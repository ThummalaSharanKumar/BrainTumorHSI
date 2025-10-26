# evaluate_ssfnet_v4.py

import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import spectral
import cv2
from sklearn.metrics import f1_score

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- All necessary classes and functions are included here ---
from src.models.ssf_net import SSFNet # Import your custom model

# --- Dataset Class Definition ---
class HSIDataset(Dataset):
    def __init__(self, csv_file, reducer, patch_size=256, apply_augmentation=False):
        self.manifest = pd.read_csv(csv_file); self.reducer = reducer
        self.patch_size = patch_size; self.apply_augmentation = apply_augmentation
        self.n_bands = 128; self.standard_wavelengths = np.linspace(450, 900, self.n_bands)
        self.patches = []
        # Suppress spectral warning
        spectral.settings.envi_support_nonlowercase_params = True
        print(f"Calculating patches for {len(self.manifest)} images...")
        for idx in range(len(self.manifest)):
            gt_path = self.manifest.loc[idx, 'gt_path']
            try:
                gt_info = spectral.envi.read_envi_header(gt_path)
                h, w = int(gt_info['lines']), int(gt_info['samples'])
                stride = patch_size // 2
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride): self.patches.append((idx, y, x))
            except Exception as e: print(f"--- WARNING: Could not read {gt_path}. Skipping. Error: {e}")
        print(f"Initialized SSF-Net dataset from '{csv_file}'. Found {len(self.patches)} total patches.")

    def __len__(self): return len(self.patches)
    def __getitem__(self, idx):
        image_idx, y_start, x_start = self.patches[idx]; y_end, x_end = y_start + self.patch_size, x_start + self.patch_size
        hsi_path = self.manifest.loc[image_idx, 'hsi_path']
        try:
            hsi_image_file = spectral.open_image(hsi_path); raw_h, raw_w, _ = hsi_image_file.shape
            gt_mask_file = spectral.open_image(self.manifest.loc[image_idx, 'gt_path'])
            base_dir = os.path.dirname(hsi_path)
            white_ref_file = spectral.open_image(os.path.join(base_dir, 'whiteReference.hdr'))
            dark_ref_file = spectral.open_image(os.path.join(base_dir, 'darkReference.hdr'))

            white_ref = white_ref_file.read_band(0); dark_ref = dark_ref_file.read_band(0)
            if white_ref.shape != (raw_h, raw_w): white_ref = cv2.resize(white_ref, (raw_w, raw_h), cv2.INTER_NEAREST)
            if dark_ref.shape != (raw_h, raw_w): dark_ref = cv2.resize(dark_ref, (raw_w, raw_h), cv2.INTER_NEAREST)

            white_ref_patch = white_ref[y_start:y_end, x_start:x_end][:, :, np.newaxis]
            dark_ref_patch = dark_ref[y_start:y_end, x_start:x_end][:, :, np.newaxis]
            hsi_patch = hsi_image_file[y_start:y_end, x_start:x_end, :]
            mask_patch = gt_mask_file.read_band(0)[y_start:y_end, x_start:x_end]

            calibrated_patch = np.divide(hsi_patch - dark_ref_patch, white_ref_patch - dark_ref_patch + 1e-8)
            calibrated_patch = np.clip(calibrated_patch, 0, 1)

            original_wavelengths = np.array([float(w) for w in hsi_image_file.metadata['wavelength']])
            resampled_patch = np.apply_along_axis(lambda s: np.interp(self.standard_wavelengths, original_wavelengths, s), 2, calibrated_patch)

            hsi_tensor_spectral = torch.from_numpy(resampled_patch.copy().transpose(2, 0, 1)).float()
            h, w, c = resampled_patch.shape; pixels = resampled_patch.reshape(h * w, c)
            # Ensure correct dtype for reducer
            pixel_data = pixels
            if 'NMF' in str(self.reducer.__class__):
                 pixel_data = pixels.astype(np.float64)
            reduced_pixels = self.reducer.transform(pixel_data)
            spatial_image = reduced_pixels.reshape(h, w, -1)
            hsi_tensor_spatial = torch.from_numpy(spatial_image.copy().transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_patch.copy()).long()
            return hsi_tensor_spatial, hsi_tensor_spectral, mask_tensor
        except Exception as e:
            print(f"ERROR loading patch from {hsi_path} at (y:{y_start}, x:{x_start}): {e}")
            n_components = getattr(self.reducer, 'n_components_', 15)
            dummy_spatial = torch.zeros((n_components, self.patch_size, self.patch_size))
            dummy_spectral = torch.zeros((self.n_bands, self.patch_size, self.patch_size))
            dummy_mask = torch.zeros((self.patch_size, self.patch_size)).long()
            return dummy_spatial, dummy_spectral, dummy_mask

# --- Metrics Calculation Function ---
def calculate_metrics(pred, target, num_classes):
    pred = torch.softmax(pred, dim=1); pred_labels = torch.argmax(pred, dim=1)
    pred_labels_flat = pred_labels.cpu().numpy().flatten(); target_flat = target.cpu().numpy().flatten()
    mask = target_flat != 0
    if not np.any(mask): return 0, 0, 0
    f1 = f1_score(target_flat[mask], pred_labels_flat[mask], average='macro', zero_division=0)
    iou_per_class, dice_per_class = [], []
    for cls in range(1, num_classes):
        pred_inds = pred_labels == cls; target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item(); union = (pred_inds | target_inds).sum().item()
        dice = (2. * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-8)
        iou = intersection / (union + 1e-8)
        iou_per_class.append(iou); dice_per_class.append(dice)
    return f1, np.mean(iou_per_class), np.mean(dice_per_class)

# --- SSF-NET V4 (CW) EVALUATION CONFIGURATION ---
EVAL_CONFIG = {
    "BATCH_SIZE": 2,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_SPATIAL_COMPONENTS": 15,
    "NUM_SPECTRAL_BANDS": 128,
    "NUM_CLASSES": 5,
    "REDUCER_PATH": 'pca_model.pkl',
    # --- Path to the class-weighted model ---
    "MODEL_PATH": 'best_ssfnet_model_v4_cw.pth',
    # ----------------------------------------
}

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting Final Evaluation of SSF-Net Model (v4 with Class Weighting) ---")
    print(f"Evaluating model: {EVAL_CONFIG['MODEL_PATH']}")

    if not os.path.exists(EVAL_CONFIG['MODEL_PATH']):
        print(f"!!! ERROR: Model file not found at {EVAL_CONFIG['MODEL_PATH']}")
        print("!!! Please ensure the SSF-Net (v4) training completed successfully.")
        sys.exit()

    reducer = joblib.load(EVAL_CONFIG["REDUCER_PATH"])
    test_dataset = HSIDataset(csv_file='test.csv', reducer=reducer)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)

    # Make sure SSFNet is imported correctly from the file containing the SEBlock
    model = SSFNet(
        n_spatial_channels=EVAL_CONFIG["NUM_SPATIAL_COMPONENTS"],
        n_spectral_channels=EVAL_CONFIG["NUM_SPECTRAL_BANDS"],
        n_classes=EVAL_CONFIG["NUM_CLASSES"]
    ).to(EVAL_CONFIG["DEVICE"])
    
    try:
        model.load_state_dict(torch.load(EVAL_CONFIG["MODEL_PATH"], map_location=EVAL_CONFIG["DEVICE"]))
    except RuntimeError as e:
        print(f"!!! ERROR loading model state_dict: {e}")
        print("!!! This might happen if the saved model architecture doesn't match the current SSFNet class.")
        sys.exit()
        
    model.eval()

    all_f1, all_iou, all_dice = [], [], []
    with torch.no_grad():
        for hsi_spatial, hsi_spectral, mask in tqdm(test_loader, desc="Evaluating SSF-Net v4 (CW) on Test Set"):
            hsi_spatial = hsi_spatial.to(EVAL_CONFIG["DEVICE"])
            hsi_spectral = hsi_spectral.to(EVAL_CONFIG["DEVICE"])
            mask = mask.to(EVAL_CONFIG["DEVICE"])
            
            outputs = model(hsi_spatial, hsi_spectral)
            f1, iou, dice = calculate_metrics(outputs, mask, EVAL_CONFIG["NUM_CLASSES"])
            all_f1.append(f1); all_iou.append(iou); all_dice.append(dice)

    print("\n--- SSF-Net v4 (Class Weighting) Evaluation Complete ---")
    print(f"📈 Average Macro F1-Score: {np.mean(all_f1):.4f}")
    print(f"📐 Average Intersection over Union (IoU): {np.mean(all_iou):.4f}")
    print(f"🎯 Average Dice Score (DSC): {np.mean(all_dice):.4f}")
    print("----------------------------------")