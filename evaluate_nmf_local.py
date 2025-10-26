# evaluate_nmf_local.py

import torch
import os
import sys
from torch.utils.data import DataLoader

# Add src to Python path to find the model/data classes if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- All necessary classes and functions are included here ---
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
import spectral
import cv2
from sklearn.metrics import f1_score

# --- Dataset Class Definition ---
class HSIDataset(Dataset):
    def __init__(self, csv_file, patch_size=256, apply_augmentation=False):
        self.manifest = pd.read_csv(csv_file); self.patch_size = patch_size; self.apply_augmentation = apply_augmentation
        self.n_bands = 128; self.standard_wavelengths = np.linspace(450, 900, self.n_bands)
        self.patches = []
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
        print(f"Initialized dataset from '{csv_file}'. Found {len(self.patches)} total patches.")

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

            if self.apply_augmentation:
                if np.random.rand() > 0.5: resampled_patch, mask_patch = np.fliplr(resampled_patch), np.fliplr(mask_patch)
                if np.random.rand() > 0.5: resampled_patch, mask_patch = np.flipud(resampled_patch), np.flipud(mask_patch)
            
            return torch.from_numpy(resampled_patch.copy().transpose(2, 0, 1)).float(), torch.from_numpy(mask_patch.copy()).long()
        except Exception as e:
            print(f"ERROR loading patch from {hsi_path} at (y:{y_start}, x:{x_start}): {e}")
            return torch.zeros((self.n_bands, self.patch_size, self.patch_size)), torch.zeros((self.patch_size, self.patch_size)).long()

# --- U-Net Model Definition ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(x)
class Down(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.mpconv(x)
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__(); self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]); x = torch.cat([x2, x1], dim=1); return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__(); self.inc = InConv(n_channels, 64); self.down1 = Down(64, 128); self.down2 = Down(128, 256); self.down3 = Down(256, 512); self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256); self.up2 = Up(512, 128); self.up3 = Up(256, 64); self.up4 = Up(128, 64); self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1); return self.outc(x)

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

# --- NMF EVALUATION CONFIGURATION ---
EVAL_CONFIG_NMF = {
    "REDUCER_TYPE": "NMF", "BATCH_SIZE": 2,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_COMPONENTS": 15, "NUM_CLASSES": 5,
    "REDUCER_PATH": 'nmf_model.pkl',
    "MODEL_PATH": 'best_unet_nmf_model.pth',
}

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting Final Evaluation of NMF Model ---")
    print(f"Evaluating model: {EVAL_CONFIG_NMF['MODEL_PATH']}")

    reducer = joblib.load(EVAL_CONFIG_NMF["REDUCER_PATH"])
    test_dataset = HSIDataset(csv_file='test.csv')
    test_loader = DataLoader(test_dataset, batch_size=EVAL_CONFIG_NMF["BATCH_SIZE"], shuffle=False, num_workers=0)

    model = UNet(n_channels=EVAL_CONFIG_NMF["NUM_COMPONENTS"], n_classes=EVAL_CONFIG_NMF["NUM_CLASSES"]).to(EVAL_CONFIG_NMF["DEVICE"])
    model.load_state_dict(torch.load(EVAL_CONFIG_NMF["MODEL_PATH"], map_location=EVAL_CONFIG_NMF["DEVICE"]))
    model.eval()

    all_f1, all_iou, all_dice = [], [], []
    with torch.no_grad():
        for hsi, mask in tqdm(test_loader, desc="Evaluating NMF Model on Test Set"):
            hsi, mask = hsi.to(EVAL_CONFIG_NMF["DEVICE"]), mask.to(EVAL_CONFIG_NMF["DEVICE"])
            b, c, h, w = hsi.shape; pixels = hsi.permute(0, 2, 3, 1).reshape(b * h * w, c)
            
            # Convert to float64 for NMF
            pixel_data = pixels.cpu().numpy().astype(np.float64)
            
            reduced_pixels = torch.from_numpy(reducer.transform(pixel_data)).float().to(EVAL_CONFIG_NMF["DEVICE"])
            reduced_hsi = reduced_pixels.reshape(b, h, w, EVAL_CONFIG_NMF["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
            outputs = model(reduced_hsi)
            f1, iou, dice = calculate_metrics(outputs, mask, EVAL_CONFIG_NMF["NUM_CLASSES"])
            all_f1.append(f1); all_iou.append(iou); all_dice.append(dice)

    print("\n--- NMF Evaluation Complete ---")
    print(f"📈 Average Macro F1-Score: {np.mean(all_f1):.4f}")
    print(f"📐 Average Intersection over Union (IoU): {np.mean(all_iou):.4f}")
    print(f"🎯 Average Dice Score (DSC): {np.mean(all_dice):.4f}")
    print("----------------------------------")