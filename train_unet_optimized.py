# train_unet_optimized.py (v5: Advanced Augmentation & Normalization)

import torch
import os
import sys
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import joblib
import numpy as np
import spectral
import pandas as pd
import cv2
# --- NEW: Import Albumentations ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ---------------------------------

# Add spectral setting
spectral.settings.envi_support_nonlowercase_params = True

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import your U-Net model
from src.models.unet import UNet # Assuming UNet definition is in src/models/unet.py

# --- MODIFIED: HSIDataset now uses Albumentations ---
class HSIDataset(Dataset):
    def __init__(self, csv_file, reducer, patch_size=256, apply_augmentation=False, mean=None, std=None):
        self.manifest = pd.read_csv(csv_file); self.reducer = reducer
        self.patch_size = patch_size; self.apply_augmentation = apply_augmentation
        self.mean = mean; self.std = std # Store normalization stats
        self.n_bands = 128; self.standard_wavelengths = np.linspace(450, 900, self.n_bands)
        self.patches = []
        print(f"Calculating patches for {len(self.manifest)} images...")
        # ... (patch calculation logic remains the same) ...
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

        # --- Define Augmentation Pipelines ---
        self.base_transform = A.Compose([
             A.Resize(patch_size, patch_size), # Ensure patch size just in case
        ])
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
        ])
        self.normalize_transform = A.Compose([
             # Normalize uses pre-calculated mean/std
             A.Normalize(mean=self.mean if self.mean is not None else [0.0]*15,
                         std=self.std if self.std is not None else [1.0]*15,
                         max_pixel_value=1.0), # Assuming data is 0-1 before this
             ToTensorV2(), # Converts numpy (H, W, C) to torch (C, H, W) and scales
        ])
        # ------------------------------------

    def __len__(self): return len(self.patches)
    def __getitem__(self, idx):
        image_idx, y_start, x_start = self.patches[idx]; y_end, x_end = y_start + self.patch_size, x_start + self.patch_size
        hsi_path = self.manifest.loc[image_idx, 'hsi_path']
        try:
            # --- Load, Calibrate, Resample (as before) ---
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
            calibrated_patch = np.divide(hsi_patch - dark_ref_patch, white_ref_patch - dark_ref_patch + 1e-8); calibrated_patch = np.clip(calibrated_patch, 0, 1)
            original_wavelengths = np.array([float(w) for w in hsi_image_file.metadata['wavelength']])
            resampled_patch = np.apply_along_axis(lambda s: np.interp(self.standard_wavelengths, original_wavelengths, s), 2, calibrated_patch)

            # --- Apply Reducer to get Spatial Data ---
            h, w, c = resampled_patch.shape; pixels = resampled_patch.reshape(h * w, c)
            pixel_data = pixels # Assuming PCA for U-Net, needs float32
            if 'NMF' in str(self.reducer.__class__): pixel_data = pixels.astype(np.float64) # Handle NMF if used later
            reduced_pixels = self.reducer.transform(pixel_data)
            spatial_image = reduced_pixels.reshape(h, w, -1).astype(np.float32) # Ensure float32 for albumentations

            # --- Apply Albumentations ---
            # Base resize first
            transformed = self.base_transform(image=spatial_image, mask=mask_patch)
            spatial_image, mask_patch = transformed['image'], transformed['mask']

            # Apply augmentations if needed
            if self.apply_augmentation:
                augmented = self.aug_transform(image=spatial_image, mask=mask_patch)
                spatial_image, mask_patch = augmented['image'], augmented['mask']

            # Apply normalization and convert to tensor
            normalized = self.normalize_transform(image=spatial_image, mask=mask_patch)
            hsi_tensor_spatial = normalized['image']
            mask_tensor = normalized['mask'].long() # Ensure mask is LongTensor

            # --- U-Net only needs spatial data and mask ---
            return hsi_tensor_spatial, mask_tensor

        except Exception as e:
            print(f"ERROR loading patch from {hsi_path} at (y:{y_start}, x:{x_start}): {e}")
            n_components = getattr(self.reducer, 'n_components_', 15)
            dummy_spatial = torch.zeros((n_components, self.patch_size, self.patch_size))
            dummy_mask = torch.zeros((self.patch_size, self.patch_size)).long()
            return dummy_spatial, dummy_mask

# --- Calculate Normalization Stats ---
def get_normalization_stats(dataset):
    print("Calculating normalization stats (mean/std)...")
    # Use a simple loader to iterate through spatial data only
    temp_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    mean = torch.zeros(dataset.reducer.n_components_)
    std = torch.zeros(dataset.reducer.n_components_)
    nb_samples = 0
    # Only need spatial tensor for normalization calc
    for spatial_tensor, _ in tqdm(temp_loader, desc="Calculating Stats"):
        batch_samples = spatial_tensor.size(0)
        # Reshape C, H, W to B, C, H*W -> B*H*W, C
        spatial_tensor = spatial_tensor.permute(0, 2, 3, 1).reshape(-1, spatial_tensor.size(1))
        mean += spatial_tensor.mean(0) * batch_samples
        std += spatial_tensor.std(0) * batch_samples
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(f"Mean: {mean.numpy()}")
    print(f"Std: {std.numpy()}")
    return mean.numpy(), std.numpy() # Return as numpy arrays

# --- Calculate Class Weights ---
def calculate_class_weights(dataset, num_classes):
    print("Calculating class weights...")
    label_counts = torch.zeros(num_classes, dtype=torch.float)
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for _, mask in tqdm(temp_loader, desc="Counting pixels"):
        labels, counts = torch.unique(mask, return_counts=True)
        for label, count in zip(labels, counts):
            if label.item() < num_classes: label_counts[label.item()] += count.item()
    label_counts[0] = 0; valid_pixel_count = label_counts.sum()
    if valid_pixel_count == 0: return torch.ones(num_classes)
    weights = valid_pixel_count / (label_counts + 1e-6); weights[0] = 0
    weights = weights / weights.sum() * (num_classes - 1)
    print(f"Class counts (raw): {label_counts.cpu().numpy()}")
    print(f"Calculated weights: {weights.cpu().numpy()}")
    return weights

# --- Helper Functions ---
def dice_loss(pred, target, num_classes, smooth=1.):
    # ... (dice loss implementation remains the same) ...
    pred = torch.softmax(pred, dim=1); pred_flat = pred.contiguous().view(pred.shape[0], num_classes, -1); target_flat = target.contiguous().view(target.shape[0], -1)
    target_one_hot = F.one_hot(target_flat, num_classes=num_classes).permute(0, 2, 1).float()
    intersection = (pred_flat * target_one_hot).sum(2); union = pred_flat.sum(2) + target_one_hot.sum(2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
def combined_loss(pred, target, num_classes, class_weights):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(pred.device), ignore_index=0)(pred, target)
    dice = dice_loss(pred, target, num_classes)
    return ce_loss + dice

# --- Training Loop Functions (U-Net version) ---
def train_one_epoch(model, loader, optimizer, class_weights, config):
    model.train(); total_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(config["DEVICE"] == "cuda"))
    for reduced_hsi, mask in tqdm(loader, desc="Training U-Net"):
        reduced_hsi, mask = reduced_hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(config["DEVICE"] == "cuda")):
            outputs = model(reduced_hsi)
            loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, class_weights, config):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for reduced_hsi, mask in tqdm(loader, desc="Validating U-Net"):
            reduced_hsi, mask = reduced_hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
            with torch.cuda.amp.autocast(enabled=(config["DEVICE"] == "cuda")):
                outputs = model(reduced_hsi)
                loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- FINAL OPTIMIZED U-NET CONFIGURATION (v5) ---
CONFIG = {
    "MAX_EPOCHS": 50, "EARLY_STOPPING_PATIENCE": 10,
    "BATCH_SIZE": 4, # Increased batch size for potentially faster training
    "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_COMPONENTS": 15, "NUM_CLASSES": 5,
    "REDUCER_PATH": 'pca_model.pkl',
    "MODEL_SAVE_PATH": 'best_unet_final_optimized.pth',
    "CHECKPOINT_PATH": 'unet_final_optimized_checkpoint.pth',
}

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting Final Optimized U-Net Training (v5) ---")
    print(f"Configuration: {CONFIG}")
    if CONFIG["DEVICE"] == "cpu": print("WARNING: No GPU detected. Training will be very slow.")

    reducer = joblib.load(CONFIG["REDUCER_PATH"])
    print(f"Loaded reducer: {CONFIG['REDUCER_PATH']}")

    # --- NEW: Calculate normalization stats before creating final datasets ---
    # Create a temporary dataset instance *without* normalization to calculate stats
    temp_train_dataset = HSIDataset(csv_file='train.csv', reducer=reducer, apply_augmentation=False, mean=None, std=None)
    train_mean, train_std = get_normalization_stats(temp_train_dataset)
    class_weights = calculate_class_weights(temp_train_dataset, CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    del temp_train_dataset # Free up memory
    # --------------------------------------------------------------------

    # Create final datasets WITH normalization stats
    train_dataset = HSIDataset(csv_file='train.csv', reducer=reducer, apply_augmentation=True, mean=train_mean, std=train_std)
    val_dataset = HSIDataset(csv_file='val.csv', reducer=reducer, mean=train_mean, std=train_std)

    num_workers = 0; pin_memory = False
    if CONFIG["DEVICE"] == "cuda": num_workers = 2; pin_memory = True
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = UNet(n_channels=CONFIG["NUM_COMPONENTS"], n_classes=CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    try: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    except TypeError: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    start_epoch = 0; best_val_loss = float('inf'); epochs_no_improve = 0
    if os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        print(f"Resuming training from checkpoint: {CONFIG['CHECKPOINT_PATH']}")
        checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=CONFIG["DEVICE"])
        # ... (checkpoint loading logic remains the same) ...
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        print(f"Resuming from Epoch {start_epoch}")

    for epoch in range(start_epoch, CONFIG["MAX_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['MAX_EPOCHS']} (Final Optimized U-Net) ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, config=CONFIG) # Pass config instead of reducer
        val_loss = evaluate(model, val_loader, class_weights, config=CONFIG) # Pass config instead of reducer
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve, 'scheduler_state_dict': scheduler.state_dict()},
                   CONFIG["CHECKPOINT_PATH"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"🎉 New best Final Optimized U-Net model saved to {CONFIG['MODEL_SAVE_PATH']} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("\n--- Final Optimized U-Net Training Finished ---")
    if os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        print(f"Loading best model weights from {CONFIG['MODEL_SAVE_PATH']}")
        model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
    else: print("Warning: No best model was saved.")