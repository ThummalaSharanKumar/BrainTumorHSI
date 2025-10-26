# train_ssfnet.py (v4 with Class Weighting & Early Stopping)

import torch
import os
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import joblib
import numpy as np
import spectral
import pandas as pd # Import pandas for dataset

# Add spectral setting
spectral.settings.envi_support_nonlowercase_params = True

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import your custom classes
from src.data_handling.hsi_dataset import HSIDataset
from src.models.ssf_net import SSFNet

# --- NEW: Function to Calculate Class Weights ---
def calculate_class_weights(dataset, num_classes):
    print("Calculating class weights...")
    label_counts = torch.zeros(num_classes, dtype=torch.float)
    # Use a simple loader for counting
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for _, _, mask in tqdm(temp_loader, desc="Counting pixels per class"):
        labels, counts = torch.unique(mask, return_counts=True)
        for label, count in zip(labels, counts):
            if label.item() < num_classes:
                 label_counts[label.item()] += count.item()

    label_counts[0] = 0 # Ignore unlabeled
    valid_pixel_count = label_counts.sum()
    if valid_pixel_count == 0:
        print("Warning: No valid labeled pixels found for weight calculation!")
        return torch.ones(num_classes) # Return equal weights as fallback

    weights = valid_pixel_count / (label_counts + 1e-6)
    weights[0] = 0 # Zero weight for unlabeled
    # Normalize weights so they roughly sum to num_classes-1
    weights = weights / weights.sum() * (num_classes - 1)

    print(f"Class counts (raw): {label_counts.cpu().numpy()}")
    print(f"Calculated weights: {weights.cpu().numpy()}")
    return weights
# ------------------------------------------------

# --- Helper Functions ---
def dice_loss(pred, target, num_classes, smooth=1.):
    pred = torch.softmax(pred, dim=1); pred_flat = pred.contiguous().view(pred.shape[0], num_classes, -1); target_flat = target.contiguous().view(target.shape[0], -1)
    target_one_hot = F.one_hot(target_flat, num_classes=num_classes).permute(0, 2, 1).float()
    intersection = (pred_flat * target_one_hot).sum(2); union = pred_flat.sum(2) + target_one_hot.sum(2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# --- MODIFIED: Combined Loss now takes weights ---
def combined_loss(pred, target, num_classes, class_weights):
    # Pass weights to CrossEntropyLoss
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(pred.device), ignore_index=0)(pred, target)
    dice = dice_loss(pred, target, num_classes)
    return ce_loss + dice
# ------------------------------------------------

def train_one_epoch(model, loader, optimizer, class_weights, config):
    model.train(); total_loss = 0
    for hsi_spatial, hsi_spectral, mask in tqdm(loader, desc="Training SSF-Net"):
        hsi_spatial, hsi_spectral, mask = hsi_spatial.to(config["DEVICE"]), hsi_spectral.to(config["DEVICE"]), mask.to(config["DEVICE"])
        # Use calculated class_weights in the loss
        outputs = model(hsi_spatial, hsi_spectral); loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, class_weights, config):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for hsi_spatial, hsi_spectral, mask in tqdm(loader, desc="Validating SSF-Net"):
            hsi_spatial, hsi_spectral, mask = hsi_spatial.to(config["DEVICE"]), hsi_spectral.to(config["DEVICE"]), mask.to(config["DEVICE"])
            # Use calculated class_weights in the loss
            outputs = model(hsi_spatial, hsi_spectral); loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights); total_loss += loss.item()
    return total_loss / len(loader)

# --- SSF-NET V4 CONFIGURATION (CLASS WEIGHTING) ---
CONFIG = {
    "MAX_EPOCHS": 50, "EARLY_STOPPING_PATIENCE": 10,
    "BATCH_SIZE": 2, "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_SPATIAL_COMPONENTS": 15, "NUM_SPECTRAL_BANDS": 128, "NUM_CLASSES": 5,
    "REDUCER_PATH": 'pca_model.pkl',
    # --- NEW SAVE PATHS ---
    "MODEL_SAVE_PATH": 'best_ssfnet_model_v4_cw.pth', # Suffix _cw for class weighting
    "CHECKPOINT_PATH": 'ssfnet_checkpoint_v4_cw.pth',
    # ----------------------
}

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting SSF-Net Training (v4 Class Weighting + Early Stopping) ---")
    print(f"Configuration: {CONFIG}")
    
    reducer = joblib.load(CONFIG["REDUCER_PATH"])
    print(f"Loaded reducer for spatial branch: {CONFIG['REDUCER_PATH']}")

    # Create dataset instances FIRST to calculate weights
    train_dataset = HSIDataset(csv_file='train.csv', reducer=reducer, apply_augmentation=True)
    val_dataset = HSIDataset(csv_file='val.csv', reducer=reducer)
    
    # --- NEW: Calculate and send weights to the correct device ---
    class_weights = calculate_class_weights(train_dataset, CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    # ------------------------------------------------------------
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)

    model = SSFNet(
        n_spatial_channels=CONFIG["NUM_SPATIAL_COMPONENTS"],
        n_spectral_channels=CONFIG["NUM_SPECTRAL_BANDS"],
        n_classes=CONFIG["NUM_CLASSES"]
    ).to(CONFIG["DEVICE"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    try: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    except TypeError: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    start_epoch = 0; best_val_loss = float('inf'); epochs_no_improve = 0
    # Start fresh for this new experiment
    if os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        print(f"Found old checkpoint for v4, starting training from scratch.")

    for epoch in range(start_epoch, CONFIG["MAX_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['MAX_EPOCHS']} (SSF-Net v4 CW + ES) ---")
        # Pass class_weights to training function
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, CONFIG)
        # Pass class_weights to evaluation function
        val_loss = evaluate(model, val_loader, class_weights, CONFIG)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve, 'scheduler_state_dict': scheduler.state_dict()},
                   CONFIG["CHECKPOINT_PATH"])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"🎉 New best SSF-Net v4 model saved to {CONFIG['MODEL_SAVE_PATH']} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("\n--- SSF-Net Training (v4 Class Weighting + Early Stopping) Finished ---")
    if os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        print(f"Loading best model weights from {CONFIG['MODEL_SAVE_PATH']}")
        model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
    else:
        print("Warning: No best model was saved.")