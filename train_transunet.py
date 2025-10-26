# train_transunet.py (Corrected Imports)

import torch
import os
import sys
from torch.utils.data import DataLoader, Dataset # Import Dataset base class
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import joblib
import numpy as np
import spectral
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add spectral setting
spectral.settings.envi_support_nonlowercase_params = True

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Import the TransUNet model ---
from src.models.transunet import TransUNet

# --- *** THE FIX: Explicitly import the OPTIMIZED dataset class *** ---
try:
    from src.data_handling.hsi_dataset_optimized import HSIDataset
    print("Successfully imported optimized HSIDataset.")
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ERROR: Could not find 'src/data_handling/hsi_dataset_optimized.py' !!!")
    print("!!! Please ensure the final HSIDataset class (with albumentations)    !!!")
    print("!!! is saved in that file.                                           !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit()
# --- End Fix ---

# --- Calculate Normalization Stats ---
# (Function remains the same, but uses the imported HSIDataset)
def get_normalization_stats(dataset_class, csv_file, reducer):
    print("Calculating normalization stats (mean/std)...")
    temp_dataset = dataset_class(csv_file=csv_file, reducer=reducer, mean=None, std=None, apply_augmentation=False)
    if len(temp_dataset) == 0: raise ValueError("Temporary dataset empty!")
    temp_loader = DataLoader(temp_dataset, batch_size=4, shuffle=False, num_workers=0)
    sums = torch.zeros(reducer.n_components_)
    sq_sums = torch.zeros(reducer.n_components_)
    num_pixels = 0
    # Dataset yields (spatial_tensor, mask)
    for spatial_tensor, _ in tqdm(temp_loader, desc="Calculating Stats"):
        sums += torch.sum(spatial_tensor, dim=[0, 2, 3])
        sq_sums += torch.sum(spatial_tensor**2, dim=[0, 2, 3])
        num_pixels += spatial_tensor.shape[0] * spatial_tensor.shape[2] * spatial_tensor.shape[3]
    if num_pixels == 0: raise ValueError("No pixels found for stats!")
    mean = sums / num_pixels
    std = torch.sqrt((sq_sums / num_pixels) - mean**2); std = torch.maximum(std, torch.tensor(1e-6))
    print(f"Mean: {mean.numpy()}"); print(f"Std: {std.numpy()}")
    return mean.numpy(), std.numpy()

# --- Calculate Class Weights ---
# (Function remains the same, but uses the imported HSIDataset)
def calculate_class_weights(dataset, num_classes):
    print("Calculating class weights..."); label_counts = torch.zeros(num_classes, dtype=torch.float)
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for _, mask in tqdm(temp_loader, desc="Counting pixels"): # Dataset yields (spatial, mask)
        labels, counts = torch.unique(mask, return_counts=True)
        for label, count in zip(labels, counts):
            if label.item() < num_classes: label_counts[label.item()] += count.item()
    label_counts[0] = 0; valid_pixel_count = label_counts.sum()
    if valid_pixel_count == 0: return torch.ones(num_classes)
    weights = valid_pixel_count / (label_counts + 1e-6); weights[0] = 0
    weights = weights / weights.sum() * (num_classes - 1)
    print(f"Class counts (raw): {label_counts.cpu().numpy()}"); print(f"Calculated weights: {weights.cpu().numpy()}")
    return weights

# --- Helper Functions (dice_loss, combined_loss remain same) ---
def dice_loss(pred, target, num_classes, smooth=1.):
    pred = torch.softmax(pred, dim=1); pred_flat = pred.contiguous().view(pred.shape[0], num_classes, -1); target_flat = target.contiguous().view(target.shape[0], -1)
    target_one_hot = F.one_hot(target_flat, num_classes=num_classes).permute(0, 2, 1).float()
    intersection = (pred_flat * target_one_hot).sum(2); union = pred_flat.sum(2) + target_one_hot.sum(2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
def combined_loss(pred, target, num_classes, class_weights):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(pred.device), ignore_index=0)(pred, target)
    dice = dice_loss(pred, target, num_classes)
    return ce_loss + dice

# --- Training Loop Functions (Adapted for TransUNet) ---
def train_one_epoch(model, loader, optimizer, class_weights, config):
    model.train(); total_loss = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(config["DEVICE"] == "cuda"))
    # The optimized dataset yields (reduced_hsi, mask)
    for reduced_hsi, mask in tqdm(loader, desc="Training TransUNet"):
        reduced_hsi, mask = reduced_hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
        optimizer.zero_grad()
        with torch.amp.autocast(config["DEVICE"], enabled=(config["DEVICE"] == "cuda")):
            outputs = model(reduced_hsi) # Pass reduced HSI to TransUNet
            loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, class_weights, config):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for reduced_hsi, mask in tqdm(loader, desc="Validating TransUNet"):
            reduced_hsi, mask = reduced_hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
            with torch.amp.autocast(config["DEVICE"], enabled=(config["DEVICE"] == "cuda")):
                outputs = model(reduced_hsi)
                loss = combined_loss(outputs, mask, config["NUM_CLASSES"], class_weights)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- TRANSUNET CONFIGURATION ---
CONFIG = {
    "MAX_EPOCHS": 50, "EARLY_STOPPING_PATIENCE": 10, "BATCH_SIZE": 2, "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_COMPONENTS": 15, "NUM_CLASSES": 5, "PATCH_SIZE": 256,
    "REDUCER_PATH": 'pca_model.pkl',
    "VIT_PATCH_SIZE": 16, "VIT_EMBED_DIM": 768, "VIT_DEPTH": 12, "VIT_NUM_HEADS": 12,
    "MODEL_SAVE_PATH": 'best_transunet.pth', "CHECKPOINT_PATH": 'transunet_checkpoint.pth',
}

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting TransUNet Training ---")
    print(f"Configuration: {CONFIG}")
    if CONFIG["DEVICE"] == "cpu": print("WARNING: No GPU detected. Training will be very slow.")

    reducer = joblib.load(CONFIG["REDUCER_PATH"])
    print(f"Loaded reducer: {CONFIG['REDUCER_PATH']}")

    # Calculate normalization stats using the CORRECT (imported) HSIDataset class
    train_mean, train_std = get_normalization_stats(HSIDataset, 'train.csv', reducer)
    # Create a temporary dataset to calculate class weights
    temp_train_dataset = HSIDataset(csv_file='train.csv', reducer=reducer, mean=train_mean, std=train_std)
    class_weights = calculate_class_weights(temp_train_dataset, CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    del temp_train_dataset # Free memory

    # Create final datasets WITH normalization stats
    train_dataset = HSIDataset(csv_file='train.csv', reducer=reducer, apply_augmentation=True, mean=train_mean, std=train_std)
    val_dataset = HSIDataset(csv_file='val.csv', reducer=reducer, mean=train_mean, std=train_std)

    num_workers = 0; pin_memory = False
    if CONFIG["DEVICE"] == "cuda": num_workers = 2; pin_memory = True
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = TransUNet(
        n_channels=CONFIG["NUM_COMPONENTS"], n_classes=CONFIG["NUM_CLASSES"],
        img_size=CONFIG["PATCH_SIZE"], vit_patch_size=CONFIG["VIT_PATCH_SIZE"],
        vit_embed_dim=CONFIG["VIT_EMBED_DIM"], vit_depth=CONFIG["VIT_DEPTH"],
        vit_num_heads=CONFIG["VIT_NUM_HEADS"]
    ).to(CONFIG["DEVICE"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    try: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    except TypeError: scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    start_epoch = 0; best_val_loss = float('inf'); epochs_no_improve = 0
    if os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        print(f"Resuming training from checkpoint: {CONFIG['CHECKPOINT_PATH']}")
        checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=CONFIG["DEVICE"])
        model.load_state_dict(checkpoint['model_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1; best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        print(f"Resuming from Epoch {start_epoch}")

    for epoch in range(start_epoch, CONFIG["MAX_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['MAX_EPOCHS']} (TransUNet) ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, config=CONFIG)
        val_loss = evaluate(model, val_loader, class_weights, config=CONFIG)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve, 'scheduler_state_dict': scheduler.state_dict()},
                   CONFIG["CHECKPOINT_PATH"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"🎉 New best TransUNet model saved to {CONFIG['MODEL_SAVE_PATH']} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("\n--- TransUNet Training Finished ---")
    if os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        print(f"Loading best model weights from {CONFIG['MODEL_SAVE_PATH']}")
        model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
    else: print("Warning: No best TransUNet model was saved.")