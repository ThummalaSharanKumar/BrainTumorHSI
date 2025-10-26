# train_nmf_local.py

import torch
import os
import sys
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import all your classes and functions
from src.data_handling.hsi_dataset import HSIDataset
from src.models.unet import UNet
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import joblib
import numpy as np

# --- Helper Functions (copied for self-containment) ---
def dice_loss(pred, target, num_classes, smooth=1.):
    pred = torch.softmax(pred, dim=1); pred_flat = pred.contiguous().view(pred.shape[0], num_classes, -1); target_flat = target.contiguous().view(target.shape[0], -1)
    target_one_hot = F.one_hot(target_flat, num_classes=num_classes).permute(0, 2, 1).float()
    intersection = (pred_flat * target_one_hot).sum(2); union = pred_flat.sum(2) + target_one_hot.sum(2)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
def combined_loss(pred, target, num_classes): return nn.CrossEntropyLoss(ignore_index=0)(pred, target) + dice_loss(pred, target, num_classes)
def train_one_epoch(model, loader, optimizer, reducer, config):
    model.train(); total_loss = 0
    for hsi, mask in tqdm(loader, desc="Training"):
        hsi, mask = hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
        b, c, h, w = hsi.shape; pixels = hsi.permute(0, 2, 3, 1).reshape(b * h * w, c)
        pixel_data = pixels.cpu().numpy().astype(np.float64) # NMF requires float64
        reduced_pixels = torch.from_numpy(reducer.transform(pixel_data)).float().to(config["DEVICE"])
        reduced_hsi = reduced_pixels.reshape(b, h, w, config["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
        outputs = model(reduced_hsi); loss = combined_loss(outputs, mask, config["NUM_CLASSES"])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += loss.item()
    return total_loss / len(loader)
def evaluate(model, loader, reducer, config):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for hsi, mask in tqdm(loader, desc="Validating"):
            hsi, mask = hsi.to(config["DEVICE"]), mask.to(config["DEVICE"])
            b, c, h, w = hsi.shape; pixels = hsi.permute(0, 2, 3, 1).reshape(b * h * w, c)
            pixel_data = pixels.cpu().numpy().astype(np.float64) # NMF requires float64
            reduced_pixels = torch.from_numpy(reducer.transform(pixel_data)).float().to(config["DEVICE"])
            reduced_hsi = reduced_pixels.reshape(b, h, w, config["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
            outputs = model(reduced_hsi); loss = combined_loss(outputs, mask, config["NUM_CLASSES"]); total_loss += loss.item()
    return total_loss / len(loader)

# --- NMF CONFIGURATION FOR LOCAL MACHINE ---
CONFIG_NMF = {
    "REDUCER_TYPE": "NMF", "EPOCHS": 50, "BATCH_SIZE": 2, "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu", # Automatically selects CPU
    "NUM_COMPONENTS": 15, "NUM_CLASSES": 5,
}
CONFIG_NMF["REDUCER_PATH"] = 'nmf_model.pkl'
CONFIG_NMF["MODEL_SAVE_PATH"] = 'best_unet_nmf_model.pth'
CONFIG_NMF["CHECKPOINT_PATH"] = 'unet_nmf_checkpoint.pth'

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Resuming U-Net Training for NMF on Local Machine ---")
    print(f"Configuration: {CONFIG_NMF}")
    
    reducer_nmf = joblib.load(CONFIG_NMF["REDUCER_PATH"])
    print(f"Loaded reducer: {CONFIG_NMF['REDUCER_PATH']}")

    train_dataset = HSIDataset(csv_file='train.csv', apply_augmentation=True)
    val_dataset = HSIDataset(csv_file='val.csv', apply_augmentation=False)
    # Using num_workers=0 is safest for local machines to avoid memory issues.
    train_loader = DataLoader(train_dataset, batch_size=CONFIG_NMF["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG_NMF["BATCH_SIZE"], shuffle=False, num_workers=0)

    model_nmf = UNet(n_channels=CONFIG_NMF["NUM_COMPONENTS"], n_classes=CONFIG_NMF["NUM_CLASSES"]).to(CONFIG_NMF["DEVICE"])
    optimizer_nmf = torch.optim.Adam(model_nmf.parameters(), lr=CONFIG_NMF["LEARNING_RATE"])

    start_epoch_nmf = 0
    best_val_loss_nmf = float('inf')
    
    # This is the key part: it will find your downloaded checkpoint file
    if os.path.exists(CONFIG_NMF["CHECKPOINT_PATH"]):
        print(f"Resuming NMF training from checkpoint: {CONFIG_NMF['CHECKPOINT_PATH']}")
        checkpoint_nmf = torch.load(CONFIG_NMF["CHECKPOINT_PATH"], map_location=CONFIG_NMF["DEVICE"])
        model_nmf.load_state_dict(checkpoint_nmf['model_state_dict'])
        optimizer_nmf.load_state_dict(checkpoint_nmf['optimizer_state_dict'])
        start_epoch_nmf = checkpoint_nmf['epoch'] + 1
        best_val_loss_nmf = checkpoint_nmf['best_val_loss']
        print(f"Resuming from Epoch {start_epoch_nmf}")
    else:
        print("--- No checkpoint found. Starting training from scratch. ---")

    for epoch in range(start_epoch_nmf, CONFIG_NMF["EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG_NMF['EPOCHS']} (NMF) ---")
        train_loss = train_one_epoch(model_nmf, train_loader, optimizer_nmf, reducer_nmf, CONFIG_NMF)
        val_loss = evaluate(model_nmf, val_loader, reducer_nmf, CONFIG_NMF)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model_nmf.state_dict(), 'optimizer_state_dict': optimizer_nmf.state_dict(), 'best_val_loss': best_val_loss_nmf}, CONFIG_NMF["CHECKPOINT_PATH"])
        if val_loss < best_val_loss_nmf:
            best_val_loss_nmf = val_loss
            torch.save(model_nmf.state_dict(), CONFIG_NMF["MODEL_SAVE_PATH"])
            print(f"🎉 New best NMF model saved to {CONFIG_NMF['MODEL_SAVE_PATH']} (Val Loss: {best_val_loss_nmf:.4f})")

    print("\n--- NMF Training Complete! ---")