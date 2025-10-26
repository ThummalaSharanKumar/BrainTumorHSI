# train.py (UPGRADED WITH CHECKPOINTING)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import numpy as np

from src.models.unet import UNet
from src.data_handling.hsi_dataset import HSIDataset

# --- 1. CONFIGURATION ---
CONFIG = {
    "REDUCER_TYPE": "PCA",
    "EPOCHS": 50,
    "BATCH_SIZE": 2,
    "LEARNING_RATE": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_COMPONENTS": 15,
    "NUM_CLASSES": 5,
    "MODEL_SAVE_PATH": None,
    "CHECKPOINT_PATH": None, # Will be set dynamically
}

# --- Dynamically set paths ---
if CONFIG["REDUCER_TYPE"] == "PCA":
    CONFIG["REDUCER_PATH"] = 'pca_model.pkl'
    CONFIG["MODEL_SAVE_PATH"] = 'best_unet_pca_model.pth'
    CONFIG["CHECKPOINT_PATH"] = 'unet_pca_checkpoint.pth'
elif CONFIG["REDUCER_TYPE"] == "NMF":
    CONFIG["REDUCER_PATH"] = 'nmf_model.pkl'
    CONFIG["MODEL_SAVE_PATH"] = 'best_unet_nmf_model.pth'
    CONFIG["CHECKPOINT_PATH"] = 'unet_nmf_checkpoint.pth'
else:
    raise ValueError("REDUCER_TYPE must be 'PCA' or 'NMF'")

# (Loss functions and train/evaluate functions remain the same as before)
def dice_loss(pred, target, smooth=1.):
    pred = torch.softmax(pred, dim=1)
    pred_flat = pred.contiguous().view(pred.shape[0], CONFIG["NUM_CLASSES"], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)
    target_one_hot = torch.nn.functional.one_hot(target_flat, num_classes=CONFIG["NUM_CLASSES"]).permute(0, 2, 1).float()
    intersection = (pred_flat * target_one_hot).sum(2)
    union = pred_flat.sum(2) + target_one_hot.sum(2)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()

def combined_loss(pred, target):
    ce = nn.CrossEntropyLoss(ignore_index=0)(pred, target)
    dice = dice_loss(pred, target)
    return ce + dice

def train_one_epoch(model, loader, optimizer, reducer):
    model.train()
    total_loss = 0
    for hsi_tensor, mask_tensor in tqdm(loader, desc="Training"):
        hsi_tensor, mask_tensor = hsi_tensor.to(CONFIG["DEVICE"]), mask_tensor.to(CONFIG["DEVICE"])
        b, c, h, w = hsi_tensor.shape
        pixels = hsi_tensor.permute(0, 2, 3, 1).reshape(b * h * w, c)
        reduced_pixels = torch.from_numpy(reducer.transform(pixels.cpu().numpy())).float().to(CONFIG["DEVICE"])
        reduced_hsi = reduced_pixels.reshape(b, h, w, CONFIG["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
        outputs = model(reduced_hsi)
        loss = combined_loss(outputs, mask_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, reducer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for hsi_tensor, mask_tensor in tqdm(loader, desc="Validating"):
            hsi_tensor, mask_tensor = hsi_tensor.to(CONFIG["DEVICE"]), mask_tensor.to(CONFIG["DEVICE"])
            b, c, h, w = hsi_tensor.shape
            pixels = hsi_tensor.permute(0, 2, 3, 1).reshape(b * h * w, c)
            reduced_pixels = torch.from_numpy(reducer.transform(pixels.cpu().numpy())).float().to(CONFIG["DEVICE"])
            reduced_hsi = reduced_pixels.reshape(b, h, w, CONFIG["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
            outputs = model(reduced_hsi)
            loss = combined_loss(outputs, mask_tensor)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- 4. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting U-Net Training ---")
    print(f"Configuration: {CONFIG}")
    
    reducer = joblib.load(CONFIG["REDUCER_PATH"])
    print(f"Loaded reducer: {CONFIG['REDUCER_PATH']}")
    
    train_dataset = HSIDataset(csv_file='train.csv', apply_augmentation=True)
    val_dataset = HSIDataset(csv_file='val.csv', apply_augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=os.cpu_count())

    model = UNet(n_channels=CONFIG["NUM_COMPONENTS"], n_classes=CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    start_epoch = 0
    best_val_loss = float('inf')

    # --- NEW: LOAD CHECKPOINT IF IT EXISTS ---
    if os.path.exists(CONFIG["CHECKPOINT_PATH"]):
        print(f"Resuming training from checkpoint: {CONFIG['CHECKPOINT_PATH']}")
        checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from Epoch {start_epoch}")

    # --- The Training Loop ---
    for epoch in range(start_epoch, CONFIG["EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, reducer)
        val_loss = evaluate(model, val_loader, reducer)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # --- NEW: SAVE CHECKPOINT AFTER EVERY EPOCH ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, CONFIG["CHECKPOINT_PATH"])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"🎉 New best model saved to {CONFIG['MODEL_SAVE_PATH']} (Val Loss: {best_val_loss:.4f})")
            
    print("\n--- Training Complete! ---")