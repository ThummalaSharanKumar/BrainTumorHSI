# evaluate.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import numpy as np
from sklearn.metrics import f1_score

from src.models.unet import UNet
from src.data_handling.hsi_dataset import HSIDataset

# --- 1. CONFIGURATION ---
CONFIG = {
    "REDUCER_TYPE": "PCA", # Options: "PCA" or "NMF"
    "BATCH_SIZE": 2,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_COMPONENTS": 15,
    "NUM_CLASSES": 5,      # IMPORTANT: Must match your data (0-4)
    "MODEL_PATH": None,    # Will be set dynamically
    "REDUCER_PATH": None,  # Will be set dynamically
}

# Dynamically set paths based on the chosen model to evaluate
if CONFIG["REDUCER_TYPE"] == "PCA":
    CONFIG["REDUCER_PATH"] = 'pca_model.pkl'
    CONFIG["MODEL_PATH"] = 'best_unet_pca_model.pth'
elif CONFIG["REDUCER_TYPE"] == "NMF":
    # For when you run the NMF experiment later
    CONFIG["REDUCER_PATH"] = 'nmf_model.pkl'
    CONFIG["MODEL_PATH"] = 'best_unet_nmf_model.pth'

# --- 2. METRICS CALCULATION ---
# These functions will help us calculate our key metrics per batch
def calculate_metrics(pred, target, num_classes):
    pred = torch.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    
    pred_labels_flat = pred_labels.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()

    # Ignore the "unlabeled" class (0) for metrics
    mask = target_flat != 0
    
    # Calculate F1-Score
    f1 = f1_score(target_flat[mask], pred_labels_flat[mask], average='macro', zero_division=0)
    
    # Calculate IoU and Dice for each class (1, 2, 3, 4)
    iou_per_class = []
    dice_per_class = []
    
    for cls in range(1, num_classes): # Start from 1 to ignore unlabeled
        pred_inds = pred_labels == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        # Dice Score = 2 * |A intersect B| / (|A| + |B|)
        dice = (2. * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-8)
        
        iou = intersection / (union + 1e-8)
        
        iou_per_class.append(iou)
        dice_per_class.append(dice)
        
    return f1, np.mean(iou_per_class), np.mean(dice_per_class)

# --- 3. MAIN EVALUATION SCRIPT ---
if __name__ == "__main__":
    print(f"--- Starting Final Evaluation ---")
    print(f"Evaluating model: {CONFIG['MODEL_PATH']}")

    # Load reducer
    reducer = joblib.load(CONFIG["REDUCER_PATH"])
    
    # Create Test Dataset and DataLoader
    test_dataset = HSIDataset(csv_file='test.csv', apply_augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=os.cpu_count())

    # Initialize and load the trained model
    model = UNet(n_channels=CONFIG["NUM_COMPONENTS"], n_classes=CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"]))
    model.eval()

    all_f1 = []
    all_iou = []
    all_dice = []

    with torch.no_grad():
        for hsi_tensor, mask_tensor in tqdm(test_loader, desc="Evaluating on Test Set"):
            hsi_tensor, mask_tensor = hsi_tensor.to(CONFIG["DEVICE"]), mask_tensor.to(CONFIG["DEVICE"])

            # Apply dimensionality reduction
            b, c, h, w = hsi_tensor.shape
            pixels = hsi_tensor.permute(0, 2, 3, 1).reshape(b * h * w, c)
            reduced_pixels = torch.from_numpy(reducer.transform(pixels.cpu().numpy())).float().to(CONFIG["DEVICE"])
            reduced_hsi = reduced_pixels.reshape(b, h, w, CONFIG["NUM_COMPONENTS"]).permute(0, 3, 1, 2)
            
            # Get model predictions
            outputs = model(reduced_hsi)
            
            # Calculate metrics for this batch
            f1, iou, dice = calculate_metrics(outputs, mask_tensor, CONFIG["NUM_CLASSES"])
            all_f1.append(f1)
            all_iou.append(iou)
            all_dice.append(dice)

    # --- 4. PRINT FINAL REPORT ---
    avg_f1 = np.mean(all_f1)
    avg_iou = np.mean(all_iou)
    avg_dice = np.mean(all_dice)
    
    print("\n--- Evaluation Complete ---")
    print("Metrics are averaged across all test batches.")
    print("Class 0 (Unlabeled) is ignored in calculations.\n")
    
    print(f"📈 Average Macro F1-Score: {avg_f1:.4f}")
    print(f"📐 Average Intersection over Union (IoU): {avg_iou:.4f}")
    print(f"🎯 Average Dice Score (DSC): {avg_dice:.4f}")
    print("\n----------------------------------")