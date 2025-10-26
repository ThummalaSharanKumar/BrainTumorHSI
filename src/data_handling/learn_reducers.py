# src/data_handling/learn_reducers.py (SYNCHRONIZED VERSION)

import numpy as np
import pandas as pd
import os
import spectral
from sklearn.decomposition import PCA, NMF
import joblib
import cv2

print("--- Starting Phase 2.1: Learning Dimensionality Reducers (Synchronized) ---")

# --- 1. CONFIGURATION ---
TRAIN_CSV = 'train.csv'
NUM_COMPONENTS = 15
NUM_SAMPLES_TO_LEARN = 5
PCA_MODEL_PATH = 'pca_model.pkl'
NMF_MODEL_PATH = 'nmf_model.pkl'

# --- THIS IS THE CRITICAL CHANGE: We use the SAME standardization as the dataset class ---
N_BANDS_STANDARDIZED = 128
STANDARD_WAVELENGTHS = np.linspace(450, 900, N_BANDS_STANDARDIZED)

# --- 2. LOAD & PREPROCESS DATA ---
train_df = pd.read_csv(TRAIN_CSV)
sample_df = train_df.sample(n=min(NUM_SAMPLES_TO_LEARN, len(train_df)), random_state=42)
print(f"Loading {len(sample_df)} sample HSI cubes to learn transformations...")

all_pixels = []

for _, row in sample_df.iterrows():
    hsi_path = row['hsi_path']
    try:
        hsi_image_file = spectral.open_image(hsi_path)
        raw_h, raw_w, _ = hsi_image_file.shape
        
        base_dir = os.path.dirname(hsi_path)
        
        white_ref_file = spectral.open_image(os.path.join(base_dir, 'whiteReference.hdr'))
        dark_ref_file = spectral.open_image(os.path.join(base_dir, 'darkReference.hdr'))

        white_ref = white_ref_file.read_band(0)
        if white_ref.shape != (raw_h, raw_w):
            white_ref = cv2.resize(white_ref, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        dark_ref = dark_ref_file.read_band(0)
        if dark_ref.shape != (raw_h, raw_w):
            dark_ref = cv2.resize(dark_ref, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

        calibrated_hsi = np.divide(hsi_image_file.load() - dark_ref[:, :, np.newaxis], 
                                   white_ref[:, :, np.newaxis] - dark_ref[:, :, np.newaxis] + 1e-8)
        calibrated_hsi = np.clip(calibrated_hsi, 0, 1)

        # --- USE SPECTRAL RESAMPLING (SAME AS DATASET) ---
        original_wavelengths = np.array([float(w) for w in hsi_image_file.metadata['wavelength']])
        
        resampled_hsi = np.apply_along_axis(
            lambda spectrum: np.interp(STANDARD_WAVELENGTHS, original_wavelengths, spectrum),
            axis=2,
            arr=calibrated_hsi
        )
        
        h, w, c = resampled_hsi.shape
        all_pixels.append(resampled_hsi.reshape(h * w, c))

    except Exception as e:
        print(f"--- ERROR processing {hsi_path}: {e}")

pixel_matrix = np.concatenate(all_pixels, axis=0)
if pixel_matrix.shape[0] > 500000:
    rng = np.random.default_rng(42)
    pixel_matrix = rng.choice(pixel_matrix, 500000, replace=False)

print(f"Created a matrix of {pixel_matrix.shape[0]} pixels with {pixel_matrix.shape[1]} features each.")

# --- 3. LEARN AND SAVE MODELS ---
print(f"\nTraining PCA model with {NUM_COMPONENTS} components...")
pca = PCA(n_components=NUM_COMPONENTS)
pca.fit(pixel_matrix)
joblib.dump(pca, PCA_MODEL_PATH)
print(f"PCA model saved to: {os.path.abspath(PCA_MODEL_PATH)}")

print(f"\nTraining NMF model with {NUM_COMPONENTS} components...")
nmf = NMF(n_components=NUM_COMPONENTS, init='random', random_state=42, max_iter=500, tol=1e-3)
nmf.fit(pixel_matrix)
joblib.dump(nmf, NMF_MODEL_PATH)
print(f"NMF model saved to: {os.path.abspath(NMF_MODEL_PATH)}")

print("\n--- Success! Synchronized reduction models are trained and saved. ---")