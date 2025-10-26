# src/data_handling/hsi_dataset.py (UPGRADED FOR SSF-NET)

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import spectral
import cv2

class HSIDataset(Dataset):
    """
    Upgraded dataset class that returns two types of HSI data for the SSF-Net:
    1. Spatial data (dimensionality-reduced)
    2. Spectral data (raw resampled)
    """
    def __init__(self, csv_file, reducer, patch_size=256, apply_augmentation=False):
        self.manifest = pd.read_csv(csv_file)
        self.reducer = reducer # The trained PCA or NMF model
        self.patch_size = patch_size
        self.apply_augmentation = apply_augmentation
        
        self.n_bands = 128
        self.standard_wavelengths = np.linspace(450, 900, self.n_bands)
        
        self.patches = []
        
        print(f"Calculating patches for {len(self.manifest)} images...")
        for idx in range(len(self.manifest)):
            gt_path = self.manifest.loc[idx, 'gt_path']
            try:
                gt_info = spectral.envi.read_envi_header(gt_path)
                h, w = int(gt_info['lines']), int(gt_info['samples'])
                stride = patch_size // 2
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        self.patches.append((idx, y, x))
            except Exception as e:
                print(f"--- WARNING: Could not read {gt_path}. Skipping. Error: {e}")

        print(f"Initialized SSF-Net dataset from '{csv_file}'. Found {len(self.patches)} total patches.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image_idx, y_start, x_start = self.patches[idx]
        y_end, x_end = y_start + self.patch_size, x_start + self.patch_size

        hsi_path = self.manifest.loc[image_idx, 'hsi_path']
        try:
            # --- Load and Preprocess Data (as before) ---
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

            # --- *** NEW LOGIC FOR DUAL OUTPUT *** ---
            
            # 1. Prepare SPECTRAL data (128 channels)
            hsi_tensor_spectral = torch.from_numpy(resampled_patch.copy().transpose(2, 0, 1)).float()

            # 2. Prepare SPATIAL data (15 channels)
            h, w, c = resampled_patch.shape
            pixels = resampled_patch.reshape(h * w, c)
            
            # Use the provided reducer to transform the pixels
            reduced_pixels = self.reducer.transform(pixels)
            
            # Reshape back to an image format
            spatial_image = reduced_pixels.reshape(h, w, -1) # The -1 infers the number of components
            hsi_tensor_spatial = torch.from_numpy(spatial_image.copy().transpose(2, 0, 1)).float()
            
            # 3. Prepare the mask
            mask_tensor = torch.from_numpy(mask_patch.copy()).long()

            return hsi_tensor_spatial, hsi_tensor_spectral, mask_tensor

        except Exception as e:
            # Create dummy tensors for all three outputs in case of an error
            print(f"ERROR loading patch from {hsi_path} at (y:{y_start}, x:{x_start}): {e}")
            n_components = self.reducer.n_components_
            dummy_spatial = torch.zeros((n_components, self.patch_size, self.patch_size))
            dummy_spectral = torch.zeros((self.n_bands, self.patch_size, self.patch_size))
            dummy_mask = torch.zeros((self.patch_size, self.patch_size)).long()
            return dummy_spatial, dummy_spectral, dummy_mask