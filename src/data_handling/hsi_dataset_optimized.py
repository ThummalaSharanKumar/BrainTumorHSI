# src/data_handling/hsi_dataset_optimized.py

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import spectral
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress spectral warning about header parameters
spectral.settings.envi_support_nonlowercase_params = True

class HSIDataset(Dataset):
    """
    Optimized HSIDataset using Albumentations for augmentation and normalization.
    Designed for models like U-Net/TransUNet that take dimensionality-reduced spatial input.
    """
    def __init__(self, csv_file, reducer, patch_size=256, apply_augmentation=False, mean=None, std=None):
        """
        Args:
            csv_file (string): Path to the csv file (train.csv, val.csv, etc.).
            reducer (object): Trained scikit-learn dimensionality reducer (PCA/NMF).
            patch_size (int): The height and width of the square patches.
            apply_augmentation (bool): Whether to apply data augmentation.
            mean (np.array): Pre-calculated mean for normalization (per channel).
            std (np.array): Pre-calculated standard deviation for normalization (per channel).
        """
        self.manifest = pd.read_csv(csv_file)
        self.reducer = reducer
        self.patch_size = patch_size
        self.apply_augmentation = apply_augmentation
        self.mean = mean
        self.std = std
        self.n_bands_resampled = 128 # Number of bands after resampling
        self.n_components_reduced = getattr(reducer, 'n_components_', 15) # Get number of components from reducer
        self.standard_wavelengths = np.linspace(450, 900, self.n_bands_resampled)
        self.patches = []

        print(f"Calculating patches for {len(self.manifest)} images...")
        for idx in range(len(self.manifest)):
            gt_path = self.manifest.loc[idx, 'gt_path']
            try:
                # Read header only to get dimensions, avoids loading full mask
                gt_info = spectral.envi.read_envi_header(gt_path)
                h, w = int(gt_info['lines']), int(gt_info['samples'])
                stride = patch_size // 2 # Overlapping stride
                # Ensure patch fits within image dimensions
                for y in range(0, max(0, h - patch_size + 1), stride):
                    for x in range(0, max(0, w - patch_size + 1), stride):
                        self.patches.append((idx, y, x))
            except Exception as e:
                print(f"--- WARNING: Could not read header for {gt_path}. Skipping. Error: {e}")

        if not self.patches:
             raise ValueError(f"Initialization failed: No valid patches calculated from '{csv_file}'. Check file paths and image dimensions.")

        print(f"Initialized dataset from '{csv_file}'. Found {len(self.patches)} total patches.")

        # --- Define Augmentation Pipelines ---
        # Base transform (applied always, before augmentation)
        self.base_transform = A.Compose([
             A.Resize(patch_size, patch_size), # Ensure patch size consistency
        ])

        # Augmentation transform (applied only during training)
        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05), # Removed alpha_affine
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
        ]) if self.apply_augmentation else None # Only create if needed

        # Normalization and Tensor conversion (applied always, after augmentation)
        # Use placeholder mean/std if none provided, but stats should be calculated from training set
        _mean = self.mean if self.mean is not None else [0.0] * self.n_components_reduced
        _std = self.std if self.std is not None else [1.0] * self.n_components_reduced
        self.normalize_transform = A.Compose([
             A.Normalize(mean=_mean, std=_std, max_pixel_value=1.0), # Assumes input data is [0, 1]
             ToTensorV2(), # Converts numpy (H, W, C) to torch (C, H, W) and scales image to [0, 1] if max_pixel=255
        ])
        # ------------------------------------

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if idx >= len(self.patches):
            raise IndexError("Index out of bounds")

        image_idx, y_start, x_start = self.patches[idx]
        y_end = y_start + self.patch_size
        x_end = x_start + self.patch_size

        hsi_path = self.manifest.loc[image_idx, 'hsi_path']

        try:
            # --- Load necessary data using file handles ---
            hsi_image_file = spectral.open_image(hsi_path)
            raw_h, raw_w, _ = hsi_image_file.shape # Get original dimensions
            gt_mask_file = spectral.open_image(self.manifest.loc[image_idx, 'gt_path'])
            base_dir = os.path.dirname(hsi_path)
            white_ref_file = spectral.open_image(os.path.join(base_dir, 'whiteReference.hdr'))
            dark_ref_file = spectral.open_image(os.path.join(base_dir, 'darkReference.hdr'))

            # --- Load reference images and resize if needed ---
            white_ref = white_ref_file.read_band(0)
            dark_ref = dark_ref_file.read_band(0)
            if white_ref.shape != (raw_h, raw_w):
                white_ref = cv2.resize(white_ref, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
            if dark_ref.shape != (raw_h, raw_w):
                dark_ref = cv2.resize(dark_ref, (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

            # --- Extract Patches ---
            # Extract only the needed patch slice from files/resized arrays
            white_ref_patch = white_ref[y_start:y_end, x_start:x_end][..., np.newaxis]
            dark_ref_patch = dark_ref[y_start:y_end, x_start:x_end][..., np.newaxis]
            hsi_patch = hsi_image_file[y_start:y_end, x_start:x_end, :] # Load only the patch
            mask_patch = gt_mask_file.read_band(0)[y_start:y_end, x_start:x_end]

            # --- Preprocessing (Calibration, Resampling) ---
            calibrated_patch = np.divide(hsi_patch.astype(np.float32) - dark_ref_patch,
                                         white_ref_patch - dark_ref_patch + 1e-8)
            calibrated_patch = np.clip(calibrated_patch, 0, 1)

            original_wavelengths = np.array([float(w) for w in hsi_image_file.metadata['wavelength']])
            resampled_patch = np.apply_along_axis(
                lambda s: np.interp(self.standard_wavelengths, original_wavelengths, s),
                axis=2, arr=calibrated_patch
            ).astype(np.float32) # Ensure float32

            # --- Apply Reducer to get Spatial Data ---
            h, w, c = resampled_patch.shape
            pixels = resampled_patch.reshape(h * w, c)
            pixel_data = pixels
            # Handle NMF dtype requirement if reducer is NMF
            if 'NMF' in str(self.reducer.__class__):
                 pixel_data = pixels.astype(np.float64)
            reduced_pixels = self.reducer.transform(pixel_data)
            # Ensure spatial image is float32 for Albumentations
            spatial_image = reduced_pixels.reshape(h, w, self.n_components_reduced).astype(np.float32)

            # --- Apply Albumentations ---
            transformed = self.base_transform(image=spatial_image, mask=mask_patch)
            spatial_image, mask_patch = transformed['image'], transformed['mask']

            if self.apply_augmentation and self.aug_transform:
                augmented = self.aug_transform(image=spatial_image, mask=mask_patch)
                spatial_image, mask_patch = augmented['image'], augmented['mask']

            # Apply final normalization and ToTensor
            normalized = self.normalize_transform(image=spatial_image, mask=mask_patch)
            hsi_tensor_spatial = normalized['image']
            mask_tensor = normalized['mask'].long() # Ensure mask is LongTensor

            return hsi_tensor_spatial, mask_tensor

        except Exception as e:
            print(f"ERROR loading patch from {hsi_path} at (y:{y_start}, x:{x_start}): {e}")
            # Return dummy tensors with correct channel number
            dummy_spatial = torch.zeros((self.n_components_reduced, self.patch_size, self.patch_size), dtype=torch.float)
            dummy_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.long)
            return dummy_spatial, dummy_mask