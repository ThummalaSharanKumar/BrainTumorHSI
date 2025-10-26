# src/models/ssf_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.unet import DoubleConv, Down, Up, OutConv # We'll reuse the U-Net building blocks

class SpectralBranch(nn.Module):
    """
    This branch acts as the 'Spectral Expert'. It uses 1D convolutions to
    process the raw spectral signature of each pixel.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spec_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_channels, kernel_size=1) # Project to feature space
        )

    def forward(self, x_spec):
        # Input x_spec shape: (Batch, Channels, Height * Width)
        return self.spec_conv(x_spec)

class SpatialBranch(nn.Module):
    """
    This branch acts as the 'Spatial Expert'. It's essentially the
    encoder part of our U-Net, learning spatial features from the
    dimensionality-reduced image.
    """
    def __init__(self, n_channels):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

    def forward(self, x_spatial):
        x1 = self.inc(x_spatial)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Return all intermediate feature maps for the decoder (skip connections)
        return x1, x2, x3, x4, x5

class SSFNet(nn.Module):
    """
    The Spatio-Spectral Fusion Network (SSF-Net).
    It combines the two branches and uses a U-Net-like decoder to
    generate the final segmentation map.
    """
    def __init__(self, n_spatial_channels, n_spectral_channels, n_classes):
        super(SSFNet, self).__init__()
        
        # Define the two branches
        self.spatial_branch = SpatialBranch(n_spatial_channels)
        
        # The spectral branch will output 512 features to match the deepest spatial feature map
        self.spectral_branch = SpectralBranch(n_spectral_channels, out_channels=512)

        # Define the decoder (similar to U-Net's 'Up' path)
        self.up1 = Up(1024 + 512, 256) # Fusion happens here
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_spatial, x_spectral_raw):
        # x_spatial: (Batch, N_Components, Height, Width) - PCA/NMF reduced data
        # x_spectral_raw: (Batch, N_Bands, Height, Width) - Raw resampled data

        # 1. Process data through the Spatial Branch
        # This gives us the feature maps for the skip connections
        x1_s, x2_s, x3_s, x4_s, x5_s = self.spatial_branch(x_spatial)
        
        # 2. Process data through the Spectral Branch
        # Reshape the raw spectral data for the 1D CNN
        b, c, h, w = x_spectral_raw.shape
        x_spec_reshaped = x_spectral_raw.permute(0, 2, 3, 1).reshape(b, h * w, c).permute(0, 2, 1)
        
        spec_features = self.spectral_branch(x_spec_reshaped) # Shape: (B, 512, H*W)
        
        # Reshape the spectral features back into a 2D spatial format
        spec_features_map = spec_features.permute(0, 2, 1).reshape(b, h, w, 512).permute(0, 3, 1, 2)
        
        # To match the size of the deepest spatial feature map (x5_s), we downsample it
        spec_features_downsampled = F.interpolate(spec_features_map, size=x5_s.shape[2:], mode='bilinear', align_corners=False)

        # 3. Fuse the features from both branches at the bottleneck
        # We concatenate the deepest spatial features with the spectral features
        fused = torch.cat([x5_s, spec_features_downsampled], dim=1)
        
        # 4. Decode the fused features, using the spatial skip connections
        x = self.up1(fused, x4_s)
        x = self.up2(x, x3_s)
        x = self.up3(x, x2_s)
        x = self.up4(x, x1_s)
        logits = self.outc(x)
        
        return logits