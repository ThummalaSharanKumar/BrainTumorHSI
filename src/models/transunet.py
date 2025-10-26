# src/models/transunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # PyTorch Image Models library
from timm.models.vision_transformer import PatchEmbed, Block

# --- Re-used U-Net Components ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # Use bilinear interpolation for upsampling by default
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Alternatively, use ConvTranspose2d
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        # Double convolution layer after upsampling and concatenation
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # Upsample the feature map from the previous layer
        x1 = self.up(x1)
        # Pad x1 to match the spatial dimensions of x2 (from skip connection) if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate the upsampled features (x1) with the skip connection features (x2)
        x = torch.cat([x2, x1], dim=1)
        # Apply double convolution
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to map features to the number of classes"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

# --- TransUNet Specific Components ---

class ConvBnRelu(nn.Sequential):
    """Basic Conv-BatchNorm-ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class VisionTransformer(nn.Module):
    """ Simplified Vision Transformer backbone (Encoder Only) """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # Shape: (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store features from intermediate blocks if needed for skip connections (not standard in basic TransUNet)
        # features = []
        for blk in self.blocks:
            x = blk(x)
            # features.append(x) # Example if needed

        x = self.norm(x)
        # Return features excluding the CLS token, ready for reshaping
        return x[:, 1:] # Shape: (B, num_patches, embed_dim)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class TransUNet(nn.Module):
    """
    TransUNet architecture combining a ViT backbone with a U-Net decoder.
    Args:
        n_channels (int): Number of input channels (e.g., 15 for PCA components).
        n_classes (int): Number of output segmentation classes.
        img_size (int): Size of the input image patches (must be square).
        vit_patch_size (int): Patch size for the ViT encoder.
        vit_embed_dim (int): Embedding dimension for the ViT.
        vit_depth (int): Number of Transformer blocks in the ViT.
        vit_num_heads (int): Number of attention heads in the ViT.
        decoder_channels (tuple): Number of channels in each stage of the U-Net decoder.
        skip_channels (tuple): Number of channels from the CNN encoder stages for skip connections (if using a hybrid CNN encoder). TransUNet often uses feature maps derived from the ViT patch embeddings or simple Conv layers.
    """
    def __init__(self, n_channels=15, n_classes=5, img_size=256, vit_patch_size=16,
                 vit_embed_dim=768, vit_depth=12, vit_num_heads=12):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.n_classes = n_classes
        self.n_channels = n_channels # Number of input PCA/NMF components
        self.patch_size = vit_patch_size

        # --- CNN Feature Extractor (Simplified - Can be replaced with ResNet) ---
        # This part extracts initial features and skip connections before the ViT
        # We need layers that output features at different spatial resolutions
        # Example using simple Conv blocks:
        self.conv1 = DoubleConv(n_channels, 64)        # Output: B, 64, H, W
        self.pool1 = nn.MaxPool2d(2)                   # Output: B, 64, H/2, W/2
        self.conv2 = DoubleConv(64, 128)               # Output: B, 128, H/2, W/2
        self.pool2 = nn.MaxPool2d(2)                   # Output: B, 128, H/4, W/4
        self.conv3 = DoubleConv(128, 256)              # Output: B, 256, H/4, W/4
        self.pool3 = nn.MaxPool2d(2)                   # Output: B, 256, H/8, W/8
        self.conv4 = DoubleConv(256, 512)              # Output: B, 512, H/8, W/8

        # Input projection for ViT
        # The input to ViT needs 3 channels usually, or needs modification.
        # Here, we'll assume ViT takes the output of conv4, but requires projection if channel counts differ.
        # For simplicity, let's use a ViT that accepts 512 channels directly (requires custom or modified ViT).
        # OR, project 512 channels to ViT's expected input dimension if using a standard ViT (e.g., 3 channels).
        # We will adapt ViT to take 512 channels for simplicity here.
        self.vit_img_size = img_size // (2**3) # Image size fed into ViT is H/8

        # --- Vision Transformer Encoder ---
        self.vit = VisionTransformer(
            img_size=self.vit_img_size,
            patch_size=vit_patch_size,
            in_chans=512, # ViT takes the output of the CNN encoder
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads
        )

        # --- Decoder ---
        # Calculate number of patches along height/width for reshaping ViT output
        self.num_patches_side = self.vit_img_size // vit_patch_size

        # Decoder Upsampling Path (using U-Net components)
        # Input to first Up layer is the reshaped ViT output
        decoder_in_channels = vit_embed_dim
        # Channels need adjustment based on skip connection structure
        self.up1 = Up(decoder_in_channels, 256) # Input: ViT output + Skip from conv3 (512 -> 256)
        self.up2 = Up(256 + 256, 128)           # Input: Up1 output (256) + Skip from conv2 (128) -> Total 384? - Adjust Up channels
        self.up3 = Up(128 + 128, 64)            # Input: Up2 output (128) + Skip from conv1 (64) -> Total 192? - Adjust Up channels

        # Correcting Up block channel counts based on concatenation:
        self.up1_corr = Up(vit_embed_dim + 256, 256)  # ViT(768) + conv3(256) = 1024 -> 256
        self.up2_corr = Up(256 + 128, 128)           # up1_corr(256) + conv2(128) = 384 -> 128
        self.up3_corr = Up(128 + 64, 64)            # up2_corr(128) + conv1(64) = 192 -> 64

        # Final output convolution
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # --- CNN Encoder Path ---
        x1 = self.conv1(x)         # B, 64, H, W
        p1 = self.pool1(x1)        # B, 64, H/2, W/2
        x2 = self.conv2(p1)        # B, 128, H/2, W/2
        p2 = self.pool2(x2)        # B, 128, H/4, W/4
        x3 = self.conv3(p2)        # B, 256, H/4, W/4
        p3 = self.pool3(x3)        # B, 256, H/8, W/8
        x4 = self.conv4(p3)        # B, 512, H/8, W/8 (Input to ViT)

        # --- Transformer Path ---
        vit_output = self.vit(x4) # Shape: (B, num_patches, embed_dim)

        # Reshape ViT output back into spatial feature map
        batch_size = x.shape[0]
        embed_dim = vit_output.shape[-1]
        # Calculate H and W of the patch grid
        num_patches_h = self.vit_img_size // self.patch_size
        num_patches_w = self.vit_img_size // self.patch_size
        vit_output_spatial = vit_output.reshape(batch_size, num_patches_h, num_patches_w, embed_dim)
        vit_output_spatial = vit_output_spatial.permute(0, 3, 1, 2) # Shape: (B, embed_dim, H/N, W/N), where N is patch size factor (e.g., 16)
        # Upsample to match skip connection size (H/8, W/8)
        vit_output_upsampled = F.interpolate(vit_output_spatial, size=x4.shape[2:], mode='bilinear', align_corners=False)

        # --- Decoder Path ---
        # Note: Using the corrected Up blocks
        d1 = self.up1_corr(vit_output_upsampled, x3) # Upsample ViT out, concat with x3(conv3)
        d2 = self.up2_corr(d1, x2)                  # Upsample d1, concat with x2(conv2)
        d3 = self.up3_corr(d2, x1)                  # Upsample d2, concat with x1(conv1)

        # Final output
        logits = self.outc(d3)
        return logits