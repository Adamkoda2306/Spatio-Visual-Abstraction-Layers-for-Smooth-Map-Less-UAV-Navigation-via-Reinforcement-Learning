"""
Multi-scale encoder-decoder U-Net perception front-end (Section 3.2, Fig. 1-A).

Encoder:  (256,256,3) -> Conv(64,3x3,ReLU)  -> MaxPool(2x2) -> (128,128,64)
          (128,128,64)-> Conv(128,3x3,ReLU) -> MaxPool(2x2) -> (64,64,128)
Bottleneck: Conv(256,3x3,ReLU) on (64,64,128) -> (64,64,256)
Decoder:  UpSample(2x2) + skip(128,128,128) -> Conv(128,3x3,ReLU) -> (128,128,128)
          UpSample(2x2) + skip(256,256,64)  -> Conv(64,3x3,ReLU)  -> (256,256,64)
Head:     Conv(1,1x1) -> Sigmoid -> (256,256,1) obstacle probability map
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2

import config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Shallow 2-level U-Net matching Fig. 1-A of the paper."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)      # (256,256,3)  -> (256,256,64)
        self.pool1 = nn.MaxPool2d(2)                # -> (128,128,64)

        self.enc2 = ConvBlock(64, 128)              # (128,128,64) -> (128,128,128)
        self.pool2 = nn.MaxPool2d(2)                # -> (64,64,128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)       # (64,64,128)  -> (64,64,256)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")   # -> (128,128,256)
        self.dec1 = ConvBlock(256 + 128, 128)                     # skip enc2 -> (128,128,128)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")   # -> (256,256,128)
        self.dec2 = ConvBlock(128 + 64, 64)                       # skip enc1 -> (256,256,64)

        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.head(d2)
        return self.sigmoid(out)  # M_prob in [0,1]^(256x256x1)


class ObstaclePerceptionModule:
    """
    Wraps the U-Net for inference-only use inside the RL environment.
    Loads pretrained segmentation weights if available at config.UNET_WEIGHTS_PATH,
    otherwise runs with (fixed, seeded) randomly-initialized weights so the module
    still acts as a deterministic multi-scale feature extractor.
    """

    def __init__(self, device=None, weights_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)

        weights_path = weights_path or config.UNET_WEIGHTS_PATH
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[UNet] Loaded pretrained weights from {weights_path}")
        else:
            torch.manual_seed(42)
            print(f"[UNet] No weights found at {weights_path} — using fixed random init. "
                  f"Train with train_unet.py for real obstacle segmentation quality.")

        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        rgb_image: HxWx3 uint8 array (any resolution, will be resized to 256x256)
        returns:   256x256 float32 obstacle probability map in [0,1]
        """
        img = cv2.resize(rgb_image, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        prob = self.model(tensor)  # (1,1,256,256)
        return prob.squeeze(0).squeeze(0).cpu().numpy()
