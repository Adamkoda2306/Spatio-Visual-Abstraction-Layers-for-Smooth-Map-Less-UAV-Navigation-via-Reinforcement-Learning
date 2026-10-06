"""
Optional supervised pretraining for the U-Net perception front-end.

The paper uses the U-Net purely as a fixed perception module inside the RL loop;
it does not specify a labeled dataset. If you have (image, obstacle-mask) pairs
(e.g. exported from AirSim segmentation view + thresholding, or hand-labeled),
this script trains the network with a standard BCE segmentation loss and saves
weights to weights/unet.pt, which unet.ObstaclePerceptionModule will then load
automatically.

Expected data layout:
    data/images/*.png   (RGB, any size, will be resized to 256x256)
    data/masks/*.png     (single channel, same filename, 0=free, 255=obstacle)

Usage:
    env\\Scripts\\python.exe train_unet.py --data_dir data --epochs 20
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from unet import UNet


class SegDataset(Dataset):
    def __init__(self, data_dir, size=config.IMG_SIZE):
        self.images = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
        self.masks = sorted(glob.glob(os.path.join(data_dir, "masks", "*")))
        assert len(self.images) == len(self.masks) and len(self.images) > 0, \
            "images/ and masks/ must contain the same non-zero number of matching files"
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask.astype(np.float32) / 255.0)[None, ...]

        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {total_loss / len(dataset):.4f}")

    os.makedirs(os.path.dirname(config.UNET_WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.UNET_WEIGHTS_PATH)
    print(f"Saved U-Net weights to {config.UNET_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
