import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Dataset for 12-band JP2 images and PNG binary masks
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
            image = image / (image.max() + 1e-6)

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# U-Net architecture for segmentation
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.middle = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        m = self.middle(self.pool4(d4))
        u4 = self.conv4(torch.cat([self.up4(m), d4], 1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], 1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], 1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], 1))
        return self.out(u1)

# Training and Evaluation Loops
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()
    return val_loss / len(loader)

# Entry point
def run_training(data_root, epochs=10, batch_size=8, lr=1e-4, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_paths(split):
        img_dir = os.path.join(data_root, split, "images")
        mask_dir = os.path.join(data_root, split, "masks")
        image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jp2")])
        mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        return image_paths, mask_paths

    train_imgs, train_masks = get_paths("train")
    val_imgs, val_masks = get_paths("val")

    train_ds = SegmentationDataset(train_imgs, train_masks)
    val_ds = SegmentationDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "unet_best_model.pth")
            print("✔️ Best model saved: unet_best_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stopping patience: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

if __name__ == "__main__":
    run_training(data_root=r"C:\Users\wasif\Desktop\final", epochs=10, batch_size=8)
