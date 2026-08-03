"""
src/training/train_resnet.py

Trains a ResNet-18 (ImageNet-pretrained, first conv layer replaced for
single-channel CT input) on precomputed LUNA16 patches. Resizes 32x32
patches to 224x224 to match ResNet's expected input size.

RESULT FROM TESTING (subset0+1+2 train, subset3 held out as unseen test):
ResNet-18 outperformed SimpleCNN on every metric:
    SimpleCNN:  acc=77.7%  sens=71.5%  spec=81.0%
    ResNet-18:  acc=87.8%  sens=83.5%  spec=90.0%
Transfer learning from ImageNet gave a real, meaningful improvement here,
despite the domain gap between natural photos and CT scans.

Usage:
    python train_resnet.py --patches precomputed_patches.pt \
        --checkpoint_out checkpoints/resnet18_model.pt --epochs 15
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_transfer import build_resnet_model


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = loss_fn(model(patches), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device).unsqueeze(1)
        total_loss += loss_fn(model(patches), labels).item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--checkpoint_out", default="checkpoints/resnet18_model.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.patches)
    patches, labels = data["patches"], data["labels"]
    print(f"Original patches: {patches.shape}")

    # ResNet-18 expects 224x224; our patches are 32x32
    patches_resized = F.interpolate(patches, size=(224, 224), mode="bilinear",
                                     align_corners=False)
    print(f"Resized patches: {patches_resized.shape}")

    full_dataset = TensorDataset(patches_resized, labels)
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_resnet_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()  # raw logits in, sigmoid applied internally by the loss

    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, args.checkpoint_out)
            print("  -> saved new best checkpoint")

    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()