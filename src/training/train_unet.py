"""
src/training/train_unet.py

Trains MiniUNet on (CT slice, lung mask) pairs from
preprocessing/precompute_lung_masks.py. Tracks Dice score and IoU
(segmentation-specific metrics — different from classification
accuracy/sensitivity used for the nodule/malignancy classifiers).

Usage:
    python train_unet.py --data lung_mask_patches.pt \
        --checkpoint_out checkpoints/unet_model.pt --epochs 20
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import MiniUNet


def dice_score(pred_mask, true_mask, epsilon=1e-6):
    pred_binary = (pred_mask >= 0.5).float()
    intersection = (pred_binary * true_mask).sum(dim=[1, 2, 3])
    union = pred_binary.sum(dim=[1, 2, 3]) + true_mask.sum(dim=[1, 2, 3])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()


def iou_score(pred_mask, true_mask, epsilon=1e-6):
    pred_binary = (pred_mask >= 0.5).float()
    intersection = (pred_binary * true_mask).sum(dim=[1, 2, 3])
    union = ((pred_binary + true_mask) >= 1).float().sum(dim=[1, 2, 3])
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += loss_fn(preds, masks).item()
        total_dice += dice_score(preds, masks)
        total_iou += iou_score(preds, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint_out", default="checkpoints/unet_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.data)
    images, masks = data["images"], data["masks"]
    print(f"Images: {images.shape}  Masks: {masks.shape}")

    full_dataset = TensorDataset(images, masks)
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MiniUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()  # MiniUNet already applies sigmoid internally

    os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  dice: {val_dice:.4f}  iou: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou},
                       args.checkpoint_out)
            print("  -> saved new best checkpoint")

    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
    