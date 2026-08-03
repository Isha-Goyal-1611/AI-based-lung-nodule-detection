"""
src/training/train_nodule_gpu.py

Trains SimpleCNN on precomputed LUNA16 patches (see
preprocessing/precompute_luna16_patches.py). Uses GPU if available.

Usage:
    python train_nodule_gpu.py --patches precomputed_patches.pt \
        --checkpoint_out checkpoints/best_model.pt --epochs 20
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_2d import SimpleCNN


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
    parser.add_argument("--patches", required=True, help="Path to precomputed_patches.pt")
    parser.add_argument("--checkpoint_out", default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.patches)
    patches, labels = data["patches"], data["labels"]
    print(f"Patches: {patches.shape}  Labels: {labels.shape}")

    full_dataset = TensorDataset(patches, labels)
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

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
    print(f"Checkpoint saved to: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
