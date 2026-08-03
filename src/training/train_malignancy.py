"""
src/training/train_malignancy.py

Trains MalignancyClassifier on precomputed LIDC-IDRI 3D patches (see
preprocessing/lidc_extraction.py). Binarizes malignancy scores: benign
(<=2.5) vs malignant (>=3.5), dropping ambiguous middle cases.

KNOWN LIMITATION (be upfront about this in write-ups): with ~50-140 training
samples, this model (millions of parameters) overfits quickly — best val_loss
typically lands in the first few epochs, after which train_loss keeps
dropping while val_loss climbs. More patients would help; this is a real,
documented constraint of the dataset size, not a bug.

Usage:
    python train_malignancy.py --data malignancy_data.pt \
        --checkpoint_out checkpoints/malignancy_model.pt --epochs 30
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.malignancy_classifier import MalignancyClassifier


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for patch, clinical, label in loader:
        patch, clinical, label = patch.to(device), clinical.to(device), label.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = loss_fn(model(patch, clinical), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for patch, clinical, label in loader:
        patch, clinical, label = patch.to(device), clinical.to(device), label.to(device).unsqueeze(1)
        total_loss += loss_fn(model(patch, clinical), label).item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint_out", default="checkpoints/malignancy_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(args.data)
    patches, labels, clinical = data["patches"], data["labels"], data["clinical"]

    mask = (labels <= 2.5) | (labels >= 3.5)
    patches, clinical = patches[mask], clinical[mask]
    binary_labels = (labels[mask] >= 3.5).float()
    print(f"After dropping ambiguous: {len(binary_labels)} samples "
          f"(Benign: {(binary_labels == 0).sum().item()}, "
          f"Malignant: {(binary_labels == 1).sum().item()})")

    full_dataset = TensorDataset(patches, clinical, binary_labels)
    n_val = max(1, int(0.2 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MalignancyClassifier().to(device)
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


if __name__ == "__main__":
    main()
