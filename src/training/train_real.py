import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.cnn_2d import SimpleCNN
from preprocessing.luna16_dataset import LUNA16PatchDataset


def weighted_bce_loss(predictions, targets, pos_weight=10.0):
    weights = targets * pos_weight + (1 - targets) * 1.0
    loss_fn = nn.BCELoss(weight=weights)
    return loss_fn(predictions, targets)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for patches, labels in dataloader:
        patches = patches.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        predictions = model(patches)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for patches, labels in dataloader:
        patches = patches.to(device)
        labels_col = labels.to(device).unsqueeze(1)
        predictions = model(patches)
        loss = loss_fn(predictions, labels_col)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_negatives", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("\n[1/4] Loading dataset...")
    full_dataset = LUNA16PatchDataset(subsets=args.subsets, max_candidates=args.max_negatives)
    n_total = len(full_dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    print(f"      Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print("\n[2/4] Building model...")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()  # or weighted_bce_loss for class imbalance

    print("\n[3/4] Training...")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - start

        print(f"      Epoch {epoch}/{args.epochs} — "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print(f"      -> saved new best checkpoint to {ckpt_path}")

    print("\n[4/4] Done.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()