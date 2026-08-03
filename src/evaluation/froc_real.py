import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np

from models.cnn_2d import SimpleCNN
from preprocessing.luna16_dataset import LUNA16PatchDataset

CHECKPOINT_PATH = "checkpoints/best_model.pt"

# Standard LUNA16 FROC operating points: false positives allowed per scan
FP_PER_SCAN_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def load_trained_model():
    model = SimpleCNN()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    all_preds, all_labels = [], []
    for patches, labels in loader:
        predictions = model(patches)
        all_preds.append(predictions.squeeze().numpy())
        all_labels.append(labels.numpy())
    y_pred_prob = np.concatenate([np.atleast_1d(p) for p in all_preds])
    y_true = np.concatenate([np.atleast_1d(l) for l in all_labels])
    return y_true, y_pred_prob


def sensitivity_at_fp_threshold(y_true, y_pred_prob, n_scans, target_fp_per_scan):
    """
    Sweep confidence thresholds; find the highest threshold whose resulting
    false-positive rate (per scan) is <= target_fp_per_scan, and report
    sensitivity at that threshold. This is what a real FROC curve plots.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_sensitivity = 0.0
    best_threshold = None

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        fp_per_scan = FP / n_scans if n_scans > 0 else float("inf")
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        if fp_per_scan <= target_fp_per_scan and sensitivity >= best_sensitivity:
            best_sensitivity = sensitivity
            best_threshold = t

    return best_sensitivity, best_threshold


def main():
    print("[1/3] Loading trained model...")
    model = load_trained_model()

    print("\n[2/3] Loading unseen test data (subset2)...")
    dataset = LUNA16PatchDataset(subsets=[2], max_candidates=200)
    y_true, y_pred_prob = get_predictions(model, dataset)

    n_scans = len(dataset.candidates["seriesuid"].unique())
    print(f"      Samples: {len(y_true)}   Scans: {n_scans}   Positives: {int(y_true.sum())}")

    print("\n[3/3] Computing real FROC — sensitivity at standard FP/scan operating points...")
    print("=" * 55)
    print("REAL FROC EVALUATION (model confidence, not intensity)")
    print("=" * 55)
    print(f"{'FP/scan target':>15} | {'Sensitivity':>12} | {'Threshold used':>15}")
    for target in FP_PER_SCAN_POINTS:
        sens, thresh = sensitivity_at_fp_threshold(y_true, y_pred_prob, n_scans, target)
        thresh_str = f"{thresh:.2f}" if thresh is not None else "none found"
        print(f"{target:>15} | {sens:>11.1%} | {thresh_str:>15}")

    avg_sensitivity = np.mean([
        sensitivity_at_fp_threshold(y_true, y_pred_prob, n_scans, t)[0]
        for t in FP_PER_SCAN_POINTS
    ])
    print(f"\nAverage sensitivity across operating points (competition-style score): "
          f"{avg_sensitivity:.1%}")


if __name__ == "__main__":
    main()