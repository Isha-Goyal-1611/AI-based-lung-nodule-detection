import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np

from models.cnn_2d import SimpleCNN
from preprocessing.luna16_dataset import LUNA16PatchDataset
from evaluation.clinical_metrics import compute_clinical_metrics

CHECKPOINT_PATH = "checkpoints/best_model.pt"


def load_trained_model():
    model = SimpleCNN()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss at save time: {checkpoint['val_loss']:.4f}")
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


def main():
    print("[1/3] Loading trained model...")
    model = load_trained_model()

    print("\n[2/3] Loading dataset (same subset used for training)...")
    dataset = LUNA16PatchDataset(subsets=[0], max_candidates=200)
    y_true, y_pred_prob = get_predictions(model, dataset)
    print(f"      Prediction range: [{y_pred_prob.min():.4f}, {y_pred_prob.max():.4f}]")
    print(f"      Prediction mean:  {y_pred_prob.mean():.4f}")
    print(f"      Total samples evaluated: {len(y_true)}")
    print(f"      Actual positives: {int(y_true.sum())}")

    print("\n[3/3] Computing real clinical metrics...")
    metrics = compute_clinical_metrics(y_true, y_pred_prob, threshold=0.5)

    print("=" * 50)
    print("REAL MODEL EVALUATION (not synthetic)")
    print("=" * 50)
    print(f"TP (caught nodules):     {metrics['TP']}")
    print(f"FP (false alarms):       {metrics['FP']}")
    print(f"TN (correct negatives):  {metrics['TN']}")
    print(f"FN (missed nodules):     {metrics['FN']}")
    print(f"Sensitivity:  {metrics['sensitivity']:.1%}")
    print(f"Specificity:  {metrics['specificity']:.1%}")
    print(f"PPV:          {metrics['PPV']:.1%}")
    print(f"NPV:          {metrics['NPV']:.1%}")
    print(f"Accuracy:     {metrics['accuracy']:.1%}")


if __name__ == "__main__":
    main()