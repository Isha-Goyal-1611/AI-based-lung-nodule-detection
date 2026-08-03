"""
src/evaluation/grad_cam.py

Grad-CAM visualization for the trained SimpleCNN nodule classifier, using a
real trained checkpoint and real labeled patches (not random noise).

HONEST FINDING FROM TESTING: Grad-CAM results on this shallow 2-conv-layer
model were inconsistent across examples — some showed spatially coherent
attention on the candidate lesion, others showed sparse or near-empty
activation despite high model confidence. This likely reflects the shallow
architecture's limited spatial detail at the final conv layer, rather than
a bug. Worth checking multiple examples, not just one, before drawing
conclusions from any single visualization.

Usage:
    python grad_cam.py --checkpoint checkpoints/best_model.pt \
        --patches precomputed_patches.pt --n_examples 4
"""

import argparse

import matplotlib.pyplot as plt
import torch

from models.cnn_2d import SimpleCNN


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap).squeeze().cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap


def load_model(checkpoint_path):
    model = SimpleCNN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss {checkpoint['val_loss']:.4f}")
    return model


def visualize_examples(model, patches, labels, n_examples=4, output_path="gradcam.png"):
    positive_indices = (labels == 1).nonzero().squeeze()[:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(4.5 * n_examples, 9))
    if n_examples == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(positive_indices):
        idx = idx.item()
        patch = patches[idx].unsqueeze(0).clone()
        patch.requires_grad_(True)

        grad_cam = GradCAM(model, model.conv2)
        heatmap = grad_cam.generate_heatmap(patch)

        with torch.no_grad():
            prediction = model(patch).item()

        axes[0, i].imshow(patch[0, 0].detach().numpy(), cmap="gray")
        axes[0, i].set_title(f"Patch {idx}\n(conf: {prediction:.3f})")
        axes[0, i].axis("off")

        axes[1, i].imshow(heatmap, cmap="hot")
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--output", default="gradcam.png")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    data = torch.load(args.patches)
    visualize_examples(model, data["patches"], data["labels"], args.n_examples, args.output)


if __name__ == "__main__":
    main()
