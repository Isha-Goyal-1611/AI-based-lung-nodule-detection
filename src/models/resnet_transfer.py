"""
src/models/resnet_transfer.py

ResNet-18 (ImageNet-pretrained) adapted for single-channel CT patches.
First conv layer replaced to accept 1 channel instead of 3; final fc
layer replaced for binary classification.

NOTE: model creation is wrapped in build_resnet_model() rather than run
at import time — the original version instantiated the model directly
in the module body, which broke whenever this file was imported (e.g.
from ensemble.py or train_resnet.py) rather than run standalone.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 1)
    return model


if __name__ == "__main__":
    model = build_resnet_model()
    fake_input = torch.randn(1, 1, 224, 224)
    output = model(fake_input)
    sigmoid_output = torch.sigmoid(output)
    print("Output shape:", sigmoid_output.shape)
    print("Output value:", sigmoid_output.item())