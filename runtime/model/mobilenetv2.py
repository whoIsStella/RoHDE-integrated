"""MobileNetV2 variant used by the RoHDE EMG classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, expansion: int, stride: int):
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        num_classes: int,
        input_layer: int = 1,
        input_shape: tuple[int, int] = (8, 24),
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_layer, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        with torch.no_grad():
            dummy = torch.zeros(1, input_layer, *input_shape)
            feature_count = self._forward_features(dummy).reshape(1, -1).shape[1]

        self.linear = nn.Linear(feature_count, num_classes)

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for block_stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, block_stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return F.relu(self.bn2(self.conv2(out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_features(x)
        return self.linear(out.reshape(out.size(0), -1))
