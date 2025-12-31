import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        eps = 1e-7
        cos_t = torch.matmul(x_norm, w_norm.t())
        cos_t = torch.clamp(cos_t, -1.0 + eps, 1.0 - eps)

        if labels is None:
            return cos_t * self.s

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)

        sin2 = (1.0 - cos_t.pow(2)).clamp_min(eps)
        sin_t = torch.sqrt(sin2)
        phi = cos_t * cos_m - sin_t * sin_m

        th = math.cos(math.pi - self.m)
        mm = math.sin(math.pi - self.m) * self.m
        phi = torch.where(cos_t > th, phi, cos_t - mm)

        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cos_t
        output = output * self.s
        return output


def build_model(num_classes: int, head_type: str, arc_s: float = 30.0, arc_m: float = 0.5) -> Tuple[nn.Module, nn.Module]:
    backbone = SimpleCNN(in_channels=1, feature_dim=128)
    if head_type == "softmax":
        head = LinearHead(backbone.feature_dim, num_classes)
    elif head_type == "arcface":
        head = ArcFaceHead(backbone.feature_dim, num_classes, s=arc_s, m=arc_m)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
    return backbone, head
