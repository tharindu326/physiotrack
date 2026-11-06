"""The MPIIFaceGaze and MPIIGaze networks of ptgaze.

Architectures copied from ptgaze (https://github.com/hysts/pytorch_mpiigaze_demo, MIT
License, (c) 2017 hysts): ``models/mpiifacegaze`` (``resnet_simple`` with a truncated
ResNet-18 backbone and a spatial-weights head, Zhang et al., "It's Written All Over Your
Face: Full-Face Appearance-Based Gaze Estimation", CVPR-W 2017) and ``models/mpiigaze``
(``resnet_preact``, a small pre-activation ResNet on 36x60 eye patches plus the
normalised head pose, Zhang et al., "Appearance-Based Gaze Estimation in the Wild",
CVPR 2015). The ETH-XGaze model is timm's plain ``resnet18``. Parameter names match the
released checkpoints.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class _TruncatedResNet18(torchvision.models.ResNet):
    """ResNet-18 up to ``layer3`` (the backbone of the MPIIFaceGaze model)."""

    def __init__(self):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 1])
        del self.layer4
        del self.avgpool
        del self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))


class MPIIFaceGazeNet(nn.Module):
    """Full-face gaze regressor with spatial weights; input ``(N, 3, 224, 224)`` BGR."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = _TruncatedResNet18()
        channels = 256  # layer3 of ResNet-18
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(channels * 14 ** 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x * F.relu(self.conv(x))
        return self.fc(x.view(x.size(0), -1))


class _PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "conv", nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv2(F.relu(self.bn2(self.conv1(x)), inplace=True))
        return y + self.shortcut(x)


class MPIIGazeNet(nn.Module):
    """Per-eye gaze regressor; inputs ``(N, 1, 36, 60)`` eye patches and ``(N, 2)`` head pose."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage1 = nn.Sequential()
        self.stage1.add_module("block1", _PreActBlock(16, 16, 1))
        self.stage2 = nn.Sequential()
        self.stage2.add_module("block1", _PreActBlock(16, 32, 2))
        self.stage3 = nn.Sequential()
        self.stage3.add_module("block1", _PreActBlock(32, 64, 2))
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 + 2, 2)

    def forward(self, x: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.conv(x))))
        x = F.adaptive_avg_pool2d(F.relu(self.bn(x), inplace=True), output_size=1)
        return self.fc(torch.cat([x.view(x.size(0), -1), head_pose], dim=1))
