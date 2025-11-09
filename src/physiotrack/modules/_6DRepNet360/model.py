import math
import torch
import torch.nn as nn
import torchvision
from . import utils


class SixDRepNet360(nn.Module):
    """
    6D Rotation Representation Network for 360-degree head pose estimation.
    
    This model predicts head pose (roll, pitch, yaw) from face images using
    a 6D rotation representation for better continuity across angle boundaries.
    
    Args:
        block: ResNet block type (e.g., torchvision.models.resnet.Bottleneck)
        layers: List of integers specifying the number of blocks in each layer
        fc_layers: Number of fully connected layers (default: 1)
    """
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion, 6)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out


def load_model(snapshot_path=None, device='cpu'):
    """
    Load a pretrained SixDRepNet360 model.
    
    Args:
        snapshot_path: Path to model weights. If None, downloads pretrained weights.
        device: Device to load the model on ('cpu', 'cuda', etc.)
    
    Returns:
        model: Loaded SixDRepNet360 model in eval mode
    """
    from torch.hub import load_state_dict_from_url
    
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    
    # Load weights
    if snapshot_path is None or snapshot_path == '':
        # Download pretrained weights
        saved_state_dict = load_state_dict_from_url(
            "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth",
            map_location=device
        )
    else:
        saved_state_dict = torch.load(snapshot_path, map_location=device)
    
    # Handle different state dict formats
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    
    model.to(device)
    model.eval()
    
    return model

