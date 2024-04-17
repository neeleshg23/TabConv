import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        # If output size is not the same as input size, adjust with 1x1 conv
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x))

DIM = 64

class ResNet(nn.Module):
    
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super().__init__()
        self.in_channels = DIM 

        self.conv1 = nn.Conv2d(num_channels, DIM, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(DIM)

        # Dynamically add blocks to the layer
        self.layer1 = self._make_layer(block, DIM, num_blocks[0])
        self.layer2 = self._make_layer(block, DIM*2, num_blocks[1])
        self.layer3 = self._make_layer(block, DIM*4, num_blocks[2])
        self.layer4 = self._make_layer(block, DIM*8, num_blocks[3])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(DIM*8 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks):
        layers = []
        strides = [2] + (num_blocks-1)*[1]
        for i in range(num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=strides[i]))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        intermediate = []
        out = self.conv1(x)
        intermediate.append(np.asarray(out.clone().detach().numpy()))
        out = self.bn1(out)
        out = F.relu(out)
        
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_group in layers:
            for layer in layer_group:
                out = layer(out)
                intermediate.append(np.asarray(out.clone().detach().numpy()))
        
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        intermediate.append(np.asarray(out.clone().detach().numpy()))
        return out, intermediate

def resnet14(num_classes, num_channels):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, num_channels)

def resnet18(num_classes, num_channels):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)

def resnet34(num_classes, num_channels):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_channels)