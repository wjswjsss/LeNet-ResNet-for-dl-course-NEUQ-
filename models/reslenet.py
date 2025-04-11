import torch.nn as nn
import torch.nn.functional as F
from models.resblock import BasicResBlock

"""
    This file contains a def of ResLeNet class which is a neural network
    COMBINES the LeNet with the ResBlock architecture. 
    
    Structure:
           Conv2d -> BN -> ReLU 
        -> ResBlock -> Pool
        -> Conv2d -> BN -> ReLU
        -> ResBlock -> Pool
        -> classifier head (flatten then 3 fc, with ReLU)

                                            Apr 11, 25. by Chiashu @ NEU
"""

class ResLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = nn.Sequential(
            BasicResBlock(16),
            BasicResBlock(16)
        )
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.layer2 = nn.Sequential(
            BasicResBlock(32),
            BasicResBlock(32)
        )
        self.pool2 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 32x32x3 → 32x32x16
        x = self.layer1(x)
        x = self.pool1(x)                    # → 16x16x16

        x = F.relu(self.bn2(self.conv2(x)))  # → 16x16x32
        x = self.layer2(x)
        x = self.pool2(x)                    # → 8x8x32

        x = x.view(x.size(0), -1)            # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
