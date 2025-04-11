import torch.nn as nn
import torch.nn.functional as F


"""
    This is the implementation of ResBlock which is used in ResNet-18/34
    
    Structure:
        Conv2d -> BN -> ReLU -> Conv2d -> BN + skip connection -> ReLU

                                            Apr 10, 25. by Chiashu @ NEU
"""

class BasicResBlock(nn.Module):
    def __init__(self, channels):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
