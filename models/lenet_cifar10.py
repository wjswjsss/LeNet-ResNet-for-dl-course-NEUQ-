import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    This is the implementation of LeNet - 5 WITHOUT resblocks.

                                  Apr 10, 25. by Chiashu @ NEU
"""

class LeNetCIFAR10(nn.Module):
    def __init__(self):
        super(LeNetCIFAR10, self).__init__()
        # Conv Layer 1: input channels=3 (RGB), output channels=6
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 32x32 -> 28x28

        # Subsampling (avg pool) layer 1
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # Conv Layer 2: input=6, output=16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14 -> 10x10

        # Subsampling layer 2: 10x10 -> 5x5
        # Already defined above, reusing self.pool

        # Fully connected conv layer (C5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # Fully connected layer (F6)
        self.fc2 = nn.Linear(120, 84)

        # Output layer
        self.fc3 = nn.Linear(84, 10)  # 10 CIFAR-10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # 32x32x3 → 28x28x6 → 14x14x6
        x = self.pool(F.relu(self.conv2(x)))      # 14x14x6 → 10x10x16 → 5x5x16
        x = x.view(-1, 16 * 5 * 5)                 # Flatten for FC
        x = F.relu(self.fc1(x))                    # 400 → 120
        x = F.relu(self.fc2(x))                    # 120 → 84
        x = self.fc3(x)                            # 84 → 10
        return x
