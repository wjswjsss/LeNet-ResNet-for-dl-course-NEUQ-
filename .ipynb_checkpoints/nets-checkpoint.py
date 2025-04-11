import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    A simple Residual Block:
    - Two 3×3 convolution layers with batch normalization and ReLU activations.
    - Identity shortcut connection.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class LeNetWithResBlock(nn.Module):
    """
    A LeNet-inspired model for CIFAR-10 with an integrated residual block.
    
    Architecture:
      - Conv1: 3 input channels, 6 filters, kernel=5 with padding to maintain size.
      - ReLU activation.
      - Residual Block on the 6-channel feature map.
      - MaxPool reducing size by half.
      - Conv2: converts 6 channels to 16 channels (kernel=5).
      - ReLU and MaxPool again.
      - Three fully connected layers.
    """
    def __init__(self, num_classes=10):
        super(LeNetWithResBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.resblock = ResBlock(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # valid convolution (no padding)
        self.pool2 = nn.MaxPool2d(2, 2)
        # After conv2, the feature map size becomes 12×12 (assuming 32×32 input after pool),
        # and after pooling, it reduces to 6×6. With 16 channels, the flattened size is 16*6*6.
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # First convolution + residual block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.resblock(x)
        x = self.pool(x)
        # Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LeNet(nn.Module):
    """
    A LeNet-inspired model for CIFAR-10 WITHOUT THE integrated residual block.
    
    Architecture:
      - Conv1: 3 input channels, 6 filters, kernel=5 with padding to maintain size.
      - ReLU activation.
      - Residual Block on the 6-channel feature map.
      - MaxPool reducing size by half.
      - Conv2: converts 6 channels to 16 channels (kernel=5).
      - ReLU and MaxPool again.
      - Three fully connected layers.
    """

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # self.resblock = ResBlock(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # valid convolution (no padding)
        self.pool2 = nn.MaxPool2d(2, 2)
        # After conv2, the feature map size becomes 12×12 (assuming 32×32 input after pool),
        # and after pooling, it reduces to 6×6. With 16 channels, the flattened size is 16*6*6.
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # First convolution + residual block
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.resblock(x)
        x = self.pool(x)
        # Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

