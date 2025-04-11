import torch
import torch.nn as nn
import torch.optim as optim

from nets import LeNetWithResBlock, LeNet
from utils import get_dataloaders, train, test
from models.lenet_cifar10 import LeNetCIFAR10
from models.reslenet import ResLeNet

def main():
    # Device configuration: use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Hyper-parameters
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 128

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size)

    # Initialize model, loss function, and optimizer
    model = ResLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'lenet_resblock_cifar10.pth')
    print("Model saved as lenet_resblock_cifar10.pth")

if __name__ == '__main__':
    main()
