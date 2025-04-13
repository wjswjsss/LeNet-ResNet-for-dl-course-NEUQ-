import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

# from nets import LeNetWithResBlock, LeNet
from utils import get_dataloaders, train, test, train_and_evaluate
from models.lenet_cifar10 import LeNetCIFAR10
from models.reslenet import ResLeNet

"""
    The main.py contains the "main" function which used for training and plotting of two independent
    models, they are LeNet - 5, and the LeNet + ResBlocks.

                                                                        Apr 11, 25. by Chiashu @ NEU
"""

def main():

    """
    Cuda
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    """    
    Hyper-parameters
    """    
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    """
    Load data
    """
    train_loader, test_loader = get_dataloaders(batch_size)

    """
    Model 1 - LeNet-5          PLUS saving the metrics
    """
    model1 = LeNetCIFAR10().to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    lenet_metrics = train_and_evaluate(model1, device, train_loader, test_loader, optimizer1, criterion, num_epochs, "lenet")
    
    train_losses, test_losses, test_accuracies, best_epoch_le = lenet_metrics

    with open("lenet_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Test Loss", "Test Accuracy"])
        for i in range(num_epochs):
            writer.writerow([i+1, train_losses[i], test_losses[i], test_accuracies[i]])

    """
    Model 2 - LeNet + ResBlock PLUS saving the metrics
    """
    model2 = ResLeNet().to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    resblock_metrics = train_and_evaluate(model2, device, train_loader, test_loader, optimizer2, criterion, num_epochs, "lenet_resblock")

    train_losses, test_losses, test_accuracies, best_epoch_res = resblock_metrics

    with open("resblock_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Test Loss", "Test Accuracy"])
        for i in range(num_epochs):
            writer.writerow([i+1, train_losses[i], test_losses[i], test_accuracies[i]])

    """
    Plotting 
    """
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Primary y-axis for losses
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, lenet_metrics[0], 'b-o', label="LeNet Train Loss")
    ax1.plot(epochs, lenet_metrics[1], 'orange', marker='o', label="LeNet Test Loss")
    ax1.plot(epochs, resblock_metrics[0], 'cyan', marker='^', label="ResBlock Train Loss")
    ax1.plot(epochs, resblock_metrics[1], 'red', marker='^', label="ResBlock Test Loss")
    ax1.grid(True)

    # Mark best epoch for LeNet and ResBlock
    ax1.axvline(x=best_epoch_le, color='blue', linestyle='--', linewidth=1.5, label=f'LeNet Best Epoch ({best_epoch_le})')
    ax1.axvline(x=best_epoch_res, color='red', linestyle='--', linewidth=1.5, label=f'ResBlock Best Epoch ({best_epoch_res})')
    
    # Secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", color='green')
    ax2.plot(epochs, lenet_metrics[2], 'g--x', label="LeNet Test Acc")
    ax2.plot(epochs, resblock_metrics[2], 'lime', linestyle='--', marker='x', label="ResBlock Test Acc")
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Performance Comparison: LeNet vs LeNet+ResBlock")
    plt.tight_layout()
    plt.savefig("comparison_lenet_resblock.png")
    plt.show()

if __name__ == '__main__':
    main()