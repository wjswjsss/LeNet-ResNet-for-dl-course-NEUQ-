"""
This script computes and visualizes the confusion matrices
for LeNet and ResLeNet on the CIFAR-10 test set.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import get_dataloaders
from models.lenet_cifar10 import LeNetCIFAR10
from models.reslenet import ResLeNet          # You need this model implemented

# Load test data
_, test_loader = get_dataloaders(batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# Get predictions
print("Evaluating LeNet...")
lenet_preds, labels = get_predictions("best_lenet.pth", LeNetCIFAR10)

print("Evaluating ResLeNet...")
reslenet_preds, _ = get_predictions("best_lenet_resblock.pth", ResLeNet)

# Class labels
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Plot function
def plot_cm(preds, labels, title, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Save both CMs
print("Saving confusion matrices...")
plot_cm(lenet_preds, labels, 'Confusion Matrix - LeNet', 'cm_lenet.png')
plot_cm(reslenet_preds, labels, 'Confusion Matrix - ResLeNet', 'cm_reslenet.png')
print("Done!")
