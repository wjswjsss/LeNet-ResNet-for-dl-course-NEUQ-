"""
This script computes and visualizes the confusion matrix 
for the CIFAR-10 classification results using the trained LeNet model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import get_dataloaders
from models.lenet_cifar10 import LeNetCIFAR10

# Load test data
_, test_loader = get_dataloaders(batch_size=64)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LeNetCIFAR10()
model.load_state_dict(torch.load("./best_lenet.pth", map_location=device))
model.to(device)
model.eval()

# Gather predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Define class names
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on CIFAR-10 (LeNet)')
plt.tight_layout()
plt.show()
