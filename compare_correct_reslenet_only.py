"""
Find and visualize samples where ResLeNet was correct but LeNet was wrong.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import get_dataloaders
from models.lenet_cifar10 import LeNetCIFAR10
from models.reslenet import ResLeNet

# CIFAR-10 class labels
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get test data with shuffle=False to preserve order
_, test_loader = get_dataloaders(batch_size=1)

# Load and eval model
def get_model_preds(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, dim=1)
            preds.append(predicted.item())
    return np.array(preds)

print("Evaluating LeNet and ResLeNet...")
lenet_preds = get_model_preds("best_lenet.pth", LeNetCIFAR10)
reslenet_preds = get_model_preds("best_lenet_resblock.pth", ResLeNet)

# Reload labels and images (preserving order)
_, test_loader = get_dataloaders(batch_size=1)
true_labels = []
images = []

for img, label in test_loader:
    true_labels.append(label.item())
    images.append(img.squeeze(0))  # shape: [3, H, W]

true_labels = np.array(true_labels)

# Find indices where ResLeNet is right and LeNet is wrong
target_indices = np.where((reslenet_preds == true_labels) & (lenet_preds != true_labels))[0]
print(f"Found {len(target_indices)} target samples.")

# Save a log file
with open("reslenet_correct_only.txt", "w") as f:
    for idx in target_indices:
        f.write(f"Index: {idx}, True: {classes[true_labels[idx]]}, LeNet: {classes[lenet_preds[idx]]}, ResLeNet: {classes[reslenet_preds[idx]]}\n")

# Plot N samples
N = min(9, len(target_indices))  # up to 9 samples for 3x3 grid
print(f"Plotting top {N} samples...")

plt.figure(figsize=(12, 8))
for i in range(N):
    idx = target_indices[i]
    img = images[idx].numpy().transpose(1, 2, 0)  # [C, H, W] → [H, W, C]

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"GT: {classes[true_labels[idx]]}\nLeNet: {classes[lenet_preds[idx]]}\nResLeNet: {classes[reslenet_preds[idx]]}")

plt.suptitle("Samples: ResLeNet Correct, LeNet Wrong", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("compare_predictions.png", dpi=300)
plt.show()
