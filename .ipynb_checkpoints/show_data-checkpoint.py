import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
"""
    I use this py file to generate the "cifar10_sample.png"
"""
# Transform to normalize and convert to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Download the dataset
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Get class names
classes = cifar10.classes

# Create a dataloader
loader = torch.utils.data.DataLoader(cifar10, batch_size=16, shuffle=True)

# Get one batch
dataiter = iter(loader)
images, labels = next(dataiter)

# Helper to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize if needed
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title("Some CIFAR-10 Images")
    plt.tight_layout()
    plt.savefig("cifar10_sample.png", dpi=300)  # save to file
    plt.show()

# Show images
imshow(torchvision.utils.make_grid(images))
print("Labels: ", ' | '.join([classes[label] for label in labels]))
