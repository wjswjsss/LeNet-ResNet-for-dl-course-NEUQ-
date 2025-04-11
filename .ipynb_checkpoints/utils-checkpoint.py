import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

def get_dataloaders(batch_size=128):
    """
    Returns training and testing dataloaders for CIFAR-10.
    Applies data augmentation (random horizontal flips, random crops) on training images.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Trains the model for one epoch.
    Prints the running loss and training accuracy every 100 batches.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{running_loss / (batch_idx + 1):.4f}",
            'Acc': f"{100. * correct / total:.2f}%"})
    
    # Final epoch summary
    print(f"Epoch {epoch} Summary | Loss: {running_loss / len(train_loader):.4f} | Accuracy: {100. * correct / total:.2f}%")


def test(model, device, test_loader, criterion):
    """
    Evaluates the model on the test dataset.
    Prints and returns the average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{test_loss / (total / inputs.size(0)):.4f}",
                'Acc': f"{100. * correct / total:.2f}%"})

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f"Test Summary | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy
