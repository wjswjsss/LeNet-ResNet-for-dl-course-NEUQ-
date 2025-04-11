import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

"""
    This file contains a bunch of util functions. 

                                            Apr 11, 25. by Chiashu @ NEU
"""

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

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

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

    return running_loss / (batch_idx + 1)


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


def train_and_evaluate(model, device, train_loader, test_loader, optimizer, criterion, num_epochs, model_name):
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, acc = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            print(f"[{model_name}] Best model saved (Epoch {epoch}, Accuracy: {acc:.2f}%)")

    print(f"[{model_name}] Training complete. Best accuracy: {best_acc:.2f}%")

    return train_losses, test_losses, test_accuracies

