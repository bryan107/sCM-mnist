import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore

def train():
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet expects 224x224
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 3. Modify ResNet-18 for MNIST
    model = models.resnet18(weights=None)   # Do not load ImageNet weights
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST has 10 classes
    model = model.to(device)

    # 4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        running_loss = 0.0
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 6. Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # 7. Save the model
    torch.save(model.state_dict(), "models/mnist_resnet18.pth")
    print("Model saved as mnist_resnet18.pth")

if __name__ == "__main__":
    train()