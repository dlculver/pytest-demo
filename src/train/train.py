"""Training of CNN for digit recognition."""

import torch.nn as nn
import torch.optim as optim
from src.model import SmallCNN
from src.train.utils import compute_accuracy

def train_model(model: SmallCNN, train_loader, val_loader, test_loader, num_epochs=10, lr=0.001, device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        val_accuracy = compute_accuracy(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    test_accuracy = compute_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return model