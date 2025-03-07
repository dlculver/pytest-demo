import torch
from src.model import SmallCNN
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def save_model(model, path="mnist_cnn.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="mnist_cnn.pth"):
    model = SmallCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def preprocess_image(image):
    """Convert drawn image to MNIST-compatible format."""
    img = Image.fromarray(image).convert('L')  # Grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor_img = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor_img

def get_processed_image_for_display(image):
    """Return the 28x28 image before normalization for debugging."""
    img = Image.fromarray(image).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(img).squeeze().numpy()  # 28x28 numpy array