import torch
from src.model import SmallCNN
from src.data import get_data_loaders
from src.train.train import train_model
from src.deploy.utils import save_model
import subprocess

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN()
    train_loader, val_loader, test_loader = get_data_loaders()
    trained_model = train_model(model, train_loader, val_loader, test_loader, device=device)
    save_model(trained_model, "mnist_cnn.pth")
    print("Training complete and model saved!")

def run_deployment():
    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "src/deploy/app.py", "--server.fileWatcherType", "none"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|deploy]")
    elif sys.argv[1] == "train":
        run_training()
    elif sys.argv[1] == "deploy":
        run_deployment()
    else:
        print("Invalid argument. Use 'train' or 'deploy'.")