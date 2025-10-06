# src/dataset.py
import torch
from torchvision import datasets, transforms

class CustomizedDataset:
    def __init__(self, root="./dataset"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)

    def get_loaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
