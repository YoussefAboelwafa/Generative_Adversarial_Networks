import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from config import *
from model import Generator, Discriminator


transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_DIMENSIONS),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="./dataset", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./dataset", train=False, transform=transform, download=True
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR)
