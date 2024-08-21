from comet_ml import Experiment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from config import *

experiment = Experiment(
    api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
    project_name="gans",
    workspace="youssefaboelwafa",
)
experiment.set_name(EXP_NAME)

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_DIMENSIONS),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="/scratch/dr/y.aboelwafa/GAN/GANs/dataset",
    train=True,
    transform=transform,
    download=True,
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR)


for epoch in range(EPOCHS):
    for i, (real_images, _) in enumerate(train_dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        optimizer_d.zero_grad()
        label_real = torch.ones(batch_size, 1).to(device)
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, label_real)
        loss_real.backward()

        noise = torch.randn(batch_size, LATENT_DIMENSIONS, 1, 1).to(device)
        fake_images = generator(noise)
        label_fake = torch.zeros(batch_size, 1).to(device)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, label_fake)
        loss_fake.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        output = discriminator(fake_images)
        loss_g = criterion(output, label_real)
        loss_g.backward()
        optimizer_g.step()

        loss_D = (loss_real.item() + loss_fake.item()) / 2
        loss_G = loss_g.item()

    if (i + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}]\n"
            f"Loss_D: {loss_D:.4f}, Loss_G: {loss_G:.4f}\n"
        )

    experiment.log_metric("loss_D", loss_D, step=epoch)
    experiment.log_metric("loss_G", loss_G, step=epoch)


torch.save(generator.state_dict(), "generator.pth")
