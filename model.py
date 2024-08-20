import torch.nn as nn
from config import *
from config import *


def gen_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            gen_block(LATENT_DIMENSIONS, 256, 7, 1, 0),
            gen_block(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, IMAGE_CHANNELS, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


def disc_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            disc_block(IMAGE_CHANNELS, 128, 4, 2, 1),
            disc_block(128, 256, 4, 2, 1),
            nn.Conv2d(256, 1, 7, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1)
