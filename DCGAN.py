import torch
import torch.nn as nn
import torch.nn.functional as F

def gen_block(in_dim, out_dim, kernel_size, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )
    
def disc_block(in_dim, out_dim, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2)
    )

class Generator(nn.Module):
    def __init__(self, in_dim, kernel_size=2, stride=2):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.gen = nn.Sequential(
            gen_block(in_dim, 1024, kernel_size, stride),
            gen_block(1024, 512, kernel_size, stride),
            gen_block(512, 256, kernel_size, stride),
            nn.ConvTranspose2d(256, 3, kernel_size, stride=stride),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(len(x), self.in_dim, 1, 1)
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            disc_block(3, 256, kernel_size, stride),
            disc_block(256, 512, kernel_size, stride),
            disc_block(512, 1024, kernel_size, stride),
            nn.Conv2d(1024, 1, kernel_size, stride=stride),
        )

    def forward(self, x):
        x = self.disc(x)
        return x.view(len(x), -1)
    

batch_size = 16
noise_dim = 100

noise = torch.randn(batch_size, noise_dim)

gen = Generator(in_dim=noise_dim)
generated_images = gen(noise)
print("Generated Images Shape:", generated_images.shape)

disc = Discriminator()
disc_output = disc(generated_images)
print("Discriminator Output Shape:", disc_output.shape)
print("Discriminator Output:", F.sigmoid(disc_output))