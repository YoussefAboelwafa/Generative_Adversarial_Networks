import torch
import numpy as np

EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0002
IMAGE_DIMENSIONS = 28
IMAGE_CHANNELS = 1
LATENT_DIMENSIONS = 100

torch.manual_seed(5)
np.random.seed(5)
