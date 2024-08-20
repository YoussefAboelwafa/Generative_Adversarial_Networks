import torch
import numpy as np
import os

EPOCHS = 200
BATCH_SIZE = 64
LR = 0.0002
IMAGE_DIMENSIONS = 28
IMAGE_CHANNELS = 1
LATENT_DIMENSIONS = 100

EXP_NAME = os.getenv("SLURM_JOB_ID")

torch.manual_seed(5)
np.random.seed(5)
