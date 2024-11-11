import numpy as np
import random
import torch

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(RANDOM_SEED)
