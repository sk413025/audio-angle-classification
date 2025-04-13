import torch
import numpy as np
import random

def worker_init_fn(worker_id):
    """Function to make DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Check for MPS availability and set seed
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
