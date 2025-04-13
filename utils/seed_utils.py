import random
import numpy as np
import torch

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader to ensure
    reproducibility across different workers.
    
    Args:
        worker_id (int): Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 