import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Sets global deterministic random seeds for Python, NumPy, and PyTorch (CPU & CUDA).
    Ensures bitwise reproducible simulation runs and model training.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
