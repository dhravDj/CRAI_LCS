import torch
from torch.utils.data import Dataset
import numpy as np

class ECGTaskDataset(Dataset):
    """
    PyTorch Dataset for Phase 4 CNN-LSTM Clinical Risk Network Training.
    Yields:
        x: (1, 1800) float32 tensor
        y: (1,) float32 binary label
        r: (1,) float32 continuous risk target R_i in [0.0, 1.0]
    """
    def __init__(self, segments: np.ndarray, labels: np.ndarray, risks: np.ndarray):
        assert len(segments) == len(labels) == len(risks), "Length mismatch in dataset inputs"
        self.segments = segments.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.risks = risks.astype(np.float32)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.segments[idx]).unsqueeze(0)  # (1, 1800) for Conv1D
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        r = torch.tensor([self.risks[idx]], dtype=torch.float32)
        return x, y, r
