import torch
import torch.nn as nn

class DeadlineMissPredictor(nn.Module):
    """
    MLP Pre-Execution Deadline Violation Forecasting Network (FR-010, Eq. 12, Table 12).
    Predicts probability P_{miss,i} in [0.0, 1.0] given system state and task features.
    Parameter budget: ~0.18M parameters.
    Target latency: <= 3.6 ms.
    """
    def __init__(self, in_features: int = 54):
        super().__init__()
        self.in_features = in_features

        # 4-layer MLP with batch normalization and dropout
        self.net = nn.Sequential(
            nn.Linear(in_features, 360),
            nn.BatchNorm1d(360),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(360, 360),
            nn.BatchNorm1d(360),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(360, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unscaled logits for numerically stable BCEWithLogitsLoss."""
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probability P_{miss,i} in [0.0, 1.0]."""
        return torch.sigmoid(self.forward_logits(x))

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
