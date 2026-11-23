import torch
import torch.nn as nn
from typing import Tuple

class ClinicalRiskCNNLSTM(nn.Module):
    """
    1D-CNN-LSTM Clinical Urgency and Risk Model (FR-005, Table 12, Equations 17-26).
    Parameter budget: ~0.74M parameters.
    Execution latency: <= 14.8 ms on CPU / Edge.
    Input shape: (B, 1, 1800) normalized ECG segment.
    Output shape: (B, 1) continuous clinical risk score R_i in [0.0, 1.0].
    """
    def __init__(self):
        super().__init__()

        # Feature Extraction: 3-layer 1D CNN with Batch Normalization
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Output: (B, 64, 900)

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Output: (B, 128, 450)

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)           # Output: (B, 256, 225)
        )

        # Temporal Sequence Modeling: 2-layer Bidirectional/Deep LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=170,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Classification & Risk Estimation Head (Dual Sigmoid & Logits)
        self.fc = nn.Sequential(
            nn.Linear(170, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw un-activated logits (used for BCEWithLogitsLoss)."""
        c = self.conv(x)
        c = c.permute(0, 2, 1)  # (B, 225, 256) for LSTM
        _, (hn, _) = self.lstm(c)
        return self.fc(hn[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns bounded continuous risk score R_i in [0.0, 1.0]."""
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
