"""
Pure-EEG-CRAI-LCS: Spatial-Temporal 1D-CNN-BiLSTM Neurological Risk Model
Detects Acute Epileptic Seizures, Ictal Bursts, and Status Epilepticus from 16-channel Scalp EEG.
Input shape: (B, 16, 1280) [16 Channels @ 256 Hz x 5 Seconds]
Output shape: (B, 1) Continuous Neurological Risk Index R_i in [0.0, 1.0]
"""

import torch
import torch.nn as nn
from typing import Tuple


class ClinicalRiskCNNLSTM(nn.Module):
    """
    Spatial-Temporal 1D-CNN-BiLSTM for Neurological Telemetry Risk Assessment.
    Compatible with CHB-MIT & Temple University Hospital EEG datasets.
    """
    def __init__(self, in_channels: int = 16, num_samples: int = 1280):
        super().__init__()
        self.in_channels = in_channels
        self.num_samples = num_samples

        # 1. Spatial Channel Mixing (Cross-channel electrode correlation)
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 2. Temporal Multi-Scale Feature Extraction
        self.conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Output: (B, 64, 320)

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Output: (B, 128, 80)

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)           # Output: (B, 256, 40)
        )

        # 3. Temporal Sequence Modeling: 2-layer Bidirectional LSTM
        # Models 3 Hz spike-and-wave rhythm dynamics and rhythmic ictal discharge patterns
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 4. Neurological Risk Estimation Head
        # Output of bidirectional LSTM is hidden_size * 2 = 256
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing: (B, 16, 1280) -> (B, 32, 1280)
        s = self.spatial(x)
        # Temporal conv: (B, 32, 1280) -> (B, 256, 40)
        c = self.conv(s)
        # Permute for LSTM: (B, 40, 256)
        c = c.permute(0, 2, 1)
        out, (hn, _) = self.lstm(c)
        # Concatenate forward and backward final hidden states: (B, 256)
        last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        return self.fc(last_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns bounded continuous risk score R_i in [0.0, 1.0]."""
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)
