"""
Pure-EMG-CRAI-LCS: Spatial-Temporal sEMG-CNN-BiLSTM Architecture
Predicts Neuromuscular Clinical Risk Index R_i in [0.0, 1.0] from 4-channel surface EMG telemetry.
Features:
- Cross-channel spatial mixing (Conv1D 1x1)
- Multi-scale temporal feature extraction
- 2-Layer Bidirectional LSTM for neuromuscular activation dynamics
- Fully vectorized inference buffer (< 3.0 ms latency budget)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ClinicalRiskCNNLSTM(nn.Module):
    """
    Multichannel sEMG Spatial-Temporal Neural Network for Neuromuscular Triage.
    Contract: Input (B, 4, 2000) -> Output (B, 1) in [0.0, 1.0].
    """
    def __init__(self, in_channels: int = 4, seq_len: int = 2000):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        # 1. Spatial channel mixer across 4 muscle sensor sites
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # 2. Multi-stage temporal convolutional feature extractor
        self.temporal = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. 2-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 4. Clinical Risk Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 4, 2000)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.shape[1] == 1 and self.in_channels == 4:
            x = x.repeat(1, 4, 1)

        x_sp = self.spatial(x)
        x_conv = self.temporal(x_sp)  # (B, 128, T')
        x_lstm_in = x_conv.permute(0, 2, 1)  # (B, T', 128)

        lstm_out, _ = self.lstm(x_lstm_in)
        last_hidden = lstm_out[:, -1, :]  # (B, 128)
        return last_hidden

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return self.sigmoid(logits)

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
