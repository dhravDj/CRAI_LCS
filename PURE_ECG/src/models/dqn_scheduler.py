import os
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
from src.baselines.heuristics import BaseScheduler
from src.simulator.task import HealthcareTask

class DuelingDQN(nn.Module):
    """
    Dueling Double Deep Q-Network for Task Offloading (OBJ-5, FR-013, Eq. 30-32).
    Separates state-value V(s) and action-advantage A(s, a) streams.
    Parameter budget: 0.320M parameters (320,045 parameters).
    Latency budget: <= 5.7 ms on CPU.
    """
    def __init__(self, state_dim: int = 54, action_dim: int = 26):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared representation layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 358),
            nn.ReLU(),
            nn.Linear(358, 506),
            nn.ReLU()
        )

        # Value stream: V(s) -> scalar state value
        self.value_stream = nn.Sequential(
            nn.Linear(506, 114),
            nn.ReLU(),
            nn.Linear(114, 1)
        )

        # Advantage stream: A(s, a) -> relative advantage per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(506, 114),
            nn.ReLU(),
            nn.Linear(114, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a'))).
        state: Tensor of shape (batch_size, 54) or (54,)
        returns: Q-values of shape (batch_size, 26)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extractor(state)
        values = self.value_stream(features)         # (batch_size, 1)
        advantages = self.advantage_stream(features) # (batch_size, 26)

        # Wang et al. (2016) mean-centered advantage aggregation for identifiability
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0, device: Optional[torch.device] = None) -> int:
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.action_dim))

        if device is None:
            device = next(self.parameters()).device

        state_tensor = torch.from_numpy(state).float().to(device)
        q_values = self.forward(state_tensor)
        return int(torch.argmax(q_values, dim=-1).item())

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Aliases for architectural flexibility and project compatibility
DQNNetwork = DuelingDQN


class DQNScheduler(BaseScheduler):
    """
    Standard BaseScheduler adapter for trained Double DQN Offloading Agent.
    Plug-and-play compatible with the continuum simulator and comparative benchmark suite.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/dqn_scheduler.pt",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = DuelingDQN(state_dim=54, action_dim=26).to(self.device)
        self.is_trained = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_trained = True
        else:
            self.model.eval()

    def schedule(self, observation: np.ndarray, task: Optional[HealthcareTask] = None) -> int:
        """Selects greedy optimal dispatch node a* = argmax_a Q(s, a)."""
        return self.model.act(observation, epsilon=0.0, device=self.device)

    def reset(self):
        pass
