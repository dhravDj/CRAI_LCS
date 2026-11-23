from typing import Tuple, Optional, Union
import numpy as np
import torch

class PrioritizedReplayBuffer:
    """
    Vectorized Prioritized Experience Replay (PER) Buffer (T6-01, FR-013).
    Circular ring buffer holding up to 50,000 transitions (s, a, r, s', done)
    with rank/proportional priority sampling and importance-sampling bias correction.
    """
    def __init__(
        self,
        capacity: int = 50000,
        state_dim: int = 54,
        alpha: float = 0.6,
        seed: Optional[int] = None
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

        # Preallocated contiguous flat numpy arrays for zero garbage collection overhead
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Appends a new transition with maximal priority to guarantee at least one training visit."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int = 64,
        beta: float = 0.4,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Samples a mini-batch with probability proportional to priority^alpha.
        Computes importance-sampling weights w_i = (N * P(i))^(-beta) / max(w).
        """
        assert self.size >= batch_size, f"Buffer contains {self.size} samples, fewer than batch_size {batch_size}"

        if self.alpha == 0.0:
            # Uniform sampling
            indices = self.rng.randint(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            # Proportional priority sampling
            prios = self.priorities[:self.size]
            probs = prios ** self.alpha
            sum_probs = np.sum(probs)
            if sum_probs == 0.0 or np.isnan(sum_probs):
                probs = np.ones(self.size, dtype=np.float32) / self.size
            else:
                probs = probs / sum_probs

            indices = self.rng.choice(self.size, size=batch_size, p=probs, replace=False if self.size >= batch_size else True)

            # Importance-sampling weights
            total = self.size
            weights = (total * probs[indices]) ** (-beta)
            max_weight = weights.max()
            if max_weight > 0:
                weights = weights / max_weight

        b_states = torch.from_numpy(self.states[indices]).float()
        b_actions = torch.from_numpy(self.actions[indices]).long()
        b_rewards = torch.from_numpy(self.rewards[indices]).float()
        b_next_states = torch.from_numpy(self.next_states[indices]).float()
        b_dones = torch.from_numpy(self.dones[indices]).float()
        b_weights = torch.from_numpy(weights).float()

        if device is not None:
            b_states = b_states.to(device)
            b_actions = b_actions.to(device)
            b_rewards = b_rewards.to(device)
            b_next_states = b_next_states.to(device)
            b_dones = b_dones.to(device)
            b_weights = b_weights.to(device)

        return b_states, b_actions, b_rewards, b_next_states, b_dones, b_weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: Union[np.ndarray, torch.Tensor], eps: float = 1e-5):
        """Updates transition priorities based on observed Bellman TD errors."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        new_priorities = np.abs(td_errors).flatten() + eps
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def __len__(self) -> int:
        return self.size
