from typing import Optional
import numpy as np
from src.simulator.task import HealthcareTask

class BaseScheduler:
    """Abstract base scheduler interface for all baseline algorithms (FR-012)."""
    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        raise NotImplementedError

    def reset(self):
        pass


class RandomScheduler(BaseScheduler):
    """
    Uniform random baseline scheduler.
    Uniformly samples an action from {0..25} (25 Edge nodes + 1 Cloud).
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        return int(self.rng.randint(0, 26))

    def reset(self):
        pass


class RoundRobinScheduler(BaseScheduler):
    """
    Cyclic round-robin baseline scheduler.
    Cycles sequentially across the 25 Edge Gateways (0..24).
    """
    def __init__(self, num_edge: int = 25):
        self.num_edge = num_edge
        self.current_idx = -1

    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        self.current_idx = (self.current_idx + 1) % self.num_edge
        return self.current_idx

    def reset(self):
        self.current_idx = -1


class FIFOScheduler(BaseScheduler):
    """
    Greedy Shortest Queue First (SQF) / FIFO baseline scheduler.
    Dispatches task to the edge node currently possessing the minimal queue workload.
    """
    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        edge_queues = observation[0:25]
        return int(np.argmin(edge_queues))

    def reset(self):
        pass


class EDFScheduler(BaseScheduler):
    """
    Earliest Deadline First (EDF) heuristic scheduler.
    High-urgency tasks (deadline < 200 ms) are routed to edge nodes with minimal queue delay.
    Routine tasks (deadline >= 200 ms) are offloaded to Central Cloud (action 25).
    """
    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        if task.deadline_ms < 200.0:
            return int(np.argmin(observation[0:25]))
        return 25

    def reset(self):
        pass


class RiskPriorityScheduler(BaseScheduler):
    """
    Clinical Risk-Aware Heuristic scheduler.
    Emergency cardiac events (R_i >= 0.50) are dispatched to the shortest-queue local edge node.
    Routine tasks (R_i < 0.50) are offloaded to Cloud (action 25) to preserve edge capacity.
    """
    def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
        if task.is_urgent:
            return int(np.argmin(observation[0:25]))
        return 25

    def reset(self):
        pass
