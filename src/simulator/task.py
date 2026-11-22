from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class HealthcareTask:
    """
    Healthcare Telemetry Task representation in the CRAI-LCS Continuum.
    4-Tuple: T_i = {delta_i, psi_i, D_i, R_i}
    """
    task_id: int
    delta_i: float            # Payload size in Bytes (~3600 bytes for 5s @ 360 Hz)
    psi_i: float              # Computational demand in CPU cycles (5e6 to 2e7)
    deadline_ms: float        # Task completion deadline in milliseconds
    clinical_risk: float      # Clinical urgency index R_i in [0.0, 1.0]
    arrival_time: float       # Simulation arrival timestamp in seconds
    predicted_miss_prob: float = 0.0  # Pre-execution deadline violation probability P_{miss,i} in [0.0, 1.0]
    assigned_node: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    actual_latency_ms: Optional[float] = None
    energy_consumed_j: Optional[float] = None
    is_violated: Optional[bool] = None
    is_preempted: bool = False

    @property
    def is_urgent(self) -> bool:
        """Returns True if the task represents a critical clinical condition (R_i >= 0.5)."""
        return self.clinical_risk >= 0.50

    @property
    def deadline_seconds(self) -> float:
        """Returns task deadline in seconds."""
        return self.deadline_ms / 1000.0


def create_synthetic_task(
    task_id: int,
    arrival_time: float,
    clinical_risk: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None
) -> HealthcareTask:
    """
    Generates a single synthetic task conforming strictly to FR-006 and Table 3.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Clinical risk R_i in [0.0, 1.0] if not supplied
    if clinical_risk is None:
        # In hospital monitoring, majority are normal (~75%) and minority are urgent (~25%)
        is_high_risk = rng.rand() < 0.25
        if is_high_risk:
            clinical_risk = float(rng.uniform(0.50, 0.99))
        else:
            clinical_risk = float(rng.uniform(0.01, 0.49))

    # Computation demand: psi_i in [5e6, 2e7] cycles
    psi_i = float(rng.uniform(5.0e6, 2.0e7))

    # Payload delta_i: ~3600 Bytes
    delta_i = float(rng.normal(3600.0, 150.0))
    delta_i = max(1800.0, min(delta_i, 7200.0))

    # Deadline D_i according to clinical risk
    if clinical_risk >= 0.50:
        # Emergency: 100 - 150 ms
        deadline_ms = float(rng.uniform(100.0, 150.0))
    else:
        # Routine: 400 - 800 ms
        deadline_ms = float(rng.uniform(400.0, 800.0))

    return HealthcareTask(
        task_id=task_id,
        delta_i=delta_i,
        psi_i=psi_i,
        deadline_ms=deadline_ms,
        clinical_risk=clinical_risk,
        arrival_time=arrival_time,
        predicted_miss_prob=0.0
    )
