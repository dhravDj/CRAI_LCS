"""
U-CRAI-LCS: Universal Multimodal Telemetry Data Contract
Defines modality-agnostic telemetry task representations and clinical metadata.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


class ModalityType(str, Enum):
    """Clinical Telemetry Modality Categories."""
    CARDIAC_ECG = "ECG"         # 1-lead / 2-lead Electrocardiogram (360 Hz)
    NEURO_EEG = "EEG"           # 16-channel Electroencephalogram (256 Hz)
    RESPIRATORY_PPG = "PPG"     # Photoplethysmogram / Pulse Oximetry (100 Hz)


# Clinical Time-to-Harm (T_harm) references in seconds based on emergency medicine guidelines
TIME_TO_HARM_SECONDS: Dict[ModalityType, float] = {
    ModalityType.CARDIAC_ECG: 30.0,       # Ventricular fibrillation / acute arrest (~30s)
    ModalityType.NEURO_EEG: 60.0,         # Status epilepticus / severe seizure (~60s)
    ModalityType.RESPIRATORY_PPG: 180.0,  # Acute desaturation / severe hypoxia (~180s)
}


@dataclass
class UniversalTelemetryTask:
    """
    Standardized Universal Telemetry Workload Contract.
    Encapsulates raw signal, physiological metadata, calibrated urgency, and continuum scheduling states.
    """
    task_id: int
    modality: ModalityType
    patient_id: int
    raw_signal: np.ndarray             # 1D for ECG/PPG, (Channels, Samples) for EEG
    sampling_rate_hz: int
    arrival_timestamp_s: float
    payload_bytes: int
    computational_demand_cycles: float # Estimated CPU cycles for analysis
    
    # Dynamic clinical attributes (computed during pipeline execution)
    clinical_risk: float = 0.0         # Raw disease-specific probability R_i in [0.0, 1.0]
    clinical_urgency: float = 0.0      # Calibrated time-to-harm index U_i in [0.0, 1.0]
    predicted_miss_prob: float = 0.0   # Pre-execution deadline failure risk P_miss in [0.0, 1.0]
    deadline_ms: float = 500.0         # Dynamic SLA deadline
    
    # Continuum execution states
    assigned_node: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    actual_latency_ms: Optional[float] = None
    energy_consumed_j: Optional[float] = None
    is_violated: Optional[bool] = None
    is_preempted: bool = False

    @property
    def is_urgent(self) -> bool:
        """Determines if task qualifies for emergency local fail-safe preemption."""
        return self.clinical_urgency >= 0.50

    @property
    def deadline_seconds(self) -> float:
        return self.deadline_ms / 1000.0
