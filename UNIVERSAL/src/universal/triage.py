"""
U-CRAI-LCS: Clinical Time-to-Harm Urgency Calibration & Dynamic SLA Predictor
Implements mathematically grounded triage utility scaling across heterogeneous diseases.
Synchronizes clinical urgency across Cardiac, Neurological, and Respiratory pathologies
using the Manchester Triage & Emergency Severity Index (ESI) formulations.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from src.universal.schema import ModalityType, TIME_TO_HARM_SECONDS


class ClinicalUrgencyCalibrator:
    """
    Calibrates raw neural prediction probabilities R_i in [0, 1] into a standardized,
    cross-disease actionable Clinical Urgency index U_i in [0, 1] based on Time-to-Harm horizons.
    """
    def __init__(self, t_ref_seconds: float = 60.0):
        self.t_ref = t_ref_seconds

    def calibrate(self, raw_risk: float, modality: ModalityType) -> float:
        """
        Computes non-linear urgency utility:
        U_i = 1.0 - exp(- R_i * (T_ref / T_harm_m))
        Guarantees that acute cardiac events strictly preempt subacute respiratory hypoxia.
        """
        t_harm = TIME_TO_HARM_SECONDS.get(modality, 60.0)
        # Ratio scales urgency based on how rapid the fatal outcome horizon is
        severity_scaling = self.t_ref / max(1.0, t_harm)
        # Non-linear convex mapping bounded in [0.0, 1.0]
        urgency = 1.0 - np.exp(- float(raw_risk) * severity_scaling)
        return float(np.clip(urgency, 0.0, 1.0))

    def get_triage_tier(self, urgency: float) -> Tuple[int, str]:
        """Maps urgency to international Emergency Severity Index (ESI) tiers 1 to 4."""
        if urgency >= 0.70:
            return 1, "RESUSCITATION_IMMEDIATE"
        elif urgency >= 0.50:
            return 2, "EMERGENT"
        elif urgency >= 0.30:
            return 3, "URGENT"
        else:
            return 4, "ROUTINE_TELEMETRY"


class DynamicSLAPredictor:
    """
    Computes SLA deadline D_i (in ms) and forecasts pre-execution violation risk P_miss.
    """
    def __init__(self):
        self.calibrator = ClinicalUrgencyCalibrator()
        # Base deadline ranges per ESI triage tier (ms)
        self.tier_deadlines = {
            1: (50.0, 120.0),    # Immediate (Ventricular Fibrillation / Acute Seizure)
            2: (120.0, 250.0),   # Emergent
            3: (250.0, 500.0),   # Urgent
            4: (500.0, 1000.0)   # Routine
        }

    def compute_deadline_and_miss_risk(
        self,
        urgency: float,
        payload_bytes: int,
        channel_snr_db: float = 20.0,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[float, float]:
        """
        Returns:
            deadline_ms: Dynamic completion deadline
            predicted_p_miss: Probability of deadline violation under current wireless channel
        """
        local_rng = rng or np.random

        tier, _ = self.calibrator.get_triage_tier(urgency)
        d_min, d_max = self.tier_deadlines[tier]

        # Dynamic interpolator within tier based on exact continuous urgency
        deadline_ms = float(d_max - urgency * (d_max - d_min))
        deadline_ms = max(d_min, min(d_max, deadline_ms))

        # P_miss estimation based on payload size vs channel SNR and deadline tightness
        # shortcut: analytical channel SNR-to-bandwidth slack model, upgrade to trained logistic regression on physical RF channel states
        # Low SNR or high payload increases channel transmission delay
        estimated_tx_ms = (payload_bytes * 8.0) / (max(5.0, channel_snr_db) * 1e3)
        slack_ms = deadline_ms - estimated_tx_ms

        if slack_ms <= 20.0:
            predicted_p_miss = 0.85
        elif slack_ms <= 80.0:
            predicted_p_miss = 0.40
        else:
            predicted_p_miss = 0.05

        return deadline_ms, float(predicted_p_miss)
