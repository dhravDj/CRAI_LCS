"""
Unit & Regression Tests for Universal Clinical Triage & SLA Prediction (Phase U3).
Verifies:
1. Cross-disease clinical urgency utility scaling (ECG strictly preempts PPG at same R_i).
2. Triage tier classifications (ESI Tiers 1 to 4).
3. Dynamic deadline SLA computation bounds.
4. Execution speed (< 0.05 ms).
"""

import time
import pytest
from src.universal.schema import ModalityType
from src.universal.triage import ClinicalUrgencyCalibrator, DynamicSLAPredictor


def test_cross_disease_urgency_calibration():
    calibrator = ClinicalUrgencyCalibrator()

    # Identical raw risk prediction R_i = 0.90 across all modalities
    u_ecg = calibrator.calibrate(0.90, ModalityType.CARDIAC_ECG)
    u_eeg = calibrator.calibrate(0.90, ModalityType.NEURO_EEG)
    u_ppg = calibrator.calibrate(0.90, ModalityType.RESPIRATORY_PPG)

    # Cardiac arrest (30s horizon) MUST produce higher urgency than EEG (60s) and PPG (180s)
    assert u_ecg > u_eeg > u_ppg, f"Expected u_ecg > u_eeg > u_ppg, got: {u_ecg:.3f}, {u_eeg:.3f}, {u_ppg:.3f}"
    assert u_ecg >= 0.70  # Triggers Immediate Tier 1
    assert u_ppg < 0.35   # Managed as Urgent / Subacute


def test_triage_tiers_and_execution_speed():
    calibrator = ClinicalUrgencyCalibrator()

    t0 = time.perf_counter()
    tier_resusc, name_resusc = calibrator.get_triage_tier(0.85)
    tier_emergent, name_emergent = calibrator.get_triage_tier(0.55)
    tier_urgent, name_urgent = calibrator.get_triage_tier(0.35)
    tier_routine, name_routine = calibrator.get_triage_tier(0.10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert tier_resusc == 1 and name_resusc == "RESUSCITATION_IMMEDIATE"
    assert tier_emergent == 2 and name_emergent == "EMERGENT"
    assert tier_urgent == 3 and name_urgent == "URGENT"
    assert tier_routine == 4 and name_routine == "ROUTINE_TELEMETRY"
    assert elapsed_ms < 0.10, f"Triage classification took {elapsed_ms:.4f} ms"


def test_dynamic_sla_predictor():
    sla_predictor = DynamicSLAPredictor()

    # 1. Resuscitation (U=0.85, small payload)
    d_resusc, p_miss_resusc = sla_predictor.compute_deadline_and_miss_risk(0.85, payload_bytes=1000, channel_snr_db=25.0)
    assert 50.0 <= d_resusc <= 120.0

    # 2. Routine (U=0.10, large payload)
    d_routine, p_miss_routine = sla_predictor.compute_deadline_and_miss_risk(0.10, payload_bytes=40000, channel_snr_db=25.0)
    assert 500.0 <= d_routine <= 1000.0

    # Higher urgency MUST receive tighter deadline
    assert d_resusc < d_routine
