"""
Unit & Integration Tests for U-CRAI-LCS Research Countermeasures:
1. Asymmetric Duty-Cycle Fallback (60s) & SQI Motion Artifact Gatekeeper
2. Squeeze-and-Excitation (SE) 1D Attention Latent Encoder
3. Time-to-Harm Dynamic Priority Aging with Hard Tier-1 Resuscitation Ceiling
4. Memory-Safe Clinical Decision Support (CDS) Ring Buffer Audit Logger
"""

import pytest
import numpy as np
import torch
import time

from src.universal.schema import ModalityType, UniversalTelemetryTask
from src.universal.asymmetric_gate import AsymmetricSensorGate, GatingDecision
from src.universal.models import UniversalBiosignalEncoder, ECGBranch, PPGBranch, SEBlock1D
from src.universal.two_tier_scheduler import TwoTierUniversalScheduler
from src.universal.audit_logger import ClinicalAuditLogger
from src.universal.pipeline import UniversalHealthcarePipeline


def test_duty_cycle_fallback_and_pacemaker_sqi():
    gate = AsymmetricSensorGate(duty_cycle_interval=12)
    t = np.linspace(0, 5, 1800)
    normal_ecg = (np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(1800)).astype(np.float32)

    # 1. First 11 windows should be routine
    for i in range(11):
        dec = gate.screen_window(normal_ecg, ModalityType.CARDIAC_ECG, patient_id=1)
        assert dec == GatingDecision.NORMAL_ROUTINE

    # 2. Window 12 MUST trigger DUTY_CYCLE_FALLBACK (guarantees silent arrhythmia detection)
    dec_12 = gate.screen_window(normal_ecg, ModalityType.CARDIAC_ECG, patient_id=1)
    assert dec_12 == GatingDecision.DUTY_CYCLE_FALLBACK

    # 3. Severe rail-to-rail saturation (sensor off / broad motion collision) without pacemaker
    # Injected baseline + intermittent heavy rail spikes
    corrupted_signal = np.zeros(1800, dtype=np.float32)
    corrupted_signal[:600] = 10.0  # 33% of window rail-clipped
    dec_corrupt = gate.screen_window(corrupted_signal, ModalityType.CARDIAC_ECG, patient_id=2, has_pacemaker=False)
    assert dec_corrupt == GatingDecision.CORRUPTED_ARTIFACT

    # 4. Same sharp signal with has_pacemaker=True is preserved
    dec_pacemaker = gate.screen_window(corrupted_signal, ModalityType.CARDIAC_ECG, patient_id=3, has_pacemaker=True)
    assert dec_pacemaker != GatingDecision.CORRUPTED_ARTIFACT


def test_se_attention_encoder():
    ecg_branch = ECGBranch(embedding_dim=128)
    params = sum(p.numel() for p in ecg_branch.parameters())
    # Squeeze-and-Excitation adds channel attention, keeping total under 45k params (vs 741k in previous)
    assert params < 45000

    x = torch.randn(2, 1, 1800)
    out = ecg_branch(x)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()

    # Full universal encoder with SE attention
    encoder = UniversalBiosignalEncoder(embedding_dim=128)
    z_ecg = encoder(x, ModalityType.CARDIAC_ECG)
    assert z_ecg.shape == (2, 128)
    # Check L2 normalization
    norms = torch.norm(z_ecg, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_priority_aging_with_hard_tier1_ceiling():
    scheduler = TwoTierUniversalScheduler(num_edge_nodes=25)
    obs = np.zeros(54, dtype=np.float32)

    # 1. Subacute Respiratory task (Urgency = 0.15, Arrival = 0.0s)
    task_resp = UniversalTelemetryTask(
        task_id=101,
        modality=ModalityType.RESPIRATORY_PPG,
        patient_id=10,
        raw_signal=np.zeros(500, dtype=np.float32),
        sampling_rate_hz=100,
        arrival_timestamp_s=0.0,
        payload_bytes=1000,
        computational_demand_cycles=5e6,
        clinical_urgency=0.15
    )

    # At t=0s: task is fresh, handled normally
    act_fresh, is_fs_fresh, desc_fresh = scheduler.select_action(task_resp, obs, current_time=0.0)
    assert not is_fs_fresh

    # At t=90s: task has waited half of T_harm (180s), aging elevates priority to AGED PRIORITY
    act_aged, is_fs_aged, desc_aged = scheduler.select_action(task_resp, obs, current_time=90.0)
    assert "[AGED PRIORITY]" in desc_aged
    assert not is_fs_aged  # Subacute task NEVER gets emergency fail-safe

    # At t=500s: even with massive wait, Hard Tier-1 Ceiling caps effective urgency at 0.499
    # so it NEVER preempts true resuscitation emergencies
    act_ancient, is_fs_ancient, desc_ancient = scheduler.select_action(task_resp, obs, current_time=500.0)
    assert not is_fs_ancient

    # 2. Fresh Acute Cardiac Arrest task (Urgency = 0.95, Arrival = 500.0s)
    task_cardiac = UniversalTelemetryTask(
        task_id=102,
        modality=ModalityType.CARDIAC_ECG,
        patient_id=20,
        raw_signal=np.zeros(1800, dtype=np.float32),
        sampling_rate_hz=360,
        arrival_timestamp_s=500.0,
        payload_bytes=3600,
        computational_demand_cycles=1e7,
        clinical_urgency=0.95
    )
    # Cardiac arrest strictly triggers immediate local fail-safe preemption
    act_cardiac, is_fs_cardiac, desc_cardiac = scheduler.select_action(task_cardiac, obs, current_time=500.0)
    assert is_fs_cardiac
    assert "[FAIL-SAFE LOCAL]" in desc_cardiac


def test_clinical_audit_logger_ring_buffer(tmp_path):
    logger = ClinicalAuditLogger(buffer_size=100)

    # Log 10 critical emergency events
    for i in range(10):
        t = UniversalTelemetryTask(
            task_id=i + 1,
            modality=ModalityType.CARDIAC_ECG,
            patient_id=i + 100,
            raw_signal=np.zeros(10),
            sampling_rate_hz=360,
            arrival_timestamp_s=i * 0.1,
            payload_bytes=3600,
            computational_demand_cycles=1e7,
            clinical_risk=0.85,
            clinical_urgency=0.90,
            actual_latency_ms=12.5,
            deadline_ms=80.0
        )
        logger.log_task_event(t, action=0, is_fail_safe=True, target_description="Edge Gateway 00 [FAIL-SAFE]")

    # Log 50 routine events
    for i in range(50):
        t_rout = UniversalTelemetryTask(
            task_id=100 + i,
            modality=ModalityType.RESPIRATORY_PPG,
            patient_id=i + 200,
            raw_signal=np.zeros(10),
            sampling_rate_hz=100,
            arrival_timestamp_s=i * 0.1,
            payload_bytes=1000,
            computational_demand_cycles=5e6,
            clinical_risk=0.05,
            clinical_urgency=0.10,
            actual_latency_ms=8.0,
            deadline_ms=800.0
        )
        logger.log_task_event(t_rout, action=19, is_fail_safe=False, target_description="Edge Gateway 19")

    # Critical ring buffer should have exactly 10 entries
    assert len(logger.critical_events_ring) == 10
    recent = logger.get_recent_critical_events(5)
    assert len(recent) == 5

    # Routine rollups aggregate data without memory bloat
    log_file = tmp_path / "audit_test.json"
    logger.flush_to_disk(str(log_file))
    assert log_file.exists()
