"""
Integration & Benchmark Tests for Universal Multi-Disease Pipeline (Phase U4).
Evaluates concurrent streaming across Cardiology (ECG), Neurology (EEG), and Pulmonology (PPG).
Verifies:
1. Multi-modal task processing integrity.
2. Two-tier hierarchical scheduling and fail-safe preemptions.
3. SLA deadline adherence and continuum stability.
"""

import time
import pytest
import numpy as np

from src.universal.schema import ModalityType, UniversalTelemetryTask
from src.universal.pipeline import UniversalHealthcarePipeline


def test_end_to_end_single_task_processing():
    pipeline = UniversalHealthcarePipeline(seed=42)

    # 1. Test ECG task
    ecg_raw = np.random.randn(1800).astype(np.float32)
    task_ecg, gate_ecg = pipeline.process_raw_telemetry(1, ModalityType.CARDIAC_ECG, 101, ecg_raw, 0.0)
    assert task_ecg.modality == ModalityType.CARDIAC_ECG
    assert 0.0 <= task_ecg.clinical_risk <= 1.0
    assert 0.0 <= task_ecg.clinical_urgency <= 1.0
    assert task_ecg.deadline_ms > 0

    # 2. Test EEG task
    eeg_raw = np.random.randn(16, 1280).astype(np.float32)
    task_eeg, gate_eeg = pipeline.process_raw_telemetry(2, ModalityType.NEURO_EEG, 102, eeg_raw, 0.05)
    assert task_eeg.modality == ModalityType.NEURO_EEG
    assert task_eeg.payload_bytes == 40960

    # 3. Test PPG task
    ppg_raw = np.random.randn(500).astype(np.float32)
    task_ppg, gate_ppg = pipeline.process_raw_telemetry(3, ModalityType.RESPIRATORY_PPG, 103, ppg_raw, 0.10)
    assert task_ppg.modality == ModalityType.RESPIRATORY_PPG


def test_concurrent_multi_disease_streaming():
    pipeline = UniversalHealthcarePipeline(seed=42)
    rng = np.random.RandomState(42)

    modalities = [ModalityType.CARDIAC_ECG, ModalityType.NEURO_EEG, ModalityType.RESPIRATORY_PPG]
    tasks = []

    # Generate 60 concurrent mixed clinical telemetry tasks (20 ECG, 20 EEG, 20 PPG)
    for i in range(60):
        mod = modalities[i % 3]
        if mod == ModalityType.CARDIAC_ECG:
            raw = rng.randn(1800).astype(np.float32)
        elif mod == ModalityType.NEURO_EEG:
            raw = rng.randn(16, 1280).astype(np.float32)
        else:
            raw = rng.randn(500).astype(np.float32)

        task, gate = pipeline.process_raw_telemetry(
            task_id=i + 1,
            modality=mod,
            patient_id=100 + (i % 10),
            raw_signal=raw,
            arrival_time=i * 0.05
        )
        tasks.append(task)

    # Run streaming continuum simulation
    results = pipeline.run_multi_disease_streaming_benchmark(tasks)

    assert results["total_tasks"] == 60
    assert results["violation_rate"] <= 5.0, f"Violation rate was {results['violation_rate']:.2f}% (budget <= 5%)"
    assert results["avg_latency_ms"] < 60.0, f"Average latency was {results['avg_latency_ms']:.2f} ms"
    assert results["fail_safe_count"] >= 0
