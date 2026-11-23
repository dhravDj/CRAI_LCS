"""
Unit & Regression Tests for Universal Telemetry Adapters & Schema (Phase U1).
Verifies:
1. Multi-modal task representation compliance.
2. Digital filtering (Butterworth bandpass + notch) stability.
3. Execution speed (< 5ms per 5-second window).
"""

import time
import pytest
import numpy as np

from src.universal.schema import ModalityType, UniversalTelemetryTask, TIME_TO_HARM_SECONDS
from src.universal.adapters import (
    CardiacECGAdapter,
    NeurologicalEEGAdapter,
    RespiratoryPPGAdapter,
    UniversalTelemetryIngestor
)


def test_universal_schema_contract():
    dummy_signal = np.zeros(1800, dtype=np.float32)
    task = UniversalTelemetryTask(
        task_id=1,
        modality=ModalityType.CARDIAC_ECG,
        patient_id=101,
        raw_signal=dummy_signal,
        sampling_rate_hz=360,
        arrival_timestamp_s=0.0,
        payload_bytes=3600,
        computational_demand_cycles=1.0e7,
        clinical_urgency=0.65
    )
    assert task.is_urgent is True
    assert task.deadline_seconds == 0.5
    assert TIME_TO_HARM_SECONDS[ModalityType.CARDIAC_ECG] == 30.0


def test_cardiac_ecg_adapter():
    adapter = CardiacECGAdapter(fs=360)
    # Synthetic 5-second ECG with 50 Hz noise
    t = np.linspace(0, 5, 1800)
    clean = np.sin(2 * np.pi * 1.2 * t)
    noisy = clean + 0.5 * np.sin(2 * np.pi * 50.0 * t) + 0.3 * np.sin(2 * np.pi * 0.1 * t)

    t0 = time.perf_counter()
    filtered = adapter.condition_signal(noisy)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert filtered.shape == (1800,)
    assert abs(np.mean(filtered)) < 1e-4
    assert abs(np.std(filtered) - 1.0) < 1e-2
    assert elapsed_ms < 5.0, f"ECG conditioning took {elapsed_ms:.2f} ms (budget <= 5.0 ms)"

    payload, cycles = adapter.estimate_demand(filtered)
    assert payload == 3600
    assert cycles == 1.0e7


def test_neurological_eeg_adapter():
    adapter = NeurologicalEEGAdapter(fs=256, channels=16)
    # 16-channel 5-second EEG (1280 samples per channel)
    raw_eeg = np.random.randn(16, 1280).astype(np.float32)

    t0 = time.perf_counter()
    filtered = adapter.condition_signal(raw_eeg)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert filtered.shape == (16, 1280)
    for ch in range(16):
        assert abs(np.mean(filtered[ch])) < 1e-4
        assert abs(np.std(filtered[ch]) - 1.0) < 1e-2
    assert elapsed_ms < 15.0, f"EEG 16-ch conditioning took {elapsed_ms:.2f} ms"

    payload, cycles = adapter.estimate_demand(filtered)
    assert payload == 16 * 1280 * 2
    assert cycles == 3.5e7


def test_respiratory_ppg_adapter():
    adapter = RespiratoryPPGAdapter(fs=100)
    # 5-second PPG (500 samples)
    t = np.linspace(0, 5, 500)
    raw_ppg = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.random.randn(500)

    t0 = time.perf_counter()
    filtered = adapter.condition_signal(raw_ppg)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert filtered.shape == (500,)
    assert abs(np.mean(filtered)) < 1e-4
    assert elapsed_ms < 5.0

    payload, cycles = adapter.estimate_demand(filtered)
    assert payload == 1000
    assert cycles == 5.0e6


def test_universal_telemetry_ingestor():
    ingestor = UniversalTelemetryIngestor()

    # 1. Ingest ECG
    ecg_raw = np.random.randn(1800).astype(np.float32)
    t_ecg = ingestor.ingest(1, ModalityType.CARDIAC_ECG, 101, ecg_raw, 0.0)
    assert t_ecg.modality == ModalityType.CARDIAC_ECG
    assert t_ecg.payload_bytes == 3600
    assert t_ecg.raw_signal.shape == (1800,)

    # 2. Ingest EEG
    eeg_raw = np.random.randn(16, 1280).astype(np.float32)
    t_eeg = ingestor.ingest(2, ModalityType.NEURO_EEG, 102, eeg_raw, 0.05)
    assert t_eeg.modality == ModalityType.NEURO_EEG
    assert t_eeg.payload_bytes == 40960
    assert t_eeg.raw_signal.shape == (16, 1280)

    # 3. Ingest PPG
    ppg_raw = np.random.randn(500).astype(np.float32)
    t_ppg = ingestor.ingest(3, ModalityType.RESPIRATORY_PPG, 103, ppg_raw, 0.10)
    assert t_ppg.modality == ModalityType.RESPIRATORY_PPG
    assert t_ppg.payload_bytes == 1000
    assert t_ppg.raw_signal.shape == (500,)


def test_eeg_single_channel_broadcast_and_flatline():
    adapter = NeurologicalEEGAdapter(fs=256, channels=16)
    # 1D single-channel input
    single_ch = np.random.randn(1280).astype(np.float32)
    out_16ch = adapter.condition_signal(single_ch)
    assert out_16ch.shape == (16, 1280)

    # Flatline / zero-amplitude signal handling
    zero_ecg = np.zeros(1800, dtype=np.float32)
    ecg_adapter = CardiacECGAdapter(fs=360)
    ecg_out = ecg_adapter.condition_signal(zero_ecg)
    assert not np.isnan(ecg_out).any()
    assert not np.isinf(ecg_out).any()
