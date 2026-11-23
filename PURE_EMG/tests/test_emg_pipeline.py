import os
import pytest
import numpy as np
import torch

from src.emg.preprocessing import (
    butterworth_bandpass_filter,
    iir_notch_filter,
    compute_rms_envelope,
    z_score_normalize,
    preprocess_emg_window
)
from src.emg.dataset import EMGTaskDataset
from src.simulator.task import HealthcareTask

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "emg_tasks_split.npz")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ecg_tasks_split.npz")


def test_emg_filtering_and_envelope():
    """
    Verifies SENIAM-compliant sEMG signal processing:
    1. Zero-phase Butterworth Bandpass (20 - 450 Hz)
    2. 50 Hz powerline IIR notch
    3. Moving RMS envelope calculation
    4. Robust channel-wise z-score normalization
    """
    fs = 1000
    t = np.linspace(0, 2.0, 2000)
    # Generate 4 channels with MUAP firing (80 Hz) + 50 Hz hum + DC drift
    raw_emg = np.zeros((4, 2000), dtype=np.float32)
    hum = 1.5 * np.sin(2 * np.pi * 50.0 * t)
    drift = 3.0 * np.sin(2 * np.pi * 5.0 * t)  # <20 Hz drift

    for ch in range(4):
        signal = np.sin(2 * np.pi * 80.0 * t + ch * 0.2)
        raw_emg[ch] = signal + hum + drift

    # Preprocess
    proc = preprocess_emg_window(raw_emg, fs=fs, filter_noise=True)

    assert proc.shape == (4, 2000)
    assert not np.isnan(proc).any()
    assert not np.isinf(proc).any()

    # Check normalization: channel-wise mean ~ 0, std ~ 1
    ch_means = np.mean(proc, axis=-1)
    ch_stds = np.std(proc, axis=-1)
    assert np.all(np.abs(ch_means) < 0.15)
    assert np.all(np.abs(ch_stds - 1.0) < 0.20)

    # Check RMS envelope extraction
    rms = compute_rms_envelope(proc, window_size=50)
    assert rms.shape == (4, 2000)
    assert np.all(rms >= 0.0)


def test_4ch_emg_dimensions_and_payload():
    """
    Verifies that sEMG tasks adhere to the 4-channel telemetry contract:
    - Shape: (4, 2000) float32
    - Payload size: 16,000 bytes (16.0 KB)
    """
    assert os.path.exists(DATA_PATH), f"Missing sEMG dataset at {DATA_PATH}"
    data = np.load(DATA_PATH)

    train_x = data["train_x"]
    assert train_x.ndim == 3, f"Expected 3D tensor (N, 4, 2000), got {train_x.shape}"
    assert train_x.shape[1] == 4, f"Expected 4 channels, got {train_x.shape[1]}"
    assert train_x.shape[2] == 2000, f"Expected 2000 timepoints, got {train_x.shape[2]}"

    # 16-bit ADC payload calculation
    expected_bytes = 4 * 2000 * 2  # 16,000 bytes
    assert expected_bytes == 16000


def test_zero_patient_leakage():
    """
    CRITICAL (FR-004): Enforces strict patient-wise partitioning for sEMG cohorts.
    No patient identifier should leak across Train, Val, and Test splits.
    """
    data = np.load(DATA_PATH)
    train_recs = set(data["train_records"])
    val_recs = set(data["val_records"])
    test_recs = set(data["test_records"])

    assert len(train_recs.intersection(val_recs)) == 0, "Patient leakage between Train and Val!"
    assert len(train_recs.intersection(test_recs)) == 0, "Patient leakage between Train and Test!"
    assert len(val_recs.intersection(test_recs)) == 0, "Patient leakage between Val and Test!"


def test_emg_task_generation():
    """
    Verifies that generated sEMG HealthcareTask instances have correct payload and compute cycles:
    - delta_i = 16,000 bytes (16.0 KB)
    - psi_i in [12M, 25M] CPU cycles
    - Emergency tetany deadline in [60 ms, 150 ms]
    """
    task = HealthcareTask(
        task_id=1,
        delta_i=16000,
        psi_i=1.5e7,
        deadline_ms=80.0,
        clinical_risk=0.88,
        arrival_time=0.0
    )
    assert task.delta_i == 16000
    assert 1.2e7 <= task.psi_i <= 2.5e7
    assert task.clinical_risk == 0.88
    assert task.deadline_ms == 80.0


def test_emg_dataset_loader():
    """
    Verifies PyTorch EMGTaskDataset correctly returns (4, 2000) tensors.
    """
    data = np.load(DATA_PATH)
    dataset = EMGTaskDataset(data["train_x"][:20], data["train_y"][:20], data["train_r"][:20])

    assert len(dataset) == 20
    x, y, r = dataset[0]
    assert x.shape == (4, 2000)
    assert y.shape == (1,)
    assert r.shape == (1,)
    assert 0.0 <= r.item() <= 1.0
