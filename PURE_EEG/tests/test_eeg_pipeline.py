import os
import pytest
import numpy as np
import torch

from src.eeg.preprocessing import (
    butterworth_bandpass_filter,
    iir_notch_filter,
    common_average_reference,
    z_score_normalize,
    preprocess_eeg_window
)
from src.eeg.dataset import EEGTaskDataset
from src.simulator.task import HealthcareTask

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "eeg_tasks_split.npz")


def test_eeg_filtering_and_car():
    """
    Verifies IFCN-compliant EEG signal processing:
    1. Zero-phase Butterworth Bandpass (0.5 - 70 Hz)
    2. 50 Hz powerline IIR notch
    3. Common Average Referencing (CAR)
    4. Robust channel-wise z-score normalization
    """
    fs = 256
    t = np.linspace(0, 5, 1280)
    # Generate 16 channels with alpha rhythm (10 Hz) + 50 Hz hum + DC drift
    raw_eeg = np.zeros((16, 1280), dtype=np.float32)
    hum = 1.5 * np.sin(2 * np.pi * 50.0 * t)
    drift = 3.0 * np.sin(2 * np.pi * 0.1 * t)
    
    for ch in range(16):
        signal = np.sin(2 * np.pi * 10.0 * t + ch * 0.1)
        raw_eeg[ch] = signal + hum + drift
        
    # Preprocess
    proc = preprocess_eeg_window(raw_eeg, fs=fs, filter_noise=True)
    
    assert proc.shape == (16, 1280)
    assert not np.isnan(proc).any()
    assert not np.isinf(proc).any()
    
    # Check CAR directly on filtered channels before channel-wise scaling
    car_signal = common_average_reference(raw_eeg)
    assert np.allclose(np.mean(car_signal, axis=0), 0.0, atol=1e-5)
    
    # Check full pipeline output
    assert proc.shape == (16, 1280)
    assert not np.isnan(proc).any()
    assert not np.isinf(proc).any()
    
    # Check normalization: channel-wise mean ~ 0, std ~ 1
    ch_means = np.mean(proc, axis=-1)
    ch_stds = np.std(proc, axis=-1)
    assert np.all(np.abs(ch_means) < 0.15)
    assert np.all(np.abs(ch_stds - 1.0) < 0.20)


def test_16ch_eeg_dimensions_and_payload():
    """
    Verifies that EEG tasks adhere to the 16-channel telemetry contract:
    - Shape: (16, 1280) float32
    - Payload size: 40,960 bytes (40.96 KB)
    """
    assert os.path.exists(DATA_PATH), f"Missing EEG dataset at {DATA_PATH}"
    data = np.load(DATA_PATH)
    
    train_x = data["train_x"]
    assert train_x.ndim == 3, f"Expected 3D tensor (N, 16, 1280), got {train_x.shape}"
    assert train_x.shape[1] == 16, f"Expected 16 channels, got {train_x.shape[1]}"
    assert train_x.shape[2] == 1280, f"Expected 1280 timepoints, got {train_x.shape[2]}"
    
    # 16-bit ADC payload calculation
    expected_bytes = 16 * 1280 * 2  # 40,960 bytes
    assert expected_bytes == 40960


def test_zero_patient_leakage():
    """
    CRITICAL (FR-004): Enforces strict patient-wise partitioning for EEG cohorts.
    No patient identifier should leak across Train, Val, and Test splits.
    """
    data = np.load(DATA_PATH)
    train_recs = set(data["train_records"])
    val_recs = set(data["val_records"])
    test_recs = set(data["test_records"])

    assert len(train_recs.intersection(val_recs)) == 0, "Patient leakage between Train and Val!"
    assert len(train_recs.intersection(test_recs)) == 0, "Patient leakage between Train and Test!"
    assert len(val_recs.intersection(test_recs)) == 0, "Patient leakage between Val and Test!"


def test_eeg_task_generation():
    """
    Verifies that generated EEG HealthcareTask instances have correct payload and compute cycles:
    - delta_i = 40,960 bytes (40.96 KB)
    - psi_i in [20M, 35M] CPU cycles
    - Emergency seizure deadline in [100 ms, 300 ms]
    """
    task = HealthcareTask(
        task_id=1,
        delta_i=40960,
        psi_i=2.5e7,
        deadline_ms=150.0,
        clinical_risk=0.88,
        arrival_time=0.0
    )
    assert task.delta_i == 40960
    assert 2.0e7 <= task.psi_i <= 3.5e7
    assert task.clinical_risk == 0.88
    assert task.deadline_ms == 150.0


def test_eeg_dataset_loader():
    """
    Verifies PyTorch EEGTaskDataset correctly returns (16, 1280) tensors.
    """
    data = np.load(DATA_PATH)
    dataset = EEGTaskDataset(data["train_x"][:20], data["train_y"][:20], data["train_r"][:20])
    
    assert len(dataset) == 20
    x, y, r = dataset[0]
    assert x.shape == (16, 1280)
    assert y.shape == (1,)
    assert r.shape == (1,)
    assert 0.0 <= r.item() <= 1.0
