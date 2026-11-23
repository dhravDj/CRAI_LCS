import os
import pytest
import numpy as np
import torch
from src.ecg.preprocessing import (
    z_score_normalize,
    extract_record_segments,
    create_ecg_healthcare_task
)
from src.ecg.dataset import ECGTaskDataset

PROCESSED_DATA_PATH = "/home/dhanrraj/Documents/Default Project/CRAI_LCS/data/processed/ecg_tasks_split.npz"


def test_z_score_normalization():
    """Verifies that Z-score normalization achieves mean ~ 0 and std ~ 1, and handles flatlines."""
    # Test random signal
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    norm = z_score_normalize(signal)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-6)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)

    # Test flatline / zero-variance signal (electrode detachment)
    flatline = np.ones(1800, dtype=np.float32) * 2.5
    norm_flat = z_score_normalize(flatline)
    assert not np.isnan(norm_flat).any(), "Normalization must not produce NaNs on flatline"
    assert not np.isinf(norm_flat).any(), "Normalization must not produce Infs on flatline"


def test_segment_dimensions_and_payload():
    """Verifies that extracted ECG segments conform to 5-second 1800-sample windowing (T=1800, 3.6KB)."""
    assert os.path.exists(PROCESSED_DATA_PATH), f"Missing {PROCESSED_DATA_PATH}"
    data = np.load(PROCESSED_DATA_PATH)

    train_x = data["train_x"]
    assert train_x.ndim == 2
    assert train_x.shape[1] == 1800, f"Expected segment length 1800, got {train_x.shape[1]}"
    assert train_x.dtype == np.float32


def test_zero_patient_data_leakage():
    """
    CRITICAL (FR-004): Enforces strict record-wise patient partitioning.
    No patient record may appear in more than one split (Train, Val, Test).
    """
    data = np.load(PROCESSED_DATA_PATH)
    train_recs = set(data["train_records"])
    val_recs = set(data["val_records"])
    test_recs = set(data["test_records"])

    assert len(train_recs & val_recs) == 0, f"Leakage detected between train and val: {train_recs & val_recs}"
    assert len(train_recs & test_recs) == 0, f"Leakage detected between train and test: {train_recs & test_recs}"
    assert len(val_recs & test_recs) == 0, f"Leakage detected between val and test: {val_recs & test_recs}"


def test_clinical_risk_mapping_and_bounds():
    """Verifies that clinical labels y_i in {0, 1} and risk scores R_i in [0.0, 1.0] adhere to AAMI EC57 bounds."""
    data = np.load(PROCESSED_DATA_PATH)
    train_y = data["train_y"]
    train_r = data["train_r"]

    # Labels must be binary
    unique_labels = set(np.unique(train_y))
    assert unique_labels.issubset({0, 1}), f"Labels must be binary {0, 1}, got {unique_labels}"

    # Risk scores must be within [0.0, 1.0]
    assert np.all(train_r >= 0.0) and np.all(train_r <= 1.0), "Risk scores outside [0.0, 1.0]"

    # Urgent cases must have R_i >= 0.50
    urgent_indices = np.where(train_y == 1)[0]
    if len(urgent_indices) > 0:
        assert np.all(train_r[urgent_indices] >= 0.50), "Urgent cardiac cases must have R_i >= 0.50"


def test_ecg_to_healthcare_task_conversion():
    """Verifies conversion of real ECG segments into Gymnasium HealthcareTask instances."""
    segment = np.random.randn(1800).astype(np.float32)

    # Urgent Task
    urgent_task = create_ecg_healthcare_task(task_id=1, segment=segment, clinical_risk=0.88, arrival_time=2.0)
    assert urgent_task.delta_i == 3600.0
    assert 5e6 <= urgent_task.psi_i <= 2e7
    assert 100.0 <= urgent_task.deadline_ms <= 150.0
    assert urgent_task.is_urgent is True

    # Routine Task
    routine_task = create_ecg_healthcare_task(task_id=2, segment=segment, clinical_risk=0.20, arrival_time=3.0)
    assert routine_task.delta_i == 3600.0
    assert 400.0 <= routine_task.deadline_ms <= 800.0
    assert routine_task.is_urgent is False


def test_pytorch_dataset_loader():
    """Verifies PyTorch Dataset class output shapes and tensor types for Phase 4 CNN-LSTM training."""
    data = np.load(PROCESSED_DATA_PATH)
    ds = ECGTaskDataset(data["train_x"][:20], data["train_y"][:20], data["train_r"][:20])

    assert len(ds) == 20
    x, y, r = ds[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 1800), f"Expected Conv1D shape (1, 1800), got {x.shape}"
    assert y.shape == (1,)
    assert r.shape == (1,)
    assert y.item() in [0.0, 1.0]
    assert 0.0 <= r.item() <= 1.0
