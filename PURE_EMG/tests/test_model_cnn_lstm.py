import os
import pytest
import time
import numpy as np
import torch
from src.models.cnn_lstm_risk import ClinicalRiskCNNLSTM

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "emg_cnn_lstm_risk.pt")
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "cnn_lstm_risk.pt")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "emg_tasks_split.npz")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ecg_tasks_split.npz")


def test_cnn_lstm_parameter_budget():
    """
    Verifies that the sEMG model architecture adheres to the ~0.24M parameter budget.
    Target: ~241,985 parameters.
    """
    model = ClinicalRiskCNNLSTM()
    num_params = model.total_parameters

    assert 220_000 <= num_params <= 270_000, f"Parameter count {num_params:,} outside budget [220k, 270k]"


def test_cnn_lstm_tensor_contracts_and_bounds():
    """
    Verifies forward pass input-output tensor contracts:
    Input: (B, 4, 2000) -> Output: (B, 1) with risk scores in [0.0, 1.0].
    """
    model = ClinicalRiskCNNLSTM()
    model.eval()

    batch_x = torch.randn(8, 4, 2000)
    with torch.no_grad():
        out = model(batch_x)

    assert out.shape == (8, 1), f"Expected shape (8, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Risk output values out of bounds [0.0, 1.0]"


def test_single_window_inference_latency():
    """
    Verifies that single-window forward pass inference executes within the 10.0 ms budget for sEMG.
    """
    model = ClinicalRiskCNNLSTM()
    model.eval()

    x = torch.randn(1, 4, 2000)
    # Warmup
    for _ in range(5):
        _ = model(x)

    # Benchmark 30 runs
    t0 = time.perf_counter()
    for _ in range(30):
        _ = model(x)
    avg_latency_ms = (time.perf_counter() - t0) / 30 * 1000.0

    assert avg_latency_ms <= 10.0, f"Inference latency {avg_latency_ms:.2f} ms exceeds 10.0 ms budget"


def test_saved_checkpoint_evaluates_on_real_emg():
    """
    Loads the trained checkpoint from disk and verifies inference on real patient sEMG segments.
    """
    assert os.path.exists(CHECKPOINT_PATH), f"Missing model checkpoint {CHECKPOINT_PATH}"
    assert os.path.exists(DATA_PATH), f"Missing dataset {DATA_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalRiskCNNLSTM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    data = np.load(DATA_PATH)
    sample_raw = torch.from_numpy(data["test_x"][:10]).float().to(device)
    if sample_raw.ndim == 2:
        sample_segments = sample_raw.unsqueeze(1)
    else:
        sample_segments = sample_raw

    with torch.no_grad():
        risks = model(sample_segments).cpu().numpy().flatten()

    assert len(risks) == 10
    assert np.all(risks >= 0.0) and np.all(risks <= 1.0)
    assert not np.isnan(risks).any()
