import os
import pytest
import time
import numpy as np
import torch
from src.models.cnn_lstm_risk import ClinicalRiskCNNLSTM

CHECKPOINT_PATH = "/home/dhanrraj/Documents/Default Project/CRAI_LCS/checkpoints/cnn_lstm_risk.pt"
DATA_PATH = "/home/dhanrraj/Documents/Default Project/CRAI_LCS/data/processed/ecg_tasks_split.npz"


def test_cnn_lstm_parameter_budget():
    """
    Verifies that the model architecture adheres to the 0.74M parameter budget (Table 12).
    Target: ~741,201 parameters within +/- 2%.
    """
    model = ClinicalRiskCNNLSTM()
    num_params = model.total_parameters

    assert 720_000 <= num_params <= 760_000, f"Parameter count {num_params:,} outside budget [720k, 760k]"


def test_cnn_lstm_tensor_contracts_and_bounds():
    """
    Verifies forward pass input-output tensor contracts:
    Input: (B, 1, 1800) -> Output: (B, 1) with risk scores in [0.0, 1.0].
    """
    model = ClinicalRiskCNNLSTM()
    model.eval()

    batch_x = torch.randn(8, 1, 1800)
    with torch.no_grad():
        out = model(batch_x)

    assert out.shape == (8, 1), f"Expected shape (8, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Risk output values out of bounds [0.0, 1.0]"


def test_single_window_inference_latency():
    """
    Verifies that single-window forward pass inference executes within the 14.8 ms budget (Table 12).
    """
    model = ClinicalRiskCNNLSTM()
    model.eval()

    x = torch.randn(1, 1, 1800)
    # Warmup
    for _ in range(5):
        _ = model(x)

    # Benchmark 30 runs
    t0 = time.perf_counter()
    for _ in range(30):
        _ = model(x)
    avg_latency_ms = (time.perf_counter() - t0) / 30 * 1000.0

    assert avg_latency_ms <= 14.8, f"Inference latency {avg_latency_ms:.2f} ms exceeds 14.8 ms budget"


def test_saved_checkpoint_evaluates_on_real_ecg():
    """
    Loads the trained checkpoint from disk and verifies inference on real patient ECG segments.
    """
    assert os.path.exists(CHECKPOINT_PATH), f"Missing model checkpoint {CHECKPOINT_PATH}"
    assert os.path.exists(DATA_PATH), f"Missing dataset {DATA_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalRiskCNNLSTM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    data = np.load(DATA_PATH)
    sample_segments = torch.from_numpy(data["test_x"][:10]).unsqueeze(1).to(device)

    with torch.no_grad():
        risks = model(sample_segments).cpu().numpy().flatten()

    assert len(risks) == 10
    assert np.all(risks >= 0.0) and np.all(risks <= 1.0)
    assert not np.isnan(risks).any()
