import os
import pytest
import time
import numpy as np
import torch
from src.models.deadline_predictor import DeadlineMissPredictor

CHECKPOINT_PATH = "/home/dhanrraj/Documents/Default Project/CRAI_LCS/checkpoints/deadline_predictor.pt"
DATA_PATH = "/home/dhanrraj/Documents/Default Project/CRAI_LCS/data/processed/predictor_training_data.npz"


def test_predictor_parameter_budget():
    """
    Verifies that DeadlineMissPredictor adheres to the 0.18M parameter budget (Table 12).
    Target: ~174,497 weights (+/- 5%).
    """
    model = DeadlineMissPredictor(in_features=54)
    num_params = model.total_parameters

    assert 160_000 <= num_params <= 190_000, f"Parameter count {num_params:,} outside [160k, 190k]"


def test_predictor_tensor_contracts_and_bounds():
    """
    Verifies forward pass input-output contracts:
    Input: (B, 54) -> Output: (B, 1) with probabilities in [0.0, 1.0].
    """
    model = DeadlineMissPredictor(in_features=54)
    model.eval()

    batch_x = torch.randn(16, 54)
    with torch.no_grad():
        out = model(batch_x)

    assert out.shape == (16, 1), f"Expected shape (16, 1), got {out.shape}"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output probabilities out of bounds [0.0, 1.0]"


def test_predictor_inference_latency():
    """
    Verifies that single-sample inference latency is well under the 3.6 ms budget (Table 12).
    """
    model = DeadlineMissPredictor(in_features=54)
    model.eval()

    x = torch.randn(1, 54)
    # Warmup
    for _ in range(10):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = model(x)
    avg_latency_ms = (time.perf_counter() - t0) / 50 * 1000.0

    assert avg_latency_ms <= 3.6, f"Latency {avg_latency_ms:.3f} ms exceeds 3.6 ms budget"


def test_predictor_checkpoint_prediction():
    """
    Loads saved checkpoint from disk and verifies predictions on validation/test features.
    """
    assert os.path.exists(CHECKPOINT_PATH), f"Missing checkpoint {CHECKPOINT_PATH}"
    assert os.path.exists(DATA_PATH), f"Missing dataset {DATA_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeadlineMissPredictor(in_features=54).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    data = np.load(DATA_PATH)
    sample_x = torch.from_numpy(data["test_x"][:20]).to(device)

    with torch.no_grad():
        probs = model(sample_x).cpu().numpy().flatten()

    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert not np.isnan(probs).any()
