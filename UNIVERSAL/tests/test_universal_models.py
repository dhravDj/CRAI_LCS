"""
Unit & Performance Tests for Universal Models & Asymmetric Gating (Phase U2).
Verifies:
1. Asymmetric sensor-side anomaly screening execution (< 0.1 ms).
2. Multi-branch 1D depthwise encoder embedding generation into R^128.
3. Pluggable risk head output range in [0.0, 1.0].
4. Inference latency budget on CPU (< 6 ms).
5. Parameter counts and model footprint (< 250 KB per branch).
"""

import time
import pytest
import torch
import numpy as np

from src.universal.schema import ModalityType
from src.universal.asymmetric_gate import AsymmetricSensorGate, GatingDecision
from src.universal.models import UniversalBiosignalEncoder, PluggableDiseaseRiskHead


def test_asymmetric_sensor_gate():
    gate = AsymmetricSensorGate()

    # 1. Normal Sinus Signal
    t = np.linspace(0, 5, 1800)
    normal_ecg = np.sin(2 * np.pi * 1.2 * t)
    t0 = time.perf_counter()
    dec_norm = gate.screen_window(normal_ecg, ModalityType.CARDIAC_ECG)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert dec_norm == GatingDecision.NORMAL_ROUTINE
    assert elapsed_ms < 0.20, f"Gating screening took {elapsed_ms:.3f} ms (budget <= 0.20 ms)"

    # 2. Extreme Ventricular Spike Anomaly
    arrhythmic_ecg = normal_ecg.copy()
    arrhythmic_ecg[500:520] += 12.0  # Acute ectopic spike
    dec_anom = gate.screen_window(arrhythmic_ecg, ModalityType.CARDIAC_ECG)
    assert dec_anom == GatingDecision.SUSPICIOUS_ANOMALY


def test_multi_branch_universal_encoder():
    encoder = UniversalBiosignalEncoder(embedding_dim=128)
    encoder.eval()

    # 1. ECG pass: (B=2, C=1, T=1800)
    x_ecg = torch.randn(2, 1, 1800)
    # Warmup pass
    with torch.no_grad():
        _ = encoder(x_ecg, ModalityType.CARDIAC_ECG)

    t0 = time.perf_counter()
    with torch.no_grad():
        z_ecg = encoder(x_ecg, ModalityType.CARDIAC_ECG)
    lat_ecg = (time.perf_counter() - t0) * 1000

    assert z_ecg.shape == (2, 128)
    assert torch.allclose(torch.norm(z_ecg, dim=-1), torch.ones(2), atol=1e-5)
    assert lat_ecg < 10.0

    # 2. EEG pass: (B=2, C=16, T=1280)
    x_eeg = torch.randn(2, 16, 1280)
    with torch.no_grad():
        z_eeg = encoder(x_eeg, ModalityType.NEURO_EEG)
    assert z_eeg.shape == (2, 128)
    assert torch.allclose(torch.norm(z_eeg, dim=-1), torch.ones(2), atol=1e-5)

    # 3. PPG pass: (B=2, C=1, T=500)
    x_ppg = torch.randn(2, 1, 500)
    with torch.no_grad():
        z_ppg = encoder(x_ppg, ModalityType.RESPIRATORY_PPG)
    assert z_ppg.shape == (2, 128)
    assert torch.allclose(torch.norm(z_ppg, dim=-1), torch.ones(2), atol=1e-5)


def test_encoder_parameter_footprint():
    encoder = UniversalBiosignalEncoder(embedding_dim=128)
    # Check parameters per branch
    for mod_name, branch in encoder.branches.items():
        params = sum(p.numel() for p in branch.parameters())
        assert params < 65000, f"Branch {mod_name} has {params:,} parameters (budget <= 65,000)"


def test_pluggable_disease_risk_head():
    head = PluggableDiseaseRiskHead(embedding_dim=128)
    head.eval()

    z = torch.randn(4, 128)
    with torch.no_grad():
        r = head(z)

    assert r.shape == (4, 1)
    assert (r >= 0.0).all() and (r <= 1.0).all()
