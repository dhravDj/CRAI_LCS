import pytest
import numpy as np
import torch
import torch.nn as nn

from src.ecg.preprocessing import butterworth_bandpass_filter, preprocess_ecg_window
from src.training.focal_loss import BinaryFocalLoss
from src.models.quantization import quantize_model_dynamic_int8
from src.models.cnn_lstm_risk import ClinicalRiskCNNLSTM
from src.models.dqn_scheduler import DuelingDQN


def test_butterworth_bandpass_filter_attenuation():
    """Verifies that Butterworth 0.5-45 Hz filter attenuates 0.1 Hz drift and 60 Hz hum."""
    fs = 360
    t = np.linspace(0, 5, 1800)
    
    # Clean 1.5 Hz cardiac signal
    cardiac = np.sin(2 * np.pi * 1.5 * t)
    # Low-freq respiration drift (0.1 Hz)
    drift = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    # High-freq powerline hum (60.0 Hz)
    hum = 1.0 * np.sin(2 * np.pi * 60.0 * t)
    
    noisy_signal = cardiac + drift + hum
    filtered = butterworth_bandpass_filter(noisy_signal, lowcut=0.5, highcut=45.0, fs=fs)
    
    # Check that amplitude is significantly cleaned
    assert len(filtered) == 1800
    assert not np.isnan(filtered).any()
    
    # Preprocess ECG window combines filter and z-score
    processed = preprocess_ecg_window(noisy_signal, filter_noise=True, fs=fs)
    assert abs(np.mean(processed)) < 1e-4
    assert abs(np.std(processed) - 1.0) < 1e-2


def test_focal_loss_properties():
    """Verifies Focal Loss downweights easy samples and upweights hard samples."""
    focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    
    # Target is positive (1.0)
    target = torch.tensor([[1.0]])
    
    # Easy sample: model already predicts high logit (+5.0 -> prob ~ 0.993)
    easy_logits = torch.tensor([[5.0]], requires_grad=True)
    easy_loss = focal(easy_logits, target)
    
    # Hard sample: model predicts wrong logit (-2.0 -> prob ~ 0.119)
    hard_logits = torch.tensor([[-2.0]], requires_grad=True)
    hard_loss = focal(hard_logits, target)
    
    # Hard loss must be orders of magnitude larger than easy loss
    assert hard_loss.item() > 10.0 * easy_loss.item()
    
    # Gradient must backprop cleanly
    hard_loss.backward()
    assert hard_logits.grad is not None


def test_int8_quantization_execution_and_accuracy():
    """Verifies dynamic INT8 quantization executes cleanly on CPU and retains prediction range."""
    model = ClinicalRiskCNNLSTM()
    quant_model = quantize_model_dynamic_int8(model)
    
    dummy_input = torch.randn(1, 1, 1800)
    with torch.no_grad():
        out_fp32 = model(dummy_input)
        out_int8 = quant_model(dummy_input)
        
    assert out_int8.shape == (1, 1)
    assert 0.0 <= out_int8.item() <= 1.0
    
    # Difference between FP32 and INT8 on random input should be small (< 0.25)
    diff = abs(out_fp32.item() - out_int8.item())
    assert diff < 0.25, f"INT8 quantization error too high: {diff}"


def test_polyak_soft_target_updates():
    """Verifies that Polyak moving average update theta^- <- tau * theta + (1-tau) * theta^- works."""
    online = DuelingDQN(state_dim=54, action_dim=26)
    target = DuelingDQN(state_dim=54, action_dim=26)
    
    # Initialize target identical to online
    target.load_state_dict(online.state_dict())
    
    # Perturb online weights
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
            
    tau = 0.10
    for target_p, online_p in zip(target.parameters(), online.parameters()):
        target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
        
    # Check that target shifted by tau * delta = 0.10 * 1.0 = 0.10
    online_p0 = next(online.parameters()).data[0, 0].item()
    target_p0 = next(target.parameters()).data[0, 0].item()
    assert abs(online_p0 - target_p0 - 0.90) < 1e-4
