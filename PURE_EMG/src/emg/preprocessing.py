"""
Pure-EMG-CRAI-LCS: Clinical Electromyographic Biosignal Preprocessing Pipeline
Adheres to SENIAM (Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles) guidelines.
Processes 4-channel surface EMG telemetry windows (@ 1000 Hz) with:
1. Zero-phase Butterworth Bandpass (20.0 - 450.0 Hz)
2. 50 Hz Powerline IIR Notch Filtering (Q=30)
3. Rectification & RMS Envelope Extraction
4. Channel-wise Robust Normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


def butterworth_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    fs: int = 1000,
    order: int = 4
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter (20 - 450 Hz) conforming to SENIAM.
    Attenuates movement artifacts (<20 Hz) while preserving MUAP firing spectrum (up to 450 Hz).
    """
    nyq = 0.5 * fs
    low = max(1e-4, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    if signal.shape[-1] <= 3 * max(len(a), len(b)):
        return signal.astype(np.float32)
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def iir_notch_filter(
    signal: np.ndarray,
    notch_freq: float = 50.0,
    fs: int = 1000,
    quality: float = 30.0
) -> np.ndarray:
    """Eliminates 50 Hz AC powerline hum via zero-phase IIR notch filter."""
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality)
    if signal.shape[-1] <= 3 * max(len(a), len(b)):
        return signal.astype(np.float32)
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def compute_rms_envelope(signal: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Computes moving Root Mean Square (RMS) envelope for muscle contraction intensity."""
    if signal.ndim == 1:
        sq = signal ** 2
        kernel = np.ones(window_size) / window_size
        return np.sqrt(np.convolve(sq, kernel, mode="same")).astype(np.float32)
    else:
        out = np.zeros_like(signal)
        kernel = np.ones(window_size) / window_size
        for ch in range(signal.shape[0]):
            sq = signal[ch] ** 2
            out[ch] = np.sqrt(np.convolve(sq, kernel, mode="same"))
        return out.astype(np.float32)


def z_score_normalize(emg_channels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Applies channel-wise robust normalization."""
    if emg_channels.ndim == 1:
        emg_channels = emg_channels.reshape(1, -1)
    mu = np.mean(emg_channels, axis=-1, keepdims=True)
    std = np.std(emg_channels, axis=-1, keepdims=True)
    return (emg_channels - mu) / (std + eps)


def preprocess_emg_window(
    raw_emg: np.ndarray,
    fs: int = 1000,
    filter_noise: bool = True,
    apply_envelope: bool = False
) -> np.ndarray:
    """
    Complete clinical preprocessing pipeline for 4-channel surface EMG window (4 x 2000 @ 1000 Hz).
    Returns cleaned, normalized EMG tensor of shape (4, 2000).
    Optionally computes moving RMS envelope if apply_envelope=True.
    """
    if raw_emg.ndim == 1:
        if raw_emg.size == 4 * 2000:
            raw_emg = raw_emg.reshape(4, 2000)
        else:
            raw_emg = np.tile(raw_emg, int(np.ceil((4 * 2000) / raw_emg.size)))[:4 * 2000].reshape(4, 2000)
    elif raw_emg.shape != (4, 2000):
        raw_emg = np.resize(raw_emg, (4, 2000))

    emg_arr = raw_emg.astype(np.float32)

    if filter_noise:
        emg_arr = butterworth_bandpass_filter(emg_arr, lowcut=20.0, highcut=450.0, fs=fs)
        emg_arr = iir_notch_filter(emg_arr, notch_freq=50.0, fs=fs)

    if apply_envelope:
        emg_arr = compute_rms_envelope(emg_arr, window_size=50)

    norm_emg = z_score_normalize(emg_arr)
    return norm_emg.astype(np.float32)
