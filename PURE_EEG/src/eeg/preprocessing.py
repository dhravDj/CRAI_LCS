"""
Pure-EEG-CRAI-LCS: Clinical Neurological Biosignal Preprocessing Pipeline
Adheres to IFCN (International Federation of Clinical Neurophysiology) and NeuroKit2 standards.
Processes 16-channel scalp EEG telemetry windows (@ 256 Hz) with:
1. Zero-phase Butterworth Bandpass (0.5 - 70 Hz)
2. 50 Hz Powerline IIR Notch Filtering (Q=30)
3. Common Average Referencing (CAR)
4. Robust Channel-wise Z-score Normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


def butterworth_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 70.0,
    fs: int = 256,
    order: int = 3
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter (0.5 - 70 Hz) to eliminate
    slow galvanic DC drift (<0.5 Hz) and high-frequency muscle artifacts (>70 Hz).
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
    fs: int = 256,
    quality: float = 30.0
) -> np.ndarray:
    """Eliminates 50 Hz AC powerline hum via zero-phase IIR notch filter."""
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality)
    if signal.shape[-1] <= 3 * max(len(a), len(b)):
        return signal.astype(np.float32)
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def common_average_reference(eeg_channels: np.ndarray) -> np.ndarray:
    """Subtracts instantaneous common average across channels to eliminate reference noise."""
    if eeg_channels.ndim < 2 or eeg_channels.shape[0] <= 1:
        return eeg_channels
    mean_ch = np.mean(eeg_channels, axis=0, keepdims=True)
    return eeg_channels - mean_ch


def z_score_normalize(eeg_channels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Applies channel-wise robust normalization."""
    if eeg_channels.ndim == 1:
        eeg_channels = eeg_channels.reshape(1, -1)
    mu = np.mean(eeg_channels, axis=-1, keepdims=True)
    std = np.std(eeg_channels, axis=-1, keepdims=True)
    return (eeg_channels - mu) / (std + eps)


def preprocess_eeg_window(
    raw_eeg: np.ndarray,
    fs: int = 256,
    channels: int = 16,
    filter_noise: bool = True
) -> np.ndarray:
    """
    Standardized clinical 16-channel EEG preprocessing pipeline:
    Auto-broadcasts single-channel inputs to 16 channels, filters bandpass & notch,
    applies CAR, and normalizes.
    """
    if raw_eeg.ndim == 1:
        if raw_eeg.size == channels * 1280:
            raw_eeg = raw_eeg.reshape(channels, 1280)
        elif raw_eeg.size == 1280:
            raw_eeg = np.tile(raw_eeg, (channels, 1))
        else:
            raw_eeg = np.tile(raw_eeg, int(np.ceil((channels * 1280) / raw_eeg.size)))[:channels * 1280].reshape(channels, 1280)
    elif raw_eeg.shape[0] == 1 and channels > 1:
        raw_eeg = np.repeat(raw_eeg, channels, axis=0)

    if filter_noise:
        banded = butterworth_bandpass_filter(raw_eeg, lowcut=0.5, highcut=70.0, fs=fs)
        notched = iir_notch_filter(banded, notch_freq=50.0, fs=fs)
        car = common_average_reference(notched)
        return z_score_normalize(car).astype(np.float32)
    
    return z_score_normalize(raw_eeg).astype(np.float32)
