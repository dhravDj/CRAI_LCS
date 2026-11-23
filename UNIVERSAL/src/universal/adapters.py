"""
U-CRAI-LCS: Universal Biosignal Ingestion & Signal Conditioning Adapters
Implements high-speed vectorized filtering, artifact suppression, and physiological normalization.
Follows NeuroKit2 and AAMI/IFCN biomedical standards.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from src.universal.schema import ModalityType, UniversalTelemetryTask, TIME_TO_HARM_SECONDS


class BaseModalityAdapter(ABC):
    """Abstract interface for modality-specific ingestion and filtering."""
    
    @abstractmethod
    def condition_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Applies digital filtering and normalization to clean raw sensor telemetry."""
        pass

    @abstractmethod
    def estimate_demand(self, conditioned_signal: np.ndarray) -> Tuple[int, float]:
        """Returns (payload_bytes, estimated_cpu_cycles) for continuum scheduling."""
        pass


class CardiacECGAdapter(BaseModalityAdapter):
    """
    Cardiac Telemetry Adapter (1-Lead / Lead MLII @ 360 Hz).
    Bandpass: 0.5 - 45.0 Hz Butterworth 3rd-order.
    """
    def __init__(self, fs: int = 360):
        self.fs = fs
        nyq = 0.5 * fs
        self.b, self.a = butter(3, [0.5 / nyq, 45.0 / nyq], btype='band')

    def condition_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        # Zero-phase Butterworth filtering
        filtered = filtfilt(self.b, self.a, raw_signal.astype(np.float32))
        # Robust z-score normalization
        mu, std = np.mean(filtered), np.std(filtered)
        return (filtered - mu) / (std + 1e-8)

    def estimate_demand(self, conditioned_signal: np.ndarray) -> Tuple[int, float]:
        # 5s window = 1800 samples. 16-bit ADC = 3600 bytes. ~1.0e7 cycles for 1D CNN-LSTM
        payload_bytes = len(conditioned_signal) * 2
        cycles = 1.0e7
        return payload_bytes, cycles


class NeurologicalEEGAdapter(BaseModalityAdapter):
    """
    Neurological Telemetry Adapter (16-Channel Scalp EEG @ 256 Hz).
    Bandpass: 0.5 - 70.0 Hz Butterworth 3rd-order + 50 Hz IIR Notch filter.
    """
    def __init__(self, fs: int = 256, channels: int = 16):
        self.fs = fs
        self.channels = channels
        nyq = 0.5 * fs
        self.b_band, self.a_band = butter(3, [0.5 / nyq, 70.0 / nyq], btype='band')
        # 50 Hz powerline notch (Q=30)
        self.b_notch, self.a_notch = iirnotch(50.0 / nyq, 30.0)

    def condition_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        # Expected input shape: (Channels, Samples). Auto-broadcast single-channel to 16 channels.
        if raw_signal.ndim == 1:
            raw_signal = np.tile(raw_signal, (self.channels, 1))
        elif raw_signal.shape[0] == 1 and self.channels > 1:
            raw_signal = np.repeat(raw_signal, self.channels, axis=0)
        
        filtered = np.zeros_like(raw_signal, dtype=np.float32)
        for ch in range(raw_signal.shape[0]):
            ch_sig = raw_signal[ch].astype(np.float32)
            # Apply bandpass then notch
            banded = filtfilt(self.b_band, self.a_band, ch_sig)
            notched = filtfilt(self.b_notch, self.a_notch, banded)
            mu, std = np.mean(notched), np.std(notched)
            filtered[ch] = (notched - mu) / (std + 1e-8)
        return filtered

    def estimate_demand(self, conditioned_signal: np.ndarray) -> Tuple[int, float]:
        # 16 channels x 1280 samples x 2 bytes = 40,960 bytes (~41 KB). ~3.5e7 cycles
        payload_bytes = conditioned_signal.size * 2
        cycles = 3.5e7
        return payload_bytes, cycles


class RespiratoryPPGAdapter(BaseModalityAdapter):
    """
    Respiratory / Pulse Oximetry Telemetry Adapter (1-Channel PPG @ 100 Hz).
    Bandpass: 0.5 - 5.0 Hz (captures arterial pulse wave while rejecting motion).
    """
    def __init__(self, fs: int = 100):
        self.fs = fs
        nyq = 0.5 * fs
        self.b, self.a = butter(2, [0.5 / nyq, 5.0 / nyq], btype='band')

    def condition_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        filtered = filtfilt(self.b, self.a, raw_signal.astype(np.float32))
        mu, std = np.mean(filtered), np.std(filtered)
        return (filtered - mu) / (std + 1e-8)

    def estimate_demand(self, conditioned_signal: np.ndarray) -> Tuple[int, float]:
        # 5s window = 500 samples. 2 bytes = 1,000 bytes. ~5.0e6 cycles
        payload_bytes = len(conditioned_signal) * 2
        cycles = 5.0e6
        return payload_bytes, cycles


class UniversalTelemetryIngestor:
    """
    Central Ingestion Gateway for heterogeneous medical IoT sensors.
    Auto-routes incoming telemetry to the matching clinical conditioning adapter.
    """
    def __init__(self):
        self.adapters = {
            ModalityType.CARDIAC_ECG: CardiacECGAdapter(fs=360),
            ModalityType.NEURO_EEG: NeurologicalEEGAdapter(fs=256, channels=16),
            ModalityType.RESPIRATORY_PPG: RespiratoryPPGAdapter(fs=100)
        }

    def ingest(
        self,
        task_id: int,
        modality: ModalityType,
        patient_id: int,
        raw_signal: np.ndarray,
        arrival_time: float
    ) -> UniversalTelemetryTask:
        adapter = self.adapters.get(modality)
        if adapter is None:
            raise ValueError(f"Unsupported modality: {modality}. Choose from {list(self.adapters.keys())}")

        conditioned = adapter.condition_signal(raw_signal)
        payload_bytes, cycles = adapter.estimate_demand(conditioned)

        # Baseline sampling rate from adapter
        fs = getattr(adapter, "fs", 100)

        return UniversalTelemetryTask(
            task_id=task_id,
            modality=modality,
            patient_id=patient_id,
            raw_signal=conditioned,
            sampling_rate_hz=fs,
            arrival_timestamp_s=arrival_time,
            payload_bytes=payload_bytes,
            computational_demand_cycles=cycles
        )
