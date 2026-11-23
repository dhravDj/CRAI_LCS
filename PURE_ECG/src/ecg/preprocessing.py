import os
from typing import Tuple, List, Dict, Optional, Set
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from src.simulator.task import HealthcareTask

# AAMI EC57 Ventricular Ectopy / Emergency arrhythmia annotation symbols
URGENT_SYMBOLS: Set[str] = {'V', 'E', '!', '[', ']', 'r'}

# Default zero-leakage patient-record partitions (FR-004)
DEFAULT_TRAIN_RECORDS: List[str] = ['100', '101', '102', '103', '200', '201']
DEFAULT_VAL_RECORDS: List[str] = ['219', '220']
DEFAULT_TEST_RECORDS: List[str] = ['230', '231', '233', '234']


def butterworth_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: int = 360,
    order: int = 3
) -> np.ndarray:
    """
    Applies zero-phase Butterworth bandpass filter (0.5 - 45 Hz) to eliminate
    respiratory baseline drift (<0.5 Hz) and 50/60 Hz powerline AC hum (>45 Hz).
    """
    nyq = 0.5 * fs
    low = max(1e-4, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    if signal.shape[-1] <= 3 * max(len(a), len(b)):
        return signal.astype(np.float32)
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def z_score_normalize(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Applies per-segment Z-score normalization (Equation 17):
    x_norm = (x - mean) / (std + eps).
    Eliminates respiration baseline drift and electrode gain discrepancies.
    """
    mean = np.mean(segment)
    std = np.std(segment)
    return (segment - mean) / (std + eps)


def preprocess_ecg_window(
    raw_segment: np.ndarray,
    filter_noise: bool = True,
    fs: int = 360,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Standardized clinical preprocessing:
    1. Zero-phase Butterworth Bandpass filtering (0.5 - 45 Hz)
    2. Per-segment Z-score normalization.
    """
    if filter_noise:
        cleaned = butterworth_bandpass_filter(raw_segment, lowcut=0.5, highcut=45.0, fs=fs)
    else:
        cleaned = raw_segment
    return z_score_normalize(cleaned, eps=eps)


def extract_record_segments(
    record_path: str,
    fs: int = 360,
    window_sec: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts non-overlapping 5-second windows (1800 samples) from an MIT-BIH record.
    Returns:
        segments: (N, 1800) float32 normalized signals
        labels: (N,) int64 binary labels (1: Urgent/Ventricular, 0: Routine)
        risks: (N,) float32 clinical risk scores R_i in [0.0, 1.0]
    """
    signals, fields = wfdb.rdsamp(record_path)
    ann = wfdb.rdann(record_path, 'atr')

    # Select Lead MLII (channel 0 or matching name)
    lead_idx = 0
    if 'sig_name' in fields:
        for idx, name in enumerate(fields['sig_name']):
            if 'MLII' in name.upper() or 'II' in name.upper():
                lead_idx = idx
                break

    lead_signal = signals[:, lead_idx].astype(np.float32)
    seg_len = fs * window_sec  # 1800 samples

    total_samples = len(lead_signal)
    num_segments = total_samples // seg_len

    ann_samples = np.array(ann.sample)
    ann_symbols = np.array(ann.symbol)

    segments = []
    labels = []
    risks = []

    for i in range(num_segments):
        start_idx = i * seg_len
        end_idx = start_idx + seg_len

        raw_segment = lead_signal[start_idx:end_idx]
        norm_segment = z_score_normalize(raw_segment)

        # Find beat annotations falling inside this 5-second window
        mask = (ann_samples >= start_idx) & (ann_samples < end_idx)
        seg_symbols = ann_symbols[mask]

        # Calculate clinical urgency based on ventricular ectopic beats
        urgent_beats = [s for s in seg_symbols if s in URGENT_SYMBOLS]
        num_urgent = len(urgent_beats)
        is_urgent = num_urgent > 0

        if is_urgent:
            label = 1
            # Severity scales with number of ventricular ectopic events in 5s window
            risk = min(0.99, 0.50 + 0.10 * num_urgent)
        else:
            label = 0
            # Routine telemetry risk [0.05, 0.45]
            risk = min(0.45, 0.05 + 0.02 * len(seg_symbols))

        segments.append(norm_segment)
        labels.append(label)
        risks.append(float(risk))

    return (
        np.array(segments, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(risks, dtype=np.float32)
    )


def build_mitbih_dataset(
    raw_dir: str,
    output_path: str,
    train_records: Optional[List[str]] = None,
    val_records: Optional[List[str]] = None,
    test_records: Optional[List[str]] = None,
    fs: int = 360,
    window_sec: int = 5
) -> Dict[str, np.ndarray]:
    """
    Processes and partitions MIT-BIH records with strict ZERO PATIENT DATA LEAKAGE (FR-004).
    Saves a compressed .npz archive containing train, val, and test partitions.
    """
    train_recs = train_records or DEFAULT_TRAIN_RECORDS
    val_recs = val_records or DEFAULT_VAL_RECORDS
    test_recs = test_records or DEFAULT_TEST_RECORDS

    # Hard zero-leakage assertion: Record sets must be completely disjoint
    assert len(set(train_recs) & set(val_recs)) == 0, "Patient leakage detected between train and val!"
    assert len(set(train_recs) & set(test_recs)) == 0, "Patient leakage detected between train and test!"
    assert len(set(val_recs) & set(test_recs)) == 0, "Patient leakage detected between val and test!"

    def process_records(record_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_x, all_y, all_r = [], [], []
        for r in record_list:
            rec_path = os.path.join(raw_dir, str(r))
            if os.path.exists(f"{rec_path}.dat"):
                x, y, r_scores = extract_record_segments(rec_path, fs=fs, window_sec=window_sec)
                all_x.append(x)
                all_y.append(y)
                all_r.append(r_scores)
            else:
                print(f"Warning: Record {r} not found at {rec_path}.dat, skipping.")

        if not all_x:
            return np.empty((0, fs * window_sec), dtype=np.float32), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        return (
            np.concatenate(all_x, axis=0),
            np.concatenate(all_y, axis=0),
            np.concatenate(all_r, axis=0)
        )

    print("Processing Training records...")
    train_x, train_y, train_r = process_records(train_recs)

    print("Processing Validation records...")
    val_x, val_y, val_r = process_records(val_recs)

    print("Processing Testing records...")
    test_x, test_y, test_r = process_records(test_recs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = {
        "train_x": train_x, "train_y": train_y, "train_r": train_r,
        "val_x": val_x, "val_y": val_y, "val_r": val_r,
        "test_x": test_x, "test_y": test_y, "test_r": test_r,
        "train_records": np.array(train_recs),
        "val_records": np.array(val_recs),
        "test_records": np.array(test_recs)
    }

    np.savez_compressed(output_path, **dataset)
    print(f"Dataset successfully created and saved to {output_path}!")
    print(f"Train: {train_x.shape[0]} samples | Val: {val_x.shape[0]} samples | Test: {test_x.shape[0]} samples")
    return dataset


def create_ecg_healthcare_task(
    task_id: int,
    segment: np.ndarray,
    clinical_risk: float,
    arrival_time: float,
    rng: Optional[np.random.RandomState] = None
) -> HealthcareTask:
    """
    Converts a real 1800-sample ECG segment into a HealthcareTask instance
    strictly conforming to Equations (1)-(2) and Table 3.
    """
    if rng is None:
        rng = np.random.RandomState(task_id)

    # Payload size: exactly 1800 float16/int16 samples = 3600.0 Bytes
    delta_i = 3600.0

    # Computational demand psi_i in [5e6, 2e7] cycles
    psi_i = float(rng.uniform(5.0e6, 2.0e7))

    # Deadlines according to clinical urgency R_i
    if clinical_risk >= 0.50:
        # Urgent cardiac condition: 100 - 150 ms
        deadline_ms = float(rng.uniform(100.0, 150.0))
    else:
        # Routine telemetry: 400 - 800 ms
        deadline_ms = float(rng.uniform(400.0, 800.0))

    return HealthcareTask(
        task_id=task_id,
        delta_i=delta_i,
        psi_i=psi_i,
        deadline_ms=deadline_ms,
        clinical_risk=float(clinical_risk),
        arrival_time=arrival_time,
        predicted_miss_prob=0.0
    )
