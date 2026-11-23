#!/usr/bin/env python3
"""
CRAI-LCS: 2,000-Sample Synthetic ECG Telemetry Generation, Labeling & Continuum Stress Benchmark
Generates 2,000 randomized clinical ECG telemetry windows (1800 samples @ 360 Hz)
with realistic physiological wave morphology, cardiac arrhythmias, and physical noise.
Tests end-to-end AI triage, deadline forecasting, DQN continuum scheduling, and fail-safe preemption.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import CRAILCSPipeline
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.simulator.task import HealthcareTask
from src.utils.seed import set_global_seed


def generate_synthetic_2000_cohort(
    num_samples: int = 2000,
    segment_len: int = 1800,
    fs: int = 360,
    arrhythmia_ratio: float = 0.25,
    seed: int = 42,
    output_npz: str = "data/synthetic_2000_dataset.npz"
):
    """
    Synthesizes 2,000 clinical 5-second ECG windows with:
    - Normal Sinus Rhythm (~75%, Label 0)
    - Acute Ventricular Arrhythmias / Ectopy (~25%, Label 1)
    - Realistic physical artifacts: 50 Hz AC hum, respiration baseline wander, sensor noise.
    """
    rng = np.random.RandomState(seed)
    signals = []
    labels = []
    true_risks = []
    deadlines_ms = []
    patient_ids = []

    t = np.linspace(0, 5, segment_len)

    print(f"\nGenerating {num_samples} synthetic physiological ECG telemetry windows...")
    t0 = time.time()

    num_patients = 100
    for i in range(num_samples):
        p_id = i % num_patients
        patient_ids.append(p_id)

        is_arrhythmia = rng.rand() < arrhythmia_ratio
        heart_rate_hz = rng.uniform(1.0, 1.4)  # 60 - 84 bpm

        # 1. Base cardiac wave (P-wave, QRS, T-wave representation)
        cardiac_cycle = (t * heart_rate_hz) % 1.0
        # P-wave
        p_wave = 0.15 * np.exp(-((cardiac_cycle - 0.20)**2) / 0.002)
        # QRS spike
        q_wave = -0.15 * np.exp(-((cardiac_cycle - 0.38)**2) / 0.0005)
        r_wave = 1.0 * np.exp(-((cardiac_cycle - 0.40)**2) / 0.0004)
        s_wave = -0.25 * np.exp(-((cardiac_cycle - 0.42)**2) / 0.0005)
        # T-wave
        t_wave = 0.30 * np.exp(-((cardiac_cycle - 0.65)**2) / 0.008)

        clean_ecg = p_wave + q_wave + r_wave + s_wave + t_wave

        if is_arrhythmia:
            labels.append(1)
            # Ventricular Ectopic / Polymorphic Tachycardia deformation:
            # Wide bizarre QRS complex, inverted deep T-wave, high-frequency flutter
            ventricular_burst = (
                1.4 * np.sin(2 * np.pi * heart_rate_hz * 3.5 * t) * np.exp(-((cardiac_cycle - 0.42)**2) / 0.015)
                - 0.6 * np.exp(-((cardiac_cycle - 0.60)**2) / 0.02)
            )
            raw_signal = clean_ecg + ventricular_burst
            risk = float(rng.uniform(0.55, 0.99))
            deadline = float(rng.uniform(100.0, 150.0))
        else:
            labels.append(0)
            raw_signal = clean_ecg
            risk = float(rng.uniform(0.01, 0.45))
            deadline = float(rng.uniform(400.0, 800.0))

        # 2. Add realistic physical telemetry noise:
        # Baseline wander (< 0.5 Hz breathing)
        baseline_wander = float(rng.uniform(0.1, 0.3)) * np.sin(2 * np.pi * float(rng.uniform(0.15, 0.3)) * t)
        # 50 Hz powerline interference
        powerline_hum = 0.05 * np.sin(2 * np.pi * 50.0 * t)
        # Random motion/muscle noise
        sensor_noise = rng.normal(0, 0.05, segment_len)

        noisy_ecg = raw_signal + baseline_wander + powerline_hum + sensor_noise
        signals.append(noisy_ecg.astype(np.float32))
        true_risks.append(risk)
        deadlines_ms.append(deadline)

    signals = np.array(signals, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    true_risks = np.array(true_risks, dtype=np.float32)
    deadlines_ms = np.array(deadlines_ms, dtype=np.float32)
    patient_ids = np.array(patient_ids, dtype=np.int64)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(
        output_npz,
        signals=signals,
        labels=labels,
        true_risks=true_risks,
        deadlines_ms=deadlines_ms,
        patient_ids=patient_ids
    )

    elapsed = time.time() - t0
    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    print(f"Generated {num_samples} samples in {elapsed:.2f}s | Saved to: {output_npz}")
    print(f"Label Distribution: Normal Sinus (0) = {neg_count} ({neg_count/num_samples*100:.1f}%) | Arrhythmia (1) = {pos_count} ({pos_count/num_samples*100:.1f}%)\n")

    return signals, labels, true_risks, deadlines_ms, patient_ids


def run_synthetic_cohort_benchmark(
    num_samples: int = 2000,
    quantize: bool = False,
    arrival_rate: float = 20.0,
    seed: int = 42
):
    """
    Executes full closed-loop stress evaluation on 2,000 synthetic clinical tasks.
    """
    set_global_seed(seed)
    signals, labels, true_risks, deadlines_ms, patient_ids = generate_synthetic_2000_cohort(
        num_samples=num_samples,
        seed=seed
    )

    print("=" * 86)
    print(f"🏥 CRAI-LCS: 2,000-TASK CLINICAL CONTINUUM STRESS TEST")
    print(f" Workload: {num_samples} Telemetry Windows | Arrival Rate: lambda = {arrival_rate} tasks/s")
    print(f" Hardware Mode: {'Dynamic INT8 Quantized' if quantize else 'FP32 Standard'}")
    print("=" * 86)

    pipeline = CRAILCSPipeline(device="cpu", quantize=quantize, seed=seed)
    env = HealthcareEdgeEnv(seed=seed, arrival_rate=arrival_rate, max_tasks=num_samples)
    env.reset(seed=seed)

    predicted_risks = []
    latencies = []
    ai_latencies = []
    violations = 0
    fail_safe_count = 0
    actions = []

    header = f"{'Sample':<7} | {'Ground Truth':<14} | {'AI Risk (Ri)':<13} | {'Deadline':<10} | {'Dispatch Target':<22} | {'Latency':<9} | {'Status'}"
    print(header)
    print("-" * 86)

    t0_eval = time.time()

    for i in range(num_samples):
        raw_win = signals[i]
        true_label = labels[i]
        gt_str = "🚨 ARRHYTHMIA" if true_label == 1 else "🟢 Normal Sinus"

        step_res = pipeline.execute_task_pipeline(
            ecg_window=raw_win,
            env=env,
            patient_id=int(patient_ids[i])
        )

        p_risk = step_res["clinical_risk"]
        predicted_risks.append(p_risk)
        lat = step_res["step_latency_ms"]
        latencies.append(lat)
        ai_latencies.append(step_res["total_ai_latency_ms"])
        act = step_res["action"]
        actions.append(act)

        if step_res["step_violated"]:
            violations += 1

        if step_res["fail_safe_triggered"]:
            fail_safe_count += 1

        deadline = env.tasks[i].deadline_ms
        target_str = "Central Cloud (WAN)" if act == 25 else f"Edge Gateway {act:02d}"
        if step_res["fail_safe_triggered"]:
            target_str += " [FAIL-SAFE]"

        status_str = "VIOLATED" if step_res["step_violated"] else "SUCCESS"
        risk_tag = f"{p_risk:.3f}" + (" *" if p_risk >= 0.50 else "  ")

        # Print first 15 tasks and last 5 tasks
        if i < 15 or i >= num_samples - 5:
            print(f"#{i+1:04d}  | {gt_str:<14} | {risk_tag:<13} | {deadline:6.1f} ms  | {target_str:<22} | {lat:6.1f} ms | {status_str}")
        elif i == 15:
            print(f"... ({num_samples - 20} tasks streaming and scheduling across 25 Edge Nodes in background) ...")

    total_eval_time = time.time() - t0_eval
    predicted_risks = np.array(predicted_risks)
    binary_preds = (predicted_risks >= 0.50).astype(int)

    # Calculate Machine Learning Performance
    acc = accuracy_score(labels, binary_preds) * 100.0
    f1 = f1_score(labels, binary_preds, zero_division=0) * 100.0
    prec = precision_score(labels, binary_preds, zero_division=0) * 100.0
    rec = recall_score(labels, binary_preds, zero_division=0) * 100.0
    auc = roc_auc_score(labels, predicted_risks) * 100.0

    cm = confusion_matrix(labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    spec = (tn / max(1, tn + fp)) * 100.0

    # Calculate Systems & Continuum Performance
    urgent_indices = np.where(labels == 1)[0]
    urgent_total = len(urgent_indices)
    urgent_on_time = sum(1 for idx in urgent_indices if not env.tasks[idx].is_violated)
    urgent_coverage = (urgent_on_time / max(1, urgent_total)) * 100.0
    violation_rate = (violations / num_samples) * 100.0

    env_info = env._get_info()

    print("-" * 86)
    print("\n" + "=" * 86)
    print(" 📊 2,000-SAMPLE SYNTHETIC BENCHMARK EVALUATION REPORT")
    print("=" * 86)
    print(" 🧠 Machine Learning & Diagnostic Metrics:")
    print(f"   • Clinical Accuracy             : {acc:.2f}%")
    print(f"   • Clinical F1-Score             : {f1:.2f}%")
    print(f"   • Precision (PPV)               : {prec:.2f}%")
    print(f"   • Recall / Sensitivity (TPR)    : {rec:.2f}% ({tp}/{tp+fn} arrhythmias detected)")
    print(f"   • Specificity (TNR)             : {spec:.2f}% ({tn}/{tn+fp} normal rhythms preserved)")
    print(f"   • Area Under ROC Curve (AUC)    : {auc:.2f}%")
    print(f"   • Confusion Matrix [TN, FP]     : [{tn}, {fp}]")
    print(f"                      [FN, TP]     : [{fn}, {tp}]")

    print("\n ⚡ Edge-Cloud Continuum & Systems Performance:")
    print(f"   • Total Tasks Evaluated         : {num_samples:,}")
    print(f"   • SLA Deadline Violation Rate   : {violation_rate:.2f}% ({violations}/{num_samples})")
    print(f"   • Urgent Risk Coverage          : {urgent_coverage:.2f}% ({urgent_on_time}/{urgent_total} on-time)")
    print(f"   • Clinical Fail-Safe Triggers   : {fail_safe_count} emergency local preemptions (< 20 ms)")
    print(f"   • Average End-to-End Latency    : {np.mean(latencies):.2f} ms")
    print(f"   • 95th Percentile (P95) Latency : {np.percentile(latencies, 95):.2f} ms")
    print(f"   • Average AI Decision Overhead  : {np.mean(ai_latencies):.2f} ms (Paper budget: <= 24.1 ms)")
    print(f"   • Total Continuum Energy        : {env_info['total_energy_j']:.2f} Joules")
    print(f"   • Total Evaluation Wall Time    : {total_eval_time:.2f} seconds ({num_samples/total_eval_time:.1f} tasks/sec)")
    print("=" * 86 + "\n")

    return {
        "num_samples": num_samples,
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc,
        "sensitivity": rec,
        "specificity": spec,
        "violation_rate": violation_rate,
        "urgent_coverage": urgent_coverage,
        "fail_safe_count": fail_safe_count,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "avg_ai_latency_ms": float(np.mean(ai_latencies)),
        "total_time_sec": total_eval_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2,000-Sample Synthetic ECG Telemetry Test")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples (default: 2000)")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic INT8 quantization")
    parser.add_argument("--lambda-val", type=float, default=20.0, help="Task arrival rate (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_synthetic_cohort_benchmark(
        num_samples=args.samples,
        quantize=args.quantize,
        arrival_rate=args.lambda_val,
        seed=args.seed
    )
