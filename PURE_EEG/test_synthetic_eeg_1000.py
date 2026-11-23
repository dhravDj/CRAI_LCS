#!/usr/bin/env python3
"""
Pure-EEG-CRAI-LCS: 1,000-Task Neurological EEG Continuum Stress Benchmark
Generates 1,000 randomized 16-channel clinical EEG telemetry windows (1280 samples @ 256 Hz, 40.96 KB)
with IFCN filtering, 3 Hz epileptic spike-wave bursts, CAR referencing, and fail-safe preemption.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import CRAILCSPipeline
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.utils.seed import set_global_seed


def generate_synthetic_eeg_cohort(num_samples: int = 1000, seed: int = 42):
    rng = np.random.RandomState(seed)
    signals = np.zeros((num_samples, 16, 1280), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    true_risks = np.zeros(num_samples, dtype=np.float32)
    t = np.linspace(0, 5, 1280)

    for i in range(num_samples):
        p = rng.rand()
        if p < 0.75:
            # Normal background: alpha + beta + sensor noise
            labels[i] = 0
            true_risks[i] = float(rng.uniform(0.05, 0.35))
            noise = rng.normal(0, 0.5, (16, 1280))
            alpha = np.sin(2 * np.pi * rng.uniform(8, 12) * t)
            beta = 0.5 * np.sin(2 * np.pi * rng.uniform(15, 25) * t)
            for ch in range(16):
                signals[i, ch] = (1.0 if ch in [6, 7, 8, 9] else 0.2) * alpha + (1.0 if ch in [0, 1, 2, 3] else 0.3) * beta + noise[ch]
        else:
            # Ictal Seizure / Status Epilepticus (3 Hz spike-and-wave)
            labels[i] = 1
            true_risks[i] = float(rng.uniform(0.75, 0.98))
            burst = 4.0 * np.sin(2 * np.pi * 3.0 * t) + 2.5 * np.sin(2 * np.pi * 6.0 * t)
            noise = rng.normal(0, 0.5, (16, 1280))
            for ch in range(16):
                signals[i, ch] = burst + noise[ch]

    return signals, labels, true_risks


def run_benchmark():
    print("=" * 78)
    print(" Pure-EEG-CRAI-LCS: 1,000-Task Continuum Stress Benchmark")
    print(" 16 Channels @ 256 Hz (40.96 KB Payload) | Status Epilepticus Detection")
    print("=" * 78)

    signals, labels, true_risks = generate_synthetic_eeg_cohort(1000, seed=42)
    pipeline = CRAILCSPipeline(device="cpu", quantize=True)
    env = HealthcareEdgeEnv(seed=42, arrival_rate=20.0, max_tasks=1000)
    env.reset(seed=42)

    latencies_ms = []
    ai_latencies_ms = []
    violations = 0
    fail_safe_triggers = 0
    total_energy_j = 0.0

    t_start = time.perf_counter()

    for i in range(1000):
        p_id = i % 100
        res = pipeline.execute_task_pipeline(eeg_window=signals[i], env=env, patient_id=p_id)

        latencies_ms.append(res["step_latency_ms"])
        ai_latencies_ms.append(res["total_ai_latency_ms"])
        total_energy_j += res["step_energy_j"]

        if res["fail_safe_triggered"]:
            fail_safe_triggers += 1
        if res["step_violated"]:
            violations += 1

    t_end = time.perf_counter()
    wall_clock_s = t_end - t_start
    throughput = 1000.0 / wall_clock_s

    print("\n" + "=" * 78)
    print(" BENCHMARK RESULTS (1,000 TASKS CONTINUOUS STREAM)")
    print("=" * 78)
    print(f"Total Tasks Processed:       1,000")
    print(f"Mean AI Inference Latency:   {np.mean(ai_latencies_ms):.2f} ms")
    print(f"Mean Turnaround Latency:     {np.mean(latencies_ms):.2f} ms")
    print(f"P95 Turnaround Latency:      {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"SLA Violation Rate:          {violations / 1000.0 * 100.0:.2f}% ({violations}/1000)")
    print(f"Fail-Safe Preemptions:       {fail_safe_triggers}")
    print(f"Total Continuum Energy:      {total_energy_j:.2f} J")
    print(f"Continuum Throughput:        {throughput:.1f} tasks/sec")
    print(f"Wall-Clock Benchmark Time:   {wall_clock_s:.2f} s")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
