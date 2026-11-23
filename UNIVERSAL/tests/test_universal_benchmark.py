"""
Unit test for Universal Multi-Disease Continuum Stress Benchmark.
"""

import pytest
import numpy as np
from src.universal.benchmark import UniversalStressBenchmark
from src.universal.schema import ModalityType


def test_synthetic_cohort_generation():
    bench = UniversalStressBenchmark(seed=42)
    cohort = bench.generate_synthetic_cohort(num_tasks=30, arrival_rate=10.0)
    assert len(cohort) == 30

    modalities = {c[0] for c in cohort}
    assert ModalityType.CARDIAC_ECG in modalities
    assert ModalityType.NEURO_EEG in modalities
    assert ModalityType.RESPIRATORY_PPG in modalities


def test_benchmark_execution():
    bench = UniversalStressBenchmark(seed=42)
    results = bench.run_benchmark(num_tasks=60)

    assert results["num_tasks"] == 60
    u_res = results["u_crai_lcs"]
    c_res = results["cloud_only"]
    g_res = results["greedy_edge"]

    # U-CRAI-LCS must beat Cloud-Only substantially in latency
    assert u_res["avg_latency_ms"] < c_res["avg_latency_ms"]

    # SLA violation rate on U-CRAI-LCS must be low (<= 5%)
    assert u_res["violation_rate"] <= 5.0

    # Urgent violation rate must be zero or near zero
    assert u_res["urgent_violation_rate"] <= 2.0

    # Battery extension factor must be > 1.5x due to asymmetric gating
    assert u_res["battery_extension_factor"] >= 1.5
