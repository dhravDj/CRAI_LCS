import os
import pytest
import pandas as pd
import numpy as np

from src.pipeline import CRAILCSPipeline
from src.experiments.load_sweep import (
    run_single_simulation,
    execute_peak_load_benchmarks,
    PipelineSchedulerAdapter
)
from src.baselines.heuristics import FIFOScheduler

def test_single_simulation_metrics_contract():
    """Verifies that run_single_simulation produces valid finite bounded metrics."""
    pipeline = CRAILCSPipeline(device="cpu")
    sched = PipelineSchedulerAdapter(pipeline)

    res = run_single_simulation(sched, seed=42, arrival_rate=20.0, num_tasks=200)

    for k in ["avg_latency_ms", "violation_rate_pct", "risk_coverage_pct", "edge_utilization_pct", "energy_joules"]:
        assert k in res, f"Missing metric key {k}"
        assert np.isfinite(res[k]), f"Non-finite value for metric {k}: {res[k]}"

    assert 0.0 <= res["violation_rate_pct"] <= 100.0
    assert 0.0 <= res["risk_coverage_pct"] <= 100.0
    assert 0.0 <= res["edge_utilization_pct"] <= 100.0


def test_peak_load_benchmark_csv_integrity():
    """
    Verifies that results/benchmarks_lambda20.csv contains all evaluated algorithms
    with expected columns and valid values.
    """
    csv_path = "results/benchmarks_lambda20.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"

    df = pd.read_csv(csv_path)
    assert not df.empty, "Benchmark CSV is empty!"
    assert "scheduler" in df.columns
    assert "seed" in df.columns
    assert "avg_latency_ms" in df.columns
    assert "violation_rate_pct" in df.columns

    schedulers = df["scheduler"].unique()
    assert "CRAI-LCS (Proposed)" in schedulers
    assert "EADRL" in schedulers
    assert "Risk-Priority Heuristic" in schedulers

    # Verify CRAI-LCS achieves superior latency vs static baseline
    crai_lat = df[df["scheduler"] == "CRAI-LCS (Proposed)"]["avg_latency_ms"].mean()
    static_lat = df[df["scheduler"] == "Risk-Priority Heuristic"]["avg_latency_ms"].mean()
    assert crai_lat < static_lat, f"CRAI-LCS latency ({crai_lat:.1f}) should be lower than Static Priority ({static_lat:.1f})!"


def test_load_sweep_results_csv_integrity():
    """
    Verifies that results/load_sweep_results.csv contains sweeps across all arrival rates.
    """
    csv_path = "results/load_sweep_results.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "arrival_rate" in df.columns

    rates = sorted(df["arrival_rate"].unique())
    assert len(rates) >= 5, f"Expected at least 5 arrival rates, got {rates}"
    assert 8.0 in rates and 20.0 in rates and 25.0 in rates
