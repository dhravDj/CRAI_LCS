import os
import subprocess
import pytest
import torch

CHECKPOINTS = [
    "checkpoints/cnn_lstm_risk.pt",
    "checkpoints/deadline_predictor.pt",
    "checkpoints/dqn_scheduler.pt"
]

RESULTS_CSVS = [
    "results/benchmarks_lambda20.csv",
    "results/load_sweep_results.csv",
    "results/statistical_wilcoxon_results.csv",
    "results/ablation_results.csv",
    "results/ptb_xl_generalization.csv"
]

PUBLICATION_FIGURES = [
    "results/graphs/figure3_latency_sweep.png",
    "results/graphs/figure4_violations_sweep.png",
    "results/graphs/figure5_risk_coverage_sweep.png",
    "results/graphs/figure6_energy_sweep.png",
    "results/graphs/figure8_ablation_comparison.png"
]


def test_production_checkpoints_exist_and_loadable():
    """Verifies that all three pre-trained neural network checkpoints exist and are valid PyTorch models."""
    for ckpt_path in CHECKPOINTS:
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt is not None, f"Failed to load checkpoint: {ckpt_path}"


def test_production_result_datasets_exist():
    """Verifies that all experimental benchmark and statistical CSV datasets are generated."""
    for csv_path in RESULTS_CSVS:
        assert os.path.exists(csv_path), f"Result CSV missing: {csv_path}"
        assert os.path.getsize(csv_path) > 50, f"CSV {csv_path} is unexpectedly small or empty!"


def test_publication_figures_exist():
    """Verifies that all 300 DPI publication plots exist and are non-empty PNG images."""
    for fig_path in PUBLICATION_FIGURES:
        assert os.path.exists(fig_path), f"Publication figure missing: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Figure {fig_path} is unexpectedly small or corrupt!"


def test_unified_cli_demo_mode_execution():
    """Verifies that main.py --mode demo runs cleanly with exit code 0."""
    res = subprocess.run(
        ["python3", "main.py", "--mode", "demo"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert res.returncode == 0, f"CLI demo mode failed with error:\n{res.stderr}"
    assert "LIVE CLINICAL TELEMETRY SCHEDULING DEMO" in res.stdout
    assert "Summary:" in res.stdout
