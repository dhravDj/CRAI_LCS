#!/usr/bin/env python3
"""
CRAI-LCS: Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling
Nature Scientific Reports (2026) - Master Production Entry Point
"""

import os
import sys
import argparse
import time
import numpy as np

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.seed import set_global_seed
from src.pipeline import CRAILCSPipeline
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.experiments.load_sweep import execute_peak_load_benchmarks, execute_load_sweep
from src.experiments.statistical_tests import run_wilcoxon_statistical_analysis
from src.experiments.ablation_study import execute_ablation_study
from src.experiments.generalization_test import evaluate_cross_dataset_generalization
from src.experiments.plot_figures import generate_all_publication_figures


def run_interactive_demo(num_tasks: int = 20, seed: int = 42, quantize: bool = False):
    """Executes a live terminal demo visualizing per-task risk, deadline, and dispatch destination."""
    set_global_seed(seed)
    print("\n" + "=" * 78)
    print(" CRAI-LCS: LIVE CLINICAL TELEMETRY SCHEDULING DEMO")
    print(" Closed-Loop Multi-Model Continuum Orchestration (Nature Scientific Reports 2026)")
    if quantize:
        print(" [OPTIMIZATION: INT8 Dynamic Quantization + Butterworth Filter Active]")
    print("=" * 78)

    pipeline = CRAILCSPipeline(device="cpu", quantize=quantize, seed=seed)
    env = HealthcareEdgeEnv(seed=seed, arrival_rate=20.0, max_tasks=num_tasks)
    obs, _ = env.reset(seed=seed)

    print(f"\n{'Step':<5} | {'Risk Ri':<9} | {'Deadline':<10} | {'P_miss':<8} | {'Dispatch Target':<22} | {'Latency':<9} | {'Status'}")
    print("-" * 78)

    for i in range(num_tasks):
        dummy_emg = np.random.randn(4, 2000).astype(np.float32)
        step_res = pipeline.execute_task_pipeline(emg_window=dummy_emg, env=env, patient_id=i)

        risk_val = step_res["clinical_risk"]
        deadline_ms = env.tasks[i].deadline_ms
        p_miss = step_res["predicted_miss_prob"]
        act = step_res["action"]
        lat = step_res["step_latency_ms"]
        violated = step_res["step_violated"]
        fail_safe = step_res["fail_safe_triggered"]

        if act == 25:
            target_str = "Central Cloud (WAN)"
        else:
            target_str = f"Edge Gateway {act:02d}"

        if fail_safe:
            target_str += " [FAIL-SAFE]"

        status_str = "VIOLATED" if violated else "SUCCESS"
        risk_tag = f"{risk_val:.3f}" + (" *" if risk_val >= 0.50 else "  ")

        print(f"#{i+1:03d}  | {risk_tag:<9} | {deadline_ms:6.1f} ms  | {p_miss:6.3f} | {target_str:<22} | {lat:6.1f} ms | {status_str}")
        time.sleep(0.04)

    info = env._get_info()
    print("-" * 78)
    print(f"Summary: {num_tasks} Tasks | Avg Latency: {info['avg_latency_ms']:.2f} ms | Violations: {info['violation_rate']:.2f}% | Risk Coverage: {info['risk_coverage']:.1f}%")
    print(f"Total AI Inference Overhead: {step_res['total_ai_latency_ms']:.2f} ms (Budget: <= 24.1 ms)")
    print("=" * 78 + "\n")


def run_test_suite():
    """Runs full automated pytest suite across all modules."""
    import pytest
    print("\nExecuting Complete Automated QA Test Pyramid (Phases 0-10)...")
    exit_code = pytest.main(["-v", "tests/"])
    if exit_code == 0:
        print("\nAll 50+ Test Pyramid Assertions PASSED successfully (100% Green).")
    else:
        print(f"\nWarning: Test suite exited with code {exit_code}")
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="CRAI-LCS: Clinical Risk-Aware IoT Scheduling Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Operating Modes:
  demo        Live terminal streaming visualization (20 tasks)
  stream      100-patient continuous closed-loop simulation (1,000 tasks)
  benchmark   Peak-load (lambda=20) and arrival sweeps across lambda in [8, 25]
  stats       Wilcoxon signed-rank statistical hypothesis test (p < 0.05)
  ablation    4-variant architectural ablation study
  transfer    PTB-XL cross-domain generalization benchmark
  plot        Generate high-resolution 300 DPI publication figures
  test        Execute complete pytest test suite (50+ tests)
  all         Run benchmark, stats, ablation, transfer, and render publication figures
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "stream", "benchmark", "stats", "ablation", "transfer", "plot", "test", "all"],
        help="Execution mode (default: demo)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--tasks", type=int, default=1000, help="Number of tasks for simulation")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic INT8 quantization for ultra-fast edge execution")

    args = parser.parse_args()

    if args.mode == "demo":
        run_interactive_demo(num_tasks=20, seed=args.seed, quantize=args.quantize)

    elif args.mode == "stream":
        mode_str = "INT8 Quantized" if args.quantize else args.device
        print(f"\nRunning 100-Patient Streaming Closed-Loop Simulation ({mode_str})...")
        pipeline = CRAILCSPipeline(device=args.device, quantize=args.quantize, seed=args.seed)
        res = pipeline.run_streaming_simulation(num_patients=100, tasks_per_patient=10, seed=args.seed)
        print("\nStreaming Run Metrics:")
        for k, v in res.items():
            print(f"  {k:<30}: {v}")

    elif args.mode == "benchmark":
        pipeline = CRAILCSPipeline(device=args.device, seed=args.seed)
        execute_peak_load_benchmarks(pipeline, num_tasks=args.tasks)
        execute_load_sweep(pipeline, num_tasks=args.tasks)

    elif args.mode == "stats":
        run_wilcoxon_statistical_analysis()

    elif args.mode == "ablation":
        pipeline = CRAILCSPipeline(device=args.device, seed=args.seed)
        execute_ablation_study(pipeline, num_tasks=args.tasks)

    elif args.mode == "transfer":
        evaluate_cross_dataset_generalization(device=args.device, seed=args.seed)

    elif args.mode == "plot":
        generate_all_publication_figures()

    elif args.mode == "test":
        run_test_suite()

    elif args.mode == "all":
        print("\nExecuting End-to-End Scientific Validation Pipeline...")
        pipeline = CRAILCSPipeline(device=args.device, seed=args.seed)
        execute_peak_load_benchmarks(pipeline, num_tasks=args.tasks)
        execute_load_sweep(pipeline, num_tasks=args.tasks)
        run_wilcoxon_statistical_analysis()
        execute_ablation_study(pipeline, num_tasks=args.tasks)
        evaluate_cross_dataset_generalization(device=args.device, seed=args.seed)
        generate_all_publication_figures()
        print("\nComplete scientific validation pipeline successfully executed!")


if __name__ == "__main__":
    main()
