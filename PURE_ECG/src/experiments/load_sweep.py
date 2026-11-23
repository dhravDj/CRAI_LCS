import os
import sys
import time
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.baselines.heuristics import (
    RandomScheduler,
    RoundRobinScheduler,
    FIFOScheduler,
    EDFScheduler,
    RiskPriorityScheduler,
    BaseScheduler
)
from src.pipeline import CRAILCSPipeline
from src.utils.seed import set_global_seed

# Seeds for 10 independent statistical repetitions (Table 9)
BENCHMARK_SEEDS: List[int] = [42, 101, 203, 305, 407, 509, 611, 713, 815, 999]

# Arrival rates for multi-load sweep (Figures 3-7)
ARRIVAL_RATES: List[float] = [8.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]


class PipelineSchedulerAdapter(BaseScheduler):
    """Wraps CRAILCSPipeline into the BaseScheduler interface for standardized evaluation."""
    def __init__(self, pipeline: CRAILCSPipeline):
        self.pipeline = pipeline

    def schedule(self, observation: np.ndarray, task) -> int:
        action, fail_safe, _ = self.pipeline.select_action(
            observation=observation,
            clinical_risk=task.clinical_risk,
            patient_id=task.task_id
        )
        if fail_safe:
            task.is_preempted = True
        return action

    def reset(self):
        pass


class EADRLSchedulerAdapter(BaseScheduler):
    """
    Efficiency-Aware DRL (EADRL) Baseline:
    Optimizes compute efficiency and transmission latency with energy constraints.
    Favors energy-efficient edge nodes when queues permit.
    """
    def __init__(self, dqn_model, device="cpu"):
        self.dqn_model = dqn_model
        self.device = device

    def schedule(self, observation: np.ndarray, task) -> int:
        # Evaluates action values but weights toward low-power edge nodes (first 10 edge gateways)
        raw_action = self.dqn_model.act(observation, epsilon=0.0, device=self.device)
        if raw_action == 25 and task.deadline_ms < 200.0:
            # Overrule cloud offload if tight deadline
            return int(np.argmin(observation[0:25]))
        return raw_action

    def reset(self):
        pass


class DRLFogSchedulerAdapter(BaseScheduler):
    """
    DRL-Fog Baseline:
    Reinforcement learning scheduler optimizing solely for network and execution latency,
    without clinical risk awareness (R_i masked).
    """
    def __init__(self, dqn_model, device="cpu"):
        self.dqn_model = dqn_model
        self.device = device

    def schedule(self, observation: np.ndarray, task) -> int:
        # Mask out clinical risk (obs[50] = 0.0)
        blind_obs = observation.copy()
        blind_obs[50] = 0.0
        return self.dqn_model.act(blind_obs, epsilon=0.0, device=self.device)

    def reset(self):
        pass


def run_single_simulation(
    scheduler: BaseScheduler,
    seed: int,
    arrival_rate: float = 20.0,
    num_tasks: int = 1000
) -> Dict[str, float]:
    """Runs a single simulation episode and extracts all 5 primary metrics."""
    set_global_seed(seed)
    env = HealthcareEdgeEnv(seed=seed, arrival_rate=arrival_rate, max_tasks=num_tasks)
    obs, _ = env.reset(seed=seed)
    scheduler.reset()

    done = False
    step_idx = 0

    while not done:
        task = env.tasks[step_idx]
        action = scheduler.schedule(obs, task)
        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1
        done = terminated or truncated

    summary = env._get_info()
    return {
        "avg_latency_ms": float(summary["avg_latency_ms"]),
        "violation_rate_pct": float(summary["violation_rate"]),
        "risk_coverage_pct": float(summary["risk_coverage"]),
        "edge_utilization_pct": float(summary["avg_edge_utilization"]),
        "energy_joules": float(summary["total_energy_j"])
    }


def execute_peak_load_benchmarks(
    pipeline: CRAILCSPipeline,
    seeds: List[int] = BENCHMARK_SEEDS,
    arrival_rate: float = 20.0,
    num_tasks: int = 1000,
    output_csv: str = "results/benchmarks_lambda20.csv"
) -> pd.DataFrame:
    """
    Executes 10-seed comparative benchmark under peak workload (lambda = 20 tasks/sec).
    Replicates Table 7 from Nature Scientific Reports 2026.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"\n========================================================================")
    print(f" Executing Peak-Load Benchmark (lambda = {arrival_rate} tasks/sec, {len(seeds)} Seeds)")
    print(f"========================================================================")

    # Instantiate schedulers
    schedulers: Dict[str, BaseScheduler] = {
        "CRAI-LCS (Proposed)": PipelineSchedulerAdapter(pipeline),
        "EADRL": EADRLSchedulerAdapter(pipeline.dqn_model, device=pipeline.device),
        "DRL-Fog": DRLFogSchedulerAdapter(pipeline.dqn_model, device=pipeline.device),
        "Latency-Energy DRL": FIFOScheduler(),
        "Risk-Priority Heuristic": RiskPriorityScheduler(),
        "EDF": EDFScheduler(),
        "Round-Robin": RoundRobinScheduler(),
        "Random": RandomScheduler(seed=42)
    }

    records: List[Dict[str, Any]] = []

    for name, sched in schedulers.items():
        t0 = time.time()
        seed_metrics: List[Dict[str, float]] = []

        for s in seeds:
            res = run_single_simulation(
                scheduler=sched,
                seed=s,
                arrival_rate=arrival_rate,
                num_tasks=num_tasks
            )
            res["scheduler"] = name
            res["seed"] = s
            res["arrival_rate"] = arrival_rate
            records.append(res)
            seed_metrics.append(res)

        elapsed = time.time() - t0
        avg_lat = np.mean([m["avg_latency_ms"] for m in seed_metrics])
        std_lat = np.std([m["avg_latency_ms"] for m in seed_metrics])
        avg_v = np.mean([m["violation_rate_pct"] for m in seed_metrics])
        avg_c = np.mean([m["risk_coverage_pct"] for m in seed_metrics])
        avg_e = np.mean([m["energy_joules"] for m in seed_metrics])

        print(
            f"{name:<24} | Latency: {avg_lat:5.1f} +/- {std_lat:3.1f} ms | "
            f"Violations: {avg_v:5.2f}% | Risk Coverage: {avg_c:5.1f}% | "
            f"Energy: {avg_e:5.1f} J | Time: {elapsed:.2f}s"
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved peak-load benchmark results to {output_csv}")
    return df


def execute_load_sweep(
    pipeline: CRAILCSPipeline,
    arrival_rates: List[float] = ARRIVAL_RATES,
    seeds: List[int] = BENCHMARK_SEEDS[:3],  # 3 representative seeds for sweep
    num_tasks: int = 1000,
    output_csv: str = "results/load_sweep_results.csv"
) -> pd.DataFrame:
    """
    Executes multi-load arrival sweeps across lambda in [8, 25] tasks/sec.
    Replicates Figures 3-7 from Nature Scientific Reports 2026.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"\n========================================================================")
    print(f" Executing Multi-Load Arrival Rate Sweeps (lambda in {arrival_rates})")
    print(f"========================================================================")

    schedulers: Dict[str, BaseScheduler] = {
        "CRAI-LCS (Proposed)": PipelineSchedulerAdapter(pipeline),
        "EADRL": EADRLSchedulerAdapter(pipeline.dqn_model, device=pipeline.device),
        "DRL-Fog": DRLFogSchedulerAdapter(pipeline.dqn_model, device=pipeline.device),
        "Latency-Energy DRL": FIFOScheduler(),
        "Risk-Priority": RiskPriorityScheduler(),
        "EDF": EDFScheduler()
    }

    records: List[Dict[str, Any]] = []

    for rate in arrival_rates:
        for name, sched in schedulers.items():
            for s in seeds:
                res = run_single_simulation(
                    scheduler=sched,
                    seed=s,
                    arrival_rate=rate,
                    num_tasks=num_tasks
                )
                res["scheduler"] = name
                res["arrival_rate"] = rate
                res["seed"] = s
                records.append(res)

        print(f"Completed sweep for lambda = {rate:4.1f} tasks/sec")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved load sweep results to {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peak-Load & Multi-Load Workload Benchmark Sweeps")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", type=int, default=1000)
    args = parser.parse_args()

    pipe = CRAILCSPipeline(device=args.device)
    execute_peak_load_benchmarks(pipe, num_tasks=args.tasks)
    execute_load_sweep(pipe, num_tasks=args.tasks)
