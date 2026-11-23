#!/usr/bin/env python3
"""
U-CRAI-LCS Full 3,000-Task Empirical Multi-Disease Benchmark.
Runs 3,000 heterogeneous clinical tasks (1,000 ECG, 1,000 EEG, 1,000 PPG)
across 25 Edge Nodes and 2 Central Cloud Instances.
"""

import sys
import os
import json
import time

# Ensure project root in pythonpath
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.universal.benchmark import UniversalStressBenchmark


def main():
    print("=" * 80)
    print("  U-CRAI-LCS: UNIVERSAL MULTI-DISEASE HEALTHCARE CONTINUUM BENCHMARK")
    print("  Modality Signals: Cardiology ECG, Neurology EEG, Pulmonology PPG")
    print("  Scale: 3,000 Tasks | 25 Edge Gateways | 2 Cloud Datacenter Instances")
    print("=" * 80)

    num_tasks = 3000
    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])

    bench = UniversalStressBenchmark(seed=42)
    start_time = time.time()
    results = bench.run_benchmark(num_tasks=num_tasks)
    elapsed = time.time() - start_time

    u = results["u_crai_lcs"]
    c = results["cloud_only"]
    g = results["greedy_edge"]

    print("\n" + "=" * 80)
    print("                      EMPIRICAL BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"{'Metric':<35} | {'U-CRAI-LCS':<14} | {'Cloud-Only':<12} | {'Greedy Edge':<12}")
    print("-" * 80)
    print(f"{'Mean End-to-End Latency (ms)':<35} | {u['avg_latency_ms']:<14.2f} | {c['avg_latency_ms']:<12.2f} | {g['avg_latency_ms']:<12.2f}")
    print(f"{'P95 End-to-End Latency (ms)':<35} | {u['p95_latency_ms']:<14.2f} | {c['p95_latency_ms']:<12.2f} | {g['p95_latency_ms']:<12.2f}")
    print(f"{'Overall SLA Violation Rate (%)':<35} | {u['violation_rate']:<14.2f} | {c['violation_rate']:<12.2f} | {g['violation_rate']:<12.2f}")
    print(f"{'Urgent SLA Violation Rate (%)':<35} | {u['urgent_violation_rate']:<14.2f} | {c['urgent_violation_rate']:<12.2f} | {g['urgent_violation_rate']:<12.2f}")
    print(f"{'Fail-Safe Preemptions triggered':<35} | {u['fail_safe_preemptions']:<14} | {c['fail_safe_preemptions']:<12} | {g['fail_safe_preemptions']:<12}")
    print(f"{'Total Continuum Energy (J)':<35} | {u['total_energy_j']:<14.2f} | {c['total_energy_j']:<12.2f} | {g['total_energy_j']:<12.2f}")
    print(f"{'Asymmetric Sensor Gating Ratio':<35} | {u['gating_ratio_pct']:<14.1f}% | {'N/A (0.0%)':<12} | {'N/A (0.0%)':<12}")
    print(f"{'Edge Battery Life Extension':<35} | {u['battery_extension_factor']:<14.2f}x | {'1.00x':<12} | {'1.00x':<12}")
    print(f"{'Continuum Throughput (tasks/s)':<35} | {u['throughput_tps']:<14.1f} | {'N/A':<12} | {'N/A':<12}")
    print("=" * 80)
    print(f"Total benchmark execution time: {elapsed:.2f} seconds.")

    out_file = "reports/universal_benchmark_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}\n")


if __name__ == "__main__":
    main()
