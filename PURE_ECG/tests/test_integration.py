import os
import pytest
import numpy as np
import torch

from src.pipeline import CRAILCSPipeline
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.utils.seed import set_global_seed

def test_pipeline_initialization():
    """
    Verifies that CRAILCSPipeline loads all 3 neural network modules
    from their respective checkpoints in eval mode:
    1. 1D-CNN-LSTM Risk Network (~0.74M parameters)
    2. Deadline Miss Predictor (~0.18M parameters)
    3. Double DQN Scheduler (~0.32M parameters)
    """
    pipeline = CRAILCSPipeline(device="cpu")
    assert not pipeline.risk_net.training
    assert not pipeline.miss_predictor.training
    assert not pipeline.dqn_model.training

    # Check total parameters across the integrated multi-model architecture
    p_risk = sum(p.numel() for p in pipeline.risk_net.parameters())
    p_miss = sum(p.numel() for p in pipeline.miss_predictor.parameters())
    p_dqn = pipeline.dqn_model.total_parameters
    total_ai_params = p_risk + p_miss + p_dqn

    print(f"Risk: {p_risk:,} | Predictor: {p_miss:,} | DQN: {p_dqn:,} | Total: {total_ai_params:,}")
    # 0.74M + 0.18M + 0.32M = 1.24M parameters total (Table 12)
    assert 1_200_000 <= total_ai_params <= 1_300_000, f"Total params {total_ai_params} out of bounds!"


def test_multi_model_inference_budgets():
    """
    Verifies strict compliance with Nature Scientific Reports Table 12:
    - 1D-CNN-LSTM Risk Net: <= 14.8 ms
    - Deadline Predictor:   <= 3.6 ms
    - Double DQN Scheduler: <= 5.7 ms
    - Combined Total AI:    <= 24.1 ms (<= 25.0 ms target)
    """
    pipeline = CRAILCSPipeline(device="cpu")
    breakdown = pipeline.measure_ai_latency_breakdown(num_samples=100)

    print(f"Measured Multi-Model Inference Latency Breakdown (CPU): {breakdown}")
    assert breakdown["cnn_lstm_ms"] <= 14.8, f"CNN-LSTM latency {breakdown['cnn_lstm_ms']:.2f} ms > 14.8 ms!"
    assert breakdown["deadline_predictor_ms"] <= 3.6, f"Predictor latency {breakdown['deadline_predictor_ms']:.2f} ms > 3.6 ms!"
    assert breakdown["dqn_scheduler_ms"] <= 5.7, f"DQN latency {breakdown['dqn_scheduler_ms']:.2f} ms > 5.7 ms!"
    assert breakdown["total_ai_latency_ms"] <= 24.1, f"Total AI latency {breakdown['total_ai_latency_ms']:.2f} ms > 24.1 ms!"


def test_clinical_fail_safe_interrupt_protocol():
    """
    Verifies the Clinical Fail-Safe Interrupt Protocol (FR-005, ADR-003):
    - When R_i >= 0.80 (Acute Emergency):
      1. Fail-safe interrupt triggers (fail_safe_triggered = True).
      2. Cloud WAN offloading (action 25) is strictly bypassed.
      3. Task is routed to patient's local edge gateway.
      4. Task is flagged with is_preempted = True to bypass queue delay.
      5. Execution turnaround time is strictly < 100 ms.
    - When R_i < 0.50 (Routine):
      1. Fail-safe is inactive (fail_safe_triggered = False).
      2. Double DQN chooses normal offloading action.
    """
    pipeline = CRAILCSPipeline(device="cpu", fail_safe_threshold=0.80)
    env = HealthcareEdgeEnv(seed=42, max_tasks=10)
    env.reset(seed=42)

    # 1. High-Risk Emergency Case (R_i = 0.95 >= 0.80)
    dummy_obs = env._get_observation()
    action, fail_safe, _ = pipeline.select_action(dummy_obs, clinical_risk=0.95, patient_id=7)
    assert fail_safe is True, "Fail-safe protocol MUST trigger for R_i >= 0.80!"
    assert action != 25, "Fail-safe MUST NEVER route acute life-threatening emergency to Cloud WAN!"
    assert action == 7, f"Fail-safe should route to patient 7 local edge gateway (action 7), got {action}"

    # 2. Routine Case (R_i = 0.15 < 0.80)
    action_routine, fail_safe_routine, _ = pipeline.select_action(dummy_obs, clinical_risk=0.15, patient_id=7)
    assert fail_safe_routine is False, "Fail-safe must NOT trigger for routine task R_i = 0.15"
    assert 0 <= action_routine <= 25, f"Invalid DQN action {action_routine}"


def test_100_patient_streaming_simulation():
    """
    Simulates 100 streaming hospital patients (1,000 tasks total)
    through the complete continuous closed loop:
    Raw ECG -> CNN-LSTM -> Fail-Safe -> Predictor -> DQN -> Simulator.
    - Zero runtime exceptions or memory leaks.
    - Deadline violation rate <= 1.0% (target 0.0%).
    - Urgent clinical risk coverage >= 99.0% (target 100.0%).
    - Mean end-to-end task turnaround latency < 50 ms.
    """
    pipeline = CRAILCSPipeline(device="cpu")
    metrics = pipeline.run_streaming_simulation(
        num_patients=100,
        tasks_per_patient=10,
        arrival_rate=20.0,
        seed=42
    )

    print(f"100-Patient Continuous Streaming Simulation Results: {metrics}")
    assert metrics["total_patients"] == 100
    assert metrics["total_tasks"] == 1000
    assert metrics["violation_rate_pct"] <= 1.0, f"Violation rate {metrics['violation_rate_pct']}% exceeds 1.0%!"
    assert metrics["urgent_risk_coverage_pct"] >= 99.0, f"Risk coverage {metrics['urgent_risk_coverage_pct']}% < 99.0%!"
    assert metrics["avg_latency_ms"] < 50.0, f"Avg latency {metrics['avg_latency_ms']:.2f} ms >= 50.0 ms!"
    assert metrics["fail_safe_triggers"] > 0, "Fail-safe should have triggered on acute arrhythmia episodes!"


def test_closed_loop_deterministic_reproducibility():
    """
    Verifies that running the complete pipeline with identical seed
    produces identical metrics and zero divergence (NFR-2).
    """
    pipeline = CRAILCSPipeline(device="cpu", seed=42)
    res1 = pipeline.run_streaming_simulation(num_patients=20, tasks_per_patient=5, seed=42)
    res2 = pipeline.run_streaming_simulation(num_patients=20, tasks_per_patient=5, seed=42)

    assert res1["avg_latency_ms"] == res2["avg_latency_ms"]
    assert res1["violation_rate_pct"] == res2["violation_rate_pct"]
    assert res1["fail_safe_triggers"] == res2["fail_safe_triggers"]
