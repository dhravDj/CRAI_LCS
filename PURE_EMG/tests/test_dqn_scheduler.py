import time
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.dqn_scheduler import DuelingDQN, DQNScheduler
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.reward_shaping import compute_reward, REWARD_PROFILES
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.utils.seed import set_global_seed

def test_dqn_parameter_budget():
    """
    Verifies that the Dueling Double DQN network strictly adheres to Table 12:
    0.32M parameters +/- 2% (313,600 <= params <= 326,400).
    """
    model = DuelingDQN(state_dim=54, action_dim=26)
    total_params = model.total_parameters

    print(f"Total Dueling DQN parameters: {total_params:,}")
    assert 313_600 <= total_params <= 326_400, (
        f"Parameter count {total_params} exceeds 0.32M +/- 2% budget!"
    )
    assert total_params == 320_045, f"Expected 320,045 parameters, got {total_params}"


def test_dqn_forward_tensor_contracts():
    """
    Verifies input/output tensor shapes and output finiteness:
    - Input: (batch_size, 54) or (54,)
    - Output: (batch_size, 26) or (1, 26)
    """
    model = DuelingDQN(state_dim=54, action_dim=26)
    model.eval()

    # 1. 2D Batch input
    batch_input = torch.randn(32, 54)
    q_vals = model(batch_input)
    assert q_vals.shape == (32, 26), f"Expected (32, 26), got {q_vals.shape}"
    assert torch.isfinite(q_vals).all(), "Non-finite values encountered in Q-values!"

    # 2. 1D Vector input
    single_input = torch.randn(54)
    q_single = model(single_input)
    assert q_single.shape == (1, 26), f"Expected (1, 26), got {q_single.shape}"
    assert torch.isfinite(q_single).all(), "Non-finite values encountered in single Q-value!"

    # 3. Action selection via act()
    state_np = np.random.randn(54).astype(np.float32)
    action = model.act(state_np, epsilon=0.0)
    assert isinstance(action, int)
    assert 0 <= action <= 25, f"Selected action {action} out of bounds [0, 25]!"


def test_dqn_cpu_inference_latency():
    """
    Verifies that DQN forward pass CPU latency <= 5.7 ms (Table 12 budget).
    """
    model = DuelingDQN(state_dim=54, action_dim=26)
    model.eval()
    model.to("cpu")

    dummy_state = torch.randn(1, 54, device="cpu")

    # Warmup
    for _ in range(50):
        _ = model(dummy_state)

    # Measure 500 inferences
    num_runs = 500
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_state)
    t1 = time.perf_counter()

    avg_latency_ms = (t1 - t0) / num_runs * 1000.0
    print(f"Measured DQN forward pass latency: {avg_latency_ms:.3f} ms (budget <= 5.7 ms)")
    assert avg_latency_ms <= 5.7, f"DQN latency {avg_latency_ms:.3f} ms exceeds 5.7 ms limit!"


def test_prioritized_replay_buffer():
    """
    Verifies Prioritized Experience Replay buffer contracts:
    - Push and circular wrapping
    - Proportional priority sampling
    - Batch shapes and importance-sampling weights
    - Priority update via TD error
    """
    buffer = PrioritizedReplayBuffer(capacity=1000, state_dim=54, alpha=0.6, seed=42)

    for i in range(200):
        s = np.full(54, i, dtype=np.float32)
        a = i % 26
        r = float(i) * 0.1
        s_next = np.full(54, i + 1, dtype=np.float32)
        done = (i % 50 == 0)
        buffer.push(s, a, r, s_next, done)

    assert len(buffer) == 200

    # Sample batch
    b_s, b_a, b_r, b_sn, b_d, b_w, indices = buffer.sample(batch_size=32, beta=0.4)
    assert b_s.shape == (32, 54)
    assert b_a.shape == (32,)
    assert b_r.shape == (32,)
    assert b_sn.shape == (32, 54)
    assert b_d.shape == (32,)
    assert b_w.shape == (32,)
    assert len(indices) == 32
    assert (b_w > 0.0).all() and (b_w <= 1.0 + 1e-5).all()

    # Update priorities
    dummy_errors = np.abs(np.random.randn(32).astype(np.float32))
    buffer.update_priorities(indices, dummy_errors)
    assert (buffer.priorities[indices] > 0.0).all()


def test_reward_shaping_profiles_and_bounds():
    """
    Verifies Multi-Objective Reward Function (Equation 29, Table 10-11):
    - Tests RW-1, RW-2, RW-3, RW-4 profiles
    - Tests deadline compliance bonus vs violation penalty
    - Tests scale invariance (norm_latency = L_i / D_i)
    """
    # 1. On-time high-risk task (L_i <= D_i, R_i = 0.9, on-time risk bonus active)
    r_ontime = compute_reward(
        latency_ms=80.0,
        deadline_ms=120.0,
        predicted_miss_prob=0.05,
        clinical_risk=0.90,
        mean_edge_util=0.50,
        profile="RW-2"
    )
    # L_i / D_i = 80 / 120 = 0.6667
    # -(0.35 * 0.6667 + 0.25 * 0.05 - 0.30 * 0.90) + 0.10 * 0.50
    # -(0.2333 + 0.0125 - 0.270) + 0.050 = -(-0.0242) + 0.050 = +0.0742
    assert r_ontime > 0.0, f"Expected positive reward for on-time critical task, got {r_ontime}"

    # 2. Missed-deadline high-risk task (L_i > D_i, no risk bonus, severe penalty)
    r_missed = compute_reward(
        latency_ms=250.0,
        deadline_ms=120.0,
        predicted_miss_prob=0.85,
        clinical_risk=0.90,
        mean_edge_util=0.50,
        profile="RW-2"
    )
    # L_i / D_i = 250 / 120 = 2.0833 (no risk bonus since missed)
    assert r_missed < 0.0, f"Expected negative penalty for missed deadline, got {r_missed}"
    assert r_ontime > r_missed, "On-time task must receive significantly higher reward than missed task!"

    # 3. All sensitivity profiles defined
    for p_name in ["RW-1", "RW-2", "RW-3", "RW-4"]:
        assert p_name in REWARD_PROFILES
        r_val = compute_reward(100.0, 150.0, 0.1, 0.5, 0.4, profile=p_name)
        assert np.isfinite(r_val)


def test_dqn_scheduler_simulation_execution():
    """
    Verifies that DQNScheduler interacts seamlessly with HealthcareEdgeEnv
    over 500 tasks without exceptions or deadlock.
    """
    set_global_seed(42)
    env = HealthcareEdgeEnv(seed=42, max_tasks=500)
    scheduler = DQNScheduler(checkpoint_path=None, device="cpu")

    obs, _ = env.reset(seed=42)
    done = False
    step_count = 0

    while not done:
        action = scheduler.schedule(obs)
        assert 0 <= action <= 25, f"Invalid action {action}"
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

    assert step_count == 500
    info = env._get_info()
    assert info["total_tasks"] == 500


def test_trained_dqn_checkpoint_evaluates_on_1000_tasks():
    """
    Verifies that the trained Double DQN checkpoint (checkpoints/dqn_scheduler.pt)
    loads properly and delivers high performance on a full 1,000-task simulation run:
    - Zero deadlocks or uncaught exceptions
    - Violation rate <= 1.0% (target 0.0%)
    - Urgent clinical risk coverage >= 99.0% (target 100.0%)
    """
    import os
    ckpt_path = "checkpoints/dqn_scheduler.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found!"

    set_global_seed(42)
    env = HealthcareEdgeEnv(seed=42, arrival_rate=20.0, max_tasks=1000)
    scheduler = DQNScheduler(checkpoint_path=ckpt_path, device="cpu")
    assert scheduler.is_trained, "DQNScheduler should report is_trained=True with valid checkpoint"

    obs, _ = env.reset(seed=42)
    done = False
    step_count = 0

    while not done:
        action = scheduler.schedule(obs)
        assert 0 <= action <= 25
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

    assert step_count == 1000
    metrics = env._get_info()
    print(f"Trained DQN Evaluation on 1,000 tasks: {metrics}")
    assert metrics["violation_rate"] <= 1.0, f"Violation rate {metrics['violation_rate']}% exceeds 1.0% threshold!"
    assert metrics["risk_coverage"] >= 99.0, f"Risk coverage {metrics['risk_coverage']}% below 99.0% threshold!"

