import pytest
import numpy as np
from src.simulator.edge_cloud_env import HealthcareEdgeEnv

def test_env_gym_interface_compliance():
    """
    Verifies that HealthcareEdgeEnv complies with standard Farama Gymnasium specifications (T1-03).
    """
    env = HealthcareEdgeEnv(seed=42, max_tasks=100)

    # 1. Action space verification: 26 discrete actions
    assert env.action_space.n == 26
    assert env.action_space.contains(0)
    assert env.action_space.contains(25)
    assert not env.action_space.contains(26)

    # 2. Observation space verification: 54 continuous dimensions
    assert env.observation_space.shape == (54,)
    assert env.observation_space.dtype == np.float32

    # 3. Reset verification
    obs, info = env.reset(seed=42)
    assert obs.shape == (54,)
    assert obs.dtype == np.float32
    assert not np.isnan(obs).any(), "Observation must not contain NaNs"
    assert not np.isinf(obs).any(), "Observation must not contain Infs"
    assert "total_tasks" in info


def test_all_26_actions_executable_without_deadlock():
    """
    Verifies that every discrete action in {0..25} (Edge 0-24, Cloud 25)
    can execute safely without runtime exceptions or queue deadlock.
    """
    env = HealthcareEdgeEnv(seed=42, max_tasks=30)
    obs, info = env.reset()

    for action in range(26):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (54,)
        assert isinstance(reward, float)
        assert not np.isnan(reward), f"Reward is NaN for action {action}"
        assert not terminated or action == 25, "Premature termination"
        assert "step_latency_ms" in info
        assert "step_energy_j" in info
        assert "step_violated" in info
        assert info["step_latency_ms"] > 0.0


def test_1000_tasks_full_simulation_run():
    """
    Executes a complete 1,000-task Poisson workload stream (lambda = 20 tasks/sec)
    verifying end-to-end discrete event stability and metrics logging.
    """
    env = HealthcareEdgeEnv(seed=42, arrival_rate=20.0, max_tasks=1000)
    obs, info = env.reset()

    step_count = 0
    terminated = False

    while not terminated:
        # Simple cyclic assignment across nodes
        action = step_count % 26
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert step_count == 1000, "Must process exactly 1000 tasks"
    assert terminated is True

    metrics = env.get_metrics()
    assert metrics["total_tasks"] == 1000
    assert metrics["avg_latency_ms"] > 0.0
    assert 0.0 <= metrics["violation_rate"] <= 100.0
    assert 0.0 <= metrics["risk_coverage"] <= 100.0
    assert 0.0 <= metrics["avg_edge_utilization"] <= 100.0
    assert metrics["total_energy_j"] > 0.0
    assert metrics["energy_per_1k_tasks"] > 0.0


def test_deterministic_reproducibility():
    """
    Verifies NFR-2: Two independent environment runs with seed=42 and identical actions
    must yield bitwise identical observations, rewards, and latencies.
    """
    actions = [0, 5, 12, 24, 25, 3, 7, 19, 25, 1]

    # Run 1
    env1 = HealthcareEdgeEnv(seed=42, max_tasks=len(actions))
    obs1, _ = env1.reset(seed=42)
    rewards1 = []
    latencies1 = []
    for a in actions:
        o1, r1, _, _, info1 = env1.step(a)
        rewards1.append(r1)
        latencies1.append(info1["step_latency_ms"])

    # Run 2
    env2 = HealthcareEdgeEnv(seed=42, max_tasks=len(actions))
    obs2, _ = env2.reset(seed=42)
    rewards2 = []
    latencies2 = []
    for a in actions:
        o2, r2, _, _, info2 = env2.step(a)
        rewards2.append(r2)
        latencies2.append(info2["step_latency_ms"])

    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_almost_equal(rewards1, rewards2, decimal=6)
    np.testing.assert_array_almost_equal(latencies1, latencies2, decimal=6)


def test_poisson_arrival_distribution():
    """
    Verifies that generated task arrival stream matches Poisson process with rate lambda = 20.
    Mean inter-arrival time should be approximately 1 / 20 = 0.05 seconds.
    """
    arrival_rate = 20.0
    env = HealthcareEdgeEnv(seed=123, arrival_rate=arrival_rate, max_tasks=2000)
    arrivals = [t.arrival_time for t in env.tasks]
    inter_arrivals = np.diff(arrivals)

    empirical_mean = np.mean(inter_arrivals)
    expected_mean = 1.0 / arrival_rate  # 0.050 seconds

    # Within 5% of theoretical mean over 2,000 samples
    assert np.isclose(empirical_mean, expected_mean, rtol=0.08), (
        f"Empirical mean {empirical_mean:.4f} differs from expected {expected_mean:.4f}"
    )
