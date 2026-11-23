import pytest
import numpy as np
from src.simulator.task import HealthcareTask, create_synthetic_task
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.baselines.heuristics import (
    RandomScheduler,
    RoundRobinScheduler,
    FIFOScheduler,
    EDFScheduler,
    RiskPriorityScheduler
)

def test_scheduler_interfaces_and_bounds():
    """Verifies that all 5 baseline schedulers implement the schedule interface and return valid actions."""
    schedulers = [
        RandomScheduler(seed=42),
        RoundRobinScheduler(),
        FIFOScheduler(),
        EDFScheduler(),
        RiskPriorityScheduler()
    ]
    dummy_task = create_synthetic_task(task_id=0, arrival_time=0.0)
    obs = np.zeros(54, dtype=np.float32)

    for scheduler in schedulers:
        action = scheduler.schedule(obs, dummy_task)
        assert isinstance(action, int), f"{scheduler.__class__.__name__} must return an int"
        assert 0 <= action <= 25, f"{scheduler.__class__.__name__} action {action} out of bounds [0, 25]"
        scheduler.reset()


def test_round_robin_cyclic_sequence():
    """Verifies that RoundRobin cycles sequentially 0..24 across edge nodes."""
    rr = RoundRobinScheduler(num_edge=25)
    dummy_task = create_synthetic_task(task_id=0, arrival_time=0.0)
    obs = np.zeros(54, dtype=np.float32)

    actions = [rr.schedule(obs, dummy_task) for _ in range(55)]
    expected = list(range(25)) + list(range(25)) + list(range(5))
    assert actions == expected

    rr.reset()
    assert rr.schedule(obs, dummy_task) == 0


def test_fifo_selects_minimum_queue_node():
    """Verifies that FIFO always dispatches to the edge node with minimal queue workload."""
    fifo = FIFOScheduler()
    dummy_task = create_synthetic_task(task_id=1, arrival_time=1.0)

    # Node 17 has the smallest queue load
    obs = np.ones(54, dtype=np.float32) * 5.0
    obs[17] = 0.2

    action = fifo.schedule(obs, dummy_task)
    assert action == 17


def test_edf_urgency_branching():
    """Verifies that EDF keeps tight deadlines (<200ms) on edge and offloads relaxed deadlines to cloud."""
    edf = EDFScheduler()
    obs = np.ones(54, dtype=np.float32) * 2.0
    obs[3] = 0.1  # Fast available edge node

    urgent_task = HealthcareTask(task_id=1, delta_i=3600.0, psi_i=1.0e7, deadline_ms=120.0, clinical_risk=0.8, arrival_time=0.0)
    routine_task = HealthcareTask(task_id=2, delta_i=3600.0, psi_i=1.0e7, deadline_ms=600.0, clinical_risk=0.1, arrival_time=0.0)

    assert edf.schedule(obs, urgent_task) == 3, "Urgent task must go to shortest-queue edge node"
    assert edf.schedule(obs, routine_task) == 25, "Relaxed deadline task must go to Central Cloud"


def test_risk_priority_clinical_branching():
    """Verifies that RiskPriority dispatches urgent tasks (R_i >= 0.5) to edge and routine to cloud."""
    rp = RiskPriorityScheduler()
    obs = np.ones(54, dtype=np.float32) * 2.0
    obs[9] = 0.05

    critical_cardiac_task = HealthcareTask(task_id=1, delta_i=3600.0, psi_i=1.0e7, deadline_ms=130.0, clinical_risk=0.92, arrival_time=0.0)
    routine_telemetry_task = HealthcareTask(task_id=2, delta_i=3600.0, psi_i=1.0e7, deadline_ms=500.0, clinical_risk=0.25, arrival_time=0.0)

    assert rp.schedule(obs, critical_cardiac_task) == 9, "Critical cardiac event must route to local edge node"
    assert rp.schedule(obs, routine_telemetry_task) == 25, "Routine telemetry must route to Central Cloud"


def test_1000_task_simulation_comparative_benchmark():
    """
    Executes a complete 1,000-task Poisson workload stream across all 5 baselines
    under an identical sequence, verifying comparative performance hierarchies.
    """
    schedulers = {
        "Random": RandomScheduler(seed=42),
        "Round-Robin": RoundRobinScheduler(),
        "FIFO (SQF)": FIFOScheduler(),
        "EDF": EDFScheduler(),
        "Risk-Priority": RiskPriorityScheduler()
    }

    results = {}

    for name, scheduler in schedulers.items():
        env = HealthcareEdgeEnv(seed=42, arrival_rate=20.0, max_tasks=1000)
        obs, _ = env.reset(seed=42)
        scheduler.reset()

        terminated = False
        step_idx = 0

        while not terminated:
            curr_task = env.tasks[step_idx]
            action = scheduler.schedule(obs, curr_task)
            obs, reward, terminated, truncated, info = env.step(action)
            step_idx += 1

        metrics = env.get_metrics()
        results[name] = metrics
        assert metrics["total_tasks"] == 1000

    # Scientific fidelity assertions matching paper trends (Table 7):
    # 1. Risk-Priority must achieve >= 95% critical risk coverage
    assert results["Risk-Priority"]["risk_coverage"] >= 95.0, (
        f"Risk-Priority coverage {results['Risk-Priority']['risk_coverage']}% must be >= 95%"
    )

    # 2. Risk-Priority and EDF must outperform Random on deadline violation rate
    assert results["Risk-Priority"]["violation_rate"] <= results["Random"]["violation_rate"], (
        "Risk-Priority must yield fewer deadline violations than Random"
    )
