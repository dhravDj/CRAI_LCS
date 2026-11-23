import pytest
import math
from src.simulator.task import HealthcareTask
from src.simulator.nodes import EdgeNode, CloudInstance
from src.simulator.latency_energy import (
    compute_edge_latency_and_energy,
    compute_cloud_latency_and_energy
)

def test_hand_calculated_edge_latency_and_energy():
    """
    Verifies mathematical precision of Equation 8 & 10 against exact hand calculation.
    """
    task = HealthcareTask(
        task_id=1,
        delta_i=3600.0,       # 3600 * 8 = 28,800 bits
        psi_i=1.0e7,          # 10,000,000 cycles
        deadline_ms=100.0,
        clinical_risk=0.8,
        arrival_time=1.0
    )

    node = EdgeNode(
        node_id=0,
        mips=1000.0,          # 1.0e9 cycles/sec
        ram_gb=16.0,
        bandwidth_mbps=20.0,  # 20.0e6 bps
        tx_latency_ms=5.0,    # 0.005 sec
        idle_power_w=50.0,
        active_power_w=100.0
    )

    t_comm, t_queue, t_exec, total_lat_ms, energy_j = compute_edge_latency_and_energy(
        task=task,
        node=node,
        current_time=1.0
    )

    # Hand calculation verification
    expected_t_comm = 0.005 + (28800.0 / 20.0e6)  # 0.005 + 0.00144 = 0.00644 sec
    expected_t_queue = 0.0
    expected_t_exec = 1.0e7 / 1.0e9               # 0.010 sec
    expected_total_lat_ms = (expected_t_comm + expected_t_exec) * 1000.0  # 16.44 ms
    expected_energy_j = (100.0 * 0.010) + (0.50 * 0.00644)              # 1.00322 J

    assert math.isclose(t_comm, expected_t_comm, rel_tol=1e-5)
    assert math.isclose(t_queue, expected_t_queue, rel_tol=1e-5)
    assert math.isclose(t_exec, expected_t_exec, rel_tol=1e-5)
    assert math.isclose(total_lat_ms, expected_total_lat_ms, rel_tol=1e-5)
    assert math.isclose(energy_j, expected_energy_j, rel_tol=1e-5)


def test_hand_calculated_cloud_latency_and_energy():
    """
    Verifies mathematical precision of Equation 9 & 10 against exact hand calculation.
    """
    task = HealthcareTask(
        task_id=2,
        delta_i=3600.0,
        psi_i=1.0e7,
        deadline_ms=500.0,
        clinical_risk=0.2,
        arrival_time=1.0
    )

    cloud = CloudInstance(
        instance_id=0,
        mips=20000.0,          # 2.0e10 cycles/sec
        ram_gb=128.0,
        bandwidth_mbps=200.0,  # 200.0e6 bps
        wan_latency_ms=50.0,   # 0.050 sec
        tx_latency_ms=5.0,     # 0.005 sec
        idle_power_w=200.0,
        active_power_w=400.0
    )

    t_comm, t_queue, t_exec, total_lat_ms, energy_j = compute_cloud_latency_and_energy(
        task=task,
        cloud=cloud,
        current_time=1.0
    )

    # Hand calculation verification
    expected_t_comm = 0.005 + 0.050 + (28800.0 / 200.0e6)  # 0.055144 sec
    expected_t_queue = 0.0
    expected_t_exec = 1.0e7 / 2.0e10                       # 0.0005 sec
    expected_total_lat_ms = (expected_t_comm + expected_t_exec) * 1000.0  # 55.644 ms
    expected_energy_j = (400.0 * 0.0005) + (2.50 * 0.055144)             # 0.33786 J

    assert math.isclose(t_comm, expected_t_comm, rel_tol=1e-5)
    assert math.isclose(t_queue, expected_t_queue, rel_tol=1e-5)
    assert math.isclose(t_exec, expected_t_exec, rel_tol=1e-5)
    assert math.isclose(total_lat_ms, expected_total_lat_ms, rel_tol=1e-5)
    assert math.isclose(energy_j, expected_energy_j, rel_tol=1e-5)


def test_edge_vs_cloud_clinical_urgency_tradeoff():
    """
    Verifies the fundamental Healthcare IoT dilemma:
    - High-urgency task (deadline 120 ms) succeeds on local edge (latency ~15-25 ms)
    - Cloud offloading suffers WAN delay (45-120 ms transmission alone), threatening deadline breach under high congestion.
    """
    urgent_task = HealthcareTask(
        task_id=3,
        delta_i=3600.0,
        psi_i=1.5e7,
        deadline_ms=120.0,
        clinical_risk=0.9,
        arrival_time=0.0
    )

    edge = EdgeNode(0, 3000.0, 32.0, 30.0, 5.0, 50.0, 110.0)
    cloud = CloudInstance(0, 25000.0, 128.0, 300.0, 100.0, 5.0, 250.0, 500.0)

    _, _, _, edge_lat, _ = compute_edge_latency_and_energy(urgent_task, edge, 0.0)
    _, _, _, cloud_lat, _ = compute_cloud_latency_and_energy(urgent_task, cloud, 0.0)

    assert edge_lat < urgent_task.deadline_ms, f"Edge latency {edge_lat} must meet urgent deadline"
    assert edge_lat < cloud_lat, "Fast edge node must offer lower end-to-end latency than remote cloud for single task"
