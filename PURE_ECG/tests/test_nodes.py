import pytest
import numpy as np
from src.simulator.nodes import EdgeNode, CloudInstance, create_continuum_nodes
from src.simulator.task import HealthcareTask

def test_edge_node_initialization():
    node = EdgeNode(
        node_id=0,
        mips=2000.0,
        ram_gb=32.0,
        bandwidth_mbps=25.0,
        tx_latency_ms=5.0,
        idle_power_w=50.0,
        active_power_w=110.0,
        capacity_score=0.5
    )
    assert node.node_id == 0
    assert node.mips == 2000.0
    assert node.ips == 2.0e9
    assert node.ram_gb == 32.0
    assert node.bandwidth_mbps == 25.0
    assert node.tx_latency_ms == 5.0
    assert node.available_time == 0.0

def test_cloud_instance_initialization():
    cloud = CloudInstance(
        instance_id=0,
        mips=25000.0,
        ram_gb=128.0,
        bandwidth_mbps=500.0,
        wan_latency_ms=80.0,
        tx_latency_ms=6.0,
        idle_power_w=250.0,
        active_power_w=500.0
    )
    assert cloud.instance_id == 0
    assert cloud.mips == 25000.0
    assert cloud.ips == 2.5e10
    assert cloud.ram_gb == 128.0
    assert cloud.wan_latency_ms == 80.0
    assert cloud.available_time == 0.0

def test_continuum_nodes_generation_bounds():
    edge_nodes, cloud_instances = create_continuum_nodes(num_edge=25, num_cloud=2, seed=42)

    assert len(edge_nodes) == 25, "Must generate exactly 25 edge nodes (FR-007)"
    assert len(cloud_instances) == 2, "Must generate exactly 2 cloud instances (FR-008)"

    # Verify Edge Gateway parameter ranges from Table 4
    for node in edge_nodes:
        assert 800.0 <= node.mips <= 4500.0, f"Edge MIPS {node.mips} outside [800, 4500]"
        assert node.ram_gb in [16.0, 32.0, 64.0], f"Edge RAM {node.ram_gb} outside [16, 64]"
        assert 10.0 <= node.bandwidth_mbps <= 50.0, f"Edge Bandwidth {node.bandwidth_mbps} outside [10, 50]"
        assert 3.0 <= node.tx_latency_ms <= 12.0, f"Edge Tx latency {node.tx_latency_ms} outside [3, 12]"
        assert 40.0 <= node.idle_power_w <= 80.0, f"Edge Idle power {node.idle_power_w} outside [40, 80]"
        assert 0.0 <= node.capacity_score <= 1.0, f"Capacity score {node.capacity_score} outside [0, 1]"

    # Verify Cloud Instance parameter ranges from Table 4
    for cloud in cloud_instances:
        assert 12000.0 <= cloud.mips <= 35000.0, f"Cloud MIPS {cloud.mips} outside [12000, 35000]"
        assert cloud.ram_gb == 128.0, "Cloud RAM must be 128 GB"
        assert 100.0 <= cloud.bandwidth_mbps <= 1000.0, f"Cloud Bandwidth {cloud.bandwidth_mbps} outside [100, 1000]"
        assert 45.0 <= cloud.wan_latency_ms <= 120.0, f"Cloud WAN latency {cloud.wan_latency_ms} outside [45, 120]"
        assert 180.0 <= cloud.idle_power_w <= 350.0, f"Cloud Idle power {cloud.idle_power_w} outside [180, 350]"

def test_anti_clock_desynchronization_rsk006():
    """
    Verifies RSK-006: Available time updates correctly:
    NodeAvailableTime = max(NodeAvailableTime, arrival_at_node) + t_exec.
    No reverse chronological executions or negative queuing delays.
    """
    node = EdgeNode(
        node_id=0,
        mips=1000.0,
        ram_gb=16.0,
        bandwidth_mbps=20.0,
        tx_latency_ms=5.0,
        idle_power_w=50.0,
        active_power_w=100.0
    )

    dummy_task = HealthcareTask(
        task_id=1,
        delta_i=3600.0,
        psi_i=1.0e7,
        deadline_ms=150.0,
        clinical_risk=0.8,
        arrival_time=0.0
    )

    # Task 1 arrives at t = 1.0, takes 10.0 seconds
    start_1 = node.schedule_task(dummy_task, arrival_at_node=1.0, exec_time_sec=10.0, energy_j=100.0)
    assert start_1 == 1.0
    assert node.available_time == 11.0

    # Task 2 arrives at t = 5.0 (while node is busy until 11.0)
    queue_delay_2 = node.get_queue_delay(current_time=5.0, arrival_at_node=5.0)
    assert queue_delay_2 == 6.0  # Must wait 11.0 - 5.0 = 6.0 seconds

    start_2 = node.schedule_task(dummy_task, arrival_at_node=5.0, exec_time_sec=5.0, energy_j=50.0)
    assert start_2 == 11.0
    assert node.available_time == 16.0

    # Task 3 arrives at t = 20.0 (node has been idle since 16.0)
    queue_delay_3 = node.get_queue_delay(current_time=20.0, arrival_at_node=20.0)
    assert queue_delay_3 == 0.0  # Zero queuing delay

    start_3 = node.schedule_task(dummy_task, arrival_at_node=20.0, exec_time_sec=2.0, energy_j=20.0)
    assert start_3 == 20.0
    assert node.available_time == 22.0
