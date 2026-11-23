from typing import Tuple, Union
from src.simulator.task import HealthcareTask
from src.simulator.nodes import EdgeNode, CloudInstance

def compute_edge_latency_and_energy(
    task: HealthcareTask,
    node: EdgeNode,
    current_time: float
) -> Tuple[float, float, float, float, float]:
    """
    Computes exact transmission delay, queueing delay, execution delay, total latency,
    and energy consumption for offloading a task to an Edge Node (Equations 8 & 10).

    Returns:
        (t_comm_sec, t_queue_sec, t_exec_sec, total_latency_ms, energy_consumed_j)
    """
    # 1. Transmission Latency: Local wireless hop (T_tr in ms + payload transmission)
    t_tr_sec = node.tx_latency_ms / 1000.0
    payload_bits = task.delta_i * 8.0
    bandwidth_bps = node.bandwidth_mbps * 1.0e6
    t_comm_sec = t_tr_sec + (payload_bits / bandwidth_bps)

    # Arrival timestamp at edge gateway
    arrival_at_node = task.arrival_time + t_comm_sec

    # 2. Queueing Delay (RSK-006: discrete-event clock preservation)
    # If task is preempted by clinical fail-safe protocol, queue delay is bypassed
    if getattr(task, "is_preempted", False):
        t_queue_sec = 0.0
    else:
        t_queue_sec = node.get_queue_delay(current_time=current_time, arrival_at_node=arrival_at_node)

    # 3. Execution Delay: psi_i / (f_j * 1e6)
    t_exec_sec = task.psi_i / node.ips

    # 4. Total End-to-End Latency: T_comm + T_queue + T_exec
    total_latency_sec = t_comm_sec + t_queue_sec + t_exec_sec
    total_latency_ms = total_latency_sec * 1000.0

    # 5. Energy Model: E = P_active * T_exec + P_tx_sensor * T_comm
    p_tx_sensor = 0.50  # 0.5W IoT sensor RF transmission power
    energy_consumed_j = (node.active_power_w * t_exec_sec) + (p_tx_sensor * t_comm_sec)

    return t_comm_sec, t_queue_sec, t_exec_sec, total_latency_ms, energy_consumed_j


def compute_cloud_latency_and_energy(
    task: HealthcareTask,
    cloud: CloudInstance,
    current_time: float
) -> Tuple[float, float, float, float, float]:
    """
    Computes exact transmission delay, WAN transit delay, queueing delay, execution delay,
    total latency, and energy consumption for offloading a task to the Cloud (Equations 9 & 10).

    Returns:
        (t_comm_sec, t_queue_sec, t_exec_sec, total_latency_ms, energy_consumed_j)
    """
    # 1. Transmission Latency: Local wireless hop + WAN Transit Latency (T_{e->c})
    t_tr_sec = cloud.tx_latency_ms / 1000.0
    t_wan_sec = cloud.wan_latency_ms / 1000.0
    payload_bits = task.delta_i * 8.0
    bandwidth_bps = cloud.bandwidth_mbps * 1.0e6
    t_comm_sec = t_tr_sec + t_wan_sec + (payload_bits / bandwidth_bps)

    # Arrival timestamp at cloud instance
    arrival_at_cloud = task.arrival_time + t_comm_sec

    # 2. Queueing Delay
    t_queue_sec = cloud.get_queue_delay(current_time=current_time, arrival_at_cloud=arrival_at_cloud)

    # 3. Execution Delay: psi_i / (f_c * 1e6)
    t_exec_sec = task.psi_i / cloud.ips

    # 4. Total End-to-End Latency
    total_latency_sec = t_comm_sec + t_queue_sec + t_exec_sec
    total_latency_ms = total_latency_sec * 1000.0

    # 5. Energy Model (Cloud server compute + WAN router transmission)
    p_tx_wan = 2.50  # 2.5W WAN backhaul gateway transmission power
    energy_consumed_j = (cloud.active_power_w * t_exec_sec) + (p_tx_wan * t_comm_sec)

    return t_comm_sec, t_queue_sec, t_exec_sec, total_latency_ms, energy_consumed_j
