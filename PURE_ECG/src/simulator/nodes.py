from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.simulator.task import HealthcareTask

class EdgeNode:
    """
    Heterogeneous Edge Gateway in the 3-Tier HIoT Continuum (Table 4, FR-007).
    Deploys 1D-CNN-LSTM risk inference and local task execution.
    """
    def __init__(
        self,
        node_id: int,
        mips: float,
        ram_gb: float,
        bandwidth_mbps: float,
        tx_latency_ms: float,
        idle_power_w: float,
        active_power_w: float,
        capacity_score: float = 0.0
    ):
        self.node_id = node_id
        self.mips = mips                          # f_j in MIPS [800.0, 4500.0]
        self.ram_gb = ram_gb                      # m_j in GB [16.0, 64.0]
        self.bandwidth_mbps = bandwidth_mbps      # B_j in Mbps [10.0, 50.0]
        self.tx_latency_ms = tx_latency_ms        # Wireless transmission baseline [3.0, 12.0] ms
        self.idle_power_w = idle_power_w          # P_idle in Watts [40.0, 80.0] W
        self.active_power_w = active_power_w      # P_active in Watts [100.0, 200.0] W
        self.capacity_score = capacity_score      # Normalized capacity a_j in [0.0, 1.0]

        # Dynamic state & queue management (RSK-006 anti-clock-desynchronization)
        self.available_time = 0.0                 # Timestamp when this node becomes idle (seconds)
        self.total_tasks_processed = 0
        self.total_busy_time = 0.0
        self.total_energy_j = 0.0

    @property
    def ips(self) -> float:
        """Instructions / cycles per second."""
        return self.mips * 1.0e6

    def get_queue_delay(self, current_time: float, arrival_at_node: float) -> float:
        """
        Calculates queuing delay (seconds) for a task arriving at this node at 'arrival_at_node'.
        Delay = max(0.0, available_time - arrival_at_node)
        """
        return max(0.0, self.available_time - arrival_at_node)

    def schedule_task(
        self,
        task: HealthcareTask,
        arrival_at_node: float,
        exec_time_sec: float,
        energy_j: float
    ) -> float:
        """
        Enqueues and processes task on this edge node.
        Updates available_time = max(available_time, arrival_at_node) + exec_time_sec.
        Returns execution start time.
        """
        # If task is preempted by clinical fail-safe protocol, execution starts immediately
        if getattr(task, "is_preempted", False):
            start_time = arrival_at_node
            finish_time = start_time + exec_time_sec
            self.available_time = max(self.available_time, finish_time)
        else:
            start_time = max(self.available_time, arrival_at_node)
            finish_time = start_time + exec_time_sec
            self.available_time = finish_time
        self.total_busy_time += exec_time_sec
        self.total_tasks_processed += 1
        self.total_energy_j += energy_j

        return start_time

    def get_utilization(self, current_time: float) -> float:
        """
        Calculates cumulative processor utilization U_j in [0.0, 1.0].
        """
        if current_time <= 0.0:
            return 0.0
        return min(1.0, self.total_busy_time / max(current_time, 1e-6))

    def get_queue_length(self, current_time: float) -> float:
        """
        Returns estimated remaining queue workload (normalized to cycles/1e7).
        """
        remaining_time = max(0.0, self.available_time - current_time)
        remaining_cycles = remaining_time * self.ips
        return remaining_cycles / 1.0e7

    def reset(self):
        self.available_time = 0.0
        self.total_tasks_processed = 0
        self.total_busy_time = 0.0
        self.total_energy_j = 0.0


class CloudInstance:
    """
    Centralized Cloud Datacenter Instance in the 3-Tier HIoT Continuum (Table 4, FR-008).
    Executes global orchestration, MLP deadline violation forecasting, and Double DQN.
    """
    def __init__(
        self,
        instance_id: int,
        mips: float,
        ram_gb: float,
        bandwidth_mbps: float,
        wan_latency_ms: float,
        tx_latency_ms: float,
        idle_power_w: float,
        active_power_w: float
    ):
        self.instance_id = instance_id
        self.mips = mips                          # f_c in MIPS [12000.0, 35000.0]
        self.ram_gb = ram_gb                      # 128.0 GB
        self.bandwidth_mbps = bandwidth_mbps      # B_c in Mbps [100.0, 1000.0]
        self.wan_latency_ms = wan_latency_ms      # T_{e->c} WAN propagation delay [45.0, 120.0] ms
        self.tx_latency_ms = tx_latency_ms        # Local wireless hop baseline [3.0, 12.0] ms
        self.idle_power_w = idle_power_w          # P_idle in Watts [180.0, 350.0] W
        self.active_power_w = active_power_w      # P_active in Watts [400.0, 750.0] W

        # Dynamic state & queue management
        self.available_time = 0.0
        self.total_tasks_processed = 0
        self.total_busy_time = 0.0
        self.total_energy_j = 0.0

    @property
    def ips(self) -> float:
        return self.mips * 1.0e6

    def get_queue_delay(self, current_time: float, arrival_at_cloud: float) -> float:
        return max(0.0, self.available_time - arrival_at_cloud)

    def schedule_task(
        self,
        task: HealthcareTask,
        arrival_at_cloud: float,
        exec_time_sec: float,
        energy_j: float
    ) -> float:
        start_time = max(self.available_time, arrival_at_cloud)
        finish_time = start_time + exec_time_sec

        self.available_time = finish_time
        self.total_busy_time += exec_time_sec
        self.total_tasks_processed += 1
        self.total_energy_j += energy_j

        return start_time

    def get_utilization(self, current_time: float) -> float:
        if current_time <= 0.0:
            return 0.0
        return min(1.0, self.total_busy_time / max(current_time, 1e-6))

    def reset(self):
        self.available_time = 0.0
        self.total_tasks_processed = 0
        self.total_busy_time = 0.0
        self.total_energy_j = 0.0


def create_continuum_nodes(
    num_edge: int = 25,
    num_cloud: int = 2,
    seed: int = 42
) -> Tuple[List[EdgeNode], List[CloudInstance]]:
    """
    Instantiates the 25 heterogeneous Edge Gateways and 2 Cloud Instances
    strictly adhering to Table 4, FR-007, and FR-008.
    """
    rng = np.random.RandomState(seed)

    # Edge Nodes Generation (FR-007)
    edge_nodes = []
    raw_mips = rng.uniform(800.0, 4500.0, size=num_edge)
    raw_ram = rng.choice([16.0, 32.0, 64.0], size=num_edge)
    raw_bw = rng.uniform(10.0, 50.0, size=num_edge)
    raw_tx = rng.uniform(3.0, 12.0, size=num_edge)
    raw_p_idle = rng.uniform(40.0, 80.0, size=num_edge)

    # Max limits for normalized capacity score (Equations 4-7)
    f_max = 4500.0
    m_max = 64.0
    w_c = 0.50
    w_m = 0.50

    for j in range(num_edge):
        mips_j = float(raw_mips[j])
        ram_j = float(raw_ram[j])
        bw_j = float(raw_bw[j])
        tx_j = float(raw_tx[j])
        p_idle = float(raw_p_idle[j])
        p_active = p_idle * 2.2  # Active power factor

        # Capacity score Eq. (7)
        a_j = float(w_c * (mips_j / f_max) + w_m * (ram_j / m_max))

        node = EdgeNode(
            node_id=j,
            mips=mips_j,
            ram_gb=ram_j,
            bandwidth_mbps=bw_j,
            tx_latency_ms=tx_j,
            idle_power_w=p_idle,
            active_power_w=p_active,
            capacity_score=a_j
        )
        edge_nodes.append(node)

    # Cloud Instances Generation (FR-008)
    cloud_instances = []
    cloud_mips = [15000.0, 30000.0] if num_cloud == 2 else list(rng.uniform(12000.0, 35000.0, size=num_cloud))

    for c in range(num_cloud):
        mips_c = float(cloud_mips[c])
        ram_c = 128.0
        bw_c = float(rng.uniform(100.0, 1000.0))
        wan_c = float(rng.uniform(45.0, 120.0))
        tx_c = float(rng.uniform(3.0, 12.0))
        p_idle = float(rng.uniform(180.0, 350.0))
        p_active = p_idle * 2.0

        instance = CloudInstance(
            instance_id=c,
            mips=mips_c,
            ram_gb=ram_c,
            bandwidth_mbps=bw_c,
            wan_latency_ms=wan_c,
            tx_latency_ms=tx_c,
            idle_power_w=p_idle,
            active_power_w=p_active
        )
        cloud_instances.append(instance)

    return edge_nodes, cloud_instances
