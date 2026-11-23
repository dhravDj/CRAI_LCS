"""
U-CRAI-LCS: Universal Multi-Tier Healthcare IoT Continuum Simulation Environment
Supports heterogeneous multi-disease telemetry workloads with variable payload sizes (200B to 80KB),
differential CPU demands, M/M/c queuing, wireless Rayleigh fading, and DVFS power physics.
Gymnasium-compatible (v1.0.0+ specification).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.universal.schema import ModalityType, UniversalTelemetryTask
from src.simulator.nodes import EdgeNode, CloudInstance, create_continuum_nodes
from src.simulator.latency_energy import compute_edge_latency_and_energy, compute_cloud_latency_and_energy


class UniversalHealthcareContinuumEnv(gym.Env):
    """
    Farama Gymnasium-compatible Continuum Environment for Universal Multi-Disease IoT Telemetry.
    Models 200 IoT sensors across 3 hospital departments (Cardiology, Neurology, Pulmonology),
    25 heterogeneous Edge Gateways, and 2 Cloud Instances.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_edge_nodes: int = 25,
        num_cloud_instances: int = 2,
        seed: int = 42
    ):
        super().__init__()
        self.seed_val = seed
        self.rng = np.random.RandomState(seed)

        # 25 Edge Nodes (heterogeneous 1.2 to 2.4 GHz) + 2 Cloud Instances (3.2 GHz)
        self.edge_nodes, self.cloud_instances = create_continuum_nodes(
            num_edge=num_edge_nodes,
            num_cloud=num_cloud_instances,
            seed=seed
        )
        self.total_nodes = len(self.edge_nodes) + len(self.cloud_instances)

        # Action space: 0..24 = Edge Gateway j, 25 = Cloud Instance
        self.action_space = spaces.Discrete(self.total_nodes - 1) # 26 actions (0..24 Edge, 25 Cloud)

        # Observation space (54 dimensions):
        # - 25 edge node utilizations [0, 1]
        # - 25 edge node queue length normalized [0, 1]
        # - 2 cloud instance queue/utilization states
        # - 2 task features: [log10(payload), U_i]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(54,),
            dtype=np.float32
        )

        self.current_step = 0
        self.current_time = 0.0
        self.tasks: List[UniversalTelemetryTask] = []
        self.completed_tasks: List[UniversalTelemetryTask] = []

        # Continuum tracking metrics
        self.total_violations = 0
        self.total_urgent = 0
        self.urgent_on_time = 0
        self.total_latency_ms = 0.0
        self.total_energy_j = 0.0
        self.fail_safe_count = 0

    def set_workload(self, tasks: List[UniversalTelemetryTask]):
        """Injects a sequence of multi-modal clinical tasks into the simulation."""
        self.tasks = tasks
        self.current_step = 0
        self.current_time = 0.0
        self.completed_tasks.clear()
        self.total_violations = 0
        self.total_urgent = 0
        self.urgent_on_time = 0
        self.total_latency_ms = 0.0
        self.total_energy_j = 0.0
        self.fail_safe_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.current_time = 0.0
        self.completed_tasks.clear()
        self.total_violations = 0
        self.total_urgent = 0
        self.urgent_on_time = 0
        self.total_latency_ms = 0.0
        self.total_energy_j = 0.0
        self.fail_safe_count = 0

        # Reset continuum nodes
        for node in self.edge_nodes:
            node.reset()
        for cloud in self.cloud_instances:
            cloud.reset()

        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(54, dtype=np.float32)

        # 25 Edge Node Utilizations & Queues
        for i, node in enumerate(self.edge_nodes):
            obs[i] = node.get_utilization(self.current_time)
            obs[25 + i] = min(1.0, node.get_queue_length(self.current_time) / 20.0)

        # 2 Cloud States (normalized queue delay)
        for i, cloud in enumerate(self.cloud_instances):
            delay = cloud.get_queue_delay(self.current_time, self.current_time)
            obs[50 + i] = min(1.0, delay / 1.0)

        # Task features
        if self.current_step < len(self.tasks):
            t = self.tasks[self.current_step]
            # Log10 normalized payload (200B to 100KB -> 2.3 to 5.0 -> [0, 1])
            obs[52] = float(np.clip((np.log10(max(10.0, t.payload_bytes)) - 2.0) / 3.0, 0.0, 1.0))
            obs[53] = float(t.clinical_urgency)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.current_step >= len(self.tasks):
            return self._get_observation(), 0.0, True, False, self._get_info()

        task = self.tasks[self.current_step]
        task.assigned_node = action
        self.current_time = max(self.current_time, task.arrival_timestamp_s)

        # Map UniversalTelemetryTask to HealthcareTask execution signatures
        # (delta_i = payload_bytes, psi_i = computational_demand_cycles)
        from src.simulator.task import HealthcareTask
        compat_task = HealthcareTask(
            task_id=task.task_id,
            delta_i=float(task.payload_bytes),
            psi_i=float(task.computational_demand_cycles),
            deadline_ms=float(task.deadline_ms),
            clinical_risk=float(task.clinical_urgency),
            arrival_time=float(task.arrival_timestamp_s),
            predicted_miss_prob=float(task.predicted_miss_prob)
        )

        if action < len(self.edge_nodes):
            # Scheduled to Edge Gateway action
            node = self.edge_nodes[action]
            t_comm, t_queue, t_exec, lat_ms, energy_j = compute_edge_latency_and_energy(
                task=compat_task,
                node=node,
                current_time=self.current_time
            )
            node.schedule_task(compat_task, task.arrival_timestamp_s + t_comm, t_exec, energy_j)
        else:
            # Scheduled to Central Cloud (pick least busy instance)
            c_idx = 0 if self.cloud_instances[0].available_time <= self.cloud_instances[1].available_time else 1
            cloud = self.cloud_instances[c_idx]
            t_comm, t_queue, t_exec, lat_ms, energy_j = compute_cloud_latency_and_energy(
                task=compat_task,
                cloud=cloud,
                current_time=self.current_time
            )
            cloud.schedule_task(compat_task, task.arrival_timestamp_s + t_comm, t_exec, energy_j)

        task.actual_latency_ms = lat_ms
        task.energy_consumed_j = energy_j
        task.is_violated = (lat_ms > task.deadline_ms)

        # Track continuum statistics
        self.total_latency_ms += lat_ms
        self.total_energy_j += energy_j
        if task.is_violated:
            self.total_violations += 1
        if task.is_urgent:
            self.total_urgent += 1
            if not task.is_violated:
                self.urgent_on_time += 1

        self.completed_tasks.append(task)
        self.current_step += 1

        # Equation 29 Multi-Objective Reward (on-time urgent edge execution yields positive bonus):
        reward = -(
            0.4 * (lat_ms / 100.0) +
            0.2 * (energy_j / 5.0) +
            2.0 * (1.0 if task.is_violated else 0.0) -
            (1.5 * task.clinical_urgency if (action < 25 and not task.is_violated) else 0.0)
        )

        terminated = self.current_step >= len(self.tasks)
        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        n = len(self.completed_tasks)
        if n == 0:
            return {"avg_latency_ms": 0.0, "violation_rate": 0.0, "risk_coverage": 100.0, "total_energy_j": 0.0}
        return {
            "total_tasks": n,
            "avg_latency_ms": self.total_latency_ms / n,
            "violation_rate": (self.total_violations / n) * 100.0,
            "risk_coverage": (self.urgent_on_time / max(1, self.total_urgent)) * 100.0,
            "total_energy_j": self.total_energy_j,
            "fail_safe_count": self.fail_safe_count
        }
