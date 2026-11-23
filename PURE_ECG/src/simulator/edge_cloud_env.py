import os
import yaml
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.simulator.task import HealthcareTask, create_synthetic_task
from src.simulator.nodes import EdgeNode, CloudInstance, create_continuum_nodes
from src.simulator.latency_energy import (
    compute_edge_latency_and_energy,
    compute_cloud_latency_and_energy
)

class HealthcareEdgeEnv(gym.Env):
    """
    Farama Gymnasium-compatible Discrete-Event Continuum Simulation Environment (OBJ-1, FR-007..FR-011).
    Models 200 IoT sensors, 25 heterogeneous Edge Gateways, and 2 Cloud Instances.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: Optional[str] = None,
        seed: int = 42,
        arrival_rate: float = 20.0,
        max_tasks: int = 1000
    ):
        super().__init__()
        self.seed_val = seed
        self.arrival_rate = arrival_rate
        self.max_tasks = max_tasks
        self.rng = np.random.RandomState(seed)

        # Load YAML config if supplied, else sensible defaults matching Table 3 & 4
        self.config = self._load_config(config_path)

        # Initialize physical 3-tier continuum nodes (FR-007, FR-008)
        self.edge_nodes, self.cloud_instances = create_continuum_nodes(
            num_edge=25,
            num_cloud=2,
            seed=self.seed_val
        )

        # Action Space: 26 discrete actions (0-24: Edge Nodes, 25: Central Cloud)
        self.action_space = spaces.Discrete(26)

        # Observation Space: 54 dimensions (25 Queues, 25 Utilizations, R_i, P_{miss,i}, D_i, psi_i)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(54,),
            dtype=np.float32
        )

        # Multi-Objective Reward Weights (RW-2 Balanced default from Table 10)
        self.alpha = float(self.config.get("reward", {}).get("alpha_latency", 0.35))
        self.beta = float(self.config.get("reward", {}).get("beta_miss", 0.25))
        self.gamma = float(self.config.get("reward", {}).get("gamma_risk", 0.30))
        self.eta = float(self.config.get("reward", {}).get("eta_utilization", 0.10))

        # Dynamic simulation tracking
        self.tasks: List[HealthcareTask] = []
        self.current_step = 0
        self.current_time = 0.0

        # Performance Metrics Accumulators
        self.completed_tasks: List[HealthcareTask] = []
        self.total_latency_ms = 0.0
        self.total_violations = 0
        self.total_urgent_tasks = 0
        self.urgent_on_time_tasks = 0
        self.total_energy_consumed = 0.0

        # Pre-generate workload stream
        self._generate_workload_stream()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {
            "reward": {
                "alpha_latency": 0.35,
                "beta_miss": 0.25,
                "gamma_risk": 0.30,
                "eta_utilization": 0.10
            }
        }

    def _generate_workload_stream(self):
        """
        Generates Poisson task arrival stream: Delta_t ~ Exponential(1 / lambda).
        """
        self.tasks.clear()
        t_arr = 0.0

        for i in range(self.max_tasks):
            # Inter-arrival exponential step
            inter_arrival = self.rng.exponential(1.0 / self.arrival_rate)
            t_arr += inter_arrival
            task = create_synthetic_task(
                task_id=i,
                arrival_time=t_arr,
                rng=self.rng
            )
            self.tasks.append(task)

    def set_workload(self, task_list: List[HealthcareTask]):
        """Sets an external clinical task stream (e.g., from MIT-BIH pipeline)."""
        self.tasks = task_list
        self.max_tasks = len(task_list)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.RandomState(seed)
            self.edge_nodes, self.cloud_instances = create_continuum_nodes(
                num_edge=25,
                num_cloud=2,
                seed=self.seed_val
            )
            self._generate_workload_stream()

        # Reset all nodes
        for node in self.edge_nodes:
            node.reset()
        for cloud in self.cloud_instances:
            cloud.reset()

        self.current_step = 0
        self.current_time = 0.0
        self.completed_tasks.clear()
        self.total_latency_ms = 0.0
        self.total_violations = 0
        self.total_urgent_tasks = 0
        self.urgent_on_time_tasks = 0
        self.total_energy_consumed = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the 54-dimensional state vector (Equation 27):
        [Q_1..Q_25 (25), U_1..U_25 (25), R_i (1), P_{miss,i} (1), D_i (1), psi_i (1)]
        """
        obs = np.zeros(54, dtype=np.float32)

        # 1. 25 Edge Node Queue Lengths
        for j in range(25):
            obs[j] = self.edge_nodes[j].get_queue_length(self.current_time)

        # 2. 25 Edge Node Processor Utilizations
        for j in range(25):
            obs[25 + j] = self.edge_nodes[j].get_utilization(self.current_time)

        # 3. Current Task Features
        if self.current_step < len(self.tasks):
            curr_task = self.tasks[self.current_step]
            obs[50] = curr_task.clinical_risk
            obs[51] = curr_task.predicted_miss_prob
            obs[52] = curr_task.deadline_seconds  # in seconds for numerical scale
            obs[53] = curr_task.psi_i / 1.0e7     # normalized CPU cycles ~ [0.5, 2.0]
        else:
            obs[50:54] = 0.0

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes scheduling action a_t in {0..25}:
        - 0..24: Dispatch to Edge Gateway j
        - 25: Dispatch to Cloud Instance
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.current_step >= len(self.tasks):
            # Terminated
            return self._get_observation(), 0.0, True, False, self._get_info()

        task = self.tasks[self.current_step]
        task.assigned_node = action

        # Advance current time to task arrival time if it has advanced
        self.current_time = max(self.current_time, task.arrival_time)

        # Compute Latency, Energy, and Execution
        if action < 25:
            # Scheduled to Edge Node action
            target_node = self.edge_nodes[action]
            t_comm, t_queue, t_exec, lat_ms, energy_j = compute_edge_latency_and_energy(
                task=task,
                node=target_node,
                current_time=self.current_time
            )
            # Commit schedule to node
            start_time = target_node.schedule_task(
                task=task,
                arrival_at_node=task.arrival_time + t_comm,
                exec_time_sec=t_exec,
                energy_j=energy_j
            )
            finish_time = start_time + t_exec
        else:
            # Scheduled to Central Cloud (pick lowest queue instance among the 2)
            c_idx = 0 if self.cloud_instances[0].available_time <= self.cloud_instances[1].available_time else 1
            target_cloud = self.cloud_instances[c_idx]
            t_comm, t_queue, t_exec, lat_ms, energy_j = compute_cloud_latency_and_energy(
                task=task,
                cloud=target_cloud,
                current_time=self.current_time
            )
            start_time = target_cloud.schedule_task(
                task=task,
                arrival_at_cloud=task.arrival_time + t_comm,
                exec_time_sec=t_exec,
                energy_j=energy_j
            )
            finish_time = start_time + t_exec

        # Populate Task Lifecycle Attributes
        task.start_time = start_time
        task.finish_time = finish_time
        task.actual_latency_ms = lat_ms
        task.energy_consumed_j = energy_j
        is_violated = lat_ms > task.deadline_ms
        task.is_violated = is_violated

        # Update Metrics
        self.completed_tasks.append(task)
        self.total_latency_ms += lat_ms
        self.total_energy_consumed += energy_j

        if is_violated:
            self.total_violations += 1

        if task.is_urgent:
            self.total_urgent_tasks += 1
            if not is_violated:
                self.urgent_on_time_tasks += 1

        # Calculate Mean Edge Utilization across 25 nodes
        edge_utils = [node.get_utilization(self.current_time) for node in self.edge_nodes]
        mean_edge_util = float(np.mean(edge_utils))

        # Normalized Multi-Objective Reward Function (Equation 29, ADR-002)
        # Normalized Latency: L_i / D_i (~1.0 scale prevents RSK-007 gradient explosion)
        norm_latency = lat_ms / max(task.deadline_ms, 1e-3)
        on_time_flag = 1.0 if not is_violated else 0.0

        reward = -(
            self.alpha * norm_latency
            + self.beta * task.predicted_miss_prob
            - self.gamma * (task.clinical_risk * on_time_flag)
        ) + (self.eta * mean_edge_util)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.tasks)
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        info["step_latency_ms"] = lat_ms
        info["step_energy_j"] = energy_j
        info["step_violated"] = is_violated

        return obs, float(reward), terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        total_done = len(self.completed_tasks)
        if total_done == 0:
            return {
                "total_tasks": 0,
                "avg_latency_ms": 0.0,
                "violation_rate": 0.0,
                "risk_coverage": 100.0,
                "avg_edge_utilization": 0.0,
                "total_energy_j": 0.0,
                "energy_per_1k_tasks": 0.0
            }

        avg_lat = self.total_latency_ms / total_done
        violation_rate = (self.total_violations / total_done) * 100.0

        if self.total_urgent_tasks > 0:
            risk_coverage = (self.urgent_on_time_tasks / self.total_urgent_tasks) * 100.0
        else:
            risk_coverage = 100.0

        edge_utils = [node.get_utilization(self.current_time) for node in self.edge_nodes]
        avg_edge_util = float(np.mean(edge_utils)) * 100.0
        energy_per_1k = (self.total_energy_consumed / total_done) * 1000.0

        return {
            "total_tasks": total_done,
            "avg_latency_ms": round(avg_lat, 2),
            "violation_rate": round(violation_rate, 2),
            "risk_coverage": round(risk_coverage, 2),
            "avg_edge_utilization": round(avg_edge_util, 2),
            "total_energy_j": round(self.total_energy_consumed, 2),
            "energy_per_1k_tasks": round(energy_per_1k, 2)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Public getter for cumulative performance statistics."""
        return self._get_info()
