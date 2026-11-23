import os
from typing import Tuple, Dict, Any, List
import numpy as np
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.utils.seed import set_global_seed

def harvest_predictor_data(
    num_samples: int = 50000,
    output_path: str = "data/processed/predictor_training_data.npz",
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Simulates multi-policy exploratory continuum workloads under varied arrival rates
    and queue conditions to collect 50,000 state-outcome tuples S_i -> y_{miss,i} (FR-010, T5-01).
    """
    set_global_seed(seed)
    rng = np.random.RandomState(seed)

    arrival_rates = [20.0, 30.0, 45.0, 60.0]
    samples_per_run = 5000
    runs = num_samples // samples_per_run

    states: List[np.ndarray] = []
    labels: List[int] = []

    print(f"Collecting {num_samples} exploratory simulation steps across {runs} runs...")

    for r in range(runs):
        rate = float(rng.choice(arrival_rates))
        env = HealthcareEdgeEnv(seed=seed + r, arrival_rate=rate, max_tasks=samples_per_run)
        obs, _ = env.reset(seed=seed + r)

        for step_idx in range(samples_per_run):
            task = env.tasks[step_idx]
            current_obs = obs.copy()

            # Multi-regime exploration policy:
            # 1. 30% Cloud offloading (learns WAN delay boundaries vs tight deadlines)
            # 2. 25% Overloaded edge exploration (learns queue buildup impact)
            # 3. 25% Random action sampling (broad exploration)
            # 4. 20% Shortest queue first (normal clean state)
            strat = rng.rand()
            if strat < 0.30:
                action = 25  # Central Cloud
            elif strat < 0.55:
                # Target currently busy node to simulate hotspot queue congestion
                busy_nodes = np.argsort(obs[0:25])[-3:]  # Top 3 busiest nodes
                action = int(rng.choice(busy_nodes))
            elif strat < 0.80:
                action = int(rng.randint(0, 26))
            else:
                action = int(np.argmin(obs[0:25]))

            obs, reward, term, trunc, info = env.step(action)
            is_violated = 1 if info.get("step_violated", False) else 0

            states.append(current_obs)
            labels.append(is_violated)

            if term:
                break

    states_arr = np.array(states, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32)

    total = len(states_arr)
    indices = rng.permutation(total)

    n_train = int(0.70 * total)
    n_val = int(0.15 * total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_x, train_y = states_arr[train_idx], labels_arr[train_idx]
    val_x, val_y = states_arr[val_idx], labels_arr[val_idx]
    test_x, test_y = states_arr[test_idx], labels_arr[test_idx]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "test_x": test_x, "test_y": test_y
    }

    np.savez_compressed(output_path, **dataset)
    print(f"Data harvesting complete! Saved {total} samples to {output_path}")
    print(f"Train: {len(train_x)} (Violations: {np.sum(train_y):.0f} / {np.sum(train_y)/len(train_y)*100:.2f}%)")
    print(f"Val:   {len(val_x)} (Violations: {np.sum(val_y):.0f} / {np.sum(val_y)/len(val_y)*100:.2f}%)")
    print(f"Test:  {len(test_x)} (Violations: {np.sum(test_y):.0f} / {np.sum(test_y)/len(test_y)*100:.2f}%)")

    return dataset

if __name__ == "__main__":
    harvest_predictor_data()
