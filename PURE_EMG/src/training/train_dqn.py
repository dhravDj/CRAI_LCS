import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.dqn_scheduler import DuelingDQN
from src.models.deadline_predictor import DeadlineMissPredictor
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.reward_shaping import compute_reward
from src.simulator.edge_cloud_env import HealthcareEdgeEnv
from src.utils.seed import set_global_seed

def train_double_dqn(
    total_episodes: int = 50,
    tasks_per_episode: int = 1000,
    batch_size: int = 64,
    gamma_discount: float = 0.99,
    lr: float = 1e-3,
    buffer_capacity: int = 50000,
    warmup_steps: int = 1000,
    target_sync_freq: int = 200,
    use_polyak: bool = True,
    tau: float = 0.005,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 25000,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    predictor_checkpoint: str = "checkpoints/deadline_predictor.pt",
    seed: int = 42,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains 26-Action Double DQN Scheduling Agent with Prioritized Experience Replay
    and Multi-Objective Reward Shaping (OBJ-5, FR-013, FR-014, T6-01..T6-03).
    """
    set_global_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    dev = torch.device(device_str)
    print(f"================================================================")
    print(f" Starting Double DQN Training on hardware device: {dev}")
    print(f" Episodes: {total_episodes} | Tasks/Episode: {tasks_per_episode}")
    print(f" Target sync frequency: every {target_sync_freq} steps")
    print(f" Exploration schedule: {eps_start} -> {eps_end} over {eps_decay_steps} steps")
    print(f"================================================================")

    # 1. Initialize Online Policy Network and Target Network
    policy_net = DuelingDQN(state_dim=54, action_dim=26).to(dev)
    target_net = DuelingDQN(state_dim=54, action_dim=26).to(dev)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    print(f"Policy Network parameters: {policy_net.total_parameters:,} (Budget: ~0.32M)")

    # 2. Load Optional Pre-Execution Deadline Violation Predictor (Phase 5)
    predictor: Optional[DeadlineMissPredictor] = None
    if os.path.exists(predictor_checkpoint):
        try:
            p_model = DeadlineMissPredictor(in_features=54).to(dev)
            ckpt = torch.load(predictor_checkpoint, map_location=dev, weights_only=True)
            if "model_state_dict" in ckpt:
                p_model.load_state_dict(ckpt["model_state_dict"])
            else:
                p_model.load_state_dict(ckpt)
            p_model.eval()
            predictor = p_model
            print(f"Loaded pre-execution deadline predictor from {predictor_checkpoint}")
        except Exception as e:
            print(f"Warning: could not load predictor checkpoint: {e}")

    # 3. Initialize Prioritized Experience Replay Buffer and Optimizer
    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_capacity,
        state_dim=54,
        alpha=0.6,
        seed=seed
    )
    optimizer = Adam(policy_net.parameters(), lr=lr, eps=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss for gradient stability (RSK-007)

    # Training telemetry
    total_steps = 0
    best_reward = -float("inf")
    best_violation_rate = float("inf")
    saved_checkpoint_path = os.path.join(checkpoint_dir, "dqn_scheduler.pt")

    history = {
        "episode": [],
        "reward": [],
        "violation_rate": [],
        "risk_coverage": [],
        "avg_loss": [],
        "epsilon": []
    }

    start_time = time.time()

    for ep in range(1, total_episodes + 1):
        env = HealthcareEdgeEnv(
            seed=seed + ep,
            arrival_rate=20.0,
            max_tasks=tasks_per_episode
        )
        obs, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        ep_losses = []

        # Run episode
        for step_idx in range(tasks_per_episode):
            total_steps += 1

            # Dynamic Epsilon & Beta Annealing
            eps = max(
                eps_end,
                eps_start - (eps_start - eps_end) * (total_steps / float(eps_decay_steps))
            )
            beta = min(
                beta_end,
                beta_start + (beta_end - beta_start) * (total_steps / float(eps_decay_steps))
            )

            # Enrich observation with deadline predictor if available
            if predictor is not None:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                    p_miss = float(predictor(obs_t).squeeze().item())
                obs[51] = p_miss
                if env.current_step < len(env.tasks):
                    env.tasks[env.current_step].predicted_miss_prob = p_miss

            # Action Selection: epsilon-greedy
            current_obs = obs.copy()
            action = policy_net.act(current_obs, epsilon=eps, device=dev)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # If predictor is active, enrich next_obs as well
            if predictor is not None and not terminated:
                with torch.no_grad():
                    next_t = torch.from_numpy(next_obs).float().unsqueeze(0).to(dev)
                    next_p_miss = float(predictor(next_t).squeeze().item())
                next_obs[51] = next_p_miss
                if env.current_step < len(env.tasks):
                    env.tasks[env.current_step].predicted_miss_prob = next_p_miss

            # Store transition in Prioritized Replay Buffer
            replay_buffer.push(
                state=current_obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=terminated
            )

            obs = next_obs

            # Optimization Step (if warmup completed)
            if len(replay_buffer) >= warmup_steps:
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_weights, b_indices = replay_buffer.sample(
                    batch_size=batch_size,
                    beta=beta,
                    device=dev
                )

                # Current Q-values: Q(s, a; theta)
                current_q = policy_net(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Double DQN Target Calculation (Equations 30-31):
                # 1. Greedy action selection from Online Network theta: a* = argmax_a Q(s', a; theta)
                # 2. Value evaluation from Target Network theta^-: Q(s', a*; theta^-)
                with torch.no_grad():
                    next_online_actions = policy_net(b_next_states).argmax(dim=-1, keepdim=True)
                    next_target_q = target_net(b_next_states).gather(1, next_online_actions).squeeze(1)
                    target_q = b_rewards + (1.0 - b_dones) * gamma_discount * next_target_q

                # Bellman TD error & Huber loss weighted by PER importance sampling
                td_errors = torch.abs(current_q - target_q).detach()
                elementwise_loss = loss_fn(current_q, target_q)
                weighted_loss = torch.mean(elementwise_loss * b_weights)

                optimizer.zero_grad()
                weighted_loss.backward()
                # Gradient clipping to prevent exploding gradients (RSK-007)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                # Update Priorities in Replay Buffer
                replay_buffer.update_priorities(b_indices, td_errors)

                ep_losses.append(weighted_loss.item())

                # Synchronize Target Network (Polyak soft update or periodic hard sync)
                if use_polyak:
                    for target_p, online_p in zip(target_net.parameters(), policy_net.parameters()):
                        target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
                elif total_steps % target_sync_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if terminated:
                break

        # Episode Metrics
        info = env._get_info()
        v_rate = info.get("violation_rate", 0.0)
        r_cov = info.get("risk_coverage", 100.0)
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["violation_rate"].append(v_rate)
        history["risk_coverage"].append(r_cov)
        history["avg_loss"].append(mean_loss)
        history["epsilon"].append(eps)

        if ep % 5 == 0 or ep == 1 or ep == total_episodes:
            elapsed = time.time() - start_time
            print(
                f"Ep [{ep:03d}/{total_episodes:03d}] | Steps: {total_steps:05d} | "
                f"Eps: {eps:.3f} | Reward: {ep_reward:+8.2f} | "
                f"Loss: {mean_loss:.4f} | Violations: {v_rate:5.2f}% | "
                f"Risk Coverage: {r_cov:5.1f}% | Time: {elapsed:.1f}s"
            )

        # Checkpoint: Save best policy (highest reward or lowest violations)
        if ep_reward > best_reward or ep == total_episodes:
            best_reward = ep_reward
            torch.save({
                "episode": ep,
                "total_steps": total_steps,
                "model_state_dict": policy_net.state_dict(),
                "reward": ep_reward,
                "violation_rate": v_rate,
                "risk_coverage": r_cov,
                "history": history
            }, saved_checkpoint_path)

    total_training_time = time.time() - start_time
    print(f"\n================================================================")
    print(f" Training complete in {total_training_time:.2f}s!")
    print(f" Best Episode Reward: {best_reward:+.2f}")
    print(f" Final Violation Rate: {history['violation_rate'][-1]:.2f}%")
    print(f" Final Risk Coverage: {history['risk_coverage'][-1]:.1f}%")
    print(f" Checkpoint saved to: {saved_checkpoint_path}")
    print(f"================================================================")

    return {
        "checkpoint_path": saved_checkpoint_path,
        "total_training_time": total_training_time,
        "final_reward": history["reward"][-1],
        "final_violation_rate": history["violation_rate"][-1],
        "final_risk_coverage": history["risk_coverage"][-1],
        "history": history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Double DQN Scheduling Agent")
    parser.add_argument("--episodes", type=int, default=50, help="Total training episodes")
    parser.add_argument("--tasks", type=int, default=1000, help="Tasks per episode")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_double_dqn(
        total_episodes=args.episodes,
        tasks_per_episode=args.tasks,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device
    )
