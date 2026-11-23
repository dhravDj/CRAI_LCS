# CRAI-LCS: Architecture Decision Record (ADR)

**Status:** APPROVED  
**Date:** 2026-11-21  
**Author:** AI Research & Engineering Team (Antigravity)  
**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

---

## 1. Context & Motivation

The CRAI-LCS paper introduces a closed-loop healthcare IoT scheduling paradigm that unites:
1. Real ECG physiological signals from PhysioNet MIT-BIH.
2. A CNN-LSTM model estimating clinical risk $R_i \in [0, 1]$.
3. A deadline violation predictor calculating $P_{miss,i} \in [0, 1]$.
4. A discrete-event edge-cloud simulator (25 edge nodes, 2 cloud instances, 200 IoT sensors).
5. A risk-conditioned Deep Q-Network (DQN) scheduler with multi-objective reward shaping.
6. A rigorous benchmarking suite against 5 baselines and ablation models across 10 random seeds.

To balance scientific fidelity, engineering maintainability, and implementation velocity, we audited 6 open-source GitHub repositories alongside standard scientific libraries (`wfdb`, `gymnasium`, `torch`). This ADR establishes our exact component selections and structural rationale.

---

## 2. Component Selection Decisions

### Decision 1: ECG Preprocessing & Task Workload Pipeline
* **Selected Foundation:** Hybrid approach combining `PanosLevedogiannis/ECG-Classification` (patient-aware record grouping) and `pr6thv3/ecg-ai-platform` (signal normalization).
* **Why Selected:**
  - `PanosLevedogiannis` prevents catastrophic patient data leakage by segregating patients by record ID.
  - `pr6thv3` provides clean Z-score normalization and high-throughput NumPy vectorization.
* **Paper-Specific Customization:** We explicitly enforce 5-second windowing at 360 Hz ($T = 1800$ samples, $\delta_i \approx 3.6$ KB) and map arrhythmia annotations to binary clinical ground truth ($y_i \in \{0, 1\}$).
* **Location in Project:** `src/ecg/dataset.py`, `src/ecg/preprocessing.py`, `src/ecg/task_generator.py`.

### Decision 2: CNN-LSTM Clinical Risk Estimator Architecture
* **Selected Foundation:** Custom PyTorch implementation referencing `mrrohan4321/ECG-Project-Deeplearning`.
* **Why Selected:** Off-the-shelf models from GitHub either lacked the exact parameter budget (0.74M parameters) or were coupled to Streamlit/web frameworks. Writing a clean, focused PyTorch model ensures exact compliance with paper equations (18)–(25).
* **Specifications:**
  - Layer 1: 1D Convolution (1 $\to$ 32 channels, kernel=5) + ReLU + MaxPool1D (stride=2).
  - Layer 2: 1D Convolution (32 $\to$ 64 channels, kernel=5) + ReLU + MaxPool1D (stride=2).
  - Layer 3: 2-layer LSTM (hidden_size=64, batch_first=True).
  - Layer 4: Linear (64 $\to$ 32) + ReLU + Dropout(0.2).
  - Output Layer: Linear (32 $\to$ 1) + Sigmoid $\to R_i \in [0, 1]$.
* **Optimization:** Adam ($\text{lr}=10^{-3}$, batch_size=32), Binary Cross-Entropy loss.
* **Location in Project:** `src/models/cnn_lstm_risk.py`.

### Decision 3: Edge-Cloud Simulation Engine
* **Selected Foundation:** Gymnasium-compatible discrete-event queueing engine adapted from `Dharmendra2022/DRL-EdgeCloud` and `PalashSharma19/Reinforcement-Learning-Job-Scheduling`.
* **Why Selected:** Farama Foundation's `gymnasium` provides standard RL interfaces (`step()`, `reset()`, `spaces.Box`, `spaces.Discrete`). Adapting the simulation queue dynamics allows exact implementation of equations (8)–(11) and (16).
* **Specifications:**
  - 25 Heterogeneous Edge Nodes: $f_j \in [800, 4500]$ MIPS, RAM $16-64$ GB, Base Power $40-80$ W.
  - 2 High-Performance Cloud Instances: $f_c \in [12000, 35000]$ MIPS, RAM $128$ GB, Base Power $180-350$ W.
  - Latency: Device-to-Edge $T_{tr} \in [3, 12]$ ms; Edge-to-Cloud $T_{e \to c} \in [45, 120]$ ms.
  - Workload Arrivals: Stochastic Poisson process $\lambda \in [8, 25]$ tasks/sec.
* **Location in Project:** `src/simulator/edge_cloud_env.py`, `src/simulator/nodes.py`.

### Decision 4: Deadline Violation Predictor Module
* **Selected Foundation:** Custom 3-layer MLP classifier ($0.18$M parameters, $<4$ ms inference).
* **Why Selected:** None of the open-source scheduling repositories had an explicit pre-execution deadline violation predictor. We author this module directly according to equation (12): $P_{miss,i} = f_\theta(S_i)$ where $S_i = \{U_t, Q_t, \psi_i, D_i, R_i\}$.
* **Location in Project:** `src/models/deadline_predictor.py`.

### Decision 5: DQN Scheduling Agent & Reward Shaping
* **Selected Foundation:** Double DQN (DDQN) architecture adapted from `PadalaSravan14/drl_edge_cloud_scheduler`.
* **Why Selected:** `PadalaSravan14` provides a clean implementation of the Replay Buffer and target network update mechanics. We adapt it to support:
  - State space ($53$-dim or compact $8$-dim observation vector): $\{Q_t, U_t, R_i, P_{miss,i}, D_i\}$.
  - Action space ($26$ discrete actions): Assigning to $E_1 \dots E_{25}$ or Cloud $C$.
  - Exact reward function (equation 29):
    $$R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$$
    with normalized latency and default coefficients $\alpha=0.35, \beta=0.25, \gamma=0.30, \eta=0.10$.
* **Location in Project:** `src/schedulers/dqn_scheduler.py`, `src/training/replay_buffer.py`.

### Decision 6: Baseline Scheduling Algorithms
* **Selected Foundation:** Unified baseline architecture adapted from `PalashSharma19/baselines.py` and `PadalaSravan14/baselines/`.
* **Why Selected:** `PalashSharma19` enforces a strict, stateless rule interface: `schedule(observation) -> int`. This guarantees 100% fair comparability across all methods under identical workload seeds.
* **Included Baselines:**
  1. `RandomScheduler`
  2. `RoundRobinScheduler`
  3. `FIFOScheduler`
  4. `EarliestDeadlineFirstScheduler` (EDF)
  5. `RiskPriorityScheduler` (heuristic prioritizing $R_i \ge 0.5$)
  6. `DRL-Fog` (Latency-only RL)
  7. `EADRL` (Efficiency-Aware DRL)
* **Location in Project:** `src/schedulers/baselines/`.

### Decision 7: Experiment Tracking & Reproducibility
* **Selected Tooling:** Lightweight JSON/CSV metric logging with reproducible Matplotlib/Seaborn graph generation.
* **Why Selected:** Avoids third-party SaaS dependencies (like cloud WandB) that require login tokens, ensuring 100% offline reproducibility across 10 random seeds.
* **Location in Project:** `src/experiments/`, `results/logs/`, `results/graphs/`.

---

## 3. Architecture Reuse Map

```
Open-Source Repositories            CRAI-LCS Production Architecture
────────────────────────            ────────────────────────────────
PanosLevedogiannis/ECG-Class. ───►  src/ecg/preprocessing.py (Record-wise split)
pr6thv3/ecg-ai-platform       ───►  src/ecg/dataset.py (360 Hz Z-Score Normalization)
mrrohan4321/ECG-Project-DL    ───►  src/models/cnn_lstm_risk.py (1D-CNN + LSTM 0.74M)
PalashSharma19/RL-Job-Sched.  ───►  src/schedulers/baselines/ (Unified Baseline Interface)
PadalaSravan14/drl_edge_cloud ───►  src/schedulers/dqn_scheduler.py (DDQN & Buffer)
Dharmendra2022/DRL-EdgeCloud  ───►  src/simulator/edge_cloud_env.py (Gymnasium Engine)
```
