# CRAI-LCS: Open-Source Repository Reuse Audit

**Document Status:** VERIFIED  
**Audit Date:** 2026-11-21  
**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

---

## 1. Executive Summary & Audit Matrix

Every candidate repository requested in the project brief was programmatically inspected using the GitHub API and raw content inspection. None were assumed or fabricated.

| Repository | Accessible? | License | Relevant Files | Reusable Components | Compatibility | Recommendation / Decision |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **mrrohan4321/ECG-Project-Deeplearning** | Yes | None specified (All rights reserved / default GitHub) | `models/`, `dataset/`, `graphs/` | 1D-CNN & CNN-LSTM PyTorch layers for normal vs. abnormal ECG classification | Moderate | **REFERENCE ONLY.** Use architecture structure as a reference for 1D-CNN + LSTM layer configurations. Do not copy directly due to lack of open-source license and absence of patient-wise data partitioning. |
| **PanosLevedogiannis/ECG-Classification** | Yes | None specified | `compare_models.py`, `ecg_mitbih.py` | Patient-aware stratified cross-validation logic; class-imbalance mitigation (Focal Loss) | High | **ADAPT SPECIFIC LOGIC.** Adapt patient-aware cross-validation splitting logic to enforce record-wise splitting (records 100-124 train, 200-215 val, 217-234 test) to prevent patient data leakage. |
| **pr6thv3/ecg-ai-platform** | Yes | MIT License | `src/preprocessing/`, `src/inference/` | 360 Hz signal buffering, normalization, ONNX/PyTorch inference patterns | Moderate | **SELECTIVE UTILITIES.** Reference signal filtering and 360 Hz sample windowing utilities. Exclude fullstack UI, Docker, WebSocket, and FastAPI bloat. |
| **Dharmendra2022/DRL-EdgeCloud** | Yes | MIT License | `drl_edgecloud/envs/continuum_env.py`, `metrics.py`, `replay_buffer.py` | Gymnasium-compatible environment structure, edge-cloud continuum state modeling | High | **ADAPT SIMULATION PATTERNS.** Adapt Gymnasium environment wrapper structure and metrics calculation for edge-cloud resource modeling. |
| **PadalaSravan14/drl_edge_cloud_scheduler** | Yes | None specified | `drl_edge_cloud_scheduler/agents/dqn_agent.py`, `ddqn_agent.py`, `baselines/fifo.py`, `round_robin.py` | DQN/DDQN agent architecture, Replay Buffer, modular baseline scheduling classes | Very High | **ADAPT DQN & BASELINE STRUCTURE.** Adopt the clean modular baseline scheduler class pattern (`FIFOScheduler`, `RoundRobinScheduler`) and DDQN structure, customized to our 26-action space and risk-conditioned reward function. |
| **PalashSharma19/Reinforcement-Learning-Job-Scheduling** | Yes | MIT License | `baselines.py`, `environment.py`, `evaluate.py` | Classical scheduling baselines: Earliest Deadline First (EDF), Shortest Job First (SJF), Random; deadline miss rate tracking | Very High | **ADAPT BASELINES & METRIC TRACKING.** Direct adaptation of the clean `schedule(observation) -> node_idx` interface for EDF, Random, and Round-Robin, plus deadline-violation rate calculation logic. |

---

## 2. Detailed Candidate Analysis

### Candidate 1: `mrrohan4321/ECG-Project-Deeplearning`
* **Repository URL:** `https://github.com/mrrohan4321/ECG-Project-Deeplearning`
* **Description:** GPU-accelerated 1D-CNN and CNN-LSTM for ECG beat classification (normal vs abnormal) on the MIT-BIH Arrhythmia Database.
* **Inspected Structure:** 48 files including `demo/app.py`, training scripts, and accuracy/loss curves.
* **License:** No license file in repository root.
* **Technical Assessment:**
  - *Strengths:* Implements a hybrid 1D-CNN + LSTM PyTorch model that matches the morphological feature extraction requirements of CRAI-LCS.
  - *Weaknesses:* Beat-level segmentation is performed without patient-wise separation, creating severe risk of patient data leakage if reused verbatim.
  - *Decision:* **Reference Only**. We will hand-craft our own clean PyTorch model in `src/models/cnn_lstm_risk.py` following paper specifications (0.74M parameters, 14.8 ms target inference) using its layer chaining as an architectural reference.

### Candidate 2: `PanosLevedogiannis/ECG-Classification`
* **Repository URL:** `https://github.com/PanosLevedogiannis/ECG-Classification`
* **Description:** Deep learning pipeline for ECG arrhythmia classification featuring patient-aware stratified cross-validation and Focal Loss.
* **Inspected Structure:** 162 files including raw MIT-BIH sample records (`100.dat`, `100.hea`, etc.) and `compare_models.py`.
* **License:** None specified.
* **Technical Assessment:**
  - *Strengths:* Emphasizes patient-aware data splitting across records to ensure patient separation between train and test sets.
  - *Weaknesses:* Focuses on beat classification rather than 5-second windowed continuous workload generation.
  - *Decision:* **Adapt Logic**. Reuse the patient-record grouping logic in `src/ecg/preprocessing.py` to guarantee zero data leakage between training (70%), validation (15%), and testing (15%).

### Candidate 3: `pr6thv3/ecg-ai-platform`
* **Repository URL:** `https://github.com/pr6thv3/ecg-ai-platform`
* **Description:** Real-Time AI-Driven ECG Arrhythmia Analysis Platform with ONNX Runtime, Grad-CAM, FastAPI, and WebSocket streaming.
* **Inspected Structure:** 477 files including fullstack web dashboards and dockerfiles.
* **License:** MIT License (Permissive, commercial-friendly).
* **Technical Assessment:**
  - *Strengths:* Robust, production-grade 360 Hz sample processing and normalization.
  - *Weaknesses:* Extreme bloat for an academic research simulation (FastAPI, Next.js, Docker, WebSockets).
  - *Decision:* **Selective Utilities**. Do NOT adopt the platform architecture. Extract only the clean numerical normalization and signal windowing routines.

### Candidate 4: `Dharmendra2022/DRL-EdgeCloud`
* **Repository URL:** `https://github.com/Dharmendra2022/DRL-EdgeCloud`
* **Description:** Topology-Aware Hierarchical Deep Reinforcement Learning for Multi-Objective Scheduling in the Edge–Cloud Continuum.
* **Inspected Structure:** 26 files including `drl_edgecloud/envs/continuum_env.py` and `metrics.py`.
* **License:** MIT License.
* **Technical Assessment:**
  - *Strengths:* Implements a standard Gymnasium environment interface (`step()`, `reset()`, observation/action spaces) modeling edge-to-cloud offloading.
  - *Weaknesses:* Uses GAT (Graph Attention Networks) and PPO tailored to hierarchical networking rather than the discrete DQN required by CRAI-LCS.
  - *Decision:* **Adapt Simulation Patterns**. Reuse the Gymnasium environment state management patterns and multi-objective metrics tracking in `src/simulator/`.

### Candidate 5: `PadalaSravan14/drl_edge_cloud_scheduler`
* **Repository URL:** `https://github.com/PadalaSravan14/drl_edge_cloud_scheduler`
* **Description:** DRL-based edge-cloud task scheduler optimizing latency, energy, and SLA using adaptive DQN.
* **Inspected Structure:** 38 files including `dqn_agent.py`, `ddqn_agent.py`, and `baselines/fifo.py`.
* **License:** None specified.
* **Technical Assessment:**
  - *Strengths:* Direct architectural match with DQN edge-cloud scheduling, clean replay buffer implementation, and modular baseline schedulers.
  - *Weaknesses:* Missing the clinical-risk state dimension ($R_i$) and deadline-violation prediction ($P_{miss,i}$) in the state space.
  - *Decision:* **Adapt Structure**. Adapt the replay buffer and modular baseline interface (`select_action(state)`). Extend the state space and reward logic to match CRAI-LCS Equation 29.

### Candidate 6: `PalashSharma19/Reinforcement-Learning-Job-Scheduling`
* **Repository URL:** `https://github.com/PalashSharma19/Reinforcement-Learning-Job-Scheduling`
* **Description:** Gymnasium environment for heterogeneous compute job scheduling with PPO and classical baselines.
* **Inspected Structure:** 17 files including `environment.py`, `baselines.py`, `evaluate.py`.
* **License:** MIT License.
* **Technical Assessment:**
  - *Strengths:* Exceptionally clean, battle-tested implementations of Earliest Deadline First (EDF), Random, and Round-Robin schedulers with a unified interface: `schedule(observation) -> int`.
  - *Weaknesses:* Simple synthetic tasks rather than ECG-grounded physiological workloads.
  - *Decision:* **Directly Adapt Baselines**. Adapt `baselines.py` for our Phase 2 baseline benchmark suite (`Random`, `RoundRobin`, `FIFO`, `EDF`, `RiskPriority`).

---

## 3. Official External Libraries to Integrate

Rather than writing infrastructure from scratch:
1. **`gymnasium` (Farama Foundation):** Standard RL environment API (`step`, `reset`, `spaces.Box`, `spaces.Discrete`).
2. **`wfdb` (PhysioNet):** Official WaveForm DataBase package for downloading and parsing MIT-BIH recordings (`.dat`, `.hea`, `.atr`).
3. **`torch` (PyTorch):** Deep learning tensor engine for CNN-LSTM, MLP Predictor, and DQN.
4. **`scipy` & `scikit-learn`:** For ROC-AUC, classification metrics, and Wilcoxon signed-rank testing.
5. **`matplotlib` & `seaborn`:** For generation of publication-standard comparison curves.
