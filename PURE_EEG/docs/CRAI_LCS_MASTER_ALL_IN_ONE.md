# CRAI-LCS: Consolidated Master Engineering Compendium

**Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling**  
*Comprehensive All-in-One Engineering Blueprint, Specifications, Implementation Roadmap, and AI Delegation Guide*  
*Provenance: Nature Portfolio Scientific Reports 2026 (DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9))*

---

## Master Table of Contents
1. [Chapter 1: Executive Overview & Repository Index](#chapter-1-executive-overview--repository-index)
2. [Chapter 2: Candidate Repositories Reuse Guide (Konsa Repo Kahan Use Karna Hai)](#chapter-2-candidate-repositories-reuse-guide-konsa-repo-kahan-use-karna-hai)
3. [Chapter 3: AI Automation Matrix: AI Se Kya Kya Kaam Kara Sakte Hain?](#chapter-3-ai-automation-matrix-ai-se-kya-kya-kaam-kara-sakte-hain)
4. [Chapter 4: Product Requirements Document (PRD) & Continuum Specs](#chapter-4-product-requirements-document-prd--continuum-specs)
5. [Chapter 5: Architecture Decision Records (ADRs)](#chapter-5-architecture-decision-records-adrs)
6. [Chapter 6: Scientific Reports Paper Compliance Matrix](#chapter-6-scientific-reports-paper-compliance-matrix)
7. [Chapter 7: Master 11-Phase Development Roadmap (Phases 0 to 10)](#chapter-7-master-11-phase-development-roadmap-phases-0-to-10)
8. [Chapter 8: Exhaustive Phase-by-Phase Practical Blueprint (Start-to-End Hinglish Guide)](#chapter-8-exhaustive-phase-by-phase-practical-blueprint-start-to-end-hinglish-guide)
9. [Chapter 9: Code Implementation & Snippets Guide](#chapter-9-code-implementation--snippets-guide)
10. [Chapter 10: Risk Register, Critical Gotchas & Architectural Mitigations](#chapter-10-risk-register-critical-gotchas--architectural-mitigations)
11. [Chapter 11: 7-Tier Testing Strategy & QA Hierarchy](#chapter-11-7-tier-testing-strategy--qa-hierarchy)
12. [Chapter 12: Empirical Benchmark Log & Baseline Comparisons](#chapter-12-empirical-benchmark-log--baseline-comparisons)
13. [Chapter 13: Task Tracking & AI/Human Ownership Board](#chapter-13-task-tracking--aihuman-ownership-board)
14. [Chapter 14: Turnkey Reproduction CLI Commands](#chapter-14-turnkey-reproduction-cli-commands)

---



---

## Chapter 1: Executive Overview & Repository Index

[![Paper: Nature Scientific Reports](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--68894--9-blue)](https://doi.org/10.1038/s41598-026-68894-9)
[![Zenodo Archive](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22036231-green)](https://doi.org/10.5281/zenodo.22036231)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28%2B-blueviolet)](https://gymnasium.farama.org/)

---

### 1. Overview

**CRAI-LCS** is an AI-enhanced scheduling framework for Healthcare Internet of Things (HIoT) environments that coordinates tasks across a 3-tier IoT-Edge-Cloud computing continuum. Unlike traditional system-centric schedulers that prioritize solely on queue length or generic latency, CRAI-LCS integrates:
1. **Authentic Physiological Workloads:** 5-second ECG windows from PhysioNet MIT-BIH ($T=1800$ samples @ 360 Hz).
2. **Clinical Risk Estimator:** Hybrid 1D-CNN + 2-layer LSTM ($0.74$M parameters) predicting continuous patient urgency score $R_i \in [0, 1]$.
3. **Deadline Violation Predictor:** MLP ($0.18$M parameters) forecasting deadline failure probability $P_{miss,i} \in [0, 1]$.
4. **Risk-Aware Double DQN Scheduler:** 26-action discrete RL agent ($0.32$M parameters) dynamically dispatching tasks across 25 edge gateways and 2 cloud servers using multi-objective reward shaping.

---

### 2. Project Engineering Documentation

This repository follows a strict, progressive research-and-development workflow:

* 📄 **[PRD.md](PRD.md):** Complete Product Requirements Document & System Specifications.
* 📋 **[PROJECT_PLAN.md](PROJECT_PLAN.md):** 11-Phase Implementation Roadmap with exit criteria.
* 📌 **[TASK_BOARD.md](TASK_BOARD.md):** Live task tracker with AI/Human ownership and statuses.
* 🔍 **[REUSE_AUDIT.md](REUSE_AUDIT.md):** Open-source audit of 6 candidate GitHub repositories and licenses.
* 🏛️ **[ARCHITECTURE_DECISION.md](ARCHITECTURE_DECISION.md):** Architecture Decision Records (ADR).
* 📐 **[PAPER_COMPLIANCE.md](PAPER_COMPLIANCE.md):** Equation-by-equation and table-by-table compliance matrix.
* 🧠 **[DECISIONS.md](DECISIONS.md):** Technical and architectural decision logs.
* ⚠️ **[RISK_REGISTER.md](RISK_REGISTER.md):** Failure modes, root causes, and prevention protocols.
* 🧪 **[TESTING_STRATEGY.md](TESTING_STRATEGY.md):** 7-level layered testing hierarchy and 12-step debugging protocol.
* 🐞 **[BUG_LOG.md](BUG_LOG.md):** Bug reporting and post-mortem resolution ledger.
* 📊 **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md):** Benchmark run records and published baseline comparisons.
* 🤝 **[AI_HANDOFF.md](AI_HANDOFF.md):** Project continuation state and next immediate action.

---

### 3. Directory Layout

```
CRAI_LCS/
├── data/
│   ├── raw/mitbih/               # PhysioNet MIT-BIH recordings (.dat, .hea, .atr)
│   └── processed/                # 5s windowed segments & patient-wise splits
├── src/
│   ├── ecg/                      # Preprocessing, normalization, task generation
│   ├── models/                   # CNN-LSTM risk model, MLP deadline predictor
│   ├── simulator/                # 25 Edge nodes + 2 Cloud discrete-event environment
│   ├── schedulers/               # DQN agent, Replay buffer, and 5 baseline schedulers
│   ├── training/                 # Model training loops and reward shaping
│   └── experiments/              # Multi-load sweeps, Wilcoxon testing, plot scripts
├── configs/                      # YAML simulation & model configurations
├── checkpoints/                  # Saved PyTorch model weights (.pt)
├── results/                      # Experiment logs, CSV metrics, and generated graphs
├── tests/                        # 7-level progressive test suite
├── main.py                       # Unified CLI entrypoint
├── requirements.txt              # Project dependencies
└── README.md                     # Master project guide
```

---

### 4. Quick Start & Setup

#### Install Dependencies:
```bash
pip install -r requirements.txt
```

#### Run Tests:
```bash
pytest tests/
```

#### Execute Demonstration:
```bash
python3 main.py --mode demo
```


---

## Chapter 2: Candidate Repositories Reuse Guide (Konsa Repo Kahan Use Karna Hai)

**Document Status:** VERIFIED  
**Audit Date:** 2026-11-21  
**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

---

### 1. Executive Summary & Audit Matrix

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

### 2. Detailed Candidate Analysis

#### Candidate 1: `mrrohan4321/ECG-Project-Deeplearning`
* **Repository URL:** `https://github.com/mrrohan4321/ECG-Project-Deeplearning`
* **Description:** GPU-accelerated 1D-CNN and CNN-LSTM for ECG beat classification (normal vs abnormal) on the MIT-BIH Arrhythmia Database.
* **Inspected Structure:** 48 files including `demo/app.py`, training scripts, and accuracy/loss curves.
* **License:** No license file in repository root.
* **Technical Assessment:**
  - *Strengths:* Implements a hybrid 1D-CNN + LSTM PyTorch model that matches the morphological feature extraction requirements of CRAI-LCS.
  - *Weaknesses:* Beat-level segmentation is performed without patient-wise separation, creating severe risk of patient data leakage if reused verbatim.
  - *Decision:* **Reference Only**. We will hand-craft our own clean PyTorch model in `src/models/cnn_lstm_risk.py` following paper specifications (0.74M parameters, 14.8 ms target inference) using its layer chaining as an architectural reference.

#### Candidate 2: `PanosLevedogiannis/ECG-Classification`
* **Repository URL:** `https://github.com/PanosLevedogiannis/ECG-Classification`
* **Description:** Deep learning pipeline for ECG arrhythmia classification featuring patient-aware stratified cross-validation and Focal Loss.
* **Inspected Structure:** 162 files including raw MIT-BIH sample records (`100.dat`, `100.hea`, etc.) and `compare_models.py`.
* **License:** None specified.
* **Technical Assessment:**
  - *Strengths:* Emphasizes patient-aware data splitting across records to ensure patient separation between train and test sets.
  - *Weaknesses:* Focuses on beat classification rather than 5-second windowed continuous workload generation.
  - *Decision:* **Adapt Logic**. Reuse the patient-record grouping logic in `src/ecg/preprocessing.py` to guarantee zero data leakage between training (70%), validation (15%), and testing (15%).

#### Candidate 3: `pr6thv3/ecg-ai-platform`
* **Repository URL:** `https://github.com/pr6thv3/ecg-ai-platform`
* **Description:** Real-Time AI-Driven ECG Arrhythmia Analysis Platform with ONNX Runtime, Grad-CAM, FastAPI, and WebSocket streaming.
* **Inspected Structure:** 477 files including fullstack web dashboards and dockerfiles.
* **License:** MIT License (Permissive, commercial-friendly).
* **Technical Assessment:**
  - *Strengths:* Robust, production-grade 360 Hz sample processing and normalization.
  - *Weaknesses:* Extreme bloat for an academic research simulation (FastAPI, Next.js, Docker, WebSockets).
  - *Decision:* **Selective Utilities**. Do NOT adopt the platform architecture. Extract only the clean numerical normalization and signal windowing routines.

#### Candidate 4: `Dharmendra2022/DRL-EdgeCloud`
* **Repository URL:** `https://github.com/Dharmendra2022/DRL-EdgeCloud`
* **Description:** Topology-Aware Hierarchical Deep Reinforcement Learning for Multi-Objective Scheduling in the Edge–Cloud Continuum.
* **Inspected Structure:** 26 files including `drl_edgecloud/envs/continuum_env.py` and `metrics.py`.
* **License:** MIT License.
* **Technical Assessment:**
  - *Strengths:* Implements a standard Gymnasium environment interface (`step()`, `reset()`, observation/action spaces) modeling edge-to-cloud offloading.
  - *Weaknesses:* Uses GAT (Graph Attention Networks) and PPO tailored to hierarchical networking rather than the discrete DQN required by CRAI-LCS.
  - *Decision:* **Adapt Simulation Patterns**. Reuse the Gymnasium environment state management patterns and multi-objective metrics tracking in `src/simulator/`.

#### Candidate 5: `PadalaSravan14/drl_edge_cloud_scheduler`
* **Repository URL:** `https://github.com/PadalaSravan14/drl_edge_cloud_scheduler`
* **Description:** DRL-based edge-cloud task scheduler optimizing latency, energy, and SLA using adaptive DQN.
* **Inspected Structure:** 38 files including `dqn_agent.py`, `ddqn_agent.py`, and `baselines/fifo.py`.
* **License:** None specified.
* **Technical Assessment:**
  - *Strengths:* Direct architectural match with DQN edge-cloud scheduling, clean replay buffer implementation, and modular baseline schedulers.
  - *Weaknesses:* Missing the clinical-risk state dimension ($R_i$) and deadline-violation prediction ($P_{miss,i}$) in the state space.
  - *Decision:* **Adapt Structure**. Adapt the replay buffer and modular baseline interface (`select_action(state)`). Extend the state space and reward logic to match CRAI-LCS Equation 29.

#### Candidate 6: `PalashSharma19/Reinforcement-Learning-Job-Scheduling`
* **Repository URL:** `https://github.com/PalashSharma19/Reinforcement-Learning-Job-Scheduling`
* **Description:** Gymnasium environment for heterogeneous compute job scheduling with PPO and classical baselines.
* **Inspected Structure:** 17 files including `environment.py`, `baselines.py`, `evaluate.py`.
* **License:** MIT License.
* **Technical Assessment:**
  - *Strengths:* Exceptionally clean, battle-tested implementations of Earliest Deadline First (EDF), Random, and Round-Robin schedulers with a unified interface: `schedule(observation) -> int`.
  - *Weaknesses:* Simple synthetic tasks rather than ECG-grounded physiological workloads.
  - *Decision:* **Directly Adapt Baselines**. Adapt `baselines.py` for our Phase 2 baseline benchmark suite (`Random`, `RoundRobin`, `FIFO`, `EDF`, `RiskPriority`).

---

### 3. Official External Libraries to Integrate

Rather than writing infrastructure from scratch:
1. **`gymnasium` (Farama Foundation):** Standard RL environment API (`step`, `reset`, `spaces.Box`, `spaces.Discrete`).
2. **`wfdb` (PhysioNet):** Official WaveForm DataBase package for downloading and parsing MIT-BIH recordings (`.dat`, `.hea`, `.atr`).
3. **`torch` (PyTorch):** Deep learning tensor engine for CNN-LSTM, MLP Predictor, and DQN.
4. **`scipy` & `scikit-learn`:** For ROC-AUC, classification metrics, and Wilcoxon signed-rank testing.
5. **`matplotlib` & `seaborn`:** For generation of publication-standard comparison curves.


---

## Chapter 3: AI Automation Matrix: AI Se Kya Kya Kaam Kara Sakte Hain?

Is project mein AI (Autonomous Coding Agents, LLMs, PyTorch Scripting Assistants) ko kis-kis step par deploy kiya ja sakta hai aur human lead ko kya oversee karna hai:

| Domain / Pipeline Area | AI Kya Kaam Kar Sakta Hai? (Autonomous Capabilities) | Human Developer / Lead Ka Role (Validation & Oversight) |
| :--- | :--- | :--- |
| **1. Data Engineering & Preprocessing** | • PhysioNet WFDB download scripts (`download_mitbih.py`) likhna.<br>• Z-score voltage normalization ($\tilde{x}_i = (x_i - \mu)/\sigma$) implement karna.<br>• 5s sliding windowing ($1800$ samples @ 360 Hz) automate karna.<br>• Zero-leakage patient-record split logic likhna aur verify karna. | • Verify karna ki raw recordings complete hain.<br>• Clinical label mapping (AAMI EC57) ko confirm karna. |
| **2. Deep Learning Neural Models** | • 1D-CNN-LSTM architecture (`src/models/cnn_lstm_risk.py`) construct karna.<br>• Class-imbalance weighted BCE loss ($w_{pos}=3.42$) aur Adam optimizer configure karna.<br>• 3-layer MLP Deadline Violation Predictor (`64 -> 32 -> 16 -> 1`) author karna.<br>• Model parameters ($0.74$M aur $0.18$M) aur latency budget ($\le 25$ ms) assert karna. | • Train/Val/Test loss curves inspect karna.<br>• Check karna ki test accuracy $\ge 94.2\%$ aur AUC $\ge 0.962$ cross-validated hai. |
| **3. Simulator & RL Scheduling** | • 200 monitors, 25 edge nodes, aur 2 cloud instances ka discrete-event simulation code karna.<br>• Farama `gymnasium.Env` wrapper (`step()`, `reset()`) likhna.<br>• Double DQN agent (`src/models/dqn_scheduler.py`) aur Experience Replay Buffer (50,000 capacity) likhna.<br>• Multi-objective normalized reward function (Eq. 29 with $L_i/D_i$) implement karna. | • Simulation assumptions (MIPS, bandwidth, queueing models) ko paper se cross-verify karna. |
| **4. Baseline Schedulers** | • Sabhi 5 baseline schedulers (`Random`, `RoundRobin`, `FIFO`, `EDF`, `RiskPriority`) ka unified `schedule(obs) -> int` code likhna. | • Sanity check karna ki baselines fair aur non-crippled hain. |
| **5. Benchmarking & Statistical Rigor** | • Poisson task arrival streams ($\lambda=5$ to $25$ tasks/sec) simulate karna.<br>• Stress tests run karke Latency, Violation Rate, Risk Coverage, aur Energy log karna.<br>• Wilcoxon Signed-Rank Test automatically run karke $p$-value calculate karna ($p = 0.00195 \ll 0.05$).<br>• 4 ablation variants aur PTB-XL cross-testing scripts likhna. | • Review karna ki benchmarks paper ke Table 10, 11, aur 13 se bitwise align ho rahe hain. |
| **6. Testing, CI/CD & QA Pyramid** | • 7-level testing suite author karna (`pytest tests/`).<br>• Flake8 linting aur Mypy static typing pipeline automate karna.<br>• Edge node queue overflow, tensor shape boundaries, aur state space bounds ke assertions lagana. | • Final test suite pass report approve karna. |
| **7. Scientific Documentation & Typesetting** | • LaTeX book monographs, technical slide presentations, aur PRD/ADR documents autonomously author aur compile karna.<br>• Architectural TikZ flowcharts aur publication tables typeset karna. | • Final PDF quality aur visual formatting review karna. |


---

## Chapter 4: Product Requirements Document (PRD) & Continuum Specs

### Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

**Document Version:** 1.0.0  
**Status:** APPROVED & BASELINED  
**Date:** 2026-11-21  
**Target Publication Reference:** *Scientific Reports (Nature Portfolio)*, 2026. DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9)

---

### 1. Project Overview & Context

In the modern Healthcare Internet of Things (HIoT), wearable devices and bedside monitors continuously generate vital physiological signals, such as electrocardiograms (ECG). These time-critical signals are vital for the early detection of life-threatening events, including acute arrhythmias and hypoxia.

Conventional cloud-centric architectures introduce substantial round-trip communication delays ($45 - 120$ ms backhaul), which can compromise clinical outcomes during cardiac emergencies. Mobile Edge Computing (MEC) mitigates this latency by deploying distributed compute gateways closer to patients. However, existing edge schedulers remain almost exclusively **system-centric**—optimizing infrastructure metrics like CPU utilization or generic queue length—while treating all computational tasks uniformly. 

**CRAI-LCS** resolves this fundamental flaw by introducing a **closed-loop clinical-to-computational decision paradigm**. It embeds patient-specific physiological risk ($R_i$) directly into the Markov Decision Process (MDP) of a Deep Reinforcement Learning (DRL) scheduler, coordinating with a pre-execution deadline violation predictor ($P_{miss,i}$) across a 3-tier IoT-Edge-Cloud hierarchy.

---

### 2. Problem Statement

1. **Failure to Distinguish Criticality:** In conventional fog/edge schedulers, a routine monitoring packet and a lethal ventricular fibrillation packet are queued identically if their file sizes and cycle counts match.
2. **Stringent Healthcare Latencies:** High-risk cardiac events require immediate intervention (deadlines $100 - 150$ ms), whereas transmission to distant cloud datacenters routinely breaches these constraints.
3. **Edge Resource Scarcity:** Edge gateways possess constrained computational and memory capacity ($800 - 4500$ MIPS). Uncontrolled local queuing triggers buffer overflows, while excessive cloud offloading increases backhaul latency and energy waste.
4. **Stochastic Clinical Surges:** Real-world patient physiological emergencies occur unpredictably, requiring dynamic, real-time task re-prioritization rather than rigid static heuristics.

---

### 3. Measurable Objectives

The completed CRAI-LCS implementation must satisfy the following verifiable targets:
* **OBJ-1:** Construct a high-fidelity discrete-event Edge-Cloud simulation modeling 200 IoT sensors, 25 heterogeneous edge gateways, and 2 cloud instances.
* **OBJ-2:** Implement the ECG data pipeline extracting 5-second windows ($T=1800$ samples @ 360 Hz) from the PhysioNet MIT-BIH dataset with strict patient-wise separation (zero leakage).
* **OBJ-3:** Train a CNN-LSTM clinical risk estimator achieving test accuracy $\ge 93.4\%$, F1-score $\ge 92.5\%$, and AUC $\ge 0.95$ within a $0.74$M parameter budget.
* **OBJ-4:** Implement an MLP deadline violation predictor ($0.18$M parameters) achieving AUC $\ge 0.90$.
* **OBJ-5:** Train a 26-action Double DQN scheduler ($0.32$M parameters) with multi-objective reward shaping.
* **OBJ-6:** Achieve superior performance over 5 baseline schedulers under peak load ($\lambda = 20$ tasks/sec):
  - Average latency $\le 150$ ms (vs. $183.1$ ms for best baseline).
  - Deadline violation rate $\le 3.0\%$ (vs. $6.08\%$ for best baseline).
  - Clinical risk coverage $\ge 95.8\%$ (vs. $84.6\%$ for best baseline).
  - Energy consumption $\le 290$ J / 1000 tasks.
* **OBJ-7:** Complete statistical significance validation across 10 independent random seeds confirming $p < 0.05$ (Wilcoxon signed-rank test).

---

### 4. Scope

#### In Scope
- Full data ingestion, segmentation, and patient-wise partitioning of PhysioNet MIT-BIH and PTB-XL datasets.
- PyTorch implementation and training of the CNN-LSTM risk model and MLP deadline predictor.
- Gymnasium-compatible discrete-event simulation of the 25-node edge cluster and 2-instance cloud infrastructure.
- Double DQN scheduler training with experience replay and multi-objective reward shaping.
- Implementation of 5 baseline schedulers: Random, Round-Robin, FIFO, Earliest Deadline First (EDF), and Risk Priority.
- Ablation experiments, reward sensitivity analysis, and statistical tests.
- Comprehensive technical documentation, architecture diagrams, and reproduction CLI.

#### Out of Scope
- Direct deployment onto physical hospital clinical networks.
- Patient diagnosis or clinical decision support intended for real-time human treatment without physician oversight.
- Production FDA/CE medical-device regulatory submissions.

---

### 5. Stakeholder & Responsibility Matrix

| Entity | Role & Ownership | Key Responsibilities |
| :--- | :--- | :--- |
| **Human Developer / Researcher** | Project Lead & Reviewer | Approves architecture decisions, reviews research assumptions, validates final scientific metrics, manages external credentials/downloads. |
| **AI Agent (Antigravity)** | Lead Research & Implementation Engineer | Autonomously authors modular Python code, configures simulation, trains models, runs tests, creates documentation, profiles performance. |
| **PhysioNet / MIT-BIH** | Data Provider | Provides raw physiological cardiac recordings under open data use policies. |
| **CRAI-LCS Framework** | Software System | Ingests ECG, estimates risk, predicts violations, schedules tasks, and logs performance. |

---

### 6. Functional Requirements (FR)

| Req ID | Description | Priority | Implemented By | Dependencies | Acceptance Criteria | Verification Method |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **FR-001** | Download MIT-BIH Arrhythmia dataset from PhysioNet | High | AI + Human | Network | All 48 records downloaded into `data/raw/mitbih/` with `.dat`, `.hea`, `.atr` files. | File integrity check & `wfdb.rdsamp` test. |
| **FR-002** | Z-score normalization of raw ECG signals | High | AI | FR-001 | Signal zero-centered with unit variance per segment: $\tilde{x}_i = (x_i - \mu)/\sigma$. | Unit test checking mean $\approx 0$ and std $\approx 1$. |
| **FR-003** | 5-second window task segmentation ($T=1800$ @ 360 Hz) | High | AI | FR-002 | Tasks contain exactly 1800 samples, size $\approx 3.6$ KB. | Shape assertion `(1800,)` on output tensors. |
| **FR-004** | Record-wise patient split (Train 70%, Val 15%, Test 15%) | Critical | AI | FR-003 | Zero overlap of patient records between train, val, and test subsets. | Set intersection test on patient IDs == $\emptyset$. |
| **FR-005** | Predict clinical risk score $R_i \in [0, 1]$ | High | AI | FR-004 | CNN-LSTM forward pass outputs risk score in range $[0.0, 1.0]$. | Test on synthetic tensor; range check $[0, 1]$. |
| **FR-006** | Generate task computation demand $\psi_i$ and deadlines $D_i$ | High | AI | FR-005 | $\psi_i \in [5\times 10^6, 2\times 10^7]$; $D_i \in [100, 150]$ ms for high risk, $[400, 800]$ ms normal. | Boundary assertion tests on generated tasks. |
| **FR-007** | Simulate 25 heterogeneous Edge Gateways | High | AI | None | Node MIPS $\in [800, 4500]$, RAM $\in [16, 64]$ GB, Base power $\in [40, 80]$ W. | Unit test verifying node attributes. |
| **FR-008** | Simulate 2 Cloud Datacenter instances | High | AI | None | Node MIPS $\in [12k, 35k]$, RAM $128$ GB, Base power $\in [180, 350]$ W. | Unit test verifying cloud specs. |
| **FR-009** | Compute transmission and execution latency | High | AI | FR-007, 008 | Implements Equations (8)-(10); $T_{tr} \in [3, 12]$ ms, $T_{e\to c} \in [45, 120]$ ms. | Hand-calculated mathematical assertion tests. |
| **FR-010** | Predict deadline violation probability $P_{miss,i}$ | High | AI | FR-006, 009 | MLP outputs violation probability given queue, load, cycles, and risk. | Test output shape `(Batch, 1)` and range $[0, 1]$. |
| **FR-011** | Discrete-event Poisson workload generator | High | AI | None | Generates task streams with $\lambda \in [8, 25]$ tasks/sec. | Empirical inter-arrival time distribution test. |
| **FR-012** | Implement 5 Baseline Schedulers | High | AI | FR-009 | Random, Round-Robin, FIFO, EDF, and Risk-Priority expose `schedule(obs) -> int`. | Determinism and order correctness unit tests. |
| **FR-013** | Implement 26-Action Double DQN Agent | High | AI | FR-009, 010 | Action space size 26; Experience replay (50k); Target network updates every 200 steps. | Forward pass and backward gradient flow tests. |
| **FR-014** | Multi-objective reward shaping calculation | High | AI | FR-013 | $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ with normalized latency. | Numerical sanity checks on positive/negative bounds. |
| **FR-015** | Peak-load comparative benchmark ($\lambda=20$) | High | AI | FR-012, 013 | Benchmarks all 6 algorithms under identical task sequence. | Script outputs comparative results table. |
| **FR-016** | 4-Model Ablation Study implementation | High | AI | FR-013 | Evaluates w/o risk, w/o predictor, w/o reward shaping, and static baseline. | Script runs and generates Table 8 reproduction. |
| **FR-017** | 10-Seed Statistical Significance Evaluation | High | AI | FR-015 | Runs 10 seeds, computes Mean $\pm$ Std and Wilcoxon signed-rank test. | Test confirms $p < 0.05$ across main metrics. |
| **FR-018** | Publication graph generation | Medium | AI | FR-015, 016 | Generates Figures 3 to 8 using Matplotlib/Seaborn. | Image integrity & axis value validation. |

---

### 7. Non-Functional Requirements (NFR)

* **NFR-1 (Modularity):** All modules must be decoupled behind abstract interfaces (`src/ecg/`, `src/models/`, `src/simulator/`, `src/schedulers/`).
* **NFR-2 (Reproducibility):** Setting a global random seed (`--seed 42`) must guarantee identical data splits, task arrival sequences, and initial network weights.
* **NFR-3 (Execution Efficiency):** Real-time inference budget for the combined models must not exceed $25$ ms on standard CPU hardware (CNN-LSTM $\le 15$ ms, Predictor $\le 4$ ms, DQN $\le 6$ ms).
* **NFR-4 (Memory Footprint):** Peak system memory during simulation and training must not exceed $6$ GB RAM.
* **NFR-5 (Test Coverage):** Every functional module must have a corresponding test in `tests/` with strict pass/fail assertions.
* **NFR-6 (Code Quality):** Code must be strictly typed, well-commented, and comply with PEP 8 standards.

---

### 8. System Architecture & Detailed Data Flow

```
[PhysioNet MIT-BIH ECG (360 Hz)]
           │
           ▼
[Z-Score Normalization & 5s Windowing (T=1800)]
           │
           ▼
[CNN-LSTM Risk Estimator (0.74M Params)] ──► Output: Clinical Risk R_i in [0, 1]
           │
           ▼
[Healthcare Task Generator] ──► Task T_i = {delta_i, psi_i, D_i, R_i}
           │
           ▼
[Edge Cluster & Cloud Simulator (25 Edge + 2 Cloud)]
     │                                │
     ├─► System State [Q_t, U_t]      │
     │                                │
     ▼                                ▼
[Deadline Violation Predictor]   [DQN Scheduler (0.32M Params)]
Output: P_miss,i in [0, 1]              │ (State: Q_t, U_t, R_i, P_miss, D_i)
     │                                │
     └────────────────────────────────┘
                      │
                      ▼
        [Action Dispatch: Node j in {0..25}]
                      │
                      ▼
         [Simulated Task Execution]
                      │
                      ▼
     [Latency L_i, Energy E_i, Violations]
                      │
                      ▼
      [Reward Feedback & Replay Buffer]
```

---

### 9. Data Requirements & Clinical Risk Mapping

* **Dataset:** PhysioNet MIT-BIH Arrhythmia Database.
* **Sampling Rate:** 360 samples/sec.
* **Segment Duration:** 5 seconds ($1800$ continuous scalar float values).
* **Clinical Risk Labeling ($y_i \in \{0, 1\}$):**
  - High Risk ($y_i = 1$): Segments containing Ventricular Premature Contractions (V), Ventricular Tachycardia (VT), Atrial Fibrillation (AFIB), Ventricular Flutter (VFL), or ventricular escape beats.
  - Normal ($y_i = 0$): Segments containing exclusively Normal Sinus Rhythm (N), Left/Right Bundle Branch Blocks without ectopy, or pacing.
* **Patient Leakage Prevention:** Non-overlapping record partition:
  - Training records (70%): `100, 101, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215`
  - Validation records (15%): `219, 220, 221, 222, 223, 228`
  - Testing records (15%): `230, 231, 232, 233, 234`


---

## Chapter 5: Architecture Decision Records (ADRs)

**Status:** APPROVED  
**Date:** 2026-11-21  
**Author:** AI Research & Engineering Team (Antigravity)  
**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

---

### 1. Context & Motivation

The CRAI-LCS paper introduces a closed-loop healthcare IoT scheduling paradigm that unites:
1. Real ECG physiological signals from PhysioNet MIT-BIH.
2. A CNN-LSTM model estimating clinical risk $R_i \in [0, 1]$.
3. A deadline violation predictor calculating $P_{miss,i} \in [0, 1]$.
4. A discrete-event edge-cloud simulator (25 edge nodes, 2 cloud instances, 200 IoT sensors).
5. A risk-conditioned Deep Q-Network (DQN) scheduler with multi-objective reward shaping.
6. A rigorous benchmarking suite against 5 baselines and ablation models across 10 random seeds.

To balance scientific fidelity, engineering maintainability, and implementation velocity, we audited 6 open-source GitHub repositories alongside standard scientific libraries (`wfdb`, `gymnasium`, `torch`). This ADR establishes our exact component selections and structural rationale.

---

### 2. Component Selection Decisions

#### Decision 1: ECG Preprocessing & Task Workload Pipeline
* **Selected Foundation:** Hybrid approach combining `PanosLevedogiannis/ECG-Classification` (patient-aware record grouping) and `pr6thv3/ecg-ai-platform` (signal normalization).
* **Why Selected:**
  - `PanosLevedogiannis` prevents catastrophic patient data leakage by segregating patients by record ID.
  - `pr6thv3` provides clean Z-score normalization and high-throughput NumPy vectorization.
* **Paper-Specific Customization:** We explicitly enforce 5-second windowing at 360 Hz ($T = 1800$ samples, $\delta_i \approx 3.6$ KB) and map arrhythmia annotations to binary clinical ground truth ($y_i \in \{0, 1\}$).
* **Location in Project:** `src/ecg/dataset.py`, `src/ecg/preprocessing.py`, `src/ecg/task_generator.py`.

#### Decision 2: CNN-LSTM Clinical Risk Estimator Architecture
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

#### Decision 3: Edge-Cloud Simulation Engine
* **Selected Foundation:** Gymnasium-compatible discrete-event queueing engine adapted from `Dharmendra2022/DRL-EdgeCloud` and `PalashSharma19/Reinforcement-Learning-Job-Scheduling`.
* **Why Selected:** Farama Foundation's `gymnasium` provides standard RL interfaces (`step()`, `reset()`, `spaces.Box`, `spaces.Discrete`). Adapting the simulation queue dynamics allows exact implementation of equations (8)–(11) and (16).
* **Specifications:**
  - 25 Heterogeneous Edge Nodes: $f_j \in [800, 4500]$ MIPS, RAM $16-64$ GB, Base Power $40-80$ W.
  - 2 High-Performance Cloud Instances: $f_c \in [12000, 35000]$ MIPS, RAM $128$ GB, Base Power $180-350$ W.
  - Latency: Device-to-Edge $T_{tr} \in [3, 12]$ ms; Edge-to-Cloud $T_{e \to c} \in [45, 120]$ ms.
  - Workload Arrivals: Stochastic Poisson process $\lambda \in [8, 25]$ tasks/sec.
* **Location in Project:** `src/simulator/edge_cloud_env.py`, `src/simulator/nodes.py`.

#### Decision 4: Deadline Violation Predictor Module
* **Selected Foundation:** Custom 3-layer MLP classifier ($0.18$M parameters, $<4$ ms inference).
* **Why Selected:** None of the open-source scheduling repositories had an explicit pre-execution deadline violation predictor. We author this module directly according to equation (12): $P_{miss,i} = f_\theta(S_i)$ where $S_i = \{U_t, Q_t, \psi_i, D_i, R_i\}$.
* **Location in Project:** `src/models/deadline_predictor.py`.

#### Decision 5: DQN Scheduling Agent & Reward Shaping
* **Selected Foundation:** Double DQN (DDQN) architecture adapted from `PadalaSravan14/drl_edge_cloud_scheduler`.
* **Why Selected:** `PadalaSravan14` provides a clean implementation of the Replay Buffer and target network update mechanics. We adapt it to support:
  - State space ($53$-dim or compact $8$-dim observation vector): $\{Q_t, U_t, R_i, P_{miss,i}, D_i\}$.
  - Action space ($26$ discrete actions): Assigning to $E_1 \dots E_{25}$ or Cloud $C$.
  - Exact reward function (equation 29):
    $$R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$$
    with normalized latency and default coefficients $\alpha=0.35, \beta=0.25, \gamma=0.30, \eta=0.10$.
* **Location in Project:** `src/schedulers/dqn_scheduler.py`, `src/training/replay_buffer.py`.

#### Decision 6: Baseline Scheduling Algorithms
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

#### Decision 7: Experiment Tracking & Reproducibility
* **Selected Tooling:** Lightweight JSON/CSV metric logging with reproducible Matplotlib/Seaborn graph generation.
* **Why Selected:** Avoids third-party SaaS dependencies (like cloud WandB) that require login tokens, ensuring 100% offline reproducibility across 10 random seeds.
* **Location in Project:** `src/experiments/`, `results/logs/`, `results/graphs/`.

---

### 3. Architecture Reuse Map

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


---

## Chapter 6: Scientific Reports Paper Compliance Matrix

**Paper Reference:** *Clinical risk-aware reinforcement learning for latency-constrained healthcare IoT scheduling*, Scientific Reports 2026. DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9)  
**Verification Date:** 2026-11-21  
**Status:** AUDITED & TRACKED

---

### 1. Compliance Classification Scheme
Every element in the project is audited and assigned one of three labels:
* **[VERIFIED]:** Explicitly specified in the published paper text, tables, or equations.
* **[PROPOSED / DECISION]:** Reasonable engineering implementation decision introduced by our team where the paper left minor low-level implementation details unspecified.
* **[ASSUMPTION / UNVERIFIED]:** Unspecified in the paper, requiring empirical validation or human review.

---

### 2. Equation-by-Equation Compliance Audit

| Item / Eq | Description in Paper | Paper Value / Form | Project Implementation | Compliance Status |
| :--- | :--- | :--- | :--- | :---: |
| **Eq. (1)-(2)** | Task Representation | $T_i = \{\delta_i, \psi_i, D_i, R_i\}$ | Implemented as `HealthcareTask` dataclass with these exact 4 attributes. | **[VERIFIED]** |
| **Sec. Task Modeling** | Window Length & Sampling | 5-second window at 360 Hz = 1800 samples; $\delta_i \approx 3.6$ KB | `window_seconds=5`, `fs=360`, `length=1800`, `input_size=3600` bytes. | **[VERIFIED]** |
| **Eq. (3)** | CPU Computational Demand | $\psi_i \in [5 \times 10^6, 2 \times 10^7]$ CPU cycles | Sampled uniformly from range $[5\times 10^6, 2\times 10^7]$ during task generation. | **[VERIFIED]** |
| **Sec. Deadlines** | Urgency-Aware Deadlines | High-risk: $100 - 150$ ms; Normal: up to $800$ ms | $D_i \in [100, 150]$ if $R_i \ge 0.5$; else $D_i \in [400, 800]$. | **[VERIFIED]** |
| **Eq. (4)-(7)** | Edge Topology & Capacity | $M=25$ nodes; $a_j = w_c \frac{\psi_j}{\psi_{max}} + w_m \frac{m_j}{m_{max}}$ | 25 heterogeneous `EdgeNode` objects with CPU/RAM weights $w_c=w_m=0.5$. | **[VERIFIED]** |
| **Eq. (8)-(10)** | Edge & Cloud Latency | $L_i^{edge} = T_{tr} + \frac{\delta_i}{B_j} + \frac{\psi_i}{f_j}$; $L_i^{cloud} = T_{tr} + T_{e\to c} + \frac{\delta_i}{B_c} + \frac{\psi_i}{f_c}$ | Modeled with $T_{tr} \in [3, 12]$ ms, $T_{e\to c} \in [45, 120]$ ms, $B_j \in [10, 50]$ Mbps, $B_c \in [100, 1000]$ Mbps. | **[VERIFIED]** |
| **Eq. (12)** | Deadline Violation Predictor | $P_{miss,i} = f_\theta(S_i)$ with $S_i = \{U_t, Q_t, \psi_i, D_i, R_i\}$ | 3-layer MLP predicting failure probability $P_{miss,i} \in [0, 1]$. | **[VERIFIED]** |
| **Eq. (13)-(16)** | Multi-Objective Optimization | $\min \sum (\alpha L_i + \beta P_{miss,i} - \gamma R_i) - \eta U_{edge}$ | Directly incorporated into DRL reward function and system objective. | **[VERIFIED]** |
| **Eq. (17)-(26)** | CNN-LSTM Clinical Risk Model | Z-score normalization, Conv1D, LSTM, Sigmoid, BCE loss, Adam lr=$10^{-3}$, batch=32 | Implemented in `src/models/cnn_lstm_risk.py` with 0.74M parameters. | **[VERIFIED]** |
| **Eq. (27)** | RL State Representation | $s_t = \{Q_t, U_t, R_i, P_{miss,i}, D_i\}$ | Vector concatenation of edge queues, utilizations, risk, violation probability, and deadline. | **[VERIFIED]** |
| **Eq. (28)** | RL Action Space | $a_t \in \{E_1, E_2, \dots, E_{25}, C\}$ | Discrete space of size 26 ($0-24$ for edge nodes, $25$ for cloud). | **[VERIFIED]** |
| **Eq. (29)** | Immediate Reward Function | $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ | Default weights: $\alpha=0.35, \beta=0.25, \gamma=0.30, \eta=0.10$. Normalized latency. | **[VERIFIED]** |
| **Eq. (30)-(32)** | DQN Action-Value Objective | $Q(s, a; \theta)$, target network $\theta^-$, TD loss | Double DQN with experience replay (capacity 50,000) and target network updates. | **[VERIFIED]** |
| **Constraints C1-C4** | Latency, Load, Energy, Critical Task Serving | $L_i \le D_i$; $Load_j \le Cap_j$; $\sum E_i \le E_{max}$; $R_i^{served} \ge \forall i \in T_{crit}$ | Enforced in environment step logic and penalty shaping. | **[VERIFIED]** |
| **Table 3** | System Environment Config | 200 devices, 25 edge nodes, 2 cloud, 8-25 tasks/sec | Embedded into `configs/simulation_config.yaml`. | **[VERIFIED]** |
| **Table 4** | Node Specifications | Edge: 800-4500 MIPS, 16-64 GB, 40-80W; Cloud: 12k-35k MIPS, 128 GB, 180-350W | Node attributes populated randomly within specified ranges with fixed seed. | **[VERIFIED]** |
| **Table 10-11** | Sensitivity Reward Weights | RW-1 (0.50, 0.20, 0.20, 0.10); RW-2 default; RW-3 (0.25, 0.20, 0.45, 0.10); RW-4 (0.25, 0.20, 0.25, 0.30) | Implemented as configurable sensitivity experiment profiles. | **[VERIFIED]** |
| **Table 12** | Model Inference Budgets | CNN-LSTM: 0.74M, 14.8 ms; Predictor: 0.18M, 3.6 ms; DQN: 0.32M, 5.7 ms; Total: 1.24M, 24.1 ms | Layer sizes and channel counts tuned to achieve exact parameter bounds. | **[VERIFIED]** |
| **Table 9** | Statistical Evaluation | 10 independent runs, Wilcoxon signed-rank test ($\alpha=0.05, p<0.05$) | `src/experiments/statistical_tests.py` using `scipy.stats.wilcoxon`. | **[VERIFIED]** |

---

### 3. Implementation Decisions & Assumptions Requiring Tracking

| Item | Paper Description | Our Implementation Decision | Rationale |
| :--- | :--- | :--- | :--- |
| **ECG Record List** | Mentions PhysioNet MIT-BIH Arrhythmia without full explicit record enumeration | Selected standard 44 benchmark records (e.g., 100, 101, ..., 234) partitioned by patient ID | Prevents patient data leakage while ensuring broad arrhythmia representation. |
| **Task Queue Policy within Nodes** | Mentions queue condition $Q_t$ without specifying internal node queue discipline | Implemented FIFO queue per edge node with preemptive urgency tag for $R_i \ge 0.8$ | Matches Figure 2 ("Immediate response to Life-threatening conditions"). |
| **Reward Normalization Scale** | Paper defines formula $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ without scaling $L_i$ | Normalize latency $L_i$ by deadline ($L_i / D_i$ or $L_i / 1000.0$) | Prevents millisecond latency values ($100-800$) from overpowering $[0, 1]$ risk terms. |
| **PTB-XL Cross-Dataset Windowing** | Evaluated on PTB-XL (Table 13) | Downsample PTB-XL records to 360 Hz to match MIT-BIH 1800-sample input size | Ensures CNN-LSTM input layer compatibility without retraining network weights. |


---

## Chapter 7: Master 11-Phase Development Roadmap (Phases 0 to 10)

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Date:** 2026-11-21  
**Status:** APPROVED FOR EXECUTION

---

### Complete Phase Roadmap Overview

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
(Audit)     (Simulator) (Baselines) (ECG Data)  (CNN-LSTM)  (Predictor)
                                                                 │
Phase 10 ◄── Phase 9 ◄── Phase 8 ◄── Phase 7 ◄── Phase 6 ◄───────┘
(Delivery)  (Ablation)  (Benchmark) (Integrate) (DQN Agent)
```

---

### Phase 0: Setup, Repository Audit & Research Verification

#### A. Purpose
Establish a clean development environment, verify original paper claims and Zenodo code, audit the 6 candidate GitHub repositories, document licenses and reuse decisions, and initialize project tracking files.

#### B. Prerequisites
Linux OS, Python 3.10+, Git, workspace access.

#### C. Tasks
* **T0-01 (Audit & Licensing):** Inspect 6 candidate GitHub repositories via GitHub API; document licenses, relevant files, and reuse suitability.
  - *Assigned to:* AI Autonomous.
  - *Files:* `REUSE_AUDIT.md`.
  - *Effort:* 15 mins.
  - *Completion Criteria:* All 6 repositories verified with zero hallucinations.
* **T0-02 (Architecture & PRD):** Author master PRD, ADR, Paper Compliance matrix, Risk Register, and Testing Strategy.
  - *Assigned to:* AI + Human Review.
  - *Files:* `PRD.md`, `ARCHITECTURE_DECISION.md`, `PAPER_COMPLIANCE.md`, `RISK_REGISTER.md`, `TESTING_STRATEGY.md`.
  - *Effort:* 30 mins.
  - *Completion Criteria:* Complete specifications matching Nature Scientific Reports 2026 paper.
* **T0-03 (Environment Initialization):** Create modular directory structure, install `wfdb` and `gymnasium`, and verify PyTorch environment.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/`, `configs/`, `tests/`, `requirements.txt`.
  - *Effort:* 10 mins.
  - *Completion Criteria:* All packages imported successfully without syntax or version errors.

#### D. Expected Output
Clean workspace skeleton, complete project management documentation, dependency installation.

#### E. Problems We May Encounter
Package dependency conflicts, Python 3.14 deprecation warnings.

#### F. Problem Resolution Plan
Pin exact versions in `requirements.txt`; verify imports inside a test script.

#### G. Testing Plan
Run `python3 -c "import torch, wfdb, gymnasium, scipy, sklearn; print('OK')"` (Level 1).

#### H. Exit Criteria
All documentation created, libraries verified, Git initialized.

#### I. Human Review
Human reviews `PRD.md` and `REUSE_AUDIT.md` to confirm architecture direction.

#### J. Checkpoint
`git tag phase0-audit-complete`.

---

### Phase 1: Edge-Cloud Simulation Engine

#### A. Purpose
Build the core discrete-event simulation modeling 25 heterogeneous edge gateways and 2 cloud instances, implementing exact equations for latency, queueing, and energy.

#### B. Prerequisites
Phase 0 complete.

#### C. Tasks
* **T1-01 (Node & Task Representation):** Implement `HealthcareTask` dataclass and `EdgeNode` / `CloudInstance` classes.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/simulator/task.py`, `src/simulator/nodes.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Attributes match Table 4 (800-4500 MIPS edge, 12k-35k MIPS cloud).
* **T1-02 (Latency & Energy Engine):** Implement mathematical formulas for transmission delay, execution delay, and base/active energy consumption (Equations 8-10).
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/simulator/latency_energy.py`.
  - *Effort:* 25 mins.
  - *Criteria:* Hand-calculated unit tests pass with zero discrepancy.
* **T1-03 (Gymnasium Simulation Environment):** Implement `HealthcareEdgeEnv` with `step()` and `reset()` conforming to Gymnasium API.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/simulator/edge_cloud_env.py`.
  - *Effort:* 30 mins.
  - *Criteria:* Synthetic tasks schedule across 26 discrete actions without deadlock.

#### D. Expected Output
Fully functional discrete-event simulator executable with synthetic workloads.

#### E. Problems We May Encounter
Queue overflow, millisecond-to-second unit confusion, infinite execution loops.

#### F. Problem Resolution Plan
Enforce strict unit tests with known deterministic arrival times and hand-calculated outputs.

#### G. Testing Plan
Run `tests/test_simulator.py` checking latency, capacity limits, and energy conservation.

#### H. Exit Criteria
100% of deterministic simulation tests pass; all 26 actions assign tasks correctly.

#### I. Human Review
Verify that simulated latencies ($3-12$ ms edge, $45-120$ ms cloud) match clinical intuition.

#### J. Checkpoint
`git tag phase1-simulator-verified`.

---

### Phase 2: Baseline Schedulers

#### A. Purpose
Implement 5 classical and heuristic scheduling baselines to establish a benchmark standard before developing RL.

#### B. Prerequisites
Phase 1 complete.

#### C. Tasks
* **T2-01 (Stateless Baselines):** Implement `RandomScheduler`, `RoundRobinScheduler`, and `FIFOScheduler`.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/schedulers/baselines/random_rr_fifo.py`.
  - *Effort:* 20 mins.
  - *Criteria:* All implement `schedule(observation) -> int`.
* **T2-02 (Urgency Baselines):** Implement `EarliestDeadlineFirstScheduler` (EDF) and `RiskPriorityScheduler`.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/schedulers/baselines/edf_risk.py`.
  - *Effort:* 25 mins.
  - *Criteria:* EDF prioritizes minimum $D_i$; RiskPriority prioritizes $R_i \ge 0.5$.
* **T2-03 (Baseline Harness):** Benchmark all 5 baselines on a synthetic 1,000-task workload stream.
  - *Assigned to:* AI Autonomous.
  - *Files:* `tests/test_baselines.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Relative performance shows EDF reducing violation rate over Random/FIFO.

#### D. Expected Output
5 working baseline schedulers tested in the Phase 1 environment.

#### E. Problems We May Encounter
Tie-breaking ambiguities when deadlines are identical.

#### F. Problem Resolution Plan
Apply secondary priority sorting based on arrival order.

#### G. Testing Plan
Execute `pytest tests/test_baselines.py`.

#### H. Exit Criteria
All 5 baselines execute without runtime exceptions and produce valid metric dictionaries.

#### I. Human Review
Inspect baseline rankings to confirm expected algorithmic behavior.

#### J. Checkpoint
`git tag phase2-baselines-complete`.

---

### Phase 3: ECG Dataset Pipeline

#### A. Purpose
Download, normalize, segment, and partition real cardiac signals from PhysioNet MIT-BIH without patient data leakage.

#### B. Prerequisites
Phase 0 complete.

#### C. Tasks
* **T3-01 (Dataset Downloader):** Download 44 MIT-BIH records using `wfdb`.
  - *Assigned to:* AI + Human (Human monitors download).
  - *Files:* `src/ecg/download.py`.
  - *Effort:* 15 mins.
  - *Criteria:* 44 `.dat` and `.atr` files stored in `data/raw/mitbih/`.
* **T3-02 (Signal Preprocessing & Windowing):** Extract 5-second segments ($T=1800$ samples @ 360 Hz) with Z-score normalization.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/ecg/preprocessing.py`.
  - *Effort:* 25 mins.
  - *Criteria:* Normalization produces zero mean and unit variance.
* **T3-03 (Patient-Wise Partitioning & Task Labeling):** Map arrhythmia annotations to binary labels ($y_i \in \{0, 1\}$) and split strictly by patient record (70/15/15).
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/ecg/task_generator.py`.
  - *Effort:* 25 mins.
  - *Criteria:* Patient record IDs in train, val, and test are completely disjoint.

#### D. Expected Output
Processed task dataset tensors stored in `data/processed/`.

#### E. Problems We May Encounter
Memory exhaustion when holding all continuous signals in RAM.

#### F. Problem Resolution Plan
Stream segments using memory-mapped NumPy files or PyTorch `Dataset` generators.

#### G. Testing Plan
Run `tests/test_ecg_pipeline.py` testing shapes, sample rates, and partition independence.

#### H. Exit Criteria
Dataset generated, verified zero-leakage, and validated with summary statistics.

#### I. Human Review
Review class balance between normal and pathological beats.

#### J. Checkpoint
`git tag phase3-ecg-pipeline-verified`.

---

### Phase 4: CNN-LSTM Clinical Risk Model ($R_i$)

#### A. Purpose
Train and validate the 1D-CNN + 2-layer LSTM network to output clinical risk scores $R_i \in [0, 1]$ adhering to the $0.74$M parameter budget.

#### B. Prerequisites
Phase 3 complete.

#### C. Tasks
* **T4-01 (Model Architecture):** Build PyTorch `CNNLSTMRisk` with 1D-CNN feature extractor, LSTM temporal layer, and sigmoid head.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/models/cnn_lstm_risk.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Exact parameter count $\approx 0.74$M; forward pass takes $<15$ ms.
* **T4-02 (Training Loop & BCE Loss):** Implement Adam optimizer ($\text{lr}=10^{-3}$, batch=32) with binary cross-entropy loss and checkpointing.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/training/train_risk_model.py`.
  - *Effort:* 30 mins.
  - *Criteria:* Loss decreases monotonically; validation loss monitored for early stopping.
* **T4-03 (Model Evaluation):** Evaluate on test patient records; compute Accuracy, Precision, Recall, F1, and ROC-AUC.
  - *Assigned to:* AI Autonomous.
  - *Files:* `tests/test_risk_model.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Test Accuracy $\ge 93.4\%$, AUC $\ge 0.95$ matching Table 6.

#### D. Expected Output
Trained weights saved to `checkpoints/cnn_lstm_risk.pt` with evaluation metrics report.

#### E. Problems We May Encounter
Overfitting on training patients due to morphological similarities.

#### F. Problem Resolution Plan
Apply dropout ($0.2$) and weight decay ($10^{-4}$); verify performance exclusively on test records.

#### G. Testing Plan
Execute `pytest tests/test_risk_model.py`.

#### H. Exit Criteria
Model loads, executes in $<15$ ms, and satisfies performance criteria.

#### I. Human Review
Verify ROC-AUC curve and confusion matrix.

#### J. Checkpoint
`git tag phase4-risk-model-trained`.

---

### Phase 5: Deadline Violation Predictor ($P_{miss,i}$)

#### A. Purpose
Implement and train an MLP module predicting the probability that an incoming task will breach its deadline based on dynamic system state.

#### B. Prerequisites
Phase 1 and Phase 4 complete.

#### C. Tasks
* **T5-01 (Historical Data Generation):** Run simulator with baseline schedulers to harvest 50,000 state-outcome tuples $(S_i, y_{miss,i})$.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/training/generate_predictor_data.py`.
  - *Effort:* 25 mins.
  - *Criteria:* Generates balanced violation dataset.
* **T5-02 (MLP Predictor Implementation):** Implement 3-layer MLP ($0.18$M parameters) taking $S_i = \{U_t, Q_t, \psi_i, D_i, R_i\}$ and outputting $P_{miss,i} \in [0, 1]$.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/models/deadline_predictor.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Inference latency $<4$ ms.
* **T5-03 (Predictor Training & Integration):** Train predictor with BCE loss and integrate into simulator state vector.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/training/train_predictor.py`.
  - *Effort:* 25 mins.
  - *Criteria:* Validation AUC $\ge 0.90$.

#### D. Expected Output
Trained weights saved to `checkpoints/deadline_predictor.pt`.

#### E. Problems We May Encounter
Feature colinearity causing prediction instability.

#### F. Problem Resolution Plan
Normalize input features to standard scalar range $[0, 1]$ before passing into MLP.

#### G. Testing Plan
Run `tests/test_predictor.py`.

#### H. Exit Criteria
Predictor inference outputs valid probabilities in $<4$ ms.

#### I. Human Review
Review calibration curve of predicted vs. observed violation frequencies.

#### J. Checkpoint
`git tag phase5-predictor-complete`.

---

### Phase 6: DQN Scheduler Implementation

#### A. Purpose
Implement and train the 26-action Double DQN agent with experience replay and multi-objective reward shaping.

#### B. Prerequisites
Phases 1, 4, and 5 complete.

#### C. Tasks
* **T6-01 (Q-Network & Replay Buffer):** Build `DQNNetwork` ($0.32$M parameters) and `PrioritizedReplayBuffer` (capacity 50,000).
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/schedulers/dqn_scheduler.py`, `src/training/replay_buffer.py`.
  - *Effort:* 25 mins.
  - *Criteria:* State dimension matches observation vector; action dimension = 26.
* **T6-02 (Multi-Objective Reward Function):** Implement Equation 29: $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ with normalized latency.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/training/reward_shaping.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Numerical unit tests verify correct penalty for missed deadlines and reward for high-risk completions.
* **T6-03 (Training Loop & Target Updates):** Train DQN over 50,000 steps with $\epsilon$-greedy exploration (1.0 to 0.05) and target network updates every 200 steps.
  - *Assigned to:* AI + Human (Human reviews training stability).
  - *Files:* `src/training/train_dqn.py`.
  - *Effort:* 45 mins.
  - *Criteria:* TD loss converges; average episode reward increases.

#### D. Expected Output
Trained checkpoint `checkpoints/dqn_scheduler.pt` and training curve logs.

#### E. Problems We May Encounter
Q-value divergence, policy collapse to a single node.

#### F. Problem Resolution Plan
Use gradient clipping (max norm 1.0) and Double DQN action evaluation.

#### G. Testing Plan
Run `tests/test_dqn.py` verifying state shape, action bounds, and loss reduction.

#### H. Exit Criteria
Agent completes training without NaN values and demonstrates superior scheduling policy.

#### I. Human Review
Inspect reward curve and node allocation distribution.

#### J. Checkpoint
`git tag phase6-dqn-trained`.

---

### Phase 7: System Integration

#### A. Purpose
Connect all modules into a unified, end-to-end pipeline: ECG $\to$ Risk Model $\to$ Task Generator $\to$ Predictor $\to$ DQN $\to$ Simulation $\to$ Metrics.

#### B. Prerequisites
Phases 1 through 6 complete.

#### C. Tasks
* **T7-01 (Pipeline Orchestrator):** Build `CRAILCSPipeline` executing continuous tasks through the full closed loop.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/pipeline.py`.
  - *Effort:* 30 mins.
  - *Criteria:* End-to-end data flow passes all assertions.
* **T7-02 (Integration Test Suite):** Run 100-patient streaming simulation test.
  - *Assigned to:* AI Autonomous.
  - *Files:* `tests/test_integration.py`.
  - *Effort:* 20 mins.
  - *Criteria:* No runtime exceptions, memory leaks, or unhandled task exceptions.

#### D. Expected Output
Seamless end-to-end pipeline callable via Python API or CLI.

#### E. Problems We May Encounter
Interface impedance (tensor vs numpy type mismatches).

#### F. Problem Resolution Plan
Enforce explicit type conversion (`.detach().cpu().numpy()`) at module boundaries.

#### G. Testing Plan
Execute `pytest tests/test_integration.py`.

#### H. Exit Criteria
1,000 continuous tasks processed end-to-end with complete metric accounting.

#### I. Human Review
Observe real-time console log of task scheduling decisions.

#### J. Checkpoint
`git tag phase7-system-integrated`.

---

### Phase 8: Experimental Evaluation & Benchmarking

#### A. Purpose
Replicate the paper's primary experimental evaluations (Tables 5, 7, 9) under peak workload ($\lambda = 20$) and multi-load sweeps ($\lambda \in [8, 25]$).

#### B. Prerequisites
Phase 7 complete.

#### C. Tasks
* **T8-01 (Multi-Load Workload Sweeps):** Run simulations across arrival rates $\lambda \in [8, 10, 12.5, 15, 17.5, 20, 22.5, 25]$.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/load_sweep.py`.
  - *Effort:* 35 mins.
  - *Criteria:* Logs Latency, Violation Rate, Risk Coverage, Utilization, and Energy.
* **T8-02 (Statistical Testing Across 10 Seeds):** Run 10 independent random seeds for CRAI-LCS and baselines; compute Mean $\pm$ Std and Wilcoxon signed-rank test.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/statistical_tests.py`.
  - *Effort:* 30 mins.
  - *Criteria:* Computes $p$-values matching Table 9 ($p < 0.05$).
* **T8-03 (Plot Generation):** Generate publication-quality figures (Figures 3, 4, 5, 6, 7).
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/plot_figures.py`.
  - *Effort:* 20 mins.
  - *Criteria:* High-DPI PNGs saved in `results/graphs/`.

#### D. Expected Output
Benchmark logs in `results/logs/` and figures in `results/graphs/`.

#### E. Problems We May Encounter
Variance across seeds exceeding expected boundaries.

#### F. Problem Resolution Plan
Verify environment resets use seeded random number generators (`np.random.seed`, `torch.manual_seed`).

#### G. Testing Plan
Execute `pytest tests/test_experiments.py`.

#### H. Exit Criteria
CRAI-LCS outperforms baselines with statistically significant p-values.

#### I. Human Review
Examine comparison tables and graphs against published paper figures.

#### J. Checkpoint
`git tag phase8-experiments-complete`.

---

### Phase 9: Ablation & Reliability Stress Testing

#### A. Purpose
Replicate the paper's ablation studies (Table 8) and evaluate system reliability under network jitter, node failures, and risk skew.

#### B. Prerequisites
Phase 8 complete.

#### C. Tasks
* **T9-01 (Ablation Suite):** Implement and evaluate:
  1. Full CRAI-LCS
  2. w/o Clinical Risk Estimator ($R_i = 0$)
  3. w/o Deadline Violation Predictor ($P_{miss} = 0$)
  4. w/o Constrained Reward Shaping
  5. Static Priority baseline
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/ablation_study.py`.
  - *Effort:* 30 mins.
  - *Criteria:* Results demonstrate that removing any component causes metric degradation matching Table 8.
* **T9-02 (Reward Sensitivity Analysis):** Test reward weight sets RW-1, RW-2 (default), RW-3, RW-4 (Table 10 & 11).
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/sensitivity_analysis.py`.
  - *Effort:* 25 mins.
  - *Criteria:* RW-3 produces highest risk coverage; RW-1 produces lowest latency.
* **T9-03 (Robustness Stress Testing):** Test under simulated edge node failures and high network jitter.
  - *Assigned to:* AI Autonomous.
  - *Files:* `src/experiments/stress_tests.py`.
  - *Effort:* 20 mins.
  - *Criteria:* Demonstrates system grace under partial failure.

#### D. Expected Output
Ablation table, sensitivity results, and Figure 8 plots.

#### E. Problems We May Encounter
Ablation variants collapsing into non-functional states.

#### F. Problem Resolution Plan
Isolate ablation flags in `simulation_config.yaml` to ensure clean toggling.

#### G. Testing Plan
Run `pytest tests/test_ablations.py`.

#### H. Exit Criteria
Ablation table generated showing exact comparative drop-offs.

#### I. Human Review
Confirm ablation percentages match published paper claims.

#### J. Checkpoint
`git tag phase9-ablations-complete`.

---

### Phase 10: Final Documentation & Demonstration CLI

#### A. Purpose
Provide unified CLI, clean documentation, reproducible execution scripts, and a final technical project report.

#### B. Prerequisites
Phases 0 through 9 complete.

#### C. Tasks
* **T10-01 (Unified CLI Entrypoint):** Build `main.py` allowing single-command training, evaluation, and plotting.
  - *Assigned to:* AI Autonomous.
  - *Files:* `main.py`.
  - *Effort:* 20 mins.
  - *Criteria:* `--mode demo`, `--mode train`, `--mode benchmark` operate cleanly.
* **T10-02 (Documentation & Guides):** Create comprehensive `README.md` and `docs/TECHNICAL_REPORT.md`.
  - *Assigned to:* AI + Human.
  - *Files:* `README.md`, `docs/TECHNICAL_REPORT.md`.
  - *Effort:* 25 mins.
  - *Criteria:* Clear installation, architecture explanation, and reproduction steps.
* **T10-03 (Final QA & Clean Install Test):** Run a clean execution test in an isolated test script to verify zero path dependencies.
  - *Assigned to:* AI Autonomous.
  - *Files:* `tests/test_clean_run.py`.
  - *Effort:* 15 mins.
  - *Criteria:* Entire project runs out-of-the-box.

#### D. Expected Output
Fully packaged, demonstrable research repository with complete documentation.

#### E. Problems We May Encounter
Hardcoded absolute paths breaking clean installation.

#### F. Problem Resolution Plan
Enforce `pathlib.Path` relative to project root across all file operations.

#### G. Testing Plan
Execute full test suite `pytest tests/`.

#### H. Exit Criteria
All unit, integration, and regression tests pass 100%.

#### I. Human Review
Final human review and project acceptance.

#### J. Checkpoint
`git tag v1.0.0-final-release`.


---

## Chapter 8: Exhaustive Phase-by-Phase Practical Blueprint (Start-to-End Hinglish Guide)

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Provenance:** Nature Portfolio *Scientific Reports* 2026 (DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9))  
**Document Purpose:** Scratch se production tak har phase ko start-to-end kaise build aur verify karna hai uska complete practical blueprint.

---

### Complete Phase Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CRAI-LCS COMPLETE 11-PHASE ROADMAP                   │
└────────────────────────────────────────────────────────────────────────┘
 [Phase 0: Setup & Seed] ──► [Phase 1: Continuum Sim] ──► [Phase 2: Baselines]
                                                                 │
 [Phase 5: Predictor MLP] ◄── [Phase 4: CNN-LSTM] ◄── [Phase 3: MIT-BIH Data]
        │
 [Phase 6: Double DQN] ──► [Phase 7: Integration] ──► [Phase 8: Benchmarking]
                                                                 │
 [Phase 10: Production QA] ◄── [Phase 9: Statistical Rigor & PTB-XL] ◄───┘
```

---

### Table of Contents
1. [Phase 0: Environment Setup, Global Determinism & Repository Audit](#phase-0-environment-setup-global-determinism--repository-audit)
2. [Phase 1: Discrete-Event Continuum Simulator (Hardware & Latency Engine)](#phase-1-discrete-event-continuum-simulator-hardware--latency-engine)
3. [Phase 2: Five Heuristic Baseline Schedulers](#phase-2-five-heuristic-baseline-schedulers)
4. [Phase 3: PhysioNet MIT-BIH Ingestion & Zero-Leakage Pipeline](#phase-3-physionet-mit-bih-ingestion--zero-leakage-pipeline)
5. [Phase 4: 1D-CNN-LSTM Clinical Risk Network Training ($R_i$)](#phase-4-1d-cnn-lstm-clinical-risk-network-training-r_i)
6. [Phase 5: Pre-Execution Deadline Predictor & 3-Stage Bootstrapping](#phase-5-pre-execution-deadline-predictor--3-stage-bootstrapping)
7. [Phase 6: Double DQN Scheduler & Multi-Objective Reward Shaping](#phase-6-double-dqn-scheduler--multi-objective-reward-shaping)
8. [Phase 7: End-to-End Closed-Loop System Integration & Fail-Safe Protocol](#phase-7-end-to-end-closed-loop-system-integration--fail-safe-protocol)
9. [Phase 8: Peak-Load Experimental Benchmarking ($\lambda = 20$ tasks/sec)](#phase-8-peak-load-experimental-benchmarking-lambda--20-taskssec)
10. [Phase 9: Statistical Rigor, Ablation Study & PTB-XL Generalization](#phase-9-statistical-rigor-ablation-study--ptb-xl-generalization)
11. [Phase 10: Production Delivery, Automated QA Suite & Unified CLI](#phase-10-production-delivery-automated-qa-suite--unified-cli)
12. [Critical 4 Engineering Gotchas & Solutions](#critical-4-engineering-gotchas--solutions)
13. [Turnkey Reproduction CLI Commands](#turnkey-reproduction-cli-commands)

---

### Phase 0: Environment Setup, Global Determinism & Repository Audit

#### 1. Goal (Kya achieve karna hai)
Ek isolated development environment banana, dependencies configure karna, aur bitwise reproducibility lock karna taaki har run pe identical ECG batches aur Poisson arrival sequences generate hon.

#### 2. Prerequisites & Inputs
* Linux OS, Python 3.10+, Git, workspace access.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Virtual Environment Create Karein:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Dependencies Install Karein:**
   `requirements.txt` file banayein aur install karein:
   ```text
   torch>=2.0.0
   gymnasium>=0.28.1
   wfdb>=4.1.0
   numpy>=1.24.0
   scipy>=1.10.0
   scikit-learn>=1.2.0
   matplotlib>=3.7.0
   pytest>=7.3.0
   pyyaml>=6.0
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. **Global Determinism Utility Code Karein:**
   `src/utils/seed.py` create karein jo exact seeds set kare:
   ```python
   import random, os, numpy as np, torch

   def set_global_seed(seed: int = 42):
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```
4. **Folder Architecture Initialize Karein:**
   ```bash
   mkdir -p src/{ecg,models,simulator,baselines,training,experiments,utils} data/{raw,processed} checkpoints results/graphs tests
   ```

#### 4. Deliverable Files
* `requirements.txt`
* `src/utils/seed.py`
* `configs/config.yaml`

#### 5. Verification & Exit Criteria
* `python3 -c "import torch, gymnasium, wfdb; from src.utils.seed import set_global_seed; set_global_seed(42); print('Phase 0 Passed')"` smoothly run ho aur exit code 0 de.

---

### Phase 1: Discrete-Event Continuum Simulator (Hardware & Latency Engine)

#### 1. Goal (Kya achieve karna hai)
Hospital ke physical 3-tier computing hierarchy (200 monitors, 25 edge gateways, 2 cloud servers) aur mathematical latency/energy models ko ek Farama Gymnasium environment mein simulate karna.

#### 2. Prerequisites & Inputs
* Phase 0 completed.
* Paper Equations (8)--(10) and Hardware Parameter Table.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Task Representation (`src/simulator/task.py`):**
   ```python
   from dataclasses import dataclass

   @dataclass
   class HealthcareTask:
       task_id: int
       delta_i: float       # Payload size in Bytes (~3.6 KB)
       psi_i: float         # Compute demand in CPU cycles (5e6 to 2e7)
       deadline_ms: float   # 100-150 ms if urgent; 400-800 ms if routine
       clinical_risk: float # R_i in [0.0, 1.0]
       arrival_time: float
   ```
2. **Node Hardware Modeling (`src/simulator/nodes.py`):**
   * **25 Edge Nodes:** Heterogeneous computing capacity $f_j \in [800, 4500]$ MIPS, local bandwidth $B_j \in [10, 50]$ Mbps.
   * **2 Cloud Instances:** High compute capacity $f_c \in [12\text{k}, 35\text{k}]$ MIPS, WAN bandwidth $B_c \in [100, 500]$ Mbps.
3. **Latency & Energy Equations (`src/simulator/latency_energy.py`):**
   * **Edge Offloading Latency:**
     $$L_i^{edge}(j) = T_{tr} + \frac{\delta_i}{B_j} + \frac{Q_j(t)}{f_j} + \frac{\psi_i}{f_j}$$
   * **Cloud Offloading Latency:**
     $$L_i^{cloud} = T_{tr} + T_{e\to c} + \frac{\delta_i}{B_c} + \frac{Q_c(t)}{f_c} + \frac{\psi_i}{f_c}$$
     *(Yahan $T_{e\to c} \sim \mathcal{U}(45, 120)\text{ ms}$ WAN propagation delay hai).*
   * **Energy Model:**
     $$E_j = P_{idle} \cdot \Delta t + P_{active} \cdot \left(\frac{\psi_i}{f_j}\right)$$
4. **Farama Gymnasium Environment (`src/simulator/edge_cloud_env.py`):**
   * `action_space = spaces.Discrete(26)` ($0 \dots 24$: Edge Nodes, $25$: Cloud).
   * `observation_space = spaces.Box(low=0.0, high=np.inf, shape=(54,), dtype=np.float32)`.
   * `step(action)` method tasks ko process kare, queues update kare, latency calculate kare, aur observation return kare.

#### 4. Deliverable Files
* `src/simulator/task.py`
* `src/simulator/nodes.py`
* `src/simulator/latency_energy.py`
* `src/simulator/edge_cloud_env.py`

#### 5. Verification & Exit Criteria
* Synthetic 1,000 tasks run karne par queues stable rahein aur hand-calculated latency values simulation output se $100\%$ match karein.

---

### Phase 2: Five Heuristic Baseline Schedulers

#### 1. Goal (Kya achieve karna hai)
Benchmarking ke liye 5 standard heuristic schedulers ka unified `schedule(observation) -> int` interface banana.

#### 2. Prerequisites & Inputs
* Phase 1 Gymnasium simulator ready.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Abstract Interface Banayein:**
   ```python
   class BaseScheduler:
       def schedule(self, observation: np.ndarray, task: HealthcareTask) -> int:
           raise NotImplementedError
   ```
2. **5 Heuristics Implement Karein (`src/baselines/heuristics.py`):**
   * **`RandomScheduler`:** Uniform randomly $a \in [0, 25]$ select kare.
   * **`RoundRobinScheduler`:** Edge nodes par cyclic pointer maintain kare: `a = (last_a + 1) % 25`.
   * **`FIFOScheduler` (Shortest Queue First):** Sabhi 25 edge nodes ki queue check kare aur minimum queue length wale node par assign kare: $a = \arg\min_j Q_j(t)$.
   * **`EDFScheduler` (Earliest Deadline First):** Sorted priority buffer se minimum deadline wale task ko sabse fast node ($f_j$) par dispatch kare.
   * **`RiskPriorityScheduler`:** Rule-based decision: agar $R_i \ge 0.5$ (urgent) hai toh fastest local edge node par bheje; agar normal hai toh cloud offload kare.
3. **Baseline Benchmark Harness:**
   `tests/test_baselines.py` mein standard 1,000-task Poisson arrival run karke latency aur violation rate compare karein.

#### 4. Deliverable Files
* `src/baselines/heuristics.py`
* `tests/test_baselines.py`

#### 5. Verification & Exit Criteria
* Schedulers bina error execute hon aur expected hierarchy verify ho: EDF aur Risk-Priority schedulers Random/FIFO se significantly behtar violation rate show karein.

---

### Phase 3: PhysioNet MIT-BIH Ingestion & Zero-Leakage Pipeline

#### 1. Goal (Kya achieve karna hai)
PhysioNet MIT-BIH arrhythmia dataset se continuous raw ECG waveforms download karna, Z-score standardizing karna, 5-second windows banana, aur patient leakage rokna.

#### 2. Prerequisites & Inputs
* Python package `wfdb`.
* PhysioNet open-access repository (`mitdb/1.0.0`).

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Dataset Downloader (`download_mitbih.py`):**
   ```python
   import wfdb
   wfdb.dl_database('mitdb', dl_dir='data/raw/mitbih')
   ```
   Isse 48 records (`.dat`, `.hea`, `.atr` files) download honge.
2. **Signal Segmentation & Normalization (`src/ecg/preprocessing.py`):**
   * Sampling frequency $f_s = 360\text{ Hz}$.
   * **5-Second Windowing:** Segment length = $360 \times 5 = 1800\text{ samples}$ ($\approx 3.6\text{ KB}$ payload).
   * **Z-Score Normalization (Equation 17):**
     $$\tilde{x}_i[k] = \frac{x_i[k] - \mu_i}{\sigma_i + 10^{-8}}, \quad k \in \{1 \dots 1800\}$$
     *(Respiration aur electrode baseline wander ko eliminate karta hai).*
3. **AAMI EC57 Clinical Risk Ground Truth Mapping:**
   * **Emergency ($y_i = 1$):** Segments containing Ventricular Fibrillation (VF), PVC ('V'), Ventricular Tachycardia (VT), Flutter ('!'), AFIB.
   * **Routine ($y_i = 0$):** Segments containing Normal Sinus Rhythm ('N'), Bundle Branch Blocks ('L', 'R'), Paced rhythms ('/').
4. **Zero-Leakage Patient-Record Partitioning:**
   Random shuffling strictly prohibited hai! Records ko partition karein:
   * **Training Set (70%, 31 records):** `100..124, 200..215`
   * **Validation Set (15%, 6 records):** `219, 220, 221, 222, 223, 228`
   * **Testing Set (15%, 5 records):** `230, 231, 232, 233, 234`

#### 4. Deliverable Files
* `download_mitbih.py`
* `src/ecg/preprocessing.py`
* `data/processed/ecg_tasks_split.npz`

#### 5. Verification & Exit Criteria
* Hard assertion check pass kare: `assert len(set(train_ids) & set(test_ids)) == 0`.
* Segments array shape: `(N, 1800)` with mean $\approx 0$ and std $\approx 1$.

---

### Phase 4: 1D-CNN-LSTM Clinical Risk Network Training ($R_i$)

#### 1. Goal (Kya achieve karna hai)
Raw 1800-sample ECG segment ko analyze karke patient ka continuous Clinical Risk Score ($R_i \in [0, 1]$) predict karne wala deep neural network train karna.

#### 2. Prerequisites & Inputs
* Phase 3 ka preprocessed MIT-BIH dataset (`ecg_tasks_split.npz`).
* Parameter budget limit: $\le 0.74\text{M}$ weights; Execution budget: $\le 14.8\text{ ms}$.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **PyTorch Model Architecture (`src/models/cnn_lstm_risk.py`):**
   ```python
   import torch
   import torch.nn as nn

   class ClinicalRiskCNNLSTM(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv = nn.Sequential(
               nn.Conv1d(1, 32, kernel_size=5, padding=2),
               nn.ReLU(),
               nn.MaxPool1d(2),          # Output: (B, 32, 900)
               nn.Conv1d(32, 64, kernel_size=5, padding=2),
               nn.ReLU(),
               nn.MaxPool1d(2)           # Output: (B, 64, 450)
           )
           self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
           self.fc = nn.Sequential(
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(32, 1),
               nn.Sigmoid()              # Continuous Risk R_i in [0, 1]
           )

       def forward(self, x):
           # x: (B, 1, 1800)
           c = self.conv(x)             # (B, 64, 450)
           c = c.permute(0, 2, 1)       # (B, 450, 64) for LSTM
           _, (hn, _) = self.lstm(c)    # hn: (2, B, 64)
           out = self.fc(hn[-1])        # (B, 1)
           return out
   ```
2. **Handling Class Imbalance:**
   Dataset mein normal beats zyada hoti hain. Weighted Binary Cross-Entropy loss use karein:
   ```python
   pos_weight = torch.tensor([3.42])
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```
3. **Training Protocol (`src/training/train_risk_model.py`):**
   * Optimizer: Adam ($lr = 10^{-3}$, weight decay $= 10^{-4}$).
   * Batch size: 64; Epochs: 50 with Early Stopping on validation AUC-ROC.
   * Model checkpoint save karein: `checkpoints/cnn_lstm_risk.pt`.

#### 4. Deliverable Files
* `src/models/cnn_lstm_risk.py`
* `src/training/train_risk_model.py`
* `checkpoints/cnn_lstm_risk.pt`

#### 5. Verification & Exit Criteria
* Unseen test set par: **Accuracy $\ge 94.2\%$**, **AUC-ROC $\ge 0.962$**.
* PyTorch parameter check: `sum(p.numel() for p in model.parameters()) == 741,409` ($\approx 0.74\text{M}$).
* Single-window CPU forward pass latency $\le 14.8\text{ ms}$.

---

### Phase 5: Pre-Execution Deadline Predictor & 3-Stage Bootstrapping

#### 1. Goal (Kya achieve karna hai)
Task assign hone se pehle hi forecast karna ki kya wo current system load ke chalte deadline miss karega ($P_{miss,i} \in [0, 1]$).

#### 2. Prerequisites & Inputs
* Phase 1 simulator aur Phase 4 risk model.
* Parameter budget: $\le 0.18\text{M}$; Inference budget: $\le 3.6\text{ ms}$.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Cold-Start Dilemma Solve Karein (3-Stage Bootstrapping):**
   * DQN agent ko observation ke andar $P_{miss,i}$ chahiye hota hai, lekin start mein bina simulation chalaye violation labels nahi hote.
2. **Stage 1: Warm-Up Data Collection (`src/simulator/bootstrap_predictor.py`):**
   * Simulator ko heuristic policies (FIFO aur Random) ke saath 50,000 steps run karein.
   * Har step par input features $[Q_j, U_j, \psi_i, D_i, R_i]$ aur actual outcome ($y_{miss} \in \{0, 1\}$) log karein.
   * Save karein: `data/processed/predictor_warmup_data.npz`.
3. **Stage 2: 3-Layer MLP Training (`src/training/train_predictor.py`):**
   * Architecture: `Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(16) -> Linear(1) -> Sigmoid`.
   * Standard Binary Cross-Entropy loss par train karein ($lr = 5 \times 10^{-4}$).
4. **Stage 3: Online Deployment:**
   * Checkpoint save karein: `checkpoints/deadline_predictor.pt`.
   * Fast inference hook banayein jo gym environment state vector mein $P_{miss,i}$ append kare.

#### 4. Deliverable Files
* `src/models/deadline_predictor.py`
* `src/simulator/bootstrap_predictor.py`
* `src/training/train_predictor.py`
* `checkpoints/deadline_predictor.pt`

#### 5. Verification & Exit Criteria
* Predictor test set par **AUC-ROC $\ge 0.924$** achieve kare.
* Total weights $\approx 0.18\text{M}$ aur CPU inference time $\le 3.6\text{ ms}$.

---

### Phase 6: Double DQN Scheduler & Multi-Objective Reward Shaping

#### 1. Goal (Kya achieve karna hai)
54-dimensional state space ko analyze karke 26 discrete actions mein se optimal dispatch decision lene wala Double Deep Q-Network train karna.

#### 2. Prerequisites & Inputs
* Pre-trained `cnn_lstm_risk.pt` aur `deadline_predictor.pt`.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **54-Dimensional State Vector:**
   $$s_t = [Q_1 \dots Q_{25}, U_1 \dots U_{25}, R_i, P_{miss,i}, D_i, \psi_i]$$
   *(25 Edge node queues, 25 Edge utilizations, clinical risk score, predicted miss probability, deadline, compute demand).*
2. **Normalized Multi-Objective Reward Function (Equation 29):**
   $$R_t = -\left[ \alpha \left(\frac{L_i}{D_i}\right) + \beta P_{miss,i} - \gamma R_i \cdot \mathbf{1}_{\{L_i \le D_i\}} \right] + \eta \bar{U}_{edge}$$
   Default weights (Balanced Profile RW-2): $\alpha = 0.35, \beta = 0.25, \gamma = 0.30, \eta = 0.10$.
   > **Critical Gotcha Solved:** Latency $L_i$ milliseconds mein hoti hai ($100-800$), jabki risk $[0, 1]$ mein hai. Agar $L_i$ ko deadline $D_i$ se divide na kiya jaye, toh reward explode ho jata hai aur agent risk ko ignore kar deta hai!
3. **Double DQN Architecture & Decoupling (`src/models/dqn_scheduler.py`):**
   * Q-Network: 3 hidden layers (`54 -> 128 -> 128 -> 64 -> 26`).
   * Overestimation bias khatam karne ke liye:
     * Online network $\theta$ action choose kare: $a^* = \arg\max_a Q(s', a; \theta)$.
     * Target network $\theta^-$ target Q-value evaluate kare: $Y_t = r + \gamma Q(s', a^*; \theta^-)$.
4. **Training Loop (`src/training/train_dqn.py`):**
   * Replay buffer capacity: 50,000 transitions; Batch size: 64.
   * Epsilon-greedy exploration schedule: $1.0 \to 0.05$ (decay over 500 episodes).
   * Target network sync frequency: Every 200 steps.
   * Total episodes: 1,000. Save checkpoint: `checkpoints/dqn_scheduler.pt`.

#### 4. Deliverable Files
* `src/models/dqn_scheduler.py`
* `src/training/train_dqn.py`
* `checkpoints/dqn_scheduler.pt`

#### 5. Verification & Exit Criteria
* TD loss smoothly converge kare ($<0.01$).
* Average episode reward negative se positive territory mein enter kare.
* Total DQN weights $\le 0.32\text{M}$ aur CPU forward pass $\le 5.7\text{ ms}$.

---

### Phase 7: End-to-End Closed-Loop System Integration & Fail-Safe Protocol

#### 1. Goal (Kya achieve karna hai)
Data ingestion, neural networks, simulator, aur scheduler ko sub-25 ms pipeline mein connect karna aur clinical fail-safe hardware preemption provide karna.

#### 2. Prerequisites & Inputs
* Phases 1 se 6 ke saare pre-trained checkpoints aur simulator modules.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Master Pipeline Orchestrator (`src/pipeline.py`):**
   Continuous closed loop banayein:
   $$\text{Raw ECG} \xrightarrow{\text{CNN-LSTM}} R_i \xrightarrow{\text{MLP}} P_{miss,i} \xrightarrow{\text{DQN}} a_t \xrightarrow{\text{Simulator}} \text{Execution} \to R_t$$
2. **Zero-Latency Tensor-NumPy Boundary:**
   Gymnasium state NumPy arrays aur PyTorch GPU/CPU tensors ke beech pinned memory use karein taaki zero garbage-collection pauses hon.
3. **Clinical Fail-Safe Interrupt Protocol:**
   Agar CNN-LSTM predict kare ki $R_i \ge 0.80$ (Acute Ventricular Fibrillation / Tachycardia):
   * DQN agent aur Cloud WAN offloading ko **immediately bypass** karein.
   * Patient ke local edge gateway par hardware preemption trigger karein ($<100\text{ ms}$ hard guarantee).
4. **Integration Testing (`tests/test_integration.py`):**
   100 streaming patients ka continuous 30-minute simulation run karein.

#### 4. Deliverable Files
* `src/pipeline.py`
* `tests/test_integration.py`

#### 5. Verification & Exit Criteria
* Pipeline ka total end-to-end AI latency budget $\le 24.1\text{ ms}$ ($14.8 + 3.6 + 5.7\text{ ms}$).
* 100-patient streaming simulation bina single unhandled exception ya memory leak ke complete ho.

---

### Phase 8: Peak-Load Experimental Benchmarking ($\lambda = 20$ tasks/sec)

#### 1. Goal (Kya achieve karna hai)
Paper ke primary benchmarks replicate karna aur prove karna ki CRAI-LCS peak hospital load par baselines se superior hai.

#### 2. Prerequisites & Inputs
* Phase 7 integrated pipeline.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Multi-Load Poisson Workload Sweeps (`src/experiments/load_sweep.py`):**
   Arrival rates simulate karein: $\lambda \in [8, 10, 12.5, 15, 17.5, 20, 22.5, 25]$ tasks/sec.
2. **10 Independent Random Seeds:**
   CRAI-LCS aur sabhi 5 baselines ko 10 seeds (`seeds = [42, 100, 2024, ...]`) par run karein taaki mean aur standard deviation measure ho sake.
3. **5 Key Metrics Log Karein:**
   * End-to-End Latency (ms)
   * Deadline Violation Rate (%)
   * Clinical Risk Coverage (%)
   * Edge Core Utilization (%)
   * Energy Consumption per 1,000 tasks (Joules)
4. **Publication Graphs Plot Karein (`src/experiments/plot_figures.py`):**
   * Figure 3: Latency vs Arrival Rate ($\lambda$)
   * Figure 4: Deadline Violations vs Arrival Rate ($\lambda$)
   * Figure 5: Critical Risk Coverage vs Arrival Rate ($\lambda$)

#### 4. Deliverable Files
* `src/experiments/load_sweep.py`
* `src/experiments/plot_figures.py`
* `results/benchmarks_lambda20.csv`
* `results/graphs/figure3_latency_sweep.png`

#### 5. Verification & Exit Criteria
* Peak load ($\lambda = 20$) par metrics paper se bitwise align karein:
  * **Latency:** $148.0\text{ ms}$ (vs $182.0\text{ ms}$ for best baseline EADRL).
  * **Violations:** $2.80\%$ (vs $5.90\%$ for EADRL).
  * **Risk Coverage:** $96.40\%$ (vs $84.20\%$ for EADRL).
  * **Energy:** $284.0\text{ Joules}$ (vs $342.0\text{ Joules}$ for EADRL).

---

### Phase 9: Statistical Rigor, Ablation Study & PTB-XL Generalization

#### 1. Goal (Kya achieve karna hai)
Prove karna ki system ka performance statistical fluke nahi hai, aur model unseen clinical datasets par generalizable hai.

#### 2. Prerequisites & Inputs
* Benchmark CSV data aur PhysioNet PTB-XL database.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Wilcoxon Signed-Rank Test (`src/experiments/statistical_tests.py`):**
   10 paired runs par non-parametric test run karein:
   ```python
   from scipy.stats import wilcoxon
   stat, p_value = wilcoxon(crai_lcs_latencies, eadrl_latencies)
   print(f"Wilcoxon p-value: {p_value}")
   assert p_value < 0.05, "Null hypothesis H0 could not be rejected!"
   ```
   *(Paper result: $p = 0.00195$).*
2. **4-Variant Ablation Study (`src/experiments/ablation_study.py`):**
   * **Variant A (Without Risk Estimator $R_i=0$):** Risk coverage $96.4\% \to 78.4\%$ gir jati hai.
   * **Variant B (Without Predictor $P_{miss}=0$):** Deadline violations double ho jati hain ($2.80\% \to 4.85\%$).
   * **Variant C (Without Reward Normalization):** Policy explode hoti hai, latency $+50.4\text{ ms}$ barh jati hai.
3. **PTB-XL Cross-Dataset Generalization (`src/experiments/generalization_test.py`):**
   Unseen $21{,}837$ records wale PTB-XL dataset par zero-shot test run karein aur metric retention verify karein:
   * **MIT-BIH Test:** Accuracy $94.20\%$, AUC $0.962$.
   * **PTB-XL Test:** Accuracy $92.60\%$, AUC $0.948$.
   * **Performance Retention Rate:** $\mathbf{98.3\%}$ (Hardware-invariance confirmed).

#### 4. Deliverable Files
* `src/experiments/statistical_tests.py`
* `src/experiments/ablation_study.py`
* `src/experiments/generalization_test.py`
* `results/ablation_results.csv`

#### 5. Verification & Exit Criteria
* Wilcoxon $p$-value $0.00195 < 0.01$ assert pass kare.
* PTB-XL retention rate $\ge 98.0\%$ report ho.

---

### Phase 10: Production Delivery, Automated QA Suite & Unified CLI

#### 1. Goal (Kya achieve karna hai)
Entire codebase ko single turnkey entrypoint aur automated CI/CD test pyramid ke saath package karna.

#### 2. Prerequisites & Inputs
* Phases 0 through 9 complete.

#### 3. Step-by-Step Kaise Karna Hai (Start to End)
1. **Unified CLI Author Karein (`main.py`):**
   Argparse ke through user ko intuitive commands provide karein:
   ```bash
   python main.py --mode demo        # Run quick 100-task simulation with summary
   python main.py --mode train       # Run full 3-model sequential training pipeline
   python main.py --mode benchmark   # Execute 10-seed peak load test & generate CSV
   python main.py --mode plot        # Re-render publication-quality PNG charts
   ```
2. **7-Tier QA Test Pyramid:**
   * Tier 1: Static code linting (`flake8 src/ tests/`).
   * Tier 2: Static type assertion (`mypy src/`).
   * Tier 3: Neural network unit tests (shape bounds, budget checks).
   * Tier 4: Latency & queue mathematics unit tests.
   * Tier 5: Farama Gymnasium API contract tests.
   * Tier 6: Multi-baseline comparative sanity tests.
   * Tier 7: Clean-environment end-to-end integration test (`tests/test_clean_run.py`).
3. **Master Compendium & Documentation Check:**
   Verify karein ki [`CRAI_LCS_MASTER_ALL_IN_ONE.md`](file:///home/dhanrraj/Documents/Default%20Project/CRAI_LCS/CRAI_LCS_MASTER_ALL_IN_ONE.md) updated hai.

#### 4. Deliverable Files
* `main.py`
* `tests/test_clean_run.py`
* `docs/REPRODUCTION_CHECKLIST.md`

#### 5. Verification & Exit Criteria
* Isolated environment mein `pytest tests/ -v` run karne par all 24+ automated tests green pass karein.

---

### Critical 4 Engineering Gotchas & Solutions

| Gotcha ID | Critical Pitfall | Root Cause | CRAI-LCS Architectural Solution |
| :--- | :--- | :--- | :--- |
| **RSK-001** | **Patient Data Leakage** | Beats ko randomly shuffle karne se same patient train aur test dono mein chala jata hai $\to$ False $99\%$ accuracy. | PhysioNet dataset ko strictly Patient Record ID ke basis pe partition karein: $\text{Records}_{train} \cap \text{Records}_{test} \equiv \emptyset$. |
| **RSK-007** | **Reward Scale Explosion** | Latency milliseconds mein hoti hai ($100-800$), jabki clinical risk $[0, 1]$ mein hai. Bada latency term gradient dominate kar deta hai. | Latency ko task deadline se normalize karein: $\tilde{L}_i = L_i / D_i$. Isse saare terms $\approx \mathcal{O}(1)$ scale pe rehte hain. |
| **RSK-005** | **Predictor Cold-Start** | MLP ko train karne ke liye historical violation data chahiye, par start mein labels nahi hote. | **3-Stage Bootstrapping:** Pehle 50k steps baseline heuristics se warm-up data log karein, phir MLP train karein, phir DRL start karein. |
| **RSK-008** | **Maximization Bias** | Standard Q-learning 26 discrete actions mein Q-values ko systematically overestimate kar deta hai. | **Double DQN Decoupling:** Action selection online network $\theta$ karega, target return evaluation target network $\theta^-$ karega. |

---

### Turnkey Reproduction CLI Commands

```bash
# 1. Environment & Dependencies Setup
git clone <repo_url> && cd CRAI_LCS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v

# 2. MIT-BIH Ingestion & Zero-Leakage 5s Windowing (@ 360 Hz)
python3 download_mitbih.py
python3 src/ecg/preprocessing.py --seed 42

# 3. Model Training & Sequential Bootstrapping
python3 -m src.training.train_risk_model --output checkpoints/cnn_lstm_risk.pt
python3 -m src.simulator.bootstrap_predictor --steps 50000 --output data/processed/predictor.npz
python3 -m src.training.train_predictor --input data/processed/predictor.npz
python3 -m src.training.train_dqn --episodes 1000 --reward_profile RW-2

# 4. Comparative Benchmarks & Statistical Significance Testing
python3 evaluation.py --arrival_rate 20 --seeds 10 --output results/benchmarks_lambda20.csv
python3 -m src.experiments.statistical_tests --input results/benchmarks_lambda20.csv
python3 -m src.experiments.generalization_test --dataset ptb-xl
```


---

## Chapter 9: Code Implementation & Snippets Guide

This guide details the complete engineering process to build **CRAI-LCS (Clinical Risk-Aware AI-Enhanced Latency-Constrained Scheduling)** as formulated in the *Scientific Reports* 2026 paper.

---

### 1. Project Directory Layout

Organize [`CRAI_LCS/`](file:///home/dhanrraj/Documents/Default%20Project/CRAI_LCS) into the following modular structure:

```
CRAI_LCS/
├── data/
│   ├── raw/                      # Downloaded MIT-BIH & PTB-XL records (.dat, .hea, .atr)
│   └── processed/                # Segmented 5s windows, task dictionaries, splits
├── models/
│   ├── cnn_lstm_risk.py          # CNN-LSTM clinical risk estimator (R_i)
│   ├── deadline_predictor.py     # MLP deadline-violation predictor (P_miss,i)
│   └── dqn_scheduler.py          # DQN agent, Q-network, and replay buffer
├── simulation/
│   ├── nodes.py                  # EdgeNode and CloudInstance classes (MIPS, RAM, Power)
│   ├── task.py                   # HealthcareTask representation (delta_i, psi_i, D_i, R_i)
│   └── edge_cloud_env.py         # Discrete-event simulation environment (Gymnasium compatible)
├── baselines/
│   ├── static_priority.py        # Rule-based priority baseline (no RL)
│   ├── eadrl.py                  # Efficiency-Aware Adaptive DRL baseline
│   └── drl_fog.py                # Traditional system-centric DRL baseline
├── scripts/
│   ├── download_dataset.py       # Automated PhysioNet data downloader
│   ├── train_risk_model.py       # CNN-LSTM training & validation pipeline
│   ├── train_scheduler.py        # DQN training loop with reward shaping
│   └── run_benchmarks.py         # Comparative evaluation, ablation, Wilcoxon test
├── config/
│   └── simulation_config.yaml    # 200 devices, 25 edge nodes, 2 cloud instances, weights
└── main.py                       # Unified entrypoint CLI
```

---

### 2. Step-by-Step Implementation Blueprint

#### Step 1: Environment Setup
Install the necessary domain dependencies alongside PyTorch:
```bash
pip install wfdb gymnasium pyyaml scipy scikit-learn matplotlib
```

---

#### Step 2: Physiological Data Acquisition & Preprocessing
The system creates tasks from authentic ECG recordings rather than synthetic random generators.

1. **Download:** Query PhysioNet MIT-BIH Arrhythmia Database (`mitdb`) using `wfdb`:
   ```python
   import wfdb
   records = ['100', '101', '102', '103', '105', '106', '108', '109', '111', '112', '114', '115', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
   wfdb.dl_database('mitdb', dl_dir='data/raw', records=records)
   ```
2. **Segmentation:**
   - Sampling rate: $f_s = 360$ Hz.
   - Window duration: $5$ seconds $\to T = 1800$ samples.
   - Input size: $\delta_i \approx 3.6$ KB (1800 samples $\times$ 2 bytes).
3. **Annotation Ground Truth:**
   - Segments containing arrhythmia annotations (PVCs, ventricular flutter, atrial fibrillation) are labeled $y_i = 1$ (High Risk).
   - Normal sinus rhythm segments are labeled $y_i = 0$ (Normal).
4. **Data Normalization & Leakage Prevention:**
   - Z-score normalization: $\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$.
   - **Record-wise partitioning:** Patient records are partitioned into Train (70%), Validation (15%), and Test (15%) such that no patient's recordings appear across multiple splits.

---

#### Step 3: CNN-LSTM Clinical Risk Estimator
The risk estimator computes $R_i \in [0, 1]$ directly from normalized raw segments:

```
Input (Batch, 1, 1800)
    │
    ▼
Conv1D (Channels: 1 -> 32, Kernel: 5, Padding: 2) -> ReLU -> MaxPool1D (Stride: 2)
    │   Output shape: (Batch, 32, 900)
    ▼
Conv1D (Channels: 32 -> 64, Kernel: 5, Padding: 2) -> ReLU -> MaxPool1D (Stride: 2)
    │   Output shape: (Batch, 64, 450)
    ▼
Bidirectional or 2-Layer LSTM (Hidden: 64, batch_first=True)
    │   Last time step output: (Batch, 64)
    ▼
Fully Connected (64 -> 32) -> ReLU -> Dropout(0.2)
    │
    ▼
Linear (32 -> 1) -> Sigmoid -> Risk Score R_i in [0, 1]
```
- **Training Objective:** Binary Cross-Entropy Loss with Adam optimizer ($\text{lr} = 10^{-3}$, batch size = 32).
- **Target Performance:** $\ge 93.4\%$ accuracy, $0.95$ AUC.

---

#### Step 4: Deadline Violation Predictor Module
Prior to scheduling decisions, this module predicts whether a candidate allocation is likely to violate deadline $D_i$:
- **Input vector $S_i$:** Edge cluster queue levels $\sum Q_t$, average cluster utilization $\bar{U}_t$, task CPU demand $\psi_i$, task deadline $D_i$, and clinical risk $R_i$.
- **Model:** 3-layer MLP (`Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1) -> Sigmoid`).
- **Target:** Produces probability $P_{miss,i} \in [0, 1]$.

---

#### Step 5: Edge-Cloud Discrete-Event Simulation Environment
The environment implements the physical queuing and transmission mathematics:

1. **Infrastructure Configuration:**
   - **25 Edge Nodes ($E_1 \dots E_{25}$):** Heterogeneous capacities:
     - CPU: $800 - 4500$ MIPS
     - RAM: $16 - 64$ GB
     - Base Power: $40 - 80$ W
   - **2 Cloud Instances ($C_1, C_2$):**
     - CPU: $12,000 - 35,000$ MIPS
     - RAM: $128$ GB
     - Base Power: $180 - 350$ W
2. **Workload Arrival Engine:**
   - Poisson process generating task arrival rates $\lambda \in [8, 25]$ tasks/sec.
   - Task CPU cycles: $\psi_i \in [5 \times 10^6, 2 \times 10^7]$ cycles.
   - Deadlines:
     - High-risk tasks ($R_i \ge 0.5$): $D_i = 100 - 150$ ms.
     - Normal tasks ($R_i < 0.5$): $D_i = 400 - 800$ ms.
3. **Execution Latency Formulas:**
   - Edge Node $j$: $L_i^{edge} = T_{tr}^{(i,j)} + \frac{\delta_i}{B_j} + \frac{\psi_i}{f_j}$
     - Transmission delay $T_{tr}^{(i,j)} \in [3, 12]$ ms, bandwidth $B_j \in [10, 50]$ Mbps.
   - Cloud Instance: $L_i^{cloud} = T_{tr}^{(i,j)} + T_{e \to c} + \frac{\delta_i}{B_c} + \frac{\psi_i}{f_c}$
     - Cloud backhaul delay $T_{e \to c} \in [45, 120]$ ms, bandwidth $B_c \in [100, 1000]$ Mbps.
4. **Energy Model:**
   - Node energy $E_j = P_{base,j} \cdot t_{idle} + P_{active,j} \cdot t_{busy}$.

---

#### Step 6: Deep Q-Network (DQN) Scheduler Implementation
1. **State Space ($s_t$):**
   - Cluster queue lengths $Q_t \in \mathbb{R}^{25}$
   - Edge node utilization levels $U_t \in \mathbb{R}^{25}$
   - Incoming task clinical risk $R_i \in [0, 1]$
   - Predicted deadline violation probability $P_{miss,i} \in [0, 1]$
   - Task deadline constraint $D_i$
2. **Action Space ($a_t$):**
   - Discrete index $a_t \in \{0, 1, \dots, 25\}$:
     - Actions $0 - 24$: Assign task to Edge Node $E_1 \dots E_{25}$.
     - Action $25$: Offload to Cloud.
3. **Multi-Objective Reward Function:**
   $$R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$$
   Default weights:
   - $\alpha = 0.35$ (Latency penalty)
   - $\beta = 0.25$ (Deadline violation penalty)
   - $\gamma = 0.30$ (Clinical risk prioritization bonus)
   - $\eta = 0.10$ (Edge resource utilization incentive)
4. **Training Hyperparameters:**
   - Experience Replay Capacity: $50,000$ transitions
   - Mini-batch Size: $64$
   - Discount Factor $\lambda$: $0.98$
   - Target Network Update Frequency: every $200$ steps
   - Exploration: $\epsilon$-greedy from $1.0$ decaying to $0.05$ with rate $0.995$

---

#### Step 7: Baseline Schedulers & Ablation Models
To validate performance as reported in the publication:
- **Baseline 1 (DRL-Fog):** Standard DRL minimizing system latency without clinical risk score.
- **Baseline 2 (EADRL):** Efficiency-aware adaptive DRL.
- **Baseline 3 (Static Priority):** Rule-based earliest-deadline-first (EDF) scheduler with static priority bands.
- **Ablation Variants:**
  - CRAI-LCS without clinical risk estimator ($R_i = 0$)
  - CRAI-LCS without deadline violation predictor ($P_{miss,i} = 0$)
  - CRAI-LCS without constrained reward shaping

---

#### Step 8: Benchmarking & Statistical Validation
1. **Multi-Load Sweep:** Run task arrival rates $\lambda \in [8, 10, 12.5, 15, 17.5, 20, 22.5, 25]$ tasks/sec.
2. **Evaluation Metrics:**
   - Average End-to-End Latency (ms)
   - Deadline Violation Rate (%)
   - Clinical Risk Coverage (%) $\to \frac{\text{High-risk tasks completed before } D_i}{\text{Total high-risk tasks}}$
   - Edge Resource Utilization (%)
   - Energy Consumption (J / 1000 tasks)
3. **Statistical Verification:** Repeat across 10 random seeds and calculate the Wilcoxon signed-rank test ($p < 0.05$).


---

## Chapter 10: Risk Register, Critical Gotchas & Architectural Mitigations

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Tracking Status:** ACTIVE

---

### 1. Risk Summary Matrix

| Risk ID | Domain | Phase | Severity | Probability | Risk Summary |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RSK-001** | Dataset | Phase 3 | **Critical** | High | Patient Data Leakage across train/test splits |
| **RSK-002** | Dataset | Phase 3 | **High** | High | Severe class imbalance (Normal >> Arrhythmia) |
| **RSK-003** | Model | Phase 4 | **High** | Medium | Vanishing gradients / slow LSTM training on 1800 samples |
| **RSK-004** | Model | Phase 4 | **Medium** | Low | Inference latency exceeding 15 ms edge gateway budget |
| **RSK-005** | Predictor | Phase 5 | **High** | Medium | Cold-start bootstrapping (no historical violation data) |
| **RSK-006** | Simulator | Phase 1 | **High** | Medium | Discrete-event clock desynchronization across nodes |
| **RSK-007** | RL | Phase 6 | **Critical** | High | Latency term scale explosion in reward function |
| **RSK-008** | RL | Phase 6 | **High** | Medium | DQN Q-value overestimation across 26 discrete actions |
| **RSK-009** | Integration| Phase 7 | **Medium** | Medium | PyTorch tensor / NumPy float type mismatches |
| **RSK-010** | Benchmark | Phase 8 | **High** | Medium | High variance across random seeds invalidating Wilcoxon test |

---

### 2. Detailed Risk Specifications

#### RSK-001: Patient Data Leakage Across Splits
* **Domain:** Dataset & Data Engineering
* **Phase:** Phase 3
* **Severity:** **CRITICAL** | **Probability:** High
* **Symptoms:** Model achieves artificially inflated test accuracy ($\ge 99\%$) on MIT-BIH, but completely fails when tested on unseen patient data (PTB-XL).
* **Root Cause:** Randomly shuffling 5-second segments between train and test sets without grouping by patient record ID.
* **Detection Method:** Automated validation script asserting that the set intersection of patient IDs in `train_records`, `val_records`, and `test_records` is strictly empty.
* **Resolution:** Partition by record: records 100-124 (train), 200-215 (val), 217-234 (test).
* **Prevention:** Enforce record-wise splitting inside `src/ecg/preprocessing.py`.
* **Human Involvement:** Human verifies record partition lists before model training.

#### RSK-002: Severe Class Imbalance (Normal vs. Pathological)
* **Domain:** Dataset & Data Engineering
* **Phase:** Phase 3
* **Severity:** **HIGH** | **Probability:** High
* **Symptoms:** High classification accuracy ($90\%+$), but near-zero recall for life-threatening arrhythmias ($R_i \approx 0$).
* **Root Cause:** Normal sinus rhythm beats vastly outnumber pathological arrhythmias in ambulatory monitoring.
* **Detection Method:** Monitor Precision, Recall, and confusion matrix during validation.
* **Resolution:** Use weighted Binary Cross-Entropy loss (`pos_weight`) to penalize false negatives on critical cardiac events.
* **Prevention:** Calculate positive weight dynamically based on training split class ratio.
* **Human Involvement:** Human approves loss weighting factor.

#### RSK-003: Vanishing Gradients & Slow Recurrent Training
* **Domain:** Deep Learning Model
* **Phase:** Phase 4
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Training loss flatlines; epoch duration exceeds 20 minutes on CPU.
* **Root Cause:** Feeding 1800 raw temporal samples directly into an uncompressed LSTM layer.
* **Detection Method:** Monitor gradient norms and training step execution time.
* **Resolution:** Apply 1D-CNN feature extraction with MaxPool1D downsampling to reduce 1800 steps to 450 steps before passing into LSTM.
* **Prevention:** Enforce the hybrid CNN-LSTM architecture defined in Equation (18).
* **Human Involvement:** None required if parameter count stays within $0.74$M.

#### RSK-004: Inference Latency Exceeding Edge Budget
* **Domain:** Deep Learning Model
* **Phase:** Phase 4
* **Severity:** **MEDIUM** | **Probability:** Low
* **Symptoms:** Edge gateway inference latency exceeds 20 ms, causing task scheduling deadline breaches before assignment.
* **Root Cause:** Excessive model parameterization or oversized dense layers.
* **Detection Method:** Measure 1,000-pass forward inference time on CPU with PyTorch benchmark timer.
* **Resolution:** Cap LSTM hidden units to 64 and prune dense layers to strictly achieve $0.74$M parameters.
* **Prevention:** Profile parameter count and inference latency in unit tests.
* **Human Involvement:** None.

#### RSK-005: Cold-Start Bootstrapping for Violation Predictor
* **Domain:** Predictor & Simulation
* **Phase:** Phase 5
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Predictor cannot train because no historical deadline violation labels exist before simulation.
* **Root Cause:** Cyclic dependency between simulator execution and predictor input.
* **Detection Method:** Missing training dataset file `predictor_training_data.npz`.
* **Resolution:** Execute an exploratory warm-up simulation phase using baseline schedulers (FIFO, Random) for 50,000 steps to generate labeled state-outcome pairs.
* **Prevention:** Sequence Phase 2 (Baselines) before Phase 5 (Predictor Training).
* **Human Involvement:** None.

#### RSK-006: Discrete-Event Clock Desynchronization
* **Domain:** Simulation Engine
* **Phase:** Phase 1
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Parallel edge node queues report negative delays or tasks complete in reverse chronological order.
* **Root Cause:** Incrementing simulation time synchronously per task rather than maintaining an event priority queue.
* **Detection Method:** Automated simulation test comparing simulated task start times with prior task finish times.
* **Resolution:** Track each node's available time individually: $\text{NodeAvailableTime}_j = \max(\text{NodeAvailableTime}_j, t_{arrival}) + t_{exec}$.
* **Prevention:** Implement strict queue availability timestamp tracking in `src/simulator/nodes.py`.
* **Human Involvement:** None.

#### RSK-007: Reward Scale Explosion in RL Objective
* **Domain:** Reinforcement Learning
* **Phase:** Phase 6
* **Severity:** **CRITICAL** | **Probability:** High
* **Symptoms:** DQN completely ignores clinical risk ($R_i$) and only minimizes latency; Q-values explode to large negative values.
* **Root Cause:** Raw latency is in hundreds of milliseconds ($100-800$ ms), while risk $R_i \in [0, 1]$. The latency penalty overwhelms the clinical reward by a factor of $500\times$.
* **Detection Method:** Print individual reward components $(\alpha L_i, \beta P_{miss}, \gamma R_i, \eta U_{edge})$ during early training.
* **Resolution:** Normalize latency by the task deadline: $L_{norm} = L_i / D_i$.
* **Prevention:** Enforce reward normalization in `src/training/reward_shaping.py`.
* **Human Involvement:** Human reviews reward normalization formula.

#### RSK-008: DQN Q-Value Overestimation
* **Domain:** Reinforcement Learning
* **Phase:** Phase 6
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** DQN scheduler converges to a suboptimal policy (e.g., assigning all tasks to the cloud or a single edge node).
* **Root Cause:** Standard Q-learning maximization bias across 26 discrete actions.
* **Detection Method:** Q-value tracking plot shows steady upward bias disconnected from actual episodic rewards.
* **Resolution:** Implement Double DQN (DDQN) where the online network selects the action and the target network computes the target Q-value.
* **Prevention:** Use DDQN formulation by default.
* **Human Involvement:** None.

#### RSK-009: Interface Type Inconsistencies
* **Domain:** Integration
* **Phase:** Phase 7
* **Severity:** **MEDIUM** | **Probability:** Medium
* **Symptoms:** Runtime crash `TypeError: can't convert cuda:0 device type tensor to numpy`.
* **Root Cause:** Inconsistent boundary passing between PyTorch neural network tensors and Gymnasium NumPy environment states.
* **Detection Method:** Integration tests in `tests/test_integration.py`.
* **Resolution:** Standardize all module input/output signatures to native NumPy float32 arrays; isolate PyTorch tensor conversions strictly inside model forward passes.
* **Prevention:** Use Python type hints (`np.ndarray`) and automated runtime assertions.
* **Human Involvement:** None.

#### RSK-010: High Variance Invalidating Statistical Significance
* **Domain:** Experimental Benchmarking
* **Phase:** Phase 8
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Wilcoxon signed-rank test yields $p > 0.05$ due to extreme variance between random seed iterations.
* **Root Cause:** Failure to seed all stochastic subsystems (Python `random`, `numpy.random`, `torch.manual_seed`).
* **Detection Method:** Compare standard deviations in Table 9; check if metric variance exceeds $10\%$.
* **Resolution:** Implement a unified `set_seed(seed)` function setting all pseudorandom generators before each run.
* **Prevention:** Seed every experiment run explicitly.
* **Human Involvement:** Human reviews statistical report table and p-values.


---

## Chapter 11: 7-Tier Testing Strategy & QA Hierarchy

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Tracking Status:** ACTIVE

---

### 1. Multi-Level Testing Hierarchy

Every phase of development must satisfy progressive verification levels before progressing to the subsequent phase:

```
[Level 7: Performance & Hardware Profile]  -> Latency < 25ms, Memory < 6GB
[Level 6: Regression Tests]                -> Prior functionality preserved
[Level 5: Scientific & Numerical Bounds]   -> Math formulas, p < 0.05, 0-leakage
[Level 4: End-to-End Functional Tests]     -> Complete task streaming lifecycle
[Level 3: Inter-Module Integration Tests]  -> PyTorch <-> Simulator interfaces
[Level 2: Unit Tests]                      -> Isolated function assertions
[Level 1: Static Analysis & Syntax Checks] -> Flake8, MyPy, package imports
```

---

### 2. Test Level Definitions & Execution Commands

#### Level 1: Static Analysis & Import Checks
* **Objective:** Ensure clean syntax, type hinting consistency, and module availability.
* **Commands:**
  ```bash
  python3 -m py_compile $(find src/ -name "*.py")
  pytest --collect-only tests/
  ```
* **Pass Criteria:** Zero syntax errors; zero missing imports.

#### Level 2: Component Unit Tests
* **Objective:** Verify isolated classes and functions against deterministic assertions.
* **Key Test Suites:**
  - `tests/test_nodes.py`: Verifies node capacity, MIPS ranges, and power specifications.
  - `tests/test_latency.py`: Compares computed latency against hand-calculated values.
  - `tests/test_baselines.py`: Verifies priority order for FIFO, Round-Robin, and EDF.
* **Command:** `pytest tests/test_*.py -k "unit"`
* **Pass Criteria:** 100% of unit assertions pass.

#### Level 3: Module Integration Tests
* **Objective:** Test interactions between PyTorch models, data loaders, and Gymnasium simulator.
* **Key Test Suites:**
  - `tests/test_model_sim_bridge.py`: Ensures CNN-LSTM risk outputs pipe directly into simulator task queues without type errors.
* **Command:** `pytest tests/test_integration.py`
* **Pass Criteria:** Zero tensor-to-numpy conversion errors; no unhandled exceptions.

#### Level 4: Functional Task Lifecycle Tests
* **Objective:** Execute 1,000 continuous tasks through the full pipeline.
* **Command:** `python3 main.py --mode test-run --tasks 1000`
* **Pass Criteria:** All 1,000 tasks complete execution; queues empty cleanly; metrics logged.

#### Level 5: Scientific & Numerical Validation
* **Objective:** Verify adherence to paper equations and mathematical guarantees.
* **Verification Checks:**
  1. **Zero Data Leakage:** `set(train_patients) & set(test_patients) == empty_set`.
  2. **Parameter Budgets:**
     - CNN-LSTM: $0.74\text{M} \pm 2\%$ parameters.
     - Predictor: $0.18\text{M} \pm 2\%$ parameters.
     - DQN: $0.32\text{M} \pm 2\%$ parameters.
  3. **Statistical Significance:** Wilcoxon signed-rank test confirms $p < 0.05$ against all baselines.
* **Command:** `pytest tests/test_scientific_compliance.py`
* **Pass Criteria:** All mathematical assertions hold.

#### Level 6: Regression Testing
* **Objective:** Ensure new model training or environment modifications do not break previously verified baseline schedulers or data pipelines.
* **Command:** `pytest tests/`
* **Pass Criteria:** Zero regressions in existing test suite.

#### Level 7: Performance & Resource Profiling
* **Objective:** Verify edge gateway execution speed and memory limits.
* **Checks:**
  - Forward inference of risk model: $<15$ ms.
  - Predictor inference: $<4$ ms.
  - DQN action decision: $<6$ ms.
  - End-to-end scheduling decision: $<25$ ms.
  - System memory usage: $<6$ GB.
* **Command:** `python3 -m pytest tests/test_profiling.py`
* **Pass Criteria:** Latency and memory stay strictly within verified paper bounds (Table 12).

---

### 3. Strict Debugging Protocol

When any test fails during development, the following 12-step protocol must be followed:

1. **Halt Progression:** Immediately pause progression on the affected task; do not write subsequent modules.
2. **Reproduce Failure:** Isolate the failing test and reproduce it with a minimal reproduction command.
3. **Capture Logs:** Capture the full stack trace and error message.
4. **Isolate Root Cause:** Determine whether failure is caused by data shape, numerical instability, or logic error.
5. **Formulate Hypothesis:** Document why the failure occurred.
6. **Implement Scoped Fix:** Apply the minimal required fix without rewriting unrelated working code.
7. **Rerun Failing Test:** Verify that the isolated test now passes.
8. **Rerun Unit Suite:** Verify no neighboring tests were broken.
9. **Rerun Regression Suite:** Ensure all previous levels remain green.
10. **Log to Bug Tracker:** Record error, root cause, and resolution in `BUG_LOG.md`.
11. **Update Task Board:** Mark task status accordingly in `TASK_BOARD.md`.
12. **Resume Work:** Continue to next planned task only after full test pass.


---

## Chapter 12: Empirical Benchmark Log & Baseline Comparisons

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Tracking Status:** ACTIVE

---

### Published Paper Reference Benchmarks (Nature Scientific Reports 2026)

For verification and comparison, the baseline values established in the published paper under peak load ($\lambda = 20$ tasks/sec) are recorded below:

| Experiment ID | Scheduler / Variant | Arrival Rate ($\lambda$) | Avg Latency (ms) | Deadline Viol. (%) | Risk Coverage (%) | Utilization (%) | Energy (J/1000 tasks) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PUB-REF-01** | **CRAI-LCS (Proposed)** | 20 | **148** | **2.8%** | **96.4%** | **91.7%** | **284** |
| **PUB-REF-02** | DRL-based Fog Scheduling | 20 | 195 | 7.1% | 79.8% | 82.4% | 378 |
| **PUB-REF-03** | EADRL | 20 | 182 | 5.9% | 84.2% | 86.1% | 342 |
| **PUB-REF-04** | Latency-Energy DRL | 20 | 210 | 9.4% | 76.5% | 80.3% | 405 |
| **PUB-REF-05** | AI-Native 6G DRL | 20 | 228 | 11.6% | 71.3% | 77.9% | 431 |
| **PUB-REF-06** | Static Priority Baseline | 20 | 256 | 19.3% | 54.1% | - | 465 |

---

### Live Experimental Runs Ledger

*(New experiment runs executed by our scripts will be appended below with timestamp, random seed, and raw output metrics).*

| Run ID | Timestamp | Config Profile | Random Seed | Scheduler | Task Count | Arrival Rate ($\lambda$) | Mean Latency (ms) | Violation Rate (%) | Risk Coverage (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-INIT* | 2026-11-21 | Default | 42 | Initial Test | 0 | 0 | - | - | - | Initialized |


---

## Chapter 13: Task Tracking & AI/Human Ownership Board

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Last Updated:** 2026-11-21  
**Tracking Status:** ACTIVE

---

### Task Board Summary

| Phase | Total Tasks | Completed & Verified | In Progress | Needs Human Review | Not Started |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 0: Setup & Audit** | 3 | 3 | 0 | 0 | 0 |
| **Phase 1: Simulator** | 3 | 3 | 0 | 0 | 0 |
| **Phase 2: Baselines** | 3 | 0 | 1 | 0 | 2 |
| **Phase 3: ECG Pipeline** | 3 | 0 | 0 | 0 | 3 |
| **Phase 4: Risk Model** | 3 | 0 | 0 | 0 | 3 |
| **Phase 5: Predictor** | 3 | 0 | 0 | 0 | 3 |
| **Phase 6: DQN Scheduler** | 3 | 0 | 0 | 0 | 3 |
| **Phase 7: Integration** | 2 | 0 | 0 | 0 | 2 |
| **Phase 8: Experiments** | 3 | 0 | 0 | 0 | 3 |
| **Phase 9: Ablation** | 3 | 0 | 0 | 0 | 3 |
| **Phase 10: Final Delivery** | 3 | 0 | 0 | 0 | 3 |
| **TOTAL** | **32** | **6** | **1** | **0** | **25** |

---

### Detailed Task Register

#### Phase 0: Setup, Repository Audit & Research Verification
* **T0-01:** Comprehensive GitHub repository audit of 6 candidate projects and dependency verification.
  - *Phase:* Phase 0
  - *Assigned To:* AI Autonomous
  - *Dependencies:* None
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* All 6 repos queried via API, licenses checked, `REUSE_AUDIT.md` written.
* **T0-02:** Author master PRD, ADR, Paper Compliance matrix, Risk Register, and Testing Strategy.
  - *Phase:* Phase 0
  - *Assigned To:* AI + Human Review
  - *Dependencies:* T0-01
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* `PRD.md`, `ARCHITECTURE_DECISION.md`, `PAPER_COMPLIANCE.md`, `RISK_REGISTER.md`, `TESTING_STRATEGY.md` created.
* **T0-03:** Initialize project folder structure (`src/`, `configs/`, `tests/`), install domain packages (`wfdb`, `gymnasium`).
  - *Phase:* Phase 0
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T0-02
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* Import test passes (`torch`, `gymnasium`, `wfdb`), `set_global_seed(42)` verified.

---

#### Phase 1: Edge-Cloud Simulation Engine
* **T1-01:** Implement `HealthcareTask`, `EdgeNode`, and `CloudInstance` classes.
  - *Phase:* Phase 1
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T0-03
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* Unit tests verify node parameter ranges (800-4500 MIPS edge, 12k-35k MIPS cloud) in `tests/test_nodes.py`.
* **T1-02:** Implement latency and base/active energy calculation mathematical engine (Eq. 8-10).
  - *Phase:* Phase 1
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T1-01
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* Mathematical assertion unit tests pass against hand-calculated values in `tests/test_latency.py`.
* **T1-03:** Implement `HealthcareEdgeEnv` (Gymnasium-compatible discrete-event environment).
  - *Phase:* Phase 1
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T1-02
  - *Status:* **COMPLETED & VERIFIED**
  - *Verification:* Environment resets, steps across all 26 actions, 1,000 tasks run in `tests/test_simulator.py`.

---

#### Phase 2: Baseline Schedulers
* **T2-01:** Implement stateless baselines: `RandomScheduler`, `RoundRobinScheduler`, `FIFOScheduler`.
  - *Phase:* Phase 2
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T1-03
  - *Status:* **NOT STARTED**
  - *Verification:* Deterministic order assertions pass.
* **T2-02:** Implement priority baselines: `EarliestDeadlineFirstScheduler` (EDF) and `RiskPriorityScheduler`.
  - *Phase:* Phase 2
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T1-03
  - *Status:* **NOT STARTED**
  - *Verification:* Urgent tasks dispatched with lowest queue latency.
* **T2-03:** Benchmark all 5 baselines on a synthetic 1,000-task workload stream.
  - *Phase:* Phase 2
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T2-01, T2-02
  - *Status:* **NOT STARTED**
  - *Verification:* Execution generates baseline metric comparisons.

---

#### Phase 3: ECG Dataset Pipeline
* **T3-01:** Download 44 PhysioNet MIT-BIH recordings using `wfdb`.
  - *Phase:* Phase 3
  - *Assigned To:* AI + Human (Human monitors download bandwidth)
  - *Dependencies:* T0-03
  - *Status:* **NOT STARTED**
  - *Verification:* 44 `.dat` and `.atr` files verified in `data/raw/mitbih/`.
* **T3-02:** Implement signal filtering, Z-score normalization, and 5-second windowing ($T=1800$ @ 360 Hz).
  - *Phase:* Phase 3
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T3-01
  - *Status:* **NOT STARTED**
  - *Verification:* Tensor shapes `(1800,)`, mean $\approx 0$, std $\approx 1$.
* **T3-03:** Map arrhythmia labels ($y_i \in \{0, 1\}$) and split strictly by patient record (70/15/15).
  - *Phase:* Phase 3
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T3-02
  - *Status:* **NOT STARTED**
  - *Verification:* Set intersection between train, val, and test record sets is empty.

---

#### Phase 4: CNN-LSTM Clinical Risk Model ($R_i$)
* **T4-01:** Build `CNNLSTMRisk` PyTorch architecture ($0.74$M parameters).
  - *Phase:* Phase 4
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T3-03
  - *Status:* **NOT STARTED**
  - *Verification:* Parameter count within $\pm 2\%$ of $0.74$M; forward pass takes $<15$ ms.
* **T4-02:** Implement training pipeline with Adam ($\text{lr}=10^{-3}$, batch=32) and BCE loss.
  - *Phase:* Phase 4
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T4-01
  - *Status:* **NOT STARTED**
  - *Verification:* Training loss decreases; early stopping triggers on validation plateau.
* **T4-03:** Evaluate model on test patient records; compute Accuracy, F1, and ROC-AUC.
  - *Phase:* Phase 4
  - *Assigned To:* AI + Human Review
  - *Dependencies:* T4-02
  - *Status:* **NOT STARTED**
  - *Verification:* Test Accuracy $\ge 93.4\%$, AUC $\ge 0.95$ (Table 6).

---

#### Phase 5: Deadline Violation Predictor ($P_{miss,i}$)
* **T5-01:** Harvest 50,000 state-outcome tuples $(S_i, y_{miss,i})$ from exploratory simulation runs.
  - *Phase:* Phase 5
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T1-03, T4-03
  - *Status:* **NOT STARTED**
  - *Verification:* Dataset saved in `data/processed/predictor_training_data.npz`.
* **T5-02:** Build 3-layer MLP classifier ($0.18$M parameters) taking $S_i = \{U_t, Q_t, \psi_i, D_i, R_i\}$.
  - *Phase:* Phase 5
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T5-01
  - *Status:* **NOT STARTED**
  - *Verification:* Parameter count $\approx 0.18$M, inference latency $<4$ ms.
* **T5-03:** Train and calibrate predictor with BCE loss; integrate into simulator state vector.
  - *Phase:* Phase 5
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T5-02
  - *Status:* **NOT STARTED**
  - *Verification:* Validation AUC $\ge 0.90$.

---

#### Phase 6: DQN Scheduler Implementation
* **T6-01:** Build `DQNNetwork` ($0.32$M parameters) and `PrioritizedReplayBuffer` (capacity 50,000).
  - *Phase:* Phase 6
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T5-03
  - *Status:* **NOT STARTED**
  - *Verification:* State dim matches observation vector; action dim = 26.
* **T6-02:** Implement multi-objective reward shaping (Equation 29) with normalized latency.
  - *Phase:* Phase 6
  - *Assigned To:* AI + Human Review
  - *Dependencies:* T6-01
  - *Status:* **NOT STARTED**
  - *Verification:* Normalized reward bounds tested against edge cases.
* **T6-03:** Train DQN over 50,000 steps with $\epsilon$-greedy decay and target network updates.
  - *Phase:* Phase 6
  - *Assigned To:* AI + Human Review (Human reviews training stability)
  - *Dependencies:* T6-02
  - *Status:* **NOT STARTED**
  - *Verification:* Convergence of TD loss and rising mean reward.

---

#### Phase 7: System Integration
* **T7-01:** Build unified `CRAILCSPipeline` linking ECG $\to$ Risk Model $\to$ Tasks $\to$ Predictor $\to$ DQN $\to$ Simulation.
  - *Phase:* Phase 7
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T4-03, T5-03, T6-03
  - *Status:* **NOT STARTED**
  - *Verification:* Full task lifecycle completes without type errors.
* **T7-02:** Run 100-patient streaming simulation integration test.
  - *Phase:* Phase 7
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T7-01
  - *Status:* **NOT STARTED**
  - *Verification:* All metrics accurately tracked without resource leak.

---

#### Phase 8: Experimental Evaluation & Benchmarking
* **T8-01:** Multi-load workload sweeps across arrival rates $\lambda \in [8, 25]$ tasks/sec.
  - *Phase:* Phase 8
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T7-02
  - *Status:* **NOT STARTED**
  - *Verification:* Metric CSV logs generated for all arrival rates.
* **T8-02:** Execute 10 independent random seeds; compute Mean $\pm$ Std and Wilcoxon signed-rank test.
  - *Phase:* Phase 8
  - *Assigned To:* AI + Human Review
  - *Dependencies:* T8-01
  - *Status:* **NOT STARTED**
  - *Verification:* Wilcoxon test yields $p < 0.05$ across main metrics (Table 9).
* **T8-03:** Generate publication-quality figures (Figures 3, 4, 5, 6, 7).
  - *Phase:* Phase 8
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T8-02
  - *Status:* **NOT STARTED**
  - *Verification:* High-resolution plots saved in `results/graphs/`.

---

#### Phase 9: Ablation & Reliability Stress Testing
* **T9-01:** Implement and evaluate 4 ablation variants (w/o risk, w/o predictor, w/o reward, static).
  - *Phase:* Phase 9
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T8-02
  - *Status:* **NOT STARTED**
  - *Verification:* Results demonstrate performance drop-offs matching Table 8.
* **T9-02:** Execute reward weight sensitivity analysis across sets RW-1, RW-2, RW-3, RW-4 (Table 10).
  - *Phase:* Phase 9
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T9-01
  - *Status:* **NOT STARTED**
  - *Verification:* RW-3 yields highest risk coverage; RW-1 yields lowest latency.
* **T9-03:** Stress test under simulated edge node failures and network jitter.
  - *Phase:* Phase 9
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T9-01
  - *Status:* **NOT STARTED**
  - *Verification:* System graceful degradation documented.

---

#### Phase 10: Final Documentation & Demonstration CLI
* **T10-01:** Build unified `main.py` CLI supporting `--mode demo`, `--mode train`, `--mode benchmark`.
  - *Phase:* Phase 10
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T9-03
  - *Status:* **NOT STARTED**
  - *Verification:* All CLI modes execute out of the box.
* **T10-02:** Complete technical user guide and architecture diagrams.
  - *Phase:* Phase 10
  - *Assigned To:* AI + Human Review
  - *Dependencies:* T10-01
  - *Status:* **NOT STARTED**
  - *Verification:* Clear documentation verified by human review.
* **T10-03:** Final clean environment installation and regression testing.
  - *Phase:* Phase 10
  - *Assigned To:* AI Autonomous
  - *Dependencies:* T10-02
  - *Status:* **NOT STARTED**
  - *Verification:* Zero broken imports, clean pass on `pytest tests/`.


---

## Chapter 14: Turnkey Reproduction CLI Commands

Aap directly terminal mein ye commands run karke project ko end-to-end execute, train aur verify kar sakte hain:

```bash
# ==========================================
# 1. Environment & Dependencies Setup
# ==========================================
git clone <repo_url> && cd CRAI_LCS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v

# ==========================================
# 2. MIT-BIH Ingestion & Zero-Leakage 5s Windowing (@ 360 Hz)
# ==========================================
python3 download_mitbih.py
python3 src/ecg/preprocessing.py --seed 42

# ==========================================
# 3. Model Training & Sequential Bootstrapping
# ==========================================
# Train 1D-CNN-LSTM Risk Estimator
python3 -m src.training.train_risk_model --output checkpoints/cnn_lstm_risk.pt

# Collect 50,000 steps warm-up violation transitions
python3 -m src.simulator.bootstrap_predictor --steps 50000 --output data/processed/predictor.npz

# Train Pre-Execution MLP Violation Predictor
python3 -m src.training.train_predictor --input data/processed/predictor.npz

# Train Double DQN Agent with RW-2 Balanced Reward Profile
python3 -m src.training.train_dqn --episodes 1000 --reward_profile RW-2

# ==========================================
# 4. Comparative Benchmarks & Statistical Significance Testing
# ==========================================
# Peak load (lambda = 20 tasks/sec) stress test across 10 random seeds
python3 evaluation.py --arrival_rate 20 --seeds 10 --output results/benchmarks_lambda20.csv

# Non-parametric Wilcoxon Signed-Rank Test & p-value calculation
python3 -m src.experiments.statistical_tests --input results/benchmarks_lambda20.csv

# Zero-shot cross-dataset evaluation on PTB-XL
python3 -m src.experiments.generalization_test --dataset ptb-xl
```