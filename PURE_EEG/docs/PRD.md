# Product Requirements Document (PRD) & Engineering Specification

## Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling (CRAI-LCS)

**Document Version:** 1.0.0  
**Status:** APPROVED & BASELINED  
**Date:** 2026-11-21  
**Target Publication Reference:** *Scientific Reports (Nature Portfolio)*, 2026. DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9)

---

## 1. Project Overview & Context

In the modern Healthcare Internet of Things (HIoT), wearable devices and bedside monitors continuously generate vital physiological signals, such as electrocardiograms (ECG). These time-critical signals are vital for the early detection of life-threatening events, including acute arrhythmias and hypoxia.

Conventional cloud-centric architectures introduce substantial round-trip communication delays ($45 - 120$ ms backhaul), which can compromise clinical outcomes during cardiac emergencies. Mobile Edge Computing (MEC) mitigates this latency by deploying distributed compute gateways closer to patients. However, existing edge schedulers remain almost exclusively **system-centric**—optimizing infrastructure metrics like CPU utilization or generic queue length—while treating all computational tasks uniformly. 

**CRAI-LCS** resolves this fundamental flaw by introducing a **closed-loop clinical-to-computational decision paradigm**. It embeds patient-specific physiological risk ($R_i$) directly into the Markov Decision Process (MDP) of a Deep Reinforcement Learning (DRL) scheduler, coordinating with a pre-execution deadline violation predictor ($P_{miss,i}$) across a 3-tier IoT-Edge-Cloud hierarchy.

---

## 2. Problem Statement

1. **Failure to Distinguish Criticality:** In conventional fog/edge schedulers, a routine monitoring packet and a lethal ventricular fibrillation packet are queued identically if their file sizes and cycle counts match.
2. **Stringent Healthcare Latencies:** High-risk cardiac events require immediate intervention (deadlines $100 - 150$ ms), whereas transmission to distant cloud datacenters routinely breaches these constraints.
3. **Edge Resource Scarcity:** Edge gateways possess constrained computational and memory capacity ($800 - 4500$ MIPS). Uncontrolled local queuing triggers buffer overflows, while excessive cloud offloading increases backhaul latency and energy waste.
4. **Stochastic Clinical Surges:** Real-world patient physiological emergencies occur unpredictably, requiring dynamic, real-time task re-prioritization rather than rigid static heuristics.

---

## 3. Measurable Objectives

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

## 4. Scope

### In Scope
- Full data ingestion, segmentation, and patient-wise partitioning of PhysioNet MIT-BIH and PTB-XL datasets.
- PyTorch implementation and training of the CNN-LSTM risk model and MLP deadline predictor.
- Gymnasium-compatible discrete-event simulation of the 25-node edge cluster and 2-instance cloud infrastructure.
- Double DQN scheduler training with experience replay and multi-objective reward shaping.
- Implementation of 5 baseline schedulers: Random, Round-Robin, FIFO, Earliest Deadline First (EDF), and Risk Priority.
- Ablation experiments, reward sensitivity analysis, and statistical tests.
- Comprehensive technical documentation, architecture diagrams, and reproduction CLI.

### Out of Scope
- Direct deployment onto physical hospital clinical networks.
- Patient diagnosis or clinical decision support intended for real-time human treatment without physician oversight.
- Production FDA/CE medical-device regulatory submissions.

---

## 5. Stakeholder & Responsibility Matrix

| Entity | Role & Ownership | Key Responsibilities |
| :--- | :--- | :--- |
| **Human Developer / Researcher** | Project Lead & Reviewer | Approves architecture decisions, reviews research assumptions, validates final scientific metrics, manages external credentials/downloads. |
| **AI Agent (Antigravity)** | Lead Research & Implementation Engineer | Autonomously authors modular Python code, configures simulation, trains models, runs tests, creates documentation, profiles performance. |
| **PhysioNet / MIT-BIH** | Data Provider | Provides raw physiological cardiac recordings under open data use policies. |
| **CRAI-LCS Framework** | Software System | Ingests ECG, estimates risk, predicts violations, schedules tasks, and logs performance. |

---

## 6. Functional Requirements (FR)

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

## 7. Non-Functional Requirements (NFR)

* **NFR-1 (Modularity):** All modules must be decoupled behind abstract interfaces (`src/ecg/`, `src/models/`, `src/simulator/`, `src/schedulers/`).
* **NFR-2 (Reproducibility):** Setting a global random seed (`--seed 42`) must guarantee identical data splits, task arrival sequences, and initial network weights.
* **NFR-3 (Execution Efficiency):** Real-time inference budget for the combined models must not exceed $25$ ms on standard CPU hardware (CNN-LSTM $\le 15$ ms, Predictor $\le 4$ ms, DQN $\le 6$ ms).
* **NFR-4 (Memory Footprint):** Peak system memory during simulation and training must not exceed $6$ GB RAM.
* **NFR-5 (Test Coverage):** Every functional module must have a corresponding test in `tests/` with strict pass/fail assertions.
* **NFR-6 (Code Quality):** Code must be strictly typed, well-commented, and comply with PEP 8 standards.

---

## 8. System Architecture & Detailed Data Flow

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

## 9. Data Requirements & Clinical Risk Mapping

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
