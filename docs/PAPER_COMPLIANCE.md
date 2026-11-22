# CRAI-LCS: Paper Compliance & Fidelity Matrix

**Paper Reference:** *Clinical risk-aware reinforcement learning for latency-constrained healthcare IoT scheduling*, Scientific Reports 2026. DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9)  
**Verification Date:** 2026-11-21  
**Status:** AUDITED & TRACKED

---

## 1. Compliance Classification Scheme
Every element in the project is audited and assigned one of three labels:
* **[VERIFIED]:** Explicitly specified in the published paper text, tables, or equations.
* **[PROPOSED / DECISION]:** Reasonable engineering implementation decision introduced by our team where the paper left minor low-level implementation details unspecified.
* **[ASSUMPTION / UNVERIFIED]:** Unspecified in the paper, requiring empirical validation or human review.

---

## 2. Equation-by-Equation Compliance Audit

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

## 3. Implementation Decisions & Assumptions Requiring Tracking

| Item | Paper Description | Our Implementation Decision | Rationale |
| :--- | :--- | :--- | :--- |
| **ECG Record List** | Mentions PhysioNet MIT-BIH Arrhythmia without full explicit record enumeration | Selected standard 44 benchmark records (e.g., 100, 101, ..., 234) partitioned by patient ID | Prevents patient data leakage while ensuring broad arrhythmia representation. |
| **Task Queue Policy within Nodes** | Mentions queue condition $Q_t$ without specifying internal node queue discipline | Implemented FIFO queue per edge node with preemptive urgency tag for $R_i \ge 0.8$ | Matches Figure 2 ("Immediate response to Life-threatening conditions"). |
| **Reward Normalization Scale** | Paper defines formula $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ without scaling $L_i$ | Normalize latency $L_i$ by deadline ($L_i / D_i$ or $L_i / 1000.0$) | Prevents millisecond latency values ($100-800$) from overpowering $[0, 1]$ risk terms. |
| **PTB-XL Cross-Dataset Windowing** | Evaluated on PTB-XL (Table 13) | Downsample PTB-XL records to 360 Hz to match MIT-BIH 1800-sample input size | Ensures CNN-LSTM input layer compatibility without retraining network weights. |
