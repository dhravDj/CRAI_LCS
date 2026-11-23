# CRAI-LCS: Reverse-Engineered Architecture & Resource Specification
**Framework:** Clinical Risk-Aware AI-Enhanced Latency-Constrained Scheduling for Healthcare IoT  
**Reference Paper:** *Scientific Reports (Nature Portfolio)*, 2026. DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9)  
**Official Codebase Archive:** Zenodo DOI [10.5281/zenodo.22036231](https://doi.org/10.5281/zenodo.22036231)

---

## 1. Executive Summary & REA Architecture Decomposition

The **CRAI-LCS** system breaks away from traditional system-centric edge schedulers (which treat all computational requests homogeneously) by integrating **real physiological data**, **clinical risk assessment**, and **deadline violation prediction** directly into a **Deep Reinforcement Learning (DRL)** edge-cloud scheduling loop.

### Architectural Tiers:
```
+-----------------------------------------------------------------------------------+
| 1. IoT Sensing Layer                                                              |
|    - 200 Medical IoT & Bedside Sensors                                            |
|    - 5-second ECG windows @ 360 Hz (1800 samples/task, ~3.6 KB input)             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. Edge Computing Layer (25 Distributed Edge Nodes)                              |
|    - Signal Normalization & Feature Extraction                                    |
|    - CNN-LSTM Clinical Risk Estimator (Inference: 14.8 ms, 0.74M params) -> Risk R_i|
|    - Local Task Queue Management & Priority Labeling (Critical, Important, Normal)|
|    - Latency Evaluation & Immediate Dispatch for Life-Threatening Events          |
+-----------------------------------------+-----------------------------------------+
                                          |
                        Offloaded Tasks   v   Feedback (Latency, Rewards, Errors)
+-----------------------------------------------------------------------------------+
| 3. AI-Enhanced Cloud Layer (2 High-Performance Cloud Instances)                   |
|    - Multi-Priority Task Queues                                                   |
|    - Deadline Violation Predictor (Inference: 3.6 ms, 0.18M params) -> P_miss,i   |
|    - Deep Q-Network (DQN) Scheduler (Decision: 5.7 ms, 0.32M params) -> Action a_t |
|    - Action Space: M=25 Edge Nodes + 1 Cloud Node (26 discrete actions)           |
|    - Reward-Shaping Mechanism (Balance Latency, Violations, Risk, Energy)         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Reconstruction Obligation Ledger (ROL)

To construct this project from scratch or enhance the downloaded Zenodo boilerplate into a full-scale pipeline, the following concrete obligations must be satisfied:

| Module / Component | Obligation & Specification | Evidence Source |
| :--- | :--- | :--- |
| **Data Ingestion** | Fetch MIT-BIH Arrhythmia & PTB-XL ECG databases from PhysioNet. Segment at $T=1800$ samples (5 sec at 360 Hz). | Paper Sec. "Problem formulation", Table 3 |
| **Signal Preprocessing** | Z-score normalization: $\tilde{x}_i = (x_i - \mu_i)/\sigma_i$. Record-wise patient split (70% train, 15% val, 15% test). | Eq. 17, Sec. "Clinical risk estimation model" |
| **Risk Estimator (CNN-LSTM)** | 1D Conv + ReLU + MaxPool $\to$ LSTM $\to$ Dense $\to$ Sigmoid output $R_i \in [0, 1]$. Budget: 0.74M params, <15 ms inference. | Eq. 18-26, Table 6, Table 12 |
| **Deadline Predictor** | Input state $S_i$ (system load, queue state, clinical risk) $\to P_{miss,i} \in [0, 1]$. Budget: 0.18M params, <4 ms inference. | Eq. 12, Table 12 |
| **Simulation Environment** | 200 IoT sensors, 25 edge nodes (800-4500 MIPS, 16-64 GB RAM, 40-80 W), 2 cloud instances (12000-35000 MIPS, 128 GB RAM, 180-350 W). | Table 3, Table 4 |
| **Latency & Energy Equations** | $L_i^{edge} = T_{tr}^{(i,j)} + \frac{\delta_i}{B_j} + \frac{\psi_i}{f_j}$, $L_i^{cloud} = T_{tr}^{(i,j)} + T_{e \to c} + \frac{\delta_i}{B_c} + \frac{\psi_i}{f_c}$. | Eq. 8-10, Table 3 |
| **DQN Scheduling Agent** | State: $s_t = \{Q_t, U_t, R_i, P_{miss,i}, D_i\}$. Action: $a_t \in \{E_1, \dots, E_{25}, C\}$. Budget: 0.32M params. | Eq. 27-32, Table 12 |
| **Reward Function** | $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ with default weights $\alpha=0.35, \beta=0.25, \gamma=0.30, \eta=0.10$. | Eq. 29, Table 10 |
| **Baselines & Benchmarking** | Implement DRL-Fog, EADRL, Latency-Energy DRL, AI-Native 6G DRL, and Static Priority rule-based scheduler. | Table 5, Table 8 |
| **Statistical Analysis** | 10 independent random seeds, Wilcoxon signed-rank test ($\alpha=0.05, p<0.05$). | Table 9, Sec. "Statistical significance analysis" |

---

## 3. Complete Resource Inventory Required

### A. Data Resources
1. **PhysioNet MIT-BIH Arrhythmia Database**:
   - Access URL: `https://physionet.org/content/mitdb/1.0.0/`
   - Records: 48 half-hour two-channel ambulatory ECG recordings (e.g., records 100 through 234).
   - Annotations: Beat and rhythm annotations (Normal 'N', Ventricular Arrhythmias 'V', Supraventricular 'A', etc.).
   - Tooling: Python `wfdb` library (`wfdb.rdrecord`, `wfdb.rdann`, `wfdb.dl_database`).
2. **PhysioNet PTB-XL ECG Database (For Generalization Testing)**:
   - Access URL: `https://physionet.org/content/ptb-xl/`
   - 21,837 clinical 12-lead ECG records from 18,885 patients (sampled at 100 Hz or 500 Hz).
3. **Task Attributes Generation Rules**:
   - Segment length: 5 seconds @ 360 Hz = 1800 samples.
   - Input size $\delta_i \approx 3.6$ KB (1800 samples $\times$ 2 bytes).
   - Computational cycles $\psi_i \in [5 \times 10^6, 2 \times 10^7]$ CPU cycles.
   - Urgency-aware Deadlines $D_i$:
     - High-Risk Tasks (Arrhythmia / abnormal $R_i \ge 0.5$): $100 - 150$ ms.
     - Normal Monitoring Tasks ($R_i < 0.5$): $400 - 800$ ms.

---

### B. Machine Learning & Model Architectures

#### 1. Clinical Risk Estimator (CNN-LSTM)
- **Input shape:** `(Batch, 1, 1800)`
- **Feature Extractor:**
  - `Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)` + `ReLU()`
  - `MaxPool1d(kernel_size=2, stride=2)` -> shape `(Batch, 32, 900)`
  - Optional second Conv1D block to achieve 0.74M parameters.
- **Temporal Sequence Modeling:**
  - `LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)`
- **Risk Score Head:**
  - `Linear(64, 32)` + `ReLU()` + `Dropout(0.2)`
  - `Linear(32, 1)` + `Sigmoid()` -> $R_i \in [0, 1]$
- **Optimization:** Adam, learning rate = $1 \times 10^{-3}$, batch size = 32, Binary Cross-Entropy Loss.

#### 2. Deadline Violation Predictor ($f_\theta(S_i)$)
- **Input vector:** System load ($U_t$), total queue delay ($\sum Q_t$), task computational cycles ($\psi_i$), task deadline ($D_i$), clinical risk ($R_i$).
- **Architecture:** 3-layer MLP (`Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1) -> Sigmoid`). Total parameters $\approx 0.18$M.
- **Output:** Predicted probability of deadline violation $P_{miss,i} \in [0, 1]$.

#### 3. Deep Q-Network (DQN) Scheduler
- **State Space (Dimension = $25 + 25 + 3 = 53$ or compact 8-dim):**
  - Edge queue lengths: $Q_t \in \mathbb{R}^M$ ($M=25$)
  - Edge node utilization: $U_t \in \mathbb{R}^M$ ($M=25$)
  - Incoming task features: $\{R_i, P_{miss,i}, D_i\}$
- **Action Space (Dimension = $25 + 1 = 26$):**
  - Actions 0 to 24: Assign to Edge Node $E_1 \dots E_{25}$.
  - Action 25: Offload to Cloud Server $C$.
- **Network Architecture:**
  - Fully Connected layers: `Linear(State_Dim, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, Action_Dim)`
  - Parameter count: $\approx 0.32$M parameters.
- **Training Strategy:**
  - Replay memory: capacity 50,000 transitions.
  - Target network update frequency: every 100 steps.
  - Discount factor: $\lambda = 0.95$ to $0.99$.
  - Exploration rate: $\epsilon$-greedy decaying from 1.0 to 0.05.

---

### C. Simulation Environment Parameters

| Parameter Category | Parameter Name | Value in Paper |
| :--- | :--- | :--- |
| **Topology** | IoT Devices | 200 devices |
| | Edge Nodes ($M$) | 25 nodes |
| | Cloud Instances | 2 high-performance instances |
| **Workload** | Arrival Rate ($\lambda$) | 8 to 25 tasks/sec (Peak test at $\lambda=20$) |
| | Input Data Size ($\delta_i$) | ~3.6 KB |
| | Task CPU Demand ($\psi_i$) | $5 \times 10^6$ to $2 \times 10^7$ cycles |
| **Deadlines** | Critical/High-Risk | 100 – 150 ms |
| | Normal Monitoring | up to 800 ms |
| **Edge Hardware** | CPU Capacity ($f_j$) | 800 – 4500 MIPS |
| | RAM Capacity ($m_j$) | 16 – 64 GB |
| | Idle/Base Power | 40 – 80 W |
| **Cloud Hardware**| CPU Capacity ($f_c$) | 12,000 – 35,000 MIPS |
| | RAM Capacity ($m_c$) | 128 GB |
| | Idle/Base Power | 180 – 350 W |
| **Network** | Device-to-Edge Latency ($T_{tr}^{(i,j)}$) | 3 – 12 ms |
| | Edge-to-Cloud Latency ($T_{e \to c}$) | 45 – 120 ms |
| | Edge Bandwidth ($B_j$) | 10 – 50 Mbps |
| | Cloud Bandwidth ($B_c$) | 100 – 1000 Mbps |

---

### D. Software Stack & Dependencies

| Tool / Library | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Core development runtime |
| **PyTorch (`torch`)** | >= 2.0.0 | CNN-LSTM, Predictor, DQN implementation |
| **WFDB (`wfdb`)** | >= 4.1.0 | PhysioNet ECG downloading and annotation decoding |
| **NumPy (`numpy`)** | >= 1.23.0 | Array manipulation, signal math, state buffering |
| **Pandas (`pandas`)** | >= 2.0.0 | Tabular metrics, evaluation reporting |
| **Scikit-learn (`scikit-learn`)**| >= 1.2.0 | ROC-AUC, classification metrics, data splitting |
| **Gymnasium (`gymnasium`)** | >= 0.28.0 | Standard RL environment wrapper interface |
| **SciPy (`scipy`)** | >= 1.10.0 | Signal processing (filtering) & Wilcoxon signed-rank test |
| **Matplotlib / Seaborn** | >= 3.7.0 | Plotting latency vs arrival rate, ablation charts |

---

### E. Compute & Hardware Budget

- **Model Footprint:**
  - CNN-LSTM Risk Estimator: 0.74 M parameters, 14.8 ms inference, 16.7 MB memory.
  - Deadline Predictor: 0.18 M parameters, 3.6 ms inference, 4.9 MB memory.
  - DQN Scheduler: 0.32 M parameters, 5.7 ms decision time, 6.8 MB memory.
  - **Complete Integrated System:** 1.24 M parameters, 24.1 ms end-to-end decision latency, 28.4 MB memory usage.
- **Training Compute:** Standard workstation with a multi-core CPU (or 1 modest GPU like RTX 3060/4060 or CPU inference). Real-time simulation runs efficiently in Python without heavy multi-GPU infrastructure.
- **Edge Deployment Target:** Can comfortably execute on lightweight edge gateways (e.g., Raspberry Pi 4/5, NVIDIA Jetson Nano/Orin, or edge server blades).

---

## 4. Evaluation Targets & Benchmarks

To replicate or exceed the results published in *Scientific Reports (2026)*:

| Evaluation Metric | Target CRAI-LCS Performance (Peak Load $\lambda=20$) | Best Baseline Performance |
| :--- | :--- | :--- |
| **Avg End-to-End Latency** | **148 – 149.6 ms** | 182 – 183.1 ms (EADRL) |
| **Deadline Violation Rate** | **2.8 – 2.94%** | 5.9 – 6.08% (EADRL) |
| **Clinical Risk Coverage** | **95.87 – 96.4%** | 84.2 – 84.62% (EADRL) |
| **Edge Resource Utilization** | **91.24 – 91.7%** | 85.93 – 86.1% (EADRL) |
| **Energy Consumption** | **284 – 287.5 J / 1000 tasks** | 342 – 345.8 J (EADRL) |
| **Worst-Case Scalability** | **Up to 48–57% faster execution** under stress | Degrades super-linearly |
