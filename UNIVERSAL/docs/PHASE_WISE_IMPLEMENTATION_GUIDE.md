# CRAI-LCS: End-to-End Phase-wise Implementation Guide (Hinglish)

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Provenance:** Nature Portfolio *Scientific Reports* 2026 (DOI: [10.1038/s41598-026-68894-9](https://doi.org/10.1038/s41598-026-68894-9))  
**Document Purpose:** Scratch se production tak har phase ko start-to-end kaise build aur verify karna hai uska complete practical blueprint.

---

## Complete Phase Roadmap

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

## Table of Contents
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

## Phase 0: Environment Setup, Global Determinism & Repository Audit

### 1. Goal (Kya achieve karna hai)
Ek isolated development environment banana, dependencies configure karna, aur bitwise reproducibility lock karna taaki har run pe identical ECG batches aur Poisson arrival sequences generate hon.

### 2. Prerequisites & Inputs
* Linux OS, Python 3.10+, Git, workspace access.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `requirements.txt`
* `src/utils/seed.py`
* `configs/config.yaml`

### 5. Verification & Exit Criteria
* `python3 -c "import torch, gymnasium, wfdb; from src.utils.seed import set_global_seed; set_global_seed(42); print('Phase 0 Passed')"` smoothly run ho aur exit code 0 de.

---

## Phase 1: Discrete-Event Continuum Simulator (Hardware & Latency Engine)

### 1. Goal (Kya achieve karna hai)
Hospital ke physical 3-tier computing hierarchy (200 monitors, 25 edge gateways, 2 cloud servers) aur mathematical latency/energy models ko ek Farama Gymnasium environment mein simulate karna.

### 2. Prerequisites & Inputs
* Phase 0 completed.
* Paper Equations (8)--(10) and Hardware Parameter Table.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/simulator/task.py`
* `src/simulator/nodes.py`
* `src/simulator/latency_energy.py`
* `src/simulator/edge_cloud_env.py`

### 5. Verification & Exit Criteria
* Synthetic 1,000 tasks run karne par queues stable rahein aur hand-calculated latency values simulation output se $100\%$ match karein.

---

## Phase 2: Five Heuristic Baseline Schedulers

### 1. Goal (Kya achieve karna hai)
Benchmarking ke liye 5 standard heuristic schedulers ka unified `schedule(observation) -> int` interface banana.

### 2. Prerequisites & Inputs
* Phase 1 Gymnasium simulator ready.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/baselines/heuristics.py`
* `tests/test_baselines.py`

### 5. Verification & Exit Criteria
* Schedulers bina error execute hon aur expected hierarchy verify ho: EDF aur Risk-Priority schedulers Random/FIFO se significantly behtar violation rate show karein.

---

## Phase 3: PhysioNet MIT-BIH Ingestion & Zero-Leakage Pipeline

### 1. Goal (Kya achieve karna hai)
PhysioNet MIT-BIH arrhythmia dataset se continuous raw ECG waveforms download karna, Z-score standardizing karna, 5-second windows banana, aur patient leakage rokna.

### 2. Prerequisites & Inputs
* Python package `wfdb`.
* PhysioNet open-access repository (`mitdb/1.0.0`).

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `download_mitbih.py`
* `src/ecg/preprocessing.py`
* `data/processed/ecg_tasks_split.npz`

### 5. Verification & Exit Criteria
* Hard assertion check pass kare: `assert len(set(train_ids) & set(test_ids)) == 0`.
* Segments array shape: `(N, 1800)` with mean $\approx 0$ and std $\approx 1$.

---

## Phase 4: 1D-CNN-LSTM Clinical Risk Network Training ($R_i$)

### 1. Goal (Kya achieve karna hai)
Raw 1800-sample ECG segment ko analyze karke patient ka continuous Clinical Risk Score ($R_i \in [0, 1]$) predict karne wala deep neural network train karna.

### 2. Prerequisites & Inputs
* Phase 3 ka preprocessed MIT-BIH dataset (`ecg_tasks_split.npz`).
* Parameter budget limit: $\le 0.74\text{M}$ weights; Execution budget: $\le 14.8\text{ ms}$.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/models/cnn_lstm_risk.py`
* `src/training/train_risk_model.py`
* `checkpoints/cnn_lstm_risk.pt`

### 5. Verification & Exit Criteria
* Unseen test set par: **Accuracy $\ge 94.2\%$**, **AUC-ROC $\ge 0.962$**.
* PyTorch parameter check: `sum(p.numel() for p in model.parameters()) == 741,409` ($\approx 0.74\text{M}$).
* Single-window CPU forward pass latency $\le 14.8\text{ ms}$.

---

## Phase 5: Pre-Execution Deadline Predictor & 3-Stage Bootstrapping

### 1. Goal (Kya achieve karna hai)
Task assign hone se pehle hi forecast karna ki kya wo current system load ke chalte deadline miss karega ($P_{miss,i} \in [0, 1]$).

### 2. Prerequisites & Inputs
* Phase 1 simulator aur Phase 4 risk model.
* Parameter budget: $\le 0.18\text{M}$; Inference budget: $\le 3.6\text{ ms}$.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/models/deadline_predictor.py`
* `src/simulator/bootstrap_predictor.py`
* `src/training/train_predictor.py`
* `checkpoints/deadline_predictor.pt`

### 5. Verification & Exit Criteria
* Predictor test set par **AUC-ROC $\ge 0.924$** achieve kare.
* Total weights $\approx 0.18\text{M}$ aur CPU inference time $\le 3.6\text{ ms}$.

---

## Phase 6: Double DQN Scheduler & Multi-Objective Reward Shaping

### 1. Goal (Kya achieve karna hai)
54-dimensional state space ko analyze karke 26 discrete actions mein se optimal dispatch decision lene wala Double Deep Q-Network train karna.

### 2. Prerequisites & Inputs
* Pre-trained `cnn_lstm_risk.pt` aur `deadline_predictor.pt`.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/models/dqn_scheduler.py`
* `src/training/train_dqn.py`
* `checkpoints/dqn_scheduler.pt`

### 5. Verification & Exit Criteria
* TD loss smoothly converge kare ($<0.01$).
* Average episode reward negative se positive territory mein enter kare.
* Total DQN weights $\le 0.32\text{M}$ aur CPU forward pass $\le 5.7\text{ ms}$.

---

## Phase 7: End-to-End Closed-Loop System Integration & Fail-Safe Protocol

### 1. Goal (Kya achieve karna hai)
Data ingestion, neural networks, simulator, aur scheduler ko sub-25 ms pipeline mein connect karna aur clinical fail-safe hardware preemption provide karna.

### 2. Prerequisites & Inputs
* Phases 1 se 6 ke saare pre-trained checkpoints aur simulator modules.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/pipeline.py`
* `tests/test_integration.py`

### 5. Verification & Exit Criteria
* Pipeline ka total end-to-end AI latency budget $\le 24.1\text{ ms}$ ($14.8 + 3.6 + 5.7\text{ ms}$).
* 100-patient streaming simulation bina single unhandled exception ya memory leak ke complete ho.

---

## Phase 8: Peak-Load Experimental Benchmarking ($\lambda = 20$ tasks/sec)

### 1. Goal (Kya achieve karna hai)
Paper ke primary benchmarks replicate karna aur prove karna ki CRAI-LCS peak hospital load par baselines se superior hai.

### 2. Prerequisites & Inputs
* Phase 7 integrated pipeline.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/experiments/load_sweep.py`
* `src/experiments/plot_figures.py`
* `results/benchmarks_lambda20.csv`
* `results/graphs/figure3_latency_sweep.png`

### 5. Verification & Exit Criteria
* Peak load ($\lambda = 20$) par metrics paper se bitwise align karein:
  * **Latency:** $148.0\text{ ms}$ (vs $182.0\text{ ms}$ for best baseline EADRL).
  * **Violations:** $2.80\%$ (vs $5.90\%$ for EADRL).
  * **Risk Coverage:** $96.40\%$ (vs $84.20\%$ for EADRL).
  * **Energy:** $284.0\text{ Joules}$ (vs $342.0\text{ Joules}$ for EADRL).

---

## Phase 9: Statistical Rigor, Ablation Study & PTB-XL Generalization

### 1. Goal (Kya achieve karna hai)
Prove karna ki system ka performance statistical fluke nahi hai, aur model unseen clinical datasets par generalizable hai.

### 2. Prerequisites & Inputs
* Benchmark CSV data aur PhysioNet PTB-XL database.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `src/experiments/statistical_tests.py`
* `src/experiments/ablation_study.py`
* `src/experiments/generalization_test.py`
* `results/ablation_results.csv`

### 5. Verification & Exit Criteria
* Wilcoxon $p$-value $0.00195 < 0.01$ assert pass kare.
* PTB-XL retention rate $\ge 98.0\%$ report ho.

---

## Phase 10: Production Delivery, Automated QA Suite & Unified CLI

### 1. Goal (Kya achieve karna hai)
Entire codebase ko single turnkey entrypoint aur automated CI/CD test pyramid ke saath package karna.

### 2. Prerequisites & Inputs
* Phases 0 through 9 complete.

### 3. Step-by-Step Kaise Karna Hai (Start to End)
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

### 4. Deliverable Files
* `main.py`
* `tests/test_clean_run.py`
* `docs/REPRODUCTION_CHECKLIST.md`

### 5. Verification & Exit Criteria
* Isolated environment mein `pytest tests/ -v` run karne par all 24+ automated tests green pass karein.

---

## Critical 4 Engineering Gotchas & Solutions

| Gotcha ID | Critical Pitfall | Root Cause | CRAI-LCS Architectural Solution |
| :--- | :--- | :--- | :--- |
| **RSK-001** | **Patient Data Leakage** | Beats ko randomly shuffle karne se same patient train aur test dono mein chala jata hai $\to$ False $99\%$ accuracy. | PhysioNet dataset ko strictly Patient Record ID ke basis pe partition karein: $\text{Records}_{train} \cap \text{Records}_{test} \equiv \emptyset$. |
| **RSK-007** | **Reward Scale Explosion** | Latency milliseconds mein hoti hai ($100-800$), jabki clinical risk $[0, 1]$ mein hai. Bada latency term gradient dominate kar deta hai. | Latency ko task deadline se normalize karein: $\tilde{L}_i = L_i / D_i$. Isse saare terms $\approx \mathcal{O}(1)$ scale pe rehte hain. |
| **RSK-005** | **Predictor Cold-Start** | MLP ko train karne ke liye historical violation data chahiye, par start mein labels nahi hote. | **3-Stage Bootstrapping:** Pehle 50k steps baseline heuristics se warm-up data log karein, phir MLP train karein, phir DRL start karein. |
| **RSK-008** | **Maximization Bias** | Standard Q-learning 26 discrete actions mein Q-values ko systematically overestimate kar deta hai. | **Double DQN Decoupling:** Action selection online network $\theta$ karega, target return evaluation target network $\theta^-$ karega. |

---

## Turnkey Reproduction CLI Commands

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
