# CRAI-LCS: Master Risk Register & Failure Mode Analysis

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Tracking Status:** ACTIVE

---

## 1. Risk Summary Matrix

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

## 2. Detailed Risk Specifications

### RSK-001: Patient Data Leakage Across Splits
* **Domain:** Dataset & Data Engineering
* **Phase:** Phase 3
* **Severity:** **CRITICAL** | **Probability:** High
* **Symptoms:** Model achieves artificially inflated test accuracy ($\ge 99\%$) on MIT-BIH, but completely fails when tested on unseen patient data (PTB-XL).
* **Root Cause:** Randomly shuffling 5-second segments between train and test sets without grouping by patient record ID.
* **Detection Method:** Automated validation script asserting that the set intersection of patient IDs in `train_records`, `val_records`, and `test_records` is strictly empty.
* **Resolution:** Partition by record: records 100-124 (train), 200-215 (val), 217-234 (test).
* **Prevention:** Enforce record-wise splitting inside `src/ecg/preprocessing.py`.
* **Human Involvement:** Human verifies record partition lists before model training.

### RSK-002: Severe Class Imbalance (Normal vs. Pathological)
* **Domain:** Dataset & Data Engineering
* **Phase:** Phase 3
* **Severity:** **HIGH** | **Probability:** High
* **Symptoms:** High classification accuracy ($90\%+$), but near-zero recall for life-threatening arrhythmias ($R_i \approx 0$).
* **Root Cause:** Normal sinus rhythm beats vastly outnumber pathological arrhythmias in ambulatory monitoring.
* **Detection Method:** Monitor Precision, Recall, and confusion matrix during validation.
* **Resolution:** Use weighted Binary Cross-Entropy loss (`pos_weight`) to penalize false negatives on critical cardiac events.
* **Prevention:** Calculate positive weight dynamically based on training split class ratio.
* **Human Involvement:** Human approves loss weighting factor.

### RSK-003: Vanishing Gradients & Slow Recurrent Training
* **Domain:** Deep Learning Model
* **Phase:** Phase 4
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Training loss flatlines; epoch duration exceeds 20 minutes on CPU.
* **Root Cause:** Feeding 1800 raw temporal samples directly into an uncompressed LSTM layer.
* **Detection Method:** Monitor gradient norms and training step execution time.
* **Resolution:** Apply 1D-CNN feature extraction with MaxPool1D downsampling to reduce 1800 steps to 450 steps before passing into LSTM.
* **Prevention:** Enforce the hybrid CNN-LSTM architecture defined in Equation (18).
* **Human Involvement:** None required if parameter count stays within $0.74$M.

### RSK-004: Inference Latency Exceeding Edge Budget
* **Domain:** Deep Learning Model
* **Phase:** Phase 4
* **Severity:** **MEDIUM** | **Probability:** Low
* **Symptoms:** Edge gateway inference latency exceeds 20 ms, causing task scheduling deadline breaches before assignment.
* **Root Cause:** Excessive model parameterization or oversized dense layers.
* **Detection Method:** Measure 1,000-pass forward inference time on CPU with PyTorch benchmark timer.
* **Resolution:** Cap LSTM hidden units to 64 and prune dense layers to strictly achieve $0.74$M parameters.
* **Prevention:** Profile parameter count and inference latency in unit tests.
* **Human Involvement:** None.

### RSK-005: Cold-Start Bootstrapping for Violation Predictor
* **Domain:** Predictor & Simulation
* **Phase:** Phase 5
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Predictor cannot train because no historical deadline violation labels exist before simulation.
* **Root Cause:** Cyclic dependency between simulator execution and predictor input.
* **Detection Method:** Missing training dataset file `predictor_training_data.npz`.
* **Resolution:** Execute an exploratory warm-up simulation phase using baseline schedulers (FIFO, Random) for 50,000 steps to generate labeled state-outcome pairs.
* **Prevention:** Sequence Phase 2 (Baselines) before Phase 5 (Predictor Training).
* **Human Involvement:** None.

### RSK-006: Discrete-Event Clock Desynchronization
* **Domain:** Simulation Engine
* **Phase:** Phase 1
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Parallel edge node queues report negative delays or tasks complete in reverse chronological order.
* **Root Cause:** Incrementing simulation time synchronously per task rather than maintaining an event priority queue.
* **Detection Method:** Automated simulation test comparing simulated task start times with prior task finish times.
* **Resolution:** Track each node's available time individually: $\text{NodeAvailableTime}_j = \max(\text{NodeAvailableTime}_j, t_{arrival}) + t_{exec}$.
* **Prevention:** Implement strict queue availability timestamp tracking in `src/simulator/nodes.py`.
* **Human Involvement:** None.

### RSK-007: Reward Scale Explosion in RL Objective
* **Domain:** Reinforcement Learning
* **Phase:** Phase 6
* **Severity:** **CRITICAL** | **Probability:** High
* **Symptoms:** DQN completely ignores clinical risk ($R_i$) and only minimizes latency; Q-values explode to large negative values.
* **Root Cause:** Raw latency is in hundreds of milliseconds ($100-800$ ms), while risk $R_i \in [0, 1]$. The latency penalty overwhelms the clinical reward by a factor of $500\times$.
* **Detection Method:** Print individual reward components $(\alpha L_i, \beta P_{miss}, \gamma R_i, \eta U_{edge})$ during early training.
* **Resolution:** Normalize latency by the task deadline: $L_{norm} = L_i / D_i$.
* **Prevention:** Enforce reward normalization in `src/training/reward_shaping.py`.
* **Human Involvement:** Human reviews reward normalization formula.

### RSK-008: DQN Q-Value Overestimation
* **Domain:** Reinforcement Learning
* **Phase:** Phase 6
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** DQN scheduler converges to a suboptimal policy (e.g., assigning all tasks to the cloud or a single edge node).
* **Root Cause:** Standard Q-learning maximization bias across 26 discrete actions.
* **Detection Method:** Q-value tracking plot shows steady upward bias disconnected from actual episodic rewards.
* **Resolution:** Implement Double DQN (DDQN) where the online network selects the action and the target network computes the target Q-value.
* **Prevention:** Use DDQN formulation by default.
* **Human Involvement:** None.

### RSK-009: Interface Type Inconsistencies
* **Domain:** Integration
* **Phase:** Phase 7
* **Severity:** **MEDIUM** | **Probability:** Medium
* **Symptoms:** Runtime crash `TypeError: can't convert cuda:0 device type tensor to numpy`.
* **Root Cause:** Inconsistent boundary passing between PyTorch neural network tensors and Gymnasium NumPy environment states.
* **Detection Method:** Integration tests in `tests/test_integration.py`.
* **Resolution:** Standardize all module input/output signatures to native NumPy float32 arrays; isolate PyTorch tensor conversions strictly inside model forward passes.
* **Prevention:** Use Python type hints (`np.ndarray`) and automated runtime assertions.
* **Human Involvement:** None.

### RSK-010: High Variance Invalidating Statistical Significance
* **Domain:** Experimental Benchmarking
* **Phase:** Phase 8
* **Severity:** **HIGH** | **Probability:** Medium
* **Symptoms:** Wilcoxon signed-rank test yields $p > 0.05$ due to extreme variance between random seed iterations.
* **Root Cause:** Failure to seed all stochastic subsystems (Python `random`, `numpy.random`, `torch.manual_seed`).
* **Detection Method:** Compare standard deviations in Table 9; check if metric variance exceeds $10\%$.
* **Resolution:** Implement a unified `set_seed(seed)` function setting all pseudorandom generators before each run.
* **Prevention:** Seed every experiment run explicitly.
* **Human Involvement:** Human reviews statistical report table and p-values.
