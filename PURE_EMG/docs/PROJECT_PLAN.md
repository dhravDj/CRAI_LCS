# CRAI-LCS: Master Project Plan & Phase Roadmap

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Date:** 2026-11-21  
**Status:** APPROVED FOR EXECUTION

---

## Complete Phase Roadmap Overview

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
(Audit)     (Simulator) (Baselines) (ECG Data)  (CNN-LSTM)  (Predictor)
                                                                 │
Phase 10 ◄── Phase 9 ◄── Phase 8 ◄── Phase 7 ◄── Phase 6 ◄───────┘
(Delivery)  (Ablation)  (Benchmark) (Integrate) (DQN Agent)
```

---

## Phase 0: Setup, Repository Audit & Research Verification

### A. Purpose
Establish a clean development environment, verify original paper claims and Zenodo code, audit the 6 candidate GitHub repositories, document licenses and reuse decisions, and initialize project tracking files.

### B. Prerequisites
Linux OS, Python 3.10+, Git, workspace access.

### C. Tasks
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

### D. Expected Output
Clean workspace skeleton, complete project management documentation, dependency installation.

### E. Problems We May Encounter
Package dependency conflicts, Python 3.14 deprecation warnings.

### F. Problem Resolution Plan
Pin exact versions in `requirements.txt`; verify imports inside a test script.

### G. Testing Plan
Run `python3 -c "import torch, wfdb, gymnasium, scipy, sklearn; print('OK')"` (Level 1).

### H. Exit Criteria
All documentation created, libraries verified, Git initialized.

### I. Human Review
Human reviews `PRD.md` and `REUSE_AUDIT.md` to confirm architecture direction.

### J. Checkpoint
`git tag phase0-audit-complete`.

---

## Phase 1: Edge-Cloud Simulation Engine

### A. Purpose
Build the core discrete-event simulation modeling 25 heterogeneous edge gateways and 2 cloud instances, implementing exact equations for latency, queueing, and energy.

### B. Prerequisites
Phase 0 complete.

### C. Tasks
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

### D. Expected Output
Fully functional discrete-event simulator executable with synthetic workloads.

### E. Problems We May Encounter
Queue overflow, millisecond-to-second unit confusion, infinite execution loops.

### F. Problem Resolution Plan
Enforce strict unit tests with known deterministic arrival times and hand-calculated outputs.

### G. Testing Plan
Run `tests/test_simulator.py` checking latency, capacity limits, and energy conservation.

### H. Exit Criteria
100% of deterministic simulation tests pass; all 26 actions assign tasks correctly.

### I. Human Review
Verify that simulated latencies ($3-12$ ms edge, $45-120$ ms cloud) match clinical intuition.

### J. Checkpoint
`git tag phase1-simulator-verified`.

---

## Phase 2: Baseline Schedulers

### A. Purpose
Implement 5 classical and heuristic scheduling baselines to establish a benchmark standard before developing RL.

### B. Prerequisites
Phase 1 complete.

### C. Tasks
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

### D. Expected Output
5 working baseline schedulers tested in the Phase 1 environment.

### E. Problems We May Encounter
Tie-breaking ambiguities when deadlines are identical.

### F. Problem Resolution Plan
Apply secondary priority sorting based on arrival order.

### G. Testing Plan
Execute `pytest tests/test_baselines.py`.

### H. Exit Criteria
All 5 baselines execute without runtime exceptions and produce valid metric dictionaries.

### I. Human Review
Inspect baseline rankings to confirm expected algorithmic behavior.

### J. Checkpoint
`git tag phase2-baselines-complete`.

---

## Phase 3: ECG Dataset Pipeline

### A. Purpose
Download, normalize, segment, and partition real cardiac signals from PhysioNet MIT-BIH without patient data leakage.

### B. Prerequisites
Phase 0 complete.

### C. Tasks
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

### D. Expected Output
Processed task dataset tensors stored in `data/processed/`.

### E. Problems We May Encounter
Memory exhaustion when holding all continuous signals in RAM.

### F. Problem Resolution Plan
Stream segments using memory-mapped NumPy files or PyTorch `Dataset` generators.

### G. Testing Plan
Run `tests/test_ecg_pipeline.py` testing shapes, sample rates, and partition independence.

### H. Exit Criteria
Dataset generated, verified zero-leakage, and validated with summary statistics.

### I. Human Review
Review class balance between normal and pathological beats.

### J. Checkpoint
`git tag phase3-ecg-pipeline-verified`.

---

## Phase 4: CNN-LSTM Clinical Risk Model ($R_i$)

### A. Purpose
Train and validate the 1D-CNN + 2-layer LSTM network to output clinical risk scores $R_i \in [0, 1]$ adhering to the $0.74$M parameter budget.

### B. Prerequisites
Phase 3 complete.

### C. Tasks
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

### D. Expected Output
Trained weights saved to `checkpoints/cnn_lstm_risk.pt` with evaluation metrics report.

### E. Problems We May Encounter
Overfitting on training patients due to morphological similarities.

### F. Problem Resolution Plan
Apply dropout ($0.2$) and weight decay ($10^{-4}$); verify performance exclusively on test records.

### G. Testing Plan
Execute `pytest tests/test_risk_model.py`.

### H. Exit Criteria
Model loads, executes in $<15$ ms, and satisfies performance criteria.

### I. Human Review
Verify ROC-AUC curve and confusion matrix.

### J. Checkpoint
`git tag phase4-risk-model-trained`.

---

## Phase 5: Deadline Violation Predictor ($P_{miss,i}$)

### A. Purpose
Implement and train an MLP module predicting the probability that an incoming task will breach its deadline based on dynamic system state.

### B. Prerequisites
Phase 1 and Phase 4 complete.

### C. Tasks
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

### D. Expected Output
Trained weights saved to `checkpoints/deadline_predictor.pt`.

### E. Problems We May Encounter
Feature colinearity causing prediction instability.

### F. Problem Resolution Plan
Normalize input features to standard scalar range $[0, 1]$ before passing into MLP.

### G. Testing Plan
Run `tests/test_predictor.py`.

### H. Exit Criteria
Predictor inference outputs valid probabilities in $<4$ ms.

### I. Human Review
Review calibration curve of predicted vs. observed violation frequencies.

### J. Checkpoint
`git tag phase5-predictor-complete`.

---

## Phase 6: DQN Scheduler Implementation

### A. Purpose
Implement and train the 26-action Double DQN agent with experience replay and multi-objective reward shaping.

### B. Prerequisites
Phases 1, 4, and 5 complete.

### C. Tasks
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

### D. Expected Output
Trained checkpoint `checkpoints/dqn_scheduler.pt` and training curve logs.

### E. Problems We May Encounter
Q-value divergence, policy collapse to a single node.

### F. Problem Resolution Plan
Use gradient clipping (max norm 1.0) and Double DQN action evaluation.

### G. Testing Plan
Run `tests/test_dqn.py` verifying state shape, action bounds, and loss reduction.

### H. Exit Criteria
Agent completes training without NaN values and demonstrates superior scheduling policy.

### I. Human Review
Inspect reward curve and node allocation distribution.

### J. Checkpoint
`git tag phase6-dqn-trained`.

---

## Phase 7: System Integration

### A. Purpose
Connect all modules into a unified, end-to-end pipeline: ECG $\to$ Risk Model $\to$ Task Generator $\to$ Predictor $\to$ DQN $\to$ Simulation $\to$ Metrics.

### B. Prerequisites
Phases 1 through 6 complete.

### C. Tasks
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

### D. Expected Output
Seamless end-to-end pipeline callable via Python API or CLI.

### E. Problems We May Encounter
Interface impedance (tensor vs numpy type mismatches).

### F. Problem Resolution Plan
Enforce explicit type conversion (`.detach().cpu().numpy()`) at module boundaries.

### G. Testing Plan
Execute `pytest tests/test_integration.py`.

### H. Exit Criteria
1,000 continuous tasks processed end-to-end with complete metric accounting.

### I. Human Review
Observe real-time console log of task scheduling decisions.

### J. Checkpoint
`git tag phase7-system-integrated`.

---

## Phase 8: Experimental Evaluation & Benchmarking

### A. Purpose
Replicate the paper's primary experimental evaluations (Tables 5, 7, 9) under peak workload ($\lambda = 20$) and multi-load sweeps ($\lambda \in [8, 25]$).

### B. Prerequisites
Phase 7 complete.

### C. Tasks
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

### D. Expected Output
Benchmark logs in `results/logs/` and figures in `results/graphs/`.

### E. Problems We May Encounter
Variance across seeds exceeding expected boundaries.

### F. Problem Resolution Plan
Verify environment resets use seeded random number generators (`np.random.seed`, `torch.manual_seed`).

### G. Testing Plan
Execute `pytest tests/test_experiments.py`.

### H. Exit Criteria
CRAI-LCS outperforms baselines with statistically significant p-values.

### I. Human Review
Examine comparison tables and graphs against published paper figures.

### J. Checkpoint
`git tag phase8-experiments-complete`.

---

## Phase 9: Ablation & Reliability Stress Testing

### A. Purpose
Replicate the paper's ablation studies (Table 8) and evaluate system reliability under network jitter, node failures, and risk skew.

### B. Prerequisites
Phase 8 complete.

### C. Tasks
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

### D. Expected Output
Ablation table, sensitivity results, and Figure 8 plots.

### E. Problems We May Encounter
Ablation variants collapsing into non-functional states.

### F. Problem Resolution Plan
Isolate ablation flags in `simulation_config.yaml` to ensure clean toggling.

### G. Testing Plan
Run `pytest tests/test_ablations.py`.

### H. Exit Criteria
Ablation table generated showing exact comparative drop-offs.

### I. Human Review
Confirm ablation percentages match published paper claims.

### J. Checkpoint
`git tag phase9-ablations-complete`.

---

## Phase 10: Final Documentation & Demonstration CLI

### A. Purpose
Provide unified CLI, clean documentation, reproducible execution scripts, and a final technical project report.

### B. Prerequisites
Phases 0 through 9 complete.

### C. Tasks
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

### D. Expected Output
Fully packaged, demonstrable research repository with complete documentation.

### E. Problems We May Encounter
Hardcoded absolute paths breaking clean installation.

### F. Problem Resolution Plan
Enforce `pathlib.Path` relative to project root across all file operations.

### G. Testing Plan
Execute full test suite `pytest tests/`.

### H. Exit Criteria
All unit, integration, and regression tests pass 100%.

### I. Human Review
Final human review and project acceptance.

### J. Checkpoint
`git tag v1.0.0-final-release`.
