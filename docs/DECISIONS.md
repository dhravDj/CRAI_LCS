# CRAI-LCS: Engineering Decisions & Architectural Rationale

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Tracking Status:** ACTIVE

---

## Decision Log

### DEC-001: Record-Wise Patient Partitioning vs. Random Sample Splitting
* **Date:** 2026-11-21
* **Context:** In ECG machine learning, randomly shuffling 5-second segments between train and test sets leads to segments from the *same patient* appearing in both splits.
* **Decision:** Enforce strict **Record-wise Partitioning** where records 100-124 are allocated to training (70%), 200-215 to validation (15%), and 217-234 to testing (15%).
* **Paper Compliance:** Strictly adheres to Section "Clinical risk estimation model" (Eq. 26).
* **Alternatives Considered:** Random k-fold cross-validation (rejected due to severe data leakage).
* **Impact:** Guarantees that the reported 93.4% accuracy reflects true unseen-patient generalization.

### DEC-002: Double DQN (DDQN) over Standard DQN
* **Date:** 2026-11-21
* **Context:** The action space consists of 26 discrete actions (25 edge nodes + 1 cloud node). Standard DQN has a known tendency to significantly overestimate action-values in large action spaces.
* **Decision:** Implement Double DQN where the online network $\theta$ selects the best action, but the target network $\theta^-$ evaluates its Q-value: $y_t = R_t + \lambda Q(s_{t+1}, \text{argmax}_{a'} Q(s_{t+1}, a'; \theta); \theta^-)$.
* **Paper Compliance:** Compatible extension to the DQN formulation in Section "Reinforcement learning framework" (Eq. 30-31).
* **Alternatives Considered:** Vanilla DQN (rejected due to Q-value overestimation), PPO (rejected because paper specifies discrete DQN).
* **Impact:** Greatly improves training stability and prevents policy collapse to a single node.

### DEC-003: Reward Latency Normalization
* **Date:** 2026-11-21
* **Context:** Equation (29) defines $R_t = -(\alpha L_i + \beta P_{miss,i} - \gamma R_i) + \eta U_{edge}$ with default weights $\alpha=0.35, \beta=0.25, \gamma=0.30, \eta=0.10$. However, latency $L_i$ is measured in milliseconds ($100-800$ ms), while $P_{miss,i}, R_i, U_{edge} \in [0, 1]$.
* **Decision:** Normalize latency by the task deadline: $L_{norm} = L_i / D_i$. If $L_i \le D_i$, $L_{norm} \in [0, 1]$; if a deadline is missed, $L_{norm} > 1$.
* **Paper Compliance:** Mathematically preserves the exact multi-objective trade-off of Equation 29 while preventing the millisecond latency magnitude from drowning out the clinical risk bonus $\gamma R_i$.
* **Alternatives Considered:** Raw millisecond latency (rejected because risk term would be ignored by gradient updates).
* **Impact:** Allows the agent to learn balanced trade-offs between speed and patient risk.

### DEC-004: Open-Source Baseline Scheduling Interface
* **Date:** 2026-11-21
* **Context:** Evaluating CRAI-LCS requires benchmarking against classical baselines (FIFO, Round Robin, Random, EDF, Risk Priority) under identical workloads.
* **Decision:** Adopt the stateless interface pattern from `PalashSharma19/baselines.py`: `schedule(observation) -> int`.
* **Paper Compliance:** Ensures fair, unbiased benchmarking matching Table 5 and Table 8.
* **Alternatives Considered:** Ad-hoc simulation functions inside the environment loop (rejected due to lack of modularity).
* **Impact:** Any baseline can be swapped in with one line of code.

### DEC-005: 1D-CNN Downsampling before LSTM
* **Date:** 2026-11-21
* **Context:** Raw ECG segments contain 1800 samples (5 seconds @ 360 Hz). Feeding 1800 sequential steps directly into an LSTM causes slow training and vanishing gradients.
* **Decision:** Pass the input through two 1D-CNN + MaxPool1D stages to compress the sequence from 1800 to 450 feature tokens before feeding into the LSTM.
* **Paper Compliance:** Follows Equation (18) and Table 12, maintaining a $0.74$M parameter budget and $<15$ ms inference latency.
* **Alternatives Considered:** Pure LSTM (rejected due to $60+$ ms inference latency), pure CNN (rejected because paper specifies hybrid CNN-LSTM).
* **Impact:** Meets edge real-time inference latency constraints ($\approx 14.8$ ms).
