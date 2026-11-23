# CRAI-LCS: Master Experiment Log

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Tracking Status:** ACTIVE

---

## Published Paper Reference Benchmarks (Nature Scientific Reports 2026)

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

## Live Experimental Runs Ledger

*(New experiment runs executed by our scripts will be appended below with timestamp, random seed, and raw output metrics).*

| Run ID | Timestamp | Config Profile | Random Seed | Scheduler | Task Count | Arrival Rate ($\lambda$) | Mean Latency (ms) | Violation Rate (%) | Risk Coverage (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-INIT* | 2026-11-21 | Default | 42 | Initial Test | 0 | 0 | - | - | - | Initialized |
