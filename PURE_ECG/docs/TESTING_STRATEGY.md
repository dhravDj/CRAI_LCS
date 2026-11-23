# CRAI-LCS: Layered Testing Framework & QA Strategy

**Project:** Clinical Risk-Aware Reinforcement Learning for Latency-Constrained Healthcare IoT Scheduling  
**Version:** 1.0.0  
**Tracking Status:** ACTIVE

---

## 1. Multi-Level Testing Hierarchy

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

## 2. Test Level Definitions & Execution Commands

### Level 1: Static Analysis & Import Checks
* **Objective:** Ensure clean syntax, type hinting consistency, and module availability.
* **Commands:**
  ```bash
  python3 -m py_compile $(find src/ -name "*.py")
  pytest --collect-only tests/
  ```
* **Pass Criteria:** Zero syntax errors; zero missing imports.

### Level 2: Component Unit Tests
* **Objective:** Verify isolated classes and functions against deterministic assertions.
* **Key Test Suites:**
  - `tests/test_nodes.py`: Verifies node capacity, MIPS ranges, and power specifications.
  - `tests/test_latency.py`: Compares computed latency against hand-calculated values.
  - `tests/test_baselines.py`: Verifies priority order for FIFO, Round-Robin, and EDF.
* **Command:** `pytest tests/test_*.py -k "unit"`
* **Pass Criteria:** 100% of unit assertions pass.

### Level 3: Module Integration Tests
* **Objective:** Test interactions between PyTorch models, data loaders, and Gymnasium simulator.
* **Key Test Suites:**
  - `tests/test_model_sim_bridge.py`: Ensures CNN-LSTM risk outputs pipe directly into simulator task queues without type errors.
* **Command:** `pytest tests/test_integration.py`
* **Pass Criteria:** Zero tensor-to-numpy conversion errors; no unhandled exceptions.

### Level 4: Functional Task Lifecycle Tests
* **Objective:** Execute 1,000 continuous tasks through the full pipeline.
* **Command:** `python3 main.py --mode test-run --tasks 1000`
* **Pass Criteria:** All 1,000 tasks complete execution; queues empty cleanly; metrics logged.

### Level 5: Scientific & Numerical Validation
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

### Level 6: Regression Testing
* **Objective:** Ensure new model training or environment modifications do not break previously verified baseline schedulers or data pipelines.
* **Command:** `pytest tests/`
* **Pass Criteria:** Zero regressions in existing test suite.

### Level 7: Performance & Resource Profiling
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

## 3. Strict Debugging Protocol

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
