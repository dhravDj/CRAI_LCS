import os
import pytest
import pandas as pd
import numpy as np

def test_wilcoxon_statistical_significance():
    """
    Verifies that Wilcoxon signed-rank test results reject null hypothesis (p < 0.05)
    across all primary metrics (Table 9 replication).
    """
    csv_path = "results/statistical_wilcoxon_results.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"

    df = pd.read_csv(csv_path)
    assert not df.empty

    for _, row in df.iterrows():
        p_val = float(row["p_value"])
        h0_rejected = bool(row["h0_rejected"])
        metric = row["metric"]

        print(f"Checking statistical significance for {metric}: p = {p_val:.5f}")
        assert p_val < 0.05, f"Metric {metric} failed p < 0.05 threshold with p={p_val}!"
        assert h0_rejected is True, f"Hypothesis H0 was not rejected for {metric}!"


def test_ablation_study_integrity_and_rankings():
    """
    Verifies that results/ablation_results.csv reflects theoretical ranking:
    - Full CRAI-LCS achieves lowest latency and best balance.
    - Variant D (Static Priority) has significantly higher latency than Full CRAI-LCS.
    """
    csv_path = "results/ablation_results.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "variant" in df.columns

    variants = df["variant"].unique()
    assert "Full CRAI-LCS (Proposed)" in variants
    assert "Variant A (w/o Clinical Risk)" in variants
    assert "Variant B (w/o Predictor)" in variants
    assert "Variant D (Static Priority)" in variants

    crai_lat = df[df["variant"] == "Full CRAI-LCS (Proposed)"]["avg_latency_ms"].mean()
    static_lat = df[df["variant"] == "Variant D (Static Priority)"]["avg_latency_ms"].mean()
    assert crai_lat < static_lat, f"Full CRAI-LCS ({crai_lat:.1f}ms) must outperform Static Priority ({static_lat:.1f}ms)!"


def test_ptb_xl_generalization_retention():
    """
    Verifies that PTB-XL cross-dataset generalization maintains >= 98.0% performance retention
    confirming hardware invariance without model retraining (Table 13 replication).
    """
    csv_path = "results/ptb_xl_generalization.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"

    df = pd.read_csv(csv_path)
    assert not df.empty

    retention_row = df[df["dataset"] == "Performance Retention"]
    assert not retention_row.empty, "Missing 'Performance Retention' row in generalization results!"

    acc_retention = float(retention_row["accuracy_pct"].values[0])
    auc_retention = float(retention_row["auc_roc"].values[0])

    print(f"PTB-XL Retention: Accuracy: {acc_retention:.2f}%, AUC: {auc_retention:.2f}%")
    assert acc_retention >= 98.0, f"Accuracy retention {acc_retention:.2f}% is below 98.0% threshold!"
    assert auc_retention >= 98.0, f"AUC retention {auc_retention:.2f}% is below 98.0% threshold!"
