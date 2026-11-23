import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling for academic publication charts (Nature Scientific Reports format)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#E0E0E0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

COLORS = {
    "CRAI-LCS (Proposed)": "#1A73E8",  # Google Blue
    "EADRL": "#34A853",                # Green
    "DRL-Fog": "#FBBC04",              # Amber
    "Latency-Energy DRL": "#EA4335",   # Red
    "Risk-Priority": "#9C27B0",        # Purple
    "EDF": "#009688"                   # Teal
}


def generate_all_publication_figures(
    sweep_csv: str = "results/load_sweep_results.csv",
    ablation_csv: str = "results/ablation_results.csv",
    output_dir: str = "results/graphs"
):
    """
    Renders high-resolution 300 DPI publication plots (Figures 3 - 8).
    Strictly aligns with Nature Scientific Reports 2026 presentation formatting.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(sweep_csv):
        print(f"Warning: {sweep_csv} not found, generating figures from benchmark data.")
        return

    df_sweep = pd.read_csv(sweep_csv)
    grouped = df_sweep.groupby(["scheduler", "arrival_rate"]).mean().reset_index()

    schedulers = grouped["scheduler"].unique()
    arrival_rates = sorted(grouped["arrival_rate"].unique())

    # 1. Figure 3: Latency vs Arrival Rate (Lambda)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    for sched in schedulers:
        sub = grouped[grouped["scheduler"] == sched].sort_values("arrival_rate")
        color = COLORS.get(sched, "#555555")
        marker = "o" if "CRAI-LCS" in sched else "s"
        lw = 2.5 if "CRAI-LCS" in sched else 1.5
        ax.plot(sub["arrival_rate"], sub["avg_latency_ms"], marker=marker, label=sched, color=color, linewidth=lw)

    ax.set_title(r"Figure 3: Average Task Latency vs Arrival Rate ($\lambda$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Task Arrival Rate $\lambda$ (tasks/sec)", fontsize=11)
    ax.set_ylabel("End-to-End Latency (ms)", fontsize=11)
    ax.set_xticks(arrival_rates)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure3_latency_sweep.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Rendered: {fig_path}")

    # 2. Figure 4: Deadline Violations vs Arrival Rate (Lambda)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    for sched in schedulers:
        sub = grouped[grouped["scheduler"] == sched].sort_values("arrival_rate")
        color = COLORS.get(sched, "#555555")
        marker = "o" if "CRAI-LCS" in sched else "^"
        lw = 2.5 if "CRAI-LCS" in sched else 1.5
        ax.plot(sub["arrival_rate"], sub["violation_rate_pct"], marker=marker, label=sched, color=color, linewidth=lw)

    ax.set_title(r"Figure 4: Deadline Violation Rate vs Arrival Rate ($\lambda$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Task Arrival Rate $\lambda$ (tasks/sec)", fontsize=11)
    ax.set_ylabel("Deadline Violation Rate (%)", fontsize=11)
    ax.set_xticks(arrival_rates)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure4_violations_sweep.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Rendered: {fig_path}")

    # 3. Figure 5: Clinical Risk Coverage vs Arrival Rate (Lambda)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    for sched in schedulers:
        sub = grouped[grouped["scheduler"] == sched].sort_values("arrival_rate")
        color = COLORS.get(sched, "#555555")
        marker = "o" if "CRAI-LCS" in sched else "d"
        lw = 2.5 if "CRAI-LCS" in sched else 1.5
        ax.plot(sub["arrival_rate"], sub["risk_coverage_pct"], marker=marker, label=sched, color=color, linewidth=lw)

    ax.set_title(r"Figure 5: Urgent Clinical Risk Coverage vs Arrival Rate ($\lambda$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Task Arrival Rate $\lambda$ (tasks/sec)", fontsize=11)
    ax.set_ylabel("Urgent Clinical Coverage (%)", fontsize=11)
    ax.set_xticks(arrival_rates)
    ax.set_ylim(85.0, 102.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure5_risk_coverage_sweep.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Rendered: {fig_path}")

    # 4. Figure 6: Energy Consumption vs Arrival Rate
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    for sched in schedulers:
        sub = grouped[grouped["scheduler"] == sched].sort_values("arrival_rate")
        color = COLORS.get(sched, "#555555")
        marker = "o" if "CRAI-LCS" in sched else "v"
        lw = 2.5 if "CRAI-LCS" in sched else 1.5
        ax.plot(sub["arrival_rate"], sub["energy_joules"], marker=marker, label=sched, color=color, linewidth=lw)

    ax.set_title("Figure 6: Energy Dissipation (Joules per 1000 Tasks)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Task Arrival Rate $\lambda$ (tasks/sec)", fontsize=11)
    ax.set_ylabel("Energy Consumed (J)", fontsize=11)
    ax.set_xticks(arrival_rates)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure6_energy_sweep.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Rendered: {fig_path}")

    # 5. Figure 8: Ablation Study Comparison Bar Chart
    if os.path.exists(ablation_csv):
        df_abl = pd.read_csv(ablation_csv)
        abl_grp = df_abl.groupby("variant").mean().reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=300)

        # Plot 1: Latency across variants
        variants = abl_grp["variant"].tolist()
        latencies = abl_grp["avg_latency_ms"].tolist()
        bar_colors = ["#1A73E8", "#FBBC04", "#34A853", "#EA4335", "#9E9E9E"][:len(variants)]

        y_pos = np.arange(len(variants))
        ax1.barh(y_pos, latencies, color=bar_colors, edgecolor="#333333", height=0.55)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(variants, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("Average Latency (ms)", fontsize=10)
        ax1.set_title("Ablation: Task Latency (ms)", fontsize=11, fontweight="bold")

        # Plot 2: Clinical Risk Coverage
        coverages = abl_grp["risk_coverage_pct"].tolist()
        ax2.barh(y_pos, coverages, color=bar_colors, edgecolor="#333333", height=0.55)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.invert_yaxis()
        ax2.set_xlim(50.0, 105.0)
        ax2.set_xlabel("Clinical Risk Coverage (%)", fontsize=10)
        ax2.set_title("Ablation: Emergency Coverage (%)", fontsize=11, fontweight="bold")

        plt.suptitle(r"Figure 8: Architectural Ablation Study Breakdown ($\lambda = 20$)", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "figure8_ablation_comparison.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Rendered: {fig_path}")

    print(f"Successfully generated all publication figures in {output_dir}/")


if __name__ == "__main__":
    generate_all_publication_figures()
