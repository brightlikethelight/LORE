#!/usr/bin/env python3
"""Generate plots for CRH CoT elicitation experiment."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

def main():
    # Load raw data
    data_path = Path("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_cot_elicit_n100/raw_data.json")
    with open(data_path) as f:
        data = json.load(f)

    baseline_results = data["baseline"]
    evolved_results = data["evolved"]

    # Extract scores
    baseline_cot = np.array([r["cot_suspicion"] for r in baseline_results])
    evolved_cot = np.array([r["cot_suspicion"] for r in evolved_results])

    n = len(baseline_cot)
    print(f"Loaded {n} paired samples")

    # Output directory
    output_dir = Path("/home/mila/t/thomas.jiralerspong/LORE/experiments/assets/2026-01-21_crh_cot_elicitation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # Figure 1: Overlaid distributions (histogram + KDE)
    # =========================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    bins = np.arange(0, 105, 5)

    # Histograms with transparency
    ax1.hist(baseline_cot, bins=bins, alpha=0.5, density=True,
             label=f'Baseline (μ={np.mean(baseline_cot):.1f})',
             color='#3274A1', edgecolor='#3274A1', linewidth=1.5)
    ax1.hist(evolved_cot, bins=bins, alpha=0.5, density=True,
             label=f'Evolved (μ={np.mean(evolved_cot):.1f})',
             color='#E1812C', edgecolor='#E1812C', linewidth=1.5)

    # Add KDE curves for smoother visualization
    x_range = np.linspace(0, 100, 200)

    # KDE for baseline
    baseline_kde = stats.gaussian_kde(baseline_cot, bw_method=0.15)
    ax1.plot(x_range, baseline_kde(x_range), color='#3274A1', linewidth=2.5, linestyle='-')

    # KDE for evolved
    evolved_kde = stats.gaussian_kde(evolved_cot, bw_method=0.15)
    ax1.plot(x_range, evolved_kde(x_range), color='#E1812C', linewidth=2.5, linestyle='-')

    ax1.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Detection threshold')

    ax1.set_xlabel('CoT Suspicion Score', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title(f'Distribution of CoT Suspicion Scores (n={n}, with CoT Elicitation)', fontsize=15)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig1.savefig(output_dir / "distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved distribution.png")

    # =========================================
    # Figure 2: Scatter with jitter
    # =========================================
    fig2, ax2 = plt.subplots(figsize=(9, 9))

    # Add jitter
    jitter_amount = 2.0
    baseline_jittered = baseline_cot + np.random.uniform(-jitter_amount, jitter_amount, n)
    evolved_jittered = evolved_cot + np.random.uniform(-jitter_amount, jitter_amount, n)

    # Color by improvement
    colors = np.where(evolved_cot < baseline_cot, 'green',
                      np.where(evolved_cot > baseline_cot, 'red', 'gray'))

    ax2.scatter(baseline_jittered, evolved_jittered, alpha=0.4, s=25, c=colors,
                edgecolors='none')

    # Diagonal line
    ax2.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='No change', zorder=10)

    # Detection threshold lines
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.axvline(x=50, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='Detection threshold')

    ax2.set_xlabel('Baseline CoT Suspicion', fontsize=14)
    ax2.set_ylabel('Evolved CoT Suspicion', fontsize=14)
    ax2.set_title(f'Paired Comparison (n={n})\\nGreen = evolved lower, Red = evolved higher', fontsize=14)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)

    # Counts
    evolved_lower = np.sum(evolved_cot < baseline_cot)
    evolved_higher = np.sum(evolved_cot > baseline_cot)
    ax2.text(85, 10, f'Lower: {evolved_lower}\n({evolved_lower/n*100:.1f}%)',
             fontsize=11, ha='center', color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(10, 90, f'Higher: {evolved_higher}\n({evolved_higher/n*100:.1f}%)',
             fontsize=11, ha='center', color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig2.savefig(output_dir / "scatter.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved scatter.png")

    # =========================================
    # Figure 3: Combined figure
    # =========================================
    fig3, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Overlaid distributions
    ax1 = axes[0]
    ax1.hist(baseline_cot, bins=bins, alpha=0.5, density=True,
             label=f'Baseline (μ={np.mean(baseline_cot):.1f})',
             color='#3274A1', edgecolor='#3274A1', linewidth=1)
    ax1.hist(evolved_cot, bins=bins, alpha=0.5, density=True,
             label=f'Evolved (μ={np.mean(evolved_cot):.1f})',
             color='#E1812C', edgecolor='#E1812C', linewidth=1)
    ax1.plot(x_range, baseline_kde(x_range), color='#3274A1', linewidth=2.5)
    ax1.plot(x_range, evolved_kde(x_range), color='#E1812C', linewidth=2.5)
    ax1.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Detection threshold')
    ax1.set_xlabel('CoT Suspicion Score', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('Overlaid Suspicion Score Distributions', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Right: Scatter with jitter
    ax2 = axes[1]
    ax2.scatter(baseline_jittered, evolved_jittered, alpha=0.4, s=20, c=colors, edgecolors='none')
    ax2.plot([0, 100], [0, 100], 'k--', linewidth=1.5, label='No change', zorder=10)
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.axvline(x=50, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='Detection threshold')
    ax2.set_xlabel('Baseline CoT Suspicion', fontsize=13)
    ax2.set_ylabel('Evolved CoT Suspicion', fontsize=13)
    ax2.set_title('Paired Comparison (jittered)', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_aspect('equal')
    ax2.grid(alpha=0.3)
    ax2.text(85, 10, f'Lower: {evolved_lower}', fontsize=10, ha='center', color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(10, 90, f'Higher: {evolved_higher}', fontsize=10, ha='center', color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'CRH Evaluation with CoT Elicitation (n={n}, STEALTH Policy)', fontsize=15, y=1.02)
    plt.tight_layout()
    fig3.savefig(output_dir / "combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved combined.png")

    # Print summary stats
    print(f"\nSummary Statistics:")
    print(f"  Baseline mean: {np.mean(baseline_cot):.1f}")
    print(f"  Evolved mean: {np.mean(evolved_cot):.1f}")
    print(f"  Delta: {np.mean(evolved_cot) - np.mean(baseline_cot):.1f}")
    print(f"  Evolved lower: {evolved_lower}/{n} ({evolved_lower/n*100:.1f}%)")
    print(f"  Evolved higher: {evolved_higher}/{n} ({evolved_higher/n*100:.1f}%)")

    # Statistical tests
    stat, p_value = stats.wilcoxon(baseline_cot, evolved_cot)
    print(f"\nWilcoxon signed-rank: p={p_value:.6f}")

    # Detection rates
    baseline_detected = sum(1 for r in baseline_results if r["cot_detected"])
    evolved_detected = sum(1 for r in evolved_results if r["cot_detected"])
    print(f"\nDetection rates:")
    print(f"  Baseline: {baseline_detected}/{n} ({baseline_detected/n*100:.1f}%)")
    print(f"  Evolved: {evolved_detected}/{n} ({evolved_detected/n*100:.1f}%)")


if __name__ == "__main__":
    main()
