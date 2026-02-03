#!/usr/bin/env python3
"""Clean plots for n=500 suspicion score distributions."""

import json
import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    # Load raw data
    data_path = Path("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_v6_stealth_n500/raw_data.json")
    with open(data_path) as f:
        data = json.load(f)

    baseline_results = data["baseline"]
    evolved_results = data["evolved"]

    # Extract scores
    baseline_cot = np.array([r["cot_suspicion"] for r in baseline_results])
    evolved_cot = np.array([r["cot_suspicion"] for r in evolved_results])

    n = len(baseline_cot)
    print(f"Loaded {n} paired samples")

    # Initialize wandb
    wandb.init(
        project="lore-crh-evaluation",
        name=f"crh_v6_n500_clean_plots_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config={
            "prompt_version": "v6",
            "attack_policy": "STEALTH",
            "n_samples": n,
        }
    )

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
    from scipy import stats

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
    ax1.set_title(f'Distribution of CoT Suspicion Scores (n={n}, STEALTH Policy)', fontsize=15)
    ax1.legend(fontsize=12, loc='upper center')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    wandb.log({"overlaid_distributions": wandb.Image(fig1)})
    fig1.savefig("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_v6_stealth_n500/overlaid_hist.png",
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

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
    ax2.set_title(f'Paired Comparison with Jitter (n={n})\nGreen = evolved lower, Red = evolved higher', fontsize=14)
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
    wandb.log({"scatter_jittered": wandb.Image(fig2)})
    fig2.savefig("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_v6_stealth_n500/scatter_jitter.png",
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # =========================================
    # Figure 3: 2D density/hexbin plot
    # =========================================
    fig3, ax3 = plt.subplots(figsize=(9, 9))

    # Hexbin for density
    hb = ax3.hexbin(baseline_cot, evolved_cot, gridsize=20, cmap='YlOrRd',
                    mincnt=1, extent=[0, 100, 0, 100])
    cb = plt.colorbar(hb, ax=ax3, label='Count')

    # Diagonal line
    ax3.plot([0, 100], [0, 100], 'k--', linewidth=2, label='No change', zorder=10)

    # Detection threshold lines
    ax3.axhline(y=50, color='blue', linestyle=':', alpha=0.7, linewidth=1.5)
    ax3.axvline(x=50, color='blue', linestyle=':', alpha=0.7, linewidth=1.5, label='Detection threshold')

    ax3.set_xlabel('Baseline CoT Suspicion', fontsize=14)
    ax3.set_ylabel('Evolved CoT Suspicion', fontsize=14)
    ax3.set_title(f'Density Plot: Baseline vs Evolved (n={n})', fontsize=14)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.set_aspect('equal')

    plt.tight_layout()
    wandb.log({"scatter_density": wandb.Image(fig3)})
    fig3.savefig("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_v6_stealth_n500/scatter_density.png",
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # =========================================
    # Figure 4: Combined clean figure
    # =========================================
    fig4, axes = plt.subplots(1, 2, figsize=(16, 7))

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

    plt.suptitle(f'CRH Monitorability: V6 Evolved Prompt (n={n}, STEALTH Policy)', fontsize=15, y=1.02)
    plt.tight_layout()
    wandb.log({"combined_clean": wandb.Image(fig4)})
    fig4.savefig("/home/mila/t/thomas.jiralerspong/LORE/results/crh_eval_v6_stealth_n500/combined_clean.png",
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)

    print(f"\nPlots saved!")
    print(f"  Evolved lower: {evolved_lower}/{n} ({evolved_lower/n*100:.1f}%)")
    print(f"  Evolved higher: {evolved_higher}/{n} ({evolved_higher/n*100:.1f}%)")
    print(f"  No change: {n - evolved_lower - evolved_higher}/{n}")

    wandb.finish()


if __name__ == "__main__":
    main()
