"""
Reproduce paper figures — Figures 1, 2, 3 (actual results).

All figures use the publication-quality style consistent with the paper.
Output: PNG files at 300 dpi saved to outputs/figures/
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

BLUE_DARK  = "#185FA5"; BLUE_LIGHT = "#B5D4F4"; BLUE_PALE = "#E6F1FB"
RED_MID    = "#E24B4A"; RED_LIGHT  = "#F09595"; GRAY_MID  = "#888780"
GRAY_DARK  = "#5F5E5A"; GREEN_DARK = "#0F6E56"; GREEN_MID = "#1D9E75"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def figure1_sharpe_sortino(
    sharpe:  dict,   # {strategy: sr_value}
    sortino: dict,   # {strategy: so_value}
    rw_sr:   float = 0.24,
    out_dir: str   = "outputs/figures",
):
    """Figure 1: Sharpe and Sortino ratios by customer segment."""
    labels = list(sharpe.keys())
    sr_vals = [sharpe[k] for k in labels]
    so_vals = [sortino[k] for k in labels]
    x = np.arange(len(labels)); w = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sr_colors = [RED_LIGHT if v < 0 else (BLUE_PALE if k == "Random Walk" else BLUE_DARK)
                 for k, v in zip(labels, sr_vals)]
    sr_edges  = ["#A32D2D" if v < 0 else ("#378ADD" if k == "Random Walk" else "#0C447C")
                 for k, v in zip(labels, sr_vals)]
    so_colors = [RED_LIGHT if v < 0 else BLUE_LIGHT for v in so_vals]
    so_edges  = ["#A32D2D" if v < 0 else "#378ADD" for v in so_vals]

    bars1 = ax.bar(x - w/2, sr_vals, w, color=sr_colors, edgecolor=sr_edges,
                   linewidth=0.8, zorder=3, label="Sharpe ratio")
    bars2 = ax.bar(x + w/2, so_vals, w, color=so_colors, edgecolor=so_edges,
                   linewidth=0.8, zorder=3, label="Sortino ratio")

    ax.axhline(rw_sr, color=RED_MID, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(0, color=GRAY_MID, linewidth=0.6, zorder=1)

    for bar, val in zip(bars1, sr_vals):
        yp = val + 0.04 if val >= 0 else val - 0.09
        ax.text(bar.get_x() + bar.get_width()/2, yp, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, color=GRAY_DARK)
    for bar, val in zip(bars2, so_vals):
        yp = val + 0.04 if val >= 0 else val - 0.09
        ax.text(bar.get_x() + bar.get_width()/2, yp, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, color=GRAY_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=10)
    ax.set_ylabel("Ratio", fontsize=10); ax.set_ylim(-0.65, 1.65)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

    p1 = mpatches.Patch(facecolor=BLUE_DARK, edgecolor="#0C447C", linewidth=0.8,
                        label="Sharpe ratio")
    p2 = mpatches.Patch(facecolor=BLUE_LIGHT, edgecolor="#378ADD", linewidth=0.8,
                        label="Sortino ratio")
    p3 = mlines.Line2D([], [], color=RED_MID, linewidth=1.2, linestyle="--",
                       label=f"Random walk SR ({rw_sr:.2f})")
    ax.legend(handles=[p1, p2, p3], fontsize=9, frameon=False, loc="upper right")
    ax.set_title(
        "Figure 1: Out-of-sample portfolio performance by customer order flow segment\n"
        "(Monthly rebalancing; target volatility σ* = 10%)",
        fontsize=10, pad=10, loc="left",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/figure1_sharpe_sortino.png")
    plt.close(fig)
    print(f"  Saved {out_dir}/figure1_sharpe_sortino.png")


def figure2_r2_decomposition(
    r2_results: dict,   # {segment: {"r2": float, "orthogonal": float}}
    out_dir: str = "outputs/figures",
):
    """Figure 2: R² decomposition bar chart (PRIMARY NMT RESULT)."""
    # Sort by ascending R² (most informed first)
    segments = sorted(r2_results, key=lambda k: r2_results[k]["r2"])
    r2_vals  = [r2_results[s]["r2"] for s in segments]
    orth_vals = [r2_results[s]["orthogonal"] for s in segments]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(segments)); h = 0.45

    ax.barh(y, r2_vals,   h, color=BLUE_DARK,  edgecolor="#0C447C",
            linewidth=0.8, zorder=3)
    ax.barh(y, orth_vals, h, left=r2_vals, color=BLUE_PALE,
            edgecolor=BLUE_LIGHT, linewidth=0.8, zorder=3)

    for i, (rv, ov) in enumerate(zip(r2_vals, orth_vals)):
        ax.text(rv/2,       i, f"{rv:.1f}%", ha="center", va="center",
                fontsize=9.5, color="white", fontweight="500")
        ax.text(rv + ov/2,  i, f"{ov:.1f}%", ha="center", va="center",
                fontsize=9.5, color=BLUE_DARK)

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ").title() for s in segments], fontsize=10)
    ax.set_xlim(0, 100); ax.set_xlabel("Percentage (%)", fontsize=10)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

    p1 = mpatches.Patch(facecolor=BLUE_DARK,  edgecolor="#0C447C",
                        linewidth=0.8, label="Explained by macro benchmarks (R²)")
    p2 = mpatches.Patch(facecolor=BLUE_PALE,  edgecolor=BLUE_LIGHT,
                        linewidth=0.8, label="Unexplained variation")
    ax.legend(handles=[p1, p2], fontsize=9, frameon=False, loc="lower right")
    ax.set_title(
        "Figure 2: Share of order flow strategy returns explained by macro benchmarks\n"
        "(PRIMARY NMT RESULT — lower R² indicates information orthogonal to public data)",
        fontsize=10, pad=10, loc="left",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/figure2_r2_decomposition.png")
    plt.close(fig)
    print(f"  Saved {out_dir}/figure2_r2_decomposition.png")
