#!/usr/bin/env python3
"""
Step 6.5: Publication-Quality Visualizations
=============================================
Generates 7 PNG charts from the experiment results.

Usage:
    python experiments/generate_figures.py

Outputs:
    results/figures/structural_comparison.png
    results/figures/win_rate_overall.png
    results/figures/win_rate_by_type.png
    results/figures/dimension_breakdown.png
    results/figures/degree_distribution.png
    results/figures/entity_fragmentation.png
    results/figures/graph_topology.png
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GRAPHS_DIR = os.path.join(PROJECT_DIR, "graphs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# === Color Palette ===
COL_FULL = "#4FC3F7"      # Light blue
COL_INC = "#FF7043"        # Orange-red
COL_TIE = "#81C784"        # Green
COL_BG = "#1a1a2e"         # Dark background
COL_CARD = "#16213e"       # Card background
COL_TEXT = "#e0e0e0"       # Light text
COL_GRID = "#333355"       # Grid lines

DPI = 300


def setup_style():
    """Apply dark academic theme."""
    plt.rcParams.update({
        "figure.facecolor": COL_BG,
        "axes.facecolor": COL_CARD,
        "axes.edgecolor": COL_GRID,
        "axes.labelcolor": COL_TEXT,
        "text.color": COL_TEXT,
        "xtick.color": COL_TEXT,
        "ytick.color": COL_TEXT,
        "grid.color": COL_GRID,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
    })


def fig_structural_comparison(metrics: dict) -> None:
    """Grouped bar chart comparing structural metrics."""
    full = metrics["full"]
    inc = metrics["inc"]

    labels = ["Nodes", "Edges", "Components", "Isolated\nNodes", "Small\nComponents"]
    keys = ["num_nodes", "num_edges", "num_components", "isolated_nodes", "small_components"]

    full_vals = [full[k] for k in keys]
    inc_vals = [inc[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, full_vals, width, label="Full Build", color=COL_FULL, alpha=0.9, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, inc_vals, width, label="Incremental Build", color=COL_INC, alpha=0.9, edgecolor="white", linewidth=0.5)

    # Add delta annotations
    for i, (fv, iv) in enumerate(zip(full_vals, inc_vals)):
        if fv > 0:
            delta_pct = (iv - fv) / fv * 100
            color = "#ff4444" if abs(delta_pct) > 10 else COL_TEXT
            ax.annotate(f"{delta_pct:+.1f}%",
                       xy=(x[i] + width/2, iv),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_ylabel("Count")
    ax.set_title("Structural Metrics: Full Build vs Incremental Build")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "structural_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ structural_comparison.png")


def fig_win_rate_overall(summary: dict) -> None:
    """Stacked horizontal bar showing overall win rates."""
    all_q = summary["all_queries"]

    categories = ["Overall"]
    full_pcts = [all_q["full_pct"]]
    inc_pcts = [all_q["inc_pct"]]
    tie_pcts = [all_q["tie_pct"]]

    # Add cross-doc and single-doc if available
    if summary.get("cross_document", {}).get("n", 0) > 0:
        cross = summary["cross_document"]
        categories.append("Cross-Document")
        full_pcts.append(cross["full_pct"])
        inc_pcts.append(cross["inc_pct"])
        tie_pcts.append(cross["tie_pct"])

    if summary.get("single_document", {}).get("n", 0) > 0:
        single = summary["single_document"]
        categories.append("Single-Document")
        full_pcts.append(single["full_pct"])
        inc_pcts.append(single["inc_pct"])
        tie_pcts.append(single["tie_pct"])

    fig, ax = plt.subplots(figsize=(10, 3 + len(categories) * 0.6))

    y = np.arange(len(categories))
    height = 0.5

    bars_full = ax.barh(y, full_pcts, height, label="Full Build Wins", color=COL_FULL, alpha=0.9)
    bars_inc = ax.barh(y, inc_pcts, height, left=full_pcts, label="Incremental Wins", color=COL_INC, alpha=0.9)
    left_for_tie = [f + i for f, i in zip(full_pcts, inc_pcts)]
    bars_tie = ax.barh(y, tie_pcts, height, left=left_for_tie, label="Tie", color=COL_TIE, alpha=0.9)

    # Add count labels
    for idx, cat in enumerate(categories):
        total_n = summary["all_queries"]["n"] if idx == 0 else (
            summary.get("cross_document", {}).get("n", 0) if idx == 1 else
            summary.get("single_document", {}).get("n", 0)
        )
        s = summary["all_queries"] if idx == 0 else (
            summary.get("cross_document", {}) if idx == 1 else
            summary.get("single_document", {})
        )

        # Label inside bars if wide enough
        if s.get("full_pct", 0) > 12:
            ax.text(s["full_pct"] / 2, idx, f'{s["full_wins"]}', ha="center", va="center", fontweight="bold", fontsize=10)
        if s.get("inc_pct", 0) > 12:
            ax.text(s["full_pct"] + s["inc_pct"] / 2, idx, f'{s["inc_wins"]}', ha="center", va="center", fontweight="bold", fontsize=10)
        if s.get("tie_pct", 0) > 12:
            ax.text(s["full_pct"] + s["inc_pct"] + s["tie_pct"] / 2, idx, f'{s["ties"]}', ha="center", va="center", fontweight="bold", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Win Rate (%)")
    ax.set_title("LLM Judge Win Rates: Full vs Incremental Build")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "win_rate_overall.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ win_rate_overall.png")


def fig_win_rate_by_type(summary: dict) -> None:
    """Grouped bar chart: win rates by query type."""
    by_type = summary.get("by_type", {})
    if not by_type:
        print("  ⚠ Skipping win_rate_by_type.png (no type data)")
        return

    types = sorted(by_type.keys())
    full_pcts = [by_type[t]["full_pct"] for t in types]
    inc_pcts = [by_type[t]["inc_pct"] for t in types]
    tie_pcts = [by_type[t]["tie_pct"] for t in types]
    counts = [by_type[t]["n"] for t in types]

    x = np.arange(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, full_pcts, width, label="Full Wins %", color=COL_FULL, alpha=0.9)
    ax.bar(x, inc_pcts, width, label="Inc Wins %", color=COL_INC, alpha=0.9)
    ax.bar(x + width, tie_pcts, width, label="Tie %", color=COL_TIE, alpha=0.9)

    # Add count labels at top
    for i, (t, c) in enumerate(zip(types, counts)):
        ax.text(i, max(full_pcts[i], inc_pcts[i], tie_pcts[i]) + 3,
                f"n={c}", ha="center", fontsize=9, color=COL_TEXT)

    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rates by Query Type")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "win_rate_by_type.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ win_rate_by_type.png")


def fig_dimension_breakdown(summary: dict) -> None:
    """Radar chart showing per-dimension win rates."""
    all_dims = summary.get("all_queries", {}).get("per_dimension", {})
    if not all_dims:
        print("  ⚠ Skipping dimension_breakdown.png (no dimension data)")
        return

    dims = list(all_dims.keys())
    labels = [d.replace("_", " ").title() for d in dims]

    full_vals = [all_dims[d].get("full_pct", 0) for d in dims]
    inc_vals = [all_dims[d].get("inc_pct", 0) for d in dims]

    # Close the radar
    labels_r = labels + [labels[0]]
    full_vals_r = full_vals + [full_vals[0]]
    inc_vals_r = inc_vals + [inc_vals[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor(COL_CARD)

    ax.fill(angles, full_vals_r, alpha=0.25, color=COL_FULL)
    ax.plot(angles, full_vals_r, "o-", color=COL_FULL, linewidth=2, label="Full Build", markersize=6)

    ax.fill(angles, inc_vals_r, alpha=0.25, color=COL_INC)
    ax.plot(angles, inc_vals_r, "s-", color=COL_INC, linewidth=2, label="Incremental Build", markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r[:-1], fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Per-Dimension Win Rate (%)", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
    ax.grid(color=COL_GRID, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dimension_breakdown.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ dimension_breakdown.png")


def fig_degree_distribution(metrics: dict) -> None:
    """Overlaid histogram of node degree distributions."""
    if nx is None:
        print("  ⚠ Skipping degree_distribution.png (networkx not available)")
        return

    full_dir = os.path.join(GRAPHS_DIR, "graph_full")
    inc_dir = os.path.join(GRAPHS_DIR, "graph_inc")

    try:
        G_full = nx.read_graphml(os.path.join(full_dir, "graph_chunk_entity_relation.graphml"))
        G_inc = nx.read_graphml(os.path.join(inc_dir, "graph_chunk_entity_relation.graphml"))
    except Exception as e:
        print(f"  ⚠ Skipping degree_distribution.png: {e}")
        return

    deg_full = [d for _, d in G_full.degree()]
    deg_inc = [d for _, d in G_inc.degree()]

    fig, ax = plt.subplots(figsize=(10, 5))

    max_deg = max(max(deg_full, default=0), max(deg_inc, default=0))
    bins = np.arange(0, min(max_deg + 2, 50), 1)

    ax.hist(deg_full, bins=bins, alpha=0.6, color=COL_FULL, label=f"Full Build (n={len(deg_full)})", edgecolor="white", linewidth=0.5)
    ax.hist(deg_inc, bins=bins, alpha=0.6, color=COL_INC, label=f"Incremental Build (n={len(deg_inc)})", edgecolor="white", linewidth=0.5)

    # Add mean lines
    mean_full = np.mean(deg_full) if deg_full else 0
    mean_inc = np.mean(deg_inc) if deg_inc else 0
    ax.axvline(mean_full, color=COL_FULL, linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(mean_inc, color=COL_INC, linestyle="--", linewidth=2, alpha=0.8)
    ax.text(mean_full + 0.3, ax.get_ylim()[1] * 0.9, f"μ={mean_full:.1f}", color=COL_FULL, fontsize=10)
    ax.text(mean_inc + 0.3, ax.get_ylim()[1] * 0.8, f"μ={mean_inc:.1f}", color=COL_INC, fontsize=10)

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Frequency")
    ax.set_title("Node Degree Distribution: Full vs Incremental")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "degree_distribution.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ degree_distribution.png")


def fig_entity_fragmentation(name_analysis: dict) -> None:
    """Horizontal bar chart of fuzzy-duplicate entity names."""
    matches = name_analysis.get("fuzzy_matches", [])
    if not matches:
        # Create a simple info chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No entity fragmentation detected\nbetween Full and Incremental graphs",
                ha="center", va="center", fontsize=14, color=COL_TEXT,
                transform=ax.transAxes)
        ax.set_title("Entity Name Fragmentation Analysis")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "entity_fragmentation.png"), dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  ✓ entity_fragmentation.png (no fragmentation)")
        return

    # Top 12 matches
    top = matches[:12]
    labels = [f"{m['inc_name']}\n↔ {m['full_name']}" for m in top]
    sims = [m["similarity"] if m["similarity"] > 0 else 0.5 for m in top]

    fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.5)))

    y = np.arange(len(labels))
    colors = [COL_INC if s > 0.8 else COL_TIE for s in sims]
    ax.barh(y, sims, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for i, s in enumerate(sims):
        ax.text(s + 0.01, i, f"{s:.0%}", va="center", fontsize=9, color=COL_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Name Similarity")
    ax.set_title("Entity Fragmentation: Potential Alias Mismatches")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "entity_fragmentation.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ entity_fragmentation.png")


def fig_graph_topology() -> None:
    """Side-by-side graph topology visualizations."""
    if nx is None:
        print("  ⚠ Skipping graph_topology.png (networkx not available)")
        return

    full_path = os.path.join(GRAPHS_DIR, "graph_full", "graph_chunk_entity_relation.graphml")
    inc_path = os.path.join(GRAPHS_DIR, "graph_inc", "graph_chunk_entity_relation.graphml")

    try:
        G_full = nx.read_graphml(full_path)
        G_inc = nx.read_graphml(inc_path)
    except Exception as e:
        print(f"  ⚠ Skipping graph_topology.png: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, G, title in [(ax1, G_full, "Full Build (Control)"), (ax2, G_inc, "Incremental Build (Test)")]:
        ax.set_facecolor(COL_CARD)

        if G.is_directed():
            G_draw = G.to_undirected()
        else:
            G_draw = G

        # Color by connected component
        components = list(nx.connected_components(G_draw))
        component_colors = {}
        palette = plt.cm.Set2(np.linspace(0, 1, min(len(components), 8)))

        for ci, comp in enumerate(sorted(components, key=len, reverse=True)):
            for node in comp:
                component_colors[node] = palette[min(ci, 7)]

        node_colors = [component_colors.get(n, palette[0]) for n in G_draw.nodes()]

        # Layout — limit nodes for performance
        if G_draw.number_of_nodes() > 200:
            # Sample largest component + some small ones
            largest = max(components, key=len)
            sample = set(list(largest)[:150])
            for comp in sorted(components, key=len, reverse=True)[1:]:
                if len(sample) < 200:
                    sample.update(list(comp)[:10])
            G_draw = G_draw.subgraph(sample)
            node_colors = [component_colors.get(n, palette[0]) for n in G_draw.nodes()]

        try:
            pos = nx.spring_layout(G_draw, seed=42, k=1.5/math.sqrt(max(G_draw.number_of_nodes(), 1)), iterations=50)
        except Exception:
            pos = nx.random_layout(G_draw, seed=42)

        nx.draw_networkx_edges(G_draw, pos, ax=ax, alpha=0.15, edge_color=COL_GRID, width=0.5)
        nx.draw_networkx_nodes(G_draw, pos, ax=ax, node_color=node_colors,
                              node_size=20, alpha=0.8, linewidths=0.3, edgecolors="white")

        n_comp = len(components)
        ax.set_title(f"{title}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_comp} components", fontsize=11)
        ax.axis("off")

    fig.suptitle("Graph Topology Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "graph_topology.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ graph_topology.png")


def main():
    print("=" * 60)
    print("  Step 6.5: Generating Visualizations")
    print("=" * 60)
    print()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_style()

    # Load data
    metrics_path = os.path.join(GRAPHS_DIR, "structural_metrics.json")
    summary_path = os.path.join(RESULTS_DIR, "analysis_summary.json")
    names_path = os.path.join(GRAPHS_DIR, "entity_name_analysis.json")

    metrics = None
    summary = None
    name_analysis = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        print(f"  ⚠ Metrics not found: {metrics_path}")

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        print(f"  ⚠ Summary not found: {summary_path}")

    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            name_analysis = json.load(f)
    else:
        print(f"  ⚠ Name analysis not found: {names_path}")

    # Generate each figure
    print("  Generating figures:")

    if metrics:
        fig_structural_comparison(metrics)
        fig_degree_distribution(metrics)
    else:
        print("  ⚠ Skipping structural charts (no metrics data)")

    if summary:
        fig_win_rate_overall(summary)
        fig_win_rate_by_type(summary)
        fig_dimension_breakdown(summary)
    else:
        print("  ⚠ Skipping win rate charts (no summary data)")

    if name_analysis:
        fig_entity_fragmentation(name_analysis)
    else:
        print("  ⚠ Skipping entity fragmentation chart (no name data)")

    fig_graph_topology()

    print()
    print(f"  ✓ All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
