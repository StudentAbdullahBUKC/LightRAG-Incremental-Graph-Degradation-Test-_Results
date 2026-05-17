#!/usr/bin/env python3
"""
Step 3: Structural Graph Metrics
==================================
Compares the full-build and incremental-build graphs on structural health metrics.

Usage:
    python experiments/graph_metrics.py

Outputs:
    graphs/structural_metrics.json     — raw metric values
    graphs/entity_name_analysis.json   — fuzzy duplicate entity analysis
"""

import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher

import networkx as nx

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GRAPHS_DIR = os.path.join(PROJECT_DIR, "graphs")


def load_lightrag_graph(working_dir: str) -> nx.Graph:
    """Load the LightRAG GraphML file."""
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(graphml_path):
        print(f"  ✗ Graph file not found: {graphml_path}")
        print(f"    Run 'python experiments/build_graphs.py' first.")
        sys.exit(1)

    G = nx.read_graphml(graphml_path)

    if G.number_of_nodes() == 0:
        print(f"  ⚠ WARNING: Graph has 0 nodes: {graphml_path}")

    return G


def compute_metrics(G: nx.Graph, label: str) -> dict:
    """Compute comprehensive structural metrics for a graph."""
    metrics = {"label": label}

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    metrics["num_nodes"] = n_nodes
    metrics["num_edges"] = n_edges
    metrics["density"] = round(nx.density(G), 6) if n_nodes > 1 else 0.0

    # Connected components
    if n_nodes > 0:
        # Handle both directed and undirected graphs
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G

        components = list(nx.connected_components(G_undirected))
        metrics["num_components"] = len(components)
        component_sizes = sorted([len(c) for c in components], reverse=True)
        metrics["largest_component_size"] = component_sizes[0] if component_sizes else 0
        metrics["largest_component_pct"] = round(
            component_sizes[0] / n_nodes * 100, 1
        ) if n_nodes > 0 else 0.0
        metrics["isolated_nodes"] = sum(1 for c in components if len(c) == 1)
        metrics["small_components"] = sum(1 for c in components if len(c) <= 3)
    else:
        metrics["num_components"] = 0
        metrics["largest_component_size"] = 0
        metrics["largest_component_pct"] = 0.0
        metrics["isolated_nodes"] = 0
        metrics["small_components"] = 0

    # Degree distribution
    if n_nodes > 0:
        if G.is_directed():
            degrees = [d for _, d in G.degree()]
        else:
            degrees = [d for _, d in G.degree()]
        metrics["avg_degree"] = round(sum(degrees) / len(degrees), 3)
        metrics["max_degree"] = max(degrees)
        metrics["min_degree"] = min(degrees)
        metrics["median_degree"] = sorted(degrees)[len(degrees) // 2]
        variance = sum((d - metrics["avg_degree"]) ** 2 for d in degrees) / len(degrees)
        metrics["degree_std"] = round(variance ** 0.5, 3)
    else:
        metrics["avg_degree"] = 0
        metrics["max_degree"] = 0
        metrics["min_degree"] = 0
        metrics["median_degree"] = 0
        metrics["degree_std"] = 0

    # Clustering coefficient
    try:
        if G.is_directed():
            metrics["avg_clustering"] = round(
                nx.average_clustering(G.to_undirected()), 4
            )
        else:
            metrics["avg_clustering"] = round(nx.average_clustering(G), 4)
    except Exception:
        metrics["avg_clustering"] = 0.0

    # Entity type distribution
    type_counter = Counter()
    for node_id in G.nodes():
        etype = G.nodes[node_id].get("entity_type", "UNKNOWN")
        type_counter[etype] += 1
    metrics["entity_type_distribution"] = dict(type_counter.most_common(10))

    return metrics


def compare_metrics(m_full: dict, m_inc: dict) -> dict:
    """Compare metrics and compute deltas."""
    key_metrics = [
        "num_nodes", "num_edges", "num_components",
        "largest_component_size", "largest_component_pct",
        "isolated_nodes", "small_components",
        "avg_degree", "density", "avg_clustering",
    ]

    deltas = {}
    print(f"\n{'=' * 65}")
    print(f"  {'METRIC':<30s} {'FULL':>8s} {'INC':>8s} {'DELTA':>10s} {'%':>8s}")
    print(f"  {'-' * 62}")

    for k in key_metrics:
        v_full = m_full.get(k, 0)
        v_inc = m_inc.get(k, 0)
        delta = v_inc - v_full if isinstance(v_full, (int, float)) else 0
        pct = (delta / v_full * 100) if v_full != 0 else 0

        # Format values
        if isinstance(v_full, float):
            full_s = f"{v_full:.4f}"
            inc_s = f"{v_inc:.4f}"
            delta_s = f"{delta:+.4f}"
        else:
            full_s = f"{v_full}"
            inc_s = f"{v_inc}"
            delta_s = f"{delta:+d}" if isinstance(delta, int) else f"{delta:+.1f}"

        flag = " ⚠️" if abs(pct) > 10 else ""
        print(f"  {k:<30s} {full_s:>8s} {inc_s:>8s} {delta_s:>10s} {pct:>+7.1f}%{flag}")

        deltas[k] = {
            "full": v_full,
            "inc": v_inc,
            "delta": round(delta, 4) if isinstance(delta, float) else delta,
            "pct": round(pct, 2),
            "flagged": abs(pct) > 10,
        }

    print(f"  {'-' * 62}")
    return deltas


def analyze_entity_names(G_full: nx.Graph, G_inc: nx.Graph) -> dict:
    """Fuzzy-match entity names between graphs to detect fragmentation."""
    full_nodes = set(G_full.nodes())
    inc_nodes = set(G_inc.nodes())

    # Nodes unique to each graph
    only_full = full_nodes - inc_nodes
    only_inc = inc_nodes - full_nodes
    common = full_nodes & inc_nodes

    print(f"\n{'=' * 65}")
    print(f"  ENTITY NAME ANALYSIS")
    print(f"  {'-' * 62}")
    print(f"  {'Common entities:':<35s} {len(common):>8d}")
    print(f"  {'Only in Full graph:':<35s} {len(only_full):>8d}")
    print(f"  {'Only in Incremental graph:':<35s} {len(only_inc):>8d}")
    print(f"  {'Jaccard similarity:':<35s} {len(common) / len(full_nodes | inc_nodes) * 100:>7.1f}%")

    # Find fuzzy matches: entities that exist in one graph but have a similar
    # name in the other — potential alias fragmentation
    fuzzy_matches = []

    # Check inc-only nodes against full-only nodes
    for inc_name in sorted(only_inc):
        best_match = None
        best_ratio = 0
        for full_name in only_full:
            ratio = SequenceMatcher(None, inc_name.upper(), full_name.upper()).ratio()
            if ratio > best_ratio and ratio > 0.6:
                best_ratio = ratio
                best_match = full_name
        if best_match:
            fuzzy_matches.append({
                "inc_name": inc_name,
                "full_name": best_match,
                "similarity": round(best_ratio, 3),
            })

    # Also check for substring matches
    for inc_name in sorted(only_inc):
        for full_name in only_full:
            if (inc_name.upper() in full_name.upper() or
                    full_name.upper() in inc_name.upper()):
                # Avoid duplicates with fuzzy matches
                if not any(m["inc_name"] == inc_name and m["full_name"] == full_name
                           for m in fuzzy_matches):
                    fuzzy_matches.append({
                        "inc_name": inc_name,
                        "full_name": full_name,
                        "similarity": -1.0,  # substring match marker
                    })

    # Sort by similarity descending
    fuzzy_matches.sort(key=lambda x: x["similarity"], reverse=True)

    if fuzzy_matches:
        print(f"\n  Potential alias fragmentation ({len(fuzzy_matches)} candidates):")
        for m in fuzzy_matches[:15]:
            sim_s = f"{m['similarity']:.0%}" if m["similarity"] > 0 else "substr"
            print(f"    [{sim_s:>6s}] INC: '{m['inc_name']}' ↔ FULL: '{m['full_name']}'")
        if len(fuzzy_matches) > 15:
            print(f"    ... and {len(fuzzy_matches) - 15} more")
    else:
        print(f"\n  No obvious alias fragmentation detected.")

    analysis = {
        "common_count": len(common),
        "only_full_count": len(only_full),
        "only_inc_count": len(only_inc),
        "jaccard_similarity": round(
            len(common) / max(len(full_nodes | inc_nodes), 1) * 100, 2
        ),
        "only_full": sorted(list(only_full))[:50],
        "only_inc": sorted(list(only_inc))[:50],
        "fuzzy_matches": fuzzy_matches[:30],
    }

    return analysis


def main():
    print("=" * 65)
    print("  Step 3: Structural Graph Metrics")
    print("=" * 65)

    # Load graphs
    print("\n  Loading graphs...")
    full_dir = os.path.join(GRAPHS_DIR, "graph_full")
    inc_dir = os.path.join(GRAPHS_DIR, "graph_inc")

    G_full = load_lightrag_graph(full_dir)
    G_inc = load_lightrag_graph(inc_dir)

    print(f"  ✓ Full graph:        {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    print(f"  ✓ Incremental graph: {G_inc.number_of_nodes()} nodes, {G_inc.number_of_edges()} edges")

    # Compute individual metrics
    m_full = compute_metrics(G_full, "Full (Control)")
    m_inc = compute_metrics(G_inc, "Incremental (Test)")

    # Compare
    deltas = compare_metrics(m_full, m_inc)

    # Entity name analysis
    name_analysis = analyze_entity_names(G_full, G_inc)

    # Save results
    results = {
        "full": m_full,
        "inc": m_inc,
        "deltas": deltas,
    }

    metrics_path = os.path.join(GRAPHS_DIR, "structural_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✓ Metrics saved to: {metrics_path}")

    names_path = os.path.join(GRAPHS_DIR, "entity_name_analysis.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(name_analysis, f, indent=2)
    print(f"  ✓ Entity analysis saved to: {names_path}")

    # Print interpretation
    flagged = [k for k, v in deltas.items() if v.get("flagged")]
    print()
    if flagged:
        print(f"  ⚠ {len(flagged)} metrics show >10% divergence: {', '.join(flagged)}")
        print(f"    → Evidence of structural degradation in incremental build.")
    else:
        print(f"  ✓ No metrics show >10% divergence.")
        print(f"    → Incremental build appears structurally similar to full build.")

    return results, name_analysis


if __name__ == "__main__":
    main()
