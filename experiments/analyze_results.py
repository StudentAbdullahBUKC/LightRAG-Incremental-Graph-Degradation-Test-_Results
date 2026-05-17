#!/usr/bin/env python3
"""
Step 6: Results Analysis
=========================
Aggregates win rates from the LLM judge output.

Usage:
    python experiments/analyze_results.py

Outputs:
    results/analysis_summary.json — aggregated statistics
"""

import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "raw_results.json")


def tally_results(results: list, label: str) -> dict:
    """Compute win rates for a subset of results."""
    n = len(results)
    if n == 0:
        return {"n": 0, "full_wins": 0, "inc_wins": 0, "ties": 0}

    wins = defaultdict(int)
    dims = ["comprehensiveness", "factual_specificity", "coherence"]
    dim_wins = {d: defaultdict(int) for d in dims}

    for r in results:
        winner = r.get("winner", "Tie")
        wins[winner] += 1

        judgment = r.get("judgment", {})
        label_order = r.get("label_order", "full_is_A")

        for dim in dims:
            if dim in judgment and isinstance(judgment[dim], dict):
                raw_w = judgment[dim].get("winner", "Tie")
                if raw_w == "Tie":
                    dim_wins[dim]["Tie"] += 1
                elif label_order == "full_is_A":
                    dim_wins[dim]["full" if raw_w == "A" else "inc"] += 1
                else:
                    dim_wins[dim]["inc" if raw_w == "A" else "full"] += 1

    # Print table
    print(f"\n  --- {label} (n={n}) ---")
    print(f"  {'Overall Win Rate:'}")
    print(f"    Full wins:   {wins['full']:>3d}/{n} = {wins['full']/n*100:>5.1f}%")
    print(f"    Inc wins:    {wins['inc']:>3d}/{n} = {wins['inc']/n*100:>5.1f}%")
    print(f"    Ties:        {wins['Tie']:>3d}/{n} = {wins['Tie']/n*100:>5.1f}%")

    print(f"  {'Per-Dimension Breakdown:'}")
    dim_data = {}
    for dim in dims:
        dw = dim_wins[dim]
        dn = sum(dw.values())
        if dn > 0:
            full_pct = dw["full"] / dn * 100
            inc_pct = dw["inc"] / dn * 100
            tie_pct = dw["Tie"] / dn * 100
            print(f"    {dim:25s}: full={full_pct:>4.0f}%  inc={inc_pct:>4.0f}%  tie={tie_pct:>4.0f}%")
            dim_data[dim] = {
                "full": dw["full"], "inc": dw["inc"], "tie": dw["Tie"],
                "full_pct": round(full_pct, 1),
                "inc_pct": round(inc_pct, 1),
                "tie_pct": round(tie_pct, 1),
            }

    return {
        "label": label,
        "n": n,
        "full_wins": wins["full"],
        "inc_wins": wins["inc"],
        "ties": wins["Tie"],
        "full_pct": round(wins["full"] / n * 100, 1),
        "inc_pct": round(wins["inc"] / n * 100, 1),
        "tie_pct": round(wins["Tie"] / n * 100, 1),
        "per_dimension": dim_data,
    }


def main():
    print("=" * 60)
    print("  Step 6: Results Analysis")
    print("=" * 60)

    if not os.path.exists(RESULTS_PATH):
        print(f"\n  ✗ Results not found: {RESULTS_PATH}")
        print(f"    Run 'python experiments/run_evaluation.py' first.")
        sys.exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Filter valid results
    valid = [r for r in results if r.get("parsed", False)]
    errors = [r for r in results if not r.get("parsed", False)]

    print(f"\n  Total results: {len(results)} | Valid: {len(valid)} | Parse errors: {len(errors)}")

    if len(valid) == 0:
        print("  ✗ No valid results to analyze.")
        return

    # All queries
    summary_all = tally_results(valid, "ALL QUERIES")

    # Cross-document queries (primary signal)
    cross_doc = [r for r in valid if r.get("requires_both_halves", False)]
    single_doc = [r for r in valid if not r.get("requires_both_halves", False)]

    summary_cross = tally_results(cross_doc, "CROSS-DOCUMENT QUERIES (requires both halves)")
    summary_single = tally_results(single_doc, "SINGLE-DOCUMENT QUERIES")

    # By query type
    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("type", "unknown")].append(r)

    type_summaries = {}
    for qtype, subset in sorted(by_type.items()):
        type_summaries[qtype] = tally_results(subset, f"TYPE: {qtype}")

    # Save summary
    summary = {
        "total_results": len(results),
        "valid_results": len(valid),
        "parse_errors": len(errors),
        "all_queries": summary_all,
        "cross_document": summary_cross,
        "single_document": summary_single,
        "by_type": type_summaries,
        "error_ids": [r["id"] for r in errors],
    }

    summary_path = os.path.join(RESULTS_DIR, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ Analysis saved to: {summary_path}")

    # Interpretation
    print()
    if summary_all["full_pct"] > summary_all["inc_pct"] + 15:
        print("  📊 FINDING: Full build significantly outperforms incremental build.")
        print("     → Evidence of degradation from incremental graph updates.")
    elif summary_all["inc_pct"] > summary_all["full_pct"] + 15:
        print("  📊 FINDING: Incremental build outperforms full build (unexpected).")
        print("     → The incremental merge may provide beneficial deduplication.")
    else:
        print("  📊 FINDING: Full and incremental builds perform comparably.")
        print("     → LightRAG's incremental update mechanism appears robust.")

    if summary_cross["n"] > 0 and summary_cross["full_pct"] > summary_cross["inc_pct"] + 20:
        print("  ⚠ Cross-document queries show significant degradation!")
        print("     → Graph fragmentation likely impacts multi-hop reasoning.")

    return summary


if __name__ == "__main__":
    main()
