#!/usr/bin/env python3
"""
Master Orchestrator: run_all.py
================================
Runs the entire microstudy pipeline in sequence.

Usage:
    python experiments/run_all.py              # run everything
    python experiments/run_all.py --from 3     # resume from step 3
    python experiments/run_all.py --only 6.5   # run only step 6.5

Steps:
    1   - Prepare dataset
    2   - Build graphs (CORE — takes 1-3 hours)
    3   - Compute structural metrics
    5   - Run LLM evaluation (30-60 min)
    6   - Analyze results
    6.5 - Generate figures
    7   - Generate report
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Add project root to path
sys.path.insert(0, PROJECT_DIR)


def check_disk_space(min_gb: float = 2.0) -> bool:
    """Check if sufficient disk space is available."""
    try:
        total, used, free = shutil.disk_usage(PROJECT_DIR)
        free_gb = free / (1024 ** 3)
        if free_gb < min_gb:
            print(f"  ⚠ Low disk space: {free_gb:.1f}GB free (need {min_gb}GB)")
            return False
        print(f"  ✓ Disk space: {free_gb:.1f}GB free")
        return True
    except Exception:
        print("  ⚠ Could not check disk space")
        return True


def check_ollama() -> bool:
    """Check if Ollama is running and has required models."""
    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            print(f"  ✓ Ollama responding with {len(models)} models")
            return True
    except Exception as e:
        print(f"  ✗ Ollama not reachable at {host}: {e}")
        print(f"    Start with: ollama serve")
        return False


def check_imports() -> bool:
    """Validate all required Python imports."""
    required = [
        ("lightrag", "LightRAG"),
        ("lightrag.llm.ollama", "ollama_model_complete"),
        ("networkx", None),
        ("matplotlib", None),
        ("numpy", None),
        ("pandas", None),
        ("dotenv", None),
    ]

    ok = True
    for module, attr in required:
        try:
            mod = importlib.import_module(module)
            if attr:
                getattr(mod, attr)
            print(f"  ✓ {module}")
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {module}: {e}")
            ok = False

    return ok


def run_step(step_num: str, script_name: str, description: str) -> tuple:
    """Run a single step and return (success, elapsed_time)."""
    print()
    print(f"{'=' * 65}")
    print(f"  STEP {step_num}: {description}")
    print(f"{'=' * 65}")

    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  ✗ Script not found: {script_path}")
        return False, 0

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_DIR,
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            return True, elapsed
        else:
            print(f"\n  ✗ Step {step_num} failed (exit code {result.returncode}) after {elapsed:.1f}s")
            return False, elapsed

    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(f"\n  ⚠ Step {step_num} interrupted by user after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ Step {step_num} error: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full microstudy pipeline")
    parser.add_argument("--from", dest="from_step", type=str, default=None,
                       help="Resume from this step (e.g., --from 3)")
    parser.add_argument("--only", type=str, default=None,
                       help="Run only this step (e.g., --only 6.5)")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="Auto-answer yes to prompts (for unattended runs)")
    args = parser.parse_args()

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(PROJECT_DIR, ".env"), override=False)
    except ImportError:
        print("  ⚠ python-dotenv not installed, using system env vars")

    print()
    print("╔" + "═" * 63 + "╗")
    print("║  LightRAG Incremental Graph Degradation — Microstudy Runner  ║")
    print("╚" + "═" * 63 + "╝")
    print()

    # Pre-flight checks
    print("PRE-FLIGHT CHECKS:")
    print("-" * 40)

    checks_ok = True
    checks_ok &= check_disk_space()
    checks_ok &= check_ollama()
    checks_ok &= check_imports()

    if not checks_ok:
        print()
        print("  ⚠ Some pre-flight checks failed.")
        print("    Run 'bash experiments/setup_environment.sh' to fix.")
        print()
        if args.yes:
            print("  --yes flag set, continuing anyway...")
        else:
            response = input("  Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                print("  Aborted.")
                sys.exit(1)

    # Define steps
    steps = [
        ("1", "prepare_dataset.py", "Prepare Dataset"),
        ("2", "build_graphs.py", "Build Full & Incremental Graphs"),
        ("3", "graph_metrics.py", "Compute Structural Metrics"),
        ("5", "run_evaluation.py", "Run LLM Judge Evaluation"),
        ("6", "analyze_results.py", "Analyze Results"),
        ("6.5", "generate_figures.py", "Generate Visualizations"),
        ("7", "generate_report.py", "Generate Final Report"),
    ]

    # Filter steps based on args
    if args.only:
        steps = [s for s in steps if s[0] == args.only]
        if not steps:
            print(f"  ✗ Unknown step: {args.only}")
            sys.exit(1)

    if args.from_step:
        step_order = [s[0] for s in steps]
        if args.from_step not in step_order:
            print(f"  ✗ Unknown step: {args.from_step}")
            sys.exit(1)
        idx = step_order.index(args.from_step)
        steps = steps[idx:]

    # Run
    overall_start = time.time()
    results = []

    for step_num, script, description in steps:
        success, elapsed = run_step(step_num, script, description)
        results.append((step_num, description, success, elapsed))

        # Don't stop on non-critical failures (metrics, figures, report)
        if not success and step_num in ("1", "2"):
            print(f"\n  ✗ Critical step {step_num} failed. Stopping pipeline.")
            break

    # Final summary
    overall_elapsed = time.time() - overall_start
    print()
    print("╔" + "═" * 63 + "╗")
    print("║  PIPELINE SUMMARY                                           ║")
    print("╚" + "═" * 63 + "╝")
    print()
    print(f"  {'Step':<8s} {'Description':<35s} {'Status':<8s} {'Time':>10s}")
    print(f"  {'-' * 65}")

    for step_num, desc, success, elapsed in results:
        status = "✓ Pass" if success else "✗ Fail"
        time_s = f"{elapsed:.0f}s" if elapsed < 120 else f"{elapsed/60:.1f}m"
        print(f"  {step_num:<8s} {desc:<35s} {status:<8s} {time_s:>10s}")

    print(f"  {'-' * 65}")
    total_s = f"{overall_elapsed:.0f}s" if overall_elapsed < 120 else f"{overall_elapsed/60:.1f}m"
    print(f"  {'TOTAL':<8s} {'':35s} {'':8s} {total_s:>10s}")

    # Point to deliverable
    report_path = os.path.join(RESULTS_DIR, "REPORT.md")
    if os.path.exists(report_path):
        print()
        print(f"  📄 Report: {report_path}")
        print(f"  📊 Figures: {os.path.join(RESULTS_DIR, 'figures')}")
    print()


if __name__ == "__main__":
    main()
