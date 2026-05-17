#!/usr/bin/env python3
"""
Step 2: Dual Graph Construction
=================================
Builds two LightRAG knowledge graphs from the same corpus:
  - graph_full: ingested from full.txt in a single pass (CONTROL)
  - graph_inc:  ingested from half_a.txt, then half_b.txt incrementally (TEST)

Usage:
    python experiments/build_graphs.py
    python experiments/build_graphs.py --skip-full    # skip full build (if already done)
    python experiments/build_graphs.py --skip-inc     # skip incremental build

Outputs:
    graphs/graph_full/   — full-build working directory
    graphs/graph_inc/    — incremental-build working directory
"""

import asyncio
import json
import os
import shutil
import sys
import time
from functools import partial

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Add project to path for imports
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(PROJECT_DIR, ".env"), override=False)

# --- Configuration (from .env or defaults) ---
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBED_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
LLM_TIMEOUT = int(os.getenv("TIMEOUT", "600"))
MAX_ASYNC = int(os.getenv("MAX_ASYNC", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))

SPLITS_DIR = os.path.join(PROJECT_DIR, "data", "splits")
GRAPHS_DIR = os.path.join(PROJECT_DIR, "graphs")
CHECKPOINT_DIR = os.path.join(GRAPHS_DIR, ".checkpoints")


def check_ollama_health() -> bool:
    """Verify Ollama server is reachable."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{LLM_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            print(f"  ✓ Ollama server responding. Available models: {len(models)}")

            # Check required models
            llm_found = any(LLM_MODEL.split(":")[0] in m for m in models)
            embed_found = any(EMBED_MODEL.split(":")[0] in m for m in models)

            if not llm_found:
                print(f"  ✗ LLM model '{LLM_MODEL}' not found! Run: ollama pull {LLM_MODEL}")
                return False
            if not embed_found:
                print(f"  ✗ Embedding model '{EMBED_MODEL}' not found! Run: ollama pull {EMBED_MODEL}")
                return False

            print(f"  ✓ Required models found: {LLM_MODEL}, {EMBED_MODEL}")
            return True
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"  ✗ Cannot reach Ollama at {LLM_HOST}: {e}")
        print(f"    Start Ollama with: ollama serve")
        return False


def load_text(path: str) -> str:
    """Load text file with validation."""
    if not os.path.exists(path):
        print(f"  ✗ File not found: {path}")
        print(f"    Run 'python experiments/prepare_dataset.py' first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) < 100:
        print(f"  ✗ File too small ({len(text)} chars): {path}")
        sys.exit(1)
    return text


async def create_rag_instance(working_dir: str) -> "LightRAG":
    """Create a LightRAG instance with Ollama backend."""
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            "host": LLM_HOST,
            "options": {"num_ctx": 8192},
            "timeout": LLM_TIMEOUT,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=partial(
                ollama_embed.func,
                embed_model=EMBED_MODEL,
                host=EMBED_HOST,
            ),
        ),
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=100,
        summary_max_tokens=4096,
        summary_context_size=4096,
        llm_model_max_async=MAX_ASYNC,
        max_parallel_insert=1,
        entity_extract_max_gleaning=1,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )

    await rag.initialize_storages()
    return rag


async def test_embedding(rag: "LightRAG") -> bool:
    """Run a test embedding to catch dimension mismatches early."""
    try:
        test_texts = ["This is a test sentence for embedding validation."]
        result = await rag.embedding_func(test_texts)
        dim = result.shape[1]
        print(f"  ✓ Embedding test passed (dimension: {dim})")
        if dim != EMBED_DIM:
            print(f"  ⚠ WARNING: Expected {EMBED_DIM} dimensions, got {dim}")
            print(f"    Update EMBEDDING_DIM in .env to {dim}")
        return True
    except Exception as e:
        print(f"  ✗ Embedding test failed: {e}")
        return False


def load_checkpoint(label: str) -> int:
    """Load checkpoint — returns number of chars already ingested."""
    cp_file = os.path.join(CHECKPOINT_DIR, f"{label}.json")
    if os.path.exists(cp_file):
        with open(cp_file, "r") as f:
            data = json.load(f)
            return data.get("chars_ingested", 0)
    return 0


def save_checkpoint(label: str, chars_ingested: int) -> None:
    """Save checkpoint for crash recovery."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cp_file = os.path.join(CHECKPOINT_DIR, f"{label}.json")
    with open(cp_file, "w") as f:
        json.dump({
            "label": label,
            "chars_ingested": chars_ingested,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


def clear_checkpoint(label: str) -> None:
    """Remove checkpoint after successful completion."""
    cp_file = os.path.join(CHECKPOINT_DIR, f"{label}.json")
    if os.path.exists(cp_file):
        os.remove(cp_file)


async def build_full_graph() -> None:
    """Pipeline 1: Full build (control) — ingest everything in one pass."""
    label = "graph_full"
    working_dir = os.path.join(GRAPHS_DIR, label)

    print()
    print("=" * 60)
    print(f"  PIPELINE 1: FULL BUILD (Control)")
    print("=" * 60)

    # Check if already completed
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graphml_path) and os.path.getsize(graphml_path) > 100:
        print(f"  ⚠ Graph already exists at {working_dir}")
        print(f"    Delete the directory to rebuild, or use --skip-full")
        return

    text = load_text(os.path.join(SPLITS_DIR, "full.txt"))
    print(f"  Corpus: {len(text):,} chars (~{len(text)//4:,} tokens)")

    rag = await create_rag_instance(working_dir)

    if not await test_embedding(rag):
        print("  ✗ FATAL: Embedding test failed. Aborting full build.")
        await rag.finalize_storages()
        return

    print(f"  Starting ingestion...")
    start_time = time.time()

    try:
        await rag.ainsert(text)
        elapsed = time.time() - start_time
        print(f"  ✓ Full build complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        clear_checkpoint(label)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Full build failed after {elapsed:.1f}s: {e}")
        save_checkpoint(label, 0)
        raise
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass


async def build_incremental_graph() -> None:
    """Pipeline 2: Incremental build (test) — ingest A, then B separately."""
    label = "graph_inc"
    working_dir = os.path.join(GRAPHS_DIR, label)

    print()
    print("=" * 60)
    print(f"  PIPELINE 2: INCREMENTAL BUILD (Test)")
    print("=" * 60)

    # Check if already completed
    graphml_path = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graphml_path) and os.path.getsize(graphml_path) > 100:
        print(f"  ⚠ Graph already exists at {working_dir}")
        print(f"    Delete the directory to rebuild, or use --skip-inc")
        return

    text_a = load_text(os.path.join(SPLITS_DIR, "half_a.txt"))
    text_b = load_text(os.path.join(SPLITS_DIR, "half_b.txt"))

    print(f"  Half A: {len(text_a):,} chars (~{len(text_a)//4:,} tokens)")
    print(f"  Half B: {len(text_b):,} chars (~{len(text_b)//4:,} tokens)")

    # --- Phase 1: Ingest Half A ---
    print()
    print(f"  --- Phase 1: Ingesting Half A ---")
    rag = await create_rag_instance(working_dir)

    if not await test_embedding(rag):
        print("  ✗ FATAL: Embedding test failed. Aborting incremental build.")
        await rag.finalize_storages()
        return

    start_time = time.time()
    try:
        await rag.ainsert(text_a)
        elapsed = time.time() - start_time
        print(f"  ✓ Phase 1 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Phase 1 failed after {elapsed:.1f}s: {e}")
        save_checkpoint(label, 0)
        raise
    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass

    # --- Phase 2: Incrementally add Half B ---
    print()
    print(f"  --- Phase 2: Incrementally adding Half B ---")
    print(f"  Re-opening existing graph at {working_dir}...")

    # Re-instantiate on the SAME working_dir — this is the real-world incremental scenario
    rag_inc = await create_rag_instance(working_dir)

    start_time = time.time()
    try:
        await rag_inc.ainsert(text_b)
        elapsed = time.time() - start_time
        print(f"  ✓ Phase 2 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        clear_checkpoint(label)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Phase 2 failed after {elapsed:.1f}s: {e}")
        save_checkpoint(label, len(text_a))
        raise
    finally:
        try:
            await rag_inc.finalize_storages()
        except Exception:
            pass

    print()
    print(f"  ✓ Incremental build complete. Both phases finished.")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build full and incremental LightRAG graphs")
    parser.add_argument("--skip-full", action="store_true", help="Skip the full build")
    parser.add_argument("--skip-inc", action="store_true", help="Skip the incremental build")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 2: Dual Graph Construction")
    print("=" * 60)
    print()

    # Pre-flight checks
    print("[Pre-flight] Checking Ollama...")
    if not check_ollama_health():
        print("\n  FATAL: Ollama pre-flight check failed. Fix the issues above and retry.")
        sys.exit(1)

    print()
    print(f"  Configuration:")
    print(f"    LLM Model:        {LLM_MODEL}")
    print(f"    LLM Host:         {LLM_HOST}")
    print(f"    Embedding Model:  {EMBED_MODEL}")
    print(f"    Embedding Dim:    {EMBED_DIM}")
    print(f"    Chunk Size:       {CHUNK_SIZE} tokens")
    print(f"    Max Async:        {MAX_ASYNC}")
    print(f"    LLM Timeout:      {LLM_TIMEOUT}s")

    overall_start = time.time()

    if not args.skip_full:
        await build_full_graph()
    else:
        print("\n  [Skipped] Full build (--skip-full)")

    if not args.skip_inc:
        await build_incremental_graph()
    else:
        print("\n  [Skipped] Incremental build (--skip-inc)")

    overall_elapsed = time.time() - overall_start
    print()
    print("=" * 60)
    print(f"  ✓ All builds complete in {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
