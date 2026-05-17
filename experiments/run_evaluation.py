#!/usr/bin/env python3
"""
Step 5: V2 Evaluation (Similarity Threshold)
=======================================
Queries both graphs with the same questions and checks for 
exact or near-exact response similarity. Overcomes LLM judge
hallucinations by enforcing a hard 95% similarity threshold.

Usage:
    python experiments/run_evaluation.py

Outputs:
    results/raw_results.json
"""

import asyncio
import json
import os
import random
import sys
import time
from functools import partial
import difflib

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(PROJECT_DIR, ".env"), override=False)

GRAPHS_DIR = os.path.join(PROJECT_DIR, "graphs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
QUERIES_PATH = os.path.join(SCRIPT_DIR, "queries.json")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:7b")
LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
LLM_TIMEOUT = int(os.getenv("TIMEOUT", "600"))
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
EMBED_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
MAX_ASYNC = int(os.getenv("MAX_ASYNC", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
SIMILARITY_THRESHOLD = 0.95

async def create_rag_instance(working_dir: str) -> "LightRAG":
    """Create a LightRAG instance for querying."""
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

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
        summary_max_tokens=4096,
        summary_context_size=4096,
        llm_model_max_async=MAX_ASYNC,
    )
    await rag.initialize_storages()
    return rag

async def query_rag(rag: "LightRAG", query: str) -> str:
    from lightrag import QueryParam
    try:
        result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        return str(result) if result else "[No response generated]"
    except Exception as e:
        return f"[Query error: {e}]"

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    matcher = difflib.SequenceMatcher(None, text1.strip(), text2.strip())
    return matcher.ratio()

async def main():
    print("=" * 60)
    print("  Step 5: V2 Evaluation (Similarity Threshold)")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(QUERIES_PATH):
        print(f"  ✗ Queries not found: {QUERIES_PATH}")
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  Loaded {len(queries)} queries")

    print("  Loading Full graph...")
    rag_full = await create_rag_instance(os.path.join(GRAPHS_DIR, "graph_full"))
    print("  Loading Incremental graph...")
    rag_inc = await create_rag_instance(os.path.join(GRAPHS_DIR, "graph_inc"))
    print("  ✓ Both graphs loaded\n")

    results_path = os.path.join(RESULTS_DIR, "raw_results.json")
    results = []
    overall_start = time.time()

    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q['id']}: {q['query'][:55]}...")
        
        t0 = time.time()
        resp_full = await query_rag(rag_full, q["query"])
        resp_inc = await query_rag(rag_inc, q["query"])
        query_time = time.time() - t0

        similarity = calculate_similarity(resp_full, resp_inc)
        
        if similarity >= SIMILARITY_THRESHOLD:
            winner = "Tie"
            judgment = {"reason": f"Auto-tied due to high similarity ({similarity:.3f})"}
            status = f"✓ Auto-Tie (Sim: {similarity:.3f})"
        else:
            # Fallback placeholder if actual differences emerge in future sets
            winner = "Needs LLM"
            judgment = {"reason": "Significant divergence, requires manual/LLM review."}
            status = f"⚠ Diverged (Sim: {similarity:.3f})"
            
        print(f"    {status} in {query_time:.1f}s")
        
        results.append({
            "id": q["id"],
            "query": q["query"],
            "type": q.get("type", "unknown"),
            "requires_both_halves": q.get("requires_both_halves", False),
            "response_full": resp_full,
            "response_inc": resp_inc,
            "response_similarity": similarity,
            "judgment": judgment,
            "winner": winner,
            "parsed": True,
            "query_time_s": round(query_time, 1),
            "judge_time_s": 0.0,
        })
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        await rag_full.finalize_storages()
        await rag_inc.finalize_storages()
    except Exception:
        pass

    elapsed = time.time() - overall_start
    winners = [r["winner"] for r in results]
    
    print(f"\n  {'=' * 55}")
    print(f"  EVALUATION SUMMARY")
    print(f"  {'=' * 55}")
    print(f"  Total queries:    {len(results)}")
    print(f"  Average Sim:      {sum(r['response_similarity'] for r in results) / len(results):.3f}")
    print(f"  Ties (Sim > {SIMILARITY_THRESHOLD}): {winners.count('Tie')}")
    print(f"  Total time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to: {results_path}")

if __name__ == "__main__":
    asyncio.run(main())
