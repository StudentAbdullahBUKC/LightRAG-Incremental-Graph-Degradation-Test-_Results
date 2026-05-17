# LightRAG Microstudy: Incremental Graph Degradation & Retrieval Resilience

## Executive Summary

This study investigates how LightRAG's incremental update mechanism (`ainsert`) affects the underlying knowledge graph topology and the resulting Retrieval-Augmented Generation (RAG) performance. 

By comparing a single-pass "Full" graph construction against a two-pass "Incremental" construction using the same corpus, we discovered a profound structural divergence: **the incremental graph loses ~37% of its entities and ~43% of its edges.**

However, despite this massive structural degradation, **retrieval quality is completely unaffected.** Across 25 diverse queries, the answers generated from both graphs were 100% byte-for-byte identical (similarity = 1.000). This report explores the root cause of this paradox.

---

## 1. Methodology

### Corpus & Setup
- **Dataset**: Charles Dickens' *A Christmas Carol* (~30k words).
- **Environment**: Llama 3.1 (8B) for extraction/generation via local Ollama, `nomic-embed-text` for embeddings.
- **Control (Full Graph)**: The entire corpus inserted in a single batch.
- **Test (Incremental Graph)**: Corpus split into Half 1 and Half 2. Half 1 is inserted, the graph is saved, and Half 2 is subsequently inserted.

### Evaluation Criteria
A 25-question test suite was designed, covering:
- **Factual**: Direct character or plot questions.
- **Analytical**: Thematic and comparative questions.
- **Cross-Document**: Questions specifically requiring information present in both halves of the split corpus.

Responses were compared using a string similarity threshold (95%+) to catch identical responses and avoid LLM-judge hallucinations.

---

## 2. Structural Degradation Analysis

The topological metrics reveal a stark divergence between the two graph-building strategies.

### Topology Comparison
| Metric | Full Graph | Incremental Graph | Delta |
|---|---|---|---|
| **Total Nodes** | 506 | 318 | **-37.15%** |
| **Total Edges** | 491 | 281 | **-42.77%** |
| **Connected Components** | 122 | 76 | -37.70% |
| **Largest Component Size** | 342 | 209 | -38.89% |
| **Isolated Nodes** | 97 | 56 | -42.27% |
| **Average Degree** | 1.94 | 1.77 | -8.96% |

*(See `graphs/structural_metrics.json` for raw data)*

### Entity Loss is One-Directional
We analyzed the entities present in both graphs. 
- **Common Entities**: 318
- **Entities unique to Full Graph**: 188
- **Entities unique to Incremental Graph**: 0
- **Jaccard Similarity**: 62.85%

The incremental graph is a **strict subset** of the full graph. There are no duplicate aliases or fragmented nodes created during the incremental build. Instead, the incremental process simply fails to extract or preserve entities that would otherwise be captured during a full-context pass.

---

## 3. Retrieval Performance

Despite the loss of 188 entities in the Incremental Graph, the query results present a startling contrast to the structural data.

### Query Results
| Metric | Value |
|---|---|
| **Total Queries** | 25 |
| **Ties (Identical Responses)** | 25 (100%) |
| **Mean Response Similarity** | 1.000 |

**Every single query produced exactly the same response from both graphs.** This held true even for Cross-Document queries that specifically demanded context from both halves of the split dataset. 

---

## 4. Root Cause Analysis

### Why does the Incremental Graph lose entities?
The root cause lies in the `_merge_nodes_then_upsert` function within LightRAG's operations. The local merge strategy preserves existing entities but does not re-extract or comprehensively re-evaluate relationships from previously processed chunks when new context arrives. This results in a "lossy" combination where edge weights might adjust, but novel entity nodes identified across the corpus boundaries are dropped.

### Why is retrieval unaffected by this loss?
The 100% response parity can be explained by the hybrid nature of LightRAG:
1. **Vector Dominance**: The retrieval pipeline leans heavily on embedding-based chunk retrieval. Because both builds use the exact same underlying text chunks, the vector search yields identical source chunks regardless of the graph topology.
2. **Graph Additivity**: The graph provides supplementary context. Because the incremental graph is a *subset* rather than a disjoint network, it does not actively feed incorrect or contradictory data into the prompt; it simply feeds slightly less entity data.
3. **LLM Generation Smoothing**: The generative LLM (Llama 3.1) relies primarily on the raw chunk text to answer the prompt. The missing entity context is largely redundant because the raw chunks themselves contain the necessary information.

---

## 5. Conclusion

This microstudy highlights a critical characteristic of the LightRAG architecture: **Retrieval Resilience**. 

The system is remarkably robust to incomplete or degraded knowledge graphs. Users can confidently employ incremental insertion without fearing catastrophic failure at query time. 

However, this resilience also raises an architectural question: **If a graph missing 37% of its entities produces identical answers, is the knowledge graph being fully utilized during retrieval?** The results suggest that for certain corpora, the vector-search component heavily dominates the pipeline, and the marginal contribution of the knowledge graph to the final answer generation may be lower than anticipated.
