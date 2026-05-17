# LightRAG Incremental Graph Degradation Microstudy

This repository contains the code and reproducible pipeline for an empirical microstudy on the **LightRAG** framework. The study investigates the structural and retrieval performance differences between building a knowledge graph in a single full pass versus an incremental, multi-pass construction.

## Key Findings

1. **Structural Degradation**: The incremental build results in a structurally degraded graph with **~37% fewer nodes and ~43% fewer edges** compared to the full build.
2. **Entity Loss is One-Directional**: The incremental graph is a strict subset of the full graph (zero unique entities, no alias fragmentation).
3. **Retrieval Resilience**: Despite massive structural loss, **retrieval quality remains completely unaffected**. Both graphs produced 100% identical responses across 25 test queries.

For a comprehensive breakdown, please refer to the `DETAILED_REPORT.md`.

## Repository Structure

```text
.
├── experiments/
│   ├── prepare_dataset.py   # Downloads and splits the test corpus
│   ├── build_graphs.py      # Constructs the 'full' and 'inc' graphs
│   ├── graph_metrics.py     # Analyzes graph topology and outputs structural_metrics.json
│   ├── run_evaluation.py    # Queries both graphs and evaluates using a local LLM judge
│   ├── analyze_results.py   # Aggregates evaluation data
│   ├── generate_figures.py  # Produces PNG charts of the results
│   └── run_all.py           # Automated pipeline script
├── graphs/                  # Generated knowledge graphs and metrics
├── results/                 # Evaluation results and output figures
├── DETAILED_REPORT.md       # Full analysis and methodology
└── README.md
```

## Quick Start (Linux / WSL2)

**Requirements**:
- Python 3.10+
- A local Ollama instance running `llama3.1:8b` (or `7b`) and `nomic-embed-text`
- ~16GB RAM / 10GB Disk Space

**1. Setup Environment**
```bash
cd experiments
chmod +x setup_environment.sh
./setup_environment.sh
```

**2. Run the Full Pipeline**
This will download the data, build both graphs, run the evaluation, and generate all figures.
```bash
python experiments/run_all.py
```

## Methodology

This study uses a controlled A/B test on Charles Dickens' *A Christmas Carol*. 
- **Control (Full)**: The entire document is processed and inserted into an empty LightRAG instance.
- **Test (Incremental)**: The document is split into two halves. Half 1 is inserted, the graph is saved, and then Half 2 is inserted (`ainsert`) into the existing working directory.

Both graphs are then queried using 25 hand-crafted questions. The answers are evaluated for semantic similarity to determine if the structural divergence impacts the final retrieval quality.
