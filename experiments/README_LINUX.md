# LightRAG Microstudy — Quick Start (Linux + RTX 4090)

## What This Does
Tests whether LightRAG's incremental graph update degrades retrieval quality
compared to building the graph from scratch. Everything runs locally on your GPU.

**Cost: $0 | Runtime: ~2-4 hours | Fully automated**

---

## Requirements
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA RTX 4090 (or any GPU with 8GB+ VRAM)
- Python 3.10+
- ~10GB free disk space
- Internet (only for first-time model downloads)

---

## Step 1: Fix line endings (IMPORTANT — files came from Windows)

```bash
cd LightRAG

# Fix Windows CRLF line endings
sed -i 's/\r$//' experiments/*.sh experiments/*.py experiments/*.json .env
chmod +x experiments/setup_environment.sh
```

## Step 2: Run setup (one-time, ~10 min)

```bash
bash experiments/setup_environment.sh
```

This will:
- Install Ollama (if not installed)
- Download 2 AI models (~5GB total)
- Create a Python virtual environment
- Install all dependencies
- Run smoke tests to verify GPU works

## Step 3: Run the experiment (~2-4 hours)

```bash
source venv/bin/activate
python3 experiments/run_all.py
```

You'll see progress in the terminal. You can safely leave it running.

**If it crashes or you need to stop:**
Just re-run the same command — it picks up where it left off.

**To skip the graph building (if already done):**
```bash
python3 experiments/run_all.py --from 3
```

## Step 4: Send back the results

When it's done, zip and send me these 2 folders:
```
results/     (contains REPORT.md + PNG charts + JSON data)
graphs/      (contains structural_metrics.json + entity_name_analysis.json)
```

```bash
tar -czf microstudy_results.tar.gz results/ graphs/
```

---

## Troubleshooting

**"Ollama not found"**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

**"Cannot reach Ollama server"**
```bash
# Start Ollama in another terminal
ollama serve
```

**GPU out of memory**
This shouldn't happen (Llama 7B only uses ~5GB of your 24GB),
but if it does, close other GPU-intensive programs.

**"Python not found"**
```bash
sudo apt update && sudo apt install python3 python3-venv python3-pip
```

---

That's it. Thanks!
