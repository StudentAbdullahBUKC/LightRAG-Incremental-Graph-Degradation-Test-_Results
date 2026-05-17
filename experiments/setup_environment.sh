#!/usr/bin/env bash
#
# LightRAG Microstudy — Environment Setup (Linux)
# ==================================================
# Run this ONCE before running the experiment.
#
# IMPORTANT: If you transferred this from Windows, first run:
#   sed -i 's/\r$//' experiments/*.sh experiments/*.py experiments/*.json .env
#
# Then:   bash experiments/setup_environment.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "  LightRAG Microstudy — Environment Setup"
echo "=============================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# -----------------------------------------------------------
# 0. Fix Windows CRLF line endings on all experiment files
# -----------------------------------------------------------
echo "[Step 0/7] Fixing Windows line endings (CRLF → LF)..."
if command -v dos2unix &> /dev/null; then
    find "$SCRIPT_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.json" \) -exec dos2unix -q {} \;
    [ -f "$PROJECT_DIR/.env" ] && dos2unix -q "$PROJECT_DIR/.env"
    echo "  ✓ Line endings fixed (dos2unix)"
else
    # Fallback: use sed
    find "$SCRIPT_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.json" \) -exec sed -i 's/\r$//' {} \;
    [ -f "$PROJECT_DIR/.env" ] && sed -i 's/\r$//' "$PROJECT_DIR/.env"
    echo "  ✓ Line endings fixed (sed)"
fi

# -----------------------------------------------------------
# 1. Check Python
# -----------------------------------------------------------
echo ""
echo "[Step 1/7] Checking Python..."
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
    PY_VER=$($PY_CMD --version 2>&1)
    echo "  ✓ $PY_VER"
elif command -v python &> /dev/null; then
    PY_CMD="python"
    PY_VER=$($PY_CMD --version 2>&1)
    echo "  ✓ $PY_VER (using 'python' command)"
else
    echo "  ✗ FATAL: Python not found. Install Python 3.10+."
    exit 1
fi

# Check version >= 3.10
$PY_CMD -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ required, got {sys.version}'" 2>/dev/null || {
    echo "  ⚠ WARNING: Python 3.10+ is recommended."
}

# -----------------------------------------------------------
# 2. Check / Install Ollama
# -----------------------------------------------------------
echo ""
echo "[Step 2/7] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "  ✓ Ollama is installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    if command -v ollama &> /dev/null; then
        echo "  ✓ Ollama installed successfully"
    else
        echo "  ✗ FATAL: Ollama installation failed."
        echo "    Please install manually: https://ollama.com/download/linux"
        exit 1
    fi
fi

# -----------------------------------------------------------
# 3. Ensure Ollama is running
# -----------------------------------------------------------
echo ""
echo "[Step 3/7] Ensuring Ollama server is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama server is responding"
else
    echo "  Starting Ollama server in background..."
    nohup ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "  Waiting for server to start (PID: $OLLAMA_PID)..."
    for i in {1..15}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "  ✓ Ollama server started successfully"
            break
        fi
        sleep 2
    done
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✗ FATAL: Cannot reach Ollama server at http://localhost:11434"
        echo "    Try running 'ollama serve' manually in another terminal."
        exit 1
    fi
fi

# -----------------------------------------------------------
# 4. Pull required models
# -----------------------------------------------------------
echo ""
echo "[Step 4/7] Pulling required models (this may take a while on first run)..."
echo "  Pulling llama3.1:8b (~4.7GB)..."
ollama pull llama3.1:8b
echo "  ✓ llama3.1:8b ready"

echo "  Pulling nomic-embed-text:latest (~274MB)..."
ollama pull nomic-embed-text:latest
echo "  ✓ nomic-embed-text ready"

# -----------------------------------------------------------
# 5. Install Python dependencies
# -----------------------------------------------------------
echo ""
echo "[Step 5/7] Installing Python dependencies..."
cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    $PY_CMD -m venv venv
fi

# Activate venv
source venv/bin/activate
echo "  ✓ Virtual environment activated: $(which python3)"

pip install --upgrade pip setuptools wheel -q
pip install -e . 2>&1 | tail -3
pip install numpy pandas tqdm networkx python-dotenv tiktoken matplotlib seaborn ollama 2>&1 | tail -3
echo "  ✓ Python dependencies installed"

# -----------------------------------------------------------
# 6. Smoke test
# -----------------------------------------------------------
echo ""
echo "[Step 6/7] Running smoke tests..."

# Test LLM generation
echo "  Testing LLM generation..."
RESULT=$(ollama run llama3.1:8b "Reply with only the word OK" 2>/dev/null | head -1)
if [ -n "$RESULT" ]; then
    echo "  ✓ LLM generation works (response: $RESULT)"
else
    echo "  ⚠ WARNING: LLM generation returned empty response"
fi

# Test embedding
echo "  Testing embedding..."
EMBED_RESULT=$(curl -s http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text:latest",
  "input": ["test sentence"]
}' 2>/dev/null)
if echo "$EMBED_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['embeddings'][0])>0" 2>/dev/null; then
    EMBED_DIM=$(echo "$EMBED_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['embeddings'][0]))")
    echo "  ✓ Embedding works (dimension: $EMBED_DIM)"
else
    echo "  ⚠ WARNING: Embedding test failed. Check Ollama logs."
fi

# Test Python imports
echo "  Testing Python imports..."
python3 -c "
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
import networkx, matplotlib, numpy, pandas
print('  ✓ All Python imports successful')
" 2>/dev/null || echo "  ⚠ WARNING: Some Python imports failed"

# -----------------------------------------------------------
# 7. Create directories
# -----------------------------------------------------------
echo ""
echo "[Step 7/7] Creating project directories..."
mkdir -p "$PROJECT_DIR/data/splits"
mkdir -p "$PROJECT_DIR/graphs/graph_full"
mkdir -p "$PROJECT_DIR/graphs/graph_inc"
mkdir -p "$PROJECT_DIR/results/figures"
echo "  ✓ Directories created"

# -----------------------------------------------------------
# Check GPU
# -----------------------------------------------------------
echo ""
echo "GPU CHECK:"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU_NAME"
    echo "  Total VRAM: $GPU_MEM"
    echo "  Free VRAM: $GPU_FREE"
else
    echo "  ⚠ nvidia-smi not found — cannot verify GPU"
fi

# -----------------------------------------------------------
# Final Status
# -----------------------------------------------------------
echo ""
echo "=============================================="
echo "  ✅ SETUP COMPLETE"
echo "=============================================="
echo ""
echo "  To run the full experiment:"
echo ""
echo "    source venv/bin/activate"
echo "    python3 experiments/run_all.py"
echo ""
echo "  Estimated runtime: 2-4 hours"
echo "  Estimated cost: \$0 (everything runs locally)"
echo ""
echo "  When done, send back these folders:"
echo "    results/    (REPORT.md + PNG charts)"
echo "    graphs/     (structural_metrics.json)"
echo ""
