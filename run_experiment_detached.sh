#!/usr/bin/env bash
#
# LightRAG Microstudy — Detached Runner
# =======================================
# Runs the full experiment pipeline in the background using nohup.
# You can safely close the terminal. Monitor progress with:
#
#   tail -f /home/danish/LightRAG\ —\ Incremental\ Graph\ Degradation\ Test/LightRAG/experiment.log
#
# Check if it's still running:
#   ps aux | grep run_all.py
#
# To stop it:
#   kill $(cat /home/danish/LightRAG\ —\ Incremental\ Graph\ Degradation\ Test/LightRAG/experiment.pid)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/experiment.log"
PID_FILE="$SCRIPT_DIR/experiment.pid"

echo "=================================================="
echo "  LightRAG Microstudy — Detached Runner"
echo "=================================================="
echo ""

# -----------------------------------------------------------
# Step 0: Fix CRLF line endings
# -----------------------------------------------------------
echo "[Pre] Fixing line endings..."
find "$SCRIPT_DIR/experiments" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.json" \) -exec sed -i 's/\r$//' {} \;
[ -f "$SCRIPT_DIR/.env" ] && sed -i 's/\r$//' "$SCRIPT_DIR/.env"
chmod +x "$SCRIPT_DIR/experiments/setup_environment.sh"
echo "  ✓ Line endings fixed"

# -----------------------------------------------------------
# Step 1: Run setup if venv doesn't exist
# -----------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo ""
    echo "[Pre] Virtual environment not found. Running setup_environment.sh first..."
    echo "  This may take ~10 minutes (downloading models, installing deps)..."
    echo ""
    bash "$SCRIPT_DIR/experiments/setup_environment.sh" 2>&1 | tee "$LOG_FILE"
    echo ""
    echo "  ✓ Setup complete"
else
    echo "[Pre] Virtual environment found, skipping setup."
fi

# -----------------------------------------------------------
# Step 2: Launch the experiment in background
# -----------------------------------------------------------
echo ""
echo "[Launch] Starting experiment pipeline in background..."
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

# Use 'yes y' to auto-answer any interactive prompts
# Redirect all output to log file, run via nohup so it survives terminal close
nohup bash -c "
    source '$SCRIPT_DIR/venv/bin/activate'
    echo '=== Experiment started at \$(date) ===' >> '$LOG_FILE'
    echo '' >> '$LOG_FILE'
    # Pipe 'yes y' to handle the 'Continue anyway?' prompt if it appears
    yes y 2>/dev/null | python3 '$SCRIPT_DIR/experiments/run_all.py' >> '$LOG_FILE' 2>&1
    EXIT_CODE=\$?
    echo '' >> '$LOG_FILE'
    echo '=== Experiment finished at \$(date) with exit code \$EXIT_CODE ===' >> '$LOG_FILE'
    rm -f '$PID_FILE'
" &

EXPERIMENT_PID=$!
echo "$EXPERIMENT_PID" > "$PID_FILE"

echo "  ✓ Experiment launched (PID: $EXPERIMENT_PID)"
echo ""
echo "=================================================="
echo "  You can now safely close this terminal!"
echo "=================================================="
echo ""
echo "  📋 Monitor progress:"
echo "    tail -f '$LOG_FILE'"
echo ""
echo "  🔍 Check if still running:"
echo "    ps -p \$(cat '$PID_FILE') 2>/dev/null && echo 'RUNNING' || echo 'FINISHED'"
echo ""
echo "  🛑 Stop the experiment:"
echo "    kill \$(cat '$PID_FILE')"
echo ""
echo "  📄 View full log:"
echo "    less '$LOG_FILE'"
echo ""
