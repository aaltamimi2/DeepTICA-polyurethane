#!/bin/bash
#===============================================================================
# Stage 2: Train DeepTICA from Bootstrap Data
# Enhanced version with parameter sweep and extensive visualization
#===============================================================================

set -e

echo "========================================================================"
echo "STAGE 2: DeepTICA Training"
echo "========================================================================"

# ==================== CONFIGURATION ====================
# Training mode: "single" or "sweep"
MODE="${1:-single}"

# Single run parameters (used if MODE=single)
LEARNING_RATE="${LR:-1e-4}"
LAG_TIME="${LAG:-50}"
HIDDEN_SIZE="${HIDDEN:-64}"
N_LAYERS="${LAYERS:-2}"
MAX_EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH:-256}"
N_CVS="${NCVS:-1}"

# Input/Output
COLVAR_FILE="${COLVAR:-COLVAR_BOOTSTRAP}"
OUTPUT_DIR="${OUTPUT:-deeptica_output}"

# ==================== CHECK REQUIREMENTS ====================
echo ""
echo "Checking requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# Check required packages
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || { echo "ERROR: PyTorch not installed"; exit 1; }
python3 -c "import mlcolvar; print(f'  mlcolvar: installed')" || { echo "ERROR: mlcolvar not installed"; exit 1; }
python3 -c "import lightning; print(f'  Lightning: {lightning.__version__}')" || { echo "ERROR: lightning not installed"; exit 1; }

# Check COLVAR file
if [ ! -f "$COLVAR_FILE" ]; then
    echo "ERROR: COLVAR file not found: $COLVAR_FILE"
    echo "  Run Stage 1 (bootstrap) first!"
    exit 1
fi
echo "  COLVAR file: $COLVAR_FILE ($(wc -l < "$COLVAR_FILE") lines)"

# ==================== RUN TRAINING ====================
echo ""
echo "========================================================================"

if [ "$MODE" == "sweep" ]; then
    echo "Running parameter sweep..."
    echo "========================================================================"
    echo ""

    python3 submit-deep-tica-2.py \
        --sweep \
        --colvar "$COLVAR_FILE" \
        --output "$OUTPUT_DIR" \
        --epochs "$MAX_EPOCHS" \
        --batch "$BATCH_SIZE" \
        --n_cvs "$N_CVS" \
        --layers "$N_LAYERS"

else
    echo "Running single training..."
    echo "========================================================================"
    echo ""
    echo "Configuration:"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Lag time: $LAG_TIME frames"
    echo "  Hidden size: $HIDDEN_SIZE"
    echo "  Hidden layers: $N_LAYERS"
    echo "  Max epochs: $MAX_EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Number of CVs: $N_CVS"
    echo ""

    python3 submit-deep-tica-2.py \
        --lr "$LEARNING_RATE" \
        --lag "$LAG_TIME" \
        --hidden "$HIDDEN_SIZE" \
        --layers "$N_LAYERS" \
        --epochs "$MAX_EPOCHS" \
        --batch "$BATCH_SIZE" \
        --n_cvs "$N_CVS" \
        --colvar "$COLVAR_FILE" \
        --output "$OUTPUT_DIR"
fi

# ==================== POST-PROCESSING ====================
echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Output files in $OUTPUT_DIR/:"
ls -la "$OUTPUT_DIR/"

echo ""
echo "========================================================================"
echo "USAGE EXAMPLES"
echo "========================================================================"
echo ""
echo "Single run with defaults:"
echo "  ./submit-deep-tica-2.sh"
echo ""
echo "Single run with custom parameters:"
echo "  LR=1e-3 LAG=100 HIDDEN=128 ./submit-deep-tica-2.sh"
echo ""
echo "Parameter sweep:"
echo "  ./submit-deep-tica-2.sh sweep"
echo ""
echo "Custom COLVAR file:"
echo "  COLVAR=my_colvar.dat OUTPUT=my_output ./submit-deep-tica-2.sh"
echo ""
echo "========================================================================"
