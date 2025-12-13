#!/bin/bash
#===============================================================================
# Stage 3: Production Run with DeepTICA CV
# Purpose: Enhanced sampling using learned collective variable
#===============================================================================

set -e

echo "========================================================================"
echo "STAGE 3: Production Run with DeepTICA CV"
echo "========================================================================"

# ==================== CONFIGURATION ====================
TPR_NAME="production-deeptica"
GRO_FILE="PE-PEUS-LONG-EQ-1us.gro"  # CHANGE THIS
TOP_FILE="PE-PEUS-HOH-0CMC.top"           # CHANGE THIS
NDX_FILE="PE-PEUS-HOH.ndx"
MODEL_FILE="hpc_export/model.ptc"
PLUMED_FILE="hpc_export/plumed-production.dat"

# Check files exist
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file not found: $MODEL_FILE"
    echo "Run train_deeptica_stage2.py first!"
    exit 1
fi

if [ ! -f "$PLUMED_FILE" ]; then
    echo "ERROR: PLUMED file not found: $PLUMED_FILE"
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL_FILE"
echo "  PLUMED: $PLUMED_FILE"
echo "  Structure: $GRO_FILE"
echo ""

# ==================== CREATE MDP FILE ====================
cat > md-production.mdp << 'EOF'
; Production MD with DeepTICA CV
integrator              = md
dt                      = 0.02
nsteps                  = 5000000        ; 10 ns (adjust as needed)
nstxtcout               = 10000
nstvout                 = 0
nstfout                 = 0
nstcalcenergy           = 10000
nstenergy               = 10000
nstlog                  = 10000

cutoff-scheme           = Verlet
nstlist                 = 20
ns_type                 = grid
pbc                     = xyz
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw                    = 1.1
verlet-buffer-tolerance = 0.005

coulombtype             = reaction-field
rcoulomb                = 1.1
fourierspacing          = 0.12
epsilon-r               = 15
DispCorr                = No

tcoupl                  = V-Rescale
tc_grps                 = SYSTEM
tau_t                   = 1.0
ref_t                   = 300

pcoupl                  = no

gen-vel                 = yes
gen-temp                = 300
gen-seed                = -1

refcoord-scaling        = all
constraints             = none
constraint-algorithm    = Lincs
EOF

echo "✓ Created md-production.mdp"

# ==================== GROMPP ====================
echo ""
echo "Running grompp..."
gmx_mpi grompp -f md-production.mdp \
             -c $GRO_FILE \
             -p $TOP_FILE \
             -n $NDX_FILE \
             -o ${TPR_NAME}.tpr \
             -maxwarn 4

echo "✓ grompp successful"

# ==================== MDRUN ====================
echo ""
echo "Running production MD with DeepTICA CV..."
echo "This will take a while..."
echo ""

gmx_mpi mdrun -v -deffnm ${TPR_NAME} \
             -ntomp 8 \
             -plumed $PLUMED_FILE

echo ""
echo "✓ Production run complete!"

# ==================== SUMMARY ====================
echo ""
echo "========================================================================"
echo "STAGE 3 COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated files:"
ls -lh ${TPR_NAME}.*
ls -lh COLVAR_PRODUCTION
ls -lh OPES_PRODUCTION
echo ""
echo "Next: Analyze results with analysis scripts"
echo "========================================================================"