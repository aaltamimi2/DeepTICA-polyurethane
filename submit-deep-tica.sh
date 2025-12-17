#!/bin/bash
#===============================================================================
# Stage 1: Bootstrap with OPES on Distance
# Purpose: Sample desorption events while computing descriptors for DeepTICA
# System: Coarse-grained polymer (dt=20fs)
#===============================================================================

set -e  # Exit on any error

echo "========================================================================"
echo "STAGE 1: Bootstrap Sampling with OPES on Distance"
echo "========================================================================"

# ==================== CONFIGURATION ====================
# Adjust these to match your system
TPR_NAME="bootstrap-stage1"
GRO_FILE="PE-PEUS-LONG-EQ-1us.gro"  # CHANGE THIS
TOP_FILE="PE-PEUS-HOH-0CMC.top"           # CHANGE THIS
NDX_FILE="PE-PEUS-HOH.ndx"

# Simulation parameters
NSTEPS=2500000        # 3 ns (150k steps * 20 fs)
TARGET_TIME_NS=50     # Total simulation time

echo ""
echo "Configuration:"
echo "  Initial structure: $GRO_FILE"
echo "  Topology: $TOP_FILE"
echo "  Index file: $NDX_FILE"
echo "  Simulation time: $TARGET_TIME_NS ns"
echo ""

# ==================== CREATE MDP FILE ====================
echo "Creating MDP file..."

cat > md-bootstrap.mdp << 'EOF'
; Molecular dynamics parameters for bootstrap sampling
; Coarse-grained system with OPES enhanced sampling

integrator              = md
dt                      = 0.02           ; 20 fs timestep (CG system)
nsteps                  = 25000000         ; 3 ns total
nstxtcout               = 10000          ; Save coordinates every 200 ps
nstvout                 = 0
nstfout                 = 0
nstcalcenergy           = 10000
nstenergy               = 10000
nstlog                  = 5000

; Cutoffs
cutoff-scheme           = Verlet
nstlist                 = 20
ns_type                 = grid
pbc                     = xyz
vdwtype                 = Cut-off
vdw-modifier            = Potential-shift-Verlet
rvdw                    = 1.1
verlet-buffer-tolerance = 0.005

; Electrostatics (CG parameters)
coulombtype             = reaction-field
rcoulomb                = 1.1
fourierspacing          = 0.12
epsilon-r               = 15             ; CG setting
DispCorr                = No

; Temperature coupling
tcoupl                  = V-Rescale
tc_grps                 = SYSTEM
tau_t                   = 1.0
ref_t                   = 300

; No pressure coupling
pcoupl                  = no

; Initial velocities
gen-vel                 = yes
gen-temp                = 300
gen-seed                = -1

; Constraints
refcoord-scaling        = all
constraints             = none
constraint-algorithm    = Lincs
EOF

echo "✓ Created md-bootstrap.mdp"

# ==================== CREATE PLUMED INPUT ====================
echo ""
echo "Creating PLUMED input..."

cat > plumed-bootstrap.dat << 'EOF'
UNITS LENGTH=nm ENERGY=kj/mol TIME=ps

#===============================================================================
# AGGRESSIVE DESORPTION CONFIGURATION
# Goal: Force polymer desorption within 10-20 ns
# Strategy: High biasfactor (no height decay) + fast deposition + direct dZ bias
#===============================================================================

PE: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PE
LIG: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PIS_PUS_PTS
HOH: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=W

WHOLEMOLECULES ENTITY0=PE,LIG

COM_PE: COM ATOMS=PE
COM_LIG: COM ATOMS=LIG

# Descriptors for DeepTICA training
rg_lig: GYRATION TYPE=RADIUS ATOMS=LIG
asph_lig: GYRATION TYPE=ASPHERICITY ATOMS=LIG
acyl_lig: GYRATION TYPE=ACYLINDRICITY ATOMS=LIG
dist_lig_pe: DISTANCE ATOMS=COM_LIG,COM_PE

FIRST_BEAD: GROUP ATOMS=17001
LAST_BEAD: GROUP ATOMS=17495
ree: DISTANCE ATOMS=FIRST_BEAD,LAST_BEAD

# Z-component is the desorption coordinate
dist_components: DISTANCE ATOMS=COM_LIG,COM_PE COMPONENTS
dX: COMBINE ARG=dist_components.x PERIODIC=NO
dY: COMBINE ARG=dist_components.y PERIODIC=NO
dZ: COMBINE ARG=dist_components.z PERIODIC=NO

nw: COORDINATION GROUPA=LIG GROUPB=HOH R_0=0.6 NN=6 MM=12
nw_tight: COORDINATION GROUPA=LIG GROUPB=HOH R_0=0.45 NN=6 MM=12
contacts: COORDINATION GROUPA=LIG GROUPB=PE R_0=0.6 NN=6 MM=12

#===============================================================================
# AGGRESSIVE OPES_METAD - Force Desorption
# Very aggressive settings to escape deep adsorption well
#===============================================================================

# Gentle push away from surface to help initiate desorption
# Acts like a soft wall at dZ = -3.0 nm pushing toward more negative dZ
UPPER_WALLS ARG=dZ AT=-3.0 KAPPA=100.0 EXP=2 LABEL=push_off

opes: OPES_METAD ...
    ARG=dZ
    PACE=200
    SIGMA=0.5
    FILE=HILLS_bootstrap
    BARRIER=1000
    BIASFACTOR=100
    SIGMA_MIN=0.2
    NLIST
...

PRINT STRIDE=100 FILE=COLVAR_BOOTSTRAP ARG=rg_lig,asph_lig,acyl_lig,dist_lig_pe,dZ,ree,nw,contacts,opes.bias,push_off.bias FMT=%12.6f

PRINT STRIDE=500 FILE=COLVAR_DETAILED ARG=rg_lig,asph_lig,acyl_lig,dist_lig_pe,dX,dY,dZ,ree,nw,nw_tight,contacts,opes.*,push_off.* FMT=%12.6f

ENDPLUMED
EOF

echo "✓ Created plumed-bootstrap.dat"

# ==================== RUN GROMACS ====================
echo ""
echo "========================================================================"
echo "Running GROMACS with OPES Enhanced Sampling"
echo "========================================================================"
echo ""

# Step 1: Grompp (prepare simulation)
echo "Step 1: Running grompp..."
gmx_mpi grompp -f md-bootstrap.mdp \
             -c $GRO_FILE \
             -p $TOP_FILE \
             -n $NDX_FILE \
             -o ${TPR_NAME}.tpr \
             -maxwarn 4

if [ $? -eq 0 ]; then
    echo "✓ grompp successful"
else
    echo "❌ grompp failed!"
    exit 1
fi

echo ""
echo "Step 2: Running mdrun with PLUMED..."
echo "  This will take a while (approximately 3 ns)..."
echo "  Monitor progress: tail -f ${TPR_NAME}.log"
echo ""

# Step 2: Mdrun (run simulation)
# Adjust -ntomp based on your workstation CPU cores
gmx_mpi mdrun -v -deffnm ${TPR_NAME} \
             -ntomp 8 \
             -plumed plumed-bootstrap.dat

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Simulation completed successfully!"
else
    echo ""
    echo "❌ Simulation failed!"
    exit 1
fi

# ==================== POST-PROCESSING ====================
echo ""
echo "========================================================================"
echo "Post-processing Results"
echo "========================================================================"

# Check output files
echo ""
echo "Generated files:"
ls -lh ${TPR_NAME}.*
ls -lh COLVAR_BOOTSTRAP
ls -lh HILLS_bootstrap

# Quick statistics
echo ""
echo "Quick statistics from COLVAR_BOOTSTRAP:"

if command -v python3 &> /dev/null; then
    python3 << 'PYTHON_EOF'
import pandas as pd
import numpy as np

try:
    colvar = pd.read_csv('COLVAR_BOOTSTRAP', sep='\s+', comment='#')
    
    print(f"  Frames collected: {len(colvar):,}")
    print(f"  Time range: {colvar['time'].min():.1f} - {colvar['time'].max():.1f} ps")
    print(f"\n  Descriptor ranges:")
    print(f"    rg_lig:      [{colvar['rg_lig'].min():.3f}, {colvar['rg_lig'].max():.3f}] nm")
    print(f"    dist_lig_pe: [{colvar['dist_lig_pe'].min():.3f}, {colvar['dist_lig_pe'].max():.3f}] nm")
    print(f"    asph_lig:    [{colvar['asph_lig'].min():.3f}, {colvar['asph_lig'].max():.3f}]")
    print(f"\n  Bias statistics:")
    print(f"    Min bias:  {colvar['opes.bias'].min():.2f} kJ/mol")
    print(f"    Max bias:  {colvar['opes.bias'].max():.2f} kJ/mol")
    print(f"    Mean bias: {colvar['opes.bias'].mean():.2f} kJ/mol")
    
    # Check if desorption was sampled
    max_dist = colvar['dist_lig_pe'].max()
    if max_dist > 2.0:
        print(f"\n  ✓ Desorption sampled! (max distance: {max_dist:.2f} nm)")
    else:
        print(f"\n  ⚠️  Limited desorption (max distance: {max_dist:.2f} nm)")
        print(f"     Consider longer simulation or stronger bias")
    
except Exception as e:
    print(f"Could not analyze COLVAR: {e}")
PYTHON_EOF
else
    echo "  (Python not available for quick analysis)"
    echo "  Check COLVAR_BOOTSTRAP manually"
fi

# ==================== SUMMARY ====================
echo ""
echo "========================================================================"
echo "STAGE 1 COMPLETE!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Bootstrap sampling with OPES on distance completed"
echo "  ✓ Generated COLVAR_BOOTSTRAP with all descriptors"
echo "  ✓ Generated HILLS_bootstrap with bias information"
echo ""
echo "Next Steps:"
echo "  1. Review COLVAR_BOOTSTRAP to ensure desorption was sampled"
echo "  2. Visualize trajectory to check sampling quality"
echo "  3. Run Stage 2: Train DeepTICA on this data"
echo "     → python train_deeptica_stage2.py"
echo ""
echo "Files for DeepTICA training:"
echo "  • COLVAR_BOOTSTRAP (contains descriptors + bias for reweighting)"
echo "  • ${TPR_NAME}.xtc (trajectory for visualization)"
echo ""
echo "========================================================================"
