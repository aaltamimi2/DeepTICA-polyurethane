#!/usr/bin/env python3
"""
Stage 2: Train DeepTICA from Bootstrap OPES Data
================================================
Trains DeepTICA on ligand descriptors with proper bias reweighting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mlcolvar.cvs import DeepTICA
import lightning as L
from mlcolvar.data import DictModule, DictDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

print("="*80)
print("STAGE 2: Training DeepTICA from Bootstrap Data")
print("="*80)

# ==================== 1. LOAD BOOTSTRAP DATA ====================
print("\n1. Loading bootstrap trajectory data...")

# PLUMED COLVAR files have '#! FIELDS' header that pandas skips
# Read it properly:
with open('COLVAR_BOOTSTRAP', 'r') as f:
    for line in f:
        if line.startswith('#! FIELDS'):
            # Extract column names
            columns = line.split()[2:]  # Skip '#!' and 'FIELDS'
            break

# Now read data with correct column names
colvar = pd.read_csv('COLVAR_BOOTSTRAP', sep='\s+', comment='#', names=columns, header=None)

# Alternative: Let pandas auto-detect but skip the comment
# colvar = pd.read_csv('COLVAR_BOOTSTRAP', sep='\s+', skiprows=1, names=columns)

print(f"✓ Loaded {len(colvar):,} frames")
print(f"  Columns: {colvar.columns.tolist()}")
print(f"  Time range: {colvar['time'].min():.1f} - {colvar['time'].max():.1f} ps")
print(f"  Duration: {(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns")

# ==================== 2. DEFINE DESCRIPTORS ====================
print("\n2. Preparing features for DeepTICA...")

# CRITICAL: These MUST match what's in COLVAR_BOOTSTRAP and what you'll use in PLUMED later!
descriptor_cols = ['rg_lig', 'asph_lig', 'acyl_lig', 'dist_lig_pe', 'dZ']

# Check columns exist
missing = [col for col in descriptor_cols if col not in colvar.columns]
if missing:
    print(f"ERROR: Missing columns in COLVAR: {missing}")
    print(f"Available columns: {colvar.columns.tolist()}")
    exit(1)

# Extract features
X = colvar[descriptor_cols].values
t = colvar['time'].values

print(f"✓ Feature matrix shape: {X.shape}")
print(f"  Features: {descriptor_cols}")

print(f"\nFeature ranges:")
for i, col in enumerate(descriptor_cols):
    print(f"  {col:15s}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]")

# ==================== 3. COMPUTE REWEIGHTING ====================
print("\n3. Computing reweighting factors from OPES bias...")

# Extract bias (for reweighting)
bias = colvar['opes.bias'].values

# Physical constants
kb = 0.008314  # kJ/(mol·K)
temp = 300.0   # K
beta = 1.0 / (kb * temp)

# Compute log weights for reweighting
# For OPES: logweight = beta * bias
logweight = beta * bias

# Clip extreme values for numerical stability
logweight = np.clip(logweight, -10, 10)

print(f"✓ Bias statistics:")
print(f"  Bias range: [{bias.min():.2f}, {bias.max():.2f}] kJ/mol")
print(f"  Logweight range: [{logweight.min():.2f}, {logweight.max():.2f}]")
print(f"  Mean logweight: {logweight.mean():.2f}")

# Check effective sample size
weights = np.exp(logweight - logweight.max())
weights = weights / weights.sum()
eff_sample_size = 1.0 / np.sum(weights**2)
print(f"  Effective sample size: {eff_sample_size:.0f} / {len(X)} ({100*eff_sample_size/len(X):.1f}%)")

# ==================== 4. CREATE TIME-LAGGED DATASET ====================
print("\n4. Creating time-lagged dataset...")

def create_timelagged_dataset_v2(X, lag_time, t=None, logweights=None):
    """Create time-lagged dataset with optional reweighting"""
    n_samples = len(X) - lag_time
    
    X_t0 = torch.FloatTensor(X[:-lag_time])
    X_t1 = torch.FloatTensor(X[lag_time:])
    
    data_dict = {
        'data': X_t0,
        'data_lag': X_t1
    }
    
    if logweights is not None:
        # Weights at t=0 and t=lag
        weights_t0 = np.exp(logweights[:-lag_time] - logweights[:-lag_time].max())
        weights_t1 = np.exp(logweights[lag_time:] - logweights[lag_time:].max())
        
        # Normalize to mean=1
        weights_t0 = weights_t0 / weights_t0.mean()
        weights_t1 = weights_t1 / weights_t1.mean()
        
        data_dict['weights'] = torch.FloatTensor(weights_t0)
        data_dict['weights_lag'] = torch.FloatTensor(weights_t1)
    else:
        data_dict['weights'] = torch.ones(n_samples)
        data_dict['weights_lag'] = torch.ones(n_samples)
    
    return DictDataset(data_dict)

# Lag time selection
# For CG systems with dt=20fs and STRIDE=100: 1 frame = 2 ps
# lag_time=50 means 100 ps lag time
lag_time = 50  # Adjust based on your system dynamics

dataset = create_timelagged_dataset_v2(X, lag_time=lag_time, logweights=logweight)

print(f"✓ Dataset created with reweighting")
print(f"  Total time-lagged pairs: {len(dataset):,}")
print(f"  Lag time: {lag_time} frames = {lag_time * 0.002:.1f} ps")

# ==================== 5. SETUP DEEPTICA MODEL ====================
print("\n5. Setting up DeepTICA model...")

datamodule = DictModule(
    dataset=dataset,
    lengths=[0.8, 0.2],  # 80% train, 20% validation
    batch_size=256,
    random_split=True,
    shuffle=True,
)

n_input = X.shape[1]
n_cvs = 1  # Start with 1 CV for simplicity
layers = [n_input, 50, 50, n_cvs]  # Larger hidden layers for complex polymer systems

print(f"  Architecture: {layers}")
print(f"  Number of CVs: {n_cvs}")
print(f"  Activation: tanh")

model = DeepTICA(
    layers=layers,
    n_cvs=n_cvs,
    options={'nn': {'activation': 'tanh'}},
)

# CRITICAL: Use low learning rate with gradient clipping for stability
model.configure_optimizers = lambda: torch.optim.Adam(model.parameters(), lr=1e-4)

# Setup logging and callbacks
logger = CSVLogger("training_logs", name="deeptica_bootstrap")

checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss',
    dirpath='checkpoints_bootstrap',
    filename='deeptica-{epoch:02d}-{valid_loss:.4f}',
    save_top_k=1,
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='valid_loss',
    patience=50,
    min_delta=0.001,
    mode='min',
)

trainer = L.Trainer(
    max_epochs=500,
    accelerator='auto',
    devices=1,
    gradient_clip_val=1.0,  # Prevent exploding gradients
    enable_progress_bar=True,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
)

# ==================== 6. TRAIN MODEL ====================
print("\n6. Training DeepTICA model...")
print("="*80)

trainer.fit(model, datamodule)

print("\n" + "="*80)
print("✓ TRAINING COMPLETE!")
print("="*80)
print(f"  Best valid loss: {checkpoint_callback.best_model_score:.6f}")
print(f"  Total epochs: {trainer.current_epoch}")
print(f"  Early stopping: {'Yes' if trainer.current_epoch < 500 else 'No'}")

# ==================== 7. VALIDATE EIGENVALUES ====================
print("\n7. Validating eigenvalues...")

with torch.no_grad():
    evals = model.tica.evals.cpu().numpy()
    
    print(f"\nFinal Eigenvalues:")
    all_positive = True
    for i, val in enumerate(evals):
        status = "✓ Valid" if val > 0 else "❌ INVALID (negative)"
        print(f"  λ{i+1} = {val:.6f} {status}")
        if val <= 0:
            all_positive = False
    
    if not all_positive:
        print("\n⚠️  WARNING: Negative eigenvalues detected!")
        print("   Consider: reducing lag_time, lowering learning rate, or longer training")

# ==================== 8. COMPUTE CVs ====================
print("\n8. Computing CVs on full trajectory...")

with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    cvs = model.forward_cv(X_tensor).detach().cpu().numpy()

print(f"✓ CVs computed")
print(f"  CV1 range: [{cvs[:, 0].min():.3f}, {cvs[:, 0].max():.3f}]")

# ==================== 9. VISUALIZE CVs ====================
print("\n9. Creating visualizations...")

fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=120)

# CV1 time evolution
ax = axs[0, 0]
sc = ax.scatter(t/1000, cvs[:, 0], s=1, c=t/1000, cmap='viridis', alpha=0.6)
ax.set_xlabel('Time [ns]')
ax.set_ylabel('DeepTICA CV1')
ax.set_title('CV1 Time Evolution')
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label='Time [ns]')

# CV1 histogram
ax = axs[0, 1]
ax.hist(cvs[:, 0], bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('DeepTICA CV1')
ax.set_ylabel('Count')
ax.set_title('CV1 Distribution')
ax.axvline(cvs[:, 0].mean(), color='r', linestyle='--', label=f'Mean: {cvs[:, 0].mean():.2f}')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# CV1 vs Rg
ax = axs[0, 2]
ax.scatter(colvar['rg_lig'], cvs[:, 0], s=1, alpha=0.5, c=colvar['dist_lig_pe'], cmap='coolwarm')
ax.set_xlabel('Rg (nm)')
ax.set_ylabel('DeepTICA CV1')
ax.set_title('CV1 vs Rg')
ax.grid(alpha=0.3)

# CV1 vs Distance
ax = axs[1, 0]
sc = ax.scatter(colvar['dist_lig_pe'], cvs[:, 0], s=1, alpha=0.5, c=t/1000, cmap='viridis')
ax.set_xlabel('Distance to PE (nm)')
ax.set_ylabel('DeepTICA CV1')
ax.set_title('CV1 vs Distance')
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label='Time [ns]')

# Training loss
ax = axs[1, 1]
try:
    metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
    if 'train_loss_step' in metrics.columns:
        train_loss = metrics.dropna(subset=['train_loss_step'])
        ax.plot(train_loss['step'], train_loss['train_loss_step'], label='Train', alpha=0.7)
    if 'valid_loss_step' in metrics.columns:
        valid_loss = metrics.dropna(subset=['valid_loss_step'])
        ax.plot(valid_loss['step'], valid_loss['valid_loss_step'], label='Valid', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
except:
    ax.text(0.5, 0.5, 'Training metrics\nnot available', 
            ha='center', va='center', transform=ax.transAxes)

# Eigenvalue evolution
ax = axs[1, 2]
try:
    eig_cols = [c for c in metrics.columns if 'train_eigval' in c and 'epoch' in c]
    for col in eig_cols:
        eig_data = metrics.dropna(subset=[col])
        ax.plot(eig_data['epoch'], eig_data[col], label=col.replace('train_eigval_', 'λ'), alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
except:
    ax.text(0.5, 0.5, 'Eigenvalue data\nnot available', 
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('deeptica_bootstrap_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deeptica_bootstrap_analysis.png")

# ==================== 10. EXPORT MODEL FOR PLUMED ====================
print("\n10. Exporting model for PLUMED...")

output_dir = Path("hpc_export")
output_dir.mkdir(exist_ok=True)

# Wrapper class for export (only norm_in + nn, not full TICA)
class ModelForExport(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.norm_in = original_model.norm_in
        self.nn = original_model.nn
    
    def forward(self, x):
        x = self.norm_in(x)
        x = self.nn(x)
        return x

export_model = ModelForExport(model)
export_model.eval()

# Create dummy input for tracing
dummy_input = torch.randn(1, n_input)

# Trace model (not script - script doesn't work with loss functions)
with torch.no_grad():
    traced_model = torch.jit.trace(export_model, dummy_input)
    
    # Verify it works
    test_output = traced_model(dummy_input)
    print(f"✓ Model traced successfully")
    print(f"  Test output shape: {test_output.shape}")

# Save model
model_path = output_dir / "model.ptc"
traced_model.save(str(model_path))
print(f"✓ Model saved: {model_path}")

# ==================== 11. SAVE METADATA ====================
print("\n11. Saving model metadata...")

metadata_path = output_dir / "model_info.txt"

with open(metadata_path, 'w') as f:
    f.write("DeepTICA Model - Bootstrap Training\n")
    f.write("="*60 + "\n\n")
    
    f.write("TRAINING DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Source: COLVAR_BOOTSTRAP\n")
    f.write(f"Frames: {len(colvar):,}\n")
    f.write(f"Duration: {(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns\n")
    f.write(f"Bias type: OPES_METAD on distance\n")
    f.write(f"Reweighting: Yes (beta * bias)\n")
    f.write(f"Effective sample size: {eff_sample_size:.0f} ({100*eff_sample_size/len(X):.1f}%)\n\n")
    
    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Layers: {layers}\n")
    f.write(f"Number of CVs: {n_cvs}\n")
    f.write(f"Lag time: {lag_time} frames ({lag_time * 0.002:.1f} ps)\n")
    f.write(f"Learning rate: 1e-4\n")
    f.write(f"Gradient clipping: 1.0\n\n")
    
    f.write("TRAINING RESULTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Final epochs: {trainer.current_epoch}\n")
    f.write(f"Best valid loss: {checkpoint_callback.best_model_score:.6f}\n")
    f.write(f"Eigenvalues: {evals}\n\n")
    
    f.write("DESCRIPTORS (in order for PLUMED)\n")
    f.write("-" * 40 + "\n")
    for i, col in enumerate(descriptor_cols):
        f.write(f"{i+1}. {col}\n")
    
    f.write(f"\nCV1 RANGE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Training data: [{cvs[:, 0].min():.3f}, {cvs[:, 0].max():.3f}]\n")
    f.write(f"Recommended for PLUMED:\n")
    f.write(f"  CV_MIN = {cvs[:, 0].min() - 0.5:.2f}\n")
    f.write(f"  CV_MAX = {cvs[:, 0].max() + 0.5:.2f}\n")

print(f"✓ Metadata saved: {metadata_path}")

# ==================== 12. CREATE PRODUCTION PLUMED FILE ====================
print("\n12. Creating production PLUMED file...")

cv_min = cvs[:, 0].min() - 0.5
cv_max = cvs[:, 0].max() + 0.5

plumed_production = f"""# PLUMED input for Production Run with DeepTICA CV
# Stage 2: Enhanced sampling using learned collective variable
UNITS LENGTH=nm ENERGY=kj/mol TIME=ps

# ==================== DEFINE GROUPS ====================
PE: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PE
LIG: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PIS_PUS_PTS

WHOLEMOLECULES ENTITY0=PE,LIG

COM_PE: COM ATOMS=PE
COM_LIG: COM ATOMS=LIG

# ==================== COMPUTE DESCRIPTORS ====================
# CRITICAL: These MUST be in the SAME ORDER as training!
# Order: {', '.join(descriptor_cols)}

rg_lig: GYRATION TYPE=RADIUS ATOMS=LIG
asph_lig: GYRATION TYPE=ASPHERICITY ATOMS=LIG
acyl_lig: GYRATION TYPE=ACYLINDRICITY ATOMS=LIG
dist_lig_pe: DISTANCE ATOMS=COM_LIG,COM_PE
dist_components: DISTANCE ATOMS=COM_LIG,COM_PE COMPONENTS
dZ: COMBINE ARG=dist_components.z PERIODIC=NO

# ==================== DEEPTICA CV ====================
# Load trained DeepTICA model
deep: PYTORCH_MODEL FILE=model.ptc ARG=rg_lig,asph_lig,acyl_lig,dist_lig_pe,dZ

# Extract CV1
cv1: COMBINE ARG=deep.node-0 PERIODIC=NO

# ==================== ENERGY ====================
ene: ENERGY

# ==================== OPES EXPANDED ENSEMBLE ====================
# Multithermal expanded ensemble
ecv: ECV_MULTITHERMAL ARG=ene TEMP_MAX=600 TEMP_MIN=300

# Multiumbrellas on learned DeepTICA CV
umb: ECV_UMBRELLAS_LINE ARG=cv1 SIGMA=0.15 CV_MIN={cv_min:.2f} CV_MAX={cv_max:.2f} BARRIER=35

# Combined OPES (temperature + CV exploration)
opes: OPES_EXPANDED ARG=ecv.ene,umb.cv1 PACE=500

# ==================== OUTPUT ====================
# Main output
PRINT STRIDE=100 FILE=COLVAR_PRODUCTION ARG=rg_lig,asph_lig,acyl_lig,dist_lig_pe,dZ,cv1,ene,ecv.ene,umb.cv1,opes.bias FMT=%12.6f

# OPES diagnostics
PRINT STRIDE=500 FILE=OPES_PRODUCTION ARG=opes.*,ecv.*,umb.* FMT=%12.6f

ENDPLUMED
"""

plumed_path = output_dir / "plumed-production.dat"
with open(plumed_path, 'w') as f:
    f.write(plumed_production)

print(f"✓ Production PLUMED file saved: {plumed_path}")

# ==================== 13. SUMMARY ====================
print("\n" + "="*80)
print("STAGE 2 COMPLETE!")
print("="*80)

summary = f"""
Training Summary:
  • Data: {len(colvar):,} frames from bootstrap ({(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns)
  • Reweighting: OPES bias (Eff. N = {eff_sample_size:.0f})
  • CV1 eigenvalue: {evals[0]:.4f} {'✓' if evals[0] > 0 else '❌'}
  • CV1 range: [{cvs[:, 0].min():.3f}, {cvs[:, 0].max():.3f}]

Files Created:
  ✓ {model_path}
  ✓ {metadata_path}
  ✓ {plumed_path}
  ✓ deeptica_bootstrap_analysis.png
  ✓ training_logs/deeptica_bootstrap/
  ✓ checkpoints_bootstrap/

Next Steps:
  1. Review deeptica_bootstrap_analysis.png
  2. Check that CV1 eigenvalue is positive
  3. Verify CV1 captures meaningful dynamics
  4. Copy files to HPC:
     - cp {model_path} /path/to/hpc/
     - cp {plumed_path} /path/to/hpc/plumed.dat
  5. Run production simulation with learned CV!

Production Run Command:
  gmx_mpi mdrun -v -deffnm production-deeptica \\
                -ntomp 8 -plumed plumed-production.dat
"""

print(summary)
print("="*80)