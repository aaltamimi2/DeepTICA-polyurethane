#!/usr/bin/env python3
"""
Stage 2: Train DeepTICA from Bootstrap Data (UNBIASED VERSION)
===============================================================
For use when OPES bias was underestimated and reweighting would be inaccurate.
Uses uniform weights - all frames treated equally.

Key focus:
- dZ: Primary desorption coordinate (z-component of ligand-surface distance)
- contacts: PE-LIG contacts (decreases during desorption)
- Rg variables: Polymer shape changes during desorption

Usage:
  python submit-deep-tica-UNBIASED-2.py                    # Single run with defaults
  python submit-deep-tica-UNBIASED-2.py --sweep            # Parameter sweep
  python submit-deep-tica-UNBIASED-2.py --lr 1e-3          # Custom learning rate
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from mlcolvar.cvs import DeepTICA
import lightning as L
from mlcolvar.data import DictModule, DictDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import argparse
import json
from itertools import product
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== ARGUMENT PARSING ====================
parser = argparse.ArgumentParser(description='Train DeepTICA WITHOUT reweighting (uniform weights)')
parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
parser.add_argument('--lag', type=int, default=50, help='Lag time in frames (default: 50)')
parser.add_argument('--hidden', type=int, default=64, help='Hidden layer size (default: 64)')
parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers (default: 2)')
parser.add_argument('--epochs', type=int, default=500, help='Max epochs (default: 500)')
parser.add_argument('--batch', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--n_cvs', type=int, default=1, help='Number of CVs to learn (default: 1)')
parser.add_argument('--colvar', type=str, default='COLVAR_BOOTSTRAP', help='COLVAR file path')
parser.add_argument('--output', type=str, default='deeptica_unbiased_output', help='Output directory')
args = parser.parse_args()

print("="*80)
print("STAGE 2: Training DeepTICA (UNBIASED - No Reweighting)")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNOTE: Using uniform weights - all frames weighted equally")
print("      This is appropriate when OPES bias was underestimated")

# ==================== 1. LOAD BOOTSTRAP DATA ====================
print("\n1. Loading bootstrap trajectory data...")

# Parse PLUMED COLVAR header for column names
with open(args.colvar, 'r') as f:
    for line in f:
        if line.startswith('#! FIELDS'):
            columns = line.split()[2:]
            break

colvar = pd.read_csv(args.colvar, sep=r'\s+', comment='#', names=columns, header=None)

print(f"   Loaded {len(colvar):,} frames")
print(f"   Columns: {colvar.columns.tolist()}")
print(f"   Time range: {colvar['time'].min():.1f} - {colvar['time'].max():.1f} ps")
print(f"   Duration: {(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns")

# ==================== 2. DEFINE DESCRIPTORS ====================
print("\n2. Preparing features for DeepTICA...")

# KEY DESCRIPTORS FOR DESORPTION:
# - dZ: z-component of ligand-surface distance (PRIMARY desorption coordinate)
# - contacts: PE-LIG contacts (decreases during desorption)
# - rg_lig: Radius of gyration (polymer size/compactness)
# - asph_lig: Asphericity (polymer shape)
# - acyl_lig: Acylindricity (polymer shape)
# - dist_lig_pe: Total 3D distance (secondary to dZ)
# - ree: End-to-end distance (polymer extension)
# - nw: Water coordination (hydration changes during desorption)

# Priority order - most important for desorption first
descriptor_priority = ['dZ', 'contacts', 'rg_lig', 'asph_lig', 'acyl_lig', 'dist_lig_pe', 'ree', 'nw']

# Check which columns exist and use only those
available_cols = [col for col in descriptor_priority if col in colvar.columns]
missing_cols = [col for col in descriptor_priority if col not in colvar.columns]

if missing_cols:
    print(f"   Note: Missing columns (will skip): {missing_cols}")

descriptor_cols = available_cols
print(f"   Using descriptors: {descriptor_cols}")

# Validate critical columns
critical_cols = ['dZ', 'contacts']
missing_critical = [col for col in critical_cols if col not in descriptor_cols]
if missing_critical:
    print(f"\n   WARNING: Missing critical desorption descriptors: {missing_critical}")
    print("   Results may not accurately capture desorption!")

if len(descriptor_cols) < 3:
    print("ERROR: Need at least 3 descriptors for meaningful DeepTICA!")
    exit(1)

# Extract features
X = colvar[descriptor_cols].values
t = colvar['time'].values

print(f"   Feature matrix shape: {X.shape}")

print(f"\n   Feature statistics:")
feature_stats = {}
for i, col in enumerate(descriptor_cols):
    mean_val = X[:, i].mean()
    std_val = X[:, i].std()
    min_val = X[:, i].min()
    max_val = X[:, i].max()
    feature_stats[col] = {'mean': float(mean_val), 'std': float(std_val),
                          'min': float(min_val), 'max': float(max_val)}
    print(f"   {col:15s}: [{min_val:.3f}, {max_val:.3f}] (mean: {mean_val:.3f}, std: {std_val:.3f})")

# ==================== 3. VALIDATE DESORPTION DATA ====================
print("\n3. Validating desorption sampling...")

if 'dZ' in descriptor_cols:
    dZ_idx = descriptor_cols.index('dZ')
    dZ_range = X[:, dZ_idx].max() - X[:, dZ_idx].min()
    dZ_min = X[:, dZ_idx].min()
    dZ_max = X[:, dZ_idx].max()

    print(f"   dZ range: [{dZ_min:.2f}, {dZ_max:.2f}] nm (span: {dZ_range:.2f} nm)")

    if dZ_range < 2.0:
        print("   WARNING: Limited dZ range - desorption may not be fully sampled!")
    elif dZ_range < 5.0:
        print("   MODERATE: dZ range suggests partial desorption")
    else:
        print("   GOOD: dZ range suggests significant desorption sampled")

if 'contacts' in descriptor_cols:
    cont_idx = descriptor_cols.index('contacts')
    cont_min = X[:, cont_idx].min()
    cont_max = X[:, cont_idx].max()
    cont_range = cont_max - cont_min

    print(f"   Contacts range: [{cont_min:.0f}, {cont_max:.0f}] (span: {cont_range:.0f})")

    if cont_min > cont_max * 0.5:
        print("   WARNING: Contacts didn't decrease much - may still be adsorbed!")
    else:
        print("   GOOD: Significant contact reduction observed")

# ==================== 4. NO REWEIGHTING (UNIFORM WEIGHTS) ====================
print("\n4. Setting uniform weights (NO reweighting)...")
print("   Reason: OPES bias was underestimated, reweighting would be inaccurate")
print("   All frames will be treated equally for DeepTICA training")

# Uniform weights - no reweighting
logweight = np.zeros(len(colvar))
eff_sample_size = len(colvar)  # All samples equally weighted

print(f"   Effective sample size: {eff_sample_size:,} / {len(X):,} (100.0%)")

# ==================== 5. HELPER FUNCTIONS ====================

def create_timelagged_dataset(X, lag_time, logweights=None):
    """Create time-lagged dataset with optional reweighting"""
    n_samples = len(X) - lag_time

    X_t0 = torch.FloatTensor(X[:-lag_time])
    X_t1 = torch.FloatTensor(X[lag_time:])

    data_dict = {'data': X_t0, 'data_lag': X_t1}

    if logweights is not None and np.any(logweights != 0):
        weights_t0 = np.exp(logweights[:-lag_time] - logweights[:-lag_time].max())
        weights_t1 = np.exp(logweights[lag_time:] - logweights[lag_time:].max())
        weights_t0 = weights_t0 / weights_t0.mean()
        weights_t1 = weights_t1 / weights_t1.mean()
        data_dict['weights'] = torch.FloatTensor(weights_t0)
        data_dict['weights_lag'] = torch.FloatTensor(weights_t1)
    else:
        # Uniform weights
        data_dict['weights'] = torch.ones(n_samples)
        data_dict['weights_lag'] = torch.ones(n_samples)

    return DictDataset(data_dict)


def train_model(X, logweight, config, descriptor_cols, verbose=True):
    """Train a single DeepTICA model with given configuration"""

    dataset = create_timelagged_dataset(X, lag_time=config['lag_time'], logweights=logweight)

    datamodule = DictModule(
        dataset=dataset,
        lengths=[0.8, 0.2],
        batch_size=config['batch_size'],
        random_split=True,
        shuffle=True,
    )

    n_input = X.shape[1]
    layers = [n_input] + [config['hidden_size']] * config['n_layers'] + [config['n_cvs']]

    model = DeepTICA(
        layers=layers,
        n_cvs=config['n_cvs'],
        options={'nn': {'activation': 'tanh'}},
    )

    # Custom optimizer with specified learning rate
    model.configure_optimizers = lambda: torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-6
    )

    run_name = f"unbiased_lr{config['learning_rate']}_lag{config['lag_time']}_h{config['hidden_size']}"
    logger = CSVLogger("training_logs", name=run_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath=f'checkpoints/{run_name}',
        filename='best-{epoch:02d}-{valid_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    early_stop = EarlyStopping(
        monitor='valid_loss',
        patience=30,
        min_delta=0.0001,
        mode='min',
    )

    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=verbose,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule)

    # Get results
    with torch.no_grad():
        evals = model.tica.evals.cpu().numpy()
        X_tensor = torch.FloatTensor(X)
        cvs = model.forward_cv(X_tensor).detach().cpu().numpy()

    results = {
        'config': config,
        'model': model,
        'trainer': trainer,
        'logger': logger,
        'checkpoint': checkpoint_callback,
        'eigenvalues': evals,
        'cvs': cvs,
        'best_loss': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else float('inf'),
        'epochs': trainer.current_epoch,
        'layers': layers,
    }

    return results


def compute_feature_importance(model, X, descriptor_cols):
    """Compute feature importance using gradient-based sensitivity analysis"""

    model.eval()
    X_tensor = torch.FloatTensor(X)
    X_tensor.requires_grad = True

    # Forward pass
    cvs = model.forward_cv(X_tensor)

    # Compute gradients for each CV
    importance = {}
    for cv_idx in range(cvs.shape[1]):
        # Backprop for this CV
        model.zero_grad()
        cv_sum = cvs[:, cv_idx].sum()
        cv_sum.backward(retain_graph=True)

        # Get absolute gradients averaged over samples
        grads = X_tensor.grad.abs().mean(dim=0).detach().numpy()
        X_tensor.grad.zero_()

        importance[f'CV{cv_idx+1}'] = {col: float(grads[i]) for i, col in enumerate(descriptor_cols)}

    return importance


def compute_feature_correlations(cvs, X, descriptor_cols):
    """Compute Pearson and Spearman correlations between features and CVs"""

    correlations = {'pearson': {}, 'spearman': {}}

    for cv_idx in range(cvs.shape[1]):
        cv_name = f'CV{cv_idx+1}'
        correlations['pearson'][cv_name] = {}
        correlations['spearman'][cv_name] = {}

        for i, col in enumerate(descriptor_cols):
            r_pearson, p_pearson = stats.pearsonr(X[:, i], cvs[:, cv_idx])
            r_spearman, p_spearman = stats.spearmanr(X[:, i], cvs[:, cv_idx])

            correlations['pearson'][cv_name][col] = {'r': float(r_pearson), 'p': float(p_pearson)}
            correlations['spearman'][cv_name][col] = {'r': float(r_spearman), 'p': float(p_spearman)}

    return correlations


def create_desorption_visualizations(results, X, t, colvar, descriptor_cols, output_dir):
    """Create visualizations focused on desorption analysis"""

    cvs = results['cvs']
    model = results['model']
    n_cvs = cvs.shape[1]

    kb = 0.008314  # kJ/(mol·K)
    temp = 300.0

    # ===== FIGURE 1: CV Time Evolution and Distribution =====
    fig1, axes = plt.subplots(2, n_cvs + 1, figsize=(5*(n_cvs+1), 10))
    if n_cvs == 1:
        axes = axes.reshape(2, -1)

    for cv_idx in range(n_cvs):
        # Time evolution
        ax = axes[0, cv_idx]
        sc = ax.scatter(t/1000, cvs[:, cv_idx], s=1, c=t/1000, cmap='viridis', alpha=0.6)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel(f'DeepTICA CV{cv_idx+1}')
        ax.set_title(f'CV{cv_idx+1} Time Evolution')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time [ns]')

        # Distribution
        ax = axes[1, cv_idx]
        ax.hist(cvs[:, cv_idx], bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(cvs[:, cv_idx].mean(), color='r', linestyle='--',
                   label=f'Mean: {cvs[:, cv_idx].mean():.2f}')
        ax.axvline(np.median(cvs[:, cv_idx]), color='g', linestyle=':',
                   label=f'Median: {np.median(cvs[:, cv_idx]):.2f}')
        ax.set_xlabel(f'DeepTICA CV{cv_idx+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'CV{cv_idx+1} Distribution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Training loss in last column
    ax = axes[0, -1]
    try:
        metrics_path = Path(results['logger'].log_dir) / 'metrics.csv'
        metrics = pd.read_csv(metrics_path)

        train_cols = [c for c in metrics.columns if 'train_loss' in c and 'epoch' in c]
        valid_cols = [c for c in metrics.columns if 'valid_loss' in c and 'epoch' in c]

        if train_cols:
            train_data = metrics.dropna(subset=[train_cols[0]])
            ax.plot(train_data['epoch'], train_data[train_cols[0]], 'b-', label='Train', alpha=0.7)
        if valid_cols:
            valid_data = metrics.dropna(subset=[valid_cols[0]])
            ax.plot(valid_data['epoch'], valid_data[valid_cols[0]], 'r-', label='Valid', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Metrics not available\n{e}', ha='center', va='center', transform=ax.transAxes)

    # Eigenvalue info
    ax = axes[1, -1]
    evals = results['eigenvalues']
    colors = ['green' if e > 0 else 'red' for e in evals]
    bars = ax.bar(range(len(evals)), evals, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('CV Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('TICA Eigenvalues')
    ax.set_xticks(range(len(evals)))
    ax.set_xticklabels([f'CV{i+1}' for i in range(len(evals))])
    for i, (bar, val) in enumerate(zip(bars, evals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cv_evolution_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: cv_evolution_distribution.png")

    # ===== FIGURE 2: Feature Importance Analysis =====
    importance = compute_feature_importance(model, X, descriptor_cols)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    cv_name = 'CV1'
    imp_values = list(importance[cv_name].values())
    imp_names = list(importance[cv_name].keys())

    # Normalize to percentages
    imp_pct = 100 * np.array(imp_values) / np.sum(imp_values)

    # Sort by importance
    sort_idx = np.argsort(imp_pct)[::-1]
    imp_pct_sorted = imp_pct[sort_idx]
    names_sorted = [imp_names[i] for i in sort_idx]

    # Color critical descriptors differently
    colors = []
    for name in names_sorted:
        if name in ['dZ', 'contacts']:
            colors.append('darkgreen')
        elif name in ['rg_lig', 'asph_lig', 'acyl_lig']:
            colors.append('steelblue')
        else:
            colors.append('gray')

    bars = ax.barh(range(len(names_sorted)), imp_pct_sorted, color=colors, edgecolor='black')
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title(f'{cv_name} Feature Importance\n(Green=desorption, Blue=shape)')
    ax.grid(alpha=0.3, axis='x')

    # Add percentage labels
    for bar, pct in zip(bars, imp_pct_sorted):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=8)

    # Cumulative importance plot
    ax = axes[1]
    cumsum = np.cumsum(imp_pct_sorted)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'bo-', markersize=8)
    ax.axhline(80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Cumulative Feature Importance')
    ax.set_xticks(range(1, len(cumsum)+1))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: feature_importance.png")

    # ===== FIGURE 3: Desorption-Focused Analysis =====
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CV1 vs dZ (primary desorption coordinate)
    if 'dZ' in descriptor_cols:
        ax = axes[0, 0]
        dZ_idx = descriptor_cols.index('dZ')
        sc = ax.scatter(X[:, dZ_idx], cvs[:, 0], s=2, c=t/1000, cmap='viridis', alpha=0.5)
        ax.set_xlabel('dZ (nm)')
        ax.set_ylabel('DeepTICA CV1')
        ax.set_title('CV1 vs dZ (Desorption Coordinate)')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time [ns]')

        # Add correlation
        r = stats.pearsonr(X[:, dZ_idx], cvs[:, 0])[0]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # CV1 vs contacts (secondary desorption indicator)
    if 'contacts' in descriptor_cols:
        ax = axes[0, 1]
        cont_idx = descriptor_cols.index('contacts')
        sc = ax.scatter(X[:, cont_idx], cvs[:, 0], s=2, c=t/1000, cmap='viridis', alpha=0.5)
        ax.set_xlabel('PE-LIG Contacts')
        ax.set_ylabel('DeepTICA CV1')
        ax.set_title('CV1 vs Contacts')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time [ns]')

        r = stats.pearsonr(X[:, cont_idx], cvs[:, 0])[0]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # dZ vs contacts colored by CV1
    if 'dZ' in descriptor_cols and 'contacts' in descriptor_cols:
        ax = axes[1, 0]
        dZ_idx = descriptor_cols.index('dZ')
        cont_idx = descriptor_cols.index('contacts')
        sc = ax.scatter(X[:, dZ_idx], X[:, cont_idx], s=2, c=cvs[:, 0], cmap='RdYlBu', alpha=0.5)
        ax.set_xlabel('dZ (nm)')
        ax.set_ylabel('PE-LIG Contacts')
        ax.set_title('dZ vs Contacts (colored by CV1)')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='CV1')

    # CV1 vs Rg
    if 'rg_lig' in descriptor_cols:
        ax = axes[1, 1]
        rg_idx = descriptor_cols.index('rg_lig')
        sc = ax.scatter(X[:, rg_idx], cvs[:, 0], s=2, c=t/1000, cmap='viridis', alpha=0.5)
        ax.set_xlabel('Rg (nm)')
        ax.set_ylabel('DeepTICA CV1')
        ax.set_title('CV1 vs Radius of Gyration')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time [ns]')

        r = stats.pearsonr(X[:, rg_idx], cvs[:, 0])[0]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(output_dir / 'desorption_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: desorption_analysis.png")

    # ===== FIGURE 4: Feature Correlations =====
    correlations = compute_feature_correlations(cvs, X, descriptor_cols)

    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pearson correlation heatmap
    ax = axes[0]
    pearson_matrix = np.array([[correlations['pearson'][f'CV{cv+1}'][col]['r']
                                 for col in descriptor_cols] for cv in range(n_cvs)])
    sns.heatmap(pearson_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=descriptor_cols, yticklabels=[f'CV{i+1}' for i in range(n_cvs)],
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Pearson Correlation: CVs vs Features')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Feature-feature correlation
    ax = axes[1]
    feature_corr = np.corrcoef(X.T)
    sns.heatmap(feature_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=descriptor_cols, yticklabels=descriptor_cols,
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature-Feature Correlations')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: feature_correlations.png")

    # ===== FIGURE 5: 2D Free Energy Surfaces =====
    fig5, axes = plt.subplots(1, 3, figsize=(15, 5))

    # FES: CV1 vs dZ
    if 'dZ' in descriptor_cols:
        ax = axes[0]
        dZ_idx = descriptor_cols.index('dZ')
        h, xedges, yedges = np.histogram2d(X[:, dZ_idx], cvs[:, 0], bins=50, density=True)
        h = np.ma.masked_where(h == 0, h)
        fes = -kb * temp * np.log(h.T)
        fes -= np.nanmin(fes)

        im = ax.imshow(fes, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='viridis', vmax=25)
        ax.set_xlabel('dZ (nm)')
        ax.set_ylabel('CV1')
        ax.set_title('Free Energy: CV1 vs dZ')
        plt.colorbar(im, ax=ax, label='F (kJ/mol)')

    # FES: CV1 vs contacts
    if 'contacts' in descriptor_cols:
        ax = axes[1]
        cont_idx = descriptor_cols.index('contacts')
        h, xedges, yedges = np.histogram2d(X[:, cont_idx], cvs[:, 0], bins=50, density=True)
        h = np.ma.masked_where(h == 0, h)
        fes = -kb * temp * np.log(h.T)
        fes -= np.nanmin(fes)

        im = ax.imshow(fes, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='viridis', vmax=25)
        ax.set_xlabel('Contacts')
        ax.set_ylabel('CV1')
        ax.set_title('Free Energy: CV1 vs Contacts')
        plt.colorbar(im, ax=ax, label='F (kJ/mol)')

    # FES: dZ vs contacts
    if 'dZ' in descriptor_cols and 'contacts' in descriptor_cols:
        ax = axes[2]
        h, xedges, yedges = np.histogram2d(X[:, dZ_idx], X[:, cont_idx], bins=50, density=True)
        h = np.ma.masked_where(h == 0, h)
        fes = -kb * temp * np.log(h.T)
        fes -= np.nanmin(fes)

        im = ax.imshow(fes, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='viridis', vmax=25)
        ax.set_xlabel('dZ (nm)')
        ax.set_ylabel('Contacts')
        ax.set_title('Free Energy: dZ vs Contacts')
        plt.colorbar(im, ax=ax, label='F (kJ/mol)')

    plt.tight_layout()
    plt.savefig(output_dir / 'free_energy_surfaces.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: free_energy_surfaces.png")

    # ===== FIGURE 6: Time Evolution of Key Features =====
    fig6, axes = plt.subplots(2, 2, figsize=(12, 8))

    key_features = ['dZ', 'contacts', 'rg_lig', 'nw']
    key_features = [f for f in key_features if f in descriptor_cols]

    for idx, feat in enumerate(key_features[:4]):
        ax = axes[idx // 2, idx % 2]
        feat_idx = descriptor_cols.index(feat)

        ax.plot(t/1000, X[:, feat_idx], 'b-', alpha=0.3, linewidth=0.5)

        # Running average
        window = min(500, len(t)//10)
        if window > 1:
            running_avg = np.convolve(X[:, feat_idx], np.ones(window)/window, mode='valid')
            t_avg = t[window//2:window//2+len(running_avg)]
            ax.plot(t_avg/1000, running_avg, 'r-', linewidth=2, label='Running avg')

        ax.set_xlabel('Time [ns]')
        ax.set_ylabel(feat)
        ax.set_title(f'{feat} Time Evolution')
        ax.grid(alpha=0.3)
        if window > 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_time_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: feature_time_evolution.png")

    return importance, correlations


# ==================== 6. PARAMETER SWEEP ====================
if args.sweep:
    print("\n5. Running parameter sweep...")

    sweep_params = {
        'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5],
        'lag_time': [25, 50, 100],
        'hidden_size': [32, 64, 128],
    }

    fixed_params = {
        'n_layers': args.layers,
        'n_cvs': args.n_cvs,
        'max_epochs': args.epochs,
        'batch_size': args.batch,
    }

    print(f"   Sweep parameters: {sweep_params}")
    print(f"   Fixed parameters: {fixed_params}")

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combinations = list(product(*param_values))

    print(f"   Total combinations: {len(combinations)}")

    sweep_results = []

    for i, combo in enumerate(combinations):
        config = {**fixed_params}
        for name, value in zip(param_names, combo):
            config[name] = value

        print(f"\n   [{i+1}/{len(combinations)}] Training: lr={config['learning_rate']}, "
              f"lag={config['lag_time']}, hidden={config['hidden_size']}")

        try:
            result = train_model(X, logweight, config, descriptor_cols, verbose=False)

            sweep_results.append({
                'config': config,
                'best_loss': result['best_loss'],
                'eigenvalues': result['eigenvalues'].tolist(),
                'cv_range': [float(result['cvs'][:, 0].min()), float(result['cvs'][:, 0].max())],
                'epochs': result['epochs'],
                'valid': all(e > 0 for e in result['eigenvalues']),
            })

            print(f"       Loss: {result['best_loss']:.6f}, Eigenvalue: {result['eigenvalues'][0]:.6f}, "
                  f"Valid: {'Yes' if sweep_results[-1]['valid'] else 'No'}")

        except Exception as e:
            print(f"       FAILED: {e}")
            sweep_results.append({
                'config': config,
                'best_loss': float('inf'),
                'eigenvalues': [],
                'cv_range': [0, 0],
                'epochs': 0,
                'valid': False,
                'error': str(e),
            })

    valid_results = [r for r in sweep_results if r['valid']]

    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_loss'])
        print(f"\n   Best configuration (lowest loss with valid eigenvalues):")
        print(f"   {best_result['config']}")
        print(f"   Loss: {best_result['best_loss']:.6f}")
        best_config = best_result['config']
    else:
        print("\n   WARNING: No configurations produced valid eigenvalues!")
        best_config = {
            'learning_rate': args.lr,
            'lag_time': args.lag,
            'hidden_size': args.hidden,
            'n_layers': args.layers,
            'n_cvs': args.n_cvs,
            'max_epochs': args.epochs,
            'batch_size': args.batch,
        }

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'sweep_results.json', 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"   Saved: {output_dir}/sweep_results.json")

else:
    best_config = {
        'learning_rate': args.lr,
        'lag_time': args.lag,
        'hidden_size': args.hidden,
        'n_layers': args.layers,
        'n_cvs': args.n_cvs,
        'max_epochs': args.epochs,
        'batch_size': args.batch,
    }

# ==================== 7. FINAL TRAINING ====================
print(f"\n6. Training final model with configuration:")
print(f"   {best_config}")

results = train_model(X, logweight, best_config, descriptor_cols, verbose=True)

print(f"\n   Training complete!")
print(f"   Best validation loss: {results['best_loss']:.6f}")
print(f"   Epochs: {results['epochs']}")
print(f"   Eigenvalues: {results['eigenvalues']}")

# Validate eigenvalues
if all(e > 0 for e in results['eigenvalues']):
    print("   Eigenvalues: VALID (all positive)")
else:
    print("   Eigenvalues: WARNING - some negative (may indicate issues)")

# ==================== 8. EXTENSIVE VISUALIZATION ====================
print("\n7. Creating desorption-focused visualizations...")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

importance, correlations = create_desorption_visualizations(
    results, X, t, colvar, descriptor_cols, output_dir
)

# ==================== 9. EXPORT MODEL ====================
print("\n8. Exporting model for PLUMED...")

model = results['model']
cvs = results['cvs']
n_input = X.shape[1]

class ModelForExport(nn.Module):
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

dummy_input = torch.randn(1, n_input)

with torch.no_grad():
    traced_model = torch.jit.trace(export_model, dummy_input)
    test_output = traced_model(dummy_input)
    print(f"   Model traced successfully")
    print(f"   Test output shape: {test_output.shape}")

model_path = output_dir / "model.ptc"
traced_model.save(str(model_path))
print(f"   Model saved: {model_path}")

# ==================== 10. SAVE METADATA ====================
print("\n9. Saving comprehensive metadata...")

metadata = {
    'training_type': 'UNBIASED (no reweighting)',
    'reason': 'OPES bias underestimated - reweighting would be inaccurate',
    'training_data': {
        'source': args.colvar,
        'frames': len(colvar),
        'duration_ns': float((colvar['time'].max() - colvar['time'].min())/1000),
        'effective_sample_size': len(colvar),
        'reweighting': 'None (uniform weights)',
    },
    'model': {
        'architecture': results['layers'],
        'n_cvs': best_config['n_cvs'],
        'learning_rate': best_config['learning_rate'],
        'lag_time': best_config['lag_time'],
        'batch_size': best_config['batch_size'],
        'epochs_trained': results['epochs'],
        'best_loss': float(results['best_loss']),
    },
    'eigenvalues': results['eigenvalues'].tolist(),
    'descriptors': descriptor_cols,
    'feature_importance': importance,
    'cv_range': {
        'min': float(cvs[:, 0].min()),
        'max': float(cvs[:, 0].max()),
        'mean': float(cvs[:, 0].mean()),
        'std': float(cvs[:, 0].std()),
    },
    'feature_stats': feature_stats,
    'correlations': correlations,
}

with open(output_dir / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"   Saved: model_metadata.json")

# ==================== 11. CREATE PRODUCTION PLUMED ====================
print("\n10. Creating production PLUMED file...")

cv_min = cvs[:, 0].min() - 0.5
cv_max = cvs[:, 0].max() + 0.5

plumed_production = f"""# PLUMED input for Production Run with DeepTICA CV
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Training: UNBIASED (no reweighting applied)
# Training loss: {results['best_loss']:.6f}
# CV1 eigenvalue: {results['eigenvalues'][0]:.6f}

UNITS LENGTH=nm ENERGY=kj/mol TIME=ps

# ==================== DEFINE GROUPS ====================
PE: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PE
LIG: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PIS_PUS_PTS
HOH: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=W

WHOLEMOLECULES ENTITY0=PE,LIG

COM_PE: COM ATOMS=PE
COM_LIG: COM ATOMS=LIG

# ==================== COMPUTE DESCRIPTORS ====================
# CRITICAL: These MUST be in the SAME ORDER as training!
# Order: {', '.join(descriptor_cols)}

"""

# Add descriptor definitions based on what was used
if 'dZ' in descriptor_cols:
    plumed_production += """dist_components: DISTANCE ATOMS=COM_LIG,COM_PE COMPONENTS
dZ: COMBINE ARG=dist_components.z PERIODIC=NO

"""

if 'contacts' in descriptor_cols:
    plumed_production += """contacts: COORDINATION GROUPA=LIG GROUPB=PE R_0=0.6 NN=6 MM=12

"""

if 'rg_lig' in descriptor_cols:
    plumed_production += """rg_lig: GYRATION TYPE=RADIUS ATOMS=LIG

"""

if 'asph_lig' in descriptor_cols:
    plumed_production += """asph_lig: GYRATION TYPE=ASPHERICITY ATOMS=LIG

"""

if 'acyl_lig' in descriptor_cols:
    plumed_production += """acyl_lig: GYRATION TYPE=ACYLINDRICITY ATOMS=LIG

"""

if 'dist_lig_pe' in descriptor_cols:
    plumed_production += """dist_lig_pe: DISTANCE ATOMS=COM_LIG,COM_PE

"""

if 'ree' in descriptor_cols:
    plumed_production += """FIRST_BEAD: GROUP ATOMS=17001
LAST_BEAD: GROUP ATOMS=17495
ree: DISTANCE ATOMS=FIRST_BEAD,LAST_BEAD

"""

if 'nw' in descriptor_cols:
    plumed_production += """nw: COORDINATION GROUPA=LIG GROUPB=HOH R_0=0.6 NN=6 MM=12

"""

plumed_production += f"""# ==================== DEEPTICA CV ====================
deep: PYTORCH_MODEL FILE=model.ptc ARG={','.join(descriptor_cols)}

cv1: COMBINE ARG=deep.node-0 PERIODIC=NO

# ==================== OPES ENHANCED SAMPLING ====================
opes: OPES_METAD ...
    ARG=cv1
    PACE=500
    SIGMA=0.1
    FILE=HILLS_production
    BARRIER=100
    BIASFACTOR=20
...

# ==================== OUTPUT ====================
PRINT STRIDE=100 FILE=COLVAR_PRODUCTION ARG={','.join(descriptor_cols)},cv1,opes.bias FMT=%12.6f

ENDPLUMED
"""

plumed_path = output_dir / "plumed-production.dat"
with open(plumed_path, 'w') as f:
    f.write(plumed_production)
print(f"   Saved: plumed-production.dat")

# ==================== 12. SUMMARY ====================
print("\n" + "="*80)
print("TRAINING COMPLETE (UNBIASED)")
print("="*80)

# Print desorption analysis
print("\nDesorption Analysis:")
if 'dZ' in descriptor_cols:
    dZ_idx = descriptor_cols.index('dZ')
    r_dZ = correlations['pearson']['CV1']['dZ']['r']
    print(f"  dZ correlation with CV1: r = {r_dZ:+.3f}")

if 'contacts' in descriptor_cols:
    r_cont = correlations['pearson']['CV1']['contacts']['r']
    print(f"  Contacts correlation with CV1: r = {r_cont:+.3f}")

# Print feature importance summary
print("\nFeature Importance (CV1):")
imp_cv1 = importance['CV1']
sorted_imp = sorted(imp_cv1.items(), key=lambda x: x[1], reverse=True)
total_imp = sum(imp_cv1.values())
for name, val in sorted_imp:
    pct = 100 * val / total_imp
    marker = '*' if name in ['dZ', 'contacts'] else ' '
    bar = '#' * int(pct / 2)
    print(f"  {marker}{name:15s}: {pct:5.1f}% {bar}")

print(f"""
Output Files ({output_dir}/):
  - model.ptc                    : Traced model for PLUMED
  - model_metadata.json          : Complete training metadata
  - plumed-production.dat        : Production PLUMED input
  - cv_evolution_distribution.png: CV time series and histograms
  - feature_importance.png       : Feature importance analysis
  - desorption_analysis.png      : dZ and contacts analysis
  - feature_correlations.png     : Correlation heatmaps
  - free_energy_surfaces.png     : 2D FES projections
  - feature_time_evolution.png   : Key feature time series

NOTE: This model was trained WITHOUT reweighting.
      For accurate FES, run production with proper OPES bias.
""")

print("="*80)
