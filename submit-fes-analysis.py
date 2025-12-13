#!/usr/bin/env python3
"""
Analyze Production Run and Compute Free Energy Surface
=======================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats

print("="*80)
print("PRODUCTION RUN ANALYSIS")
print("="*80)

# ==================== 1. LOAD DATA ====================
print("\n1. Loading COLVAR_PRODUCTION...")

colvar = pd.read_csv('COLVAR_PRODUCTION', sep='\s+', comment='#',
                     names=['time', 'rg_lig', 'asph_lig', 'acyl_lig', 'dist_lig_pe', 
                            'dZ', 'cv1', 'ene', 'ecv.ene', 'umb.cv1', 'opes.bias'])

print(f"✓ Loaded {len(colvar):,} frames")
print(f"  Time range: {colvar['time'].min():.1f} - {colvar['time'].max():.1f} ps")
print(f"  Duration: {(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns")

# ==================== 2. BASIC STATISTICS ====================
print("\n2. Sampling Statistics:")
print(f"  CV1 range: [{colvar['cv1'].min():.3f}, {colvar['cv1'].max():.3f}]")
print(f"  Distance range: [{colvar['dist_lig_pe'].min():.2f}, {colvar['dist_lig_pe'].max():.2f}] nm")
print(f"  Temperature range: {colvar['ecv.ene'].min():.1f} - {colvar['ecv.ene'].max():.1f} K (effective)")
print(f"  Max bias: {colvar['opes.bias'].max():.2f} kJ/mol")

# Check if desorption occurred
max_dist = colvar['dist_lig_pe'].max()
if max_dist > 8.0:
    print(f"  ✓ Desorption observed! (max distance: {max_dist:.2f} nm)")
else:
    print(f"  ⚠️ Limited desorption (max distance: {max_dist:.2f} nm)")

# ==================== 3. COMPUTE FREE ENERGY ====================
print("\n3. Computing Free Energy Surface...")

# Constants
kB = 0.008314  # kJ/(mol·K)
T = 300.0      # K
beta = 1.0 / (kB * T)

# Method 1: Direct reweighting from OPES bias
# F(s) = -1/beta * log P(s), where P(s) is reweighted histogram

# Compute weights for reweighting
weights = np.exp(beta * colvar['opes.bias'])
weights = weights / weights.sum()  # Normalize

# Create CV bins
cv1_bins = np.linspace(colvar['cv1'].min() - 0.1, colvar['cv1'].max() + 0.1, 50)
cv1_centers = (cv1_bins[:-1] + cv1_bins[1:]) / 2

# Compute reweighted histogram
hist_reweighted, _ = np.histogram(colvar['cv1'], bins=cv1_bins, weights=weights, density=True)

# Compute free energy
# F = -kT ln(P)
# Add small constant to avoid log(0)
hist_reweighted = np.maximum(hist_reweighted, 1e-10)
fes = -kB * T * np.log(hist_reweighted)

# Shift minimum to zero
fes = fes - fes.min()

print(f"✓ FES computed")
print(f"  Free energy range: 0 - {fes.max():.2f} kJ/mol")
print(f"  Number of bins: {len(cv1_bins)-1}")

# ==================== 4. IDENTIFY METASTABLE STATES ====================
print("\n4. Identifying Metastable States...")

# Find local minima (states)
# Smooth FES first
fes_smooth = gaussian_filter1d(fes, sigma=2)

# Find minima (where derivative changes sign)
minima_idx = []
for i in range(1, len(fes_smooth)-1):
    if fes_smooth[i] < fes_smooth[i-1] and fes_smooth[i] < fes_smooth[i+1]:
        if fes_smooth[i] < 15:  # Only significant minima
            minima_idx.append(i)

print(f"  Found {len(minima_idx)} metastable states:")
for i, idx in enumerate(minima_idx):
    cv_val = cv1_centers[idx]
    fe_val = fes_smooth[idx]
    # Find corresponding distance
    mask = np.abs(colvar['cv1'] - cv_val) < 0.1
    avg_dist = colvar.loc[mask, 'dist_lig_pe'].mean() if mask.sum() > 0 else np.nan
    print(f"    State {i+1}: CV1={cv_val:.3f}, F={fe_val:.2f} kJ/mol, dist≈{avg_dist:.2f} nm")

# ==================== 5. COMPUTE BARRIERS ====================
if len(minima_idx) >= 2:
    print("\n5. Transition Barriers:")
    for i in range(len(minima_idx)-1):
        idx1 = minima_idx[i]
        idx2 = minima_idx[i+1]
        # Find barrier between states
        barrier_idx = idx1 + np.argmax(fes_smooth[idx1:idx2+1])
        barrier_height = fes_smooth[barrier_idx] - max(fes_smooth[idx1], fes_smooth[idx2])
        print(f"    State {i+1} → State {i+2}: {barrier_height:.2f} kJ/mol")

# ==================== 6. CREATE COMPREHENSIVE PLOTS ====================
print("\n6. Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Free Energy Surface
ax1 = plt.subplot(3, 4, 1)
ax1.plot(cv1_centers, fes, 'b-', linewidth=2, label='FES')
ax1.plot(cv1_centers, fes_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Smoothed')
# Mark minima
for idx in minima_idx:
    ax1.plot(cv1_centers[idx], fes_smooth[idx], 'ro', markersize=10)
ax1.set_xlabel('DeepTICA CV1', fontsize=12)
ax1.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
ax1.set_title('Free Energy Surface', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim([0, min(50, fes.max()*1.1)])

# Plot 2: CV1 Time Evolution
ax2 = plt.subplot(3, 4, 2)
ax2.plot(colvar['time']/1000, colvar['cv1'], linewidth=0.5, alpha=0.7)
ax2.set_xlabel('Time (ns)', fontsize=12)
ax2.set_ylabel('CV1', fontsize=12)
ax2.set_title('CV1 Time Evolution', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: CV1 Distribution
ax3 = plt.subplot(3, 4, 3)
ax3.hist(colvar['cv1'], bins=50, alpha=0.7, edgecolor='black', density=True)
ax3.axvline(colvar['cv1'].mean(), color='r', linestyle='--', 
            label=f'Mean: {colvar["cv1"].mean():.3f}')
ax3.set_xlabel('CV1', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title('CV1 Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# Plot 4: Distance Time Evolution
ax4 = plt.subplot(3, 4, 4)
ax4.plot(colvar['time']/1000, colvar['dist_lig_pe'], linewidth=0.5, alpha=0.7)
ax4.axhline(8.0, color='r', linestyle='--', label='Desorption threshold')
ax4.set_xlabel('Time (ns)', fontsize=12)
ax4.set_ylabel('Distance (nm)', fontsize=12)
ax4.set_title('Ligand-Surface Distance', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 5: CV1 vs Distance (2D FES)
ax5 = plt.subplot(3, 4, 5)
# Create 2D histogram with reweighting
H, xedges, yedges = np.histogram2d(colvar['cv1'], colvar['dist_lig_pe'], 
                                    bins=50, weights=weights, density=True)
H = np.maximum(H, 1e-10)
fes_2d = -kB * T * np.log(H.T)
fes_2d = fes_2d - fes_2d.min()
fes_2d = np.minimum(fes_2d, 30)  # Cap at 30 kJ/mol for visualization

im = ax5.contourf((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2, 
                   fes_2d, levels=15, cmap='jet')
plt.colorbar(im, ax=ax5, label='Free Energy (kJ/mol)')
ax5.set_xlabel('CV1', fontsize=12)
ax5.set_ylabel('Distance (nm)', fontsize=12)
ax5.set_title('2D Free Energy Surface', fontsize=14, fontweight='bold')

# Plot 6: Bias Evolution
ax6 = plt.subplot(3, 4, 6)
ax6.plot(colvar['time']/1000, colvar['opes.bias'], linewidth=0.5, alpha=0.7)
ax6.set_xlabel('Time (ns)', fontsize=12)
ax6.set_ylabel('OPES Bias (kJ/mol)', fontsize=12)
ax6.set_title('Bias Evolution', fontsize=14, fontweight='bold')
ax6.grid(alpha=0.3)

# Plot 7: Temperature Sampling
ax7 = plt.subplot(3, 4, 7)
# Approximate temperature from energy
# This is simplified - actual temp requires proper conversion
temp_approx = 300 + (colvar['ecv.ene'] - colvar['ecv.ene'].min()) / colvar['ecv.ene'].std() * 100
ax7.hist(temp_approx, bins=30, alpha=0.7, edgecolor='black')
ax7.set_xlabel('Effective Temperature (K)', fontsize=12)
ax7.set_ylabel('Count', fontsize=12)
ax7.set_title('Temperature Sampling', fontsize=14, fontweight='bold')
ax7.grid(alpha=0.3, axis='y')

# Plot 8: Rg vs CV1
ax8 = plt.subplot(3, 4, 8)
ax8.scatter(colvar['cv1'], colvar['rg_lig'], s=1, alpha=0.3, c=colvar['time'], cmap='viridis')
ax8.set_xlabel('CV1', fontsize=12)
ax8.set_ylabel('Rg (nm)', fontsize=12)
ax8.set_title('Rg vs CV1', fontsize=14, fontweight='bold')
ax8.grid(alpha=0.3)

# Plot 9: Free Energy vs Distance
ax9 = plt.subplot(3, 4, 9)
dist_bins = np.linspace(colvar['dist_lig_pe'].min(), colvar['dist_lig_pe'].max(), 40)
dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
hist_dist, _ = np.histogram(colvar['dist_lig_pe'], bins=dist_bins, weights=weights, density=True)
hist_dist = np.maximum(hist_dist, 1e-10)
fes_dist = -kB * T * np.log(hist_dist)
fes_dist = fes_dist - fes_dist.min()
ax9.plot(dist_centers, fes_dist, 'b-', linewidth=2)
ax9.set_xlabel('Distance (nm)', fontsize=12)
ax9.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
ax9.set_title('FES vs Distance', fontsize=14, fontweight='bold')
ax9.grid(alpha=0.3)
ax9.set_ylim([0, min(30, fes_dist.max()*1.1)])

# Plot 10: Sampling Quality (block analysis)
ax10 = plt.subplot(3, 4, 10)
n_blocks = 10
block_size = len(colvar) // n_blocks
cv1_block_means = []
for i in range(n_blocks):
    block_data = colvar['cv1'].iloc[i*block_size:(i+1)*block_size]
    cv1_block_means.append(block_data.mean())
ax10.plot(range(1, n_blocks+1), cv1_block_means, 'o-', markersize=8)
ax10.axhline(colvar['cv1'].mean(), color='r', linestyle='--', label='Overall mean')
ax10.set_xlabel('Block Number', fontsize=12)
ax10.set_ylabel('Mean CV1', fontsize=12)
ax10.set_title('Convergence (Block Analysis)', fontsize=14, fontweight='bold')
ax10.legend()
ax10.grid(alpha=0.3)

# Plot 11: Trajectory in CV space
ax11 = plt.subplot(3, 4, 11)
scatter = ax11.scatter(colvar['cv1'], colvar['rg_lig'], 
                       c=colvar['time']/1000, s=2, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, ax=ax11, label='Time (ns)')
ax11.set_xlabel('CV1', fontsize=12)
ax11.set_ylabel('Rg (nm)', fontsize=12)
ax11.set_title('Trajectory in CV Space', fontsize=14, fontweight='bold')

# Plot 12: Desorption Events
ax12 = plt.subplot(3, 4, 12)
# Count transitions above threshold
threshold = 8.0
above_threshold = colvar['dist_lig_pe'] > threshold
transitions = np.diff(above_threshold.astype(int))
desorption_times = colvar['time'].iloc[np.where(transitions == 1)[0]].values / 1000
adsorption_times = colvar['time'].iloc[np.where(transitions == -1)[0]].values / 1000

ax12.axhline(threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
ax12.plot(colvar['time']/1000, colvar['dist_lig_pe'], linewidth=0.5, alpha=0.7)
for t in desorption_times:
    ax12.axvline(t, color='g', alpha=0.3, linewidth=1)
for t in adsorption_times:
    ax12.axvline(t, color='b', alpha=0.3, linewidth=1)
ax12.set_xlabel('Time (ns)', fontsize=12)
ax12.set_ylabel('Distance (nm)', fontsize=12)
ax12.set_title(f'Desorption Events (n={len(desorption_times)})', fontsize=14, fontweight='bold')
ax12.legend()
ax12.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('production_fes_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: production_fes_analysis.png")

# ==================== 7. SAVE FES DATA ====================
print("\n7. Saving FES data...")

fes_data = pd.DataFrame({
    'cv1': cv1_centers,
    'fes_kj_mol': fes,
    'fes_smooth': fes_smooth,
    'probability': hist_reweighted / hist_reweighted.sum()
})
fes_data.to_csv('fes_cv1.dat', sep='\t', index=False, float_format='%.6f')
print("✓ Saved: fes_cv1.dat")

# Save 2D FES
np.savetxt('fes_2d.dat', fes_2d, fmt='%.6f')
print("✓ Saved: fes_2d.dat")

# ==================== 8. SUMMARY REPORT ====================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

summary = f"""
Simulation Details:
  • Duration: {(colvar['time'].max() - colvar['time'].min())/1000:.2f} ns
  • Frames: {len(colvar):,}
  • Sampling interval: {(colvar['time'].iloc[1] - colvar['time'].iloc[0]):.2f} ps

CV1 Statistics:
  • Range: [{colvar['cv1'].min():.3f}, {colvar['cv1'].max():.3f}]
  • Mean: {colvar['cv1'].mean():.3f} ± {colvar['cv1'].std():.3f}
  • Training range: [-0.673, 0.195]
  • Exploration: {'✓ Good' if colvar['cv1'].max() > 0.15 and colvar['cv1'].min() < -0.6 else '⚠️ Limited'}

Distance Statistics:
  • Range: [{colvar['dist_lig_pe'].min():.2f}, {colvar['dist_lig_pe'].max():.2f}] nm
  • Mean: {colvar['dist_lig_pe'].mean():.2f} ± {colvar['dist_lig_pe'].std():.2f} nm
  • Desorption events: {len(desorption_times)}
  • Max separation: {colvar['dist_lig_pe'].max():.2f} nm

Free Energy Landscape:
  • Number of states: {len(minima_idx)}
  • Maximum barrier: {fes.max():.2f} kJ/mol
  • Deepest minimum: {cv1_centers[fes.argmin()]:.3f} (CV1)

Enhanced Sampling Performance:
  • Temperature range: 300-600 K (multithermal)
  • Max bias deposited: {colvar['opes.bias'].max():.2f} kJ/mol
  • CV exploration: {((colvar['cv1'].max() - colvar['cv1'].min()) / (0.195 - (-0.673)) * 100):.1f}% of training range

Files Generated:
  ✓ production_fes_analysis.png  (comprehensive analysis)
  ✓ fes_cv1.dat                  (1D free energy data)
  ✓ fes_2d.dat                   (2D free energy surface)
"""

print(summary)

# Save report
with open('production_analysis_report.txt', 'w') as f:
    f.write("PRODUCTION RUN ANALYSIS REPORT\n")
    f.write("="*80 + "\n")
    f.write(summary)

print("✓ Saved: production_analysis_report.txt")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)