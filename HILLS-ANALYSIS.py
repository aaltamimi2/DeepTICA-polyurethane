#!/usr/bin/env python3
"""
Analyze OPES HILLS File - CORRECTED
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

print("="*80)
print("OPES HILLS FILE ANALYSIS")
print("="*80)

# ==================== 1. LOAD HILLS FILE ====================
print("\n1. Loading HILLS file...")

hills_file = 'HILLS_bootstrap'

# Parse header
metadata = {}
with open(hills_file, 'r') as f:
    for line in f:
        if line.startswith('#! SET'):
            parts = line.split()
            key = parts[2]
            # Handle the weird "1" on compression_threshold line
            value = parts[3] if len(parts) > 3 else None
            metadata[key] = value
        elif not line.startswith('#'):
            break

print(f"\nMetadata:")
for key, val in metadata.items():
    print(f"  {key:25s}: {val}")

# Load data - handle variable whitespace
data = pd.read_csv(hills_file, sep='\s+', comment='#', 
                   names=['time', 'dZ', 'sigma_dZ', 'height', 'logweight'])

# Remove any invalid rows
data = data.dropna()

n_kernels = len(data)
print(f"\n✓ Loaded {n_kernels:,} kernels")

# ==================== 2. BASIC STATISTICS ====================
print("\n2. Kernel Statistics:")
print(f"  Time range: {data['time'].min():.1f} - {data['time'].max():.1f} ps ({data['time'].max()/1000:.2f} ns)")
print(f"  dZ range: {data['dZ'].min():.3f} - {data['dZ'].max():.3f} nm")
print(f"  Sigma range: {data['sigma_dZ'].min():.3f} - {data['sigma_dZ'].max():.3f} nm")
print(f"  Height range: {data['height'].min():.6f} - {data['height'].max():.6f}")
print(f"  Total deposited bias: {data['height'].sum():.2f} kJ/mol")

# Deposition rate
dt = np.diff(data['time'].values)
mean_dt = np.mean(dt)
print(f"  Mean kernel spacing: {mean_dt:.1f} ps")
print(f"  Deposition rate: {1000/mean_dt:.1f} kernels/ns")

# ==================== 3. PLOT TIME EVOLUTION ====================
print("\n3. Creating plots...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: CV coverage over time
ax1 = plt.subplot(3, 4, 1)
scatter = ax1.scatter(data['time']/1000, data['dZ'], 
                     c=data['height'], s=10, alpha=0.6, cmap='viridis')
ax1.set_xlabel('Time (ns)', fontsize=10)
ax1.set_ylabel('dZ (nm)', fontsize=10)
ax1.set_title('Kernel Deposition Over Time', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Height')

# Plot 2: Kernel heights over time
ax2 = plt.subplot(3, 4, 2)
ax2.plot(data['time']/1000, data['height'], linewidth=0.8, alpha=0.7)
ax2.set_xlabel('Time (ns)', fontsize=10)
ax2.set_ylabel('Kernel Height', fontsize=10)
ax2.set_title('Kernel Heights (Well-Tempering)', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(alpha=0.3)

# Plot 3: Kernel widths (adaptive)
ax3 = plt.subplot(3, 4, 3)
ax3.plot(data['time']/1000, data['sigma_dZ'], linewidth=0.8, alpha=0.7, color='orange')
ax3.set_xlabel('Time (ns)', fontsize=10)
ax3.set_ylabel('Sigma (nm)', fontsize=10)
ax3.set_title('Kernel Width (Adaptive)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Cumulative bias
ax4 = plt.subplot(3, 4, 4)
cumulative_bias = np.cumsum(data['height'].values)
ax4.plot(data['time']/1000, cumulative_bias, linewidth=1.5, color='red')
ax4.set_xlabel('Time (ns)', fontsize=10)
ax4.set_ylabel('Cumulative Bias (kJ/mol)', fontsize=10)
ax4.set_title('Total Deposited Bias', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# Plot 5: dZ histogram (coverage)
ax5 = plt.subplot(3, 4, 5)
ax5.hist(data['dZ'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
ax5.set_xlabel('dZ (nm)', fontsize=10)
ax5.set_ylabel('Number of Kernels', fontsize=10)
ax5.set_title('CV Space Coverage', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3, axis='y')

# Plot 6: Kernel size distribution
ax6 = plt.subplot(3, 4, 6)
ax6.hist(data['sigma_dZ'], bins=50, alpha=0.7, edgecolor='black', color='orange')
ax6.set_xlabel('Sigma (nm)', fontsize=10)
ax6.set_ylabel('Count', fontsize=10)
ax6.set_title('Kernel Size Distribution', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3, axis='y')

# Plot 7: Height distribution
ax7 = plt.subplot(3, 4, 7)
ax7.hist(data['height'], bins=50, alpha=0.7, edgecolor='black', color='green')
ax7.set_xlabel('Height', fontsize=10)
ax7.set_ylabel('Count', fontsize=10)
ax7.set_title('Kernel Height Distribution', fontsize=12, fontweight='bold')
ax7.set_yscale('log')
ax7.grid(alpha=0.3, axis='y')

# Plot 8: Deposition rate over time - FIXED
ax8 = plt.subplot(3, 4, 8)
window = min(100, len(dt)//10)  # Adaptive window
if len(dt) > window:
    # Compute rates
    rates = 1000 / np.convolve(dt, np.ones(window)/window, mode='valid')
    # Match time array length
    times = data['time'].values[window//2 : window//2 + len(rates)] / 1000
    
    ax8.plot(times, rates, linewidth=1, alpha=0.7)
    ax8.set_xlabel('Time (ns)', fontsize=10)
    ax8.set_ylabel('Kernels/ns', fontsize=10)
    ax8.set_title('Deposition Rate', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'Not enough data', ha='center', va='center',
             transform=ax8.transAxes, fontsize=12)

# Plot 9: 2D histogram (time vs dZ)
ax9 = plt.subplot(3, 4, 9)
h, xedges, yedges = np.histogram2d(data['time']/1000, data['dZ'], bins=[50, 50])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax9.imshow(h.T, aspect='auto', origin='lower', extent=extent, cmap='hot')
ax9.set_xlabel('Time (ns)', fontsize=10)
ax9.set_ylabel('dZ (nm)', fontsize=10)
ax9.set_title('Exploration Density', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax9, label='Kernels')

# Plot 10: Reconstructed Free Energy
ax10 = plt.subplot(3, 4, 10)
dz_grid = np.linspace(data['dZ'].min()-0.5, data['dZ'].max()+0.5, 200)
fes = np.zeros_like(dz_grid)

# Sum Gaussians
for _, row in data.iterrows():
    gaussian = row['height'] * np.exp(-0.5 * ((dz_grid - row['dZ']) / row['sigma_dZ'])**2)
    fes += gaussian

# Convert to free energy
fes = -fes
fes -= fes.min()

ax10.plot(dz_grid, fes, linewidth=2, color='darkblue')
ax10.set_xlabel('dZ (nm)', fontsize=10)
ax10.set_ylabel('Free Energy (kJ/mol)', fontsize=10)
ax10.set_title('Approximate FES', fontsize=12, fontweight='bold')
ax10.grid(alpha=0.3)
ax10.axhline(0, color='k', linestyle='--', alpha=0.3)

# Plot 11: Coverage statistics
ax11 = plt.subplot(3, 4, 11)
bins = np.linspace(data['dZ'].min(), data['dZ'].max(), 30)
coverage, _ = np.histogram(data['dZ'], bins=bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

ax11.bar(bin_centers, coverage, width=np.diff(bins)[0]*0.9, 
         alpha=0.7, edgecolor='black', color='purple')
ax11.set_xlabel('dZ (nm)', fontsize=10)
ax11.set_ylabel('Visits', fontsize=10)
ax11.set_title('Sampling Distribution', fontsize=12, fontweight='bold')
ax11.grid(alpha=0.3, axis='y')

# Plot 12: Summary
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

well_sampled = coverage > coverage.mean()
coverage_percent = 100 * well_sampled.sum() / len(coverage)

summary = f"""
HILLS FILE SUMMARY

Kernels: {n_kernels:,}
Duration: {data['time'].max()/1000:.2f} ns
Rate: {n_kernels/(data['time'].max()/1000):.1f} kernels/ns

CV Space (dZ):
  Range: [{data['dZ'].min():.2f}, {data['dZ'].max():.2f}] nm
  Span: {data['dZ'].max() - data['dZ'].min():.2f} nm
  Coverage: {coverage_percent:.1f}%

Kernels:
  Height: {data['height'].min():.2e} to {data['height'].max():.2e}
  Sigma: {data['sigma_dZ'].min():.3f} to {data['sigma_dZ'].max():.3f} nm
  Total bias: {data['height'].sum():.1f} kJ/mol

Settings:
  Biasfactor: {metadata.get('biasfactor', 'N/A')}

Status:
  {"✓ Good coverage" if coverage_percent > 70 else "⚠ Needs more time"}
"""

ax12.text(0.05, 0.95, summary, transform=ax12.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('hills_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hills_analysis.png")

# ==================== 4. CONVERGENCE CHECK ====================
print("\n4. Convergence Analysis:")

dz_range = data['dZ'].max() - data['dZ'].min()
print(f"  CV range explored: {dz_range:.2f} nm")

n_half = len(data) // 2
early_range = data['dZ'].iloc[:n_half].max() - data['dZ'].iloc[:n_half].min()
late_range = data['dZ'].iloc[n_half:].max() - data['dZ'].iloc[n_half:].min()
print(f"  Early half: {early_range:.2f} nm")
print(f"  Late half: {late_range:.2f} nm")

if late_range > 0.9 * early_range:
    print("  ✓ Still exploring")
else:
    print("  ⚠ Converging")

early_heights = data['height'].iloc[:n_half].mean()
late_heights = data['height'].iloc[n_half:].mean()
print(f"  Early height: {early_heights:.3f}")
print(f"  Late height: {late_heights:.3f}")
print(f"  Reduction: {100*(1 - late_heights/early_heights):.1f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFiles created:")
print(f"  hills_analysis.png")