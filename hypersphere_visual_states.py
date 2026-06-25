import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

# 1. Plot Continuous Reality
radius = np.sqrt(65.0)
theta = np.linspace(0, np.pi/2, 200)
ax.plot(radius * np.cos(theta), radius * np.sin(theta), color='#dc2626', linestyle='-', linewidth=3, label='True Continuous Failure Boundary')
ax.fill_between(radius * np.cos(theta), radius * np.sin(theta), 12, color='#fee2e2', alpha=0.4)

# 2. Plot the Mesh Structure
ax.plot([1.0, 4.2, 4.2, 1.0, 1.0], [1.0, 1.0, 4.2, 4.2, 1.0], color='#94a3b8', linestyle='--', linewidth=1.5, label='Quantum Grid Mesh')

# 3. Plot Quantum Grid Points Colored by Mask Classification (k_crit = 5.5)
# Coordinates: (x1, x2, simulated_total_k, color, label)
quantum_states = [
    (1.0, 1.0, 0, '#16a34a', '(k=0): SAFE'),
    (4.2, 1.0, 1, '#16a34a', '(k=1): SAFE'),
    (1.0, 4.2, 1, '#16a34a', '(k=1): SAFE'),
    (4.2, 4.2, 2, '#16a34a', '(k=2): SAFE'),
    # Let's plot a point that has 4 hidden background spikes active (Total k = 6)
    #(4.2, 4.2, 6, '#dc2626', 'Hidden Layer State (k=6): FAILURE')
]

for x, y, k, color, label in quantum_states:
    ax.scatter(x, y, color=color, s=150, edgecolors='black', linewidth=1.5, zorder=5)
    ax.text(x + 0.15, y + 0.15, label, fontsize=9, fontweight='bold', color='#1e293b')

# 4. Plot the Single-Variable Hidden 10D Failure (Blue Triangle)
ax.scatter(7.5, 1.0, color='#2563eb', s=150, marker='^', edgecolors='black', linewidth=1.5, zorder=5, label='Continuous 10D Failure Axis')

# Formatting
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_xlabel(r'Environmental Load Variable $x_1$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Environmental Load Variable $x_2$', fontsize=12, fontweight='bold')
ax.set_title('Quantum Classification Mask Layer (Green=Safe, Red=Failure)', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0')

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/quantum_mask_visual.png", dpi=300, bbox_inches='tight')
plt.close()
print("Successfully generated the colored mask visualization!")