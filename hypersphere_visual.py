import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')

# 1. Generate the Continuous World (The Smooth Failure Boundary)
radius = np.sqrt(65.0)
theta = np.linspace(0, np.pi/2, 200)  # Positive orthant only
x_cont = radius * np.cos(theta)
y_cont = radius * np.sin(theta)

ax.plot(x_cont, y_cont, color='#dc2626', linestyle='-', linewidth=3, label=r'Continuous Failure Surface ($\sum x_i^2 = 65.0$)')
ax.fill_between(x_cont, y_cont, 12, color='#fee2e2', alpha=0.4, label='Continuous Failure Region')

# 2. Generate the Discrete World (The Quantized Quantum Grid Mesh)
grid_points = [
    (1.0, 1.0, 'Safe State (k=0)', '#16a34a', 'o'),
    (4.2, 1.0, 'Safe State (k=1)', '#16a34a', 'o'),
    (1.0, 4.2, 'Safe State (k=1)', '#16a34a', 'o'),
    (4.2, 4.2, 'Grid Corner (k=2: Math=35.28 < 65)', '#16a34a', 's'),
    (7.5, 1.0, 'True Axis Breach (Continuous Spike)', '#2563eb', '^'),
    (1.0, 7.5, 'True Axis Breach (Continuous Spike)', '#2563eb', '^')
]

# Draw the pixelated hypercube boundary box lines to show the grid structure
ax.plot([1.0, 4.2, 4.2, 1.0, 1.0], [1.0, 1.0, 4.2, 4.2, 1.0], color='#64748b', linestyle='--', linewidth=1.5, label='Quantum Grid Mesh Boundary')

# Plot individual quantum basis states
for x, y, label, color, marker in grid_points:
    ax.scatter(x, y, color=color, s=120, marker=marker, edgecolors='black', zorder=5)
    ax.text(x + 0.15, y + 0.15, label, fontsize=9, fontweight='bold', color='#1e293b')

# 3. Annotate the "Smoothing" effect of the Continuous Distribution
ax.annotate('Continuous Probability Mass\nSmoothly Fills This Space\nBelow Threshold', 
            xy=(3.0, 3.0), xytext=(4.2, 6.0),
            fontweight='bold', color='#1e293b', fontsize=10,
            arrowprops=dict(facecolor='#475569', shrink=0.05, width=1.5, headwidth=7))

ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_xlabel(r'Environmental Load Variable $x_1$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Environmental Load Variable $x_2$', fontsize=12, fontweight='bold')
ax.set_title('VQIS Domain Mapping: Continuous Space vs. Discrete Grid Mesh', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0')

# Save high-resolution file
os.makedirs("figures", exist_ok=True)
output_path = "figures/cont_vs_discrete_space.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Success: Image saved to {output_path}")