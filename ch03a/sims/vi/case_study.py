# Value Iteration Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks V(s) at each of 9 iterations, showing value propagation.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, CMAP_SEQ, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import GridworldEnv, run_value_iteration, v_to_array

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 5
GAMMA = 0.95
env = GridworldEnv(N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0,
                   symmetry_break=0.001)

# --- Run VI ---
V, policy, metrics = run_value_iteration(env)
V_optimal = v_to_array(V, env)

print("=" * 60)
print("VALUE ITERATION CASE STUDY")
print("=" * 60)
print(f"Iterations to convergence: {metrics.iterations}")
print(f"Final residual: {metrics.final_residual:.2e}")
print(f"Wall time: {metrics.wall_time:.4f}s")

# --- Figure 1: Bellman residual per iteration ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
iters = list(range(1, metrics.iterations + 1))
residuals = metrics.residual_history

# Theoretical bound: gamma^k * ||V_0 - V*||_inf
V0_dist = V_optimal.max()  # since V_0 = 0
theoretical = [V0_dist * GAMMA**k for k in iters]

ax.semilogy(iters, residuals, 'o-', color=ALGO_COLORS['VI'], label='Actual residual', markersize=6)
ax.semilogy(iters, theoretical, '--', color='gray', label=r'$\gamma^k \|V_0 - V^*\|_\infty$', alpha=0.7)
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Bellman residual $\|V_{k+1} - V_k\|_\infty$')
ax.set_title('Value Iteration: Bellman Residual')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'vi_bellman_residual.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: vi_bellman_residual.png")

# Print residual table
print("\nResidual per iteration:")
print(f"  {'Iter':>4s}  {'Residual':>12s}  {'gamma^k bound':>14s}")
for k, (r, t) in enumerate(zip(residuals, theoretical), 1):
    print(f"  {k:4d}  {r:12.6e}  {t:14.6e}")

# --- Figure 2: Value wavefront heatmaps ---
show_iters = [1, 3, 5, 7, 9]
# Only show iterations that exist
show_iters = [i for i in show_iters if i in metrics.value_snapshots_per_iter]
ncols = len(show_iters)

fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0))
if ncols == 1:
    axes = [axes]

vmin = 0.0
vmax = V_optimal.max()

for j, it in enumerate(show_iters):
    ax = axes[j]
    V_snap = metrics.value_snapshots_per_iter[it].reshape(N, N)
    im = ax.imshow(V_snap, cmap=CMAP_SEQ, vmin=vmin, vmax=vmax, origin='upper')
    for r in range(N):
        for c in range(N):
            val = V_snap[r, c]
            color = 'white' if val < vmax * 0.5 else 'black'
            ax.text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Iter {it}', fontsize=10)

fig.suptitle('Value Iteration: $V(s)$ at selected iterations', fontsize=11)
fig.tight_layout(rect=[0, 0, 0.92, 0.94])
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=CMAP_SEQ),
             cax=cbar_ax, label='$V(s)$')
fig.savefig(os.path.join(OUTPUT_DIR, 'vi_value_wavefront.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: vi_value_wavefront.png")

# Print V snapshots
for it in show_iters:
    V_snap = metrics.value_snapshots_per_iter[it].reshape(N, N)
    print(f"\nV(s) at iteration {it}:")
    for r in range(N):
        vals = '  '.join(f'{V_snap[r, c]:6.3f}' for c in range(N))
        print(f"  row {r}: {vals}")

# --- Figure 3: Convergence order heatmap ---
fig, ax = plt.subplots(figsize=(5, 4.5))

conv_order = np.full((N, N), metrics.iterations, dtype=int)
for it in sorted(metrics.value_snapshots_per_iter.keys()):
    V_snap = metrics.value_snapshots_per_iter[it]
    errors = np.abs(V_snap - V_optimal)
    for s_idx in range(env.num_states):
        r, c = env.index_to_state(s_idx)
        if conv_order[r, c] == metrics.iterations and errors[s_idx] < 0.1:
            conv_order[r, c] = it

im = ax.imshow(conv_order, cmap=CMAP_SEQ, origin='upper',
               vmin=1, vmax=metrics.iterations)
for r in range(N):
    for c in range(N):
        val = conv_order[r, c]
        color = 'white' if val > metrics.iterations * 0.6 else 'black'
        ax.text(c, r, str(val), ha='center', va='center', fontsize=10, color=color)
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_title('Iteration at which each state converges ($|V(s)-V^*(s)| < 0.1$)')
fig.colorbar(im, ax=ax, label='Convergence iteration')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'vi_convergence_order.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: vi_convergence_order.png")

# Print convergence order
print("\nConvergence order (iteration where |V(s)-V*(s)| < 0.1):")
for r in range(N):
    vals = '  '.join(f'{conv_order[r, c]:4d}' for c in range(N))
    print(f"  row {r}: {vals}")

# --- Summary table ---
lines = []
lines.append(r'\begin{tabular}{lr}')
lines.append(r'\hline')
lines.append(r'Metric & Value \\')
lines.append(r'\hline')
lines.append(f'Iterations & {metrics.iterations} \\\\')
lines.append(f'Final residual & {metrics.final_residual:.2e} \\\\')
lines.append(f'Wall time (s) & {metrics.wall_time:.4f} \\\\')
lines.append(f'MDP diameter & 8 \\\\')
lines.append(r'\hline')
lines.append(r'\end{tabular}')

tex_path = os.path.join(OUTPUT_DIR, 'vi_summary.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"Saved: vi_summary.tex")

print("\nDone.")
