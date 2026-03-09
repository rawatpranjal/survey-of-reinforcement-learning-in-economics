# Policy Iteration Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks pi(s) and V^pi(s) at each of 9 PI steps.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_policy_iteration,
                                   v_to_array, policy_to_array)

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 5
GAMMA = 0.95
np.random.seed(42)
env = GridworldEnv(N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0,
                   symmetry_break=0.001)

# Get optimal V* and pi* from VI
V_vi, policy_vi, _ = run_value_iteration(env)
V_optimal = v_to_array(V_vi, env)
policy_optimal = policy_to_array(policy_vi, env)

# Run PI
np.random.seed(42)
V_pi, policy_pi, metrics = run_policy_iteration(env)

print("=" * 60)
print("POLICY ITERATION CASE STUDY")
print("=" * 60)
print(f"Iterations to convergence: {metrics.iterations}")
print(f"Policy changes per iter: {metrics.policy_changes_per_iter}")
print(f"Wall time: {metrics.wall_time:.4f}s")

# --- Figure 1: Policy changes per iteration ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
iters = list(range(1, metrics.iterations + 1))
changes = metrics.policy_changes_per_iter

ax.bar(iters, changes, color=ALGO_COLORS['PI'], alpha=0.8, edgecolor='black', linewidth=0.5)
for i, c in enumerate(changes):
    ax.text(i + 1, c + 0.3, str(c), ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Policy Improvement Step')
ax.set_ylabel('States with changed action')
ax.set_title('Policy Iteration: Policy Changes per Step')
ax.set_xticks(iters)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pi_policy_changes.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: pi_policy_changes.png")

# --- Figure 2: Policy evolution arrow grids ---
show_iters = [1, 3, 5, 7, 9]
show_iters = [i for i in show_iters if i in metrics.policy_snapshots_per_iter]
ncols = len(show_iters) + 1  # +1 for optimal

arrow_dx = [0, 0, -1, 1, 0]   # Left, Right, Up, Down, Stay
arrow_dy = [-1, 1, 0, 0, 0]

fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0))

pi_opt = policy_optimal.reshape(N, N)

for j, it in enumerate(show_iters):
    ax = axes[j]
    pi_snap = metrics.policy_snapshots_per_iter[it].reshape(N, N)

    for r in range(N):
        for c in range(N):
            if (r, c) == env.terminal:
                ax.plot(c, r, 'ks', markersize=8)
                continue
            a = pi_snap[r, c]
            matches = (a == pi_opt[r, c])
            color = '#2ca02c' if matches else '#d62728'
            ax.annotate('', xy=(c + 0.3 * arrow_dy[a], r + 0.3 * arrow_dx[a]),
                        xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    n_match = sum(1 for r in range(N) for c in range(N)
                  if (r, c) != env.terminal and pi_snap[r, c] == pi_opt[r, c])
    ax.set_title(f'Iter {it} ({n_match}/24)', fontsize=9)

# Optimal column
ax = axes[-1]
for r in range(N):
    for c in range(N):
        if (r, c) == env.terminal:
            ax.plot(c, r, 'ks', markersize=8)
            continue
        a = pi_opt[r, c]
        ax.annotate('', xy=(c + 0.3 * arrow_dy[a], r + 0.3 * arrow_dx[a]),
                    xytext=(c, r),
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
ax.set_xlim(-0.5, N - 0.5)
ax.set_ylim(N - 0.5, -0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title(r'$\pi^*$', fontsize=10)

fig.suptitle('Policy Iteration: policy at selected steps (green=optimal, red=suboptimal)', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUTPUT_DIR, 'pi_policy_evolution.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: pi_policy_evolution.png")

# Print policy agreement per iteration
print("\nPolicy agreement per iteration:")
for it in sorted(metrics.policy_snapshots_per_iter.keys()):
    pi_snap = metrics.policy_snapshots_per_iter[it]
    n_match = sum(1 for s_idx in range(env.num_states)
                  if s_idx != env.state_to_index(env.terminal)
                  and pi_snap[s_idx] == policy_optimal[s_idx])
    print(f"  Iter {it}: {n_match}/24 states optimal ({100*n_match/24:.1f}%)")

# --- Figure 3: V^pi(s_0) at each PI iteration ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
v_s0_values = []
for it in sorted(metrics.value_snapshots_per_iter.keys()):
    v_s0_values.append(metrics.value_snapshots_per_iter[it][0])

ax.step(range(1, len(v_s0_values) + 1), v_s0_values, where='mid',
        color=ALGO_COLORS['PI'], linewidth=2)
ax.axhline(V_optimal[0], color='black', ls='--', lw=0.8, alpha=0.5,
           label=f'$V^*(s_0) = {V_optimal[0]:.2f}$')
ax.set_xlabel('Policy Improvement Step')
ax.set_ylabel('$V^\\pi(s_0)$')
ax.set_title('Policy Iteration: Value at Start State')
ax.legend()
ax.set_xticks(range(1, len(v_s0_values) + 1))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pi_value_jumps.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: pi_value_jumps.png")

# Print V(s0) per iteration
print("\nV^pi(s_0) per iteration:")
for it, v in zip(sorted(metrics.value_snapshots_per_iter.keys()), v_s0_values):
    print(f"  Iter {it}: {v:.4f}")
print(f"  V*(s_0): {V_optimal[0]:.4f}")

# --- Summary table ---
lines = []
lines.append(r'\begin{tabular}{lr}')
lines.append(r'\hline')
lines.append(r'Metric & Value \\')
lines.append(r'\hline')
lines.append(f'PI steps & {metrics.iterations} \\\\')
lines.append(f'Total policy changes & {sum(changes)} \\\\')
lines.append(f'Wall time (s) & {metrics.wall_time:.4f} \\\\')
lines.append(r'\hline')
lines.append(r'\end{tabular}')

tex_path = os.path.join(OUTPUT_DIR, 'pi_summary.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"Saved: pi_summary.tex")

print("\nDone.")
