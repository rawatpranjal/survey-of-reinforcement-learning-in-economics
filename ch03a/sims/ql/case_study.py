# Q-Learning Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks Q(s,a) table, visit counts, TD errors, action gaps.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_q_learning,
                                   v_to_array, policy_to_array)

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 5
GAMMA = 0.95
NUM_EPISODES = 500_000
HORIZON = 50

CHECKPOINTS = sorted(set(
    list(range(1, 11)) +
    list(range(20, 101, 10)) +
    list(range(200, 1001, 100)) +
    list(range(2000, 10001, 1000)) +
    list(range(20000, NUM_EPISODES + 1, 10000))
))

env = GridworldEnv(N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0,
                   symmetry_break=0.001)

# Get optimal V* from VI
V_vi, policy_vi, _ = run_value_iteration(env)
V_optimal = v_to_array(V_vi, env)
policy_optimal = policy_to_array(policy_vi, env)

print("=" * 60)
print("Q-LEARNING CASE STUDY")
print("=" * 60)

# Run Q-Learning with fine checkpoints
_, policy, metrics = run_q_learning(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.1, alpha_decay=0.99995,
    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
    V_optimal=V_optimal, policy_optimal=policy_optimal,
    optimal_return=9.30, eval_freq=500, eval_episodes=100,
    snapshot_episodes=CHECKPOINTS
)

print(f"Final return: {metrics.eval_returns[-1]:.2f}")
print(f"Final RMSE: {metrics.value_errors[-1]:.4f}")
print(f"Wall time: {metrics.wall_time:.2f}s")

# --- Figure 1: Action gaps at 4 selected states ---
# Pick states: (0,0) start, (2,2) center, (0,4) corner, (3,3) near goal
selected_states = [(0, 0), (2, 2), (0, 4), (3, 3)]
state_indices = [env.state_to_index(s) for s in selected_states]
action_names = ['Left', 'Right', 'Up', 'Down', 'Stay']

fig, axes = plt.subplots(2, 2, figsize=FIG_DOUBLE)
axes = axes.flatten()

snap_eps = sorted(metrics.q_table_snapshots.keys())

for idx, (s, s_idx) in enumerate(zip(selected_states, state_indices)):
    ax = axes[idx]
    for a in range(5):
        q_vals = [metrics.q_table_snapshots[ep][s_idx, a] for ep in snap_eps]
        ax.semilogx(snap_eps, q_vals, label=action_names[a], alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('$Q(s, a)$')
    ax.set_title(f'State {s}')
    if idx == 0:
        ax.legend(fontsize=7)

fig.suptitle('Q-Learning: Q-value evolution at selected states', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_action_gaps.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_action_gaps.png")

# --- Figure 2: Mean |TD error| per episode + epsilon schedule ---
fig, ax1 = plt.subplots(figsize=FIG_SINGLE)

# TD errors (use checkpoint-level data)
td_eps = list(range(1, len(metrics.mean_td_errors) + 1))
ax1.loglog(td_eps, metrics.mean_td_errors, color=ALGO_COLORS['Q-Learning'],
           alpha=0.3, linewidth=0.5)
# Smooth with rolling mean
window = max(1, len(metrics.mean_td_errors) // 200)
if window > 1:
    smoothed = np.convolve(metrics.mean_td_errors, np.ones(window)/window, mode='valid')
    ax1.loglog(np.arange(window//2 + 1, window//2 + 1 + len(smoothed)), smoothed,
               color=ALGO_COLORS['Q-Learning'], linewidth=1.5, label='Mean |TD error|')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Mean |TD error|', color=ALGO_COLORS['Q-Learning'])

# Epsilon on twin axis
ax2 = ax1.twinx()
if metrics.effective_epsilon:
    ckpt_eps = list(range(500, NUM_EPISODES + 1, 500))[:len(metrics.effective_epsilon)]
    ax2.semilogx(ckpt_eps, metrics.effective_epsilon, color='gray', ls='--',
                 alpha=0.6, label=r'$\epsilon$')
    ax2.set_ylabel(r'$\epsilon$', color='gray')

ax1.set_title('Q-Learning: TD Error Decay')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_td_errors.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_td_errors.png")

# --- Figure 3: Bellman residual at checkpoints ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
br_eps = sorted(metrics.bellman_residual_snapshots.keys())
br_vals = [metrics.bellman_residual_snapshots[ep] for ep in br_eps]
ax.semilogy(br_eps, br_vals, 'o-', color=ALGO_COLORS['Q-Learning'], markersize=2)
ax.set_xlabel('Episode')
ax.set_ylabel(r'$\max_{s,a} |Q(s,a) - \mathcal{T}^* Q(s,a)|$')
ax.set_title('Q-Learning: Bellman Residual')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_bellman_residual.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_bellman_residual.png")

# --- Figure 4: State visit heatmap ---
fig, ax = plt.subplots(figsize=(5, 4.5))
visit_grid = np.zeros((N, N))
for s, count in metrics.state_visit_counts.items():
    r, c = s
    visit_grid[r, c] = count

im = ax.imshow(np.log10(visit_grid + 1), cmap='YlOrRd', origin='upper')
for r in range(N):
    for c in range(N):
        val = int(visit_grid[r, c])
        color = 'white' if visit_grid[r, c] > visit_grid.max() * 0.6 else 'black'
        ax.text(c, r, f'{val//1000}k' if val >= 1000 else str(val),
                ha='center', va='center', fontsize=7, color=color)
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_title('Q-Learning: State Visit Counts (500K episodes)')
fig.colorbar(im, ax=ax, label='$\\log_{10}$(visits)')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_visit_heatmap.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_visit_heatmap.png")

# Print visit counts
print("\nState visit counts (thousands):")
for r in range(N):
    vals = '  '.join(f'{visit_grid[r, c]/1000:6.1f}k' for c in range(N))
    print(f"  row {r}: {vals}")

print("\nDone.")
