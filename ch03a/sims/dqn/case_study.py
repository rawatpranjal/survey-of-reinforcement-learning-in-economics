# DQN Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks neural network Q-values, training loss, overestimation, target gap.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_dqn_tabular_comparison,
                                   v_to_array, policy_to_array)

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 5
GAMMA = 0.95
NUM_EPISODES = 500_000
HORIZON = 50
EVAL_FREQ = 500
TARGET_UPDATE_FREQ = 100

CHECKPOINTS = sorted(set(
    list(range(500, 10001, 500)) +
    list(range(20000, NUM_EPISODES + 1, 10000))
))

env = GridworldEnv(N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0,
                   symmetry_break=0.001)

V_vi, policy_vi, _ = run_value_iteration(env)
V_optimal = v_to_array(V_vi, env)
policy_optimal = policy_to_array(policy_vi, env)

print("=" * 60)
print("DQN CASE STUDY")
print("=" * 60)

_, policy, metrics = run_dqn_tabular_comparison(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100,
    replay_size=50000, batch_size=64, lr=1e-3,
    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999,
    target_update_freq=TARGET_UPDATE_FREQ, hidden_dim=64,
    snapshot_episodes=CHECKPOINTS
)

print(f"Final return: {metrics.eval_returns[-1]:.2f}")
print(f"Final RMSE: {metrics.value_errors[-1]:.4f}")
print(f"Wall time: {metrics.wall_time:.2f}s")

# --- Figure 1: Training loss per episode ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
n_loss = len(metrics.dqn_loss)
step = max(1, n_loss // 3000)
eps_sub = list(range(1, n_loss + 1, step))
loss_sub = metrics.dqn_loss[::step]

ax.semilogy(eps_sub, loss_sub, color=ALGO_COLORS['DQN'], alpha=0.2, linewidth=0.5)
# Smooth
window = max(1, len(loss_sub) // 100)
if window > 1 and len(loss_sub) > window:
    smoothed = np.convolve(loss_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax.semilogy(x_smooth, smoothed, color=ALGO_COLORS['DQN'], linewidth=1.5)

# Vertical lines at target network sync points
for sync_ep in range(TARGET_UPDATE_FREQ, min(5000, NUM_EPISODES), TARGET_UPDATE_FREQ):
    ax.axvline(sync_ep, color='gray', alpha=0.15, linewidth=0.5)

ax.set_xlabel('Episode')
ax.set_ylabel('Training Loss')
ax.set_title('DQN: Training Loss (vertical lines = target sync)')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'dqn_loss.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: dqn_loss.png")

# --- Figure 2: Overestimation bias ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
if metrics.q_overestimation:
    ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))[:len(metrics.q_overestimation)]
    ax.plot(ckpts, metrics.q_overestimation, color=ALGO_COLORS['DQN'])
    ax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Mean $[\max_a Q_{net}(s,a) - V^*(s)]$')
    ax.set_title('DQN: Overestimation Bias')
    ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'dqn_overestimation.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: dqn_overestimation.png")

# --- Figure 3: Target network gap ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
if metrics.target_network_gap:
    ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))[:len(metrics.target_network_gap)]
    ax.semilogy(ckpts, metrics.target_network_gap, color=ALGO_COLORS['DQN'])

    # Vertical lines at sync points (first 50)
    for sync_ep in range(TARGET_UPDATE_FREQ, min(NUM_EPISODES, 50000), TARGET_UPDATE_FREQ):
        ax.axvline(sync_ep, color='gray', alpha=0.1, linewidth=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\|Q_{net} - Q_{target}\|$')
    ax.set_title('DQN: Online vs Target Network Gap')
    ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'dqn_target_gap.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: dqn_target_gap.png")

# --- Figure 4: Scatter: learned Q vs V* ---
fig, ax = plt.subplots(figsize=(6, 6))
if metrics.value_snapshots:
    last_ep = max(metrics.value_snapshots.keys())
    V_learned = metrics.value_snapshots[last_ep]

    ax.scatter(V_optimal, V_learned, c=ALGO_COLORS['DQN'], alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    # Identity line
    vmin = min(V_optimal.min(), V_learned.min()) - 0.5
    vmax = max(V_optimal.max(), V_learned.max()) + 0.5
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.8, alpha=0.5)

    # Annotate worst error
    errors = np.abs(V_learned - V_optimal)
    worst_idx = np.argmax(errors)
    ax.annotate(f'{env.index_to_state(worst_idx)}\nerr={errors[worst_idx]:.2f}',
                xy=(V_optimal[worst_idx], V_learned[worst_idx]),
                fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('$V^*(s)$')
    ax.set_ylabel('$\\max_a Q_{DQN}(s,a)$')
    ax.set_title(f'DQN: Learned vs Optimal Values (ep {last_ep})')
    ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'dqn_scatter.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: dqn_scatter.png")

print("\nDone.")
