# REINFORCE Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks gradient norms, policy entropy, baseline quality, return variance.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, CMAP_SEQ, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_reinforce,
                                   v_to_array, policy_to_array)

apply_style()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

N = 5
GAMMA = 0.95
NUM_EPISODES = 500_000
HORIZON = 50
EVAL_FREQ = 500

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
print("REINFORCE CASE STUDY")
print("=" * 60)

_, policy, metrics = run_reinforce(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.01, alpha_decay=0.99995, temperature=1.0, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=CHECKPOINTS
)

print(f"Final return: {metrics.eval_returns[-1]:.2f}")
print(f"Final RMSE: {metrics.value_errors[-1]:.4f}")
print(f"Wall time: {metrics.wall_time:.2f}s")

# --- Figure 1: Gradient norms per episode ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
n = len(metrics.gradient_norms)
step = max(1, n // 3000)
eps_sub = list(range(1, n + 1, step))
gn_sub = metrics.gradient_norms[::step]

ax.semilogy(eps_sub, gn_sub, color=ALGO_COLORS['REINFORCE'], alpha=0.2, linewidth=0.5)
window = max(1, len(gn_sub) // 100)
if window > 1 and len(gn_sub) > window:
    smoothed = np.convolve(gn_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax.semilogy(x_smooth, smoothed, color=ALGO_COLORS['REINFORCE'], linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Gradient norm')
ax.set_title('REINFORCE: Gradient Norm Over Training')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'reinforce_gradient_norms.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: reinforce_gradient_norms.png")

# --- Figure 2: Policy entropy ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ent_sub = metrics.policy_entropy_per_ep[::step]
ax.semilogx(eps_sub[:len(ent_sub)], ent_sub, color=ALGO_COLORS['REINFORCE'],
            alpha=0.2, linewidth=0.5)
if window > 1 and len(ent_sub) > window:
    smoothed = np.convolve(ent_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax.semilogx(x_smooth, smoothed, color=ALGO_COLORS['REINFORCE'], linewidth=1.5)

# Reference lines
ax.axhline(np.log(5), color='gray', ls='--', lw=0.8, alpha=0.5, label=f'$\\ln(5) = {np.log(5):.2f}$ (uniform)')
ax.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5, label='0 (deterministic)')
ax.set_xlabel('Episode')
ax.set_ylabel('Mean policy entropy')
ax.set_title('REINFORCE: Policy Entropy')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'reinforce_entropy.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: reinforce_entropy.png")

# --- Figure 3: Baseline V(s) heatmaps at selected episodes ---
show_eps = [10000, 100000, 500000]
show_eps = [ep for ep in show_eps if ep in metrics.value_snapshots]

if show_eps:
    ncols = len(show_eps)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.0))
    if ncols == 1:
        axes = [axes]

    vmin = 0.0
    vmax = V_optimal.max()

    for j, ep in enumerate(show_eps):
        ax = axes[j]
        V_snap = metrics.value_snapshots[ep].reshape(N, N)
        im = ax.imshow(V_snap, cmap=CMAP_SEQ, vmin=vmin, vmax=vmax, origin='upper')
        for r in range(N):
            for c in range(N):
                val = V_snap[r, c]
                color = 'white' if val < vmax * 0.5 else 'black'
                ax.text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        label = f'{ep//1000}K' if ep >= 1000 else str(ep)
        ax.set_title(f'Ep {label}', fontsize=10)

    fig.suptitle('REINFORCE: Baseline $V(s)$ at selected episodes', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=CMAP_SEQ),
                 cax=cbar_ax, label='$V(s)$')
    fig.savefig(os.path.join(OUTPUT_DIR, 'reinforce_baseline.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved: reinforce_baseline.png")

# --- Figure 4: Return variance ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
if metrics.return_std_window:
    ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))[:len(metrics.return_std_window)]
    ax.plot(ckpts, metrics.return_std_window, color=ALGO_COLORS['REINFORCE'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Std of returns (100-episode window)')
    ax.set_title('REINFORCE: Return Variance')
    ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'reinforce_return_variance.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: reinforce_return_variance.png")

print("\nDone.")
