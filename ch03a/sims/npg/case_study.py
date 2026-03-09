# Natural Policy Gradient Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks critic quality, gradient norms vs REINFORCE, entropy, theta evolution.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_reinforce,
                                   run_npg, v_to_array, policy_to_array)

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
print("NPG CASE STUDY")
print("=" * 60)

print("\n[NPG]")
_, _, m_npg = run_npg(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    eta=0.2, eta_decay=0.99995, temperature=1.0,
    alpha_critic=0.5, alpha_critic_decay=0.99995, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=CHECKPOINTS
)
print(f"  Final return: {m_npg.eval_returns[-1]:.2f}")
print(f"  Final RMSE: {m_npg.value_errors[-1]:.4f}")

print("\n[REINFORCE (reference)]")
_, _, m_reinforce = run_reinforce(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.01, alpha_decay=0.99995, temperature=1.0, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=CHECKPOINTS
)
print(f"  Final return: {m_reinforce.eval_returns[-1]:.2f}")

# --- Figure 1: Critic quality (RMSE of V_critic vs V*) ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

n = min(len(ckpts), len(m_npg.value_errors))
ax.semilogy(ckpts[:n], m_npg.value_errors[:n], color=ALGO_COLORS['NPG'], label='NPG critic')
n_r = min(len(ckpts), len(m_reinforce.value_errors))
ax.semilogy(ckpts[:n_r], m_reinforce.value_errors[:n_r], color=ALGO_COLORS['REINFORCE'],
            label='REINFORCE baseline', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('RMSE $V$ vs $V^*$')
ax.set_title('NPG: Critic Quality Over Training')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'npg_critic_quality.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: npg_critic_quality.png")

# --- Figure 2: Gradient norms: NPG vs REINFORCE ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

for m, label, color in [
    (m_npg, 'NPG', ALGO_COLORS['NPG']),
    (m_reinforce, 'REINFORCE', ALGO_COLORS['REINFORCE']),
]:
    n = len(m.gradient_norms)
    step = max(1, n // 2000)
    eps_sub = list(range(1, n + 1, step))
    gn_sub = m.gradient_norms[::step]
    window = max(1, len(gn_sub) // 80)
    if window > 1 and len(gn_sub) > window:
        smoothed = np.convolve(gn_sub, np.ones(window)/window, mode='valid')
        x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
        ax.semilogy(x_smooth, smoothed, color=color, linewidth=1.5, label=label)

ax.set_xlabel('Episode')
ax.set_ylabel('Gradient norm (smoothed)')
ax.set_title('Gradient Norms: NPG vs REINFORCE')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'npg_gradient_comparison.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: npg_gradient_comparison.png")

# --- Figure 3: Entropy comparison ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

for m, label, color in [
    (m_npg, 'NPG', ALGO_COLORS['NPG']),
    (m_reinforce, 'REINFORCE', ALGO_COLORS['REINFORCE']),
]:
    n = len(m.policy_entropy_per_ep)
    step = max(1, n // 2000)
    eps_sub = list(range(1, n + 1, step))
    ent_sub = m.policy_entropy_per_ep[::step]
    window = max(1, len(ent_sub) // 80)
    if window > 1 and len(ent_sub) > window:
        smoothed = np.convolve(ent_sub, np.ones(window)/window, mode='valid')
        x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
        ax.semilogx(x_smooth, smoothed, color=color, linewidth=1.5, label=label)

ax.axhline(np.log(5), color='gray', ls='--', lw=0.8, alpha=0.5, label=f'$\\ln(5)$ (uniform)')
ax.set_xlabel('Episode')
ax.set_ylabel('Mean policy entropy')
ax.set_title('Policy Entropy: NPG vs REINFORCE')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'npg_entropy_comparison.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: npg_entropy_comparison.png")

# --- Figure 4: Theta evolution at state (2,2) ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
s_idx = env.state_to_index((2, 2))
action_names = ['Left', 'Right', 'Up', 'Down', 'Stay']

snap_eps = sorted(m_npg.theta_snapshots.keys())
if snap_eps:
    for a in range(5):
        theta_vals = [m_npg.theta_snapshots[ep][s_idx, a] for ep in snap_eps]
        ax.semilogx(snap_eps, theta_vals, label=action_names[a], alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\theta(s, a)$')
    ax.set_title(r'NPG: $\theta$ evolution at state (2,2)')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'No theta snapshots available', transform=ax.transAxes,
            ha='center', va='center')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'npg_theta_evolution.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: npg_theta_evolution.png")

print("\nDone.")
