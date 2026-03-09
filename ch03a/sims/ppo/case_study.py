# PPO Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks clip fractions, importance ratios, entropy, convergence.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_ppo,
                                   run_reinforce, run_npg,
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
print("PPO CASE STUDY")
print("=" * 60)

print("\n[PPO]")
_, _, m_ppo = run_ppo(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.01, alpha_decay=0.99995, clip_ratio=0.2,
    n_epochs=4, gae_lambda=0.95, temperature=1.0, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=CHECKPOINTS
)
print(f"  Final return: {m_ppo.eval_returns[-1]:.2f}")
print(f"  Final RMSE: {m_ppo.value_errors[-1]:.4f}")
print(f"  Wall time: {m_ppo.wall_time:.2f}s")

# Also run REINFORCE and NPG for entropy comparison
print("\n[REINFORCE (reference)]")
_, _, m_reinforce = run_reinforce(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.01, alpha_decay=0.99995, temperature=1.0, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=None
)
print(f"  Final return: {m_reinforce.eval_returns[-1]:.2f}")

print("\n[NPG (reference)]")
_, _, m_npg = run_npg(
    env, num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    eta=0.2, eta_decay=0.99995, temperature=1.0,
    alpha_critic=0.5, alpha_critic_decay=0.99995, baseline=True,
    V_optimal=V_optimal, policy_optimal=policy_optimal, optimal_return=9.30,
    eval_freq=EVAL_FREQ, eval_episodes=100, snapshot_episodes=None
)
print(f"  Final return: {m_npg.eval_returns[-1]:.2f}")

# --- Figure 1: Clip fractions ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
n = len(m_ppo.clip_fractions)
step = max(1, n // 3000)
eps_sub = list(range(1, n + 1, step))
cf_sub = m_ppo.clip_fractions[::step]

ax.semilogx(eps_sub, cf_sub, color=ALGO_COLORS['PPO'], alpha=0.2, linewidth=0.5)
window = max(1, len(cf_sub) // 100)
if window > 1 and len(cf_sub) > window:
    smoothed = np.convolve(cf_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax.semilogx(x_smooth, smoothed, color=ALGO_COLORS['PPO'], linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Fraction of steps clipped')
ax.set_title('PPO: Clip Fraction Over Training')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ppo_clip_fractions.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ppo_clip_fractions.png")

# --- Figure 2: Importance ratios ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

ratio_mean_sub = m_ppo.ppo_ratios_mean[::step]
ratio_std_sub = m_ppo.ppo_ratios_std[::step]

# Smooth
if window > 1 and len(ratio_mean_sub) > window:
    mean_smooth = np.convolve(ratio_mean_sub, np.ones(window)/window, mode='valid')
    std_smooth = np.convolve(ratio_std_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(mean_smooth)]
    ax.semilogx(x_smooth, mean_smooth, color=ALGO_COLORS['PPO'], linewidth=1.5, label='Mean ratio')
    ax.fill_between(x_smooth,
                     np.array(mean_smooth) - np.array(std_smooth),
                     np.array(mean_smooth) + np.array(std_smooth),
                     alpha=0.2, color=ALGO_COLORS['PPO'])
else:
    ax.semilogx(eps_sub[:len(ratio_mean_sub)], ratio_mean_sub,
                color=ALGO_COLORS['PPO'], linewidth=1.5, label='Mean ratio')

ax.axhline(1.0 + 0.2, color='gray', ls='--', lw=0.8, alpha=0.5, label='Clip bounds')
ax.axhline(1.0 - 0.2, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.axhline(1.0, color='black', ls=':', lw=0.5, alpha=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel(r'Importance ratio $\pi_{new}/\pi_{old}$')
ax.set_title(r'PPO: Importance Sampling Ratios ($\pm$ 1 std)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ppo_ratios.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ppo_ratios.png")

# --- Figure 3: Entropy comparison (PPO, REINFORCE, NPG) ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

for m, label, color in [
    (m_ppo, 'PPO', ALGO_COLORS['PPO']),
    (m_reinforce, 'REINFORCE', ALGO_COLORS['REINFORCE']),
    (m_npg, 'NPG', ALGO_COLORS['NPG']),
]:
    n_ent = len(m.policy_entropy_per_ep)
    s = max(1, n_ent // 2000)
    eps_s = list(range(1, n_ent + 1, s))
    ent_s = m.policy_entropy_per_ep[::s]
    w = max(1, len(ent_s) // 80)
    if w > 1 and len(ent_s) > w:
        smoothed = np.convolve(ent_s, np.ones(w)/w, mode='valid')
        x_s = eps_s[w//2:w//2 + len(smoothed)]
        ax.semilogx(x_s, smoothed, color=color, linewidth=1.5, label=label)

ax.axhline(np.log(5), color='gray', ls='--', lw=0.8, alpha=0.5, label=f'$\\ln(5)$ (uniform)')
ax.set_xlabel('Episode')
ax.set_ylabel('Mean policy entropy')
ax.set_title('Policy Entropy: PPO vs REINFORCE vs NPG')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ppo_entropy.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ppo_entropy.png")

# --- Figure 4: Convergence (RMSE + policy agreement) ---
fig, ax1 = plt.subplots(figsize=FIG_SINGLE)
ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

n = min(len(ckpts), len(m_ppo.value_errors))
ax1.semilogy(ckpts[:n], m_ppo.value_errors[:n], color=ALGO_COLORS['PPO'], label='RMSE')
ax1.set_xlabel('Episode')
ax1.set_ylabel('RMSE $V$ vs $V^*$', color=ALGO_COLORS['PPO'])
ax1.set_xscale('log')

ax2 = ax1.twinx()
n_a = min(len(ckpts), len(m_ppo.policy_agreements))
agreements_pct = [a * 100 for a in m_ppo.policy_agreements[:n_a]]
ax2.plot(ckpts[:n_a], agreements_pct, color='gray', ls='--', alpha=0.7, label='Policy agreement')
ax2.set_ylabel('Policy agreement (%)', color='gray')
ax2.set_ylim(-5, 105)

ax1.set_title('PPO: Convergence Profile')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ppo_convergence.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ppo_convergence.png")

# Print summary
print(f"\nPPO clip statistics:")
print(f"  Mean clip fraction: {np.mean(m_ppo.clip_fractions):.4f}")
print(f"  Max clip fraction: {np.max(m_ppo.clip_fractions):.4f}")
print(f"  Mean ratio: {np.mean(m_ppo.ppo_ratios_mean):.4f}")
print(f"  Mean ratio std: {np.mean(m_ppo.ppo_ratios_std):.4f}")

print("\nDone.")
