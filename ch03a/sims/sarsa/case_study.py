# SARSA (GLIE) Case Study: 5x5 Gridworld
# Chapter 3a -- Compares SARSA with epsilon_end=0.01 vs 0.0 (GLIE),
# and against Q-Learning as reference.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, CMAP_SEQ, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_q_learning,
                                   run_sarsa, v_to_array, policy_to_array)

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
print("SARSA CASE STUDY (GLIE vs non-GLIE)")
print("=" * 60)

common_kwargs = dict(
    num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.1, alpha_decay=0.99995,
    epsilon_start=1.0, epsilon_decay=0.9995,
    V_optimal=V_optimal, policy_optimal=policy_optimal,
    optimal_return=9.30, eval_freq=EVAL_FREQ, eval_episodes=100,
    snapshot_episodes=CHECKPOINTS
)

print("\n[SARSA eps_end=0.01]")
_, _, m_sarsa_01 = run_sarsa(env, epsilon_end=0.01, **common_kwargs)
print(f"  Final RMSE: {m_sarsa_01.value_errors[-1]:.4f}, Agreement: {m_sarsa_01.policy_agreements[-1]:.2%}")

print("\n[SARSA eps_end=0.0 (GLIE)]")
_, _, m_sarsa_glie = run_sarsa(env, epsilon_end=0.0, **common_kwargs)
print(f"  Final RMSE: {m_sarsa_glie.value_errors[-1]:.4f}, Agreement: {m_sarsa_glie.policy_agreements[-1]:.2%}")

print("\n[Q-Learning (reference)]")
_, _, m_ql = run_q_learning(env, epsilon_end=0.01, **common_kwargs)
print(f"  Final RMSE: {m_ql.value_errors[-1]:.4f}, Agreement: {m_ql.policy_agreements[-1]:.2%}")

# --- Figure 1: GLIE comparison heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

for ax, m, title in zip(axes, [m_sarsa_01, m_sarsa_glie],
                          [r'SARSA ($\epsilon_{end}=0.01$)', r'SARSA ($\epsilon_{end}=0.0$, GLIE)']):
    last_ep = max(m.value_snapshots.keys())
    V_final = m.value_snapshots[last_ep]
    errors = np.abs(V_final - V_optimal).reshape(N, N)

    im = ax.imshow(errors, cmap=CMAP_SEQ, vmin=0, vmax=V_optimal.max(), origin='upper')
    for r in range(N):
        for c in range(N):
            val = errors[r, c]
            color = 'white' if val > V_optimal.max() * 0.6 else 'black'
            ax.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)

fig.suptitle('Final $|V(s) - V^*(s)|$ at episode 500K', fontsize=11)
fig.tight_layout(rect=[0, 0, 0.92, 0.94])
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, V_optimal.max()), cmap=CMAP_SEQ),
             cax=cbar_ax, label='$|V - V^*|$')
fig.savefig(os.path.join(OUTPUT_DIR, 'sarsa_glie_comparison.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: sarsa_glie_comparison.png")

# --- Figure 2: RMSE convergence curves ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

for m, label, color in [
    (m_sarsa_01, r'SARSA ($\epsilon_{end}=0.01$)', '#ff7f0e'),
    (m_sarsa_glie, r'SARSA (GLIE, $\epsilon_{end}=0.0$)', '#d62728'),
    (m_ql, 'Q-Learning', ALGO_COLORS['Q-Learning']),
]:
    n = min(len(ckpts), len(m.value_errors))
    ax.semilogy(ckpts[:n], m.value_errors[:n], label=label, color=color)

ax.set_xlabel('Episode')
ax.set_ylabel('RMSE $V$ vs $V^*$')
ax.set_title('Convergence: SARSA variants vs Q-Learning')
ax.set_xscale('log')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'sarsa_convergence_curves.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: sarsa_convergence_curves.png")

# --- Figure 3: Policy agreement over episodes ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

for m, label, color in [
    (m_sarsa_01, r'SARSA ($\epsilon_{end}=0.01$)', '#ff7f0e'),
    (m_sarsa_glie, r'SARSA (GLIE)', '#d62728'),
    (m_ql, 'Q-Learning', ALGO_COLORS['Q-Learning']),
]:
    n = min(len(ckpts), len(m.policy_agreements))
    agreements_pct = [a * 100 for a in m.policy_agreements[:n]]
    ax.plot(ckpts[:n], agreements_pct, label=label, color=color)

ax.set_xlabel('Episode')
ax.set_ylabel('Policy agreement (%)')
ax.set_title('Policy Optimality Over Training')
ax.set_xscale('log')
ax.set_ylim(-5, 105)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'sarsa_policy_agreement.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: sarsa_policy_agreement.png")

# --- Figure 4: Per-state Bellman residual for GLIE SARSA at checkpoints ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
br_eps = sorted(m_sarsa_glie.bellman_residual_snapshots.keys())
br_vals = [m_sarsa_glie.bellman_residual_snapshots[ep] for ep in br_eps]
ax.semilogy(br_eps, br_vals, 'o-', color='#d62728', markersize=2)
ax.set_xlabel('Episode')
ax.set_ylabel(r'$\max_{s,a} |Q - \mathcal{T}^* Q|$')
ax.set_title('SARSA (GLIE): Bellman Residual')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'sarsa_per_state_errors.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: sarsa_per_state_errors.png")

# Summary
print(f"\nFinal comparison (episode {NUM_EPISODES}):")
print(f"  {'Method':30s}  {'RMSE':>8s}  {'Agree':>8s}  {'Max|V-V*|':>10s}")
for m, label in [(m_sarsa_01, 'SARSA (eps=0.01)'), (m_sarsa_glie, 'SARSA (GLIE)'), (m_ql, 'Q-Learning')]:
    last_ep = max(m.value_snapshots.keys())
    max_err = np.abs(m.value_snapshots[last_ep] - V_optimal).max()
    print(f"  {label:30s}  {m.value_errors[-1]:8.4f}  {m.policy_agreements[-1]:7.1%}  {max_err:10.4f}")

print("\nDone.")
