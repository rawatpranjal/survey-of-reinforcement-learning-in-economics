# Q(lambda) Case Study: 5x5 Gridworld
# Chapter 3a -- Tracks eligibility traces, trace resets, and speedup vs Q-Learning.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ch03_theory', 'sims'))
from gridworld_algorithms import (GridworldEnv, run_value_iteration, run_q_learning,
                                   run_q_lambda, v_to_array, policy_to_array)

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
print("Q(lambda) CASE STUDY")
print("=" * 60)

common_kwargs = dict(
    num_episodes=NUM_EPISODES, horizon=HORIZON, seed=42,
    alpha=0.1, alpha_decay=0.99995,
    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
    V_optimal=V_optimal, policy_optimal=policy_optimal,
    optimal_return=9.30, eval_freq=EVAL_FREQ, eval_episodes=100,
    snapshot_episodes=CHECKPOINTS
)

print("\n[Q(lambda)]")
_, _, m_ql = run_q_lambda(env, lambda_=0.9, **common_kwargs)
print(f"  Final RMSE: {m_ql.value_errors[-1]:.4f}")

print("\n[Q-Learning (reference)]")
_, _, m_q = run_q_learning(env, **common_kwargs)
print(f"  Final RMSE: {m_q.value_errors[-1]:.4f}")

# --- Figure 1: Mean active traces per episode ---
fig, ax1 = plt.subplots(figsize=FIG_SINGLE)

# Subsample for readability
n_traces = len(m_ql.mean_active_traces)
step = max(1, n_traces // 2000)
eps_sub = list(range(1, n_traces + 1, step))
traces_sub = m_ql.mean_active_traces[::step]

ax1.semilogx(eps_sub, traces_sub, color=ALGO_COLORS['Q(λ)'], alpha=0.3, linewidth=0.5)
# Smooth
window = max(1, len(traces_sub) // 100)
if window > 1 and len(traces_sub) > window:
    smoothed = np.convolve(traces_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax1.semilogx(x_smooth, smoothed, color=ALGO_COLORS['Q(λ)'], linewidth=1.5,
                 label='Mean active traces')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Mean active traces', color=ALGO_COLORS['Q(λ)'])

# Epsilon on twin axis
ax2 = ax1.twinx()
if m_ql.effective_epsilon:
    ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))[:len(m_ql.effective_epsilon)]
    ax2.semilogx(ckpts, m_ql.effective_epsilon, color='gray', ls='--', alpha=0.5, label=r'$\epsilon$')
    ax2.set_ylabel(r'$\epsilon$', color='gray')

ax1.set_title(r'Q($\lambda$): Active Traces Over Training')
ax1.legend(loc='upper left', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_trace_active.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_trace_active.png")

# --- Figure 2: Trace resets per episode ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

resets_sub = m_ql.trace_resets[::step]
ax.semilogx(eps_sub, resets_sub, color=ALGO_COLORS['Q(λ)'], alpha=0.3, linewidth=0.5)
if window > 1 and len(resets_sub) > window:
    smoothed = np.convolve(resets_sub, np.ones(window)/window, mode='valid')
    x_smooth = eps_sub[window//2:window//2 + len(smoothed)]
    ax.semilogx(x_smooth, smoothed, color=ALGO_COLORS['Q(λ)'], linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Trace resets per episode')
ax.set_title(r'Q($\lambda$): Trace Resets (exploratory action count)')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_trace_resets.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_trace_resets.png")

# --- Figure 3: RMSE convergence: Q(lambda) vs Q-Learning ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ckpts = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

n_ql = min(len(ckpts), len(m_ql.value_errors))
n_q = min(len(ckpts), len(m_q.value_errors))
ax.semilogy(ckpts[:n_ql], m_ql.value_errors[:n_ql],
            label=r'Q($\lambda$)', color=ALGO_COLORS['Q(λ)'])
ax.semilogy(ckpts[:n_q], m_q.value_errors[:n_q],
            label='Q-Learning', color=ALGO_COLORS['Q-Learning'])
ax.set_xlabel('Episode')
ax.set_ylabel('RMSE $V$ vs $V^*$')
ax.set_title(r'Convergence Speedup: Q($\lambda$) vs Q-Learning')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_trace_speedup.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_trace_speedup.png")

# --- Figure 4: Q(s0, a_optimal) over episodes ---
fig, ax = plt.subplots(figsize=FIG_SINGLE)

s0_idx = env.state_to_index((0, 0))
a_opt = policy_optimal[s0_idx]

snap_eps_ql = sorted(m_ql.q_table_snapshots.keys())
snap_eps_q = sorted(m_q.q_table_snapshots.keys())

q_vals_ql = [m_ql.q_table_snapshots[ep][s0_idx, a_opt] for ep in snap_eps_ql]
q_vals_q = [m_q.q_table_snapshots[ep][s0_idx, a_opt] for ep in snap_eps_q]

ax.semilogx(snap_eps_ql, q_vals_ql, label=r'Q($\lambda$)', color=ALGO_COLORS['Q(λ)'])
ax.semilogx(snap_eps_q, q_vals_q, label='Q-Learning', color=ALGO_COLORS['Q-Learning'])
ax.axhline(V_optimal[s0_idx], color='black', ls='--', lw=0.8, alpha=0.5,
           label=f'$V^*(s_0)$ = {V_optimal[s0_idx]:.2f}')
ax.set_xlabel('Episode')
ax.set_ylabel(f'$Q(s_0, a^*)$')
ax.set_title(r'Credit Propagation: Q($\lambda$) vs Q-Learning')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ql_trace_qvalue_comparison.png'), bbox_inches='tight')
plt.close(fig)
print("Saved: ql_trace_qvalue_comparison.png")

print("\nDone.")
