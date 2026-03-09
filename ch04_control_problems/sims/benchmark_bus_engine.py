# Bus Engine Replacement Benchmark: Scaling Analysis (Chapter 4)
# Fleet bus engine replacement problem (inspired by Rust 1987).
# Scaling sweep across fleet sizes N=1..6, comparing DP, DQN, and heuristics.

import itertools
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, COLORS, FIG_DOUBLE
apply_style()

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic,
    compute_q_error, compute_policy_agreement, capture_stdout,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MILEAGE_STATES = 6
ALPHA = 1.0       # operating cost weight (per unit mileage)
BETA = 5.0        # replacement cost per engine
GAMMA = 0.95
CAPACITY = 3      # max engines replaceable per period

COMPLEXITY_SWEEP = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 123, 7]
TRAIN_HORIZON = 50      # training episode length (92% of discounted value)
EVAL_HORIZON = 100      # evaluation horizon (99.4% of discounted value)
EVAL_EPISODES = 200

DP_FEASIBLE_THRESHOLD = 10_000  # DP for N=1..4 (max 1,296 states)
DP_TIMEOUT = 300                # seconds, for N=5 attempt

# DQN hyperparameters scaled by N
DQN_CONFIG = {
    1: dict(episodes=3000,  replay=5_000,  hidden1=128, hidden2=64,  lr=1e-3, batch=64,  eps_decay=0.5),
    2: dict(episodes=4000,  replay=8_000,  hidden1=128, hidden2=64,  lr=1e-3, batch=64,  eps_decay=0.5),
    3: dict(episodes=6000,  replay=10_000, hidden1=128, hidden2=64,  lr=1e-3, batch=64,  eps_decay=0.55),
    4: dict(episodes=8000,  replay=15_000, hidden1=128, hidden2=64,  lr=5e-4, batch=128, eps_decay=0.55),
    5: dict(episodes=10000, replay=20_000, hidden1=256, hidden2=128, lr=5e-4, batch=128, eps_decay=0.6),
    6: dict(episodes=12000, replay=25_000, hidden1=256, hidden2=128, lr=5e-4, batch=128, eps_decay=0.6),
}

OUTPUT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = 'benchmark_bus_engine'


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BusEngine(EconBenchmark):
    """Multi-engine bus replacement problem with mileage-dependent costs.

    State: tuple of N mileage levels, each in {0, ..., MILEAGE_STATES-1}
    Action: subset of engines to replace (size <= CAPACITY)
    Reward: -cost where cost = alpha * sum(mileages) + beta * |replacements|
    """

    def __init__(self, N):
        self.N = N
        self.complexity_param = N

        self._all_states = list(itertools.product(range(MILEAGE_STATES), repeat=N))
        self._state_to_idx = {s: i for i, s in enumerate(self._all_states)}

        self._all_actions = []
        engines = list(range(N))
        for k in range(min(CAPACITY, N) + 1):
            for combo in itertools.combinations(engines, k):
                self._all_actions.append(combo)

        self.dp_feasible = len(self._all_states) <= DP_FEASIBLE_THRESHOLD
        self.state = None
        self.reset()

    def reset(self):
        self.state = tuple(np.random.randint(0, MILEAGE_STATES, size=self.N))
        return self.state

    def step(self, action):
        action_tuple = self._all_actions[action]
        cost = self._cost(self.state, action_tuple)
        self.state = self._transition(self.state, action_tuple)
        return self.state, -cost, False

    @property
    def num_states(self):
        return len(self._all_states)

    @property
    def num_actions(self):
        return len(self._all_actions)

    def state_to_index(self, state):
        return self._state_to_idx[state]

    def index_to_state(self, idx):
        return self._all_states[idx]

    def state_to_features(self, state):
        return np.array(state, dtype=np.float32) / (MILEAGE_STATES - 1)

    def transition_distribution(self, state, action):
        action_tuple = self._all_actions[action]
        next_state = self._transition(state, action_tuple)
        return [(next_state, 1.0)]

    def expected_reward(self, state, action):
        action_tuple = self._all_actions[action]
        return -self._cost(state, action_tuple)

    def _cost(self, state, action_tuple):
        operating = sum(state)  # mileage-dependent: cost = alpha * sum(m_i)
        return ALPHA * operating + BETA * len(action_tuple)

    def _transition(self, state, action_tuple):
        replace_set = set(action_tuple)
        new_state = []
        for i, m in enumerate(state):
            if i in replace_set:
                new_state.append(0)
            else:
                new_state.append(min(MILEAGE_STATES - 1, m + 1))
        return tuple(new_state)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def make_threshold_heuristic(env, threshold=3):
    """Replace engines with mileage >= threshold (up to CAPACITY)."""
    def heuristic(state):
        to_replace = tuple(sorted(
            [i for i, m in enumerate(state) if m >= threshold]
        )[:CAPACITY])
        for idx, action_tuple in enumerate(env._all_actions):
            if action_tuple == to_replace:
                return idx
        return 0
    return heuristic


def make_never_replace_heuristic(env):
    """Never replace any engine."""
    def heuristic(state):
        return 0
    return heuristic


# ---------------------------------------------------------------------------
# Run single complexity level
# ---------------------------------------------------------------------------
def run_single_complexity(N, seeds):
    """Run benchmark for a single fleet size N."""
    env = BusEngine(N)
    cfg = DQN_CONFIG[N]
    result = {
        'complexity': N,
        'states': env.num_states,
        'actions': env.num_actions,
    }

    print(f"\n  N={N}: {env.num_states:,} states, {env.num_actions} actions")

    # --- Value Iteration (if feasible) ---
    V, policy = None, None
    if env.dp_feasible or env.num_states <= 10_000:
        print("    Running Value Iteration...")
        t0 = time.time()
        try:
            V, policy, vi_metrics = run_value_iteration(
                env, gamma=GAMMA, tol=1e-8, max_iter=1000
            )
            elapsed = time.time() - t0
            if elapsed > DP_TIMEOUT and N > 4:
                print(f"    VI timed out ({elapsed:.1f}s > {DP_TIMEOUT}s)")
                V, policy = None, None
                result['dp_reward'] = None
                result['dp_time'] = None
            else:
                dp_reward = evaluate_dp_policy(
                    env, policy, gamma=GAMMA,
                    n_episodes=EVAL_EPISODES, horizon=EVAL_HORIZON,
                    discount=GAMMA
                )
                result['dp_reward'] = dp_reward
                result['dp_time'] = vi_metrics.wall_time
                print(f"    VI: {vi_metrics.iterations} iter, "
                      f"residual={vi_metrics.final_residual:.2e}, "
                      f"time={vi_metrics.wall_time:.2f}s, "
                      f"discounted reward={dp_reward:.2f}")
        except Exception as e:
            print(f"    VI failed: {e}")
            V, policy = None, None
            result['dp_reward'] = None
            result['dp_time'] = None
    else:
        result['dp_reward'] = None
        result['dp_time'] = None
        print("    DP skipped (state space too large)")

    # --- DQN (multiple seeds) ---
    print(f"    Running DQN ({len(seeds)} seeds)...")
    dqn_rewards = []
    dqn_times = []
    dqn_q_errors = []
    dqn_agreements = []

    for i, seed in enumerate(seeds):
        env_dqn = BusEngine(N)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA,
            num_episodes=cfg['episodes'],
            episode_horizon=TRAIN_HORIZON,
            seed=seed,
            replay_size=cfg['replay'],
            batch_size=cfg['batch'],
            lr=cfg['lr'],
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_frac=cfg['eps_decay'],
            target_update_freq=50,
            hidden1=cfg['hidden1'],
            hidden2=cfg['hidden2'],
            eval_freq=200,
            eval_episodes=20,
            desc=f"N={N} seed {i+1}/{len(seeds)}"
        )

        reward = evaluate_dqn_policy(
            env, q_net, n_episodes=EVAL_EPISODES,
            horizon=EVAL_HORIZON, discount=GAMMA
        )
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)

        if V is not None:
            q_err = compute_q_error(q_net, V, env, n_samples=500, gamma=GAMMA)
            dqn_q_errors.append(q_err)
            agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
            dqn_agreements.append(agreement)

        print(f"      seed={seed}: reward={reward:.2f}, time={metrics.wall_time:.1f}s")

    result['dqn_reward_mean'] = np.mean(dqn_rewards)
    result['dqn_reward_std'] = np.std(dqn_rewards)
    result['dqn_rewards'] = dqn_rewards
    result['dqn_time_mean'] = np.mean(dqn_times)

    if dqn_q_errors:
        result['q_error'] = np.mean(dqn_q_errors)
        result['agreement'] = np.mean(dqn_agreements)
    else:
        result['q_error'] = None
        result['agreement'] = None

    # --- Heuristics ---
    print("    Running Heuristics...")
    threshold_h = make_threshold_heuristic(env, threshold=3)
    h_threshold = evaluate_heuristic(
        env, threshold_h, n_episodes=EVAL_EPISODES,
        horizon=EVAL_HORIZON, discount=GAMMA
    )
    never_h = make_never_replace_heuristic(env)
    h_never = evaluate_heuristic(
        env, never_h, n_episodes=EVAL_EPISODES,
        horizon=EVAL_HORIZON, discount=GAMMA
    )

    result['h_threshold'] = h_threshold
    result['h_never'] = h_never
    print(f"      Threshold(3)={h_threshold:.2f}, Never={h_never:.2f}")

    return result


# ---------------------------------------------------------------------------
# Scaling sweep
# ---------------------------------------------------------------------------
def run_scaling_sweep():
    """Run benchmark across all fleet sizes."""
    print("=" * 70)
    print("  Bus Engine Replacement: Scaling Analysis")
    print("=" * 70)
    print(f"  Fleet sizes: N in {COMPLEXITY_SWEEP}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Mileage states: {MILEAGE_STATES}, Capacity: {CAPACITY}")
    print(f"  Alpha={ALPHA}, Beta={BETA}, Gamma={GAMMA}")
    print(f"  Train horizon: {TRAIN_HORIZON}, Eval horizon: {EVAL_HORIZON}")
    print(f"  Cost function: c(s,a) = alpha * sum(m_i) + beta * |a|")

    results = []
    for N in COMPLEXITY_SWEEP:
        result = run_single_complexity(N, SEEDS)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(results):
    """Generate two-panel scaling figure."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    Ns = [r['complexity'] for r in results]
    dp_times = []
    dqn_times = []
    dp_rewards = []
    dqn_means = []
    dqn_stds = []
    h_thresh = []
    h_never = []

    for r in results:
        dp_times.append(r['dp_time'])
        dqn_times.append(r['dqn_time_mean'])
        dp_rewards.append(r['dp_reward'])
        dqn_means.append(r['dqn_reward_mean'])
        dqn_stds.append(r['dqn_reward_std'])
        h_thresh.append(r['h_threshold'])
        h_never.append(r['h_never'])

    # Left panel: Wall-clock time vs N (log scale)
    ax = axes[0]
    dp_t_valid = [(n, t) for n, t in zip(Ns, dp_times) if t is not None]
    if dp_t_valid:
        ax.plot([x[0] for x in dp_t_valid], [x[1] for x in dp_t_valid],
                's-', color=COLORS['blue'], label='DP (VI)', markersize=7)
    ax.plot(Ns, dqn_times, 'o-', color=COLORS['cyan'], label='DQN', markersize=7)
    ax.set_yscale('log')
    ax.set_xlabel('Fleet Size $N$')
    ax.set_ylabel('Wall-Clock Time (s)')
    ax.set_title('Computation Time')
    ax.legend()
    ax.set_xticks(Ns)

    # Right panel: Discounted reward vs N
    ax = axes[1]
    dp_r_valid = [(n, r) for n, r in zip(Ns, dp_rewards) if r is not None]
    if dp_r_valid:
        ax.plot([x[0] for x in dp_r_valid], [x[1] for x in dp_r_valid],
                's-', color=COLORS['blue'], label='DP (VI)', markersize=7)
    ax.errorbar(Ns, dqn_means, yerr=dqn_stds, fmt='o-',
                color=COLORS['cyan'], label='DQN', markersize=7, capsize=3)
    ax.plot(Ns, h_thresh, '^--', color=COLORS['green'],
            label='Threshold(3)', markersize=6, alpha=0.8)
    ax.plot(Ns, h_never, 'v--', color=COLORS['red'],
            label='Never Replace', markersize=6, alpha=0.8)
    ax.set_xlabel('Fleet Size $N$')
    ax.set_ylabel('Discounted Return')
    ax.set_title('Policy Performance')
    ax.legend()
    ax.set_xticks(Ns)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'bus_engine_scaling.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def make_latex_table(results):
    """Generate LaTeX scaling table."""
    lines = []
    lines.append(r'\begin{tabular}{rrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'$N$ & $|\mathcal{S}|$ & DP Time & DP Return & '
                 r'DQN Return & Thresh(3) & Never Replace \\')
    lines.append(r'\midrule')

    for r in results:
        N = r['complexity']
        states = r['states']

        if r['dp_reward'] is not None:
            dp_t = f"${r['dp_time']:.2f}$s"
            dp_r = f"${r['dp_reward']:.1f}$"
        else:
            dp_t = "---"
            dp_r = "---"

        dqn_r = f"${r['dqn_reward_mean']:.1f} \\pm {r['dqn_reward_std']:.1f}$"
        h_t = f"${r['h_threshold']:.1f}$"
        h_n = f"${r['h_never']:.1f}$"

        lines.append(f"{N} & {states:,} & {dp_t} & {dp_r} & "
                     f"{dqn_r} & {h_t} & {h_n} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'bus_engine_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")


# ---------------------------------------------------------------------------
# Detailed stdout
# ---------------------------------------------------------------------------
def print_detailed_results(results):
    """Print detailed results to stdout."""
    print("\n" + "=" * 70)
    print("  SCALING SUMMARY")
    print("=" * 70)

    print("\n  " + "-" * 80)
    print(f"  {'N':>3} | {'|S|':>7} | {'DP Time':>8} | {'DP Return':>10} | "
          f"{'DQN Return':>18} | {'Thresh(3)':>10} | {'Never':>8}")
    print("  " + "-" * 80)

    for r in results:
        N = r['complexity']
        states = r['states']
        dp_t = f"{r['dp_time']:.2f}s" if r['dp_time'] is not None else "---"
        dp_r = f"{r['dp_reward']:.2f}" if r['dp_reward'] is not None else "---"
        dqn_r = f"{r['dqn_reward_mean']:.2f} +/- {r['dqn_reward_std']:.2f}"
        h_t = f"{r['h_threshold']:.2f}"
        h_n = f"{r['h_never']:.2f}"
        print(f"  {N:>3} | {states:>7,} | {dp_t:>8} | {dp_r:>10} | "
              f"{dqn_r:>18} | {h_t:>10} | {h_n:>8}")

    print("  " + "-" * 80)

    # Per-seed DQN results
    print("\n  Per-Seed DQN Returns:")
    for r in results:
        N = r['complexity']
        rewards = r['dqn_rewards']
        print(f"    N={N}: " + ", ".join(f"{rw:.2f}" for rw in rewards))

    # Q-error and agreement where available
    has_q = any(r['q_error'] is not None for r in results)
    if has_q:
        print("\n  Q-Error and Policy Agreement (where DP available):")
        print("  " + "-" * 40)
        print(f"  {'N':>3} | {'Q-Error':>10} | {'Agreement':>10}")
        print("  " + "-" * 40)
        for r in results:
            if r['q_error'] is not None:
                print(f"  {r['complexity']:>3} | {r['q_error']:>10.3f} | "
                      f"{r['agreement']:>9.1%}")
        print("  " + "-" * 40)

    # DQN gap vs DP
    print("\n  DQN Gap vs DP (where both available):")
    for r in results:
        if r['dp_reward'] is not None:
            gap = (r['dqn_reward_mean'] - r['dp_reward']) / abs(r['dp_reward']) * 100
            print(f"    N={r['complexity']}: DQN is {gap:+.1f}% vs DP")

    print("\n  Output Files:")
    print(f"    {OUTPUT_DIR / 'bus_engine_scaling.png'}")
    print(f"    {OUTPUT_DIR / 'bus_engine_results.tex'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    stdout_path = OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'
    with capture_stdout(stdout_path):
        print("Bus Engine Replacement Benchmark")
        print(f"Cost function: c(s,a) = {ALPHA} * sum(m_i) + {BETA} * |a|")
        print(f"Fleet sizes: N in {COMPLEXITY_SWEEP}")
        print(f"Seeds: {SEEDS}")
        print(f"Gamma={GAMMA}, Train horizon={TRAIN_HORIZON}, Eval horizon={EVAL_HORIZON}")

        results = run_scaling_sweep()
        make_figures(results)
        make_latex_table(results)
        print_detailed_results(results)

        print("\nBenchmark complete.")
