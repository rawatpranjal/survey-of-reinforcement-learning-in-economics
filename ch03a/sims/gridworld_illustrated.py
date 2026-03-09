# Illustrated Example: 9 Algorithms on a 5x5 Gridworld (2 planning + 7 learning)
# Chapter 3a -- Per-state value function convergence analysis
# Produces 5 figures, 2 LaTeX tables, and comprehensive stdout diagnostics.
#
# Supports incremental execution:
#   python gridworld_illustrated.py                  # run all (or load from cache)
#   python gridworld_illustrated.py --only dqn,ppo   # run only DQN and PPO
#   python gridworld_illustrated.py --plots-only      # regenerate figures/tables from cache

import os
import sys
import time
import pickle
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Shared style module lives at repo root sims/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE, FIG_DOUBLE

# gridworld_algorithms lives in ch03_theory/sims/
sys.path.insert(0, str(Path(__file__).resolve().parent / '../../ch03_theory/sims'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
apply_style()
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable

from gridworld_algorithms import (
    GridworldEnv,
    run_value_iteration, run_policy_iteration,
    run_q_learning, run_sarsa, run_q_lambda,
    run_reinforce, run_npg, run_ppo, run_dqn_tabular_comparison,
    v_to_array, policy_to_array, evaluate_policy,
    AlgorithmMetrics, compute_value_error
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N = 5
GAMMA = 0.95
STEP_PENALTY = -0.1
TERMINAL_REWARD = 10.0
SYMMETRY_BREAK = 0.001

NUM_EPISODES = 500_000
EPISODE_HORIZON = 50
EVAL_EPISODES = 10   # 10 greedy rollouts per eval (deterministic env → stable)

# Per-algorithm eval frequency: tabular methods are cheap to evaluate,
# DQN/policy gradient need neural forward passes so eval less often
EVAL_FREQ_TABULAR = 10    # QL, SARSA, Q(λ)
EVAL_FREQ_DQN = 100       # DQN (neural net eval is expensive)
EVAL_FREQ_PG = 50         # REINFORCE, NPG, PPO

ALGO_EVAL_FREQ = {
    'Q-Learning': EVAL_FREQ_TABULAR, 'SARSA': EVAL_FREQ_TABULAR,
    'Q(λ)': EVAL_FREQ_TABULAR, 'DQN': EVAL_FREQ_DQN,
    'REINFORCE': EVAL_FREQ_PG, 'NPG': EVAL_FREQ_PG, 'PPO': EVAL_FREQ_PG,
}

ALPHA = 0.1
ALPHA_DECAY = 0.99995
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995
LAMBDA = 0.9

M = 1
SEEDS = [42]

# Dense early snapshots for precise convergence measurement, sparse late for figures
SNAPSHOT_EPISODES = sorted(set(
    list(range(10, 1010, 10)) +        # every 10 episodes for first 1000
    list(range(1000, 10001, 100)) +    # every 100 up to 10K
    list(range(10000, 500001, 2000))   # every 2000 after 10K
))

OUTPUT_DIR = Path(__file__).resolve().parent

ALGO_KEYS = ['Q-Learning', 'SARSA', 'Q(λ)', 'DQN', 'REINFORCE', 'NPG', 'PPO']

# Filesystem-safe cache keys
ALGO_CACHE_KEYS = {
    'Q-Learning': 'ql', 'SARSA': 'sarsa', 'Q(λ)': 'ql_trace',
    'DQN': 'dqn', 'REINFORCE': 'reinforce', 'NPG': 'npg', 'PPO': 'ppo',
}
CACHE_KEY_TO_ALGO = {v: k for k, v in ALGO_CACHE_KEYS.items()}

ALGORITHM_NAMES = {
    'Q-Learning': 'Q-Learning',
    'SARSA': 'SARSA',
    'Q(λ)': r'Q($\lambda$)',
    'DQN': 'DQN',
    'REINFORCE': 'REINFORCE',
    'NPG': 'NPG',
    'PPO': 'PPO',
}

COLORS = {k: ALGO_COLORS[k] for k in ALGO_KEYS}
COLORS_PLANNING = {'VI': ALGO_COLORS['VI'], 'PI': ALGO_COLORS['PI']}

# Per-state convergence threshold: max_s |V(s) - V*(s)| < 0.1
VALUE_CONV_THRESHOLD = 0.1

CACHE_DIR = OUTPUT_DIR / 'cache'


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------

class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.stdout = sys.__stdout__

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Caching infrastructure
# ---------------------------------------------------------------------------

def _algo_config(name):
    """Return the hyperparameter dict for a given algorithm (for hashing)."""
    configs = {
        'Q-Learning': dict(alpha=ALPHA, alpha_decay=ALPHA_DECAY,
                           epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                           epsilon_decay=EPSILON_DECAY),
        'SARSA': dict(alpha=ALPHA, alpha_decay=ALPHA_DECAY,
                      epsilon_start=EPSILON_START, epsilon_end=0.0,
                      epsilon_decay=EPSILON_DECAY),
        'Q(λ)': dict(alpha=ALPHA, alpha_decay=ALPHA_DECAY, lambda_=LAMBDA,
                      epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                      epsilon_decay=EPSILON_DECAY),
        'DQN': dict(replay_size=50000, batch_size=64, lr=1e-3,
                    epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                    epsilon_decay=0.9999, target_update_freq=100, hidden_dim=64),
        'REINFORCE': dict(alpha=0.01, alpha_decay=ALPHA_DECAY, temperature=1.0,
                          baseline=True),
        'NPG': dict(eta=0.2, eta_decay=ALPHA_DECAY, temperature=1.0,
                    alpha_critic=0.5, alpha_critic_decay=ALPHA_DECAY, baseline=True),
        'PPO': dict(alpha=0.01, alpha_decay=0.99995, clip_ratio=0.2,
                    n_epochs=4, gae_lambda=0.95, temperature=1.0, baseline=True),
    }
    return configs[name]


def compute_config_hash(name):
    """Hash of algorithm config + shared params to detect stale caches."""
    shared = dict(N=N, GAMMA=GAMMA, STEP_PENALTY=STEP_PENALTY,
                  TERMINAL_REWARD=TERMINAL_REWARD, SYMMETRY_BREAK=SYMMETRY_BREAK,
                  NUM_EPISODES=NUM_EPISODES, EPISODE_HORIZON=EPISODE_HORIZON,
                  EVAL_FREQ=ALGO_EVAL_FREQ[name], EVAL_EPISODES=EVAL_EPISODES,
                  SEEDS=SEEDS, M=M)
    combined = {**shared, **_algo_config(name)}
    raw = str(sorted(combined.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_algo_cache(name, agg_result, detailed_metric):
    """Pickle an algorithm's results to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = ALGO_CACHE_KEYS[name]
    data = {
        'config_hash': compute_config_hash(name),
        'agg_result': agg_result,
        'detailed_metrics': detailed_metric,
    }
    path = CACHE_DIR / f'{cache_key}_results.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached: {path.name}")


def load_algo_cache(name):
    """Load cached results if valid. Returns (agg_result, detailed_metric) or None."""
    cache_key = ALGO_CACHE_KEYS[name]
    path = CACHE_DIR / f'{cache_key}_results.pkl'
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        expected_hash = compute_config_hash(name)
        if data.get('config_hash') != expected_hash:
            print(f"  Cache stale for {name} (hash mismatch), will re-run")
            return None
        return data['agg_result'], data['detailed_metrics']
    except Exception as e:
        print(f"  Cache load failed for {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------

@dataclass
class AggResult:
    name: str
    eval_returns_all: List[List[float]]
    value_errors_all: List[List[float]]       # RMSE per checkpoint per seed
    agreements_all: List[List[float]]
    per_state_errors_all: list                 # List of dicts: {checkpoint_ep: array(num_states)}
    # Finals (mean ± std across seeds)
    return_mean: float
    return_std: float
    agreement_mean: float
    agreement_std: float
    value_error_mean: float
    value_error_std: float
    max_per_state_error_mean: float
    max_per_state_error_std: float
    time_mean: float
    time_std: float
    episodes_to_return_99: float
    episodes_to_v_s0: Optional[float]          # first ep where |V(s0) - V*(s0)| < threshold
    episodes_to_v_s0_std: float
    episodes_to_vstar_conv: Optional[float]    # max_s convergence; None if never
    episodes_to_vstar_conv_std: float
    # Per-state policy convergence: episode where each state first has pi(s)=pi*(s)
    policy_conv_per_state: Optional[np.ndarray]  # shape (num_states,), -1 if never


# ---------------------------------------------------------------------------
# Per-state error extraction
# ---------------------------------------------------------------------------

def extract_per_state_errors(metrics, V_optimal, env):
    """Extract per-state |V(s) - V*(s)| at each snapshot checkpoint."""
    per_state = {}
    if not metrics.value_snapshots:
        return per_state
    for ep, V_snap in sorted(metrics.value_snapshots.items()):
        errors = np.abs(V_snap - V_optimal)
        per_state[ep] = errors
    return per_state


def compute_max_per_state_error(per_state_errors, episode):
    """Get max_s |V(s) - V*(s)| at a given episode."""
    if episode in per_state_errors:
        return per_state_errors[episode].max()
    return None


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

def run_planning(env):
    print("=" * 70)
    print("PHASE 1: PLANNING METHODS")
    print("=" * 70)

    print("\n[VI] Running Value Iteration...")
    V_vi, policy_vi, metrics_vi = run_value_iteration(env)
    V_optimal = v_to_array(V_vi, env)
    policy_optimal = policy_to_array(policy_vi, env)
    optimal_return, optimal_steps = evaluate_policy(
        env, policy_vi, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)

    print(f"  Iterations:     {metrics_vi.iterations}")
    print(f"  Residual:       {metrics_vi.final_residual:.2e}")
    print(f"  Time:           {metrics_vi.wall_time:.4f}s")
    print(f"  Optimal return: {optimal_return:.4f}")
    print(f"  Optimal steps:  {optimal_steps:.2f}")

    # Print V* for all states
    print(f"\n  V* (5x5 grid):")
    V_grid = V_optimal.reshape(N, N)
    for r in range(N):
        vals = '  '.join(f'{V_grid[r, c]:6.3f}' for c in range(N))
        print(f"    row {r}: {vals}")

    print("\n[PI] Running Policy Iteration...")
    V_pi, policy_pi, metrics_pi = run_policy_iteration(env)
    pi_return, _ = evaluate_policy(
        env, policy_pi, n_episodes=EVAL_EPISODES, horizon=EPISODE_HORIZON)

    print(f"  Iterations:     {metrics_pi.iterations}")
    print(f"  Time:           {metrics_pi.wall_time:.4f}s")
    print(f"  Return:         {pi_return:.4f}")
    print(f"  Policy changes: {metrics_pi.policy_changes_per_iter}")

    return (V_optimal, policy_optimal, optimal_return,
            metrics_vi, metrics_pi, V_vi, policy_vi)


def run_learning_seed(name, run_fn, env, V_optimal, policy_optimal,
                      optimal_return, seed, snapshot=False, **kwargs):
    env_copy = GridworldEnv(N, GAMMA, STEP_PENALTY, TERMINAL_REWARD,
                            symmetry_break=SYMMETRY_BREAK)

    eval_freq = ALGO_EVAL_FREQ[name]
    call_kwargs = dict(
        num_episodes=NUM_EPISODES,
        horizon=EPISODE_HORIZON,
        seed=seed,
        V_optimal=V_optimal,
        policy_optimal=policy_optimal,
        optimal_return=optimal_return,
        eval_freq=eval_freq,
        eval_episodes=EVAL_EPISODES,
        **kwargs
    )

    call_kwargs['snapshot_episodes'] = SNAPSHOT_EPISODES if snapshot else None

    _, policy, metrics = run_fn(env_copy, **call_kwargs)
    return policy, metrics


def _get_algo_configs():
    """Algorithm name -> (run_function, kwargs_dict)."""
    return {
        'Q-Learning': (run_q_learning, dict(
            alpha=ALPHA, alpha_decay=ALPHA_DECAY,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY)),
        'SARSA': (run_sarsa, dict(
            alpha=ALPHA, alpha_decay=ALPHA_DECAY,
            epsilon_start=EPSILON_START, epsilon_end=0.0,
            epsilon_decay=EPSILON_DECAY)),
        'Q(λ)': (run_q_lambda, dict(
            alpha=ALPHA, alpha_decay=ALPHA_DECAY, lambda_=LAMBDA,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY)),
        'DQN': (run_dqn_tabular_comparison, dict(
            replay_size=50000, batch_size=64, lr=1e-3,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=0.9999, target_update_freq=100,
            hidden_dim=64)),
        'REINFORCE': (run_reinforce, dict(
            alpha=0.01, alpha_decay=ALPHA_DECAY, temperature=1.0,
            baseline=True)),
        'NPG': (run_npg, dict(
            eta=0.2, eta_decay=ALPHA_DECAY, temperature=1.0,
            alpha_critic=0.5, alpha_critic_decay=ALPHA_DECAY,
            baseline=True)),
        'PPO': (run_ppo, dict(
            alpha=0.01, alpha_decay=0.99995, clip_ratio=0.2,
            n_epochs=4, gae_lambda=0.95, temperature=1.0, baseline=True)),
    }


def run_single_algo(name, env, V_optimal, policy_optimal, optimal_return):
    """Run one algorithm across all seeds, return (AggResult, detailed_metric)."""
    algo_configs = _get_algo_configs()
    run_fn, kwargs = algo_configs[name]
    eval_freq = ALGO_EVAL_FREQ[name]
    print(f"\n[{name}] Running {M} seeds (eval_freq={eval_freq})...")
    t0 = time.time()

    all_eval_returns = []
    all_value_errors = []
    all_agreements = []
    all_per_state_errors = []
    final_returns = []
    final_agreements = []
    final_value_errors = []
    final_max_per_state = []
    times = []
    ep_to_return_99_list = []
    ep_to_v_s0_list = []
    ep_to_vstar_list = []
    policy_conv_per_state_list = []
    detail_metric = None

    for i, seed in enumerate(SEEDS):
        is_detail_seed = (i == 0)
        _, metrics = run_learning_seed(
            name, run_fn, env, V_optimal, policy_optimal,
            optimal_return, seed, snapshot=is_detail_seed, **kwargs)

        all_eval_returns.append(metrics.eval_returns)
        all_value_errors.append(metrics.value_errors)
        all_agreements.append(metrics.policy_agreements)

        if is_detail_seed:
            pse = extract_per_state_errors(metrics, V_optimal, env)
            all_per_state_errors.append(pse)
            detail_metric = metrics

        if metrics.eval_returns:
            final_returns.append(metrics.eval_returns[-1])
        if metrics.policy_agreements:
            final_agreements.append(metrics.policy_agreements[-1])
        if metrics.value_errors:
            final_value_errors.append(metrics.value_errors[-1])
        times.append(metrics.wall_time)

        threshold = 0.99 * optimal_return
        found = False
        for j, ret in enumerate(metrics.eval_returns):
            if ret >= threshold:
                ep_to_return_99_list.append((j + 1) * eval_freq)
                found = True
                break
        if not found:
            ep_to_return_99_list.append(NUM_EPISODES)

        if is_detail_seed and pse:
            found_v = False
            for ep in sorted(pse.keys()):
                if pse[ep].max() < VALUE_CONV_THRESHOLD:
                    ep_to_vstar_list.append(ep)
                    found_v = True
                    break
            if not found_v:
                ep_to_vstar_list.append(None)
            last_ep = max(pse.keys())
            final_max_per_state.append(pse[last_ep].max())

            found_s0 = False
            for ep in sorted(pse.keys()):
                if pse[ep][0] < VALUE_CONV_THRESHOLD:
                    ep_to_v_s0_list.append(ep)
                    found_s0 = True
                    break
            if not found_s0:
                ep_to_v_s0_list.append(None)

            state_pol_conv = np.full(env.num_states, -1, dtype=int)
            for ep in sorted(metrics.policy_snapshots.keys()):
                pi_snap = metrics.policy_snapshots[ep]
                for s_idx in range(env.num_states):
                    if state_pol_conv[s_idx] < 0 and pi_snap[s_idx] == policy_optimal[s_idx]:
                        state_pol_conv[s_idx] = ep
            policy_conv_per_state_list.append(state_pol_conv)
        else:
            found_v = False
            for j, rmse in enumerate(metrics.value_errors):
                if rmse < VALUE_CONV_THRESHOLD / np.sqrt(env.num_states):
                    ep_to_vstar_list.append((j + 1) * eval_freq)
                    found_v = True
                    break
            if not found_v:
                ep_to_vstar_list.append(None)
            ep_to_v_s0_list.append(None)

    elapsed = time.time() - t0

    valid_vstar = [x for x in ep_to_vstar_list if x is not None]
    ep_vstar_mean = np.mean(valid_vstar) if valid_vstar else None
    ep_vstar_std = (np.std(valid_vstar) if len(valid_vstar) > 1 else 0.0) if valid_vstar else 0.0

    valid_v_s0 = [x for x in ep_to_v_s0_list if x is not None]
    ep_v_s0_mean = np.mean(valid_v_s0) if valid_v_s0 else None
    ep_v_s0_std = (np.std(valid_v_s0) if len(valid_v_s0) > 1 else 0.0) if valid_v_s0 else 0.0

    pol_conv = policy_conv_per_state_list[0] if policy_conv_per_state_list else None

    agg = AggResult(
        name=name,
        eval_returns_all=all_eval_returns,
        value_errors_all=all_value_errors,
        agreements_all=all_agreements,
        per_state_errors_all=all_per_state_errors,
        return_mean=np.mean(final_returns),
        return_std=np.std(final_returns),
        agreement_mean=np.mean(final_agreements) if final_agreements else 0.0,
        agreement_std=np.std(final_agreements) if final_agreements else 0.0,
        value_error_mean=np.mean(final_value_errors) if final_value_errors else 0.0,
        value_error_std=np.std(final_value_errors) if final_value_errors else 0.0,
        max_per_state_error_mean=np.mean(final_max_per_state) if final_max_per_state else 0.0,
        max_per_state_error_std=np.std(final_max_per_state) if final_max_per_state else 0.0,
        time_mean=np.mean(times),
        time_std=np.std(times),
        episodes_to_return_99=np.mean(ep_to_return_99_list),
        episodes_to_v_s0=ep_v_s0_mean,
        episodes_to_v_s0_std=ep_v_s0_std,
        episodes_to_vstar_conv=ep_vstar_mean,
        episodes_to_vstar_conv_std=ep_vstar_std,
        policy_conv_per_state=pol_conv,
    )

    vstar_str = f'{ep_vstar_mean:.0f}' if ep_vstar_mean is not None else '---'
    v_s0_str = f'{ep_v_s0_mean:.0f}' if ep_v_s0_mean is not None else '---'
    print(f"  Return:          {agg.return_mean:.3f} +/- {agg.return_std:.3f}")
    print(f"  Agreement:       {agg.agreement_mean:.2%} +/- {agg.agreement_std:.2%}")
    print(f"  V* RMSE:         {agg.value_error_mean:.4f} +/- {agg.value_error_std:.4f}")
    print(f"  Max |V-V*|:      {agg.max_per_state_error_mean:.4f}")
    print(f"  Ep to 99% ret:   {agg.episodes_to_return_99:.0f}")
    print(f"  Ep to V(s0):     {v_s0_str}")
    print(f"  Ep to V* conv:   {vstar_str}")
    print(f"  Time:            {agg.time_mean:.2f} +/- {agg.time_std:.2f}s")
    print(f"  Total ({M} seeds): {elapsed:.1f}s")

    return agg, detail_metric


def run_all_learning(env, V_optimal, policy_optimal, optimal_return,
                     only_algos=None):
    """Run learning algorithms with per-algorithm caching.

    Args:
        only_algos: if set, list of cache keys (e.g. ['dqn','ppo']) to force re-run.
                    All others loaded from cache if available.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: LEARNING ALGORITHMS (M={M} seeds)")
    print("=" * 70)

    # Resolve which algos to force-run
    force_run = set()
    if only_algos is not None:
        for ck in only_algos:
            if ck in CACHE_KEY_TO_ALGO:
                force_run.add(CACHE_KEY_TO_ALGO[ck])
            else:
                print(f"  Warning: unknown algorithm key '{ck}', skipping")

    agg_results = {}
    detailed_metrics = {}

    for name in ALGO_KEYS:
        should_run = (only_algos is None) or (name in force_run)

        # Try cache first (unless force-running this algo)
        if name not in force_run:
            cached = load_algo_cache(name)
            if cached is not None:
                agg, detail = cached
                agg_results[name] = agg
                if detail is not None:
                    detailed_metrics[name] = detail
                print(f"\n[{name}] Loaded from cache")
                _print_agg_summary(agg)
                continue
            elif not should_run:
                print(f"\n[{name}] No cache, skipping (not in --only list)")
                continue

        # Run fresh
        agg, detail = run_single_algo(
            name, env, V_optimal, policy_optimal, optimal_return)
        agg_results[name] = agg
        if detail is not None:
            detailed_metrics[name] = detail
        save_algo_cache(name, agg, detail)

    return agg_results, detailed_metrics


def _print_agg_summary(agg):
    """Print summary line for a cached/loaded algorithm."""
    vstar_str = f'{agg.episodes_to_vstar_conv:.0f}' if agg.episodes_to_vstar_conv is not None else '---'
    v_s0_str = f'{agg.episodes_to_v_s0:.0f}' if agg.episodes_to_v_s0 is not None else '---'
    print(f"  Return:          {agg.return_mean:.3f} +/- {agg.return_std:.3f}")
    print(f"  Agreement:       {agg.agreement_mean:.2%} +/- {agg.agreement_std:.2%}")
    print(f"  V* RMSE:         {agg.value_error_mean:.4f} +/- {agg.value_error_std:.4f}")
    print(f"  Max |V-V*|:      {agg.max_per_state_error_mean:.4f}")
    print(f"  Ep to 99% ret:   {agg.episodes_to_return_99:.0f}")
    print(f"  Ep to V(s0):     {v_s0_str}")
    print(f"  Ep to V* conv:   {vstar_str}")
    print(f"  Time:            {agg.time_mean:.2f} +/- {agg.time_std:.2f}s")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _mean_with_band(data_all, checkpoints=None):
    min_len = min(len(d) for d in data_all) if data_all else 0
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])
    arr = np.array([d[:min_len] for d in data_all])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = checkpoints[:min_len] if checkpoints is not None else np.arange(min_len)
    return x, mean, std


def _make_checkpoints(eval_freq):
    return list(range(eval_freq, NUM_EPISODES + 1, eval_freq))


# ---------------------------------------------------------------------------
# Figure 1: Learning curves
# ---------------------------------------------------------------------------

def fig_learning_curves(agg_results):
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name in ALGO_KEYS:
        if name not in agg_results:
            continue
        agg = agg_results[name]
        ckpts = _make_checkpoints(ALGO_EVAL_FREQ[name])
        x, mean, std = _mean_with_band(agg.eval_returns_all, ckpts)
        if len(x) == 0:
            continue
        ax.plot(x, mean, label=name, color=COLORS[name])
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=COLORS[name])
    ax.set_xscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Evaluation Return')
    ax.set_title(f'Learning Curves ({M}-seed mean ± 1 std)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gridworld_learning_curves.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gridworld_learning_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: Value error heatmaps (5x5, error magnitude)
# ---------------------------------------------------------------------------

def fig_value_heatmaps(detailed_metrics, V_optimal, env, optimal_return):
    """Value heatmaps: V(s) at percentage-based checkpoints, same scale as V*."""
    pct_labels = ['10%', '25%', '50%', '75%', 'Final']
    pct_fracs = [0.10, 0.25, 0.50, 0.75, 1.0]

    all_algos = ALGO_KEYS
    available = [a for a in all_algos if a in detailed_metrics
                 and detailed_metrics[a].value_snapshots]
    if not available:
        print("  Skipped: gridworld_value_heatmaps.png (no snapshot data)")
        return

    nrows = len(available)
    ncols = len(pct_labels) + 1  # +1 for V* reference

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Unified colormap: all columns use Blues with same scale
    val_cmap = 'Blues'
    vmin_v = min(V_optimal.min(), 0)
    vmax_v = V_optimal.max()

    V_grid_opt = V_optimal.reshape(N, N)

    for i, name in enumerate(available):
        display_name = ALGORITHM_NAMES[name]
        m = detailed_metrics[name]

        # T = first snapshot where max_s |V(s)-V*(s)| < 0.1; else NUM_EPISODES
        conv_ep = NUM_EPISODES
        snap_keys_sorted = sorted(m.value_snapshots.keys())
        for ep in snap_keys_sorted:
            if np.abs(m.value_snapshots[ep] - V_optimal).max() < VALUE_CONV_THRESHOLD:
                conv_ep = ep
                break

        for j, (frac, label) in enumerate(zip(pct_fracs, pct_labels)):
            ax = axes[i, j]
            target_ep = max(10, int(round(frac * conv_ep)))
            target_ep = min(target_ep, NUM_EPISODES)

            snap_keys = snap_keys_sorted
            nearest = min(snap_keys, key=lambda k: abs(k - target_ep))

            V_snap = m.value_snapshots[nearest]
            V_grid = V_snap.reshape(N, N)

            ax.imshow(V_grid, cmap=val_cmap, vmin=vmin_v, vmax=vmax_v,
                      origin='upper')

            # Annotate each cell with V(s) value
            for r in range(N):
                for c in range(N):
                    val = V_grid[r, c]
                    color = 'white' if val > (vmin_v + vmax_v) * 0.6 else 'black'
                    ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                            fontsize=6, color=color)

            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(display_name, fontsize=9)
            # Episode number
            ax.text(0.97, 0.03, f'ep {nearest}', fontsize=5, color='black',
                    ha='right', va='bottom', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))

        # V* column
        ax = axes[i, -1]
        ax.imshow(V_grid_opt, cmap=val_cmap, vmin=vmin_v, vmax=vmax_v, origin='upper')
        for r in range(N):
            for c in range(N):
                val = V_grid_opt[r, c]
                color = 'white' if val > (vmin_v + vmax_v) * 0.6 else 'black'
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                        fontsize=6, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title('$V^*$', fontsize=10)

    fig.suptitle('Learned $V(s)$ at percentage of $V^*$-convergence time',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])

    # Colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    fig.colorbar(ScalarMappable(Normalize(vmin_v, vmax_v), val_cmap),
                 cax=cbar_ax, label='$V(s)$')

    fig.savefig(OUTPUT_DIR / 'gridworld_value_heatmaps.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gridworld_value_heatmaps.png")


# ---------------------------------------------------------------------------
# Figure 3: Per-state convergence fan (max, mean, min error over episodes)
# ---------------------------------------------------------------------------

def fig_convergence_fan(detailed_metrics, V_optimal, env):
    """Plot max/mean/min per-state error over episodes for each algorithm."""
    available = [a for a in ALGO_KEYS if a in detailed_metrics
                 and detailed_metrics[a].value_snapshots]
    if not available:
        print("  Skipped: gridworld_convergence.png (no snapshot data)")
        return

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for name in available:
        m = detailed_metrics[name]
        episodes = sorted(m.value_snapshots.keys())

        max_errors = []
        mean_errors = []
        min_errors = []
        for ep in episodes:
            errs = np.abs(m.value_snapshots[ep] - V_optimal)
            max_errors.append(errs.max())
            mean_errors.append(errs.mean())
            min_errors.append(errs.min())

        eps_arr = np.array(episodes)
        max_arr = np.array(max_errors)
        mean_arr = np.array(mean_errors)
        min_arr = np.array(min_errors)

        color = COLORS[name]
        ax.semilogy(eps_arr, mean_arr, label=name, color=color)
        ax.fill_between(eps_arr, np.maximum(min_arr, 1e-6), max_arr,
                        alpha=0.15, color=color)

    # Threshold line
    ax.axhline(VALUE_CONV_THRESHOLD, color='black', ls='--', lw=0.8, alpha=0.5)
    ax.text(NUM_EPISODES * 0.7, VALUE_CONV_THRESHOLD * 1.3,
            f'threshold = {VALUE_CONV_THRESHOLD}', fontsize=8, alpha=0.7)

    ax.set_xlabel('Episode')
    ax.set_ylabel('$|V(s) - V^*(s)|$ (log scale)')
    ax.set_title('Per-state value error: mean (line) with min/max range (band)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gridworld_convergence.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gridworld_convergence.png")


# ---------------------------------------------------------------------------
# Figure 4: Policy convergence (fraction of states with pi(s) = pi*(s))
# ---------------------------------------------------------------------------

def fig_policy_convergence(detailed_metrics, policy_optimal, env):
    """Line plot: fraction of states where pi(s) = pi*(s) over episodes."""
    available = [a for a in ALGO_KEYS if a in detailed_metrics
                 and detailed_metrics[a].policy_snapshots]
    if not available:
        print("  Skipped: gridworld_policy_convergence.png (no snapshot data)")
        return

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Exclude terminal state from count (policy is trivial there)
    terminal_idx = env.state_to_index(env.terminal)
    non_terminal = [s for s in range(env.num_states) if s != terminal_idx]
    n_non_terminal = len(non_terminal)

    for name in available:
        m = detailed_metrics[name]
        episodes = sorted(m.policy_snapshots.keys())
        fracs = []
        for ep in episodes:
            pi_snap = m.policy_snapshots[ep]
            matches = sum(1 for s in non_terminal if pi_snap[s] == policy_optimal[s])
            fracs.append(matches / n_non_terminal)

        ax.plot(episodes, fracs, label=name, color=COLORS[name])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Fraction of states with $\\pi(s) = \\pi^*(s)$')
    ax.set_title('Per-state policy optimality over training')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gridworld_policy_convergence.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gridworld_policy_convergence.png")


# ---------------------------------------------------------------------------
# Figure 5: Policy heatmaps (5x5 with arrows)
# ---------------------------------------------------------------------------

def _draw_policy_arrows(ax, pi_grid, pi_opt, env, marker_size=8, lw=1.2):
    arrow_dx = [-1, 1, 0, 0, 0]   # row delta: UP=-1, DOWN=+1
    arrow_dy = [0, 0, -1, 1, 0]   # col delta: LEFT=-1, RIGHT=+1
    agree = 0
    total = 0
    for r in range(env.N):
        for c in range(env.N):
            if (r, c) == env.terminal:
                ax.plot(c, r, 'ks', markersize=marker_size)
                continue
            a = pi_grid[r, c]
            matches = (a == pi_opt[r, c])
            if matches:
                agree += 1
            total += 1
            color = '#2ca02c' if matches else '#d62728'
            if a == 4:  # STAY: draw dot instead of zero-length arrow
                ax.plot(c, r, 'o', color=color, markersize=marker_size * 0.6)
            else:
                ax.annotate('', xy=(c + 0.3 * arrow_dy[a], r + 0.3 * arrow_dx[a]),
                            xytext=(c, r),
                            arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    ax.set_xlim(-0.5, env.N - 0.5)
    ax.set_ylim(env.N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    return agree, total


def fig_policy_heatmaps(detailed_metrics, policy_optimal, env, V_optimal):
    """Policy arrow grid at percentage checkpoints."""
    pct_labels = ['10%', '25%', '50%', '75%', 'Final']
    pct_fracs = [0.10, 0.25, 0.50, 0.75, 1.0]

    available = [a for a in ALGO_KEYS if a in detailed_metrics
                 and detailed_metrics[a].policy_snapshots]
    if not available:
        print("  Skipped: gridworld_policy_heatmaps.png (no snapshot data)")
        return

    pi_opt = policy_optimal.reshape(env.N, env.N)
    nrows = len(available)
    ncols = len(pct_labels) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i, name in enumerate(available):
        display_name = ALGORITHM_NAMES[name]
        m = detailed_metrics[name]

        # T = first snapshot where max_s |V(s)-V*(s)| < 0.1; else NUM_EPISODES
        conv_ep = NUM_EPISODES
        if m.value_snapshots:
            snap_keys_sorted = sorted(m.value_snapshots.keys())
            for ep in snap_keys_sorted:
                if np.abs(m.value_snapshots[ep] - V_optimal).max() < VALUE_CONV_THRESHOLD:
                    conv_ep = ep
                    break

        for j, (frac, label) in enumerate(zip(pct_fracs, pct_labels)):
            ax = axes[i, j]
            target_ep = max(10, int(round(frac * conv_ep)))
            target_ep = min(target_ep, NUM_EPISODES)
            snap_keys = sorted(m.policy_snapshots.keys())
            nearest = min(snap_keys, key=lambda k: abs(k - target_ep))
            pi_snap = m.policy_snapshots[nearest].reshape(env.N, env.N)
            agree, total = _draw_policy_arrows(ax, pi_snap, pi_opt, env)
            pct_agree = 100.0 * agree / total if total > 0 else 0
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(display_name, fontsize=9)
            ax.text(0.97, 0.03, f'ep {nearest}\n{pct_agree:.0f}%',
                    fontsize=6, color='white', ha='right', va='bottom',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5))

        # Optimal column
        ax = axes[i, -1]
        _draw_policy_arrows(ax, pi_opt, pi_opt, env)
        if i == 0:
            ax.set_title('$\\pi^*$', fontsize=10)

    fig.suptitle('Policy evolution (percentage of $V^*$-convergence time, seed 42)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'gridworld_policy_heatmaps.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gridworld_policy_heatmaps.png")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_results_table(agg_results, optimal_return, metrics_vi, metrics_pi):
    lines = []
    lines.append(r'\begin{tabular}{lrrrrrrr}')
    lines.append(r'\hline')
    lines.append(r'Algorithm & Return & Agreement (\%) '
                 r'& $V^*$ RMSE & Max $|V{-}V^*|$ & Ep to $V(s_0)$ & Ep to $V^*$ conv & Time (s) \\')
    lines.append(r'\hline')

    # Planning methods
    lines.append(r'\multicolumn{8}{l}{\emph{Planning (full model)}} \\')
    lines.append(f'VI & {optimal_return:.2f} & 100.0 & 0.00 & 0.00 '
                 f'& --- & {metrics_vi.iterations} iter & {metrics_vi.wall_time:.3f} \\\\')
    lines.append(f'PI & {optimal_return:.2f} & 100.0 & 0.00 & 0.00 '
                 f'& --- & {metrics_pi.iterations} iter & {metrics_pi.wall_time:.3f} \\\\')
    lines.append(r'\hline')

    def _algo_row(name, agg):
        tex_name = name.replace('λ', r'$\lambda$')
        ep_vstar_str = f'{agg.episodes_to_vstar_conv:.0f}' if agg.episodes_to_vstar_conv is not None else '---'
        ep_v_s0_str = f'{agg.episodes_to_v_s0:.0f}' if agg.episodes_to_v_s0 is not None else '---'
        return (
            f'{tex_name} & {agg.return_mean:.2f} '
            f'& {agg.agreement_mean*100:.1f} '
            f'& {agg.value_error_mean:.2f} '
            f'& {agg.max_per_state_error_mean:.2f} '
            f'& {ep_v_s0_str} '
            f'& {ep_vstar_str} '
            f'& {agg.time_mean:.1f} \\\\'
        )

    # TD methods
    lines.append(r'\multicolumn{8}{l}{\emph{Temporal difference}} \\')
    for name in ['Q-Learning', 'SARSA']:
        lines.append(_algo_row(name, agg_results[name]))
    lines.append(r'\hline')

    # Multi-step
    lines.append(r'\multicolumn{8}{l}{\emph{Multi-step}} \\')
    lines.append(_algo_row('Q(λ)', agg_results['Q(λ)']))
    lines.append(r'\hline')

    # Deep RL
    lines.append(r'\multicolumn{8}{l}{\emph{Deep RL}} \\')
    lines.append(_algo_row('DQN', agg_results['DQN']))
    lines.append(r'\hline')

    # Policy gradient
    lines.append(r'\multicolumn{8}{l}{\emph{Policy gradient}} \\')
    for name in ['REINFORCE', 'NPG', 'PPO']:
        lines.append(_algo_row(name, agg_results[name]))

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    path = OUTPUT_DIR / 'gridworld_study_results.tex'
    path.write_text('\n'.join(lines))
    print(f"  Saved: gridworld_study_results.tex")


def generate_hyperparams_table():
    lines = []
    lines.append(r'\begin{tabular}{ll}')
    lines.append(r'\hline')
    lines.append(r'Parameter & Value \\')
    lines.append(r'\hline')
    lines.append(f'Grid size & ${N} \\times {N}$ ({N*N} states, 5 actions) \\\\')
    lines.append(f'Discount $\\gamma$ & {GAMMA} \\\\')
    lines.append(f'Step penalty & {STEP_PENALTY} \\\\')
    lines.append(f'Terminal reward & {TERMINAL_REWARD} \\\\')
    lines.append(f'Training episodes & {NUM_EPISODES:,} \\\\')
    lines.append(f'Episode horizon & {EPISODE_HORIZON} \\\\')
    lines.append(f'Learning rate $\\alpha$ & {ALPHA} (decay {ALPHA_DECAY}/ep) \\\\')
    lines.append(f'Exploration $\\epsilon$ & {EPSILON_START} $\\to$ {EPSILON_END} '
                 f'(decay {EPSILON_DECAY}/ep) \\\\')
    lines.append(f'Trace decay $\\lambda$ & {LAMBDA} \\\\')
    lines.append(f'Eval frequency & {EVAL_FREQ_TABULAR} (tabular), '
                 f'{EVAL_FREQ_DQN} (DQN), {EVAL_FREQ_PG} (PG) \\\\')
    lines.append(f'Eval episodes & {EVAL_EPISODES} \\\\')
    lines.append(f'Seeds & {M} (seeds {SEEDS[0]}--{SEEDS[-1]}) \\\\')
    lines.append(f'$V^*$ convergence threshold & $\\max_s |V(s) - V^*(s)| < {VALUE_CONV_THRESHOLD}$ \\\\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    path = OUTPUT_DIR / 'gridworld_hyperparams.tex'
    path.write_text('\n'.join(lines))
    print(f"  Saved: gridworld_hyperparams.tex")


def _windowed_metric(snapshots, target_ep, metric_fn, window=3):
    """Compute metric at target_ep using a rolling average over nearby snapshots.

    Args:
        snapshots: dict {episode: np.ndarray}
        target_ep: target episode number
        metric_fn: callable(snapshot_array) -> float
        window: number of snapshots on each side (total 2*window+1)
    Returns:
        float: mean of metric_fn across the window of snapshots
    """
    sorted_eps = sorted(snapshots.keys())
    if not sorted_eps:
        return None
    # Find index of nearest snapshot
    idx = min(range(len(sorted_eps)), key=lambda i: abs(sorted_eps[i] - target_ep))
    lo = max(0, idx - window)
    hi = min(len(sorted_eps), idx + window + 1)
    values = [metric_fn(snapshots[sorted_eps[i]]) for i in range(lo, hi)]
    return np.mean(values)


def generate_value_convergence_table(detailed_metrics, V_optimal, output_dir):
    """Table of max |V(s) - V*(s)| at episode checkpoints, with rolling average."""
    checkpoints = [100, 1_000, 10_000, 100_000, 500_000]
    cp_labels = ['100', '1K', '10K', '100K', '500K']

    lines = []
    lines.append(r'\begin{tabular}{l' + 'r' * len(checkpoints) + '}')
    lines.append(r'\hline')
    header = 'Algorithm & ' + ' & '.join(f'Ep {l}' for l in cp_labels) + r' \\'
    lines.append(header)
    lines.append(r'\hline')

    for name in ALGO_KEYS:
        if name not in detailed_metrics or not detailed_metrics[name].value_snapshots:
            continue
        m = detailed_metrics[name]
        tex_name = name.replace('λ', r'$\lambda$')
        cells = []
        for cp in checkpoints:
            val = _windowed_metric(
                m.value_snapshots, cp,
                lambda snap: np.max(np.abs(snap - V_optimal))
            )
            cells.append(f'{val:.2f}' if val is not None else '---')
        lines.append(tex_name + ' & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    path = output_dir / 'gridworld_value_convergence.tex'
    path.write_text('\n'.join(lines))
    print(f"  Saved: gridworld_value_convergence.tex")


def generate_policy_convergence_table(detailed_metrics, policy_optimal, output_dir):
    """Table of % states where pi(s) = pi*(s) at episode checkpoints, with rolling average."""
    checkpoints = [100, 1_000, 10_000, 100_000, 500_000]
    cp_labels = ['100', '1K', '10K', '100K', '500K']
    n_states = len(policy_optimal)

    lines = []
    lines.append(r'\begin{tabular}{l' + 'r' * len(checkpoints) + '}')
    lines.append(r'\hline')
    header = 'Algorithm & ' + ' & '.join(f'Ep {l}' for l in cp_labels) + r' \\'
    lines.append(header)
    lines.append(r'\hline')

    for name in ALGO_KEYS:
        if name not in detailed_metrics or not detailed_metrics[name].policy_snapshots:
            continue
        m = detailed_metrics[name]
        tex_name = name.replace('λ', r'$\lambda$')
        cells = []
        for cp in checkpoints:
            val = _windowed_metric(
                m.policy_snapshots, cp,
                lambda snap: 100.0 * np.mean(snap == policy_optimal)
            )
            cells.append(f'{val:.0f}' if val is not None else '---')
        lines.append(tex_name + ' & ' + ' & '.join(cells) + r' \\')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')

    path = output_dir / 'gridworld_policy_convergence.tex'
    path.write_text('\n'.join(lines))
    print(f"  Saved: gridworld_policy_convergence.tex")


# ---------------------------------------------------------------------------
# Per-state convergence summary (stdout)
# ---------------------------------------------------------------------------

def print_per_state_summary(detailed_metrics, V_optimal, policy_optimal, env):
    print("\n" + "=" * 70)
    print("PHASE 3: PER-STATE VALUE CONVERGENCE (seed 42)")
    print("=" * 70)

    for name in ALGO_KEYS:
        if name not in detailed_metrics:
            continue
        m = detailed_metrics[name]
        if not m.value_snapshots:
            continue

        print(f"\n[{name}]")

        # Final snapshot
        last_ep = max(m.value_snapshots.keys())
        V_final = m.value_snapshots[last_ep]
        errors = np.abs(V_final - V_optimal)
        errors_grid = errors.reshape(N, N)

        print(f"  Final episode: {last_ep}")
        print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.4f}")
        print(f"  Max |V-V*|: {errors.max():.4f} at state {np.argmax(errors)} "
              f"({env.index_to_state(np.argmax(errors))})")
        print(f"  Mean |V-V*|: {errors.mean():.4f}")

        # V(s0) error
        print(f"  |V(s0) - V*(s0)|: {errors[0]:.4f}")

        print(f"  Per-state errors (5x5 grid):")
        for r in range(N):
            vals = '  '.join(f'{errors_grid[r, c]:6.3f}' for c in range(N))
            print(f"    row {r}: {vals}")

        # Find convergence episode for each state (value)
        print(f"  Per-state value convergence episode (threshold={VALUE_CONV_THRESHOLD}):")
        state_conv = np.full(env.num_states, -1, dtype=int)
        for ep in sorted(m.value_snapshots.keys()):
            errs = np.abs(m.value_snapshots[ep] - V_optimal)
            for s_idx in range(env.num_states):
                if state_conv[s_idx] < 0 and errs[s_idx] < VALUE_CONV_THRESHOLD:
                    state_conv[s_idx] = ep

        conv_grid = state_conv.reshape(N, N)
        for r in range(N):
            vals = '  '.join(
                f'{conv_grid[r, c]:6d}' if conv_grid[r, c] >= 0 else '  ----'
                for c in range(N)
            )
            print(f"    row {r}: {vals}")

        print(f"  States never converged (value): "
              f"{np.sum(state_conv < 0)}/{env.num_states}")

        # Per-state policy convergence
        if m.policy_snapshots:
            print(f"  Per-state policy convergence episode (pi(s)=pi*(s)):")
            state_pol_conv = np.full(env.num_states, -1, dtype=int)
            for ep in sorted(m.policy_snapshots.keys()):
                pi_snap = m.policy_snapshots[ep]
                for s_idx in range(env.num_states):
                    if state_pol_conv[s_idx] < 0 and pi_snap[s_idx] == policy_optimal[s_idx]:
                        state_pol_conv[s_idx] = ep

            pol_conv_grid = state_pol_conv.reshape(N, N)
            for r in range(N):
                vals = '  '.join(
                    f'{pol_conv_grid[r, c]:6d}' if pol_conv_grid[r, c] >= 0 else '  ----'
                    for c in range(N)
                )
                print(f"    row {r}: {vals}")
            print(f"  States never converged (policy): "
                  f"{np.sum(state_pol_conv < 0)}/{env.num_states}")

        print(f"  Wall time: {m.wall_time:.2f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='9-algorithm gridworld study')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated cache keys to force re-run '
                             '(e.g. --only dqn,ppo). Others loaded from cache.')
    parser.add_argument('--plots-only', action='store_true',
                        help='Skip computation, regenerate figures/tables from cache.')
    return parser.parse_args()


def main():
    args = parse_args()
    only_algos = args.only.split(',') if args.only else None

    tee = TeeOutput(OUTPUT_DIR / 'gridworld_illustrated_stdout.txt')
    sys.stdout = tee

    print("=" * 70)
    print(f"ILLUSTRATED EXAMPLE: 9 ALGORITHMS ON A {N}x{N} GRIDWORLD")
    print(f"  (2 planning + 7 learning)")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Grid: {N}x{N}, gamma={GAMMA}, episodes={NUM_EPISODES}, seeds={M}")
    if only_algos:
        print(f"Mode: selective re-run ({', '.join(only_algos)})")
    elif args.plots_only:
        print(f"Mode: plots-only (loading all from cache)")
    else:
        print(f"Mode: full run (cache where valid)")
    print()

    env = GridworldEnv(N, GAMMA, STEP_PENALTY, TERMINAL_REWARD,
                       symmetry_break=SYMMETRY_BREAK)

    # Phase 1: Planning (always runs, <1 second)
    (V_optimal, policy_optimal, optimal_return,
     metrics_vi, metrics_pi, V_vi, policy_vi) = run_planning(env)

    # Phase 2: Learning (with caching)
    if args.plots_only:
        # Load everything from cache
        agg_results, detailed_metrics = run_all_learning(
            env, V_optimal, policy_optimal, optimal_return,
            only_algos=[])  # empty list = force-run nothing
    else:
        agg_results, detailed_metrics = run_all_learning(
            env, V_optimal, policy_optimal, optimal_return,
            only_algos=only_algos)

    # Phase 3: Per-state diagnostics
    print_per_state_summary(detailed_metrics, V_optimal, policy_optimal, env)

    # Phase 4: Figures
    print("\n" + "=" * 70)
    print("PHASE 4: GENERATING FIGURES")
    print("=" * 70)

    fig_learning_curves(agg_results)
    fig_value_heatmaps(detailed_metrics, V_optimal, env, optimal_return)
    fig_convergence_fan(detailed_metrics, V_optimal, env)
    fig_policy_convergence(detailed_metrics, policy_optimal, env)
    fig_policy_heatmaps(detailed_metrics, policy_optimal, env, V_optimal)

    # Phase 5: Tables
    print("\n" + "=" * 70)
    print("PHASE 5: GENERATING TABLES")
    print("=" * 70)

    generate_results_table(agg_results, optimal_return, metrics_vi, metrics_pi)
    generate_hyperparams_table()
    generate_value_convergence_table(detailed_metrics, V_optimal, OUTPUT_DIR)
    generate_policy_convergence_table(detailed_metrics, policy_optimal, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    outputs = sorted(OUTPUT_DIR.glob('gridworld_*.png')) + \
              sorted(OUTPUT_DIR.glob('gridworld_*.tex'))
    for f in outputs:
        print(f"  {f.name}")
    cache_files = sorted(CACHE_DIR.glob('*.pkl')) if CACHE_DIR.exists() else []
    if cache_files:
        print(f"\nCACHE ({CACHE_DIR.name}/):")
        for f in cache_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.0f} KB)")

    print("\nDone.")
    sys.stdout = tee.stdout
    tee.close()


if __name__ == '__main__':
    main()
