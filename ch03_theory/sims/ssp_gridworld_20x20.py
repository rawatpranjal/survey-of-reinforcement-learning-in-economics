# SSP Gridworld 20x20 - Comprehensive Algorithm Comparison
# Chapter 2 -- Planning and Learning: Unified Theory
# Compares model-free RL, model-based planning, and Bertsekas framework methods.

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import load_results, save_results, add_cache_args

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import all algorithms from gridworld_algorithms
from gridworld_algorithms import (
    GridworldEnv, AlgorithmMetrics,
    run_value_iteration, run_policy_iteration,
    run_q_learning, run_sarsa, run_expected_sarsa,
    run_mc_control, run_reinforce,
    run_sarsa_lambda, run_q_lambda,
    run_rollout, run_lookahead, run_mpc, run_truncated_rollout,
    v_to_array, policy_to_array, evaluate_policy
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'ssp_gridworld_20x20'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N = 20  # Grid size
GAMMA = 0.99
NUM_EPISODES = 20000
NUM_SEEDS = 10
HORIZON = 200  # Max steps per episode (20x20 needs more steps)
EVAL_FREQ = 500  # Evaluate every 500 episodes
EVAL_EPISODES = 50

# Output paths
OUTPUT_DIR = "/Users/pranjal/Code/rl/ch02_planning_learning/sims"

CONFIG = {
    'N': N, 'GAMMA': GAMMA, 'NUM_EPISODES': NUM_EPISODES,
    'NUM_SEEDS': NUM_SEEDS, 'HORIZON': HORIZON,
    'EVAL_FREQ': EVAL_FREQ, 'EVAL_EPISODES': EVAL_EPISODES,
    'step_penalty': -0.1, 'terminal_reward': 10.0,
    'version': 1,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def smooth_curve(y: List[float], window: int = 100) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(y) < window:
        return np.array(y)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def aggregate_metrics(metrics_list: List[AlgorithmMetrics]) -> Dict:
    """Aggregate metrics across seeds."""
    result = {}

    # Episode returns
    all_returns = [m.episode_returns for m in metrics_list]
    min_len = min(len(r) for r in all_returns)
    returns_arr = np.array([r[:min_len] for r in all_returns])
    result['returns_mean'] = np.mean(returns_arr, axis=0)
    result['returns_std'] = np.std(returns_arr, axis=0)

    # Eval returns
    all_eval = [m.eval_returns for m in metrics_list]
    min_eval_len = min(len(e) for e in all_eval)
    if min_eval_len > 0:
        eval_arr = np.array([e[:min_eval_len] for e in all_eval])
        result['eval_mean'] = np.mean(eval_arr, axis=0)
        result['eval_std'] = np.std(eval_arr, axis=0)
        result['checkpoint_episodes'] = metrics_list[0].checkpoint_episodes[:min_eval_len]
    else:
        result['eval_mean'] = np.array([])
        result['eval_std'] = np.array([])
        result['checkpoint_episodes'] = []

    # Wall time
    result['wall_time_mean'] = np.mean([m.wall_time for m in metrics_list])
    result['wall_time_std'] = np.std([m.wall_time for m in metrics_list])

    # Final eval return
    final_returns = [m.eval_returns[-1] if m.eval_returns else np.nan for m in metrics_list]
    result['final_return_mean'] = np.mean(final_returns)
    result['final_return_std'] = np.std(final_returns)

    # Value errors (if available)
    all_verr = [m.value_errors for m in metrics_list if m.value_errors]
    if all_verr:
        min_verr_len = min(len(v) for v in all_verr)
        if min_verr_len > 0:
            verr_arr = np.array([v[:min_verr_len] for v in all_verr])
            result['value_error_mean'] = np.mean(verr_arr, axis=0)
            result['value_error_std'] = np.std(verr_arr, axis=0)

    return result


def compute_data():
    """Run all experiments and cache results."""
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("=" * 70)
    print("SSP Gridworld 20x20 - Comprehensive Algorithm Comparison")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid size: {N}x{N} ({N*N} states)")
    print(f"  Discount factor: {GAMMA}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"  Episode horizon: {HORIZON}")
    print(f"  Evaluation frequency: every {EVAL_FREQ} episodes")
    print()

    # Step 1: Compute optimal solution via VI
    print("-" * 70)
    print("Step 1: Computing optimal solution via Value Iteration")
    print("-" * 70)

    env = GridworldEnv(N=N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0)
    V_opt, pi_opt, vi_metrics = run_value_iteration(env, tol=1e-10, max_iter=2000)

    V_opt_arr = v_to_array(V_opt, env)
    pi_opt_arr = policy_to_array(pi_opt, env)

    opt_return, opt_steps = evaluate_policy(env, pi_opt, n_episodes=1000, horizon=HORIZON)

    print(f"  VI converged in {vi_metrics.iterations} iterations")
    print(f"  Final residual: {vi_metrics.final_residual:.2e}")
    print(f"  Optimal return: {opt_return:.4f}")
    print(f"  Optimal steps to goal: {opt_steps:.2f}")
    print(f"  Wall time: {vi_metrics.wall_time:.4f}s")
    print()

    # Step 2: Define algorithm configurations
    print("-" * 70)
    print("Step 2: Algorithm Configurations")
    print("-" * 70)

    learning_algorithms = {
        'Q-Learning': {
            'runner': run_q_learning,
            'kwargs': {'alpha': 0.1, 'alpha_decay': 0.9999, 'epsilon_start': 1.0,
                       'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
        'SARSA': {
            'runner': run_sarsa,
            'kwargs': {'alpha': 0.1, 'alpha_decay': 0.9999, 'epsilon_start': 1.0,
                       'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
        'Expected SARSA': {
            'runner': run_expected_sarsa,
            'kwargs': {'alpha': 0.1, 'alpha_decay': 0.9999, 'epsilon_start': 1.0,
                       'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
        'MC Control': {
            'runner': run_mc_control,
            'kwargs': {'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
        'REINFORCE': {
            'runner': run_reinforce,
            'kwargs': {'alpha': 0.005, 'alpha_decay': 0.9999, 'temperature': 0.5}
        },
        'SARSA(0.9)': {
            'runner': run_sarsa_lambda,
            'kwargs': {'alpha': 0.1, 'alpha_decay': 0.9999, 'lambda_': 0.9,
                       'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
        'Q(0.9)': {
            'runner': run_q_lambda,
            'kwargs': {'alpha': 0.1, 'alpha_decay': 0.9999, 'lambda_': 0.9,
                       'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.9995}
        },
    }

    planning_algorithms = {
        'Rollout (heuristic)': {
            'runner': run_rollout,
            'kwargs': {'V_base': None}
        },
        'Rollout (VI-5)': {
            'runner': run_rollout,
            'kwargs': {}
        },
        'Lookahead (l=2)': {
            'runner': run_lookahead,
            'kwargs': {'depth': 2, 'V_base': None}
        },
        'MPC (H=5)': {
            'runner': run_mpc,
            'kwargs': {'horizon_H': 5}
        },
        'MPC (H=10)': {
            'runner': run_mpc,
            'kwargs': {'horizon_H': 10}
        },
        'Truncated Rollout (m=5)': {
            'runner': run_truncated_rollout,
            'kwargs': {'m_warmup': 5}
        },
        'Truncated Rollout (m=10)': {
            'runner': run_truncated_rollout,
            'kwargs': {'m_warmup': 10}
        },
    }

    print("  Computing partial VI solutions for planning baselines...")
    V_vi5 = {}
    for s in env.states:
        if env.is_terminal(s):
            V_vi5[s] = 0.0
        else:
            dist = abs(env.terminal[0] - s[0]) + abs(env.terminal[1] - s[1])
            V_vi5[s] = env.step_penalty * dist + env.terminal_reward * (GAMMA ** dist)

    for _ in range(5):
        V_new = {}
        for s in env.states:
            if env.is_terminal(s):
                V_new[s] = 0.0
                continue
            best_val = float('-inf')
            for a in range(env.num_actions):
                ns = env.get_next_state(s, a)
                r = env.get_reward(s, a, ns)
                val = r + GAMMA * V_vi5[ns]
                if val > best_val:
                    best_val = val
            V_new[s] = best_val
        V_vi5 = V_new

    planning_algorithms['Rollout (VI-5)']['kwargs']['V_base'] = V_vi5

    print(f"  Learning algorithms: {len(learning_algorithms)}")
    for name in learning_algorithms:
        print(f"    - {name}")
    print(f"  Planning algorithms: {len(planning_algorithms)}")
    for name in planning_algorithms:
        print(f"    - {name}")
    print()

    # Step 3: Run all algorithms across seeds
    print("-" * 70)
    print("Step 3: Running Experiments")
    print("-" * 70)

    all_results = {}
    t_start = time.time()

    common_kwargs = {
        'num_episodes': NUM_EPISODES,
        'horizon': HORIZON,
        'V_optimal': V_opt_arr,
        'policy_optimal': pi_opt_arr,
        'optimal_return': opt_return,
        'eval_freq': EVAL_FREQ,
        'eval_episodes': EVAL_EPISODES
    }

    print("\n  Running learning algorithms...")
    learning_pbar = tqdm(learning_algorithms.items(), desc="Learning algorithms", leave=True)
    for alg_name, alg_config in learning_pbar:
        learning_pbar.set_description(f"Learning: {alg_name}")
        metrics_list = []
        seed_pbar = tqdm(range(NUM_SEEDS), desc=f"Seeds", leave=False)
        for seed in seed_pbar:
            env_run = GridworldEnv(N=N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0)
            kwargs = {**common_kwargs, **alg_config['kwargs'], 'seed': seed}
            _, _, metrics = alg_config['runner'](env_run, **kwargs)
            metrics_list.append(metrics)
        all_results[alg_name] = aggregate_metrics(metrics_list)
        tqdm.write(f"    {alg_name}: {all_results[alg_name]['final_return_mean']:.4f} ({all_results[alg_name]['wall_time_mean']:.2f}s/seed)")

    print("\n  Running planning algorithms...")
    planning_pbar = tqdm(planning_algorithms.items(), desc="Planning algorithms", leave=True)
    for alg_name, alg_config in planning_pbar:
        planning_pbar.set_description(f"Planning: {alg_name}")
        metrics_list = []
        seed_pbar = tqdm(range(NUM_SEEDS), desc=f"Seeds", leave=False)
        for seed in seed_pbar:
            env_run = GridworldEnv(N=N, gamma=GAMMA, step_penalty=-0.1, terminal_reward=10.0)
            kwargs = {**alg_config['kwargs'], 'seed': seed}
            kwargs['num_episodes'] = NUM_EPISODES
            kwargs['horizon'] = HORIZON if 'horizon' in alg_config['runner'].__code__.co_varnames else None
            if kwargs['horizon'] is None:
                kwargs.pop('horizon', None)
                if 'episode_horizon' in alg_config['runner'].__code__.co_varnames:
                    kwargs['episode_horizon'] = HORIZON
            kwargs['V_optimal'] = V_opt_arr
            kwargs['policy_optimal'] = pi_opt_arr
            kwargs['optimal_return'] = opt_return
            kwargs['eval_freq'] = EVAL_FREQ
            kwargs['eval_episodes'] = EVAL_EPISODES
            _, _, metrics = alg_config['runner'](env_run, **kwargs)
            metrics_list.append(metrics)
        all_results[alg_name] = aggregate_metrics(metrics_list)
        tqdm.write(f"    {alg_name}: {all_results[alg_name]['final_return_mean']:.4f} ({all_results[alg_name]['wall_time_mean']:.2f}s/seed)")

    t_total = time.time() - t_start
    print(f"\n  Total experiment time: {t_total:.1f}s")

    # Serialize: convert numpy arrays to lists for pickling safety
    serialized_results = {}
    for alg_name, res in all_results.items():
        sr = {}
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serialized_results[alg_name] = sr

    data = {
        'all_results': serialized_results,
        'opt_return': opt_return,
        'opt_steps': opt_steps,
        'vi_wall_time': vi_metrics.wall_time,
        'vi_iterations': vi_metrics.iterations,
        'vi_final_residual': vi_metrics.final_residual,
        'learning_alg_names': list(learning_algorithms.keys()),
        'planning_alg_names': list(planning_algorithms.keys()),
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    """Generate all figures, tables, and analysis from cached data."""
    # Reconstruct numpy arrays
    all_results = {}
    for alg_name, res in data['all_results'].items():
        r = {}
        for k, v in res.items():
            if isinstance(v, list):
                r[k] = np.array(v) if k not in ('checkpoint_episodes',) else v
            else:
                r[k] = v
        # Convert checkpoint_episodes to list if needed
        if 'checkpoint_episodes' in r and isinstance(r['checkpoint_episodes'], np.ndarray):
            r['checkpoint_episodes'] = r['checkpoint_episodes'].tolist()
        all_results[alg_name] = r

    opt_return = data['opt_return']
    vi_wall_time = data['vi_wall_time']
    learning_alg_names = data['learning_alg_names']
    planning_alg_names = data['planning_alg_names']

    # Step 4: Results Summary Table
    print("-" * 70)
    print("Step 4: Results Summary")
    print("-" * 70)

    print("\n  Algorithm Performance Summary (20x20 SSP Gridworld)")
    print("  " + "=" * 68)
    print(f"  {'Algorithm':<25} {'Final Return':>14} {'Std':>8} {'Time (s)':>10}")
    print("  " + "-" * 68)

    sorted_algs = sorted(all_results.items(), key=lambda x: -x[1]['final_return_mean'])

    for alg_name, res in sorted_algs:
        print(f"  {alg_name:<25} {res['final_return_mean']:>14.4f} {res['final_return_std']:>8.4f} {res['wall_time_mean']:>10.2f}")

    print("  " + "=" * 68)
    print(f"  {'Optimal (VI)':<25} {opt_return:>14.4f} {0.0:>8.4f} {vi_wall_time:>10.4f}")
    print()

    # Step 5: Generate Figures
    print("-" * 70)
    print("Step 5: Generating Figures")
    print("-" * 70)

    # Figure 1: Learning curves
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    colors_learning = plt.cm.tab10(np.linspace(0, 1, len(learning_alg_names)))
    colors_planning = plt.cm.Set2(np.linspace(0, 1, len(planning_alg_names)))

    for i, alg_name in enumerate(learning_alg_names):
        if alg_name in all_results:
            d = all_results[alg_name]
            cp = d.get('checkpoint_episodes', [])
            if len(cp) > 0:
                em = np.array(d['eval_mean']) if not isinstance(d['eval_mean'], np.ndarray) else d['eval_mean']
                es = np.array(d['eval_std']) if not isinstance(d['eval_std'], np.ndarray) else d['eval_std']
                ax1.plot(cp, em, label=alg_name, color=colors_learning[i], linewidth=2)
                ax1.fill_between(cp, em - es, em + es, color=colors_learning[i], alpha=0.15)

    for i, alg_name in enumerate(planning_alg_names):
        if alg_name in all_results:
            d = all_results[alg_name]
            cp = d.get('checkpoint_episodes', [])
            if len(cp) > 0:
                em = np.array(d['eval_mean']) if not isinstance(d['eval_mean'], np.ndarray) else d['eval_mean']
                ax1.plot(cp, em, label=alg_name, color=colors_planning[i], linewidth=2, linestyle='--')

    ax1.axhline(y=opt_return, color='k', linestyle=':', linewidth=2, label='Optimal')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Evaluation Return', fontsize=12)
    ax1.set_title(f'Learning Curves: 20x20 SSP Gridworld (N={N}, \u03b3={GAMMA})', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, NUM_EPISODES)

    fig1.tight_layout()
    fig1_path = f"{OUTPUT_DIR}/ssp_gridworld_20x20_learning_curves.png"
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")

    # Figure 2: Planning vs Learning comparison
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax2a = axes[0]
    early_eps = 5000
    for i, alg_name in enumerate(['Q-Learning', 'SARSA', 'Expected SARSA']):
        if alg_name in all_results:
            d = all_results[alg_name]
            eps = np.array(d['checkpoint_episodes'])
            mask = eps <= early_eps
            if np.any(mask):
                em = np.array(d['eval_mean']) if not isinstance(d['eval_mean'], np.ndarray) else d['eval_mean']
                ax2a.plot(eps[mask], em[mask], label=alg_name, linewidth=2)

    for i, alg_name in enumerate(['Rollout (heuristic)', 'MPC (H=5)', 'Truncated Rollout (m=5)']):
        if alg_name in all_results:
            d = all_results[alg_name]
            eps = np.array(d['checkpoint_episodes'])
            mask = eps <= early_eps
            if np.any(mask):
                em = np.array(d['eval_mean']) if not isinstance(d['eval_mean'], np.ndarray) else d['eval_mean']
                ax2a.plot(eps[mask], em[mask], label=alg_name, linewidth=2, linestyle='--')

    ax2a.axhline(y=opt_return, color='k', linestyle=':', linewidth=2, label='Optimal')
    ax2a.set_xlabel('Episode', fontsize=12)
    ax2a.set_ylabel('Evaluation Return', fontsize=12)
    ax2a.set_title('Early Learning: Planning vs Model-Free RL', fontsize=12)
    ax2a.legend(fontsize=9)
    ax2a.grid(True, alpha=0.3)

    ax2b = axes[1]
    alg_names_sorted = list(sorted_algs)[:10]
    returns = [all_results[name]['final_return_mean'] for name, _ in alg_names_sorted]
    stds = [all_results[name]['final_return_std'] for name, _ in alg_names_sorted]
    names = [name for name, _ in alg_names_sorted]

    y_pos = np.arange(len(names))
    colors_bar = ['#2ecc71' if 'Rollout' in n or 'MPC' in n or 'Truncated' in n or 'Lookahead' in n
                  else '#3498db' for n in names]
    ax2b.barh(y_pos, returns, xerr=stds, color=colors_bar, alpha=0.8, capsize=3)
    ax2b.axvline(x=opt_return, color='r', linestyle='--', linewidth=2, label='Optimal')
    ax2b.set_yticks(y_pos)
    ax2b.set_yticklabels(names, fontsize=10)
    ax2b.set_xlabel('Final Return', fontsize=12)
    ax2b.set_title('Final Performance Comparison', fontsize=12)
    ax2b.legend()
    ax2b.invert_yaxis()

    fig2.tight_layout()
    fig2_path = f"{OUTPUT_DIR}/ssp_gridworld_20x20_planning_vs_learning.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")

    plt.close('all')

    # Step 6: Generate LaTeX Table
    print("-" * 70)
    print("Step 6: Generating LaTeX Table")
    print("-" * 70)

    latex_table = r"""\begin{tabular}{lrrr}
\toprule
Algorithm & Final Return & Std. Error & Time (s) \\
\midrule
"""
    latex_table += f"Optimal (VI) & {opt_return:.4f} & -- & {vi_wall_time:.4f} \\\\\n"
    latex_table += r"\midrule" + "\n"

    latex_table += r"\multicolumn{4}{l}{\textit{Model-Free RL}} \\" + "\n"
    for alg_name in learning_alg_names:
        if alg_name in all_results:
            res = all_results[alg_name]
            se = res['final_return_std'] / np.sqrt(NUM_SEEDS)
            latex_table += f"{alg_name} & {res['final_return_mean']:.4f} & {se:.4f} & {res['wall_time_mean']:.2f} \\\\\n"

    latex_table += r"\midrule" + "\n"

    latex_table += r"\multicolumn{4}{l}{\textit{Planning Methods (Bertsekas Framework)}} \\" + "\n"
    for alg_name in planning_alg_names:
        if alg_name in all_results:
            res = all_results[alg_name]
            se = res['final_return_std'] / np.sqrt(NUM_SEEDS)
            latex_table += f"{alg_name} & {res['final_return_mean']:.4f} & {se:.4f} & {res['wall_time_mean']:.2f} \\\\\n"

    latex_table += r"""\bottomrule
\end{tabular}
"""

    latex_path = f"{OUTPUT_DIR}/ssp_gridworld_20x20_results.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"  Saved: {latex_path}")

    # Step 7: Detailed Analysis
    print("-" * 70)
    print("Step 7: Detailed Analysis")
    print("-" * 70)

    print("\n  Analysis 1: Planning methods achieve near-optimal from episode 1")
    print("  " + "-" * 60)
    for alg_name in ['Rollout (heuristic)', 'Rollout (VI-5)', 'MPC (H=5)', 'MPC (H=10)',
                     'Truncated Rollout (m=5)', 'Truncated Rollout (m=10)']:
        if alg_name in all_results:
            res = all_results[alg_name]
            em = np.array(res['eval_mean']) if not isinstance(res['eval_mean'], np.ndarray) else res['eval_mean']
            if len(em) > 0:
                first_eval = em[0]
                gap = opt_return - first_eval
                pct_opt = (first_eval / opt_return) * 100
                print(f"    {alg_name:<30}: {first_eval:.4f} ({pct_opt:.1f}% of optimal, gap={gap:.4f})")

    print("\n  Analysis 2: Sample efficiency (episodes to reach 95% of optimal)")
    print("  " + "-" * 60)
    target = 0.95 * opt_return
    for alg_name, res in sorted_algs:
        em = np.array(res['eval_mean']) if not isinstance(res['eval_mean'], np.ndarray) else res['eval_mean']
        if len(em) > 0:
            eps = np.array(res['checkpoint_episodes'])
            vals = em
            reached = np.where(vals >= target)[0]
            if len(reached) > 0:
                first_idx = reached[0]
                print(f"    {alg_name:<30}: episode {eps[first_idx]}")
            else:
                print(f"    {alg_name:<30}: never reached")

    print("\n  Analysis 3: Truncated rollout warmup benefit")
    print("  " + "-" * 60)
    if 'Rollout (heuristic)' in all_results and 'Truncated Rollout (m=5)' in all_results:
        r_heur = all_results['Rollout (heuristic)']['final_return_mean']
        r_trunc5 = all_results['Truncated Rollout (m=5)']['final_return_mean']
        r_trunc10 = all_results.get('Truncated Rollout (m=10)', {}).get('final_return_mean', np.nan)
        print(f"    Rollout (heuristic):      {r_heur:.4f}")
        print(f"    Truncated Rollout (m=5):  {r_trunc5:.4f} (improvement: {r_trunc5 - r_heur:.4f})")
        print(f"    Truncated Rollout (m=10): {r_trunc10:.4f} (improvement: {r_trunc10 - r_heur:.4f})")

    print("\n  Analysis 4: MPC horizon effect")
    print("  " + "-" * 60)
    if 'MPC (H=5)' in all_results and 'MPC (H=10)' in all_results:
        mpc5 = all_results['MPC (H=5)']['final_return_mean']
        mpc10 = all_results['MPC (H=10)']['final_return_mean']
        print(f"    MPC (H=5):  {mpc5:.4f}")
        print(f"    MPC (H=10): {mpc10:.4f} (improvement: {mpc10 - mpc5:.4f})")
        print(f"    Optimal:    {opt_return:.4f}")

    print()

    # Final summary
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {fig1_path}")
    print(f"  - {fig2_path}")
    print(f"  - {latex_path}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSP Gridworld 20x20 Algorithm Comparison')
    add_cache_args(parser)
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()

    if not args.data_only:
        generate_outputs(data)
