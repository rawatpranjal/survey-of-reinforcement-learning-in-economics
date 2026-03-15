# Gridworld Algorithm Comparison Study
# Chapter 3 -- Comprehensive Simulation Study
# Compares classical RL algorithms on 10x10 gridworld with M=30 Monte Carlo replications.

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import load_results, save_results, add_cache_args

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'gridworld_study'
CONFIG = {
    'N': 10, 'GAMMA': 0.95,
    'STEP_PENALTY': -0.1, 'TERMINAL_REWARD': 10.0,
    'NUM_EPISODES': 5000, 'EPISODE_HORIZON': 100,
    'EVAL_FREQ': 100, 'EVAL_EPISODES': 100,
    'ALPHA': 0.1, 'ALPHA_DECAY': 0.999,
    'EPSILON_START': 1.0, 'EPSILON_END': 0.01, 'EPSILON_DECAY': 0.995,
    'LAMBDA': 0.9, 'M': 30,
    'version': 1,
}

from gridworld_algorithms import (
    GridworldEnv,
    run_value_iteration, run_policy_iteration,
    run_q_learning, run_sarsa, run_expected_sarsa,
    run_mc_control, run_sarsa_lambda, run_q_lambda,
    run_reinforce, run_dqn_tabular_comparison,
    v_to_array, policy_to_array, evaluate_policy,
    AlgorithmMetrics
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment parameters
N = 10                          # Grid size (100 states)
GAMMA = 0.95                    # Discount factor
STEP_PENALTY = -0.1             # Per-step cost
TERMINAL_REWARD = 10.0          # Goal reward

# Training parameters
NUM_EPISODES = 5000             # Training episodes
EPISODE_HORIZON = 100           # Max steps per episode
EVAL_FREQ = 100                 # Evaluation frequency
EVAL_EPISODES = 100             # Evaluation rollouts

# TD/MC hyperparameters
ALPHA = 0.1                     # Learning rate
ALPHA_DECAY = 0.999             # Per-episode decay
EPSILON_START = 1.0             # Initial exploration
EPSILON_END = 0.01              # Final exploration
EPSILON_DECAY = 0.995           # Per-episode decay
LAMBDA = 0.9                    # Eligibility trace decay

# Monte Carlo replications
M = 30                          # Number of seeds
SEEDS = list(range(42, 42 + M))  # Seeds: 42, 43, ..., 71

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent

# Algorithm names for display
ALGORITHM_NAMES = [
    'VI', 'PI', 'Q-Learning', 'SARSA', 'Expected SARSA',
    'MC Control', 'SARSA(λ)', 'Q(λ)', 'REINFORCE', 'DQN'
]

# Colors for plotting
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmResult:
    """Aggregated results for one algorithm across all seeds."""
    name: str
    # Final performance
    return_mean: float
    return_std: float
    steps_mean: float
    steps_std: float
    # Policy quality
    agreement_mean: float
    agreement_std: float
    value_error_mean: float
    value_error_std: float
    # Efficiency
    time_mean: float
    time_std: float
    coverage_mean: float
    coverage_std: float
    # Convergence
    episodes_to_95: float  # Mean episodes to 95% of optimal
    # Learning curves (for plotting)
    eval_returns_all: List[List[float]]  # [seed][checkpoint]
    value_errors_all: List[List[float]]
    agreements_all: List[List[float]]
    regrets_all: List[List[float]]


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------

class TeeOutput:
    """Write to both file and stdout."""
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
# Main study functions
# ---------------------------------------------------------------------------

def print_header():
    """Print study configuration."""
    print("=" * 70)
    print("GRIDWORLD ALGORITHM COMPARISON STUDY")
    print("=" * 70)
    print()
    print("Environment Configuration:")
    print(f"  Grid size:        {N}x{N} ({N*N} states)")
    print(f"  Actions:          5 (Left, Right, Up, Down, Stay)")
    print(f"  Discount (γ):     {GAMMA}")
    print(f"  Step penalty:     {STEP_PENALTY}")
    print(f"  Terminal reward:  {TERMINAL_REWARD}")
    print()
    print("Training Configuration:")
    print(f"  Episodes:         {NUM_EPISODES}")
    print(f"  Horizon:          {EPISODE_HORIZON}")
    print(f"  Eval frequency:   {EVAL_FREQ}")
    print(f"  Eval episodes:    {EVAL_EPISODES}")
    print()
    print("Hyperparameters:")
    print(f"  α (learning rate):    {ALPHA}")
    print(f"  α decay:              {ALPHA_DECAY}")
    print(f"  ε (start/end/decay):  {EPSILON_START}/{EPSILON_END}/{EPSILON_DECAY}")
    print(f"  λ (trace decay):      {LAMBDA}")
    print()
    print("Monte Carlo:")
    print(f"  Replications (M):     {M}")
    print(f"  Seeds:                {SEEDS[0]} to {SEEDS[-1]}")
    print()


def run_planning_methods(env: GridworldEnv) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """Run VI and PI, return optimal value/policy."""
    print("-" * 70)
    print("PLANNING METHODS (Model-Based)")
    print("-" * 70)

    results = {}

    # Value Iteration
    print("\n[VI] Running Value Iteration...")
    V_vi, policy_vi, metrics_vi = run_value_iteration(env)
    V_optimal = v_to_array(V_vi, env)
    policy_optimal = policy_to_array(policy_vi, env)
    optimal_return, optimal_steps = evaluate_policy(env, policy_vi,
                                                    n_episodes=EVAL_EPISODES,
                                                    horizon=EPISODE_HORIZON)

    print(f"      Iterations: {metrics_vi.iterations}")
    print(f"      Residual:   {metrics_vi.final_residual:.2e}")
    print(f"      Time:       {metrics_vi.wall_time:.4f}s")
    print(f"      Return:     {optimal_return:.4f}")
    print(f"      Steps:      {optimal_steps:.2f}")

    results['VI'] = AlgorithmResult(
        name='VI',
        return_mean=optimal_return, return_std=0.0,
        steps_mean=optimal_steps, steps_std=0.0,
        agreement_mean=1.0, agreement_std=0.0,
        value_error_mean=0.0, value_error_std=0.0,
        time_mean=metrics_vi.wall_time, time_std=0.0,
        coverage_mean=1.0, coverage_std=0.0,
        episodes_to_95=0,
        eval_returns_all=[[optimal_return]],
        value_errors_all=[[0.0]],
        agreements_all=[[1.0]],
        regrets_all=[[]]
    )

    # Policy Iteration
    print("\n[PI] Running Policy Iteration...")
    V_pi, policy_pi, metrics_pi = run_policy_iteration(env)
    pi_return, pi_steps = evaluate_policy(env, policy_pi,
                                          n_episodes=EVAL_EPISODES,
                                          horizon=EPISODE_HORIZON)

    print(f"      Iterations: {metrics_pi.iterations}")
    print(f"      Time:       {metrics_pi.wall_time:.4f}s")
    print(f"      Return:     {pi_return:.4f}")
    print(f"      Steps:      {pi_steps:.2f}")

    # Verify PI matches VI
    policy_pi_arr = policy_to_array(policy_pi, env)
    pi_agreement = np.mean(policy_pi_arr == policy_optimal)
    print(f"      Agreement with VI: {pi_agreement:.2%}")

    results['PI'] = AlgorithmResult(
        name='PI',
        return_mean=pi_return, return_std=0.0,
        steps_mean=pi_steps, steps_std=0.0,
        agreement_mean=pi_agreement, agreement_std=0.0,
        value_error_mean=0.0, value_error_std=0.0,
        time_mean=metrics_pi.wall_time, time_std=0.0,
        coverage_mean=1.0, coverage_std=0.0,
        episodes_to_95=0,
        eval_returns_all=[[pi_return]],
        value_errors_all=[[0.0]],
        agreements_all=[[pi_agreement]],
        regrets_all=[[]]
    )

    return V_optimal, policy_optimal, optimal_return, results


def run_learning_algorithm(name: str, run_fn, env: GridworldEnv,
                           V_optimal: np.ndarray, policy_optimal: np.ndarray,
                           optimal_return: float, **kwargs) -> AlgorithmResult:
    """Run learning algorithm across all seeds."""
    print(f"\n[{name}] Running {M} seeds...")

    all_metrics = []
    final_returns = []
    final_steps = []
    final_agreements = []
    final_value_errors = []
    times = []
    coverages = []
    episodes_to_95_list = []

    for i, seed in enumerate(SEEDS):
        env_copy = GridworldEnv(N, GAMMA, STEP_PENALTY, TERMINAL_REWARD)
        _, policy, metrics = run_fn(
            env_copy,
            num_episodes=NUM_EPISODES,
            horizon=EPISODE_HORIZON,
            seed=seed,
            V_optimal=V_optimal,
            policy_optimal=policy_optimal,
            optimal_return=optimal_return,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            **kwargs
        )

        all_metrics.append(metrics)

        # Final evaluation
        eval_return, eval_steps = evaluate_policy(env_copy, policy,
                                                   n_episodes=EVAL_EPISODES,
                                                   horizon=EPISODE_HORIZON)
        final_returns.append(eval_return)
        final_steps.append(eval_steps)

        # Final agreement and value error
        if metrics.policy_agreements:
            final_agreements.append(metrics.policy_agreements[-1])
        else:
            final_agreements.append(0.0)

        if metrics.value_errors:
            final_value_errors.append(metrics.value_errors[-1])
        else:
            final_value_errors.append(float('inf'))

        times.append(metrics.wall_time)
        coverages.append(len(metrics.states_visited) / env_copy.num_states)

        # Episodes to 95% of optimal
        threshold = 0.95 * optimal_return
        ep_95 = NUM_EPISODES
        for j, r in enumerate(metrics.eval_returns):
            if r >= threshold:
                ep_95 = metrics.checkpoint_episodes[j]
                break
        episodes_to_95_list.append(ep_95)

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"      Seed {seed}: return={eval_return:.2f}, "
                  f"agr={final_agreements[-1]:.1%}, time={metrics.wall_time:.2f}s")

    # Aggregate learning curves
    eval_returns_all = [m.eval_returns for m in all_metrics]
    value_errors_all = [m.value_errors for m in all_metrics]
    agreements_all = [m.policy_agreements for m in all_metrics]
    regrets_all = [m.cumulative_regret for m in all_metrics]

    result = AlgorithmResult(
        name=name,
        return_mean=np.mean(final_returns),
        return_std=np.std(final_returns),
        steps_mean=np.mean(final_steps),
        steps_std=np.std(final_steps),
        agreement_mean=np.mean(final_agreements),
        agreement_std=np.std(final_agreements),
        value_error_mean=np.mean(final_value_errors),
        value_error_std=np.std(final_value_errors),
        time_mean=np.mean(times),
        time_std=np.std(times),
        coverage_mean=np.mean(coverages),
        coverage_std=np.std(coverages),
        episodes_to_95=np.mean(episodes_to_95_list),
        eval_returns_all=eval_returns_all,
        value_errors_all=value_errors_all,
        agreements_all=agreements_all,
        regrets_all=regrets_all
    )

    print(f"      Mean return: {result.return_mean:.3f} ± {result.return_std:.3f}")
    print(f"      Mean agreement: {result.agreement_mean:.1%} ± {result.agreement_std:.1%}")
    print(f"      Mean time: {result.time_mean:.2f}s")
    print(f"      Episodes to 95%: {result.episodes_to_95:.0f}")

    return result


def run_all_learning_methods(env: GridworldEnv, V_optimal: np.ndarray,
                             policy_optimal: np.ndarray,
                             optimal_return: float) -> Dict[str, AlgorithmResult]:
    """Run all learning algorithms."""
    print()
    print("-" * 70)
    print("LEARNING METHODS (Model-Free)")
    print("-" * 70)

    results = {}

    # Q-Learning
    results['Q-Learning'] = run_learning_algorithm(
        'Q-Learning', run_q_learning, env, V_optimal, policy_optimal, optimal_return,
        alpha=ALPHA, alpha_decay=ALPHA_DECAY,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # SARSA
    results['SARSA'] = run_learning_algorithm(
        'SARSA', run_sarsa, env, V_optimal, policy_optimal, optimal_return,
        alpha=ALPHA, alpha_decay=ALPHA_DECAY,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # Expected SARSA
    results['Expected SARSA'] = run_learning_algorithm(
        'Expected SARSA', run_expected_sarsa, env, V_optimal, policy_optimal, optimal_return,
        alpha=ALPHA, alpha_decay=ALPHA_DECAY,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # MC Control
    results['MC Control'] = run_learning_algorithm(
        'MC Control', run_mc_control, env, V_optimal, policy_optimal, optimal_return,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # SARSA(λ)
    results['SARSA(λ)'] = run_learning_algorithm(
        'SARSA(λ)', run_sarsa_lambda, env, V_optimal, policy_optimal, optimal_return,
        alpha=ALPHA, alpha_decay=ALPHA_DECAY, lambda_=LAMBDA,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # Q(λ)
    results['Q(λ)'] = run_learning_algorithm(
        'Q(λ)', run_q_lambda, env, V_optimal, policy_optimal, optimal_return,
        alpha=ALPHA, alpha_decay=ALPHA_DECAY, lambda_=LAMBDA,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY
    )

    # REINFORCE
    results['REINFORCE'] = run_learning_algorithm(
        'REINFORCE', run_reinforce, env, V_optimal, policy_optimal, optimal_return,
        alpha=0.01, alpha_decay=0.9995, temperature=1.0, baseline=True
    )

    # DQN
    results['DQN'] = run_learning_algorithm(
        'DQN', run_dqn_tabular_comparison, env, V_optimal, policy_optimal, optimal_return,
        replay_size=10000, batch_size=64, lr=1e-3,
        epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY,
        target_update_freq=50, hidden_dim=64
    )

    return results


def print_results_table(results: Dict[str, AlgorithmResult], optimal_return: float):
    """Print formatted results table."""
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Header
    header = f"{'Algorithm':<16} {'Return':>12} {'Steps':>10} {'Agreement':>12} {'V Error':>10} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    # Sort by return (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1].return_mean, reverse=True)

    for name, r in sorted_results:
        if r.return_std > 0:
            ret_str = f"{r.return_mean:.2f}±{r.return_std:.2f}"
        else:
            ret_str = f"{r.return_mean:.2f}"

        if r.steps_std > 0:
            steps_str = f"{r.steps_mean:.1f}±{r.steps_std:.1f}"
        else:
            steps_str = f"{r.steps_mean:.1f}"

        if r.agreement_std > 0:
            agr_str = f"{r.agreement_mean:.1%}±{r.agreement_std:.1%}"
        else:
            agr_str = f"{r.agreement_mean:.0%}"

        if r.value_error_mean < float('inf'):
            if r.value_error_std > 0:
                verr_str = f"{r.value_error_mean:.3f}±{r.value_error_std:.3f}"
            else:
                verr_str = f"{r.value_error_mean:.3f}"
        else:
            verr_str = "---"

        if r.time_std > 0:
            time_str = f"{r.time_mean:.2f}±{r.time_std:.2f}"
        else:
            time_str = f"{r.time_mean:.4f}"

        print(f"{name:<16} {ret_str:>12} {steps_str:>10} {agr_str:>12} {verr_str:>10} {time_str:>10}")

    print()
    print(f"Optimal return (VI): {optimal_return:.4f}")


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_learning_curves_figure(results: Dict[str, AlgorithmResult],
                                optimal_return: float):
    """Learning curves: episode return vs training episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    checkpoint_episodes = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

    for i, (name, r) in enumerate(results.items()):
        if name in ['VI', 'PI']:
            continue  # Skip planning methods

        curves = r.eval_returns_all
        if not curves or not curves[0]:
            continue

        # Align curves to same length
        min_len = min(len(c) for c in curves)
        curves_arr = np.array([c[:min_len] for c in curves])
        eps = checkpoint_episodes[:min_len]

        mean = np.mean(curves_arr, axis=0)
        std = np.std(curves_arr, axis=0)

        color = COLORS[ALGORITHM_NAMES.index(name) % len(COLORS)]
        ax.plot(eps, mean, color=color, linewidth=1.5, label=name)
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)

    # Optimal reference
    ax.axhline(optimal_return, color='black', linestyle='--', linewidth=1.5,
               label=f'Optimal ({optimal_return:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Evaluation Return', fontsize=11)
    ax.set_title('Gridworld Learning Curves (M=30 seeds)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = OUTPUT_DIR / 'gridworld_learning_curves.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_convergence_figure(results: Dict[str, AlgorithmResult]):
    """Value error vs training episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    checkpoint_episodes = list(range(EVAL_FREQ, NUM_EPISODES + 1, EVAL_FREQ))

    for i, (name, r) in enumerate(results.items()):
        if name in ['VI', 'PI']:
            continue

        curves = r.value_errors_all
        if not curves or not curves[0]:
            continue

        min_len = min(len(c) for c in curves)
        curves_arr = np.array([c[:min_len] for c in curves])
        eps = checkpoint_episodes[:min_len]

        mean = np.mean(curves_arr, axis=0)
        std = np.std(curves_arr, axis=0)

        color = COLORS[ALGORITHM_NAMES.index(name) % len(COLORS)]
        ax.plot(eps, mean, color=color, linewidth=1.5, label=name)
        ax.fill_between(eps, np.maximum(0, mean - std), mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Value RMSE', fontsize=11)
    ax.set_title('Value Function Convergence (M=30 seeds)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = OUTPUT_DIR / 'gridworld_convergence.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_regret_figure(results: Dict[str, AlgorithmResult]):
    """Cumulative regret vs episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, r) in enumerate(results.items()):
        if name in ['VI', 'PI']:
            continue

        regrets = r.regrets_all
        if not regrets or not regrets[0]:
            continue

        # Downsample for plotting (every 10 episodes)
        min_len = min(len(c) for c in regrets)
        step = 10
        indices = list(range(0, min_len, step))
        regrets_arr = np.array([[c[j] for j in indices] for c in regrets])
        eps = [j + 1 for j in indices]

        mean = np.mean(regrets_arr, axis=0)
        std = np.std(regrets_arr, axis=0)

        color = COLORS[ALGORITHM_NAMES.index(name) % len(COLORS)]
        ax.plot(eps, mean, color=color, linewidth=1.5, label=name)
        ax.fill_between(eps, np.maximum(0, mean - std), mean + std, color=color, alpha=0.15)

    ax.set_xlabel('Episodes', fontsize=11)
    ax.set_ylabel('Cumulative Regret', fontsize=11)
    ax.set_title('Cumulative Regret vs Optimal (M=30 seeds)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = OUTPUT_DIR / 'gridworld_regret.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_sample_efficiency_figure(results: Dict[str, AlgorithmResult],
                                  optimal_return: float):
    """Bar chart: episodes to 95% of optimal."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    eps_95 = []
    colors = []

    for name, r in results.items():
        if name in ['VI', 'PI']:
            continue
        names.append(name)
        eps_95.append(r.episodes_to_95)
        colors.append(COLORS[ALGORITHM_NAMES.index(name) % len(COLORS)])

    # Sort by episodes (ascending)
    sorted_idx = np.argsort(eps_95)
    names = [names[i] for i in sorted_idx]
    eps_95 = [eps_95[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(names, eps_95, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Episodes to 95% of Optimal', fontsize=11)
    ax.set_title('Sample Efficiency (M=30 seeds)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, eps_95):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=9)

    out_path = OUTPUT_DIR / 'gridworld_sample_efficiency.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_compute_time_figure(results: Dict[str, AlgorithmResult]):
    """Bar chart: wall-clock time comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    times_mean = []
    times_std = []
    colors = []

    for name, r in results.items():
        names.append(name)
        times_mean.append(r.time_mean)
        times_std.append(r.time_std)
        idx = ALGORITHM_NAMES.index(name) if name in ALGORITHM_NAMES else 0
        colors.append(COLORS[idx % len(COLORS)])

    # Sort by time (ascending)
    sorted_idx = np.argsort(times_mean)
    names = [names[i] for i in sorted_idx]
    times_mean = [times_mean[i] for i in sorted_idx]
    times_std = [times_std[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(names, times_mean, xerr=times_std, color=colors,
                   edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=11)
    ax.set_title('Computation Time (M=30 seeds)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = OUTPUT_DIR / 'gridworld_compute_time.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def make_results_table(results: Dict[str, AlgorithmResult], optimal_return: float):
    """Generate main results LaTeX table."""
    lines = []
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Algorithm & Return & Steps & Agreement & Value Err & Time (s) & Ep to 95\% \\')
    lines.append(r'\midrule')

    # Sort by return (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1].return_mean, reverse=True)

    for name, r in sorted_results:
        if r.return_std > 0:
            ret_str = f"${r.return_mean:.2f} \\pm {r.return_std:.2f}$"
        else:
            ret_str = f"${r.return_mean:.2f}$"

        if r.steps_std > 0:
            steps_str = f"${r.steps_mean:.1f} \\pm {r.steps_std:.1f}$"
        else:
            steps_str = f"${r.steps_mean:.1f}$"

        agr_pct = r.agreement_mean * 100
        if r.agreement_std > 0:
            agr_str = f"${agr_pct:.1f}\\%$"
        else:
            agr_str = f"${agr_pct:.0f}\\%$"

        if r.value_error_mean < float('inf') and r.value_error_mean > 0:
            if r.value_error_std > 0:
                verr_str = f"${r.value_error_mean:.3f}$"
            else:
                verr_str = f"${r.value_error_mean:.3f}$"
        else:
            verr_str = "$0.000$"

        if r.time_std > 0:
            time_str = f"${r.time_mean:.2f}$"
        else:
            time_str = f"${r.time_mean:.4f}$"

        if name in ['VI', 'PI']:
            ep95_str = "---"
        else:
            ep95_str = f"${r.episodes_to_95:.0f}$"

        # Escape special characters
        name_tex = name.replace('(λ)', '($\\lambda$)')

        lines.append(f"{name_tex} & {ret_str} & {steps_str} & {agr_str} & "
                     f"{verr_str} & {time_str} & {ep95_str} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'gridworld_study_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Saved: {out_path}")


def make_hyperparams_table():
    """Generate hyperparameter table."""
    lines = []
    lines.append(r'\begin{tabular}{lr}')
    lines.append(r'\toprule')
    lines.append(r'Parameter & Value \\')
    lines.append(r'\midrule')
    lines.append(f"Grid size ($N$) & ${N}$ \\\\")
    lines.append(f"Discount ($\\gamma$) & ${GAMMA}$ \\\\")
    lines.append(f"Step penalty & ${STEP_PENALTY}$ \\\\")
    lines.append(f"Terminal reward & ${TERMINAL_REWARD}$ \\\\")
    lines.append(f"Training episodes & ${NUM_EPISODES}$ \\\\")
    lines.append(f"Episode horizon & ${EPISODE_HORIZON}$ \\\\")
    lines.append(f"Learning rate ($\\alpha$) & ${ALPHA}$ \\\\")
    lines.append(f"$\\alpha$ decay & ${ALPHA_DECAY}$ \\\\")
    lines.append(f"$\\epsilon$ (start) & ${EPSILON_START}$ \\\\")
    lines.append(f"$\\epsilon$ (end) & ${EPSILON_END}$ \\\\")
    lines.append(f"$\\epsilon$ decay & ${EPSILON_DECAY}$ \\\\")
    lines.append(f"Trace decay ($\\lambda$) & ${LAMBDA}$ \\\\")
    lines.append(f"Replications ($M$) & ${M}$ \\\\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'gridworld_hyperparams.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_data():
    """Run all computation (planning + learning). Returns cached results if available."""
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    t_start = time.time()

    print_header()

    # Create environment
    env = GridworldEnv(N, GAMMA, STEP_PENALTY, TERMINAL_REWARD)

    # Run planning methods (get optimal solution)
    V_optimal, policy_optimal, optimal_return, planning_results = run_planning_methods(env)

    # Run all learning methods
    learning_results = run_all_learning_methods(env, V_optimal, policy_optimal, optimal_return)

    # Combine results
    all_results = {**planning_results, **learning_results}

    # Print summary table
    print_results_table(all_results, optimal_return)

    total_time = time.time() - t_start
    print(f"\nComputation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Convert AlgorithmResult dataclass instances to dicts for pickling
    serialized_results = {}
    for name, r in all_results.items():
        serialized_results[name] = {
            'name': r.name,
            'return_mean': r.return_mean,
            'return_std': r.return_std,
            'steps_mean': r.steps_mean,
            'steps_std': r.steps_std,
            'agreement_mean': r.agreement_mean,
            'agreement_std': r.agreement_std,
            'value_error_mean': r.value_error_mean,
            'value_error_std': r.value_error_std,
            'time_mean': r.time_mean,
            'time_std': r.time_std,
            'coverage_mean': r.coverage_mean,
            'coverage_std': r.coverage_std,
            'episodes_to_95': r.episodes_to_95,
            'eval_returns_all': r.eval_returns_all,
            'value_errors_all': r.value_errors_all,
            'agreements_all': r.agreements_all,
            'regrets_all': r.regrets_all,
        }

    data = {
        'all_results': serialized_results,
        'optimal_return': optimal_return,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def _reconstruct_results(data):
    """Reconstruct AlgorithmResult objects from cached dict."""
    all_results = {}
    for name, d in data['all_results'].items():
        all_results[name] = AlgorithmResult(**d)
    return all_results, data['optimal_return']


def generate_outputs(data):
    """Generate all figures, tables, and printed output from cached data."""
    all_results, optimal_return = _reconstruct_results(data)

    # Print summary table
    print_results_table(all_results, optimal_return)

    # Generate figures
    print()
    print("-" * 70)
    print("GENERATING FIGURES")
    print("-" * 70)
    make_learning_curves_figure(all_results, optimal_return)
    make_convergence_figure(all_results)
    make_regret_figure(all_results)
    make_sample_efficiency_figure(all_results, optimal_return)
    make_compute_time_figure(all_results)

    # Generate LaTeX tables
    print()
    print("-" * 70)
    print("GENERATING LATEX TABLES")
    print("-" * 70)
    make_results_table(all_results, optimal_return)
    make_hyperparams_table()

    # Final summary
    print()
    print("=" * 70)
    print(f"STUDY COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  {OUTPUT_DIR / 'gridworld_learning_curves.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_convergence.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_regret.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_sample_efficiency.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_compute_time.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_study_results.tex'}")
    print(f"  {OUTPUT_DIR / 'gridworld_hyperparams.tex'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gridworld Algorithm Comparison Study')
    add_cache_args(parser)
    args = parser.parse_args()

    # Capture stdout to file
    stdout_path = OUTPUT_DIR / 'gridworld_study_stdout.txt'
    tee = TeeOutput(stdout_path)
    sys.stdout = tee

    try:
        if args.plots_only:
            data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
            assert data is not None, "No cache found. Run without --plots-only first."
        else:
            data = compute_data()

        if not args.data_only:
            generate_outputs(data)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"\nStdout captured to: {stdout_path}")
