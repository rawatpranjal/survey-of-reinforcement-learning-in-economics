# Comprehensive Gridworld Algorithm Comparison Study
# Chapter 3 -- Economic Applications Benchmarks
# Compares classical RL algorithms on 10x10 gridworld with M=30 Monte Carlo seeds.

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gridworld_algorithms import (
    GridworldEnv, ALGORITHMS, AlgorithmMetrics,
    run_value_iteration, evaluate_policy, extract_policy_from_Q
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environment
N = 10                      # Grid size (100 states)
GAMMA = 0.95                # Discount factor
STEP_PENALTY = -0.1         # Per-step cost
TERMINAL_REWARD = 10.0      # Goal reward

# Training
NUM_EPISODES = 5000         # Episodes for learning algorithms
EPISODE_HORIZON = 100       # Max steps per episode
EVAL_FREQ = 100             # Checkpoint frequency
EVAL_EPISODES = 100         # Episodes per evaluation

# Hyperparameters
ALPHA = 0.1                 # TD/MC learning rate
ALPHA_DECAY = 0.999         # Per-episode decay
EPSILON_START = 1.0         # Initial exploration
EPSILON_END = 0.01          # Final exploration
EPSILON_DECAY = 0.995       # Per-episode decay
LAMBDA = 0.9                # Trace decay for eligibility methods

# Monte Carlo replications
M = 30                      # Seeds per algorithm
SEEDS = list(range(42, 42 + M))

# Output
OUTPUT_DIR = Path(__file__).resolve().parent

# Algorithms to compare (subset for tractability)
ALGORITHMS_TO_RUN = [
    'VI',
    'PI',
    'Q-Learning',
    'SARSA',
    'Expected SARSA',
    'MC Control',
    'SARSA(λ)',
    'Q(λ)',
    'REINFORCE',
    'DQN',
]

# Figure styling
COLORS = {
    'VI': '#1f77b4',
    'PI': '#1f77b4',
    'Q-Learning': '#ff7f0e',
    'SARSA': '#2ca02c',
    'Expected SARSA': '#d62728',
    'MC Control': '#9467bd',
    'SARSA(λ)': '#8c564b',
    'Q(λ)': '#e377c2',
    'REINFORCE': '#7f7f7f',
    'DQN': '#bcbd22',
}

LINESTYLES = {
    'VI': '--',
    'PI': '--',
    'Q-Learning': '-',
    'SARSA': '-',
    'Expected SARSA': '-',
    'MC Control': '-',
    'SARSA(λ)': '-',
    'Q(λ)': '-',
    'REINFORCE': '-',
    'DQN': '-',
}


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmResult:
    """Aggregated results for one algorithm across M seeds."""
    name: str
    # Means
    return_mean: float
    return_std: float
    steps_mean: float
    steps_std: float
    agreement_mean: float
    agreement_std: float
    value_error_mean: float
    value_error_std: float
    time_mean: float
    time_std: float
    # Learning curves (per seed, per checkpoint)
    checkpoint_episodes: List[int]
    checkpoint_returns: List[List[float]]  # [seed][checkpoint]
    checkpoint_value_errors: List[List[float]]
    checkpoint_agreements: List[List[float]]
    # Raw data
    final_returns: List[float]
    final_steps: List[float]
    final_agreements: List[float]
    final_value_errors: List[float]
    wall_times: List[float]


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    """Run full comparison study."""
    print("=" * 70)
    print("Gridworld Algorithm Comparison Study")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Grid size: {N}x{N} ({N*N} states)")
    print(f"  Discount factor: γ = {GAMMA}")
    print(f"  Step penalty: {STEP_PENALTY}")
    print(f"  Terminal reward: {TERMINAL_REWARD}")
    print(f"  Training episodes: {NUM_EPISODES}")
    print(f"  Horizon: {EPISODE_HORIZON}")
    print(f"  Monte Carlo seeds: M = {M}")
    print(f"  Seeds: {SEEDS[0]} to {SEEDS[-1]}")
    print()
    print("Hyperparameters:")
    print(f"  α (learning rate): {ALPHA}")
    print(f"  α decay: {ALPHA_DECAY}")
    print(f"  ε start: {EPSILON_START}")
    print(f"  ε end: {EPSILON_END}")
    print(f"  ε decay: {EPSILON_DECAY}")
    print(f"  λ (trace decay): {LAMBDA}")
    print()

    # Create environment and get optimal solution
    env = GridworldEnv(N, step_penalty=STEP_PENALTY, terminal_reward=TERMINAL_REWARD)

    print("-" * 70)
    print("Phase 1: Computing Optimal Solution (Value Iteration)")
    print("-" * 70)

    policy_opt, Q_opt, metrics_opt = run_value_iteration(env, gamma=GAMMA)

    # Convert to arrays for comparison
    V_optimal = np.array([max(Q_opt[env.index_to_state(i)].values())
                          for i in range(env.num_states)])
    optimal_policy = np.array([policy_opt[env.index_to_state(i)]
                               for i in range(env.num_states)])

    opt_return, opt_steps = evaluate_policy(env, policy_opt, n_episodes=100, horizon=EPISODE_HORIZON)

    print(f"  VI converged in {metrics_opt.iterations} iterations")
    print(f"  Optimal return: {opt_return:.3f}")
    print(f"  Optimal steps to goal: {opt_steps:.1f}")
    print(f"  Wall time: {metrics_opt.wall_time:.3f}s")
    print()

    # Store results
    all_results: Dict[str, AlgorithmResult] = {}

    # Run each algorithm
    print("-" * 70)
    print("Phase 2: Running Algorithms")
    print("-" * 70)

    for algo_name in ALGORITHMS_TO_RUN:
        print()
        print(f"Algorithm: {algo_name}")
        print("-" * 40)

        algo_fn = ALGORITHMS[algo_name]

        # Special handling for DP methods (no stochasticity)
        if algo_name in ['VI', 'PI']:
            env_run = GridworldEnv(N, step_penalty=STEP_PENALTY, terminal_reward=TERMINAL_REWARD)
            policy, Q, metrics = algo_fn(env_run, gamma=GAMMA)

            result = AlgorithmResult(
                name=algo_name,
                return_mean=metrics.final_return,
                return_std=0.0,
                steps_mean=metrics.final_steps,
                steps_std=0.0,
                agreement_mean=1.0,
                agreement_std=0.0,
                value_error_mean=0.0,
                value_error_std=0.0,
                time_mean=metrics.wall_time,
                time_std=0.0,
                checkpoint_episodes=[],
                checkpoint_returns=[],
                checkpoint_value_errors=[],
                checkpoint_agreements=[],
                final_returns=[metrics.final_return],
                final_steps=[metrics.final_steps],
                final_agreements=[1.0],
                final_value_errors=[0.0],
                wall_times=[metrics.wall_time]
            )

            print(f"  Return: {metrics.final_return:.3f}")
            print(f"  Steps: {metrics.final_steps:.1f}")
            print(f"  Iterations: {metrics.iterations}")
            print(f"  Time: {metrics.wall_time:.3f}s")

        else:
            # Learning algorithms: run M seeds
            final_returns = []
            final_steps = []
            final_agreements = []
            final_value_errors = []
            wall_times = []
            all_checkpoint_returns = []
            all_checkpoint_value_errors = []
            all_checkpoint_agreements = []
            checkpoint_episodes = None

            for i, seed in enumerate(SEEDS):
                env_run = GridworldEnv(N, step_penalty=STEP_PENALTY, terminal_reward=TERMINAL_REWARD)

                # Build kwargs
                kwargs = {
                    'gamma': GAMMA,
                    'num_episodes': NUM_EPISODES,
                    'horizon': EPISODE_HORIZON,
                    'alpha': ALPHA,
                    'alpha_decay': ALPHA_DECAY,
                    'epsilon_start': EPSILON_START,
                    'epsilon_end': EPSILON_END,
                    'epsilon_decay': EPSILON_DECAY,
                    'lmbda': LAMBDA,
                    'eval_freq': EVAL_FREQ,
                    'eval_episodes': EVAL_EPISODES,
                    'V_optimal': V_optimal,
                    'optimal_policy': optimal_policy,
                    'seed': seed,
                }

                policy, Q, metrics = algo_fn(env_run, **kwargs)

                final_returns.append(metrics.final_return)
                final_steps.append(metrics.final_steps)
                final_agreements.append(metrics.final_policy_agreement)
                final_value_errors.append(metrics.final_value_error)
                wall_times.append(metrics.wall_time)

                if metrics.checkpoint_returns:
                    all_checkpoint_returns.append(metrics.checkpoint_returns)
                if metrics.checkpoint_value_errors:
                    all_checkpoint_value_errors.append(metrics.checkpoint_value_errors)
                if metrics.checkpoint_policy_agreements:
                    all_checkpoint_agreements.append(metrics.checkpoint_policy_agreements)
                if checkpoint_episodes is None and metrics.checkpoint_episodes:
                    checkpoint_episodes = metrics.checkpoint_episodes

                # Progress output
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Seed {seed}: return={metrics.final_return:.3f}, "
                          f"agreement={metrics.final_policy_agreement:.1%}, "
                          f"time={metrics.wall_time:.2f}s")

            result = AlgorithmResult(
                name=algo_name,
                return_mean=np.mean(final_returns),
                return_std=np.std(final_returns),
                steps_mean=np.mean(final_steps),
                steps_std=np.std(final_steps),
                agreement_mean=np.mean(final_agreements),
                agreement_std=np.std(final_agreements),
                value_error_mean=np.mean(final_value_errors),
                value_error_std=np.std(final_value_errors),
                time_mean=np.mean(wall_times),
                time_std=np.std(wall_times),
                checkpoint_episodes=checkpoint_episodes or [],
                checkpoint_returns=all_checkpoint_returns,
                checkpoint_value_errors=all_checkpoint_value_errors,
                checkpoint_agreements=all_checkpoint_agreements,
                final_returns=final_returns,
                final_steps=final_steps,
                final_agreements=final_agreements,
                final_value_errors=final_value_errors,
                wall_times=wall_times
            )

            print(f"  Summary (M={M}):")
            print(f"    Return: {result.return_mean:.3f} ± {result.return_std:.3f}")
            print(f"    Steps: {result.steps_mean:.1f} ± {result.steps_std:.1f}")
            print(f"    Agreement: {result.agreement_mean:.1%} ± {result.agreement_std:.1%}")
            print(f"    Value Error: {result.value_error_mean:.3f} ± {result.value_error_std:.3f}")
            print(f"    Time: {result.time_mean:.2f}s ± {result.time_std:.2f}s")

        all_results[algo_name] = result

    return all_results, opt_return, opt_steps, V_optimal, optimal_policy


# ---------------------------------------------------------------------------
# Output: Tables
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, AlgorithmResult], opt_return: float, opt_steps: float):
    """Print formatted results table."""
    print()
    print("-" * 70)
    print("Phase 3: Results Summary")
    print("-" * 70)
    print()

    # Header
    print(f"{'Algorithm':<16} {'Return':>12} {'Steps':>10} {'Agreement':>12} "
          f"{'Val Err':>10} {'Time (s)':>10}")
    print("-" * 70)

    # Optimal reference
    print(f"{'Optimal':<16} {opt_return:>12.3f} {opt_steps:>10.1f} "
          f"{'100.0%':>12} {'0.000':>10} {'---':>10}")
    print("-" * 70)

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if r.return_std > 0:
            ret_str = f"{r.return_mean:.2f}±{r.return_std:.2f}"
        else:
            ret_str = f"{r.return_mean:.3f}"

        if r.steps_std > 0:
            steps_str = f"{r.steps_mean:.1f}±{r.steps_std:.1f}"
        else:
            steps_str = f"{r.steps_mean:.1f}"

        if r.agreement_std > 0:
            agr_str = f"{100*r.agreement_mean:.1f}±{100*r.agreement_std:.1f}%"
        else:
            agr_str = f"{100*r.agreement_mean:.1f}%"

        if r.value_error_std > 0:
            ve_str = f"{r.value_error_mean:.3f}±{r.value_error_std:.3f}"
        else:
            ve_str = f"{r.value_error_mean:.3f}"

        time_str = f"{r.time_mean:.2f}"

        print(f"{name:<16} {ret_str:>12} {steps_str:>10} {agr_str:>12} "
              f"{ve_str:>10} {time_str:>10}")

    print()


def make_latex_results_table(results: Dict[str, AlgorithmResult], opt_return: float, opt_steps: float):
    """Generate LaTeX results table."""
    lines = []
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Algorithm & Return & Steps & Agreement (\%) & Value Error & Time (s) \\')
    lines.append(r'\midrule')

    # Optimal
    lines.append(f"Optimal (VI) & ${opt_return:.2f}$ & ${opt_steps:.1f}$ & $100.0$ & $0.00$ & --- \\\\")
    lines.append(r'\midrule')

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if name in ['VI', 'PI']:
            ret_str = f"${r.return_mean:.2f}$"
            steps_str = f"${r.steps_mean:.1f}$"
            agr_str = "$100.0$"
            ve_str = "$0.00$"
            time_str = f"${r.time_mean:.3f}$"
        else:
            ret_str = f"${r.return_mean:.2f} \\pm {r.return_std:.2f}$"
            steps_str = f"${r.steps_mean:.1f} \\pm {r.steps_std:.1f}$"
            agr_str = f"${100*r.agreement_mean:.1f} \\pm {100*r.agreement_std:.1f}$"
            ve_str = f"${r.value_error_mean:.2f} \\pm {r.value_error_std:.2f}$"
            time_str = f"${r.time_mean:.1f}$"

        lines.append(f"{name} & {ret_str} & {steps_str} & {agr_str} & {ve_str} & {time_str} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'gridworld_study_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Results table saved to {out_path}")
    return tex


def make_latex_hyperparams_table():
    """Generate LaTeX hyperparameters table."""
    lines = []
    lines.append(r'\begin{tabular}{lr}')
    lines.append(r'\toprule')
    lines.append(r'Parameter & Value \\')
    lines.append(r'\midrule')
    lines.append(f"Grid size $N$ & {N} \\\\")
    lines.append(f"State space $|\\mathcal{{S}}|$ & {N*N} \\\\")
    lines.append(f"Action space $|\\mathcal{{A}}|$ & 5 \\\\")
    lines.append(f"Discount $\\gamma$ & {GAMMA} \\\\")
    lines.append(f"Step penalty & {STEP_PENALTY} \\\\")
    lines.append(f"Terminal reward & {TERMINAL_REWARD} \\\\")
    lines.append(r'\midrule')
    lines.append(f"Training episodes & {NUM_EPISODES} \\\\")
    lines.append(f"Horizon & {EPISODE_HORIZON} \\\\")
    lines.append(f"Monte Carlo seeds $M$ & {M} \\\\")
    lines.append(r'\midrule')
    lines.append(f"Learning rate $\\alpha$ & {ALPHA} \\\\")
    lines.append(f"$\\alpha$ decay & {ALPHA_DECAY} \\\\")
    lines.append(f"$\\varepsilon$ start & {EPSILON_START} \\\\")
    lines.append(f"$\\varepsilon$ end & {EPSILON_END} \\\\")
    lines.append(f"$\\varepsilon$ decay & {EPSILON_DECAY} \\\\")
    lines.append(f"Trace decay $\\lambda$ & {LAMBDA} \\\\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    out_path = OUTPUT_DIR / 'gridworld_hyperparams.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"Hyperparams table saved to {out_path}")
    return tex


# ---------------------------------------------------------------------------
# Output: Figures
# ---------------------------------------------------------------------------

def make_learning_curves_figure(results: Dict[str, AlgorithmResult], opt_return: float):
    """Generate learning curves plot (return vs episodes)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if not r.checkpoint_returns:
            continue

        episodes = r.checkpoint_episodes
        # Pad curves to same length
        min_len = min(len(c) for c in r.checkpoint_returns)
        curves = np.array([c[:min_len] for c in r.checkpoint_returns])
        episodes = episodes[:min_len]

        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)

        ax.plot(episodes, mean, color=COLORS[name], linestyle=LINESTYLES[name],
                linewidth=1.5, label=name)
        ax.fill_between(episodes, mean - std, mean + std,
                       color=COLORS[name], alpha=0.15)

    # Optimal reference
    ax.axhline(opt_return, color='black', linestyle='--', linewidth=1.0,
               label=f'Optimal ({opt_return:.2f})')

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Return', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N}: Learning Curves (M={M} seeds)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'gridworld_learning_curves.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Learning curves figure saved to {out_path}")


def make_convergence_figure(results: Dict[str, AlgorithmResult]):
    """Generate value error convergence plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if not r.checkpoint_value_errors:
            continue

        episodes = r.checkpoint_episodes
        min_len = min(len(c) for c in r.checkpoint_value_errors)
        curves = np.array([c[:min_len] for c in r.checkpoint_value_errors])
        episodes = episodes[:min_len]

        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)

        ax.plot(episodes, mean, color=COLORS[name], linestyle=LINESTYLES[name],
                linewidth=1.5, label=name)
        ax.fill_between(episodes, mean - std, mean + std,
                       color=COLORS[name], alpha=0.15)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Value Error (RMSE)', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N}: Value Convergence (M={M} seeds)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    out_path = OUTPUT_DIR / 'gridworld_convergence.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Convergence figure saved to {out_path}")


def make_regret_figure(results: Dict[str, AlgorithmResult], opt_return: float):
    """Generate cumulative regret plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if not r.checkpoint_returns:
            continue

        episodes = r.checkpoint_episodes
        min_len = min(len(c) for c in r.checkpoint_returns)
        curves = np.array([c[:min_len] for c in r.checkpoint_returns])
        episodes = episodes[:min_len]

        # Regret = optimal - achieved
        regret_curves = opt_return - curves

        # Cumulative regret (approximate: multiply by eval_freq)
        cum_regret = np.cumsum(regret_curves * EVAL_FREQ, axis=1)

        mean = np.mean(cum_regret, axis=0)
        std = np.std(cum_regret, axis=0)

        ax.plot(episodes, mean, color=COLORS[name], linestyle=LINESTYLES[name],
                linewidth=1.5, label=name)
        ax.fill_between(episodes, mean - std, mean + std,
                       color=COLORS[name], alpha=0.15)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Cumulative Regret', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N}: Cumulative Regret (M={M} seeds)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / 'gridworld_regret.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Regret figure saved to {out_path}")


def make_sample_efficiency_figure(results: Dict[str, AlgorithmResult], opt_return: float):
    """Generate sample efficiency bar chart (episodes to 95% optimal)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    means = []
    stds = []

    threshold = 0.95 * opt_return

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        if not r.checkpoint_returns:
            continue

        # Find episode to reach threshold for each seed
        convergence_episodes = []
        for curve in r.checkpoint_returns:
            found = False
            for i, ret in enumerate(curve):
                if ret >= threshold:
                    convergence_episodes.append(r.checkpoint_episodes[i])
                    found = True
                    break
            if not found:
                convergence_episodes.append(NUM_EPISODES)

        names.append(name)
        means.append(np.mean(convergence_episodes))
        stds.append(np.std(convergence_episodes))

    x = np.arange(len(names))
    colors = [COLORS[n] for n in names]

    ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Episodes to 95% Optimal', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N}: Sample Efficiency (M={M} seeds)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')

    out_path = OUTPUT_DIR / 'gridworld_sample_efficiency.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Sample efficiency figure saved to {out_path}")


def make_compute_time_figure(results: Dict[str, AlgorithmResult]):
    """Generate computation time bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    means = []
    stds = []

    for name in ALGORITHMS_TO_RUN:
        r = results[name]
        names.append(name)
        means.append(r.time_mean)
        stds.append(r.time_std)

    x = np.arange(len(names))
    colors = [COLORS[n] for n in names]

    ax.bar(x, means, yerr=stds, color=colors, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Wall-Clock Time (s)', fontsize=11)
    ax.set_title(f'Gridworld {N}x{N}: Computation Time (M={M} seeds)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    out_path = OUTPUT_DIR / 'gridworld_compute_time.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Compute time figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    # Run experiment
    results, opt_return, opt_steps, V_optimal, optimal_policy = run_experiment()

    # Print results
    print_results_table(results, opt_return, opt_steps)

    # Generate outputs
    print("-" * 70)
    print("Phase 4: Generating Output Files")
    print("-" * 70)
    print()

    make_latex_results_table(results, opt_return, opt_steps)
    make_latex_hyperparams_table()
    make_learning_curves_figure(results, opt_return)
    make_convergence_figure(results)
    make_regret_figure(results, opt_return)
    make_sample_efficiency_figure(results, opt_return)
    make_compute_time_figure(results)

    total_time = time.time() - start_time

    print()
    print("-" * 70)
    print("Study Complete")
    print("-" * 70)
    print(f"Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print("Output files:")
    print(f"  {OUTPUT_DIR / 'gridworld_study_results.tex'}")
    print(f"  {OUTPUT_DIR / 'gridworld_hyperparams.tex'}")
    print(f"  {OUTPUT_DIR / 'gridworld_learning_curves.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_convergence.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_regret.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_sample_efficiency.png'}")
    print(f"  {OUTPUT_DIR / 'gridworld_compute_time.png'}")


if __name__ == '__main__':
    main()
