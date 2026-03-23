# TD(λ) Credit Assignment in a Corridor
# Chapter 3 — Theory of Reinforcement Learning
# Demonstrates how eligibility traces accelerate credit assignment in sparse-reward settings.

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set
from sims.plot_style import apply_style, COLORS, FIG_SINGLE

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'td_lambda_corridor'
CONFIG = {
    'n_states': 20,
    'gamma': 0.99,
    'alpha': 0.05,
    'n_episodes': 200,
    'n_seeds': 20,
    'lambdas': [0.0, 0.4, 0.8, 1.0],
    'rmsve_threshold': 0.05,
    'version': 3,
}

OUTPUT_DIR = os.path.dirname(__file__)

# Lambda display colors (use generic palette for 4 curves)
LAMBDA_COLORS = {
    0.0: COLORS['blue'],
    0.4: COLORS['orange'],
    0.8: COLORS['green'],
    1.0: COLORS['red'],
}

# ---------------------------------------------------------------------------
# Environment: 20-state corridor
# ---------------------------------------------------------------------------

def true_values(n_states, gamma):
    """V*(s) = gamma^(n_states - 1 - s) for s = 0, ..., n_states - 1."""
    return np.array([gamma ** (n_states - 1 - s) for s in range(n_states)])


def run_td_lambda(n_states, gamma, alpha, lam, n_episodes, seed):
    """Run TD(lambda) on the corridor. Returns RMSVE per episode."""
    rng = np.random.RandomState(seed)
    V_true = true_values(n_states, gamma)
    V = rng.uniform(0, 0.5, n_states)
    V[n_states - 1] = 0.0  # terminal state value fixed
    rmsve_history = np.zeros(n_episodes)

    for ep in range(n_episodes):
        e = np.zeros(n_states)  # eligibility trace
        s = 0  # start state

        while s < n_states - 1:
            s_next = s + 1
            r = 1.0 if s_next == n_states - 1 else 0.0
            V_next = 0.0 if s_next == n_states - 1 else V[s_next]

            delta = r + gamma * V_next - V[s]

            # Accumulating trace
            e[s] += 1.0

            # Update all states
            V[:n_states - 1] += alpha * delta * e[:n_states - 1]

            # Decay trace
            e *= gamma * lam

            s = s_next

        # RMSVE after this episode (exclude terminal state)
        rmsve = np.sqrt(np.mean((V[:n_states - 1] - V_true[:n_states - 1]) ** 2))
        rmsve_history[ep] = rmsve

    return rmsve_history


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _run_td_lambda_experiment():
    """Run TD(lambda) for all lambda values and seeds."""
    n_states = CONFIG['n_states']
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    n_episodes = CONFIG['n_episodes']
    n_seeds = CONFIG['n_seeds']
    lambdas = CONFIG['lambdas']
    threshold = CONFIG['rmsve_threshold']

    print(f"Running TD(lambda) corridor experiment")
    print(f"  States: {n_states}, gamma: {gamma}, alpha: {alpha}")
    print(f"  Episodes: {n_episodes}, Seeds: {n_seeds}")
    print(f"  Lambda values: {lambdas}")
    print()

    results = {}
    for lam in lambdas:
        all_rmsve = np.zeros((n_seeds, n_episodes))
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            all_rmsve[seed_idx] = run_td_lambda(n_states, gamma, alpha, lam, n_episodes, seed)

        mean_rmsve = np.mean(all_rmsve, axis=0)
        se_rmsve = np.std(all_rmsve, axis=0) / np.sqrt(n_seeds)
        final_rmsve_per_seed = all_rmsve[:, -1]
        final_mean = np.mean(final_rmsve_per_seed)
        final_se = np.std(final_rmsve_per_seed) / np.sqrt(n_seeds)

        # Episodes to threshold (per seed)
        eps_to_thresh = []
        for seed_idx in range(n_seeds):
            below = np.where(all_rmsve[seed_idx] < threshold)[0]
            if len(below) > 0:
                eps_to_thresh.append(below[0] + 1)  # 1-indexed
            else:
                eps_to_thresh.append(np.nan)
        eps_to_thresh = np.array(eps_to_thresh)
        n_reached = int(np.sum(~np.isnan(eps_to_thresh)))

        results[str(lam)] = {
            'mean_rmsve': mean_rmsve,
            'se_rmsve': se_rmsve,
            'final_mean': final_mean,
            'final_se': final_se,
            'eps_to_thresh_mean': float(np.nanmean(eps_to_thresh)) if n_reached > 0 else float('nan'),
            'eps_to_thresh_se': float(np.nanstd(eps_to_thresh) / np.sqrt(n_reached)) if n_reached > 1 else 0.0,
            'eps_to_thresh_count': n_reached,
        }

        print(f"  lambda={lam:.1f}: final RMSVE = {final_mean:.4f} +/- {final_se:.4f}")

    return {'results': results, 'config': CONFIG}


def compute_data(force=None):
    force = force or set()
    return compute_or_load(CACHE_DIR, SCRIPT_NAME, 'td_lambda', CONFIG,
                            _run_td_lambda_experiment, force=('td_lambda' in force))


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_outputs(data):
    """Generate figure and LaTeX table."""
    results = data['results']
    config = data['config']
    lambdas = config['lambdas']
    n_episodes = config['n_episodes']
    threshold = config['rmsve_threshold']

    # --- Figure: RMSVE vs episodes ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for lam in lambdas:
        r = results[str(lam)]
        mean = r['mean_rmsve']
        se = r['se_rmsve']
        episodes = np.arange(1, n_episodes + 1)
        color = LAMBDA_COLORS[lam]
        label = f'$\\lambda = {lam}$'

        ax.plot(episodes, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(episodes, mean - se, mean + se, color=color, alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('RMSVE')
    ax.set_title('TD($\\lambda$) on 20-State Corridor')
    ax.legend(loc='upper right')
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(bottom=0)

    fig_path = os.path.join(OUTPUT_DIR, 'td_lambda_corridor.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # --- LaTeX table ---
    tex_path = os.path.join(OUTPUT_DIR, 'td_lambda_corridor.tex')
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')
        f.write('\\caption{TD($\\lambda$) on 20-state corridor. ')
        f.write(f'Mean $\\pm$ SE over {config["n_seeds"]} seeds, ')
        f.write(f'{config["n_episodes"]} episodes, ')
        f.write(f'$\\gamma = {config["gamma"]}$, ')
        f.write(f'$\\alpha = {config["alpha"]}$.}}\n')
        f.write('\\label{tab:td_lambda_corridor}\n')
        f.write('\\begin{tabular}{ccc}\n')
        f.write('\\hline\n')
        f.write(f'$\\lambda$ & Final RMSVE & Episodes to RMSVE $< {threshold}$ \\\\\n')
        f.write('\\hline\n')

        for lam in lambdas:
            r = results[str(lam)]
            rmsve_str = f'{r["final_mean"]:.4f} $\\pm$ {r["final_se"]:.4f}'
            if r['eps_to_thresh_count'] == config['n_seeds']:
                thresh_str = f'{r["eps_to_thresh_mean"]:.0f} $\\pm$ {r["eps_to_thresh_se"]:.0f}'
            elif r['eps_to_thresh_count'] > 0:
                thresh_str = f'{r["eps_to_thresh_mean"]:.0f} $\\pm$ {r["eps_to_thresh_se"]:.0f} ({r["eps_to_thresh_count"]}/{config["n_seeds"]})'
            else:
                thresh_str = '$> ' + str(config['n_episodes']) + '$'
            f.write(f'{lam} & {rmsve_str} & {thresh_str} \\\\\n')

        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"  Table saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TD(lambda) Credit Assignment in a Corridor')
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)

    print("=" * 70)
    print("TD(LAMBDA) CREDIT ASSIGNMENT IN A CORRIDOR")
    print("=" * 70)
    print()
    print("Environment: 20-state corridor (chain MDP)")
    print(f"  States: {CONFIG['n_states']}, Terminal: state {CONFIG['n_states'] - 1}")
    print(f"  Reward: +1 at terminal, 0 elsewhere")
    print(f"  True V*(s) = gamma^(19 - s)")
    print()
    print(f"Parameters: gamma={CONFIG['gamma']}, alpha={CONFIG['alpha']}")
    print(f"  Episodes: {CONFIG['n_episodes']}, Seeds: {CONFIG['n_seeds']}")
    print(f"  Lambda values: {CONFIG['lambdas']}")
    print()

    if force:
        print(f"Force recompute: {sorted(force)}")

    if args.plots_only:
        data = compute_data()  # cache hit
        generate_outputs(data)
    elif args.data_only:
        compute_data(force=force)
    else:
        data = compute_data(force=force)
        generate_outputs(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
