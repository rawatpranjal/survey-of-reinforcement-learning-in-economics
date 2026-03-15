"""
Cournot and Bertrand Duopoly: Multi-Agent Q-Learning Convergence
Chapter 6 — RL in Games
Compares IQL, Nash-Q, and WoLF-PHC on two canonical IO games with unique Nash equilibria.
"""

import argparse
import sys, os
import numpy as np
from itertools import product
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sims'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from plot_style import apply_style, COLORS, DOMAIN_COLORS, BENCH_STYLE, FIG_DOUBLE
from sims.sim_cache import load_results, save_results, add_cache_args
import matplotlib.pyplot as plt

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'cournot_bertrand_marl'
CONFIG = {
    'n_iter': 50000,
    'n_seeds': 20,
    'cournot_a': 10, 'cournot_c': 1, 'cournot_Q_max': 9,
    'bertrand_a': 10, 'bertrand_b': 2, 'bertrand_e': 1, 'bertrand_c': 1, 'bertrand_P_max': 9,
    'tail': 5000,
    'version': 1,
}

# ============================================================================
# Game definitions
# ============================================================================

class CournotDuopoly:
    """Quantity competition: pi_i = q_i * (a - q_i - q_j - c)."""
    def __init__(self, a=10, c=1, Q_max=9):
        self.a, self.c, self.Q_max = a, c, Q_max
        self.n_actions = Q_max + 1  # {0, 1, ..., Q_max}
        self.name = "Cournot"
        # Precompute payoff matrices: payoff[a0, a1, i] = player i's profit
        # when player 0 plays a0 and player 1 plays a1
        self.payoff = np.zeros((self.n_actions, self.n_actions, 2))
        for q0 in range(self.n_actions):
            for q1 in range(self.n_actions):
                price = max(a - q0 - q1, 0)
                self.payoff[q0, q1, 0] = q0 * (price - c)
                self.payoff[q0, q1, 1] = q1 * (price - c)
        # Nash equilibrium: q* = (a-c)/3
        self.nash_action = (a - c) / 3.0
        self.nash_profit = (a - c)**2 / 9.0

class BertrandDuopoly:
    """Price competition with differentiated products.
    d_i = a - b*p_i + e*p_j; pi_i = (p_i - c) * d_i.
    """
    def __init__(self, a=10, b=2, e=1, c=1, P_max=9):
        self.a, self.b, self.e, self.c, self.P_max = a, b, e, c, P_max
        self.n_actions = P_max + 1
        self.name = "Bertrand"
        # payoff[a0, a1, i] = player i's profit when player 0 plays a0, player 1 plays a1
        self.payoff = np.zeros((self.n_actions, self.n_actions, 2))
        for p0 in range(self.n_actions):
            for p1 in range(self.n_actions):
                d0 = max(a - b * p0 + e * p1, 0)
                d1 = max(a - b * p1 + e * p0, 0)
                self.payoff[p0, p1, 0] = (p0 - c) * d0
                self.payoff[p0, p1, 1] = (p1 - c) * d1
        # Nash: p* = (a + bc + ec) / (2b - e)
        self.nash_action = (a + b * c + e * c) / (2 * b - e)
        # Nash profit
        d_nash = a - b * self.nash_action + e * self.nash_action
        self.nash_profit = (self.nash_action - c) * d_nash

# ============================================================================
# Algorithm 1: Independent Q-Learning (IQL)
# ============================================================================

def run_iql(game, n_iter, seed, tau_init=5.0, tau_decay=0.9995):
    """Each agent maintains Q_i(a_i) over own actions only. Boltzmann exploration."""
    rng = np.random.RandomState(seed)
    n = game.n_actions
    Q = [np.zeros(n), np.zeros(n)]
    alpha0 = 0.5
    history = np.zeros((n_iter, 2))  # expected action per iteration

    for t in range(n_iter):
        tau = max(tau_init * (tau_decay ** t), 0.01)
        alpha = alpha0 / (1 + t * 0.0001)

        # Boltzmann action selection
        actions = []
        policies = []
        for i in range(2):
            logits = Q[i] / tau
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            policies.append(probs)
            actions.append(rng.choice(n, p=probs))

        # Observe rewards
        for i in range(2):
            r = game.payoff[actions[0], actions[1], i]
            Q[i][actions[i]] += alpha * (r - Q[i][actions[i]])

        # Record expected action under current policy
        for i in range(2):
            history[t, i] = np.dot(policies[i], np.arange(n))

    return history

# ============================================================================
# Algorithm 2: Nash-Q Learning
# ============================================================================

def solve_nash_2player(payoff_0, payoff_1):
    """Solve 2-player general-sum game via support enumeration.
    Returns (pi_0, pi_1) mixed strategies. Falls back to pure NE."""
    n = payoff_0.shape[0]
    m = payoff_0.shape[1]

    best_ne = None
    best_value = -np.inf  # pick highest joint payoff NE

    # Check all pure strategy NE first
    for i in range(n):
        for j in range(m):
            # Check if (i,j) is a pure NE
            if (payoff_0[i, j] >= payoff_0[:, j].max() - 1e-10 and
                payoff_1[i, j] >= payoff_1[i, :].max() - 1e-10):
                pi0 = np.zeros(n); pi0[i] = 1.0
                pi1 = np.zeros(m); pi1[j] = 1.0
                val = payoff_0[i, j] + payoff_1[i, j]
                if val > best_value:
                    best_value = val
                    best_ne = (pi0, pi1)

    # Try 2x2 mixed strategy NE for all pairs of supports
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    # Player 1 mixes over {i1, i2}, player 2 over {j1, j2}
                    # Player 2 must be indifferent: p*U0[i1,j1]+(1-p)*U0[i2,j1] = p*U0[i1,j2]+(1-p)*U0[i2,j2]
                    # For player 1's mixture to make player 2 indifferent:
                    denom1 = (payoff_1[i1, j1] - payoff_1[i2, j1] - payoff_1[i1, j2] + payoff_1[i2, j2])
                    if abs(denom1) < 1e-12:
                        continue
                    p = (payoff_1[i2, j2] - payoff_1[i2, j1]) / denom1
                    # For player 2's mixture to make player 1 indifferent:
                    denom2 = (payoff_0[i1, j1] - payoff_0[i1, j2] - payoff_0[i2, j1] + payoff_0[i2, j2])
                    if abs(denom2) < 1e-12:
                        continue
                    q = (payoff_0[i2, j2] - payoff_0[i1, j2]) / denom2  # swapped to fix
                    # Actually: q makes player 1 indifferent between i1 and i2
                    # q * U0[i1,j1] + (1-q)*U0[i1,j2] = q*U0[i2,j1] + (1-q)*U0[i2,j2]
                    q = (payoff_0[i2, j2] - payoff_0[i1, j2]) / denom2

                    if 0 <= p <= 1 and 0 <= q <= 1:
                        pi0 = np.zeros(n); pi0[i1] = p; pi0[i2] = 1 - p
                        pi1 = np.zeros(m); pi1[j1] = q; pi1[j2] = 1 - q
                        val0 = pi0 @ payoff_0 @ pi1
                        val1 = pi0 @ payoff_1 @ pi1
                        val = val0 + val1
                        if val > best_value:
                            best_value = val
                            best_ne = (pi0, pi1)

    if best_ne is None:
        # Fallback: use joint action maximizing sum of payoffs
        total = payoff_0 + payoff_1
        idx = np.unravel_index(total.argmax(), total.shape)
        pi0 = np.zeros(n); pi0[idx[0]] = 1.0
        pi1 = np.zeros(m); pi1[idx[1]] = 1.0
        best_ne = (pi0, pi1)

    return best_ne


def run_nash_q(game, n_iter, seed, nash_update_freq=50):
    """Nash-Q: each agent maintains Q_i(a_i, a_j), backs up via Nash of stage game.
    In a stateless game, Q-backup is just the immediate reward; Nash only needed for policy.
    Nash equilibrium recomputed every nash_update_freq steps for efficiency."""
    rng = np.random.RandomState(seed)
    n = game.n_actions
    # Q-tables over joint actions
    Q = [np.zeros((n, n)), np.zeros((n, n))]
    alpha0 = 0.5
    history = np.zeros((n_iter, 2))

    # Initial Nash policy (uniform)
    pi0 = np.ones(n) / n
    pi1 = np.ones(n) / n

    for t in range(n_iter):
        alpha = alpha0 / (1 + t * 0.0001)

        # Recompute Nash periodically
        if t % nash_update_freq == 0:
            pi0, pi1 = solve_nash_2player(Q[0], Q[1])

        # Epsilon-greedy exploration mixed with Nash policy
        eps = max(0.3 * (0.999 ** t), 0.01)
        policies = []
        actions = []
        for i, pi in enumerate([pi0, pi1]):
            if rng.random() < eps:
                a = rng.randint(n)
            else:
                a = rng.choice(n, p=pi)
            actions.append(a)
            policy = (1 - eps) * pi + eps / n
            policies.append(policy)

        # Observe rewards and update Q (stateless: backup = reward, no discount)
        a0, a1 = actions
        for i in range(2):
            r = game.payoff[a0, a1, i]
            Q[i][a0, a1] += alpha * (r - Q[i][a0, a1])

        # Record expected action
        history[t, 0] = np.dot(policies[0], np.arange(n))
        history[t, 1] = np.dot(policies[1], np.arange(n))

    return history

# ============================================================================
# Algorithm 3: WoLF-PHC
# ============================================================================

def run_wolf_phc(game, n_iter, seed, delta_w=0.002, delta_l=0.02):
    """WoLF-PHC: policy hill-climbing with variable learning rate."""
    rng = np.random.RandomState(seed)
    n = game.n_actions
    Q = [np.zeros(n), np.zeros(n)]       # Q over own actions
    pi = [np.ones(n) / n, np.ones(n) / n]  # current policy
    pi_bar = [np.ones(n) / n, np.ones(n) / n]  # average policy
    counts = [np.zeros(n), np.zeros(n)]   # action counts for average
    alpha0 = 0.5
    history = np.zeros((n_iter, 2))
    n_visits = 0

    for t in range(n_iter):
        alpha = alpha0 / (1 + t * 0.0001)

        # Select actions from current policies
        actions = []
        for i in range(2):
            a = rng.choice(n, p=pi[i])
            actions.append(a)

        a0, a1 = actions

        # Observe rewards and update Q (using own-action Q, like IQL)
        for i in range(2):
            r = game.payoff[a0, a1, i]
            Q[i][actions[i]] += alpha * (r - Q[i][actions[i]])

        n_visits += 1

        for i in range(2):
            # Update average policy
            counts[i][actions[i]] += 1
            pi_bar[i] = counts[i] / counts[i].sum()

            # Determine if winning or losing
            v_pi = np.dot(pi[i], Q[i])
            v_bar = np.dot(pi_bar[i], Q[i])
            delta = delta_l if v_pi < v_bar else delta_w

            # Move policy toward greedy action
            a_star = np.argmax(Q[i])
            for a in range(n):
                if a == a_star:
                    pi[i][a] += delta
                else:
                    pi[i][a] -= delta / (n - 1)
            # Project onto simplex
            pi[i] = np.clip(pi[i], 0.0, 1.0)
            pi[i] /= pi[i].sum()

        # Record expected action
        for i in range(2):
            history[t, i] = np.dot(pi[i], np.arange(n))

    return history

# ============================================================================
# Experiment runner
# ============================================================================

def smooth(x, window=500):
    """Running average for plotting."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')

def run_experiment(game, n_iter=50000, n_seeds=20):
    """Run all three algorithms on a game across multiple seeds."""
    algos = {
        'IQL': run_iql,
        'Nash-Q': run_nash_q,
        'WoLF-PHC': run_wolf_phc,
    }
    results = {}
    for name, fn in algos.items():
        all_hist = []
        print(f"  Running {name}...")
        for s in trange(n_seeds, desc=f"    {name}", leave=False):
            h = fn(game, n_iter, seed=42 + s)
            all_hist.append(h)
        results[name] = np.array(all_hist)  # (n_seeds, n_iter, 2)
    return results

def compute_stats(results, game, n_iter, tail=5000):
    """Compute final action, profit, distance to Nash over last `tail` iterations."""
    stats = {}
    for name, data in results.items():
        # data: (n_seeds, n_iter, 2)
        # Mean action over last tail iterations, per seed
        tail_actions = data[:, -tail:, :].mean(axis=1)  # (n_seeds, 2)
        mean_action = tail_actions.mean(axis=0)  # (2,)
        se_action = tail_actions.std(axis=0) / np.sqrt(len(tail_actions))

        # Approximate profit: use payoff matrix at nearest integer action
        profits = []
        for seed_idx in range(data.shape[0]):
            a_i = int(round(tail_actions[seed_idx, 0]))
            a_j = int(round(tail_actions[seed_idx, 1]))
            a_i = np.clip(a_i, 0, game.n_actions - 1)
            a_j = np.clip(a_j, 0, game.n_actions - 1)
            profits.append(game.payoff[a_i, a_j, 0])
        profits = np.array(profits)

        dist = np.abs(mean_action - game.nash_action).mean()

        # Convergence iteration: first time mean action stays within 0.5 of Nash
        mean_traj = data[:, :, :].mean(axis=0)  # (n_iter, 2)
        avg_action = (mean_traj[:, 0] + mean_traj[:, 1]) / 2
        smoothed = smooth(avg_action, window=1000)
        conv_iter = n_iter
        for i in range(len(smoothed)):
            if abs(smoothed[i] - game.nash_action) < 0.5:
                conv_iter = i + 1000
                break

        stats[name] = {
            'mean_action': mean_action.mean(),
            'se_action': se_action.mean(),
            'mean_profit': profits.mean(),
            'se_profit': profits.std() / np.sqrt(len(profits)),
            'dist_nash': dist,
            'conv_iter': conv_iter,
        }
    return stats

# ============================================================================
# Output generation
# ============================================================================

def make_figure(results_cournot, results_bertrand, game_c, game_b, n_iter, outpath):
    """2-panel convergence figure."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    algo_colors = {
        'IQL': DOMAIN_COLORS['IQL'],
        'Nash-Q': DOMAIN_COLORS['Nash-Q'],
        'WoLF-PHC': DOMAIN_COLORS['WoLF-PHC'],
    }

    for ax, results, game, title in [
        (axes[0], results_cournot, game_c, 'Cournot (Quantity)'),
        (axes[1], results_bertrand, game_b, 'Bertrand (Price)'),
    ]:
        for name, data in results.items():
            # Average across seeds and players
            mean_traj = data.mean(axis=(0, 2))  # (n_iter,)
            smoothed = smooth(mean_traj, window=1000)
            ax.plot(np.arange(len(smoothed)) + 1000, smoothed,
                    label=name, color=algo_colors[name], linewidth=1.5)

        ax.axhline(game.nash_action, **BENCH_STYLE,
                   label=f'Nash ($a^*={game.nash_action:.2f}$)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$\mathbb{E}[a_i]$')
        ax.set_title(title)
        ax.legend(loc='best', framealpha=0.8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"  Figure saved: {outpath}")
    plt.close(fig)

def make_table(stats_c, stats_b, game_c, game_b, outpath):
    """LaTeX results table."""
    lines = []
    lines.append(r'\begin{tabular}{ll rr rr}')
    lines.append(r'\hline')
    lines.append(r'Game & Algorithm & Action & Profit & $|a - a^*|$ & Conv.\ iter \\')
    lines.append(r'\hline')
    for game, stats, gobj in [('Cournot', stats_c, game_c), ('Bertrand', stats_b, game_b)]:
        first = True
        for algo in ['IQL', 'Nash-Q', 'WoLF-PHC']:
            s = stats[algo]
            gname = game if first else ''
            lines.append(
                f'{gname} & {algo} & '
                f'${s["mean_action"]:.2f} \\pm {s["se_action"]:.2f}$ & '
                f'${s["mean_profit"]:.1f} \\pm {s["se_profit"]:.1f}$ & '
                f'${s["dist_nash"]:.2f}$ & '
                f'{s["conv_iter"]:,} \\\\'
            )
            first = False
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Table saved: {outpath}")

# ============================================================================
# Compute / cache
# ============================================================================

def compute_data():
    """Run all computation and return results dict."""
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    np.random.seed(42)
    N_ITER = CONFIG['n_iter']
    N_SEEDS = CONFIG['n_seeds']

    print("=" * 70)
    print("Cournot and Bertrand Duopoly: Multi-Agent Q-Learning Convergence")
    print("=" * 70)

    # --- Game parameters ---
    game_c = CournotDuopoly(a=CONFIG['cournot_a'], c=CONFIG['cournot_c'], Q_max=CONFIG['cournot_Q_max'])
    game_b = BertrandDuopoly(a=CONFIG['bertrand_a'], b=CONFIG['bertrand_b'], e=CONFIG['bertrand_e'],
                             c=CONFIG['bertrand_c'], P_max=CONFIG['bertrand_P_max'])

    print(f"\nCournot parameters: a={game_c.a}, c={game_c.c}, Q_max={game_c.Q_max}")
    print(f"  Nash equilibrium: q*={game_c.nash_action:.4f}, profit*={game_c.nash_profit:.4f}")
    print(f"\nBertrand parameters: a={game_b.a}, b={game_b.b}, e={game_b.e}, c={game_b.c}, P_max={game_b.P_max}")
    print(f"  Nash equilibrium: p*={game_b.nash_action:.4f}, profit*={game_b.nash_profit:.4f}")

    print(f"\nAlgorithms: IQL, Nash-Q, WoLF-PHC")
    print(f"Seeds: {N_SEEDS}, Iterations: {N_ITER:,}")
    print()

    # --- Run experiments ---
    print("Running Cournot experiments...")
    results_c = run_experiment(game_c, N_ITER, N_SEEDS)
    print("\nRunning Bertrand experiments...")
    results_b = run_experiment(game_b, N_ITER, N_SEEDS)

    # --- Compute statistics ---
    stats_c = compute_stats(results_c, game_c, N_ITER)
    stats_b = compute_stats(results_b, game_b, N_ITER)

    # --- Print results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for game_name, stats, game in [("Cournot", stats_c, game_c), ("Bertrand", stats_b, game_b)]:
        print(f"\n{game_name} Duopoly (Nash action = {game.nash_action:.2f}, Nash profit = {game.nash_profit:.2f})")
        print(f"{'Algorithm':<12} {'Action':>14} {'Profit':>14} {'|a-a*|':>8} {'Conv iter':>10}")
        print("-" * 60)
        for algo in ['IQL', 'Nash-Q', 'WoLF-PHC']:
            s = stats[algo]
            print(f"{algo:<12} {s['mean_action']:>6.2f} +/- {s['se_action']:.2f}"
                  f"   {s['mean_profit']:>6.1f} +/- {s['se_profit']:.1f}"
                  f"   {s['dist_nash']:>6.2f}   {s['conv_iter']:>8,}")

    # --- Serialize for caching ---
    # Convert numpy arrays in results to lists for pickling
    def serialize_experiment(results_dict):
        serialized = {}
        for name, arr in results_dict.items():
            serialized[name] = arr.tolist()
        return serialized

    data = {
        'results_c': serialize_experiment(results_c),
        'results_b': serialize_experiment(results_b),
        'stats_c': stats_c,
        'stats_b': stats_b,
        'game_c_params': {'a': game_c.a, 'c': game_c.c, 'Q_max': game_c.Q_max,
                          'nash_action': game_c.nash_action, 'nash_profit': game_c.nash_profit},
        'game_b_params': {'a': game_b.a, 'b': game_b.b, 'e': game_b.e, 'c': game_b.c, 'P_max': game_b.P_max,
                          'nash_action': game_b.nash_action, 'nash_profit': game_b.nash_profit},
        'n_iter': N_ITER,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    """Generate all plots and tables from precomputed data."""
    OUTDIR = os.path.dirname(os.path.abspath(__file__))

    # Reconstruct numpy arrays from lists
    results_c = {name: np.array(arr) for name, arr in data['results_c'].items()}
    results_b = {name: np.array(arr) for name, arr in data['results_b'].items()}
    stats_c = data['stats_c']
    stats_b = data['stats_b']

    # Reconstruct game objects for plotting
    gcp = data['game_c_params']
    gbp = data['game_b_params']
    game_c = CournotDuopoly(a=gcp['a'], c=gcp['c'], Q_max=gcp['Q_max'])
    game_b = BertrandDuopoly(a=gbp['a'], b=gbp['b'], e=gbp['e'], c=gbp['c'], P_max=gbp['P_max'])
    N_ITER = data['n_iter']

    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    fig_path = os.path.join(OUTDIR, 'cournot_bertrand_marl.png')
    tab_path = os.path.join(OUTDIR, 'cournot_bertrand_results.tex')

    make_figure(results_c, results_b, game_c, game_b, N_ITER, fig_path)
    make_table(stats_c, stats_b, game_c, game_b, tab_path)

    print(f"\nOutput files:")
    print(f"  {fig_path}")
    print(f"  {tab_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()
    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
