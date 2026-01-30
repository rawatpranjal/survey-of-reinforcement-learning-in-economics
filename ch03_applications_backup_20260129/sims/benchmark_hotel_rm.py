# Hotel Revenue Management Benchmark: Scaling Analysis
# Chapter 3 -- Economic Benchmarks
# Dynamic pricing of perishable hotel rooms over a finite booking horizon.
# Systematic scaling sweep across capacity levels.

import matplotlib
matplotlib.use('Agg')

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import poisson, binom

from econ_benchmark import (
    EconBenchmark, run_value_iteration, run_dqn,
    evaluate_dp_policy, evaluate_dqn_policy, evaluate_heuristic,
    compute_policy_entropy, compute_policy_agreement, state_coverage,
    compute_q_error, evaluate_with_decomposition, capture_stdout,
    make_scaling_table, DP_FEASIBLE_THRESHOLD
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T_HORIZON = 20
D_BINS = 3
PRICE_LEVELS = [50, 80, 120, 180, 250]
N_PRICES = len(PRICE_LEVELS)
ALPHA_LOGIT = 0.02
REF_PRICE = 120.0
BASE_ARRIVAL_RATE = 3.0
MAX_ARRIVALS = 8
GAMMA = 1.0

# Scaling sweep configuration
# C=10: 660 states (DP feasible), C=50: 3060 states (DP feasible but slower)
# Note: Hotel RM has linear state scaling, so DP remains tractable at larger C
COMPLEXITY_SWEEP = [10, 50]  # Capacity: small vs large
SEEDS = [42, 123, 7]  # 3 seeds for faster runs
NUM_EPISODES = 500
EVAL_FREQ = 50
EVAL_EPISODES = 10

OUTPUT_DIR = Path(__file__).resolve().parent
SCRIPT_NAME = 'hotel_rm'

LAMBDA_RATES = {0: 0.5 * BASE_ARRIVAL_RATE,
                1: BASE_ARRIVAL_RATE,
                2: 1.8 * BASE_ARRIVAL_RATE}

# Figure styling
DP_COLOR = '#1f77b4'
DQN_COLOR = '#ff7f0e'
HEURISTIC_COLORS = ['#2ca02c', '#d62728', '#9467bd']


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class HotelRMEnv(EconBenchmark):
    """Hotel revenue management MDP.

    State: (t, I, d) -- days until check-in, remaining rooms, demand intensity.
    Action: price tier index in {0,...,4}.
    """

    def __init__(self, capacity):
        self.C = capacity
        self.T = T_HORIZON
        self._num_states = self.T * (self.C + 1) * D_BINS
        self._num_actions = N_PRICES
        self.current_state = None
        self.steps = 0

        self.complexity_param = capacity
        self.dp_feasible = self._num_states <= DP_FEASIBLE_THRESHOLD

        # Track last step for decomposition
        self._last_price = 0
        self._last_bookings = 0
        self._last_arrivals = 0

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    def reset(self):
        self.current_state = (self.T - 1, self.C, 1)
        self.steps = 0
        return self.current_state

    def step(self, action):
        t, I, d = self.current_state
        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]

        arrivals = min(np.random.poisson(lam), MAX_ARRIVALS)
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        bookings = 0
        for _ in range(arrivals):
            if np.random.rand() < q_p:
                bookings += 1
        bookings = min(bookings, I)
        reward = price * bookings

        # Store for decomposition
        self._last_price = price
        self._last_bookings = bookings
        self._last_arrivals = arrivals

        next_I = I - bookings
        next_d = _demand_transition(arrivals)
        next_t = t - 1
        self.steps += 1

        done = (next_t < 0) or (next_I == 0) or (self.steps >= self.T)
        if done:
            self.current_state = (0, next_I, next_d)
        else:
            self.current_state = (next_t, next_I, next_d)

        return self.current_state, reward, done

    def state_to_index(self, state):
        t, I, d = state
        return t * ((self.C + 1) * D_BINS) + I * D_BINS + d

    def index_to_state(self, idx):
        d = idx % D_BINS
        idx //= D_BINS
        I = idx % (self.C + 1)
        t = idx // (self.C + 1)
        return (t, I, d)

    def state_to_features(self, state):
        t, I, d = state
        return np.array([t / max(self.T - 1, 1),
                         I / max(self.C, 1),
                         d / max(D_BINS - 1, 1)], dtype=np.float32)

    def expected_reward(self, state, action):
        t, I, d = state
        if I == 0 or t <= 0:
            return 0.0
        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        exp_rew = 0.0
        for arr in range(MAX_ARRIVALS + 1):
            p_arr = poisson.pmf(arr, lam)
            if p_arr < 1e-10:
                continue
            exp_book = min(arr * q_p, I)
            exp_rew += p_arr * price * exp_book
        return exp_rew

    def transition_distribution(self, state, action):
        t, I, d = state
        if I == 0 or t <= 0:
            return [(state, 1.0)]

        price = PRICE_LEVELS[action]
        lam = LAMBDA_RATES[d]
        q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
        next_t = t - 1

        result = {}
        for arr in range(MAX_ARRIVALS + 1):
            p_arr = poisson.pmf(arr, lam)
            if p_arr < 1e-10:
                continue
            next_d = _demand_transition(arr)
            max_book = min(arr, I)
            for book in range(max_book + 1):
                p_book = binom.pmf(book, arr, q_p)
                if p_book < 1e-10:
                    continue
                prob = p_arr * p_book
                ns = (next_t, I - book, next_d)
                result[ns] = result.get(ns, 0.0) + prob

        return list(result.items())


def _demand_transition(arrivals):
    if arrivals < 0.7 * BASE_ARRIVAL_RATE:
        return 0
    elif arrivals < 1.5 * BASE_ARRIVAL_RATE:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Decomposition function for hotel RM
# ---------------------------------------------------------------------------
def hotel_decompose(state, action, reward, next_state, env):
    """Decompose reward into components for hotel RM environment."""
    t, I, d = state
    price = env._last_price
    bookings = env._last_bookings
    arrivals = env._last_arrivals

    components = {
        'revenue': reward,
        'price_chosen': price,
        'bookings': bookings,
        'arrivals': arrivals,
        'inventory_start': I,
        'conversion_rate': bookings / max(arrivals, 1),
        'occupancy_delta': bookings / max(env.C, 1),
    }

    # Track which price tier was chosen
    for i, p in enumerate(PRICE_LEVELS):
        components[f'price_tier_{i}'] = 1.0 if price == p else 0.0

    # Demand state tracking
    for ds in range(D_BINS):
        components[f'demand_state_{ds}'] = 1.0 if d == ds else 0.0

    return components


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def make_heuristic_fixed_price(capacity):
    """Always charge the reference price ($120, index 2)."""
    def heuristic(state):
        return 2
    return heuristic


def make_heuristic_emsr_b(capacity):
    """EMSR-b: price based on inventory scarcity relative to time."""
    def heuristic(state):
        t, I, d = state
        if t <= 0 or I == 0:
            return 0
        inv_ratio = I / max(capacity, 1)
        time_ratio = t / T_HORIZON
        if inv_ratio > time_ratio + 0.2:
            return 0
        elif inv_ratio > time_ratio:
            return 1
        elif inv_ratio > time_ratio - 0.2:
            return 2
        elif inv_ratio > time_ratio - 0.4:
            return 3
        else:
            return 4
    return heuristic


def make_heuristic_myopic(capacity):
    """Maximize immediate expected revenue."""
    def heuristic(state):
        t, I, d = state
        if t <= 0 or I == 0:
            return 0
        lam = LAMBDA_RATES[d]
        best_a, best_r = 0, -1.0
        for a, price in enumerate(PRICE_LEVELS):
            q_p = 1.0 / (1.0 + np.exp(ALPHA_LOGIT * (price - REF_PRICE)))
            exp_book = 0.0
            for arr in range(MAX_ARRIVALS + 1):
                p_arr = poisson.pmf(arr, lam)
                exp_book += p_arr * min(arr * q_p, I)
            er = price * exp_book
            if er > best_r:
                best_r = er
                best_a = a
        return best_a
    return heuristic


# ---------------------------------------------------------------------------
# Run single complexity level
# ---------------------------------------------------------------------------
def run_single_complexity(C, seeds):
    """Run benchmark for a single complexity level (C capacity)."""
    env = HotelRMEnv(capacity=C)
    result = {
        'complexity': C,
        'states': env.num_states,
        'dp_feasible': env.dp_feasible,
    }

    print(f"\n  C={C}: {env.num_states:,} states, DP feasible={env.dp_feasible}")

    # --- Value Iteration (if feasible) ---
    V, policy, vi_metrics = None, None, None
    if env.dp_feasible:
        print("    Running Value Iteration...")
        V, policy, vi_metrics = run_value_iteration(env, gamma=GAMMA)
        dp_reward = evaluate_dp_policy(env, policy, gamma=GAMMA,
                                       n_episodes=200, horizon=T_HORIZON)
        result['dp_reward'] = dp_reward
        result['dp_time'] = vi_metrics.wall_time
        result['dp_iterations'] = vi_metrics.iterations
        print(f"      VI: {vi_metrics.iterations} iter, time={vi_metrics.wall_time:.2f}s, "
              f"reward={dp_reward:.1f}")
    else:
        result['dp_reward'] = None
        result['dp_time'] = None
        result['dp_iterations'] = None
        print("    DP skipped (state space too large)")

    # --- DQN (multiple seeds) ---
    print(f"    Running DQN ({len(seeds)} seeds)...")
    dqn_rewards = []
    dqn_times = []
    dqn_q_errors = []
    dqn_agreements = []
    all_curves = []
    checkpoint_episodes = None

    for i, seed in enumerate(seeds):
        env_dqn = HotelRMEnv(capacity=C)
        q_net, metrics = run_dqn(
            env_dqn, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=T_HORIZON, seed=seed, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            desc=f"C={C} seed {i+1}/{len(seeds)}"
        )

        reward = evaluate_dqn_policy(env, q_net, n_episodes=200, horizon=T_HORIZON)
        dqn_rewards.append(reward)
        dqn_times.append(metrics.wall_time)
        all_curves.append(metrics.eval_checkpoints)
        checkpoint_episodes = metrics.checkpoint_episodes

        if V is not None:
            q_err = compute_q_error(q_net, V, env, n_samples=500, gamma=GAMMA)
            dqn_q_errors.append(q_err)
            agreement = compute_policy_agreement(q_net, policy, env, n_samples=500)
            dqn_agreements.append(agreement)

        print(f"      seed={seed}: reward={reward:.1f}, time={metrics.wall_time:.2f}s")

    result['dqn_reward_mean'] = np.mean(dqn_rewards)
    result['dqn_reward_std'] = np.std(dqn_rewards)
    result['dqn_rewards'] = dqn_rewards
    result['dqn_time_mean'] = np.mean(dqn_times)
    result['dqn_curves'] = all_curves
    result['checkpoint_episodes'] = checkpoint_episodes

    if dqn_q_errors:
        result['q_error'] = np.mean(dqn_q_errors)
        result['agreement'] = np.mean(dqn_agreements)
    else:
        result['q_error'] = None
        result['agreement'] = None

    # --- Heuristics ---
    print("    Running Heuristics...")
    h_fixed = evaluate_heuristic(env, make_heuristic_fixed_price(C),
                                 n_episodes=200, horizon=T_HORIZON)
    h_emsr = evaluate_heuristic(env, make_heuristic_emsr_b(C),
                                n_episodes=200, horizon=T_HORIZON)
    h_myopic = evaluate_heuristic(env, make_heuristic_myopic(C),
                                  n_episodes=200, horizon=T_HORIZON)

    result['heuristics'] = {
        'fixed_price': h_fixed,
        'emsr_b': h_emsr,
        'myopic': h_myopic,
    }
    print(f"      Fixed={h_fixed:.1f}, EMSR-b={h_emsr:.1f}, Myopic={h_myopic:.1f}")

    # --- Decomposition (using best DQN policy) ---
    if all_curves:
        print("    Running decomposition analysis...")
        env_decomp = HotelRMEnv(capacity=C)
        q_net_decomp, _ = run_dqn(
            env_decomp, gamma=GAMMA, num_episodes=NUM_EPISODES,
            episode_horizon=T_HORIZON, seed=42, replay_size=10_000,
            batch_size=64, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01,
            epsilon_decay_frac=0.6, target_update_freq=50,
            eval_freq=EVAL_FREQ, eval_episodes=EVAL_EPISODES,
            show_progress=False
        )

        def dqn_policy(s):
            s_feat = env_decomp.state_to_features(s)
            with torch.no_grad():
                q_vals = q_net_decomp(torch.FloatTensor(s_feat).unsqueeze(0))
                return q_vals.argmax(dim=1).item()

        decomp = evaluate_with_decomposition(
            env_decomp, dqn_policy, n_episodes=100, horizon=T_HORIZON,
            decompose_fn=hotel_decompose
        )
        result['decomposition'] = decomp.component_rewards

        # Compute occupancy rate
        total_bookings = decomp.component_rewards.get('bookings', 0) * T_HORIZON
        result['occupancy_rate'] = total_bookings / C

    return result


# ---------------------------------------------------------------------------
# Scaling sweep
# ---------------------------------------------------------------------------
def run_scaling_sweep():
    """Run benchmark across all complexity levels."""
    print("=" * 70)
    print("  Hotel Revenue Management: Scaling Analysis")
    print("=" * 70)
    print(f"  Complexity sweep: C in {COMPLEXITY_SWEEP}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Episodes: {NUM_EPISODES}, Horizon: {T_HORIZON}")

    results = []
    for C in COMPLEXITY_SWEEP:
        result = run_single_complexity(C, SEEDS)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figures(results):
    """Generate scaling comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Revenue vs complexity
    ax = axes[0]
    complexities = [r['complexity'] for r in results]
    dp_rewards = [r['dp_reward'] if r['dp_reward'] else np.nan for r in results]
    dqn_means = [r['dqn_reward_mean'] for r in results]
    dqn_stds = [r['dqn_reward_std'] for r in results]

    x = np.arange(len(complexities))
    width = 0.35

    ax.bar(x - width/2, dp_rewards, width, label='DP', color=DP_COLOR, alpha=0.8)
    ax.bar(x + width/2, dqn_means, width, yerr=dqn_stds, label='DQN',
           color=DQN_COLOR, alpha=0.8, capsize=3)

    ax.set_xlabel('Capacity (C)', fontsize=11)
    ax.set_ylabel('Episode Revenue', fontsize=11)
    ax.set_title('Hotel RM: Revenue vs Capacity', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(complexities)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Learning curves for each complexity
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    for i, r in enumerate(results):
        C = r['complexity']
        curves = r['dqn_curves']
        episodes = r['checkpoint_episodes']

        if curves and episodes:
            min_len = min(len(c) for c in curves)
            curves_arr = np.array([c[:min_len] for c in curves])
            episodes = episodes[:min_len]

            mean_rewards = np.mean(curves_arr, axis=0)
            std_rewards = np.std(curves_arr, axis=0)

            ax.plot(episodes, mean_rewards, color=colors[i], linewidth=2, label=f'C={C}')
            ax.fill_between(episodes, mean_rewards - std_rewards,
                           mean_rewards + std_rewards, color=colors[i], alpha=0.15)

        if r['dp_reward']:
            ax.axhline(r['dp_reward'], color=colors[i], linestyle='--',
                      linewidth=1, alpha=0.7)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Episode Revenue', fontsize=11)
    ax.set_title('Hotel RM: Learning Curves by Capacity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def make_latex_tables(results):
    """Generate LaTeX tables for scaling results."""
    tex = make_scaling_table(results, 'C', 'Hotel RM')
    out_path = OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")

    # Decomposition table for largest complexity
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        lines = []
        lines.append(r'\begin{tabular}{lr}')
        lines.append(r'\toprule')
        lines.append(r'Metric & Mean per Episode \\')
        lines.append(r'\midrule')

        decomp = largest['decomposition']
        lines.append(f"Revenue & ${decomp.get('revenue', 0):.1f}$ \\\\")
        lines.append(f"Bookings & ${decomp.get('bookings', 0):.2f}$ \\\\")
        lines.append(f"Avg Price & ${decomp.get('price_chosen', 0):.0f}$ \\\\")
        lines.append(f"Arrivals & ${decomp.get('arrivals', 0):.2f}$ \\\\")
        lines.append(f"Conversion Rate & ${decomp.get('conversion_rate', 0):.2%}$ \\\\")

        # Price tier distribution
        for i in range(N_PRICES):
            key = f'price_tier_{i}'
            if key in decomp:
                lines.append(f"Price Tier {i} (\\${PRICE_LEVELS[i]}) & "
                           f"${decomp[key]:.2%}$ \\\\")

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        decomp_tex = '\n'.join(lines)
        decomp_path = OUTPUT_DIR / f'{SCRIPT_NAME}_decomposition.tex'
        with open(decomp_path, 'w') as f:
            f.write(decomp_tex)
        print(f"Decomposition table saved to {decomp_path}")


# ---------------------------------------------------------------------------
# Detailed stdout output
# ---------------------------------------------------------------------------
def print_detailed_results(results):
    """Print detailed results to stdout."""
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)

    # Scaling summary table
    print("\n  Scaling Summary:")
    print("  " + "-" * 70)
    print(f"  {'C':>4} | {'States':>8} | {'DP Revenue':>11} | {'DQN Revenue':>18} | "
          f"{'Q-Error':>8} | {'Agr.':>6}")
    print("  " + "-" * 70)

    for r in results:
        C = r['complexity']
        states = r['states']
        dp_r = f"{r['dp_reward']:.1f}" if r['dp_reward'] else "---"
        dqn_r = f"{r['dqn_reward_mean']:.1f} +/- {r['dqn_reward_std']:.1f}"
        q_err = f"{r['q_error']:.3f}" if r['q_error'] else "---"
        agr = f"{r['agreement']:.1%}" if r['agreement'] else "---"
        print(f"  {C:>4} | {states:>8,} | {dp_r:>11} | {dqn_r:>18} | {q_err:>8} | {agr:>6}")

    print("  " + "-" * 70)

    # Per-seed results
    print("\n  Per-Seed DQN Revenues:")
    for r in results:
        C = r['complexity']
        rewards = r['dqn_rewards']
        print(f"    C={C}: " + ", ".join(f"{rw:.1f}" for rw in rewards))

    # Heuristic comparison
    print("\n  Heuristic Comparison:")
    print("  " + "-" * 55)
    print(f"  {'C':>4} | {'Fixed':>10} | {'EMSR-b':>10} | {'Myopic':>10}")
    print("  " + "-" * 55)
    for r in results:
        C = r['complexity']
        h = r['heuristics']
        print(f"  {C:>4} | {h['fixed_price']:>10.1f} | {h['emsr_b']:>10.1f} | "
              f"{h['myopic']:>10.1f}")
    print("  " + "-" * 55)

    # Decomposition for largest C
    largest = results[-1]
    if 'decomposition' in largest and largest['decomposition']:
        print(f"\n  Revenue Decomposition (C={largest['complexity']}, DQN policy):")
        decomp = largest['decomposition']
        print(f"    Revenue per episode: {decomp.get('revenue', 0):.1f}")
        print(f"    Bookings per step: {decomp.get('bookings', 0):.2f}")
        print(f"    Avg price: ${decomp.get('price_chosen', 0):.0f}")
        print(f"    Conversion rate: {decomp.get('conversion_rate', 0):.2%}")
        if 'occupancy_rate' in largest:
            print(f"    Occupancy rate: {largest['occupancy_rate']:.2%}")

    # Output files
    print("\n  Output Files:")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_scaling.png'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_results.tex'}")
    print(f"    {OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    stdout_path = OUTPUT_DIR / f'{SCRIPT_NAME}_stdout.txt'
    with capture_stdout(stdout_path):
        print("Hotel Revenue Management Benchmark")
        print(f"Complexity sweep: C in {COMPLEXITY_SWEEP}")
        print(f"Seeds: {SEEDS}")

        results = run_scaling_sweep()
        make_figures(results)
        make_latex_tables(results)
        print_detailed_results(results)

        print("\nBenchmark complete.")
