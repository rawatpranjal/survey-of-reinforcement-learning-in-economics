# Kuhn Poker Equilibrium Computation — Chapter 5, RL in Games
# Compares CFR, CFR+, and Fictitious Play convergence to Nash equilibrium

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
import time

np.random.seed(42)

# =============================================================================
# GAME DEFINITION
# =============================================================================

CARDS = ['J', 'Q', 'K']
CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}
NUM_ACTIONS = 2  # 0 = pass/fold, 1 = bet/call


def is_terminal(history: str) -> bool:
    return history in ['pp', 'pbp', 'pbb', 'bp', 'bb']


def get_payoff(cards: Tuple[str, str], history: str) -> float:
    """Get payoff for player 0. cards = (p0_card, p1_card)."""
    if history == 'pp':
        return 1 if CARD_RANK[cards[0]] > CARD_RANK[cards[1]] else -1
    elif history == 'bp':
        return 1
    elif history == 'pbp':
        return -1
    elif history in ['bb', 'pbb']:
        return 2 if CARD_RANK[cards[0]] > CARD_RANK[cards[1]] else -2
    raise ValueError(f"Unknown history: {history}")


def get_current_player(history: str) -> int:
    return len(history) % 2


# =============================================================================
# EXPLOITABILITY COMPUTATION (CORRECT)
# =============================================================================

def compute_exploitability(strategy: Dict[str, np.ndarray]) -> float:
    """
    Compute exploitability = BR_value(P0) + BR_value(P1).

    The BR for each player is computed at the INFORMATION SET level:
    for each info set, we compute the EV of each action averaged over
    opponent cards, then choose the action with higher EV.

    At Nash equilibrium, exploitability = 0.
    """
    expl = 0.0
    for br_player in [0, 1]:
        br_value = compute_br_value(strategy, br_player)
        expl += br_value
    return expl


def compute_br_value(opp_strategy: Dict[str, np.ndarray], br_player: int) -> float:
    """
    Compute BR value for br_player against opponent's strategy.

    1. For each info set, compute EV of each action averaged over opponent cards
    2. Pick the action with higher EV to form the BR strategy
    3. Compute overall EV under this BR strategy
    """
    br_strategy = compute_br_strategy(opp_strategy, br_player)

    # Compute EV under BR strategy
    total_ev = 0.0
    for my_card in CARDS:
        for opp_card in CARDS:
            if my_card == opp_card:
                continue
            cards = (my_card, opp_card) if br_player == 0 else (opp_card, my_card)
            ev = compute_value_with_strategies(opp_strategy, br_strategy, br_player, cards, '')
            total_ev += ev / 6

    return total_ev


def compute_br_strategy(opp_strategy: Dict[str, np.ndarray], br_player: int) -> Dict[str, np.ndarray]:
    """Compute best response strategy for br_player."""
    br_strategy = {}

    if br_player == 0:
        info_sets = ['J', 'Q', 'K', 'Jpb', 'Qpb', 'Kpb']
    else:
        info_sets = ['Jp', 'Qp', 'Kp', 'Jb', 'Qb', 'Kb']

    for info_set in info_sets:
        my_card = info_set[0]
        history = info_set[1:] if len(info_set) > 1 else ''

        # Compute EV for each action, averaged over opponent cards
        action_evs = [0.0, 0.0]

        for opp_card in CARDS:
            if opp_card == my_card:
                continue

            cards = (my_card, opp_card) if br_player == 0 else (opp_card, my_card)

            for action in [0, 1]:
                new_h = history + ('p' if action == 0 else 'b')
                # For downstream decisions, use optimal play
                ev = compute_value_br_optimal(opp_strategy, br_player, cards, new_h, br_strategy)
                action_evs[action] += ev / 2

        # BR action is one with higher EV
        if action_evs[1] > action_evs[0]:
            br_strategy[info_set] = np.array([0.0, 1.0])
        else:
            br_strategy[info_set] = np.array([1.0, 0.0])

    return br_strategy


def compute_value_br_optimal(opp_strategy, br_player, cards, history, br_strategy):
    """Compute value using BR strategy where defined, optimal at downstream nodes."""
    if is_terminal(history):
        payoff = get_payoff(cards, history)
        return payoff if br_player == 0 else -payoff

    player = get_current_player(history)

    if player == br_player:
        my_card = cards[br_player]
        my_info_set = my_card + history

        if my_info_set in br_strategy:
            probs = br_strategy[my_info_set]
            value = 0.0
            for a in [0, 1]:
                new_h = history + ('p' if a == 0 else 'b')
                value += probs[a] * compute_value_br_optimal(opp_strategy, br_player, cards, new_h, br_strategy)
            return value
        else:
            # Downstream: use max
            values = []
            for a in [0, 1]:
                new_h = history + ('p' if a == 0 else 'b')
                values.append(compute_value_br_optimal(opp_strategy, br_player, cards, new_h, br_strategy))
            return max(values)
    else:
        opp_card = cards[1 - br_player]
        opp_info_set = opp_card + history
        probs = opp_strategy.get(opp_info_set, np.array([0.5, 0.5]))
        value = 0.0
        for a in [0, 1]:
            new_h = history + ('p' if a == 0 else 'b')
            value += probs[a] * compute_value_br_optimal(opp_strategy, br_player, cards, new_h, br_strategy)
        return value


def compute_value_with_strategies(opp_strategy, my_strategy, br_player, cards, history):
    """Compute value for br_player using specified strategies."""
    if is_terminal(history):
        payoff = get_payoff(cards, history)
        return payoff if br_player == 0 else -payoff

    player = get_current_player(history)

    if player == br_player:
        my_card = cards[br_player]
        info_set = my_card + history
        probs = my_strategy.get(info_set, np.array([0.5, 0.5]))
    else:
        opp_card = cards[1 - br_player]
        info_set = opp_card + history
        probs = opp_strategy.get(info_set, np.array([0.5, 0.5]))

    value = 0.0
    for a in [0, 1]:
        new_h = history + ('p' if a == 0 else 'b')
        value += probs[a] * compute_value_with_strategies(opp_strategy, my_strategy, br_player, cards, new_h)
    return value


# =============================================================================
# NASH EQUILIBRIUM (closed-form)
# =============================================================================

def nash_equilibrium_strategy(alpha: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Return a Nash equilibrium strategy for Kuhn Poker.
    Alpha parameterizes a family of equilibria, alpha in [0, 1/3].
    """
    return {
        'J': np.array([1 - alpha, alpha]),
        'Q': np.array([1.0, 0.0]),
        'K': np.array([1 - 3*alpha, 3*alpha]),
        'Jpb': np.array([1.0, 0.0]),
        'Qpb': np.array([1 - (alpha + 1/3), alpha + 1/3]),
        'Kpb': np.array([0.0, 1.0]),
        'Jp': np.array([2/3, 1/3]),
        'Qp': np.array([1 - alpha, alpha]),
        'Kp': np.array([0.0, 1.0]),
        'Jb': np.array([1.0, 0.0]),
        'Qb': np.array([1 - (alpha + 1/3), alpha + 1/3]),
        'Kb': np.array([0.0, 1.0]),
    }


# =============================================================================
# CFR IMPLEMENTATION
# =============================================================================

class CFR:
    """Vanilla Counterfactual Regret Minimization."""

    def __init__(self):
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def get_strategy(self, info_set: str) -> np.ndarray:
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        avg = {}
        for info_set, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            if total > 0:
                avg[info_set] = strat_sum / total
            else:
                avg[info_set] = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return avg

    def cfr(self, cards: Tuple[str, str], history: str, reach: np.ndarray) -> np.ndarray:
        if is_terminal(history):
            payoff = get_payoff(cards, history)
            return np.array([payoff, -payoff])

        player = get_current_player(history)
        info_set = cards[player] + history
        strategy = self.get_strategy(info_set)

        self.strategy_sum[info_set] += reach[player] * strategy

        utilities = np.zeros((NUM_ACTIONS, 2))
        for a in [0, 1]:
            new_h = history + ('p' if a == 0 else 'b')
            new_reach = reach.copy()
            new_reach[player] *= strategy[a]
            utilities[a] = self.cfr(cards, new_h, new_reach)

        node_util = strategy @ utilities
        opp = 1 - player
        for a in [0, 1]:
            regret = utilities[a, player] - node_util[player]
            self.regret_sum[info_set][a] += reach[opp] * regret

        return node_util

    def train(self, iterations: int) -> List[Tuple[int, float]]:
        history = []
        for i in range(iterations):
            for c0 in CARDS:
                for c1 in CARDS:
                    if c0 != c1:
                        self.cfr((c0, c1), "", np.ones(2))
            if (i + 1) % 10 == 0 or i == 0:
                expl = compute_exploitability(self.get_average_strategy())
                history.append((i + 1, expl))
        return history


# =============================================================================
# CFR+ IMPLEMENTATION
# =============================================================================

class CFRPlus:
    """CFR+ with linear averaging and regret flooring."""

    def __init__(self):
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def get_strategy(self, info_set: str) -> np.ndarray:
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        avg = {}
        for info_set, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            if total > 0:
                avg[info_set] = strat_sum / total
            else:
                avg[info_set] = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return avg

    def cfr(self, cards: Tuple[str, str], history: str, reach: np.ndarray, t: int) -> np.ndarray:
        if is_terminal(history):
            payoff = get_payoff(cards, history)
            return np.array([payoff, -payoff])

        player = get_current_player(history)
        info_set = cards[player] + history
        strategy = self.get_strategy(info_set)

        self.strategy_sum[info_set] += t * reach[player] * strategy

        utilities = np.zeros((NUM_ACTIONS, 2))
        for a in [0, 1]:
            new_h = history + ('p' if a == 0 else 'b')
            new_reach = reach.copy()
            new_reach[player] *= strategy[a]
            utilities[a] = self.cfr(cards, new_h, new_reach, t)

        node_util = strategy @ utilities
        opp = 1 - player
        for a in [0, 1]:
            regret = utilities[a, player] - node_util[player]
            self.regret_sum[info_set][a] += reach[opp] * regret

        return node_util

    def train(self, iterations: int) -> List[Tuple[int, float]]:
        history = []
        for i in range(iterations):
            for c0 in CARDS:
                for c1 in CARDS:
                    if c0 != c1:
                        self.cfr((c0, c1), "", np.ones(2), i + 1)
            for info_set in self.regret_sum:
                self.regret_sum[info_set] = np.maximum(self.regret_sum[info_set], 0)
            if (i + 1) % 10 == 0 or i == 0:
                expl = compute_exploitability(self.get_average_strategy())
                history.append((i + 1, expl))
        return history


# =============================================================================
# FICTITIOUS PLAY IMPLEMENTATION
# =============================================================================

class FictitiousPlay:
    """Fictitious Play for two-player zero-sum games."""

    def __init__(self):
        self.action_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.ones(NUM_ACTIONS))

    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        avg = {}
        for info_set, counts in self.action_counts.items():
            total = counts.sum()
            avg[info_set] = counts / total
        return avg

    def get_br_action(self, opp_strategy: Dict[str, np.ndarray], br_player: int,
                      my_card: str, history: str) -> int:
        """Get BR action at info set, averaged over opponent cards."""
        action_evs = [0.0, 0.0]

        for opp_card in CARDS:
            if opp_card == my_card:
                continue
            cards = (my_card, opp_card) if br_player == 0 else (opp_card, my_card)

            for action in [0, 1]:
                new_h = history + ('p' if action == 0 else 'b')
                ev = self.evaluate_subtree_optimal(opp_strategy, br_player, cards, new_h)
                action_evs[action] += ev / 2

        return 1 if action_evs[1] > action_evs[0] else 0

    def evaluate_subtree_optimal(self, opp_strategy, br_player, cards, history):
        """Evaluate subtree value with optimal play at BR nodes."""
        if is_terminal(history):
            payoff = get_payoff(cards, history)
            return payoff if br_player == 0 else -payoff

        player = get_current_player(history)

        if player == br_player:
            values = []
            for a in [0, 1]:
                new_h = history + ('p' if a == 0 else 'b')
                values.append(self.evaluate_subtree_optimal(opp_strategy, br_player, cards, new_h))
            return max(values)
        else:
            opp_card = cards[1 - br_player]
            info_set = opp_card + history
            probs = opp_strategy.get(info_set, np.array([0.5, 0.5]))
            value = 0.0
            for a in [0, 1]:
                new_h = history + ('p' if a == 0 else 'b')
                value += probs[a] * self.evaluate_subtree_optimal(opp_strategy, br_player, cards, new_h)
            return value

    def train(self, iterations: int) -> List[Tuple[int, float]]:
        for info_set in ['J', 'Q', 'K', 'Jpb', 'Qpb', 'Kpb', 'Jp', 'Qp', 'Kp', 'Jb', 'Qb', 'Kb']:
            self.action_counts[info_set] = np.ones(NUM_ACTIONS)

        history = []
        for i in range(iterations):
            avg_strat = self.get_average_strategy()

            for card in CARDS:
                for h in ['', 'pb']:
                    info_set = card + h
                    br_a = self.get_br_action(avg_strat, 0, card, h)
                    self.action_counts[info_set][br_a] += 1

            for card in CARDS:
                for h in ['p', 'b']:
                    info_set = card + h
                    br_a = self.get_br_action(avg_strat, 1, card, h)
                    self.action_counts[info_set][br_a] += 1

            if (i + 1) % 10 == 0 or i == 0:
                expl = compute_exploitability(self.get_average_strategy())
                history.append((i + 1, expl))

        return history


# =============================================================================
# EXPERIMENT AND OUTPUT
# =============================================================================

def run_experiment(num_iterations: int = 2000, num_seeds: int = 10):
    results = {
        'CFR': {'exploitability': [], 'time': [], 'final_strategy': None},
        'CFR+': {'exploitability': [], 'time': [], 'final_strategy': None},
        'FP': {'exploitability': [], 'time': [], 'final_strategy': None}
    }

    print(f"Running experiments: {num_iterations} iterations, {num_seeds} seeds")
    print("-" * 50)

    for seed in range(num_seeds):
        np.random.seed(42 + seed)

        for name, Cls in [('CFR', CFR), ('CFR+', CFRPlus), ('FP', FictitiousPlay)]:
            agent = Cls()
            start = time.time()
            hist = agent.train(num_iterations)
            elapsed = time.time() - start
            results[name]['exploitability'].append(hist)
            results[name]['time'].append(elapsed)
            if seed == 0:
                results[name]['final_strategy'] = agent.get_average_strategy()

        print(f"Seed {seed + 1}/{num_seeds} complete")

    return results


def aggregate_results(results):
    aggregated = {}
    for method in results:
        histories = results[method]['exploitability']
        iterations = [h[0] for h in histories[0]]
        expl_by_iter = defaultdict(list)
        for hist in histories:
            for it, expl in hist:
                expl_by_iter[it].append(expl)
        mean_expl = [np.mean(expl_by_iter[it]) for it in iterations]
        std_expl = [np.std(expl_by_iter[it]) for it in iterations]
        aggregated[method] = {
            'iterations': iterations,
            'mean_exploitability': mean_expl,
            'std_exploitability': std_expl,
            'mean_time': np.mean(results[method]['time']),
            'std_time': np.std(results[method]['time']),
            'final_strategy': results[method]['final_strategy']
        }
    return aggregated


def find_convergence_iteration(iterations, exploitability, threshold=0.01):
    for it, expl in zip(iterations, exploitability):
        if expl < threshold:
            return it
    return None


def generate_exploitability_plot(aggregated, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'CFR': '#1f77b4', 'CFR+': '#2ca02c', 'FP': '#ff7f0e'}
    labels = {'CFR': 'Vanilla CFR', 'CFR+': 'CFR+', 'FP': 'Fictitious Play'}

    for method in ['CFR', 'CFR+', 'FP']:
        data = aggregated[method]
        iters = data['iterations']
        mean = data['mean_exploitability']
        std = data['std_exploitability']
        ax.semilogy(iters, mean, color=colors[method], label=labels[method], linewidth=2)
        ax.fill_between(iters, np.maximum(np.array(mean) - np.array(std), 1e-6),
                        np.array(mean) + np.array(std), color=colors[method], alpha=0.2)

    ax.axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='ε = 0.01')
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Exploitability (log scale)', fontsize=12)
    ax.set_title('Convergence to Nash Equilibrium in Kuhn Poker', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_strategy_plot(aggregated, save_path):
    nash = nash_equilibrium_strategy(alpha=0)
    info_sets = ['J', 'Q', 'K', 'Jp', 'Qp', 'Kp', 'Jb', 'Qb', 'Kb', 'Jpb', 'Qpb', 'Kpb']
    info_set_labels = ['J (P0)', 'Q (P0)', 'K (P0)', 'J|p (P1)', 'Q|p (P1)', 'K|p (P1)',
                       'J|b (P1)', 'Q|b (P1)', 'K|b (P1)', 'J|pb (P0)', 'Q|pb (P0)', 'K|pb (P0)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    methods = ['CFR', 'CFR+', 'FP']
    titles = ['Vanilla CFR', 'CFR+', 'Fictitious Play']

    for ax, method, title in zip(axes, methods, titles):
        strategy = aggregated[method]['final_strategy']
        nash_probs = [nash[is_][1] for is_ in info_sets]
        learned_probs = [strategy.get(is_, np.array([0.5, 0.5]))[1] for is_ in info_sets]

        x = np.arange(len(info_sets))
        width = 0.35
        ax.bar(x - width/2, nash_probs, width, label='Nash', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, learned_probs, width, label=title, color='#ff7f0e', alpha=0.8)
        ax.set_ylabel('P(bet/call)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(info_set_labels, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Learned Strategies vs Nash Equilibrium (after 2000 iterations)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_results_table(aggregated, save_path):
    checkpoints = [100, 500, 1000, 2000]
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & \multicolumn{4}{c}{Exploitability at Iteration} & Iter to $\varepsilon < 0.01$ & Time (s) \\",
        r"\cmidrule(lr){2-5}",
        r" & 100 & 500 & 1000 & 2000 & & \\",
        r"\midrule"
    ]

    for method, label in [('CFR', 'Vanilla CFR'), ('CFR+', 'CFR+'), ('FP', 'Fictitious Play')]:
        data = aggregated[method]
        iters = data['iterations']
        mean_expl = data['mean_exploitability']
        expl_vals = []
        for cp in checkpoints:
            idx = iters.index(cp) if cp in iters else -1
            expl_vals.append(f"{mean_expl[idx]:.4f}" if idx >= 0 else "--")
        conv = find_convergence_iteration(iters, mean_expl, 0.01)
        conv_str = str(conv) if conv else "$>$2000"
        time_str = f"{data['mean_time']:.2f} $\\pm$ {data['std_time']:.2f}"
        lines.append(f"{label} & {' & '.join(expl_vals)} & {conv_str} & {time_str} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("Kuhn Poker Equilibrium Computation")
    print("Chapter 5: RL in Games")
    print("=" * 60)

    # Verify Nash exploitability
    nash = nash_equilibrium_strategy(alpha=0)
    nash_expl = compute_exploitability(nash)
    print(f"\nNash equilibrium exploitability: {nash_expl:.6f}")
    print("(Should be ~0 for exact Nash equilibrium)")

    # Run experiments
    results = run_experiment(num_iterations=2000, num_seeds=10)
    aggregated = aggregate_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for method in ['CFR', 'CFR+', 'FP']:
        data = aggregated[method]
        final = data['mean_exploitability'][-1]
        conv = find_convergence_iteration(data['iterations'], data['mean_exploitability'], 0.01)
        print(f"\n{method}:")
        print(f"  Final exploitability: {final:.6f}")
        print(f"  Iterations to ε < 0.01: {conv if conv else '>2000'}")
        print(f"  Time: {data['mean_time']:.2f}s ± {data['std_time']:.2f}s")

    # Generate outputs
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)
    output_dir = "/Users/pranjal/Code/rl/ch05_rl_in_games/sims"
    generate_exploitability_plot(aggregated, f"{output_dir}/kuhn_poker_exploitability.png")
    generate_strategy_plot(aggregated, f"{output_dir}/kuhn_poker_strategies.png")
    generate_results_table(aggregated, f"{output_dir}/kuhn_poker_results.tex")
    print("\nDone!")


if __name__ == "__main__":
    main()
