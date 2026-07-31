"""Ch09 RLHF sim. BT-MLE vs Leximax Copeland subject to PO on the
6-candidate construction of Ge et al. (NeurIPS 2024). Reproduces Theorem
3.1 (BT-MLE fails PO and PMC) and Theorem 4.3 (LCPO satisfies both)."""

import argparse
import os
import sys
from itertools import combinations, permutations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE
from sims.sim_cache import (
    compute_or_load, add_component_args, parse_force_set,
)

apply_style()

SCRIPT_NAME = 'axiom_aware_aggregation'
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
OUT_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Environment: the 6-candidate construction (Theorem 3.1 of Ge et al. 2024)
# ---------------------------------------------------------------------------
# Candidates a, b, c are at feature positions x_a=(2,1), x_b=(1,1), x_c=(0,0).
# Candidates a', b', c' are nearby perturbations parametrized by eps, delta.
# Voter type 1 (fraction p) ranks a > a' > b > b' > c' > c (induced by theta=(1,1)).
# Voter type 2 (fraction 1-p) ranks c' > c > b' > b > a' > a (induced by theta=(-1,0)).
# Both voter types agree c' > c, so Pareto requires the output to rank c' above c.
# The majority ranking is type 1's ranking when p > 1/2 (so PMC asks for that).

CANDIDATES = ['a', 'b', 'c', 'ap', 'bp', 'cp']
K = len(CANDIDATES)

ENV_PARAMS = {
    # Small eps so that Theorem 3.1 of Ge et al. (2024) bites: at the BT-MLE
    # optimum, the linear reward family is forced to rank c above c' despite
    # both voter types preferring c' to c (a PO violation). At larger eps the
    # asymptotic minimum shifts and BT-MLE may accidentally satisfy PO.
    'eps': 0.01,
    # delta > 1 so type 1 also ranks c' above c (matching the paper's Pareto
    # dominance requirement at lines 177--179 of the docling extraction).
    'delta': 2.0,
    'p': 0.6,             # fraction of voter type 1 (majority)
    # Large beta so per-vote labels are nearly deterministic w.r.t. each
    # voter's strict ranking. Otherwise small-reward-gap pairs collapse to
    # near-50/50 sampling noise and the axiomatic distinction is lost.
    'beta': 50.0,
    'n_pairs_per_pair': None,  # set per sweep step
}


def feature_matrix(eps, delta):
    return np.array([
        [2.0, 1.0],          # a
        [1.0, 1.0],          # b
        [0.0, 0.0],          # c
        [2.0 - eps, 1.0],    # a'
        [1.0 - eps, 1.0],    # b'
        [-eps, delta * eps], # c'
    ])


VOTER_TYPES = {
    'type1': np.array([1.0, 1.0]),   # induces a > a' > b > b' > c' > c
    'type2': np.array([-1.0, 0.0]),  # induces c' > c > b' > b > a' > a
}


def voter_ranking(theta, X):
    """Return permutation of CANDIDATES ranked by descending r_theta(c) = <theta, x_c>."""
    rewards = X @ theta
    # break ties by candidate index (stable + deterministic)
    order = np.argsort(-rewards, kind='stable')
    return order


def all_majority_pairs(p):
    """Return dict {(i, j): P(i beats j in population)} for i < j.

    Computed from the two voter types and their population fractions.
    """
    X = feature_matrix(ENV_PARAMS['eps'], ENV_PARAMS['delta'])
    # type 1 (fraction p), type 2 (fraction 1-p)
    r1 = X @ VOTER_TYPES['type1']
    r2 = X @ VOTER_TYPES['type2']
    pair_probs = {}
    for i, j in combinations(range(K), 2):
        # P(i > j) = p * 1{r1[i] > r1[j]} + (1-p) * 1{r2[i] > r2[j]}
        prob_i_beats_j = (p * (r1[i] > r1[j])
                          + (1.0 - p) * (r2[i] > r2[j]))
        pair_probs[(i, j)] = prob_i_beats_j
    return pair_probs


def true_pairwise_majority(p):
    """Return dict {(i, j): +1 if pop-majority prefers i over j, -1 otherwise} for i<j."""
    pair_probs = all_majority_pairs(p)
    return {pair: (1 if prob > 0.5 else -1) for pair, prob in pair_probs.items()}


def pareto_dominance(p):
    """Return set of pairs (i, j) such that i Pareto-dominates j (every voter ranks i above j)."""
    X = feature_matrix(ENV_PARAMS['eps'], ENV_PARAMS['delta'])
    r1 = X @ VOTER_TYPES['type1']
    r2 = X @ VOTER_TYPES['type2']
    dominated = set()
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if r1[i] > r1[j] and r2[i] > r2[j]:
                dominated.add((i, j))
    return dominated


# ---------------------------------------------------------------------------
# Data generation: sample pairwise comparison labels
# ---------------------------------------------------------------------------

def sample_comparisons(n_per_pair, p, beta, rng):
    """Draw pairwise-comparison labels.

    For each ordered pair (i, j) with i < j: sample n_per_pair voters; each voter
    is type 1 with probability p, else type 2. The voter labels (i, j) by
    Bradley-Terry on their type's reward, P(i > j | voter) = sigmoid(beta*(r(i)-r(j))).
    Returns array of shape (K, K, n_per_pair) with values in {0, 1}: a[i,j,t] = 1
    means in trial t voter labelled "i > j". Only entries with i < j are populated
    (the i > j entries are the flipped complement).
    """
    X = feature_matrix(ENV_PARAMS['eps'], ENV_PARAMS['delta'])
    r1 = X @ VOTER_TYPES['type1']
    r2 = X @ VOTER_TYPES['type2']
    counts_i_beats_j = np.zeros((K, K), dtype=int)
    n_obs = np.zeros((K, K), dtype=int)
    for i, j in combinations(range(K), 2):
        types = rng.uniform(size=n_per_pair) < p  # True = type 1
        diff = np.where(types, r1[i] - r1[j], r2[i] - r2[j])
        prob = 1.0 / (1.0 + np.exp(-beta * diff))
        labels = rng.uniform(size=n_per_pair) < prob
        counts_i_beats_j[i, j] = labels.sum()
        n_obs[i, j] = n_per_pair
    return counts_i_beats_j, n_obs


# ---------------------------------------------------------------------------
# Method 1: Bradley-Terry MLE over the linear reward family r_theta(c)=<theta,x_c>
# ---------------------------------------------------------------------------

def bt_mle_linear(counts, n_obs, eps, delta):
    """Maximum-likelihood Bradley-Terry over linear-in-features reward.

    Parameters: theta in R^d (d=2). Reward of candidate c is r_c = <theta, x_c>.
    For each (i,j) pair we observed counts[i,j] wins for i and n_obs[i,j]-counts[i,j] for j.
    Negative log-likelihood is the standard logistic loss. Use L-BFGS-B.
    Output: ranking (descending r_c). Ties broken by candidate index.
    """
    X = feature_matrix(eps, delta)

    def nll(theta):
        rewards = X @ theta
        loss = 0.0
        for i, j in combinations(range(K), 2):
            n_ij = n_obs[i, j]
            if n_ij == 0:
                continue
            wins_i = counts[i, j]
            diff = rewards[i] - rewards[j]
            # NLL of binary outcome with prob sigma(diff)
            # log_p = -log(1 + exp(-diff)); log_1mp = -log(1 + exp(diff))
            log_p = -np.logaddexp(0.0, -diff)
            log_1mp = -np.logaddexp(0.0, diff)
            loss -= wins_i * log_p + (n_ij - wins_i) * log_1mp
        # mild L2 to keep optimizer well-behaved
        return loss + 1e-4 * np.sum(theta**2)

    res = minimize(nll, x0=np.zeros(2), method='L-BFGS-B')
    theta = res.x
    rewards = X @ theta
    ranking = np.argsort(-rewards, kind='stable')
    return ranking, theta


# ---------------------------------------------------------------------------
# Method 2: Leximax Copeland subject to Pareto Optimality
# ---------------------------------------------------------------------------

def feasibility_lp(constraints, X, slack=1e-3):
    """Check if there exists theta inducing the given ordered constraints.

    constraints: list of (a, b) pairs meaning r_theta(a) > r_theta(b).
    Equivalent LP: find theta s.t. <theta, x_a - x_b> >= slack for each (a, b).
    Returns (feasible: bool, theta: np.ndarray or None).
    """
    if not constraints:
        return True, np.zeros(X.shape[1])
    A_ub = []
    b_ub = []
    for a, b in constraints:
        A_ub.append(-(X[a] - X[b]))   # -<theta, x_a - x_b> <= -slack
        b_ub.append(-slack)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    # Minimize 0 subject to constraints, with bounded box on theta to avoid unboundedness.
    bounds = [(-100.0, 100.0)] * X.shape[1]
    res = linprog(c=np.zeros(X.shape[1]), A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method='highs')
    if res.success:
        return True, res.x
    return False, None


def leximax_copeland_po(majority_dirs, counts, n_obs, X, dominance):
    """LCPO from Ge et al. 2024 Section 4.

    majority_dirs: dict (i,j) -> +1 if empirical majority prefers i over j, -1 if j over i.
    counts, n_obs: pairwise sample counts (used for tiebreaks via margin magnitude).
    X: feature matrix.
    dominance: set of (i, j) pairs where i Pareto-dominates j (from oracle).
            All such pairs are added as hard constraints (the algorithm enforces PO).

    Returns ranking (list of candidate indices in descending order).
    """
    # Copeland score: |{ j : majority(i, j) = +1 }|
    copeland = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            sign = majority_dirs[pair] if i < j else -majority_dirs[pair]
            if sign > 0:
                copeland[i] += 1
    # Sorted-margins for leximax tiebreak (per-candidate vector of |margin| sorted desc)
    margins = {}
    for i in range(K):
        ms = []
        for j in range(K):
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            wins_i = counts[i, j] if i < j else (n_obs[j, i] - counts[j, i])
            n_ij = n_obs[i, j] if i < j else n_obs[j, i]
            margin = (wins_i / n_ij) - 0.5 if n_ij > 0 else 0.0
            ms.append(margin)
        margins[i] = sorted(ms, reverse=True)

    # Sequential rank-assignment with feasibility constraints.
    ranked = []
    remaining = list(range(K))
    while remaining:
        # Build hard constraints: partial-ranking + Pareto dominance.
        partial_constraints = []
        for idx in range(len(ranked) - 1):
            partial_constraints.append((ranked[idx], ranked[idx + 1]))
        if ranked:
            for j in remaining:
                partial_constraints.append((ranked[-1], j))  # last ranked > each remaining
        for (a, b) in dominance:
            if a in remaining and b in remaining:
                partial_constraints.append((a, b))
            elif a in remaining and b in ranked:
                # b already placed; nothing to enforce here
                pass
            elif a in ranked and b in remaining:
                # a already placed (necessarily above b in partial ordering if PO respected)
                pass
        # Find candidates that can be placed next
        candidates_feasible = []
        for c in remaining:
            extra = [(c, j) for j in remaining if j != c]
            feasible, _ = feasibility_lp(partial_constraints + extra, X)
            if feasible:
                candidates_feasible.append(c)
        if not candidates_feasible:
            # Should never happen: at least one candidate must be feasibly rankable next.
            candidates_feasible = remaining[:]
        # Pick highest Copeland score; tiebreak by leximax of sorted margin vectors
        best = max(
            candidates_feasible,
            key=lambda c: (copeland[c], tuple(margins[c]))
        )
        ranked.append(best)
        remaining.remove(best)
    return ranked


# ---------------------------------------------------------------------------
# Axiom evaluation
# ---------------------------------------------------------------------------

def evaluate_axioms(ranking, p):
    """Check PO and PMC on the output ranking.

    ranking: list/array of candidate indices in descending order.
    Returns dict with PO_violation (bool), PMC_violation (bool), worst_group_utility (float).
    """
    # PO violation: exists (i, j) such that i Pareto-dominates j but ranking places j above i
    dominance = pareto_dominance(p)
    pos = {c: r for r, c in enumerate(ranking)}
    po_violation = any(pos[i] > pos[j] for (i, j) in dominance)
    # PMC: a PMC ranking exists iff the majority relation is a total order on candidates
    # In our 2-type construction, type 1 ranking is the PMC ranking (when p > 1/2)
    # since type 1 has majority on every pair where the two types disagree only when
    # they share orientation. Compute it explicitly.
    X = feature_matrix(ENV_PARAMS['eps'], ENV_PARAMS['delta'])
    if p > 0.5:
        pmc_ranking = voter_ranking(VOTER_TYPES['type1'], X)
    else:
        pmc_ranking = voter_ranking(VOTER_TYPES['type2'], X)
    pmc_violation = not np.array_equal(np.asarray(ranking), pmc_ranking)
    # Worst-group utility: min over voter types of E[r_type(top candidate)]
    top = ranking[0]
    r1 = X @ VOTER_TYPES['type1']
    r2 = X @ VOTER_TYPES['type2']
    worst_group_utility = float(min(r1[top], r2[top]))
    return {
        'PO_violation': bool(po_violation),
        'PMC_violation': bool(pmc_violation),
        'worst_group_utility': worst_group_utility,
        'top': int(top),
        'ranking': [int(c) for c in ranking],
    }


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

N_PER_PAIR_SWEEP = [5, 10, 20, 50, 100, 500, 2000]
N_SEEDS = 30

SHARED_CONFIG = {
    **ENV_PARAMS,
    'CANDIDATES': CANDIDATES,
    'N_PER_PAIR_SWEEP': N_PER_PAIR_SWEEP,
    'N_SEEDS': N_SEEDS,
}

BT_CONFIG = {**SHARED_CONFIG, 'method': 'BT_MLE_linear'}
COPELAND_CONFIG = {**SHARED_CONFIG, 'method': 'Leximax_Copeland_PO'}


# ---------------------------------------------------------------------------
# Shared compute (oracle quantities)
# ---------------------------------------------------------------------------

def compute_shared(config):
    eps = config['eps']
    delta = config['delta']
    p = config['p']
    X = feature_matrix(eps, delta)
    r1 = X @ VOTER_TYPES['type1']
    r2 = X @ VOTER_TYPES['type2']
    dominance = pareto_dominance(p)
    true_pairwise = true_pairwise_majority(p)
    pmc_ranking = voter_ranking(
        VOTER_TYPES['type1'] if p > 0.5 else VOTER_TYPES['type2'], X
    )
    return {
        'X': X,
        'r1': r1,
        'r2': r2,
        'dominance': sorted(list(dominance)),
        'true_pairwise': {f'{i}_{j}': int(v) for (i, j), v in true_pairwise.items()},
        'pmc_ranking': [int(c) for c in pmc_ranking],
        'C_NAMES': CANDIDATES,
    }


# ---------------------------------------------------------------------------
# Per-method compute (the sweep)
# ---------------------------------------------------------------------------

def _run_one_method(config, shared, method_name):
    eps = config['eps']
    delta = config['delta']
    p = config['p']
    beta = config['beta']
    X = shared['X']
    dominance = set(tuple(pair) for pair in shared['dominance'])
    results = {n: {'po_violations': [], 'pmc_violations': [], 'worst_utility': [],
                   'top': [], 'rankings': []}
               for n in N_PER_PAIR_SWEEP}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n_per_pair in N_PER_PAIR_SWEEP:
            counts, n_obs = sample_comparisons(n_per_pair, p, beta, rng)
            # Estimated majority direction per pair
            est_majority = {}
            for i, j in combinations(range(K), 2):
                est_majority[(i, j)] = +1 if counts[i, j] > 0.5 * n_obs[i, j] else -1
            if method_name == 'BT_MLE_linear':
                ranking, _ = bt_mle_linear(counts, n_obs, eps, delta)
            elif method_name == 'Leximax_Copeland_PO':
                ranking = leximax_copeland_po(est_majority, counts, n_obs, X, dominance)
            else:
                raise ValueError(method_name)
            axioms = evaluate_axioms(ranking, p)
            results[n_per_pair]['po_violations'].append(int(axioms['PO_violation']))
            results[n_per_pair]['pmc_violations'].append(int(axioms['PMC_violation']))
            results[n_per_pair]['worst_utility'].append(axioms['worst_group_utility'])
            results[n_per_pair]['top'].append(axioms['top'])
            results[n_per_pair]['rankings'].append(axioms['ranking'])
    # Summary stats
    summary = {}
    for n in N_PER_PAIR_SWEEP:
        po = np.array(results[n]['po_violations'])
        pmc = np.array(results[n]['pmc_violations'])
        wu = np.array(results[n]['worst_utility'])
        summary[n] = {
            'po_mean': float(po.mean()),
            'po_se': float(po.std(ddof=1) / np.sqrt(len(po))),
            'pmc_mean': float(pmc.mean()),
            'pmc_se': float(pmc.std(ddof=1) / np.sqrt(len(pmc))),
            'wu_mean': float(wu.mean()),
            'wu_se': float(wu.std(ddof=1) / np.sqrt(len(wu))),
            'top_mode_idx': int(np.bincount(results[n]['top']).argmax()),
        }
    return {'summary': summary, 'raw': results}


def compute_bt(config, shared):
    return _run_one_method(config, shared, 'BT_MLE_linear')


def compute_copeland(config, shared):
    return _run_one_method(config, shared, 'Leximax_Copeland_PO')


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'shared', SHARED_CONFIG,
        compute_shared, SHARED_CONFIG, force=('shared' in force),
    )
    bt = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'BT_MLE', BT_CONFIG,
        compute_bt, BT_CONFIG, shared,
        force=('BT_MLE' in force or 'shared' in force),
    )
    copeland = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'Leximax_Copeland', COPELAND_CONFIG,
        compute_copeland, COPELAND_CONFIG, shared,
        force=('Leximax_Copeland' in force or 'shared' in force),
    )
    return {'shared': shared, 'BT_MLE': bt, 'Leximax_Copeland': copeland}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def generate_outputs(data):
    shared = data['shared']
    bt = data['BT_MLE']['summary']
    cp = data['Leximax_Copeland']['summary']

    Ns = np.array(N_PER_PAIR_SWEEP)
    bt_po = np.array([bt[n]['po_mean'] for n in Ns])
    bt_pmc = np.array([bt[n]['pmc_mean'] for n in Ns])
    cp_po = np.array([cp[n]['po_mean'] for n in Ns])
    cp_pmc = np.array([cp[n]['pmc_mean'] for n in Ns])
    bt_po_se = np.array([bt[n]['po_se'] for n in Ns])
    bt_pmc_se = np.array([bt[n]['pmc_se'] for n in Ns])
    cp_po_se = np.array([cp[n]['po_se'] for n in Ns])
    cp_pmc_se = np.array([cp[n]['pmc_se'] for n in Ns])

    bt_wu = np.array([bt[n]['wu_mean'] for n in Ns])
    cp_wu = np.array([cp[n]['wu_mean'] for n in Ns])
    bt_wu_se = np.array([bt[n]['wu_se'] for n in Ns])
    cp_wu_se = np.array([cp[n]['wu_se'] for n in Ns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    c_bt = COLORS['orange']
    c_cp = COLORS['blue']
    ax1.errorbar(Ns, bt_po, yerr=bt_po_se, color=c_bt, marker='o',
                 linestyle='-', label='BT-MLE: PO violation rate')
    ax1.errorbar(Ns, bt_pmc, yerr=bt_pmc_se, color=c_bt, marker='s',
                 linestyle='--', label='BT-MLE: PMC violation rate')
    ax1.errorbar(Ns, cp_po, yerr=cp_po_se, color=c_cp, marker='o',
                 linestyle='-', label='LCPO: PO violation rate')
    ax1.errorbar(Ns, cp_pmc, yerr=cp_pmc_se, color=c_cp, marker='s',
                 linestyle='--', label='LCPO: PMC violation rate')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Comparisons per pair $N$')
    ax1.set_ylabel('Violation rate (over 30 seeds)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='center right', fontsize=8)
    ax1.set_title('Axiom violations vs sample size')

    ax2.errorbar(Ns, bt_wu, yerr=bt_wu_se, color=c_bt, marker='o',
                 label='BT-MLE')
    ax2.errorbar(Ns, cp_wu, yerr=cp_wu_se, color=c_cp, marker='o',
                 label='LCPO')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'Comparisons per pair $N$')
    ax2.set_ylabel(r'Worst-group utility of top-ranked candidate')
    ax2.legend(loc='best')
    ax2.set_title('Worst-group utility of winner')

    png_path = os.path.join(OUT_DIR, f'{SCRIPT_NAME}.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure: {png_path}')

    # Table at the largest N
    N_table = Ns[-1]
    bt_row = bt[N_table]
    cp_row = cp[N_table]

    def _row(name, s):
        return (f"{name} & {s['po_mean']:.2f} ({s['po_se']:.2f}) & "
                f"{s['pmc_mean']:.2f} ({s['pmc_se']:.2f}) & "
                f"{s['wu_mean']:.3f} ({s['wu_se']:.3f}) & "
                f"{shared['C_NAMES'][s['top_mode_idx']]} \\\\")

    table_tex = (
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        f"Method & PO viol. & PMC viol. & Worst-group util. & Modal top \\\\\n"
        "\\hline\n"
        + _row("BT-MLE (linear)", bt_row) + "\n"
        + _row("Leximax Copeland (LCPO)", cp_row) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
    )
    tex_path = os.path.join(OUT_DIR, f'{SCRIPT_NAME}.tex')
    with open(tex_path, 'w') as f:
        f.write(table_tex)
    print(f'  Table: {tex_path}')

    # Stdout summary
    print()
    print(f"Construction: 6 candidates, eps={ENV_PARAMS['eps']}, delta={ENV_PARAMS['delta']}, "
          f"p={ENV_PARAMS['p']}, beta={ENV_PARAMS['beta']}")
    print(f"PMC (majority) ranking: {[shared['C_NAMES'][i] for i in shared['pmc_ranking']]}")
    print(f"Pareto-dominance pairs (i dominates j): "
          f"{[(shared['C_NAMES'][a], shared['C_NAMES'][b]) for (a, b) in shared['dominance']]}")
    print()
    header = f"{'N':>6} | {'BT_PO':>7} {'BT_PMC':>7} {'CP_PO':>7} {'CP_PMC':>7} | {'BT_wu':>7} {'CP_wu':>7}"
    print(header)
    print('-' * len(header))
    for n in N_PER_PAIR_SWEEP:
        print(f"{n:>6} | {bt[n]['po_mean']:>7.3f} {bt[n]['pmc_mean']:>7.3f} "
              f"{cp[n]['po_mean']:>7.3f} {cp[n]['pmc_mean']:>7.3f} | "
              f"{bt[n]['wu_mean']:>7.3f} {cp[n]['wu_mean']:>7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    if args.plots_only:
        # Cache hit on every component; do not force
        data = compute_data(force=set())
        generate_outputs(data)
        return
    force = parse_force_set(args)
    data = compute_data(force=force)
    if not args.data_only:
        generate_outputs(data)


if __name__ == '__main__':
    main()
