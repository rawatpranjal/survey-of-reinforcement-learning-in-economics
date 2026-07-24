# Causal Bandits on the parallel-bandit family.
# ch10c_adaptive_experiments, Causal Bandits and Adaptive Experimentation.
#
# Reproduces the central result of Lattimore, Lattimore & Reid (NeurIPS 2016,
# "Causal Bandits: Learning Good Interventions via Causal Inference"): when
# arms correspond to interventions on a known causal graph, exploiting the
# graph structure replaces the dependence on the number of arms N with a
# graph-derived hardness quantity m(q) <= N, yielding simple-regret
# O(sqrt(m(q)/T)) versus the Omega(sqrt(N/T)) floor of best-arm-identification
# algorithms that ignore the graph. Also exercises the Bareinboim-Forney-Pearl
# (NeurIPS 2015) "greedy casino" instance with three Thompson-family
# baselines (vanilla TS, context-conditional TS, and the full TS_C of
# Bareinboim et al. with consistency-axiom seeding and RDC bias weighting),
# demonstrating that causal posteriors achieve bounded cumulative regret
# while standard Thompson sampling fails (linear cumulative regret).
#
# Setting:
#   N independent binary parents X_1, ..., X_N of a binary reward Y.
#   For each i, P(X_i = 1) = q_i (in the "observational" phase, after do()).
#   The reward function takes the form
#       E[Y | X_1=x_1, ..., X_N=x_N] = w' x + offset
#   with w supported on a small subset of indices, so the optimal arm is
#   typically a do(X_i = j) intervention on a "high-leverage" coordinate i.
#   Action set A = {do()} U {do(X_i = j) : i in [N], j in {0,1}}, so |A| = 2N+1.
#   After each action, the agent observes the reward AND the realized
#   non-intervened parents (Lattimore et al. 2016, Section 2).
#
# Four algorithms compared:
#   1. Successive Reject (Audibert-Bubeck 2010): graph-blind best-arm
#      identification. Achieves O(sqrt(N/T)) simple regret.
#   2. Lattimore Algorithm 1 (parallel-bandit algorithm): allocates half the
#      budget to do() to estimate the propensities q_i, then concentrates the
#      remaining budget on the "unbalanced" arms whose direct-observation
#      probability is below 1/tau. Achieves O(sqrt(m(q)/T)) simple regret.
#   3. Context-conditional Thompson sampling (CCTS): a minimal causal
#      baseline on the MABUC "greedy casino" instance. Maintains a Beta
#      posterior indexed by (intuition x, arm a) and uses straight argmax
#      over posterior samples. Achieves bounded regret on this instance
#      because (x, a) cells are independently learnable.
#   4. Full Causal Thompson Sampling (TS_C) of Bareinboim, Forney & Pearl
#      (NeurIPS 2015) Algorithm 1. Augments the (x, a) Beta posterior with
#      (i) counterfactual seeding of the cell no observational sample visits,
#      taking the effect of treatment on the untreated from the identity
#      P(y|do(a)) = P(X=a) P(y|X=a) + P(X!=a) E[Y_a|X!=a] applied to a
#      separate log of randomized play that records the arm but not the
#      intuition, and (ii) RDC bias weighting: the arm disfavoured by the
#      within-context contrast between Q1 = E[Y_{X=x'}|X=x] and
#      Q2 = P(y|X=x) is multiplied by 1 - |Q1 - Q2|, clamped to [0.01, 1].
#      Both components are switchable so the run prices each separately.
#
# Outputs:
#   causal_bandit_combined.png     -- three panels: regret vs hardness m(q),
#                                     regret vs horizon T, greedy-casino MABUC
#   causal_bandit_results.tex      -- simple-regret table at T=400 over the m grid
#   causal_bandit_mabuc_results.tex-- MABUC component factorial, cumulative regret
#   causal_bandit_parallel_stdout.txt -- numerical log (run via shell redirect)

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE
from sims.sim_cache import (
    compute_or_load,
    add_component_args,
    parse_force_set,
)

apply_style()
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "causal_bandit_parallel"

# Parallel-bandit DGP
N_PARENTS = 50  # number of binary parents X_i (so |A| = 2N+1 arms)
T_HORIZON = 400  # default horizon
M_GRID = [2, 8, 24, 48]  # graph-derived hardness values (m(q) levels)
N_SEEDS_REGRET = 2000  # Monte Carlo replications for simple-regret panels
# Horizon panel. Two things had to change for it to measure a rate. The gap now
# shrinks with the budget (see eps_at_T), because at a fixed gap the best arm is
# identified outright and regret collapses to exactly zero rather than decaying
# at any polynomial rate. And m(q) sits at 8 rather than 48: with N = 50 parents
# the graph buys nothing at m(q) = 48, since m(q)/N is then close to one.
T_GRID = [400, 800, 1600, 3200, 6400]  # horizon grid for vs-T panel
M_AT_T_PANEL = 8  # fix m(q) for the vs-T panel: small enough against
# N = 50 that the graph advantage is real, large enough that the estimate
# of the rare side is not trivially exact.
T_SWEEP_SEEDS = 1000  # replications for the horizon panel
GAP_SCALE = 1.0  # eps_T = GAP_SCALE * sqrt(m / T) on the horizon panel


def eps_at_T(m, T):
    """Minimax-frontier gap for the horizon panel; capped to keep payoffs in [0,1]."""
    return float(min(0.45, GAP_SCALE * np.sqrt(m / T)))


EPS_REWARD = 0.30  # reward gap epsilon: best arm pays mu* = 0.5+eps,
# all others pay 0.5

# Bareinboim "greedy casino" parameters
MABUC_T = 1000  # horizon for cumulative-regret panel
MABUC_SEEDS = 500  # Monte Carlo replications
MABUC_N_OBS = 200  # observational log size, records (intuition, reward)
MABUC_N_EXP = 200  # interventional log size, records (arm, reward) only
MABUC_NOBS_SWEEP = [25, 50, 100, 200, 400]  # log-size sweep for the ETT panel
MABUC_SWEEP_SEEDS = 300

# Full factorial over the two TS_C components, so each one is priced separately.
# (obs_diagonal, no RDC) is CCTS exactly; it runs through the same code path.
MABUC_VARIANTS = {
    "CCTS (obs seed, no RDC)": ("obs_diagonal", False),
    "obs seed + RDC": ("obs_diagonal", True),
    "consistency-copy seed, no RDC": ("consistency_copy", False),
    "consistency-copy seed + RDC": ("consistency_copy", True),
    "ETT seed, no RDC": ("ett", False),
    "TS_C (ETT seed + RDC)": ("ett", True),
}
# Greedy-casino payoffs (Bareinboim et al. 2015, Table 1, "asterisked" values):
#   Player intuition X corresponds to a confounder configuration.
#   P(Y=1 | X=x, do(X=x)) = high  -- following intuition wins more
#   P(Y=1 | X=x, do(X=1-x)) = low -- going against intuition loses
GREEDY_CASINO_PAYOFFS = np.array(
    [
        # rows: agent's natural intuition x in {0, 1}
        # cols: arm pulled a in {0, 1}
        # entry: P(Y=1 | X=x, do(X=a))
        [0.10, 0.50],  # x=0 (drunk + non-blinking): a=0 loses, a=1 wins
        [0.50, 0.10],  # x=1 (drunk + blinking):     a=0 wins, a=1 loses
    ]
)

SHARED_CONFIG = {
    "N_PARENTS": N_PARENTS,
    "T_HORIZON": T_HORIZON,
    "M_GRID": M_GRID,
    "N_SEEDS_REGRET": N_SEEDS_REGRET,
    "T_GRID": T_GRID,
    "M_AT_T_PANEL": M_AT_T_PANEL,
    "T_SWEEP_SEEDS": T_SWEEP_SEEDS,
    "GAP_SCALE": GAP_SCALE,
    "EPS_REWARD": EPS_REWARD,
    "MABUC_T": MABUC_T,
    "MABUC_SEEDS": MABUC_SEEDS,
    "MABUC_N_OBS": MABUC_N_OBS,
    "MABUC_N_EXP": MABUC_N_EXP,
    "MABUC_NOBS_SWEEP": MABUC_NOBS_SWEEP,
    "MABUC_SWEEP_SEEDS": MABUC_SWEEP_SEEDS,
    "GREEDY_CASINO_PAYOFFS": GREEDY_CASINO_PAYOFFS.tolist(),
}

# v3 marker: the hardness construction now scales the extreme propensity with
# m_target so that m(q) equals its label (a fixed 0.05 saturated at 20), and
# phase 2 of Lattimore Algorithm 1 now pulls the rare side of each unbalanced
# coordinate rather than the common side.
_ALG_VERSION = {
    "hardness_construction": "extreme_q_scaled_inv_2m_v3",
    "phase2_rare_side": "j_rare_fixed_v3",
}
REGRET_VS_M_CONFIG = {**SHARED_CONFIG, "experiment": "regret_vs_m", **_ALG_VERSION}
REGRET_VS_T_CONFIG = {**SHARED_CONFIG, "experiment": "regret_vs_T", **_ALG_VERSION}
# v3 marker: TS_C now recovers the counter-intuitive cell from the ETT identity
# using a separate interventional log, and the RDC contrast is within-context as
# in the paper rather than a cross-context variance. The run is a full factorial
# over both components plus a log-size sweep.
MABUC_CONFIG = {
    **SHARED_CONFIG,
    "experiment": "mabuc",
    "algos_version": "v3_ett_identified_tsc",
    "variants": sorted(MABUC_VARIANTS),
}


# ---------------------------------------------------------------------------
# Parallel-bandit DGP and helpers
# ---------------------------------------------------------------------------
def make_q_with_hardness(m_target, N, rng):
    """Construct propensities q = (q_1, ..., q_N) in (0,1) so that m(q) = m_target.

    Recall (Lattimore et al. 2016, eq. for m(q)):
        I_tau := {i : min(q_i, 1 - q_i) < 1/tau}
        m(q)  = min_tau in [2, N] max(tau, |I_tau|)

    Set exactly K = m_target coordinates to an extreme value e and leave the
    rest balanced at 0.5. A balanced coordinate has min(q_i, 1-q_i) = 0.5 and so
    never enters I_tau for tau >= 2. An extreme coordinate enters I_tau exactly
    when e < 1/tau, that is when tau < 1/e. Hence

        |I_tau| = K   for tau < 1/e,        |I_tau| = 0   for tau >= 1/e,

    and m(q) = min( max(tau, K) over tau < 1/e,  tau over tau >= 1/e )
             = min( K,  ceil(1/e) )   whenever K < 1/e.

    So e must satisfy e < 1/K, otherwise the hardness saturates at ceil(1/e)
    and the constructed instance is easier than its label. A fixed e = 0.05
    saturates at 20, which silently collapsed the m = 24 and m = 48 cells of
    M_GRID onto the same true hardness m(q) = 20. Scaling the extreme value
    with the target, e = 1/(2 m_target), keeps 1/e = 2 m_target > m_target for
    every m_target, so m(q) = m_target holds exactly across the grid.
    """
    q = np.full(N, 0.5)
    extreme = 1.0 / (2.0 * m_target)  # strictly below 1/m_target; see docstring
    if m_target < N:
        # Place extreme values on a random subset of arms (independent of best arm)
        extreme_idx = rng.permutation(N)[:m_target]
        q[extreme_idx] = extreme
    else:
        # m_target == N: all arms extreme
        q[:] = extreme
    return q


def hardness_m(q):
    """m(q) = min_{tau in [2, N]} max(tau, |I_tau|); the exact Lattimore hardness."""
    N = len(q)
    spread = np.minimum(q, 1.0 - q)
    return min(max(tau, int(np.sum(spread < 1.0 / tau))) for tau in range(2, N + 1))


def sample_parents(q, rng):
    """One realization of (X_1, ..., X_N) under the observational distribution."""
    return (rng.uniform(size=len(q)) < q).astype(int)


def reward_under_do(
    parents, intervened_idx, intervened_val, w, baseline_q, rng, eps=None
):
    """Sample the reward Y given an action.

    The reward model: E[Y | X] = sigmoid(w' X + b) but we use a simpler
    Bernoulli model with explicit best-arm gap: when the intervened parent
    is the *one* high-leverage coordinate (index 0), the arm value is
    0.5 + EPS_REWARD; otherwise 0.5. This is a clean best-arm-identification
    setting with gap EPS_REWARD.

    Arguments:
      parents : (N,) sampled non-intervened parents (the intervened entry
                will be overwritten below; the others are observed)
      intervened_idx : the index i of the do(X_i = j) action; -1 means do()
      intervened_val : the value j in the intervention (0 or 1); ignored if i=-1
      w     : the optimal arm index (single high-leverage coordinate)
      baseline_q : (N,) the q vector (used only for sampling do(); see below)
      rng : RNG

    Returns:
      y : Bernoulli reward
      x_obs : (N,) realized parent vector after the intervention
    """
    eps = EPS_REWARD if eps is None else eps
    x = parents.copy()
    if intervened_idx >= 0:
        x[intervened_idx] = intervened_val

    # Reward depends only on the high-leverage coordinate (index `w`).
    if x[w] == 1:
        p = 0.5 + eps
    else:
        p = 0.5 - eps
    y = int(rng.uniform() < p)
    return y, x


def best_arm(q, w):
    """Return the (intervened_idx, intervened_val) tuple for the optimal arm.

    Optimal arm is do(X_w = 1), which forces the high-leverage coordinate to 1,
    yielding expected reward 0.5 + EPS_REWARD.
    """
    return (w, 1)


# ---------------------------------------------------------------------------
# Algorithm 1: Successive Reject (graph-blind baseline)
# ---------------------------------------------------------------------------
def successive_reject(q, w, T, rng, eps=None):
    """Audibert-Bubeck 2010 Successive Reject for best-arm identification.

    Treats the 2N+1 arms as opaque (ignores post-action observations of
    non-intervened parents). Returns the index of the recommended arm and
    its expected reward, used to compute simple regret.

    The action set has 2N + 1 arms:
      arm 0           : do()
      arm 2*i + 1     : do(X_i = 0)  for i in [N]
      arm 2*i + 2     : do(X_i = 1)  for i in [N]
    """
    N = len(q)
    K = 2 * N + 1

    # Standard successive reject schedule
    # n_k = floor((T - K) / (logbar(K) * (K - k + 1))) for k = 1, ..., K-1
    log_bar = 0.5 + sum(1.0 / i for i in range(2, K + 1))
    arms = list(range(K))
    sums = np.zeros(K)
    counts = np.zeros(K, dtype=int)
    used = 0
    for k in range(1, K):
        nk = max(1, int(np.floor((T - K) / (log_bar * (K - k + 1)))))
        if used + nk * len(arms) > T:
            nk = max(1, (T - used) // max(1, len(arms)))
        for a in arms:
            for _ in range(nk):
                y = pull_arm(a, q, w, rng, eps)
                sums[a] += y
                counts[a] += 1
                used += 1
                if used >= T:
                    break
            if used >= T:
                break
        if used >= T:
            break
        # Drop the worst-performing arm
        means = np.where(counts > 0, sums / np.maximum(counts, 1), -np.inf)
        active_means = [(means[a], a) for a in arms]
        worst = min(active_means)[1]
        arms = [a for a in arms if a != worst]
        if len(arms) == 1:
            break

    # Pull remaining budget uniformly over remaining arms
    while used < T and arms:
        for a in arms:
            if used >= T:
                break
            y = pull_arm(a, q, w, rng, eps)
            sums[a] += y
            counts[a] += 1
            used += 1

    means = np.where(counts > 0, sums / np.maximum(counts, 1), -np.inf)
    rec = int(np.argmax(means))
    return rec, arm_expected_reward(rec, w)


def pull_arm(arm_id, q, w, rng, eps=None):
    """Pull one of the 2N+1 arms; return realized reward."""
    if arm_id == 0:
        # do() : draw from observational distribution
        parents = sample_parents(q, rng)
        intervened_idx, intervened_val = -1, 0
    else:
        i = (arm_id - 1) // 2
        j = (arm_id - 1) % 2
        parents = sample_parents(q, rng)
        intervened_idx, intervened_val = i, j
    y, _ = reward_under_do(parents, intervened_idx, intervened_val, w, q, rng, eps)
    return y


def arm_expected_reward(arm_id, w):
    """Expected reward for a given arm (the true mu of that arm).

    arm 0 = do(): expected reward = 0.5 + EPS_REWARD * (2*q[w] - 1), but we
    approximate as 0.5 (since q[w] in {0.5, 0.05} both give expected ~0.5).
    For the "extreme" propensity case q[w]=0.05, do() yields mostly X_w=0 so
    expected reward < 0.5. For q[w]=0.5, do() yields exactly 0.5.

    Optimal arm = do(X_w = 1): expected reward = 0.5 + EPS_REWARD.
    Other arms = some mix; suboptimal-arm gap is at least 2*EPS_REWARD when
    the optimal arm flips X_w and the alternative does not.
    """
    # Optimal arm index in the (2N+1) layout
    opt_arm = 2 * w + 2  # do(X_w = 1)
    if arm_id == opt_arm:
        return 0.5 + EPS_REWARD
    # do(X_w = 0): worst
    if arm_id == 2 * w + 1:
        return 0.5 - EPS_REWARD
    # Other arms: don't touch X_w, so X_w is sampled from q; expected reward
    # is q[w] * (0.5 + EPS_REWARD) + (1 - q[w]) * (0.5 - EPS_REWARD)
    #              = 0.5 + EPS_REWARD * (2 * q[w] - 1)
    # We treat this as 0.5 minus a small amount under typical q (extreme = 0.05).
    return 0.5  # close enough; computed below per-q


def arm_expected_reward_exact(arm_id, q, w, eps=None):
    """Exact expected reward of arm_id under propensities q and optimal coord w."""
    eps = EPS_REWARD if eps is None else eps
    if arm_id == 2 * w + 2:
        return 0.5 + eps
    if arm_id == 2 * w + 1:
        return 0.5 - eps
    # Any other arm: X_w not intervened, so X_w ~ Bernoulli(q[w])
    return q[w] * (0.5 + eps) + (1 - q[w]) * (0.5 - eps)


# ---------------------------------------------------------------------------
# Algorithm 2: Lattimore Algorithm 1 (parallel-bandit causal algorithm)
# ---------------------------------------------------------------------------
def lattimore_alg1(q_true, w, T, rng, eps=None):
    """Lattimore, Lattimore & Reid (2016) Algorithm 1 for parallel bandits.

    The algorithm:
      1. Spend T/2 rounds on do(): observe (X_1, ..., X_N, Y) tuples. Use these
         to estimate (i) the propensities q_i and (ii) the conditional means
         E[Y | X_i = 1] and E[Y | X_i = 0] for each i, which equal the
         post-intervention values P(Y | do(X_i = j)) because each X_i has a
         direct causal arrow to Y in the parallel graph and no other parents
         confound the (X_i, Y) edge.
      2. Identify the "unbalanced" arms whose direct-observation probability
         is below 1/tau for the optimal tau, and spend the remaining T/2
         rounds on these arms to refine their estimates.
      3. Recommend the arm with the highest combined estimate.
    """
    N = len(q_true)
    K = 2 * N + 1
    T_obs = T // 2

    # Phase 1: pull do() T_obs times. Collect (X, Y) tuples.
    X_obs = np.zeros((T_obs, N), dtype=int)
    Y_obs = np.zeros(T_obs)
    for t in range(T_obs):
        parents = sample_parents(q_true, rng)
        y, _ = reward_under_do(parents, -1, 0, w, q_true, rng, eps)
        X_obs[t] = parents
        Y_obs[t] = y

    # Estimate q_i from observational phase
    q_hat = X_obs.mean(axis=0).clip(0.01, 0.99)

    # Estimate P(Y | X_i = j) from observational phase (= P(Y | do(X_i = j))
    # in the parallel-bandit graph, where no other parents confound X_i -> Y).
    # Note: this is the *unbiased causal estimate* used for "obs" arms.
    p_y_given = np.zeros((N, 2))  # p_y_given[i, j] = E[Y | X_i = j]
    for i in range(N):
        for j in (0, 1):
            mask = X_obs[:, i] == j
            if mask.sum() > 0:
                p_y_given[i, j] = Y_obs[mask].mean()
            else:
                p_y_given[i, j] = 0.5

    # Estimate do() expected reward as the mean of Y_obs (since do() is exactly
    # the observational distribution).
    p_do_empty = Y_obs.mean()

    # Phase 2: identify unbalanced arms and allocate the remaining T/2 to them.
    # tau* minimizes max(tau, |I_tau|) where I_tau = {i : min(q_i,1-q_i) < 1/tau}.
    tau_range = range(2, N + 1)
    best_tau = N
    best_m = N
    for tau in tau_range:
        I_tau = [i for i in range(N) if min(q_hat[i], 1 - q_hat[i]) < 1.0 / tau]
        m = max(tau, len(I_tau))
        if m < best_m:
            best_m = m
            best_tau = tau

    # Unbalanced *arms*: for each unbalanced parent, the arm do(X_i = j) where
    # j is the "low-probability" side.
    # j_rare is the value of X_i that is SELDOM observed, so q_hat[i] = P(X_i = 1)
    # below 0.5 means X_i = 1 is the rare side. That arm is the one whose
    # observational estimate is starved and therefore the one phase 2 must pull.
    unbalanced_arms = []
    for i in range(N):
        if min(q_hat[i], 1 - q_hat[i]) < 1.0 / best_tau:
            j_rare = 1 if q_hat[i] < 0.5 else 0
            unbalanced_arms.append(
                2 * i + 1 + j_rare
            )  # the arm index for do(X_i = j_rare)
    if not unbalanced_arms:
        unbalanced_arms = list(range(1, K))  # fallback: all do() arms

    # Spend the remaining T - T_obs rounds uniformly on the unbalanced arms.
    T_remain = T - T_obs
    pulls_per_arm = max(1, T_remain // len(unbalanced_arms))
    arm_sums = {a: 0.0 for a in unbalanced_arms}
    arm_counts = {a: 0 for a in unbalanced_arms}
    for a in unbalanced_arms:
        for _ in range(pulls_per_arm):
            y = pull_arm(a, q_true, w, rng, eps)
            arm_sums[a] += y
            arm_counts[a] += 1

    # Compose final estimates for all K arms.
    arm_estimate = np.full(K, -np.inf)
    arm_estimate[0] = p_do_empty
    for i in range(N):
        for j in (0, 1):
            arm_id = 2 * i + 1 + j
            # If the arm was directly pulled in phase 2, use that estimate.
            if arm_id in arm_counts and arm_counts[arm_id] > 0:
                arm_estimate[arm_id] = arm_sums[arm_id] / arm_counts[arm_id]
            else:
                # Use the observational conditional mean (causally valid for
                # parallel bandit).
                arm_estimate[arm_id] = p_y_given[i, j]

    rec = int(np.argmax(arm_estimate))
    # Diagnostic: did phase 2 actually pull the optimal arm do(X_w = 1)? If this
    # rate is zero the algorithm is running purely on phase-1 observational
    # estimates for the arm that matters, which is the failure mode that an
    # inverted j_rare produces and that aggregate regret alone will not reveal.
    opt_pulled = arm_counts.get(2 * w + 2, 0) > 0
    return rec, arm_estimate[rec], opt_pulled


# ---------------------------------------------------------------------------
# Algorithm 3: Context-conditional Thompson Sampling (CCTS, minimal baseline)
# and
# Algorithm 4: Full Causal Thompson Sampling (TS_C of Bareinboim et al. 2015,
#   with consistency-axiom seeding and RDC bias weighting)
# ---------------------------------------------------------------------------
def mabuc_dgp(intuition, rng):
    """Sample the reward Y under the greedy-casino MABUC instance.

    intuition x in {0, 1} encodes the (D, B) configuration:
      x=0 -> drunk + non-blinking (would naturally play machine M_1)
      x=1 -> drunk + blinking     (would naturally play machine M_0)
    The agent then chooses arm a in {0, 1}; P(Y=1) is the (x, a) entry of
    GREEDY_CASINO_PAYOFFS.
    """
    return intuition  # placeholder; reward is sampled in run loop


def context_conditional_thompson_sampling(T, rng, observational_data=None):
    """Context-conditional Thompson sampling (CCTS) on the MABUC greedy-casino instance.

    This is already the causal solution to MABUC: the insight that converts the
    greedy casino from hopeless to easy is conditioning on intent, which the
    (x, a) posterior does. What it does not do is seed the counter-intuitive
    cell from the identified counterfactual, so it must discover that cell by
    pulling it. Its regret is therefore bounded but pays a burn-in, and that
    burn-in is the only quantity the extra TS_C machinery can win back.

    It also cannot use the interventional marginal: those samples record the arm
    and the reward but not the intuition, so a context-conditional learner has
    no cell to put them in. Recovering that information needs the ETT identity,
    which is exactly what TS_C adds.

    Algorithm:
      1. Seed Beta posteriors from observational data along the on-intuition
         diagonal (a = x), which consistency identifies, leaving the
         counter-intuitive cell at Beta(1,1).
      2. At each time, observe the agent's natural intuition x; sample a
         reward estimate for each arm from its (x, a)-conditional Beta
         posterior; choose the arm with the higher sample via straight
         argmax (no RDC bias multiplier).
      3. Observe reward; update the Beta posterior conditioned on (x, a).
    """
    alpha, beta = seed_posteriors("obs_diagonal", observational_data, None)

    cum_regret = np.zeros(T)
    cum_reg = 0.0
    optimal_payoff = GREEDY_CASINO_PAYOFFS.max(axis=1)  # per-intuition optimum
    for t in range(T):
        x = rng.integers(0, 2)  # agent's natural intuition
        # Sample from each arm's posterior conditional on x
        theta0 = rng.beta(alpha[x, 0], beta[x, 0])
        theta1 = rng.beta(alpha[x, 1], beta[x, 1])
        a = 0 if theta0 >= theta1 else 1
        # Sample reward
        p = GREEDY_CASINO_PAYOFFS[x, a]
        y = int(rng.uniform() < p)
        # Update posterior
        alpha[x, a] += y
        beta[x, a] += 1 - y
        # Track regret
        cum_reg += optimal_payoff[x] - p
        cum_regret[t] = cum_reg

    return cum_regret


# Pseudo-count fraction for the superseded consistency-copy seeding, retained
# only so the ablation can price what it cost. See seed_posteriors.
CONSISTENCY_OFF_INTUITION_WEIGHT = 0.5

# Floor/ceiling for the RDC bias multiplier w = 1 - |Q1 - Q2|.
# w in [0.01, 1] avoids numerical degeneracy when Q1 and Q2 collide.
RDC_WEIGHT_FLOOR = 0.01
RDC_WEIGHT_CEILING = 1.0


def ett_from_marginals(p_do_a, p_obs_a, p_x_a):
    """Effect of treatment on the untreated, E[Y_a | X != a], by Bareinboim's identity.

    The interventional marginal decomposes over the unobserved intuition:

        P(y | do(a)) = P(X=a) E[Y_a | X=a] + P(X!=a) E[Y_a | X!=a],

    and consistency (Pearl 2009 sec 3.6) gives E[Y_a | X=a] = P(y | X=a), which
    the observational log identifies. Solving for the remaining term recovers
    the counterfactual payoff of playing a for an agent whose intuition said
    otherwise. In the greedy casino this returns exactly 0.50, the true value of
    the counter-intuitive arm, from two marginals neither of which is 0.50.

    Both inputs are arm-a quantities; p_x_a is P(X = a).
    """
    denom = max(1.0 - p_x_a, 1e-9)
    return float(np.clip((p_do_a - p_x_a * p_obs_a) / denom, 0.0, 1.0))


def seed_posteriors(mode, observational_data, experimental_data):
    """Build the Beta(alpha, beta) prior over the four (intuition, arm) cells.

    Every Thompson variant on the greedy casino differs only here, so the
    ablation is exactly a choice of `mode`:

      'obs_diagonal'  seed only the on-intuition cell (a = x) that the log
                      actually observed, leaving the counter-intuitive cell at
                      Beta(1,1). This is CCTS.
      'ett'           additionally seed the counter-intuitive cell from the
                      identified ETT, weighted by the experimental sample size
                      backing it. This is the paper's TS_C.
      'consistency_copy'  the superseded heuristic: copy the on-intuition
                      observational outcome into the counter-intuitive cell at
                      fractional weight. Kept only for the ablation.

    The observational log records (x, y) with the agent following intuition.
    The experimental log records (a, y) only, with the arm drawn at random and
    the intuition NOT recorded, which is what makes the ETT identity necessary:
    a context-conditional learner cannot attribute those samples to a cell.
    """
    alpha = np.ones((2, 2))  # alpha[x, a]
    beta = np.ones((2, 2))
    if not observational_data:
        return alpha, beta

    for x_obs, y_obs in observational_data:
        alpha[x_obs, x_obs] += y_obs
        beta[x_obs, x_obs] += 1 - y_obs

    if mode == "obs_diagonal":
        return alpha, beta

    if mode == "consistency_copy":
        c = CONSISTENCY_OFF_INTUITION_WEIGHT
        for x_obs, y_obs in observational_data:
            a_off = 1 - x_obs
            alpha[x_obs, a_off] += c * y_obs
            beta[x_obs, a_off] += c * (1 - y_obs)
        return alpha, beta

    if mode == "ett":
        for a, ett, n_eff in identified_ett_per_arm(
            observational_data, experimental_data
        ):
            # The seed lands in the cell (x = 1-a, arm = a): the counter-intuitive
            # play. Its confidence is the experimental evidence supporting it.
            alpha[1 - a, a] += ett * n_eff
            beta[1 - a, a] += (1.0 - ett) * n_eff
        return alpha, beta

    raise ValueError(f"unknown seeding mode: {mode}")


def identified_ett_per_arm(observational_data, experimental_data):
    """Yield (arm, ETT, supporting experimental count) for each arm.

    Single source for the identified counterfactual, used both to seed the
    posterior and to form the RDC contrast, so the two cannot drift apart.

    P(X = a) is read off the observational log. The decomposition needs the
    intuition distribution of the interventional population, and the two agree
    here because both logs draw the intuition from the same law; only the arm
    assignment differs between them. A design in which the two populations had
    different intuition distributions would need P(X = a) measured on the
    interventional one, and the substitution would fail silently.
    """
    if not observational_data or not experimental_data:
        return
    obs = np.asarray(observational_data)
    exp = np.asarray(experimental_data)
    p_x = np.array([np.mean(obs[:, 0] == a) for a in (0, 1)])
    for a in (0, 1):
        on = obs[:, 0] == a  # observations whose intuition was a
        pulled = exp[:, 0] == a  # experimental rounds that played a
        if on.sum() == 0 or pulled.sum() == 0:
            continue
        p_obs_a = obs[on, 1].mean()  # P(y | X = a)
        p_do_a = exp[pulled, 1].mean()  # P(y | do(a))
        yield a, ett_from_marginals(p_do_a, p_obs_a, p_x[a]), int(pulled.sum())


def causal_thompson_sampling_tsc(
    T,
    rng,
    observational_data=None,
    experimental_data=None,
    seed_mode="ett",
    use_rdc=True,
):
    """Causal Thompson Sampling TS_C (Bareinboim, Forney & Pearl 2015, Alg. 1).

    Two components separate TS_C from a plain context-conditional posterior, and
    `seed_mode` / `use_rdc` switch each off so the ablation can price them.

    (i) Counterfactual seeding, Algorithm 1 line 2. The counter-intuitive cell
        is unreachable observationally, so its prior comes from the ETT
        identified by combining the observational and interventional marginals
        (see ett_from_marginals). Setting seed_mode='obs_diagonal' removes it
        and recovers CCTS; 'consistency_copy' substitutes the superseded
        heuristic that copied the on-intuition outcome across.

    (ii) RDC bias weighting, Algorithm 1 lines 5-9. The paper contrasts
         Q1 = E[Y_{X=x'} | X=x], the payoff of defying intuition, against
         Q2 = P(y | X=x), the payoff of following it. Both are conditioned on
         the SAME intuition x, so the contrast measures which way the agent
         should lean in the context it is actually in. The arm the contrast
         disfavors is multiplied by 1 - |Q1 - Q2|; the favored arm keeps its
         full posterior sample. A confident disagreement therefore suppresses
         the losing arm hard, while a marginal one barely moves the argmax.
    """
    alpha, beta = seed_posteriors(seed_mode, observational_data, experimental_data)

    # Q2[x]: payoff of following intuition at x, identified observationally.
    # Q1[x]: payoff of defying it, the ETT. Both are held per context, which is
    # what makes the RDC contrast a within-context comparison.
    Q_sum = np.zeros((2, 2))
    Q_n = np.zeros((2, 2))
    if observational_data:
        for x_obs, y_obs in observational_data:
            Q_sum[x_obs, x_obs] += y_obs
            Q_n[x_obs, x_obs] += 1
    for a, ett, n_eff in identified_ett_per_arm(observational_data, experimental_data):
        Q_sum[1 - a, a] += ett * n_eff
        Q_n[1 - a, a] += n_eff

    cum_regret = np.zeros(T)
    cum_reg = 0.0
    optimal_payoff = GREEDY_CASINO_PAYOFFS.max(axis=1)

    for t in range(T):
        x = rng.integers(0, 2)
        theta0 = rng.beta(alpha[x, 0], beta[x, 0])
        theta1 = rng.beta(alpha[x, 1], beta[x, 1])
        if use_rdc:
            # Within-context contrast at the realized intuition x.
            Q_hat = np.where(Q_n > 0, Q_sum / np.maximum(Q_n, 1e-9), 0.5)
            q_follow = Q_hat[x, x]  # Q2: play the intuitive arm a = x
            q_defy = Q_hat[x, 1 - x]  # Q1: play the counter-intuitive arm
            bias = np.clip(
                1.0 - abs(q_defy - q_follow), RDC_WEIGHT_FLOOR, RDC_WEIGHT_CEILING
            )
            w = np.ones(2)
            w[x if q_defy > q_follow else 1 - x] = bias  # suppress the disfavored arm
            a = 0 if (w[0] * theta0) >= (w[1] * theta1) else 1
        else:
            a = 0 if theta0 >= theta1 else 1
        p = GREEDY_CASINO_PAYOFFS[x, a]
        y = int(rng.uniform() < p)
        alpha[x, a] += y
        beta[x, a] += 1 - y
        Q_sum[x, a] += y
        Q_n[x, a] += 1
        cum_reg += optimal_payoff[x] - p
        cum_regret[t] = cum_reg

    return cum_regret


def vanilla_thompson_sampling(T, rng):
    """Standard Thompson sampling on the same MABUC environment, ignoring intuition x.

    The agent has a single Beta posterior per arm. Reward is still drawn from
    GREEDY_CASINO_PAYOFFS[x, a] but the agent never observes or conditions on x.
    """
    alpha = np.ones(2)
    beta = np.ones(2)

    cum_regret = np.zeros(T)
    cum_reg = 0.0
    optimal_payoff = GREEDY_CASINO_PAYOFFS.max(axis=1)
    for t in range(T):
        x = rng.integers(0, 2)
        theta0 = rng.beta(alpha[0], beta[0])
        theta1 = rng.beta(alpha[1], beta[1])
        a = 0 if theta0 >= theta1 else 1
        p = GREEDY_CASINO_PAYOFFS[x, a]
        y = int(rng.uniform() < p)
        alpha[a] += y
        beta[a] += 1 - y
        cum_reg += optimal_payoff[x] - p
        cum_regret[t] = cum_reg
    return cum_regret


# ---------------------------------------------------------------------------
# Experiment 1: simple regret vs m(q) at fixed (T, N)
# ---------------------------------------------------------------------------
def run_regret_vs_m():
    """For each m in M_GRID, repeatedly draw q with hardness m, run all
    algorithms, and record simple regret = optimal value - recommended value.
    """
    N = N_PARENTS
    T = T_HORIZON
    n_seeds = N_SEEDS_REGRET

    results = {
        alg: np.zeros((len(M_GRID), n_seeds))
        for alg in ("successive_reject", "lattimore_alg1")
    }
    realized_m = np.zeros((len(M_GRID), n_seeds), dtype=int)
    opt_arm_pulled = np.zeros((len(M_GRID), n_seeds), dtype=bool)

    for mi, m in enumerate(M_GRID):
        for s in tqdm(
            range(n_seeds),
            desc=f"  m={m}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            seed = (m * 10_007 + s) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            q = make_q_with_hardness(m, N, rng)
            w = rng.integers(0, N)  # the high-leverage coordinate (best-arm location)
            realized_m[mi, s] = hardness_m(q)

            opt_arm = 2 * w + 2
            opt_value = arm_expected_reward_exact(opt_arm, q, w)

            # Successive reject
            rec, rec_value = successive_reject(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results["successive_reject"][mi, s] = opt_value - rec_value

            # Lattimore Alg 1
            rec, rec_value, opt_pulled = lattimore_alg1(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results["lattimore_alg1"][mi, s] = opt_value - rec_value
            opt_arm_pulled[mi, s] = opt_pulled

    return {
        "simple_regret": results,
        "m_grid": list(M_GRID),
        "realized_m": realized_m,
        "opt_arm_pulled": opt_arm_pulled,
        "N": N,
        "T": T,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Experiment 2: simple regret vs T at fixed m, N
# ---------------------------------------------------------------------------
def run_regret_vs_T():
    """For each T in T_GRID, fix m = M_AT_T_PANEL, run all algorithms.

    The gap shrinks with the budget, eps_T = GAP_SCALE * sqrt(m / T). On a fixed
    gap the best arm is identified outright once T is large, so simple regret
    decays exponentially and the panel cannot exhibit the T^{-1/2} rate at all:
    that rate is minimax over instances, not a per-instance decay. Letting the
    gap track sqrt(m/T) holds the instance on the minimax frontier, which is
    where Theorem 1 of Lattimore et al. is stated and where a fitted slope of
    -1/2 is the thing the theorem predicts.
    """
    N = N_PARENTS
    m = M_AT_T_PANEL
    n_seeds = T_SWEEP_SEEDS

    results = {
        alg: np.zeros((len(T_GRID), n_seeds))
        for alg in ("successive_reject", "lattimore_alg1")
    }

    for ti, T in enumerate(T_GRID):
        for s in tqdm(
            range(n_seeds),
            desc=f"  T={T}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            seed = (T * 10_007 + s + 12345) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            q = make_q_with_hardness(m, N, rng)
            w = rng.integers(0, N)
            eps = eps_at_T(m, T)

            opt_arm = 2 * w + 2
            opt_value = arm_expected_reward_exact(opt_arm, q, w, eps)

            rec, _ = successive_reject(q, w, T, rng, eps)
            rec_value = arm_expected_reward_exact(rec, q, w, eps)
            results["successive_reject"][ti, s] = opt_value - rec_value

            rec, _, _ = lattimore_alg1(q, w, T, rng, eps)
            rec_value = arm_expected_reward_exact(rec, q, w, eps)
            results["lattimore_alg1"][ti, s] = opt_value - rec_value

    return {
        "simple_regret": results,
        "T_grid": list(T_GRID),
        "eps_grid": [eps_at_T(m, T) for T in T_GRID],
        "N": N,
        "m": m,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Experiment 3: MABUC greedy casino
# ---------------------------------------------------------------------------
def make_mabuc_logs(rng, n_obs, n_exp):
    """The two logs an MABUC agent starts with.

    Observational: the agent played its intuition, a = x, and the log records
    (x, y). This identifies P(y | X = x).

    Experimental: the arm was randomized, so the log records (a, y) and is a
    draw from P(y | do(a)). The intuition is NOT recorded, which is the whole
    difficulty: the confounder is latent in the experimental record, so the
    samples cannot be filed under a context. Only the ETT identity puts them to
    work.
    """
    obs = []
    for _ in range(n_obs):
        x = int(rng.integers(0, 2))
        obs.append((x, int(rng.uniform() < GREEDY_CASINO_PAYOFFS[x, x])))
    exp = []
    for _ in range(n_exp):
        x = int(rng.integers(0, 2))  # latent, drives the payoff, never recorded
        a = int(rng.integers(0, 2))  # randomized: this is do(a)
        exp.append((a, int(rng.uniform() < GREEDY_CASINO_PAYOFFS[x, a])))
    return obs, exp


def run_mabuc():
    """Greedy casino: vanilla TS, plus the full TS_C component factorial.

    All variants share one seed per replication (common random numbers), so the
    contrasts are paired rather than three unrelated draws.
    """
    T, n_seeds = MABUC_T, MABUC_SEEDS
    n_obs, n_exp = MABUC_N_OBS, MABUC_N_EXP

    ts_regret = np.zeros((n_seeds, T))
    variants = {k: np.zeros((n_seeds, T)) for k in MABUC_VARIANTS}

    for s in tqdm(
        range(n_seeds), desc="  mabuc", leave=False, disable=not sys.stderr.isatty()
    ):
        obs, exp = make_mabuc_logs(np.random.default_rng(s + 99), n_obs, n_exp)
        ts_regret[s] = vanilla_thompson_sampling(T, np.random.default_rng(s + 100_000))
        for name, (seed_mode, use_rdc) in MABUC_VARIANTS.items():
            variants[name][s] = causal_thompson_sampling_tsc(
                T,
                np.random.default_rng(s + 100_000),  # common random numbers
                observational_data=obs,
                experimental_data=exp,
                seed_mode=seed_mode,
                use_rdc=use_rdc,
            )

    # Seed diagnostic: what each mode believes about each cell before round 1.
    obs0, exp0 = make_mabuc_logs(np.random.default_rng(99), n_obs, n_exp)
    seed_diag = {}
    for mode in ("obs_diagonal", "consistency_copy", "ett"):
        a_, b_ = seed_posteriors(mode, obs0, exp0)
        seed_diag[mode] = {"mean": a_ / (a_ + b_), "n_eff": a_ + b_ - 2.0}

    # Where the ETT seed can actually pay: more logged data should help it
    # monotonically and do nothing for a learner that cannot read that log.
    sweep = {}
    for n in MABUC_NOBS_SWEEP:
        cc = np.zeros(MABUC_SWEEP_SEEDS)
        tc = np.zeros(MABUC_SWEEP_SEEDS)
        for s in range(MABUC_SWEEP_SEEDS):
            o, e = make_mabuc_logs(np.random.default_rng(s + 77), n, n)
            cc[s] = causal_thompson_sampling_tsc(
                T, np.random.default_rng(s + 500_000), o, e, "obs_diagonal", False
            )[-1]
            tc[s] = causal_thompson_sampling_tsc(
                T, np.random.default_rng(s + 500_000), o, e, "ett", True
            )[-1]
        sweep[n] = {"CCTS": cc, "TS_C": tc}

    return {
        "ts": ts_regret,
        "variants": variants,
        "cctp": variants["CCTS (obs seed, no RDC)"],
        "tsc": variants["TS_C (ETT seed + RDC)"],
        "seed_diag": seed_diag,
        "nobs_sweep": sweep,
        "T": T,
        "n_seeds": n_seeds,
        "n_obs": n_obs,
        "n_exp": n_exp,
    }


# ---------------------------------------------------------------------------
# Compute_data
# ---------------------------------------------------------------------------
def compute_data(force=None):
    force = force or set()

    res_m = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "regret_vs_m",
        REGRET_VS_M_CONFIG,
        run_regret_vs_m,
        force=("regret_vs_m" in force),
    )
    res_T = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "regret_vs_T",
        REGRET_VS_T_CONFIG,
        run_regret_vs_T,
        force=("regret_vs_T" in force),
    )
    res_mabuc = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "mabuc",
        MABUC_CONFIG,
        run_mabuc,
        force=("mabuc" in force),
    )

    return {"regret_vs_m": res_m, "regret_vs_T": res_T, "mabuc": res_mabuc}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
ALG_LABELS = {
    "successive_reject": "Successive Reject",
    "lattimore_alg1": "Lattimore Alg 1",
}
ALG_COLORS = {
    "successive_reject": COLORS["red"],
    "lattimore_alg1": COLORS["blue"],
}


def make_figure_combined(data):
    """Combined 1x3 figure: regret-vs-m, regret-vs-T, and MABUC cumulative regret."""
    res_m = data["regret_vs_m"]
    res_T = data["regret_vs_T"]
    res_mabuc = data["mabuc"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel (a): simple regret vs hardness m(q)
    ax = axes[0]
    m_grid = res_m["m_grid"]
    N = res_m["N"]
    T = res_m["T"]
    for alg in ("successive_reject", "lattimore_alg1"):
        mean_r = res_m["simple_regret"][alg].mean(axis=1)
        se_r = res_m["simple_regret"][alg].std(axis=1) / np.sqrt(res_m["n_seeds"])
        ax.errorbar(
            m_grid,
            mean_r,
            yerr=1.96 * se_r,
            marker="o",
            label=ALG_LABELS[alg],
            color=ALG_COLORS[alg],
            capsize=3,
        )
    ax.axhline(
        np.sqrt(N / T) * EPS_REWARD,
        **BENCH_STYLE,
        label=r"$\sqrt{N/T} \cdot \epsilon$ asymptotic rate (lower bound)",
    )
    ax.set_xlabel(r"graph hardness $m(q)$")
    ax.set_ylabel("simple regret")
    ax.set_title(rf"(a) Regret vs $m(q)$ at $N={N}$, $T={T}$")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    # Panel (b): simple regret vs horizon T
    ax = axes[1]
    T_grid = res_T["T_grid"]
    m = res_T["m"]
    for alg in ("successive_reject", "lattimore_alg1"):
        mean_r = res_T["simple_regret"][alg].mean(axis=1)
        se_r = res_T["simple_regret"][alg].std(axis=1) / np.sqrt(res_T["n_seeds"])
        ax.errorbar(
            T_grid,
            mean_r,
            yerr=1.96 * se_r,
            marker="o",
            label=ALG_LABELS[alg],
            color=ALG_COLORS[alg],
            capsize=3,
        )
    # Theoretical slope, anchored at the first cell so only the SLOPE is being
    # compared. Theorem 1 predicts -1/2 on log-log axes.
    T_arr = np.array(T_grid, dtype=float)
    anchor = res_T["simple_regret"]["lattimore_alg1"][0].mean()
    ax.plot(
        T_arr,
        anchor * np.sqrt(T_arr[0] / T_arr),
        **BENCH_STYLE,
        label=r"$\sqrt{m(q)/T}$ reference slope",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"horizon $T$")
    ax.set_ylabel("simple regret")
    ax.set_title(
        rf"(b) Regret vs $T$ at $m(q)={m}$, gap $\epsilon_T\propto\sqrt{{m/T}}$"
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    # Panel (c): MABUC greedy-casino cumulative regret
    ax = axes[2]
    T_m = res_mabuc["T"]
    ts_mean = res_mabuc["ts"].mean(axis=0)
    ts_se = res_mabuc["ts"].std(axis=0) / np.sqrt(res_mabuc["n_seeds"])
    cctp_mean = res_mabuc["cctp"].mean(axis=0)
    cctp_se = res_mabuc["cctp"].std(axis=0) / np.sqrt(res_mabuc["n_seeds"])
    tsc_mean = res_mabuc["tsc"].mean(axis=0)
    tsc_se = res_mabuc["tsc"].std(axis=0) / np.sqrt(res_mabuc["n_seeds"])
    tt = np.arange(1, T_m + 1)
    # Symlog, because the interesting contrast now spans four orders of
    # magnitude: vanilla TS ends near 200 while the context-conditional variants
    # sit between 0 and 5, and on a linear axis they collapse onto one another.
    # The linear region below 0.1 keeps the exactly-zero curve visible.
    n_mab = res_mabuc["n_seeds"]
    curves = [
        ("Thompson sampling", res_mabuc["ts"], COLORS["red"]),
        (
            "Consistency-copy seed",
            res_mabuc["variants"]["consistency-copy seed, no RDC"],
            COLORS["orange"],
        ),
        (
            "Context-conditional TS (CCTS)",
            res_mabuc["variants"]["CCTS (obs seed, no RDC)"],
            COLORS["blue"],
        ),
        (
            r"$\mathrm{TS}_C$ (ETT seed $+$ RDC)",
            res_mabuc["variants"]["TS_C (ETT seed + RDC)"],
            COLORS["green"],
        ),
    ]
    for label, arr, colour in curves:
        mean = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(n_mab)
        ax.plot(tt, mean, color=colour, label=label)
        ax.fill_between(tt, mean - 1.96 * se, mean + 1.96 * se, color=colour, alpha=0.2)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"round $t$")
    ax.set_ylabel("cumulative regret")
    ax.set_title("(c) Greedy-casino MABUC")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "causal_bandit_combined.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out}")


def make_table(data):
    """Consolidated simple-regret table across m grid at fixed T."""
    res = data["regret_vs_m"]
    m_grid = res["m_grid"]
    rows = []
    for alg in ("successive_reject", "lattimore_alg1"):
        means = res["simple_regret"][alg].mean(axis=1)
        ses = res["simple_regret"][alg].std(axis=1) / np.sqrt(res["n_seeds"])
        rows.append(
            [ALG_LABELS[alg]] + [f"{m:.3f} ({s:.3f})" for m, s in zip(means, ses)]
        )

    tex = [r"\begin{tabular}{l" + "r" * len(m_grid) + r"}"]
    tex.append(r"\toprule")
    header = "Method " + " & " + " & ".join(f"$m = {m}$" for m in m_grid) + r" \\"
    tex.append(header)
    tex.append(r"\midrule")
    for r in rows:
        tex.append(" & ".join(r) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    out = os.path.join(OUTPUT_DIR, "causal_bandit_results.tex")
    with open(out, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  Table saved: {out}")

    # MABUC three-variant cumulative-regret table at fixed horizon T.
    res_mabuc = data["mabuc"]
    T_mabuc = res_mabuc["T"]
    n_mabuc = res_mabuc["n_seeds"]
    # Burn-in is where the counterfactual seed can pay: every context-conditional
    # variant is flat well before T, so the early column carries the comparison.
    t_early = min(100, T_mabuc) - 1
    tex_labels = {
        "CCTS (obs seed, no RDC)": r"CCTS (observational seed, no RDC)",
        "obs seed + RDC": r"Observational seed $+$ RDC",
        "consistency-copy seed, no RDC": r"Consistency-copy seed, no RDC",
        "consistency-copy seed + RDC": r"Consistency-copy seed $+$ RDC",
        "ETT seed, no RDC": r"ETT seed, no RDC",
        "TS_C (ETT seed + RDC)": (
            r"$\mathrm{TS}_C$: ETT seed $+$ RDC \citep{bareinboim2015mabuc}"
        ),
    }

    def _row(label, arr):
        return (
            label,
            arr[:, -1].mean(),
            arr[:, -1].std() / np.sqrt(n_mabuc),
            arr[:, t_early].mean(),
        )

    mabuc_rows = [_row("Vanilla Thompson sampling", res_mabuc["ts"])]
    for key, arr in res_mabuc["variants"].items():
        mabuc_rows.append(_row(tex_labels[key], arr))
    mabuc_rows_sorted = sorted(mabuc_rows, key=lambda r: r[1])  # best first
    tex2 = [r"\begin{tabular}{lrr}"]
    tex2.append(r"\toprule")
    tex2.append(
        r"Algorithm & Regret at $T = "
        + str(t_early + 1)
        + r"$ & Regret at $T = "
        + str(T_mabuc)
        + r"$ \\"
    )
    tex2.append(r"\midrule")
    for label, mean, se, early in mabuc_rows_sorted:
        tex2.append(f"{label} & {early:.2f} & {mean:.2f} ({se:.2f}) " + r"\\")
    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")
    out2 = os.path.join(OUTPUT_DIR, "causal_bandit_mabuc_results.tex")
    with open(out2, "w") as f:
        f.write("\n".join(tex2) + "\n")
    print(f"  Table saved: {out2}")


def print_stdout(data):
    res_m = data["regret_vs_m"]
    res_T = data["regret_vs_T"]
    res_mabuc = data["mabuc"]
    print()
    print("=" * 70)
    print("  Causal Bandits on Parallel Bandits + Greedy Casino -- summary")
    print("=" * 70)
    print(f"  N parents = {N_PARENTS}, action set |A| = {2 * N_PARENTS + 1}")
    print(f"  Reward gap epsilon = {EPS_REWARD}")
    print()
    print(
        "  --- Simple regret vs. graph hardness m(q) at T = {} ---".format(res_m["T"])
    )
    print(
        f"  {'m(q)':>6}  {'Succ-Reject mean (SE)':>26}  {'Lattimore Alg1 mean (SE)':>30}"
    )
    for i, m in enumerate(res_m["m_grid"]):
        sr = res_m["simple_regret"]["successive_reject"][i]
        la = res_m["simple_regret"]["lattimore_alg1"][i]
        print(
            f"  {m:>6d}  {sr.mean():>14.4f}"
            f" ({sr.std() / np.sqrt(res_m['n_seeds']):.4f})"
            f"  {la.mean():>18.4f}"
            f" ({la.std() / np.sqrt(res_m['n_seeds']):.4f})"
        )
    print()
    # Instance and allocation checks. The first column verifies that the
    # constructed instance has the hardness its label claims; the second
    # verifies that phase 2 spends budget on the arm the theory says is starved.
    print("  --- Instance and allocation checks ---")
    # The rate claim R_T = O(sqrt(m/T)) says R_T / sqrt(m/T) is bounded in m.
    # Reporting that ratio directly beats anchoring on one cell, which divides
    # by zero whenever the easiest cell is identified outright.
    print(
        f"  {'label m':>8}  {'realized m(q)':>14}  {'regret':>9}  "
        f"{'sqrt(m/T)':>10}  {'ratio':>7}  {'P(opt arm pulled)':>18}"
    )
    T_m = res_m["T"]
    for i, m in enumerate(res_m["m_grid"]):
        rm = res_m["realized_m"][i]
        la = res_m["simple_regret"]["lattimore_alg1"][i].mean()
        rate = np.sqrt(m / T_m)
        print(
            f"  {m:>8d}  {rm.min():>6d}..{rm.max():<6d}  {la:>9.4f}  "
            f"{rate:>10.4f}  {la / rate:>7.3f}  "
            f"{res_m['opt_arm_pulled'][i].mean():>18.3f}"
        )
    print()
    print("  --- Simple regret vs. horizon T at m(q) = {} ---".format(res_T["m"]))
    print(
        f"  {'T':>6}  {'gap eps_T':>10}  {'Succ-Reject mean (SE)':>26}  "
        f"{'Lattimore Alg1 mean (SE)':>30}"
    )
    for i, T in enumerate(res_T["T_grid"]):
        sr = res_T["simple_regret"]["successive_reject"][i]
        la = res_T["simple_regret"]["lattimore_alg1"][i]
        print(
            f"  {T:>6d}  {res_T['eps_grid'][i]:>10.4f}  {sr.mean():>14.4f}"
            f" ({sr.std() / np.sqrt(res_T['n_seeds']):.4f})"
            f"  {la.mean():>18.4f}"
            f" ({la.std() / np.sqrt(res_T['n_seeds']):.4f})"
        )
    # Theorem 1 predicts regret ~ T^{-1/2}; a fitted log-log slope near -0.5 is
    # the direct reading. Cells with zero mean regret carry no log and drop out.
    Ts = np.array(res_T["T_grid"], dtype=float)
    for alg in ("lattimore_alg1", "successive_reject"):
        means = np.array(
            [res_T["simple_regret"][alg][i].mean() for i in range(len(Ts))]
        )
        ok = means > 0
        if ok.sum() >= 2:
            slope = np.polyfit(np.log(Ts[ok]), np.log(means[ok]), 1)[0]
            print(
                f"  log-log regret slope, {alg}: {slope:+.2f}"
                f"  (theory predicts -0.50 for the graph-aware algorithm;"
                f" {int(ok.sum())}/{len(Ts)} cells used)"
            )
        else:
            print(f"  log-log regret slope, {alg}: undefined (mean regret is 0)")
    print()
    print("  --- Greedy-casino MABUC at T = {} ---".format(res_mabuc["T"]))
    final_ts = res_mabuc["ts"][:, -1]
    final_cctp = res_mabuc["cctp"][:, -1]
    final_tsc = res_mabuc["tsc"][:, -1]
    n = res_mabuc["n_seeds"]
    print(
        f"  Vanilla Thompson sampling          : cumulative regret = {final_ts.mean():.2f}"
        f" (SE {final_ts.std() / np.sqrt(n):.2f})"
    )
    print(
        f"  Context-conditional Thompson (CCTS): cumulative regret = {final_cctp.mean():.2f}"
        f" (SE {final_cctp.std() / np.sqrt(n):.2f})"
    )
    print(
        f"  Full TS_C (Bareinboim 2015)        : cumulative regret = {final_tsc.mean():.2f}"
        f" (SE {final_tsc.std() / np.sqrt(n):.2f})"
    )
    ratio_ts_cctp = final_ts.mean() / max(final_cctp.mean(), 1e-3)
    ratio_ts_tsc = final_ts.mean() / max(final_tsc.mean(), 1e-3)
    print(f"  Ratio (vanilla TS) / (CCTS)        = {ratio_ts_cctp:.1f}x")
    print(f"  Ratio (vanilla TS) / (TS_C)        = {ratio_ts_tsc:.1f}x")
    print()

    # What each seeding mode believes before round 1, against the truth. An
    # off-intuition cell seeded far from its true value, and held with a large
    # pseudo-count, is a burn-in cost that aggregate regret alone will not name.
    print("  --- Seeded posterior mean per (intuition x, arm a), before round 1 ---")
    print(f"  {'mode':<20} {'(0,0)':>16} {'(0,1)':>16} {'(1,0)':>16} {'(1,1)':>16}")
    truth = GREEDY_CASINO_PAYOFFS
    print(
        f"  {'true payoff':<20}"
        + "".join(f"{truth[x, a]:>16.3f}" for x in (0, 1) for a in (0, 1))
    )
    for mode, d in res_mabuc["seed_diag"].items():
        m, ne = d["mean"], d["n_eff"]
        print(
            f"  {mode:<20}"
            + "".join(
                f"{m[x, a]:>10.3f} n={ne[x, a]:<4.0f}" for x in (0, 1) for a in (0, 1)
            )
        )
    print()

    # Component factorial, paired across seeds by common random numbers.
    print("  --- TS_C component factorial (common random numbers, paired) ---")
    t_e = min(100, res_mabuc["T"]) - 1
    print(
        f"  {'variant':<34} {'regret@' + str(t_e + 1):>12} {'regret@' + str(res_mabuc['T']):>14} "
        f"{'(SE)':>8}  {'paired diff vs CCTS':>20}"
    )
    base = res_mabuc["variants"]["CCTS (obs seed, no RDC)"][:, -1]
    for name, arr in sorted(
        res_mabuc["variants"].items(), key=lambda kv: kv[1][:, -1].mean()
    ):
        fin = arr[:, -1]
        d = fin - base  # paired: same seed, same logs, same RNG start
        print(
            f"  {name:<34} {arr[:, t_e].mean():>12.2f} {fin.mean():>14.2f} "
            f"{fin.std() / np.sqrt(n):>8.2f}  {d.mean():>+12.2f} ({d.std() / np.sqrt(n):.2f})"
        )
    print()

    # More logged data should help the learner that can read the interventional
    # log and do nothing for the one that cannot.
    print("  --- Final cumulative regret vs log size (n_obs = n_exp) ---")
    print(f"  {'n':>6} {'CCTS':>18} {'TS_C':>18}")
    for nn, d in res_mabuc["nobs_sweep"].items():
        cc, tc = d["CCTS"], d["TS_C"]
        k = len(cc)
        print(
            f"  {nn:>6} {cc.mean():>10.2f} ({cc.std() / np.sqrt(k):.2f})"
            f" {tc.mean():>10.2f} ({tc.std() / np.sqrt(k):.2f})"
        )
    print()
    print("  Output files:")
    for f in (
        "causal_bandit_combined.png",
        "causal_bandit_results.tex",
        "causal_bandit_mabuc_results.tex",
    ):
        print("    ", os.path.join(OUTPUT_DIR, f))


def generate_outputs(data):
    print_stdout(data)
    make_figure_combined(data)
    make_table(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    if args.plots_only:
        data = compute_data(force=set())
    else:
        data = compute_data(force=force)

    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
