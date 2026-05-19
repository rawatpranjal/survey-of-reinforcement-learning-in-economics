# Causal Bandits on the parallel-bandit family.
# Chapter 11, RL for Causal Inference, Section 4a (Causal Bandits).
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
#      (i) consistency-axiom seeding (each observation (X=x, A=a, Y=y)
#      contributes a fractional pseudo-count to the off-intuition arm's
#      posterior at the same context, per Pearl 2009 §3.6), and (ii) RDC
#      bias weighting: each posterior sample theta_a is multiplied by
#      w_a = 1 - |Q1(a) - Q2(a)| (clamped to [0.01, 1]) where Q1, Q2 are
#      running empirical means of the do-effect under each context. Arms
#      with high cross-context disagreement are suppressed.
#
# Outputs:
#   causal_bandit_regret_vs_m.png  -- simple regret vs hardness m at fixed T, N
#   causal_bandit_regret_vs_T.png  -- simple regret vs horizon T at fixed m, N
#   causal_bandit_mabuc.png        -- cumulative regret in greedy casino
#   causal_bandit_results.tex      -- consolidated table at T=400, m grid
#   causal_bandit_stdout.txt       -- numerical log (run via shell redirect)

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, BENCH_STYLE, FIG_DOUBLE, FIG_SINGLE
from sims.sim_cache import (
    compute_or_load, add_component_args, parse_force_set,
)
apply_style()
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'causal_bandit_parallel'

# Parallel-bandit DGP
N_PARENTS = 50              # number of binary parents X_i (so |A| = 2N+1 arms)
T_HORIZON = 400             # default horizon
M_GRID = [2, 8, 24, 48]     # graph-derived hardness values (m(q) levels)
N_SEEDS_REGRET = 2000       # Monte Carlo replications for simple-regret panels
T_GRID = [100, 200, 400, 800, 1600]  # horizon grid for vs-T panel
M_AT_T_PANEL = 2            # fix m(q) for the vs-T panel
EPS_REWARD = 0.30           # reward gap epsilon: best arm pays mu* = 0.5+eps,
                            # all others pay 0.5

# Bareinboim "greedy casino" parameters
MABUC_T = 1000              # horizon for cumulative-regret panel
MABUC_SEEDS = 500           # Monte Carlo replications
# Greedy-casino payoffs (Bareinboim et al. 2015, Table 1, "asterisked" values):
#   Player intuition X corresponds to a confounder configuration.
#   P(Y=1 | X=x, do(X=x)) = high  -- following intuition wins more
#   P(Y=1 | X=x, do(X=1-x)) = low -- going against intuition loses
GREEDY_CASINO_PAYOFFS = np.array([
    # rows: agent's natural intuition x in {0, 1}
    # cols: arm pulled a in {0, 1}
    # entry: P(Y=1 | X=x, do(X=a))
    [0.10, 0.50],   # x=0 (drunk + non-blinking): a=0 loses, a=1 wins
    [0.50, 0.10],   # x=1 (drunk + blinking):     a=0 wins, a=1 loses
])

SHARED_CONFIG = {
    'N_PARENTS': N_PARENTS,
    'T_HORIZON': T_HORIZON,
    'M_GRID': M_GRID,
    'N_SEEDS_REGRET': N_SEEDS_REGRET,
    'T_GRID': T_GRID,
    'M_AT_T_PANEL': M_AT_T_PANEL,
    'EPS_REWARD': EPS_REWARD,
    'MABUC_T': MABUC_T,
    'MABUC_SEEDS': MABUC_SEEDS,
    'GREEDY_CASINO_PAYOFFS': GREEDY_CASINO_PAYOFFS.tolist(),
}

REGRET_VS_M_CONFIG = {**SHARED_CONFIG, 'experiment': 'regret_vs_m'}
REGRET_VS_T_CONFIG = {**SHARED_CONFIG, 'experiment': 'regret_vs_T'}
# v2 marker forces a refresh of MABUC cache since results now include
# the full Bareinboim TS_C (consistency seeding + RDC weighting) alongside CCTS.
MABUC_CONFIG       = {**SHARED_CONFIG, 'experiment': 'mabuc', 'algos_version': 'v2_full_tsc'}


# ---------------------------------------------------------------------------
# Parallel-bandit DGP and helpers
# ---------------------------------------------------------------------------
def make_q_with_hardness(m_target, N, rng):
    """Construct propensities q = (q_1, ..., q_N) in (0,1) so that m(q) = m_target.

    Recall (Lattimore et al. 2016, eq. for m(q)):
        I_tau := {i : min(q_i, 1 - q_i) < 1/tau}
        m(q)  = min_tau in [2, N] max(tau, |I_tau|)

    If we make exactly K of the q_i extreme (e.g. = 0.05 so 1/tau threshold
    captures them) and the rest balanced at 0.5, then for tau in [K, N]
    we get |I_tau| = K (extreme arms below 1/tau = small), so m(q) = max(tau, K)
    minimized at tau = K, giving m(q) = K. So setting K = m_target works
    for m_target in [2, N].
    """
    q = np.full(N, 0.5)
    # Mark exactly m_target arms as extreme (low probability)
    if m_target < N:
        # Place extreme values on a random subset of arms (independent of best arm)
        extreme_idx = rng.permutation(N)[:m_target]
        q[extreme_idx] = 0.05  # well below any 1/tau for tau <= N=50
    else:
        # m_target == N: all arms extreme
        q[:] = 0.05
    return q


def sample_parents(q, rng):
    """One realization of (X_1, ..., X_N) under the observational distribution."""
    return (rng.uniform(size=len(q)) < q).astype(int)


def reward_under_do(parents, intervened_idx, intervened_val, w, baseline_q, rng):
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
    x = parents.copy()
    if intervened_idx >= 0:
        x[intervened_idx] = intervened_val

    # Reward depends only on the high-leverage coordinate (index `w`).
    if x[w] == 1:
        p = 0.5 + EPS_REWARD
    else:
        p = 0.5 - EPS_REWARD
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
def successive_reject(q, w, T, rng):
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
                y = pull_arm(a, q, w, rng)
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
            y = pull_arm(a, q, w, rng)
            sums[a] += y
            counts[a] += 1
            used += 1

    means = np.where(counts > 0, sums / np.maximum(counts, 1), -np.inf)
    rec = int(np.argmax(means))
    return rec, arm_expected_reward(rec, w)


def pull_arm(arm_id, q, w, rng):
    """Pull one of the 2N+1 arms; return realized reward."""
    N = len(q)
    if arm_id == 0:
        # do() : draw from observational distribution
        parents = sample_parents(q, rng)
        intervened_idx, intervened_val = -1, 0
    else:
        i = (arm_id - 1) // 2
        j = (arm_id - 1) % 2
        parents = sample_parents(q, rng)
        intervened_idx, intervened_val = i, j
    y, _ = reward_under_do(parents, intervened_idx, intervened_val, w, q, rng)
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


def arm_expected_reward_exact(arm_id, q, w):
    """Exact expected reward of arm_id under propensities q and optimal coord w."""
    N = len(q)
    if arm_id == 2 * w + 2:
        return 0.5 + EPS_REWARD
    if arm_id == 2 * w + 1:
        return 0.5 - EPS_REWARD
    # Any other arm: X_w not intervened, so X_w ~ Bernoulli(q[w])
    return q[w] * (0.5 + EPS_REWARD) + (1 - q[w]) * (0.5 - EPS_REWARD)


# ---------------------------------------------------------------------------
# Algorithm 2: Lattimore Algorithm 1 (parallel-bandit causal algorithm)
# ---------------------------------------------------------------------------
def lattimore_alg1(q_true, w, T, rng):
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
        y, _ = reward_under_do(parents, -1, 0, w, q_true, rng)
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
    unbalanced_arms = []
    for i in range(N):
        if min(q_hat[i], 1 - q_hat[i]) < 1.0 / best_tau:
            j_rare = 0 if q_hat[i] < 0.5 else 1
            unbalanced_arms.append(2 * i + 1 + j_rare)  # the arm index for do(X_i = j_rare)
    if not unbalanced_arms:
        unbalanced_arms = list(range(1, K))  # fallback: all do() arms

    # Spend the remaining T - T_obs rounds uniformly on the unbalanced arms.
    T_remain = T - T_obs
    pulls_per_arm = max(1, T_remain // len(unbalanced_arms))
    arm_sums = {a: 0.0 for a in unbalanced_arms}
    arm_counts = {a: 0 for a in unbalanced_arms}
    for a in unbalanced_arms:
        for _ in range(pulls_per_arm):
            y = pull_arm(a, q_true, w, rng)
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
    return rec, arm_estimate[rec]


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

    Minimal causal baseline. Retains the context-specific posterior indexed
    by (intuition x, arm a), but does not implement the consistency-axiom
    seeding of the off-intuition arm or the RDC bias weighting of the full
    TS_C. Bounded cumulative regret is achieved on this instance because
    each (x, a) cell is independently learnable from on-intuition observations
    plus direct experimental pulls.

    Algorithm:
      1. (Optional) Seed Beta posteriors from observational data along
         the on-intuition diagonal (a = x), per the consistency axiom for
         the on-intuition arm only.
      2. At each time, observe the agent's natural intuition x; sample a
         reward estimate for each arm from its (x, a)-conditional Beta
         posterior; choose the arm with the higher sample via straight
         argmax (no RDC bias multiplier).
      3. Observe reward; update the Beta posterior conditioned on (x, a).
    """
    alpha = np.ones((2, 2))  # alpha[x, a] = pseudo-success count
    beta = np.ones((2, 2))   # beta[x, a]  = pseudo-failure count

    # Seed from observational data (sample-on-intuition consistency only)
    if observational_data is not None:
        for x_obs, y_obs in observational_data:
            alpha[x_obs, x_obs] += y_obs
            beta[x_obs, x_obs] += 1 - y_obs

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


# Pseudo-count fraction for consistency-axiom seeding of the off-intuition arm.
# Bareinboim et al. (2015) Algorithm 1 line 2 seeds the on-intuition arm
# directly from P_obs(y|x); the consistency-axiom extension to the
# off-intuition arm a != x has no canonical pseudo-count in the paper, so we
# use a conservative fractional weight. We disclose this in the tex.
CONSISTENCY_OFF_INTUITION_WEIGHT = 0.5

# Floor/ceiling for the RDC bias multiplier w_a = 1 - |Q1(a) - Q2(a)|.
# w_a in [0.01, 1] avoids numerical degeneracy when Q1 and Q2 collide.
RDC_WEIGHT_FLOOR = 0.01
RDC_WEIGHT_CEILING = 1.0


def causal_thompson_sampling_tsc(T, rng, observational_data=None):
    """Full Causal Thompson Sampling TS_C (Bareinboim, Forney & Pearl 2015, Alg. 1).

    Implements the two distinguishing features of the paper's TS_C beyond the
    minimal context-conditional baseline:

    (i) Consistency-axiom seeding (Pearl 2009 sec 3.6, Bareinboim 2015 Alg. 1
        line 2). For an observation (X=x, A=a, Y=y) the on-intuition cell
        (a = x) receives a full pseudo-count, and the off-intuition cell
        (a != x) at the same context x receives a fractional pseudo-count
        equal to CONSISTENCY_OFF_INTUITION_WEIGHT * y (success) or
        CONSISTENCY_OFF_INTUITION_WEIGHT * (1-y) (failure). The
        consistency-derived seed on the off-intuition arm is conservative
        because we do not have direct observational access to E[Y_{X=a'} | X=x]
        for a' != x; the binary-arm symmetry of the greedy casino means a
        weak positive prior in that cell is a useful summary of "if the agent
        had played against intuition, the payoff distribution is not
        radically different from the observed on-intuition distribution".

    (ii) RDC (Relative Difference of Confounders) bias weighting
         (Bareinboim 2015 Alg. 1 lines 5-10). Maintain running empirical
         estimates Q_hat[x, a] = E[Y | X=x, do(A=a)] for each (x, a) cell.
         At each round, sample theta_a from the (x, a)-conditional Beta
         posterior as usual, then multiply by
             w_a = clamp(1 - |Q_hat[1, a] - Q_hat[2, a]|, 0.01, 1).
         Arms whose do-effect varies sharply across contexts (high
         cross-context disagreement) are suppressed; arms with stable
         cross-context effects retain their full posterior signal. Pick
         argmax_a (w_a * theta_a).

    The paper's Algorithm 1 lines 5-9 use Q1 = E(Y_{X=x'} | X=x) and
    Q2 = P(y | X=x), assigning the bias multiplier to the arm favored by
    the higher Q value. The Q_hat[x, a] formulation above is the natural
    generalization to running estimates that update as data accumulates.
    """
    alpha = np.ones((2, 2))  # alpha[x, a]
    beta = np.ones((2, 2))   # beta[x, a]

    # Q_hat[x, a]: running empirical mean reward at (x, a).
    Q_sum = np.zeros((2, 2))
    Q_n = np.zeros((2, 2))

    # Consistency-axiom seeding from observational data.
    # On-intuition cell: full pseudo-count (canonical Bareinboim 2015 line 2).
    # Off-intuition cell at same context: fractional pseudo-count.
    if observational_data is not None:
        c = CONSISTENCY_OFF_INTUITION_WEIGHT
        for x_obs, y_obs in observational_data:
            # On-intuition (a = x_obs): full count
            alpha[x_obs, x_obs] += y_obs
            beta[x_obs, x_obs] += 1 - y_obs
            Q_sum[x_obs, x_obs] += y_obs
            Q_n[x_obs, x_obs] += 1
            # Off-intuition (a != x_obs) at same context: fractional count
            a_off = 1 - x_obs
            alpha[x_obs, a_off] += c * y_obs
            beta[x_obs, a_off] += c * (1 - y_obs)
            Q_sum[x_obs, a_off] += c * y_obs
            Q_n[x_obs, a_off] += c

    cum_regret = np.zeros(T)
    cum_reg = 0.0
    optimal_payoff = GREEDY_CASINO_PAYOFFS.max(axis=1)

    for t in range(T):
        x = rng.integers(0, 2)
        # Posterior samples
        theta0 = rng.beta(alpha[x, 0], beta[x, 0])
        theta1 = rng.beta(alpha[x, 1], beta[x, 1])
        # RDC bias weights from running Q_hat estimates.
        # Q_hat[x, a] defaults to 0.5 (uninformative) when n=0.
        Q_hat = np.where(Q_n > 0, Q_sum / np.maximum(Q_n, 1e-9), 0.5)
        # Disagreement of arm a across contexts {0, 1}
        disagree_0 = abs(Q_hat[0, 0] - Q_hat[1, 0])
        disagree_1 = abs(Q_hat[0, 1] - Q_hat[1, 1])
        w0 = np.clip(1.0 - disagree_0, RDC_WEIGHT_FLOOR, RDC_WEIGHT_CEILING)
        w1 = np.clip(1.0 - disagree_1, RDC_WEIGHT_FLOOR, RDC_WEIGHT_CEILING)
        # Arm selection: argmax of weighted posterior samples
        a = 0 if (w0 * theta0) >= (w1 * theta1) else 1
        # Pull and observe
        p = GREEDY_CASINO_PAYOFFS[x, a]
        y = int(rng.uniform() < p)
        # Update Beta posterior
        alpha[x, a] += y
        beta[x, a] += 1 - y
        # Update Q_hat running mean
        Q_sum[x, a] += y
        Q_n[x, a] += 1
        # Track regret
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

    results = {alg: np.zeros((len(M_GRID), n_seeds)) for alg in
               ('successive_reject', 'lattimore_alg1')}

    for mi, m in enumerate(M_GRID):
        for s in tqdm(range(n_seeds), desc=f'  m={m}', leave=False,
                       disable=not sys.stderr.isatty()):
            seed = (m * 10_007 + s) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            q = make_q_with_hardness(m, N, rng)
            w = rng.integers(0, N)  # the high-leverage coordinate (best-arm location)

            opt_arm = 2 * w + 2
            opt_value = arm_expected_reward_exact(opt_arm, q, w)

            # Successive reject
            rec, rec_value = successive_reject(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results['successive_reject'][mi, s] = opt_value - rec_value

            # Lattimore Alg 1
            rec, rec_value = lattimore_alg1(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results['lattimore_alg1'][mi, s] = opt_value - rec_value

    return {'simple_regret': results, 'm_grid': list(M_GRID),
            'N': N, 'T': T, 'n_seeds': n_seeds}


# ---------------------------------------------------------------------------
# Experiment 2: simple regret vs T at fixed m, N
# ---------------------------------------------------------------------------
def run_regret_vs_T():
    """For each T in T_GRID, fix m = M_AT_T_PANEL, run all algorithms."""
    N = N_PARENTS
    m = M_AT_T_PANEL
    n_seeds = N_SEEDS_REGRET

    results = {alg: np.zeros((len(T_GRID), n_seeds)) for alg in
               ('successive_reject', 'lattimore_alg1')}

    for ti, T in enumerate(T_GRID):
        for s in tqdm(range(n_seeds), desc=f'  T={T}', leave=False,
                       disable=not sys.stderr.isatty()):
            seed = (T * 10_007 + s + 12345) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            q = make_q_with_hardness(m, N, rng)
            w = rng.integers(0, N)

            opt_arm = 2 * w + 2
            opt_value = arm_expected_reward_exact(opt_arm, q, w)

            rec, _ = successive_reject(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results['successive_reject'][ti, s] = opt_value - rec_value

            rec, _ = lattimore_alg1(q, w, T, rng)
            rec_value = arm_expected_reward_exact(rec, q, w)
            results['lattimore_alg1'][ti, s] = opt_value - rec_value

    return {'simple_regret': results, 'T_grid': list(T_GRID),
            'N': N, 'm': m, 'n_seeds': n_seeds}


# ---------------------------------------------------------------------------
# Experiment 3: MABUC greedy casino
# ---------------------------------------------------------------------------
def run_mabuc():
    """Run vanilla TS, context-conditional TS, and full TS_C on the greedy-casino
    environment. Record cumulative regret."""
    T = MABUC_T
    n_seeds = MABUC_SEEDS

    # Observational seed: simulate n_obs samples where the agent follows its
    # intuition (i.e., plays arm x for each intuition x). Both CCTS and TS_C
    # consume this data, but TS_C additionally derives a fractional
    # pseudo-count for the off-intuition arm via the consistency axiom.
    n_obs = 200

    ts_regret = np.zeros((n_seeds, T))
    cctp_regret = np.zeros((n_seeds, T))
    tsc_regret = np.zeros((n_seeds, T))

    for s in tqdm(range(n_seeds), desc='  mabuc', leave=False,
                   disable=not sys.stderr.isatty()):
        rng = np.random.default_rng(s + 99)
        # Generate observational data: for each iteration, sample intuition x,
        # the agent follows intuition (a = x), observe reward.
        obs = []
        for _ in range(n_obs):
            x = rng.integers(0, 2)
            p = GREEDY_CASINO_PAYOFFS[x, x]
            y = int(rng.uniform() < p)
            obs.append((x, y))
        # Run all three algorithms with independent RNGs to remove shared randomness
        rng_ts = np.random.default_rng(s + 100_000)
        rng_cctp = np.random.default_rng(s + 200_000)
        rng_tsc = np.random.default_rng(s + 300_000)
        ts_regret[s] = vanilla_thompson_sampling(T, rng_ts)
        cctp_regret[s] = context_conditional_thompson_sampling(
            T, rng_cctp, observational_data=obs)
        tsc_regret[s] = causal_thompson_sampling_tsc(
            T, rng_tsc, observational_data=obs)

    return {'ts': ts_regret, 'cctp': cctp_regret, 'tsc': tsc_regret,
            'T': T, 'n_seeds': n_seeds}


# ---------------------------------------------------------------------------
# Compute_data
# ---------------------------------------------------------------------------
def compute_data(force=None):
    force = force or set()

    res_m = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'regret_vs_m', REGRET_VS_M_CONFIG,
        run_regret_vs_m, force=('regret_vs_m' in force),
    )
    res_T = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'regret_vs_T', REGRET_VS_T_CONFIG,
        run_regret_vs_T, force=('regret_vs_T' in force),
    )
    res_mabuc = compute_or_load(
        CACHE_DIR, SCRIPT_NAME, 'mabuc', MABUC_CONFIG,
        run_mabuc, force=('mabuc' in force),
    )

    return {'regret_vs_m': res_m, 'regret_vs_T': res_T, 'mabuc': res_mabuc}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
ALG_LABELS = {
    'successive_reject': 'Successive Reject',
    'lattimore_alg1':    'Lattimore Alg 1',
}
ALG_COLORS = {
    'successive_reject': COLORS['red'],
    'lattimore_alg1':    COLORS['blue'],
}


def make_figure_combined(data):
    """Combined 1x3 figure: regret-vs-m, regret-vs-T, and MABUC cumulative regret."""
    res_m = data['regret_vs_m']
    res_T = data['regret_vs_T']
    res_mabuc = data['mabuc']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel (a): simple regret vs hardness m(q)
    ax = axes[0]
    m_grid = res_m['m_grid']
    N = res_m['N']
    T = res_m['T']
    for alg in ('successive_reject', 'lattimore_alg1'):
        mean_r = res_m['simple_regret'][alg].mean(axis=1)
        se_r = res_m['simple_regret'][alg].std(axis=1) / np.sqrt(res_m['n_seeds'])
        ax.errorbar(m_grid, mean_r, yerr=1.96 * se_r, marker='o',
                    label=ALG_LABELS[alg], color=ALG_COLORS[alg], capsize=3)
    ax.axhline(np.sqrt(N / T) * EPS_REWARD, **BENCH_STYLE,
               label=r'$\sqrt{N/T} \cdot \epsilon$ asymptotic rate (lower bound)')
    ax.set_xlabel(r'graph hardness $m(q)$')
    ax.set_ylabel('simple regret')
    ax.set_title(fr'(a) Regret vs $m(q)$ at $N={N}$, $T={T}$')
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    # Panel (b): simple regret vs horizon T
    ax = axes[1]
    T_grid = res_T['T_grid']
    m = res_T['m']
    for alg in ('successive_reject', 'lattimore_alg1'):
        mean_r = res_T['simple_regret'][alg].mean(axis=1)
        se_r = res_T['simple_regret'][alg].std(axis=1) / np.sqrt(res_T['n_seeds'])
        ax.errorbar(T_grid, mean_r, yerr=1.96 * se_r, marker='o',
                    label=ALG_LABELS[alg], color=ALG_COLORS[alg], capsize=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'horizon $T$')
    ax.set_ylabel('simple regret')
    ax.set_title(fr'(b) Regret vs $T$ at $m(q)={m}$')
    ax.legend(frameon=False, loc='upper right', fontsize=9)

    # Panel (c): MABUC greedy-casino cumulative regret
    ax = axes[2]
    T_m = res_mabuc['T']
    ts_mean = res_mabuc['ts'].mean(axis=0)
    ts_se = res_mabuc['ts'].std(axis=0) / np.sqrt(res_mabuc['n_seeds'])
    cctp_mean = res_mabuc['cctp'].mean(axis=0)
    cctp_se = res_mabuc['cctp'].std(axis=0) / np.sqrt(res_mabuc['n_seeds'])
    tsc_mean = res_mabuc['tsc'].mean(axis=0)
    tsc_se = res_mabuc['tsc'].std(axis=0) / np.sqrt(res_mabuc['n_seeds'])
    tt = np.arange(1, T_m + 1)
    ax.plot(tt, ts_mean, color=COLORS['red'], label='Thompson sampling')
    ax.fill_between(tt, ts_mean - 1.96 * ts_se, ts_mean + 1.96 * ts_se,
                    color=COLORS['red'], alpha=0.2)
    ax.plot(tt, cctp_mean, color=COLORS['blue'],
            label='Context-conditional TS')
    ax.fill_between(tt, cctp_mean - 1.96 * cctp_se, cctp_mean + 1.96 * cctp_se,
                    color=COLORS['blue'], alpha=0.2)
    ax.plot(tt, tsc_mean, color=COLORS['green'],
            label=r'Full $\mathrm{TS}_C$ (Bareinboim 2015)')
    ax.fill_between(tt, tsc_mean - 1.96 * tsc_se, tsc_mean + 1.96 * tsc_se,
                    color=COLORS['green'], alpha=0.2)
    ax.set_xlabel(r'round $t$')
    ax.set_ylabel('cumulative regret')
    ax.set_title('(c) Greedy-casino MABUC')
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'causal_bandit_combined.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out}")


def make_table(data):
    """Consolidated simple-regret table across m grid at fixed T."""
    res = data['regret_vs_m']
    m_grid = res['m_grid']
    rows = []
    for alg in ('successive_reject', 'lattimore_alg1'):
        means = res['simple_regret'][alg].mean(axis=1)
        ses = res['simple_regret'][alg].std(axis=1) / np.sqrt(res['n_seeds'])
        rows.append([ALG_LABELS[alg]] + [f'{m:.3f} ({s:.3f})' for m, s in zip(means, ses)])

    tex = [r'\begin{tabular}{l' + 'r' * len(m_grid) + r'}']
    tex.append(r'\toprule')
    header = 'Method ' + ' & ' + ' & '.join(f'$m = {m}$' for m in m_grid) + r' \\'
    tex.append(header)
    tex.append(r'\midrule')
    for r in rows:
        tex.append(' & '.join(r) + r' \\')
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')

    out = os.path.join(OUTPUT_DIR, 'causal_bandit_results.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Table saved: {out}")


def print_stdout(data):
    res_m = data['regret_vs_m']
    res_T = data['regret_vs_T']
    res_mabuc = data['mabuc']
    print()
    print('=' * 70)
    print('  Causal Bandits on Parallel Bandits + Greedy Casino -- summary')
    print('=' * 70)
    print(f'  N parents = {N_PARENTS}, action set |A| = {2*N_PARENTS+1}')
    print(f'  Reward gap epsilon = {EPS_REWARD}')
    print()
    print('  --- Simple regret vs. graph hardness m(q) at T = {} ---'.format(res_m['T']))
    print(f"  {'m(q)':>6}  {'Succ-Reject mean (SE)':>26}  {'Lattimore Alg1 mean (SE)':>30}")
    for i, m in enumerate(res_m['m_grid']):
        sr = res_m['simple_regret']['successive_reject'][i]
        la = res_m['simple_regret']['lattimore_alg1'][i]
        print(f'  {m:>6d}  {sr.mean():>14.4f}'
              f' ({sr.std()/np.sqrt(res_m["n_seeds"]):.4f})'
              f'  {la.mean():>18.4f}'
              f' ({la.std()/np.sqrt(res_m["n_seeds"]):.4f})')
    print()
    print('  --- Simple regret vs. horizon T at m(q) = {} ---'.format(res_T['m']))
    print(f"  {'T':>6}  {'Succ-Reject mean (SE)':>26}  {'Lattimore Alg1 mean (SE)':>30}")
    for i, T in enumerate(res_T['T_grid']):
        sr = res_T['simple_regret']['successive_reject'][i]
        la = res_T['simple_regret']['lattimore_alg1'][i]
        print(f'  {T:>6d}  {sr.mean():>14.4f}'
              f' ({sr.std()/np.sqrt(res_T["n_seeds"]):.4f})'
              f'  {la.mean():>18.4f}'
              f' ({la.std()/np.sqrt(res_T["n_seeds"]):.4f})')
    print()
    print('  --- Greedy-casino MABUC at T = {} ---'.format(res_mabuc['T']))
    final_ts = res_mabuc['ts'][:, -1]
    final_cctp = res_mabuc['cctp'][:, -1]
    final_tsc = res_mabuc['tsc'][:, -1]
    n = res_mabuc['n_seeds']
    print(f'  Vanilla Thompson sampling          : cumulative regret = {final_ts.mean():.2f}'
          f' (SE {final_ts.std()/np.sqrt(n):.2f})')
    print(f'  Context-conditional Thompson (CCTS): cumulative regret = {final_cctp.mean():.2f}'
          f' (SE {final_cctp.std()/np.sqrt(n):.2f})')
    print(f'  Full TS_C (Bareinboim 2015)        : cumulative regret = {final_tsc.mean():.2f}'
          f' (SE {final_tsc.std()/np.sqrt(n):.2f})')
    ratio_ts_cctp = final_ts.mean() / max(final_cctp.mean(), 1e-3)
    ratio_ts_tsc = final_ts.mean() / max(final_tsc.mean(), 1e-3)
    ratio_cctp_tsc = final_cctp.mean() / max(final_tsc.mean(), 1e-3)
    print(f'  Ratio (vanilla TS) / (CCTS)        = {ratio_ts_cctp:.1f}x')
    print(f'  Ratio (vanilla TS) / (TS_C)        = {ratio_ts_tsc:.1f}x')
    print(f'  Ratio (CCTS)      / (TS_C)         = {ratio_cctp_tsc:.2f}x')
    if final_tsc.mean() <= final_cctp.mean():
        print(f'  TS_C beats CCTS by {final_cctp.mean() - final_tsc.mean():.2f}'
              f' regret units ({(1 - final_tsc.mean()/max(final_cctp.mean(), 1e-9))*100:.1f}%).')
    else:
        print(f'  CCTS beats TS_C by {final_tsc.mean() - final_cctp.mean():.2f}'
              f' regret units ({(1 - final_cctp.mean()/max(final_tsc.mean(), 1e-9))*100:.1f}%).')
    print()
    print('  Output files:')
    for f in ('causal_bandit_combined.png', 'causal_bandit_results.tex'):
        print('    ', os.path.join(OUTPUT_DIR, f))


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


if __name__ == '__main__':
    main()
