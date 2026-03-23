"""
Unified Identification in a Confounded Retail Pricing MDP
Chapter 10: Causal Inference and RL
Compares 6 estimators (Oracle, Naive, Backdoor, Front-door, IV, Proximal)
across confounding strengths in a 5-state engagement funnel.
"""

import argparse
import numpy as np
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, DOMAIN_COLORS, BENCH_STYLE, FIG_SINGLE, FIG_DOUBLE, FIG_TRIPLE
from sims.sim_cache import load_results, save_results, add_cache_args
apply_style()
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(OUTPUT_DIR, 'cache')
SCRIPT_NAME = 'confounded_ope'

# MDP parameters
N_STATES = 5          # states 0..4; state 4 is absorbing (converted)
N_ACTIONS = 2         # 0 = promote, 1 = hold price
GAMMA = 0.9           # discount factor

# Structural equation parameters
P_Z1 = 0.5            # P(Z=1), market conditions

# Unobserved confounder U (consumer sentiment), caused by Z
P_U1_GIVEN_Z1 = 0.9   # P(U=1 | Z=1)
P_U1_GIVEN_Z0 = 0.1   # P(U=1 | Z=0)

# Independent instrument: cost shock IV ~ Bernoulli(0.5)
P_IV1 = 0.5

# Mediator M (marketing effort), caused by A
P_M1_GIVEN_A0 = 0.8   # P(M=1 | A=0 promote)
P_M1_GIVEN_A1 = 0.2   # P(M=1 | A=1 hold)

# Proxies of U
P_W1_GIVEN_U1 = 0.85  # CRM score proxy
P_W1_GIVEN_U0 = 0.15
P_W2_GIVEN_U1 = 0.75  # browse behavior proxy
P_W2_GIVEN_U0 = 0.25

# Transition probabilities: depend on M and Z only (NOT A or U directly)
P_TRANS = {
    (1, 1): 0.90,  # P(s+1 | s, M=1, Z=1)
    (1, 0): 0.50,  # P(s+1 | s, M=1, Z=0)
    (0, 1): 0.40,  # P(s+1 | s, M=0, Z=1)
    (0, 0): 0.15,  # P(s+1 | s, M=0, Z=0)
}

# Rewards
STEP_COST = -1.0    # r(s, a) for s < 4
GOAL_REWARD = 0.0   # r(4, a)

# Behavioral policy parameters
MU_BASE = 0.55
MU_DELTA = 0.25
MU_IV_COEFF = 0.15   # IV effect on behavioral policy

# Experimental design
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 20
BASE_SEED = 42
N_TRAJ = 2000
T_MAX = 50

# Estimator names and colors
ESTIMATOR_NAMES = ["Oracle", "Naive", "Backdoor", "Front-door", "IV", "Proximal"]
EST_COLORS = {
    "Naive":      DOMAIN_COLORS['Naive'],
    "Backdoor":   DOMAIN_COLORS['Backdoor'],
    "Front-door": DOMAIN_COLORS['Front-door'],
    "IV":         DOMAIN_COLORS['IV'],
    "Proximal":   DOMAIN_COLORS['Proximal'],
}
EST_MARKERS = {
    "Naive":      "s",
    "Backdoor":   "^",
    "Front-door": "D",
    "IV":         "v",
    "Proximal":   "o",
}

CONFIG = {
    'n_states': N_STATES,
    'n_actions': N_ACTIONS,
    'gamma': GAMMA,
    'p_z1': P_Z1,
    'p_u1_given_z1': P_U1_GIVEN_Z1,
    'p_u1_given_z0': P_U1_GIVEN_Z0,
    'p_iv1': P_IV1,
    'p_m1_given_a0': P_M1_GIVEN_A0,
    'p_m1_given_a1': P_M1_GIVEN_A1,
    'p_w1_given_u1': P_W1_GIVEN_U1,
    'p_w1_given_u0': P_W1_GIVEN_U0,
    'p_w2_given_u1': P_W2_GIVEN_U1,
    'p_w2_given_u0': P_W2_GIVEN_U0,
    'p_trans': {f"{k}": v for k, v in P_TRANS.items()},
    'step_cost': STEP_COST,
    'goal_reward': GOAL_REWARD,
    'mu_base': MU_BASE,
    'mu_delta': MU_DELTA,
    'mu_iv_coeff': MU_IV_COEFF,
    'rho_values': RHO_VALUES,
    'n_seeds': N_SEEDS,
    'base_seed': BASE_SEED,
    'n_traj': N_TRAJ,
    't_max': T_MAX,
    'iv_strengths': [0.05, 0.15, 0.30],
    'version': 1,
}


# ============================================================
# Oracle: analytical computation of V^pi under target policy
# ============================================================

def compute_true_transition_promote():
    """Compute P(s+1 | s, do(promote)).

    Since A affects S' only through M, and Z is independent:
    P(s+1|s, do(a)) = sum_{m,z} P(s+1|s,m,z) * P(m|a) * P(z)
    """
    p_m1 = P_M1_GIVEN_A0  # P(M=1 | promote)
    p_m0 = 1 - p_m1
    p = (P_TRANS[(1, 1)] * p_m1 * P_Z1 +
         P_TRANS[(1, 0)] * p_m1 * (1 - P_Z1) +
         P_TRANS[(0, 1)] * p_m0 * P_Z1 +
         P_TRANS[(0, 0)] * p_m0 * (1 - P_Z1))
    return p


def compute_true_transition_hold():
    """Compute P(s+1 | s, do(hold))."""
    p_m1 = P_M1_GIVEN_A1  # P(M=1 | hold)
    p_m0 = 1 - p_m1
    p = (P_TRANS[(1, 1)] * p_m1 * P_Z1 +
         P_TRANS[(1, 0)] * p_m1 * (1 - P_Z1) +
         P_TRANS[(0, 1)] * p_m0 * P_Z1 +
         P_TRANS[(0, 0)] * p_m0 * (1 - P_Z1))
    return p


def compute_oracle_value():
    """Compute true V^pi(s) for target policy pi(promote|s)=1 for all s<4."""
    p_adv_true = compute_true_transition_promote()

    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES - 1):
        P_pi[s, s + 1] = p_adv_true
        P_pi[s, s] = 1.0 - p_adv_true
    P_pi[N_STATES - 1, N_STATES - 1] = 1.0

    r_pi = np.full(N_STATES, STEP_COST)
    r_pi[N_STATES - 1] = GOAL_REWARD

    V = np.linalg.solve(np.eye(N_STATES) - GAMMA * P_pi, r_pi)
    return V, p_adv_true


def solve_bellman_with_transitions(P_est):
    """Solve V = (I - gamma * P_pi)^{-1} r_pi given estimated transition matrix."""
    r_pi = np.full(N_STATES, STEP_COST)
    r_pi[N_STATES - 1] = GOAL_REWARD
    V = np.linalg.solve(np.eye(N_STATES) - GAMMA * P_est, r_pi)
    return V


# ============================================================
# Data generating process
# ============================================================

def generate_trajectories(rng, rho, n_traj=N_TRAJ, t_max=T_MAX, iv_coeff=None):
    """Generate trajectories from the confounded retail pricing MDP.

    Per-step structural equations:
      Z_t  ~ Bernoulli(P_Z1)                              market conditions
      U_t  ~ Bernoulli(P_U1|Z_t)                          consumer sentiment (LATENT)
      IV_t ~ Bernoulli(P_IV1)                              cost shock (independent)
      A_t  ~ mu(.|s, U, IV; rho)                           behavioral policy (confounded)
      M_t  ~ Bernoulli(P_M1|A_t)                           marketing effort (mediator)
      W1_t ~ Bernoulli(P_W1|U_t)                           CRM score proxy
      W2_t ~ Bernoulli(P_W2|U_t)                           browse behavior proxy
      S_{t+1} ~ P_trans(.|s, M_t, Z_t)                    transition (NO direct A or U)

    Args:
        rng: numpy random generator
        rho: confounding strength
        n_traj: number of trajectories
        t_max: max steps per trajectory
        iv_coeff: instrument strength coefficient (overrides MU_IV_COEFF if provided)

    Returns list of trajectories, each a list of
    (s, a, z, iv, m, w1, w2, s_next, r) tuples.
    """
    _iv_coeff = iv_coeff if iv_coeff is not None else MU_IV_COEFF
    trajectories = []
    for _ in range(n_traj):
        traj = []
        s = 0
        for t in range(t_max):
            if s == N_STATES - 1:
                break

            # Draw exogenous variables
            z = rng.binomial(1, P_Z1)
            p_u1 = P_U1_GIVEN_Z1 if z == 1 else P_U1_GIVEN_Z0
            u = rng.binomial(1, p_u1)
            iv = rng.binomial(1, P_IV1)

            # Behavioral policy: depends on U (confounded) and IV
            mu_promote = MU_BASE + rho * MU_DELTA * (2 * u - 1) + _iv_coeff * (iv - 0.5)
            mu_promote = np.clip(mu_promote, 0.01, 0.99)
            a = 0 if rng.random() < mu_promote else 1

            # Mediator: marketing effort depends on action
            p_m1 = P_M1_GIVEN_A0 if a == 0 else P_M1_GIVEN_A1
            m = rng.binomial(1, p_m1)

            # Proxies of U
            p_w1 = P_W1_GIVEN_U1 if u == 1 else P_W1_GIVEN_U0
            w1 = rng.binomial(1, p_w1)
            p_w2 = P_W2_GIVEN_U1 if u == 1 else P_W2_GIVEN_U0
            w2 = rng.binomial(1, p_w2)

            # Reward
            r = STEP_COST

            # Transition depends on M and Z only (NOT A or U)
            p_advance = P_TRANS[(m, z)]
            s_next = s + 1 if rng.random() < p_advance else s

            traj.append((s, a, z, iv, m, w1, w2, s_next, r))
            s = s_next

        trajectories.append(traj)
    return trajectories


def flatten_data(trajectories):
    """Flatten trajectories into arrays for vectorized estimation."""
    all_steps = []
    for traj in trajectories:
        for step in traj:
            all_steps.append(step)
    if len(all_steps) == 0:
        return None
    arr = np.array(all_steps, dtype=np.float64)
    return {
        's': arr[:, 0].astype(int),
        'a': arr[:, 1].astype(int),
        'z': arr[:, 2].astype(int),
        'iv': arr[:, 3].astype(int),
        'm': arr[:, 4].astype(int),
        'w1': arr[:, 5].astype(int),
        'w2': arr[:, 6].astype(int),
        's_next': arr[:, 7].astype(int),
        'r': arr[:, 8],
    }


# ============================================================
# Estimator 1: Naive OPE (biased)
# ============================================================

def naive_ope(data):
    """Estimate P_obs(s'|s,a) from counts, solve Bellman."""
    counts_sa = np.zeros((N_STATES, N_ACTIONS))
    counts_sas = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    for i in range(len(data['s'])):
        s, a, sn = data['s'][i], data['a'][i], data['s_next'][i]
        counts_sa[s, a] += 1
        counts_sas[s, a, sn] += 1

    P_obs = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if counts_sa[s, a] > 0:
                P_obs[s, a, :] = counts_sas[s, a, :] / counts_sa[s, a]
            else:
                P_obs[s, a, s] = 1.0

    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P_pi[s, :] = P_obs[s, 0, :]  # target: always promote (a=0)

    return solve_bellman_with_transitions(P_pi)


# ============================================================
# Estimator 2: Backdoor-adjusted OPE (condition on Z)
# ============================================================

def backdoor_ope(data):
    """P(s'|s,do(a)) = sum_z P(s'|s,a,z) P(z|s)."""
    counts_saz = np.zeros((N_STATES, N_ACTIONS, 2))
    counts_sazs = np.zeros((N_STATES, N_ACTIONS, 2, N_STATES))
    counts_s = np.zeros(N_STATES)
    counts_sz = np.zeros((N_STATES, 2))

    for i in range(len(data['s'])):
        s, a, z, sn = data['s'][i], data['a'][i], data['z'][i], data['s_next'][i]
        counts_saz[s, a, z] += 1
        counts_sazs[s, a, z, sn] += 1
        counts_s[s] += 1
        counts_sz[s, z] += 1

    P_saz = np.zeros((N_STATES, N_ACTIONS, 2, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for z in range(2):
                if counts_saz[s, a, z] > 0:
                    P_saz[s, a, z, :] = counts_sazs[s, a, z, :] / counts_saz[s, a, z]
                else:
                    P_saz[s, a, z, s] = 1.0

    P_z_given_s = np.zeros((N_STATES, 2))
    for s in range(N_STATES):
        if counts_s[s] > 0:
            P_z_given_s[s, :] = counts_sz[s, :] / counts_s[s]
        else:
            P_z_given_s[s, :] = 0.5

    P_bd = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for z in range(2):
                P_bd[s, a, :] += P_saz[s, a, z, :] * P_z_given_s[s, z]

    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P_pi[s, :] = P_bd[s, 0, :]

    return solve_bellman_with_transitions(P_pi)


# ============================================================
# Estimator 3: Front-door adjusted OPE (through mediator M)
# ============================================================

def frontdoor_ope(data):
    """P(s'|s,do(a)) = sum_m P(m|a) sum_{a'} P(s'|s,m,a') P(a'|s).

    Front-door criterion: A affects S' only through M.
    """
    counts_a = np.zeros(N_ACTIONS)
    counts_am = np.zeros((N_ACTIONS, 2))
    counts_s = np.zeros(N_STATES)
    counts_sa_fd = np.zeros((N_STATES, N_ACTIONS))
    counts_sma = np.zeros((N_STATES, 2, N_ACTIONS))
    counts_smas = np.zeros((N_STATES, 2, N_ACTIONS, N_STATES))

    for i in range(len(data['s'])):
        s, a, m, sn = data['s'][i], data['a'][i], data['m'][i], data['s_next'][i]
        counts_a[a] += 1
        counts_am[a, m] += 1
        counts_s[s] += 1
        counts_sa_fd[s, a] += 1
        counts_sma[s, m, a] += 1
        counts_smas[s, m, a, sn] += 1

    # P(m|a)
    P_m_given_a = np.zeros((N_ACTIONS, 2))
    for a in range(N_ACTIONS):
        if counts_a[a] > 0:
            P_m_given_a[a, :] = counts_am[a, :] / counts_a[a]
        else:
            P_m_given_a[a, :] = 0.5

    # P(a'|s)
    P_a_given_s = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        if counts_s[s] > 0:
            P_a_given_s[s, :] = counts_sa_fd[s, :] / counts_s[s]
        else:
            P_a_given_s[s, :] = 0.5

    # P(s'|s,m,a')
    P_sma = np.zeros((N_STATES, 2, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for m in range(2):
            for a in range(N_ACTIONS):
                if counts_sma[s, m, a] > 0:
                    P_sma[s, m, a, :] = counts_smas[s, m, a, :] / counts_sma[s, m, a]
                else:
                    P_sma[s, m, a, s] = 1.0

    # Front-door formula
    P_fd = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for m in range(2):
                # sum_{a'} P(s'|s,m,a') P(a'|s)
                inner = np.zeros(N_STATES)
                for a_prime in range(N_ACTIONS):
                    inner += P_sma[s, m, a_prime, :] * P_a_given_s[s, a_prime]
                P_fd[s, a, :] += P_m_given_a[a, m] * inner

    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P_pi[s, :] = P_fd[s, 0, :]

    return solve_bellman_with_transitions(P_pi)


# ============================================================
# Estimator 4: IV (Wald estimator using cost shock)
# ============================================================

def iv_ope(data):
    """Wald estimator using the cost shock IV.

    Linear structural model: P(s+1|s) = alpha + beta * 1{A=0} + eps
    where E[eps|IV] = 0. The Wald estimator recovers beta, then:
    P(s+1|s, do(a=0)) = P(s+1|s, IV=z) + beta * (1 - P(A=0|s, IV=z))
    """
    counts_s_iv = np.zeros((N_STATES, 2))
    counts_s_iv_advance = np.zeros((N_STATES, 2, N_STATES))
    counts_s_iv_a0 = np.zeros((N_STATES, 2))

    for i in range(len(data['s'])):
        s, a, iv, sn = data['s'][i], data['a'][i], data['iv'][i], data['s_next'][i]
        counts_s_iv[s, iv] += 1
        counts_s_iv_advance[s, iv, sn] += 1
        if a == 0:
            counts_s_iv_a0[s, iv] += 1

    P_pi = np.zeros((N_STATES, N_STATES))
    P_pi[N_STATES - 1, N_STATES - 1] = 1.0

    for s in range(N_STATES - 1):
        # Reduced form: P(s+1|s, IV=z)
        p_advance_iv1 = 0.0
        p_advance_iv0 = 0.0
        if counts_s_iv[s, 1] > 0:
            p_advance_iv1 = counts_s_iv_advance[s, 1, s + 1] / counts_s_iv[s, 1]
        if counts_s_iv[s, 0] > 0:
            p_advance_iv0 = counts_s_iv_advance[s, 0, s + 1] / counts_s_iv[s, 0]

        # First stage: P(A=0|s, IV=z)
        p_a0_iv1 = 0.0
        p_a0_iv0 = 0.0
        if counts_s_iv[s, 1] > 0:
            p_a0_iv1 = counts_s_iv_a0[s, 1] / counts_s_iv[s, 1]
        if counts_s_iv[s, 0] > 0:
            p_a0_iv0 = counts_s_iv_a0[s, 0] / counts_s_iv[s, 0]

        # Wald estimator: beta = reduced_form / first_stage
        first_stage = p_a0_iv1 - p_a0_iv0
        if abs(first_stage) > 0.01:
            reduced_form = p_advance_iv1 - p_advance_iv0
            beta = reduced_form / first_stage
            # Recover interventional: P(s+1|s,do(a=0)) using IV=0 as reference
            p_do = p_advance_iv0 + beta * (1.0 - p_a0_iv0)
            p_do = np.clip(p_do, 0.0, 1.0)
        else:
            # Weak instrument: fall back to naive
            total = counts_s_iv[s, 0] + counts_s_iv[s, 1]
            if total > 0:
                p_do = (counts_s_iv_advance[s, 0, s + 1] + counts_s_iv_advance[s, 1, s + 1]) / total
            else:
                p_do = 0.5

        P_pi[s, s + 1] = p_do
        P_pi[s, s] = 1.0 - p_do

    return solve_bellman_with_transitions(P_pi)


# ============================================================
# Estimator 5: Proximal causal inference (bridge function)
# ============================================================

def proximal_ope(data):
    """Discrete bridge function approach using W1 (treatment proxy) and W2 (outcome proxy).

    Solve for h(w2, s, a) from the conditional moment equation:
      E[1{S_{t+1}=s+1} | W1=w1, S=s, A=a] = E[h(W2, s, a) | W1=w1, S=s, A=a]

    Then: P(s+1|s, do(a)) = sum_{w2} h(w2) * P(W2=w2 | S=s)
    where P(W2|S) is the MARGINAL proxy distribution (not conditioned on A).
    """
    counts_w1sa = np.zeros((2, N_STATES, N_ACTIONS))
    counts_w1sa_advance = np.zeros((2, N_STATES, N_ACTIONS))
    counts_w1sa_w2 = np.zeros((2, N_STATES, N_ACTIONS, 2))
    counts_s_w2 = np.zeros((N_STATES, 2))
    counts_s_total = np.zeros(N_STATES)

    for i in range(len(data['s'])):
        s = data['s'][i]
        a = data['a'][i]
        w1 = data['w1'][i]
        w2 = data['w2'][i]
        sn = data['s_next'][i]

        counts_w1sa[w1, s, a] += 1
        if s < N_STATES - 1 and sn == s + 1:
            counts_w1sa_advance[w1, s, a] += 1
        counts_w1sa_w2[w1, s, a, w2] += 1
        counts_s_w2[s, w2] += 1
        counts_s_total[s] += 1

    P_pi = np.zeros((N_STATES, N_STATES))
    P_pi[N_STATES - 1, N_STATES - 1] = 1.0

    for s in range(N_STATES - 1):
        a = 0  # target action: promote
        # Build the 2x2 linear system for h(w2=0, s, a) and h(w2=1, s, a)
        # For each w1 in {0,1}:
        #   P(s+1|w1, s, a) = sum_{w2} P(w2|w1, s, a) * h(w2)
        A_mat = np.zeros((2, 2))
        b_vec = np.zeros(2)

        valid = True
        for w1 in range(2):
            n = counts_w1sa[w1, s, a]
            if n < 5:
                valid = False
                break
            b_vec[w1] = counts_w1sa_advance[w1, s, a] / n
            for w2 in range(2):
                A_mat[w1, w2] = counts_w1sa_w2[w1, s, a, w2] / n

        if valid and abs(np.linalg.det(A_mat)) > 1e-10:
            h = np.linalg.solve(A_mat, b_vec)
            # P(s+1|s, do(a)) = sum_{w2} h(w2) * P(W2=w2|S=s)
            # Use marginal P(W2|S), NOT P(W2|S,A) which is confounded
            if counts_s_total[s] > 0:
                P_w2_given_s = counts_s_w2[s, :] / counts_s_total[s]
            else:
                P_w2_given_s = np.array([0.5, 0.5])
            p_do = np.dot(h, P_w2_given_s)
            p_do = np.clip(p_do, 0.0, 1.0)
        else:
            # Fallback: use marginal
            total = counts_s_total[s]
            if total > 0:
                adv_count = sum(counts_w1sa_advance[w1, s, a] for w1 in range(2))
                p_do = adv_count / total
            else:
                p_do = 0.5

        P_pi[s, s + 1] = p_do
        P_pi[s, s] = 1.0 - p_do

    return solve_bellman_with_transitions(P_pi)


# ============================================================
# Population-level verification
# ============================================================

def print_population_verification(rho):
    """Print population-level transition probabilities for verification."""
    p_true_promote = compute_true_transition_promote()
    p_true_hold = compute_true_transition_hold()

    # Compute population observational P(s+1|s, a=0)
    # Need P(M=m, Z=z | A=0) which requires integrating over U and IV
    # P(A=0 | U=u, IV=iv) = base + rho*delta*(2u-1) + iv_coeff*(iv-0.5)
    # P(U=u, Z=z) = P(U=u|Z=z)*P(Z=z)
    # P(A=0, U=u, Z=z, IV=iv) = P(A=0|u,iv) * P(u|z) * P(z) * P(iv)

    p_advance_obs = 0.0
    p_a0_total = 0.0
    for z in range(2):
        p_z = P_Z1 if z == 1 else (1 - P_Z1)
        for u in range(2):
            p_u_z = (P_U1_GIVEN_Z1 if z == 1 else P_U1_GIVEN_Z0) if u == 1 else \
                    (1 - P_U1_GIVEN_Z1 if z == 1 else 1 - P_U1_GIVEN_Z0)
            for iv in range(2):
                p_iv = P_IV1 if iv == 1 else (1 - P_IV1)
                mu_promote = MU_BASE + rho * MU_DELTA * (2 * u - 1) + MU_IV_COEFF * (iv - 0.5)
                mu_promote = np.clip(mu_promote, 0.01, 0.99)
                # Joint probability of this (z,u,iv) config
                p_joint = p_z * p_u_z * p_iv
                p_a0 = mu_promote * p_joint
                p_a0_total += p_a0
                # P(s+1|s, A=0, Z=z, U=u, IV=iv) = sum_m P(s+1|s,m,z) * P(m|A=0)
                for m in range(2):
                    p_m = P_M1_GIVEN_A0 if m == 1 else (1 - P_M1_GIVEN_A0)
                    p_advance_obs += p_a0 * p_m * P_TRANS[(m, z)]

    p_obs = p_advance_obs / p_a0_total if p_a0_total > 0 else 0.0

    print(f"  rho = {rho:.1f}")
    print(f"    True interventional P(s+1|s,do(promote)) = {p_true_promote:.4f}")
    print(f"    True interventional P(s+1|s,do(hold))    = {p_true_hold:.4f}")
    print(f"    Naive observational P(s+1|s,A=promote)   = {p_obs:.4f}  (bias = {p_obs - p_true_promote:+.4f})")
    print()


# ============================================================
# compute_data
# ============================================================

def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    print("=" * 72)
    print("Unified Identification: Confounded Retail Pricing MDP")
    print("Chapter 10: Causal Inference and RL")
    print("=" * 72)
    print()
    print("MDP Parameters:")
    print(f"  States                    = {N_STATES} (0..{N_STATES-1}, state {N_STATES-1} absorbing)")
    print(f"  Actions                   = promote (0), hold price (1)")
    print(f"  Discount gamma            = {GAMMA}")
    print(f"  Step cost r(s<4,a)        = {STEP_COST}")
    print(f"  Goal reward r(4,a)        = {GOAL_REWARD}")
    print()
    print("Transition probabilities (depend on M and Z only):")
    for (m, z), p in sorted(P_TRANS.items()):
        print(f"  P(s+1|s, M={m}, Z={z}) = {p}")
    print()
    print("Structural equations:")
    print(f"  Z_t  ~ Bernoulli({P_Z1})")
    print(f"  U_t  | Z=1 ~ Bernoulli({P_U1_GIVEN_Z1}), Z=0 ~ Bernoulli({P_U1_GIVEN_Z0})")
    print(f"  IV_t ~ Bernoulli({P_IV1})")
    print(f"  M_t  | A=0 ~ Bernoulli({P_M1_GIVEN_A0}), A=1 ~ Bernoulli({P_M1_GIVEN_A1})")
    print(f"  W1_t | U=1 ~ Bernoulli({P_W1_GIVEN_U1}), U=0 ~ Bernoulli({P_W1_GIVEN_U0})")
    print(f"  W2_t | U=1 ~ Bernoulli({P_W2_GIVEN_U1}), U=0 ~ Bernoulli({P_W2_GIVEN_U0})")
    print(f"  mu(promote|s,U,IV;rho) = {MU_BASE} + rho*{MU_DELTA}*(2U-1) + {MU_IV_COEFF}*(IV-0.5)")
    print()
    print("Causal Structure:")
    print("  Z -> U -> A, IV -> A            (Z observed, U unobserved, IV independent)")
    print("  A -> M -> S'                     (A affects S' ONLY through M)")
    print("  Z -> S' (through transition)     (market conditions affect transitions)")
    print("  U -> W1, U -> W2                 (proxies of latent confounder)")
    print()
    print("Experimental Design:")
    print(f"  rho values                = {RHO_VALUES}")
    print(f"  Seeds per rho             = {N_SEEDS}")
    print(f"  Trajectories per seed     = {N_TRAJ}")
    print(f"  Max steps per trajectory  = {T_MAX}")
    print()

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------
    V_oracle, p_adv_true = compute_oracle_value()
    print("Oracle (analytical):")
    print(f"  True P(s+1|s,do(promote)) = {p_adv_true:.4f}")
    for s in range(N_STATES):
        print(f"  V^pi(s={s}) = {V_oracle[s]:.4f}")
    print()

    # --------------------------------------------------------
    # Population-level verification
    # --------------------------------------------------------
    print("Population-level transition verification:")
    print("-" * 50)
    for rho in [0.0, 0.5, 1.0]:
        print_population_verification(rho)

    # --------------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------------
    results = {rho: {name: [] for name in ESTIMATOR_NAMES}
               for rho in RHO_VALUES}

    total_runs = len(RHO_VALUES) * N_SEEDS
    pbar = tqdm(total=total_runs, desc="Running experiments")

    for rho in RHO_VALUES:
        for seed_idx in range(N_SEEDS):
            seed = BASE_SEED + seed_idx
            rng = np.random.default_rng(seed)
            trajectories = generate_trajectories(rng, rho)
            data = flatten_data(trajectories)

            # Oracle
            results[rho]["Oracle"].append(V_oracle[0])

            # Naive
            V_naive = naive_ope(data)
            results[rho]["Naive"].append(V_naive[0])

            # Backdoor
            V_bd = backdoor_ope(data)
            results[rho]["Backdoor"].append(V_bd[0])

            # Front-door
            V_fd = frontdoor_ope(data)
            results[rho]["Front-door"].append(V_fd[0])

            # IV
            V_iv = iv_ope(data)
            results[rho]["IV"].append(V_iv[0])

            # Proximal
            V_prox = proximal_ope(data)
            results[rho]["Proximal"].append(V_prox[0])

            pbar.update(1)
    pbar.close()
    print()

    # --------------------------------------------------------
    # Compute bias and RMSE
    # --------------------------------------------------------
    summary = {}
    for rho in RHO_VALUES:
        summary[rho] = {}
        oracle_val = V_oracle[0]
        for name in ESTIMATOR_NAMES:
            vals = np.array(results[rho][name])
            biases = vals - oracle_val
            summary[rho][name] = {
                "mean": float(vals.mean()),
                "bias": float(biases.mean()),
                "se_bias": float(biases.std(ddof=1) / np.sqrt(N_SEEDS)),
                "rmse": float(np.sqrt((biases ** 2).mean())),
            }

    # --------------------------------------------------------
    # Print results tables
    # --------------------------------------------------------
    print("=" * 72)
    print("RESULTS: Estimated V^pi(s=0) (mean over seeds)")
    print("=" * 72)
    header = f"{'rho':>5s}"
    for name in ESTIMATOR_NAMES:
        header += f"  {name:>12s}"
    print(header)
    print("-" * len(header))
    for rho in RHO_VALUES:
        row = f"{rho:5.1f}"
        for name in ESTIMATOR_NAMES:
            row += f"  {summary[rho][name]['mean']:12.4f}"
        print(row)
    print()

    print("=" * 72)
    print("RESULTS: Bias (Estimator - Oracle), mean +/- SE")
    print("=" * 72)
    header = f"{'rho':>5s}"
    for name in ESTIMATOR_NAMES[1:]:
        header += f"  {name:>18s}"
    print(header)
    print("-" * len(header))
    for rho in RHO_VALUES:
        row = f"{rho:5.1f}"
        for name in ESTIMATOR_NAMES[1:]:
            b = summary[rho][name]["bias"]
            se = summary[rho][name]["se_bias"]
            row += f"  {b:+8.4f} ({se:.4f})"
        print(row)
    print()

    print("=" * 72)
    print("RESULTS: RMSE relative to Oracle")
    print("=" * 72)
    header = f"{'rho':>5s}"
    for name in ESTIMATOR_NAMES[1:]:
        header += f"  {name:>12s}"
    print(header)
    print("-" * len(header))
    for rho in RHO_VALUES:
        row = f"{rho:5.1f}"
        for name in ESTIMATOR_NAMES[1:]:
            row += f"  {summary[rho][name]['rmse']:12.4f}"
        print(row)
    print()

    # --------------------------------------------------------
    # IV Strength Experiment: variance vs instrument strength
    # --------------------------------------------------------
    print("=" * 72)
    print("IV STRENGTH EXPERIMENT: Bias distribution vs instrument strength")
    print("  Fixed rho = 1.0 (maximum confounding)")
    print("=" * 72)
    print()

    IV_STRENGTHS = [0.05, 0.15, 0.30]
    IV_LABELS = ["Weak\n(0.05)", "Moderate\n(0.15)", "Strong\n(0.30)"]
    IV_N_SEEDS = 20
    IV_N_TRAJ = N_TRAJ
    IV_RHO = 1.0

    # Also compute naive bias at rho=1.0 for reference line
    naive_bias_rho1 = summary[1.0]["Naive"]["bias"]

    iv_strength_biases = {coeff: [] for coeff in IV_STRENGTHS}
    iv_strength_first_stage = {coeff: [] for coeff in IV_STRENGTHS}

    for coeff in IV_STRENGTHS:
        for seed_idx in range(IV_N_SEEDS):
            seed = BASE_SEED + seed_idx
            rng = np.random.default_rng(seed)
            trajectories = generate_trajectories(rng, IV_RHO, n_traj=IV_N_TRAJ, iv_coeff=coeff)
            data = flatten_data(trajectories)

            # Run IV estimator
            V_iv = iv_ope(data)
            bias = V_iv[0] - V_oracle[0]
            iv_strength_biases[coeff].append(bias)

            # Compute average first-stage coefficient: P(A=0|IV=1) - P(A=0|IV=0)
            iv_vals = data['iv']
            a_vals = data['a']
            mask_iv1 = iv_vals == 1
            mask_iv0 = iv_vals == 0
            p_a0_iv1 = np.mean(a_vals[mask_iv1] == 0) if mask_iv1.sum() > 0 else 0.0
            p_a0_iv0 = np.mean(a_vals[mask_iv0] == 0) if mask_iv0.sum() > 0 else 0.0
            iv_strength_first_stage[coeff].append(p_a0_iv1 - p_a0_iv0)

    print(f"{'IV Coeff':>10s}  {'Mean Bias':>10s}  {'SE':>8s}  {'RMSE':>8s}  {'First-Stage':>12s}")
    print("-" * 56)
    for coeff in IV_STRENGTHS:
        biases_arr = np.array(iv_strength_biases[coeff])
        fs_arr = np.array(iv_strength_first_stage[coeff])
        mean_bias = biases_arr.mean()
        se_bias = biases_arr.std(ddof=1) / np.sqrt(IV_N_SEEDS)
        rmse = np.sqrt((biases_arr ** 2).mean())
        mean_fs = fs_arr.mean()
        print(f"{coeff:10.2f}  {mean_bias:+10.4f}  {se_bias:8.4f}  {rmse:8.4f}  {mean_fs:+12.4f}")
    print()
    print(f"  Naive estimator bias at rho=1.0 (reference): {naive_bias_rho1:+.4f}")
    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Total configurations: {len(RHO_VALUES)} rho x {N_SEEDS} seeds = {total_runs} runs")
    print()
    print("Validation at rho=0.0 (no confounding):")
    for name in ESTIMATOR_NAMES:
        m = summary[0.0][name]["mean"]
        b = summary[0.0][name]["bias"]
        print(f"  {name:>12s}: mean={m:.4f}, bias={b:+.4f}")
    print()
    print("Bias at rho=1.0 (full confounding):")
    for name in ESTIMATOR_NAMES[1:]:
        b = summary[1.0][name]["bias"]
        se = summary[1.0][name]["se_bias"]
        rmse = summary[1.0][name]["rmse"]
        print(f"  {name:>12s}: bias={b:+.4f} (SE {se:.4f}), RMSE={rmse:.4f}")
    print()

    # --------------------------------------------------------
    # Pack data for caching
    # --------------------------------------------------------
    # Convert results dict with float rho keys to str keys for JSON serialization
    results_serializable = {}
    for rho in RHO_VALUES:
        results_serializable[str(rho)] = {name: vals for name, vals in results[rho].items()}

    summary_serializable = {}
    for rho in RHO_VALUES:
        summary_serializable[str(rho)] = summary[rho]

    iv_biases_serializable = {str(k): v for k, v in iv_strength_biases.items()}
    iv_fs_serializable = {str(k): v for k, v in iv_strength_first_stage.items()}

    cache_data = {
        'V_oracle': V_oracle.tolist(),
        'p_adv_true': float(p_adv_true),
        'results': results_serializable,
        'summary': summary_serializable,
        'iv_strength_biases': iv_biases_serializable,
        'iv_strength_first_stage': iv_fs_serializable,
        'naive_bias_rho1': float(naive_bias_rho1),
    }

    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, cache_data)
    return cache_data


# ============================================================
# generate_outputs
# ============================================================

def generate_outputs(data):
    V_oracle = np.array(data['V_oracle'])
    naive_bias_rho1 = data['naive_bias_rho1']

    # Reconstruct summary with float rho keys
    summary = {}
    for rho_str, val in data['summary'].items():
        summary[float(rho_str)] = val

    # Reconstruct iv_strength_biases with float keys
    iv_strength_biases = {}
    for k_str, v in data['iv_strength_biases'].items():
        iv_strength_biases[float(k_str)] = v

    IV_STRENGTHS = [0.05, 0.15, 0.30]

    # --------------------------------------------------------
    # Combined figure: bias vs rho (left) + RMSE vs rho (center) + IV strength box plot (right)
    # --------------------------------------------------------
    fig, (ax, ax_rmse, ax_iv) = plt.subplots(1, 3, figsize=FIG_TRIPLE)

    # Left panel: bias vs confounding strength
    for name in ESTIMATOR_NAMES[1:]:
        biases = np.array([summary[rho][name]["bias"] for rho in RHO_VALUES])
        ses = np.array([summary[rho][name]["se_bias"] for rho in RHO_VALUES])
        ax.plot(RHO_VALUES, biases, color=EST_COLORS[name], marker=EST_MARKERS[name],
                label=name, markersize=7, zorder=3)
        ax.fill_between(RHO_VALUES, biases - 1.96 * ses, biases + 1.96 * ses,
                        color=EST_COLORS[name], alpha=0.15, zorder=2)
    ax.axhline(0, **BENCH_STYLE, label="Oracle (zero bias)")
    ax.set_xlabel(r"Confounding strength $\rho$")
    ax.set_ylabel(r"Bias: $\hat{V}^{\pi}(s_0) - V^{\pi}(s_0)$")
    ax.legend(loc="best", framealpha=0.9, fontsize=7)
    ax.set_xticks(RHO_VALUES)
    ax.set_title("(a) Bias vs. confounding strength")

    # Center panel: RMSE vs confounding strength
    for name in ESTIMATOR_NAMES[1:]:
        rmses = np.array([summary[rho][name]["rmse"] for rho in RHO_VALUES])
        ax_rmse.plot(RHO_VALUES, rmses, color=EST_COLORS[name], marker=EST_MARKERS[name],
                     label=name, markersize=7, zorder=3)
    ax_rmse.axhline(0, **BENCH_STYLE, label="Oracle (zero RMSE)")
    ax_rmse.set_xlabel(r"Confounding strength $\rho$")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.legend(loc="best", framealpha=0.9, fontsize=7)
    ax_rmse.set_xticks(RHO_VALUES)
    ax_rmse.set_title("(b) RMSE vs. confounding strength")

    # Right panel: IV strength box plot
    box_data = [iv_strength_biases[coeff] for coeff in IV_STRENGTHS]
    positions = [1, 2, 3]

    bp = ax_iv.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                       showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                       markeredgecolor='black', markersize=6),
                       medianprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=4, alpha=0.5))

    box_colors = [COLORS['orange'], COLORS['blue'], COLORS['green']]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_iv.axhline(0, **BENCH_STYLE, label="Oracle (zero bias)")
    ax_iv.axhline(naive_bias_rho1, color=COLORS['red'], linestyle="--", linewidth=1.2,
                  zorder=1, label=f"Naive bias at $\\rho=1$ ({naive_bias_rho1:+.2f})")
    ax_iv.set_xticks(positions)
    ax_iv.set_xticklabels([f"{c}" for c in IV_STRENGTHS])
    ax_iv.set_xlabel("Instrument strength (IV coefficient)")
    ax_iv.set_ylabel(r"Bias: $\hat{V}^{\pi}(s_0) - V^{\pi}(s_0)$")
    ax_iv.legend(loc="best", framealpha=0.9, fontsize=7)
    ax_iv.set_title("(c) IV variance vs. instrument strength")

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "confounded_ope_bias.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # --------------------------------------------------------
    # LaTeX table
    # --------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "confounded_ope_results.tex")
    with open(tex_path, "w") as f:
        ncols = len(ESTIMATOR_NAMES) - 1
        col_spec = "c" + "cc" * ncols
        f.write("\\small\n")
        f.write("\\begin{tabular}{" + col_spec + "}\n")
        f.write("\\toprule\n")
        f.write("& ")
        parts = []
        for name in ESTIMATOR_NAMES[1:]:
            parts.append(f"\\multicolumn{{2}}{{c}}{{{name}}}")
        f.write(" & ".join(parts))
        f.write(" \\\\\n")
        cmidrule_parts = []
        for i in range(ncols):
            cs = 2 + 2 * i
            cmidrule_parts.append(f"\\cmidrule(lr){{{cs}-{cs + 1}}}")
        f.write(" ".join(cmidrule_parts) + "\n")
        f.write("$\\rho$")
        for _ in ESTIMATOR_NAMES[1:]:
            f.write(" & Bias (SE) & RMSE")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        for rho in RHO_VALUES:
            f.write(f"{rho:.1f}")
            for name in ESTIMATOR_NAMES[1:]:
                b = summary[rho][name]["bias"]
                se = summary[rho][name]["se_bias"]
                rmse = summary[rho][name]["rmse"]
                f.write(f" & ${b:+.3f}$ (${se:.3f}$) & ${rmse:.3f}$")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Table saved: {tex_path}")

    print()
    print("Output files:")
    print(f"  {fig_path}")
    print(f"  {tex_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
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
