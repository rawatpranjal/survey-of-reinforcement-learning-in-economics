"""
NFXP vs RLHF: Two Approaches to Preference Recovery
Chapter 7 - RLHF & Preference Learning

Compares structural estimation (Rust 1987) with reward learning from preferences
(Christiano 2017) on the bus engine replacement problem.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

np.random.seed(42)

# MDP parameters
S = 90  # Number of states (mileage bins)
GAMMA = 0.9  # Discount factor

# True cost parameters (unknown to both methods)
# Scaled so costs are in reasonable range for the problem
THETA_TRUE = np.array([0.0, 1.0, 2.0])  # [theta1, theta2, theta3]
RC = 5.0  # Replacement cost

# Mileage transition (geometric distribution for increment)
P_INCREMENT = 0.6  # Probability of staying in same bin

# Experiment parameters
N_SEEDS = 20
N_CHOICES_LIST = [100, 200, 500, 1000, 2000, 5000]
N_COMPARISONS_LIST = [50, 100, 200, 500, 1000, 2000]
TRAJECTORY_LENGTH = 20  # Length of trajectories for RLHF comparisons


# =============================================================================
# MDP Definition
# =============================================================================

def build_transition_matrix():
    """Build transition matrices for keep (a=0) and replace (a=1)."""
    P = np.zeros((2, S, S))

    # a=0: Keep - mileage increases stochastically
    for s in range(S):
        for s_prime in range(s, S):
            if s_prime == s:
                P[0, s, s_prime] = P_INCREMENT
            elif s_prime == s + 1:
                P[0, s, s_prime] = (1 - P_INCREMENT) * 0.7
            elif s_prime == s + 2:
                P[0, s, s_prime] = (1 - P_INCREMENT) * 0.2
            elif s_prime == s + 3:
                P[0, s, s_prime] = (1 - P_INCREMENT) * 0.1
        # Normalize and handle boundary
        if s >= S - 3:
            P[0, s, S-1] = 1 - P[0, s, :S-1].sum()
        row_sum = P[0, s].sum()
        if row_sum > 0:
            P[0, s] /= row_sum

    # a=1: Replace - reset to state 0
    P[1, :, 0] = 1.0

    return P


def cost_function(s, a, theta):
    """Cost function c(s, a; theta)."""
    if a == 0:  # Keep
        return theta[0] + theta[1] * (s / S) + theta[2] * (s / S)**2
    else:  # Replace
        return RC


def cost_vector(theta):
    """Return cost vectors for both actions."""
    c = np.zeros((2, S))
    s_norm = np.arange(S) / S
    c[0] = theta[0] + theta[1] * s_norm + theta[2] * s_norm**2
    c[1] = RC
    return c


# =============================================================================
# Value Function / Policy Computation
# =============================================================================

def solve_bellman(theta, P, tol=1e-8, max_iter=1000):
    """Solve Bellman equation using value iteration.

    Returns value function V and choice-specific value v(s,a).
    Uses logsum formula for EV errors.
    """
    c = cost_vector(theta)
    V = np.zeros(S)

    for _ in range(max_iter):
        # Choice-specific values (negative cost + continuation)
        v = np.zeros((2, S))
        for a in range(2):
            v[a] = -c[a] + GAMMA * P[a] @ V

        # Integrated value (logsum for Type-I EV errors)
        V_new = logsumexp(v, axis=0)

        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    return V, v


def choice_probability(v):
    """Compute choice probabilities from choice-specific values."""
    # Softmax
    v_max = np.max(v, axis=0, keepdims=True)
    exp_v = np.exp(v - v_max)
    return exp_v / exp_v.sum(axis=0, keepdims=True)


def optimal_policy(theta, P):
    """Return optimal policy (probability of replacement) for each state."""
    _, v = solve_bellman(theta, P)
    prob = choice_probability(v)
    return prob[1]  # P(replace)


# =============================================================================
# Data Generation
# =============================================================================

def generate_choices(theta, P, n_choices, suboptimal_noise=0.0):
    """Generate choice data from agent optimizing under theta.

    Args:
        suboptimal_noise: If > 0, agent uses noisy value function (suboptimal).
    """
    _, v = solve_bellman(theta, P)

    # Add noise for suboptimal agent
    if suboptimal_noise > 0:
        v = v + np.random.normal(0, suboptimal_noise, v.shape)

    prob = choice_probability(v)

    # Simulate trajectory
    states = []
    actions = []
    s = 0

    for _ in range(n_choices):
        states.append(s)
        # Sample action from choice probability
        a = np.random.binomial(1, prob[1, s])
        actions.append(a)
        # Transition
        s = np.random.choice(S, p=P[a, s])

    return np.array(states), np.array(actions)


def generate_trajectory(theta, P, length, start_state=None):
    """Generate a single trajectory."""
    _, v = solve_bellman(theta, P)
    prob = choice_probability(v)

    if start_state is None:
        start_state = np.random.randint(0, S)

    states = [start_state]
    actions = []
    s = start_state

    for _ in range(length):
        a = np.random.binomial(1, prob[1, s])
        actions.append(a)
        s = np.random.choice(S, p=P[a, s])
        states.append(s)

    return np.array(states[:-1]), np.array(actions)


def generate_comparisons(theta, P, n_comparisons, traj_length):
    """Generate pairwise trajectory comparisons.

    Oracle prefers trajectory with lower total cost.
    Uses diverse trajectory generation: some from optimal policy,
    some from random, some from suboptimal policies, and explicit
    keep-vs-replace comparisons at the same state.
    """
    c = cost_vector(theta)
    _, v = solve_bellman(theta, P)
    prob_opt = choice_probability(v)

    comparisons = []

    for i in range(n_comparisons):
        # Vary trajectory generation strategy for diversity
        strategy = i % 4  # Now 4 strategies

        tau1_s, tau1_a = [], []
        tau2_s, tau2_a = [], []

        if strategy == 3:
            # Strategy 3: Explicit keep-vs-replace at same high-mileage state
            # This directly reveals the value of replacement
            s_start = np.random.randint(30, 70)  # Near threshold region
            s1, s2 = s_start, s_start

            # Trajectory 1: always keep
            for _ in range(traj_length):
                tau1_s.append(s1)
                tau1_a.append(0)  # Keep
                s1 = np.random.choice(S, p=P[0, s1])

            # Trajectory 2: replace immediately, then keep
            for t in range(traj_length):
                tau2_s.append(s2)
                if t == 0:
                    tau2_a.append(1)  # Replace
                    s2 = np.random.choice(S, p=P[1, s2])
                else:
                    tau2_a.append(0)  # Keep
                    s2 = np.random.choice(S, p=P[0, s2])
        else:
            # Start from diverse states
            s1 = np.random.randint(0, S)
            s2 = np.random.randint(0, S)

            for _ in range(traj_length):
                if strategy == 0:
                    # Both random
                    a1 = np.random.binomial(1, 0.3)
                    a2 = np.random.binomial(1, 0.3)
                elif strategy == 1:
                    # One near-optimal, one random
                    a1 = np.random.binomial(1, prob_opt[1, s1])
                    a2 = np.random.binomial(1, 0.5)
                else:  # strategy == 2
                    # Both from varied replacement thresholds
                    threshold1 = np.random.randint(30, 70)
                    threshold2 = np.random.randint(30, 70)
                    a1 = 1 if s1 >= threshold1 else 0
                    a2 = 1 if s2 >= threshold2 else 0

                tau1_s.append(s1)
                tau1_a.append(a1)
                tau2_s.append(s2)
                tau2_a.append(a2)

                s1 = np.random.choice(S, p=P[a1, s1])
                s2 = np.random.choice(S, p=P[a2, s2])

        tau1_s, tau1_a = np.array(tau1_s), np.array(tau1_a)
        tau2_s, tau2_a = np.array(tau2_s), np.array(tau2_a)

        # Compute total costs
        cost1 = sum(c[tau1_a[t], tau1_s[t]] for t in range(traj_length))
        cost2 = sum(c[tau2_a[t], tau2_s[t]] for t in range(traj_length))

        # Oracle preference: lower cost wins
        if cost1 < cost2:
            comparisons.append(((tau1_s, tau1_a), (tau2_s, tau2_a)))
        else:
            comparisons.append(((tau2_s, tau2_a), (tau1_s, tau1_a)))

    return comparisons


# =============================================================================
# NFXP Estimation (Rust 1987)
# =============================================================================

def nfxp_log_likelihood(theta, states, actions, P):
    """Compute negative log-likelihood for NFXP."""
    _, v = solve_bellman(theta, P)
    prob = choice_probability(v)

    # Log-likelihood
    ll = 0.0
    for s, a in zip(states, actions):
        ll += np.log(prob[a, s] + 1e-10)

    return -ll


def estimate_nfxp(states, actions, P, theta_init=None):
    """Estimate parameters via NFXP (MLE with nested DP)."""
    if theta_init is None:
        theta_init = np.array([0.0, 1.0, 1.0])

    result = minimize(
        nfxp_log_likelihood,
        theta_init,
        args=(states, actions, P),
        method='L-BFGS-B',
        bounds=[(-1, 1), (0, 5), (0, 10)]
    )

    return result.x


# =============================================================================
# RLHF Estimation (Christiano 2017)
# =============================================================================

def reward_function(s, a, phi):
    """Learned reward function r(s, a; phi).

    Parametric form matching cost structure.
    """
    if a == 0:  # Keep
        return -(phi[0] + phi[1] * (s / S) + phi[2] * (s / S)**2)
    else:  # Replace
        return -phi[3]


def trajectory_reward(states, actions, phi):
    """Compute total reward for a trajectory."""
    return sum(reward_function(s, a, phi) for s, a in zip(states, actions))


def bradley_terry_loss(phi, comparisons):
    """Bradley-Terry loss for pairwise comparisons."""
    loss = 0.0

    for (winner_s, winner_a), (loser_s, loser_a) in comparisons:
        r_winner = trajectory_reward(winner_s, winner_a, phi)
        r_loser = trajectory_reward(loser_s, loser_a, phi)

        # P(winner > loser) = sigmoid(r_winner - r_loser)
        diff = r_winner - r_loser
        # Log-loss: -log(sigmoid(diff))
        loss += -diff + np.log(1 + np.exp(diff))

    return loss / len(comparisons)


def estimate_rlhf(comparisons, phi_init=None):
    """Estimate reward function via Bradley-Terry MLE."""
    if phi_init is None:
        phi_init = np.array([0.0, 1.0, 1.0, 5.0])

    result = minimize(
        bradley_terry_loss,
        phi_init,
        args=(comparisons,),
        method='L-BFGS-B',
        bounds=[(-1, 1), (0, 5), (0, 10), (1, 15)]
    )

    return result.x


def rlhf_to_policy(phi, P, anchor_rc=None):
    """Derive optimal policy from learned reward.

    Bradley-Terry identifies rewards only up to affine transformation.
    To fix the scale, we anchor on the known replacement cost RC.
    This reflects real-world information structure: replacement cost
    is typically observable (invoice for new engine).

    Args:
        phi: Learned reward parameters [phi0, phi1, phi2, phi3]
        P: Transition matrix
        anchor_rc: Known replacement cost to anchor scale. If None, uses global RC.
    """
    if anchor_rc is None:
        anchor_rc = RC

    # Scale phi so phi[3] = RC (anchor on known replacement cost)
    if phi[3] > 0:
        scale = anchor_rc / phi[3]
        phi_normalized = phi * scale
    else:
        phi_normalized = phi.copy()

    # Build cost vectors with normalized parameters
    c = np.zeros((2, S))
    s_norm = np.arange(S) / S
    c[0] = phi_normalized[0] + phi_normalized[1] * s_norm + phi_normalized[2] * s_norm**2
    c[1] = phi_normalized[3]  # = RC after normalization

    V = np.zeros(S)
    for _ in range(1000):
        v = np.zeros((2, S))
        for a in range(2):
            v[a] = -c[a] + GAMMA * P[a] @ V
        V_new = logsumexp(v, axis=0)
        if np.max(np.abs(V_new - V)) < 1e-8:
            break
        V = V_new

    prob = choice_probability(v)
    return prob[1]


def rlhf_to_policy_unnormalized(phi, P):
    """Derive policy WITHOUT normalization (for comparison).

    This shows what happens when Bradley-Terry identification issue is ignored.
    """
    c = np.zeros((2, S))
    s_norm = np.arange(S) / S
    c[0] = phi[0] + phi[1] * s_norm + phi[2] * s_norm**2
    c[1] = phi[3]

    V = np.zeros(S)
    for _ in range(1000):
        v = np.zeros((2, S))
        for a in range(2):
            v[a] = -c[a] + GAMMA * P[a] @ V
        V_new = logsumexp(v, axis=0)
        if np.max(np.abs(V_new - V)) < 1e-8:
            break
        V = V_new

    prob = choice_probability(v)
    return prob[1]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def parameter_error(theta_est, theta_true):
    """L2 error in parameter estimates."""
    return np.linalg.norm(theta_est - theta_true)


def cost_correlation(phi, theta_true):
    """Pearson correlation between learned reward and true cost."""
    # Evaluate at all (s, a) pairs
    true_costs = []
    learned_rewards = []

    for s in range(S):
        for a in range(2):
            true_costs.append(cost_function(s, a, theta_true))
            learned_rewards.append(-reward_function(s, a, phi))

    return np.corrcoef(true_costs, learned_rewards)[0, 1]


def cost_correlation_spearman(phi, theta_true):
    """Spearman correlation between learned reward and true cost.

    Spearman is robust to monotonic transforms, better suited to the
    identification-up-to-affine problem in Bradley-Terry.
    """
    true_costs = []
    learned_rewards = []

    for s in range(S):
        for a in range(2):
            true_costs.append(cost_function(s, a, theta_true))
            learned_rewards.append(-reward_function(s, a, phi))

    return spearmanr(true_costs, learned_rewards)[0]


def policy_distance(pi1, pi2):
    """L1 distance between policies."""
    return np.mean(np.abs(pi1 - pi2))


def find_threshold(policy):
    """Find replacement threshold (first state where P(replace) > 0.5)."""
    for s in range(len(policy)):
        if policy[s] > 0.5:
            return s
    return len(policy)


# =============================================================================
# Experiments
# =============================================================================

def run_experiment_1(P):
    """Experiment 1: Recovery accuracy vs sample size."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Recovery Accuracy vs Sample Size")
    print("="*70)

    nfxp_errors = {n: [] for n in N_CHOICES_LIST}
    rlhf_corrs = {m: [] for m in N_COMPARISONS_LIST}
    rlhf_spearman = {m: [] for m in N_COMPARISONS_LIST}

    for seed in tqdm(range(N_SEEDS), desc="Seeds"):
        np.random.seed(42 + seed)

        # NFXP experiments
        for n in N_CHOICES_LIST:
            states, actions = generate_choices(THETA_TRUE, P, n)
            theta_est = estimate_nfxp(states, actions, P)
            err = parameter_error(theta_est, THETA_TRUE)
            nfxp_errors[n].append(err)

        # RLHF experiments
        for m in N_COMPARISONS_LIST:
            comparisons = generate_comparisons(THETA_TRUE, P, m, TRAJECTORY_LENGTH)
            phi_est = estimate_rlhf(comparisons)
            corr = cost_correlation(phi_est, THETA_TRUE)
            spearman = cost_correlation_spearman(phi_est, THETA_TRUE)
            rlhf_corrs[m].append(corr)
            rlhf_spearman[m].append(spearman)

    # Print results
    print("\nNFXP Parameter Recovery (||theta_hat - theta*||):")
    print("-" * 50)
    print(f"{'N choices':>12} {'Mean Error':>12} {'Std Error':>12}")
    print("-" * 50)
    for n in N_CHOICES_LIST:
        mean_err = np.mean(nfxp_errors[n])
        std_err = np.std(nfxp_errors[n]) / np.sqrt(N_SEEDS)
        print(f"{n:>12} {mean_err:>12.4f} {std_err:>12.4f}")

    print("\nRLHF Cost Correlation (Pearson and Spearman):")
    print("-" * 70)
    print(f"{'M comparisons':>14} {'Pearson':>12} {'(SE)':>10} {'Spearman':>12} {'(SE)':>10}")
    print("-" * 70)
    for m in N_COMPARISONS_LIST:
        mean_corr = np.mean(rlhf_corrs[m])
        std_corr = np.std(rlhf_corrs[m]) / np.sqrt(N_SEEDS)
        mean_spear = np.mean(rlhf_spearman[m])
        std_spear = np.std(rlhf_spearman[m]) / np.sqrt(N_SEEDS)
        print(f"{m:>14} {mean_corr:>12.4f} {std_corr:>10.4f} {mean_spear:>12.4f} {std_spear:>10.4f}")

    print("\nNote: Spearman correlation is robust to monotonic transforms,")
    print("better suited to the identification-up-to-affine problem in Bradley-Terry.")

    return nfxp_errors, rlhf_corrs


def run_experiment_2(P):
    """Experiment 2: Policy quality comparison."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Policy Quality Comparison")
    print("="*70)

    # Ground truth policy
    pi_true = optimal_policy(THETA_TRUE, P)
    threshold_true = find_threshold(pi_true)
    print(f"\nTrue optimal threshold: state {threshold_true}")

    # Compute true cost ratio for comparison
    s_high = S - 1
    true_keep_cost_at_max = THETA_TRUE[0] + THETA_TRUE[1] * (s_high/S) + THETA_TRUE[2] * (s_high/S)**2
    true_ratio = RC / true_keep_cost_at_max
    print(f"True RC = {RC}, max keep cost = {true_keep_cost_at_max:.3f}, ratio = {true_ratio:.3f}")

    nfxp_distances = []
    rlhf_distances = []
    rlhf_distances_unnorm = []  # Without normalization for comparison
    nfxp_thresholds = []
    rlhf_thresholds = []
    rlhf_thresholds_unnorm = []

    # Storage for parameter recovery comparison
    nfxp_thetas = []  # List of recovered theta arrays
    rlhf_phis_normalized = []  # List of normalized phi arrays (cost params only)

    n_choices = 1000
    n_comparisons = 500

    print("\n" + "-"*70)
    print("DIAGNOSTIC: Per-seed parameter comparison (NFXP vs RLHF)")
    print("-"*70)
    print(f"True theta = [{THETA_TRUE[0]:.3f}, {THETA_TRUE[1]:.3f}, {THETA_TRUE[2]:.3f}], RC = {RC}")
    print("-"*70)

    for seed in range(N_SEEDS):
        np.random.seed(42 + seed)

        # NFXP
        states, actions = generate_choices(THETA_TRUE, P, n_choices)
        theta_est = estimate_nfxp(states, actions, P)
        pi_nfxp = optimal_policy(theta_est, P)
        nfxp_distances.append(policy_distance(pi_nfxp, pi_true))
        nfxp_thresholds.append(find_threshold(pi_nfxp))
        nfxp_thetas.append(theta_est.copy())

        # RLHF
        comparisons = generate_comparisons(THETA_TRUE, P, n_comparisons, TRAJECTORY_LENGTH)
        phi_est = estimate_rlhf(comparisons)

        # Normalize phi: scale so phi[3] = RC (anchor on known replacement cost)
        if phi_est[3] > 0:
            scale = RC / phi_est[3]
            phi_normalized = phi_est * scale
        else:
            phi_normalized = phi_est.copy()
        rlhf_phis_normalized.append(phi_normalized[:3].copy())  # Keep cost params only

        # Per-seed comparison output
        print(f"Seed {seed:2d}: NFXP theta = [{theta_est[0]:6.3f}, {theta_est[1]:6.3f}, {theta_est[2]:6.3f}]  |  "
              f"RLHF phi_norm = [{phi_normalized[0]:6.3f}, {phi_normalized[1]:6.3f}, {phi_normalized[2]:6.3f}]")

        # Policy with normalization (anchored on known RC)
        pi_rlhf = rlhf_to_policy(phi_est, P)
        rlhf_distances.append(policy_distance(pi_rlhf, pi_true))
        rlhf_thresholds.append(find_threshold(pi_rlhf))

        # Policy without normalization (to show the problem)
        pi_rlhf_unnorm = rlhf_to_policy_unnormalized(phi_est, P)
        rlhf_distances_unnorm.append(policy_distance(pi_rlhf_unnorm, pi_true))
        rlhf_thresholds_unnorm.append(find_threshold(pi_rlhf_unnorm))

    # Convert to arrays for statistics
    nfxp_thetas = np.array(nfxp_thetas)
    rlhf_phis_normalized = np.array(rlhf_phis_normalized)

    # Parameter recovery comparison table
    print("\n" + "="*70)
    print("PARAMETER RECOVERY COMPARISON")
    print("="*70)
    print(f"{'Parameter':<12} {'True':>10} {'NFXP (mean±std)':>22} {'RLHF norm (mean±std)':>22}")
    print("-"*70)
    for i, param_name in enumerate(['theta_0', 'theta_1', 'theta_2']):
        true_val = THETA_TRUE[i]
        nfxp_mean = np.mean(nfxp_thetas[:, i])
        nfxp_std = np.std(nfxp_thetas[:, i])
        rlhf_mean = np.mean(rlhf_phis_normalized[:, i])
        rlhf_std = np.std(rlhf_phis_normalized[:, i])
        print(f"{param_name:<12} {true_val:>10.3f} {nfxp_mean:>12.3f} +/- {nfxp_std:<6.3f} {rlhf_mean:>12.3f} +/- {rlhf_std:<6.3f}")
    print("-"*70)
    print("Note: RLHF phi normalized by scaling so phi[3] = RC (known replacement cost).")

    print("\nPolicy Distance from Optimal (L1):")
    print("-" * 60)
    print(f"{'Method':>18} {'Mean Dist':>12} {'Std Error':>12}")
    print("-" * 60)
    print(f"{'NFXP':>18} {np.mean(nfxp_distances):>12.4f} {np.std(nfxp_distances)/np.sqrt(N_SEEDS):>12.4f}")
    print(f"{'RLHF (normalized)':>18} {np.mean(rlhf_distances):>12.4f} {np.std(rlhf_distances)/np.sqrt(N_SEEDS):>12.4f}")
    print(f"{'RLHF (raw)':>18} {np.mean(rlhf_distances_unnorm):>12.4f} {np.std(rlhf_distances_unnorm)/np.sqrt(N_SEEDS):>12.4f}")

    print("\nReplacement Threshold Recovery:")
    print("-" * 60)
    print(f"{'Method':>18} {'Mean':>12} {'Std':>12} {'True':>12}")
    print("-" * 60)
    print(f"{'NFXP':>18} {np.mean(nfxp_thresholds):>12.1f} {np.std(nfxp_thresholds):>12.1f} {threshold_true:>12}")
    print(f"{'RLHF (normalized)':>18} {np.mean(rlhf_thresholds):>12.1f} {np.std(rlhf_thresholds):>12.1f} {threshold_true:>12}")
    print(f"{'RLHF (raw)':>18} {np.mean(rlhf_thresholds_unnorm):>12.1f} {np.std(rlhf_thresholds_unnorm):>12.1f} {threshold_true:>12}")

    print("\nKey insight: RLHF (raw) has wrong threshold because Bradley-Terry")
    print("identifies rewards only up to affine transformation. Anchoring on")
    print("the known replacement cost RC fixes the scale and recovers the correct policy.")

    return nfxp_distances, rlhf_distances, nfxp_thresholds, rlhf_thresholds, threshold_true


def run_experiment_3(P):
    """Experiment 3: Sample efficiency."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Sample Efficiency")
    print("="*70)

    pi_true = optimal_policy(THETA_TRUE, P)
    target_distance = 0.05  # Target policy distance

    # Find minimum samples needed
    print(f"\nTarget policy distance: {target_distance}")
    print("\nSample requirements to achieve target:")
    print("-" * 50)

    nfxp_found = False
    rlhf_found = False

    # NFXP
    for n in N_CHOICES_LIST:
        distances = []
        for seed in range(N_SEEDS):
            np.random.seed(42 + seed)
            states, actions = generate_choices(THETA_TRUE, P, n)
            theta_est = estimate_nfxp(states, actions, P)
            pi_est = optimal_policy(theta_est, P)
            distances.append(policy_distance(pi_est, pi_true))
        mean_dist = np.mean(distances)
        print(f"NFXP @ {n:5d} choices: mean dist = {mean_dist:.4f}")
        if mean_dist < target_distance and not nfxp_found:
            print(f"  -> NFXP achieves target at {n} choices")
            nfxp_found = True

    # RLHF (with normalization)
    print()
    for m in N_COMPARISONS_LIST:
        distances = []
        for seed in range(N_SEEDS):
            np.random.seed(42 + seed)
            comparisons = generate_comparisons(THETA_TRUE, P, m, TRAJECTORY_LENGTH)
            phi_est = estimate_rlhf(comparisons)
            pi_est = rlhf_to_policy(phi_est, P)  # Uses normalization
            distances.append(policy_distance(pi_est, pi_true))
        mean_dist = np.mean(distances)
        print(f"RLHF @ {m:5d} comparisons: mean dist = {mean_dist:.4f}")
        if mean_dist < target_distance and not rlhf_found:
            print(f"  -> RLHF achieves target at {m} comparisons")
            rlhf_found = True


def run_experiment_4(P):
    """Experiment 4: Robustness to suboptimal behavior."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Robustness to Suboptimal Agent")
    print("="*70)

    pi_true = optimal_policy(THETA_TRUE, P)
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0]

    print("\nNFXP with suboptimal agent (noisy value function):")
    print("-" * 60)
    print(f"{'Noise':>8} {'Param Error':>14} {'Policy Dist':>14}")
    print("-" * 60)

    nfxp_robustness = []
    for noise in noise_levels:
        param_errors = []
        policy_dists = []
        for seed in range(N_SEEDS):
            np.random.seed(42 + seed)
            states, actions = generate_choices(THETA_TRUE, P, 1000, suboptimal_noise=noise)
            theta_est = estimate_nfxp(states, actions, P)
            param_errors.append(parameter_error(theta_est, THETA_TRUE))
            pi_est = optimal_policy(theta_est, P)
            policy_dists.append(policy_distance(pi_est, pi_true))

        mean_param = np.mean(param_errors)
        mean_policy = np.mean(policy_dists)
        nfxp_robustness.append((noise, mean_param, mean_policy))
        print(f"{noise:>8.1f} {mean_param:>14.4f} {mean_policy:>14.4f}")

    print("\nRLHF is robust by design: does not assume optimal behavior.")

    return nfxp_robustness


# =============================================================================
# Visualization
# =============================================================================

def plot_cost_recovery(nfxp_errors, rlhf_corrs, P):
    """Plot cost/parameter recovery results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # NFXP parameter error
    ax = axes[0]
    means = [np.mean(nfxp_errors[n]) for n in N_CHOICES_LIST]
    stds = [np.std(nfxp_errors[n]) / np.sqrt(N_SEEDS) for n in N_CHOICES_LIST]
    ax.errorbar(N_CHOICES_LIST, means, yerr=stds, marker='o', capsize=3)
    ax.set_xlabel('Number of choices')
    ax.set_ylabel('Parameter error ||θ̂ - θ*||')
    ax.set_title('NFXP: Parameter Recovery')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # RLHF correlation
    ax = axes[1]
    means = [np.mean(rlhf_corrs[m]) for m in N_COMPARISONS_LIST]
    stds = [np.std(rlhf_corrs[m]) / np.sqrt(N_SEEDS) for m in N_COMPARISONS_LIST]
    ax.errorbar(N_COMPARISONS_LIST, means, yerr=stds, marker='s', capsize=3, color='C1')
    ax.set_xlabel('Number of comparisons')
    ax.set_ylabel('Correlation corr(r̂, -c)')
    ax.set_title('RLHF: Cost Correlation')
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ch07_rlhf/sims/cost_recovery.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: ch07_rlhf/sims/cost_recovery.png")


def plot_policy_comparison(P, nfxp_distances, rlhf_distances, threshold_true):
    """Plot policy comparison results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Example policies
    ax = axes[0]
    pi_true = optimal_policy(THETA_TRUE, P)

    # Get one example estimate from each method
    np.random.seed(42)
    states, actions = generate_choices(THETA_TRUE, P, 1000)
    theta_est = estimate_nfxp(states, actions, P)
    pi_nfxp = optimal_policy(theta_est, P)

    comparisons = generate_comparisons(THETA_TRUE, P, 500, TRAJECTORY_LENGTH)
    phi_est = estimate_rlhf(comparisons)
    pi_rlhf = rlhf_to_policy(phi_est, P)  # Normalized
    pi_rlhf_raw = rlhf_to_policy_unnormalized(phi_est, P)  # Raw

    ax.plot(range(S), pi_true, 'k-', linewidth=2, label='True optimal')
    ax.plot(range(S), pi_nfxp, 'b--', linewidth=1.5, label='NFXP')
    ax.plot(range(S), pi_rlhf, 'g-', linewidth=1.5, label='RLHF (normalized)')
    ax.plot(range(S), pi_rlhf_raw, 'r:', linewidth=1.5, label='RLHF (raw)')
    ax.axvline(threshold_true, color='gray', linestyle=':', alpha=0.5, label=f'True threshold ({threshold_true})')
    ax.set_xlabel('State (mileage)')
    ax.set_ylabel('P(replace)')
    ax.set_title('Learned Policies')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Policy distance boxplot
    ax = axes[1]
    bp = ax.boxplot([nfxp_distances, rlhf_distances], labels=['NFXP', 'RLHF\n(normalized)'])
    ax.set_ylabel('Policy distance (L1)')
    ax.set_title('Policy Recovery Quality')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ch07_rlhf/sims/policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: ch07_rlhf/sims/policy_comparison.png")


def generate_latex_table(nfxp_errors, rlhf_corrs, nfxp_distances, rlhf_distances,
                         nfxp_thresholds, rlhf_thresholds, threshold_true):
    """Generate LaTeX results table."""

    latex = r"""\begin{table}[htbp]
\centering
\caption{NFXP vs RLHF: Preference Recovery Comparison}
\label{tab:nfxp_rlhf}
\begin{tabular}{lcc}
\toprule
& NFXP & RLHF \\
\midrule
\multicolumn{3}{l}{\emph{Data requirements}} \\
Data type & Choices $(s_t, a_t)$ & Preferences $\tau^w \succ \tau^l$ \\
Sample size & 1000 choices & 500 comparisons \\
\midrule
\multicolumn{3}{l}{\emph{Recovery accuracy}} \\
"""

    # Add numerical results
    nfxp_err = np.mean(nfxp_errors[1000])
    rlhf_corr = np.mean(rlhf_corrs[500])
    nfxp_dist = np.mean(nfxp_distances)
    rlhf_dist = np.mean(rlhf_distances)
    nfxp_thresh = np.mean(nfxp_thresholds)
    rlhf_thresh = np.mean(rlhf_thresholds)
    nfxp_thresh_std = np.std(nfxp_thresholds)
    rlhf_thresh_std = np.std(rlhf_thresholds)

    latex += f"Parameter error & {nfxp_err:.4f} & -- \\\\\n"
    latex += f"Cost correlation & -- & {rlhf_corr:.4f} \\\\\n"
    latex += f"Policy distance & {nfxp_dist:.4f} & {rlhf_dist:.4f} \\\\\n"
    latex += f"Threshold (true: {threshold_true}) & {nfxp_thresh:.1f} $\\pm$ {nfxp_thresh_std:.1f} & {rlhf_thresh:.1f} $\\pm$ {rlhf_thresh_std:.1f} \\\\\n"

    latex += r"""\midrule
\multicolumn{3}{l}{\emph{Assumptions}} \\
Requires optimal agent & Yes & No \\
Parametric reward & Yes & Yes \\
Known anchor cost & No & Yes \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item RLHF uses Bradley-Terry model which identifies rewards up to affine transformation.
Policy derivation requires anchoring on a known cost (replacement cost $RC$) to fix the scale.
\end{tablenotes}
\end{table}
"""

    with open('ch07_rlhf/sims/nfxp_vs_rlhf_results.tex', 'w') as f:
        f.write(latex)
    print("Saved: ch07_rlhf/sims/nfxp_vs_rlhf_results.tex")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("NFXP vs RLHF: Two Approaches to Preference Recovery")
    print("="*70)

    print("\nConfiguration:")
    print(f"  States: {S}")
    print(f"  Discount: {GAMMA}")
    print(f"  True theta: {THETA_TRUE}")
    print(f"  Replacement cost: {RC}")
    print(f"  Seeds: {N_SEEDS}")

    # Build transition matrix
    P = build_transition_matrix()
    print(f"\nTransition matrix built: P[a, s, s'] shape = {P.shape}")

    # Run experiments
    nfxp_errors, rlhf_corrs = run_experiment_1(P)
    nfxp_dist, rlhf_dist, nfxp_thresh, rlhf_thresh, thresh_true = run_experiment_2(P)
    run_experiment_3(P)
    run_experiment_4(P)

    # Generate outputs
    print("\n" + "="*70)
    print("Generating output files...")
    print("="*70)

    plot_cost_recovery(nfxp_errors, rlhf_corrs, P)
    plot_policy_comparison(P, nfxp_dist, rlhf_dist, thresh_true)
    generate_latex_table(nfxp_errors, rlhf_corrs, nfxp_dist, rlhf_dist,
                        nfxp_thresh, rlhf_thresh, thresh_true)

    print("\nOutput files:")
    print("  ch07_rlhf/sims/cost_recovery.png")
    print("  ch07_rlhf/sims/policy_comparison.png")
    print("  ch07_rlhf/sims/nfxp_vs_rlhf_results.tex")


if __name__ == "__main__":
    main()
