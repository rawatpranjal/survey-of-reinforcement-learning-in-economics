"""
Confounded Off-Policy Evaluation in a 5-State Chain MDP
Chapter 9: Causal Inference and RL
Validates Theorem 1 (backdoor identification of interventional transitions)
and Lemma 1 (bias of naive OPE) via four estimators across confounding strengths.
"""

import numpy as np
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
apply_style()
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# MDP parameters
N_STATES = 5          # states 0..4; state 4 is absorbing goal
N_ACTIONS = 2         # 0 = advance, 1 = stay
GAMMA = 0.9           # discount factor

# Observed covariate Z ~ Bernoulli(0.5), i.i.d. per step
# Causal graph: Z_t -> U_t -> A_t, Z_t -> S_{t+1}
# Z satisfies the backdoor criterion relative to (A_t, S_{t+1})
P_Z1 = 0.5

# Unobserved confounder U, caused by Z: P(U=1|Z=z)
P_U1_GIVEN_Z1 = 0.9   # P(U=1 | Z=1)
P_U1_GIVEN_Z0 = 0.1   # P(U=1 | Z=0)

# Transition probabilities for 'advance' (action 0), depend on Z (not U)
P_ADV_Z1 = 0.9        # P(s+1 | s, advance, Z=1)
P_ADV_Z0 = 0.5        # P(s+1 | s, advance, Z=0)
# 'stay' (action 1): deterministic self-loop

# Rewards (deterministic, NO confounding)
STEP_COST = -1.0    # r(s, a) for s < 4
GOAL_REWARD = 0.0   # r(4, a)

# Behavioral policy: mu(advance | s, U) = base + rho * delta * (2U - 1)
# Depends on unobserved U, creating confounding through Z -> U correlation
MU_BASE = 0.55
MU_DELTA = 0.25
# At rho=0: mu(adv) = 0.55 for all U (no confounding)
# At rho=1: mu(adv|U=1) = 0.80, mu(adv|U=0) = 0.30

# Target policy: always advance
# pi(advance | s) = 1 for s < 4

# Experimental design
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS = 20
BASE_SEED = 42
N_TRAJ = 2000         # trajectories per seed per rho
T_MAX = 50            # max steps per trajectory
W_MAX = 50.0          # IS weight truncation

# Estimator names and colors
ESTIMATOR_NAMES = ["Oracle", "Naive", "Backdoor", "Naive IS"]
EST_COLORS = {
    "Naive": COLORS['orange'],
    "Backdoor": COLORS['blue'],
    "Naive IS": COLORS['red'],
}
EST_MARKERS = {
    "Naive": "s",
    "Backdoor": "^",
    "Naive IS": "D",
}


# ============================================================
# Oracle: analytical computation of V^pi under target policy
# ============================================================

def compute_oracle_value():
    """Compute true V^pi(s) for target policy pi(advance|s)=1 for all s<4.

    True interventional transition for 'advance':
      P(s+1|s, do(adv)) = P(s+1|s,adv,Z=1)*P(Z=1) + P(s+1|s,adv,Z=0)*P(Z=0)
                        = P_ADV_Z1 * P_Z1 + P_ADV_Z0 * (1-P_Z1)

    Solve V = r + gamma * P_pi * V via matrix inversion.
    """
    p_adv_true = P_ADV_Z1 * P_Z1 + P_ADV_Z0 * (1 - P_Z1)

    # Build transition matrix under target policy (always advance)
    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES - 1):
        P_pi[s, s + 1] = p_adv_true
        P_pi[s, s] = 1.0 - p_adv_true
    P_pi[N_STATES - 1, N_STATES - 1] = 1.0  # absorbing

    # Reward vector under target policy
    r_pi = np.full(N_STATES, STEP_COST)
    r_pi[N_STATES - 1] = GOAL_REWARD

    # V = (I - gamma * P_pi)^{-1} r_pi
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

def generate_trajectories(rng, rho, n_traj=N_TRAJ, t_max=T_MAX):
    """Generate trajectories from the confounded MDP.

    Causal structure per step:
      Z_t ~ Bernoulli(P_Z1)                    [observed covariate]
      U_t ~ Bernoulli(P(U=1|Z_t))              [unobserved confounder]
      A_t ~ mu(a|s, U_t)                       [behavioral policy]
      S_{t+1} ~ P(s'|s, A_t, Z_t)             [transition depends on Z]

    Returns list of trajectories, each a list of (s, a, z, s_next, r) tuples.
    """
    trajectories = []
    for _ in range(n_traj):
        traj = []
        s = 0
        for t in range(t_max):
            if s == N_STATES - 1:  # absorbing goal
                break

            # Draw Z_t and U_t fresh each step (i.i.d.)
            z = rng.binomial(1, P_Z1)
            p_u1 = P_U1_GIVEN_Z1 if z == 1 else P_U1_GIVEN_Z0
            u = rng.binomial(1, p_u1)

            # Behavioral policy depends on U (unobserved)
            mu_adv = MU_BASE + rho * MU_DELTA * (2 * u - 1)
            a = 0 if rng.random() < mu_adv else 1  # a=0 advance, a=1 stay

            # Reward (deterministic, no confounding)
            r = STEP_COST

            # Transition depends on Z (observed), not U
            if a == 0:  # advance
                p_success = P_ADV_Z1 if z == 1 else P_ADV_Z0
                s_next = s + 1 if rng.random() < p_success else s
            else:  # stay
                s_next = s

            traj.append((s, a, z, s_next, r))
            s = s_next

        trajectories.append(traj)
    return trajectories


# ============================================================
# Estimator 1: Naive OPE (biased per Lemma 1)
# ============================================================

def naive_ope(trajectories):
    """Estimate P_obs(s'|s,a) from counts, solve Bellman.

    Biased because action a carries information about U (through behavioral
    policy), and U is correlated with Z (through Z -> U), and Z affects
    transitions. So P(s'|s,a) != P(s'|s,do(a)).
    """
    counts_sa = np.zeros((N_STATES, N_ACTIONS))
    counts_sas = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    for traj in trajectories:
        for (s, a, z, s_next, r) in traj:
            counts_sa[s, a] += 1
            counts_sas[s, a, s_next] += 1

    P_obs = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if counts_sa[s, a] > 0:
                P_obs[s, a, :] = counts_sas[s, a, :] / counts_sa[s, a]
            else:
                P_obs[s, a, s] = 1.0

    # Build transition matrix under target policy (always advance, a=0)
    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P_pi[s, :] = P_obs[s, 0, :]

    V = solve_bellman_with_transitions(P_pi)
    return V, P_obs


# ============================================================
# Estimator 2: Backdoor-adjusted OPE (Theorem 1)
# ============================================================

def backdoor_ope(trajectories):
    """Apply Theorem 1: P(s'|s,do(a)) = sum_z P(s'|s,a,z) P(z|s).

    Z satisfies the backdoor criterion in the causal graph
    Z -> U -> A, Z -> S', so this formula exactly recovers
    the interventional transition probability.
    """
    counts_saz = np.zeros((N_STATES, N_ACTIONS, 2))
    counts_sazs = np.zeros((N_STATES, N_ACTIONS, 2, N_STATES))
    counts_s = np.zeros(N_STATES)
    counts_sz = np.zeros((N_STATES, 2))

    for traj in trajectories:
        for (s, a, z, s_next, r) in traj:
            counts_saz[s, a, z] += 1
            counts_sazs[s, a, z, s_next] += 1
            counts_s[s] += 1
            counts_sz[s, z] += 1

    # Estimate P(s'|s,a,z) and P(z|s)
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

    # Backdoor adjustment: P(s'|s,do(a)) = sum_z P(s'|s,a,z) P(z|s)
    P_bd = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for z in range(2):
                P_bd[s, a, :] += P_saz[s, a, z, :] * P_z_given_s[s, z]

    # Build transition matrix under target policy
    P_pi = np.zeros((N_STATES, N_STATES))
    for s in range(N_STATES):
        P_pi[s, :] = P_bd[s, 0, :]

    V = solve_bellman_with_transitions(P_pi)
    return V, P_bd


# ============================================================
# Estimator 3: Naive IS (biased per Lemma 1 second claim)
# ============================================================

def naive_is(trajectories):
    """IS with marginal propensity mu_hat(a|s) from counts.

    Biased because mu_hat(a|s) != mu(a|s,u), the true per-step propensity.
    The marginal averages over U, but each step's action was generated
    conditional on a specific U_t that the analyst cannot observe.
    """
    counts_s = np.zeros(N_STATES)
    counts_sa = np.zeros((N_STATES, N_ACTIONS))
    for traj in trajectories:
        for (s, a, z, s_next, r) in traj:
            counts_s[s] += 1
            counts_sa[s, a] += 1

    mu_hat = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        if counts_s[s] > 0:
            mu_hat[s, :] = counts_sa[s, :] / counts_s[s]
        else:
            mu_hat[s, :] = 0.5

    weighted_returns = []
    for traj in trajectories:
        if len(traj) == 0:
            continue
        w = 1.0
        G = 0.0
        for t, (s, a, z, s_next, r) in enumerate(traj):
            # Target policy: pi(advance|s) = 1 for s < 4
            if s < N_STATES - 1:
                pi_a = 1.0 if a == 0 else 0.0
            else:
                pi_a = 1.0

            mu_a = mu_hat[s, a]
            if mu_a < 1e-10:
                w = 0.0
                break
            w *= pi_a / mu_a
            w = min(w, W_MAX)

            if pi_a == 0.0:
                w = 0.0
                break

            G += (GAMMA ** t) * r

        weighted_returns.append(w * G)

    if len(weighted_returns) == 0:
        return 0.0
    return np.mean(weighted_returns)


# ============================================================
# Population-level verification
# ============================================================

def print_population_verification(rho):
    """Print population-level transition probabilities for verification."""
    mu_adv_u1 = MU_BASE + rho * MU_DELTA
    mu_adv_u0 = MU_BASE - rho * MU_DELTA

    # P(advance | Z=z) = sum_u mu(adv|u) P(u|z)
    p_adv_z1 = mu_adv_u1 * P_U1_GIVEN_Z1 + mu_adv_u0 * (1 - P_U1_GIVEN_Z1)
    p_adv_z0 = mu_adv_u1 * P_U1_GIVEN_Z0 + mu_adv_u0 * (1 - P_U1_GIVEN_Z0)
    p_adv_marginal = p_adv_z1 * P_Z1 + p_adv_z0 * (1 - P_Z1)

    # True interventional: P(s+1|s, do(adv)) = P_ADV_Z1*P_Z1 + P_ADV_Z0*(1-P_Z1)
    p_true = P_ADV_Z1 * P_Z1 + P_ADV_Z0 * (1 - P_Z1)

    # Naive observational: P(s+1|s, adv) = sum_z P(s+1|s,adv,z) P(z|adv)
    p_z1_given_adv = p_adv_z1 * P_Z1 / p_adv_marginal
    p_obs = P_ADV_Z1 * p_z1_given_adv + P_ADV_Z0 * (1 - p_z1_given_adv)

    # Backdoor: exact recovery since Z satisfies the criterion
    p_bd = P_ADV_Z1 * P_Z1 + P_ADV_Z0 * (1 - P_Z1)

    print(f"  rho = {rho:.1f}")
    print(f"    mu(adv|U=1) = {mu_adv_u1:.3f}, mu(adv|U=0) = {mu_adv_u0:.3f}")
    print(f"    P(adv|Z=1) = {p_adv_z1:.4f}, P(adv|Z=0) = {p_adv_z0:.4f}")
    print(f"    P(adv) marginal = {p_adv_marginal:.4f}")
    print(f"    P(Z=1|adv) = {p_z1_given_adv:.4f}")
    print(f"    True interventional P(s+1|s,do(adv)) = {p_true:.4f}")
    print(f"    Naive observational P(s+1|s,adv)     = {p_obs:.4f}  (bias = {p_obs - p_true:+.4f})")
    print(f"    Backdoor-adjusted   P(s+1|s,do(adv)) = {p_bd:.4f}  (bias = {p_bd - p_true:+.4f})")
    print()

    return p_true, p_obs, p_bd


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Confounded Off-Policy Evaluation: 5-State Chain MDP")
    print("Chapter 9: Causal Inference and RL")
    print("=" * 72)
    print()
    print("MDP Parameters:")
    print(f"  States                    = {N_STATES} (0..{N_STATES-1}, state {N_STATES-1} absorbing)")
    print(f"  Actions                   = advance (0), stay (1)")
    print(f"  Discount gamma            = {GAMMA}")
    print(f"  Step cost r(s<4,a)        = {STEP_COST}")
    print(f"  Goal reward r(4,a)        = {GOAL_REWARD}")
    print(f"  P(s+1|s,adv,Z=1)         = {P_ADV_Z1}")
    print(f"  P(s+1|s,adv,Z=0)         = {P_ADV_Z0}")
    print(f"  P(Z=1)                    = {P_Z1}")
    print(f"  P(U=1|Z=1)               = {P_U1_GIVEN_Z1}")
    print(f"  P(U=1|Z=0)               = {P_U1_GIVEN_Z0}")
    print(f"  Behavioral base           = {MU_BASE}")
    print(f"  Behavioral delta          = {MU_DELTA}")
    print()
    print("Causal Structure:")
    print("  Z_t -> U_t -> A_t   (Z observed, U unobserved)")
    print("  Z_t -> S_{t+1}      (transitions depend on Z)")
    print("  Z satisfies the backdoor criterion relative to (A_t, S_{t+1})")
    print()
    print("Experimental Design:")
    print(f"  rho values                = {RHO_VALUES}")
    print(f"  Seeds per rho             = {N_SEEDS}")
    print(f"  Trajectories per seed     = {N_TRAJ}")
    print(f"  Max steps per trajectory  = {T_MAX}")
    print(f"  IS weight truncation      = {W_MAX}")
    print()

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------
    V_oracle, p_adv_true = compute_oracle_value()
    print("Oracle (analytical):")
    print(f"  True P(s+1|s,do(adv)) = {p_adv_true:.4f}")
    for s in range(N_STATES):
        print(f"  V^pi(s={s}) = {V_oracle[s]:.4f}")
    print()

    # --------------------------------------------------------
    # Population-level transition verification
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

    per_state_results = {rho: {name: {s: [] for s in range(N_STATES)}
                               for name in ["Oracle", "Naive", "Backdoor"]}
                         for rho in RHO_VALUES}

    total_runs = len(RHO_VALUES) * N_SEEDS
    pbar = tqdm(total=total_runs, desc="Running experiments")

    for rho in RHO_VALUES:
        for seed_idx in range(N_SEEDS):
            seed = BASE_SEED + seed_idx
            rng = np.random.default_rng(seed)
            trajectories = generate_trajectories(rng, rho)

            # Oracle (analytical, same for all seeds)
            results[rho]["Oracle"].append(V_oracle[0])
            for s in range(N_STATES):
                per_state_results[rho]["Oracle"][s].append(V_oracle[s])

            # Naive OPE
            V_naive, P_obs = naive_ope(trajectories)
            results[rho]["Naive"].append(V_naive[0])
            for s in range(N_STATES):
                per_state_results[rho]["Naive"][s].append(V_naive[s])

            # Backdoor OPE
            V_bd, P_bd = backdoor_ope(trajectories)
            results[rho]["Backdoor"].append(V_bd[0])
            for s in range(N_STATES):
                per_state_results[rho]["Backdoor"][s].append(V_bd[s])

            # Naive IS
            V_is = naive_is(trajectories)
            results[rho]["Naive IS"].append(V_is)

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
                "mean": vals.mean(),
                "bias": biases.mean(),
                "se_bias": biases.std(ddof=1) / np.sqrt(N_SEEDS),
                "rmse": np.sqrt((biases ** 2).mean()),
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
    # Compounding analysis: bias at each state for rho=1.0
    # --------------------------------------------------------
    print("=" * 72)
    print("COMPOUNDING: Per-state bias at rho=1.0 (Naive vs Oracle)")
    print("  Bias should increase for states further from goal")
    print("=" * 72)
    header = f"{'State':>6s}  {'Oracle':>10s}  {'Naive':>10s}  {'Backdoor':>10s}  {'Naive Bias':>12s}  {'BD Bias':>12s}"
    print(header)
    print("-" * len(header))
    for s in range(N_STATES):
        oracle_s = np.mean(per_state_results[1.0]["Oracle"][s])
        naive_s = np.mean(per_state_results[1.0]["Naive"][s])
        bd_s = np.mean(per_state_results[1.0]["Backdoor"][s])
        print(f"{s:6d}  {oracle_s:10.4f}  {naive_s:10.4f}  {bd_s:10.4f}  {naive_s - oracle_s:+12.4f}  {bd_s - oracle_s:+12.4f}")
    print()

    # --------------------------------------------------------
    # Transition estimate verification at rho=1.0
    # --------------------------------------------------------
    print("=" * 72)
    print("TRANSITION VERIFICATION at rho=1.0 (last seed)")
    print(f"  Expected: P_obs(s+1|s,adv) ~ 0.773, P_bd(s+1|s,do(adv)) ~ {p_adv_true:.2f}")
    print("=" * 72)
    rng = np.random.default_rng(BASE_SEED + N_SEEDS - 1)
    trajs = generate_trajectories(rng, 1.0)
    _, P_obs_check = naive_ope(trajs)
    _, P_bd_check = backdoor_ope(trajs)
    for s in range(N_STATES - 1):
        p_obs_adv = P_obs_check[s, 0, s + 1]
        p_bd_adv = P_bd_check[s, 0, s + 1]
        print(f"  State {s}: P_obs(s+1|s,adv) = {p_obs_adv:.4f}, P_bd(s+1|s,do(adv)) = {p_bd_adv:.4f}")
    print()

    # --------------------------------------------------------
    # Figure: Bias vs rho
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for name in ESTIMATOR_NAMES[1:]:
        biases = np.array([summary[rho][name]["bias"] for rho in RHO_VALUES])
        ses = np.array([summary[rho][name]["se_bias"] for rho in RHO_VALUES])
        ax.plot(RHO_VALUES, biases, color=EST_COLORS[name], marker=EST_MARKERS[name],
                label=name, markersize=7, zorder=3)
        ax.fill_between(RHO_VALUES, biases - 1.96 * ses, biases + 1.96 * ses,
                        color=EST_COLORS[name], alpha=0.15, zorder=2)
    ax.axhline(0, color=COLORS['gray'], linestyle="--", linewidth=0.8, zorder=1,
               label="Oracle (zero bias)")
    ax.set_xlabel(r"Confounding strength $\rho$")
    ax.set_ylabel(r"Bias: $\hat{V}^{\pi}(s_0) - V^{\pi}(s_0)$")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xticks(RHO_VALUES)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "confounded_ope_bias.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # --------------------------------------------------------
    # LaTeX table
    # --------------------------------------------------------
    tex_path = os.path.join(OUTPUT_DIR, "confounded_ope_results.tex")
    with open(tex_path, "w") as f:
        ncols = len(ESTIMATOR_NAMES) - 1
        col_spec = "c" + "cc" * ncols
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
    print("Output files:")
    print(f"  {fig_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()
