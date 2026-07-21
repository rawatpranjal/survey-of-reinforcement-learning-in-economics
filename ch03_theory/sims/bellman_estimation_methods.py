"""
Comparing Estimators of the Bellman Equation
Chapter 3 (Theory) -- companion to subsection sec:bellman_estimation.

One small stochastic MDP, one linear feature class, several estimators that all
begin from the same Bellman equation but define different population targets or
use different solvers: orthogonal value projection, LSTD (projected fixed
point), raw Bellman-residual minimization (BRM), the Antos-Szepesvari-Munos
corrected/minimax criterion, semi-gradient TD(0), fitted value iteration, and
TDC (gradient-TD). The three-state machine-maintenance MDP and every population
target reproduce the hand-computed values in the study note that motivates the
subsection, so the population block doubles as a verification anchor.

Population objects are closed form. Empirical estimators are averaged over
N_SEEDS with standard errors. Sampling is i.i.d. from the stationary
distribution mu of the fixed policy, so D = diag(mu) is exact and the on-policy
weighting matches the theory (Tsitsiklis-Van Roy 1997; Sutton et al. 2009).
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTDIR, "cache")
SCRIPT_NAME = "bellman_estimation_methods"

np.random.seed(42)

# ======================================================================
# Layer 1: configuration
# ======================================================================
# Three-state machine-maintenance MDP (study note running example).
# States G, F, W (good, fatigued, worn) -> indices 0, 1, 2.
# Actions O (operate), M (maintain) -> indices 0, 1.
# Reward deterministic given (s, a); transition stochastic.
GAMMA = 0.9

# Reward[s, a]
REWARD = np.array(
    [
        [5.0, -1.0],  # G: O, M
        [3.0, -2.0],  # F: O, M
        [0.0, -4.0],  # W: O, M
    ]
)

# P[s, a] -> row over next states (G, F, W)
P = np.array(
    [
        [[0.75, 0.25, 0.00], [0.95, 0.05, 0.00]],  # G: O, M
        [[0.10, 0.60, 0.30], [0.75, 0.20, 0.05]],  # F: O, M
        [[0.00, 0.10, 0.90], [0.65, 0.30, 0.05]],  # W: O, M
    ]
)

# Fixed evaluation policy pi: operate in G and F, maintain in W.
PI = np.array([0, 0, 1])  # action index per state

# Linear feature map phi(s) = [1, x(s)] with deterioration score x = 0, 1, 2.
X_SCORE = np.array([0.0, 1.0, 2.0])
PHI = np.column_stack([np.ones(3), X_SCORE])  # 3 x 2

# The MDP tables are hashed into the cache key. Editing a reward, a transition
# row, the policy or the features must invalidate every cached component, so
# they cannot be left out and patched up by hand-bumping a version number.
ENV_PARAMS = {
    "gamma": GAMMA,
    "reward": REWARD.tolist(),
    "transitions": P.tolist(),
    "policy": PI.tolist(),
    "features": PHI.tolist(),
}

SHARED_CONFIG = {
    **ENV_PARAMS,
    "N_SEEDS": 30,
    "N_SAMPLES": 20000,  # i.i.d. transitions per seed for batch estimators
}

# Online stochastic-approximation schedule for TD(0) and TDC.
ONLINE_CONFIG = {
    **SHARED_CONFIG,
    "N_STEPS": 400000,
    "ALPHA0": 0.05,
    "TAU": 10000.0,  # step size alpha_t = ALPHA0 / (1 + t / TAU)
    # TDC auxiliary-weight ratio: beta_t = ETA_W * alpha_t. The w recursion is a
    # least-mean-squares fit, so it is stable only while beta_t ||phi||^2 < 2;
    # here max ||phi||^2 = 5, so beta_0 must stay below 0.4. ETA_W = 10 (the
    # first setting tried) gives beta_0 = 1.0 and diverges.
    "ETA_W": 2.0,
    "LOG_STRIDE": 1000,
    # Fitted value iteration is a contraction with modulus gamma, so 60 sweeps
    # leave 0.9^60 * 34 = 0.06 of truncation error and make FVI look separated
    # from LSTD when it is not. 300 sweeps puts the truncation below 1e-12.
    "FVI_ITERS": 300,
}


# ======================================================================
# Layer 2: population objects (closed form)
# ======================================================================


def policy_matrices():
    """Return P_pi (3x3), r_pi (3,) induced by the fixed policy PI."""
    P_pi = np.array([P[s, PI[s]] for s in range(3)])
    r_pi = np.array([REWARD[s, PI[s]] for s in range(3)])
    return P_pi, r_pi


def stationary_dist(P_pi):
    """Stationary distribution mu of P_pi (left eigenvector, normalized)."""
    vals, vecs = np.linalg.eig(P_pi.T)
    idx = np.argmin(np.abs(vals - 1.0))
    mu = np.real(vecs[:, idx])
    return mu / mu.sum()


def bellman_pi(v, P_pi, r_pi):
    """Apply the policy Bellman operator T^pi to a value vector v."""
    return r_pi + GAMMA * P_pi @ v


def weighted_norm(x, mu):
    """D-weighted norm ||x||_D with D = diag(mu)."""
    return float(np.sqrt(np.sum(mu * x**2)))


def d_projection(target, mu):
    """theta minimizing ||Phi theta - target||_D and the fitted values."""
    D = np.diag(mu)
    C = PHI.T @ D @ PHI
    theta = np.linalg.solve(C, PHI.T @ D @ target)
    return theta, PHI @ theta


def compute_population():
    P_pi, r_pi = policy_matrices()
    mu = stationary_dist(P_pi)
    D = np.diag(mu)
    M = np.eye(3) - GAMMA * P_pi

    # Exact value of the fixed policy.
    v_pi = np.linalg.solve(M, r_pi)

    # (1) Orthogonal projection of the true value.
    theta_proj, v_proj = d_projection(v_pi, mu)

    # (2) LSTD projected fixed point: A theta = b.
    A = PHI.T @ D @ M @ PHI
    b = PHI.T @ D @ r_pi
    theta_lstd = np.linalg.solve(A, b)
    v_lstd = PHI @ theta_lstd

    # (3) Raw Bellman-residual minimization: min ||r - M Phi theta||_D^2.
    A_brm = PHI.T @ M.T @ D @ M @ PHI
    b_brm = PHI.T @ M.T @ D @ r_pi
    theta_brm = np.linalg.solve(A_brm, b_brm)
    v_brm = PHI @ theta_brm

    # (4) Antos corrected/minimax criterion. Proposition (Antos et al. 2008,
    #     Prop 2): its minimizer over a linear class coincides with the LSTD
    #     projected fixed point. Verify STRUCTURALLY, not by re-solving A theta=b.
    #     The corrected criterion is
    #       J_corr(theta) = ||v_theta - T^pi v_theta||_D^2
    #                       - inf_h ||h - T^pi v_theta||_D^2
    #                     = ||v_theta - Pi_D T^pi v_theta||_D^2   (Pythagoras),
    #     a convex quadratic. Confirm it is ~0 at theta_LSTD (so v = Pi T^pi v,
    #     the projected fixed point) and strictly larger at perturbations, i.e.
    #     theta_LSTD is its unique minimizer -- an independent check of Prop 2.
    def J_corrected(theta):
        v = PHI @ theta
        Tv = bellman_pi(v, P_pi, r_pi)
        raw = float(np.sum(mu * (Tv - v) ** 2))  # ||v - T^pi v||_D^2
        _, proj_Tv = d_projection(Tv, mu)
        unrep = float(np.sum(mu * (Tv - proj_Tv) ** 2))  # ||T^pi v - Pi T^pi v||_D^2
        return raw - unrep

    antos_J_at_lstd = J_corrected(theta_lstd)
    antos_perturbs = [
        np.array(p) for p in [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]
    ]
    antos_J_at_perturbed = [J_corrected(theta_lstd + p) for p in antos_perturbs]

    # Minimize the corrected criterion in its own right, so the table row is an
    # independent computation rather than a copy of the LSTD row. J_corr is
    # ||M theta - c||_D^2 with M = Phi - gamma Pi_D P_pi Phi and c = Pi_D r_pi,
    # whose normal equations never reference A theta = b.
    Pi_D = PHI @ np.linalg.solve(PHI.T @ D @ PHI, PHI.T @ D)
    M_corr = PHI - GAMMA * Pi_D @ P_pi @ PHI
    c_corr = Pi_D @ r_pi
    theta_antos = np.linalg.solve(M_corr.T @ D @ M_corr, M_corr.T @ D @ c_corr)
    v_antos = PHI @ theta_antos

    # Antos decomposition for a specific candidate v = (30, 26, 22),
    # theta = (30, -4). Reproduces the study note's worked numbers.
    theta_cand = np.array([30.0, -4.0])
    v_cand = PHI @ theta_cand
    Tv_cand = bellman_pi(v_cand, P_pi, r_pi)
    raw_resid_sq = float(np.sum(mu * (Tv_cand - v_cand) ** 2))
    _, proj_Tv = d_projection(Tv_cand, mu)  # best representable Bellman target
    unrep_sq = float(np.sum(mu * (Tv_cand - proj_Tv) ** 2))
    corrected_sq = raw_resid_sq - unrep_sq
    corrected_direct = float(np.sum(mu * (v_cand - proj_Tv) ** 2))

    # Conditional-variance decomposition at state F under theta = (30, -4).
    # E[delta^2 | F] = (Bellman residual)^2 + Var(target | F).
    f = 1  # index of state F
    a_f = PI[f]
    succ = P[f, a_f]  # next-state distribution
    targets_f = REWARD[f, a_f] + GAMMA * v_cand  # target per realized successor
    td_f = targets_f - v_cand[f]  # delta per successor
    e_delta = float(np.sum(succ * td_f))
    e_delta_sq = float(np.sum(succ * td_f**2))
    var_target = e_delta_sq - e_delta**2

    # Expected TD update direction E[phi delta] at theta = (30, -4).
    resid_cand = bellman_pi(v_cand, P_pi, r_pi) - v_cand  # E[delta | s]
    e_phi_delta = PHI.T @ (mu * resid_cand)

    # Action values (control) for the summary display.
    q_O = REWARD[:, 0] + GAMMA * P[:, 0, :] @ v_pi
    q_M = REWARD[:, 1] + GAMMA * P[:, 1, :] @ v_pi

    # Summary norms for each population solution.
    def norms(v):
        return weighted_norm(v - v_pi, mu), weighted_norm(
            bellman_pi(v, P_pi, r_pi) - v, mu
        )

    summary = {
        "Orthogonal projection": (theta_proj, v_proj, *norms(v_proj)),
        "LSTD projected fixed point": (theta_lstd, v_lstd, *norms(v_lstd)),
        "Antos corrected BRM": (theta_antos, v_antos, *norms(v_antos)),
        "Raw Bellman-residual min.": (theta_brm, v_brm, *norms(v_brm)),
    }

    return {
        "P_pi": P_pi,
        "r_pi": r_pi,
        "mu": mu,
        "v_pi": v_pi,
        "theta_proj": theta_proj,
        "v_proj": v_proj,
        "theta_lstd": theta_lstd,
        "v_lstd": v_lstd,
        "theta_brm": theta_brm,
        "v_brm": v_brm,
        "theta_antos": theta_antos,
        "v_antos": v_antos,
        "antos_J_at_lstd": antos_J_at_lstd,
        "antos_J_at_perturbed": antos_J_at_perturbed,
        "antos_decomp": {
            "raw_resid_sq": raw_resid_sq,
            "unrep_sq": unrep_sq,
            "corrected_sq": corrected_sq,
            "corrected_direct": corrected_direct,
        },
        "variance_decomp": {
            "td_errors_F": td_f,
            "e_delta": e_delta,
            "e_delta_sq": e_delta_sq,
            "var_target": var_target,
        },
        "e_phi_delta": e_phi_delta,
        "q_O": q_O,
        "q_M": q_M,
        "summary": summary,
    }


# ======================================================================
# Layer 3: empirical estimators (finite sample, multiple seeds)
# ======================================================================


def _draw_successors(states, rng):
    """Vectorized draw of s' ~ P[s, pi(s)] for an array of states (inverse-CDF)."""
    u = rng.random(len(states))
    succ = np.empty(len(states), dtype=int)
    for s in range(3):
        mask = states == s
        if mask.any():
            cdf = np.cumsum(P[s, PI[s]])
            succ[mask] = np.searchsorted(cdf, u[mask], side="right")
    return np.clip(succ, 0, 2)


def sample_transitions(mu, rng, n):
    """i.i.d. states from mu; one successor each, plus an independent second
    successor for the double-sampling estimator. Rewards deterministic."""
    states = rng.choice(3, size=n, p=mu)
    rewards = REWARD[states, PI[states]]
    succ1 = _draw_successors(states, rng)
    succ2 = _draw_successors(states, rng)
    return states, rewards, succ1, succ2


def emp_lstd(phi, r, phi_n):
    """Empirical LSTD: A_hat theta = b_hat."""
    n = len(r)
    A = (phi[:, :, None] * (phi - GAMMA * phi_n)[:, None, :]).sum(0) / n
    b = (phi * r[:, None]).sum(0) / n
    return np.linalg.solve(A, b)


def emp_brm_single(phi, r, phi_n):
    """Single-successor BRM: OLS of r on (phi - gamma phi'). This minimizes the
    empirical squared TD error, a biased proxy for the Bellman residual because
    the same successor appears in both factors of the quadratic form."""
    psi = phi - GAMMA * phi_n
    A = psi.T @ psi
    b = psi.T @ r
    return np.linalg.solve(A, b)


def emp_brm_double(phi, r, psi1, psi2):
    """Double-successor BRM: independent successors in the two factors debias
    the quadratic form (E[psi1 psi2^T | s] = psi(s) psi(s)^T). Requires a
    generative model. Converges to the population raw-BRM solution."""
    n = len(r)
    G = (psi1[:, :, None] * psi2[:, None, :]).sum(0) / n
    G = 0.5 * (G + G.T)  # symmetrize (true matrix is symmetric)
    b = (psi1 * r[:, None]).sum(0) / n
    return np.linalg.solve(G, b)


def emp_fitted_vi(phi, r, phi_n, n_iters):
    """Fitted value iteration: regress on targets from a frozen previous
    iterate. Fixed point is the projected Bellman fixed point (= LSTD)."""
    theta = np.zeros(2)
    pinv = np.linalg.pinv(phi)
    for _ in range(n_iters):
        targets = r + GAMMA * (phi_n @ theta)
        theta = pinv @ targets
    return theta


def run_online(mu, rng, cfg, theta_star):
    """Semi-gradient TD(0) and TDC over a stream of i.i.d. transitions.
    Returns per-checkpoint distance ||theta_t - theta_star|| for each method."""
    n_steps = cfg["N_STEPS"]
    a0, tau, eta = cfg["ALPHA0"], cfg["TAU"], cfg["ETA_W"]
    stride = cfg["LOG_STRIDE"]

    theta_td = np.zeros(2)
    theta_tdc = np.zeros(2)
    w_tdc = np.zeros(2)
    steps, dist_td, dist_tdc = [], [], []

    # Pre-sample the whole i.i.d. stream (vectorized); the update loop stays
    # sequential but does no per-step RNG, so long runs are cheap.
    states = rng.choice(3, size=n_steps, p=mu)
    succ = _draw_successors(states, rng)
    rewards = REWARD[states, PI[states]]
    phi_all = PHI[states]
    phin_all = PHI[succ]

    for t in range(n_steps):
        r = rewards[t]
        ph, phn = phi_all[t], phin_all[t]
        alpha = a0 / (1.0 + t / tau)

        # semi-gradient TD(0)
        delta_td = r + GAMMA * phn @ theta_td - ph @ theta_td
        theta_td = theta_td + alpha * delta_td * ph

        # TDC (Sutton et al. 2009, eq. 10). This is the practical constant-ratio
        # form beta_t = eta * alpha_t, not the strict two-timescale condition
        # alpha_t / beta_t -> 0 of their Theorem 2; on-policy the constant-ratio
        # version still converges to the TD fixed point A theta = b.
        beta = eta * alpha
        delta_c = r + GAMMA * phn @ theta_tdc - ph @ theta_tdc
        theta_tdc = theta_tdc + alpha * (delta_c * ph - GAMMA * phn * (ph @ w_tdc))
        w_tdc = w_tdc + beta * (delta_c - ph @ w_tdc) * ph

        if (t + 1) % stride == 0:
            steps.append(t + 1)
            dist_td.append(np.linalg.norm(theta_td - theta_star))
            dist_tdc.append(np.linalg.norm(theta_tdc - theta_star))

    return (
        np.array(steps),
        np.array(dist_td),
        np.array(dist_tdc),
        theta_td,
        theta_tdc,
    )


def compute_empirical(pop):
    mu = pop["mu"]
    theta_star = pop["theta_lstd"]
    n_seeds = SHARED_CONFIG["N_SEEDS"]
    n = SHARED_CONFIG["N_SAMPLES"]

    est = {k: [] for k in ["lstd", "brm_single", "brm_double", "fvi", "td", "tdc"]}
    curves_td, curves_tdc = [], []
    steps_ref = None

    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        s, r, s1, s2 = sample_transitions(mu, rng, n)
        phi = PHI[s]
        phi1, phi2 = PHI[s1], PHI[s2]
        psi1 = phi - GAMMA * phi1
        psi2 = phi - GAMMA * phi2

        est["lstd"].append(emp_lstd(phi, r, phi1))
        est["brm_single"].append(emp_brm_single(phi, r, phi1))
        est["brm_double"].append(emp_brm_double(phi, r, psi1, psi2))
        est["fvi"].append(emp_fitted_vi(phi, r, phi1, ONLINE_CONFIG["FVI_ITERS"]))

        steps, d_td, d_tdc, th_td, th_tdc = run_online(
            mu, rng, ONLINE_CONFIG, theta_star
        )
        est["td"].append(th_td)
        est["tdc"].append(th_tdc)
        curves_td.append(d_td)
        curves_tdc.append(d_tdc)
        steps_ref = steps

    est = {k: np.array(v) for k, v in est.items()}
    return {
        "theta": est,
        "steps": steps_ref,
        "curve_td_mean": np.mean(curves_td, 0),
        "curve_td_se": np.std(curves_td, 0) / np.sqrt(n_seeds),
        "curve_tdc_mean": np.mean(curves_tdc, 0),
        "curve_tdc_se": np.std(curves_tdc, 0) / np.sqrt(n_seeds),
    }


# ======================================================================
# Orchestration
# ======================================================================


def compute_data(force=None):
    force = force or set()
    pop = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "population",
        ENV_PARAMS,
        compute_population,
        force=("population" in force),
    )
    emp = compute_or_load(
        CACHE_DIR,
        SCRIPT_NAME,
        "empirical",
        ONLINE_CONFIG,
        compute_empirical,
        pop,
        force=("empirical" in force or "population" in force),
    )
    return {"population": pop, "empirical": emp}


# ======================================================================
# Layer 5: outputs (table + figure)
# ======================================================================


def _emp_norms(theta_rows, pop):
    """Mean +/- SE of value error and Bellman-residual norm across seeds."""
    mu, v_pi = pop["mu"], pop["v_pi"]
    P_pi, r_pi = pop["P_pi"], pop["r_pi"]
    ve, be = [], []
    for th in theta_rows:
        v = PHI @ th
        ve.append(weighted_norm(v - v_pi, mu))
        be.append(weighted_norm(bellman_pi(v, P_pi, r_pi) - v, mu))
    ve, be = np.array(ve), np.array(be)
    n = len(theta_rows)
    return ve.mean(), ve.std() / np.sqrt(n), be.mean(), be.std() / np.sqrt(n)


def generate_outputs(data):
    pop = data["population"]
    emp = data["empirical"]

    # ---- LaTeX table ----
    rows_pop = [
        (
            "Orthogonal projection of $v^\\pi$",
            "Pop.",
            *pop["summary"]["Orthogonal projection"][2:],
        ),
        (
            "LSTD projected fixed point",
            "Pop.",
            *pop["summary"]["LSTD projected fixed point"][2:],
        ),
        (
            "Antos corrected BRM",
            "Pop.",
            *pop["summary"]["Antos corrected BRM"][2:],
        ),
        (
            "Raw Bellman-residual min.",
            "Pop.",
            *pop["summary"]["Raw Bellman-residual min."][2:],
        ),
    ]
    emp_specs = [
        ("LSTD", "lstd"),
        ("Semi-gradient TD(0)", "td"),
        ("TDC (gradient TD)", "tdc"),
        ("Fitted value iteration", "fvi"),
        ("Raw BRM, two-sample", "brm_double"),
        ("Raw BRM, one-sample", "brm_single"),
    ]
    rows_emp = []
    for label, key in emp_specs:
        ve, ve_se, be, be_se = _emp_norms(emp["theta"][key], pop)
        rows_emp.append((label, ve, ve_se, be, be_se))

    # House convention: rank order by performance, best first. The ranking
    # column is the distance to the true value function.
    rows_pop.sort(key=lambda r: r[2])
    rows_emp.sort(key=lambda r: r[1])

    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Estimator & Type & $\\|\\hat v - v^\\pi\\|_D$ & $\\|T^\\pi\\hat v - \\hat v\\|_D$\\\\",
        "\\midrule",
    ]
    for label, typ, ve, be in rows_pop:
        lines.append(f"{label} & {typ} & ${ve:.4f}$ & ${be:.4f}$\\\\")
    lines.append("\\midrule")
    for label, ve, ve_se, be, be_se in rows_emp:
        lines.append(
            f"{label} & Emp. & ${ve:.4f}\\,(\\pm{ve_se:.4f})$ "
            f"& ${be:.4f}\\,(\\pm{be_se:.4f})$\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = os.path.join(OUTDIR, "bellman_estimation_methods.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- learning-curve figure ----
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    steps = emp["steps"]
    for mean, se, color, label in [
        (
            emp["curve_td_mean"],
            emp["curve_td_se"],
            COLORS["blue"],
            "Semi-gradient TD(0)",
        ),
        (
            emp["curve_tdc_mean"],
            emp["curve_tdc_se"],
            COLORS["orange"],
            "TDC (gradient TD)",
        ),
    ]:
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.2)
    # No reference line: the y axis is distance to the fixed point, so the
    # benchmark is zero, which a log scale cannot draw.
    ax.set_xlabel("Transitions")
    ax.set_ylabel(r"$\|\theta_t - \theta_{\mathrm{LSTD}}\|_2$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_path = os.path.join(OUTDIR, "bellman_estimation_methods.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return tex_path, fig_path


# ======================================================================
# stdout report
# ======================================================================


def print_report(data):
    pop = data["population"]
    emp = data["empirical"]
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 70)
    print("Comparing Estimators of the Bellman Equation")
    print("Three-state machine-maintenance MDP, gamma =", GAMMA)
    print("=" * 70)

    print("\n-- Population objects (verification anchors, study note) --")
    print(f"mu (stationary)         : {pop['mu']}   note (0.4813, 0.3942, 0.1245)")
    print(f"v^pi                    : {pop['v_pi']}   note (34.7903, 28.0305, 25.0477)")
    print(f"theta_proj              : {pop['theta_proj']}   note (34.4030, -5.4265)")
    print(
        f"v_proj                  : {pop['v_proj']}   note (34.4030, 28.9764, 23.5499)"
    )
    print(f"theta_LSTD              : {pop['theta_lstd']}   note (34.3673, -5.3711)")
    print(
        f"v_LSTD                  : {pop['v_lstd']}   note (34.3673, 28.9962, 23.6252)"
    )
    print(f"theta_BRM               : {pop['theta_brm']}   note (33.8835, -4.6189)")
    print(
        f"v_BRM                   : {pop['v_brm']}   note (33.8835, 29.2646, 24.6458)"
    )
    print(
        f"theta_Antos (own normal eqs)    : {pop['theta_antos']}   "
        f"deviation from theta_LSTD {np.linalg.norm(pop['theta_antos'] - pop['theta_lstd']):.2e}"
    )
    print(
        f"Antos corrected J at theta_LSTD : {pop['antos_J_at_lstd']:.2e}   (Prop 2: minimized, == 0)"
    )
    print(
        f"Antos corrected J at perturbations : "
        f"{[round(j, 4) for j in pop['antos_J_at_perturbed']]}   (all > 0)"
    )

    ad = pop["antos_decomp"]
    print("\n-- Antos decomposition at v=(30,26,22) --")
    print(f"raw squared Bellman residual : {ad['raw_resid_sq']:.4f}   note 0.6469")
    print(f"unrepresentable component    : {ad['unrep_sq']:.4f}   note 0.0834")
    print(f"corrected (raw - unrep)      : {ad['corrected_sq']:.4f}   note 0.5634")
    print(
        f"corrected (direct ||v-PiTv||): {ad['corrected_direct']:.4f}   (should match)"
    )

    vd = pop["variance_decomp"]
    print("\n-- Conditional-variance decomposition at state F, theta=(30,-4) --")
    print(f"TD errors by successor (G,F,W): {vd['td_errors_F']}   note (4, 0.4, -3.2)")
    print(f"E[delta | F]                  : {vd['e_delta']:.4f}   note -0.32")
    print(f"E[delta^2 | F]                : {vd['e_delta_sq']:.4f}   note 4.768")
    print(f"squared residual = E[delta|F]^2 : {vd['e_delta'] ** 2:.4f}   note 0.1024")
    print(f"Var(target | F)               : {vd['var_target']:.4f}   note 4.6656")
    print(
        f"E[phi delta] at theta=(30,-4) : {pop['e_phi_delta']}   note (0.3485, -0.2357)"
    )

    print("\n-- Action values (control) --")
    print(f"q^pi(.,O) : {pop['q_O']}   note (34.7903, 28.0305, 22.8114)")
    print(f"q^pi(.,M) : {pop['q_M']}   note (30.0071, 27.6561, 25.0477)")

    print("\n-- Summary norms (population) --")
    print(f"{'Method':<32}{'||v-v^pi||_D':>14}{'||Tv-v||_D':>14}")
    for name, tup in pop["summary"].items():
        print(f"{name:<32}{tup[2]:>14.4f}{tup[3]:>14.4f}")

    print("\n-- Empirical estimators (mean +/- SE over seeds) --")
    print(
        f"{'Method':<28}{'theta_0':>10}{'theta_1':>10}{'||v-v^pi||_D':>14}{'||Tv-v||_D':>13}"
    )
    for label, key in [
        ("LSTD", "lstd"),
        ("Semi-gradient TD(0)", "td"),
        ("TDC (gradient TD)", "tdc"),
        ("Fitted value iteration", "fvi"),
        ("Raw BRM two-sample", "brm_double"),
        ("Raw BRM one-sample", "brm_single"),
    ]:
        th = emp["theta"][key]
        ve, _, be, _ = _emp_norms(th, pop)
        print(
            f"{label:<28}{th[:, 0].mean():>10.4f}{th[:, 1].mean():>10.4f}{ve:>14.4f}{be:>13.4f}"
        )

    print("\n-- Cross-checks --")
    lstd_emp = emp["theta"]["lstd"].mean(0)
    print(
        f"empirical LSTD -> population LSTD : {np.linalg.norm(lstd_emp - pop['theta_lstd']):.4f}"
    )
    bd = emp["theta"]["brm_double"].mean(0)
    bs = emp["theta"]["brm_single"].mean(0)
    print(
        f"two-sample BRM -> population BRM  : {np.linalg.norm(bd - pop['theta_brm']):.4f}"
    )
    print(
        f"one-sample BRM -> population BRM  : {np.linalg.norm(bs - pop['theta_brm']):.4f}  (biased, larger)"
    )
    td = emp["theta"]["td"].mean(0)
    tdc = emp["theta"]["tdc"].mean(0)
    # Two distinct quantities. The first is the distance of the seed-AVERAGED
    # parameter, in which zero-mean sampling scatter cancels. The second is the
    # average over seeds of the per-seed distance, which is what the figure
    # plots and is necessarily the larger of the two.
    print(
        f"||mean_seeds(theta_TD) - theta_LSTD||   : {np.linalg.norm(td - pop['theta_lstd']):.4f}"
    )
    print(
        f"||mean_seeds(theta_TDC) - theta_LSTD||  : {np.linalg.norm(tdc - pop['theta_lstd']):.4f}"
    )
    print(
        f"mean_seeds||theta_TD - theta_LSTD||     : {emp['curve_td_mean'][-1]:.4f}   (figure endpoint)"
    )
    print(
        f"mean_seeds||theta_TDC - theta_LSTD||    : {emp['curve_tdc_mean'][-1]:.4f}   (figure endpoint)"
    )


# ======================================================================
# Inline verification (fails loudly on drift)
# ======================================================================


def verify(data):
    """Validation gate = primary-source STRUCTURAL properties, computed from the
    MDP, independent of any stated constant. The study note is a secondary source,
    so its published numbers are a soft transcription guard (loose print + one
    loose check on v_pi), never the gate the code is shaped to satisfy."""
    pop = data["population"]
    emp = data["empirical"]
    P_pi, r_pi, mu = pop["P_pi"], pop["r_pi"], pop["mu"]
    D = np.diag(mu)

    # --- MDP well-formedness (transcription integrity, not a result) ---
    assert np.allclose(P.sum(axis=2), 1.0), "transition rows must sum to 1"
    assert np.allclose(mu @ P_pi, mu, atol=1e-10), "mu must be stationary for P_pi"
    assert np.all(mu > 0), mu

    # --- LSTD: the projected Bellman residual is orthogonal to the features ---
    # Phi^T D (T^pi v_LSTD - v_LSTD) = 0  (defining property, Antos L297 / LSTD).
    resid_lstd = bellman_pi(pop["v_lstd"], P_pi, r_pi) - pop["v_lstd"]
    assert np.allclose(PHI.T @ (D @ resid_lstd), 0.0, atol=1e-9), PHI.T @ (
        D @ resid_lstd
    )

    # --- raw BRM: satisfies its own normal equations A_brm theta = b_brm ---
    M = np.eye(3) - GAMMA * P_pi
    A_brm = PHI.T @ M.T @ D @ M @ PHI
    b_brm = PHI.T @ M.T @ D @ r_pi
    assert np.allclose(A_brm @ pop["theta_brm"], b_brm, atol=1e-9)

    # --- Antos Prop 2: theta_LSTD minimizes the corrected criterion (== 0),
    #     strictly positive at perturbations (independent structural check) ---
    assert abs(pop["antos_J_at_lstd"]) < 1e-9, pop["antos_J_at_lstd"]
    assert all(j > 1e-6 for j in pop["antos_J_at_perturbed"]), pop[
        "antos_J_at_perturbed"
    ]
    ad = pop["antos_decomp"]
    assert abs(ad["corrected_sq"] - ad["corrected_direct"]) < 1e-9

    # --- conditional-variance identity: E[d^2|F] = E[d|F]^2 + Var (algebraic) ---
    vd = pop["variance_decomp"]
    assert abs(vd["e_delta_sq"] - (vd["e_delta"] ** 2 + vd["var_target"])) < 1e-12

    # --- empirical: LSTD -> population LSTD; two-sample BRM debiases one-sample ---
    lstd_emp = emp["theta"]["lstd"].mean(0)
    assert np.linalg.norm(lstd_emp - pop["theta_lstd"]) < 0.2, lstd_emp
    err_double = np.linalg.norm(emp["theta"]["brm_double"].mean(0) - pop["theta_brm"])
    err_single = np.linalg.norm(emp["theta"]["brm_single"].mean(0) - pop["theta_brm"])
    assert err_double < err_single, (err_double, err_single)

    # --- soft transcription guard vs the secondary study note (NOT the gate) ---
    note_v_pi = np.array([34.7903, 28.0305, 25.0477])
    dev = float(np.max(np.abs(pop["v_pi"] - note_v_pi)))
    tag = "ok" if dev < 1e-2 else "DRIFT -- check MDP transcription"
    print(
        f"\n[verify] structural properties hold. v_pi vs note: max dev {dev:.2e} ({tag})."
    )
    assert dev < 1e-2, (
        "v_pi drifted from the study note; MDP table likely mis-transcribed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)

    if args.plots_only:
        data = compute_data()  # all cache hits
    else:
        data = compute_data(force=force)

    if not args.data_only:
        print_report(data)
        verify(data)
        tex_path, fig_path = generate_outputs(data)
        print(f"\nOutputs:\n  {tex_path}\n  {fig_path}")
