# Sampling the Bellman operator: Q-learning, TD(lambda), and inherent Bellman error
# Chapter 3 - The Theory of Reinforcement Learning
# Closed-form calculator on the Engine Replacement MDP, with fixed-seed Monte Carlo layers.
# The script (1) checks that the expected Q-learning target at the engine's one
# stochastic pair equals the value-iteration Bellman-optimality backup, and that a
# sample average of the same bracket concentrates on it as the sample size grows;
# (2) runs tabular Q-learning over ten fixed seeds against the exact Q*;
# (3) compares the projected TD(lambda) operator norm for a one-dimensional feature
# with the contraction bound in Theorem thm:proj_bellman; (4) computes state-action
# concentrability for the target and logging policies used in the appendix.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style, ALGO_COLORS, FIG_SINGLE
from sims.sim_cache import add_cache_args, load_results, save_results
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    discounted_occupancy,
    policy_matrices,
    q_values,
    solve_optimal,
    stationary_distribution,
    stochastic_policy_matrices,
)

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_value_learning"

SEEDS = list(range(42, 52))
Q_ITERATIONS = 5
MC_SAMPLE_SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
QLEARN_T = 40_000  # synchronous seeded sweeps
QLEARN_OMEGA = 0.6  # step size alpha_n(s,a) = n(s,a)^{-omega}, omega in (1/2, 1)
LAMBDAS = [0.0, 0.5, 0.9, 1.0]
PHI_CONTRAST = np.array([1.0, -1.0])  # the V(low) - V(high) contrast feature
BEHAVIOR_KEEP_PROB = 0.1
CONFIG = {
    "version": 1,
    "seeds": SEEDS,
    "q_iterations": Q_ITERATIONS,
    "mc_sample_sizes": MC_SAMPLE_SIZES,
    "qlearn_t": QLEARN_T,
    "qlearn_omega": QLEARN_OMEGA,
    "lambdas": LAMBDAS,
    "phi_contrast": PHI_CONTRAST.tolist(),
    "behavior_keep_prob": BEHAVIOR_KEEP_PROB,
    "gamma": GAMMA,
}


def projected_td_lambda_norm(P_pi, d_pi, phi, gamma, lam):
    """Weighted norm of Pi T^(lambda)' for the declared feature class."""
    n = P_pi.shape[0]
    L = gamma * (1 - lam) * P_pi @ np.linalg.inv(np.eye(n) - lam * gamma * P_pi)
    Phi = np.asarray(phi, dtype=float).reshape(-1, 1)
    D = np.diag(d_pi)
    projection = Phi @ np.linalg.inv(Phi.T @ D @ Phi) @ Phi.T @ D
    projected = projection @ L
    d_sqrt = np.sqrt(d_pi)
    M = np.diag(d_sqrt) @ projected @ np.diag(1.0 / d_sqrt)
    return np.linalg.norm(M, ord=2)


def state_action_concentrability(P, r, greedy_star):
    """Discounted target-to-log occupancy ratio from the shared low-mileage start."""
    nu = np.array([1.0, 0.0])
    P_star, _ = policy_matrices(P, r, list(greedy_star))
    P_mu, _, behavior = stochastic_policy_matrices(P, r, BEHAVIOR_KEEP_PROB)
    occ_star = discounted_occupancy(P_star, GAMMA, nu)
    occ_mu = discounted_occupancy(P_mu, GAMMA, nu)
    d_star = np.zeros((2, 2))
    for state, action in enumerate(greedy_star):
        d_star[state, action] = occ_star[state]
    d_mu = occ_mu[:, None] * behavior
    ratios = np.divide(d_star, d_mu, out=np.zeros_like(d_star), where=d_mu > 0)
    return float(ratios.max()), d_star, d_mu, ratios


def compute_data(force=False):
    if not force:
        cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if cached is not None:
            print("  Cache hit: engine_value_learning")
            return cached

    P, r = build_mdp()
    V_star, greedy_star, Q_star = solve_optimal(P, r, GAMMA)
    print(f"V* = ({V_star[LOW]:.4f}, {V_star[HIGH]:.4f}), Q* =")
    print(f"  {Q_star}")
    print()

    # --- item 1: the expected Q-learning update is the Q-iteration update ---
    Q5 = np.zeros((2, 2))
    for _ in range(Q_ITERATIONS):
        Q5 = q_values(P, r, Q5.max(axis=1), GAMMA)
    print(
        f"Q_{Q_ITERATIONS} = F^{Q_ITERATIONS} Q_0 after {Q_ITERATIONS} "
        "Q-iteration backups from Q_0 = 0:"
    )
    print(f"  {Q5}")
    V5 = Q5.max(axis=1)
    exact_next_Q = q_values(P, r, V5, GAMMA)
    exact_target = exact_next_Q[LOW, KEEP]
    print(
        f"exact Bellman-optimality target (F Q_{Q_ITERATIONS})(low, keep) = "
        f"{exact_target:.6f}"
    )
    print()

    mc_estimates = {}
    print(
        "Monte Carlo estimate of the same target, r(low,keep) + gamma * E[max_a' Q_5(s',a')]:"
    )
    print("  N sampled    mean estimate       SE     mean |error|")
    for N in MC_SAMPLE_SIZES:
        estimates = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            s_next = rng.choice([LOW, HIGH], size=N, p=P[LOW, KEEP])
            samples = r[LOW, KEEP] + GAMMA * V5[s_next]
            estimates.append(float(samples.mean()))
        estimates = np.asarray(estimates)
        mean_estimate = float(estimates.mean())
        estimate_se = float(estimates.std(ddof=1) / np.sqrt(len(SEEDS)))
        mean_abs_error = float(np.mean(np.abs(estimates - exact_target)))
        mc_estimates[N] = {
            "mean": mean_estimate,
            "se": estimate_se,
            "mean_abs_error": mean_abs_error,
        }
        print(
            f"  {N:10d}  {mean_estimate:13.6f}  {estimate_se:8.6f}  "
            f"{mean_abs_error:12.6f}"
        )
    print()
    assert abs(mc_estimates[MC_SAMPLE_SIZES[-1]]["mean"] - exact_target) < 0.01, (
        "largest-N Monte Carlo estimate did not concentrate on the exact target"
    )

    # --- item 2: ten-seed tabular Q-learning trace against Q* ---
    seed_errors = np.zeros((len(SEEDS), QLEARN_T))
    final_tables = []
    for seed_idx, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        Q = np.zeros((2, 2))
        visits = np.zeros((2, 2))
        for t in range(QLEARN_T):
            Q_next = Q.copy()
            for s in (LOW, HIGH):
                for a in (KEEP, REPLACE):
                    s_next = rng.choice(2, p=P[s, a])
                    target = r[s, a] + GAMMA * Q[s_next].max()
                    visits[s, a] += 1
                    alpha = visits[s, a] ** (-QLEARN_OMEGA)
                    Q_next[s, a] = Q[s, a] + alpha * (target - Q[s, a])
            Q = Q_next
            seed_errors[seed_idx, t] = np.max(np.abs(Q - Q_star))
        final_tables.append(Q.copy())
    mean_errors = seed_errors.mean(axis=0)
    error_se = seed_errors.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    print(
        f"Tabular Q-learning over {len(SEEDS)} fixed seeds, step size "
        f"alpha_n = n^-{QLEARN_OMEGA} per (s,a), {QLEARN_T} synchronous sweeps:"
    )
    print("  sweep      mean error          SE")
    checkpoints = [0, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, QLEARN_T - 1]
    for t in checkpoints:
        print(f"  {t:6d}  {mean_errors[t]:14.6f}  {error_se[t]:10.6f}")
    final_errors = seed_errors[:, -1]
    final_error = float(final_errors.mean())
    final_error_se = float(final_errors.std(ddof=1) / np.sqrt(len(SEEDS)))
    print(f"  final error range: [{final_errors.min():.6f}, {final_errors.max():.6f}]")
    print(f"  seed-mean final Q table: {np.mean(final_tables, axis=0)}")
    print()
    assert np.max(final_errors) < 0.05, (
        f"Q-learning did not converge for every seed: max final error {final_errors.max()}"
    )

    # --- item 3: projected TD(lambda) modulus against the theorem's bound ---
    P_star, _ = policy_matrices(P, r, list(greedy_star))
    d_star = stationary_distribution(P_star)
    print(
        f"Stationary distribution of P^(pi*): d = ({d_star[LOW]:.4f}, {d_star[HIGH]:.4f})"
    )
    print(
        "Projected TD(lambda) contraction modulus for phi = (1,-1), compared with "
        "kappa_lambda = gamma(1-lambda)/(1-lambda*gamma)"
    )
    print("  lambda       bound     projected norm")
    kappas = {}
    for lam in LAMBDAS:
        bound = GAMMA * (1 - lam) / (1 - lam * GAMMA)
        projected_norm = projected_td_lambda_norm(
            P_star, d_star, PHI_CONTRAST, GAMMA, lam
        )
        kappas[lam] = (bound, projected_norm)
        print(f"  {lam:6.2f}  {bound:10.6f}  {projected_norm:14.6f}")
        assert projected_norm <= bound + 1e-12, (
            f"projection exceeded bound at lambda={lam}"
        )
    print()

    # --- item 4: state-action concentrability ---
    C_inf, d_sa_star, d_sa_mu, ratios = state_action_concentrability(P, r, greedy_star)
    print("State-action discounted occupancies and target-to-log ratios:")
    print("  state action        target         log       ratio")
    names_s = ["low", "high"]
    names_a = ["keep", "replace"]
    for s in (LOW, HIGH):
        for a in (KEEP, REPLACE):
            print(
                f"  {names_s[s]:5s} {names_a[a]:7s}  {d_sa_star[s, a]:10.6f}  "
                f"{d_sa_mu[s, a]:10.6f}  {ratios[s, a]:10.6f}"
            )
    print(f"  C_inf = {C_inf:.6f}")
    assert abs(C_inf - 1910 / 261) < 1e-9
    print()

    data = {
        "V_star": V_star,
        "Q_star": Q_star,
        "Q5": Q5,
        "exact_target": exact_target,
        "mc_estimates": mc_estimates,
        "mean_errors": mean_errors,
        "error_se": error_se,
        "final_error": final_error,
        "final_error_se": final_error_se,
        "d_star": d_star,
        "kappas": kappas,
        "C_inf": C_inf,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    errors = data["mean_errors"]
    error_se = data["error_se"]
    ax.semilogy(
        errors,
        color=ALGO_COLORS["Q-Learning"],
        linewidth=1.3,
        label=r"mean over 10 seeds",
    )
    lower = np.maximum(errors - error_se, np.finfo(float).tiny)
    upper = errors + error_se
    ax.fill_between(
        np.arange(1, len(errors) + 1),
        lower,
        upper,
        color=ALGO_COLORS["Q-Learning"],
        alpha=0.2,
        linewidth=0,
        label=r"mean $\pm$ standard error",
    )
    ax.set_xlabel("sweep $t$")
    ax.set_ylabel(r"$\|Q_t - Q^*\|_\infty$")
    ax.set_title("Q-learning error on the Engine Replacement MDP")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "engine_value_learning.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    N_report = MC_SAMPLE_SIZES[-1]
    estimate = data["mc_estimates"][N_report]
    kappas = data["kappas"]
    tex_path = os.path.join(OUTPUT_DIR, "engine_value_learning.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Value-learning quantities computed on the Engine Replacement MDP. "
            "The Bellman-operator identity and Monte Carlo estimate use the intermediate "
            f"table $Q_{{{Q_ITERATIONS}}}$ at $(s,a) = (\\text{{low}}, \\text{{keep}})$, "
            f"the engine's one stochastic transition. The Q-learning trace uses "
            f"{len(SEEDS)} fixed seeds and {QLEARN_T:,} synchronous sweeps with step size "
            "$\\alpha_n = n^{-0.6}$. Projected TD($\\lambda$) moduli use "
            "$\\phi=(1,-1)$ and the stationary distribution of $P^{\\pi^\\star}$. "
            "Concentrability uses the discounted occupancy from the low-mileage start.}\n"
        )
        f.write("\\label{tab:engine_value_learning}\n")
        f.write("\\begin{tabular}{p{0.68\\textwidth}r}\n\\hline\n")
        f.write("quantity & value \\\\\n\\hline\n")
        f.write(
            f"exact target $(FQ_{Q_ITERATIONS})(\\text{{low}},\\text{{keep}})$ "
            f"& {data['exact_target']:.4f} \\\\\n"
        )
        f.write(
            f"Monte Carlo mean, $N = {N_report:,}$ & "
            f"{estimate['mean']:.4f} $\\pm$ {estimate['se']:.4f} \\\\\n"
        )
        f.write(
            f"mean absolute sampling error, $N = {N_report:,}$ "
            f"& {estimate['mean_abs_error']:.4f} \\\\\n"
        )
        f.write(
            f"Q-learning final error, mean $\\pm$ SE "
            f"& {data['final_error']:.4f} $\\pm$ {data['final_error_se']:.4f} \\\\\n"
        )
        for lam in LAMBDAS:
            bound, projected_norm = kappas[lam]
            f.write(
                f"projected TD($\\lambda={lam:g}$) norm, bound {bound:.4f} "
                f"& {projected_norm:.4f} \\\\\n"
            )
        f.write(
            f"state-action concentrability $C_\\infty$ & {data['C_inf']:.4f} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Value-learning objects on the Engine Replacement MDP"
    )
    add_cache_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute the fixed-seed experiment even when a valid cache exists",
    )
    args = parser.parse_args()
    print("=" * 70)
    print("SAMPLING THE BELLMAN OPERATOR ON THE ENGINE REPLACEMENT MDP")
    print("=" * 70)
    print()
    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if data is None:
            raise FileNotFoundError("No cache found. Run without --plots-only first.")
    else:
        data = compute_data(force=args.force)
    if not args.data_only:
        generate_outputs(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
