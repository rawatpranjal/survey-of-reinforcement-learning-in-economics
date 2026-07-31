# Every importance weight on the Engine Replacement MDP
# Chapter 10b - Off-Policy Evaluation and Dynamic Treatment Effects
# Evaluates the optimal replacement policy from an independent behavior-policy log,
# prints every demonstration weight, and measures horizon-driven variance.

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    ACTION_NAMES,
    STATE_NAMES,
    build_mdp,
    solve_optimal,
)
from sims.plot_style import (
    ALGO_COLORS,
    COLORS,
    FIG_DOUBLE,
    apply_style,
)
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()

import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_ope"

DEMO_HORIZON = 4
DEMO_TRAJECTORIES = 12
DEMO_SEED = 42
MODEL_TRAJECTORIES = 200
MODEL_HORIZON = 10
MODEL_SEED = 31415
HORIZONS = (2, 4, 6, 8, 10)
MC_TRAJECTORIES = 500
MC_SEEDS = tuple(range(1000, 1040))
DIRICHLET_ALPHA = 0.5

CONFIG = {
    "demo_horizon": DEMO_HORIZON,
    "demo_trajectories": DEMO_TRAJECTORIES,
    "demo_seed": DEMO_SEED,
    "model_trajectories": MODEL_TRAJECTORIES,
    "model_horizon": MODEL_HORIZON,
    "model_seed": MODEL_SEED,
    "horizons": HORIZONS,
    "mc_trajectories": MC_TRAJECTORIES,
    "mc_seeds": MC_SEEDS,
    "dirichlet_alpha": DIRICHLET_ALPHA,
    "gamma": GAMMA,
    "version": 2,
    "wis_zero_mass": "conditional_rmse_with_valid_count",
}


def policy_matrices():
    target = np.zeros((2, 2))
    target[LOW, KEEP] = 1.0
    target[HIGH, REPLACE] = 1.0
    behavior = np.full((2, 2), 0.5)
    return target, behavior


def finite_horizon_dp(P, r, target, horizon):
    Q = np.zeros((horizon, 2, 2))
    V = np.zeros((horizon + 1, 2))
    for t in range(horizon - 1, -1, -1):
        Q[t] = r + GAMMA * P @ V[t + 1]
        V[t] = (target * Q[t]).sum(axis=1)
    return float(V[0, LOW]), Q, V


def simulate_log(P, r, policy, horizon, n_trajectories, seed):
    rng = np.random.default_rng(seed)
    states = np.zeros((n_trajectories, horizon), dtype=np.int64)
    actions = np.zeros_like(states)
    rewards = np.zeros((n_trajectories, horizon))
    next_states = np.zeros_like(states)
    state = np.full(n_trajectories, LOW, dtype=np.int64)

    for t in range(horizon):
        states[:, t] = state
        action_draw = rng.random(n_trajectories)
        actions[:, t] = (action_draw >= policy[state, KEEP]).astype(np.int64)
        rewards[:, t] = r[state, actions[:, t]]
        transition_draw = rng.random(n_trajectories)
        transition_cdf = np.cumsum(P[state, actions[:, t]], axis=1)
        state = (transition_draw[:, None] > transition_cdf).sum(axis=1)
        next_states[:, t] = state

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
    }


def fit_tabular_model(log, r):
    P_hat = np.full((2, 2, 2), 0.5)
    r_hat = np.zeros((2, 2))
    counts = np.zeros((2, 2), dtype=np.int64)
    states = log["states"]
    actions = log["actions"]

    for state in (LOW, HIGH):
        for action in (KEEP, REPLACE):
            mask = (states == state) & (actions == action)
            count = int(mask.sum())
            counts[state, action] = count
            if count == 0:
                continue
            next_counts = np.bincount(log["next_states"][mask], minlength=2)
            P_hat[state, action] = (next_counts + DIRICHLET_ALPHA) / (
                count + 2.0 * DIRICHLET_ALPHA
            )
            r_hat[state, action] = float(log["rewards"][mask].mean())

    assert np.all(counts > 0), "Independent nuisance log missed a state-action pair"
    assert np.max(np.abs(r_hat - r)) < 1e-12
    return P_hat, r_hat, counts


def score_log(log, target, behavior, Q_hat, V_hat):
    states = log["states"]
    actions = log["actions"]
    rewards = log["rewards"]
    horizon = states.shape[1]
    ratios = target[states, actions] / behavior[states, actions]
    cumulative = np.cumprod(ratios, axis=1)
    discounts = GAMMA ** np.arange(horizon)
    returns = (rewards * discounts).sum(axis=1)

    is_scores = cumulative[:, -1] * returns
    pdis_scores = (cumulative * rewards * discounts).sum(axis=1)
    weight_mass = cumulative.sum(axis=0)
    if np.any(weight_mass <= 0.0):
        wis = np.nan
    else:
        wis = float(
            sum(
                discounts[t] * np.dot(cumulative[:, t], rewards[:, t]) / weight_mass[t]
                for t in range(horizon)
            )
        )

    dr_scores = np.zeros(states.shape[0])
    for t in range(horizon - 1, -1, -1):
        state = states[:, t]
        action = actions[:, t]
        dr_scores = V_hat[t, state] + ratios[:, t] * (
            rewards[:, t] + GAMMA * dr_scores - Q_hat[t, state, action]
        )

    return {
        "ratios": ratios,
        "cumulative": cumulative,
        "returns": returns,
        "is_scores": is_scores,
        "pdis_scores": pdis_scores,
        "dr_scores": dr_scores,
        "IS": float(is_scores.mean()),
        "PDIS": float(pdis_scores.mean()),
        "WIS": wis,
        "DR": float(dr_scores.mean()),
    }


def exact_weight_variance(P, target, behavior, horizon):
    second_moment_kernel = np.zeros((2, 2))
    for state in (LOW, HIGH):
        for action in (KEEP, REPLACE):
            second_moment_kernel[state] += (
                target[state, action] ** 2 / behavior[state, action] * P[state, action]
            )
    second_moment = (
        np.array([1.0, 0.0])
        @ np.linalg.matrix_power(second_moment_kernel, horizon)
        @ np.ones(2)
    )
    return float(second_moment - 1.0)


def compute_fresh():
    P, r = build_mdp()
    _, optimal_policy, _ = solve_optimal(P, r, GAMMA)
    assert tuple(optimal_policy) == (KEEP, REPLACE)
    target, behavior = policy_matrices()

    nuisance_log = simulate_log(
        P, r, behavior, MODEL_HORIZON, MODEL_TRAJECTORIES, MODEL_SEED
    )
    P_hat, r_hat, nuisance_counts = fit_tabular_model(nuisance_log, r)

    truth, _, _ = finite_horizon_dp(P, r, target, DEMO_HORIZON)
    dm, Q_hat, V_hat = finite_horizon_dp(P_hat, r_hat, target, DEMO_HORIZON)
    demo_log = simulate_log(P, r, behavior, DEMO_HORIZON, DEMO_TRAJECTORIES, DEMO_SEED)
    demo_scores = score_log(demo_log, target, behavior, Q_hat, V_hat)
    estimates = {
        "Exact DP": truth,
        "DM": dm,
        "IS": demo_scores["IS"],
        "PDIS": demo_scores["PDIS"],
        "WIS": demo_scores["WIS"],
        "DR": demo_scores["DR"],
    }
    finite_estimates = np.array(
        [estimates[name] for name in ("DM", "IS", "PDIS", "WIS", "DR")]
    )
    assert finite_estimates.min() < truth < finite_estimates.max()

    horizon_results = []
    for horizon in HORIZONS:
        horizon_truth, _, _ = finite_horizon_dp(P, r, target, horizon)
        dm_h, Q_hat_h, V_hat_h = finite_horizon_dp(P_hat, r_hat, target, horizon)
        replications = {name: [] for name in ("IS", "PDIS", "WIS", "DR")}
        weight_variances = []
        for seed in MC_SEEDS:
            log = simulate_log(P, r, behavior, horizon, MC_TRAJECTORIES, seed)
            scores = score_log(log, target, behavior, Q_hat_h, V_hat_h)
            for name in replications:
                replications[name].append(scores[name])
            weight_variances.append(float(np.var(scores["cumulative"][:, -1], ddof=1)))

        exact_variance = exact_weight_variance(P, target, behavior, horizon)
        weight_variances = np.asarray(weight_variances)
        row = {
            "horizon": horizon,
            "truth": horizon_truth,
            "DM": dm_h,
            "exact_weight_variance": exact_variance,
            "empirical_weight_variance": float(weight_variances.mean()),
            "empirical_weight_variance_se": float(
                weight_variances.std(ddof=1) / np.sqrt(len(MC_SEEDS))
            ),
        }
        for name, values in replications.items():
            values = np.asarray(values)
            finite = np.isfinite(values)
            row[f"{name}_valid"] = int(finite.sum())
            if name != "WIS":
                assert finite.all(), f"{name} produced a non-finite replication"
            row[f"{name}_rmse"] = float(
                np.sqrt(np.mean((values[finite] - horizon_truth) ** 2))
            )
        assert row["WIS_valid"] > 0, "WIS was undefined in every replication"
        horizon_results.append(row)

    return {
        "P_hat": P_hat,
        "nuisance_counts": nuisance_counts,
        "truth": truth,
        "demo_log": demo_log,
        "demo_scores": demo_scores,
        "estimates": estimates,
        "horizon_results": horizon_results,
    }


def compute_data(force=False):
    if not force:
        cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if cached is not None:
            print("Cache hit")
            return cached
    data = compute_fresh()
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def print_results(data):
    print("=" * 78)
    print("ENGINE REPLACEMENT OFF-POLICY EVALUATION")
    print("=" * 78)
    print("Parameters")
    print(f"  gamma                         {GAMMA:.1f}")
    print(f"  demonstration horizon         {DEMO_HORIZON}")
    print(f"  demonstration trajectories    {DEMO_TRAJECTORIES}")
    print(f"  demonstration seed            {DEMO_SEED}")
    print(f"  nuisance trajectories          {MODEL_TRAJECTORIES}")
    print(f"  nuisance seed                  {MODEL_SEED}")
    print(f"  Monte Carlo seeds              {len(MC_SEEDS)}")
    print(f"  trajectories per MC seed       {MC_TRAJECTORIES}")
    print("  behavior keep probability      0.5000 at both states")
    print("  target policy                  keep at low, replace at high")
    print()
    print("Independent nuisance-log counts")
    print("  state      keep    replace")
    for state in (LOW, HIGH):
        counts = data["nuisance_counts"][state]
        print(f"  {STATE_NAMES[state]:<8s} {counts[KEEP]:7d} {counts[REPLACE]:10d}")
    print()
    print("Every demonstration importance weight")
    print("  traj time state action   rho_t   rho_1:t  reward")
    log = data["demo_log"]
    scores = data["demo_scores"]
    for i in range(DEMO_TRAJECTORIES):
        for t in range(DEMO_HORIZON):
            state = log["states"][i, t]
            action = log["actions"][i, t]
            print(
                f"  {i + 1:4d} {t + 1:4d} {STATE_NAMES[state]:>5s} "
                f"{ACTION_NAMES[action]:>7s} "
                f"{scores['ratios'][i, t]:7.3f} "
                f"{scores['cumulative'][i, t]:9.3f} "
                f"{log['rewards'][i, t]:7.3f}"
            )
    print()
    print("Point estimates on the demonstration log")
    print("  estimator       estimate       error")
    truth = data["truth"]
    for name in ("Exact DP", "DM", "IS", "PDIS", "WIS", "DR"):
        estimate = data["estimates"][name]
        error = estimate - truth
        print(f"  {name:<12s} {estimate:10.6f} {error:11.6f}")
    print()
    print("Horizon experiment")
    print(
        "  H  exact Var(weight)  empirical Var(weight)      SE       "
        "IS RMSE  PDIS RMSE  WIS RMSE  WIS n   DR RMSE"
    )
    for row in data["horizon_results"]:
        print(
            f"  {row['horizon']:2d} {row['exact_weight_variance']:18.3f} "
            f"{row['empirical_weight_variance']:22.3f} "
            f"{row['empirical_weight_variance_se']:8.3f} "
            f"{row['IS_rmse']:9.4f} {row['PDIS_rmse']:10.4f} "
            f"{row['WIS_rmse']:9.4f} {row['WIS_valid']:6d} "
            f"{row['DR_rmse']:9.4f}"
        )


def generate_table(data):
    path = os.path.join(OUTPUT_DIR, "engine_ope.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Every per-step and cumulative importance weight in the twelve-trajectory Engine Replacement MDP log. Panel B compares the five estimates with the exact finite-horizon dynamic-programming value.}\n"
        )
        f.write("\\label{tab:engine_ope}\n")
        f.write("{\\scriptsize\\renewcommand{\\arraystretch}{1.05}\n")
        f.write("\\begin{tabular}{crrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "& \\multicolumn{4}{c}{$\\rho_t\\,/\\,\\rho_{1:t}$} & discounted return \\\\\n"
        )
        f.write("\\cmidrule(lr){2-5}\n")
        f.write("trajectory & $t=1$ & $t=2$ & $t=3$ & $t=4$ & $G_i$ \\\\\n")
        f.write("\\midrule\n")
        ratios = data["demo_scores"]["ratios"]
        cumulative = data["demo_scores"]["cumulative"]
        returns = data["demo_scores"]["returns"]
        for i in range(DEMO_TRAJECTORIES):
            cells = " & ".join(
                f"{ratios[i, t]:.0f}\\,/\\,{cumulative[i, t]:.0f}"
                for t in range(DEMO_HORIZON)
            )
            f.write(f"{i + 1} & {cells} & {returns[i]:.3f} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{6}{c}{Panel B. Point estimates} \\\\\n")
        f.write("\\midrule\n")
        f.write(
            "estimator & \\multicolumn{2}{c}{estimate} & \\multicolumn{3}{c}{estimate minus exact value} \\\\\n"
        )
        truth = data["truth"]
        for name in ("Exact DP", "DM", "IS", "PDIS", "WIS", "DR"):
            estimate = data["estimates"][name]
            f.write(
                f"{name} & \\multicolumn{{2}}{{c}}{{{estimate:.4f}}} "
                f"& \\multicolumn{{3}}{{c}}{{{estimate - truth:+.4f}}} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table}\n")
    return path


def generate_figure(data):
    rows = data["horizon_results"]
    horizons = np.array([row["horizon"] for row in rows])
    exact_variance = np.array([row["exact_weight_variance"] for row in rows])
    empirical_variance = np.array([row["empirical_weight_variance"] for row in rows])
    empirical_se = np.array([row["empirical_weight_variance_se"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    axes[0].plot(
        horizons,
        exact_variance,
        color=COLORS["black"],
        marker="o",
        label="exact variance",
    )
    axes[0].errorbar(
        horizons,
        empirical_variance,
        yerr=empirical_se,
        color=COLORS["orange"],
        marker="s",
        capsize=3,
        label="Monte Carlo mean $\\pm$ SE",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("horizon")
    axes[0].set_ylabel("variance of $\\rho_{1:H}$")
    axes[0].set_title("(a) Trajectory-weight variance")
    axes[0].legend()

    method_styles = {
        "IS": (COLORS["gray"], "o"),
        "PDIS": (COLORS["orange"], "s"),
        "WIS": (COLORS["green"], "^"),
        "DR": (ALGO_COLORS["Q-Learning"], "D"),
    }
    for name, (color, marker) in method_styles.items():
        rmse = np.array([row[f"{name}_rmse"] for row in rows])
        label = "WIS, defined logs" if name == "WIS" else name
        axes[1].plot(
            horizons,
            rmse,
            color=color,
            marker=marker,
            label=label,
        )
        if name == "WIS":
            for horizon, error, row in zip(horizons, rmse, rows):
                axes[1].annotate(
                    f"{row['WIS_valid']}/{len(MC_SEEDS)}",
                    (horizon, error),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("horizon")
    axes[1].set_ylabel("RMSE of the value estimate")
    axes[1].set_title("(b) Estimator error")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "engine_ope.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_outputs(data):
    table_path = generate_table(data)
    figure_path = generate_figure(data)
    print()
    print("Output files")
    print(f"  {table_path}")
    print(f"  {figure_path}")


def main():
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()
    data = compute_data(force=not args.plots_only)
    print_results(data)
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
