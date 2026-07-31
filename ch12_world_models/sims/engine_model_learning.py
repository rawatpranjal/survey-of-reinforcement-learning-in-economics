# Learning the Engine Replacement MDP transition model
# Chapter 12 - World Models and Model-Based Reinforcement Learning
# Estimates all four binary transition rows, plans in each empirical model, and
# compares optimal-value error with a discounted simulation-lemma bound.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import ACTION_NAMES, GAMMA, STATE_NAMES, build_mdp, solve_optimal
from sims.plot_style import COLORS, FIG_SINGLE, apply_style
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_model_learning"

CONFIG = {
    "sample_sizes": [25, 50, 100, 250, 500, 1_000, 2_500, 5_000],
    "n_seeds": 200,
    "seed_start": 2900,
}


def empirical_kernel(samples, sample_size):
    P_hat = np.empty((2, 2, 2))
    for s in range(2):
        for a in range(2):
            p_high = float(samples[s, a, :sample_size].mean())
            P_hat[s, a] = [1.0 - p_high, p_high]
    return P_hat


def compute_data(force=None):
    if not force:
        cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if cached is not None:
            print("Cache hit")
            return cached

    P, rewards = build_mdp()
    V_star, policy_star, _ = solve_optimal(P, rewards, GAMMA)
    true_p_high = P[:, :, 1]
    max_sample_size = max(CONFIG["sample_sizes"])
    n_sizes = len(CONFIG["sample_sizes"])
    estimates = np.empty((n_sizes, CONFIG["n_seeds"], 2, 2))
    value_errors = np.empty((n_sizes, CONFIG["n_seeds"]))
    bounds = np.empty((n_sizes, CONFIG["n_seeds"]))
    policy_losses = np.empty((n_sizes, CONFIG["n_seeds"]))
    policy_matches = np.empty((n_sizes, CONFIG["n_seeds"]), dtype=bool)
    kernel_l1_errors = np.empty((n_sizes, CONFIG["n_seeds"]))
    reward_bound = float(np.max(np.abs(rewards)))

    for seed_index in range(CONFIG["n_seeds"]):
        rng = np.random.default_rng(CONFIG["seed_start"] + seed_index)
        uniforms = rng.random((2, 2, max_sample_size))
        samples = uniforms < true_p_high[:, :, None]
        for size_index, sample_size in enumerate(CONFIG["sample_sizes"]):
            P_hat = empirical_kernel(samples, sample_size)
            V_hat, policy_hat, _ = solve_optimal(P_hat, rewards, GAMMA)
            estimates[size_index, seed_index] = P_hat[:, :, 1]
            value_errors[size_index, seed_index] = np.max(np.abs(V_hat - V_star))
            kernel_l1 = np.max(np.sum(np.abs(P_hat - P), axis=2))
            kernel_l1_errors[size_index, seed_index] = kernel_l1
            bounds[size_index, seed_index] = (
                GAMMA * reward_bound * kernel_l1 / (1.0 - GAMMA) ** 2
            )
            policy_matches[size_index, seed_index] = np.array_equal(
                policy_hat, policy_star
            )
            P_true_policy = np.array([P[s, policy_hat[s]] for s in range(P.shape[0])])
            r_true_policy = np.array(
                [rewards[s, policy_hat[s]] for s in range(P.shape[0])]
            )
            V_true_policy = np.linalg.solve(
                np.eye(P.shape[0]) - GAMMA * P_true_policy, r_true_policy
            )
            policy_losses[size_index, seed_index] = np.max(V_star - V_true_policy)

    if np.any(value_errors > bounds + 1e-10):
        raise AssertionError(
            "a realized value error exceeds the simulation-lemma bound"
        )

    mean_errors = value_errors.mean(axis=1)
    mean_bounds = bounds.mean(axis=1)
    log_sizes = np.log(np.asarray(CONFIG["sample_sizes"], dtype=float))
    error_slope = float(np.polyfit(log_sizes, np.log(mean_errors), 1)[0])
    bound_slope = float(np.polyfit(log_sizes, np.log(mean_bounds), 1)[0])
    assert mean_errors[-1] < mean_errors[0]
    assert mean_bounds[-1] < mean_bounds[0]
    assert error_slope < 0.0 and bound_slope < 0.0

    data = {
        "P": P,
        "rewards": rewards,
        "V_star": V_star,
        "policy_star": policy_star,
        "true_p_high": true_p_high,
        "estimates": estimates,
        "value_errors": value_errors,
        "bounds": bounds,
        "policy_losses": policy_losses,
        "policy_matches": policy_matches,
        "kernel_l1_errors": kernel_l1_errors,
        "error_slope": error_slope,
        "bound_slope": bound_slope,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def standard_error(values):
    return values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])


def print_results(data):
    print("LEARNING THE ENGINE REPLACEMENT MDP TRANSITIONS")
    print(
        f"seeds {CONFIG['n_seeds']}  seed start {CONFIG['seed_start']}  "
        f"gamma {GAMMA:.3f}"
    )
    print("true P(next=high | state, action)")
    for s, state in enumerate(STATE_NAMES):
        for a, action in enumerate(ACTION_NAMES):
            print(f"  {state:5s} {action:7s} {data['true_p_high'][s, a]:.6f}")
    print()
    print(
        "N      p_LK      p_LR      p_HK      p_HR"
        "   value error       SE       bound       SE   policy match"
    )
    error_se = standard_error(data["value_errors"])
    bound_se = standard_error(data["bounds"])
    for i, sample_size in enumerate(CONFIG["sample_sizes"]):
        estimate_means = data["estimates"][i].mean(axis=0)
        print(
            f"{sample_size:5d} {estimate_means[0, 0]:10.6f}"
            f" {estimate_means[0, 1]:10.6f} {estimate_means[1, 0]:10.6f}"
            f" {estimate_means[1, 1]:10.6f}"
            f" {data['value_errors'][i].mean():13.6f} {error_se[i]:9.6f}"
            f" {data['bounds'][i].mean():11.6f} {bound_se[i]:9.6f}"
            f" {data['policy_matches'][i].mean():14.6f}"
        )
    print()
    print(f"log-log value-error slope {data['error_slope']:.6f}")
    print(f"log-log bound slope {data['bound_slope']:.6f}")
    print(
        f"maximum realized error-minus-bound "
        f"{np.max(data['value_errors'] - data['bounds']):.3e}"
    )
    print(f"maximum true-policy loss {data['policy_losses'].max():.6f}")


def generate_outputs(data):
    import matplotlib.pyplot as plt

    sample_sizes = np.asarray(CONFIG["sample_sizes"])
    mean_errors = data["value_errors"].mean(axis=1)
    error_se = standard_error(data["value_errors"])
    mean_bounds = data["bounds"].mean(axis=1)
    bound_se = standard_error(data["bounds"])

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(
        sample_sizes,
        mean_errors,
        marker="o",
        color=COLORS["blue"],
        label=r"$\|\widehat V^*-V^*\|_\infty$",
    )
    ax.fill_between(
        sample_sizes,
        mean_errors - error_se,
        mean_errors + error_se,
        color=COLORS["blue"],
        alpha=0.18,
    )
    ax.plot(
        sample_sizes,
        mean_bounds,
        marker="s",
        color=COLORS["orange"],
        label="simulation-lemma bound",
    )
    ax.fill_between(
        sample_sizes,
        mean_bounds - bound_se,
        mean_bounds + bound_se,
        color=COLORS["orange"],
        alpha=0.18,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("transition samples per state-action pair $N$")
    ax.set_ylabel("sup-norm value difference")
    ax.legend()
    fig.tight_layout()
    figure_path = os.path.join(OUTPUT_DIR, "engine_model_learning.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved {figure_path}")

    table_path = os.path.join(OUTPUT_DIR, "engine_model_learning.tex")
    with open(table_path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{Learning the four Engine Replacement MDP transition rows. "
            "Value errors, bounds, and standard errors use two hundred fixed seeds.}\n"
        )
        f.write("\\label{tab:engine_model_learning}\n")
        f.write("\\begin{tabular}{rrrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "$N$ & mean kernel error & value error & s.e. "
            "& simulation bound & s.e. \\\\\n"
        )
        f.write("\\hline\n")
        for i, sample_size in enumerate(CONFIG["sample_sizes"]):
            f.write(
                f"{sample_size:,} & {data['kernel_l1_errors'][i].mean():.4f} "
                f"& {mean_errors[i]:.4f} & {error_se[i]:.4f} "
                f"& {mean_bounds[i]:.4f} & {bound_se[i]:.4f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write(
            "\\multicolumn{6}{l}{Final mean estimates of $P(\\mathrm{high}\\mid s,a)$} \\\\\n"
        )
        f.write("\\hline\n")
        f.write(
            "low, keep & low, replace & high, keep & high, replace "
            "& \\multicolumn{2}{c}{optimal-policy matches} \\\\\n"
        )
        f.write("\\hline\n")
        final_estimates = data["estimates"][-1].mean(axis=0)
        f.write(
            f"{final_estimates[0, 0]:.4f} & {final_estimates[0, 1]:.4f} "
            f"& {final_estimates[1, 0]:.4f} & {final_estimates[1, 1]:.4f} "
            f"& \\multicolumn{{2}}{{c}}{{{data['policy_matches'][-1].mean():.3f}}} \\\\\n"
        )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Table saved {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Learn the Engine Replacement MDP transition model"
    )
    add_cache_args(parser)
    args = parser.parse_args()
    cache_path = os.path.join(CACHE_DIR, f"{SCRIPT_NAME}.pkl")
    if args.plots_only and not os.path.exists(cache_path):
        raise FileNotFoundError("plots-only requires a matching cache")
    data = compute_data(force=not args.plots_only)
    print_results(data)
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
