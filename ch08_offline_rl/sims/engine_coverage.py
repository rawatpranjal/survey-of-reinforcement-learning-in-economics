# Coverage and extrapolation on the Engine Replacement MDP
# Chapter 8 - Offline Reinforcement Learning
# Shows the four discounted state-action occupancies, one absent data pair,
# FQI's unsupported prediction, and CQL's explicit pessimism penalty.

import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import (
    ACTION_NAMES,
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    STATE_NAMES,
    build_mdp,
    discounted_occupancy,
    policy_matrices,
    solve_optimal,
    stochastic_policy_matrices,
)
from sims.plot_style import ALGO_COLORS, BENCH_STYLE, FIG_SINGLE, apply_style
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()

import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_coverage"

N_PER_OBSERVED_PAIR = 40
N_ITERATIONS = 100
CQL_ALPHA = 0.01
RIDGE = 1e-5
LOGGING_KEEP_PROBABILITY = 0.1
EXPECTED_C_INF = 7.318007662835235

CONFIG = {
    "n_per_observed_pair": N_PER_OBSERVED_PAIR,
    "n_iterations": N_ITERATIONS,
    "cql_alpha": CQL_ALPHA,
    "ridge": RIDGE,
    "logging_keep_probability": LOGGING_KEEP_PROBABILITY,
    "gamma": GAMMA,
    "version": 1,
}


def one_hot_features(states, actions):
    indices = 2 * np.asarray(states, dtype=np.int64) + np.asarray(
        actions, dtype=np.int64
    )
    return np.eye(4)[indices]


def build_three_pair_log(r):
    states = []
    actions = []
    rewards = []
    next_states = []
    for state, action in ((LOW, KEEP), (LOW, REPLACE), (HIGH, KEEP)):
        for j in range(N_PER_OBSERVED_PAIR):
            if (state, action) == (LOW, KEEP):
                next_state = LOW if j < N_PER_OBSERVED_PAIR // 2 else HIGH
            elif (state, action) == (LOW, REPLACE):
                next_state = LOW
            else:
                next_state = HIGH
            states.append(state)
            actions.append(action)
            rewards.append(r[state, action])
            next_states.append(next_state)
    return {
        "states": np.asarray(states, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards),
        "next_states": np.asarray(next_states, dtype=np.int64),
    }


def cql_objective(theta, X, targets, dataset_states):
    predictions = X @ theta
    residuals = predictions - targets
    Q = theta.reshape(2, 2)
    conservative_penalty = np.mean(logsumexp(Q[dataset_states], axis=1) - predictions)
    objective = (
        0.5 * np.mean(residuals**2)
        + CQL_ALPHA * conservative_penalty
        + 0.5 * RIDGE * np.dot(theta, theta)
    )
    probabilities = softmax(Q[dataset_states], axis=1)
    all_features = np.eye(4).reshape(2, 2, 4)
    softmax_features = np.einsum(
        "na,nad->nd", probabilities, all_features[dataset_states]
    )
    gradient = (
        X.T @ residuals / len(X)
        + CQL_ALPHA * (softmax_features - X).mean(axis=0)
        + RIDGE * theta
    )
    return objective, gradient


def fit_fqi_and_cql(log):
    X = one_hot_features(log["states"], log["actions"])
    fqi_theta = np.zeros(4)
    cql_theta = np.zeros(4)
    fqi_path = []
    cql_path = []

    for _ in range(N_ITERATIONS):
        fqi_Q = fqi_theta.reshape(2, 2)
        fqi_targets = log["rewards"] + GAMMA * fqi_Q[log["next_states"]].max(axis=1)
        fqi_theta = np.linalg.solve(
            X.T @ X / len(X) + RIDGE * np.eye(4),
            X.T @ fqi_targets / len(X),
        )

        cql_Q = cql_theta.reshape(2, 2)
        cql_targets = log["rewards"] + GAMMA * cql_Q[log["next_states"]].max(axis=1)
        result = minimize(
            lambda theta: cql_objective(theta, X, cql_targets, log["states"])[0],
            cql_theta,
            jac=lambda theta: cql_objective(theta, X, cql_targets, log["states"])[1],
            method="L-BFGS-B",
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
        )
        cql_theta = result.x
        gradient_norm = np.linalg.norm(
            cql_objective(cql_theta, X, cql_targets, log["states"])[1]
        )
        assert gradient_norm < 1e-5, (
            f"CQL optimizer stopped with gradient norm {gradient_norm:.3e}"
        )
        fqi_path.append(fqi_theta.reshape(2, 2).copy())
        cql_path.append(cql_theta.reshape(2, 2).copy())

    return np.asarray(fqi_path), np.asarray(cql_path)


def compute_fresh():
    P, r = build_mdp()
    _, optimal_policy, Q_star = solve_optimal(P, r, GAMMA)
    assert tuple(optimal_policy) == (KEEP, REPLACE)

    nu = np.array([1.0, 0.0])
    P_target, _ = policy_matrices(P, r, optimal_policy)
    target_state_occupancy = discounted_occupancy(P_target, GAMMA, nu)
    target_policy = np.zeros((2, 2))
    target_policy[LOW, KEEP] = 1.0
    target_policy[HIGH, REPLACE] = 1.0
    target_sa_occupancy = target_state_occupancy[:, None] * target_policy

    P_log, _, behavior = stochastic_policy_matrices(P, r, LOGGING_KEEP_PROBABILITY)
    logging_state_occupancy = discounted_occupancy(P_log, GAMMA, nu)
    logging_sa_occupancy = logging_state_occupancy[:, None] * behavior
    ratios = np.divide(
        target_sa_occupancy,
        logging_sa_occupancy,
        out=np.zeros_like(target_sa_occupancy),
        where=logging_sa_occupancy > 0,
    )
    C_inf = float(ratios.max())
    assert abs(C_inf - EXPECTED_C_INF) < 1e-12
    assert abs(logging_sa_occupancy[LOW, REPLACE] - 0.8575916230) < 1e-10
    assert abs(logging_sa_occupancy[HIGH, REPLACE] - 0.0424083770) < 1e-10

    log = build_three_pair_log(r)
    observed = np.zeros((2, 2), dtype=bool)
    observed[log["states"], log["actions"]] = True
    assert observed.sum() == 3
    assert not observed[HIGH, REPLACE]

    fqi_path, cql_path = fit_fqi_and_cql(log)
    fqi_Q = fqi_path[-1]
    cql_Q = cql_path[-1]
    assert fqi_Q[HIGH, REPLACE] == 0.0
    assert cql_Q[HIGH, REPLACE] < fqi_Q[HIGH, REPLACE]

    return {
        "logging_sa_occupancy": logging_sa_occupancy,
        "target_sa_occupancy": target_sa_occupancy,
        "ratios": ratios,
        "C_inf": C_inf,
        "observed": observed,
        "Q_star": Q_star,
        "fqi_path": fqi_path,
        "cql_path": cql_path,
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
    print("ENGINE REPLACEMENT COVERAGE AND EXTRAPOLATION")
    print("=" * 78)
    print("Parameters")
    print(f"  gamma                         {GAMMA:.1f}")
    print(f"  logging keep probability      {LOGGING_KEEP_PROBABILITY:.1f}")
    print(f"  transitions per observed pair {N_PER_OBSERVED_PAIR}")
    print(f"  fitted-Q iterations           {N_ITERATIONS}")
    print(f"  CQL alpha                     {CQL_ALPHA:.3f}")
    print(f"  ridge coefficient             {RIDGE:.1e}")
    print()
    print("Discounted state-action occupancies")
    print("  state action     log pct  target pct observed  ratio")
    for state in (LOW, HIGH):
        for action in (KEEP, REPLACE):
            print(
                f"  {STATE_NAMES[state]:<5s} {ACTION_NAMES[action]:<7s} "
                f"{100 * data['logging_sa_occupancy'][state, action]:8.4f} "
                f"{100 * data['target_sa_occupancy'][state, action]:10.4f} "
                f"{str(bool(data['observed'][state, action])):>8s} "
                f"{data['ratios'][state, action]:7.4f}"
            )
    print(f"  C_inf = {data['C_inf']:.6f}")
    print()
    print("Final action values")
    print("  state action       exact       FQI       CQL")
    fqi_Q = data["fqi_path"][-1]
    cql_Q = data["cql_path"][-1]
    for state in (LOW, HIGH):
        for action in (KEEP, REPLACE):
            print(
                f"  {STATE_NAMES[state]:<5s} {ACTION_NAMES[action]:<7s} "
                f"{data['Q_star'][state, action]:10.4f} "
                f"{fqi_Q[state, action]:9.4f} "
                f"{cql_Q[state, action]:9.4f}"
            )
    print()
    print("Missing-pair path by iteration")
    print("  iteration       FQI       CQL")
    for iteration in (1, 2, 5, 10, 20, 50, 100):
        print(
            f"  {iteration:9d} "
            f"{data['fqi_path'][iteration - 1, HIGH, REPLACE]:9.4f} "
            f"{data['cql_path'][iteration - 1, HIGH, REPLACE]:9.4f}"
        )


def generate_table(data):
    path = os.path.join(OUTPUT_DIR, "engine_coverage.tex")
    fqi_Q = data["fqi_path"][-1]
    cql_Q = data["cql_path"][-1]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Discounted state-action occupancy under the logging and target policies, finite-log coverage, and final action values. The log contains forty transitions for each marked pair and none for high mileage with replacement.}\n"
        )
        f.write("\\label{tab:engine_coverage}\n")
        f.write("{\\small\\renewcommand{\\arraystretch}{1.12}\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "state & action & log share & target share & in log & exact $Q^*$ & FQI & CQL \\\\\n"
        )
        f.write("\\midrule\n")
        for state in (LOW, HIGH):
            for action in (KEEP, REPLACE):
                in_log = "yes" if data["observed"][state, action] else "no"
                f.write(
                    f"{STATE_NAMES[state]} & {ACTION_NAMES[action]} & "
                    f"{100 * data['logging_sa_occupancy'][state, action]:.1f}\\% & "
                    f"{100 * data['target_sa_occupancy'][state, action]:.1f}\\% & "
                    f"{in_log} & {data['Q_star'][state, action]:.3f} & "
                    f"{fqi_Q[state, action]:.3f} & {cql_Q[state, action]:.3f} \\\\\n"
                )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table}\n")
    return path


def generate_figure(data):
    iterations = np.arange(1, N_ITERATIONS + 1)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(
        iterations,
        data["fqi_path"][:, HIGH, REPLACE],
        color=ALGO_COLORS["Q-Learning"],
        label="FQI",
    )
    ax.plot(
        iterations,
        data["cql_path"][:, HIGH, REPLACE],
        color=ALGO_COLORS["SAC"],
        label="CQL",
    )
    ax.axhline(
        data["Q_star"][HIGH, REPLACE],
        **BENCH_STYLE,
        label="exact $Q^*$",
    )
    ax.set_xlabel("fitted-Q iteration")
    ax.set_ylabel("$Q(\\mathrm{high},\\mathrm{replace})$")
    ax.set_title("The state-action pair absent from the finite log")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "engine_coverage.png")
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
