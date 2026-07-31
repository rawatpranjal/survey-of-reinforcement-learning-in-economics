# Structural estimation on the +EV Engine Replacement MDP
# Chapter 5 - Structural Estimation with Reinforcement Learning
# Compares NFXP, Hotz-Miller CCP inversion, and a full-basis TD-CCP estimator
# at population moments and across fixed finite-sample replications.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    REPLACE,
    build_mdp,
    solve_ev,
    stationary_distribution,
)
from sims.plot_style import apply_style
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()

import numpy as np
from scipy.optimize import least_squares, minimize

np.random.seed(42)

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_estimation"

SHOCK_SCALE = 0.4
TRUE_THETA = np.array([0.2, 0.5])
PARAMETER_NAMES = ["high-mileage keep reward", "replacement cost"]
METHOD_NAMES = ["NFXP", "CCP", "TD-CCP"]
SEEDS = list(range(20))
N_AGENTS = 300
HORIZON = 30
CCP_PSEUDOCOUNT = 0.5

CONFIG = {
    "shock_scale": SHOCK_SCALE,
    "true_theta": TRUE_THETA.tolist(),
    "seeds": SEEDS,
    "n_agents": N_AGENTS,
    "horizon": HORIZON,
    "ccp_pseudocount": CCP_PSEUDOCOUNT,
    "version": 1,
}


def model_from_theta(theta):
    return build_mdp(
        r_keep_high=float(theta[0]),
        replace_cost=float(theta[1]),
    )


def empirical_ccp(choice_counts):
    numerator = choice_counts + CCP_PSEUDOCOUNT
    denominator = choice_counts.sum(axis=1, keepdims=True) + 2 * CCP_PSEUDOCOUNT
    return numerator / denominator


def choice_counts(states, actions):
    counts = np.zeros((2, 2))
    np.add.at(counts, (states, actions), 1.0)
    return counts


def log_odds(ccp):
    return np.log(ccp[:, KEEP] / ccp[:, REPLACE])


def fit_nfxp(counts):
    P, _ = model_from_theta(TRUE_THETA)

    def negative_log_likelihood(theta):
        _, r = model_from_theta(theta)
        _, _, ccp = solve_ev(P, r, GAMMA, SHOCK_SCALE)
        return -float(np.sum(counts * np.log(np.clip(ccp, 1e-15, 1.0))))

    result = minimize(
        negative_log_likelihood,
        x0=np.array([0.1, 0.4]),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"NFXP optimization failed: {result.message}")
    return result.x


def ccp_implied_log_odds(theta, ccp):
    P, r = model_from_theta(theta)
    P_policy = np.einsum("sa,sat->st", ccp, P)
    entropy_reward = -SHOCK_SCALE * np.sum(ccp * np.log(ccp), axis=1)
    policy_reward = np.einsum("sa,sa->s", ccp, r)
    W = np.linalg.solve(
        np.eye(2) - GAMMA * P_policy,
        policy_reward + entropy_reward,
    )
    continuation_gap = np.einsum("st,t->s", P[:, KEEP] - P[:, REPLACE], W)
    return (r[:, KEEP] - r[:, REPLACE] + GAMMA * continuation_gap) / SHOCK_SCALE


def fit_ccp(ccp):
    result = least_squares(
        lambda theta: ccp_implied_log_odds(theta, ccp) - log_odds(ccp),
        x0=np.array([0.1, 0.4]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=1000,
    )
    if not result.success:
        raise RuntimeError(f"CCP inversion failed: {result.message}")
    return result.x


def reward_basis(state, action, ccp):
    return np.array(
        [
            float(state == LOW and action == KEEP)
            - SHOCK_SCALE * np.log(ccp[state, action]),
            float(state == HIGH and action == KEEP),
            -float(action == REPLACE),
        ]
    )


def solve_td_components(A, b):
    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("TD normal matrix is singular")
    return np.array([np.linalg.solve(A, component) for component in b])


def theta_from_td_components(value_components, transition_kernel, ccp):
    intercept = np.zeros(2)
    design = np.zeros((2, 2))
    for state in range(2):
        transition_gap = (
            transition_kernel[state, KEEP] - transition_kernel[state, REPLACE]
        )
        intercept[state] = (
            float(state == LOW) + GAMMA * transition_gap @ value_components[0]
        ) / SHOCK_SCALE
        design[state, 0] = (
            float(state == HIGH) + GAMMA * transition_gap @ value_components[1]
        ) / SHOCK_SCALE
        design[state, 1] = (
            1.0 + GAMMA * transition_gap @ value_components[2]
        ) / SHOCK_SCALE
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("TD-CCP parameter system is singular")
    return np.linalg.solve(design, log_odds(ccp) - intercept)


def fit_td_ccp_population(P, ccp):
    P_policy = np.einsum("sa,sat->st", ccp, P)
    state_weights = stationary_distribution(P_policy)
    features = np.eye(2)
    A = np.zeros((2, 2))
    b = np.zeros((3, 2))

    for state in range(2):
        for action in range(2):
            for next_state in range(2):
                weight = (
                    state_weights[state]
                    * ccp[state, action]
                    * P[state, action, next_state]
                )
                A += weight * np.outer(
                    features[state],
                    features[state] - GAMMA * features[next_state],
                )
                b += weight * np.outer(
                    reward_basis(state, action, ccp),
                    features[state],
                )

    value_components = solve_td_components(A, b)
    return theta_from_td_components(value_components, P, ccp)


def fit_td_ccp_sample(states, actions, next_states, ccp):
    features = np.eye(2)
    A = np.zeros((2, 2))
    b = np.zeros((3, 2))
    transition_counts = np.zeros((2, 2, 2))
    state_action_counts = np.zeros((2, 2))

    for state, action, next_state in zip(states, actions, next_states):
        A += np.outer(
            features[state],
            features[state] - GAMMA * features[next_state],
        )
        b += np.outer(
            reward_basis(state, action, ccp),
            features[state],
        )
        transition_counts[state, action, next_state] += 1.0
        state_action_counts[state, action] += 1.0

    if np.any(state_action_counts == 0):
        raise ValueError("TD-CCP sample misses a state-action pair")
    A /= len(states)
    b /= len(states)
    transition_kernel = transition_counts / state_action_counts[:, :, None]
    value_components = solve_td_components(A, b)
    return theta_from_td_components(value_components, transition_kernel, ccp)


def simulate_panel(seed, P, ccp):
    rng = np.random.default_rng(seed)
    states = []
    actions = []
    next_states = []
    for _ in range(N_AGENTS):
        state = LOW
        for _ in range(HORIZON):
            action = rng.choice(2, p=ccp[state])
            next_state = rng.choice(2, p=P[state, action])
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
    return (
        np.asarray(states, dtype=int),
        np.asarray(actions, dtype=int),
        np.asarray(next_states, dtype=int),
    )


def compute_population_estimates(P, ccp):
    P_policy = np.einsum("sa,sat->st", ccp, P)
    state_weights = stationary_distribution(P_policy)
    population_counts = state_weights[:, None] * ccp
    return np.array(
        [
            fit_nfxp(population_counts),
            fit_ccp(ccp),
            fit_td_ccp_population(P, ccp),
        ]
    )


def compute_data(force=None):
    cached = None if force else load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print("Loaded from cache.")
        return cached

    P, r = model_from_theta(TRUE_THETA)
    W, choice_values, ccp = solve_ev(P, r, GAMMA, SHOCK_SCALE)
    population = compute_population_estimates(P, ccp)
    population_error = np.abs(population - TRUE_THETA)
    if np.max(population_error) >= 1e-6:
        raise AssertionError(
            f"population recovery error {np.max(population_error):.3e} exceeds 1e-6"
        )

    finite_estimates = np.zeros((len(SEEDS), len(METHOD_NAMES), 2))
    print("=" * 72)
    print("STRUCTURAL ESTIMATION ON THE +EV ENGINE REPLACEMENT MDP")
    print("=" * 72)
    print(f"discount factor: {GAMMA:.1f}")
    print(f"Type-I extreme-value shock scale: {SHOCK_SCALE:.1f}")
    print(
        "generating parameters: "
        f"high-mileage keep reward = {TRUE_THETA[0]:.4f}, "
        f"replacement cost = {TRUE_THETA[1]:.4f}"
    )
    print(f"population CCP at low mileage: {ccp[LOW].tolist()}")
    print(f"population CCP at high mileage: {ccp[HIGH].tolist()}")
    print(f"integrated values: {W.tolist()}")
    print(f"choice-specific values: {choice_values.tolist()}")
    print()
    print("Population recovery")
    print(f"{'method':<10} {'high keep':>12} {'replacement':>12} {'max abs error':>16}")
    for method_index, method in enumerate(METHOD_NAMES):
        print(
            f"{method:<10} {population[method_index, 0]:>12.8f} "
            f"{population[method_index, 1]:>12.8f} "
            f"{population_error[method_index].max():>16.3e}"
        )

    print()
    print(
        f"Finite samples: {len(SEEDS)} seeds, {N_AGENTS} agents, "
        f"{HORIZON} periods, {N_AGENTS * HORIZON} transitions per seed"
    )
    print(f"{'seed':>5} {'method':<10} {'high keep':>12} {'replacement':>12}")
    for seed_index, seed in enumerate(SEEDS):
        states, actions, next_states = simulate_panel(seed, P, ccp)
        counts = choice_counts(states, actions)
        ccp_hat = empirical_ccp(counts)
        estimates = np.array(
            [
                fit_nfxp(counts),
                fit_ccp(ccp_hat),
                fit_td_ccp_sample(states, actions, next_states, ccp_hat),
            ]
        )
        finite_estimates[seed_index] = estimates
        for method_index, method in enumerate(METHOD_NAMES):
            print(
                f"{seed:>5} {method:<10} "
                f"{estimates[method_index, 0]:>12.6f} "
                f"{estimates[method_index, 1]:>12.6f}"
            )

    finite_mean = finite_estimates.mean(axis=0)
    finite_se = finite_estimates.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    finite_rmse = np.sqrt(
        np.mean((finite_estimates - TRUE_THETA[None, None, :]) ** 2, axis=0)
    )

    print()
    print("Finite-sample summary")
    print(f"{'method':<10} {'parameter':<27} {'mean':>10} {'SE':>10} {'RMSE':>10}")
    for method_index, method in enumerate(METHOD_NAMES):
        for parameter_index, parameter in enumerate(PARAMETER_NAMES):
            print(
                f"{method:<10} {parameter:<27} "
                f"{finite_mean[method_index, parameter_index]:>10.6f} "
                f"{finite_se[method_index, parameter_index]:>10.6f} "
                f"{finite_rmse[method_index, parameter_index]:>10.6f}"
            )

    data = {
        "P": P,
        "ccp": ccp,
        "W": W,
        "choice_values": choice_values,
        "population": population,
        "population_error": population_error,
        "finite_estimates": finite_estimates,
        "finite_mean": finite_mean,
        "finite_se": finite_se,
        "finite_rmse": finite_rmse,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def generate_outputs(data):
    path = os.path.join(OUTPUT_DIR, "engine_estimation_results.tex")
    with open(path, "w") as handle:
        handle.write("\\begin{table}[h]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{Structural parameter recovery in the $+\\mathrm{EV}$ "
            "Engine Replacement MDP. Population columns use exact choice and "
            "transition probabilities. Finite-sample columns report means and "
            "standard errors across 20 fixed seeds, with 9,000 transitions per seed.}\n"
        )
        handle.write("\\label{tab:engine_estimation}\n")
        handle.write("\\small\n")
        handle.write("\\begin{tabular}{lrrrr}\n")
        handle.write("\\hline\n")
        handle.write(
            " & \\multicolumn{2}{c}{high-mileage keep reward} "
            "& \\multicolumn{2}{c}{replacement cost} \\\\\n"
        )
        handle.write(
            "method & population & finite sample & population & finite sample \\\\\n"
        )
        handle.write("\\hline\n")
        for method_index, method in enumerate(METHOD_NAMES):
            handle.write(
                f"{method} & {data['population'][method_index, 0]:.6f} "
                f"& {data['finite_mean'][method_index, 0]:.4f} "
                f"({data['finite_se'][method_index, 0]:.4f}) "
                f"& {data['population'][method_index, 1]:.6f} "
                f"& {data['finite_mean'][method_index, 1]:.4f} "
                f"({data['finite_se'][method_index, 1]:.4f}) \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write(
            "generating value & \\multicolumn{2}{c}{0.200000} "
            "& \\multicolumn{2}{c}{0.500000} \\\\\n"
        )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
    print()
    print("Output files")
    print("  ch05_econ_models/sims/engine_estimation_results.tex")


def main():
    parser = argparse.ArgumentParser(
        description="NFXP, CCP, and TD-CCP on the +EV Engine Replacement MDP"
    )
    add_cache_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute finite-sample estimates even when a valid cache exists",
    )
    args = parser.parse_args()

    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, "No cache found. Run without --plots-only first."
    else:
        data = compute_data(force=args.force)
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
