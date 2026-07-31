# Bias from an unobserved grade in the Engine Replacement MDP
# Chapter 10 - Causal Inference for Reinforcement Learning
# Compares the observational transition plug-in with a backdoor adjustment on
# the +U engine and checks both against their population formulas.

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
    build_mdp_confounded,
    exact_value,
    policy_matrices,
)
from sims.plot_style import apply_style
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
SCRIPT_NAME = "engine_confounding"

CONFIG = {
    "delta": 0.20,
    "latent_probability": 0.50,
    "keep_probability_u0": 0.25,
    "keep_probability_u1": 0.75,
    "adjustment_weight": "estimated_low_state_latent_share",
    "n_seeds": 20,
    "n_trajectories": 2_000,
    "horizon": 40,
    "seed_start": 1700,
}
TARGET_POLICY = [KEEP, REPLACE]


def target_value_from_degradation(probability):
    P, r = build_mdp(degrade_prob=probability)
    P_pi, r_pi = policy_matrices(P, r, TARGET_POLICY)
    return exact_value(P_pi, r_pi, GAMMA)


def simulate_one_seed(seed, P_by_u, rewards):
    rng = np.random.default_rng(seed)
    q = CONFIG["latent_probability"]
    keep_probability = np.array(
        [CONFIG["keep_probability_u0"], CONFIG["keep_probability_u1"]]
    )
    total = 0
    high_next = 0
    total_by_u = np.zeros(2, dtype=int)
    high_by_u = np.zeros(2, dtype=int)
    low_state_by_u = np.zeros(2, dtype=int)

    for _ in range(CONFIG["n_trajectories"]):
        state = LOW
        for _ in range(CONFIG["horizon"]):
            latent = int(rng.random() < q)
            if state == LOW:
                low_state_by_u[latent] += 1
            action = KEEP if rng.random() < keep_probability[latent] else REPLACE
            next_state = int(rng.choice(2, p=P_by_u[latent, state, action]))
            if state == LOW and action == KEEP:
                total += 1
                high_next += int(next_state == HIGH)
                total_by_u[latent] += 1
                high_by_u[latent] += int(next_state == HIGH)
            state = next_state

    if total == 0 or np.any(total_by_u == 0):
        raise RuntimeError("insufficient low-mileage keep observations")

    naive_probability = high_next / total
    stratum_probabilities = high_by_u / total_by_u
    q_hat = low_state_by_u[1] / low_state_by_u.sum()
    adjusted_probability = float(
        (1.0 - q_hat) * stratum_probabilities[0] + q_hat * stratum_probabilities[1]
    )
    naive_value = target_value_from_degradation(naive_probability)
    adjusted_value = target_value_from_degradation(adjusted_probability)
    return {
        "seed": seed,
        "count": total,
        "count_u0": int(total_by_u[0]),
        "count_u1": int(total_by_u[1]),
        "q_hat": q_hat,
        "naive_probability": naive_probability,
        "adjusted_probability": adjusted_probability,
        "naive_value": naive_value,
        "adjusted_value": adjusted_value,
    }


def compute_data(force=None):
    if not force:
        cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        if cached is not None:
            print("Cache hit")
            return cached

    delta = CONFIG["delta"]
    q = CONFIG["latent_probability"]
    P_by_u, rewards, _ = build_mdp_confounded(delta=delta, q=q)
    degradation_by_u = P_by_u[:, LOW, KEEP, HIGH]
    keep_probability = np.array(
        [CONFIG["keep_probability_u0"], CONFIG["keep_probability_u1"]]
    )
    latent_weights = np.array([1.0 - q, q])

    interventional_probability = float(latent_weights @ degradation_by_u)
    observational_probability = float(
        (latent_weights * keep_probability)
        @ degradation_by_u
        / (latent_weights @ keep_probability)
    )
    true_value = target_value_from_degradation(interventional_probability)
    population_naive_value = target_value_from_degradation(observational_probability)
    population_bias = float(population_naive_value[LOW] - true_value[LOW])

    seed_results = [
        simulate_one_seed(CONFIG["seed_start"] + i, P_by_u, rewards)
        for i in range(CONFIG["n_seeds"])
    ]
    naive_values = np.array([row["naive_value"][LOW] for row in seed_results])
    adjusted_values = np.array([row["adjusted_value"][LOW] for row in seed_results])
    naive_probabilities = np.array([row["naive_probability"] for row in seed_results])
    adjusted_probabilities = np.array(
        [row["adjusted_probability"] for row in seed_results]
    )
    naive_biases = naive_values - true_value[LOW]
    adjusted_biases = adjusted_values - true_value[LOW]
    naive_bias_se = float(naive_biases.std(ddof=1) / np.sqrt(CONFIG["n_seeds"]))
    adjusted_bias_se = float(adjusted_biases.std(ddof=1) / np.sqrt(CONFIG["n_seeds"]))

    population_gap = abs(float(naive_biases.mean()) - population_bias)
    adjustment_gap = abs(float(adjusted_biases.mean()))
    assert population_gap <= 2.0 * naive_bias_se, (
        "simulated naive bias does not match the population formula within two MC SE"
    )
    assert adjustment_gap <= 2.0 * adjusted_bias_se, (
        "backdoor adjustment does not recover the target value within two MC SE"
    )

    data = {
        "degradation_by_u": degradation_by_u,
        "interventional_probability": interventional_probability,
        "observational_probability": observational_probability,
        "true_value": true_value,
        "population_naive_value": population_naive_value,
        "population_bias": population_bias,
        "seed_results": seed_results,
        "naive_probabilities": naive_probabilities,
        "adjusted_probabilities": adjusted_probabilities,
        "naive_values": naive_values,
        "adjusted_values": adjusted_values,
        "naive_biases": naive_biases,
        "adjusted_biases": adjusted_biases,
        "naive_bias_se": naive_bias_se,
        "adjusted_bias_se": adjusted_bias_se,
        "population_gap": population_gap,
        "adjustment_gap": adjustment_gap,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    return data


def print_results(data):
    print("BIAS FROM AN UNOBSERVED GRADE IN THE ENGINE REPLACEMENT MDP")
    print(
        f"delta {CONFIG['delta']:.3f}  P(U=1) {CONFIG['latent_probability']:.3f}  "
        f"trajectories {CONFIG['n_trajectories']}  horizon {CONFIG['horizon']}  "
        f"seeds {CONFIG['n_seeds']}"
    )
    print("latent grade       P(degrade | U)       behavior P(keep | U)")
    for u in (0, 1):
        print(
            f"{u:12d} {data['degradation_by_u'][u]:20.6f}"
            f" {CONFIG[f'keep_probability_u{u}']:24.6f}"
        )
    print()
    print(
        f"interventional P(degrade | do(keep)) {data['interventional_probability']:.6f}"
    )
    print(f"observational P(degrade | keep) {data['observational_probability']:.6f}")
    print(f"true target value at low mileage {data['true_value'][LOW]:.6f}")
    print(
        f"population naive value at low mileage "
        f"{data['population_naive_value'][LOW]:.6f}"
    )
    print(f"closed-form naive bias {data['population_bias']:.6f}")
    print()
    print(
        "seed   low-keep     U=0     U=1   q_hat   naive p  adjusted p"
        "   naive value  adjusted value"
    )
    for row in data["seed_results"]:
        print(
            f"{row['seed']:4d} {row['count']:10d} {row['count_u0']:7d}"
            f" {row['count_u1']:7d} {row['q_hat']:8.6f}"
            f" {row['naive_probability']:10.6f}"
            f" {row['adjusted_probability']:11.6f}"
            f" {row['naive_value'][LOW]:13.6f}"
            f" {row['adjusted_value'][LOW]:15.6f}"
        )
    print()
    print("estimator                  mean bias       MC SE")
    print(
        f"naive transition plug-in {data['naive_biases'].mean():12.6f}"
        f" {data['naive_bias_se']:12.6f}"
    )
    print(
        f"backdoor adjustment      {data['adjusted_biases'].mean():12.6f}"
        f" {data['adjusted_bias_se']:12.6f}"
    )
    print(
        f"naive formula gap {data['population_gap']:.6f}  "
        f"two MC SE {2 * data['naive_bias_se']:.6f}"
    )
    print(
        f"adjustment gap {data['adjustment_gap']:.6f}  "
        f"two MC SE {2 * data['adjusted_bias_se']:.6f}"
    )


def generate_outputs(data):
    path = os.path.join(OUTPUT_DIR, "engine_confounding.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Target-policy evaluation on the $+U$ Engine Replacement "
            "MDP. Monte Carlo entries are means and standard errors over twenty "
            "fixed seeds.}\n"
        )
        f.write("\\label{tab:engine_confounding}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write(
            "transition estimate & $P(\\mathrm{degrade})$ "
            "& $V(\\mathrm{low})$ & bias \\\\\n"
        )
        f.write("\\hline\n")
        f.write(
            f"interventional truth & {data['interventional_probability']:.4f} "
            f"& {data['true_value'][LOW]:.4f} & 0.0000 \\\\\n"
        )
        f.write(
            f"naive population formula & {data['observational_probability']:.4f} "
            f"& {data['population_naive_value'][LOW]:.4f} "
            f"& {data['population_bias']:.4f} \\\\\n"
        )
        f.write(
            f"naive simulation & {data['naive_probabilities'].mean():.4f} "
            f"& {data['naive_values'].mean():.4f} "
            f"& {data['naive_biases'].mean():.4f} "
            f"$\\,({data['naive_bias_se']:.4f})$ \\\\\n"
        )
        f.write(
            f"backdoor simulation & {data['adjusted_probabilities'].mean():.4f} "
            f"& {data['adjusted_values'].mean():.4f} "
            f"& {data['adjusted_biases'].mean():.4f} "
            f"$\\,({data['adjusted_bias_se']:.4f})$ \\\\\n"
        )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Table saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Confounding bias on the Engine Replacement MDP"
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
