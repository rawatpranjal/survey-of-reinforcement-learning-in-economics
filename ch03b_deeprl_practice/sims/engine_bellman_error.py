# Bellman error and Jensen bias on the Engine Replacement MDP
# Chapter 3b - The Empirics of Deep Reinforcement Learning
# Constructs an incomplete-data zero-residual solution and computes the
# Gaussian maximum bias against the Engine Replacement MDP's true action gap.

import argparse
import math
import os
import sys

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
    exact_value,
    policy_matrices,
    q_values,
)
from sims.plot_style import apply_style

apply_style()

import numpy as np
from scipy.special import ndtr

OUTPUT_DIR = os.path.dirname(__file__)

TARGET_POLICY = np.array([KEEP, REPLACE])
MISSING_PAIR = (HIGH, REPLACE)
VALUE_SHIFT = 10.0
NOISE_SCALES = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])


def pair_index(state, action):
    return 2 * state + action


def construct_zero_residual_solution(P, r, Q_true):
    observed_pairs = [
        (LOW, KEEP),
        (LOW, REPLACE),
        (HIGH, KEEP),
    ]
    A = np.zeros((4, 4))
    b = np.zeros(4)

    for row, (state, action) in enumerate(observed_pairs):
        A[row, pair_index(state, action)] = 1.0
        for next_state in range(2):
            next_action = TARGET_POLICY[next_state]
            A[row, pair_index(next_state, next_action)] -= (
                GAMMA * P[state, action, next_state]
            )
        b[row] = r[state, action]

    missing_index = pair_index(*MISSING_PAIR)
    A[-1, missing_index] = 1.0
    b[-1] = Q_true[MISSING_PAIR] + VALUE_SHIFT
    Q_hat = np.linalg.solve(A, b).reshape(2, 2)

    continuation = np.array([Q_hat[state, TARGET_POLICY[state]] for state in range(2)])
    residual = Q_hat - (r + GAMMA * P @ continuation)
    return observed_pairs, Q_hat, residual


def gaussian_max_bias(action_gap, sigma):
    standardized_gap = action_gap / (math.sqrt(2.0) * sigma)
    density = math.exp(-(standardized_gap**2) / 2.0) / math.sqrt(2.0 * math.pi)
    return math.sqrt(2.0) * sigma * density - action_gap * ndtr(-standardized_gap)


def compute_data(force=None):
    P, r = build_mdp()
    P_pi, r_pi = policy_matrices(P, r, TARGET_POLICY)
    V_pi = exact_value(P_pi, r_pi, GAMMA)
    Q_true = q_values(P, r, V_pi, GAMMA)

    observed_pairs, Q_hat, residual = construct_zero_residual_solution(P, r, Q_true)
    observed_mask = np.ones((2, 2), dtype=bool)
    observed_mask[MISSING_PAIR] = False
    value_error = Q_hat - Q_true

    assert P[LOW, KEEP, HIGH] > 0.0
    assert TARGET_POLICY[HIGH] == REPLACE
    assert np.max(np.abs(residual[observed_mask])) < 1e-12
    assert np.max(np.abs(value_error[observed_mask])) > 5.0
    assert abs(Q_true[HIGH, REPLACE] - V_pi[HIGH]) < 1e-12

    action_gap = float(Q_true[HIGH, REPLACE] - Q_true[HIGH, KEEP])
    biases = np.array([gaussian_max_bias(action_gap, sigma) for sigma in NOISE_SCALES])
    assert abs(action_gap - 67.0 / 290.0) < 1e-10
    assert abs(biases[0] - 0.000010183805602143568) < 1e-12
    assert abs(biases[-1] - 0.1815023688874949) < 1e-12

    print("=" * 74)
    print("BELLMAN ERROR AND JENSEN BIAS ON THE ENGINE REPLACEMENT MDP")
    print("=" * 74)
    print(f"discount factor: {GAMMA:.1f}")
    print("target policy: low -> keep, high -> replace")
    print("dataset coverage: 3 of 4 state-action pairs")
    print("missing pair: high, replace")
    print(f"imposed value shift at missing pair: {VALUE_SHIFT:.1f}")
    print()
    print("Incomplete-data construction")
    print(
        f"{'state':<8} {'action':<9} {'in data':<8} "
        f"{'Q^pi':>10} {'Q_hat':>10} {'residual':>12} {'abs error':>12}"
    )
    for state in range(2):
        for action in range(2):
            in_data = bool(observed_mask[state, action])
            print(
                f"{STATE_NAMES[state]:<8} {ACTION_NAMES[action]:<9} "
                f"{str(in_data):<8} {Q_true[state, action]:>10.4f} "
                f"{Q_hat[state, action]:>10.4f} {residual[state, action]:>12.4e} "
                f"{abs(value_error[state, action]):>12.4f}"
            )
    print()
    print(
        "maximum absolute residual on observed pairs: "
        f"{np.max(np.abs(residual[observed_mask])):.4e}"
    )
    print(
        "maximum absolute value error on observed pairs: "
        f"{np.max(np.abs(value_error[observed_mask])):.4f}"
    )
    print()
    print(f"true high-state action gap: {action_gap:.10f}")
    print("Gaussian maximum bias")
    print(f"{'sigma':>10} {'bias':>14} {'bias / gap':>14}")
    for sigma, bias in zip(NOISE_SCALES, biases):
        print(f"{sigma:>10.2f} {bias:>14.6f} {bias / action_gap:>14.4f}")

    return {
        "Q_true": Q_true,
        "Q_hat": Q_hat,
        "residual": residual,
        "value_error": value_error,
        "observed_mask": observed_mask,
        "observed_pairs": observed_pairs,
        "action_gap": action_gap,
        "noise_scales": NOISE_SCALES,
        "biases": biases,
    }


def generate_outputs(data):
    path = os.path.join(OUTPUT_DIR, "engine_bellman_error_results.tex")
    with open(path, "w") as handle:
        handle.write("\\begin{table}[H]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{Bellman residuals and Gaussian maximum bias in the "
            "Engine Replacement MDP. The target policy keeps at low mileage "
            "and replaces at high mileage. The data omit the high-mileage "
            "replacement pair.}\n"
        )
        handle.write("\\label{tab:engine_bellman_error}\n")
        handle.write("\\small\n")
        handle.write("\\begin{tabular}{llcrrr}\n")
        handle.write("\\hline\n")
        handle.write(
            "state & action & in data & $Q^\\pi$ & $\\widehat Q$ "
            "& $|\\widehat Q-Q^\\pi|$ \\\\\n"
        )
        handle.write("\\hline\n")
        for state in range(2):
            for action in range(2):
                in_data = "yes" if data["observed_mask"][state, action] else "no"
                handle.write(
                    f"{STATE_NAMES[state]} & {ACTION_NAMES[action]} & {in_data} "
                    f"& {data['Q_true'][state, action]:.4f} "
                    f"& {data['Q_hat'][state, action]:.4f} "
                    f"& {abs(data['value_error'][state, action]):.4f} \\\\\n"
                )
        handle.write("\\hline\n")
        handle.write(
            "\\multicolumn{3}{l}{maximum observed-pair residual} "
            f"& \\multicolumn{{3}}{{c}}{{"
            f"{np.max(np.abs(data['residual'][data['observed_mask']])):.1e}}} \\\\\n"
        )
        handle.write("\\hline\n")
        handle.write(
            "$\\sigma$ & \\multicolumn{2}{c}{Gaussian maximum bias} "
            "& \\multicolumn{3}{c}{bias divided by the $0.2310$ gap} \\\\\n"
        )
        handle.write("\\hline\n")
        for sigma, bias in zip(data["noise_scales"], data["biases"]):
            handle.write(
                f"{sigma:.2f} & \\multicolumn{{2}}{{c}}{{{bias:.6f}}} "
                f"& \\multicolumn{{3}}{{c}}{{{bias / data['action_gap']:.4f}}} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
    print()
    print("Output files")
    print("  ch03b_deeprl_practice/sims/engine_bellman_error_results.tex")


def main():
    parser = argparse.ArgumentParser(
        description="Bellman error and Jensen bias on the Engine Replacement MDP"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    data = compute_data()
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
