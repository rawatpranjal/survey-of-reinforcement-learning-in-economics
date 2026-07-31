# Search as a partial improvement step: a one-step evaluate-improve proxy
# Chapter 3 - The Theory of Reinforcement Learning
# Closed-form calculator on the Engine Replacement MDP. It uses the uniform policy's
# exact value as an approximate evaluation, applies one greedy lookahead, and checks
# the policy loss against the greedy error amplification bound of Theorem
# thm:singh_yee (Singh and Yee, 1994). This is a proxy for the evaluate-improve pattern,
# not a tree-search or AlphaZero simulation. No Monte Carlo, no cache.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.plot_style import apply_style
from sims.engine import (
    GAMMA,
    HIGH,
    KEEP,
    LOW,
    build_mdp,
    exact_value,
    policy_matrices,
    policy_performance,
    q_values,
    solve_optimal,
)

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)

NU = np.array(
    [1.0, 0.0]
)  # start at low mileage, the appendix and policy-square convention
UNIFORM_POLICY = np.array([[0.5, 0.5], [0.5, 0.5]])


def compute_data(force=None):
    P, r = build_mdp()
    V_star, greedy_star, Q_star = solve_optimal(P, r, GAMMA)

    # approximate evaluation: the uniform policy's exact value
    J0, V0, rho0 = policy_performance(P, r, GAMMA, UNIFORM_POLICY, NU)
    print(f"V* = ({V_star[LOW]:.4f}, {V_star[HIGH]:.4f})")
    print(f"network value v = V^(uniform) = ({V0[LOW]:.4f}, {V0[HIGH]:.4f})")

    # the "search" step: one-step greedy lookahead from v, exactly Table tab:pi_alphazero's
    # evaluate-then-improve pair, and exactly the alpha -> infinity limit already plotted
    # as the orange dots in Figure fig:engine_policy_square
    Q0 = q_values(P, r, V0, GAMMA)
    pi_hat = list(Q0.argmax(axis=1))
    names = ["keep", "replace"]
    print(f"search step: greedy(v) = ({names[pi_hat[LOW]]}, {names[pi_hat[HIGH]]})")
    assert pi_hat == [KEEP, KEEP], "search step did not land on the (keep, keep) vertex"

    P_hat, r_hat = policy_matrices(P, r, pi_hat)
    V_hat = exact_value(P_hat, r_hat, GAMMA)
    print(
        f"actual value of the searched policy V^(pi_hat) = ({V_hat[LOW]:.4f}, {V_hat[HIGH]:.4f})"
    )

    eps = float(np.max(np.abs(V0 - V_star)))
    actual_loss = float(np.max(np.abs(V_star - V_hat)))
    sy_const = 2 * GAMMA / (1 - GAMMA)
    sy_bound = sy_const * eps
    ratio = actual_loss / sy_bound
    holds = actual_loss <= sy_bound + 1e-9

    print()
    print(f"epsilon = ||v - V*||_inf = {eps:.4f}")
    print(f"actual loss ||V* - V^(pi_hat)||_inf = {actual_loss:.4f}")
    print(f"Singh-Yee constant 2*gamma/(1-gamma) at gamma={GAMMA} = {sy_const:.4f}")
    print(f"Singh-Yee bound (Theorem thm:singh_yee) = {sy_bound:.4f}")
    print(f"actual loss / bound = {ratio:.4f}")
    print(f"bound holds: {holds}")

    assert holds, "the searched policy violates the Singh-Yee bound"

    return {
        "V_star": V_star,
        "V0": V0,
        "pi_hat": pi_hat,
        "V_hat": V_hat,
        "eps": eps,
        "actual_loss": actual_loss,
        "sy_const": sy_const,
        "sy_bound": sy_bound,
        "ratio": ratio,
    }


def generate_outputs(data):
    tex_path = os.path.join(OUTPUT_DIR, "engine_hybrid_search.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{A one-step evaluate-improve proxy on the Engine Replacement MDP. "
            "The approximate value $v$ is the uniform policy's exact value; one greedy "
            "lookahead lands on the $(\\text{keep},\\text{keep})$ vertex of "
            "Figure~\\ref{fig:engine_policy_square}. The bound is Theorem~"
            "\\ref{thm:singh_yee} at $\\gamma = 0.9$.}\n"
        )
        f.write("\\label{tab:engine_hybrid_search}\n")
        f.write("\\begin{tabular}{lrr}\n\\hline\n")
        f.write(" & low & high \\\\\n\\hline\n")
        v0 = data["V0"]
        vh = data["V_hat"]
        vs = data["V_star"]
        f.write(
            f"network value $v = V^{{(\\mathrm{{unif}})}}$ & {v0[LOW]:.4f} & {v0[HIGH]:.4f} \\\\\n"
        )
        f.write(
            f"searched policy value $V^{{\\hat\\pi}}$ & {vh[LOW]:.4f} & {vh[HIGH]:.4f} \\\\\n"
        )
        f.write(f"optimal value $V^\\star$ & {vs[LOW]:.4f} & {vs[HIGH]:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write(" & \\multicolumn{2}{c}{value} \\\\\n\\hline\n")
        f.write(
            f"$\\varepsilon = \\|v - V^\\star\\|_\\infty$ & \\multicolumn{{2}}{{c}}{{{data['eps']:.4f}}} \\\\\n"
        )
        f.write(
            f"actual loss $\\|V^\\star - V^{{\\hat\\pi}}\\|_\\infty$ "
            f"& \\multicolumn{{2}}{{c}}{{{data['actual_loss']:.4f}}} \\\\\n"
        )
        f.write(
            f"Singh--Yee constant $2\\gamma/(1-\\gamma)$ "
            f"& \\multicolumn{{2}}{{c}}{{{data['sy_const']:.4f}}} \\\\\n"
        )
        f.write(
            f"Singh--Yee bound $2\\gamma\\varepsilon/(1-\\gamma)$ "
            f"& \\multicolumn{{2}}{{c}}{{{data['sy_bound']:.4f}}} \\\\\n"
        )
        f.write(
            f"actual loss / bound & \\multicolumn{{2}}{{c}}{{{data['ratio']:.4f}}} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  Table saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="One-step evaluate-improve proxy on the Engine Replacement MDP"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    print("=" * 70)
    print("SEARCH AS A PARTIAL IMPROVEMENT STEP ON THE ENGINE REPLACEMENT MDP")
    print("=" * 70)
    print()
    data = compute_data()
    if not args.data_only:
        generate_outputs(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
