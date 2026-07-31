# DPO reward equivalence on the Engine Replacement MDP
# Chapter 9 - RLHF and AI Alignment
# Computes the KL-regularized optimal policy and inverts it to exhibit the
# state-specific reward shift that preference comparisons cannot identify.

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sims.engine import ACTION_NAMES, STATE_NAMES, build_mdp
from sims.plot_style import apply_style

apply_style()

import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)
BETA = 0.5
REFERENCE_POLICY = np.array([[0.60, 0.40], [0.40, 0.60]])


def compute_data(force=None):
    _, rewards = build_mdp()
    if np.any(REFERENCE_POLICY <= 0.0):
        raise ValueError("DPO inversion requires a full-support reference policy")

    exponentials = np.exp(rewards / BETA)
    unnormalized = REFERENCE_POLICY * exponentials
    normalizers = unnormalized.sum(axis=1)
    optimal_policy = unnormalized / normalizers[:, None]
    inverse_reward = BETA * np.log(optimal_policy / REFERENCE_POLICY)
    shifts = inverse_reward - rewards
    reconstructed_reward = inverse_reward + BETA * np.log(normalizers)[:, None]

    assert np.allclose(optimal_policy.sum(axis=1), 1.0)
    assert np.allclose(shifts[:, 0], shifts[:, 1])
    assert np.allclose(shifts, -BETA * np.log(normalizers)[:, None])
    assert np.allclose(reconstructed_reward, rewards)
    assert np.allclose(np.diff(inverse_reward, axis=1), np.diff(rewards, axis=1))

    print("DPO REWARD INVERSION ON THE ENGINE REPLACEMENT MDP")
    print(f"beta = {BETA:.3f}")
    print("reference policy rows [keep, replace]")
    for state, row in zip(STATE_NAMES, REFERENCE_POLICY):
        print(f"  {state:4s}  {row[0]:.6f}  {row[1]:.6f}")
    print()
    print(
        "state action    reward    pi_ref   exp(r/beta)   unnormalized"
        "    pi_star   inverse_r"
    )
    for s, state in enumerate(STATE_NAMES):
        for a, action in enumerate(ACTION_NAMES):
            print(
                f"{state:5s} {action:7s} {rewards[s, a]:9.6f}"
                f" {REFERENCE_POLICY[s, a]:9.6f} {exponentials[s, a]:13.6f}"
                f" {unnormalized[s, a]:14.6f} {optimal_policy[s, a]:10.6f}"
                f" {inverse_reward[s, a]:11.6f}"
            )
    print()
    print("state       Z(s)    beta log Z(s)   inverse shift")
    for s, state in enumerate(STATE_NAMES):
        print(
            f"{state:5s} {normalizers[s]:10.6f}"
            f" {BETA * np.log(normalizers[s]):16.6f} {shifts[s, 0]:15.6f}"
        )
    print()
    print(
        "maximum reconstruction error "
        f"{np.max(np.abs(reconstructed_reward - rewards)):.3e}"
    )
    print(
        "maximum within-state reward-difference error "
        f"{np.max(np.abs(np.diff(inverse_reward, axis=1) - np.diff(rewards, axis=1))):.3e}"
    )

    return {
        "rewards": rewards,
        "exponentials": exponentials,
        "unnormalized": unnormalized,
        "normalizers": normalizers,
        "optimal_policy": optimal_policy,
        "inverse_reward": inverse_reward,
        "shifts": shifts,
    }


def generate_outputs(data):
    path = os.path.join(OUTPUT_DIR, "engine_preferences.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{DPO reward inversion on the Engine Replacement MDP at "
            "$\\lambda_{KL}=0.5$. The four rows report the exponential tilt and "
            "the normalized policy. The final column is the reward representative "
            "selected by the DPO normalization.}\n"
        )
        f.write("\\label{tab:engine_preferences}\n")
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "state & action & $r$ & $\\pi^{SFT}$ & $e^{r/\\lambda_{KL}}$ "
            "& $\\pi^{SFT}e^{r/\\lambda_{KL}}$ & $\\pi^*$ "
            "& $\\lambda_{KL}\\log(\\pi^*/\\pi^{SFT})$ \\\\\n"
        )
        f.write("\\hline\n")
        for s, state in enumerate(STATE_NAMES):
            for a, action in enumerate(ACTION_NAMES):
                f.write(
                    f"{state} & {action} & {data['rewards'][s, a]:.4f} "
                    f"& {REFERENCE_POLICY[s, a]:.4f} "
                    f"& {data['exponentials'][s, a]:.4f} "
                    f"& {data['unnormalized'][s, a]:.4f} "
                    f"& {data['optimal_policy'][s, a]:.4f} "
                    f"& {data['inverse_reward'][s, a]:.4f} \\\\\n"
                )
        f.write("\\hline\n")
        f.write(
            "state & \\multicolumn{2}{l}{$Z(s)$} "
            "& \\multicolumn{2}{l}{$\\lambda_{KL}\\log Z(s)$} "
            "& \\multicolumn{3}{l}{inverse reward minus $r(s,a)$} \\\\\n"
        )
        f.write("\\hline\n")
        for s, state in enumerate(STATE_NAMES):
            f.write(
                f"{state} & \\multicolumn{{2}}{{l}}{{{data['normalizers'][s]:.4f}}} "
                f"& \\multicolumn{{2}}{{l}}{{"
                f"{BETA * np.log(data['normalizers'][s]):.4f}}} "
                f"& \\multicolumn{{3}}{{l}}{{{data['shifts'][s, 0]:.4f}}} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Table saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="DPO reward equivalence on the Engine Replacement MDP"
    )
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    data = compute_data()
    if not args.data_only:
        generate_outputs(data)


if __name__ == "__main__":
    main()
