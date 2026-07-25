#!/usr/bin/env python3
"""Run all simulation scripts with optional filtering and flag passthrough.

Usage:
    python3 scripts/run_all_sims.py                    # full run (compute + output)
    python3 scripts/run_all_sims.py --plots-only       # refresh all figures/tables from cache
    python3 scripts/run_all_sims.py --data-only        # compute only, skip output generation
    python3 scripts/run_all_sims.py --chapter ch07     # one chapter (partial match)
    python3 scripts/run_all_sims.py --script bandit    # one script (partial name match)
    python3 scripts/run_all_sims.py --list             # show registry
    python3 scripts/run_all_sims.py --script offline_rl --algo CQL  # recompute one component
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Registry: (chapter, script_path_relative_to_repo_root, category)
#   A = compute-heavy (supports --data-only / --plots-only with caching)
#   B = diagram-only  (--data-only is no-op, --plots-only runs normally)
# ---------------------------------------------------------------------------

REGISTRY = [
    # ch02
    ("ch02", "ch02_rl_algorithms/sims/algorithm_architectures.py", "B"),
    # ch03_theory
    ("ch03_theory", "ch03_theory/sims/brock_mirman_newton.py", "A"),
    ("ch03_theory", "ch03_theory/sims/lqc_fvi_fqi.py", "A"),
    ("ch03_theory", "ch03_theory/sims/trust_region_lqc.py", "A"),
    ("ch03_theory", "ch03_theory/sims/gridworld_study.py", "A"),
    ("ch03_theory", "ch03_theory/sims/deadly_triad_geometry.py", "B"),
    ("ch03_theory", "ch03_theory/sims/qlearning_geometry.py", "B"),
    ("ch03_theory", "ch03_theory/sims/info_geometry_npg.py", "B"),
    ("ch03_theory", "ch03_theory/sims/mm_surrogate_trpo.py", "B"),
    ("ch03_theory", "ch03_theory/sims/td_lambda_corridor.py", "A"),
    ("ch03_theory", "ch03_theory/sims/wind_farm_curse_study.py", "A"),
    ("ch03_theory", "ch03_theory/sims/curse_arithmetic.py", "B"),
    # ch03a_bm
    ("ch03a_bm", "ch03a_bm/sims/bm_fvi_fqi.py", "A"),
    # ch03b
    ("ch03b", "ch03b_deeprl_practice/sims/bellman_vs_return.py", "A"),
    ("ch03b", "ch03b_deeprl_practice/sims/brock_mirman_bellman.py", "A"),
    ("ch03b", "ch03b_deeprl_practice/sims/brock_mirman_dqn.py", "A"),
    ("ch03b", "ch03b_deeprl_practice/sims/overestimation_bias.py", "B"),
    # ch04
    ("ch04", "ch04_control_problems/sims/benchmark_bus_engine.py", "A"),
    # ch05
    ("ch05", "ch05_econ_models/sims/bus_engine_dp_vs_dqn.py", "A"),
    ("ch05", "ch05_econ_models/sims/nfxp_ccp_td.py", "A"),
    ("ch05", "ch05_econ_models/sims/estimation_flowcharts.py", "B"),
    # ch06_macro
    ("ch06_macro", "ch06_macro/sims/rbc_dp_vs_drl.py", "A"),
    ("ch06_macro", "ch06_macro/sims/lq_mfg.py", "B"),
    ("ch06_macro", "ch06_macro/sims/mfax_lq_run_grid.py", "A"),
    # ch06
    ("ch06", "ch06_games/sims/durable_goods_monopoly.py", "A"),
    ("ch06", "ch06_games/sims/kuhn_poker_equilibrium.py", "A"),
    ("ch06", "ch06_games/sims/coase_stress_tests.py", "A"),
    ("ch06", "ch06_games/sims/cournot_bertrand_marl.py", "A"),
    # ch07
    ("ch07", "ch07_bandits/sims/knowledge_ladder.py", "A"),
    ("ch07", "ch07_bandits/sims/curve_learning_pricing.py", "A"),
    ("ch07", "ch07_bandits/sims/regret_rates.py", "B"),
    ("ch07", "ch07_bandits/sims/uninformative_price.py", "B"),
    # ch08 (Offline RL)
    ("ch08_offline", "ch08_offline_rl/sims/offline_rl_pricing.py", "A"),
    # ch09 (RLHF)
    ("ch09", "ch09_rlhf/sims/job_search_preference_learning.py", "A"),
    ("ch09", "ch09_rlhf/sims/axiom_aware_aggregation.py", "A"),
    ("ch09", "ch09_rlhf/sims/rlhf_dpo_pipeline.py", "B"),
    # ch10
    ("ch10", "ch10_causal/sims/confounded_ope.py", "A"),
    ("ch10", "ch10_causal/sims/identification_dags.py", "B"),
    # ch10b (OPE and dynamic treatment effects)
    ("ch10b", "ch10b_rl_for_ci/sims/dtr_qlearning_vs_murphy.py", "A"),
    ("ch10b", "ch10b_rl_for_ci/sims/dtr_dags.py", "B"),
    ("ch10b", "ch10b_rl_for_ci/sims/ope_estimators.py", "A"),
    ("ch10b", "ch10b_rl_for_ci/sims/dynamic_dml_snmm.py", "A"),
    ("ch10b", "ch10b_rl_for_ci/sims/dtr_policy_learning.py", "A"),
    # ch10c (causal bandits and adaptive experimentation)
    ("ch10c", "ch10c_adaptive_experiments/sims/causal_bandit_parallel.py", "A"),
    # ch11 (Quantile, Robust, Constrained)
    ("ch11", "ch11_dist_robust_constrained/sims/risk_sensitive_inventory.py", "A"),
    ("ch11", "ch11_dist_robust_constrained/sims/robust_consumption_savings.py", "A"),
    ("ch11", "ch11_dist_robust_constrained/sims/carbon_constrained_production.py", "A"),
    # ch12 (World Models and Model-Based RL)
    ("ch12", "ch12_world_models/sims/dyna_maze.py", "A"),
    ("ch12", "ch12_world_models/sims/cobweb_paradigms.py", "A"),
    ("ch12", "ch12_world_models/sims/fishery_paradigms.py", "A"),
    ("ch12", "ch12_world_models/sims/multi_echelon_paradigms.py", "A"),
    ("ch12", "ch12_world_models/sims/draw_maze_layout.py", "B"),
    # appA (Mathematical Preliminaries).
    # The running example is data-only and every other appA figure that quotes a number
    # from it reads that number here, so it runs first.
    ("appA", "appA_preliminaries/sims/running_example.py", "A"),
    ("appA", "appA_preliminaries/sims/elementary_concepts.py", "B"),
    ("appA", "appA_preliminaries/sims/appendix_geometry.py", "B"),
    ("appA", "appA_preliminaries/sims/discount_cost.py", "A"),
    ("appA", "appA_preliminaries/sims/projection_geometry.py", "A"),
    # Retired from the appendix text but kept runnable: each still verifies a result the
    # appendix now states without proof, and the scripts are the evidence trail for those
    # statements even though their figures no longer appear.
    ("appA", "appA_preliminaries/sims/rl_theory_geometry.py", "B"),
    ("appA", "appA_preliminaries/sims/spectral_radius.py", "A"),
    ("appA", "appA_preliminaries/sims/neumann_series.py", "A"),
    ("appA", "appA_preliminaries/sims/markov_stationary.py", "A"),
    ("appA", "appA_preliminaries/sims/hilbert_projection.py", "B"),
    ("appA", "appA_preliminaries/sims/jensen_gap.py", "A"),
    ("appA", "appA_preliminaries/sims/lln_clt.py", "A"),
    ("appA", "appA_preliminaries/sims/martingale_convergence.py", "A"),
    ("appA", "appA_preliminaries/sims/gradient_descent.py", "A"),
    ("appA", "appA_preliminaries/sims/lagrangian_duality.py", "A"),
    ("appA", "appA_preliminaries/sims/envelope_theorem.py", "A"),
    ("appA", "appA_preliminaries/sims/lipschitz_continuity.py", "A"),
    ("appA", "appA_preliminaries/sims/banach_contraction.py", "A"),
    ("appA", "appA_preliminaries/sims/robbins_monro.py", "A"),
    # ch13 OPE-reliability sim is NOT listed here: it needs its own pinned venv
    # (scope-rl / d3rlpy / torch) that the default sys.executable runner does not have.
    # Run it directly:
    #   cd ch13_field_deployments/sims && ./.venv/bin/python field_ope_reliability.py \
    #       > field_ope_reliability_stdout.txt 2>&1
]


def print_registry():
    print(f"{'Chapter':<12} {'Category':<5} {'Script'}")
    print("-" * 70)
    for ch, path, cat in REGISTRY:
        print(f"{ch:<12} {cat:<5} {path}")
    print(f"\nTotal: {len(REGISTRY)} scripts")


def run_script(script_path, flags, repo_root):
    """Run a single script, capture stdout to _stdout.txt, return (success, elapsed)."""
    abs_path = os.path.join(repo_root, script_path)
    if not os.path.exists(abs_path):
        print(f"  SKIP (not found): {script_path}")
        return False, 0.0

    # Build stdout capture path: same dir as script, script_name_stdout.txt
    script_dir = os.path.dirname(abs_path)
    script_base = os.path.splitext(os.path.basename(abs_path))[0]
    stdout_path = os.path.join(script_dir, f"{script_base}_stdout.txt")

    cmd = [sys.executable, abs_path] + flags
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=repo_root,
        )
        elapsed = time.perf_counter() - t0

        # Write stdout (combined stdout + stderr)
        with open(stdout_path, "w") as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write("\n--- stderr ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            print(f"  FAIL ({elapsed:.1f}s): {script_path}")
            # Print last few lines of stderr for diagnostics
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
            return False, elapsed

        print(f"  OK   ({elapsed:.1f}s): {script_path}")
        return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  TIMEOUT ({elapsed:.1f}s): {script_path}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all simulation scripts")
    parser.add_argument("--list", action="store_true", help="Show script registry")
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        help='Filter by chapter (partial match, e.g. "ch07")',
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help='Filter by script name (partial match, e.g. "bandit")',
    )
    parser.add_argument(
        "--data-only", action="store_true", help="Pass --data-only to all scripts"
    )
    parser.add_argument(
        "--plots-only", action="store_true", help="Pass --plots-only to all scripts"
    )
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        default=None,
        help="Pass --algo to scripts (force-recompute component). "
        "Repeat for multiple: --algo CQL --algo FQI",
    )
    args = parser.parse_args()

    if args.list:
        print_registry()
        return

    # Build flag list to pass through
    flags = []
    if args.data_only:
        flags.append("--data-only")
    if args.plots_only:
        flags.append("--plots-only")
    if args.algo:
        for a in args.algo:
            flags.extend(["--algo", a])

    # Filter registry
    scripts = REGISTRY
    if args.chapter:
        scripts = [(ch, p, c) for ch, p, c in scripts if args.chapter in ch]
    if args.script:
        scripts = [
            (ch, p, c) for ch, p, c in scripts if args.script in os.path.basename(p)
        ]

    if not scripts:
        print("No scripts matched filters.")
        return

    print(
        f"Running {len(scripts)} scripts"
        + (f" with flags: {' '.join(flags)}" if flags else "")
        + "\n"
    )

    passed = 0
    failed = 0
    total_time = 0.0

    for ch, path, cat in scripts:
        ok, elapsed = run_script(path, flags, REPO_ROOT)
        if ok:
            passed += 1
        else:
            failed += 1
        total_time += elapsed

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {total_time:.1f}s total")


if __name__ == "__main__":
    main()
