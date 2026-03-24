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
    ('ch02', 'ch02_rl_algorithms/sims/algorithm_architectures.py', 'B'),

    # ch03_theory
    ('ch03_theory', 'ch03_theory/sims/brock_mirman_newton.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/lqr_convergence.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/lqc_fvi_fqi.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/trust_region_lqc.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/gridworld_study.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/theory_validation.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/ssp_gridworld_20x20.py', 'A'),
    ('ch03_theory', 'ch03_theory/sims/deadly_triad_geometry.py', 'B'),
    ('ch03_theory', 'ch03_theory/sims/qlearning_geometry.py', 'B'),
    ('ch03_theory', 'ch03_theory/sims/info_geometry_npg.py', 'B'),
    ('ch03_theory', 'ch03_theory/sims/mm_surrogate_trpo.py', 'B'),
    ('ch03_theory', 'ch03_theory/sims/td_lambda_corridor.py', 'A'),

    # ch03a
    ('ch03a', 'ch03a/sims/gridworld_illustrated.py', 'A'),

    # ch03a_bm
    ('ch03a_bm', 'ch03a_bm/sims/bm_illustrated.py', 'A'),
    ('ch03a_bm', 'ch03a_bm/sims/bm_fvi_fqi.py', 'A'),

    # ch03b
    ('ch03b', 'ch03b_deeprl_practice/sims/bellman_vs_return.py', 'A'),
    ('ch03b', 'ch03b_deeprl_practice/sims/brock_mirman_bellman.py', 'A'),
    ('ch03b', 'ch03b_deeprl_practice/sims/brock_mirman_dqn.py', 'A'),
    ('ch03b', 'ch03b_deeprl_practice/sims/overestimation_bias.py', 'B'),

    # ch04
    ('ch04', 'ch04_control_problems/sims/benchmark_bus_engine.py', 'A'),

    # ch05
    ('ch05', 'ch05_econ_models/sims/bus_engine_dp_vs_dqn.py', 'A'),
    ('ch05', 'ch05_econ_models/sims/nfxp_ccp_td.py', 'A'),
    ('ch05', 'ch05_econ_models/sims/estimation_flowcharts.py', 'B'),

    # ch06
    ('ch06', 'ch06_games/sims/durable_goods_monopoly.py', 'A'),
    ('ch06', 'ch06_games/sims/kuhn_poker_equilibrium.py', 'A'),
    ('ch06', 'ch06_games/sims/coase_stress_tests.py', 'A'),
    ('ch06', 'ch06_games/sims/cournot_bertrand_marl.py', 'A'),

    # ch07
    ('ch07', 'ch07_bandits/sims/bandit_fundamentals.py', 'A'),
    ('ch07', 'ch07_bandits/sims/strategic_pricing.py', 'A'),
    ('ch07', 'ch07_bandits/sims/auction_reserve_price.py', 'A'),
    ('ch07', 'ch07_bandits/sims/knowledge_ladder.py', 'A'),
    ('ch07', 'ch07_bandits/sims/structural_pricing_misra.py', 'A'),
    ('ch07', 'ch07_bandits/sims/regret_rates.py', 'B'),
    ('ch07', 'ch07_bandits/sims/uninformative_price.py', 'B'),

    # ch08 (Offline RL)
    ('ch08_offline', 'ch08_offline_rl/sims/offline_rl_pricing.py', 'A'),

    # ch09 (RLHF)
    ('ch09', 'ch09_rlhf/sims/job_search_rlhf.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/job_search_dpo.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/job_search_preference_learning.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/preference_learning.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/nfxp_vs_rlhf.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/gridworld_rlhf.py', 'A'),
    ('ch09', 'ch09_rlhf/sims/rlhf_dpo_pipeline.py', 'B'),

    # ch10
    ('ch10', 'ch10_causal/sims/confounded_ope.py', 'A'),
    ('ch10', 'ch10_causal/sims/identification_dags.py', 'B'),

    # ch11 (Quantile, Robust, Constrained)
    ('ch11', 'ch11_dist_robust_constrained/sims/risk_sensitive_inventory.py', 'A'),
    ('ch11', 'ch11_dist_robust_constrained/sims/robust_consumption_savings.py', 'A'),
    ('ch11', 'ch11_dist_robust_constrained/sims/carbon_constrained_production.py', 'A'),
]


def print_registry():
    print(f"{'Chapter':<12} {'Category':<5} {'Script'}")
    print('-' * 70)
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
    stdout_path = os.path.join(script_dir, f'{script_base}_stdout.txt')

    cmd = [sys.executable, abs_path] + flags
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            cwd=repo_root,
        )
        elapsed = time.perf_counter() - t0

        # Write stdout (combined stdout + stderr)
        with open(stdout_path, 'w') as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write('\n--- stderr ---\n')
                f.write(result.stderr)

        if result.returncode != 0:
            print(f"  FAIL ({elapsed:.1f}s): {script_path}")
            # Print last few lines of stderr for diagnostics
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"    {line}")
            return False, elapsed

        print(f"  OK   ({elapsed:.1f}s): {script_path}")
        return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"  TIMEOUT ({elapsed:.1f}s): {script_path}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='Run all simulation scripts')
    parser.add_argument('--list', action='store_true', help='Show script registry')
    parser.add_argument('--chapter', type=str, default=None,
                        help='Filter by chapter (partial match, e.g. "ch07")')
    parser.add_argument('--script', type=str, default=None,
                        help='Filter by script name (partial match, e.g. "bandit")')
    parser.add_argument('--data-only', action='store_true',
                        help='Pass --data-only to all scripts')
    parser.add_argument('--plots-only', action='store_true',
                        help='Pass --plots-only to all scripts')
    parser.add_argument('--algo', type=str, action='append', default=None,
                        help='Pass --algo to scripts (force-recompute component). '
                             'Repeat for multiple: --algo CQL --algo FQI')
    args = parser.parse_args()

    if args.list:
        print_registry()
        return

    # Build flag list to pass through
    flags = []
    if args.data_only:
        flags.append('--data-only')
    if args.plots_only:
        flags.append('--plots-only')
    if args.algo:
        for a in args.algo:
            flags.extend(['--algo', a])

    # Filter registry
    scripts = REGISTRY
    if args.chapter:
        scripts = [(ch, p, c) for ch, p, c in scripts if args.chapter in ch]
    if args.script:
        scripts = [(ch, p, c) for ch, p, c in scripts
                    if args.script in os.path.basename(p)]

    if not scripts:
        print("No scripts matched filters.")
        return

    print(f"Running {len(scripts)} scripts" +
          (f" with flags: {' '.join(flags)}" if flags else "") + "\n")

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

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {total_time:.1f}s total")


if __name__ == '__main__':
    main()
