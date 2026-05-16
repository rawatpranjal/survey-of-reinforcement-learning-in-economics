"""Summarise the official MFAX Linear-Quadratic RSPG grid.

The heavy lifting for this artifact is done by the public MFAX implementation
of Wibault et al. (RSPG), using the paper's Linear-Quadratic POMFG configs:

  * ``configs/linear_quadratic_spg.yaml``
  * ``configs/linear_quadratic_rspg.yaml``

The raw grid is stored in ``mfax_lq_grid_results.json``.  This script keeps the
chapter artifact reproducible without requiring the chapter build to rerun the
external JAX training jobs every time.  It regenerates the table and figure used
in the text from that official-output JSON.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE

apply_style()

SCRIPT_NAME = 'lq_mfg'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRID_JSON = os.path.join(SCRIPT_DIR, 'mfax_lq_grid_results.json')

METHOD_ORDER = ('SPG', 'RSPG')
METHOD_COLORS = {
    'SPG': COLORS['orange'],
    'RSPG': COLORS['blue'],
}


def _sem(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _format_lr(lr):
    exponent = round(-math.log10(lr))
    if math.isclose(lr, 10 ** (-exponent), rel_tol=0.0, abs_tol=1e-12):
        return f'1e-{exponent}'
    return f'{lr:.2g}'


def _format_lr_tex(lr):
    exponent = round(-math.log10(lr))
    if math.isclose(lr, 10 ** (-exponent), rel_tol=0.0, abs_tol=1e-12):
        return f'10^{{-{exponent}}}'
    return f'{lr:.2g}'


def load_grid(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing official MFAX grid JSON: {path}. "
            "Regenerate it from the MFAX scripts before rebuilding this artifact."
        )
    with open(path, 'r') as f:
        data = json.load(f)

    bad = [
        res for res in data.get('results', [])
        if res.get('returncode') != 0 or 'final' not in res
    ]
    if bad:
        raise RuntimeError(f"MFAX grid contains {len(bad)} failed jobs")
    return data


def group_results(data):
    grouped = defaultdict(list)
    for res in data['results']:
        grouped[(res['algo'], float(res['lr']))].append(res)
    return dict(grouped)


def aggregate_runs(runs):
    finals = [run['final'] for run in runs]
    exploit = np.array([row['exploitability'] for row in finals], dtype=np.float64)
    ret = np.array([row['return'] for row in finals], dtype=np.float64)
    train_time = np.array([row['train_time'] for row in finals], dtype=np.float64)
    wall_time = np.array([run['wall_clock_seconds'] for run in runs], dtype=np.float64)

    curve_iters = np.array([row['iteration'] for row in runs[0]['curves']], dtype=np.int64)
    curve_exploit = np.array([
        [row['exploitability'] for row in run['curves']]
        for run in runs
    ], dtype=np.float64)

    return {
        'n': len(runs),
        'iterations': curve_iters,
        'curve_exploitability_mean': curve_exploit.mean(axis=0),
        'curve_exploitability_sem': curve_exploit.std(axis=0, ddof=1) / math.sqrt(len(runs)),
        'exploitability_mean': float(exploit.mean()),
        'exploitability_sem': _sem(exploit),
        'return_mean': float(ret.mean()),
        'return_sem': _sem(ret),
        'train_time_mean': float(train_time.mean()),
        'train_time_sem': _sem(train_time),
        'wall_time_mean': float(wall_time.mean()),
        'wall_time_sem': _sem(wall_time),
    }


def aggregate_grid(data):
    grouped = group_results(data)
    by_algo = defaultdict(dict)
    for (algo, lr), runs in grouped.items():
        by_algo[algo][lr] = aggregate_runs(runs)
    return dict(by_algo)


def select_learning_rates(aggregates):
    selected = {}
    for algo, by_lr in aggregates.items():
        selected[algo] = min(
            by_lr,
            key=lambda lr: by_lr[lr]['exploitability_mean'],
        )
    return selected


def write_table(aggregates, selected):
    rows = []
    for algo in METHOD_ORDER:
        lr = selected[algo]
        res = aggregates[algo][lr]
        rows.append(
            f"{algo} & ${_format_lr_tex(lr)}$ & "
            f"${res['exploitability_mean']:.2f} \\pm {res['exploitability_sem']:.2f}$ & "
            f"${res['return_mean']:.1f} \\pm {res['return_sem']:.1f}$ & "
            f"${res['train_time_mean']:.2f} \\pm {res['train_time_sem']:.2f}$ \\\\"
        )
    table = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Method & LR & Exploitability & Expected return & Train time (s) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    path = os.path.join(SCRIPT_DIR, f'{SCRIPT_NAME}_results.tex')
    with open(path, 'w') as f:
        f.write(table)
    print(f"  Table saved: {path}")


def write_figure(aggregates, selected):
    fig, (ax_curve, ax_grid) = plt.subplots(
        1, 2, figsize=FIG_DOUBLE, gridspec_kw={'width_ratios': [1.4, 1.0]}
    )

    for algo in METHOD_ORDER:
        lr = selected[algo]
        res = aggregates[algo][lr]
        steps = res['iterations']
        mean = res['curve_exploitability_mean']
        sem = res['curve_exploitability_sem']
        ax_curve.plot(
            steps, mean, color=METHOD_COLORS[algo], linewidth=1.7,
            label=f'{algo}, LR={_format_lr(lr)}'
        )
        ax_curve.fill_between(
            steps, mean - sem, mean + sem,
            color=METHOD_COLORS[algo], alpha=0.16, linewidth=0
        )
    ax_curve.set_xlabel('Policy-gradient update')
    ax_curve.set_ylabel('Approximate exploitability')
    ax_curve.set_yscale('log')
    ax_curve.legend(loc='upper right', fontsize=8)

    offsets = {'SPG': -0.012, 'RSPG': 0.012}
    for algo in METHOD_ORDER:
        lrs = sorted(aggregates[algo])
        means = [aggregates[algo][lr]['exploitability_mean'] for lr in lrs]
        sems = [aggregates[algo][lr]['exploitability_sem'] for lr in lrs]
        x = np.arange(len(lrs), dtype=np.float64) + offsets[algo]
        ax_grid.errorbar(
            x, means, yerr=sems, marker='o', linestyle='-',
            capsize=3, color=METHOD_COLORS[algo], label=algo
        )
    ax_grid.set_xticks(np.arange(len(sorted(next(iter(aggregates.values()))))))
    ax_grid.set_xticklabels([_format_lr(lr) for lr in sorted(next(iter(aggregates.values())))])
    ax_grid.set_xlabel('Learning rate')
    ax_grid.set_ylabel('Final exploitability')
    ax_grid.set_yscale('log')
    ax_grid.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f'{SCRIPT_NAME}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")


def print_summary(data, aggregates, selected):
    source = data.get('source', {})
    config = data.get('config', {})
    print(f"Official MFAX LQ-POMFG artifact: {SCRIPT_NAME}")
    print("=" * 72)
    print("Source:")
    print(f"  repo       = {source.get('repo', 'unknown')}")
    print(f"  commit     = {source.get('commit', 'unknown')}")
    print("  scripts    = hsm/algos/spg.py, hsm/algos/rspg.py")
    print()
    print("Environment and training:")
    print("  task       = linear_quadratic")
    print("  state      = indices, public MFAX num_states=99")
    print(f"  gamma      = {config.get('discount_factor')}")
    print(f"  num_envs   = {config.get('num_envs')}")
    print(f"  iterations = {config.get('num_iterations')}")
    print(f"  eval_every = {config.get('eval_frequency')}")
    print(f"  seeds      = {config.get('seeds')}")
    print(f"  LR grid    = {config.get('lrs')}")
    print()
    print(f"{'Method':8s} {'LR':>8s} {'Exploitability':>22s} "
          f"{'Return':>18s} {'Train time':>16s}")
    for algo in METHOD_ORDER:
        lr = selected[algo]
        res = aggregates[algo][lr]
        print(
            f"{algo:8s} {_format_lr(lr):>8s}"
            f" {res['exploitability_mean']:>9.2f} +/- {res['exploitability_sem']:<7.2f}"
            f" {res['return_mean']:>9.1f} +/- {res['return_sem']:<6.1f}"
            f" {res['train_time_mean']:>8.2f} +/- {res['train_time_sem']:<5.2f}"
        )
    print()
    print("Final exploitability by LR:")
    for algo in METHOD_ORDER:
        parts = []
        for lr in sorted(aggregates[algo]):
            res = aggregates[algo][lr]
            parts.append(f"{_format_lr(lr)}={res['exploitability_mean']:.2f}")
        print(f"  {algo}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--grid-json', default=DEFAULT_GRID_JSON,
        help='Official MFAX grid JSON to summarise',
    )
    args = parser.parse_args()

    data = load_grid(args.grid_json)
    aggregates = aggregate_grid(data)
    selected = select_learning_rates(aggregates)
    print_summary(data, aggregates, selected)
    write_table(aggregates, selected)
    write_figure(aggregates, selected)


if __name__ == '__main__':
    main()
