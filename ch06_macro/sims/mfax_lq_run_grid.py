"""Run the official MFAX Linear-Quadratic SPG/RSPG grid.

This is a thin orchestration wrapper around the public MFAX scripts.  It does
not reimplement the algorithms; it launches:

  * mfax/algos/hsm/algos/spg.py
  * mfax/algos/hsm/algos/rspg.py

Set ``MFAX_ROOT`` to a local MFAX checkout and ``MFAX_PYTHON`` to the Python
interpreter for that checkout.  The resulting JSON is consumed by
``lq_mfg.py`` to build the chapter table and figure.

The run used for the checked-in artifact was from MFAX commit ``9acc1eb``.
Under Python 3.11, the temporary MFAX checkout also needed dataclass
``default_factory`` compatibility fixes in unused environments, plus a small
stdout patch so the scripts print ``Return`` in each evaluation line.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / 'mfax_lq_grid_results.json'
DEFAULT_OUTPUT_STDOUT = SCRIPT_DIR / 'mfax_lq_grid_stdout.txt'

ALGOS = (
    ('SPG', 'mfax/algos/hsm/algos/spg.py', 'hsm_spg'),
    ('RSPG', 'mfax/algos/hsm/algos/rspg.py', 'hsm_rspg'),
)

LINE_RE = re.compile(
    r'Iteration:\s*(?P<iter>\d+),\s*Train Time:\s*(?P<time>[-+0-9.eE]+),\s*'
    r'Exploitability:\s*(?P<exploit>[-+0-9.eE]+),\s*Return:\s*(?P<ret>[-+0-9.eE]+)'
)


def parse_csv_numbers(raw, cast):
    return [cast(part.strip()) for part in raw.split(',') if part.strip()]


def run_one(job, mfax_root, mfax_python, base_args):
    algo_name, script, algo_flag, lr, seed = job
    cmd = [
        str(mfax_python), script, *base_args,
        '--algo', algo_flag,
        '--seed', str(seed),
        '--lr', str(lr),
    ]
    env = os.environ.copy()
    env.setdefault('WANDB_MODE', 'disabled')

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=mfax_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    elapsed = time.perf_counter() - t0
    curves = [
        {
            'iteration': int(match['iter']),
            'train_time': float(match['time']),
            'exploitability': float(match['exploit']),
            'return': float(match['ret']),
        }
        for match in (m.groupdict() for m in LINE_RE.finditer(proc.stdout))
    ]
    result = {
        'algo': algo_name,
        'script': script,
        'mfax_algo': algo_flag,
        'lr': lr,
        'seed': seed,
        'returncode': proc.returncode,
        'wall_clock_seconds': elapsed,
        'curves': curves,
        'stdout': proc.stdout,
    }
    if proc.returncode == 0 and curves and curves[-1]['iteration'] == 200:
        result['final'] = curves[-1]
    else:
        result['error'] = 'missing final metric or nonzero return code'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mfax-root', default=os.environ.get('MFAX_ROOT', '/tmp/mfax'))
    parser.add_argument(
        '--mfax-python',
        default=os.environ.get('MFAX_PYTHON', '/tmp/mfax-venv/bin/python'),
    )
    parser.add_argument('--output-json', default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument('--output-stdout', default=str(DEFAULT_OUTPUT_STDOUT))
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--lrs', default='0.0001,0.001,0.01')
    parser.add_argument('--max-parallel', type=int, default=4)
    args = parser.parse_args()

    mfax_root = Path(args.mfax_root)
    mfax_python = Path(args.mfax_python)
    if not mfax_root.exists():
        raise FileNotFoundError(f'MFAX_ROOT does not exist: {mfax_root}')
    if not mfax_python.exists():
        raise FileNotFoundError(f'MFAX_PYTHON does not exist: {mfax_python}')

    seeds = parse_csv_numbers(args.seeds, int)
    lrs = parse_csv_numbers(args.lrs, float)
    base_args = [
        '--task', 'linear_quadratic',
        '--state-type', 'indices',
        '--discount-factor', '0.99',
        '--normalize-obs',
        '--normalize-states',
        '--common-noise',
        '--num-envs', '8',
        '--num-iterations', '200',
        '--anneal-lr',
        '--max-grad-norm', '1.0',
        '--eval-frequency', '20',
        '--no-log',
        '--no-save',
    ]
    jobs = [
        (algo_name, script, algo_flag, lr, seed)
        for algo_name, script, algo_flag in ALGOS
        for lr in lrs
        for seed in seeds
    ]

    print(f'running {len(jobs)} official MFAX HSM jobs with max_parallel={args.max_parallel}')
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(run_one, job, mfax_root, mfax_python, base_args): job
            for job in jobs
        }
        for future in cf.as_completed(futures):
            res = future.result()
            results.append(res)
            if 'final' in res:
                final = res['final']
                print(
                    f"done {res['algo']} seed={res['seed']} lr={res['lr']}: "
                    f"expl={final['exploitability']:.3f}, "
                    f"ret={final['return']:.3f}, "
                    f"train={final['train_time']:.2f}s, "
                    f"wall={res['wall_clock_seconds']:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"FAILED {res['algo']} seed={res['seed']} lr={res['lr']}: "
                    f"{res.get('error')}",
                    flush=True,
                )

    commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=mfax_root,
        text=True,
    ).strip()
    payload = {
        'source': {
            'repo': 'https://github.com/CWibault/mfax.git',
            'local_root': str(mfax_root),
            'commit': commit,
            'notes': [
                'Official HSM configs: linear_quadratic_spg.yaml and linear_quadratic_rspg.yaml.',
                'MFAX scripts must print Return in the Iteration lines for this parser.',
                'Python 3.11 may require default_factory compatibility fixes in MFAX dataclasses.',
            ],
        },
        'config': {
            'task': 'linear_quadratic',
            'state_type': 'indices',
            'discount_factor': 0.99,
            'normalize_obs': True,
            'normalize_states_requested': True,
            'common_noise': True,
            'num_envs': 8,
            'num_iterations': 200,
            'eval_frequency': 20,
            'lrs': lrs,
            'seeds': seeds,
        },
        'results': sorted(
            results,
            key=lambda row: (row.get('algo', ''), row.get('seed', -1), row.get('lr', -1)),
        ),
    }

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    stdout_lines = [
        'Official MFAX Linear Quadratic HSM grid',
        f'source_commit={commit}',
        f'jobs={len(jobs)}',
        '',
    ]
    for res in payload['results']:
        stdout_lines.append('=' * 72)
        stdout_lines.append(f"{res.get('algo')} seed={res.get('seed')} lr={res.get('lr')}")
        stdout_lines.append(res.get('stdout', res.get('error', '')))
    Path(args.output_stdout).write_text('\n'.join(stdout_lines), encoding='utf-8')

    errors = [res for res in results if 'final' not in res]
    print(f'wrote {output_json}')
    print(f'errors={len(errors)}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
