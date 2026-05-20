# Counterfactual off-policy evaluation under a misspecified linear SCM.
# Chapter 10 (causal inference for reinforcement learning), §cfope_sim.
# Compares three estimators of V(pi_tilde) from logged data under pi_obs:
#   IS  Per-decision importance sampling
#   MB  Model-based (fit hat_f, plug pi_tilde actions)
#   CF  Residual-corrected estimator labelled "CF" in the tex.
#       This is algebraically the classical doubly-robust / AIPW estimator
#       (Robins-Rotnitzky-Zhao 1994; Bang-Robins 2005), not the Buesing
#       (2019) abduction-action-prediction estimator, which operates at the
#       trajectory level under a known/learned SCM and is not implemented
#       here. The naive Buesing-style average of
#         y'_i = hat_f(x_i, pi_tilde(x_i)) + (y_i - hat_f(x_i, a_i))
#       collapses to V_MB under OLS with intercept because residuals sum to
#       zero. Weighting the residual by the importance ratio rho_i restores
#       bias cancellation, yielding the AIPW form below. Unbiased under
#       correct propensity OR correct outcome model.
# Two scenarios: well-specified outcome model and misspecified outcome model.
# The propensity is held fixed at the true DGP value in both scenarios, so
# only the outcome-model side of double robustness is stressed here.

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_DOUBLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'counterfactual_ope'
OUT_DIR = os.path.dirname(__file__)

# ----------------------------------------------------------------------------
# Configuration

# True data-generating process parameters
#   y = beta0 + bx1*x1 + bx2*x2 + ba*a + bxa1*(a*x1) + u, u ~ N(0, sigma_u^2)
DGP_PARAMS = dict(
    beta0=1.0,
    bx1=0.5,
    bx2=-0.3,
    ba=2.0,
    bxa1=1.0,
    sigma_u=0.5,
    # Behavior policy logit: alpha0 + a1*x1 + a2*x2
    alpha0=0.5,
    a1=1.0,
    a2=-0.5,
    # Target policy threshold: pi_tilde(x) = 1{x1 + x2 > 0}
)

SAMPLE_SIZES = [200, 500, 1000, 2000]
N_SEEDS = 20

SHARED_CONFIG = {**DGP_PARAMS, 'sample_sizes': SAMPLE_SIZES,
                  'n_seeds': N_SEEDS, 'oracle_mc': 1_000_000}

# Per-scenario configs share the same logged data; only the estimator's
# feature set changes. Two scenarios:
#   well_spec  model includes a*x1 interaction (matches DGP)
#   misspec    model omits a*x1 (the true heterogeneous-effect term)
WELL_SPEC_CONFIG = {**SHARED_CONFIG, 'scenario': 'well_spec',
                     'model_features': ['intercept', 'x1', 'x2', 'a', 'a_x1']}
MISSPEC_CONFIG   = {**SHARED_CONFIG, 'scenario': 'misspec',
                     'model_features': ['intercept', 'x1', 'x2', 'a']}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def target_policy(x):
    """Deterministic target policy: take action 1 iff x1 + x2 > 0."""
    return (x[..., 0] + x[..., 1] > 0).astype(float)


def behavior_prob_a1(x, cfg):
    """Behavior policy probability of a=1 given features x."""
    z = cfg['alpha0'] + cfg['a1'] * x[..., 0] + cfg['a2'] * x[..., 1]
    return sigmoid(z)


def outcome_mean(x, a, cfg):
    """True conditional mean E[y | x, a] under the DGP."""
    return (cfg['beta0'] + cfg['bx1'] * x[..., 0] + cfg['bx2'] * x[..., 1]
            + cfg['ba'] * a + cfg['bxa1'] * a * x[..., 0])


def generate_data(seed, n, cfg):
    """Generate one logged dataset under the behavior policy."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, 2))
    p_a1 = behavior_prob_a1(x, cfg)
    u = rng.uniform(size=n)
    a = (u < p_a1).astype(float)
    eps = rng.normal(0.0, cfg['sigma_u'], size=n)
    y = outcome_mean(x, a, cfg) + eps
    return {'x': x, 'a': a, 'y': y, 'p_a1': p_a1}


def compute_oracle(cfg):
    """Monte-Carlo estimate of V(pi_tilde) under the true DGP."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(cfg['oracle_mc'], 2))
    a_tilde = target_policy(x)
    return float(np.mean(outcome_mean(x, a_tilde, cfg)))


# ----------------------------------------------------------------------------
# Estimators

def design_matrix(x, a, features):
    """Build the design matrix for a chosen feature set."""
    cols = []
    for f in features:
        if f == 'intercept':
            cols.append(np.ones(len(x)))
        elif f == 'x1':
            cols.append(x[..., 0])
        elif f == 'x2':
            cols.append(x[..., 1])
        elif f == 'a':
            cols.append(a)
        elif f == 'a_x1':
            cols.append(a * x[..., 0])
        else:
            raise ValueError(f'unknown feature {f}')
    return np.stack(cols, axis=-1)


def fit_ols(x, a, y, features):
    """OLS fit returning coefficient vector."""
    X = design_matrix(x, a, features)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def predict(theta, x, a, features):
    """Predict y under fitted model at given (x, a)."""
    X = design_matrix(x, a, features)
    return X @ theta


def estimator_IS(data, cfg):
    """Per-decision importance sampling with deterministic target.

    rho_i = pi_tilde(a_i | x_i) / pi_obs(a_i | x_i)
    With deterministic pi_tilde, rho_i = 1{a_i == pi_tilde(x_i)} /
    pi_obs(a_i | x_i).
    """
    x, a, y, p_a1 = data['x'], data['a'], data['y'], data['p_a1']
    a_tilde = target_policy(x)
    p_obs = np.where(a > 0.5, p_a1, 1 - p_a1)
    matched = (a == a_tilde).astype(float)
    rho = matched / np.maximum(p_obs, 1e-6)
    return float(np.mean(rho * y))


def estimator_MB(data, features):
    """Model-based estimator with the given feature set."""
    x, a, y = data['x'], data['a'], data['y']
    theta = fit_ols(x, a, y, features)
    a_tilde = target_policy(x)
    return float(np.mean(predict(theta, x, a_tilde, features)))


def estimator_CF(data, features):
    """Counterfactual-augmented (doubly-robust) estimator.

    Under correct SCM with known noise, Buesing's abduction-action-prediction
    rule sets y_cf_i = hat_f(x_i, a'_i) + (y_i - hat_f(x_i, a_i)). Averaging
    over i with an OLS hat_f returns V_MB exactly because OLS residuals sum
    to zero. The finite-sample fix is to weight the residual correction by
    the importance ratio rho_i, recovering the doubly-robust estimator
    V_CF = mean(hat_f(x_i, a'_i)) + mean(rho_i * (y_i - hat_f(x_i, a_i))),
    which is unbiased under correct propensity OR correct model.
    """
    x, a, y, p_a1 = data['x'], data['a'], data['y'], data['p_a1']
    theta = fit_ols(x, a, y, features)
    a_tilde = target_policy(x)
    # Predictions under observed and counterfactual actions
    y_hat_obs = predict(theta, x, a, features)
    y_hat_cf = predict(theta, x, a_tilde, features)
    # Importance ratio with deterministic target
    p_obs = np.where(a > 0.5, p_a1, 1 - p_a1)
    matched = (a == a_tilde).astype(float)
    rho = matched / np.maximum(p_obs, 1e-6)
    # Doubly-robust combination
    return float(np.mean(y_hat_cf) + np.mean(rho * (y - y_hat_obs)))


# ----------------------------------------------------------------------------
# Per-scenario evaluation

def run_scenario(scenario_cfg, shared):
    """Run IS, MB, CF for all (n, seed) cells in one scenario."""
    features = scenario_cfg['model_features']
    sample_sizes = shared['sample_sizes']
    n_seeds = shared['n_seeds']
    oracle = shared['oracle']

    results = {
        'IS': np.zeros((len(sample_sizes), n_seeds)),
        'MB': np.zeros((len(sample_sizes), n_seeds)),
        'CF': np.zeros((len(sample_sizes), n_seeds)),
    }
    for i, n in enumerate(sample_sizes):
        for s in range(n_seeds):
            seed = 1000 * (i + 1) + s
            data = generate_data(seed, n, scenario_cfg)
            results['IS'][i, s] = estimator_IS(data, scenario_cfg)
            results['MB'][i, s] = estimator_MB(data, features)
            results['CF'][i, s] = estimator_CF(data, features)
    # Convert to bias/std/RMSE arrays
    summary = {}
    for est, vals in results.items():
        bias = vals.mean(axis=1) - oracle
        std = vals.std(axis=1, ddof=1)
        rmse = np.sqrt(((vals - oracle) ** 2).mean(axis=1))
        summary[est] = {'estimates': vals, 'bias': bias, 'std': std,
                         'rmse': rmse}
    return summary


# ----------------------------------------------------------------------------
# Compute pipeline

def compute_shared(cfg):
    return {'oracle': compute_oracle(cfg),
            'sample_sizes': cfg['sample_sizes'],
            'n_seeds': cfg['n_seeds']}


def compute_well_spec(shared, cfg):
    return run_scenario(cfg, shared)


def compute_misspec(shared, cfg):
    return run_scenario(cfg, shared)


SCENARIO_REGISTRY = {
    'well_spec': (compute_well_spec, WELL_SPEC_CONFIG),
    'misspec':   (compute_misspec,   MISSPEC_CONFIG),
}


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'shared', SHARED_CONFIG,
                              compute_shared, SHARED_CONFIG,
                              force=('shared' in force))
    results = {}
    for name, (fn, ccfg) in SCENARIO_REGISTRY.items():
        results[name] = compute_or_load(
            CACHE_DIR, SCRIPT_NAME, name, ccfg,
            fn, shared, ccfg,
            force=(name in force or 'shared' in force),
        )
    return {'shared': shared, 'results': results}


# ----------------------------------------------------------------------------
# Outputs

EST_COLORS = {
    'IS': COLORS['gray'],
    'MB': COLORS['red'],
    'CF': COLORS['blue'],
}

EST_LABELS = {
    'IS': 'IS (importance sampling)',
    'MB': 'MB (model-based)',
    'CF': 'CF (counterfactual)',
}


def generate_outputs(data):
    shared = data['shared']
    results = data['results']
    sample_sizes = shared['sample_sizes']
    oracle = shared['oracle']

    # Figure: two panels, log-log RMSE vs n
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)
    for ax, scenario in zip(axes, ['well_spec', 'misspec']):
        for est in ('IS', 'MB', 'CF'):
            rmse = results[scenario][est]['rmse']
            ax.loglog(sample_sizes, rmse, 'o-',
                      label=EST_LABELS[est] if scenario == 'well_spec' else None,
                      color=EST_COLORS[est], linewidth=1.4, markersize=5)
        title = 'Well-specified model' if scenario == 'well_spec' else 'Misspecified model'
        ax.set_xlabel('sample size $n$')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('RMSE of $\\widehat V$')
    axes[0].legend(loc='best', frameon=False)
    fig_path = os.path.join(OUT_DIR, 'counterfactual_ope.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure: {fig_path}')

    # Table: bias/std/RMSE at n=1000 for both scenarios
    n_target = 1000
    n_idx = sample_sizes.index(n_target)
    rows = []
    for scenario in ('well_spec', 'misspec'):
        for est in ('IS', 'MB', 'CF'):
            r = results[scenario][est]
            rows.append((scenario, est, r['bias'][n_idx], r['std'][n_idx],
                          r['rmse'][n_idx]))

    tex_path = os.path.join(OUT_DIR, 'counterfactual_ope_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Generated by counterfactual_ope.py\n')
        f.write('\\begin{tabular}{llccc}\n')
        f.write('\\hline\\hline\n')
        f.write('Scenario & Estimator & Bias & Std & RMSE \\\\\n')
        f.write('\\hline\n')
        prev_scenario = None
        for scenario, est, bias, std, rmse in rows:
            scen_label = ''
            if scenario != prev_scenario:
                scen_label = 'Well-specified' if scenario == 'well_spec' else 'Misspecified'
                prev_scenario = scenario
            f.write(f'{scen_label} & {est} & ${bias:+.3f}$ & '
                    f'${std:.3f}$ & ${rmse:.3f}$ \\\\\n')
            if est == 'CF' and scenario == 'well_spec':
                f.write('\\hline\n')
        f.write('\\hline\\hline\n')
        f.write('\\end{tabular}\n')
    print(f'  Table:  {tex_path}')

    # Stdout summary
    print()
    print(f'Oracle V(pi_tilde) = {oracle:.4f}')
    print()
    print(f'Summary at n = {n_target} over {shared["n_seeds"]} seeds:')
    print(f'{"Scenario":<16} {"Est":<6} {"Bias":>10} {"Std":>10} {"RMSE":>10}')
    for scenario, est, bias, std, rmse in rows:
        scen = 'Well-spec' if scenario == 'well_spec' else 'Misspec'
        print(f'{scen:<16} {est:<6} {bias:>+10.4f} {std:>10.4f} {rmse:>10.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_component_args(parser)
    args = parser.parse_args()
    force = parse_force_set(args)
    print('Config:')
    print(f'  sample_sizes={SAMPLE_SIZES}, n_seeds={N_SEEDS}')
    print(f'  sigma_u={DGP_PARAMS["sigma_u"]}, '
          f'true TE: tau(x) = {DGP_PARAMS["ba"]} + '
          f'{DGP_PARAMS["bxa1"]}*x1')
    if force:
        print(f'  forcing recompute of: {sorted(force)}')
    if args.plots_only:
        data = compute_data()
    else:
        data = compute_data(force=force)
    if not args.data_only:
        generate_outputs(data)


if __name__ == '__main__':
    main()
