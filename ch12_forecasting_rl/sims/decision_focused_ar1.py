# Decision-focused forecasting for the contextual newsvendor under AR(1) demand.
# Chapter 12 (forecasting and reinforcement learning), main simulation.
# Compares three procedures sharing the same parametric family:
#   A. OLS / MLE plus critical-fractile plug-in (classical predict-then-optimize)
#   B. Task-loss subgradient descent on the realized newsvendor loss
#      (decision-focused learning, Donti-Amos-Kolter 2017)
#   C. Bertsekas-style one-step rollout using empirical residual quantiles

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_SINGLE, BENCH_STYLE
from sims.sim_cache import compute_or_load, add_component_args, parse_force_set

apply_style()

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
SCRIPT_NAME = 'decision_focused_ar1'
OUT_DIR = os.path.dirname(__file__)

# ----------------------------------------------------------------------------
# Configuration

ENV_PARAMS = dict(
    phi=0.7,           # AR(1) persistence
    mu=10.0,           # demand mean
    sigma0=1.0,        # baseline noise scale
    beta_dim=10,       # covariate dimension
    beta_scale=0.5,    # covariate effect scale
    gamma_scale=0.4,   # heteroscedasticity coefficient (misspec only)
    misspec=True,      # whether DGP is heteroscedastic (model is homoscedastic either way)
    T_train=2000,
    T_test=500,
    c_u=4.0,
    c_o=1.0,
)

SHARED_CONFIG = {**ENV_PARAMS, 'n_seeds': 20}
TAU_STAR = ENV_PARAMS['c_u'] / (ENV_PARAMS['c_u'] + ENV_PARAMS['c_o'])

# Method-level configs (one per algorithm)
A_CONFIG = {**SHARED_CONFIG, 'method': 'OLS'}
B_CONFIG = {**SHARED_CONFIG, 'method': 'SPO+', 'lr': 0.005, 'epochs': 400, 'init_from_ols': True}
C_CONFIG = {**SHARED_CONFIG, 'method': 'rollout', 'residual_quantile': True}

# ----------------------------------------------------------------------------
# Data generating process

def generate_data(seed, cfg):
    """Return dict with train and test arrays for one seed."""
    rng = np.random.default_rng(seed)
    p = cfg['beta_dim']
    beta_true = rng.normal(0.0, cfg['beta_scale'], size=p)
    gamma_true = rng.normal(0.0, cfg['gamma_scale'], size=p)

    def simulate(T):
        x = rng.normal(0.0, 1.0, size=(T, p))
        d = np.zeros(T)
        d[0] = cfg['mu']
        for t in range(1, T):
            sig = cfg['sigma0']
            if cfg['misspec']:
                # heteroscedastic noise depending on covariates; OLS sigma_hat is wrong
                sig = sig * np.exp(gamma_true @ x[t])
            mean = (cfg['mu'] + cfg['phi'] * (d[t - 1] - cfg['mu'])
                    + beta_true @ x[t])
            d[t] = mean + sig * rng.normal()
        return x, d

    x_tr, d_tr = simulate(cfg['T_train'])
    x_te, d_te = simulate(cfg['T_test'])
    return {
        'x_train': x_tr, 'd_train': d_tr,
        'x_test': x_te, 'd_test': d_te,
        'beta_true': beta_true, 'gamma_true': gamma_true,
    }


# ----------------------------------------------------------------------------
# Shared setup: data + oracle regret

def compute_shared(cfg):
    """Generate seed-indexed datasets and compute oracle costs."""
    seeds = list(range(cfg['n_seeds']))
    datasets = []
    oracle_costs = np.zeros((cfg['n_seeds'], cfg['T_test']))
    for i, s in enumerate(seeds):
        ds = generate_data(s, cfg)
        datasets.append(ds)
        oracle_costs[i] = oracle_cost_path(ds, cfg)
    return {'datasets': datasets, 'oracle_costs': oracle_costs, 'seeds': seeds}


def oracle_cost_path(ds, cfg):
    """Optimal expected newsvendor cost path on test set (known DGP)."""
    p = cfg['beta_dim']
    beta_true = ds['beta_true']
    gamma_true = ds['gamma_true']
    x_te = ds['x_test']
    d_te = ds['d_test']
    T_te = len(d_te)
    costs = np.zeros(T_te)
    for t in range(1, T_te):
        mean = (cfg['mu'] + cfg['phi'] * (d_te[t - 1] - cfg['mu'])
                + beta_true @ x_te[t])
        if cfg['misspec']:
            sig = cfg['sigma0'] * np.exp(gamma_true @ x_te[t])
        else:
            sig = cfg['sigma0']
        q_star = mean + sig * norm.ppf(TAU_STAR)
        costs[t] = newsvendor_cost(q_star, d_te[t], cfg['c_u'], cfg['c_o'])
    return costs


def newsvendor_cost(q, d, c_u, c_o):
    """Pointwise newsvendor cost."""
    return c_u * np.maximum(d - q, 0.0) + c_o * np.maximum(q - d, 0.0)


# ----------------------------------------------------------------------------
# Method A: OLS / MLE plus critical-fractile plug-in

def build_design(x, d_lag, mu_centered=True):
    """Return feature matrix [1, d_{t-1}-mu_centered_term, x_t]."""
    n = x.shape[0]
    if mu_centered:
        # We fit intercept + AR(1) + beta'x as linear regression on (1, d_{t-1}, x_t)
        ones = np.ones((n, 1))
        return np.hstack([ones, d_lag.reshape(-1, 1), x])
    else:
        return np.hstack([d_lag.reshape(-1, 1), x])


def fit_ols_one(ds, cfg):
    """Fit linear regression d_t = a + phi*d_{t-1} + beta'x_t + noise on training set."""
    x_tr = ds['x_train']
    d_tr = ds['d_train']
    T = len(d_tr)
    X = build_design(x_tr[1:], d_tr[:-1])
    y = d_tr[1:]
    # Closed-form OLS
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    sigma_hat = float(np.std(resid, ddof=X.shape[1]))
    return {
        'a': float(theta[0]),
        'phi': float(theta[1]),
        'beta': theta[2:].copy(),
        'sigma': sigma_hat,
        'residuals': resid,
    }


def predict_mean(params, x_t, d_prev):
    """Single-step mean forecast."""
    return params['a'] + params['phi'] * d_prev + params['beta'] @ x_t


def order_plug(params, x_t, d_prev, tau):
    """Plug-in critical-fractile order using parametric sigma."""
    mu_hat = predict_mean(params, x_t, d_prev)
    return mu_hat + params['sigma'] * norm.ppf(tau)


def train_method_A(shared, cfg):
    """Method A: OLS plus critical-fractile plug-in."""
    n_seeds = cfg['n_seeds']
    T_te = cfg['T_test']
    costs = np.zeros((n_seeds, T_te))
    mse_in = np.zeros(n_seeds)
    mse_out = np.zeros(n_seeds)
    fitted = []
    for i, ds in enumerate(shared['datasets']):
        params = fit_ols_one(ds, cfg)
        mse_in[i] = float(np.mean(params['residuals'] ** 2))
        # Test-time orders
        x_te = ds['x_test']
        d_te = ds['d_test']
        sq = 0.0
        for t in range(1, T_te):
            q = order_plug(params, x_te[t], d_te[t - 1], TAU_STAR)
            costs[i, t] = newsvendor_cost(q, d_te[t], cfg['c_u'], cfg['c_o'])
            mu_hat = predict_mean(params, x_te[t], d_te[t - 1])
            sq += (d_te[t] - mu_hat) ** 2
        mse_out[i] = sq / (T_te - 1)
        fitted.append(params)
    return {'costs': costs, 'mse_in': mse_in, 'mse_out': mse_out,
            'params_by_seed': fitted}


# ----------------------------------------------------------------------------
# Method B: SPO+ end-to-end (gradient descent on realized newsvendor loss)

def train_method_B(shared, cfg):
    """Method B: task-loss subgradient descent on the realised newsvendor loss
    (decision-focused learning, Donti-Amos-Kolter 2017), in the same
    parametric family as Method A."""
    n_seeds = cfg['n_seeds']
    T_te = cfg['T_test']
    costs = np.zeros((n_seeds, T_te))
    mse_in = np.zeros(n_seeds)
    mse_out = np.zeros(n_seeds)
    fitted = []
    for i, ds in enumerate(shared['datasets']):
        # Initialise from OLS for stability
        if cfg.get('init_from_ols', True):
            init = fit_ols_one(ds, cfg)
            theta = np.concatenate([
                [init['a'], init['phi']],
                init['beta'],
                [np.log(init['sigma'])],   # log-sigma for positivity
            ])
        else:
            theta = np.zeros(2 + cfg['beta_dim'] + 1)
            theta[1] = 0.5  # init phi
            theta[-1] = np.log(cfg['sigma0'])
        theta = task_loss_train(theta, ds, cfg)
        params = unpack_theta(theta, cfg['beta_dim'])
        # In-sample residual MSE
        x_tr = ds['x_train']
        d_tr = ds['d_train']
        sq_in = 0.0
        for t in range(1, len(d_tr)):
            mu_hat = predict_mean(params, x_tr[t], d_tr[t - 1])
            sq_in += (d_tr[t] - mu_hat) ** 2
        mse_in[i] = sq_in / (len(d_tr) - 1)
        # Test-time costs
        x_te = ds['x_test']
        d_te = ds['d_test']
        sq_out = 0.0
        for t in range(1, T_te):
            q = order_plug(params, x_te[t], d_te[t - 1], TAU_STAR)
            costs[i, t] = newsvendor_cost(q, d_te[t], cfg['c_u'], cfg['c_o'])
            mu_hat = predict_mean(params, x_te[t], d_te[t - 1])
            sq_out += (d_te[t] - mu_hat) ** 2
        mse_out[i] = sq_out / (T_te - 1)
        fitted.append(params)
    return {'costs': costs, 'mse_in': mse_in, 'mse_out': mse_out,
            'params_by_seed': fitted}


def unpack_theta(theta, p):
    return {
        'a': float(theta[0]),
        'phi': float(theta[1]),
        'beta': theta[2:2 + p].copy(),
        'sigma': float(np.exp(theta[-1])),
    }


def task_loss_train(theta, ds, cfg):
    """Subgradient descent on the realised newsvendor loss (task-loss training).

    Decision rule q = a + phi*d_{t-1} + beta'x + sigma * Phi_inv(tau).
    Loss L_t = c_u * max(d_t - q_t, 0) + c_o * max(q_t - d_t, 0).
    Subgradient: dL/dq = -c_u if d_t > q_t else c_o (with d_t = q_t -> 0).
    Backpropagate through q's parameters. This is the
    Donti-Amos-Kolter (2017) decision-focused-learning approach; not the
    Elmachtoub-Grigas SPO+ surrogate, which requires linear-in-decision cost.
    """
    p = cfg['beta_dim']
    x_tr = ds['x_train']
    d_tr = ds['d_train']
    T = len(d_tr)
    lr = cfg['lr']
    epochs = cfg['epochs']
    tau_q = norm.ppf(TAU_STAR)
    n = T - 1
    # Batch SGD across epochs over all (t -> t+1) pairs
    for epoch in range(epochs):
        a, phi, beta, log_sigma = theta[0], theta[1], theta[2:2 + p], theta[-1]
        sigma = np.exp(log_sigma)
        # Vectorised forward and gradient
        d_prev = d_tr[:-1]
        d_now = d_tr[1:]
        x_now = x_tr[1:]
        mu_hat = a + phi * d_prev + x_now @ beta
        q_hat = mu_hat + sigma * tau_q
        excess = d_now > q_hat
        # dL/dq = -c_u where excess else c_o
        g_q = np.where(excess, -cfg['c_u'], cfg['c_o'])
        # parameter gradients via chain rule
        g_a = g_q.mean()
        g_phi = (g_q * d_prev).mean()
        g_beta = (g_q[:, None] * x_now).mean(axis=0)
        g_log_sigma = (g_q * sigma * tau_q).mean()
        grad = np.concatenate([[g_a, g_phi], g_beta, [g_log_sigma]])
        theta = theta - lr * grad
    return theta


# ----------------------------------------------------------------------------
# Method C: rollout with empirical residual quantile

def train_method_C(shared, cfg):
    """Method C: OLS mean prediction + empirical residual quantile for the order.

    This is a one-step rollout under the certainty-equivalent fitted AR(1) mean,
    where the cost-to-go is approximated by Monte Carlo over training residuals.
    """
    n_seeds = cfg['n_seeds']
    T_te = cfg['T_test']
    costs = np.zeros((n_seeds, T_te))
    mse_in = np.zeros(n_seeds)
    mse_out = np.zeros(n_seeds)
    fitted = []
    for i, ds in enumerate(shared['datasets']):
        params = fit_ols_one(ds, cfg)
        mse_in[i] = float(np.mean(params['residuals'] ** 2))
        residuals = params['residuals']
        q_residual = float(np.quantile(residuals, TAU_STAR))
        x_te = ds['x_test']
        d_te = ds['d_test']
        sq_out = 0.0
        for t in range(1, T_te):
            mu_hat = predict_mean(params, x_te[t], d_te[t - 1])
            q = mu_hat + q_residual
            costs[i, t] = newsvendor_cost(q, d_te[t], cfg['c_u'], cfg['c_o'])
            sq_out += (d_te[t] - mu_hat) ** 2
        mse_out[i] = sq_out / (T_te - 1)
        fitted.append({**params, 'q_residual': q_residual})
    return {'costs': costs, 'mse_in': mse_in, 'mse_out': mse_out,
            'params_by_seed': fitted}


# ----------------------------------------------------------------------------
# Compute pipeline

ALGO_REGISTRY = {
    'OLS': (train_method_A, A_CONFIG),
    'SPO': (train_method_B, B_CONFIG),
    'Rollout': (train_method_C, C_CONFIG),
}

LABELS = {
    'OLS': r'A: OLS + critical fractile',
    'SPO': r'B: Task-loss fine-tune',
    'Rollout': r'C: Bertsekas rollout',
}

METHOD_COLOR = {
    'OLS': COLORS['gray'],
    'SPO': COLORS['blue'],
    'Rollout': COLORS['orange'],
}


def compute_data(force=None):
    force = force or set()
    shared = compute_or_load(CACHE_DIR, SCRIPT_NAME, 'shared', SHARED_CONFIG,
                             compute_shared, SHARED_CONFIG,
                             force=('shared' in force))
    results = {}
    for name, (fn, ccfg) in ALGO_REGISTRY.items():
        results[name] = compute_or_load(
            CACHE_DIR, SCRIPT_NAME, name, ccfg,
            fn, shared, ccfg,
            force=(name in force or 'shared' in force),
        )
    return {'shared': shared, 'results': results}


# ----------------------------------------------------------------------------
# Output generation

def cumulative_regret(costs, oracle):
    """Per-step regret accumulated over the test horizon."""
    return np.cumsum(costs - oracle, axis=1)


def summary_table(data):
    """Produce per-method (mean, se) of MSE_in, MSE_out, mean_regret."""
    oracle = data['shared']['oracle_costs']
    rows = []
    for name, res in data['results'].items():
        reg = (res['costs'] - oracle).mean(axis=1)
        mse_in_mean, mse_in_se = res['mse_in'].mean(), res['mse_in'].std(ddof=1) / np.sqrt(len(res['mse_in']))
        mse_out_mean, mse_out_se = res['mse_out'].mean(), res['mse_out'].std(ddof=1) / np.sqrt(len(res['mse_out']))
        reg_mean, reg_se = reg.mean(), reg.std(ddof=1) / np.sqrt(len(reg))
        rows.append((name, mse_in_mean, mse_in_se, mse_out_mean, mse_out_se, reg_mean, reg_se))
    return rows


def generate_outputs(data):
    rows = summary_table(data)
    # ----- Figure: cumulative regret with shaded SE -----
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    oracle = data['shared']['oracle_costs']
    t = np.arange(SHARED_CONFIG['T_test'])
    for name, res in data['results'].items():
        cr = cumulative_regret(res['costs'], oracle)
        mean = cr.mean(axis=0)
        se = cr.std(axis=0, ddof=1) / np.sqrt(cr.shape[0])
        ax.plot(t, mean, label=LABELS[name], color=METHOD_COLOR[name], linewidth=1.4)
        ax.fill_between(t, mean - se, mean + se,
                        color=METHOD_COLOR[name], alpha=0.2, linewidth=0)
    ax.axhline(0, **BENCH_STYLE, label='oracle')
    ax.set_xlabel('test horizon $t$')
    ax.set_ylabel('cumulative regret over oracle')
    ax.set_title(f'Decision-focused forecasting under AR(1) demand '
                 f'({SHARED_CONFIG["n_seeds"]} seeds, '
                 f'{"misspec" if SHARED_CONFIG["misspec"] else "well-spec"})')
    ax.legend(loc='upper left', frameon=False)
    fig_path = os.path.join(OUT_DIR, 'decision_focused_ar1.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure: {fig_path}')

    # ----- LaTeX table -----
    tex_path = os.path.join(OUT_DIR, 'decision_focused_ar1_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Generated by decision_focused_ar1.py\n')
        f.write('\\begin{tabular}{lccc}\n')
        f.write('\\hline\\hline\n')
        f.write('Method & In-sample MSE & Out-of-sample MSE & Mean newsvendor regret \\\\\n')
        f.write('\\hline\n')
        for name, mi, mi_se, mo, mo_se, r, r_se in rows:
            label = LABELS[name].replace('$^+$', '$^{+}$')
            f.write(f'{label} & ${mi:.3f}\\,({mi_se:.3f})$ & ${mo:.3f}\\,({mo_se:.3f})$ & '
                    f'${r:.3f}\\,({r_se:.3f})$ \\\\\n')
        f.write('\\hline\\hline\n')
        f.write('\\end{tabular}\n')
    print(f'  Table:  {tex_path}')

    # ----- Stdout summary table -----
    print('\nSummary (means with standard errors over '
          f'{SHARED_CONFIG["n_seeds"]} seeds):')
    print(f'{"Method":<22} {"MSE_in":>14} {"MSE_out":>14} {"Regret":>14}')
    for name, mi, mi_se, mo, mo_se, r, r_se in rows:
        print(f'{LABELS[name]:<22} {mi:>8.3f} ({mi_se:.3f})'
              f' {mo:>8.3f} ({mo_se:.3f})'
              f' {r:>8.3f} ({r_se:.3f})')


# ----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_component_args(parser)
    args = parser.parse_args()

    force = parse_force_set(args)
    print(f'Config:')
    print(f'  T_train={SHARED_CONFIG["T_train"]}, T_test={SHARED_CONFIG["T_test"]}, '
          f'n_seeds={SHARED_CONFIG["n_seeds"]}, misspec={SHARED_CONFIG["misspec"]}')
    print(f'  c_u={SHARED_CONFIG["c_u"]}, c_o={SHARED_CONFIG["c_o"]}, '
          f'tau*={TAU_STAR:.3f}')
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
