#!/usr/bin/env python3
"""
curve_learning_pricing.py
Chapter 7: Economic Bandits

Minimal Weaver-style dynamic-pricing replication.

The simulation follows the core ten-price Beta-WTP experiment in Weaver, Kumar,
and Jain (2025): demand is learned at the curve level, prices are chosen from
P={0.1,...,1.0}, and the firm updates prices every 10 consumers. The code is an
independent finite-grid implementation, not the authors' replication package.
"""

import argparse
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta as beta_dist
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sims.plot_style import apply_style, COLORS, FIG_TRIPLE
from sims.sim_cache import add_cache_args, load_results, save_results

apply_style()


# =============================================================================
# Configuration
# =============================================================================
K = 10
T = 2_500
N_SEEDS = 1_000
N_WORKERS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 10
PRICES = np.arange(1, K + 1, dtype=float) / K
CHECKPOINTS = [500, 2_500]
SAMPLE_INTERVAL = 10

# GP kernel hyperparameters. Listed in CONFIG below so the cache invalidates
# correctly when any of these are edited.
KERNEL_LENGTHSCALE = 0.18
KERNEL_VARIANCE = 0.20
OBSERVATION_NOISE = 0.25  # Bernoulli worst-case Var(y) bound, scaled by 1/n_a
GP_UCB_BETA = 1.8

SCENARIOS = {
    'b29': {
        'label': r'$B(2,9)$',
        'description': 'Low optimal price',
        'beta': (2.0, 9.0),
    },
    'b22': {
        'label': r'$B(2,2)$',
        'description': 'Middle optimal price',
        'beta': (2.0, 2.0),
    },
    'b92': {
        'label': r'$B(9,2)$',
        'description': 'High optimal price',
        'beta': (9.0, 2.0),
    },
}

ALG_NAMES = [
    'ts',
    'gp_ucb',
    'gp_ts',
    'mono_gp_ucb',
    'mono_gp_ts',
]

ALG_LABELS = {
    'ts': 'TS',
    'gp_ucb': 'GP-UCB',
    'gp_ts': 'GP-TS',
    'mono_gp_ucb': 'GP-UCB-M',
    'mono_gp_ts': 'GP-TS-M',
}

ALG_COLORS = {
    'ts': COLORS['orange'],
    'gp_ucb': COLORS['green'],
    'gp_ts': COLORS['cyan'],
    'mono_gp_ucb': COLORS['red'],
    'mono_gp_ts': COLORS['brown'],
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
SCRIPT_NAME = 'curve_learning_pricing'
CONFIG = {
    'K': K,
    'T': T,
    'N_SEEDS': N_SEEDS,
    'BATCH_SIZE': BATCH_SIZE,
    'PRICES': PRICES.tolist(),
    'CHECKPOINTS': CHECKPOINTS,
    'SAMPLE_INTERVAL': SAMPLE_INTERVAL,
    'SCENARIOS': SCENARIOS,
    'KERNEL_LENGTHSCALE': KERNEL_LENGTHSCALE,
    'KERNEL_VARIANCE': KERNEL_VARIANCE,
    'OBSERVATION_NOISE': OBSERVATION_NOISE,
    'GP_UCB_BETA': GP_UCB_BETA,
    'version': 10,
}


# =============================================================================
# Demand model
# =============================================================================
class BetaWTPDemand:
    """Unit-demand customers with WTP drawn from a Beta distribution."""

    def __init__(self, beta_params):
        self.a, self.b = beta_params
        grid = np.linspace(0.0, 1.0, 200_001)
        dense_profit = grid * beta_dist.sf(grid, self.a, self.b)
        self.true_opt_price = float(grid[int(np.argmax(dense_profit))])
        self.true_opt_profit = float(np.max(dense_profit))

    def draw_wtp(self, rng, n):
        return rng.beta(self.a, self.b, size=n)

    def demand(self, prices):
        return beta_dist.sf(np.asarray(prices), self.a, self.b)

    def profit(self, prices):
        prices = np.asarray(prices)
        return prices * self.demand(prices)


# =============================================================================
# Independent-arm bandits
# =============================================================================
class PricingTS:
    """Beta-Bernoulli Thompson sampling on demand, scaled by price."""

    def __init__(self, prices):
        self.prices = prices
        self.K = len(prices)
        self.alpha = np.ones(self.K)
        self.beta = np.ones(self.K)

    def select_arm(self, rng):
        demand_draw = rng.beta(self.alpha, self.beta)
        return int(np.argmax(self.prices * demand_draw))

    def update_batch(self, arm, sales, n_customers):
        self.alpha[arm] += sales
        self.beta[arm] += n_customers - sales


# =============================================================================
# GP curve-learning algorithms
# =============================================================================
def rbf_kernel_cross(x, z, lengthscale=KERNEL_LENGTHSCALE, variance=KERNEL_VARIANCE):
    x = np.asarray(x)
    z = np.asarray(z)
    sqdist = (x[:, None] - z[None, :]) ** 2
    return variance * np.exp(-0.5 * sqdist / lengthscale ** 2)


def rbf_kernel(x, lengthscale=KERNEL_LENGTHSCALE, variance=KERNEL_VARIANCE):
    return rbf_kernel_cross(x, x, lengthscale, variance)


def cov_f_deriv(x, z, lengthscale=KERNEL_LENGTHSCALE, variance=KERNEL_VARIANCE):
    """Covariance between f(x) and f'(z) for an RBF kernel."""
    kxz = rbf_kernel_cross(x, z, lengthscale, variance)
    return ((x[:, None] - z[None, :]) / lengthscale ** 2) * kxz


def cov_deriv_f(x, z, lengthscale=KERNEL_LENGTHSCALE, variance=KERNEL_VARIANCE):
    """Covariance between f'(x) and f(z) for an RBF kernel."""
    kxz = rbf_kernel_cross(x, z, lengthscale, variance)
    return (-(x[:, None] - z[None, :]) / lengthscale ** 2) * kxz


def cov_deriv_deriv(x, z, lengthscale=KERNEL_LENGTHSCALE, variance=KERNEL_VARIANCE):
    """Covariance between f'(x) and f'(z) for an RBF kernel."""
    diff = x[:, None] - z[None, :]
    kxz = rbf_kernel_cross(x, z, lengthscale, variance)
    return (1 / lengthscale ** 2 - diff ** 2 / lengthscale ** 4) * kxz


def gaussian_update(mu, cov, obs_index, y, noise_var):
    """One Gaussian update for a scalar noisy demand-rate observation."""
    s = cov[:, obs_index].copy()
    pred_var = cov[obs_index, obs_index] + noise_var
    gain = s / pred_var
    mu = mu + gain * (y - mu[obs_index])
    cov = cov - np.outer(gain, s)
    cov = 0.5 * (cov + cov.T)
    diag = np.maximum(np.diag(cov), 1e-9)
    np.fill_diagonal(cov, diag)
    return mu, cov


def sample_normal_below(rng, mean, sd, upper=0.0):
    """Fast inverse-CDF draw from N(mean, sd^2) truncated above at upper."""
    upper_cdf = ndtr((upper - mean) / sd)
    upper_cdf = np.clip(upper_cdf, 1e-300, 1.0)
    u = rng.uniform(0.0, upper_cdf)
    return mean + sd * ndtri(np.clip(u, 1e-300, 1.0 - 1e-15))


class DemandGP:
    """GP-UCB/GP-TS on demand values over the tested price grid."""

    def __init__(self, prices, mode='ucb', noise_var=OBSERVATION_NOISE,
                 beta=GP_UCB_BETA, seed=0):
        self.prices = prices
        self.K = len(prices)
        self.mode = mode
        self.noise_var = noise_var
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        self.mu = np.clip(1.0 - prices, 0.0, 1.0)
        self.cov = rbf_kernel(prices) + 1e-6 * np.eye(self.K)
        self.n_obs = 0

    def select_arm(self, rng):
        if self.n_obs == 0:
            return int(rng.randint(self.K))
        if self.mode == 'ucb':
            sd = np.sqrt(np.maximum(np.diag(self.cov), 1e-10))
            demand_curve = self.mu + self.beta * sd
        elif self.mode == 'ts':
            cov = 0.5 * (self.cov + self.cov.T) + 1e-8 * np.eye(self.K)
            demand_curve = self.rng.multivariate_normal(self.mu, cov)
        else:
            raise ValueError(f'Unknown GP mode: {self.mode}')
        demand_curve = np.clip(demand_curve, 0.0, 1.0)
        return int(np.argmax(self.prices * demand_curve))

    def update_batch(self, arm, sales, n_customers):
        y = sales / n_customers
        noise_var = self.noise_var / n_customers
        self.mu, self.cov = gaussian_update(self.mu, self.cov, arm, y, noise_var)
        self.n_obs += n_customers


class MonotoneDemandGP:
    """Derivative-sign-constrained monotone GP bandit.

    The state vector is [D(0), D(p_1), ..., D(p_K), D'(0), D'(p_1), ...,
    D'(p_K)]. Purchases update the demand-value component at the chosen price.
    GP-TS-M draws derivative paths constrained to D'(p) <= 0 and reconstructs
    demand by integration. GP-UCB-M uses high quantiles from constrained draws.
    """

    def __init__(self, prices, mode='ucb', noise_var=OBSERVATION_NOISE, seed=0,
                 gibbs_sweeps=2, ucb_samples=8, ucb_quantile=0.9):
        self.prices = prices
        self.K = len(prices)
        self.mode = mode
        self.noise_var = noise_var
        self.rng = np.random.RandomState(seed)
        self.derivative_grid = np.r_[0.0, prices]
        self.n_deriv = len(self.derivative_grid)
        self.f0_index = 0
        self.f_start = 1
        self.d_start = 1 + self.K
        self.deriv_slice = slice(self.d_start, self.d_start + self.n_deriv)
        self.gibbs_sweeps = gibbs_sweeps
        self.ucb_samples = ucb_samples
        self.ucb_quantile = ucb_quantile
        self.n_obs = 0
        self._precision_cache = None
        self._precision_age = 1

        self.mu = np.concatenate([
            np.array([1.0]),
            np.clip(1.0 - prices, 0.0, 1.0),
            -np.ones(self.n_deriv),
        ])
        self.cov = self._prior_covariance()

    def _prior_covariance(self):
        zero = np.array([0.0])
        k00 = rbf_kernel_cross(zero, zero)
        k0p = rbf_kernel_cross(zero, self.prices)
        k0d = cov_f_deriv(zero, self.derivative_grid)
        kpp = rbf_kernel(self.prices)
        kpd = cov_f_deriv(self.prices, self.derivative_grid)
        kd0 = cov_deriv_f(self.derivative_grid, zero)
        kdp = cov_deriv_f(self.derivative_grid, self.prices)
        kdd = cov_deriv_deriv(self.derivative_grid, self.derivative_grid)

        top = np.concatenate([k00, k0p, k0d], axis=1)
        middle = np.concatenate([k0p.T, kpp, kpd], axis=1)
        bottom = np.concatenate([kd0, kdp, kdd], axis=1)
        cov = np.concatenate([top, middle, bottom], axis=0)
        cov = 0.5 * (cov + cov.T)
        cov += 1e-6 * np.eye(cov.shape[0])
        return cov

    def _derivative_precision(self, cov_d):
        if self._precision_cache is None or self._precision_age:
            cov_d = 0.5 * (cov_d + cov_d.T) + 1e-7 * np.eye(cov_d.shape[0])
            self._precision_cache = np.linalg.inv(cov_d)
            self._precision_age = 0
        return self._precision_cache

    def _sample_negative_derivatives(self, mu_d, cov_d):
        precision = self._derivative_precision(cov_d)
        draw = np.minimum(mu_d, -1e-4)
        centered = draw - mu_d
        order = np.arange(len(draw))
        for _ in range(self.gibbs_sweeps):
            self.rng.shuffle(order)
            for i in order:
                qii = max(precision[i, i], 1e-10)
                rest_dot = precision[i, :] @ centered - qii * centered[i]
                cond_mean = mu_d[i] - rest_dot / qii
                cond_sd = np.sqrt(1.0 / qii)
                draw[i] = sample_normal_below(self.rng, cond_mean, cond_sd)
                centered[i] = draw[i] - mu_d[i]
        return draw

    def _intercept_conditional_draw(self, derivatives, mu_d, cov_d):
        mu0 = self.mu[self.f0_index]
        var0 = self.cov[self.f0_index, self.f0_index]
        cov0d = self.cov[self.f0_index, self.deriv_slice]
        precision = self._derivative_precision(cov_d)
        cond_mean = mu0 + cov0d @ precision @ (derivatives - mu_d)
        cond_var = var0 - cov0d @ precision @ cov0d.T
        cond_sd = np.sqrt(max(cond_var, 1e-8))
        return float(np.clip(self.rng.normal(cond_mean, cond_sd), 0.0, 1.0))

    def _integrate_derivatives(self, intercept, derivatives):
        demand_grid = np.empty(self.n_deriv)
        demand_grid[0] = intercept
        x = self.derivative_grid
        for i in range(1, self.n_deriv):
            dx = x[i] - x[i - 1]
            demand_grid[i] = demand_grid[i - 1] + 0.5 * dx * (
                derivatives[i - 1] + derivatives[i]
            )
        return np.clip(demand_grid[1:], 0.0, 1.0)

    def _sample_curve(self):
        mu_d = self.mu[self.deriv_slice]
        cov_d = self.cov[self.deriv_slice, self.deriv_slice]
        derivatives = self._sample_negative_derivatives(mu_d, cov_d)
        intercept = self._intercept_conditional_draw(derivatives, mu_d, cov_d)
        return self._integrate_derivatives(intercept, derivatives)

    def select_arm(self, rng):
        if self.n_obs == 0:
            return int(rng.randint(self.K))
        if self.mode == 'ts':
            demand_curve = self._sample_curve()
        elif self.mode == 'ucb':
            curves = np.array([self._sample_curve() for _ in range(self.ucb_samples)])
            demand_curve = np.quantile(curves, self.ucb_quantile, axis=0)
        else:
            raise ValueError(f'Unknown monotone GP mode: {self.mode}')
        return int(np.argmax(self.prices * demand_curve))

    def update_batch(self, arm, sales, n_customers):
        y = sales / n_customers
        noise_var = self.noise_var / n_customers
        obs_index = self.f_start + arm
        self.mu, self.cov = gaussian_update(self.mu, self.cov, obs_index, y, noise_var)
        self.n_obs += n_customers
        self._precision_age = 1


# =============================================================================
# Simulation
# =============================================================================
def make_algorithms(seed):
    return {
        'ts': (PricingTS(PRICES), np.random.RandomState(seed + 2_000)),
        'gp_ucb': (
            DemandGP(PRICES, mode='ucb', seed=seed + 3_000),
            np.random.RandomState(seed + 3_001),
        ),
        'gp_ts': (
            DemandGP(PRICES, mode='ts', seed=seed + 4_000),
            np.random.RandomState(seed + 4_001),
        ),
        'mono_gp_ucb': (
            MonotoneDemandGP(PRICES, mode='ucb', seed=seed + 5_000),
            np.random.RandomState(seed + 5_001),
        ),
        'mono_gp_ts': (
            MonotoneDemandGP(PRICES, mode='ts', seed=seed + 6_000),
            np.random.RandomState(seed + 6_001),
        ),
    }


def run_one(scenario_config, seed):
    rng = np.random.RandomState(seed)
    demand = BetaWTPDemand(scenario_config['beta'])
    price_profits = demand.profit(PRICES)
    price_set_opt_arm = int(np.argmax(price_profits))
    price_set_opt_price = float(PRICES[price_set_opt_arm])
    price_set_opt_profit = float(price_profits[price_set_opt_arm])

    algorithms = make_algorithms(seed)
    n_samples = T // SAMPLE_INTERVAL
    cum_profit = {name: np.zeros(n_samples) for name in ALG_NAMES}
    running_profit = {name: 0.0 for name in ALG_NAMES}

    for batch_start in range(0, T, BATCH_SIZE):
        n_customers = min(BATCH_SIZE, T - batch_start)
        valuations = demand.draw_wtp(rng, n_customers)

        for name, (alg, alg_rng) in algorithms.items():
            arm = alg.select_arm(alg_rng)
            price = PRICES[arm]
            sales = int(np.sum(valuations >= price))
            running_profit[name] += price * sales
            alg.update_batch(arm, sales, n_customers)

        for t in range(batch_start + SAMPLE_INTERVAL, batch_start + n_customers + 1,
                       SAMPLE_INTERVAL):
            idx = t // SAMPLE_INTERVAL - 1
            for name in ALG_NAMES:
                cum_profit[name][idx] = running_profit[name]

    return {
        'cum_profit': cum_profit,
        'price_set_opt_price': price_set_opt_price,
        'price_set_opt_profit': price_set_opt_profit,
        'true_opt_price': demand.true_opt_price,
        'true_opt_profit': demand.true_opt_profit,
    }


def run_one_for_pool(args):
    scenario_config, seed = args
    return run_one(scenario_config, seed)


def _regret_slope(time_points, profit_curve, oracle):
    """Log-log slope of cumulative regret R_t = t * oracle - profit_t.

    Returns (slope, stderr) computed on the regression log R_t = a + b log t,
    restricted to the second half of the trajectory to avoid the initial burn-in
    where regret is non-monotone. Theoretical rates: O(sqrt(T)) gives slope 0.5,
    O(log T) gives slope -> 0.
    """
    regret = np.asarray(time_points, dtype=float) * float(oracle) - np.asarray(profit_curve)
    mask = regret > 1e-6
    mask[: len(mask) // 2] = False
    if mask.sum() < 5:
        return float('nan'), float('nan')
    x = np.log(np.asarray(time_points)[mask])
    y = np.log(regret[mask])
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - y_mean))
    slope = sxy / sxx
    resid = y - (y_mean + slope * (x - x_mean))
    sigma2 = np.sum(resid ** 2) / max(n - 2, 1)
    stderr = float(np.sqrt(sigma2 / sxx))
    return float(slope), stderr


def _print_summary(data):
    print('=' * 72)
    print('WEAVER-STYLE CURVE-LEARNING PRICING REPLICATION')
    print('=' * 72)
    print(f'K={K}, T={T:,}, seeds={N_SEEDS}, batch={BATCH_SIZE}, workers={N_WORKERS}')
    print(f'GP kernel: lengthscale={KERNEL_LENGTHSCALE}, variance={KERNEL_VARIANCE}, '
          f'obs_noise={OBSERVATION_NOISE}, gp_ucb_beta={GP_UCB_BETA}')
    print('Algorithms:', ', '.join(ALG_LABELS[name] for name in ALG_NAMES))
    print()
    results = data['results']
    time_points = data['time_points']
    for scenario_name, scenario_config in SCENARIOS.items():
        scenario = results[scenario_name]
        print(f"Scenario: {scenario_config['label']} ({scenario_config['description']})")
        print(f"  Price-set optimum: {scenario['price_set_opt_prices'][0]:.2f}")
        oracle = scenario['price_set_opt_profits'][0]
        for name in ALG_NAMES:
            final_profit = scenario['profit_arrays'][name][:, -1].mean()
            pct = 100 * final_profit / (T * oracle)
            print(f'  {ALG_LABELS[name]:<10} final profit {pct:6.1f}% of grid oracle')
        print()
        print('  Empirical regret-rate slope (log R_t vs log t, second-half fit):')
        for name in ALG_NAMES:
            mean_profit = scenario['profit_arrays'][name].mean(axis=0)
            slope, stderr = _regret_slope(time_points, mean_profit, oracle)
            if np.isnan(slope):
                print(f'    {ALG_LABELS[name]:<10} slope=N/A (regret never positive)')
            else:
                print(f'    {ALG_LABELS[name]:<10} slope={slope:+.3f} +/- {stderr:.3f}')
        print()


def compute_data():
    cached = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
    if cached is not None:
        print('Loaded from cache.')
        _print_summary(cached)
        return cached

    time_points = np.arange(SAMPLE_INTERVAL, T + 1, SAMPLE_INTERVAL)
    results = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        print(f"Running scenario: {scenario_config['label']} ({scenario_config['description']})")
        jobs = [(scenario_config, seed) for seed in range(N_SEEDS)]
        if N_WORKERS > 1:
            with mp.Pool(processes=N_WORKERS) as pool:
                runs = list(tqdm(
                    pool.imap(run_one_for_pool, jobs, chunksize=4),
                    total=N_SEEDS,
                    desc=scenario_name,
                    unit='seed',
                ))
        else:
            runs = [
                run_one_for_pool(job)
                for job in tqdm(jobs, desc=scenario_name, unit='seed')
            ]

        profit_arrays = {
            name: np.array([run['cum_profit'][name] for run in runs])
            for name in ALG_NAMES
        }
        results[scenario_name] = {
            'profit_arrays': profit_arrays,
            'mean_profit': {name: arr.mean(axis=0) for name, arr in profit_arrays.items()},
            'se_profit': {name: arr.std(axis=0) / np.sqrt(N_SEEDS)
                          for name, arr in profit_arrays.items()},
            'price_set_opt_prices': np.array([run['price_set_opt_price'] for run in runs]),
            'price_set_opt_profits': np.array([run['price_set_opt_profit'] for run in runs]),
            'true_opt_prices': np.array([run['true_opt_price'] for run in runs]),
            'true_opt_profits': np.array([run['true_opt_profit'] for run in runs]),
        }

    data = {
        'time_points': time_points,
        'results': results,
    }
    save_results(CACHE_DIR, SCRIPT_NAME, CONFIG, data)
    _print_summary(data)
    return data


# =============================================================================
# Outputs
# =============================================================================
def pct_of_oracle(profit, t, oracle_profit):
    return 100 * profit / (t * oracle_profit)


def generate_outputs(data):
    time_points = data['time_points']
    results = data['results']
    tex_end = r'\\' + '\n'

    print('Generating Weaver-style curve-learning figure...')
    fig, axes = plt.subplots(1, 3, figsize=FIG_TRIPLE, sharey=True)

    for ax, (scenario_name, scenario_config) in zip(axes, SCENARIOS.items()):
        scenario = results[scenario_name]
        oracle = scenario['price_set_opt_profits'][0]
        scale = time_points * oracle
        for name in ALG_NAMES:
            mean = 100 * scenario['mean_profit'][name] / scale
            se = 100 * scenario['se_profit'][name] / scale
            lower = np.maximum(mean - 2 * se, 0.0)
            upper = np.minimum(mean + 2 * se, 110.0)
            ax.plot(time_points, mean, label=ALG_LABELS[name],
                    color=ALG_COLORS[name], linewidth=1.5)
            ax.fill_between(time_points, lower, upper, color=ALG_COLORS[name],
                            alpha=0.10, linewidth=0)
        p_star = scenario['price_set_opt_prices'][0]
        ax.set_title(
            f"{scenario_config['label']}\n{scenario_config['description']}, "
            f"$p^*_P={p_star:.1f}$"
        )
        ax.set_xlabel('Customers $t$')
        ax.set_xlim(0, T)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel('Cumulative profit (% of price-set oracle)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, 'curve_learning_pricing_pct_oracle.png'),
                bbox_inches='tight')
    print('  Saved: curve_learning_pricing_pct_oracle.png')

    print('Generating Weaver-style curve-learning tables...')
    tex_path = os.path.join(OUT_DIR, 'curve_learning_pricing_results.tex')
    with open(tex_path, 'w') as f:
        f.write('\\begin{tabular}{llrrrr}\n')
        f.write('\\toprule\n')
        f.write('Scenario & Algorithm & $p^*_P$ & $\\Pi_{500}/\\Pi^*_P$ & '
                '$\\Pi_{2{,}500}/\\Pi^*_P$ & $\\Pi_{2{,}500}/\\Pi^*$ '
                + tex_end)
        f.write('\\midrule\n')
        for scenario_name, scenario_config in SCENARIOS.items():
            scenario = results[scenario_name]
            p_star = scenario['price_set_opt_prices'][0]
            price_oracle = scenario['price_set_opt_profits'][0]
            true_oracle = scenario['true_opt_profits'][0]
            for name in ALG_NAMES:
                row = f"{scenario_config['label']} & {ALG_LABELS[name]} & {p_star:.1f}"
                for cp in CHECKPOINTS:
                    idx = cp // SAMPLE_INTERVAL - 1
                    mean_profit = scenario['profit_arrays'][name][:, idx].mean()
                    row += f" & {pct_of_oracle(mean_profit, cp, price_oracle):.1f}\\%"
                final_profit = scenario['profit_arrays'][name][:, -1].mean()
                row += f" & {pct_of_oracle(final_profit, T, true_oracle):.1f}\\%"
                row += ' ' + tex_end
                f.write(row)
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
    print(f'  Saved: {tex_path}')

    summary_path = os.path.join(OUT_DIR, 'curve_learning_pricing_summary.tex')
    with open(summary_path, 'w') as f:
        f.write('\\begin{tabular}{lrrrrr}\n')
        f.write('\\toprule\n')
        f.write('WTP & $p^*_P$ & TS & GP-TS & GP-TS-M & GP-UCB-M '
                + tex_end)
        f.write('\\midrule\n')
        for scenario_name, scenario_config in SCENARIOS.items():
            scenario = results[scenario_name]
            p_star = scenario['price_set_opt_prices'][0]
            price_oracle = scenario['price_set_opt_profits'][0]
            vals = {}
            for name in ['ts', 'gp_ts', 'mono_gp_ts', 'mono_gp_ucb']:
                final_profit = scenario['profit_arrays'][name][:, -1].mean()
                vals[name] = pct_of_oracle(final_profit, T, price_oracle)
            f.write(
                f"{scenario_config['label']} & {p_star:.1f} & "
                f"{vals['ts']:.1f}\\% & {vals['gp_ts']:.1f}\\% & "
                f"{vals['mono_gp_ts']:.1f}\\% & {vals['mono_gp_ucb']:.1f}\\% "
                + tex_end
            )
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
    print(f'  Saved: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cache_args(parser)
    args = parser.parse_args()
    if args.plots_only:
        data = load_results(CACHE_DIR, SCRIPT_NAME, CONFIG)
        assert data is not None, 'No cache found. Run without --plots-only first.'
    else:
        data = compute_data()
    if not args.data_only:
        generate_outputs(data)
