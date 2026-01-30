"""
Bradley-Terry Preference Learning: RLHF/DPO Simulation
Chapter 7 Simulation

Demonstrates reward learning from pairwise preferences in a controlled setting where
ground truth utilities are known, allowing verification of weight recovery.
Establishes the connection between RLHF and discrete choice econometrics.

Validation framework:
1. Weight recovery - MSE scales as O(1/N), consistent with MLE theory
2. DPO equivalence - DPO and explicit RM produce identical policies
3. Noise robustness - graceful degradation under label noise
4. Fisher Information - empirical variance matches theoretical asymptotic variance
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Environment
    'n_options': 5,                                  # Number of options
    'true_weights': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),  # Ground truth utilities

    # Experiment parameters
    'n_seeds': 20,                                  # Seeds per configuration
    'sample_sizes': [50, 100, 200, 500, 1000],      # Preference pair counts
    'noise_levels': [0.0, 0.05, 0.1, 0.2, 0.3],     # Label flip probabilities

    # Optimization
    'lambda_kl': 1.0,                               # KL penalty coefficient for DPO
    'optimizer': 'L-BFGS-B',                        # Optimization method
    'optimizer_maxiter': 1000,                      # Max optimizer iterations
    'optimizer_ftol': 1e-12,                        # Function tolerance
    'optimizer_gtol': 1e-8,                         # Gradient tolerance

    # Output
    'output_dir': 'ch07_rlhf/sims',
    'figure_dpi': 300,
}

# Extract frequently used config values
N_OPTIONS = CONFIG['n_options']
TRUE_WEIGHTS = CONFIG['true_weights']
N_SEEDS = CONFIG['n_seeds']
SAMPLE_SIZES = CONFIG['sample_sizes']
NOISE_LEVELS = CONFIG['noise_levels']
LAMBDA_KL = CONFIG['lambda_kl']


def print_header():
    """Print comprehensive header with configuration and version info."""
    print("=" * 70)
    print("BRADLEY-TERRY PREFERENCE LEARNING: RLHF/DPO SIMULATION")
    print("Chapter 7 - Reinforcement Learning from Human Feedback")
    print("=" * 70)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    print()
    print("Configuration:")
    print(f"  K = {N_OPTIONS} options")
    print(f"  True weights: {TRUE_WEIGHTS.tolist()}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Optimizer: {CONFIG['optimizer']}")
    print(f"  lambda_KL: {LAMBDA_KL}")
    print(f"  Identification: w[0] = 0 (standard discrete choice normalization)")
    print()


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sample_preference_pair(w_star, noise=0.0):
    """Sample a preference pair according to Bradley-Terry model.

    Args:
        w_star: True utility weights (n_options,)
        noise: Probability of flipping the preference

    Returns:
        (a_win, a_lose): Indices of winning and losing options
    """
    # Sample two distinct options uniformly
    a1, a2 = np.random.choice(N_OPTIONS, size=2, replace=False)

    # Bradley-Terry probability
    p = sigmoid(w_star[a1] - w_star[a2])

    # Apply noise (flip with probability noise)
    if np.random.rand() < noise:
        p = 1 - p

    # Generate preference
    if np.random.rand() < p:
        return (a1, a2)
    else:
        return (a2, a1)


def generate_preference_dataset(w_star, n_pairs, noise=0.0):
    """Generate a dataset of preference pairs."""
    return [sample_preference_pair(w_star, noise) for _ in range(n_pairs)]


def bradley_terry_nll(w, preferences):
    """Negative log-likelihood for Bradley-Terry model.

    Args:
        w: Utility weights (n_options-1,) with w[0] = 0 for identification
        preferences: List of (a_win, a_lose) tuples

    Returns:
        Negative log-likelihood
    """
    # Ensure identification: w[0] = 0
    w_full = np.zeros(N_OPTIONS)
    w_full[1:] = w

    nll = 0.0
    for a_w, a_l in preferences:
        diff = w_full[a_w] - w_full[a_l]
        # log sigmoid(diff) = diff - log(1 + exp(diff)) for stability
        nll -= np.log(sigmoid(diff) + 1e-10)

    return nll


def bradley_terry_gradient(w, preferences):
    """Gradient of negative log-likelihood for Bradley-Terry model."""
    w_full = np.zeros(N_OPTIONS)
    w_full[1:] = w

    grad = np.zeros(N_OPTIONS - 1)
    for a_w, a_l in preferences:
        diff = w_full[a_w] - w_full[a_l]
        p = sigmoid(diff)

        # Gradient: d(-log sigma(diff))/dw = -(1-p) * d(diff)/dw
        # For w[j] (j>0): d(diff)/dw[j] = 1{a_w=j} - 1{a_l=j}
        for j in range(1, N_OPTIONS):
            indicator = float(a_w == j) - float(a_l == j)
            grad[j-1] -= (1 - p) * indicator

    return grad


def train_reward_model(preferences, return_optimizer_info=False):
    """Train reward model via MLE on Bradley-Terry model.

    Normalizes by fixing w[0] = 0 for identification.

    Args:
        preferences: List of (a_win, a_lose) tuples
        return_optimizer_info: If True, return (weights, optimizer_result)

    Returns:
        Estimated weights (n_options,) or (weights, result) if return_optimizer_info
    """
    w0 = np.zeros(N_OPTIONS - 1)

    result = minimize(
        bradley_terry_nll,
        w0,
        args=(preferences,),
        method=CONFIG['optimizer'],
        jac=bradley_terry_gradient,
        options={
            'maxiter': CONFIG['optimizer_maxiter'],
            'ftol': CONFIG['optimizer_ftol'],
            'gtol': CONFIG['optimizer_gtol'],
        }
    )

    # Reconstruct full weight vector
    w_est = np.zeros(N_OPTIONS)
    w_est[1:] = result.x

    if return_optimizer_info:
        return w_est, result
    return w_est


def dpo_loss(theta, preferences, lambda_kl):
    """DPO loss function.

    With uniform reference policy, this simplifies to:
    -sum_i log sigma(lambda_kl * (theta[a_w] - theta[a_l]))
    """
    theta_full = np.zeros(N_OPTIONS)
    theta_full[1:] = theta

    loss = 0.0
    for a_w, a_l in preferences:
        diff = lambda_kl * (theta_full[a_w] - theta_full[a_l])
        loss -= np.log(sigmoid(diff) + 1e-10)

    return loss


def dpo_gradient(theta, preferences, lambda_kl):
    """Gradient of DPO loss."""
    theta_full = np.zeros(N_OPTIONS)
    theta_full[1:] = theta

    grad = np.zeros(N_OPTIONS - 1)
    for a_w, a_l in preferences:
        diff = lambda_kl * (theta_full[a_w] - theta_full[a_l])
        p = sigmoid(diff)

        for j in range(1, N_OPTIONS):
            indicator = float(a_w == j) - float(a_l == j)
            grad[j-1] -= lambda_kl * (1 - p) * indicator

    return grad


def train_dpo_policy(preferences, lambda_kl=1.0, return_optimizer_info=False):
    """Train policy directly via DPO with uniform reference policy."""
    theta0 = np.zeros(N_OPTIONS - 1)

    result = minimize(
        dpo_loss,
        theta0,
        args=(preferences, lambda_kl),
        method=CONFIG['optimizer'],
        jac=lambda t, p, l: dpo_gradient(t, p, l),
        options={
            'maxiter': CONFIG['optimizer_maxiter'],
            'ftol': CONFIG['optimizer_ftol'],
            'gtol': CONFIG['optimizer_gtol'],
        }
    )

    theta_est = np.zeros(N_OPTIONS)
    theta_est[1:] = result.x

    if return_optimizer_info:
        return theta_est, result
    return theta_est


def logits_to_policy(theta, temperature=1.0):
    """Convert logits to softmax policy."""
    exp_theta = np.exp((theta - np.max(theta)) / temperature)
    return exp_theta / np.sum(exp_theta)


def normalize_weights(w, w_star):
    """Normalize estimated weights to match scale of true weights using linear regression."""
    w_centered = w - np.mean(w)
    w_star_centered = w_star - np.mean(w_star)

    if np.std(w_centered) > 1e-10:
        scale = np.std(w_star_centered) / np.std(w_centered)
    else:
        scale = 1.0

    w_normalized = w_centered * scale + np.mean(w_star)
    return w_normalized


def compute_metrics(w_est, w_star):
    """Compute evaluation metrics."""
    w_norm = normalize_weights(w_est, w_star)
    mse = np.mean((w_norm - w_star) ** 2)
    rho, _ = spearmanr(w_est, w_star)
    top1 = int(np.argmax(w_est) == np.argmax(w_star))

    return {'mse': mse, 'rank_corr': rho, 'top1': top1, 'w_norm': w_norm}


def compute_regret(pi, w_star):
    """Compute expected regret of policy."""
    optimal_utility = np.max(w_star)
    expected_utility = np.dot(pi, w_star)
    return optimal_utility - expected_utility


def compute_fisher_information(w, n_comparisons_per_pair=None):
    """Compute the Fisher Information matrix for Bradley-Terry model.

    For Bradley-Terry with K options (K-1 free parameters due to identification),
    the Fisher Information for a single comparison between options i and j is:

    I_{kl}(i,j) = p(1-p) * (1{k=i} - 1{k=j}) * (1{l=i} - 1{l=j})

    where p = sigma(w[i] - w[j]).

    For uniform sampling over all pairs, we sum over all (i,j) pairs.
    """
    # Build K-1 x K-1 Fisher Information matrix
    fisher = np.zeros((N_OPTIONS-1, N_OPTIONS-1))

    # Sum over all pairs (i,j) with i < j
    for i in range(N_OPTIONS):
        for j in range(i+1, N_OPTIONS):
            diff = w[i] - w[j]
            p = sigmoid(diff)
            var = p * (1 - p)

            # Build the outer product for this pair
            # Indices are shifted by 1 because w[0] = 0 is fixed
            for k in range(1, N_OPTIONS):
                for l in range(1, N_OPTIONS):
                    indicator_k = float(k == i) - float(k == j)
                    indicator_l = float(l == i) - float(l == j)
                    fisher[k-1, l-1] += var * indicator_k * indicator_l

    # Normalize by number of pairs (uniform sampling)
    n_pairs = N_OPTIONS * (N_OPTIONS - 1) // 2
    fisher = fisher / n_pairs

    return fisher


def compute_theoretical_variance(w, n_samples):
    """Compute theoretical asymptotic variance Var(w_hat) = I(w)^{-1} / n."""
    fisher = compute_fisher_information(w)
    try:
        fisher_inv = np.linalg.inv(fisher)
        return fisher_inv / n_samples
    except np.linalg.LinAlgError:
        return None


def print_summary_statistics(values, name):
    """Print detailed summary statistics for a list of values."""
    arr = np.array(values)
    print(f"  {name}:")
    print(f"    Mean:   {np.mean(arr):.6f}")
    print(f"    SE:     {np.std(arr)/np.sqrt(len(arr)):.6f}")
    print(f"    Min:    {np.min(arr):.6f}")
    print(f"    Q1:     {np.percentile(arr, 25):.6f}")
    print(f"    Median: {np.median(arr):.6f}")
    print(f"    Q3:     {np.percentile(arr, 75):.6f}")
    print(f"    Max:    {np.max(arr):.6f}")


# =============================================================================
# EXPERIMENT 1: WEIGHT RECOVERY
# =============================================================================

def run_experiment_1():
    """Weight recovery experiment with detailed per-seed tracking."""
    print("=" * 70)
    print("EXPERIMENT 1: WEIGHT RECOVERY")
    print("=" * 70)
    print()
    print("Hypothesis: MSE scales as O(1/N), consistent with MLE theory.")
    print("Metric: MSE between normalized estimated and true weights.")
    print()

    # Storage for all results
    all_results = {n: [] for n in SAMPLE_SIZES}

    for n in SAMPLE_SIZES:
        print(f"--- N = {n} ---")
        print()

        # Per-seed results table header
        header = f"{'Seed':>4} | {'w_est (normalized)':>40} | {'MSE':>8} | {'Rho':>6} | {'Top1':>4} | {'Conv':>4} | {'Iters':>5} | {'Loss':>10}"
        print(header)
        print("-" * len(header))

        for seed in range(N_SEEDS):
            np.random.seed(seed)

            # Generate preferences
            prefs = generate_preference_dataset(TRUE_WEIGHTS, n, noise=0.0)

            # Train reward model with optimizer tracking
            w_est, opt_result = train_reward_model(prefs, return_optimizer_info=True)

            # Compute metrics
            metrics = compute_metrics(w_est, TRUE_WEIGHTS)

            # Store result
            result = {
                'seed': seed,
                'n': n,
                'w_est': w_est,
                'w_norm': metrics['w_norm'],
                'mse': metrics['mse'],
                'rank_corr': metrics['rank_corr'],
                'top1': metrics['top1'],
                'converged': opt_result.success,
                'iterations': opt_result.nit,
                'loss': opt_result.fun,
            }
            all_results[n].append(result)

            # Print per-seed row
            w_str = ', '.join([f'{w:.2f}' for w in metrics['w_norm']])
            conv_str = 'Y' if opt_result.success else 'N'
            print(f"{seed:>4} | [{w_str:>38}] | {metrics['mse']:>8.4f} | {metrics['rank_corr']:>6.3f} | {metrics['top1']:>4} | {conv_str:>4} | {opt_result.nit:>5} | {opt_result.fun:>10.2f}")

        print()

        # Summary statistics for this N
        mses = [r['mse'] for r in all_results[n]]
        rhos = [r['rank_corr'] for r in all_results[n]]
        top1s = [r['top1'] for r in all_results[n]]
        n_converged = sum(r['converged'] for r in all_results[n])
        mean_iters = np.mean([r['iterations'] for r in all_results[n]])

        print(f"Summary (N={n}):")
        print_summary_statistics(mses, "MSE")
        print(f"  Rank Corr: mean={np.mean(rhos):.4f}, SE={np.std(rhos)/np.sqrt(N_SEEDS):.4f}")
        print(f"  Top-1 Acc: {sum(top1s)}/{N_SEEDS} seeds ({100*np.mean(top1s):.0f}%)")
        print(f"  Optimizer: {n_converged}/{N_SEEDS} converged, mean {mean_iters:.1f} iterations")
        print()

    # Theoretical variance validation (for N=500)
    print("=" * 70)
    print("THEORETICAL VARIANCE VALIDATION (N=500)")
    print("=" * 70)
    print()

    n_val = 500
    results_500 = all_results[n_val]

    # Empirical variance of w_est (unnormalized, since theory is for unnormalized)
    w_ests = np.array([r['w_est'][1:] for r in results_500])  # K-1 free parameters
    empirical_var = np.var(w_ests, axis=0, ddof=1)

    # Theoretical variance
    theoretical_cov = compute_theoretical_variance(TRUE_WEIGHTS, n_val)
    if theoretical_cov is not None:
        theoretical_var = np.diag(theoretical_cov)

        print("Fisher Information Matrix I(w*):")
        fisher = compute_fisher_information(TRUE_WEIGHTS)
        for i in range(N_OPTIONS-1):
            print(f"  [{', '.join([f'{fisher[i,j]:>7.4f}' for j in range(N_OPTIONS-1)])}]")
        print()

        print(f"{'Param':>6} | {'Emp Var':>10} | {'Theory Var':>10} | {'Ratio':>8} | {'In 95% CI':>10}")
        print("-" * 60)

        coverage_count = 0
        for j in range(N_OPTIONS-1):
            ratio = empirical_var[j] / theoretical_var[j] if theoretical_var[j] > 0 else np.nan

            # Check if true value falls in 95% CI for each seed
            true_val = TRUE_WEIGHTS[j+1] - TRUE_WEIGHTS[0]  # Relative to w[0]=0
            in_ci = 0
            for r in results_500:
                w_j = r['w_est'][j+1]
                se = np.sqrt(theoretical_var[j])
                if true_val - 1.96*se <= w_j <= true_val + 1.96*se:
                    in_ci += 1
            coverage = in_ci / N_SEEDS
            if coverage >= 0.85:  # Allow some slack from 0.95
                coverage_count += 1

            print(f"  w[{j+1}] | {empirical_var[j]:>10.6f} | {theoretical_var[j]:>10.6f} | {ratio:>8.3f} | {in_ci}/{N_SEEDS} ({100*coverage:.0f}%)")

        print()
        print(f"Coverage summary: {coverage_count}/{N_OPTIONS-1} parameters have adequate CI coverage")
    else:
        print("Fisher Information matrix is singular; cannot compute theoretical variance.")

    print()
    return all_results


# =============================================================================
# EXPERIMENT 2: DPO EQUIVALENCE
# =============================================================================

def run_experiment_2():
    """DPO equivalence experiment showing RM and DPO produce identical policies."""
    print("=" * 70)
    print("EXPERIMENT 2: DPO EQUIVALENCE")
    print("=" * 70)
    print()
    print("Hypothesis: With uniform reference policy, DPO and explicit RM are")
    print("mathematically equivalent, producing identical policies.")
    print()
    print("Configuration:")
    print(f"  lambda_KL = {LAMBDA_KL}")
    print(f"  N = 500 preference pairs")
    print()

    N_EQUIV = 500

    results = {
        'explicit_rm': [],
        'dpo': [],
        'mle': [],
    }

    # Per-seed results table
    header = f"{'Seed':>4} | {'Method':>12} | {'Regret':>8} | {'P(best)':>8} | {'Conv':>4} | {'Iters':>5} | {'Policy':>30}"
    print(header)
    print("-" * len(header))

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        prefs = generate_preference_dataset(TRUE_WEIGHTS, N_EQUIV, noise=0.0)

        # Method 1: Explicit Reward Model + Softmax Policy
        w_rm, opt_rm = train_reward_model(prefs, return_optimizer_info=True)
        pi_rm = logits_to_policy(w_rm / LAMBDA_KL)
        regret_rm = compute_regret(pi_rm, TRUE_WEIGHTS)

        # Method 2: DPO (direct policy optimization)
        theta_dpo, opt_dpo = train_dpo_policy(prefs, lambda_kl=LAMBDA_KL, return_optimizer_info=True)
        pi_dpo = logits_to_policy(theta_dpo)
        regret_dpo = compute_regret(pi_dpo, TRUE_WEIGHTS)
        w_dpo = LAMBDA_KL * theta_dpo  # Implied reward

        # Method 3: MLE (same as explicit RM, for verification)
        w_mle, opt_mle = train_reward_model(prefs, return_optimizer_info=True)
        pi_mle = logits_to_policy(w_mle)
        regret_mle = compute_regret(pi_mle, TRUE_WEIGHTS)

        # Store results
        results['explicit_rm'].append({
            'seed': seed, 'weights': w_rm, 'policy': pi_rm, 'regret': regret_rm,
            'converged': opt_rm.success, 'iterations': opt_rm.nit
        })
        results['dpo'].append({
            'seed': seed, 'weights': w_dpo, 'policy': pi_dpo, 'regret': regret_dpo,
            'converged': opt_dpo.success, 'iterations': opt_dpo.nit
        })
        results['mle'].append({
            'seed': seed, 'weights': w_mle, 'policy': pi_mle, 'regret': regret_mle,
            'converged': opt_mle.success, 'iterations': opt_mle.nit
        })

        # Print for this seed (only RM and DPO to show equivalence)
        for method, data in [('Explicit RM', results['explicit_rm'][-1]),
                              ('DPO', results['dpo'][-1])]:
            pi_str = ', '.join([f'{p:.3f}' for p in data['policy']])
            conv_str = 'Y' if data['converged'] else 'N'
            print(f"{seed:>4} | {method:>12} | {data['regret']:>8.4f} | {data['policy'][4]:>8.4f} | {conv_str:>4} | {data['iterations']:>5} | [{pi_str}]")

    print()

    # Summary comparison
    print("Summary: Policy Comparison (N=500)")
    print("-" * 70)
    print(f"{'Method':>12} | {'Regret':>15} | {'P(Best Option)':>15} | {'Converged':>10}")
    print("-" * 70)

    for method in ['explicit_rm', 'dpo', 'mle']:
        regrets = [r['regret'] for r in results[method]]
        pi_best = [r['policy'][4] for r in results[method]]
        n_conv = sum(r['converged'] for r in results[method])

        regret_str = f"{np.mean(regrets):.4f} +/- {np.std(regrets)/np.sqrt(N_SEEDS):.4f}"
        pi_str = f"{np.mean(pi_best):.4f} +/- {np.std(pi_best)/np.sqrt(N_SEEDS):.4f}"

        print(f"{method:>12} | {regret_str:>15} | {pi_str:>15} | {n_conv}/{N_SEEDS}")

    print()

    # DPO-RM equivalence check
    print("DPO-RM Equivalence Check:")
    print("-" * 70)

    # Compare policies seed by seed
    policy_diffs = []
    for i in range(N_SEEDS):
        pi_rm = results['explicit_rm'][i]['policy']
        pi_dpo = results['dpo'][i]['policy']
        diff = np.max(np.abs(pi_rm - pi_dpo))
        policy_diffs.append(diff)

    print(f"  Max policy difference across all seeds: {np.max(policy_diffs):.6f}")
    print(f"  Mean policy difference: {np.mean(policy_diffs):.6f}")
    print(f"  Policies match to <0.001: {sum(d < 0.001 for d in policy_diffs)}/{N_SEEDS} seeds")
    print()

    return results


# =============================================================================
# EXPERIMENT 3: NOISE ROBUSTNESS
# =============================================================================

def run_experiment_3():
    """Noise robustness experiment."""
    print("=" * 70)
    print("EXPERIMENT 3: NOISE ROBUSTNESS")
    print("=" * 70)
    print()
    print("Hypothesis: Weight recovery degrades gracefully under label noise,")
    print("remaining usable up to noise level ~0.2.")
    print()
    print("Configuration:")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  N = 500 preference pairs")
    print()

    N_NOISE = 500

    all_results = {eps: [] for eps in NOISE_LEVELS}

    for eps in NOISE_LEVELS:
        print(f"--- Noise = {eps:.2f} ---")

        for seed in range(N_SEEDS):
            np.random.seed(seed)
            prefs = generate_preference_dataset(TRUE_WEIGHTS, N_NOISE, noise=eps)

            w_est, opt_result = train_reward_model(prefs, return_optimizer_info=True)
            metrics = compute_metrics(w_est, TRUE_WEIGHTS)

            pi = logits_to_policy(w_est)
            regret = compute_regret(pi, TRUE_WEIGHTS)

            all_results[eps].append({
                'seed': seed,
                'noise': eps,
                'mse': metrics['mse'],
                'rank_corr': metrics['rank_corr'],
                'top1': metrics['top1'],
                'regret': regret,
                'converged': opt_result.success,
                'iterations': opt_result.nit,
            })

        # Summary for this noise level
        mses = [r['mse'] for r in all_results[eps]]
        rhos = [r['rank_corr'] for r in all_results[eps]]
        top1s = [r['top1'] for r in all_results[eps]]
        regrets = [r['regret'] for r in all_results[eps]]

        print(f"  MSE: {np.mean(mses):.4f} +/- {np.std(mses)/np.sqrt(N_SEEDS):.4f}")
        print(f"  Rank Corr: {np.mean(rhos):.4f} +/- {np.std(rhos)/np.sqrt(N_SEEDS):.4f}")
        print(f"  Top-1: {sum(top1s)}/{N_SEEDS} ({100*np.mean(top1s):.0f}%)")
        print(f"  Regret: {np.mean(regrets):.4f} +/- {np.std(regrets)/np.sqrt(N_SEEDS):.4f}")
        print()

    # Comprehensive summary table
    print("Noise Robustness Summary Table:")
    print("-" * 80)
    print(f"{'Noise':>6} | {'MSE':>15} | {'Rank Corr':>15} | {'Top-1':>10} | {'Regret':>15}")
    print("-" * 80)

    for eps in NOISE_LEVELS:
        mses = [r['mse'] for r in all_results[eps]]
        rhos = [r['rank_corr'] for r in all_results[eps]]
        top1s = [r['top1'] for r in all_results[eps]]
        regrets = [r['regret'] for r in all_results[eps]]

        mse_str = f"{np.mean(mses):.4f} +/- {np.std(mses)/np.sqrt(N_SEEDS):.4f}"
        rho_str = f"{np.mean(rhos):.4f} +/- {np.std(rhos)/np.sqrt(N_SEEDS):.4f}"
        top1_str = f"{sum(top1s)}/{N_SEEDS}"
        regret_str = f"{np.mean(regrets):.4f} +/- {np.std(regrets)/np.sqrt(N_SEEDS):.4f}"

        print(f"{eps:>6.2f} | {mse_str:>15} | {rho_str:>15} | {top1_str:>10} | {regret_str:>15}")

    print()
    return all_results


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figures(recovery_results, equiv_results, noise_results):
    """Generate all publication-quality figures."""
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    print()

    output_dir = CONFIG['output_dir']
    dpi = CONFIG['figure_dpi']

    # Figure 1: Weight Recovery (MSE vs N, multiple noise levels)
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    for eps in [0.0, 0.1, 0.2]:
        mse_means = []
        mse_ses = []
        for n in SAMPLE_SIZES:
            mses = []
            for seed in range(N_SEEDS):
                np.random.seed(seed)
                prefs = generate_preference_dataset(TRUE_WEIGHTS, n, noise=eps)
                w_est = train_reward_model(prefs)
                metrics = compute_metrics(w_est, TRUE_WEIGHTS)
                mses.append(metrics['mse'])
            mse_means.append(np.mean(mses))
            mse_ses.append(np.std(mses) / np.sqrt(N_SEEDS))

        label = f'noise = {eps:.0%}' if eps > 0 else 'no noise'
        ax1.errorbar(SAMPLE_SIZES, mse_means, yerr=mse_ses, marker='o',
                     capsize=3, label=label)

    # Add 1/N reference line
    n_ref = np.array(SAMPLE_SIZES)
    mse_ref = 0.5 / n_ref
    ax1.plot(n_ref, mse_ref, 'k--', alpha=0.5, label=r'$O(1/N)$ reference')

    ax1.set_xlabel('Number of Preference Pairs (N)', fontsize=11)
    ax1.set_ylabel('Weight MSE', fontsize=11)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Weight Recovery: MSE vs Sample Size', fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f'{output_dir}/preference_weight_recovery.png', dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_dir}/preference_weight_recovery.png")

    # Figure 2: Policy Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    options = np.arange(N_OPTIONS)
    width = 0.2

    pi_true = logits_to_policy(TRUE_WEIGHTS / 0.1)  # Sharp distribution
    pi_rm = np.mean([r['policy'] for r in equiv_results['explicit_rm']], axis=0)
    pi_dpo = np.mean([r['policy'] for r in equiv_results['dpo']], axis=0)
    pi_uniform = np.ones(N_OPTIONS) / N_OPTIONS

    ax2.bar(options - 1.5*width, pi_true, width, label='Optimal (greedy on true)', color='C0', alpha=0.8)
    ax2.bar(options - 0.5*width, pi_rm, width, label='Explicit RM', color='C1', alpha=0.8)
    ax2.bar(options + 0.5*width, pi_dpo, width, label='DPO', color='C2', alpha=0.8)
    ax2.bar(options + 1.5*width, pi_uniform, width, label='Uniform', color='gray', alpha=0.5)

    ax2.set_xlabel('Option', fontsize=11)
    ax2.set_ylabel('Policy Probability', fontsize=11)
    ax2.set_xticks(options)
    ax2.set_xticklabels([f'$a_{i+1}$\n($u$={TRUE_WEIGHTS[i]:.1f})' for i in range(N_OPTIONS)])
    ax2.legend(fontsize=10)
    ax2.set_title('Policy Distributions by Method (N=500)', fontsize=12)
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(f'{output_dir}/preference_policy_comparison.png', dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_dir}/preference_policy_comparison.png")

    # Figure 3: Noise Robustness
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    mse_means = [np.mean([r['mse'] for r in noise_results[eps]]) for eps in NOISE_LEVELS]
    mse_ses = [np.std([r['mse'] for r in noise_results[eps]]) / np.sqrt(N_SEEDS) for eps in NOISE_LEVELS]
    ax3a.errorbar(NOISE_LEVELS, mse_means, yerr=mse_ses, marker='s', capsize=3, color='C0')
    ax3a.set_xlabel('Label Noise Probability', fontsize=11)
    ax3a.set_ylabel('Weight MSE', fontsize=11)
    ax3a.set_title('Weight Recovery Error vs Noise', fontsize=12)
    ax3a.grid(True, alpha=0.3)

    regret_means = [np.mean([r['regret'] for r in noise_results[eps]]) for eps in NOISE_LEVELS]
    regret_ses = [np.std([r['regret'] for r in noise_results[eps]]) / np.sqrt(N_SEEDS) for eps in NOISE_LEVELS]
    ax3b.errorbar(NOISE_LEVELS, regret_means, yerr=regret_ses, marker='s', capsize=3, color='C1')
    ax3b.set_xlabel('Label Noise Probability', fontsize=11)
    ax3b.set_ylabel('Policy Regret', fontsize=11)
    ax3b.set_title('Policy Regret vs Noise', fontsize=12)
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(f'{output_dir}/preference_noise_robustness.png', dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_dir}/preference_noise_robustness.png")

    plt.close('all')
    print()


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(equiv_results):
    """Generate LaTeX table with booktabs formatting."""
    print("=" * 70)
    print("GENERATING LATEX TABLE")
    print("=" * 70)
    print()

    output_dir = CONFIG['output_dir']

    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Weight MSE & Rank Correlation & Top-1 Accuracy & Regret \\",
        r"\midrule",
    ]

    for method, name in [('explicit_rm', 'Explicit RM'), ('dpo', 'DPO'), ('mle', 'MLE')]:
        weights = [r['weights'] for r in equiv_results[method]]
        regrets = [r['regret'] for r in equiv_results[method]]

        mses = [compute_metrics(w, TRUE_WEIGHTS)['mse'] for w in weights]
        rhos = [compute_metrics(w, TRUE_WEIGHTS)['rank_corr'] for w in weights]
        top1s = [compute_metrics(w, TRUE_WEIGHTS)['top1'] for w in weights]

        mse_str = f"{np.mean(mses):.4f} $\\pm$ {np.std(mses)/np.sqrt(N_SEEDS):.4f}"
        rho_str = f"{np.mean(rhos):.4f} $\\pm$ {np.std(rhos)/np.sqrt(N_SEEDS):.4f}"
        top1_str = f"{np.mean(top1s):.2f} $\\pm$ {np.std(top1s)/np.sqrt(N_SEEDS):.2f}"
        regret_str = f"{np.mean(regrets):.4f} $\\pm$ {np.std(regrets)/np.sqrt(N_SEEDS):.4f}"

        latex_lines.append(f"{name} & {mse_str} & {rho_str} & {top1_str} & {regret_str} \\\\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    latex_table = '\n'.join(latex_lines)

    with open(f'{output_dir}/preference_results.tex', 'w') as f:
        f.write(latex_table)

    print(f"  Saved: {output_dir}/preference_results.tex")
    print()
    print("Table contents:")
    print(latex_table)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print_header()

    # Run experiments
    recovery_results = run_experiment_1()
    equiv_results = run_experiment_2()
    noise_results = run_experiment_3()

    # Generate outputs
    generate_figures(recovery_results, equiv_results, noise_results)
    generate_latex_table(equiv_results)

    # Final summary
    print("=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)
    print()

    # Check 1: MSE decreases with N
    mse_50 = np.mean([r['mse'] for r in recovery_results[50]])
    mse_1000 = np.mean([r['mse'] for r in recovery_results[1000]])
    print(f"1. MSE decreases with N:")
    print(f"   MSE(N=50)   = {mse_50:.4f}")
    print(f"   MSE(N=1000) = {mse_1000:.4f}")
    print(f"   Ratio: {mse_50/mse_1000:.1f}x improvement (expected ~20x for O(1/N))")
    print(f"   CHECK: {'PASS' if mse_50 > mse_1000 else 'FAIL'}")
    print()

    # Check 2: DPO and RM policies match
    policy_diffs = []
    for i in range(N_SEEDS):
        pi_rm = equiv_results['explicit_rm'][i]['policy']
        pi_dpo = equiv_results['dpo'][i]['policy']
        diff = np.max(np.abs(pi_rm - pi_dpo))
        policy_diffs.append(diff)
    max_diff = np.max(policy_diffs)
    print(f"2. DPO-RM policy equivalence:")
    print(f"   Max policy difference: {max_diff:.6f}")
    print(f"   CHECK: {'PASS' if max_diff < 0.001 else 'FAIL'} (threshold: 0.001)")
    print()

    # Check 3: Rank correlation >= 0.9 for N >= 500
    rhos_500 = [r['rank_corr'] for r in recovery_results[500]]
    mean_rho = np.mean(rhos_500)
    print(f"3. Rank correlation for N=500:")
    print(f"   Mean rank correlation: {mean_rho:.4f}")
    print(f"   CHECK: {'PASS' if mean_rho >= 0.9 else 'FAIL'} (threshold: 0.9)")
    print()

    # Output files
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"  {CONFIG['output_dir']}/preference_weight_recovery.png")
    print(f"  {CONFIG['output_dir']}/preference_policy_comparison.png")
    print(f"  {CONFIG['output_dir']}/preference_noise_robustness.png")
    print(f"  {CONFIG['output_dir']}/preference_results.tex")


if __name__ == '__main__':
    main()
