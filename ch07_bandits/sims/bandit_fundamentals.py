# Bandit Fundamentals — Chapter 6, Economic Bandits.
# Baseline comparison of ε-greedy, UCB1, and Thompson Sampling on a K-armed Gaussian bandit.

import os
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────

K = 10                  # number of arms
T = 10_000              # horizon
N_SEEDS = 30            # Monte Carlo repetitions (seeds 0..29)
mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0])
mu_star = mu.max()      # optimal mean (arm 9, μ*=2.0)
optimal_arm = np.argmax(mu)
Delta = mu_star - mu    # sub-optimality gaps

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Algorithm implementations ───────────────────────────────────────────────

def run_epsilon_greedy(rewards):
    """Decaying ε-greedy: ε_t = min(1, 5K / (t+1))."""
    counts = np.zeros(K)
    mu_hat = np.zeros(K)
    actions = np.zeros(T, dtype=int)

    for t in range(T):
        eps_t = min(1.0, 5 * K / (t + 1))
        if np.random.rand() < eps_t:
            a = np.random.randint(K)
        else:
            a = np.argmax(mu_hat)
        actions[t] = a
        counts[a] += 1
        mu_hat[a] += (rewards[t, a] - mu_hat[a]) / counts[a]

    return actions


def run_ucb1(rewards):
    """UCB1: argmax_i [μ̂_i + sqrt(2 ln(t) / N_i)]. Play each arm once first."""
    counts = np.zeros(K)
    sum_rewards = np.zeros(K)
    actions = np.zeros(T, dtype=int)

    # initialization: play each arm once
    for t in range(K):
        a = t
        actions[t] = a
        counts[a] += 1
        sum_rewards[a] += rewards[t, a]

    for t in range(K, T):
        mu_hat = sum_rewards / counts
        ucb = mu_hat + np.sqrt(2 * np.log(t) / counts)
        a = np.argmax(ucb)
        actions[t] = a
        counts[a] += 1
        sum_rewards[a] += rewards[t, a]

    return actions


def run_thompson_sampling(rewards):
    """Thompson Sampling (Gaussian): posterior N(μ̂_i, 1/N_i). Sample and pick argmax."""
    counts = np.zeros(K)
    sum_rewards = np.zeros(K)
    actions = np.zeros(T, dtype=int)

    for t in range(T):
        samples = np.zeros(K)
        for i in range(K):
            if counts[i] == 0:
                # uninformative prior: sample from N(0, 1e6) effectively
                samples[i] = np.random.normal(0, 1e3)
            else:
                mu_hat_i = sum_rewards[i] / counts[i]
                sigma_i = 1.0 / np.sqrt(counts[i])
                samples[i] = np.random.normal(mu_hat_i, sigma_i)
        a = np.argmax(samples)
        actions[t] = a
        counts[a] += 1
        sum_rewards[a] += rewards[t, a]

    return actions


# ─── Monte Carlo runs ────────────────────────────────────────────────────────

algorithms = {
    r"$\varepsilon$-greedy (decaying)": run_epsilon_greedy,
    "UCB1": run_ucb1,
    "Thompson Sampling": run_thompson_sampling,
}

# Storage: (N_SEEDS, T) cumulative regret for each algorithm
cum_regret = {name: np.zeros((N_SEEDS, T)) for name in algorithms}
optimal_pulls = {name: np.zeros(N_SEEDS) for name in algorithms}

for seed in range(N_SEEDS):
    np.random.seed(seed)
    # Pre-generate reward matrix so all algorithms face identical draws
    rewards = np.random.randn(T, K) + mu[np.newaxis, :]  # shape (T, K)

    for name, algo_fn in algorithms.items():
        np.random.seed(seed + 1000)  # separate seed for algorithm randomness
        # Offset by 1000 so algorithm randomness is independent of reward draws
        # but reproducible across runs
        actions = algo_fn(rewards)

        # Instantaneous regret
        inst_regret = mu_star - mu[actions]
        cum_regret[name][seed] = np.cumsum(inst_regret)

        # Fraction of optimal arm pulls
        optimal_pulls[name][seed] = np.mean(actions == optimal_arm)

# ─── Theoretical bounds ──────────────────────────────────────────────────────

t_range = np.arange(1, T + 1)

# Lai-Robbins instance-dependent lower bound: sum_{i: Delta_i > 0} (Delta_i / KL_i) * ln(t)
# For Gaussian with unit variance, KL(mu_i, mu_star) = 0.5 * Delta_i^2
lai_robbins_coeff = 0.0
for i in range(K):
    if Delta[i] > 0:
        KL_i = 0.5 * Delta[i] ** 2
        lai_robbins_coeff += Delta[i] / KL_i  # = 2 / Delta_i

lai_robbins = lai_robbins_coeff * np.log(t_range)

# Minimax bound: c * sqrt(K * t)
c_minimax = 0.5
minimax = c_minimax * np.sqrt(K * t_range)

# ─── Plot: cumulative regret ─────────────────────────────────────────────────

colors = {"$\\varepsilon$-greedy (decaying)": "#d62728",
          "UCB1": "#1f77b4",
          "Thompson Sampling": "#2ca02c"}

fig, ax = plt.subplots(figsize=(8, 5))

for name in algorithms:
    mean_regret = cum_regret[name].mean(axis=0)
    se_regret = cum_regret[name].std(axis=0) / np.sqrt(N_SEEDS)
    ax.plot(t_range, mean_regret, label=name, color=colors[name], linewidth=1.5)
    ax.fill_between(t_range, mean_regret - se_regret, mean_regret + se_regret,
                     color=colors[name], alpha=0.15)

# Theoretical overlays
ax.plot(t_range, lai_robbins, 'k--', linewidth=1.0, alpha=0.7,
        label=r"Lai--Robbins $\sum \frac{\Delta_i}{\mathrm{KL}_i} \ln t$")
ax.plot(t_range, minimax, 'k:', linewidth=1.0, alpha=0.7,
        label=r"Minimax $0.5\sqrt{Kt}$")

ax.set_xlabel("Round $t$", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title("Cumulative Regret: $K$=10 Gaussian Bandit", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(1, T)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

fig_path = os.path.join(OUT_DIR, "bandit_fundamentals_regret.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {fig_path}")

# ─── LaTeX table: summary results ────────────────────────────────────────────

tex_lines = []
tex_lines.append(r"\begin{tabular}{lcc}")
tex_lines.append(r"\hline")
tex_lines.append(r"Algorithm & Final Regret (mean $\pm$ SE) & \% Optimal Arm Pulls \\")
tex_lines.append(r"\hline")

for name in algorithms:
    final_regrets = cum_regret[name][:, -1]
    mean_reg = final_regrets.mean()
    se_reg = final_regrets.std() / np.sqrt(N_SEEDS)
    mean_opt = optimal_pulls[name].mean() * 100
    # Escape special LaTeX characters in algorithm name for table
    table_name = name.replace("$\\varepsilon$", r"$\varepsilon$")
    tex_lines.append(f"{table_name} & {mean_reg:.1f} $\\pm$ {se_reg:.1f} & {mean_opt:.1f}\\% \\\\")

tex_lines.append(r"\hline")
tex_lines.append(r"\end{tabular}")

tex_content = "\n".join(tex_lines)
tex_path = os.path.join(OUT_DIR, "bandit_fundamentals_results.tex")
with open(tex_path, "w") as f:
    f.write(tex_content)
print(f"Saved table: {tex_path}")
