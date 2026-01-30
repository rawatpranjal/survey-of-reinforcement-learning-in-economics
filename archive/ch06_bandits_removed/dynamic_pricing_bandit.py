# Dynamic Pricing Bandit — Chapter 6, Economic Bandits.
# Unimodal demand structure yields O(log T) vs O(sqrt(T)) regret.

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 20
T = 50_000
N_SEEDS = 30
PRICES = np.linspace(0.5, 10.0, K)

# Logistic demand: theta(p) = 1 / (1 + exp(1.5*(p - 5)))
def theta(p):
    return 1.0 / (1.0 + np.exp(1.5 * (p - 5.0)))

# Expected revenue r(p) = p * theta(p)
EXPECTED_REVENUE = PRICES * theta(PRICES)
OPTIMAL_ARM = np.argmax(EXPECTED_REVENUE)
OPTIMAL_PRICE = PRICES[OPTIMAL_ARM]
OPTIMAL_REVENUE = EXPECTED_REVENUE[OPTIMAL_ARM]

print(f"Optimal arm: {OPTIMAL_ARM}, price: {OPTIMAL_PRICE:.4f}, "
      f"E[revenue]: {OPTIMAL_REVENUE:.4f}")

# Theoretical overlay constants
C1 = 1.0   # agnostic bound coefficient
C2 = 50.0  # structural bound coefficient (scaled for visibility)

# Output directory (same as script location)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Storage for results across seeds
# ---------------------------------------------------------------------------
cumregret_all = {alg: np.zeros((N_SEEDS, T)) for alg in
                 ["eps-greedy", "UCB1", "UUCB"]}
arm_counts_all = {alg: np.zeros((N_SEEDS, K)) for alg in
                  ["eps-greedy", "UCB1", "UUCB"]}
total_revenue_all = {alg: np.zeros(N_SEEDS) for alg in
                     ["eps-greedy", "UCB1", "UUCB"]}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
for seed in range(N_SEEDS):
    np.random.seed(seed)
    # Pre-generate uniform draws for demand realization.
    # For a given price choice, sale = 1 if U < theta(price).
    U = np.random.uniform(size=T)

    for alg in ["eps-greedy", "UCB1", "UUCB"]:
        counts = np.zeros(K)
        sum_rewards = np.zeros(K)
        means = np.zeros(K)
        cum_regret = np.zeros(T)
        total_rev = 0.0

        # For UUCB: leader index
        leader = 0

        for t in range(T):
            # --- Arm selection ---
            if alg == "eps-greedy":
                eps_t = min(1.0, 5.0 * K / (t + 1))
                if np.random.uniform() < eps_t:
                    arm = np.random.randint(K)
                else:
                    arm = np.argmax(means)

            elif alg == "UCB1":
                # Initialize: play each arm once
                if t < K:
                    arm = t
                else:
                    ucb_indices = means + np.sqrt(2.0 * np.log(t) /
                                                  np.maximum(counts, 1))
                    arm = np.argmax(ucb_indices)

            elif alg == "UUCB":
                # Initialize: play each arm once
                if t < K:
                    arm = t
                else:
                    if t == K:
                        # Set initial leader after initialization
                        leader = np.argmax(means)

                    # Check neighbors
                    best_neighbor = None
                    best_neighbor_ucb = -np.inf
                    for nb in [leader - 1, leader + 1]:
                        if 0 <= nb < K:
                            nb_ucb = means[nb] + np.sqrt(
                                2.0 * np.log(t) / np.maximum(counts[nb], 1))
                            if nb_ucb > best_neighbor_ucb:
                                best_neighbor_ucb = nb_ucb
                                best_neighbor = nb

                    leader_ucb = means[leader] + np.sqrt(
                        2.0 * np.log(t) / np.maximum(counts[leader], 1))

                    if best_neighbor is not None and best_neighbor_ucb > leader_ucb:
                        leader = best_neighbor
                        arm = best_neighbor
                    else:
                        arm = leader

            # --- Observe reward ---
            p = PRICES[arm]
            sale = 1.0 if U[t] < theta(p) else 0.0
            reward = p * sale

            # Update statistics
            counts[arm] += 1
            sum_rewards[arm] += reward
            means[arm] = sum_rewards[arm] / counts[arm]

            # Track regret and revenue
            instant_regret = OPTIMAL_REVENUE - EXPECTED_REVENUE[arm]
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0.0) + instant_regret
            total_rev += reward

        cumregret_all[alg][seed] = cum_regret
        arm_counts_all[alg][seed] = counts
        total_revenue_all[alg][seed] = total_rev

    if (seed + 1) % 10 == 0:
        print(f"  Completed seed {seed + 1}/{N_SEEDS}")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
t_axis = np.arange(1, T + 1)

regret_mean = {}
regret_se = {}
for alg in ["eps-greedy", "UCB1", "UUCB"]:
    regret_mean[alg] = cumregret_all[alg].mean(axis=0)
    regret_se[alg] = cumregret_all[alg].std(axis=0) / np.sqrt(N_SEEDS)

# Arm fractions
arm_frac_mean = {}
arm_frac_se = {}
for alg in ["eps-greedy", "UCB1", "UUCB"]:
    fracs = arm_counts_all[alg] / T
    arm_frac_mean[alg] = fracs.mean(axis=0)
    arm_frac_se[alg] = fracs.std(axis=0) / np.sqrt(N_SEEDS)

# Final regret, fraction on optimal, total revenue
final_regret_mean = {}
final_regret_se = {}
opt_frac_mean = {}
opt_frac_se = {}
revenue_mean = {}
revenue_se = {}
for alg in ["eps-greedy", "UCB1", "UUCB"]:
    fr = cumregret_all[alg][:, -1]
    final_regret_mean[alg] = fr.mean()
    final_regret_se[alg] = fr.std() / np.sqrt(N_SEEDS)

    of = arm_counts_all[alg][:, OPTIMAL_ARM] / T
    opt_frac_mean[alg] = of.mean()
    opt_frac_se[alg] = of.std() / np.sqrt(N_SEEDS)

    rv = total_revenue_all[alg]
    revenue_mean[alg] = rv.mean()
    revenue_se[alg] = rv.std() / np.sqrt(N_SEEDS)

# Theoretical bounds
agnostic_bound = C1 * np.sqrt(K * t_axis)
structural_bound = C2 * np.log(t_axis + 1)

# ---------------------------------------------------------------------------
# Figure: 2-panel (cumulative regret + arm histogram)
# ---------------------------------------------------------------------------
colors = {"eps-greedy": "#1f77b4", "UCB1": "#ff7f0e", "UUCB": "#2ca02c"}
labels = {"eps-greedy": r"$\varepsilon$-greedy", "UCB1": "UCB1",
          "UUCB": "Unimodal UCB"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: cumulative regret ---
for alg in ["eps-greedy", "UCB1", "UUCB"]:
    m = regret_mean[alg]
    se = regret_se[alg]
    ax1.plot(t_axis, m, color=colors[alg], label=labels[alg], linewidth=1.5)
    ax1.fill_between(t_axis, m - se, m + se, color=colors[alg], alpha=0.15)

ax1.plot(t_axis, agnostic_bound, 'k--', linewidth=1.0, alpha=0.6,
         label=r"Agnostic $O(\sqrt{Kt})$")
ax1.plot(t_axis, structural_bound, 'k:', linewidth=1.0, alpha=0.6,
         label=r"Structural $O(\log t)$")

ax1.set_xlabel("Round $t$")
ax1.set_ylabel("Cumulative Regret")
ax1.set_title("Cumulative Regret")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_xlim(1, T)
ax1.set_ylim(bottom=0)

# --- Right panel: arm selection histogram ---
bar_width = 0.25
x_pos = np.arange(K)
for i, alg in enumerate(["eps-greedy", "UCB1", "UUCB"]):
    ax2.bar(x_pos + i * bar_width, arm_frac_mean[alg], bar_width,
            color=colors[alg], label=labels[alg], alpha=0.85)

ax2.axvline(x=OPTIMAL_ARM + 1.0 * bar_width, color='red', linestyle='--',
            linewidth=1.5, label=f"Optimal price ({OPTIMAL_PRICE:.2f})")
ax2.set_xlabel("Price")
ax2.set_ylabel("Fraction of Pulls")
ax2.set_title("Price Selection Distribution")
ax2.set_xticks(x_pos + bar_width)
ax2.set_xticklabels([f"{p:.1f}" for p in PRICES], rotation=45, fontsize=7)
ax2.legend(fontsize=8)

fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "dynamic_pricing_regret.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {fig_path}")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
alg_display = {"eps-greedy": r"$\varepsilon$-greedy",
               "UCB1": "UCB1",
               "UUCB": "Unimodal UCB"}

lines = []
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\hline")
lines.append(r"Algorithm & Final Regret & Fraction on Optimal & "
             r"Total Revenue \\")
lines.append(r"\hline")
for alg in ["eps-greedy", "UCB1", "UUCB"]:
    fr = f"{final_regret_mean[alg]:.1f} $\\pm$ {final_regret_se[alg]:.1f}"
    of = f"{opt_frac_mean[alg]:.3f} $\\pm$ {opt_frac_se[alg]:.3f}"
    rv = f"{revenue_mean[alg]:.1f} $\\pm$ {revenue_se[alg]:.1f}"
    lines.append(f"{alg_display[alg]} & {fr} & {of} & {rv} \\\\")
lines.append(r"\hline")
lines.append(r"\end{tabular}")

tex_path = os.path.join(OUT_DIR, "dynamic_pricing_results.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved table: {tex_path}")
