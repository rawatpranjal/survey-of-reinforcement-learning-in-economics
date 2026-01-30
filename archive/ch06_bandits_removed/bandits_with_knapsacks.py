# Bandits with Knapsacks — Chapter 6, Economic Bandits.
# Budget-constrained bandits with primal-dual approach vs unconstrained UCB.

import os
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 10
bids = np.linspace(0.5, 5.0, K)
T = 10_000
B_total = 0.3 * T  # budget
N_SEEDS = 30
COMPETING_BID_RATE = 0.5  # Exp(rate=0.5), mean=2.0
CLICK_PROB = 0.3
MC_SAMPLES = 100_000
ETA = 1.0 / np.sqrt(T)  # dual learning rate

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Compute expected reward and cost per bid level via Monte Carlo
# ---------------------------------------------------------------------------
np.random.seed(999)
mc_competing = np.random.exponential(1.0 / COMPETING_BID_RATE, size=MC_SAMPLES)
mc_clicks = np.random.binomial(1, CLICK_PROB, size=MC_SAMPLES)

E_reward = np.zeros(K)
E_cost = np.zeros(K)
for i, b in enumerate(bids):
    wins = mc_competing <= b
    E_reward[i] = np.mean(wins * mc_clicks)
    E_cost[i] = np.mean(wins * mc_competing)  # second-price: pay competing bid

# ---------------------------------------------------------------------------
# LP Relaxation Benchmark
# ---------------------------------------------------------------------------
# max  sum_i x_i * E[reward_i]
# s.t. sum_i x_i * E[cost_i] <= B/T
#      sum_i x_i = 1
#      x_i >= 0
#
# linprog minimises, so negate the objective.
c_lp = -E_reward
A_ub = [E_cost]
b_ub = [B_total / T]
A_eq = [np.ones(K)]
b_eq = [1.0]
bounds = [(0, 1) for _ in range(K)]

res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=bounds, method='highs')
lp_reward_per_round = -res.fun
lp_benchmark = lp_reward_per_round * T

print(f"LP relaxation benchmark: {lp_benchmark:.2f} (per-round: {lp_reward_per_round:.4f})")
print(f"LP solution x*: {np.round(res.x, 4)}")
print(f"E[reward]: {np.round(E_reward, 4)}")
print(f"E[cost]:   {np.round(E_cost, 4)}")

# ---------------------------------------------------------------------------
# Pre-generate random outcomes for all seeds
# ---------------------------------------------------------------------------
all_competing_bids = np.zeros((N_SEEDS, T))
all_clicks = np.zeros((N_SEEDS, T, K))

for seed in range(N_SEEDS):
    np.random.seed(seed)
    all_competing_bids[seed] = np.random.exponential(1.0 / COMPETING_BID_RATE, size=T)
    all_clicks[seed] = np.random.binomial(1, CLICK_PROB, size=(T, K))


# ---------------------------------------------------------------------------
# Helper: evaluate a single round
# ---------------------------------------------------------------------------
def play_round(bid_idx, competing_bid, clicks_row):
    """Returns (reward, cost) for choosing bid_idx."""
    b = bids[bid_idx]
    if b >= competing_bid:
        reward = float(clicks_row[bid_idx])
        cost = competing_bid  # second-price
    else:
        reward = 0.0
        cost = 0.0
    return reward, cost


# ---------------------------------------------------------------------------
# Algorithm 1: UCB1-no-budget
# ---------------------------------------------------------------------------
def run_ucb1_no_budget(competing_bids, clicks):
    cum_reward = np.zeros(T)
    cum_cost = np.zeros(T)

    counts = np.zeros(K)
    sum_rewards = np.zeros(K)
    budget_remaining = B_total
    total_reward = 0.0
    total_cost = 0.0

    for t in range(T):
        if budget_remaining <= 0:
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        # UCB1 arm selection
        if t < K:
            arm = t  # play each arm once
        else:
            ucb_values = sum_rewards / counts + np.sqrt(2 * np.log(t) / counts)
            arm = np.argmax(ucb_values)

        r, c = play_round(arm, competing_bids[t], clicks[t])

        # Check if cost exceeds remaining budget
        if c > budget_remaining:
            # Can't afford this; budget exhausted
            budget_remaining = 0
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        total_reward += r
        total_cost += c
        budget_remaining -= c
        counts[arm] += 1
        sum_rewards[arm] += r

        cum_reward[t] = total_reward
        cum_cost[t] = total_cost

    return cum_reward, cum_cost


# ---------------------------------------------------------------------------
# Algorithm 2: Proportional Pacing
# ---------------------------------------------------------------------------
def run_proportional_pacing(competing_bids, clicks, rng):
    cum_reward = np.zeros(T)
    cum_cost = np.zeros(T)

    counts = np.zeros(K)
    sum_rewards = np.zeros(K)
    budget_remaining = B_total
    total_reward = 0.0
    total_cost = 0.0

    for t in range(T):
        if budget_remaining <= 0:
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        # Pacing: play with probability proportional to remaining budget
        f_t = budget_remaining / B_total
        if rng.random() > f_t:
            # Skip this round
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        # UCB1 arm selection
        if np.min(counts) == 0:
            arm = np.argmin(counts)
        else:
            ucb_values = sum_rewards / counts + np.sqrt(2 * np.log(t + 1) / counts)
            arm = np.argmax(ucb_values)

        r, c = play_round(arm, competing_bids[t], clicks[t])

        if c > budget_remaining:
            budget_remaining = 0
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        total_reward += r
        total_cost += c
        budget_remaining -= c
        counts[arm] += 1
        sum_rewards[arm] += r

        cum_reward[t] = total_reward
        cum_cost[t] = total_cost

    return cum_reward, cum_cost


# ---------------------------------------------------------------------------
# Algorithm 3: BwK Primal-Dual
# ---------------------------------------------------------------------------
def run_bwk_primal_dual(competing_bids, clicks):
    cum_reward = np.zeros(T)
    cum_cost = np.zeros(T)

    counts = np.zeros(K)
    sum_rewards = np.zeros(K)
    sum_costs = np.zeros(K)
    budget_remaining = B_total
    total_reward = 0.0
    total_cost = 0.0
    lam = 0.0  # dual variable
    budget_per_round = B_total / T

    for t in range(T):
        if budget_remaining <= 0:
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        # Arm selection: argmax_i [mu_hat_i - lambda * c_hat_i + exploration bonus]
        if np.min(counts) == 0:
            arm = np.argmin(counts)
        else:
            mu_hat = sum_rewards / counts
            c_hat = sum_costs / counts
            bonus = np.sqrt(2 * np.log(t + 1) / counts)
            index = mu_hat - lam * c_hat + bonus
            arm = np.argmax(index)

        r, c = play_round(arm, competing_bids[t], clicks[t])

        if c > budget_remaining:
            budget_remaining = 0
            cum_reward[t] = total_reward
            cum_cost[t] = total_cost
            continue

        total_reward += r
        total_cost += c
        budget_remaining -= c
        counts[arm] += 1
        sum_rewards[arm] += r
        sum_costs[arm] += c

        # Dual update
        lam = max(0.0, lam + ETA * (c - budget_per_round))

        cum_reward[t] = total_reward
        cum_cost[t] = total_cost

    return cum_reward, cum_cost


# ---------------------------------------------------------------------------
# Run all algorithms across seeds
# ---------------------------------------------------------------------------
algo_names = ["UCB1-no-budget", "Proportional Pacing", "BwK Primal-Dual"]
n_algos = len(algo_names)

all_cum_rewards = np.zeros((n_algos, N_SEEDS, T))
all_cum_costs = np.zeros((n_algos, N_SEEDS, T))

for seed in range(N_SEEDS):
    cb = all_competing_bids[seed]
    cl = all_clicks[seed]

    # UCB1-no-budget
    cr, cc = run_ucb1_no_budget(cb, cl)
    all_cum_rewards[0, seed] = cr
    all_cum_costs[0, seed] = cc

    # Proportional Pacing (needs its own RNG for pacing coin flips)
    pacing_rng = np.random.RandomState(seed + 1000)
    cr, cc = run_proportional_pacing(cb, cl, pacing_rng)
    all_cum_rewards[1, seed] = cr
    all_cum_costs[1, seed] = cc

    # BwK Primal-Dual
    cr, cc = run_bwk_primal_dual(cb, cl)
    all_cum_rewards[2, seed] = cr
    all_cum_costs[2, seed] = cc

    if (seed + 1) % 10 == 0:
        print(f"Completed seed {seed + 1}/{N_SEEDS}")

# ---------------------------------------------------------------------------
# Compute means and standard errors
# ---------------------------------------------------------------------------
mean_rewards = all_cum_rewards.mean(axis=1)   # (n_algos, T)
se_rewards = all_cum_rewards.std(axis=1) / np.sqrt(N_SEEDS)

mean_costs = all_cum_costs.mean(axis=1)
se_costs = all_cum_costs.std(axis=1) / np.sqrt(N_SEEDS)

# Regret vs LP benchmark (cumulative LP benchmark at time t)
lp_cum = lp_reward_per_round * np.arange(1, T + 1)
mean_regret = lp_cum[np.newaxis, :] - mean_rewards
se_regret = se_rewards  # same SE (LP is deterministic)

# Budget utilization: cumulative cost / B_total
mean_util = mean_costs / B_total
se_util = se_costs / B_total

# ---------------------------------------------------------------------------
# Figure: 3-panel plot
# ---------------------------------------------------------------------------
colors = ["#e74c3c", "#3498db", "#2ecc71"]
ts = np.arange(1, T + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: Cumulative Reward
ax = axes[0]
for i in range(n_algos):
    ax.plot(ts, mean_rewards[i], color=colors[i], label=algo_names[i])
    ax.fill_between(ts, mean_rewards[i] - se_rewards[i],
                    mean_rewards[i] + se_rewards[i], color=colors[i], alpha=0.15)
ax.plot(ts, lp_cum, 'k--', label="LP Benchmark", linewidth=1.5)
ax.set_xlabel("Round $t$")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Cumulative Reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Center: Regret vs LP
ax = axes[1]
for i in range(n_algos):
    ax.plot(ts, mean_regret[i], color=colors[i], label=algo_names[i])
    ax.fill_between(ts, mean_regret[i] - se_regret[i],
                    mean_regret[i] + se_regret[i], color=colors[i], alpha=0.15)
ax.set_xlabel("Round $t$")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Regret vs LP Relaxation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Budget Utilization
ax = axes[2]
for i in range(n_algos):
    ax.plot(ts, mean_util[i], color=colors[i], label=algo_names[i])
    ax.fill_between(ts, mean_util[i] - se_util[i],
                    mean_util[i] + se_util[i], color=colors[i], alpha=0.15)
ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label="Budget Limit")
ax.set_xlabel("Round $t$")
ax.set_ylabel("Cumulative Cost / $B$")
ax.set_title("Budget Utilization")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "bwk_regret.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {fig_path}")

# ---------------------------------------------------------------------------
# LaTeX table: bwk_results.tex
# ---------------------------------------------------------------------------
final_rewards = all_cum_rewards[:, :, -1]   # (n_algos, N_SEEDS)
final_costs = all_cum_costs[:, :, -1]
final_util_pct = 100.0 * final_costs / B_total
final_regret = lp_benchmark - final_rewards

table_lines = [
    r"\begin{tabular}{lccc}",
    r"\hline",
    r"Algorithm & Total Reward & Budget Utilization (\%) & Regret vs LP \\",
    r"\hline",
]

for i in range(n_algos):
    rew_mean = final_rewards[i].mean()
    rew_se = final_rewards[i].std() / np.sqrt(N_SEEDS)
    util_mean = final_util_pct[i].mean()
    util_se = final_util_pct[i].std() / np.sqrt(N_SEEDS)
    reg_mean = final_regret[i].mean()
    reg_se = final_regret[i].std() / np.sqrt(N_SEEDS)

    line = (f"{algo_names[i]} & "
            f"${rew_mean:.1f} \\pm {rew_se:.1f}$ & "
            f"${util_mean:.1f} \\pm {util_se:.1f}$ & "
            f"${reg_mean:.1f} \\pm {reg_se:.1f}$ \\\\")
    table_lines.append(line)

table_lines.append(r"\hline")
table_lines.append(f"LP Benchmark & ${lp_benchmark:.1f}$ & --- & $0.0$ \\\\")
table_lines.append(r"\hline")
table_lines.append(r"\end{tabular}")

tex_path = os.path.join(OUTPUT_DIR, "bwk_results.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(table_lines) + "\n")
print(f"Saved table: {tex_path}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print("\n=== Final Results ===")
print(f"{'Algorithm':<25} {'Reward':>15} {'Util %':>15} {'Regret':>15}")
print("-" * 72)
for i in range(n_algos):
    rew = f"{final_rewards[i].mean():.1f} ± {final_rewards[i].std()/np.sqrt(N_SEEDS):.1f}"
    util = f"{final_util_pct[i].mean():.1f} ± {final_util_pct[i].std()/np.sqrt(N_SEEDS):.1f}"
    reg = f"{final_regret[i].mean():.1f} ± {final_regret[i].std()/np.sqrt(N_SEEDS):.1f}"
    print(f"{algo_names[i]:<25} {rew:>15} {util:>15} {reg:>15}")
print(f"{'LP Benchmark':<25} {lp_benchmark:>15.1f} {'---':>15} {'0.0':>15}")
