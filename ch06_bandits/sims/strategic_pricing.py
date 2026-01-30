# Strategic Pricing with Reference Effects — Chapter 6, Economic Bandits.
# Two-firm logit pricing with reference price dynamics: gradient ascent vs Q-learning vs Nash oracle.

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_FIRMS = 2
T = 5_000
N_SEEDS = 30

# Consumer utility: U_t(i) = alpha_i - beta * p_i + gamma * (rho_t - p_i) + eps_i
# Logit choice with outside option normalised to 0.
ALPHA = np.array([2.0, 2.0])
BETA = 1.0
GAMMA = 0.5       # reference price sensitivity
DELTA = 0.1       # reference price smoothing
MARGINAL_COST = 0.0

# Price bounds for projection
P_MIN = 0.1
P_MAX = 5.0

# Q-learning: discretise prices
N_PRICE_LEVELS = 20
PRICE_GRID = np.linspace(P_MIN, P_MAX, N_PRICE_LEVELS)

# Q-learning hyperparameters
Q_LR = 0.1
Q_GAMMA_DISCOUNT = 0.0   # myopic (stateless bandit-like)
Q_EPS_START = 1.0
Q_EPS_END = 0.01
Q_EPS_DECAY = T * 0.8    # linear decay over 80% of horizon

# Gradient ascent hyperparameters
GA_ETA0 = 0.5            # initial step size
GA_DECAY = 50.0          # step size: eta_t = GA_ETA0 / (1 + t/GA_DECAY)
GA_PERTURBATION = 0.01   # finite-difference perturbation

# ---------------------------------------------------------------------------
# Logit demand and profit functions
# ---------------------------------------------------------------------------
def logit_shares(prices, rho):
    """Logit choice probabilities for N_FIRMS firms + outside option."""
    # V_i = alpha_i - beta * p_i + gamma * (rho - p_i)
    V = ALPHA - BETA * prices + GAMMA * (rho - prices)
    # Outside option utility = 0
    exp_V = np.exp(V)
    denom = 1.0 + exp_V.sum()
    return exp_V / denom

def expected_profit(prices, rho):
    """Expected profit for each firm (revenue only, zero marginal cost)."""
    shares = logit_shares(prices, rho)
    return (prices - MARGINAL_COST) * shares

def update_reference_price(rho, prices):
    """Exponential smoothing: rho_{t+1} = (1-delta)*rho + delta * mean(prices)."""
    return (1.0 - DELTA) * rho + DELTA * np.mean(prices)

# ---------------------------------------------------------------------------
# Compute static Nash equilibrium numerically
# ---------------------------------------------------------------------------
def compute_nash_equilibrium(rho_fixed=None, tol=1e-8, max_iter=10000):
    """Find symmetric Nash equilibrium via iterated best response.
    If rho_fixed is None, solve for the fixed point where rho = mean(p^*)."""
    prices = np.array([2.0, 2.0])
    for _ in range(max_iter):
        old_prices = prices.copy()
        for i in range(N_FIRMS):
            # Best response for firm i: maximise p_i * s_i(p_i, p_{-i}, rho)
            best_p = prices[i]
            best_profit = -np.inf
            for p_cand in np.linspace(P_MIN, P_MAX, 1000):
                test_prices = prices.copy()
                test_prices[i] = p_cand
                rho = rho_fixed if rho_fixed is not None else np.mean(test_prices)
                pi_i = expected_profit(test_prices, rho)[i]
                if pi_i > best_profit:
                    best_profit = pi_i
                    best_p = p_cand
            prices[i] = best_p
        if np.max(np.abs(prices - old_prices)) < tol:
            break
    # Compute the equilibrium reference price
    rho_eq = np.mean(prices)
    return prices, rho_eq

# Compute Nash equilibrium at the steady-state reference price
NE_PRICES, NE_RHO = compute_nash_equilibrium()
NE_PROFITS = expected_profit(NE_PRICES, NE_RHO)
print(f"Nash equilibrium prices: {NE_PRICES}")
print(f"Nash equilibrium rho:    {NE_RHO:.4f}")
print(f"Nash equilibrium profits: {NE_PROFITS}")

# ---------------------------------------------------------------------------
# Algorithm 1: Online Projected Gradient Ascent (structure-aware)
# ---------------------------------------------------------------------------
def run_gradient_ascent(seed):
    rng = np.random.RandomState(seed)
    prices = rng.uniform(P_MIN, P_MAX, size=N_FIRMS)
    rho = np.mean(prices)

    price_history = np.zeros((T, N_FIRMS))
    profit_history = np.zeros((T, N_FIRMS))
    rho_history = np.zeros(T)

    for t in range(T):
        # Record
        price_history[t] = prices
        rho_history[t] = rho

        # Realise demand (draw logit shocks, compute shares and profits)
        shares = logit_shares(prices, rho)
        # Simulate: each of N_CONSUMERS consumers makes a choice
        # For simplicity, use expected profit (law of large numbers with many consumers)
        profits = expected_profit(prices, rho)
        profit_history[t] = profits

        # Gradient estimation via finite differences
        eta_t = GA_ETA0 / (1.0 + t / GA_DECAY)
        new_prices = prices.copy()
        for i in range(N_FIRMS):
            prices_plus = prices.copy()
            prices_plus[i] += GA_PERTURBATION
            prices_minus = prices.copy()
            prices_minus[i] -= GA_PERTURBATION
            grad_i = (expected_profit(prices_plus, rho)[i] -
                      expected_profit(prices_minus, rho)[i]) / (2 * GA_PERTURBATION)
            new_prices[i] = prices[i] + eta_t * grad_i

        # Project onto [P_MIN, P_MAX]
        prices = np.clip(new_prices, P_MIN, P_MAX)

        # Update reference price
        rho = update_reference_price(rho, prices)

    return price_history, profit_history, rho_history

# ---------------------------------------------------------------------------
# Algorithm 2: Independent Q-learning (model-free)
# ---------------------------------------------------------------------------
def run_q_learning(seed):
    rng = np.random.RandomState(seed)

    # Q-table: each firm has a table of size N_PRICE_LEVELS
    # (single state, so Q(a) only)
    Q = [np.zeros(N_PRICE_LEVELS) for _ in range(N_FIRMS)]
    counts = [np.zeros(N_PRICE_LEVELS) for _ in range(N_FIRMS)]

    prices = rng.uniform(P_MIN, P_MAX, size=N_FIRMS)
    rho = np.mean(prices)

    price_history = np.zeros((T, N_FIRMS))
    profit_history = np.zeros((T, N_FIRMS))
    rho_history = np.zeros(T)

    for t in range(T):
        # Epsilon schedule (linear decay)
        eps = max(Q_EPS_END, Q_EPS_START - (Q_EPS_START - Q_EPS_END) * t / Q_EPS_DECAY)

        # Each firm selects a price level
        actions = np.zeros(N_FIRMS, dtype=int)
        for i in range(N_FIRMS):
            if rng.random() < eps:
                actions[i] = rng.randint(N_PRICE_LEVELS)
            else:
                actions[i] = np.argmax(Q[i])

        prices = PRICE_GRID[actions]
        price_history[t] = prices
        rho_history[t] = rho

        # Observe profits (with noise: simulate finite consumer pool)
        shares = logit_shares(prices, rho)
        # Add noise: realised demand from 100 consumers
        n_consumers = 100
        realised_shares = np.zeros(N_FIRMS)
        for c in range(n_consumers):
            V = ALPHA - BETA * prices + GAMMA * (rho - prices)
            exp_V = np.exp(V)
            denom = 1.0 + exp_V.sum()
            probs = np.append(exp_V / denom, 1.0 / denom)  # firms + outside
            choice = rng.choice(N_FIRMS + 1, p=probs)
            if choice < N_FIRMS:
                realised_shares[choice] += 1.0 / n_consumers

        profits = prices * realised_shares
        profit_history[t] = profits

        # Q-learning update (myopic: no discounting)
        for i in range(N_FIRMS):
            a = actions[i]
            counts[i][a] += 1
            lr = Q_LR  # fixed learning rate
            Q[i][a] += lr * (profits[i] - Q[i][a])

        # Update reference price
        rho = update_reference_price(rho, prices)

    return price_history, profit_history, rho_history

# ---------------------------------------------------------------------------
# Algorithm 3: Static Nash Oracle
# ---------------------------------------------------------------------------
def run_nash_oracle(seed):
    rng = np.random.RandomState(seed)
    prices = NE_PRICES.copy()
    rho = NE_RHO

    price_history = np.zeros((T, N_FIRMS))
    profit_history = np.zeros((T, N_FIRMS))
    rho_history = np.zeros(T)

    for t in range(T):
        price_history[t] = prices
        rho_history[t] = rho
        profits = expected_profit(prices, rho)
        profit_history[t] = profits
        rho = update_reference_price(rho, prices)

    return price_history, profit_history, rho_history

# ---------------------------------------------------------------------------
# Run experiments across seeds
# ---------------------------------------------------------------------------
algo_names = ["Gradient Ascent", "Q-learning", "Nash Oracle"]
algo_funcs = [run_gradient_ascent, run_q_learning, run_nash_oracle]
n_algos = len(algo_names)

# Storage
all_dist_to_ne = np.zeros((n_algos, N_SEEDS, T))
all_cum_profit = np.zeros((n_algos, N_SEEDS, T))

for algo_idx, (name, func) in enumerate(zip(algo_names, algo_funcs)):
    for seed in range(N_SEEDS):
        price_hist, profit_hist, rho_hist = func(seed)

        # Distance to Nash equilibrium
        dist = np.sqrt(np.sum((price_hist - NE_PRICES[np.newaxis, :]) ** 2, axis=1))
        all_dist_to_ne[algo_idx, seed] = dist

        # Cumulative profit (sum over firms)
        all_cum_profit[algo_idx, seed] = np.cumsum(profit_hist.sum(axis=1))

    print(f"Completed {name}: all {N_SEEDS} seeds")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
mean_dist = all_dist_to_ne.mean(axis=1)
se_dist = all_dist_to_ne.std(axis=1) / np.sqrt(N_SEEDS)

mean_cum_profit = all_cum_profit.mean(axis=1)
se_cum_profit = all_cum_profit.std(axis=1) / np.sqrt(N_SEEDS)

# Final-round statistics
final_dist_mean = all_dist_to_ne[:, :, -1].mean(axis=1)
final_dist_se = all_dist_to_ne[:, :, -1].std(axis=1) / np.sqrt(N_SEEDS)
final_profit_mean = all_cum_profit[:, :, -1].mean(axis=1)
final_profit_se = all_cum_profit[:, :, -1].std(axis=1) / np.sqrt(N_SEEDS)

# Average distance over last 500 rounds
avg_dist_last500_mean = all_dist_to_ne[:, :, -500:].mean(axis=2).mean(axis=1)
avg_dist_last500_se = all_dist_to_ne[:, :, -500:].mean(axis=2).std(axis=1) / np.sqrt(N_SEEDS)

print("\n=== Final Results ===")
for i in range(n_algos):
    print(f"{algo_names[i]:<20} dist_to_NE(last500): {avg_dist_last500_mean[i]:.4f} ± {avg_dist_last500_se[i]:.4f}  "
          f"cum_profit: {final_profit_mean[i]:.1f} ± {final_profit_se[i]:.1f}")

# ---------------------------------------------------------------------------
# Figure: 2-panel (convergence to NE + cumulative profit)
# ---------------------------------------------------------------------------
colors = ["#2ca02c", "#1f77b4", "#7f7f7f"]
ts = np.arange(1, T + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: distance to Nash equilibrium
for i in range(n_algos):
    ax1.plot(ts, mean_dist[i], color=colors[i], label=algo_names[i], linewidth=1.5)
    ax1.fill_between(ts, mean_dist[i] - se_dist[i], mean_dist[i] + se_dist[i],
                     color=colors[i], alpha=0.15)

ax1.set_xlabel("Round $t$")
ax1.set_ylabel(r"$\|p_t - p^*_{\mathrm{NE}}\|_2$")
ax1.set_title("Distance to Nash Equilibrium")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_xlim(1, T)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# Right panel: cumulative profit (sum over firms)
for i in range(n_algos):
    ax2.plot(ts, mean_cum_profit[i], color=colors[i], label=algo_names[i], linewidth=1.5)
    ax2.fill_between(ts, mean_cum_profit[i] - se_cum_profit[i],
                     mean_cum_profit[i] + se_cum_profit[i],
                     color=colors[i], alpha=0.15)

ax2.set_xlabel("Round $t$")
ax2.set_ylabel("Cumulative Industry Profit")
ax2.set_title("Cumulative Profit")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_xlim(1, T)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "strategic_pricing_convergence.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved figure: {fig_path}")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
tex_lines = [
    r"\begin{tabular}{lcc}",
    r"\hline",
    r"Algorithm & Avg.\ Distance to NE (last 500) & Cumulative Industry Profit \\",
    r"\hline",
]

for i in range(n_algos):
    dist_str = f"${avg_dist_last500_mean[i]:.4f} \\pm {avg_dist_last500_se[i]:.4f}$"
    prof_str = f"${final_profit_mean[i]:.1f} \\pm {final_profit_se[i]:.1f}$"
    tex_lines.append(f"{algo_names[i]} & {dist_str} & {prof_str} \\\\")

tex_lines.append(r"\hline")
tex_lines.append(r"\end{tabular}")

tex_path = os.path.join(OUTPUT_DIR, "strategic_pricing_results.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print(f"Saved table: {tex_path}")
