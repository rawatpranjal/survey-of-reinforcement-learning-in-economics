# Auction Reserve Price Optimization — Chapter 6, Economic Bandits.
# Unimodal structure in second-price auction reserve price learning.

import os
import numpy as np
from scipy import optimize
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_BIDDERS = 3
MU, SIGMA = 0.0, 0.5          # LogNormal parameters (underlying normal)
K = 25
reserves = np.linspace(0.2, 5.0, K)
T = 20_000
N_SEEDS = 30
MC_SAMPLES = 100_000

# ─── Myerson Optimal Reserve ────────────────────────────────────────────────
# For LogNormal(mu, sigma), the PDF and CDF use scipy's parameterisation:
#   lognorm(s=sigma, scale=exp(mu))
dist = lognorm(s=SIGMA, scale=np.exp(MU))

def virtual_valuation(v):
    """phi(v) = v - (1 - F(v)) / f(v).  Myerson optimal where phi(v) = 0."""
    return v - (1.0 - dist.cdf(v)) / dist.pdf(v)

# Solve phi(v) = 0 numerically
sol = optimize.brentq(virtual_valuation, 0.1, 10.0)
r_star = sol
print(f"Myerson optimal reserve: r* = {r_star:.4f}")

# ─── Expected Revenue Curve via Monte Carlo ─────────────────────────────────
np.random.seed(999)
mc_vals = dist.rvs(size=(MC_SAMPLES, N_BIDDERS))
mc_vals_sorted = np.sort(mc_vals, axis=1)  # ascending: col 0 lowest, col 2 highest
mc_highest = mc_vals_sorted[:, -1]
mc_second  = mc_vals_sorted[:, -2]

expected_revenue = np.zeros(K)
for k, r in enumerate(reserves):
    # Revenue: if highest >= r, revenue = max(r, second-highest); else 0
    sold = mc_highest >= r
    rev = np.where(sold, np.maximum(r, mc_second), 0.0)
    expected_revenue[k] = rev.mean()

# Oracle expected revenue (at Myerson optimal)
sold_star = mc_highest >= r_star
rev_star_samples = np.where(sold_star, np.maximum(r_star, mc_second), 0.0)
oracle_rev = rev_star_samples.mean()
print(f"Oracle expected revenue per round: {oracle_rev:.4f}")

# ─── Auction Revenue Function ───────────────────────────────────────────────
def auction_revenue(bids, reserve):
    """Second-price auction with reserve. bids: array of shape (N_BIDDERS,)."""
    sorted_bids = np.sort(bids)
    highest = sorted_bids[-1]
    second  = sorted_bids[-2]
    if highest >= reserve:
        return max(reserve, second)
    return 0.0

# Vectorised version for a full trajectory
def auction_revenue_vec(bid_matrix, reserve):
    """bid_matrix: (T, N_BIDDERS). Returns (T,) revenues."""
    sorted_bids = np.sort(bid_matrix, axis=1)
    highest = sorted_bids[:, -1]
    second  = sorted_bids[:, -2]
    sold = highest >= reserve
    return np.where(sold, np.maximum(reserve, second), 0.0)

# ─── Algorithm Implementations ──────────────────────────────────────────────

def run_ucb1(bid_matrix):
    """Standard UCB1 over K=25 reserve-price arms."""
    counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K)
    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    # Initialise: play each arm once
    for k in range(K):
        rev = auction_revenue(bid_matrix[k], reserves[k])
        counts[k] = 1
        sum_rewards[k] = rev
        actions[k] = k
        rewards[k] = rev

    for t in range(K, T):
        means = sum_rewards / counts
        ucb = means + np.sqrt(2.0 * np.log(t) / counts)
        arm = np.argmax(ucb)
        rev = auction_revenue(bid_matrix[t], reserves[arm])
        counts[arm] += 1
        sum_rewards[arm] += rev
        actions[t] = arm
        rewards[t] = rev

    return rewards, actions, counts


def run_uucb(bid_matrix):
    """Unimodal UCB: maintain leader L, explore only neighbours L-1, L+1.
    Exploits the unimodal structure of the expected revenue curve over
    ordered reserve prices. The leader is the current best arm; only
    its immediate neighbours are candidates for exploration."""
    counts = np.zeros(K, dtype=int)
    sum_rewards = np.zeros(K)
    actions = np.zeros(T, dtype=int)
    rewards = np.zeros(T)

    # Warm-up: play each arm multiple times for reliable initial estimates
    WARM_UP = 5
    t_idx = 0
    for rep in range(WARM_UP):
        for k in range(K):
            rev = auction_revenue(bid_matrix[t_idx], reserves[k])
            counts[k] += 1
            sum_rewards[k] += rev
            actions[t_idx] = k
            rewards[t_idx] = rev
            t_idx += 1

    # Start leader at the arm with highest warm-up mean
    leader = np.argmax(sum_rewards / counts)

    for t in range(t_idx, T):
        means = sum_rewards / counts

        # Identify neighbours of current leader
        neighbours = []
        if leader > 0:
            neighbours.append(leader - 1)
        if leader < K - 1:
            neighbours.append(leader + 1)

        # Among neighbours, find the one with highest UCB
        best_nb = None
        best_nb_ucb = -np.inf
        for nb in neighbours:
            ucb_nb = means[nb] + np.sqrt(2.0 * np.log(t + 1) / counts[nb])
            if ucb_nb > best_nb_ucb:
                best_nb_ucb = ucb_nb
                best_nb = nb

        # Explore neighbour if its UCB exceeds leader's sample mean
        if best_nb is not None and best_nb_ucb > means[leader]:
            arm = best_nb
        else:
            arm = leader

        rev = auction_revenue(bid_matrix[t], reserves[arm])
        counts[arm] += 1
        sum_rewards[arm] += rev
        actions[t] = arm
        rewards[t] = rev

        # Migrate leader: if played neighbour now has higher mean, switch
        if arm != leader:
            updated_means = sum_rewards / counts
            if updated_means[arm] > updated_means[leader]:
                leader = arm

    return rewards, actions, counts


def run_oracle(bid_matrix):
    """Always play the Myerson optimal reserve (nearest grid point)."""
    oracle_arm = np.argmin(np.abs(reserves - r_star))
    revs = auction_revenue_vec(bid_matrix, reserves[oracle_arm])
    actions = np.full(T, oracle_arm, dtype=int)
    counts = np.zeros(K, dtype=int)
    counts[oracle_arm] = T
    return revs, actions, counts

# ─── Monte Carlo Over Seeds ─────────────────────────────────────────────────
oracle_arm_idx = np.argmin(np.abs(reserves - r_star))
oracle_expected = expected_revenue[oracle_arm_idx]

all_regret_ucb1 = np.zeros((N_SEEDS, T))
all_regret_uucb = np.zeros((N_SEEDS, T))
all_total_rev_ucb1 = np.zeros(N_SEEDS)
all_total_rev_uucb = np.zeros(N_SEEDS)
all_total_rev_oracle = np.zeros(N_SEEDS)
all_counts_ucb1 = np.zeros((N_SEEDS, K))
all_counts_uucb = np.zeros((N_SEEDS, K))

for seed in range(N_SEEDS):
    np.random.seed(seed)
    bid_matrix = dist.rvs(size=(T, N_BIDDERS))

    # Oracle
    rev_oracle, _, cnt_oracle = run_oracle(bid_matrix)

    # UCB1
    rev_ucb1, _, cnt_ucb1 = run_ucb1(bid_matrix)

    # UUCB
    rev_uucb, _, cnt_uucb = run_uucb(bid_matrix)

    # Cumulative regret relative to oracle
    cum_oracle = np.cumsum(rev_oracle)
    all_regret_ucb1[seed] = cum_oracle - np.cumsum(rev_ucb1)
    all_regret_uucb[seed] = cum_oracle - np.cumsum(rev_uucb)

    all_total_rev_ucb1[seed] = rev_ucb1.sum()
    all_total_rev_uucb[seed] = rev_uucb.sum()
    all_total_rev_oracle[seed] = rev_oracle.sum()
    all_counts_ucb1[seed] = cnt_ucb1
    all_counts_uucb[seed] = cnt_uucb

    if (seed + 1) % 10 == 0:
        print(f"  Completed seed {seed + 1}/{N_SEEDS}")

# ─── Aggregate Statistics ────────────────────────────────────────────────────
mean_regret_ucb1 = all_regret_ucb1.mean(axis=0)
se_regret_ucb1   = all_regret_ucb1.std(axis=0) / np.sqrt(N_SEEDS)
mean_regret_uucb = all_regret_uucb.mean(axis=0)
se_regret_uucb   = all_regret_uucb.std(axis=0) / np.sqrt(N_SEEDS)

mean_counts_ucb1 = all_counts_ucb1.mean(axis=0)
mean_counts_uucb = all_counts_uucb.mean(axis=0)

final_regret_ucb1_mean = all_regret_ucb1[:, -1].mean()
final_regret_ucb1_se   = all_regret_ucb1[:, -1].std() / np.sqrt(N_SEEDS)
final_regret_uucb_mean = all_regret_uucb[:, -1].mean()
final_regret_uucb_se   = all_regret_uucb[:, -1].std() / np.sqrt(N_SEEDS)

pct_ucb1_mean = (all_total_rev_ucb1 / all_total_rev_oracle * 100).mean()
pct_ucb1_se   = (all_total_rev_ucb1 / all_total_rev_oracle * 100).std() / np.sqrt(N_SEEDS)
pct_uucb_mean = (all_total_rev_uucb / all_total_rev_oracle * 100).mean()
pct_uucb_se   = (all_total_rev_uucb / all_total_rev_oracle * 100).std() / np.sqrt(N_SEEDS)

print(f"\nFinal Regret  UCB1: {final_regret_ucb1_mean:.1f} +/- {final_regret_ucb1_se:.1f}")
print(f"Final Regret  UUCB: {final_regret_uucb_mean:.1f} +/- {final_regret_uucb_se:.1f}")
print(f"% Myerson Rev UCB1: {pct_ucb1_mean:.2f}% +/- {pct_ucb1_se:.2f}%")
print(f"% Myerson Rev UUCB: {pct_uucb_mean:.2f}% +/- {pct_uucb_se:.2f}%")

# ─── Theoretical Overlays ───────────────────────────────────────────────────
t_range = np.arange(1, T + 1)
c1 = 0.3
c2 = 15.0
agnostic_bound   = c1 * np.sqrt(K * t_range)
structural_bound = c2 * np.log(t_range + 1)

# ─── Figure: 2-Panel Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: cumulative regret
ax1.plot(t_range, mean_regret_ucb1, color='C0', label='UCB1')
ax1.fill_between(t_range,
                 mean_regret_ucb1 - se_regret_ucb1,
                 mean_regret_ucb1 + se_regret_ucb1,
                 alpha=0.2, color='C0')
ax1.plot(t_range, mean_regret_uucb, color='C1', label='Unimodal UCB')
ax1.fill_between(t_range,
                 mean_regret_uucb - se_regret_uucb,
                 mean_regret_uucb + se_regret_uucb,
                 alpha=0.2, color='C1')
ax1.plot(t_range, agnostic_bound, 'k--', alpha=0.6, label=r'Agnostic $O(\sqrt{Kt})$')
ax1.plot(t_range, structural_bound, 'k:', alpha=0.6, label=r'Structural $O(\log t)$')
ax1.set_xlabel('Round $t$')
ax1.set_ylabel('Cumulative Regret')
ax1.set_title('Cumulative Regret vs Myerson Optimal')
ax1.legend(loc='upper left')
ax1.set_xlim(0, T)

# Right panel: revenue curve + exploration distribution
ax2.plot(reserves, expected_revenue, 'k-', linewidth=2, label='$E[\\mathrm{rev}(r)]$')
ax2.axvline(r_star, color='red', linestyle='--', linewidth=1.5, label=f'Myerson $r^*={r_star:.2f}$')

# Normalised pull fractions as bars
bar_width = (reserves[1] - reserves[0]) * 0.35
pull_frac_ucb1 = mean_counts_ucb1 / mean_counts_ucb1.sum()
pull_frac_uucb = mean_counts_uucb / mean_counts_uucb.sum()

# Scale pull fractions to fit on the revenue axis
scale = expected_revenue.max() * 0.8
ax2.bar(reserves - bar_width / 2, pull_frac_ucb1 * scale / pull_frac_ucb1.max(),
        width=bar_width, alpha=0.35, color='C0', label='UCB1 pulls')
ax2.bar(reserves + bar_width / 2, pull_frac_uucb * scale / pull_frac_uucb.max(),
        width=bar_width, alpha=0.35, color='C1', label='UUCB pulls')

ax2.set_xlabel('Reserve Price')
ax2.set_ylabel('Expected Revenue / Pull Fraction')
ax2.set_title('Revenue Curve and Exploration')
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'auction_reserve_regret.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved figure: {fig_path}")

# ─── LaTeX Table ─────────────────────────────────────────────────────────────
tex_path = os.path.join(OUTPUT_DIR, 'auction_reserve_results.tex')
with open(tex_path, 'w') as f:
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\hline\n")
    f.write("Algorithm & Final Regret & \\% of Myerson Optimum Revenue \\\\\n")
    f.write("\\hline\n")
    f.write(f"UCB1 & ${final_regret_ucb1_mean:.1f} \\pm {final_regret_ucb1_se:.1f}$ "
            f"& ${pct_ucb1_mean:.2f} \\pm {pct_ucb1_se:.2f}\\%$ \\\\\n")
    f.write(f"Unimodal UCB & ${final_regret_uucb_mean:.1f} \\pm {final_regret_uucb_se:.1f}$ "
            f"& ${pct_uucb_mean:.2f} \\pm {pct_uucb_se:.2f}\\%$ \\\\\n")
    f.write(f"Myerson Oracle & $0.0 \\pm 0.0$ & $100.00 \\pm 0.00\\%$ \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
print(f"Saved table:  {tex_path}")
