# Chapter 10c claim-source ledger

Date: 2026-07-24

| Chapter claim | Primary source checked | Result |
|---|---|---|
| Greedy-casino observational means are 0.15 and interventional means are 0.30 | Bareinboim, Forney, and Pearl (2015), Table 1 and Section 4 | Corrected. The prior chapter stated 0.10 and 0.30 in one place and one half in another. |
| Bareinboim Algorithm 1 seeds the off-intuition posterior from an ETT identity | Bareinboim, Forney, and Pearl (2015), Algorithm 1; authors' `thompsonCausalRun.m` | Rejected. Algorithm 1 seeds only the on-intuition cells. The ETT seed is now labelled as a separate binary data-fusion extension. |
| Lattimore Algorithm 1 attains graph-dependent simple regret | Lattimore, Lattimore, and Reid (2016), Theorems 1 and 2 | Retained with the logarithmic factor shown and the parallel-bandit scope stated. |
| General-graph complexity equals the parallel hardness | Lattimore, Lattimore, and Reid (2016), Theorem 3 and proof of Corollary 1 | Corrected. The optimized general-graph complexity is bounded by twice the parallel hardness. |
| Exploration sampling is rate-optimal for frequentist expected policy regret | Kasy and Sautmann (2021), Theorem 1; authors' correction dated 2021-11-10 | Rejected. The correction withdraws item 3. The chapter now states the retained posterior-concentration result and separate posterior expected-regret bound. |
| Constant-allocation adaptive AIPW weights minimize asymptotic variance | Hadad et al. (2021), Sections 3.1 and 3.2, Theorem 2 | Corrected. The weights establish variance stabilization and asymptotic normality but are not variance-optimal. |
| Post-contextual-bandit inference permits unknown black-box logging | Bibaut et al. (2021), setup and main theorem | Rejected. The adaptive logging probabilities remain known. |

## Simulation claims

The parallel-bandit data-generating process now matches the paper's hard family.
Changing the number of zero-propensity parents changes observability but leaves
the value of every arm unchanged. The optimal coordinate is one of the
zero-propensity parents, so phase 2 must directly sample it.

Successive Reject now uses cumulative phase targets and adds only
`n_k - n_{k-1}` pulls in phase `k`. Lattimore Algorithm 1 uses the paper's
`1 / m_hat` threshold. Entry-point checks fail if budgets overflow, labelled
hardness differs from realized hardness, non-optimal arm values vary across the
hardness grid, or the greedy-casino marginals differ from Table 1.
