# Fix Report: ch06_games/sims/durable_goods_monopoly.py

**Date:** 2026-05-19
**Original score:** 65% (see `ch06_games__durable_goods_monopoly_2026-05-19.md`)
**Strategy:** Rescope + bug fixes. No reimplementation of an asymptotic Coase setup.
**Diagram-only:** no.
**Cited tex file(s):** `/Users/pranjal/Code/rl/ch06_games/tex/rl_in_games.tex` §"Screening versus Pooling in the Durable Goods Monopoly" (renamed from §"The Coase Conjecture"), lines 148--189; `\input` of `durable_goods_results.tex`.
**Cited paper PDFs read:** none. The script's docstring previously cited Ausubel-Cramton-Deneckere; the rescoped docstring now cites \citet{ausubel1989reputation} and \citet{ausubel2002bargaining} as the asymptotic-treatment references, both of which exist in `docs/refs.bib` (verified at lines 3880--3899).

## 1. Algorithm Identity

CFR implementation was already faithful (vanilla CFR with regret matching, strategy averaging). The fix preserves it. Per-seed reproducibility is now obtained by Gaussian-perturbing the initial regret table (std $0.05$), which leaves the CFR update rule untouched; this is a standard way to obtain seed-level variability for a deterministic solver and is documented in the docstring.

## 2. Environment / MDP Fidelity

The script remains a two-period game with a two-element seller price set $\{v_L, P^*(\delta)\}$. The audit's most damaging finding was the title/artifact mismatch (Coase conjecture is asymptotic; sim is two-period). Fixed by:

- Renamed the section from "The Coase Conjecture" to "Screening versus Pooling in the Durable Goods Monopoly" (rl_in_games.tex line 148).
- Added a one-sentence framing note immediately after the conjecture is stated: "The simulation below is a finite-horizon precursor to the Coase conjecture. With only two periods and a two-element seller price set $\{v_L, P^*(\delta)\}$, the artifact recovers the screening-versus-pooling threshold rather than the asymptotic price-collapse statement."
- Extended the existing footnote at line 152 to add: "The asymptotic price-collapse limit ($T \to \infty$, $\delta \to 1$) of the Coase conjecture is not reproduced in the two-period simulation that follows; for the canonical asymptotic treatment, see \citet{ausubel1989reputation} and the survey by \citet{ausubel2002bargaining}."
- Replaced "The Coase conjecture manifests as $\delta \to 1$, where $P^*(\delta) \to v_L$" (line 168, original) with an honest reading: as $\delta \to 1$, $P^*(\delta) \to v_L$ in this finite-horizon setup, but the asymptotic price-collapse mechanism does not operate in a two-period game with action set $\{v_L, P^*(\delta)\}$.

The script header docstring was rewritten end-to-end to remove the "VALIDATION FRAMEWORK / proves convergence to Nash equilibrium" claim and to state up front that this is a finite-horizon precursor, not the Coase conjecture.

## 3. Data Integrity

`compute_data()` still calls the real CFR trainer. Results in `durable_goods_results.tex` come from the actual computation. The `version` field in `CONFIG` was bumped from 1 to 2 to invalidate the stale single-seed cache; verified that loading with `version: 2` returns `None`, forcing a fresh run. The new cache contains the multi-seed (n=10) trajectories.

## 4. Comparison Fairness

The audit flagged the post-hoc "near threshold (0.45--0.60), strategies can mix during transition" carve-out at the original lines 936--938, which silently checkmarked $\pi = 0.55$ (where CFR pooled and theory predicted screening). Fixed by removing the carve-out entirely and replacing the `Status` column with a transparent $|\Delta| = |\text{P(Screen)} - \text{Theory}|$ column. The reader sees, for example, $\pi = 0.55$ → $|\Delta| = 0.599$ rather than $\checkmark$. The validation status is now visible arithmetic, not a hidden tolerance.

Old original `Status` column logic at lines 932--940:
```
theory_match = abs(prob - theory) < 0.15
near_threshold = 0.45 <= pi <= 0.60
if theory_match or near_threshold:
    status = r"\checkmark"
```
Removed and not replaced; the $|\Delta|$ column carries the comparison.

## 5. Theoretical Sanity Checks

- **NashConv framing.** Renamed the convergence plot from "Convergence to Nash Equilibrium" to "Residual exploitability over CFR iterations (linear scale)". Switched y-axis from log to linear scale (audit finding 5: log-scale framing made the 20-unit gap look like progress). Added a reference line at $0.05 \cdot v_H = 10$ to anchor the magnitude relative to the maximum single-player payoff. Updated prose at rl_in_games.tex line 188: "Residual NashConv after 5{,}000 iterations is in the range $4$--$24$ across $\pi$, equivalent to up to $12\%$ of the maximum single-player payoff; CFR is not at $\varepsilon$-Nash for small $\varepsilon$ here, and reporting NashConv on a linear utility scale rather than a log scale makes this gap visible." This is the honest framing the audit demanded.

- **$\delta = 0.75$ narrative reframing.** Previously: "the seller switches to pooling as patient buyers erode the screening premium, consistent with the Coase conjecture." Replaced with (rl_in_games.tex line 189): "The analytical column predicts pure screening (P(Screen) $= 1$) throughout this range because $\pi = 0.7 > \pi^* = 1/2$, so this drift is not a Coase regime switch. Instead, as $\delta$ grows, the screening price $P^*(\delta)$ shrinks toward $v_L$ and the seller's screening profit approaches the pooling profit; the seller becomes near-indifferent and CFR's average strategy spreads across the two corners under seed-perturbed initial regrets." This matches the script's own analytical solver (which predicts screening at every $\delta$ here) and attributes the CFR drift correctly.

- **$\pi^* = 1/2$ transition framing.** Original prose claimed "CFR recovers the sharp phase transition at $\pi^* = 0.5$." Audit pointed out that CFR's transition is slightly delayed. Updated prose to: "The empirical transition CFR finds is slightly delayed relative to the analytical $\pi^* = 1/2$: $\text{P(Screen)} \approx 0.60$ at $\pi = 0.60$ and reaches $0.90$ only at $\pi = 0.70$."

## 6. Information Leakage

No change. The audit found no leakage; the seller's info set does not condition on the buyer type.

## 7. Seed and Reproducibility

Original: $n=1$ (vanilla CFR is deterministic given zero initial regrets).

Fix: $n=10$ seeds per $(\pi, \delta)$ cell. Seed-level variability is generated by perturbing the initial regret table with Gaussian noise (std $0.05$); each seed has its own `numpy.random.default_rng(seed)`. The CFR algorithm itself is unchanged.

Mean and SE are computed across seeds and:
- Printed in the stdout table (`Mean across-seed SE on P(Screen): 0.054` for the $\pi$-sweep).
- Added as `SE` columns in `durable_goods_results.tex` (one for P(Screen), one for NashConv).
- Plotted as error bars on the $\pi$-sweep and $\delta$-sweep figures.

Observed across-seed SE is non-trivial near the indifference threshold: at $\pi = 0.55$, P(Screen) $= 0.401$ with SE $0.163$, confirming the perturbation actually drives the system across the indifference boundary on different seeds. Away from the threshold ($\pi \le 0.40$ and $\pi \ge 0.80$), SE collapses to 0 because the equilibrium is strict.

The audit also flagged 5{,}000 iterations as too few. Did not increase iteration count; this is a rescope, not a reimplementation, and the residual NashConv is now reported honestly rather than waved away.

## Re-run and Recompile

Sim re-run with version bump to invalidate stale cache:
```
python3 ch06_games/sims/durable_goods_monopoly.py > ch06_games/sims/durable_goods_monopoly_stdout.txt 2>&1
```
Exit 0. All five output files regenerated: `durable_goods_coase.png`, `durable_goods_delta_sweep.png`, `durable_goods_nashconv.png`, `durable_goods_strategies.png`, `durable_goods_results.tex`. Stdout shows mean and SE columns for both sweeps.

Chapter PDF recompiled with three pdflatex passes plus bibtex. Output: `/Users/pranjal/Code/rl/docs/ch06_games.pdf` (14 pages, 525573 bytes). No undefined citations or undefined refs in the log; only standard hyperref/caption warnings.

## What was not done

- Did not extend the simulation to $T$-period or continuous-time variants. That would require reimplementing the asymptotic Coase setup, which is explicitly out of scope per the fix strategy. The asymptotic statement is now disclaimed in tex and referred to \citet{ausubel1989reputation} and \citet{ausubel2002bargaining}.
- Did not add any durable-goods reference PDFs to `ch06_games/papers/`. Citation suffices for the survey purpose; the audit's hostile-reviewer scenario is now defused at the prose level rather than the artifact level.
- Did not increase the iteration count from 5{,}000. Residual NashConv is reported honestly in the table and figure.

## Bullshit Score

**Bullshit score: 20%** --- Reviewer 2 still notes that the section's headline result is a finite-horizon screening exercise rather than the asymptotic Coase price-collapse. But the title no longer makes the asymptotic claim, the disclaimer is in the first footnote, the $\delta = 0.75$ behavior is correctly attributed to indifference-induced CFR drift rather than to Coase dynamics, the post-hoc $\checkmark$ carve-out is gone, NashConv is reported on linear scale with an explicit utility-share anchor, and multi-seed SE columns expose the genuine near-threshold mixing. The reviewer might still grumble that the artifact is illustrative rather than original economics, but the substance, the captions, and the prose are now internally consistent and would survive a hostile re-read.
