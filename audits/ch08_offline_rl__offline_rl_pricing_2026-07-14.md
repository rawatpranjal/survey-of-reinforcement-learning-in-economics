# Audit: ch08_offline_rl/sims/offline_rl_pricing.py

**Date:** 2026-07-14
**Type:** DELTA (prior: 2026-05-19 audit 50%, 2026-05-20 polish 15%)
**Auditor stance:** hostile journal referee, evidence-only, read-only (no re-run).

## Delta summary

The only change to the chapter or the sim since the 2026-05-20 polish is one
commit, `e79f828` (2026-05-29, "humanize: surgical de-AI prose pass"). Its
touch on `ch08_offline_rl/tex/offline_rl.tex` is 6 insertions / 6 deletions,
all cosmetic (colon→semicolon, colon→period, one deleted filler sentence
"The agent forecasts what to do, and the forecast is the action."). No number,
no footnote claim, no citation changed. The sim script
`offline_rl_pricing.py` has NOT been touched since `719243f` (2026-05-19, the
Phase 2 recovery); all cached numbers and generated artifacts predate the
delta. The three deliberately-deferred reimplementations (IQL-argmax,
BCQ-D, fused-token DT) and their disclosure footnotes are the standing open
question this cycle checks.

## Files read (end to end)

- `/Users/pranjal/Code/rl/ch08_offline_rl/sims/offline_rl_pricing.py`
- `/Users/pranjal/Code/rl/ch08_offline_rl/sims/offline_rl_pricing_stdout.txt` (all 73 lines)
- `/Users/pranjal/Code/rl/ch08_offline_rl/sims/offline_rl_pricing_results.tex`
- `/Users/pranjal/Code/rl/ch08_offline_rl/sims/offline_rl_pricing_bandit.tex`
- `/Users/pranjal/Code/rl/ch08_offline_rl/sims/offline_rl_pricing_coverage.png` (viewed)
- `/Users/pranjal/Code/rl/ch08_offline_rl/tex/offline_rl.tex` (sim subsection + all method subsections + footnotes)
- `/Users/pranjal/Code/rl/docs/refs.bib` (bibkey existence checks)
- Prior audits: `.../audits/..._2026-05-19.md`, `.../..._polish_2026-05-20.md`, `.../sims/offline_rl_pricing_audit.md`
- `git show e79f828`, `git log` on script + tex

## Step 3 — what this sim is evidence FOR (in my own words)

(i) **Theoretical claim of the chapter part.** The chapter advances the
*pessimism principle*: in offline RL, naive value learning (FQI) suffers an
overestimation cascade under distributional shift, and pessimism mechanisms
(CQL's conservative penalty, IQL's expectile-V, BCQ's action constraint) exist
to suppress out-of-distribution Q-values. The §sec:dt_rvs subsection adds a
second thesis: return-conditioned supervised learning (DT, RvS) is an
alternative to pessimism that treats the policy as a supervised model rather
than a conservative value function.

(ii) **What the sim is used as evidence for.** Two propositions. First, that
FQI without any pessimism mechanism collapses (24.7% of DP optimal) while
pessimism methods stay near 92% — the overestimation cascade is real and
pessimism fixes it. Second (the honest twist the rebalanced experiment now
carries), that on a *near-on-policy* dataset with a near-optimal behavioral,
pure imitation (BC 96.8%) and supervised-conditioning (RvS 97.0%, DT 96.3%)
beat the pessimism family, because pessimism pays a robustness tax that only
returns value when coverage degrades — which the coverage sweep (Figure) then
demonstrates by collapsing BCQ-D to 25.6% at ε_b=0.9 while CQL/IQL hold ~92%.

## Criteria verdicts

### (a) CORRECTNESS — PASS (with disclosed simplifications, all accurate)

Every trained method implements what its (re)label names, and the three
relabels have footnote disclosures that match the code line-for-line:

- **IQL-argmax** (`train_iql`, lines 568–611). Expectile-V regression is
  present and correct: `diff = q_vals - v_vals; weight = where(diff>0, IQL_TAU,
  1-IQL_TAU); v_loss = (weight*diff**2).mean()` (lines 587–589), τ=0.7. Q fit
  to `r + V(s')` (line 595), never `max_a Q`. Policy extraction is
  `argmax_a Q(s,a)` (lines 604–610), NOT advantage-weighted regression. The
  footnote (tex line 85) discloses exactly this: "expectile-V learning step
  follows Kostrikov2022 verbatim ... policy-extraction ... replaced with
  argmax_a Q(s,a)." Accurate.
- **BCQ-D** (`train_bcq`, lines 614–672). Trains an MLP behavioral classifier
  (cross-entropy), then thresholds admissible actions at
  `bc_probs >= BCQ_THRESHOLD * max_prob` (lines 645–646, 664), τ=0.3. This is
  the discrete BCQ-D of Fujimoto2019b, not the continuous VAE+perturbation
  BCQ. Footnote (tex line 105) discloses this precisely, and — resolving the
  prior cycle's flag — `\citet{Fujimoto2019b}` now exists in `refs.bib`
  (line 2206), so the citation resolves.
- **DT (fused-token)** (`DecisionTransformer.forward`, lines 425–436). Each
  timestep is one token summing return+state+prev-action+positional embeddings
  (lines 428–431), with a causal mask (lines 433–434); the head predicts a_t.
  This is T tokens/trajectory, not Chen2021's 3T. Footnote (tex line 126)
  discloses the fused-token simplification and the 3× context shrink. Accurate.

Unchanged internals spot-checked: CQL penalty `logsumexp(Q) - Q_data` with
α=0.1 and Polyak target (lines 549–556) is the Kumar2020 CQL(H) form; FQI is
the plain `max_a Q(s',a)` backup with no target network (lines 491–506),
disclosed as a deliberate pedagogical choice (tex footnote line 147). Results
are consistent with offline-RL theory: FQI (no pessimism) collapses, the three
pessimism methods hold ~92%, and BCQ-D's hard threshold goes vacuous under a
near-uniform behavioral (25.6% at ε_b=0.9) — textbook behavior.

### (b) PRESENTATION / NUMBERS — PASS (published artifacts fully consistent)

Every published number traces to `stdout` and to `main_results`/`coverage_results`:

| Method | results.tex | stdout main table | figure |
|---|---|---|---|
| DP Oracle | 192.41 ± 0.33 / 100.0% | 192.41 / 100.0% | axhline @100 |
| RvS | 186.58 ± 0.34 / 97.0% | 186.58 / 97.0% | 96.7→86.7 |
| BC | 186.28 ± 0.31 / 96.8% | 186.28 / 96.8% | 96.8→95.4 |
| DT | 185.27 ± 0.33 / 96.3% | 185.27 / 96.3% | 96.7→85.5 |
| CQL | 178.08 ± 1.48 / 92.6% | 178.08 / 92.6% | ~92 flat |
| BCQ-D | 177.05 ± 0.73 / 92.0% | 177.05 / 92.0% | 92.3→25.6 |
| IQL-argmax | 176.67 ± 0.81 / 91.8% | 176.67 / 91.8% | ~92 flat |
| FQI | 47.48 ± 8.42 / 24.7% | 47.48 / 24.7% | 16.7/27.4/25.4 |

Main table is rank-ordered by mean descending (DP first); figure legend is in
rank order. Coverage-figure lines match the stdout coverage table exactly (I
verified all 7×3 cells against the PNG). Prose numbers all trace:
R⋆≈184 = `dp_init_val` 184.47 (line 3 stdout); BC/BCQ-D/DT/RvS 96.8/92.0/96.3/97.0
(line 158); FQI "17%–27%" band = 16.7/27.4/25.4. `169.27` appears once, only in
the disclosure footnote (line 147) documenting the retired collapse. No stale
live number survives. The DP-value-at-d=1 (184.47) differing from the empirical
DP-eval mean (192.41) is legitimate: eval averages over a uniform initial
demand regime while 184.47 is the single-regime backward-induction value; the
prose only ever quotes 184 as the DT/RvS conditioning target, which is correct.

### (c) CHAPTER FIT — PASS on the stated thesis, one taxonomy contradiction

The sim demonstrates both step-3 propositions: FQI's 24.7% collapse and the
pessimism family's ~92% hold validate the overestimation-cascade / pessimism
story; the coverage sweep validates "pessimism's advantage emerges only as the
behavioral degrades" (BCQ-D collapse at ε_b=0.9). One internal contradiction
degrades the fit (see Finding 1): the setup sentence and the figure caption
group FQI *into* "the pessimism family," while three other passages state FQI
has no pessimism mechanism.

### (d) EFFICIENCY / STANDARDS — PASS

Per-component `compute_or_load` caching with config decomposition
(BC_CONFIG ⊂ FQI_CONFIG ⊂ {CQL,IQL,BCQ}_CONFIG; DT/RvS separate), CONFIG_VERSION=14,
flags via `add_component_args` (`--data-only`, `--plots-only`, per-component
force). N_SEEDS=20 (≥10), mean and SE=std/√N reported for every method. Fixed
seeds; dataset RNG `seed`, eval RNG `seed+10000`, coverage RNG `seed+20000/30000`,
identical across methods (fair comparison confirmed). Palette from
`sims.plot_style`. One cosmetic ding: `stdout.txt` is a concatenation of an
interrupted run and a completing run (cache-hit lines interleave a tqdm bar at
line 31), so the file carries redundant progress noise — the final summary
tables (lines 53–73) are clean and correct.

## 7-point checklist

1. **Algorithm identity** — PASS. All 7 methods match their labels; 3 disclosed
   simplifications verified against code and against resolvable bibkeys.
2. **Environment/MDP fidelity** — PASS. State (i,d,t), Poisson demand
   λ₀=(1.5,3,5,8)·e^{−0.15p}, r=p·min(Q,i), −2.00 salvage, 4-state chain diag
   0.6 — all match tex lines 147.
3. **Data integrity** — PASS. Table/stdout read `r['mean']`, `r['se']` from
   `compute_data`; nothing hardcoded. Numbers reproduce across results.tex ↔
   stdout ↔ figure.
4. **Comparison fairness** — PASS. Same per-seed dataset and same eval RNG
   across all methods. Note: DT/RvS receive R⋆=V*(s0) (an oracle scalar) as the
   conditioning target that BC does not get; standard DT protocol, disclosed.
5. **Theoretical sanity** — PASS. FQI < pessimism ~92% < BC/DT/RvS; BCQ-D
   threshold goes vacuous under diffuse behavioral; nothing beats the oracle.
6. **Information leakage** — PASS. Training consumes only offline tensors; no
   `dp_policy`/`dp_value`/`step` in any update. R⋆ scalar is a disclosed
   deployment hyperparameter, bounded leakage at most.
7. **Seed/reproducibility** — PASS. 20 seeds, SE reported. Minor: `np.random.seed`
   set in train_dt/train_rvs but Q-methods rely on `torch.manual_seed` only —
   reproducible, inconsistent (carried from prior audit, unchanged).

## Prior-audit open-item disposition

- **Four-way 169.27 collapse (was 50%)** — RESOLVED. Phase 2 state-dependent
  behavioral (`BEHAVIORAL_MARKUPS=[5,7,8,9]`, config v14) differentiates the
  four methods (186.58/186.28/185.27/177.05). `169.27` now lives only in the
  disclosure footnote (tex line 147). Confirmed.
- **BCQ misattribution / Fujimoto2019b missing bibkey (was 25%)** — RESOLVED.
  `@article{Fujimoto2019b,` present at `refs.bib:2206`; footnote line 105 cites
  it. Citation resolves.
- **IQL argmax vs AWR (was 25%)** — RESOLVED via disclosure. Label IQL-argmax,
  footnote line 85, matches code.
- **DT fused-token vs Chen2021 (was 25%)** — RESOLVED via disclosure. Footnote
  line 126, matches code.
- **DT/RvS R⋆ sensitivity not swept** — STILL OPEN (deferred). R⋆≈184 is named
  in prose (line 158); no sweep. Disclosed, not resolved.
- **FQI collapse not diagnosed vs target-net ablation** — STILL OPEN. Disclosed
  as pedagogical (footnote line 147); no ablation. Acceptable but open.
- **"Broader coverage worsens FQI" mechanism (was 25%)** — STILL OPEN, and now
  in visible tension with the sim's own coverage sweep (see Finding 2).

## Findings (severity-ordered)

**1. FQI is simultaneously in and not-in "the pessimism family" (internal
contradiction).** Tex line 147: "The pessimism family appears as FQI, CQL,
IQL-argmax, and BCQ-D." Figure caption line 165: "the pessimism family (FQI,
CQL, IQL-argmax, BCQ-D)." But line 156: "the pessimism family (CQL, BCQ-D,
IQL-argmax)" and "FQI, with no pessimism mechanism at all"; line 160: FQI
"without any mechanism to control extrapolation error"; line 65: FQI "does not
include any explicit pessimism mechanism." A referee reading the setup and the
figure caption is told FQI is a pessimism method, then told three times it is
not. Terminological, no number affected, but a clean Reviewer-2 catch. Fix:
call the value-based group the "value-learning family" (FQI as the no-pessimism
control within it) in lines 147 and 165, or drop FQI from the pessimism list in
both.

**2. The line-156 FQI mechanism claim is contradicted by the coverage sweep.**
Line 156 says FQI "suffers an overestimation cascade made worse by the broader
action coverage: more state-action pairs ... means more opportunities for the
unconstrained max ... to select an overestimated out-of-distribution Q-value."
Broader coverage means *fewer* OOD actions, so standard theory predicts it
should help FQI — and the sim's own Figure agrees: FQI rises 16.7→27.4 as ε_b
goes 0.05→0.3 (more coverage → better). Line 169 then hedges correctly
("uniformly catastrophic ... regardless of dataset breadth"). The two FQI
explanations are in tension and line 156's causal direction is the dubious one.
This was flagged at 25% in the 2026-05-19 audit and survives.

**3. Orphan artifact `offline_rl_pricing_bandit.tex` (stale, committed,
unconsumed).** The file is tracked in git (last touched `4cdbac6`, mtime
2026-03-16), is `\input` by NO tex file (grep across the repo returns nothing),
and is produced by NO current script (no `.py` writes it; no ch08 script
mentions "IPW"/"Direct Regression"). Its contents describe a contextual-bandit
off-policy-evaluation experiment (methods: FQI, CQL, Direct Regression, IPW;
DP Oracle = 10.64) with no relationship to the current 7-method MDP sim
(DP Oracle = 192.41). It is also not rank-ordered (IPW 96.3% is listed last),
but that is moot because it is unpublished. Risk: a landmine — a later `\input`
would silently pull wrong-context numbers into the chapter. Recommend deleting
it or moving it out of `sims/`. Not a defect in the shipped paper.

**4. (minor) DT/RvS get the oracle return V*(s0) as their conditioning target;
BC does not.** Standard DT/RvS deployment and disclosed (R⋆≈184, tex line 158),
but a hostile reader can note the supervised-conditioning family receives a
scalar of privileged information the imitation baseline lacks. Bounded, low
severity, no fix required beyond the existing disclosure.

**5. (minor) stdout hygiene.** `offline_rl_pricing_stdout.txt` concatenates an
interrupted run with the completing run; progress noise interleaves. Final
summary tables are clean and correct. Cosmetic.

## Bottom line

The delta (one cosmetic humanize commit) introduced no numeric or claim
regression. The standing question — are the three disclosure footnotes present
and accurate against the shipped code — resolves cleanly: all three are present,
line-for-line accurate, and cite bibkeys that now exist. Every published number
is mutually consistent across script, stdout, table, figure, and prose, and the
table/figure are rank-ordered. What a hostile referee still catches: FQI listed
in the pessimism family in the setup/caption but declared pessimism-free
elsewhere (Finding 1), and the line-156 "more coverage hurts FQI" claim that the
chapter's own Figure contradicts (Finding 2). Both are prose-level, catchable,
and survivable in revision; neither falsifies a result and neither makes a
named method differ from the implemented method.

**Bullshit score: 25%** — Reviewer 2 catches the FQI-in-vs-out-of-the-pessimism-family contradiction and the "broader coverage worsens FQI" claim that the coverage figure rebuts; the substance (correct method identities with accurate disclosures, fully consistent rank-ordered numbers) survives revision.
