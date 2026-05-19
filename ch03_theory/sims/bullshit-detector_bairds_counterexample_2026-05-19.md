# bullshit-detector — bairds_counterexample — 2026-05-19

**Bullshit score: 35%** — Script implements the Sutton & Barto (2018) 7-state Example 11.2 with expected (not stochastic) updates, while the v3 tex describes the Baird (1995) 6-state MDP with a different value-function form (`V(6) = 2w_1 - w_6`). The MDP is the wrong one relative to the prose. Divergence is real and the Sutton & Barto setup is canonical in its own right, so the qualitative claim survives, but a hostile reviewer reading the tex and then the code will catch the swap.

## Header
- Claim sources:
  - `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex` (lines 256–283; primary)
  - `/Users/pranjal/Code/rl/ch03_theory/sims/bairds_counterexample.py` (docstring lines 1–5, 18)
- Code / artifact root: `/Users/pranjal/Code/rl/ch03_theory/sims/`
  - `bairds_counterexample.py` (292 lines)
  - `bairds_counterexample_stdout.txt` (52 lines, verified populated)
  - `bairds_counterexample.png` (4-panel figure, viewed)
- Seed audit (if any): none provided
- Run by: bullshit-detector skill (Opus 4.7)
- Date: 2026-05-19
- Diagram-only cap applied: no (Monte Carlo / numerical sim)

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Tex says "six-state star MDP"; script uses 7 states / 8 weights | DATA DRIFT | MED | no (qualitative divergence holds either way) |
| 2 | Tex value form `V(s)=2w_1+w_s`, `V(6)=2w_1-w_6` vs script `V(s)=2w_s+w_8`, `V(7)=w_7+2w_8` (sign + coefficient swap) | DATA DRIFT | MED | no (both forms produce divergence) |
| 3 | Tex says "Training samples all transitions equally often (uniform distribution)"; script uses deterministic expected-update matrix A, no sampling at all | DILUTED | MED | no (expected updates are the limit of uniform sampling, but the demo of *sampling-induced* divergence is absent) |
| 4 | Tex describes "shared weight w_1" pushed up by 5 visits vs 1; script's stdout never reports this 5:1 mechanism, only spectral radius | HOLDS | — | no |
| 5 | Divergence claim (`‖V‖→∞`) in panel 1 | HOLDS | — | no |
| 6 | No `np.random.seed`; no stochastic component anywhere | HOLDS (deterministic by construction) | — | no |
| 7 | Stdout "expected 18.50 / 10.0747 / 8.79" hardcoded as ground-truth without source — internal note reference "(Section 5a)" not findable in tex | DATA DRIFT | LOW | no |
| 8 | TDC and L2 fixes presented as Baird-applicable; tex section 11.7 resolutions match | HOLDS | — | no |
| 9 | Script docstring says "personal understanding, not for the paper" (line 3) | DILUTED | LOW | no (intent disclosure, but figure still uses chapter plot style and lives in chapter sims/) |

## Findings

### Finding 1: MDP topology disagreement — 6 states (tex) vs 7 states (code)

- **Claim source (verbatim):**
  > "\citet{Baird1995} constructed a six-state star MDP that makes this failure concrete. ... The MDP has a star topology: states 1 through 5 each transition to state 6, and state 6 transitions to itself."
  >
  > — `ch03_theory/tex/planning_learning_v3.tex:280`

- **Code evidence (verbatim):**
  ```python
  N_STATES = 7
  N_WEIGHTS = 8
  GAMMA = 0.99
  ALPHA = 0.01
  N_EPOCHS = 1000


  def make_features():
      """Feature vectors for 7-state Baird's star MDP."""
      X = np.zeros((N_STATES, N_WEIGHTS))
      for i in range(6):          # states 1-6 (0-indexed: 0-5)
          X[i, i] = 2.0          # own weight with coefficient 2
          X[i, 7] = 1.0          # shared weight w_8 with coefficient 1
      X[6, 6] = 1.0              # state 7: own weight w_7
      X[6, 7] = 2.0              # state 7: shared weight w_8 with coefficient 2
      return X
  ```
  `ch03_theory/sims/bairds_counterexample.py:20-35`

  Script docstring line 18: `# Common setup: Sutton & Barto (2018) Example 11.2`

- **Data evidence:** stdout reports 8 eigenvalues (lambda_1 through lambda_8), confirming 8 weights:
  > "lambda_1 = +0.2393 <-- unstable ... lambda_8 = -0.5714"
  > — `ch03_theory/sims/bairds_counterexample_stdout.txt:9-16`

- **Category:** DATA DRIFT — tex describes Baird (1995) 6-state original; script implements the Sutton & Barto (2018) Example 11.2 7-state variant. Both are legitimate "Baird" MDPs in the literature, but they are not the same MDP.

- **Severity:** MED — does not change the qualitative divergence claim. A reviewer comparing tex line 280 to the figure caption will see the script reports lambda_1..lambda_8 (eight eigenvalues) for what the tex insists is a six-state, six-weight MDP. That contradiction is on the page.

- **Result-changing:** no — both setups diverge under semi-gradient off-policy TD.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert bairds_counterexample.N_STATES == 7 and bairds_counterexample.N_WEIGHTS == 8
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert bairds_counterexample.N_STATES == 6 and bairds_counterexample.N_WEIGHTS == 6  # match tex line 280
  ```
  (Alternative honest fix: edit tex line 280 to say "seven-state Sutton & Barto variant"; in that case the invariant becomes a tex grep.)

### Finding 2: Value-function form disagreement — sign and coefficient swap

- **Claim source (verbatim):**
  > "Linear function approximation uses a shared weight $w_1$ across all states plus a state-specific weight, with $V(s) = 2w_1 + w_s$ for $s \in \{1,\ldots,5\}$ and $V(6) = 2w_1 - w_6$."
  >
  > — `ch03_theory/tex/planning_learning_v3.tex:280`

- **Code evidence (verbatim):** Per the feature matrix above:
  - For states 0..5 (i.e., 1..6 in 1-indexed): `X[i, i] = 2.0`, `X[i, 7] = 1.0` ⇒ `V(s) = 2 w_{s} + w_8`.
  - For state 6 (state 7): `X[6, 6] = 1.0`, `X[6, 7] = 2.0` ⇒ `V(7) = w_7 + 2 w_8`.

  Two mismatches with the tex:
  1. Tex has *coefficient 2 on the shared weight* and *coefficient 1 on the state-specific weight* for ordinary states; code has coefficient 2 on the state-specific weight and coefficient 1 on the shared weight. The coefficients are swapped.
  2. Tex has a **minus sign** on the state-specific weight at the absorbing state: `V(6) = 2w_1 - w_6`. Code has *no minus sign*: `V(7) = w_7 + 2 w_8`. Tex's `-w_6` is absent from the implementation.

- **Data evidence:** stdout line 49: `V(7) at epoch 0: 21.00 (expected 21.00)` — consistent with `1·1 + 2·10 = 21` (code's form), not with tex's `2·1 - 1 = 1` form. The "expected 21.00" anchor confirms the code form is the one tested.

- **Category:** DATA DRIFT (could be promoted to FALSE if the tex form is taken as canonical). The code does not implement the value function the tex describes.

- **Severity:** MED — the tex's `V(6) = 2w_1 - w_6` with shared-coefficient 2 is the *Baird 1995* form. The script's `V(7) = w_7 + 2 w_8` is the *Sutton & Barto 2018* form. Both diverge under uniform off-policy semi-gradient TD; the demo's qualitative payload survives. But the specific numbers in the figure and stdout cannot be reconciled with the equations on tex line 280.

- **Result-changing:** no — qualitative divergence holds for both forms. The numerical trajectories differ, but the published claim is "diverges to infinity," which both satisfy.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert make_features()[6, 6] == 1.0 and make_features()[6, 7] == 2.0  # no minus sign anywhere in code
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert make_features()[5, 5] == -1.0  # tex form: V(6) = 2 w_1 - w_6 requires negative entry
  ```
  (Alternative: change tex line 280 to match Sutton & Barto 2018 notation. Either side must move.)

### Finding 3: "Uniform sampling" claim vs deterministic expected-update implementation

- **Claim source (verbatim):**
  > "Training samples all transitions equally often (uniform distribution, not $d^\pi$)."
  >
  > — `ch03_theory/tex/planning_learning_v3.tex:280`

  > "Under behavior policy b (dashed 6/7, solid 1/7) and target policy pi (solid always), only solid transitions contribute. The IS ratio b(solid)*rho = (1/7)*7 = 1 cancels, leaving effective weight d(s) = 1/7."
  >
  > — `ch03_theory/sims/bairds_counterexample.py:47-51` (docstring)

- **Code evidence (verbatim):**
  ```python
  def make_A(X):
      x7 = X[6]
      A = np.zeros((N_WEIGHTS, N_WEIGHTS))
      for s in range(N_STATES):
          A += np.outer(X[s], GAMMA * x7 - X[s])
      A /= N_STATES
      return A

  def run_semigradient(X, n_epochs):
      """Standard semi-gradient TD with IS correction. Expected updates."""
      w = make_w0()
      A = make_A(X)
      history = [w.copy()]
      for _ in range(n_epochs):
          w = w + ALPHA * A @ w
          history.append(w.copy())
      return history
  ```
  `ch03_theory/sims/bairds_counterexample.py:45-78`

- **Data evidence:** No `np.random.seed`, no `np.random.choice`, no sampling loop anywhere in the file. `grep "seed\|np.random" bairds_counterexample.py` returns nothing. The "uniform sampling" demo is replaced by a deterministic linear iteration `w_{t+1} = (I + α A) w_t`.

- **Category:** DILUTED — uniform-sample expectations *are* the population limit of uniform sampling, and the divergence claim in the tex is about that limit. But the tex sentence "Training samples all transitions equally often" describes a stochastic process. The script demonstrates the deterministic mean-field, not the sampling regime. A hostile reviewer can argue: "you claim sampling causes divergence; you never sampled."

- **Severity:** MED — the script is honest in its docstring (`"Expected updates"` on line 71), but the tex prose around the figure does not warn the reader that the figure shows the expected-update fixed-point iteration, not Monte Carlo trajectories. The Sutton & Barto canonical Example 11.2 figure also uses expected updates, so this is defensible, but it is not what the tex sentence promises.

- **Result-changing:** no — divergence direction and magnitude in the expected-update setting bound the sample-average behavior.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "np.random" not in open("bairds_counterexample.py").read()  # no stochastic sampling
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # either: implement stochastic uniform sampling
  assert "np.random.choice" in inspect.getsource(bairds_counterexample.run_semigradient)
  # or: amend the tex to say "the expected update (limit of uniform sampling)"
  ```

### Finding 4: Unsourced "expected" anchors in stdout verification block

- **Claim source (verbatim):**
  ```
  Verification (Section 5a):
    w_8 at epoch 0:   10.0000  (expected 10.0000)
    w_8 at epoch 1:   10.0747  (expected 10.0747)
    w_8 at epoch 100: 18.50  (expected ~18.50)
    V(1) at epoch 0:  12.00  (expected 12.00)
    V(7) at epoch 0:  21.00  (expected 21.00)
    delta(1) epoch 0: 8.79  (expected 8.79)
  ```
  `ch03_theory/sims/bairds_counterexample_stdout.txt:44-50`, generated by `bairds_counterexample.py:250-261`.

- **Code evidence (verbatim):**
  ```python
  print(f'\nVerification (Section 5a):')
  print(f'  w_8 at epoch 0:   {w8_sg[0]:.4f}  (expected 10.0000)')
  print(f'  w_8 at epoch 1:   {w8_sg[1]:.4f}  (expected 10.0747)')
  print(f'  w_8 at epoch 100: {w8_sg[100]:.2f}  (expected ~18.50)')
  ```
  `ch03_theory/sims/bairds_counterexample.py:250-254`

- **Data evidence:** "Section 5a" appears nowhere in `planning_learning_v3.tex`:
  ```
  $ grep -n "Section 5a\|5a" ch03_theory/tex/planning_learning_v3.tex
  (no matches)
  ```
  These hardcoded "expected" values therefore reference notes the reader cannot access. The values `10.0747` and `18.50` and `8.79` are *outputs of this same code*; they are not independent oracles. The "verification" prints values against itself.

- **Category:** DATA DRIFT — the verification anchors point at a "Section 5a" that does not exist in the cited tex; the comparison is therefore vacuous.

- **Severity:** LOW — the docstring on line 3 says "personal understanding, not for the paper," so the verification block is private to the author. But the stdout file is checked into the repo and the figure renders into the chapter PDF.

- **Result-changing:** no.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Section 5a" in inspect.getsource(bairds_counterexample)  # references nonexistent section
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # remove the unsourced anchors, or replace with citation to a verifiable source
  assert "Section 5a" not in inspect.getsource(bairds_counterexample)
  ```

### Finding 5: Divergence figure check — does it really diverge?

- **Claim source (verbatim):** Tex line 280: "The shared weight diverges to $+\infty$."

- **Code/Data evidence:** stdout line 31:
  > "Semi-gradient TD             10.00      18.50      84.28      335.84"

  And max|V| (line 39):
  > "Semi-gradient TD             21.00      37.95     168.99      669.47"

  Spectral radius of `I + αA` is 1.002393 (stdout line 18). Growth rate (1.002393)^1000 ≈ 10.95. Empirically max|V| grows from 21 to 669, a factor of ~32 over 1000 iterations. Per-step factor ≈ 32^(1/1000) ≈ 1.00346, somewhat above the 1.002393 spectral radius (because the unstable mode has eigenvalue 0.2393, giving (1+0.01·0.2393)^1000 ≈ 11; the empirical 32 reflects nonzero projection onto faster initial-transient modes plus the second unstable eigenvalue 0.0222).

  The PNG (panel 1, top-left, red trace) shows monotonic super-linear growth from ~21 to ~670 over 1000 iterations with no plateau. Divergence is visible.

- **Category:** HOLDS — divergence claim is faithfully demonstrated.

### Finding 6: Reproducibility — deterministic by construction

- **Claim source:** None explicit; sim standards in `CLAUDE.md` require seeds and ≥10 seeds with means/SEs.

- **Code evidence:** No `np.random.seed`, no stochastic step, no seed loop. Two consecutive runs produce bit-identical output (deterministic linear algebra).

- **Category:** HOLDS for reproducibility (deterministic).
  Side-note: the `CLAUDE.md` directive "Run each method across multiple seeds (minimum 10) and report means and standard errors" does not apply because there is no stochastic component to average over — but this is also why Finding 3 (no actual sampling) lands as DILUTED. The deterministic shortcut and the missing sampling demo are two sides of the same choice.

### Finding 7: Author intent disclosure (line 3) does not exempt the artifact

- **Claim source (verbatim):**
  > '"""'
  > "Baird's Counterexample: Divergence and Three Fixes"
  > "Chapter 3 (Theory) — personal understanding, not for the paper."
  >
  > — `ch03_theory/sims/bairds_counterexample.py:1-3`

- **Code evidence:** The script lives in `ch03_theory/sims/` (the canonical chapter sim directory), uses the centralized plot style (`from sims.plot_style import apply_style, COLORS`), produces `bairds_counterexample.png` in the chapter sim directory, and has a populated `_stdout.txt` — i.e., it is structurally indistinguishable from publishable chapter sims.

- **Category:** DILUTED — the disclaimer is not load-bearing because the artifact is shipped as if it were chapter-grade. If a reader opens the chapter PDF and sees a Baird figure cross-referenced from §11.7, the disclaimer in line 3 of the script is invisible.

- **Severity:** LOW — currently no `\includegraphics{.../bairds_counterexample.png}` line was found in v3 tex (the deadly-triad section references `deadly_triad_geometry.png` instead). So the figure is not yet included in the paper. If/when it is, severity rises to MED.

- **Result-changing:** no, contingent on the figure not entering the PDF.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "bairds_counterexample.png" not in open("ch03_theory/tex/planning_learning_v3.tex").read()
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # either remove the disclaimer (commit to chapter inclusion), or move the script out of ch03_theory/sims/
  assert "personal understanding, not for the paper" not in open("ch03_theory/sims/bairds_counterexample.py").read()
  ```

## Cross-cutting patterns

- The single biggest source of friction is **MDP-variant drift between tex and code**: the v3 tex describes Baird (1995) 6-state with `V(6) = 2w_1 - w_6`; the script implements Sutton & Barto (2018) Example 11.2 7-state with no minus sign. Findings 1 and 2 are different facets of the same mismatch. A hostile reviewer reading the tex and then opening the figure will catch this in under a minute. Pick a canonical variant and align both sides — either (a) rewrite tex line 280 in Sutton & Barto 2018 notation (seven states, eight weights, no minus sign), or (b) reimplement the script as the 6-state Baird 1995 form. The drift, not divergence, is the audit's central object.
- **"Expected updates" vs "uniform sampling" terminology drift** (Finding 3): the tex sentence says samples, the script computes the population limit. This is not wrong per se — it is the cleanest way to show divergence — but the prose needs to acknowledge it. One footnote on tex line 280 saying "Figure 3.X plots the expected update $w_{t+1} = (I+\alpha A) w_t$, the population limit of uniform Monte Carlo sampling" closes this finding.
- **Unsourced verification anchors** (Finding 4) and **the "not for the paper" disclaimer** (Finding 7) are symptoms of an artifact in flux. Decide whether this sim is paper-grade or scratch-grade and commit. If paper-grade, drop the disclaimer, replace "Section 5a" with an actual `\ref{}`, and include the figure in the chapter. If scratch-grade, move the script to `ch03_theory/sims/scratch/` and exclude from the chapter sim runner.
- **No FALSE / UNIMPLEMENTED findings.** The divergence claim, the three fixes (fitted VI, TDC, L2), the spectral-radius analysis, and the figure all check out. The sim demonstrates the deadly triad faithfully; the issues are notational and definitional rather than mechanistic.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** 35% is below the 50% halt threshold. Do not stop forward work, but address Findings 1 and 2 before this artifact enters the chapter PDF.
1. For each non-HOLDS finding (1, 2, 3, 4, 7), write the violated-invariant pytest assertion as a test under `ch03_theory/sims/tests/test_bairds.py`. Confirm each PASSES on current code.
2. Convert each honest-fix pass condition into a second pytest test that FAILS on current code.
3. Decision point: align tex to code (Sutton & Barto 2018 7-state) or align code to tex (Baird 1995 6-state). Recommend tex → S&B 2018 because (a) the script already uses that variant cleanly, (b) Sutton & Barto 2018 is the canonical pedagogical reference, and (c) the figure already exists and reproduces the standard textbook result. Two tex edits cover it: change "six-state" → "seven-state" and update the value function form.
4. Add a footnote to tex line 280 clarifying that the figure shows expected (mean-field) updates, not stochastic samples (closes Finding 3).
5. Remove or replace the "Section 5a" verification anchors with a defensible source (closes Finding 4); strip the "personal understanding, not for the paper" disclaimer if and only if the figure is going into the chapter PDF (closes Finding 7).
6. Re-run the sim. Re-run this skill. Expected bullshit score after fixes: 5–10% (one residual DATA DRIFT at most).
