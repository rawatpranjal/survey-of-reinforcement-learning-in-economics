# bullshit-detector — algorithm_architectures — 2026-05-18

**Bullshit score: 20%** — Reviewer 2 catches the panel-(b)/panel-(c) training-signal asymmetry and the unanchored "low-variance" comparative claim, but every panel's literal caption assertion is honored by the drawing. Diagram-only cap (25%) applies.

## Header
- Claim sources:
  - `/Users/pranjal/Code/rl/ch02_rl_algorithms/tex/rl_algorithms.tex` (figure caption, lines 195–200)
  - Surrounding chapter prose, lines 148–193 (DQN, TRPO/PPO, SAC subsections that frame the figure)
- Code / artifact root: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.py`
- Rendered artifact: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.png` (MD5 fd6eae79d670c83afb61270802282b32 — author-verified reproducible)
- Stdout: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures_stdout.txt`
- Seed audit (if any): None
- Run by: bullshit-detector subagent (Opus 4.7)
- Date: 2026-05-18
- Diagram-only cap applied: yes (25% ceiling)

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---|---|---|---|
| 1 | "(a) DQN maps states to Q-values for all actions, selecting the argmax." | HOLDS | — | no |
| 2 | "(b) REINFORCE maps states to a probability distribution over actions, then samples." | HOLDS | — | no |
| 3 | "(c) Actor-Critic maintains separate policy and value networks." | HOLDS | — | no |
| 4 | "(c) the critic's TD error δ_t provides a low-variance learning signal to the actor." | DILUTED | LOW | no (figure caption only) |
| 5 | "Architecture comparison of the three fundamental algorithm families." | DILUTED | LOW | no |
| 6 | TD-error formula `δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)` rendered correctly. | HOLDS | — | no |
| 7 | Training-signal depiction is asymmetric: panel (c) shows δ_t→actor feedback; panel (b) omits the REINFORCE return-weighted-log-likelihood gradient. | DILUTED | LOW | no |
| 8 | Panel (c) shows the TD signal feeding back to the actor but not back to the critic (critic→δ_t is one-way). | DILUTED | LOW | no |
| 9 | Notation drift inside panel (b): policy node labelled `π_θ(a|s)` but sampling node labelled `a ~ π(·|s)` (θ subscript dropped). | MISLABELED | LOW | no |
| 10 | Stdout file matches the script's `print` statements verbatim. | HOLDS | — | no |
| 11 | `--data-only` exits with a message, `--plots-only` runs normally — interface-consistency rule for diagram-only scripts honored. | HOLDS | — | no |
| 12 | No `np.random` / seed / RNG calls in script (reproducibility by construction). | HOLDS | — | no |

## Findings

### Finding 1: "(a) DQN maps states to Q-values for all actions, selecting the argmax."

- **Claim source (verbatim):** "(a) DQN maps states to Q-values for all actions, selecting the argmax." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_circle_node(ax, nodes['s']['xy'], r'$s_t$')
  draw_rect_node(ax, nodes['qnet']['xy'], nodes['qnet']['size'],
                 r'$Q(s,\cdot\,;\theta)$', color=col, fontsize=12)
  draw_rect_node(ax, nodes['argmax']['xy'], nodes['argmax']['size'],
                 r'$\arg\max_a$', color=col, alpha=0.08, fontsize=11)
  draw_circle_node(ax, nodes['a']['xy'], r'$a_t^*$')

  _connect(ax, nodes, 's', 'qnet')
  _connect(ax, nodes, 'qnet', 'argmax')
  _connect(ax, nodes, 'argmax', 'a')
  ```
  `algorithm_architectures.py:181-190`
- **Data evidence:** PNG visually shows `s_t → Q(s,·;θ) → arg max_a → a_t*` (left panel). The `·` placeholder ranges over actions, supplying "Q-values for all actions"; the argmax node supplies "selecting the argmax."
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no
- **Violated invariant:** N/A (HOLDS)
- **Honest-fix pass condition:** N/A (HOLDS)

### Finding 2: "(b) REINFORCE maps states to a probability distribution over actions, then samples."

- **Claim source (verbatim):** "(b) REINFORCE maps states to a probability distribution over actions, then samples." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['policy']['xy'], nodes['policy']['size'],
                 r'$\pi_\theta(a|s)$', color=col, fontsize=12)
  draw_rect_node(ax, nodes['sample']['xy'], nodes['sample']['size'],
                 r'$a \sim \pi(\cdot|s)$', color=col, alpha=0.08, fontsize=10)
  draw_circle_node(ax, nodes['a']['xy'], r'$a_t$')

  _connect(ax, nodes, 's', 'policy')
  _connect(ax, nodes, 'policy', 'sample')
  _connect(ax, nodes, 'sample', 'a')
  ```
  `algorithm_architectures.py:235-243`
- **Data evidence:** PNG middle panel shows `s_t → π_θ(a|s) → a ~ π(·|s) → a_t`. Forward path matches the caption verb-for-verb.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no
- **Violated invariant:** N/A
- **Honest-fix pass condition:** N/A

### Finding 3: "(c) Actor-Critic maintains separate policy and value networks."

- **Claim source (verbatim):** "(c) Actor-Critic maintains separate policy and value networks; the critic's TD error $\delta_t$ provides a low-variance learning signal to the actor." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['actor']['xy'], nodes['actor']['size'],
                 r'Actor $\pi_\theta(a|s)$', color=col, fontsize=11)
  draw_circle_node(ax, nodes['a']['xy'], r'$a_t$')
  draw_rect_node(ax, nodes['critic']['xy'], nodes['critic']['size'],
                 r'Critic $V_w(s)$', color=col, fontsize=11)
  draw_circle_node(ax, nodes['td']['xy'], r'$\delta_t$')
  ```
  `algorithm_architectures.py:287-292`
- **Data evidence:** PNG right panel shows two distinct rect nodes labelled "Actor π_θ(a|s)" and "Critic V_w(s)". Separate networks claim holds.
- **Category:** HOLDS (for the "separate policy and value networks" half; the "low-variance" half is split into Finding 4)
- **Severity:** —
- **Result-changing:** no

### Finding 4: "the critic's TD error δ_t provides a low-variance learning signal to the actor."

- **Claim source (verbatim):** "the critic's TD error $\delta_t$ provides a low-variance learning signal to the actor." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # Feedback: td -> actor (dashed, curved upward)
  draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'],
            p1_shape='circle', p2_shape='rect',
            p1_size=NODE_RADIUS, p2_size=nodes['actor']['size'],
            dashed=True, color=col, curve=-0.35)
  ```
  `algorithm_architectures.py:305-308`
- **Data evidence:** Dashed arrow from δ_t to Actor is rendered. The "δ_t goes to the actor" half is depicted. The "low-variance" qualifier is comparative; the comparison baseline (REINFORCE's Monte-Carlo return) is not drawn anywhere in the figure. The figure cannot ground the variance comparison; only the prose can.
- **Category:** DILUTED — half of the claim ("δ_t feeds the actor") is in the figure; the comparative half ("low variance") is smuggled in by the caption with no visual referent. The caption sells the figure as making the comparison the figure does not make.
- **Severity:** LOW (caption-only; no published number depends on this)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "MC" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read() and "Monte" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read()
  # PASSES on current code (no MC-return baseline drawn) — proves the comparative claim has no visual anchor
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "low-variance" not in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read().split("\\caption")[1].split("\\label")[0]
  # PASSES on honest fix (caption no longer asserts variance comparison the figure does not show); FAILS on current caption
  ```

### Finding 5: "Architecture comparison of the three fundamental algorithm families."

- **Claim source (verbatim):** "Architecture comparison of the three fundamental algorithm families." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_dqn(axes[0])
  draw_reinforce(axes[1])
  draw_actor_critic(axes[2])
  ```
  `algorithm_architectures.py:332-334`
- **Data evidence:** Three panels, three families. "Fundamental" is interpretive language (value-based, policy-gradient, actor-critic are textbook categories — Sutton-Barto Ch.13). The figure draws three; the caption claims they are "the" three fundamental families. The definite article is the dilution: model-based methods, distributional RL, model-free hybrids are not included; the caption excludes them by phrasing.
- **Category:** DILUTED — adversarial reading: "the three fundamental" promises an exhaustive partition the figure does not deliver. Charitable reading: this is the standard model-free trichotomy. Pick the more severe per skill voice rules.
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "the three fundamental algorithm families" in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read()
  # PASSES on current text — proves the definite-article exhaustiveness claim is in the caption
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "three model-free algorithm families" in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read() or "three classical algorithm families" in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read()
  # PASSES on honest fix (caption no longer over-claims exhaustiveness); FAILS on current caption
  ```

### Finding 6: TD-error formula correctness

- **Claim source (verbatim):** Formula rendered in figure: `δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)` — `ch02_rl_algorithms/sims/algorithm_architectures.py:312`
- **Code evidence (verbatim):**
  ```python
  ax.text(nodes['td']['xy'][0], nodes['td']['xy'][1] - 0.45,
          r'$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$',
          ha='center', va='top', fontsize=9, color=col,
          fontstyle='italic')
  ```
  `algorithm_architectures.py:311-314`
- **Data evidence:** PNG shows the formula in italics below δ_t. Standard one-step TD(0) error for a V-based critic. Matches Sutton-Barto §13.5 (one-step actor-critic). Matches the variable names of the critic node (V_w).
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 7: Training-signal asymmetry between panels (b) and (c)

- **Claim source (verbatim):** None explicit; the caption describes acting-time behavior for (a) and (b) and adds a training-time arrow only for (c).
- **Code evidence (verbatim):**
  ```python
  # Panel (b) — REINFORCE: only forward path, no return-weighted gradient arrow.
  _connect(ax, nodes, 's', 'policy')
  _connect(ax, nodes, 'policy', 'sample')
  _connect(ax, nodes, 'sample', 'a')
  ```
  `algorithm_architectures.py:241-243`
  versus
  ```python
  # Panel (c) — Actor-Critic: forward path plus δ_t -> actor dashed feedback arrow.
  draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'], ... dashed=True, ...)
  ```
  `algorithm_architectures.py:305-308`
- **Data evidence:** PNG: panel (b) has no dashed return-weighted-gradient arrow back to the policy; panel (c) does have a dashed arrow back to the actor. A reader comparing the two panels would conclude that actor-critic uses a learning signal and REINFORCE does not — REINFORCE's defining update is the return-weighted log-likelihood gradient `∇_θ log π_θ(a_t|s_t) · G_t` (Williams 1992), which is structurally analogous to the actor-critic update with G_t in place of δ_t. The figure visually erases this.
- **Category:** DILUTED — the figure systematically under-represents REINFORCE's training signal while showing actor-critic's. Defensible as a simplification (each panel could be read as acting-time only), but the asymmetry creates a false visual contrast.
- **Severity:** LOW (caption does not assert training-signal content for panel b; the figure is technically free to omit it, but the reader's expected comparison is the actor-critic ↔ REINFORCE training-signal difference, and the figure cannot support that comparison without showing both)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "G_t" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read() and "return" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().lower().split("def draw_reinforce")[1].split("def draw_actor_critic")[0]
  # PASSES on current code — proves REINFORCE panel has no return-based training arrow
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "G_t" in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_reinforce")[1].split("def draw_actor_critic")[0] or "_dashed_feedback" in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_reinforce")[1].split("def draw_actor_critic")[0]
  # PASSES on honest fix (REINFORCE panel gains a return-weighted training arrow analogous to (c)); FAILS on current code
  ```

### Finding 8: Actor-Critic critic update not shown

- **Claim source (verbatim):** None explicit. Caption asserts δ_t feeds the actor; says nothing about how the critic V_w updates.
- **Code evidence (verbatim):**
  ```python
  # critic -> td
  _connect(ax, nodes, 'critic', 'td')

  # Feedback: td -> actor (dashed, curved upward)
  draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'], ... dashed=True, ...)
  ```
  `algorithm_architectures.py:301-308`
- **Data evidence:** PNG shows `critic → δ_t` (forward) and `δ_t → actor` (dashed feedback). No `δ_t → critic` arrow. Standard A2C/A3C updates the critic by minimizing `||δ_t||^2` against the V_w parameters, so δ_t feeds back to both networks. The figure shows only the actor update channel.
- **Category:** DILUTED — half the training picture for actor-critic. A reviewer asking "how does the critic learn?" gets no answer from the figure.
- **Severity:** LOW (caption does not claim to show the full training graph; the omission is internal consistency only)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  ac_block = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert ac_block.count("'critic'") == 3  # critic node + s->critic + critic->td, no td->critic
  # PASSES on current code — proves no δ_t→critic edge is drawn
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  ac_block = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert "'td'" in ac_block and ac_block.count("'critic'") >= 4  # td->critic edge added
  # PASSES on honest fix (δ_t→critic dashed arrow added); FAILS on current code
  ```

### Finding 9: Notation drift inside panel (b)

- **Claim source (verbatim):** Panel-(b) node labels: policy box `$\pi_\theta(a|s)$`; sample box `$a \sim \pi(\cdot|s)$` (θ subscript dropped) — `ch02_rl_algorithms/sims/algorithm_architectures.py:236, 238`
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['policy']['xy'], nodes['policy']['size'],
                 r'$\pi_\theta(a|s)$', color=col, fontsize=12)
  draw_rect_node(ax, nodes['sample']['xy'], nodes['sample']['size'],
                 r'$a \sim \pi(\cdot|s)$', color=col, alpha=0.08, fontsize=10)
  ```
  `algorithm_architectures.py:235-238`
- **Data evidence:** PNG renders `π_θ(a|s)` and `π(·|s)` side by side. The same policy referenced with two different symbols within one panel.
- **Category:** MISLABELED — the parameterised policy and the sampling distribution are the same object; dropping θ in the second occurrence is purely cosmetic.
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  reinforce_block = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_reinforce")[1].split("def draw_actor_critic")[0]; assert r"\pi(\cdot|s)" in reinforce_block and r"\pi_\theta(\cdot|s)" not in reinforce_block
  # PASSES on current code — proves the θ-dropped form is used in the sample node
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  reinforce_block = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_reinforce")[1].split("def draw_actor_critic")[0]; assert r"\pi_\theta(\cdot|s)" in reinforce_block or r"\pi_\theta" in reinforce_block.split("a \\sim")[1].split(",", 1)[0]
  # PASSES on honest fix (θ subscript restored in sample node); FAILS on current code
  ```

### Finding 10: stdout matches print statements

- **Claim source:** Implicit (project convention requires `_stdout.txt` to capture script output verbatim).
- **Code evidence (verbatim):**
  ```python
  print(f"Output: {os.path.abspath(outpath)}")
  print("Algorithm architectures diagram generated.")
  ```
  `algorithm_architectures.py:343-344`
- **Data evidence:**
  ```
  Output: /Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.png
  Algorithm architectures diagram generated.
  ```
  `algorithm_architectures_stdout.txt:1-2`
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 11: `--data-only` / `--plots-only` interface compliance

- **Claim source (verbatim):** "Diagram-only scripts (no Monte Carlo simulation) skip caching but still accept `--data-only` (exits with message) and `--plots-only` (runs normally) for interface consistency with the runner." — `/Users/pranjal/Code/rl/CLAUDE.md:244`
- **Code evidence (verbatim):**
  ```python
  parser.add_argument('--data-only', action='store_true',
                      help='No computation to cache (diagram-only script)')
  parser.add_argument('--plots-only', action='store_true',
                      help='Runs normally (same as no flags)')
  args = parser.parse_args()
  if args.data_only:
      print("No computation to cache (diagram-only script).")
      sys.exit(0)
  generate_outputs()
  ```
  `algorithm_architectures.py:347-357`
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 12: Reproducibility — no RNG in script

- **Claim source:** Implicit; user-stated that PNG MD5 is stable across runs (fd6eae79d670c83afb61270802282b32).
- **Code evidence:** `grep -E "random|seed|np\.random|rng" algorithm_architectures.py` returns zero matches. No stochastic operation in the script; all node positions, sizes, colors are hard-coded constants.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

## Cross-cutting patterns

- Every literal caption assertion ("(a) ... argmax", "(b) ... samples", "(c) ... separate networks", TD formula) is grounded in code with `file:line`. The literal claims hold. The dilution clusters around the *comparative* claims the caption smuggles in but the figure does not visualise.
- Two dilution findings (Finding 4 "low-variance", Finding 7 "(b)/(c) asymmetry") share the same root: the figure depicts acting-time behavior for (a) and (b) but training-time behavior for (c), then asks the reader to make a variance comparison the figure cannot ground. Either show training arrows everywhere or claim only acting-time semantics everywhere. The current state is asymmetric.
- One dilution (Finding 5) and one mislabel (Finding 9) are pure caption / notation hygiene — no figure change required, only text edits.
- All cosmetic and notation issues are LOW severity. There are no FALSE / UNIMPLEMENTED / HIGH-severity findings. There are no DATA DRIFT findings (stdout matches the script's print calls; only one numeric artifact — the formula — and it is correct).
- The diagram-only cap (25%) constrains the score regardless of the dilution count: there is no Monte Carlo, no metric, no published number that the figure can falsify.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** 20% — Reviewer-2 grade. Ship after touch-up; address in response letter. Do not halt downstream work.
1. For each non-HOLDS finding (Findings 4, 5, 7, 8, 9), turn the **violated invariant** into a pytest test under `<repo>/tests/test_algorithm_architectures_caption.py`. Confirm each test PASSES on current code (proves the gap is real).
2. Convert each **honest-fix pass condition** into a paired test that FAILS on current code. The pairs are now the red/green specs for caption/figure edits.
3. Hand off to `writing-plans` (caption rewording for Findings 4, 5; figure edits for Findings 7, 8; one-line notation fix for Finding 9). Then `executing-plans`.
4. After fixes, the violated-invariant tests should FAIL and the pass-condition tests should PASS. Green state.
5. Re-render the PNG. Compare MD5 against the current `fd6eae79d670c83afb61270802282b32` — Findings 7/8/9 fixes WILL change the hash by design; Findings 4/5 are caption-only and will not. Re-run this skill if any fix is applied and confirm the new bullshit score is ≤10%.
