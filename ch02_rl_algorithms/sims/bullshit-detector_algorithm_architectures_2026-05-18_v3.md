# bullshit-detector — algorithm_architectures (v3 recheck) — 2026-05-18

**Bullshit score: 25%** — Diagram-only cap (25%) pinned. The two v2 regressions (#12 δ_t↔Critic overlap; #13 missing env loop in panel c) were both addressed, but the v3 fix introduces new structural mismatches: panel (c) is now visibly denser than (a)/(b), the env-loop labels in (c) are repositioned to coordinates that sit on top of the new dashed feedback arrows, and the relocated TD-error formula no longer reads as belonging to a panel (c) sub-region — it floats below the Environment block at x=4.2 while the Environment is centered at x=2.5. Two residual dilutions from v1/v2 ("low-variance" without visual referent; "three core" exhaustiveness) remain unchanged.

## Header
- Claim sources:
  - `/Users/pranjal/Code/rl/ch02_rl_algorithms/tex/rl_algorithms.tex` (figure caption, line 198)
  - Compiled chapter PDF: `/Users/pranjal/Code/rl/docs/ch02_rl_algorithms.pdf` (Figure 1, page 11; caption text confirmed via `pdftotext -layout -f 11 -l 11`)
- Code / artifact root: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.py`
- Rendered artifact: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.png` (MD5 `b50a64c357becc4013a2423bf3dfd711`, 228,499 B, mtime 2026-05-18 17:40)
- Stdout: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures_stdout.txt`
- Seed audits:
  - v1 (20%): `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/bullshit-detector_algorithm_architectures_2026-05-18.md`
  - v2 (25%, cap-pinned): `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/bullshit-detector_algorithm_architectures_2026-05-18_v2.md`
- Run by: bullshit-detector (Opus 4.7, 1M)
- Date: 2026-05-18
- Diagram-only cap applied: yes (25% ceiling, stated per Pass 6 rule)

## Delta from prior audits (v1 → v2 → v3)

| Finding | v1 | v2 | v3 | Note |
|---|---|---|---|---|
| 1 — DQN forward path | HOLDS | HOLDS | HOLDS | Unchanged. |
| 2 — REINFORCE forward path | HOLDS | HOLDS | HOLDS | Unchanged. |
| 3 — Actor-Critic separate networks | HOLDS | HOLDS | HOLDS | Unchanged. |
| 4 — "low-variance" smuggled comparison | DILUTED | DILUTED (downgraded) | DILUTED | Caption still asserts variance comparison the figure does not depict. Panel (b) has no Monte-Carlo return node, no `G_t`. The caption-only anchor introduced in v2 ("REINFORCE's Monte-Carlo return") remains unsupported by the figure. |
| 5 — "three core algorithm families" exhaustiveness | DILUTED | DILUTED (softened) | DILUTED | "three core algorithm families" unchanged from v2. Still excludes model-based, distributional, evolutionary. |
| 6 — TD-error formula correctness | HOLDS | HOLDS | HOLDS | Formula text unchanged; position relocated (see Finding 15). |
| 7 — caption acknowledges acting/training split | — | HOLDS (prose) | partially regressed (caption rewritten) | Caption rewritten in v3: removed "Panels (a) and (b) depict… panel (c) additionally shows…" — replaced with "the critic's TD error δ_t feeds back into both during training" and a new sentence "Each panel includes the environment feedback loop…". The (a)/(b)-vs-(c) acknowledgment is gone; the "every panel has env loop" is now an explicit caption claim. This creates new pressure on panel (c) to match (see Finding 13). |
| 8 — critic update channel missing | DILUTED | HOLDS (with caveat) | HOLDS | δ_t→Critic dashed arrow still present (`algorithm_architectures.py:312-315`). |
| 9 — π/π_θ notation drift | MISLABELED | HOLDS | HOLDS | Unchanged. |
| 10 — stdout matches prints | HOLDS | HOLDS | HOLDS | Unchanged. |
| 11 — --data-only / --plots-only interface | HOLDS | HOLDS | HOLDS | Unchanged. |
| 12 — δ_t↔Critic visual overlap | — | DILUTED (new) | HOLDS (fixed) | v3 reroutes: `critic→td` solid with `curve=0.30` (bows up), `td→critic` dashed with `curve=0.30` (bows down, since direction is reversed). Rendered PNG: the two arcs are visually distinct, one above the line and one below. Hostile-reader objection on bidirectional ambiguity no longer holds. |
| 13 — panel (c) missing env loop | — | DILUTED (new) | partially fixed → DILUTED (residual) | Env box, a_t→Env, Env→s_t curves now drawn in `draw_actor_critic` (`algorithm_architectures.py:317-339`). But the curves use `curve=-0.55` (vs `+0.4` in panels a/b); the labels are at y=0.5 (vs y=0.35 in a/b); and the env-loop arcs sweep through regions that collide with the new dashed δ_t arrows and the actor-critic stack. See Findings 13, 16. |
| 14 — PNG reproducibility | — | HOLDS | HOLDS | Confirmed: re-ran script twice; both runs produce `b50a64c357becc4013a2423bf3dfd711`. Determinism preserved. |
| 15 — NEW: TD formula relocated; visual association weakened | — | — | DILUTED (new) | Formula moved from below δ_t (y_critic − 0.45 ≈ 0.55) to y=−0.75, below the Environment block. Formula x-position is `nodes['td']['xy'][0]` = 4.2, but the Environment box is centered at x=2.5. The formula now floats at (4.2, −0.75) — below the actor-critic right column, but to the right of the Environment box. A reader scanning the panel can plausibly read the formula as a caption for the Environment rather than for the δ_t / TD-error region. |
| 16 — NEW: Env-loop labels in panel (c) repositioned to collide with new arrows | — | — | DILUTED (new) | Labels moved from (4.3, 0.35) / (0.65, 0.35) in panels a/b to (4.6, 0.5) / (0.4, 0.5) in panel c. Label `a_t` at (4.6, 0.5) sits to the right of δ_t (4.2, 1.0) and overlaps the visual region of both the `td→critic` dashed arc and the a_t→Env curved gray arrow. Label `r_{t+1}, s_{t+1}` at (0.4, 0.5) sits in the region traversed by the Env→s_t gray curve. |
| 17 — NEW: Print legibility at \textwidth | — | — | DILUTED (new) | Compiled PDF page 11 renders the three-panel figure at `\textwidth`. The rendered figure spans the full page width but is vertically compressed (`figsize=FIG_WIDE`); panel (c) sub-elements (δ_t→critic dashed arc, δ_t→actor dashed arc, env-loop labels, TD formula) compete for ~3.5 vertical inches of paper space. The compiled image is visibly busier than panels (a)/(b) at print scale. |

Net delta v2→v3: two v2 regressions closed (#12, #13 partially); three new structural dilutions introduced (#15 formula float, #16 label-on-arrow collision, #17 print clutter); two residual dilutions from v1/v2 unchanged (#4, #5). Without the 25% diagram-only cap, this would land in the 30–40% band — same band as v2, different specific failures.

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---|---|---|---|
| 1 | "(a) DQN maps states to Q-values for all actions, selecting the argmax." | HOLDS | — | no |
| 2 | "(b) REINFORCE maps states to a probability distribution over actions, then samples." | HOLDS | — | no |
| 3 | "(c) Actor-Critic maintains separate policy and value networks." | HOLDS | — | no |
| 4 | "yielding lower-variance policy updates than REINFORCE's Monte-Carlo return." | DILUTED | LOW | no (caption only) |
| 5 | "three core algorithm families" (definite-article exhaustiveness) | DILUTED | LOW | no |
| 6 | TD-error formula correct | HOLDS | — | no |
| 7 | Caption "Each panel includes the environment feedback loop…" | HOLDS (prose) | — | no — but see Findings 13, 16 |
| 8 | δ_t feeds back to both actor and critic | HOLDS | — | no |
| 9 | π_θ notation consistency in panel (b) | HOLDS | — | no |
| 10 | Stdout matches prints | HOLDS | — | no |
| 11 | --data-only / --plots-only interface | HOLDS | — | no |
| 12 | δ_t↔Critic arrows visually distinct (v2 regression fixed) | HOLDS | — | no |
| 13 | Panel (c) env loop present but uses different curvature / label positions than (a)/(b) | DILUTED | LOW | no |
| 14 | PNG hash reproducible across two consecutive runs | HOLDS | — | no |
| 15 | TD formula relocation produces visual disassociation from δ_t | DILUTED | LOW | no |
| 16 | Env-loop labels in panel (c) collide with new dashed arcs / curved gray arrows | DILUTED | LOW | no |
| 17 | Panel (c) cluttered at print scale (\textwidth on PDF p.11) | DILUTED | LOW | no |

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
- **Data evidence:** PNG panel (a): `s_t → Q(s,·;θ) → arg max_a → a_t*`.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 2: "(b) REINFORCE maps states to a probability distribution over actions, then samples."

- **Claim source (verbatim):** "(b) REINFORCE maps states to a probability distribution over actions, then samples." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['policy']['xy'], nodes['policy']['size'],
                 r'$\pi_\theta(a|s)$', color=col, fontsize=12)
  draw_rect_node(ax, nodes['sample']['xy'], nodes['sample']['size'],
                 r'$a \sim \pi_\theta(\cdot|s)$', color=col, alpha=0.08, fontsize=10)
  ```
  `algorithm_architectures.py:235-238`
- **Data evidence:** PNG panel (b): `s_t → π_θ(a|s) → a ~ π_θ(·|s) → a_t`.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 3: "(c) Actor-Critic maintains separate policy and value networks."

- **Claim source (verbatim):** "(c) Actor-Critic maintains separate policy and value networks" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):** `algorithm_architectures.py:287-291` — two distinct `draw_rect_node` calls for actor and critic.
- **Data evidence:** PNG panel (c): "Actor π_θ(a|s)" at y=2.5; "Critic V_w(s)" at y=1.0. Two stacked boxes.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 4: "yielding lower-variance policy updates than REINFORCE's Monte-Carlo return."

- **Claim source (verbatim):** "yielding lower-variance policy updates than REINFORCE's Monte-Carlo return." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```bash
  $ grep -nE "G_t|Monte|return" ch02_rl_algorithms/sims/algorithm_architectures.py
  # (no matches)
  ```
  Panel (b) `draw_reinforce` has only the forward path `s → π_θ → sample → a` plus the env loop. There is no `G_t` node, no return-weighted gradient arrow, no Monte-Carlo branching.
- **Data evidence:** Rendered PNG panel (b) shows no training-time signal. The caption's "REINFORCE's Monte-Carlo return" is unanchored in the figure.
- **Category:** DILUTED — caption invites a comparison the figure cannot ground.
- **Severity:** LOW (caption-only)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "G_t" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read() and "Monte" not in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "lower-variance" not in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read().split("\\caption")[1].split("\\label")[0] or "G_t" in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read()
  ```

### Finding 5: "three core algorithm families" (exhaustiveness over-claim)

- **Claim source (verbatim):** "Architecture comparison of three core algorithm families: value-based, policy gradient, and actor-critic." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_dqn(axes[0])
  draw_reinforce(axes[1])
  draw_actor_critic(axes[2])
  ```
  `algorithm_architectures.py:363-365`
- **Data evidence:** Three panels, three families. The phrasing "three core algorithm families" reads as a closed partition; model-based RL, distributional RL, evolutionary policy search are excluded.
- **Category:** DILUTED — adversarial reading of "core" as exhaustive remains available.
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant:**
  ```python
  assert "three core algorithm families" in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read()
  ```
- **Honest-fix pass condition:**
  ```python
  cap = open("ch02_rl_algorithms/tex/rl_algorithms.tex").read().split("\\caption{")[1].split("}")[0]; assert ("three core algorithm families" not in cap) and ("three model-free" in cap or "three of the" in cap or "three classical" in cap)
  ```

### Finding 6: TD-error formula correctness

- **Claim source (verbatim):** Formula rendered: `δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)` — `ch02_rl_algorithms/sims/algorithm_architectures.py:343`
- **Code evidence (verbatim):**
  ```python
  ax.text(nodes['td']['xy'][0], -0.75,
          r'$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$',
          ha='center', va='top', fontsize=9, color=col,
          fontstyle='italic')
  ```
  `algorithm_architectures.py:342-345`
- **Data evidence:** PNG shows formula in italics in the panel-(c) bottom region. Symbol-level math matches Sutton-Barto §13.5 (one-step TD(0)).
- **Category:** HOLDS (mathematical correctness)
- **Severity:** —
- **Result-changing:** no
- Caveat: positioning is split into Finding 15 below.

### Finding 7: Caption "Each panel includes the environment feedback loop…"

- **Claim source (verbatim):** "Each panel includes the environment feedback loop in which the agent's action produces a reward and next state." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # draw_dqn:        env_xy = (2.5, -0.15) ... curve=0.4   (lines 192-209)
  # draw_reinforce:  env_xy = (2.5, -0.15) ... curve=0.4   (lines 245-264)
  # draw_actor_critic: env_xy = (2.5, -0.15) ... curve=-0.55 (lines 317-339)
  ```
  All three `draw_*` functions construct an Environment rect and two curved gray edges. The literal caption claim ("each panel includes …") is satisfied.
- **Data evidence:** PNG: all three panels display an "Environment" box and gray a→env→s curves.
- **Category:** HOLDS (prose-side)
- **Severity:** —
- **Result-changing:** no
- Caveat: the *manner* in which panel (c) includes the env loop differs from (a)/(b) in three measurable ways — curvature sign (`-0.55` vs `+0.4`), curvature magnitude (`0.55` vs `0.40`), and label position (y=0.5 vs y=0.35). See Findings 13 and 16.

### Finding 8: δ_t feeds back to both Actor and Critic

- **Claim source (verbatim):** "the critic's TD error $\delta_t$ feeds back into both during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # Feedback: td -> actor (dashed) for policy gradient update
  draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'],
            p1_shape='circle', p2_shape='rect',
            p1_size=NODE_RADIUS, p2_size=nodes['actor']['size'],
            dashed=True, color=col, curve=-0.35)

  # Feedback: td -> critic (dashed, opposite-arc to the critic->td forward edge above)
  draw_edge(ax, nodes['td']['xy'], nodes['critic']['xy'],
            p1_shape='circle', p2_shape='rect',
            p1_size=NODE_RADIUS, p2_size=nodes['critic']['size'],
            dashed=True, color=col, curve=0.30)
  ```
  `algorithm_architectures.py:306-315`
- **Data evidence:** PNG shows two dashed purple arrows from δ_t: one to Actor (upward), one to Critic (downward-left).
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 9: π_θ notation consistency in panel (b)

- **Claim source:** Implicit — within-panel consistency.
- **Code evidence:** `algorithm_architectures.py:236, 238` both use `\pi_\theta`.
- **Data evidence:** PNG panel (b) shows `π_θ(a|s)` and `a ~ π_θ(·|s)`.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 10: stdout matches print statements

- **Claim source:** Implicit (project convention).
- **Code evidence (verbatim):**
  ```python
  print(f"Output: {os.path.abspath(outpath)}")
  print("Algorithm architectures diagram generated.")
  ```
  `algorithm_architectures.py:374-375`
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

- **Claim source (verbatim):** "Diagram-only scripts (no Monte Carlo simulation) skip caching but still accept `--data-only` (exits with message) and `--plots-only` (runs normally) for interface consistency with the runner." — `/Users/pranjal/Code/rl/CLAUDE.md`
- **Code evidence:** `algorithm_architectures.py:379-388`. Standard pattern.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 12: δ_t↔Critic visual overlap (v2 regression fixed)

- **Claim source (verbatim):** "the critic's TD error $\delta_t$ feeds back into both during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # critic -> td (curved above) and td -> critic (dashed, curves below)
  # Opposite-direction arcs prevent the bidirectional pair from collapsing onto a single line
  _connect(ax, nodes, 'critic', 'td', curve=0.30)
  ...
  draw_edge(ax, nodes['td']['xy'], nodes['critic']['xy'],
            ...
            dashed=True, color=col, curve=0.30)
  ```
  `algorithm_architectures.py:301-315`
- **Data evidence:** Per matplotlib `arc3,rad=k` semantics, positive `curve` bends left of the line p1→p2. For `critic→td` (left→right, p1=(2,1.0), p2=(4.2,1.0)), `curve=+0.30` bows upward. For `td→critic` (right→left, p1=(4.2,1.0), p2=(2,1.0)), `curve=+0.30` bows downward (because direction is reversed). Rendered PNG confirms: solid black `critic→td` arrow arcs above the horizontal; dashed purple `td→critic` arrow arcs below. The pair is now visibly bidirectional and not collapsed.
- **Category:** HOLDS (regression closed)
- **Severity:** —
- **Result-changing:** no

### Finding 13: Panel (c) env loop drawn, but with different curvature/positioning than (a)/(b)

- **Claim source (verbatim):** "Each panel includes the environment feedback loop in which the agent's action produces a reward and next state." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):** Compare across panels:
  ```python
  # Panel (a): draw_dqn  (lines 200-209)
  draw_edge(ax, nodes['a']['xy'], env_xy, ..., curve=0.4, color=COLORS['gray'])
  draw_edge(ax, env_xy, nodes['s']['xy'], ..., curve=0.4, color=COLORS['gray'])

  # Panel (b): draw_reinforce  (lines 251-259)
  # ... identical to panel (a), curve=0.4

  # Panel (c): draw_actor_critic  (lines 324-333)
  draw_edge(ax, nodes['a']['xy'], env_xy, ..., curve=-0.55, color=COLORS['gray'])
  draw_edge(ax, env_xy, nodes['s']['xy'], ..., curve=-0.55, color=COLORS['gray'])
  ```
  Panels (a)/(b) use `+0.4`; panel (c) uses `-0.55`. Curvature sign AND magnitude differ. Panel (c)'s `a_t` is at y=2.5 (top row), so a curve down to env at y=−0.15 requires a much larger sweep than in (a)/(b) where `a_t` is at y=1.5.
- **Data evidence:** PNG: panels (a)/(b) show env curves that sweep gently below the main row. Panel (c) shows env curves that sweep WIDELY OUTSIDE the right side of the panel (the a_t→Env curve goes well past x=5) and WIDELY OUTSIDE the left side (Env→s_t curve goes near x=−0.5). The visual is asymmetric across the three panels.
- **Category:** DILUTED — the literal "env loop present" claim holds (Finding 7), but the rendered figure shows panel (c) using a substantively different env-loop geometry. A hostile reader will say "you said 'each panel'; you meant 'each panel, formally; but panel (c)'s sweep is different enough to read as inconsistent'."
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); assert "curve=-0.55" in src and src.count("curve=0.4") >= 2
  # PASSES on current code — proves panel (c) env-loop curvature differs in sign and magnitude from panels (a)/(b)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert "curve=0.4" in ac_block or "curve=-0.4" in ac_block
  # PASSES on honest fix (panel (c) env-loop curvature normalized to match panels (a)/(b) in magnitude); FAILS on current code
  ```

### Finding 14: PNG hash reproducibility across consecutive runs

- **Claim source:** Implicit — diagram-only script claimed to be reproducible (v1 and v2 reports both asserted no RNG).
- **Code evidence (verbatim):** `grep -E "random|seed|np\.random|rng" algorithm_architectures.py` → zero matches.
- **Data evidence:** Re-ran the script twice from a clean state. Both runs produced MD5 `b50a64c357becc4013a2423bf3dfd711`. Determinism preserved.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 15: TD-error formula relocation produces visual disassociation from δ_t

- **Claim source:** Implicit — figure caption refers to δ_t (panel c), so the TD formula and δ_t should read as a unit.
- **Code evidence (verbatim):**
  ```python
  # TD error formula below the env block to avoid collision with feedback arrows
  ax.text(nodes['td']['xy'][0], -0.75,
          r'$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$',
          ...)
  ```
  `algorithm_architectures.py:341-345`. The formula is placed at `(td.xy[0], -0.75)` = `(4.2, -0.75)`.
- **Data evidence:** In the rendered PNG, the formula appears in the bottom region of panel (c), to the right of and below the Environment box. The Environment box is centered at (2.5, −0.15) and extends roughly to x ∈ [1.7, 3.3]. The formula at x=4.2 sits OUTSIDE the Environment box's horizontal extent. The formula's vertical position (y=−0.75) is below both the Environment box and the env-loop labels — it is in the bottom strip of the panel. The formula is no longer directly underneath δ_t (which is at (4.2, 1.0)); there are ~1.75 vertical units of nodes/arrows between them. A reader scanning the panel can plausibly read the formula as floating "footnote-like" below everything, rather than as a label of δ_t.
- **Category:** DILUTED — the formula's mathematical content is correct (Finding 6); its visual association with δ_t has weakened.
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert "-0.75" in ac_block
  # PASSES on current code — proves formula is placed at y=-0.75, below the Environment block
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert ("y_critic - 0.45" in ac_block) or ("y_critic-0.45" in ac_block) or any(f"xy'][1] - 0.4" in ac_block for _ in [0])
  # PASSES on honest fix (formula returned to the δ_t-adjacent position, e.g., below the δ_t circle at y_critic - 0.45); FAILS on current code
  ```

### Finding 16: Env-loop labels in panel (c) collide with new dashed arcs / curved gray arrows

- **Claim source:** Implicit — figure should be legible.
- **Code evidence (verbatim):**
  ```python
  # Env-loop labels (match panels a and b)
  ax.text(4.6, 0.5, r'$a_t$', fontsize=9, color=COLORS['gray'],
          ha='center', va='center')
  ax.text(0.4, 0.5, r'$r_{t+1}, s_{t+1}$', fontsize=9, color=COLORS['gray'],
          ha='center', va='center')
  ```
  `algorithm_architectures.py:336-339`

  Compare to panels (a)/(b):
  ```python
  ax.text(4.3, 0.35, r'$a_t$', ...)
  ax.text(0.65, 0.35, r'$r_{t+1}, s_{t+1}$', ...)
  ```
  `algorithm_architectures.py:212-215, 261-264`
- **Data evidence:** Panel-(c) labels are at (4.6, 0.5) and (0.4, 0.5). The `a_t` label at (4.6, 0.5) sits to the right of δ_t (4.2, 1.0). The dashed `td→critic` feedback arrow runs from δ_t at (4.2, 1.0) DOWN-LEFT to Critic at (2.0, 1.0); its arc bows downward through approximately y ∈ [0.3, 1.0] in the x ∈ [2.0, 4.2] range. The `a_t→Env` gray curve runs from a_t at (4.2, 2.5) down to Env at (2.5, −0.15) with `curve=-0.55`, sweeping to the right of x=4.2 in its upper portion and back left toward the env block at the bottom. Visual inspection of the rendered PNG shows the gray `a_t` label sits in the visual neighborhood of multiple arrows. The `r_{t+1}, s_{t+1}` label at (0.4, 0.5) is similarly in the path of the gray Env→s_t curve. These labels were placed at y=0.5 to clear the Critic row at y=1.0, but the choice exposes them to the curved arrows now passing through that region.
- **Category:** DILUTED — the labels are readable in isolation, but a hostile reader scanning panel (c) at print scale will perceive the bottom-right region as crowded (δ_t, dashed td→critic, gray a_t→env arc, `a_t` label, formula all competing for the same area).
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert "(4.6, 0.5)" in ac_block and "(0.4, 0.5)" in ac_block
  # PASSES on current code — proves env-loop labels in panel (c) are at y=0.5, in the visual region of the dashed feedback arcs and curved env arrows
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert any(f"({x}, {y})" in ac_block for x in ["4.3", "4.4"] for y in ["0.2", "0.25", "0.3", "0.35"])
  # PASSES on honest fix (labels lowered toward y≈0.3 to match panels (a)/(b) and avoid collision); FAILS on current code
  ```

### Finding 17: Panel (c) cluttered at print scale on compiled PDF

- **Claim source:** Implicit — figure should be legible at the size at which it is rendered in the chapter.
- **Code evidence:** Figure is rendered at `\includegraphics[width=\textwidth]` (`ch02_rl_algorithms/tex/rl_algorithms.tex:197`). Source figure uses `figsize=FIG_WIDE` (from `sims/plot_style.py`).
- **Data evidence:** Compiled PDF page 11 (`docs/ch02_rl_algorithms.pdf`, extracted via `pdftoppm -r 200 -f 11 -l 11`) shows the three-panel figure spanning the page width. At this scale, panel (c) contains: s_t, Actor box, a_t, Critic box, δ_t, 3 forward arrows (s→actor, s→critic, actor→a, critic→td), 2 dashed feedback arrows (td→actor, td→critic), Environment box, 2 curved gray arrows (a→env, env→s), 2 gray labels (a_t, r_{t+1}/s_{t+1}), and the TD formula. Panels (a) and (b) each contain: 3 forward nodes/arrows in the main row plus Env loop. Panel (c) has roughly twice the element density of (a)/(b). At print scale, the panel (c) sub-elements (especially the dashed feedback arrows and the formula) are smaller and more cramped than the corresponding elements in (a)/(b). A reader at standard reading distance will see panel (c) as visibly denser.
- **Category:** DILUTED — the figure is technically all there, but the print-scale density of (c) relative to (a)/(b) creates visual asymmetry the caption does not warn about.
- **Severity:** LOW (cosmetic)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert ac_block.count("draw_edge") + ac_block.count("_connect") >= 7  # panel (c) draws ≥7 edges; panels (a)/(b) draw 5 each
  # PASSES on current code — proves panel (c) has substantially more edges than (a)/(b)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # Honest fix: either give panel (c) more horizontal space (taller figsize or different gridspec ratio) or trim a non-load-bearing element (e.g., move the TD formula into the caption).
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); assert "gridspec_kw" in src or "width_ratios" in src or "delta_t = r_t" not in src
  # PASSES on honest fix (panel-c gets its own width budget OR the TD formula is removed from the figure and moved to caption/prose); FAILS on current code
  ```

## Cross-cutting patterns

- The v3 patch closed both v2 regressions: the δ_t↔Critic overlap (v2-#12) is now visually distinct due to opposite-arc routing, and panel (c) now contains an Environment box and acting-time loop (v2-#13 partially closed). Both fixes are observable in the rendered PNG.
- The v3 patch trades the v2 regressions for three new structural dilutions: (1) panel (c)'s env loop uses curvature `curve=-0.55` vs panels (a)/(b)'s `curve=+0.4`, so the literal "each panel includes the env loop" caption is true but the panels look visibly inconsistent; (2) the env-loop labels in panel (c) were repositioned from (4.3, 0.35) / (0.65, 0.35) in (a)/(b) to (4.6, 0.5) / (0.4, 0.5) in (c), placing them in the visual region of the new dashed feedback arrows; (3) the TD-error formula was relocated from below δ_t (y_critic − 0.45) to y=−0.75, putting it below the Environment block at x=4.2 — outside the Environment block's horizontal extent.
- Two residual dilutions from v1 — "low-variance vs REINFORCE's Monte-Carlo return" (no visual referent in panel b) and "three core algorithm families" (definite-article exhaustiveness) — are unchanged in v3. Neither is touched by the panel (c) edits.
- The diagram-only cap (25%) is the binding constraint on the final score. Without the cap, the count of dilutions (4, 5, 13, 15, 16, 17) and their cumulative caption/figure-mismatch density would land this in the 30–40% band.
- The script is reproducible: re-running the script from a clean state twice produces the same PNG MD5 (`b50a64c357becc4013a2423bf3dfd711`). The v1 and v2 reports were correct that no RNG calls exist; v3 confirmation re-validates that property.
- Pattern across all three audit rounds: each editing cycle closes one or two specific findings but introduces new structural mismatches of similar severity elsewhere in the figure. The artifact is approaching a fixed point where literal caption claims hold (HOLDS) but the figure-vs-caption fidelity at the *implicit* level (panel symmetry, visual association, label placement, print legibility) keeps regressing under local edits. An honest exit from this loop requires either (a) accepting the diagram-only cap and ceasing edits, or (b) a global redesign of panel (c) layout rather than further local patches.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** 25% — diagram-only cap pinned, Reviewer-2 grade. Ship after touch-up if any further edit is performed, but the recommended action is to STOP local editing on this figure: every prior round has closed one issue and opened another of similar severity. If the user insists on another edit pass, treat it as a global redesign per pattern note above.
1. For each non-HOLDS finding (4, 5, 13, 15, 16, 17), turn the **violated invariant** into a pytest test under `tests/test_algorithm_architectures_caption.py`. Confirm each test PASSES on current code (proves the gap is real).
2. Convert each **honest-fix pass condition** into a paired test that FAILS on current code. The pairs are the red/green specs.
3. If further editing is approved, hand off to `writing-plans`:
   - Findings 4, 5: caption wording.
   - Finding 13: normalize panel-(c) env-loop curvature to match (a)/(b), or accept different geometry and update the caption to acknowledge it.
   - Finding 15: move the TD formula back below δ_t (e.g., `(td.xy[0], y_critic - 0.45)`), or remove it from the figure and place it in the prose.
   - Finding 16: lower the env-loop labels in panel (c) to y≈0.3 to match (a)/(b), or relocate them outside the curved-arrow region.
   - Finding 17: increase panel (c)'s horizontal width via `gridspec_kw={'width_ratios': [1, 1, 1.3]}`, or trim a non-load-bearing element (e.g., move TD formula out of the figure).
4. After fixes, re-render PNG. The MD5 will change for any figure edit (Findings 13, 15, 16, 17) but not for caption-only edits (Findings 4, 5).
5. Re-run this skill (v4). Target a score ≤10% on next pass, but note that the diagram-only cap caps the achievable score at 25% regardless; the meaningful target is to close all DILUTED findings, after which the report is pinned at the cap by definition.
