# bullshit-detector — algorithm_architectures (v2 recheck) — 2026-05-18

**Bullshit score: 25%** — Diagram-only cap pinned. Caption now anchors the variance comparison and acknowledges the (a)/(b)-vs-(c) acting/training split, but the figure introduces a new asymmetry the caption does not address (panels (a) and (b) draw an Environment feedback loop; panel (c) draws none), and the new dashed δ_t→Critic arrow visually overlaps the solid Critic→δ_t arrow tightly enough that a hostile reader can call panel (c) cluttered/ambiguous.

## Header
- Claim sources:
  - `/Users/pranjal/Code/rl/ch02_rl_algorithms/tex/rl_algorithms.tex` (figure caption, line 198)
  - Compiled chapter PDF figure page: `/Users/pranjal/Code/rl/docs/ch02_rl_algorithms.pdf` (Figure 1, page 11)
- Code / artifact root: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.py`
- Rendered artifact: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures.png` (MD5 `28c9fdeee4fd2a88a5a7e9a3b03deda8`, size 210,449 B, mtime 2026-05-18 17:27:32 — matches user-specified hash)
- Stdout: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/algorithm_architectures_stdout.txt`
- Seed audit: `/Users/pranjal/Code/rl/ch02_rl_algorithms/sims/bullshit-detector_algorithm_architectures_2026-05-18.md` (prior v1, scored 20%)
- Run by: bullshit-detector (Opus 4.7, 1M)
- Date: 2026-05-18
- Diagram-only cap applied: yes (25% ceiling, stated per Pass 6 rule)

## Delta from prior audit

| Prior # | Prior label | Prior category | Current category | Note |
|---|---|---|---|---|
| 1 | DQN forward path | HOLDS | HOLDS | Unchanged. |
| 2 | REINFORCE forward path | HOLDS | HOLDS | Unchanged. |
| 3 | Separate actor/critic nets | HOLDS | HOLDS | Unchanged. |
| 4 | "Low-variance" smuggled comparison | DILUTED | partially fixed → DILUTED (downgraded severity) | Caption now anchors the comparison to "REINFORCE's Monte-Carlo return" (`rl_algorithms.tex:198`), but the comparison still has no visual referent in panel (b); the figure cannot ground the variance claim. |
| 5 | "Three fundamental algorithm families" exhaustiveness | DILUTED | partially fixed → DILUTED (downgraded severity) | "fundamental" replaced with "core" (`rl_algorithms.tex:198`). The definite article "three core algorithm families" still suggests a partition that excludes model-based, distributional RL, etc. — softer claim, same shape. |
| 6 | TD-error formula | HOLDS | HOLDS | Unchanged. |
| 7 | Acting-time/training-time asymmetry between (a)/(b) and (c) | DILUTED | partially fixed → DILUTED | Caption now names the asymmetry ("Panels (a) and (b) depict the acting-time forward pass; panel (c) additionally shows…"). Acknowledgement is honest but creates a *new* internal contradiction (see Finding 13): (a)/(b) show the environment loop, (c) does not, so (c) does NOT additionally show acting-time content — it OMITS half of it. |
| 8 | Critic update channel missing | DILUTED | HOLDS (with caveat → see Finding 12) | Second dashed arrow from δ_t to Critic added (`algorithm_architectures.py:311-314`). Arrow renders in PNG. Caveat: visual overlap with solid Critic→δ_t arrow near δ_t node makes the bidirectional region cluttered. |
| 9 | π_θ / π notation drift | MISLABELED | HOLDS | Sample-node label now `\pi_\theta(\cdot|s)` (`algorithm_architectures.py:238`). Verified in source and in rendered PNG. |
| 10 | Stdout matches prints | HOLDS | HOLDS | Unchanged. |
| 11 | --data-only / --plots-only interface | HOLDS | HOLDS | Unchanged. |
| 12 | No RNG (reproducibility) | HOLDS | HOLDS | New MD5 `28c9fdeee4…` matches user spec; previous `fd6eae79d…` from v1 invalidated by design (figure changed). |
| — | NEW: Panel (c) lacks Environment loop while (a)/(b) include it | — | DILUTED (new regression) | See Finding 13. |
| — | NEW: δ_t↔Critic visual overlap on the right side of panel (c) | — | DILUTED (new regression) | See Finding 12. |

Net delta: two prior DILUTED items downgraded but not closed (4, 5); one prior DILUTED item closed conditionally (8); one MISLABELED closed (9); one prior DILUTED acknowledged in caption but converted into a new structural mismatch (7→13); one new clutter risk (12). The cap rule pins the score at 25%; absent the cap this would land in the 30–40% band.

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---|---|---|---|
| 1 | "(a) DQN maps states to Q-values for all actions, selecting the argmax." | HOLDS | — | no |
| 2 | "(b) REINFORCE maps states to a probability distribution over actions, then samples." | HOLDS | — | no |
| 3 | "(c) Actor-Critic maintains separate policy and value networks." | HOLDS | — | no |
| 4 | "yields lower-variance policy updates than REINFORCE's Monte-Carlo return." | DILUTED | LOW | no (caption only) |
| 5 | "three core algorithm families" (definite article exhaustiveness) | DILUTED | LOW | no |
| 6 | TD-error formula `δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)` | HOLDS | — | no |
| 7 | Caption acknowledges acting/training asymmetry between panels | HOLDS (prose-side) | — | no — but see #13 |
| 8 | δ_t feeds back to *both* actor and critic | HOLDS | — | no |
| 9 | Panel (b) θ-subscript notation consistent | HOLDS | — | no |
| 10 | Stdout matches `print` statements | HOLDS | — | no |
| 11 | `--data-only` / `--plots-only` interface compliance | HOLDS | — | no |
| 12 | Panel (c) δ_t↔Critic arrows visually overlap; bidirectional region ambiguous | DILUTED | LOW | no |
| 13 | Panel (c) omits the Environment forward loop that panels (a) and (b) include; "panel (c) additionally shows…" presupposes (c) is a superset of (a)/(b) | DILUTED | LOW | no |
| 14 | PNG hash matches user-stated `28c9fdeee4fd2a88a5a7e9a3b03deda8` | HOLDS | — | no |

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
- **Data evidence:** Rendered PNG (panel a, cropped) shows `s_t → Q(s,·;θ) → arg max_a → a_t*`.
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
  draw_circle_node(ax, nodes['a']['xy'], r'$a_t$')
  ```
  `algorithm_architectures.py:235-239`
- **Data evidence:** Rendered PNG (panel b, cropped) shows `s_t → π_θ(a|s) → a ~ π_θ(·|s) → a_t`. θ subscript now consistent in both nodes.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 3: "(c) Actor-Critic maintains separate policy and value networks."

- **Claim source (verbatim):** "(c) Actor-Critic maintains separate policy and value networks." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['actor']['xy'], nodes['actor']['size'],
                 r'Actor $\pi_\theta(a|s)$', color=col, fontsize=11)
  draw_rect_node(ax, nodes['critic']['xy'], nodes['critic']['size'],
                 r'Critic $V_w(s)$', color=col, fontsize=11)
  ```
  `algorithm_architectures.py:287-291`
- **Data evidence:** Rendered PNG (panel c, cropped) shows two distinct rect nodes labelled "Actor π_θ(a|s)" and "Critic V_w(s)".
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 4: "yields lower-variance policy updates than REINFORCE's Monte-Carlo return."

- **Claim source (verbatim):** "which yields lower-variance policy updates than REINFORCE's Monte-Carlo return." — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):** The caption now names a baseline ("REINFORCE's Monte-Carlo return") that does not appear in the figure. Searching the script:
  ```bash
  grep -nE "G_t|Monte|return" algorithm_architectures.py
  # (no matches)
  ```
  `algorithm_architectures.py:1-364`
- **Data evidence:** PNG panel (b) draws no `G_t` node, no `∇ log π · G_t` arrow, no training-time signal. The variance comparison is asserted in prose; the figure has no Monte-Carlo return to visually compare against.
- **Category:** DILUTED — caption now names the baseline (improvement over v1), but the named baseline is still not depicted; reader cannot ground the variance claim from the figure alone. Severity downgraded because the prose anchor is now explicit; not closed because the figure still cannot support the comparison.
- **Severity:** LOW
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
  `algorithm_architectures.py:338-340`
- **Data evidence:** Three panels, three families. "Three core algorithm families" reads as a closed partition; model-based RL, distributional RL, evolutionary policy search are excluded.
- **Category:** DILUTED — "fundamental" → "core" softens the claim but does not eliminate the definite-article exhaustiveness. Adversarial reading still flags it.
- **Severity:** LOW
- **Result-changing:** no
- **Violated invariant:**
  ```python
  assert "three core algorithm families" in open("ch02_rl_algorithms/tex/rl_algorithms.tex").read()
  ```
- **Honest-fix pass condition:**
  ```python
  cap = open("ch02_rl_algorithms/tex/rl_algorithms.tex").read().split("\\caption{")[2].split("}")[0]; assert ("three core algorithm families" not in cap) and ("three model-free" in cap or "three of the" in cap or "three classical" in cap)
  ```

### Finding 6: TD-error formula correctness

- **Claim source (verbatim):** Formula rendered: `δ_t = r_t + γ V_w(s_{t+1}) - V_w(s_t)` — `ch02_rl_algorithms/sims/algorithm_architectures.py:318`
- **Code evidence (verbatim):**
  ```python
  ax.text(nodes['td']['xy'][0], nodes['td']['xy'][1] - 0.45,
          r'$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$',
          ha='center', va='top', fontsize=9, color=col,
          fontstyle='italic')
  ```
  `algorithm_architectures.py:317-320`
- **Data evidence:** Rendered PNG shows formula in italics below δ_t. One-step TD(0) error, V-based critic. Matches Sutton-Barto §13.5.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 7: Caption now acknowledges acting-time vs training-time split

- **Claim source (verbatim):** "Panels (a) and (b) depict the acting-time forward pass; panel (c) additionally shows the critic's TD error $\delta_t$ feeding back to both networks during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):** Panel (b) has only forward arrows (`_connect 's'→'policy'→'sample'→'a'`); panel (c) has forward arrows (`s→actor`, `s→critic`, `actor→a`, `critic→td`) plus two dashed feedback arrows (`td→actor`, `td→critic`). `algorithm_architectures.py:241-243, 295-314`.
- **Data evidence:** PNG: panel (b) shows no training arrow; panel (c) shows two dashed feedback arrows from δ_t.
- **Category:** HOLDS (the caption sentence is technically accurate prose-side)
- **Severity:** —
- **Result-changing:** no — but see Finding 13: "additionally shows" presupposes (c) is a superset of (a)/(b) and the figure violates that presupposition along a different axis (environment loop).

### Finding 8: δ_t feeds back to both Actor and Critic

- **Claim source (verbatim):** "the critic's TD error $\delta_t$ feeding back to both networks during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # Feedback: td -> actor (dashed, curved upward) for policy gradient update
  draw_edge(ax, nodes['td']['xy'], nodes['actor']['xy'],
            p1_shape='circle', p2_shape='rect',
            p1_size=NODE_RADIUS, p2_size=nodes['actor']['size'],
            dashed=True, color=col, curve=-0.35)

  # Feedback: td -> critic (dashed, curved downward) for value update
  draw_edge(ax, nodes['td']['xy'], nodes['critic']['xy'],
            p1_shape='circle', p2_shape='rect',
            p1_size=NODE_RADIUS, p2_size=nodes['critic']['size'],
            dashed=True, color=col, curve=0.35)
  ```
  `algorithm_architectures.py:305-314`
- **Data evidence:** PNG (panel c) shows TWO dashed purple arrows from δ_t: one curves up to Actor; one curves down to Critic. Both have arrowheads on the network ends. The "feeding back to both networks" caption claim is visually grounded.
- **Category:** HOLDS (subject to the visual-clutter caveat raised in Finding 12)
- **Severity:** —
- **Result-changing:** no

### Finding 9: π_θ / π notation consistency restored

- **Claim source (verbatim):** Implicit — within-panel consistency in panel (b).
- **Code evidence (verbatim):**
  ```python
  draw_rect_node(ax, nodes['policy']['xy'], nodes['policy']['size'],
                 r'$\pi_\theta(a|s)$', color=col, fontsize=12)
  draw_rect_node(ax, nodes['sample']['xy'], nodes['sample']['size'],
                 r'$a \sim \pi_\theta(\cdot|s)$', color=col, alpha=0.08, fontsize=10)
  ```
  `algorithm_architectures.py:235-238`
- **Data evidence:** Rendered PNG (panel b) shows `π_θ(a|s)` and `a ~ π_θ(·|s)`. θ subscript present in both.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 10: stdout matches print statements

- **Claim source (verbatim):** Implicit (project convention requires `_stdout.txt` to capture script output verbatim).
- **Code evidence (verbatim):**
  ```python
  print(f"Output: {os.path.abspath(outpath)}")
  print("Algorithm architectures diagram generated.")
  ```
  `algorithm_architectures.py:349-350`
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
  `algorithm_architectures.py:353-363`
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

### Finding 12: NEW REGRESSION — panel (c) δ_t↔Critic visual overlap

- **Claim source (verbatim):** "the critic's TD error $\delta_t$ feeding back to both networks during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):**
  ```python
  # critic -> td  (solid forward)
  _connect(ax, nodes, 'critic', 'td')
  ...
  # Feedback: td -> critic (dashed, curved downward) for value update
  draw_edge(ax, nodes['td']['xy'], nodes['critic']['xy'],
            ... dashed=True, color=col, curve=0.35)
  ```
  `algorithm_architectures.py:302, 311-314`
- **Data evidence:** PNG panel (c) right-bottom region: the solid `Critic→δ_t` arrow and the dashed `δ_t→Critic` arrow run almost antiparallel along the same horizontal axis, separated by only the small `curve=0.35` bow of the dashed arrow. The two arrowheads land on opposite endpoints of the same near-segment. A hostile reader can read this as a single ambiguous bidirectional edge between Critic and δ_t rather than two distinct edges (one forward residual computation, one backward parameter update). The `td→actor` feedback uses `curve=-0.35` and is well separated from the solid `actor→a` edge (they are not antiparallel), so the actor side is clean; the critic side is where the overlap lives.
- **Category:** DILUTED — the second arrow is *present* (so Finding 8 holds), but its placement is close enough to the existing solid arrow that a worst-case reader will dismiss it as drawing noise or claim the diagram is misleading.
- **Severity:** LOW (cosmetic; no published number depends on it)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "curve=0.35" in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read() and "curve=-0.35" in open("ch02_rl_algorithms/sims/algorithm_architectures.py").read()
  # PASSES on current code; the two feedback arrows use symmetric ±0.35 curvature, which on the critic side overlaps the solid forward edge
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); assert "td_to_critic_offset" in src or any(c in src for c in ["curve=0.6", "curve=0.55", "curve=0.5"]) or "via=" in src or "y_offset" in src
  # PASSES on honest fix (the δ_t→critic arrow is routed with larger curvature, an explicit offset, or a routing waypoint so it does not visually overlap the solid critic→δ_t edge); FAILS on current code
  ```

### Finding 13: NEW REGRESSION — panel (c) omits the Environment loop that panels (a)/(b) include

- **Claim source (verbatim):** "Panels (a) and (b) depict the acting-time forward pass; panel (c) additionally shows the critic's TD error $\delta_t$ feeding back to both networks during training" — `ch02_rl_algorithms/tex/rl_algorithms.tex:198`
- **Code evidence (verbatim):** Panels (a) and (b) construct an Environment box and two curved gray arrows for the acting-time loop:
  ```python
  # In draw_dqn / draw_reinforce:
  env_xy = (2.5, -0.15); env_size = (1.6, 0.55)
  draw_rect_node(ax, env_xy, env_size, 'Environment', color=COLORS['gray'], alpha=0.10, fontsize=10)
  # a -> env, env -> s (curved)
  draw_edge(ax, nodes['a']['xy'], env_xy, ... curve=0.4, ...)
  draw_edge(ax, env_xy, nodes['s']['xy'], ... curve=0.4, ...)
  ```
  `algorithm_architectures.py:192-209, 245-264`. Panel (c)'s `draw_actor_critic` function (`algorithm_architectures.py:273-322`) contains zero references to `Environment`, `env_xy`, or any curved gray edge connecting `a_t` back to `s_t`.
- **Data evidence:** Rendered PNG: panel (a) shows an "Environment" box with curved gray arrows from `a_t* → Environment → s_t` (labels `a_t`, `r_{t+1}, s_{t+1}`). Panel (b) shows the same. Panel (c) shows NO Environment box, NO curved acting-time loop. The caption's phrase "panel (c) additionally shows" presupposes that (c) includes the content of (a)/(b) plus extra training-time arrows; the figure shows that (c) DROPS the acting-time loop and adds training-time arrows — a substitution, not an addition. A hostile reader will call this out: "(c) does not 'additionally show' anything; it shows a *different* slice of the graph."
- **Category:** DILUTED — caption presupposition not honored by figure. Defensible as a layout-space tradeoff (panel (c) is already busy with actor/critic/δ_t/formula), but the caption now actively misdescribes the relationship between the panels.
- **Severity:** LOW (figure caption only; no quantitative claim hinges on it)
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  ac_block = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read().split("def draw_actor_critic")[1].split("def generate_outputs")[0]; assert "Environment" not in ac_block
  # PASSES on current code (panel (c) draws no Environment box)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  src = open("ch02_rl_algorithms/sims/algorithm_architectures.py").read(); ac_block = src.split("def draw_actor_critic")[1].split("def generate_outputs")[0]; cap = open("ch02_rl_algorithms/tex/rl_algorithms.tex").read().split("\\caption{")[2].split("}")[0]; assert ("Environment" in ac_block) or ("additionally shows" not in cap)
  # PASSES on honest fix (either panel (c) gains an Environment loop, or the caption stops claiming (c) "additionally shows" content beyond (a)/(b)); FAILS on current code+caption pair
  ```

### Finding 14: PNG hash reproducibility

- **Claim source:** User-stated: PNG MD5 is `28c9fdeee4fd2a88a5a7e9a3b03deda8`.
- **Code evidence:** `grep -E "random|seed|np\.random|rng" algorithm_architectures.py` returns zero matches; node positions, sizes, colors are deterministic literals.
- **Data evidence:** `md5 algorithm_architectures.png` → `MD5 (...) = 28c9fdeee4fd2a88a5a7e9a3b03deda8`. Matches user spec.
- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** no

## Cross-cutting patterns

- The patch closed two prior findings outright (8: critic update arrow added; 9: notation drift fixed) and softened two more (4: variance baseline named; 5: "fundamental" → "core"). The score would have dropped to ~10% if no regressions were introduced.
- The patch introduces two new structural problems on panel (c). Both share the same root cause: panel (c) was treated as the "training-time" panel and edited locally, but the caption was rewritten to assert a global relationship between panels ("(a) and (b) depict the acting-time forward pass; panel (c) additionally shows…"). The local edit no longer matches the global claim. Specifically:
  1. Panel (c) drops the Environment loop, so "additionally shows" is false in the set-theoretic sense — panel (c) is not a superset of (a)/(b)'s content along the environment axis.
  2. Panel (c) adds a second arrow on top of an existing forward arrow without re-routing, producing a visual region near δ_t where two arrowheads collide.
- The dilution markers from the v1 audit ("low-variance", "three fundamental") were softened but not eliminated. The caption is now longer and contains more verifiable claims, which is good for fidelity in absolute terms but also gives a hostile reader more surface area to attack.
- The diagram-only cap (25%) is the binding constraint on the final score. Without the cap, the new structural mismatches (Findings 12 and 13) plus the residual dilutions (Findings 4 and 5) would push the score to roughly 35–40%.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** 25% — Reviewer-2 grade, diagram-only cap pinned. Ship after touch-up; surface to user before any deeper code rewrite. Do not halt downstream work on other chapters.
1. For each non-HOLDS finding (4, 5, 12, 13), turn the **violated invariant** into a pytest test under `tests/test_algorithm_architectures_caption.py`. Confirm each test PASSES on current code (proves the gap is real).
2. Convert each **honest-fix pass condition** into a paired test that FAILS on current code. The pairs are the red/green specs.
3. Hand off to `writing-plans`:
   - Findings 4 and 5: caption wording (e.g., drop the variance comparison or remove the definite article "three core").
   - Finding 12: route the dashed `δ_t→Critic` arrow with larger curvature, an explicit waypoint, or terminate on the Critic's left/top edge instead of the right edge to avoid collision with the solid `Critic→δ_t` arrow.
   - Finding 13: either add an Environment loop to panel (c) (matches the caption's "additionally shows"), or rewrite the caption to describe panel (c) as showing a different slice rather than a superset (e.g., "panels (a) and (b) show the acting-time forward pass with environment feedback; panel (c) abstracts the environment and instead shows the training-time TD signal feeding back to both networks").
4. After fixes, re-render the PNG. The MD5 will change for any figure edit (Findings 12, 13) but not for caption-only edits (Findings 4, 5).
5. Re-run this skill on the new code. Target a score ≤10% on the next pass.
