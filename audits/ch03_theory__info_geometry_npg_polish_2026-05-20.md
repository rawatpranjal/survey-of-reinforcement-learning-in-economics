# Polish: ch03_theory/sims/info_geometry_npg.py

**Date:** 2026-05-20
**Prior audit:** `audits/ch03_theory__info_geometry_npg_2026-05-19.md` (25%)
**Diagram-only:** YES. Cap remains 25% per CLAUDE.md unless the diagram visually contradicts the caption.
**Cited tex file:** `ch03_theory/tex/planning_learning_v3.tex` (lines 371–388), Figure `\label{fig:info_geometry}`.

## Fixes applied

### Fix 1 — Misleading-ratio caveat in stdout

`info_geometry_npg.py` previously printed `g^T nat_step / g^T euc_step = 0.68` with no caveat, so a reviewer reading the stdout alone would conclude NPG was *worse* than the Euclidean step. The Euclidean step is actually disqualified because its KL is `1.088 > δ = 1.0`. Added two print lines naming the disqualification explicitly and stating that the natural step saturates the KL budget by construction, so the linear-gain comparison is not apples-to-apples.

New stdout tail:

```
Note: Euclidean step violates the KL constraint (KL = 1.088 > delta = 1.0); not a valid trust-region step.
The natural step saturates the KL budget exactly (KL = 0.500 = delta/2), so the comparison of linear gains
above is not apples-to-apples; the Euclidean step buys its higher linear gain by leaving the trust region.
```

### Fix 2 — Ellipse rotation pairing (the real one)

Confirmed the prior audit's suspicion. Before the fix, the code was:

```python
angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))  # rotate to eigvec[:,1]
width = 2 * np.sqrt(delta / eigvals[0])   # LONG semi-axis
height = 2 * np.sqrt(delta / eigvals[1])  # SHORT semi-axis
```

`np.linalg.eigh` returns eigenvalues in ascending order, so `eigvals[0] ≈ 0.619` (small) and `eigvals[1] ≈ 2.181` (large). The KL ellipse `{x : x^T F x ≤ δ}` in the eigenbasis is `λ_0 c_0² + λ_1 c_1² ≤ δ`, so the semi-axis along `eigvecs[:, i]` is `sqrt(δ/λ_i)`. The LONG axis (small λ) is along `eigvecs[:, 0]`. matplotlib's `Ellipse(width, height, angle)` places `width` along the local x-axis before rotating counter-clockwise by `angle`. The old code rotated to `eigvecs[:, 1]` (large-λ direction) but assigned the long `width` to that direction — placing the long axis along the *steep* direction of F, which is backwards.

Hand check confirmed:

```
eigvecs columns:
[[ 0.34, -0.94],
 [-0.94, -0.34]]
old angle (atan2 on eigvecs[:,1]) = -160.10°
new angle (atan2 on eigvecs[:,0]) =  -70.10°  (differs by 90°, correct)

x = sqrt(δ/λ_0) * eigvecs[:,0] = (0.433, -1.195)
x^T F x = 1.000  ✓  (this point lies on the ellipse boundary along the long axis)

nat_step components in eigenbasis: c_0 = 0.133, c_1 = -0.673
(c_0/long)² + (c_1/short)² = 1.000  ✓  (nat_step on the ellipse boundary)
```

Fix:

```python
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
width = 2 * np.sqrt(delta / eigvals[0])   # long axis along eigvecs[:,0]
height = 2 * np.sqrt(delta / eigvals[1])  # short axis along eigvecs[:,1]
```

Visual check on the rendered PNG: the KL ellipse's long axis now points along the upper-left / lower-right diagonal, matching the small-eig eigenvector `(0.34, -0.94)` (slope -2.76). The natural-gradient arrow tip lies on the ellipse boundary. The Euclidean unit circle clearly extends past the ellipse along the short (steep) axis, making the KL violation visually explicit. The geometry now matches F's spectrum.

A six-line comment in the script explains the eigenbasis derivation and the matplotlib convention so the pairing is no longer load-bearing on the reader.

### Fix 3 — Disclose left panel as schematic

Updated the figure caption in `planning_learning_v3.tex` (lines 374–388). The left panel's "natural gradient" curve is `linear-interp(π_old, π*) + 0.4 sin(πt)` with hand-placed Euclidean direction `(2.0, 0.3)` — purely decorative, not derived from F. New caption text:

> *Left*: schematic of the policy manifold M with Euclidean gradient (red) and natural gradient (green) from the current iterate toward the optimal policy π\*. The two arrows are placed schematically to convey the convergence advantage of NPG; quantitative behavior depends on the loss landscape and is shown in the right panel.

Also added a closing sentence to the right-panel caption naming the KL violation explicitly, mirroring the new stdout caveat:

> The Euclidean step (red arrow) leaves the KL ball, violating the trust-region constraint; the natural step (green arrow) saturates the KL budget by construction.

## Verification

- Script re-run: `python3 ch03_theory/sims/info_geometry_npg.py > ch03_theory/sims/info_geometry_npg_stdout.txt 2>&1`, exit 0. New stdout 28 lines (was 26), KL caveat present.
- PNG regenerated: `ch03_theory/sims/info_geometry_npg.png`. Ellipse orientation visually correct, long axis along small-eig direction.
- Chapter PDF recompiled: `docs/ch03_theory.pdf`, 31 pages, 2.5 MB. New caption text renders cleanly.

## Hostile-reviewer summary (post-polish)

Reviewer 2's three prior snarks:

1. **"Why did you print numbers that contradict your headline?"** — resolved. Stdout now names the KL violation in plain text right after the ratio.
2. **"The KL ellipse orientation may be 90° off."** — resolved. Long axis now along the shallow direction of F, as the spectrum dictates. Verified analytically and visually.
3. **"The left panel is decorative but the caption reads as if it's derived."** — resolved. Caption now says "schematic ... arrows are placed schematically ... quantitative behavior ... is shown in the right panel."

Residual: the left panel is still a sin-bumped linear interpolation, by design. With the caption fixed, that's no longer a snark — it's an honest illustration. The right panel is now consistent with F's spectrum, and the figure tells the story the prose (and the prior audit) says it should.

**Bullshit score: 10%** — A skeptical reader might still note that the chosen `ε = 1` for the Euclidean ball and `δ = 1` for the KL ball aren't "matched" in any principled sense (a point made in §4 of the prior audit). But the new caption and stdout name the constraint violation explicitly, the ellipse renders correctly, and the schematic disclosure makes the left panel honest. Nothing left for Reviewer 2 to weaponize beyond a stylistic preference for matched budgets.
