# Humanizer DETECT report — `ch07_bandits/tex/dynamic_pricing.tex`

Mode: detect (diagnostic only, no edits). Target voice: **technical** (academic arXiv survey). Judged against a technical-academic register, not a casual/blog register. LaTeX commands, math, `\citet`/`\citep`, `\ref`/`\label`, theorem environments, equations, captions, and footnote evidence-trails were excluded from prose scoring per the task brief and the project CLAUDE.md style rules.

---

## 1. Overall AI-tell score

**Score: 34 / 100 (Mostly human, leaning low-mixed).**

The prose is genuine technical writing with high perplexity (dense domain jargon, real citations, specific numbers like "83.6%", "$6.4\times10^{33}$ assortments", "7,870 customers per month"); the residual AI smell is almost entirely *structural*, not lexical: a repeated "The [noun] is …" topic-sentence template, colon-after-claim emphasis, and aphoristic one-line paragraph closers. None of the heavy lexical tells (delve, leverage, tapestry, vibrant, "it's worth noting") appear in the running prose.

---

## 2. Per-pattern table (only patterns that fire)

| Pattern ID | Name | Count | Severity |
|---|---|---|---|
| P13 | Em Dash Ban | 1 (line 3, `---` → renders as em dash) | low |
| P30 / P38 | "The [noun] is/changes/appears…" topic-sentence template (paragraph-reshuffling immunity) | ~12 paragraph openers | med |
| P1 | Significance Inflation ("the canonical … problem", "the central measure") | 2 | low–med |
| P40 | Symbolic gloss / aphoristic meaning-telling ("demand learning is also mechanism design") | 3 | med |
| P9 | Negative parallelism ("is not just a constant factor"; "does not merely say X; it says Y"; "is not a cosmetic modeling choice") | 3 | low–med |
| P7 | AI-vocabulary discourse adverbs ("Crucially,", "Formally,", "Together,", "Notably"-class) | 3–4 | low |
| P18 | Colon-after-claim emphatic register ("This is not a cosmetic modeling choice:"; "is the outlier: it produces…"; "The regret is linear in $T$:") | 4–5 | low–med |
| P10 | Rule of three (forced abstract triads) | 2 | low |

Patterns checked and **not** firing in main prose: P2/P37 (name-dropping — citations are substantive, each says what the paper *did*), P4 (promotional), P5 (vague attribution — all claims are cited), P14/P15/P28/P42 (boldface/bullets/markdown — none; the file obeys the no-`\textbf` rule), P17 (curly quotes — straight quotes throughout), P19/P20/P21 (chatbot artifacts/disclaimers/sycophancy), P24 (generic positive conclusion), P27 (question headings), P29/P32 ("comprehensive overview"/instructional framing), P33/P34/P35 (placeholders/citation-markup/UTM), P43 (treadmill restatement — density is high, little padding).

---

## 3. Line-anchored examples (high/med patterns)

### P30 / P38 — repetitive "The [noun] is …" topic-sentence template (med, highest-impact)
The single strongest tell. Many paragraphs open or pivot on a flat abstract subject + copula, and several are self-contained mini-theses that could be reordered without breaking the argument (the P38 reshuffle test).
- L18: *"The issue is not whether the model has a finite-dimensional parameter; it is whether profitable prices are also informative…"*
- L20: *"The picture changes under a ``well-separated'' condition…"*
- L106: *"The lesson is that demand learning is also mechanism design."*
- L111: *"The dominant pattern is that stronger assumptions yield faster learning…"*
- L148: *"The same pattern appears beyond single-product posted pricing."*
- L165: *"The finite-sample pattern is mixed."* … *"The right conclusion is narrower:"*
- L176: *"The low-optimal-price case is the clearest diagnostic."* … *"The conclusion is therefore narrower…"*
Why it reads as AI: the model favors a uniform "topic sentence = abstract noun + is" scaffold; a human technical writer varies the entry point (leads with the mechanism, the citation, or a concrete number more often).

### P40 — aphoristic symbolic gloss as paragraph closer (med)
Sentences that translate a result into its "real meaning" as a punchy standalone line.
- L106: *"The lesson is that demand learning is also mechanism design."* (then L106: *"Incentive compatibility matters even in settings where the seller is ``just'' learning demand."*)
- L62: *"That is why the incremental gains from partial identification can be small once a curve-learning bandit is already transferring information efficiently across prices."*
- L34: *"Even with hundreds of features, if only a handful matter, learning is fast."*
Why it reads as AI: the gloss restates the maths in a quotable register ("X is also Y", "the lesson is"); a human would more often let the theorem/number stand.

### P9 — negative parallelism (low–med)
Theatrical "not X; (it is) Y" build-ups. Once is fine; three instances make it a tic.
- L62: *"It does not merely say which prices can be discarded; it says how evidence at one price should move beliefs…"*
- L67: *"This is not a cosmetic modeling choice: the error distribution determines…"*
- L111: *"…so the distinction is not just a constant factor."* (paired with L69 footnote-adjacent main text *"This is not a constant-factor tuning issue; it is a change in the learning rate."*)
Why it reads as AI: the rhetorical negation-then-correction is a signature LLM emphasis move.

### P18 — colon-after-claim emphatic register (low–med)
Declarative claim, colon, then the payoff clause. Recurs enough to feel templated.
- L90: *"The regret is linear in $T$: every standard dynamic pricing algorithm … systematically underprices…"*
- L111: *"Strategic behavior is the outlier: it produces linear regret that no amount of data can overcome…"*
- L67: *"This is not a cosmetic modeling choice: the error distribution determines…"*
Why it reads as AI: the colon-as-drumroll is common in chatbot expository prose; commas or a second sentence read more human.

### P1 — significance inflation (low–med)
- L1: *"Dynamic pricing is **the canonical** in-field reinforcement-learning problem for economics."*
- L3: *"\textit{Regret} … is **the central measure**."*
Why it reads as AI: superlative framing ("the canonical", "the central") asserts importance rather than showing it. Mild here because the rest of the sentence is concrete.

### P13 — em dash (low, but a project hard-rule)
- L3: `With enough structure---well-separated parametric demand … smooth monotone demand curves---learning can be much faster.` The `---` renders as an em dash. The project CLAUDE.md explicitly bans em dashes in prose, so this is a real (if cosmetic) hit even though the technical target is lenient on P13. Fix: replace with commas or parentheses (e.g., a colon is also banned here).

---

## 4. Top 5 highest-impact fixes (described only — NOT applied)

1. **Break the "The [noun] is …" opener monotony (P30/P38).** Rewrite ~6–8 of the ~12 flat openers to lead with the concrete mechanism, the number, or the citing author instead of an abstract subject + copula (e.g., L111 "The dominant pattern is that stronger assumptions…" → "Stronger assumptions yield faster learning, but only when…"). Also add one explicit callback so adjacent paragraphs depend on each other (defeats the reshuffle test). **Expected delta: −8 to −10.** Largest single lever.

2. **Defuse the aphoristic closers (P40).** Soften or fold the 3 "the lesson is / X is also Y" punchlines into the preceding analytical sentence so the paragraph ends on its strongest specific point, not a quotable gloss (esp. L106 "demand learning is also mechanism design"). **Expected delta: −4 to −5.**

3. **Cut two of the three negative parallelisms (P9).** Keep at most one "not X; it is Y"; state the others directly (e.g., L67 "This is not a cosmetic modeling choice: …" → "The error distribution matters because it determines how purchase probabilities translate into valuations."). **Expected delta: −3 to −4.**

4. **Reduce colon-as-drumroll (P18).** Convert 2–3 of the claim-colon-payoff constructions (L90, L111, L67) into two plain sentences. Note the project also bans colons in prose, so this doubles as a style-compliance fix. **Expected delta: −2 to −3.**

5. **Replace the line-3 em dash and trim the two superlatives (P13 + P1).** Swap `---` for commas/parens; soften "the canonical" / "the central measure" to a factual statement of what the object is. **Expected delta: −2.**

Cumulative if all five applied: roughly **34 → ~14–18** (solidly "pristine/mostly-human").

---

## 5. What is already human/good (leave alone)

- **All citation handling.** Every `\citet`/`\citep` is load-bearing — it says what the paper *did* (rates, theorems, algorithm names), never name-dropping. Do not touch (no P2/P5/P37).
- **Footnote evidence-trails** (L11, L18, L20, L34, L67, L69, L73, L106, L163, L174). These carry hyperparameters, theorem numbers, and proof sketches exactly as the project style mandates. Not prose tells; do not flag or "humanize."
- **Concrete-number density is excellent and very human:** "98% of oracle profit", "43% higher profits", "176 products and 30 display slots", "$\binom{176}{30}\approx 6.4\times10^{33}$", "roughly 5,000 items", "83.6% / 97.5% / 98.3%". This is the chief reason the score is low; preserve all of it.
- **The honest hedging on the sim results** (L111, L165, L176): "the finite-sample pattern is mixed", "the right conclusion is narrower", "illustrative rather than predictive". This is genuine scientific caution and reads as a real author with stakes in the claims — keep it (only de-template the *openers*, not the content).
- **The WARP intuition paragraph (L43)** with the \$80/\$120 worked example — concrete, specific, well-paced. Leave intact.
- **Sentence burstiness is acceptable** (short punches like "The contextual benchmark is harder." and "Every price is informative…" sit beside 35-word sentences). No P30 uniform-length failure at the sentence level; the issue is opener *templating*, not length variance.
- **Zero formatting slop:** no bold, no bullets, no markdown bleed, straight quotes throughout. The file already complies with the project's no-`\textbf`/no-bullets rules.
