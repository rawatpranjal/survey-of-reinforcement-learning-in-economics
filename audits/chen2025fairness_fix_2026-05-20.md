# Hallucinated bibliography fix: `Chen2025fairness` -> `Cohen2025fairness`

Date: 2026-05-20
Chapter: ch07 (Bandits and Dynamic Pricing), `dynamic_pricing.tex`

## 1. Original (wrong) entry and claim

The audited entry was a hallucination. It did not match any real paper.

Original BibTeX entry (`arxiv_submission 2/refs.bib`):

```
@article{Chen2025fairness,
  author    = {Chen, Yuxin and Mao, Jieming and Miao, Rui},
  title     = {Dynamic Pricing with Fairness Constraints},
  journal   = {arXiv preprint arXiv:2402.07834},
  year      = {2025}
}
```

Original claim, in a footnote at `ch07_bandits/tex/dynamic_pricing.tex:94`:

> \citet{Chen2025fairness} show that imposing fairness constraints (requiring
> similar prices for similar customers) raises the regret floor to
> $\Theta(T^{2/3})$, a social cost of equitable treatment.

Both the metadata (authors, venue, arXiv id) and the cited result (a
$\Theta(T^{2/3})$ regret floor) were fabricated.

## 2. Scope of the problem

The fabricated key appeared only in the frozen submission snapshot
`arxiv_submission 2/` — in three files: `refs.bib`, `main.bbl`, and
`ch07_bandits/tex/dynamic_pricing.tex`. The canonical `docs/refs.bib` and the
canonical `ch07_bandits/tex/dynamic_pricing.tex` do NOT contain the entry or
any citation of it; neither do `arxiv_submission/`, `journals/shared/refs.bib`,
`thesis/docs/refs.bib`, or `thesis_v2/docs/refs.bib`. The fix is therefore
confined to `arxiv_submission 2/`.

## 3. What Perplexity found

Tool: `mcp__perplexity__perplexity_ask`, queried 2026-05-20.

The real paper titled "Dynamic Pricing with Fairness Constraints" is:

- Authors: Maxime C. Cohen, Shanshan Miao, Yining Wang
- Journal: Operations Research, volume 73, issue 6, pages 3027-3043
- Year: 2025
- DOI: 10.1287/opre.2023.0123
- Sources: pubsonline.informs.org/doi/10.1287/opre.2023.0123 ;
  ideas.repec.org/a/inm/oropre/v73y2025i6p3027-3043.html

Result that paper actually proves: under group fairness constraints (prices
may depend on context but must satisfy ex ante fairness across groups), an
infrequently updated UCB algorithm achieves $\tilde{O}(\sqrt{T})$ regret with
a matching lower bound, so the minimax regret order is $\tilde{\Theta}(\sqrt{T})$.
Fairness constraints do NOT worsen the regret order relative to unconstrained
contextual dynamic pricing.

On the $\Theta(T^{2/3})$ claim: Perplexity found NO paper establishing a
$\Theta(T^{2/3})$ regret rate or lower bound for fairness-constrained dynamic
pricing. The rigorous regret results in this line of work all give
$\sqrt{T}$-order regret, including Cohen-Miao-Wang (2025) and the related
Liu & Sun (2025), "Fairness-aware Contextual Dynamic Pricing with Strategic
Buyers", arXiv:2501.15338, which proves an $O(\sqrt{T})$ upper bound and an
$\Omega(\sqrt{T})$ lower bound.

Conclusion: the $\Theta(T^{2/3})$ rate is unsupported by any paper. It must be
dropped, not re-attributed.

## 4. Option chosen

Option B. The intended reference is Cohen-Miao-Wang (2025), but that paper
proves a different result ($\sqrt{T}$, not $T^{2/3}$). The bib entry was
replaced with correct Cohen-Miao-Wang metadata, and the tex sentence was
rewritten so the claim matches what the paper actually proves. The false
$\Theta(T^{2/3})$ rate was removed.

## 5. Exact edits

### `arxiv_submission 2/refs.bib`

New entry, key renamed `Chen2025fairness` -> `Cohen2025fairness`:

```
@article{Cohen2025fairness,
  author    = {Cohen, Maxime C. and Miao, Shanshan and Wang, Yining},
  title     = {Dynamic Pricing with Fairness Constraints},
  journal   = {Operations Research},
  volume    = {73},
  number    = {6},
  pages     = {3027--3043},
  year      = {2025},
  doi       = {10.1287/opre.2023.0123}
}
```

### `arxiv_submission 2/ch07_bandits/tex/dynamic_pricing.tex` (line 94)

New footnote sentence:

> \citet{Cohen2025fairness} study dynamic pricing under group fairness
> constraints (prices may depend on context but must satisfy ex ante fairness
> across groups) and show, via an infrequently updated UCB algorithm with a
> matching lower bound, that the constrained problem retains
> $\tilde{\Theta}(\sqrt{T})$ minimax regret; fairness does not worsen the
> regret order relative to unconstrained contextual pricing.

### `arxiv_submission 2/main.bbl`

The stale `\bibitem` for `Chen2025fairness` was updated to the
`Cohen2025fairness` entry so the pre-existing .bbl is consistent. bibtex
also regenerated it correctly during recompilation.

## 6. Recompile result

`arxiv_submission 2/` has no `compile_chapter.tex` (chapters there are
`\input` into `main.tex`), so verification used the full document build:
`pdflatex -> bibtex -> pdflatex -> pdflatex` on `arxiv_submission 2/main.tex`.

- All four passes exited 0.
- bibtex ran with no errors or warnings.
- Final pass: no "Citation undefined" errors.
- `Chen2025fairness` no longer appears anywhere in `arxiv_submission 2/`.
- `Cohen2025fairness` resolves correctly; regenerated `main.bbl:431` has the
  Cohen-Miao-Wang entry.
- Output: `arxiv_submission 2/main.pdf` (7.6 MB).

Note: the canonical survey (`docs/`, `ch07_bandits/tex/dynamic_pricing.tex`)
was already clean and required no changes.
