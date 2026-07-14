# Bloat Detection and Trimming Guide

This guide supplements the Chapter Editing Pass rules in `CLAUDE.md`. It provides a systematic approach to detecting and removing bloat from dense academic writing that bridges economics and computer science.

## Phase 1: Macro-Trimming (Structural Bloat)

Before looking at sentences, look at the architecture. Whole paragraphs can be deleted if they don't serve the core argument.

### Rule M1: Define the Audience Floor

The bloat: explaining basic concepts your target audience already knows. This survey targets economists curious about RL and RL researchers curious about economics. Both hold PhDs or are advanced graduate students.

The fix: never explain what a derivative, a regression, a Markov chain, or a neural network layer is. Do explain RL-specific concepts (Bellman operator, policy gradient) to economists and economics-specific concepts (DDC, WARP, McFadden logit) to RL researchers. If a concept falls below the audience floor, delete the explanation entirely.

### Rule M2: Synthesize, Don't Serialize (The Literature Review Rule)

The bloat: listing papers one by one. "Smith (2018) found X. Then, Jones (2019) found Y. Later, Doe (2020) found Z."

The fix: group by concept or tension. "Recent work demonstrates X (Smith 2018; Jones 2019), though Doe (2020) highlights limitation Z." Let the ideas drive the paragraph, not the chronological list of authors. Each paragraph should advance one claim, supported by multiple citations, not walk through papers sequentially.

### Rule M3: Eliminate Redundant Signposting

The bloat: overusing meta-discourse to tell the reader what you are doing. "In this section, we will first attempt to show that..." or "As we previously discussed in Chapter 3..." or "We now turn to the question of..."

The fix: just do the thing. Trust headings and subheadings to do the structural work. If a cross-reference is needed, one parenthetical suffices: "(Section~\ref{sec:foo})". Never open a subsection by re-motivating the chapter.

### Rule M4: Let Figures and Tables Do the Talking

The bloat: spending a full paragraph describing the exact data points visible in a chart or table. "DQN achieved -331.68, Q-learning reached 2.95, and Nash-Q reached 2.89 with higher variance..."

The fix: if the table shows the exact numbers, the prose should only state the insight or the anomaly. One to two sentences per table: what it shows and what the reader should conclude. Never re-list numbers that appear in the table. Example:

- Fat: "Table 3 shows that AMZN achieved -18.2% vs TWAP and -14.6% vs AC, QCOM achieved -22.1% vs TWAP and -19.3% vs AC, and NVDA achieved -15.8% vs TWAP and -12.1% vs AC."
- Trimmed: "Table 3 shows RL reduces execution cost by 12-19% over the Almgren-Chriss baseline across all three stocks."

### Rule M5: One Definition Per Concept

The bloat: defining the same concept multiple times across the chapter. The contraction property appears in Section 3.1, then is re-explained in Section 3.3, then restated in Section 3.5.

The fix: define each concept once at its canonical location. All subsequent uses reference the definition via `\ref{}`. Acceptable: "Recall that $T$ is a $\gamma$-contraction (Theorem~\ref{thm:contraction})." Not acceptable: re-deriving or re-explaining the property.

## Phase 2: Micro-Trimming (Sentence-Level Fat)

Once the structure is sound, tighten the prose sentence by sentence.

### Rule S1: Destroy Zombie Nouns (Nominalizations)

The bloat: turning strong verbs into clunky nouns paired with weak verbs (make, have, do, perform, provide, achieve).

The fix: resurrect the verb.

- Fat: "The algorithm makes an estimation of the value function."
- Trimmed: "The algorithm estimates the value function."
- Fat: "We provide a demonstration of convergence."
- Trimmed: "We demonstrate convergence."
- Fat: "The method achieves a reduction in regret."
- Trimmed: "The method reduces regret."

### Rule S2: Prune Hedges and Qualifiers

The bloat: stacking qualifiers to avoid absolute claims. Words: very, rather, somewhat, quite, generally, it seems that, it could be argued that, it is worth noting that, it is important to note that.

The fix: limit to one qualifier per claim. If the claim needs hedging, hedge once. If it doesn't, state it flat.

- Fat: "It might perhaps be argued that this algorithm is generally somewhat faster."
- Trimmed: "This algorithm is faster."
- Fat: "It is worth noting that the deadly triad causes divergence."
- Trimmed: "The deadly triad causes divergence."

### Rule S3: Fix Prepositional Chains

The bloat: stringing nouns together with "of the" constructions.

The fix: use possessives or adjectives.

- Fat: "The convergence of the algorithm of the agent in the environment..."
- Trimmed: "The agent's algorithm converges in the environment..."
- Fat: "The estimation of the parameters of the model..."
- Trimmed: "The model's parameter estimates..."

### Rule S4: Cut Expletive Constructions

The bloat: starting sentences with empty placeholder subjects: "There are", "There is", "It is".

The fix: put the true subject at the front.

- Fat: "There are three algorithms that restore convergence."
- Trimmed: "Three algorithms restore convergence."
- Fat: "It is the case that off-policy methods diverge."
- Trimmed: "Off-policy methods diverge."
- Fat: "There exists a unique Nash equilibrium in pure strategies."
- Trimmed: "A unique Nash equilibrium in pure strategies exists."

### Rule S5: Compress Relative Clauses

The bloat: using "which/that/who" clauses that can be reduced to adjectives or participles.

The fix: compress the clause.

- Fat: "The policy that was learned by the agent..."
- Trimmed: "The agent's learned policy..."
- Fat: "Algorithms which are off-policy..."
- Trimmed: "Off-policy algorithms..."
- Fat: "The reward function that is estimated from preferences..."
- Trimmed: "The preference-estimated reward function..."

### Rule S6: Kill Redundant Pairs and Triplets

The bloat: using near-synonyms in pairs or triplets for emphasis. "Each and every", "first and foremost", "various and sundry", "goals and objectives", "methods and techniques".

The fix: pick one word.

## Phase 3: Plain Construction (De-Strain)

Micro-trimming removes fat. This pass removes *strain*. A sentence can be short and still read strange, because it leans on writerly scaffolding instead of naming things plainly. These constructions read as over-written or "AI-ish" even when every fact is correct. Fix them by saying the thing directly.

### Rule P1: Kill Frame-Noun Scaffolding

The bloat: filler nouns that frame content instead of carrying it. "instance", "line" (of work), "slot", "foil", "lens", "the X that occupies the rest of the chapter".

The fix: name the thing, or delete the frame.

- Strained: "the reinforcement-learning line that occupies the rest of the chapter is the same loop pursued at scale."
- Plain: "The rest of the chapter runs the same loop in reinforcement learning, at larger scale."
- Strained: "This is the architectural slot that subsequent work fills with deep ensembles..."
- Plain: "This open design is what subsequent work fills with deep ensembles..."
- Strained: "the reinforcement-learning instances of the model-based loop, with the model-free schemes as their foil."
- Plain: "the reinforcement-learning versions of the model-based loop." (drop the foil; the contrast is already drawn)

### Rule P2: No Imperative Math-Speak

The bloat: "Write X as Y", "Let us denote", "Consider the object Z" used as connective tissue rather than genuine setup.

The fix: state the definition as fact.

- Strained: "Write the agent's model at date $t$ as a parameter $\theta_t$ indexing..."
- Plain: "The agent's model at date $t$ is a parameter $\theta_t$ that indexes..."

### Rule P3: Do Not Personify the Argument

The bloat: making an abstraction the subject of an active verb, or narrating the text's own moves. "Two questions decide whether...", "the examples answer one question each", "This is the economic statement of...", "the section isolates the commitments that survive".

The fix: put the real actor first, or state the claim plainly.

- Strained: "Two questions decide whether the learner succeeds."
- Plain: "Success hinges on two questions."
- Strained: "This is the economic statement of the world-model failure the synthesis returns to."
- Plain: "This is the economic version of the world-model failure discussed in the synthesis."

### Rule P4: Unpack Terse Appositions

The bloat: cramming a contrast into a comma-apposition the reader has to decode.

The fix: use two plain sentences.

- Strained: "The model class differs, a tabular Dirac in Dyna against a differentiable network in Schmidhuber."
- Plain: "The model class differs. Dyna uses a tabular Dirac, Schmidhuber a differentiable network."
- Strained: "differ in two places and agree in a third."
- Plain: "differ in two ways and agree in one."

### Rule P5: No Idioms or Cliches

The bloat: stock phrases that read as filler. "not a free lunch", "the whole point", "at the end of the day", "when it comes to".

The fix: state the mechanism.

- Strained: "The sample-efficiency claim that follows is not a free lunch."
- Plain: "This sample efficiency is not free."

### Rule P6: Cross-Reference, Never Re-Derive

The bloat: restating a theorem, operator definition, or proof already established earlier (a special case of Rule M5). The Watkins-Dayan Q-learning convergence result, the Bellman operator, and its $\gamma$-contraction were spelled out in full inside a later chapter that had already cited them.

The fix: cite the first statement.

- Strained: "...who prove that under $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$..., $Q$ converges to $Q^\star$, the fixed point of $\mathcal{T}^\star$ defined by... . The operator is a $\gamma$-contraction in the sup norm."
- Plain: "It converges to $Q^\star$ under the standard step-size and visitation conditions, with the $\gamma$-contraction of the Bellman operator supplying the argument (Theorem~\ref{thm:qlearning_convergence}, Section~\ref{sec:planning_learning})."

## The 10% Constraint (Final Test)

When a section appears fully edited, force a further 10% word count reduction without losing any fact or argument. This constraint shifts attention from what is being said to how it is being said, exposing hidden bloat invisible during content-focused editing.

## Applying This Guide

When performing an editing pass on a chapter:

1. First pass (macro): scan for structural bloat using Rules M1-M5. Delete whole paragraphs, merge redundant subsection openers, collapse serialized literature reviews.
2. Second pass (micro): tighten remaining prose using Rules S1-S6. This is line-by-line work.
3. Third pass (de-strain): apply Rules P1-P6 to remove strained, writerly constructions (frame-noun scaffolding, imperative math-speak, personified arguments, terse appositions, idioms) and any re-derived result that should be a cross-reference. Prose can pass Phase 2 and still read strange; this pass catches that.
4. Final pass: apply the 10% constraint to any section that still feels heavy.

This guide works in conjunction with the Chapter Editing Pass rules in `CLAUDE.md` (one canonical location, cross-references over restatement, footnotes for details, figure/table caption discipline, no re-motivation in section openings).
