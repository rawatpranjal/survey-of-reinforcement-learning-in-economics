# Scrutiny Report Round 2

This memo restarts the review after the latest edits. It supersedes the earlier first-chapter comments in `thesis_v2/intro_scrutiny_report.md`.

## Chapter 1: introduction

Scope of review:
- Reviewed `thesis_v2/ch00_introduction/tex/intro.tex`
- Cross-checked chapter scope against `thesis_v2/docs/main.tex`
- Cross-checked framing against `thesis_v2/ch00_introduction/tex/abstract.tex`
- Did not edit thesis source

### Overall judgment

The introduction is improved in one important respect. The bridge between economics and RL is now more visible, especially in the `Two Cultures` section and the `Structural Equivalences` section. The chapter still has two main problems. It remains too RL-first in its opening paragraphs, and it still carries several dense blocks of jargon that many economists will not absorb easily in a first chapter.

### What is working

- The chapter now has a clearer bridge spine. `thesis_v2/ch00_introduction/tex/intro.tex:20` through `thesis_v2/ch00_introduction/tex/intro.tex:26` gives a clean language-and-purpose contrast between economics and RL.
- The formal bridge is strong. `thesis_v2/ch00_introduction/tex/intro.tex:98` through `thesis_v2/ch00_introduction/tex/intro.tex:114` is still one of the best parts of the chapter for an economist audience.
- The notation table remains useful as a translation device. See `thesis_v2/ch00_introduction/tex/intro.tex:119`.

### Inconsistencies and conceptual mismatches

- **Discount-factor inconsistency.** `thesis_v2/ch00_introduction/tex/intro.tex:9` defines `\gamma \in [0,1)`, but the footnote at `thesis_v2/ch00_introduction/tex/intro.tex:132` says RL permits `\gamma = 1` in episodic settings. Those two statements conflict.
- **Roadmap framing is still too narrow.** `thesis_v2/ch00_introduction/tex/intro.tex:13` says the chapter addresses the forward problem in a known or simulated environment, but `thesis_v2/docs/main.tex:165` through `thesis_v2/docs/main.tex:167` shows that the thesis still includes `Offline RL under Unobserved Confounding`. That is not naturally described as a known-or-simulated-environment problem.
- **Off-policy evaluation is still equated too strongly with causal counterfactuals.** `thesis_v2/ch00_introduction/tex/intro.tex:31` says off-policy evaluation is `precisely counterfactual policy evaluation`. For economists, that sounds stronger than the chapter can support because it blurs the line between policy evaluation under maintained assumptions and causal identification.
- **Lifecycle table label mismatch.** `thesis_v2/ch00_introduction/tex/intro.tex:61` labels the row `Live market ("in-field")`, but `thesis_v2/ch00_introduction/tex/intro.tex:63` places `AlphaGo vs. Lee Sedol` there. That is a live deployment example, but not a market example.
- **Pipeline sentence overgeneralizes.** `thesis_v2/ch00_introduction/tex/intro.tex:68` presents one pipeline, from historical logs to simulator refinement to frozen deployment, as the typical applied RL path. That is too broad for the chapter’s own applications and for RL more generally.
- **Model paragraph still has typesetting artifacts.** `thesis_v2/ch00_introduction/tex/intro.tex:35` contains `\"` quote escapes in running prose.

### Jargon economists may not understand quickly

- **Opening paragraphs are too technical too early.** `thesis_v2/ch00_introduction/tex/intro.tex:3` and `thesis_v2/ch00_introduction/tex/intro.tex:5` begin with `average Bellman error`, `sampled Bellman error`, `geometric rate`, `sublinear convergence`, `sufficient exploration`, and `hyperparameters` before the economic bridge is established.
- **Control acronyms are unexplained.** `thesis_v2/ch00_introduction/tex/intro.tex:22` uses `PID`, `LQR`, and `MPC` without expansion.
- **RL acronyms are unexplained.** `thesis_v2/ch00_introduction/tex/intro.tex:37` uses `MDP` without expansion. `thesis_v2/ch00_introduction/tex/intro.tex:83` uses `UCB` without expansion. `thesis_v2/ch00_introduction/tex/intro.tex:85` uses `TD` without expansion.
- **The adaptive-learning paragraph is too dense for an introduction.** `thesis_v2/ch00_introduction/tex/intro.tex:81` compresses `Robbins-Monro`, `E-stability`, `ODE method`, `temporal-difference`, `Q-learning`, and `actor-critic` into one paragraph.
- **The exploration and bootstrapping paragraphs read more like glossary entries than introduction prose.** See `thesis_v2/ch00_introduction/tex/intro.tex:83` and `thesis_v2/ch00_introduction/tex/intro.tex:85`.
- **The policy paragraph still leans on RL-native infrastructure language.** `thesis_v2/ch00_introduction/tex/intro.tex:87` uses `digital twin`, `sim-to-real transfer`, and `domain adaptation`, which many economists will not parse immediately.
- **The convex-analysis footnote is too specialized for this location.** `thesis_v2/ch00_introduction/tex/intro.tex:112` introduces `Fenchel conjugates` and `negative Shannon entropy` in a first-chapter bridge section.
- **The final table footnote is technically correct but too fine-grained for a chapter opener.** `thesis_v2/ch00_introduction/tex/intro.tex:139` distinguishes `TD error`, `Bellman residual`, and `BRM` at a level many economists will not need yet.

### Framing issues for an economics audience

- **The opening is still RL-first.** `thesis_v2/ch00_introduction/tex/intro.tex:1` through `thesis_v2/ch00_introduction/tex/intro.tex:7` explains RL mechanics before stating plainly what economists gain from the chapter.
- **The notation paragraph arrives too early.** `thesis_v2/ch00_introduction/tex/intro.tex:9` inserts formal definitions before the motivating bridge has fully formed.
- **The scope paragraph still has survey residue.** `thesis_v2/ch00_introduction/tex/intro.tex:11` still reads partly like a map of adjacent literatures rather than a tight statement of the thesis contribution.
- **The abstract is slightly narrower than the chapter roadmap.** `thesis_v2/ch00_introduction/tex/abstract.tex:1` mentions structural estimation, games, and preference learning, but not the retained causal-RL chapter now listed in `thesis_v2/ch00_introduction/tex/intro.tex:13`.

### Best material to preserve

- `thesis_v2/ch00_introduction/tex/intro.tex:20`, because the economics-versus-control distinction is the cleanest conceptual bridge.
- `thesis_v2/ch00_introduction/tex/intro.tex:37`, because the point that `model-free` is not the same as `reduced-form` is genuinely useful for economists, even though the paragraph needs cleanup.
- `thesis_v2/ch00_introduction/tex/intro.tex:87`, because the different meaning of `policy evaluation` is one of the most important translation points in the chapter.
- `thesis_v2/ch00_introduction/tex/intro.tex:98`, because the softmax-logit equivalence is one of the most persuasive bridges in the thesis.
- `thesis_v2/ch00_introduction/tex/intro.tex:105`, because the entropy-regularization and inclusive-value connection gives economists a familiar object.
- `thesis_v2/ch00_introduction/tex/intro.tex:119`, because the notation table reduces repeated exposition later.

### Lowest-value density if more trimming is needed

- `thesis_v2/ch00_introduction/tex/intro.tex:3` through `thesis_v2/ch00_introduction/tex/intro.tex:7`, which are technically competent but too method-first for the chapter opening.
- `thesis_v2/ch00_introduction/tex/intro.tex:43` through `thesis_v2/ch00_introduction/tex/intro.tex:68`, especially the lifecycle grid and the `typical pipeline` sentence.
- `thesis_v2/ch00_introduction/tex/intro.tex:81`, which is a valid bridge but too dense for this location.
- `thesis_v2/ch00_introduction/tex/intro.tex:83` and `thesis_v2/ch00_introduction/tex/intro.tex:85`, which feel more like glossary compression than core argument.
- The footnotes at `thesis_v2/ch00_introduction/tex/intro.tex:112` and `thesis_v2/ch00_introduction/tex/intro.tex:139`, which are technically interesting but low-value for a first chapter aimed at economists.

### Bottom line

The first chapter is directionally better, but it still does not fully read like an economist-facing introduction. The strongest parts are the translation passages and the formal equivalences. The weakest parts are the RL-heavy opening, the unexplained acronyms, the still-too-strong causal language, and a few internal inconsistencies introduced by the latest edits.

Next chapter review is pending.

## Chapter 2: reinforcement learning algorithms

Scope of review:
- Reviewed `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex`
- Checked chapter cross-references against labels present in `thesis_v2/ch00_introduction/tex/intro.tex` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex`
- Did not edit thesis source

### Overall judgment

This chapter is substantively strong and better aligned with the retained thesis than a generic RL survey chapter would be. It gives the reader the algorithmic backbone needed for the theory, structural-estimation, games, and offline-RL chapters. The main remaining problem is audience. The chapter still reads primarily for an RL-literate reader, not for an economics committee. It starts abruptly, moves quickly through dense RL terminology, and leans heavily on games and control examples before explaining why these methods matter for the economist-facing applications later in the thesis.

### What is working

- The chapter has a coherent historical spine from Monte Carlo and TD through Q-learning, policy gradients, fitted methods, and deep RL. This makes the sequencing legible.
- The bridge-relevant material is present. `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:114` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:130` on FQI/FVI is useful for later economist-facing chapters, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:187` links max-ent RL to discrete choice cleanly.
- The AlphaGo Zero section now ends with an econometric interpretation at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:207`, which helps justify its place in the thesis.
- I did not find broken cross-references in this chapter. References to `section:language`, `sec:stochastic_approx`, `sec:deadly_triad`, `sec:policy_gradient`, `sec:actor_critic`, and `sec:fvi_fqi_theory` all resolve to labels present in the thesis copy.

### Inconsistencies and wording slips

- **The chapter opens without an economist-facing setup.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:5` starts immediately with `The Classical Synthesis` and then Monte Carlo estimation. There is no opening paragraph under the section heading explaining why an economist reader needs this chapter.
- **There is a sentence-level typo in the Watkins paragraph.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:39` reads `Instead of learning $V(s')$ learn $Q(s,a)$`, which is missing connective wording, and it contains the quote artifact `quality\"`.
- **The SARSA convergence sentence is broken.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:73` reads `not $Q^*$ ... policies and standard step-size conditions`, which appears to be a drafting error.
- **The chapter’s examples and explanations are still somewhat misaligned with its eventual payoff.** Early emphasis falls on backgammon, cliff-walking, Atari, and Go, while the economist-facing payoff only becomes explicit later in FQI/FVI, control-as-inference, and the final sentence of AlphaGo Zero.

### Jargon economists may not understand quickly

- **The opening drops readers into RL vocabulary immediately.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:11` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:17` uses `episodes`, `returns`, `first-visit`, `exploring starts`, and `GLIE` with little scaffolding.
- **TD language arrives fast and stays technical.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:23` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:33` introduces `bootstrapping`, `TD(0)`, `TD(\lambda)`, `eligibility trace`, and `TD error`. Some of this is defined, but it still reads like textbook compression.
- **Policy-gradient material assumes ML familiarity.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:51` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:57` uses `gradient ascent`, `log-derivative trick`, `Gaussian policy`, and `REINFORCE` with little economist-facing translation.
- **Actor-critic and natural-gradient sections are technically dense.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:81` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:109` introduces `critic`, `advantage`, `two-timescale learning`, `compatibility condition`, `Fisher information matrix`, and `conjugate gradient methods` in quick succession.
- **Deep-RL acronyms pile up quickly.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:151` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:181` uses `TRPO`, `PPO`, `A2C`, `KL divergence`, `SAC`, and entropy regularization. Some are defined, but the density remains high for an economics audience.
- **Control-as-inference is a strong bridge but still very compressed.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:187` uses `graphical model`, `optimality variable`, `structured variational inference`, and `ELBO`. Many economists will not know these terms.
- **AlphaGo Zero contains dense engineering jargon.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:194` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:205` uses `deep residual network`, `binary planes`, `MCTS`, `forward pass`, `PUCT`, `visit count`, `cross-entropy loss`, and `L_2 regularization` at a level many economist readers will not need.

### Framing and proportion issues

- **The chapter is still method-first, not bridge-first.** The content is sensible, but `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:11` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:109` mostly explains RL as RL before connecting it to the parts economists care about.
- **The examples are still disproportionately game- and benchmark-oriented.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:63`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:138`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:192` focus on backgammon, Atari, and Go. These examples are standard, but they are not the reader’s eventual destination in this thesis.
- **The strongest economics bridge arrives late.** The chapter’s cleanest economics-facing material is concentrated in `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:114` and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:187`, rather than shaping the chapter from the start.
- **AlphaGo Zero is justified conceptually but still overweight in implementation detail.** The section matters because of the planning-plus-function-approximation architecture and the link to later theory, but the engineering specifics at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:194` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:205` remain more detailed than most economists need.

### Best material to preserve

- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:23`, because TD learning is the conceptual basis for much of the rest of the thesis.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:39`, because Q-learning is a core bridge from Bellman optimality to model-free computation.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:81`, because actor-critic methods matter later and cannot be dropped without weakening the theory and offline-RL linkage.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:114`, because FQI/FVI is one of the most economist-relevant algorithm blocks in the chapter.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:169`, because SAC sets up the entropy-regularization bridge that is later made explicit.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:187`, because control-as-inference is one of the cleanest bridges to discrete choice and should remain.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:189`, because AlphaGo Zero is worth keeping for the planning-and-learning architecture, especially given its later Bertsekas-style significance.

### Lowest-value density if more tightening is needed

- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:63`, because the TD-Gammon paragraph is historically important but less central to the economist-facing bridge than FQI/FVI or control-as-inference.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:138` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:149`, where the Atari engineering benchmark story is useful background but not one of the stronger bridges in this thesis.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:194` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:205`, where the AlphaGo engineering details could likely be reduced without losing the main conceptual point.
- Footnotes explaining implementation specifics, especially at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:194`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:198`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:205`, if page pressure remains high.

### Bottom line

This is a solid algorithms chapter, but it still needs more translation for an economics audience. The core algorithmic choices belong here. The main residual issue is not relevance. It is presentation. The chapter should do a better job telling the economist reader why these methods matter before immersing them in RL-native terminology and benchmark examples.

## Chapter 3: theory of reinforcement learning

Scope of review:
- Reviewed `thesis_v2/ch03_theory/tex/planning_learning_v3.tex`
- Checked major internal references and theorem labels within the chapter
- Did not edit thesis source

### Overall judgment

This remains one of the strongest chapters in the thesis. It has a clear intellectual core: RL is presented as an extension of dynamic programming through stochastic approximation, approximation theory, and policy optimization. It also serves your bridge objective better than most technical theory chapters because several sections connect directly to economic computation. The main residual problem is density. The chapter is mathematically rich, but it still explains too many advanced concepts at once for an economics committee reader, especially in footnotes and optimization-heavy subsections.

### What is working

- The chapter has a strong organizing claim from the start. `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:12` makes the PI-as-Newton argument explicit and then carries it through the chapter.
- The Brock-Mirman simulation is a good economist-facing anchor. `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:45` gives a familiar economic model rather than a generic control benchmark.
- The rollout, lookahead, and AlphaZero section is one of the best bridge sections in the thesis. `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:142` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:179` justifies keeping AlphaZero by tying it directly to Bertsekas-style planning logic.
- The deadly-triad section is conceptually strong. `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:181` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:225` explains the source of instability more clearly than many RL expositions do.
- The actor-critic and entropy-regularization sections connect well to later chapters. `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:321` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:353` explains why these hybrid methods matter without reading as pure benchmark reporting.

### Concrete problems

- **An internal table reference still appears unresolved.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:55` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:61` refer to `Table~\ref{tab:brock_mirman}`, but I did not find a corresponding `\label{tab:brock_mirman}` in the thesis copy.
- **The chapter opens at a very high technical level.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:12` is conceptually strong, but the supporting material quickly moves into `Picard iteration`, `supporting hyperplane`, `semismooth`, and `B-subdifferential`. That is a steep entry point for economists.
- **The off-policy paragraph repeats an over-strong causal formulation.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:186` says off-policy evaluation answers the counterfactual question at the heart of policy comparison. That is too close to the same overstatement already present in the introduction.
- **The final tradeoffs subsection re-expands into survey mode.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:370` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:373` reads more like a broad RL summary than a conclusion to this chapter’s main theoretical argument.

### Jargon economists may not understand quickly

- **Optimization language is very dense in the opening section.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:21` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:30` uses `Picard iteration`, `supporting hyperplane`, `semismooth`, and `B-subdifferential` very early.
- **The stochastic-approximation section assumes mathematical background beyond most applied readers.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:71` uses `Robbins-Monro`, `ODE method`, and `Lyapunov theory` in the opening footnote.
- **The projected-operator discussion is conceptually important but technically loaded.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:188` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:205` uses `orthogonal projection`, `oblique projection`, `stationary distribution`, and `double-sampling` in quick succession.
- **The policy-gradient section becomes optimization-heavy.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:277` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:309` introduces `gradient domination`, `Polyak-Łojasiewicz`, `dimension-free convergence`, `information geometry`, and `majorization-minimization`.
- **The planning-complexity section is jargon-rich and citation-heavy.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:364` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:368` uses `generative model`, `effective horizon`, `minimax-optimal sample complexity`, and recent rate refinements that many economists will not need.
- **Footnotes often carry their own miniature theory lectures.** This is especially true at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:30`, `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:71`, `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:287`, `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:309`, and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:366`.

### Framing and proportion issues

- **The core argument is excellent, but some local sections are too encyclopedic.** The chapter works best when it pushes one of three claims: PI is Newton-like, value learning is stochastic approximation under approximation error, or modern policy methods approximate classical planning/improvement. It weakens when it broadens into general optimization geometry or survey-style tradeoffs.
- **The Brock-Mirman simulation belongs here, but some exposition around it is heavier than necessary.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:50` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:61` is valuable, but the scalar-envelope explanation at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:59` is dense and probably longer than many readers need.
- **The rollout/lookahead/AlphaZero section is worth preserving almost entirely.** Unlike some other retained AlphaGo material, this block earns its space because it directly supports the Bertsekas bridge and explains why search helps.
- **The planning-complexity section is informative but less central to the economist-facing bridge.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:364` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:368` is mathematically interesting, but less central than the Newton, deadly-triad, and rollout/lookahead arguments.
- **The `Fundamental Tradeoffs` subsection is the least essential part of the chapter.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:370` feels like inherited survey breadth rather than the main theory spine.

### Best material to preserve

- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:12`, because the PI-as-Newton claim is the chapter’s central contribution.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:45`, because Brock-Mirman is a strong economist-facing simulation anchor.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:142`, because rollout, lookahead, and AlphaZero are where the Bertsekas insight becomes operational.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:181`, because the deadly triad is essential for understanding why approximate RL is hard.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:321`, because actor-critic is necessary for later chapters and is explained in a way that still connects back to theory.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:355`, because the Singh-Yee error-amplification bound is a useful economic-computation message.

### Lowest-value density if more tightening is needed

- The most technical footnote under the Newton interpretation at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:30`, especially the `semismooth` and `B-subdifferential` discussion.
- The scalar-envelope geometry paragraph at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:59`, if the chapter needs local compression without losing the main Newton argument.
- The planning-complexity block at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:364` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:368`, if the goal is economist readability rather than full survey completeness.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:370`, the `Fundamental Tradeoffs` subsection, which still feels the least necessary to the chapter’s own argument.

### Bottom line

This chapter is substantively excellent and is one of the strongest retained parts of the thesis. It earns its place. The main remaining issue is not conceptual weakness. It is density. The chapter often says the right things, but in a register that is more mathematically specialized than many economists will want in one sitting. If you want one theory chapter to survive the trimming intact, this is probably it. But if you want it to land better with economists, the biggest gains now come from trimming dense footnotes, softening causal-style phrasing, and reducing the residual survey-style tradeoffs ending.

## Chapter 4: structural estimation with reinforcement learning

Scope of review:
- Reviewed `thesis_v2/ch05_econ_models/tex/rl_in_se.tex`
- Checked chapter structure, economist-facing clarity, and broad scope fit
- Did not edit thesis source

### Overall judgment

This is one of the most economist-friendly chapters in the thesis. It opens with the right claim: RL is a computational method embedded inside structural estimation, not a live decision-maker interacting with real markets. The DDC material and the simulation study are especially strong. The main remaining problem is scope. The chapter is titled and introduced as structural estimation, but it still bundles together several distinct topics: DDC estimation, dynamic oligopoly computation, mechanism design, and macro/policy applications. That breadth makes the chapter feel more like a curated survey than a tightly organized thesis chapter.

### What is working

- The opening frame is good. `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1` makes clear that RL is being used as a numerical method inside an econometric model.
- The NFXP comparison is exactly the right economist-facing anchor. `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:3` situates the entire chapter relative to a canonical econometric benchmark.
- The single-agent DDC material is strong and clearly central. `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:18` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:93` reads as a genuine economics-RL bridge rather than imported RL content.
- The simulation study is highly valuable. `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:191` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:207` gives the reader a direct computational comparison against NFXP and CCP, which is exactly what economists care about.
- The chapter is materially more readable for economists than the earlier algorithms and theory chapters.

### Concrete problems

- **Survey language still appears in a thesis chapter.** The footnote at `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1` says `the rest of this survey`, which is inconsistent with the document’s thesis framing.
- **The chapter title and content do not fully match.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:18`, `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:96`, `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:145`, and `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:186` cover fairly different objects. Dynamic procurement auctions, merger analysis, sequential price mechanisms, and macro policy design are related applications, but they are not all naturally read as `structural estimation`.
- **The macro/policy subsection is the weakest fit.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:186` broadens into macro solution methods and automated tax design. That material may be interesting, but it is the least tightly connected to the chapter’s NFXP-to-RL bridge.
- **Some companion-work framing remains visible.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1` and the collusion footnote at `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:142` still point outward to companion survey or companion chapter material, which keeps a bit of the original survey architecture alive.

### Jargon economists may not understand quickly

- **The chapter is relatively accessible, but the first paragraph is dense.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1` introduces `Q-learning`, `temporal-difference learning`, `policy gradient`, `actor-critic`, `MDP`, and `action-value function` all at once.
- **The TD-CCP subsection uses RL and semiparametric jargon quickly.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:22` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:47` includes `semi-gradient`, `AVI`, `LASSO`, `random forests`, `PMLE`, and `locally robust correction`. Most are explained, but the density is still high.
- **The policy-gradient DDC subsection assumes familiarity with RL optimization.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:77` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:91` uses `REINFORCE-style gradient ascent`, `action-value function`, and forward simulation of latent states.
- **The mechanism-design subsections introduce several specialized acronyms.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:150` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:183` uses `SPM`, `SOSP`, `POMDP`, `PPO`, `SAC`, `DDPG`, and `DQN`. Some economists will know the mechanism-design side better than the RL side, but many will still find the acronym load heavy.

### Framing and proportion issues

- **The DDC material is the chapter’s clearest center of gravity.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:18` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:93` plus the simulation at `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:191` form the strongest internal arc.
- **The dynamic-oligopoly section still fits reasonably well, but it is already broader than the title suggests.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:96` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:142` is still economist-facing and connected to equilibrium computation, but it is more about solving dynamic games than estimating single-agent structural models.
- **The auction-mechanism section is interesting but more distant from the chapter’s opening frame.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:145` through `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:183` is relevant to economics, but it tilts away from `RL inside structural estimation` and toward `RL for computational mechanism design`.
- **The macro/policy subsection feels like residual breadth.** `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:186` is the most obvious candidate for scope trimming if the chapter needs to feel tighter.

### Best material to preserve

- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1`, because the opening correctly frames RL as a numerical method rather than an economic agent.
- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:3`, because the NFXP comparison gives economists the right benchmark immediately.
- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:22`, because the Adusumilli-Eckardt TD-CCP section is one of the cleanest bridges in the thesis.
- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:67`, because the SMM-plus-policy-gradient section shows a distinct route through latent-state DDC estimation.
- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:191`, because the DDC scaling simulation is exactly the kind of evidence an economics committee will value.

### Lowest-value density if more tightening is needed

- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:186`, the macro and optimal-policy subsection, which is the least tightly connected to the chapter’s main bridge argument.
- `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:145`, if additional narrowing is needed. The mechanism-design material is interesting, but less central than DDC estimation and the scaling comparison.
- Some outward-pointing companion-work footnotes, especially `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:1` and `thesis_v2/ch05_econ_models/tex/rl_in_se.tex:142`, if the goal is to make this read as a self-contained thesis chapter rather than a reduced survey.

### Bottom line

This chapter fits the thesis well, especially in its DDC sections and simulation study. It is already much closer to economist-facing prose than the earlier chapters. The main remaining weakness is not jargon so much as scope discipline. If you want the chapter to feel tighter and more thesis-like, the gains come from centering the DDC-and-computation bridge even more firmly and treating the macro and mechanism-design spillovers as optional breadth rather than coequal content.

## Chapter 5: reinforcement learning in games

Scope of review:
- Reviewed `thesis_v2/ch06_games/tex/rl_in_games.tex`
- Checked chapter structure and economist-facing clarity
- Did not edit thesis source

### Overall judgment

This chapter fits the thesis reasonably well, but less tightly than the structural-estimation chapter. Its best material is strongly economist-facing: the Cournot-Bertrand benchmark and the Coase-conjecture application. Its weaker material is the poker and neural-games layer, which still reads more like inherited survey breadth than core thesis argument. The chapter is cleaner than a generic MARL survey, but it still opens in a game-learning taxonomy rather than in the economic questions that justify its inclusion.

### What is working

- The chapter has two strong economist-facing anchors. `thesis_v2/ch06_games/tex/rl_in_games.tex:79` gives Cournot and Bertrand benchmarks, and `thesis_v2/ch06_games/tex/rl_in_games.tex:122` gives a durable-goods monopoly application with a clear economic payoff.
- The Cournot-Bertrand simulation is useful because it grounds the abstract learning algorithms in canonical industrial-organization games. See `thesis_v2/ch06_games/tex/rl_in_games.tex:82` and `thesis_v2/ch06_games/tex/rl_in_games.tex:95`.
- The Coase section is one of the strongest retained applications in the whole thesis. `thesis_v2/ch06_games/tex/rl_in_games.tex:126` through `thesis_v2/ch06_games/tex/rl_in_games.tex:159` gives a clean case where CFR recovers a textbook economics result.
- I did not find broken internal references in this chapter.

### Concrete problems

- **The chapter opens in RL/game-theory survey voice rather than economics-first voice.** `thesis_v2/ch06_games/tex/rl_in_games.tex:3` through `thesis_v2/ch06_games/tex/rl_in_games.tex:5` begins with desiderata, paradigm classification, and complexity hardness before stating why economists should care.
- **The poker/neural block is still the least thesis-like part of the chapter.** `thesis_v2/ch06_games/tex/rl_in_games.tex:116` through `thesis_v2/ch06_games/tex/rl_in_games.tex:120` is coherent, but it is more prestige literature review than economist-facing bridge work.
- **Some claims remain stronger than the rest of the thesis voice.** `thesis_v2/ch06_games/tex/rl_in_games.tex:114` says CFR+ enabled heads-up limit hold'em to be `essentially solved`, and `thesis_v2/ch06_games/tex/rl_in_games.tex:120` highlights Libratus and Pluribus defeating top professionals. These are standard descriptions, but they also give poker more prominence than the economics material needs.
- **The chapter’s best economic material arrives after a long technical setup.** A committee reader has to move through Minimax-Q, Nash-Q, WoLF-PHC, and CFR theory before reaching the strongest economics content.

### Jargon economists may not understand quickly

- **The opening is acronym- and concept-heavy.** `thesis_v2/ch06_games/tex/rl_in_games.tex:3` through `thesis_v2/ch06_games/tex/rl_in_games.tex:5` uses `stationary-transition assumption`, `Bellman operator`, `CFR`, and `PPAD-complete` immediately.
- **The stochastic-games section is technically dense.** `thesis_v2/ch06_games/tex/rl_in_games.tex:11` through `thesis_v2/ch06_games/tex/rl_in_games.tex:46` introduces `stochastic game`, `rationality`, `convergent`, `Minimax-Q`, `Nash-Q`, and stage-game equilibrium computation.
- **WoLF-PHC is hard to parse for non-specialists.** `thesis_v2/ch06_games/tex/rl_in_games.tex:68` through `thesis_v2/ch06_games/tex/rl_in_games.tex:77` uses `policy hill-climbing`, `WoLF-IGA`, `piecewise ellipses`, and evolutionary-game analogies.
- **The CFR section is important but mathematically compressed.** `thesis_v2/ch06_games/tex/rl_in_games.tex:98` through `thesis_v2/ch06_games/tex/rl_in_games.tex:114` introduces `counterfactual value`, `reach probability`, `regret matching`, `exploitability`, and asymptotic equilibrium rates.
- **The poker subsection adds more specialized jargon.** `thesis_v2/ch06_games/tex/rl_in_games.tex:116` through `thesis_v2/ch06_games/tex/rl_in_games.tex:120` uses `Deep CFR`, `NFSP`, `fictitious play`, `subgame solving`, and `mbb/g`.

### Framing and proportion issues

- **The chapter’s core bridge is narrower than the chapter’s full content.** The strongest bridge is not `multi-agent RL` in general. It is `RL and game-learning methods applied to economic games economists recognize`.
- **Cournot-Bertrand and Coase should dominate the reader’s memory of the chapter.** Those are the places where the chapter feels most thesis-like and least survey-like.
- **The poker material still occupies prestige space rather than bridge space.** `thesis_v2/ch06_games/tex/rl_in_games.tex:116` through `thesis_v2/ch06_games/tex/rl_in_games.tex:120` is informative, but it is much less central for an economics committee than the IO and bargaining sections.
- **The convergence-comparison material is useful but could feel abstract to economists.** `thesis_v2/ch06_games/tex/rl_in_games.tex:48` through `thesis_v2/ch06_games/tex/rl_in_games.tex:77` helps organize the learning algorithms, but it is still more methodological than economic.

### Best material to preserve

- `thesis_v2/ch06_games/tex/rl_in_games.tex:79`, because Cournot and Bertrand are canonical economist-facing benchmarks.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:95`, because the interpretation is restrained and economically meaningful.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:98`, because CFR is necessary for the later Coase application and for imperfect-information games.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:122`, because the Coase-conjecture application is one of the best bridge examples in the chapter.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:159`, because the computational results show the method recovering a classic economics prediction rather than merely winning a benchmark game.

### Lowest-value density if more tightening is needed

- `thesis_v2/ch06_games/tex/rl_in_games.tex:3` through `thesis_v2/ch06_games/tex/rl_in_games.tex:5`, if you want a more economics-first opening.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:68`, if the convergence-taxonomy discussion needs compression.
- `thesis_v2/ch06_games/tex/rl_in_games.tex:116` through `thesis_v2/ch06_games/tex/rl_in_games.tex:120`, because the poker/neural block remains the least economist-facing part of the chapter.

### Bottom line

This chapter belongs in the thesis, but mainly because of its economics applications, not because of its poker coverage. The Cournot-Bertrand and Coase sections justify it. The main remaining issue is emphasis. If the chapter is meant to serve an economics committee, it should read less like a compact multi-agent-RL survey and more like a chapter about learning and equilibrium computation in economic games.
