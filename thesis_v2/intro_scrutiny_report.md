# Intro Scrutiny Report

Scope of review:
- Reviewed `thesis_v2/ch00_introduction/tex/intro.tex`
- Cross-checked framing against `thesis_v2/ch00_introduction/tex/abstract.tex`
- Verified all intro `\ref{...}` targets against labels present in `thesis_v2/`
- Did not edit thesis source

## Overall judgment

The introduction is structurally coherent and its internal cross-references resolve, but it still carries several tensions from the pre-cut survey version. The main issues are not broken LaTeX references inside the intro. They are conceptual overstatements, scope drift toward material no longer in the thesis, and framing that still reads as a broad survey rather than a narrower economist-facing thesis.

## Confirmed clean

- The roadmap paragraph in `thesis_v2/ch00_introduction/tex/intro.tex:11` matches the chapters retained in `thesis_v2/docs/main.tex`.
- The merged `Two Cultures` material is now internal to the introduction rather than a separate top-level section, centered at `thesis_v2/ch00_introduction/tex/intro.tex:14`.
- All `\ref{...}` calls used in the intro resolve in the thesis copy. I found no broken intro cross-references.

## Major inconsistencies

- **Training/deployment contradiction.** `thesis_v2/ch00_introduction/tex/intro.tex:37` says every RL system has a frozen execution phase with no further updates, but the same paragraph says bandit algorithms update during deployment. Those two claims are not consistent as stated.
- **Forward-problem framing is too narrow for the retained chapters.** `thesis_v2/ch00_introduction/tex/intro.tex:11` says the thesis studies optimal policies in a known or simulated environment, but the retained thesis still includes structural estimation and offline RL, where the environment is not simply known and the central object is often not only policy computation.
- **Off-policy evaluation is equated too strongly with causal counterfactuals.** `thesis_v2/ch00_introduction/tex/intro.tex:29` says off-policy evaluation is precisely counterfactual policy evaluation. That is too strong after cutting the dedicated causal chapter, because it risks collapsing statistical policy evaluation into causal identification.
- **Simulator claim is too broad.** `thesis_v2/ch00_introduction/tex/intro.tex:87` says RL typically assumes access to a high-fidelity simulator, but the thesis itself retains offline RL and discussion of live-market learning, which sit outside that framing.

## Scope drift after cuts

- **Broad omitted-literature paragraph remains too prominent.** `thesis_v2/ch00_introduction/tex/intro.tex:9` still spends substantial space on collusion, heterogeneous-agent macro, finance, and IRL. As written, it reads like a master-survey disclaimer rather than an intro to the reduced thesis version.
- **Bandit material still anchors several explanations.** Bandits appear as organizing examples in `thesis_v2/ch00_introduction/tex/intro.tex:37` and `thesis_v2/ch00_introduction/tex/intro.tex:83`, even though bandits are no longer a thesis chapter.
- **AlphaGo-style examples remain more visible than economist-facing examples.** `thesis_v2/ch00_introduction/tex/intro.tex:37` and the lifecycle table beginning at `thesis_v2/ch00_introduction/tex/intro.tex:42` still lean on AlphaGo and bandit examples more than on the retained structural-estimation or offline-RL examples.
- **IRL still receives notable conceptual attention.** `thesis_v2/ch00_introduction/tex/intro.tex:9` and `thesis_v2/ch00_introduction/tex/intro.tex:89` devote visible framing space to IRL and identifiability even though that topic is delegated to the companion survey.

## Framing mismatches

- **Still written as a survey.** The opening at `thesis_v2/ch00_introduction/tex/intro.tex:1`, the scope paragraph at `thesis_v2/ch00_introduction/tex/intro.tex:9`, and the roadmap at `thesis_v2/ch00_introduction/tex/intro.tex:11` all speak in the voice of a broad survey rather than a thesis derivative.
- **Opening is RL-first, not economics-first.** `thesis_v2/ch00_introduction/tex/intro.tex:1` starts with RL methods and Bellman connections before stating the economic question or why economists should care about the retained chapter set.
- **Abstract and intro are aligned with each other, but both are still survey-framed.** See `thesis_v2/ch00_introduction/tex/abstract.tex:1` and `thesis_v2/ch00_introduction/tex/intro.tex:1`.

## Precision and overstatement issues

- **DP versus RL contrast is too sharp.** `thesis_v2/ch00_introduction/tex/intro.tex:3` presents the distinction in a hard binary way that is rhetorically effective but technically oversimplified.
- **Convergence claim is too sweeping.** `thesis_v2/ch00_introduction/tex/intro.tex:5` says RL has only sublinear convergence guarantees. For an intro that later develops more nuanced theory, that statement is too blunt.
- **The “key takeaway” sentence is more didactic than thesis-like.** `thesis_v2/ch00_introduction/tex/intro.tex:39` reads like explanatory workshop prose rather than measured thesis prose.

## Style and wording issues

- `thesis_v2/ch00_introduction/tex/intro.tex:3` has `a incremental`.
- `thesis_v2/ch00_introduction/tex/intro.tex:5` has awkward punctuation in `This is however, quite sufficient in practice`.
- `thesis_v2/ch00_introduction/tex/intro.tex:39` has `whether in a computer or in field`.
- `thesis_v2/ch00_introduction/tex/intro.tex:3` uses `This allows us`, which is slightly off the single-author thesis voice.

## Priority order

If this introduction gets one more revision pass, the highest-value fixes are:

1. Resolve the contradiction in `thesis_v2/ch00_introduction/tex/intro.tex:37`.
2. Reframe `thesis_v2/ch00_introduction/tex/intro.tex:1` and `thesis_v2/ch00_introduction/tex/intro.tex:11` so the retained thesis scope is described accurately.
3. Trim or demote the omitted-literature paragraph in `thesis_v2/ch00_introduction/tex/intro.tex:9`.
4. Remove or soften the strongest overstatements in `thesis_v2/ch00_introduction/tex/intro.tex:29`, `thesis_v2/ch00_introduction/tex/intro.tex:5`, and `thesis_v2/ch00_introduction/tex/intro.tex:87`.
5. Replace non-retained anchor examples, especially bandits and AlphaGo, with examples from the chapters that remain central to the thesis.

## Algorithms chapter review

Scope of review:
- Reviewed `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex`
- Verified chapter `\ref{...}` targets against labels present in `thesis_v2/`
- Did not edit thesis source

### Overall judgment

The chapter is internally coherent and its references resolve, but it still reflects the wider survey rather than the narrowed thesis. The main problems are not broken cross-references. They are conceptual overreach in a few explanations, residual emphasis on domains that were cut elsewhere, and disproportionate space given to material that does not appear central to the retained economist-facing thesis.

### Confirmed clean

- All `\ref{...}` calls in `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex` resolve in the thesis copy.
- The chapter no longer contains `Decision Transformers`, which was one of the intended cuts.
- The retained algorithm spine is visible and coherent: Monte Carlo, TD, Q-learning, SARSA, actor-critic, natural policy gradient, fitted methods, DQN, PPO/TRPO, and SAC.

### Major inconsistencies or precision issues

- **Monte Carlo opening is conceptually off.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:11` says that when `P(s'|s,a)` and `r(s,a)` are unknown, the obvious approach is to use Monte Carlo to approximate them. The paragraph that follows does not approximate `P` or `r`; it estimates returns and value functions from sampled episodes. The first sentence promises one object and the paragraph delivers another.
- **Independence claim is too strong.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:15` says each first-visit return is an independent draw from the return distribution. That is stronger than the surrounding exposition supports and reads too cleanly for sequential data generated under a policy.
- **Control-as-inference section overreaches conceptually.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:206` says the framework recasts the preceding algorithms in a single model and that the forward and inverse problems become two queries in the same model. In the reduced thesis, this is broader than the chapter needs, and it reopens the IRL connection after that material was intentionally moved out.
- **Framework-recovery claims are expansive.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:268` presents a long “single model recovers familiar algorithms” argument, including SAC, soft Q-learning, TRPO, PPO, hard-max limits, and inverse RL. The line of argument is mathematically interesting, but in this chapter it reads as a unifying claim larger than the later thesis scope can fully support.

### Scope drift after cuts

- **Continuous-control framing remains prominent.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:57` uses a robot-arm torque example to motivate policy gradients, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:193` emphasizes continuous-control benchmarks for SAC. Since optimal control was cut as a thesis chapter, these examples now pull the chapter back toward the broader survey.
- **Control-as-inference is still a substantial standalone subsection.** The subsection beginning at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:203` is mathematically rich, but it is also one of the clearest remnants of the material that had already been identified as less essential to the narrowed thesis.
- **AlphaGo Zero still occupies a very large share of the deep-learning era.** The subsection begins at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:272` and runs through dense architectural, search, and training details up to `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:383`. For the reduced thesis, this is a lot of real estate devoted to a case study that is not one of the retained economist-facing applications.
- **Bandit logic re-enters through AlphaGo.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:279` explains MCTS using a UCB analogy from bandit theory, which is accurate but also reintroduces a cut domain as a core explanatory anchor.

### Framing mismatches

- **The chapter still reads like a broad RL methods survey.** That is most visible in the sweep from `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:145` through `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:383`, where the narrative moves from Atari to probabilistic inference to AlphaGo rather than staying tightly tied to the later thesis chapters.
- **Economist-facing linkage is uneven.** The chapter occasionally reconnects to structural estimation and discrete choice, for example at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:187`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:266`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:383`, but much of the deep-learning-era exposition still foregrounds canonical ML successes rather than the economic problems retained in the thesis.

### Style and proportion issues

- **AlphaGo subsection is disproportionately detailed.** The material at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:277`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:279`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:377` goes deep into input planes, residual blocks, PUCT, Dirichlet noise, TPUs, and match history. This level of detail is hard to justify relative to the trimmed thesis scope.
- **A few statements are more promotional than measured.** `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:179` calls PPO the default for large-scale RL applications, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:381` describes a “virtuous cycle.” Both are understandable in context, but they are less understated than the rest of the thesis now aims to be.
- **The chapter’s examples skew away from the retained thesis applications.** Robot arms, Atari, backgammon, and Go dominate the exposition in `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:57`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:150`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:63`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:275`, while the later thesis centers structural estimation, games, and offline RL.

### Bottom line on this chapter

The chapter is technically intact and substantially cleaner than the old survey version, but it is not yet fully aligned with the narrowed thesis philosophy. The core algorithm spine is fine. The main residual issue is that the deep-learning-era material still spends too much time on control-as-inference and AlphaGo-style exposition relative to what the rest of the thesis actually uses.

### Priority order for a future pass

1. Fix the opening conceptual mismatch at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:11`.
2. Decide whether `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:203` should remain as a full subsection or be reduced sharply.
3. Reduce the weight of the AlphaGo subsection beginning at `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:272`.
4. Replace or rebalance non-retained examples, especially continuous-control and Go examples, with examples that point forward to structural estimation, games, and offline RL.
5. Soften broad or promotional claims in `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:15`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:179`, `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:206`, and `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:381`.

## Theory chapter review

Scope of review:
- Reviewed `thesis_v2/ch03_theory/tex/planning_learning_v3.tex`
- Verified chapter `\ref{...}` targets against labels present in `thesis_v2/`
- Did not edit thesis source

### Overall judgment

This is one of the strongest substantive chapters in the thesis. The central argument, that RL extends classical dynamic programming through stochastic approximation and approximation theory, is clear and valuable. The main issues are not conceptual confusion at the chapter level. They are unresolved internal references, residual dependence on examples from cut areas, and a tendency for some sections to keep the breadth of the original survey rather than the narrower thesis.

### Confirmed clean

- The chapter has a coherent spine from Bellman geometry through stochastic approximation, fitted methods, the deadly triad, policy gradients, actor-critic methods, and tradeoffs.
- Most internal and cross-chapter references resolve correctly.
- The chapter’s core thesis remains well aligned with the reduced document: RL is presented as an extension of dynamic programming rather than a disconnected ML toolkit.

### Concrete problems

- **Two internal references are unresolved.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:55` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:61` refer to `Table~\ref{tab:brock_mirman}`, but I did not find a corresponding `\label{tab:brock_mirman}` in the thesis copy. Likewise, `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:143` refers to `Table~\ref{tab:td_lambda_corridor}`, but I did not find a corresponding `\label{tab:td_lambda_corridor}`.
- **RLHF enters the theory chapter as an application anchor.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:419` says PPO is dominant in large-scale applications including RLHF. The reference resolves, but it pulls the theory chapter toward a modern-application narrative rather than the narrower mathematical thread the chapter otherwise maintains.

### Scope drift after cuts

- **Control-flavored simulations remain prominent.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:173` is a full simulation study on linear-quadratic control, and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:419` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:426` use an LQC monetary-policy trust-region figure to explain PPO and TRPO. Since the separate optimal-control chapter was cut, these examples now carry more weight than the narrowed thesis framing would suggest.
- **AlphaZero remains a substantial theory case study.** The subsection beginning at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:217` develops rollout, lookahead, and AlphaZero in detail, and it depends on the already-large AlphaGo exposition in the algorithms chapter via `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:235`. This is mathematically interesting, but it is also one of the clearest remnants of the broad survey architecture.
- **Bandit material re-enters through the tradeoffs section.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:496` uses Lai-Robbins and UCB as the main explanation of exploration-exploitation tradeoffs. That is standard and useful, but after the bandits chapter was cut it again makes a removed domain central to the thesis narrative.
- **The chapter still accumulates many worked examples from outside the retained applications.** Brock-Mirman is highly relevant, but the chapter also leans on corridor credit assignment, linear-quadratic control, AlphaZero, and monetary-policy trust-region geometry. Together, these examples preserve the full-survey breadth inside a chapter that otherwise fits the reduced thesis very well.

### Framing and proportion issues

- **Some simulations look more like survey carryovers than thesis necessities.** The fitted-method and trust-region sections rely heavily on control-style examples at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:173` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:426`, even though the thesis no longer includes a standalone control chapter.
- **AlphaZero appears twice in force across the thesis.** It already occupies substantial space in `Algorithms`, and the theory chapter gives it another substantial conceptual role at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:217`. The duplication is not exact repetition, but it still gives AlphaZero more cumulative weight than any of the retained economics-facing applications.
- **The tradeoffs section broadens outward again at the end.** `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:496` to `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:502` reads like a compact general-RL survey of canonical tradeoffs rather than a distilled setup for the chapters that follow.

### Style and precision issues

- **A few captions carry interpretation rather than only description.** The LQC caption at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:426` through `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:438` explains the substantive lessons of the figure in detail, not just what is shown. That is informative, but it is heavier than the thesis’s stated caption discipline.
- **Some prose is more rhetorical than the current thesis voice elsewhere.** The conclusion begins `RL algorithms are not mysterious` at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:507`. The claim is clear, but the phrasing is more assertive and essay-like than the quieter thesis tone now aimed for in the rest of the trimmed document.
- **The chapter remains dense even after cuts.** The mathematical density is appropriate, but several long explanatory blocks still read like comprehensive survey exposition rather than a tighter thesis theory chapter.

### Bottom line on this chapter

This chapter is substantively strong and probably the best-kept part of the reduced thesis. It earns its place. The main issues are narrower than in the introduction or algorithms chapter. They are mostly about cleanup and scope discipline: fix the broken internal references, decide how much control- and AlphaZero-oriented material the thesis still wants to foreground, and keep the chapter from re-expanding into a general RL theory survey at the end.

### Priority order for a future pass

1. Fix the unresolved internal references at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:55`, `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:61`, and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:143`.
2. Reassess how much of the control-flavored material centered at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:173` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:426` still belongs in the trimmed thesis.
3. Reassess the size of the AlphaZero subsection beginning at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:217`, especially given the parallel coverage in `Algorithms`.
4. Decide whether the tradeoffs section at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:496` should remain as a broad RL wrap-up or be tightened toward the applications that actually remain in the thesis.
5. Soften the most rhetorical phrasing and overly interpretive caption material, especially at `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:426` and `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:507`.

## Structural estimation chapter review

Scope of review:
- Reviewed `ch05_econ_models/tex/rl_in_se.tex`, which is the source pulled into the thesis build via `thesis_v2/docs/main.tex:155`
- Verified chapter `\ref{...}` targets against labels present in the thesis copy
- Checked the thesis-copy folder structure for this chapter
- Did not edit thesis source

### Overall judgment

This chapter contains some of the most economist-facing material in the thesis, and the DDC estimation simulation is one of the clearest retained pieces of evidence for why RL matters here. The main issues are structural rather than purely local. The chapter title in the thesis is `Structural Estimation with Reinforcement Learning`, but the file now ranges well beyond structural estimation into dynamic oligopoly, auction design, macro solution methods, and tax-policy design. That breadth made sense in the full survey. In the trimmed thesis, it makes the chapter feel less tightly scoped than the title promises.

### Important architecture note

- **This chapter is not isolated in the thesis copy.** `thesis_v2/ch05_econ_models` is a symlink to `../ch05_econ_models`, not an independent thesis-specific copy. This means the thesis version is not fully separated from the original for this chapter, unlike the copied intro, algorithms, and theory chapters. That matters if future thesis-specific edits are supposed to leave the original untouched.

### Confirmed clean

- I did not find broken chapter cross-references in `ch05_econ_models/tex/rl_in_se.tex`.
- The opening footnote at `ch05_econ_models/tex/rl_in_se.tex:1` does useful conceptual work by clarifying that the RL loop here runs inside the econometrician’s model rather than in a real environment.
- The simulation study at `ch05_econ_models/tex/rl_in_se.tex:202` through `ch05_econ_models/tex/rl_in_se.tex:224` is well aligned with the thesis’s economist-facing motivation.

### Major inconsistencies

- **Chapter title versus chapter contents.** The thesis labels this section `Structural Estimation with Reinforcement Learning`, but the chapter includes strategic auction equilibrium computation at `ch05_econ_models/tex/rl_in_se.tex:107`, merger-and-innovation dynamics at `ch05_econ_models/tex/rl_in_se.tex:130`, mechanism design at `ch05_econ_models/tex/rl_in_se.tex:151`, macro solution methods at `ch05_econ_models/tex/rl_in_se.tex:192`, and tax-policy design at `ch05_econ_models/tex/rl_in_se.tex:197`. Those are not all naturally “structural estimation.”
- **The simulation supports only part of the chapter’s scope.** The empirical anchor at `ch05_econ_models/tex/rl_in_se.tex:202` is a DDC estimation scaling exercise. It strongly supports the single-agent structural-estimation story, but it does not anchor the broader claims made in the oligopoly, mechanism design, macro, or policy-design subsections.

### Scope drift after cuts

- **The chapter still reflects the full survey’s breadth.** The single-agent estimation material at `ch05_econ_models/tex/rl_in_se.tex:24` and the DDC simulation at `ch05_econ_models/tex/rl_in_se.tex:202` fit the trimmed thesis very well. The later sections, especially `Macroeconomic Models` at `ch05_econ_models/tex/rl_in_se.tex:192` and `Optimal Policy Design` at `ch05_econ_models/tex/rl_in_se.tex:197`, reopen the wider “RL across economics” survey logic.
- **The macro subsection is especially survey-like.** `ch05_econ_models/tex/rl_in_se.tex:194` is a compact literature sweep rather than a tightly integrated chapter component. It reads more like “further territory exists” than like part of the retained thesis argument.
- **The AI Economist subsection stretches the chapter furthest from its title.** `ch05_econ_models/tex/rl_in_se.tex:199` is interesting and economists may care about it, but it is not structural estimation in the ordinary sense. In the trimmed thesis, it looks like a holdover from the broader survey rather than a natural end point of the chapter.

### Framing and organization issues

- **The opening paragraph is overloaded.** `ch05_econ_models/tex/rl_in_se.tex:1` does many things at once: scope restriction, exclusion of IRL, notation unification, definition of the MDP, and value-function notation. It is informative, but dense for a chapter whose readership is now supposed to be economists first.
- **The chapter may now contain two different organizing logics.** One logic is “RL as a numerical tool for structural estimation,” visible in the opening and DDC sections. The other is “RL in computational industrial organization and mechanism design more broadly,” visible from `ch05_econ_models/tex/rl_in_se.tex:102` onward. Those are related, but not identical.
- **Some sections are descriptive literature review rather than integrated argument.** This is most evident in `ch05_econ_models/tex/rl_in_se.tex:194`, where the macro subsection summarizes several papers without clearly tying them back to the chapter’s main estimation story.

### Style and tone issues

- **A few claims are stronger than the current thesis voice elsewhere.** `ch05_econ_models/tex/rl_in_se.tex:199` says the learned tax policies `Pareto-dominate` analytical baselines. If retained, that kind of wording bears more argumentative weight than the understated thesis voice usually wants.
- **The simulation summary is strong but still somewhat conclusion-like.** `ch05_econ_models/tex/rl_in_se.tex:224` states the numerical findings cleanly, but the phrase `validating the AVI approach` is more assertive than the report-like tone used elsewhere in the trimmed thesis.

### Bottom line on this chapter

This is a valuable chapter for the thesis, but it is now somewhat misnamed relative to what it contains. Its best core is the DDC and single-agent structural-estimation material. Its main weakness is not bad prose or broken references. It is that the chapter still bundles several adjacent but distinct topics under the structural-estimation umbrella, and it remains linked to the original survey through a symlink rather than a fully separate thesis copy.

### Priority order for a future pass

1. Decide whether this chapter is truly about structural estimation only, or whether its title should reflect its broader scope.
2. If the thesis is meant to remain isolated from the original, replace the symlinked chapter source with a real thesis-specific copy before making further chapter-specific edits.
3. Reassess whether `Macroeconomic Models` at `ch05_econ_models/tex/rl_in_se.tex:192` and `Optimal Policy Design` at `ch05_econ_models/tex/rl_in_se.tex:197` still belong in this trimmed thesis chapter.
4. Tighten the opening paragraph at `ch05_econ_models/tex/rl_in_se.tex:1` so the organizing claim is visible earlier.
5. Soften the strongest evaluative wording, especially at `ch05_econ_models/tex/rl_in_se.tex:199` and `ch05_econ_models/tex/rl_in_se.tex:224`.

## Games chapter review

Scope of review:
- Reviewed `ch06_games/tex/rl_in_games.tex`, which is the source pulled into the thesis build via `thesis_v2/docs/main.tex:159`
- Verified chapter `\ref{...}` targets against labels present in the thesis copy
- Checked the thesis-copy folder structure for this chapter
- Did not edit thesis source

### Overall judgment

This chapter fits the trimmed thesis better than several of the broader-survey chapters. The Cournot-Bertrand simulation and the Coase conjecture section are both clearly economist-facing and well chosen. The main residual issue is that the chapter still carries a substantial poker and neural-games layer from the original survey. That material is coherent, but it is less obviously central for an economics thesis committee than the industrial organization and bargaining pieces.

### Important architecture note

- **This chapter is not isolated in the thesis copy.** `thesis_v2/ch06_games` is a symlink to `../ch06_games`, not an independent thesis-specific copy. As with the structural-estimation chapter, this means thesis-specific edits here would also affect the original survey source unless the symlink is replaced.

### Confirmed clean

- I did not find broken chapter cross-references in `ch06_games/tex/rl_in_games.tex`.
- The chapter’s economic anchors are strong. `ch06_games/tex/rl_in_games.tex:82` gives Cournot and Bertrand benchmarks, and `ch06_games/tex/rl_in_games.tex:150` gives the durable-goods monopoly and Coase conjecture.
- The final substantive claim of the chapter is easy to see: RL-style learning dynamics can recover or approximate equilibrium behavior in game settings that economists care about.

### Scope drift after cuts

- **The poker block still reads like the broader survey.** The CFR section at `ch06_games/tex/rl_in_games.tex:104`, the neural extensions at `ch06_games/tex/rl_in_games.tex:122`, and the poker results paragraph at `ch06_games/tex/rl_in_games.tex:146` are technically relevant, but they shift the chapter away from the economist-facing applications that otherwise justify its place in the trimmed thesis.
- **The neural poker material is not anchored by the chapter’s own simulations.** The retained simulations are industrial organization and bargaining examples, while the poker discussion serves more as a literature tour through CFR, Deep CFR, NFSP, Libratus, and Pluribus. In the reduced thesis, that asymmetry makes the poker block feel more like inherited survey coverage than a central chapter component.
- **The discussion subsection is a residual survey ending.** `ch06_games/tex/rl_in_games.tex:187` closes with a general comparative remark about stochastic-game Q-learning and CFR. It is brief, but it still has the flavor of a chapter-summary coda rather than the more stripped-down thesis style used elsewhere.

### Framing and proportion issues

- **The chapter’s best material is the economics material, not the poker material.** If the chapter is being read by economists, the strongest path runs from stochastic games to Cournot-Bertrand to the Coase conjecture. The poker-heavy middle at `ch06_games/tex/rl_in_games.tex:120` through `ch06_games/tex/rl_in_games.tex:146` is interesting, but it likely carries less value for the intended thesis audience.
- **The opening frame is broad and method-oriented.** `ch06_games/tex/rl_in_games.tex:3` and `ch06_games/tex/rl_in_games.tex:5` begin with a general multi-agent learning taxonomy. That is not wrong, but it means the chapter opens in a more RL-survey voice before arriving at the economist-facing benchmarks.

### Precision and tone issues

- **A few claims are stronger than needed.** `ch06_games/tex/rl_in_games.tex:120` says CFR+ enabled heads-up limit hold’em to be `essentially solved`, and `ch06_games/tex/rl_in_games.tex:146` says the methods achieved `superhuman performance in poker`. Both are standard descriptions in the literature, but they also add to the sense that poker occupies prestige space in the chapter.
- **The final sentence overstates algorithmic agnosticism slightly.** `ch06_games/tex/rl_in_games.tex:189` says the simulations confirm convergence to known equilibria `without encoding domain structure into the algorithms`. That is broadly true at a high level, but it compresses a lot of modeling structure and equilibrium setup into a single sentence.

### Bottom line on this chapter

This is a good thesis chapter. It probably needs less surgery than `Intro`, `Algorithms`, or the shared structural-estimation file. The main question is one of emphasis. If the trimmed thesis wants to keep only what economists are most likely to care about, then the Cournot-Bertrand and Coase material looks essential, while the poker and neural-extension block looks more optional.

### Priority order for a future pass

1. If thesis/source separation is a hard requirement, replace the symlinked games chapter with a real thesis-specific copy before making targeted edits.
2. Reassess how much of the poker and neural-extension block centered at `ch06_games/tex/rl_in_games.tex:122` through `ch06_games/tex/rl_in_games.tex:146` still belongs in the trimmed thesis.
3. Consider whether the chapter should reach the Cournot-Bertrand benchmark earlier, since `ch06_games/tex/rl_in_games.tex:82` is one of the most economist-facing parts.
4. Tighten or demote the residual discussion subsection at `ch06_games/tex/rl_in_games.tex:187` if the thesis is trying to avoid chapter-level recap material.
5. Soften the strongest poker-centered prestige framing at `ch06_games/tex/rl_in_games.tex:120`, `ch06_games/tex/rl_in_games.tex:146`, and `ch06_games/tex/rl_in_games.tex:189`.

## Offline RL and human feedback chapter review

Scope of review:
- Reviewed `thesis_v2/ch08_offline_rl/tex/offline_rl.tex`, which is the source pulled into the thesis build via `thesis_v2/docs/main.tex:163`
- Verified chapter `\ref{...}` targets against labels present in the thesis copy
- Checked the thesis-copy folder structure for this chapter
- Did not edit thesis source

### Overall judgment

This is one of the strongest and most thesis-aligned chapters in the reduced document. The opening economic motivation is clear, the offline RL theory is focused, and the job-search preference-learning simulation gives the RLHF material a direct economic anchor. The remaining issues are mostly about emphasis, architecture, and tone rather than broken logic.

### Important architecture note

- **The text is thesis-specific, but assets are still partly shared.** `thesis_v2/ch08_offline_rl/tex/offline_rl.tex` is a real local file, not a symlink. However, `thesis_v2/ch08_offline_rl/papers` and `thesis_v2/ch08_offline_rl/sims` are symlinked to the original chapter assets, and several RLHF figures and tables are still sourced from `../ch09_rlhf/sims/...` at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:194`, `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:211`, `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:220`, `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:231`, and `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:238`. So the chapter is more isolated than the shared games and structural-estimation chapters, but it still depends on legacy asset locations.

### Confirmed clean

- I did not find broken chapter cross-references in `thesis_v2/ch08_offline_rl/tex/offline_rl.tex`.
- The opening at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:1` is one of the clearest economics-first openings in the thesis.
- The chapter’s internal progression is strong: offline-RL problem setup, pessimism principle, algorithm sketches, dynamic-pricing simulation, then transition to human feedback and preference learning.
- The job-search simulation provides a good economist-facing landing point for the RLHF material.

### Scope and emphasis issues

- **The offline RL half is tightly thesis-aligned; the RLHF half is more expansive.** Up through `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:139`, the chapter is disciplined and focused. From `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:144` onward, the chapter expands into a broader RLHF/LLM-alignment tour, including the full pipeline, DPO, variational interpretation, and broader alignment discussion.
- **The RLHF pipeline section may be more detailed than the reduced thesis needs.** The block from `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:162` through `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:196` is technically interesting, but it gives substantial space to the canonical LLM-alignment framing before the chapter returns to the economist-facing job-search example.
- **The chapter still depends on an LLM-centered narrative despite the economic simulation.** `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:151`, `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:180`, and `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:203` repeatedly foreground language-model alignment. That is relevant, but it may still be more ML-facing than the trimmed thesis ideally wants.

### Framing and organization issues

- **The chapter effectively contains two related but distinct arguments.** One is about offline RL under distributional shift. The other is about reward learning from preferences and direct preference optimization. They belong together, but the seam is visible at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:144`, where the chapter pivots from one problem class to another.
- **The economic payoff is strongest when the chapter stays near known transitions and policy design.** That is clearest in the offline dynamic-pricing simulation and in the final job-search preference-learning simulation. The more abstract LLM-alignment framing in the middle is the least naturally connected to the rest of the thesis.

### Style and tone issues

- **A few claims are stronger than the trimmed thesis voice elsewhere.** `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:5` says naive offline application fails `catastrophically`, which is vivid but slightly more forceful than the rest of the reduced thesis usually is.
- **The results prose occasionally becomes more assertive than report-like.** `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:225` begins `Three findings stand out`, and the first finding says the structural model `dominates`. The claim may be correct, but the phrasing is more declarative than the understated style used elsewhere.
- **The recent-developments section is somewhat general.** `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:203` discusses open issues in RLHF in a broad way. It is sensible, but it also nudges the chapter back toward survey mode.

### Bottom line on this chapter

This chapter belongs in the thesis and is one of the better-fitting retained pieces. The main remaining question is not whether to keep it. It is how much of the LLM-centered RLHF exposition the thesis wants relative to the clearly economist-facing offline-pricing and job-search components.

### Priority order for a future pass

1. If thesis/source separation is meant to be strict, decide whether the legacy asset dependencies on `ch09_rlhf/sims` should be relocated into the thesis copy.
2. Reassess how much of the RLHF pipeline block at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:162` through `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:196` the reduced thesis really needs.
3. Keep the economic simulations central, especially the dynamic-pricing and job-search studies, since they are the clearest bridges to the thesis audience.
4. Soften the strongest evaluative wording, especially at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:5`, `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:225`, and `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:245`.
5. Consider tightening the general RLHF open-issues paragraph at `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:203` if the thesis wants to avoid survey-style outward expansion.

## Final synthesis: bridge criterion

The clearest guiding rule for the trimmed thesis is the following:

- Do not keep economics material that has no real RL counterpart.
- Do not keep RL material that has no real economics counterpart.
- Keep and foreground the places where the two fields genuinely illuminate each other.

Under that rule, the thesis should become a document about translation and bridge work, not a general survey of either field.

### Updated judgment

This bridge criterion changes the trimming logic in an important way.

- It strengthens the case for keeping `AlphaZero` and the Bertsekas-style lookahead interpretation, because that material is not being kept as benchmark prestige. It is being kept because it provides a deep dynamic-programming bridge between RL and economic computation.
- It also strengthens the case for keeping `Control as Probabilistic Inference`, provided it is framed tightly around the soft Bellman equations, entropy regularization, and the connection to discrete choice.
- It weakens the case for keeping material that is strong only on one side of the bridge, even if it is impressive in its own literature.

### Protect

These look like material to protect because they are genuine bridge sections.

- `thesis_v2/ch00_introduction/tex/intro.tex:14`, but only in a narrowed form focused on language and conceptual translation.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:203`, because `Control as Probabilistic Inference` is one of the cleanest bridges from RL to econometric logit-style structure.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:272` together with `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:217`, because the `AlphaGo/AlphaZero` material matters here as a bridge from approximate planning and policy improvement to Bertsekas-style theory.
- `ch05_econ_models/tex/rl_in_se.tex:24` and `ch05_econ_models/tex/rl_in_se.tex:202`, because DDC estimation is exactly where RL becomes a computational tool for economics rather than a separate field.
- `ch06_games/tex/rl_in_games.tex:79` and `ch06_games/tex/rl_in_games.tex:148`, because Cournot-Bertrand and Coase are economist-facing equilibrium applications of RL/game-learning ideas.
- `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:119` and `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:205`, because offline pricing and job-search preference learning are among the most persuasive economist-facing bridge applications in the thesis.

### Compress

These sections are worth keeping, but only in the form needed to support the bridge argument.

- `thesis_v2/ch00_introduction/tex/intro.tex:14` should be reduced to terminology and concept mapping. The broader sociology of fields, the more expansive examples, and the longer lifecycle framing are lower value under the bridge criterion.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:203` should keep the optimality variable, soft Bellman equations, and discrete-choice connection, but can lose the more expansive “single framework explains everything” sweep.
- `thesis_v2/ch02_rl_algorithms/tex/rl_algorithms.tex:272` should keep the network-plus-search structure that matters for the theory chapter, but can lose architecture and training detail that is not part of the bridge argument.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:173` should either be cut or reduced sharply, because the LQC example is less central once the optimal-control chapter is gone.
- `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:162` should keep DPO and the minimum RLHF machinery needed for the job-search application, but can lose more general LLM-pipeline detail.

### Cut

These sections are now the best candidates for removal because they are mostly one-sided rather than true bridges.

- `thesis_v2/ch00_introduction/tex/intro.tex:9`, the omitted-territory survey paragraph.
- `thesis_v2/ch03_theory/tex/planning_learning_v3.tex:491`, `Fundamental Tradeoffs`, which re-expands into a general RL survey frame.
- `ch05_econ_models/tex/rl_in_se.tex:192`, `Macroeconomic Models`, which reads mainly as economics spillover rather than bridge exposition.
- `ch05_econ_models/tex/rl_in_se.tex:197`, `Optimal Policy Design`, for the same reason.
- `ch06_games/tex/rl_in_games.tex:122`, or at least most of it. Under the bridge criterion, `Neural Extensions` and the poker prestige material are weaker than they first appear, because they are much more RL-facing than economics-facing.
- `ch06_games/tex/rl_in_games.tex:187`, `Discussion`.
- `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:144`, `From Offline RL to Human Feedback`.
- `thesis_v2/ch08_offline_rl/tex/offline_rl.tex:199`, `Recent Developments`.

### Revised strategic implication

The trimmed thesis should not try to be a smaller version of the original survey. It should become a narrower argument:

- RL matters for economists when it extends dynamic programming to high-dimensional state spaces, strategic environments, or fixed-data settings.
- Economics matters for RL when it contributes structure, identification discipline, interpretable reward models, and equilibrium reasoning.
- The best chapters are the ones where those two claims meet directly.

Under that logic, the thesis should sacrifice breadth in favor of high-value bridge sections. The right losses are not the conceptual pillars the author identified as important. The right losses are the parts of the document that display only one side of the bridge.

## Re-inspection: first chapter after cleanup

Scope of review:
- Re-read `thesis_v2/ch00_introduction/tex/intro.tex`
- Checked the current introduction for economist-facing jargon, framing drift, and internal consistency
- Did not edit thesis source

### Overall judgment

The first chapter is cleaner than before, and the bridge logic is now more visible. The strongest material is the language-translation function in `thesis_v2/ch00_introduction/tex/intro.tex:18` and the structural-equivalence bridge in `thesis_v2/ch00_introduction/tex/intro.tex:96`. The main remaining problem is not broken structure. It is that the introduction still explains RL in a dense, tutorial-like register that many economists will find more technical than they need at the chapter-opening stage.

### What now works well

- **The training-versus-deployment contradiction appears resolved.** The current phrasing at `thesis_v2/ch00_introduction/tex/intro.tex:37` is clearer than the earlier draft.
- **The bridge material is now identifiable.** `thesis_v2/ch00_introduction/tex/intro.tex:18` through `thesis_v2/ch00_introduction/tex/intro.tex:24` gives a useful economics-versus-control framing, and `thesis_v2/ch00_introduction/tex/intro.tex:96` through `thesis_v2/ch00_introduction/tex/intro.tex:110` gives a concrete mathematical bridge economists can recognize.
- **The notation table is useful.** `thesis_v2/ch00_introduction/tex/intro.tex:117` gives a practical translation device for readers who know dynamic choice or econometrics better than RL.

### Main remaining issues

- **The opening is still RL-first rather than economist-first.** `thesis_v2/ch00_introduction/tex/intro.tex:1` through `thesis_v2/ch00_introduction/tex/intro.tex:7` begins with Bellman equations, value iteration, Q-learning, policy gradients, Bellman error, convergence rates, and hyperparameters before stating the economic value of the chapter in plain terms.
- **The omitted-territory paragraph still reads like survey carryover.** `thesis_v2/ch00_introduction/tex/intro.tex:9` still spends visible space on what the thesis does not cover.
- **The causal claim remains too strong.** `thesis_v2/ch00_introduction/tex/intro.tex:29` still says off-policy evaluation is `precisely counterfactual policy evaluation`. For an economics audience, that risks sounding like a claim about identification rather than a looser analogy.
- **The lifecycle grid has a category mismatch.** `thesis_v2/ch00_introduction/tex/intro.tex:59` labels the row `Live market ("in-field")`, but `thesis_v2/ch00_introduction/tex/intro.tex:61` places `AlphaGo vs. Lee Sedol` in that row. That is a live environment, but not a market, so the label and example do not fit each other.
- **The pipeline sentence overgeneralizes.** `thesis_v2/ch00_introduction/tex/intro.tex:66` says a typical applied RL pipeline moves from historical logs to simulator refinement to frozen deployment. That is plausible for some modern systems, but it is not a typical path for RL in general and not the natural frame for several retained economist-facing applications.
- **The `model-free` paragraph contains quote-escape artifacts.** `thesis_v2/ch00_introduction/tex/intro.tex:33` has `"` forms in `model-free`, `in-field`, and `model`, which reads like a typesetting glitch rather than intended prose.
- **The simulator framing is still too broad.** `thesis_v2/ch00_introduction/tex/intro.tex:85` says RL typically assumes access to a high-fidelity simulator or digital twin. That is too sweeping once the thesis also keeps offline RL and fixed-data applications.

### Jargon an economist may not understand immediately

- **Undefined acronyms.** `thesis_v2/ch00_introduction/tex/intro.tex:20` uses `PID`, `LQR`, and `MPC` without expansion. `thesis_v2/ch00_introduction/tex/intro.tex:35` uses `MDP` without expansion. `thesis_v2/ch00_introduction/tex/intro.tex:81` uses `UCB` without expansion.
- **RL-native optimization language appears too early.** `thesis_v2/ch00_introduction/tex/intro.tex:3` and `thesis_v2/ch00_introduction/tex/intro.tex:5` use `average Bellman error`, `sampled Bellman error`, `geometric rate`, `sublinear convergence`, and `sufficient exploration` before the chapter has translated them into economist-facing language.
- **Control-theory vocabulary may lose readers.** `thesis_v2/ch00_introduction/tex/intro.tex:20` uses `plant physics` and `robustness constraints`, which are natural in control but not in economics.
- **The adaptive-learning paragraph is mathematically dense for an introduction.** `thesis_v2/ch00_introduction/tex/intro.tex:79` packs in `Robbins-Monro`, `E-stability`, `ODE method`, `temporal-difference`, and `actor-critic`. This is valuable material, but it reads more like a specialized bridge note than introductory framing.
- **The exploration paragraph is jargon-heavy.** `thesis_v2/ch00_introduction/tex/intro.tex:81` introduces `$\varepsilon$-greedy`, `UCB`, and `Thompson sampling` in a chapter opener where only the economic analogy may matter.
- **The bootstrapping paragraph is also dense.** `thesis_v2/ch00_introduction/tex/intro.tex:83` introduces `TD algorithm`, `function approximation`, `sieve estimation`, `series estimation`, and `probability calibration` in quick succession.
- **The discount-factor paragraph is longer and more technical than it needs to be here.** `thesis_v2/ch00_introduction/tex/intro.tex:91` moves through Koopmans axioms, quasi-hyperbolic discounting, contraction arguments, and `Pitis2019`. The core bridge is good, but the paragraph is still heavy for an introduction.
- **The convex-analysis footnote is likely too specialized for this location.** `thesis_v2/ch00_introduction/tex/intro.tex:110` introduces `Fenchel conjugates` and `negative Shannon entropy`. For most economists, that is more technical than the introduction needs.
- **The notation table ends with a dense technical footnote.** `thesis_v2/ch00_introduction/tex/intro.tex:137` distinguishes `TD error`, `Bellman residual`, and `BRM`. That distinction is correct, but it is probably too fine-grained for a first-chapter translation table.

### Best bridge material to preserve

- `thesis_v2/ch00_introduction/tex/intro.tex:18` through `thesis_v2/ch00_introduction/tex/intro.tex:24`, because this is the cleanest statement of the language problem between economics and RL.
- `thesis_v2/ch00_introduction/tex/intro.tex:35`, because the point that `model-free` is not the same as `reduced-form` is genuinely useful for economists.
- `thesis_v2/ch00_introduction/tex/intro.tex:85`, because the contrast between policy evaluation in economics and in RL is conceptually important, even if the current wording is too expansive.
- `thesis_v2/ch00_introduction/tex/intro.tex:96` through `thesis_v2/ch00_introduction/tex/intro.tex:110`, because the softmax/logit and entropy/inclusive-value equivalences are among the most persuasive bridges in the whole introduction.
- `thesis_v2/ch00_introduction/tex/intro.tex:117`, because the notation table is one of the few compression devices that actually helps the economist reader.

### Lowest-value density if more tightening is needed

- `thesis_v2/ch00_introduction/tex/intro.tex:9`, the omitted-territory paragraph.
- `thesis_v2/ch00_introduction/tex/intro.tex:41` through `thesis_v2/ch00_introduction/tex/intro.tex:66`, especially the lifecycle grid and the `typical pipeline` sentence, which are less central to the economics-RL bridge than the terminology and structural-equivalence material.
- `thesis_v2/ch00_introduction/tex/intro.tex:79`, which could likely be reduced to a shorter adaptive-learning bridge sentence or paragraph.
- `thesis_v2/ch00_introduction/tex/intro.tex:81` and `thesis_v2/ch00_introduction/tex/intro.tex:83`, which read more like glossary expansions than essential first-chapter argument.
- `thesis_v2/ch00_introduction/tex/intro.tex:91`, which can likely be shortened substantially without losing the economist-facing point.
- The footnotes at `thesis_v2/ch00_introduction/tex/intro.tex:110` and `thesis_v2/ch00_introduction/tex/intro.tex:137`, which are accurate but probably too technical for this chapter-opening role.

### Bottom line on chapter one

The introduction is now conceptually closer to the right thesis. Its best parts are the translation paragraphs and the formal equivalences that show economists they have seen parts of this machinery before. Its weakest parts are not wrong. They are too dense, too acronym-heavy, and occasionally too RL-native for a first chapter aimed at economists. If the goal is economist readability, the next gains come less from further restructuring and more from pruning jargon-heavy explanatory blocks that feel like a mini textbook inside the introduction.
