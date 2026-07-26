# AI-tell audit of recently updated chapters

Date: 2026-07-25

## Scope and authority

This is a read-only audit of the chapters created or substantially revised in the most
recent work cycle. The authority surface follows the live worktree that contains each
change, rather than the stale local `main` snapshot.

| Chapter | Authority surface | Why included |
|---|---|---|
| 3, Theory | `/Users/pranjal/Code/rl` | Substantially revised on 2026-07-25; current file also has live uncommitted changes |
| 10b, OPE and Dynamic Treatment Effects | `/Users/pranjal/Code/rl-ope-split` | Expanded and reorganized on 2026-07-24 |
| 10c, Causal Bandits and Adaptive Experimentation | `/Users/pranjal/Code/rl-ope-split` | New chapter created and rebuilt on 2026-07-24 |
| 11, Quantile, Robust, and Constrained RL | `/Users/pranjal/Code/rl` | Contains a current uncommitted proof-explanation edit |
| 13, Field Deployments | `/Users/pranjal/Code/rl-ch13-repair` | New 2026-07 field chapter; latest OPE-repair and Horizon-anatomy surface |

Chapter 2 was touched on 2026-07-25 only to add pointers to the revised appendix and theory,
so it is excluded. Older chapters and the appendix are also excluded.

## Method

The audit uses the high-confidence patterns relevant to academic prose from the
`humanizer` detection catalog together with the plain-construction rules P1-P6 in
`docs/bloat.md`. It excludes comments, equations, necessary theorem setup, captions that
only identify content, and simulation-generated table fragments. It does not assign
AI-probability scores.

`Definite` means the construction matches a repo rule and adds no necessary mathematical
or historical function in its current form. `Borderline` means it resembles an AI tell but
may be justified by exposition or technical precision. A flag is an editorial candidate,
not evidence about authorship. No LaTeX source was changed during this audit.

The older `docs/humanizer_edits_report.md` covered a different snapshot. Rejected rewrites
listed in `docs/destrain_sweep.md` remain rejected and are not recommended again.

## Summary

| Chapter | Definite | Borderline |
|---|---:|---:|
| 3 | 12 | 8 |
| 10b | 6 | 4 |
| 10c | 2 | 2 |
| 11 | 1 | 0 |
| 13 | 28 | 13 |
| **Total** | **49** | **27** |

The findings are concentrated rather than paper-wide. Chapter 13 accounts for 28 of the
49 definite candidates, mostly in its new opening, evidence synthesis, and closing
deployment lessons. Chapter 3 contributes 12, primarily from document narration,
personified mathematical objects, and explanatory metaphors in the recent rewrite.
Chapter 10b has six. The new Chapter 10c has two, and the live Chapter 11 edit has one.

## Chapter 3: The Theory of Reinforcement Learning

Authority: `/Users/pranjal/Code/rl/ch03_theory/tex/planning_learning_v3.tex`, snapshot
SHA-256 `98ef8266c78ab8e23f5cf4a96749fcb24466cbb6aecf6d23d8326ed2bf6c8558`.

This is a bounded recent-diff audit against commit `126c180` and the current worktree.
`curse_of_dimensionality.tex` has no July 25 or uncommitted prose changes, so it contributes
no rows. The two Chapter 3 lines changed directly by `126c180` also produced no findings.
No prose em dashes were found. The two rejected Chapter 3 rows in
`docs/destrain_sweep.md` are not proposed again.

### Definite candidates

1. `ch03_theory/tex/planning_learning_v3.tex:9` — Formulaic meta opener. `This chapter
   reuses the Markov decision process notation fixed in
   Section~\ref{section:rl_algorithms} ... and adds notation needed for the convergence
   analysis that follows.` describes the chapter's action rather than stating the notation.
   Minimal action: delete the process framing and begin with the symbols. Prior-report
   overlap: none.

2. `ch03_theory/tex/planning_learning_v3.tex:82` — Frame noun and personification.
   `$\tilde{V}$'s role ends there. It has picked out one action per state` assigns a role and
   agency to a value estimate. Minimal action: state the policy's dependence on
   $\tilde V$ directly. Prior-report overlap: none.

3. `ch03_theory/tex/planning_learning_v3.tex:86` — Idiomatic description. `plus a borrowed
   guess $V(s')$ for everything after` replaces the continuation-value concept with
   `borrowed guess` and `everything after`. Minimal action: name the continuation-value
   estimate directly. Prior-report overlap: none.

4. `ch03_theory/tex/planning_learning_v3.tex:90` — Personified travel metaphor. `The
   geometry accounts for the speed but not for the destination.` uses speed and destination
   to transition between rate and optimality. Minimal action: state that convergence rate
   does not establish optimality. Prior-report overlap: none.

5. `ch03_theory/tex/planning_learning_v3.tex:125` — Personified argument and strained
   idiom. `The two arguments rest on different foundations and cost different assumptions.`
   makes arguments incur assumptions. Minimal action: state which assumptions each result
   requires. Prior-report overlap: none.

6. `ch03_theory/tex/planning_learning_v3.tex:163` — Formulaic meta transition. `Everything
   to this point has been planning.`, `The rest of the chapter drops the assumption that
   $P$ is known.`, and `the next subsections rebuild the same fixed-point calculations on
   that stream` narrate the chapter's movement. Minimal action: introduce sampled-transition
   learning directly. Prior-report overlap: none.

7. `ch03_theory/tex/planning_learning_v3.tex:210` — Frame nouns and personified proof. `The
   logic of Q-learning is best understood`, `The averaging picture is heuristic`, and `The
   proof of Theorem~\ref{thm:qlearning_convergence} below makes it exact` wrap the sampling
   and martingale claims in scaffolding. Minimal action: state those claims directly.
   Prior-report overlap: none.

8. `ch03_theory/tex/planning_learning_v3.tex:392` — Imperative math-speak. `In chess this is
   the familiar tree search, expand $\ell$ plies of moves, apply the evaluation function at
   the leaves, and back the values up.` switches into instructions. Minimal action: describe
   the tree-search operations declaratively. Prior-report overlap: none.

9. `ch03_theory/tex/planning_learning_v3.tex:437` — Formulaic meta opener and frame noun.
   `The theorem that follows plays the role of Lemma~\ref{lem:bellman_contraction} under
   function approximation.` announces the theorem and assigns a textual role. Minimal
   action: state the operator and norm correspondence directly. Prior-report overlap: none.

10. `ch03_theory/tex/planning_learning_v3.tex:580` — Frame noun, significance framing, and
    personified theorem. `The value of the theorem lies in what is absent from
    ~\eqref{eq:policy_gradient}.` and `The theorem transforms a sensitivity analysis problem
    ... into a simpler expectation problem` make the theorem act. Minimal action: state the
    absent transition derivative and resulting estimability directly. Prior-report overlap:
    none.

11. `ch03_theory/tex/planning_learning_v3.tex:632` — Formulaic signposting. `TRPO
    \citep{Schulman2015} and PPO \citep{Schulman2017} are its two practical descendants,
    developed in turn below.` ends with pure navigation. Minimal action: delete `developed
    in turn below`. Prior-report overlap: none.

12. `ch03_theory/tex/planning_learning_v3.tex:719` — Terse apposition and frame noun. `The
    division separates estimation from optimization, the critic solves a regression
    problem ... while the actor solves an optimization problem` makes `division` the subject
    and joins the gloss by comma. Minimal action: split into direct critic and actor
    statements. Prior-report overlap: none.

### Borderline candidates

1. `ch03_theory/tex/planning_learning_v3.tex:78` — Negative parallelism and colon reveal.
   `roughly tenfold rather than by ten percent: shrinking the error a thousandfold` adds
   rhetorical staging to a numerical example. Minimal action if selected: use two
   declarative sentences.

2. `ch03_theory/tex/planning_learning_v3.tex:88` — Formulaic reason opener. `The reason this
   process converges so quickly is geometric` mildly promotes the setup before the convexity
   argument. Minimal action if selected: begin with the geometric fact.

3. `ch03_theory/tex/planning_learning_v3.tex:143` — Repeated colon cadence. `Panel~(a)
   shows VI:` and `Panel~(b) shows PI:` use the same reveal structure. Minimal action if
   selected: use declarative panel descriptions.

4. `ch03_theory/tex/planning_learning_v3.tex:300` — Personification. `the class loses the
   ability to represent its own Bellman images` assigns ability and possession to a
   function class. Minimal action if selected: state the representability condition.

5. `ch03_theory/tex/planning_learning_v3.tex:305` — Personified abstraction. `The bracket
   collects three error sources` makes notation act and partially revives a previously
   cleaned three-term construction. Minimal action if selected: name the three terms
   directly. Prior overlap: related to, but not a reversion of, the older line-168 edit.

6. `ch03_theory/tex/planning_learning_v3.tex:474` — Personification and comma apposition.
   `The projection becomes \emph{oblique} in the $d^\pi$-norm rather than orthogonal, it
   reaches the subspace along directions that are not perpendicular in that norm.` uses
   `reaches` and strains the definition. Minimal action if selected: split and define the
   projection direction.

7. `ch03_theory/tex/planning_learning_v3.tex:602` — Promotional metaphor. `The alignment
   with policy iteration also has a statistical payoff.` lightly inflates the transition to
   the dimension-free result. Minimal action if selected: state that result directly.

8. `ch03_theory/tex/planning_learning_v3.tex:734` — Frame noun and theatrical metaphor.
   `The fast-slow separation is the same device that stabilizes the two-timescale methods
   ... with policy evaluation now cast as the fast process.` obscures the mathematical
   correspondence. Minimal action if selected: state the correspondence directly.

## Chapter 10b: Off-Policy Evaluation and Dynamic Treatment Effects

Authority: `/Users/pranjal/Code/rl-ope-split/ch10b_rl_for_ci/tex/rl_for_ci.tex`, snapshot
SHA-256 `96de5f5ba5bd1d8dbcf39eecf76e4153b8c71b96eec972447291ebdc4ef29595`.

The file was read in full. No prose em dashes or colon-drumrolls were found. None of the
rejected rows in `docs/destrain_sweep.md` is proposed again.

### Definite candidates

1. `ch10b_rl_for_ci/tex/rl_for_ci.tex:210` — Negative parallelism. `The empirical
   distinction is therefore not between having and lacking an estimated rule. It is between
   histories where the data distinguish its prescribed action and histories where they do
   not.` stages a precise empirical contrast as a rhetorical reversal. Minimal action:
   state the interval-based distinction directly. Prior-report overlap: none.

2. `ch10b_rl_for_ci/tex/rl_for_ci.tex:252` — Personified argument. `The regression reads
   the greater needs of frequently visited families as a harmful treatment association`
   turns omitted-confounder bias into an actor. Minimal action: name the confounding
   mechanism directly. Prior-report overlap: none.

3. `ch10b_rl_for_ci/tex/rl_for_ci.tex:368` — Personified method and loose metaphor. `the
   \emph{doubly robust} (DR) estimator uses each to repair the other` substitutes `repair`
   for the estimator's complementary regression and weighting corrections. Minimal action:
   state those two roles literally. Prior-report overlap: none.

4. `ch10b_rl_for_ci/tex/rl_for_ci.tex:397` — Personified argument. `the curse defeating
   its own measurement` makes a statistical phenomenon act on its measurement. The
   preceding numerical sentence already gives the rare-trajectory mechanism. Minimal
   action: name that sampling failure directly. Prior-report overlap: none.

5. `ch10b_rl_for_ci/tex/rl_for_ci.tex:703` — Personification and significance inflation.
   `AIPW also survives the cell in which both nuisances are wrong. The learned thresholds
   show that this is genuine recovery rather than a flat value surface.` interprets the
   result through `survives` and `genuine recovery` instead of reporting regret and
   thresholds. Minimal action: report those measurements directly. Prior-report overlap:
   none.

6. `ch10b_rl_for_ci/tex/rl_for_ci.tex:719` — Formulaic challenges opener and meta prose.
   `The methods above address several parts of the sequential causal problem, but important
   gaps remain. Table~\ref{tab:rlci_open_issues} separates problems that are often
   conflated.` uses the stock progress-versus-gaps frame and narrates the table. Minimal
   action: begin with the first substantive distinction or point directly to the table.
   Prior-report overlap: none.

### Borderline candidates

1. `ch10b_rl_for_ci/tex/rl_for_ci.tex:19` — Meta roadmap. `Each section pairs the formal
   result with a computational experiment and separates point estimation from uncertainty
   quantification. The final section describes open problems.` narrates organization, but
   may help orient readers in a heavily revised chapter. Minimal action if selected: retain
   only the navigation that materially helps.

2. `ch10b_rl_for_ci/tex/rl_for_ci.tex:250` — Redundant explanation. `No recovery ratio can
   establish this. A ratio measures how fast an estimator approaches the optimum, which is
   a statement about estimation and is consistent with the identity being false.` partly
   restates the first sentence, though the false-identity point is useful. Minimal action if
   selected: carry both claims in one sentence. This is not the rejected old MABUC row that
   once occupied line 250.

3. `ch10b_rl_for_ci/tex/rl_for_ci.tex:414` — Personified cross-reference.
   `Section~\ref{subsec:sim_ope} shows this ranking under correct specification and its
   reversal when the model is wrong.` is common scholarly shorthand, but makes the section
   the actor. Minimal action if selected: attribute the result to the simulation evidence.

4. `ch10b_rl_for_ci/tex/rl_for_ci.tex:703` — Strained negative framing. `The both-wrong
   cell is not the failure that double robustness predicts for value estimation.` is
   conceptually important but treats `failure` as an object predicted by double robustness.
   Minimal action if selected: state the estimand-specific limitation directly.

## Chapter 10c: Causal Bandits and Adaptive Experimentation

Authority:
`/Users/pranjal/Code/rl-ope-split/ch10c_adaptive_experiments/tex/adaptive_experiments.tex`,
snapshot SHA-256 `133c46096be430296922981c268232e968a143c3e387bb478f600060dc2a9683`.

The new file was read in full. No finding overlaps the older humanizer or de-strain reports.

### Definite candidates

1. `ch10c_adaptive_experiments/tex/adaptive_experiments.tex:164` — Redundant construction.
   `The factorial separates observational seeding from RDC weighting and reports the
   additional ETT warm start separately.` repeats `separates` and `separately`. Minimal
   action: remove the duplicated separation wording while preserving the factorial design.
   Prior-report overlap: none.

2. `ch10c_adaptive_experiments/tex/adaptive_experiments.tex:210` — Formulaic conclusion
   and treadmill effect. The four-sentence Discussion beginning `Causal structure can reduce
   exploration cost when one action reveals outcomes relevant to others.` restates the
   chapter opener and facts already established in the two body subsections. Minimal action:
   delete the subsection or retain only new synthesis. Prior-report overlap: none.

### Borderline candidates

1. `ch10c_adaptive_experiments/tex/adaptive_experiments.tex:6` — Personified, formulaic
   opener. `Sequential experiments join two problems.` makes experiments join problems and
   sets up a balanced first-problem/second-problem template. Minimal action if selected:
   state the two requirements directly.

2. `ch10c_adaptive_experiments/tex/adaptive_experiments.tex:60` — Vague frame.
   `\citet{lattimore2016causal} obtain a different gain from causal structure.` delays the
   concrete parallel-bandit mechanism supplied immediately afterward. Minimal action if
   selected: delete the transition or name the mechanism directly.

## Chapter 11: Quantile, Robust, and Constrained Reinforcement Learning

Authority:
`/Users/pranjal/Code/rl/ch11_dist_robust_constrained/tex/dist_robust_constrained.tex`,
snapshot SHA-256 `254309491081d778a7e2a158f4e461b1ec65e631c2b3974aafd41f0536ed8fa3`.

This focused check covers the current uncommitted proof-explanation edit and its surrounding
context. The older, unchanged chapter prose was not re-audited.

### Definite candidate

1. `ch11_dist_robust_constrained/tex/dist_robust_constrained.tex:462` —
   Colon-drumroll and strained compound sentence. `An equivalent route is Blackwell's
   sufficient conditions (Theorem~\ref{thm:blackwell}): monotonicity holds because every
   $p \in \mathcal{P}(s,a)$ is nonnegative, and discounting holds with equality, $T(V+c) =
   TV + \gamma c$, because every $p \in \mathcal{P}(s,a)$ sums to one, so the constant shift
   passes unchanged through both the inner minimum and the outer maximum.` stages two
   conditions after a colon and stacks `because`, `and`, another `because`, and `so`.
   Minimal action: split the two checks into declarative sentences while preserving all
   mathematics. Prior-report overlap: none; this is newly added prose.

### Borderline candidates

None. The application-specific check of Blackwell's conditions is not a P6 re-derivation.

## Chapter 13: Reinforcement Learning in the Field

Authority:
`/Users/pranjal/Code/rl-ch13-repair/ch13_field_deployments/tex/field_deployments.tex`
on branch `ch13-ope-repair`. Snapshot SHA-256:
`01175832f41e448a184568177ccad977233154974553a2092bebde648cda0d54`.

The file was read in full, including its two live uncommitted edits. It contains no prose
em dashes. Chapter 13 is absent from the older humanizer and de-strain trackers, so none of
the findings below has prior overlap.

### Definite candidates

1. `ch13_field_deployments/tex/field_deployments.tex:1` — Formulaic meta opener and idiom.
   `This chapter looks at where reinforcement learning actually survives contact with a real
   production system.` narrates the chapter's task and uses `survives contact`. Minimal
   action: state the production distinction literally.

2. `ch13_field_deployments/tex/field_deployments.tex:1` — Treadmill effect. `This chapter
   investigates the best known cases of industrial scale deployment of reinforcement
   learning.` repeats the preceding sentence's scope. Minimal action: delete.

3. `ch13_field_deployments/tex/field_deployments.tex:3` — Formulaic recap. `There is, in
   short, a lot of infrastracture and prior experimentation that is necessarily for business
   deployment of reinforcement learning.` recaps the preceding enumeration without new
   information. Minimal action: delete or retain only distinct content.

4. `ch13_field_deployments/tex/field_deployments.tex:34` — Frame-noun scaffolding. `the
   deployments in the rest of this chapter each occupy part of it rather than all of it`
   places cases in abstract parts of a loop. Minimal action: name the pipeline stages each
   case documents.

5. `ch13_field_deployments/tex/field_deployments.tex:217` — Idiom. `an under-trained bidder
   cannot be let loose on live traffic` uses informal stock phrasing. Minimal action: state
   the deployment restriction literally.

6. `ch13_field_deployments/tex/field_deployments.tex:322` — Frame noun. `An earlier line of
   DiDi systems learned the value with a neural network rather than a table` frames the
   history rather than naming the systems. Minimal action: name the earlier systems
   directly.

7. `ch13_field_deployments/tex/field_deployments.tex:359` — Colon-drumroll. `The project is
   not a standard A/B test: the source states that showing different prices to different
   customers at the same time was not legally available` joins two complete claims through
   a reveal colon. Minimal action: split or use a causal connective.

8. `ch13_field_deployments/tex/field_deployments.tex:369` — Treadmill effect. `The
   pre-DeepStock inventory evidence is a benchmark result, not a deployment.`, `The paper is
   valuable precisely because it does not claim deployment.`, and `It shows that stylized
   inventory benchmarks are not enough to justify field adoption.` repeat the same
   benchmark-versus-deployment classification. Minimal action: state the classification
   once and retain only the distinct result.

9. `ch13_field_deployments/tex/field_deployments.tex:372` — Frame-noun scaffolding. `The
   move that makes it deployable is to regularize the learned policy with the classical
   base-stock structure of inventory theory` delays the mechanism. Minimal action: make
   base-stock regularization the subject.

10. `ch13_field_deployments/tex/field_deployments.tex:418` — Colon-drumroll. `I treat
    MuZero-RC as the canonical physical/design runtime-control case: it replaced a narrow
    codec control component` stages the factual control-surface claim as a reveal. Minimal
    action: separate the classification from the fact.

11. `ch13_field_deployments/tex/field_deployments.tex:421` — Colon-drumroll. `AlphaChip
    treats chip floorplanning as a sequential placement problem: starting from an empty
    layout` introduces the setup theatrically. Minimal action: split or subordinate the
    setup.

12. `ch13_field_deployments/tex/field_deployments.tex:468` — Colon-drumroll. `It is
    post-training: demonstrations train a supervised policy` turns the pipeline description
    into a reveal. Minimal action: split.

13. `ch13_field_deployments/tex/field_deployments.tex:470` — Colon-drumroll. `The deployment
    evidence should be read narrowly: RL shaped the policy before deployment` dramatizes the
    evidentiary qualification. Minimal action: state the limitation and evidence separately.

14. `ch13_field_deployments/tex/field_deployments.tex:475` — Treadmill effect. `not through
    a live production trading deployment. It is therefore a contrast case, not a confirmed
    field deployment.` repeats the same classification. Minimal action: keep it once.

15. `ch13_field_deployments/tex/field_deployments.tex:481` — Negative parallelism. `the
    question is not whether a policy is good but whether offline evaluation can tell`
    supplies rhetorical contrast. Minimal action: state the evaluation question directly.

16. `ch13_field_deployments/tex/field_deployments.tex:495` — Personified argument.
    `Table~\ref{tab:field_ope_reliability} reports the outcome, and the logging regime decides
    it.` and `Table~\ref{tab:field_ope_candidates} shows the mechanism per candidate.` make
    the regime decide and the table reveal a mechanism. Minimal action: put the measured
    estimator or coverage result in subject position.

17. `ch13_field_deployments/tex/field_deployments.tex:497` — Formulaic meta opener. `The
    practical implication for the field is specific.` announces the implication. Minimal
    action: delete and begin with the substantive claim.

18. `ch13_field_deployments/tex/field_deployments.tex:521` — Formulaic framing. `the public
    evidence comes in two kinds that are worth keeping apart` narrates an organizational
    judgment. Minimal action: state the two categories directly.

19. `ch13_field_deployments/tex/field_deployments.tex:523` — Frame noun and personification.
    `A gap runs between reinforcement learning as it is studied and reinforcement learning
    as it is deployed.` makes an abstract gap act. Minimal action: name the concrete
    differences.

20. `ch13_field_deployments/tex/field_deployments.tex:525` — Negative parallelism and
    aphoristic close. `None of it is the learning algorithm, and all of it is why a system
    reaches production.` turns the infrastructure claim into a none/all slogan. Minimal
    action: state the infrastructure requirement directly.

21. `ch13_field_deployments/tex/field_deployments.tex:527` — Formulaic grand synthesis.
    `Taken together, this is why the confirmed list is short, why so many ``real-world''
    papers stop at a simulator or a backtest, and why the field's honest claim is a narrow
    one.` repeats `why` and inflates a conclusion already established. Minimal action:
    retain only the narrow empirical conclusion.

22. `ch13_field_deployments/tex/field_deployments.tex:532` — Formulaic and redundant setup.
    `faces a prior question, and it is worth answering first. The question is whether the
    firm's own actions move the state.` announces and then repeats the question. Minimal
    action: begin with the decision criterion.

23. `ch13_field_deployments/tex/field_deployments.tex:538` — Idiom. `which is where the
    deployed systems of Section~\ref{sec:field_lessons} were won or lost` uses a cliché for
    engineering viability. Minimal action: name the determining infrastructure work.

24. `ch13_field_deployments/tex/field_deployments.tex:543` — Meta opener. `It helps to see
    the confirmed record in one place.` only announces the table. Minimal action: delete.

25. `ch13_field_deployments/tex/field_deployments.tex:567` — Negative parallelism and
    aphorism. `What is absent is as informative as what is present.` delays the concrete
    absence claim. Minimal action: delete and name the missing method class.

26. `ch13_field_deployments/tex/field_deployments.tex:571` — Personified argument and
    aphoristic gloss. `The magnitude itself is a tell.` and `the warning the simulation of
    Section~\ref{sec:field_ope_sim} makes precise` turn the number into a tell and the
    simulation into a warning. Minimal action: state the measured contrast and limitation.

27. `ch13_field_deployments/tex/field_deployments.tex:573` — Frame-noun scaffolding and
    personification. `Around that headline number sit measurements a return curve never
    shows, and any one of them can decide whether a policy is deployable.` stages
    measurements around a headline and makes them decide. Minimal action: name the
    additional deployment criteria directly.

28. `ch13_field_deployments/tex/field_deployments.tex:573` — Negative parallelism and
    aphoristic close. `None of these appear in a simulator's return, and together they are
    the difference between a method that wins a benchmark and one that runs a business.`
    ends on a benchmark-versus-business slogan. Minimal action: end on the specific omitted
    operational measurements.

### Borderline candidates

1. `ch13_field_deployments/tex/field_deployments.tex:3` — Vague significance inflation.
   `The gap between academic discussion and application in controlled simulators against
   large scale industrial application is quite large.` asserts magnitude without specifying
   the observable difference. Minimal action if selected: name the infrastructure contrast.

2. `ch13_field_deployments/tex/field_deployments.tex:34` — Significance inflation and meta
   framing. `is the clearest public account of what that infrastructure contains, so it is
   worth setting out as a template` editorializes the source choice. Minimal action if
   selected: state why Horizon supplies the common schema.

3. `ch13_field_deployments/tex/field_deployments.tex:34` — Significance gloss and formulaic
   list. `What makes it deployable is everything around it, namely how decisions are logged,
   how transitions are assembled, how the formulation is screened, how the policy is
   evaluated before anyone sees it, and how the loop is closed by retraining.` contains
   substantive infrastructure but uses an exhaustive `what makes it` frame. Minimal action
   if selected: put the infrastructure components in subject position.

4. `ch13_field_deployments/tex/field_deployments.tex:43` — Meta narration. `The objects can
   be named once and reused for the later cases.` narrates notation management. Minimal
   action if selected: begin with the common objects.

5. `ch13_field_deployments/tex/field_deployments.tex:53` — Aphoristic opener. `Formulation
   comes before any algorithm.` reads as a maxim but accurately introduces Horizon's
   screening stage. Minimal action if selected: attach the claim directly to Horizon.

6. `ch13_field_deployments/tex/field_deployments.tex:165` — Significance inflation. `The
   live evidence is unusually explicit for a production reinforcement-learning paper.`
   makes an undefined comparison. Minimal action if selected: state the explicit deployment
   facts.

7. `ch13_field_deployments/tex/field_deployments.tex:195` — Significance gloss. `which is
   what lets a learned bidder reach production` interprets one design choice as the enabling
   cause. Minimal action if selected: state the documented deployment consequence.

8. `ch13_field_deployments/tex/field_deployments.tex:270` — Symbolic gloss. `the case
   exemplifies reinforcement learning used to tune a trusted controller rather than replace
   it` tells the reader what the case represents, though synthesis is legitimate in a
   survey. Minimal action if selected: tie the conclusion to the deployed artifact.

9. `ch13_field_deployments/tex/field_deployments.tex:386` — Frame noun and significance
   inflation. `The pattern is a Pareto improvement` and `the clearest public case of a
   learned policy running an entire replenishment operation` frame factual results through
   editorial judgments. Minimal action if selected: state the result and evidence limit.

10. `ch13_field_deployments/tex/field_deployments.tex:438` — Meta narration. `it is the
    case this chapter counts for reinforcement learning in commercial cooling` narrates a
    classification decision. Minimal action if selected: state the evidence category.

11. `ch13_field_deployments/tex/field_deployments.tex:513` — Significance gloss. `The
    method improves a trusted component in place, and that is what makes it safe to ship.`
    compresses several safety mechanisms into one causal consequence. Minimal action if
    selected: retain the component and qualify the safety mechanism.

12. `ch13_field_deployments/tex/field_deployments.tex:534` — Abstract frame nouns. `the
    deployed record marks out where reinforcement learning has actually paid off, and the
    profile is consistent` announces a pattern instead of naming it. Minimal action if
    selected: begin with the recurring deployment conditions.

13. `ch13_field_deployments/tex/field_deployments.tex:536` — Meta framing and frame noun.
    `Read in reverse, the same profile is a list of warning signs.` tells the reader how to
    invert an abstract profile. Minimal action if selected: introduce the warning conditions
    directly.
