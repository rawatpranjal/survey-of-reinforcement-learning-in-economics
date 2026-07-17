# Reinforcement Learning in the Field: Master Redesign Plan

Date: 2026-07-16

Status: implemented 2026-07-16. This file consolidates the earlier chapter plan, the redesign decision, the source/read-note discipline, the user's preferences, and the implementation/audit outcome.

Related files:

- `docs/plans/2026-07-16-rl-in-field-chapter-plan.md`: earlier initial plan and sourcing kickoff.
- `ch13_field_deployments/source_manifest.md`: operational source/status checklist.
- `ch13_field_deployments/tex/field_deployments.tex`: implemented chapter text.
- `ch13_field_deployments/papers/`: downloaded originals and markdown conversions.
- `ch13_field_deployments/papers/read_notes/`: source-by-source read notes before prose.

## Implementation and Audit Outcome

Implemented on 2026-07-16.

- Rewrote `ch13_field_deployments/tex/field_deployments.tex` as the canonical field-deployments chapter.
- Retired the old `ch04_control_problems/tex/applications.tex` body and removed it from the master, journal, and thesis wrappers.
- Added source originals, markdown conversions, read notes, and bibliography entries for the current source corpus.
- Verified `scripts/check-sourced.sh ch13_field_deployments/papers` passes for the converted source markdowns.
- Verified `docs/main.tex`, `thesis/docs/main.tex`, `journals/csur_full_50pp/main.tex`, and `journals/fntml_monograph_100pp/main.tex` build to PDFs.
- Patched `journals/build.sh` so journal wrappers can input master chapter sources while resolving their existing chapter-relative figure and table paths.
- Residual build warnings remain outside the field-deployments rewrite: the compact journal wrappers omit some sections and preliminary theorem labels they still reference, and the thesis wrapper has pre-existing unresolved citation/reference warnings. No fatal LaTeX errors or missing-source-file errors remain in the checked logs.

## Claim Verification Pass (2026-07-17)

Every Tier-1/Tier-2/contrast case was independently re-verified against the on-disk primary source. One fresh reader per paper read the full markdown and ruled each chapter claim SUPPORTED / CONTRADICTED / NOT_FOUND with a verbatim quote, checking every number and date exactly. Fourteen sources, roughly sixty atomic claims.

- Thirteen sources CLEAN. All numeric claims matched the source verbatim: the five YouTube A/B figures (+0.07/+0.53/+0.85/-0.66/-0.52), Taobao 8.67%/18.03% and the January 26 2021 start, Meta bidding 200k campaigns / 1.2B steps / 50B impressions / +0.17%/+0.16%, Alibaba RTB m=24 and 100M instances for 1000 ads, DiDi 1.3%/5.3%, DeepStock 100% and >1M SKU-warehouse, MuZero-RC 6.28% offline versus 4% production, BCOOLER 9%/13%, hotel 11.80% RevPAR, Tmall July 2018.
- One correction applied. The CVNet clause attributed "distillation" to value stabilization; the source uses distillation for feature marginalization (a reduced-feature value for real-time planning when destination features are absent). Only Lipschitz regularization is tied to stability, context randomization to temporal invariance. Sentence rewritten accordingly.
- Advisory, no change needed: Wu2018rtb's own text claims industry deployment but reports no live-traffic metrics, so the contrast label holds; the AlphaChip sequential-placement mechanic is verbatim in the DeepMind post rather than the paywalled Nature body, but both are cited together.
- Cleanup in the same pass: the fifteen `\paragraph{}` case headers became `\subsubsection{}` (matches the rest of the monograph, hidden from the ToC by tocdepth 2), one em-dash pair and two judgment words ("sobering") removed.

## Executive Decision

Collapse the old standalone "RL for Optimal Control" / applied-operations chapter into the late **Reinforcement Learning in the Field** chapter, except for material that is clearly theoretical or not actually RL.

The field chapter becomes the canonical home for real-world RL and deployment-shaped RL examples. It should not be a flat catalog. It should be organized into clusters, with one deeply worked canonical case per cluster and shorter lead-up, contrast, or follow-up cases around it.

The old title "RL for Optimal Control" should not survive unless the manuscript actually covers classical optimal control: HJB, Pontryagin, LQR, MPC, continuous-time stochastic control, and related theory. The current material is not that. It is applied economic and operational control.

## User Preferences Captured

- The new chapter should be late in the monograph, after the methods chapters, so it reads as the downstream synthesis of the whole survey.
- The chapter should be longer and deeper, but not by giving every paper equal weight.
- There should be one canonical example fleshed out per cluster; the other examples should lead up to it, contrast with it, or follow from it.
- Source all papers first.
- Convert each source one by one, not in a combined batch.
- Write one read note per source before using it for prose.
- Do not drop important older or near-real examples; assign each one a home and a label.
- Do not let simulator, offline replay, or deployment-shaped examples masquerade as production deployment.
- Keep the final lessons section substantial and source-driven.
- The final lessons section should answer: how field RL differs from theory, which RL is actually useful, what is missing, what has been tried, which domain conditions matter, and what engineering requirements recur.
- Bus engine belongs in theory as DP versus DQN.
- Google data-center cooling should be removed from the confirmed RL path unless a source proves RL rather than learned predictive control or MPC.
- We need one canonical place for field deployments, not a duplicate applications chapter and deployment chapter saying similar things.

## Presentation Style From the Removed Applications Chapter

The removed `ch04_control_problems/tex/applications.tex` chapter had a stronger worked-case presentation style than the first field-deployments draft. Recover that style during the writing pass, but keep the new evidence screen.

### What the old style did well

- It opened each case with the operational problem in plain economic/engineering language before naming the algorithm.
- It gave the reader a compact MDP anatomy: state, action, reward/cost, transition/horizon, and baseline or incumbent policy.
- It used equations to make the control object concrete, for example DiDi value functions and matching weights, Tmall DRCR, Almgren-Chriss execution schedules, Q-learning updates, inventory costs, RTB budget dynamics, and bus-engine scaling.
- It put the main empirical result into a small table per case, with units and the comparison baseline in the caption.
- It used footnotes as teaching aids for terms that slow down the main prose: semi-MDPs, coarse coding, transfer learning, synthetic control, implementation shortfall, echelon inventory, credit assignment, residual RL, and pricing A/B constraints.
- It connected each case to economic or operational significance rather than just reporting the percentage lift.
- It presented negative or contrast evidence as useful evidence, especially inventory benchmarks and RTB simulation, instead of treating them as failures to mention.

### What should not carry over

- Do not give every case full worked-treatment depth. The old chapter made many cases feel equally central.
- Do not mix field deployment, field experiment, historical backtest, and simulator benchmark without explicit labels.
- Do not keep unsupported deployment language, especially around Google data-center cooling or financial execution.
- Do not let old tables survive unless the numbers are rechecked against the new read notes.
- Do not keep the bus-engine simulation in the field chapter; its style is useful, but the case belongs in theory.

### Target hybrid style for the new chapter

The new chapter should combine the old case-study mechanics with the new source discipline:

1. Start each cluster with the evidence label and cluster thesis.
2. Give one canonical case the old full-treatment structure: domain problem, MDP anatomy, equation, deployment path, results table, integration constraints, and evidence limits.
3. Give supporting cases shorter capsules, but still include the control surface and evidence label.
4. Use compact tables for canonical-case outcomes and one consolidated evidence table for the full chapter.
5. Use footnotes for explanatory machinery and caveats; keep tier decisions in the main text.
6. Close each cluster with a short evidence summary only if needed; reserve cross-case synthesis for the final lessons section.
7. Keep the final lessons section source-driven, but write it in the old chapter's explanatory style: concrete mechanisms first, abstraction second.

### Canonical case presentation template

Use this template for the full-treatment case in each cluster:

1. Operational problem and incumbent baseline.
2. What RL controlled and what it did not control.
3. MDP anatomy: state, action, reward/objective, transition/horizon.
4. One equation or optimization expression that makes the control surface legible.
5. Training data and learning method.
6. Deployment path and evidence label.
7. Result table with baseline, metric, scale, and source.
8. Guardrails, constraints, fallback, or integration details.
9. Why this case worked where generic benchmark RL would not.
10. Evidence limits and contribution to later synthesis.

## Strict Inclusion Standard

Use a strict screen. The chapter is not about "papers that mention industry." It is about systems that crossed into the real world, plus carefully labeled deployment-shaped precursors.

### Tier 1: Confirmed Production or Manufactured Artifact

Tier 1 requires primary-source evidence that the RL-shaped system did at least one of the following:

- served real production traffic,
- launched as a primary operational mode,
- controlled a real production workflow,
- controlled a real marketplace or platform operation,
- produced or shaped a real manufactured or shipped artifact.

Examples expected to qualify if the read notes confirm the current understanding:

- YouTube Top-K off-policy REINFORCE recommender.
- Meta/Facebook Horizon notification policies.
- Meta production bidding-policy optimization.
- DiDi scalable RL dispatch.
- Alibaba/Taobao RL-LTV cold-start recommendation.
- Alibaba sponsored-search real-time bidding.
- Alibaba/Tmall DeepStock inventory replenishment.
- Google DeepMind/YouTube MuZero-RC video rate control.
- Google AlphaChip floorplanning for real chips.
- OpenAI InstructGPT/ChatGPT RLHF post-training, with the caveat that RL is training-time rather than inference-time control.

### Tier 2: Real Field Trial

Tier 2 means direct operational control in a real facility, market, or business setting, but without public evidence of broad or durable production rollout.

Expected examples:

- DeepMind/Trane BCOOLER commercial-building cooling.
- Hotel revenue management field experiment, unless source evidence supports a stronger production label.
- Tmall dynamic pricing field experiment, unless source evidence supports a stronger production label.

### Tier 3: Deployment-Shaped but Not Confirmed

Tier 3 means realistic state/action/reward and realistic data, but no public proof that RL controlled live production.

Expected examples:

- Financial order execution on real limit-order-book data.
- Facebook 360 video adaptive bitrate, unless stronger evidence is sourced.

### Contrast or Negative Evidence

These cases are useful because they show why naive RL is not enough.

Expected examples:

- RTB simulator calibrated to production data as a lead-up to Alibaba/Meta bidding.
- Generic inventory benchmark as negative evidence before DeepStock.
- Classical base-stock and operations-research baselines that outperform generic DRL.

### Excluded

Exclude from confirmed deployment evidence:

- simulator-only papers,
- offline replay only,
- benchmark environments,
- vague "industrial application" claims,
- production ML where the source does not establish RL,
- Google data-center cooling if the public source is MPC/learned predictive control rather than RL.

Excluded material can appear only as a cautionary footnote or contrast if it directly helps the reader avoid misclassification.

## What Happens to the Old Applications Chapter

Target end state:

1. Remove `ch04_control_problems` from the main manuscript wrappers, or reduce it to a very short bridge only if the manuscript flow needs it.
2. Move the genuinely field-relevant sections into `ch13_field_deployments/tex/field_deployments.tex`.
3. Preserve deployment-shaped non-deployments as short lead-up or contrast capsules inside the relevant clusters.
4. Keep bus engine only in `ch03_theory`.
5. Do not resurrect Google data-center cooling as confirmed RL.

If a short bridge remains, it should be titled **Economic Control Problems as MDPs**, not "RL for Optimal Control." It should only introduce domain anatomy:

- state,
- action,
- reward,
- constraints,
- baseline policy,
- why exact DP is hard,
- what RL approximates.

It should not duplicate the field chapter or contain deployment adjudication.

## Migration Map for Existing Material

| Existing material | Current/old home | Final home | Role | Evidence label |
|---|---|---|---|---|
| Bus engine replacement | `ch04_control_problems`; now moved to `ch03_theory` | `ch03_theory` only | DP vs DQN scaling example | Theory/simulation, not field |
| Google data-center cooling comment | Retired comments in old applications chapter | Remove or cautionary footnote only | Misclassification warning | Excluded unless RL is proven |
| DiDi dispatch | Old applications chapter | Field deployments, marketplace cluster | Canonical or major marketplace-control case | Tier 1 candidate |
| Hotel revenue management | Old applications chapter | Field deployments, marketplace/pricing cluster | Field experiment follow-up | Tier 2 / field experiment |
| Tmall dynamic pricing | Old applications chapter | Field deployments, marketplace/pricing cluster | Field pricing follow-up | Tier 2 or applied deployment, source-dependent |
| Financial order execution | Old applications chapter | Field deployments, finance capsule | Deployment-shaped near-real case | Tier 3 unless stronger evidence |
| Supply-chain inventory benchmark | Old applications chapter | Field deployments, inventory cluster | Negative evidence before DeepStock | Contrast / negative evidence |
| RTB simulator | Old applications chapter | Field deployments, bidding cluster | Academic precursor to production bidding | Contrast / precursor |
| DiDi older CVNet material | Old applications chapter | Field deployments, DiDi support | Background/technical support for dispatch | Supporting evidence |
| Lyft dispatch material | Old applications chapter | Field deployments if sourced | Follow-up/comparison only | Tier depends on source |

## Field Chapter Architecture

The chapter should be organized by deployment surface, not by algorithm family. Each cluster should have one canonical case treated deeply and supporting cases treated briefly.

| Cluster | Canonical case | Supporting cases | Purpose |
|---|---|---|---|
| Recommendations and notifications | YouTube Top-K off-policy REINFORCE recommender | Meta notifications; Taobao RL-LTV | Show RL in large-scale digital traffic and ranking systems. |
| Bidding and auctions | Meta production bidding | RTB simulator as precursor; Alibaba sponsored-search RTB as field follow-up | Show narrow bid/budget control surfaces and conservative production integration. |
| Marketplaces and pricing | DiDi dispatch | Hotel revenue management; Tmall dynamic pricing | Show online marketplace/control problems with real economic consequences. |
| Inventory and operations | DeepStock | Generic inventory benchmark as negative evidence | Show why operations deployments need structure, regularization, and classical baselines. |
| Physical and design systems | MuZero-RC or AlphaChip | BCOOLER; the other of MuZero-RC/AlphaChip if space permits | Show narrow physical/control/design surfaces where RL can survive engineering constraints. |
| RLHF and post-training | InstructGPT/ChatGPT RLHF | Product-post caveats and deployment distinction | Show deployed training-time RL, not online runtime control. |

## Canonical Case Depth Rule

Only the canonical case in each cluster receives full treatment. Full treatment means:

1. What RL actually controlled.
2. Whether control happened at serving time, operations time, design time, or training time.
3. State, action, reward, and horizon.
4. RL method and training data.
5. Deployment evidence from primary source text.
6. Measured outcome and why the effect size matters.
7. Safety, constraints, guardrails, counterfactual evaluation, and fallback mechanisms.
8. How the RL component integrated with legacy infrastructure.
9. Why this case worked where textbook or benchmark RL alone would not be enough.
10. What evidence the case contributes to the final synthesis.

Supporting cases should usually be one to three paragraphs each. They should not become mini-chapters unless the source evidence is unusually strong.

## Draft Chapter Outline

### 1. Opening Screen

Core claim: real deployments exist, but public evidence supports a narrower and more conservative story than survey language often implies.

Points to make:

- Fielded RL is not "train an unconstrained neural policy and let it run the company."
- Successful examples use narrow levers, bounded action spaces, strong logging, extensive evaluation, and legacy-system integration.
- Offline/simulator success is not enough.
- The chapter classifies systems by evidence standard before drawing lessons.

### 2. Recommendations and Notifications

Canonical case: YouTube Top-K off-policy REINFORCE recommender.

Full treatment:

- Candidate generator chooses video slate/candidates for homepage/watch-page recommendation.
- Trained from logged production data with behavior-policy correction and top-K off-policy correction.
- Production traffic and ViewTime lift are the evidence hooks.
- Important engineering themes: propensities, logged data, exploration slices, off-policy correction, separation between candidate generation and ranking.

Supporting cases:

- Meta/Facebook Horizon notification policies: DQN send/drop decisions, PID thresholding, daily retraining, production serving.
- Taobao RL-LTV: cold-start recommendation and lifetime-value score blended with existing ranking.

Later synthesis material:

- Digital traffic deployments work when RL controls a narrow ranking/gating component, not the whole product.

### 3. Bidding and Auctions

Canonical case: Meta production bidding-policy optimization.

Full treatment:

- RL optimizes parameters of a trusted bidding controller rather than deploying a raw neural policy.
- Offline RL/CQL-style training on production logs.
- Production A/B tests over very large impression counts.
- Safety mechanism is architectural: tune a base policy, do not replace the serving stack with a black-box neural controller.

Supporting cases:

- RTB simulator from old applications chapter: deployment-shaped academic precursor, not live deployment.
- Alibaba sponsored-search RTB: stronger field/production evidence and useful comparison to Meta.

Later synthesis material:

- Bidding deployments favor parameterized control surfaces, budget/bid multipliers, and conservative integration with auction infrastructure.

### 4. Marketplaces and Pricing

Canonical case: DiDi dispatch.

Full treatment:

- Real-time driver-order matching.
- RL value function feeds matching weights rather than replacing the whole dispatch optimizer.
- State/value estimation, cancellation risk, bandit pruning, online marketplace feedback.
- Evidence: deployed in multiple cities and launched as primary dispatch mode in a major market, subject to read-note confirmation.

Supporting cases:

- Hotel revenue management: real field experiment, scalar discount/control recommendation plus LP allocation layer.
- Tmall dynamic pricing: field pricing experiment, DQN/DDPG price control, but label carefully based on source.

Later synthesis material:

- Marketplace RL works when the RL signal is embedded inside an optimization/control layer that operations teams can reason about.

### 5. Inventory and Operations

Canonical case: DeepStock.

Full treatment:

- Action is SKU-warehouse replenishment order quantity.
- Deployed policy regularized toward inventory-theoretic structures.
- Source claims broad rollout across Tmall SKU-warehouse pairs, subject to read-note confirmation.
- Important contrast: naive DRL often loses to base-stock policies.

Supporting cases:

- Generic inventory benchmark from old applications chapter: negative evidence, not deployment.
- Classical base-stock and echelon-inventory policies as required baselines.

Later synthesis material:

- Operations deployments need theory-shaped RL, not generic DRL. The baseline is often a strong OR policy, not a weak heuristic.

### 6. Physical and Design Systems

Canonical choice to decide after read notes:

- MuZero-RC if the emphasis is online production traffic in a codec subsystem.
- AlphaChip if the emphasis is RL producing manufactured artifacts.

MuZero-RC treatment:

- Frame-level quantization/rate-control decision in VP9.
- Narrow subsystem replacement, not entire codec replacement.
- Evidence from DeepMind post and arXiv paper.

AlphaChip treatment:

- RL places chip blocks during design.
- Deployment is design-time, not runtime.
- Real manufactured TPU/chip artifacts make it field-relevant.

Supporting case:

- BCOOLER: Tier 2 real HVAC field trial, useful because it shows physical-control guardrails, fallback, uncertainty penalties, and constraints.

Later synthesis material:

- Physical/design deployments work when RL is restricted to a well-scoped subsystem inside conventional engineering verification.

### 7. RLHF and Post-Training

Canonical case: InstructGPT/ChatGPT RLHF.

Full treatment:

- RL optimizes model parameters during post-training.
- RL policy is the language model during training, but deployment is ordinary inference from the trained model.
- This is real deployed RL influence, but not online sequential control at serving time.
- Use the OpenAI paper as the local source; product posts are browser-verifiable but local `curl` is blocked and should remain marked accordingly unless separately sourced.

Later synthesis material:

- RL can be production-critical even when it is not an online runtime controller.

### 8. Lower-Confidence and Excluded Cases

Include a short section or table for:

- Facebook 360 adaptive bitrate: Tier 3 unless better source evidence is found.
- Google data-center cooling: exclude from confirmed RL unless source proves RL rather than MPC/learned predictive control.
- Simulator/offline-only cases from surveys: not counted.

Purpose:

- Make the screen credible by showing what is not counted.

### 9. Long Lessons from the Field

Write only after read notes are complete.

This is the most important section. It should be a real synthesis, not a list of case summaries.

Required questions:

- How does field RL differ from theoretical RL?
- Which RL families have actually proved useful?
- What is missing from public evidence?
- What has been tried but remained simulation, replay, or field-trial status?
- Which domain conditions make RL viable?
- What engineering requirements recur across successful deployments?
- How should economists and applied researchers interpret production RL effect sizes?
- Why do small percentage lifts matter in production systems?
- What is the gap between academic offline RL and deployed offline RL?

## Long Lessons Section: Detailed Content Plan

### Field RL versus theoretical RL

Expected claims to test against read notes:

- Theory often studies unconstrained MDPs; field systems expose a narrow control surface.
- Theory often treats reward as known; field systems fight delayed, noisy, proxy, or multi-objective rewards.
- Theory often assumes clean exploration; field systems need logged propensities, guarded exploration, or no online exploration.
- Theory asks for optimality; field systems ask for reliable improvement over a strong incumbent.
- Theory treats the policy as the whole agent; field systems embed RL inside rankers, matchers, controllers, encoders, or post-training pipelines.

### Which RL has been useful

Expected categories:

- Off-policy policy gradient / REINFORCE with correction in recommender systems.
- DQN-style value learning for narrow discrete gates such as notifications.
- Offline RL as optimizer of production-controller parameters, not necessarily a deployed neural policy.
- TD value estimation inside optimization layers for dispatch.
- Actor-critic methods when action spaces are scalar or low-dimensional and heavily constrained.
- Model-based RL/planning in narrow engineered subsystems such as rate control.
- RLHF/PPO for training-time alignment of LLMs.
- RL for design-time optimization such as chip floorplanning.

### What is missing

Expected claims:

- Little public evidence that canonical academic offline-RL algorithms such as CQL, IQL, MOPO, COMBO, RAMBO, or ARMOR are deployed as unconstrained black-box policies controlling major consumer systems.
- Limited public evidence for continuously online-learning physical controllers without strong wrappers.
- Sparse evidence in healthcare and robotics compared with survey rhetoric.
- Many papers stop at simulators, logged replay, or field-like datasets.

### Domain conditions that seem necessary

Expected conditions:

- Repeated decisions at scale.
- Clear action logging.
- Recoverable or bounded mistakes.
- Measurable business or physical outcomes.
- Strong incumbent baseline.
- A narrow intervention surface.
- Sufficient data volume.
- Ability to run A/B tests, field trials, or credible counterfactual evaluation.
- Ability to impose action masks, constraints, fallbacks, or human override.
- Long-run effects that myopic supervised learning or one-step optimization misses.

### Engineering requirements

Long section. Required ingredients:

- Propensity logging.
- Stable reward definitions and delayed-outcome attribution.
- Counterfactual/off-policy evaluation.
- Exploration discipline.
- Guardrails and action masks.
- Constraint handling.
- Fallback policy.
- Shadow/canary/A/B testing.
- Monitoring after launch.
- Retraining cadence.
- Ownership by product/ops/infra teams.
- Integration with existing controllers, optimizers, rankers, or design flows.
- Interpretability sufficient for operators.
- Incident response and rollback.
- Clear distinction between training-time RL and runtime RL.

## Source Discipline

No case can support prose until the source has:

1. A downloaded original on disk.
2. A same-stem markdown conversion on disk.
3. A read note written by the main agent.
4. A tier decision tied to primary-source text.

Conversion rule:

- Convert one source at a time.
- Keep the original beside the markdown.
- Do not hand-write markdown from memory or a web snippet.
- If Docling fails, use `tomd` fallback and record the caveat.
- If a source is blocked, mark it blocked or failed. Do not silently substitute a secondary source.

Current known caveat:

- Updated 2026-07-16: the local corpus in `ch13_field_deployments/papers/` has been reconverted with Docling 2.45.0. The conversion used placeholder image export plus whole-document, chunked, and page-level fallback as needed. The conversion report is `ch13_field_deployments/papers/_websource_logs/docling-2.45-final-report.tsv`.
- Resolved 2026-07-17: the OpenAI product pages are Cloudflare-403 to `curl` and WebFetch on the live domain, but the InstructGPT (Jan 27 2022) and ChatGPT (Nov 30 2022) posts were sourced through their Internet Archive snapshots and are now local originals plus markdown under `ch13_field_deployments/papers/`, with a read note at `papers/read_notes/openai-product-posts.md`. The InstructGPT post supplies the deployment claim the paper alone did not, that the RLHF models were the default API models.

## Current Source Corpus

The source gate currently passes for the local originals and markdown files in `ch13_field_deployments/papers/`. The operational checklist with exact read-note paths is `ch13_field_deployments/source_manifest.md`.

| ID | System | Current source status | Read status |
|---|---|---|---|
| youtube-recs | YouTube Top-K off-policy REINFORCE | DONE | READ_DONE |
| meta-horizon | Meta/Facebook Horizon notifications | DONE | READ_DONE |
| meta-bidding | Meta production bidding | DONE | READ_DONE |
| alibaba-rtb | Alibaba sponsored-search RTB | DONE | READ_DONE |
| taobao-rltv | Taobao RL-LTV cold start | DONE | READ_DONE |
| didi-dispatch | DiDi scalable RL dispatch | DONE | READ_DONE |
| didi-ride-hailing-dispatch | DiDi ride-hailing order dispatch | DONE | READ_DONE |
| didi-cvnet | DiDi CVNet multi-driver dispatch | DONE | READ_DONE |
| didi-mean-field-ridesharing | DiDi mean-field ridesharing dispatch | DONE | READ_DONE |
| lyft-rl-matching | Lyft driver-rider RL matching | DONE | READ_DONE |
| deepstock | Alibaba/Tmall DeepStock | DONE | READ_DONE |
| muzero-rc-paper | MuZero-RC paper | DONE | READ_DONE |
| muzero-rc-post | MuZero-RC DeepMind post | DONE | READ_DONE |
| alphachip-nature | AlphaChip Nature page | DONE | READ_DONE |
| alphachip-post | AlphaChip DeepMind post | DONE | READ_DONE |
| openai-instructgpt-paper | OpenAI InstructGPT paper | DONE | READ_DONE |
| bcooler | DeepMind/Trane BCOOLER | DONE | READ_DONE |
| tmall-pricing | Tmall dynamic pricing field experiment | DONE | READ_DONE |
| hotel-rm | Hotel revenue management field experiment | DONE | READ_DONE |
| execution | Financial order execution | DONE | READ_DONE |
| rtb-simulator | Budget-constrained RTB simulator | DONE | READ_DONE |
| inventory-benchmark | DRL inventory benchmark | DONE | READ_DONE |
| facebook-abr | Facebook 360 adaptive bitrate via Horizon source | DONE | READ_DONE |

Remaining caveat:

- Closed 2026-07-17. The OpenAI product posts were sourced through Internet Archive snapshots (the live pages are Cloudflare-403 to `curl` and WebFetch), converted to markdown, and read. `openai-product-posts` is now READ_DONE in the source manifest. The RLHF subsection now cites both posts for the deployment claim while preserving the training-time-versus-runtime distinction.

## Read Note Template

Create one markdown file per source in `ch13_field_deployments/papers/read_notes/`.

Template:

```markdown
# {System}

- Source original:
- Source markdown:
- Canonical URL:
- Source type:
- What RL controlled:
- State:
- Action:
- Reward/objective:
- Horizon:
- RL method:
- Training data:
- Deployment evidence:
- Measured outcomes:
- Safety/constraints/integration:
- What is not proven:
- Tier decision:
- Cluster role:
- Prose implications:
- Quotes or exact source anchors:
```

Read-note rule:

- The note should separate direct evidence from inference.
- It should record exact phrasing only in short compliant snippets.
- It should state when a case is deployment-shaped but not confirmed deployed.

## Ten-Step Implementation Plan

1. Keep this master plan as the durable source of truth.
2. Keep `ch13_field_deployments/` as the chapter home.
3. Maintain the source manifest with every candidate case and its status.
4. Source missing old-chapter cases one by one, keeping originals beside markdown.
5. Write read notes for every Tier 1 and Tier 2 candidate before prose.
6. Write read notes for deployment-shaped/negative-evidence cases before using them as contrast.
7. Decide the canonical case for each cluster after read notes, not before.
8. Move old applications material into field clusters, preserving only useful content and labels.
9. Remove or stop including the old applications chapter once useful content has migrated.
10. Draft the long lessons section last, then build and audit all wrappers.

## Detailed Implementation Phases

### Phase 1: Source and Read

- Verify the source gate still passes for the current corpus.
- Source old-chapter cases that are going to survive.
- Write read notes for YouTube, Meta Horizon, Meta bidding, Alibaba RTB, Taobao RL-LTV, DiDi, DeepStock, MuZero-RC, AlphaChip, OpenAI RLHF, and BCOOLER.
- Write shorter read notes for Tmall pricing, hotel RM, financial execution, RTB simulator, and inventory benchmark.

### Phase 2: Decide Canonical Cases

Provisional canonical choices:

- Recommendations/notifications: YouTube.
- Bidding/auctions: Meta bidding.
- Marketplaces/pricing: DiDi.
- Inventory/operations: DeepStock.
- Physical/design: decide between MuZero-RC and AlphaChip after read notes.
- RLHF/post-training: InstructGPT/ChatGPT.

Do not lock these until read notes are complete.

### Phase 3: Migrate Old Chapter Content

- Move DiDi into marketplace cluster.
- Move hotel revenue management into marketplace/pricing supporting case.
- Move Tmall dynamic pricing into marketplace/pricing supporting case.
- Move RTB simulator into bidding precursor paragraph.
- Move financial execution into finance/deployment-shaped capsule.
- Move inventory benchmark into negative-evidence setup for DeepStock.
- Leave bus engine in theory.
- Delete or stop including the old applications chapter.

### Phase 4: Draft Field Chapter

Draft in this order:

1. Evidence screen and classification table.
2. Recommendations and notifications.
3. Bidding and auctions.
4. Marketplaces and pricing.
5. Inventory and operations.
6. Physical and design systems.
7. RLHF and post-training.
8. Lower-confidence/excluded cases.
9. Long lessons section.

### Phase 5: Build and Audit

- Run source gate.
- Build `docs/main.tex`.
- Build journal wrappers if they include the new chapter.
- Check undefined references.
- Search for old "RL for Optimal Control" title.
- Search for stale references to `section:applications` if the chapter is removed.
- Check that no deployment-shaped case is accidentally presented as Tier 1.

## Prose Rules

- Use concrete evidence labels.
- Do not use survey rhetoric as proof of deployment.
- Do not write "deployed" unless the source says the system served, launched, controlled, or produced a real artifact.
- Use "field experiment," "field trial," "simulator," "offline replay," "deployment-shaped," or "negative evidence" when those are the accurate labels.
- Keep source and inference separate.
- Make effect sizes legible in business terms when small percentages are large at scale.
- Emphasize engineering integration, not just algorithm names.
- Avoid a flat literature-review list.
- Avoid overstating older or ambiguous cases.

## Final Acceptance Criteria

- There is one canonical **Reinforcement Learning in the Field** chapter.
- There is no duplicate "RL for Optimal Control" chapter unless reduced to a short MDP-domain bridge.
- Bus engine appears only in theory.
- Google data-center cooling is not counted as confirmed RL.
- Every included case has a clear evidence label.
- Every Tier 1/Tier 2 claim is traceable to primary-source text.
- Deployment-shaped but non-deployed examples are clearly marked as precursor, contrast, near-real, or negative evidence.
- The reader can distinguish production deployment, field trial, simulator, offline replay, and theory at a glance.
- The long lessons section is written after read notes, not from intuition.
- All wrappers build after the chapter topology changes.
