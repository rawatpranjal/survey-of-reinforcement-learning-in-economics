# Proofread Report: Field Deployments Chapter

File checked: `ch13_field_deployments/tex/field_deployments.tex`

Included files checked:

- `ch13_field_deployments/sims/field_ope_reliability_table.tex`
- `ch13_field_deployments/sims/field_ope_reliability_candidates.tex`
- `ch13_field_deployments/sims/field_ope_reliability_macros.tex`

Reference file checked for cited entries: `docs/refs.bib`

Date: 2026-07-19

Mode: default proofread report, no in-place fixes

Total findings: 47

## Summary

| Rule | Count |
|---|---:|
| R3 reference entry incomplete | 1 |
| R4 first person | 1 |
| R5 slash in prose | 4 |
| R6 acronym before expansion | 16 |
| R12 directional language | 4 |
| R13 cross-reference by number | 10 |
| R14 grammar or word choice | 8 |
| R15 ambiguous citation grouping | 3 |

## R3 - Reference Entry Incomplete

[R3 reference-entry] `docs/refs.bib:1816` | "`@inproceedings{SadeghiEshkevari2022DiDiScalable`" | conference reference has `booktitle` and `year` but no page range -> add the proceedings page range if available; do not guess the values.

## R4 - First Person

[R4 first-person] `ch13_field_deployments/tex/field_deployments.tex:334` | "I treat MuZero-RC as the canonical physical/design runtime-control case" | journal style bans first person -> recast as "This chapter treats MuZero-RC as the canonical ..." or "MuZero-RC is treated here as the canonical ...".

## R5 - Slash or "and/or" in Prose

[R5 slash] `ch13_field_deployments/tex/field_deployments.tex:20` | "A/B performance" | slash form recurs throughout the chapter in prose and table text; if house style disallows slash compounds, batch-recast as "randomized-test performance," "online experiment," or "split test" where appropriate.

[R5 slash] `ch13_field_deployments/tex/field_deployments.tex:334` | "DeepMind/YouTube post" | slash in prose leaves the relationship implicit -> use "DeepMind and YouTube post" or "joint DeepMind-YouTube post," depending on the source attribution.

[R5 slash] `ch13_field_deployments/tex/field_deployments.tex:334` | "physical/design runtime-control case" | slash in prose leaves the category ambiguous -> write "physical or design runtime-control case" or name the intended category directly.

[R5 slash] `ch13_field_deployments/tex/field_deployments.tex:391` | "historical/generative methodology" | slash in prose leaves the method ambiguous -> write "historical or generative methodology" or specify the exact evaluation design.

## R6 - Acronym Used Before Being Spelled Out

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:3` | "RLHF" | acronym appears before expansion -> spell out "reinforcement learning from human feedback (RLHF)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:5` | "ML controller" | acronym appears before expansion -> use "machine-learning controller" or introduce "machine learning (ML)" if the acronym is needed.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:38` | "RL-LTV" | acronym appears before expansion in the case table -> spell out "reinforcement-learning lifetime value (RL-LTV)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:40` | "RTB" | acronym appears before expansion -> spell out "real-time bidding (RTB)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:40` | "Robust-MDP" | acronym appears before expansion -> write "robust Markov decision process (robust MDP)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:42` | "DRL" | acronym appears before expansion -> spell out "deep reinforcement learning (DRL)" or avoid the acronym.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:42` | "SKU-warehouse" | acronym appears before expansion -> introduce "stock-keeping unit (SKU)" before the case table or spell it out in the row.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:45` | "PPO" | acronym appears before expansion -> spell out "proximal policy optimization (PPO)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:46` | "HVAC" | acronym appears before expansion -> spell out "heating, ventilation, and air-conditioning (HVAC)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:78` | "RNN candidate-generation model" | acronym appears before expansion -> spell out "recurrent neural network (RNN)" at first use.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:99` | "DQN-style policy" | acronym appears before expansion -> spell out "deep-Q-network (DQN)-style policy" or "deep Q-network style policy."

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:102` | "Vanilla-CTR baseline" | acronym appears before expansion -> spell out "vanilla click-through-rate (CTR) baseline" or avoid the acronym.

[R6 acronym] `ch13_field_deployments/tex/field_deployments.tex:204` | "95\% CI" | acronym appears before expansion in a table header -> use "95\% confidence interval" or define "confidence interval (CI)" before the table.

[R6 acronym] `ch13_field_deployments/sims/field_ope_reliability_table.tex:3` | "DM, per-decision importance sampling, doubly robust" | estimator abbreviations are not fully introduced in the standalone caption/table -> write "direct method (DM), per-decision importance sampling (PDIS), and doubly robust (DR)."

[R6 acronym] `ch13_field_deployments/sims/field_ope_reliability_candidates.tex:4` | "DM error is ...; IS effective sample size" | `IS` is not expanded in the standalone caption -> write "importance-sampling (IS) effective sample size."

[R6 acronym] `ch13_field_deployments/sims/field_ope_reliability_candidates.tex:16` | "CQL (offline RL)" | acronym appears in a table row without expansion -> write "conservative Q-learning (CQL; offline RL)" if the row width permits.

## R12 - Directional Language

[R12 directional] `ch13_field_deployments/tex/field_deployments.tex:5` | "The cases below" | directional language depends on layout -> recast as "The cases in this chapter."

[R12 directional] `ch13_field_deployments/tex/field_deployments.tex:288` | "the benchmark above" | directional language depends on layout -> refer to "the inventory benchmark" or the named subsection.

[R12 directional] `ch13_field_deployments/tex/field_deployments.tex:413` | "the cases above" | directional language depends on layout -> recast as "the deployment cases in this chapter."

[R12 directional] `ch13_field_deployments/tex/field_deployments.tex:485` | "the shape above" | directional language depends on layout -> recast as "this profile" or "the profile in Table~\\ref{tab:field_shape}."

## R13 - Cross-Reference by Number Rather Than Section Title

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:125` | "Section~\\ref{section:rl_algorithms}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:145` | "Section~\\ref{section:offline_rl}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:153` | "Section~\\ref{section:offline_rl}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:226` | "Section~\\ref{section:rl_algorithms}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:409` | "Section~\\ref{section:offline_rl} ... Section~\\ref{section:causal_rl}" | ORE-style cross-references use section titles rather than section numbers -> replace with quoted section titles if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:443` | "Section~\\ref{sec:field_ope_sim}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:448` | "Section~\\ref{subsec:overlapping_terminology} ... Section~\\ref{sec:liu} ... Section~\\ref{section:language}" | ORE-style cross-references use section titles rather than section numbers -> replace with quoted section titles if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:452` | "Section~\\ref{section:offline_rl} ... Section~\\ref{sec:field_ope_sim}" | ORE-style cross-references use section titles rather than section numbers -> replace with quoted section titles if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:454` | "Section~\\ref{sec:field_lessons}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

[R13 cross-ref] `ch13_field_deployments/tex/field_deployments.tex:487` | "Section~\\ref{sec:field_ope_sim}" | ORE-style cross-references use section titles rather than section numbers -> replace with the quoted section title if this venue requires that style.

## R14 - Grammar, Missing Verb, or Flagged Word Choice

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:1` | "Their success conditions, closed game rules, dense simulated experience, preference-labeled model training, or specialized robotic hardware loops, do not transfer" | comma separates the subject from the verb and the list blurs "success conditions" with examples -> recast, e.g. "Those success conditions do not transfer: closed game rules, dense simulated experience, preference-labeled model training, and specialized robotic hardware loops are different from ..."

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:78` | "which are sub-percent ViewTime movements, component tests rather than a wholesale replacement" | appositive lacks a clear predicate -> recast as "The key point is that these are sub-percent ViewTime movements and component tests rather than a wholesale replacement ..."

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:302` | "a policy whose base-stock parameterization operators can read" | phrase can be misread as "parameterization operators" -> recast as "a policy with a base-stock parameterization that operators can read."

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:340` | "Cooling is a large and climate-relevant load, space cooling alone is about a tenth of world electricity demand" | comma splice joins two independent clauses -> split the sentence or use a semicolon.

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:342` | "The reinforcement-learning agent does not replace those controllers, it recommends the setpoints they track" | comma splice joins two independent clauses -> use a semicolon, "rather," or split the sentence.

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:441` | "Logged data has to carry the probability each action was taken" | inconsistent singular treatment of "data" in a chapter that elsewhere uses plural "data are" -> change to "Logged data have to carry ..." or recast as "The log has to carry ..."

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:454` | "because the domain, not the modeler, sets what is possible, a randomized test for a digital product" | comma splice turns the examples into a second clause -> use a colon before the examples or split the sentence.

[R14 grammar] `ch13_field_deployments/tex/field_deployments.tex:483` | "The data a deployment needs is domain-dependent" | inconsistent singular treatment of "data" -> change to "The data a deployment needs are domain-dependent" or recast as "A deployment's data needs are domain-dependent."

## R15 - Ambiguous Citation Grouping

[R15 citation-grouping] `ch13_field_deployments/tex/field_deployments.tex:1` | "Games, language-model post-training, and robotics are set aside, each already carrying its own deep survey tradition \\citep{Shao2019VideoGamesSurvey,Kaufmann2023RLHFSurvey,Ibarz2021robot,Tang2024RoboticsSurvey}" | one citation group is asked to support three different survey traditions -> split the citations by domain so readers can tell which source maps to games, RLHF, and robotics.

[R15 citation-grouping] `ch13_field_deployments/tex/field_deployments.tex:47` | "Tmall pricing, hotel revenue management ... \\citep{Liu2019,Chen2023hotelrl}" | two case labels share one grouped citation -> attach the Liu citation to Tmall and the Chen citation to hotels, or add parenthetical labels.

[R15 citation-grouping] `ch13_field_deployments/tex/field_deployments.tex:48` | "Financial execution, RTB simulator, inventory benchmark ... \\citep{Nevmyvaka2006execution,Wu2018rtb,Gijsbrechts2022inventory}" | three contrast cases share one grouped citation -> attach each citation to its case or add parenthetical labels.

## Notes

- R6b and R13 are ORE-specific. If this monograph keeps numbered LaTeX cross-references by design, the R13 findings can be ignored as a venue-style choice.
- No R1 or R2 findings were recorded. The quotation marks in this chapter are scare quotes or terms of art rather than direct quoted source passages.
- No in-place fixes were applied.
