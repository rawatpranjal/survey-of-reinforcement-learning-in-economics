# Reinforcement Learning in the Field: Chapter Plan

Date: 2026-07-16

## Goal

Build a late monograph chapter, **Reinforcement Learning in the Field**, that answers a narrower question than the existing applications chapter: which reinforcement-learning systems have actually controlled production traffic, physical systems, marketplaces, or manufactured artifacts?

The chapter should come after the method chapters, especially offline RL, RLHF, robust/constrained RL, and world models. The reader should arrive with enough machinery to understand why fielded RL looks more conservative than textbook or benchmark RL.

## Inclusion Standard

Use a strict evidence screen.

- **Tier 1:** the primary source says the RL system served production traffic, launched as a primary operational mode, controlled a real production workflow, or produced a real manufactured/design artifact.
- **Tier 2:** the system directly controlled a real facility, marketplace, or operational process in a live field trial, but the source does not establish broad or durable production rollout.
- **Tier 3:** the source hints at internal use or deployment but lacks enough public detail about scope, metrics, or whether RL was on the critical path.
- **Excluded:** simulator-only, offline replay only, benchmark-only, vague "industrial application," or production ML where the public source does not establish RL.

## Ten-Step Implementation Plan

1. Create this planning document and make it the durable source of truth for the migration.
2. Add a new chapter home at `ch13_field_deployments/`.
3. Build a source manifest before prose, with one row per candidate deployment.
4. Source papers one at a time, keeping the downloaded original beside the converted markdown.
5. Convert each source with `tomd`/Docling sequentially, never as a batch.
6. Read each converted source and write a short read note before using it in prose.
7. Move the bus-engine simulation out of the applications chapter and into the theory chapter as a DP-vs-DQN scaling example.
8. Assign every old applications-chapter case a home: field-deployment core, domain-method chapter, background note, or explicit exclusion.
9. Draft the deployment chapter from the verified manifest, organized by deployment surface.
10. Write a long lessons section after all cases are read, then build and audit the manuscript.

## Migration Map for Existing Material

| Existing material | Current home | Target home | Status |
|---|---|---|---|
| DiDi dispatch | `ch04_control_problems` | Field deployments, marketplace control | Keep; re-source against newest dispatch deployment paper before expanding. |
| Hotel revenue management | `ch04_control_problems` | Field deployments or operations appendix | Keep as non-tech field evidence unless space requires appendix treatment. |
| Tmall dynamic pricing | `ch04_control_problems` | Field deployments plus bandits/pricing cross-reference | Keep; distinguish from Taobao RL-LTV and DeepStock. |
| Financial execution | `ch04_control_problems` | Domain-method background unless deployment evidence strengthens | Keep out of Tier 1 unless source proves production use. |
| Inventory comparison | `ch04_control_problems` | World models / operations simulation background | Keep as "what RL does not dominate" evidence, not as deployment. |
| Real-time bidding simulator | `ch04_control_problems` | Background to Alibaba sponsored-search deployment | Keep only if useful as contrast with confirmed Alibaba RTB deployment. |
| Bus engine replacement | `ch04_control_problems` | Theory chapter, DP vs DQN scaling | Move now; it is a theory/simulation benchmark, not a deployment. |
| Google data-center cooling | retired comments | Exclude or footnote | Exclude from confirmed list unless source establishes RL rather than predictive control/MPC. |

## Source Manifest

The manifest below is the working checklist. A case cannot support chapter prose until `source_status`, `markdown_status`, and `read_status` are complete.

| ID | System | Domain | Tier target | Primary source target | Source status | Markdown status | Read status | Prose home |
|---|---|---|---|---|---|---|---|---|
| youtube-recs | YouTube Top-K off-policy REINFORCE recommender | Recommendation | Tier 1 candidate | arXiv:1812.02353 | DONE | DONE | TODO | Digital traffic |
| meta-horizon | Facebook/Meta Horizon notification policies | Notifications | Tier 1 candidate | arXiv:1811.00260 | DONE | DONE | TODO | Digital traffic |
| meta-bidding | Meta production bidding-policy optimization | Ads and bidding | Tier 1 candidate | arXiv:2310.09426 | DONE | DONE | TODO | Digital traffic |
| alibaba-rtb | Alibaba sponsored-search real-time bidding | Ads and bidding | Tier 1 candidate | arXiv:1803.00259 / KDD 2018 | DONE | DONE | TODO | Digital traffic |
| taobao-rltv | Alibaba/Taobao RL-LTV cold-start recommendation | Recommendation | Tier 1 candidate | arXiv:2108.09141 | DONE | DONE | TODO | Digital traffic |
| didi-dispatch | DiDi scalable RL dispatch | Marketplace dispatch | Tier 1 candidate | arXiv:2202.05118 / KDD 2022 | DONE | DONE | TODO | Marketplace control |
| deepstock | Alibaba/Tmall DeepStock replenishment | Inventory operations | Tier 1 candidate | arXiv:2603.19621 | DONE | DONE | TODO | Operations |
| muzero-rc | Google DeepMind/YouTube MuZero-RC | Video encoding | Tier 1 candidate | DeepMind post + arXiv paper | DONE | DONE | TODO | Physical/design systems |
| alphachip | Google AlphaChip | Chip floorplanning | Tier 1 candidate | Nature 2021 + official follow-ups | DONE | DONE | TODO | Physical/design systems |
| openai-rlhf | OpenAI InstructGPT/ChatGPT RLHF | LLM post-training | Tier 1 candidate | Ouyang 2022 official PDF sourced; product posts verified in browser but local `curl` download blocked by Cloudflare 403 | PARTIAL | PARTIAL | TODO | Post-training |
| bcooler | DeepMind/Trane BCOOLER | HVAC control | Tier 2 candidate | arXiv:2211.07357 | DONE | DONE | TODO | Physical control |
| tmall-pricing | Alibaba/Tmall dynamic pricing field experiment | Pricing | Secondary field evidence | Liu 2019 | Existing source in ch04 | Existing markdown in ch04 | TODO | Marketplace/operations |
| hotel-rm | China Lodging Group hotel revenue management | Revenue management | Secondary field evidence | Chen 2023 | Existing source in ch04 | Existing markdown in ch04 | TODO | Operations appendix/background |
| execution | Adaptive financial order execution | Financial execution | Tier 3/background unless stronger evidence | Nevmyvaka 2006 | Existing source in ch04 | Existing markdown in ch04 | TODO | Background |
| facebook-abr | Facebook 360 video adaptive bitrate | Video serving | Tier 3 candidate | Horizon-related source | TODO | TODO | TODO | Footnote only unless verified |

## Target Chapter Outline

1. **Opening screen:** production RL is real but smaller and more conservative than survey language implies.
2. **Digital traffic:** recommenders, notifications, bidding, sponsored search, and cold-start ranking.
3. **Marketplaces and operations:** dispatch, pricing, replenishment, and revenue management.
4. **Physical and design systems:** codec control, chip floorplanning, and HVAC control.
5. **Post-training:** RLHF as deployed training-time RL, not online runtime RL.
6. **Lessons from the field:** the long synthesis, written only after source reading.

## Lessons Section Questions

Answer these from the sourced cases, not from prior intuition.

- How does field RL differ from theoretical RL?
- Which RL families have actually been useful in production?
- What is missing from public evidence?
- What has been tried but stayed at simulation, replay, or limited field-trial status?
- Which domain conditions make RL viable?
- What logging, evaluation, safety, monitoring, and organizational machinery is required?

## Acceptance Criteria

- Every core paper has a downloaded original and non-trivial markdown conversion; product posts that are blocked by site defenses are explicitly marked, not silently treated as sourced.
- Every Tier 1/Tier 2 classification is traceable to primary-source text.
- Every old applications-chapter section has a documented destination.
- The bus-engine simulation is no longer framed as field deployment.
- The manuscript builds with the new topology.
