# Source Manifest: Reinforcement Learning in the Field

Status vocabulary:

- `TODO`: not yet sourced/read.
- `DOWNLOADED`: original source file exists.
- `MD_DONE`: markdown conversion exists and is non-trivial.
- `READ_DONE`: source has been read and summarized.
- `FAILED`: source or conversion failed; reason must be recorded.

| ID | System | Domain | Tier target | Primary source target | Original | Markdown | Read note | Prose home | Status |
|---|---|---|---|---|---|---|---|---|---|
| youtube-recs | YouTube Top-K off-policy REINFORCE recommender | Recommendation | Tier 1 | arXiv:1812.02353 | `papers/youtube-recs-top-k-off-policy-reinforce.pdf` | `papers/youtube-recs-top-k-off-policy-reinforce.md` | `papers/read_notes/youtube-recs.md` | Digital traffic | READ_DONE |
| meta-horizon | Facebook/Meta Horizon notification policies | Notifications | Tier 1 | arXiv:1811.00260 | `papers/meta-horizon-facebook-applied-rl-platform.pdf` | `papers/meta-horizon-facebook-applied-rl-platform.md` | `papers/read_notes/meta-horizon.md` | Digital traffic | READ_DONE |
| meta-bidding | Meta production bidding-policy optimization | Ads and bidding | Tier 1 | arXiv:2310.09426 | `papers/meta-production-bidding-offline-rl.pdf` | `papers/meta-production-bidding-offline-rl.md` | `papers/read_notes/meta-bidding.md` | Digital traffic | READ_DONE |
| alibaba-rtb | Alibaba sponsored-search real-time bidding | Ads and bidding | Tier 1 | arXiv:1803.00259 / KDD 2018 | `papers/alibaba-sponsored-search-rtb.pdf` | `papers/alibaba-sponsored-search-rtb.md` | `papers/read_notes/alibaba-rtb.md` | Digital traffic | READ_DONE |
| taobao-rltv | Alibaba/Taobao RL-LTV cold-start recommendation | Recommendation | Tier 1 | arXiv:2108.09141 | `papers/taobao-rltv-cold-start-recommendation.pdf` | `papers/taobao-rltv-cold-start-recommendation.md` | `papers/read_notes/taobao-rltv.md` | Digital traffic | READ_DONE |
| didi-dispatch | DiDi scalable RL dispatch | Marketplace dispatch | Tier 1 | arXiv:2202.05118 / KDD 2022 | `papers/didi-scalable-rl-dispatch.pdf` | `papers/didi-scalable-rl-dispatch.md` | `papers/read_notes/didi-dispatch.md` | Marketplace control | READ_DONE |
| didi-ride-hailing-dispatch | Ride-Hailing Order Dispatching at DiDi via RL | Marketplace dispatch | Tier 1 support | Qin et al. 2021 / INFORMS Journal on Applied Analytics | `papers/didi-ride-hailing-order-dispatching-rl.pdf` | `papers/didi-ride-hailing-order-dispatching-rl.md` | `papers/read_notes/didi-ride-hailing-order-dispatching-rl.md` | Marketplace control support | READ_DONE |
| didi-cvnet | CVNet multi-driver order dispatch | Marketplace dispatch | Supporting evidence | Tang et al. 2019 / arXiv:2106.04493 | `papers/didi-cvnet-multi-driver-order-dispatching.pdf` | `papers/didi-cvnet-multi-driver-order-dispatching.md` | `papers/read_notes/didi-cvnet-multi-driver-order-dispatching.md` | DiDi technical support | READ_DONE |
| didi-mean-field-ridesharing | Mean-field multi-agent ridesharing dispatch | Marketplace dispatch | Supporting/contrast evidence | Li et al. 2019 / arXiv:1901.11454 | `papers/didi-mean-field-ridesharing-dispatch.pdf` | `papers/didi-mean-field-ridesharing-dispatch.md` | `papers/read_notes/didi-mean-field-ridesharing-dispatch.md` | DiDi support/contrast | READ_DONE |
| lyft-rl-matching | A Better Match for Drivers and Riders: RL at Lyft | Marketplace dispatch | Follow-up/comparison | arXiv:2310.13810 / Lyft matching paper | `papers/lyft-driver-rider-rl-matching.pdf` | `papers/lyft-driver-rider-rl-matching.md` | `papers/read_notes/lyft-driver-rider-rl-matching.md` | Marketplace comparison | READ_DONE |
| deepstock | Alibaba/Tmall DeepStock replenishment | Inventory operations | Tier 1 | arXiv:2603.19621 | `papers/alibaba-deepstock-inventory.pdf` | `papers/alibaba-deepstock-inventory.md` | `papers/read_notes/deepstock.md` | Operations | READ_DONE |
| muzero-rc-paper | Google DeepMind/YouTube MuZero-RC paper | Video encoding | Tier 1 support | arXiv:2202.06626 | `papers/deepmind-youtube-muzero-rc-vp9.pdf` | `papers/deepmind-youtube-muzero-rc-vp9.md` | `papers/read_notes/muzero-rc-paper.md` | Physical/design systems | READ_DONE |
| muzero-rc-post | Google DeepMind/YouTube MuZero-RC official post | Video encoding | Tier 1 | DeepMind post | `papers/deepmind-muzero-real-world.html` | `papers/deepmind-muzero-real-world.md` | `papers/read_notes/muzero-rc-post.md` | Physical/design systems | READ_DONE |
| alphachip-nature | Google AlphaChip Nature paper | Chip floorplanning | Tier 1 design artifact | Nature 2021 | `papers/nature-alphachip-graph-placement.html` | `papers/nature-alphachip-graph-placement.md` | `papers/read_notes/alphachip-nature.md` | Physical/design systems | READ_DONE |
| alphachip-post | Google DeepMind AlphaChip official follow-up | Chip floorplanning | Tier 1 design artifact | DeepMind post | `papers/deepmind-alphachip-transformed.html` | `papers/deepmind-alphachip-transformed.md` | `papers/read_notes/alphachip-post.md` | Physical/design systems | READ_DONE |
| openai-instructgpt-paper | OpenAI InstructGPT RLHF paper | LLM post-training | Tier 1 training-time | Ouyang et al. 2022 official PDF | `papers/openai-instructgpt-paper.pdf` | `papers/openai-instructgpt-paper.md` | `papers/read_notes/openai-instructgpt-paper.md` | Post-training | READ_DONE |
| openai-product-posts | OpenAI InstructGPT/ChatGPT product posts | LLM post-training | Tier 1 candidate | OpenAI posts (via Wayback; live pages Cloudflare-403) | `papers/openai-instructgpt-post.html`, `papers/openai-chatgpt-post.html` | `papers/openai-instructgpt-post/*.md`, `papers/openai-chatgpt-post/*.md` | `papers/read_notes/openai-product-posts.md` | Post-training | READ_DONE |
| bcooler | DeepMind/Trane BCOOLER | HVAC control | Tier 2 | arXiv:2211.07357 | `papers/deepmind-trane-bcooler.pdf` | `papers/deepmind-trane-bcooler.md` | `papers/read_notes/bcooler.md` | Physical control | READ_DONE |
| tmall-pricing | Alibaba/Tmall dynamic pricing field experiment | Pricing | Tier 2 | Liu 2019 | `papers/tmall-dynamic-pricing-field-experiment.pdf` | `papers/tmall-dynamic-pricing-field-experiment.md` | `papers/read_notes/tmall-pricing.md` | Marketplace/operations | READ_DONE |
| hotel-rm | China Lodging Group hotel revenue management | Revenue management | Tier 2 | Chen 2023 | `papers/hotel-revenue-management-field-experiment.pdf` | `papers/hotel-revenue-management-field-experiment.md` | `papers/read_notes/hotel-rm.md` | Operations/revenue management | READ_DONE |
| execution | Adaptive financial order execution | Financial execution | Tier 3 | Nevmyvaka 2006 | `papers/financial-order-execution-q-learning.pdf` | `papers/financial-order-execution-q-learning.md` | `papers/read_notes/financial-execution.md` | Contrast/background | READ_DONE |
| rtb-simulator | Budget-constrained display-ad RTB simulator | Ads and bidding | Contrast | Wu 2018 | `papers/rtb-budget-constrained-bidding-simulator.pdf` | `papers/rtb-budget-constrained-bidding-simulator.md` | `papers/read_notes/rtb-simulator.md` | Contrast/background | READ_DONE |
| inventory-benchmark | DRL inventory benchmark | Inventory operations | Contrast | Gijsbrechts et al. 2022 | `papers/inventory-drl-benchmark.pdf` | `papers/inventory-drl-benchmark.md` | `papers/read_notes/inventory-benchmark.md` | Contrast/background | READ_DONE |
| facebook-abr | Facebook 360 video adaptive bitrate | Video serving | Tier 3 | Horizon-related source | `papers/meta-horizon-facebook-applied-rl-platform.pdf` | `papers/meta-horizon-facebook-applied-rl-platform.md` | `papers/read_notes/facebook-abr.md` | Footnote only unless verified | READ_DONE |

## Read Note Template

Use this template once a source is converted:

```markdown
## {system}

- Source:
- What RL controlled:
- RL method and training data:
- Deployment evidence:
- Metrics:
- Safety/constraints/integration:
- Tier decision:
- Prose implications:
```
