# Learning to Summarize with Human Feedback - Implementation Summary

## Citation
```bibtex
@article{stiennon2020learning,
  title={Learning to summarize with human feedback},
  author={Stiennon, Nisan and Ouyang, Long and Wu, Jeffrey and Ziegler, Daniel and Lowe, Ryan and Voss, Chelsea and Radford, Alec and Amodei, Dario and Christiano, Paul F},
  journal={NeurIPS},
  year={2020}
}
```

## Core Contribution

This paper demonstrates RLHF for text summarization, showing that human feedback can train models that produce summaries preferred over those from much larger supervised models. The key finding is that optimizing for human preferences produces qualitatively different and better summaries than optimizing for reference summary likelihood (MLE). The paper also introduces important practical techniques: reward model ensembling, KL penalty scheduling, and careful handling of length biases.

The approach trains on Reddit TL;DR posts and CNN/DailyMail articles, with extensive human evaluation showing preference for RLHF summaries over supervised baselines.

## Key Equations

### Equation 1: Reward Model Training
$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_0, y_1, b) \sim \mathcal{D}}\left[b \log \sigma(r_\phi(x, y_1) - r_\phi(x, y_0)) + (1-b) \log \sigma(r_\phi(x, y_0) - r_\phi(x, y_1))\right]
$$
**Notation:**
- $b \in \{0, 1\}$: binary label indicating which summary is preferred
- $x$: source document
- $y_0, y_1$: two candidate summaries

**Implementation notes:** Equivalent to Bradley-Terry. Can also use soft labels $b \in [0, 1]$.

### Equation 2: Policy Optimization Objective
$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]
$$
**Implementation notes:** Standard KL-regularized objective. $\pi_{ref}$ is the supervised baseline.

### Equation 3: Adaptive KL Coefficient
$$
\beta_{t+1} = \beta_t \cdot \exp\left(\alpha \cdot (D_{KL,t} - D_{KL,target})\right)
$$
**Notation:**
- $D_{KL,t}$: current KL divergence from reference
- $D_{KL,target}$: target KL (hyperparameter)
- $\alpha$: adaptation rate

**Implementation notes:** Automatically adjust $\beta$ to maintain target KL. Prevents reward hacking.

### Equation 4: Reward Model Ensemble
$$
\hat{r}(x, y) = \frac{1}{K} \sum_{k=1}^K r_{\phi_k}(x, y)
$$
**Implementation notes:** Ensemble of $K$ independently trained RMs. Also provides uncertainty: $\sigma^2 = \frac{1}{K}\sum_k (r_{\phi_k} - \hat{r})^2$.

### Equation 5: Length-Adjusted Comparison (Analysis)
$$
P[y_w \succ y_l | \text{len}(y_w), \text{len}(y_l)] = \sigma(r(y_w) - r(y_l) + \gamma(\text{len}(y_w) - \text{len}(y_l)))
$$
**Implementation notes:** Humans exhibit length bias. Can probe for and partially correct this.

## Testable Claims

### Claim 1: RLHF Outperforms Supervised Learning
**Statement:** RLHF summaries are preferred over supervised summaries even when supervised model is larger.
**Validation approach:** Human evaluation comparing RLHF vs MLE models of same/different sizes.
**Expected result:** 1.3B RLHF preferred over 6.7B supervised $> 55\%$ of time.

### Claim 2: Reward Model Correlates with Human Preference
**Statement:** Higher RM score predicts higher human preference.
**Validation approach:** Compute correlation between RM scores and human preference on held-out data.
**Expected result:** Spearman correlation $\rho > 0.6$.

### Claim 3: Adaptive KL Maintains Quality
**Statement:** Adaptive $\beta$ prevents degenerate outputs while maximizing reward.
**Validation approach:** Compare fixed vs adaptive $\beta$, measure output quality at matched reward.
**Expected result:** Adaptive produces more coherent outputs at same reward level.

### Claim 4: Longer Summaries are Not Always Better
**Statement:** Controlling for length, RLHF summaries still preferred.
**Validation approach:** Match summary lengths, compare human preference.
**Expected result:** Preference advantage persists after length matching.

## Algorithm Pseudocode

```
Algorithm: RLHF for Summarization
Input:
  - Pretrained LM π_0
  - Source documents D
  - Human comparators (or oracle)
  - Target KL: D_KL_target

Output: Trained policy π_θ

# Phase 1: Supervised Baseline
1. Fine-tune π_0 on reference summaries → π_ref

# Phase 2: Collect Comparisons
2. For i = 1 to N_comparisons:
3.   Sample document x from D
4.   Generate y_0, y_1 ~ π_ref(·|x) (or from policy pool)
5.   Collect human preference b ∈ {0, 1}
6.   Add (x, y_0, y_1, b) to comparison dataset

# Phase 3: Train Reward Model Ensemble
7. For k = 1 to K:
8.   Initialize r_φ_k from π_ref
9.   Train r_φ_k on comparison dataset (different random seed)

# Phase 4: RL Training
10. Initialize π_θ ← π_ref
11. Initialize β ← β_0 (initial KL coefficient)
12. For each iteration:
13.   Sample batch of documents x_1, ..., x_B
14.   Generate summaries y_i ~ π_θ(·|x_i)
15.   Compute ensemble reward: r_i = mean_k(r_φ_k(x_i, y_i))
16.   Compute KL: kl_i = KL(π_θ(·|x_i) || π_ref(·|x_i))
17.   Adjusted reward: r_adj_i = r_i - β * kl_i
18.   Update π_θ via PPO on r_adj
19.   # Adapt β
20.   avg_kl = mean(kl_i)
21.   β ← β * exp(α * (avg_kl - D_KL_target))

22. Return π_θ
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Initial KL coef | $\beta_0$ | 0.05 | [0.01, 0.1] | Starting value before adaptation |
| Target KL | $D_{KL,target}$ | 6 nats | [1, 20] | Per-token KL budget |
| Adaptation rate | $\alpha$ | 0.1 | [0.01, 0.5] | How fast β adjusts |
| RM ensemble size | $K$ | 4 | [2, 8] | More = more robust |
| Comparisons collected | $N$ | 60,000 | [10k, 200k] | Task-dependent |
| Summary max length | - | 48-128 tokens | - | Task-dependent |
| PPO batch size | - | 512-2048 | - | Large for stability |

## Connections to Other Papers
- **Christiano et al. 2017**: First RLHF paper; this applies to NLP
- **Ouyang et al. 2022**: Generalizes from summarization to instruction-following
- **Ziegler et al. 2019**: Earlier LM + human feedback work
- **Rafailov et al. 2023**: DPO could replace PPO stage here

## Simulation Validation Checklist
- [ ] Verify RM accuracy improves with more comparisons
- [ ] Test that ensemble outperforms single RM
- [ ] Confirm adaptive KL prevents reward hacking
- [ ] Measure length bias in human and model preferences
- [ ] Compare RLHF vs supervised at matched model sizes
- [ ] Validate that RM score predicts human preference
