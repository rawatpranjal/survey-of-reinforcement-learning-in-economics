# Fine-Tuning Language Models from Human Preferences - Implementation Summary

## Citation
```bibtex
@article{ziegler2019fine,
  title={Fine-tuning language models from human preferences},
  author={Ziegler, Daniel M and Stiennon, Nisan and Wu, Jeffrey and Brown, Tom B and Radford, Alec and Amodei, Dario and Christiano, Paul and Irving, Geoffrey},
  journal={arXiv preprint arXiv:1909.08593},
  year={2019}
}
```

## Core Contribution

This paper is one of the first to apply RLHF to language models, predating the summarization and InstructGPT work. It demonstrates the approach on four tasks: text continuation with positive sentiment, text continuation with specific topics, CNN/DailyMail summarization, and TL;DR summarization. The key contributions are practical techniques for training reward models and policies on LMs, including importance sampling for KL estimation and early stopping based on reward model validation.

The paper establishes that reward model training requires careful regularization (dropout, early stopping) to avoid overfitting to limited human feedback.

## Key Equations

### Equation 1: Reward Model Objective
$$
\mathcal{L}_{RM}(\phi) = -\sum_{(x, y_+, y_-)} \log \sigma(r_\phi(x, y_+) - r_\phi(x, y_-))
$$
**Notation:**
- $(y_+, y_-)$: preferred and dispreferred completions
- $r_\phi$: reward model (final layer scalar from fine-tuned LM)

**Implementation notes:** Standard Bradley-Terry. They use 10% dropout in reward model.

### Equation 2: KL-Regularized Policy Objective
$$
\mathcal{J}(\theta) = \mathbb{E}_{x, y \sim \pi_\theta}[r_\phi(x, y)] - \beta \mathbb{E}_{x}[D_{KL}[\pi_\theta(\cdot|x) \| \pi_0(\cdot|x)]]
$$
**Notation:**
- $\pi_0$: initial (pretrained) policy
- $\beta$: KL penalty coefficient

**Implementation notes:** KL estimated via samples, not computed exactly.

### Equation 3: Importance-Weighted KL Estimation
$$
\hat{D}_{KL} = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_0(y|x)}\right] \approx \frac{1}{N}\sum_{i=1}^N \log \frac{\pi_\theta(y_i|x)}{\pi_0(y_i|x)}
$$
**Implementation notes:** Sample from $\pi_\theta$, compute log-ratio. Simple and unbiased.

### Equation 4: Reward Baseline (Variance Reduction)
$$
A(x, y) = r_\phi(x, y) - b(x)
$$
where $b(x) = \mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)]$

**Implementation notes:** Baseline reduces gradient variance. Estimate via exponential moving average.

### Equation 5: Entropy Bonus
$$
\mathcal{J}_{ent}(\theta) = \mathcal{J}(\theta) + \alpha \mathbb{E}_{x}[H(\pi_\theta(\cdot|x))]
$$
**Notation:**
- $H(\cdot)$: entropy
- $\alpha$: entropy coefficient

**Implementation notes:** Encourages exploration and prevents collapse to deterministic policy.

## Testable Claims

### Claim 1: RLHF Improves Sentiment Control
**Statement:** RL-trained model generates more positive continuations than supervised model.
**Validation approach:** Measure sentiment classifier scores on continuations.
**Expected result:** RLHF continuations score $> 0.8$ positive vs $< 0.5$ for baseline.

### Claim 2: Reward Model Overfits with Small Data
**Statement:** Without regularization, RM overfits to small comparison datasets.
**Validation approach:** Track RM train/val accuracy with and without dropout.
**Expected result:** Dropout reduces val accuracy gap by $> 50\%$.

### Claim 3: KL Penalty Necessary for Coherence
**Statement:** Without KL penalty, generated text becomes incoherent.
**Validation approach:** Generate samples with $\beta = 0$, measure perplexity under base model.
**Expected result:** Perplexity increases $> 5\times$ without KL penalty.

### Claim 4: More Comparisons Improve RM
**Statement:** Reward model accuracy scales with number of comparisons.
**Validation approach:** Train RM with 1k, 5k, 20k, 50k comparisons, measure held-out accuracy.
**Expected result:** Log-linear improvement in accuracy.

## Algorithm Pseudocode

```
Algorithm: LM Fine-Tuning from Human Preferences (Ziegler et al.)
Input:
  - Pretrained LM π_0
  - Task prompts D
  - Comparison data collection budget N

Output: Trained policy π_θ

# Phase 1: Collect Initial Comparisons
1. For i = 1 to N:
2.   Sample prompt x from D
3.   Generate y_0, y_1 ~ π_0(·|x)  # or from current policy
4.   Collect human preference label
5.   Store comparison

# Phase 2: Train Reward Model
6. Initialize r_φ from π_0
7. Add dropout (p=0.1) to r_φ
8. Train r_φ with early stopping on validation loss
9. Save best r_φ

# Phase 3: Policy Training
10. Initialize π_θ ← π_0
11. For each iteration:
12.   Sample prompts x_1, ..., x_B from D
13.   Generate y_i ~ π_θ(·|x_i) for each prompt
14.   Compute rewards: r_i = r_φ(x_i, y_i)
15.   Compute KL: kl_i = log π_θ(y_i|x_i) - log π_0(y_i|x_i)
16.   Compute baseline: b = EMA of recent rewards
17.   Advantage: A_i = r_i - β * kl_i - b
18.   Policy gradient update on π_θ
19.   Optionally: collect more comparisons and retrain r_φ

20. Return π_θ
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| KL coefficient | $\beta$ | 0.1 to 0.5 | [0.01, 1.0] | Task-dependent |
| RM dropout | - | 0.1 | [0, 0.3] | Prevents overfitting |
| RM early stopping patience | - | 3 epochs | [1, 10] | On validation loss |
| Entropy bonus | $\alpha$ | 0.01 | [0, 0.1] | Often set to 0 |
| Batch size | - | 64-256 | - | Per-step samples |
| Learning rate | - | 1e-5 | [1e-6, 1e-4] | For both RM and policy |
| Total comparisons | $N$ | 5k-60k | [1k, 200k] | Task complexity dependent |

## Connections to Other Papers
- **Christiano et al. 2017**: Adapts their approach from control to language
- **Stiennon et al. 2020**: Builds on this work for summarization
- **Ouyang et al. 2022**: Scales to instruction-following
- **Rafailov et al. 2023**: Eliminates RL phase entirely with DPO

## Simulation Validation Checklist
- [ ] Verify RM accuracy improves with more data
- [ ] Confirm dropout prevents RM overfitting
- [ ] Test that KL penalty maintains coherence
- [ ] Compare policy performance vs RM accuracy
- [ ] Measure effect of entropy bonus on diversity
- [ ] Validate early stopping criterion for RM
