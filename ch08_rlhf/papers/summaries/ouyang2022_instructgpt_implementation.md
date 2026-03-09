# Training Language Models to Follow Instructions with Human Feedback - Implementation Summary

## Citation
```bibtex
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={NeurIPS},
  year={2022}
}
```

## Core Contribution

InstructGPT introduces the three-stage RLHF pipeline that became the standard for aligning large language models: (1) Supervised Fine-Tuning (SFT) on demonstration data, (2) Reward Model (RM) training on comparison data, and (3) Reinforcement Learning via PPO against the reward model. The paper demonstrates that a 1.3B parameter InstructGPT model is preferred over a 175B GPT-3 model, showing that alignment can compensate for scale.

A key contribution is the measurement of "alignment tax" - the tradeoff between alignment and capability on traditional NLP benchmarks - and techniques to minimize it through mixing pretraining gradients.

## Key Equations

### Equation 1: SFT Objective (Stage 1)
$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x, y^*) \sim \mathcal{D}_{demo}}\left[\log \pi_\theta(y^* | x)\right]
$$
**Notation:**
- $\mathcal{D}_{demo}$: demonstration dataset (human-written responses)
- $y^*$: target response
- $x$: prompt

**Implementation notes:** Standard language modeling cross-entropy loss.

### Equation 2: Reward Model Objective (Stage 2)
$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{comp}}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]
$$
**Notation:**
- $\mathcal{D}_{comp}$: comparison dataset
- $y_w, y_l$: preferred and dispreferred responses
- $r_\phi$: reward model (scalar output from LM + linear head)

**Implementation notes:** Same as Bradley-Terry loss. Train from SFT checkpoint, remove final unembedding layer, add scalar head.

### Equation 3: PPO Objective with KL Penalty (Stage 3)
$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[r_\phi(x, y) - \beta D_{KL}[\pi_\theta(y|x) \| \pi_{SFT}(y|x)]\right]
$$
**Implementation notes:** Reference policy is SFT model. KL penalty prevents reward hacking.

### Equation 4: Per-Token KL Penalty (Practical Form)
$$
r_{total}(x, y) = r_\phi(x, y) - \beta \sum_t \left[\log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{SFT}(y_t | x, y_{<t})\right]
$$
**Implementation notes:** KL is computed token-by-token for sequences. More stable than sequence-level KL.

### Equation 5: PPO Clipped Objective
$$
\mathcal{L}_{PPO}(\theta) = \mathbb{E}\left[\min\left(\frac{\pi_\theta(y|x)}{\pi_{old}(y|x)} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]
$$
**Notation:**
- $A$: advantage estimate (from GAE)
- $\epsilon$: clipping parameter (typically 0.2)
- $\pi_{old}$: policy from previous iteration

**Implementation notes:** Standard PPO with learned value function baseline.

### Equation 6: Pretraining Mix (Reduce Alignment Tax)
$$
\mathcal{L}_{total}(\theta) = \mathcal{L}_{PPO}(\theta) + \gamma \mathcal{L}_{pretrain}(\theta)
$$
**Notation:**
- $\gamma$: mixing coefficient
- $\mathcal{L}_{pretrain}$: language modeling loss on pretraining distribution

**Implementation notes:** Mixing pretraining gradients preserves capabilities during RL.

## Testable Claims

### Claim 1: Alignment Outweighs Scale
**Statement:** InstructGPT 1.3B is preferred over GPT-3 175B in human evaluations.
**Validation approach:** Compare win rates between aligned small model and unaligned large model.
**Expected result:** Aligned model wins $> 60\%$ of comparisons.

### Claim 2: Alignment Tax is Small with Mixing
**Statement:** Pretraining gradient mixing reduces performance degradation on NLP benchmarks.
**Validation approach:** Measure benchmark accuracy with and without pretraining mix.
**Expected result:** $< 5\%$ degradation with mixing vs $> 20\%$ without.

### Claim 3: KL Penalty Prevents Reward Hacking
**Statement:** Without KL penalty, policy finds degenerate high-reward outputs.
**Validation approach:** Train with and without KL penalty, inspect outputs qualitatively.
**Expected result:** KL-penalized outputs remain coherent; unpenalized outputs degenerate.

### Claim 4: RM Accuracy Predicts Policy Quality
**Statement:** Higher reward model accuracy leads to better final policy.
**Validation approach:** Train RMs to different accuracy levels, correlate with downstream preference win rate.
**Expected result:** Positive correlation $r > 0.8$ between RM accuracy and policy win rate.

### Claim 5: Three Stages are All Necessary
**Statement:** Skipping any stage degrades final performance.
**Validation approach:** Ablate each stage, measure human preference win rate.
**Expected result:** Full pipeline outperforms ablations by $> 10\%$.

## Algorithm Pseudocode

```
Algorithm: InstructGPT Three-Stage Training
Input:
  - Pretrained LM π_0
  - Demonstration data D_demo
  - Comparison data D_comp
  - Prompt distribution D_prompts

Output: Aligned policy π_RLHF

# Stage 1: Supervised Fine-Tuning
1. Initialize π_SFT ← π_0
2. For each epoch:
3.   For each (x, y*) in D_demo:
4.     L = -log π_SFT(y*|x)
5.     Update π_SFT via gradient descent on L
6. Save π_SFT

# Stage 2: Reward Model Training
7. Initialize r_φ ← π_SFT (with scalar head)
8. For each epoch:
9.   For each (x, y_w, y_l) in D_comp:
10.    L = -log σ(r_φ(x, y_w) - r_φ(x, y_l))
11.    Update r_φ via gradient descent on L
12. Save r_φ

# Stage 3: PPO Fine-Tuning
13. Initialize π_θ ← π_SFT, V_ω ← random
14. For each iteration:
15.   Sample batch of prompts x_1, ..., x_B from D_prompts
16.   Generate responses: y_i ~ π_θ(·|x_i)
17.   Compute rewards: r_i = r_φ(x_i, y_i)
18.   Compute KL penalty: kl_i = β * KL(π_θ || π_SFT)
19.   Adjusted reward: r_adj_i = r_i - kl_i
20.   Compute advantages using GAE with V_ω
21.   Update π_θ, V_ω using PPO objective
22. Return π_θ as π_RLHF
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| SFT epochs | - | 2-4 | [1, 10] | Small to avoid overfitting |
| SFT learning rate | - | 1e-5 | [5e-6, 5e-5] | Lower than pretraining |
| RM epochs | - | 1-2 | [1, 5] | Single pass often sufficient |
| RM learning rate | - | 1e-5 | [1e-6, 5e-5] | Similar to SFT |
| PPO learning rate | - | 1e-6 | [5e-7, 5e-6] | Much lower than SFT |
| KL coefficient | $\beta$ | 0.02 | [0.001, 0.1] | Critical hyperparameter |
| PPO clip | $\epsilon$ | 0.2 | [0.1, 0.3] | Standard PPO value |
| Batch size (PPO) | - | 512 | [64, 2048] | Large batches stabilize |
| PPO epochs per batch | - | 4 | [1, 10] | Multiple passes per batch |
| GAE lambda | $\lambda$ | 0.95 | [0.9, 1.0] | Advantage estimation |
| Pretraining mix | $\gamma$ | 0.1-0.5 | [0, 1] | Higher preserves capabilities |

## Connections to Other Papers
- **Christiano et al. 2017**: InstructGPT scales this framework to LMs
- **Rafailov et al. 2023**: DPO replaces Stage 2+3 with single objective
- **Stiennon et al. 2020**: Earlier LM RLHF, summarization-focused
- **Korbak et al. 2022**: Theoretical justification for KL penalty

## Simulation Validation Checklist
- [ ] Verify SFT improves instruction-following over base model
- [ ] Confirm RM accuracy increases with more comparison data
- [ ] Test that KL penalty prevents reward hacking
- [ ] Measure alignment tax with and without pretraining mix
- [ ] Compare reward-KL Pareto frontiers across β values
- [ ] Validate that full pipeline outperforms ablations
