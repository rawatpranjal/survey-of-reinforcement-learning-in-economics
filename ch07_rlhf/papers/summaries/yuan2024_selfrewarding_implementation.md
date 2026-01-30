# Self-Rewarding Language Models - Implementation Summary

## Citation
```bibtex
@article{yuan2024self,
  title={Self-rewarding language models},
  author={Yuan, Weizhe and Pang, Richard Yuanzhe and Cho, Kyunghyun and Sukhbaatar, Sainbayar and Xu, Jing and Weston, Jason},
  journal={arXiv preprint arXiv:2401.10020},
  year={2024}
}
```

## Core Contribution

This paper proposes a self-improvement loop where the language model itself serves as both the policy and the reward model. The model generates candidate responses, judges them (LLM-as-a-Judge), creates preference pairs from these judgments, and trains on them using DPO. The process iterates, with each iteration producing a stronger model that generates better training data for the next iteration. This eliminates the need for human preference data after initialization.

The key insight is that instruction-following capability includes the ability to judge response quality, so a sufficiently capable model can bootstrap its own improvement through self-generated preference data.

## Key Equations

### Equation 1: LLM-as-a-Judge Scoring
$$
s(x, y) = f_\theta(x, y, \text{prompt}_{judge})
$$
**Notation:**
- $f_\theta$: the language model used as judge
- $\text{prompt}_{judge}$: evaluation prompt template
- $s \in \{1, 2, 3, 4, 5\}$: quality score (or continuous)

**Implementation notes:** Parse score from model's text output. Use specific rubric in prompt.

### Equation 2: Preference Pair Construction
$$
(y_w, y_l) = \begin{cases}
(y_i, y_j) & \text{if } s(x, y_i) > s(x, y_j) \\
(y_j, y_i) & \text{if } s(x, y_j) > s(x, y_i)
\end{cases}
$$
**Implementation notes:** For each prompt $x$, generate multiple responses, score each, pair highest with others.

### Equation 3: DPO Training Objective
$$
\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$
**Implementation notes:** Standard DPO loss. Reference policy is the model from previous iteration.

### Equation 4: Iterative Self-Improvement
$$
\pi_\theta^{(t+1)} = \text{DPO}\left(\pi_\theta^{(t)}, \mathcal{D}^{(t)}\right)
$$
where $\mathcal{D}^{(t)} = \{(x, y_w, y_l) : y_w, y_l \sim \pi_\theta^{(t)}, s^{(t)}(x, y_w) > s^{(t)}(x, y_l)\}$

**Implementation notes:** Each iteration uses the current model to both generate and judge.

### Equation 5: Score Margin Filtering
$$
\mathcal{D}_{filtered}^{(t)} = \{(x, y_w, y_l) \in \mathcal{D}^{(t)} : s(x, y_w) - s(x, y_l) \geq \tau\}
$$
**Notation:**
- $\tau$: minimum score margin for inclusion

**Implementation notes:** Only include pairs with clear preference signal. Reduces noise.

## Testable Claims

### Claim 1: Iterative Improvement
**Statement:** Each iteration improves model performance on benchmarks.
**Validation approach:** Track MT-Bench/AlpacaEval scores across iterations.
**Expected result:** Monotonic improvement for 2-3 iterations before plateau.

### Claim 2: Self-Reward Accuracy Improves
**Statement:** The model's ability to judge improves alongside its generation ability.
**Validation approach:** Measure agreement with human preferences at each iteration.
**Expected result:** Judge accuracy increases $> 5\%$ per iteration.

### Claim 3: Score Margin Predicts Preference Quality
**Statement:** Higher score margins between $y_w$ and $y_l$ yield better training signal.
**Validation approach:** Train with different margin thresholds, compare final performance.
**Expected result:** Optimal threshold at $\tau \in [1, 2]$ for 5-point scale.

### Claim 4: Self-Training Outperforms Static Data
**Statement:** Iterative self-training outperforms training on fixed preference dataset.
**Validation approach:** Compare 3-iteration self-reward vs 1x training on 3x data.
**Expected result:** Self-reward achieves higher final performance.

### Claim 5: Diversity Maintained Across Iterations
**Statement:** Response diversity does not collapse during iterative training.
**Validation approach:** Measure distinct n-grams and entropy at each iteration.
**Expected result:** Diversity metrics remain within 80% of initial values.

## Algorithm Pseudocode

```
Algorithm: Self-Rewarding Language Model Training
Input:
  - Base instruction-tuned model M_0
  - Prompt dataset D
  - Number of iterations T
  - Responses per prompt K
  - DPO temperature β

Output: Improved model M_T

1. M ← M_0
2. For iteration t = 1 to T:
3.   # Generate training data
4.   P ← empty preference dataset
5.   For each prompt x in D:
6.     # Generate K candidate responses
7.     responses = [M.generate(x) for _ in range(K)]
8.
9.     # Score each response using M as judge
10.    scores = []
11.    For each y in responses:
12.      prompt_eval = format_judge_prompt(x, y)
13.      score = parse_score(M.generate(prompt_eval))
14.      scores.append(score)
15.
16.    # Create preference pairs from top and non-top responses
17.    best_idx = argmax(scores)
18.    y_w = responses[best_idx]
19.    For j in range(K) where j ≠ best_idx:
20.      if scores[best_idx] - scores[j] >= τ:
21.        P.add((x, y_w, responses[j]))
22.
23.  # Train with DPO
24.  M_ref ← copy(M)  # Reference for this iteration
25.  M ← train_DPO(M, M_ref, P, β)
26.
27. Return M
```

## Parameter Specifications

| Parameter | Symbol | Typical Value | Range | Notes |
|-----------|--------|---------------|-------|-------|
| Iterations | $T$ | 3 | [2, 5] | Diminishing returns after 3-4 |
| Responses per prompt | $K$ | 4 | [2, 8] | More = better pairs, higher cost |
| Score margin threshold | $\tau$ | 1.0 | [0, 2] | For 5-point scale |
| DPO temperature | $\beta$ | 0.1 | [0.05, 0.5] | Standard DPO value |
| Prompts per iteration | - | 20,000 | [5k, 100k] | Scales with compute budget |
| DPO learning rate | - | 5e-7 | [1e-7, 1e-6] | Low to prevent forgetting |
| DPO epochs | - | 1-2 | [1, 3] | Per iteration |

## Connections to Other Papers
- **Rafailov et al. 2023**: Uses DPO as the training algorithm
- **Ouyang et al. 2022**: Self-rewarding replaces human feedback collection
- **Constitutional AI (Bai et al. 2022)**: Related self-critique approach
- **SPIN (Chen et al. 2024)**: Another self-play improvement method

## Simulation Validation Checklist
- [ ] Verify iterative improvement on held-out benchmarks
- [ ] Measure judge accuracy agreement with humans across iterations
- [ ] Test score margin filtering effect on training quality
- [ ] Confirm diversity is maintained (not collapsed)
- [ ] Compare self-reward iterations vs more data (compute-matched)
- [ ] Analyze failure modes: when does iteration hurt?
