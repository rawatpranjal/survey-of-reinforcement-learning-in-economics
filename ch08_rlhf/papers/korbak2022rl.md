# RL with KL Penalties Is Better Viewed as Bayesian Inference

Tomasz Korbak, Ethan Perez, Christopher L. Buckley

Findings of EMNLP 2022. arXiv:2205.11275

## Abstract

This paper argues that KL-regularised reinforcement learning, widely used for fine-tuning large language models (LLMs) with human feedback, should be understood through a Bayesian lens. The standard RLHF objective maximizes expected reward while penalizing divergence from a reference policy. The authors show that this is mathematically equivalent to variational inference: the process approximates a Bayesian posterior that specifies how to update a prior LM to conform with evidence provided by the reward function.

## Key Result: The Bayesian Posterior Interpretation

The central contribution is the observation that the optimal policy under the KL-regularized objective has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

where:
- $\pi_{\text{ref}}$ is the reference (prior) policy (typically the SFT model)
- $r(x, y)$ is the reward function
- $\beta$ is the KL penalty weight (temperature)
- $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x, y)/\beta)$ is the partition function

This directly parallels Bayes' rule: the posterior is proportional to the prior times the likelihood, where $\exp(r(x,y)/\beta)$ plays the role of the likelihood.

## RLHF as Variational Inference

The KL-regularized RL objective:

$$J(\phi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\phi(\cdot|x)} [r(x, y)] - \beta \, D_{KL}(\pi_\phi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

is shown to be equivalent (up to constants) to the Evidence Lower Bound (ELBO):

$$J(\phi) = -\beta \, D_{KL}(\pi_\phi(\cdot|x) \| \pi^*(\cdot|x)) + \text{const}$$

Maximizing $J(\phi)$ is therefore equivalent to minimizing the KL divergence between the learned policy $\pi_\phi$ and the Bayesian posterior $\pi^*$. This means that PPO-based RLHF training is performing variational inference, seeking the best approximation to the ideal posterior within the parametric family of $\pi_\phi$.

## Implications

1. The KL penalty is not an ad-hoc regularizer but a fundamental component of Bayesian inference, controlling how much the posterior deviates from the prior.

2. The temperature $\beta$ controls the strength of the evidence (reward) relative to the prior: as $\beta \to \infty$, the posterior converges to the prior; as $\beta \to 0$, the posterior concentrates on the reward-maximizing response.

3. The framework suggests that RL may not be the most natural formalism for LM alignment. The problem is better characterized as posterior inference, which opens the door to alternative optimization methods beyond policy gradient.

4. Distribution collapse (reward hacking) can be understood as the posterior concentrating too sharply, which happens when $\beta$ is too small relative to the reward scale.

## Connection to DPO

The closed-form posterior is exactly the relationship exploited by Rafailov et al. (2023) in Direct Preference Optimization. DPO reparameterizes the reward in terms of the optimal policy and reference policy, then substitutes into the Bradley-Terry loss to eliminate the reward model entirely. The Korbak et al. posterior equation is the key identity that makes this reparameterization possible.

## Relation to This Chapter

The chapter's Equation (3) for the Bayesian posterior $\pi^*(y|s) \propto \pi^{SFT}(y|s) \exp(r_\theta(s,y)/\lambda_{KL})$ matches the paper's central result. The chapter uses $\lambda_{KL}$ where the paper uses $\beta$, as noted in the chapter's footnote on notation conventions.
