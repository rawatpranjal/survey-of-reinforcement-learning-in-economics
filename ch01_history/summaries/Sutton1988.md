# Learning to Predict by the Methods of Temporal Differences (Sutton, 1988)

## The Problem

Multi-step prediction requires estimating the expected value of some outcome $z$ given a sequence of observations $x_1, x_2, \ldots, x_m$. The prediction at time $t$ is $P_t = P(x_t, w)$, parameterized by weights $w$. The outcome $z$ is revealed only at sequence end.

Standard supervised learning updates predictions toward observed targets: $\Delta w \propto (z - P_t) \nabla_w P_t$. But this requires waiting until sequence termination to observe $z$. For long sequences, this delays learning and produces high-variance updates because a single outcome $z$ provides the training signal for all $m$ predictions.

The core question: can we update predictions incrementally, using information from intermediate observations before the final outcome is known?

## What Didn't Work (Alternatives)

Supervised learning methods required explicit teacher signals at each time step. For prediction problems, the "correct" prediction at intermediate time steps is not available; only the final outcome is observed. This made standard supervised learning inapplicable.

Monte Carlo methods waited for the final outcome and then updated all predictions that preceded it. This worked but was slow: credit had to propagate backward through many time steps, and variance was high because a single outcome determined updates to many predictions.

Widrow-Hoff/LMS learning updated predictions toward observed outcomes. But in multi-step prediction, intermediate outcomes are not observed; only the next prediction is available. Standard error-correction methods could not be directly applied.

## The Key Insight

Temporal difference (TD) learning updates predictions based on the difference between successive predictions, not on the difference between predictions and outcomes. If my current prediction is $P_t$ and my next prediction is $P_{t+1}$, I update $P_t$ toward $P_{t+1}$:

$$P_t \leftarrow P_t + \alpha(P_{t+1} - P_t)$$

This seemingly circular update makes sense because later predictions incorporate more information than earlier ones. As time passes, predictions converge toward the actual outcome. By chaining updates, information propagates backward from the outcome through intermediate predictions without waiting for the final result.

The eligibility trace mechanism extends this idea to update all recent predictions based on the current TD error, with older predictions receiving smaller updates. This allows a single outcome to update an entire sequence of predictions in a principled way.

## The Method

Consider predicting outcome $z$ from observations $x_1, x_2, \ldots, x_m$ with $P_{m+1} = z$. The prediction at time $t$ is $P_t = w^\top x_t$ (linear in features).

**TD error.** The one-step temporal difference error is:

$$\delta_t = P_{t+1} - P_t$$

This measures how much the prediction changed from one step to the next. At the final step, $\delta_m = z - P_m$ compares prediction to actual outcome.

**Eligibility trace.** The eligibility trace accumulates a decaying record of which weights contributed to recent predictions:

$$e_t = \sum_{k=1}^{t} \lambda^{t-k} \nabla_w P_k = \lambda e_{t-1} + \nabla_w P_t$$

For linear predictions, $\nabla_w P_t = x_t$, so $e_t = \lambda e_{t-1} + x_t$.

**TD(λ) update (backward view).** The weight update at time $t$ is:

$$\Delta w_t = \alpha \delta_t e_t = \alpha (P_{t+1} - P_t) \sum_{k=1}^{t} \lambda^{t-k} x_k$$

**λ-return (forward view).** Define the $n$-step return as:

$$G_t^{(n)} = P_{t+1} + P_{t+2} + \cdots + P_{t+n-1} + P_{t+n} - (n-1)P_t$$

For prediction problems where intermediate rewards are zero, this simplifies. The **λ-return** is the exponentially weighted average of all $n$-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Forward-backward equivalence.** The total weight change over a complete episode is identical whether computed by the forward view (target each $P_t$ toward $G_t^\lambda$) or the backward view (accumulate eligibility traces and apply TD errors):

$$\sum_{t=1}^{m} \alpha (G_t^\lambda - P_t) \nabla_w P_t = \sum_{t=1}^{m} \alpha \delta_t e_t$$

**Special cases:**
- **TD(0):** $\lambda = 0$, so $e_t = x_t$. Updates based only on one-step TD error: $\Delta w_t = \alpha(P_{t+1} - P_t) x_t$
- **TD(1):** $\lambda = 1$, equivalent to Monte Carlo. All predictions updated toward final outcome $z$.

## The Result

Sutton demonstrated TD methods on a random walk prediction task. A particle starts in the center of 7 states and moves randomly left or right until reaching a terminal state. The goal is to predict the probability of reaching the right terminal state from each non-terminal state.

TD($\lambda$) with intermediate values of $\lambda$ learned faster than either extreme:
- TD(0) learned slowly because information propagated only one step at a time
- TD(1) learned slowly due to high variance (entire trajectory determines update)
- TD(0.3) to TD(0.7) achieved the best balance of bias and variance

With appropriate $\lambda$, TD methods converged faster and required less memory than supervised learning alternatives. The optimal $\lambda$ depended on the problem structure.

## Worked Example

Consider a simple 5-state random walk: A - B - C - D - E, with terminal states at both ends. Starting from state C, the particle moves left or right with equal probability. Reaching E yields reward 1; reaching A yields reward 0.

True probabilities: $V(B) = 0.25$, $V(C) = 0.5$, $V(D) = 0.75$

Initial estimates: $V(B) = V(C) = V(D) = 0.5$

Episode 1: C → D → E (outcome = 1)

TD(0) updates:
- At C→D: $\Delta V(C) = \alpha(V(D) - V(C)) = \alpha(0.5 - 0.5) = 0$ (no update)
- At D→E: $\Delta V(D) = \alpha(1 - 0.5) = 0.5\alpha$ (increases V(D))

After episode: $V(D) = 0.5 + 0.5\alpha$

Notice that $V(C)$ was not updated, even though the episode started from C and ended successfully. Information must propagate one step at a time.

TD(1) updates (equivalent to Monte Carlo):
- $\Delta V(C) = \alpha(1 - 0.5) = 0.5\alpha$
- $\Delta V(D) = \alpha(1 - 0.5) = 0.5\alpha$

Both states receive the same update toward the outcome. This is faster for this single episode but has higher variance across episodes.

TD($\lambda$ = 0.5) updates:
- $\Delta V(D) = \alpha(1 - 0.5) = 0.5\alpha$
- $\Delta V(C) = 0.5 \cdot \alpha(1 - 0.5) = 0.25\alpha$

State C receives a discounted update, balancing the benefits of both approaches.

## Subtleties

TD methods are not gradient descent on any fixed objective function. The target $P_{t+1}$ depends on the weights being trained, making this "semi-gradient" or "bootstrapping" learning. Despite this, TD methods converge under suitable conditions, as later theoretical work showed.

The relationship between TD($\lambda$) and Monte Carlo is not simply that TD(1) equals Monte Carlo. In TD(1), updates occur at each step but depend on future predictions; in Monte Carlo, a single update occurs at episode end. They are equivalent only after the complete episode is processed.

The eligibility trace has two interpretations:
1. Forward view: look ahead and weight future TD errors
2. Backward view: accumulate eligibility as states are visited, then distribute TD errors backward

The backward view enables online, incremental learning; the forward view provides theoretical clarity.

TD learning exploits the Markov property: the current state contains all information needed to predict the future. If the process is non-Markov, TD learning may converge to the wrong predictions. In practice, state representations are often designed to be approximately Markov.

## Critical Debates

TD versus Monte Carlo: TD methods have lower variance (update based on one-step differences rather than full returns) but potential bias (bootstrapping from potentially incorrect predictions). Monte Carlo is unbiased but high variance. The optimal choice depends on the problem.

Function approximation: Sutton's analysis assumed tabular representations. With function approximation, TD methods can diverge (the "deadly triad" when combined with off-policy learning and bootstrapping). This limitation would only be understood later.

Credit assignment: TD($\lambda$) provides one solution to temporal credit assignment, but the eligibility trace decays exponentially. For problems with long-delayed consequences, this may not assign credit appropriately.

Biological plausibility: TD learning resembles the dopamine prediction error signal discovered in neuroscience. Schultz et al. (1997) showed that dopamine neurons encode a TD-like error signal, suggesting TD might be implemented in biological brains.

## Key Quotes

"The basic operation of TD methods is to make the system's predictions better match other, more accurate predictions rather than waiting to match actual outcomes." (Section 1)

"The methods considered here can be directly related to the Widrow-Hoff or LMS learning rule in that they use the difference between two successive predictions as an error signal rather than the difference between the final, correct outcome and the current prediction." (Abstract)

"TD methods converge more rapidly than supervised-learning methods on certain problems." (Section 5)

## Citation

Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9-44.
