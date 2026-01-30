# A Theory of Pavlovian Conditioning: Variations in the Effectiveness of Reinforcement and Nonreinforcement (Rescorla & Wagner, 1972)

## The Problem

Classical conditioning theory held that associative strength grows whenever a conditioned stimulus (CS) is paired with an unconditioned stimulus (US). But Kamin's (1969) blocking experiments contradicted this: prior conditioning of stimulus A to asymptote prevented subsequent learning about stimulus X when the compound AX was reinforced. The CS-US pairing occurred, yet no association formed.

Blocking implies that learning depends on prediction error, not mere contiguity. When A already fully predicts the US, the US is expected, and no learning occurs. The theoretical challenge: formalize this error-driven learning in a model that quantitatively predicts acquisition, extinction, blocking, overshadowing, and conditioned inhibition.

## What Didn't Work (Alternatives)

Traditional contiguity-based theories held that learning depended simply on the pairing of conditioned and unconditioned stimuli. If stimulus and reward co-occurred, association should form. These theories could not explain why learning was blocked when the reward was already predicted by another stimulus. The reward still occurred, it still followed the new stimulus, yet no learning took place.

Attention-based theories proposed that animals stop attending to stimuli that are uninformative. But this merely redescribed the phenomenon without explaining the mechanism. How does an animal know which stimuli are informative before learning about them?

## The Key Insight

Learning is driven by prediction error: the discrepancy between what actually happens and what was expected. When an outcome is surprising (different from expectation), learning occurs. When an outcome is fully predicted, no learning occurs regardless of what stimuli are present. The key equation is:

$$\Delta V_A = \alpha_A \beta (\lambda - V_{AX})$$

The change in associative strength ($\Delta V_A$) of stimulus A depends on the discrepancy between the maximum associative strength the unconditioned stimulus can support ($\lambda$) and the current combined associative strength of all stimuli present ($V_{AX}$). Learning occurs when this discrepancy is nonzero.

The crucial insight is that learning about any single stimulus depends on the total associative strength of the entire compound. If other stimuli already predict the outcome, there is no prediction error, and no learning occurs.

## The Method

**Notation.** Let $V_A$ denote the associative strength of stimulus A. For a compound stimulus AX, the aggregate associative strength is the sum of components:

$$V_{AX} = V_A + V_X$$

Let $\lambda$ denote the asymptotic associative strength that the US can support (e.g., $\lambda = 1$ for a strong US, $\lambda = 0$ for no US).

**Learning rule.** On each trial, the change in associative strength for each present stimulus is:

$$\Delta V_A = \alpha_A \beta (\lambda - V_{AX})$$
$$\Delta V_X = \alpha_X \beta (\lambda - V_{AX})$$

where:
- $\alpha_A, \alpha_X \in (0,1]$ are salience parameters (stimulus-specific learning rates)
- $\beta \in (0,1]$ is a US intensity parameter
- $(\lambda - V_{AX})$ is the prediction error

**Reinforcement vs. nonreinforcement.** The US determines $\lambda$:
- Reinforcement (US present): $\lambda = \lambda_1 > 0$
- Nonreinforcement (US absent): $\lambda = \lambda_2 = 0$

**Single-stimulus acquisition.** With only stimulus A present, $V_{AX} = V_A$:

$$\Delta V_A = \alpha_A \beta (\lambda - V_A)$$

This is a linear difference equation with solution:

$$V_A^{(n)} = \lambda (1 - (1 - \alpha_A \beta)^n)$$

Associative strength approaches $\lambda$ exponentially.

**Blocking.** In Phase 1, A is conditioned to asymptote: $V_A \approx \lambda$. In Phase 2, compound AX is reinforced. The prediction error is:

$$\lambda - V_{AX} = \lambda - (V_A + V_X) = \lambda - (\lambda + 0) = 0$$

With zero prediction error, $\Delta V_X = 0$. Stimulus X acquires no associative strength.

**Overshadowing.** When AX is conditioned from scratch with $\alpha_A > \alpha_X$ (A is more salient), A gains associative strength faster. As $V_{AX} \to \lambda$, the remaining prediction error diminishes, leaving X with less than it would acquire alone.

**Conditioned inhibition.** Suppose A is conditioned ($V_A > 0$) and then AX is nonreinforced ($\lambda = 0$):

$$\Delta V_X = \alpha_X \beta (0 - V_{AX}) = -\alpha_X \beta V_{AX} < 0$$

Stimulus X acquires negative associative strength, becoming a conditioned inhibitor.

**Extinction.** Repeated nonreinforcement of A alone:

$$\Delta V_A = \alpha_A \beta (0 - V_A) = -\alpha_A \beta V_A$$

Associative strength decays exponentially toward zero.

## The Result

The model unified a wide range of conditioning phenomena:

Blocking: Prior conditioning of A to asymptote means $V_A \approx \lambda$, so $V_{AX} \approx \lambda$ even with X at zero, producing no prediction error and no learning about X.

Overshadowing: When AX is reinforced and A is more salient ($\alpha_A > \alpha_X$), A gains associative strength faster, leaving less discrepancy for X to capture.

Conditioned inhibition: If A predicts reward ($V_A > 0$) and AX is nonreinforced ($\lambda = 0$), then $V_{AX} > \lambda$, so X acquires negative associative strength to reduce the compound toward zero.

Extinction: Repeated nonreinforcement of A drives $V_A$ toward zero as the discrepancy $(\lambda - V_A) = (0 - V_A)$ is negative.

## Worked Example

Consider the blocking experiment:

Phase 1: Stimulus A is paired with shock until $V_A \approx 1.0$ (asymptote).

Phase 2: Compound AX is paired with shock. The prediction error is:
$$\delta = \lambda - V_{AX} = 1.0 - (1.0 + 0) = 0$$

With zero prediction error, $\Delta V_X = \alpha_X \beta (0) = 0$. Stimulus X gains no associative strength despite being paired with shock on every trial.

Now consider what happens if a more intense shock is used in Phase 2, supporting a higher asymptote $\lambda = 1.5$:
$$\delta = 1.5 - 1.0 = 0.5$$

Now $\Delta V_X > 0$, and X gains associative strength. The unexpected increase in shock intensity creates prediction error, enabling learning.

This explains Kamin's finding that blocking is eliminated when shock intensity is increased between phases.

## Subtleties

The sum rule for combining associative strengths ($V_{AX} = V_A + V_X$) is an assumption, not a derivation. It implies that associative strengths are additive and independent. Alternative rules (e.g., multiplicative combination) would yield different predictions.

The model allows negative associative strength, which corresponds to conditioned inhibition. A stimulus can actively predict the absence of reward, suppressing responses to other excitatory stimuli. This bidirectional nature of learning (both excitatory and inhibitory) is crucial for the model's explanatory power.

The model specifies learning rules but not performance rules. How associative strength translates into behavioral responses requires additional assumptions. The model predicts learning, not responding directly.

The model assumes that prediction error is computed instantly based on current associative strengths. This implies that animals have access to the summed associative strength of all present stimuli, which raises questions about the neural implementation.

## Critical Debates

The Rescorla-Wagner model assumes learning depends on aggregate prediction error computed at the time of the US. An alternative view, temporal difference learning, suggests that prediction errors are computed moment by moment as the trial unfolds. TD learning can account for timing effects that Rescorla-Wagner cannot explain.

The model treats stimuli as having fixed identities. Configural theories propose that compounds AX are represented as distinct entities from their components, allowing for more flexible pattern recognition at the cost of greater representational complexity.

The model's prediction that associative strength is bounded by $\lambda$ has been challenged. Under some conditions, associative strength appears to exceed asymptotic values, suggesting that the model's simple learning rule may require modification.

The deepest question is what the model tells us about the brain. Schultz and colleagues showed that dopamine neurons encode prediction error signals remarkably similar to those specified by Rescorla-Wagner and TD learning. This neural implementation gives the abstract model concrete biological meaning.

## Key Quotes

"The central notion suggested here can also be phrased in somewhat more cognitive terms. One version might read: organisms only learn when events violate their expectations." (p. 75)

"The effect of a reinforcement or nonreinforcement in changing the associative strength of a stimulus depends upon the existing associative strength, not only of that stimulus, but also of other stimuli concurrently present." (p. 72)

"It appears that the changes in associative strength of a stimulus as a result of a trial can be well-predicted from the composite strength resulting from all stimuli present on that trial." (p. 72)

## Citation

Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical Conditioning II: Current Research and Theory* (pp. 64-99). New York: Appleton-Century-Crofts.
