# Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems (Barto, Sutton & Anderson, 1983)

## The Problem

Learning control from sparse, delayed reinforcement poses two distinct credit assignment problems. The **temporal credit assignment problem**: when a failure signal arrives after a sequence of actions, which actions caused the failure? The **structural credit assignment problem**: in a network of adaptive elements, which elements' parameters should change?

The pole-balancing task exemplifies these challenges. The system observes state $x(t) = (\theta, \dot{\theta}, x, \dot{x})$ (pole angle, angular velocity, cart position, cart velocity) and selects a binary action $y(t) \in \{-1, +1\}$ (push left or right). The only feedback is $r(t) = 0$ during balancing and $r(t) = -1$ at failure. The optimal control law is unknown and must be learned from experience.

## What Didn't Work (Alternatives)

Supervised learning methods like the Perceptron and ADALINE required a teacher to specify the correct output for every input. But for control problems, no teacher knows the correct action at each moment. The optimal control decision depends on future consequences that cannot be observed at decision time.

Standard adaptive control methods assumed known system dynamics or required continuous error signals measuring distance from a desired trajectory. The pole-balancing formulation provided neither: the dynamics were unknown, and the only feedback was a binary failure signal arriving after many decisions.

Previous neural network approaches used Hebbian learning (correlation-based) or clustering (unsupervised). Neither addressed the problem of learning from evaluative feedback to improve performance on externally defined tasks.

## The Key Insight

The solution combined two ideas: an associative search element (ASE) that learns stimulus-response associations through trial and error, and an adaptive critic element (ACE) that learns to predict future reinforcement and provides more informative feedback signals. Together, they solve the credit assignment problem.

The ASE addresses the immediate question: given the current state, which action should I take? It learns by trying random actions and remembering which ones preceded good outcomes. But "good outcomes" are measured by the ACE, not by external failure signals.

The ACE addresses the deeper question: how do I evaluate states before final outcomes are known? It learns to predict eventual reinforcement from each state. When the system moves from state A to state B, the ACE compares B's predicted value to A's predicted value. If B is better, the preceding action was good; if worse, it was bad. This provides moment-by-moment feedback even when external feedback comes only at episode end.

## The Method

The system comprises two adaptive elements: the **Associative Search Element (ASE)** learns a control policy; the **Adaptive Critic Element (ACE)** learns a value function and provides internal reinforcement.

**State representation.** The continuous state space is discretized into boxes. A decoder maps state $x(t)$ to a binary feature vector: $x_i(t) = 1$ if state is in box $i$, else 0. (Only one $x_i = 1$ at each time.)

### Associative Search Element (ASE)

**Output (stochastic policy).** The ASE produces a binary action:

$$y(t) = f\left(\sum_i w_i(t) x_i(t) + \text{noise}(t)\right)$$

where $f$ is a threshold function and noise provides exploration. In practice:

$$y(t) = \text{sign}\left(\sum_i w_i x_i + n(t)\right)$$

with $n(t)$ drawn from a zero-mean distribution.

**Eligibility trace.** The trace records which state-action pairs occurred:

$$e_i(t+1) = \delta e_i(t) + (1 - \delta) y(t) x_i(t)$$

where $\delta \in [0,1)$ is the trace decay rate. When $x_i(t) = 1$ and action $y(t)$ is taken, eligibility $e_i$ increases; otherwise it decays.

**Weight update.** Weights change in proportion to internal reinforcement and eligibility:

$$\Delta w_i(t) = \alpha \hat{r}(t) e_i(t)$$

where $\alpha > 0$ is the learning rate and $\hat{r}(t)$ is provided by the ACE.

### Adaptive Critic Element (ACE)

**Prediction.** The ACE predicts cumulative future reinforcement from each state:

$$p(t) = \sum_i v_i(t) x_i(t)$$

where $v_i$ is the predicted value of box $i$.

**Internal reinforcement (TD error).** The ACE provides the learning signal for the ASE:

$$\hat{r}(t) = r(t) + \gamma p(t) - p(t-1)$$

where:
- $r(t)$ is external reinforcement (0 during balancing, $-1$ at failure)
- $\gamma \in [0,1)$ is the discount factor
- $p(t) - p(t-1)$ compares current prediction to previous prediction

This is the temporal difference error. When $\hat{r}(t) > 0$, the transition was better than expected; when $\hat{r}(t) < 0$, worse.

**Trace for ACE.** The ACE maintains its own eligibility trace:

$$\bar{x}_i(t+1) = \lambda \bar{x}_i(t) + (1 - \lambda) x_i(t)$$

**ACE weight update:**

$$\Delta v_i(t) = \beta \hat{r}(t) \bar{x}_i(t)$$

where $\beta > 0$ is the ACE learning rate.

### Combined System

At each time step:
1. Observe state $x(t)$
2. ASE computes action $y(t)$ with exploration noise
3. Environment transitions to next state, provides $r(t)$
4. ACE computes TD error $\hat{r}(t)$
5. Both ASE and ACE update weights using their respective eligibility traces

## The Result

The ASE/ACE system learned to balance the pole far more effectively than the boxes system of Michie and Chambers, which lacked the adaptive critic. The boxes system learned only from failure signals, which became less frequent as learning progressed. The ACE provided learning signal on every time step by evaluating state transitions.

In simulations, the ASE/ACE system typically solved the problem (balancing for extended periods) within 100 trials, often requiring fewer than 50. The boxes system required many more trials and sometimes failed to converge.

The critical difference was that ACE-provided feedback enabled learning throughout each trial, not just at failure. As the system learned, it moved toward "safe" states and away from "dangerous" ones, receiving continuous guidance from the learned value function.

## Worked Example

Consider a simplified version with two states: "pole nearly vertical" (safe) and "pole tilted" (dangerous).

Initially, $p(\text{safe}) = 0$ and $p(\text{dangerous}) = 0$. No predictions.

Episode 1: Start in safe state. Random action moves system to dangerous state. Then to failure.
- At failure: $\hat{r} = -1 + 0 - 0 = -1$. Update $p(\text{dangerous})$ to predict failure.
- At transition to dangerous: $\hat{r} = 0 + \gamma p(\text{dangerous}) - p(\text{safe})$. Now negative because dangerous predicts failure.

Episode 2: Start in safe state.
- If action keeps us safe: $\hat{r} = 0 + \gamma p(\text{safe}) - p(\text{safe}) \approx 0$. Neutral.
- If action moves to dangerous: $\hat{r} = 0 + \gamma p(\text{dangerous}) - p(\text{safe}) < 0$. Punishment signal.

The ASE now receives punishment when moving toward danger, even before actual failure. It learns to avoid actions leading to dangerous states. The value predictions propagate backward from failure, creating a gradient that guides learning.

## Subtleties

The random noise in action selection is not a bug but a feature. It provides the exploration necessary to discover good actions. Without noise, the system would repeat the same actions indefinitely, unable to discover improvements. The noise implements trial-and-error search.

The eligibility trace decay rate $\delta$ controls the temporal extent of credit assignment. High $\delta$ means distant past decisions receive credit; low $\delta$ focuses credit on recent decisions. This parameter must be tuned to the problem's temporal structure.

The ACE's value predictions bootstrap on themselves: the target for learning includes the next state's predicted value, which is itself learned. This circular dependence works because the system starts from failure (whose value is known) and propagates backward. But it means convergence is not guaranteed in general.

The state representation matters crucially. Barto, Sutton, and Anderson used a hand-designed decoder that divided the continuous state space into discrete boxes. The learning algorithm cannot discover useful state representations; they must be provided.

## Critical Debates

The division between ASE and ACE anticipates the modern actor-critic architecture, but the relationship to policy gradient methods was not fully understood in 1983. The ASE performs a form of stochastic policy search, while the ACE implements temporal difference learning for value prediction.

The credit assignment solution via eligibility traces is one approach among several. Alternatives include full backpropagation through time (requires a model of dynamics) and Monte Carlo methods (wait for episode end). The trace-based approach offers a middle ground: it does not require a model but can learn from incomplete episodes.

The biological plausibility of these elements was a central motivation. The ACE's behavior resembles classical conditioning (learning to predict future outcomes from current stimuli), while the ASE resembles instrumental conditioning (learning which actions produce rewards). This suggested that neural circuits might implement similar computations.

The deeper question is whether complex adaptive behavior can emerge from networks of such elements. Barto et al. demonstrated that a single ASE/ACE pair could solve a non-trivial control problem, but extending to networks remained future work. The scaling properties of such systems are still not fully understood.

## Key Quotes

"The learning problems faced by adaptive elements that are components of adaptive networks are at least as difficult as this version of the pole-balancing problem." (Abstract)

"It is shown how a system consisting of two neuronlike adaptive elements can solve a difficult learning control problem... without explicitly computing gradient estimates or even storing information from which such estimates could be computed." (Abstract)

"The reinforcement feedback received by a network component at any time will generally depend upon factors other than its own action taken some fixed time earlier; it will additionally depend upon the actions of a large number of components taken at a variety of earlier times." (p. 835)

## Citation

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, 13(5), 834-846.
