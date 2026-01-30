# Mastering the Game of Go without Human Knowledge (Silver et al., 2017)

## The Problem (Layperson)

AlphaGo defeated the world's best Go players but relied heavily on human knowledge: it was trained on millions of human expert games and used hand-crafted features describing Go concepts. Could a program achieve superhuman performance starting from nothing, learning entirely through self-play?

This question goes beyond Go. In many domains, human expertise is scarce, expensive, or potentially suboptimal. If AI systems can only match human knowledge, they are fundamentally limited. The goal was to demonstrate that a pure reinforcement learning approach, starting tabula rasa, could surpass human-trained systems.

## What Didn't Work (Alternatives)

AlphaGo's training pipeline had several limitations:

The initial policy network required millions of labeled expert moves. Collecting such data was expensive, and the network was bounded by the quality of human play.

Separate policy and value networks required training two architectures with different objectives. This was computationally expensive and potentially suboptimal.

Monte Carlo rollouts for position evaluation added computational overhead and required a separate fast rollout policy with hand-tuned features.

Hand-crafted input features (liberties, capture patterns, ladder detection) encoded human Go knowledge. While helpful, they constrained the system to human conceptualizations.

## The Key Insight

A single neural network can serve as both policy and value function, trained purely from self-play using MCTS as a policy improvement operator. The training loop is:

1. Play games using current network + MCTS
2. Train network to predict MCTS move probabilities and game outcomes
3. Repeat

MCTS improves upon the raw network policy by looking ahead. The network then learns to predict MCTS's improved policy, internalizing the search. This creates a virtuous cycle: better networks produce better MCTS policies, which train better networks.

No human data is needed. The network starts with random weights and learns everything from self-play: opening patterns, tactical sequences, positional evaluation, endgame technique. Within days, it surpasses thousands of years of accumulated human Go knowledge.

## The Method

**Neural Network Architecture**

A single network $f_\theta(s) = (p, v)$ outputs both move probabilities $p$ and position value $v$. The architecture uses residual blocks:
- Input: 17 binary planes (8 for current player's stones across 8 time steps, 8 for opponent, 1 for current color)
- Tower: 20 or 40 residual blocks with 256 filters each
- Policy head: convolution → fully connected → softmax over 362 moves
- Value head: convolution → fully connected → tanh scalar output

**Self-Play Training**

Games are generated using MCTS guided by the current network:
- At each position $s_t$, run MCTS with 1,600 simulations
- Select move proportional to visit counts: $a_t \sim \pi_t$ where $\pi_t(a) \propto N(s_t, a)^{1/\tau}$
- At game end, record outcome $z \in \{-1, +1\}$
- Store training examples $(s_t, \pi_t, z_t)$

**Neural Network Training**

Minimize combined loss:
$$\ell = (z - v)^2 - \pi^T \log p + c\|\theta\|^2$$

The network learns to predict both MCTS's improved policy (cross-entropy on $\pi$) and game outcomes (mean-squared error on $z$).

**MCTS Details**

Selection: $a_t = \arg\max_a[Q(s,a) + U(s,a)]$ where $U(s,a) = c_{puct} P(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}$

Evaluation: leaf nodes are evaluated by network forward pass, no rollouts.

Backup: $W(s,a) \leftarrow W(s,a) + v$, $Q(s,a) = W(s,a)/N(s,a)$

## The Result

AlphaGo Zero achieved superhuman performance after just 3 days of training, defeating AlphaGo Lee (which beat Lee Sedol) 100-0.

After 40 days of training, AlphaGo Zero defeated AlphaGo Master (which beat top professionals 60-0 in online play) 89-11.

Remarkably, the self-play trained network achieved higher move prediction accuracy on human professional games than a network trained directly on those games with supervised learning. AlphaGo Zero learned strategies that matched and exceeded human understanding.

The tabula rasa approach was not only simpler but substantially stronger. Training required only 4 TPUs on a single machine, compared to AlphaGo's distributed system with 176 GPUs.

## Worked Example

Training progression over 72 hours:

**Hour 0:** Random initialization. The network outputs uniform probabilities over moves. Self-play games are random and long (random moves until termination). Value predictions are meaningless.

**Hour 3:** Games still chaotic, but the network begins to recognize that capturing stones is good. Self-play games show greedy capturing behavior, like a human beginner.

**Hour 10:** Basic tactics emerge: atari, connections, eye formation. The network can evaluate simple life-and-death positions. Games show purposeful local fighting.

**Hour 24:** Whole-board concepts appear: territory, influence, invasion timing. AlphaGo Zero surpasses the supervised learning network trained on human games.

**Hour 36:** AlphaGo Zero defeats AlphaGo Lee (the version that beat Lee Sedol).

**Hour 72:** Sophisticated strategy: joseki (corner patterns), fuseki (opening strategy), ko fights. AlphaGo Zero has rediscovered human Go knowledge and discovered novel variations.

At each iteration:
1. Network generates self-play data
2. MCTS improves upon network policy (search probabilities $\pi$ differ from raw policy $p$)
3. Network trains to match MCTS probabilities, improving its policy
4. Better network → better MCTS → even better network

## Subtleties

Combining policy and value in a single network provided implicit regularization. The shared representation must support both move selection and position evaluation, forcing it to capture fundamental aspects of position rather than task-specific artifacts.

Residual networks were crucial for training depth. The 40-block network (79 parameterized layers) significantly outperformed shallower architectures. Skip connections enabled gradient flow through this deep network.

No rollouts were needed. The value network alone, trained on self-play outcomes, provided sufficient position evaluation. This simplified the system and eliminated the need for a hand-tuned rollout policy.

Temperature scheduling controlled exploration versus exploitation. Early in games ($t \leq 30$), moves were sampled proportionally to visit counts ($\tau = 1$). Later, moves were selected deterministically ($\tau \to 0$). This ensured diverse openings while playing optimally in critical positions.

Dirichlet noise at the root provided additional exploration: $P(s,a) = (1-\varepsilon)p_a + \varepsilon\eta_a$ where $\eta \sim \text{Dir}(0.03)$. This ensured all moves could be tried despite the network's strong prior.

## Critical Debates

Human knowledge as initialization: AlphaGo Zero demonstrated that human data was unnecessary, but was it actually harmful? Supervised learning provides faster initial progress; pure RL requires exploring from scratch. The answer depends on the domain and compute budget.

Compute requirements: Training AlphaGo Zero required substantial compute (4 TPUs for 40 days, generating 29 million self-play games). Whether tabula rasa learning is practical for domains where simulation is expensive remains questionable.

Transfer to other domains: Go has perfect information, deterministic dynamics, and cheap simulation. Extending to partial information, stochastic environments, or expensive real-world interactions introduces additional challenges.

Emergent knowledge: AlphaGo Zero discovered joseki that human professionals subsequently adopted. This suggests machine learning can contribute to human understanding, not just replicate it. But it also raises questions about whether the discovered strategies are truly optimal or merely locally optimal given the training procedure.

The broader question: what constitutes "no human knowledge"? AlphaGo Zero still used human-designed neural network architectures, hyperparameters chosen based on experience, and the structure of the Go board encoded in the input representation. Pure tabula rasa would require discovering even these design choices.

## Key Quotes

"Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100-0 against the previously published, champion-defeating AlphaGo." (Abstract)

"Surprisingly, AlphaGo Zero outperformed AlphaGo Lee after just 36 hours." (Section 2)

"AlphaGo Zero was able to rediscover much of this Go knowledge, as well as novel strategies that provide new insights into the oldest of games." (Conclusion)

## Citation

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
