# Mastering the Game of Go with Deep Neural Networks and Tree Search (Silver et al., 2016)

## The Problem (Layperson)

Go had been AI's "grand challenge" for decades. Unlike chess, where computers achieved superhuman play in 1997, Go resisted all attempts at computerization. The game's vast search space (approximately $10^{170}$ possible positions) and the difficulty of evaluating positions made traditional game-playing approaches ineffective.

In chess, computers could evaluate positions using hand-crafted rules about piece values, king safety, and pawn structure. In Go, position evaluation defied such formalization. The concepts that experts use (influence, thickness, eye shape) are holistic and contextual. A group of stones might be strong or weak depending on subtle interactions across the entire board.

## What Didn't Work (Alternatives)

Minimax search with alpha-beta pruning, successful in chess, was impractical for Go. The branching factor ($b \approx 250$ legal moves per position) and game length ($d \approx 150$ moves) made exhaustive search impossible. Even with aggressive pruning, the search tree was too large.

Monte Carlo tree search (MCTS) had achieved amateur-level play by using random simulations to estimate position values. But MCTS programs relied on hand-crafted patterns and heuristics for their rollout policies. Progress had stalled at strong amateur level; professional play seemed out of reach.

Previous neural network approaches predicted moves from game records but played at weak amateur level. The networks could mimic expert moves with modest accuracy but lacked the strategic depth for strong play.

## The Key Insight

The solution combined deep neural networks with Monte Carlo tree search, using networks for both move selection (policy) and position evaluation (value):

- A **policy network** $p_\sigma(a|s)$ predicts the probability of moves, trained first on human expert games and then improved by self-play reinforcement learning
- A **value network** $v_\theta(s)$ estimates the probability of winning from position $s$, trained on self-play games

These networks reduce both the breadth (by focusing search on promising moves) and depth (by evaluating positions without playing to game end) of the search tree.

The training pipeline proceeds in stages: supervised learning from human games establishes a foundation, then reinforcement learning from self-play improves beyond human play, and finally a value network is trained to predict self-play outcomes.

## The Method

**Stage 1: Supervised Learning of Policy Network**

Train a convolutional neural network to predict expert moves from 30 million positions from the KGS Go server:
$$\Delta\sigma = \frac{\alpha}{m} \sum_{k=1}^{m} \frac{\partial \log p_\sigma(a_k|s_k)}{\partial \sigma}$$

The network achieved 57% accuracy predicting expert moves, vastly exceeding prior methods (44.4%).

**Stage 2: Reinforcement Learning of Policy Network**

Initialize $\rho = \sigma$ and improve through self-play, using REINFORCE:
$$\Delta\rho = \frac{\alpha}{n} \sum_{i=1}^{n} \sum_{t=1}^{T_i} \frac{\partial \log p_\rho(a_t|s_t)}{\partial \rho} (z_t - v(s_t))$$

where $z_t = \pm 1$ indicates the game winner. The RL policy network won 80% against the SL network and 85% against the strongest open-source program Pachi (using no search at all).

**Stage 3: Value Network**

Train a network to predict game outcomes from self-play:
$$\Delta\theta = \frac{\alpha}{m} \sum_{k=1}^{m} (z_k - v_\theta(s_k)) \frac{\partial v_\theta(s_k)}{\partial \theta}$$

Critically, each training position is sampled from a different game to avoid overfitting to correlated positions within games.

**Stage 4: MCTS with Neural Networks**

During search, use the policy network to guide action selection and the value network (combined with rollout evaluation) to evaluate leaf positions:
$$V(s_L) = (1-\lambda)v_\theta(s_L) + \lambda z_L$$

where $z_L$ is the rollout outcome. The mixture ($\lambda = 0.5$) outperformed either component alone.

## The Result

AlphaGo defeated Fan Hui, the European Go champion, 5-0 in October 2015. This was the first time a computer program had defeated a human professional player without handicap in the full game of Go.

In an internal tournament, AlphaGo won 99.8% of games against other Go programs, including the strongest commercial and open-source programs. With four handicap stones, it still won 77-99% against these programs.

In March 2016, AlphaGo defeated Lee Sedol, holder of 18 world titles, 4-1 in a highly publicized match. This result was considered a landmark achievement in AI.

## Worked Example

Consider a mid-game position where Black must decide among several candidates: extend a group, invade opponent's territory, or strengthen a weak group.

**Without neural networks (traditional MCTS):**
- Generate random rollouts from each candidate
- After 100,000 simulations, moves that happen to lead to favorable random play get higher values
- But random play is poor, so evaluations are noisy

**With AlphaGo:**

Policy network evaluation:
- $p_\sigma(\text{extend}) = 0.35$
- $p_\sigma(\text{invade}) = 0.25$
- $p_\sigma(\text{strengthen}) = 0.20$
- ...other moves share remaining probability

The MCTS focuses on these high-probability moves, exploring the extend variation most heavily.

Value network evaluation at a leaf node:
- $v_\theta(s_{\text{after extend}}) = 0.62$ (62% win probability for Black)
- This evaluation uses no simulation, just a single network forward pass

Combined evaluation:
- $V(s) = 0.5 \cdot 0.62 + 0.5 \cdot z_{\text{rollout}}$
- If the rollout wins, $V(s) = 0.5 \cdot 0.62 + 0.5 \cdot 1.0 = 0.81$

The search tree grows toward promising moves, with accurate position evaluations guiding exploration. After 1,600 simulations per move (compared to 100,000 for traditional MCTS), AlphaGo selects the most-visited root action.

## Subtleties

The SL policy network was used for move selection in MCTS rather than the stronger RL policy network. This counterintuitive choice worked because supervised learning from diverse human games produces a broader distribution of reasonable moves, while RL optimizes for the single best move. Diversity helps exploration during search.

However, the value network trained from RL self-play outperformed one trained from SL games. The RL policy's value predictions were more accurate because RL optimizes for winning rather than move prediction.

The value network had to be trained on positions from separate games to avoid overfitting. Within a game, consecutive positions differ by only one stone but share the same outcome label. Training on correlated positions caused the network to memorize game outcomes rather than learn general position evaluation.

The rollout policy used fast, simple features (3x3 patterns) that could be computed incrementally, enabling thousands of rollouts per second. This complemented the slow but accurate value network.

Combining value network and rollout evaluation ($\lambda = 0.5$) outperformed either alone. The value network provided stable evaluation from limited computation; rollouts provided precise evaluation of tactical sequences.

## Critical Debates

The role of human knowledge: AlphaGo's foundation was supervised learning from human games. Critics questioned whether this was "true" AI or just sophisticated imitation. AlphaGo Zero (2017) would address this by learning from scratch.

Compute requirements: AlphaGo required 1,202 CPUs and 176 GPUs for the distributed version. This raised questions about the scalability and accessibility of such methods. Single-machine versions were substantially weaker.

Generalization: Would the techniques transfer to other domains? The specific architecture assumed a grid-structured board and perfect information. Extensions to other games and real-world problems remained to be demonstrated.

Understanding: AlphaGo's play often surprised human experts, revealing new strategies. But the neural networks were black boxes; why specific moves were chosen remained opaque. This interpretability gap concerned some researchers.

The match against Lee Sedol included one famous loss (Game 4), where Lee found a brilliant move that exploited a blind spot in AlphaGo's evaluation. This demonstrated that superhuman performance did not mean perfect play.

## Key Quotes

"We introduce a new approach to computer Go that uses 'value networks' to evaluate board positions and 'policy networks' to select moves." (Abstract)

"Without any lookahead search, the neural networks play Go at the level of state-of-the-art Monte Carlo tree search programs that simulate thousands of random games of self-play." (Abstract)

"This is the first time that a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away." (Abstract)

## Citation

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.
