# TD-Gammon: A Self-Teaching Backgammon Program Achieves Master-Level Play (Tesauro, 1994)

## The Problem (Layperson)

Can a computer learn to play a complex game at the level of human experts, starting from zero knowledge and learning entirely from self-play? Backgammon presents a formidable challenge: positions have billions of possible configurations, the optimal strategy depends on subtle positional features that take humans years to master, and the dice introduce randomness that makes evaluating positions difficult.

Previous game-playing programs relied on human expertise: hand-crafted evaluation functions encoding what experts knew about good positions. TD-Gammon asked whether a neural network could discover such knowledge on its own, using only the rules of the game and the feedback from winning or losing.

## What Didn't Work (Alternatives)

Supervised learning from expert games was the approach used in Neurogammon, Tesauro's earlier program. It learned to mimic expert moves but was limited by the quality and quantity of training data. Even with championship-level game records, Neurogammon played at only a strong intermediate level.

Minimax search with hand-crafted evaluation functions, successful in chess, was less effective in backgammon. The branching factor (due to dice) made deep search impractical, and the evaluation function required extensive expert knowledge to construct.

Random self-play with simple learning rules failed entirely. With random initial weights, the network's play was terrible, games lasted thousands of moves instead of the normal 50-60, and it seemed impossible for sensible learning to emerge from such chaos.

## The Key Insight

Temporal difference learning could train a neural network to evaluate backgammon positions by playing against itself. The key equation is:

$$V(s_t) \leftarrow V(s_t) + \alpha [V(s_{t+1}) - V(s_t)]$$

The network's evaluation of the current position moves toward its evaluation of the next position. No external teacher is needed: the network learns from the discrepancy between consecutive evaluations, using the final game outcome only to anchor the value of terminal states.

The insight that made TD-Gammon work was that random initial play, despite being terrible, still generated learning signal. As long as games eventually end (with wins and losses), TD learning propagates value information backward. The network gradually improves, its self-play generates better games, which generates better training data, creating a virtuous cycle.

## The Method

TD-Gammon used a multilayer neural network to evaluate positions. Input: raw board position (number of pieces at each point) plus optional hand-crafted features. Output: probability of winning for each player.

Training proceeded by self-play: the network played both sides, choosing moves that maximized its evaluation of the resulting position. After each move, backpropagation updated weights based on the TD error:

$$\Delta w = \alpha (V_{t+1} - V_t) \sum_{k=1}^{t} \lambda^{t-k} \nabla_w V_k$$

This is TD($\lambda$) learning, where eligibility traces allow credit assignment across multiple time steps. The gradient $\nabla_w V_k$ was computed by standard backpropagation.

Training involved hundreds of thousands of self-play games. Each version of TD-Gammon represented a checkpoint after a certain number of training games: 300,000 for version 1.0, 800,000 for version 2.0, 1,500,000 for version 2.1.

## The Result

TD-Gammon achieved world-class play. Version 2.1, with 1.5 million training games, achieved near-parity with Bill Robertie, one of the world's best players, in a 40-game match (losing by one point, 39-40).

Testing against expert humans showed steady improvement:
- Version 1.0 (300K games): -0.25 points per game vs. grandmasters
- Version 2.0 (800K games): -0.18 points per game
- Version 2.1 (1.5M games): -0.02 points per game

For comparison, the best previous programs lost around -0.5 to -1.0 points per game against top humans. TD-Gammon represented a quantum leap in playing strength.

Remarkably, the zero-knowledge version (raw board input only) achieved strong intermediate play, roughly equal to Neurogammon. Adding hand-crafted features pushed performance to grandmaster level.

## Worked Example

Consider a simplified endgame position where White has 2 pieces left and Black has 1. The network must learn that being ahead in the race is good.

Initial random weights: $V(\text{White ahead}) \approx V(\text{Black ahead}) \approx 0.5$

Game plays out. White's turn: rolls dice, moves pieces. Network evaluates new position.

After White's move: $V_{t+1} = 0.52$ (slightly higher, maybe White moved advantageously)
Before White's move: $V_t = 0.50$

TD error: $\delta = 0.52 - 0.50 = 0.02$

Weights update to make $V_t$ closer to $V_{t+1}$. The network learns that the pre-move position was slightly undervalued.

Eventually, White wins (game ends with reward 1 for White).
TD error at final position: $\delta = 1.0 - V_{t}$

Weights update strongly, increasing evaluations of positions that led to victory.

Over thousands of games, the network learns: fewer opponent pieces = higher value, more pieces in home board = higher value, etc. These discovered features approximate what human experts know.

## Subtleties

The success of TD-Gammon was unexpected and, in some ways, not fully explained. Random self-play generates terrible training data initially. Why does learning succeed rather than collapse?

One hypothesis: the stochastic dice provide natural exploration. Even if the evaluation function is wrong, dice rolls force the agent into diverse positions, preventing collapse to a single bad strategy. This "forced exploration" may be crucial to TD-Gammon's success.

The choice of $\lambda$ matters. TD-Gammon used $\lambda = 0.7$, allowing credit to propagate over multiple moves but not the entire game. Too small $\lambda$ learns slowly; too large $\lambda$ has high variance.

Network architecture was modest by modern standards: 40-80 hidden units. Larger networks consistently performed better, suggesting that capacity was a limiting factor. Modern deep networks might achieve even higher performance.

TD-Gammon discovered novel strategies that human experts subsequently adopted. The network's aggressive early play, initially dismissed by experts, was later recognized as theoretically superior. Machine learning contributed to human understanding of the game.

## Critical Debates

TD-Gammon's success did not generalize easily to other games. Attempts to apply TD learning to chess, Go, and checkers with self-play had mixed results. Why did backgammon work so well?

Hypotheses include:
1. Dice provide natural exploration
2. The evaluation function is smoother (small position changes cause small value changes)
3. The game tree is narrower due to dice constraints
4. Draw outcomes are rare, providing clear learning signal

The role of hand-crafted features remained debated. Zero-knowledge TD-Gammon reached intermediate play, but grandmaster play required human-engineered features. Was TD learning doing the hard work, or were the features?

TD-Gammon influenced the development of AlphaGo and AlphaZero, which combined TD learning with Monte Carlo tree search and convolutional neural networks. These successors achieved superhuman play in Go and chess, finally realizing the promise TD-Gammon hinted at.

The theoretical foundations remain shaky. TD learning with function approximation can diverge (the deadly triad). TD-Gammon's convergence was empirical, not guaranteed. Understanding why it worked is still an active research area.

## Key Quotes

"Despite starting from random initial weights (and hence random initial strategy), TD-Gammon achieves a surprisingly strong level of play." (Abstract)

"From an a priori point of view, this methodology appeared unlikely to produce any sensible learning, because random strategy is exceedingly bad, and because the games end up taking an incredibly long time." (p. 2)

"TD-Gammon is now probably as good at backgammon as the grandmaster chess machine Deep Thought is at chess." (p. 4, quoting Robertie)

## Citation

Tesauro, G. (1994). TD-Gammon, a self-teaching backgammon program, achieves master-level play. *Neural Computation*, 6(2), 215-219.

Note: This summary references the 1994 short paper. Extended technical details appear in Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM*, 38(3), 58-68.
