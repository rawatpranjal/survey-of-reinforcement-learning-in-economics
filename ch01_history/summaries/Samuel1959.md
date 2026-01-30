# Some Studies in Machine Learning Using the Game of Checkers (Samuel, 1959)

## The Problem (Layperson)

Can a computer program improve its own performance through experience? In 1959, this was a radical question. Computers were seen as devices that executed predetermined instructions; the idea that they could learn, that they could become better at a task than their programmers had explicitly made them, seemed almost fantastical.

Arthur Samuel chose checkers (draughts) as his domain. The game was complex enough to require genuine intelligence but simple enough that a program could play many games quickly. The goal was not merely to write a program that played checkers, but to write a program that learned to play checkers better through practice.

## What Didn't Work (Alternatives)

Exhaustive search was impractical. Checkers has approximately $5 \times 10^{20}$ possible positions. Even with the fastest computers, examining every possibility was impossible.

Hand-coded rules were limited by the programmer's knowledge. A program could only be as good as the rules its creator could articulate. Expert players often cannot explain why a position is good; their knowledge is intuitive and holistic.

Static evaluation without learning could achieve moderate play but had no mechanism for improvement. The program would play at a fixed level regardless of experience.

## The Key Insight

Samuel introduced two key mechanisms for learning:

**Rote learning**: Store positions encountered during play along with their evaluations. When a position is seen again, retrieve the stored value instead of computing it. Over many games, the program builds a library of evaluated positions.

**Generalization learning**: Adjust the weights in a scoring polynomial based on experience. The evaluation function $V(s) = \sum_i w_i \phi_i(s)$ combines features $\phi_i$ (like piece count, center control, advancement) with learned weights $w_i$. By adjusting weights to make the evaluation function more consistent with observed outcomes, the program improves.

The breakthrough was using the program's own play to generate training data. The program played against itself or against modified versions of itself, learning from the outcomes without requiring human supervision.

## The Method

**The Evaluation Function**

The score for a position is a polynomial combination of board features:
$$V = w_1\phi_1 + w_2\phi_2 + \cdots + w_n\phi_n$$

Samuel experimented with up to 38 features including:
- Piece advantage (number of pieces minus opponent's pieces)
- Advancement (how far pieces have progressed toward promotion)
- Center control (control of central board squares)
- Mobility (number of available moves)
- Threat (pieces that could be captured)
- Various patterns encoding common configurations

**Rote Learning**

Positions and their alpha-beta minimax evaluations are stored. When a position is encountered:
1. Check if it (or a symmetric equivalent) is in memory
2. If found, use the stored value
3. If not, compute the value and store it

Over time, the library grows, reducing computation and improving consistency.

**Generalization Learning**

During play, if the backed-up minimax value of a position differs from its polynomial evaluation, adjust the polynomial weights. Specifically:
1. Compare evaluation at ply $n$ with backed-up value from ply $n+k$
2. If they differ, the deeper search presumably provides a better estimate
3. Adjust polynomial weights to reduce the discrepancy

Samuel also used a "correlation" procedure that adjusted weights based on which features correlated with ultimate game outcomes.

**Self-Play**

The program played against a copy of itself or against a version with different parameters. This provided unlimited training data without requiring human opponents.

## The Result

After training, the program achieved strong amateur-level play. Samuel reported:

"The program beat a master-loss record holder at checkers... It has played hundreds of games against many opponents and has clearly improved its level of play."

The program could defeat its own creator and held its own against accomplished players. It was featured on television, demonstrating machine learning to the public.

Perhaps more significant than the playing strength was the demonstration that self-improvement through experience was possible. The program that emerged from training was substantially different from the initial version, having discovered effective feature weights through its own play.

## Worked Example

Consider learning the weight for "piece advantage" (number of pieces minus opponent's pieces).

Initial weight: $w_1 = 1.0$

Game position: Program has 8 pieces, opponent has 7 pieces. Feature value: $\phi_1 = +1$.

Current evaluation (using only this feature): $V = 1.0 \times 1 = 1.0$ (favors program)

After minimax search to depth 4, backed-up value: $V' = 0.3$ (still favors program, but less)

The deeper search revealed that the extra piece provides less advantage than the polynomial suggested (perhaps it's poorly positioned, or the opponent has a strong threat).

Weight adjustment: $w_1 \leftarrow w_1 + \alpha(V' - V)\phi_1 = 1.0 + 0.1 \times (-0.7) \times 1 = 0.93$

The piece advantage feature's weight decreases slightly, indicating that raw piece count is somewhat less important than originally assumed.

Over thousands of positions, such adjustments tune all weights simultaneously, discovering which features truly correlate with winning.

## Subtleties

Samuel's learning rule anticipated temporal difference learning, though the connection was not made explicit until Sutton's work decades later. The idea of adjusting predictions based on subsequent, presumably more accurate predictions is the core of TD learning.

The minimax structure of checkers provided natural training signals. The backed-up value from deeper search served as a "teacher" for shallower evaluation, similar to how TD(λ) uses successor state values as targets.

Book moves (openings) and endgame databases were incorporated, demonstrating that learning systems could be augmented with human knowledge. The learning mechanism adjusted play in the middle game, where rote memory of known positions was impractical.

The feature engineering was crucial. Samuel's choice of 38 features encoded significant human knowledge about checkers. The learning only adjusted weights; it could not discover new features. This limitation would persist in AI systems until deep learning enabled automatic feature discovery.

## Critical Debates

How much learning versus engineering? Samuel's program combined hand-crafted features with learned weights. Critics argued this was mostly engineering, with learning providing only fine-tuning. Supporters emphasized that the learned weights discovered knowledge the programmer could not articulate.

Generalization: The learned weights transferred across positions, but the rote learning was specific to encountered positions. How these two mechanisms interacted, and which provided more benefit, was debated.

The term "machine learning": Samuel popularized this phrase, but its meaning was contested. Did adjustment of numerical weights constitute genuine learning? The debate over what distinguishes "true" learning from sophisticated parameter fitting continues today.

Comparison to human learning: Samuel explicitly compared his program to human learners. But humans learn from far less experience and generalize more broadly. The program's learning, while real, was narrow compared to human cognitive flexibility.

Significance: Some viewed Samuel's work as a landmark demonstration that machines could learn. Others saw it as a successful engineering project that proved little about intelligence or learning in general.

## Key Quotes

"The studies reported here have been concerned with programming a digital computer to behave in a way which, if done by human beings or animals, would be described as involving the process of learning." (Introduction)

"Enough work has been done to verify the fact that a computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program." (Section 5)

"The program has now been made to play against itself, learning from its own experience." (Section 4)

## Citation

Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. *IBM Journal of Research and Development*, 3(3), 210-229.

Note: Samuel published a follow-up paper in 1967 ("Some Studies in Machine Learning Using the Game of Checkers II: Recent Progress") describing further developments.
