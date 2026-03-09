## Monte-Carlo tree search as regularized policy optimization

Jean-Bastien Grill * 1 Florent Altch´ e * 1 Yunhao Tang * 1 2 Thomas Hubert 3 Michal Valko 1 Ioannis Antonoglou 3 R´ emi Munos 1

## Abstract

The combination of Monte-Carlo tree search (MCTS) with deep reinforcement learning has led to significant advances in artificial intelligence. However, AlphaZero, the current stateof-the-art MCTS algorithm, still relies on handcrafted heuristics that are only partially understood. In this paper, we show that AlphaZero's search heuristics, along with other common ones such as UCT, are an approximation to the solution of a specific regularized policy optimization problem. With this insight, we propose a variant of AlphaZero which uses the exact solution to this policy optimization problem, and show experimentally that it reliably outperforms the original algorithm in multiple domains.

## 1. Introduction

Policy gradient is at the core of many state-of-the-art deep reinforcement learning (RL) algorithms. Among many successive improvements to the original algorithm (Sutton et al., 2000), regularized policy optimization encompasses a large family of such techniques. Among them trust region policy optimization is a prominent example (Schulman et al., 2015; 2017; Abdolmaleki et al., 2018; Song et al., 2019). These algorithmic enhancements have led to significant performance gains in various benchmark domains (Song et al., 2019).

As another successful RL framework, the AlphaZero family of algorithms (Silver et al., 2016; 2017b;a; Schrittwieser et al., 2019) have obtained groundbreaking results on challenging domains by combining classical deep learning (He et al., 2016) and RL (Williams, 1992) techniques with Monte-Carlo tree search (Kocsis and Szepesv´ ari, 2006). To search efficiently, the MCTS action selection criteria takes inspiration from bandits (Auer, 2002). Interestingly,

* Equal contribution 1 DeepMind, Paris, FR 2 Columbia University, New York, USA 3 DeepMind, London, UK. Correspondence to: Jean-Bastien Grill &lt; jbgrill@google.com &gt; .

Proceedings of the 37 th International Conference on Machine Learning , Vienna, Austria, PMLR 119, 2020. Copyright 2020 by the author(s).

AlphaZero employs an alternative handcrafted heuristic to achieve super-human performance on board games (Silver et al., 2016). Recent MCTS-based MuZero (Schrittwieser et al., 2019) has also led to state-of-the-art results in the Atari benchmarks (Bellemare et al., 2013).

Our main contribution is connecting MCTS algorithms, in particular the highly-successful AlphaZero, with MPO, a state-of-the-art model-free policy-optimization algorithm (Abdolmaleki et al., 2018). Specifically, we show that the empirical visit distribution of actions in AlphaZero's search procedure approximates the solution of a regularized policy-optimization objective. With this insight, our second contribution a modified version of AlphaZero that comes significant performance gains over the original algorithm, especially in cases where AlphaZero has been observed to fail, e.g., when per-search simulation budgets are low (Hamrick et al., 2020).

In Section 2, we briefly present MCTS with a focus on AlphaZero and provide a short summary of the model-free policy-optimization. In Section 3, we show that AlphaZero (and many other MCTS algorithms) computes approximate solutions to a family of regularized policy optimization problems. With this insight, Section 4 introduces a modified version of AlphaZero which leverages the benefits of the policy optimization formalism to improve upon the original algorithm. Finally, Section 5 shows that this modified algorithm outperforms AlphaZero on Atari games and continuous control tasks.

## 2. Background

Consider a standard RL setting tied to a Markov decision process (MDP) with state space X and action space A . At a discrete round t ≥ 0 , the agent in state x t ∈ X takes action a t ∈ A given a policy a t ∼ π ( ·| s t ) , receives reward r t , and transitions to a next state x t +1 ∼ p ( ·| x t , a t ) . The RL problem consists in finding a policy which maximizes the discounted cumulative return E π [ ∑ t ≥ 0 γ t r t ] for a discount factor γ ∈ (0 , 1) . To scale the method to large environments, we assume that the policy π θ ( a | x ) is parameterized by a neural network θ .

## 2.1. AlphaZero

We focus on the AlphaZero family, comprised of AlphaGo (Silver et al., 2016), AlphaGo Zero (Silver et al., 2017b), AlphaZero (Silver et al., 2017a), and MuZero (Schrittwieser et al., 2019), which are among the most successful algorithms in combining model-free and model-based RL. Although they make different assumptions, all of these methods share the same underlying search algorithm, which we refer to as AlphaZero for simplicity.

From a state x , AlphaZero uses MCTS (Browne et al., 2012) to compute an improved policy ˆ π ( ·| x ) at the root of the search tree from the prior distribution predicted by a policy network π θ ( ·| x ) 1 ; see Eq. 3 for the definition. This improved policy is then distilled back into π θ by updating θ as θ ← θ -η ∇ θ E x [ D (ˆ π ( ·| x ) , π θ ( ·| x ))] for a certain divergence D . In turn, the distilled parameterized policy π θ informs the next local search by predicting priors, further improving the local policy over successive iterations. Therefore, such an algorithmic procedure is a special case of generalized policy improvement (Sutton and Barto, 1998).

One of the main differences between AlphaZero and previous MCTS algorithms such as UCT (Kocsis and Szepesv´ ari, 2006) is the introduction of a learned prior π θ and value function v θ . Additionally, AlphaZero's search procedure applies the following action selection heuristic,

<!-- formula-not-decoded -->

where c is a numerical constant, 2 n ( x, a ) is the number of times that action a has been selected from state x during search, and Q ( x, a ) is an estimate of the Q-function for state-action pair ( x, a ) computed from search statistics and using v θ for bootstrapping.

Intuitively, this selection criteria balances exploration and exploitation, by selecting the most promising actions (high Q-value Q ( x, a ) and prior policy π θ ( a | x ) ) or actions that have rarely been explored (small visit count n ( x, a ) ). We denote by N sim the simulation budget, i.e., the search is run with N sim simulations. A more detailed presentation of AlphaZero is in Appendix A; for a full description of the algorithm, refer to Silver et al. (2017a).

## 2.2. Policy optimization

Policy optimization aims at finding a globally optimal policy π θ , generally using iterative updates. Each iteration

1 We note here that terminologies such as prior follow Silver et al. (2017a) and do not relate to concepts in Bayesian statistics.

2 Schrittwieser et al. (2019) uses a c that has a slow-varying dependency on ∑ b n ( x, b ) , which we omit here for simplicity, as it was the case of Silver et al. (2017a).

updates the current policy π θ by solving a local maximization problem of the form

<!-- formula-not-decoded -->

where Q π θ is an estimate of the Q-function, S is the |A| -dimensional simplex and R : S 2 → R a convex regularization term (Neu et al., 2017; Grill et al., 2019; Geist et al., 2019). Intuitively, Eq. 2 updates π θ to maximize the value Q T π θ y while constraining the update with a regularization term R ( y , π θ ) .

Without regularizations, i.e., R = 0 , Eq. 2 reduces to policy iteration (Sutton and Barto, 1998). When π θ is updated using a single gradient ascent step towards the solution of Eq. 2, instead of using the solution directly, the above formulation reduces to (regularized) policy gradient (Sutton et al., 2000; Levine, 2018).

Interestingly, the regularization term has been found to stabilize, and possibly to speed up the convergence of π θ . For instance, trust region policy search algorithms (TRPO, Schulman et al., 2015; MPO Abdolmaleki et al., 2018; VMPO, Song et al., 2019), set R to be the KL-divergence between consecutive policies KL[ y , π θ ] ; maximum entropy RL (Ziebart, 2010; Fox et al., 2015; O'Donoghue et al., 2016; Haarnoja et al., 2017) sets R to be the negative entropy of y to avoid collapsing to a deterministic policy.

## 3. MCTS as regularized policy optimization

In Section 2, we presented AlphaZero that relies on modelbased planning. We also presented policy optimization, a framework that has achieved good performance in modelfree RL. In this section, we establish our main claim namely that AlphaZero's action selection criteria can be interpreted as approximating the solution to a regularized policy-optimization objective.

## 3.1. Notation

First, let us define the empirical visit distribution ˆ π as

<!-- formula-not-decoded -->

Note that in Eq. 3, we consider an extra visit per action compared to the acting policy and distillation target in the original definition (Silver et al., 2016). This extra visit is introduced for convenience in the upcoming analysis (to avoid divisions by zero) and does not change the generality of our results.

We also define the multiplier λ N as

<!-- formula-not-decoded -->

where the shorthand notation n a is used for n ( x, a ) , and N ( x ) glyph[defines] ∑ b n b denotes the number of visits to x during search. With this notation, the action selection formula of Eq. 1 can be written as selecting the action a glyph[star] such that

<!-- formula-not-decoded -->

Note that in Eq. 5 and in the rest of the paper (unless otherwise specified), we use Q to denote the search Q-values, i.e., those estimated by the search algorithm as presented in Section 2.1. For more compact notation, we use bold fonts to denote vector quantities, with the convention that u v [ a ] = u [ a ] v [ a ] for two vectors u and v with the same dimension. Additionally, we omit the dependency of quantities on state x when the context is clear. In particular, we use q ∈ R |A| to denote the vector of search Q-function Q ( x, a ) such that q a = Q ( x, a ) . With this notation, we can rewrite the action selection formula of Eq. 5 simply as 3

<!-- formula-not-decoded -->

## 3.2. A related regularized policy optimization problem

We now define ¯ π as the solution to a regularized policy optimization problem; we will see in the next subsection that the visit distribution ˆ π is a good approximation of ¯ π .

Definition 1 ( ¯ π ) . Let ¯ π be the solution to the following objective

<!-- formula-not-decoded -->

where S is the |A|dimensional simplex and KL is the KL-divergence. 4

We can see from Eq. 2 and Definition 1 that ¯ π is the solution to a policy optimization problem where Q is set to the search Q-values, and the regularization term R is a reversed KLdivergence weighted by factor λ N .

In addition, note that ¯ π is as a smooth version of the arg max associated to the search Q-values q . In fact, ¯ π can be computed as (Appendix B.3 gives a detailed derivation of ¯ π )

<!-- formula-not-decoded -->

where α ∈ R is such that ¯ π is a proper probability vector. This is slightly different from the softmax distribution obtained with KL[ y , π θ ] , which is written as

<!-- formula-not-decoded -->

3 When the context is clear, we simplify for any x ∈ R |A| , that arg max [ x ] glyph[defines] arg max a { x [ a ] , a ∈ A} .

4 We apply the definition KL[ x , y ] glyph[defines] ∑ a x [ a ] log x [ a ] y [ a ] ·

Remark The factor λ N is a decreasing function of N . Asymptotically, λ N = ˜ O (1 / √ N ) . Therefore, the influence of the regularization term decreases as the number of simulation increases, which makes ¯ π rely increasingly more on search Q-values q and less on the policy prior π θ . As we explain next, λ N follows the design choice of AlphaZero, and may be justified by a similar choice done in bandits (Bubeck et al., 2012).

## 3.3. AlphaZero as policy optimization

We now analyze the action selection formula of AlphaZero (Eq. 1). Interestingly, we show that this formula, which was handcrafted 5 independently of the policy optimization research, turns out to result in a distribution ˆ π that closely relates to the policy optimization solution ¯ π .

The main formal claim of this section that AlphaZero's search policy ˆ π tracks the exact solution ¯ π of the regularized policy optimization problem of Definition 1. We show that Proposition 1 and Proposition 2 support this claim from two complementary perspectives.

First, with Proposition 1, we show that ˆ π approximately follows the gradient of the concave objective for which ¯ π is the optimum.

Proposition 1. For any action a ∈ A , visit count n ∈ R A , policy prior π θ &gt; 0 and Q-values q ,

<!-- formula-not-decoded -->

with a glyph[star] being the action realizing Eq. 1 as defined in Eq. 5 and ˆ π = (1 + n ) / ( |A| + ∑ b n b ) as defined in Eq. 3, is a function of the count vector extended to real values.

The only thing that the search algorithm eventually influences through the tree search is the visit count distribution. If we could do an infinitesimally small update, then the greedy update maximizing Eq. 8 would be in the direction of the partial derivative of Eq. 9. However, as we are restricted by a discrete update , then increasing the visit count as in Proposition 1 makes ˆ π track ¯ π . Below, we further characterize the selected action a glyph[star] and assume π θ &gt; 0 .

Proposition 2. The action a glyph[star] realizing Eq. 1 is such that

<!-- formula-not-decoded -->

To acquire intuition from Proposition 2, note that once a glyph[star] is selected, its count n a glyph[star] increases and so does the total count N . As a result, ˆ π ( a glyph[star] ) increases (in the order of O (1 /N ) ) and further approximates ¯ π ( a glyph[star] ) . As such, Proposition 2 shows that the action selection formula encourages

5 Nonetheless, this heuristic could be interpreted as loosely inspired by bandits (Rosin, 2011), but was adapted to accommodate a prior term π θ .

the shape of ˆ π to be close to that of ¯ π , until in the limit the two distributions coincide.

Note that Proposition 1 and Proposition 2 are a special case of a more general result that we formally prove in Appendix D.1. In this particular case, the proof relies on noticing that

<!-- formula-not-decoded -->

Then, since ∑ a ˆ π ( a ) = ∑ a ¯ π ( a ) and ˆ π &gt; 0 and ¯ π &gt; 0 , there exists at least one action for which 0 &lt; ˆ π ( a ) ≤ ¯ π ( a ) , i.e., 1 / ˆ π ( a ) -1 / ¯ π ( a ) ≥ 0 .

To state a formal statement on ˆ π approximating ¯ π , in Appendix D.3 we expand the conclusion under the assumption that ¯ π is a constant. In this case we can derive a bound for the convergence rate of these two distributions as N increases over the search,

<!-- formula-not-decoded -->

with O (1 /N ) matching the lowest possible approximation error (see Appendix D.3) among discrete distributions of the form ( k i /N ) i for k i ∈ N .

## 3.4. Generalization to common MCTS algorithms

Besides AlphaZero, UCT (Kocsis and Szepesv´ ari, 2006) is another heuristic with a selection criteria inspired by UCB, defined as

<!-- formula-not-decoded -->

Contrary to AlphaZero, the standard UCT formula does not involve a prior policy. In this section, we consider a slightly modified version of UCT with a (learned) prior π θ , as defined in Eq. 14. By setting the prior π θ to the uniform distribution, we recover the original UCT formula,

<!-- formula-not-decoded -->

Using the same reasoning as in Section 3.3, we now show that this modified UCT formula also tracks the solution to a regularized policy optimization problem, thus generalizing our result to commonly used MCTS algorithms.

First, we introduce ¯ π UCT, which is tracked by the UCT visit distribution, as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to AlphaZero, λ UCT N behaves 7 as ˜ O ( 1 / √ N ) and therefore the regularization gets weaker as N increases. We can also derive tracking properties between ¯ π UCT and the UCT empirical visit distribution ˆ π UCT as we did for AlphaZero in the previous section, with Proposition 3; as in the previous section, this is a special case of the general result with any f -divergence in Appendix D.1.

## Proposition 3. We have that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

To sum up, similar to the previous section, we show that UCT's search policy ˆ π UCT tracks the exact solution ¯ π UCT of the regularized policy optimization problem of Eq. 15.

## 4. Algorithmic benefits

In Section 3, we introduced a distribution ¯ π as the solution to a regularized policy optimization problem. We then showed that AlphaZero, along with general MCTS algorithms, select actions such that the empirical visit distribution ˆ π actively approximates ¯ π . Building on this insight, below we argue that ¯ π is preferable to ˆ π , and we propose three complementary algorithmic changes to AlphaZero.

## 4.1. Advantages of using ¯ π over ˆ π

MCTS algorithms produce Q-values as a by-product of the search procedure. However, MCTS does not directly use search Q-values to compute the policy, but instead uses the visit distribution ˆ π (search Q-values implicitly influence ˆ π by guiding the search). We postulate that this degrades the performance especially at low simulation budgets N sim for several reasons:

6 In particular D ( x, y ) ≥ 0 , D ( x, y ) = 0 = ⇒ x = y and D ( x, y ) is jointly convex in x and y (Csisz´ ar, 1964; Liese and Vajda, 2006).

7 We ignore logarithmic terms.

1. When a promising new (high-value) leaf is discovered, many additional simulations might be needed before this information is reflected in ˆ π ; since ¯ π is directly computed from Q-values, this information is updated instantly.
2. By definition (Eq. 3), ˆ π is the ratio of two integers and has limited expressiveness when N sim is low, which might lead to a sub-optimal policy; ¯ π does not have this constraint.
3. The prior π θ is trained against the target ˆ π , but the latter is only improved for actions that have been sampled at least once during search. Due to the deterministic action selection (Eq. 1), this may be problematic for certain actions that would require a large simulation budget to be sampled even once.

The above downsides cause MCTS to be highly sensitive to simulation budgets N sim . When N sim is high relative to the branching factor of the tree search, i.e., number of actions, MCTS algorithms such as AlphaZero perform well. However, this performance drops significantly when N sim is low as showed by Hamrick et al. (2020); see also e.g. , Figure 3.D. by Schrittwieser et al. (2019).

We illustrate the effect of simulation budgets in Figure 1, where x -axis shows the budgets N sim and y -axis shows the episodic performance of algorithms applying ˆ π vs. ¯ π ; see the details of these algorithms in the following sections. We see that ˆ π is highly sensitive to simulation budgets while ¯ π performs consistently well across all budget values.

## 4.2. Proposed improvements to AlphaZero

We have pointed out potential issues due to ˆ π . We now detail how to use ¯ π as a replacement to resolve such issues. 8 Appendix B.3 shows how to compute ¯ π in practice.

ACT: acting with ¯ π AlphaZero acts in the real environment by sampling actions according to a ∼ ˆ π ( ·| x root ) . Instead, we propose to to sample actions sampling according to a ∼ ¯ π ( ·| x root ) . We label this variant as ACT.

SEARCH: searching with ¯ π During search, we propose to stochastically sample actions according to ¯ π instead of the deterministic action selection rule of Eq. 1. At each node x in the tree, ¯ π ( · ) is computed with Q-values and total visit counts at the node based on Definition 1. We label this variant as SEARCH.

LEARN: learning with ¯ π AlphaZero computes locally improved policy with tree search and distills such improved

8 Recall that we have identified three issues. Each algorithmic variant below helps in addressing issue 1 and 2. Furthermore, the LEARN variant helps address issue 3.

<!-- image -->

⋃]⌉√⌈∐√]}{√(√˜√(√√˜√

Figure 1. Comparison of the score (median score over 3 seeds) of MuZero (red: using ˆ π ) and ALL (blue: using ¯ π ) after 100k learner steps as a function of N sim on Cheetah Run of the Control Suite.

policy into π θ . We propose to use ¯ π as the target policy in place of ˆ π to train our prior policy. As a result, the parameters are updated as

<!-- formula-not-decoded -->

where x root is sampled from a prioritized replay buffer as in AlphaZero. We label this variant as LEARN.

ALL: combining them all We refer to the combination of these three independent variants as ALL. Appendix B provides additional implementation details.

Remark Note that AlphaZero entangles search and learning, which is not desirable. For example, when the action selection formula changes, this impacts not only intermediate search results but also the root visit distribution ˆ π ( ·| x root ) , which is also the learning target for π θ . However, the LEARN variant partially disentangles these components. Indeed, the new learning target is ¯ π ( ·| x root ) which is computed from search Q-values, rendering it less sensitive to e.g., the action selection formula.

## 4.3. Connections between AlphaZero and model-free policy optimization.

Next, we make the explicit link between proposed algorithmic variants and existing policy optimization algorithms. First, we provide two complementary interpretations.

LEARN as policy optimization For this interpretation, we treat SEARCH as a blackbox, i.e., a subroutine that takes a root node x and returns statistics such as search Q-values.

Recall that policy optimization (Eq. 2) maximizes the objective ≈ Q T π θ y with the local policy y . There are many model-

free methods for the estimation of Q π θ , ranging from MonteCarlo estimates of cumulative returns Q π θ ≈ ∑ t ≥ 0 γ t r t (Schulman et al., 2015; 2017) to using predictions from a Q-value critic Q π θ ≈ q θ trained with off-policy samples (Abdolmaleki et al., 2018; Song et al., 2019). When solving ¯ π for the update (Eq. 17), we can interpret LEARN as a policy optimization algorithm using tree search to estimate Q π θ . Indeed, LEARN could be interpreted as building a Q-function 9 critic q θ with a tree-structured inductive bias. However, this inductive bias is not built-in a network architecture (Silver et al., 2017c; Farquhar et al., 2017; Oh et al., 2017; Guez et al., 2018), but constructed online by an algorithm, i.e., MCTS. Next, LEARN computes the locally optimal policy ¯ π to the regularized policy optimization objective and distills ¯ π into π θ . This is exactly the approach taken by MPO (Abdolmaleki et al., 2018).

SEARCH as policy optimization We now unpack the algorithmic procedure of the tree search, and show that it can also be interpreted as policy optimization.

During the forward simulation phase of SEARCH, the action at each node x is selected by sampling a ∼ ¯ π ( ·| x ) . As a result, the full imaginary trajectory is generated consistently according to policy ¯ π . During backward updates, each encountered node x receives a backup value from its child node, which is an exact estimate of Q ¯ π ( x, a ) . Finally, the local policy ¯ π ( ·| x ) is updated by solving the constrained optimization problem of Definition 1, leading to an improved policy over previous ¯ π ( ·| x ) . Overall, with N sim simulated trajectories, SEARCH optimizes the root policy ¯ π ( ·| x root ) and root search Q-values, by carrying out N sim sequences of MPO-style updates across the entire tree. 10 A highly related approach is to update local policies via policy gradients (Anthony et al., 2019).

By combining the above two interpretations, we see that the ALL variant is very similar to a full policy optimization algorithm. Specifically, on a high level, ALL carries out MPO updates with search Q-values. These search Q-values are also themselves obtained via MPO-style updates within the tree search. This paves the way to our major revelation stated next.

Observation 1. ALL can be interpreted as regularized policy optimization. Further, since ˆ π approximates ¯ π , AlphaZero and other MCTS algorithms can be interpreted as approximate regularized policy optimization.

9 During search, because child nodes have fewer simulations than the root, the Q-function estimate at the root slightly underestimates the acting policy Q-function.

10 Note that there are several differences from typical model-free implementations of policy optimization: most notably, unlike a fully-parameterized policy, the tree search policy is tabular at each node. This also entails that the MPO-style distillation is exact.

<!-- image -->

⊕˜∐√{˜√(√√˜√√

Figure 2. Comparison of median scores of MuZero (red) and ALL (blue) at N sim = 5 (dotted line) and N sim = 50 (solid line) simulations per step on Ms Pacman (Atari). Averaged across 8 seeds.

## 5. Experiments

In this section, we aim to address several questions: (1) How sensitive are state-of-the-art hybrid algorithms such as AlphaZero to low simulation budgets and can the ALL variant provide a more robust alternative? (2) What changes among ACT, SEARCH, and LEARN are most critical in this variant performance? (3) How does the performance of the ALL variant compare with AlphaZero in environments with large branching factors?

Baseline algorithm Throughout the experiments, we take MuZero (Schrittwieser et al., 2019) as the baseline algorithm. As a variant of AlphaZero, MuZero applies tree search in learned models instead of real environments, which makes it applicable to a wider range of problems. Since MuZero shares the same search procedure as AlphaGo, AlphaGo Zero, and AlphaZero, we expect the performance gains to be transferable to these algorithms. Note that the results below were obtained with a scaled-down version of MuZero, which is described in Appendix B.1.

Hyper-parameters The hyper-parameters of the algorithms are tuned to achieve the maximum possible performance for baseline MuZero on the Ms Pacman level of the Atari suite (Bellemare et al., 2013), and are identical in all experiments with the exception of the number of simulations per step N sim . 11 In particular, no further tuning was required for the LEARN, SEARCH, ACT, and ALL variants, as was expected from the theoretical considerations of Section 3.

11 The number of actors is scaled linearly with N sim to maintain the same total number of generated frames per second.

Figure 3. Comparison of median score (solid lines) over 6 seeds of MuZero and ALL on four Atari games with N sim = 50 . The shaded areas correspond to the range of the best and worst seed. ALL (blue) performs consistently better than MuZero (red).

<!-- image -->

## 5.1. Search with low simulation budgets

Since AlphaZero solely relies on the ˆ π for training targets, it may misbehave when simulation budgets N sim are low. On the other hand, our new algorithmic variants might perform better in this regime. To confirm these hypotheses, we compare the performance of MuZero and the ALL variant on the Ms Pacman level of the Atari suite at different levels of simulation budgets.

Result In Figure 2, we compare the episodic return of ALL vs. MuZero averaged over 8 seeds, with a simulation budget N sim = 5 and N sim = 50 for an action set of size |A| ≤ 18 ; thus, we consider that N sim = 5 and N sim = 50 respectively correspond to a low and high simulation budgets relative to the number of actions. We make several observations: (1) At a relatively high simulation budget, N sim = 50 , same as Schrittwieser et al. (2019), both MuZero and ALL exhibit reasonably close levels of performance; though ALL obtains marginally better performance than MuZero; (2) At low simulation budget, N sim = 5 , though both algorithms suffer in performance relative to high budgets, ALL significantly outperforms MuZero both in terms of learning speed and asymptotic performance; (3) Figure 6 in Appendix C.1 shows that this behavior is consistently observed at intermediate simulation budgets, with the two algorithms starting to reach comparable levels of performance when N sim ≥ 24 simulations. These observations confirm the intuitions from Section 3. (4) We provide results on a subset of Atari games in Figure 3, which show that the performance gains due to ¯ π over ˆ π are also observed in other levels than Ms Pacman; see

Figure 4. Ablation study at 5 and 50 simulations per step on Ms Pacman (Atari); average across 8 seeds.

<!-- image -->

Appendix C.2 for results on additional levels. This subset of levels are selected based on the experiment setup in Figure S1 of Schrittwieser et al. (2019). Importantly, note that the performance gains of ALL are consistently significant across selected levels, even at a higher simulation budget of N sim = 50 .

## 5.2. Ablation study

To better understand which component of the ALL contributes the most to the performance gains, Figure 4 presents the results of an ablation study where we compare individual component LEARN, SEARCH, or ACT.

Result The comparison is shown in Figure 4, we make several observations: (1) At N sim = 5 (Figure 4a), the main improvement comes from using the policy optimization solution ¯ π as the learning target (LEARN variant); using ¯ π during search or acting leads to an additional marginal improve-

ment; (2) Interestingly, we observe a different behavior at N sim = 50 (Figure 4b). In this case, using ¯ π for learning or acting does not lead to a noticeable improvement. However, the superior performance of ALL is mostly due to sampling according to ¯ π during search (SEARCH).

The improved performance when using ¯ π as the learning target (LEARN) illustrates the theoretical considerations of Section 3: at low simulation budgets, the discretization noise in ˆ π makes it a worse training target than ¯ π , but this advantage vanishes when the number of simulations per step increases. As predicted by the theoretical results of Section 3, learning and acting using ¯ π and ˆ π becomes equivalent when the simulation budget increases.

On the other hand, we see a slight but significant improvement when sampling the next node according to ¯ π during search (SEARCH) regardless of the simulation budget. This could be explained by the fact that even at high simulations budget, the SEARCH modification also affect deeper node that have less simulations.

## 5.3. Search with large action space - continuous control

The previous results confirm the intuitions presented in Sections 3 and 4; namely, the ALL variation greatly improves performance at low simulation budgets, and obtain marginally higher performance at high simulation budgets. Since simulation budgets are relative to the number of action, these improvements are critical in tasks with a high number of actions, where MuZero might require a prohibitively high simulation budgets; prior work (Dulac-Arnold et al., 2015; Metz et al., 2017; Van de Wiele et al., 2020) has already identified continuous control tasks as an interesting testbed.

Benchmarks We select high-dimensional environments from DeepMind Control Suite (Tassa et al., 2018). The observations are images and action space A = [ -1 , 1] m with m dimensions. We apply an action discretization method similar to that of Tang and Agrawal (2019). In short, for a continuous action space m dimensions, each dimension is discretized into K evenly spaced atomic actions. With proper parameterization of the policy network (see, e.g. , Appendix B.2), we can reduce the effective branching factor to Km glyph[lessmuch] K m , though this still results in a much larger action space than Atari benchmarks. In Appendix C.2, we provide additional descriptions of the tasks.

Result In Figure 5, we compare MuZero with the ALL variant on the CheetahRun environment of the DeepMind Control Suite (Tassa et al., 2018). We evaluate the performance at low ( N sim = 4 ), medium ( N sim = 12 ) and 'high' ( N sim = 50 ) simulation budgets, for an effective action space of size 30 ( m = 6 , K = 5 ). The horizontal line corresponds to the performance of model-free D4PG also

Figure 5. Comparison of the median score over 3 seeds of MuZero (red) and ALL (blue) at 4 (dotted) and 50 (solid line) simulations per step on Cheetah Run (Control Suite).

<!-- image -->

trained on pixel observations (Barth-Maron et al., 2018), as reported in (Tassa et al., 2018). Appendix C.2 provides experimental results on additional tasks. We again observe that ALL outperforms the original MuZero at low simulation budgets and still achieves faster convergence to the same asymptotic performance with more simulations. Figure 1 compares the asymptotic performance of MuZero and ALL as a function of the simulation budget at 100k learner steps.

Conclusion In this paper, we showed that the action selection formula used in MCTS algorithms, most notably AlphaZero, approximates the solution to a regularized policy optimization problem formulated with search Q-values. From this theoretical insight, we proposed variations of the original AlphaZero algorithm by explicitly using the exact policy optimization solution instead of the approximation. We show experimentally that these variants achieve much higher performance at low simulation budget, while also providing statistically significant improvements when this budget increases.

Our analysis on the behavior of model-based algorithms (i.e., MCTS) has made explicit connections to model-free algorithms. We hope that this sheds light on new ways of combining both paradigms and opens doors to future ideas and improvements.

Acknowledgements The authors would like to thank Alaa Saade, Bernardo Avila Pires, Bilal Piot, Corentin Tallec, Daniel Guo, David Silver, Eugene Tarassov, Florian Strub, Jessica Hamrick, Julian Schrittwieser, Katrina McKinney, Mohammad Gheshlaghi Azar, Nathalie Beauguerlange, Pierre M´ enard, Shantanu Thakoor, Th´ eophane Weber, Thomas Mesnard, Toby Pohlen and the DeepMind team.

## References

- Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N., and Riedmiller, M. (2018). Maximum a posteriori policy optimisation. arXiv preprint arXiv:1806.06920 .
- Andrychowicz, O. M., Baker, B., Chociej, M., Jozefowicz, R., McGrew, B., Pachocki, J., Petron, A., Plappert, M., Powell, G., Ray, A., et al. (2020). Learning dexterous inhand manipulation. The International Journal of Robotics Research , 39(1):3-20.
- Anthony, T., Nishihara, R., Moritz, P., Salimans, T., and Schulman, J. (2019). Policy gradient search: Online planning and expert iteration without search trees. arXiv preprint arXiv:1904.03646 .
- Auer, P. (2002). Using confidence bounds for exploitationexploration trade-offs. Journal of Machine Learning Research , 3(Nov):397-422.
- Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W., Horgan, D., TB, D., Muldal, A., Heess, N., and Lillicrap, T. (2018). Distributional policy gradients. In International Conference on Learning Representations .
- Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. (2013). The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47:253-279.
- Boyd, S. and Vandenberghe, L. (2004). Convex optimization . Cambridge university press.
- Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M., Cowling, P. I., Rohlfshagen, P., Tavener, S., Perez, D., Samothrakis, S., and Colton, S. (2012). A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games , 4(1):1-43.
- Bubeck, S., Cesa-Bianchi, N., et al. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends R © in Machine Learning , 5(1):1-122.
- Csisz´ ar, I. (1964). Eine informationstheoretische ungleichung und ihre anwendung auf beweis der ergodizitaet von markoffschen ketten. Magyer Tud. Akad. Mat. Kutato Int. Koezl. , 8:85-108.
- Dhariwal, P., Hesse, C., Klimov, O., Nichol, A., Plappert, M., Radford, A., Schulman, J., Sidor, S., Wu, Y., and Zhokhov, P. (2017). Openai baselines.
- Dulac-Arnold, G., Evans, R., van Hasselt, H., Sunehag, P., Lillicrap, T., Hunt, J., Mann, T., Weber, T., Degris, T., and Coppin, B. (2015). Deep reinforcement learning in large discrete action spaces. arXiv preprint arXiv:1512.07679 .
- Farquhar, G., Rockt¨ aschel, T., Igl, M., and Whiteson, S. (2017). TreeQN and ATreeC: Differentiable treestructured models for deep reinforcement learning. arXiv preprint arXiv:1710.11417 .
- Fox, R., Pakman, A., and Tishby, N. (2015). Taming the noise in reinforcement learning via soft updates. arXiv preprint arXiv:1512.08562 .
- Geist, M., Scherrer, B., and Pietquin, O. (2019). A theory of regularized markov decision processes. arXiv preprint arXiv:1901.11275 .
- Google (2020). Cloud TPU - Google Cloud. https://cloud.google.com/tpu/.
- Grill, J.-B., Domingues, O. D., M´ enard, P., Munos, R., and Valko, M. (2019). Planning in entropy-regularized Markov decision processes and games. In Neural Information Processing Systems .
- Guez, A., Weber, T., Antonoglou, I., Simonyan, K., Vinyals, O., Wierstra, D., Munos, R., and Silver, D. (2018). Learning to search with mctsnets. arXiv preprint arXiv:1802.04697 .
- Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. (2017). Reinforcement learning with deep energy-based policies. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 1352-1361. JMLR. org.
- Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Pfaff, T., Weber, T., Buesing, L., and Battaglia, P. W. (2020). Combining Q-learning and search with amortized value estimates. In International Conference on Learning Representations .
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778.
- Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel, M., van Hasselt, H., and Silver, D. (2018). Distributed prioritized experience replay. In International Conference on Learning Representations .
- Kocsis, L. and Szepesv´ ari, C. (2006). Bandit based MonteCarlo planning. In European conference on machine learning , pages 282-293. Springer.
- Levine, S. (2018). Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909 .
- Liese, F. and Vajda, I. (2006). On divergences and informations in statistics and information theory. IEEE Transactions on Information Theory , 52(10):4394-4412.

- Metz, L., Ibarz, J., Jaitly, N., and Davidson, J. (2017). Discrete sequential prediction of continuous actions for deep rl. arXiv preprint arXiv:1705.05035 .
- Neu, G., Jonsson, A., and G´ omez, V. (2017). A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 .
- O'Donoghue, B., Munos, R., Kavukcuoglu, K., and Mnih, V. (2016). Combining policy gradient and Q-learning. arXiv preprint arXiv:1611.01626 .
- Oh, J., Singh, S., and Lee, H. (2017). Value prediction network. In Advances in Neural Information Processing Systems , pages 6118-6128.
- Rosin, C. D. (2011). Multi-armed bandits with episode context. Annals of Mathematics and Artificial Intelligence , 61(3):203-230.
- Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al. (2019). Mastering Atari, go, chess and shogi by planning with a learned model. arXiv preprint arXiv:1911.08265 .
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning , pages 1889-1897.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. (2016). Mastering the game of go with deep neural networks and tree search. nature , 529(7587):484.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., et al. (2017a). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815 .
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017b). Mastering the game of go without human knowledge. Nature , 550(7676):354-359.
- Silver, D., van Hasselt, H., Hessel, M., Schaul, T., Guez, A., Harley, T., Dulac-Arnold, G., Reichert, D., Rabinowitz, N., Barreto, A., et al. (2017c). The predictron: End-toend learning and planning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 3191-3199. JMLR. org.
- Song, H. F., Abdolmaleki, A., Springenberg, J. T., Clark, A., Soyer, H., Rae, J. W., Noury, S., Ahuja, A., Liu, S., Tirumala, D., et al. (2019). V-MPO: On-policy maximum a posteriori policy optimization for discrete and continuous control. arXiv preprint arXiv:1909.12238 .
- Sutton, R. S. and Barto, A. G. (1998). Reinforcement Learning: An Introduction . MIT Press.
- Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , pages 1057-1063.
- Tang, Y. and Agrawal, S. (2019). Discretizing continuous action space for on-policy optimization. arXiv preprint arXiv:1901.10500 .
- Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D. d. L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq, A., et al. (2018). DeepMind control suite. arXiv preprint arXiv:1801.00690 .
- Van de Wiele, T., Warde-Farley, D., Mnih, A., and Mnih, V. (2020). Q-learning in enormous action spaces via amortized approximate maximization. arXiv preprint arXiv:2001.08116 .
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256.
- Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy . PhD thesis, Carnegie Mellon University, USA.

## A. Details on search for AlphaZero

Below we briefly present details of the search procedure for AlphaZero. Please refer to the original work (Silver et al., 2017a) for more comprehensive explanations.

As explained in the main text, the search procedure starts with a MDP state x 0 , which is used as the root node of the tree. The rest of this tree is progressively built as more simulations are generated. In addition to Q-function Q ( x, a ) , prior π θ ( x, a ) and visit counts n ( x, a ) , each node also maintains a reward R ( x, a ) = r ( x, a ) and value V ( x ) estimate.

In each simulation, the search consists of several parts: Selection , Expansion and Backup , as below.

Selection. From the root node x 0 , the search traverses the tree using the action selection formula of Eq. 1 until a leaf node x l is reached.

Expansion. After a leaf node x l is reached, the search selects an action from the leaf node, generates the corresponding child node x c and appends it to the tree T . The statistics for the new node are then initialized to Q ( x c , a ) = min x ∈T ,a ′ ∈A Q ( x, a ′ ) (pessimistic initialization), n ( x, a ) = 0 for ∀ a ∈ A .

Back-up. The back-up consists of updating statistics of nodes encountered during the forward traversal. Statistics that need updating include the Q-function Q ( x, a ) , count n ( x, a ) and value V ( x ) . The newly expanded node n c updates its value V ( x ) to be either the Monte-Carlo estimation from random rollouts (e.g. board games) or a prediction of the value network (e.g. Atari games). For the other nodes encountered during the forward traversal, all other statistics are updated as follows:

<!-- formula-not-decoded -->

where child ( x, a ) refers to the child node obtained by taking action a from node x .

Note that, in order to make search parameters agnostic to the scale of the numerical rewards (and, therefore, values), Q-function statistics Q ( x, a ) are always normalized by statistics in the search tree before applying the action selection formula; in practice, Eq. 1 uses the normalized Q z ( x, a ) defined as:

<!-- formula-not-decoded -->

## B. Implementation details

## B.1. Agent

For ease of implementation and availability of computational resources, the experimental results from Section 5 were obtained with a scaled-down version of MuZero (Schrittwieser et al., 2019). In particular, our implementation uses smaller networks compared to the architecture described in Appendix F of (Schrittwieser et al., 2019): we use only 5 residual blocks with 128 hidden layers for the dynamics function, and the residual blocks in the representation functions have half the number of channels. Furthermore, we use a stack of only 4 past observations instead of 32. Additionally, some algorithmic refinements (such as those described in Appendix H of (Schrittwieser et al., 2019)) have not been implemented in the version that we use in this paper.

Our experimental results have been obtained using either 4 or 8 Tesla v100 GPUs for learning (compared to 8 third-generation Google Cloud TPUs (Google, 2020) in the original MuZero paper, which are approximately equivalent to 64 v100 GPUs). Each learner GPU receives data from a separated, prioritized experience replay buffer (Horgan et al., 2018) storing the last 500000 transitions. Each of these buffers is filled by 512 dedicated CPU actors 12 , each running a different environment instance. Finally, each actor receives updated parameters from the learner every 500 learner steps (corresponding to

12 For 50 simulations per step; this number is scaled linearly as 12 + 10 · N sim to maintain a constant total number of frames per second when varying N sim .

approximately 4 minutes of wall-clock time); because episodes can potentially last several minutes of wall-clock time, weights updating will usually occur within the duration of an episode. The total score at the end of an episode is associated to the version of the weights that were used to select the final action in the episode.

Hyperparameters choice generally follows those of (Schrittwieser et al., 2019), with the exception that we use the Adam optimizer with a constant learning rate of 0 . 001 .

## B.2. Details on discretizing continuous action space

AlphaZero (Silver et al., 2017a) is designed for discrete action spaces. When applying this algorithm to continuous control, we use the method described in (Tang and Agrawal, 2019) to discretize the action space. Although the idea is simple, discretizing continuous action space has proved empirically efficient (Andrychowicz et al., 2020; Tang and Agrawal, 2019). We present the details below for completeness.

Discretizing the action space We consider a continuous action space A = [ -1 , 1] m with m dimensions. Each dimension is discretized into K = 5 bins; specifically, the continuous action along each dimension is replaced by K atomic categorical actions, evenly spaced between [ -1 , 1] . This leads to a total of K m actions, which grows exponentially fast (e.g. m = 6 leads to about 10 4 joint actions). To avoid the curse of dimensionality, we assume that the parameterized policy can be factorized as π θ ( a | x ) = Π i m =1 π ( i ) θ ( a i | x ) , where π ( i ) θ ( a i | x ) is the marginal distribution for dimension i , a i ∈ { 1 , 2 ...K } is the discrete action along dimension i and a = [ a 1 , a 2 ...a m ] is the joint action.

Modification to the search procedure Though it is convenient to assume a factorized form of the parameterized policy (Andrychowicz et al., 2020; Tang and Agrawal, 2019), it is not as straightforward to apply the same factorization assumption to the Q-function Q ( x, a ) . A most naive way of applying the search procedure is to maintain a Q-table of size K m with one entry for each joint action, which may not be tractable in practice. Instead, we maintain m separate Q-tables each with K entries Q i ( x, a i ) . We also maintain m count tables n ( x, a i ) with K entries for each dimension.

To make the presentation clear, we detail on how the search is applied. At each node of the search tree, we maintain m tables each with K entries as introduced above. The three core components of the tree search are modified as follows.

- Selection. During forward action selection, the algorithm needs to select an action a at node x . This joint action a has all its components a i selected independently, using the action selection formula applied to each dimension. To select action at dimension i , we need the Q-table Q i ( x, a i ) , the prior π ( i ) θ ( a i | x ) and count n ( x, a i ) for dimension i .
- Expansion. The expansion part does not change.
- Back-up. During the value back-up, we update Q-tables of each dimension independently. At a node x , given the downstream reward R ( x, a ) and child value V ( child ( x, a )) , we generate the target update for each Q-table and count table as Q ( x, a i ) ← R ( x, a ) + γV ( child ( x, a )) and n ( x, a i ) ← n ( x, a i ) + 1 .

glyph[negationslash]

The m small Q-tables can be interpreted as maintaining the marginalized values of the joint Q-table. Indeed, let us denote by Q ( x, a ) the joint Q-table with K m entries. At dimension i , the Q-table Q ( x, a i ) increments its values purely based on the choice of a i , regardless of actions in other dimension a j , j = i . This implies that the Q-table Q ( x, a i ) marginalizes the joint Q-table Q ( x, a ) via the visit count distribution.

Details on the learning At the end of the tree search, a distribution target ˆ π or ¯ π is computed from the root node. In the discretized case, each component of the target distribution is computed independently. For example, ˆ π i is computed from N ( x 0 , a i ) . The target distribution derived from constrained optimization ¯ π i is also computed independently across dimensions, from Q ( x 0 , a i ) and N ( x 0 , a i ) . In general, let π target ( ·| x ) be the target distribution and π ( i ) target ( ·| x ) its marginal for dimension i . Due to the factorized assumption on the policy distribution, the update can be carried out independently for each dimension. Indeed, KL[ π target ( ·| x ) , π θ ( ·| x )] = ∑ m i =1 KL[ π ( i ) target ( ·| x ) , π ( i ) θ ( ·| x )] , sums over dimensions.

## B.3. Practical computation of ¯ π

The vector ¯ π is defined as the solution to a multi-dimensional optimization problem; however, we show that it can be computed easily by dichotomic search. We first restate the definition of ¯ π ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(23)

As ∑ b ¯ π α [ b ] is strictly decreasing on α ∈ ( α min , α max ) , Proposition 4 guarantees that ¯ π can be computed easily using dichotomic search over ( α min , α max ) .

## Proof of (i).

Proof. The proof start the same as the one of Lemma 3 of Appendix D.1 setting f ( x ) = -log( x ) to get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with ✶ being the the vector such that ∀ a ✶ a = 1 . Therefore there exists α ∈ R such that

Then α is set such that ∑ b ¯ π b = 1 and ∀ b ¯ π b ≥ 0 .

Proof of (ii).

Proof.

Proof of (iii).

Proof.

Let us define

Proposition 4.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We combine this with the fact that ∑ b π α [ b ] is a decreasing function of α for any α &gt; max b q [ b ] , and ∑ b π α glyph[star] [ b ] = 1 .

Figure 6. Dispersion between seeds at different number of simulations per step on Ms Pacman.

<!-- image -->

## C. Additional experimental results

## C.1. Complements to Section 5.1

Figure 6 presents a comparison of the score obtained by our MuZero implementation and the proposed ALL variant at different simulation budgets on the Ms. Pacman level; the results from Figure 2 are also included fore completeness. In this experiment, we used 8 seeds with 8 GPUs per seed and a batch size of 256 per GPU. We use the same set of hyper-parameters for MuZero and ALL; these parameters were tuned on MuZero. The solid line corresponds to the average score (solid line) and the 95% confidence interval (shaded area) over the 8 seeds, averaged for each seed over buckets of 2000 learner steps without additional smoothing. Interestingly, we observe that ALL provides improved performance at low simulation budgets while also reducing the dispersion between seeds.

Figure 7 presents a comparison of the score obtained by our MuZero implementation and the proposed ALL variant on six Atari games, using 6 seeds per game and a batch size of 512 per GPU and 8 GPUs; we use the same set of hyper-parameters as in the other experiments. Because the distribution of scores across seeds is skewed towards higher values, we represent dispersion between seeds using the min-max interval over the 6 seeds (shaded area) instead of using the standard deviation; the solid line represents the median score over the seeds.

## C.2. Complements to Section 5.3

Details on the environments The DeepMind Control Suite environments (Tassa et al., 2018) are control tasks with continuous action space A = [ -1 , 1] m . These tasks all involve simulated robotic systems and the reward functions are designed so as to guide the system for accomplish e.g. locomotion tasks. Typically, these robotic systems have relatively low-dimensional sensory recordings which summarize the environment states. To make the tasks more challenging, for observations, we take the third-person camera of the robotic system and use the image recordings as observations to the RL agent. These images are of dimension 64 × 64 × 3 .

Figures 9 to 12 present a comparison of MuZero and ALL on a subset of 4 of the medium-difficulty (Van de Wiele et al., 2020) DeepMind Control Suite (Tassa et al., 2018) tasks chosen for their relatively high-dimensional action space among these medium-difficulty problems ( n dim = 6 ). Figure 8 compare the score of MuZero and ALL after 100k learner steps on

Figure 7. Comparison of median score over 6 seeds of MuZero and ALL on six Atari games with 50 simulations per step. The shaded area correspond the the best and worst seeds.

<!-- image -->

these four medium difficulty Control problems. These continuous control problems are cast to a discrete action formulation using the method presented in Appendix B.2; note that these experiments only use pixel renderings and not the underlying scalar states.

These curves present the median (solid line) and min-max interval (shaded area) computed over 3 seeds in the same settings as described in Appendix C.1. The hyper-parameters are the same as in the other experiments; no specific tuning was performed for the continuous control domain. The horizontal dashed line corresponds to the performance of the D4PG algorithm when trained on pixel observations only (Barth-Maron et al., 2018), as reported by (Tassa et al., 2018).

## C.3. Complemantary experiments on comparison with PPO

Since we interpret the MCTS-based algorithms as regularized policy optimization algorithms, as a sanity check for the proposal's performance gains, we compare it with state-of-the-art proximal policy optimization (PPO) (Schulman et al., 2017). Since PPO is a near on-policy optimization algorithm, whose gradient updates are purely based on on-policy data, we adopt a lighter network architecture to ensure its stability. Please refer to the public code base (Dhariwal et al., 2017) for a review of the neural network architecture and algorithmic details.

To assess the performance of PPO, we train with both state-based inputs and image-based inputs. State-based inputs are low-dimensional sensor data of the environment, which renders the input sequence strongly Markovian (Tassa et al., 2018). For image-based training, we adopt the same inputs as in the main paper. The performance is reported in Table 1 where each score is the evaluation performance of PPO after the convergence takes place. We observe that state-based PPO performs significantly better than image-based PPO, while in some cases it matches the performance of ALL. In general, image-based PPO significantly underperforms ALL.

Figure 8. Score of MuZero and ALL on Continuous control tasks after 100k learner steps as a function of the number of simulations N sim .

<!-- image -->

Figure 9. Comparison of MuZero and ALL on Cheetah Run.

<!-- image -->

Figure 10. Comparison of MuZero and ALL on Walker Stand.

<!-- image -->

Figure 11. Comparison of MuZero and ALL on Walker Walk.

<!-- image -->

Figure 12. Comparison of MuZero and ALL on Walker Run.

<!-- image -->

Table 1. Comparison to the performance of PPO baselines on benchmark tasks. The inputs to PPO are either state-based or image-based. The performance is computed as the evaluated returns after the training is completed, averaged across 3 random seeds.

| Benchmarks   |   PPO (state) |   PPO (image) |   MuZero (image) |   ALL(image) |
|--------------|---------------|---------------|------------------|--------------|
| WALKER-WALK  |           406 |           270 |              925 |          941 |
| WALKER-STAND |           937 |           357 |              959 |          951 |
| WALKER-RUN   |           340 |            71 |              533 |          644 |
| CHEETAH-RUN  |           538 |           285 |              887 |          882 |

## D. Derivations for Section 3

## D.1. Proof of Proposition 1, Eq. 11 and Proposition 2.

We start with a definition of the f -divergence (Csisz´ ar, 1964).

Definition 2 ( f -divergence ) . For any probability distributions p and q on A and function f : R → R such that f is a convex function on R and f (1) = 0 , the f -divergence D f between p and q is defined as

<!-- formula-not-decoded -->

Remark 1. Let D f be a f -divergence,

<!-- formula-not-decoded -->

We states four lemmas that we formally prove in Appendix D.2.

Lemma 1.

Lemma 3.

Lemma 4.

<!-- formula-not-decoded -->

Where π θ is assumed to be non zero. We now restate the definition of ˆ π [ a ] glyph[defines] n a +1 N + |A| .

Lemma 2.

<!-- formula-not-decoded -->

Now we consider a more general definition of ¯ π using any f -divergence for some λ f &gt; 0 and assume π θ &gt; 0 ,

<!-- formula-not-decoded -->

We also consider the following action selection formula based on f -divergence D f .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Lemmas 2 and 4 with the appropriate function f directly leads to Proposition 1, Proposition 2, and Proposition 3. In particular, we use

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

| Algorithm   | Function f ( x )   | Derivative f ′ ( x )   | Associated f -divergence                | Associated action selection formula             |
|-------------|--------------------|------------------------|-----------------------------------------|-------------------------------------------------|
| -           | x · log( x )       | log( x )+1             | D f ( p, q ) = KL ( p, q )              | argmax a q a + c √ N · log ( π θ ( a ) n a +1 ) |
| UCT         | 2 - 2 √ x          | - 1 √ x                | D f ( p, q ) = 2 - 2 ∑ b ∈A √ p a · q a | argmax a q a + c · √ π θ n a +1 √               |
| AlphaZero   | - log( x )         | - 1 x                  | D f ( p, q ) = KL ( q, p )              | argmax a q a + c · π θ · N n a +1               |

## D.2. Proofs of Lemmas 1 to 4

## Proof of Lemma 1

Proof. For any action a ∈ A using basic differentiation rules we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof of Lemma 2

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β = -∑ b ∈A [ q b -λ f f ′ ( ˆ π ( b ) π θ ( b ) )] ( |A| + ∑ c ∈A n c ) 2 is independent of a . Also because 1 |A| + ∑ c ∈A n c &gt; 0 then

<!-- formula-not-decoded -->

(45)

## Proof of Lemma 3

Proof. The Eq. 31 is a differentiable strictly convex optimization problem, its unique solution satisfies the KKT condition requires (see Section 5.5.3 of Boyd and Vandenberghe, 2004) therefore there exists α ∈ R such that for all actions a,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

✶ where ✶ is the vector constant equal one: ∀ a ✶ a = 1 . Using Lemma 1 setting π to ¯ π we get

## Proof of Lemma 4

Proof. Since ∑ a ˆ π ( a | x ) = ∑ a ¯ π ( a | x ) = 1 , there exists at least an action a 0 for which 0 ≤ ˆ π ( a 0 | x ) ≤ ¯ π ( a 0 | x ) then 0 ≤ ˆ π ( a 0 | x ) π θ ( a | x ) ≤ ¯ π ( a 0 | x ) π θ ( a | x ) as π θ ( a | x ) &gt; 0 . Because f is convex then f ′ is increasing and therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We put Equations (53) and (54) together

Now using Lemma 3

Then, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(52)

<!-- formula-not-decoded -->

Finally we use again that f ′ is increasing and π θ &gt; 0 to conclude the proof

<!-- formula-not-decoded -->

## D.3. Tracking property in the constant ¯ π case

Let π be some target distribution independent of the round t ≥ 0 . At each round t, starting from t = 1 , an action a t ∈ A is selected and for any t ≥ 0 , we define

<!-- formula-not-decoded -->

where for any action a ∈ A , n t ( a ) is the number of rounds the action a has been selected,

<!-- formula-not-decoded -->

Proposition 5. Assume that for all rounds t ≥ 1 , and for the chosen action a t ∈ A we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Before proving the proposition above, note that O (1 /t ) is the best approximation w.r.t. t, since for any integer k ≥ 0 , taking π ( a ) = ( 1 2 + k ) / ( |A| + t ) , we have that for all n ≥ 0 ,

<!-- formula-not-decoded -->

which follows from the fact that ∀ k, n ∈ N , ∣ ∣ 1 2 + k -( n +1) ∣ ∣ = ∣ ∣ k -n -1 2 ∣ ∣ ≥ 1 2 ·

Proof. By induction on the round t, we prove that

<!-- formula-not-decoded -->

At round t = 1 , Eq. 58 holds as for any action a , n t ( a ) ≥ 0 therefore p t ( a ) ≤ 1 . Now, let us assume that Eq. 58 holds for some t ≥ 1 . We have that for all a,

<!-- formula-not-decoded -->

Note that at each round, there is exactly one action chosen and therefore, ∑ b n t ( b ) = t . Furthermore, for a ′ = a t +1 , we have that n t +1 ( a ′ ) = n t ( a ′ ) , since a ′ has not been chosen at round t +1 . Therefore, for a ′ = a t +1 , glyph[negationslash]

<!-- formula-not-decoded -->

Now, for the chosen action, n t +1 ( a t +1 ) = n t ( a t +1 ) + 1 . Using our assumption stated in Eq. 57, we have that

<!-- formula-not-decoded -->

which concludes the induction. Next, we compute a lower bound. For any action a ∈ A and round t ≥ 1 , glyph[negationslash]

<!-- formula-not-decoded -->

We have for any action a ∈ A , glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

Since when |A| = 1 , then by definition, p t ( a ) = π ( a ) = 1 and for all rounds t ≥ 1 , we get

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]