## An Optimal Tightness Bound for the Simulation Lemma

Sam Lobel samuel\_lobel@brown.edu Department of Computer Science Brown University

Ronald Parr parr@cs.duke.edu Department of Computer Science Duke University

## Abstract

We present a bound for value-prediction error with respect to model misspecification that is tight, including constant factors. This is a direct improvement of the 'simulation lemma,' a foundational result in reinforcement learning. We demonstrate that existing bounds are quite loose, becoming vacuous for large discount factors, due to the suboptimal treatment of compounding probability errors. By carefully considering this quantity on its own, instead of as a subcomponent of value error, we derive a bound that is sub-linear with respect to transition function misspecification. We then demonstrate broader applicability of this technique, improving a similar bound in the related subfield of hierarchical abstraction.

## 1 Introduction

In reinforcement learning, an agent is frequently tasked with making decisions in an environment that it cannot model perfectly. This may occur because the environment is learned about through sampled data, or because the agent's environment model is simplified through some abstraction. In such cases it is natural to ask, how might the quality of this approximation impact an agent's decision making? This is the subject of the 'simulation lemma,' a foundational result in reinforcement learning that bounds the error in value estimation when the transition and reward function are known only with some specified degree of precision.

The simulation lemma was introduced in the context of exploration and finds use in a variety of domains that utilize imperfect models, such as hierarchical abstraction (Abel et al., 2016) and offline policy evaluation (Yin et al., 2021). Frequently, results of this kind rely on developing a recursive relationship between the value error at subsequent timesteps. We show that this approach implicitly overestimates how probability errors compound over time. By more directly approximating this quantity, we produce a bound on value-estimation error that is demonstrably tight. We then show that existing bounds can be derived as a linearization of our result, and finally apply our result to a hierarchical setting to demonstrate broader applicability.

## 2 Background and Related Work

We develop our results in the framework of Markov Decision Processes (MDPs): M = ( S , A , R, T, γ ), where S is the state space, A is the action space, and γ ∈ [0 , 1] is the discount factor. The nextstate transition probabilities are given by T ( s ′ | s, a ), and the reward function by R ( s, a ) ∈ [0 , 1]. A policy π ( a | s ) gives the probability of taking an action from a given state. The objective in the MDP framework is generally either to construct a policy π that maximizes the expected γ -discounted sum of reward, or to evaluate a given policy on this same measure.

When a model of the environment is given, these quantities can be computed exactly, for example through policy iteration or dynamic programming (Howard, 1960). In reinforcement learning, however, the agent generally is not given this model, and instead must learn about the environment

through interaction. A common approach to this is model-based reinforcement learning (Moerland et al., 2023; Auer &amp; Ortner, 2006), which aims to estimate the environment's transitions and rewards from gathered data. However, when using finite data, the learned model is generally imperfect. This work concerns itself with developing optimal bounds on policy evaluation error in the setting of misspecified models. Here we detail a variety of areas in which such a bound is useful, along with related lines of study.

Exploration The original simulation lemma was introduced in the context of efficient exploration (Kearns &amp; Singh, 2002), to quantify policy evaluation error as a function of state-action visitation counts. Understanding the effect of imperfect modelling is central to efficient exploration (Auer &amp; Ortner, 2006; Auer et al., 2008; Brafman &amp; Tennenholtz, 2002). Methods that use these measures include count-based exploration (Strehl &amp; Littman, 2008) and its pseudocount approximations (Bellemare et al., 2016; Lobel et al., 2023).

Abstraction Model approximation frequently appears in the field of abstraction, where a full model of an MDP is replaced by one that is simpler in some respect. As we show later, our methodology can be used to improve the value error bounds when performing this replacement with state-action abstracted options (Sutton et al., 1999). A simple form of state abstraction is discretization , where sets of states are grouped by some measure of similarity. A common example of this occurs in the partially observable MDP framework (Lee et al., 2007; Grover &amp; Dimitrakakis, 2021), where the continuous belief-state space can be discretized into an approximate, finite MDP.

Offline Policy Evaluation The goal of offline policy evaluation (OPE) is to estimate the value of a policy using a fixed dataset of transitions, often generated by a different policy. Model-based OPE involves fitting an empirical model of transitions and rewards from this dataset, and using this to estimate value (Gottesman et al., 2019). In this setting, the simulation lemma often is a key step in constructing accuracy bounds of the estimated value (Yin &amp; Wang, 2020; Yin et al., 2021).

We also note that a variety of results in the literature bound the value error using different measures of similarity than the original simulation lemma. Perhaps most closely related to our contribution is work that bounds multi-step transition error of imperfectly-modelled Lipschitz transition functions (Asadi et al., 2018). This results in a similar sum of compounding errors to ours, albiet in a different setting. Bisimulation metrics (Ferns et al., 2004) unify transition and reward error into a single quantity that can be used to measure the similarity of MDPs with entirely different state spaces.

## 3 Main Result

We begin by stating the conditions of the original simulation lemma. We consider two MDPs: M = ( S , A , R, T, γ ), and ˆ M = ( S , A , ˆ R, ˆ T, γ ), which share a state-action space, but have (boundedly) different transition and reward functions. We are interested in the effect of running the same policy π on these two related MDPs. Let P π be a matrix that contains the policy-conditioned state-state transition probabilities, and R π be a vector that contains the per-state expected reward:

<!-- formula-not-decoded -->

We define ˆ P π and ˆ R π analogously for MDP ˆ M . Throughout this work, a single index on a matrix (or vector) extracts the specified row vector (or scalar). Furthermore, P a and R a refer to the transition probabilities, and expected reward, of executing action a from each state. Using this notation, we can quantify the difference between two transition or reward functions with the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are interested in the value difference between running π on each MDP. The value of a state for a given policy and MDP is defined as the expected discounted sum of rewards:

<!-- formula-not-decoded -->

where s t is a random variable representing the state at timestep t . Noting that Pr( s t = s ′ | s 0 = s, π ) = ( P π ) t s,s ′ , we can concisely represent value in vectorized notation as follows:

<!-- formula-not-decoded -->

where 〈 · , · 〉 denotes the inner product between two vectors. We define ˆ V π analogously for ˆ M .

## 3.1 Original Simulation Lemma

We are interested in quantifying the maximum value difference between running the same policy on two different MDPs. The original simulation lemma bounds this quantity as follows:

<!-- formula-not-decoded -->

Existing proofs of the simulation lemma frequently take advantage of a recursive representation of value (the Bellman Equation) (Howard, 1960):

<!-- formula-not-decoded -->

For a complete proof, please refer to Jiang (2018) or see Appendix A. The key mathematical idea is to establish the following recursive relationship:

<!-- formula-not-decoded -->

which can then be easily transformed into the simulation lemma's bound. Analyzing the recursive relationship above, the first term ( /epsilon1 R ) represents a one-step reward-prediction error. The second term ( γ/epsilon1 T 2(1 -γ ) ) represents the maximum value error that results from misspecifying /epsilon1 T of the nextstate distribution's probability mass. However, by defining the recursive relationship as such, this bound implicitly assumes that the process can continually misspecify /epsilon1 T of its probability at each timestep. This quickly amounts to misspecifying more than the entire probability mass, leading to a vast overestimate of the value error, in particular when /epsilon1 T &gt; 1 -γ . In contrast, we carefully track the probability drift at each timestep to avoid this issue.

## 3.2 Bounding Probability Distance

We seek to bound the probability distance tightly at any timestep t . To do so effectively, it is useful to frame distances between probability vectors in terms of their overlap, instead of their L 1 distance.

0.9

0.8

0.7

0.6

0.3

0.2

0.1

0.0

-1.00

-0.75 -0.50 -0.25

0.00

0.25

Figure 1: Visualization of relation between L 1 distance and overlap of two probability distributions (Equation 7). The blue and orange shaded regions together comprise the L 1 distance. The brown region represents overlap. Overlap plus either the blue or orange sections constitutes a probability distribution, and therefore has total area 1. Thus the blue and orange regions both individually have area ‖ p -ˆ p ‖ 1 / 2, and so ‖ ¯ p ‖ 1 = 1 -‖ p -ˆ p ‖ 1 / 2.

<!-- image -->

We note that Jiang et al. (2016) uses similar machinery to bound compounding probability error (Lemma 1), though applies this insight in a different context. For two probability vectors p, ˆ p , we define their overlap as ¯ p , such that for each index i :

<!-- formula-not-decoded -->

Usefully, because each element of p -¯ p (and likewise ˆ p -¯ p ) is non-negative, the L 1 norm of the difference between these two vectors is equal to the difference between the L 1 norms:

<!-- formula-not-decoded -->

We use this to derive an equivalence between overlap and L 1 distance, related to the concept of total variation distance (Levin &amp; Peres, 2017). Below, we use the notation [ p ] + to indicate a thresholded version of p that retains only the non-negative parts, [ p ] + i = max( p i , 0):

<!-- formula-not-decoded -->

See Figure 1 for a demonstration and explanation of this equivalence. This relationship allows for a simple rewriting of the transition-error condition of the simulation lemma (Equation 2):

<!-- formula-not-decoded -->

Using this framing, we can now lower-bound the overlap of state-distributions at timestep t when starting from s 0 , by demonstrating that at every timestep, at least 1 -/epsilon1 T / 2 fraction of the prior timestep's distributional overlap is retained. For notational convenience, P t s 0 ,s = ( P π ) t s 0 ,s , and

¯ M t s 0 ,s = min( P t s 0 ,s , ˆ P t s 0 ,s ). Thus,

<!-- formula-not-decoded -->

The third line can be understood as providing the minimum operator more options to choose from, in that after bringing the minimum inside of the sum, the two elements in the second line are both still possible choices and so the inequality holds. The fourth line can be understood similarly for multiplication.

With ¯ M 0 = I as the base case, applying recursion yields

<!-- formula-not-decoded -->

We contrast this with the equivalent recursive proof of distributional drift using the L 1 formulation of transition misspecification, akin to the recursion employed by the original simulation lemma (Equation 5):

<!-- formula-not-decoded -->

where ‖ · ‖ 1 above refers to both the matrix and vector 1-norm, and on the third line we use the identity ‖ Ax ‖ 1 ≤ ‖ A ‖ 1 ‖ x ‖ 1 . This result makes clear the contrast between the two methods for computing distributional drift: Naïvely using the L 1 formulation leads to unbounded accumulation of drift as horizon approaches infinity, while the overlap formulation smoothly decays from 1 to 0. This difference is crucial to generating the tighter bound in the next section.

## 3.3 A Tight Bound on Value Error

We are now ready to prove our main result, a tight bound on the value error.

Theorem 1 For two MDPs M and ˆ M related as described in Equations 2 and 3, the following inequality holds:

<!-- formula-not-decoded -->

Furthermore, this bound is tight.

Proof: Since the conditions of the simulation lemma (Equations 2,3) are symmetric with respect to M and ˆ M , without loss of generality we assume V π s 0 ≥ ˆ V π s 0 , and thus | V π s 0 -ˆ V π s 0 | = V π s 0 -ˆ V π s 0 . We now add and subtract the same quantity in a way that allows for discarding a strictly non-positive term:

<!-- formula-not-decoded -->

By construction, ¯ M t s 0 is the overlap between P t s 0 and ˆ P t s 0 , and thus and entries of ¯ M t s 0 -ˆ P t s 0 are strictly non-positive. Since rewards are likewise non-negative, the third inner product in the above sum is always non-positive. Thus, we can drop this term to significantly tighten our bound.

<!-- formula-not-decoded -->

This proof makes use of Hölder's inequality to bound inner products with L 1 and L ∞ norms, as well as the identity in Equation 6 to split ‖ P t s 0 -¯ M t s 0 ‖ 1 into ‖ P t s 0 ‖ 1 -‖ ¯ M t s 0 ‖ 1 . We provide a parallel

Normalized Value Error with Varying ER, ET (V = 0.9)

Normalized Value Error with Varying V (ER, ET/2 = 0.1)

10

10

8

IV-41 / VMAX

8

6

6

4

4

2

2

0

• Original Bound

Original Bound

Our Bound

Our Bound

0.2

0.2

0.4

Figure 2: Bounds on value error given by original simulation lemma as well as our tighter bounds, normalized by V MAX . (Left) Bound on value error with increasing gamma shows the original lemma's suboptimality with respect to discount. (Right) Bound on value error with increasing misspecification shows looseness of linear approximation compared to the tight bound.

<!-- image -->

proof for the finite-horizon undiscounted setting in Appendix B. We briefly remark that this bound matches intuition:

- When γ = 0, then | V π s -ˆ V π s | ≤ /epsilon1 R since only the first step contributes to value.
- When /epsilon1 R = 1, the MDPs can have completely different reward functions and thus | V π s -ˆ V π s | ≤ 1 1 -γ = V MAX .
- When /epsilon1 R = /epsilon1 T = 0, the MDPs are identical and thus | V π s -ˆ V π s | = 0.

Additionally we note the the original simulation lemma can be reproduced as a Taylor expansion of our bound around /epsilon1 R = 0 and /epsilon1 T = 0, proving that the original bound is the tightest possible linear approximation to the maximal error as model misspecification approaches 0. Figure 2 presents a comparison of our bound with the original simulation lemma, demonstrating superiority in the large-misspecification and large-discount limits.

## 3.4 Proof of Tightness

We now demonstrate that this is the tightest possible bound, including constant factors, by constructing a pair of MDPs with exactly this value error. M consists of two states, both of which transition to themselves, with R ( s 1 ) = 1 and R ( s 2 ) = 0. We construct ˆ M so that ˆ V ( s 1 ) is as small as possible given /epsilon1 R , /epsilon1 T , by setting ˆ R ( s 1 ) = 1 -/epsilon1 R , and transitioning from s 1 to s 2 with probability /epsilon1 T / 2 (and thus self-transitions with /epsilon1 T / 2 less probability, so ‖ P π s 1 -ˆ P π s 1 ‖ 1 = /epsilon1 T ). Hence, V ( s 0 ) = 1 1 -γ and ˆ V ( s 0 ) = 1 -/epsilon1 R .

<!-- formula-not-decoded -->

Intuitively, this result makes clear the role of /epsilon1 T as modifying the discount factor of ˆ M . A discount can be interpreted as entering an absorbing state with probability 1 -γ at each timestep (Sutton &amp; Barto, 2018). In ˆ M , this instead occurs more frequently, with probability 1 -γ (1 -/epsilon1 T / 2).

## 3.5 Value Loss of Optimal Policy

The simulation lemma directly applies to bounding the value difference of executing the same policy on two related MDPs. However, in reinforcement learning the task is frequently to learn an optimal policy π ∗ , that has the following property:

<!-- formula-not-decoded -->

0.0

It is natural to ask, if one learns the optimal policy ˆ π ∗ by training on an approximate MDP ˆ M , how much worse will this policy do than π ∗ when executed on the actual MDP M ? In contrast to the simulation lemma, we are comparing the value loss of different policies on the same MDP. Noting that ˆ V ˆ π ∗ s ≥ ˆ V π ∗ s :

<!-- formula-not-decoded -->

This is simply twice the value error of executing the same policy on different MDPs. Thus, by improving the simulation lemma bound, we similarly tighten the estimated value loss when training on an approximate MDP. Similar results are common in inverse RL, e.g., Burchfiel et al. (2016), and have been noted in the context of the simulation lemma as well (Jiang, 2018).

## 3.6 Application to Hierarchy

Analogs to the simulation lemma exist throughout the reinforcement learning literature; here, we present an extension of our proof to one such instance in the field of hierarchical reinforcement learning. We use the formalism of φ -relative options (Abel et al., 2020), a form of approximately value preserving state and action abstractions.

Let O ∗ φ be a set of options o ∗ over abstract states s φ ∈ S φ , that can be composed to form a policy that is optimal in the base MDP. Let ˆ O φ be a set of options that approximates O ∗ φ in that

<!-- formula-not-decoded -->

where R o s and P o s,s ′ represent the reward and multi-time models of Sutton et al. (1999). We define V π o ∗ as the value of executing the best policy over O ∗ φ , and V π ˆ o as the value of executing an approximately equivalent policy using options from ˆ O φ . By bounding probability distances we arrive at the following relation:

<!-- formula-not-decoded -->

This improves on the existing bound (Abel et al., 2020):

<!-- formula-not-decoded -->

in much the same way as our original result improves upon the simulation lemma. A proof, more complete definitions, and an example demonstrating tightness are deferred to Appendix C. The main difference in applying our technique to this domain is careful treatment of the multi-time transition function, where ∑ s ′ P o s,s ′ = 1.

## 4 Conclusion

The simulation lemma is a widely used result in reinforcement learning that quantifies the effect of model misspecification on value. We demonstrate that the originally provided bound is quite loose,

/negationslash

becoming vacuous when applied to large discount factors frequently used in reinforcement learning. In this work we present a version of this lemma that is optimally tight, along with an example application of this method to hierarchical reinforcement learning. We expect that our bound can be applied to a variety of results throughout the literature, and that the general proof technique can be useful in other domains.

## Acknowledgements

We would like to thank George Konidaris for valuable input during the early stages of this work, as well as Tuluhan Akbulut and Ruo Yu Tao for helping inspire the question we ask here. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under grant #2040433 and ARO grant #W911NF2210251.

## References

- David Abel, David Hershkowitz, and Michael Littman. Near optimal behavior via approximate state abstraction. In International Conference on Machine Learning , pp. 2915-2923. PMLR, 2016.
- David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup, and Michael Littman. Value preserving state-action abstractions. In International Conference on Artificial Intelligence and Statistics , pp. 1639-1650. PMLR, 2020.
- Kavosh Asadi, Dipendra Misra, and Michael Littman. Lipschitz continuity in model-based reinforcement learning. In International Conference on Machine Learning , pp. 264-273. PMLR, 2018.
- Peter Auer and Ronald Ortner. Logarithmic online regret bounds for undiscounted reinforcement learning. Advances in neural information processing systems , 19, 2006.
- Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems , 21, 2008.
- Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems , 29, 2016.
- Ronen I Brafman and Moshe Tennenholtz. R-max a general polynomial time algorithm for nearoptimal reinforcement learning. Journal of Machine Learning Research , 3(Oct):213-231, 2002.
- Benjamin Burchfiel, Carlo Tomasi, and Ronald Parr. Distance minimization for reward learning from scored trajectories. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 30, 2016.
- Norman Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite Markov decision processes. Proceedings of the 20th conference on Uncertainty in Artificial Intelligence , 2004.
- Omer Gottesman, Yao Liu, Scott Sussex, Emma Brunskill, and Finale Doshi-Velez. Combining parametric and nonparametric models for off-policy evaluation. In International Conference on Machine Learning , pp. 2366-2375. PMLR, 2019.
- Divya Grover and Christos Dimitrakakis. Adaptive belief discretization for POMDP planning. arXiv preprint arXiv:2104.07276 , 2021.
- Ronald A Howard. Dynamic programming and Markov processes. 1960.
- Nan Jiang. Notes on tabular methods. 2018. URL https://nanjiang.cs.illinois.edu/files/ cs542f22/note3.pdf .
- Nan Jiang, Satinder Singh, and Ambuj Tewari. On structural properties of mdps that bound loss due to shallow planning. In IJCAI , volume 8, pp. 1, 2016.

- Sham Kakade, Michael J Kearns, and John Langford. Exploration in metric state spaces. In International Conference on Machine Learning , pp. 306-312. PMLR, 2003.
- Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning , 49:209-232, 2002.
- Wee Lee, Nan Rong, and David Hsu. What makes some POMDP problems easy to approximate? Advances in neural information processing systems , 20, 2007.
- David A Levin and Yuval Peres. Markov chains and mixing times , volume 107. American Mathematical Soc., 2017.
- Sam Lobel, Akhil Bagaria, and George Konidaris. Flipping coins to estimate pseudocounts for exploration in reinforcement learning. In International Conference on Machine Learning , pp. 22594-22613. PMLR, 2023.
- Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Model-based reinforcement learning: A survey. Foundations and Trends® in Machine Learning , 16(1):1-118, 2023.
- Alexander L Strehl and Michael L Littman. An analysis of model-based interval estimation for Markov decision processes. Journal of Computer and System Sciences , 74(8):1309-1331, 2008.
- Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.
- Richard S Sutton, Doina Precup, and Satinder Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial intelligence , 112(1-2):181-211, 1999.
- Ming Yin and Yu-Xiang Wang. Asymptotically efficient off-policy evaluation for tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pp. 3948-3958. PMLR, 2020.
- Ming Yin, Yu Bai, and Yu-Xiang Wang. Near-optimal provable uniform convergence in offline policy evaluation for reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pp. 1567-1575. PMLR, 2021.

## A Full proof of Simulation Lemma

For completeness, we include the proof of the simulation lemma found in Jiang (2018). We adopt notation from Section 3.

<!-- formula-not-decoded -->

This proof makes use of Hölder's inequality to bound inner products with L 1 and L ∞ norms, as well as centers the value 0 ≤ V π s ≤ 1 1 -γ through subtracting the midpoint for improved bounds.

## B Application to the Finite-Horizon Setting

We now extend our improved bound to the finite-horizon, undiscounted setting, where an agent interacts with an environment for H steps. One difference in this setting is that policies are conditioned on timestep as well as state; hence we define π = [ π 0 , . . . , π H -1 ]. Existing bounds in the finite-horizon setting establish a relationship between values at subsequent timesteps. Noting that 0 ≤ V π h,s ≤ H -h (and defining V π H,s = 0), Then,

<!-- formula-not-decoded -->

For our bound, the only change from the discounted setting is replacing the discounted infinite sums of Section 3.3 with finite undiscounted ones. Redefining P t = ∏ 0 ≤ i&lt;t P π i , and WLOG assuming that V π 0 ,s 0 ≥ ˆ V π 0 ,s 0 we can show:

<!-- formula-not-decoded -->

Again, we note that Taylor expanding this relation at /epsilon1 R = 0 and /epsilon1 T = 0 recovers the original bound.

## C Proof of Hierarchy Bound

This proof exactly mirrors the one in the main body, with additional care taken to handle multi-time models. We first describe the φ -relative options framework (definitions largely taken from Abel et al. (2020)), and then provide a tighter bound on value loss.

An option o ∈ O is an abstract action defined by the tuple ( I o , β o , π o ), where I o ⊆ S is the subset of the state space the option can initiate in, β 0 ⊆ S is the subset the option terminates in, and π o is a policy. For a given state abstraction φ : S → S φ , an option o φ is said to be φ -relative if and only if ∃ s φ ∈ S φ such that

<!-- formula-not-decoded -->

In words, a φ -relative option is one that executes from anywhere in one abstract state, and terminates upon leaving that abstract state. Furthermore, O φ denotes a set of only φ -relative options, with at least one option that executes at each abstract state.

Let O ∗ φ be a set of φ -relative options o ∗ that can be composed to form an optimal policy in the base MDP. Let ˆ O φ be a set of options that approximates O ∗ φ in that

<!-- formula-not-decoded -->

where R o s and P o s,s ′ represent the multi-time reward and transition functions described in Sutton et al. (1999):

<!-- formula-not-decoded -->

In words, R o s is the expected discounted reward accumulated over the course of an option execution, and P o s,s ′ is the total discounted probability that an option terminates in s ′ when starting from s . Crucially, ∑ s ′ ∈S P o s,s ′ ≤ γ &lt; 1. We also note that the /epsilon1 T bound is per-entry, not per-vector. This was the form of the conditions in the original simulation lemma (Kearns &amp; Singh, 2002), which was replaced with a vectorized version in subsequent work (Kakade et al., 2003).

Since ‖ P o s ‖ 1 may take on different values for different options and starting states, we can no longer directly use a relation similar to Equation 7. However, we can augment the MDP by adding an absorbing state s x , and modify each option such that

/negationslash

<!-- formula-not-decoded -->

By doing this, ‖ P o s ‖ 1 = γ without modifying the behavior of the given option in the base MDP. This allows our proof to proceed treating options in roughly the same way as we do actions in the main body. Noting that since P o s,s ≡ 0 by construction, for two options o ∗ , ˆ o ∗ satisfying the relations of Equation 11 we have that:

/negationslash

<!-- formula-not-decoded -->

Thus we can recover a condition similar to that of Equation 2:

<!-- formula-not-decoded -->

Due to the addition of s x , we can now describe the above bound in terms of overlap. Defining ¯ P o ∗ , ˆ o s,s ′ = min( P o ∗ s,s ′ , P ˆ o s,s ′ ), we can produce a similar relation to Equation 8:

<!-- formula-not-decoded -->

Let Π O φ be the set of abstract policies representable by O φ . Let π o ∗ be a policy within Π O ∗ φ that is optimal in the base MDP. Let π ˆ o be a policy in Π ˆ O φ produced by replacing each o ∗ chosen by π o ∗ with an option ˆ o satisfying the relations of Equation 11. Then, we can follow the same algebraic steps as in the main body to produce the following bound:

<!-- formula-not-decoded -->

## C.1 Proof of Tightness

We can generate an abstract MDP that achieves this bound using a similar recipe as in Section 3.4. We construct an abstract MDP where each option o ∗ transitions uniformly to each other state with discounted probability γ | S |-1 , receiving a reward of R MAX . We then construct a new set of options that uniformly transition with discounted probability γ | S |-1 -/epsilon1 T , receiving reward R MAX -/epsilon1 R . This exactly reproduces the provided bound.