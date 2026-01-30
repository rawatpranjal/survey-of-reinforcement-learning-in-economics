## Learning True Objectives: Linear Algebraic Characterizations of Identifiability in Inverse Reinforcement Learning

Mohamad Louai Shehab Antoine Aspeel Nikos Ar´ echiga Andrew Best Necmiye Ozay

Editors:

A. Abate, K. Margellos, A. Papachristodoulou

## Abstract

Inverse reinforcement Learning (IRL) has emerged as a powerful paradigm for extracting expert skills from observed behavior, with applications ranging from autonomous systems to humanrobot interaction. However, the identifiability issue within IRL poses a significant challenge, as multiple reward functions can explain the same observed behavior. This paper provides a linear algebraic characterization of several identifiability notions for an entropy-regularized finite horizon Markov decision process (MDP). Moreover, our approach allows for the seamless integration of prior knowledge, in the form of featurized reward functions, to enhance the identifiability of IRL problems. The results are demonstrated with experiments on a grid world environment.

Keywords: Markov decision process, inverse reinforcement learning, identifiability

## 1. Introduction

Inverse reinforcement learning (IRL) is the problem of finding the reward function of an agent from its behavior Ng and Russell (2000). IRL has gained significant attention in the research community since having access to expert demonstrations can alleviate the burden of manually specifying a reward function Abbeel and Ng (2004) and improve generalizability. A primary problem with IRL is that it is fundamentally ill-posed. Indeed, there are multiple reward functions leading to any observed behavior. Prior work has generally dealt with this ambiguity in reward learning by using heuristics, e.g., Max Margin IRL Ratliff et al. (2006), Bayesian IRL Ramachandran and Amir (2007), Max Entropy IRL Ziebart et al. (2008), Relative Entropy IRL Boularias et al. (2011), and Deep Max Entropy IRL Wulfmeier et al. (2015) (see Arora and Doshi (2021) for a comprehensive overview). These approaches are well-suited for learning an imitation policy since the learned reward is guaranteed to induce a learned policy at least as good as the expert one. However, when IRL is used for behavior modeling Li et al. (2022); Ashwood et al. (2022); Babes et al. (2011); Ramponi et al. (2020); Jenner and Gleave (2021), or for policy transfer to novel environments Cao et al. (2021); Rolland et al. (2022); Fu et al. (2018), it becomes crucial to address the reward ambiguity problem. In such settings, finding one reward function that explains the agent's behavior is not enough since different reward functions can lead to different interpretations of the agent's preferences or completely different behaviors on a modified environment. Instead, it is necessary to find the set of all possible rewards Metelli et al. (2021) that can explain the behavior.

This leads to different notions of reward equivalence. For example, two rewards are said to be trajectory equivalent if they lead to the same distribution of trajectories under the optimal policy.

MLSHEHAB@UMICH.EDU ANTOINAS@UMICH.EDU NIKOS.ARECHIGA@TRI.GLOBAL ANDREW.BEST@TRI.GLOBAL NECMIYE@UMICH.EDU

Equivalence classes of rewards allow us to formalize the concept of identifiability of an MDP as follows: identifiability holds when a reward can be identified up to the corresponding equivalence class . In this context, our contribution is two-fold. First, we derive linear algebraic characterizations of weak, almost-strong, and strong trajectory equivalence classes of a reward function. This leads to necessary and sufficient conditions for the corresponding notions of identifiability. Then, we show how incorporating prior knowledge -in the form of featurized reward functions- can be seamlessly integrated into the framework to enhance the identifiability of rewards in certain environments.

## 2. Preliminaries

## 2.1. Notation

We denote by N and R the sets of natural and real numbers, respectively. The identity matrix in R n × n is denoted by I n , the zero matrix in R m × n is denoted by 0 m × n , and the vector of ones in R n is denoted by 1 n . For matrices A and B , [ A B ] is the horizontal concatenation of A and B . We denote by ker ( A ) and ran ( A ) the null space and the column span of the matrix A respectively. For a matrix A and a set X , AX is the set { Ax | x ∈ X } . For any two sets X and Y , X × Y is their Cartesian product and X ⊕ Y is their Minkowski sum. For a vector x , x ⊕ Y denotes { x } ⊕ Y . We denote by dim ( V ) the dimension of a vector space V . The cardinality of a set Ω is denoted by | Ω | , and ∆(Ω) denotes the set of probability measures over the set Ω . The support of a measure µ ∈ ∆(Ω) is the set support ( µ ) = { x ∈ Ω | µ ( x ) &gt; 0 } . The 'Dirac' distribution that sets a point mass at state s ∈ Ω is denoted δ s, Ω ∈ ∆(Ω) . It will be denoted δ s when Ω is clear from the context. The indicator function ✶ ( · ) is ✶ ( a = b ) = 1 if a = b , and 0 otherwise. Given a function f : X → Y , and a set A ⊆ X , we denote by f | A the restriction of f to A .

## 2.2. Markov Decision Processes

A Markov Decision Process (MDP) is a tuple ( S , A , T , µ 0 , r, γ, T ) , where S = { s (1) , . . . , s ( n ) } is a finite set of states with cardinality |S| = n ; A = { a (1) , . . . , a ( m ) } is a finite set of actions with cardinality |A| = m ; T : S × A → ∆( S ) is a Markov transition kernel; µ 0 ∈ ∆( S ) is an initial distribution over the set of states; r : S×A → R is a reward function (or reward for short); γ ∈ [0 , 1] is a discount factor; and T ∈ N is the non-negative time horizon. A policy π t : S → ∆( A ) is a function that describes an agent's behavior at time step t by specifying an action distribution at each state. We denote by π = ( π t ) T -1 t =0 the time-varying stochastic policy throughout the entire horizon. A trajectory τ (of length T ) is an alternating sequence of states and actions (ending with a state), i.e., τ = ( s 0 , a 0 , s 1 , a 1 , . . . , s T -1 , a T -1 , s T ) with s t ∈ S and a t ∈ A . Under a policy π , a trajectory τ occurs with probability

<!-- formula-not-decoded -->

which depends on the distribution of initial states, the policy, and the Markov transition kernel. We consider the Maximum Entropy Reinforcement Learning (MaxEntRL) objective given by:

<!-- formula-not-decoded -->

where λ &gt; 0 is a regularization parameter, and H ( π t ( . | s t )) = -∑ a ∈A π t ( a | s t ) log( π t ( a | s t )) is the entropy of the policy π t . The expectation is with respect to the probability measure P π µ 0 . We denote by Ω the support of P π µ 0 . Similarly, we denote by Ω( s 0 ) the support of P π δ s 0 , for some s 0 ∈ support ( µ 0 ) . The reward of a trajectory τ is given by overloading the reward function r ( τ ) = T -1 ∑ t =0 γ t r ( s t , a t ) . We define the optimal policy set Π ∗ r , corresponding to a reward function r , as the set of maximizers of (1), i.e.,

<!-- formula-not-decoded -->

The non-uniqueness of the optimal policy stems from the fact that the policy can be arbitrarily specified for the non-accessible states without changing the objective value. However, the policy is unique over the accessible state-action pairs Kim et al. (2021). To formalize this, we define the accessible states at time step t and those throughout the horizon T as:

<!-- formula-not-decoded -->

respectively. When we restrict the policies in Π ∗ r to the accessible states, we obtain a unique policy, denoted by π ∗ r | Access 1 . Since the trajectory distribution for a given policy depends only on the accessible states, we define the optimal trajectory distribution for a reward r as p r = P π ∗ r µ 0 , where π ∗ r ∈ Π ∗ r is arbitrary. In particular, p r is the distribution of trajectories when using an optimal policy corresponding to r and starting from the support of µ 0 . Finally, we define an MDPModel as a tuple ( S , A , T , µ 0 , R, γ, T ) where R is a set of reward functions, and S , A , T , µ 0 , γ , and T are defined as for an MDP.

## 2.3. Reward identifiability and Equivalence Classes

As in many identification problems, rewards can only be identified up to an equivalence class. Roughly speaking, an MDP model is more identifiable when the equivalence class is smaller. In what follows, we define a set of equivalence classes and use them to define different notions of identifiability. Let R ⊆ R nm be the set of reward functions for the given MDP model. Let ∼⊆ R × R denote an equivalence relation on R . For a given reward r ∈ R , the equivalence class of r with respect to the relation ∼ is defined as [ r ] ∼ = { ˆ r ∈ R | ˆ r ∼ r } , where we use the shorthand ˆ r ∼ r for (ˆ r, r ) ∈∼ . Some of the equivalence relations of interest are as follows.

Definition 1 (Distribution Equivalence ∼ d ) Given an MDP model, two rewards r and ˆ r in R are distribution equivalent, denoted by r ∼ d ˆ r , if p r = p ˆ r .

In words, two rewards are distribution equivalent when they induce the same optimal trajectory distribution.

Definition 2 (Policy Equivalence ∼ π ) Given an MDP model, two rewards in R are policy equivalent, denoted by r ∼ π ˆ r , if π ∗ r | Access = π ∗ ˆ r | Access .

1. The notation π t , the policy at time step t , is overloaded with π r , the policy throughout the horizon [0 , T -1] corresponding to r .

Two rewards are policy equivalent if they induce the same optimal time-varying policy over the accessible states. Since p r = p ˆ r ⇐⇒ π ∗ r | Access = π ∗ ˆ r | Access , distribution equivalence class and policy equivalence class are the same, hence we use them interchangeably.

Definition 3 (Weak Trajectory Equivalence ∼ τ Kim et al. (2021)) Given an MDP model, two rewards in R are weak trajectory equivalent, denoted by r ∼ τ ˆ r , if for all s 0 ∈ support ( µ 0 ) , there exists c s 0 ∈ R such that r ( τ ) = ˆ r ( τ ) + c s 0 , for all τ ∈ Ω( s 0 ) .

Weak trajectory equivalence means that the two rewards are equivalent if their discounted sums along trajectories starting from the same initial state are a unique constant apart.

Definition 4 (Strong Trajectory Equivalence ∼ ω ) Given an MDP model, two rewards in R are strong trajectory equivalent, denoted by r ∼ ω ˆ r , if there exists some c ∈ R such that r ( τ ) = ˆ r ( τ )+ c , for all τ ∈ Ω .

Strong trajectory equivalence is similar to weak trajectory equivalence but requires the discounted sums of rewards along all possible trajectories to be a unique constant apart independent of the initial state.

Definition 5 (State-Action Equivalence ∼ s,a Kim et al. (2021)) Given an MDP model, two rewards in R are state-action equivalent, denoted by r ∼ s,a ˆ r , if there exists c ∈ R s.t. r ( s, a ) = ˆ r ( s, a ) + c, for all ( s, a ) ∈ S × A .

State-action equivalence means that the two rewards are equivalent if they are a unique constant c apart at all state-action pairs. When the reward set is R = R nm , state-action equivalence class is the smallest equivalence class up to which it is possible to identify a reward. Indeed, from the definitions, it is easy to see that:

<!-- formula-not-decoded -->

Different notions of identifiability of MDP models in the literature deal with the question of when the reverse implications hold. In particular, we have the following definitions.

Definition 6 (Identifiability) An MDP model is said to be:

- i. weakly identifiable if for all r, ˆ r ∈ R , r ∼ π ˆ r ⇐⇒ r ∼ τ ˆ r .
- ii. almost-strongly identifiable if for all r, ˆ r ∈ R , r ∼ π ˆ r ⇐⇒ r ∼ ω ˆ r .
- iii. strongly identifiable if for all r, ˆ r ∈ R , r ∼ π ˆ r ⇐⇒ r ∼ s,a ˆ r .

The definitions of weak and strong identifiability were introduced in Kim et al. (2021). It follows from Equation (3) that strong identifiability implies almost-strong identifiability, which implies weak identifiability.

## 3. Linear Algebraic Characterizations of Identifiability

In this section, we derive linear algebraic characterizations for the different notions of reward equivalence defined in Section 2.3. The different notions of identifiability are characterized by comparing the corresponding equivalence classes. Throughout this section, we assume that R = R mn .

## 3.1. Policy-Preserving Equivalence

We first recall that the solutions of finite horizon MaxEntRL problems are usually time-varying policies. However, in general not every time-varying policy is a solution to Problem (2) for some reward. Therefore, we first characterize the conditions a time-varying policy should satisfy to be a solution. Given a policy π = ( π t ) T -1 t =0 , we vectorize it as follows:

<!-- formula-not-decoded -->

Furthermore, we define the matrices Γ ∈ R Tmn × ( Tn + mn ) and Ξ ∈ R Tmn as:

<!-- formula-not-decoded -->

with I = I mn , E = [ I n · · · I n ] ⊺ ∈ R nm × n and P = [ P ⊺ a (1) · · · P ⊺ a ( m ) ] ⊺ ∈ R nm × n , where P a ( k ) ∈ R n × n is such that its ij -th entry is given by T ( s ( j ) | s ( i ) , a ( k ) ) , k ∈ { 1 , . . . , m } . Given Γ , we construct Γ Access by only keeping the rows in Γ corresponding to accessible states. Similarly, we construct Ξ Access . Details of this construction is given in Appendix A.1. Observe that Γ Access and Ξ Access have ∑ T -1 t =0 m | Access t | rows, which simplifies to Tnm when all states are accessible at all times. Then, we have the following necessary and sufficient condition for a time-varying policy π to be a solution of Problem (2) for some reward.

Proposition 7 A time-varying policy π = ( π t ) T -1 t =0 solves Problem (2) for some reward if and only if Ξ Access ∈ ran (Γ Access ) .

Proof See Appendix A.1.

We use this result to first characterize the set of rewards that can induce π then derive the finitehorizon policy-preserving equivalence class. To this end, we define the following affine subspace:

<!-- formula-not-decoded -->

Then, the set of rewards r such that π ∈ arg max π J MaxEnt ( π ; r ) , denoted by R , is given by:

<!-- formula-not-decoded -->

where P = [ I mn 0 mn × Tn ] is the projection operator of a mn + Tn dimensional vector onto its first mn components. By defining the following subspace:

<!-- formula-not-decoded -->

we arrive at the following result.

Corollary 8 Given a time-varying policy π = ( π t ) T -1 t =0 and an MDP model, let r be a reward that induces π . Then, the policy-preserving equivalence class of r is

<!-- formula-not-decoded -->

Proof See Appendix A.2.

## 3.2. Weak Trajectory Equivalence and Weak Identifiability

Let K = | support ( µ 0 ) | , which denotes the number of initial states in the MDP. We denote these states by { s ( k ) 0 } K k =1 . Consider { Ω( s ( k ) 0 ) } K k =1 , where each Ω( s ( k ) 0 ) corresponds to the set of all trajectories starting from s ( k ) 0 . For each s ( k ) 0 , we construct the matrix M s ( k ) 0 ∈ R | Ω( s ( k ) 0 ) |× mn as:

<!-- formula-not-decoded -->

where τ ( k ) i ( t ) denotes the state action pair at time step t of the i -th trajectory of Ω( s ( k ) 0 ) , for some arbitrary ordering of trajectories. Using the definition above, we can characterize the weak-trajectory equivalence class of a reward function.

Theorem 9 The weak-trajectory equivalence class for a reward r is given by:

<!-- formula-not-decoded -->

Proof See Appendix A.3.

The following characterization of weak identifiability follows directly from Theorem 9.

Corollary 10 An MDP model with R = R mn is weakly identifiable if and only if

<!-- formula-not-decoded -->

## 3.3. Strong Trajectory Equivalence and Almost-Strong Identifiability

The strong trajectory equivalence class of a reward can be characterized using a similar derivation to that of Section 3.2. To this end, define the matrix M = [ M ⊺ s (1) 0 M ⊺ s (2) 0 · · · M ⊺ s ( K ) 0 ] ⊺ .

Theorem 11 The strong-trajectory equivalence class for a reward r is given by:

<!-- formula-not-decoded -->

Proof See Appendix A.4

Using Theorem 11, we can directly characterize almost-strong identifiability as follows.

Corollary 12 An MDP model with R = R mn is almost-strongly identifiable if and only if

<!-- formula-not-decoded -->

The conditions given in Corollaries 11 and 12 can be computationally expensive to verify since the number of trajectories in a stochastic MDP typically grows exponentially with the horizon length. Hence, storing the matrices ( M s ( i ) 0 ) K i =1 and computing their null-space can quickly become computationally infeasible, even for moderately sized MDPs. This means that verifying weak- and almoststrong identifiability can be prohibitive. However, given our linear algebraic characterizations, we can design an incremental algorithm to mitigate the aforementioned problem. The algorithm is based on the following result for almost-strong identifiability.

Proposition 13 Given an MDP model, let { k 1 , . . . , k r } be a basis for K Γ . Then the MDP model is almost-strongly identifiable if and only if

<!-- formula-not-decoded -->

where M i is the i -th row of M corresponding to the i -th trajectory in Ω .

Proof See Appendix A.5.

Proposition 13 says that we can check for almost-strong identifiability by checking a property for individual trajectories instead of storing a large matrix of trajectories and computing its null-space. The procedure is summarized in Algorithm 1 of Appendix B.1. The same algorithm can be directly adapted to test weak identifiability by running it for each starting state { s ( k ) 0 } K k =1 .

## 3.4. State-Action Equivalence and Strong Identifiability

For state-action equivalence, the following result follows directly from its definition.

Theorem 14 The state-action equivalence class for a reward r is given by:

<!-- formula-not-decoded -->

Strong identifiability can be characterized using this theorem as follows.

Corollary 15 An MDP model with R = R mn is strongly identifiable if and only if

<!-- formula-not-decoded -->

Corollary 15 gives an efficient way to check if an MDP model is strongly identifiable. Indeed, we can (i) compute the accessible states Access , (ii) compute the matrix Γ Access , (iii) compute a basis of its kernel, and (iv) compute the dimension of K Γ . This dimension is one if and only if the MDP model is strongly identifiable. This consists of a polynomial time algorithm to check the strong identifiability of an MDP model. In fact, the computational complexity can be further improved in the fully accessible case as detailed in Appendix C. We note that this is in contrast to the strong identifiability condition in Cao et al. (2021), which is exponential in the horizon T .

## 4. Feature-Based Identifiability

So far, we have studied identifiability of rewards in inverse reinforcement learning for the reward set R = R mn . However, a common assumption in reinforcement learning is that the agent is trying to optimize a reward function that can be expressed as a linear combination of known features. This means that the conditions in Corollaries 10, 12 and 15 can be made tighter, since not every reward function in R can be written as a linear combination of the pre-determined features. Given that features describe a subspace in the reward space, incorporating feature-based rewards into our framework becomes just a matter of intersecting these subspaces with our previous results. In particular, consider a feature function f : S × A → R k . Define the mn × k matrix describing the feature function as F = [ f 1 ( · ) f 2 ( · ) · · · f k ( · ) ] , where f i ( · ) is the i -th feature evaluated at all the state-action pairs. Let R f = { r ∈ R nm |∃ ω ∈ R k s.t. r ( s, a ) = ω ⊺ f ( s, a ) , ∀ ( s, a ) ∈ S × A} be the space of featurized reward functions. We can directly see that r ∈ R f ⇐⇒ r ∈ ran ( F ) .

Moreover, to distinguish the equivalence classes when using R = R f from the ones when R = R mn , we use [ r ] ∼ π,f , [ r ] ∼ τ,f , [ r ] ∼ ω,f and [ r ] ∼ ( s,a ) ,f . As in Section 3.1, where it is stated that not every time-varying policy is induced by a reward, clearly not every time-varying policy is induced by a featurized reward.

Theorem 16 Given a time-varying policy π = ( π t ) T -1 t =0 and an MDP Model, the set of featurized rewards r such that π ∈ arg max π J MaxEnt ( π ; r ) , denoted by R f , is given by:

<!-- formula-not-decoded -->

Proof Follows from the construction of R with the added constraint that r ∈ ran ( F ) .

We note that in Theorem 16, if π is not induced by a featurized reward, then Equation (8) gives the empty set. As in the unconstrained reward case, we can show that the featurized equivalence classes can be derived simply by taking the intersection between the equivalence classes studied in Section 3 with ran ( F ) :

## Theorem 17

<!-- formula-not-decoded -->

Proof Similar to the proofs of Section 3, while noting the new structure of R f .

Equation (9) reveals that if ran ( 1 mn ) ⊆ ran ( F ) , then [ r ] ∼ ( s,a ) ,f = [ r ] ∼ s,a . Otherwise, [ r ] ∼ ( s,a ) ,f = { r } . That is, if the vector of ones is not in the range of the feature matrix, it might be possible to exactly identify a unique reward in the featurized setting. Moreover, the results in Theorem 17 are not restricted to rewards constrained to subspaces via features but can easily be generalized to arbitrary reward sets R by taking the intersection with R instead of ran ( F ) .

## 5. Numerical Experiments

In this section, we test our framework on different grid world examples with different dynamics. The code to generate the results is available at https://github.com/mlshehab/learning\_ true\_objectives .

## 5.1. Unconstrained Reward Functions

We demonstrate our framework on three versions of a 5 by 5 grid world shown in Figure 1. The four possible actions available for the agent are: UP,DOWN,LEFT,RIGHT . Each action succeeds with a probability 0 . 9 , and with probability 0 . 1 the agent moves randomly to one of the 4 neighboring cells or stays in the same cell. The first grid world, shown in Figure 1( a ), is the original grid world where all transitions are admissible. In the second grid world, shown in Figure 1( b ), we introduce a strip blocking (denoted by the dashed line and red area) that the agent can not enter from outside, but can escape if started inside. Note that all actions are still available at all states, but the outcome of a blocked action is uniformly distributed over the available neighboring cells. Lastly, we introduce a wall in the grid world of Figure 1( c ) which forces the only possible transition on the left column to be upward. For example, if the agent starts at the lower left corner, then the only way they can reach the right side of the grid world is by first traveling along the left border until the blocking is

Figure 1: Three grid worlds considered in this section: (a) the original grid world with no blocking, (b) the red strip is blocked from outside, and (c) the thick line only blocks transitions from the left side.

<!-- image -->

cleared. We take the horizon length to be 15 and the initial distribution to be a single starting state; results with varying horizon lengths and starting states are given in Appendix B.2. For the MDP described by Figure 1( a ), with any starting state, we find that K Γ = ran ( 1 mn ) , which means that the MDP model is strongly identifiable. We note that if we remove self-transitions, the MDP model is not strongly identifiable anymore. On the other hand, we get that dim( K Γ ) &gt; 1 for the MDPs of Figures 1( b ) and 1( c ), with a starting state inside the blocking and on the bottom left corner respectively, and hence both are not strongly identifiable. We observe that the subspace K Γ is along the states in the red strip in Figure 1( b ) and along the states on the left most wall of Figure 1( c ), meaning that we can arbitrarily change the reward at these states and still induce the same optimal policy. Additional results with weak and almost-strong identifiability are given in Appendices B.3 and B.4.

## 5.2. Featurized Reward Functions

In this section, we show how prior information, in the form of featurized rewards, can improve identifiability. Consider a scenario where the rewards depend on landmarks in a grid world and we want to place the landmarks in a way to understand how much agents value different landmarks. In particular, we present four such cases in Figures 2( a ), 2( b ), 2( c ) and 2( d ), where the important landmarks are a burger joint and a vehicle charging station. We denote the two landmarks by l 1 and l 2 . The feature function f : S × A → R 2 is given by f i ( s, a ) = -manhattan distance ( s, l i ) , ∀ a ∈ A . F is constructed by stacking the feature function values for all state-action pairs. We report the results with a horizon of 15 and the varying horizon results are given in Appendix B.2. The starting state is the lower left corner. Our framework shows that the any placement of the landmarks, e.g. Figures 2( a ), 2( b ), 2( c ) and 2( d ), makes the MDP model strongly identifiable . In particular, we find that K Γ ∩ ran ( F ) = ran ( 1 mn ) for Figure 2( a ). For Figures 2( b ), 2( c ) and 2( d ), we find that K Γ ∩ ran ( F ) = 0 . Since ran ( 1 mn ) ̸⊆ ran ( F ) for all these placements, we conclude that the true reward function can be exactly recoverable. Sparse feature results are given in Appendix B.5.

## 6. Related Works

Here we compare our results with some recent work on the reward ambiguity problem of IRL. In their work, Cao et al. (2021) derive necessary and sufficient conditions for strong-identifiability in infinite and finite horizons. For finite horizon, they characterize strong identifiability in terms of

Figure 2: The blocked grid world of Figure 1( c ) with features. The colored cells denote the position of important landmarks.

<!-- image -->

the properties of 'full-action rank' and 'full access'. Our work builds on Cao et al. (2021) by first deriving explicitly the set of rewards inducing a policy. Additionally, we demonstrate how a linear algebraic characterization enables a polynomial-complexity test for strong identifiability and extends to different notions of identifiability. Rolland et al. (2022) extend the work of Cao et al. (2021) to find linear algebraic characterizations for strong-identifiability in infinite horizon settings. Amin et al. (2017) studied how access to sequential tasks could enhance identifiability and reduce the mismatch between the demonstrator's objective and the learned reward function. However, these previous works assume access either to demonstrations of the agents in multiple sufficiently distinct environments, or multiple tasks. Instead, our work presents unified necessary and sufficient conditions for weak and strong identifiability (with and without features) using the policy in one single environment. Schlaginhaufen and Kamgarpour (2023) also derive a linear algebraic characterization of strong identifiability in the infinite horizon constrained MDP setting. The major commonality between these prior works is assuming an infinite horizon setting, for which the optimal policy is known to be stationary and thus simplifies the analysis. Kim et al. (2021) studied identifiability using the notions of weak and strong identifiability. However, their necessary and sufficient conditions for strong identifiability requires the MDP model to be weakly identifiable, for which a means of verification was not presented except for deterministic MDPs. Our results allow verifying weak identifiability for any MDP. Finally, Skalse et al. (2023) generalize most of the previous works by characterizing transformations on the rewards that preserve optimality under different RL objectives. Our work is complementary to theirs by focusing on MaxEntRL objective and extracting computable linear algebraic characterizations for different equivalence classes.

## 7. Conclusion

In this work, we established linear algebraic characterizations of weak-, almost-strong, and strongidentifiability of MDPs. Our numerical examples illustrate how these new theoretical results can be leveraged to choose features making the underlying MDP identifiable. In the future, we will build on this approach to design identifiability preserving abstractions. Finally, we will investigate the problem of reward identifiability from a finite set of expert trajectories, instead of knowing the exact expert policy.

## Acknowledgments

Toyota Research Institute ('TRI') provided funds to assist the authors with their research but this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity. MLS and NO were also supported in part by NSF grants CNS-1931982 and CNS-1918123.

## References

- Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In International Conference on Machine Learning , 2004.
- Kareem Amin, Nan Jiang, and Satinder Singh. Repeated inverse reinforcement learning. Advances in Neural Information Processing Systems , 30, 2017.
- Saurabh Arora and Prashant Doshi. A survey of inverse reinforcement learning: Challenges, methods and progress. Artificial Intelligence , 297, 2021.
- Zoe Ashwood, Aditi Jha, and Jonathan W Pillow. Dynamic inverse reinforcement learning for characterizing animal behavior. Advances in Neural Information Processing Systems , 35:2966329676, 2022.
- Monica Babes, Vukosi Marivate, Kaushik Subramanian, and Michael L Littman. Apprenticeship learning about multiple intentions. In International Conference on Machine Learning , pages 897-904, 2011.
- Abdeslam Boularias, Jens Kober, and Jan Peters. Relative entropy inverse reinforcement learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics , pages 182-189, 2011.
- Haoyang Cao, Samuel Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. Advances in Neural Information Processing Systems , 34:12362-12373, 2021.
- Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adverserial inverse reinforcement learning. In International Conference on Learning Representations , 2018.
- Erik Jenner and Adam Gleave. Preprocessing reward functions for interpretability. Advances in Neural Information Processing Systems Workshop on Cooperative AI , 2021.
- Kuno Kim, Shivam Garg, Kirankumar Shiragur, and Stefano Ermon. Reward identification in inverse reinforcement learning. In International Conference on Machine Learning , pages 54965505, 2021.
- Dan Li, Mohamad Louai Shehab, Zexiang Liu, Nikos Ar´ echiga, Jonathan DeCastro, and Necmiye Ozay. Outlier-robust inverse reinforcement learning and reward-based detection of anomalous driving behaviors. In 25th International Conference on Intelligent Transportation Systems (ITSC) , pages 4175-4182, 2022.
- Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, and Marcello Restelli. Provably efficient learning of transferable rewards. In International Conference on Machine Learning , pages 7665-7676. PMLR, 2021.

- Andrew Y Ng and Stuart Russell. Algorithms for inverse reinforcement learning. In International Conference on Machine Learning , 2000.
- Deepak Ramachandran and Eyal Amir. Bayesian inverse reinforcement learning. In International Joint Conferences on Artificial Intelligence , volume 7, pages 2586-2591, 2007.
- Giorgia Ramponi, Amarildo Likmeta, Alberto Maria Metelli, Andrea Tirinzoni, and Marcello Restelli. Truly batch model-free inverse reinforcement learning about multiple intentions. In International conference on artificial intelligence and statistics , pages 2359-2369. PMLR, 2020.
- Nathan D Ratliff, J Andrew Bagnell, and Martin A Zinkevich. Maximum margin planning. In International Conference on Machine Learning , pages 729-736, 2006.
- Paul Rolland, Luca Viano, Norman Sch¨ urhoff, Boris Nikolov, and Volkan Cevher. Identifiability and generalizability from multiple experts in inverse reinforcement learning. Advances in Neural Information Processing Systems , 35:550-564, 2022.
- Andreas Schlaginhaufen and Maryam Kamgarpour. Identifiability and generalizability in constrained inverse reinforcement learning. In International Conference on Machine Learning , 2023.
- Joar Max Viktor Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, and Adam Gleave. Invariance in policy optimisation and partial identifiability in reward learning. In International Conference on Machine Learning , pages 32033-32058, 2023.
- Markus Wulfmeier, Peter Ondruska, and Ingmar Posner. Deep inverse reinforcement learning. CoRR, abs/1507.04888 , 2015.
- Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum entropy inverse reinforcement learning. In AAAI Conference on Artificial Intelligence , volume 8, pages 14331438, 2008.

## Appendix A. Proofs

## A.1. Proof of Proposition 7

We build on the following result adapted from (Cao et al., 2021) by setting the terminal reward to zero.

Lemma 18 For any time-varying policy π = ( π t ) T -1 t =0 , and for any function ν : { 0 , . . . , T -1 } × S → R , the reward function given by

<!-- formula-not-decoded -->

with ν T = 0 , is the only reward function for which π is the optimal solution of (2) with value function ν .

Lemma 18 describes implicitly all the possible reward functions for which a given policy π is optimal. Since Equation (10) is linear in r and ν for all t in [0 , T -1] , we can construct a linear system of equations which the reward and value function have to satisfy in order to induce a given policy π . This allows us to explicitly describe all the possible rewards inducing π . Proposition 7 is essentially doing this by also taking the ambiguities due to (in)accessibility into account.

Proof [of Proposition 7] We create vectorized versions of the reward and value function as:

<!-- formula-not-decoded -->

Equation (10) gives necessary and sufficient conditions that a reward and value have to satisfy in order to induce a given time-varying policy π . Using the definitions of I , E , P and π log t , we can write the equation as:

<!-- formula-not-decoded -->

If a state i is not accessible at time t , we delete all its corresponding rows from Equation (11). The indices of the deleted rows are I = { nl + i | l = 0 , · · · , m -1 } . This amounts to deleting m rows for each inaccessible state, corresponding to m state-action pairs that have no constraint at time step t due to the state not being accessible. Finally, a reward r and value function ν satisfying this equation exist if and only if Ξ Access ∈ ran (Γ Access ) , which concludes the proof.

## A.2. Proof of Corollary 8

Proof We first prove the ⊆ direction. Let r, ˆ r be two rewards that induce the same time-varying policy π . By Equation (5), we know that:

<!-- formula-not-decoded -->

Since x, ˆ x ∈ X , then x -ˆ x ∈ ker (Γ Access ) . Thus, r -ˆ r = P ( x -ˆ x ) ∈ K Γ , hence we have [ r ] ∼ π ⊆ r ⊕K Γ . For the ⊇ direction, let r be a reward inducing π and let ˆ r = r + v, v ∈ K Γ . Since

r ∈ R , then there exists x ∈ X such that Γ Access x = Ξ Access and r = P x . Define ˆ x as:

<!-- formula-not-decoded -->

Then Γ Access ˆ x = Γ Access x +Γ Access η = Ξ Access , hence ˆ x ∈ X and ˆ r = P ˆ x , so ˆ r ∈ R . Thus ˆ r ∈ [ r ] ∼ π , and thus r ⊕K Γ ⊆ [ r ] ∼ π .

## A.3. Proof of Theorem 9

We make use of the following lemma for general subspaces S i and a vector r :

## Lemma 19

concluding the proof.

<!-- formula-not-decoded -->

Proof We proceed by proving inclusion in both directions:

⊆ : Let v ∈ ⋂ i =1 ,...,K r ⊕ S i . Then, ∀ i, v ∈ r ⊕ S i . It follows that for every i , there exists s i such that v = r + s i . Hence, v -r = s i and then v -r ∈ S i for all i . Consequently, v -r ∈ ⋂ i =1 ,...,K S i and it follows that v = r + s , with s ∈ ⋂ i =1 ,...,K S i .

⊇ : Let v ∈ r ⊕ ⋂ i =1 ,...,K S i . Then, v = r + s, s ∈ ⋂ i =1 ,...,K S i . Hence, for all i , s ∈ S i . Thus, for all i , v ∈ r ⊕ S i which leads to v ∈ ⋂ i =1 ,...,K r ⊕ S i , concluding the proof.

Now, we can prove Theorem 9.

Proof [of Theorem 9] Let r be a reward in R . Using Definition 3 and Equation (7), a reward ˆ r is weak-trajectory equivalent to r if and only if for all k = 1 , . . . , K , there exists c k ∈ R such that

<!-- formula-not-decoded -->

Using M s ( k ) 0 1 mn = ( ∑ T -1 t =0 γ t ) 1 | Ω( s ( k ) 0 ) | and defining ˜ c k = c k / ∑ T -1 t =0 γ t , Equation (12) can be rewritten M s ( k ) 0 ( r -ˆ r -˜ c k 1 mn ) = 0 . This holds for some ˜ c k if and only if ˆ r ∈ r ⊕ ran (1 mn ) ⊕ ker ( M s ( k ) 0 ) . Since this must hold for all k = 1 , . . . , K , it gives

<!-- formula-not-decoded -->

Using Lemma 19, this can be rewritten as

<!-- formula-not-decoded -->

## A.4. Proof of Theorem 11

Proof Let r be a reward. Using Definition 4 and the definition of the matrix M , a reward ˆ r is strong-trajectory equivalent to r if and only if it exists c ∈ R such that

<!-- formula-not-decoded -->

Defining ˜ c = c/ ∑ T -1 t =0 γ t , and using M (˜ c 1 mn ) = c 1 | Ω | , Equation (13) can be rewritten M ( r -ˆ r -˜ c 1 mn ) = 0 . Such a ˜ c exists if and only if ˆ r ∈ r ⊕ ran ( 1 mn ) ⊕ ker ( M ) .

## A.5. Proof of Proposition 13

Proof Let { k 1 , . . . , k r } be a basis for K Γ . Then:

<!-- formula-not-decoded -->

which concludes the proof.

## Appendix B. Algorithmic Details and Additional Examples

## B.1. Test of Almost-Strong Identifiability

In Algorithm 1, we present an incremental procedure for testing almost-strong identifiability. The same algorithm can be adapted to test weak-identifiability by running it for each starting state { s ( k ) 0 } K k =1 and making sure the output is 1 for all starting states. We only have to keep track of the variables ( ξ j ) r j =1 , and compute the state-visitation row of a trajectory at each time step.

## B.2. Results with Varying Horizon Length and Starting States

In this section, we show the effect of horizon length and starting state on strong identifiability results. Changing the starting state and horizon essentially changes Access , yielding different identifiability results for different start state/horizon combinations. We generally expect longer horizons and starting states with larger accessible sets to result in better identifiability. In Figures 3( a ), 3( b ) and 3( c ), we show these changes for the examples of Section 5.1. In particular, we plot the dimension of K Γ with varying horizons for different starting states. Since 1 mn ∈ K Γ , we can equivalently say that an MDP model is strongly identifiable if, and only if, dim ( K Γ ) = 1 . Wenotice that the MDP model is strongly identifiable for all starting states in Figure 1( a ) beyond a horizon of 9 . For Figures 1( b ) and 1( c ), starting states that are most covering (i.e., states 7 and 4) yield the best identifiability results beyond horizons 7 and 13 . In Table 1, we show the results for those of Section 5.2.

̸

```
Algorithm 1: Test of Almost-Strong Identifiability Input: basis for K Γ : { k i } r i =1 Output: 1 , if MDP model is almost-strongly identifiable, 0 otherwise. 1 τ 1 ← any starting trajectory 2 r 1 ← corresponding row of τ 1 in M, constructed using (7) 3 for j ← 1 to r do 4 ξ j ← r ⊺ 1 k j 5 end 6 for each trajectory τ i do 7 r i ← corresponding row of τ i in M, constructed using (7) 8 for j ← 1 to r do 9 if r ⊺ i k j = ξ j then 10 return 0 11 end 12 end 13 end 14 return 1
```

Table 1: Identifiability with dense features from different initial states and different horizon lengths for the grid world of Figure 1( c ). strong: strongly identifiable , not strong: not strongly identifiable , exact: exactly identifiable .

|                    | Identifiability Status   | Identifiability Status   | Identifiability Status   | Identifiability Status   |
|--------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| Landmarks Location | Starting State = 4       | Starting State = 4       | Starting State = 4       | Starting State = 15      |
|                    | T ∈ [1 , 5]              | T ∈ [6 , 7]              | T ∈ [8 , 20]             | T ∈ [1 , 20]             |
| (0, 24)            | not strong               | strong                   | strong                   | strong                   |
| (3, 1)             | not strong               | not strong               | exact                    | exact                    |
| (3, 19)            | not strong               | exact                    | exact                    | exact                    |
| (14, 14)           | not strong               | exact                    | exact                    | exact                    |

## B.3. Example of Weakly Identifiable But Not Strongly Identifiable

Before giving out an illustrative example of an MDP model that is weakly identifiable but not strongly identifiable, it is worth mentioning the following remark.

̸

Remark 20 Given Equation (11) and the fact that the reward at the last time step is given by r = ¯ π log t + E ν T -1 , we can show that E ker ( P ) ⊆ K Γ . Thus, if ker ( P ) ̸⊆ ran ( 1 n ) , the MDP model is not strongly identifiable. Since P1 n = 1 mn , the previous condition is equivalent to ker ( P ) = { 0 } . Hence, we can equivalently say that an MDP model is strongly identifiable only if P is full rank.

Given our linear algebraic characterizations, it is possible to come up with examples that are weakly identifiable, but not strongly identifiable. For example, consider an MDP with 3 states ( s 1 , s 2 , s 3 )

Dimension of Kr

60

10

12

Horizon

(a) Original start state = 0

start state = 4

start state = 7

start state = 12

start state = 15

start state = 20

start state = 24

min dim = 1

* I

start state = 0

## LEARNING TRUE OBJECTIVES

start state = 12

start state = 15

start state = 20

start state = 24

min dim = 14

Figure 3: The three grid worlds of Figure 1 with varying horizons and varying starting states. The starting state numbering is such that the state on the top left is 0 , and increases by 1 going south, and by 5 going east.

<!-- image -->

and 2 actions a 1 , a 2 . s 1 is the only starting state. Assume that the transition matrices are given by:

<!-- formula-not-decoded -->

With a horizon of 2, the trajectory matrix M is given by:

<!-- formula-not-decoded -->

We can directly see that ker ( M ) = ran ( e 2 , e 5 ) , where e i is the i -th euclidean vector in R 6 . By constructing Ψ , we get that K Γ = ran ( e 1 + e 3 + e 4 + e 6 , e 2 + e 5 ) . Thus, E K Γ ⊆ ran ( 1 6 ) ⊕ ker ( M ) , meaning that the MDP model is weakly identifiable (also follows from Kim et al. (2021) since

Dimension of Kr

deterministic MDPs are weakly identifiable). However, since P is not full-rank, the MDP model is not strongly identifiable using Remark 20.

## B.4. Weak and Almost-Strong Identifiability Results

We run Algorithm 1 on the MDP models in Figures 1( a ), 1( b ) and 1( c ). We take the horizon length to be 15. We get:

- The MDP model of Figure 1( a ) is strongly identifiable , and thus it is trivially both weakly identifiable and almost-strongly identifiable.
- The MDP model of Figure 1( b ) is almost-strongly identifiable if the set of starting states is completely outside the blocking, or a single state inside the blocking. It is weakly identifiable for any set of starting states. We note that the basis of K Γ is exactly along the states in the red strip (meaning that we can arbitrarily change the reward at these states and still induce the same optimal policy). This means that if the starting state is outside the red strip, all trajectories never visit these blocked states, and thus the inner product in line 9 of Algorithm 1 stays the same (in fact, equals 0). Also, if a trajectory starts inside the red strip, it has to leave in 1 step and can not re-enter, and thus again the value of line 9 stays the same. If we allow transitions inside the strip blocking, then the MDP model becomes strongly identifiable .
- Similarly, the MDP model of Figure 1( c ) is almost-strongly identifiable if the set of starting states is right of the wall, or a single state on the left column. It is weakly identifiable for any set of starting states. The same reasoning as Figure 1( b ) applies, since the basis of K Γ is exactly along the states on the left-most wall.

## B.5. Sparse Feature Function Setting

To highlight the importance of the feature function, we consider a more sparse feature function given by f i ( s, a ) = 1 , ∀ a ∈ A if s = l i , and 0 otherwise. We consider the MDP model of Figure 1( c ). We find that with this feature function, if we place any of the burger joint or the charging station on the left most column, the MDP model is not strongly identifiable. However, we are free to place them at any position on the right of the thick wall and obtain an exactly identifiable MDP model with a sufficiently long horizon. This means that if the underlying reward function of agents is a linear combination of these sparse features, then placing any of the burger joint or the charging station on the left-most column is not ideal since we can not disambiguate which landmark the agent prefers. Detailed results with varying horizons are given in Table 2.

## Appendix C. Fully Accessible Case

In this section, we derive a closed form for K Γ in the case where Access t = S for t = 0 , · · · , T -1 . This allows us to directly derive interpretable sufficient conditions that the MDP model has to satisfy in order to be strongly identifiable. We argue that full accessibility is a necessary condition for knowing the policy π everywhere, which is the assumption in Cao et al. (2021). The first result allows us to write K Γ in a more compact form.

Lemma 21 Let K Γ be defined as in Equation (6). Then

<!-- formula-not-decoded -->

Table 2: Identifiability with sparse features from different initial states and different horizons lengths for the grid world of Figure 1( c ). strong: strongly identifiable , not strong: not strongly identifiable , exact: exactly identifiable .

|                    | Identifiability Status   | Identifiability Status   | Identifiability Status   | Identifiability Status   | Identifiability Status   |
|--------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| Landmarks Location | Starting State = 4       | Starting State = 4       | Starting State = 4       | Starting State = 15      | Starting State = 15      |
|                    | T ∈ [1 , 10]             | T ∈ [11 , 12]            | T ∈ [13 , 20]            | T ∈ [1 , 5]              | T ∈ [6 , 20]             |
| (0, 24)            | not strong               | not strong               | exact                    | not strong               | exact                    |
| (3, 1)             | not strong               | not strong               | not strong               | not strong               | not strong               |
| (3, 19)            | not strong               | not strong               | not strong               | not strong               | not strong               |
| (14, 14)           | not strong               | exact                    | exact                    | not strong               | exact                    |

where

<!-- formula-not-decoded -->

Proof E S ⊆ K Γ : Let x T -1 ∈ S . We want to show that E x T -1 ∈ K Γ . Since x T -1 ∈ S , then there exists x T -2 ∈ R n such that E x T -2 = γ P x T -1 . Given that E † P = L (where E † denotes the pseudo-inverse of E ), we can write x T -2 = γ L x T -1 . This gives that P x T -2 = γ PL x T -1 , combined with x T -1 ∈ S , means that there exists x T -3 ∈ R n such that E x T -3 = γ P x T -2 , resulting in x T -3 = γ 2 L 2 x T -1 . Repeating the same process, we can construct ( x t ) T -1 t =0 satisfying:

<!-- formula-not-decoded -->

Now, construct the vector k = [ r ⊺ ν ⊺ 0 ν ⊺ 1 · · · ν ⊺ T -1 ] ⊺ where:

<!-- formula-not-decoded -->

Then, we can verify that:

<!-- formula-not-decoded -->

Then, k ∈ ker (Γ) and thus r = P k ∈ K Γ . Since r = E ν T -1 = E x T -1 , we conclude that E x T -1 ∈ K Γ .

K Γ ⊆ E S : Let r ∈ K Γ . We want to prove that r ∈ E S , i.e., r = E x for some x ∈ S . Since r ∈ K Γ , then there exists k ∈ ker (Γ) such that r = P k . The vector k can be written as [ r ⊺ ν ⊺ 0 ν ⊺ 1 · · · ν ⊺ T -1 ] ⊺ , with r and ( ν t ) T -1 t =0 satisfying conditions (15). Define x T -1 = ν T -1 and x t = ν t -∑ T -1 i = t +1 x i , t ∈ [0 , T -2] . Then E x t = γ P x t +1 for all t ∈ [0 , T -2] , yielding x T -1 ∈ S . Since r = E x T -1 , we conclude that r ∈ E S .

We also make use of the following lemma.

Lemma 22 Let x ∈ R n . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Proof Let x ∈ R n . Then:

<!-- formula-not-decoded -->

Finally, we can write K Γ compactly as follows.

Proposition 23 Let L and D be defined as in Equations (14) and (16) respectively. Then:

<!-- formula-not-decoded -->

Proof Follows directly from Lemmas 21 and 22 by noting that PL t x ∈ ran ( E ) ⇐⇒ M t x ∈ ker ( D ) ⇐⇒ x ∈ ker ( DM t ) .

The main implication of Proposition 23 is that checking the necessary and sufficient condition for strong identifiability in MDP models can be done by computing the kernel of a Tmn by n matrix as compared to Γ , which is Tmn by mn + Tn . We can directly arrive at the following results:

̸

Corollary 24 Assume γ = 0 . If any of the following conditions is true:

1. There exists t ≥ 0 such that rank ( D L t ) = n -1 ,

̸

2. There exists two actions a i ∈ A , a j ∈ A , i = j such that rank ( [ P a i -P a j ] ) = n -1 ,

Then the MDP model is strongly identifiable for all horizons T ≥ T ∗ (where T ∗ = t + 1 for the first condition, and T ∗ = 1 for the second).

Proof Follows directly from the closed form of K Γ given by Proposition 23.

̸

Remark 25 A particular case of Corollary 24 is that a fully accessible MDP model is strongly identifiable for all horizons T ≥ 1 if rank ( D ) = n -1 . Interestingly, the same condition on D is required in order to identify a reward function up to a constant by observing an expert act in two identical MDP models with only different discount factors γ 1 = γ 2 [Rolland et al. (2022), Corollary 5]. It is also equivalent to the condition for identification of an action-independent reward from a single expert [Cao et al. (2021), Corollary 3].