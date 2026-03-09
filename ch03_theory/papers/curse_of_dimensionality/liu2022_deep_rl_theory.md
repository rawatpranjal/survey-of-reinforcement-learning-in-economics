## Understanding Deep Neural Function Approximation in Reinforcement Learning via /epsilon1 -Greedy Exploration

## Fanghui Liu ∗ , Luca Viano, Volkan Cevher

Laboratory for Information and Inference Systems

École Polytechnique Fédérale de Lausanne (EPFL), Switzerland {first}.{last}@epfl.ch

## Abstract

This paper provides a theoretical study of deep neural function approximation in reinforcement learning (RL) with the /epsilon1 -greedy exploration under the online setting. This problem setting is motivated by the successful deep Q-networks (DQN) framework that falls in this regime. In this work, we provide an initial attempt on theoretical understanding deep RL from the perspective of function class and neural networks architectures (e.g., width and depth) beyond the 'linear' regime. To be specific, we focus on the value based algorithm with the /epsilon1 -greedy exploration via deep (and two-layer) neural networks endowed by Besov (and Barron) function spaces, respectively, which aims at approximating an α -smooth Q-function in a d -dimensional feature space. We prove that, with T episodes, scaling the width m = ˜ O ( T d 2 α + d ) and the depth L = O (log T ) of the neural network for deep RL is sufficient for learning with sublinear regret in Besov spaces. Moreover, for a two layer neural network endowed by the Barron space, scaling the width Ω( √ T ) is sufficient. To achieve this, the key issue in our analysis is how to estimate the temporal difference error under deep neural function approximation as the /epsilon1 -greedy exploration is not enough to ensure 'optimism'. Our analysis reformulates the temporal difference error in an L 2 (d µ ) -integrable space over a certain averaged measure µ , and transforms it to a generalization problem under the non-iid setting. This might have its own interest in RL theory for better understanding /epsilon1 -greedy exploration in deep RL.

## 1 Introduction

Efficient reinforcement learning (RL) under the large (or even infinite) state space and action space setting is increasingly important and relevant challenge [1, 2, 3]. One of the first successful approaches towards this problem is the deep Q-network (DQN) [4, 5] framework, which deploys powerful nonlinear function approximation techniques via Deep Neural Networks (DNNs) [6] to concisely approximate state and action spaces. Despite its impressive practical success, there is still a gap between practical uses and theoretical understanding on deep RL with regard to the function class and the employed /epsilon1 -greedy policy.

In the perspective of function class, many theoretical works center around linear function approximation [7, 8] and linear mixtures [9, 10]. Existing non-linear function approximation results on RL are largely based on neural tangent kernel (NTK) [11, 12], Bellman rank [13, 14], and Eluder dimension [15, 16, 17]. Nevertheless, these approaches fail in truly capturing the highly non-linear properties of deep RL. For example, NTK (or lazy training [18]) essentially works in a 'linear' regime [19, 20, 21], and can not efficiently learn even a single ReLU neuron [22, 23, 24] as it requires Ω( ε -d ) samples to achieve ε approximation error, where d is the (original) or transformed

∗ Correspondence to: Fanghui Liu &lt;fanghui.liu@epfl.ch&gt; and Luca Viano &lt;luca.viano@epfl.ch&gt; .

feature dimension input; the Bellman rank is normally difficult to be estimated for neural networks as suggested by [25]; the Eluder dimension is at least in an exponential order [26, 27] even for twolayer neural networks. The above general function approximation schemes appear difficult to fully demonstrate the success of practical deep RL both theoretically and empirically.

In the perspective of exploration schemes, DQN is directly equipped with the /epsilon1 -greedy policy instead of confidence-bound based scheme that are commonly used in RL theory. The /epsilon1 -greedy exploration is theoretically demonstrated to have exponential sample complexity in the worst case [28] but is still popular in practical deep RL due to its simple implementation. In this case, theoretical analyses of /epsilon1 -greedy in deep RL are still required. Besides, to ensure a sublinear regret, under the NTK regime, the width of neural networks is required to be m = Ω( T 13 ) [12], where T is the number of episodes. This does not match deep RL in practice with small width/depth under large episodes [4, 29].

To bridge the large theory-practice gap, we study the value iteration algorithm with deep neural function approximation and the /epsilon1 -greedy policy under the online setting, which broadly captures the key features of DQN. Our analysis framework is based on DNNs (as well as two-layer neural networks) where the target Q function lies in the Besov space [30] or the Barron space [31], respectively. These function classes can fully capture the properties of Q-functions, e.g., smoothness by neural networks. Our results demonstrate that the sublinear regret can be achieved for deep neural function approximation under the /epsilon1 -greedy exploration with reasonably finite width and depth in practice. Besides, the relationship between the problem-dependent smoothness of Q-function and regret bounds is also developed. These results could also motivate practitioners to consider different architectures of implementations of deep RL.

## 1.1 Technical challenges and contributions

Most previous RL theory results on function approximation in the online setting work with 'optimism in the face of uncertainty' principle for exploration, leading to a series of upper confidence bound (UCB)-type algorithms to ensure the temporal difference (TD) error smaller than zero.

Conceptually, optimism is sometimes too aggressive and UCB-style algorithms can suffer exponential sample complexity even for nonlinear bandits [27]. Technically, UCB-type algorithms in linear/kernel function approximation [7, 12, 32] depend on a known feature mapping or the NTK kernel, which appears invalid for deep neural function approximation beyond the 'linear' regime. This is because, the used confidence ellipsoid and elliptical potential lemma are not applicable for data-dependent feature mapping of DNNs. To avoid explicitly designing a bonus function, Thompson sampling [33, 34] appears promising in a Bayesian perspective by using randomized (i.e., perturbed) versions of the estimated model or value function [35]. Nevertheless, the bonus function is still implicitly included in confidence estimate of perturbations.

In this work, we center around deep neural function approximation with the /epsilon1 -greedy exploration. Since this exploration scheme is not enough to ensure the TD error smaller than zero, the technical challenge in our analysis is how to estimate it to ensure the sublinear regret. In our proof framework, by a measure transform, the TD error is analysed in an L 2 (d¯ µ ) -integrable space, where ¯ µ is the averaged measure wrt a mini-batch of historical state-action pairs. To break the dependence between the episodes for neural networks training, we utilize the experience replay scheme [36] from DQN, and then transform the TD error estimation to generalization error under the independent but non-identically distributed data setting and approximation error in the respective function spaces. Note that in practice, experience replay makes observations to be (nearly) iid, but our analysis only requires the independence of observations, that is weaker than iid. Such generalization problem can be addressed by uniform convergence via (local) Rademacher complexity of the Besov/Barron spaces under the averaged measure. This considered function spaces in this work is more general than Hölder spaces used in offline RL [37].

Our results show that ( i ) the problem-dependent smoothness of Q-function affects the efficiency of learning with deep RL, which can be improved by increasing the model capacity (width and depth). We use α as a parameter indicating the smoothness degree of Q-function. A larger α indicates smoother functions, easier RL tasks, and smaller exploration times, which coincides with our theory. ( ii ) for deep neural networks under the Besov space, the width m = ˜ O ( T d 2 α + d ) and the depth L = ˜ O (1) are enough for sublinear regret under the /epsilon1 -greedy policy, where ˜ O ( · ) omits the log terms. ( iii ) for two-layer neural networks under the Barron space, the width m = Ω( √ T ) suffices to

ensure sublinear regret. Furthermore, our regret bounds can be independent of the feature dimension, supporting the premise of practical, high-dimensional data in RL.

## 1.2 Related work

Recent work on neural network function approximation beyond NTK (or the Eluder dimension) mainly restrict on the generative setting [25, 38] by assuming a simulator in which the agent can require any state and action, and the offline setting [37, 39]. In sequel, we review RL with function approximation under the online setting that DQN falls into this regime. We also mention that, theoretical understanding of DQN can be conducted by from the perspective of neural fitted Q-iteration algorithm [40, 37, 41], and Q learning [42] in the perspective of understanding the target network [43] and experience replay [44, 45, 46] with linear function approximation. Note that, for notational consistency with previous work, in this subsection, T denotes the total number of steps (i.e., interactions with the environment) instead of the number of episodes in our paper.

RL with linear/kernel function approximation: RL with linear function approximation achieves a sublinear regret bound with ˜ O ( √ d 3 H 3 T ) under a low-rank MDP in a model-free setting [7] and ˜ O ( dH 2 √ T ) in a model-based setting [32], where H is the length of each episode. The regret can be improved to ˜ O ( dH √ T ) under a low inherent Bellman error by assuming a global planning oracle [47] or under a Bernstein-type exploration bonus and controlling extra uniform convergence cost [48]. This nearly optimal regret can be also achieved under the linear mixtures setting [10]. In the kernel regime, the regret can be achieved with ˜ O ( δ F √ H 3 T ) [32, 12], where δ F is the intrinsic complexity (e.g., effective dimension) of the function class RKHS F . The above bounds are based on confidence ellipsoid to quantify the uncertainty in an explicit bonus function by feature mapping/kernel function; while Thompson sampling [34, 33] utilizes an implicit bonus function in probability estimation on uncertainty quantification, which leads to an ˜ O ( d 2 H 2 √ T ) [35] regret in linear function approximation.

Overall, the above metrics are difficult to the nonlinear spaces of DNNs beyond 'linear' regime that concern us.

RL with general function approximation: One prototypical scheme uses the Eluder dimension [15], which measures the degree of dependence among action rewards, resulting in an ˜ O ( poly ( δ F H ) √ T ) regret [16, 17], where the complexity δ F depends on the Eluder dimension. Using this metric, the sublinear regret under the /epsilon1 -greedy exploration can be achieved by [49]. Besides, the low Bellman rank assumption [13], where the Bellman error 'matrix' admits a low-rank factorization, can be also used general function approximation [14] by measuring the error of the function class under the Bellman operator. Combining Bellman rank and Eluder dimension results in a new metric, Bellman Eluder dimension [50], achieving ˜ O ( H √ δ F T ) -regret, where δ F depends on this metric.

## 2 Background and preliminaries

In this section, we introduce the necessary background and definitions with respect to online reinforcement learning based on episodic Markov decision processes (MDPs) and function spaces of deep (and two-layer) ReLU neural networks.

Notation: We denote by a ( n ) /lessorsimilar b ( n ) : there exists a positive constant c independent of n such that a ( n ) /lessorequalslant cb ( n ) ; a ( n ) /equivasymptotic b ( n ) : there exists two positive constant c 1 and c 2 independent of n such that c 1 b ( n ) /lessorequalslant a ( n ) /lessorequalslant c 2 b ( n ) . We use the shorthand [ n ] := { 1 , 2 , . . . , n } for some positive n and /ceilingleft x /ceilingright denotes the smallest integer exceeding x . Let X = [0 , 1] d be a domain of the functions, we denote the L p -integrable space by L p ( X ) endowed by the norm ‖ f ‖ L p ( X ) = ( ∫ X | f ( x ) | p d x ) 1 /p , and the µ -integrable L p space by L p (d µ ) for a probability measure µ on X and the norm is given by ‖ f ‖ L p (d µ ) = ( ∫ X | f ( x ) | p d µ ) 1 /p .

## 2.1 Episodic Markov decision processes

A (finite-horizon) episodic MDPs is denoted as MDP ( S , A , H, P , r ) , where S is the state space with possibly infinite states; A is the finite action space; H is the number of steps in each episode;

P := { P h } H h =1 is the Markov transition kernel with the transition probability P h ( ·| s, a ) on action a taken at state s ∈ S in the h -th step; the reward functions r := { r h } H h =1 are assumed to be deterministic. For notational simplicity, denote X = S × A and x = ( s, a ) , we assume X = [0 , 1] d as a compact space of R d and r h : S × A → [0 , 1] at h -th step.

A non-stationary policy π is a collection of H functions π := { π h : S → A} H h =1 . Given a policy π , the (state) value function V π h : S → [0 , H ] is defined as the expected cumulative reward of the MDP starting from step h ∈ [ H ] , i.e., V π h ( s ) = E π [ ∑ H h ′ = h r h ′ ( s h ′ , a h ′ ) ∣ ∣ s h = s ] , ∀ s ∈ S , h ∈ [ H ] where E π [ · ] denotes the expectation with respect to the randomness of the trajectory { ( s h , a h ) } H h =1 obtained by the policy π . Likewise, the action-value function Q π h : S × A → [0 , H ] is defined as Q π h ( s, a ) = E π [ ∑ H h ′ = h r h ′ ( s h ′ , a h ′ ) ∣ ∣ s h = s, a h = a ] . Moreover, since the action space and episode length are both finite, there always exists an optimal policy π /star [51] such that V /star h ( s ) = sup π V π h ( s ) for all s ∈ S and h ∈ [ H ] . To simplify the notation, denote ( P h V )( s, a ) := E s ′ ∼ P h ( ·| s,a ) [ V ( s ′ )] and the Bellman operator ( T h V )( s, a ) = r h ( s, a ) + ( P h V )( s, a ) for any measurable function V : S → [0 , H ] . Using this notation, the Bellman equation associated with a policy π can be formulated as

<!-- formula-not-decoded -->

Similarly, the Bellman optimality equation is given by

<!-- formula-not-decoded -->

Accordingly, the optimal policy π /star is the greedy policy with respect to { Q /star h } H h =1 . Hence the Bellman optimality operator T /star h is defined as

<!-- formula-not-decoded -->

In the online setting , the goal is to learn the optimal policy π /star by minimizing the cumulative regret under the interaction with the environment over a number of episodes. For any policy π , the difference between V π 1 and V /star 1 quantifies its sub-optimality. Thus, after T (fixed but large) episodes, the total (expected) regret is defined as Regret ( T ) = ∑ T t =1 [ V /star 1 ( s t 1 ) -V ˜ π t 1 ( s t 1 ) ] , where ˜ π t is the policy executed in the t -th episode and s t 1 is the initial state.

By definition, the Bellman equation in Eq. (2) is equivalent to Q /star h = T /star h Q /star h +1 , ∀ h ∈ [ H ] .

## 2.2 Function spaces

We give an overview of Besov spaces for deep neural networks and the Barron space for two-layer neural networks. More details refer to Appendix A. For description simplicity, we focus on the ReLU activation function in this work.

Besov spaces: Previous work in approximation theory focuses on the 'smoothness' of the function, e.g., Hölder spaces [52, 37] and Sobolev spaces [53]. Here we consider the concept of α -smooth from modulus of smoothness [30], cf. , Appendix A.

Based on this, we consider a more general function space beyond Hölder spaces and Sobolev spaces, i.e., Besov spaces [54, 30], which allows for spatially inhomogeneous smoothness with spikes and jumps. The Besov space is defined by B α p,q ( X ) = { f ∈ L p ( X ) | ‖ f ‖ B α p,q &lt; ∞} , where the Besov norm is ‖ f ‖ B α p,q := ‖ f ‖ L p ( X ) + | f | B α p,q . The smoothness parameter α indicates which function at a certain smoothness degree can be represented. For example, if α &gt; d/p , then the related Besov space is continuously embedded in the set of the continuous functions; if α &lt; d/p , then the functions in the Besov space are no longer continuous. The formal definition and relations to Hölder spaces and Sobolev spaces are deferred to Appendix A.

Barron spaces: A two-layer neural network with m neurons can be represented as f ( x ) = 1 m ∑ m k =1 b k σ ( w /latticetop k x + c k ) with the ReLU activation function σ ( · ) used in this work and the neural network parameters { b k , w k , c k } m k =1 . It admits the integral representation f ( x ) = ∫ Ω bσ ( w /latticetop x + c ) ρ (d b, d w , d c ) , x ∈ X , where Ω = R × R d × R and ρ is a probability measure over Ω . Then the Barron space [31] endowed by the Barron norm is defined as

<!-- formula-not-decoded -->

The Barron space ˜ P [31] can be (roughly) equipped with the /lscript 1 -path norm, i.e., ‖ f ‖ ˜ P /lessorequalslant ‖ f ‖ P := 1 m ∑ m k =1 | b k | ( ‖ w k ‖ 1 + c k ) /lessorequalslant 2 ‖ f ‖ ˜ P . Accordingly, it is natural to use ‖ f ‖ P to denote the Barron norm, as the discrete version.

The Barron space [31] can be regarded as the largest function space for two-layer ReLU neural networks. Here the 'largest' terminology [31, 55] means that the approximation ability can avoid curse of dimensionality , i.e., 1) any function in Barron spaces can be efficiently approximated by two-layer neural networks with bounded norm; 2) any continuous function that can be efficiently approximated by two-layer neural networks with bounded norm belongs to a Barron space.

We remark that, avoiding curse of dimensionality is important in theory for practical highdimensional data in RL. However, Besov spaces are too large and thus do not enjoy this property for deep ReLU neural networks.

## 3 Algorithm: Value iteration via DNNs under /epsilon1 -greedy exploration

In this section, we lay out our algorithm 1 via value iteration by DNNs under the /epsilon1 -greedy policy. Though our value iteration algorithm is different from one gradient-step for deep Q-learning in DQN, it still shares the key spirit with DQN in terms of function approximation via DNNs, /epsilon1 -greedy exploration, and experience replay.

Function class: We define the function class F given by F = F 1 ×··· × F H , including F SNN for two-layer (Shallow) ReLU neural networks and F DNN for deep ReLU neural networks as below

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the weight parameters are W (1) ∈ R m × d , W ( l ) ∈ R m × m , ∀ l ∈ { 2 , 3 , . . . , L -1 } , and W ( L ) ∈ R m ; the bias parameter are b ( l ) ∈ R m , ∀ l ∈ [ L -1] and b ( L ) ∈ R . Such sparselyconnected neural networks require most of the network parameters to be zero or non-active, which can be verified [56]. The depth L , the width m , the sparsity parameter S and the norm parameter B can be determined later in our proof to achieve good approximation and estimation performance.

Experience replay: In our setting, after initialization, at t -th episode, at h -th time step, we have observed t -1 transition tuples, { ( s τ h , a τ h , s τ h +1 ) } t -1 τ =1 and attempt to estimate { Q /star h } H h =1 via DNNs. Note that, at each time step h , these t -1 transition tuples are neither independent nor identically distributed due to the interaction with value functions and stochastic transition. To pursue the independence among the transition tuples that is required in our analysis, we follow the experience replay scheme [36] that is successfully applied in DQN [4]. The intuition behind experience replay is to break (or weaken) the temporal dependency among the observations for neural networks training. When the replay memory is large (e.g., 10 6 in DQN [4]), experience replay is close to sampling independent transitions. To be specific, at t -th episode, we store transition { ( s t h , a t h , r h , s t h +1 ) } H h =1 in the replay memory D , and then sample a mini-batch of independent observations from D with { ( s τ j h , a τ j h , s τ j h +1 ) } ( j,h ) ∈ [ ˜ t ] × [ H ] for DNNs training. Here the number of mini-batch is denoted as ˜ t := /ceilingleft /rho1t /ceilingright with the mini-batch ratio /rho1 ∈ (0 , 1) , and { τ j } ˜ t j =1 is the index for the mino-batch of ˜ t independent samples. Note that such independence assumption from experience replay is also used in RL theory, e.g., [37, 44] and theoretically demonstrated to be a good de-correlator [57]. In fact, our analysis only requires independence via experience replay, which is still weaker than the standard iid assumption.

Value iteration via neural networks: In our algorithm, we apply the classical least squares value iteration via neural networks for value function learning [28]. We solve the following least squares

## Algorithm 1 Value Iteration via DNNs under /epsilon1 -greedy exploration with experience replay

- 1: Input: Function class F , the number of episodes T , the /epsilon1 -greedy parameter /epsilon1 ∈ (0 , 1) , minibatch ratio /rho1 ∈ (0 , 1) .
- 3: for episode t = 1 , . . . , T do
- 2: Initialize replay memory D .
- 4: Receive the initial state s t 1 .
- 6: Set the minibatch size ˜ t := /ceilingleft /rho1t /ceilingright for experience replay.
- 5: Set V t H +1 as the zero function.
- 7: for step h = H,.. . , 1 do
- 12: Take the policy { ˜ π t h } H h =1 to be greedy policy with probability 1 -/epsilon1 or any policy with probability /epsilon1 .
- 8: Obtain ̂ Q t h := argmin f ∈F ∑ ˜ t j =1 [ f ( s τ j h , a τ j h ) -r h ( s τ j h , a τ j h ) -V t h +1 ( s τ j h +1 ) ] 2 . 9: Obtain Q t h := ̂ Q t h and V t h ( · ) = max a ∈A Q t h ( · , a ) . 10: end for 11: // /epsilon1 -greedy for exploration
- 13: for step h = 1 , . . . , H do
- 15: Observe the reward r h ( s t h , a t h ) and obtain the next state s t h +1 .
- 14: Take a t h ∼ ˜ π t h ( ·| s t h ) .
- 16: end for
- 17: // experience replay
- 19: Sample random mini-batch of transitions from D with ˜ t pairs { ( s τ j h , a τ j h , s τ j h +1 ) } ( j,h ) ∈ [ ˜ t ] × [ H ] .
- 18: Store transition { ( s t h , a t h , r h , s t h +1 ) } H h =1 in D .
- 20: end for

regression problem via ˜ t independent samples

<!-- formula-not-decoded -->

For ease of simplicity for analyses, we directly assume that the global minima solution of problem (5) can be obtained, that follows [58, 52, 59] in deep learning theory. Nevertheless, our result could be extended to allow small optimization error in each episode that will be discussed in Section 6.

Besides, we also need the expectation version of ̂ E t h in problem (5) for our analysis. Formally, we assume each state-action pair in the mini-batch is sampled from a respective (unknown) probability measure, i.e., ( s τ j h , a τ j h ) ∼ µ τ j h , ∀ j ∈ [ ˜ t ] , where µ τ j h ∈ P ( S × A ) is from the collection of all probability distribution on S ×A . Taking the averaged measure ¯ µ ˜ t h := 1 ˜ t ∑ ˜ t j =1 µ τ j h , the expectation of ̂ E t h is defined as

Note that, ̂ Q t h in Eq. (5) is not an unbiased estimator of the squared Bellman error minimizer [60, 61]. Indeed, E t h differs from the squared Bellman error because of an extra variance term caused by the stochastic transition [62]. This biased estimation issue can be avoided (or alleviated) in practice by introducing target networks in DQN [63]. Some variants [64] of DQN can also reduce the biased estimate and performs well without target networks. Nevertheless, in our analysis, we center around the uniform bound sup f ∈F |E t h ( f ) -E t h ( f ) | instead of the Bellman error.

<!-- formula-not-decoded -->

̂ /epsilon1 -greedy exploration: In order to work in the online setting, we need to ensure that the learner visits 'good' state action pairs in the sense that are almost maximizers of the value function for unseen state, a.k.a. , exploration. In RL theory, a classical way is to design an optimistic estimate of the value function via a bonus function b t h [12, 65] such that Q t h = min { ̂ Q t h + b t h , H } + . Instead, we employ the /epsilon1 -greedy exploration that follows DQN-like algorithms. Using the /epsilon1 -greedy exploration will ensure each state-action pair can be visited with positive probability and favor independence among samples. In our algorithm, we directly set Q t h := min { ̂ Q t h , H } + , and then naturally incorporate the truncation operation in neural networks training, see Eqs. (3) and (4).

Based on the above description, our algorithm centers around deep neural function approximation via value iteration under the /epsilon1 -greedy exploration and experience replay under the online setting. This problem setting matches the spirit of practical DQN, which allows for better understanding deep RL.

## 4 Main results

This section presents our results for value iteration under deep (as well as two-layer) ReLU neural networks via the Besov spaces and Barron spaces, respectively. Our theory is based on the independence assumption via experience replay and achieves sublinear regret under the /epsilon1 -greedy exploration.

## 4.1 Efficient value iteration via DNNs in Besov spaces

In this setting, we consider ̂ Q t h = argmin f ∈F DNN ̂ E t h ( f ) in Eq. (5), where F DNN is the function space of deep ReLU neural networks defined in Eq. (4). We make the following assumption on the Besov space B , similar to [7, 12], where the Bellman optimality operator maps any bounded value function to a bounded Besov space ball.

Assumption 1. Let ˜ R be a fixed constant. Define B ˜ R = { f ∈ B α p,q ( X ) : ‖ f ‖ B /lessorequalslant ˜ R } in the Besov space and assume that for any h ∈ [ H ] and Q : S × A → [0 , H ] , we have T /star h Q ∈ B ˜ R .

Remark: Due to Q ∈ [0 , H ] , the radius ˜ R in fact depends on H , i.e., ˜ R /equivasymptotic H . Based on this assumption, we have the following theorem on the regret bound in the Besov space for deep RL under the /epsilon1 -greedy exploration.

Theorem 1. Under Assumption 1 with the smoothness parameter α &gt; d (1 /p -1 / 4) + in the Besov space B α p,q ( X ) , considering value function learning (5) via DNNs defined by Eq. (4) in Algorithm 1 under the /epsilon1 -greedy exploration and the mini-batch ratio /rho1 ∈ (0 , 1) , and taking

<!-- formula-not-decoded -->

then given a MDP-dependent constant K ∈ [1 , H ] , for any δ ∈ (0 , 1) , the total regret can be upper bounded with probability at least 1 -δ

Remark: We make the following remarks.

<!-- formula-not-decoded -->

i ) The constant K describes the 'myopic' level of MDPs under the /epsilon1 -greedy policy, e.g., the worst case ( K := H ) under the sparse rewards setting; the benign case K := c (for some small constant c ) under the helpful dense rewards setting as discussed in [49]. The exponential dependence on H (in the worst case for any MDP) can be avoided at an additional cost of worsening T dependence. In fact, whether in the benign/worst case, the sublinear regret is always achieved under some certain /epsilon1 values in Eq. (8), which theoretically demonstrates the efficiency of deep RL. Note that the chosen /epsilon1 ∈ (0 , 1) is always satisfied under a large episode T .

ii ) Clearly, the regret bound is a non-increasing function of the smoothness parameter α , which shows that an easier task (i.e., the target Q function is more smooth) leads to regret bounds with faster rates. Specially, if we take α →∞ (i.e., the target Q function is sufficiently smooth), which holds for linear function approximation

Regret ( T ) /lessorsimilar ˜ O ( H H +4 H +2 K 2 K +2 A K K +2 T K +1 K +2 ) , which recovers the regret bound ˜ O ( T K +1 K +2 ) in [49, Theorem 3] via Eluder dimension. In the best case ( K = 1 ), our regret bound implies ˜ O ( H 4 3 A 1 3 T 2 3 ) with H /greaterorequalslant 4 , which matches the optimal regret bound for the contextual bandits problem in terms of dependence on T or A under the /epsilon1 -greedy

exploration [66]. In the worst case ( K := H ), we can still obtain the sublinear regret at a certain O ( T H +1 H +2 ) rate.

˜ Theorem 1 demonstrates that the sublinear regret can be achieved by choosing O (log T ) depth and ˜ O ( T d 2 α + d ) width, but the sublinear regret bound ˜ O ( T αK +( α + d )( K +2) (2 α + d )( K +2) ) heavily depends on the feature dimension d , failing in the curse of dimensionality , which appears ineffective on high dimensional data in deep RL. In the next, we consider the Barron spaces, i.e., the 'largest' function space for two-layer neural networks to avoid the curse of dimensionality. In this case, the rate of the sublinear regret can get rid of d , which is useful for high dimensional data in practical RL.

## 4.2 Efficient value iteration via two-layer neural networks in Barron spaces

As mentioned before, Barron spaces are the 'largest' function space for two-layer neural networks. In this setting, we consider ̂ Q t h = argmin f ∈F SNN ̂ E t h ( f ) in Eq. (5), where F SNN is the function space of two-layer ReLU neural networks defined in Eq. (3). We give a similar assumption on the Bellman optimality operator in the Barron space.

Assumption 2. Let ˜ R &gt; 0 be a fixed constant. Define P ˜ R = { f ∈ P : ‖ f ‖ P ≤ ˜ R } in the Barron space, and assume that for any h ∈ [ H ] and Q : S × A → [0 , H ] , we have T /star h Q ∈ P ˜ R .

Based on this assumption, we have the following regret bounds for two-layer ReLU neural networks.

Theorem 2. Under Assumption 2, considering value function learning (5) by two-layer ReLU neural networks with width m and bounded /lscript 1 norm B defined by Eq. (3) in Algorithm 1 under the /epsilon1 -greedy exploration and the mini-batch ratio /rho1 ∈ (0 , 1) , then given a MDP-dependent constant K ∈ [1 , H ] , for any δ ∈ (0 , 1) , the total regret can be upper bounded with probability at least 1 -δ

Remark: In our result, taking m = Ω( √ T ) is suffice to achieve the sublinear regret bound ˜ O ( T 2 K +3 2 K +4 ) , which also gets rid of the feature dimension d , allowing for high-dimensional image data in practice.

<!-- formula-not-decoded -->

## 5 Discussion on architecture guidelines in deep RL

In this section, we present a detailed discussion on how our results provide the architecture guidelines in practical deep RL, in the perspective of the width, the depth, and problem-dependent smoothness of the Q function.

Width-depth and DQN: According to Theorem 1, the O (log T ) depth and ˜ O ( T d 2 α + d ) width are enough for sublinear regret in deep RL. Interestingly, we notice that this result is closely matching practical implementation of DQN. For example, the choices of [4] m = 512 and L = 5 can be explained by our theory, indeed log(512) ≈ 6 . Specially, when taking α → ∞ , this setting holds for linear function approximation. For two-layer neural networks endowed by the Barron space, the curse of dimensionality in terms of width and regret bound can be avoided in Theorem 2, supporting the premise of practical, high-dimensional RL.

Problem-dependent smoothness and exploration: The problem-dependent smoothness, determined by α , largely affects our regret bounds. The difficulty of a task in deep RL can be defined in two views: one is the smoothness of the target Q function; and the other is the degree of exploration. Intuitively speaking, if a RL task is difficult, then the target Q function is often complicated, and thus admits a relative lower smoothness; or we need conduct more exploration in a complex scenario. Our results coincide with these two views. One hand, the regret bound in Theorem 1 is a non-increasing function of the smoothness parameter α . A more difficult task in deep RL (i.e., a smaller α ) leads to a slower rate of the sublinear regret, which indicates that more episodes are required. On the other hand, Theorem 1 shows that the parameter /epsilon1 is also a non-increasing function of

<!-- formula-not-decoded -->

α . That means, a more difficult task in deep RL requires a larger /epsilon1 , i.e., we need conduct exploration more frequently.

Besides, the exploration parameter /epsilon1 is also affected by K for MDPs with different situations. For example, compared to the best case K = 1 , more frequent exploration (a larger α ) is required in MDPs under difficult cases, which coincides with our certain /epsilon1 value in Theorems 1 and 2.

Width and depth trade-off: Under a limit parameter budget, according to the width-depth ratio m/L = T d 2 α + d in Theorem 1, our theory indicates that less problem-dependent smoothness of Qfunction requires DNNs to be wider. In practice, if we work in the limited budget of parameters N in neural networks, e.g., N /equivasymptotic m 2 L , our theory implies that there is a tradeoff between the depth and width on smoothness, i.e., the depth L := N 1 / 3 T -2 d 3(2 α + d ) increasing with α (or T ) and the width m = N 1 / 3 T d 3(2 α + d ) decreasing with α (or T ).

Besides, according to the width-depth ratio, it can be found that, the change of α leads to less changes on the depth but more changes on the width. This shows that width and depth admit different levels of parameter sensitivity under the change of problem-dependent smoothness.

## 6 Proof outline

In this section, we outline the proof of our theoretical results presented in Section 4. As mentioned before, the technical challenge in our analysis is how to estimate the TD error without bonus function design. Apart from the regret decomposition, our proof framework includes two main parts: transformation of TD error estimation to generalization bounds, see Figure 1; and generalization bounds on non-iid data in certain Besov/Barron spaces for TD error analysis, see Figure 2. The complete proof is reported in the appendix.

Regret decomposition: This part is standard and commonly studied in RL theory, e.g., [65, 7, 12]. We briefly include here for self-completeness. Define the temporal-difference (TD) error as

<!-- formula-not-decoded -->

where Γ t h is a function on S × A for all h ∈ [ H ] and t ∈ [ T ] . Accordingly, the regret can be decomposed into ( c.f. Lemma 1)

<!-- formula-not-decoded -->

where the first term relates to the TD error and the second term is the statistical error based on the standard martingale difference sequences, which can be upper bounded by the Hoeffding-Azuma inequality with O ( √ H 3 T ) regret ( c.f. Lemma 2). The last term /epsilon1H T is due to the /epsilon1 -greedy exploration.

Transforming TD error to generalization bounds: To bound the TD error, we first introduce Lemma 3 with ¯ µ ˜ t h ( C ) &gt; 0 , where the event C denotes that all state-action pairs have been visited at all time steps under the /epsilon1 -greedy policy. Then we are able to build the connection between Term ( i ) and ‖ Γ t h ‖ L 2 (d¯ µ ˜ t h ) in the L 2 (d¯ µ ˜ t h ) -integrable space ( c.f. Lemma 4). After analysis of E t h ( f ) in Proposition 1, we transform the estimation of ‖ Γ t h ‖ L 2 (d¯ µ ˜ t h ) to the following two terms: generalization error and approximation error, respectively ( c.f. Lemma 5)

<!-- formula-not-decoded -->

```
     Rademacher complexity on non-iid data: Lem. 7 ⇐ Lem. 6 two-layer NNs: Thm. 2 ⇐O ( H 2 B √ log d/n ) ⇐ Lem. 13: Rademacher complexity of Barron spaces DNNs: Thm. 1 ⇐O ( n -2 α 2 α + d ) ⇐ Prop. 2 ⇐ Lem. 12 on LRC for Besov spaces
```

Figure 2: Proof framework of the TD error via generalization bounds on n non-iid data. We denote LRC by local Rademacher complexity for short.

where the first term is the generalization error which we elucidate in the next and the second term is the approximation error and can be considered in an L p ( X ) space for Besov spaces in Corollary 1. For example, the approximation error in the Besov space admits the certain O ( N -2 α/d ) rate in [30] for deep ReLU networks with L /equivasymptotic log N, S /equivasymptotic N, m /equivasymptotic N log N .

Generalization bounds on non-iid data: The key part left is to bound the generalization error on non-i.i.d data for the TD error estimation, see the proof framework in Figure 2. In our proof, we firstly verify that the maximum error in estimating the mean of any function f ∈ F can be still bounded by the Rademacher complexity of F in Lemma 6, and then generalization bounds by Rademacher complexity still holds by Lemma 7 via the averaged measure ¯ µ ˜ t h , which only requires the data to be independent. These results can be easily extended to local Rademacher complexity.

For deep neural networks, by computing the local Rademacher complexity of F DNN in Lemma 12 and choosing proper neural network parameters in Eq. (4), we derive the convergence rate of generalization bounds at a certain O ( n -2 α 2 α + d ) rate in Besov spaces ( c.f. Proposition 2) with n non-iid data. Combining the result of approximation error and taking the depth and width in Eq. (7), Term ( i ) can be upper bounded with high probability. Finally we conclude the proof of Theorem 1 by combining with the statistical error.

For two-layer neural networks, by computing the Rademacher complexity of F SNN in Lemma 13, we obtain the generalization error at a certain O ( H 2 B √ log d/n ) convergence rate. Combining the result of approximation error in Barron spaces with other terms in the regret decomposition, we conclude the proof of Theorem 2.

Regret bounds effected by optimization error: Here we briefly discuss the regret bound affected by a solution (denoted as ˜ Q t h ) that is not a global minimum of problem (5). Assume that the optimization error is small in the functional view, i.e., ‖ ˜ Q t h -̂ Q t h ‖ L 2 (d¯ µ ˜ t h ) /lessorequalslant ε opt , that will appear in Eq. (11), and accordingly Term ( i ) incurs in an extra regret bound O ( H 2 log T ) if we take ε opt := H/ √ ˜ t . This condition is fair and reasonable as the optimization error decreases with the mini-batch size ˜ t for neural network training but requires a refined analysis under non-iid data [67, 68].

## 7 Conclusion

This paper provides an in-depth understanding on neural network function approximation with the /epsilon1 -greedy exploration under the online setting beyond the 'linear' regime. Our results provide theoretical guarantees of sublinear regret bounds, and shed light on some guidelines for understanding deep RL in the perspective of the width-depth configuration and the problem-dependent smoothness of RL tasks.

The analysis of this work is built on the /epsilon1 -greedy policy for exploration, which are satisfied in practical cases when employing DQN. Nevertheless, designing a provably efficient exploration mechanism for deep RL could be an interesting future direction in both practice and theory. Besides, our theory requires state-action pairs to be independent, which (approximately) holds via experience replay and could be improved by reverse experience replay [69]. Furthermore, our work is built on the value iteration based algorithm, which is different from practical DQN that adapts Q-learning via one-step gradient descent. Towards a better understanding DQN in terms of Q learning and target networks [43, 44] would be an interesting direction.

## Acknowledgement

The authors would like to thank anonymous reviewers for their constructive suggestions to improve the presentation and point out the independence issue.

This work was supported by SNF project - Deep Optimisation of the Swiss National Science Foundation (SNSF) under grant number 200021\_205011; the Enterprise for Society Center (E4S); the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement n°725594 - time-data).

## References

- [1] Csaba Szepesvári. Algorithms for reinforcement learning. Synthesis lectures on artificial intelligence and machine learning , 4(1):1-103, 2010. 1
- [2] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018. 1
- [3] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning , pages 1587-1596. PMLR, 2018. 1
- [4] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature , 518(7540):529-533, 2015. 1, 2, 5, 8
- [5] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature , 529(7587):484-489, 2016. 1
- [6] LeCun Yann, Bengio Yoshua, and Hinton Geoffrey. Deep learning. Nature , 521(7553):436444, 2015. 1
- [7] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 21372143. PMLR, 2020. 1, 2, 3, 7, 9
- [8] Yining Wang, Ruosong Wang, Simon Shaolei Du, and Akshay Krishnamurthy. Optimism in reinforcement learning with generalized linear function approximation. In International Conference on Learning Representations , 2020. 1
- [9] Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pages 463-474. PMLR, 2020. 1
- [10] Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory , pages 4532-4576. PMLR, 2021. 1, 3
- [11] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems , pages 8571-8580, 2018. 1
- [12] Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, and Michael I Jordan. On function approximation in reinforcement learning: Optimism in the face of large state spaces. In Advances in Neural Information Processing Systems , 2020. 1, 2, 3, 6, 7, 9, 17, 18
- [13] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning , pages 1704-1713. PMLR, 2017. 1, 3
- [14] Kefan Dong, Jian Peng, Yining Wang, and Yuan Zhou. Root-n-regret for learning in markov decision processes with function approximation and low bellman rank. In Conference on Learning Theory , pages 1554-1557. PMLR, 2020. 1, 3
- [15] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems , 26, 2013. 1, 3
- [16] Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. In Advances in Neural Information Processing Systems , volume 33, pages 6123-6135, 2020. 1, 3

- [17] Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina Precup, and Lin Yang. Randomized exploration in reinforcement learning with general value function approximation. In International Conference on Machine Learning , pages 4607-4616. PMLR, 2021. 1, 3
- [18] Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In Advances in Neural Information Processing Systems , pages 2933-2943, 2019. 1
- [19] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in Neural Information Processing Systems , pages 85708581, 2019. 1
- [20] Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In Conference on Learning Theory , pages 3635-3673. PMLR, 2020. 1
- [21] Mario Geiger, Stefano Spigler, Arthur Jacot, and Matthieu Wyart. Disentangling feature and lazy training in deep neural networks. Journal of Statistical Mechanics: Theory and Experiment , 2020(11):113301, 2020. 1
- [22] Francis Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research , 18(1):629-681, 2017. 1
- [23] Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. In Advances in Neural Information Processing Systems , pages 6594-6604, 2019. 1
- [24] Michael Celentano, Theodor Misiakiewicz, and Andrea Montanari. Minimum complexity interpolation in random features models. arXiv preprint arXiv:2103.15996 , 2021. 1
- [25] Baihe Huang, Kaixuan Huang, Sham Kakade, Jason D Lee, Qi Lei, Runzhe Wang, and Jiaqi Yang. Going beyond linear rl: Sample efficient neural function approximation. In Advances in Neural Information Processing Systems , 2021. 2, 3
- [26] Gene Li, Pritish Kamath, Dylan J Foster, and Nathan Srebro. Eluder dimension and generalized rank. arXiv preprint arXiv:2104.06970 , 2021. 2
- [27] Kefan Dong, Jiaqi Yang, and Tengyu Ma. Provable model-based nonlinear bandit and reinforcement learning: Shelve optimism, embrace virtual curvature. In Advances in Neural Information Processing Systems , volume 34, 2021. 2
- [28] Ian Osband, Benjamin Van Roy, Daniel J Russo, Zheng Wen, et al. Deep exploration via randomized value functions. Journal of Machine Learning Research , 20(124):1-62, 2019. 2, 5
- [29] Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, et al. Deep q-learning from demonstrations. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018. 2
- [30] Taiji Suzuki. Adaptivity of deep ReLU network for learning in Besov and mixed smooth Besov spaces: optimal rate and curse of dimensionality. In International Conference on Learning Representations , 2019. 2, 4, 5, 10, 16, 25
- [31] Weinan E, Chao Ma, and Lei Wu. The barron space and the flow-induced function spaces for neural network models. Constructive Approximation , pages 1-38, 2021. 2, 4, 5, 17, 28, 29
- [32] Lin Yang and Mengdi Wang. Reinforcement learning in feature space: Matrix bandit, kernels, and regret bound. In International Conference on Machine Learning , pages 10746-10756. PMLR, 2020. 2, 3
- [33] Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al. A tutorial on thompson sampling. Foundations and Trends® in Machine Learning , 11(1):1-96, 2018. 2, 3
- [34] Shipra Agrawal and Navin Goyal. Analysis of thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory , pages 1-39. JMLR Workshop and Conference Proceedings, 2012. 2, 3

- [35] Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro Lazaric. Frequentist regret bounds for randomized least-squares value iteration. In International Conference on Artificial Intelligence and Statistics , pages 1954-1964. PMLR, 2020. 2, 3
- [36] Long-Ji Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning , 8(3):293-321, 1992. 2, 5
- [37] Jianqing Fan, Zhaoran Wang, Yuchen Xie, and Zhuoran Yang. A theoretical analysis of deep q-learning. In Learning for Dynamics and Control , pages 486-489, 2020. 2, 3, 4, 5
- [38] Jihao Long, Jiequn Han, et al. An l 2 analysis of reinforcement learning in high dimensions with kernel and neural network approximation. arXiv preprint arXiv:2104.07794 , 2021. 3
- [39] Thanh Nguyen-Tang, Sunil Gupta, Hung Tran-The, and Svetha Venkatesh. Sample complexity of offline reinforcement learning with deep relu networks. arXiv preprint arXiv:2103.06671 , 2021. 3
- [40] Martin Riedmiller. Neural fitted Q iteration-first experiences with a data efficient neural reinforcement learning method. In European Conference on Machine Learning , pages 317-328. Springer, 2005. 3
- [41] Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. In International Conference on Machine Learning , pages 10555-10565. PMLR, 2020. 3
- [42] Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Advances in neural information processing systems , 2018. 3
- [43] Andrea Zanette and Martin J Wainwright. Stabilizing Q-learning with linear architectures for provably efficient learning. arXiv preprint arXiv:2206.00796 , 2022. 3, 10
- [44] Diogo Carvalho, Francisco S Melo, and Pedro Santos. A new convergent variant of Q-learning with linear function approximation. In Advances in Neural Information Processing Systems , volume 33, pages 19412-19421, 2020. 3, 5, 10
- [45] Naman Agarwal, Syomantak Chaudhuri, Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. Online target Q-learning with reverse experience replay: Efficiently finding the optimal policy for linear mdps. arXiv preprint arXiv:2110.08440 , 2021. 3
- [46] Liran Szlak and Ohad Shamir. Convergence results for Q-learning with experience replay. arXiv preprint arXiv:2112.04213 , 2021. 3
- [47] Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning , pages 10978-10989. PMLR, 2020. 3
- [48] Pihe Hu, Yu Chen, and Longbo Huang. Nearly minimax optimal reinforcement learning with linear function approximation. In International Conference on Machine Learning , pages 89719019, 2022. 3
- [49] Chris Dann, Yishay Mansour, Mehryar Mohri, Ayush Sekhari, and Karthik Sridharan. Guarantees for epsilon-greedy reinforcement learning with function approximation. In International Conference on Machine Learning , pages 4666-4689, 2022. 3, 7, 19
- [50] Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. Advances in Neural Information Processing Systems , 34, 2021. 3
- [51] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014. 4
- [52] Minshuo Chen, Haoming Jiang, Wenjing Liao, and Tuo Zhao. Efficient approximation of deep ReLU networks for functions on low dimensional manifolds. Advances in neural information processing systems , 32:8174-8184, 2019. 4, 6
- [53] Ahmed Abdeljawad and Philipp Grohs. Approximations with deep neural networks in sobolev time-space. arXiv preprint arXiv:2101.06115 , 2020. 4
- [54] Yoshihiro Sawano. Theory of Besov spaces , volume 56. Springer, 2018. 4, 16
- [55] Weinan E and Stephan Wojtowytsch. Representation formulas and pointwise properties for barron functions. arXiv preprint arXiv:2006.05982 , 2020. 5, 17

- [56] Boris Hanin and David Rolnick. Deep relu networks have surprisingly few activation patterns. Advances in Neural Information Processing Systems , 32, 2019. 5
- [57] Shirli Di-Castro, Shie Mannor, and Dotan Di Castro. Analysis of stochastic processes through replay buffers. In International Conference on Machine Learning , pages 5039-5060. PMLR, 2022. 5
- [58] Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. Annals of Statistics , 48(4):1875-1897, 2020. 6
- [59] Taiji Suzuki and Atsushi Nitanda. Deep learning is adaptive to intrinsic dimensionality of model smoothness in anisotropic besov space. In Advances in Neural Information Processing Systems , 2021. 6
- [60] András Antos, Csaba Szepesvári, and Rémi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning , 71(1):89-129, 2008. 6
- [61] Yaqi Duan, Chi Jin, and Zhiyuan Li. Risk bounds and rademacher complexity in batch reinforcement learning. In International Conference on Machine Learning , pages 2892-2902. PMLR, 2021. 6, 21
- [62] Steven J Bradtke and Andrew G Barto. Linear least-squares algorithms for temporal difference learning. Machine learning , 22(1):33-57, 1996. 6
- [63] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013. 6
- [64] Seungchan Kim, Kavosh Asadi, Michael Littman, and George Konidaris. Deepmellow: removing the need for a target network in deep q-learning. In International Joint Conference on Artificial Intelligence , 2019. 6
- [65] Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. In International Conference on Machine Learning , pages 1283-1294, 2020. 6, 9, 17, 18, 19
- [66] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020. 8
- [67] Suhas Kowshik, Dheeraj Nagaraj, Prateek Jain, and Praneeth Netrapalli. Streaming linear system identification with reverse experience replay. Advances in Neural Information Processing Systems , 34:30140-30152, 2021. 10
- [68] Ahmet Alacaoglu and Hanbaek Lyu. Convergence and complexity of stochastic subgradient methods with dependent data for nonconvex optimization. arXiv preprint arXiv:2203.15797 , 2022. 10
- [69] Naman Agarwal, Syomantak Chaudhuri, Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. Online target Q-learning with reverse experience replay: Efficiently finding the optimal policy for linear MDPs. In International Conference on Learning Representations , 2022. 10
- [70] Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Conference on Learning Theory , pages 1376-1401. PMLR, 2015. 17
- [71] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning . MIT press, 2018. 22
- [72] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge University Press, 2019. 22, 29
- [73] Shahar Mendelson. Improving the sample complexity using global data. IEEE transactions on Information Theory , 48(7):1977-1991, 2002. 22, 25
- [74] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms . Cambridge University Press, 2014. 22, 24
- [75] Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. Annals of Statistics , 33(4):1497-1537, 2005. 24, 25
- [76] Yunwen Lei, Lixin Ding, and Yingzhou Bi. Local rademacher complexity bounds based on covering numbers. Neurocomputing , 218:320-330, 2016. 24

## Checklist

1. For all authors...
2. (a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]
3. (b) Did you describe the limitations of your work? [Yes] We clearly discuss the limitation of this work in the Conclusion section.
4. (c) Did you discuss any potential negative societal impacts of your work? [No] Our work is theoretical and generally will have no negative societal impacts.
5. (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]
2. If you are including theoretical results...
7. (a) Did you state the full set of assumptions of all theoretical results? [Yes] The assumptions are clearly stated and well discussed.
8. (b) Did you include complete proofs of all theoretical results? [Yes] All of the proofs can be found in the Appendix.
3. If you ran experiments...
10. (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [N/A]
11. (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [N/A]
12. (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [N/A]
13. (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [N/A]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
15. (a) If your work uses existing assets, did you cite the creators? [N/A]
16. (b) Did you mention the license of the assets? [N/A]
17. (c) Did you include any new assets either in the supplemental material or as a URL? [N/A]
18. (d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]
19. (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]
5. If you used crowdsourcing or conducted research with human subjects...
21. (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]
22. (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]
23. (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

The appendix is organized as follows

- Appendix A: preliminaries on Besov spaces and Barron spaces;
- Appendix B: proofs related to regret decomposition;
- Appendix C: proofs related to the temporal difference error and generalization error;
- Appendix D: proofs related to generalization bounds on non-iid data;
- Appendix E: proofs related to sublinear regret bounds for deep ReLU neural networks endowed by Besov spaces;
- Appendix F: proofs related to sublinear regret bounds for two-layer neural networks endowed by Barron spaces.

## A Preliminaries: Besov spaces and Barron spaces

In this section, we give an overview of Besov spaces for deep ReLU neural networks and the Barron spaces for two-layer ReLU neural networks.

## A.1 Besov spaces

Here we briefly introduce a general function space for deep ReLU neural networks according to the 'smoothness' of the function, i.e., Besov spaces.

To define Besov functions, we need introduce the modulus of smoothness.

Definition 1. [30, modulus of smoothness] For a function f ∈ L p ( X ) with some p ∈ (0 , ∞ ] , the k -th modulus of smoothness of f is defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

The quantity ∆ k h ( f ) captures the local oscillation of function f that is not necessarily differentiable. Based on this, the Besov space is defined as below.

Definition 2. [54, 30, Besov space B α p,q ( X ) ] For 0 &lt; p, q /lessorequalslant ∞ , the smoothness parameter α &gt; 0 , k := /floorleft α /floorright +1 , define the semi-norm | · | B α p,q as

<!-- formula-not-decoded -->

The norm of the Besov space B α p,q ( X ) is defined by ‖ f ‖ B α p,q := ‖ f ‖ L p ( X ) + | f | B α p,q , and the Besov space is B α p,q ( X ) = { f ∈ L p ( X ) | ‖ f ‖ B α p,q &lt; ∞} .

The smoothness parameter α indicates which function at a certain smoothness degree can be represented. For example, if α &gt; d/p , then the related Besov space is continuously embedded in the set of the continuous functions. However, if α &lt; d/p , then the functions in the Besov space are no longer continuous. In particular, the Besov space reduces to the Hölder space C α when p = q = ∞ and α is a positive non-integer; degenerates to the Sobolev space W α 2 when p = q = 2 and α is a positive integer. The Besov space is more general than these two spaces as it allows for spatially inhomogeneous smoothness with spikes and jumps. More properties of Besov spaces and relations to other function spaces refer to [30] for details.

## A.2 Barron spaces

The study for deep ReLU neural networks is endowed by Besov spaces, but the complete of function space for deep ReLU neural networks to avoid the curse of dimensionality is still open. Luckily, the

complete of function space for two-layer neural networks can be conducted by Barron spaces. Here we briefly introduce the basic definition and property of Barron spaces [55, 31].

<!-- formula-not-decoded -->

We consider a typical two-layer neural network f ( x ) = 1 m ∑ m k =1 b k σ ( w /latticetop k x + c k ) , where m is the number of neurons in the hidden layer and σ ( x ) = max { x, 0 } is the ReLU activation function used in this work. Accordingly, the two-layer neural network admits the following representation where Ω = R × R d × R and ρ is a probability measure over Ω . Then the Barron space [31] endowed by the p -Barron norm with p ∈ [1 , + ∞ ] is defined as

Specifically, when using ReLU, these function spaces under different p are the same, i.e., ˜ P 1 = ˜ P 2 = · · · = ˜ P ∞ , and thus we directly use ˜ P for short. The is the main reason why we study ReLU activation functions in this work. Besides, the Barron norm is close to the /lscript 1 -path norm [70]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As suggested by [55], Barron space can be regarded as the largest function space for two layer neural networks in two folds [31]: 1) direct approximation: Any function in Barron spaces can be efficiently approximated by two-layer neural networks with bounded /lscript 1 path norm at O (1 /m ) rate without curse of dimensionality ; 2) inverse approximation: Any continuous function that can be efficiently approximated by two-layer neural networks with bounded /lscript 1 -path norm belongs to a Barron space.

Based on this, for description simplicity, we do not strictly distinguish the Barron norm and the /lscript 1 -path norm, and regard ‖ f ‖ P as the discrete version of the Barron norm.

## B Regret decomposition

We present the regret decomposition under the /epsilon1 -greedy policy by constructing the martingale difference sequence and giving error bounds for this. Apart from an extra /epsilon1H T regret, this decomposition result appears in [65, 12], and we include them here just for self-completeness.

To establish the regret decomposition, we need some notations. Remember the definition of the regret, ˜ π t is the /epsilon1 -greedy policy and π t is the greedy policy at the t -th episode, and then we have

<!-- formula-not-decoded -->

where /epsilon1H T stems from the fact that the return of greedy and /epsilon1 -greedy policies can differ at most /epsilon1H in each episode. In the next, we aim to estimate the first term in the above equation. It involves the greedy policy π t at the t -th episode, which leads to a trajectory { ( s t h , a t h ) } H h =1 . Note that this trajectory is different from Algorithm 1 that uses the /epsilon1 -greedy policy but we use the same notation on state-action pairs for notational simplicity in this section.

Following [65, 12], we define two quantities ζ 1 t,h , ζ 2 t,h ∈ R for any h ∈ [ H ] and t ∈ [ T ] based on the greedy policy

By definition, ζ 1 t,h depends on the randomness of choosing an action a t h ∼ π t h ( ·| s t h ) ; and ζ 2 t,h captures the stochastic transition, i.e., the randomness of drawing the next state s t h +1 from P h ( ·| s t h , a t h ) . Based on the following definition of filtration, { ζ 1 t,h , ζ 2 t,h } forms a bounded martingale difference sequence.

<!-- formula-not-decoded -->

Definition 3. [65, Filtration] For any ( t, h ) ∈ [ T ] × [ H ] , define σ -algebras M t,h, 1 and M t,h, 2 generated by the following respective state-action sequence as where we identify F t, 0 , 2 with M t -1 ,H, 2 for all t /greaterorequalslant 2 and let M 1 , 0 , 2 be the empty set. Further, for any t ∈ [ T ] , h ∈ [ H ] and m ∈ [2] , we define the time-step index τ ( t, h, m ) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which offers an partial ordering over the triplets ( t, h, m ) ∈ [ T ] × [ H ] × [2] . Moreover, according to Eq. (14) , for any ( t, h, m ) and ( t ′ , h ′ , m ′ ) satisfying τ ( k, h, m ) /lessorequalslant τ ( k ′ , h ′ , m ′ ) , it holds that M k,h,m ⊆ M k ′ ,h ′ ,m ′ . Thus, the sequence of σ -algebras {M t,h,m } ( t,h,m ) ∈ [ T ] × [ H ] × [2] forms a filtration.

Accordingly, we have the following regret decomposition result.

Lemma 1 (Regret Decomposition [65, 12]) . Recall the definition of the temporal-difference error Γ t h : S × A → in Eq. (9) for all ( t, h ) ∈ [ T ] × [ H ] , then the regret can be decomposed as

<!-- formula-not-decoded -->

where ζ 1 t,h and ζ 2 t,h are defined in Eq. (13) .

Proof. Remember the definition of the regret, ˜ π t is the /epsilon1 -greedy policy and π t is the greedy policy at the t -th episode, and then we have where the first term (*) can be bounded by [65, 12]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the fact that π t is the greedy policy with respect to Q t h for any ( t, h ) ∈ [ T ] × [ H ] . The second term (**) is also bounded by [65, 12]

<!-- formula-not-decoded -->

Finally, we conclude the proof.

In the next, it is natural to employ Azuma-Hoeffding inequality for martingale difference sequences as below.

Lemma 2. [65, statistical error] For ζ 1 t,h and ζ 2 t,h defined in Eq. (13) and for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

## C Proofs of transformation on the temporal difference error

In this section, we aim to transform the temporal difference error in Term ( i ) to generalization bounds. This is the key part in our proof without bonus function design.

## C.1 TD error under the averaged measure

Here we build the connection between Term ( i ) in the regret decomposition and the TD error Γ t h in the L 2 (d¯ µ ˜ t h ) -integrable space.

To this end, we need study the relationship between L 2 (d µ ) -norm and L ∞ -norm, where µ can be any probability measure over S × A . For any f ∈ L 2 (d µ ) with δ /lessorequalslant ‖ f ‖ ∞ , denote then we have the following lemma that ¯ µ ˜ t h ( G δ ) can be lower bounded under the /epsilon1 -greedy policy.

<!-- formula-not-decoded -->

Lemma 3. Under the /epsilon1 -greedy policy, considering the set in Eq. (18) and the averaged measure ¯ µ ˜ t h based on a mini-batch of ˜ t historical state-action pairs, we have

Remark: Clearly, in the best case, we have ¯ µ ˜ t h ( G δ ) /greaterorequalslant Ω ( /epsilon1 A ) . Accordingly, we denote K ∈ [1 , H ] as a MDP-dependent constant to describe the 'myopic' level of MDPs [49] such that ¯ µ ˜ t h ( G δ ) /greaterorequalslant Ω ( ( /epsilon1/ A ) K ) .

<!-- formula-not-decoded -->

Proof. For any f ∈ L 2 (d µ ) with δ /lessorequalslant ‖ f ‖ ∞ , we have

<!-- formula-not-decoded -->

which is also valid to ¯ µ ˜ t h . Clearly, ¯ µ ˜ t h ( G δ ) ∈ [0 , 1] .

<!-- formula-not-decoded -->

To prove ¯ µ ˜ t h ( G δ ) &gt; 0 with the lower bound, we consider the worst case with δ = 0 and every time step taking non-greedy action with probability /epsilon1 . That means, we need to find the optimal stateaction pair in Eq. (18), which can be achieved by the fact that all state-action pairs have been visited at all time steps. It is clear that the cardinality of G δ is a non-decreasing function of δ . Accordingly, there exists j ∈ [ ˜ t ] such that where µ π τ j h h is the occupancy measure of the policy π τ j h at the h -step and t -th episode. Accordingly, µ π τ j h h admits the following representation

<!-- formula-not-decoded -->

Accordingly, in the worst case, at every time step we take any one action with probability /epsilon1/ A such that

<!-- formula-not-decoded -->

which implies that and accordingly we conclude the proof.

<!-- formula-not-decoded -->

Lemma 4. Given a MDP-dependent constant K ∈ [1 , H ] , for the temporal-difference error Γ t h defined in Eq. (9) for all ( t, h ) ∈ [ T ] × [ H ] , under the /epsilon1 -greedy policy, then Term ( i ) can be upper bounded by

<!-- formula-not-decoded -->

Proof. According to the definition of Term ( i ) in Lemma 1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, by taking δ := t -1 / 2 such that ∫ T 1 t -1 / 2 d t = O ( √ T ) , and using ¯ µ ˜ t h ( G δ ) /greaterorequalslant Ω ( ( /epsilon1/ A ) K ) with K ∈ [1 , H ] in Lemma 3, the above equation can be further expressed as which concludes the proof.

## C.2 Connection between the TD error and generalization bounds

Based on Lemma 4, the key issue left is to bound ∑ T t =1 ‖ Γ t h ‖ 2 L 2 (d¯ µ ˜ t h ) /lessorsimilar o ( T ) for a sublinear regret. To this end, we build the connection between ‖ Γ t h ‖ 2 L 2 (d¯ µ ˜ t h ) and generalization bounds. We first the study the decomposition of E t h ( f ) in Eq. (6) by the following proposition: there exists an extra variance term in the expected risk E t h ( f ) .

<!-- formula-not-decoded -->

Proposition 1. According to the definition of E t h ( f ) in Eq. (6) , then we have where the variance Var [ V t h +1 ( s h +1 )] := [ E s h +1 [ V t h +1 ( s h +1 )] -V t h +1 ( s h +1 ) ] 2 .

<!-- formula-not-decoded -->

Proof. Denote s ′ := s h +1 for short, we expand E t h ( f ) as the following expression where we use

E

′

s

∼

P

(

·|

s

,a

)

[

E

′

s

[

V

t

h

+1

(

s

′

)]

-

V

t

h

+1

(

s

′

)

]

= 0

and conclude the proof.

h

h

h

According to the decomposition of E t h ( f ) Proposition 1, ¯ E t h ( f ) in Eq. (21) is close to the squared Bellman error [61]. We are able to transform the estimation of the TD error to generalization error and approximation error as below.

Lemma 5. For the temporal-difference error Γ t h defined in Eq. (9) for all ( t, h ) ∈ [ T ] × [ H ] , it can be upper bounded in the L 2 (d µ ˜ t h ) space with

<!-- formula-not-decoded -->

where the first term is the generalization error of ̂ Q t h , the second term is the approximation error in the function class F .

Proof. According to the definition of the TD error Γ t h and taking f := ̂ Q t h in Eq. (21) given by Proposition 1, we have where the second equality holds by the definition of the averaged measure ¯ µ ˜ t h = 1 ˜ t ∑ ˜ t j =1 µ τ j h ; and we use Q t h = ̂ Q t h in the last equality as the truncation operation has been given in function classes, see Eqs. (3) and (4). Then, taking the infimum on both sides of Eq. (21), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equality holds by V t h +1 ( s h +1 ) = max a ∈A Q t h +1 ( s h +1 , a ) .

Combining Eqs. (22) and (23), we have

<!-- formula-not-decoded -->

which concludes the proof.

Based on Lemma 5, we have the following corollary if we consider the approximation error in L p ( X ) -integrable space, which is needed for our results on deep ReLU neural networks.

Corollary 1. Under the same setting of Lemma 5, we have

<!-- formula-not-decoded -->

Proof. Following the proof of Lemma 5, this result can be easily obtained by Cauchy-Schwartz inequality. To be specific, for any probability measure µ , we have

<!-- formula-not-decoded -->

where g is the probability density function associated with the probability measure µ . Note that the result here still holds true for the approximation error in L ∞ ( X ) if we use Hölder inequality, but this condition is much stronger as it requires the target Q function to be continuous.

## D Generalization bounds on non-iid data

In this section, we prove that the traditional Rademacher complexity is still valid for independent but non-identically distributed data under a well-defined measure. Similarly, such result is also valid to local Rademacher complexity. The key fact is that, the classical Rademacher complexity [71] is still valid as McDiarmid's bound only requires the independent property.

For description simplicity, we consider a general setting beyond our reinforcement learning task, i.e., learning with n independent but non-identical distributed data X = { x i } n i =1 in R d with x i ∼ µ i , ∀ i ∈ [ n ] . Define the average measure ¯ µ := 1 n ∑ n i =1 µ i , we have

Accordingly, the empirical Rademacher complexity of a function class F on the sample set X is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the expectation is taken over ξ = { ξ 1 , ξ 2 , · · · , ξ n } , i.e., Rademacher random variables, with Pr( ξ i = 1) = Pr( ξ i = -1) = 1 / 2 . The related Rademacher complexity under our non-iid setting is defined as

<!-- formula-not-decoded -->

where the expectation is taken over { x i } n i =1 with respect to each probability measure { µ i } n i =1 . This definition follows the classical Rademacher complexity [71] on iid samples to intuitively indicates how expressive the function class is. Besides, in our proof, we also need a notation of local Rademacher complexity on a set of vectors, where 'local' means that the class over which the Rademacher process is defined is a subset of the original class. Following the same style with Rademacher complexity, the local Rademacher complexity under the non-iid setting is defined as R n { f ∈ F : E ¯ µ f 2 /lessorequalslant R } , and the empirical local Rademacher complexity is defined as ̂ R n { f ∈ F : P n f 2 /lessorequalslant R } , where we denote P n f := 1 n ∑ n i =1 f ( x i ) for short. Besides, Rademacher complexity is also related to covering number, a metric for estimation of a hypothesis space. Here we give the definition of covering number, that is also used in this work.

Definition 4. [72, Definition 5.1, covering number] Let ( F , ‖ · ‖ ) be a norm space. A δ -cover of the set F with respect to ‖ · ‖ is a set { θ 1 , · · · , θ n } ⊆ F such that for each θ ∈ F , there exists some i ∈ [ n ] such that ‖ θ -θ i ‖ /lessorequalslant δ . The δ -covering number N ( δ, F , ‖ · ‖ ) is the cardinality of the minimal δ -cover.

In this work, we consider the covering number with two types of norms, one is N ( /epsilon1, F , ‖ · ‖ ∞ ) and the other is N ( /epsilon1, F , ‖ · ‖ 2 ) := sup n sup P n N ( /epsilon1, F , ‖ · ‖ L 2 ( P n ) ) [73].

## D.1 Rademacher complexity on non-iid data

Based on the definition of Rademacher complexity and its empirical version, we have the following lemma.

Lemma 6. Let X = { x i } n i =1 be an independent but non-identical distributed data set with x i ∼ µ i , ∀ i ∈ [ n ] , and R n ( F ) be the Rademacher complexity of the function class F on X , denote the averaged probability measure as ¯ µ := 1 n ∑ n i =1 µ i , then we have

<!-- formula-not-decoded -->

Proof. The proof follows with the classical Rademacher complexity [74, Chapter 26] apart from the averaged measure. Take a copy of X , i.e., X ′ = { x ′ i } n i =1 such that X ′ is independent but

x ′ i ∼ µ i , ∀ i ∈ [ n ] . According to Eq. (25), we have

Note that every possible configuration/value of ξ has probability of 1 / 2 n due to ξ ∈ {-1 , 1 } n . Without loss of generality, we can permute any configuration of ξ of such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where u = { u 1 , u 2 , · · · , u n } is a permutation of { 1 , 2 , . . . , n } . Accordingly, for any configuration of ξ , we have

<!-- formula-not-decoded -->

where we use the fact that x u i and x ′ u i are independent and symmetric. Based on this, we obtain

<!-- formula-not-decoded -->

where the last inequality holds by the fact that ξ i and -ξ i , i ∈ [ n ] admit the same distribution, and multiplying each term in the summation by a Rademacher variable ξ i will not change the expectation due to E ξ i = 0 .

Based on the above lemma, we demonstrate that the Rademacher complexity can be well approximated by the empirical Rademacher complexity under our non-iid setting.

Lemma 7. Under the same setting of Lemma 6, for any f ∈ F , assume that | f ( x ) -f ( x ′ ) | /lessorequalslant c, ∀ x , x ′ ∈ dom( f ) for some constant c &gt; 0 , for any δ ∈ (0 , 1) , the following proposition holds with probability at least 1 -δ

<!-- formula-not-decoded -->

Proof. The proof follows with the classical Rademacher complexity [74, Chapter 26] apart from the averaged measure. Recall the definition of the empirical Rademacher complexity in Eq. (26), ̂ R n ( F , X ) is a function of n random variables { x i } n i =1 . Moreover, due to | f ( x ) -f ( x ′ ) | /lessorequalslant c , ̂ R n ( F , X ) satisfies the precondition for McDiarmid's inequality by at most c/n , which only requires independence of random variables without the identically distributed condition which implies

<!-- formula-not-decoded -->

By Lemma 6, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we use McDiarmid's inequality again to obtain Pr( A ) /lessorequalslant e -2 nδ 2 /c 2 since E x ∼ ¯ µ [ f ( x )] -1 n ∑ n i =1 f ( x i ) can be regarded as a function of { x i } n i =1 and any variations of { x i } n i =1 would change the outcome by at most c/n . Denote event B as E x ∼ ¯ µ [ f ( x )] -1 n ∑ n i =1 f ( x i ) -2 R n ( F ) /greaterorequalslant δ , we have Pr( B ) /lessorequalslant Pr( A ) /lessorequalslant e -2 nδ 2 /c 2 .

Further, denote the event C as ̂ R n ( F , X ) /greaterorequalslant R n ( F ) -δ , we have Pr( C ) /greaterorequalslant 1 -exp( -2 nδ 2 /c 2 ) by Eq. (31). Denote the event D as E x ∼ ¯ µ ( f ( x )) -1 n ∑ n i =1 f ( x i ) /greaterorequalslant 2 R n ( F ) + 3 δ , we have

<!-- formula-not-decoded -->

Similar to the proof of Lemma 7, it is easy to verify that, the standard Massart's lemma and the Talagrand's Contraction Lemma (empirical Rademacher complexity of Lipschitz function class) in [74, Chapter 26] are valid to our independent but non-iid setting.

## D.2 Local Rademacher complexity

Here we present some results on local Rademacher complexity [75] that is needed in this work. The used lemmas here are still valid for our independent but non-identically distributed data. Since the proof framework is similar to what we present for Rademacher complexity, we omit the proofs here.

When applying local Rademacher complexity, we need the following definition.

Definition 5. A function ψ : R + → R + is sub-root if it is non-negative, non-decreasing, and if ψ ( x ) / √ x is non-increasing.

Lemma 8. [76, Theorem 2] Let F be a function class with ‖ f ‖ ∞ /lessorequalslant b, ∀ f ∈ F and ˜ F := { f -g : f, g ∈ F} , and P n f := 1 n ∑ n i =1 f ( x i ) , then taking the average measure ¯ µ , we have

<!-- formula-not-decoded -->

Lemma 9. [75, Theorem 3.3, modified version] Let f be a class of functions with ranges in [ a, b ] and assume that there exists some functional T : F → R + and some constant B such that Var ( f ) /lessorequalslant T ( f ) /lessorequalslant BPf for every f ∈ F . Let P n be the empirical measure supported on the independent data points { x i } n i =1 with the averaged measure ¯ µ := 1 n µ i , Let ψ be a sub-root function with the fixed point R ∗ . If for any R /greaterorequalslant R ∗ , ψ satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then for any J &gt; 1 and δ ∈ (0 , 1) , with probability at least 1 -δ , we have where c 1 , c 2 , c 3 are some positive constants.

Lemma 10. [73, Refined entropy integral] Let P n be the empirical measure supported on the independent data points { x i } n i =1 . For any function class F and any monotone sequence { /epsilon1 k } ∞ k =0 decreasing to 0 such that /epsilon1 0 /greaterorequalslant sup f ∈F √ P n f 2 , the following inequality holds for every non-negative integer N

<!-- formula-not-decoded -->

## E Proofs of regret bounds via deep ReLU neural networks

In this section, we give the proofs of regret bounds via deep ReLU neural networks according to the function class of T /star h Q in Besov spaces.

To conclude our proof, we need the following lemma that how well the functions in the Besov space can be approximated by deep neural networks with the ReLU activation. Here the approximation error is defined in the L 4 ( X ) -integrable space ( c.f. Corollary 1).

Lemma 11. (Approximation error in Besov space) [30, Proposition 1, modified version] Assume that the smoothness parameter α satisfies

<!-- formula-not-decoded -->

then there exists a deep neural network architecture F DNN ( L, m, S, B ) with ν := ( α -η ) / (2 η ) and a large N such that then it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In our proof, we need the following result on local Rademacher complexity of deep ReLU neural networks.

Lemma 12. Let X = { x i } n i =1 ⊆ [0 , 1] d be an independent but non-identical distributed data set with x i ∼ µ i , ∀ i ∈ [ n ] , and R n { f ∈ F DNN : Pf 2 /lessorequalslant R } be the local Rademacher complexity of the function class F DNN on X defined in Eq. (4) , denote the averaged measure as ¯ µ := 1 n ∑ n i =1 µ i , then for a large N , we have

<!-- formula-not-decoded -->

Remark: The parameter N depends on the number of the training data n , but it will be determined later.

Proof. According to [30, Lemma 3], the covering number of F DNN can be bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Lemma 10, taking ε j = 2 -j ε , and using the inequality

<!-- formula-not-decoded -->

then the following inequality holds for any J ∈ N + :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we choose ε := n -1 / 2 in the last inequality, and then we conclude the proof.

Based on the above result, we have the following proposition on generalization bounds in Besov spaces under non-iid state-action pairs.

Proposition 2. Given the solution ̂ Q t h = argmin f ∈F DNN ̂ E t h ( f ) in Eq. (5) , then for a large N and any δ ∈ (0 , 1) , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Proof. It is clear that ψ ( R ) defined in Eq. (37) in Lemma 12 is a sub-root function. Therefore, the fixed point R ∗ of ψ ( R ) can be analytically solved by the equation R ∗ = ψ ( R ∗ ) , which leads to

<!-- formula-not-decoded -->

Strictly speaking, there is an extra term N [ (log N ) 2 +log n ] 3 4 /n in the above equation, but we can omit it as we only concern the smallest and largest order. By verifying the variance-expectation condition, we have

E [ E t h ( ̂ Q t h ) -E t h ( f /star h )] 2 /lessorequalslant 16 H 2 E [ E t h ( ̂ Q t h ) -E t h ( f /star h )] , (38) where f /star h := argmin f ∈F DNN E t h ( f ) and we use the fact E t h ( f ) is 4 H -Lipschitz. Denote the function space F DNN with the following function formulation for any j ∈ [ n ]

we have P n ˆ g t h = ̂ E t h ( ̂ Q t h ) -̂ E t h ( f /star h ) /lessorequalslant 0 due to ̂ Q t h = argmin f ∈F ̂ E t h ( f ) . Then using E g 2 /lessorequalslant H 2 Pg , for any g ∈ ̂ F DNN by Eq. (38), according to Lemma 9, the following inequality holds with probability at least 1 -δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where which further implies

E t h ( ̂ Q t h ) -min f ∈F DNN E t h ( f ) /lessorsimilar N [ (log N ) 2 +log n ] n + H √ N [(log N ) 2 +log n ] n + H 2 log(1 /δ ) n . Finally, we conclude the proof.

Proof of Theorem 1. Using the approximation error in L 4 ( X ) by Corollary 1, the smoothness parameter satisfies α &gt; d (1 /p -1 / 4) + . By taking δ/ 2 in Proposition 2, we have

<!-- formula-not-decoded -->

where in the second inequality, taking α &gt; d (1 /p -1 / 4) + , the approximation error can be estimated by Lemma 11

<!-- formula-not-decoded -->

Accordingly, the right hand side of Eq. (39) can be minimized by taking N /equivasymptotic ˜ t d 2 α + d up to (log ˜ t ) 3 -order in Eq. (33) for choosing suitable L, m, S, B . To make the architecture of deep RL independent of a variable ˜ t (or t ) during different episodes, here we directly choose N /equivasymptotic T d 2 α + d log 3 T , in this case, Eq. (39) can be formulated as

<!-- formula-not-decoded -->

which requires the depth L and the width m up to

<!-- formula-not-decoded -->

Recall ˜ t := /ceilingleft /rho1t /ceilingright , according to Lemma 4, if α &gt; d (1 /p -1 / 4) + , then for any δ ∈ (0 , 1) , with probability at least 1 -δ/ 2 , the Term ( i ) can be upper bounded by

<!-- formula-not-decoded -->

Then taking δ/ 2 in the statistical error Term ( ii ) in Lemma 1, if α &gt; d (1 /p -1 / 4) + , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Then taking

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Finally we conclude the proof.

## F Proofs of regret bounds via two-layer neural networks

In this section, we focus on generalization bounds under the independent but non-identically distributed data setting in the Barron space, and it is useful to present estimates of our regret bound.

Lemma 13. For two-layer ReLU neural networks with bounded /lscript 1 path norm defined in Eq. (3) given the function class F SNN and n independent but non-identically distributed data points X = { x i } n i =1 ⊆ X , then we have

<!-- formula-not-decoded -->

Proof. Here we directly focus on the /lscript 1 path norm, which is different from [31, Theorem 3]. Based on the definition of two-layer ReLU neural networks defined in Eq. (3), denote ˜ w k := ( w /latticetop k , c k ) /latticetop and ˜ x = ( x /latticetop , 1) /latticetop for simplicity, the empirical Rademacher complexity of F SNN under our setting

can be upper bounded by

<!-- formula-not-decoded -->

where the first inequality holds by the homogeneity of ReLU for any ˜ w ∈ R d / { 0 } . Since the Massart's lemma is still valid under our independent but non-identically distributed data, R n ( F SNN ) can be further expressed by

<!-- formula-not-decoded -->

where the last inequality holds by the maximum of n sub-Gaussian random variables [72] since Rademacher random variables are sub-Gaussian, and finally we conclude the proof.

Proof of Theorem 2. Denote X := { ( s τ j h , a τ j h , s τ j h +1 ) } ˜ t j =1 for simplicity and notice that the function [ f ( s h , a h ) -r h ( s h , a h ) -V t h +1 ( s h +1 )] 2 is 4 H -Lipschitz. Then according to Lemma 7, for any δ ∈ (0 , 1) , the following result holds with probability at least 1 -δ/ 2

<!-- formula-not-decoded -->

where we use the empirical Rademacher complexity in Lemma 13. Accordingly, by Lemma 5 and Eq. (41), then with probability at least 1 -δ/ 2 , we have

<!-- formula-not-decoded -->

where the second inequality uses the approximation result for two-layer ReLU neural networks and the Barron space in [31, Theorem 4]. Accordingly, by Lemma 4, for any δ ∈ (0 , 1) , the Term ( i ) in

the regret decomposition can be upper bounded with probability at least 1 -δ/ 2

<!-- formula-not-decoded -->

where we use ˜ R /equivasymptotic H and ∫ T 1 ( t -1) -1 / 2 d t = O ( √ T ) .

<!-- formula-not-decoded -->

Accordingly, taking δ/ 2 in the statistical error Term ( ii ) in Lemma 1, then with the probability at least 1 -δ , the total regret can be upper bounded by

Taking m = Ω( √ T ) and /epsilon1 = O ( H 2 K +2 T -1 2( K +2) ) , the regret bound can be further represented as Regret ( T ) /lessorsimilar ˜ O ( H K +4 K +2 T 2 K +3 2 K +4 ) , which concludes the proof.