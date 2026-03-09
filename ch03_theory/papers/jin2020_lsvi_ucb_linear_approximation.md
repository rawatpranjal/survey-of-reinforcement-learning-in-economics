## Provably Efficient Reinforcement Learning with Linear Function Approximation

Chi Jin University of California, Berkeley chijin@cs.berkeley.edu

Zhaoran Wang Northwestern University zhaoranwang@gmail.com

Zhuoran Yang Princeton University zy6@princeton.edu

Michael I. Jordan University of California, Berkeley jordan@cs.berkeley.edu

## Abstract

Modern Reinforcement Learning (RL) is commonly applied to practical problems with an enormous number of states, where function approximation must be deployed to approximate either the value function or the policy. The introduction of function approximation raises a fundamental set of challenges involving computational and statistical efficiency, especially given the need to manage the exploration/exploitation tradeoff. As a result, a core RL question remains open: how can we design provably efficient RL algorithms that incorporate function approximation? This question persists even in a basic setting with linear dynamics and linear rewards, for which only linear function approximation is needed.

This paper presents the first provable RL algorithm with both polynomial runtime and polynomial sample complexity in this linear setting, without requiring a 'simulator' or additional assumptions. Concretely, we prove that an optimistic modification of Least-Squares Value Iteration (LSVI)-a classical algorithm frequently studied in the linear setting-achieves ˜ O ( √ d 3 H 3 T ) regret, where d is the ambient dimension of feature space, H is the length of each episode, and T is the total number of steps. Importantly, such regret is independent of the number of states and actions.

## 1 Introduction

Reinforcement Learning (RL) is a control-theoretic problem in which an agent tries to maximize its expected cumulative reward by interacting with an unknown environment over time [41]. Modern RL commonly engages practical problems with an enormous number of states, where function approximation must be deployed to approximate the (action-)value function -the expected cumulative reward starting from a stateaction pair-or the policy -the mapping from a state to its subsequent action. Function approximation, especially based on deep neural networks, lies at the heart of the recent practical successes of RL in domains such as Atari games [30], Go [38], robotics [23], and dialogue systems [27]. Moreover, deep neural networks serve as essential components of generic deep RL algorithms, including Deep Q-Network (DQN) [30], Asynchronous Advantage Actor-Critic (A3C) [31], and Trust Region Policy Optimization (TRPO) [36].

Despite the empirical successes of function approximation in RL, most existing theoretical guarantees apply only to tabular RL [see, e.g., 20, 33, 8, 22], in which the states and actions are discrete, and the value function is represented by a table. Due to the curse of dimensionality, only relatively small problems can be tackled by tabular RL. Thus, researchers have turned to function approximation [see, e.g., 40, 12, 43], in theory and in practice. While function approximation greatly expands the potential reach of RL,

particularly via deep RL architectures, it raises a number of fundamental theoretical challenges. For example, while the effective state and action spaces can be much larger when function approximation is used, the neighborhoods of most states are not visited even once during a set of learning episodes, which makes it difficult to obtain reliable estimates of value functions [see, e.g., 41, 42, 26]. To cope with this challenge, relatively simple function classes, including linear function classes, are often used. This introduces, however, a bias, even in the limit of infinite training data, given that the optimal value function and policy may not be linear [see, e.g., 10, 11, 43]. Thus, both in theory and in practice, the design of RL systems must cope with fundamental statistical problems of sparsity and misspecification, all in the context of a dynamical system. Moreover, a core distinguishing feature of RL is that it requires addressing the tradeoff between exploration and exploitation. Addressing this tradeoff algorithmically requires exactly the kinds of statistical estimates that are challenging to obtain in the RL setting due to sparsity, misspecification, and dynamics. Thus the following fundamental question remains open:

## Is it possible to design provably efficient RL algorithms in the function approximation setting?

By 'efficient' we mean efficient in both runtime and sample complexity-the runtime and the sample complexity should not depend on the number of states, but should depend instead on an intrinsic complexity measure of the function class.

Several recent attempts have been made to attack this fundamental problem. However, they either require the access to a 'simulator' [49] which alleviates the difficulty of exploration, or assume the transition dynamics to be deterministic [47, 48], to have a low variance [19], or are parametrizable by a relatively small matrix [50], which alleviates the difficulty in estimating the transition dynamics (see Section 1.1 for more details).

Focusing on a linear setting in which the transition dynamics and reward function are assumed to be linear, we present the first algorithm that is provably efficient in both runtime and sample complexity, without requiring additional oracles or stronger assumptions. Concretely, in the general setting of an episodic Markov Decision Process (MDP), we prove that an optimistic version of Least-Squares Value Iteration (LSVI) [12, 33]-a classical algorithm frequently studied in the linear setting-achieves ˜ O ( √ d 3 H 3 T ) regret, where d is the ambient dimension of feature space, H is the length of each episode, T is the total number of steps, and ˜ O ( · ) hides only absolute constant and poly-logarithmic factors. Importantly, such regret is independent of S and A -the number of states and actions. Our algorithm runs in O ( d 2 AKT ) time and O ( d 2 H + dAT ) space, which are again independent of S and thus efficient in practice. In addition, our result is robust to the linear assumption: When the underlying transition model is not linear, but ζ -close to linear in total variation distance (Assumption B), our algorithm achieves ˜ O ( √ d 3 H 3 T + ζdHT ) regret. That is, in addition to the standard √ T regret, the algorithm also suffers from a linear regret term that scales with an error ζ that arises due to the function class misspecification.

## 1.1 Related Work

Tabular RL: Tabular RL is well studied in both model-based [20, 33, 8, 17] and model-free settings [39, 22]. See also [24, 6, 7, 25, 37, 45] for a simplified setting with access to a 'simulator' (also called a generative model), which is a strong oracle that allows the algorithm to query arbitrary state-action pairs and return the reward and the next state. The 'simulator' significantly alleviates the difficulty of exploration, since a naive exploration strategy which queries all state-action pairs uniformly at random already leads to the most efficient algorithm for finding an optimal policy [7].

In the episodic setting with nonstationary dynamics and no 'simulators,' the best regrets achieved by existing model-based and model-free algorithms are ˜ O ( √ H 2 SAT ) [8] and ˜ O ( √ H 3 SAT ) [22], respectively,

both of which (nearly) attain the minimax lower bound Ω( √ H 2 SAT ) [20, 32, 22]. Here S and A denote the numbers of states and actions, respectively. Although these algorithms are (nearly) minimax-optimal, they can not cope with large state spaces, as their regret scales linearly in √ S , where S is often exponentially large in practice [see, e.g., 30, 38, 23, 27]. Moreover, the minimax lower bound suggests that, informationtheoretically, a large state space cannot be handled efficiently unless further problem-specific structure is exploited. Compared with this line of work, in the current paper we exploit the linear structure of the reward and transition functions and show that the regret of optimistic LSVI scales polynomially in the ambient dimension d rather than the number of states S .

Linear bandits: To enable function approximation, another line of related work studies stochastic linear bandits or stochastic linear contextual bandits [see, e.g., 5, 16, 28, 35, 14, 2], which is a special case of the linear MDP studied in this paper (Assumption A) with the episode length H set equal to one. See [13, 26] and the references therein for a detailed survey. The best regrets achieved by existing algorithms are ˜ O ( d √ T ) for linear bandits [2] and ˜ O ( √ dT ) for linear contextual bandits [5, 14], both of which scale polynomially in the ambient dimension d . We note, however, that while an MDP has state transition, linear bandits do not. This temporal structure captures the fundamental difference in their difficulties of exploration: a naive adaptation of existing linear bandit algorithms to the linear MDP setting yields a regret exponential in H -the length of each episode.

RL with function approximation: In the setting of linear function approximation, there is a long line of classical work on the design of algorithms, but this work does not provide polynomial sample efficiency guarantees [see, e.g., 12, 29, 41, 33, 9]. Recently, Yang and Wang [49] revisited the setting of linear transitions and rewards [12, 29] (Assumption A), and presented a sample-efficient algorithm assuming the access to a 'simulator'. Similar to the case of tabular setting, the 'simulator' greatly alleviates the difficulty of exploration. We also note that their very recent work [50], developed independently of the current paper, provides sample efficiency guarantees for exploration in the linear MDP setting. Compared with the current paper, [50] differs in that requires one additional key assumption-that the transition model can be parameterized by a relatively small matrix. This additional assumption reduces the number of free parameters in the transition model from potentially being infinite (for the case with an infinite number of states) to small and finite, and thus mitigates the challenges in estimating the transition model. As a result, their algorithm and main mechanism are based on estimating the unknown matrix, which differs from our approach. Finally, in a broader context, without the assumption of a linear MDP, sample efficiency guarantees have been established for RL under other assumptions, such as that the transition dynamics are fully deterministic [47, 48], or have low variances [19]. These assumptions can be potentially restrictive in practice, and may not hold even in the tabular setting. In contrast, our results directly cover the standard tabular case with no extra assumptions.

In the setting of general function approximation, Jiang et al. [21] present a generic algorithm Olive, which enjoys sample efficiency if a complexity measure that they refer to as 'Bellman rank' is small. It can be shown that Bellman rank is at most d under Assumption A, and thus Olive is sample efficient in our setting. In contrast to our results, Olive is not computationally efficient in general and it does not provide a √ T regret bound. Meanwhile, a recent line of work [51, 46] studies a nonparametric setting with H¨ older smooth reward and transition model. The sample complexities provided therein are exponential in dimensionality in the worst case.

## 2 Preliminaries

We consider the setting of an episodic Markov decision process, denoted by MDP( S , A , H , P , r) , where S and A are the sets of possible states and actions, respectively, H ∈ Z + is the length of each episode, P = { P h } H h =1 and r = { r h } H h =1 are the state transition probability measures and the reward functions, respectively. We assume that S is a measurable space with possibly infinite number of elements and A is a finite set with cardinality A . Moreover, for each h ∈ [ H ] , P h ( ·| x, a ) denotes the transition kernel over the next states if action a is taken for state x at step h ∈ [ H ] , and r h : S ×A → [0 , 1] is the deterministic reward function at step h . 1

An agent interacts with this episodic MDP as follows. In each episode, an initial state x 1 is picked arbitrarily by an adversary. Then, at each step h ∈ [ H ] , the agent observes the state x h ∈ S , picks an action a h ∈ A , and receives a reward r h ( x h , a h ) . Moreover, the MDP evolves into a new state x h +1 that is drawn from the probability measure P h ( ·| x h , a h ) . The episode terminates when x H +1 is reached. We note that the agent cannot take an action at x H +1 and hence receives no reward.

A policy π of an agent is a function π : S × [ H ] →A , where π ( x, h ) is the action that the agent takes at state x and at the h th step in the episode. Moreover, for each h ∈ [ H ] , we define the value function V π h : S → R as the expected value of cumulative rewards received under policy π when starting from an arbitrary state at the h th step. Specifically, we have

<!-- formula-not-decoded -->

Accordingly, we also define the action-value function Q π h : S × A → R which gives the expected value of cumulative rewards when the agent starts from an arbitrary state-action pair at the h -th step and follows policy π afterwards; that is,

<!-- formula-not-decoded -->

Since the action spaces and the episode length are both finite, there always exists an optimal policy π /star which gives the optimal value V /star h ( x ) = sup π V π h ( x ) for all x ∈ S and h ∈ [ H ] [see, e.g., 34]). To simplify the notation, we denote [ P h V h +1 ]( x, a ) := E x ′ ∼ P h ( ·| x,a ) V h +1 ( x ′ ) . Using this notation, the Bellman equation associated with a policy π becomes

<!-- formula-not-decoded -->

which holds for all ( x, a ) ∈ S × A . Similarly, the Bellman optimality equation is

<!-- formula-not-decoded -->

This implies that the optimal policy π /star is the greedy policy with respect to the optimal action-value function { Q /star h } h ∈ [ H ] . Thus, to find the optimal policy π /star , it suffices to estimate the optimal action-value functions.

Furthermore, under the setting of an episodic MDP, the agent aims to learn the optimal policy by interacting with the environment during a set of episodes. For each k ≥ 1 , at the beginning of the k th episode,

1 While we study deterministic reward functions for notational simplicity, our results readily generalize to random reward functions. Also, we assume the reward lies in [0 , 1] without loss of generality.

the adversary picks the initial state x k 1 and the agent chooses policy π k . The difference in values between V π k 1 ( x k 1 ) and V /star 1 ( x k 1 ) serves as the expected regret or the suboptimality of the agent at the k -th episode. Thus, after playing for K episodes, the total (expected) regret is

<!-- formula-not-decoded -->

## 2.1 Linear Markov decision processes

We focus on a setting of a linear Markov decision process , where the transition kernels and the reward function are assumed to be linear. This assumption implies that the action-value function is linear, as we will show. Note that this is not the same as the assumption that the policy is a linear function-an assumption that has been the focus of much of the literature. Rather, it is akin to a statistical modeling assumption, in which we make assumptions about how data are generated and then study various estimators. Formally, we make the following definition.

Assumption A (Linear MDP [12, 29]) . MDP( S , A , H , P , r) is a linear MDP with a feature map φ : S × A → R d , if for any h ∈ [ H ] , there exist d unknown (signed) measures µ h = ( µ (1) h , . . . , µ ( d ) h ) over S and an unknown vector θ h ∈ R d , such that for any ( x, a ) ∈ S × A , we have

<!-- formula-not-decoded -->

Without loss of generality, we assume ‖ φ ( x, a ) ‖ ≤ 1 for all ( x, a ) ∈ S × A , and max {‖ µ h ( S ) ‖ , ‖ θ h ‖} ≤ √ d for all h ∈ [ H ] .

By definition, in a linear MDP, both the Markov transition model and the reward functions are linear in a feature mapping φ . We remark that despite being linear, the Markov transition model P h ( ·| x, a ) can still have infinite degrees of freedom as the measure µ h is unknown. This is a key difference from the linear quadratic regulator [1, 18, 4, 3, 15] or the recent work of Yang and Wang [50], whose transition models are completely specified by a finite-dimensional matrix such that the degrees of freedom are bounded.

Recall that we assume the reward functions are bounded in [0 , 1] , which implies that the value functions are bounded in [0 , H ] . Our choice of normalization conditions in Assumption A implies that the following concrete examples serve as special cases of a linear MDP.

Example 2.1 (Tabular MDP) . For the scenario with finitely many states and actions, letting d = |S| × |A| , then each coordinate can be indexed by state-action pair ( x, a ) ∈ S × A . Let φ ( x, a ) = e ( x,a ) be the canonical basis in R d . Then if we set e /latticetop ( x,a ) µ h ( · ) = P h ( ·| x, a ) and e /latticetop ( x,a ) θ h = r h ( x, a ) for any h ∈ [ H ] , we recover the tabular MDP.

Example 2.2 (Simplex Feature Space) . When the feature space, { φ ( x, a ): ( x, a ) ∈ S × A} , is a subset of the d -dimensional simplex, { ψ | ∑ d i =1 ψ i = 1 and ψ i ≥ 0 for all i } , a linear MDP can be instantiated by choosing e /latticetop i µ h to be an arbitrary probability measure over S and letting θ h be any vector such that ‖ θ h ‖ ∞ ≤ 1 .

As mentioned earlier, a crucial property of the linear MDP is that, for all policies, the action-value functions are always linear in the feature map φ . Therefore, when designing RL algorithms, it suffices to focus on linear action-value functions.

## Algorithm 1 Least-Squares Value Iteration with UCB (LSVI-UCB)

```
1: for episode k = 1 , . . . , K do 2: Receive the initial state x k 1 . 3: for step h = H,... , 1 do 4: Λ h ← ∑ k -1 τ =1 φ ( x τ h , a τ h ) φ ( x τ h , a τ h ) /latticetop + λ · I . 5: w h ← Λ -1 h ∑ k -1 τ =1 φ ( x τ h , a τ h )[ r h ( x τ h , a τ h ) + max a Q h +1 ( x τ h +1 , a )] . 6: Q h ( · , · ) ← min { w /latticetop h φ ( · , · ) + β [ φ ( · , · ) /latticetop Λ -1 h φ ( · , · )] 1 / 2 , H } . 7: for step h = 1 , . . . , H do 8: Take action a k argmax Q h ( x k , a ) , and observe x k .
```

- h ← a ∈A h h +1

Proposition 2.3. For a linear MDP, for any policy π , there exist weights { w π h } h ∈ [ H ] such that for any ( x, a, h ) ∈ S × A × [ H ] , we have Q π h ( x, a ) = 〈 φ ( x, a ) , w π h 〉 .

We provide a proof of this proposition in Appendix A, where we also present additional discussion of the basic properties of a linear MDP.

## 3 Main Results

In this section, we present our main results, which provide sample complexity guarantees for Algorithm 1 in the linear MDP setting (Theorem 3.1) and in a misspecified setting (Theorem 3.2).

Wefirst lay out our algorithm (Algorithm 1)-an optimistic modification of Least-Square Value Iteration (LSVI), where the optimism is realized by Upper-Confidence Bounds (UCB). At a high level, each episode consists of two passes (or loops) over all steps. The first pass (line 3-6) updates the parameters ( w h , Λ h ) that are used to form the action-value function Q h . The second pass (line 7-8) executes the greedy policy, a h = argmax a ∈A Q h ( x h , a ) , according to the Q h obtained in the first pass. We note Q H +1 ( · , · ) ≡ 0 since the agent receives no reward after the H th step. For the first episode k = 1 , since the summation in line 4-5 is from τ = 1 to 0 , we simply have Λ h ← λ I and w h ← 0 . Line 6 specifies the dependency of the action-value function Q h on the parameters w h and Λ h , and no actual updates need to be performed.

The idea of Least-Square Value Iteration [12, 33] stems from the classical value-iteration algorithm, which finds the optimal policy (or action-value function) by applying the Bellman optimality equation Eq. (2) recursively:

In practical RL with linear function approximation, there are two challenges to face in implementing the updates: First, P h is unknown, and it is replaced by the samples observed empirically. Second, in the setting of large state space, we cannot iterate over all ( x, a ) . We parametrize Q /star h ( x, a ) by a linear form w /latticetop h φ ( x, a ) instead. A natural idea here is to replace the Bellman update by solving for w h in a least-squares problem. In fact, the update of w h in Algorithm 1 solves precisely the following regularized least-squares problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Algorithm 1 additionally adds an UCB bonus term of form β ( φ /latticetop Λ -1 h φ ) 1 / 2 to encourage exploration, where Λ h is the Gram matrix of the regularized least-squares problem, and β is a scalar. This form of bonus is

common in the literature on linear bandits [13, 26]. Intuitively, m := ( φ /latticetop Λ -1 h φ ) -1 represents the effective number of samples the agent has observed so far along the φ direction, and thus the bonus term β/ √ m represents the uncertainty along the φ direction. It is called an upper confidence bound because, by choosing a proper value for β we can prove that, with high probability, Q h in line 5 of Algorithm 1 is always an upper bound of Q /star h for all state-action pair (see Lemma B.5).

We are now ready to state our main theorem, which gives a √ T -regret bound in the linear MDP setting without any further assumptions. Here, T = KH is the total number of steps.

Theorem 3.1. Under Assumption A, there exists an absolute constant c &gt; 0 such that, for any fixed p ∈ (0 , 1) , if we set λ = 1 and β = c · dH √ ι in Algorithm 1 with ι := log(2 dT/p ) , then with probability 1 -p , the total regret of LSVI-UCB (Algorithm 1) is at most O ( √ d 3 H 3 Tι 2 ) , where O ( · ) hides only absolute constants.

Theorem 3.1 asserts that when λ and β are set properly, LSVI-UCB will suffer total regret at most ˜ O ( √ d 3 H 3 T ) . We emphasize that while a naive adaptation of existing linear bandit algorithms to this linear MDP setting easily yields a regret exponential in H , our regret is only polynomial in H . Avoiding this exponential dependency on the planning horizon is a key step in efficiently solving the sequential RL problem. Additionally, comparing to the minimax regret in a tabular setting, ˜ Θ( √ H 2 SAT ) , our regret replaces the number of state-action pairs SA by a polynomial dependency on the intrinsic complexity measure of feature space, d . In fact, our regret is completely independent of S and A , which is crucial in the large state-space setting where function approximation is necessary. Please see also Section 5 for more discussion on the optimal dependencies on d and H .

We remark that Algorithm 1 only needs to store Λ h , w h , r ( x k h , a k h ) and { φ ( x k h , a ) } a ∈A for all ( h, k ) ∈ [ H ] × [ K ] , which takes O ( d 2 H + dAT ) space. When we compute Λ -1 h by the Sherman-Morrison formula, the computational complexity of Algorithm 1 is dominated by line 5 in computing max a Q h +1 ( x τ h +1 , a ) for all τ ∈ [ k ] . This takes O ( d 2 AK ) time per step, which gives a total runtime O ( d 2 AKT ) .

Finally, similarly to the discussion in Section 3.1 of [22], our regret bound (Theorem 3.1) directly translates to a sample complexity guarantee (or a PAC guarantee) in the following sense. When the initial state x 1 is fixed for all episodes, then, with at least constant probability, we can learn an ε -optimal policy π which satisfies V /star ( x 1 ) -V π ( x 1 ) ≤ ε using ˜ O ( d 3 H 4 /ε 2 ) samples. The algorithm to achieve this is to simply run Algorithm 1 for K = ˜ O ( d 3 H 3 /ε 2 ) episodes, and then output the greedy policy according to the action-value function Q at the k th episode, where k is sampled uniformly from [ K ] .

## 3.1 Results for a misspecified setting

Theorem 3.1 hinges on the fact that the MDP has a linear structure. A natural follow-up question arises: what would happen if the underlying MDP is not linear, and thus misspecified? We first present a definition for an approximate linear model.

Assumption B ( ζ -Approximate Linear MDP) . For any ζ ≤ 1 , we say that MDP( S , A , H , P , r) is a ζ -approximate linear MDP with a feature map φ : S × A → R d , if for any h ∈ [ H ] , there exist d unknown (signed) measures µ h = ( µ (1) h , . . . , µ ( d ) h ) over S and an unknown vector θ h ∈ R d such that for any ( x, a ) ∈ S × A , we have

<!-- formula-not-decoded -->

Without loss of generality, we assume that ‖ φ ( x, a ) ‖ ≤ 1 for all ( x, a ) ∈ S×A , and max {‖ µ h ( S ) ‖ , ‖ θ h ‖} ≤ √ d for all h ∈ [ H ] .

By definition, an MDP is an ζ -approximately linear MDP if there exists a linear MDP such that their Markov transition dynamics and reward functions are close. Here the closeness between transition dynamics is measured in terms of total variation distance.

In general, an algorithm designed for a linear MDP could break down entirely if the underlying MDP is not linear. The following theorem states that this is not the case for our algorithm. It is in fact robust to small model misspecification. To achieve this, we need only to adopt a different hyperparameter β in different episodes.

Theorem 3.2. Under Assumption B, there exists an absolute constant c &gt; 0 such that, for any fixed p ∈ (0 , 1) , if we set λ = 1 and β k = c · ( d √ ι + ζ √ kd ) H in Algorithm 1 with ι := log(2 dT/p ) , then with probability 1 -p , the total regret of LSVI-UCB (Algorithm 1) is at most O ( √ d 3 H 3 Tι 2 + ζdHT √ ι ) .

Compared with Theorem 3.1, Theorem 3.2 asserts that the LSVI-UCB algorithm will incur at most an additional ˜ O ( ζdHT ) regret when the model is misspecified. This additional term is inevitably linear in T due the intrinsic bias introduced by linear approximation. When ζ is sufficiently small, i.e., the underlying MDP is not far away from being linear, our algorithm will still enjoy good theoretical guarantees.

Theorem 3.2 can also be converted to a PAC guarantee with a similar flavor. When the initial state x 1 is fixed for all episodes, then, with at least constant probability, we can learn an ε -optimal policy π which satisfies V /star ( x 1 ) -V π ( x 1 ) ≤ ε + ˜ O ( ζdH 2 ) using ˜ O ( d 3 H 4 /ε 2 ) samples.

## 4 Mechanisms

In this section, we overview several of the key ideas behind the regret bound in Theorem 3.1. We defer the full proof of Theorem 3.1 and Theorem 3.2 to Appendix B and Appendix C respectively.

In Section 3, we mentioned that the LSVI algorithm is motivated from the Bellman optimality equation Eq. (2). It remains to verify that line 5 in Algorithm 1 indeed well approximates the Bellman optimality equation, which turns out to require not only the linear MDP structure but also hinges on several other facts.

To simplify our presentation, in this section we treat the regularization parameter λ loosely as being sufficiently small so that Λ -1 h ∑ k -1 τ =1 φ ( x τ h , a τ h ) φ ( x τ h , a τ h ) /latticetop ≈ I . We will focus in this section on a fixed episode k , and drop the dependency of parameters and value functions on k when it is clear from the context. Now, ignoring the UCB bonus, the least-squares solution (line 5) gives the following estimate of the actionvalue function:

<!-- formula-not-decoded -->

where V h +1 ( · ) = max a ∈A Q h +1 ( · , a ) . Plugging in r h ( · , · ) = φ ( · , · ) /latticetop θ h , we know the first term on the right-hand side approximates r h ( x, a ) . Comparing this to Eq. (2), it remains to show why the second term of right-hand side approximates P h V h +1 ( x, a ) . We thus define our empirical Markov transition measure as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the δ -measure δ ( · , x ) puts an atom on element x . It remains to verify that ̂ P h V h +1 ( x, a ) ≈ P h V h +1 ( x, a ) . To establish this, we use a measure ¯ P h to bridge these two quantities:

Our analysis depends on the following two key steps.

Step 1: Prove ̂ P h V h +1 ( x, a ) ≈ ¯ P h V h +1 ( x, a ) via Value-Aware Uniform Concentration. Computing the difference, we have ( ̂ P h -¯ P h ) V h +1 ( x, a ) = φ ( x, a ) /latticetop Λ -1 h ∑ k -1 τ =1 φ ( x τ h , a τ h )[ V h +1 ( x τ h +1 ) -P h V h +1 ( x τ h , a τ h )] . Since x τ h +1 is a sample from the distribution P h ( ·| x τ h , a τ h ) , we would expect this term to be small due to concentration. This would be the case if function V h +1 is fixed and independent of the samples { x τ h +1 } k -1 τ =1 . Then, V h +1 ( x τ h +1 ) -P h V h +1 ( x τ h , a τ h ) is a zero-mean random variable in [ -H,H ] , and we could aim to use a concentration inequality for self-normalized processes to bound ( ̂ P h -¯ P h ) V h +1 ( x, a ) . Please see Theorem D.3 or [2] for more detail on this approach.

However, the function V h +1 in Algorithm 1 is again computed by least-squares value iteration in later steps [ h +1 , H ] and it thus inevitably depends on the choices of actions { a τ h +1 } k -1 τ =1 , and thus also samples { x τ h +1 } k -1 τ =1 . Therefore, the concentration of self-normalized process does not apply directly. To resolve this issue, we establish the uniform concentration over all value functions in the following class:

<!-- formula-not-decoded -->

Step 2: Show ¯ P h V h +1 ( x, a ) ≈ P h V h +1 ( x, a ) due to Linear Markov Transitions. One big challenge in RL with function approximation is that, due to the large state space, the learner may never visit the neighborhood of a state-action pair twice. This raises a question of how to use the experiences from other state-action pairs to infer information about a state-action pair of interest. In Eq. (5), ¯ P h ( ·| x, a ) provides such an estimate via regularized least-squares. Our modeling assumption of a linear MDP (Assumption A) ensures that this least-square estimate is valid: since P h ( ·| x, a ) = φ ( x, a ) /latticetop µ h ( · ) for any ( x, a ) pair, we have where the parameters w , β, Λ are all bounded. We ensure that Algorithm 1 only uses value functions within this class V , which has a reasonably small covering number. This gives, with high probability, | ( ̂ P h -¯ P h ) V h +1 ( x, a ) | ≤ ˜ O ( dH ) · ( φ ( x, a )Λ -1 h φ ( x, a )) 1 / 2 (Lemma B.3).

<!-- formula-not-decoded -->

In summary, combining step 1 and step 2, we establish ̂ P h V h +1 ( x, a ) ≈ P h V h +1 ( x, a ) , and hence show that LSVI approximates the optimal Bellman equation. We emphasize that despite being linear, the Markov transition model P h ( ·| x, a ) = φ ( x, a ) /latticetop µ h ( · ) can still have infinite degrees of freedom since the measure µ h is unknown. Therefore, within a finite number of samples, no algorithm can establish that ̂ P h and P h are close in total variation distance. In contrast, our algorithm only requires ̂ P h V h +1 ( x, a ) ≈ P h V h +1 ( x, a ) for all value functions V h +1 in a small function class V (especially in step 1). This bypasses the need for fully learning the transition model P h . Thus, our algorithm can also be viewed as ' model-free ' in this sense.

Finally, with the above key observations in mind, our proof proceeds by leveraging and adapting techniques from the literature on tabular MDP and linear bandits. Please see Appendix B and C for the details.

## 5 Conclusion

In this paper, we have presented the first provable RL algorithm with both polynomial runtime and polynomial sample complexity for linear MDPs, without requiring a 'simulator' or additional assumptions. The

algorithm is simply Least-Squares Value Iteration-a classical RL algorithm commonly studied in the setting of linear function approximation-with a UCB bonus. We hope that our work may serve as a first step towards a better understanding of efficient RL with function approximation.

We provide a few additional concluding observations.

On the optimal dependencies on d and H . Theorem 3.1 claims the total regret to be upper bounded by ˜ O ( √ d 3 H 3 T ) . One immediate question is what the optimal dependencies on d and H are. Since our setting covers the standard tabular setting, as in shown in Example 2.1, a lower bound can be directly obtained through a reduction from the tabular setting, which gives Ω( √ dH 2 T ) for the case of nonstationary transitions [22]. We believe the √ H difference between this lower bound and our upper bound is expected because the exploration bonus used in this paper is intrinsically 'Hoeffding-type.' Using a 'Bernstein-type' bonus can potentially help shave off one √ H factor (see [8, 22] for a similar phenomenon in the tabular setting).

In contrast, the optimal dependency on dimension d is more important but is also less clear. In the case where the number of actions is very large, one may attempt to use the lower bound in the linear bandit setting, Ω( d √ T ) , for the case H = 1 . We comment that as soon as H ≥ 2 (where the Markov transition matters), the assumption of a linear MDP imposes structure on the feature space { φ ( x, a ) | ( x, a ) ∈ S × A} (see Proposition A.1). Technically, the standard constructions for the hard instances in the linear bandit lower bound do not respect this structure, so the lower bound does not directly apply. It remains an interesting future direction to determine this optimal dependency on d .

On the assumption of linear transition dynamics. The main assumption in this paper is the linear MDP assumption (Assumption A), which requires the Markov transition P h ( ·| x, a ) to be linear in φ ( x, a ) . This requirement could be strong in practice. It turns out that our proof only relies on a weaker version of this assumption:

<!-- formula-not-decoded -->

where w V is a vector independent of ( x, a ) and V is the class of value functions considered in this paper, as in Eq. (6). That is, we effectively only need that P h ( ·| x, a ) appears to be linear when we apply it to a value function V . When there is additional problem structure in the feature map φ so that V is relatively small and structured, Eq. (7) can potentially provide a usefully weaker condition compared to Assumption A.

When both the feature map φ and the policy π are fully generic, we comment that under mild conditions, the assumption of linear transition is then in fact necessary for the Bellman error to be zero for all policies π . Indeed, defining the Bellman operator T π h associated with π as

<!-- formula-not-decoded -->

/negationslash for any Q : S × A → R , we have the following proposition.

Proposition 5.1. Let Q = { Q | Q ( · , · ) = φ ( · , · ) /latticetop w , w ∈ R d } be the family of linear action-value functions. Suppose that S is a finite set, and for any x ∈ S , there exist two actions a, ¯ a ∈ A such that φ ( x, a ) = φ ( x, ¯ a ) . Then, T π h Q ⊂ Q for all π only if the Markov transition measures P h are linear in φ .

Finally, it remains an interesting future question whether an RL algorithm can be proved to be efficient without assuming a linear structure in the transition dynamics.

## Acknowledgements

We thank Alekh Agarwal, Zeyuan Allen-Zhu, Sebastian Bubeck, Nan Jiang and Akshay Krishnamurthy for valuable discussions. This work was supported in part by the DARPA program on Lifelong Learning Machines.

## References

- [1] Y. Abbasi-Yadkori and C. Szepesv´ ari. Regret bounds for the adaptive control of linear quadratic systems. In Conference on Learning Theory , pages 1-26, 2011.
- [2] Y. Abbasi-Yadkori, D. P´ al, and C. Szepesv´ ari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , pages 2312-2320, 2011.
- [3] Y. Abbasi-Yadkori, N. Lazic, and C. Szepesv´ ari. Model-free linear quadratic control via reduction to expert prediction. In International Conference on Artificial Intelligence and Statistics , pages 31083117, 2019.
- [4] M. Abeille and A. Lazaric. Improved regret bounds for Thompson sampling in linear quadratic control problems. In International Conference on Machine Learning , pages 1-9, 2018.
- [5] P. Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research , 3(Nov):397-422, 2002.
- [6] M. G. Azar, R. Munos, M. Ghavamzadaeh, and H. J. Kappen. Speedy Q-learning. In Advances in Neural Information Processing Systems , 2011.
- [7] M. G. Azar, R. Munos, and B. Kappen. On the sample complexity of reinforcement learning with a generative model. arXiv preprint arXiv:1206.6461 , 2012.
- [8] M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272, 2017.
- [9] K. Azizzadenesheli, E. Brunskill, and A. Anandkumar. Efficient exploration through bayesian deep q-networks. In 2018 Information Theory and Applications Workshop (ITA) , pages 1-9. IEEE, 2018.
- [10] L. Baird. Residual algorithms: Reinforcement learning with function approximation. In International Conference on Machine Learning , pages 30-37, 1995.
- [11] J. A. Boyan and A. W. Moore. Generalization in reinforcement learning: Safely approximating the value function. In Advances in Neural Information Processing Systems , pages 369-376, 1995.
- [12] S. J. Bradtke and A. G. Barto. Linear least-squares algorithms for temporal difference learning. Machine Learning , 22(1-3):33-57, 1996.
- [13] S. Bubeck and N. Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends R © in Machine Learning , 5(1):1-122, 2012.
- [14] W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In International Conference on Artificial Intelligence and Statistics , pages 208-214, 2011.

- [15] A. Cohen, T. Koren, and Y. Mansour. Learning linear-quadratic regulators efficiently with only √ T regret. arXiv preprint arXiv:1902.06223 , 2019.
- [16] V. Dani, T. P. Hayes, and S. M. Kakade. Stochastic linear optimization under bandit feedback. In Conference on Learning Theory , 2008.
- [17] C. Dann, T. Lattimore, and E. Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , pages 5713-5723, 2017.
- [18] S. Dean, H. Mania, N. Matni, B. Recht, and S. Tu. Regret bounds for robust adaptive control of the linear quadratic regulator. In Advances in Neural Information Processing Systems , pages 4188-4197, 2018.
- [19] S. S. Du, Y. Luo, R. Wang, and H. Zhang. Provably efficient Q-learning with function approximation via distribution shift error checking oracle. arXiv preprint arXiv:1906.06321 , 2019.
- [20] T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(4):1563-1600, 2010.
- [21] N. Jiang, A. Krishnamurthy, A. Agarwal, J. Langford, and R. E. Schapire. Contextual decision processes with low bellman rank are pac-learnable. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 1704-1713. JMLR. org, 2017.
- [22] C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873, 2018.
- [23] J. Kober and J. Peters. Reinforcement learning in robotics: A survey. In Reinforcement Learning , pages 579-610. Springer, 2012.
- [24] S. Koenig and R. G. Simmons. Complexity analysis of real-time reinforcement learning. In Association for the Advancement of Artificial Intelligence , pages 99-107, 1993.
- [25] T. Lattimore and M. Hutter. PAC bounds for discounted MDPs. In International Conference on Algorithmic Learning Theory , pages 320-334, 2012.
- [26] T. Lattimore and C. Szepesv´ ari. Bandit algorithms. preprint , 2018.
- [27] J. Li, W. Monroe, A. Ritter, M. Galley, J. Gao, and D. Jurafsky. Deep reinforcement learning for dialogue generation. arXiv preprint arXiv:1606.01541 , 2016.
- [28] L. Li, W. Chu, J. Langford, and R. E. Schapire. A contextual-bandit approach to personalized news article recommendation. In International Conference on World Wide Web , pages 661-670, 2010.
- [29] F. S. Melo and M. I. Ribeiro. Q-learning with linear function approximation. In International Conference on Computational Learning Theory , pages 308-322. Springer, 2007.
- [30] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller. Playing Atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.

- [31] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning , pages 1928-1937, 2016.
- [32] I. Osband and B. Van Roy. On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 , 2016.
- [33] I. Osband, B. Van Roy, and Z. Wen. Generalization and exploration via randomized value functions. arXiv preprint arXiv:1402.0635 , 2014.
- [34] M. L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp;Sons, 2014.
- [35] P. Rusmevichientong and J. N. Tsitsiklis. Linearly parameterized bandits. Mathematics of Operations Research , 35(2):395-411, 2010.
- [36] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In International Conference on Machine Learning , pages 1889-1897, 2015.
- [37] A. Sidford, M. Wang, X. Wu, and Y. Ye. Variance reduced value iteration and faster algorithms for solving Markov decision processes. In ACM-SIAMSymposium on Discrete Algorithms , pages 770-787, 2018.
- [38] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of Go with deep neural networks and tree search. Nature , 529(7587):484-489, 2016.
- [39] A. L. Strehl, L. Li, E. Wiewiora, J. Langford, and M. L. Littman. PAC model-free reinforcement learning. In International Conference on Machine Learning , pages 881-888, 2006.
- [40] R. S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning , 3(1): 9-44, 1988.
- [41] R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction . MIT Press, 2011.
- [42] C. Szepesv´ ari. Algorithms for reinforcement learning. Synthesis Lectures on Artificial Intelligence and Machine Learning , 4(1):1-103, 2010.
- [43] J. N. Tsitsiklis and B. Van Roy. Analysis of temporal-diffference learning with function approximation. In Advances in Neural Information Processing Systems , pages 1075-1081, 1997.
- [44] R. Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 , 2010.
- [45] M. J. Wainwright. Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 , 2019.
- [46] T. Wang, W. Ye, D. Geng, and C. Rudin. Towards practical Lipschitz stochastic bandits. arXiv preprint arXiv:1901.09277 , 2019.
- [47] Z. Wen and B. Van Roy. Efficient exploration and value function generalization in deterministic systems. In Advances in Neural Information Processing Systems , pages 3021-3029, 2013.

- [48] Z. Wen and B. Van Roy. Efficient reinforcement learning in deterministic systems with value function generalization. Mathematics of Operations Research , 42(3):762-782, 2017.
- [49] L. Yang and M. Wang. Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004, 2019.
- [50] L. F. Yang and M. Wang. Reinforcement leaning in feature space: Matrix bandit, kernels, and regret bound. arXiv preprint arXiv:1905.10389 , 2019.
- [51] X. Zhu and D. B. Dunson. Lipschitz bandit optimization with improved efficiency. arXiv preprint arXiv:1904.11131 , 2019.

## A Properties of Linear MDP

In this section, we present some of the basic properties of linear MDPs.

We start with the most important property of a linear MDP: the action-value function is always linear in the feature map φ for any policy.

Proposition 2.3. For a linear MDP, for any policy π , there exist weights { w π h } h ∈ [ H ] such that for any ( x, a, h ) ∈ S × A × [ H ] , we have Q π h ( x, a ) = 〈 φ ( x, a ) , w π h 〉 .

Proof. The linearity of the action-value functions directly follows from the Bellman equation in Eq. (1):

<!-- formula-not-decoded -->

Second, we show that, under mild conditions, the assumption of a linear transition is necessary for the Bellman error to be zero for all policies π .

Therefore, we have Q π h ( x, a ) = 〈 φ ( x, a ) , w π h 〉 where w π h is given by w π h = θ h + ∫ S V π h +1 ( x ′ ) d µ h ( x ′ ) . /square

/negationslash

Proposition 5.1. Let Q = { Q | Q ( · , · ) = φ ( · , · ) /latticetop w , w ∈ R d } be the family of linear action-value functions. Suppose that S is a finite set, and for any x ∈ S , there exist two actions a, ¯ a ∈ A such that φ ( x, a ) = φ ( x, ¯ a ) . Then, T π h Q ⊂ Q for all π only if the Markov transition measures P h are linear in φ .

/negationslash

Proof. For any fixed state x 0 ∈ S , by assumption, there exist two actions a 0 and ¯ a 0 such that φ ( x 0 , a 0 ) = φ ( x 0 , ¯ a 0 ) . Then there exists w 0 ∈ R d such that

<!-- formula-not-decoded -->

We define the function Q 0 ( · , · ) = φ ( · , · ) /latticetop w 0 . Additionally, let two policies π 1 and π 2 satisfy

<!-- formula-not-decoded -->

Now consider T π 1 h Q 0 -T π 2 h Q 0 for any h . By the definition of Bellman operator in Eq. (8), for any ( x, a ) ∈ S × A , we have where the second equality holds due to Eq. (10). Thus, by combining Eq. (9) and Eq. (11), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since T π h Q ⊂ Q for all π , we know both T π 1 h Q 0 and T π 2 h Q 0 are elements of Q , so is P h ( x 0 | · , · ) , which implies that P h ( x 0 | · , · ) is a linear function of φ ( · , · ) . That is, there exists a vector µ ( x 0 ) independent of ( x, a ) so that P h ( x 0 | x, a ) = 〈 φ ( x, a ) , µ ( x 0 ) 〉 for all ( x, a ) . Because this holds for all x 0 ∈ S , we have P h ( · | x, a ) = 〈 φ ( x, a ) , µ ( · ) 〉 . This concludes the proof. /square

Finally, we note Assumption A also implicitly enforces the following structure on the feature space since P h ( ·| x, a ) must be a probability measure over S for any ( x, a ) ∈ S × A .

Proposition A.1. For a linear MDP, for any ( x, a, h ) ∈ S × A × [ H ] , we have

<!-- formula-not-decoded -->

Proof. This proposition immediately follows from the fact that P h ( · | x, a ) is a probability measure over S for any ( x, a, h ) ∈ S × A × [ H ] . /square

In particular, the first condition in Eq. (12) requires the image of φ , { φ ( x, a ) | ( x, a ) ∈ S × A} , to be contained in a ( d -1) -dimensional hyperphane.

## B Proof of Theorem 3.1

In this section, we prove Theorem 3.1. We first introduce the notation that is used throughout this section. Then, we present lemmas and their proofs. Finally, we combine the lemmas to prove Theorem 3.1.

Notation: Throughout this section, we denote Λ k h , w k h , and Q k h as the parameters and the Q-value function estimate in episode k . Denote value function V k h as V k h ( x ) = max a Q k h ( x, a ) . We also denote π k as the greedy policy induced by { Q k h } H h =1 . To simplify our presentation, we always denote φ k h := φ ( x k h , a k h ) .

First, we prove two lemmas which state that the linear weights w h in both the action-value functions and Algorithm 1 are bounded.

Lemma B.1 (Bound on Weights of Value Functions) . Under Assumption A, for any fixed policy π , let { w π h } h ∈ [ H ] be the corresponding weights such that Q π h ( x, a ) = 〈 φ ( x, a ) , w π h 〉 for all ( x, a, h ) ∈ S×A× [ H ] . Then, we have

Proof. By the Bellman equation in Eq. (1), we know, for any h ∈ [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since MDP is linear, by definition, this gives:

Under the normalization conditions of Assumption A, the reward at each step is in [0 , 1] , thus V π h +1 ( x ′ ) ≤ H for any state x ′ . Therefore, ‖ θ h ‖ ≤ √ d , and ∥ ∫ V π h +1 ( x ′ )d µ h ( x ′ ) ∥ ≤ H √ d , which finishes the proof. /square

<!-- formula-not-decoded -->

∥ ∥ Lemma B.2 (Bound on Weights in Algorithm) . For any ( k, h ) ∈ [ K ] × [ H ] , the weight w k h in Algorithm 1 satisfies:

Proof. For any vector v ∈ R d , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step is due to Lemma D.1. The remainder of the proof follows from the fact that ∥ ∥ w k h ∥ ∥ = max v : ‖ v ‖ =1 | v /latticetop w k h | . /square

Second, we present our main concentration lemma, which is crucial in controlling the fluctuations in least-squares value iteration.

Lemma B.3. Under the setting of Theorem 3.1, let c β be the constant in our definition of β (i.e., β = c β · dH √ ι ). There exists an absolute constant C that is independent of c β such that for any fixed p ∈ [0 , 1] , if we let E be the event that:

<!-- formula-not-decoded -->

Proof. For all ( k, h ) ∈ [ K ] × [ H ] , by Lemma B.2 we have ‖ w k h ‖ ≤ 2 H √ dk/λ . In addition, by the construction of Λ k h , the minimum eigenvalue of Λ k h is lower bounded by λ . Thus, by combining Lemmas D.4 and D.6, we have for any fixed ε &gt; 0 that:

where χ = log[2( c β +1) dT/p ] , then P ( E ) ≥ 1 -p/ 2 .

<!-- formula-not-decoded -->

Notice that we choose the hyperparameters λ = 1 and β = C · dHι where C is an absolute constant. Finally, picking ε = dH/k , by Eq. (13), there exists a absolute constant C &gt; 0 that is independent of c β such that

<!-- formula-not-decoded -->

which concludes the proof.

/square

Next, we recursively bound the difference between the value function maintained in Algorithm 1 (without bonus) and the true value function of any policy π . We bound this difference using their expected difference at next step, plus a error term. This error term can be upper bounded by our bonus with high probability. This is the key technical lemma in this section.

Lemma B.4. There exists an absolute constant c β such that for β = c β · dH √ ι where ι = log(2 dT/p ) , and for any fixed policy π , on the event E defined in Lemma B.3, we have for all ( x, a, h, k ) ∈ S ×A× [ H ] × [ K ] that:

<!-- formula-not-decoded -->

for some ∆ k h ( x, a ) that satisfies | ∆ k h ( x, a ) | ≤ β φ ( x, a ) /latticetop (Λ k h ) -1 φ ( x, a )

<!-- formula-not-decoded -->

Proof. By Proposition 2.3 and the Bellman equation Eq. (1), we know for any ( x, a, h ) ∈ S × A × [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This gives:

Now, we bound the terms on the right-hand side individually. For the first term,

<!-- formula-not-decoded -->

For the second term, given the event E defined in Lemma B.3, we have:

<!-- formula-not-decoded -->

for an absolute constant c 0 independent of c β , and χ = log[2( c β +1) dT/p ] . For the third term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, since 〈 φ ( x, a ) , w k h 〉 -Q π h ( x, a ) = 〈 φ ( x, a ) , w k h -w π h 〉 = 〈 φ ( x, a ) , q 1 + q 2 + q 3 〉 , by Lemma B.1 and our choice of parameter λ , we have

<!-- formula-not-decoded -->

for an absolute constant c ′ independent of c β . Finally, to prove this lemma, we only need to show that there exists a choice of absolute constant c β so that

<!-- formula-not-decoded -->

where ι = log(2 dT/p ) . Weknow ι ∈ [log 2 , ∞ ) by its definition, and c ′ is an absolute constant independent of c β . Therefore, we can pick an absolute constant c β which satisfies c ′ √ log 2 + log( c β +1) ≤ c β √ log 2 . This choice of c β will make Eq. (14) hold for all ι ∈ [log 2 , ∞ ) , which finishes the proof. /square

Lemma B.4 implies that by adding appropriate bonuses, Q k h in Algorithm 1 can be always an upper bound of Q /star h with high confidence.

Lemma B.5 (UCB) . Under the setting of Theorem 3.1, on the event E defined in Lemma B.3, we have Q k h ( x, a ) ≥ Q /star h ( x, a ) for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] .

Proof. We prove this lemma by induction.

First, we prove the base case, at the last step H . The statement holds because Q k H ( x, a ) ≥ Q /star H ( x, a ) . Since the value function at H +1 step is zero, by Lemma B.4, we have:

Therefore, we know:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, suppose the statement holds true at step h +1 and consider step h . Again, by Lemma B.4, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the induction assumption that P h ( V k h +1 -V /star h +1 )( x, a ) ≥ 0 , we have:

which concludes the proof.

/square

Lemma B.4 also easily transforms to a recursive formula for δ k h = V k h ( x k h ) -V π k h ( x k h ) . This formula will be very useful in proving the main theorem.

Lemma B.6 (Recursive formula) . Let δ k h = V k h ( x k h ) -V π k h ( x k h ) , and ζ k h +1 = E [ δ k h +1 | x k h , a k h ] -δ k h +1 . Then, on the event E defined in Lemma B.3, we have the following for any ( k, h ) ∈ [ K ] × [ H ] :

Proof. By Lemma B.4 we have that for any ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and finally, by Algorithm 1 and the definition of V π k we have which finishes the proof.

<!-- formula-not-decoded -->

Finally, we are ready to prove the main theorem. We restate our main theorem as follows.

Theorem 3.1. Under Assumption A, there exists an absolute constant c &gt; 0 such that, for any fixed p ∈ (0 , 1) , if we set λ = 1 and β = c · dH √ ι in Algorithm 1 with ι := log(2 dT/p ) , then with probability 1 -p , the total regret of LSVI-UCB (Algorithm 1) is at most O ( √ d 3 H 3 Tι 2 ) , where O ( · ) hides only absolute constants.

Proof. We use the notion of δ k h and ζ k h as in Lemma B.6. We condition on the event E defined in Lemma B.3 with δ = p/ 2 . By Lemmas B.5 and B.6, we have

<!-- formula-not-decoded -->

We now bound the two terms on the right-hand side of Eq. (15) separately. For the first term, since the computation of V k h is independent of the new observation x k h at episode k , we obtain that { ζ k h } is a martingale difference sequence satisfying | ζ k h | ≤ 2 H for all ( k, h ) . Therefore, by the Azuma-Hoeffding inequality, for any t &gt; 0 , we have

<!-- formula-not-decoded -->

Hence, with probability at least 1 -p/ 2 , we have

<!-- formula-not-decoded -->

where ι = log(2 dT/p ) . Furthermore, for the second term, note that the minimum eigenvalue of Λ k h is at least λ (which equals to 1) for all ( k, h ) ∈ [ K ] × [ H ] . Also notice that ‖ φ k h ‖ ≤ 1 . By Lemma D.2, for any h ∈ [ H ] , we have

Moreover, note that ‖ Λ k +1 h ‖ = ‖ ∑ k τ =1 φ k h ( φ k h ) /latticetop + λ I ‖ ≤ λ + k ; this gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, by the Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

which yields an upper bound on the second term in Eq. (15). Finally, combining Eq. (15), Eq. (16), Eq. (18), and with our choice of β = c · dH √ ι for some absolute constant c, we conclude that with probability 1 -p :

<!-- formula-not-decoded -->

for some absolute constant c ′ . This concludes the proof.

/square

## C Proof of Theorem 3.2

In this section, we prove Theorem 3.2. At a high level, the proof structure is similar to the structure in Appendix B. We will particularly focus on the parts that require different treatments in the misspecified setting.

Notation: Throughout this section, we denote Λ k h , w k h , and Q k h as the parameters and the Q-value functions estimated in episode k . Denote the value function V k h as V k h ( x ) = max a Q k h ( x, a ) . We denote π k as the greedy policy induced by { Q k h } H h =1 . To simplify the presentation, we denote φ k h := φ ( x k h , a k h ) .

First, we establish a lemma that is the counterpart of Lemma 2.3 in the misspecified setting: for any policy π , its action-value function is always close to a linear function.

Lemma C.1. For a ζ -nearly linear MDP, for any policy π , there exist corresponding weights { w π h } h ∈ [ H ] where w π h = θ h + ∫ V π h +1 ( x ′ )d µ h ( x ′ ) such that for any ( x, a, h ) ∈ S × A × [ H ] :

<!-- formula-not-decoded -->

Proof. Since µ h and θ h satisfy Eq.(4), we have:

<!-- formula-not-decoded -->

which finishes the proof.

We can again show that the linear weights defined in Lemma C.1 are bounded.

LemmaC.2 (Bound on Weights of Value Functions) . Under Assumption B, for any policy π , let { w π h } h ∈ [ H ] be the corresponding weights as defined in Lemma C.1. Then, we have

<!-- formula-not-decoded -->

Proof. Under the normalization conditions of Assumption B, the reward at each step is in [0 , 1] , thus V π h +1 ( x ′ ) ≤ H for any state x ′ . Therefore, ‖ θ h ‖ ≤ √ d , and ∥ ∥ ∫ V π h +1 ( x ′ )d µ h ( x ′ ) ∥ ∥ ≤ H √ d , which finishes the proof. /square

Similar to Lemma B.3, we also bound the stochastic noise in concentration.

Lemma C.3. Under the setting of Theorem 3.2, let c β be the constant in our choice of β k (i.e. β k = c β · ( d √ ι + ζ √ kd ) H ), There exists an absolute constant C that is independent of c β such that for any fixed p ∈ [0 , 1] , if we let E be the event that:

where χ = log[2( c β +1) dT/p ] , then P ( E ) ≥ 1 -p/ 2 .

<!-- formula-not-decoded -->

/square

Proof. The proof is essentially the same as the proof for Lemma B.3, with the only difference that β k is now bounded by c β ( d √ ι + ζ √ Kd ) H instead of c β dH √ ι as in Lemma B.3. Because ζ ≤ 1 as in Assumption B, the new bound of β k only affects the choice of absolute C in Lemma C.3. /square

In the misspecified case, we also need to bound an error term where the noise can be potentially adversarial instead of stochastic. The adversarial noise is precisely due to model misspecification.

LemmaC.4. Let { ε τ } be any sequence so that | ε τ | ≤ B for any τ . Then, we have for any ( h, k ) ∈ [ H ] × [ K ] and any φ ∈ R d that:

Proof. By the Cauchy-Schwarz inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to Lemma D.1.

/square

Now we are ready to prove the key lemma, which is the counterpart of Lemma B.4.

Lemma C.5. There exists an absolute constant c β such that for β k = c β · ( d √ ι + ζ √ kd ) H where ι = log(2 dT/p ) , and for any fixed policy π , on the event E defined in Lemma C.3, we have for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] that:

<!-- formula-not-decoded -->

Proof. By Lemma C.1, there exists w π h = θ h + ∫ V π h +1 ( x ′ )d µ h ( x ′ ) so that for any ( x, a, h ) ∈ S ×A× [ H ] :

for some ∆ k h ( x, a ) that satisfies | ∆ k h ( x, a ) | ≤ β k √ φ ( x, a ) /latticetop (Λ k h ) -1 φ ( x, a ) + 4 Hζ .

<!-- formula-not-decoded -->

On the other hand, let ˜ P ( ·| x, a ) = 〈 φ ( x, a ) , µ h ( · ) 〉 . Then, for any ( x, a ) ∈ S × A , we have

<!-- formula-not-decoded -->

This further gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second term, given the event E defined in Lemma C.3, we have:

<!-- formula-not-decoded -->

for an absolute constant c 0 independent of c β , and χ = log[2( c β +1) dT/p ] . For the third term,

<!-- formula-not-decoded -->

where by definition of P h , we have

Since ‖ ˜ P h -P h ‖ TV ≤ ζ , we have

<!-- formula-not-decoded -->

For the fourth term, by Lemma C.4, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, since 〈 φ ( x, a ) , w k h -w π h 〉 = 〈 φ ( x, a ) , q 1 + q 2 + q 3 + q 4 〉 , we have:

for an absolute constant c ′ independent of c β . As in the proof of Lemma B.4, to prove this lemma, we only need to show that there exists a choice of absolute constant c β so that c β ≥ 2 , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given Lemma C.5, we can now easily proceed to prove that Q k h is a upper bound of Q /star h up to an error that depends linearly on the misspecification ζ .

This can be done by an picking absolute constant c β that satisfies c ′ √ log 2 + log( c β +1) ≤ c β √ log 2 . /square

Lemma C.6 (UCB) . Under the setting of Theorem 3.2, on the event E defined in Lemma C.3, we have Q k h ( x, a ) ≥ Q /star h ( x, a ) -4 H ( H +1 -h ) ζ for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] .

Proof. We prove this lemma by induction.

First, we consider the base case. The statement holds for the last step H , i.e., Q k H ( x, a ) ≥ Q /star H ( x, a ) -4 Hζ . Since the value function at H +1 step is zero, by Lemma C.5, we have:

<!-- formula-not-decoded -->

Therefore, we obtain that

<!-- formula-not-decoded -->

Now, suppose the statement holds true at step h +1 , and consider step h . Again, by Lemma C.5, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the induction assumption that P h ( V k h +1 -V /star h +1 )( x, a ) ≥ -4 H ( H -h ) ζ , we have:

Therefore, we conclude the proof of this lemma.

/square

The gap δ k h = V k h ( x k h ) -V π k h ( x k h ) also has a recursive formula similar to Lemma B.6.

<!-- formula-not-decoded -->

Lemma C.7 (Recursive formula) . Let δ k h = V k h ( x k h ) -V π k h ( x k h ) , and ζ k h +1 = E [ δ k h +1 | x k h , a k h ] -δ k h +1 . Then, on the event E defined in Lemma C.3, we have the following for any ( k, h ) ∈ [ K ] × [ H ] :

Proof. This is because by Lemma C.5, we have for any ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] :

Finally, by Algorithm 1 and the definition of V π k we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which finishes the proof.

Finally, we are ready to combine all previous lemmas to prove the main theorem in the misspecified setting.

Theorem 3.2. Under Assumption B, there exists an absolute constant c &gt; 0 such that, for any fixed p ∈ (0 , 1) , if we set λ = 1 and β k = c · ( d √ ι + ζ √ kd ) H in Algorithm 1 with ι := log(2 dT/p ) , then with probability 1 -p , the total regret of LSVI-UCB (Algorithm 1) is at most O ( √ d 3 H 3 Tι 2 + ζdHT √ ι ) .

Proof of Theorem 3.2. The proof of this theorem is similar to that of Theorem 3.1. We condition on the event E defined in Lemma C.3. For for any ( k, h ) ∈ [ K ] × [ H ] , we define δ k h = V k h ( x k h ) -V π k h ( x k h ) . ByLemmaC.6, we have Q k 1 ( x, a ) ≥ Q ∗ 1 ( x, a ) -4 H 2 ζ for all k ∈ [ K ] , which implies that V /star 1 ( x k 1 ) -V π k 1 ( x k 1 ) ≤ δ k 1 +4 H 2 ζ . Furthermore, by Lemma C.7, on the event E we have:

<!-- formula-not-decoded -->

where we use the fact that T = HK . Since { ζ k h } is a martingale difference sequence with each term bounded by 2 H , the Azuma-Hoeffding inequality implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds with probability at least 1 -p/ 2 , where ι = log(2 dT/p ) . Moreover, by the Cauchy-Schwarz inequality, we have

Similarly to Eq. (17), we have

Moreover, since we set β k = c · ( d √ ι + ζ √ kd ) H for some absolute constant c &gt; 0 , we have which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, combining Eq. (21), Eq. (22), and Eq. (23), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, combining Eq. (19), Eq. (20), and Eq. (24), we obtain

′′ /square for some absolute constant c . This concludes the proof of the theorem.

## D Auxiliary Lemmas

This section presents several auxiliary lemmas and their proofs.

## D.1 Important inequalities for summations

First, we present a few important short inequalities for summations.

Lemma D.1. Let Λ t = λ I + ∑ t i =1 φ i φ /latticetop i where φ i ∈ R d and λ &gt; 0 . Then:

Proof. We have ∑ t i =1 φ /latticetop i (Λ t ) -1 φ i = ∑ t i =1 tr( φ /latticetop i (Λ t ) -1 φ i ) = tr((Λ t ) -1 ∑ t i =1 φ i φ /latticetop i ) . Given the eigenvalue decomposition ∑ t i =1 φ i φ /latticetop i = U diag( λ 1 , . . . , λ d ) U /latticetop , we have Λ t = U diag( λ 1 + λ, . . . , λ d + λ ) U /latticetop , and tr((Λ t ) -1 ∑ t i =1 φ i φ /latticetop i ) = ∑ d j =1 λ j / ( λ j + λ ) ≤ d . /square

<!-- formula-not-decoded -->

Lemma D.2 ([2]) . Let { φ t } t ≥ 0 be a bounded sequence in R d satisfying sup t ≥ 0 ‖ φ t ‖ ≤ 1 . Let Λ 0 ∈ R d × d be a positive definite matrix. For any t ≥ 0 , we define Λ t = Λ 0 + ∑ t j =1 φ /latticetop j φ j . Then, if the smallest eigenvalue of Λ 0 satisfies λ min (Λ 0 ) ≥ 1 , we have

<!-- formula-not-decoded -->

Proof. Since λ min (Λ 0 ) ≥ 1 and ‖ φ t ‖ ≤ 1 for all j ≥ 0 , we have

<!-- formula-not-decoded -->

Note that, for any x ∈ [0 , 1] , it holds that log(1 + x ) ≤ x ≤ 2log(1 + x ) . Therefore, we have

<!-- formula-not-decoded -->

Moreover, for any t ≥ 0 , by the definition of Λ t , we have

<!-- formula-not-decoded -->

Since det( I +Λ -1 / 2 t -1 φ t φ /latticetop t Λ -1 / 2 t -1 ) = 1 + φ /latticetop t Λ -1 t -1 φ t , the recursion gives:

<!-- formula-not-decoded -->

Therefore, combining Eq. (25) and Eq. (26), we conclude the proof.

/square

## D.2 Concentration inequalities for self-normalized processes

Next, we present a few concentration inequalities. The following one provides a concentration inequality for the standard self-normalized processes.

Theorem D.3 (Concentration of Self-Normalized Processes [2]) . Let { ε t } ∞ t =1 be a real-valued stochastic process with corresponding filtration {F t } ∞ t =0 . Let ε t |F t -1 be zero-mean and σ -subGaussian; i.e. E [ ε t |F t -1 ] = 0 , and

Let { φ t } ∞ t =0 be an R d -valued stochastic process where φ t ∈ F t -1 . Assume Λ 0 is a d × d positive definite matrix, and let Λ t = Λ 0 + ∑ t s =1 φ s φ /latticetop s . Then for any δ &gt; 0 , with probability at least 1 -δ , we have for all t ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When specializing this concentration inequality to our setting, we require uniform concentration over all value functions V within a function class V . This uniform concentration incurs an additional term that depends logarithmically on the covering number of V .

Lemma D.4. Let { x τ } ∞ τ =1 be a stochastic process on state space S with corresponding filtration {F τ } ∞ τ =0 . Let { φ τ } ∞ τ =0 be an R d -valued stochastic process where φ τ ∈ F τ -1 , and ‖ φ τ ‖ ≤ 1 . Let Λ k = λI + ∑ k τ =1 φ τ φ /latticetop τ . Then for any δ &gt; 0 , with probability at least 1 -δ , for all k ≥ 0 , and any V ∈ V so that sup x | V ( x ) | ≤ H , we have:

V ∈ V V ε

<!-- formula-not-decoded -->

This gives following decomposition:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we can apply Theorem D.3 and a union bound to the first term. Also, it is not hard to bound the second term by 8 k 2 ε 2 /λ . /square

To compute the covering number of function class V , we first require a basic result on the covering number of a Euclidean ball as follows. We refer readers to classical material, such as Lemma 5.2 in [44], for its proof.

Lemma D.5 (Covering Number of Euclidean Ball) . For any ε &gt; 0 , the ε -covering number of the Euclidean ball in R d with radius R &gt; 0 is upper bounded by (1 + 2 R/ε ) d .

Now, we are ready to compute the covering number of V .

Lemma D.6. Let V denote a class of functions mapping from S to R with following parametric form where the parameters ( w , β, Λ) satisfy ‖ w ‖ ≤ L , β ∈ [0 , B ] and the minimum eigenvalue satisfies λ min (Λ) ≥ λ . Assume ‖ φ ( x, a ) ‖ ≤ 1 for all ( x, a ) pairs, and let N ε be the ε -covering number of V with respect to the distance dist( V, V ′ ) = sup x | V ( x ) -V ′ ( x ) | . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Equivalently, we can reparametrize the function class V by let A = β 2 Λ -1 , so we have for ‖ w ‖ ≤ L and ‖ A ‖ ≤ B 2 λ -1 . For any two functions V 1 , V 2 ∈ V , let them take the form in Eq. (27) with parameters ( w 1 , A 1 ) and ( w 2 , A 2 ) , respectively. Then, since both min {· , H } and max a are contraction maps, we have

<!-- formula-not-decoded -->

Let C w be an ε/ 2 -cover of { w ∈ R d | ‖ w ‖ ≤ L } with respect to the 2-norm, and C A be an ε 2 / 4 -cover of { A ∈ R d × d | ‖ A ‖ F ≤ d 1 / 2 B 2 λ -1 } with respect to the Frobenius norm. By Lemma D.5, we know:

where the second last inequality follows from the fact that | √ x - √ y | ≤ √ | x -y | holds for any x, y ≥ 0 . For matrices, ‖·‖ and ‖·‖ F denote the matrix operator norm and Frobenius norm respectively.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Eq. (28), for any V 1 ∈ V , there exists w 2 ∈ C w and A 2 ∈ C A such that V 2 parametrized by ( w 2 , A 2 ) satisfies dist( V 1 , V 2 ) ≤ ε . Hence, it holds that N ε ≤ |C w | · |C A | , which gives:

This concludes the proof.

/square