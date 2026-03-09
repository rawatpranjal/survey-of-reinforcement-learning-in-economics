## Near-optimal Regret Bounds for Reinforcement Learning ∗

Thomas Jaksch Ronald Ortner Peter Auer

Chair for Information Technology University of Leoben Franz-Josef-Strasse 18 8700 Leoben, Austria

Editor:

Sham Kakade

TJAKSCH@UNILEOBEN.AC.AT

RORTNER@UNILEOBEN.AC.AT AUER@UNILEOBEN.AC.AT

## Abstract

For undiscounted reinforcement learning in Markov decision processes (MDPs) we consider the total regret of a learning algorithm with respect to an optimal policy. In order to describe the transition structure of an MDP we propose a new parameter: An MDP has diameter D if for any pair of states s , s ′ there is a policy which moves from s to s ′ in at most D steps (on average). We present a reinforcement learning algorithm with total regret ˜ O ( DS √ AT ) after T steps for any unknown MDP with S states, A actions per state, and diameter D . A corresponding lower bound of Ω ( √ DSAT ) on the total regret of any learning algorithm is given as well.

These results are complemented by a sample complexity bound on the number of suboptimal steps taken by our algorithm. This bound can be used to achieve a (gap-dependent) regret bound that is logarithmic in T .

Finally, we also consider a setting where the MDP is allowed to change a fixed number of /lscript times. We present a modification of our algorithm that is able to deal with this setting and show a regret bound of ˜ O ( /lscript 1 / 3 T 2 / 3 DS √ A ) .

Keywords: undiscounted reinforcement learning, Markov decision process, regret, online learning, sample complexity

## 1. Introduction

In a Markov decision process (MDP) M with finite state space S and finite action space A , a learner in some state s ∈ S needs to choose an action a ∈ A . When executing action a in state s , the learner receives a random reward r drawn independently from some distribution on [ 0 , 1 ] with mean ¯ r ( s , a ) . Further, according to the transition probabilities p ( s ′ | s , a ) , a random transition to a state s ′ ∈ S occurs.

Reinforcement learning of MDPs is a standard model for learning with delayed feedback. In contrast to important other work on reinforcement learning-where the performance of the learned policy is considered (see, e.g., Sutton and Barto 1998, Kearns and Singh 1999, and also the discussion and references given in the introduction of Kearns and Singh 2002)-we are interested in the performance of the learning algorithm during learning . For that, we compare the rewards collected by the algorithm during learning with the rewards of an optimal policy.

∗ . Anextended abstract of this paper appeared in Advances in Neural Information Processing Systems 21 (2009), pp. 8996.

An algorithm A starting in an initial state s of an MDP M chooses at each time step t (possibly randomly) an action at . As the MDP is assumed to be unknown except the sets S and A , usually an algorithm will map the history up to step t to an action at or, more generally, to a probability distribution over A . Thus, an MDP M and an algorithm A operating on M with initial state s constitute a stochastic process described by the states st visited at time step t , the actions at chosen by A at step t , and the rewards rt obtained ( t ∈ N ). In this paper we will consider undiscounted rewards. Thus, the accumulated reward of an algorithm A after T steps in an MDP M with initial state s , defined as

<!-- formula-not-decoded -->

is a random variable with respect to the mentioned stochastic process. The value 1 T E [ R ( M , A , s , T )] then is the expected average reward of the process up to step T . The limit

<!-- formula-not-decoded -->

is called the average reward and can be maximized by an appropriate stationary policy π : S → A which determines an optimal action for each state (see Puterman, 1994). Thus, in what follows we will usually consider policies to be stationary.

The difficulty of learning an optimal policy in an MDP does not only depend on the MDP's size (given by the number of states and actions), but also on its transition structure. In order to measure this transition structure we propose a new parameter, the diameter D of an MDP. The diameter D is the time it takes to move from any state s to any other state s ′ , using an appropriate policy for each pair of states s , s ′ :

Definition 1 Consider the stochastic process defined by a stationary policy π : S → A operating on an MDP M with initial state s. Let T ( s ′ | M , π , s ) be the random variable for the first time step in which state s ′ is reached in this process. Then the diameter of M is defined as

/negationslash

In Appendix A we show that the diameter is at least log | A | | S | -3. On the other hand, depending on the existence of states that are hard to reach under any policy, the diameter may be arbitrarily large. (For a comparison of the diameter to other mixing time parameters see below.)

<!-- formula-not-decoded -->

In any case, a finite diameter seems necessary for interesting bounds on the regret of any algorithm with respect to an optimal policy. When a learner explores suboptimal actions, this may take him into a 'bad part' of the MDP from which it may take up to D steps to reach again a 'good part' of the MDP. Thus, compared to the simpler multi-armed bandit problem where each arm a is typically explored log T g times (depending on the gap g between the optimal reward and the reward for arm a )-see, for example, the regret bounds of Auer et al. (2002a) for the UCB algorithms and the lower bound of Mannor and Tsitsiklis (2004)-the best one would expect for the general MDP setting is a regret bound of Θ ( D | S || A | log T ) . The alternative gap-independent regret bounds of ˜ O ( √ | B | T ) and Ω ( √ | B | T ) for multi-armed bandits with | B | arms (Auer et al., 2002b) correspondingly translate into a regret bound of Θ ( D | S || A | T ) for MDPs with diameter D .

√ For MDPs with finite diameter (which usually are called communicating , see, e.g., Puterman 1994) the optimal average reward ρ ∗ does not depend on the initial state (cf. Puterman 1994, Section 8.3.3), and we set

<!-- formula-not-decoded -->

The optimal average reward is the natural benchmark 1 for a learning algorithm A , and we define the total regret of A after T steps as

<!-- formula-not-decoded -->

In the following, we present our reinforcement learning algorithm UCRL2 (a variant of the UCRL algorithm of Auer and Ortner, 2007) which uses upper confidence bounds to choose an optimistic policy. We show that the total regret of UCRL2 after T steps is ˜ O ( D | S | √ | A | T ) . A corresponding lower bound of Ω ( √ D | S || A | T ) on the total regret of any learning algorithm is given as well. These results establish the diameter as an important parameter of an MDP. Unlike other parameters that have been proposed for various PAC and regret bounds, such as the mixing time (Kearns and Singh, 2002; Brafman and Tennenholtz, 2002) or the hitting time of an optimal policy (Tewari and Bartlett, 2008) (cf. the discussion below) the diameter only depends on the MDP's transition structure.

## 1.1 Relation to Previous Work

We first compare our results to the PAC bounds for the well-known algorithms E 3 of Kearns and Singh (2002), and R-Max of Brafman and Tennenholtz (2002) (see also Kakade, 2003). These algorithms achieve ε -optimal average reward with probability 1 -δ after time polynomial in 1 δ , 1 ε , | S | , | A | , and the mixing time T mix ε (see below). As the polynomial dependence on ε is of order 1 ε 3 , the PAC bounds translate into T 2 / 3 regret bounds at the best. Moreover, both algorithms need the ε -return mixing time T mix ε of an optimal policy π ∗ as input parameter. 2 This parameter T mix ε is the number of steps until the average reward of π ∗ over these T mix ε steps is ε -close to the optimal average reward ρ ∗ . It is easy to construct MDPs of diameter D with T mix ε ≈ D ε . This additional dependence on ε further increases the exponent in the above mentioned regret bounds for E 3 and R-max. Also, the exponents of the parameters | S | and | A | in the PAC bounds of Kearns and Singh (2002) and Brafman and Tennenholtz (2002) are substantially larger than in our bound. However, there are algorithms with better dependence on these parameters. Thus, in the sample complexity bounds for the Delayed Q-Learning algorithm of Strehl et al. (2006) the dependence on states and actions is of order | S || A | , however at the cost of a worse dependence of order 1 ε 4 on ε .

The MBIE algorithm of Strehl and Littman (2005, 2008)-similarly to our approach-applies confidence bounds to compute an optimistic policy. However, Strehl and Littman consider only a discounted reward setting. Their definition of regret measures the difference between the rewards 3 of an optimal policy and the rewards of the learning algorithm along the trajectory taken by the learning algorithm . In contrast, we are interested in the regret of the learning algorithm in respect to the rewards of the optimal policy along the trajectory of the optimal policy . 4 Generally, in discounted reinforcement learning only a finite number of steps is relevant, depending on the discount

1. It can be shown that max A E [ R ( M , A , s , T )] = T ρ ∗ ( M ) + O ( D ( M )) and max A R ( M , A , s , T ) = T ρ ∗ ( M ) + ˜ O ( √ T ) with high probability.

2. The knowledge of this parameter can be eliminated by guessing T mix ε to be 1 , 2 , . . . , so that sooner or later the correct T mix ε will be reached (cf. Kearns and Singh 2002; Brafman and Tennenholtz 2002). However, since there is no condition on when to stop increasing T mix ε , the assumed mixing time eventually becomes arbitrarily large, so that the PAC bounds become exponential in the true T mix ε (cf. Brafman and Tennenholtz, 2002).

3. Actually, the state values.

4. Indeed, one can construct MDPs for which these two notions of regret differ significantly. E.g., set the discount factor γ = 0. Then any policy which maximizes immediate rewards achieves 0 regret in the notion of Strehl and Littman. But such a policy may not move to states where the optimal reward is obtained.

factor. This makes discounted reinforcement learning similar to the setting with trials of constant length from a fixed initial state as considered by Fiechter (1994). For this case logarithmic online regret bounds in the number of trials have already been given by Auer and Ortner (2005). Also, the notion of regret is less natural than in undiscounted reinforcement learning: when summing up the regret in the individual visited states to obtain the total regret in the discounted setting, somehow contrary to the principal idea of discounting, the regret at each time step counts the same.

Tewari and Bartlett (2008) propose a generalization of the index policies of Burnetas and Katehakis (1997). These index policies choose actions optimistically by using confidence bounds only for the estimates in the current state. The regret bounds for the index policies of Burnetas and Katehakis (1997) and the OLP algorithm of Tewari and Bartlett (2008) are asymptotically logarithmic in T . However, unlike our bounds, these bounds depend on the gap between the 'quality' of the best and the second best action, and these asymptotic bounds also hide an additive term which is exponential in the number of states. Actually, it is possible to prove a corresponding gap-dependent logarithmic bound for our UCRL2 algorithm as well (cf. Theorem 4 below). This bound holds uniformly over time and under weaker assumptions: While Tewari and Bartlett (2008) and Burnetas and Katehakis (1997) consider only ergodic MDPs in which any policy will reach every state after a sufficient number of steps, we make only the more natural assumption of a finite diameter.

Recently, Bartlett and Tewari (2009) have introduced the REGAL algorithm (inspired by our UCRL2 algorithm) and show-based on the methods we introduce in this paper-regret bounds where the diameter is replaced with a smaller transition parameter D 1 (that is basically an upper bound on the span of the bias of an optimal policy). Moreover, this bound also allows the MDP to have some transient states that are not reachable under any policy. However, the bound holds only when the learner knows an upper bound on this parameter D 1. In case the learner has no such upper bound, a doubling trick can be applied, but then the bound's dependence on | S | deteriorates from | S | to | S | 3 / 2 . Bartlett and Tewari (2009) also modify our lower bound example to obtain a lower bound of Ω ( D 1 √ | S || A | T ) with respect to their new transition parameter D 1. Still, in the given example D 1 = √ D , so that in this case their lower bound matches our lower bound.

## 2. Results

We summarize the results achieved for our algorithm UCRL2 (which will be described in the next section), and also state a corresponding lower bound. We assume an unknown MDP M to be learned, with S : = | S | states, A : = | A | actions, and finite diameter D : = D ( M ) . Only S and A are known to the learner, and UCRL2 is run with confidence parameter δ .

Theorem 2 With probability of at least 1 -δ it holds that for any initial state s ∈ S and any T &gt; 1 , the regret of UCRL2 is bounded by

<!-- formula-not-decoded -->

It is straightforward to obtain from Theorem 2 the following sample complexity bound.

Corollary 3 With probability of at least 1 -δ the average per-step regret of UCRL2 is at most ε for any steps.

<!-- formula-not-decoded -->

It is also possible to give a sample complexity bound on the number of suboptimal steps UCRL2 takes, which allows to derive the following gap-dependent logarithmic bound on the expected regret.

Theorem 4 For any initial state s ∈ S , any T ≥ 1 and any ε &gt; 0 , with probability of at least 1 -3 δ the regret of UCRL2 is

<!-- formula-not-decoded -->

Moreover setting

<!-- formula-not-decoded -->

to be the gap in average reward between best and second best policy in M, the expected regret of UCRL2 (with parameter δ : = 1 3 T ) for any initial state s ∈ S is

<!-- formula-not-decoded -->

where T π is the smallest natural number such that for all T ≥ T π the expected average reward after T steps is g 2 -close to the average reward of π . Using the doubling trick to set the parameter δ , one obtains a corresponding bound (with larger constant) without knowledge of the horizon T.

These new bounds are improvements over the bounds that have been achieved by Auer and Ortner (2007) for the original UCRL algorithm in various respects: the exponents of the relevant parameters have been decreased considerably, the parameter D we use here is substantially smaller than the corresponding mixing time of Auer and Ortner (2007), and finally, the ergodicity assumption is replaced by the much weaker and more natural assumption that the MDP has finite diameter.

The following is an accompanying lower bound on the expected regret.

Theorem 5 For any algorithm A , any natural numbers S , A ≥ 10 , D ≥ 20log A S, and T ≥ DSA, there is an MDP M with S states, A actions, and diameter D, 5 such that for any initial state s ∈ S the expected regret of A after T steps is

<!-- formula-not-decoded -->

Finally, we consider a modification of UCRL2 that is also able to deal with changing MDPs.

Theorem 6 Assume that the MDP (i.e., its transition probabilities and reward distributions) is allowed to change ( /lscript -1 ) times up to step T, such that the diameter is always at most D. Restarting UCRL2 with parameter δ /lscript 2 at steps ⌈ i 3 /lscript 2 ⌉ for i = 1 , 2 , 3 . . . , the regret (now measured as the sum of missed rewards compared to the /lscript optimal policies in the periods during which the MDP remains constant) is upper bounded by with probability of at least 1 -δ .

<!-- formula-not-decoded -->

5. As already mentioned, the diameter of any MDP with S states and A actions is at least log A S -3.

## JAKSCH, ORTNER AND AUER

For the simpler multi-armed bandit problem, similar settings have already been considered by Auer et al. (2002b), and more recently by Garivier and Moulines (2008), and Yu and Mannor (2009). The achieved regret bounds are O ( √ /lscript T log T ) in the first two mentioned papers, while Yu and Mannor (2009) derive regret bounds of O ( /lscript log T ) for a setting with side observations on past rewards in which the number of changes /lscript need not be known in advance.

MDPs with a different model of changing rewards have already been considered by Even-Dar et al. (2005) and Even-Dar et al. (2009), respectively. There, the transition probabilities are assumed to be fixed and known to the learner, but the rewards are allowed to change at every step (however, independently of the history). In this setting, an upper bound of O ( √ T ) on the regret against an optimal stationary policy (with the reward changes known in advance) is best possible and has been derived by Even-Dar et al. (2005). This setting recently has been further investigated by Yu et al. (2009), who also show that for achieving sublinear regret it is essential that the changing rewards are chosen obliviously, as an opponent who chooses the rewards depending on the learner's history may inflict linear loss on the learner. It should be noted that although the definition of regret in the nonstochastic setting looks the same as in the stochastic setting, there is an important difference to notice. While in the stochastic setting the average reward of an MDP is always maximized by a stationary policy π : S → A , in the nonstochastic setting obviously a dynamic policy adapted to the reward sequence would in general earn more than a stationary policy. However, obviously no algorithm will be able to compete with the best dynamic policy for all possible reward sequences, so that-similar to the nonstochastic bandit problem, compare to Auer et al. (2002b)-one usually competes only with a finite set of experts, in the case of MDPs the set of stationary policies π : S → A . For different notions of regret in the nonstochastic MDP setting see Yu et al. (2009).

Note that all our results scale linearly with the rewards. That is, if the rewards are not bounded in [ 0 , 1 ] but taken from some interval [ r min , r max ] , the rewards can simply be normalized, so that the given regret bounds hold with additional factor ( r max -r min ) .

## 3. The UCRL2 Algorithm

Our algorithm is a variant of the UCRL algorithm of Auer and Ortner (2007). As its predecessor, UCRL2 implements the paradigm of 'optimism in the face of uncertainty'. That is, it defines a set M of statistically plausible MDPs given the observations so far, and chooses an optimistic MDP ˜ M (with respect to the achievable average reward) among these plausible MDPs. Then it executes a policy ˜ π which is (nearly) optimal for the optimistic MDP ˜ M . More precisely, UCRL2 (see Figure 1) proceeds in episodes and computes a new policy ˜ π k only at the beginning of each episode k . The lengths of the episodes are not fixed a priori, but depend on the observations made. In Steps 2-3, UCRL2 computes estimates ˆ rk ( s , a ) and ˆ pk ( s ′ | s , a ) for the mean rewards and the transition probabilities from the observations made before episode k . In Step 4, a set M k of plausible MDPs is defined in terms of confidence regions around the estimated mean rewards ˆ rk ( s , a ) and transition probabilities ˆ pk ( s ′ | s , a ) . This guarantees that with high probability the true MDP M is in M k . In Step 5, extended value iteration (see below) is used to choose a near-optimal policy ˜ π k on an optimistic MDP ˜ Mk ∈ M k . This policy ˜ π k is executed throughout episode k (Step 6). Episode k ends when a state s is visited in which the action a = ˜ π k ( s ) induced by the current policy has been chosen in episode k equally often as before episode k . Thus, the total number of occurrences of

any state-action pair is at most doubled during an episode. The counts vk ( s , a ) keep track of these occurrences in episode k . 6

## 3.1 Extended Value Iteration: Finding Optimistic Model and Optimal Policy

In Step 5 of the UCRL2 algorithm we need to find a near-optimal policy ˜ π k for an optimistic MDP ˜ Mk . While value iteration typically calculates an optimal policy for a fixed MDP, we also need to select an optimistic MDP ˜ Mk that gives almost maximal optimal average reward among all plausible MDPs.

## 3.1.1 PROBLEM FORMULATION

Wecan formulate this as a general problem as follows. Let M be the set of all MDPs with (common) state space S , (common) action space A , transition probabilities ˜ p ( ·| s , a ) , and mean rewards ˜ r ( s , a ) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for given probability distributions ˆ p ( ·| s , a ) , values ˆ r ( s , a ) in [ 0 , 1 ] , d ( s , a ) &gt; 0, and d ′ ( s , a ) ≥ 0. Further, we assume that M contains at least one communicating MDP, that is, an MDP with finite diameter.

In Step 5 of UCRL2, the d ( s , a ) and d ′ ( s , a ) are obviously the confidence intervals as given by (4) and (3), while the communicating MDP assumed to be in M k is the true MDP M . The task is to find an MDP ˜ M ∈ M and a policy ˜ π : S → A which maximize ρ ( ˜ M , ˜ π , s ) for all states s . 7 This task is similar to optimistic optimality in bounded parameter MDPs as considered by Tewari and Bartlett (2007). A minor difference is that in our case the transition probabilities are bounded not individually but by the 1-norm. More importantly, while Tewari and Bartlett (2007) give a converging algorithm for computing the optimal value function, they do not bound the error when terminating their algorithm after finitely many steps. In the following, we will extend standard undiscounted value iteration (Puterman, 1994) to solve the set task.

First, note that we may combine all MDPs in M to get a single MDP with extended action set A ′ . That is, we consider an MDP ˜ M + with continuous action space A ′ , where for each action a ∈ A , each admissible transition probability distribution ˜ p ( ·| s , a ) according to (1) and each admissible mean reward ˜ r ( s , a ) according to (2) there is an action in A ′ with transition probabilities ˜ p ( ·| s , a ) and mean reward ˜ r ( s , a ) . 8 Then for each policy ˜ π + on ˜ M + there is an MDP ˜ M ∈ M and a policy ˜ π : S → A on ˜ M such that the policies ˜ π + and ˜ π induce the same transition probabilities and mean rewards on the respective MDP. (The other transition probabilities in ˜ M can be set to ˆ p ( ·| s , a ) .) On the other hand, for any given MDP ˜ M ∈ M and any policy ˜ π : S → A there is a policy ˜ π + on ˜ M + so that again the same transition probabilities and rewards are induced by ˜ π on ˜ M and ˜ π + on ˜ M + . Thus, finding an MDP ˜ M ∈ M and a policy ˜ π on ˜ M such that ρ ( ˜ M , ˜ π , s ) = max M ′ ∈ M , π , s ′ ρ ( M ′ , π , s ′ ) for all initial states s , corresponds to finding an average reward optimal policy on ˜ M + .

/negationslash

7. Note that, as we assume that M contains a communicating MDP, if an average reward of ρ is achievable in one state, it is achievable in all states.

6. Since the policy ˜ π k is fixed for episode k , vk ( s , a ) = 0 only for a = ˜ π k ( s ) . Nevertheless, we find it convenient to use a notation which explicitly includes the action a in vk ( s , a ) .

8. Note that in ˜ M + the set of available actions now depends on the state.

Input:

A confidence parameter δ ∈ ( 0 , 1 ) , S and A .

Initialization: Set t : = 1, and observe the initial state s

1.

For episodes k = 1 , 2 , . . . do

## Initialize episode k :

1. Set the start time of episode k , t k : = t .
2. For all ( s , a ) in S × A initialize the state-action counts for episode k , vk ( s , a ) : = 0. Further, set the state-action counts prior to episode k ,

<!-- formula-not-decoded -->

3. For s , s ′ ∈ S and a ∈ A set the observed accumulated rewards and the transition counts prior to episode k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Compute estimates ˆ rk ( s , a ) : = Rk ( s , a ) max { 1 , Nk ( s , a ) } , ˆ pk ( s ′ | s , a ) : = Pk ( s , a , s ′ ) max { 1 , Nk ( s , a ) } .

## Compute policy ˜ π k :

4. Let M k be the set of all MDPs with states and actions as in M , and with transition probabilities ˜ p ( ·| s , a ) close to ˆ pk ( ·| s , a ) , and rewards ˜ r ( s , a ) ∈ [ 0 , 1 ] close to ˆ rk ( s , a ) , that is,

<!-- formula-not-decoded -->

- ∥ ∥ 5. Use extended value iteration (see Section 3.1) to find a policy ˜ π k and an optimistic MDP ˜ Mk ∈ M k such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Execute policy ˜ π k :

6. While vk ( st , ˜ π k ( st )) &lt; max { 1 , Nk ( st , ˜ π k ( st )) } do
2. (a) Choose action at = ˜ π k ( st ) , obtain reward rt , and observe next state st + 1.
3. (b) Update vk ( st , at ) : = vk ( st , at ) + 1.
4. (c) Set t : = t + 1.

Figure 1: The UCRL2 algorithm.

Input: Estimates ˆ p ( ·| s , a ) and distance d ( s , a ) for a state-action pair ( s , a ) , and the states in S sorted descendingly according to their ui value.

That is, let S : = { s ′ 1 , s ′ 2 , . . . , s ′ n } with ui ( s ′ 1 ) ≥ ui ( s ′ 2 ) ≥ . . . ≥ ui ( s ′ n ) .

1. Set

<!-- formula-not-decoded -->

2. Set /lscript : = n .
3. While ∑ s ′ j ∈ S p ( s ′ j ) &gt; 1 do
3. (b) Set /lscript : = /lscript -1.
4. (a) Reset p ( s ′ /lscript ) : = max { 0 , 1 -∑ s ′ j = s ′ /lscript p ( s ′ j ) } .

/negationslash

Figure 2: Computing the inner maximum in the extended value iteration (5).

## 3.1.2 EXTENDED VALUE ITERATION

Wedenote the state values of the i -th iteration by ui ( s ) . Then we get for undiscounted value iteration (Puterman, 1994) on ˜ M + for all s ∈ S :

<!-- formula-not-decoded -->

where ˜ r ( s , a ) : = ˆ r ( s , a ) + d ′ ( s , a ) are the maximal possible rewards according to condition (2), and P ( s , a ) is the set of transition probabilities ˜ p ·| s , a satisfying condition (1).

( ) While (5) is a step of value iteration with an infinite action space, max p p · u i is actually a linear optimization problem over the convex polytope P ( s , a ) . This implies that (5) can be evaluated considering only the finite number of vertices of this polytope.

/negationslash

Indeed, for a given state-action pair the inner maximum of (5) can be computed in O ( S ) computation steps by an algorithm introduced by Strehl and Littman (2008). For the sake of completeness we display the algorithm in Figure 2. The idea is to put as much transition probability as possible to the state with maximal value ui ( s ) at the expense of transition probabilities to states with small values ui ( s ) . That is, one starts with the estimates ˆ p ( s ′ j | s , a ) for p ( s ′ j ) except for the state s ′ 1 with maximal ui ( s ) , for which we set p ( s ′ 1 ) : = ˆ p ( s ′ 1 | s , a ) + 1 2 d ( s , a ) . In order to make p correspond to a probability distribution again, the transition probabilities from s to states with small ui ( s ) are reduced in total by 1 2 d ( s , a ) , so that ‖ p -ˆ p ( ·| s , a ) ‖ 1 = d ( s , a ) . This is done iteratively. Updating ∑ s ′ j ∈ S p ( s ′ j ) with every change of p for the computation of ∑ s ′ j = s ′ /lscript p ( s ′ j ) , this iterative procedure takes O ( S ) steps. Thus, sorting the states according to their value ui ( s ) at each iteration i once, u i + 1 can be computed from u i in at most O ( S 2 A ) steps.

<!-- formula-not-decoded -->

## 3.1.3 CONVERGENCE OF EXTENDED VALUE ITERATION

We have seen that value iteration on the MDP ˜ M + with continuous action is equivalent to value iteration on an MDP with finite action set. Thus, in order to guarantee convergence, it is sufficient to assure that extended value iteration never chooses a policy with periodic transition matrix. (Intuitively, it is clear that optimal policies with periodic transition matrix do not matter as long as it is guaranteed that such a policy is not chosen by value iteration, compare to Sections 8.5, 9.4, and 9.5.3. of Puterman 1994. For a proof see Appendix B.) Indeed, extended value iteration always chooses a policy with aperiodic transition matrix: In each iteration there is a single fixed state s ′ 1 which is regarded as the 'best' target state. For each state s , in the inner maximum an action with positive transition probability to s ′ 1 will be chosen. In particular, the policy chosen by extended value iteration will have positive transition probability from s ′ 1 to s ′ 1 . Hence, this policy is aperiodic and has state independent average reward. Thus we obtain the following result.

Theorem 7 Let M be the set of all MDPs with state space S , action space A , transition probabilities ˜ p ( ·| s , a ) , and mean rewards ˜ r ( s , a ) that satisfy (1) and (2) for given probability distributions ˆ p ( ·| s , a ) , values ˆ r ( s , a ) in [ 0 , 1 ] , d ( s , a ) &gt; 0 , and d ′ ( s , a ) ≥ 0 . If M contains at least one communicating MDP, extended value iteration converges. Further, stopping extended value iteration when

<!-- formula-not-decoded -->

the greedy policy with respect to u i is ε -optimal.

Remark 8 When value iteration converges, a suitable transformation of u i converges to the bias vector of an optimal policy. Recall that for a policy π the bias λ ( s ) in state s is basically the expected advantage in total reward (for T → ∞ ) of starting in state s over starting in the stationary distribution (the long term probability of being in a state) of π . For a fixed policy π , the Poisson equation

<!-- formula-not-decoded -->

relates the bias vector λ to the average reward ρ , the mean reward vector r , and the transition matrix P . Now when value iteration converges, the vector u i -min s ui ( s ) 1 converges to λ -min s λ ( s ) 1 . As we will see in inequality (11) below, the so-called span max s ui ( s ) -min s ui ( s ) of the vector u i is upper bounded by the diameter D, so that this also holds for the span of the bias vector λ of the optimal policy found by extended value iteration, that is, max s λ ( s ) -min s λ ( s ) ≤ D. Indeed, one can show that this holds for any optimal policy (cf. also Section 4 of Bartlett and Tewari, 2009).

Remark 9 We would like to note that the algorithm of Figure 2 can easily be adapted to the alternative setting of Tewari and Bartlett (2007), where each single transition probability p ( s ′ | s , a ) is bounded as 0 ≤ b -( s ′ , s , a ) ≤ p ( s ′ | s , a ) ≤ b + ( s ′ , s , a ) ≤ 1 . However, concerning convergence one needs to make some assumptions to exclude the possibility of choosing optimal policies with periodic transition matrices. For example, one may assume (apart from other assumptions already made by Tewari and Bartlett 2007) that for all s ′ , s , a there is an admissible probability distribution p ( ·| s , a ) with p ( s ′ | s , a ) &gt; 0 . Note that for Theorem 7 to hold, it is similarly essential that d ( s , a ) &gt; 0 . Alternatively, one may apply an aperiodicity transformation as described in Section 8.5.4 of Puterman (1994).

Now returning to Step 5 of UCRL2, we stop value iteration when

<!-- formula-not-decoded -->

which guarantees by Theorem 7 that the greedy policy with respect to u i is 1 √ t k -optimal.

## 4. Analysis of UCRL2 (Proofs of Theorem 2 and Corollary 3)

We start with a rough outline of the proof of Theorem 2. First, in Section 4.1, we deal with the random fluctuation of the rewards. Further, the regret is expressed as the sum of the regret accumulated in the individual episodes. That is, we set the regret in episode k to be

<!-- formula-not-decoded -->

where vk ( s , a ) now denotes the final counts of state-action pair ( s , a ) in episode k . Then it is shown that the total regret can be bounded by with high probability.

<!-- formula-not-decoded -->

In Section 4.2, we consider the regret that is caused by failing confidence regions. We show that this term can be upper bounded by √ T with high probability. After this intermezzo, the regret of episodes for which the true MDP M ∈ M k is examined in Section 4.3. Analyzing the extended value iteration scheme in Section 4.3.1 and using vector notation, we show that

<!-- formula-not-decoded -->

where ˜ P k is the assumed transition matrix (in ˜ Mk ) of the applied policy in episode k , v k are the visit counts at the end of that episode, and w k is a vector with ‖ w k ‖ ∞ ≤ D ( M ) 2 . The last two terms in the above expression stem from the reward confidence intervals (3) and the approximation error of value iteration. These are bounded in Section 4.3.3 when summing over all episodes. The first term on the right hand side is analyzed further in Section 4.3.2 and split into

<!-- formula-not-decoded -->

∥ ∥ where P k is the true transition matrix (in M ) of the policy applied in episode k . Substituting for ˜ P k -P k the lengths of the confidence intervals as given in (4), the remaining term that needs analysis is v k ( P k -I ) w k . For the sum of this term over all episodes we obtain in Section 4.3.2 a high probability bound of

<!-- formula-not-decoded -->

where m is the number of episodes-a term shown to be logarithmic in T in Appendix C.2. Section 4.3.3 concludes the analysis of episodes with M ∈ M k by summing the individual regret terms over all episodes k with M ∈ M k . In the final Section 4.4 we finish the proof by combining the results of Sections 4.1-4.3.

## 4.1 Splitting into Episodes

Recall that rt is the (random) reward UCRL2 receives at step t when starting in some initial state s 1. For given state-action counts N ( s , a ) after T steps, the rt are independent random variables, so that by Hoeffding's inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ Thus we get for the regret of UCRL2 (now omitting explicit references to M and UCRL2)

with probability at least 1 -δ 12 T 5 / 4 . Denoting the number of episodes started up to step T by m , we have ∑ m k = 1 vk ( s , a ) = N ( s , a ) and ∑ s , a N ( s , a ) = T . Therefore, writing ∆ k : = ∑ s , a vk ( s , a ) ( ρ ∗ -¯ r ( s , a ) ) , it follows that with probability at least 1 -δ 12 T 5 / 4 .

<!-- formula-not-decoded -->

## 4.2 Dealing with Failing Confidence Regions

Let us now consider the regret of episodes in which the set of plausible MDPs M k does not contain the true MDP M , ∑ m k = 1 ∆ k 1 M /negationslash∈ M k . By the stopping criterion for episode k we have (except for episodes where vk ( s , a ) = 1 and Nk ( s , a ) = 0, when ∑ s , a vk ( s , a ) = 1 ≤ t k holds trivially)

<!-- formula-not-decoded -->

Hence, denoting M ( t ) to be the set of plausible MDPs as given by (3) and (4) using the estimates available at step t , we have due to ρ ∗ ≤ 1 that

Now, P { M /negationslash∈ M ( t ) } ≤ δ 15 t 6 (see Appendix C.1), and since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have P {∃ t : T 1 / 4 &lt; t ≤ T : M /negationslash∈ M ( t ) } ≤ δ 12 T 5 / 4 . It follows that with probability at least 1 -δ 12 T 5 / 4 ,

## 4.3 Episodes with M ∈ M k

Nowweassume that M ∈ M k and start by considering the regret in a single episode k . The optimistic average reward ˜ ρ k of the optimistically chosen policy ˜ π k is essentially larger than the true optimal average reward ρ ∗ , and thus it is sufficient to calculate by how much the optimistic average reward ˜ ρ k overestimates the actual rewards of policy ˜ π k . By the assumption M ∈ M k , the choice of ˜ π k and ˜ Mk in Step 5 of UCRL2, and Theorem 7 we get that ˜ ρ k ≥ ρ ∗ -1 √ t k . Thus for the regret ∆ k accumulated in episode k we obtain

<!-- formula-not-decoded -->

## 4.3.1 EXTENDED VALUE ITERATION REVISITED

To proceed, we reconsider the extended value iteration of Section 3.1. As an important observation for our analysis, we find that for any iteration i the range of the state values is bounded by the diameter of the MDP M , that is,

<!-- formula-not-decoded -->

To see this, observe that ui ( s ) is the total expected i -step reward of an optimal non-stationary i -step policy starting in state s on the MDP ˜ M + with extended action set (as considered for extended value iteration). The diameter of this extended MDP is at most D as it contains by assumption the actions of the true MDP M . Now, if there were states s ′ , s ′′ with ui ( s ′′ ) -ui ( s ′ ) &gt; D , then an improved value for ui ( s ′ ) could be achieved by the following nonstationary policy: First follow a policy which moves from s ′ to s ′′ most quickly, which takes at most D steps on average. Then follow the optimal i -step policy for s ′′ . Since only D of the i rewards of the policy for s ′′ are missed, this policy gives ui ( s ′ ) ≥ ui ( s ′′ ) -D , contradicting our assumption and thus proving (11).

It is a direct consequence of Theorem 8.5.6. of Puterman (1994), that when the convergence criterion (6) holds at iteration i , then

<!-- formula-not-decoded -->

for all s ∈ S , where ˜ ρ k is the average reward of the policy ˜ π k chosen in this iteration on the optimistic MDP ˜ Mk . 9 Expanding ui + 1 ( s ) according to (5), we get and hence by (12)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

9. This is quite intuitive. We expect to receive average reward ˜ ρ k per step, such that the difference of the state values after i + 1 and i steps should be about ˜ ρ k .

∣ ∣ Setting r k : = ( ˜ rk ( s , ˜ π k ( s ) )) s to be the (column) vector of rewards for policy ˜ π k , ˜ P k : = ( ˜ pk ( s ′ | s , ˜ π k ( s )) ) s , s ′ the transition matrix of ˜ π k on ˜ Mk , and v k : = ( vk ( s , ˜ π k ( s ) )) s the (row)

vector of visit counts for each state and the corresponding action chosen by ˜ π k , we can use (13)recalling that vk ( s , a ) = 0 for a = ˜ π k ( s ) -to rewrite (10) as

/negationslash

<!-- formula-not-decoded -->

Since the rows of ˜ P k sum to 1, we can replace u i by w k where we set

<!-- formula-not-decoded -->

such that it follows from (11) that ‖ w k ‖ ≤ D 2 . Further, since we assume M ∈ M k , ˜ rk ( s , a ) -¯ r ( s , a ) ≤ | ˜ rk ( s , a ) -ˆ rk ( s , a ) | + | ¯ r ( s , a ) -ˆ rk ( s , a ) | is bounded according to (3), so that

<!-- formula-not-decoded -->

Noting that max { 1 , Nk ( s , a ) } ≤ t k ≤ T we get from (14) that

<!-- formula-not-decoded -->

## 4.3.2 THE TRUE TRANSITION MATRIX

Now we want to replace the transition matrix ˜ P k of the policy ˜ π k in the optimistic MDP ˜ Mk by the transition matrix P k : = ( p ( s ′ | s , ˜ π k ( s )) ) s , s ′ of ˜ π k in the true MDP M . Thus, we write

The first term. Since by assumption ˜ Mk and M are in the set of plausible MDPs M k , the first term in (16) can be bounded using condition (4). Thus, also using that ‖ w k ‖ ∞ ≤ D 2 we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This term will turn out to be the dominating contribution in our regret bound.

The second term. The intuition about the second term in (16) is that the counts of the state visits v k are relatively close to the stationary distribution µ k of the transition matrix P k , for which µ k P k = µ k , such that v k ( P k -I ) should be small. For the proof we define a suitable martingale and make use of the Azuma-Hoeffding inequality.

Lemma 10 (Azuma-Hoeffding inequality, Hoeffding 1963) Let X 1 , X 2 , . . . be a martingale difference sequence with | Xi | ≤ c for all i. Then for all ε &gt; 0 and n ∈ N ,

<!-- formula-not-decoded -->

Denote the unit vectors with i -th coordinate 1 and all other coordinates 0 by e i . Let s 1 , a 1 , s 2 , . . . , aT , sT + 1 be the sequence of states and actions, and let k ( t ) be the episode which contains step t . Consider the sequence Xt : = ( p ( ·| st , at ) -e st + 1 ) w k ( t ) 1 M ∈ M k ( t ) for t = 1 , . . . , T . Then for any episode k with M ∈ M k , we have due to ‖ w k ‖ ∞ ≤ D 2 that

<!-- formula-not-decoded -->

Also due to ‖ w k ‖ ∞ ≤ D 2 , we have | Xt | ≤ ( ‖ p ( ·| st , at ) ‖ 1 + ‖ e st + 1 ‖ 1 ) D 2 ≤ D . Further, E [ Xt ∣ ∣ s 1 , a 1 , . . . , st , at ] = 0, so that Xt is a sequence of martingale differences, and application of Lemma 10 gives

<!-- formula-not-decoded -->

Since for the number of episodes we have m ≤ SA log 2 ( 8 T SA ) as shown in Appendix C.2, summing over all episodes yields

<!-- formula-not-decoded -->

with probability at least 1 -δ 12 T 5 / 4 .

## 4.3.3 SUMMING OVER EPISODES WITH M ∈ M k

To conclude Section 4.3, we sum (15) over all episodes with M ∈ M k , using (16), (17), and (18), which yields that with probability at least 1 -δ 12 T 5 / 4

<!-- formula-not-decoded -->

Recall that N ( s , a ) : = ∑ k vk ( s , a ) such that ∑ s , a N ( s , a ) = T and Nk ( s , a ) = ∑ i &lt; k vi ( s , a ) . By the criterion for episode termination in Step 6 of the algorithm, we have that vk ( s , a ) ≤ Nk ( s , a ) . Using that for Zk = max { 1 , ∑ k i = 1 zi } and 0 ≤ zk ≤ Zk -1 it holds that (see Appendix C.3)

we get

<!-- formula-not-decoded -->

By Jensen's inequality we thus have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we get from (19) after some minor simplifications that with probability at least 1 -δ 12 T 5 / 4

<!-- formula-not-decoded -->

## 4.4 Completing the Proof of Theorem 2

Finally, evaluating (8) by summing ∆ k over all episodes, we get by (9) and (21)

<!-- formula-not-decoded -->

with probability at least 1 -δ 12 T 5 / 4 -δ 12 T 5 / 4 -δ 12 T 5 / 4 . Further simplifications (given in Appendix C.4) yield that for any T &gt; 1 with probability at least 1 -δ 4 T 5 / 4

<!-- formula-not-decoded -->

Since ∑ ∞ T = 2 δ 4 T 5 / 4 &lt; δ the statement of Theorem 2 follows by a union bound over all possible values of T .

## 4.5 Proof of Corollary 3

In order to obtain the PAC bound of Corollary 3 we simply have to find a sufficiently large T 0 such that for all T ≥ T 0 the per-step regret is smaller than ε . By Theorem 2 this means that for all T ≥ T 0 we shall have

<!-- formula-not-decoded -->

( δ ) ε 0)

Setting T 0 : = 2 α log α for α : = 34 2 D 2 S 2 A 2 we have due to x &gt; 2log x (for x &gt;

<!-- formula-not-decoded -->

so that (24) as well as the corollary follow.

## 5. The Logarithmic Bound (Proof of Theorem 4)

To show the logarithmic upper bound on the expected regret, we start with a bound on the number of steps in suboptimal episodes (in the spirit of sample complexity bounds as given by Kakade, 2003). We say that an episode k is ε -bad if its average regret is more than ε , where the average regret of an episode of length /lscript k is ∆ k /lscript k with 10 ∆ k = ∑ t k + 1 -1 t = t k ( ρ ∗ -rt ) . The following result gives an upper bound on the number of steps taken in ε -bad episodes.

Theorem 11 Let L ε ( T ) be the number of steps taken by UCRL2 in ε -bad episodes up to step T. Then for any initial state s ∈ S , any T &gt; 1 and any ε &gt; 0 , with probability of at least 1 -3 δ

<!-- formula-not-decoded -->

Proof The proof is an adaptation of the proof of Theorem 2 which gives an upper bound of O ( DS √ L ε A log ( AT / δ ) ) on the regret ∆ ′ ε ( s , T ) in ε -bad episodes in terms of L ε . The theorem then follows due to ε L ε ≤ ∆ ′ ε ( s , T ) .

Fix some T &gt; 1, and let K ε and J ε be two random sets that contain the indices of the ε -bad episodes up to step T and the corresponding time steps taken in these episodes, respectively. Then by an application of Hoeffding's inequality similar to (7) in Section 4.1 and a union bound over all possible values of L ε , one obtains that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

10. In the following we use the same notation as in the proof of Theorem 2.

Further, by summing up all error probabilities P { M /negationslash∈ M ( t ) } ≤ δ 15 t 6 for t = 1 , 2 , . . . one has

<!-- formula-not-decoded -->

It follows that with probability at least 1 -2 δ

<!-- formula-not-decoded -->

In order to bound the regret of a single episode with M ∈ M k we follow the lines of the proof of Theorem 2 in Section 4.3. Combining (15), (16), and (17) we have that

<!-- formula-not-decoded -->

In Appendix D we prove an analogon of (20), that is,

<!-- formula-not-decoded -->

Then from (25), (26), and (27) it follows that with probability at least 1 -2 δ

<!-- formula-not-decoded -->

For the regret term of ∑ k ∈ K ε v k ( P k -I ) w k 1 M ∈ M k we use an argument similar to the one applied to obtain (18) in Section 4.3.2. Here we have to consider a slightly modified martingale difference sequence for t = 1 , . . . , T to get (using the bound on the number of episodes given in Appendix C.2)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we set T ( L ) : = min { t : # { τ ≤ t , τ ∈ J ε } = L } . The application of the Azuma-Hoeffding inequality in Section 4.3.2 is replaced with the following consequence of Bernstein's inequality for martingales.

Lemma 12 (Freedman 1975) Let X 1 , X 2 , . . . be a martingale difference sequence. Then

<!-- formula-not-decoded -->

Application of Lemma 12 with κ = 2 D √ L log ( T / δ ) and γ = D 2 L yields that for L ≥ log ( T / δ ) D 2 it holds that

<!-- formula-not-decoded -->

On the other hand, if L &lt; log ( T / δ ) D 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, (30) and (31) give by a union bound over all possible values of L ε that with probability at least 1 -δ

Together with (29) this yields that with probability at least 1 -δ

<!-- formula-not-decoded -->

Thus by (28) we obtain that with probability at least 1 -3 δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This can be simplified to by similar arguments as given in Appendix C.4. Since ε L ε ≤ ∆ ′ ε ( s , T ) , we get

<!-- formula-not-decoded -->

which proves the theorem.

Now we apply Theorem 11 to obtain the claimed logarithmic upper bound on the expected regret.

Proof of Theorem 4 Upper bounding L ε in (32) by (33), we obtain for the regret ∆ ′ ε ( s , T ) accumulated in ε -bad episodes that with probability at least 1 -3 δ . Noting that the regret accumulated outside of ε -bad episodes is at most ε T implies the first statement of the theorem.

<!-- formula-not-decoded -->

For the bound on the expected regret, first note that the expected regret of each episode in which an optimal policy is executed is at most D , whereas due to Theorem 11 the expected regret in g 2 -bad

episodes is upper bounded by 34 2 · 2 · D 2 S 2 A log ( T ) g + 1, as δ = 1 3 T . What remains to do is to consider episodes k with expected average regret smaller than g 2 in which however a non-optimal policy ˜ π k was chosen.

First, note that for each policy π there is a T π such that for all T ≥ T π the expected average reward after T steps is g 2 -close to the average reward of π . Thus, when a policy π is played in an episode of length ≥ T π either the episode is g 2 -bad (in expectation) or the policy π is optimal. Now we fix a state-action pair ( s , a ) and consider the episodes k in which the number of visits vk ( s , a ) in ( s , a ) is doubled. The corresponding episode lengths /lscript k ( s , a ) are not necessarily increasing, but the vk ( s , a ) are monotonically increasing, and obviously /lscript k ( s , a ) ≥ vk ( s , a ) . Since the vk ( s , a ) are at least doubled, it takes at most /ceilingleft 1 + log 2 ( max π : π ( s )= a T π ) /ceilingright episodes until /lscript k ( s , a ) ≥ vk ( s , a ) ≥ max π : π ( s )= a T π , when any policy π with π ( s ) = a applied in episode k that is not g 2 -bad (in expectation) will be optimal. Consequently, as only episodes of length smaller than max π : π ( s )= a T π have to be considered, the regret of episodes k where vk ( s , a ) &lt; max π : π ( s )= a T π is upper bounded by /ceilingleft 1 + log 2 ( max π : π ( s )= a T π ) /ceilingright max π : π ( s )= a T π . Summing over all state-action pairs, we obtain an additional additive regret term of

<!-- formula-not-decoded -->

which concludes the proof of the theorem.

## 6. The Lower Bound (Proof of Theorem 5)

We first consider the two-state MDP depicted in Figure 3. That is, there are two states, the initial state s ◦ and another state s /barshort , and A ′ = ⌊ A -1 2 ⌋ actions. For each action a , let the deterministic rewards be r ( s ◦ , a ) = 0 and r ( s /barshort , a ) = 1. For all but a single 'good' action a ∗ let p ( s /barshort | s ◦ , a ) = δ : = 4 D , whereas p ( s /barshort | s ◦ , a ∗ ) = δ + ε for some 0 &lt; ε &lt; δ specified later in the proof. Further, let p ( s ◦| s /barshort , a ) = δ for all a . The diameter of this MDP is D ′ = 1 δ = D 4 . For the rest of the proof we assume that 11 δ ≤ 1 3 .

Figure 3: The MDP for the lower bound. The single action a ∗ with higher transition probability from state s ◦ to state s /barshort is shown as dashed line.

<!-- image -->

Consider k : = ⌊ S 2 ⌋ copies of this MDP where only one of the copies has such a 'good' action a ∗ . To complete the construction, we connect the k copies into a single MDP with diameter less than D ,

11. Otherwise we have D &lt; 12, so that due to the made assumptions A &gt; 2 S . In this case we employ a different construction: Using S -1 actions, we connect all states to get an MDP with diameter 1. With the remaining A -S + 1 actions we set up a bandit problem in each state as in the proof of the lower bound of Auer et al. (2002b) where only one state has a better action. This yields Ω ( √ SAT ) regret, which is sufficient, since D is bounded in this case.

Figure 4: The composite MDP for the lower bound. Copies of the MDP of Figure 3 are arranged in an A ′ -ary tree, where the s ◦ -states are connected.

<!-- image -->

using at most A -A ′ additional actions. This can be done by introducing A ′ + 1 additional actions per state with deterministic transitions which do not leave the s /barshort -states and connect the s ◦ -states of the k copies by inducing an A ′ -ary tree structure on the s ◦ -states (one action for going toward the root, A ′ actions going toward the leaves-see Figure 4 for a schematic representation of the composite MDP). The reward for each of those actions is zero in any state. The diameter of the resulting MDP is at most 2 ( D 4 + /ceilingleft log A ′ k /ceilingright ) , which is twice the time it takes to travel to or from the root for any state in the MDP. Thus we have constructed an MDP M with ≤ S states, ≤ A actions, and diameter ≤ D , for which we will show the claimed lower bound on the regret.

Actually, in the analysis we will consider the simpler MDP where all s ◦ -states are identified. We set this state to be the initial state. This MDP is equivalent to a single MDP M ′ like the one in Figure 3 with kA ′ actions which we assume in the following to be taken from { 1 , . . . , kA ′ } . Note that learning this MDP is easier (as the learner is allowed to switch between different s ◦ -states without any cost for transition), while its optimal average reward is the same.

Weprove the theorem by applying the same techniques as in the proof of the lower bound for the multi-armed bandit problem of Auer et al. (2002b). The pair ( s ∗ ◦ , a ∗ ) identifying the copy with the better action and the better action is considered to be chosen uniformly at random from { 1 , . . . , k }× { 1 , . . . , A ′ } , and we denote the expectation with respect to the random choice of ( s ∗ ◦ , a ∗ ) as E ∗ [ · ] . We show that ε can be chosen such that M ′ and consequently also the composite MDP M forces regret E ∗ [ ∆ ( M , A , s ◦ , T )] ≥ E ∗ [ ∆ ( M ′ , A , s ∗ ◦ , T )] &gt; 0 . 015 √ D ′ kA ′ T on any algorithm A .

We write E unif [ · ] for the expectation when there is no special action (i.e., the transition probability from s ◦ to s /barshort is δ for all actions), and E a [ · ] for the expectation conditioned on a being the special action a ∗ in M ′ . As already argued by Auer et al. (2002b), it is sufficient to consider deterministic strategies for choosing actions. Indeed, any randomized strategy is equivalent to an (apriori) random choice from the set of all deterministic strategies. Thus, we may assume that any algorithm A maps the sequence of observations up to step t to an action at .

Now we follow the lines of the proof of Theorem A.2 as given by Auer et al. (2002b). Let the random variables N /barshort , N ◦ and N ∗ ◦ denote the total number of visits to state s /barshort , the total number of visits to state s ◦ , and the number of times action a ∗ is chosen in state s ◦ , respectively. Further, write st as

usual for the state observed at step t . Then since s ◦ is assumed to be the initial state, we have

/negationslash

<!-- formula-not-decoded -->

Taking into account that choosing a ∗ instead of any other action in s ◦ reduces the probability of staying in state s ◦ , it follows that (using D ′ = 1 δ )

<!-- formula-not-decoded -->

Now denoting the step where the first transition from s ◦ to s /barshort occurs by τ ◦ /barshort , we may lower bound E unif [ N /barshort ] by the law of total expectation as

<!-- formula-not-decoded -->

Therefore, combining (34) and (35) we obtain

<!-- formula-not-decoded -->

As A chooses its actions deterministically based on the observations so far, N ∗ ◦ is a function of the observations up to step T , too. A slight difference to Auer et al. (2002b) is that in our setting the sequence of observations consists not just of the rewards but also of the next state, that is, upon playing action at the algorithm observes st + 1 and rt . However, since the immediate reward is fully determined by the current state, N ∗ ◦ is also a function of just the state sequence, and we may bound E a [ N ∗ ◦ ] by the following lemma, adapted from Auer et al. (2002b).

Lemma 13 Let f : { s ◦ , s /barshort } T + 1 → [ 0 , B ] be any function defined on state sequences s ∈ { s ◦ , s /barshort } T + 1 observed in the MDP M ′ . Then for any 0 ≤ δ ≤ 1 2 , any 0 ≤ ε ≤ 1 -2 δ , and any a ∈ { 1 , . . . , kA ′ } ,

<!-- formula-not-decoded -->

The proof of Lemma 13 is a straightforward modification of the respective proof given by Auer et al. (2002b). For details we refer to Appendix E.

Now let us assume that ε ≤ δ . (Our final choice of ε below will satisfy this requirement.) By our assumption of δ ≤ 1 3 this yields that ε ≤ δ ≤ 1 3 ≤ 1 -2 δ . Then, since N ∗ ◦ is a function of the state sequence with N ∗ ◦ ∈ [ 0 , T ] , we may apply Lemma 13 to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

An immediate consequence of (35) is that ∑ kA ′ a = 1 E unif [ N ∗ ◦ ] ≤ T 2 + D ′ 2 , which yields by Jensen's inequality that ∑ kA ′ a = 1 √ 2 E unif [ N ∗ ◦ ] ≤ √ kA ′ ( T + D ′ ) . Thus we have from (37)

Together with (36) this gives

<!-- formula-not-decoded -->

Calculating the stationary distribution, we find that the optimal average reward for the MDP M ′ is δ + ε 2 δ + ε . Hence, the expected regret with respect to the random choice of a ∗ is at least

<!-- formula-not-decoded -->

Since by assumption we have T ≥ DSA ≥ 16 D ′ kA ′ and thus D ′ ≤ T 16 kA ′ , it follows that

<!-- formula-not-decoded -->

Now we choose ε : = c √ kA ′ TD ′ , where c : = 1 5 . Then because of 1 δ = D ′ ≤ T 16 kA ′ it follows that ε ≤ c 1 4 D ′ = δ 20 (so that also ε ≤ δ as needed to get (37)), and further 1 4 δ + 2 ε ≥ 1 4 + 1 / 8 D ′ . Hence we obtain

<!-- formula-not-decoded -->

Finally, we note that

<!-- formula-not-decoded -->

and since by assumption S , A ≥ 10 so that kA ′ ≥ 20, it follows that which concludes the proof.

<!-- formula-not-decoded -->

## 7. Regret Bounds for Changing MDPs (Proof of Theorem 6)

Consider the learner operates in a setting where the MDP is allowed to change /lscript times, such that the diameter never exceeds D (we assume an initial change at time t = 1). For this task we define the regret of an algorithm A up to step T with respect to the average reward ρ ∗ ( t ) of an optimal policy at step t as

<!-- formula-not-decoded -->

where rt is as usual the reward received by A at step t when starting in state s .

The intuition behind our approach is the following: When restarting UCRL2 every ( T /lscript ) 2 / 3 steps, the total regret for periods in which the MDP changes is at most /lscript 1 / 3 T 2 / 3 . For each other period we have regret of ˜ O ( ( T /lscript ) 1 / 3 ) by Theorem 2. Since UCRL2 is restarted only T 1 / 3 /lscript 2 / 3 times, the total regret is ˜ O /lscript 1 / 3 T 2 / 3 .

( ) Because the horizon T is usually unknown, we use an alternative approach for restarting which however exhibits similar properties: UCRL2 ′ restarts UCRL2 with parameter δ /lscript 2 at steps τ i = ⌈ i 3 /lscript 2 ⌉ for i = 1 , 2 , 3 , . . . . Now we prove Theorem 6, which states that the regret of UCRL2 ′ is bounded by

<!-- formula-not-decoded -->

Let n be the largest natural number such that ⌈ n 3 /lscript 2 ⌉ ≤ T , that is, n is the number of restarts up to step T . Then n 3 /lscript 2 ≤ τ n ≤ T ≤ τ n + 1 -1 &lt; ( n + 1 ) 3 /lscript 2 and consequently with probability at least 1 -δ in the considered setting.

<!-- formula-not-decoded -->

The regret ∆ c incurred due to changes of the MDP can be bounded by the number of steps taken in periods in which the MDP changes. This is maximized when the changes occur during the /lscript

longest periods, which contain at most τ n + 1 -1 -τ n -/lscript + 1 steps. Hence we have

<!-- formula-not-decoded -->

For /lscript ≥ 2 we get by (39) and (38) that

<!-- formula-not-decoded -->

while for /lscript = 1 we obtain also from (39) and (38) that

<!-- formula-not-decoded -->

Thus the contribution to the regret from changes of the MDP is at most

<!-- formula-not-decoded -->

On the other hand, if the MDP does not change between the steps τ i and min { T , τ i + 1 } , the regret ∆ ( s τ i , Ti ) for these Ti : = min { T , τ i + 1 }-τ i steps is bounded according to Theorem 2 (or more precisely (23)). Therefore, recalling that the confidence parameter is chosen to be δ /lscript 2 , this gives

<!-- formula-not-decoded -->

with probability 1 -δ 4 /lscript 2 T 5 / 4 i . As ∑ n i = 1 Ti = T , we have by Jensen's inequality ∑ n i = 1 √ Ti ≤ √ n √ T . Thus, summing over all i = 1 , . . . , n , the regret ∆ f in periods in which the MDP does not change is by (38)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -∑ n i = 1 δ 4 /lscript 2 T 5 / 4 i . We conclude the proof by bounding this latter probability. For ⌊ /lscript 2 3 ⌋ &lt; i &lt; n ,

and consequently 1 /lscript 2 T 5 / 4 i ≤ 1 i 2 . This together with Ti ≥ 1 then yields

<!-- formula-not-decoded -->

As ∆ ′ ( UCRL2 ′ , s , T ) ≤ ∆ c + ∆ f , combining (40) and (41) yields with probability at least 1 -δ , and Theorem 6 follows, since the claimed bound holds trivially for A log ( T δ ) &lt; log4.

<!-- formula-not-decoded -->

## 8. Open Problems

There is still a gap between the upper bound on the regret of Theorem 2 and the lower bound of Theorem 5. We conjecture that the lower bound gives the right exponents for the parameters S and D (concerning the dependence on S compare also the sample complexity bounds of Strehl et al., 2006). The recent research of Bartlett and Tewari (2009) also poses the question whether the diameter in our bounds can be replaced by a smaller parameter, that is, by the span of the bias of an optimal policy. As the algorithm REGAL.C of Bartlett and Tewari (2009) demonstrates, this is at least true when this value is known to the learner. However, in the case of ignorance, currently this replacement of the diameter D can only be achieved at the cost of an additional factor of √ S in the regret bounds (Bartlett and Tewari, 2009). The difficulty in the proof is that while the span of an optimal policy's bias vector in the assumed optimistic MDP can be upper bounded by the diameter of the true MDP (cf. Remark 8), it is not clear how the spans of optimal policies in the assumed and the true MDP relate to each other.

A somehow related question is that of transient states, that is, the possibility that some of the states are not reachable under any policy. In this case the diameter is unbounded, so that our bounds become vacuous. Indeed, our algorithm cannot handle transient states: for any time step and any transient state s , UCRL2 optimistically assumes maximal possible reward in s and a very small but still positive transition probability to s from any other state. Thus insisting on the possibility of a transition to s , the algorithm fails to detect an optimal policy. 12 The assumption of having an upper bound on an optimal policy's bias resolves this problem, as this bound indirectly also gives some information on what the learner may expect from a state that has not been reached so far and thus may be transient. Consequently, with the assumed knowledge of such an upper bound, the REGAL.C algorithm of Bartlett and Tewari (2009) is also able to deal with transient states.

12. Actually, one can modify UCRL2 to deal with transient states by assuming transition probability 0 for all transitions not observed so far. This is complemented by an additional exploration phase between episodes where, for example, the state-action pair with the fewest number of visits is probed. While this algorithm gives asymptotically the same bounds, these however contain a large additive constant for all the episodes that occur before the transition structure assumed by the algorithm is correct.

## Acknowledgments

We would like to thank the anonymous reviewers for their valuable comments. This work was supported in part by the Austrian Science Fund FWF (S9104-N13 SP4). The research leading to these results has also received funding from the European Community's Seventh Framework Programme (FP7/2007-2013) under grant agreements n ◦ 216886 (PASCAL2 Network of Excellence), and n ◦ 216529 (Personal Information Navigator Adapting Through Viewing, PinView), and the Austrian Federal Ministry of Science and Research. This publication only reflects the authors' views.

## Appendix A. A Lower Bound on the Diameter

We are going to show a more general result, from which the bound on the diameter follows. For a given MDP, let T ∗ ( s | s 0 ) be the minimal expected time it takes to move from state s 0 to state s .

Theorem 14 Consider an MDP with state space S and A states. Let d 0 be an arbitrary distribution over S , and U ⊆ S be any subset of states. Then the sum of the minimal expected transition times to states in U when starting in an initial state distributed according to d 0 is bounded as follows:

<!-- formula-not-decoded -->

We think this bound is tight. The minimum on the right hand side is attained when the nk are maximized for small k until | U | is exhausted. For A ≥ 2, this gives an average (over the states in U ) expected transition time of at least log A | U | -3 to states in U . Indeed, for | U | = ∑ m -1 k = 0 A k + nm we have A m + 1 -A ( A -1 ) 2 &lt; | U | ( 1 + 1 A -1 ) as well as m ≥ log A ( | U | 2 ) , so that

<!-- formula-not-decoded -->

In particular, choosing U = S gives the claimed lower bound on the diameter.

Corollary 15 In any MDP with S states and A ≥ 2 actions, the diameter D is lower bounded by log A S -3 .

Remark 16 For given S , A the minimal diameter is not always assumed by an MDP with deterministic transitions. Consider for example S = 4 and A = 2 . Any deterministic MDP with four states and two actions has diameter at least 2. However, Figure 5 shows a corresponding MDP whose diameter is 3 2 .

Figure 5: An MDP with four states and two actions whose diameter is 3 2 . In each state two actions are available. One action leads to another state deterministically, while the other action causes a random transition to each of the two other states with probability 1 2 (indicated as dashed lines).

<!-- image -->

Proof of Theorem 14 Let a ∗ ( s 0 , s ) be the optimal action in state s 0 for reaching state s , and let p ( s | s 0 , a ) be the transition probability to state s when choosing action a in state s 0.

For | U | &gt; 1 we have

The proof is by induction on the size of U . For | U | = 0 , 1 the statement holds.

<!-- formula-not-decoded -->

where U s 0 , a : = { s ∈ U \{ s 0 } : a ∗ ( s 0 , s ) = a } .

If all U s 0 , a ⊂ U , we apply the induction hypothesis and obtain for suitable nk ( s 0 , a )

<!-- formula-not-decoded -->

since ∑ k nk ( s 0 , a ) = | U s 0 , a | . Furthermore, nk ( s 0 , a ) ≤ A k and | U | -1 ≤ ∑ a | U s 0 , a | ≤ | U | . Thus setting n ′ k = ∑ s 0 d 0 ( s 0 ) ∑ a nk -1 ( s 0 , a ) for k ≥ 1 and n ′ 0 = | U | -∑ k ≥ 1 n ′ k satisfies the conditions of the statement. This completes the induction step for this case.

If U s 0 , a = U for some pair ( s 0 , a ) (i.e., for all target states s ∈ U the same action is optimal in s 0), then we construct a modified MDP with shorter transition times. This is achieved by modifying one of the actions to give a deterministic transition from s 0 to some state in U (which is not reached deterministically by choosing action a ). For the modified MDP the induction step works and the lower bound can be proven, which then also holds for the original MDP.

## Appendix B. Convergence of Value Iteration (Proof of Theorem 7)

As sufficient condition for convergence of value iteration, Puterman (1994) assumes only that all optimal policies have aperiodic transition matrices. Actually, the proof of Theorem 9.4.4 of Puterman (1994)-the main result on convergence of value iteration-needs this assumption only at one step, that is, to guarantee that the optimal policy identified at the end of the proof has aperiodic transition matrix. In the following we give a proof sketch of Theorem 7 that concentrates on the differences to the convergence proof given by Puterman (1994).

Lemma 9.4.3 of Puterman (1994) shows that value iteration eventually chooses only policies π that satisfy P π ρ ∗ = ρ ∗ , where P π is the transition matrix of π and ρ ∗ is the optimal average reward vector. More precisely, there is an i 0 such that for all i ≥ i 0

<!-- formula-not-decoded -->

where r π is the reward vector of the policy π , and E : = { π : S → A | P π ρ ∗ = ρ ∗ } .

Unlike standard value iteration, extended value iteration always chooses policies with aperiodic transition matrix (cf. the discussion in Section 3.1.3). Thus when considering only aperiodic policies F : = { π : S → A | P π is aperiodic } in the proof of Lemma 9.4.3, the same argument shows that there is an i ′ 0 such that for all i ≥ i ′ 0

<!-- formula-not-decoded -->

Intuitively, (42) shows that extended value iteration eventually chooses only policies from E ∩ F .

Then by the aperiodicity of P π ∗ , the result of Theorem 9.4.4 follows, and one obtains analogously to Theorem 9.4.5 (a) of Puterman (1994) that

With (42) accomplished, the proof of Theorem 9.4.4, the main result on convergence of value iteration, can be rewritten word by word from Puterman (1994), with E replaced with E ∩ F and using (42) instead of Lemma 9.4.3. Thus, unlike in the original proof where the optimal policy π ∗ identified at the end of the proof is in E , in our case π ∗ is in E ∩ F . Here Puterman (1994) uses the assumption that all optimal policies have aperiodic transition matrices to guarantee that π ∗ has aperiodic transition matrix. In our case, π ∗ has aperiodic transition matrix by definition, as it is in E ∩ F .

<!-- formula-not-decoded -->

As the underlying MDP ˜ M + is assumed to be communicating (so that ρ ∗ is state-independent), analogously to Corollary 9.4.6 of Puterman (1994) convergence of extended value iteration follows from (43). Finally, with the convergence of extended value iteration established, the error bound for the greedy policy follows from Theorem 8.5.6 of Puterman (1994).

## Appendix C. Technical Details for the Proof of Theorem 2

This appendix collects some technical details, starting with an error bound for our confidence intervals.

## C.1 Confidence Intervals

Lemma 17 For any t ≥ 1 , the probability that the true MDP M is not contained in the set of plausible MDPs M ( t ) at time t (as given by the confidence intervals in (3) and (4) ) is at most δ 15 t 6 , that is

Proof Consider a fixed state-action pair ( s , a ) and assume some given number of visits n &gt; 0 in ( s , a ) before step t . Denote the estimates for transition probabilities and rewards obtained from these n observations by ˆ p ( ·| s , a ) and ˆ r ( s , a ) , respectively. Let us first consider the probability with which a confidence interval for the transition probabilities fails. The random event observed for the transition probability estimates is the state to which the transition occurs. Generally, the L 1 -deviation of the true distribution and the empirical distribution over m distinct events from n samples is bounded according to Weissman et al. (2003) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ Thus, in our case we have m = S (for each possible transition there is a respective event), so that setting we get from (44)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the rewards we observe real-valued, independent identically distributed (i.i.d.) random variables with support in [ 0 , 1 ] . Hoeffding's inequality gives for the deviation between the true mean ¯ r and the empirical mean ˆ r from n i.i.d. samples with support in [ 0 , 1 ]

Setting

<!-- formula-not-decoded -->

we get for state-action pair ( s , a )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that when there haven't been any observations, the confidence intervals trivially hold with probability 1 (for transition probabilities as well as for rewards). Hence a union bound over all possible values of n = 1 , . . . , t -1 gives (now writing N ( s , a ) for the number of visits in ( s , a ) )

∥ ∥ Summing these error probabilities over all state-action pairs we obtain the claimed bound P { M / ∈ M ( t ) } &lt; δ 15 t 6 .

<!-- formula-not-decoded -->

## C.2 A Bound on the Number of Episodes

Since in each episode the total number of visits to at least one state-action pair doubles, the number of episodes m is logarithmic in T . Actually, the number of episodes becomes maximal when all state-action pairs are visited equally often, which results in the following bound.

Proposition 18 The number m of episodes of UCRL2 up to step T ≥ SA is upper bounded as m ≤ SA log 2 ( 8 T SA ) . Proof Let N ( s , a ) : = # { τ &lt; T + 1 : s τ = s , a τ = a } be the total number of observations of the stateaction pair ( s , a ) up to step T . In each episode k &lt; m there is a state-action pair ( s , a ) with vk ( s , a ) = Nk ( s , a ) (or vk ( s , a ) = 1, Nk ( s , a ) = 0). Let K ( s , a ) be the number of episodes with vk ( s , a ) = Nk ( s , a ) and Nk ( s , a ) &gt; 0. If N ( s , a ) &gt; 0, then vk ( s , a ) = Nk ( s , a ) implies Nk + 1 ( s , a ) = 2 Nk ( s , a ) , so that

<!-- formula-not-decoded -->

On the other hand, if N ( s , a ) = 0, then obviously K ( s , a ) = 0, so that generally, N ( s , a ) ≥ 2 K ( s , a ) -1 for any state-action pair ( s , a ) . It follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, in each episode a state-action pair ( s , a ) is visited for which either Nk ( s , a ) = 0 or Nk ( s , a ) = vk ( s , a ) . Hence, m ≤ 1 + SA + ∑ s , a K ( s , a ) , or equivalently ∑ s , a K ( s , a ) ≥ m -1 -SA . This implies

Together with (45) this gives which yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the claimed bound on m follows for T ≥ SA .

## C.3 The Sum in (19)

Lemma 19 For any sequence of numbers z 1 , . . . , zn with 0 ≤ zk ≤ Zk -1 : = max { 1 , ∑ k -1 i = 1 zi }

<!-- formula-not-decoded -->

Proof We prove the statement by induction over n .

Base case: We first show that the lemma holds for all n with ∑ n -1 k = 1 zk ≤ 1. Indeed, in this case Zk = 1 for k ≤ n -1 and hence zn ≤ 1. It follows that

<!-- formula-not-decoded -->

Note that this also shows that the lemma holds for n = 1, since ∑ 0 k = 1 zk = 0 ≤ 1. Inductive step: Now let us consider natural numbers n such that ∑ n -1 k = 1 zk &gt; 1. By the induction hypothesis we have

Since zn ≤ Zn -1 = ∑ n -1 k = 1 zk and Zn -1 + zn = Zn , we further have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves the lemma.

## C.4 Simplifying (22)

Combining similar terms, (22) yields that with probability at least 1 -δ 4 T 5 / 4

<!-- formula-not-decoded -->

We assume A ≥ 2, since the bound is trivial otherwise. Also, for 1 &lt; T ≤ 34 2 A log ( T δ ) we have ∆ ( s 1 , T ) ≤ 34 √ AT log ( T δ ) trivially. Considering T &gt; 34 A log ( T δ ) wehave A &lt; 1 34log ( T δ ) √ AT log ( T δ ) and also log 2 ( 8 T ) &lt; 2log ( T ) , so that

<!-- formula-not-decoded -->

Further, T &gt; 34 A log ( T δ ) also implies log ( 2 AT δ ) ≤ 2log ( T δ ) and log ( 8 T δ ) ≤ 2log ( T δ ) . Thus, we have by (46) that for any T &gt; 1 with probability at least 1 -δ 4 T 5 / 4

<!-- formula-not-decoded -->

## Appendix D. Technical Details for the Proof of Theorem 4: Proof of (27)

For a given index set K ε of episodes we would like to bound the sum

<!-- formula-not-decoded -->

We will do this by modifying the sum so that Lemma 19 becomes applicable. Compared to the setting of Lemma 19 there are some 'gaps' in the sum caused by episodes / ∈ K ε . In the following we show that the contribution of episodes that occur after step L ε : = ∑ k ∈ K ε ∑ s , a v k ( s , a ) is not larger than the missing contributions of the episodes / ∈ K ε . Intuitively speaking, one may fill the episodes that occur after step L ε into the gaps of episodes / ∈ K ε as Figure 6 suggests.

Figure 6: Illustration of the proof idea. Shaded boxes stand for episodes ∈ K ε , empty boxes for episodes / ∈ K ε . The contribution of episodes after step L ε can be 'filled into the gaps' of episodes / ∈ K ε before step L ε .

<!-- image -->

Let /lscript ε ( s , a ) : = ∑ k ∈ K ε v k ( s , a ) , so that ∑ s , a /lscript ε ( s , a ) = L ε . Weconsider a fixed state-action pair ( s , a ) and skip the reference to it for ease of reading, so that Nk refers to the number of visits to ( s , a ) up to episode k , and N denotes the total number of visits to ( s , a ) . Further, we abbreviate dk : = √ max { 1 , Nk } , and let m ε : = max { k : Nk &lt; /lscript ε } be the episode containing the /lscript ε -th visit to ( s , a ) . Due to vk = Nk + 1 -Nk we have

<!-- formula-not-decoded -->

Since Nm ε = ∑ m ε -1 k = 1 vk , this yields

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

By (47) and due to dk ≥ dm ε for k ≥ m ε we have

<!-- formula-not-decoded -->

Hence, we get together with (48), using that dk ≤ dm ε for k ≤ m ε

<!-- formula-not-decoded -->

Now define v ′ k as follows: let v ′ k : = vk for k &lt; m ε and v ′ m ε : = /lscript ε -Nm ε . Then we have just seen that

<!-- formula-not-decoded -->

Since further ∑ m ε k = 1 v ′ k = /lscript ε we get by Lemma 19 that

<!-- formula-not-decoded -->

As ∑ s , a /lscript ε ( s , a ) = L ε , we finally obtain by Jensen's inequality as claimed.

<!-- formula-not-decoded -->

## Appendix E. Proof of Lemma 13

Let us first recall some notation from Section 6. Thus P a [ · ] denotes the probability conditioned on a being the 'good' action, while the probability with respect to a setting where all actions in state s ◦ are equivalent (i.e., ε = 0) is denoted by P unif [ · ] . Let S : = { s ◦ , s /barshort } and denote the state

observed at step τ by s τ and the state-sequence up to step τ by s τ = s 1 , . . . , s τ . Basically, the proof follows along the lines of the proof of Lemma A.1 of Auer et al. (2002b). The first difference is that our observations now consist of the sequence of T + 1 states instead of a sequence of T observed rewards. Still it is straightforward to get analogously to the proof of Auer et al. (2002b), borrowing the notation, that for any function f from { s ◦ , s /barshort } T + 1 to [ 0 , B ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ where KL ( P ‖ Q ) denotes for two distributions P , Q the Kullback-Leibler divergence defined as KL ( P ‖ Q ) : = ∑ s ∈ S T + 1 P { s } log 2 ( P { s } Q { s } ) . It holds that (cf. Auer et al., 2002b)

where KL ( P { st + 1 | s t }‖ Q { st + 1 | s t } ) : = ∑ s t + 1 ∈ S t + 1 P { s t + 1 } log 2 ( P { st + 1 | s t } Q { st + 1 | s t } ) . By the Markov property and the fact that the action at is determined by a sequence s t ∈ S t we have (similar to Auer et al., 2002b)

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

Since log 2 ( P unif [ s ′ | s ′′ , a ′ ] P a [ s ′ | s ′′ , a ′ ] ) = 0 only for s ′′ = s ◦ and a ′ being the special action a , we get

To complete the proof we use the following lemma.

Lemma 20 For any 0 ≤ δ ≤ 1 2 and ε ≤ 1 -2 δ we have

<!-- formula-not-decoded -->

Indeed, application of Lemma 20 together with (50) and (51) gives that

<!-- formula-not-decoded -->

which together with (49) yields

<!-- formula-not-decoded -->

as claimed by Lemma 13.

## Proof of Lemma 20 Consider

<!-- formula-not-decoded -->

We show that h δ ( ε ) ≥ 0 for δ ≤ 1 2 and 0 ≤ ε ≤ ε 0, where

<!-- formula-not-decoded -->

Indeed, h δ ( 0 ) = 0 for all δ , while for the first derivative

<!-- formula-not-decoded -->

we have h ′ δ ( ε ) ≥ 0 for δ ≤ 1 2 and 0 ≤ ε ≤ ε 0. It remains to show that δ ≤ 1 2 and ε ≤ 1 -2 δ imply ε ≤ ε 0. Indeed, for δ ≤ 1 2 and ε ≤ 1 -2 δ we have

<!-- formula-not-decoded -->

## References

Peter Auer and Ronald Ortner. Logarithmic online regret bounds for reinforcement learning. In Advances in Neural Information Processing Systems 19 (NIPS 2006) , pages 49-56. MIT Press, 2007.

- Peter Auer and Ronald Ortner. Online regret bounds for a new reinforcement learning algorithm. In Proceedings 1st Austrian Cognitive Vision Workshop (ACVW 2005) , pages 35-42. ÖCG, 2005.
- Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multi-armed bandit problem. Mach. Learn. , 47:235-256, 2002a.
- Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM J. Comput. , 32:48-77, 2002b.
- Peter L. Bartlett and Ambuj Tewari. REGAL: A regularization based algorithm for reinforcement learning in weakly communicating MDPs. In Proceedings of the 25th Annual Conference on Uncertainty in Artificial Intelligence (UAI 2009) , 2009.
- Ronen I. Brafman and Moshe Tennenholtz. R-max - a general polynomial time algorithm for nearoptimal reinforcement learning. J. Mach. Learn. Res. , 3:213-231, 2002.
- Apostolos N. Burnetas and Michael N. Katehakis. Optimal adaptive policies for Markov decision processes. Math. Oper. Res. , 22(1):222-255, 1997.
- Eyal Even-Dar, Sham M. Kakade, and Yishay Mansour. Experts in a Markov decision process. In Advances in Neural Information Processing Systems 17 (NIPS 2004) , pages 401-408. MIT Press, 2005.
- Eyal Even-Dar, Sham M. Kakade, and Yishay Mansour. Online Markov decision processes. Math. Oper. Res. , 34(3):726-736, 2009.
- Claude-Nicolas Fiechter. Efficient reinforcement learning. In Proceedings of the Seventh Annual ACM Conference on Computational Learning Theory (COLT 1994) , pages 88-97. ACM, 1994.
- David A. Freedman. On tail probabilities for martingales. Ann. Probab. , 3:100-118, 1975.
- Aurélien Garivier and Eric Moulines. On upper-confidence bound polic ies for non-stationary bandit problems. Preprint, 2008. URL http://arxiv.org/pdf/0805.3415 .
- Wassily Hoeffding. Probability inequalities for sums of bounded random variables. J. Amer. Statist. Assoc. , 58:13-30, 1963.
- Sham M. Kakade. On the Sample Complexity of Reinforcement Learning . PhD thesis, University College London, 2003.
- Michael J. Kearns and Satinder P. Singh. Near-optimal reinforcement learning in polynomial time. Mach. Learn. , 49:209-232, 2002.
- Michael J. Kearns and Satinder P. Singh. Finite-sample convergence rates for Q-learning and indirect algorithms. In Advances in Neural Information Processing Systems 11 (NIPS 1998) , pages 996-1002. MIT Press, 1999.
- Shie Mannor and John N. Tsitsiklis. The sample complexity of exploration in the multi-armed bandit problem. J. Mach. Learn. Res. , 5:623-648, 2004.

- Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, Inc., New York, NY, USA, 1994.
- Alexander L. Strehl and Michael L. Littman. A theoretical analysis of model-based interval estimation. In Machine Learning, Proceedings of the Twenty-Second International Conference (ICML 2005) , pages 857-864. ACM, 2005.
- Alexander L. Strehl and Michael L. Littman. An analysis of model-based interval estimation for Markov decision processes. J. Comput. System Sci. , 74(8):1309-1331, 2008.
- Alexander L. Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L. Littman. PAC model-free reinforcement learning. In Machine Learning, Proceedings of the Twenty-Third International Conference (ICML 2006) , pages 881-888. ACM, 2006.
- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . MIT Press, 1998.
- Ambuj Tewari and Peter Bartlett. Optimistic linear programming gives logarithmic regret for irreducible MDPs. In Advances in Neural Information Processing Systems 20 (NIPS 2007) , pages 1505-1512. MIT Press, 2008.
- Ambuj Tewari and Peter L. Bartlett. Bounded parameter Markov decision processes with average reward criterion. In Learning Theory, 20th Annual Conference on Learning Theory (COLT 2007) , pages 263-277, 2007.
- Tsachy Weissman, Erik Ordentlich, Gadiel Seroussi, Sergio Verdu, and Marco L. Weinberger. Inequalities for the L1 deviation of the empirical distribution. Technical Report HPL-2003-97, HP Laboratories Palo Alto, 2003. URL www.hpl.hp.com/techreports/2003/HPL-2003-97R1. pdf .
- Jia Yuan Yu and Shie Mannor. Piecewise-stationary bandit problems with side observations. In Proceedings of the 26th Annual International Conference on Machine Learning (ICML 2009) , pages 1177-1184, 2009.
- Jia Yuan Yu, Shie Mannor, and Nahum Shimkin. Markov decision processes with arbitrary reward processes. Math. Oper. Res. , 34(3):737-757, 2009.