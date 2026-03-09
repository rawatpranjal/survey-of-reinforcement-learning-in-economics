## A new convergent variant of Q -learning with linear function approximation

## Diogo S. Carvalho Francisco S. Melo Pedro A. Santos

INESC-ID &amp; Instituto Superior Técnico, University of Lisbon Lisbon, Portugal

{diogo.s.carvalho, pedro.santos}@tecnico.ulisboa.pt fmelo@inesc-id.pt

## Abstract

In this work, we identify a novel set of conditions that ensure convergence with probability 1 of Q -learning with linear function approximation, by proposing a two time-scale variation thereof. In the faster time scale, the algorithm features an update similar to that of DQN, where the impact of bootstrapping is attenuated by using a Q -value estimate akin to that of the target network in DQN. The slower time-scale, in turn, can be seen as a modified target network update. We establish the convergence of our algorithm, provide an error bound and discuss our results in light of existing convergence results on reinforcement learning with function approximation. Finally, we illustrate the convergent behavior of our method in domains where standard Q -learning has previously been shown to diverge.

## 1 Introduction

In this paper, we investigate the convergence of reinforcement learning with linear function approximation in control settings. Specifically, we analyze the convergence of Q -learning when combined with linear function approximation. Several well-known counter-examples exist in the literature that showcase the divergence of this algorithm when used with even such a relatively 'benign' form of function approximation [2, 6, 21]. The divergent behavior has been blamed on the so-called 'deadly triad' [18, 24]-function approximation, bootstrapping and off-policy learning. Bootstrapping means that successive estimates for the Q -function are built on previous estimates; over-estimation errors for action values thus critically propagate across iterations [23]. Off-policy means that the policy used to sample the environment differs from that which the algorithm is evaluating.

The few results that establish the convergence of Q -learning with function approximation either restrict the approximation architecture, eventually minimizing the impact of over-estimation errors [20, 21] or require a very restrictive coupling between the approximation architecture and the sampling distribution which, in practice, occurs only when the sampling policy is very close to the optimal policy, rendering Q -learning almost on-policy [13]. More recently, Q -learning was attributed finitetime error bounds when certain fixed behaviour policies are used [8]. Unfortunately, such policies are scarce or may not even exist as the number of features grows.

Other convergence results for the control problem in RL with function approximation propose algorithms in which at least one of the elements in the 'deadly triad' is not present. For example, in a work of Perkins and Precup (2003), the authors propose a novel algorithm that converges with arbitrary function approximation, but is restricted to on-policy sampling. Greedy-GQ [12] is a variant of Q -learning with convergence guarantees. However, the associated solution may not be globally optimal, following from the fact it minimizes a non-convex objective function. Finite-time error bounds for Asynchronous Dynamic Programming methods [3], including Fitted Q -iteration (FQI)

[9], assume not only realizability of the optimal Q -function but also closedness under Bellman update [19, 7]. Finally, convergence results have been established for the return-base setting [16].

Our work is motivated by the success of DQN [14], which can be viewed as an instance of FQI [25]. In their work, Mnih et al. explore two important ideas in order to circumvent (or, at least, mitigate) the negative impact of bootstrapping and off-policy sampling:

- The use of a target network to compute the target value for the updates. In the original version, the target network is updated only rarely, by copying the values of the original network, although posterior implementations have adopted Polyak updates, in this sense bringing DQN closer to a two time-scale update scheme;
- The use of experience replay , where the samples are drawn from a replay buffer , thus minimizing the correlation between samples observed in trajectory-based learning and enabling the use of supervised learning techniques that assume sample independence.

Building on these ideas, we propose a two time-scale variation of Q -learning with linear function approximation. Our proposed algorithm keeps two sets of parameters.

- The first set of parameters, corresponding to the 'main' iteration, follows a faster time-scale and uses a DQN-like update, where the targets are built from the second set of parametersthus minimizing the impact of bootstrapping.
- The second set of parameters, corresponding to the 'target network', proceeds at a slower time-scale-in a sense mimicking the slower updates of a target network. However, unlike DQN, the second set of parameters does not directly copy the 'main' but, instead, a transformed version thereof, reminiscent of the preconditioning process discussed in [1].

We contribute with a convergence analysis of the resulting algorithm, showing convergence with probability 1 , or w.p.1 , with much less stringent assumptions than previous works [13] and provide an interpretation and performance bounds for the resulting limit solution.

## Notation:

We denote random variables (r.v.s) using upright letters, as in x or a , and instances of r.v.s as slanted letters, as in x or a . We use uppercase letters to denote functions, as in V or Q , and calligraphic letters to denote sets, as in X or A . Vectors are represented as bold lowercase letters. For example, c denotes a random vector and c an instance thereof. Matrices are represented using bold uppercase letters, as in Q . We write E x , a ∼ p [ f (x , a)] or simply E p [ f (x , a)] to denote the expectation of f when the r.v.s x and a follow distribution p .

## 2 Background

A Markov decision problem (MDP) is a tuple ( X , A , { P a } , R, γ ) , where X is the countable state space, A is the finite action space, P a is the transition probability matrix associated with action a ∈ A , with component xx ′ given by

<!-- formula-not-decoded -->

The random variable (r.v.) x t denotes the state of the MDP at time step t ; similarly, the r.v. a t denotes the action of the agent at time step t . We generally write P ( x ′ | x, a ) to denote [ P a ] xx ′ .

The function R : X × A → R is the expected reward for performing action a in state x . We write r t to denote the reward at time step t . It is a r.v. with expected value R (x t , a t ) and we assume throughout that | r t | ≤ ρ for some value ρ &gt; 0 . Finally, γ is a discount factor taking values in [0 , 1) .

Solving an MDP consists in finding a policy (i.e., a decision rule) that yields the maximal total discounted reward . A policy is a (possibly stochastic) mapping π : X → A , where π ( x ) is the action selected by π at state x . The total discounted reward associated with a policy π is

<!-- formula-not-decoded -->

We refer to V π as the value function associated with policy π . The optimal policy π ∗ is such that V π ∗ ( x ) ≥ V π ( x ) for any other policy π . We write V ∗ to compactly denote V π ∗ and refer to V ∗ as the optimal value function . The optimal value function verifies the recursive relation

<!-- formula-not-decoded -->

where the quantity in square brackets is known as the optimal action-value for the state-action pair ( x, a ) , and is denoted as Q ∗ ( x, a ) . Since V ∗ ( x ) = max a ∈A Q ∗ ( x, a ) , it holds that Q ∗ verifies

<!-- formula-not-decoded -->

An optimal policy is any policy π ∗ such that π ∗ ( x ) ∈ arg max a ∈A Q ∗ ( x, a ) .

V ∗ and Q ∗ can be computed, respectively, from (1) and (2) using dynamic programming. Alternatively, if { P a , a ∈ A} and R are unknown, they can be computed using stochastic approximation.

In this paper, we are particularly interested in the stochastic approximation approach to the computation of Q ∗ , an algorithm known as Q -learning . Given a sequence of observed tuples B = { ( x 0 , a 0 , r 0 , x ′ 0 ) , ( x 1 , a 1 , r 1 , x ′ 1 ) , . . . , ( x t , a t , r t , x ′ t ) , . . . } , obtained by running some learning policy in the MDP, Q -learning proceeds by performing, at each time step t , the update

<!-- formula-not-decoded -->

where δ t is the temporal difference at time step t , given by

<!-- formula-not-decoded -->

Convergence of Q -learning can be established using standard stochastic approximation arguments.

## 2.1 Q -learning with function approximation

We now address the problem of control in reinforcement learning with function approximation , where the optimal Q -function, Q ∗ , cannot be represented exactly and, therefore, some form of approximation must be used.

Let us consider a parameterized family of functions Q = { Q w , w ∈ R K } , with Q w : X × A → R . The extension of Q -learning to accommodate for one such representation takes the general form

<!-- formula-not-decoded -->

The update (3) can be seen as a single-sample stochastic gradient update over the error E = E [ δ 2 t ] , if we assume the target value R ( x t , a t ) + γ max a ′ ∈A Q w t ( x ′ t , a ′ ) is fixed. Intuitively, the error E conveys the notion that the parameters should be adjusted so that the output, Q w t ( x t , a t ) , approaches such target value as well as possible.

Unfortunately, it is a well known phenomenon that having the target value, R ( x t , a t ) + γ max a ′ ∈A Q w t ( x ′ t , a ) , built from the output Q w t of the learning algorithm (bootstrapping) may cause the resulting algorithm to diverge.

The recently proposed DQN architecture [14] seeks to alleviate the potential negative impact of bootstrapping by using a target network to construct the value of the target above. Such target network is updated infrequently; the targets used to train DQN are, therefore, mostly static. Building on that idea, we contribute a novel method which can be seen as implementing a two time-scale equivalent to the target network in DQN. We analyze the convergence of our proposed algorithm when used with linear function approximators, and contribute with:

- Anovel proof of convergence of Q -learning with linear function approximation that requires significantly less stringent conditions that those currently available in the literature;
- A better theoretical understanding for the use of the target network in DQN.

## 3 Coupled Q -learning

We are given a set of basis functions { φ 1 , . . . , φ K } , where φ k : X ×A → R for k = 1 , . . . , K . The function Q ∗ is then approximated by a function Q w ∈ { φ · w , w ∈ R K } , where φ = ( φ 1 , . . . , φ K ) . We designate our method as coupled Q -learning , or CQL , as it consists of the two coupled updates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now use the temporal difference

<!-- formula-not-decoded -->

and α t &lt;&lt; β t . In (4), Q u plays the role of target network and Q v implements the role of the 'main' network, implementing the same Q -learning update used in DQN. The update (4a) takes place at a slower time scale than the update (4b), emulating the infrequent updates in the target network of DQN [14]. Finally, we do not directly copy the values of the 'main' network, but instead match the projection of the output along the feature space, much like the 'pre-conditioning' step from [1].

## 3.1 Convergence analysis

We now analyze the convergence of our proposed algorithm. We establish convergence w.p.1 under the following assumptions.

- (I) For all t , (x t , a t , x ′ t , r t ) is independently sampled from a replay buffer B = { ( x i , a i , x ′ i , r i ) , i ∈ N 0 } according to a fixed distribution µ . Each tuple ( x i , a i , x ′ i , r i ) is sampled from the MDP, i.e., x ′ i is distributed according to P ( · | x i , a i ) , and r i is such that E µ [r i | x i , a i ] = R ( x i , a i ) .
- (II) Σ µ def = E µ [ φ (x t , a t ) φ T (x t , a t ) ] is non-singular and ‖ φ ( x, a ) ‖ 2 ≤ 1 , for all ( x, a ) .
- (III) The step size sequences { α t , t ∈ N } and { β t , t ∈ N } verify the conditions ∑ ∞ t =0 α t = ∑ ∞ t =0 β t = ∞ and ∑ ∞ t =0 α 2 t + ∑ ∞ t =0 β 2 t &lt; ∞ . Additionally, α t = o( β t ) .

Our main result follows.

Theorem 1. Under Assumptions (I) through (III), the CQL algorithm defined by the updates (4) converges w.p.1.

Before moving to the proof of Theorem 1, let us consider the implications of Assumptions (I)-(III).

Assumption (I) is similar to the setting considered by Chen and Jiang [7]. It indicates that the tuples ( x t , a t , r t , x ′ t ) used to perform the updates are mutually independent-which can be implemented, for example, through a replay buffer similar to what is done in many current deep RL approaches. Several previous works have considered the distribution µ to be the stationary distribution for the Markov chain induced by the sampling policy [13, 20], which typically requires the sampling policy to induce an ergodic Markov chain. In this sense, Assumption (I) is less restrictive, as it makes no particular assumption on the sampling policy but simplifies the convergence analysis.

Assumption (II) requires ‖ φ ( x, a ) ‖ 2 ≤ 1 for all ( x, a ) ∈ X × A . This is a relatively straightforward condition to ensure by a simple scaling of the features φ 1 , . . . , φ K and is also present in recent related work [26, 8]. Assumption (II) also requires that the matrix Σ µ is non-singular, a condition which is tantamount to the linear independence requirement found in previous works [22]. Assumption (II) is, therefore, significantly weaker than those imposed, for example, in previous works [13], which seldom (if ever) hold in practice.

Assumption (III) is standard for two time-scale algorithms. In practice, the use of small constant step sizes α and β is usual, as long as α &lt;&lt; β .

## 3.1.1 Proof of Theorem 1

Proof. We establish Theorem 1 by directly applying the well-established two time-scale stochastic approximation result from Borkar [4, Chapter 6] provided in the supplementary material. To do such

application, it amounts to show that our algorithm satisfies each and every condition of the theorem. We present the main steps of the proof and refer to the supplementary material once more for details.

We start by defining the mean fields F, G : R K × R K → R K as

<!-- formula-not-decoded -->

Using F and G above, we also define martingale difference sequences of noise { m t } and { n t } as

<!-- formula-not-decoded -->

Both F and G are Lipschitz continuous, and letting the sigma-algebra F t = σ ( { ( u τ , v τ , m τ , n τ ) , τ = 0 , . . . , t } ) , we can show that

<!-- formula-not-decoded -->

for some constants c m , c n &gt; 0 .

In analyzing two time-scale algorithms we follow the standard notion that, since α t glyph[lessmuch] β t , the updates to v t proceeds at a 'faster' timescale than those to u t . Thus, when viewed from the faster time-scale, u t appears to be quasi-static . The update for u t takes the general form u t +1 = u t + β t ∆ u t , where

<!-- formula-not-decoded -->

as long as v t and u t remain bounded. With this in mind, for a fixed u ∈ R K , we have the o.d.e.

<!-- formula-not-decoded -->

which has a unique globally asymptotically stable equilibrium

<!-- formula-not-decoded -->

The global asymptotic stability of v ∗ can be established by a Lyapunov argument, using the Lyapunov function L ( v ) = 1 2 ‖ v -v ∗ ‖ 2 . Additionally, λ ( u ) can also be shown to be Lipschitz continuous. Finally, defining

<!-- formula-not-decoded -->

we can show that the origin is an asymptotically stable equilibrium for the o.d.e. ˙ v t = G ∞ ( v t ) . From Theorem 2.1 of Borkar and Meyn (2000), we conclude that sup t ‖ v t ‖ &lt; ∞ w.p.1 .

Conversely, when viewed from the slower time-scale, v t appears to have already reached its equilibrium point . With this in mind, we have the o.d.e. ˙ u t = F ( u t , λ ( u t )) , which also has a unique globally asymptotically stable equilibrium

<!-- formula-not-decoded -->

The existence of the fixed point in (5) can be established from the Banach fixed-point theorem, since the right-hand side is a contraction in u ; that u ∗ is globally asymptotically stable can again be established by a Lyapunov argument, using the function L ( u ) = 1 2 ‖ u -u ∗ ‖ 2 . Finally, we define

<!-- formula-not-decoded -->

We can show that the origin is an asymptotically stable equilibrium for the o.d.e. ˙ u t = F ∞ ( u t ) , from where we repeat our previous argument to conclude that sup t ‖ u t ‖ &lt; ∞ w.p.1 . Since all conditions have been verified, the conclusion follows.

Proje

Qu

Proje

HQu*

Proje

Qui

Figure 1: Relation between ˆ Q , Q ∗ and the coupled solutions Q u ∗ and Q v ∗ .

<!-- image -->

## 3.2 Performance analysis

While Theorem 1 establishes the convergence of our algorithm w.p.1, it says nothing about the corresponding limit point. The limit considered in the context of linear function approximation is

<!-- formula-not-decoded -->

where Proj Φ is the orthogonal projection into the span of { φ k : k = 1 , . . . , K } , which we denote by Φ , and H is the Bellman operator. In our case we have, instead,

<!-- formula-not-decoded -->

The difference is the normalizing term Σ -1 µ . Therefore, Q u is the fixed point of the combined operator obtained from an 'un-normalized' orthogonal projection and the Bellman operator.

Taking this analysis one step further, and keeping the parallel between our approach and the DQN architecture, it is also interesting to analyze Q v ∗ , which corresponds to the actual output of the learning algorithm. We have

<!-- formula-not-decoded -->

The three functions ( ˆ Q , Q u ∗ and Q v ∗ ) coincide when Σ µ = I . However, this case does not fall, in general, under Assumption (II). Nevertheless, consider the case where the set of basis functions { φ k } is orthogonal (i.e., E µ [ φ i φ j ] = 0 ) and uniformly excited (i.e., E µ [ φ 2 i ] = E µ [ φ 2 j ] = σ ) by a factor σ taking values in (0 , 1] . Equivalently, let the following assumption hold.

<!-- formula-not-decoded -->

We note that Assumption (IV) does not impose any additional constraint on the features considered, since we can make them orthogonal and scale them to ensure that the latter assumption holds. Under assumptions (I) through (IV), we can now establish a bound on the error obtained when approximating Q ∗ by Q v ∗ . Consider the infinity norm ‖·‖ in the space of bounded real-valued functions on X × A .

Theorem 2. The limit v ∗ of the sequence { v t , t ∈ N } generated by the iterations in (4) is such that

<!-- formula-not-decoded -->

Before presenting the proof of Theorem 2, let us discuss its implications. Consider the right-hand side of (6). Intuitively, the first error term only depends on the proximity between Q ∗ and its best linear approximation for the given sampling policy and choice of features, within a factor of 1 -γ . But, more importantly, as σ approaches 1 , the second error term ξ σ goes to 0 . Additionally, σ is a relevant parameter when we consider the relation between the solution ˆ Q and the coupled solutions ( Q u ∗ , Q v ∗ ) , as suggested before. In fact, as σ converges to 1 , both Q u ∗ and Q v ∗ converge to ˆ Q . Figure 1 illustrates, geometrically, the interpretation just discussed. Dashed arrows refer to displacement components as σ approaches 1 .

1043-

1033-

1021-

10'

10-3.

10-13,

10-27

10-39

Q-learning

CQL

GGO

2000

4000

Episodes

103

101 1

10-1 -

10-31

<!-- image -->

- (a) Results on the θ → 2 θ example.
- (b) Results on the star counterexample.

Figure 2: Comparison of the proposed method with standard Q -learning and GGQ.

## 3.2.1 Proof of Theorem 2

Proof. We have that

<!-- formula-not-decoded -->

Consider the second term on the right-hand side. Since Q ∗ = H Q ∗ and Q v ∗ = Proj Φ H Q ∗ u , we get

<!-- formula-not-decoded -->

by means of the Cauchy-Schwarz and Jensen inequalities and assumption (II).

Since Q u ∗ = σQ v ∗ and ‖ Q ∗ ‖ ∞ ≤ ρ/ (1 -γ ) , we can put everything together to get

<!-- formula-not-decoded -->

Solving the inequality for ‖ Q ∗ -Q v ∗ ‖ ∞ , the desired result follows.

## 4 Experimental results

We evaluated the CQL algorithm on three domains with increasing complexity. The first was the θ → 2 θ example [21] and the second was the 7-star version of the star counterexample [2]. Both problems are known two cause divergence of Q -learning with linear function approximation. We also tested the algorithm on the mountain car problem [15]. On each domain, we compare CQL with standard Q -learning and GGQ [12]. We performed online learning on the second and third tests, showing that the use of a replay buffer satisfying Assumption (I) is not necessary for convergence.

On the first two domains, results were averaged over 30 runs of 10 3 episodes, considered γ = 0 . 99 and constant learning-rates: α = 0 . 1 for the original algorithm; α = 0 . 05 , β = 0 . 25 for CQL and GGQ. After each episode, we compute ‖ Q ‖ F = ∑ x ∈X ,a ∈A Q 2 ( x, a ) .

<!-- formula-not-decoded -->

In the most simple example [21] there are only two states and one action, and the reward is always zero. The only feature has value 1 for the first state and 2 for the second state. We would expect with the use of a simple architecture such as this one, Q -learning would converge to Q ∗ = 0 .

We scaled the feature by a factor of 1 / 2 , set the initial weight as 1, and randomly initialized every episode with equal probability for each state. Each episode consisted of a transition and the update. Figure 2a shows Q -learning caused divergence of the Q -values, in contrast with the other two methods. The convergence of CQL is considerably faster than of GGQ.

## 4.2 Star counterexample

A more complex domain [2] considers seven states and two actions: the solid and the dotted action. State seven is absorbing for the solid action, and the dotted action uniformly transitions the agent to any of the first six states. Both actions, on any state, incur zero reward, and thus Q ∗ = 0 .

Table 1: Results on the mountain car problem. For each architecture, the best result is bolden.

| Architecture   | Architecture   | Average cumulative reward ( ± std. dev.)   | Average cumulative reward ( ± std. dev.)   | Average cumulative reward ( ± std. dev.)   |
|----------------|----------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| l              | η              | Q -learning                                | GGQ                                        | CQL                                        |
| 2              | 0.125          | -190.95 ± 23.42                            | -167.93 ± 39.28                            | -172.66 ± 35.82                            |
| 2              | 0.25           | -181.67 ± 30.95                            | -187.02 ± 24.02                            | -168.03 ± 39.17                            |
| 2              | 0.5            | -175.54 ± 35.95                            | -180.55 ± 31.08                            | -166.48 ± 38.87                            |
| 2              | 1              | -167.98 ± 39.22                            | -174.15 ± 36.70                            | -148.99 ± 36.33                            |
| 4              | 0.125          | -181.10 ± 31.74                            | -173.75 ± 36.23                            | -175.51 ± 34.52                            |
| 4              | 0.25           | -174.48 ± 36.57                            | -168.26 ± 34.10                            | -175.55 ± 37.35                            |
| 4              | 0.5            | -158.12 ± 38.68                            | -163.31 ± 35.91                            | -150.59 ± 37.99                            |
| 4              | 1              | -159.80 ± 40.20                            | -169.76 ± 35.36                            | -164.58 ± 32.55                            |
| 8              | 0.125          | -149.12 ± 36.25                            | -133.38 ± 19.25                            | -163.42 ± 30.11                            |
| 8              | 0.25           | -121.43 ± 14.50                            | -143.83 ± 15.83                            | -144.94 ± 33.76                            |
| 8              | 0.5            | -181.59 ± 25.35                            | -171.11 ± 33.46                            | -182.81 ± 22.97                            |
| 8              | 1              | -187.21 ± 20.05                            | -193.46 ± 13.60                            | -184.57 ± 15.61                            |

We used the same features as those described in the original work [2] but scaled by a factor of 1 / √ 5 . To initiate each run, we set the vectors as (1 , 1 , 1 , 1 , 1 , 1 , 10 , 1) for the solid action and the rest as 1 [12]. The auxiliary set of parameters of GGQ was initialized as 0 . On every run, we initialized the trajectory in any of the six states with equal probability. The dotted action and the solid action were then chosen with probability 5 / 6 and 1 / 6 , respectively, on every episode. Figure 2b shows that the original Q -learning caused the Q-values to diverge, whereas CQL and GGQ led them to converge. The GGQ algorithm stabilizes in a sub-optimal solution but CQL converges to the true solution.

## 4.3 Mountain car

The mountain car is a more realistic and complex control problem: a car is placed in the middle of two hills and on each time step it can either accelerate left, do nothing, or accelerate right, forming the action space A . In this environment, gravity is stronger than the engine, and therefore some strategy must be found to climb the hill. Every time step results in a -1 reward, except for reaching the goal.

The basis functions used were bi-dimensional Gaussians. Each of the Gaussians had mean in the center of a square on a l × l grid over the state space X of position and velocity pairs. The standard deviations were σ p and σ v on the position and velocity dimensions, respectively. The basis functions were normalized so that ‖ φ ( x, a ) ‖ 2 = 1 for each ( x, a ) . Three such functions, in each state, were then associated with the three possible actions. On each run, the initial vectors were 0 . The learning parameters used were pairs ( α, β ) ∈ { (10 -i , 10 -j ) , i = 1 , . . . , 4 , j = i, . . . , 4 } . In the case of Q -learning, only α is used. During training, each run consisted of 10 3 episodes. Each episode ended when the car successfully climbed the hill or 200 transitions were made. An glyph[epsilon1] -greedy policy was used to learn, with glyph[epsilon1] = 0 . 3 . For testing, after each run, we computed the average cumulative reward of the greedy policy obtained over 100 episodes. Finally, results were averaged over 10 runs.

Table 1 shows, for each algorithm and values of l and η = σ p 1 . 8 = σ v 0 . 14 , the results obtained from the best learning-rate pair ( α, β ) . For CQL, the most selected learning-rate pair was α = β = 10 -4 . Also, CQL performed the best when l = 2 , which testifies for the benefits of simple approximation architectures to the proposed method, as suggested by Theorem 2.

## 5 Conclusions and future work

By proposing a two time-scale variant of Q -learning able to combine linear function approximation and off-policy sampling of trajectories, and establishing its convergence under general assumptions, we revived the discussion of convergence for this broadly employed algorithm and introduced a theoretical foundation regarding the use of DQN. We validated the effectiveness of the construction on classical examples where Q -learning in its standard configuration fails.

Assumption (II) bounds the feature vectors by the 2-norm. Even though it is achieved without loss of generality, a bound on a non-Euclidean norm, not agnostic to the sampling distribution µ , would be preferable, allowing the limit solutions to be closer to Q ∗ .

## Broader impact

Even though our work is mostly theoretical, we include a reflection on the general impacts of the field.

Artificial intelligence (AI) and machine learning (ML) adoption in society is rapidly increasing. Within the classical application domains-such as robotics, natural language processing, computer vision, predictive models, and others, AI algorithms are now a part of our daily lives. In some of those applications, AI and ML-driven algorithms can surpass human-level performance. Additionally, these algorithms are being used to address critical problems in our world. For example, deep learning algorithms are being used to predict poverty from satellite images [10], and to predict and manage traffic patterns to avoid pollution and congestion in cities [11].

While the impacts of intelligent algorithms are immense, many solid empirical successes are not supported by an equally solid theoretical understanding. This gap between theory and practice is notorious in the field of reinforcement learning (RL), particularly with respect to the recent successes of deep RL. The present work contributes a new algorithm-a modification of Q -learning inspired by successful architectures such as DQN-along with the theoretical analysis of its properties. Such contribution has the potential to impact the broader area of RL, including deep RL, by providing a sharper understanding of the theoretical properties of RL algorithms and potentially pushing research towards a new class of stable RL algorithms which make use of more complex approximation architectures than the ones considered in this worke.g. , neural networks.

Given the fundamental nature of the work, we expect its impact on our daily lives to be as far-reaching as that of AI and ML.

## Acknowledgments and Disclosure of Funding

This work was partially supported by national funds through Fundação para a Ciência e Tecnologia under project SLICE with reference PTDC/CCI-COM/30787/2017 and INESC-ID multi annual funding with reference UIDB/50021/2020.

## References

- [1] J. Achiam, E. Knight, and P. Abbeel. Towards characterizing divergence in deep Q -learning. CoRR , abs/1903.08894, 2019.
- [2] L. Baird. Residual algorithms: Reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Machine Learning , pages 30-37, 1995.
- [3] D. Bertsekas and J. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- [4] V. Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint . Cambridge University Press, 2008.
- [5] V. Borkar and S. Meyn. The O.D.E. method for convergence of stochastic approximation and reinforcement learning. SIAM J. Control and Optimization , 38(2):447-469, 2000.
- [6] J. Boyan and A. Moore. Generalization in reinforcement learning: Safely approximating the value function. In Advances in Neural Information Processing Systems 7 , pages 369-376, 1995.
- [7] Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In Proceedings of the 36th International Conference on Machine Learning , pages 1042-1051, 2019.
- [8] Z. Chen, S. Zhang, T. T. Doan, S. T. Maguluri, and J. Clarke. Performance of q-learning with linear function approximation: Stability and finite-time analysis. arXiv preprint arXiv:1905.11425 , 2019.
- [9] D. Ernst, P. Geurts, and L. Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6(Apr):503-556, 2005.

- [10] N. Jean, M. Burke, M. Xie, W. M. Davis, D. B. Lobell, and S. Ermon. Combining satellite imagery and machine learning to predict poverty. Science , 353(6301):790-794, 2016.
- [11] L. Li, Y. Lv, and F. Wang. Traffic signal timing via deep reinforcement learning. IEEE/CAA Journal of Automatica Sinica , 3(3):247-254, 2016.
- [12] H. R. Maei, C. Szepesvári, S. Bhatnagar, and R. S. Sutton. Toward off-policy learning control with function approximation. In Proceedings of the 27th International Conference on Machine Learning , pages 719-726, 2010.
- [13] F.S. Melo, S. Meyn, and M.I. Ribeiro. An analysis of reinforcement learning with function approximation. In Proceedings of the 25th International Conference on Machine learning , pages 664-671, 2008.
- [14] V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature , 518:529-533, 2015.
- [15] A. Moore. Efficient memory-based learning for robot control. Technical Report UCAM-CLTR-209, University of Cambridge, 1990.
- [16] R. Munos, T. Stepleton, A. Harutyunyan, and M. Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems , pages 10541062, 2016.
- [17] T. J. Perkins and D. Precup. A convergent form of approximate policy iteration. In Advances in neural information processing systems , pages 1627-1634, 2003.
- [18] R. Sutton and A. Barto. Reinforcement Learning: An Introduction . MIT Press, 2nd edition, 2018.
- [19] C. Szepesvári and R. Munos. Finite time bounds for sampling based fitted value iteration. In Proceedings of the 22nd international conference on Machine learning , pages 880-887, 2005.
- [20] C. Szepesvári and W. Smart. Interpolation-based Q -learning. In Proceedings of the 21st International Conference on Machine learning , pages 100-107, 2004.
- [21] J. Tsitsiklis and B. Van Roy. Feature-based methods for large scale dynamic programming. Machine Learning , 22:59-94, 1996.
- [22] J. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Trans. Automatic Control , 42(5):674-690, 1996.
- [23] H. van Hasselt. Double Q -learning. In Advances in Neural Information Processing Systems , pages 2613-2621, 2010.
- [24] H. van Hasselt, Y. Doron, F. Strub, M. Hessel, N. Sonnerat, and J. Modayil. Deep reinforcement learning and the deadly triad. CoRR , abs/1812.02648, 2018.
- [25] Z. Yang, Y. Xie, and Z. Wang. A theoretical analysis of deep q-learning. arXiv preprint arXiv:1901.00137 , 2019.
- [26] S. Zou, T. Xu, and Y. Liang. Finite-sample analysis for sarsa with linear function approximation. In Advances in Neural Information Processing Systems , pages 8665-8675, 2019.