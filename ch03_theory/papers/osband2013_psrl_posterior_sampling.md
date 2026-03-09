## (More) Efficient Reinforcement Learning via Posterior Sampling

Ian Osband

Stanford University

Stanford, CA 94305 iosband@stanford.edu

## Benjamin Van Roy

Stanford University Stanford, CA 94305

bvr@stanford.edu

## Abstract

Most provably-efficient reinforcement learning algorithms introduce optimism about poorly-understood states and actions to encourage exploration. We study an alternative approach for efficient exploration: posterior sampling for reinforcement learning (PSRL). This algorithm proceeds in repeated episodes of known duration. At the start of each episode, PSRL updates a prior distribution over Markov decision processes and takes one sample from this posterior. PSRL then follows the policy that is optimal for this sample during the episode. The algorithm is conceptually simple, computationally efficient and allows an agent to encode prior knowledge in a natural way. We establish an ˜ O ( τS √ AT ) bound on expected regret, where T is time, τ is the episode length and S and A are the cardinalities of the state and action spaces. This bound is one of the first for an algorithm not based on optimism, and close to the state of the art for any reinforcement learning algorithm. We show through simulation that PSRL significantly outperforms existing algorithms with similar regret bounds.

## 1 Introduction

We consider the classical reinforcement learning problem of an agent interacting with its environment while trying to maximize total reward accumulated over time [1, 2]. The agent's environment is modeled as a Markov decision process (MDP), but the agent is uncertain about the true dynamics of the MDP. As the agent interacts with its environment, it observes the outcomes that result from previous states and actions, and learns about the system dynamics. This leads to a fundamental tradeoff: by exploring poorly-understood states and actions the agent can learn to improve future performance, but it may attain better short-run performance by exploiting its existing knowledge.

Na¨ ıve optimization using point estimates for unknown variables overstates an agent's knowledge, and can lead to premature and suboptimal exploitation. To offset this, the majority of provably efficient learning algorithms use a principle known as optimism in the face of uncertainty [3] to encourage exploration. In such an algorithm, each state and action is afforded some optimism bonus such that their value to the agent is modeled to be as high as is statistically plausible. The agent will then choose a policy that is optimal under this 'optimistic' model of the environment. This incentivizes exploration since poorly-understood states and actions will receive a higher optimism bonus. As the agent resolves its uncertainty, the effect of optimism is reduced and the agent's behavior approaches optimality. Many authors have provided strong theoretical guarantees for optimistic algorithms [4, 5, 6, 7, 8]. In fact, almost all reinforcement learning algorithms with polynomial bounds on sample complexity employ optimism to guide exploration.

## Daniel Russo

Stanford University Stanford, CA 94305

djrusso@stanford.edu

We study an alternative approach to efficient exploration, posterior sampling , and provide finite time bounds on regret. We model the agent's initial uncertainty over the environment through a prior distribution. 1 At the start of each episode , the agent chooses a new policy, which it follows for the duration of the episode. Posterior sampling for reinforcement learning (PSRL) selects this policy through two simple steps. First, a single instance of the environment is sampled from the posterior distribution at the start of an episode. Then, PSRL solves for and executes the policy that is optimal under the sampled environment over the episode. PSRL randomly selects policies according to the probability they are optimal; exploration is guided by the variance of sampled policies as opposed to optimism.

The idea of posterior sampling goes back to 1933 [9] and has been applied successfully to multi-armed bandits. In that literature, the algorithm is often referred to as Thompson sampling or as probability matching . Despite its long history, posterior sampling was largely neglected by the multi-armed bandit literature until empirical studies [10, 11] demonstrated that the algorithm could produce state of the art performance. This prompted a surge of interest, and a variety of strong theoretical guarantees are now available [12, 13, 14, 15]. Our results suggest this method has great potential in reinforcement learning as well.

PSRL was originally introduced in the context of reinforcement learning by Strens [16] under the name 'Bayesian Dynamic Programming', 2 where it appeared primarily as a heuristic method. In reference to PSRL and other 'Bayesian RL' algorithms, Kolter and Ng [17] write 'little is known about these algorithms from a theoretical perspective, and it is unclear, what (if any) formal guarantees can be made for such approaches.' Those Bayesian algorithms for which performance guarantees exist are guided by optimism. BOSS [18] introduces a more complicated version of PSRL that samples many MDPs, instead of just one, and then combines them into an optimistic environment to guide exploration. BEB [17] adds an exploration bonus to states and actions according to how infrequently they have been visited. We show it is not always necessary to introduce optimism via a complicated construction, and that the simple algorithm originally proposed by Strens [16] satisfies strong bounds itself.

Our work is motivated by several advantages of posterior sampling relative to optimistic algorithms. First, since PSRL only requires solving for an optimal policy for a single sampled MDP, it is computationally efficient both relative to many optimistic methods, which require simultaneous optimization across a family of plausible environments [4, 5, 18], and to computationally intensive approaches that attempt to approximate the Bayes-optimal solutions directly [18, 19, 20]. Second, the presence of an explicit prior allows an agent to incorporate known environment structure in a natural way. This is crucial for most practical applications, as learning without prior knowledge requires exhaustive experimentation in each possible state. Finally, posterior sampling allows us to separate the algorithm from the analysis . In any optimistic algorithm, performance is greatly influenced by the manner in which optimism is implemented. Past works have designed algorithms, at least in part, to facilitate theoretical analysis for toy problems. Although our analysis of posterior sampling is closely related to the analysis in [4], this worst-case bound has no impact on the algorithm's actual performance. In addition, PSRL is naturally suited to more complex settings where design of an efficiently optimistic algorithm might not be possible. We demonstrate through a computational study in Section 6 that PSRL outperforms the optimistic algorithm UCRL2 [4]: a competitor with similar regret bounds over some example MDPs.

## 2 Problem formulation

We consider the problem of learning to optimize a random finite horizon MDP M = ( S , A , R M , P M , τ, ρ ) in repeated finite episodes of interaction. S is the state space, A is the action space, R a M ( s ) is a probability distribution over reward realized when selecting action a while in state s whose support is [0 , 1], P a M ( s ′ | s ) is the probability of transitioning to state s ′ if action a is selected while at state s , τ is the time horizon, and ρ the initial state distribution. We define the MDP and all other random variables we will consider with

1 For an MDP, this might be a prior over transition dynamics and reward distributions.

2 We alter terminology since PSRL is neither Bayes-optimal, nor a direct approximation of this.

respect to a probability space (Ω , F , P ). We assume S , A , and τ are deterministic so the agent need not learn the state and action spaces or the time horizon.

A deterministic policy µ is a function mapping each state s ∈ S and i = 1 , . . . , τ to an action a ∈ A . For each MDP M = ( S , A , R M , P M , τ, ρ ) and policy µ , we define a value function

<!-- formula-not-decoded -->

where R M a ( s ) denotes the expected reward realized when action a is selected while in state s , and the subscripts of the expectation operator indicate that a j = µ ( s j , j ), and s j +1 ∼ P a M j ( ·| s j ) for j = i, . . . , τ . A policy µ is said to be optimal for MDP M if V M µ,i ( s ) = max µ ′ V µ M ′ ,i ( s ) for all s ∈ S and i = 1 , . . . , τ . We will associate with each MDP M a policy µ M that is optimal for M .

The reinforcement learning agent interacts with the MDP over episodes that begin at times t k = ( k -1) τ + 1, k = 1 , 2 , . . . . At each time t , the agent selects an action a t , observes a scalar reward r t , and then transitions to s t +1 . If an agent follows a policy µ then when in state s at time t during episode k , it selects an action a t = µ ( s, t -t k ). Let H t = ( s 1 , a 1 , r 1 , . . . , s t -1 , a t -1 , r t -1 ) denote the history of observations made prior to time t . A reinforcement learning algorithm is a deterministic sequence { π k | k = 1 , 2 , . . . } of functions, each mapping H t k to a probability distribution π k ( H t k ) over policies. At the start of the k th episode, the algorithm samples a policy µ k from the distribution π k ( H t k ). The algorithm then selects actions a t = µ k ( s t , t -t k ) at times t during the k th episode.

We define the regret incurred by a reinforcement learning algorithm π up to time T to be

<!-- formula-not-decoded -->

where ∆ k denotes regret over the k th episode, defined with respect to the MDP M ∗ by

<!-- formula-not-decoded -->

with µ ∗ = µ M ∗ and µ k ∼ π k ( H t k ). Note that regret is not deterministic since it can depend on the random MDP M ∗ , the algorithm's internal random sampling and, through the history H t k , on previous random transitions and random rewards. We will assess and compare algorithm performance in terms of regret and its expectation.

## 3 Posterior sampling for reinforcement learning

The use of posterior sampling for reinforcement learning (PSRL) was first proposed by Strens [16]. PSRL begins with a prior distribution over MDPs with states S , actions A and horizon τ . At the start of each k th episode, PSRL samples an MDP M k from the posterior distribution conditioned on the history H t k available at that time. PSRL then computes and follows the policy µ k = µ M k over episode k .

## Algorithm: Posterior Sampling for Reinforcement Learning (PSRL)

```
Data : Prior distribution f , t=1 for episodes k = 1 , 2 , . . . do sample M k ∼ f ( ·| H t k ) compute µ k = µ M k for timesteps j = 1 , . . . , τ do sample and apply a t = µ k ( s t , j ) observe r t and s t +1 t = t +1 end end
```

We show PSRL obeys performance guarantees intimately related to those for learning algorithms based upon OFU, as has been demonstrated for multi-armed bandit problems [15]. We believe that a posterior sampling approach offers some inherent advantages. Optimistic algorithms require explicit construction of the confidence bounds on V M ∗ µ, 1 ( s ) based on observed data, which is a complicated statistical problem even for simple models. In addition, even if strong confidence bounds for V M ∗ µ, 1 ( s ) were known, solving for the best optimistic policy may be computationally intractable. Algorithms such as UCRL2 [4] are computationally tractable, but must resort to separately bounding R M a ( s ) and P a M ( s ) with high probability for each s, a . These bounds allow a 'worst-case' mis-estimation simultaneously in every state-action pair and consequently give rise to a confidence set which may be far too conservative.

By contrast, PSRL always selects policies according to the probability they are optimal. Uncertainty about each policy is quantified in a statistically efficient way through the posterior distribution. The algorithm only requires a single sample from the posterior, which may be approximated through algorithms such as Metropolis-Hastings if no closed form exists. As such, we believe PSRL will be simpler to implement, computationally cheaper and statistically more efficient than existing optimistic methods.

## 3.1 Main results

The following result establishes regret bounds for PSRL. The bounds have ˜ O ( τS √ AT ) expected regret, and, to our knowledge, provide the first guarantees for an algorithm not based upon optimism:

Theorem 1. If f is the distribution of M ∗ then,

<!-- formula-not-decoded -->

This result holds for any prior distribution on MDPs, and so applies to an immense class of models. To accommodate this generality, the result bounds expected regret under the prior distribution (sometimes called Bayes risk or Bayesian regret ). We feel this is a natural measure of performance, but should emphasize that it is more common in the literature to bound regret under a worst-case MDP instance. The next result provides a link between these notions of regret. Applying Markov's inequality to (1) gives convergence in probability.

Corollary 1. If f is the distribution of M ∗ then for any α &gt; 1 2 ,

<!-- formula-not-decoded -->

/negationslash

As shown in the appendix, this also bounds the frequentist regret for any MDP with non-zero probability. State-of-the-art guarantees similar to Theorem 1 are satisfied by the algorithms UCRL2 [4] and REGAL [5] for the case of non-episodic RL. Here UCRL2 gives regret bounds ˜ O ( DS √ AT ) where D = max s ′ = s min π E [ T ( s ′ | M,π,s )] and T ( s ′ | M,π,s ) is the first time step where s ′ is reached from s under the policy π . REGAL improves this result to ˜ O (Ψ S √ AT ) where Ψ ≤ D is the span of the of the optimal value function. However, there is so far no computationally tractable implementation of this algorithm.

In many practical applications we may be interested in episodic learning tasks where the constants D and Ψ could be improved to take advantage of the episode length τ . Simple modifications to both UCRL2 and REGAL will produce regret bounds of ˜ O ( τS √ AT ), just as PSRL. This is close to the theoretical lower bounds of √ SAT -dependence.

## 4 True versus sampled MDP

A simple observation, which is central to our analysis, is that, at the start of each k th episode, M ∗ and M k are identically distributed. This fact allows us to relate quantities that depend on the true, but unknown, MDP M ∗ , to those of the sampled MDP M k , which is

fully observed by the agent. We introduce σ ( H t k ) as the σ -algebra generated by the history up to t k . Readers unfamiliar with measure theory can think of this as 'all information known just before the start of period t k . ' When we say that a random variable X is σ ( H t k )-measurable, this intuitively means that although X is random, it is deterministically known given the information contained in H t k . The following lemma is an immediate consequence of this observation [15].

Lemma 1 (Posterior Sampling) . If f is the distribution of M ∗ then, for any σ ( H t k ) -measurable function g ,

<!-- formula-not-decoded -->

Note that taking the expectation of (2) shows E [ g ( M ∗ )] = E [ g ( M k )] through the tower property.

Recall, we have defined ∆ k = ∑ s ∈S ρ ( s )( V M ∗ µ ∗ , 1 ( s ) -V M ∗ µ k , 1 ( s )) to be the regret over period k . A significant hurdle in analyzing this equation is its dependence on the optimal policy µ ∗ , which we do not observe. For many reinforcement learning algorithms, there is no clean way to relate the unknown optimal policy to the states and actions the agent actually observes. The following result shows how we can avoid this issue using Lemma 1. First, define

<!-- formula-not-decoded -->

as the difference in expected value of the policy µ k under the sampled MDP M k , which is known, and its performance under the true MDP M ∗ , which is observed by the agent.

Theorem 2 (Regret equivalence) .

<!-- formula-not-decoded -->

and for any δ &gt; 0 with probability at least 1 -δ ,

Proof. Note, ∆ k -˜ ∆ k = ∑ s ∈S ρ ( s )( V M ∗ µ ∗ , 1 ( s ) -V M k µ k , 1 ( s )) ∈ [ -τ, τ ]. By Lemma 1, E [∆ k -˜ ∆ k | H t k ] = 0. Taking expectations of these sums therefore establishes the claim.

This result bounds the agent's regret in epsiode k by the difference between the agent's estimate V M k µ k , 1 ( s t k ) of the expected reward in M k from the policy it chooses, and the expected reward V M ∗ µ k , 1 ( s t k ) in M ∗ . If the agent has a poor estimate of the MDP M ∗ , we expect it to learn as the performance of following µ k under M ∗ differs from its expectation under M k . As more information is gathered, its performance should improve. In the next section, we formalize these ideas and give a precise bound on the regret of posterior sampling.

## 5 Analysis

An essential tool in our analysis will be the dynamic programming, or Bellman operator T µ M , which for any MDP M = ( S , A , R M , P M , τ, ρ ), stationary policy µ : S → A and value function V : S → R , is defined by

<!-- formula-not-decoded -->

This operation returns the expected value of state s where we follow the policy µ under the laws of M , for one time step. The following lemma gives a concise form for the dynamic programming paradigm in terms of the Bellman operator.

Lemma 2 (Dynamic programming equation) . For any MDP M = ( S , A , R M , P M , τ, ρ ) and policy µ : S × { 1 , . . . , τ } → A , the value functions V µ M satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to streamline our notation we will let V ∗ µ,i := V M ∗ µ,i , V k µ,i ( s ) := V M k µ,i ( s ), T k µ := T M k µ , T ∗ µ := T M ∗ µ and P ∗ µ ( ·| s ) := P M ∗ µ ( s ) ( ·| s ).

## 5.1 Rewriting regret in terms of Bellman error

<!-- formula-not-decoded -->

To see why (6) holds, simply apply the Dynamic programming equation inductively:

<!-- formula-not-decoded -->

where d t k + i := ∑ s ′ ∈S { P ∗ µ k ( · ,i ) ( s ′ | s t k + i )( V ∗ µ k ,i +1 -V k µ k ,i +1 )( s ′ ) } -( V ∗ µ k ,i +1 -V k µ k ,i +1 )( s t k + i ).

This expresses the regret in terms two factors. The first factor is the one step Bellman error [ ( T k µ k ( · ,i ) -T ∗ µ k ( · ,i ) ) V k µ k ,i +1 ( s t k + i ) ] under the sampled MDP M k . Crucially, (6) depends only the Bellman error under the observed policy µ k and the states s 1 , .., s T that are actually visited over the first T periods. We go on to show the posterior distribution of M k concentrates around M ∗ as these actions are sampled, and so this term tends to zero.

The second term captures the randomness in the transitions of the true MDP M ∗ . In state s t under policy µ k , the expected value of ( V ∗ µ k ,i +1 -V k µ k ,i +1 )( s t k + i ) is exactly ∑ s ′ ∈S { P ∗ µ k ( · ,i ) ( s ′ | s t k + i )( V ∗ µ k ,i +1 -V k µ k ,i +1 )( s ′ ) } . Hence, conditioned on the true MDP M ∗ and the sampled MDP M k , the term ∑ τ i =1 d t k + i has expectation zero.

## 5.2 Introducing confidence sets

The last section reduced the algorithm's regret to its expected Bellman error. We will proceed by arguing that the sampled Bellman operator T k µ k ( · ,i ) concentrates around the true Bellman operatior T ∗ µ k ( · ,i ) . To do this, we introduce high probability confidence sets similar to those used in [4] and [5]. Let ˆ P t a ( ·| s ) denote the emprical distribution up period t of transitions observed after sampling ( s, a ), and let ˆ R t a ( s ) denote the empirical average reward. Finally, define N t k ( s, a ) = ∑ t k -1 t =1 1 { ( s t ,a t )=( s,a ) } to be the number of times ( s, a ) was sampled prior to time t k . Define the confidence set for episode k :

<!-- formula-not-decoded -->

Where β k ( s, a ) := √ 14 S log(2 SAmt k ) max { 1 ,N t k ( s,a ) } is chosen conservatively so that M k contains both M ∗ and M k with high probability. It's worth pointing out that we have not tried to optimize this confidence bound, and it can be improved, at least by a numerical factor, with more careful analysis. Now, using that ˜ ∆ k ≤ τ we can decompose regret as follows:

<!-- formula-not-decoded -->

(1, r =

5

1000)

0.4

S1

0.6

0.05

0.6

$2

0.35

0.05

0.6

$3

0.35

0.05

0.6

$4

0.35

0.05

1

Now, since M k is σ ( H t k )-measureable, by Lemma 1, E [ 1 { M k / ∈M k } | H t k ] = E [ 1 { M ∗ / ∈M k } | H t k ]. Lemma 17 of [4] shows 3 P ( M ∗ / ∈ M k ) ≤ 1 /m for this choice of β k ( s, a ), which implies

<!-- formula-not-decoded -->

We also have the worst-case bound ∑ m k =1 ˜ ∆ k ≤ T . In the technical appendix we go on to provide a worst case bound on min { τ ∑ m k =1 ∑ τ i =1 min { β k ( s t k + i , a t k + i ) , 1 } , T } of order τS √ AT log( SAT ), which completes our analysis.

## 6 Simulation results

We compare performance of PSRL to UCRL2 [4]: an optimistic algorithm with similar regret bounds. We use the standard example of RiverSwim [21], as well as several randomly generated MDPs. We provide results in both the episodic case, where the state is reset every τ = 20 steps, as well as the setting without episodic reset.

Figure 1: RiverSwim - continuous and dotted arrows represent the MDP under the actions 'right' and 'left'.

<!-- image -->

RiverSwim consists of six states arranged in a chain as shown in Figure 1. The agent begins at the far left state and at every time step has the choice to swim left or right. Swimming left (with the current) is always successful, but swimming right (against the current) often fails. The agent receives a small reward for reaching the leftmost state, but the optimal policy is to attempt to swim right and receive a much larger reward. This MDP is constructed so that efficient exploration is required in order to obtain the optimal policy. To generate the random MDPs, we sampled 10-state, 5-action environments according to the prior.

We express our prior in terms of Dirichlet and normal-gamma distributions over the transitions and rewards respectively. 4 In both environments we perform 20 Monte Carlo simulations and compute the total regret over 10,000 time steps. We implement UCRL2 with δ = 0 . 05 and optimize the algorithm to take account of finite episodes where appropriate. PSRL outperformed UCRL2 across every environment, as shown in Table 1. In Figure 2, we show regret through time across 50 Monte Carlo simulations to 100,000 time-steps in the RiverSwim environment: PSRL's outperformance is quite extreme.

3 Our confidence sets are equivalent to those of [4] when the parameter δ = 1 /m .

4 These priors are conjugate to the multinomial and normal distribution. We used the values α = 1 /S, µ = σ 2 = 1 and pseudocount n = 1 for a diffuse uniform prior.

0.6

0.35

0.4

$6

(0.6, r = 1)

Regret

14000.

12000

10000

8000

6000

4000

2000

• 2000%

•- PSRL Regret

- - UCRL2 Regret|

2

4

Time elapsed

1000

900

800

700

600 -

500

Table 1: Total regret in simulation. PSRL outperforms UCRL2 over different environments.

300

| Algorithm   | Random MDP τ -episodes   | Random MDP ∞ -horizo n   | RiverSwim τ -episodes   | RiverSwim ∞ -horizon   |
|-------------|--------------------------|--------------------------|-------------------------|------------------------|
| PSRL        | 1 . 04 × 10 4            | 7 . 30 × 10 3            | 6 . 88 × 10 1           | 1 . 06 × 10 2          |
| UCRL2       | 5 . 92 × 10 4            | 1 . 13 × 10 5            | 1 . 26 × 10 3           | 3 . 64 × 10 3          |

Time elapsed

## 6.1 Learning in MDPs without episodic resets

The majority of practical problems in reinforcement learning can be mapped to repeated episodic interactions for some length τ . Even in cases where there is no actual reset of episodes, one can show that PSRL's regret is bounded against all policies which work over horizon τ or less [6]. Any setting with discount factor α can be learned for τ ∝ (1 -α ) -1 .

One appealing feature of UCRL2 [4] and REGAL [5] is that they learn this optimal timeframe τ . Instead of computing a new policy after a fixed number of periods, they begin a new episode when the total visits to any state-action pair is doubled. We can apply this same rule for episodes to PSRL in the ∞ -horizon case, as shown in Figure 2. Using optimism with KL-divergence instead of L 1 balls has also shown improved performance over UCRL2 [22], but its regret remains orders of magnitude more than PSRL on RiverSwim .

<!-- image -->

- (a) PSRL outperforms UCRL2 by large margins (b) PSRL learns quickly despite misspecified prior

Figure 2: Simulated regret on the ∞ -horizon RiverSwim environment.

## 7 Conclusion

We establish posterior sampling for reinforcement learning not just as a heuristic, but as a provably efficient learning algorithm. We present ˜ O ( τS √ AT ) Bayesian regret bounds, which are some of the first for an algorithm not motivated by optimism and are close to state of the art for any reinforcement learning algorithm. These bounds hold in expectation irrespective of prior or model structure. PSRL is conceptually simple, computationally efficient and can easily incorporate prior knowledge. Compared to feasible optimistic algorithms we believe that PSRL is often more efficient statistically, simpler to implement and computationally cheaper. We demonstrate that PSRL performs well in simulation over several domains. We believe there is a strong case for the wider adoption of algorithms based upon posterior sampling in both theory and practice.

## Acknowledgments

Osband and Russo are supported by Stanford Graduate Fellowships courtesy of PACCAR inc., and Burt and Deedee McMurty, respectively. This work was supported in part by Award CMMI-0968707 from the National Science Foundation.

An25/1529

516/3

-PSRL Regret (ave)

- PSRL Regret (worst)

•- PSRL Regret (best)

## References

- [1] A. N. Burnetas and M. N. Katehakis. Optimal adaptive policies for markov decision processes. Mathematics of Operations Research , 22(1):222-255, 1997.
- [2] P. R. Kumar and P. Varaiya. Stochastic systems: estimation, identification and adaptive control . Prentice-Hall, Inc., 1986.
- [3] T.L. Lai and H. Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics , 6(1):4-22, 1985.
- [4] T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning. The Journal of Machine Learning Research , 99:1563-1600, 2010.
- [5] P. L. Bartlett and A. Tewari. Regal: A regularization based algorithm for reinforcement learning in weakly communicating mdps. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence , pages 35-42. AUAI Press, 2009.
- [6] R. I. Brafman and M. Tennenholtz. R-max-a general polynomial time algorithm for nearoptimal reinforcement learning. The Journal of Machine Learning Research , 3:213-231, 2003.
- [7] S. M. Kakade. On the sample complexity of reinforcement learning . PhD thesis, University of London, 2003.
- [8] M. Kearns and S. Singh. Near-optimal reinforcement learning in polynomial time. Machine Learning , 49(2-3):209-232, 2002.
- [9] W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3/4):285-294, 1933.
- [10] O. Chapelle and L. Li. An empirical evaluation of Thompson sampling. In Neural Information Processing Systems (NIPS) , 2011.
- [11] S.L. Scott. A modern Bayesian look at the multi-armed bandit. Applied Stochastic Models in Business and Industry , 26(6):639-658, 2010.
- [12] S. Agrawal and N. Goyal. Further optimal regret bounds for Thompson sampling. arXiv preprint arXiv:1209.3353 , 2012.
- [13] S. Agrawal and N. Goyal. Thompson sampling for contextual bandits with linear payoffs. arXiv preprint arXiv:1209.3352 , 2012.
- [14] E. Kauffmann, N. Korda, and R. Munos. Thompson sampling: an asymptotically optimal finite time analysis. In International Conference on Algorithmic Learning Theory , 2012.
- [15] D. Russo and B. Van Roy. Learning to optimize via posterior sampling. CoRR , abs/1301.2609, 2013.
- [16] M. Strens. A Bayesian framework for reinforcement learning. In Proceedings of the 17th International Conference on Machine Learning , pages 943-950, 2000.
- [17] J. Z. Kolter and A. Y. Ng. Near-Bayesian exploration in polynomial time. In Proceedings of the 26th Annual International Conference on Machine Learning , pages 513-520. ACM, 2009.
- [18] T. Wang, D. Lizotte, M. Bowling, and D. Schuurmans. Bayesian sparse sampling for on-line reward optimization. In Proceedings of the 22nd international conference on Machine learning , pages 956-963. ACM, 2005.
- [19] A. Guez, D. Silver, and P. Dayan. Efficient bayes-adaptive reinforcement learning using samplebased search. arXiv preprint arXiv:1205.3109 , 2012.
- [20] J. Asmuth and M. L. Littman. Approaching bayes-optimalilty using monte-carlo tree search. In Proc. 21st Int. Conf. Automat. Plan. Sched., Freiburg, Germany , 2011.
- [21] A. L. Strehl and M. L. Littman. An analysis of model-based interval estimation for markov decision processes. Journal of Computer and System Sciences , 74(8):1309-1331, 2008.
- [22] S. Filippi, O. Capp´ e, and A. Garivier. Optimism in reinforcement learning based on kullbackleibler divergence. CoRR , abs/1004.5229, 2010.

## A Relating Bayesian to frequentist regret

Let M be any family of MDPs with non-zero probability under the prior. Then, for any /epsilon1 &gt; 0, α &gt; 1 2 :

<!-- formula-not-decoded -->

This provides regret bounds even if M ∗ is not distributed according to f . As long as the true MDP is not impossible under the prior, we will have an asymptotic frequentist regret close to the theoretical lower bounds of in T -dependence of O ( √ T ).

Proof. We have for any /epsilon1 &gt; 0:

<!-- formula-not-decoded -->

Therefore via theorem (1), for any α &gt; 1 2 :

<!-- formula-not-decoded -->

## B Bounding the sum of confidence set widths

We are interested in bounding min { τ ∑ m k =1 ∑ τ i =1 min { β k s t k + i , a t k + i ) , 1 } , T } which we claim is O ( τS √ AT log( SAT ) for β k ( s, a ) := √ 14 S log(2 SAmt k ) max { 1 ,N t k ( s,a ) } .

Proof. In a manner similar to [4] we can say:

<!-- formula-not-decoded -->

Now, the consider the event ( s t , a t ) = ( s, a ) and ( N t k ( s, a ) ≤ τ ). This can happen fewer than 2 τ times per state action pair. Therefore, ∑ m k =1 ∑ τ i =1 1 ( N t k ( s, a ) ≤ τ ) ≤ 2 τSA .Now, suppose N t k ( s, a ) &gt; τ . Then for any t ∈ { t k , .., t k +1 -1 } , N t ( s, a ) + 1 ≤ N t k ( s, a ) + τ ≤ 2 N t k ( s, a ). Therefore:

<!-- formula-not-decoded -->

Note that since all rewards and transitions are absolutely constrained ∈ [0 , 1] our regret

<!-- formula-not-decoded -->

Which is our required result.