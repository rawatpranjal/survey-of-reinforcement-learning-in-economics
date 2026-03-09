## Model-Based Reinforcement Learning with Value-Targeted Regression

Alex Ayoub ∗ 1 , Zeyu Jia † 2 , Csaba Szepesv´ ari ‡ 1,6 , Mengdi Wang § 3,4,6 and Lin F. Yang ¶ 5

1

Department of Computing Science, University of Alberta 2 School of Mathematical Science, Peking University 3 Department of Electrical Engineering, Princeton University 4 Center for Statistics and Machine Learning, Princeton University 5 Department of Electrical and Computer Engineering, University of California, Los Angeles 6

DeepMind

June 2, 2020

## Abstract

This paper studies model-based reinforcement learning (RL) for regret minimization. We focus on finite-horizon episodic RL where the transition model P belongs to a known family of models P , a special case of which is when models in P take the form of linear mixtures: P θ = ∑ d i =1 θ i P i . We propose a model based RL algorithm that is based on optimism principle: In each episode, the set of models that are 'consistent' with the data collected is constructed. The criterion of consistency is based on the total squared error of that the model incurs on the task of predicting values as determined by the last value estimate along the transitions. The next value function is then chosen by solving the optimistic planning problem with the constructed set of models. We derive a bound on the regret, which, in the special case of linear mixtures, the regret bound takes the form ˜ O ( d √ H 3 T ), where H , T and d are the horizon, total number of steps and dimension of θ , respectively. In particular, this regret bound is independent of the total number of states or actions, and is close to a lower bound Ω( √ HdT ). For a general model family P , the regret bound is derived using the notion of the so-called Eluder dimension proposed by Russo &amp; Van Roy (2014).

## 1 Introduction

Reinforcement learning (RL) enables learning to control complex environments through trial and error. It is a core problem in artificial intelligence (Russel &amp; Norvig, 2003; Sutton &amp; Barto, 2018) and recent years has witnessed phenomenal empirical advances in various areas such as: games, robotics and science (e.g., Mnih et al., 2015; Silver et al., 2017; AlQuraishi, 2019; Arulkumaran et al., 2019). In online RL, an agent has to learn to act in an unknown environment 'from scratch', collect data as she acts, and adapt the policy to maximize the reward collected. An important problem is to design algorithms that provably achieve sublinear regret in a large class of environments. Regret minimization for RL has received considerable attention in recent years (e.g., Jaksch et al. 2010; Osband et al. 2014; Azar et al. 2017; Dann et al. 2017, 2018; Agrawal &amp;

∗ aayoub@ualberta.ca

† jiazy@pku.edu.cn

‡ szepesva@ualberta.ca

§ mengdiw@princeton.edu

¶ linyang@ee.ucla.edu

Jia 2017; Osband et al. 2017; Jin et al. 2018; Yang &amp; Wang 2019a; Jin et al. 2019). While most of these existing works focus on the tabular or linear-factored MDP, only a handful of prior efforts have studied RL with general model classes. In particular, in a pioneering paper Strens (2000) proposed to use posterior sampling, which was later analyzed in the Bayesian setting by Osband &amp; Van Roy (2014); Abbasi-Yadkori &amp; Szepesv´ ari (2015); Theocharous et al. (2017). The reader is referred to Section 5 for a discussion of these and other related works.

In this paper, we study episodic reinforcement learning in an environment where the unknown probability transition model is known to belong to a family of models, i.e., P ∈ P . The model family P is a general set of models, and it may be either finitely parametrized or nonparametric. In particular, our approach accommodates working with smoothly parameterized models (e.g., Abbasi-Yadkori &amp; Szepesv´ ari, 2015), and can find use in both robotics (Kober et al., 2013) and queueing systems (Kovalenko, 1968). An illuminating special case is the case of linear parametrization when elements of P take the form P θ = ∑ i θ i P i where P 1 , P 2 , . . . , P d are fixed, known basis models and θ = ( θ 1 , . . . , θ d ) are unknown, real-valued parameters. Model P θ can be viewed as a mixture model that aggregates a finite family of known basic dynamical models (Modi et al., 2019). As an important special case, linear mixture models include the linear-factor MDP model of Yang &amp; Wang (2019a), a model that allows the embedding of possible transition kernels into an appropriate space of finite matrices.

The main contribution of this paper is a model-based upper confidence RL algorithm where the main novelty is the criterion to select models that are deemed consistent with past data. As opposed to standard practice where the models are selected based on their ability to predict next states or raw observations there (cf. Jaksch et al. (2010); Yang &amp; Wang (2019a) or (Strens, 2000; Osband &amp; Van Roy, 2014; Abbasi-Yadkori &amp; Szepesv´ ari, 2015; Ouyang et al., 2017; Agrawal &amp; Jia, 2017) in a Bayesian setting), we propose to evaluate models based on their ability to predict the values at next states as computed using the last value function estimate produced by our algorithm. In effect, the algorithm aims to select models based on their ability to produce small losses in a value-targeted regression problem.

Value-targeted regression is attractive for multiple reasons: (i) First and foremost, value-targeted regression holds the promise that model learning will focus on task-relevant aspects of the transition dynamics and can ignore aspects of the dynamics that are not relevant for the task. This is important as the dynamics can be quite complicated and modelling irrelevant aspects of the dynamics can draw valuable resources away from modelling task-relevant aspects. (ii) A related advantage is that building faithful probability models with high-dimensional state variables (or observations) can be challenging. Value-targeted regression sets up model learning as a real-valued regression problem, which intuitively feels easier than either building a model with maximum likelihood or setting up a vector-valued regression problem to model next state probabilities. (iii) Value-targeted regression aims at directly what matters in terms of the model accuracy or regret. Specifically the objective used in value-targeted is obtained from an expression that upper bounds the regret, hence it is natural to expect that minimizing this will lead to a small regret.

In addition, our approach is attractive as the algorithm has a modular structure and this allows any advances on components (optimistic planning, improvements in designing confidence sets) to be directly translated into a decreased regret. One may also question whether value-targeted regression is going 'too far' in ignoring details of the dynamics. Principally, one may think that since the value function used in defining the regression targets is derived based on imperfect knowledge, the model may never be sufficiently refined in a way that allows the regret to be kept under control. Secondly, one may worry about that by ignoring the rich details of observations (in our simple model, the identity of the state), the approach advocated is ignoring information available in the data, which may slow down learning. To summarize, the main question, to which we seek an answer in this paper, is the following:

## Is value-targeted regression sufficient and efficient for model-based online RL?

Based on the theoretical and the experimental evidence that we provide in this paper, our conclusion is that the answer is 'yes'.

Firstly, the regret bounds we derive conclusively show that the despite the imperfection and non-stationarity of the value targets, the algorithm cannot get 'stuck' (i.e., it enjoys sublinear regret). Our results also

suggest that perhaps there is no performance degradation as compared to the performance of competing algorithms. We are careful here as this conclusion is based on comparing worst-case upper bounds, which cannot provide a definitive answer.

To complement the theoretical findings, our experiments also confirm that our algorithm is competitive. The experiments also allow us to conclude that it is value-targeted regression together with optimistic planning that is effective. In particular, if optimism is taken away (i.e., glyph[epsilon1] -greedy is applied for the purpose of providing sufficient exploration), value-targeted regression performs worse than using a canonical approach to estimate the model. Similarly, if value-targeted regression is taken away, optimism together with the canonical model-estimation approach is less sample-efficient.

This still leaves open the possibility that certain combinations of value-targeted regression and canonical model building can be more effective than value-targeted regression. In fact, given the vast number of possibilities, we find this to be a quite plausible hypothesis. We note in passing that our proofs can be adjusted to deal with adding simultaneous alternative targets for constraining the set of data-consistent models. However, sadly, our current theoretical tools are unable to exhibit the tradeoffs that one expects to see here.

It is interesting to note that, in an independent and concurrent work, value-targeted regression has also been suggested as the main model building tool of the MuZero algorithm. The authors of this algorithm empirically evaluated MuZero on a number of RL benchmarks, such as the 57 Atari 'games', the game of 'Go', chess and shogi (Schrittwieser et al., 2019). In these benchmarks, despite the fact that MuZero does not use optimistic exploration or any other 'smart' exploration technique, MuZero was found to be highly competitive with its state-of-the-art alternatives, which reinforces the conclusion that training models using value-targeted regression is indeed a good approach to build effective model-based RL algorithms. The good results of MuZero on these benchmark may seem to contradict our experimental findings that value-targeted regression is ineffective without an appropriate, 'smart' exploration component. However, there is no contradiction: Smart exploration may be optional in some environments; our experiments show that it is not optional on some environments. In short, for robust performance across a wide range of environments, smart exploration is necessary but smart exploration may be optional in some environments.

As to the organization of the rest of the paper, the next section (Section 2) introduces the formal problem definition. This is followed by the description of the new algorithm (Section 3) and the main theoretical results (Section 4). In particular, we first give a regret bound for the general case where the regret is expressed as a function of the 'richness' of the model class P . This analysis is based on the Eluder dimension of an appropriately defined function class and its metric entropy at an appropriate scale. It is worth noting that the regret bound does not depend on either the size of the state or the size of the action space. To illustrate the strength of this general technique, we specialize the regret bound for the case of linear mixture models, √

for which we prove that the expected cumulative regret is at most O ( d H 3 T ), where H is the episode length, d is the number of model parameters and T is the total number of steps that the RL algorithm interacts with its environment. To complement the upper bound, for the linear case we also provide a regret lower bound Ω( √ HdT ) by adapting a lower bound that has been derived earlier for tabular RL. After these results, we discuss the connection of our work to prior art (Section 5). This is followed by the presentation of our empirical results (Section 6), where, as it was alluded to earlier, the aim is to explore how the various parts of the algorithm interact with each other. Section 7 concludes the paper.

## 2 Problem Formulation

We study episodic Markov decision processes (MDPs, for short), described by a tuple M = ( S , A , P, r, H, s ◦ ). Here, S is the state space, A is the action space, P is the transition kernel, r is a reward function, H &gt; 0 is the episode length, or horizon, and s ◦ ∈ S is the initial state. In the online RL problem, the learning agent is given S , A , H and r but does not know P . 1 The agent interacts with its environment described by M in episodes. Each episode begins at state s ◦ and ends after the agent made H decisions. At state

1 Our results are easy to extend to the case when r is not known.

s ∈ S , the agent, after observing the state s , can choose an action a ∈ A . As a result, the immediate reward r ( s, a ) is incurred. Then the process transitions to a random next state s ′ ∈ S according to the transition law P ( ·| s, a ). 2

The agent's goal is to maximize the total expected reward received over time.

If P is known, the behavior that achieves this over any number of episodes can be described by applying a deterministic policy π . Such a policy is a mapping from S × [ H ] into A , where we use the convention that for a natural number n , [ n ] = { 1 , . . . , n } . Following the policy means that the agent upon encountering state s in stage h will choose action π ( s, h ). In what follows, we will use π h ( s ) as an alternate notation, as this makes some of the formulae more readable. We will also follow this convention when it comes to other functions whose domain is S × [ H ]. We will find it convenient to move the stage h into the index. In particular, for policies, we will also write π h ( s ) for π ( s, h ) but we will use the same convention for other similar objects, like the value function, defined next.

The value function V π : S × [ H ] → R of a policy π is defined via

<!-- formula-not-decoded -->

where the subscript π (which we will often suppress) signifies that the probabilities underlying the expectation are governed by π . An optimal policy π ∗ and the optimal value function V ∗ are defined to be a policy and the value function such that V π h ( s ) achieves the maximum among all possible policies for any s ∈ S and h ∈ [ H ]. As noted above, there is no loss of generality in restricting the search of optimal policies to deterministic policies.

In online RL, a good agent of course uses all past observations to come up with its decisions. The performance of such an agent is measured by its regret, which is the total reward the agent misses because they did not follow the optimal policy from the beginning. In particular, the total expected regret of an agent A across K episodes is given by

<!-- formula-not-decoded -->

where T = KH is the total number of time steps that the agent interacts with its environment, s k 1 = s ◦ is the initial state at the start of the k -th episode, and s 1 1 , a 1 1 , . . . , s k H , a k H , . . . , s K 1 , a K 1 , . . . , s K H , a K H are the T = KH state-action pairs in the order that they are encountered by the agent. The regret is sublinear of R ( T ) /T → 0 as T →∞ . For a fixed T , let R ∗ ( T ) denote the worst-case regret. As is well known, no matter the algorithm used, R ∗ ( T ), grows at least as fast as √ T (e.g., Jaksch et al., 2010).

In this paper, we aim to design a general model-based reinforcement learning algorithm, with a guaranteed sublinear regret, for any given family of transition models.

Assumption 1 (Known Transition Model Family) . The unknown transition model P belongs to a family of models P which is available to the learning agent. The elements of P are transition kernels mapping state-action pairs to signed distributions over S .

That we allow signed distributions increases the generality; this may be important when one is given a model class that can be compactly represented but only when it also includes non-probability kernels (see Pires &amp; Szepesv´ ari 2016 for a discussion of this).

An important special case is the class of linear mixture models:

Definition 1 (Linear Mixture Models) . We say that P is the class of linear mixture models with component models P 1 , . . . , P d if P 1 , . . . , P d are transition kernels that map state-action pairs to signed measures and

2 The precise definitions require measure-theoretic concepts (Bertsekas &amp; Shreve, 1978), i.e., P is a Markov kernel, mapping from S × A to distributions over S , hence, all these spaces need to be properly equipped with a measurability structure. For the sake of readability and also because they are well understood, we omit these technical details.

P ∈ P if and only if there exists θ ∈ R d such that

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A .

Parametric and nonparametric transition models are common in modelling complex stochastic controlled systems. For one example, robotic systems are often smoothly parameterized by unknown mechanical parameters such as friction, or just parameters that describe the geometry of the robot.

The linear mixture model can be viewed as a way of aggregating a number of known basis models as considered by Modi et al. (2019). We can view each P j ( ·|· ) as a basis latent 'mode' and the actual transition is a probabilistic mixture of these latent modes. For another example, consider large-scale queueing networks where the arrival rate and job processing speed for each queue is not known. By using a discrete-time Bernoulli approximation, the transition probability matrix from time t to t +∆ t becomes increasingly close to linear with respect to the unknown arrival/processing rates as ∆ t → 0. In this case, it is common to model the discrete-time state transition as a linear aggregation of arrival/processing processes with unknown parameters Kovalenko (1968).

Another interesting special case is the linear-factored MDP model of Yang &amp; Wang (2019a) where, assuming a discrete state space for a moment, P takes the form

<!-- formula-not-decoded -->

where φ ( s, a ) ∈ R d 1 , ψ ( s ′ ) ∈ R d 2 are given features for every s, s ′ ∈ S and a ∈ A (when the state space is continuous, ψ becomes an R d 2 -valued measure over S ). The matrix M ∈ R d 1 × d 2 is an unknown matrix and is to be learned. It is easy to see that the factored MDP model is a special case of the linear mixture model (2) with each ψ j ( s ′ ) φ i ( s, a ) being a basis model (this should be replaced by ψ j ( ds ′ ) φ i ( s, a ) when the state space is continuous). In this case, the number of unknown parameters in the transition model is d = d 1 × d 2 . In this setting, without any additional assumption, our regret bound matches the result of Yang &amp; Wang (2019a).

## 3 Upper Confidence RL with Value-Targeted Model Regression

Our algorithm can be viewed as a generalization of UCRL (Jaksch et al., 2010), following ideas of Osband &amp; Van Roy (2014).

In particular, at the beginning of episode k = 1 , 2 , . . . , K , the algorithm first computes a subset B k of the model class P that contains the set of models that are deemed to be consistent with all the data that has been collected in the past. The new idea, value-targeted regression is used in the construction of B k . The details of how this is done are postponed to a later section.

Next, the algorithm needs to find the model that maximizes the optimal value, and the corresponding optimal policy. Denoting by V ∗ P the optimal value function under a model P , this amounts to finding the model P ∈ B k that maximizes the value V ∗ P, 1 ( s k 1 ). Given the model P k that maximizes this value, an optimal policy is extracted from the model as described in the next section (this is standard dynamic programming). At the end of the episode, the data collected is used to refine the confidence set B k . The pseudocode of the algorithm can be found in Algorithm 1.

## 3.1 Model-Based Optimistic Planning

Upper confidence methods are prominent in online learning. In our algorithm, we will maintain a confidence set B k for the estimated transition model and use it for optimistic planning:

<!-- formula-not-decoded -->

## Algorithm 1 UCRL-VTR

- 1: Input: Family of MDP models P , d, H, T = KH ;
- 2: Initialize: pick the sequence { β k } as in Eq. (6) of Theorem 1
- 3: B 1 = P
- 4: for k = 1 , 2 , . . . , K do
- 5: Observe the initial state s k 1 of episode k
- 6: Optimistic planning:

<!-- formula-not-decoded -->

- 7: for h = 1 , 2 , . . . , H do
- 8: Choose the next action greedily with respect to Q h,k :

<!-- formula-not-decoded -->

- 9: Observe state s k h +1
- 10: Compute and store value predictions: y h,k ← V h +1 ,k ( s k h +1 )
- 11: end for
- 12: Construct confidence set using value-targeted regression as described in Section 3.2 :

<!-- formula-not-decoded -->

## 13: end for

where s k 1 is the initial state at the beginning of episode k and we use V ∗ P ′ , 1 to denote the optimal value function of stage one, when the transition model is P ′ . Given model P k , the optimal policy for P k can be computed using dynamic programming. In particular, for 1 ≤ h ≤ H +1, define

<!-- formula-not-decoded -->

where we with a measure µ and function f over the same domain, we use 〈 µ, f 〉 to denote the integral of f with respect to µ . Then, taking the action at stage h and state s that maximizes Q h,k ( s, · ) gives an optimal policy for model P k . As long as P ∈ B k with high probability, the preceding calculation gives an optimistic (that is, upper) estimate of value of an episode. Next we show how to construct the confidence set B k .

## 3.2 Value-Targeted Regression for Confidence Set Construction

Every time we observe a transition ( s, a, s ′ ) with s ′ ∼ P ( ·| s, a ), we receive information about the model P . Instead of regression onto fixed target like probabilities or raw states, we will refresh the model estimate by regression using the estimated value functions as target.

This leads to the model

<!-- formula-not-decoded -->

In the above regression procedure, the regret target keeps changing as the algorithm constructs increasingly accurate value estimates. This is in contrast to typical supervised learning for building models, where the

regression targets are often fixed objects (such as raw observations, features or keypoints; e.g. Jaksch et al. (2010); Osband &amp; Van Roy (2014); Abbasi-Yadkori &amp; Szepesv´ ari (2015); Xie et al. (2016); Agrawal &amp; Jia (2017); Yang &amp; Wang (2019a); Kaiser et al. (2019)).

For a confidence set construction, we get inspiration from Proposition 5 in the paper of Osband &amp; Van Roy (2014). The set is centered at ˆ P k +1 .Define

<!-- formula-not-decoded -->

Then we let

<!-- formula-not-decoded -->

and the value of β k can be obtained using a calculation similar to that done in Proposition 5 of the paper of Osband &amp; Van Roy (2014), which is based on the nonlinear least-squares confidence set construction from Russo &amp; Van Roy (2014), which we describe in the appendix.

It is not hard to see that the confidence set can also be written in the alternative form

<!-- formula-not-decoded -->

with a suitably defined ˜ β k +1 and where

<!-- formula-not-decoded -->

Note that the above formulation strongly exploits that the MDP is time-homogeneous: The same transition model is used at all stages of an episode. When the MDP is time-inhomogeneous, the construction can be easily modified to accommodate that the transition kernel may depend on the stage index.

## 3.3 Implementation of UCRL-VTR

Algorithm 1 gives a general and modular template for model-based RL that is compatible with regression methods/optimistic planners. While the algorithms is conceptually simple, and the optimization and evaluation of the loss in value-targeted regression appears to be at advantage in terms of computation as compared to standard losses typically used in model-based RL, the implementation of UCRL-VTR is nontrivial in general and for now it requires a case-by-base design.

Computation efficiency of the algorithm depends on the specific family of models chosen. For the linear-factor MDP model considered by Yang &amp; Wang (2019a), the regression is linear and admits efficient implementation; further, optimistic planning for this model can be implemented in poly( d ) time by using Monte-Carlo simulation and sketching as argued in the cited paper. Other ideas include loosening the confidence set to come up with computationally tractable methods, or relaxing the requirement that the same model is used in all stages.

In the general case, optimistic planning is computationally intractable. However, we expect that randomized (eg Osband et al. (2017, 2014); Lu &amp; Van Roy (2017)) and approximate dynamic programming methods (tree search, roll out, see eg Bertsekas &amp; Tsitsiklis (1996)) will often lead to tractable and good approximations. As was mentioned above, in some special cases these have been rigorously shown to work. In similar settings, the approximation errors are known to mildly impact the regret Abbasi-Yadkori &amp; Szepesv´ ari (2015) and we expect the same will hold in our setting.

If we look beyond methods with rigorous guarantees, there are practical deep RL algorithms that implement parts of UCRL-VTR. As mentioned earlier, the Muzero algorithm of Schrittwieser et al. (2019) is a stateof-the-art algorithm on the Atari domain and this algorithm implements both value-targeted-regression to learn a model and Monte Carlo tree search for planning based on the learned model, although it does not incorporate optimistic planning.

## 4 Theoretical Analysis

We will need the concept of Eluder dimension. Let F be a set of real-valued functions with domain X . To measure the complexity of interactively identify an element of F , Russo &amp; Van Roy (2014) defines the Eluder dimension of F at scale glyph[epsilon1] &gt; 0. For f ∈ F , x 1 , . . . , x t ∈ X , introduce the notation f | ( x 1 ,...,x t ) = ( f ( x 1 ) , . . . , f ( x t )). We say that x ∈ X is glyph[epsilon1] -independent of x 1 , . . . , x t ∈ X given F if there exists f, f ′ ∈ F such that ‖ ( f -f ′ ) | ( x 1 ,...,x t ) ‖ 2 ≤ glyph[epsilon1] while f ( x ) -f ′ ( x ) &gt; glyph[epsilon1] .

Definition 2 (Eluder dimension Russo &amp; Van Roy (2014)) . The Eluder dimension dim E ( F , glyph[epsilon1] ) of F at scale glyph[epsilon1] is the length of the longest sequence ( x 1 , . . . , x n ) in X such that for some glyph[epsilon1] ′ ≥ glyph[epsilon1] , for any 2 ≤ t ≤ n , x t is glyph[epsilon1] ′ -independent of ( x 1 , . . . , x t -1 ) given F .

Let V be the set of optimal value functions under some model in P : V = { V ∗ P ′ : P ′ ∈ P} . Note that V ⊂ B ( S , H ), where B ( S , H ) denotes the set of real-valued measurable functions with domain S that are bounded by H . We let X = S × A × V . In order to analyze the confidence of nonlinear regression, choose F to be the collection of functions f : X → R as

<!-- formula-not-decoded -->

Note that F ⊂ B ( X , H ). For a norm ‖ · ‖ on F and α &gt; 0 let N ( F , α, ‖ · ‖ ) denote the ( α, ‖ · ‖ )-covering number of F . That is, this if m = N ( F , α, ‖ · ‖ ) then one can find m points of F such that any point in F is at most α away from one of these points in norm ‖ · ‖ . Denote by ‖ · ‖ ∞ the supremum norm: ‖ f ‖ ∞ = sup x ∈X | f ( x ) | .

Now we analyze the regret of UCRL-VTR. Define the K -episode pseudo-regret as

<!-- formula-not-decoded -->

Clearly, R ( KH ) = E R K holds for any K &gt; 0 where R ( T ) is the expected regret after T steps of interaction as defined in (1). Thus, to study the expected regret, it suffices to study R K .

Our main result is as follows.

Theorem 1 (Regret of Algorithm 1) . Let Assumption 1 hold and let α ∈ (0 , 1) . For k ∈ [ K ] , let β k be

<!-- formula-not-decoded -->

Then, with probability 1 -2 δ ,

<!-- formula-not-decoded -->

where d = dim E ( F , α ) is the Eluder dimension with F given by (5) .

A typical choice of α is α = 1 / ( KH ). In the special case of linear transition model, Theorem 1 implies a worst-case regret bound that depends linearly on the number of parameters.

Corollary 2 (Regret of Algorithm 1 for Linearly-Parametrized Transition Model) . Let P 1 , . . . , P d be d transition models, Θ ⊂ R d a nonempty set with diameter R measured in ‖·‖ 1 and let P = { ∑ j θ j P j : θ ∈ Θ } . Then, for any 0 &lt; δ &lt; 1 , with probability at least 1 -δ , the pseudo-regret R K of Algorithm 1 when it uses the confidence sets given in Theorem 1 satisfies

<!-- formula-not-decoded -->

We also provide a lower bound for the regret in our model. The proof is by reduction to a known lower bound and is left to Appendix B.

Theorem 3 (Regret Lower Bound) . For any H ≥ 1 and d ≥ 8 , there exist a state space S and action set A , a reward function r : S × A → [0 , 1] , d transition models P 1 , . . . , P d and a set Θ of diameter at most one such that for any algorithm there exists θ ∈ Θ such that for sufficiently large number of episodes K , the expected regret of the algorithm on the H -horizon MDP with reward r and transition model P = ∑ j θ j P j is at least Ω( H √ dK ) .

√

Rusmevichientong &amp; Tsitsiklis (2010) gave a regret lower bound of Ω( d T ) for linearly parameterized bandit with actions on the unit sphere (see also Section 24.2 of Lattimore &amp; Szepesv´ ari (2020)). Our regret upper bound matches this bandit lower bound in d, T . Whether the upper or lower bound is tight (or none of them) remains to be seen.

The theorems validate that, in the setting we consider, it is sufficient to use the predicted value functions as regression targets. That for the special case of linear mixture models the lower bound is close to the upper bound appears to suggest that little benefit if any can be derived from fitting the transition model to predict well future observations. We conjecture that this is in fact true when considering the worst-case regret. Of course, a conclusion that is concerned with the worst-case regret has no implication for the behavior of the respective methods on particular MDP instances.

We note in passing that by appropriately increasing β k , the regret upper bounds can be extended to the so-called misspecified case when P can be outside of P (for related results, see, e.g., Jin et al. 2019; Lattimore &amp; Szepesv´ ari 2019). However, the details of this are left for future work.

## 5 Related Work

A number of prior efforts have established efficient RL methods with provable regret bounds. For tabular MDPs with S states and A actions, building on the pioneering work of Jaksch et al. (2010) who studied the technically more challenging continuing setting, a number of works obtained results for the episodic setting, both with model-based (e.g., Osband et al. 2014; Azar et al. 2017; Dann et al. 2017, 2018; Agrawal &amp; Jia 2017), and model-free methods (e.g., Jin et al. 2018; Russo 2019; Zhang et al. 2020), both for the time-homogeneous case (i.e., the same transition kernel governs the dynamics in all stages of the H -horizon episode) and the time-inhomogeneous case. Results developed for the time-inhomogeneous case apply to the time-inhomogeneous case and since in this case the number of free parameters to learn is at least H times larger than for the time-homogeneous case, the regret bounds are expected to be √ H larger.

As far as regret lower bounds are concerned, Jaksch et al. (2010) established a worst-case regret lower bound of Ω( √ DSAT ) for the continuing case for MDPs with diameter bounded by D (see also Chapter 38 of Lattimore &amp; Szepesv´ ari 2020). This lower bound can be adapted to the episodic by setting D = H . This way one obtains a lower bound of Ω( √ HSAT ) for the homogeneous, and Ω( √ H 2 SAT ) for the inhomogeneous case (because here the state space size is effectively HS ). This lower bound, up to lower order terms, is matched by upper bounds both for the time-homogeneous case (Azar et al., 2017; Kakade et al., 2018) and the time inhomogeneous case (Dann et al., 2018; Zhang et al., 2020). Except the work of Zhang et al. (2020), these results are achieved by algorithms that estimate models. With a routine adjustment, the near-optimal model-based algorithms available for the homogeneous case are also expected to deliver near-optimal worstcase regret growth in the inhomogeneous case. A further variation is obtained by considering different scalings of the reward (Wang et al., 2020a).

Moving beyond tabular MDP, there have been significant theoretical and empirical advances on RL with function approximation, including but not limited to Baird (1995); Tsitsiklis &amp; Van Roy (1997); Parr et al. (2008); Mnih et al. (2013, 2015); Silver et al. (2017); Yang &amp; Wang (2019b); Bradtke &amp; Barto (1996). Among these works, many papers aim to uncover algorithms that are provably efficient. Under the assumption that the optimal action-value function is captured by linear features, Zanette et al. (2019) considers the case when the features are 'extrapolation friendly' and a simulation oracle is available, Wen &amp; Van Roy (2013, 2017) tackle problems where the transition model is deterministic, Du et al. (2019) deals with a relaxation of the deterministic case when the transition model has low variance. Yang &amp; Wang (2019b) considers the case of linear factor models, while Lattimore &amp; Szepesv´ ari (2019) considers the case when all the action-value

functions of all deterministic policies are well-approximated using a linear function approximator. These latter works handle problems when the algorithm has access to a simulation oracle of the MDP. As for regret minimization in RL using linear function approximation, Yang &amp; Wang (2019a) assumed the transition model admits a matrix embedding of the form P ( s ′ | s, a ) = φ ( s, a ) glyph[latticetop] Mψ ( s ′ ), and proposed a model-based MatrixRL method with regret bounds ˜ O ( H 2 d √ T ) with stronger assumptions and ˜ O ( H 2 d 2 √ T ) in general, where d is the dimension of state representation φ ( s, a ).

Jin et al. (2019) studied the setting of linear MDPs and constructed a model-free least-squares action-value iteration algorithm, which was proved to achieve the regret bound ˜ O ( √ H 3 d 3 T ). (Modi et al., 2019) considered a related setting where the transition model is an ensemble involving state-action-dependent features and basis models and proved a sample complexity d 3 K 2 H 2 glyph[epsilon1] 2 where d is the feature dimension, K is the number of basis models and d · K is their total model complexity. Very recently, Wang et al. (2020b) propose an model-free algorithm for general reward function approximation and show that the learning complexity of the function class can be bounded by the eluder dimension, which is similar to our model-based setting.

As for RL with a general model class, the seminal paper Osband &amp; Van Roy (2014) provided a general posterior sampling RL method that works for any given classes of reward and transition functions. It established a Bayesian regret upper bound O ( √ d K d E T ), where d K and d E are the Kolmogorov and the Eluder dimensions of the model class. In the case of linearly parametrized transition model (Assumption 2 of this paper), this Bayesian regret becomes O ( d √ T ), and our worst-case regret result matches with the Bayesian one. Abbasi-Yadkori &amp; Szepesv´ ari (2015); Theocharous et al. (2017) also considered the Bayesian regret and in particular Abbasi-Yadkori &amp; Szepesv´ ari (2015) considered a smooth parameterization with a somewhat unusual definition of smoothness. To the authors' best knowledge, there are no prior works addressing the problem of designing low-regret algorithms for MDPs with a general model family. In particular, while Osband &amp; Van Roy (2014) sketch the main ideas of an optimistic model-based optimistic algorithm for a general model class, they left out the details. When the details are filled based on their approach for the Bayesian case, unlike in the present work, the confidence sets would be constructed by losses that measure how well the model predict future observations and not by the value-targeted regression loss studied here. A preliminary version of the present paper appeared at L4DC 2020, which included results for the linear transition model only.

## 6 Numerical Experiments

The goal of our experiments is to provide insight into the benefits and/or pitfalls of using value-targets for fitting models, both with and without optimistic planning. We run our experiments in the tabular setting as in this setup it is easy to keep all the aspects of the test environments under control and the tabular setting also lets us avoid approximate computations. Note that tabular environments are a special case of the linear model where P j ( s ′ | s, a ) = I ( j = f ( s, a, s ′ )), where j ∈ [ S 2 A ] and f is a bijection that maps its arguments to the set [ S 2 A ]. Thus, d = S 2 A in this case.

The algorithms that we compare have a model-fitting objective which is either used to fit a nominal model or to calculate confidence sets. The objective is either to minimize mean-squared error of predicting next states (alternatively, maximize log-likelihood of observed data), which leads to standard frequency based model estimates, or it is based on minimizing the value targets as proposed in our paper. The other component of the algorithms is whether they implement optimistic planning, or planning with the nominal model and then implementing an glyph[epsilon1] -greedy policy with respect to the estimated model ('dithering'). We also consider mixing value targets and next state targets. In the case of optimistic planning, the algorithm that uses mixed targets uses a union bound and takes the smallest value upper confidence bounds amongst the two bounds obtained with the two model-estimation methods. These leads to six algorithms, as shown in Table 1. Results for the 'mixed' variants are very similar to the variant that uses value-targeted regression. As such, the results for the 'mixed' variants are shown in the appendix only, as they would otherwise make the graphs overly cluttered.

In the experiments we use confidence bounds that are specialized to the linear case. For the details, see Appendix C. For glyph[epsilon1] -greedy, we optimize the value of glyph[epsilon1] in each environment to get the best results. This gives

Table 1: Legend to the algorithms compared. Note that UC-MatrixRL of Yang &amp; Wang (2019a) in the tabular case essentially becomes UCRL of Jaksch et al. (2010). The mixed targets use both targets.

| Exploration/ Targets   | Optimism    | Dithering   |
|------------------------|-------------|-------------|
| Next states            | UC-MatrixRL | EG-Freq     |
| Values                 | UCRL-VTR    | EG-VTR      |
| Mixed                  | UCRL-Mixed  | EG-Mixed    |

glyph[epsilon1] -greedy an unfair advantage; but as we shall see despite this advantage, glyph[epsilon1] -greedy will not fair particularly well in our experiments.

## 6.1 Environments

We compare these algorithms on the episodic RiverSwim environment due to Strehl &amp; Littman (2008) and a novel finite horizon MDP we tentatively call WideTree. The RiverSwim environment, whose detailed description is given below in Section 6.3, is chosen because it is known that in this environment 'dithering' type exploration methods (e.g., glyph[epsilon1] -greedy) are ineffective. We vary the number of states in the RiverSwim environment in order to highlight some of the advantages and disadvantages of Value-Targeted Regression.

WideTree is designed in order it highlight the advantages of Value-Targeted Regression when compared with more tradition frequency based methods. In this environment, only one action effects the outcome thus the other actions are non-informative. The detailed description of WideTree is given in Section 6.4.

## 6.2 Measurements

We report the cumulative regret as a function of the number of episodes and the weighted model error to indicate how well the model is learned. The results are obtained from 30 independent runs for the glyph[epsilon1] -greedy algorithms and 10 independent runs for the UC algorithms, The weighted model error reported is as follows. Given the model estimate ˆ P , its weighted error is

<!-- formula-not-decoded -->

where N ( s, a ) is the observation-count of the state-action pair ( s, a ), N ( s, a, s ′ ) is the count of transitioning to s ′ from ( s, a ), and P ∗ ( s ′ | s, a ) is the probability of s ′ when action a is chosen in state s , according to the true model. Here, for the algorithms that use value-targeted regression the estimated model ˆ P is the model obtained through Eq. (4). The weighting is introduced so that an algorithm that discards a state-action pair is not penalized. This is meant to prevent penalizing good exploration algorithms that may quickly discard some state-action pairs. We are interested in this error metric to monitor whether UCLR-VTR, which is not forced to model next-state distributions, will learn the proper next state distribution. In fact, we will see one example both for the case when this and also when this does not happen.

## 6.3 Results for RiverSwim

The schematic diagram of the RiverSwim environment is shown in Figure 1. RiverSwim consists of S states arranged in a chain. The agent begins on the far left and has the choice of swimming left or right at each state. There is a current that makes swimming left much easier than swimming right. Swimming left with the current always succeeds in moving the agent left, but swimming right against the current sometimes moves the agent right (with a small probability of moving left as well), but more often than not leaves the agent in the current state. Thus smart exploration is a necessity to learn a good policy in this environment. We experiment with small environments with S ∈ { 3 , 4 , 5 } states and set the horizon to 4 S for each case.

Figure 1: The 'RiverSwim' environment with 6 states. State s 1 has a small associated reward, state s 6 has a large associated reward. The action whose effect is shown with the dashed arrow deterministically 'moves the agent' towards state s 1 . The other action is stochastic, and with relatively high probability moves the agent towards state s 6 : This represents swimming 'against the current'. None of these actions incur a reward.

<!-- image -->

The optimal values of the initial state are 5 . 72, 5 . 66 and 5 . 6, respectively, in these cases. The initial state is the leftmost state ( s 1 in the diagram). The value that we found to work the best for glyph[epsilon1] greedy is glyph[epsilon1] = 0 . 01.

Results are shown in Figure 2, except for UCRL-Mixed and EG-Mixed, whose results are given in Appendix D. As noted before, the results of these algorithms are very close to those of the VTR-versions, hence, they are not included here. The columns correspond to environments with S = 3, S = 4 and S = 5, respectively, which are increasingly more challenging. The first row shows the algorithm's performance measured in terms of their respective cumulative regrets, the second row shows results for the weighted model error as defined above. The regret per episode for an algorithm that 'does not learn' is expected to be in the same range as the respective optimal values. Based on this we see that 10 5 episodes is barely sufficient for the algorithms other than UCRL-VTR to learn a good policy. Looking at the model errors we see that EGRL-VTR is doing quite poorly, EG-Freq is also lacking (especially on the environment with 5 states), the others are doing reasonably well. That EG-Freq is not doing well is perhaps surprising. However, this is because EG-Freq visits more uniformly than the other methods the various state-action pairs.

The results clearly indicate that (i) fitting to the state-value function alone provides enough of a signal for learning as evident by UCRL-VTR obtaining low regret as predicted by our theoretical results, and that (ii) optimism is necessary when using value targeted regression to achieve good results, as evident by UCRL-VTR achieving significantly better regret than EGRL-VTR and even in the smaller RiverSwim environment where EG-Freq performed best.

It is also promising that value-targeted regression with optimistic exploration outperformed optimism based on the 'canonical' model estimation procedure. We attribute this to the fact that value-targeted regression will learn a model faster that predicts the optimal values well than the canonical, frequency based approach.

That value-targeted regression also learns a model with small weighted error appears to be an accidental feature of this environment. Our next experiments are targeted at further exploring this effect.

## 6.4 WideTree

We introduce a novel tabular MDP we call WideTree. The WideTree environment has a fixed horizon H = 2 but can vary in the number of states. A visualization of an eleven state WideTree environment is shown in Figure 3.

R

:RL

R

764

3.0 105

2.5

2.0

1.5

1.0

0.5

0.0

Cumulative Regret for a 4 state RiverSwim

2

3.0 105

2.5

2.0

1.5

1.0

0.5

Figure 2: The results for the glyph[epsilon1] -greedy algorithms were averaged over thirty runs and the results for the UC algorithms were averaged over ten runs. Error bars are only reported for the regret plots.

<!-- image -->

Figure 3: An eleven state WideTree MDP. The algorithm starts in the initial state s 1 . From the initial state s 1 the algorithm has a choice of either deterministically transitioning to either state s 2 or state s 3 . Finally from either state s 2 or state s 3 the algorithm picks one of two possible actions and transitions to one of the terminal states e i . Depending on which state the algorithm transitioned to from the initial state s 1 , determines which delayed reward the algorithm will observe. The delayed reward is observed at the final stage h = 2 of this MDP.

<!-- image -->

In WideTree, an agent starts at the initial state s 1 . The agent then progresses to one of the many bottom terminal states and collects a reward of either 0 or 1 depending on the action selected at state s 1 . The only significant action is whether to transition from s 1 to either s 2 or s 3 . Note that the model in the second layer is irrelevant for making a good decision: Once in s 3 , all actions lead to a reward of one, and once in s 2 , all actions lead to a reward of zero. We vary the number of bottom states reachable from states s 2 and s 3 while still maintaining a reward structure depending on whether the algorithm choose to transition to either s 2 or s 3 from the initial state s 1 .

We set glyph[epsilon1] = 0 . 1 in this environment, though choosing smaller glyph[epsilon1] but as long as glyph[epsilon1] &gt; 0 then both EGRL-VTR and EG-Freq will incur linear regret dependent on the choice of glyph[epsilon1] . One could also change the reward function in order to make learning for a given glyph[epsilon1] hard.

Cumulative Regret for a 5 state RiverSwim

Figure 4: As with the RiverSwim experiments, the results for the glyph[epsilon1] -greedy algorithms were averaged over thirty runs and the results for the UC algorithms were averaged over ten runs. Error bars are reported for the regret plots.

<!-- image -->

The results are shown in Figure 4, except for UCRL-Mixed and EG-Mixed, whose results are given in Appendix D. Both UCRL-VTR and EG-VTR learn equally poor models (their graphs are 'on the top of each other'). Yet, UCRL-VTR manages to quickly learn a good policy, as attested by its low regret.

EG-Freq and EG-VTR perform equally poorly and UC-MatrixRL is even slower as it keeps exploring the environment. These experiments clearly illustrate that UCRL-VTR is able to achieve good results without learning a good model - its focus on values makes pays off swiftly in this well-chosen environment.

## 7 Conclusions

We considered online learning in episodic MDPs and proposed an optimistic model-based reinforcement learning method (UCRL-VTR) with the unique characteristic to evaluate and select models based on their ability to predict value functions that the algorithm constructs during learning. The regret of the algorithm was shown to be bounded by a quantity that relates to the richness of the model class through the Eluder dimension and the metric entropy of an appropriately construction function space. For the case of linear mixture models, the regret bound simplifies to ˜ O ( H 3 / 2 d √ T ) where d is the number of model parameters, H is the horizon, and T is the total number of interaction steps. Our experiments confirmed that the value-targeted regression objective is not only theoretically sound, but also yields a competitive method which allows task-focused model-tuning: In a carefully chosen environment we demonstrated that the algorithm achieves low regret despite that it ignores modeling a major part of the environment.

## 8 Acknowledgements

Csaba Szepesv´ ari gratefully acknowledges funding from the Canada CIFAR AI Chairs Program, Amii and NSERC.

## References

Abbasi-Yadkori, Y. and Szepesv´ ari, C. Bayesian optimal control of smoothly parameterized systems. In UAI , pp. 1-11, 2015.

Abbasi-Yadkori, Y., P´ al, D., and Szepesv´ ari, C. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , pp. 2312-2320, 2011.

- Agrawal, S. and Jia, R. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Advances in Neural Information Processing Systems , pp. 1184-1194, 2017.
- AlQuraishi, M. AlphaFold at CASP13. Bioinformatics , 35(22):4862-4865, 2019.
- Arulkumaran, K., Cully, A., and Togelius, J. Alphastar: An evolutionary computation perspective. arXiv preprint arXiv:1902.01724 , 2019.
- Azar, M. G., Osband, I., and Munos, R. Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pp. 263-272. JMLR. org, 2017.
- Baird, L. Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995 , pp. 30-37. Elsevier, 1995.
- Bertsekas, D. P. and Shreve, S. Stochastic optimal control: the discrete-time case . Academic Press, 1978.
- Bertsekas, D. P. and Tsitsiklis, J. N. Neuro-dynamic programming . Athena Scientific, 1996.
- Bradtke, S. J. and Barto, A. G. Linear least-squares algorithms for temporal difference learning. Machine learning , 22(1-3):33-57, 1996.
- Dann, C., Lattimore, T., and Brunskill, E. Unifying PAC and regret: Uniform PAC bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , pp. 5713-5723, 2017.
- Dann, C., Li, L., Wei, W., and Brunskill, E. Policy certificates: Towards accountable reinforcement learning. arXiv preprint arXiv:1811.03056 , 2018.
- Du, S. S., Luo, Y., Wang, R., and Zhang, H. Provably efficient Q -learning with function approximation via distribution shift error checking oracle. arXiv preprint arXiv:1906.06321 , 2019.
- Jaksch, T., Ortner, R., and Auer, P. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600, 2010.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. Is Q -learning provably efficient? In Advances in Neural Information Processing Systems , pp. 4863-4873, 2018.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. Provably efficient reinforcement learning with linear function approximation. arXiv preprint arXiv:1907.05388 , 2019.
- Kaiser, L., Babaeizadeh, M., Milos, P., Osinski, B., Campbell, R. H., Czechowski, K., Erhan, D., Finn, C., Kozakowski, P., Levine, S., Mohiuddin, A., Sepassi, R., Tucker, G., and Michalewski, H. Model-based reinforcement learning for Atari. In ICLR , 2019.
- Kakade, S., Wang, M., and Yang, L. F. Variance reduction methods for sublinear reinforcement learning. arXiv preprint arXiv:1802.09184 , 2018.
- Kober, J., Bagnell, J. A., and Peters, J. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013.
- Kovalenko, B. G. I. N. Introduction to queueing theory . Israel Program for Scientific Translation, Jerusalem, 1968.
- Lattimore, T. and Szepesv´ ari, C. Learning with good feature representations in bandits and in RL with a generative model, 2019.
- Lattimore, T. and Szepesv´ ari, C. Bandit Algorithms . Cambridge University Press, 2020. (to appear).
- Lu, X. and Van Roy, B. Ensemble sampling. In Advances in neural information processing systems , pp. 3258-3266, 2017.

- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. Playing Atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.
- Modi, A., Jiang, N., Tewari, A., and Singh, S. Sample complexity of reinforcement learning using linearly combined model ensembles. arXiv preprint arXiv:1910.10597 , 2019.
- Osband, I. and Van Roy, B. Model-based reinforcement learning and the Eluder dimension. In Advances in Neural Information Processing Systems , pp. 1466-1474, 2014.
- Osband, I. and Van Roy, B. On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 , 2016.
- Osband, I., Van Roy, B., and Wen, Z. Generalization and exploration via randomized value functions. arXiv preprint arXiv:1402.0635 , 2014.
- Osband, I., Van Roy, B., Russo, D., and Wen, Z. Deep exploration via randomized value functions. arXiv preprint arXiv:1703.07608 , 2017.
- Ouyang, Y., Gagrani, M., Nayyar, A., and Jain, R. Learning unknown Markov decision processes: A Thompson sampling approach. In Advances in Neural Information Processing Systems , pp. 1333-1342, 2017.
- Parr, R., Li, L., Taylor, G., Painter-Wakefield, C., and Littman, M. L. An analysis of linear models, linear value-function approximation, and feature selection for reinforcement learning. In Proceedings of the 25th international conference on Machine learning , pp. 752-759. ACM, 2008.
- Pires, B. and Szepesv´ ari, C. Policy error bounds for model-based reinforcement learning with factored linear models. In COLT , pp. 121-151, 2016.
- Rusmevichientong, P. and Tsitsiklis, J. N. Linearly parameterized bandits. Mathematics of Operations Research , 35(2):395-411, 2010.
- Russel, S. and Norvig, P. Artificial Intelligence - a modern approach . Prentice Hall, 2003.
- Russo, D. Worst-case regret bounds for exploration via randomized value functions. In Advances in Neural Information Processing Systems , pp. 14410-14420, 2019.
- Russo, D. and Van Roy, B. Learning to optimize via posterior sampling. Mathematics of Operations Research , 39(4):1221-1243, 2014.
- Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., , Lillicrap, T., and Silver, D. Mastering Atari, Go, chess and shogi by planning with a learned model. arXiv preprint arXiv:1911.08265 , 2019.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., Driessche, G. v. d., Graepel, T., and Hassabis, D. Mastering the game of go without human knowledge. Nature , 550(7676):354, 2017.
- Strehl, A. and Littman, M. An analysis of model-based interval estimation for Markov decision processes. Journal of Computer and System Sciences , 74(8):1309-1331, 2008.
- Strens, M. J. A. A Bayesian framework for reinforcement learning. In ICML , pp. 943-950, 2000.
- Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction . The MIT Press, 2 edition, 2018.

- Theocharous, G., Wen, Z., Abbasi-Yadkori, Y., and Vlassis, N. Posterior sampling for large scale reinforcement learning. arXiv preprint arXiv:1711.07979 , 2017.
- Tsitsiklis, J. N. and Van Roy, B. Analysis of temporal-diffference learning with function approximation. In Advances in neural information processing systems , pp. 1075-1081, 1997.
- Wang, R., Du, S. S., Yang, L. F., and Kakade, S. M. Is long horizon reinforcement learning more difficult than short horizon reinforcement learning? arXiv preprint arXiv:2005.00527 , 2020a.
- Wang, R., Salakhutdinov, R., and Yang, L. F. Provably efficient reinforcement learning with general value function approximation. arXiv preprint arXiv:2005.10804 , 2020b.
- Wen, Z. and Van Roy, B. Efficient exploration and value function generalization in deterministic systems. In Advances in Neural Information Processing Systems , pp. 3021-3029, 2013.
- Wen, Z. and Van Roy, B. Efficient reinforcement learning in deterministic systems with value function generalization. Mathematics of Operations Research , 42(3):762-782, 2017.
- Xie, C., Patil, S., Moldovan, T., Levine, S., and Abbeel, P. Model-based reinforcement learning with parametrized physical models and optimism-driven exploration. In 2016 IEEE International Conference on Robotics and Automation (ICRA) , pp. 504-511. IEEE, 2016.
- Yang, L. F. and Wang, M. Reinforcement leaning in feature space: Matrix bandit, kernels, and regret bound. arXiv preprint arXiv:1905.10389 , 2019a.
- Yang, L. F. and Wang, M. Sample-optimal parametric Q -learning with linear transition models. International Conference on Machine Learning , 2019b.
- Zanette, A., Lazaric, A., Kochenderfer, M. J., and Brunskill, E. Limiting extrapolation in linear approximate value iteration. In Advances in Neural Information Processing Systems 32 , pp. 5616-5625. Curran Associates, Inc., 2019.
- Zhang, Z., Zhou, Y., and Ji, X. Almost optimal model-free reinforcement learning via reference-advantage decomposition. arXiv preprint arXiv:2004.10019 , April 2020.

## A Proof of Theorem 1

In this section, we provide the regret analysis of the UCRL-VTR Algorithm (Algorithm 1). We will explain the motivation for our construction of confidence sets for general nonlinear squared estimation, and establish the regret bound for a general class of transition models, P .

## A.1 Preliminaries

Recall that a finite horizon MDP is M = ( S , A , P, r, H, s ◦ ) where S is the state space, A is the action space, P = ( P a ) a ∈A is a collection of P a : S → M 1 ( S ) Markov kernels, r : S × A → [0 , 1] is the reward function, H &gt; 0 is the horizon and s ◦ ∈ S is the initial state. For a state s ∈ S and an action a ∈ A , P a ( s ) gives the distribution of the next state that is obtained when action a is executed in state s . For a bounded (measurable) function V : S → R , we will use 〈 P a ( s ) , V 〉 as the shorthand for the expected value of V at a random next state s ′ whose distribution is P a ( s ).

Given any policy π (which may or may not use the history), its value function is

<!-- formula-not-decoded -->

where E π,δ s is the expectation operator underlying the probability measure P π,δ s induced over sequences of state-action pairs of length H by executing policy π starting at state s in the MDP M and s h is the state visited in stage h and action a h is the action taken in that stage after visiting s h . For a nonstationary Markov policy π = ( π 1 , . . . , π H ), we also let

<!-- formula-not-decoded -->

be the value function of π from stage h to H . Here, π h : H denotes the policy ( π h , . . . , π H ). The optimal value function V ∗ = ( V ∗ 1 , . . . , V ∗ H ) is defined via V ∗ h ( s ) = max π V π h ( s ), s ∈ S .

For simplicity assume that r is known. To indicate the dependence of V ∗ on the transition model P , we will write V ∗ P = ( V ∗ P, 1 , . . . , V ∗ P,H ). For convenience, we define V ∗ P,H +1 = 0.

Algorithm 1 is an instance of the following general model-based optimistic algorithm:

```
Algorithm 2 Generic Algorithm 1-Schema for finite horizon problems 1: Input: P - a set of transition models, K - number of episodes, s 0 - initial state 2: Set B 1 = P 3: for k = 1 , . . . , K do 4: P k = argmax { V ∗ ˜ P ( s 0 ) : ˜ P ∈ B k } 5: V k = V ∗ P k 6: s k 1 = s 0 7: for h = 1 , . . . , H do 8: Choose a k h = argmax a ∈A r ( s k h , a ) + 〈 P k a ( s k h ) , V h +1 ,k 〉 9: Observe transition to s k h +1 10: end for 11: Construct B k +1 based on ( s k 1 , a k 1 , . . . , s k H , a k H ) 12: end for
```

Specific instances of Algorithm 2 differ in terms of how B k +1 is constructed. In particular, UCRL-VTR uses the construction described in Section 3.2.

Recall that V k = ( V 1 ,k , . . . , V H,k , V H +1 ,k ) (with V H +1 ,k = 0) in Algorithm 2. Let π k be the nonstationary Markov policy chosen in episode k by Algorithm 2. Let

<!-- formula-not-decoded -->

be the pseudo-regret of Algorithm 1 for K episodes. The following standard lemma bounds the k th term of the expression on the right-hand side.

Lemma 4. Assuming that P ∈ B k , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that ( ξ 2 , 1 , ξ 3 , 1 , . . . , ξ H, 1 , ξ 2 , 2 , ξ 3 , 2 , . . . , ξ H, 2 , ξ 2 , 3 , . . . ) is a sequence of martingale differences.

Proof. Because P ∈ B k , V ∗ 1 ( s k 1 ) ≤ V 1 ,k ( s k 1 ) by the definition of the algorithm. Hence,

<!-- formula-not-decoded -->

Fix h ∈ [ H ]. In what follows we bound V h,k ( s k h ) -V π k h ( s k h ). By the definition of π k , P k and a k h , we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Therefore, by induction, noting that V H +1 ,k = 0, we get that

<!-- formula-not-decoded -->

## A.2 The confidence sets for Algorithm 1

The previous lemma suggests that at the end of the k th episode, the model could be estimated using

<!-- formula-not-decoded -->

For a confidence set construction, we get inspiration from Proposition 5 in the paper of Osband &amp; Van Roy (2014). The set is centered at ˆ P k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that this is the same confidence set as described in Section 3.2. To obtain the value of β k , we now consider the nonlinear least-squares confidence set construction from Russo &amp; Van Roy (2014). The next section is devoted to this construction.

## A.3 Confidence sets for general nonlinear least-squares

Let ( X p , Y p ) p =1 , 2 ,... be a sequence of random elements, X p ∈ X for some measurable set X and Y p ∈ R . Let F be a subset of the set of real-valued measurable functions with domain X . Let F = ( F p ) p =0 , 1 ,... be a filtration such that for all p ≥ 1, ( X 1 , Y 1 , . . . , X p -1 , Y p -1 , X p ) is F p -1 measurable and such that there exists some function f ∗ ∈ F such that E [ Y p | F p -1 ] = f ∗ ( X p ) holds for all p ≥ 1. The (nonlinear) least-squares predictor given ( X 1 , Y 1 , . . . , X t , Y t ) is ˆ f t = argmin f ∈F ∑ t p =1 ( f ( X p ) -Y p ) 2 . We say that Z is conditionally ρ -subgaussian given the σ -algebra F if for all λ ∈ R , log E [exp( λZ ) | F ] ≤ 1 2 λ 2 ρ 2 . For α &gt; 0, let N α be the ‖ · ‖ ∞ -covering number of F at scale α . That is, N α is the smallest integer for which there exist G ⊂ F with N α elements such that for any f ∈ F , min g ∈G ‖ f -g ‖ ∞ ≤ α . For β &gt; 0, define

<!-- formula-not-decoded -->

We have the following theorem, the proof of which is given in Section A.6.

Theorem 5. Let F be the filtration defined above and assume that the functions in F are bounded by the positive constant C &gt; 0 . Assume that for each s ≥ 1 , ( Y p -f ∗ ( X p )) p is conditionally σ -subgaussian given F p -1 . Then, for any α &gt; 0 , with probability 1 -δ , for all t ≥ 1 , f ∗ ∈ F t ( β t ( δ, α )) , where

<!-- formula-not-decoded -->

The proof follows that of Proposition 6, Russo &amp; Van Roy (2014), with minor improvements, which lead to a slightly better bound. In particular, with our notation, Russo &amp; Van Roy stated their result with

<!-- formula-not-decoded -->

While β t ( δ, α ) ≤ β RvR t ( δ, α ), the improvement is only in terms of smaller constants.

## A.4 The choice of β k in Algorithm 1

To use this result in our RL problem recall that P is the set of transition probabilities parameterized by θ ∈ Θ. We index time t = 1 , 2 , . . . in a continuous fashion. Episode k = 1 , 2 , . . . and stage h = 1 , . . . , H -1 corresponds to time t = ( k -1)( H -1) + h :

| episode ( k )   |   1 |   1 | . . .   | 1     | 2   | 2    | . . .   | 2       | 3       | . . .   |
|-----------------|-----|-----|---------|-------|-----|------|---------|---------|---------|---------|
| stage ( h )     |   1 |   2 | . . .   | H - 1 | 1   | 2    | . . .   | H - 1   | 1       | . . .   |
| time step ( t ) |   1 |   2 | . . .   | H - 1 | H   | H +1 | . . .   | 2 H - 2 | 2 H - 1 | . . .   |

where

Note that the transitions at stage h = H are skipped and the time index at the end of episode k ≥ 1 is k ( H -1).

Let V ( t ) be the value function used by Algorithm 1 at time t ( V ( t ) is constant in periods of length H -1), while let ( s ( t ) , a ( t ) ) be the state-action pair visited at time t .

Let V be the set of optimal value functions under some model in P : V = { V ∗ P ′ : P ′ ∈ P} . Note that V ⊂ B ( S , H ), where B ( S , H ) denotes the set of real-valued measurable functions with domain S that are bounded by H . Note also that for all t , V ( t ) ∈ V . Define X = S × A × V . We also let X t = ( s ( t ) , a ( t ) , V ( t ) ), Y t = V ( t ) ( s ( t +1) ) when t +1 glyph[negationslash]∈ { H +1 , 2 H +1 , . . . } and Y t = V ( t ) ( s k H +1 ), and choose

<!-- formula-not-decoded -->

Note that F ⊂ B ∞ ( X , H ).

Let φ : P → F be the natural surjection to F : φ ( P ) = f where f ( s, a, v ) = ∫ P a ( ds ′ | s ) v ( s ′ ) for ( s, a, v ) ∈ X . We know show that φ is in fact a bijection. If P = P ′ , this means that for some ( s, a ) ∈ S × A and U ⊂ S measurable, P a ( U | s ) = P ′ a ( U | s ). Choosing v to be the indicator of U , note that ( s, a, v ) ∈ X . Hence, φ ( P )( s, a, v ) = P a ( U | s ) = P ′ a ( U | s ) = φ ( P ′ )( s, a, v ), and hence φ ( P ) = φ ( P ′ ): φ is indeed a bijection. For convenience and to reduce clutter, we will write f P = φ ( P ).

glyph[negationslash]

glyph[negationslash]

Choose F = ( F t ) t ≥ 0 so that F t -1 is generated by ( s (1) , a (1) , V (1) , . . . , s ( t ) , a ( t ) , V ( t ) ). Then E [ Y t | F t -1 ] = ∫ P a ( t ) ( ds ′ | s ( t ) ) V ( t ) ( s ′ ) = f P ( X t ) and by definition f P ∈ F . Now, Y t ∈ [0 , H ], hence, Z t = Y t -f P ( X t ) is conditionally H/ 2-subgaussian given F t -1 .

Let t = k ( H -1) for some k ≥ 1. Thus, this time step corresponds to finishing episode k and thus V ( t ) = V k . Furthermore, letting ˆ f t = argmin f ∈F ∑ t p =1 ( f ( X p ) -Y p ) 2 , since φ is an injection, we see that ˆ f t = f ˆ P k where ˆ P k is defined using (8). For P ′ , P ′′ ∈ P , we have L k ( P ′ , P ′′ ) = ∑ t p =1 ( f P ′ ( X p ) -f P ′′ ( X p )) 2 and thus

<!-- formula-not-decoded -->

Corollary 6. For α &gt; 0 and k ≥ 1 let

<!-- formula-not-decoded -->

Then, with probability 1 -δ , for any k ≥ 1 , P ∈ B k where B k is defined by (9) .

## A.5 Regret of Algorithm 1

Recall that X = S×A×V where V ⊂ B ∞ ( S , H ) is the set of value functions that are optimal under some model in P . We will abbreviate ( x 1 , . . . , x t ) ∈ X t as x 1: t . Further, we let F| x 1: t = { ( f ( x 1 ) , . . . , f ( x t )) : f ∈ F} ( ⊂ R t ) and for S ⊂ R t , let diam( S ) = sup u,v ∈ S ‖ u -v ‖ 2 be the diameter of S . We will need the following lemma, extracted from Russo &amp; Van Roy (2014):

Lemma 7 (Lemma 5 of Russo &amp; Van Roy (2014) ) . Let F ⊂ B ∞ ( X , C ) be a set of functions bounded by C &gt; 0 , ( F t ) t ≥ 1 and ( x t ) t ≥ 1 be sequences such that F t ⊂ F and x t ∈ X hold for t ≥ 1 . Then, for any T ≥ 1 and α &gt; 0 it holds that

<!-- formula-not-decoded -->

where δ T = max 1 ≤ t ≤ T diam( F t | x 1: t ) and d = dim E ( F , α ) .

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

Let

From Lemma 4, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 8. Let α &gt; 0 and d = dim E ( F , α ) where F is given by (10) . Then, for any nondecreasing sequence ( β 2 k ) K k =1 , on the event when P ∈ ∩ k ∈ [ K ] B k ,

<!-- formula-not-decoded -->

Proof. Let P ∈ ∩ k ∈ [ K ] B k holds. Using the notation of the previous section, letting ˜ F t = F t ( β k ) for ( k -1)( H -1) + 1 ≤ t ≤ k ( H -1), we have

<!-- formula-not-decoded -->

where X t is defined in Section A.4 and where the last inequality is by Lemma 7, which is applicable because F ⊂ B ∞ ( X , H ) holds by choice, and δ K ( H -1) = max 1 ≤ t ≤ K ( H -1) diam( ˜ F t | X 1: t ). Thanks to the definition of ˜ F t , δ K ( H -1) ≤ 2 √ β K . Plugging this into the previous display finishes the proof.

## A.5.1 Proof of Theorem 1

Proof. Note that for any k ∈ [ K ] and h ∈ [ H -1], ξ h +1 , k ∈ [ -H,H ]. As noted beforehand, ξ 2 , 1 , ξ 3 , 1 , . . . , ξ H, 1 , ξ 2 , 2 , ξ 3 , 2 , . . . , ξ H, 2 , ξ 2 , 3 , . . . is a martingale difference sequence. Thus, with probability 1 -δ , ∑ K k =1 ∑ H -1 h =1 ξ h +1 ,k ≤ H √ 2 K ( H -1) log(1 /δ ). Consider the event when this inequality holds and when P ∈ ∩ k ∈ [ K ] B k . By using Corollary 6 and a union bound, this event holds with probability at least 1 -2 δ . On this event, by (11) and Lemma 8, we obtain

<!-- formula-not-decoded -->

Using α ≤ 1, which holds by assumption, finishes the proof.

## A.5.2 Proof of Corollary 2

Proof. Note that

<!-- formula-not-decoded -->

For α &gt; 0 let N ( P , α, ‖ · ‖ ∞ , 1 ) denote the ( α, ‖ · ‖ ∞ , 1 )-covering number of P . Then we have

<!-- formula-not-decoded -->

Then, by Corollary 6,

<!-- formula-not-decoded -->

with some universal constant C &gt; 0. Let f : (Θ , ‖ · ‖ ) → ( P , ‖ · ‖ ∞ , 1 ) be defined by θ ↦→ ∑ j θ j P j . Note that ‖ f ( θ ) -f ( θ ′ ) ‖ ∞ , 1 ≤ sup s,a ∑ j ‖ ( θ j -θ ′ j ) P j,a ( s ) ‖ 1 = ∑ j | θ j -θ ′ j | = ‖ θ -θ ′ ‖ 1 . Hence, any ( glyph[epsilon1], ‖ · ‖ 1 ) covering of Θ induces an ( glyph[epsilon1], ‖ · ‖ ∞ , 1 )-covering of P and so N ( P , α/H, ‖ · ‖ ∞ , 1 ) ≤ N (Θ , α/H, ‖ · ‖ 1 ) ≤ C ′ ( RH/α ) d with some universal constant C ′ &gt; 0.

Now, choose 1 /α = K √ log( KH/δ ). Hence,

<!-- formula-not-decoded -->

Suppressing log factors (e.g., log( RH )), log log terms and constants, we have β K = H 2 ( d +log(1 /δ )).

Let F be given by (10). We now bound dim E ( F , α ). Let X = S × A × B ( S ) as before. Define z : S × A × B ( S ) → R d using z ( s, a, v ) j = 〈 P j,a ( s ) , v 〉 and note that if x ∈ X is ( glyph[epsilon1], F )-independent of x 1 , . . . , x k ∈ X then z ( x ) ∈ R d is ( glyph[epsilon1], Θ)-independent of z ( x 1 ) , . . . , z ( x k ) ∈ R d . This holds because if P = ∑ j θ j P j ∈ P then f P ( s, a, v ) = 〈 θ, z ( s, a, v ) 〉 for any ( s, a, v ) ∈ X . Hence, dim E ( F , α ) ≤ dim E (Lin( Z , Θ) , α ), where Lin( Z , Θ) is the set of linear maps with domain Z = { z ( x ) : x ∈ X} ⊂ R d and parameter from Θ: Lin( Z , Θ) = { h : h : Z → R s.t. ∃ θ ∈ Θ : h ( z ) = 〈 θ, z 〉 , z ∈ Z} . Now, by Proposition 11 of Russo &amp; Van Roy (2014), dim E (Lin( Z , Θ) , α ) = O ( d log(1 + ( Sγ/α ) 2 ) where S is the ‖ · ‖ 2 diameter of Θ and γ = sup z ∈Z ‖ z ‖ 2 . We have

<!-- formula-not-decoded -->

hence γ ≤ H √ d . By the relation between the 1 and 2 norms, the 2-norm diameter of Θ is at most √ dR . Dropping log terms, dim E ( F , α ) = ˜ O ( d ).

Plugging into Theorem 1 gives the desired result.

## A.6 Proof of Theorem 5

Recall the following:

Definition 3. A random variable X is σ -subgaussian if for all λ ∈ R , it holds that E [exp( λX )] ≤ exp ( λ 2 σ 2 / 2 ) .

The proof of the next couple of statements is standard and is included only for completeness.

Theorem 9. If X is σ -subgaussian, then for any λ &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. Let λ &gt; 0. We have, { X ≥ glyph[epsilon1] } = { exp( λ ( X -glyph[epsilon1] )) ≥ 0 } . Hence, Markov's inequality gives P ( X ≥ glyph[epsilon1] ) ≤ exp( -λglyph[epsilon1] ) E [exp( λX )] ≤ exp( -λglyph[epsilon1] + 1 2 λ 2 σ 2 ). Equating the right-hand side with δ and solving for glyph[epsilon1] , we get that log( δ ) = -λglyph[epsilon1] + 1 2 λ 2 σ 2 . Solving for glyph[epsilon1] gives glyph[epsilon1] = log(1 /δ ) /λ + σ 2 2 λ , finishing the proof.

Choosing the λ that minimizes the right-hand side of the bound gives the usual form:

<!-- formula-not-decoded -->

Lemma 10 (Lemma 5.4 of Lattimore &amp; Szepesv´ ari (2020)) . Suppose that X is σ -subgaussian and X 1 and X 2 are independent and σ 1 and σ 2 -subgaussian, respectively, then:

1. E [ X ] = 0 .

2. cX is | c | σ -subgaussian for all c ∈ R .

<!-- formula-not-decoded -->

Let ( Z p ) p be an F = ( F p ) p -adapted process. Recall that ( Z p ) p is conditionally σ -subgaussian given F if for all p ≥ 1,

<!-- formula-not-decoded -->

A standard calculation gives that S t = ∑ t p =1 Z p is √ tσ -subgaussian (essentially, a refinement of the calculation that is need to show Part (3) of Lemma 10) and thus, in particular, for any t ≥ 1 and λ &gt; 0, with probability 1 -δ ,

<!-- formula-not-decoded -->

In fact, by slightly strengthening the argument, one can show that the above inequality holds simultaneously for all t ≥ 1:

Theorem 11 (E.g., Lemma 7 of Russo &amp; Van Roy (2014)) . Let F be a filtration and let ( Z p ) p be an F -adapted, conditionally σ -subgaussian process. Then for any λ &gt; 0 , with probability at least 1 -δ , for all t ≥ 1 ,

<!-- formula-not-decoded -->

where S t = ∑ t p =1 Z p .

Proof of Theorem 5 Let us introduce the following helpful notation: For vectors x, y ∈ R t , let 〈 x, y 〉 t = ∑ t p =1 x p y p , ‖ x ‖ 2 t = 〈 x, x 〉 t , and for f : X → R , ‖ f ‖ 2 t = ∑ t p =1 f 2 ( X p ). More generally, we will overload addition and subtraction such that for x ∈ R t , x + f ∈ R t is the vector whose p th coordinate is x p + f ( X p ) ( x p and X p both appear on purpose here). We also overload 〈· , ·〉 t such that 〈 x, f 〉 t = 〈 f, x 〉 t = ∑ t p =1 x p f ( X p ).

Define Z p using Y p = f ∗ ( X p ) + Z p and collect ( Y p ) t p =1 and ( Z p ) t p =1 into the vectors Y and Z . As in the statement of the theorem, let F = ( F p ) p =0 , 1 ,... be such that for any s ≥ 1, ( X 1 , Y 1 , . . . , X p -1 , Y p -1 , X p ) is F p -1 -measurable. Note that for any p ≥ 1, Z p = Y p -f ∗ ( X p ) is F p -measurable, hence ( Z p ) p ≥ 1 is F -adapted.

With this, elementary calculation gives

<!-- formula-not-decoded -->

Splitting ‖ f ∗ -f ‖ 2 t and rearranging gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that ˆ f t = argmin f ∈F ‖ Y -f ‖ 2 t . Plugging ˆ f t into 15 in place of f and using that thanks to f ∗ ∈ F , ‖ Y -ˆ f t ‖ 2 t ≤ ‖ Y -f ∗ ‖ 2 t , we get

<!-- formula-not-decoded -->

where

Thus, it remains to bound E ( ˆ f t ). For this fix some α &gt; 0 to be chosen later and let G ( α ) ⊂ F be an α -cover of F in ‖ · ‖ ∞ . Let g ∈ G ( α ) be a random function, also to be chosen later. We have

<!-- formula-not-decoded -->

We start by bounding the last term above. A simple calculation gives that for any fixed f ∈ F , w.p. 1 -δ , 2 〈 Z, f -f ∗ 〉 t is 2 σ ‖ f -f ∗ ‖ t -subgaussian. Hence, with probability 1 -δ , simultaneously for all t ≥ 1,

<!-- formula-not-decoded -->

where the equality follows by choosing λ = 1 / (4 σ 2 ) (which makes the first and last terms cancel). (Note how splitting ‖ f -f ∗ ‖ 2 t into two halves allowed us to bound the 'error term' E ( f ) independently of t .) Now, by a union bound, it follows that with probability at least 1 -δ , the second term is bounded by 4 σ 2 log( |G ( α ) | /δ ).

Let us now turn to bounding the first term. We calculate

<!-- formula-not-decoded -->

where for the last inequality we chose g = argmin ˜ g ∈G ( α ) ‖ ˆ f t -˜ g ‖ ∞ so that ‖ ˆ f t -g ‖ t ≤ α √ t and used Cauchy-Schwartz, together with that ‖ g ‖ t , ‖ ˆ f t ‖ t , ‖ f ∗ ‖ t ≤ C √ t , which follows from g, ˆ f t , f ∗ ∈ F and that by assumption all functions in F are bounded by C .

It remains to bound ‖ Z ‖ t . For this, we observe that with probability 1 -δ , simultaneously for all t ≥ 1,

<!-- formula-not-decoded -->

Indeed, this follows because with probability 1 -δ , simultaneously for any s ≥ 1, | Z p | 2 ≤ 2 σ 2 log(2 s ( s +1) /δ ) holds because of a union bound and Eq. (13). Therefore, for the above choice g , with probability 1 -δ , simultaneously for all t ≥ 1, it holds that

<!-- formula-not-decoded -->

Merging this with Eqs. (16) and (17) and with another union bound, we get that with probability 1 -δ , for any t ≥ 1,

<!-- formula-not-decoded -->

where N α is the ( α, ‖ · ‖ ∞ )-covering number of F .

## B Proof of Theorem 3

In this section we establish a regret lower bound by reduction to a known result for tabular MDP.

Proof. We assume without loss of generality that d is a multiple of 4 and d ≥ 8. We set S = 2 and A = d/ 4 ≥ 2. According to Azar et al. (2017), Osband &amp; Van Roy (2016), there exists an MDP M ( S , A , P, r, H ) with S states, A actions and horizon H such that any algorithm has regret at least Ω( √ HSAT ). In this case, we have |S × A × S| = d . We use σ ( s, a, s ′ ) to denote the index of ( s, a, s ′ ) in S × A × S . Letting

<!-- formula-not-decoded -->

and θ i = P ( s ′ | s, a ) if σ ( s, a, s ′ ) = i , we will have P ( s ′ | s, a ) = ∑ d i =1 θ i P i ( s ′ | s, a ) . Therefore P can be parametrized using (2). Therefore, the known lower bound Ω( √ HSAT ) implies a worst-case lower bound of Ω( √ H · d/ 2 · T ) = Ω( √ HdT ) for our model.

## C Implementation

## C.1 Analysis of Implemented Confidence Bounds

In the implementation of UCRL-VTR used in Section 6, we used different confidence intervals then the ones stated in the paper. The confidence intervals used in our implementation are the ones introduced in Abbasi-Yadkori et al. (2011). These confidence intervals are much tighter in the linear setting than the ones introduced in Section 3 and thus have better practical performance. The purpose of this section is to formally introduce the confidence intervals used in our implementation of UCRL-VTR as well as show how these confidence intervals were adapted from the linear bandit setting to the linear MDP setting.

## C.1.1 Linear MDP Assumptions

For our implementation of UCRL-VTR we used different confidence then was introduced in the paper. These are the tighter confidence bounds from the seminal work done by Abbasi-Yadkori et al. (2011) and further expanded upon in Chapter 20 of Lattimore &amp; Szepesv´ ari (2020). Now we will state some assumptions in the MDP setting, then we will state the equivalent assumptions from the linear bandit setting, and lastly we will make the connections between the two that allow us to use the confidence bounds from the linear bandit setting in the RL setting.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where t is defined in the table of A.4. Also note that in this section ( · ) ∗ denotes the true parameter or model, ( · ) MDP denotes something derived or used in the linear MDP setting, and ( · ) LIN denotes something derived or used in the linear bandit setting. Now, under 1-3 of C.1.1 we hope to construct a confidence set C MDP t such that

<!-- formula-not-decoded -->

with high probability. Now the choice of how to choose both C MDP t and β t comes from the linear bandit literature. We will introduce the necessary theorems and assumptions to derive both C LIN t and β t in the linear bandit setting and then adapt the results from the linear bandit setting to the linear MDP setting.

## C.1.2 Tighter Confidence Bounds for Linear Bandits

The following results are introduced in the paper by Abbasi-Yadkori et al. (2011) and are further explained in Chapter 20 of the book by Lattimore &amp; Szepesv´ ari (2020). In this section, we will introduce the theorems and lemmas that allows us to derive tighter confidence intervals for the linear bandit setting. Then we will carefully adapt the confidence intervals to the linear bandit setting. Now supposed a bandit algorithm has chosen actions A 1 , ..., A t ∈ R d and received rewards X LIN 1 , ..., X LIN t with X LIN s = 〈 A t , θ LIN ∗ 〉 + η s where η s is some zero mean noise. The least squares estimator of θ LIN ∗ is the minimizer of the following loss function

<!-- formula-not-decoded -->

where λ &gt; 0 is the regularizer. This loss function is minimized by

<!-- formula-not-decoded -->

notice how this linear bandit problem is very similar to the linear MDP problem introduced in section 3 of our paper. In our linear MDP setting, it is convenient to think of M and W as serving equivalent purposes (storing rank one updates) thus it is also convenient to think of A t and X MDP t as serving equivalent purposes (the features by which we use to make our predictions), where X MDP t is defined in section 3 of our paper with some added notation to distinguish it from the X LIN t used here in the linear bandit setting. We will now build up some intuition by making some simplifying assumptions.

1. No regularization: λ = 0 and W t is invertible.
2. Independent subgaussian noise: ( η s ) s are independent and σ -subgaussian
3. Fixed Design: A 1 , ..., A t are deterministically chosen without the knowledge of X LIN 1 , ..., X LIN t

finally it is also convenient to think of X LIN t and V t +1 ( s t +1 ) as serving equivalent purposes (the target of our predictions). Thus the statements we prove in the linear bandit setting can be easily adapted to the linear MDP setting. While none of the assumptions stated above is plausible in the bandit setting, the simplifications eases the analysis and provides insight.

Comparing θ LIN ∗ and ˆ θ LIN t in the direction x ∈ R d , we have

<!-- formula-not-decoded -->

Since ( η s ) s are independent and σ -subgaussian, by Lemma 5.4 and Theorem 5.3 (need to be stated),

<!-- formula-not-decoded -->

A little linear algebra shows that ∑ t s =1 〈 x, W -1 t A s 〉 2 = ‖ x ‖ 2 W -1 t and so,

<!-- formula-not-decoded -->

We now remove the limiting assumptions we stated above and use the newly stated assumptions for the rest of this section

1. There exists a θ LIN ∗ ∈ R d such that X LIN t = 〈 θ LIN ∗ , A t 〉 + η t for all t ≥ 1.
2. The noise is conditionally σ -subgaussian:

<!-- formula-not-decoded -->

where F t -1 is such that A 1 , X LIN 1 , ..., A t -1 , X LIN t -1 are F t -1 -measurable.

## 3. In addition, we now assume λ &gt; 0.

The inclusion of A t in the definition of F t -1 allows the noise to depend on past choices, including the most recent action. Since we want exponentially decaying tail probabilities, one is tempted to try the Cramer-Chernoff method:

<!-- formula-not-decoded -->

Sadly, we do not know how to bound this expectation. Can we still somehow use the Cramer-Chernoff method? We take inspiration from looking at the special case of λ = 0 one last time, assuming that W t = ∑ t s =1 A s A glyph[latticetop] s is invertible. Let

<!-- formula-not-decoded -->

Recall that ˆ θ LIN t = W -1 t ∑ t s =1 X LIN s A s = θ LIN ∗ + W -1 t S t . Hence,

<!-- formula-not-decoded -->

The next lemma shows that the exponential of the term inside the maximum is a supermartingale even when λ ≥ 0.

Lemma 12. For all x ∈ R d the process D t ( x ) = exp( 〈 x, S t 〉 -1 2 ‖ x ‖ W 2 t ) is an F -adapted non-negative supermartingale with D 0 ( x ) ≤ 1 .

The proof for this Lemma can be found in Chapter 20 of the book by Lattimore &amp; Szepesv´ ari (2020). For simplicity, consider now again the case when λ = 0. Combining the lemma and the linearisation idea almost works. The Cramer-Chernoff method leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now Lemma 12 shows that E [ D t ( x )] ≤ 1. Now using Laplace's approximation we write

<!-- formula-not-decoded -->

where h is some measure on R d chosen so that the integral can be calculated in closed form. This is not a requirement of the method, but it does make the argument shorter. The main benefit of replacing the maximum with an integral is that we obtain the following lemma

Lemma 13. Let h be a probability measure on R d ; then; ¯ D t = ∫ R d D t ( x ) dh ( x ) is an F -adapted non-negative supermartingale with ¯ D 0 = 1 .

The proof of Lemma 13 can, again, be found in Chapter 20 of the book by Lattimore &amp; Szepesv´ ari (2020). Now the following theorem is the key result from which the confidence set will be derived.

Theorem 14. For all λ &gt; 0 , and δ ∈ (0 , 1)

<!-- formula-not-decoded -->

Furthermore, if ‖ θ LIN ∗ ‖ 2 ≤ m 2 , then P ( exists t ∈ N + : θ LIN ∗ / ∈ C LIN t ) ≤ δ with

<!-- formula-not-decoded -->

The proof of Theorem 14 can be found in Chapter 20 of the book by Lattimore &amp; Szepesv´ ari (2020).

## C.1.3 Adaptation of the Confidence Bounds to our Linear MDP Setting

Now with the Lemmas and Theorems introduced in the previous section we are ready to derive the confidence bounds used in our implementation of UCRL-VTR. Now using the notation from the linear bandit setting we set

1. The target X MDP t = ∫ j V t ( s ′ ) P j ( ds ′ | s t , a t )
2. Y t = V t ( s t +1 )
3. F t -1 = σ ( s 1 , a 1 , ..., s t -1 , a t -1 ), which just means the filtration is set to be the sigma-algebra generated by all past states and actions observed.
4. η t = Y t - 〈 X MDP t , θ MDP ∗ 〉 = V t ( s t +1 ) -∫ j V t ( s ′ ) P ∗ j ( ds ′ | s t , a t ), since θ MDP ∗ is the true model of the MDP.
5. M t in the linear MDP setting is defined equivalently to W t in the linear bandit setting, i.e. they are both the sums of a regularizer term and a bunch of rank one updates.

it can be seen that our the noise in our system η t has zero mean E [ η t | F t -1 ] = 0 finally the noise in our system has variance H/ 2 thus our system in H/ 2-subgaussian.

Lemma 15. (Hoeffding's lemma) Let Z = Z -E [ Z ] be a real centered random variable such that Z ∈ [ a, b ] almost surely. Then E [exp( αZ )] ≤ exp( α 2 ( b -a ) 2 8 ) for any α ∈ R or Z is subgaussian with variance σ 2 = ( b -a ) 2 4 .

Proof Define ψ ( α ) = log E [exp( αZ )] we can then compute

<!-- formula-not-decoded -->

Thus ψ ′′ ( α ) can be interpreted as the variance of the random variable Z under the probability measure d Q = exp( αZ ) E [exp( αZ )] d P , but since Z ∈ [ a, b ] almost surely, we have, under any probability

<!-- formula-not-decoded -->

The fundamental theorem of calculus yields

<!-- formula-not-decoded -->

using ψ (0) = log 1 = 0 and ψ ′ (0) = E [ Z

<!-- formula-not-decoded -->

Now using Lemma 15 and the fact that Y t is bounded in the range of [0 , H ], E [ Y t ] = 〈 X MDP t , θ MDP ∗ 〉 , and η t = Y t -〈 X MDP t , θ MDP ∗ 〉 = Y t -E [ Y t ], the noise η t in our linear MDP setting is H/ 2-subgaussian. This result is also stated in a proof from A.4.

Putting this all together we can derive the tighter confidence set for UCRL-VTR in the linear setting,

<!-- formula-not-decoded -->

where here in the linear MDP setting M t replaces W t from the linear bandit setting and ‖ θ MDP ∗ ‖ 2 ≤ m 2 . The justification of using these bounds in the linear MDP setting follow exactly from the justification given above for using these bounds in the linear bandit setting.

## C.2 UCRL-VTR

In the proceeding subsections we discuss the implementation of the algorithms studied in Section 6 of the paper. The first algorithm we present is the algorithm used to generate the results for UCRL-VTR.

## Algorithm 3 UCRL-VTR with Tighter Confidence Bounds

- 1: Input: MDP, d, H, T = KH ;
- 2: Initialize: M 1 , 1 ← I , w 1 , 1 ← 0 ∈ R d × 1 , θ 1 ← M -1 1 , 1 w 1 , 1 for 1 ≤ h ≤ H , d 1 = |S| × |A| ;
- 3: Initialize: δ ← 1 /K , and for 1 ≤ k ≤ K ,
- 4: Compute Q-function Q h, 1 using θ 1 , 1 according to (3);
- 5: for k = 1 : K do
- 6: Obtain initial state s k 1 for episode k ;
- 7: for h = 1 : H do
- 8: Choose action greedily by

and observe the next state s k h +1 .

- 9: Compute the predicted value vector:

glyph[triangleright] Evaluate the expected value of next state

<!-- formula-not-decoded -->

- 10: y h,k ← V h +1 ,k ( s k h +1 )
- 11: M h +1 ,k ← M h,k + X h,k X glyph[latticetop] h,k
- 12: w h +1 ,k ← w h,k + y h,k · X h,k
- 13: end for
- 14: Update at the end of episode:

glyph[triangleright] Update regression parameters glyph[triangleright] Update Model Parameters

16: end for

The iterative Q-update for Algorithm 3 is

<!-- formula-not-decoded -->

The choice of the confidence bounds used in Algorithm 3 comes from the tight bounds derived in AbbasiYadkori et al. (2011) for linear bandits and further expanded upon in Chapter 20 of Lattimore &amp; Szepesv´ ari (2020). The details of which are shown and stated in C.1. We slightly tighten the values for the noise at each stage by using the fact that for each stage in the horizon, h ∈ [ H ], the value V k h ( · ) is capped as to never be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 15: Compute Q h,k +1 for h = H,.. . , 1 , using θ k +1 according to (21) using

<!-- formula-not-decoded -->

glyph[triangleright] Computing Q functions

greater than H -h +1. The appearance of the √ d 1 comes from the fact that ‖ θ ∗ ‖ 2 ≤ √ d 1 for all θ ∗ ∈ R d in the tabular setting since θ ∗ in the tabular setting is equal to the true model of the environment.

## C.3 EGRL-VTR

In this section we discuss the algorithm EGRL-VTR. This algorithm is very similar to UCRL-VTR expect it performs ε -greedy value iteration instead of optimistic value iteration and acts ε -greedy with respect to Q h,k .

## Algorithm 4 EGRL-VTR

- 1: Input: MDP, d, H, T = KH,ε &gt; 0;
- 2: Initialize: M 1 , 1 ← I , w 1 , 1 ← 0 ∈ R d × 1 , θ 1 ← M -1 1 , 1 w 1 , 1 for 1 ≤ h ≤ H ;
- 3: Compute Q-function Q h, 1 using θ 1 , 1 according to (22);
- 4: for k = 1 : K do
- 5: Obtain initial state s k 1 for episode k ;
- 6: for h = 1 : H do
- 7: With probability 1 -ε do
- 8:

<!-- formula-not-decoded -->

else pick a uniform random action a k h ∈ A . Observe the next state s k h +1 . Compute the predicted value vector: glyph[triangleright] Evaluate the expected value of next state

<!-- formula-not-decoded -->

- 9: y h,k ← V h +1 ,k ( s k h +1 )
- 10: M h +1 ,k ← M h,k + X h,k X glyph[latticetop] h,k
- 11: w h +1 ,k ← w h,k + y h,k · X h,k
- 12: end for
- 13: Update at the end of episode:

glyph[triangleright] Update regression parameters glyph[triangleright] Update Model Parameters

glyph[triangleright] Computing Q functions

<!-- formula-not-decoded -->

- 14: Compute Q h,k +1 for h = H,.. . , 1 , using θ k +1 according to (22) 15: end for

The iterative value update for EGRL-VTR is

<!-- formula-not-decoded -->

## C.4 EG-Frequency

In this section we discuss the algorithm EG-Frequency. This algorithm is the ε -greedy version of UC-MatrixRL Yang &amp; Wang (2019a).

## Algorithm 5 EG-Frequency

```
1: Input: MDP, Features φ : S × A → R |S||A| and ψ : S → R |S| , ε > 0, and the total number of episodes K ; 2: Initialize: A 1 ← I ∈ R |S||A|×|S||A| , M 1 ← 0 ∈ R |S||A|×|S| , and K ψ ← ∑ s ′ ∈S ψ ( s ′ ) ψ ( s ′ ) glyph[latticetop] ; 3: for k = 1 : K do 4: Let Q h,k be given in (23) using M k ; 5: for h = 1 : H do 6: Let the current state be s k h ; 7: With probability (1 -ε ) play action a k h = arg max a ∈A Q h,k ( s k h , a ) else pick a uniform random action a k h ∈ A . 8: Record the next state s k h +1 9: end for 10: A k +1 ← A k + ∑ h ≤ H φ ( s k h , a k h ) φ ( s k h , a k h ) glyph[latticetop] 11: M k +1 ← M k + A -1 k +1 ∑ h ≤ H φ ( s k h , a k h ) ψ ( s k h +1 ) glyph[latticetop] K -1 ψ 12: end for
```

The iterative Q-update for EG-Frequency is

<!-- formula-not-decoded -->

Note that Ψ is a |S| × |S| whose rows are the features ψ ( s ′ ) and Φ is a |S||A| × |S||A| whose rows are the features φ ( s, a ). In the tabular RL setting both Ψ and Φ are the identity matrix which is what we used in our numerical experiments. In the tabular RL setting, EG-Frequency stores the counts of the number of times it transitioned to next state s ′ from the state-action pair ( s, a ) and fits the estimated model M k accordingly.

## C.4.1 Futher Implementation Notes

In this section, we include some further details on how we implemented Algorithms 3, 4, and 5. All code was written in Python 3 and used the Numpy and Scipy libraries. All plots were generated using MatPlotLib. In Algorithm 3, Numpy's logdet function was used to calculate the determinate in step 15 for numerical stability purposes. No matrix inversion was performed in our code, instead a Sherman-Morrison update was performed for each matrix in which a matrix inversion is performed at each ( k, h ) in order to save on computation. To read more about the Sherman Morrison update in the context of RL, we refer to the reader to Eqn (9.22) of Sutton &amp; Barto (2018). When computing the weighted L1-norm, we added a small constant to each summation in the denominator to avoid dividing by zero. Finally, when computing UC-MatrixRL we also used the self-normalize bounds introduced in the beginning of this section. Some pseudocode for using self-normalized bounds with UC-MatrixRL can be found in step 5 of Alg 6.

## D Mixture Model

In this section, we introduce, analyze, and evaluate a Linear model-based RL algorithm that used both the canonical model and the VTR model for planning. We call this algorithm UCRL-MIX.

## D.1 UCRL-MIX

Below a meta-algorithm for UCRL-MIX

## Algorithm 6 UCRL-MIX

- 1: Compute Algorithm 3 and UC-MatrixRL Yang &amp; Wang (2019a) simultaneously.
- 2: At end of episode k , perform value iteration and set V H +1 ,k ( s ) = 0.
- 3: for h = H +1 : 1 do
- 4: for s ∈ |S| and a ∈ |A| do
- 5: Compute the confidence set bonuses as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 7: Perform one step of value iteration using the VTR model as follows: Q h,k ( s, a ) = r ( s, a ) + X glyph[latticetop] h,k θ k + √
- 8: else
- 9: Update Q h,k ( s, a ) according to Equation 8 Yang &amp; Wang (2019a) using the UC-MatrixRL model A k . Note that in Yang &amp; Wang (2019a) they use n to denote the current episode, in our paper we use k to denote the current episode.
- 10: end if

<!-- formula-not-decoded -->

- 12: end for
- 13: end for

We are now using multiple models instead of a single model, we must adjust our confidence sets accordingly. By using a union bound we replace δ with δ/ 2 for our confidence parameter. This updated confidence parameter changes the term inside the logarithm. We now have log(2 /δ ) where as before we had log(1 /δ ).

## D.2 Numerical Results

We will include the cumulative regret and the weighted L1 norm of UCRL-MIX on the RiverSwim environment as in Section 6. We also include a bar graph of the relative frequency with which the algorithm used the VTR-model for planning and the canonical model for planning.

<!-- formula-not-decoded -->

Figure 5: In the plots for the model error we include model error for both the VTR-model and the canonical model. Even though only one is used during planning both are updated at the end of each episode.

<!-- image -->

If we compare the results of Figure 5 with the results of Figure 2 from Section 6.3 we see that the cumulative regret of UCRL-MIX is almost identical to the cumulative regret of UCRL-VTR. The model errors of both the VTR and the canonical models are almost identical to the model errors of UCRL-VTR and UC-MatrixRL respectively.

Figure 6: UCRL-MIX rarely, if ever, chooses the canonical model for planning on the RiverSwim environments.

<!-- image -->

From Figure 6, we see that on the RiverSwim environment, UCRL-MIX almost always uses the VTR-model for planning. We calculate this frequency by counting the number of times Step 7 of Alg 6 was observed up until episode k and by counting the number of times Step 9 of Alg 6 was observed up until episode k . We then divide these counts by the sum of the counts to get a percentage. We believe the reason the algorithm overwhelming chose the VTR-model was due to the fact that the confidence intervals for the VTR-model shrink much faster than the confidence intervals for the canonical model. The canonical model is forced to explore much longer than the VTR-model as its objective is to learn a globally optimal model rather than a model that yields high reward. Thus, the canonical model is forced to explore all state-action-next state tuples, even ones that do not yield high reward, in order to meet its objective of learning a globally optimal model while the VTR-model is only forced to explore state-action-next state tuples that fall in-line with its objective of accumulating high reward. The set of all state-action-next state tuples is much larger then the set of state-action-next state tuples that yield high reward which means the confidence intervals for the canonical model shrink slower than the confidence sets of the VTR-model on the RiverSwim environment.