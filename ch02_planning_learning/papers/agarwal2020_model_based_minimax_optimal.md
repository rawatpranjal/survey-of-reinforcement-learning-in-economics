## Model-Based Reinforcement Learning with a Generative Model is Minimax Optimal

Alekh Agarwal Microsoft alekha@microsoft.com

Sham Kakade University of Washington sham@cs.washington.edu

Lin F. Yang University of California, Los Angeles linyang@ee.ucla.edu

April 7, 2020

## Abstract

This work considers the sample and computational complexity of obtaining an ε -optimal policy in a discounted Markov Decision Process (MDP), given only access to a generative model. In this model, the learner accesses the underlying transition model via a sampling oracle that provides a sample of the next state, when given any state-action pair as input. This widely studied setting provides a natural abstraction which permits the investigation of sample-based planning over a long horizon, decoupled from the complexity of exploration. In this work, we study the effectiveness of the most natural plug-in approach to model-based planning: we build the maximum likelihood estimate of the transition model in the MDP from observations and then find an optimal policy in this empirical MDP. We ask arguably the most basic and unresolved question in model based planning: is the naïve 'plug-in' approach, nonasymptotically, minimax optimal in the quality of the policy it finds, given a fixed sample size? Here, the non-asymptotic regime refers to when the sample size is sublinear in the model size.

With access to a generative model, we resolve this question in the strongest possible sense: our main result shows that any high accuracy solution in the plug-in model constructed with N samples, provides an ε -optimal policy in the true underlying MDP (where ε is the minimax accuracy with N samples at every state, action pair). In comparison, all prior (non-asymptotically) minimax optimal results use model free approaches, such as the Variance Reduced Q-value iteration algorithm (Sidford et al., 2018a), while the best known model-based results (e.g. Azar et al. (2013)) require larger sample sizes in their dependence on the planning horizon or the state space. Notably, we show that the model-based approach allows the use of any efficient planning algorithm in the empirical MDP, which simplifies algorithm design as this approach does not tie the algorithm to the sampling procedure. The core of our analysis is a novel 'absorbing MDP' construction to address the statistical dependency issues that arise in the analysis of model-based planning approaches, a construction which may be helpful more generally.

## 1 Introduction

How best to plan across a long-horizon with access to an approximate model of a Markov Decision Process? This is a fundamental question at the heart of reinforcement learning, and understanding it is essential to tackling even more complex challenges such as sample-efficient exploration (see e.g. Kakade et al. (2003); Strehl et al. (2006); Strehl (2007); Jaksch et al. (2010); Osband and Van Roy (2014); Azar et al. (2017); Sidford et al. (2018b,a)). When the approximate model is arbitrary, these questions are studied, for example, in the approximate dynamic programming literature (Bertsekas, 1976). Before moving to approximation questions, a more basic question is an information theoretic one: how many samples from the Markov Decision Process are required to yield a near optimal policy? Our work studies this question in the generative model framework introduced in the work of Kearns and Singh (1999).

In the generative model setting, the learning agent has sampling access to a generative model of the Markov Decision Process (henceforth MDP), and it can query the next state s ′ sampled from the transition process, given as input any state-action pair. The information theoretic question is to quantify how many samples from the generative model are required in order to obtain a near optimal policy; this question is analogous to the classical question of sample complexity in the supervised learning setting.

Arguably, the simplest approach here is a model-based one: the approach is to first build the maximum likelihood estimate of the transition model in the MDP from observations and then find an optimal policy in this empirical MDP. This work seeks to address the following unresolved question: is the naïve 'plug-in' approach, non-asymptotically, minimax optimal in the quality of the policy it finds, given a fixed sample size? Throughout, we refer to the non-asymptotic regime as one where the sample size is sublinear in the model size. This work answers this question affirmatively showing that a model based planning approach is non-asymptotically minimax optimal.

We note that the first provably, non-asymptotically, minimax optimal algorithm is the Variance Reduced Q-value iteration algorithm (Sidford et al., 2018a), a model free approach. The significance of the optimality of our model-based result is that it allows the use of any efficient planning algorithm in the empirical MDP, which simplifies algorithm design due to that the algorithm utilized need not be tied to the sampling procedure. We now discuss our contributions and the related work more broadly.

## 1.1 Our Contributions

There exists a large body of literature on MDPs and RL (see e.g. Kakade et al. (2003); Strehl et al. (2009); Kalathil et al. (2014); Dann and Brunskill (2015) and reference therein). A summary of our result relative to the prior works using a generative model is presented in Table 1. Here, /epsilon1 is a desired accuracy parameter; |S| and |A| are the cardinalities of the (finite) state and actions spaces; γ is a discount factor. We refer to ε -optimal policy the one whose discounted cumulative value in the MDP is ε close to the optimal value.

Before discussing the sample complexity of finding an ε -optimal policy, let us review the results on computing an ε -optimal value function. This refers to the problem of finding a function ̂ Q which approximates Q /star to an error of /epsilon1 at all states. The work of Azar et al. (2012) shows that for /epsilon1 ∈ (0 , 1) it suffices to use at most ˜ O ( |S||A| (1 -γ ) 3 /epsilon1 2 ) calls to the generative model in order to return an ε -optimal value function 1 . Furthermore, the work of Azar et al. (2012) shows this sample complexity is minimax optimal.

1 We conjecture that our techniques can be used to broaden the range of ε to go beyond /epsilon1 ∈ (0 , 1) , as needed in Azar et al.

Obtaining an ε -optimal policy (rather than just estimating the value itself) is more subtle; naïvely, a policy obtained in a greedy manner from an ε -optimal value will incur a further degradation in its quality by a factor of 1 -γ (Singh and Yee, 1994). The work of Azar et al. (2013) shows that this additional error amplification is avoidable provided that the number of samples is at least O ( |S| 2 |A| ) (see Table 1); note that such a sample size is actually linear in the model size.

Our work avoids this error amplification and shows that for a desired accuracy threshold of /epsilon1 , we can find an /epsilon1 -optimal policy for any /epsilon1 ∈ (0 , 1 √ 1 -γ ] using at most ˜ O ( |S||A| (1 -γ ) 3 /epsilon1 2 ) samples. Our result holds for any planning algorithm that finds a near optimal policy in the empirically constructed MDP. Due to existing lower bounds (Azar et al., 2012; Sidford et al., 2018a), this bound is known to be minimax optimal for /epsilon1 ∈ (0 , 1] . Notably, this sample complexity is o ( S 2 A ) whenever /epsilon1 2 ≥ 1 / ((1 -γ ) 3 |S| ) , meaning that we can use the model to find a near optimal policy even in sample regimes where no meaningfully accurate approximation to the actual transition probabilities can be constructed.

Prior to this work, the only other non-asymptotically minimax optimal approach takes a different algorithmic path: Sidford et al. (2018a) (also see Sidford et al. (2018b)) use a modification of the Q -value iteration method, with explicit control of variance in value estimates, to obtain an optimal sample complexity for /epsilon1 ∈ (0 , 1] . Our guarantees hold for a broader range of /epsilon1 values (though we conjecture that our techniques could also improve the ε dependence in Sidford et al. (2018a). See Footnote 1.).

Importantly, our work highlights that the sub-optimality of the prior model-based results was not due to any inherent limitation of the approach, but instead due to a matter of analysis. As a by-product, we retain a conceptually and algorithmically simpler solution strategy relative to Sidford et al. (2018a). On a more technical note, our analysis is based on a novel absorbing MDP construction to deal with the dependence issues which arise in the analysis of Azar et al. (2012, 2013), and this argument might be more broadly useful.

## 2 Setting

Markov Decision Process We denote a discounted Markov decision process (MDP) as a tuple M = ( S , A , P M , r M , γ ) , where S is a finite set of states, A is a finite set of actions, P M : S × A → R S is the transition kernel (that is, P M ( s ′ | s, a ) is the probability of obtaining state s ′ when we take action a in state s ), r M : S × A → [0 , 1] is the reward function 2 , and γ ∈ (0 , 1) is a discount factor. For any ( s, a ) , we denote P M ( · | s, a ) ∈ R |S| as the probability vector conditioning on state-action pair ( s, a ) . A (deterministic) stationary policy is a map π : S → A that maps a state to an action. The value function of a policy π is a vector V π M ∈ R |S| , defined as follows.

<!-- formula-not-decoded -->

(2012). In particular, the proof of Lemma 10 (used to prove Theorem 1) uses a self-bounding approach which we conjecture can be used to broaden the range of ε to allow for /epsilon1 ∈ (0 , 1 √ 1 -γ ] . We do not focus on this improvement in this work, as our main focus is on the value of the policy itself.

2 We consider the setting where the rewards are in [0 , 1] . Our results can be generalized to other ranges of reward function via a standard reduction (see e.g. Sidford et al. (2018a))

Table 1: Sample Complexity to Compute /epsilon1 -Optimal Policies Using the Generative Sampling Model : Here |S| is the number of states, |A| is the number of actions per state, γ ∈ (0 , 1) is the discount factor, and C is an upper bound on the ergodicity. We ignore poly log( |S||A| /δ/ (1 -γ )) factors in the sample complexity. Rewards are bounded between 0 and 1.

| Algorithm                                        | Sample Complexity                                      | /epsilon1 -Range         | References              |
|--------------------------------------------------|--------------------------------------------------------|--------------------------|-------------------------|
| Phased Q-Learning                                | C |S||A| (1 - γ ) 7 /epsilon1 2                        | (0 , (1 - γ ) - 1 ]      | Kearns and Singh (1999) |
| Empirical QVI                                    | |S||A| (1 - γ ) 5 /epsilon1 2                          | (0 , 1]                  | Azar et al. (2013)      |
| Empirical QVI                                    | |S||A| (1 - γ ) 3 /epsilon1 2                          | ( 0 , 1 √ (1 - γ ) |S| ] | Azar et al. (2013)      |
| Randomized Primal-Dual Method                    | C |S||A| (1 - γ ) 4 /epsilon1 2                        | (0 , (1 - γ ) - 1 ]      | Wang (2017)             |
| Sublinear Randomized Value Iteration             | |S||A| (1 - γ ) 4 /epsilon1 2 · poly log /epsilon1 - 1 | (0 , 1]                  | Sidford et al. (2018b)  |
| Variance Reduced QVI                             | |S||A| (1 - γ ) 3 /epsilon1 2 · poly log /epsilon1 - 1 | (0 , 1]                  | Sidford et al. (2018a)  |
| Empirical MDP + any ac- curate black-box planner | |S||A| (1 - γ ) 3 /epsilon1 2                          | (0 , (1 - γ ) - 1 / 2 ]  | This work               |

where a t = π ( s t ) and s 1 , s 2 , s 3 , . . . are generated from the distribution s t +1 ∼ P M ( ·| s t , a t ) . We also define an action value function Q π M ∈ R S×A for policy π :

<!-- formula-not-decoded -->

When the MDP M is clear from the context, we drop the subscript to avoid clutter. The goal of a planning algorithm is to find a stationary policy in the MDP which maximizes the expected reward, denoted by π /star . The famous theorem of ? shows that there exists a policy π /star which simultaneously maximizes V π ( s 0 ) for all s 0 ∈ S . We also use Q /star and V /star to denote the value functions induced by π /star We call a policy, π , /epsilon1 -optimal, if V π ( s ) ≥ V ∗ ( s ) -/epsilon1 for all s ∈ S .

Generative Model Assume we have a access to a generative model or a sampler , which can provide us with samples s ′ ∼ P ( · | s, a ) . Suppose we call our sampler N times at each state action pair. Let ̂ P be our empirical model, defined as follows:

where count ( s ′ , s, a ) is the number of times the state-action pair ( s, a ) transitions to state s ′ . We define ̂ M to be the empirical MDP that is identical to the original M , except that it uses ̂ P instead of P for the transition kernel. We let ̂ V π and ̂ Q π to denote the value functions of a policy π in ̂ M , and ̂ π /star , ̂ Q /star and ̂ V /star refer to the optimal policy and its value functions in ̂ M . The reward function r is assumed to be known and deterministic 3 , and hence is identical in M and M .

<!-- formula-not-decoded -->

̂ 3 If r is unknown, we can use additional |S||A| samples to obtain the exact value of r . If r is stochastic, we can query |S||A| //epsilon1 2 / (1 -γ ) 2 samples to obtain a sufficiently accurate estimate of its mean. In both cases, the complexity contributed by r is only a lower order term to the present case. We can therefore assume, without loss of generality, r is known and deterministic.

Optimization Oracle Our goal in this paper is to determine the smallest sample size N , such that a planner run in ̂ M returns a near-optimal policy in M . In order to decouple the statistical and computational aspects of planning with respect to an approximate model ̂ M , we will make use of an optimization oracle which takes as input an MDP M and returns a policy π satisfying: ‖ V π M -V /star M ‖ ∞ ≤ /epsilon1 opt and ‖ Q π M -Q /star M ‖ ∞ ≤ /epsilon1 opt .

We will use this optimization oracle for the empirical MDP ̂ M , and analyze the performance of the returned policy in the original MDP M . Classical algorithms such as value or policy iteration (Puterman, 2014) are the most common examples, though we discuss more sophisticated oracles as well in the next section.

## 3 Main results

In this section we present our main results. Before presenting our main theorem, we review some of the key challenges and our approach. Our high-level approach is to invoke any reasonable optimization oracle for the sample-based MDP ̂ M , and understand the sub-optimality of the returned policy π in the original MDP M . The key challenge is that π depends on the randomness in ̂ M , and hence, its value estimate from ̂ M is not an unbiased estimator of its value in M . Ausual way to address such issues is via uniform convergence, that is, we first establish that the values of all policies are similar in ̂ M and M . This then implies that the high value of π in ̂ M translates to a high value in M . Unfortunately, a naïve application of this argument yields bounds scaling as |S| 2 . Azar et al. (2013) do establish uniform convergence, but use a more careful argument which yields a bound scaling linearly in |S| , but only when the desired accuracy /epsilon1 ≤ √ 1 / ((1 -γ ) |S| ) , where the |S| factor in the condition of /epsilon1 is due to uniform convergence. Sidford et al. (2018a,b) instead use a more complex algorithmic modification using variance reduction to get a sharper uniform convergence over a smaller class of policies with small variance in their value functions. In our result, we instead rely on a novel technique to directly establish uniform convergence of our value estimates, while utilizing the most natural algorithmic scheme of running a black-box optimization oracle on the sample-based MDP ̂ M . We will show the following result for this scheme.

Theorem 1. Suppose δ &gt; 0 and /epsilon1 ∈ (0 , (1 -γ ) -1 / 2 ] . Let ̂ π be any /epsilon1 opt-optimal policy for ̂ M , i.e. ‖ ̂ Q ̂ π -̂ Q /star ‖ ∞ ≤ /epsilon1 opt . If we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ , where c is an absolute constant, provided γ ≥ 1 / 2 .

Thus, the theorem shows that if /epsilon1 opt is made suitably small (roughly (1 -γ ) /epsilon1 ), then we will find an O ( /epsilon1 ) sub-optimal policy with O ( log |S||A| (1 -γ ) δ / (1 -γ ) 3 //epsilon1 2 ) samples in each s, a pair. The total number of samples from the generative model then is |S||A| N which amounts to O ( |S||A| log |S||A| (1 -γ ) δ / (1 -γ ) 3 //epsilon1 2 ) samples. As remarked before, this is known to be unimprovable (up to a logarithmic factor) in the regime /epsilon1 ∈ (0 , 1] due to the lower bounds of (Azar et al., 2012; Sidford et al., 2018a).

We have so far focused on the statistical aspects of our estimators, since the use of a black-box optimization method in ̂ M allows us to leverage the best possible solutions available. We now discuss some specific

implications on the computational complexity of sparse model-based planning, instantiating the bound for some of the natural methods that may be used. Throughout we focus on attaining /epsilon1 opt = O ((1 -γ ) /epsilon1 ) , since that equates the statistical and optimization errors. A very natural idea is to use value iteration (see e.g. Puterman (2014)), which requires O [(1 -γ ) -1 · log /epsilon1 -1 opt ] iterations, with each iteration taking O ( |S||A| N ) time. Thus the overall running time for this algorithm is

<!-- formula-not-decoded -->

Policy iteration methods (see again Puterman (2014)) can obtain an /epsilon1 opt-optimal policy within the same iteration complexity bound as value iteration. However, each iteration of the policy iteration requires solving a linear system of size |S| 2 , which can be expensive. This computation time can be additionally improved. For instance, after initial phase of reading O ( |S||A| N ) data points, Sidford et al. (2018b) give a randomized algorithm to obtain an /epsilon1 opt-optimal policy with probability at least 1 -δ in time

<!-- formula-not-decoded -->

where ˜ O hides poly log log factors and nnz( P ) means the number of non-zero entries in P . Thus, the computational complexity of this scheme is nearly-linear in the total sample size up to additional logarithmic factors. There are other results for obtaining an exactly optimal policy for the MDP ̂ M as well, for instance the SIMPLEX policy iteration Ye (2011), which runs in time O (poly( |S||A| N/ (1 -γ )) .

## 4 Analysis

We begin with some notation needed for our analysis, and then give a high-level outline of the proof, along with some basic lemmas. We then present our main technical novelty, which is a construction of an auxiliary MDPas a device to guarantee uniform convergence of value functions. We conclude by providing the proof of the theorem in terms of the key lemmas, deferring the proofs of the lemmas to the appendix.

Additional Notation For a vector v , we let ( v ) 2 , √ v , and | v | be the component-wise square, square root, and absolute value operations. We let 1 denotes the vector of all ones (adapting to dimensions based on the context). It is helpful to overload notation and let P be a matrix of size ( S ×A ) ×S where the entry P ( s,a ) ,s ′ is equal to P ( s ′ | s, a ) . Also, let P s,a denote the vector P ( · | s, a ) . We also define P π to be the transition matrix on state-action pairs induced by a deterministic policy π . In particular,

<!-- formula-not-decoded -->

With this notation, we have

<!-- formula-not-decoded -->

Slightly abusing the notation, for V ∈ R S , we define the vector Var P ( V ) ∈ R S×A as:

<!-- formula-not-decoded -->

/negationslash

where the squares are applied componentwise. We also define Σ π M as the variance of the discounted reward, i.e.

where the expectation is induced under the trajectories induced by π in M . It can be verified that, for all π , Σ π satisfies the following Bellman style, self-consistency conditions (see Lemma 6 in Azar et al. (2013)):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is straightforward to verify that ‖ Σ π M ‖ ∞ ≤ γ 2 / (1 -γ ) 2 .

## 4.1 Errors in empirical estimates

We begin the analysis by stating some basic results about empirical estimates of values derived from ̂ M relative to their true values in M . We start with stating a lemma on componentwise error bounds. Its proof has been postponed to the appendix.

Lemma 2 (Componentwise bounds) . For any policy π , we have

In addition, we have:

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For any policy π ,

For the second claim,

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

Another application of

<!-- formula-not-decoded -->

completes the proof of the first inequality.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We hope to invoke the second part of the lemma to establish Theorem 1, where the middle term is the optimization error, and we will focus on bounding the other two terms. We next state another basic lemma.

Lemma3. For any policy π , MDP M and vector v ∈ R |S|×|A| , we have ‖ ( I -γP π ) -1 v ‖ ∞ ≤ ‖ v ‖ ∞ / (1 -γ ) .

Proof. Note that v = ( I -γP π )( I -γP π ) -1 v = ( I -γP π ) w , where w = ( I -γP π ) -1 v . By triangle inequality, we have

<!-- formula-not-decoded -->

where the final inequality follows since P π w is an average of the elements of w by the definition of P π so that ‖ P π w ‖ ∞ ≤ ‖ w ‖ ∞ . Rearranging terms completes the proof.

Our next lemma is a key observation in Lemma 6 of Azar et al. (2012), namely the Bellman property of a policy's variance and its accumulation under the transition operator of the corresponding policy. We provide a short proof in the appendix for completeness.

Lemma 4. For any policy π and MDP M , where P is the transition model of M .

<!-- formula-not-decoded -->

Proof. Note that (1 -γ )( I -γP π ) -1 is matrix whose rows are a probability distribution. For a positive vector v and a distribution ν (where ν is vector of the same dimension of v ), Jensen's inequality implies that ν · √ v ≤ √ ν · v . This implies:

∥ ∥ where we have used that ‖ ( I -γP π ) -1 v ‖ ∞ ≤ 2 ‖ ( I -γ 2 P π ) -1 v ‖ ∞ (which we will prove shortly). The proof is completed as follows: by Equation 2, Σ π M = γ 2 ( I -γ 2 P π ) -1 Var P ( V π M ) , so taking v = Var P ( V π M ) and using that ‖ Σ π M ‖ ∞ ≤ γ 2 / (1 -γ ) 2 completes the proof.

<!-- formula-not-decoded -->

Finally, to see that ‖ ( I -γP π ) -1 v ‖ ∞ ≤ 2 ‖ ( I -γ 2 P π ) -1 v ‖ ∞ , observe:

<!-- formula-not-decoded -->

which proves the claim.

Finally, it will be useful to also have more direct bounds on the errors in our value estimates which follow directly from Hoeffding's inequality, even though we are eventually after more careful bounds that account for variance. This result can be also be found as Lemma 4 in Azar et al. (2013), and is a standard concentration argument. For completeness, we provide its proof here.

Lemma 5 (Crude Value Bounds, Lemma 4 in Azar et al. (2013)) . Let δ ≥ 0 . With probability greater than 1 -δ ,

Proof. Note that V /star is a fixed vector independent with the randomness in ̂ P . Moreover, ‖ V /star ‖ ∞ ≤ (1 -γ ) -2 . Thus, by Hoeffding bound and a union bound over all S × A , we have, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the rest of the proof, we condition on the event that the above inequality holds.

Next we show the first inequality. Note that for any π , we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ Consider π /star . Since ( I -γ ̂ P π ) -1 = ∑ i =0 γ i ( ̂ P π ) i and ( ̂ P π ) i is a probability matrix, we have as desired.

Now we consider the second inequality. Let T be the Bellman optimality operator on M , i.e., for any V ∈ R S

<!-- formula-not-decoded -->

Let ̂ T be the Bellman optimality operator on ̂ M . Further recalling our notations P π and ̂ P π , we have

<!-- formula-not-decoded -->

We observe that these simple bounds are worse than what Theorem 1 posits by a factor of √ 1 / (1 -γ ) , and removing this additional factor requires a significantly more careful analysis as we will see in the remainder of this section.

̂ Solving for ‖ Q /star -̂ Q /star ‖ , we complete the proof.

## 4.2 An s -absorbing MDP M

In order to improve upon the crude bounds in Lemma 5, we would like to directly bound the errors in our value estimates using the componentwise bounds of Lemma 2. Doing so requires an understanding of quantities such as | ( P -̂ P ) ̂ V /star | and | ( P -̂ P ) ̂ V π /star | , which we will do next. However ̂ V /star and ̂ V π /star depend on ̂ P , so that we are not able to directly apply a standard concentration argument. We now address this challenge by providing a method to decouple these dependencies.

For a state s and a scalar u , define the MDP M s,u as follows: M s,u is identical to M except that state s is absorbing in M s,u , i.e. P M s,u ( s | s, a ) = 1 for all a , and the instantaneous reward at state s in M s,u is (1 -γ ) u ; the remainder of the transition model and reward function are identical to those in M . In order to avoid notational clutter, we use V π s,u to denote the value function V π M s,u and correspondingly for Q and reward and transition functions. This implies that for all policies π :

<!-- formula-not-decoded -->

since s is absorbing with instantaneous reward (1 -γ ) u .

For some state s , we will only consider M s,u for u in a finite set U s , where

<!-- formula-not-decoded -->

In particular, we will set U s to consist of evenly spaced elements in this interval, where we set the size of | U s | appropriately later on. As before, we let ̂ M s,u denote the MDP that uses the empirical model ̂ P instead of P , at all non-absorbing states and abbreviate the value functions in M s,u as V π s,u .

<!-- formula-not-decoded -->

̂ ̂ Lemma 6. Fix a state s , an action a , a finite set U s , and δ ≥ 0 . With probability greater than 1 -δ , it holds that for all u ∈ U s ,

Proof. The random variables ̂ P s,a 4 and ̂ V /star s,u are independent. The result now follows from Bernstein's inequality along with a union bound over all U s .

This independence of ̂ P s,a from the value function ̂ V /star s,u is the biggest upshot of our construction. Note that a similar statement does not hold for ̂ V /star . Wenext need to understand how to construct U s so that ̂ V /star s,u provides a good approximation for ̂ V /star , for some u ∈ U s . The following two lemmas provide helpful properties of these absorbing state MDPs to build towards this goal.

Lemma 7. Let u ∗ = V /star M ( s ) and u π = V π M ( s ) . We have

<!-- formula-not-decoded -->

/negationslash

Proof. To prove the first claim, it suffices to verify that V /star M satisfies the Bellman optimality conditions in M s,u /star . To see this, observe that at state s , the Bellman equations are trivially satisfied as s is absorbing with value u /star = V /star M ( s ) at state s by construction. For state s ′ = s , the outgoing transition model at s ′ in M s,u /star is identical to that in M . Since V /star M satisfies the Bellman optimality conditions at state s ′ in M , it must also satisfy Bellman optimality conditions at state s ′ in M . The proof of the second claim is analogous.

This lemma gives a good setting for u , but we also need robustness to misspecification of u as we seek to construct a cover. The next lemma provides this result.

Lemma 8. For all states s , u, u ′ ∈ R , and policies π ,

<!-- formula-not-decoded -->

Proof. First observe

<!-- formula-not-decoded -->

since these two reward functions differ only in state s , in which case r s,u ( s, a ) = (1 -γ ) u and r s,u ′ ( s, a ) = (1 -γ ) u ′ . Let π s,u be the optimal policy in M s,u . Note

<!-- formula-not-decoded -->

where the equality ( a ) follows since P s,u only depends on the state s and not the value u . The proof of the lower bound is analogous, which completes the proof of the first claim. The proof of the second claim can be obtained with a similar argument.

With these two lemmas, we now show the main result of this section.

4 Note that P s,a and ̂ P s,a are from the original MDPs M and ̂ M and not the absorbing versions, as the latter induce degenerate transitions in s for all actions a .

Proposition 1. Fix a state s , an action a , a finite set U s , and δ ≥ 0 . With probability greater than 1 -2 δ , it holds that for all u ∈ U s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By Lemma 6, with probability greater than 1 -δ , we have that for all u ∈ U s .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

using the triangle inequality, √ Var P s,a ( V 1 + V 2 ) ≤ √ Var P s,a ( V 1 ) + √ Var P s,a ( V 2 ) By Lemmas 7 and 8,

.

Since the above holds for all u ∈ U s , we may take the best possible choice, which completes the proof of the first claim. The proof of the second claim is analogous.

The proposition, combined with an accounting of the discretization level yields the following result.

Lemma 9. With probability greater than 1 -δ , where

<!-- formula-not-decoded -->

with c being an absolute constant.

<!-- formula-not-decoded -->

Proof. We take U s to be the evenly spaced elements in the interval [ V /star ( s ) -∆ δ/ 2 ,N V /star ( s ) + ∆ δ/ 2 ,N ] , and we take the size of U s to be | U s | = 1 (1 -γ ) 2 . By Lemma 5, with probability greater than 1 -δ/ 2 , we have V /star ( s ) [ V /star ( s ) ∆ V /star ( s ) + ∆ ] for all s . This implies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used that that ̂ V /star ( s ) will land in one of | U s | -1 evenly sized sub-intervals of length 2∆ δ/ 2 ,N / ( | U s | -1) . Now we use δ/ (2 |S||A| ) , so that the claims in Proposition 1 hold with probability greater than 1 -δ/ 2 for all state action pairs. The first claim follows by substitution and noting that probability of either event failing is less than δ/ 2 . The proof of the second claim is analogous; note that Lemma 5 and Proposition 1 hold simultaneously with regards to the both claims regarding π /star and ̂ π /star so no further modifications to the failure probability are required.

## 4.3 The proof of Theorem 1

Theorem 1 immediately follows from the following lemma combined with Lemma 2.

<!-- formula-not-decoded -->

where c is an absolute constant and where:

Proof. We have:

<!-- formula-not-decoded -->

where (a) uses Lemma 2; (b) is the triangle inequality; (c) uses Lemma 3; (d) uses that ( I -γP ̂ π /star ) -1 has all positive entries.

<!-- formula-not-decoded -->

Focusing on the first term, we see that

<!-- formula-not-decoded -->

Let us now show that Theorem 1 follows from this Lemma. From the condition on ̂ π in the theorem statement, along with Lemma 2, we have where the inequality (c) uses Lemma 9; (d) uses √ Var P ( X + Y ) = √ E P [( X + Y -E P [ X + Y ]) 2 ] ≤ √ Var P ( X ) + √ Var P ( Y ) , by triangle inequality of norms, using √ E P [ Z 2 ] as the norm; (e) uses Lemma 4. Solving for ‖ Q ̂ π -̂ Q ̂ π ‖ ∞ proves the first claim. The proof of the second claim is analogous.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The condition on N in Theorem 1 (for an appropriately chosen absolute contant) implies that α δ,N = γ 1 -γ √ 8 log(8 |S||A| / ((1 -γ ) δ )) N &lt; 1 / 2 . This and Lemma 10 implies:

Plugging in the choice of N in Theorem 1 (where the absolute constant in Theorem 1 need not be the same as that in Lemma 2) completes the proof of the theorem.

## 5 Conclusion

This paper sheds new light on a long-studied basic question in reinforcement learning, which is that of a good approach to planning, given an approximate model of the world. While this is a fundamental question

in itself, previous advances have also resulted in improved algorithms for harder questions such as sampleefficient exploration. For instance, the Bellman structure of variances in an MDP, observed in Azar et al. (2013) has subsequently formed a crucial component of minimax optimal exploration algorithms (Azar et al., 2017; Jin et al., 2018; Zanette and Brunskill, 2019; Wainwright, 2019). We hope that the new technical components in our work can be similarly reused in broader contexts in future work, beyond their utility in analyzing sparse, model-based planning.

## Acknowledgments

Sham Kakade thanks Rong Ge for numerous helpful discussions. We thank Csaba Szepesvari, Kaiqing Zhang, and Mohammad Gheshlaghi Azar for helpful discussions and pointing out typos in the initial version of the paper. S. K. gratefully acknowledges funding from the Washington Research Foundation for Innovation in Data-intensive Discover, the ONR award N00014-18-1-2247, and NSF Award CCF-1703574.

## References

- Azar, M. G., Munos, R., and Kappen, B. (2012). On the sample complexity of reinforcement learning with a generative model. arXiv preprint arXiv:1206.6461 .
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 263-272. JMLR. org.
- Bertsekas, D. P. (1976). Dynamic programming and stochastic control.
- Dann, C. and Brunskill, E. (2015). Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826.
- Jaksch, T., Ortner, R., and Auer, P. (2010). Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873.
- Kakade, S. M. et al. (2003). On the sample complexity of reinforcement learning . PhD thesis, University of London London, England.
- Kalathil, D., Borkar, V. S., and Jain, R. (2014). Empirical q-value iteration. arXiv preprint arXiv:1412.0180 .
- Kearns, M. J. and Singh, S. P. (1999). Finite-sample convergence rates for q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002.
- Osband, I. and Van Roy, B. (2014). Model-based reinforcement learning and the eluder dimension. In Advances in Neural Information Processing Systems , pages 1466-1474.

- Puterman, M. L. (2014). Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp;Sons.
- Sidford, A., Wang, M., Wu, X., Yang, L., and Ye, Y. (2018a). Near-optimal time and sample complexities for solving markov decision processes with a generative model. In Advances in Neural Information Processing Systems 31 , pages 5186-5196. Curran Associates, Inc.
- Sidford, A., Wang, M., Wu, X., and Ye, Y. (2018b). Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM.
- Singh, S. and Yee, R. (1994). An upper bound on the loss from approximate optimal-value functions. Machine Learning , 16(3):227-233.
- Strehl, A. L. (2007). Probably approximately correct (PAC) exploration in reinforcement learning . PhD thesis, Rutgers University-Graduate School-New Brunswick.
- Strehl, A. L., Li, L., and Littman, M. L. (2009). Reinforcement learning in finite mdps: Pac analysis. Journal of Machine Learning Research , 10(Nov):2413-2444.
- Strehl, A. L., Li, L., Wiewiora, E., Langford, J., and Littman, M. L. (2006). Pac model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888. ACM.
- Wainwright, M. J. (2019). Stochastic approximation with cone-contractive operators: Sharp /lscript -infty -bounds for q-learning. arXiv preprint arXiv:1905.06265 .
- Wang, M. (2017). Randomized linear programming solves the discounted Markov decision problem in nearly-linear running time. arXiv preprint arXiv:1704.01869 .
- Ye, Y. (2011). The simplex and policy-iteration methods are strongly polynomial for the Markov decision problem with a fixed discount rate. Mathematics of Operations Research , 36(4):593-603.
- Zanette, A. and Brunskill, E. (2019). Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. arXiv preprint arXiv:1901.00210 .