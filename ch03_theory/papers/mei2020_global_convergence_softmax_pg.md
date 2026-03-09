## On the Global Convergence Rates of Softmax Policy Gradient Methods

Jincheng Mei ♣♠ * Chenjun Xiao ♣ Csaba Szepesv´ ari ♥♣ Dale Schuurmans ♠♣

♣ University of Alberta ♥ DeepMind ♠ Google Research, Brain Team

## Abstract

We make three contributions toward better understanding policy gradient methods in the tabular setting. First, we show that with the true gradient, policy gradient with a softmax parametrization converges at a O (1 /t ) rate, with constants depending on the problem and initialization. This result significantly expands the recent asymptotic convergence results. The analysis relies on two findings: that the softmax policy gradient satisfies a Łojasiewicz inequality, and the minimum probability of an optimal action during optimization can be bounded in terms of its initial value. Second, we analyze entropy regularized policy gradient and show that it enjoys a significantly faster linear convergence rate O ( e -c · t ) toward softmax optimal policy ( c &gt; 0) . This result resolves an open question in the recent literature. Finally, combining the above two results and additional new Ω(1 /t ) lower bound results, we explain how entropy regularization improves policy optimization, even with the true gradient, from the perspective of convergence rate. The separation of rates is further explained using the notion of non-uniform Łojasiewicz degree. These results provide a theoretical understanding of the impact of entropy and corroborate existing empirical studies.

## 1. Introduction

The policy gradient is one of the most foundational concepts in Reinforcement Learning (RL), lying at the core of policysearch and actor-critic methods. This paper is concerned with the analysis of the convergence rate of policy gradient

Work done as an intern at Google Research, Brain Team. This version (v3) generalizes Lemma 8 to multiple optimal actions, i.e., Eq. (260). Jincheng Mei would like to thank Ziheng Wang at University of Oxford, Mathematical Institute for asking related questions. Correspondence to: Jincheng Mei &lt; jmei2@ualberta.ca &gt; .

Proceedings of the 37 th International Conference on Machine Learning , Online, PMLR 119, 2020. Copyright 2020 by the author(s).

methods (Sutton et al., 2000). As an approach to RL, the appeal of policy gradient methods is that they are conceptually straightforward and under some regularity conditions they guarantee monotonic improvement of the value. A secondary appeal is that policy gradient methods were shown to achieve effective empirical performance (e.g., Schulman et al., 2015; 2017).

Despite the prevalence and importance of policy optimization in RL, the theoretical understanding of policy gradient method has, until recently, been severely limited. A key barrier to understanding is the inherent non-convexity of the value landscape with respect to standard policy parametrizations. As a result, little has been known about the global convergence behavior of policy gradient method. Recently, important new progress in understanding the convergence behavior of policy gradient has been achieved. As in this paper we will restrict ourselves to the tabular setting, we analyze the part of the literature that also deals with this setting. While the tabular setting is clearly limiting, this is the setting where so far the cleanest results have been achieved and understanding this setting is a necessary first step towards the bigger problem of understanding RL algorithms. Returning to the discussion of recent work, Bhandari &amp; Russo (2019) showed that, without parametrization, projected gradient ascent on the simplex does not suffer from spurious local optima. In concurrent work, Agarwal et al. (2019) showed that (i) without parametrization, projected gradient ascent converges at rate O (1 / √ t ) to a global optimum; and (ii) with softmax parametrization, policy gradient converges asymptotically. Agarwal et al. also analyze other variants of policy gradient, and show that policy gradient with relative entropy regularization converges at rate O (1 / √ t ) , natural policy gradient (mirror descent) converges at rate O (1 /t ) , and given a 'compatible' function approximation (thus, going beyond the tabular case) natural policy gradient converges at rate O (1 / √ t ) . Shani et al. (2020) obtains the slower rate O (1 / √ t ) for mirror descent. They also proposed a variant that adds entropy regularization and prove a rate of O (1 /t ) for this modified problem.

Despite these advances, many open questions remain in understanding the behavior of policy gradient methods, even in the tabular setting and even when the true gradient is avail-

able in the updates. In this paper, we provide answers to the following three questions left open by previous work in this area: (i) What is the convergence rate of policy gradient methods with softmax parametrization? The best previous result, due to Agarwal et al. (2019), established asymptotic convergence but gave no rates. (ii) What is the convergence rate of entropy regularized softmax policy gradient? Figuring out the answer to this question was explicitly stated as an open problem by Agarwal et al. (2019). (iii) Empirical results suggest that entropy helps optimization (Ahmed et al., 2019). Can this empirical observation be turned into a rigorous theoretical result? 1

First , we prove that with the true gradient, policy gradient methods with a softmax parametrization converge to the optimal policy at a O (1 /t ) rate, with constants depending on the problem and initialization. This result significantly strengthens the recent asymptotic convergence results of Agarwal et al. (2019). Our analysis relies on two novel findings: (i) that softmax policy gradient satisfies what we call a non-uniform Łojasiewicz-type inequality with the constant in the inequality depending on the optimal action probability under the current policy; (ii) the minimum probability of an optimal action during optimization can be bounded in terms of its initial value. Combining these two findings, with a few other properties we describe, it can be shown that softmax policy gradient method achieves a O (1 /t ) convergence rate.

Second , we analyze entropy regularized policy gradient and show that it enjoys a linear convergence rate of O ( e -t ) toward the softmax optimal policy, which is significantly faster than that of the unregularized version. This result resolves an open question in Agarwal et al. (2019), where the authors analyzed a more aggressive relative entropy regularization rather than the more common entropy regularization. A novel insight is that entropy regularized gradient updates behave similarly to the contraction operator in value learning, with a contraction factor that depends on the current policy.

Third , we provide a theoretical understanding of entropy regularization in policy gradient methods. (i) We prove a new lower bound of Ω(1 /t ) for softmax policy gradient, implying that the upper bound of O (1 /t ) that we established, apart from constant factors, is unimprovable. This result also provides a theoretical explanation of the optimization advantage of entropy regularization: even with access to the true gradient, entropy helps policy gradient converge faster than any achievable rate of softmax policy gradient method without regularization . (ii) We study the concept of non-uniform Łojasiewicz degree and show that, without

1 While Shani et al. (2020) suggest that entropy regularization speeds up mirror descent to achieve the rate of O (1 /t ) , in light of the corresponding result of Agarwal et al. (2019) who established the same rate for the unregularized version of mirror descent, their conclusion needs further support (e.g., lower bounds).

regularization, the Łojasiewicz degree of expected reward cannot be positive, which allows O (1 /t ) rates to be established. We then show that with entropy regularization, the Łojasiewicz degree of maximum entropy reward becomes 1 / 2 , which is sufficient to obtain linear O ( e -t ) rates. This change of the relationship between gradient norm and suboptimality reveals a deeper reason for the improvement in convergence rates. The theoretical study we provide corroborates existing empirical studies on the impact of entropy in policy optimization (Ahmed et al., 2019).

The remainder of the paper is organized as follows. After introducing notation and defining the setting in Section 2, we present the three main contributions in Sections 3 to 5 as aforementioned. Section 6 gives our conclusions.

## 2. Notations and Settings

For a finite set X , we use ∆( X ) to denote the set of probability distributions over X . A finite Markov decision process (MDP) M = ( S , A , P , r, γ ) is determined by a finite state space S , a finite action space A , transition function P : S × A → ∆( S ) , reward function r : S × A → R , and discount factor γ ∈ [0 , 1) . Given a policy π : S → ∆( A ) , the value of state s under π is defined as

<!-- formula-not-decoded -->

We also let V π ( ρ ) := E s ∼ ρ [ V π ( s )] , where ρ ∈ ∆( S ) is an initial state distribution. The state-action value of π at ( s, a ) ∈ S × A is defined as

<!-- formula-not-decoded -->

We let A π ( s, a ) := Q π ( s, a ) -V π ( s ) be the so-called advantage function of π . The (discounted) state distribution of π is defined as

<!-- formula-not-decoded -->

and we let d π ρ ( s ) := E s 0 ∼ ρ [ d π s 0 ( s ) ] . Given ρ , there exists an optimal policy π ∗ such that

<!-- formula-not-decoded -->

We denote V ∗ ( ρ ) := V π ∗ ( ρ ) for conciseness. Since S × A is finite, for convenience, without loss of generality, we assume that the one step reward lies in the [0 , 1] interval:

Assumption 1 (Bounded reward) . r ( s, a ) ∈ [0 , 1] , ∀ ( s, a ) .

The softmax transform of a vector exponentiates the components of the vector and normalizes it so that the result lies in the simplex. This can be used to transform vectors assigned to state-action pairs into policies:

Softmax transform. Given the function θ : S × A → R , the softmax transform of θ is defined as π θ ( ·| s ) := softmax( θ ( s, · )) , where for all a ∈ A ,

<!-- formula-not-decoded -->

Due to its origin in logistic regression, we call the values θ ( s, a ) the logit values and the function θ itself a logit function. We also extend this notation to the case when there are no states: For θ : [ K ] → R , we define π θ := softmax( θ ) using π θ ( a ) = exp { θ ( a ) } / ∑ a ′ exp { θ ( a ′ ) } ( a ∈ [ K ] ).

Hmatrix. Given any distribution π over [ K ] , let H ( π ) := diag ( π ) -ππ glyph[latticetop] ∈ R K × K , where diag ( x ) ∈ R K × K is the diagonal matrix that has x ∈ R K at its diagonal. The H matrix will play a central role in our analysis because H ( π θ ) is the Jacobian of the θ ↦→ π θ := softmax( θ ) map that maps R [ K ] to the ( K -1) -simplex:

<!-- formula-not-decoded -->

Here, we are using the standard convention that derivatives give row-vectors. Finally, we recall the definition of smoothness from convex analysis:

Smoothness. A function f : Θ → R with Θ ⊂ R d is β -smooth (w.r.t. glyph[lscript] 2 norm, β &gt; 0 ) if for all θ , θ ′ ∈ Θ ,

<!-- formula-not-decoded -->

## 3. Policy Gradient

Policy gradient is a special policy search method. In policy search, one considers a family of policies parametrized by finite-dimensional parameter vectors, reducing the search for a good policy to searching in the space of parameters. This search is usually accomplished by making incremental changes (additive updates) to the parameters. Representative policy-based RL methods include REINFORCE (Williams, 1992), natural policy gradient (Kakade, 2002), deterministic policy gradient (Silver et al., 2014), and trust region policy optimization (Schulman et al., 2015). In policy gradient methods, the parameters are updated by following the gradient of the map that maps policy parameters to values. Under mild conditions, the gradient can be reexpressed in a convenient form in terms of the policy's action-value function and the gradients of the policy parametrization:

Theorem 1 (Policy gradient theorem (Sutton et al., 2000)) . Fix a map θ ↦→ π θ ( a | s ) that for any ( s, a ) is differentiable and fix an initial distribution µ ∈ ∆( S ) . Then,

<!-- formula-not-decoded -->

## 3.1. Vanilla Softmax Policy Gradient

We focus on the policy gradient method that uses the softmax parametrization. Since we consider the tabular case, the policy is then parametrized using the logit θ : S × A → R function and π θ ( ·| s ) = softmax( θ ( s, · )) . The vanilla form of policy gradient for this case is shown in Algorithm 1.

## Algorithm 1 Policy Gradient Method

Input: Learning rate η &gt; 0 .

Initialize logit θ 1 ( s, a ) for all ( s, a ) .

<!-- formula-not-decoded -->

With some calculation, Theorem 1 can be used to show that the gradient takes the following special form in this case:

Lemma 1. Softmax policy gradient w.r.t. θ

is

<!-- formula-not-decoded -->

Due to space constraints, the proof of this, as well as of all the remaining results are given in the appendix. While this lemma was known (Agarwal et al., 2019), we included a proof for the sake of completeness.

Recently, Agarwal et al. (2019) showed that softmax policy gradient asymptotically converges to π ∗ , i.e., V π θ t ( ρ ) → V ∗ ( ρ ) as t → ∞ provided that µ ( s ) &gt; 0 holds for all states s ∈ S . We strengthen this result to show that the rate of convergence (in terms of value sub-optimality) is O (1 /t ) . The next section is devoted to this result. For better accessibility, we start with the result for the bandit case which presents an opportunity to explaining the main ideas underlying our result in a clean fashion.

## 3.2. Convergence Rates

## 3.2.1. THE INSTRUCTIVE CASE OF BANDITS

As promised, in this section we consider 'bandit case': In particular, assume that the MDP has a single state and the discount factor γ is zero: γ = 0 . In this case, Eq. (1) reduces to maximizing the expected reward,

<!-- formula-not-decoded -->

With π θ = softmax( θ ) , even in this simple setting, the objective is non-concave in θ , as shown by a simple example: Proposition 1. On some problems, θ ↦→ E a ∼ π θ [ r ( a )] is a non-concave function over R K .

As γ = 0 and there is a single state, Lemma 1 simplifies to

<!-- formula-not-decoded -->

Putting things together, we see that in this case the update in Algorithm 1 takes the following form:

Update 1 (Softmax policy gradient, expected reward) . θ t +1 ( a ) ← θ t ( a ) + η · π θ t ( a ) · ( r ( a ) -π glyph[latticetop] θ t r ) , ∀ a ∈ [ K ] .

As is well known, if a function is smooth, then a small gradient update will be guaranteed to improve the objective value. As it turns out, for the softmax parametrization, the expected reward objective is β -smooth with β ≤ 5 / 2 :

Lemma 2 (Smoothness) . ∀ r ∈ [0 , 1] K , θ ↦→ π glyph[latticetop] θ r is 5 / 2 -smooth.

Smoothness alone (as is also well known) is not sufficient to guarantee that gradient updates converge to a global optimum. For non-concave objectives, the next best thing to guarantee convergence to global maxima is to establish that the gradient of the objective at any parameter dominates the sub-optimality of the parameter. Inequalities of this form are known as a Łojasiewicz inequality (Łojasiewicz, 1963). The reason gradient dominance helps is because it prevents the gradient vanishing before reaching a maximum. The objective function of our problem also satisfies such an inequality, although of a weaker, 'non-uniform' form. For the following result, for simplicity, we assume that the optimal action is unique. This assumption can be lifted with a little extra work, which is discussed at the end of this section.

Lemma 3 (Non-uniform Łojasiewicz) . Assume r has one unique maximizing action a ∗ . Let π ∗ = arg max π ∈ ∆ π glyph[latticetop] r . Then,

<!-- formula-not-decoded -->

The weakness of this inequality is that the right-hand side scales with π θ ( a ∗ ) - hence we call it non-uniform. As a result, Lemma 3 is not very useful if π θ t ( a ∗ ) , the optimal action's probability, becomes very small during the updates.

Nevertheless, the inequality still suffices to get an following intermediate result. The proof of this result combines smoothness and the Łojasiewicz inequality we derived.

Lemma 4 (Pseudo-rate) . Let c t = min 1 ≤ s ≤ t π θ s ( a ∗ ) . Using Update 1 with η = 2 / 5 , for all t ≥ 1 ,

<!-- formula-not-decoded -->

In the remainder of this section we assume that η = 2 / 5 .

Remark 1. The value of π θ t ( a ∗ ) , while it is nonzero (and so is c t ) can be small (e.g., because of the choice of θ 1 ). Consequently, its minimum c t can be quite small and the upper bound in Lemma 4 can be large, or even vacuous. The dependence of the previous result on π θ t ( a ∗ ) comes from Lemma 3. As it turns out, it is not possible to eliminate or improve the dependence on π θ ( a ∗ ) in Lemma 3. To see this consider r = (5 , 4 , 4) glyph[latticetop] , π θ = (2 glyph[epsilon1], 1 / 2 -2 glyph[epsilon1], 1 / 2) where glyph[epsilon1] &gt; 0 is small number. By algebra, ( π ∗ -π θ ) glyph[latticetop] r = 1 -2 glyph[epsilon1] &gt; 1 / 2 , dπ glyph[latticetop] θ r dθ = (2 glyph[epsilon1] -4 glyph[epsilon1] 2 , -glyph[epsilon1] +4 glyph[epsilon1] 2 , -glyph[epsilon1] ) glyph[latticetop] , ∥ ∥ ∥ dπ glyph[latticetop] θ r dθ ∥ ∥ ∥ 2 = glyph[epsilon1] · √ 6 -24 glyph[epsilon1] +32 glyph[epsilon1] 2 ≤ 3 glyph[epsilon1] . Hence, for any constant C &gt; 0 ,

<!-- formula-not-decoded -->

which means for any Łojasiewicz-type inequality, C necessarily depends on glyph[epsilon1] and hence on π θ ( a ∗ ) = 2 glyph[epsilon1] .

The necessary dependence on π θ t ( a ∗ ) makes it clear that Lemma 4 is insufficient to conclude a O (1 /t ) rate. since c t may vanish faster than O (1 /t ) as t increases. Our next result eliminates this possibility. In particular, the result follows from the asymptotic convergence result of Agarwal et al. (2019) which states that π θ t ( a ∗ ) → 1 as t →∞ . From this and because π θ ( a ) &gt; 0 for any θ ∈ R K and action a , we conclude that π θ t ( a ∗ ) remains bounded away from zero during the course of the updates:

Lemma 5. We have inf t ≥ 1 π θ t ( a ∗ ) &gt; 0 .

With some extra work, one can also show that eventually θ t enters a region where π θ t ( a ∗ ) can only increase:

Proposition 2. For any initialization there exist t 0 ≥ 1 such that for any t ≥ t 0 , t ↦→ π θ t ( a ∗ ) is increasing. In particular, when π θ 1 is the uniform distribution, t 0 = 1 .

With Lemmas 4 and 5, we can now obtain an O (1 /t ) convergence rate for softmax policy gradient method 2 :

Theorem 2 (Arbitrary initialization) . Using Update 1 with η = 2 / 5 , for all t ≥ 1 ,

<!-- formula-not-decoded -->

where c = inf t ≥ 1 π θ t ( a ∗ ) &gt; 0 is a constant that depends on r and θ 1 , but it does not depend on the time t .

Proposition 2 suggests that one should set θ 1 so that π θ 1 is uniform. Using this initialization, we can show that inf t ≥ 1 π θ t ( a ∗ ) ≥ 1 /K , strengthening Theorem 2:

Theorem 3 (Uniform initialization) . Using Update 1 with η = 2 / 5 and θ 1 such that π θ 1 ( a ) = 1 /K , ∀ a , for all t ≥ 1 ,

<!-- formula-not-decoded -->

2 For a continuous version of Update 1, Walton (2020) proves a O (1 /t ) rate, using a Lyapunov function argument.

Figure 1. Visualization of proof idea for Lemma 5.

<!-- image -->

<!-- image -->

Remark 2. In Section 5, we prove a lower bound Ω(1 /t ) for the same update rule, showing that the upper bound O (1 /t ) of Theorem 2, apart from constant factors, is unimprovable.

In general it is difficult to characterize how the constant C in Theorem 2 depends on the problem and initialization. For the simple 3 -armed case, this dependence is relatively clear:

Lemma 6. Let r (1) &gt; r (2) &gt; r (3) . Then, a ∗ = 1 and inf t ≥ 1 π θ t ( a ∗ ) = min 1 ≤ t ≤ t 0 π θ t (1) , where

<!-- formula-not-decoded -->

Note that the smaller r (1) -r (2) and π θ 1 (1) are, the larger t 0 is, which potentially means c in Theorem 2 can be smaller.

Visualization. Let r = (1 . 0 , 0 . 9 , 0 . 1) glyph[latticetop] . In Fig. 1(a), the region below the red line corresponds to R = { π θ : π θ (1) /π θ (3) ≥ ( r (2) -r (3)) / (2 · ( r (1) -r (2))) } . Any globally convergent iteration will enter R within finite time (the closure of R contains π ∗ ) and never leaves R (this is the main idea in Lemma 5). Subfigure (b) shows the behavior of the gradient updates with 'good' ( π θ 1 = (0 . 05 , 0 . 01 , 0 . 94) glyph[latticetop] ) and 'bad' ( π θ 1 = (0 . 01 , 0 . 05 , 0 . 94) glyph[latticetop] ) initial policies. While these are close to each other, the iterates behave quite differently (in both cases η = 2 / 5 ). From the good initialization, the iterates converge quickly: after 100 iterations the distance to the optimal policy is already quite small. At the same time, starting from a 'bad' initial value, the iterates are first attracted toward a sub-optimal action. It takes more than 7000 iterations for the algorithm to escape this sub-optimal corner! In subfigure (c), we see that π θ t ( a ∗ ) increases for the good initialization, while in subfigure (d), for the bad initialization, we see that it initially decreases. These experiments confirm that the dependence of the error bound in Theorem 2 on the initial values cannot be removed.

Non-unique optimal actions. When the optimal action is non-unique, the arguments need to be slightly modified. Instead of using a single π θ ( a ∗ ) , we need to consider ∑ a ∗ ∈A ∗ π θ ( a ∗ ) , i.e., the sum of probabilities of all optimal actions. Details are given in the appendix.

## 3.2.2. GENERAL MDPS

For general MDPs, the optimization problem takes the form

<!-- formula-not-decoded -->

Here, as before, π θ ( ·| s ) = softmax( θ ( s, · )) , s ∈ S . Following Agarwal et al. (2019), the values here are defined with respect to an initial state distribution ρ which may not be the same as the initial state distribution µ used in the gradient updates (cf. Algorithm 1), allowing for greater flexibility in our analysis. While the initial state distributions do not play any role in the bandit case, here, in the multi-state case, they have a strong influence. In particular, for the rest of this section, we will assume that the initial state distribution µ used in the gradient updates is bounded away from zero:

Assumption 2 (Sufficient exploration) . The initial state distribution satisfies min s µ ( s ) &gt; 0 .

Assumption 2 was also adapted by Agarwal et al. (2019), which ensures 'sufficient exploration' in the sense that the occupancy measure d π µ of any policy π when started from µ will be guaranteed to be positive over the whole state space. Agarwal et al. (2019) asked whether this assumption is necessary for convergence to global optimality.

Proposition 3. There exists an MDP and µ with min s µ ( s ) = 0 such that there exists θ ∗ : S × A → [0 , ∞ ] such that θ ∗ is the stationary point of θ ↦→ V π θ ( µ ) while π θ ∗ is not an optimal policy. Furthermore, this stationary point is an attractor, hence, starting gradient ascent in a small enough vicinity of θ ∗ will make it converge to θ ∗ .

The MDP of this proposition is S bandit problems: Each state s ∈ S under each action deterministically gives itself as the next state. The reward is selected so that in each s there is a unique optimal action. If µ leaves out state s (i.e., µ ( s ) = 0 ), clearly, the gradient of θ ↦→ V π θ ( µ ) w.r.t. θ is zero regardless of the choice of θ . Hence, any θ such that θ ( s, a ) = + ∞ for a optimal in state s with µ ( s ) &gt; 0 and θ ( s, a ) finite otherwise will satisfy the properties of the proposition. It remains open whether the sufficient exploration condition is necessary for unichain MDPs.

According to Assumption 1, r ( s, a ) ∈ [0 , 1] , Q ( s, a ) ∈ [0 , 1 / (1 -γ )] , and hence the objective function is still smooth, as was also shown by Agarwal et al. (2019):

Lemma 7 (Smoothness) . V π θ ( ρ ) is 8 / (1 -γ ) 3 -smooth.

As mentioned in Section 3.2.1, smoothness and (uniform) Łojasiewicz inequality are sufficient to prove a convergence rate. As noted by Agarwal et al. (2019), the main difficulty is to establish a (uniform) Łojasiewicz inequality for softmax parametrization. As it turns out, the results from the bandit case carry over to multi-state MDPs.

For stating this and the remaining results, we fix a deterministic optimal policy π ∗ and denote by a ∗ ( s ) the action that π ∗ selects in state s . With this, the promised result on the non-uniform Łojasiewicz inequality is as follows:

Lemma 8 (Non-uniform Łojasiewicz) . We have,

<!-- formula-not-decoded -->

By Assumption 2, d π θ µ is also bounded away from zero on the whole state space and thus the multiplier of the suboptimality in the above inequality is positive.

Generalizing Lemma 5, we show that min s π θ t ( a ∗ ( s ) | s ) is uniformly bounded away from zero:

Lemma 9. Let Assumption 2 hold. Using Algorithm 1, we have, c := inf s ∈S ,t ≥ 1 π θ t ( a ∗ ( s ) | s ) &gt; 0 .

Using Lemmas 7 to 9, we prove that softmax policy gradient converges to an optimal policy at a O (1 /t ) rate in MDPs, just like what we have seen in the bandit case:

Theorem 4. Let Assumption 2 hold and let { θ t } t ≥ 1 be generated using Algorithm 1 with η = (1 -γ ) 3 / 8 , c the positive constant from Lemma 9. Then, for all t ≥ 1 ,

<!-- formula-not-decoded -->

As far as we know, this is the first convergence-rate result for softmax policy gradient for MDPs.

Remark 3. Theorem 4 implies that the iteration complexity of Algorithm 1 to achieve O ( glyph[epsilon1] ) sub-optimality is O ( S c 2 (1 -γ ) 6 glyph[epsilon1] · ∥ ∥ ∥ d π ∗ µ µ ∥ ∥ ∥ 2 ∞ · ∥ ∥ ∥ 1 µ ∥ ∥ ∥ ∞ ) , which, as a function of glyph[epsilon1] , is better than the results of Agarwal et al. (2019) for (i) projected gradient ascent on the simplex ( O ( SA (1 -γ ) 6 glyph[epsilon1] 2 · ∗

∥ ∥ ∥ d π ρ µ ∥ ∥ ∥ 2 ∞ ) ) or for (ii) softmax policy gradient with relativeentropy regularization ( O ( S 2 A 2 (1 -γ ) 6 glyph[epsilon1] 2 · ∥ ∥ ∥ d π ∗ ρ µ ∥ ∥ ∥ 2 ∞ ) ). The improved dependence on glyph[epsilon1] (or t ) in our result follows from Lemmas 8 and 9 and a different proof technique utilized to prove Theorem 4, while we pay a price because our bound depends on c , which adds an extra dependence on the MDP as well as on the initialization of the algorithm.

## 4. Entropy Regularized Policy Gradient

Agarwal et al. (2019) considered relative-entropy regularization in policy gradient to get an O (1 / √ t ) convergence rate. As they note, relative-entropy is more 'agressive' in penalizing small probabilities than the more 'common' entropy regularizer (cf. Remark 5.5 in their paper) and it remains unclear whether this latter regularizer leads to an algorithm with the same rate. In this section, we answer this positively and in fact prove a much better rate. In particular, we show that entropy regularized policy gradient with the softmax parametrization enjoys a linear rate of O ( e -t ) . In retrospect, perhaps this is unsurprising as entropy regularization bears a strong similarity to introducing a strongly convex regularizer in convex optimization, where this change is known to significantly improve the rate of convergence of first-order methods (e.g., Nesterov, 2018, Chapter 2).

## 4.1. Maximum Entropy RL

In entropy regularized RL, or sometimes called maximum entropy RL, near-deterministic policies are penalized (Williams &amp; Peng, 1991; Mnih et al., 2016; Nachum et al., 2017; Haarnoja et al., 2018; Mei et al., 2019), which is achieved by modifying the value of a policy π to

<!-- formula-not-decoded -->

where H ( ρ, π ) is the 'discounted entropy', defined as

<!-- formula-not-decoded -->

and τ ≥ 0 , the 'temperature', determines the strength of the penalty. 3 Clearly, the value of any policy can be obtained by adding an entropy penalty to the rewards (as proposed originally by Williams &amp; Peng (1991)). Hence, similarly to Lemma 1, one can obtain the following expression for the gradient of the entropy regularized objective under the softmax policy parametrization:

Lemma 10. It holds that for all ( s, a ) ,

<!-- formula-not-decoded -->

3 To better align with naming conventions in information-theory, discounted entropy should be rather called the discounted actionentropy rate as entropy itself in the literature on Markov chain information theory would normally refer to the entropy of the stationary distribution of the chain, while entropy rate refers to what is being used here.

where ˜ A π θ ( s, a ) is the 'soft' advantage function defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 4.2. Convergence Rates

As in the non-regularized case, to gain insight, we first consider MDPs with a single state and γ = 0 .

## 4.2.1. BANDIT CASE

In the one-state case with γ = 0 , Eq. (15) reduces to maximizing the entropy-regularized reward,

<!-- formula-not-decoded -->

Again, Eq. (20) is a non-concave function of θ . In this case, regularized policy gradient reduces to

<!-- formula-not-decoded -->

where H ( π θ ) is the same as in Eq. (6). Using the above gradient in Algorithm 1 we have the following update rule: Update 2 (Softmax policy gradient, maximum entropy reward) . θ t +1 ← θ t + η · H ( π θ t )( r -τ log π θ t ) .

Due to the presence of regularization, the optimal solution will be biased with the bias disappearing as τ → 0 :

Softmax optimal policy. π ∗ τ := softmax( r/τ ) is the optimal solution of Eq. (20).

Remark 4. At this stage, we could use arguments similar to those of Section 3 to show the O (1 /t ) convergence of π θ t to π ∗ τ . However, we can use an alternative idea to show that entropy-regularized policy gradient converges significantly faster. The issue of bias will be discussed later.

Our alternative idea is to show that Update 2 defines a contraction but with a contraction coefficient that depends on the parameter that the update is applied to:

Lemma 11 (Non-uniform contraction) . Using Update 2 with τη ≤ 1 , ∀ t &gt; 0 ,

<!-- formula-not-decoded -->

where ζ t := τθ t -r -( τθ t -r ) glyph[latticetop] 1 K · 1 .

This lemma immediately implies the following bound:

Lemma 12. Using Update 2 with τη ≤ 1 , ∀ t &gt; 0 ,

<!-- formula-not-decoded -->

Similarly to Lemma 5, we can show that the minimum action probability can be lower bounded by its initial value.

Lemma 13. There exists c = c ( τ, K, ‖ θ 1 ‖ ∞ ) &gt; 0 , such that for all t ≥ 1 , min a π θ t ( a ) ≥ c . Thus, ∑ t -1 s =1 min a π θ s ( a ) ≥ c · ( t -1) .

A closed-form expression for c is given in the appendix. Note that when τ = 0 (no regularization), the result would no longer hold true. The key here is that min a π θ t ( a ) → min a π ∗ τ ( a ) &gt; 0 as t →∞ and the latter inequality holds thanks to τ &gt; 0 . From Lemmas 12 and 13, it follows that entropy regularized softmax policy gradient enjoys a linear convergence rate:

Theorem 5. Using Update 2 with η ≤ 1 /τ , for all t ≥ 1 ,

<!-- formula-not-decoded -->

where ˜ δ t := π ∗ τ glyph[latticetop] ( r -τ log π ∗ τ ) -π glyph[latticetop] θ t ( r -τ log π θ t ) and c &gt; 0 is from Lemma 13.

## 4.2.2. GENERAL MDPS

For general MDPs, the problem is to maximize ˜ V π θ ( ρ ) in Eq. (15). The softmax optimal policy π ∗ τ is known to satisfy the following consistency conditions (Nachum et al., 2017):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using a somewhat lengthy calculation, we show that the discounted entropy in Eq. (16) is smooth:

Lemma 14 (Smoothness) . H ( ρ, π θ ) is (4 + 8 log A ) / (1 -γ ) 3 -smooth, where A := |A| is the total number of actions.

Our next key result shows that the augmented value function ˜ V π θ ( ρ ) satisfies a 'better type' of Łojasiewicz inequality:

Lemma 15 (Non-uniform Łojasiewicz) . Suppose µ ( s ) &gt; 0 for all state s ∈ S . Then,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The main difference to the previous versions of the nonuniform Łojasiewicz inequality is that the sub-optimality gap appears under the square root. For small sub-optimality gaps this means that the gradient must be larger - a stronger 'signal'. Next, we show that action probabilities are still uniformly bounded away from zero:

Lemma16. Using Algorithm 1 with the entropy regularized objective, we have c := inf t ≥ 1 min s,a π θ t ( a | s ) &gt; 0 .

With Lemmas 14 to 16, we show a O ( e -t ) rate for entropy regularized policy gradient in general MDPs:

Theorem 6. Suppose µ ( s ) &gt; 0 for all state s . Using Algorithm 1 with the entropy regularized objective and softmax parametrization and η = (1 -γ ) 3 / (8 + τ (4 + 8 log A )) , there exists a constant C &gt; 0 such that for all t ≥ 1 ,

<!-- formula-not-decoded -->

The value of the constant C in this theorem appears in the proof of the result in the appendix in a closed form.

## 4.2.3. CONTROLLING THE BIAS

As noted in Remark 4, π ∗ τ is biased, i.e., π ∗ τ = π ∗ for fixed τ &gt; 0 . We discuss two possible approaches to deal with the bias, but much remains to be done to properly address the bias. For simplicity, we consider the bandit case.

glyph[negationslash]

glyph[negationslash]

A two-stage approach. Note that for any fixed τ &gt; 0 , π ∗ τ ( a ∗ ) ≥ π ∗ τ ( a ) for all a = a ∗ . Therefore, using policy gradient with π θ 1 = π ∗ τ , we have π θ t ( a ∗ ) ≥ c t ≥ 1 /K . This suggests a two-stage method: first, to ensure π θ t ( a ∗ ) ≥ max a π θ t ( a ) , use entropy-regularized policy gradient some iterations and then turn off regularization.

glyph[negationslash]

Theorem 7. Denote ∆ = r ( a ∗ ) -max a = a ∗ r ( a ) &gt; 0 . Using Update 2 for t 1 ∈ O ( e 1 /τ · log ( τ +1 ∆ )) iterations and then Update 1 for t 2 ≥ 1 iterations, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This approach removes the nasty dependence on the choice of the initial parameters. While this dependence is also removed if we initialize with the uniform policy, uniform initialization is insufficient if only noisy estimates of the gradients are available. However, we leave the study of this case for future work. An obvious problem with this approach is that ∆ is unknown. This can be helped by exiting the first phase when we detect 'convergence' e.g. by detecting that the relative change of the policy is small.

Decreasing the penalty. Another simple idea is to decrease the strength of regularization, e.g., set τ t ∈ O (1 / log t ) . Consider the following update, which is a slight variation of the previous one:

<!-- formula-not-decoded -->

The rationale for the scaling factor is that it allows one to prove a variant of Lemma 11. While this is promising, the proof cannot be finished as before. The difficulty is that π θ t → π ∗ (which is what we want to achieve) implies that min a π θ t ( a ) → 0 , which prevents the use of our previous proof technique. We show the following partial results.

Theorem 8. Using Update 3 with τ t = α · ∆ log t for t ≥ 2 , where α &gt; 0 , and η t = 1 /τ t , we have, for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The final rates then depend on how fast min a π θ t ( a ) diminishes as function of t . We conjecture that the rate in some cases degenerates to O ( log t t 1 /α ) , which is strictly faster than O (1 /t ) in non-regularized case when α ∈ (0 , 1) and is observed in simulations in the appendix. We leave it as an open problem to study decaying entropy in general MDPs.

## 5. Does Entropy Regularization Really Help?

The previous section indicated that entropy regularization may speed up convergence. In addition, ample empirical evidence suggest that this may be the case (e.g., Williams &amp; Peng, 1991; Mnih et al., 2016; Nachum et al., 2017; Haarnoja et al., 2018; Mei et al., 2019). In this section, we aim to provide new insights into why entropy may help policy optimization, taking an optimization perspective.

We start by establishing a lower bound that shows that the O (1 /t ) rate we established earlier for softmax policy gradient without entropy regularization cannot be improved. Next, we introduce the notion of Łojasiewicz degree, which we show to increase in the presence of entropy regularization. We then connect a higher degree to faster convergence rates. Note that our proposal to view entropy regularization as an optimization aid is somewhat conflicting with the more common explanation that entropy regularization helps by encouraging exploration. While it is definitely true that entropy regularization encourages exploration, the form of exploration it encourages is not sensitive to epistemic uncertainty and as such it fails to provide a satisfactory solution to the exploration problem (e.g., O'Donoghue et al., 2020).

## 5.1. Lower Bounds

The purpose of this section is to establish that the O (1 /t ) rates established earlier for unpenalized policy gradient is tight. To get lower bounds, we need to show that progress in every iteration cannot be too large. This holds when we can reverse the inequality in the Łojasiewicz inequality. To this regard, in bandit problems we have the following result:

Lemma 17 (Reversed Łojasiewicz) . Take any r ∈ [0 , 1] K .

glyph[negationslash]

Denote ∆ = r ( a ) -max a = a ∗ r ( a ) &gt; 0

<!-- formula-not-decoded -->

Using this result gives the desired lower bound:

Theorem 9 (Lower bound) . Take any r ∈ [0 , 1] K . For large enough t ≥ 1 , using Update 1 with learning rate η t ∈ (0 , 1] ,

<!-- formula-not-decoded -->

Note that Theorem 9 is a special case of general MDPs. Next, we strengthen this result and show that the Ω(1 /t ) lower bound also holds for any MDP:

Theorem 10 (Lower bound) . Take any MDP. For large enough t ≥ 1 , using Algorithm 1 with η t ∈ (0 , 1] ,

<!-- formula-not-decoded -->

glyph[negationslash]

where ∆ ∗ := min s ∈S ,a = a ∗ ( s ) { Q ∗ ( s, a ∗ ( s )) -Q ∗ ( s, a ) } &gt; 0 is the optimal value gap of the MDP.

Remark 5. Our convergence rates in Section 3 match the lower bounds up to constant. However, the constant gap is large, e.g., K 2 in Theorem 3, and ∆ 2 in Theorem 9. The gap is because the reversed Łojasiewicz inequality of Lemma 17 uses ∆ , which is unavoidable when π θ is close to π ∗ . We leave it as an open problem to close this gap.

With the lower bounds established, we confirm that entropy regularization helps policy optimization by speeding up convergence, though the question remains as to the mechanism by which the improved convergence rate manifests itself.

## 5.2. Non-uniform Łojasiewicz Degree

To gain further insight into how entropy regularization helps, we introduce the non-uniform Łojasiewicz degree:

Definition 1 (Non-uniform Łojasiewicz degree) . A function f : X → R has Łojasiewicz degree ξ ∈ [0 , 1] if 4

<!-- formula-not-decoded -->

∀ x ∈ X , where C ( x ) &gt; 0 holds for all x ∈ X .

The uniform degree, where C ( x ) is a positive constant, has previously been connected to convergence speed in the optimization literature. B´ arta (2017) studied this effect for first-, while Nesterov &amp; Polyak (2006); Zhou et al. (2018) studied this for second-order methods. As noted beforehand, a larger degree (smaller exponent of the sub-optimality) is expected to improve the convergence speed of algorithms

4 In literature (Łojasiewicz, 1963), C cannot depend on x . Based on the examples we have seen, we relax this requirement.

that rely on gradient information. Intuitively, we expect this to continue to hold for the non-uniform Łojasiewicz degree as well. With this, we now study what Łojasiewicz degrees can one obtain with and without entropy regularization.

Our first result shows that the Łojasiewicz degree of the expected reward objective (in bandits) cannot be positive:

Proposition 4. Let r ∈ [0 , 1] K be arbitrary and consider θ ↦→ E a ∼ π θ [ r ( a )] . The non-uniform Łojasiewicz degree of this map with constant C ( θ ) = π θ ( a ∗ ) is zero.

Note that according to Remark 1, it is necessary that C ( θ ) depends on π θ ( a ∗ ) . The difference between Proposition 4 and the reversed Łojasiewicz inequality of Lemma 17 is subtle. Lemma 17 is a condition that implies impossibility to get rates faster than O (1 /t ) , while Proposition 4 says it is not sufficient to get rates faster than O (1 /t ) using the same technique as in Lemma 4 . However, this does not preclude that other techniques could give faster rates.

Next, we show that the Łojasiewicz degree of the entropyregularized expected reward objective is at least 1 / 2 :

Proposition 5. Fix τ &gt; 0 . With C ( θ ) = √ 2 τ · min a π θ ( a ) , the Łojasiewicz degree of θ ↦→ E a ∼ π θ [ r ( a ) -τ log π θ ( a )] is at least 1 / 2 .

## 6. Conclusions and Future Work

We set out to study the convergence speed of softmax policy gradient methods with and without entropy regularization in the tabular setting. Here, the error is measured in terms of the sub-optimality of the policy obtained after some number of updates. Our main findings is that without entropy regularization, the rate is Θ(1 /t ) , which is faster than rates previously obtained. Our analysis also uncovered an unpleasant dependence on the initial parameter values. With entropy regularization, the rate becomes linear, where now the constant in the exponent is influenced by the initial choice of parameters. Thus, our analysis shows that entropy regularization substantially changes the rate at which gradient methods converge. Our main technical innovation is the introduction of a non-uniform variant of the Łojasiewicz inequality. Our work leaves open a number of interesting questions: While we have some lower bounds, there remains some gaps to be filled between the lower and upper bounds. Other interesting directions are extending the results for alternative (e.g., restricted) policy parametrizations or studying policy gradient when the gradient must be estimated from data. One also expects that non-uniform Łojasiewicz inequalities and the Łojasiewicz degree could also be put to good use in other areas of non-convex optimization.

## Acknowledgements

Jincheng Mei would like to thank Bo Dai and Lihong Li for helpful discussions and for providing feedback on a draft of this manuscript. Jincheng Mei would like to thank Ruitong Huang for enlightening early discussions. Csaba Szepesv´ ari gratefully acknowledges funding from the Canada CIFAR AI Chairs Program, Amii and NSERC.

## References

- Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. Optimality and approximation with policy gradient methods in Markov decision processes, 2019.
- Ahmed, Z., Le Roux, N., Norouzi, M., and Schuurmans, D. Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning , pp. 151-160, 2019.
- B´ arta, T. Rate of convergence to equilibrium and Łojasiewicz-type estimates. Journal of Dynamics and Differential Equations , 29(4):1553-1568, 2017.
- Bhandari, J. and Russo, D. Global optimality guarantees for policy gradient methods, 2019.
- Golub, G. H. Some modified matrix eigenvalue problems. SIAM Review , 15(2):318-334, 1973.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning , pp. 1861-1870, 2018.
- Kakade, S. and Langford, J. Approximately optimal approximate reinforcement learning. In ICML , volume 2, pp. 267-274, 2002.
- Kakade, S. M. A natural policy gradient. In Advances in neural information processing systems , pp. 1531-1538, 2002.
- Łojasiewicz, S. Une propri´ et´ e topologique des sousensembles analytiques r´ eels. Les ´ equations aux d´ eriv´ ees partielles , 117:87-89, 1963.
- Mei, J., Xiao, C., Huang, R., Schuurmans, D., and M¨ uller, M. On principled entropy exploration in policy optimization. In Proceedings of the 28th International Joint Conference on Artificial Intelligence , pp. 3130-3136. AAAI Press, 2019.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pp. 19281937, 2016.
- Nachum, O., Norouzi, M., Xu, K., and Schuurmans, D. Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems , pp. 2775-2785, 2017.
- Nesterov, Y. Lectures on convex optimization , volume 137. Springer, 2018.
- Nesterov, Y. and Polyak, B. T. Cubic regularization of Newton method and its global performance. Mathematical Programming , 108(1):177-205, 2006.
- O'Donoghue, B., Osband, I., and Ionescu, C. Making sense of reinforcement learning and probabilistic inference. arXiv preprint arXiv:2001.00805 , 2020.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International conference on machine learning , pp. 1889-1897, 2015.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- Shani, L., Efroni, Y., and Mannor, S. Adaptive trust region policy optimization: Global convergence and faster rates for regularized MDPs. In AAAI , 2020.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. Deterministic policy gradient algorithms. In International Conference on Machine Learning , pp. 387-395, 2014.
- Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , pp. 1057-1063, 2000.
- Walton, N. A short note on soft-max and policy gradients in bandits problems. arXiv preprint arXiv:2007.10297 , 2020.
- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256, 1992.
- Williams, R. J. and Peng, J. Function optimization using connectionist reinforcement learning algorithms. Connectionist Science , 3(3):241-268, 1991.
- Xiao, C., Huang, R., Mei, J., Schuurmans, D., and M¨ uller, M. Maximum entropy monte-carlo planning. In Advances in Neural Information Processing Systems , pp. 9516-9524, 2019.
- Zhou, Y., Wang, Z., and Liang, Y. Convergence of cubic regularization for nonconvex optimization under KL property. In Advances in Neural Information Processing Systems , pp. 3760-3769, 2018.

The appendix is organized as follows.

- Appendix A: proofs for the technical results in the main paper.
- -Appendix A.1: proofs for the results of softmax policy gradient in Section 3.
* Appendix A.1.1: Preliminaries.
* Appendix A.1.2: One-state MDPs (bandits).
* Appendix A.1.3: General MDPs.
- -Appendix A.2: proofs for the results of entropy regularized softmax policy gradient in Section 4.
* Appendix A.2.1: Preliminaries.
* Appendix A.2.2: One-state MDPs (bandits).
* Appendix A.2.3: General MDPs.
* Appendix A.2.4: Two-stage and decaying entropy regularization.
- -Appendix A.3: proofs for Section 5 (does entropy regularization really help?).
* Appendix A.3.1: One-state MDPs (bandits).
* Appendix A.3.2: General MDPs.
* Appendix A.3.3: Non-uniform Łojasiewicz degree.
- Appendix B: miscellaneous extra supporting results that are not mentioned in the main paper.
- Appendix C: further remarks on sub-optimality guarantees for other entropy-based RL methods beyond those presented in the main paper.
- Appendix D: simulation results to verify the convergence rates, which are not presented in the main paper.

## A. Proofs

## A.1. Proofs for Section 3: softmax parametrization

## A.1.1. PRELIMINARIES

Lemma 1. Consider the map θ ↦→ V π θ ( µ ) where θ ∈ R S×A and π θ ( ·| s ) = softmax( θ ( s, · )) . The derivative of this map satisfies

<!-- formula-not-decoded -->

Note that this is given as Agarwal et al. (2019, Lemma C.1); we include a proof for completeness.

Proof. According to the policy gradient theorem (Theorem 1),

<!-- formula-not-decoded -->

For s ′ = s , ∂π θ ( a | s ′ ) ∂θ ( s, · ) = 0 since π θ ( a | s ′ ) does not depend on θ ( s, · ) . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

Since H ( π θ ( ·| s )) = diag ( π θ ( ·| s )) -π θ ( ·| s ) π θ ( ·| s ) glyph[latticetop] , for each component a , we have

<!-- formula-not-decoded -->

## A.1.2. PROOFS FOR SOFTMAX PARAMETRIZATION IN BANDITS

Proposition 1. On some problems, θ ↦→ E a ∼ π θ [ r ( a )] is a non-concave function over R K .

Proof. Consider the following example: r = (1 , 9 / 10 , 1 / 10) glyph[latticetop] , θ 1 = (0 , 0 , 0) glyph[latticetop] , π θ 1 = softmax( θ 1 ) = (1 / 3 , 1 / 3 , 1 / 3) glyph[latticetop] , θ 2 = (ln 9 , ln 16 , ln 25) glyph[latticetop] , and π θ 2 = softmax( θ 2 ) = (9 / 50 , 16 / 50 , 25 / 50) glyph[latticetop] . We have,

<!-- formula-not-decoded -->

On the other hand, defining ¯ θ = 1 2 · ( θ 1 + θ 2 ) = (ln 3 , ln 4 , ln 5) glyph[latticetop] we have π ¯ θ = softmax( ¯ θ ) = (3 / 12 , 4 / 12 , 5 / 12) glyph[latticetop] and

<!-- formula-not-decoded -->

Since 1 2 · ( π glyph[latticetop] θ 1 r + π glyph[latticetop] θ 2 r ) &gt; π glyph[latticetop] ¯ θ r , θ ↦→ E a ∼ π θ ( · ) [ r ( a )] is a non-concave function of θ .

Lemma 2 (Smoothness) . Let π θ = softmax( θ ) and π θ ′ = softmax( θ ′ ) . For any r ∈ [0 , 1] K , θ ↦→ π glyph[latticetop] θ r is 5 / 2 -smooth, i.e.,

<!-- formula-not-decoded -->

Proof. Let S := S ( r, θ ) ∈ R K × K be the second derivative of the value map θ ↦→ π glyph[latticetop] θ r . By Taylor's theorem, it suffices to show that the spectral radius of S (regardless of r and θ ) is bounded by 5 / 2 . Now, by its definition we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Continuing with our calculation fix i, j ∈ [ K ] . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

is Kronecker's δ -function. To show the bound on the spectral radius of S , pick y ∈ R K . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where glyph[circledot] is Hadamard (component-wise) product, and the last inequality uses H¨ older's inequality together with the triangle inequality. Note that ‖ y glyph[circledot] y ‖ 1 = ‖ y ‖ 2 2 , ‖ π θ ‖ 1 = 1 , and ‖ y ‖ ∞ ≤ ‖ y ‖ 2 . For i ∈ [ K ] , denote by H i, : ( π θ ) the i -th row of H ( π θ ) as a row vector. Then, glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand,

Therefore we have, finishing the proof.

Lemma 3 (Non-uniform Łojasiewicz) . Assume r has a single maximizing action a ∗ . Let π ∗ := arg max π ∈ ∆ π glyph[latticetop] r , and π θ = softmax( θ ) . Then, for any θ ,

<!-- formula-not-decoded -->

When there are multiple optimal actions, we have

<!-- formula-not-decoded -->

where A ∗ = { a ∗ : r ( a ∗ ) = max a r ( a ) } is the set of optimal actions.

Proof. We give the proof for the general case, as the case of a single maximizing action is a corollary to this case. Using the expression we got for the gradient earlier,

<!-- formula-not-decoded -->

For the remaining results in this section, for simplicity, we assume that A ∗ = { a ∗ } , i.e., there is a unique optimal action a ∗ . Lemma 4 (Pseudo-rate) . Let π θ t = softmax( θ t ) , and c t = min 1 ≤ s ≤ t π θ s ( a ∗ ) . Using Update 1 with η = 2 / 5 , for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. According to Lemma 2,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let δ t = ( π ∗ -π θ t ) glyph[latticetop] r . To prove the first part, we need to show that δ t ≤ 5 c 2 t · 1 t holds for any t ≥ 1 . We prove this by induction on t .

Base case: Since δ t ≤ 1 and c t ∈ (0 , 1) , the result trivially holds up to t ≤ 5 .

Inductive step: Now, let t ≥ 2 and suppose that δ t ≤ 5 c 2 t · 1 t . Consider f t : R → R defined using f t ( x ) = x -c 2 t 5 · x 2 . We which implies

<!-- formula-not-decoded -->

which is equivalent to

have that f t is monotonically increasing in [ 0 , 5 2 · c 2 t ] . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the induction and the proof of the first part of the lemma.

For the second part, summing up δ t ≤ 5 c 2 t · 1 t ≤ 5 c 2 T · 1 t , we have

<!-- formula-not-decoded -->

On the other hand, rearranging Eq. (78) and summing up δ 2 t ≤ 5 c 2 t · ( δ t -δ t +1 ) ≤ 5 c 2 T · ( δ t -δ t +1 ) from t = 1 to T ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 5. For η = 2 / 5 , we have inf t ≥ 1 π θ t ( a ∗ ) &gt; 0 .

Proof. Let and

Therefore, by Cauchy-Schwarz,

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

denote the reward gap of r . We will prove that inf t ≥ 1 π θ t ( a ∗ ) = min 1 ≤ t ≤ t 0 π θ t ( a ∗ ) , where t 0 = min { t : π θ t ( a ∗ ) ≥ c c +1 } . Note that t 0 depends only on θ 1 and c , and c depends only on the problem. Define the following regions, glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We make the following three-part claim.

## Claim 1. The following hold:

- a) R 1 is a 'nice' region, in the sense that if θ t ∈ R 1 then, with any η &gt; 0 , following a gradient update (i) θ t +1 ∈ R 1 and (ii) π θ t +1 ( a ∗ ) ≥ π θ t ( a ∗ ) .
- b) We have R 2 ⊂ R 1 and N c ⊂ R 1 .
- c) For η = 2 / 5 , there exists a finite time t 0 ≥ 1 , such that θ t 0 ∈ N c , and thus θ t 0 ∈ R 1 , which implies that inf t ≥ 1 π θ t ( a ∗ ) = min 1 ≤ t ≤ t 0 π θ t ( a ∗ ) .

Claim a) Part (i): We want to show that if θ t ∈ R 1 , then θ t +1 ∈ R 1 . Let

<!-- formula-not-decoded -->

Note that R 1 = ∩ a = a ∗ R 1 ( a ) . Pick a = a ∗ . Clearly, it suffices to show that if θ t ∈ R 1 ( a ) then θ t +1 ∈ R 1 ( a ) . Hence, suppose that θ t ∈ R 1 ( a ) . We consider two cases.

glyph[negationslash]

glyph[negationslash]

Case (a): π θ t ( a ∗ ) ≥ π θ t ( a ) . Since π θ t ( a ∗ ) ≥ π θ t ( a ) , we also have θ t ( a ∗ ) ≥ θ t ( a ) . After an update of the parameters,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that π θ t +1 ( a ∗ ) ≥ π θ t +1 ( a ) . Since r ( a ∗ ) -π glyph[latticetop] θ t +1 r &gt; 0 and r ( a ∗ ) &gt; r ( a ) ,

<!-- formula-not-decoded -->

which is equivalent to dπ glyph[latticetop] θ t +1 r dθ t +1 ( a ∗ ) ≥ dπ glyph[latticetop] θ t +1 r dθ t +1 ( a ) , i.e., θ t +1 ∈ R 1 ( a ) .

Case (b): Suppose now that π θ t ( a ∗ ) &lt; π θ t ( a ) . First note that for any θ and a = a ∗ , θ ∈ R 1 ( a ) holds if and only if glyph[negationslash]

<!-- formula-not-decoded -->

Indeed, from the condition dπ glyph[latticetop] θ r dθ ( a ∗ ) ≥ dπ glyph[latticetop] θ r dθ ( a ) , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, after rearranging, is equivalent to Eq. (98). Hence, it suffices to show that Eq. (98) holds for θ t +1 provided it holds for θ t .

From the latter condition, we get

<!-- formula-not-decoded -->

After an update of the parameters, according to the ascent lemma for smooth function (Lemma 18), π glyph[latticetop] θ t +1 r ≥ π glyph[latticetop] θ t r , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand,

which implies that which is equivalent to

<!-- formula-not-decoded -->

Furthermore, by our assumption that π θ t ( a ∗ ) &lt; π θ t ( a ) , we have 1 -exp { θ t ( a ∗ ) -θ t ( a ) } = 1 -π θ t ( a ∗ ) π θ t ( a ) &gt; 0 . Putting things together, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and thus by our previous remark, θ t +1 ∈ R 1 ( a ) , thus, finishing the proof of part (i).

glyph[negationslash]

Part (ii): Assume again that θ t ∈ R 1 . We want to show that π θ t +1 ( a ∗ ) ≥ π θ t ( a ∗ ) . Since θ t ∈ R 1 , we have dπ glyph[latticetop] θ t r dθ t ( a ∗ ) ≥ dπ glyph[latticetop] θ t r dθ t ( a ) , ∀ a = a ∗ . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim b) We start by showing that R 2 ⊂ R 1 . For this, let θ ∈ R 2 , i.e., π θ ( a ∗ ) ≥ π θ ( a ) . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, θ ∈ R 1 and thus R 2 ⊂ R 1 as desired.

Now, let us prove that N c ⊂ R 1 . Take θ ∈ N c . We want to show that θ ∈ R 1 . If θ ∈ R 2 , by R 2 ⊂ R 1 , we also have that θ ∈ R 1 . Hence, it remains to show that θ ∈ R 1 holds when θ ∈ N c and θ glyph[negationslash]∈ R 2 .

glyph[negationslash]

Thus, take any θ that satisfies these two conditions. Pick a = a ∗ . It suffices to show that θ ∈ R 1 ( a ) . Without loss of

generality, assume that a ∗ = 1 and a = 2 . Then, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equation is because

<!-- formula-not-decoded -->

glyph[negationslash]

the first inequality is by 0 &lt; r (1) -r ( i ) ≤ 1 and the second inequality is because of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Plugging ∑ K i =3 π θ ( i ) = 1 -π θ (1) -π θ (2) into Eq. (116) and rearranging the resulting expression we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that θ ∈ R 1 ( a ) , thus, finishing the proof.

Claim c) We claim that π θ t ( a ∗ ) → 1 as t →∞ . For this, we wish to use the asymptotic convergence results of Agarwal et al. (2019, Theorem 5.1), which states this, but the stepsize there is η ≤ 1 / 5 while here we have η = 2 / 5 . We claim that their asymptotic result still hold with the larger η . In fact, the restriction on η comes from that they can only prove the ascent lemma (Lemma 18) for η ≤ 1 / 5 . Other than this, their proof does not rely on the choice of η . Since we can prove the ascent lemma with η ≤ 2 / 5 (and in particular with η = 2 / 5 ), their result continues to hold even with η = 2 / 5 .

Thus, π θ t ( a ∗ ) → 1 as t → ∞ . Hence, there exists t 0 ≥ 1 , such that π θ t 0 ( a ∗ ) ≥ c c +1 , which means θ t 0 ∈ N c ⊂ R 1 . According to the first part in our proof, i.e., once θ t is in R 1 , following gradient update θ t +1 will be in R 1 , and π θ t ( a ∗ ) is increasing in R 1 , we have inf t π θ t ( a ∗ ) = min 1 ≤ t ≤ t 0 π θ t ( a ∗ ) . t 0 depends on initialization and c , which only depends on the problem.

Proposition 2. For any initialization there exist t 0 ≥ 1 such that for any t ≥ t 0 , t ↦→ π θ t ( a ∗ ) is increasing. In particular, when π θ 1 is the uniform distribution, t 0 = 1 .

Proof. We have t 0 = min { t ≥ 1 : π θ t ( a ∗ ) ≥ c c +1 } , where c = K 2∆ · ( 1 -∆ K ) in the proof for Lemma 5 satisfies for any t ≥ t 0 , t ↦→ π θ t ( a ∗ ) is increasing.

Now, let θ 1 be so that π θ 1 is the uniform distribution. We show that t 0 = 1 . Recall from Claim 1 that R 2 is the region where the probability of the optimal action exceeds that of the suboptimal ones and R 1 is the region where the gradient of the optimal action exceeds those of the suboptimal ones and that R 2 ⊂ R 1 . Clearly, θ 1 ∈ R 2 and hence also θ 1 ∈ R 1 . Now, by Part a) of Claim 1, R 1 is invariant under the updates, showing that t 0 = 1 holds as required.

Theorem 2 (Arbitrary initialization) . Using Update 1 with η = 2 / 5 , for all t ≥ 1 ,

<!-- formula-not-decoded -->

where c = inf t ≥ 1 π θ t ( a ∗ ) &gt; 0 is a constant that depends on r and θ 1 , but it does not depend on the time t .

Proof. According to Lemmas 4 and 5, the claim immediately holds, with c = inf t ≥ 1 π θ t ( a ∗ ) &gt; 0 .

Theorem 3 (Uniform initialization) . Using Update 1 with η = 2 / 5 and π θ 1 ( a ) = 1 /K , ∀ a , for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Since the initial policy is uniform policy, π θ 1 ( a ∗ ) ≥ 1 /K . According to Proposition 2, for all t ≥ t 0 = 1 , t ↦→ π θ t ( a ∗ ) is increasing. Hence, we have π θ t ( a ∗ ) ≥ 1 /K , ∀ t ≥ 1 , and c t = min 1 ≤ s ≤ t π θ s ( a ∗ ) ≥ 1 /K . According to Lemma 4,

<!-- formula-not-decoded -->

we have ( π ∗ -π θ t ) glyph[latticetop] r ≤ 5 K 2 /t , ∀ t ≥ 1 . The remaining results follow from Eq. (70) and c T ≥ 1 /K .

Lemma 6. Let r (1) &gt; r (2) &gt; r (3) . Then, a ∗ = 1 and inf t ≥ 1 π θ t (1) = min 1 ≤ t ≤ t 0 π θ t (1) , where

<!-- formula-not-decoded -->

In general, for K -action bandit cases, let r (1) &gt; r (2) &gt; · · · &gt; r ( K ) , we have, glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Proof. 3 -action case. Recall the definition of R 1 from the proof for Lemma 5:

glyph[negationslash]

<!-- formula-not-decoded -->

By Part a) of Claim 1, it suffices to prove that θ ∈ R 1 . Thus, our goal is to show that any θ such that π θ (1) π θ (3) ≥ r (2) -r (3) 2 · ( r (1) -r (2)) is in fact an element of R 1 . Suppose π θ (1) π θ (3) ≥ r (2) -r (3) 2 · ( r (1) -r (2)) . There are two cases.

Case (a): If π θ (1) π θ (3) ≥ r (2) -r (3) r (1) -r (2) , then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that since r (1) &gt; π glyph[latticetop] θ r , and r (3) &lt; π glyph[latticetop] θ r , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ (2) and dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ (3) , i.e., θ ∈ R 1 .

Case (b): If r (2) -r (3) 2 · ( r (1) -r (2)) ≤ π θ (1) π θ (3) &lt; r (2) -r (3) r (1) -r (2) , then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equation is according to

<!-- formula-not-decoded -->

and the second inequality is because of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the last inequality is from

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies,

Now we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ (2) . According to Eq. (141), we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ (3) . Therefore we have θ ∈ R 1 .

glyph[negationslash]

glyph[negationslash]

K -action case. Suppose for each action i ∈ { 2 , 3 , . . . K -1 } , π θ (1) ≥ ∑ j =1 ,j = i π θ ( j ) · ( r ( i ) -r ( j )) 2 · ( r (1) -r ( i )) . There are two cases.

glyph[negationslash]

glyph[negationslash]

Case (a): If π θ (1) ≥ ∑ j =1 ,j = i π θ ( j ) · ( r ( i ) -r ( j )) r (1) -r ( i ) , then we have,

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

which implies, for all i ∈ { 2 , 3 , . . . K -1 } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar with Eq. (141), since r (1) &gt; π glyph[latticetop] θ r , and r ( K ) &lt; π glyph[latticetop] θ r , we have dπ

glyph[latticetop]

θ

dθ

r

(1)

dπ

glyph[latticetop]

θ

dθ

(

r

K

)

-

=

π

θ

(1)

·

(

r

(1)

-

π

glyph[latticetop]

θ

r

)

-

π

θ

(

K

)

·

(

r

(

K

)

-

π

glyph[latticetop]

θ

r

)

(160)

<!-- formula-not-decoded -->

Therefore we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ ( i ) , for all i ∈ { 2 , 3 , . . . K } , i.e., θ ∈ R 1 .

glyph[negationslash]

Case (b): If ∑ j =1 ,j = i π θ ( j ) · ( r ( i ) -r ( j )) 2 · ( r (1) -r ( i )) ≤ π θ (1) &lt; ∑ j =1 ,j = i π θ ( j ) · ( r ( i ) -r ( j )) r (1) -r ( i ) , then we have, for all i ∈ { 2 , 3 , . . . K -1 } , glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

where the second equation is according to

<!-- formula-not-decoded -->

and the first inequality is by r (1) -π glyph[latticetop] θ r &gt; 0 and,

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

and the second inequality is because of glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

and the last inequality is from ∑ j =1 ,j = i π θ ( j ) · ( r ( i ) -r ( j )) r (1) -r ( i ) &gt; π θ (1) &gt; 0 and, glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Now we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ ( i ) , for all i ∈ { 2 , 3 , . . . K -1 } . According to Eq. (160), we have dπ glyph[latticetop] θ r dθ (1) ≥ dπ glyph[latticetop] θ r dθ ( K ) . Therefore we have θ ∈ R 1 .

## A.1.3. PROOFS FOR SOFTMAX PARAMETRIZATION IN MDPS

Lemma 7 (Smoothness) . V π θ ( ρ ) is 8 / (1 -γ ) 3 -smooth.

Proof. See Agarwal et al. (2019, Lemma E.4). Our proof is for completeness. Denote θ α = θ + αu , where α ∈ R and u ∈ R SA . For any s ∈ S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, glyph[negationslash]

Since ∂π θ ( a | s ) ∂θ ( s ′ , · ) = 0 , for s ′ = s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

Let S ( a, θ ) = ∂ 2 π θ ( a | s ) ∂θ 2 ( s, · ) ∈ R A × A . ∀ i, j ∈ [ A ] , the value of S ( a, θ ) is,

<!-- formula-not-decoded -->

where the δ notation is as defined in Eq. (50). Then we have,

<!-- formula-not-decoded -->

Therefore we have,

<!-- formula-not-decoded -->

Define P ( α ) ∈ R S × S , where ∀ ( s, s ′ ) ,

<!-- formula-not-decoded -->

The derivative w.r.t. α is

<!-- formula-not-decoded -->

For any vector x ∈ R S , we have

<!-- formula-not-decoded -->

The glyph[lscript] ∞ norm is upper bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, taking second derivative w.r.t. α ,

<!-- formula-not-decoded -->

The glyph[lscript] ∞ norm is upper bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, consider the state value function of π θ α ,

<!-- formula-not-decoded -->

which implies, where

and r θ α ∈ R S for s ∈ S is given by

Since [ P ( α )] ( s,s ′ ) ≥ 0 , ∀ ( s, s ′ ) , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have [ M ( α )] ( s,s ′ ) ≥ 0 , ∀ ( s, s ′ ) . Denote [ M ( α )] i, : as the i -th row vector of M ( α ) . We have

<!-- formula-not-decoded -->

which implies, ∀ i ,

<!-- formula-not-decoded -->

Therefore, for any vector x ∈ R S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Assumption 1, r ( s, a ) ∈ [0 , 1] , ∀ ( s, a ) . We have,

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

Therefore we have,

Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly to Eq. (59), the glyph[lscript] 1 norm is upper bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking derivative w.r.t. α in Eq. (202),

<!-- formula-not-decoded -->

Taking second derivative w.r.t. α ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the last term,

For the second last term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the first term, according to Eq. (192), Eqs. (208) and (211),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Eqs. (230), (233), (238) and (243) with Eq. (228),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies for all y ∈ R SA and θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote θ ξ = θ + ξ ( θ ′ -θ ) , where ξ ∈ [0 , 1] . According to Taylor's theorem, ∀ s , ∀ θ, θ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since V π θ ( s ) is 8 / (1 -γ ) 3 -smooth, for any state s , V π θ ( ρ ) = E s ∼ ρ [ V π θ ( s )] is also 8 / (1 -γ ) 3 -smooth.

Lemma 8 (Non-uniform Łojasiewicz) . Let π θ ( ·| s ) = softmax( θ ( s, · )) , s ∈ S and fix an arbitrary optimal policy π ∗ . We have,

<!-- formula-not-decoded -->

where a ∗ ( s ) = arg max a π ∗ ( a | s ) ( s ∈ S ). Furthermore,

<!-- formula-not-decoded -->

where ¯ A π ( s ) = { ¯ a ( s ) ∈ A : Q π ( s, ¯ a ( s )) = max a Q π ( s, a ) } is the greedy action set for state s given policy π . Finally,

<!-- formula-not-decoded -->

where A ∗ ( s ) is the 'optimal action set' under state s ∈ S , defined by,

<!-- formula-not-decoded -->

and π ∗ θ is the globally optimal policy induced by π θ , where for all s ∈ S ,

<!-- formula-not-decoded -->

Proof. We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define the distribution mismatch coefficient as ∥ ∥ ∥ ∥ d π ∗ ρ d π θ µ ∥ ∥ ∥ ∥ ∞ = max s d π ∗ ρ ( s ) d π θ µ ( s ) . We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the one but last equality used that π ∗ is deterministic and in state s chooses a ∗ ( s ) with probability one, and the last equality uses the performance difference formula (Lemma 19).

To prove the second claim, given a policy π , define the greedy action set for each state s ,

<!-- formula-not-decoded -->

By similar arguments that were used in the first part, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is because for any ¯ a ( s ) ∈ ¯ A π θ ( s ) we have

<!-- formula-not-decoded -->

which is the same value across all ¯ a ( s ) ∈ ¯ A π θ ( s ) . Then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equation is again according to Lemma 19.

To prove the third claim, using the similar arguments in the second part, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking minimum over all states, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(290)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equation is because of π ∗ θ is an optimal policy, for all s ∈ S ,

<!-- formula-not-decoded -->

thus finishing the proofs.

Lemma 9. Let Assumption 2 hold. Using Algorithm 1, we have c := inf s ∈S ,t ≥ 1 π θ t ( a ∗ ( s ) | s ) &gt; 0 .

Proof. The proof is an extension of the proof for Lemma 5. Denote ∆ ∗ ( s ) = Q ∗ ( s, a ∗ ( s )) -max a = a ∗ ( s ) Q ∗ ( s, a ) &gt; 0 as the optimal value gap of state s , where a ∗ ( s ) is the action that the optimal policy selects under state s , and ∆ ∗ = min s ∈S ∆ ∗ ( s ) &gt; 0 as the optimal value gap of the MDP. For each state s ∈ S , define the following sets:

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly to the previous proof, we have the following claims:

- Claim I. R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) is a 'nice' region, in the sense that, following a gradient update, (i) if θ t ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , then θ t +1 ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) ; while we also have (ii) π θ t +1 ( a ∗ ( s ) | s ) ≥ π θ t ( a ∗ ( s ) | s ) .
- Claim II. N c ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) ⊂ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) .
- Claim III. There exists a finite time t 0 ( s ) ≥ 1 , such that θ t 0 ( s ) ∈ N c ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , and thus θ t 0 ( s ) ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , which implies inf t ≥ 1 π θ t ( a ∗ ( s ) | s ) = min 1 ≤ t ≤ t 0 ( s ) π θ t ( a ∗ ( s ) | s ) .
- Claim IV. Define t 0 = max s t 0 ( s ) . Then, we have inf s ∈S ,t ≥ 1 π θ t ( a ∗ ( s ) | s ) = min 1 ≤ t ≤ t 0 min s π θ t ( a ∗ ( s ) | s ) .

Clearly, claim IV suffices to prove the lemma since for any θ , min s,a π θ ( a | s ) &gt; 0 . In what follows we provide the proofs of these four claims.

Claim I. First we prove part (i) of the claim. If θ t ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , then θ t +1 ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) . Suppose θ t ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) . We have θ t +1 ∈ R 3 ( s ) by the definition of R 3 ( s ) . We have,

<!-- formula-not-decoded -->

According to smoothness arguments as Eq. (345), we have V π θ t +1 ( s ′ ) ≥ V π θ t ( s ′ ) , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

which means θ t +1 ∈ R 2 ( s ) . Next we prove θ t +1 ∈ R 1 ( s ) . Note that ∀ a = a ∗ ( s ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using similar arguments we also have Q π θ t +1 ( s, a ∗ ( s )) -Q π θ t +1 ( s, a ) ≥ ∆ ∗ ( s ) / 2 . According to Lemma 1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, since ∂V π θ t ( µ ) ∂θ t ( s,a ∗ ( s )) ≥ ∂V π θ t ( µ ) ∂θ t ( s,a ) , we have

<!-- formula-not-decoded -->

Similarly to the first part in the proof for Lemma 5. There are two cases.

Case (a): If π θ t ( a ∗ ( s ) | s ) ≥ π θ t ( a | s ) , then θ t ( s, a ∗ ( s )) ≥ θ t ( s, a ) . After an update of the parameters,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies π θ t +1 ( a ∗ ( s ) | s ) ≥ π θ t +1 ( a | s ) . Since Q π θ t +1 ( s, a ∗ ( s )) -Q π θ t +1 ( s, a ) ≥ ∆ ∗ ( s ) / 2 ≥ 0 , ∀ a , we have Q π θ t +1 ( s, a ∗ ( s )) -V π θ t +1 ( s ) = Q π θ t +1 ( s, a ∗ ( s )) -∑ a π θ t +1 ( a | s ) · Q π θ t +1 ( s, a ) ≥ 0 , and

<!-- formula-not-decoded -->

which is equivalent to ∂V π θ t +1 ( µ ) ∂θ t +1 ( s,a ∗ ( s )) ≥ ∂V π θ t +1 ( µ ) ∂θ t +1 ( s,a ) , i.e., θ t +1 ∈ R 1 ( s ) .

Case (b): If π θ t ( a ∗ ( s ) | s ) &lt; π θ t ( a | s ) , then by ∂V π θ t ( µ ) ∂θ t ( s,a ∗ ( s )) ≥ ∂V π θ t ( µ ) ∂θ t ( s,a ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, after rearranging, is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since θ t +1 ∈ R 3 ( s ) , we have,

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Furthermore, since 1 -exp { θ t ( s, a ∗ ( s )) -θ t ( s, a ) } = 1 -π θ t ( a ∗ ( s ) | s ) π θ t ( a | s ) &gt; 0 (in this case π θ t ( a ∗ ( s ) | s ) &lt; π θ t ( a | s )) ,

<!-- formula-not-decoded -->

which after rearranging is equivalent to

<!-- formula-not-decoded -->

which means ∂V π θ t +1 ( µ ) ∂θ t +1 ( s,a ∗ ( s )) ≥ ∂V π θ t +1 ( µ ) ∂θ t +1 ( s,a ) i.e., θ t +1 ∈ R 1 ( s ) . Now we have (i) if θ t ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , then θ t +1 ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) .

glyph[negationslash]

Let us now turn to proving part (ii). We have π θ t +1 ( a ∗ ( s ) | s ) ≥ π θ t ( a ∗ ( s ) | s ) . If θ t ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , then ∂V π θ t ( µ ) ∂θ t ( s,a ∗ ( s )) ≥ ∂V π θ t ( µ ) ∂θ t ( s,a ) , ∀ a = a ∗ . After an update of the parameters,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim II. N c ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) ⊂ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) . Suppose θ ∈ R 2 ( s ) ∩ R 3 ( s ) and π θ ( a ∗ ( s ) | s ) ≥ c ( s ) c ( s )+1 . There are two cases.

glyph[negationslash]

Case (a): If π θ ( a ∗ ( s ) | s ) ≥ max a = a ∗ ( s ) { π θ ( a | s ) } , then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

where the inequality is since Q π θ ( s, a ∗ ( s )) -Q π θ ( s, a ) ≥ ∆ ∗ ( s ) / 2 &gt; 0 , ∀ a = a ∗ ( s ) , similarly to Eq. (304).

glyph[negationslash]

Case (b): π θ ( a ∗ ( s ) | s ) &lt; max a = a ∗ ( s ) { π θ ( a | s ) } , which is not possible. Suppose there exists an a = a ∗ ( s ) , such that π θ ( a ∗ ( s ) | s ) &lt; π θ ( a | s ) . Then we have the following contradiction, glyph[negationslash]

<!-- formula-not-decoded -->

where the last inequality is according to A ≥ 2 (there are at least two actions), and ∆ ∗ ( s ) ≤ 1 / (1 -γ ) .

Claim III. (1) According to the asymptotic convergence results of Agarwal et al. (2019, Theorem 5.1), which we can use thanks to Assumption 2, π θ t ( a ∗ ( s ) | s ) → 1 . Hence, there exists t 1 ( s ) ≥ 1 , such that π θ t 1 ( s ) ( a ∗ ( s ) | s ) ≥ c ( s ) c ( s )+1 . (2) Q π θ t ( s, a ∗ ( s )) → Q ∗ ( s, a ∗ ( s )) , as t →∞ . There exists t 2 ( s ) ≥ 1 , such that Q π θ t 2 ( s ) ( s, a ∗ ( s )) ≥ Q ∗ ( s, a ∗ ( s )) -∆ ∗ ( s ) / 2 . (3) Q π θ t ( s, a ∗ ( s )) → V ∗ ( s ) , and V π θ t ( s ) → V ∗ ( s ) , as t → ∞ . There exists t 3 ( s ) ≥ 1 , such that ∀ t ≥ t 3 ( s ) , Q π θ t ( s, a ∗ ( s )) -V π θ t ( s ) ≤ ∆ ∗ ( s ) / 2 .

Define t 0 ( s ) = max { t 1 ( s ) , t 2 ( s ) , t 3 ( s ) } . We have θ t 0 ( s ) ∈ N c ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , and thus θ t 0 ( s ) ∈ R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) . According to the first part in our proof, i.e., once θ t is in R 1 ( s ) ∩ R 2 ( s ) ∩ R 3 ( s ) , following gradient update θ t +1 will be in R 1 ( s ) ∩R 2 ( s ) ∩R 3 ( s ) , and π θ t ( a ∗ ( s ) | s ) is increasing in R 1 ( s ) ∩R 2 ( s ) ∩R 3 ( s ) , we have inf t π θ t ( a ∗ ( s ) | s ) = min 1 ≤ t ≤ t 0 ( s ) π θ t ( a ∗ ( s ) | s ) . t 0 ( s ) depends on initialization and c ( s ) , which only depends on the MDP and state s .

<!-- formula-not-decoded -->

Theorem 4. Let Assumption 2 hold and let { θ t } t ≥ 1 be generated using Algorithm 1 with η = (1 -γ ) 3 / 8 , c the positive constant from Lemma 9. Then, for all t ≥ 1 ,

<!-- formula-not-decoded -->

Proof. Let us first note that for any θ and µ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to the value sub-optimality lemma of Lemma 21,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality is because of which leads to the final result,

<!-- formula-not-decoded -->

thus, finishing the proof.

## A.2. Proofs for Section 4: entropy regularized softmax policy gradient

## A.2.1. PRELIMINARIES

Lemma 10. Entropy regularized policy gradient w.r.t. θ is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ A π θ ( s, a ) is soft advantage function defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the last equation is again by Lemma 21. According to Lemma 7, V π θ ( µ ) is β -smooth with β = 8 / (1 -γ ) 3 . Denote δ t = V ∗ ( µ ) -V π θ t ( µ ) . And note η = (1 -γ ) 3 8 . We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second to last inequality is by d π θ t µ ( s ) ≥ (1 -γ ) · µ ( s ) (cf. Eq. (335)). According to Lemma 9, c = inf s ∈S ,t ≥ 1 π θ t ( a ∗ ( s ) | s ) &gt; 0 . Using similar induction arguments as in Eq. (79),

<!-- formula-not-decoded -->

Proof. According to the definition of ˜ V π θ ,

<!-- formula-not-decoded -->

Taking derivative w.r.t. θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equation is because of

<!-- formula-not-decoded -->

glyph[negationslash]

Using similar arguments as in the proof for Lemma 1, i.e., for s ′ = s , ∂π θ ( a | s ) ∂θ ( s ′ , · ) = 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For each component a , we have

<!-- formula-not-decoded -->

## A.2.2. PROOFS FOR BANDITS AND NON-UNIFORM CONTRACTION

Lemma 11 (Non-uniform contraction) . Using Update 2 with τη ≤ 1 , ∀ t ≥ 1 ,

<!-- formula-not-decoded -->

where ζ t = τθ t -r -( τθ t -r ) glyph[latticetop] 1 K · 1 .

Proof. Update 2 can be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last two equations are from H ( π θ t ) 1 = 0 as shown in Lemma 22. For all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the last term,

<!-- formula-not-decoded -->

where the last equation is again by H ( π θ t ) glyph[latticetop] 1 = H ( π θ t ) 1 = 0 . Using the update rule and combining the above,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Lemma 23, with τη ≤ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 12. Let π θ t = softmax( θ t ) . Using Update 2 with τη ≤ 1 , ∀ t ≥ 1 ,

<!-- formula-not-decoded -->

Proof. According to Lemma 11, for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the initial logit θ 1 , finishing the proof.

Lemma 13. There exists c = c ( τ, K, ‖ θ 1 ‖ ∞ ) &gt; 0 , such that for all t ≥ 1 , min a π θ t ( a ) ≥ c . Thus, ∑ t -1 s =1 min a π θ s ( a ) ≥ c · ( t -1) .

Proof. Define the constant c = c ( τ, K, ‖ θ 1 ‖ ∞ ) as

<!-- formula-not-decoded -->

First, according to Eq. (389), we have,

Next, according to Lemma 11, with τη ≤ 1 ,

<!-- formula-not-decoded -->

Therefore, for all t ≥ 1 , we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove min a π θ t ( a ) ≥ c . We have, ∀ a ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote a 1 = arg min a θ t ( a ) , and a 2 = arg max a θ t ( a ) . According to the above, we have the following results,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which can be used to lower bound the minimum probability as,

<!-- formula-not-decoded -->

which can be further lower bounded using the above results,

<!-- formula-not-decoded -->

Theorem 5. Let π θ t = softmax( θ t ) . Using Update 2 with η ≤ 1 /τ , for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ δ t := π ∗ τ glyph[latticetop] ( r -τ log π ∗ τ ) -π θ t glyph[latticetop] ( r -τ log π θ t ) and c &gt; 0 is from Lemma 13.

Proof. According to H¨ older's inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2.3. PROOFS FOR MDPS AND ENTROPY REGULARIZATION

Lemma 14 (Smoothness) . H ( ρ, π θ ) is (4 + 8 log A ) / (1 -γ ) 3 -smooth, where A = |A| is the total number of actions.

Proof. Denote H π θ ( s ) = H ( s, π θ ) . Also denote θ α = θ + αu , where α ∈ R and u ∈ R SA . According to Eq. (16),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ( α ) = ( Id -γP ( α )) -1 is defined in Eq. (203), P ( α ) is defined in Eq. (189), and h θ α ∈ R S for s ∈ S is given by

<!-- formula-not-decoded -->

According to Eq. (431), h θ α ( s ) ∈ [0 , log A ] , ∀ s . Then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies,

For any state s ∈ S ,

The glyph[lscript] 1 norm is upper bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second derivative w.r.t. α is

Therefore we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote the Hessian T ( s, θ α ) = ∂ 2 h θα ( s ) ∂θ 2 ( s, · ) . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note T ( s, θ α ) ∈ R A × A , and ∀ i, j ∈ A , the value of T ( s, θ α ) is,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any vector y ∈ R A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by H¨ older's inequality. Note that ‖ y glyph[circledot] y ‖ 1 = ‖ y ‖ 2 2 , ‖ π θ α ( ·| s ) ‖ ∞ ≤ ‖ π θ α ( ·| s ) ‖ 1 , ‖ π θ α ( ·| s ) ‖ 2 ≤ ‖ π θ α ( ·| s ) ‖ 1 = 1 , and ‖ y ‖ ∞ ≤ ‖ y ‖ 2 . The glyph[lscript] ∞ norm is upper bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to the above results,

<!-- formula-not-decoded -->

Taking derivative w.r.t. α in Eq. (430),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking second derivative w.r.t. α ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the last term,

For the second last term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the first term, according to Eqs. (192) and (208), Eq. (432),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Eqs. (478), (481), (486) and (491) with Eq. (476),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies for all y ∈ R SA and θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote θ ξ = θ + ξ ( θ ′ -θ ) , where ξ ∈ [0 , 1] . According to Taylor's theorem, ∀ s , ∀ θ, θ ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since H π θ ( s ) is (4 + 8 log A ) / (1 -γ ) 3 -smooth, ∀ s , H ( ρ, π θ ) = E s ∼ ρ [ H π θ ( s )] is also (4 + 8 log A ) / (1 -γ ) 3 -smooth.

Lemma 15 (Non-uniform Łojasiewicz) . Suppose µ ( s ) &gt; 0 for all states s ∈ S and π θ ( ·| s ) = softmax( θ ( s, · )) . Then,

<!-- formula-not-decoded -->

Proof. According to the definition of soft value functions,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, define the 'soft greedy policy' ¯ π θ ( ·| s ) = softmax( ˜ Q π θ ( s, · ) /τ ) , ∀ s , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have, ∀ s ,

Also note that,

Combining the above,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A = |A| is the total number of actions. Taking square root of soft sub-optimality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, the entropy regularized policy gradient norm is lower bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is further lower bounded as

<!-- formula-not-decoded -->

Denote ζ θ ( s ) = ˜ Q π θ ( s, · ) -τθ ( s, · ) -( ˜ Q π θ ( s, · ) -τθ ( s, · )) glyph[latticetop] 1 K · 1 . We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is by d θ ( s ) ≥ (1 -γ ) · µ ( s )

π µ (cf. Eq. (335)).

Lemma 16. Using Algorithm 1 with the entropy regularized objective, we have c := inf t ≥ 1 min s,a π θ t ( a | s ) &gt; 0 .

Proof. The augmented value function ˜ V π θ t ( ρ ) is monotonically increasing following gradient update due to smoothness, i.e., Lemmas 7 and 14. It follows then that ˜ V π θ t ( ρ ) is upper bounded. Indeed,

<!-- formula-not-decoded -->

According to the monotone convergence theorem, ˜ V π θ t ( ρ ) converges to a finite value. Suppose π θ t ( a | s ) → π θ ∞ ( a | s ) . For any state s ∈ S , define the following sets,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that A = A 0 ( s ) ∪A + ( s ) since π ∞ ( a | s ) ≥ 0 , ∀ a ∈ A . We prove that for any state s ∈ S , A 0 ( s ) = ∅ by contradiction. Suppose ∃ s ∈ S , such that A 0 ( s ) is non-empty. For any a 0 ∈ A 0 ( s ) , we have π θ t ( a 0 | s ) → π θ ∞ ( a 0 | s ) = 0 , which implies -log π θ t ( a 0 | s ) →∞ . There exists t 0 ≥ 1 , such that ∀ t ≥ t 0 ,

<!-- formula-not-decoded -->

According to Lemma 10, ∀ t ≥ t 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality is by

<!-- formula-not-decoded -->

This means that θ t ( s, a 0 ) is increasing for any t ≥ t 0 , which in turn implies that θ ∞ ( s, a 0 ) is lower bounded by constant, i.e., θ ∞ ( s, a 0 ) ≥ c for some constant c , and thus exp { θ ∞ ( a 0 | s ) } ≥ e c &gt; 0 . According to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, for any a + ∈ A + ( s ) , according to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that ∀ t , the summation of logit incremental over all actions is zero:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have, we have,

which implies,

According to Eq. (548), ∀ t ≥ t 0 ,

According to Eq. (558), ∀ t ≥ t 0 , which is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means ∑ a + ∈A + ( s ) θ t ( s, a + ) will decrease for all large enough t ≥ 1 . This contradicts with Eq. (557), i.e., ∑ a + ∈A + ( s ) θ t ( s, a + ) →∞ .

To this point, we have shown that A 0 ( s ) = ∅ for any state s ∈ S , i.e., π θ t ( ·| s ) will converge in the interior of probabilistic simplex ∆( A ) . Furthermore, at the convergent point π θ ∞ ( ·| s ) , the gradient is zero, otherwise by smoothness the objective can be further improved, which is a contradiction with convergence. According to Lemma 10, ∀ s ,

<!-- formula-not-decoded -->

We have d π θ ∞ µ ( s ) ≥ (1 -γ ) · µ ( s ) &gt; 0 for all states s (cf. Eq. (335)). Therefore we have, ∀ s ,

<!-- formula-not-decoded -->

According to Lemma 22, H ( π θ ∞ ( ·| s )) has eigenvalue 0 with multiplicity 1 , and its corresponding eigenvector is c · 1 for some constant c ∈ R . Therefore, the gradient is zero implies that for all states s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, according to Nachum et al. (2017, Theorem 3), is the softmax optimal policy π ∗ τ . Since τ ∈ Ω(1) &gt; 0 and,

<!-- formula-not-decoded -->

we have π θ ∞ ( a | s ) ∈ Ω(1) , ∀ ( s, a ) . Since π θ t ( a | s ) → π θ ∞ ( a | s ) , there exists t 0 ≥ 1 , such that ∀ t ≥ t 0 ,

<!-- formula-not-decoded -->

which means inf t ≥ t 0 min s,a π θ t ( a | s ) ∈ Ω(1) , and thus

<!-- formula-not-decoded -->

Theorem 6. Suppose µ ( s ) &gt; 0 for all state s . Using Algorithm 1 with the entropy regularized objective and softmax parametrization and η = (1 -γ ) 3 / (8 + τ (4 + 8 log A )) , there exists a constant C &gt; 0 such that for all t ≥ 1 ,

<!-- formula-not-decoded -->

Proof. According to the soft sub-optimality lemma of Lemma 26,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equation is again by Lemma 26, and the first inequality is according to d π θ t µ ( s ) ≥ (1 -γ ) · µ ( s ) (cf. Eq. (335)). According to Lemmas 7 and 14, V π θ ( µ ) is 8 / (1 -γ ) 3 -smooth, and H ( µ, π θ ) is (4 + 8 log A ) / (1 -γ ) 3 -smooth. Therefore, ˜ V π θ ( µ ) = V π θ ( µ ) + τ · H ( µ, π θ ) is β -smooth with β = (8 + τ (4 + 8 log A )) / (1 -γ ) 3 . Denote ˜ δ t = ˜ V π ∗ τ ( µ ) -˜ V π θ t ( µ ) . And note η = (1 -γ ) 3 8+ τ (4+8log A ) . We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Lemma 16, c = inf t ≥ 1 min s,a π θ t ( a | s ) &gt; 0 is independent with t . We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is according to Eq. (541). Therefore we have the final result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where is independent with t .

## A.2.4. PROOFS FOR TWO-STAGE AND DECAYING ENTROPY REGULARIZATION

glyph[negationslash]

Theorem 7 (Two-stage) . Denote ∆ = r ( a ∗ ) -max a = a ∗ r ( a ) &gt; 0 . Using Update 2 for t 1 ∈ O ( e 1 /τ · log ( τ +1 ∆ )) iterations and then Update 1 for t 2 ≥ 1 iterations, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where t = t 1 + t 2 , and C ∈ [1 /K, 1) .

Proof. In particular, using Update 2 with η ≤ 1 /τ for the following number of iterations,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have,

<!-- formula-not-decoded -->

Therefore we have, which is equivalent to,

Then we have, for all a ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies, glyph[negationslash]

Then we have, for all a = a ∗ ,

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means π θ t 1 ( a ∗ ) ≥ π θ t 1 ( a ) . Now we turn off the regularization and use Update 1 for t 2 ≥ 1 iterations. According to similar arguments as in Theorem 3, we have,

<!-- formula-not-decoded -->

where t = t 1 + t 2 , and C ∈ [1 /K, 1) .

Theorem 8 (Decaying entropy regularization) . Using Update 3 with τ t = α · ∆ log t for t ≥ 2 , where α &gt; 0 , and η t = 1 /τ t , we have, for all t ≥ 1 ,

<!-- formula-not-decoded -->

Proof. Denote π ∗ τ t = softmax( r/τ t ) as the softmax optimal policy at time t . We have,

<!-- formula-not-decoded -->

glyph[negationslash]

'decaying' part. Note a ∗ is the optimal action. Denote ∆( a ) = r ( a ∗ ) -r ( a ) , and ∆ = min a = a ∗ ∆( a ) . We have, glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

Using the decaying temperature τ t = α · ∆ log t , for t ≥ 2 , where α &gt; 0 , we have,

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

'tracking' part. Using Update 3, we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.3. Proofs for Section 5 (Does Entropy Regularization Really Help?)

## A.3.1. PROOFS FOR THE BANDIT CASE

Lemma 17 (Reversed Łojasiewicz) . Take any r ∈ [0 , 1] K . Denote ∆ = r ( a ∗ ) -max a = a ∗ r ( a ) &gt; 0 . Then, glyph[negationslash]

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

Proof. Note a ∗ is the optimal action. Denote ∆( a ) = r ( a ∗ ) -r ( a ) , and ∆ = min a = a ∗ ∆( a ) .

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Therefore the glyph[lscript] 2 norm of gradient can be upper bounded as

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Combining the results, we have

<!-- formula-not-decoded -->

glyph[negationslash]

Theorem 9 (Lower bound) . Take any r ∈ [0 , 1] K . For large enough t ≥ 1 , using Update 1 with learning rate η t ∈ (0 , 1] ,

<!-- formula-not-decoded -->

Proof. Denote δ t = ( π ∗ -π θ t ) glyph[latticetop] r &gt; 0 . Let θ t +1 = θ t + η t · dπ glyph[latticetop] θ t r dθ t , and π θ t +1 = softmax( θ t +1 ) be the next policy after one step gradient update. We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

According to convergence result Theorem 2 we have δ t &gt; 0 , δ t → 0 as t →∞ . We prove that for all large enough t ≥ 1 , δ t ≤ 10 9 · δ t +1 by contradiction. Suppose δ t &gt; 10 9 · δ t +1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies δ t +1 &gt; ∆ 2 50 for large enough t ≥ 1 . This is a contradiction with δ t → 0 as t → ∞ . Now we have δ t ≤ 10 9 · δ t +1 . Divide both sides of δ t -δ t +1 ≤ 9 2 · 1 ∆ 2 · δ 2 t by δ t · δ t +1 ,

<!-- formula-not-decoded -->

Summing up from T 1 (some large enough time) to T 1 + t , we have

<!-- formula-not-decoded -->

Since T 1 is a finite time, δ T 1 ≥ 1 /C for some constant C &gt; 0 . Rearranging, we have

<!-- formula-not-decoded -->

By abusing notation t := T 1 + t and C ≤ t ∆ 2 , we have

<!-- formula-not-decoded -->

for all large enough t ≥ 1 .

## A.3.2. PROOFS FOR GENERAL MDPS

Theorem 10 (Lower bound) . Take any MDP. For large enough t ≥ 1 , using Algorithm 1 with η t ∈ (0 , 1] ,

<!-- formula-not-decoded -->

where ∆ ∗ = min s ∈S ,a = a ∗ ( s ) { Q ∗ ( s, a ∗ ( s )) -Q ∗ ( s, a ) } &gt; 0 is the optimal value gap of the MDP, and a ∗ ( s ) = arg max a π ∗ ( a | s ) is the action that the optimal policy selects under state s .

glyph[negationslash]

Proof. Suppose Algorithm 1 can converge faster than O (1 /t ) for general MDPs, then it can converge faster than O (1 /t ) for any one-state MDPs, which are special cases of general MDPs. This is a contradiction with Theorem 9.

The above one-sentence argument implies a Ω(1 /t ) rate lower bound. To calculate the constant in the lower bound, we need results similar to Lemma 17. According to the reversed Łojasiewicz inequality of Lemma 28,

<!-- formula-not-decoded -->

where δ t = V ∗ ( µ ) -V π θ t ( µ ) &gt; 0 . Let θ t +1 = θ t + η t · ∂V π θ t ( µ ) ∂θ t , and π θ t +1 ( ·| s ) = softmax( θ t +1 ( s, · )) , ∀ s ∈ S be the next policy after one step gradient update. Using similar calculations as in Eq. (637),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Theorem 4, we have δ t &gt; 0 , δ t → 0 as t →∞ . Using similar arguments as in Eq. (641), we can show that for all large enough t ≥ 1 , δ t ≤ 11 10 · δ t +1 . Divide both sides of δ t -δ t +1 ≤ 10 (1 -γ ) 5 · 1 (∆ ∗ ) 2 · δ 2 t by δ t · δ t +1 ,

<!-- formula-not-decoded -->

Using similar calculations as in the proof of Theorem 9, we have,

<!-- formula-not-decoded -->

for all large enough t ≥ 1 .

## A.3.3. PROOFS FOR THE NON-UNIFORM ŁOJASIEWICZ DEGREE

Proposition 4. Let r ∈ [0 , 1] K be arbitrary and consider θ ↦→ E a ∼ π θ [ r ( a )] . The non-uniform Łojasiewicz degree of this map with constant C ( θ ) = π θ ( a ∗ ) is zero.

Proof. We prove by contradiction. Suppose the Łojasiewicz degree of E a ∼ π θ [ r ( a )] can be larger than 0 . Then there exists ξ &gt; 0 , such that,

<!-- formula-not-decoded -->

Consider the following example, r = (0 . 6 , 0 . 4 , 0 . 2) glyph[latticetop] , π θ = (1 -3 glyph[epsilon1], 2 glyph[epsilon1], glyph[epsilon1] ) glyph[latticetop] with small number glyph[epsilon1] &gt; 0 .

<!-- formula-not-decoded -->

According to the reversed Łojasiewicz inequality of Lemma 17,

<!-- formula-not-decoded -->

Also note that π θ ( a ∗ ) = 1 -3 glyph[epsilon1] &gt; 1 / 4 . Then for ξ ∈ (0 , 1] , we have

<!-- formula-not-decoded -->

Next, since glyph[epsilon1] &gt; 0 can be very small,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality is by (0 . 8 · glyph[epsilon1] ) ξ &lt; 1 / 3 for small glyph[epsilon1] &gt; 0 since ξ &gt; 0 . This is a contradiction with the assumption. Therefore the Łojasiewicz degree ξ cannot be larger than 0 .

Proposition 5. Fix τ &gt; 0 . With C ( θ ) = √ 2 τ · min a π θ ( a ) , the Łojasiewicz degree of θ ↦→ E a ∼ π θ [ r ( a ) -τ log π θ ( a )] is at least 1 / 2 .

Proof. Denote δ θ = E a ∼ π ∗ τ [ r ( a ) -τ log π ∗ τ ( a )] -E a ∼ π θ [ r ( a ) -τ log π θ ( a )] as the soft sub-optimality. We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, the entropy regularized policy gradient w.r.t. θ is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last two equations are by H ( π θ ) 1 = 0 as shown in Lemma 22. Then we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means the Łojasiewicz degree of E a ∼ π θ [ r ( a ) -τ log π θ ( a )] is 1 / 2 and C ( θ ) = √ 2 τ · min a π θ ( a ) .

## B. Miscellaneous Extra Supporting Results

Lemma 18 (Ascent lemma for smooth function) . Let f : R d → R be a β -smooth function, θ ∈ R d and θ ′ = θ + 1 β · ∂f ( θ ) ∂θ . We have,

<!-- formula-not-decoded -->

Proof. According to the definition of smoothness, we have,

<!-- formula-not-decoded -->

which implies,

<!-- formula-not-decoded -->

Lemma 19 (First performance difference lemma (Kakade &amp; Langford, 2002)) . For any policies π and π ′ ,

<!-- formula-not-decoded -->

Proof. According to the definition of value function,

<!-- formula-not-decoded -->

Lemma 20 (Second performance difference lemma) . For any policies π and π ′ ,

<!-- formula-not-decoded -->

Proof. According to the definition of value function,

<!-- formula-not-decoded -->

Lemma 21 (Value sub-optimality lemma) . For any policy π ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. According to the second performance difference lemma of Lemma 20, the result immediately holds.

Lemma 22 (Spectrum of H matrix) . Let π ∈ ∆( A ) . Denote H ( π ) = diag ( π ) -ππ glyph[latticetop] . Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. According to Golub (1973, Section 5),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus 1 is an eigenvector of H ( π ) which corresponds to eigenvalue 0 . Furthermore, for any vector x ∈ R K ,

<!-- formula-not-decoded -->

which means all the eigenvalues of H ( π )

are non-negative.

Lemma 23. Let π ∈ ∆( A ) . Denote H ( π ) = diag ( π ) -ππ glyph[latticetop] . For any vector x ∈ R K ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. x can be written as linear combination of eigenvectors of H ( π ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since H ( π ) is symmetric, { 1 √ K , v 2 , . . . , v K } are orthonormal. The last equation is because the representation is unique, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote the eigenvalues of H ( π ) as

Then we have,

We show λ 1 = 0 . Note

Denote

We have

On the other hand,

Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality is by 0 ≤ π (1) ≤ λ 2 ≤ ·· · ≤ λ K ≤ π ( K ) ≤ 1 , and the last inequality is according to λ 2 ≥ π (1) = min a π ( a ) , and both are shown in Lemma 22. Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 24. Let π θ = softmax( θ ) and π θ ′ = softmax( θ ′ ) . Then for any constant c ∈ R ,

<!-- formula-not-decoded -->

Proof. This results improves the results of ‖ π θ -π θ ′ ‖ ∞ ≤ 2 · ‖ θ -θ ′ ‖ ∞ in Xiao et al. (2019, Lemma 5). According to the glyph[lscript] 1 norm strong convexity of negative entropy over probabilistic simplex, i.e., for any policies π , π ′ ,

<!-- formula-not-decoded -->

we have (letting π = π θ , and π ′ = π θ ′ ),

<!-- formula-not-decoded -->

which is the Pinsker's inequality. Then we have,

<!-- formula-not-decoded -->

Lemma 25 (Soft performance difference lemma) . For any policies π and π ′ ,

<!-- formula-not-decoded -->

Proof. According to the definition of soft value function,

<!-- formula-not-decoded -->

Lemma 26 (Soft sub-optimality lemma) . For any policy π ,

<!-- formula-not-decoded -->

Proof. According to Nachum et al. (2017, Theorem 1), ∀ ( s, a ) ,

<!-- formula-not-decoded -->

According to the soft performance difference lemma of Lemma 25,

<!-- formula-not-decoded -->

Lemma 27 (KL-Logit inequality) . Let π θ = softmax( θ ) and π θ ′ = softmax( θ ′ ) . Then for any constant c ∈ R ,

<!-- formula-not-decoded -->

In particular, let c = ( θ ′ -θ ) glyph[latticetop] 1 K , we have

<!-- formula-not-decoded -->

Proof. According to the glyph[lscript] 1 norm strong convexity of negative entropy over probabilistic simplex, i.e., for any policies π , π ′ ,

<!-- formula-not-decoded -->

we have (letting π = π θ , and π ′ = π θ ′ ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is according to ax -bx 2 ≤ a 2 4 b , ∀ a, b &gt; 0 .

Lemma 28 (Reversed Łojasiewicz) . Denote ∆ ∗ ( s ) = Q ∗ ( s, a ∗ ( s )) -max a = a ∗ ( s ) Q ∗ ( s, a ) &gt; 0 as the optimal value gap of state s , where a ∗ ( s ) is the action that the optimal policy selects under state s , and ∆ ∗ = min s ∈S ∆ ∗ ( s ) &gt; 0 as the optimal value gap of the MDP. Then we have, glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Proof. Denote ∆ ∗ ( s, a ) = Q ∗ ( s, a ∗ ( s )) -Q ∗ ( s, a ) , and ∆ ∗ ( s ) = min a = a ∗ ( s ) ∆ ∗ ( s, a ) . We have,

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

π π π

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

Therefore the glyph[lscript] 2 norm of gradient can be upper bounded as glyph[negationslash]

<!-- formula-not-decoded -->

Combining the results, we have

<!-- formula-not-decoded -->

glyph[negationslash]

## C. Sub-optimality Guarantees for Other Entropy-Based RL Methods

Some interesting insight worth mentioning in the proof of Lemma 15 is that the intermediate results provide sub-optimality guarantees for existing entropy regularized RL methods. In particular, Eqs. (513) and (523) provides policy improvement guarantee for Soft Actor-Critic (Haarnoja et al., 2018, SAC), and Eqs. (524) and (529) provide sub-optimality guarantees for Patch Consistency Learning (Nachum et al., 2017, PCL).

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

Remark 6 (Soft policy improvement inequality) . In Haarnoja et al. (2018, Eq. (4) and Lemma 2), the policy is updated by

<!-- formula-not-decoded -->

which is exactly the KL divergence in Eq. (523) , with ¯ π θ ( ·| s ) defined in Eq. (513) . The soft policy improvement inequality of Eq. (523) guarantees that if the soft policy improvement is small, then the sub-optimality is small.

Remark 7 (Path inconsistency inequality) . In Nachum et al. (2017, Theorems 1 and 3), it is shown that

- (i) soft optimal policy π ∗ τ satisfies the consistency conditions Eqs. (25) and (26) ;
- (ii) for any policy π that satisfies the consistency conditions, i.e., if ∀ s, a ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

However, Nachum et al. (2017) does not show if the consistency is violated during learning, how the violation is related to the sub-optimality. To see why Lemma 15 provides insight, define the following 'path inconsistency',

<!-- formula-not-decoded -->

which captures the violation of consistency conditions during learning. Note that for softmax policy π θ ( ·| s ) = softmax( θ ( s, · )) , the r.h.s. of Eq. (760) can be written in vector form as

<!-- formula-not-decoded -->

Denote c θ ( s ) = ˜ V π θ ( s ) τ -log ∑ a exp { θ ( s, a ) } , and using Lemma 27 in the proof of Lemma 15, in particular, Eq. (524) ,

<!-- formula-not-decoded -->

Using the above results in Eq. (529) ,

<!-- formula-not-decoded -->

where (square of) ∣ ∣ ∣ r ( s, a ) + γ ∑ s ′ P ( s ′ | s, a ) ˜ V π θ ( s ′ ) -τ log π θ ( a | s ) -˜ V π θ ( s ) ∣ ∣ ∣ is exactly the (one-step) path inconsistency objective used in PCL (Nachum et al., 2017, Eq. (14)). Therefore, minimizing path inconsistency guarantees small sub-optimality. The path inconsistency inequality of Eq. (762) implies path consistency of Nachum et al. (2017).

## D. Simulation Results

To verify the convergence rates in the main paper, we conducted experiments on one-state MDPs, which have K actions, with randomly generated reward r ∈ [0 , 1] K , and randomly initialized policy π θ 1 .

Figure 2. Softmax policy gradient, Update 1.

<!-- image -->

## D.1. Softmax Policy Gradient

K = 20 , r ∈ [0 , 1] K is randomly generated, and π θ 1 is randomly initialized. Softmax policy gradient, i.e., Update 1 is used with learning rate η = 2 / 5 and T = 3 × 10 5 . As shown in Fig. 2(a), the sub-optimality δ t = ( π ∗ -π θ t ) glyph[latticetop] r approaches 0 . Subfigures (b) and (c) show log δ t as a function of log t . As log t increases, the slope is approaching -1 , indicating that log δ t = -log t + C , which is equivalent to δ t = C ′ /t . Subfigure (d) shows π θ t ( a ∗ ) as a function of t .

## D.2. Entropy Regularized Softmax Policy Gradient

K = 20 , r ∈ [0 , 1] K and π θ 1 are the same as above. Entropy regularized softmax policy gradient, i.e., Update 2 is used with temperature τ = 0 . 2 , learning rate η = 2 / 5 and T = 5 × 10 4 . As shown in Fig. 3(a), the soft sub-optimality ˜ δ t = π ∗ τ glyph[latticetop] ( r -τ log π ∗ τ ) -π θ t glyph[latticetop] ( r -τ log π θ t ) approaches 0 . Subfigure (b) shows log ˜ δ t as a function of t . As t increases, the curve approaches a straight line, indicating that log ˜ δ t = -C 1 · t + C 2 , which is equivalent to ˜ δ t = C ′ 2 / exp { C ′ 1 · t } . Subfigure (c) shows ζ t as defined in Lemma 11 as a function of t , which verifies Lemma 12. Subfigure (d) shows min a π θ t ( a ) as a function of t . As t increases, min a π θ t ( a ) approaches constant values, which verifies Lemma 13.

Figure 3. Entropy regularized softmax policy gradient, Update 2.

<!-- image -->

## D.3. 'Bad' Initializations for Softmax Policy Gradient (PG)

As illustrated in Fig. 1, 'bad' initializations lead to attraction toward sub-optimal corners and slowly escaping for softmax policy gradient. Fig. 4 shows one example with K = 5 . Softmax policy gradient takes about 8 × 10 6 iterations around a sub-optimal corner. While with entropy regularization ( τ = 0 . 2 ), the convergence is significantly faster.

Figure 4. Bad initialization for softmax policy gradient.

<!-- image -->

## D.4. Decaying Entropy Regularization

We run entropy regularized policy gradient with decaying temperature τ t = α · ∆ log t for t ≥ 2 , i.e., Update 3. Fig. 5 shows one example with K = 10 and different α values. The actual rate is O ( 1 t -slope ) , and the partial rate in Theorem 8 is O ( 1 t 1 /α ) .

Figure 5. Decaying entropy regularization, Update 3.

<!-- image -->