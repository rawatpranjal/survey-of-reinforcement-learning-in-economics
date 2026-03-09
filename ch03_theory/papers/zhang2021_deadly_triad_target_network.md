## Breaking the Deadly Triad with a Target Network

Shangtong Zhang 1 Hengshuai Yao 2 3 Shimon Whiteson 1

## Abstract

The deadly triad refers to the instability of a reinforcement learning algorithm when it employs off-policy learning, function approximation, and bootstrapping simultaneously. In this paper, we investigate the target network as a tool for breaking the deadly triad, providing theoretical support for the conventional wisdom that a target network stabilizes training. We first propose and analyze a novel target network update rule which augments the commonly used Polyak-averaging style update with two projections. We then apply the target network and ridge regularization in several divergent algorithms and show their convergence to regularized TD fixed points. Those algorithms are off-policy with linear function approximation and bootstrapping, spanning both policy evaluation and control, as well as both discounted and average-reward settings. In particular, we provide the first convergent linear Q -learning algorithms under nonrestrictive and changing behavior policies without bi-level optimization.

## 1. Introduction

The deadly triad (see, e.g., Chapter 11.3 of Sutton &amp; Barto (2018)) refers to the instability of a value-based reinforcement learning (RL, Sutton &amp; Barto (2018)) algorithm when it employs off-policy learning, function approximation, and bootstrapping simultaneously. Different from on-policy methods, where the policy of interest is executed for data collection, off-policy methods execute a different policy for data collection, which is usually safer (Dulac-Arnold et al., 2019) and more data efficient (Lin, 1992; Sutton et al., 2011). Function approximation methods use parameterized functions, instead of a look-up table, to represent quantities of interest, which usually cope better with large-scale problems (Mnih et al., 2015; Silver et al., 2016). Bootstrap-

1 University of Oxford 2 Huawei Technologies 3 University of Alberta. Correspondence to: Shangtong Zhang &lt; shangtong.zhang@cs.ox.ac.uk &gt; .

Proceedings of the 38 th International Conference on Machine Learning , PMLR 139, 2021. Copyright 2021 by the author(s).

ping methods construct update targets for an estimate by using the estimate itself recursively, which usually has lower variance than Monte Carlo methods (Sutton, 1988). However, when an algorithm employs all those three preferred ingredients (off-policy learning, function approximation, and bootstrapping) simultaneously, there is usually no guarantee that the resulting algorithm is well behaved and the value estimates can easily diverge (see, e.g., Baird (1995); Tsitsiklis &amp; Van Roy (1997); Zhang et al. (2021)), yielding the notorious deadly triad.

An example of the deadly triad is Q -learning (Watkins &amp; Dayan, 1992) with linear function approximation, whose divergence is well documented in Baird (1995). However, DeepQ -Networks (DQN, Mnih et al. (2015)), a combination of Q -learning and deep neural network function approximation, has enjoyed great empirical success. One major improvement of DQN over linear Q -learning is the use of a target network, a copy of the neural network function approximator (the main network) that is periodically synchronized with the main network. Importantly, the bootstrapping target in DQN is computed via the target network instead of the main network. As the target network changes slowly, it provides a stable bootstrapping target which in turn stabilizes the training of DQN. Instead of the periodical synchronization, Lillicrap et al. (2015) propose a Polyak-averaging style target network update, which has also enjoyed great empirical success (Fujimoto et al., 2018; Haarnoja et al., 2018).

Inspired by the empirical success of the target network in RL with deep networks, in this paper, we theoretically investigate the target network as a tool for breaking the deadly triad. We consider a two-timescale framework, where the main network is updated faster than the target network. By using a target network to construct the bootstrapping target, the main network update becomes least squares regression. After adding ridge regularization (Tikhonov et al., 2013) to this least squares problem, we show convergence for both the target and main networks.

Our main contributions are twofold. First, we propose a novel target network update rule augmenting the Polyakaveraging style update with two projections. The balls for the projections are usually large so most times they are just identity mapping. However, those two projections offer sig-

nificant theoretical advantages making it possible to analyze where the target network converges to (Section 3). Second, we apply the target network in various existing divergent algorithms and show their convergence to regularized TD (Sutton, 1988) fixed points. Those algorithms are off-policy algorithms with linear function approximation and bootstrapping, spanning both policy evaluation and control, as well as both discounted and average-reward settings. In particular, we provide the first convergent linear Q -learning algorithms under nonrestrictive and changing behavior policies without bi-level optimization, for both discounted and average-reward settings.

## 2. Background

Let M be a real positive definite matrix and x be a vector, we use ∥ x ∥ M . = √ x ⊤ Mx to denote the norm induced by M and ∥·∥ M to denote the corresponding induced matrix norm. When M is the identity matrix I , we ignore the subscript I for simplicity. We use vectors and functions interchangeably when it does not cause confusion, e.g., given f : X → R , we also use f to denote the corresponding vector in R |X| . All vectors are column vectors. We use 1 to denote an all one vector, whose dimension can be deduced from the context. 0 is similarly defined.

We consider an infinite horizon Markov Decision Process (MDP, see, e.g., Puterman (2014)) consisting of a finite state space S , a finite action space A , a transition kernel p : S × S × A → [0 , 1] , and a reward function r : S × A → R . At time step t , an agent at a state S t executes an action A t ∼ π ( ·| S t ) , where π : A × S → [0 , 1] is the policy followed by the agent. The agent then receives a reward R t +1 . = r ( S t , A t ) and proceeds to a new state S t +1 ∼ p ( ·| S t , A t ) .

In the discounted setting, we consider a discount factor γ ∈ [0 , 1) and define the return at time step t as G t . = ∑ ∞ i =1 γ i -1 R t + i , which allows us to define the actionvalue function q π ( s, a ) . = E π,p [ G t | S t = s, A t = a ] . The action-value function q π is the unique fixed point of the Bellman operator T π , i.e., q π = T π q π . = r + γP π q π , where P π ∈ R |S||A|×|S||A| is the transition matrix, i.e., P π (( s, a ) , ( s ′ , a ′ )) . = ∑ a p ( s ′ | s, a ) π ( a ′ | s ′ ) .

In the average-reward setting, we assume:

Assumption 2.1. The chain induced by π is ergodic.

This allows us to define the reward rate ¯ r π . = lim T →∞ 1 T ∑ T t =1 E [ R t | p, π ] . The differential actionvalue function ¯ q π ( s, a ) is defined as

<!-- formula-not-decoded -->

The differential Bellman equation is

<!-- formula-not-decoded -->

where ¯ q ∈ R |S||A| and ¯ r ∈ R are free variables. It is well known that all solutions to (1) form a set { (¯ q, ¯ r ) | ¯ r = ¯ r π , ¯ q = q π + c 1 , c ∈ R } (Puterman, 2014).

The policy evaluation problem refers to estimating q π or (¯ q π , ¯ r π ) . The control problem refers to finding a policy π maximizing q π ( s, a ) for each ( s, a ) or maximizing ¯ r π . With linear function approximation, we approximate q π ( s, a ) or ¯ q π ( s, a ) with x ( s, a ) ⊤ w , where x : S × A → R K is a feature mapping and w ∈ R K is the learnable parameter. We use X ∈ R |S||A|× K to denote the feature matrix, each row of which is x ( s, a ) ⊤ , and assume:

Assumption 2.2. X has linearly independent columns.

In the average-reward setting, we use an additional parameter ¯ r ∈ R to approximate ¯ r π . In the off-policy learning setting, the data for policy evaluation or control is collected by executing a policy µ (behavior policy) in the MDP, which is different from π (target policy). In the rest of the paper, we consider the off-policy linear function approximation setting thus always assume A t ∼ µ ( ·| S t ) . We use as shorthand x t . = x ( S t , A t ) , ¯ x t . = ∑ a π ( a | S t ) x ( S t , a ) .

Policy Evaluation . In the discounted setting, similar to Temporal Difference Learning (TD, Sutton (1988)), one can use Off-Policy Expected SARSA to estimate q π , which updates w as

<!-- formula-not-decoded -->

where { α t } are learning rates. In the average-reward setting, (1) implies that ¯ r π = d ⊤ ( r + P π ¯ q π -¯ q π ) holds for any probability distribution d . In particular, it holds for d = d µ . Consequently, to estimate ¯ q π and ¯ r π , Wan et al. (2020); Zhang et al. (2021) update w and ¯ r as

<!-- formula-not-decoded -->

Unfortunately, both (2) and (3) can possibly diverge (see, e.g., Tsitsiklis &amp; Van Roy (1997); Zhang et al. (2021)), which exemplifies the deadly triad in discounted and average-reward settings respectively.

Control . In the discounted setting, Q -learning with linear function approximation yields

<!-- formula-not-decoded -->

In the average-reward setting, Differential Q -learning (Wan et al., 2020) with linear function approximation yields

<!-- formula-not-decoded -->

Unfortunately, both (4) and (5) can possibly diverge as well (see, e.g., Baird (1995); Zhang et al. (2021)), exemplifying the deadly triad again.

Motivated by the empirical success of the target network in deep RL, one can apply the target network in the linear function approximation setting. For example, using a target network in (4) yields

<!-- formula-not-decoded -->

where θ denotes the target network, { β t } are learning rates, and we consider the Polyak-averaging style target network update. The convergence of (6) and (7), however, remains unknown. Besides target networks, regularization has also been widely used in deep RL, e.g., Mnih et al. (2015) consider a Huber loss instead of a mean-squared loss; Lillicrap et al. (2015) consider ℓ 2 weight decay in updating Q -values.

## 3. Analysis of the Target Network

In Sections 4 &amp; 5, we consider the merits of using a target network in several linear RL algorithms (e.g., (2) (3) (4) (5)). To this end, in this section, we start by proposing and analyzing a novel target network update rule:

<!-- formula-not-decoded -->

In (8), w denotes the main network and θ denotes the target network. Γ B 1 : R K → R K is a projection to the ball B 1 . = { x ∈ R K | ∥ x ∥ ≤ R B 1 } , i.e.,

<!-- formula-not-decoded -->

where I is the indicator function. Γ B 2 is a projection onto the ball B 2 with a radius R B 2 . We make the following assumption about the learning rates:

Assumption 3.1. { β t } is a deterministic positive nonincreasing sequence satisfying ∑ t β t = ∞ , ∑ t β 2 t &lt; ∞ .

While (8) specifies only how θ is updated, we assume w is updated such that w can track θ in the sense that

Assumption 3.2. There exists w ∗ : R K → R K such that lim t →∞ ∥ w t -w ∗ ( θ t ) ∥ = 0 almost surely.

After making some additional assumptions on w ∗ , we arrive at our general convergent results.

Assumption 3.3. sup θ ∥ w ∗ ( θ ) ∥ &lt; R B 2 &lt; R B 1 &lt; ∞ .

Theorem 1. (Convergence of Target Network) Under Assumptions 3.1-3.4, the iterate { θ t } generated by (8) satisfies

Assumption 3.4. w ∗ is a contraction mapping w.r.t. ∥·∥ .

<!-- formula-not-decoded -->

where θ ∗ is the unique fixed point of w ∗ ( · ) .

Assumptions 3.2 - 3.4 are assumed only for now. Once the concrete update rules for w are specified in the algorithms in Sections 4 &amp; 5, we will prove that those assumptions indeed hold. Assumption 3.2 is expected to hold because we will later require that the target network to be updated much slower than the main network. Consequently, the update of the main network will become a standard least-square regression, whose solution w ∗ usually exists. Assumption 3.4 is expected to hold becuase we will later apply ridge regularization to the least-square regression. Consequently, its solution w ∗ will not change too fast w.r.t. the change of the regression target.

The target network update (8) is the same as that in (7) except for the two projections, where the first projection Γ B 1 is standard in optimization literature. The second projection Γ B 2 , however, appears novel and plays a crucial role in our analysis. First , if we have only Γ B 1 , the iterate { θ t } would converge to the invariant set of the ODE

<!-- formula-not-decoded -->

where ζ ( t ) is a reflection term that moves θ ( t ) back to B 1 when θ ( t ) becomes too large (see, e.g., Section 5 of Kushner &amp; Yin (2003)). Due to this reflection term, it is possible that θ ( t ) visits the boundary of B 1 infinitely often. It thus becomes unclear what the invariant set of (9) is even if w ∗ is contractive. By introducing the second projection Γ B 2 and ensuring R B 1 &gt; R B 2 , we are able to remove the reflection term and show that the iterate { θ t } tracks the ODE

<!-- formula-not-decoded -->

whose invariant set is a singleton { θ ∗ } when Assumption 3.4 holds. See the proof of Theorem 1 in Section A.1 based on the ODE approach (Kushner &amp; Yin, 2003; Borkar, 2009) for more details. Second , to ensure the main network tracks the target network in the sense of Assumption 3.2 in our applications in Sections 4 &amp; 5, it is crucial that the target network changes sufficiently slowly in the following sense:

<!-- formula-not-decoded -->

Lemma 1 would not be feasible without the second projection Γ B 2 and we defer its proof to Section A.2

In Sections 4 &amp; 5, we provide several applications of Theorem 1 in both discounted and average-reward settings, for both policy evaluation and control. We consider a twotimescale framework, where the target network is updated more slowly than the main network. Let { α t } be the learning rates for updating the main network w ; we assume

Assumption 3.5. { α t } is a deterministic positive nonincreasing sequence satisfying ∑ t α t = ∞ , ∑ t α 2 t &lt; ∞ . Further, for some d &gt; 0 , ∑ t ( β t /α t ) d &lt; ∞ .

## 4. Application to Off-Policy Policy Evaluation

In this paper, we consider estimating the action-value q π instead of the state-value v π for unifying notations of policy evaluation and control. The algorithms for estimating v π are straightforward up to change of notations and introduction of importance sampling ratios.

Discounted Setting . Using a target network for bootstrapping in (2) yields

<!-- formula-not-decoded -->

As θ t is quasi-static for w t (Lemma 1 and Assumption 3.5), this update becomes least squares regression. Motivated by the success of ridge regularization in least squares and the widespread use of weight decay in deep RL, which is essentially ridge regularization, we add ridge regularization to this least squares, yielding Q -evaluation with a Target Network (Algorithm 1).

## Algorithm 1 Q -evaluation with a Target Network

<!-- formula-not-decoded -->

Let A = X ⊤ D µ ( I -γP π ) X,b = X ⊤ D µ r , where D µ is a diagonal matrix whose diagonal entry is d µ , the stationary state-action distribution of the chain induced by µ . Let Π D µ . = X ( X ⊤ D µ X ) -1 X ⊤ D µ be the projection to the column space of X . We have

Assumption 4.1. The chain in S × A induced by µ is ergodic.

Theorem 2. Under Assumptions 2.2, 3.1, 3.5, &amp; 4.1, for any ξ ∈ (0 , 1) , let C 0 . = 2(1 -ξ ) √ η γ ∥ P π ∥ Dµ , C 1 . = ∥ r ∥ 2 ξ √ η +1 , then for all ∥ X ∥ &lt; C 0 , C 1 &lt; R B 1 , R B 1 -ξ &lt; R B 2 &lt; R B 1 the iterate { w t } generated by Algorithm 1 satisfies

<!-- formula-not-decoded -->

where w ∗ η is the unique solution of ( A + ηI ) w -b = 0 , and

<!-- formula-not-decoded -->

where σ max ( · ) , σ min ( · ) denotes the largest and minimum singular values.

We defer the proof to Section A.3. Theorem 2 requires that the balls for projection are sufficiently large, which is completely feasible in practice. Theorem 2 also requires that the feature norm ∥ X ∥ is not too large. Similar assumptions on feature norms also appear in Zou et al. (2019); Du et al. (2019); Chen et al. (2019b); Carvalho et al. (2020); Wang &amp; Zou (2020); Wu et al. (2020) and can be easily achieved by scaling.

The solutions to Aw -b = 0 , if they exist, are TD fixed points for off-policy policy evaluation in the discounted setting (Sutton et al., 2009b;a). Theorem 2 shows that Algorithm 1 finds a regularized TD fixed point w ∗ η , which is also the solution of Least-Squares TD methods (LSTD, Boyan (1999); Yu (2010)). LSTD maintains estimates for A and b (referred to as ˆ A and ˆ b ) in an online fashion, which requires O ( K 2 ) computational and memory complexity per step. As ˆ A is not guaranteed to be invertible, LSTD usually uses ( ˆ A + ηI ) -1 ˆ b as the solution and η plays a key role in its performance (see, e.g, Chapter 9.8 of Sutton &amp; Barto (2018)). By contrast, Algorithm 1 finds the LSTD solution (i.e., w ∗ η ) with only O ( K ) computational and memory complexity per step. Moreover, Theorem 2 provides a performance bound for w ∗ η . Let w ∗ 0 . = A -1 b ; Kolter (2011) shows with a counterexample that the approximation error of TD fixed points (i.e., ∥ Xw ∗ 0 -q π ∥ ) can be arbitrarily large if µ is far from π , as long as there is representation error (i.e., ∥ ∥ Π D µ q π -q π ∥ ∥ &gt; 0 ) (see Section 6 for details). By contrast, Theorem 2 guarantees that ∥ ∥ Xw ∗ η -q π ∥ ∥ is bounded from above, which is one possible advantage of regularized TD fixed points.

## Algorithm 2 Diff. Q -evaluation with a Target Network

<!-- formula-not-decoded -->

Average-reward Setting . In the average-reward setting, we need to learn both ¯ r and w . Hence, we consider target networks θ r and θ w for ¯ r and w respectively. Plugging θ r and θ w into (3) for bootstrapping yields Differential Q -evaluation with a Target Network (Algorithm 2), where { B i } are now balls in R K +1 . In Algorithm 2, we impose ridge regularization only on w as ¯ r is a scalar and thus does

not have any representation capacity limit.

Theorem 3. Under Assumptions 2.1, 2.2, 3.1, 3.5, &amp; 4.1, for any ξ ∈ (0 , 1) , there exist constants C 0 and C 1 such that for all ∥ X ∥ &lt; C 0 , C 1 &lt; R B 1 , R B 1 -ξ &lt; R B 2 &lt; R B 1 , the iterates { ¯ r t } and { w t } generated by Algorithm 2 satisfy

<!-- formula-not-decoded -->

where w ∗ η is the unique solution of ( ¯ A + ηI ) w -¯ b = 0 with

<!-- formula-not-decoded -->

If features are zero-centered (i.e., X ⊤ d µ = 0 ), then

<!-- formula-not-decoded -->

where ¯ q c π . = ¯ q π + c 1 .

We defer the proof to Section A.4. As the differential Bellman equation (1) has infinitely many solutions for ¯ q , all of which differ only by some constant offsets, we focus on analyzing the quality of Xw ∗ η w.r.t. ¯ q c π in Theorem 3. The zero-centered feature assumption is also used in Zhang et al. (2021), which can be easily fulfilled in practice by subtracting all features with the estimated mean. In the on-policy case (i.e., µ = π ), we have d ⊤ µ ( P π -I ) = 0 , indicating ¯ r ∗ η = ¯ r π , i.e., the regularization on the value estimate does not pose any bias on the reward rate estimate.

As shown by Zhang et al. (2021), if the update (3) converges, it converges to w ∗ 0 , the TD fixed point for off-policy policy evaluation in the average-reward setting, which satisfies ¯ Aw ∗ 0 + ¯ b = 0 . Theorem 3 shows that Algorithm 2 converges to a regularized TD fixed point. Though Zhang et al. (2021) give a bound on ∥ Xw ∗ 0 -¯ q c π ∥ , their bound holds only if µ is sufficiently close to π . By contrast, our bound on w ∗ η in Theorem 3 holds for all µ .

## 5. Application to Off-Policy Control

Discounted Setting . Introducing a target network and ridge regularization in (4) yields Q -learning with a Target Network (Algorithm 3), where the behavior policy µ θ depends on θ through the action-value estimate Xθ and can be any policy satisfying the following two assumptions.

Assumption 5.1. Let P be the closure of { P µ θ | θ ∈ R K } . For any P ∈ P , the Markov chain evolving in S×A induced by P is ergodic.

Assumption 5.2. µ θ ( a | s ) is Lipschitz continuous in X s θ , where X s ∈ R |A|× K is the feature matrix for the state s , i.e., its a -th row is x ( s, a ) ⊤ .

Assumption 5.1 is standard. When the behavior policy µ is fixed (independent of θ ), the induced chain is usually assumed to be ergodic when analyzing the behavior of Q -learning (see, e.g., Melo et al. (2008); Chen et al. (2019b); Cai et al. (2019)). In Algorithm 3, the behavior policy µ θ changes every step, so it is natural to assume that any of those behavior policies induces an ergodic chain. A similar assumption is also used by Zou et al. (2019) in their analysis of on-policy linear SARSA. Moreover, Zou et al. (2019) assume not only the ergodicity but also the uniform ergodicity of their sampling policies. Similarly, in Assumption 5.1, we assume ergodicity for not only all the transition matrices, but also their limits (c.f. the closure P ). A similar assumption is also used by Marbach &amp; Tsitsiklis (2001) in their analysis of on-policy actor-critic methods. Assumption 5.2 can be easily fulfilled, e.g., by using a softmax policy w.r.t. x ( s, · ) ⊤ θ .

## Algorithm 3 Q -learning with a Target Network

<!-- formula-not-decoded -->

end for

Theorem 4. Under Assumptions 2.2, 3.1, 3.5, 5.1, &amp; 5.2, for any ξ ∈ (0 , 1) , R B 1 &gt; R B 2 &gt; R B 1 -ξ &gt; 0 , there exists a constant C 0 such that for all ∥ X ∥ &lt; C 0 , the iterate { w t } generated by Algorithm 3 satisfies

<!-- formula-not-decoded -->

where w ∗ η is the unique solution of

<!-- formula-not-decoded -->

inside B 1 . Here

<!-- formula-not-decoded -->

and π w denotes the greedy policy w.r.t. x ( s, · ) ⊤ w .

Wedefer the proof to Section A.5. Analogously to the policy evaluation setting, if we call the solutions of A π w ,µ w w -b µ w = 0 TD fixed points for control in the discounted setting, then Theorem 4 asserts that Algorithm 3 finds a regularized TD fixed point.

Algorithm 3 and Theorem 4 are significant in two aspects. First , in Algorithm 3, the behavior policy is a function of the target network and thus changes every time step. By contrast, previous work on Q -learning with function approximation (e.g., Melo et al. (2008); Maei et al. (2010); Chen et al. (2019b); Cai et al. (2019); Chen et al. (2019a); Lee &amp; He (2019); Xu &amp; Gu (2020); Carvalho et al. (2020); Wang &amp;Zou (2020)) usually assumes the behavior policy is fixed. Though Fan et al. (2020) also adopt a changing behavior policy, they consider bi-level optimization. At each time step, the nested optimization problem must be solved exactly, which is computationally expensive and sometimes unfeasible. To the best of our knowledge, we are the first to analyze Q -learning with function approximation under a changing behavior policy and without nested optimization problems. Compared with the fixed behavior policy setting or the bi-level optimization setting, our two-timescale setting with a changing behavior policy is more closely related to actual practice (e.g., Mnih et al. (2015); Lillicrap et al. (2015)).

Second , Theorem 4 does not enforce any similarity between µ θ and π w ; they can be arbitrarily different. By contrast, previous work (e.g., Melo et al. (2008); Chen et al. (2019b); Cai et al. (2019); Xu &amp; Gu (2020); Lee &amp; He (2019)) usually requires the strong assumption that the fixed behavior policy µ is sufficiently close to the target policy π w . As the target policy (i.e., the greedy policy) can change every time step due to the changing action-value estimates, this strong assumption rarely holds. While some work removes this strong assumption, it introduces other problems instead. In Greedy-GQ, Maei et al. (2010) avoid this strong assumption by computing sub-gradients of an MSPBE objective MSPBE ( w ) . = ∥ A π w ,µ w -b µ ∥ 2 C -1 µ directly, where C µ . = X ⊤ D µ X . If linear Q -learning (4) under a fixed behavior policy µ converges, it converges to the minimizer of MSPBE ( w ) . Greedy-GQ, however, converges only to a stationary point of MSPBE ( w ) . By contrast, Algorithm 3 converges to a minimizer of our regularized MSPBE (c.f. (11)). In Coupled Q -learning, Carvalho et al. (2020) avoid this strong assumption by using a target network as well, which they update as

<!-- formula-not-decoded -->

This target network update deviates much from the commonly used Polyak-averaging style update, while our (8) is identical to the Polyak-averaging style update most times if the balls for projection are sufficiently large. Coupled Q -learning updates the main network w as usual (see (6)). With the Coupled Q -learning updates (6) and (12), Carvalho et al. (2020) prove that the main network and the target network converge to ¯ w and ¯ θ respectively, which satisfy

<!-- formula-not-decoded -->

It is, however, not clear how ¯ w and ¯ θ relate to TD fixed points. Yang et al. (2019) also use a target network to avoid this strong assumption. Their target network update is the same as (8) except that they have only one projection Γ B 1 . Consequently, they face the problem of the reflection term ζ ( t ) (c.f. (9)). They also assume the main network { w t } is always bounded, a strong assumption that we do not require. Moreover, they consider a fixed sampling distribution for obtaining i.i.d. samples, while our data collection is done by executing the changing behavior policy µ θ in the MDP.

One limit of Theorem 4 is that the bound on ∥ X ∥ (i.e., C 0 ) depends on 1 /R B 1 (see the proof in Section A.5 for the analytical expression), which means C 0 could potentially be small. Though we can use a small η accordingly to ensure that the regularization effect of η is modest, a small C 0 may not be desirable in some cases. To address this issue, we propose Gradient Q -learning with a Target Network, inspired by Greedy-GQ. We first equip MSPBE ( w ) with a changing behavior policy µ w , yielding the following objective ∥ A π w ,µ w w -b µ w ∥ 2 C -1 µw . We then use the target network θ in place of w in the non-convex components, yielding

<!-- formula-not-decoded -->

where we have also introduced a ridge term. At time step t , we update w t following the gradient ∇ w L ( w,θ t ) and update the target network θ t as usual. Details are provided in Algorithm 4, where the additional weight vector u ∈ R K results from a weight duplication trick (see Sutton et al. (2009b;a) for details) to address a double sampling issue in estimating ∇ w L ( w,θ ) .

## Algorithm 4 Gradient Q -learning with a Target Network

<!-- formula-not-decoded -->

In Algorithm 3, the target policy π w is a greedy policy, which is not continuous in w . This discontinuity is not a problem there but requires sub-gradients in the analysis of Algorithm 4, which complicates the presentation. We, therefore, impose Assumption 5.2 on π w as well.

Assumption 5.3. π θ ( a | s ) is Lipschitz continuous in X s θ .

Though a greedy policy no longer satisfies Assumption 5.3, we can simply use a softmax policy with any temperature.

Theorem 5. Under Assumptions 2.2, 3.1, 3.5, &amp; 5.1-5.3, there exist positive constants C 0 and C 1 such that for all ∥ X ∥ &lt; C 0 , R B 1 &gt; R B 2 &gt; C 1 , the iterate { w t } generated by Algorithm 4 satisfies

<!-- formula-not-decoded -->

where w ∗ η is the unique solution of

<!-- formula-not-decoded -->

We defer the proof to Section A.6. Importantly, the C 0 here does not depend on R B 1 and R B 2 . More importantly, the condition on ∥ X ∥ (or equivalently, η ) in Theorem 5 is only used to fulfill Assumption 3.4, without which { θ t } in Algorithm 4 still converges to an invariant set of the ODE (10). This condition is to investigate where the iterate converges to instead of whether it converges or not. If we assume w ∗ 0 . = lim η → 0 w ∗ η exists and A π w ∗ 0 ,µ w ∗ 0 is invertible, we can see A π w ∗ 0 ,µ w ∗ 0 w ∗ 0 -b µ w ∗ 0 = 0 , indicating w ∗ 0 is a TD fixed point. w ∗ η can therefore be regarded as a regularized TD fixed point, though how the regularization is imposed here (c.f. (13)) is different from that in Algorithm 3 (c.f. (11)).

Average-reward Setting . Similar to Algorithm 2, introducing a target network and ridge regularization in (5) yields Differential Q -learning with a Target Network (Algorithm 5). Similar to Algorithm 2, { B i } are now balls in R K +1 .

## Algorithm 5 Diff. Q -learning with a Target Network

<!-- formula-not-decoded -->

Theorem 6. Under Assumptions 2.2, 3.1, 3.5, 5.1, &amp; 5.2, let L µ denote the Lipschitz constant of µ θ , for any ξ ∈ (0 , 1) , R B 1 &gt; R B 2 &gt; R B 1 -ξ &gt; 0 , there exist constants C 0 and C 1 such that for all ∥ X ∥ &lt; C 0 , L µ &lt; C 1 , the iterate { w t } generated by Algorithm 5 satisfies

<!-- formula-not-decoded -->

where w ∗ η is the unique solution of ( ¯ A π w ,µ w + ηI ) w -¯ b µ w = 0 inside B 1 , where

<!-- formula-not-decoded -->

and π w is a greedy policy w.r.t. x ( s, · ) ⊤ w .

We defer the proof to Section A.7. Theorem 6 requires µ θ to be sufficiently smooth, which is a standard assumption even in the on-policy setting (e.g., Melo et al. (2008); Zou et al. (2019)). It is easy to see that if (5) converges, it converges to a solution of ¯ A π w ,µ w w -¯ b µ w = 0 , which we call a TD fixed point for control in the average-reward setting. Theorem 6, which shows that Algorithm 5 finds a regularized TD fixed point, is to the best of our knowledge the first theoretical study for linear Q -learning in the average-reward setting.

## 6. Experiments

All the implementations are publicly available. 1

We first use Kolter's example (Kolter, 2011) to investigate how η influences the performance of w ∗ η in the policy evaluation setting. Details are provided in Section D.1. This example is a two-state MDP with small representation error (i.e., ∥ ∥ Π D µ v π -v π ∥ ∥ is small). We vary the sampling probability of one state ( d µ ( s 1 ) ) and compute corresponding w ∗ η analytically. Figure 1a shows that with η = 0 , the performance of w ∗ η becomes arbitrarily poor when d µ ( s 1 ) approaches around 0.71. With η = 0 . 01 , the spike exists as well. If we further increase η to 0 . 02 and 0 . 03 , the performance for w ∗ η becomes well bounded. This confirms the potential advantage of the regularized TD fixed points.

We then use Baird's example (Baird, 1995) to empirically investigate the convergence of the algorithms we propose. We use exactly the same setup as Chapter 11.2 of Sutton &amp; Barto (2018). Details are provided in Section D.2. In particular, we consider three settings: policy evaluation (Figure 1b), control with a fixed behavior policy (Figure 1c), and control with an action-value dependent behavior policy (Figure 1d). For the policy evaluation setting, we compare a TD version of Algorithm 1 and standard Off-Policy Linear TD (possibly with ridge regularization). For the two control settings, we compare Algorithm 3 with standard linear Q -learning (possibly with ridge regularization). We use constant learning rates and do not use any projection in all the compared algorithms. The exact update rules are provided in Section D.2. Interestingly, Figures 1b-d show that even with η = 0 , i.e., no ridge regularization, our algorithms with target network still converge in the tested domains. By contrast, without a target network, even when mild regularization is imposed, standard off-policy algorithms still

1 https://github.com/ShangtongZhang/DeepRL

Figure 1. (a) Effect of regularization on Kolter's example. v π is the true state-value function. (b) Policy evaluation on Baird's example. (c) Control on Baird's example with a fixed behavior policy. (d) Control on Baird's example with an action-value-dependent behavior policy. In (b)(c)(d), the curves are averaged over 30 independent runs with shaded region indicating one standard deviation. q ∗ is the optimal action-value function. η is the weight for the ridge term. Those marked 'ours' are curves of algorithms we propose; those marked 'standard' are standard semi-gradient off-policy algorithms. Interestingly, the three 'standard' curves overlap and get unbounded quickly.

<!-- image -->

diverge. This confirms the importance of the target network.

## 7. Discussion and Related Work

For all the algorithms we propose, both the target network and the ridge regularization are at play. One may wonder if it is possible to ensure convergence with only ridge regularization without the target network. In the policy evaluation setting, the answer is affirmative. Applying ridge regularization in (2) directly yields

<!-- formula-not-decoded -->

where δ t is defined in (2). The expected update of (14) is

<!-- formula-not-decoded -->

If its Jacobian w.r.t. w , denoted as J w (∆ w ) , is negative definite, the convergence of { w t } is expected (see, e.g., Section 5.5 of Vidyasagar (2002)). This negative definiteness can be easily achieved by ensuring η &gt; ∥ X ∥ 2 ∥ D µ ( I -γP π ) ∥ (see Diddigi et al. (2019) for similar techniques). This direct ridge regularization, however, would not work in the control setting. Consider, for example, linear Q -learning with ridge regularization (i.e., (14) with δ t defined in (4)). The Jacobian of its expected update is J w ( b µ w -( A π w ,µ w + ηI ) w ) . It is, however, not clear how to ensure this Jacobian is negative definite by tuning η . By using a target network for bootstrapping, P π Xw becomes P π Xθ . So J w (∆ w ) becomes -J w ( X ⊤ D µ Xw + ηw ) , which is always negative definite. Similarly, J w ( b µ w -( A π w ,µ w + η ) w ) becomes -J w ( X ⊤ D µ θ Xw + ηw ) in Algorithm 3, which is always negative definite regardless of θ . The convergence of the main network { w t } can, therefore, be expected. The convergence of the target network { θ t } is then delegated to Theorem 1. Now it is clear that in the deadly triad setting, the target network stabilizes training by ensuring the Jacobian of the expected update is negative definite. One may also wonder if it is possible to ensure convergence with only the target network without ridge regularization. The answer is unclear. In our analysis, the conditions on ∥ X ∥ (or equivalently, η ) are only sufficient and not necessarily necessary. We do see in Figure 1 that even with η = 0 , our algorithms still converge in the tested domains. How small η can be in general and under what circumstances η can be 0 are still open problems, which we leave for future work. Further, ridge regularization usually affects the convergence rate of the algorithm, which we also leave for future work.

In this paper, we investigate target network as one possible solution for the deadly triad. Other solutions include Gradient TD methods (Sutton et al. (2009b;a; 2016) for the discounted setting; Zhang et al. (2021) for the averagereward setting) and Emphatic TD methods (Sutton et al. (2016) for the discounted setting). Other convergence results of Q -learning with function approximation include Tsitsiklis &amp; Van Roy (1996); Szepesv´ ari &amp; Smart (2004), which require special approximation architectures, Wen &amp; Van Roy (2013); Du et al. (2020), which consider deterministic MDPs, Li et al. (2011); Du et al. (2019), which require a special oracle to guide exploration, Chen et al. (2019a), which require matrix inversion every time step, and Wang et al. (2019); Yang &amp; Wang (2019; 2020); Jin et al. (2020), which consider linear MDPs (i.e., both p and r are assumed to be linear). Achiam et al. (2019) characterize the divergence of Q -learning with nonlinear function approximation via Taylor expansions and use preconditioning to empirically stabilize training. Van Hasselt et al. (2018) empirically study the role of a target network in the deadly triad setting in deep RL, which is complementary to our theoretical analysis.

Regularization is also widely used in RL. Yu (2017) introduce a general regularization term to improve the robustness of Gradient TD algorithms. Du et al. (2017) use ridge regularization in MSPBE to improve its convexity. Zhang et al.

(2020) use ridge regularization to stabilize the training of critic in an off-policy actor-critic algorithm. Kolter &amp; Ng (2009); Johns et al. (2010); Petrik et al. (2010); PainterWakefield et al. (2012); Liu et al. (2012) use Lasso regularization in policy evaluation, mainly for feature selection.

## 8. Conclusion

In this paper, we proposed and analyzed a novel target network update rule, with which we improved several linear RL algorithms that are known to diverge previously due to the deadly triad. Our analysis provided a theoretical understanding, in the deadly triad setting, of the conventional wisdom that a target network stabilizes training. A possibility for future work is to introduce nonlinear function approximation, possibly over-parameterized neural networks, into our analysis.

## Acknowledgments

The authors thank Handong Lim for an insightful discussion. SZ is generously funded by the Engineering and Physical Sciences Research Council (EPSRC). SZ was also partly supported by DeepDrive. Inc from September to December 2020 during an internship. This project has received funding from the European Research Council under the European Union's Horizon 2020 research and innovation programme (grant agreement number 637713). The experiments were made possible by a generous equipment grant from NVIDIA.

## References

- Achiam, J., Knight, E., and Abbeel, P. Towards characterizing divergence in deep q-learning. arXiv preprint arXiv:1903.08894 , 2019.
- Baird, L. Residual algorithms: Reinforcement learning with function approximation. Machine Learning , 1995.
- Borkar, V. S. Stochastic approximation: a dynamical systems viewpoint . Springer, 2009.
- Boyan, J. A. Least-squares temporal difference learning. In Proceedings of the 16th International Conference on Machine Learning , 1999.
- Cai, Q., Yang, Z., Lee, J. D., and Wang, Z. Neural temporaldifference and q-learning provably converge to global optima. arXiv preprint arXiv:1905.10027 , 2019.
- Carvalho, D., Melo, F. S., and Santos, P. A new convergent variant of q-learning with linear function approximation. Advances in Neural Information Processing Systems , 33, 2020.
- Chen, S., Devraj, A. M., Buˇ si´ c, A., and Meyn, S. Zap qlearning with nonlinear function approximation. arXiv preprint arXiv:1910.05405 , 2019a.
- Chen, Z., Zhang, S., Doan, T. T., Clarke, J.-P., and Maguluri, S. T. Finite-sample analysis of nonlinear stochastic approximation with applications in reinforcement learning. arXiv preprint arXiv:1905.11425 , 2019b.
- Diddigi, R. B., Kamanchi, C., and Bhatnagar, S. A convergent off-policy temporal difference algorithm. arXiv preprint arXiv:1911.05697 , 2019.
- Du, S. S., Chen, J., Li, L., Xiao, L., and Zhou, D. Stochastic variance reduction methods for policy evaluation. In Proceedings of the 34th International Conference on Machine Learning , 2017.
- Du, S. S., Luo, Y., Wang, R., and Zhang, H. Provably efficient q-learning with function approximation via distribution shift error checking oracle. In Advances in Neural Information Processing Systems , 2019.
- Du, S. S., Lee, J. D., Mahajan, G., and Wang, R. Agnostic q -learning with function approximation in deterministic systems: Near-optimal bounds on approximation error and sample complexity. Advances in Neural Information Processing Systems , 2020.
- Dulac-Arnold, G., Mankowitz, D., and Hester, T. Challenges of real-world reinforcement learning. arXiv preprint arXiv:1904.12901 , 2019.
- Fan, J., Wang, Z., Xie, Y., and Yang, Z. A theoretical analysis of deep q-learning. In Learning for Dynamics and Control . PMLR, 2020.
- Fujimoto, S., van Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477 , 2018.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290 , 2018.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pp. 2137-2143. PMLR, 2020.
- Johns, J., Painter-Wakefield, C., and Parr, R. Linear complementarity for regularized policy evaluation and improvement. Advances in neural information processing systems , 2010.
- Kolter, J. Z. The fixed points of off-policy td. In Advances in Neural Information Processing Systems , 2011.

- Kolter, J. Z. and Ng, A. Y. Regularization and feature selection in least-squares temporal difference learning. In Proceedings of the 26th annual international conference on machine learning , 2009.
- Konda, V. R. Actor-critic algorithms . PhD thesis, Massachusetts Institute of Technology, 2002.
- Kushner, H. and Yin, G. G. Stochastic approximation and recursive algorithms and applications . Springer Science &amp;Business Media, 2003.
- Lee, D. and He, N. A unified switching system perspective and ode analysis of q-learning algorithms. arXiv preprint arXiv:1912.02270 , 2019.
- Li, L., Littman, M. L., Walsh, T. J., and Strehl, A. L. Knows what it knows: a framework for self-aware learning. Machine learning , 2011.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 , 2015.
- Lin, L.-J. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning , 1992.
- Liu, B., Mahadevan, S., and Liu, J. Regularized off-policy td-learning. Advances in Neural Information Processing Systems , 2012.
- Maei, H. R., Szepesv´ ari, C., Bhatnagar, S., and Sutton, R. S. Toward off-policy learning control with function approximation. In Proceedings of the 27th International Conference on Machine Learning , 2010.
- Marbach, P. and Tsitsiklis, J. N. Simulation-based optimization of markov reward processes. IEEE Transactions on Automatic Control , 2001.
- Melo, F. S., Meyn, S. P., and Ribeiro, M. I. An analysis of reinforcement learning with function approximation. In Proceedings of the 25th International Conference on Machine Learning , 2008.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. Nature , 2015.
- Painter-Wakefield, C., Parr, R., and Durham, N. L1 regularized linear temporal difference learning. Technical report: Department of Computer Science, Duke University, Durham, NC, TR-2012-01 , 2012.
- Petrik, M., Taylor, G., Parr, R., and Zilberstein, S. Feature selection using regularization in approximate linear programs for markov decision processes. arXiv preprint arXiv:1005.1860 , 2010.
- Puterman, M. L. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014.
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. Mastering the game of go with deep neural networks and tree search. Nature , 2016.
- Sutton, R. S. Learning to predict by the methods of temporal differences. Machine Learning , 1988.
- Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction (2nd Edition) . MIT press, 2018.
- Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesv´ ari, C., and Wiewiora, E. Fast gradientdescent methods for temporal-difference learning with linear function approximation. In Proceedings of the 26th International Conference on Machine Learning , 2009a.
- Sutton, R. S., Maei, H. R., and Szepesv´ ari, C. A convergent o ( n ) temporal-difference algorithm for off-policy learning with linear function approximation. In Advances in Neural Information Processing Systems , 2009b.
- Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., and Precup, D. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In Proceedings of the 10th International Conference on Autonomous Agents and Multiagent Systems , 2011.
- Sutton, R. S., Mahmood, A. R., and White, M. An emphatic approach to the problem of off-policy temporal-difference learning. The Journal of Machine Learning Research , 2016.
- Szepesv´ ari, C. and Smart, W. D. Interpolation-based qlearning. In Proceedings of the twenty-first international conference on Machine learning , pp. 100, 2004.
- Tikhonov, A. N., Goncharsky, A., Stepanov, V., and Yagola, A. G. Numerical methods for the solution of ill-posed problems . Springer Science &amp; Business Media, 2013.
- Tsitsiklis, J. N. and Van Roy, B. Feature-based methods for large scale dynamic programming. Machine Learning , 1996.
- Tsitsiklis, J. N. and Van Roy, B. Analysis of temporaldiffference learning with function approximation. In Advances in Neural Information Processing Systems , 1997.

- Van Hasselt, H., Doron, Y., Strub, F., Hessel, M., Sonnerat, N., and Modayil, J. Deep reinforcement learning and the deadly triad. arXiv preprint arXiv:1812.02648 , 2018.
- Vidyasagar, M. Nonlinear systems analysis . SIAM, 2002.
- Wan, Y., Naik, A., and Sutton, R. S. Learning and planning in average-reward markov decision processes. arXiv preprint arXiv:2006.16318 , 2020.
- Wang, Y. and Zou, S. Finite-sample analysis of greedygq with linear function approximation under markovian noise. arXiv preprint arXiv:2005.10175 , 2020.
- Wang, Y., Wang, R., Du, S. S., and Krishnamurthy, A. Optimism in reinforcement learning with generalized linear function approximation. arXiv preprint arXiv:1912.04136 , 2019.
- Watkins, C. J. and Dayan, P. Q-learning. Machine Learning , 1992.
- Wen, Z. and Van Roy, B. Efficient exploration and value function generalization in deterministic systems. Advances in Neural Information Processing Systems , 2013.
- Wu, Y., Zhang, W., Xu, P., and Gu, Q. A finite time analysis of two time-scale actor critic methods. arXiv preprint arXiv:2005.01350 , 2020.
- Xu, P. and Gu, Q. A finite-time analysis of q-learning with neural network function approximation. In International Conference on Machine Learning , pp. 1055510565. PMLR, 2020.
- Yang, L. and Wang, M. Reinforcement learning in feature space: Matrix bandit, kernels, and regret bound. In International Conference on Machine Learning , pp. 1074610756. PMLR, 2020.
- Yang, L. F. and Wang, M. Sample-optimal parametric qlearning using linearly additive features. arXiv preprint arXiv:1902.04779 , 2019.
- Yang, Z., Fu, Z., Zhang, K., and Wang, Z. Convergent reinforcement learning with function approximation: A bilevel optimization perspective, 2019. URL https: //openreview.net/forum?id=ryfcCo0ctQ .
- Yu, H. Convergence of least squares temporal difference methods under general conditions. In ICML , 2010.
- Yu, H. On convergence of some gradient-based temporaldifferences algorithms for off-policy learning. arXiv preprint arXiv:1712.09652 , 2017.
- Zhang, S., Liu, B., Yao, H., and Whiteson, S. Provably convergent two-timescale off-policy actor-critic with function approximation. In Proceedings of the 37th International Conference on Machine Learning , 2020.
- Zhang, S., Wan, Y., Sutton, R. S., and Whiteson, S. Averagereward off-policy policy evaluation with function approximation. arXiv preprint arXiv:2101.02808 , 2021.
- Zou, S., Xu, T., and Liang, Y. Finite-sample analysis for sarsa with linear function approximation. In Advances in Neural Information Processing Systems , 2019.

## A. Convergence of Target Networks

We first state a result from Borkar (2009) regarding the convergence of a linear system. Consider updating the parameter y ∈ R K recursively as

<!-- formula-not-decoded -->

where h : R K → R K and { ϵ t } is a deterministic or random bounded sequence satisfying lim t →∞ ∥ ϵ t ∥ = 0 . Assuming

Assumption A.1. h is Lipschitz continuous.

Assumption A.2. The learning rates { β t } satisfies ∑ t β t = ∞ , ∑ t β 2 t &lt; ∞ .

Theorem 7. (The third extension of Theorem 2 in Chapter 2 of Borkar (2009)) Under Assumptions A.1- A.3, almost surely, the sequence { y t } generated by (15) converges to a compact connected internally chain transitive invariant set of the ODE

<!-- formula-not-decoded -->

Assumption A.3. sup t ∥ y t ∥ &lt; ∞ almost surely.

## A.1. Proof of Theorem 1

Proof. Similar to Chapter 5.4 of Borkar (2009), we consider ˙ Γ B 1 , the directional derivative of Γ B 1 . At a point x ∈ R K , given a direction y ∈ R K , we have

<!-- formula-not-decoded -->

where int ( B 1 ) is the interior of B 1 , ∂B 1 is the boundary of B 1 , F x ( B 1 ) . = { y ∈ R K | ∃ δ &gt; 0 , s.t. x + δy ∈ B 1 } is the feasible directions of B 1 w.r.t. x . The first two cases are trivial and are easy to deal with. The third case is complicated and is the source of the reflection term ζ ( t ) in (9). However, thanks to the projection Γ B 2 , we succeeded in getting rid of it.

By (8), θ t ∈ B 1 always holds. With the directional derivative, we can rewrite the update rule of { θ t } as

<!-- formula-not-decoded -->

We now compute ˙ Γ B 1 ( θ t , Γ B 2 ( w t ) -θ t ) . We proceed by showing that only the first two cases in ˙ Γ B 1 ( x, y ) can happen and the third case will never occur.

For θ t ∈ int ( B 1 ) , we have

For θ t ∈ ∂B 1 ,

Let y 0 . = Γ B 2 ( w t ) -θ t , (17) implies that we can decompose y 0 as y 0 = y 1 + y 2 , where ⟨ θ t , y 1 ⟩ = 0 and ⟨ θ t , y 2 ⟩ = -∥ θ t ∥∥ y 2 ∥ . Here y 2 is the projection of y 0 onto θ t , which is in the opposite direction of θ t and y 1 is the remaining orthogonal component. By Pythagoras's theorem, for any δ &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For sufficiently small δ , e.g., δ 2 ∥ y 1 ∥ 2 -2 δ ∥ θ t ∥∥ y 2 ∥ + δ 2 ∥ y 2 ∥ 2 &lt; 0 , we have

<!-- formula-not-decoded -->

implying Γ B 2 ( w t ) -θ t ∈ F θ t ( B 1 ) . So we have

<!-- formula-not-decoded -->

Combining (16) and (18) yields

<!-- formula-not-decoded -->

Assumption A.1 is verified by Assumption 3.4; Assumption A.2 is verified by Assumption 3.1; Assumption A.3 is verified directly by the projection in (8). By Theorem 7, almost surely, { θ t } converges to a compact connected internally chain transitive invariant set of the ODE

<!-- formula-not-decoded -->

Under Assumption 3.4, the Banach fixed-point theorem asserts that there is a unique θ ∗ satisfying w ∗ ( θ ∗ ) = θ ∗ , i.e., θ ∗ is the unique equilibrium of the ODE above. We now show θ ∗ is globally asymptotically stable. Consider the candidate Lyapunov function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϱ &lt; 1 is the Lipschitz constant of w ∗ . It is easy to see

- V ( θ ) ≥ 0
- V ( θ ) = 0 ⇐⇒ θ = θ ∗
- d d t V ( θ ( t )) ≤ 0
- d d t V ( θ ( t )) = 0 ⇐⇒ θ = θ ∗

Consequently, θ ∗ is globally asymptotically stable, implying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have

## A.2. Proof of Lemma 1

Proof. By (8), θ t ∈ B 1 holds for all t . So

<!-- formula-not-decoded -->

## A.3. Proof of Theorem 2

Proof. Consider the Markov process Y t . = ( S t , A t , S t +1 ) . By Assumption 4.1, Y t adopts a unique stationary distribution, which we refer to as d Y . We have d Y ( s, a, s ′ ) = d µ ( s ) µ ( a | s ) p ( s ′ | s, a ) . We define

<!-- formula-not-decoded -->

As θ t ∈ B 1 holds for all t , we can rewrite the update of w t in Algorithm 1 as

<!-- formula-not-decoded -->

The asymptotic behavior of { w t } is then governed by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now verify Assumptions 3.2, 3.3, &amp; 3.4 to invoke Theorem 1.

Assumption 3.2 is verified in Lemma 2.

To verify Assumption 3.4, we use SVD and get

Define

<!-- formula-not-decoded -->

where U, V are two orthogonal matrices, Σ = [ Σ + 0 ] is a rectangular diagonal matrix with Σ + . = diag ([ . . . , σ i , . . . ]) being a diagonal matrix. Assumptions 4.1 &amp; 2.2 imply that σ i &gt; 0 . We have

<!-- formula-not-decoded -->

According to (19), it is then easy to see

<!-- formula-not-decoded -->

Take any ξ ∈ (0 , 1) , assuming then

Assumption 3.4, therefore, holds.

We now select proper R B 1 and R B 2 to fulfill Assumption 3.3. Plugging (20) and (21) into (19) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have sup θ ∥ w ∗ ( θ ) ∥ ≤ R B 1 -ξ . Selecting R B 2 ∈ ( R B 1 -ξ, R B 1 ) then fulfills Assumption 3.3.

With Assumptions 3.1 - 3.4 satisfied, Theorem 1 then implies that there exists a unique θ ∞ such that

<!-- formula-not-decoded -->

Next we show what θ ∞ is. We define

<!-- formula-not-decoded -->

Note this is just the right side of equation (19) without the projection. (20) and (21) imply that f is a contraction. The Banach fixed-point theorem then asserts that f adopts a unique fixed point, which we refer to as w ∗ η . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For sufficiently large R B 1 , e.g.,

Then for sufficiently large R B 1 , e.g.,

<!-- formula-not-decoded -->

we have w ∗ η = Γ B 1 ( w ∗ η ) , implying w ∗ η is a fixed point of w ∗ ( · ) (i.e., the right side of (19)) as well. As w ∗ ( · ) is a contraction, we have θ ∞ = w ∗ η . Rewriting f ( w ∗ η ) = w ∗ η yields

<!-- formula-not-decoded -->

In other words, w ∗ η is the unique (due to the contraction of f ) solution of ( A + ηI ) w -b = 0 . Combining (21), (22), and (23), the desired constants are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now bound ∥ ∥ Xw ∗ η -q π ∥ ∥ . For any y ∈ R |S| , we define the ridge regularized projection Π η D µ as

<!-- formula-not-decoded -->

Π η D µ is connected with f as Π η D µ T π ( Xw ) = Xf ( w ) . We have

<!-- formula-not-decoded -->

The above equation implies

<!-- formula-not-decoded -->

where Π D µ is shorthand for Π η =0 D µ . We now bound ∥ ∥ ∥ Π η D µ -Π D µ ∥ ∥ ∥ .

<!-- formula-not-decoded -->

( σ max ( · ) and σ min ( · ) indicate the largest and smallest singular values)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we arrive at

.

<!-- formula-not-decoded -->

which completes the proof.

## A.4. Proof of Theorem 3

Proof. The proof is similar to the proof of Theorem 2 in Section A.3. We, therefore, highlight only the difference to avoid verbatim repetition. Define

<!-- formula-not-decoded -->

We can then rewrite the update of ¯ r and w in Algorithm 2 as

<!-- formula-not-decoded -->

Similarly, we define then

Assumption 3.4, therefore, holds.

We now select proper R B 1 and R B 2 to fulfill Assumption 3.3. Using (25), it is easy to see

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have sup θ ∥ w ∗ ( θ ) ∥ ≤ R B 1 -ξ . Selecting R B 2 ∈ ( R B 1 -ξ, R B 1 ) then fulfills Assumption 3.3.

<!-- formula-not-decoded -->

We proceed to verifying Assumptions 3.2, 3.3, &amp; 3.4 to invoke Theorem 1.

Assumption 3.2 is verified in Lemma 3.

For Assumption 3.4 to hold, note

<!-- formula-not-decoded -->

The above equations suggest that

<!-- formula-not-decoded -->

Take any ξ ∈ (0 , 1) , assuming

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For sufficiently large R B 1 , e.g.,

With Assumptions 3.1 - 3.4 satisfied, Theorem 1 then implies that there exists a unique θ ∞ such that

<!-- formula-not-decoded -->

Next we show what θ ∞ is. We define

<!-- formula-not-decoded -->

Note this is just u ∗ ( θ ) without the projection. Under (25), it is easy to show f is a contraction. The Banach fixed-point theorem then asserts that f adopts a unique fixed point, which we refer to as u ∗ η . Using (25) again, we get

<!-- formula-not-decoded -->

Then for sufficiently large R B 1 , e.g.,

<!-- formula-not-decoded -->

we have u ∗ η = Γ B 1 ( u ∗ η ) , implying u ∗ η is a fixed point of u ∗ ( · ) as well. As u ∗ ( · ) is a contraction, we have θ ∞ = u ∗ η . Writing u ∗ η as [ ¯ r ∗ η w ∗ η ] and expanding f ( u ∗ η ) = u ∗ η yields

<!-- formula-not-decoded -->

Rearranging terms yields ( ¯ A + ηI ) w ∗ η -¯ b = 0 , i.e., w ∗ η is the unique (due to the contraction of f ) solution of ( ¯ A + ηI ) w -¯ b = 0 .

We now bound ∥ ∥ Xw ∗ η -¯ q c π ∥ ∥ .

<!-- formula-not-decoded -->

Assuming we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is then easy to see

<!-- formula-not-decoded -->

Taking infimum for c ∈ R then yields the desired results.

Combining (25), (26), (27), and (28), the desired constants are

<!-- formula-not-decoded -->

which completes the proof.

## A.5. Proof of Theorem 4

Proof. The proof is similar to the proof of Theorem 2 in Section A.3 but is more involving. We define

<!-- formula-not-decoded -->

where ¯ θ . = Γ B 1 ( θ ) is shorthand. As θ t ∈ B 1 holds for all t , we can rewrite the update of w t in Algorithm 3 as

<!-- formula-not-decoded -->

The expected update given θ is then controlled by

<!-- formula-not-decoded -->

where Assumption 5.1 ensures the existence of d µ θ and π θ is the target policy, i.e. a greedy policy with random tie breaking defined as follows.

Let A max s,θ . = arg max a x ( s, a ) ⊤ θ be the set of maximizing actions for state s , we define

<!-- formula-not-decoded -->

Similar to the proof in Section A.3, we define

<!-- formula-not-decoded -->

and proceed to verify Assumptions 3.2 - 3.4 to invoke Theorem 1.

Assumption 3.2 is proved in Lemma 4.

For Assumption 3.4, Lemma 10 shows that w ∗ ( θ ) is Lipschitz continuous in θ with

<!-- formula-not-decoded -->

being a Lipschitz constant. Here L D , L 0 , and U P are positive constants detailed in the proof of Lemma 10. Assuming

<!-- formula-not-decoded -->

we have

Take any ξ ∈ (0 , 1) , assuming

<!-- formula-not-decoded -->

it then follows that C w ≤ 1 -ξ . Assumptions 3.4, therefore, holds.

We now select proper R B 1 and R B 2 to fulfill Assumption 3.3. Similar to Lemma 10 (see, e.g., the last three rows of Table 1 in the proof of Lemma 10), we can easily get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have sup θ ∥ w ∗ ( θ ) ∥ &lt; R B 1 -ξ . Taking R B 2 ∈ ( R B 1 -ξ, R B 1 ) then fulfills Assumption 3.3.

With Assumptions 3.1 - 3.4 satisfied, Theorem 1 implies that there exists a unique θ ∞ such that

<!-- formula-not-decoded -->

We now show what θ ∞ is. We define

<!-- formula-not-decoded -->

and consider a ball B 0 . = { θ ∈ R K | ∥ θ ∥ ≤ R B 0 } with R B 0 to be tuned (for the Brouwer fixed-point theorem). We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for sufficiently large R B 0 , e.g., we have

Using (30) yields

For sufficiently large R B 1 , e.g.,

Assuming we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Brouwer fixed-point theorem then asserts that there exists a w ∗ η ∈ B 0 such that f ( w ∗ η ) = w ∗ η . For sufficiently large R B 1 , e.g.,

<!-- formula-not-decoded -->

we have Γ B 1 ( w ∗ η ) = w ∗ η , i.e., w ∗ η is also a fixed point of w ∗ ( · ) . The contraction of w ∗ ( · ) then implies θ ∞ = w ∗ η . Rewriting f ( w ∗ η ) = w ∗ η yields

<!-- formula-not-decoded -->

In other words, w ∗ η is the unique solution of ( A π w ,µ w + ηI ) w -b µ w = 0 inside B 1 (due to the contraction of w ∗ ( · ) ). Combining (30) (31) (33) (32) (34), the desired constant is

<!-- formula-not-decoded -->

which completes the proof. As R B 1 is usually large, in general C 0 is O ( R -1 B 1 ) . Though C 0 is potentially small, we can use small η as well. So a small C 0 (i.e., ∥ X ∥ ) does not necessarily implies a large regularization bias.

## A.6. Proof of Theorem 5

Proof. Let κ t . = [ u ⊤ t , w ⊤ t ] ⊤ , Algorithms 4 implies that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [ · ] K +1:2 K is the subvector indexed from K +1 to 2 K .

We proceed to verifying Assumptions 3.2, 3.3, &amp; 3.4 thus invoke Theorem 1.

Assumption 3.2 is verified in Lemma 5. Lemma 11 shows that if ∥ X ∥ ≤ 1 , there exist a constant L w &gt; 0 , which depends on X through only X ∥ X ∥ , such that

<!-- formula-not-decoded -->

As w ∗ ( · ) is independent of R B 1 , so does L w . So as long as

<!-- formula-not-decoded -->

w ∗ ( · ) is contractive and Assumption 3.4 is satisfied. Since L w depends on X only through X ∥ X ∥ , there are indeed X satisfying (36). E.g., if some X ′ does not satisfy (36), we can simply scale X ′ down by some scalar. In the proof of Lemma 11, we show sup θ ∥ ∥ C -1 µ θ ∥ ∥ &lt; ∞ . It is then easy to see sup θ ∥ w ∗ ( θ ) ∥ &lt; ∞ . Consequently, we can choose sufficiently large R B 1 and R B 2 such that Assumption 3.3 holds.

With Assumptions 3.1 - 3.4 satisfied, Theorem 1 implies that there exists a unique w ∗ η such that

<!-- formula-not-decoded -->

Expanding w ∗ ( w ∗ η ) = w ∗ η yields

<!-- formula-not-decoded -->

which completes the proof.

## A.7. Proof of Theorem 6

Proof. The proof is combination of the proofs of Theorem 3 and Theorem 4. To avoid verbatim repetition, in this proof, we show only the existence of the constants C 0 and C 1 without showing the exact expressions. We define

<!-- formula-not-decoded -->

We can then rewrite the update of ¯ r and w in Algorithm 5 as

<!-- formula-not-decoded -->

In the rest of this proof, we write µ θ and π θ as shorthand for µ θ w and π θ w . We define

<!-- formula-not-decoded -->

We proceed to verifying Assumptions 3.2, 3.3, &amp; 3.4 to invoke Theorem 1.

Assumption 3.2 is verified in Lemma 6.

For Assumption 3.4, Lemma 12 suggests that assuming ∥ X ∥ ≤ 1 , L µ ≤ 1 , then

<!-- formula-not-decoded -->

is a Lipschitz constant of u ∗ ( θ ) . Take any ξ ∈ (0 , 1) , it is easy to see there exists positive constants C 2 and C 3 such that

<!-- formula-not-decoded -->

Assumption 3.4, therefore, holds.

We now select proper R B 1 and R B 2 to fulfill Assumption 3.3. Using (38), it is easy to see

<!-- formula-not-decoded -->

for some positive constant C 4 . For sufficiently large R B 1 , e.g.,

<!-- formula-not-decoded -->

we have sup θ ∥ u ∗ ( θ ) ∥ ≤ R B 1 -ξ . Selecting R B 2 ∈ ( R B 1 -ξ, R B 1 ) then fulfills Assumption 3.3.

With Assumptions 3.1 - 3.4 satisfied, Theorem 1 then implies that there exists a unique θ ∞ such that

<!-- formula-not-decoded -->

We now show what θ ∞ is. We define

<!-- formula-not-decoded -->

Similar to the proof of Theorem 4 in Section A.5, we can use the Brouwer fixed point theorem to find a u ∗ η ∈ Γ B 1 such that f ( u ∗ η ) = u ∗ η if

<!-- formula-not-decoded -->

for some constant C 5 . Then it is easy to see u ∗ η is also the fixed point of u ∗ ( · ) , implying θ ∞ = u ∗ η . Rearranging terms of u ∗ η = f ( u ∗ η ) yields

<!-- formula-not-decoded -->

and the desired constants C 0 and C 1 can be deduced from (38), (39), &amp; (40).

## B. Convergence of Main Networks

This section is a collection of several convergence proofs of main networks. We first state a general convergence result regarding the convergence of time varying linear systems from Konda (2002), which will be repeatedly used.

Consider a stochastic process { Y t } taking values in a finite space Y . Let { P θ ∈ R |Y|×|Y| | θ ∈ R K } be a parameterized family of transition kernels on Y . We update the parameter w ∈ R K recursively as

<!-- formula-not-decoded -->

where h θ : Y → R K and G θ : Y → R K × K are two vector- and matrix-valued functions.

Assumption B.2. The learning rates { α t } is positive deterministic nonincreasing and satisfies

Assumption B.1. Pr( Y t +1 | Y 0 , w 0 , θ 0 , · · · , Y t , w t , θ t ) = P θ t ( Y t , Y t +1 ) .

<!-- formula-not-decoded -->

Assumption B.3. The random sequence { θ t } satisfies

<!-- formula-not-decoded -->

where C &gt; 0 is a constant, { β t } is a deterministic sequence satisfying

<!-- formula-not-decoded -->

for some d &gt; 0 .

Assumption B.4. For each θ , there exists ¯ h ( θ ) ∈ R K , ¯ G ( θ ) ∈ R K × K , ˆ h θ : Y → R K , ˆ G θ : Y → R K × K such that for each y ∈ Y ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption B.5. sup θ ∥ ∥ ¯ h ( θ ) ∥ ∥ &lt; ∞ , sup θ ∥ ∥ ¯ G ( θ ) ∥ ∥ &lt; ∞ .

Assumption B.6. For any f θ ∈ { h θ , ˆ h θ , G θ , ˆ G θ } , sup θ,y ∥ f θ ( y ) ∥ &lt; ∞ .

Assumption B.7. There exists some constant C &gt; 0 such that

<!-- formula-not-decoded -->

Assumption B.8. There exists some constant C &gt; 0 such that for any f θ ∈ { h θ , ˆ h θ , G θ , ˆ G θ } and y ∈ Y ,

<!-- formula-not-decoded -->

Assumption B.9. There exists a constant ζ &gt; 0 such that for all w,θ ,

<!-- formula-not-decoded -->

Theorem 8. (Theorem 3.2 in Konda (2002)) Under Assumptions B.1- B.9, almost surely,

<!-- formula-not-decoded -->

We start with the convergence of the main network in Q -evaluation with a Target Network (Algorithm 1).

Lemma 2. Almost surely, lim t →∞ ∥ w t -w ∗ ( θ t ) ∥ = 0 .

Proof. This proof is a special case of the proof of Lemma 4. We, therefore, omit it to avoid verbatim repetition.

We then show the convergence of the main network in Differential Q -evaluation with a Target Network (Algorithm 2).

Lemma 3. Almost surely, lim t →∞ ∥ u t -u ∗ ( θ t ) ∥ = 0 .

Proof. This proof is the same as the proof of Lemma 2 up to change of notations. We, therefore, omit it to avoid verbatim repetition.

We now proceed to show the convergence of the main network in Q -learning with a Target Network (Algorithm 3). In Lemma 4 and its proof, we continue using the notations in Theorem 4 and its proof (Section A.5).

Lemma 4. Almost surely, lim t →∞ ∥ w t -w ∗ ( θ t ) ∥ = 0 .

Proof. We proceed by verifying Assumptions B.1- B.9. Let Y t . = ( S t , A t , S t +1 ) and

<!-- formula-not-decoded -->

Let y 1 = ( s 1 , a 1 , s ′ 1 ) , y 2 = ( s 2 , a 2 , s ′ 2 ) , we define P θ ∈ R |Y|×|Y| as

Recall in Section A.5, we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to Algorithm 3, we have

<!-- formula-not-decoded -->

Assumption B.1 therefore holds. Assumptions B.2 &amp; B.3 are satisfied automatically by Assumptions 3.1, 3.5, and Lemma 1.

Consider a Markov Reward Process (MRP) in Y with transition kernel P θ and reward function h θ ( y ) i , the i -th element of h θ ( y ) (we need an MRP for each i ). The stationary distribution of this MRP is d Y ,θ ( s, a, s ′ ) = d µ θ ( s, a ) p ( s ′ | s, a ) . The reward rate of this MRP is ∑ y d Y ,θ ( y ) h θ ( y ) i = ¯ h ( θ ) i , the i -the element of ¯ h ( θ ) . We define

<!-- formula-not-decoded -->

By definition, ˆ h θ ( y ) i is the differential value function of this MRP. The differential Bellman equation (1) then implies the first equation in Assumption B.4 holds. Similarly, we define

<!-- formula-not-decoded -->

the second equation in Assumption B.4 holds as well.

Assumption B.5 follows directly from the definition of ¯ h ( θ ) and ¯ G ( θ ) . The boundedness of h θ ( y ) and G θ ( y ) in Assumption B.6 follows directly from their definitions. The differential value function of the MRP can be equivalently expressed as (see, e.g, Eq (8.2.2) in Puterman (2014))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h θ,i denotes the vector in R |Y| whose y -th element is h θ ( y ) i and ˆ h θ,i , ˆ G θ,ij , G θ,ij are similarly defined as h θ,i . Note the existence of the matrix inversion results directly from the ergodicity of the chain (see, e.g., Section A.5 in the appendix of Puterman (2014)). To show the boundedness of ˆ h θ ( y ) and ˆ G θ ( y ) required in Assumption B.6, it suffices to show the boundedness of ( I -P θ + 1 d ⊤ Y ,θ ) -1 ( I -1 d ⊤ Y ,θ ) . The intuition is simple: it is a continuous function in a compact set P (defined in Assumption 5.1), so it obtains its maximum. We now formalize this intuition. We first define a function f : P → R |Y|×|Y| that maps a transition matrix in S × A to a transition matrix in Y . For P ∈ P , y 1 ∈ Y , y 2 ∈ Y , we have f ( P )( y 1 , y 2 ) . = I s ′ 1 = s 2 P (( s 1 , a 1 ) , ( s 2 , a 2 )) p ( s ′ 2 | s 2 , a 2 ) /p ( s ′ 1 | s 1 , a 1 ) . Let P Y . = { f ( P ) | P ∈ P} . As f is continuous and P is compact, it is easy to see P Y is also compact. It is easy to verify that f ( P ) is a valid transition kernel in Y . As Assumption 5.1 asserts P is ergodic, we use d P to denote the stationary distribution of the chain induced by P . Let d f ( P ) ( s, a, s ′ ) . = d P ( s, a ) p ( s ′ | s, a ) , it is easy to verify that d f ( P ) is the stationary distribution of the chain induced by f ( P ) , from which it is easy to see the chain in Y induced by f ( P ) is also ergodic, i.e., any P ′ ∈ P Y is ergodic. Consider the function g ( P ′ ) . = ( I -P ′ + 1 d ⊤ P ′ ) -1 ( I -1 d ⊤ P ′ ) . The ergodicity of P ′ ensures g ( P ′ ) is well defined. As P Y is compact and g is continuous, g attains its maximum in P Y , say, e.g., U g . For any θ ∈ R K , it is easy to verify that P θ = f ( P µ θ ) ∈ P Y , where P θ is defined in (41). Consequently, ∥ g ( P θ ) ∥ ≤ U g . The boundedness of ˆ h θ ( y ) and ˆ G θ ( y ) then follows directly from the boundedness of h θ ( y ) and G θ ( y ) . Assumption B.6 therefore holds.

The Lipschitz continuity of h θ ( y ) , ¯ h ( θ ) , G θ ( y ) , ¯ G ( θ ) in θ follows directly from the Lipschitz continuity of D µ θ (Lemma 9), the Lipschitz continuity of P π ¯ θ X ¯ θ (the proof of Lemma 10), and Lemma 7. We can easily show the boundedness of ( I -P θ + 1 d ⊤ Y ,θ ) -1 similar to the boundedness of ( I -P θ + 1 d ⊤ Y ,θ ) -1 ( I -1 d ⊤ Y ,θ ) . The Lipschitz continuity of ˆ h θ ( y ) and ˆ G θ ( y ) in θ is then an exercise of Lemmas 7 &amp; 8. Assumptions B.7 &amp; B.8 are then satisfied.

Assumptions B.9 follows directly from the positive definiteness of ¯ G ( θ ) .

With Assumptions B.1- B.9 satisfied, invoking Theorem 8 completes the proof.

We now show the convergence of the main network in Gradient Q -learning with a Target Network (Algorithm 4). In Lemma 5 and its proof, we continue using the notations in Theorem 5 and its proof (Section A.6).

Lemma 5. Almost surely, lim t →∞ ∥ w t -w ∗ ( θ t ) ∥ = 0 .

Proof. We proceed by verifying Assumptions B.1-B.9. Assumptions B.1-B.8 can be verified in the same way as the proof of Lemma 4. We, therefore, omit it to avoid verbatim repetition. For Assumption B.9, consider κ . = [ u ⊤ , w ⊤ ] ⊤ , we have

<!-- formula-not-decoded -->

where C µ θ = X ⊤ D µ θ X . We first show that there exists a constant C 1 &gt; 0 such that u ⊤ C µ θ u ≥ C 1 ∥ u ∥ 2 holds for any u and θ , which is equivalent to showing for some C 1 &gt; 0 ,

<!-- formula-not-decoded -->

holds for any u ∈ U . = { u | ∥ u ∥ = 1 } and θ ∈ R K . Consider P defined in 5.1. For any P ∈ P , we use D P to denote a diagonal matrix whose diagonal term is the stationary distribution of the chain induced by P . Assumptions 2.2 &amp; 5.1 then ensure that X ⊤ D P X is positive definite, i.e.,

<!-- formula-not-decoded -->

holds for any u ∈ U and P ∈ P . As u ⊤ X ⊤ D P Xu is a continuous function in both u and P , it obtains its minimum in the compact set U × P , say, e.g., C 1 &gt; 0 , i.e.

<!-- formula-not-decoded -->

holds for any u ∈ U and P ∈ P . It is then easy to see

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption B.9, therefore, holds. Theorem 8 then implies that

<!-- formula-not-decoded -->

holds for any u ∈ R K and θ ∈ R K , implying which completes the proof.

We now show the convergence of the main network in Differential Q -learning with a Target Network (Algorithm 5).

Lemma 6. Almost surely, lim t →∞ ∥ u t -u ∗ ( θ t ) ∥ = 0 .

Proof. The proof is the same as the proof of Lemma 4 up to change of notations. We, therefore, omit it to avoid verbatim repetition.

## C. Technical Lemmas

Lemma 7. Let f 1 ( x ) and f 2 ( x ) be two Lipschitz continuous functions with Lipschitz constants C 1 and C 2 . If they are also bounded by U 1 and U 2 , then their product f 1 ( x ) f 2 ( x ) is also Lipschitz continuous with C 1 U 2 + C 2 U 1 being a Lipschitz constant.

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Lemma 9. Under Assumption 5.1 &amp; 5.2, D µ θ is Lipschitz continuous in θ .

Proof. By the definition of stationary distribution, we have ensures that for any θ , L ( P µ ) has full column rank. So

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is easy to see L ( P µ θ ) is Lipschitz continuous in θ and sup θ ∥ L ( P µ θ ) ∥ &lt; ∞ . We now show ( L ( P µ θ ) ⊤ L ( P µ θ ) ) -1 is Lipschitz continuous in θ and is uniformly bounded for all θ , invoking Lemma 7 with which will then complete the proof. Note

<!-- formula-not-decoded -->

where adj ( · ) denotes the adjugate matrix. By the properties of determinants and adjugate matrices, it is easy to see both the numerator and the denominator are Lipschitz continuous in θ and are uniformly bounded for all θ . It thus remains to show the denominator is bounded away from 0 . Consider the compact set P defined in Assumption 5.1, for any P ∈ P , Assumption 5.1 ensures L ( P ) ⊤ L ( P ) is invertible, i.e., | det ( L ( P ) ⊤ L ( P ) ) | &gt; 0 . As | det ( L ( P ) ⊤ L ( P ) ) | is continuous in P , so it obtains its minimum in the compact set P , say C 0 &gt; 0 . It then follows inf θ | det ( L ( P µ θ ) ⊤ L ( P µ θ ) ) | ≥ C 0 &gt; 0 , which completes the proof.

Lemma 10. The w ∗ ( θ ) defined in (29) is Lipschitz continuous in θ .

Proof. Recall

<!-- formula-not-decoded -->

We first show P π θ Xθ is Lipschitz continuous in θ . By definition of π θ ,

<!-- formula-not-decoded -->

Let X s ′ ∈ R |A|× K be a matrix whose a ′ -th row is x ( s ′ , a ′ ) ⊤ . Then

<!-- formula-not-decoded -->

The equivalence between norms then asserts that there exists a constant L 0 &gt; 0 such that

<!-- formula-not-decoded -->

It is easy to see L 0 ∥ X ∥ is also a Lipschitz constant of P π ¯ θ X ¯ θ by the property of projection.

Lemma 9 ensures that D µ θ is Lipschitz continuous in θ and we use L D to denote a Lipschitz constant. We remark that if we assume ∥ X ∥ ≤ 1 , we can indeed select an L D that is independent of X . To see this, let L µ be the Lipschitz constant in Assumption 5.2, then we have

<!-- formula-not-decoded -->

Due to the equivalence between norms, there exists a constant L ′ µ &gt; 0 such that

<!-- formula-not-decoded -->

We can now use Lemmas 7 &amp; 8 to compute the bounds and Lipschitz constants for several terms of interest, which are detailed in Table 1. From Table 1 and Lemma 7, a Lipschitz constant of w ∗ ( θ ) is

<!-- formula-not-decoded -->

which completes the proof.

|                                      | Bound            | Lipschitz constant                             |
|--------------------------------------|------------------|------------------------------------------------|
| D µ θ                                | 1                | L D                                            |
| ( X ⊤ D µ θ X + ηI ) - 1             | η - 1            | η - 2 ∥ X ∥ 2 L D                              |
| X ⊤ D µ θ r                          | ∥ X ∥∥ r ∥       | ∥ X ∥∥ r ∥ L D                                 |
| ( X ⊤ D µ θ X + ηI ) - 1 X ⊤ D µ θ r | η - 1 ∥ X ∥∥ r ∥ | η - 1 ∥ X ∥∥ r ∥ L D + η - 2 ∥ X ∥ 3 ∥ r ∥ L D |
| ( X ⊤ D µ θ X + ηI ) - 1 X ⊤ D µ θ   | η - 1 ∥ X ∥      | η - 1 ∥ X ∥ L D + η - 2 ∥ X ∥ 3 L D            |
| γP π ¯ θ X ⊤ ¯ θ                     | γU P X R B 1     | γL 0 X                                         |

∥

∥

Table 1. U P . = sup θ ∥ P π θ ∥ .

Lemma 11. Under Assumptions 2.2, 5.1 - 5.3, if ∥ X ∥ ≤ 1 , then w ∗ ( θ ) defined in (35) satisfies

<!-- formula-not-decoded -->

where L w is a positive constant that depends on X through only X ∥ X ∥ .

Proof. We first recall that if ∥ X ∥ ≤ 1 , the Lipschitz constants for both µ θ and π θ in θ can be selected to be independent of X (c.f. (42)). Recall

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now show w ∗ ( θ ) / ∥ X ∥ is Lipschitz continuous in θ by invoking Lemmas 7 &amp; 8. Let D P ∈ R |S||A|×|S||A| be a diagonal matrix whose diagonal entry is the stationary distribution of the chain induced by P . For any P ∈ P , Assumption 5.1

Let then

∥

∥

ensures that D P is positive definite. Consequently, ∥ ∥ ∥ ( ˜ X ⊤ D P ˜ X ) -1 ∥ ∥ ∥ is well defined and is continuous in P , implying it obtains its maximum in the compact set P , say U g . So ∥ ∥ ∥ ˜ C -1 µ θ ∥ ∥ ∥ ≤ U g and importantly, U g depends on X through only X ∥ X ∥ . Using Lemma 7, it is easy to see the bound and the Lipschitz constant of ˜ A ⊤ π θ ,µ θ ˜ C -1 µ θ ˜ A π θ ,µ θ depend on X through only X ∥ X ∥ . It is easy to see ∥ ∥ ∥ ( ηI + ∥ X ∥ 2 ˜ A ⊤ π θ ,µ θ ˜ C -1 µ θ ˜ A π θ ,µ θ ) -1 ∥ ∥ ∥ ≤ 1 /η . If we further assuming ∥ X ∥ ≤ 1 , Lemma 8 then implies that ( ηI + ∥ X ∥ 2 ˜ A ⊤ π θ ,µ θ ˜ C -1 µ θ ˜ A π θ ,µ θ ) -1 has a Lipschitz constant that depends on X through only X ∥ X ∥ . It is then easy to see there exists a constant L w &gt; 0 , which depends on X only through X ∥ X ∥ , such that

<!-- formula-not-decoded -->

which completes the proof.

Lemma 12. The u ∗ ( θ ) defined in (37) is Lipschitz continuous in θ .

Proof. Recall

<!-- formula-not-decoded -->

We now use Lemmas 7 &amp; 8 to compute the bounds and Lipschitz constants for several terms of interest, which are detailed in Table 2. From Table 2 and Lemma 7, a Lipschitz constant of u ∗ ( θ ) is

<!-- formula-not-decoded -->

which completes the proof.

|               | Bound              | Lipschitz constant              |
|---------------|--------------------|---------------------------------|
| D µ θ         | 1                  | L D                             |
| ¯ G ( θ ) - 1 | max { 1 , η - 1 }  | max { 1 , η - 2 } O ( ∥ X ∥ 2 ) |
| ¯ h 1 ( θ )   | O ( ∥ X ∥ )+ O (1) | O ( L µ )                       |
| ¯ H 2 ( θ )   | O ( ∥ X ∥ )        | O ( ∥ X ∥ )                     |
| ¯ h ( θ )     | ( X )+ (1)         | ( X )+ ( L µ )                  |

O

∥

∥

O

O

∥

∥

O

Table 2. Bounds and Lipschitz constants of several terms, assuming ∥ X ∥ ≤ 1 , L µ ≤ 1 .

## D. Experiment Details

## D.1. Kolter's Example

Kolter's example is a simple two-state Markov chain with P π . = [ 0 . 5 0 . 5 0 . 5 0 . 5 ] . The reward is set in a way such that the state-value function is v π = [ 1 1 . 05 ] . The feature matrix is X . = [ 1 1 . 05 + ϵ ] . Kolter (2011) shows that for any ϵ &gt; 0 , C &gt; 0 , there exists a D µ = diag ( [ d µ ( s 1 ) d µ ( s 2 ) ] ) such that

<!-- formula-not-decoded -->

where w ∗ 0 is the TD fixed point. This suggests that as long as there is representation error (i.e., ϵ &gt; 0 ), the performance of the TD fixed point can be arbitrarily poor. In our experiments, we set ϵ = 0 . 01 and find when d µ ( s 1 ) approaches around 0.71, ∥ Xw ∗ 0 -v π ∥ becomes unbounded.

## D.2. Baird's Example

y = 0.99

Figure 2. The Baird's example used in Chapter 11.2 of Sutton &amp; Barto (2018). The figure is taken from Zhang et al. (2020). At each state, there are two actions. The solid action always leads to s 7 ; the dashed action leads to one of { s 1 , . . . , s 6 } with equal probability. The discount factor γ is 0.99.

<!-- image -->

Figure 2 shows Baird's example. In the policy evaluation setting (corresponding to Figure 1b), we use the same state features as Sutton &amp; Barto (2018), i.e.,

<!-- formula-not-decoded -->

The weight w is initialized as [1 , 1 , 1 , 1 , 1 , 1 , 10 , 1] ⊤ as suggested by Sutton &amp; Barto (2018). The behavior policy is µ 0 ( dashed | s i ) = 6 / 7 and µ 0 ( solid | s i ) = 1 / 7 for i = 1 , . . . , 7 . And the target policy always selects the solid action. The standard Off-Policy Linear TD (with ridge regularization) updates w t as

<!-- formula-not-decoded -->

where we use α = 0 . 01 as used by Sutton &amp; Barto (2018). Note as long as α &gt; 0 , it diverges. Here we overload x to denote the state feature and x t . = x ( S t ) . We use a TD version of Algorithm 1 to estimate v π , which updates w t as

<!-- formula-not-decoded -->

where we set α = β = 0 . 01 , θ 0 = w 0 .

For the control setting (corresponding to Figures 1c &amp; 1d), we construct the state-action feature in the same way as the Errata of Baird (1995), i.e.,

<!-- formula-not-decoded -->

The first 7 rows of X are the features of the solid action; the second 7 rows are the dashed action. The weight w is initialized as [1 , 1 , 1 , 1 , 1 , 1 , 10 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1] ⊤ . The standard linear Q -learning (with ridge regularization) updates w t as

<!-- formula-not-decoded -->

where we set α = 0 . 01 . The variant of Algorithm 3 we use in this experiment updates w t as

<!-- formula-not-decoded -->

where we set α = 0 . 01 , β = 0 . 001 , θ 0 = w 0 . The behavior policy for Figure 1c is still µ 0 ; the behavior policy for Figure 1d is 0 . 9 µ 0 +0 . 1 µ w , where µ w is a softmax policy w.r.t. x ( s, · ) ⊤ w . For our algorithm with target network, the softmax policy is computed using the target network as shown in Algorithm 3.