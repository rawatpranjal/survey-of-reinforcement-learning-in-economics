## Non-Asymptotic Gap-Dependent Regret Bounds for Tabular MDPs

Max Simchowitz UC Berkeley msimchow@berkeley.edu

Kevin Jamieson University of Washington jamieson@cs.washington.edu

October 30, 2019

## Abstract

This paper establishes that optimistic algorithms attain gap-dependent and non-asymptotic logarithmic regret for episodic MDPs. In contrast to prior work, our bounds do not suffer a dependence on diameter-like quantities or ergodicity, and smoothly interpolate between the gap dependent logarithmic-regret, and the ˜ O ( √ HSAT )-minimax rate. The key technique in our analysis is a novel 'clipped' regret decomposition which applies to a broad family of recent optimistic algorithms for episodic MDPs.

## 1 Introduction

Reinforcement learning (RL) is a powerful paradigm for modeling a learning agent's interactions with an unknown environment, in an attempt to accumulate as much reward as possible. Because of its flexibility, RL can encode such a vast array of different problem settings - many of which are entirely intractable. Therefore, it is crucial to understand what conditions enable an RL agent to effectively learn about its environment, and to account for the success of RL methods in practice.

In this paper, we consider tabular Markov decision processes (MDPs), a canonical RL setting where the agent seeks to learn a policy mapping discrete states x ∈ S to one of finitely many actions a ∈ A , in an attempt to maximize cumulative reward over an episode horizon H . We shall study the regret setting, where the learner plays a policy π k for a sequence of episodes k = 1 , . . . , K , and suffers a regret proportional to the average sub-optimality of the policies π 1 , . . . , π K .

In recent years, the vast majority of literature has focused on obtaining minimax regret bounds that match the worst-case dependence on the number states |S| , actions |A| , and horizon length H ; namely, a cumulative regret of √ H |S||A| T , where T = KH denotes the total number of rounds of the game [Azar et al., 2017]. While these bounds are succinct and easy to interpret, they paint an overly pessimistic account of the complexity of these problems, and do not elucidate the favorable structural properties of which a learning agent can hope to take advantage.

The earlier literature, on the other hand, establishes a considerable more favorable regret of the form C log T , where C is an instance-dependent constant given in terms of the sub-optimality gaps associated with each action at a given state, defined as

<!-- formula-not-decoded -->

where V π glyph[star] and Q π glyph[star] denote the value and Q functions for an optimal policy π glyph[star] , and the subscript-∞ denotes these bounds hold for a non-episodic, infinite horizon setting. Depending on the constant C , the regret C log T can yield a major improvement over the √ T minimax scaling. Unfortunately, these analyses are asymptotic in nature, and only take effect after a large number of rounds,

depending on other potentially-large, highly-conservative, or difficult-to-verify problem-dependent quantities such as hitting times or measures of uniform ergodicity Jaksch et al. [2010], Tewari and Bartlett [2008], Ok et al. [2018].

To fully account for the empirical performance of RL algorithms, we seek regret bounds which take advantage of favorable problem instances, but apply in finite time and for practically realistic numbers of rounds T .

## 1.1 Contributions

As a first step in this direction, Zanette and Brunskill [2019] introduced a novel algorithm called EULER , which enjoys reduced dependence on the episode horizon H for favorable instances, while maintaining the same worst-case dependence for other parameters in their analysis as in Azar et al. [2017].

In this paper, we take the next step by demonstrating that a common class of algorithms for solving MDPs, based on the optimism principle, attains gap-dependent, problem-specific bounds similar to those previously found only in the asymptotic regime. For concreteness, we specialize our analysis to a minor modification of the EULER algorithm we call StrongEuler ; as we explain in Section 3, our analysis extends more broadly to other optimistic algorithms as well. We show that

- For any episodic MDP M , StrongEuler enjoys a high probability regret bound of C M log(1 /δ ) for all rounds T ≥ 1, where the constant C M depends on the sub-optimality gaps between actions at different states, as well as the horizon length, and contains an additive almost-gapindependent term that scales as AS 2 poly( H ) (Corollary 2.1).

Unlike previous gap-dependent regret bounds,

- The constant C M does not suffer worst-case dependencies on other problem dependent quantities such as mixing times, hitting times or measures of ergodicity. However, the constant C M does take advantage of benign problem instances (Definition 2.2).
- The regret bound of C M log(1 /δ ) is valid for any total number of rounds T ≥ 1. Selecting δ = 1 /T , this implies a non-asymptotic expected regret bound of C M log T 1 .
- The regret of StrongEuler interpolates between instance-dependent regret C M log T and minimax regret ˜ O ( √ H |S||A| T ), the latter of which may be sharper for smaller T (Theorem 2.4). Following Zanette and Brunskill [2019], this dependence on H may also be refined for benign instances.

Lastly, while the StrongEuler algorithm affords sharper regret bounds than past algorithms, our analysis techniques extend more generally to other optimism based algorithms:

- We introduce a novel 'clipped' regret decomposition (Proposition 3.1) which applies to a broad family of optimistic algorithms, including the algorithms analyzed in [Zanette and Brunskill, 2019, Dann et al., 2018, 2017, Jin et al., 2018, Azar et al., 2017].
- Following our analysis of StrongEuler , the clipped regret decomposition can establish analogous gap-dependent log T -regret bounds for many of the algorithms mentioned above.

1 By this, we mean that for any fixed T ≥ 1, one can attain C M log T regret. Extending the bound to anytime regret is left to future work

What is C M ? In many settings, we show that C M is dominated by an analogue to the sum over the reciprocals of the gaps defined in (1). This is known to be optimal for non-dynamic MDP settings like contextual bandits, and we prove a lower bound (Proposition 2.2) which shows that this is unimprovable for general MDPs as well. Furthermore, building on Zanette and Brunskill [2019], we show this adapts to problems with additional structure, yielding, for example, a horizon H -free bound for contextual bandits.

However, our gap-dependent bound also suffers from a certain dependence on the smallest nonzero gap gap min (see Definition 2.1), which may dominate in some settings. We prove a lower bound (Theorem 2.3) which shows that optimistic algorithms in the recent literature - including StrongEuler - necessarily suffer a similar term in their regret. We believe this insight will motivate new algorithms for which this dependence can be removed, leading to new design principles and actionable insights for practitioners. Finally, our regret bound incurs an (almost) gap-independent burn-in term, which is standard for optimistic algorithms, and which we believe is an exciting direction of research to remove.

Altogether, we believe that the results in our paper serve as a preliminary but significant step to attaining sharp, instance-dependent, and non-asymptotic bounds for tabular MDPs, and hope that our analysis will guide the design of future algorithms that attain these bounds.

## 1.2 Related Work

Like the multi-armed bandit setting, regret bounds for MDP algorithms have been characterized both in gap-independent forms that rely solely on S := |S| , A := |A| , H, T , and in gap-dependent forms which take into account the gaps (1), as well as other instance-specific properties of the rewards and transition probabilities.

Finite Sample Bounds, Gap-Independent Bounds: A number of notable recent works give undiscounted regret bounds for finite-horizon, tabular MDPs, nearly all of them relying on the principle of optimism which we describe in Section 3 [Dann and Brunskill, 2015, Azar et al., 2017, Dann et al., 2017, Jin et al., 2018, Zanette and Brunskill, 2019]. Many of the more recent works Azar et al. [2017], Zanette and Brunskill [2019], Dann et al. [2018] attain a regret of √ HSAT , matching the known lower bound of √ HSAT established in Osband and Van Roy [2016], Jaksch et al. [2010], Dann and Brunskill [2015]. As mentioned above, the EULER algorithm of Zanette and Brunskill [2019] attains the minimax rates and simultaneously enjoys a reduced dependence on H in benign problem instances, such as the contextual bandits setting where the transition probabilities do not depend on the current state or learners actions, or when the total cumulative rewards over any roll-out are bounded by 1 in magnitude.

Diameter Dependent Bounds: In the setting of infinite horizon MDPs with discounted regret, many previous works have established logarithmic regret bounds of the form C ( M ) log T , where C ( M ) is a constant depending on the underlying MDP. Notably, Jaksch et al. [2010] give an algorithm which attains a ˜ O ( √ D 2 S 2 AT ) gap-independent regret, and an ˜ O ( D 2 S 2 A gap ∗ log( T )) gapdependent regret bound, where gap ∗ is the difference between the mean infinite-horizon reward of π ∗ and the next-best stationary policy, and where D denotes the maximum expected traversal time between any two states x, x ′ , under the policy which attains the minimal traversal time between those two states. We note that if gap ∞ ( x, a ) denotes the sub-optimality of any action a at state x as in (1), then gap ∗ ≤ min x,a gap ∞ ( x, a ). The bounds in this work, on the other hand, depend on an average over inverse gaps, rather than a worst case. Moreover, the diameter D can be quite large when there exist difficult-to-access states. We stress that the bound due to Jaksch et al. [2010] is

non-asympotic, but the bound in terms of gap ∗ dependences other worst-case quantities measuring ergodicity.

Asymptotic Bounds: Prior to Jaksch et al. [2010], and building on the bounds of Burnetas and Katehakis [1997], Tewari and Bartlett [2008] presented bounds in terms of a diameterrelated quantity ¯ D ≥ D , which captures the minimal hitting time between states when restricted to optimal policies. Tewari and Bartlett [2008] prove that their algorithm enjoys a regret 2 of ∑ ( s,a ) ∈ CRIT ¯ D 2 gap ∞ ( x,a ) log( T ) asymptotically in T where CRIT contains those sub-optimal state-action pairs ( x, a ) such that a can be made to the the unique, optimal action at x by replacing p ( s ′ | s, a ) with some other vector on the S -simplex. Recently, Ok et al. [2018] present per-instance lower bounds for both structured and unstructured MDPs, which apply to any algorithm which enjoys sub-linear regret on any problem instance, and an algorithm which matches these bounds asymptotically. This bound replaces ¯ D 2 with ¯ H 2 , where ¯ H denotes the range of the bias functions, an analogue of H for the non-episodic setting Bartlett and Tewari [2009]. We further stress that whereas the logarithmic regret bounds of Jaksch et al. [2010] hold for finite time with polynomial dependence on the problem parameters, the number of episodes needed for the bounds of Burnetas and Katehakis [1997], Tewari and Bartlett [2008], Ok et al. [2018] to hold may be exponentially large, and depend on additional, pessimistic problem-dependent quantities (e.g. a uniform hitting time in Tewari [2007, Proposition 29]).

Novelty of this work: The major contribution of our work is showing problem-dependent log( T ) regret bounds which i) attain a refined dependence on the gaps, as in Tewari and Bartlett [2008], ii) apply in finite time after a burn-in time only polynomial in S , A , H and the gaps, iii) depend only on H and not on the diameter D (and thus, are not adversely affected by difficult to access states), and iv) smoothly interpolate between log T regret and the minimax √ HSAT rate attained by Azar et al. [2017] et seq.

## 1.3 Problem Setting, Notation, and Organization

Episodic MDP: A stationary , episodic MDP is a tuple M := ( S , A , H, r, p, p 0 , R ), where for each x ∈ S , a ∈ A we have that R ( x, a ) ∈ [0 , 1] is a random reward with expectation r ( x, a ), p : S × A → ∆ S denotes transition probabilities, p 0 ∈ ∆ S is an initial distribution over states, and H is the horizon, or length of the episode. A policy π is a sequence of mappings π h : S → A . For our given MDP M , we let E π and P π denote the expectation and probability operator with respect to the law of sequence ( x 1 , a 1 ) , . . . , ( x H , a H ), where x 1 ∼ p 0 , a h = π h ( x h ), x h +1 ∼ p ( x h , a h ). We define the value of π as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2 Tewari and Bartlett [2008] actually presents a bound of the form ¯ D 2 SA min ( s,a ) ∈ CRIT gap ∞ ( x,a ) log( T ) but it is straightforward to extract the claimed form from the proof.

and for h ∈ [ H ] and x ∈ S ,

which we identify with a vector in R S . We define the associated Q-function Q π : S × A → R ,

<!-- formula-not-decoded -->

so that Q π h ( x, π h ( x )) = V π h ( x ). We denote the set of optimal policies

<!-- formula-not-decoded -->

and let π glyph[star] h ( x ) := { a : π h ( x ) = a, π ∈ π glyph[star] } denote the set of optimal actions. Lastly, given any optimal π ∈ π glyph[star] , we introduce the shorthand V glyph[star] h = V π h and Q glyph[star] h = Q π h , where we note that even when π is not unique, V glyph[star] h and Q glyph[star] h do not depend on the choice of optimal policy.

Episodic Regret: We consider a game that proceeds in rounds k = 1 , . . . , K , where at each state an algorithm Alg selects a policy π k , and observes a roll out ( x 1 , a 1 ) , . . . , ( x H , a H ) ∼ P π k . The goal is to minimize the cumulative simple regret, defined as

<!-- formula-not-decoded -->

Notation and Organization: For n ∈ N , we define [ n ] = { 1 , . . . , n } . For two expressions f, g that are functions of any problem-dependent variables of M , we say f glyph[lessorsimilar] g ( f glyph[greaterorsimilar] g , respectively) if there exists a universal constant c &gt; 0 independent of M such that f ≤ cg ( f ≥ cg , respectively). glyph[lessorapproxeql] will denote an informal, approximate inequality. Section 2 presents our main results, and Section 3 sketches the proof and highlights the novelty of our techniques. All formal proofs, and many rigorous statement of results, are deferred to the appendix, whose organization and notation are described at length in Appendix A.

## 1.4 Optimistic Algorithms

Lastly, we introduce optimistic algorithms which select a policy which is optimal for an overestimated, or optimistic , estimate of the true Q -function, Q glyph[star] .

Definition 1.1 (Optimistic Algorithm) . We say that an algorithm Alg satisifes optimism if, for each round k ∈ [ K ] and stage h ∈ [ H ], it constructs an optimistic Q -function Q k,h ( x, a ) and policy π k = ( π k,h ) satisfying

<!-- formula-not-decoded -->

The associated optimistic value function is V k,h ( x ) := Q k,h ( x, π k,h ( x )).

We shall colloquially refer to an algorithm as optimistic if it satsifies optimism with high probability. Optimism has become the dominant approach for learning finite-horizon MDPs, and all recent low-regret algorithms are optimistic [Dann et al., 2017, 2018, Azar et al., 2017, Zanette and Brunskill, 2019, Jin et al., 2018]. In model-based algorithms, the overestimates Q k,h are constructed recursively as Q k,h ( x, a ) = ̂ r k ( x, a ) + ̂ p k ( x, a ) glyph[latticetop] V k,h +1 + b k,h ( x, a ), where ̂ r k ( x, a ) and ̂ p k ( x, a ) are empirical estimates of the mean rewards and transition probabilities, and b k,h ( x, a ) ≥ 0 is a confidence bonus designed to ensure that Q k,h ( x, a ) ≥ Q glyph[star] ( x, a ). Letting n k ( x, a ) denote the total number of times a given state-action pair is visited, a simple bonus b k,h ( x, a ) glyph[equalorsimilar] √ H log( SAHK/δ ) n k ( x,a ) suffices to induce optimism, yielding the UCBVI-CH algorithm of [Azar et al., 2017]. This leads

to an episodic regret bound of √ H 2 SAT , a factor of √ H greater than the minimax rate. More refined bonuses based on the 'Bernstein trick' achieve the optimal H -dependence [Azar et al., 2017], and the EULER algorithm of Zanette and Brunskill [2019] adopts further refinements to replace worst-case H dependence with more adaptive quantities.

The StrongEuler algorithm considered in this work applies similarly adaptive bonuses, but our analysis extends to all aforementioned bonus configurations. We remark that there are also modelfree optimistic algorithms based on Q-learning (see, e.g. Jin et al. [2018]) that construct overestimates in a slightly different fashion. While our main technical contribution, the clipped regret decomposition (Proposition 3.1), applies to all optimistic algorithms, our subsequent analysis is tailored to model-based approaches, and may not extend straightforwardly to Q-learning methods.

## 2 Main Results

Logarithmic Regret for Optimistic Algorithms: We now state regret bounds that describe the performance of StrongEuler , an instance of the model-based, optimistic algorithms described above. StrongEuler is based on carefully selected bonuses from Zanette and Brunskill [2019], and formally instantiated in Algorithm 1 in Appendix E. We emphasize that other optimistic algorithms enjoy similar regret bounds, but we restrict our analysis to StrongEuler to attain the sharpest H -dependence.The key quantities at play are the suboptimality-gaps between the Q-functions:

Definition 2.1 (Suboptimality Gaps) . For h ∈ [ H ], define the stage-dependent suboptimality gap

<!-- formula-not-decoded -->

as well as the minimal stage-independent gap gap ( x, a ) := min h gap h ( x, a ), and the minimal gap gap min := min x,a,h { gap h ( x, a ) : gap h ( x, a ) &gt; 0 } .

Note that any optimal a glyph[star] ∈ π glyph[star] h ( x ) satisfies the Bellman equation Q glyph[star] h ( x, a glyph[star] ) = max a Q glyph[star] h ( x, a )= V glyph[star] h ( x ), and thus gap h ( x, a glyph[star] ) = 0 if and only if a glyph[star] ∈ π glyph[star] h ( x ). Following Zanette and Brunskill [2019], we consider two illustrative benign problem settings under which we obtain an improved dependence on the horizon H :

Definition 2.2 (Benign Settings) . We say that an MDP M is a contextual bandit instance if p ( x ′ | x, a ) does not depend on x or a . An MDP M has G -bounded rewards if, for any policy π , ∑ H h =1 R ( x h , a h ) ≤ G holds with probability 1 over trajectories (( x h , a h )) ∼ P π .

Lastly, we define Z opt as the set of pairs ( x, a ) for which a is optimal at x for some stage h ∈ [ H ], and its complement Z sub :

<!-- formula-not-decoded -->

Note that typically |Z opt | glyph[lessorsimilar] H |S| or even |Z opt | glyph[lessorsimilar] |S| (see Remark B.2 in the appendix). We now state our first result, which gives a gap-dependent regret bound that scales as log(1 /δ ) with probability at least 1 -δ . The result is a consequence of a more general result stated as Theorem 2.4, itself a simplified version of more precise bounds stated in Appendix B.1.

Corollary 2.1. Fix δ ∈ (0 , 1 / 2) , and let A = |A| , S = |S| , M = ( SAH ) 2 . Then with probability at least 1 -δ , StrongEuler run with confidence parameter δ enjoys the following regret bound for all

K ≥ 1 :

<!-- formula-not-decoded -->

Moreover, if M is either a contextual bandits instance, or has G -bounded rewards for G glyph[lessorsimilar] 1 , then the factors of H 3 on the first line can be sharped to H . In addition, if M is a contextual bandits instance, the factor of H 3 in the first term (summing over ( x, a ) ∈ Z sub ) can be sharped to 1 .

Setting δ = 1 /T and noting that ∑ K k =1 V glyph[star] 0 -V π k 0 ≤ KH = T with probability 1 (recall R ( x, a ) ∈ [0 , 1]), we see that the expected regret E [ ∑ K k =1 V glyph[star] 0 -V π k 0 ] can be bounded by replacing 1 /δ with T in right hand side of the inequality (2); this yields an expected regret that scales as log T .

Three regret terms: The first term in Corollary 2.1 reflects the sum over sub-optimal stateaction pairs, which a lower bound (Proposition 2.2) shows is unimprovable in general. In the infinite horizon setting, Ok et al. [2018] gives an algorithm whose regret is asymptotically bounded by an analogue of this term. The third term characterizes the burn-in time suffered by nearly all modelbased finite-time analyses and is the number of rounds necessary before standard concentration of measure arguments kick in. The second term is less familiar and is addressed in Section 2.2 below.

H dependence: Comparing to known results from the infinite-horizon setting, one expects the optimal dependence of the first term on the horizon to be H 2 . However, we cannot rule out that the optimal dependence is H 3 for the following three reasons: (i) the infinite-horizon analogues D, ¯ D, ¯ H (Section 1.2) are not directly comparable to the horizon H ; (ii) in the episodic setting, we have a potentially different value function V glyph[star] h for each h ∈ [ H ], whereas the value functions of the infinite horizon setting are constant across time; (iii) the H 3 may be unavoidable for nonasymptotic (in T ) bounds, even if H 2 is the optimal asymptotic dependence after sufficient burn-in (possibly depending on diameter-like quantities). Resolving the optimal H dependence is left as future work. We also note that for contextual bandits, we incur no H dependence on the first term; and thus the first term coincides with the known asymptotically optimal (in T ), instance-specific regret [Garivier et al., 2018].

Guarantees for other optimistic algorithms: To make the exposition concrete, we only provide regret bounds for the StrongEuler algorithm. However, the 'gap-clipping' trick (Proposition 3.1) and subsequent analysis template described in Section 3.1 can be applied to obtain similar bounds for other recent optimistic algorithms, as in [Azar et al., 2017, Dann et al., 2017, Zanette and Brunskill, 2019, Dann et al., 2018]. 3

## 2.1 Sub-optimality Gap Lower Bound

Our first lower bound shows that when the total number of rounds T = KH is large, the first term of Corollary 2.1 is unavoidable in terms of regret. Specifically, for every possible choice of gaps, there exists an instance whose regret scales on the order of the first term in (2).

3 To achieve logarithmic regret, some of these algorithms require a minor modification to their confidence intervals; otherwise, the gap-dependent regret scales as log 2 T . See Appendix E for details.

Following standard convention in the literature, the lower bound is stated for algorithms which have sublinear worst case regret. Namely, we say than an algorithm Alg is α -uniformly good if, for any MDP instance M , there exists a constant C M &gt; 0 such that E M [Regret K ] ≤ C M K α for all K . 4

Proposition 2.2 (Regret Lower Bound) . Let S ≥ 2 , and A ≥ 2 , and let { ∆ x,a } x,a ∈ [ S ] × [ A ] ⊂ (0 , H/ 8) denote a set of gaps. Then, for any H ≥ 1 , there exists an MDP M with states S = [ S +2] , actions A = [ A ] , and H stages, such that,

<!-- formula-not-decoded -->

and any α -uniformly good algorithm satisfies

<!-- formula-not-decoded -->

The above proposition is proven in Appendix H, using a construction based on Dann and Brunskill [2015]. For simplicity, we stated an asymptotic lower bound. We remark that if the constant C M is poly( |S| , |A| , H ), then one can show that the above asymptotic bound holds as soon as K ≥ ( |S||A| H/ gap ∗ ) O (1 / (1 -α )) , where gap ∗ := { min gap 1 ( x, a ) : gap 1 ( x, a ) &gt; 0 } . More refined non-asymptotic regret bounds can be obtained by following Garivier et al. [2018].

## 2.2 Why the dependence on gap min ?

Without the second term, Corollary 2.1 would only suffer one factor of 1 / gap min due to the sum over state-actions pairs ( x, a ) ∈ Z sub (when the minimum is achieved by a single pair). However, as remarked above, |Z opt | typically scales like |S| and therefore the second term scales like |S| / gap min , with a dependence on 1 / gap min that is at least a factor of |S| more than we would expect. Here, we show that |S| / gap min is unavoidable for the sorts of optimistic algorithms that we typically see in the literature; a rigorous proof is deferred to Appendix G.

Theorem 2.3 (Informal Lower Bound) . Fix δ ∈ (0 , 1 / 8) . For universal constants c 1 , c 2 , c 3 , c 4 , if glyph[epsilon1] ∈ (0 , c 1 ) , and S satisfies c 2 log( glyph[epsilon1] -1 /δ ) ≤ S ≤ c 3 glyph[epsilon1] -1 / log( glyph[epsilon1] -1 /δ ) , there exists an MDP with |S| = S , |A| = 2 and horizon H = 2 , such that exactly one state has a sub-optimality gap of gap min = glyph[epsilon1] and all other states have a minimum sub-optimality gap gap h ( x, a ) ≥ 1 / 2 . For this MDP, ∑ h,x,a : gap h ( x,a ) &gt; 0 1 gap h ( x,a ) glyph[lessorsimilar] S + 1 gap min but all existing optimistic algorithms for finite-horizon MDPs which are δ -correct suffer a regret of at least S gap min log(1 /δ ) glyph[greaterorsimilar] ∑ h,x,a : gap h ( x,a ) &gt; 0 log(1 /δ ) gap h ( x,a ) + S log(1 /δ ) gap min with probability at least 1 -c 4 δ .

The particular instance described in Appendix G that witnesses this lower bound is instructive because it demonstrates a case where optimism results in over -exploration.

## 2.3 Interpolating with Minimax Regret for Small T

We remark that while the logarithmic regret in Corollary 2.1 is non-asymptotic, the expression can be loose for a number of rounds T that is small relative to the sum of the inverse gaps. Our more general result interpolates between the log T gap-dependent and √ T gap-independent regret regimes.

4 We may assume as well that Alg is allowed to take the number of episodes K as a parameter.

Theorem 2.4 (Main Regret Bound for StrongEuler ) . Fix δ ∈ (0 , 1 / 2) , and let A = |A| , S = |S| , M = ( SAH ) 2 . Futher, define for all glyph[epsilon1] &gt; 0 the set Z sub ( glyph[epsilon1] ) := { ( x, a ) ∈ Z sub : gap ( x, a ) &lt; glyph[epsilon1] } . Then with probability at least 1 -δ , StrongEuler run with confidence parameter δ enjoys the following regret bound for all K ≥ 2 :

<!-- formula-not-decoded -->

where the second inequality follows from the first with max { max glyph[epsilon1] |Z sub ( glyph[epsilon1] ) | , |Z opt |} ≤ SA . Moreover, if M is an instance of contextual bandits, then the factors of H under the square roots can be refined to a 1 , and if M has glyph[lessorsimilar] 1 -bounded rewards, then these same factors of H can be replaced by a 1 /H . In both settings, logarithmic terms can be refined as in Corollary 2.1.

By the same argument as above, Theorem 2.4 with δ = 1 /T implies an expected regret scaling like gap-dependent log T or worst-case √ HSAT . In Appendix B.1, we state a more refined bound given in terms of the reward bound G , and the maximal variance of any state-action pair (Theorem B.2).

## 3 Gap-Dependent bounds via 'clipping'

In this section, we (i) introduce the key properties of optimistic algorithms, (ii) explain existing approaches to the analysis of such algorithms, and (iii) introduce the 'clipping trick', and sketch how this technique yields gap-dependent, non-asymptotic bounds.

Definition 3.1 (Optimistic Surplus) . Given an optimistic algorithm Alg , we define the (optimistic) surplus

<!-- formula-not-decoded -->

We further say that Alg is strongly optimistic if E k,h ( x, a ) ≥ 0 for all k ≥ 1, and ( x, a, h ) ∈ S × A × [ H ], which implies that Alg is also optimistic.

While the nomenclature 'suplus' is unique to our work, surplus-like terms arise in many prior regret analyses Dann et al. [2017], Zanette and Brunskill [2019]. The notion of strong optimism is novel to this work, and facilitates a sharper H -dependence in contextual bandit setting of Definition 2.2; intuitively, strong optimism means that the Q-function Q k,h at stage h over-estimates Q glyph[star] h more than Q k,h +1 does Q glyph[star] k,h +1 .

The Regret Decomposition for Optimistic Algorithms: Under optimism alone, we can see that for any h and any a glyph[star] ∈ π glyph[star] ( x ),

<!-- formula-not-decoded -->

and therefore, we can bound the sub-optimality of π k as V glyph[star] 0 -V π k 0 ≤ V k, 0 -V π k 0 .

We can decompose the regret further by introducing the following notation: we let ω k,h ( x, a ) := P π k [( x h , a h ) = ( x, a )] denote the probability of visiting x and playing a at time h in episode k . We note that since π k ( x ) is a deterministic function, ω k,h ( x, a ) is supported on only one action a for each state x and stage h . A standard regret decomposition (see e.g. Dann et al. [2017, Lemma E.15]) then shows that for a trajectory ( x h , a h ) H h =1 ,

<!-- formula-not-decoded -->

yielding a regret bound of

<!-- formula-not-decoded -->

Existing Analysis of MDPs: We begin by sketching the flavor of minimax analyses. Introducing the notation

<!-- formula-not-decoded -->

existing analyses carefully manipulate the surpluses E k,h ( x, a ) to show that

<!-- formula-not-decoded -->

where typically C M = poly( H, log( T/δ ). Finally, they replace n k ( x, a ) with an 'idealized analogue', n k ( x, a ) := ∑ k j =1 ∑ H h =1 ω j,h ( x, a ) := ∑ k j =1 ω j ( x, a ), where we introduce ω j ( x, a ) := ∑ H h =1 ω j,h ( x, a ) denote the expected number of visits of ( x, a ) at episode j . Letting {F k } denote the filtration capturing all events up to the end episode k , we see that E [ n k ( x, a ) -n k -1 |F k -1 ] = ω k ( x, a ), and thus by standard concentration arguments (see Lemma B.7, or Dann et al. [2018, Lemma 6]), n k ( x, a ) and n k ( x, a ) are within a constant factor of each other for all k such that n k ( x, a ) is sufficiently large. Hence, by replacing n k ( x, a ) with n k ( x, a ), we have (up to lower order terms)

<!-- formula-not-decoded -->

A √ SAK poly( H ) bound is typically concluded using a careful application of Cauchy-Schwartz, and an integration-type lemma (e.g., Lemma C.1). An analysis of this flavor is used in Appendix B.4. On the other hand, one can exactly establish the identity

<!-- formula-not-decoded -->

Then one can achieve a gap dependent bound as soon as one can show that the algorithm ceases to select suboptimal actions a at ( x, h ) after sufficiently large T . Crucially, determining if action a is (sub)optimal at ( x, h ) requires precise knowledge about the value function at other states in the MDP at future stages h ′ &gt; h . This difficulty is why previous gap-dependent analyses appeal to diameter or ergodicity assumptions, which ensure sufficient uniform exploration of the MDP to reason about the value function at subsequent stages.

## 3.1 The Clipping Trick

We now introduce the 'clipping trick', a technique which merges both the minimax analysis in terms of the surpluses E k,h ( x, a ), and the gap-dependent strategy, which attempts to control how many times a given suboptimal action is selected. Core to our analysis, define the clipping operator

<!-- formula-not-decoded -->

for all x, glyph[epsilon1] &gt; 0. We can now state our first main technical result, which states that the suboptimality V glyph[star] 0 -V π k 0 can be controlled by a sum over surpluses which have been clipped to zero whenever they are sufficiently small.

Proposition 3.1. Let ˇ gap h ( x, a ) := gap min 2 H ∨ gap h ( x,a ) 4 H . Then, if π k is induced by an optimistic algorithm with surpluses E k,h ( x, a ) ,

<!-- formula-not-decoded -->

If the algorithm is strongly optimistic , and M is a contextual bandits instance, we can replace ˇ gap h ( x, a ) with ˇ gap h ( x, a ) := gap min 2 H ∨ gap h ( x,a ) 4 .

The above proposition is a consequence of a more general bound, Theorem B.3, given in Section B. Unlike gap-dependent bounds that appeal to hitting-time arguments, we do not reason about when a suboptimal action a will cease to be taken. Indeed, an algorithm may still choose a suboptimal action a even if the surplus E k,h ( x, a ) is small, because future surpluses may be large. Instead, we argue in two parts:

1. A sub-optimal action a / ∈ π glyph[star] h ( x ) is taken only if Q k,h ( x, a ) ≥ Q glyph[star] h ( x, a glyph[star] ) for some a glyph[star] ∈ π glyph[star] h ( x ), or equivalently in terms of the surplus, only if E k,h ( x, a )+ p ( x, a ) glyph[latticetop] ( V k,h +1 -V glyph[star] k,h +1 ) &gt; gap h ( x, a ). Thus if Alg selects a suboptimal action, then this is because either the current surplus E k,h ( x, a ) is larger than Ω( gap h ( x,a ) H ), or the expectation over future surpluses, captured by p ( x, a ) glyph[latticetop] ( V k,h +1 -V glyph[star] k,h +1 ) is larger than (1 - O ( 1 H ) ) gap h ( x, a ). Intuitively, the first case occurs when ( x, a ) has not been visited enough times, and the second when the future state/action pairs have not experienced sufficient visitation. In the first case, we can clip the surplus at Ω( gap h ( x,a ) H ); in the second, E k,h ( x, a ) + p ( x, a ) glyph[latticetop] ( V k,h +1 -V glyph[star] k,h +1 ) ≤ (1 + O ( 1 H ) ) p ( x, a ) glyph[latticetop] ( V k,h +1 -V glyph[star] k,h +1 ), and push the the contribution of E k,h ( x, a ) into the contribution of future surpluses. This incurs a factor of at most (1 + O ( 1 H ) ) H glyph[lessorsimilar] 1, avoiding an exponential dependence on H .
2. Clipping surpluses for pairs ( x, a ) for optimal a ∈ π glyph[star] h ( x ) requires more care. We introduce 'half-clipped' surpluses ¨ E k,h ( x, a ) := clip [ E k,h ( x, a ) | gap min 2 H ] where all actions are clipped at gap min / 2 H , and recursively define value functions ¨ V π k h ( · ) corresponding to these clipped surpluses (see Definition D.1). We then show that, for ¨ V π k 0 := E x ∼ p 0 [ ¨ V 1 ( x ) ] , we have (Lemma D.2)

<!-- formula-not-decoded -->

This argument is based on carefully analyzing when π k,h first recommends a suboptimal action π k,h ( x ) / ∈ π glyph[star] ( x ), and showing that when this occurs, V glyph[star] 0 -V π k 0 is roughly lower bounded by

gap min H times the probability of visiting a state x where π k,h ( x ) plays suboptimally. We can then subtract off gap min 2 H from all the surplus terms at the expense of at most halving the suboptimality, and using the fact E k,h -gap min 2 H ≤ clip [ E k,h | gap min 2 H ] concludes the bound. This step is crucial, because it allows us to clip the surpluses even at pairs ( x, a ) where a ∈ π glyph[star] h ( x ) is the optimal action. We note that in the formal proof of Proposition 3.1, this half-clipping precedes the clipping of suboptimal actions described above.

Unfortunately, the first step involving the half-clipping is rather coarse, and leads to S/ gap min term in the final regret bound. As argued in Theorem 2.3, this is unavoidable for existing optimistic algorithms, and suggests that Proposition 3.1 cannot be significantly improved in general.

## 3.2 Analysis of StrongEuler

Recall that StrongEuler is precisely described by Definition 1.1 up to our particular choice of confidence intervals defined (see Algorithm 1 in Appendix E). We now state a surplus bound (proved in Appendix F) that holds for these particular choice of confidence intervals, and which ensures that the strong optimism criterion of Definition 1.1 is satisfied:

Proposition 3.2 (Surplus Bound for Strong Euler (Informal)) . Let M = SAH , and define the variances Var glyph[star] h,x,a := Var[ R ( x, a )] + Var x ′ ∼ p ( x,a ) [ V glyph[star] h +1 ( x ′ )] . Then, with probability at least 1 -δ/ 2 , the following holds for all ( x, a ) ∈ S × A , h ∈ [ H ] and k ≥ 1 ,

<!-- formula-not-decoded -->

We emphasize that Proposition 3.2, and its formal analogue Proposition B.4 in Appendix B.2, are the only part of the analysis that relies upon the particular form of the StrongEuler confidence intervals; to analyze other model-based optimistic algorithms, one would simply establish an analogue of this proposition, and continue the analysis in much the same fashion. While Q-learning Jin et al. [2018] also satisfies optimism, it induces a more intricate surplus structure, which may require a different analysis.

Recalling the clipping from Proposition 3.1, we begin the gap-dependent bound with ∑ K k =1 V glyph[star] 0 -V π k 0 glyph[lessorsimilar] ∑ x,a,k,h ω k,h ( x, a ) clip [ E k,h ( x, a ) | ˇ gap h ( x, a )]. Neglecting lower order terms, Proposition 3.2 ensures that this is approximately less than ∑ x,a,k,h ω k,h ( x, a ) clip [ B lead k,h ( x, a ) | ˇ gap h ( x, a ) ] . Introduce the minimal (over h ) clipping-gaps ˇ gap ( x, a ) := min h ˇ gap ( x, a ) ≥ gap ( x,a ) ∨ gap min 4 H and maximal variances Var glyph[star] x,a := max h Var glyph[star] h,x,a . We can then render B lead k,h ( x, a ) ≤ f ( n k ( x, a )), where f ( u ) glyph[lessorsimilar] clip [√ 1 u Var glyph[star] x,a log( Mu/δ ) | ˇ gap ( x, a ) ] . Recalling the approximation n k ( x, a ) ≈ n k ( x, a ) described above, we have, to first order,

<!-- formula-not-decoded -->

where we recall the expected visitations ω k ( x, a ) := ∑ H h =1 ω k,h ( x, a ). Since n k ( x, a ) := ∑ k j =1 ω j ( x, a ), we can regard the above as an integral of the function f ( u ) (see Lemma C.1), with respect to the

density ω k ( x, a ). Evaluating this integral (Lemma B.9) yields (up to lower order terms)

<!-- formula-not-decoded -->

Finally, bounding Var glyph[star] x,a ≤ H 2 and splitting the bound into the states Z sub := { ( x, a ) : gap ( x, a ) &gt; 0 } and Z opt := { ( x, a ) : gap ( x, a ) = 0 } recovers the first two terms in Corollary 2.1. In benign instances (Definition 2.2) , we can bound Var glyph[star] h,x,a glyph[lessorsimilar] 1, improving the H -dependence. In contextual bandits, we save an addition H factor via ˇ gap h ( x, a ) glyph[greaterorsimilar] ( gap min /H ) ∨ gap ( x, a ). The interpolation with the minimax rate in Theorem 2.4 is decribed in greater detail in Appendix B.4.

## 4 Conclusion

In this paper, we proposed a new approach for providing logarithmic, gap dependent bounds for tabular MDPs in the episodic, non-generative setting. Our approach extends to any of the model-based , optimistic algorithms in the present literature. Extending these bounds to model-free approaches based on Q-learning (e.g., Jin et al. [2018]), and resolving the optimal horizon dependence are left for future work.

While we found that our models nearly matched information-theoretic lower bounds analogous, we also demonstrated that existing optimistic algorithms (both model-based and model-free) necessarily incur an additional dependence on S/ gap min for worst-case instances. It would be interesting to understand if one can circumvent this limitation by augmenting the principle of optimistism with new algorithmic ideas.

Lastly, it would be exciting to extend logarithmic bounds to settings where taking advantage of the suboptimality gaps is indispensable for attaining non-trivial regret guarantees. For example, we hope our techniques might enable sublinear regret in 'infinite arm' settings, where A is a countably infinite set, and actions a ∈ A are drawn from an (unknown) reservoir distribution. It would also be interesting to extend our tools to adaptive discretization when the state-action pairs embed into a metric space [Song and Sun, 2019], and to the function approximation settings [Jin et al., 2018].

## References

- Mohammad Gheshlaghi Azar, Ian Osband, and R´ emi Munos. Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 263-272. JMLR. org, 2017.
- Peter L Bartlett and Ambuj Tewari. Regal: A regularization based algorithm for reinforcement learning in weakly communicating mdps. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence , pages 35-42. AUAI Press, 2009.
- Apostolos N Burnetas and Michael N Katehakis. Optimal adaptive policies for markov decision processes. Mathematics of Operations Research , 22(1):222-255, 1997.
- Christoph Dann and Emma Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826, 2015.
- Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , pages 5713-5723, 2017.
- Christoph Dann, Lihong Li, Wei Wei, and Emma Brunskill. Policy certificates: Towards accountable reinforcement learning. arXiv preprint arXiv:1811.03056 , 2018.
- Aur´ elien Garivier, Pierre M´ enard, and Gilles Stoltz. Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research , 2018.
- Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600, 2010.
- Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4868-4878, 2018.
- Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. arXiv preprint arXiv:0907.3740 , 2009.
- Jungseul Ok, Alexandre Proutiere, and Damianos Tranos. Exploration in structured reinforcement learning. In Advances in Neural Information Processing Systems , pages 8888-8896, 2018.
- Ian Osband and Benjamin Van Roy. On lower bounds for regret in reinforcement learning. stat , 1050:9, 2016.
- Max Simchowitz, Kevin Jamieson, and Benjamin Recht. Best-of-k-bandits. In Conference on Learning Theory , pages 1440-1489, 2016.
- Zhao Song and Wen Sun. Efficient model-free reinforcement learning in metric spaces. arXiv preprint arXiv:1905.00475 , 2019.
- Ambuj Tewari. Reinforcement learning in large or unknown MDPs . University of California, Berkeley, 2007.
- Ambuj Tewari and Peter L Bartlett. Optimistic linear programming gives logarithmic regret for irreducible mdps. In Advances in Neural Information Processing Systems , pages 1505-1512, 2008.

- Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. arXiv preprint arXiv:1901.00210 , 2019.

## Contents

| Introduction                                                      |                                                                   | 1   |
|-------------------------------------------------------------------|-------------------------------------------------------------------|-----|
| 1.1                                                               | Contributions . . . . . . . . . . . . . . . . . . . . . .         | 2   |
| 1.2                                                               | Related Work . . . . . . . . . . . . . . . . . . . . . .          | 3   |
| 1.3                                                               | Problem Setting, Notation, and Organization . . . .               | 4   |
| 1.4                                                               | Optimistic Algorithms . . . . . . . . . . . . . . . . .           | 5   |
| Main                                                              |                                                                   |     |
|                                                                   | Results                                                           | 6   |
| 2.1                                                               | Sub-optimality Gap Lower Bound . . . . . . . . . . .              | 7   |
| 2.2                                                               | Why the dependence on gap min ? . . . . . . . . . . .             | 8   |
| 2.3                                                               | Interpolating with Minimax Regret for Small T . . .               | 8   |
|                                                                   | 'clipping'                                                        |     |
|                                                                   | Gap-Dependent bounds via                                          | 9   |
| 3.1                                                               | The Clipping Trick . . . . . . . . . . . . . . . . . .            | 11  |
| 3.2                                                               | Analysis of StrongEuler . . . . . . . . . . . . . . . . .         | 12  |
| 4 Conclusion                                                      | 4 Conclusion                                                      | 13  |
| A Notation and Organization                                       | A Notation and Organization                                       | 18  |
| Notation Table                                                    | Notation Table                                                    | 18  |
|                                                                   | Results and Analysis                                              |     |
| Precise                                                           | Precise                                                           | 20  |
| Precise Statement and Rigorous Proof Sketch of Main Regret Bounds | Precise Statement and Rigorous Proof Sketch of Main Regret Bounds | 20  |
| B.1                                                               | More Precise Statement of Regret Bound Theorem 2.4                | 20  |
| B.2                                                               | Rigorous proof of upper bounds: Preliminaries . . .               | 22  |
| B.3                                                               | Proof of Corollary B.1: A proof via integration . . .             | 24  |
| B.4                                                               | Proof of Theorem B.2 . . . . . . . . . . . . . . . . .            | 27  |
| Proof                                                             | of Technical Lemmas                                               | 29  |
| C.1                                                               | Proof of clipping with future bounds, Lemma B.6 . .               | 29  |
|                                                                   | (Lemma B.7) . .                                                   |     |
| C.2                                                               | Proof of sampling lemma . . . . .                                 | 30  |
| C.3                                                               | Proof of integral conversion, Lemma B.8 . . . . . . .             | 31  |
| C.4                                                               | Proof of interal conversion for G -bounds, Lemma B.10             | 32  |
| C.5                                                               | General Integral computations (Lemma B.9) . . . . .               | 32  |
| Proof                                                             | of 'clipping' bound: Proposition 3.1 / Theorem B.3                | 36  |
| D.1                                                               | Proof of Lemma D.2 . . . . . . . . . . . . . . . . . .            | 37  |
| D.2                                                               | Proof of Lemma D.1 . . . . . . . . . . . . . . . . . .            | 39  |
| D.3                                                               | Proof of Lemma D.3 . . . . . . . . . . . . . . . . . .            | 40  |
| II StrongEuler                                                    | and its surpluses                                                 | 42  |
| E The StrongEuler Algorithm                                       | E The StrongEuler Algorithm                                       | 42  |

| F Analysis of StrongEuler : Proof of Proposition B.4   | F Analysis of StrongEuler : Proof of Proposition B.4        | F Analysis of StrongEuler : Proof of Proposition B.4                        |   43 |
|--------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------|------|
|                                                        | F.1 . . . . . . . .                                         | Proof of Optimism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   43 |
| F.2                                                    | Proof of Surplus Bound Upper Bound                          | . . . . . . . . . . . . . . . . . . . . . . . . . .                         |   46 |
| F.3                                                    | Definition of A conc , and proofs of supporting lemmas      | . . . . . . . . . . . . . . . . .                                           |   49 |
| III                                                    | Lower Bounds                                                | Lower Bounds                                                                |   54 |
| G                                                      | Min-Gap Lower Bound for Optimistic Algorithms (Theorem 2.3) | Min-Gap Lower Bound for Optimistic Algorithms (Theorem 2.3)                 |   54 |
| G.1                                                    | Formal . . . . . . . .                                      | Statement . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         |   54 |
| G.2                                                    | Algorithm Class . . . . . . . . .                           | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   54 |
| G.3                                                    | Formal Lower Bound Instance .                               | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   55 |
| G.4                                                    | The Lower Bound: . . . . . . . .                            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   56 |
|                                                        | G.4.1 Proof of Proposition G.3 .                            | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   56 |
| G.5                                                    | Proof of Claim G.5 . . . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   59 |
| H                                                      | Information Theoretic Lower Bound (Proposition 2.2)         | Information Theoretic Lower Bound (Proposition 2.2)                         |   62 |
| H.1                                                    | Construction of the hard instance                           | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   62 |
| H.2                                                    | Regret Lower Bound                                          | Decomposition . . . . . . . . . . . . . . . . . . . . . . . . . . .         |   63 |
| H.3                                                    | Proof of Equation (30) . . . . . .                          | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |   63 |

```
General Notation glyph[lessorsimilar] denotes inequality up to a universal constant. f glyph[equalorsimilar] g denotes f glyph[lessorsimilar] g glyph[lessorsimilar] f . log + ( x ) := log max { x, 1 } . I denotes an indicator function M = ( S , A , [ H ] , p 0 , p, R ) denotes an MDP H denote the horizon, A and S denotes the space of actions and states A := |A| and S := |S| h ∈ [ H ], a ∈ A , x ∈ S are used for stages, actions, and states R ( x, a ) ∈ [0 , 1] denotes the R.V. with reward distribution at ( x, a ). r ( x, a ) := E [ R ( x, a )] denotes expected reward p 0 ( x ) denotes initial distribution of x 1 p ( x ′ | x, a ) denotes transition probability. M = SAH K denotes number of episodes, indexed with k ∈ [ K ] T = KH denotes total length of game.
```

## A Notation and Organization

Organization: This section describes the organization of the appendix, and clarifying our notation. The remainder of the appendix is divided into three parts:

Part I presents more detailed statements of the regret upper bounds obtained by StrongEuler , and their complete proofs. Section B.2 introduces Corollary B.1 and Theorem B.2, refining Corollary 2.1 and Theorem 2.4, from the main text. The section continues to prove both results. In addition, we introduce Theorem B.3, which refines the clipped regret decomposition Proposition 3.1. The proofs in this section rely on numerous technical lemmas, whose proofs are defered to Section C. Finally, this section states Proposition B.4, which ensures that StrongEuler is optimistic and provides a precise bound on the surpluses E k,h ( x, a ), described informally in Proposition 3.2.

In Part II, we present the StrongEuler algorithm and its guarantees. Section E describes how StrongEuler instantiates the model-based examples of optimistic algorithms described in Section 1.4; the algorithm and choice of confidence bonuses are specified in pseudocode. In Section F, we prove the surplus bound Proposition B.4, and verify that StrongEuler is strongly optimistic

Lastly, Part III contains the proofs of our lower bounds. Section G proves the Ω( S/ gap min ) lower bound described in Theorem 2.3, and rigorously describes the class of algorithms to which it applies. Finally, Section H proves the information theoretic lower bound, Proposition 2.2.

Notational Rationale: Unfortunately, the regret analysis of tabular MDPs requires significant notational overhead. Here we take a moment to highlight some notational conventions that we shall use throughout. The superscript ( · ) glyph[star] denotes 'optimal' quantities, i.e. the optimal policy π glyph[star] , the optimal value V glyph[star] , and variances of the optimal policy Var glyph[star] h,x,a . The accents ( · ) will be used to denote upper bounds on quantities, e.q. an optimistic Q-function Q k,h is an upper bound on Q glyph[star] k,h , and Var is an upper bound on the variance, and so on. ( · ) will denote lower bounds on quantities. For example, StrongEuler will maintain lower bounds on the values V k,h ≤ V glyph[star] . The accent ( ˇ · ) will pertain to clipped quantities; e.g. ˇ gap is the gap-value at which surpluses are clipped. Many quantities, like gap h ( x, a ) (gaps) and Var glyph[star] h,x,a (variances) depend on the triples ( x, a, h ). The quantities gap ( x, a ) and Var glyph[star] x,a with h suppresed to denote worse-case bounds on these term over h ∈ [ H ]; e.g. gap ( x, a ) := min h ∈ [ H ] gap h ( x, a ) and Var glyph[star] x,a := max h ∈ [ H ] Var glyph[star] h,x,a .

## Policies, Value Functions, Q-functions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

V π 0 denotes the value of π

Q π h ( x, a ) denotes Q-function of π

V glyph[star] 0 , V glyph[star] h ( x ) , Q glyph[star] ( x, a ) denote optimal value, value function, Q-function π glyph[star] h ( x ) denotes the set of optimal actions at h ∈ [ H ], x ∈ S .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Problem Dependent Quantities

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

## Quantities for Analysis

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

n k ( x, a ) denotes the number of times ( x, a ) is observed up to time k -1

<!-- formula-not-decoded -->

τ ( x, a ) denotes time after which n k ( x, a ) is sufficiently large

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

<!-- formula-not-decoded -->

## Part I

## Precise Results and Analysis

## B Precise Statement and Rigorous Proof Sketch of Main Regret Bounds

In this section, we present a precise statements and formal proofs of the upper bounds, Corollary 2.1 and Theorem 2.4, from the main text. These bounds both makes the improvements in the benign instances of Definition 2.2, and takes advantage of other possibly-favorable instance-specific quantities. The remainder of the section is organized as follows. In Section B.1, we introduce the relevant problem-dependent quantitites in terms of which we state our more refined bounds. We then state Corollary B.1, a more precise analogue of the log-regret bounds in Corollary 2.1, followed by Theorem B.2, which refines the regret bound Theorem 2.4 in interpolating between the log T and √ T regimes.

Next, in Section B.2, we set up the preliminaries for the proof of our upper bound, including (a) Theorem B.3, the granular clipping bound strengthening Proposition 3.1, (b) Proposition B.4, which upper bounds the surpluses E k,h ( x, a ) for StrongEuler , and (c) Lemma B.6 which combines the two into a useful form.

Then, in Section B.2, we present a rigorous proof of Corollary B.1 based on integration tools developed in Section C. Finally, we modify the arguments slightly to obtain the interpolation in Theorem B.2. The proof of Proposition 3.1 is given in Section F, Theorem B.3 is given in Section D, and the remainder of technical results in the present section are established in Section C.

We emphasize that the tools in this section provide a general recipe for establishing similar regret bounds for the existing model-based optimistic algorithms in the literature. We have attempted to present our tools in a modular fashion in hope that they can be borrowed to automate the proofs of similar guarantees in related settings.

## B.1 More Precise Statement of Regret Bound Theorem 2.4

We shall begin by stating a more precise version of Theorem 2.4, Following Zanette and Brunskill [2019], we begin by defnining the variances of the value optimal functions::

Definition B.1 (Variance Terms) . We define the variance of a triple ( x, a, h ) as

<!-- formula-not-decoded -->

and the statewise maximal variances as Var glyph[star] x,a := max h Var glyph[star] h,x,a , and the maximal variance as Var := max x,a,h Var glyph[star] h,x,a .

Remark B.1 (Typical Bounds on the variance ) . While Var ≤ H 2 for general MDPs (see e.g. [Azar et al., 2017]), we have Var is smaller for the benign instances in Definition 2.2. We briefly summarize this discussion from Zanette and Brunskill [2019]: If M has G bounded rewards, then V glyph[star] h +1 ( x ) ≤ G for any x , and thus Var glyph[star] h,x,a ≤ 1 + G 2 , which is glyph[lessorsimilar] 1 if G glyph[lessorsimilar] 1. For contextual bandits, p = p ( x, a ) does not depend on x, a , and V glyph[star] h +1 ( x ) = (max a R ( x, a )) + ( E x ′ ∼ p V glyph[star] h +2 ( x ′ )), where the second term does not dependent Hence, Var x ′ ∼ p [ V glyph[star] h +1 ( x ′ )] ≤ Var[(max a R ( x, a ))] ≤ 1, and thus Var glyph[star] h,x,a ≤ 2.

We can then define an associated 'effective horizon', which replaces H with a possibly smaller problem dependent quantity:

Definition B.2 (Effective Horizon) . Suppose that M has G -bounded rewards, as in Definition 2.2) We define the effective horizon as

<!-- formula-not-decoded -->

Since any horizonH MDP has H -bounded rewards, H T always satisfies H T ≤ min { H 2 , H log T } .

We note that the bound Var glyph[lessorsimilar] 1 for contextual bandits implies H T glyph[lessorsimilar] 1, whereas if M has G -bounded rewards with G glyph[lessorsimilar] 1, H T glyph[lessorsimilar] 1 ∧ 1 H log T .

Lastly, we shall introduce one more condition we call transition suboptimality , which is a notion of distributional closeness that enables the improved clipping and sharper regret bounds for the special case of contextual bandits (Definition 2.2):

Definition B.3 (Transition Sub-optimality) . Given α ∈ [0 , 1], we say that a tuple ( x, a, h ) is α -transition suboptimal if there exists an a glyph[star] ∈ π glyph[star] h ( x ) such that

<!-- formula-not-decoded -->

Intuitively, the condition states that the transition distributions p ( x, a ) and p ( x, a glyph[star] ) are close in a pointwise, multiplicative sense. This is motivated by the contextual bandit setting of Definition 2.2, where each ( x, a, h ) is exactly 0-transition suboptimal. For arbitrary MDPs, the bound p ( x ′ | x, a glyph[star] ) ≥ 0 implies that every triple ( x, a, h ) is at most 1-transition suboptimal. 6

With these definitions in place, we can state the more precise analogue of Corollary 2.1 as follows:

Corollary B.1 (Logarithmic Regret Bound for StrongEuler ) . Fix δ ∈ (0 , 1 / 2) , and let A = |A| , S = |S| , M = ( SAH ) 2 . Then with probability at least 1 -δ , StrongEuler run with confidence parameter δ enjoys the following regret bound for all K ≥ 2 :

<!-- formula-not-decoded -->

In particular, if M is an instance of contextual bandits, then Var can be replaced by 1 , H T can be replaced by 1 and max { α H, 1 } = 1 . If M has G glyph[lessorsimilar] 1 bounded rewards, then Var can be replaced by 1 in the above bound,

Moreover, our more precise analogue of Theorem 2.4, which interpolates between the log T and √ T regimes, is as follows:

Theorem B.2 (Main Regret Bound for StrongEuler ) . Fix δ ∈ (0 , 1 / 2) , and let A = |A| , S = |S| , M = ( SAH ) 2 . Let H T be as in Definition B.2, and suppose that each tuple ( x, a, h ) is α -transition suboptimal. Futher, define Z sub ( glyph[epsilon1] ) := { ( x, a ) ∈ Z sub : gap ( x, a ) &lt; glyph[epsilon1] } . Then with probability at least

6 The condition can be relaxed somewhat to only needing to hold for a set S for which p ( x ′ ∈ S | x, a ) is close to 1; for simplicity, consider the unrelaxed notion as defined as above.

1 -δ , StrongEuler run with confidence parameter δ enjoys the following regret bound for all K ≥ 2 :

<!-- formula-not-decoded -->

where the second inequality follows from the first with max { max glyph[epsilon1] |Z sub ( glyph[epsilon1] ) | , |Z opt |} ≤ SA . In particular, if M is an instance of contextual bandits, then Var can be replaced by 1 , H T can be replaced by 1 and max { α H, 1 } = 1 . If M has G glyph[lessorsimilar] 1 bounded rewards, then Var can be replaced by 1 in the above bound, and H T replaced by min { 1 , log T H } .

We observe that Theorem 2.4, and Corollary 2.1 are direct consequences of the above theorem.

Remark B.2 (Bounds on |Z opt | ) . Note that |Z opt |≤ ∑ x,h | π glyph[star] h ( x ) | ; in particular if for each ( x, h ) there is exactly one optimal action, then |Z opt |≤ H |S| . If in addition the same action is optimal at x for each h ∈ [ H ], then |Z opt | = |S| . For many environments |Z opt | glyph[lessorsimilar] |S| ; for instance, a race car doing many laps around a track may have h -dependent optimal actions in the first and last laps, but for the steady-state laps the optimal action will depend just on the current state.

Remark B.3 (Coupling Variances and Gaps) . For state action pairs ( x, a ) ∈ Z sub , Corollary B.1 Theorem B.2 suffer for the term (1 ∨ α H ) Var glyph[star] x,a gap ( x,a ) , where Var glyph[star] x,a := max h Var glyph[star] h,x,a is the maximal variance over stages, and gap ( x, a ) = min h gap h ( x, a ) is the minimal gap. This quantity can be refined to defend on roughly max h Var glyph[star] h,x,a gap h ( x,a ) , coupling the variance and gap terms. To do so, one needs to bin the gaps into intervals of [2 j -1 H, 2 j ] or integers j ∈ N , and apply numerous careful manipulations. In the interest of brevity, we defer the details to a later work.

## B.2 Rigorous proof of upper bounds: Preliminaries

We now turn to a rigorous proof of the regret bounds for StrongEuler : Corollary B.1 and Theorem B.2 (and consequently Theorem 2.4 and Corollary 2.1).

We first state our generalized surplus clipping bound in terms of the transition-suboptimality condition, which generalizes Proposition 3.1:

Theorem B.3. Suppose that each tuple ( x, a, h ) is α x,a,h transition-suboptimal, and set ˇ gap h ( x, a ) := gap min 2 H ∨ gap h ( x,a ) 4( H α x,a,h ∨ 1) . Then, if π k is induced by a strongly optimistic algorithm with surpluses E k,h ( x, a ) ,

<!-- formula-not-decoded -->

If the algorithm is optimistic but not strongly optimistic, then the above holds by replacing α x,a,h with 1 in the definition of ˇ gap h ( x, a ) .

The proof of the above theorem is given in Section D. We remark that the above theorem specializes to Proposition 3.1 by noting that each tuple ( x, a, h ) is 0-transition suboptimal for contextual bandits. For simplicitiy, we shall assume in the proof of Theorem B.2 that each state is α -suboptimal for a common α ; the bound can be straightforwardly refined to allow α to vary across ( x, a, h ).

Next, in order to ensure optimal H -dependence when interpolating with the O ( √ T ) regret bounds, we introduce policy-dependent variance quantities:

## Definition B.4. Define the variances

<!-- formula-not-decoded -->

where we recall that Var glyph[star] h,x,a := Var π ∗ h,x,a . Further, define Var ( k ) h,x,a = min { Var glyph[star] h,x,a , Var π k h,x,a } .

We are now ready to state the formal version of Proposition 3.2, which upper bounds the surpluses of StrongEuler , and verifies that the algorithm satisfies strong optimism:

Proposition B.4 (Surplus Bound for StrongEuler ) . There exists a universal constant c ≥ 1 and event A conc , with P [ A conc ] ≥ 1 -δ/ 2 , such that on A conc , for all x ∈ S , a ∈ A , h ∈ [ H ] and k ≥ 1 ,

<!-- formula-not-decoded -->

where have defined the terms

<!-- formula-not-decoded -->

The above proposition is proven in Appendix F. Here B lead denotes a 'lead term' in the analysis, which contributes to the dominate factors in our regret bounds. B fut notates 'future' bound terms under a rollout of π k starting at a given triple ( x, a, h ); these terms are responsible for the lower ˜ O ( SAH 4 ( S ∨ H ) ) -term in the regret.

Remark B.4 (Remarks on Proposition B.4) . First, the dominant term in the upper bound on E k,h is B lead k,h ( x, a ), which decays as ˜ O ( n k ( x, a ) -1 / 2 ) . The terms B fut k ( x, a ) decay more rapidly ˜ O ( n k ( x, a ) -1 ) , and will thus be responsible for the (nearly gap-free) portion of the regret. Second, in order to analyze similar optimistic algorithms in the same vein (e.g. [Azar et al., 2017, Dann and Brunskill, 2015, Dann et al., 2017]), one would instead prove the appropriate analogue to Proposition B.4 and follow the remaining steps of the present proof. Little would change, except one would be forced to replace Var glyph[star] h,x,a with a more pessimistic, less problem-dependent quantity. Lastly, note that the lead term B lead k,h ( x, a ) depends on the minimum of the variance of the optimal value function, Var glyph[star] h,x,a and of the variance of the value function for π k , Var π k h,x,a . As in the aforementioned works, this dependence on Var π k h,x,a is crucial for obtaining the correct minimax ˜ O ( √ HSAT ) regret.

Next, let us combine Proposition B.4 with our main clipping theorem, Theorem B.3. Since E k,h ( x, a ) glyph[lessorsimilar] B lead k,h ( x, a ) + E π k [ ∑ ( . . . ) | ( . . . )], combining the two results into a convenient form requires that we reason about how to distribute clipping operations across sums of terms. To this end, we invoke the following technical lemma:

Lemma B.5 (Distributing the clipping operator) . Let m ≥ 2 , a 1 , . . . , a m ≥ 0 , and glyph[epsilon1] ≥ 0 . clip [ ∑ m i =1 a i | glyph[epsilon1] ] ≤ 2 ∑ m i =1 clip [ a i | glyph[epsilon1] 2 m ] .

Proof. Let us assume without loss of generality 0 ≤ a 1 ≤ . . . ≤ a m , and that ∑ m i =1 a i ≥ glyph[epsilon1] . Defining the index i ∗ := min { i : a i ≥ glyph[epsilon1] 2 m } , we observe that a i ∗ ≥ glyph[epsilon1] 2 m , and since ( a i ) are non-decreasing by assumption, ∑ m i = i ∗ a i = ∑ m i = i ∗ clip [ a i | glyph[epsilon1] 2 m ] ≤ ∑ m i =1 clip [ a i | glyph[epsilon1] 2 m ] . It therefore suffices to show that ∑ m i =1 a i ≤ 2 ∑ m i = i ∗ a i . To this end, we see that, since a i ≤ glyph[epsilon1] 2 m for i &lt; i ∗ , ∑ i ∗ -1 i =1 a i ≤ ∑ i ∗ -1 i =1 glyph[epsilon1] 2 m ≤ ( i ∗ -1) glyph[epsilon1] 2 m ≤ glyph[epsilon1]/ 2. On the other hand, since ∑ m i =1 a i ≥ glyph[epsilon1] , we must have that ∑ m i = i ∗ a i ≥ glyph[epsilon1] 2 , and thus ∑ m i =1 a i ≤ 2 ∑ m i = i ∗ a i , as needed.

Applying the above lemma careful, we arrive at the following useful regret decomposition:

Lemma B.6 (Clipped Regret Decomposition Lead and Future Bounds) . Let ˇ gap min = min x,a,h ˇ gap h ( x, a ) . Then on the event A conc the regret of StrongEuler is bounded by

<!-- formula-not-decoded -->

where c is a universal constant.

## B.3 Proof of Corollary B.1: A proof via integration

Note that Lemma B.6 bounds Regret K by a sum of local bounnds terms B lead k,h ( x, a ) and B fut k , which depend only on the number of samples n k ( x, a ) obtained from state action pair ( x, a ). More precisesly, we can represent the bound terms by defining the functions

<!-- formula-not-decoded -->

Further, define glyph[epsilon1] lead x,a := min h ˇ gap h ( x,a ) 4 , glyph[epsilon1] fut := ˇ gap min 8 SAH , and lastly set

<!-- formula-not-decoded -->

Then, recalling the definitions of B lead , B fut , and the fact that B lead k,h ( x, a ) ≤ H and B fut k ( x, a ) ≤ H 3 , we can write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As described in Section 3, the crucial step now is to relate the empirical conunts n k ( x, a ) to the visitation probabililties. Precisely, let us aggregate

<!-- formula-not-decoded -->

Note that if {F k } denotes the filtration corresponding to the episodes k , then, E [ n k ( x, a ) | F k -1 ] = n k -1 ( x, a )+ ω k -1 ( x, a ). In other words, n k -1 ( x, a ) is precise the sum of the increments E [ n j ( x, a ) -n j -1 ( x, a ) | F j -1 ] for j = 1 , . . . , k -1. 7 . Hence, by a now-standard martingale concentration argument, we find that n k ( x, a ) will be be lower bounded by n k ( x, a ), provided that the latter quantity is sufficiently large. More precisely:

Lemma B.7 (Sampling Event) . Define the event

<!-- formula-not-decoded -->

Then, for some H sample glyph[lessorsimilar] H log M δ , E samp ( H sample ) holds with probability at least 1 -δ/ 2 .

Lemma B.7 is proved in Appendix C.2 as a consequence of Dann et al. [2018, Lemma 6]. Together, the events A conc and E samp account for 1 -δ probability with which our regret bounds hold. For short, we will let E samp denote E samp ( H sample ) when clear from context, and τ := τ H sample . After neglecting the first τ ( x, a ) samples in the sum (6), we can approximately bound

<!-- formula-not-decoded -->

where glyph[lessorapproxeql] denotes an informal inequality. Now, ω k,h ( x, a ) and n k ( x, a ) / 4 are directly related via n k ( x, a ) / 4 := ∑ k j =1 ∑ H h =1 ω j,h ( x, a ). Hence, we can view the above regret bounds as discrete integrals of the functions f lead x,a and f fut ( n k ( x, a ) / 4). This argument is made precise by the following lemma, which comprises the workhorse of out argument:

Lemma B.8 (Integral Conversion) . Suppose that the event E samp ( H sample ) holds. Then, for any collection of functions f x,a ( · ) non-increasing functions from N → R bounded aboved by f max and any glyph[epsilon1] x,a,h ≥ 0 , we have that

<!-- formula-not-decoded -->

7 Note that we induce n k ( x, a ) to include a sum up to index k ; this makes the following arguments more convenient, and will only accrue constant factors in the analysis

Since H sample glyph[lessorsimilar] H log( M/δ ), and the functions f fut x,a , f lead x,a are bounded by H 3 , we see that, on E samp ∩ A conc , it holds that

<!-- formula-not-decoded -->

where for the term on the second line, we have bounded n K ( x, a ) ≤ T and used that f fut ( · ) ≥ 0.

All that remains is to evaluate the above integrals. This is directly adressed by the following technical lemma, proved in Section C.5:

Lemma B.9 (General Integration Computations) . Let f ( u ) ≤ min { f max , clip [ g ( u ) | glyph[epsilon1] ] } where glyph[epsilon1] ∈ [0 , H ] and g ( u ) is a non-increasing function is specified in each of two cases that follow. Further, let M ≥ 1 , and δ ∈ (0 , 1 / 2) be problem dependent constants. Finally, let glyph[lessorsimilar] denote inequality up to a problem independent constant. Then, the following integral computations hold:

- (a) Suppose that C &gt; 0 is a problem depedendent constant satisfying log C glyph[lessorsimilar] log(2 M ) , and that g ( u ) glyph[lessorsimilar] √ C log( Mu/δ ) u . Then,

<!-- formula-not-decoded -->

- (b) Suppose that C, C ′ &gt; 0 are a problem depedendent constant satisfying log( CC ′ ) glyph[lessorsimilar] log 2 M , and that g ( u ) glyph[lessorsimilar] C (√ C ′ log( Mu/δ ) u + C ′ log( Mu/δ ) u ) 2 . Then,

<!-- formula-not-decoded -->

Note that the special case g ( u ) glyph[lessorsimilar] C log( Mu/δ ) u can be obtained by setting C ′ = 1 in the above inequality.

Lastly, the above computations hold if f ( u/ 4) is replaced by f ( u/c ) for any universal constant c &gt; 0 . Moreover, the above computations hold if f ( u ) glyph[lessorsimilar] min { f max , g ( u ) } by taking glyph[epsilon1] = 0 and setting 1 glyph[epsilon1] = ∞ .

Remark B.5 (Integration without anytime bounds) . If instead we consider functions g ( u ) satisfying the looser bounds (a) g ( u ) glyph[lessorsimilar] √ C log( MT/δ ) u and (b) g ( u ) glyph[lessorsimilar] C (√ C ′ log( MT/δ ) u + C ′ log( MT/δ ) u ) 2 for T ≥ N , then we can recover the bounds

<!-- formula-not-decoded -->

These sorts of bounds arise when the confidence intervals are derived via union bounds over all time T , rather than via anytime estimates. In particular, we see that using a naive union bounded over all time T incurs a dependence on log T · (log log T ), and thus does not imply a strictly O (log T ) regret.

Let us conclude by applying the above lemma to the terms at hand. First, applying the Part (a) with f = f lead x,a , g = g lead x,a , C = Var glyph[star] x,a , and H ≥ glyph[epsilon1] = glyph[epsilon1] lead x,a := min h ˇ gap h ( x,a ) 4 glyph[greaterorsimilar] gap ( x,a ) (1 ∨ α H ) for ( x, a ) ∈ Z sub , and H ≥ glyph[epsilon1]x,a glyph[greaterorsimilar] gap min H for ( x, a ) ∈ Z opt , we have that

<!-- formula-not-decoded -->

Similarly, applying the Part (b) with f = f fut , g = g fut , C ′ = S and C = H 3 , and glyph[epsilon1] = glyph[epsilon1] fut := ˇ gap min 8 SAH glyph[greaterorsimilar] gap min H 2 (and also satisfying glyph[epsilon1] fut ≤ H ), we can bound

<!-- formula-not-decoded -->

Plugging the above two displays into (6) concludes the proof of Corollary B.1.

## B.4 Proof of Theorem B.2

We conclude the section by proving the regret bound of Theorem B.2, which interpolates between the √ T and log T regimes. Let us recall the subset Z sub ( glyph[epsilon1] ) := { ( x, a ) : gap ( x, a ) &lt; glyph[epsilon1] } , as well as H T := min { Var , G 2 H log T } . . Retracing the proof of Corollary B.1, it suffices to establish only two points:

<!-- formula-not-decoded -->

For both of these inequalities, we will discard the clipping, and thus the two bounds will be syntatically the same. Hence, let us simply prove the following bound:

<!-- formula-not-decoded -->

Since H T := min { Var , G 2 H log T } , it suffices to prove the above bound first with H T replaced by Var , and then replaced by G 2 H log T .

Bound with Var : To obtain a bound involving Var , we use the fact that B lead k,h ( x, a ) glyph[lessorsimilar] g lead ( n k ( x, a )), for the function g lead ( u ) = √ Var log( Mu/δ ) /u . Hence, following the integration arguments in the proof of Corollary B.1, clipped at glyph[epsilon1] = 0, we can bound

<!-- formula-not-decoded -->

Hence, by Cauchy Schwartz, and the bound ∑ ( x,a ) ∈Z opt n K ( x, a ) ≤ ∑ ( x,a ) n K ( x, a ) = T ,

<!-- formula-not-decoded -->

as needed.

Bound with H T : This bound requires a little more subtely. Define the function f ( u ) = (1 / max { u, 1 } ). Then, using the definition of B lead k,h ( x, a ) from Proposition B.4, we have

<!-- formula-not-decoded -->

Applying the recipe we used for Corollary B.1 will not quite carry over in this setting. Instead, we apply an argument based on Cauchy-Schwartz, defered to Section C.4:

Lemma B.10 (Cauchy-Schwartz Integration Lemma for G -bounds) . Let { V x,a,k,h } be a sequence of numbers, and let f ( u ) be a nonnegative, non-decreasing function, f max &gt; 0 , L a problem dependent parameter, and let Z 0 ⊂ S × A . Then, on E samp ,

<!-- formula-not-decoded -->

We apply the above lemma with V x,a,k,h = Var π k x,a,h , f max = H and f ( u ) = 1 / max { u, 1 } , and L = log( MT/δ ). It is easy to see that the term |Z 0 | H sample f max glyph[lessorsimilar] SAH 2 log( M/δ ) will already

absorbed into terms already present in the final bound. On the other hand, by a now-standard law of total variance argument,

<!-- formula-not-decoded -->

where the last inequality is from the proof of Zanette and Brunskill [2019, Proposition 6]. On the other hand, we can bound( Hf ( H ) + ∫ T 1 f ( u ) du ) ≤ 1 + log T . This finally yields

<!-- formula-not-decoded -->

as needed.

## C Proof of Technical Lemmas

## C.1 Proof of clipping with future bounds, Lemma B.6

Since strong optimistm holds on A conc , Theorem B.3 yields

<!-- formula-not-decoded -->

Applying Lemma B.5 with m = 2, a 1 = cB lead k,h ( x, a ), and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The term on the right hand side of the first line of the above display is exactly as needed. Let us turn our attention to the term on the second line. We have that

<!-- formula-not-decoded -->

Hence, applying Lemma B.5 with the terms a i -terms corresponding to B fut k ( x ′ , a ′ ) P [( x t , a t ) = ( x ′ , a ′ ) | ( x h , a h ) = ( x, a )] and the number of such terms m bounded by SAH , we have

<!-- formula-not-decoded -->

Since clip [ αx | glyph[epsilon1] ] ≤ α clip [ x | glyph[epsilon1] ] for α ≤ 1, and since the probabilities P [( x t , a t ) = ( x ′ , a ′ ) | ( x h , a h ) = ( x, a )] are bounded by 1, we can bound the above by

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Altogether,

<!-- formula-not-decoded -->

Summing over k = 1 , . . . , K proves the inequality.

## C.2 Proof of sampling lemma (Lemma B.7)

Recall ( E samp ) ′ := {∀ k, s, a : n k ( x, a ) ≥ 1 2 n k -1 ( x, a ) -H log 2 HSA δ } ; Lemma 6 in Dann et al. [2018] in shows that this event occurs with probability at least 1 -δ/ 2. We show that ( E samp ) ′ ⊆ E samp ( H sample ), for H sample = 4 H log 2 HSA δ glyph[lessorsimilar] H log M δ .

Noting that n k ≤ n k -1 + H , ( E samp ) ′ implies that n k ≥ 1 2 n k ( x, a ) -H log 2 HSA δ -H = 1 2 n k ( x, a ) -H log 2 eHSA δ . Hence, for any k ≥ τ ( x, a ), we have n k ≥ 4 H log 2 HSA eδ and thus n k ( x, a ) ≥ n k 4 + n k 4 -H log 2 eHSA δ ≥ n k 4 . Bounding log 2 HSA eδ ≤ ˜ L (1) concludes the proof.

## C.3 Proof of integral conversion, Lemma B.8

Recall that τ ( x, a ) denote inf { k : n k ( x, a ) ≥ H sample } . Then,

<!-- formula-not-decoded -->

since ∑ τ ( x,a ) -1 k =1 ω k ( x, a ) = n τ ( x,a ) -1 ( x, a ) ≤ H sample madn f ( · ) ≤ f max . We now appeal to the following integration lemma, which we prove momentarily.

Lemma C.1 (Integration over ω k ( x, a )) . Let f : [ H, ∞ ) → R &gt; 0 be a non-increasing function. Then,

<!-- formula-not-decoded -->

To conclude the proof of Lemma B.8, we apply the above for each ( x, a ) with the functions f ( u ) ← f x,a ( u/ 4), and note Hf x,a ( H/ 4) ≤ Hf max ≤ H sample f max .

Proof of Lemma C.1. The proof generalizes Lemma E.5 in Dann et al. [2017]. For ease of notation, define k 0 = τ ( x, a ). Wecan define the step function g : [ k 0 , K ] → R via g ( t ) = ∑ K -1 k = k 0 ω k +1 ( x, a ) I ( t ∈ [ k, k +1)]. Then, letting G ( t ) := n k 0 ( x, a ) + ∫ t 0 g ( u ) du , we see that G ′ ( t ) = g ( t ) almost everywhere, G is non-decreasing, and G ( k ) = n k ( x, a ) for all k ∈ [ k 0 , K ]. We can therefore express

<!-- formula-not-decoded -->

where ( i ) uses the fact that f ◦ G is non-increasing, ( ii ) is the Fundamental Theorem of Calculus, with G ′ ( t ) = g ( t ), and ( iii ) is G ( k ) = n k ( x, a ) for k ∈ [ k 0 , K ]. Hence, we have the bound

<!-- formula-not-decoded -->

where ( i ) uses ω k 0 ≤ H , and that f ( u ) ≥ 0, and ( ii ) uses the fact that f is nonincreasing, and n k 0 ( x, a ) ≥ H sample ≥ H .

## C.4 Proof of interal conversion for G -bounds, Lemma B.10

Let τ ( x, a ) = τ H sample ( x, a ). Then, as in the proof of Lemma B.8,

<!-- formula-not-decoded -->

By Cauchy-Schwartz

<!-- formula-not-decoded -->

The first term in the above product can be bounded as

<!-- formula-not-decoded -->

Using Lemma C.1, the second term can be bounded as

<!-- formula-not-decoded -->

## C.5 General Integral computations (Lemma B.9)

For convenience, let us restate the lemma we are about to prove.

Lemma B.9 (General Integration Computations) . Let f ( u ) ≤ min { f max , clip [ g ( u ) | glyph[epsilon1] ] } where glyph[epsilon1] ∈ [0 , H ] and g ( u ) is a non-increasing function is specified in each of two cases that follow. Further, let M ≥ 1 , and δ ∈ (0 , 1 / 2) be problem dependent constants. Finally, let glyph[lessorsimilar] denote inequality up to a problem independent constant. Then, the following integral computations hold:

- (a) Suppose that C &gt; 0 is a problem depedendent constant satisfying log C glyph[lessorsimilar] log(2 M ) , and that g ( u ) glyph[lessorsimilar] √ C log( Mu/δ ) u . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) Suppose that C, C ′ &gt; 0 are a problem depedendent constant satisfying log( CC ′ ) glyph[lessorsimilar] log 2 M , and that g ( u ) glyph[lessorsimilar] C (√ C ′ log( Mu/δ ) u + C ′ log( Mu/δ ) u ) 2 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the special case g ( u ) glyph[lessorsimilar] C log( Mu/δ ) u can be obtained by setting C ′ = 1 in the above inequality.

Lastly, the above computations hold if f ( u/ 4) is replaced by f ( u/c ) for any universal constant c &gt; 0 . Moreover, the above computations hold if f ( u ) glyph[lessorsimilar] min { f max , g ( u ) } by taking glyph[epsilon1] = 0 and setting 1 glyph[epsilon1] = ∞ .

Proof. By inflating C by a problem-independent constant if necessary, we may assume without loss of generality that g ( u ) = √ C log( Mu/δ ) /u in part (a) and g ( u ) = C ( √ C ′ log( Mu/δ ) /u + √ C ′ log( Mu/δ ) /u ) 2 , with equality rather than approximate inequality glyph[lessorsimilar] .

Next, define

<!-- formula-not-decoded -->

Throughout, we shall assume the case glyph[epsilon1] &gt; 0, as the glyph[epsilon1] = 0 can be derived by just taking n end = N . Note then that f ( u/ 4) = clip [ g ( u/ 4) | glyph[epsilon1] ] = 0 for all u &gt; n end . Hence, it suffices to upper bound

<!-- formula-not-decoded -->

Lastly, let us define ˜ L ( u ) := log( Mu/δ ) for u ≥ H . We shall rquire the following inversion lemma, which is standard in the multi-arm bandits literature.

Lemma C.2 (Inversion Lemma) . There exists a universal constant c &gt; 0 such that for all b ≥ 0 , ˜ L ( u ) /u ≤ b as long as u ≥ ˜ L (1 + b -1 ) /cb . Moreover, for u glyph[lessorsimilar] ˜ L ( b -1 ) /cb , it holds that ˜ L ( u ) glyph[lessorsimilar] ˜ L (1 + b -1 ) .

Proof. Let u = ˜ L (1 /b ) /cb for a constant c to be chosen shortly. Then,

<!-- formula-not-decoded -->

where we use log log( x ) ≤ x and ˜ L (1 + b -1 ) ≥ ˜ L (1) ≥ log 2. It is easy to see that this quantity is less than b for a constant c sufficiently small that does not depend on M,δ,b . The second statement follows from an analogous computation.

Proof of Part (a): Suppose g ( u ) = √ C ˜ L ( u ) u . It is straightforward to bound

<!-- formula-not-decoded -->

To conclude, let us find n end ( x, a ). By our inversion Lemma C.2, we can see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, for if log C glyph[lessorsimilar] log M and glyph[epsilon1] ≤ H , we can bound ˜ L ( 1 + C glyph[epsilon1] ) glyph[lessorsimilar] log MH glyph[epsilon1] . Hence, we haveshow

<!-- formula-not-decoded -->

To conclude, it remains to show that we can replace H glyph[epsilon1] with N . For this, we use a simpler argument:

<!-- formula-not-decoded -->

Using similar arguments to above, we can bound ∫ N H clip [ 1 √ u | glyph[epsilon1] ′ ] glyph[lessorsimilar] 1 glyph[epsilon1] ′ , yielding the bound ∫ N H f ( u/ 4) du glyph[lessorsimilar] ˜ L 1 / 2 ( T ) glyph[epsilon1] ′ = ˜ L ( T ) glyph[epsilon1] .

Proof of Part (b): A first step This proof will require slightly more care than part (b). We shall first require the following lemma:

Claim C.3. In the setting of Lemma B.9, if g ( u ) = C log Mu δ u = C ˜ L ( u ) u , then

<!-- formula-not-decoded -->

Proof of Claim C.3. Define n 0 = 2 + log( M/δ ). Then, we have

<!-- formula-not-decoded -->

Therefore,

Now take g ( u ) = C ˜ L ( u ) /u . Since ˜ L ( u ) glyph[lessorsimilar] log( M/δ ) + log( u ) for u ≥ n 0 ≥ 2, it it is straightforward to bound

<!-- formula-not-decoded -->

where in the final inequality, we use N ≤ T , M/δ ≥ 2, and n 0 ≥ 1. By the same token, we can crudely bound the above by C glyph[lessorsimilar] log 2 ( MT/δ ).

Let us now develop a more refined bound by taking advantage of n end . By our inversion lemma, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, with some algebra we can bound

<!-- formula-not-decoded -->

This leads to the more refined bound ∫ n 0 + N ∧ n end n 0 g ( u/ 4) du glyph[lessorsimilar] C log(1 + C glyph[epsilon1] ) log( ˜ MT ). Again, since log C glyph[lessorsimilar] log M and glyph[epsilon1] ≤ H , we bound again bound log(1 + C glyph[epsilon1] ) glyph[lessorsimilar] log MH glyph[epsilon1] .

## Concluding the proof of Part (b) Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that for u ≥ n 0 , g ( u/ 4) glyph[lessorsimilar] h ( u/ 4), where h ( u ) ≤ CC ′ u ˜ L ( u ). Hence, applying the bound from Lemma C.3 with C ← CC ′ , we have

<!-- formula-not-decoded -->

On the otherhand, by our inversion lemma and using C ′ ≤ ˜ M O (1) , we can bound

<!-- formula-not-decoded -->

Combining these two pieces yields the bound.

Since n 0 = 1 + log M δ ,

Then, we have

## D Proof of 'clipping' bound: Proposition 3.1 / Theorem B.3

In this section, we prove Theorem B.3 (of which Proposition 3.1 is a direct consequence), which allows us to clip the surpluses when they are below a certain value. The center of our analysis is the following lemma, which tells us that if gap h ( x, a ) &gt; 0 for a pair ( x, a, h ), then either the surplus E k,h ( x, a ) is large, or expected difference in value functions at the next stage, p ( x, a ) glyph[latticetop] ( V k,h +1 -V π k h +1 ), is large:

Lemma D.1 (Fundamental Gap Bound) . Then suppose that Alg is strongly optimistic, and consider a pair ( x, a, h ) with gap h ( x, a ) &gt; 0 which is is α -transition optimal. Then

<!-- formula-not-decoded -->

If Alg is possibly not strongly optimistic, then the above holds still holds α = 1 .

Lemma D.1 is established in Section D.2. Notice that as α gets close to zero, the above bound implies that when E k,h ( x, a ) is much smaller than the gap h ( x, a ), the difference in value functions at the next stage, p ( x, a ) glyph[latticetop] ( V k,h +1 -V π k h +1 ), must become even larger to compensate. The extreme case is α = 0, e.g. in contextual bandits, where the gap always lower bounds the surplus.

Continuing with the proof of Theorem B.3, we begin with the 'half-clipping' which clips the surpluses at at most gap min :

Definition D.1 (Half Clipped Value Function) . We define the half-clipped surplus ¨ E k,h ( x, a ) := clip [ E k,h ( x, a ) | glyph[epsilon1] clip ], where glyph[epsilon1] clip := gap min / (2 H ). We set ¨ V π k k,H +1 ( x ) = 0 for all x ∈ S , and recursively define

<!-- formula-not-decoded -->

denote the value and Q-functions of under π k associated with MDP whose transitions are transitions p ( · , · ) and non-stationary rewards r ( x, a ) + ¨ E k,h ( x, a ) at stage h .

After the half-clipping has been introduced, it is no longer the case that π k is optimal for this half clipped MDP. As a result, it is not certain that the half-clipped Q-function for π k is optimistic in the sense that ¨ Q π k k,h ( x, a ) ≥ Q glyph[star] h ( x, a ). We shall instead show that if ¨ V π k,h is approximately optimistic, in the sense that its excess relative to V π k , ¨ V π k,h 0 -V π k 0 is at least a constant factor of the regret V glyph[star] 0 -V π k 0 :

Lemma D.2 (Lower Bound on Half-Clipped Surplus) . For glyph[epsilon1] clip = gap min / 2 H , it holds that

<!-- formula-not-decoded -->

The above bound is established in Section D.1. Hence, to establish the bound of Theorem B.3, it suffices to bound the gap ¨ V π k,h 0 -V π k 0 . For a given h , and an x : π k,h ( x ) / ∈ π glyph[star] h ( x ), let us consider the difference

<!-- formula-not-decoded -->

We now introduce the following lemma, proven Section D.3, which allows us to further clip the bonus for suboptimal actions a / ∈ π glyph[star] h ( x ), i.e. , actions with gap h ( x, a ) &gt; 0:

Lemma D.3 (Gap Clipping) . Suppose either Alg is strongly optimistic and each tuple is α x,a,h -transition suboptimal. Then the fully-clipped surpluses

<!-- formula-not-decoded -->

satisfy the bound

<!-- formula-not-decoded -->

If Alg is just optimistic, then the above bound holds with α x,a,h = 1 .

Unfolding the above lemma, and noting that even when Alg is not strongly optimistic, the clipping ensures that ˇ E k,h ( x, a ) ≥ 0, so that we can bound

<!-- formula-not-decoded -->

where we recall ω k,h ( x, a ) = P π k [( x h , a h ) = ( x, a )]. Combining with our earlier bound V glyph[star] 0 -V π k 0 ≤ 2( ¨ V π k k, 0 ( x ) -V π k 0 ( x )) from Lemma D.2, we find that V glyph[star] 0 -V π k 0 ≤ 2 e ∑ x,a ∑ H h =1 ω k,h ( x, a ) ˇ E k,h ( x, a ), thereby demonstrating Theorem B.3.

## D.1 Proof of Lemma D.2

We can with a crude comparison between the clipped and optimistic value functions.

<!-- formula-not-decoded -->

Proof. The bound ¨ E k,h ( x, π k,h ( x )) ≥ E k,h ( x, π k,h ( x )) -glyph[epsilon1] clip follows directly from

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( i.a ) and ( i.b ) follow by recursively unfolding the identities ¨ V π k k,h ( x ) -V π k h ( x ) = ¨ E k,h ( x, a ) + p ( x, a ) glyph[latticetop] ( ¨ V π k k,h +1 ( x ) -V π k h ( x )) and V k,h ( x ) -V π k h ( x ) = E k,h ( x, a )+ p ( x, a ) glyph[latticetop] ( V k,h +1 ( x ) -V π k h ( x )) .

We now turn to proving Lemma D.2.

Proof. The strategy is as follows. Weshall introduce the events over P π k , E h := { π k,h ( x h ) / ∈ π glyph[star] h ( x h ) } , which is the event that the policy π k,h does not prescribe an optimal action x h . We further define the events

<!-- formula-not-decoded -->

which is the event that the policy π k agrees with an optimal action on x 1 , . . . , x h -1 , and disagrees on x h . Below, our goal will be to establish the following two formulae for the suboptimality gap V glyph[star] 0 -V π k 0 and ¨ V π k 0 -V π k 0 :

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Note that on A h , E h = { π k,h ( x h ) / ∈ π glyph[star] h ( x h ) } also occurs, and therefore gap ( x h , π k,h ( x h )) ≥ gap min . In particular, displays (10) and (11) both imply

<!-- formula-not-decoded -->

where ( i ) uses glyph[epsilon1] clip = gap min 2 H and display (10), ( ii ) uses that Q glyph[star] h ( x h , π k,h ( x h )) -V π k h ( x ) ≥ 0, and ( iii ) uses display (11).

Let us start with proving (10). First, consider a stage h , state x , and suppose that π k,h ( x ) / ∈ π glyph[star] h ( x ). Observe that by Lemma D.4, optimism, and the definition of gap h ( x, a ), we have that for any a glyph[star] ∈ π glyph[star] h ( x ),

<!-- formula-not-decoded -->

Subtracting, we find that for π k,h ( x ) / ∈ π glyph[star] h ( x ),

<!-- formula-not-decoded -->

Now, on the other hand, if π k,h ( x ) ∈ π glyph[star] h ( x ), then,

<!-- formula-not-decoded -->

where in ( i ) we have defined the increment ∂ ¨ V h := ¨ V π k,h k,h -V π k h with ∂ ¨ V H +1 = 0, and ( ii ) holds since ¨ E k,h ( x, π k,h ( x )) = E k,h ( x, π k,h ( x )) I ( E k,h ( x, π k,h ( x )) ≥ glyph[epsilon1] clip ) ≥ 0.

Now, recalling that E h denotes the event that π k,h ( x ) / ∈ π glyph[star] h ( x ), we have

<!-- formula-not-decoded -->

We continue with

<!-- formula-not-decoded -->

Recalling the event A h = E h ∩ ⋂ h ′ &lt;h E c h ′ , we can continue the above induction to find that,

<!-- formula-not-decoded -->

as needed. Now let's prove (11). We can always write

<!-- formula-not-decoded -->

where gap h ( x, a ) = 0 when π glyph[star] h ( x ) ∈ π k,h ( x ), that is, on E c . Hence, the same line of reasoning used to prove Eq. (10) (omitting the subtracted glyph[epsilon1] clip H ), verifies Eq. (11).

## D.2 Proof of Lemma D.1

Proof. For simplicity, set a = π k,h ( x ), and let a glyph[star] ∈ π glyph[star] h ( x ) be an action which witnesses the α transition-suboptimality condition. We then have

<!-- formula-not-decoded -->

where ( i ) is by definition of V k,h ( x ), ( ii ) is since a = π k,h ( x ) = arg max a ′ Q k,h ( x, a ′ ), and ( iii ) is the definition of gap h ( x, a ). Rearranging, we have

<!-- formula-not-decoded -->

If Alg is not necessarily strongly optimistic then we bound Q k,h ( x, a glyph[star] ) -Q glyph[star] ( x, a glyph[star] ) ≥ 0 and Q glyph[star] h ( x, a ) ≥ V π k h ( x ), yielding

<!-- formula-not-decoded -->

which corresponds to the desired bound for α = 1.

When Alg is strongly optimistic, we handle (15) more carefully. Specifically, we compute

<!-- formula-not-decoded -->

Moreover, recalling that a ∗ ∈ π glyph[star] h ( x ), we have

<!-- formula-not-decoded -->

where the last inequality uses strong optimism of Alg . Hence,

<!-- formula-not-decoded -->

where the lastar line uses the component-wise inequalityes p ( x, a ) -p ( x, a glyph[star] ) ≤ α p ( x, a ) due to the fact that a glyph[star] witnesses the α transition-suboptimality, and V k,h -V glyph[star] h +1 ≥ 0 due to optimism.

## D.3 Proof of Lemma D.3

Proof. For ease, we suppress the dependence of α on ( x, a, h ). By our fundamental gap bound (Lemma D.1) and then Lemma D.4, we have that

<!-- formula-not-decoded -->

where the inequality bounds α ( H -h + 1) glyph[epsilon1] clip ≤ α gap min / 2 ≤ α · gap h ( x, a ) / 2 ≤ gap h ( x, a ) / 2. This yields

<!-- formula-not-decoded -->

Now, fix a constant c ∈ (0 , 1] to be chosen later. Either we have that ¨ E k,h ( x, a ) ≥ c 2 gap h ( x, a ), or otherwise,

<!-- formula-not-decoded -->

which can be rearranged into

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

and thus,

<!-- formula-not-decoded -->

In particular, choosing c = 1 2 min { 1 , ( α H ) -1 ) } , we have (1 + c α 1 -c ) ≤ 1 + 1 H , and

<!-- formula-not-decoded -->

so that ¨ E k,h ( x, a ) I { ¨ E k,h ( x, a ) ≥ c 2 gap h ( x, a ) } = ˇ E k,h ( x, a ). This concludes the proof.

## Part II StrongEuler

## and its surpluses

## E The StrongEuler Algorithm

Before continuing, let us define a logarithmic factor we shall use throughout:

<!-- formula-not-decoded -->

where we recall that M = SAH ≥ 2. This section formally presents StrongEuler , which makes two subtle modification of the EULER algorithm of Zanette and Brunskill [2019].

First, similar to [Dann et al., 2017, 2018], StrongEuler refines the log factors in the bonuses to depend on the number of samples n k ( x, a ) via L ( n k ( x, a )) ∝ log( Mn k ( x, a ) /δ ), rather than the overall time T = KH via L ( n k ( x, a )) ∝ log( MT/δ ), which is necessary to ensure the optimal log T regret. Following [Dann et al., 2017, 2018], our confidence bounds can be slightly refined using law-of-iterated logarithm bounds, but for simplicity we do not pursue this direction here.

Second, StrongEuler satisfies strong optimism . We remind the reader that strong optimism is not necessary to achieve gap dependent bounds, but can achieve sharper bounds for settings with simple transition dynamics like contextual bandits. The EULER algorithm, or its predecessors (e.g. Azar et al. [2017]), would also achieve-gap dependent bounds due to our analysis. Moreover, running these algorithms with the refined log( Mn k ( x, a ) /δ ) log factors would also yield log T - asymptotic regret, whereas implementing log( MT/δ ) confidence intervals may yield asymptotic regret that scales as log 2 T (see Remark B.5).

The EULER algorithm proceeds by standard optimistic value iteration, with carefully chosen exploration bonuses, and keeps track of various variance-related quantities:

The RolloutAndUpdate function (Algorithm 2 below) executes one trajectory according to the policy π k , and records all count- and variance- data regarding the relevant rewards and transition probabilities. Finally, the bonuses are are defined in Algorithm 3.

## Algorithm 1: StrongEuler

```
1 Input: 2 Initialized: For each a ∈ A x, x ′ ∈ S , n 1 ( x, a ) = 0, n 1 ( x ′ | x, a ) = 0, rsum 1 = 0, rsumsq 1 = 0, ̂ p 1 ( x, a ) = 0, ̂ Var 1 [ R ( x, a )] = 0 3 for k = 1 , 2 , . . . do 4 V k,H +1 ← 0 5 for h = H,H -1 , . . . , 1 do 6 for x ∈ S do 7 for a ∈ A do 8 Call ConstructBonuses . 9 Q k,h ( x, a ) ← min { H -h +1 , ̂ r ( x, a ) + ̂ p k,h ( x, a ) glyph[latticetop] V k,h +1 + 10 b prob k,h ( x, a ) + b rw k ( x, a ) + b str k,h ( x, a ) } 11 end 12 π k,h ( x ) := arg max a Q k,h ( x, a ), ̂ a ← π k,h ( x ) 13 V k,h ( x ) := Q k,h ( x, ̂ a ) 14 V k,h ( x ) = max { 0 , ̂ r ( x, ̂ a ) -b rw k,h ( x, ̂ a )+ ̂ p k,h ( x, ̂ a ) glyph[latticetop] V k,h +1 -b prob k,h ( x, ̂ a ) -b str k,h ( x, ̂ a ) } . 15 end 16 end 17 Call RolloutAndUpdate ( k ). 18 end
```

## F Analysis of StrongEuler : Proof of Proposition B.4

Proposition B.4 requires demonstrating a lower bound on the surplus, 0 ≤ E k,h ( x, a ), thereby establishing strong optimism, as well as an upper bound on the surplus, which we shall use to analyze the same complexity. We address strong optimism first in the next subsection, and then the upper bound in the following subsection. Throughout, we will assume that a good event A conc holds. To keep the proofs modular, the event A conc will only appear as an assumption in the supporting lemmas used in Sections F.1 and F.2. Then, in Section F.3, we formally define A conc in terms of 6 constituent events, establish P [ A conc ] ≥ 1 -δ 2 , and conclude with proofs of the supporting lemmas which rely on A conc . We remark that many of the arguments in this section are similar to those from Zanette and Brunskill [2019], with the main differences being strong optimism and the additional care paid to log-factors, necessary for log T regret. Again, recall the definition L ( u ) := √ 2 log(10 M 2 max { u, 1 } /δ ).

## F.1 Proof of Optimism

Here we establish the optimism of StrongEuler , and in particular, the bound E k,h ( x, a ) ≥ 0.

Proposition F.1. Under the good event A conc ,

- (a) StrongEuler is optimistic : π k,h ( x ) = arg max a Q k,h ( x, a ) , where Q k,h ( x, a ) ≥ Q glyph[star] h ( x, a ) for all h, x, a . In particular, V k,h ( x ) ≥ V glyph[star] h ( x ) for h ∈ [0 : H ] .
- (b) StrongEuler is strongly optimistic E k,h ( x, a ) := Q k,h ( x, a ) -r ( x, a ) -p ( x, a ) glyph[latticetop] V k,h +1 ( x ) ≥ 0 .
- (c) V k,h ≤ V π k h ≤ V glyph[star] h ≤ V k,h

## Algorithm 2: RolloutAndUpdate ( k )

```
1 Input: Global current episode k , global counts and empirical probabilities. Initialize k +1-th episode counts: n k +1 ( · , · ) ← n k ( · , · ), n k +1 ( · | · , · ) ← n k ( · | · , · ), rsum k +1 ( · , · ) ← rsum k ( · , · ), rsumsq k +1 ( · , · ) ← rsumsq k ( · , · ). 2 for h = 1 , . . . , H do 3 Observe state x h , play a h = π k,h ( x h ), recieve reward R and view next state x h +1 . 4 n k ( x h , a h ) += 1, n k ( x h ′ | x h , a h ) += 1, rsum ( x, a ) += R , rsum ( x, a ) += R 2 5 end 6 for a ∈ A , x ∈ S do 7 for x ′ ∈ S do 8 ̂ p k +1 ( x ′ | x, a ) = n k ( x h ′ | x h ,a h ) n k ( x h ,a h ) 9 end 10 r k +1 ( x, a ) = rsum k +1 n k ( x h ,a h ) , ̂ Var k +1 [ R ( x, a )] = rsumsq k +1 n k ( x h ,a h ) -r k +1 ( x, a ) 2 . 11 end 12 ,
```

Proof. The policy choice π k,h ( x ) = arg max a Q k,h ( x, a ) holds by definition of the algorithm. We now give the remainder of the argument by inducting backwards on h . For h = H +1, V k,H +1 = V k,H +1 = V glyph[star] k,H +1 = V π k h +1 = 0. Now, suppose as an inductive hypothesis that V k,h +1 ≥ V glyph[star] h +1 ≥ V π k h +1 ≥ V k,h +1 , and E k,h +1 ( x, a ) ≥ 0 for all x, a .

First, we shall show that E k,h ( x, a ) ≥ 0 for all x, a . This will establish the induction for point b . It also establishes ( a ), since then Q k,h ( x, a ) ≥ r ( x, a ) + p ( x, a ) glyph[latticetop] V k,h +1 ( x ) ≥ r ( x, a ) + p ( x, a ) glyph[latticetop] V glyph[star] h +1 = Q glyph[star] h ( x, a ), proving optimism. To this end, note that

<!-- formula-not-decoded -->

Since r ( x, a ) + p ( x, a ) glyph[latticetop] V k,h +1 ( x ) ≤ H -h +1, it suffices to show that

<!-- formula-not-decoded -->

Grouping the terms, it suffices to show that ̂ r ( x, a ) -r ( x, a ) + b rw k ( x, a ) ≥ 0, and that

<!-- formula-not-decoded -->

We lower bound ̂ r ( x, a ) -r ( x, a ) + b rw k ( x, a ) and ( ̂ p k,h ( x, a ) glyph[latticetop] -p ( x, a )) glyph[latticetop] V glyph[star] h +1 ( x ) + b prob k,h ( x, a ) by zero with the following lemma:

Lemma F.2. On the good concentration event A conc , it holds that

<!-- formula-not-decoded -->

## 1 Bonuses:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude the proof of (b) with the following lemma, which lets us bound

<!-- formula-not-decoded -->

Precisely we apply the following lemma with V 2 = V k,h +1 and V 1 = V glyph[star] h +1 :

Lemma F.3. Suppose that A prob ⊃ A conc holds, and suppose that V 1 , V 2 : S → R satisfies V k,h +1 ≤ V 1 ≤ V 2 ≤ V k,h +1 . Then,

<!-- formula-not-decoded -->

This finally establishes (b). We conclude by establishing (c). Here, we note that by definition V π k h ≤ V glyph[star] h , and V glyph[star] h ≤ V k,h as show above. Hence, it suffices to show V k,h ≤ V π k h . We begin with the inequality

<!-- formula-not-decoded -->

where the last inequality uses the bounds ( r ( x, a glyph[star] ) -̂ r ( x, a glyph[star] )) ≥ -b rw k,h ( x, a ) and ( p ( x, a glyph[star] ) glyph[latticetop] -̂ p ( x, a glyph[star] ) glyph[latticetop] ) V glyph[star] h +1 ≥ -b prob k,h ( x, a ) on A conc due to Lemma F.2, and bounds ( p ( x, a glyph[star] ) -̂ p ( x, a glyph[star] )) glyph[latticetop] ( V π k h +1 -V glyph[star] h +1 ) ≥ -b str k,h ( x, a ) by applying Lemma F.3 with V 1 = V π k h +1 and V 2 = V glyph[star] h +1 , which satisfy V k,h +1 ≤ V 1 ≤ V 2 ≤ V k,h +1 by our inductive hypothesis (namely, V k,h +1 ≥ V glyph[star] h +1 ≥ V π k h +1 ≥ V k,h +1 ). Since V π k h ( x ) ≥ 0 as well, and since V π k h +1 ≥ V k,h +1 by our inductive hypothesis, we therefore have

<!-- formula-not-decoded -->

This completes the induction.

## F.2 Proof of Surplus Bound Upper Bound

Throughout, we assume the round k is fixed, and suppress the dependence of ̂ p , ̂ Var, and ̂ r on k . We use the shorthand p = p ( x, a ) and ̂ p = ̂ p ( x, a ), where the pair ( x, a ) are clear from context.

<!-- formula-not-decoded -->

where the last line is by Lemmas F.2 and F.3. Next, we state a standard lemma that lets us swap out the empirical variance for the true variance in upper bounding b rw k ( x, a ):

<!-- formula-not-decoded -->

Next, we recall from the definition of b prob ,

<!-- formula-not-decoded -->

where we replaced n k ( x, a ) -1 by n k ( x, a ) in the deminator of one of the terms by taking advantage of the ' H ∧ '. Furthermore, we can bound

<!-- formula-not-decoded -->

We can control the difference | √ | Var p ( x,a ) [ V glyph[star] h +1 ] -√ Var p ( x,a ) [ V π k h +1 ] | using the following lemma:

Lemma F.5. Let X,Y be two real valued random variables, and let ‖·‖ p, 2 := √ E [( · ) 2 ] . Then | √ Var[ X ] -√ Var[ Y ] |≤ √ Var[ X -Y ] ≤ ‖ X -Y ‖ 2 ,p .

Proof. The inequality Var[ X -Y ] ≤ E [( X -Y ) 2 ] = ‖ X -Y ‖ 2 2 ,p follows since Var[ Z ] ≤ E [ Z 2 ] for any random variable Z . For the first inequality, we can assume WLOG that X,Y are mean zero, in which case √ Var[ X ] = ‖ X ‖ 2 ,p , and similarly for Y and X -Y . The result now follows from the fact that the norm ‖·‖ p, 2 satisfies the triangle inequality.

We shall also need the following simple fact:

<!-- formula-not-decoded -->

Since V k,h +1 ≤ V π k h +1 ≤ V glyph[star] h +1 ≤ V k,h +1 by Proposition F.1, Lemma F.5 and Fact F.6 above yield

<!-- formula-not-decoded -->

Together the with the elementary inequality, √ a + b ≤ √ a + √ b glyph[lessorsimilar] √ a + b , this in turn yields

<!-- formula-not-decoded -->

where we use the shorthand ‖ V ‖ 2 , ̂ p + p = √ ‖ V ‖ 2 2 ,p + ‖ V ‖ 2 2 , ̂ p , and where in the last, we recall that Var ( k ) h,x,a = min { Var π k h,x,a , Var glyph[star] h,x,a } = Var[ R ( x, a )] + min { Var p ( x,a ) [ V glyph[star] h +1 ] , Var p ( x,a ) [ V π k h +1 ] } . Next, substituing in b str k,h ( x, a ) := ‖ V k,h +1 -V k,h +1 ‖ 2 , ̂ p ( x,a ) √ S L ( n k ( x,a )) n k ( x,a ) + 8 3 SH L ( n k ( x,a )) n k ( x,a ) , we obtain

<!-- formula-not-decoded -->

where ( i ) uses the inequality a/b ≤ a 2 + 1 b 2 , and ( ii ) uses the facts that ‖ V ‖ 2 2 , ̂ p + p = ‖ V ‖ 2 2 ,p + ‖ V ‖ 2 2 , ̂ p and ‖ V ‖ 2 2 , ̂ p = 〈 ̂ p, V 2 〉 = 〈 ̂ p, V 2 〉 + 〈 ̂ p -p, V 2 〉 = ‖ V ‖ 2 2 , ̂ p + 〈 ̂ p -p, V 2 〉 . Lastly, inequality ( iii ) uses 0 ≤ V k,h +1 ≤ V k,h +1 ≤ H .

We continue bounding H ( p -̂ p ) glyph[latticetop] ( V k,h +1 -V k,h +1 ) in much the same way that we bounded the term in Lemma F.3 in terms of b str , with the exception that we seek a term which depends on the true transition probability p ( x, a ), and not the empirical ̂ p ( x, a ):

## Lemma F.7. Under A conc ,

<!-- formula-not-decoded -->

The proof of the above lemma is ommitted for the sake of brevity, and follows from a simplified version of the proof of Lemma F.3 where we need not pass through an empirical variance. Applying the bound in Lemma F.7, we have

<!-- formula-not-decoded -->

where the last line uses the inequality ab ≤ ( a 2 + b 2 ) / 2. Finally, combining the above with our previous bound, we arrive at

<!-- formula-not-decoded -->

From first principles, it is straightforward to show that E k,h ( x, a ) glyph[lessorsimilar] H , which implies that

<!-- formula-not-decoded -->

To conclude the proof, it remains to unravel the term ‖ V k,h +1 -V k,h +1 ‖ 2 2 ,p ( x,a ) .

## Lemma F.8. Define the term

<!-- formula-not-decoded -->

Then, we have the bound

<!-- formula-not-decoded -->

As a consequence, we can compute

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

Since H Z k ( x, a ) ≥ H ∧ SH 2 L ( n k ( x,a )) n k ( x,a ) . Hence, we can bound via (21)

<!-- formula-not-decoded -->

where we note that the summation in the expectation now begins at t = h to account for the term H ∧ SH 2 L ( n k ( x,a )) n k ( x,a ) in (21), and recal that

<!-- formula-not-decoded -->

To conclude, we recall the definitions,

<!-- formula-not-decoded -->

so that, for u ≥ 1, L ( u ) glyph[lessorsimilar] log Mu δ .

## F.3 Definition of A conc , and proofs of supporting lemmas

Before proving the lemmas above, we formally express the good event A conc as a list of constituent concentration events, and verify that it occurs with probability at least 1 -δ/ 2:

Proposition F.9. The event A conc := A rw ∩A prob ∩A val ∩A var , val ∩A var , rw occurs with probability 1 -δ/ 2 , where each of the constituent events occurs with probability at least 1 -δ/ 12 :

<!-- formula-not-decoded -->

Proof. The proof of these the first four events follows from standard applications of Bernstein's and Hoeffding's inequality, and the last two from Maurer and Pontil [2009, Theorem 10]. Similar proofs can be found in [Zanette and Brunskill, 2019, Azar et al., 2017, Dann et al., 2017]. As in those works, the only subtlety is to use the appropriate concentration inequality with respect to an appropriate filtration to attain bounds that depend on L ( n k ( x, a )), rather than on L ( T ).

Let's prove A rw as an example. We it suffices to only consider rounds for which n k ( x, a ) ≥ 1, for otherwise the bound is vacuous. Fix an action ( x, a ), and let τ i ∈ { 1 , 2 , . . . } ∪ {∞} denote the round k + 1 immediately after the i -th round k at which a pair ( x, a ) is observed at least once during the rollout, and define a sub-filtration {G i } via G i = F τ i . Then, for any given i , a martingale analogue of Bernstein's inequality yields

<!-- formula-not-decoded -->

Now fix an i ≥ 1. Since ̂ r k ( x, a ) and n k ( x, a ) are constant for k ∈ { τ i , . . . , τ i +1 -1 } , we have

<!-- formula-not-decoded -->

Applying the above with η ← 2 η/i 2 and union bounding over n , we have

<!-- formula-not-decoded -->

Since n k ( x, a ) increments by at least one for each τ i , we have i ≤ n k ( x, a ) for k ∈ { τ i , . . . , τ i +1 -1 } . Thus,

<!-- formula-not-decoded -->

Lastly, since for any k , there always exist some i for which k ∈ { τ i , . . . , τ i +1 -1 } , we have

<!-- formula-not-decoded -->

We then conclude by union bounding over SA , and letting η = δ/ 12 SA , yielding the following log factor: log(48 SAn k ( x, a ) 2 /δ ) ≤ L ( n k ( x, a )), where we recall L ( u ) = √ 2 log(10 M 2 max { u, 1 } /δ ) for M = SAH . The proof for A prob is analogous, the proof for A val requires union bounding over states x ′ , incuring a log factor log(4 S 2 An k ( x, a ) 2 /δ ) ≤ L ( n k ( x, a )).

Proof of Lemma F.2. We prove the bound | ( ̂ p ( x, a ) -p ( x, a )) glyph[latticetop] V glyph[star] h +1 |≤ b prob k,h ( x, a ); the analogous bounds for rewards is similar. Note that since ̂ p ( x, a ) glyph[latticetop] V glyph[star] h +1 ∈ [0 , H ] and p ( x, a ) glyph[latticetop] V glyph[star] h +1 ∈ [0 , H ], | ( ̂ p ( x, a ) -p ( x, a )) glyph[latticetop] V glyph[star] h +1 |∈ [0 , H ]. This takes care of the first ' H ∧ ' in b prob k,h ( x, a ). Next, on A val

and A var , val ,

<!-- formula-not-decoded -->

Lastly, by Lemma F.5, we have the bound

<!-- formula-not-decoded -->

Proof of Lemma F.3. Summing up the condition of event A prob over states x ′ ∈ S , and then applying event A var , prob to control | p ( x ′ | x, a ) -̂ p ( x ′ | x, a ) | :

<!-- formula-not-decoded -->

where ( i ) uses p ( x ′ | x, a )(1 -p ( x ′ | x, a )) ≤ p ( x ′ | x, a ), ( ii ) uses event A var , prob , and where bound in the bracket is because there | V k,t +1 ( x ′ ) -V glyph[star] h +1 ( x ′ ) |≤ H by Proposition F.1 part (b), and there are

at most S terms in the summation. To bound the first term, we have

<!-- formula-not-decoded -->

( i ) bounds uses Cauchy-Schwartz, and ( ii ) uses Proposition F.1 part (c) to bound ‖ V 2 -V 1 ‖ 2 ,p ≤ ‖ V k,h +1 -V k,h +1 ‖ 2 , ̂ p for V k,h +1 ≤ V 1 ≤ V 2 ≤ V k,h +1 , in light of Fact F.6.

Proof of Lemma F.4. Under the event A conc we have b rw k ( x, a )

<!-- formula-not-decoded -->

where in the second-to-last inequality, we used the event A var , rw to control ∣ ∣ ∣ ∣ √ Var[ R ( x, a )] -√ ̂ Var[ R ( x, a )] ∣ ∣ ∣ ∣ glyph[lessorsimilar] √ L ( n k ( x,a )) n k ( x,a ) -1 .

<!-- formula-not-decoded -->

where the last line uses the fact that p ( x, a ) glyph[latticetop] ( V k,h +1 -V k,h +1 ) ≥ 0 on A conc (Proposition F.1, part (c)). Unfolding the above expression inductively, we then find that

<!-- formula-not-decoded -->

To conclude, it suffices to check that H ∧ { 2 b rw k ( x, a ) + 2 b prob k,h ( x, a ) + 2 b str k,h ( x, a ) } glyph[lessorsimilar] √ Z k ( x, a ), for any triple x, a, h . To check that this bound holds, we have from (20) that

<!-- formula-not-decoded -->

where we recall the notation ‖ V ‖ 2 , ̂ p + p = √ ‖ V ‖ 2 2 ,p + ‖ V ‖ 2 2 , ̂ p , and thus the final bound holds since Var glyph[star] t,x,a ≤ H implying that ‖ V k,t +1 -V k,t +1 ‖ 2 2 , ( ̂ p k + p )( x t ,a t ) ≤ 4 H for 0 ≤ V k,t +1 ≤ V k,t +1 ≤ H . Consolidating the terms, we have 2 b rw k ( x t , a t ) + 2 b prob k,t ( x t , a t ) + 2 b str k,t ( x t , a t ) is at most glyph[lessorsimilar] ( H √ S L ( n k ( x,a )) n k ( x,a ) + SH L ( n k ( x,a )) n k ( x,a ) ) , and thus H ∧ 2 b rw k ( x t , a t ) + 2 b prob k,t ( x t , a t ) + 2 b str k,t ( x t , a t ) is glyph[lessorsimilar] H ∧ ( H √ S L ( n k ( x,a )) n k ( x,a ) + SH L ( n k ( x,a )) n k ( x,a ) ) := √ Z k ( x, a ) .

## Part III

## Lower Bounds

## G Min-Gap Lower Bound for Optimistic Algorithms (Theorem 2.3)

## G.1 Formal Statement

We begin a formal version of the lower bound, Theorem 2.3.

Theorem G.1. Let c 1 , c 2 , c 3 be absolute constants that may depend on the constants defined in Section G.2. Let Alg denote an algorithm in the class described in Section G.2 run with confidence parameter δ ∈ (0 , 1 / 8) . For any S ≥ 1 and glyph[epsilon1] ≤ 1 / glyph[ceilingleft] c 1 S log( S/δ )) glyph[ceilingright] , fix any MDP in the class described in Section G.3 so that |S| = 2 S + 1 , |A| = 2 , H = 2 , and exactly one state has a suboptimality gap of gap min = glyph[epsilon1] and all other states have a minimum sub-optimality gap of at least 1 / 2 . Then ∑ h,x,a : gap h ( x,a ) &gt; 0 1 gap h ( x,a ) glyph[lessorsimilar] S + 1 gap min but Alg for all sufficiently large K suffers a regret

<!-- formula-not-decoded -->

with probability at least 1 -c 2 Sglyph[epsilon1] -2 log(1 /δ ) e -c 3 S -3 δ .

In particular, for any glyph[epsilon1] ∈ (0 , c ) for some constant c , if log( glyph[epsilon1] -1 /δ ) glyph[lessorsimilar] S glyph[lessorsimilar] glyph[epsilon1] -1 / log( glyph[epsilon1] -1 /δ ) then the above regret lower bound holds with probability 1 -O ( δ ).

## G.2 Algorithm Class

Optimistic Q-functions: We consider algorithms where the optimistic Q-function is constructed as follows: given a reward bonus function b rw k ( x, a ) ≥ 0 and an additional nonnegative stagedependent bonus b k,h ( x, a ), and empirical estimates ̂ r k ( x, a ) of the reward and ̂ p k ( x, a ) = ( ̂ p ( x ′ | x, a )) of the transition probabilities. We set the Q-function at stage H as Q k,H ( x, a ) = ̂ r k ( x, a )+ b rw k ( x, a ), where ̂ r k ( x, a ), and for h ∈ { 1 , . . . , H -1 } ,

<!-- formula-not-decoded -->

Lastly, suppose that b rw k ( x, a ) depends only on rewards collected when the state ( x, a ) is visited.

Note that this template subsumes the model-based approaches of Azar et al. [2017], Zanette and Brunskill [2019], Dann et al. [2018], and if b rw ( x, a ) is made to be time dependent, captures the approach of Dann et al. [2017] as well. For the specific lower bound instance we consider, each stage x ∈ S can only be visited at a single stages h ∈ [2], so b rw may be chosen to be time dependent without loss of generality. In order to capture the 'model-free' methods based on Q-learning due to Jin et al. [2018], we can instead mandate that

<!-- formula-not-decoded -->

where ̂ ( p ( x, a ) glyph[latticetop] V glyph[star] h +1 ) is a generalized estimate of p ( x, a ) glyph[latticetop] V glyph[star] h +1 , and such that ̂ ( p ( x, a ) glyph[latticetop] V glyph[star] h +1 ) is nonnegative. In Lemma 4.2 in Jin et al. [2018], one can see that we can take

<!-- formula-not-decoded -->

where k s is the round at which ( x, a ) was selected for the s -th time, α s is an appropriate weight, ̂ P k s ,h ( x, a ) is the empirical probability estimate ̂ P k s ( x, a )[ x ′ ] = I ( x ′ = x k s ,h +1 ) equal to indicator at the state x k s ,h +1 visited after playing a at x at round k s , and where V k s ,h +1 is an optimistic estimate of V glyph[star] h +1 at round k s .

For simplicity, we shall work with the model based formulation (24), though the lower bound can be extended to this more general class.

Confidence Interval Assumptions: Our class of algorithms takes in a confidence parameter δ ∈ (0 , 1 / 8). We shall also assume that there exists consants c bon , c bon such that, when the algorithm is run with parameter δ , the bonuses b rw and b rw k satisfy 8

<!-- formula-not-decoded -->

We further assume that b rw ( x, a ) is δ -correct, in the sense that,

<!-- formula-not-decoded -->

Lastly, we shall assume that the optimistic overestimate is consistent in the sense that for any MDP M with optimal value V ∗ , M 0 , for any glyph[epsilon1],δ &gt; 0 there exists a function f M such that

<!-- formula-not-decoded -->

Intuitively, this condition states that with high probability, the optimistic over-estimate of the value estimate approaches the expected reward under the optimal policy. Note that this does not assume uniform convergence of the entire value function itself, just the expected reward with respect to the initial state distribution p 0 on the optimal policy.

Remark G.1. Note that we do not require that our algorithm's confidence intervals are 'inflated', in the sense that, with high probability, ̂ r k ( x, a ) + b rw k ( x, a ) -r ( x, a ) ≥ c b rw k ( x, a ), for a universal constant c . With this stronger assumption, we note that the proof of the lower bound can be simplified, and some restrictions on S, glyph[epsilon1] removed. In the interest of generality, we refrain from making this assumption.

## G.3 Formal Lower Bound Instance

Consider the following simple game with H = 2, A = {-1 , +1 } and S = {-S, . . . , -1 , 0 , 1 , . . . , S } = S -∪ { 0 } ∪ S + , where S -= -[ S ] and S + = [ S ] (note |S| = 2 S + 1). The game always begins at state x 1 = 0 with two available actions, a ∈ {-1 , +1 } . Then, x 2 | ( x 1 = 0 , a 1 = +1) unif ∼ S + , and x 2 | ( x 1 = 0 , a 1 = -1) unif ∼ S -. Lastly, let D denote any symmetric distribution on [ -1 , 1] with Ω(1) variance. For glyph[epsilon1] ∈ (0 , 1 / 8), we formally define the reward distributions

<!-- formula-not-decoded -->

It is straightforward to verify the following fact

8 The quantity Var[ R ( x, a )] below can also be replaced with an empirical variance, but we choose the true variance for simplicity.

glyph[negationslash]

Fact G.2. The optimal action is always a = 1 . Moreover, gap 1 (0 , -1) = gap min = glyph[epsilon1] , whereas gap 2 ( x, -1) ≥ 1 2 for x = 0 .

In other words, all the gaps for suboptimal arms are Ω(1), except for the gap at state x = 0, which means for this instance with H = 2 and A = 2 we have ∑ x,a,h 1 gap h ( x,a ) glyph[equalorsimilar] S + 1 glyph[epsilon1] . Nevertheless, we shall show that any algorithm in the class above suffers regret

<!-- formula-not-decoded -->

## G.4 The Lower Bound:

The Lower Bound: We first show that the optimistic Q-function relative to the optimal value at (0 , 1) decays at a rate of at least √ S log(1 /δ ) /n k (0 , 1). This will ultimately lead to incurring a regret of S log(1 /δ ) glyph[epsilon1] , despite the fact that all but one of the Q-function gaps are Ω(1).

Proposition G.3. Let Alg denote an algorithm in the class described in Section G.2 run with confidence parameter δ ∈ (0 , 1 / 8) . Then there exists constants c 1 , c 2 , c 3 , depending only on the constants described in Section G.2, such that the following holds. For any glyph[epsilon1] ≤ 1 / glyph[ceilingleft] c 1 S log( S/δ )) glyph[ceilingright] and for N = glyph[floorleft] c 2 S log(1 /δ ) /glyph[epsilon1] 2 glyph[floorright] ,

<!-- formula-not-decoded -->

We now use Proposition G.3 to prove Theorem G.1. Note that V glyph[star] 1 (0) = V glyph[star] 0 . By assumption, with probability 1 -δ , V 0 ≤ V glyph[star] 0 + η after f ( η, δ ) rounds. Fix an appropriate glyph[epsilon1] and N in Proposition G.3 and let K ≥ f M ( glyph[epsilon1]/ 2 , δ ) + N . If n K (0 , -1) &gt; N times, then we have

<!-- formula-not-decoded -->

and the theorem is proved. Thus, suppose not so that n K (0 , -1) ≤ N . Then by Proposition G.3 we have with high probability that

<!-- formula-not-decoded -->

However, by assumption K ≥ f M ( glyph[epsilon1]/ 2 , δ ) which means that on an event that holds with probability at least 1 -δ , we have V 0 -V glyph[star] 1 (0) = max a ∈{-1 , 1 } Q k, 1 (0 , a ) -V glyph[star] 1 (0) ≤ glyph[epsilon1]/ 2, a contradiction.

## G.4.1 Proof of Proposition G.3

Throughout, we will use upper case C 1 , C 2 , . . . to do denote possibly changing numerical constants that depend on the the constants in the definition of Alg , as set in Section G.2. The lower cast constants c 1 , c 2 will be coincide with those in Proposition G.3.

Since Q glyph[star] 1 (0 , 1) = 1 2 + glyph[epsilon1] , it suffices to show that

<!-- formula-not-decoded -->

Fix an n 0 = glyph[ceilingleft] c 1 S/ log( S/δ ) glyph[ceilingright] for a constant c 1 be specified later, and let

<!-- formula-not-decoded -->

By the optimism assumption, E opt holds with probability at least 1 -δ . First we verify that Q k, 1 (0 , -1) -1 2 ≥ 2 glyph[epsilon1] for 0 ≤ n k (0 , -1) ≤ n 0 , provided that glyph[epsilon1] is sufficiently small:

Claim G.4. Suppose that glyph[epsilon1] ≤ c bon 2 n 0 . Then, with probability 1 -δ , Q k, 1 (0 , -1) -1 2 ≥ 2 glyph[epsilon1] whenever 0 ≤ n k (0 , -1) ≤ n 0 :

Proof. We have that

<!-- formula-not-decoded -->

Since p ( x | 0 , -1) = 0 for x / ∈ S -, the empirical probability ̂ p ( x | 0 , -1) is also 0, and thus

<!-- formula-not-decoded -->

where the first equality and first inequality use ∑ x ′ ∈S -̂ p ( x | 0 , -1) = 1, and the second uses the optimistic event E opt to show that V k, 2 ( x ′ ) ≥ ̂ r k ( x ′ , 1) + b rw k ( x ′ , 1) ≥ r ( x ′ , 1) = 1 2 for x ′ ∈ S -. Using the assumption that b k, 1 ( x, a ) ≥ c bon 1 ∨ n k ( x,a ) , we see that if n k ( x, a ) ≤ n 0 and glyph[epsilon1] ≤ c bon 2 n 0 , then b k, 1 ( x, a ) ≥ c bon n 0 ≥ 2 glyph[epsilon1] , as needed.

Now, we turn to the case where n k ( x, a ) ∈ { n 0 , . . . , N } for some N = glyph[floorleft] c 2 S log(1 /δ ) /glyph[epsilon1] 2 glyph[floorright] . It light of (25), it suffices to show that for n k ≤ N ,

<!-- formula-not-decoded -->

By the definition of our algorithm class, the optimistic Q-function at stage h = 2 and pair ( x, a ) depend only at rewards collected at ( x, a ), and the construction of our MDP, pairs ( x, a ) for x ∈ S -are only accessible by playing (0 , -1). Hence, to analyze Q k, 1 (0 , -1), for n 0 ≤ n k (0 , -1) ≤ N , it suffices to prove our described lower bound on Q k, 1 (0 , -1) in the simplified game, where at each round k = 1 , 2 , . . . , the algorithm always selects (0 , -1), and show that for this algorithm

<!-- formula-not-decoded -->

Turning our attention to this simplified game,for x ∈ S -let n k ( x ) denote the number of times x has been visited up to round k , and recall n k ( x, a ) is the number of times action a is played at stage s . Further, set

<!-- formula-not-decoded -->

We now make a couple of observations

- (a) The vector ( n k ( x )) x ∈S -is a uniform multinomial on the states in S -.
- (b) Conditioned on ( n k ( x )) x ∈S -, we can see that the values of V k, 2 ( x ) are independent, because for each x ∈ S -, the game decouples into n k ( x ) rounds of a two arm bandit game on actions a ∈ {-1 , 1 } .

Using these observations, we prove the following claim:

Claim G.5. There exists constants C 1 , C 2 such that for any x ∈ S -, if δ ≤ 1 / 8 and n k ( x ) ≥ C 1 log( M/δ ) , then conditioned on the history ( n j ( x ′ )) x ′ ∈S -,j ≥ 1 , the following event holds with probability at least 1 / 4 :

<!-- formula-not-decoded -->

and the events {E ∆ j ( x ) : x ∈ S -} are mutually independent (again, given ( n j ( x ′ )) x ′ ∈S -,j ≥ 1 ).

Therefore, on the optimistic event E opt , where { ∆( k, x ) ≥ 0 } , we can lower bound (again, in the simplified game where we always select action (0 , -1)),

<!-- formula-not-decoded -->

where ( i ) uses the fact that for x ∈ S -is only accessible through (0 , -1), and that (0 , -1) is always selected in the simplified game. Next, observe that in the simplified game, n k ( x ) = k/S , so that if n 0 /S ≥ C 3 log(1 /δ ) for some constant C 3 , it holds by an argument similar to Lemma B.7 that with probability 1 -δ , the event E 1 := {∀ x ∈ S -, ∀ k ≥ n 0 , n k ( x ) ≥ n k ( x ) / 4 = k/ 4 S } holds, yielding

<!-- formula-not-decoded -->

Finally, if in addition n 0 / 4 S ≥ C 1 log(1 /δ ), where C 1 is the constant from claim G.5, then on E 1 , it holds that for k ≥ n 0 , n k ( x ) ≥ C 1 log(1 /δ ). We then set the constant c 1 so that n 0 /S ≥ C 3 log(1 /δ ) and n 0 / 4 S ≥ C 1 log(1 /δ ) hold.

Lastly, since (a) E 1 is measurable with respect to the counts ( n j ( x ′ )) x ′ ∈S -,j ≥ 1 , (b) since E ∆ k ( x ) are independent given these counts, and (c) E [ I ( E ∆ k )] ≥ 1 / 4, a Chernoff bound shows that for k ≥ n 0 , the event E 2 ( k ) := { ( 1 S ∑ x ∈S -I ( E ∆ k ) ) ≥ 1 / 8 } holds with probability at least e -C 5 S conditioned on E 1 . Hence, on E opt ∩ E 1 ∩ ⋃ N k = n 0 E 2 ( k ), we have

<!-- formula-not-decoded -->

Hence, if N ≤ c 2 S log(1 /δ ) glyph[epsilon1] 2 for some constant c 2 , we see that ∆ 0 ( k ) ≥ 2 glyph[epsilon1] for all k ∈ { n 0 , . . . , N } .

Lastly, we see that

<!-- formula-not-decoded -->

Translating to the non-simplified game, we have therefore established that

<!-- formula-not-decoded -->

Combining with the additional probability of error δ for the case n k (0 , -1) ≤ n 0 concludes the proof.

## G.5 Proof of Claim G.5

We observe that conditioned on the vector ( n j ( x ′ )) x ′ ∈S -,j ≥ 1 , the games at states x and round k are equivalent to S independent two-arm bandit games with n k ( x ) rounds. Note moreover that ∆( x, k ) = V k, 2 ( x ) -1 2 ≥ b rw k ( x, 1)+ ̂ r k ( x, 1) -1 2 . Hence, restricting to a single state x (and dropping the dependence on x for simplicity), it suffices to show that for k rounds of an appropriate two-arm bandit game with a ∈ {-1 , 1 } with empirical rewards ̂ r k ( a ) and bonuses b rw k ( a ), R ( -1) = 0 and R (1) ∼ 1 2 + 1 4 D , that

<!-- formula-not-decoded -->

where we have dropped the dependence on x for simplicity. Throughout, we will also use the notation C 1 , C 2 , C 3 to denote constants specific to the proof of Claim G.5, and reserve C 1 , C 2 for the constants in the claim statement.

If δ ≤ 1 / 8, then a standard argument shows that for some constant C 1 (depending on c bon ), n k ( -1) ≤ C 1 log( S/δ ). Indeed, define the event E 0 := {∀ k ≥ 1 : b rw k (1) + ̂ r k (1) ≥ r (1) = 1 2 } ; by assumption on our confidence intervals, complement of this event occurs with probability at most δ ≤ 1 / 8. Note also that on E 0 , since R ( -1) = 0 with probability 1, it holds that for any j ≤ k with n j ( -1) ≥ C 1 log( S/δ )

<!-- formula-not-decoded -->

where in ( i ) we used the definition of the confidence interval with M glyph[lessorsimilar] S , and in ( ii ) we used n j ( -1) ≥ C 1 log( S/δ ) for an appropriately tuned constant C 1 . Since a j := arg max a ̂ r j ( a ) + b rw j ( a ), we have a j = 1. This implies that n k ( -1) ≤ max j ≥ 1 n j ( -1) ≤ C 1 log( M/δ ).

Next, set k 0 = C 1 log( M/δ ). We wish to show that for k ≥ k 0 ,

<!-- formula-not-decoded -->

There are two technical challenges: first, the confidence interval b rw k (1) might be nearly tight, so that we cannot show that with high probability, ̂ r k (1) + b rw k (1) glyph[greaterorsimilar] b rw k (1). Second, because the

algorithm adaptively chooses to sample actions a ∈ {-1 , 1 } , ̂ r k (1) does not have the distribution of n k (1) i.i.d. samples from R (1).

We can get around this as follows. We can imagine all rewards sampled from action 1 as being drawn at the start of the game, and constituting a sequence R (1) (1) , R (2) (1) , . . . and so on. Then, ̂ r k (1) is the average of the samples 1 , . . . , n k (1), where n k (1) ≤ k . Therefore

<!-- formula-not-decoded -->

where the last line uses n k (1) + n k ( -1) = k .

Now consider the event E 1 ( δ ) := { n k ( -1) ≤ k 0 } , where we recall k 0 = C 1 log( M/δ ) was our 1 -δ -probability upper bound on n k ( -1). On E 1 ( δ ), n k ( -1) = j for some j ∈ { 0 , 1 , . . . , k 0 } , and we can lower bound the above expression by

<!-- formula-not-decoded -->

Observe now that we have lower bounded n k (1)( ̂ r k (1) -1 2 ) in terms of quantities depending only on the i.i.d. reward sequence ( R ( i ) (1)), and not on the quantities n k ( -1) , n k (1).

Moreover, a standard maximal inequality implies that the following event E 2 ( δ ) holds for an appropriate constant C 2 with probability 1 -δ :

<!-- formula-not-decoded -->

Lastly, since R ( i ) is symmetric, we have that the following event E 3 holds with probability 1 / 2:

<!-- formula-not-decoded -->

Hence, on E 1 ( δ ) ∩ E 2 ( δ ) ∩ E 3 ,

<!-- formula-not-decoded -->

If we further assume that k ≥ 2 C 1 log( M/δ ), then n k ( -1) ≤ k 0 ≤ k/ 2, so that E 1 ( δ ) implies n k (1) ≥ k/ 2. Dividing both sides of the above by k and bringing 1 /k into the square root yields (again on E 1 ( δ ) ∩ E 2 ( δ ) ∩ E 3 )

<!-- formula-not-decoded -->

Moreover, by the lower bound assumption on b rw and the fact that R (1) has Ω(1) variance, there exists some constant C 3 such that

<!-- formula-not-decoded -->

where again we use n k (1) ≤ k . Combining with (28), we have on E 1 ( δ ) ∩ E 2 ( δ ) ∩ E 3 that

<!-- formula-not-decoded -->

Hence, if k 0 /k ≤ ( C 3 / C 2 ) 2 , or equivalently if k ≥ C 1 ( C 3 / C 2 ) -2 log( M/δ ), then

<!-- formula-not-decoded -->

on the event E 1 ( δ ) ∩ E 2 ( δ ) ∩ E 3 . Lastly, for δ ≤ 1 / 8, we note P [ E 1 ( δ ) ∩ E 2 ( δ ) ∩ E 3 ] ≥ 1 2 -2 δ ≥ 1 / 4. Recalling our earlier condition k ≥ 2 C 1 log( M/δ ), the claim now holds with by setting the constant C 1 in the claim statement to be C 1 max { 2 , ( C 3 / C 2 ) -2 } , and C 2 to be C 3 2 √ 2 .

## H Information Theoretic Lower Bound (Proposition 2.2)

In this section we construct give a proof of the information theoretic lower bound Proposition 2.2, as well as a non-asymptotic bound that holds even for non-uniformly good algorithms.

## H.1 Construction of the hard instance

Our construction mirrors the lower bounds due to Dann and Brunskill [2015], but with specific and non-uniform gaps. We define M as an MDP on state space S = [ S +2], with actions A = [ A ], and horizon [ H ]. We will first state the construction for H ≥ 2, and then remark on the modification for H = 1 at the end of the section. For a ∈ [ A ] , x ∈ [ S ], we set

<!-- formula-not-decoded -->

Furthermore, we set the initial state to have the distribution x 1 unif ∼ [ S ], and set

<!-- formula-not-decoded -->

Finally, the rewards are set deterministically as

<!-- formula-not-decoded -->

We may then verify that V glyph[star] h ( S +1) = ( H -h +1) and V glyph[star] h ( S +1) = ( h -H +1) / 2, which implies that that for x ∈ [ S ],

<!-- formula-not-decoded -->

and in particular that gap 1 ( x, a ) = ∆ x,a . For H = 1, the construction is modified so that S = [ S ], and

<!-- formula-not-decoded -->

Then, we see that gap 1 ( x, a ) = ∆ x,a . In what follows, we will adress the H ≥ 2 case; the case H = 1 will follow from similar, but simpler arguments.

## H.2 Regret Lower Bound Decomposition

We can now lower bound the expected regret as

<!-- formula-not-decoded -->

where inequality ( i ) follows since V π k 1 ( x ) = Q π k 1 ( x, π k, 1 ( x )) ≤ Q glyph[star] 1 ( x, π k, 1 ( x )). We now show that for all sufficiently large K ≥ K 0 ( M ), any uniformly correct algorithm must have

<!-- formula-not-decoded -->

which concludes the proof since

<!-- formula-not-decoded -->

We further note that this argument can also show that, for all K sufficiently large and all h ∈ [ H -1]

<!-- formula-not-decoded -->

as well.

## H.3 Proof of Equation (30)

Throughout, we fix a state x ∈ [ S ], and an action a : gap 1 ( x, a ) &gt; 0. We shall further introduce the shorhand

<!-- formula-not-decoded -->

where the bound on ∆ x,a follows from ∆ x,a ∈ (0 , H/ 8).

To lower bound Equation (30), we follow steps analogues to standard information theoretic lower bounds. Our exposition will follow Garivier et al. [2018]. First, we state a lemma which is the MDP analogue of Garivier et al. [2018, Equation (6)]. Its proof is analogous, and omitted for the sake of brevity:

Lemma H.1. Let M = ( S , A , H, r, p M , p 0 , R M ) and M ′ = ( S , A , H, r, p M ′ , p 0 , R M ′ ) denote two episodic MDPs with the same state space S , action space A and horizon h , and initial state distribution p 0 . For any ( x, a ) ∈ S × A , let ν M ( x, a ) denote the law of the joint distribution of ( X ′ , R ) where X ′ ∼ p M ( ·| x, a ) and R ∼ R M ( x, a ) ; define the law ν M ( x, a ) analogously. Finally, fix a horizon K ≥ 1 , and let F K denote the filtration generated by all rollouts up to episode K . Then, for any F K -measurable random variable Z ∈ [0 , 1] ,

<!-- formula-not-decoded -->

where kl( x, y ) = x log x y + (1 -x ) log 1 -x 1 -y denotes the binary KL-divergence, and KL( · , · ) denotes the KL-divergence between two probability laws.

We apply the above lemma as follows. For our fixed pair ( x, a ), define an alternate M ′ to be the MDP which coincides with M except that

<!-- formula-not-decoded -->

By construction, M and M ′ differ only at their law at ( x, a ). Thus,

<!-- formula-not-decoded -->

We the following lower bound controls the KL divergence between the laws ν M ( x, a ) , ν M ′ ( x, a ):

Claim H.2. There exists a universal constant c such that

<!-- formula-not-decoded -->

Proof. At ( x, a ), R ( x, a ) = 0 with probability under both M , M ′ . Moreover, recall that under M , ( x, a ) transition to state S +1 with probability 3 4 -∆ x,a , and to S +2 with probability 1 -( 3 4 -∆ x,a , ). On the other hand, M ′ transtion to S +1 with probability 3 4 + η , and S +2 with probability 1 -( 3 4 + η ). Consequently both laws are equivalent to Bernoulli distributions with parameters 3 4 -∆ x,a and 3 4 + η , respectively. Since kl( x, y ) is precisely KL(Bernoulli( x ) , Bernoulli( y )) for x, y ∈ (0 , 1),

<!-- formula-not-decoded -->

Lastly, set x = 3 4 -∆ x,a and y = 3 4 +min { 7 8 , ∆ x,a } .We y -x ≤ 2 ∆ x,a , and by assumption on ∆ x,a ≤ 1 / 2, Thus, 1 / 4 ≤ x ≤ y ≤ 7 / 8. Hence, a standard Taylor expansion (e.g. Simchowitz et al. [2016, Lemma E.1]) shows that there exists a universal constant c such that kl( x, y ) ≤ c ( x -y ) 2 ≤ 4 c ∆ 2 x,a , as needed.

As a consequence, we see that for any F K -measurable Z ∈ [0 , 1], we find

<!-- formula-not-decoded -->

where the last inequality uses that ∆ x,a glyph[lessorsimilar] ∆ x,a /H .

To conclude, it suffices to exhibit a random variable Z K such that, for K sufficiently large,

<!-- formula-not-decoded -->

To this end, consider Z K = Sn K ( x,a ) K . Note that since x is only visited with probability at most 1 /S at stage h = 1, and with probability 0 for stages h ≥ 2, we have

<!-- formula-not-decoded -->

which implies that, Z K ∈ [0 , 1] with probability one. Moreover, note that by an argument similar to that of (29), that under the MDP M ′ , glyph[negationslash]

<!-- formula-not-decoded -->

Hence, if Alg is α -uniformly good, then there existsa constant C M ′ such that

<!-- formula-not-decoded -->

By the same token, there exists a constant C M such that

<!-- formula-not-decoded -->

which implies that E M [ Z K ] ≤ SC M K α -1 gap 1 ( x,a ) . Furthermore, by Garivier et al. [2018, Inequality (11)], it holds that

<!-- formula-not-decoded -->

which implies that for K sufficiently large,

<!-- formula-not-decoded -->