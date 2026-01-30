## Beyond Value-Function Gaps: Improved Instance-Dependent Regret Bounds for Episodic Reinforcement Learning

Chris Dann Google Research chrisdann@google.com

Mehryar Mohri Courant Institute and Google Research mohri@google.com

## Abstract

We provide improved gap-dependent regret bounds for reinforcement learning in finite episodic Markov decision processes. Compared to prior work, our bounds depend on alternative definitions of gaps. These definitions are based on the insight that, in order to achieve a favorable regret, an algorithm does not need to learn how to behave optimally in states that are not reached by an optimal policy. We prove tighter upper regret bounds for optimistic algorithms and accompany them with new information-theoretic lower bounds for a large class of MDPs. Our results show that optimistic algorithms can not achieve the information-theoretic lower bounds even in deterministic MDPs unless there is a unique optimal policy.

## 1 Introduction

Reinforcement Learning (RL) is a general scenario where agents interact with the environment to achieve some goal. The environment and an agent's interactions are typically modeled as a Markov decision process (MDP) [29], which can represent a rich variety of tasks. But, for which MDPs can an agent or an RL algorithm succeed? This requires a theoretical analysis of the complexity of an MDP. This paper studies this question in the tabular episodic setting, where an agent interacts with the environment in episodes of fixed length H and where the size of the state and action space is finite ( S and A respectively).

While the performance of RL algorithms in tabular Markov decision processes has been the subject of many studies in the past [e.g. 11, 22, 28, 7, 4, 20, 34, 6], the vast majority of existing analyses focuses on worst-case problem-independent regret bounds, which only take into account the size of the MDP, the horizon H and the number of episodes K .

Recently, however, some significant progress has been achieved towards deriving more optimistic (problem-dependent) guarantees. This includes more refined regret bounds for the tabular episodic setting that depend on structural properties of the specific MDP considered [30, 25, 21, 13, 17]. Motivated by instance-dependent analyses in multi-armed bandits [24], these analyses derive gapdependent regret-bounds of the form O ( ∑ ( s,a ) ∈S×A H log( K ) gap( s,a ) ) , where the sum is over state-actions pairs ( s, a ) and where the gap notion is defined as the difference of the optimal value function V ∗ of the Bellman optimal policy π ∗ and the Q -function of π ∗ at a sub-optimal action: gap( s, a ) =

∗ Author was at Johns Hopkins University during part of this work.

Teodor V. Marinov ∗ Google Research tvmarinov@google.com

Julian Zimmert Google Research zimmert@google.com

| Value-function gap (prior)                                                                     | Return gap (ours)                                                                   |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| O (∑ s,a H log( K ) gap( s,a ) ) Ω ( ∑ s,a : s ∈ π ∗ log( K ) gap( s,a ) ) gap( s 1 ,a 2 ) = c | O (∑ s,a log( K ) gap( s,a ) ) Ω (∑ s,a log( K ) H gap( s,a ) )                     |
| gap( s 2 ,a 4 ) = glyph[epsilon1] O ( SH log( K ) glyph[epsilon1] )                            | gap( s 1 ,a 2 ) = c gap( s 2 ,a 4 ) = c + glyph[epsilon1] H ≈ c O ( SH log( K ) c ) |

Figure 1: Comparison of our contributions in MDPs with deterministic transitions. Bounds only include the main terms and all sums over ( s, a ) are understood to only include terms where the respective gap is nonzero. gap is our alternative return gap definition introduced later (Definition 3.1).

<!-- image -->

V ∗ ( s ) -Q ∗ ( s, a ) . We will refer to this gap definition as value-function gap in the following. We note that a similar notion of gap has been used in the infinite horizon setting to achieve instance-dependent bounds [1, 31, 2, 12, 27], however, a strong assumption about irreducibility of the MDP is required.

While regret bounds based on these value function gaps generalize the bounds available in the multiarmed bandit setting, we argue that they have a major limitation. The bound at each state-action pair depends only on the gap at the pair and treats all state-action pairs equally, ignoring their topological ordering in the MDP. This can have a major impact on the derived bound. In this paper, we address this issue and formalize the following key observation about the difficulty of RL in an episodic MDP through improved instance-dependent regret bounds:

Learning a policy with optimal return does not require an RL agent to distinguish between actions with similar outcomes (small value-function gap) in states that can only be reached by taking highly suboptimal actions (large value-function gap).

To illustrate this insight, consider autonomous driving, where each episode corresponds to driving from a start to a destination. If the RL agent decides to run a red light on a crowded intersection, then a car crash is inevitable. Even though the agent could slightly affect the severity of the car crash by steering, this effect is small and, hence, a good RL agent does not need to learn how to best steer after running a red light. Instead, it would only need a few samples to learn to obey the traffic light in the first place as the action of disregarding a red light has a very large value-function gap.

To understand how this observation translates into regret bounds, consider the toy example in Figure 1. This MDP has deterministic transitions and only terminal rewards with c glyph[greatermuch] glyph[epsilon1] &gt; 0 . There are two decision points, s 1 and s 2 , with two actions each, and all other states have a single action. There are three policies which govern the regret bounds: π ∗ (red path) which takes action a 1 in state s 1 ; π 1 which takes action a 2 at s 1 and a 3 at s 2 (blue path); and π 2 which takes action a 2 at s 1 and a 4 at s 2 (green path). Since π ∗ follows the red path, it never reaches s 2 and achieves optimal return c + glyph[epsilon1] , while π 1 and π 2 are both suboptimal with return glyph[epsilon1] and 0 respectively. Existing value-function gaps evaluate to gap( s 1 , a 2 ) = c and gap( s 2 , a 4 ) = glyph[epsilon1] which yields a regret bound of order H log( K )(1 /c +1 /glyph[epsilon1] ) . The idea behind these bounds is to capture the necessary number of episodes to distinguish the value of the optimal policy π ∗ from the value of any other sub-optimal policy on all states . However, since π ∗ will never reach s 2 it is not necessary to distinguish it from any other policy at s 2 . A good algorithm only needs to determine that a 2 is sub-optimal in s 1 , which eliminates both π 1 and π 2 as optimal policies after only log( K ) /c 2 episodes. This suggests a regret of order O (log( K ) /c ) . The bounds presented in this paper achieve this rate up to factors of H by replacing the gaps at every state-action pair with the average of all gaps along certain paths containing the state action pair. We call these averaged gaps return gaps . The return gap at ( s, a ) is denoted as gap( s, a ) . Our new bounds replace gap( s 2 , a 4 ) = glyph[epsilon1] by gap( s 2 , a 4 ) ≈ 1 2 gap( s 1 , a 2 ) + 1 2 gap( s 2 , a 4 ) = Ω( c ) . Notice that glyph[epsilon1] and c can be selected arbitrarily in this example. In particular, if we take c = 0 . 5 and glyph[epsilon1] = 1 / √ K our bounds remain logarithmic O (log( K )) , while prior regret bounds scale as √ K .

This work is motivated by the insight just discussed. First, we show that improved regret bounds are indeed possible by proving a tighter regret bound for STRONGEULER, an existing algorithm

based on the optimism-in-the-face-of-uncertainty (OFU) principle [30]. Our regret bound is stated in terms of our new return gaps that capture the problem difficulty more accurately and avoid explicit dependencies on the smallest value function gap gap min . Our technique applies to optimistic algorithms in general and as a by-product improves the dependency on episode length H of prior results. Second, we investigate the difficulty of RL in episodic MDPs from an information-theoretic perspective by deriving regret lower-bounds. We show that existing value-function gaps are indeed sufficient to capture difficulty of problems but only when each state is visited by an optimal policy with some probability. Finally, we prove a new lower bound when the transitions of the MDP are deterministic that depends only on the difference in return of the optimal policy and suboptimal policies, which is closely related to our notion of return gap.

## 2 Problem setting and notation

We consider reinforcement learning in episodic tabular MDPs with a fixed horizon. An MDP can be described as a tuple ( S , A , P, R, H ) , where S and A are state- and action-space of size S and A respectively, P is the state transition distribution with P ( ·| s, a ) ∈ ∆ S -1 the next state probability distribution, given that action a was taken in the current state s . R is the reward distribution defined over S × A and r ( s, a ) = E [ R ( s, a )] ∈ [0 , 1] . Episodes admit a fixed length or horizon H .

We consider layered MDPs: each state s ∈ S belongs to a layer κ ( s ) ∈ [ H ] and the only non-zero transitions are between states s, s ′ in consecutive layers, with κ ( s ′ ) = κ ( s ) + 1 . This common assumption [see e.g. 23] corresponds to MDPs with time-dependent transitions, as in [20, 7], but allows us to omit an explicit time-index in value-functions and policies. For ease of presentation, we assume there is a unique start state s 1 with κ ( s 1 ) = 1 but our results can be generalized to multiple (possibly adversarial) start states. Similarly, for convenience, we assume that all states are reachable by some policy with non-zero probability, but not necessarily all policies or the same policy.

We denote by K the number of episodes during which the MDP is visited. Before each episode k ∈ [ K ] , the agent selects a deterministic policy π k : S → A out of a set of all policies Π and π k is then executed for all H time steps in episode k . For each policy π , we denote by w π ( s, a ) = P ( S κ ( s ) = s, A κ ( s ) = a | A h = π ( S h ) ∀ h ∈ [ H ]) and w π ( s ) = ∑ a w π ( s, a ) probability of reaching stateaction pair ( s, a ) and state s respectively when executing π . For convenience, supp ( π ) = { s ∈ S : w π ( s ) &gt; 0 } is the set of states visited by π with non-zero probability. The Q- and value function of a policy π are

<!-- formula-not-decoded -->

and the regret incurred by the agent is the sum of its regret over K episodes

<!-- formula-not-decoded -->

where v π = V π ( s 1 ) is the expected total sum of rewards or return of π and V ∗ is the optimal value function V ∗ ( s ) = max π ∈ Π V π ( s ) . Finally, the set of optimal policies is denoted as Π ∗ = { π ∈ Π : V π = V ∗ } . Note that we only call a policy optimal if it satisfies the Bellman equation in every state, as is common in literature, but there may be policies outside of Π ∗ that also achieve maximum return because they only take suboptimal actions outside of their support. The variance of the Q function at a state-action pair ( s, a ) of the optimal policy is V ∗ ( s, a ) = V [ R ( s, a )] + V s ′ ∼ P ( ·| s,a ) [ V ∗ ( s ′ )] , where V [ X ] denotes the variance of the r.v. X . The maximum variance over all state-action pairs is V ∗ = max ( s,a ) V ∗ ( s, a ) . Finally, our proofs will make use of the following clipping operator clip[ a | b ] = χ ( a ≥ b ) a that sets a to zero if it is smaller than b , where χ is the indicator function.

## 3 Novel upper bounds for optimistic algorithms

In this section, we present tighter regret upper-bounds for optimistic algorithms through a novel analysis technique. Our technique can be generally applied to model-based optimistic algorithms such as STRONGEULER [30], UCBVI [3], ORLC [9] or EULER [34]. In the following, we will first

give a brief overview of this class of algorithms (see Appendix B for more details) and then state our main results for the STRONGEULER algorithm [30]. We focus on this algorithm for concreteness and ease of comparison.

Optimistic algorithms maintain estimators of the Q -functions at every state-action pair such that there exists at least one policy π for which the estimator, ¯ Q π , overestimates the Q -function of the optimal policy, that is ¯ Q π ( s, a ) ≥ Q ∗ ( s, a ) , ∀ ( s, a ) ∈ S ×A . During episode k ∈ [ K ] , the optimistic algorithm selects the policy π k with highest optimistic value function ¯ V k . By definition, it holds that ¯ V k ( s ) ≥ V ∗ ( s ) . The optimistic value and Q -functions are constructed through finite-sample estimators of the true rewards r ( s, a ) and the transition kernel P ( ·| s, a ) plus bias terms, similar to estimators for the UCB-I multi-armed bandit algorithm. Careful construction of these bias terms is crucial for deriving min-max optimal regret bounds in S, A and H [4]. Bias terms which yield the tightest known bounds come from concentration of martingales results such as Freedman's inequality [14] and empirical Bernstein's inequality for martingales [26].

The STRONGEULER algorithm not only satisfies optimism, i.e., ¯ V k ≥ V ∗ , but also a stronger version called strong optimism . To define strong optimism we need the notion of surplus which roughly measures the optimism at a fixed state-action pair. Formally the surplus at ( s, a ) during episode k is defined as

<!-- formula-not-decoded -->

We say that an algorithm is strongly optimistic if E k ( s, a ) ≥ 0 , ∀ ( s, a ) ∈ S × A , k ∈ [ K ] . Surpluses are also central to our new regret bounds and we will carefully discuss their use in Appendix F.

As hinted to in the introduction, the way prior regret bounds treat value-function gaps independently at each state-action pair can lead to excessively loose guarantees. Bounds that use value-function gaps [30, 25, 21] scale at least as

<!-- formula-not-decoded -->

where state-action pairs with zero gap appear, with gap min = min s,a : gap( s,a ) &gt; 0 gap( s, a ) , the smallest positive gap. To illustrate where these bounds are loose, let us revisit the example in Figure 1. Here, these bounds evaluate to H log( K ) c + H log( K ) glyph[epsilon1] + SH log( K ) glyph[epsilon1] , where the first two terms come from state-action pairs with positive value-function gaps and the last term comes from all the state-action pairs with zero gaps. There are several opportunities for improvement:

- O.1 State-action pairs that can only be visited by taking optimal actions: We should not pay the 1 / gap min factor for such ( s, a ) as there are no other suboptimal policies π to distinguish from π ∗ in such states.
- O.2 State-action pairs that can only be visited by taking at least one suboptimal action: We should not pay the 1 / gap( s 2 , a 3 ) factor for state-action pair ( s 2 , a 3 ) and the 1 / gap min factor for ( s 2 , a 4 ) because no optimal policy visits s 2 . Such state-action pairs should only be accounted for with the price to learn that a 2 is not optimal in state s 1 . After all, learning to distinguish between π 1 and π 2 is unnecessary for optimal return.

Both opportunities suggest that the price 1 gap( s,a ) or 1 gap min that each state-action pair ( s, a ) contributes to the regret bound can be reduced by taking into account the regret incurred by the time ( s, a ) is reached. Opportunity O.1 postulates that if no regret can be incurred up to (and including) the time step ( s, a ) is reached, then this state-action pair should not appear in the regret bound. Similarly, if this regret is necessarily large, then the agent can learn this with few observations and stop reaching ( s, a ) earlier than gap( s, a ) may suggest. Thus, as claimed in O.2 , the contribution of ( s, a ) to the regret should be more limited in this case.

Since the total regret incurred during one episode by a policy π is simply the expected sum of value-function gaps visited (Lemma F.1 in the appendix),

<!-- formula-not-decoded -->

we can measure the regret incurred up to reaching ( S t , A t ) by the sum of value function gaps ∑ t h =1 gap( S h , A h ) up to this point t . We are interested in the regret incurred up to visiting a certain

state-action pair ( s, a ) which π may visit only with some probability. We therefore need to take the expectation of such gaps conditioned on the event that ( s, a ) is actually visited. We further condition on the event that this regret is nonzero, which is exactly the case when the agent encounters a positive value-function gap within the first κ ( s ) time steps. We arrive at

<!-- formula-not-decoded -->

where B = min { h ∈ [ H +1]: gap( S h , A h ) &gt; 0 } is the first time a non-zero gap is visited. This quantity measures the regret incurred up to visiting ( s, a ) through suboptimal actions. If this quantity is large for all policies π , then a learner will stop visiting this state-action pair after few observations because it can rule out all actions that lead to ( s, a ) quickly. Conversely, if the event that we condition on has zero probability under any policy, then ( s, a ) can only be reached through optimal action choices (including a in s ) and incurs no regret. This motivates our new definition of gaps that combines value function gaps with the regret incurred up to visiting the state-action pair:

Definition 3.1 (Return gap) . For any state-action pair ( s, a ) ∈ S × A define B ( s, a ) ≡ { B ≤ κ ( s ) , S κ ( s ) = s, A κ ( s ) = a } , where B is the first time a non-zero gap is encountered. B ( s, a ) denotes the event that state-action pair ( s, a ) is visited and that a suboptimal action was played at any time up to visiting ( s, a ) . We define the return gap as

<!-- formula-not-decoded -->

if there is a policy π ∈ Π with P π ( B ( s, a )) &gt; 0 and gap( s, a ) ≡ 0 otherwise.

The additional 1 /H factor in the second term is a required normalization suggesting that it is the average gap rather than their sum that matters. We emphasize that Definition 3.1 is independent of the choice of RL algorithm and in particular does not depend on the algorithm being optimistic. Thus, we expect our main ideas and techniques to be useful beyond the analysis of optimistic algorithms. Equipped with this definition, we are ready to state our main upper bound which pertains to the STRONGEULER algorithm proposed by Simchowitz and Jamieson [30].

Theorem 3.2 (Main Result (Informal)) . The regret R ( K ) of STRONGEULER is bounded with high probability for all number of episodes K as

<!-- formula-not-decoded -->

In the above, we have restricted the bound to only those terms that have inverse polynomial dependence on the gaps.

Comparison with existing gap-dependent bounds. We now compare our bound to the existing gap-dependent bound for STRONGEULER by Simchowitz and Jamieson [30, Corollary B.1]

<!-- formula-not-decoded -->

We here focus only on terms that admit a dependency on K and an inverse-polynomial dependency on gaps as all other terms are comparable. Most notable is the absence of the second term of (4) in our bound in Theorem 3.2. Thus, while state-action pairs with gap( s, a ) = 0 do not contribute to our regret bound, they appear with a 1 / gap min factor in existing bounds. Therefore, our bound addresses O.1 because it does not pay for state-action pairs that can only be visited through optimal actions. Further, state-action pairs that do contribute to our bound satisfy 1 gap( s,a ) ≤ 1 gap( s,a ) ∧ H gap min and thus never contribute more than in the existing bound in (4). Therefore, our regret bound is never worse. In fact, it is significantly tighter when there are states that are only reachable by taking severely suboptimal actions, i.e., when the average value-function gaps are much larger than gap( s, a ) or

gap min . By our definition of return gaps, we only pay the inverse of these larger gaps instead of gap min . Thus, our bound also addresses O.2 and achieves the desired log( K ) /c regret bound in the motivating example of Figure 1 as opposed to the log( K ) /glyph[epsilon1] bound of prior work.

One of the limitations of optimistic algorithms is their S / gap min dependence even when there is only one state with a gap of gap min [30]. We note that even though our bound in Theorem 3.2 improves on prior work, our result does not aim to address this limitation. Very recent concurrent work [32] proposed an action-elimination based algorithm that avoids the S / gap min issue of optimistic algorithm but their regret bounds still suffer the issues illustrated in Figure 1 (e.g. O.2 ). We therefore view our contributions as complementary. In fact, we believe our analysis techniques can be applied to their algorithm as well and result similar improvements as for the example in Figure 1.

Regret bound when transitions are deterministic. We now interpret Definition 3.1 for MDPs with deterministic transitions and derive an alternative form of our bound in this case. Let Π s,a be the set of all policies that visit ( s, a ) and have taken a suboptimal action up to that visit, that is,

<!-- formula-not-decoded -->

where ( s π 1 , a π 1 , s π 2 , . . . , s π H , a π H ) are the state-action pairs visited (deterministically) by π . Further, let v ∗ s,a = max π ∈ Π s,a v π be the best return of such policies. Definition 3.1 now evaluates to gap( s, a ) = gap( s, a ) ∨ 1 H ( v ∗ -v ∗ s,a ) and the bound in Theorem 3.2 can be written as glyph[negationslash]

<!-- formula-not-decoded -->

We show in Appendix F.7, that it is possible to further improve this bound when the optimal policy is unique by only summing over state-action pairs which are not visited by the optimal policy.

## 3.1 Regret analysis with improved clipping: from minimum gap to average gap

In this section, we present the main technical innovations of our tighter regret analysis. Our framework applies to optimistic algorithms that maintain a Q -function estimate, ¯ Q k ( s, a ) , which overestimates the optimal Q -function Q ∗ ( s, a ) with high probability in all states s , actions a and episodes k . We first give an overview of gap-dependent analyses and then describe our approach.

Overview of gap-dependent analyses. A central quantity in regret analyses of optimistic algorithms are the surpluses E k ( s, a ) , defined in (2), which, roughly speaking, quantify the local amount of optimism. Worst-case regret analyses bound the regret in episode k as ∑ ( s,a ) ∈S×A w π k ( s, a ) E k ( s, a ) , the expected surpluses under the optimistic policy π k executed in that episode. Instead, gap-dependent analyses rely on a tighter version and bound the instantaneous regret by the clipped surpluses [e.g. Proposition 3.1 30]

<!-- formula-not-decoded -->

Sharper clipping with general thresholds. Our main technical contribution for achieving a regret bound in terms of return gaps gap( s, a ) is the following improved surplus clipping bound:

Proposition 3.3 (Improved surplus clipping bound) . Let the surpluses E k ( s, a ) be generated by an optimistic algorithm. Then the instantaneous regret of π k is bounded as follows:

<!-- formula-not-decoded -->

where glyph[epsilon1] k : S × A → R + 0 is any clipping threshold function that satisfies

<!-- formula-not-decoded -->

Compared to previous surplus clipping bounds in (6), there are several notable differences. First, instead of gap min / 2 H , we can now pair gap( s, a ) with more general clipping thresholds glyph[epsilon1] k ( s, a ) , as long as their expected sum over time steps after the first non-zero gap was encountered is at most half the expected sum of gaps. We will provide some intuition for this condition below. Note that glyph[epsilon1] k ( s, a ) ≡ gap min 2 H satisfies the condition because the LHS is bounded between gap min 2 H P π k ( B ≤ H ) and gap min P π k ( B ≤ H ) , and there must be at least one positive gap in the sum ∑ H h =1 gap( S h , A h ) on the RHS in event { B ≤ H } . Thus our bound recovers existing results. In addition, the first term in our clipping thresholds is 1 4 gap( s, a ) instead of 1 4 H gap( s, a ) . Simchowitz and Jamieson [30] are able to remove this spurious H factor only if the problem instance happens to be a bandit instance and the algorithm satisfies a condition called strong optimism where surpluses have to be non-negative. Our analysis does not require such conditions and therefore generalizes these existing results. 2

Choice of clipping thresholds for return gaps. The condition in Proposition 3.3 suggests that one can set glyph[epsilon1] k ( S h , A h ) to be proportional to the average expected gap under policy π k :

<!-- formula-not-decoded -->

if P π k ( B ( s, a )) &gt; 0 and glyph[epsilon1] k ( s, a ) = ∞ otherwise. Lemma F.5 in Appendix F shows that this choice indeed satisfies the condition in Proposition 3.3. If we now take the minimum over all policies for π k , then we can proceed with the standard analysis and derive our main result in Theorem 3.2. However, by avoiding the minimum over policies, we can derive a stronger policy-dependent regret bound which we discuss in the appendix.

## 4 Instance-dependent lower bounds

We here shed light on what properties on an episodic MDP determine the statistical difficulty of RL by deriving information-theoretic lower bounds on the asymptotic expected regret of any (good) algorithm. To that end, we first derive a general result that expresses a lower bound as the optimal value of a certain optimization problem and then derive closed-form lower-bounds from this optimization problem that depend on certain notions of gaps for two special cases of episodic MDPs.

Specifically, in those special cases, we assume that the rewards follow a Gaussian distribution with variance 1 / 2 . We further assume that the optimal value function is bounded in the same range as individual rewards, e.g. as 0 ≤ V ∗ ( s ) &lt; 1 for all s ∈ S . This assumption is common in the literature [e.g. 23, 19, 8] and can be considered harder than a normalization of V ∗ ( s ) ∈ [0 , H ] [18].

## 4.1 General instance-dependent lower bound as an optimization problem

The idea behind deriving instance-dependent lower bounds for the stochastic MAB problem [24, 5, 15] and infinite horizon MDPs [16, 27] are based on first assuming that the algorithm studied is uniformly good , that is, on any instance of the problem and for any α &gt; 0 , the algorithm incurs regret at most o ( T α ) , and then argue that, to achieve that guarantee, the algorithm must select a certain policy or action at least some number of times as it would otherwise not be able to distinguish the current MDP from another MDP that requires a different optimal strategy.

Since comparison between different MDPs is central to lower-bound constructions, it is convenient to make the problem-instance explicit in the notation. To that end, let Θ be the problem class of possible MDPs and we use subscripts θ and λ for value functions, return, MDP parameters etc., to denote specific problem instances θ, λ ∈ Θ of those quantities. Further, for a policy π and MDP θ , P π θ denotes the law of one episode, i.e., the distribution of ( S 1 , A 1 , R 1 , S 2 , A 2 , R 2 , . . . , S H +1 ) . To state the general regret lower-bound we need to introduce the set of confusing MDPs. This set consists of all MDPs λ in which there is at least one optimal policy π such that π glyph[negationslash]∈ Π ∗ θ , i.e., π is not optimal for the original MDP and no policy in Π ∗ θ has been changed.

Definition 4.1. For any problem instance θ ∈ Θ we define the set of confusing MDPs Λ( θ ) as glyph[negationslash]

<!-- formula-not-decoded -->

2 Our layered state space assumption changes H factors in lower-order terms of our final regret compared to Simchowitz and Jamieson [30]. However, Proposition 3.3 directly applies to their setting with no penalty in H .

We are now ready to state our general regret lower-bound for episodic MDPs:

Theorem 4.2 (General instance-dependent lower bound for episodic MDPs) . Let ψ be a uniformly good RL algorithm for Θ , that is, for all problem instances θ ∈ Θ and exponents α &gt; 0 , the regret of ψ is bounded as E [ R θ ( K )] ≤ o ( K α ) , and assume that v ∗ θ &lt; H . Then, for any θ ∈ Θ , the regret of ψ satisfies

<!-- formula-not-decoded -->

where C ( θ ) is the optimal value of the following optimization problem

<!-- formula-not-decoded -->

The optimization problem in Theorem 4.2 can be interpreted as follows. The variables η ( π ) are the (expected) number of times the algorithm chooses to play policy π which makes the objective the total expected regret incurred by the algorithm. The constraints encode that any uniformly good algorithm needs to be able to distinguish the true instance θ from all confusing instances λ ∈ Λ( θ ) , because otherwise it would incur linear regret. To do so, a uniformly good algorithm needs to play policies π that induce different behavior in λ and θ which is precisely captured by the constraints ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ 1 .

Although Theorem 4.2 has the flavor of results in the bandit and RL literature, there are a few notable differences. Compared to lower-bounds in the infinite-horizon MDP setting [16, 31, 27], we for example do not assume that the Markov chain induced by an optimal policy π ∗ is irreducible. That irreducibility plays a key role in converting the semi-infinite linear program (8), which typically has uncountably many constraints, into a linear program with only O ( SA ) constraints. While for infinite horizon MDPs, irreducibility is somewhat necessary to facilitate exploration, this is not the case for the finite horizon setting and in general we cannot obtain a convenient reduction of the set of constraints Λ( θ ) (see also Appendix E.2).

## 4.2 Gap-dependent lower bound when optimal policies visit all states

To derive closed-form gap-dependent bounds from the general optimization problem (8), we need to identify a finite subset of confusing MDPs Λ( θ ) that each require the RL agent to play a distinct set of policies that do not help to distinguish the other confusing MDPs. To do so, we restrict our attention to the special case of MDPs where every state is visited with non-zero probability by some optimal policy, similar to the irreducibility assumptions in the infinite-horizon setting [31, 27]. In this case, it is sufficient to raise the expected immediate reward of a suboptimal ( s, a ) by gap θ ( s, a ) in order to create a confusing MDP, as shown in Lemma 4.3:

Lemma 4.3. Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1 and let θ ∈ Θ be an MDP in this class. Then for any suboptimal state-action pair ( s, a ) with gap θ ( s, a ) &gt; 0 such that s is visited by some optimal policy with non-zero probability, there exists a confusing MDP λ ∈ Λ( θ ) with

- λ and θ only differ in the immediate reward at ( s, a )
- KL ( P π θ , P π λ ) ≤ gap θ ( s, a ) 2 for all π ∈ Π .

By relaxing the problem in (8) to only consider constraints from the confusing MDPs in Lemma 4.3 with KL ( P π θ , P π λ ) ≤ gap θ ( s, a ) 2 , for every ( s, a ) , we can derive the following closed-form bound:

Theorem 4.4 (Gap-dependent lower bound when optimal policies visit all states) . Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1. Let θ ∈ Θ be an instance where every state is visited by some optimal policy with non-zero probability. Then any uniformly good algorithm on Θ has expected regret on θ that satisfies

<!-- formula-not-decoded -->

Theorem 4.4 can be viewed as a generalization of Proposition 2.2 in Simchowitz and Jamieson [30], which gives a lower bound of order ∑ s,a : gap θ ( s,a ) &gt; 0 H gap θ ( s,a ) for a certain set of MDPs. 3 While our lower bound is a factor of H worse, it is significantly more general and holds in any MDP where optimal policies visit all states and with appropriate normalization of the value function. Theorem 4.4 indicates that value-function gaps characterize the instance-optimal regret when optimal policies cover the entire state space.

## 4.3 Gap-dependent lower bound for deterministic-transition MDPs

We expect that optimal policies do not visit all states in most MDPs of practical interest (e.g. because certain parts of the state space can only be reached by making an egregious error). We therefore now consider the general case where ⋃ π ∈ Π ∗ θ supp ( π ) glyph[subsetnoteql] S but restrict our attention to MDPs with deterministic transitions where we are able to give an intuitive closed-form lower bound. Note that deterministic transitions imply ∀ π, s, a : w π ( s, a ) ∈ { 0 , 1 } . Here, a confusing MDP can be created by simply raising the reward of any ( s, a ) by

<!-- formula-not-decoded -->

the regret of the best policy that visits ( s, a ) , as long as it is positive and ( s, a ) is not visited by any optimal policy. (9) is positive when no optimal policy visits ( s, a ) in which case suboptimal actions have to be taken to reach ( s, a ) and gap θ ( s, a ) &gt; 0 . Let π ∗ ( s,a ) be any maximizer in (9), which has to act optimally after visiting ( s, a ) . From the regret decomposition in (3) and the fact that π ∗ ( s,a ) visits ( s, a ) with probability 1 , it follows that v ∗ θ -v π ∗ ( s,a ) θ ≥ gap θ ( s, a ) . We further have v ∗ θ -v π ∗ ( s,a ) θ ≤ H gap θ ( s, a ) . Equipped with the subset of confusing MDPs λ that each raise the reward of a single ( s, a ) as r λ ( s, a ) = r θ ( s, a ) + v ∗ θ -v π ∗ ( s,a ) θ , we can derive the following gap-dependent lower bound:

Theorem 4.5. Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1. Let θ ∈ Θ be an instance with deterministic transitions. Then any uniformly good algorithm on Θ has expected regret on θ that satisfies

<!-- formula-not-decoded -->

where Z θ = { ( s, a ) ∈ S × A : ∀ π ∗ ∈ Π ∗ θ w π ∗ θ ( s, a ) = 0 } is the set of state-action pairs that no optimal policy in θ visits.

We now compare the above lower bound to the upper bound guaranteed by STRONGEULER in (5). The comparison is only with respect to number of episodes and gaps 4

<!-- formula-not-decoded -->

glyph[negationslash]

The difference between the two bounds, besides the extra H 2 factor, is the fact that ( s, a ) pairs that are visited by any optimal policy ( s, a = Z θ ) do not appear in the lower-bound while the upper-bound pays for such pairs if they can also be visited after playing a suboptimal action. This could result in cases where the number of terms in the lower bound is O (1) but the number of terms in the upper bound is Ω( SA ) leading to a large discrepancy. In Theorem E.11 in the appendix we show that there exists an MDP instance on which it is information-theoretically possible to achieve O (log( K ) /glyph[epsilon1] ) regret, however, any optimistic algorithm with confidence parameter δ will incur expected regret of at least Ω( S log(1 /δ ) /glyph[epsilon1] ) . Theorem E.11 has two implications for optimistic algorithms in MDPs with deterministic transitions. Specifically, optimistic algorithms

- cannot be asymptotically optimal if confidence parameter δ is tuned to the time horizon K ;
- cannot have an anytime bound that matches the information-theoretic lower bound.

3 We translated their results to our setting where V ∗ ≤ 1 which reduces the bound by a factor of H .

4 We carry out the comparison in expectation, since our lower bounds do not apply with high probability.

## 5 Conclusion

In this work, we prove that optimistic algorithms such as STRONGEULER, can suffer substantially less regret compared to what prior work had shown. We do this by introducing a new notion of gap, while greatly simplifying and generalizing existing analysis techniques. We further investigated the information-theoretic limits of learning episodic layered MDPs. We provide two new closed-form lower bounds in the special case where the MDP has either deterministic transitions or the optimal policy is supported on all states. These lower bounds suggest that our notion of gap better captures the difficulty of an episodic MDP for RL.

## References

- [1] Peter Auer and Ronald Ortner. Logarithmic online regret bounds for undiscounted reinforcement learning. In Advances in Neural Information Processing Systems , pages 49-56, 2007.
- [2] Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. In Advances in Neural Information Processing Systems , 2009.
- [3] Mohammad Gheshlaghi Azar, Rémi Munos, and Hilbert J Kappen. On the sample complexity of reinforcement learning with a generative model. In Proceedings of the 29th International Coference on International Conference on Machine Learning , pages 1707-1714. Omnipress, 2012.
- [4] Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272, 2017.
- [5] Richard Combes, Stefan Magureanu, and Alexandre Proutiere. Minimal exploration in structured stochastic bandits. In Advances in Neural Information Processing Systems , pages 1763-1771, 2017.
- [6] Christoph Dann. Strategic Exploration in Reinforcement Learning - New Algorithms and Learning Guarantees . PhD thesis, Carnegie Mellon University, 2019.
- [7] Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying PAC and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , pages 5713-5723, 2017.
- [8] Christoph Dann, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. On oracle-efficient PAC reinforcement learning with rich observations. arXiv preprint arXiv:1803.00606 , 2018.
- [9] Christoph Dann, Lihong Li, Wei Wei, and Emma Brunskill. Policy certificates: Towards accountable reinforcement learning. International Conference on Machine Learning , 2019.
- [10] Simon S Du, Jason D Lee, Gaurav Mahajan, and Ruosong Wang. Agnostic Q-learning with function approximation in deterministic systems: Tight bounds on approximation error and sample complexity. arXiv preprint arXiv:2002.07125 , 2020.
- [11] Claude-Nicolas Fiechter. Efficient reinforcement learning. In Proceedings of the seventh annual conference on Computational learning theory , pages 88-97. ACM, 1994.
- [12] Sarah Filippi, Olivier Cappé, and Aurélien Garivier. Optimism in reinforcement learning and Kullback-Leibler divergence. In 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton) , pages 115-122. IEEE, 2010.
- [13] Dylan J Foster, Alexander Rakhlin, David Simchi-Levi, and Yunzong Xu. Instance-dependent complexity of contextual bandits and reinforcement learning: A disagreement-based perspective. arXiv preprint arXiv:2010.03104 , 2020.
- [14] David A Freedman. On tail probabilities for martingales. the Annals of Probability , pages 100-118, 1975.

- [15] Aurélien Garivier, Pierre Ménard, and Gilles Stoltz. Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research , 44(2):377-399, 2019.
- [16] Todd L Graves and Tze Leung Lai. Asymptotically efficient adaptive choice of control laws incontrolled markov chains. SIAM journal on control and optimization , 35(3):715-743, 1997.
- [17] Jiafan He, Dongruo Zhou, and Quanquan Gu. Logarithmic regret for reinforcement learning with linear function approximation. arXiv preprint arXiv:2011.11566 , 2020.
- [18] Nan Jiang and Alekh Agarwal. Open problem: The dependence of sample complexity lower bounds on planning horizon. In Conference On Learning Theory , pages 3395-3398, 2018.
- [19] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning , pages 1704-1713, 2017.
- [20] Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is Q-learning provably efficient? arXiv preprint arXiv:1807.03765 , 2018.
- [21] Tiancheng Jin and Haipeng Luo. Simultaneously learning stochastic and adversarial episodic MDPs with known transition. arXiv preprint arXiv:2006.05606 , 2020.
- [22] Sham Kakade. On the sample complexity of reinforcement learning . PhD thesis, University College London, 2003.
- [23] Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Pac reinforcement learning with rich observations. In Advances in Neural Information Processing Systems , pages 1840-1848, 2016.
- [24] Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics , 6(1):4-22, 1985.
- [25] Thodoris Lykouris, Max Simchowitz, Aleksandrs Slivkins, and Wen Sun. Corruption robust exploration in episodic reinforcement learning. arXiv preprint arXiv:1911.08689 , 2019.
- [26] Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. arXiv preprint arXiv:0907.3740 , 2009.
- [27] Jungseul Ok, Alexandre Proutiere, and Damianos Tranos. Exploration in structured reinforcement learning. In Advances in Neural Information Processing Systems , pages 8874-8882, 2018.
- [28] Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via posterior sampling. In Advances in Neural Information Processing Systems , pages 3003-3011, 2013.
- [29] Martin Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley-Interscience, 1994.
- [30] Max Simchowitz and Kevin Jamieson. Non-asymptotic gap-dependent regret bounds for tabular MDPs. arXiv preprint arXiv:1905.03814 , 2019.
- [31] Ambuj Tewari and Peter L Bartlett. Optimistic linear programming gives logarithmic regret for irreducible MDPs. In Advances in Neural Information Processing Systems , pages 1505-1512, 2008.
- [32] Haike Xu, Tengyu Ma, and Simon S Du. Fine-grained gap-dependent bounds for tabular mdps via adaptive multi-step bootstrap. arXiv preprint arXiv:2102.04692 , 2021.
- [33] Kunhe Yang, Lin F Yang, and Simon S Du. Q -learning with logarithmic regret. arXiv preprint arXiv:2006.09118 , 2020.
- [34] A. Zanette and E. Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. https://arxiv.org/abs/1901.00210 , 2019.

- [35] Alexander Zimin and Gergely Neu. Online learning in episodic markovian decision processes by relative entropy policy search. In Advances in neural information processing systems , pages 1583-1591, 2013.

## Contents of main article and appendix

| 1 Introduction 1                                                                                                                                                                                                        | 1 Introduction 1                                                                                                                                                                                                        | 1 Introduction 1                                                                                                      | 1 Introduction 1                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2 Problem setting and notation                                                                                                                                                                                          | 2 Problem setting and notation                                                                                                                                                                                          | 3                                                                                                                     | 2 Problem setting and notation                                                                                                                                                                                          |
| 3 Novel upper bounds for optimistic algorithms                                                                                                                                                                          | 3 Novel upper bounds for optimistic algorithms                                                                                                                                                                          | 3                                                                                                                     | 3 Novel upper bounds for optimistic algorithms                                                                                                                                                                          |
| 3.1 Regret analysis with improved clipping: from minimum gap to average gap                                                                                                                                             | 3.1 Regret analysis with improved clipping: from minimum gap to average gap                                                                                                                                             | 6                                                                                                                     | 3.1 Regret analysis with improved clipping: from minimum gap to average gap                                                                                                                                             |
| 4 Instance-dependent lower bounds                                                                                                                                                                                       | 4 Instance-dependent lower bounds                                                                                                                                                                                       | 7                                                                                                                     | 4 Instance-dependent lower bounds                                                                                                                                                                                       |
| 4.1 General instance-dependent lower bound as an optimization problem . . .                                                                                                                                             | 4.1 General instance-dependent lower bound as an optimization problem . . .                                                                                                                                             | 7                                                                                                                     | 4.1 General instance-dependent lower bound as an optimization problem . . .                                                                                                                                             |
| 4.2 Gap-dependent lower bound when optimal policies visit all states . . . .                                                                                                                                            | 4.2 Gap-dependent lower bound when optimal policies visit all states . . . .                                                                                                                                            | 8                                                                                                                     | 4.2 Gap-dependent lower bound when optimal policies visit all states . . . .                                                                                                                                            |
| 4.3 Gap-dependent lower bound for deterministic-transition MDPs . . . . . . . . . . .                                                                                                                                   | 4.3 Gap-dependent lower bound for deterministic-transition MDPs . . . . . . . . . . .                                                                                                                                   | 9                                                                                                                     | 4.3 Gap-dependent lower bound for deterministic-transition MDPs . . . . . . . . . . .                                                                                                                                   |
| 5 Conclusion                                                                                                                                                                                                            | 5 Conclusion                                                                                                                                                                                                            | 10                                                                                                                    | 5 Conclusion                                                                                                                                                                                                            |
| A Related work                                                                                                                                                                                                          | A Related work                                                                                                                                                                                                          | 14                                                                                                                    | A Related work                                                                                                                                                                                                          |
| B Model-based optimistic algorithms for tabular RL                                                                                                                                                                      | B Model-based optimistic algorithms for tabular RL                                                                                                                                                                      | 15                                                                                                                    | B Model-based optimistic algorithms for tabular RL                                                                                                                                                                      |
| C Experimental results                                                                                                                                                                                                  | C Experimental results                                                                                                                                                                                                  | 16                                                                                                                    | C Experimental results                                                                                                                                                                                                  |
| D Additional Notation                                                                                                                                                                                                   | D Additional Notation                                                                                                                                                                                                   | 17                                                                                                                    | D Additional Notation                                                                                                                                                                                                   |
| E Proofs and extended discussion for regret lower-bounds                                                                                                                                                                | E Proofs and extended discussion for regret lower-bounds                                                                                                                                                                | 17                                                                                                                    | E Proofs and extended discussion for regret lower-bounds                                                                                                                                                                |
| E.1 Lower bound as an optimization problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                                                    |                                                                                                                                                                                                                         | 18                                                                                                                    |                                                                                                                                                                                                                         |
| policy .                                                                                                                                                                                                                | policy .                                                                                                                                                                                                                | 20                                                                                                                    | policy .                                                                                                                                                                                                                |
| E.2 Lower bounds for full support optimal E.3 Lower bounds for deterministic MDPs . . . . . . . . . . . . . . . . . .                                                                                                   | E.2 Lower bounds for full support optimal E.3 Lower bounds for deterministic MDPs . . . . . . . . . . . . . . . . . .                                                                                                   | E.2 Lower bounds for full support optimal E.3 Lower bounds for deterministic MDPs . . . . . . . . . . . . . . . . . . | E.2 Lower bounds for full support optimal E.3 Lower bounds for deterministic MDPs . . . . . . . . . . . . . . . . . .                                                                                                   |
|                                                                                                                                                                                                                         |                                                                                                                                                                                                                         | 23                                                                                                                    |                                                                                                                                                                                                                         |
| E.3.1 Lower bound for Markov decision processes with bounded value                                                                                                                                                      | E.3.1 Lower bound for Markov decision processes with bounded value                                                                                                                                                      | 24                                                                                                                    | E.3.1 Lower bound for Markov decision processes with bounded value                                                                                                                                                      |
| E.3.2 Tree-structured MDPs . . . . . . . . . . . . E.3.3 Issue with deriving a general bound . . . . . E.4 Lower bounds for optimistic algorithms in MDPs with F Proofs and extended discussion for regret upper-bounds | E.3.2 Tree-structured MDPs . . . . . . . . . . . . E.3.3 Issue with deriving a general bound . . . . . E.4 Lower bounds for optimistic algorithms in MDPs with F Proofs and extended discussion for regret upper-bounds | 31                                                                                                                    | E.3.2 Tree-structured MDPs . . . . . . . . . . . . E.3.3 Issue with deriving a general bound . . . . . E.4 Lower bounds for optimistic algorithms in MDPs with F Proofs and extended discussion for regret upper-bounds |
| F.1 . . . . . . . . . . . . . . . .                                                                                                                                                                                     | Further discussion on Opportunity O.2 . .                                                                                                                                                                               | 31                                                                                                                    | Further discussion on Opportunity O.2 . .                                                                                                                                                                               |
| F.2 . . . . . . . . . . . . . . . . . .                                                                                                                                                                                 | Useful decomposition lemmas . . . .                                                                                                                                                                                     | 32                                                                                                                    | Useful decomposition lemmas . . . .                                                                                                                                                                                     |
| F.3                                                                                                                                                                                                                     | General surplus clipping for optimistic algorithms . . . . . . . . . . . . . . . . . . .                                                                                                                                | 33                                                                                                                    | General surplus clipping for optimistic algorithms . . . . . . . . . . . . . . . . . . .                                                                                                                                |
| F.4                                                                                                                                                                                                                     | Definition of valid clipping thresholds glyph[epsilon1] . . . . . . . . . . . . . . . . . . . . . .                                                                                                                     | 36                                                                                                                    | Definition of valid clipping thresholds glyph[epsilon1] . . . . . . . . . . . . . . . . . . . . . .                                                                                                                     |
| F.5 Policy-dependent regret bound for STRONGEULER . . . . . . . . . . .                                                                                                                                                 | k                                                                                                                                                                                                                       | 39                                                                                                                    | k                                                                                                                                                                                                                       |
| F.6                                                                                                                                                                                                                     | Nearly tight bounds for deterministic transition MDPs . . . . . . . . . . . . . . .                                                                                                                                     | 43                                                                                                                    | Nearly tight bounds for deterministic transition MDPs . . . . . . . . . . . . . . .                                                                                                                                     |
| F.7                                                                                                                                                                                                                     | Tighter bounds for unique optimal policy. . . . . . . . . . . . . . . . . . . . .                                                                                                                                       | 44                                                                                                                    | Tighter bounds for unique optimal policy. . . . . . . . . . . . . . . . . . . . .                                                                                                                                       |
|                                                                                                                                                                                                                         | . . . . . .                                                                                                                                                                                                             | 47                                                                                                                    | . . . . . .                                                                                                                                                                                                             |
| F.8                                                                                                                                                                                                                     | Alternative to integration lemmas . . . . . . . . . . . . . . . . . . .                                                                                                                                                 | Alternative to integration lemmas . . . . . . . . . . . . . . . . . . .                                               | Alternative to integration lemmas . . . . . . . . . . . . . . . . . . .                                                                                                                                                 |

## Checklist

1. For all authors...
2. (a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes] See Section 3, Section 4 and corresponding sections in the appendix.
3. (b) Did you describe the limitations of your work? [Yes] See lower bounds, discussion after Equation 5, Appendix E.3.3
4. (c) Did you discuss any potential negative societal impacts of your work? [N/A] Our work is theoretical and we do not see any potential negative societal impacts.
5. (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]
2. If you are including theoretical results...
7. (a) Did you state the full set of assumptions of all theoretical results? [Yes]
8. (b) Did you include complete proofs of all theoretical results? [Yes] See Appendix E for lower bounds and Appendix F for upper bounds.
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

## A Related work

We now discuss related work carefully. Instance dependent regret lower bounds for the MAB were first introduced in Lai and Robbins [24]. Later Graves and Lai [16] extend such instance dependent lower bounds to the setting of controlled Markov chains, while assuming infinite horizon and certain properties of the stationary distribution of each policy. Building on their work, more recently Combes et al. [5] establish instance dependent lower bounds for the Structured Stochastic Bandit problem. Very recently, in the stochastic MAB, Garivier et al. [15] generalize and simplify the techniques of Lai and Robbins [24] to completely characterize the behavior of uniformly good algorithms. The work of Ok et al. [27] builds on these ideas to provide an instance dependent lower bound for infinite horizon MDPs, again under assumptions of how the stationary distributions of each policy will behave and

irreducibility of the Markov chain. The idea behind deriving the above bounds is to use the uniform goodness of the studied algorithm to argue that the algorithm must select a certain policy or action at least a fixed number of times. This number is governed by a change of environment under which said policy/action is now the best overall. The reasoning now is that unless the algorithm is able to distinguish between these two environments it will have to incur linear regret asymptotically. Since the algorithm is uniformly good this can not happen.

For infinite horizon MDPs with additional assumptions the works of Auer and Ortner [1], Tewari and Bartlett [31], Auer et al. [2], Filippi et al. [12], Ok et al. [27] establish logarithmic in horizon regret bounds of the form O ( D 2 S 2 A log( T ) /δ ) , where δ is a gap-like quantity and D is a diameter measure. We now discuss the works of [31, 27], which should give more intuition about how the infinite horizon setting differs from our setting. Both works consider the non-episodic problem and therefore make some assumptions about the MDP M . The main assumption, which allows for computationally tractable algorithms is that of irreducibility. Formally both works require that under any policy the induced Markov chain is irreducible. Intuitively, the notion of irreducibility allows for coming up with exploration strategies, which are close to min-max optimal and are easy to compute. In [27] this is done by considering the same semi-infinite LP 8 as in our work. Unlike our work, however, assuming that the Markov chain induced by the optimal policy π ∗ is irreducible allows for a nice characterization of the set Λ( θ ) of "confusing" environments. In particular the authors manage to show that at every state s it is enough to consider the change of environment which makes the reward of any action a : ( s, a ) glyph[negationslash]∈ π ∗ equal to the reward of a ′ : ( s, a ′ ) ∈ π ∗ . Because of the irreducability assumption we know that the support of P ( ·| s, a ) is the same as the support of P ( ·| s, a ′ ) and this implies that the above change of environment makes the policy π which plays ( s, a ) and then coincides with π ∗ optimal. Some more work shows that considering only such changes of environment is sufficient for an equivalent formulation to the LP8. Since this is an LP with at most S × A constraints it is solvable in polynomial time and hence a version of the algorithm in [5] results in asymptotic min-max rates for the problem. The exploration in [31] is also based on a similar LP, however, slightly more sophisticated.

Very recently there has been a renewed interest in proposing instance dependent regret bounds for finite horizon tabular MDPs [30, 25, 21]. The works of [30, 25] are based on the OFU principle and the proposed regret bounds scale as O ( ∑ ( s,a ) glyph[negationslash]∈ π ∗ H log( T ) / gap( s, a ) + SH log( T ) / gap min ) , disregarding variance terms and terms depending only poli-logarithmically on the gaps. The setting in [25] also considers adversarial corruptions to the MDP, unknown to the algorithm, and their bound scales with the amount of corruption. Jin and Luo [21] derive similar upper bounds, however, the authors assume a known transition kernel and take the approach of modelling the problem as an instance of Online Linear Optimization, through using occupancy measures [35]. For the problem of Q -learning, Yang et al. [33], Du et al. [10], also propose algorithms with regret scaling as O ( SAH 6 log( T ) / gap min ) . All of these bounds scale at least as Ω( SH log( T ) / gap min ) . Simchowitz and Jamieson [30] show an MDP instance on which no optimistic algorithm can hope to do better.

## B Model-based optimistic algorithms for tabular RL

This section is a general discussion of optimistic algorithms for the tabular setting. Our regret upper bounds can be extended to other model based optimistic algorithms or in general any optimistic algorithm for which we can show a meaningful bound on the surpluses in terms of the number of times a state-action pair has been visited throughout the K episodes.

Pseudo-code for a generic algorithm can be found in Algorithm 1. The algorithm begins by initializing an empirical transition kernel ˆ P ∈ [0 , 1] S × A × S , empirical reward kernel ˆ r ∈ [0 , 1] S × A , and bonuses b ∈ [0 , 1] S × A . If we let n k ( s, a ) be the number of times we have observed state-action pair ( s, a ) up to episode k and n k ( s ′ , s, a ) the number of times we have observed state s ′ after visiting ( s, a ) then one standard way to define the empirical kernels at episode k are as follows:

<!-- formula-not-decoded -->

where R j ( s, a ) is a sample from r ( s, a ) at episode j if ( s, a ) was visited and 0 otherwise. At every episode the generic algorithm constructs an policy π k using the empirical model together with bonus

TC*

$2,1

§1,1

T1

$2,2

Algorithm 1 Generic Model-Based Optimistic Algorithm for Tabular RL

Require: Number of episodes K , horizon H , number of states S , number of actions A , probability of failure δ .

Ensure: A sequence of policies ( π k ) K k =1 with low regret.

- 1: Initialize empirical transition kernel ˆ P ∈ [0 , 1] S × A × S , empirical reward kernel ˆ r ∈ [0 , 1] S × A , bonuses b ∈ [0 , 1] S × A .
- 2: for k ∈ [ K ] do
- 3: h = H , Q k ( s H +1 , a H +1 ) = 0 , ∀ ( s, a ) ∈ S × A .
- 4: while h &gt; 0 do
- 5: Q k ( s, a ) = ˆ r ( s, a ) + 〈 ˆ P ( ·| s, a ) , V k 〉 + b ( s, a ) .
- 6: π k ( s ) := argmax a Q k ( s, a ) .
- 7: h -= 1
- 8: Play π k , collect observations from transition kernel P and reward kernel r and update ˆ P , ˆ r , b .

terms b ( s, a ) , ∀ ( s, a ) ∈ S × A . Bonuses are constructed by using concentration of measure results relating ˆ r ( s, a ) to r ( s, a ) and ˆ P ( ·| s, a ) to P ( ·| s, a ) . These bonuses usually scale inversely with the empirical visitations n k ( s, a ) , ∀ ( s, a ) ∈ S × A , as O (1 / √ n k ( s, a )) . Further, depending on the type of concentration of measure result, the bonuses could either have a direct dependence on K,H,S,A,δ (following from Azuma-Hoeffding style concentration bounds) or replace H with the empirical estimator (following Freedman style concentration bounds). The bonus terms ensure that optimism is satisfied for π k , that is Q k ( s, a ) ≥ Q π k ( s, a ) for all ( s, a ) ∈ S × A and all episodes k ∈ [ K ] with probability at least 1 -δ . Algorithms such as UCBVI [4], EULER [34] and STRONGEULER [30] are all versions of Algorithm 1 with different instantiations of the bonus terms.

The greedy choice of π k together with optimism also ensures that V k ( s ) ≥ V ∗ ( s ) . This has been key in prior work as it is what allows to bound the instantaneous regret by the sum of surpluses and ultimately relate the regret upper bound back to the bonus terms and the number of visits of each state-action pair respectively. Our regret upper bounds are also based on this decomposition and as such are not really tied to the STRONGEULER algorithm but would work with any model-based optimistic algorithm for the tabular setting. The main novelty in this work is a way to control the surpluses by clipping them to a gap-like quantity which better captures the sub-optimality of π k compared to π ∗ . We remark that our analysis can be extended to any algorithm which follows Algorithm 1 so as long as we can control the bonus terms sufficiently well.

## C Experimental results

In this section we present experiments based on the following deterministic LP which can be found in Figure 2. In short the MDP has only deterministic transitions and 3 layers. The starting state is

Figure 2: Deterministic MDP used in experiments

<!-- image -->

4000

3500

3000

2500 -

З 2000 -

1500 1

1000

500

Return gap bound

Min-max regret bound

R(K),epsilon\_pow:0

R(K), epsilon\_ pow: 1

R(K),epsilon\_pow:3

R(K), epsilon\_pow:2

100000

200000

107

Return gap bound denoted by s 0 and the j -th state at layer i by s i,j . There are n +1 possible actions at s 0 , two possible actions at s 1 ,j , ∀ j ∈ [ n +1] , and a single possible action at s 2 ,j , ∀ j ∈ [4] . The only non-negative rewards are at state-action pairs in the final layer. The unique optimal policy reaches state s 2 , 1 and has return equal to 0 . 5 . We distinguish between two types of sub-optimal policies given by π 1 which visists s 1 , 1 and all other sub-optimal policies which visit s 1 ,j , j ≥ 2 . The return of policy π 1 determines the gap parameter in our experiments and the reward at state s 2 , 4 determines the glyph[epsilon1] parameter.

We run two sets of experiments using the UCBVI algorithm [4]. We have chosen this algorithm over Strong-EULER since UCBVI is slightly easier to implement and their differences are orthogonal to the issues studied here. The rewards in both experiments are Bernoulli with the respective mean provided below the state in Figure 2. In the first set of experiments we let the gap parameter to be equal to 0 . 5 and in the second set of experiments we let the gap parameter to be √ S K . We let glyph[epsilon1] = 4 glyph[epsilon1]pow √ K , where glyph[epsilon1] pow takes integer values between 0 and glyph[floorleft] 0 . 5 ∗ log 4 ( K ) glyph[floorright] . We have two settings for n (respectively S ) which are n = 1 and n = 250 . In all experiments we have set K = 500000 and the topology of the MDP implies H = 3 . Each experiment is repeated 5 times and we report the average regret of the algorithm, together with standard deviation of the regret. We note that in the first set of experiments we should observe regret which is close to Θ( SA log( T ) gap ) , this is because with our parameter choices the return gap is gap/ 2 for all settings of glyph[epsilon1] . In the second set of experiments we should observe regret which is close to Θ( √ SAK ) as the min-max regret bounds dominate.

Figure 3: Large gap experiments

<!-- image -->

The first set of experiments can be found in Figure 3. We plot S 2 A + SA log( T ) gap in purple and S 2 A + √ SAK in brown for reference. We include the additive term of S 2 A as this is what the theoretical regret bounds suggest. We see that for n = 1 our experiments almost perfectly match theory, including the observations made regarding Opportunity O.1 and Opportunity O.2 . In particular there is no obvious dependence on 1 / gap min = 1 /glyph[epsilon1] , especially when glyph[epsilon1] = O (1 / √ K ) , which in the plot is reflected by glyph[epsilon1] pow = 0 . In the case for n = 250 the algorithm performs better than what our theory suggests. We expect that our bounds do not accurately capture the dependence on S and A , at least for deterministic transition MDPs. The second set of experiments can be found in Figure 4. Similar observations hold as in the large gap experiment.

## D Additional Notation

We use the shorthand ( s, a ) ∈ π to indicate that π admits a non-zero probability of visiting the state-action pair ( s, a ) and abusively use π as the set of such state-action pairs, when convenient.

## E Proofs and extended discussion for regret lower-bounds

Let N ψ,π ( k ) be the random variable denoting the number of times policy π has been chosen by the strategy ψ . Let N ψ, ( s,a ) ( k ) be the number of times the state-action pair has been visited up to time k by the strategy ψ .

8000

7000 -

6000 1

5000

1 2000.

3000

2000

1000

Return gap bound

Min-max regret bound

R(K),epsilon\_pow:0

R(K),epsilon\_pow: 1

R(K), epsilon\_ pow:2

R(K),epsilon\_pow: 3

100000

200000

K

300000

107

31654

Return gap bound

Figure 4: Small gap experiments

<!-- image -->

## E.1 Lower bound as an optimization problem

We begin by formulating an LP characterizing the minimum regret incurred by any uniformly good algorithm ψ.

Theorem E.1. Let ψ be a uniformly good RL algorithm for Θ , that is, for all problem instances θ ∈ Θ and exponents α &gt; 0 , the regret of ψ is bounded as E [ R θ ( K )] ≤ o ( K α ) . Then, for any θ ∈ Θ , the regret of ψ satisfies

<!-- formula-not-decoded -->

where C ( θ ) is the optimal value of the following optimization problem

<!-- formula-not-decoded -->

where Λ ′ ( θ ) = { λ ∈ Θ: Π ∗ λ ∩ Π ∗ θ = ∅ , KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 } are all environments that share no optimal policy with θ and do not change the rewards or transition kernel on π ∗ .

Proof. We can write the expected regret as E [ R θ ( K )] = ∑ π ∈ Π E θ [ N ψ,π ( K )]( v ∗ θ -v π θ ) . We will show that η ( π ) = E θ [ N ψ,π ( K )] / log K is feasible for the optimization problem in (8). This is sufficient to prove the theorem. To do so we follow the techniques of [15]. With slight abuse of notation, let P I k θ be the law of all trajectories up to episode k , where I k is the history up to and including time k . Let Y k be the random variable which is the value function of the policy, ψ ( I k ) , selected at episode k . We have

<!-- formula-not-decoded -->

Iterating the argument we arrive at ∑ π ∈ Π E θ [ N ψ,π ( K )] KL ( P π θ , P π λ ) = KL ( P I K θ , P I K λ ) where E θ denotes expectation in problem instance θ . Next one shows that for any measurable Z ∈ [0 , 1] , with respect to the natural sigma-algebra induced by I K , it holds that KL ( P I K θ , P I K λ ) ≥ kl ( E θ [ Z ] , E λ [ Z ]) where kl ( p, q ) = p log p/q +(1 -p ) log (1 -p ) / (1 -q ) denotes the KL-divergence between two Bernoulli random variables p and q . This follows directly from Lemma 1 by Garivier et al. [15]. Finally we choose Z = N ψ, Π ∗ λ ( K ) /K as the fraction of episodes where an optimal policy for λ

was played (here we use the short-hand notation N ψ, Π ∗ λ ( K ) = ∑ π ∈ Π ∗ λ N ψ,π ( K ) ). Evaluating the kl -term we have

<!-- formula-not-decoded -->

Since ψ is a uniformly good algorithm it follows that for any α &gt; 0 , K -E λ [ N ψ, Π ∗ λ ( K )] = o ( K α ) . By assuming that Π ∗ θ ∩ Π ∗ λ = ∅ , we get E θ [ N ψ, Π ∗ λ ( K )] = o ( K ) . This implies that for K sufficiently large and all 1 ≥ α &gt; 0

<!-- formula-not-decoded -->

The set Λ ′ ( θ ) is uncountably infinite for any reasonable Θ we consider. What is worse the constraints of LP 8 will not form a closed set and thus the value of the optimization problem will actually be obtained on the boundary of the constraints. To deal with this issue it is possible to show the following.

Theorem 4.2 (General instance-dependent lower bound for episodic MDPs) . Let ψ be a uniformly good RL algorithm for Θ , that is, for all problem instances θ ∈ Θ and exponents α &gt; 0 , the regret of ψ is bounded as E [ R θ ( K )] ≤ o ( K α ) , and assume that v ∗ θ &lt; H . Then, for any θ ∈ Θ , the regret of ψ satisfies

<!-- formula-not-decoded -->

where C ( θ ) is the optimal value of the following optimization problem

<!-- formula-not-decoded -->

Proof. For the rest of this proof we identify Λ ′ ( θ ) = { λ ∈ Θ : Π ∗ λ ∩ Π ∗ θ = ∅ , KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 , ∀ π ∗ θ ∈ Π ∗ θ } as the set from Theorem E.1 and ˜ Λ( θ ) = { λ ∈ Θ : v π ∗ λ λ ≥ v π ∗ θ θ , π ∗ λ glyph[negationslash]∈ Π ∗ θ , KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 } . From the proof of Theorem E.1 it is clear that we can rewrite Λ ′ ( θ ) as the union ⋃ π ∈ Π Λ π ( θ ) , where Λ π ( θ ) = { λ ∈ Θ : KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 , v π ∗ λ &gt; v π ∗ θ θ , π ∗ λ = π } is the set of all environments which make π the optimal policy. This implies that we can equivalently write LP 8 as

<!-- formula-not-decoded -->

The above formulation now minimizes a linear function over a finite intersection of sets, however, these sets are still slightly inconvenient to work with. We are now going to try to make these sets more amenable to the proof techniques we would like to use for deriving specific lower bounds. We begin by noting that Λ π ( θ ) is bounded in the following sense. We identify each λ with a vector in [0 , 1] S 2 A × [0 , 1] SA where the first S 2 A coordinates are transition probabilities and the last SA coordinates are the expected rewards. From now on we work with the natural topology on [0 , 1] S 2 A × [0 , 1] SA , induced by the glyph[lscript] 1 norm. Further, we claim that we can assume that KL ( P π θ , P π λ ) is a continuous function over Λ π ′ ( θ ) . The only points of discontinuity are at λ for which the support of the transition kernel induced by λ does not match the support of the transition kernel induced by θ . At such points the KL ( P π θ , P π λ ) = ∞ . This implies that such λ does not achieve the infimum in the set of constraints so we can just restrict Λ π ′ ( θ ) to contain only λ for which KL ( P π θ , P π λ ) &lt; ∞ . With this restriction in hand the KL-divergence is continuous in λ .

Fix a π ′ and consider the set { η : inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ 1 } corresponding to one of the constraints in LP 13. Denote ˜ Λ π ′ ( θ ) = { λ ∈ Θ : KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 , v π ∗ λ λ ≥ v π ∗ θ θ , π ∗ λ glyph[negationslash]∈ Π ∗ θ , π ∗ λ = π ′ } . ˜ Λ π ′ ( θ ) is closed as KL ( P π ∗ θ θ , P π ∗ θ λ ) and v π ∗ λ λ -v π ∗ θ θ are both continuous in λ . To see the statement for v π ∗ λ λ , notice that this is the maximum over the continuous functions v π λ over π ∈ Π . Take any η ∈ Λ π ′ ( θ ) and let { λ j } ∞ j =1 , λ j ∈ Λ π ′ ( θ ) be a sequence of environments such that ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ j ) ≥ 1 + 2 -j . If there is no convergent subsequence of { λ j } ∞ j =1 in Λ π ′ ( θ ) we claim it is because of the constraint v π ∗ λ λ &gt; v π ∗ θ θ . Take the limit λ of any convergent subsequence of { λ j } ∞ j =1 in the closure of Λ π ′ ( θ ) . Then by continuity of the divergence we have 0 = lim j →∞ KL ( P π ∗ θ θ , P π ∗ θ λ j ) = KL ( P π ∗ θ θ , P π ∗ θ λ ) , thus it must be the case that v π ∗ λ λ ≤ v π ∗ θ θ . This shows that ˜ Λ π ′ ( θ ) is a subset of the closure of Λ π ′ ( θ ) which implies it is the closure of Λ π ′ ( θ ) , i.e., ¯ Λ π ′ ( θ ) = ˜ Λ π ′ ( θ ) .

Next, take η ∈ { η : min λ ∈ ¯ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ 1 } and let λ π ′ ,η be the environment on which the minimum is achieved. Such λ π ′ ,η exists because we just showed that ¯ Λ π ′ ( θ ) is closed and bounded and hence compact and the sum consists of a finite number of continuous functions. If λ π ′ ,η ∈ Λ π ′ ( θ ) then η ∈ { η : inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ 1 } . If λ π ′ ,η glyph[negationslash]∈ Λ π ′ ( θ ) then λ π ′ ,η must be a limit point of Λ π ′ ( θ ) . By definition we can construct a convergent sequence of { λ j } ∞ j =1 , λ j ∈ Λ π ′ ( θ ) to λ π ′ ,η such that ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ j ) ≥ 1 . This implies ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ j ) ≥ inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) . Using the continuity of the KL term and taking limits, the above implies that the minimum upper bounds the infimum. Since we argued that Λ π ′ ( θ ) is bounded and ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ j ) is also bounded from below this implies ¯ Λ π ′ ( θ ) contains the infimum inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) . This implies inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ min λ ∈ ¯ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) , and so the infimum over Λ π ( θ ) equals the minimum over ¯ Λ π ( θ ) . Which finally implies that η ∈ { η : inf λ ∈ Λ π ′ ( θ ) ∑ π ∈ Π η ( π ) KL ( P π θ , P π λ ) ≥ 1 } . This shows that LP 13 is equivalent to

<!-- formula-not-decoded -->

or equivalently that we can consider the closure of Λ( θ ) in LP 8, ¯ Λ( θ ) = { λ ∈ Θ: v π ∗ λ λ ≥ v π ∗ θ θ , π ∗ λ glyph[negationslash]∈ Π ∗ θ , KL ( P π ∗ θ θ , P π ∗ θ λ ) = 0 } i.e. the set of environments which makes any π optimal without changing the environment on state-action pairs in π ∗ θ .

## E.2 Lower bounds for full support optimal policy

Lemma 4.3. Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1 and let θ ∈ Θ be an MDP in this class. Then for any suboptimal state-action pair ( s, a ) with gap θ ( s, a ) &gt; 0 such that s is visited by some optimal policy with non-zero probability, there exists a confusing MDP λ ∈ Λ( θ ) with

- λ and θ only differ in the immediate reward at ( s, a )
- KL ( P π θ , P π λ ) ≤ gap θ ( s, a ) 2 for all π ∈ Π .

Proof. Let λ be the environment that is identical to θ except for the immediate reward for state-action pair for ( s, a ) . Specifically, let R λ ( s, a ) so that r λ ( s, a ) = r θ ( s, a ) +∆ with ∆ = gap θ ( s, a ) . Since we assume that rewards are Gaussian, it follows that

<!-- formula-not-decoded -->

for any policy π ∈ Π . We now show that the optimal value function (and thus return) of λ is uniformly upper-bounded by the optimal value function of θ . To that end, consider their difference in any state

s ′ , which we will upper-bound by their difference in s as

<!-- formula-not-decoded -->

Further, the difference in s is exactly

<!-- formula-not-decoded -->

Hence, V ∗ λ = V ∗ θ ≤ 1 and thus λ ∈ Θ . We will now show that there is a policy that is optimal in λ but not in θ . Let π ∗ ∈ Π ∗ θ be any optimal policy for θ that has non-zero probability of visiting s and consider the policy glyph[negationslash]

<!-- formula-not-decoded -->

that matches π ∗ on all states except s . We will now show that ˜ π achieves the same return as π ∗ in λ . Consider their difference

<!-- formula-not-decoded -->

where ( i ) and ( ii ) follow from the fact that ˜ π and π ∗ only differ on s and hence, their probability at arriving at s and their value for any successor state of s is identical. Step ( iii ) follows from the fact that λ and θ only differ on ( s, a ) which is not visited by π ∗ . Finally, step ( iv ) applies the definition of optimal value functions and value-function gaps. Since ∆ = gap θ ( s, ˜ π ( s )) , it follows that v ˜ π λ = v π ∗ λ = v π ∗ θ = v ∗ θ . As we have seen above, the optimal value function (and return) is identical in θ and λ and, hence, ˜ π is optimal in λ .

Note that the we can apply the chain of equalities above in the same manner to v ˜ π θ -v π ∗ θ if we consider ∆ = 0 . This yields

<!-- formula-not-decoded -->

because w π ∗ θ ( s, π ∗ ( s )) &gt; 0 and gap θ ( s, a ) &lt; 0 by assumption. Hence ˜ π is not optimal in θ , which completes the proof.

Lemma E.2 (Optimization problem over S × A instead of Π ) . Let optimal value C ( θ ) of the optimization problem (8) in Theorem 4.2 is lower-bound by the optimal value of the problem

<!-- formula-not-decoded -->

Proof. First, we rewrite the objective of (8) as

<!-- formula-not-decoded -->

where step ( i ) applies Lemma F.1 proved in Appendix F. Here, w π θ ( s, a ) is the probability of reaching s and taking a when playing policy π in MDP θ . Similarly, the LHS of the constraints of (8) can be decomposed as

<!-- formula-not-decoded -->

where the first equality follows from writing out the definition of the KL divergence. Let now η ( π ) be a feasible solution to the original problem (8). Then the two equalities we just proved show that η ( s, a ) = ∑ π ∈ Π η ( π ) w π θ ( s, a ) is a feasible solution for the problem in (14) with the same value. Hence, since (14) is a minimization problem, its optimal value cannot be larger than C ( θ ) , the optimal value of (8).

Theorem 4.4 (Gap-dependent lower bound when optimal policies visit all states) . Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1. Let θ ∈ Θ be an instance where every state is visited by some optimal policy with non-zero probability. Then any uniformly good algorithm on Θ has expected regret on θ that satisfies

<!-- formula-not-decoded -->

Proof. Let ¯ Λ( θ ) be a set of all confusing MDPs from Lemma 4.3, that is, for every suboptimal ( s, a ) , ¯ Λ( θ ) contains exactly one confusing MDP that differs with θ only in the immediate reward at ( s, a ) . Consider now the relaxation of Theorem 4.2 from Lemma E.2 and further relax it by reducing the set of constraints induced by Λ( θ ) to only the set of constraints induced by ¯ Λ( θ ) :

<!-- formula-not-decoded -->

Since all confusing MDPs only differ in rewards, we dropped the KL-term for the transition probabilities. We can simplify the constraints by noting that for each λ , only one KL-term is non-zero and it has value gap θ ( s, a ) 2 . Hence, we can write the problem above equivalently as

<!-- formula-not-decoded -->

Rearranging the constraint as η ( s, a ) ≥ 1 / gap θ ( s, a ) 2 , we see that the value is lower-bounded by

<!-- formula-not-decoded -->

which completes the proof.

We note that because the relaxation in Lemma E.2 essentially allows the algorithm to choose which state-action pairs to play instead of just policies, the final lower bound in Theorem 4.4 may be loose, especially in factors of H . However, it is unlikely that the gap min term arising in the upper bound of Simchowitz and Jamieson [30] can be recovered. We conjecture that such a term can be avoided by algorithms, which do not construct optimistic estimators for the Q -function at each state-action pair but rather just work with a class of policies and construct only optimistic estimators of the return.

## E.3 Lower bounds for deterministic MDPs

We will show that we can derive lower bounds in two cases:

1. We show that if the graph induced by the MDP is a tree, then we can formulate a finite LP which has value at most a polynomial factor of H away from the value of LP 8.
2. We show that if we assume that the value function for any policy is at most 1 and the rewards of each state-action pair are at most 1 , then we can derive a closed form lower bound. This lower bound is also at most a polynomial factor of H away from the solution to LP 8.

We begin by stating a helpful lemma, which upper and lower bounds the KL -divergence between two environments on any policy π . Since we consider Gaussian rewards with σ = 1 / √ 2 it holds that KL ( R θ ( s, a ) , R λ ( s, a )) = ( r θ ( s, a ) -r λ ( s, a )) 2 . Further for any π and λ it holds that KL ( θ ( π ) , λ ( π )) = ∑ ( s,a ) ∈ π KL ( R θ ( s, a ) , R λ ( s, a )) = ∑ ( s,a ) ∈ π ( r θ ( s, a ) -r λ ( s, a )) 2 . We can now show the following lower bound on KL ( θ ( π ) , λ ( π )) .

Lemma E.3. Fix π and suppose λ is such that π ∗ λ = π . Then ( v ∗ -v π ) 2 ≥ KL ( θ ( π ) , λ ( π )) ≥ ( v ∗ -v π ) 2 H .

Proof. The second inequality follows from the fact that the optimization problem

<!-- formula-not-decoded -->

admits a solution at θ, λ for which r λ ( s, a ) -r θ ( s, a ) = v ∗ -v π H , ∀ ( s, a ) ∈ π . The first inequality follows from considering the optimization problem

<!-- formula-not-decoded -->

and the fact that it admits a solution at θ, λ for which there exists a single state-action pair ( s, a ) ∈ π such that r θ ( s, a ) -r λ ( s, a ) = v ∗ -v π and for all other ( s, a ) it holds that r λ ( s, a ) = r θ ( s, a ) .

glyph[negationslash]

Using the above Lemma E.3 we now show that we can restrict our attention only to environments λ ∈ Λ( θ ) which make one of π ∗ ( s,a ) optimal and derive an upper bound on C ( θ ) which we will try to match, up to factors of H , later. Define the set ˜ Λ( θ ) = { λ ∈ Λ( θ ) : ∃ ( s, a ) ∈ S × A , π ∗ λ = π ∗ ( s,a ) } and Π ∗ = { π ∈ Π , π = π ∗ θ : ∃ ( s, a ) ∈ S × A , π = π ∗ ( s,a ) } . We have

Lemma E.4. Let ˜ C ( θ ) be the value of the optimization problem

<!-- formula-not-decoded -->

Then ∑ π ∈ Π ∗ H v ∗ -v π ≥ C ( θ ) ≥ ˜ C ( θ ) H .

Proof. We begin by showing C ( θ ) ≥ ˜ C ( θ ) H holds. Fix a π glyph[negationslash]∈ Π ∗ s.t. the solution of LP 8 implies η ( π ) &gt; 0 . Let λ ∈ ˜ Λ( θ ) be a change of environment for which KL ( θ ( π ) , λ ( π )) &gt; 0 . We can now shift all of the weight of η ( π ) to η ( π ∗ λ ) while still preserving the validity of the constraint. Further doing so to all π ∗ ( s,a ) for which π ∗ ( s,a ) ∩ π = ∅ will not increase the objective by more than a factor of H as v ∗ -v π ≥ 1 H ∑ ( s,a ) ∈ π v ∗ -v π ∗ ( s,a ) . Thus, we have converted the solution to LP 8 to a feasible solution to LP 15 which is only a factor of H larger.

glyph[negationslash]

Next we show that ∑ π ∈ Π ∗ H v ∗ -v π ≥ C ( θ ) . Set η ( π ) = 0 , ∀ π ∈ Π \ Π ∗ and set η ( π ) = H ( v ∗ -v π ) 2 , ∀ π ∈ Π ∗ . If π is s.t. η ( π ) &gt; 0 then for any λ which makes π optimal it holds that

<!-- formula-not-decoded -->

where the second inequality follows from Lemma E.3. Next, if π is s.t. η ( π ) = 0 then for any λ which makes π optimal it holds that

<!-- formula-not-decoded -->

π ∗ π ∗ ∗

where the second inequality follows from the fact that v λ ≤ v ( s,a ) , ∀ ( s, a ) ∈ π λ .

## E.3.1 Lower bound for Markov decision processes with bounded value function

Lemma E.5. Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1 . Consider an MDP θ ∈ Θ with deterministic transitions. Then, for any reachable state-action pair ( s, a ) that is not visited by any optimal policy, there exists a confusing MDP λ ∈ Λ( θ ) with

- λ and θ only differ in the immediate reward at ( s, a )

<!-- formula-not-decoded -->

Proof. Let ( s, a ) ∈ S × A be any state-action pair that is not visited by any optimal policy. Then v π ∗ ( s,a ) θ = max π : w π ( s,a ) &gt; 0 v π θ ≤ v ∗ θ is strictly suboptimal in θ . Let ˜ π be any policy that visits ( s, a ) and achieves the highest return v π ∗ ( s,a ) θ in θ possible among such policies.

Define λ to be the MDP that matches θ except in the immediate reward at ( s, a ) , which we set as R λ ( s, a ) = N ( r θ ( s, a ) + ∆ , 1 / 2) with ∆ = v ∗ θ -v π ∗ ( s,a ) θ . That is, the expected reward of λ in ( s, a ) is raised by ∆ . For any policy π , it then holds

<!-- formula-not-decoded -->

due to the deterministic transitions. Hence, while v ∗ λ = v ∗ θ and all optimal policies of θ are still optimal in λ , now policy ˜ π , which is not optimal in θ is optimal in λ .

By the choice of Gaussian rewards with variance 1 / 2 , we have KL ( R θ ( s, a ) , R λ ( s, a )) = ( v ∗ θ -v π ∗ ( s,a ) θ ) 2 and thus KL ( P π θ , P π λ ) = w π θ ( s, a )( v ∗ θ -v π ∗ ( s,a ) θ ) 2 for all π ∈ Π .

It only remains to show that λ ∈ Θ , i.e., that all immediate rewards and optimal value function is bounded by 1 . For rewards, we have

<!-- formula-not-decoded -->

for ( s, a ) and for all other ( s ′ , a ′ ) , r λ ( s ′ , a ′ ) = r θ ( s ′ , a ′ ) ≤ 1 . Finally, the value function at any reachable state is bounded by the optimal return v ∗ λ = v ∗ θ ≤ 1 and for any unreachable state, the optimal value function of λ is identical to the optimal value function of θ . Hence, λ ∈ Θ .

Theorem 4.5. Let Θ be the set of all episodic MDPs with Gaussian immediate rewards and optimal value function uniformly bounded by 1. Let θ ∈ Θ be an instance with deterministic transitions. Then any uniformly good algorithm on Θ has expected regret on θ that satisfies

<!-- formula-not-decoded -->

where Z θ = { ( s, a ) ∈ S × A : ∀ π ∗ ∈ Π ∗ θ w π ∗ θ ( s, a ) = 0 } is the set of state-action pairs that no optimal policy in θ visits.

Proof. The proof works by first relaxing the general LP 8 and then considering its dual. We now define the set ˇ Λ( θ ) which consists of all changes of environment which make π ∗ ( s,a ) optimal by only changing the distribution of the reward at ( s, a ) by making it v ∗ θ -v π ∗ ( s,a ) θ larger. Formally, the set is defined as glyph[negationslash]

<!-- formula-not-decoded -->

This set is guaranteed to be non-empty (for any reasonable MDP) by Lemma E.5. The relaxed LP is now give by

<!-- formula-not-decoded -->

The dual of the above LP is given by

<!-- formula-not-decoded -->

By weak duality, the value of any feasible solution to (17) produces a lower bound on C ( θ ) in Theorem 4.2. Let

<!-- formula-not-decoded -->

be the set of state-action pairs that are reachable in θ but no optimal policy visits. Then consider a dual solution µ that puts 0 on all confusing MDPs except on the |X| many MDPs from Lemma E.5. Since each such confusing MDP is associated with an ( s, a ) ∈ X , we can rewrite µ as a mapping from X to R sending ( s, a ) → λ ( s,a ) . Specifically, we set

<!-- formula-not-decoded -->

To show that this µ is feasible, consider the LHS of the constraints in (17)

<!-- formula-not-decoded -->

where the first equality applies our definition of µ and the second uses the expression for the KLdivergence from Lemma E.5. By definition of v π ∗ ( s,a ) θ , we have v π ∗ ( s,a ) θ ≥ v π θ for all policies π with w π θ ( s, a ) &gt; 0 . Thus,

<!-- formula-not-decoded -->

where the second inequality holds because each policy visits at most H states. Thus proves that µ defined above is indeed feasible. Hence, its objective value

<!-- formula-not-decoded -->

is a lower-bound for C ( θ ) from Theorem 4.2 which finishes the proof.

## E.3.2 Tree-structured MDPs

Even though Lemma E.4 restricts the set of confusing environments from Λ( θ ) to ˜ Λ( θ ) , this set could still have exponential or even infinite cardinality. In this section we show that for a type of special MDPs we can restrict ourselves to a finite subset of ˜ Λ( θ ) of size at most SA .

Arrange π ∗ ( s,a ) , ( s, a ) ∈ S × A according to the value functions v π ∗ ( s,a ) . Under this arrangement let π 1 glyph[followsequal] π 2 glyph[followsequal] , . . . , glyph[followsequal] π m . Let π 0 = π ∗ θ . We will now construct m environments λ 1 , . . . , λ m , which will constitute the finite subset. We begin by constructing λ 1 as follows. Let B 1 be the set of all ( s h , a h ) ∈ π 1 and ( s h , a h ) glyph[negationslash]∈ π 0 . Arrange the elements in B 1 in inverse dependence on horizon ( s h 1 , a h 1 ) glyph[precedesequal] ( s h 2 , a h 2 ) glyph[precedesequal] . . . glyph[precedesequal] ( s h H 1 , a h H 1 ) , where H 1 = |B 1 | , so that h 1 &gt; h 2 &gt;,. . . , h H 1 . Let λ 1 be the environment which sets

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly λ 1 makes π 1 optimal and also does not change the value of any state-action pair which belongs to π 0 so it agrees with θ on π 0 . Further π 2 , π 3 , . . . , π m are still suboptimal policies under λ 1 . This follows from the fact that for any i &gt; 1 , v π 1 &gt; v π i and there exists ( s, a ) such that ( s, a ) ∈ π i but ( s, a ) glyph[negationslash]∈ π 1 so R λ 1 ( s, a ) = R θ ( s, a ) . Further λ 1 only increases the rewards for state-action pairs in π 1 and hence v π 1 λ 1 &gt; v π i λ 1 . Notice that there exists an index ˜ H 1 at which R λ 1 ( s h ˜ H 1 , a h ˜ H 1 ) = v π 0 -( v π 1 -∑ ˜ H 1 glyph[lscript] =1 R θ ( s h glyph[lscript] , a h glyph[lscript] )) -˜ H 1 ) ≥ R θ ( a ˜ H 1 , s ˜ H 1 ) . For this index it holds that for h &lt; ˜ H 1 , R λ 1 ( s h , a h ) = 1 and for h &gt; ˜ H 1 , R λ 1 ( s h , a h ) = R θ ( s h , a h ) .

Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first define an environment ˜ λ i on ( s, a ) ∈ ˜ B i as follows. R λ i ( s, a ) = R λ glyph[lscript] ( s, a ) , where glyph[lscript] &lt; i is such that ( s, a ) ∈ B glyph[lscript] . Let v π i ˜ λ i be the value function of π i with respect to ˜ λ i .

Lemma E.6. It holds that v π i ˜ λ i ≤ v π 0 .

glyph[negationslash]

Proof. Let ˜ H i be the index for which it holds that for glyph[lscript] ≤ ˜ H i , ( s h glyph[lscript] , a h glyph[lscript] ) ∈ π i ⇐⇒ ( s h glyph[lscript] , a h glyph[lscript] ) ∈ B i . Such a ˜ H i exists as there is a unique sub-tree M i , of maximal depth, for which it holds that if π j ⋂ M i = ∅ ⇐⇒ π i glyph[followsequal] π j . The root of this subtree is exactly at depth H -h ˜ H i . Let π j be any policy such that π j glyph[followsequal] π i and ∃ ( s h ˜ Hi , a h ˜ Hi ) ∈ π j . By the maximality of M i such a π j exists. Because of the tree structure it holds that for any h ′ &gt; h ˜ H i if ( s h ′ , a h ′ ) ∈ π i = ⇒ ( s h ′ , a h ′ ) ∈ π j and hence ˜ λ i = λ j up to depth h ˜ H i . Since π i and π j match up to depth H -h ˜ H i and π j glyph[followsequal] π i it also holds that

<!-- formula-not-decoded -->

Since π j is optimal under λ j the claim holds.

For all ( s h j , a h j ) ∈ B i we now set

<!-- formula-not-decoded -->

and for all ( s h , a h ) ∈ ˜ B i we set R λ i ( s h , a h ) = R ˜ λ i ( s h , a h ) . From the definition of ˜ B i it follows that λ i agrees with all λ j for j ≤ i on state-action pairs in π i . Finally we need to show that the construction in Equation 18 yields an environment λ i for which π i is optimal.

Lemma E.7. Under λ i it holds that π i is optimal.

Proof. Let ˜ H i and π j be as in the proof of Lemma E.6. We now show that ∑ glyph[lscript] ≤ ˜ H i R λ j ( s π j h glyph[lscript] , a π j h glyph[lscript] ) ≤ ∑ glyph[lscript] ≤ ˜ H i R λ i ( s π i h glyph[lscript] , a π i h glyph[lscript] ) . We only need to show that ∑ glyph[lscript] ≤ ˜ H i R λ i ( s π i h glyph[lscript] , a π i h glyph[lscript] ) ≥ v π 0 -v π i ˜ λ i . From Equation 18 we have R λ i ( s h 1 , a h 1 ) = min(1 , v π 0 -v π i ˜ λ i ) . If R λ i ( s h 1 , a h 1 ) = v π 0 -v π i ˜ λ i then the claim is complete. Suppose R λ i ( s h 1 , a h 1 ) = 1 . This implies v π 0 -v π i ˜ λ i ≥ 1 -R θ ( s h 1 , a h 1 ) . Next the construction adds the remaining gap of v π 0 -v π i ˜ λ i + R θ ( s h 1 , a h 1 ) -1 to R θ ( s h 2 , a h 2 ) and clips R λ i ( s h 2 , a h 2 ) to 1 if necessary. Continuing in this way we see that if ever R λ i ( s h j , a h j ) = R θ ( s h j , a h j ) + v π 0 -( v π i ˜ λ i -∑ j glyph[lscript] =1 R ˜ λ i ( s h glyph[lscript] , a h glyph[lscript] )) -j then v π 0 -V π i ˜ λ i ≤ ∑ glyph[lscript] ≤ ˜ H i R λ i ( s π i h glyph[lscript] , a π i h glyph[lscript] ) . On the other hand if this never occurs, we must have R λ i ( s π i h glyph[lscript] , a π i h glyph[lscript] ) = 1 ≥ R λ j ( s π j h glyph[lscript] , a π j h glyph[lscript] ) which concludes the claim.

Let ˆ Λ( θ ) = { λ 1 , . . . , λ m } be the set of the environments constructed above. We now show that the value of the optimization problem is not too much smaller than the value of Problem 8.

## Theorem E.8. The value ˆ C ( θ ) of the LP

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The inequality C ( θ ) ≥ ˆ C ( θ ) H follows from Lemma E.4 and the fact that the above optimization problem is a relaxation to LP 15.

To show the first inequality we consider the following relaxed LP

<!-- formula-not-decoded -->

Figure 5: Issue with restricting LP to Π ∗

<!-- image -->

Any solution to the LP in the statement of the theorem is feasible for the above LP and thus the value of the above LP is no larger. We now show that the value of the above LP is greater than or equal to C ( θ ) H 2 . Fix λ ∈ ˆ Λ( θ ) . We show that for any λ ′ ∈ Λ( θ ) such that π ∗ λ = π ∗ λ ′ it holds that KL ( θ ( π ) , λ ( π )) ≤ H 2 KL ( θ ( π ) , λ ′ ( π )) , ∀ π ∈ Π . This would imply that if η is a solution to the above LP, then H 2 η is feasible for LP 8 and therefore ˆ C ( θ ) ≥ C ( θ ) H 2 .

Arrange π ∈ Π : KL ( θ ( π ) , λ ( π )) &gt; 0 according to KL ( θ ( π ) , λ ( π )) so that

<!-- formula-not-decoded -->

Consider the optimization problem

<!-- formula-not-decoded -->

If we let ∆ λ ′ ( s h , a h ) , ( s h , a h ) ∈ π ∗ λ denote the change of reward for ( s h , a h ) under environment λ ′ , then the above optimization problem can be equivalently written as

<!-- formula-not-decoded -->

It is easy to see that the solution to the above optimization problem is to set r ( s h , a h )+∆ λ ′ ( s h , a h ) = 1 for all h ∈ [ h ˜ H i +1 , H ] and spread the remaining mass of v ∗ -˜ H i -( v π ∗ λ -∑ ˜ H i glyph[lscript] =1 ) R θ ( s h glyph[lscript] , a h glyph[lscript] ) as uniformly as possible on ∆ λ ′ ( s h , a h ) , h ∈ [1 , h ˜ H i ] . Notice that under this construction the solution to the above optimization problem and λ match for h ∈ [ h ˜ H i +1 , H ] . Since the remaining mass is now the same it now holds that for any λ ′ , ∑ h ˜ Hi h =1 ∆ λ ′ ( s h , a h ) 2 ≥ 1 h 2 ˜ Hi ∑ h ˜ Hi h =1 ∆ λ ( s h , a h ) 2 . This implies KL ( θ ( π i ) , λ ′ ( π i )) ≥ 1 ˜ H 2 i KL ( θ ( π ) , λ ( π )) and the result follows as ˜ H i ≤ H, ∀ i ∈ [ H ] .

## E.3.3 Issue with deriving a general bound

We now try to give some intuition regarding why we could not derive a generic lower bound for deterministic transition MDPs. We have already outlined our general approach of restricting the set Π and Λ( θ ) to finite subsets of manageable size and then showing that the value of the LP on these restricted sets is not much smaller than the value of the original LP. One natural restriction of Π is the set Π ∗ from Theorem 4.5. Suppose we restrict ourselves to the same set and consider only environments making policies in Π ∗ optimal as the restriction for Λ( θ ) . We now give an example of an MDP for which such a restriction will lead to an Ω( SA ) multiplicative discrepancy between the value of the original semi-infinite LP and the restricted LP. The MDP can be found in Figure 5. The rewards for each action for a fixed state s are equal and are shown in the vertices corresponding to the states. The number of states in the second and last layer of the MDP are equal to ( SA -3) / 2 .

The optimal policy takes the red path and has value V π ∗ = 3 . The set Π ∗ consists of all policies π j,i which visit one of the states in green. The policies π 1 ,i , in blue, visit the green state in the second layer of the MDP and one of the states in the final layer, following the paths in blue. Similarly the policies π 2 ,i , in orange, visit one of the state in the second layer and the green state in the last layer, following the orange paths. The value function of π j,i is V π j,i = 3 -3 SA -iglyph[epsilon1] , where 0 ≤ i ≤ ( SA -4) / 2 . We claim that playing each π j,i η ( π j,i ) = Ω( SA ) times is a feasible solution to the LP restricted to Π ∗ . Fix i , the λ π 1 ,i must put weight at least 1 /SA on the green state in layer 2. Coupling with the fact that for all i ′ the rewards π 1 ,i ′ are also changed under this environment we know that the constraint of the restricted LP with respect to λ π 1 ,i is lower bounded by ∑ i ′ η ( π 1 ,i ′ ) / ( SA ) 2 . Since there are Ω( SA ) policies { π 1 ,i ′ } i ′ , this implies that η ( π 1 ,i ) = Ω( SA ) is feasible. A similar argument holds for any π 2 ,i . Thus the value of the restricted LP is at most O ( SA ) , for any glyph[epsilon1] glyph[lessmuch] SA .

However, we claim that the value of the semi-infinite LP which actually characterizes the regret is at least Ω( S 2 A 2 ) . First, to see that the above assignment of η is not feasible for the semi-infinite LP, consider any policy π glyph[negationslash]∈ Π ∗ , e.g. take the policy which visits the state in layer 2 with reward 1 -1 /SA -glyph[epsilon1] and the state in layer 4 with reward 1 -2 /SA -glyph[epsilon1] . Each of these states have been visited O ( SA ) times and η ( π ) = 0 hence the constraint for the environment λ π is upper bounded by SA ( ( 1 SA + glyph[epsilon1] ) 2 + ( ( 2 SA + glyph[epsilon1] ) 2 )) ≈ 1 /SA . In general each of the states in black in the second layer and the fourth layer have been visited 1 /SA times less than what is necessary to distinguish any π glyph[negationslash]∈ Π ∗ as sub-optimal. If we define the i -th column of the MDP as the pair consisting of the states with rewards 1 -1 /SA -iglyph[epsilon1] and 1 -2 /SA -iglyph[epsilon1] then to distinguish the policy visiting both of these states as sub-optimal we need to visit at least one of these Ω( S 2 A 2 ) times. This implies we need to visit each column of the MDP Ω( S 2 A 2 ) times and thus any strategy must incur regret at least Ω (∑ i S 2 A 2 1 SA ) = Ω( S 2 A 2 ) , leading to the promised multiplicative gap of Ω( SA ) between the values of the two LPs.

Why does such a gap arise and how can we hope to fix it this issue? Any feasible solution to the LP restricted to Π ∗ essentially needs to visit the states in green Θ( S 2 A 2 ) times. This is sufficient to distinguish the green states as sub-optimal to visit and hence any strategy visiting these states would be also deemed sub-optimal. This is achievable by playing each strategy in Π ∗ in the order of Θ( SA ) times as already discussed. Now, even though Π ∗ covers all other states, from our argument above we see that we need to play each π ∈ Π ∗ in the order of Θ( S 2 A 2 ) times to be able to determine all sub-optimal states. To solve this issue, we either have to increase the size of Π ∗ to include for example all policies visiting each column of the MDP or at the very least include changes of environments in the constraint set which make such policies optimal. This is clearly computationally feasible for the MDP in Figure 5, however, it is not clear how to proceed for general MDPs, without having to include exponentially many constraints. This begs the question about the computational hardness of achieving both upper and lower regret bounds in a factor of o ( SA ) from what is optimal.

## E.4 Lower bounds for optimistic algorithms in MDPs with deterministic transitions

In this section we prove a lower bound on the regret of optimistic algorithms, demonstrating that optimistic algorithms can not hope to achieve the information-theoretic lower bounds even if the MDPs have deterministic transitions. While the result might seem similar to the one proposed by Simchowitz and Jamieson [30] (Theorem 2.3) we would like to emphasize that the construction of Simchowitz and Jamieson [30] does not apply to MDPs with deterministic transitions, and that the idea behind our construction is significantly different.

Consider the MDP in Figure 6. This MDP has 2 n +9 states and 4 n +8 actions. The rewards for each action are either 1 / 12 or 1 / 12 + glyph[epsilon1]/ 2 and can be found next to the transitions from the respective states. We are going to label the states according to their layer and their position in the layer so that the first state is s 1 , 1 the state which is to the left of s 1 , 1 in layer 2 is s 2 , 1 and to the right s 2 , 2 . In general the i -th state in layer h is denoted as s h,i . The rewards in all states are deterministic, with a single exception of a Bernoulli reward from state s 4 , 1 to s 5 , 2 with mean 1 / 12 . From the construction it is clear that V ∗ ( s 1 , 1 ) = 1 / 2 + glyph[epsilon1] . Further there are two sets of optimal policies with the above value function - the n optimal policies which visit state s 2 , 2 and the n optimal policies which visit s 5 , 1 . Notice that the information-theoretic lower bound for this MDP is in O (log( K ) /glyph[epsilon1] ) as only the transition from state s 4 , 1 to s 5 , 2 does not belong to an optimal policy. In particular, there is no

1

12

12

+

n

1

12

S1,1

1

12

n

1

12

Figure 6: Deterministic MDP instance for optimistic lower bound

<!-- image -->

dependence on n . Next we try to show that the class of optimistic algorithms will incur regret at least Ω( n log( δ -1 ) /glyph[epsilon1] ) .

Class of algorithms. We adopt the class of algorithms from Section G.2 in [30] with an additional assumption which we clarify momentarily. Recall that the class of algorithms assumes access to an optimistic value function ¯ V k ( s ) ≥ V ∗ ( s ) and optimistic Q-functions. In particular the algorithms construct optimistic Q and value functions as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We assume that there exists a c ≥ 1 such that

<!-- formula-not-decoded -->

where M = θ ( n ) and b k ( s, a ) ∼ √ Sf k ( s, a ) b rw k ( s, a ) , where f k is a decreasing function in the number of visits to ( s, a ) given by n k ( s, a ) . For n k ( s, a ) = Ω( n log( n )) , we assume b k ( s, a ) ≤ b rw k ( s, a ) . One can verify that this is true for the the Q and value functions of StrongEuler.

Lower bound. Let glyph[epsilon1] &gt; 0 be sufficiently small to be specified later and let N be such that

<!-- formula-not-decoded -->

Lemma E.9. There exists n 0 , glyph[epsilon1] 0 such that for any pair of n ≥ n 0 and glyph[epsilon1] ≤ glyph[epsilon1] 0 and any k ≤ N , with probability at least 1 -δ , it holds that either n k ( s 5 , 1 ) &lt; N/ 4 , or ¯ Q k ( s 4 , 1 , 1) &lt; ¯ Q k ( s 4 , 1 , 2) .

1

12

1

12

1

12

1

12

+€

+

€

Proof. Assume n k ( s 5 , 1 ) ≥ N/ 4 , then we have

<!-- formula-not-decoded -->

where we assume glyph[epsilon1] is sufficiently small such that b k ( s, a ) ≤ b rw k ( s, a ) for n k ( s, a ) ≥ N/ 4 .

On the other hand, we have have with probability at least 1δ , that ∀ k : ˆ r k ( s 4 , 1 , 2) + b rw k ( s 4 , 1 , 2) ≥ 1 / 12 . Hence conditioned under that event, we have

<!-- formula-not-decoded -->

The proof is completed for n 0 = 48 2 .

We can show the same for the upper part of the MDP.

Lemma E.10. There exists n 0 , glyph[epsilon1] 0 such that for any pair of n ≥ n 0 and glyph[epsilon1] ≤ glyph[epsilon1] 0 and any k ≤ N , with probability at least 1 -δ , it holds that either n k ( s 1 , 2 ) &lt; N/ 4 , or ¯ Q k ( s 1 , 1 , 2) &lt; ¯ Q k ( s 1 , 1 , 1) .

Proof. First we split ¯ Q k ( s 1 , 1 , 2) into the observed sum of mean rewards and bonuses from s 1 , 1 to s 5 , 2 and the value ¯ V k ( s 5 , 2 ) . Then we upper bound ¯ Q k ( s 1 , 1 , 1) by ¯ V k ( s 5 , 2 ) and the maximum observed sum of mean rewards and bonuses along the paths passing by s 3 ,j for j ∈ [ n ] . Finally analogous to the proof of Lemma E.9, it is straightforward show that the latter is always larger as long as the visitation count for s 2 , 2 exceeds N/ 4 .

Theorem E.11. There exists an MDP instance with deterministic transitions on which any optimistic algorithm with confidence parameter δ will incur expected regret of at least Ω( S log( δ -1 ) /glyph[epsilon1] )) while it is asymptotically possible to achieve Ω(log( K ) /glyph[epsilon1] ) regret.

Proof. Taking the MDP from Figure 6. Applying Lemma E.9 and E.10 shows that after N episodes with probability at least 1 -2 δ , the visitation count of s 2 , 2 and s 5 , 1 each do not exceed N/ 4 . Hence there are at least N/ 2 episodes in which neither of them is visited, which means an glyph[epsilon1] -suboptimal policy is taken. Hence the expected regret after N episodes is at least

<!-- formula-not-decoded -->

Theorem E.11 has two implications for optimistic algorithms in MDPs with deterministic transitions.

- It is impossible to be asymptotically optimal if the confidence parameter δ is tuned to the time horizon K .
- It is impossible to have an anytime bound matching the information-theoretic lower bound.

## F Proofs and extended discussion for regret upper-bounds

## F.1 Further discussion on Opportunity O.2

The example in Figure 1 does not illustrate O.2 to its fullest extent. We now expand this example and elaborate why it is important to address Opportunity O.2 . Our example can be found in Figure 7. The MDP is an extension of the one presented in Figure 1 with the new addition of actions a 5 and a 6 in

T1*

A5

a1

a6

€

2

S1

a2

52

a4

r = 0

а3

= €

TC 1

Figure 7: Example for Opportunity O.2

<!-- image -->

state s 3 and the new state following action a 6 . Again there is only a single action available at all other states than s 1 , s 2 , s 3 . The reward of the state following action a 6 is set as r = c + glyph[epsilon1]/ 2 . This defines a new sub-optimal policy π 3 and the gap gap( s 3 , a 6 ) = glyph[epsilon1] 2 . Information theoretically it is impossible to distinguish π 3 as sub-optimal in less than Ω(log( K ) /glyph[epsilon1] 2 ) rounds and so any uniformly good algorithm would have to pay at least O (log( K ) /glyph[epsilon1] ) regret. However, what we observed previously still holds true, i.e., we should not have to play more than log( K ) /c 2 rounds to eliminate both π 1 and π 2 as sub-optimal policies. Prior work now suffers Opportunity O.2 as it would pay log( K ) /glyph[epsilon1] regret for all zero gap state-action pairs belonging to either π 1 or π 2 , essentially evaluating to SA log( K ) /glyph[epsilon1] . On the other hand our bounds will only pay log( K ) /glyph[epsilon1] regret for zero gap state-action pairs belonging to π 3 .

## F.2 Useful decomposition lemmas

We start by providing the following lemma that establishes that the instantaneous regret can be decomposed into gaps defined w.r.t. any optimal (and not necessarily Bellman optimal) policy.

Lemma F.1 (General policy gap decomposition) . Let gap ˆ π ( s, a ) = V ˆ π ( s ) -Q ˆ π ( s, a ) for any optimal policy ˆ π ∈ Π ∗ . Then the difference in values of ˆ π and any policy π ∈ Π is

<!-- formula-not-decoded -->

and, further, the instantaneous regret of π is

<!-- formula-not-decoded -->

Proof. We start by establishing a recursive bound for the value difference of π and ˆ π for any s

<!-- formula-not-decoded -->

Unrolling this recursion for all layers gives

<!-- formula-not-decoded -->

To show the second identity, consider s = s 1 and note that v π = V π ( s 1 ) and v ∗ = v ˆ π = V ˆ π ( s 1 ) because ˆ π is an optimal policy.

For the rest of the paper we are going to focus only on the Bellman optimal policy from each state and hence only consider gap ˆ π ( s, a ) = gap( s, a ) . All of our analysis will also go through for arbitrary gap ˆ π , ˆ π ∈ Π ∗ , however, this did not provide us with improved regret bounds.

We now show the following technical lemma which generalizes the decomposition of value function differences and will be useful in the surplus clipping analysis.

Lemma F.2. Let Ψ : S → R , ∆ : S × A → R be functions satisfying Ψ( s ) = 0 for any s with κ ( s ) = H +1 and π : S → A a deterministic policy. Further, assume that the following relation holds

<!-- formula-not-decoded -->

and let A be any event that is H h -measurable where H h = σ ( S 1 , A 1 , R 1 , . . . , S h ) is the sigmafield induced by the episode up to the state at time h . Then, for any h ∈ [ H ] and h ′ ∈ N with h ≤ h ′ ≤ H +1 , it holds that

<!-- formula-not-decoded -->

Proof. First apply the assumption of Ψ recursively to get

<!-- formula-not-decoded -->

Plugging this identity into E π [ χ ( A ) Ψ( S h ))] yields

<!-- formula-not-decoded -->

where H h = σ ( S 1 , A 1 , R 1 , . . . , S h ) is the sigma-field induced by the episode up to the state at time h . Identity ( i ) holds because of the Markov-property and ( ii ) holds because A is H h -measurable. The final identity ( iii ) uses the tower-property of conditional expectations.

## F.3 General surplus clipping for optimistic algorithms

Clipped operators. One of the main arguments to derive instance dependent bounds is to write the instantaneous regret in terms of the surpluses which are clipped to the minimum positive gap. We now define the clipping threshold glyph[epsilon1] k : S × A → R + 0 and associated clipped surpluses

<!-- formula-not-decoded -->

Next, define the clipped Q - and value-function as

<!-- formula-not-decoded -->

The random variable which is the state visited by π k at time h throughout episode k is denoted by S h and A h is the action at time h .

Events about encountered gaps Define the event E h = { gap( S h , A h ) &gt; 0 } that at time h an action with a positive gap played, the P 1: h = ⋂ h -1 h ′ =1 E c h ′ that only actions with zero gap have been played until h and the event A h = E h ∩ P 1: h that the first positive gap was encountered at time h . Let A H +1 = P 1: H be the event that only zero gaps were encountered. Further, let

<!-- formula-not-decoded -->

be the first time a non-zero gap is encountered. Note that B is a stopping time w.r.t. the filtration F h = σ ( S 1 , A 1 , . . . , S h , A h ) .

The proof of Simchowitz and Jamieson [30] consists of two main steps. First show that for their definition of clipped value functions one can bound ¨ V k ( s 1 ) -V π k ( s 1 ) ≥ 1 2 ( ¯ V k ( s 1 ) -V π k ( s 1 )) . Next, using optimism together with the fact that π k has highest value function at episode k it follows that ¯ V k ( s 1 ) -V π k ( s 1 ) ≥ V ∗ ( s 1 ) -V π k ( s 1 ) . The second main step is to use a high-probability bound on the clipped surpluses to relate them to the probability to visit the respective state-action pair and the proof is finished via an integration lemma. We now show that the first step can be carried out in greater generality by defining a less restrictive clipping operator. This operator is independent of the details in the definition of gap at each state-action pair but rather only uses a certain property which allows us to decompose the episodic regret as a sum over gaps. We will also further show that one does not need to use an integration lemma for the second step but can rather reformulate the regret bound as an optimization problem. This will allow us to clip surpluses at state-action pairs with zero gaps beyond the gap min rate.

Clipping with an arbitrary threshold. Recall the definition of the clipped surpluses and clipped value function in Equation 21 and Equation 22. We begin by showing a general relation between the clipped value function difference and the non-clipped surpluses for any clipping threshold glyph[epsilon1] k : S → R . This will help in establishing ¨ V k ( s 1 ) -V π k ( s 1 ) ≥ 1 2 ( ¯ V k ( s 1 ) -V π k ( s 1 )) .

Lemma F.3. Let glyph[epsilon1] k : S × A → R + 0 be arbitrary. Then for any optimistic algorithm it holds that

<!-- formula-not-decoded -->

Proof. Weuse W k ( s ) = ¨ V k ( s ) -V π k ( s ) in the following and first show that W ( s 1 ) ≥ E π k [ W k ( S B )] . As a precursor, we prove

E π k [ χ ( P 1: h ) W k ( S h )] ≥ E π k [ χ ( A h +1 ) W k ( S h +1 )] + E π k [ χ ( P 1: h +1 ) W k ( S h +1 )] . (24) To see this, plug the definitions into W k ( s ) which gives W k ( s ) = ¨ V k ( s ) -V π k ( s ) = ¨ E k ( s, π k ( s )) + 〈 P ( ·| s, π k ( s )) , W k 〉 and use this in the LHS of (24) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where H h = σ ( S 1 , A 1 , R 1 , . . . , S h ) is the sigma-field induced by the episode up to the state at time h . Step ( i ) follows from clip[ ·| c ] ≥ 0 for any c ≥ 0 and the Markov property and ( ii ) holds because P 1: h is H h -measurable. We now rewrite the RHS by splitting the expectation based on whether event E h +1 occurred as

E π k [ χ ( P 1: h ) W k ( S h +1 )] = E π k [ χ ( P 1: h +1 ) W k ( S h +1 )] + E π k [ χ ( A h +1 ) W k ( S h +1 )] . We have now shown (24), which we will now use to lower-bound W k ( s 1 ) as

<!-- formula-not-decoded -->

Applying Lemma F.2 with A = A h , Ψ = W k and ∆ = ¨ E k yields

<!-- formula-not-decoded -->

where we applied the definition clipped surpluses which gives ¨ E k ( s, a ) = clip[ E k ( s, a ) | glyph[epsilon1] k ( s, a )] ≥ E k ( s, a ) -glyph[epsilon1] k ( s, a ) . It only remains to show that

<!-- formula-not-decoded -->

To do so, we apply Lemma F.2 twice, first with A = A h , Ψ = ¯ V k -V π k and ∆ = E k and then again with A = A h , Ψ = V ∗ -V π k and ∆ = gap which gives

<!-- formula-not-decoded -->

Thus, we have shown that

<!-- formula-not-decoded -->

where the last equality uses the definition of B , the first time step at which a non-zero gap was encountered.

Lemma F.4 (Optimism of clipped value function) . Let the clipping thresholds glyph[epsilon1] k : S × A → R + 0 used in the definition of ¨ V k satisfy

<!-- formula-not-decoded -->

for some optimal policy ˆ π . Then scaled optimism holds for the clipped value function, i.e.,

<!-- formula-not-decoded -->

Proof. The proof works by establishing the following chain of inequalities:

<!-- formula-not-decoded -->

(

e

)

≤

¨

V

k

(

s

1

)

-

V

π

k

(

s

1

)

.

Here, ( a ) uses Lemma F.1 and ( b ) uses the definition of B . Step ( c ) is just algebra and step ( d ) uses the assumption on the threshold function. The last step ( e ) follows from Lemma F.3.

Proposition 3.3 (Improved surplus clipping bound) . Let the surpluses E k ( s, a ) be generated by an optimistic algorithm. Then the instantaneous regret of π k is bounded as follows:

<!-- formula-not-decoded -->

where glyph[epsilon1] k : S × A → R + 0 is any clipping threshold function that satisfies

<!-- formula-not-decoded -->

Proof. Applying Lemma F.4 which ensures scaled optimism of the clipped value function gives

<!-- formula-not-decoded -->

where the equality follows from the definition of ¨ V k ( s 1 ) and Lemma F.2. Subtracting 1 2 ( V ∗ ( s 1 ) -V π k ( s 1 )) from both sides gives

<!-- formula-not-decoded -->

because Lemma F.1 ensures that 1 2 ( V ∗ ( s 1 ) -V π k ( s 1 )) = 1 2 ∑ s,a w π k ( s, a ) gap( s, a ) . Reordering terms yields

<!-- formula-not-decoded -->

where the final inequality follows from the general properties of the clipping operator, which satisfies

<!-- formula-not-decoded -->

## F.4 Definition of valid clipping thresholds glyph[epsilon1] k

Proposition 3.3 establishes a sufficient condition on the clipping thresholds glyph[epsilon1] k that ensures that the penalized surplus clipping bounds holds. We now discuss several choices for this threshold that satisfy this condition.

Minimum positive gap gap min : We now make the quick observation that taking glyph[epsilon1] k ≡ gap min 2 H will satisfy the condition of Proposition 3.3, because on the event B ≡ A c H +1 there exists at least one positive gap in the sum ∑ H h =1 gap( S h , A h ) , which, by definition, is at least gap min . This shows that our results already can recover the bounds in prior work, with significantly less effort.

Average gaps: Instead of the minimum gap which was used in existing analyses, we now show that we can also use the marginalized average gap which we will define now. Recall that B = min { h ∈ [ H +1]: gap( S h , A h ) &gt; 0 } is the first time a non-zero gap is encountered. Note that B is a stopping time w.r.t. the filtration F h = σ ( S 1 , A 1 , . . . , S h , A h ) . Further let

<!-- formula-not-decoded -->

be the event that ( s, a ) was visited after a non-zero gap in the episode. We now define this clipping threshold

<!-- formula-not-decoded -->

As the following lemma shows, this is a valid choice which satisfies the condition of Proposition 3.3.

Lemma F.5. The expected sum of clipping thresholds in Equation (26) over all state-action pairs encountered after a positive gap is at most half the expected total gaps per episode. That is,

<!-- formula-not-decoded -->

Proof. Werewrite the LHS of the inequality to show as E π k [ ∑ H h =1 χ ( B ≤ h ) glyph[epsilon1] k ( S h , A h ) ] and from now on consider the random variable f h ( B,S h , A h ) = χ ( B ≤ h ) glyph[epsilon1] k ( S h , A h ) where f h ( b, s, a ) = χ ( b ≤ h ) glyph[epsilon1] k ( s, a ) is a deterministic function 5 . We will show below that E π k [ f h ( B,S h , A h )] ≤ 1 2 H E π k [ ∑ H h = B gap( S h , A h ) ] . This is sufficient to prove the statement, because

<!-- formula-not-decoded -->

To bound the expected value of f h ( B,S h , A h ) , we first write f h for all triples b, s, a such that P π k ( B = b, A h = a, S h = s ) &gt; 0 as

<!-- formula-not-decoded -->

where ( i ) expands the definition of glyph[epsilon1] k and ( ii ) decomposes the sum inside the conditional expectation and uses the Markov-property to simplify the conditioning for terms after h . Before taking the

5 It may still depend on the current policy π k which is determined by observations in episodes 1 to k -1 . But, crucially, f h does not depend on any realization in the k -th episode

expectation of f h ( B,S h , A h ) , we first rewrite the conditional expectation in the first term above, which will be useful later.

<!-- formula-not-decoded -->

Here, step ( i ) uses the property of conditional expectations with respect to an event with nonzero probability and ( ii ) follows from the definition of B : When B &gt; h , the sum of gaps until h is zero. Consider now the expectation of f h ( B,S h , A h )

<!-- formula-not-decoded -->

The term in (28) can be bounded using the tower-property of expectations as

<!-- formula-not-decoded -->

For the term in (27), we also use the tower-property to rewrite it as

<!-- formula-not-decoded -->

Summing both terms yields the required upper-bound 1 2 H E π k [ ∑ H h = B gap( S h , A h ) ] on the expectation E π k [ f h ( B,S h , A h )] .

## F.5 Policy-dependent regret bound for STRONGEULER

We now show how to derive a regret bound for STRONGEULER algorithm in Simchowitz and Jamieson [30] that depends on the gaps of the played policies throughout the K episodes.

To build on parts of the analysis in Simchowitz and Jamieson [30], we first define some useful notation analogous to Simchowitz and Jamieson [30] but adapted to our setting:

<!-- formula-not-decoded -->

We will use their following results:

Proposition F.6 (Proposition F.1, F.9 and B.4 in Simchowitz and Jamieson [30]) . There is a good event A conc that holds with probability 1 -δ/ 2 . In this event, STRONGEULER is strongly optimistic (as well as optimistic). Further, there is a universal constant c ≥ 1 so that for all k ≥ 1 , s ∈ S , a ∈ A , the surpluses are bounded as

<!-- formula-not-decoded -->

where B lead , B fut are defined as

<!-- formula-not-decoded -->

Lemma F.7 (Lemma B.3 in Simchowitz and Jamieson [30]) . Let m ≥ 2 , a 1 , . . . , a m ≥ 0 and glyph[epsilon1] ≥ 0 . Then clip [∑ m i =1 a i ∣ ∣ glyph[epsilon1] ] ≤ 2 ∑ m i =1 clip [ a i | glyph[epsilon1] 2 m ] .

Equipped with these results and our improved surplus clipping proposition in Proposition F.6, we can now derive the following bound on the regret of STRONGEULER

Lemma F.8. In event A conc , the regret of STRONGEULER is bounded for all k ≥ 1 as

<!-- formula-not-decoded -->

with a universal constant c ≥ 1 and ˘ gap k ( s, a ) = gap( s,a ) 4 ∨ glyph[epsilon1] k ( s, a ) .

Proof. We now use our improved surplus clipping result from Proposition 3.3 as a starting point to bound the instantaneous regret of STRONGEULER in the k th episode as

<!-- formula-not-decoded -->

Next, we write the bound on the surpluses from Proposition F.6 as

<!-- formula-not-decoded -->

and plugging it in (29) and applying Lemma F.7 gives

<!-- formula-not-decoded -->

The statement to show follows now by summing over k ∈ [ K ] . The form of the second term in the previous display follows from the inequality

<!-- formula-not-decoded -->

We note that if π k ≡ ˆ π for any ˆ π ∈ Π ∗ then V ∗ ( s 1 ) -V π k ( s 1 ) = 0 , and WLOG we can disregard such terms in the total regret.

The next step is to relate ¯ n k ( s, a ) to n k ( s, a ) via the following lemma.

Lemma F.9 (Lemma B.7 in Simchowitz and Jamieson [30]) . Define the event A samp

<!-- formula-not-decoded -->

where τ ( s, a ) = inf { k : ¯ n k ( s, a ) ≥ H samp } and H samp = c ′ log( M/δ ) for a universal constant c ′ . Then event A samp holds with probability 1 -δ/ 2 .

Proof. This can be proved analogously to Lemma B.7 in Simchowitz and Jamieson [30] and Lemma 6 in Dann et al. [9] with the difference that in our case, there can only be at most one observation of ( s, a ) per episode for each ( s, a ) due to our layered assumption. Thus, there is no need to sum over observations accumulated for each h ∈ [ H ] and our H samp = O (log( H )) as opposed to O ( H log( H )) .

Lemma F.10. Let f s,a : N → R be non-increasing with sup u f s,a ( u ) ≤ ˆ f &lt; ∞ for all s, a ∈ S ×A . Then on event A samp in Lemma F.9, we have

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Theorem F.11 (Regret Bound for STRONGEULER) . With probability at least 1 -δ , the regret of STRONGEULER is bounded for all number of episodes K ∈ N as

<!-- formula-not-decoded -->

Here, K ( s,a ) is the last round during which a policy π was played such that w π ( s, a ) &gt; 0 , ˘ gap t ( s, a ) = gap( s, a ) ∨ glyph[epsilon1] t ( s, a ) , ˘ gap min ( s, a ) = min k ∈ [ K ] : ˘ gap k ( s,a ) &gt; 0 ˘ gap k ( s, a ) is the smallest gap encountered for each ( s, a ) , and LOG ( M/δ,t, ˘ gap t ( s, a )) = log ( M δ ) log ( t ∧ 1 + 16 V ∗ ( s,a ) log( M/δ ) ˘ gap t ( s,a ) 2 ) .

Proof. We here consider the event A conc ∩ A samp which has probability at least 1 -δ by Proposition F.6 and Lemma F.9. We now start with the regret bound in Lemma F.8 and bound the two terms individually in the following:

Bounding the B lead term We have

<!-- formula-not-decoded -->

where step ( i ) applies Lemma F.10 and ( ii ) follows from the definition of V k ( s, a ) , the definition of K ( s,a ) and

<!-- formula-not-decoded -->

We now apply our optimization lemma (Lemma F.16) with x k = w π k ( s, a ) , v k = 2 c √ V ∗ ( s, a ) log( M/δ ) , and glyph[epsilon1] k = ˘ gap k ( s,a ) 4 v k to bound each ( s, a ) -term in (30) for any t ∈ [ K ] as

Let LOG

<!-- formula-not-decoded -->

Plugging this bound back in (30) gives

<!-- formula-not-decoded -->

where glyph[lessorsimilar] only ignores absolute constant factors.

Bounding the B fut term Consider the second term in Lemma F.8 and event A conc ∩A samp . Then by Lemma F.10

<!-- formula-not-decoded -->

where f s,a is

<!-- formula-not-decoded -->

We now apply Lemma C.1 by Simchowitz and Jamieson [30] which gives

<!-- formula-not-decoded -->

The remaining integral term is bounded with Lemma B.9 (b) by Simchowitz and Jamieson [30] with C ′ = S, C = H 3 and glyph[epsilon1] = ˘ gap min ( s, a ) = min k ∈ [ K ( s,a ) ] : ˘ gap k ( s,a ) &gt; 0 ˘ gap k ( s, a ) as follows.

<!-- formula-not-decoded -->

Comparing with the bound in Simchowitz and Jamieson [30]. We now proceed to compare our bound directly to the one stated in Corollary B.1 [30]. We will ignore the factors with only poly-logarithmic dependence on gaps as they are are common between both bounds. We now recall the regret bound presented in Corollary B.1, modulo said factors:

<!-- formula-not-decoded -->

where V ∗ = max ( s,a ) V ( s, a ) , Z opt is the set on which gap( s, π ∗ ( s )) = 0 , i.e., the set of state-action pairs assigned to π ∗ according to the Bellman optimality condition, and Z sub is the complement of Z opt . If we take t = K in Theorem F.11, we have the following upper bound:

<!-- formula-not-decoded -->

where S opt is the set of all states for s ∈ S for which gap( s, π ∗ ( s )) = 0 and there exists at least one state s ′ with κ ( s ′ ) &lt; s for which gap( s ′ , π ∗ ( s )) &gt; 0 . We note that this set is no larger than the set Z opt and further that even the smallest glyph[epsilon1] k ( s, a ) can still be much larger than gap min , as it is the conditional average of the gaps. In particular, this leads to an arbitrary improvement in our example in Figure 1 and an improvement of SA in the example in Figure 7.

## F.6 Nearly tight bounds for deterministic transition MDPs

We recall that for deterministic MDPs, glyph[epsilon1] k ( s, a ) = V ∗ ( s 1 ) -V π k ( s 1 ) 2 H , ∀ a and the definition of the set Π s,a :

<!-- formula-not-decoded -->

We note that V ( s, a ) ≤ 1 as this is just the variance of the reward at ( s, a ) . Theorem F.11 immediately yields the following regret bound by taking t = K .

Corollary F.12 (Explicit bound from (5)) . Suppose the transition kernel of the MDP consists only of point-masses. Then with probability 1 -δ , StrongEuler 's regret is bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

We now compare the above bound with the one in [30] again. For simplicity we are going to take K to be the smaller of the two quantities in the logarithm. To compare the bounds, we compare ∑ ( s,a ):Π s,a = ∅ H (log( KM/δ ))) v ∗ -v ∗ ( s,a ) to ∑ ( s,a ) ∈Z sub αH log( KM/δ ) gap( s,a ) + |Z opt | H gap min . Recall that α ∈ [0 , 1] is defined as the smallest value such that for all ( s, a, s ′ ) ∈ S × A × S it holds that

<!-- formula-not-decoded -->

For any deterministic transition MDP with more than one layer and one sub-optimal action it holds that α = 1 . We will compare V ∗ ( s 1 ) -V π ∗ ( s,a ) ( s 1 ) to gap( s, a ) = Q ∗ ( s, π ∗ ( s )) -Q ∗ ( s, a ) . This comparison is easy as by Lemma F.1 we can write

<!-- formula-not-decoded -->

Hence, our bound in the worst case matches the one in Simchowitz and Jamieson [30] and can actually be significantly better. We would further like to remark that we have essentially solved all of the issues presented in the example MDP in Figure 1. In particular we do not pay any gap-dependent factors for states which are only visited by π ∗ , we do not pay a gap min factor for any state and we never pay any factors for distinguishing between two suboptimal policies. Finally, we compare this bound to the lower bound derived Theorem 4.5 only with respect to number of episodes and gaps. Let S ∗ be the set of all states in the support of an optimal policy glyph[negationslash]

<!-- formula-not-decoded -->

The difference between the two bounds, outside of an extra H 2 factor, is in the sets S ∗ and the set { s, a : Π s,a = ∅} . We note that { s, a : Π s,a = ∅} ⊆ S ∗ . Unfortunately there are examples in which { s, a : Π s,a = ∅} is O (1) and S ∗ = Ω( S ) leading to a discrepancy between the upper and lower bounds of the order Ω( S ) . As we show in Theorem E.11 this discrepancy can not really be avoided by optimistic algorithms.

## F.7 Tighter bounds for unique optimal policy.

If we further assume that the optimal policy is unique on its support, then we can show STRONGEULER will only incur regret on sub-optimal state-action pairs. This matches the information theoretic lower bound up to horizon factors. We begin by showing a different type of upper bound on the expected gaps by the surpluses. Define the set β k = range ( B ) where B is the r.v. which is the stopping time with respect to π k . For any π ∗ , define the set

<!-- formula-not-decoded -->

This set has the following intuitive definition - whenever A B occurs we restrict our attention to the MDP with initial state S B . On this restricted MDP, O k is the set of state-action pairs which have greater probability to be visited by the optimal π ∗ than by π k .

glyph[negationslash]

Lemma F.13. Assume strong optimism and greedy ¯ V k , i.e., ¯ V k ( s ) ≥ max a ¯ Q k ( s, a ) for all s ∈ S . Then there exists an optimal π ∗ for which

<!-- formula-not-decoded -->

Proof. One can write the optimistic value function for any s and π as follows

<!-- formula-not-decoded -->

By backwards induction on H we show that for any s , κ ( s ) ≤ H ¯ V π ≤ ¯ V k . The base case holds from the fact that on all s : κ ( s ) = H , ¯ V k ( s ) is just the largest optimistic reward over all actions at s . For the induction step it holds that

<!-- formula-not-decoded -->

where the first inequality holds from the induction hypothesis and the second inequality holds by definition of the value function. We now have

<!-- formula-not-decoded -->

Let us focus on the term E π k [ ¯ V ∗ ( S B ) -V ∗ ( S B ) ]

<!-- formula-not-decoded -->

We can similarly expand the term E π k [ ¯ V k ( S B ) -V k ( S B ) ] . By the definition of O k ( π ∗ ) it holds that for any h ≥ κ ( s )

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

We next show a version of Lemma F.3 which takes into account the set O k ( π ∗ ) .

Lemma F.14. With the same assumptions as in Lemma F.13, there exists an optimal π ∗ for which

<!-- formula-not-decoded -->

where glyph[epsilon1] k is arbitrary.

Proof. Since ¨ E k is non-negative on all state-action pairs we have

<!-- formula-not-decoded -->

where the second to last inequality follows from the definition of ¨ E k and the last inequality follows from Lemma F.13.

Next, we define ¯ glyph[epsilon1] k in the following way. Let

<!-- formula-not-decoded -->

where glyph[epsilon1] k is the clipping function defined in Equation 26. Lemma F.14 now implies that

<!-- formula-not-decoded -->

glyph[negationslash]

This is sufficient to argue Lemma F.8 with ˘ gap k ( s, a ) = gap( s,a ) 4 ∨ ¯ glyph[epsilon1] k ( s, a ) and hence arrive at a version of Corollary F.12 which uses ¯ glyph[epsilon1] k as the clipping thresholds. Let us now argue that ¯ glyph[epsilon1] k ( s, a ) = ∞ for all ( s, a ) ∈ π ∗ whenever π ∗ is the unique optimal policy for the deterministic MDP. To do so consider ( s, a ) ∈ π ∗ and π k = π ∗ . Since the MDP is deterministic, β k is a singleton and is the the first state s b at which π k differs from π ∗ . We now observe that if κ ( s ) &lt; κ ( s b ) , this implies glyph[epsilon1] k ( s, a ) = ∞ as B ( s, a ) does not occur. Further, the conditional probabilities P π ∗ (( S h , A h ) = ( s, a ) | S κ ( s b ) = s b ) and P π k (( S h , A h ) = ( s, a ) | S κ ( s b ) = s b ) are both equal to 1 if κ ( s ) &gt; κ ( s b ) and so ( s, a ) ∈ O k ( π ∗ ) which implies ¯ glyph[epsilon1] k ( s, a ) = ∞ . Thus we can clip all gaps at ( s, a ) ∈ π ∗ to infinity and they will never appear in the regret bound. With the notation from Corollary F.12 we have the following tighter bound.

Corollary F.15. Suppose the transition kernel of the MDP consists only of point-masses and there exists a unique optimal π ∗ . Then with probability 1 -δ , StrongEuler 's regret is bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Comparing terms which depend polynomially on 1 / gap to the information theoretic lower bound in Theorem 4.5 we observe only a multiplicative difference of H 2 .

## F.8 Alternative to integration lemmas

The following lemma is an alternative to the integration lemmas when bounding the sum of the clipped surpluses and in some cases allows us to save additional factors of H .

Lemma F.16. Consider the following optimization problem

<!-- formula-not-decoded -->

with ( v i ) i ∈ [ K ] ∈ R K + and ( glyph[epsilon1] i ) i ∈ [ K ] ∈ R K + . Then the optimal value of Problem 32 is bounded for any t ∈ [ K ] as

<!-- formula-not-decoded -->

where ¯ v t = max k ∈ [ t ] v k and v ∗ t = max K ≥ k ≥ t v k .

Proof. Denote by X k = ∑ k t =1 x t the cumulative sum of x t . The proof consists of splitting the objective of (32) into two terms:

<!-- formula-not-decoded -->

and bounding each by the corresponding one in (33) respectively.

Before doing so, we derive the following bound on the sum of x k √ X k terms:

<!-- formula-not-decoded -->

where the inequality is due to X k being non-decreasing.

Consider now each term in the objective in (34) separately.

Summands up to t : Since X k is non-decreasing, we can bound

<!-- formula-not-decoded -->

where ( i ) follows from (35) using the convention X 0 = 0 and ( ii ) from the optimization constraint √ log( X t ) ≥ glyph[epsilon1] t √ X t . It remains to bound log( X t ) by 2 log ( t ∧ 1 + 1 glyph[epsilon1] 2 t ) . Since all increments x j are at most 1 , the bound log( X t ) ≤ log( t ) holds.

We claim the following:

Claim F.17. For any x s.t. log( x ) ≤ log(log( x ) /a ) it holds that log( x ) ≤ 2 log(1 + 1 /a ) .

Proof. First, we note that if 0 &lt; x ≤ e , then log(log( x )) &lt; 0 and thus the assumption of the claim implies log( x ) ≤ log(1 /a ) . Next, assume that x &gt; e . Then we have log(log( x )) log( x ) ≤ 1 /e , which together with the assumption of the claim implies log( x ) ≤ 1 /e log( x ) + log(1 /a ) or equivalently log( x ) ≤ e e -1 log(1 /a ) . Noting that e/ ( e -1) ≤ 2 completes the proof.

The constraints of the problem enforce √ X k ≤ √ log( X k ) glyph[epsilon1] k , which implies after squaring and taking the log : log( X k ) ≤ log(log( X k ) /glyph[epsilon1] 2 k ) . Thus, using Claim F.17 yields:

<!-- formula-not-decoded -->

Summands larger than t : Let v ∗ t = max k : t&lt;k ≤ K v k . For this term, we have

<!-- formula-not-decoded -->

where we first bounded log( X k ) ≤ log( X K ) , because X k is non-decreasing, and used the upper bound on log( X K ) . Then we applied (35) and finally used 0 ≤ x k ≤ 1 .