<!-- image -->

Contents lists available at ScienceDirect

## Results in Control and Optimization

journal homepage: www.elsevier.com/locate/rico

## Multiagent value iteration algorithms in dynamic programming and reinforcement learning

## Dimitri Bertsekas ∗

McAfee Professor of Engineering, MIT, Cambridge, MA, United States of America Fulton Professor of Computational Decision Making, ASU, Tempe, AZ, United States of America

## A B S T R A C T

We consider infinite horizon dynamic programming problems, where the control at each stage consists of several distinct decisions, each one made by one of several agents. In an earlier work we introduced a policy iteration algorithm, where the policy improvement is done one-agent-at-a-time in a given order, with knowledge of the choices of the preceding agents in the order. As a result, the amount of computation for each policy improvement grows linearly with the number of agents, as opposed to exponentially for the standard all-agents-at-once method. For the case of a finite-state discounted problem, we showed convergence to an agent-by-agent optimal policy. In this paper, this result is extended to value iteration and optimistic versions of policy iteration, as well as to more general DP problems where the Bellman operator is a contraction mapping, such as stochastic shortest path problems with all policies being proper.

## 1. Multiagent problem formulation

We consider an abstract form of infinite horizon dynamic programming (DP) problem, which contains as special case finite-state discounted Markovian decision problems (MDP), as well as more general problems where the Bellman operator is a monotone weighted sup-norm contraction. The distinguishing feature of the problem is that the control 𝑢 consists of 𝑚 components 𝑢 𝓁 , 𝓁 = 1 , … , 𝑚 , where 𝑚 &gt; 1 :

<!-- formula-not-decoded -->

Conceptually, each component may be viewed as chosen by a distinct agent, with knowledge of the selections of the other agents. We consider value iteration (VI) algorithms that involve minimization component-by-component as opposed to minimization over all components at once. This is similar to what is done in coordinate descent methods for multivariable optimization, and can lead to dramatic gains in computational efficiency for large and even moderate values of 𝑚 . We propose several methods and we show their convergence to an agent-by-agent optimal policy, a type of policy that is related to the notion of person-by-person optimality from the theory of teams. Our analysis extends and complements our earlier proposals of rollout and policy iteration (PI) algorithms [1,2].

We assume that 𝑢 is chosen from a finite constraint set 𝑈 ( 𝑥 ) when the system is at state 𝑥 . In our earlier papers [1,2], we have made a stronger assumption: we assumed that each control component 𝑢 𝓁 , 𝓁 = 1 , … , 𝑚 , is separately constrained to lie in a given finite set 𝑈 𝓁 ( 𝑥 ) . In this case 𝑈 ( 𝑥 ) is the Cartesian product set

<!-- formula-not-decoded -->

In this paper, we do not impose this assumption, except occasionally to discuss its implications. As a result our algorithms must ensure that the selection of a control component at a given state and stage does not preclude the feasibility of selection of the other control components at the same state and stage. This complicates our algorithms relative to the Cartesian product case (1.2). We will discuss the mechanism for dealing with this issue in Section 2. For the remainder of this section, we will assume no special structure for the constraint set 𝑈 ( 𝑥 ) other than finiteness.

∗ Correspondence to: Fulton Professor of Computational Decision Making, ASU, Tempe, AZ, United States of America E-mail address: dbertsek@asu.edu.

<!-- image -->

<!-- image -->

D. Bertsekas

## The 𝛼 -discounted MDP case

A major context for application of our algorithmic ideas is the standard infinite horizon discounted MDP with states 𝑥 = 1 , … , 𝑛 . Here, at state 𝑥 , a control 𝑢 is applied, and the system transitions to a next state 𝑦 with transition probabilities 𝑝 𝑥𝑦 ( 𝑢 ) and cost 𝑔 ( 𝑥, 𝑢, 𝑦 ) . The control is chosen at state 𝑥 from a finite constraint set 𝑈 ( 𝑥 ) . The cost function of a stationary policy 𝜇 that applies control 𝜇 ( 𝑥 ) ∈ 𝑈 ( 𝑥 ) at state 𝑥 is denoted by 𝐽 𝜇 ( 𝑥 ) , and the optimal cost [the minimum over 𝜇 of 𝐽 𝜇 ( 𝑥 ) ] is denoted by 𝐽 ∗ ( 𝑥 ) .

The standard VI algorithm starts from some initial guess 𝐽 0 and iterates as follows 1 :

<!-- formula-not-decoded -->

where 𝑇 is the Bellman operator, which maps a vector 𝐽 = ( 𝐽 (1) , … , 𝐽 ( 𝑛 ) ) to the vector

<!-- formula-not-decoded -->

according to

<!-- formula-not-decoded -->

Thus each VI involves a comparison of all the Q-factors

<!-- formula-not-decoded -->

A related algorithm is optimistic PI , which involves simultaneous value and policy iterations, using the Bellman operator 𝑇 𝜇 defined for each policy 𝜇 by

<!-- formula-not-decoded -->

Given a pair ( 𝜇 𝑘 , 𝐽 𝑘 ) , this algorithm generates ( 𝜇 𝑘 +1 , 𝐽 𝑘 +1 ) according to

<!-- formula-not-decoded -->

where 𝑞 is a positive integer (which in some cases may depend on 𝑘 ), and 𝑇 𝑞 𝜇 denotes the mapping obtained by 𝑞 -fold application of the mapping 𝑇 𝜇 . When 𝑞 = 1 we obtain the VI algorithm 𝐽 𝑘 +1 = 𝑇𝐽 𝑘 and when 𝑞 → ∞ , we have 𝐽 𝑘 +1 = 𝐽 𝜇 𝑘 (in the limit), so the algorithm approaches the standard PI algorithm where 𝜇 𝑘 +1 is obtained from 𝜇 𝑘 according to

<!-- formula-not-decoded -->

Unfortunately, iterating with the mapping 𝑇 is inconvenient for problems involving even a moderate number of agents, because the size of the control constraint set 𝑈 ( 𝑥 ) typically grows exponentially with 𝑚 . In particular, in the Cartesian product case (1.2), if each constraint set 𝑈 𝓁 ( 𝑥 ) consists of at most 𝑠 elements, minimization over 𝑈 ( 𝑥 ) involves a comparison of as many as 𝑠 𝑚 Q-factors of the form (1.4). This motivates us to consider versions of the preceding algorithms that involve a simpler form of minimization. For example, minimization over the component constraint sets 𝑈 𝓁 ( 𝑥 ) , one component at a time, which involves a comparison of 𝑠 Q-factors for each agent, for a total of 𝑠 ⋅ 𝑚 Q-factors.

## The general contractive DP case

It is convenient and useful to develop our algorithm in a more general setting, which involves an operator-based framework from the author's abstract DP book [3]. In particular, we consider a finite set 𝑋 of states and a finite set 𝑈 of controls , and for each 𝑥 ∈ 𝑋 , a nonempty control constraint set 𝑈 ( 𝑥 ) ⊂ 𝑈 . 2 We denote by  the set of all functions 𝜇 ∶ 𝑋 ↦ 𝑈 with 𝜇 ( 𝑥 ) ∈ 𝑈 ( 𝑥 ) for all 𝑥 ∈ 𝑋 , which we refer to as policies . We introduce a mapping 𝐻 ∶ 𝑋 × 𝑈 ×  ( 𝑋 ) ↦ ℜ , where ℜ denotes the real line and  ( 𝑋 ) denotes the set of real-valued functions 𝐽 ∶ 𝑋 ↦ ℜ . For each policy 𝜇 ∈  , we consider the mapping 𝑇 𝜇 ∶  ( 𝑋 ) ↦  ( 𝑋 ) defined by

(

)

𝑥, 𝜇

(

𝑇 𝜇 𝐽

)(

𝑥

) =

𝐻

(

𝑥

)

, 𝐽

,

𝑥

∈

𝑋.

We also consider the mapping 𝑇 defined by

(

𝑇𝐽

)(

𝑥

) =

min

𝑢

∈

𝑈

(

𝑥

)

𝐻

(

𝑥, 𝑢, 𝐽

) = min

𝜇

∈



(

𝑇 𝜇 𝐽

)(

𝑥

)

,

𝑥

∈

𝑋.

Note that the 𝛼 -discounted MDP is obtained when 𝐻 is given by

<!-- formula-not-decoded -->

1 Throughout the paper, we will be using componentwise equality and inequality notation, whereby for any pair of real-valued functions 𝐽, 𝐽 ′ of the state 𝑥 , we write 𝐽 = 𝐽 ′ (or 𝐽 ≤ 𝐽 ′ ) if 𝐽 ( 𝑥 ) = 𝐽 ′ ( 𝑥 ) [or 𝐽 ( 𝑥 ) ≤ 𝐽 ′ ( 𝑥 ) , respectively] for all 𝑥 .

2 The abstract DP framework of [3] does not require finiteness of the state and control spaces. We impose the finiteness assumption in order to obtain the most powerful algorithmic results possible. However, at several points in the paper, and particularly in Section 5, we speculate around the possibility of extending our algorithms and analysis to infinite state and control spaces.

D. Bertsekas

The problem is to find a function 𝐽 ∗ ∈  ( 𝑋 ) such that

<!-- formula-not-decoded -->

i.e., to find a fixed point of 𝑇 within  ( 𝑋 ) (we can view 𝐽 ∗ = 𝑇𝐽 ∗ as a generalized form of Bellman's equation). We also want to obtain a policy 𝜇 ∗ ∈  such that 𝑇 𝜇 ∗ 𝐽 ∗ = 𝑇𝐽 ∗ . We assume that the control 𝑢 consists of the 𝑚 components 𝑢 𝓁 , 𝓁 = 1 , … , 𝑚 , [cf. Eq. (1.1)]. Note that since the state and control spaces are assumed finite, the control constraint set 𝑈 ( 𝑥 ) and the set of policies  are also finite, so the minimum of various expressions over 𝑈 ( 𝑥 ) or  is attained.

We will adopt throughout the following monotonicity and contraction assumptions.

Assumption 1.1 ( Monotonicity ) . If 𝐽, 𝐽 ′ ∈  ( 𝑋 ) and 𝐽 ≤ 𝐽 ′ , then

<!-- formula-not-decoded -->

For the contraction assumption, we introduce a function 𝑣 ∶ 𝑋 ↦ ℜ with

<!-- formula-not-decoded -->

We consider the weighted sup-norm

<!-- formula-not-decoded -->

on  ( 𝑋 ) , the space of real-valued functions 𝐽 on 𝑋 .

Assumption 1.2 ( Contraction ) . For some 𝛼 ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

The monotonicity and contraction assumptions are satisfied in the 𝛼 -discounted finite-state MDP case (1.8), as well as other finite-state DP problems, such as stochastic shortest path problems in the special case where all policies are proper; see the books [3-5] for an extensive discussion. In particular, for the 𝛼 -discounted MDP, 𝑇 𝜇 is a contraction with respect to the unweighted sup-norm with contraction modulus 𝛼 , whereas in the stochastic shortest path case, 𝑇 𝜇 is a contraction with respect to a weighted sup-norm with weights and contraction modulus that depend on the maximum expected time to reach the destination using proper policies (see [4], Prop. 2.2).

General abstract DP models under Assumptions 1.1 and 1.2 have been investigated in detail in the author's monograph [3] [without assuming finiteness of 𝑋 and 𝑈 , but with  ( 𝑋 ) replaced by the set  ( 𝑋 ) of all uniformly bounded functions over 𝑋 , equipped with a weighted sup-norm]. The main results are that 𝑇 is a contraction mapping and has as unique fixed point the optimal cost function 𝐽 ∗ (the equation 𝐽 ∗ = 𝑇𝐽 ∗ is Bellman's equation). Also 𝐽 𝜇 is the unique fixed point of 𝑇 𝜇 . Moreover 𝜇 is optimal if and only if 𝑇 𝜇 𝐽 ∗ = 𝑇𝐽 ∗ (or equivalently 𝑇 𝜇 𝐽 𝜇 = 𝑇𝐽𝜇 ). Algorithmic results include the convergence of VI [i.e., 𝑇 𝑘 𝐽 → 𝐽 ∗ for all 𝐽 ∈  ( 𝑋 ) ], and also convergence results for the PI algorithm (1.7) and some of its variations. We will be using these results in what follows in this paper, with the monograph [3] as a general reference. For parts of our analysis, only the monotonicity and contraction Assumptions 1.1 and 1.2 are essential: the assumption of finiteness of the state and control spaces can be eliminated with minor mathematical proof modifications.

## 2. Agent-by-agent value iteration

The salient feature of the multiagent DP problem of this paper is that the control 𝑢 consists of 𝑚 components,

<!-- formula-not-decoded -->

cf. Eq. (1.1). We will aim to develop a computationally efficient variant of the standard VI algorithm 𝐽 𝑘 +1 = 𝑇𝐽 𝑘 , i.e.,

<!-- formula-not-decoded -->

Rather than simultaneous minimization over all the components 𝑢 1 , … , 𝑢 𝑚 , our multiagent VI algorithm involves sequential minimization of 𝐻 ( 𝑥, 𝑢 1 , … , 𝑢 𝑚 , 𝐽 𝑘 ) over a single component 𝑢 𝓁 , with the remaining components 𝑢 𝓁 ′ , 𝓁 ′ ≠ 𝓁 , fixed at the values obtained through the preceding minimizations . We maintain these control component values in a policy that is continually updated to incorporate the results of new minimizations.

Let 𝜇 be a given policy that applies at 𝑥 the control

<!-- formula-not-decoded -->

We define a constraint set for the 𝓁 th control component 𝑢 𝓁 that is given by

{

}

<!-- formula-not-decoded -->

Note that since a policy 𝜇 by definition satisfies the feasibility constraint

<!-- formula-not-decoded -->

D. Bertsekas the set 𝑈 𝓁 ,𝜇 ( 𝑥 ) contains 𝜇 𝓁 ( 𝑥 ) , so it is nonempty. Note also that when 𝑈 ( 𝑥 ) has the Cartesian product form 𝑈 1 ( 𝑥 ) × ⋯ × 𝑈𝑚 ( 𝑥 ) , the set 𝑈 𝓁 ,𝜇 ( 𝑥 ) is simply equal to 𝑈 𝓁 ( 𝑥 ) for all 𝜇 .

Our algorithm generates a double sequence { 𝐽 𝑘 , 𝜇 𝑘 } , starting from some pair ( 𝐽 0 , 𝜇 0 ) : at the 𝑘 th iteration, given ( 𝐽 𝑘 , 𝜇 𝑘 ) , the algorithm obtains ( 𝐽 𝑘 +1 , 𝜇 𝑘 +1 ) after 𝑚 successive minimizations, one for each of the components 𝑢 𝓁 , 𝓁 = 1 , … , 𝑚 . In particular, given the typical pair ( 𝐽, 𝜇 ) , our algorithm generates the next pair ( ̃ 𝐽, ̃ 𝜇 ) as the last of a sequence of cost function-policy component pairs

<!-- formula-not-decoded -->

to be defined shortly, i.e., it sets

<!-- formula-not-decoded -->

The cost function-policy pairs (2.2) are obtained as follows:

For every 𝓁 = 1 , … , 𝑚 , given ( 𝐽 ̂ 𝓁 -1 , ̂ 𝜇 1 , … , ̂ 𝜇 𝓁 -1 ) , the algorithm generates ( 𝐽 ̂ 𝓁 , ̂ 𝜇 𝓁 ) according to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the constraint set in the two preceding minimizations,

𝑈

𝓁

,

(

̂ 𝜇

1

,

…

, ̂ 𝜇

𝓁

-1

,𝜇

𝓁

,

…

,𝜇 𝑚

)

(

𝑥

)

,

is defined by Eq. (2.1); it is the set of 𝑢 𝓁 , which are consistent (in terms of feasibility) with the previously chosen components ̂ 𝜇 1 ( 𝑥 ) , … , ̂ 𝜇 𝓁 -1 ( 𝑥 ) and the component choices 𝜇 𝓁 +1 ( 𝑥 ) , … , 𝜇 𝑚 ( 𝑥 ) specified by the policy 𝜇 . To start this process, only the initial function 𝐽 ̂ 0 is needed (in addition to 𝜇 ), and it is given by

<!-- formula-not-decoded -->

Note that each of the minimizations (2.4) is performed for every state 𝑥 ∈ 𝑋 , and that there may be multiple possible policies ̃ 𝜇 that can be generated by this process [cf. Eq. (2.3)], since the minimum in Eq. (2.5) may not be uniquely attained. Similarly, there may be multiple possible functions 𝐽 ̃ that can be generated by this process [since the minimization (2.4) is affected by the multiplicity of possible policies ̂ 𝜇 1 , … , ̂ 𝜇 𝓁 -1 ]. In summary, our multiagent VI algorithm, starting from the pair ( 𝐽 𝑘 , 𝜇 𝑘 ) , generates the pair ( 𝐽 𝑘 +1 , 𝜇 𝑘 +1 ) according to

<!-- formula-not-decoded -->

where ̃  ( 𝐽 𝑘 , 𝜇 𝑘 ) is the set of cost function-policy pairs ( ̃ 𝐽, ̃ 𝜇 ) that can be generated by the process (2.3)-(2.6), starting with 𝐽 = 𝐽 𝑘 and 𝜇 = 𝜇 𝑘 .

## Optimistic and asynchronous PI algorithms

In the preceding algorithm (2.7), each iteration involves a policy improvement operation, i.e., an 𝑚 -step minimization that cycles through all control components one-by-one. In Section 3, we will also consider an optimistic PI variant where the 𝑚 -step minimization is performed for only an infinite subset  ⊂ {0 , 1 , …} of the iterations, while for the complementary subset of iterations, 𝑘 ∉  , we use the (less expensive) standard policy evaluation update 𝐽 𝑘 +1 = 𝑇 𝜇 𝑘 𝐽 𝑘 , and no policy update:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We call the algorithm (2.8)-(2.9) multiagent optimistic PI . It is a natural multiagent extension of the standard (single agent) optimistic PI algorithm (1.6), which is described in many sources for MDP and other problems, e.g., [3,5,6]. Note that when  = {0 , 1 , …} , the optimistic PI algorithm is the same as the multiagent VI algorithm (2.7). In cases where  is a ''small'' subset of {0 , 1 , …} , the multiagent optimistic PI algorithm involves nearly exact policy evaluations and ''approaches'' the multiagent PI algorithm proposed in our earlier papers [1,2].

In the preceding multiagent algorithms (2.7) and (2.8)-(2.9), the iterations are performed simultaneously for all states 𝑥 ∈ 𝑋 . In Section 4, we will also consider an asynchronous distributed version of the multiagent optimistic PI algorithm (2.8)-(2.9), whereby iteration 𝑘 is performed for only a subset 𝑋𝑘 of the states. A requirement here is that each state 𝑥 belongs infinitely often to some subset 𝑋𝑘 , so that there are infinitely many policy improvements at every state. This algorithm is well suited for distributed asynchronous computation, involving a partition of the state space into subsets, and with a processor assigned to each set of the partition.

## 3. Convergence to an agent-by-agent optimal policy

We will prove that the multiagent VI algorithm (2.7) converges to an agent-by-agent optimal policy, which we define as follows.

D. Bertsekas

Definition 3.1 ( Agent-by-Agent Optimality ) . We say that a policy 𝜇 = { 𝜇 1 , … , 𝜇 𝑚 } is agent-by-agent optimal if for all 𝑥 ∈ 𝑋 and 𝓁 = 1 , … , 𝑚 , we have

<!-- formula-not-decoded -->

where the constraint set 𝑈 𝓁 ,𝜇 ( 𝑥 ) is defined by Eq. (2.1).

To interpret this definition, let a policy 𝜇 = { 𝜇 1 , … , 𝜇 𝑚 } be given, and consider for every 𝓁 = 1 , … , 𝑚 the single agent DP problem where for all 𝓁 ′ ≠ 𝓁 the 𝓁 ′ th policy components are fixed at 𝜇 𝓁 ′ , while the 𝓁 th policy component is subject to optimization. The Definition 3.1 is the optimality condition for all the single agent problems; see [3], Chapter 2 [Eq. (3.1) can be written as 𝑇 𝜇, 𝓁 𝐽 𝜇 = 𝑇 𝓁 𝐽 𝜇 , where 𝑇 𝓁 and 𝑇 𝜇, 𝓁 are the Bellman operators (1.3) and (1.5) that correspond to the single agent problem involving agent 𝓁 ]. We can then conclude that 𝜇 = { 𝜇 1 , … , 𝜇 𝑚 } is agent-by-agent optimal if each component 𝜇 𝓁 is optimal for the 𝓁 th single agent problem, where it is assumed that the remaining policy components remain fixed; in other words by using 𝜇 𝓁 , each agent 𝓁 acts optimally, assuming all other agents 𝓁 ′ ≠ 𝓁 continue to use the corresponding policy components 𝜇 𝓁 ′ .

Our definition of an agent-by-agent optimal policy is related to the notion of ''person-by-person'' optimality from team theory, which has been studied primarily in the context of multiagent decision problems with nonclassical information patterns, whereby the agents may not share the information on which they base their decision. Thus team problems do not assume the shared information pattern that is characteristic of DP problems. For the origins of team theory and control with a nonclassical information pattern, we refer to Marschak [7], Radner [8], and Witsenhausen [9-11]. For a sampling of subsequent works, we refer to the survey by Ho [12], and the papers by Krainak, Speyer, and Marcus [13,14], de Waal and van Schuppen [15]. For more recent works, see Nayyar, Mahajan, and Teneketzis [16], Nayyar and Teneketzis [17], Li et al. [18], Gupta [19], the book by Zoppoli, Sanguineti, Gnecco, and Parisini [20], and the references quoted there.

Note that an (overall) optimal policy is agent-by-agent optimal, but the reverse may not be true. This is similar to properties of person-by-person optimal solutions in team theory. It is also similar to what may happen in coordinate descent methods for multivariable optimization, where it is possible (in the absence of favorable assumptions) to stop at a nonoptimal point where no progress can be made along any one coordinate; some examples involving a Cartesian product constraint set of the form (1.2) are given in the papers [1,2].

While an agent-by-agent optimal policy may be either optimal or adequate for practical purposes, it may offer no guarantees of quality. For a simple example, let 𝑈 ( 𝑥 ) be the intersection of a Cartesian product of finite subsets 𝑈 𝓁 ( 𝑥 ) of the real line and the unit simplex:

<!-- formula-not-decoded -->

Then it can be seen that the constraint set 𝑈 𝓁 ,𝜇 ( 𝑥 ) consists of just the single point 𝜇 𝓁 ( 𝑥 ) , so that all feasible policies are agent-by-agent optimal. This is due to the extreme coupling of the control components through the simplex constraint. It would not happen if the constraint set was just a Cartesian product 𝑈 1 ( 𝑥 ) × ⋯ × 𝑈𝑚 ( 𝑥 ) , in which case 𝑈 𝓁 ,𝜇 ( 𝑥 ) = 𝑈 𝓁 ( 𝑥 ) for all 𝓁 . Nonetheless, one should be aware that the method of partitioning of the control into components may seriously impact the effectiveness of our multiagent VI algorithm through the creation of spurious agent-by-agent optimal policies.

We will now prove our main convergence result, under the following assumption, which is reminiscent of strict convexity assumptions in the analysis of coordinate descent methods (see e.g., [21], Section 3.7). While we do not have a concrete counterexample, we speculate based on experience with coordinate descent methods, that the assumption cannot be easily dispensed with.

Assumption 3.1 ( Uniqueness Property ) . The cost functions of distinct policies are distinct, i.e., for any two policies 𝜇 and 𝜇 ′

𝜇

≠

𝜇

′

⟹

𝐽 𝜇

≠

𝐽 𝜇

′

.

Our convergence result also assumes that the initial condition ( 𝐽 0 , 𝜇 0 ) satisfies

<!-- formula-not-decoded -->

This assumption is unnecessary for the 𝛼 -discounted MDP where 𝑇 and 𝑇 𝜇 are given by Eqs. (1.3) and (1.5). The reason is that if we replace 𝐽 0 by a function 𝐽 0 obtained by shifting 𝐽 0 by a constant 𝑐 [i.e., replace 𝐽 0 ( 𝑥 ) by 𝐽 0 ( 𝑥 ) + 𝑐 for all 𝑥 ], we will have

(

𝑇 𝜇

0

𝐽

)(

𝑥

) = (

𝑇 𝜇

0

𝐽

0

)(

𝑥

) +

𝛼𝑐

≤

𝐽

0

(

𝑥

) +

𝑐

=

𝐽

(

𝑥

)

,

provided 𝑐 is large enough, thereby satisfying the assumption (3.2). At the same time, it can be seen that by replacing 𝐽 0 with 𝐽 0 the generated policies will not be affected, while the generated iterates 𝐽 𝑘 will just be shifted by an appropriate constant. Thus for discounted MDP the assumption (3.2) is unnecessary for the following convergence result, since the same sequence of policies will be obtained whether we use 𝐽 0 or 𝐽 0 .

For other types of problems the assumption (3.2) is needed. However, thanks to the contraction property of Assumption 1.2 , it can be typically satisfied by adding to 𝐽 0 ( 𝑥 ) a sufficiently large constant 𝑐 for all 𝑥 . In particular, any function 𝐽 that satisfies

<!-- formula-not-decoded -->

0

0

D. Bertsekas

(for example a sufficiently large constant function) also satisfies the condition (3.2). To see this, note that for 𝐽 ≤ 𝐽 and 𝑥 ∈ 𝑋 ,

<!-- formula-not-decoded -->

where the first inequality follows from the monotonicity of 𝐻 , the second inequality follows by applying the contraction property with 𝐽 = 𝐽 , 𝐽 ′ = 𝐽 𝜇 , and the third inequality is Eq. (3.3). Thus, for 𝐽 satisfying Eq. (3.3), we have 𝑇 𝜇 𝐽 ≤ 𝐽 for all 𝜇 ∈  .

Proposition 3.1 ( VI Convergence to an Agent-by-Agent Optimal Policy ) . Let Assumptions 1.1 , 1.2 , and 3.1 hold, and assume further that the state and control spaces 𝑋 and 𝑈 are finite, and that the initial pair ( 𝐽 0 , 𝜇 0 ) satisfies Eq. (3.2) . Let { 𝐽 𝑘 , 𝜇 𝑘 } be a sequence generated by the agent-by-agent VI algorithm (2.7) . Then there is an agent-by-agent optimal policy ̄ 𝜇 and an index 𝑘 such that for all 𝑘 ≥ 𝑘 , we have 𝜇 𝑘 = ̄ 𝜇 , and

<!-- formula-not-decoded -->

while the sequence { 𝐽 𝑘 } converges to 𝐽 ̄ 𝜇 .

Proof. The critical step of the proof is to show that for all ( 𝐽, 𝜇 ) with

<!-- formula-not-decoded -->

and all ( ̃ 𝐽, ̃ 𝜇 ) ∈ ̃  ( 𝐽, 𝜇 ) [cf. Eq. (2.7)], the following monotone decrease inequality holds

<!-- formula-not-decoded -->

where for all 𝓁 = 1 , … , 𝑚 , and 𝑥 ∈ 𝑋 ,

<!-- formula-not-decoded -->

with 𝐽 ̂ 0 = 𝐽 [cf. Eq. (2.4)], and

<!-- formula-not-decoded -->

[cf. Eq. (2.5)]. Indeed the relation (3.5) is proved starting from the right side, which is the assumption (3.2), and by using the definition of the algorithm, and the monotonicity Assumption 1.1 to prove first that 𝐽 ̂ 1 ≤ 𝑇 𝜇 𝐽 ≤ 𝐽 , and then by proceeding sequentially to the inequality ̂ 𝐽 𝑚 ≤ ̂ 𝐽 𝑚 -1 . In particular, at the typical step, assuming that 𝐽 ̂ 𝓁 -1 ≤ 𝐽 ̂ 𝓁 -2 , we show that 𝐽 ̂ 𝓁 ≤ 𝐽 ̂ 𝓁 -1 by writing

<!-- formula-not-decoded -->

where the first inequality follows by using the monotonicity Assumption 1.1 and the hypothesis 𝐽 ̂ 𝓁 -1 ≤ 𝐽 ̂ 𝓁 -2 . Finally, by applying 𝑇 ̃ 𝜇 to the relation ̂ 𝐽 𝑚 ≤ ̂ 𝐽 𝑚 -1 to obtain 𝑇 ̃ 𝜇 ̂ 𝐽 𝑚 ≤ 𝑇 ̃ 𝜇 ̂ 𝐽 𝑚 -1 , and by using the facts 𝐽 ̃ = ̂ 𝐽 𝑚 = 𝑇 ̃ 𝜇 ̂ 𝐽 𝑚 -1 , we obtain the leftmost relation 𝑇 ̃ 𝜇 𝐽 ̃ ≤ 𝐽 ̃ = ̂ 𝐽 𝑚 in Eq. (3.5). (Note that the contraction assumption is not needed for the preceding argument, and this is useful for applying this line of proof in other DP problem contexts.)

From Eq. (3.5), we see that the sequence of functions 𝐽 𝑘 converges monotonically to some function 𝐽 , and the same is true for all the sequences of intermediate functions 𝐽 𝑘 1 , … , 𝐽 𝑘 𝑚 -1 . For each 𝓁 , let the policies

<!-- formula-not-decoded -->

be equal to some policy ̄ 𝜇 [ 𝓁 ] = ( ̄ 𝜇 1 [ 𝓁 ] , … , ̄ 𝜇 𝑚 [ 𝓁 ] ) infinitely often, say for an infinite index set  𝓁 , (such a policy exists since the set of policies is finite). Then we will have for all 𝑥 ∈ 𝑋 and 𝓁 = 1 , … , 𝑚 ,

<!-- formula-not-decoded -->

for all 𝑘 ∈  𝓁 . By taking limit as 𝑘 → ∞ , 𝑘 ∈  𝓁 , and using the continuity of 𝐻 ( 𝑥, 𝑢, ⋅ ) (which is implied by the contraction property of 𝑇 ̄ 𝜇 [ 𝓁 ] ), we have

<!-- formula-not-decoded -->

as well as

<!-- formula-not-decoded -->

Eq. (3.7) and the contraction property of 𝑇 ̄ 𝜇 [ 𝓁 ] imply that 𝐽 is equal to the cost functions 𝐽 ̄ 𝜇 [ 𝓁 ] of all of the 𝑚 policies ̄ 𝜇 [ 𝓁 ] , 𝓁 = 1 , … , 𝑚 . In view of the uniqueness Assumption 3.1, this implies that all the policies ̄ 𝜇 [ 𝓁 ] , 𝓁 = 1 , … , 𝑚 , are equal to some policy ̄ 𝜇 , which has cost function 𝐽 , and in view of Eq. (3.8), satisfies

<!-- formula-not-decoded -->

It follows that ̄ 𝜇 is agent-by-agent optimal.

Finally, the preceding argument shows that 𝐽 is the cost function of every policy that is repeated infinitely often. Thus the uniqueness Assumption 3.1 implies that ̄ 𝜇 is the only policy that is repeated infinitely often. Since there are finitely many policies, it follows that 𝜇 𝑘 = ̄ 𝜇 for all 𝑘 after some index. Hence from the definition of the algorithm, the sequence { 𝐽 𝑘 } satisfies 𝐽 𝑘 +1 = 𝑇 ̄ 𝜇 𝐽 𝑘 for all 𝑘 after some index, which in view of the contraction Assumption 1.2, implies Eq. (3.4). □

Note that the preceding proposition does not guarantee convergence to the optimal policy (which is unique by Assumption 3.1). In particular, if our algorithm is started at a pair ( 𝐽 ̄ 𝜇 , ̄ 𝜇 ) , where ̄ 𝜇 is an agent-by-agent optimal policy, it will not move from ̄ 𝜇 [in fact it can be shown that this will happen even if the algorithm is started at a pair ( 𝐽 0 , ̄ 𝜇 ) , where 𝐽 0 is sufficiently close to 𝐽 ̄ 𝜇 ]. Thus every agent-by-agent optimal policy behaves like a ''local minimum'', with its own ''region of attraction'', and is a potential convergence limit of our algorithm. The limit will depend on the starting pair, as well as the order in which the agents select their components. The algorithm guarantees convergence to the optimal policy only under additional assumptions that guarantee that there are no additional agent-by-agent optimal policies. We postpone a discussion of this issue for Section 5.

Ensuring convergence to an optimal policy with randomization schemes

Another possibility to enhance the convergence properties of the algorithm, and ensure convergence to an optimal policy, is to enlarge the constraint sets

<!-- formula-not-decoded -->

in Eq. (2.4), to allow minimization over subsets of multiple control components. These subsets may be selected with some form of randomization: at some iterations minimize over a single control component as in iteration (2.4)-(2.5), while at some other randomly chosen iterations minimize over multiple or even all control components. Schemes of this type have been considered for the purpose of enhancing the convergence properties of asynchronous PI; see [3], Section 2.5.3. Randomization over sets of multiple control components can also be used in the context of the optimistic agent-by-agent PI methods of the next section, and they can similarly enhance their convergence properties.

We will not consider randomized control component selection schemes in this paper. Their analysis is similar to the one of [3], Section 2.5.3, their implementation is likely problem-dependent, and their practical performance is an interesting subject for further research. Their principal drawback is that simultaneous minimization over multiple control components can be very costly (depending on the number of components involved), even if it used in only a small proportion of the total number of iterations.

## 4. Agent-by-agent optimistic policy iteration

Let us now consider an optimistic PI variant where we introduce an infinite subset  ⊂ {0 , 1 , …} of the iterations, and the complementary subset of iterations 𝑘 ∉  . For the latter subset, we use the (less expensive) standard policy evaluation update 𝐽 𝑘 +1 = 𝑇 𝜇 𝑘 𝐽 𝑘 , and no policy update:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have the following convergence result:

Proposition 4.1 ( Optimistic PI Convergence to an Agent-by-Agent Optimal Policy ) . Let the assumptions of Proposition 3.1 hold, and let { 𝐽 𝑘 , 𝜇 𝑘 } be a sequence generated by the optimistic agent-by-agent PI algorithm (4.1) -(4.2) . Then there is an index 𝑘 such that for all 𝑘 ≥ 𝑘 , we will have 𝜇 𝑘 = ̄ 𝜇 , where ̄ 𝜇 is an agent-by-agent optimal policy, while the sequence { 𝐽 𝑘 } will converge to 𝐽 ̄ 𝜇 .

Proof. The proof is essentially identical to the one of Proposition 3.1. In particular, the definition of the optimistic PI algorithm allows the proof of the critical relation (3.5) to go through. □

The algorithm admits also a distributed implementation, whereby the iteration (4.1)-(4.2) is executed at the subset of times 𝑘 ∈  only for a subset 𝑋𝑘 of the states, while for the remaining states 𝑥 ∉ 𝑋𝑘 the values of 𝐽 𝑘 +1 ( 𝑥 ) and 𝜇 𝑘 +1 ( 𝑥 ) remain unchanged:

<!-- formula-not-decoded -->

In addition to the set  being infinite, there is a requirement here is that each state 𝑥 belongs infinitely often to some subset 𝑋𝑘 , so that there are infinitely many policy improvements at every state. Algorithms of this type have been proposed in the book [4], Section 2.2.3, and in [5]. The convergence proof of Proposition 3.1 still goes through; see also the proof of Prop. 2.5 of [4]. Note, however, that for this type of algorithm to be provably convergent, ( 𝐽 0 , 𝜇 0 ) must satisfy the condition 𝑇 𝜇 0 𝐽 0 ≤ 𝐽 0 [cf. Eq. (3.2)] even for discounted MDP, as demonstrated with counterexamples by Williams and Baird [22] (see also [23]).

## D. Bertsekas

In a more complex version of the algorithm, the information on the cost function iterates at each iteration is allowed to be out-of-date, while modifications are introduced to eliminate the need for the initial condition assumption of Eq. (3.2). Distributed asynchronous PI algorithms of this type have been proposed and analyzed in the paper by Bertsekas and Yu [24] [see also [25,26], and the books [5] (Section 2.6), and [3] (Section 2.6)]. See also the randomized optimistic PI algorithms of [3] (Section 2.5.3). Multiagent versions of such algorithms are a subject for further research.

## 5. Conditions for obtaining an optimal policy

We proved earlier that our multiagent VI algorithm will find an agent-by-agent optimal policy under our assumptions of Proposition 3.1, but this policy need not be optimal. We will now discuss approaches that can be used to show that the policy obtained is optimal, under the same or alternative assumptions. One possibility is to impose conditions under which every agent-by-agent optimal policy is optimal. To this end we introduce the following definition.

Definition 5.1 ( Component-by-Component Minimum ) . For a state 𝑥 and a function 𝐽 ∈  ( 𝑋 ) we say that a control 𝑢 = ( 𝑢 1 , … , 𝑢 𝑚 ) ∈ 𝑈 ( 𝑥 ) is a component-by-component minimum of 𝐻 at ( 𝑥, 𝐽 ) if

<!-- formula-not-decoded -->

where the sets 𝑈 𝓁 , ̄ 𝑢 ( 𝑥 ) are defined by

<!-- formula-not-decoded -->

Note that from the definition of agent-by-agent optimality, we have that ̄ 𝜇 is agent-by-agent optimal if for every 𝑥 ∈ 𝑋 , the control ̄ 𝜇 ( 𝑥 ) is a component-by-component minimum of 𝐻 at ( 𝑥, 𝐽 ̄ 𝜇 ) . We have the following proposition.

Proposition 5.1 ( Agent-by-Agent Optimality Criterion ) . Assume that for every state 𝑥 ∈ 𝑋 and policy ̄ 𝜇 such that ̄ 𝜇 ( 𝑥 ) is a componentby-component minimum of 𝐻 at ( 𝑥, 𝐽 ̄ 𝜇 ) , the control ̄ 𝜇 ( 𝑥 ) minimizes 𝐻 ( 𝑥, 𝑢, 𝐽 ̄ 𝜇 ) over 𝑢 ∈ 𝑈 ( 𝑥 ) . Then every agent-by-agent optimal policy is optimal.

Proof. Let ̄ 𝜇 be agent-by-agent optimal. Then from the definition of agent-by-agent optimality, we have that for all 𝑥 ∈ 𝑋 , ̄ 𝜇 ( 𝑥 ) is a component-by-component minimum of 𝐻 at ( 𝑥, 𝐽 ̄ 𝜇 ) . By our assumption, this implies that for all 𝑥 ∈ 𝑋 , ̄ 𝜇 ( 𝑥 ) minimizes 𝐻 ( 𝑥, 𝑢, 𝐽 ̄ 𝜇 ) over 𝑢 ∈ 𝑈 ( 𝑥 ) , or 𝑇 ̄ 𝜇 𝐽 ̄ 𝜇 = 𝑇𝐽 ̄ 𝜇 . From general properties of contractive abstract DP models (cf. [3], Chapter 2), we also have 𝑇 ̄ 𝜇 𝐽 ̄ 𝜇 = 𝐽 ̄ 𝜇 . Hence 𝑇 ̄ 𝜇 𝐽 ̄ 𝜇 = 𝑇𝐽 ̄ 𝜇 , which implies that 𝐽 ̄ 𝜇 = 𝐽 ∗ (cf. [3], Chapter 2), so ̄ 𝜇 is optimal. □

In view of Proposition 5.1, an important issue is to delineate sufficient conditions that guarantee that component-by-component minima of 𝐻 at ( 𝑥, 𝐽 𝜇 ) minimize 𝐻 ( 𝑥, 𝑢, 𝐽 𝜇 ) over 𝑢 ∈ 𝑈 ( 𝑥 ) . Somewhat similar questions have been addressed in two related contexts:

- (a) Team theory in connection with the notion of person-by-person optimality mentioned earlier.
- (b) The theory of convergence of coordinate descent methods in nonlinear optimization.

In the theory of teams and other related decentralized control problem formulations, the most prominent analytical issues arise when the team members select control components based on different information. By contrast in our framework the agents choose actions based on shared information, namely the current state 𝑥 𝑘 of the system. Because of this fundamental structural assumption, DP algorithms such as VI and PI apply to our problem, but do not apply to team problems with nonclassical information pattens. These problems are generally far more complicated than the ones considered here, as illustrated for linear systems and quadratic cost by the famous counterexample of [27].

In the theory of coordinate descent methods, the result most related to our context is that if a function 𝐹 ( 𝑦 1 , … , 𝑦 𝑚 ) of 𝑚 vectors 𝑦 1 , … , 𝑦 𝑚 is strictly convex and differentiable over the Cartesian product 𝑌 1 × ⋯ × 𝑌 𝑚 of closed convex sets 𝑌 1 , … , 𝑌 𝑚 , then a vector ̄ 𝑦 = ( ̄ 𝑦 1 , … , ̄ 𝑦 𝑚 ) is a global minimum of 𝐹 over 𝑌 1 × ⋯ × 𝑌 𝑚 if and only if it has the component-by-component minimization property

<!-- formula-not-decoded -->

Thus when 𝐹 is strictly convex and differentiable, the block coordinate descent method cannot get trapped into a solution that is a component-by-component minimum but is not a global minimum [this is not true, however, if 𝐹 is strictly convex but nondifferentiable, since the condition (5.2) may hold at vectors ̄ 𝑦 that are not global minima, and at which 𝐹 is nondifferentiable]. Some related results are known for the case where the sets 𝑌 𝓁 are discrete, under assumptions that can be viewed as discrete space substitutes for strict convexity; see e.g., de Waal and van Schuppen [15], and Bauso and Pesenti [28,29].

While the coordinate descent and the team theory results provide some analytical guidance, they do not apply directly to the DP context of this paper. The reason is that the mapping 𝐻 involves the functions 𝐽 𝜇 , whose properties have to be verified through analysis. We leave this line of investigation as a subject for further research, and we outline another analytical approach, which assumes continuous state and control spaces 𝑋 and 𝑈 , and is based on strict convexity and differentiability assumptions.

Continuous spaces, strict convexity, and differentiability

Let us remove the assumption that the state and control spaces 𝑋 and 𝑈 are finite, while continuing to assume that the control has 𝑚 components, 𝑢 = ( 𝑢 1 , … , 𝑢 𝑚 ) that are constrained by 𝑢 ∈ 𝑈 ( 𝑥 ) for all 𝑥 ∈ 𝑋 . We continue to adopt the monotonicity and contraction Assumptions 1.1 and 1.2, with the modification that  ( 𝑋 ) is replaced by the space  ( 𝑋 ) of bounded functions over 𝑋 , with respect to a weighted sup-norm. Moreover, we assume that the various minima over control components in the definition of the algorithms are attained. Models of this type have been analyzed extensively in the monograph [3] (Chapter 2), to which we refer for a detailed discussion. The definitions of agent-by-agent optimality and component-by-component minimum carry over without change to the continuous spaces setting, and so does the associated agent-by-agent optimality criterion (cf. Proposition 5.1). Furthermore, the key inequality (3.4) for the proof of the convergence result of Proposition 3.1 goes through, under the condition 𝑇 𝜇 0 𝐽 0 ≤ 𝐽 0 [cf. Eq. (3.2)]. As a result, the proof of monotonic decrease of the sequence { 𝐽 𝑘 } to some function 𝐽 goes through as well.

In conclusion, without assuming finiteness of the state and control spaces 𝑋 and 𝑈 , our algorithm, under the monotonicity and contraction Assumptions 1.1 and 1.2, and the condition 𝑇 𝜇 0 𝐽 0 ≤ 𝐽 0 [cf. Eq. (3.2)], converges monotonically to some 𝐽 , which can be seen to be pointwise bounded below by the optimal cost function 𝐽 ∗ , which belongs to  ( 𝑋 ) , so that 𝐽 ∈  ( 𝑋 ) . Further conditions, involving strict convexity and differentiability, need to be imposed to guarantee that 𝐽 = 𝐽 ∗ , that 𝐽 ∗ is convex and differentiable, and that an optimal policy can be obtained. A stochastic optimal control model, involving a linear system, a convex cost per stage, and convex state and control constraints, was formulated and analyzed in 1973 by the author [30], and is well suited for this purpose. We leave further analysis along this line as a subject for further research.

## 6. Concluding remarks

We have shown that in the context of multiagent problems, agent-by-agent versions of the VI algorithm and related optimistic PI algorithms have greatly reduced computational requirements, while still maintaining a meaningful convergence property. While these algorithms may terminate with a suboptimal policy that is agent-by-agent optimal, they can be dramatically more efficient than the standard VI and optimistic PI algorithms, which may be computationally intractable even for a moderate number of agents.

Several unresolved questions remain regarding algorithmic variations and conditions that guarantee that our algorithms obtain an optimal policy rather than one that is agent-by-agent optimal. Approximate versions of our algorithms of the type used in neurodynamic programming/reinforcement learning are also of interest, and are a subject for further investigation. Moreover, the basic idea of our approach, simplifying the minimization defining the VI operator while maintaining some form of convergence guarantee, can be extended in other directions to exploit special problem structures.

## Declaration of competing interest

One or more of the authors of this paper have disclosed potential or pertinent conflicts of interest, which may include receipt of payment, either direct or indirect, institutional support, or association with an entity in the biomedical field which may be perceived to have potential conflict of interest with this work. For full disclosure statements refer to https://doi.org/10.1016/j.rico. 2020.100003.

## References

- [1] Bertsekas DP. Multiagent rollout algorithms and reinforcement learning. 2020, arXiv preprint, arXiv:2002.07407.
- [2] Bertsekas DP. Multiagent reinforcement learning: Rollout and policy iteration. IEEE/CAA J Autom Sin 2020 [in press].
- [3] Bertsekas DP. Abstract dynamic programming. Belmont, MA: Athena Scientific; 2018, On-line at http://web.mit.edu/dimitrib/www/RLbook.html.
- [4] Bertsekas DP, Tsitsiklis JN. Neuro-dynamic programming. Belmont, MA: Athena Scientific; 1996.
- [5] Bertsekas DP. Dynamic programming and optimal control, vol. II. 4th ed. Belmont, MA: Athena Scientific; 2012.
- [6] Bertsekas DP. Reinforcement learning and optimal control. Belmont, MA: Athena Scientific; 2019.
- [7] Marschak J. Elements for a theory of teams. Manage Sci 1975;1:127-37.
- [8] Radner R. Team decision problems. Ann Math Stat 1962;33:857-81.
- [9] Witsenhausen H. On information structures, feedback, causality. SIAM J Control 1971;9:149-60.
- [10] Witsenhausen H. Separation of estimation and control for discrete time systems. Proc IEEE 1971;59:1557-66.
- [11] Witsenhausen H. Equivalent stochastic control problems. Math Control Signals Systems 1988;1:3-11.
- [12] Ho YC. Team decision theory and information structures. Proc IEEE 1980;68:644-54.
- [13] Krainak JC, Speyer J, Marcus S. Static team problems -part I: Sufficient conditions and the exponential cost criterion. IEEE Trans Automat Control 1982;27:839-48.
- [14] Krainak JC, Speyer J, Marcus S. Static team problems - part II: Affine control laws, projections, algorithms, and the LEGT problem. IEEE Trans Automat Control 1982;27:848-59.
- [15] de Waal PR, van Schuppen JH. A class of team problems with discrete action spaces: Optimality conditions based on multimodularity. SIAM J Control Optim 2000;38:875-92.
- [16] Nayyar A, Mahajan A, Teneketzis D. Decentralized stochastic control with partial history sharing: A common information approach. IEEE Trans Automat Control 2013;58:1644-58.
- [17] Nayyar A, Teneketzis D. Common knowledge and sequential team problems. IEEE Trans Automat Control 2019;64:5108-15.
- [18] Li Y, Tang Y, Zhang R, Li N. Distributed reinforcement learning for decentralized linear quadratic control: A derivative-free policy optimization approach. 2019, arXiv preprint arXiv:1912.09135.
- [19] Gupta A. Existence of team-optimal solutions in static teams with common information: A topology of information approach. SIAM J Control Optim 2020;58:998-1021.
- [20] Zoppoli R, Sanguineti M, Gnecco G, Parisini T. Neural approximations for optimal control and decision. Springer; 2020.
- [21] Bertsekas DP. Nonlinear programming. 3rd ed. Belmont, MA: Athena Scientific; 2016.

## D. Bertsekas

- [22] Williams RJ, Baird LC. Analysis of some incremental variants of policy iteration: First steps toward understanding actor-critic learning systems. Report NU-CCS-93-11, Boston, MA: College of Computer Science, Northeastern Univ.; 1993.
- [23] Bertsekas DP. Williams-baird counterexample for Q-factor asynchronous policy iteration. 2010, http://web.mit.edu/dimitrib/www/WilliamsBairdCounterexample.pdf.
- [24] Bertsekas DP, Yu H. Asynchronous distributed policy iteration in dynamic programming. In: Proc. of allerton conf. on communication, control and computing. Ill: Allerton Park; 2010, p. 1368-74.
- [25] Bertsekas DP, Yu H. Q-learning and enhanced policy iteration in discounted dynamic programming. Math. OR 2012;37:66-94.
- [26] Yu H, Bertsekas DP. Q-learning and policy iteration algorithms for stochastic shortest path problems. Ann Oper Res 2013;208:95-132.
- [27] Witsenhausen H. A counterexample in stochastic optimum control. SIAM J Control 1968;6:131-47.
- [28] Bauso D, Pesenti R. Generalized person-by-person optimization in team problems with binary decisions. In: Proc. 2008 American control conference. 2008. p. 717-22.
- [29] Bauso D, Pesenti R. Team theory and person-by-person optimization with binary decisions. SIAM J Control Optim 2012;50:3011-28.
- [30] Bertsekas DP. Linear convex stochastic control problems over an infinite horizon. IEEE Trans Aut Control 1973;AC-18:314-5.