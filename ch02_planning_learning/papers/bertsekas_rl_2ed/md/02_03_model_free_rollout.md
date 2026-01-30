# 2.3.7-2.3.10: Model-Free Rollout & Inference

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 208-220
**Topics:** expert rollout, model-free rollout, world model, local search, n-grams, transformers, HMM, Markov chains

---

1990s, indicate consistently good performance of rollout; see the last section of this chapter for a bibliography. The DP textbook [Ber17a] provides some detailed worked-out examples (Chapter 6, Examples 6.4.2, 6.4.5, 6.4.6, and Exercises 6.11, 6.14, 6.15, 6.16). The price for the performance improvement is extra computation that is typically equal to the computation time of the base heuristic times a factor that is a low order polynomial of N . It is generally hard to quantify the amount of performance improvement, but the computational results obtained from the case studies are consistent with the Newton step interpretations that we discussed in Chapter 1.

The books [Ber19a] (Section 2.5.1) and [Ber20a] (Section 3.1) show that the sequential improvement condition is satisfied in the context of MPC, and is the underlying reason for the stability properties of the MPC scheme. On the other hand the base heuristic underlying the classical form of the MPC scheme is not sequentially consistent (see the preceding references).

Generally, the sequential improvement condition may not hold for a given base heuristic. This is not surprising since any heuristic (no matter how inconsistent or silly) is in principle admissible to use as base heuristic. Here is an example:

## Example 2.3.3 (Sequential Improvement Violation)

Consider the 2-stage problem shown in Fig. 2.3.5, which involves two states at each of stages 1 and 2, and the controls shown. Suppose that the unique optimal trajectory is ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ), and that the base heuristic produces this optimal trajectory starting at x 0 . The rollout algorithm chooses a control at x 0 as follows: it runs the base heuristic to construct a trajectory starting from x ∗ 1 and ˜ x 1 , with corresponding costs H 1 ( x ∗ 1 ) and H 1 (˜ x 1 ). If

<!-- formula-not-decoded -->

the rollout algorithm rejects the optimal control u ∗ 0 in favor of the alternative control ˜ u 0 . The inequality above will occur if the base heuristic chooses ¯ u 1 at x ∗ 1 (there is nothing to prevent this from happening, since the base heuristic is arbitrary), and moreover the cost g 1 ( x ∗ 1 ↪ ¯ u 1 ) + g 2 (˜ x 2 ), which is equal to H 1 ( x ∗ 1 ) is high enough.

Let us also verify that if the inequality (2.15) holds then the heuristic is not sequentially improving at x 0 , i.e., that

<!-- formula-not-decoded -->

Indeed, this is true because H 0 ( x 0 ) is the optimal cost

<!-- formula-not-decoded -->

and must be smaller than both

<!-- formula-not-decoded -->

ne bast ce u1 :

Optimal Trajectory

P

к, Йк, Tk+1, Uk+

Nn"

1, IN

Chosen by Base Heuristic at xo

*5

N

High Cost Transition

Chosen by Heuristic at xi

й1

Optimal Trajectory Chosen by Base Heuristic at

High Cost Transition Chosen by Heuristic at

Violates Sequential Improvement 2.4.3, 2.4.4 2.4.2 3.3,

Violates Sequential Improvement 2.4.3, 2.4.4 2.4.2 3.3,

<!-- image -->

Rollout Choice

Figure 2.3.5 A 2-stage problem with states x ∗ 1 ↪ ˜ x 1 at stage 1, and states x ∗ 2 ↪ ˜ x 2 at stage 2. The controls and corresponding transitions are as shown in the figure. The rollout choice at the initial state x 0 is strictly suboptimal, while the base heuristic choice is optimal. The reason is that the base heuristic is not sequentially improving and makes the suboptimal choice u 1 at x ∗ 1 , but makes the di ff erent (optimal) choice u ∗ 1 when run from x 0 .

which is the cost of the trajectory ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u 1 ↪ ˜ x 2 ), and

<!-- formula-not-decoded -->

which is the cost of the trajectory ( x 0 ↪ ˜ u 0 ↪ ˜ x 1 ↪ ˜ u 1 ↪ ˜ x 2 ).

The preceding example and the monotonicity property (2.14) suggest a simple enhancement to the rollout algorithm, which detects when the sequential improvement condition is violated and takes corrective measures. In this algorithmic variant, called fortified rollout , we maintain the best trajectory obtained so far, and keep following that trajectory up to the point where we discover another trajectory that has improved cost (see the next section).

## 2.3.2 The Fortified Rollout Algorithm

In this section we describe a rollout variant that implicitly enforces the sequential improvement property. This variant, called the fortified rollout algorithm , starts at x 0 , and generates step-by-step a sequence of states ¶ x 0 ↪ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x N ♦ and corresponding sequence of controls. Upon reaching state x k we have the trajectory

<!-- formula-not-decoded -->

that has been constructed by rollout, called permanent trajectory , and we also store a tentative best trajectory

<!-- formula-not-decoded -->

Rollout

Choice

Il end- trajectory and

every

к, Йк)

algori to Tr.

., UN-1, IN.

of Tk,

Uk)

le perr d (Uk, Fk+1) to

It (It, Ut)

NIN

v. To se Tr.

inchanged: Tk+

Pk

Tk+

= Tk.

C(T1)

XO

Tentative Best Trajectory Tk

Xk+1

Current State

Xk

йк

Permanent trajectory P

Min Q-factor choice

Figure 2.3.6 Schematic illustration of fortified rollout. After k steps, we have constructed the permanent trajectory

<!-- image -->

<!-- formula-not-decoded -->

and the tentative best trajectory

<!-- formula-not-decoded -->

the best end-to-end trajectory computed so far. We now run the rollout algorithm at x k , i.e., we find the control ˜ u k that minimizes over u k the sum of g k ( x k ↪ u k ) plus the heuristic cost from the state x k +1 = f k ( x k ↪ u k ), and the corresponding trajectory

<!-- formula-not-decoded -->

If the cost of the end-to-end trajectory ˜ T k is lower than the cost of T k , we add (˜ u k ↪ ˜ x k +1 ) to the permanent trajectory and set the tentative best trajectory to T k +1 = ˜ T k . Otherwise we add ( u k ↪ x k +1 ) to the permanent trajectory and keep the tentative best trajectory unchanged: T k +1 = T k glyph[triangleright]

with corresponding cost

<!-- formula-not-decoded -->

The tentative best trajectory T k is the end-to-end trajectory that has minimum cost out of all end-to-end trajectories computed up to stage k of the algorithm. Initially, T 0 is the trajectory generated by the base heuristic starting at the initial state x 0 . The idea now is to discard the suggestion of the rollout algorithm at every state x k where it produces a trajectory that is inferior to T k , and use T k instead (see Fig. 2.3.6).

In particular, upon reaching state x k , we run the rollout algorithm as earlier, i.e., for every u k ∈ U k ( x k ) and next state x k +1 = f k ( x k ↪ u k ) ↪ we

Heuristic

run the base heuristic from x k +1 , and find the control ˜ u k that gives the best trajectory, denoted

<!-- formula-not-decoded -->

with corresponding cost

<!-- formula-not-decoded -->

Whereas the ordinary rollout algorithm would choose control ˜ u k and move to ˜ x k +1 , the fortified algorithm compares C ( T k ) and C ( ˜ T k ), and depending on which of the two is smaller, chooses u k or ˜ u k and moves to x k +1 or to ˜ x k +1 , respectively. In particular, if

<!-- formula-not-decoded -->

the algorithm sets the next state and corresponding tentative best trajectory to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it sets the next state and corresponding tentative best trajectory to

<!-- formula-not-decoded -->

In other words the fortified rollout at x k follows the current tentative best trajectory T k unless a lower cost trajectory ˜ T k is discovered by running the base heuristic from all possible next states x k +1 . It follows that at every state the tentative best trajectory has no larger cost than the initial tentative best trajectory, which is the one produced by the base heuristic starting from x 0 . Moreover, it can be seen that if the base heuristic is sequentially improving, the rollout algorithm and its fortified version coincide. Experimental evidence suggests that it is often important to use the fortified version if the base heuristic is not known to be sequentially improving. Fortunately, the fortified version involves hardly any additional computational cost.

As expected, when the base heuristic generates an optimal trajectory, the fortified rollout algorithm will also generate the same trajectory. This is illustrated by the following example.

The base heuristic may also be run from a subset of the possible next states x k +1 , as in the case where a simplified version of rollout is used; cf. Section 2.3.4. Then fortified rollout will still guarantee a cost improvement property.

and if

## Example 2.3.4

Let us consider the application of the fortified rollout algorithm to the problem of Example 2.3.3 and see how it addresses the issue of cost improvement. The fortified rollout algorithm stores as initial tentative best trajectory the optimal trajectory ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ) generated by the base heuristic at x 0 . Then, starting at x 0 , it runs the heuristic from x ∗ 1 and ˜ x 1 , and (despite the fact that the ordinary rollout algorithm prefers going to ˜ x 1 rather than x ∗ 1 ) it discards the control ˜ u 0 in favor of u ∗ 0 , which is dictated by the tentative best trajectory. It then sets the tentative best trajectory to ( x 0 ↪ u ∗ 0 ↪ x ∗ 1 ↪ u ∗ 1 ↪ x ∗ 2 ).

We finally note that the fortified rollout algorithm can be used in a di ff erent setting to restore and maintain the cost improvement property. Suppose in particular that the rollout minimization at each step is performed with approximations. For example the control u k may have multiple independently constrained components, i.e.,

<!-- formula-not-decoded -->

Then, to take advantage of distributed computation, it may be attractive to decompose the optimization over u k in the rollout algorithm,

<!-- formula-not-decoded -->

into an (approximate) parallel optimization over the components u i k (or subgroups of these components). However, as a result of approximate optimization over u k , the cost improvement property may be degraded, even if the sequential improvement assumption holds. In this case by maintaining the tentative best trajectory, starting with the one produced by the base heuristic at the initial condition, we can ensure that the fortified rollout algorithm, even with approximate minimization, will not produce an inferior solution to the one of the base heuristic.

## 2.3.3 Using Multiple Base Heuristics - Parallel Rollout

In many problems, several promising heuristics may be available. It is then possible to use all of these heuristics in the rollout framework. The idea is to construct a superheuristic , which selects the best out of the trajectories produced by the entire collection of heuristics. The superheuristic can then be used as the base heuristic for a rollout algorithm.

A related practically interesting possibility is to introduce a partition of the state space into subsets, and a collection of multiple heuristics that are specially tailored to the subsets. We may then select the appropriate heuristic to use on each subset of the partition. In fact one may use a collection of multiple heuristics tailored to each subset of the state space partition, and at each state, select out of all the heuristics that apply, the one that yields minimum cost.

X1

Current State

•••

States at

Time N

Heuristic 1

Heuristic 2

Next States

Xk+1

Uk

Heuristic 3

Minimal Q-Factor

<!-- image -->

) States at Time

Heuristic 1 Heuristic 2 Heuristic 3

Heuristic 1 Heuristic 2 Heuristic 3

Figure 2.3.7 Schematic illustration of parallel rollout. From every possible next state x k +1 , we run each of the heuristics, we compute the Q-factor corresponding to each possible control u k and heuristic, and select the control with minimal Qfactor. Thus the number of Q-factors that are compared is equal to m · h , where m is the number of controls available at state x k and h is the number of heuristics used in the parallel rollout. In the figure, there are six Q-factors to compare, and the best/minimal Q-factor corresponds to the pair ( u ′ k ↪ Heuristic 1), as indicated, thus leading to selection of u ′ k as the rollout control at x k .

In particular, let us assume that we have m heuristics, and that the /lscript th of these, given a state x k +1 , produces a trajectory

<!-- formula-not-decoded -->

and corresponding cost C ( ˜ T /lscript k +1 ). The superheuristic then produces at x k +1 the trajectory ˜ T /lscript k +1 for which C ( ˜ T /lscript k +1 ) is minimum. The rollout algorithm selects at state x k the control u k that minimizes the minimal Q-factor:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the cost of the trajectory ( x k ↪ u k ↪ ˜ T /lscript k +1 ). Note that the Q-factors of the di ff erent heuristics can be computed independently and in parallel. In view of this fact, the rollout scheme just described is sometimes referred to as parallel rollout ; see Fig. 2.3.7.

An interesting property, which can be readily verified by using the definitions, is that if all the heuristics are sequentially improving, the same is true for the superheuristic , something that is also suggested by the proof of the cost improvement property of Fig. 2.3.4. Indeed, let us write the sequential improvement condition (2.13) for each of the base heuristics

<!-- formula-not-decoded -->

l=1

l=1

., m

..., m

XO

Cont

X1

Current State

.. •

Xk

States at

Time N

Trajectory 1

Trajectory 2

Next States

Xk+1

Uk

Trajectory 3

Minimal Q-Factor

<!-- image -->

) States at Time

Trajectory 1 Trajectory 2 Trajectory 3

Trajectory 1 Trajectory 2 Trajectory 3

Figure 2.3.8 Schematic illustration of general rollout. From every possible next state x k +1 , we run a number of trajectories, we compute the Q-factor corresponding to all the possible (control, trajectory) pairs, and select the control with minimal Q-factor. The number of Q-factors compared is at most m · h , where m is the number of controls available at state x k and h is the maximum number of trajectories generated from each possible next state x k +1 . In the figure, there are five Q-factors to compare, and the best/minimal Q-factor corresponds to the pair ( u ′ k ↪ Trajectory 1), as indicated, thus leading to selection of u ′ k as the rollout control at x k .

where ˜ Q /lscript k ( x k ↪ u k ) and H /lscript k ( x k ) are Q-factors and heuristic costs that correspond to the /lscript th heuristic. Then by taking minimum over /lscript , we have

<!-- formula-not-decoded -->

for all x k and k . By interchanging the order of the minimizations of the left side, we then obtain

<!-- formula-not-decoded -->

which is precisely the sequential improvement condition (2.13) for the superheuristic.

## 2.3.4 Using Multiple Rollout Trajectories - General Rollout

A generalization of the parallel rollout scheme, called general rollout , involves the generation of multiple trajectories starting from each possible next state x k +1 (see Fig. 2.3.8). Here, similar to parallel rollout, for each possible x k +1 , we compute multiple Q-factors, but we do not require that

each of these Q-factors correspond to a distinct base heuristic. Instead the Q-factors may be associated with multiple trajectories, however generated, which start at x k +1 and end at the terminal time N .

Note that some of the trajectories that start from a state x k +1 may share some intermediate states from times k +2 to N , and they may also be randomly generated. As a result the general rollout framework is well suited for the use of large language models (LLM) as base heuristics. The reason is that the trajectories generated by LLM often involve randomizations and may also involve suboptimization over multiple trajectories. For example, LLM may consider multiple next word choices at selected states, and generate multiple trajectories corresponding to each of these choices.

Similar to parallel rollout, general rollout is not di ff erent conceptually from the basic rollout algorithm with a single superheuristic that involves optimization over multiple trajectories starting at any possible next state x k +1 (cf. Section 2.3.3). One possibility in general rollout is to carry out this optimization by DP, over a tree of trajectories that is rooted at x k +1 . We finally note an important point: with multiple trajectories, the verification of properties such as sequential consistency and sequential improvement, may become complicated. Thus the use of rollout fortification (cf. Section 2.3.2) may be essential to guarantee a cost improvement property.

## 2.3.5 Simplified Rollout Algorithms

We will now consider a rollout variant, called simplified rollout , which is motivated by problems where the control constraint set U k ( x k ) is either infinite or finite but very large. Then the minimization

<!-- formula-not-decoded -->

[cf. Eqs. (2.9) and (2.10)], may be unwieldy, since the number of Q-factors

<!-- formula-not-decoded -->

is accordingly infinite or large.

To remedy this situation, we may replace U k ( x k ) with a smaller finite subset U k ( x k ):

<!-- formula-not-decoded -->

The rollout control ˜ θ k ( x k ) in this variant is one that attains the minimum of ˜ Q k ( x k ↪ u k ) over u k ∈ U k ( x k ):

<!-- formula-not-decoded -->

An example is when U k ( x k ) results from discretization of an infinite set U k ( x k ). Another possibility is when by using some preliminary approximate optimization, perhaps using a trained neural network or other heuristic method, we can identify a subset U k ( x k ) of promising controls, and to

save computation, we restrict attention to this subset. A related possibility is to generate U k ( x k ) by some iterative or random search method that explores intelligently the set U k ( x k ) with the aim to minimize ˜ Q k ( x k ↪ u k ) [cf. Eq. (2.16)].

It turns out that the proof of the cost improvement property of Prop. 2.3.2,

<!-- formula-not-decoded -->

goes through if the following modified sequential improvement property holds:

<!-- formula-not-decoded -->

This can be seen by verifying that Eq. (2.18) is su ffi cient to guarantee that the monotone improvement Eq. (2.14) is satisfied. The condition (2.18) is very simple to satisfy if the base heuristic is sequentially consistent, in which case the control u k selected by the base heuristic satisfies

<!-- formula-not-decoded -->

In particular, for the property (2.18) to hold, it is su ffi cient that U k ( x k ) contains the base heuristic choice u k .

The idea of replacing the minimization (2.16) by the simpler minimization (2.17) can be extended. In particular, by working through the preceding argument, it can be seen that any policy

<!-- formula-not-decoded -->

such that ˜ θ k ( x k ) satisfies the condition

<!-- formula-not-decoded -->

for all x k and k , guarantees the modified sequential improvement property (2.18), and hence also the cost improvement property . A prominent example of such an algorithm arises in the multiagent case where u has m components, u = ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ), and the minimization over U 1 k ( x k ) × · · · × U m k ( x k ) is replaced by a sequence of single component minimizations, one-componentat-a-time; cf. Section 1.6.7. Of course in the multiagent case, the onecomponent-at-a-time implementation has an additional favorable property: it can be viewed as rollout (without simplification) for a modified but equivalent DP problem (see Section 1.6.7).

## 2.3.6 Truncated Rollout with Terminal Cost Approximation

An important variation of rollout algorithms is truncated rollout with terminal cost approximation. Here the rollout trajectories are obtained by running the base policy from the leaf nodes of the lookahead tree, but they

are truncated after a given number of steps, while a terminal cost approximation is added to the heuristic cost to compensate for the resulting error. This is important for problems with a large number of stages, and it is also essential for infinite horizon problems where the rollout trajectories have infinite length.

One possibility that works well for many problems is to simply set the terminal cost approximation to zero. Alternatively, the terminal cost function approximation may be obtained by using some sophisticated o ff -line training process that may involve an approximation architecture such as a neural network, or by using some heuristic calculation based on a simplified version of the problem. This form of truncated rollout may also be viewed as an intermediate approach between standard rollout where there is no truncation (and hence no cost function approximation), and approximation in value space without any rollout.

## 2.3.7 Rollout with an Expert - Model-Free Rollout

We will now consider a rollout algorithm for discrete deterministic optimization for the case where we do not know the cost function and the constraints of the problem . Instead we have access to a base heuristic, and also a human or software 'expert' who can rank any two feasible solutions without assigning numerical values to them.

We consider the general discrete optimization problem of selecting a control sequence u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) to minimize a function G ( u ). For simplicity we assume that each component u k is constrained to lie in a given constraint set U k , but extensions to more general constraint sets are possible. We assume the following:

- (a) A base heuristic with the following property is available: Given any k &lt; N -1, and a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), it generates, for every ˜ u k +1 ∈ U k +1 , a complete feasible solution by concatenating the given partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) with a sequence (˜ u k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ). This complete feasible solution is denoted

<!-- formula-not-decoded -->

The base heuristic is also used to start the algorithm from an artificial empty solution, by generating all components ˜ u 0 ∈ U 0 and a complete feasible solution (˜ u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ u N -1 ), starting from each ˜ u 0 ∈ U 0 .

- (b) An 'expert' is available that can compare any two feasible solutions u and u , in the sense that he/she can determine whether

<!-- formula-not-decoded -->

It can be seen that deterministic rollout can be applied to this problem, even though the cost function G is unknown. The reason is that the

Base

Heuristic

Complete

Solutions www

Expert Ranks Complete Solutions

Sk U0, ..., Ик, k+1), k+1 € Uk+1

Base Heuristic Expert Ranks Complete Solutions

Base Heuristic Expert Ranks Complete Solutions www

Figure 2.3.9 Schematic illustration of the rollout with an expert for minimizing G ( u ) subject to

<!-- image -->

<!-- formula-not-decoded -->

We assume that we do not know G and/or U 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ U N -1 . Instead we have a base heuristic, which given a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), outputs all next controls ˜ u k +1 ∈ U k +1 , and generates from each a complete solution

<!-- formula-not-decoded -->

Also, we have a human or software 'expert' that can rank any two complete solutions without assigning numerical values to them. The control that is selected from U k +1 by the rollout algorithm is the one whose corresponding complete solution is ranked best by the expert.

rollout algorithm uses the cost function only as a means of ranking complete solutions in terms of their cost. Hence, if the ranking of any two solutions can be revealed by the expert, this is all that is needed. In fact, the constraint sets U 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ U N -1 need not be known either, as long as they can be generated by the base heuristic. Thus, the rollout algorithm can be described as follows (see Fig. 2.3.9):

We start with an artificial empty solution, and at the typical step, given the partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), k &lt; N -1, we use the base heuristic

Note that for this to be true, it is important that the problem is deterministic, and that the expert ranks solutions using some underlying (though unknown) cost function. In particular, the expert's rankings should have a transitivity property: if u is ranked better than u ′ and u ′ is ranked better than u ′′ , then u is ranked better than u ′′ .

Current Partial Solution glyph[triangleright] ↪

to generate all possible one-step-extended solutions

<!-- formula-not-decoded -->

and the set of complete solutions

<!-- formula-not-decoded -->

We then use the expert to rank this set of complete solutions. Finally, we select the component u k +1 that is ranked best by the expert, extend the partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) by adding u k +1 , and repeat with the new partial solution ( u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ u k +1 ).

Except for the (mathematically inconsequential) use of an expert rather than a cost function, the preceding rollout algorithm can be viewed as a special case of the one given earlier. As a result several of the rollout variants that we have discussed so far (rollout with multiple heuristics, simplified rollout, and fortified rollout) can also be easily adapted.

## Example 2.3.5 (Active Learning - Dataset Enrichment)

Let us consider a machine learning context whereby we have a dataset that we want to enrich with data selected from another dataset. The enrichment is to be done sequentially, one data point at a time, up to a given maximum size of N data points. We are interested in finding a sequential enrichment policy that optimizes some terminal cost function. This is the active learning context that we discussed briefly in Section 1.7.5.

We can approach the problem in terms of the discrete optimization framework of this section. The state is a partial data set ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ), consisting of the initial data set, denoted u 0 , and the k additional data points u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k that have been selected up to time k . The current state ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ) is augmented with a data point u k +1 from the set of the remaining (unused) data points, and each possible new state is evaluated using a base heuristic and an expert, according to our framework of this section. An interesting variation here is to allow the option for an early termination of the dataset enrichment process; this can be done by allowing a dummy/empty data point selection at each time. An additional possibility for early termination is to introduce a cost for each new data point addition, so that the dataset enrichment process stops naturally, when there is little cost improvement.

## Example 2.3.6 (Using a Large Language Model for Rollout)

The problem of minimizing G ( u ) over a constraint set can be viewed as an N -gram optimization problem, discussed in Example 1.6.2, where the text window consists of a partial solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k ↪ u k +1 ) (preceded by N -k -2 'default' words to bring the total to N ). We noted in that example that a GPT can be used as a policy that generates next words within the context of N -gram optimization. Thus the problem can be addressed within the model-free rollout framework of this section, whereby a GPT is used as a

Partial Folding Software Critic Software

Complete Folding Current Partial Folding

Partial Folding Software Critic Software

Figure 2.3.10 Schematic illustration of rollout for the RNA folding problem. The current state is the partial folding depicted on the left side. There are at most three choices for control at each state.

<!-- image -->

base heuristic for completion of partial solutions. The main issues with this approach are how to train a GPT for the problem at hand, and also in the absence of an explicit cost function G ( u ), how to properly design the expert software for comparing complete solutions. Both of these issues are actively researched at present.

## Example 2.3.7 (RNA Folding)

In a classical problem from computational biology, we are given a sequence of nucleotides, represented by circles in Fig. 2.3.10, and we want to 'fold' the sequence in an 'interesting' way (introduce pairings of nucleotides that result in an 'interesting' structure). There are some constraints on which pairings are possible, but we will not go into the details of this (some types of constraints may require the use of the constrained rollout framework of Section 2.5). A common constraint is that the pairings should not 'cross,' i.e., given a pairing ( i 1 ↪ i 2 ) there should be no pairing ( i 3 ↪ i 4 ) where either i 3 &lt; i 1 and i 1 &lt; i 4 &lt; i 2 , or i 1 &lt; i 3 &lt; i 2 and i 2 &lt; i 4 . This type of problem has a long history of solution by DP, starting with the paper by Zuker and Stiegler [ZuS81]. There are several formulations, where the aim is to optimize some criterion, e.g., the number of pairings, or the 'energy' of the folding. However, biologists do not agree on a suitable criterion, and have developed software to generate 'reasonable' foldings, based on semi-heuristic reasoning. We will develop a rollout approach that makes use of such software without discussing their underlying principles.

We formulate the folding problem as a discrete optimization problem involving a pairing decision at each nucleotide in the sequence with at most three choices (open a pairing, close a pairing, do nothing); see Fig. 2.3.10. To apply rollout, we need a base heuristic, which given a partial folding, generates a complete folding (this is the partial folding software shown in Fig.

Expert Rollout with Base O

Value iterations Compares Complete Foldings