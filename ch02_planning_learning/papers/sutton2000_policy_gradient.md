glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

## Policy Gradient Methods for Reinforcement Learning with Function Approximation

Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

AT&amp;T Labs - Research, 180 Park Avenue, Florham Park, NJ 07932

## Abstract

Function approximation is essential to reinforcement learning, but the standard approach of approximating a value function and determining a policy from it has so far proven theoretically intractable. In this paper we explore an alternative approach in which the policy is explicitly represented by its own function approximator, independent of the value function, and is updated according to the gradient of expected reward with respect to the policy parameters. Williams's REINFORCE method and actor-critic methods are examples of this approach. Our main new result is to show that the gradient can be written in a form suitable for estimation from experience aided by an approximate action-value or advantage function. Using this result, we prove for the first time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a locally optimal policy.

Large applications of reinforcement learning (RL) require the use of generalizing function approximators such neural networks, decision-trees, or instance-based methods. The dominant approach for the last decade has been the value-function approach, in which all function approximation effort goes into estimating a value function, with the action-selection policy represented implicitly as the 'greedy' policy with respect to the estimated values (e.g., as the policy that selects in each state the action with highest estimated value). The value-function approach has worked well in many applications, but has several limitations. First, it is oriented toward finding deterministic policies, whereas the optimal policy is often stochastic, selecting different actions with specific probabilities (e.g., see Singh, Jaakkola, and Jordan, 1994). Second, an arbitrarily small change in the estimated value of an action can cause it to be, or not be, selected. Such discontinuous changes have been identified as a key obstacle to establishing convergence assurances for algorithms following the value-function approach (Bertsekas and Tsitsiklis, 1996). For example, Q-learning, Sarsa, and dynamic programming methods have all been shown unable to converge to any policy for simple MDPs and simple function approximators (Gordon, 1995, 1996; Baird, 1995; Tsitsiklis and van Roy, 1996; Bertsekas and Tsitsiklis, 1996). This can occur even if the best approximation is found at each step before changing the policy, and whether the notion of 'best' is in the mean-squared-error sense or the slightly different senses of residual-gradient, temporal-difference, and dynamic-programming methods.

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

In this paper we explore an alternative approach to function approximation in RL. Rather than approximating a value function and using that to compute a deterministic policy, we approximate a stochastic policy directly using an independent function approximator with its own parameters. For example, the policy might be represented by a neural network whose input is a representation of the state, whose output is action selection probabilities, and whose weights are the policy parameters. Let θ denote the vector of policy parameters and ρ the performance of the corresponding policy (e.g., the average reward per step). Then, in the policy gradient approach, the policy parameters are updated approximately proportional to the gradient:

<!-- formula-not-decoded -->

where α is a positive-definite step size. If the above can be achieved, then θ can usually be assured to converge to a locally optimal policy in the performance measure ρ . Unlike the value-function approach, here small changes in θ can cause only small changes in the policy and in the state-visitation distribution.

In this paper we prove that an unbiased estimate of the gradient (1) can be obtained from experience using an approximate value function satisfying certain properties. Williams's (1988, 1992) REINFORCE algorithm also finds an unbiased estimate of the gradient, but without the assistance of a learned value function. REINFORCE learns much more slowly than RL methods using value functions and has received relatively little attention. Learning a value function and using it to reduce the variance of the gradient estimate appears to be essential for rapid learning. Jaakkola, Singh and Jordan (1995) proved a result very similar to ours for the special case of function approximation corresponding to tabular POMDPs. Our result strengthens theirs and generalizes it to arbitrary differentiable function approximators.

Our result also suggests a way of proving the convergence of a wide variety of algorithms based on 'actor-critic' or policy-iteration architectures (e.g., Barto, Sutton, and Anderson, 1983; Sutton, 1984; Kimura and Kobayashi, 1998). In this paper we take the first step in this direction by proving for the first time that a version of policy iteration with general differentiable function approximation is convergent to a locally optimal policy. Baird and Moore (1999) obtained a weaker but superficially similar result for their VAPS family of methods. Like policy-gradient methods, VAPS includes separately parameterized policy and value functions updated by gradient methods. However, VAPS methods do not climb the gradient of performance (expected long-term reward), but of a measure combining performance and valuefunction accuracy. As a result, VAPS does not converge to a locally optimal policy, except in the case that no weight is put upon value-function accuracy, in which case VAPS degenerates to REINFORCE. Similarly, Gordon's (1995) fitted value iteration is also convergent and value-based, but does not find a locally optimal policy.

## 1 Policy Gradient Theorem

We consider the standard reinforcement learning framework (see, e.g., Sutton and Barto, 1998), in which a learning agent interacts with a Markov decision process (MDP). The state, action, and reward at each time t ∈ ¶ 0 ↪ 1 ↪ 2 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ are denoted s t ∈ S , a t ∈ A , and r t ∈ /Rfractur respectively. The environment's dynamics are characterized by state transition probabilities, P a ss ′ = Pr ¶ s t +1 = s ′ ♣ s t = s↪ a t = a ♦ , and expected rewards R a s = E ¶ r t +1 ♣ s t = s↪ a t = a ♦ , ∀ s↪ s ′ ∈ S ↪ a ∈ A . The agent's decision making procedure at each time is characterized by a policy, π ( s↪ a↪ θ ) = Pr ¶ a t = a ♣ s t = s↪ θ ♦ , ∀ s ∈ S ↪ a ∈ A , where θ ∈ /Rfractur l , for l &lt;&lt; ♣S♣ , is a parameter vector. We assume that π is diffentiable with respect to its parameter, i.e., that ∂π ( s↪a ) ∂θ exists. We also usually write just π ( s↪ a ) for π ( s↪ a↪ θ ).

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

With function approximation, two ways of formulating the agent's objective are useful. One is the average reward formulation, in which policies are ranked according to their long-term expected reward per step, ρ ( π ):

<!-- formula-not-decoded -->

where d π ( s ) = lim t →∞ Pr ¶ s t = s ♣ s 0 ↪ π ♦ is the stationary distribution of states under π , which we assume exists and is independent of s 0 for all policies. In the average reward formulation, the value of a state-action pair given a policy is defined as

<!-- formula-not-decoded -->

The second formulation we cover is that in which there is a designated start state s 0 , and we care only about the long-term reward obtained from it. We will give our results only once, but they will apply to this formulation as well under the definitions

<!-- formula-not-decoded -->

Our first result concerns the gradient of the performance metric with respect to the policy parameter:

where γ ∈ [0 ↪ 1] is a discount rate ( γ = 1 is allowed only in episodic tasks). In this formulation, we define d π ( s ) as a discounted weighting of states encountered starting at s 0 and then following π : d π ( s ) = ∑ ∞ t =0 γ t Pr ¶ s t = s ♣ s 0 ↪ π ♦ .

Theorem 1 (Policy Gradient) . For any MDP, in either the average-reward or start-state formulations,

<!-- formula-not-decoded -->

Proof : See the appendix.

Marbach and Tsitsiklis (1998) describe a related but different expression for the gradient in terms of the state-value function, citing Jaakkola, Singh, and Jordan (1995) and Cao and Chen (1997). In both that expression and ours, the key point is that their are no terms of the form ∂d π ( s ) ∂θ : the effect of policy changes on the distribution of states does not appear. This is convenient for approximating the gradient by sampling. For example, if s was sampled from the distribution obtained by following π , then ∑ a ∂π ( s↪a ) ∂θ Q π ( s↪ a ) would be an unbiased estimate of ∂ρ ∂θ . Of course, Q π ( s↪ a ) is also not normally known and must be estimated. One approach is to use the actual returns, R t = ∑ ∞ k =1 r t + k -ρ ( π ) (or R t = ∑ ∞ k =1 γ k -1 r t + k in the start-state formulation) as an approximation for each Q π ( s t ↪ a t ). This leads to Williams's episodic REINFORCE algorithm, ∆ θ t ∝ ∂π ( s t ↪a t ) ∂θ R t 1 π ( s t ↪a t ) (the 1 π ( s t ↪a t ) corrects for the oversampling of actions preferred by π ), which is known to follow ∂ρ ∂θ in expected value (Williams, 1988, 1992).

## 2 Policy Gradient with Approximation

Now consider the case in which Q π is approximated by a learned function approximator. If the approximation is sufficiently good, we might hope to use it in place of Q π in (2) and still point roughly in the direction of the gradient. For example, Jaakkola, glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

Singh, and Jordan (1995) proved that for the special case of function approximation arising in a tabular POMDP one could assure positive inner product with the gradient, which is sufficient to ensure improvement for moving in that direction. Here we extend their result to general function approximation and prove equality with the gradient.

Let f w : S × A → /Rfractur be our approximation to Q π , with parameter w . It is natural to learn f w by following π and updating w by a rule such as ∆ w t ∝ ∂ ∂w [ ˆ Q π ( s t ↪ a t ) -f w ( s t ↪ a t )] 2 ∝ [ ˆ Q π ( s t ↪ a t ) -f w ( s t ↪ a t )] ∂f w ( s t ↪a t ) ∂w , where ˆ Q π ( s t ↪ a t ) is some unbiased estimator of Q π ( s t ↪ a t ), perhaps R t . When such a process has converged to a local optimum, then

<!-- formula-not-decoded -->

Theorem 2 (Policy Gradient with Function Approximation) . If f w satisfies (3) and is compatible with the policy parameterization in the sense that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then

Proof: Combining (3) and (4) gives

<!-- formula-not-decoded -->

which tells us that the error in f w ( s↪ a ) is orthogonal to the gradient of the policy parameterization. Because the expression above is zero, we can subtract it from the policy gradient theorem (2) to yield

<!-- formula-not-decoded -->

## 3 Application to Deriving Algorithms and Advantages

Given a policy parameterization, Theorem 2 can be used to derive an appropriate form for the value-function parameterization. For example, consider a policy that is a Gibbs distribution in a linear combination of features:

<!-- formula-not-decoded -->

where each φ sa is an l -dimensional feature vector characterizing state-action pair s↪ a . Meeting the compatibility condition (4) requires that

<!-- formula-not-decoded -->

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

so that the natural parameterization of f w is

<!-- formula-not-decoded -->

In other words, f w must be linear in the same features as the policy, except normalized to be mean zero for each state. Other algorithms can easily be derived for a variety of nonlinear policy parameterizations, such as multi-layer backpropagation networks.

The careful reader will have noticed that the form given above for f w requires that it have zero mean for each state: ∑ a π ( s↪ a ) f w ( s↪ a ) = 0 ↪ ∀ s ∈ S . In this sense it is better to think of f w as an approximation of the advantage function, A π ( s↪ a ) = Q π ( s↪ a ) -V π ( s ) (much as in Baird, 1993), rather than of Q π . Our convergence requirement (3) is really that f w get the relative value of the actions correct in each state, not the absolute value, nor the variation from state to state. Our results can be viewed as a justification for the special status of advantages as the target for value function approximation in RL. In fact, our (2), (3), and (5), can all be generalized to include an arbitrary function of state added to the value function or its approximation. For example, (5) can be generalized to ∂ρ ∂θ = ∑ s d π ( s ) ∑ a ∂π ( s↪a ) ∂θ [ f w ( s↪ a ) + v ( s )] ↪ where v : S → /Rfractur is an arbitrary function. (This follows immediately because ∑ a ∂π ( s↪a ) ∂θ = 0, ∀ s ∈ S .) The choice of v does not affect any of our theorems, but can substantially affect the variance of the gradient estimators. The issues here are entirely analogous to those in the use of reinforcement baselines in earlier work (e.g., Williams, 1992; Dayan, 1991; Sutton, 1984). In practice, v should presumably be set to the best available approximation of V π . Our results establish that that approximation process can proceed without affecting the expected evolution of f w and π .

## 4 Convergence of Policy Iteration with Function Approximation

Given Theorem 2, we can prove for the first time that a form of policy iteration with function approximation is convergent to a locally optimal policy.

Theorem 3 (Policy Iteration with Function Approximation) . Let π and f w be any differentiable function approximators for the policy and value function respectively that satisfy the compatibility condition (4) and for which max θ↪s↪a↪i↪j ♣ ∂ 2 π ( s↪a ) ∂θ i ∂θ j ♣ &lt; B &lt; ∞ . Let ¶ α k ♦ ∞ k =0 be any step-size sequence such that lim k →∞ α k = 0 and ∑ k α k = ∞ . Then, for any MDP with bounded rewards, the sequence ¶ ( θ k ↪ w k ) ♦ , defined by any θ 0 , π k = π ( · ↪ · ↪ θ k ), and

<!-- formula-not-decoded -->

converges such that lim k →∞ ∂ρ ( π k ) ∂θ = 0.

Proof: Our Theorem 2 assures that the θ k update is in the direction of the gradient. The bounds on ∂ 2 π ( s↪a ) ∂θ i ∂θ j and on the MDP's rewards together assure us that ∂ 2 ρ ∂θ i ∂θ j is also bounded. These, together with the step-size requirements, are the necessary conditions to apply Proposition 3.5 from page 96 of Bertsekas and Tsitsiklis (1996), which assures convergence to a local optimum. Q.E.D.

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

glyph[suppress]

## Acknowledgements

The authors wish to thank Martha Steenstrup and Doina Precup for comments, and Michael Kearns for insights into the notion of optimal policy under function approximation.

## References

Baird, L. C. (1993). Advantage Updating. Wright Lab. Technical Report WL-TR-93-1146. Baird, L. C. (1995). Residual algorithms: Reinforcement learning with function approximation. Proc. of the Twelfth Int. Conf. on Machine Learning , pp. 30-37. Morgan Kaufmann. Baird, L. C., Moore, A. W. (1999). Gradient descent for general reinforcement learning.

NIPS 11 . MIT Press.

Barto, A. G., Sutton, R. S., Anderson, C. W. (1983). Neuronlike elements that can solve difficult learning control problems. IEEE Trans. on Systems, Man, and Cybernetics 13 :835.

Bertsekas, D. P., Tsitsiklis, J. N. (1996). Neuro-Dynamic Programming . Athena Scientific.

Cao, X.-R., Chen, H.-F. (1997). Perturbation realization, potentials, and sensitivity analysis of Markov Processes, IEEE Trans. on Automatic Control 42 (10):1382-1393.

Dayan, P. (1991). Reinforcement comparison. In D. S. Touretzky, J. L. Elman, T. J. Sejnowski, and G. E. Hinton (eds.), Connectionist Models: Proceedings of the 1990 Summer School , pp. 45-51. Morgan Kaufmann.

Gordon, G. J. (1995). Stable function approximation in dynamic programming. Proceedings of the Twelfth Int. Conf. on Machine Learning , pp. 261-268. Morgan Kaufmann.

Gordon, G. J. (1996). Chattering in SARSA( λ ). CMU Learning Lab Technical Report.

Jaakkola, T., Singh, S. P., Jordan, M. I. (1995) Reinforcement learning algorithms for partially observable Markov decision problems, NIPS 7 , pp. 345-352. Morgan Kaufman.

Kimura, H., Kobayashi, S. (1998). An analysis of actor/critic algorithms using eligibility traces: Reinforcement learning with imperfect value functions. Proceedings of the Fifteenth International Conference on Machine Learning . Morgan Kaufmann.

Marbach, P., Tsitsiklis, J. N. (1998) Simulation-based optimization of Markov reward processes, technical report LIDS-P-2411, Massachusetts Institute of Technology.

Singh, S. P., Jaakkola, T., Jordan, M. I. (1994). Learning without state-estimation in partially observable Markovian decision problems. Proceedings of the Eleventh International Conference on Machine Learning , pp. 284-292. Morgan Kaufmann.

Sutton, R. S. (1984). Temporal Credit Assignment in Reinforcement Learning . Ph.D. thesis, University of Massachusetts, Amherst.

Sutton, R. S., Barto, A. G. (1998). Reinforcement Learning: An Introduction . MIT Press. Tsitsiklis, J. N. Van Roy, B. (1996). Feature-based methods for large scale dynamic programming. Machine Learning 22 :59-94.

Williams, R. J. (1988). Toward a theory of reinforcement-learning connectionist systems. Technical Report NU-CCS-88-3, Northeastern University, College of Computer Science.

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8 :229-256.

## Appendix: Proof of Theorem 1

We prove the theorem first for the average-reward formulation and then for the startstate formulation.

<!-- formula-not-decoded -->

glyph[suppress]

glyph[suppress]

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing both sides over the stationary distribution d π ,

<!-- formula-not-decoded -->

but since d π is stationary,

<!-- formula-not-decoded -->

For the start-state formulation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

after several steps of unrolling (7), where Pr ( s → x↪ k↪ π ) is the probability of going from state s to state x in k steps under policy π . It is then immediate that

<!-- formula-not-decoded -->