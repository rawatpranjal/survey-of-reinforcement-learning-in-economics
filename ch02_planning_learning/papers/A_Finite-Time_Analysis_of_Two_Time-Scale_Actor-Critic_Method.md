## A Finite-Time Analysis of Two Time-Scale Actor-Critic Methods

## Yue Wu

Department of Computer Science University of California, Los Angeles Los Angeles, CA 90095 ywu@cs.ucla.edu

## Pan Xu

## Weitong Zhang

Department of Computer Science University of California, Los Angeles Los Angeles, CA 90095 weightzero@cs.ucla.edu

## Quanquan Gu

Department of Computer Science University of California, Los Angeles Los Angeles, CA 90095 panxu@cs.ucla.edu

Department of Computer Science University of California, Los Angeles Los Angeles, CA 90095 qgu@cs.ucla.edu

## Abstract

Actor-critic (AC) methods have exhibited great empirical success compared with other reinforcement learning algorithms, where the actor uses the policy gradient to improve the learning policy and the critic uses temporal difference learning to estimate the policy gradient. Under the two time-scale learning rate schedule, the asymptotic convergence of AC has been well studied in the literature. However, the non-asymptotic convergence and finite sample complexity of actor-critic methods are largely open. In this work, we provide a non-asymptotic analysis for two time-scale actor-critic methods under non-i.i.d. setting. We prove that the actor-critic method is guaranteed to find a first-order stationary point (i.e., ‖∇ J ( θ ) ‖ 2 2 ≤ /epsilon1 ) of the non-concave performance function J ( θ ) , with ˜ O ( /epsilon1 -2 . 5 ) sample complexity. To the best of our knowledge, this is the first work providing finite-time analysis and sample complexity bound for two time-scale actor-critic methods.

## 1 Introduction

Actor-Critic (AC) methods [2, 16] aim at combining the advantages of actor-only methods and criticonly methods, and have achieved great empirical success in reinforcement learning [31, 1]. Specifically, actor-only methods, such as policy gradient [28] and trust region policy optimization [24], utilize a parameterized policy function class and improve the policy by optimizing the parameters of some performance function using gradient ascent, whose exact form is characterized by the Policy Gradient Theorem [28]. Actor-only methods can be naturally applied to continuous setting but suffer from high variance when estimating the policy gradient. On the other hand, critic-only methods, such as temporal difference learning [26] and Q-learning [32], focus on learning a value function (expected cumulative rewards), and determine the policy based on the value function, which is recursively approximated based on the Bellman equation. Although the critic-only methods can efficiently learn a satisfying policy under tabular setting [14], they can diverge with function approximation under continuous setting [33]. Therefore, it is natural to combine actor and critic based methods to achieve the best of both worlds. The principal idea behind actor-critic methods is simple: the critic tries to learn the value function, given the policy from the actor, while the actor can estimate the policy gradient based on the approximate value function provided by the critic.

If the actor is fixed, the policy remains unchanged throughout the updates of the critic. Thus one can use policy evaluation algorithm such as temporal difference (TD) learning [27] to estimate the value function (critic). After many steps of the critic update, one can expect a good estimation of the value function, which in turn enables an accurate estimation of the policy gradient for the actor. A more favorable implementation is the so-called two time-scale actor-critic algorithm, where the actor and the critic are updated simultaneously at each iteration except that the actor changes more slowly (with a small step size) than the critic (with a large step size). In this way, one can hope the critic will be well approximated even after one step of update. From the theoretical perspective, the asymptotic analysis of two time-scale actor-critic methods has been established in [6, 16]. In specific, under the assumption that the ratio of the two time-scales goes to infinity (i.e. lim t →∞ β t /α t = ∞ ), the asymptotic convergence is guaranteed through the lens of the two time-scale ordinary differential equations(ODE), where the slower component is fixed and the faster component converges to its stationary point. This type of analysis was also applied in the context of generic two time-scale stochastic approximation [5].

However, finite-time analysis (non-asymptotic analysis) of two-time scale actor-critic is still largely missing in the literature, which is important because it can address the questions that how many samples are needed for two time-scale actor-critic to converge, and how to appropriately choose the different learning rates for the actor and the critic. Some recent work has attempted to provide the finite-time analysis for the 'decoupled' actor-critic methods [18, 23]. The term 'decoupled' means that before updating the actor at the t -th iteration, the critic starts from scratch to estimate the statevalue (or Q-value) function. At each iteration, the 'decoupled' setting requires the critic to perform multiple sampling and updating (often from another new sample trajectory). As we will see in the later comparison, this setting is sample-inefficient or even impractical. Besides, their analyses are based on either the i.i.d. assumption [18] or the partially i.i.d. assumption [23] (the actor receives i.i.d. samples), which is unrealistic in practice. In this paper, we present the first finite-time analysis on the convergence of the two time-scale actor-critic algorithm. We summarize our contributions as follows:

- We prove that, the actor in the two time-scale actor critic algorithm converges to an /epsilon1 -approximate stationary point of the non-concave performance function J after accessing at most ˜ O ( /epsilon1 -2 . 5 ) samples. Compared with existing finite-time analysis of actor-critic methods [18, 23], the algorithm we analyzed is based on two time-scale update and therefore more practical and efficient than the 'decoupled' version. Moreover, we do not need any i.i.d. data assumptions in the convergence analysis as required by Kumar et al. [18], Qiu et al. [23], which do not hold in real applications.
- From the technical viewpoint, we also present a new proof framework that can tightly characterize the estimation error in two time-scale algorithms. Compared with the proof technique used in [38], we remove the extra artificial factor O ( t ξ ) in the convergence rate introduced by their 'iterative refinement' technique. Therefore, our new proof technique may be of independent interest for analyzing the convergence of other two time-scale algorithms to get sharper rates.

Notation We use lower case letters to denote scalars, and use lower and upper case bold face letters to denote vectors and matrices respectively. For two sequences { a n } and { b n } , we write a n = O ( b n ) if there exists an absolute constant C such that a n ≤ Cb n . We use ˜ O ( · ) to further hide logarithm factors. Without other specification, ‖·‖ denotes the /lscript 2 norm of Euclidean vectors. d TV ( P, Q ) is the total variation norm between two probability measure P and Q , which is defined as d TV ( P, Q ) = 1 / 2 ∫ X | P ( dx ) -Q ( dx ) | .

## 2 Related work

In this section, we briefly review and discuss existing work, which is mostly related to ours.

Stochastic bias characterization The main difficulty in analyzing reinforcement learning algorithms under non-i.i.d. data assumptions is that the samples and the trainable parameters are correlated, which makes the noise term biased. Bhandari et al. [3] used information-theoretical techniques to bound the Markovian bias and provide a simple and explicit analysis for the temporal difference learning. Similar techniques were also established in [25] through the lens of stochastic approximation methods. Gupta et al. [12], Xu et al. [38] applied such methods to deriving the nonasymptotic convergence of two time-scale temporal difference learning algorithms (TDC). Zou et al. [44], Chen et al. [10], Xu and Gu [35] further applied these analysis methods to on-policy learning

algorithms including SARSA and Q-learning. In addition, Hu and Syed [13] formulated a family of TD learning algorithms as Markov jump linear systems and analyzed the evolution of the mean and covariance matrix of the estimation error. Cai et al. [7] studied TD learning with neural network approximation, and proved its global convergence.

Two time-scale reinforcement learning The two time-scale stochastic approximation can be seen as a general framework for analyzing reinforcement learning [5, 29, 17]. Recently, the finite-time analysis of two time-scale stochastic approximation has gained much interest. Dalal et al. [11] proved convergence rate for the two time-scale linear stochastic approximation under i.i.d. assumption. Gupta et al. [12] also provided finite-time analysis for the two time-scale linear stochastic approximation algorithms. Both can be applied to analyze two time-scale TD methods like GTD, GTD2 and TDC. Xu et al. [38] proved convergence rate and sample complexity for the TDC algorithm over Markovian samples. [15] further improved the convergence rate of two time-scale linear stochastic approximation and removed the projection step. However, since the update rule for the actor is generally not linear, we cannot apply these results to the actor-critic algorithms.

Analysis for actor-critic methods The asymptotic analysis of actor-critic methods has been well established. Konda and Tsitsiklis [16] proposed the actor-critic algorithm, and established the asymptotic convergence for the two time-scale actor-critic, with TD( λ ) learning-based critic. Bhatnagar et al. [4] proved the convergence result for the original actor-critic and natural actor-critic methods. Castro and Meir [8] proposed a single time-scale actor-critic algorithm and proved its convergence. Recently, [43] proved convergence of two time-scale off-policy actor-critic with function approximation. Recently, there has emerged some works concerning the finite-time behavior of actor-critic methods. Yang et al. [41] studied the global convergence of actor-critic algorithms under the Linear Quadratic Regulator. Yang et al. [40] analyzed the finite-sample performance of batched actor-critic, where all samples are assumed i.i.d. and the critic performs several empirical risk minimization (ERM) steps. Qiu et al. [23] treated the actor-critic algorithms as a bilevel optimization problem and established a finite sample analysis under the 'average-reward' setting, assuming that the actor has access to independent samples. Similar result has also been established by Kumar et al. [18], where they considered the sample complexity for the 'decoupled' actor-critic methods under i.i.d. assumption. Wang et al. [30] also proved the global convergence of actor-critic algorithms with both actor and critic being approximated by overparameterized neural networks.

When we were preparing this work, we noticed that there is a concurrent and independent work [39] which also analyzes the non-asymptotic convergence of two time-scale actor-critic algorithms and achieves the same sample complexity, i.e., ˜ O ( /epsilon1 -2 . 5 ) . However, there are two key differences between their work and ours. First, the two time-scale algorithms analyzed in both papers are very different. We analyze the classical two time-scale algorithm described in [27], where both actor and critic take one step update in each iteration. It is very easy to implement and has been widely used in practice, while the update rule in [39] for the critic needs to call a sub-algorithm, which involves generating a fresh episode to estimate the Q-function. Second, the analysis in [39] relies on the compatible function approximation [28], which requires the critic to be a specific linear function class, while our analysis does not require such specific approximation, and therefore is more general. This makes our analysis potentially extendable to non-linear function approximation such as neural networks [7].

## 3 Preliminaries

In this section, we present the background of the two time-scale actor-critic algorithm.

## 3.1 Markov decision processes

Reinforcement learning tasks can be modeled as a discrete-time Markov Decision Process (MDP) M = {S , A , P , r } , where S and A are the state and action spaces respectively. In this work we consider the finite action space |A| &lt; ∞ . P ( s ′ | s, a ) is the transition probability that the agent transits to state s ′ after taking action a at state s . Function r : S × A → [ -U r , U r ] emits a bounded reward after the agent takes action a at state s , where U r &gt; 0 is a constant. A policy parameterized by θ at state s is a probability function π θ ( a | s ) over action space A . µ θ denotes the stationary distribution induced by the policy π θ .

In this work we consider the 'average reward' setting [28], where under the ergodicity assumption, the average reward over time eventually converges to the expected reward under the stationary distribution:

<!-- formula-not-decoded -->

To evaluate the overall rewards given a starting state s 0 and the behavior policy π θ , we define the state-value function as

<!-- formula-not-decoded -->

where the action follows the policy a t ∼ π θ ( ·| s t ) and the next state follows the transition probability s t +1 ∼ P ( ·| s t , a t ) . Another frequently used function is the state-action value function, also called Q-value function:

<!-- formula-not-decoded -->

where the expectation is taken over s ′ ∼ P ( ·| s, a ) .

Throughout this paper, we use O to denote the tuple O = ( s, a, s ′ ) , some variants are like O t = ( s t , a t , s t +1 ) and ˜ O t = ( ˜ s t , ˜ a t , ˜ s t +1 ) . 3.2 Policy gradient theorem

Wedefine the performance function associated with policy π θ naturally as the expected reward under the stationary distribution µ θ induced by π θ , which takes the form

<!-- formula-not-decoded -->

To maximize the performance function with respect to the policy parameters, Sutton et al. [28] proved the following policy gradient theorem.

Lemma3.1 (Policy Gradient) . Consider the performance function defined in (3.1), its gradient takes the form

<!-- formula-not-decoded -->

The policy gradient also admits a neat form in expectation:

<!-- formula-not-decoded -->

A typical way to estimate the policy gradient ∇ J ( θ ) is by Monte Carlo method, namely using the summed return along the trajectory as the estimated Q-value, which is known as the 'REINFORCE' method [34].

Remark 3.2. The problem formulation in this paper is what Sutton et al. [28] had defined as 'average-reward' formulation. An alternative formulation is the 'start-state' formulation, which avoids estimating the average reward, but gives a more complicated form for the policy-gradient algorithm and the AC algorithm.

## 3.3 REINFORCE with a baseline

Note that for any function b ( s ) depending only on the state, which is usually called 'baseline' function, we have

<!-- formula-not-decoded -->

So we also have

<!-- formula-not-decoded -->

A popular choice of b ( s ) is b ( s ) = V π θ ( s ) and ∆ π θ ( s, a ) = Q π θ ( s, a ) -V π θ ( s ) is viewed as the advantage of taking a specific action a , compared with the expected reward at state s . Also note that the expectation form still holds:

<!-- formula-not-decoded -->

Based on this fact, Williams [34] also proposed a corresponding policy gradient algorithm named 'REINFORCE with a baseline' which performs better due to the reduced variance.

In practice the policy gradient method could suffer from high variance. An alternative approach is to introduce another trainable model to approximate the state-value function, which is called the actor-critic methods.

## 3.4 The two time-scale actor-critic algorithm

In previous subsection, we have seen how the policy gradient theorem appears in the form of the advantage value instead of the Q-value. Assume the critic uses linear function approximation ̂ V ( · ; ω ) = φ /latticetop ( · ) ω , and is updated by TD(0) algorithm, then this gives rise to Algorithm 1 that we are going to analyze.

Algorithm 1 has been proposed in many literature, and is clearly introduced in [27] as a classic on-line one-step actor-critic algorithm. It uses the advantage (namely temporal difference error) to update the critic and the actor simultaneously. Based on its on-line nature, this algorithm can be implemented both under episodic and continuing setting. In practice, the asynchronous variant of this algorithm, called Asynchronous Advantage Actor-Critic(A3C), is an empirically very successful parallel actor-critic algorithm.

Sometimes, Algorithm 1 is also called Advantage Actor-Critic (A2C) because it is the synchronous version of A3C and the name indicates its use of advantage instead of Q-value [20].

## Algorithm 1 Two Time-Scale Actor-Critic

- 1: Input: initial actor parameter θ 0 , initial critic parameter ω 0 , initial average reward estimator η 0 , step size α t for actor, β t for critic and γ t for the average reward estimator.
- 2: Draw s 0 from some initial distribution
- 3: for t = 0 , 1 , 2 , . . . do
- 4: Take the action a t ∼ π θ t ( ·| s t )
- 5: Observe next state s t +1 ∼ P ( ·| s t , a t ) and the reward r t = r ( s t , a t )
- 6: δ t = r t -η t + φ ( s t +1 ) /latticetop ω t -φ ( s t ) /latticetop ω t
- 7: η t +1 = η t + γ t ( r t -η t )
- 8: ω t +1 = Π R ω ( ω t + β t δ t φ ( s t ) ) 9: θ t +1 = θ t + α t δ t ∇ θ log π θ t ( a t | s t )
- 10: end for

In Line 6 of Algorithm 1, the temporal difference error δ t can be calculated based on the critic's estimation of the value function φ ( · ) /latticetop ω t , where ω t ∈ R d and φ ( · ) : S → R d is a known feature mapping. Then the critic will be updated using the semi-gradient from TD(0) method. Line 8 in Algorithm 1 also contains a projection operator. This is required to control the algorithm's convergence which also appears in some other literature [3, 38]. The actor uses the advantage δ t (estimated by critic) and the samples to get an estimation of the policy gradient.

Algorithm 1 is more general and practical than the algorithms analyzed in many previous work [23, 18]. In our algorithm, there is no need for independent samples or samples from the stationary distribution. There is only one naturally generated sample path. Also, the critic inherits from last iteration and continuously updates its parameter, without requiring a restarted sample path (or a new episode).

## 4 Main theory

In this section, we first discuss on some standard assumptions used in the literature for deriving the convergence of reinforcement learning algorithms and then present our theoretical results for two time-scale actor-critic methods.

## 4.1 Assumptions and propositions

We consider the setting where the critic uses TD [27] with linear function approximation to estimate the state-value function, namely ̂ V ( · ; ω ) = φ /latticetop ( · ) ω . We assume that the feature mapping has bounded norm ‖ φ ( · ) ‖ ≤ 1 . Denote by ω ∗ ( θ ) the limiting point of TD(0) algorithms under the behavior policy π θ , and define A and b as:

<!-- formula-not-decoded -->

where s ∼ µ θ ( · ) , a ∼ π θ ( ·| s ) , s ′ ∼ P ( ·| s, a ) . It is known that the TD limiting point satisfies:

<!-- formula-not-decoded -->

In the sequel, when there is no confusion, we will use a shorthand notation ω ∗ to denote ω ∗ ( θ ) . Based on the complexity of the feature mapping, the approximation error of this function class can vary. The approximation error of the linear function class is defined as follows:

<!-- formula-not-decoded -->

Throughout this paper, we assume the approximation error for all potential policies is uniformly bounded,

<!-- formula-not-decoded -->

for some constant /epsilon1 app ≥ 0 .

In the analysis of TD learning, the following assumption is often made to ensure the uniqueness of the limiting point of TD and the problem's solvability.

Assumption 4.1. For all potential policy parameters θ , the matrix A defined above is negative definite and has the maximum eigenvalues as -λ .

Assumption 4.1 is often made to guarantee the problem's solvability [3, 44, 38]. Note that Algorithm 1 contains a projection step at Line 8. To guarantee convergence it is required all ω ∗ lie within this projection radius R ω . Assumption 4.1 indicates that a sufficient condition is to set R ω = 2 U r /λ because ‖ b ‖ ≤ 2 U r and ‖ A -1 ‖ ≤ λ -1 .

The next assumption, first adopted by Bhandari et al. [3] in TD learning, addresses the issue of Markovian noise.

Assumption 4.2 (Uniform ergodicity) . For a fixed θ , denote µ θ ( · ) as the stationary distribution induced by the policy π θ ( ·| s ) and the transition probability measure P ( ·| s, a ) . Consider a Markov chain generated by the rule a t ∼ π θ ( ·| s t ) , s t +1 ∼ P ( ·| s t , a t ) . Then there exists m &gt; 0 and ρ ∈ (0 , 1) such that:

<!-- formula-not-decoded -->

We also need some regularity assumptions on the policy.

Assumption 4.3. Let π θ ( a | s ) be a policy parameterized by θ . There exist constants L, B, L l &gt; 0 such that for all given state s and action a it holds

- (a) ∥ ∥ ∇ log π θ ( a | s ) ∥ ∥ ≤ B , ∀ θ ∈ R d ,
- (c) ∣ ∣ π θ 1 ( a | s ) -π θ 2 ( a | s ) ∣ ∣ ≤ L ‖ θ 1 -θ 2 ‖ , ∀ θ 1 , θ 2 ∈ R d .
- (b) ∥ ∥ ∇ log π θ 1 ( a | s ) -∇ log π θ 2 ( a | s ) ∥ ∥ ≤ L l ‖ θ 1 -θ 2 ‖ , ∀ θ 1 , θ 2 ∈ R d ,

The first two inequalities are regularity conditions to guarantee actor's convergence in the literature of policy gradient [22, 42, 18, 36, 37]. The last inequality in Assumption 4.3 is also adopted by Zou et al. [44] when analyzing SARSA.

An important fact arises from our assumptions is that the limiting point ω ∗ of TD(0) , which can be viewed as a mapping of the policy's parameter θ , is Lipschitz.

Proposition 4.4. Under Assumptions 4.1 and 4.2, there exists a constant L ∗ &gt; 0 such that

Proposition 4.4 states that the target point ω ∗ moves slowly compared with the actor's update on θ . This is an observation pivotal to the two time-scale analysis. Specifically, the two time-scale analysis can be informally described as 'the actor moves slowly while the critic chases the slowly moving target determined by the actor'.

<!-- formula-not-decoded -->

Now we are ready to present the convergence result of two time-scale actor-critic methods. We first define an integer that depends on the learning rates α t and β t .

where m,ρ are defined as in Assumption 4.2. By definition, τ t is a mixing time of an ergodic Markov chain. We will use τ t to control the Markovian noise encountered in the training process.

<!-- formula-not-decoded -->

## 4.2 Convergence of the actor

At the k -th iteration of the actor's update, ω k is the critic parameter estimated by Line 7 of Algorithm 1 and ω ∗ k is the unknown parameter of value function V π θ k ( · ) defined in Assumption 4.1. The following theorem gives the convergence rate of the actor when the averaged mean squared error between ω k and ω ∗ k and the error between η k and r ( θ k ) from k = τ t to k = t are small.

Theorem 4.5. Suppose Assumptions 4.1-4.3 hold and we choose α t = c α / (1 + t ) σ in Algorithm 1, where σ ∈ (0 , 1) and c α &gt; 0 are constants. If we assume at the t -th iteration, the critic satisfies

<!-- formula-not-decoded -->

where E ( t ) is a bounded sequence, then we have where O ( · ) hides constants, whose exact forms can be found in the detailed proof in Appendix C.1.

<!-- formula-not-decoded -->

Note that E ( t ) in Theorem 4.5 is the averaged estimation error made by the critic throughout the learning process, which will be bounded in the next Theorem 4.7.

Remark 4.6. Theorem 4.5 recovers the results for the decoupled case [23, 18] by setting σ = 1 / 2 . Nevertheless, we are considering a much more practical and challenging case where the actor and critic are simultaneously updated under Markovian noises. It is worth noting that the non-i.i.d. data assumption leads to an additional logarithm term, which is also observed in [3, 44, 25, 10].

## 4.3 Convergence of the critic

The condition in (4.2) is guaranteed by the following theorem that characterizes the convergence of the critic.

Theorem 4.7. Suppose Assumptions 4.1-4.3 hold and we choose α t = c α / (1 + t ) σ and β t = c β / (1 + t ) ν in Algorithm 1, where 0 &lt; ν &lt; σ &lt; 1 , c α and c β ≤ λ -1 are positive constants. Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where O ( · ) hides constants, whose exact forms can be found in the detailed proof in Appendix C.2 and C.3.

Remark 4.8. The first term O ( t ν -1 ) on the right hand side of (4.3) and (4.4) comes from loosely bounding the error's norm, and can be removed by applying the 'iterative refinement' technique used in Xu et al. [38]. Using this technique, we can obtain a bound (also holds for η t ) E ‖ ω t -ω ∗ t ‖ 2 = O (log t/t ν ) + O (1 /t 2( σ -ν ) -ξ ) , where ξ &gt; 0 is an arbitrarily small constant. The constant ξ is an artifact due to the the 'iterative refinement' technique. Similar simplification can be done for (4.4). Nevertheless, if we plug (4.3) and (4.4) (after some transformation) into the result of Theorem 4.5, it is easy to see that the term O (1 /t 1 -ν ) is actually dominated by the term O (1 /t 1 -σ ) . Thus this term makes no difference in the total sample complexity of Algorithm 1 and we choose not to complicate the proof or introduce the extra artificial parameter ξ in the result of Theorem 4.7.

The second term in both (4.3) and (4.4) comes from the Markovian noise and the variance of the semi-gradient. The third term in these two equations comes from the slow drift of the actor. These two terms together can be interpreted as follows: if the actor moves much slower than the critic (i.e., σ -ν /greatermuch ν ), then the error is dominated by the Markovian noise and gradient variance; if the actor moves not too slowly compared with the critic (i.e. σ -ν /lessmuch ν ), then the critic's error is dominated by the slowly drifting effect of the actor.

## 4.4 Convergence rate and sample complexity

Combining Theorems 4.5 and 4.7 leads to the following convergence rate and sample complexity for Algorithm 1. The detailed proof is in Appendix C.4.

Corollary 4.9. Under the same assumptions of Theorems 4.5 and 4.7, we have

<!-- formula-not-decoded -->

If we set σ = 3 / 5 , ν = 2 / 5 , leading to the actor step size α t = O (1 /t 3 / 5 ) and the critic step size β t = O (1 /t 2 / 5 ) , Algorithm 1 can find an /epsilon1 -approximate stationary point of J ( · ) within T steps, namely,

<!-- formula-not-decoded -->

Corollary 4.9 combines the results of Theorems 4.5 and 4.7 and shows that the convergence rate of Algorithm 1 is ˜ O ( t -2 / 5 ) . Since the per iteration sample is 1 , the sample complexity of two time-scale actor-critic is O ( /epsilon1 -2 . 5 ) .

where T = ˜ O ( /epsilon1 -2 . 5 ) is the total iteration number.

˜ Remark 4.10. We compare our results with existing results on the sample complexity of actor-critic methods in the literature. Kumar et al. [18] provided a general result that after T = O ( /epsilon1 -2 ) updates for the actor, the algorithm can achieve min 0 ≤ k ≤ T E ‖∇ J ( θ k ) ‖ 2 ≤ /epsilon1 , as long as the estimation error of the critic can be bounded by O ( t -1 / 2 ) at the t -th actor's update. However, to ensure such a condition on the critic, they need to draw t samples to estimate the critic at the t -th actor's update. Therefore, the total number of samples drawn from the whole training process by the actor-critic algorithm in [18] is O ( T 2 ) , yielding a O ( /epsilon1 -4 ) sample complexity. Under the similar setting, Qiu et al. [23] proved the same sample complexity ˜ O ( /epsilon1 -4 ) when TD(0) is used for estimating the critic. Thus Corollary 4.9 suggests that the sample complexity of Algorithm 1 is significantly better than the sample complexity presented in [18, 23] by a factor of O ( /epsilon1 -1 . 5 ) .

Remark 4.11. The gap between the 'decoupled' actor-critic and the two time-scale actor-critic seems huge. Intuitively, this is due to the inefficient usage of the samples. At each iteration, the critic in the 'decoupled' algorithm starts over to evaluate the policy's value function and discards the history information, regardless of the fact that the policy might only changed slightly. The two time-scale actor-critic keeps the critic's parameter and thus takes full advantage of each samples in the trajectory.

Remark 4.12. According to [22], the sample complexity of policy gradient methods such as REINFORCE is O ( /epsilon1 -2 ) . As a comparison, if the critic converges faster than O ( t -1 / 2 ) , namely E ( t ) = O ( t -1 / 2 ) , then Theorem 4.5 combined with Corollary 4.9 implies that the complexity of two

time-scale actor-critic is ˜ O ( /epsilon1 -2 ) , which matches the result of policy gradient methods [22] up to logarithmic factors. Nevertheless, as we have discussed in the previous remarks, a smaller estimation error for critic often comes at the cost of more samples needed for the critic update [23, 18], which eventually increases the total sample complexity. Therefore, the ˜ O ( /epsilon1 -2 . 5 ) sample complexity in Corollary 4.9 is indeed the lowest we can achieve so far for classic two time-scale actor-critic methods. However, it is possible to further improve the sample complexity by using policy evaluation algorithms better than vanilla TD(0), such as GTD and TDC methods.

## 5 Conclusion and discussion

In this paper, we provided the first finite-time analysis of the two time-scale actor-critic methods, with non-i.i.d. Markovian samples and linear function approximation. The algorithm we analyzed is an on-line, one-step actor-critic algorithm which is practical and efficient. We proved its nonasymptotic convergence rate as well as its sample complexity. Our proof technique can be potentially extended to analyze other two time-scale reinforcement learning algorithms.

As one of the anonymous reviewers suggested, the compatible features are useful tools to address the function approximation error of the critic [16]. This can leads to finite-time analysis for the natural actor-critic algorithm [39], which also relates to the more general natural policy gradient methods [9]. Another possible improvement is to use regularization( e.g. ridge) for the critic to ensure the boundedness of the critic and remove the assumption on the maximum eigenvalue. The analysis can also be applied to the infinite-horizon discounted MDP, where the framework of analysis essentially remains the same.

## Broader impact

This work could positively impact the industrial application of actor-critic algorithms and other reinforcement learning algorithms. The theorem exhibits the sample complexity of actor-critic algorithms, which could be used to estimate required training time of reinforcement learning models. Another direct application of our result is to set the learning rate according to the finite-time bound, by optimizing the constant factors of the dominant terms. In this sense, the result could potentially reduce the overhead of hyper-parameter tuning, thus saving both human and computational resources. Moreover, the new analysis in this paper can potentially help people in different fields to understand the broader class of two-time scale algorithms, in addition to actor-critic methods. To our knowledge, this algorithm and theory studied in our paper do not have any ethical issues.

## Acknowledgement

We would like to thank the anonymous reviewers for their helpful comments. We also thank Xuyang Chen and Lin Zhao for pointing out a bug caused by the notation inconsistency in the proof of Theorem 4.5 and Lemma D.1 in the previous version. This research was sponsored in part by the National Science Foundation IIS-1904183 and Adobe Data Science Research Award. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

## References

- [1] Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. An actor-critic algorithm for sequence prediction. arXiv preprint arXiv:1607.07086 , 2016.
- [2] A. G. Barto, R. S. Sutton, and C. W. Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics , SMC-13(5):834-846, 1983.
- [3] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. arXiv preprint arXiv:1806.02450 , 2018.
- [4] Shalabh Bhatnagar, Richard S Sutton, Mohammad Ghavamzadeh, and Mark Lee. Natural actor-critic algorithms. Automatica , 45(11):2471-2482, 2009.

- [5] Vivek S Borkar. Stochastic approximation with two time scales. Systems &amp; Control Letters , 29(5):291-294, 1997.
- [6] Vivek S Borkar and Vijaymohan R Konda. The actor-critic algorithm as multi-time-scale stochastic approximation. Sadhana , 22(4):525-543, 1997.
- [7] Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. arXiv preprint arXiv:1912.05830 , 2019.
- [8] Dotan Di Castro and Ron Meir. A convergent online single time scale actor critic algorithm. Journal of Machine Learning Research , 11(Jan):367-410, 2010.
- [9] Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. arXiv preprint arXiv:2007.06558 , 2020.
- [10] Zaiwei Chen, Sheng Zhang, Thinh T Doan, Siva Theja Maguluri, and John-Paul Clarke. Performance of q-learning with linear function approximation: Stability and finite-time analysis. arXiv preprint arXiv: 1905.11425 , 2019.
- [11] Gal Dalal, Balazs Szorenyi, Gugan Thoppe, and Shie Mannor. Finite sample analysis of twotimescale stochastic approximation with applications to reinforcement learning. arXiv preprint arXiv:1703.05376 , 2017.
- [12] Harsh Gupta, R Srikant, and Lei Ying. Finite-time performance bounds and adaptive learning rate selection for two time-scale reinforcement learning. In Advances in Neural Information Processing Systems , pages 4706-4715, 2019.
- [13] Bin Hu and Usman Syed. Characterizing the exact behaviors of temporal difference learning algorithms using markov jump linear system theory. In Advances in Neural Information Processing Systems , pages 8477-8488, 2019.
- [14] Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873, 2018.
- [15] Maxim Kaledin, Eric Moulines, Alexey Naumov, Vladislav Tadic, and Hoi-To Wai. Finite time analysis of linear two-timescale stochastic approximation with markovian noise. arXiv preprint arXiv:2002.01268 , 2020.
- [16] Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems , pages 1008-1014, 2000.
- [17] Vijay R Konda, John N Tsitsiklis, et al. Convergence rate of linear two-time-scale stochastic approximation. The Annals of Applied Probability , 14(2):796-819, 2004.
- [18] Harshat Kumar, Alec Koppel, and Alejandro Ribeiro. On the sample complexity of actorcritic method for reinforcement learning with function approximation. arXiv preprint arXiv:1910.08412 , 2019.
- [19] A Yu Mitrophanov. Sensitivity and convergence of uniformly ergodic markov chains. Journal of Applied Probability , 42(4):1003-1014, 2005.
- [20] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937, 2016.
- [21] Yurii Nesterov. Lectures on convex optimization , volume 137. Springer, 2018.
- [22] Matteo Papini, Damiano Binaghi, Giuseppe Canonaco, Matteo Pirotta, and Marcello Restelli. Stochastic variance-reduced policy gradient. In International Conference on Machine Learning , pages 4023-4032, 2018.
- [23] Shuang Qiu, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. On the finite-time convergence of actor-critic algorithm. NeurIPS 2019 Optimization Foundations of Reinforcement Learning Workshop , 2019.

- [24] John Schulman, Sergey Levine, Pieter Abbeel, Michael I Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning , volume 37, pages 1889-1897, 2015.
- [25] R Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation andtd learning. In Conference on Learning Theory , pages 2803-2830, 2019.
- [26] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning , 3(1):9-44, 1988.
- [27] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.
- [28] Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems , pages 1057-1063, 2000.
- [29] Vladislav B Tadic and Sean P Meyn. Asymptotic properties of two time-scale stochastic approximation algorithms with constant step sizes. In Proceedings of the 2003 American Control Conference, 2003. , volume 5, pages 4426-4431. IEEE, 2003.
- [30] Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence. In International Conference on Learning Representations , 2020.
- [31] Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224 , 2016.
- [32] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning , 8(3-4):279-292, 1992.
- [33] Marco A Wiering. Convergence and divergence in standard and averaging reinforcement learning. In European Conference on Machine Learning , pages 477-488. Springer, 2004.
- [34] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8(3-4):229-256, 1992.
- [35] Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. arXiv preprint arXiv:1912.04511 , 2019.
- [36] Pan Xu, Felicia Gao, and Quanquan Gu. An improved convergence analysis of stochastic variance-reduced policy gradient. In International Conference on Uncertainty in Artificial Intelligence , 2019.
- [37] Pan Xu, Felicia Gao, and Quanquan Gu. Sample efficient policy gradient methods with recursive variance reduction. In International Conference on Learning Representations , 2020.
- [38] Tengyu Xu, Shaofeng Zou, and Yingbin Liang. Two time-scale off-policy td learning: Nonasymptotic analysis over markovian samples. In Advances in Neural Information Processing Systems , pages 10633-10643, 2019.
- [39] Tengyu Xu, Zhe Wang, and Yingbin Liang. Non-asymptotic convergence analysis of two timescale (natural) actor-critic algorithms. arXiv preprint arXiv:2005.03557 , 2020.
- [40] Zhuoran Yang, Kaiqing Zhang, Mingyi Hong, and Tamer Ba¸ sar. A finite sample analysis of the actor-critic algorithm. In 2018 IEEE Conference on Decision and Control (CDC) , pages 2759-2764. IEEE, 2018.
- [41] Zhuoran Yang, Yongxin Chen, Mingyi Hong, and Zhaoran Wang. On the global convergence of actor-critic: A case for linear quadratic regulator with ergodic cost. In Advances in Neural Information Processing Systems , 2019.
- [42] Kaiqing Zhang, Alec Koppel, Hao Zhu, and Tamer Ba¸ sar. Global convergence of policy gradient methods to (almost) locally optimal policies. arXiv preprint arXiv:1906.08383 , 2019.

- [43] Shangtong Zhang, Bo Liu, Hengshuai Yao, and Shimon Whiteson. Provably convergent twotimescale off-policy actor-critic with function approximation. arXiv , pages arXiv-1911, 2019.
- [44] Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finite-sample analysis for sarsa with linear function approximation. In Advances in Neural Information Processing Systems , pages 86658675, 2019.

## A Proof Sketch

In this section, we provide the proof roadmap of the main theory. Detailed proofs can be found in Appendix C.

## A.1 Proof Sketch of Theorem 4.5

The following lemma is important in that it enables the analysis of policy gradient method: Lemma A.1 ([42]) . For the performance function defined in (3.1), there exists a constant L J &gt; such that for all θ 1 , θ 2 ∈ R d , it holds that which by the definition of smoothness [21] is also equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This lemma enables us to perform a gradient ascent style analysis on the non-concave function J ( θ ) :

where O t = ( s t , a t , s t +1 ) is a tuple of observations. The second term ∆ h ( O t , η t , ω t , θ t ) on the right hand side of (A.1) is the bias introduced by the critic ω t and the reward estimate η t . The third term ∆ h ′ ( O t , θ t ) is from the linear approximation error. The fourth term Γ( O t , θ t ) is due to the Markovian noise. The last term can be viewed as the variance of the stochastic gradient update. Please refer to (C.1) for the definition of each notation.

<!-- formula-not-decoded -->

Now we bound each term's expectation in (A.1) respectively.

First, we have where z t := ω t -ω ∗ t and y t := η t -η ∗ t , and the inequality is due to Cauchy inequality and Lemma C.2.

<!-- formula-not-decoded -->

Second, taking expectation over the approximation error term containing ∆ h ′ , we have

<!-- formula-not-decoded -->

Third, we have

<!-- formula-not-decoded -->

where the first inequality is due to Lemma C.3, and the second inequality is due to ∥ ∥ δ t ∇ log π θ t ( a t | s t ) ∥ ∥ ≤ G θ by Lemma C.3. Taking the expectation of (C.3), plugging the above terms back into it and rearranging give

<!-- formula-not-decoded -->

0

Setting τ = τ t and summing over each term, and further dividing (1 + t -τ t ) at both sides and assuming t &gt; 2 τ t -1 , we can express the result as

<!-- formula-not-decoded -->

By Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, denote F ( t ) := 1 / (1 + t -τ t ) ∑ t k = τ t E ‖∇ J ( θ k ) ‖ 2 and Z ( t ) := 1 / (1 + t -τ t ) ∑ t k = τ t ( 8 E ‖ z t ‖ 2 +2 E [ y 2 t ] ) , and putting them back to (A.2) ( O -notation for simplicity):

which further gives

<!-- formula-not-decoded -->

Note that for a general function H ( t ) = A ( t ) + B ( t ) (with each positive), we have

This means

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Proof Sketch of Theorem 4.7

The proof of Theorem 4.7 can be divided into the following two parts.

## A.2.1 Estimating the Average Reward η k

We denote y k := η k -r ( θ k ) . First, we shall mention that many components in this step is uses the same framework and partial result as the proof regarding ω t in the next part. Also, part of the proof is intriguingly similar with the proof of Theorem 4.5. For simplicity, here we only present the final result regarding η k . Please refer to Section C.2 for the detailed proof. By setting γ k = (1 + t ) -ν , we have that

<!-- formula-not-decoded -->

## A.2.2 Approximating the TD Fixed Point

Step 1: decomposition of the estimation error. For simplicity, we denote z t := ω t -ω ∗ t , where the ω ∗ t denotes the exact parameter under policy π θ t . By the critic update in Line 7 of Algorithm 1, we have

<!-- formula-not-decoded -->

where O t := ( s t , a t , s t +1 ) is a tuple of observations, g ( O t , ω t ) and ¯ g ( θ t , ω t ) are the estimated gradient and the true gradient respectively. Λ( O t , ω t , θ t ) := 〈 ω t -ω ∗ t , g ( O t , ω t ) -¯ g ( θ t , ω t ) 〉 can be seen as the error induced by the Markovian noise. Please refer to (C.8) for formal definition of each notation.

The second term on the right hand side of (A.3) can be bounded by -2 λβ t ‖ z t ‖ 2 due to Assumption 4.1. The third term is a bias term caused by the Markovian noise. The fourth term ∆ g ( O t , η t , θ t ) is another bias term caused by inaccurate average reward estimator η t . The fifth term is caused by the slowly drifting policy parameter θ t . And the last term can be considered as the variance term.

Rewriting (A.3) and telescoping from τ = τ t to t , we have

<!-- formula-not-decoded -->

We will see that the Markovian noise I 2 , the 'slowly drifting policy" term I 3 and the estimation bias I 4 from η t are significant, and bounding the Markovian term is another challenge.

Step 2: bounding the Markovian bias. We first decompose Λ( θ t , ω t , O t ) as follows.

The motivation is to employ the uniform ergodicity defined by Assumption 4.2. This technique was first introduced by Bhandari et al. [3] to address the Markovian noise in policy evaluation. Zou et al. [44] extended to the Q-learning setting where the parameter itself both keeps updated and determines the behavior policy. In this work we take one step further to consider that the policy parameter θ t is changing, and the evaluation parameter ω t is updated. The analysis relies on the auxiliary Markov chain constructed by Zou et al. [44], which is obtained by repeatedly applying policy π θ t -τ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lipschitz conditions, we can bound the first two terms in (A.5). The third term will be bounded by the total variation between s k and ˜ s k , which is achieved by recursively bounding total variation between s k -1 and s k -1 .

˜ In fact, the Markovian noise Γ( O t , θ t ) in Section C.1 is obtained in a similar way. Due to the space limit, we only present how to bound the more complicated Λ( θ t , ω t , O t ) . We have the final form as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3: integrating the results. By some calculation, terms I 1 , I 2 and I 4 can be respectively bounded as follows (set τ = τ t defined in (4.1)). The detailed derivation can be found in Appendix C.3,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The log t comes from τ t = O (log t ) . Performing the same technique on I 3 as in Step 3 in the proof sketch of Theorem 4.5, we have

<!-- formula-not-decoded -->

After plugging each term into (A.4), we have that

<!-- formula-not-decoded -->

This inequality actually resembles (A.2). Following the same procedure as the proof of Theorem 4.5, starting from (A.2), we can finally get

<!-- formula-not-decoded -->

Note that this requires the step sizes γ t and β t should be of the same order O ( t -ν ) .

## B Preliminary Lemmas

These useful lemmas are frequently applied throughout the proof.

## B.1 Probabilistic Lemmas

The first two statements in the following lemma come from Zou et al. [44].

Lemma B.1. For any θ 1 and θ 2 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof of the first two inequality is exactly the same as Lemma A.3 in Zou et al. [44], which mainly depends on Theorem 3.1 in Mitrophanov [19]. Here we provide the proof of the third inequality. Note that

<!-- formula-not-decoded -->

so it has the same upper bound as the second inequality.

Lemma B.2. Given time indexes t and τ such that t ≥ τ &gt; 0 , consider the auxiliary Markov chain starting from s t -τ . Conditioning on s t -τ +1 and θ t -τ , the Markov chain is obtained by repeatedly applying policy π θ t -τ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Throughout this lemma, we always condition the expectation on s t -τ +1 and θ t -τ and omit this in order to simplify the presentation. Under the setting introduced above, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of (B.2) . By the Law of Total Probability,

<!-- formula-not-decoded -->

and a similar argument also holds for ˜ O t . Then we have

The last equality requires exchange of integral, which should be guaranteed by the regularity.

<!-- formula-not-decoded -->

Proof of (B.3) .

<!-- formula-not-decoded -->

Proof of (B.4) . Because θ t is also dependent on s t , we make it clear here that

Therefore, the total variance can be bounded as

<!-- formula-not-decoded -->

where the inequality holds due to the Lipschitz continuity of the policy as in Assumption 4.3.

<!-- formula-not-decoded -->

## B.2 Lipschitzness of the Optimal Parameter

This section is used to present the proof of Proposition 4.4.

Proof of Proposition 4.4. Sutton and Barto [27] has proved in Chapter 9 the fact that the linear TD(0) will converge to the optimal point (w.r.t. Mean Square Projected Bellman Error) which satisfies

<!-- formula-not-decoded -->

where A i := E [ φ ( s )( φ ( s ) -φ ( s ′ )) /latticetop ] and b i := E [( r ( s, a ) -r ( θ i )) φ ( s )] . The expectation is taken over the stationary distribution s ∼ µ θ i , the action a ∼ π θ i ( ·| s ) and the transition probability kernel s ′ ∼ P ( ·| s, a ) .

Now we denote ω ∗ 1 , ω ∗ 2 , ̂ ω 1 as the unique solutions of the following equations respectively: A 1 ω ∗ 1 = b 1 ,

First we bound ‖ ω ∗ 1 -̂ ω 1 ‖ . By definition, we have

It can be easily shown that

<!-- formula-not-decoded -->

which further gives

<!-- formula-not-decoded -->

Then we bound ‖ ̂ ω 1 -ω ∗ 2 ‖ ,

<!-- formula-not-decoded -->

‖ ̂ ω 1 -ω ∗ 2 ‖ ≤ ‖ A -1 2 ‖‖ b 1 -b 2 ‖ . By Assumption 4.1, the eigenvalues of A i are bounded from below by λ &gt; 0 , therefore ‖ A -1 i ‖ ≤ λ -1 . Also ‖ b 1 ‖ ≤ U r due to the assumption that | r ( s, a ) | ≤ U r and ‖ φ ( s ) ‖ ≤ 1 . To bound ‖ A 1 -A 2 ‖ and ‖ b 1 -b 2 ‖ , we first note that where O i is the tuple obtained by s i ∼ µ θ i ( · ) , a i ∼ π θ i ( ·| s i ) and ( s ′ ) i ∼ P ( ·| s i , a i ) . And the total variation norm can be bounded by Lemma B.1 as:

Collecting the results above gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we set L ∗ := (2 λ -2 U r + 3 λ -1 U r ) |A| L (1 + /ceilingleft log ρ m -1 /ceilingright + 1 / (1 -ρ )) to obtain the final result.

## B.3 Asymptotic Equivalence

Lemma B.3. Suppose { a i } is a non-negative, bounded sequence, τ := C 1 + C 2 log t ( C 2 &gt; 0) , then for any large enough t such that t ≥ τ &gt; 0 , we have:

<!-- formula-not-decoded -->

Proof. We know that τ = O (log t ) and the sequence is bounded: 0 &lt; a i &lt; B . For the first equation, we have

<!-- formula-not-decoded -->

and further assuming t ≥ 2 τ -2 gives a constant 2 . For the second equation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Proof of Main Theorems and Propositions

## C.1 Proof of Theorem 4.5

We first define several notations to clarify the dependence:

<!-- formula-not-decoded -->

In the following proof, we also denote η ∗ t = r ( θ t ) . When the context is clear, ω ∗ denotes ω ∗ ( θ ) . Note that ∆ h , ∆ h ′ and h -∆ h ′ together give a decomposition of the actor update ( ∆ h + h ) we use in Algorithm 1. They respectively correspond to the error caused by the critic ω t and η t , the approximation error of the linear class, and the stochastic policy gradient.

Γ( O, θ ) is the Markovian noise for h ( O, θ ) . Here O ′ = ( s, a, s ′ ) is a shorthand for an independent sample from s ∼ µ θ , a ∼ π θ , s ′ ∼ P . Using a more compact notation E O ′ [ · ] , it is clear we have

<!-- formula-not-decoded -->

and E O ′ ‖ ∆ h ′ ( O, θ ) ‖ 2 ≤ 4 B 2 /epsilon1 2 app because

<!-- formula-not-decoded -->

There are several lemmas that will be used in the proof.

Lemma C.1. For the performance function defined in (3.1), there exists a constant L J &gt; 0 such that for all θ 1 , θ 2 ∈ R d , it holds that which by the definition of smoothness [21] implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following two lemmas characterize the bias introduced by the critic's approximation and the Markovian noise.

Lemma C.2. For any t ≥ 0 ,

Lemma C.3. For any θ ∈ R d , we have ‖ δ ∇ log π θ ( a | s ) ‖ ≤ G θ := U δ · B , where U δ = 2 U r +2 R ω . Furthermore, for any t ≥ 0 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D 1 := max { 2 L J + 3 L h , 2 U δ B |A| L } and D 2 = 4 U δ B . Here L h = U δ L l + (2 + 2 λ -2 + 3 λ -1 ) BU r |A| L ( 1 + /ceilingleft log ρ m -1 /ceilingright +1 / (1 -ρ ) ) .

Proof of Theorem 4.5. Under the update rule of Algorithm 1, we have

The first inequality is by Lemma C.1 (we discard the 1 / 2 in front of the square-norm term). The first equality is by the definitions in (C.1); the second equality is by the definition of Γ( O t , θ t ) in (C.1). The last equality is due to (C.2). Here O ′ = ( s, a, s ′ ) is a shorthand for an independent sample from s ∼ µ θ t , a ∼ π θ t , s ′ ∼ P ( ·| s, a ) .

<!-- formula-not-decoded -->

We will bound the expectation of each term on the right hand side of (C.3) as follows. First, we have where z t := ω t -ω ∗ t and y t := η t -η ∗ t , and the inequality is due to Cauchy inequality and Lemma C.2.

<!-- formula-not-decoded -->

Second, we have

<!-- formula-not-decoded -->

where the first inequality is due to Lemma C.3, and the second inequality is due to ∥ ∥ δ t ∇ log π θ t ( a t | s t ) ∥ ∥ ≤ G θ by Lemma C.3. Third, by the remarks under (C.1) regarding ∆ h ′ , we have

<!-- formula-not-decoded -->

Taking the expectation of (C.3) and plugging the above terms back into it gives

<!-- formula-not-decoded -->

Rearranging the above inequality gives

<!-- formula-not-decoded -->

By setting τ = τ t , we get

<!-- formula-not-decoded -->

Summing over k from τ t to t gives

<!-- formula-not-decoded -->

For the term I 1 , we have,

<!-- formula-not-decoded -->

where the inequality holds due to | E [ J ( θ )] | ≤ U r . For the term I 2 , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Note that both upper bounds rely on the summation ∑ t -τ t k =0 1 / (1+ k ) σ ≤ ∫ t -τ t +1 0 x -σ dx = 1 / (1 -σ )( t -τ t +1) 1 -σ . Combining the results for terms I 1 and I 2 , we have

<!-- formula-not-decoded -->

Dividing (1 + t -τ t ) at both sides and assuming t &gt; 2 τ t -1 , we can express the result as

<!-- formula-not-decoded -->

By Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, denote F ( t ) := 1 / (1 + t -τ t ) ∑ t k = τ t E ‖∇ J ( θ k ) ‖ 2 and Z ( t ) := 1 / (1 + t -τ t ) ∑ t k = τ t ( 8 E ‖ z t ‖ 2 +2 E [ y 2 t ] ) , and putting them back to (C.4) ( O -notation for simplicity):

which further gives

<!-- formula-not-decoded -->

Note that for a general function H ( t ) ≤ A ( t ) + B ( t ) (with each positive), we have

This means (C.5) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma B.3, assuming t ≥ 2 τ t -1 , it holds that

<!-- formula-not-decoded -->

And finally, we have

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 4.7: Estimating the Average Reward

We define several notations to clarify the probabilistic dependency.

<!-- formula-not-decoded -->

We also write J ( θ t ) = r ( θ t ) sometimes in the proof.

Lemma C.4. For any θ 1 , θ 2 , we have

∣ ∣ where C J = 2 U r |A| L (1 + /ceilingleft log ρ m -1 /ceilingright +1 / (1 -ρ )) .

<!-- formula-not-decoded -->

Lemma C.5. Given the definition of Ξ( O t , η t , θ t ) , for any t &gt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. From the definition, η t is the average reward estimator, η ∗ t = J ( θ t ) = E [ r ( s, a )] is the average reward under the stationary distribution µ θ t ⊗ π θ t , and y t = η t -η ∗ t . From the algorithm we have the update rule as where we leave the step size γ t unspecified for now. Unrolling the recursive definition we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rearranging and summing from τ t to t , we have

<!-- formula-not-decoded -->

For I 1 , following the Abel summation formula, we have

<!-- formula-not-decoded -->

For I 2 , from Lemma C.5, we have

<!-- formula-not-decoded -->

By the choice of τ t , we have

<!-- formula-not-decoded -->

For I 3 , we have

For I 5 , we have

<!-- formula-not-decoded -->

which is because by Lemma C.4, ( η ∗ k -η ∗ k +1 ) can be linearly bounded by ‖ θ k -θ k +1 ‖ ≤ G θ · α k . For I 4 , by the same argument it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by bounding the expectation uniformly.

Now, we set γ k = 1 / (1 + t ) ν and combine all the terms together to get

<!-- formula-not-decoded -->

By applying the squaring technique already stated in the proof of Theorem 4.5, we have that

<!-- formula-not-decoded -->

## C.3 Proof of Theorem 4.7: Approximating the TD Fixed Point

Now we deal with the critic's parameter ω t . The two time-scale analysis with Markovian noise and moving behavior policy can be complicated, so we define some useful notations here that could hopefully clarify the probabilistic dependency. Note that J ( θ ) := r ( θ ) is the average reward under π θ and r ( s, a ) is the one-step reward specified by the state s and action a .

<!-- formula-not-decoded -->

A bounded lemma is used frequently in this section.

Lemma C.6. Under Assumption 4.3, for any θ , ω , O = ( s, a, s ′ ) such that ‖ ω ‖ ≤ R ω ,

<!-- formula-not-decoded -->

The following lemma is used to control the bias due to Markovian noise.

Lemma C.7. Given the definition of Λ( θ t , ω t , O t ) , for any 0 ≤ τ ≤ t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem 4.7. By the updating rule of ω t in Algorithm 1, unrolling and decomposing the squared error gives

<!-- formula-not-decoded -->

where the first inequality holds because ω ∗ t +1 is assumed to be within the R ω -ball so the projection only reduces the distance; the second one is due to ‖ x + y ‖ 2 ≤ 2 ‖ x ‖ 2 +2 ‖ y ‖ 2 and the third one is due to ‖ g ( O t , ω t , θ t ) + ∆ g ( O t , η t , θ t ) ‖ ≤ U δ .

First, note that due to Assumption 4.1, we have

<!-- formula-not-decoded -->

where the first equation is due to the fact that ¯ g ( ω ∗ , θ ) = 0 [27]. Taking expectation up to s t +1 , we have

<!-- formula-not-decoded -->

Based on the result above, we can further rewrite it as:

<!-- formula-not-decoded -->

where we denote the constant coefficient before the quadratic stepsize β 2 t as C q at the last step. The first inequality is due to Proposition 4.4 and Cauchy-Schwartz inequality. The second inequality is due to the update of θ t is bounded by G θ α t . The third inequality is from employing the fact that σ &gt; ν so α t /β t is bounded. Rearranging the inequality yields

<!-- formula-not-decoded -->

where the second inequality is due to the concavity of square root function. Telescoping from τ t to t gives:

<!-- formula-not-decoded -->

From (C.9), we can see the proof of the critic again shares the same spirit with the proof of Theorem 4.5. For term I 1 , we have

<!-- formula-not-decoded -->

where the first inequality is due to discarding the last term, and the second inequality is due to E ‖ z k ‖ 2 ≤ ( R ω + R ω ) 2 .

For term I 2 , note that due to Lemma C.7, we actually have

<!-- formula-not-decoded -->

and the summation is

<!-- formula-not-decoded -->

For term I 3 and I 4 , we will instead show it can be bounded in a different form. Using CauchySchwartz inequality we have where the second inequality is due to the monotonicity of α k and β k . The O ( · ) comes from that τ = O (log t ) and ∑ k -ν = O ( t 1 -ν ) .

<!-- formula-not-decoded -->

Collecting the upper bounds of the above five terms, and writing them using O ( · ) notation give

<!-- formula-not-decoded -->

Now, we first divide both sides by (1 + t -τ t ) , and denote

<!-- formula-not-decoded -->

and the rest as A ( t ) = O ( t ν ) + O ( t 1 -ν ) . G ( t ) 's constants appear at (C.7) in exact form. This simplification leads to

<!-- formula-not-decoded -->

which further gives

<!-- formula-not-decoded -->

This is again a similar reasoning as in the end of the proof of Theorem 4.5. We actually show that

<!-- formula-not-decoded -->

This completes the proof. To obtain the exact constant, please refer to (C.7) and (C.10).

## C.4 Proof of Corollary 4.9

Proof of Corollary 4.9. By Theorem 4.7, we have

<!-- formula-not-decoded -->

By Lemma B.3, E ( t ) in Theorem 4.5 is of the equivalent order:

<!-- formula-not-decoded -->

The same reasoning also applies to

<!-- formula-not-decoded -->

Plugging the above results into Theorem 4.5, and optimizing over the choice of σ and ν (which gives σ = 3 / 5 and ν = 2 / 5 ), we have

<!-- formula-not-decoded -->

Therefore, in order to obtain an /epsilon1 -approximate(ignoring the approximation error) stationary point of J , namely, we need to set T = O ( /epsilon1 -2 . 5 ) .

<!-- formula-not-decoded -->

## D Proof of Technical Lemmas

## D.1 Proof of Lemma C.1

Proof of Lemma C.1. The first inequality comes from Lemma 3.2 in Zhang et al. [42].

The second inequality is well known as a partial result of [ -L, L ] -smoothness of non-convex functions.

## D.2 Proof of Lemma C.2

Proof of Lemma C.2. Applying the definition of ∆ h () and Cauchy-Schwartz inequality immediately yields the result.

## D.3 Proof of Lemma C.3

The proof of Lemma C.3 will be built on the following supporting lemmas.

Lemma D.1. For any t ≥ 0 ,

<!-- formula-not-decoded -->

Lemma D.2. For any t ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.3. For any t ≥ 0 ,

<!-- formula-not-decoded -->

Proof of Lemma C.3. First note that

<!-- formula-not-decoded -->

which immediately implies

<!-- formula-not-decoded -->

where the last inequality is due to Assumption 4.3. We decompose the Markovian bias as

<!-- formula-not-decoded -->

where ˜ O t is from the auxiliary Markovian chain and O ′ t is from the stationary distribution which actually satisfy E [Γ( O ′ t , θ t -τ )] = 0 . By collecting the corresponding bounds from Lemmas D.1,

D.2 and D.3, we have that

<!-- formula-not-decoded -->

where D 1 := max { 2 L J +3 L h , 2 U δ B |A| L } and D 2 := 4 U δ B , which completes the proof.

## D.4 Proof of Lemma C.4

Proof of Lemma C.4. By definition, we have

<!-- formula-not-decoded -->

where s ( i ) ∼ µ θ i , a ( i ) ∼ π θ i . Therefore, it holds that

<!-- formula-not-decoded -->

## D.5 Proof of Lemma C.5

The proof of this lemma depends on several auxiliary lemmas as follows.

Lemma D.4. For any θ 1 , θ 2 , eta, O = ( s, a, s ′ ) , we have

Lemma D.5. For any θ , η 1 , η 2 , O , we have

<!-- formula-not-decoded -->

Lemma D.6. Consider original tuples O t = ( s t , a t , s t +1 ) and the auxiliary tuples ˜ O t = ( ˜ s t , ˜ a t , ˜ s t +1 ) . Conditioned on s t -τ +1 and θ t -τ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.7. Conditioned on s t -τ +1 and θ t -τ , we have

<!-- formula-not-decoded -->

Proof. By the Lemma D.4, D.5, D.6 and D.7, we can collect the corresponding term and get the bound

<!-- formula-not-decoded -->

## D.6 Proof of Lemma C.6

Proof of Lemma C.6. For the first inequality, apply the property of norm and the Cauchy-Schwartz inequality:

<!-- formula-not-decoded -->

For the second inequality, we can directly apply Cauchy-Schwartz inequality and obtain the result. For the third inequality, apply Cauchy-Schwartz inequality as we have

<!-- formula-not-decoded -->

which completes the proof.

## D.7 Proof of Lemma C.7

This Lemma is actually a combination of several auxiliary lemmas listed here:

Lemma D.8. For any θ 1 , θ 2 , ω and tuple O = ( s, a, s ′ ) ,

∣ ∣ where K 1 = 2 U 2 δ |A| L (1 + /ceilingleft log ρ m -1 /ceilingright +1 / (1 -ρ )) + 2 U δ L ∗ .

<!-- formula-not-decoded -->

Lemma D.9. For any θ , ω 1 , ω 2 and tuple O = ( s, a, s ′ ) ,

<!-- formula-not-decoded -->

∣ ∣ Λ( O, ω 1 , θ ) -Λ( O, ω 2 , θ ) ∣ ∣ ≤ 6 U δ ‖ ω 1 -ω 2 ‖ . Lemma D.10. Consider original tuples O t = ( s t , a t , s t +1 ) and the auxiliary tuples ˜ O t = ( ˜ s t , ˜ a t , ˜ s t +1 ) . Conditioned on s t -τ +1 and θ t -τ , we have

Lemma D.11. Conditioned on s t -τ +1 and θ t -τ ,

<!-- formula-not-decoded -->

Proof of Lemma C.7. By the Lemma D.8, D.9, D.10 and D.11, we can collect the corresponding term and get the bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E Proof of Auxiliary Lemmas

## E.1 Proof of Lemma D.1

Proof of Lemma D.1. Denote δ ( O t , θ ) := r ( s t , a t ) -r ( θ ) + ( φ ( s t +1 ) -φ ( s t )) /latticetop ω ∗ and we have h ( O t , θ ) = δ ( O t , θ ) ∇ log π θ ( a t | s t ) . It can be shown that δ ( O t , θ 1 ) -δ ( O t , θ 2 ) = ( φ ( s t +1 ) -φ ( s t )) /latticetop ( ω ∗ 1 -ω ∗ 2 ) -( r ( θ 1 ) -r ( θ 2 )) .

Denote O t = ( s t , a t , s t +1 ) , we have for any θ 1 and θ 2 , that

<!-- formula-not-decoded -->

where we use shorthand E θ to denote that O ′ = ( s, a, s ′ ) is drawn from s ∼ µ θ , a ∼ π θ , s ′ ∼ P ( ·| s, a ) . We first exhibit each term here is Lipschitz. We have by Lemma C.1 that,

‖∇ J ( θ 1 ) -∇ J ( θ 2 ) ‖ ≤ L J ‖ θ 1 -θ 2 ‖ . For h ( O t , θ 1 ) and h ( O t , θ 2 ) , we have ∥ ∥ h ( O t , θ 1 ) -h ( O t , θ 2 ) ∥ ∥ = ∥ ∥ δ ( O t , θ 1 ) ∇ log π θ 1 ( a t | s t ) -δ ( O t , θ 2 ) ∇ log π θ 2 ( a t | s t ) ∥ ∥ ≤ ∥ ∥ δ ( O t , θ 1 ) ∇ log π θ 1 ( a t | s t ) -δ ( O t , θ 1 ) ∇ log π θ 2 ( a t | s t ) ∥ ∥ ︸ ︷︷ ︸ I 1 + ∥ ∥ δ ( O t , θ 1 ) ∇ log π θ 2 ( a t | s t ) -δ ( O t , θ 2 ) ∇ log π θ 2 ( a t | s t ) ∥ ∥ ︸ ︷︷ ︸ I 2 ≤ U δ L l ‖ θ 1 -θ 2 ‖ + ∥ ∥ δ ( O t , θ 1 ) ∇ log π θ 2 ( a t | s t ) -δ ( O t , θ 2 ) ∇ log π θ 2 ( a t | s t ) ∥ ∥ ︸ ︷︷ ︸ I 2 , where the first inequality is due to the triangle inequality. The term I 1 is easily bounded by the fact that δ is bounded (see Section D.3) and Assumption 4.3. For I 2 , we have

<!-- formula-not-decoded -->

where the first inequality is due to Assumption 4.3, and the second is by unrolling the definition of δ ( O t , θ ) and invoking the triangle inequality, among them, we know φ is within the unit ball and ω ∗ is L ∗ -Lipschitz by Proposition 4.4 with L ∗ := (2 λ -2 U r +3 λ -1 U r ) |A| L (1 + /ceilingleft log ρ m -1 /ceilingright +1 / (1 -ρ )) .

For | r ( θ 1 ) -r ( θ 2 ) | , we have that

<!-- formula-not-decoded -->

where the first inequality is by the definition of the total-variation distance, and the second inequality is from Lemma B.1. To summarize, we have

<!-- formula-not-decoded -->

where L h denotes the coefficient above.

Similarly, for E [ h ( O ′ , θ )] and E [ h ( O ′ , θ )] , we have first

<!-- formula-not-decoded -->

where the first inequality is due to the triangle inequality; the second one is due to the convexity of ‖ · ‖ norm; the third inequality is from the Lipschitz-ness of h ( O, θ ) we just showed above; the fourth one is due to the property of the total variation distance; the fifth one is due to Proposition 4.4. The last inequality is just to absorb the coefficient into L h for less notation clutter.

So far, we have proved the Lipschitz-ness of all the terms in Γ( O, θ 1 ) -Γ( O, θ 2 ) . We can also show that each term is bounded: from (D.1) in Section D.3, we can see that ∇ J ( θ ) is G θ -bounded and also h ( O, θ ) -E θ [ h ( O ′ , θ )] is 2 U δ B -bounded since h ( O, θ ) is bounded by U δ B for any O and θ .

To sum up, ∇ J ( θ ) is G θ -bounded and L J -Lipschitz; h ( O, θ ) -E θ [ h ( O ′ , θ )] is 3 L h -Lipschitz and 2 U δ B -bounded. By the triangle inequality, we have

<!-- formula-not-decoded -->

This completes the proof.

## E.2 Proof of Lemma D.2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality is by the definition of total variation. By Lemma B.2 we have

<!-- formula-not-decoded -->

Plugging (E.2) into (E.1) we get

<!-- formula-not-decoded -->

## E.3 Proof of Lemma D.3

Proof of Lemma D.3.

<!-- formula-not-decoded -->

The first inequality is by the definition of total variation norm and the second inequality holds because, by the ergodicity in Assumption 4.2, it holds that and thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The equations above are derived following the same procedure in (B.1), because P ( ˜ O t = ·| s t -τ +1 , θ t -τ ) = P ( ˜ s t = ·| s t -τ +1 , θ t -τ ) ⊗ π θ t -τ ⊗P .

## E.4 Proof of Lemma D.4

Proof of Lemma D.4. By the definition of Ξ( O,η, θ ) in (C.6), we have

<!-- formula-not-decoded -->

## E.5 Proof of Lemma D.5

Proof of Lemma D.5. By definition,

<!-- formula-not-decoded -->

## E.6 Proof of Lemma D.6

Proof of Lemma D.6. By the Cauchy-Schwartz inequality and the definition of total variation norm, we have

Since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this bound, we have

<!-- formula-not-decoded -->

## E.7 Proof of Lemma D.7

Proof of Lemma D.7. We first note that according to the definition,

<!-- formula-not-decoded -->

where O ′ t = ( s ′ t , a ′ t , s ′ t +1 ) is the tuple generated by s ′ t ∼ µ θ t -τ , a ′ t ∼ π θ t -τ , s ′ t +1 ∼ P . By the ergodicity in Assumption 4.2, it holds that

It can be shown that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The argument used here also appears in the proof of Lemma D.11 and explained in detail there.

## E.8 Proof of Lemma D.8

Proof of Lemma D.8.

<!-- formula-not-decoded -->

2

For the term I 2 , we simply use the Cauchy-Schwartz inequality to get 2 U δ ‖ ω ∗ 1 -ω ∗ 2 ‖ . For the term I 1 , it can be bounded as:

<!-- formula-not-decoded -->

where the first inequality is due to Cauchy-Schwartz; the second inequality is by the definition of total variation norm; the third inequality is due to the fact U δ ≥ 2 R ω . Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality is due to Lemma B.1 and Proposition 4.4.

## E.9 Proof of Lemma D.9

Proof of Lemma D.9. By definition,

∥ ∥ Note that we have ‖ g ( O, ω 1 , θ ) -g ( O, ω 2 , θ ) ‖ = | ( φ ( s ′ ) -φ ( s )) /latticetop ( ω 1 -ω 2 ) | ≤ 2 ‖ ω 1 -ω 2 ‖ and similarly ‖ ¯ g ( ω 1 , θ ) -¯ g ( ω 2 , θ ) ‖ ≤ | E [ ( φ ( s ′ ) -φ ( s )) /latticetop ( ω 1 -ω 2 ) ] | ≤ 2 ‖ ω 1 -ω 2 ‖ . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.10 Proof of Lemma D.10

Proof of Lemma D.10. By the Cauchy-Schwartz inequality and the definition of total variation norm, we have

<!-- formula-not-decoded -->

Plugging this bound into (E.3), we have

<!-- formula-not-decoded -->

## E.11 Proof of Lemma D.11

Proof of Lemma D.11. We first note that according to the definition in Section C.3,

<!-- formula-not-decoded -->

where O ′ t = ( s ′ t , a ′ t , s ′ t +1 ) is the tuple generated by s ′ t ∼ µ θ t -τ , a ′ t ∼ π θ t -τ , s ′ t +1 ∼ P . By the ergodicity in Assumption 4.2, it holds that

It can be shown that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The third inequality holds because 2 R ω &lt; U δ and

<!-- formula-not-decoded -->

˜ This can be shown following the same procedure in (B.1), because P ( ˜ O t = ·| s t -τ +1 , θ t -τ ) = P ( ˜ s t = ·| s t -τ +1 , θ t -τ ) ⊗ π θ t -τ ⊗P .