## Neural Temporal-Difference and Q-Learning Provably Converge to Global Optima

Qi Cai ∗ Zhuoran Yang † Jason D. Lee ‡ Zhaoran Wang ∗

## Abstract

Temporal-difference learning (TD), coupled with neural networks, is among the most fundamental building blocks of deep reinforcement learning. However, due to the nonlinearity in value function approximation, such a coupling leads to nonconvexity and even divergence in optimization. As a result, the global convergence of neural TDremains unclear. In this paper, we prove for the first time that neural TD converges at a sublinear rate to the global optimum of the mean-squared projected Bellman error for policy evaluation. In particular, we show how such global convergence is enabled by the overparametrization of neural networks, which also plays a vital role in the empirical success of neural TD. Beyond policy evaluation, we establish the global convergence of neural (soft) Q-learning, which is further connected to that of policy gradient algorithms.

## 1 Introduction

Given a policy, temporal-different learning (TD) (Sutton, 1988) aims to learn the corresponding (action-)value function by following the semigradients of the mean-squared Bellman error in an online manner. As the most-used policy evaluation algorithm, TD serves as the 'critic' component of many reinforcement learning algorithms, such as the actor-critic algorithm (Konda and Tsitsiklis, 2000) and trust-region policy optimization (Schulman et al., 2015). In particular, in deep reinforcement learning, TD is often applied to learn value functions parametrized by neural networks

∗ Department of Industrial Engineering and Management Sciences, Northwestern University

† Department of Operations Research and Financial Engineering, Princeton University

‡ Department of Electrical Engineering, Princeton University

(Lillicrap et al., 2015; Mnih et al., 2016; Haarnoja et al., 2018), which gives rise to neural TD. As policy improvement relies crucially on policy evaluation, the optimization efficiency and statistical accuracy of neural TD are critical to the performance of deep reinforcement learning. Towards theoretically understanding deep reinforcement learning, the goal of this paper is to characterize the convergence of neural TD.

Despite the broad applications of neural TD, its convergence remains rarely understood. Even with linear value function approximation, the nonasymptotic convergence of TD remains open until recently (Bhandari et al., 2018; Lakshminarayanan and Szepesvari, 2018; Dalal et al., 2018; Srikant and Ying, 2019), although its asymptotic convergence is well understood (Jaakkola et al., 1994; Tsitsiklis and Van Roy, 1997; Borkar and Meyn, 2000; Kushner and Yin, 2003; Borkar, 2009). Meanwhile, with nonlinear value function approximation, TD is known to diverge in general (Baird, 1995; Boyan and Moore, 1995; Tsitsiklis and Van Roy, 1997; Chung et al., 2019; Achiam et al., 2019). To remedy such an issue, Bhatnagar et al. (2009) propose nonlinear (gradient) TD, which uses the tangent vectors of nonlinear value functions in place of the feature vectors in linear TD. Unlike linear TD, which converges to the global optimum of the mean-squared projected Bellman error (MSPBE), nonlinear TD is only guaranteed to converge to a local optimum asymptotically. As a result, the statistical accuracy of the value function learned by nonlinear TD remains unclear. In contrast to such conservative theory, neural TD, which straightforwardly combines TD with neural networks without the explicit local linearization in nonlinear TD, often learns a desired value function that generalizes well to unseen states in practice (Duan et al., 2016; Amiranashvili et al., 2018; Henderson et al., 2018). Hence, a gap separates theory from practice.

There exist three obstacles towards closing such a theory-practice gap: (i) MSPBE has an expectation with respect to the transition dynamics within the squared loss, which forbids the construction of unbiased stochastic gradients (Sutton and Barto, 2018). As a result, even with linear value function approximation, TD largely eludes the classical optimization framework, as it follows biased stochastic semigradients. (ii) When the value function is parametrized by a neural network, MSPBE is nonconvex in the weights of the neural network, which may introduce undesired stationary points such as local optima and saddle points (Jain and Kar, 2017). As a result, even an ideal algorithm that follows the population gradients of MSPBE may get trapped. (iii) Due to the interplay between the bias in stochastic semigradients and the nonlinearity in value function approximation, neural TD may even diverge (Baird, 1995; Boyan and Moore, 1995; Tsitsiklis and Van Roy, 1997), instead of converging to an undesired stationary point, as it lacks the explicit local linearization in nonlinear TD (Bhatnagar et al., 2009). Such divergence is also not captured by the classical

optimization framework.

Contribution: Towards bridging theory and practice, we establish the first nonasymptotic global rate of convergence of neural TD. In detail, we prove that randomly initialized neural TD converges to the global optimum of MSPBE at the rate of 1 /T with population semigradients and at the rate of 1 / √ T with stochastic semigradients. Here T is the number of iterations and the (action)value function is parametrized by a sufficiently wide multi-layer neural network. Moreover, we prove that the projection in MSPBE allows for a sufficiently rich class of functions, which has the same representation power of a reproducing kernel Hilbert space associated with the random initialization. As a result, for a broad class of reinforcement learning problems, neural TD attains zero MSPBE. Beyond policy evaluation, we further establish the global convergence of neural (soft) Q-learning, which allows for policy improvement. In particular, we prove that, under stronger regularity conditions, neural (soft) Q-learning converges at the same rate of neural TD to the global optimum of MSPBE for policy optimization. Also, by exploiting the connection between (soft) Qlearning and policy gradient algorithms (Schulman et al., 2017; Haarnoja et al., 2018), we establish the global convergence of a variant of the policy gradient algorithm (Williams, 1992; Szepesv´ ari, 2010; Sutton and Barto, 2018).

At the core of our analysis is the overparametrization of the multi-layer neural network for value function approximation, which enables us to circumvent the three obstacles above. In particular, overparametrization leads to an implicit local linearization that varies smoothly along the solution path, which mirrors the explicit one in nonlinear TD (Bhatnagar et al., 2009). Such an implicit local linearization enables us to circumvent the third obstacle of possible divergence. Moreover, overparametrization allows us to establish a notion of one-point monotonicity (Harker and Pang, 1990; Facchinei and Pang, 2007) for the semigradients followed by neural TD, which ensures its evolution towards the global optimum of MSPBE along the solution path. Such a notion of monotonicity enables us to circumvent the first and second obstacles of bias and nonconvexity. Broadly speaking, our theory backs the empirical success of overparametrized neural networks in deep reinforcement learning. In particular, we show that instead of being a curse, overparametrization is indeed a blessing for minimizing MSPBE in the presence of bias, nonconvexity, and even divergence.

More Related Work: There is a large body of literature on the convergence of linear TD under both asymptotic (Jaakkola et al., 1994; Tsitsiklis and Van Roy, 1997; Borkar and Meyn, 2000; Kushner and Yin, 2003; Borkar, 2009) and nonasymptotic (Bhandari et al., 2018; Lakshminarayanan and Szepesv 2018; Dalal et al., 2018; Srikant and Ying, 2019) regimes. See Dann et al. (2014) for a detailed

survey. In particular, our analysis is based on the recent breakthrough in the nonasymptotic analysis of linear TD (Bhandari et al., 2018) and its extension to linear Q-learning (Zou et al., 2019). An essential step of our analysis is bridging the evolution of linear TD and neural TD through the implicit local linearization induced by overparametrization. See also the concurrent work of Brandfonbrener and Bruna (2019a,b); Agazzi and Lu (2019) on neural TD, which however requires the state space to be finite.

To incorporate nonlinear value function approximation into TD, Bhatnagar et al. (2009) propose the first convergent nonlinear TD based on explicit local linearization, which however only converges to a local optimum of MSPBE. See Geist and Pietquin (2013); Bertsekas (2019) for a detailed survey. In contrast, we prove that, with the implicit local linearization induced by overparametrization, neural TD, which is simpler to implement and more widely used in deep reinforcement learning than nonlinear TD, provably converges to the global optimum of MSPBE.

There exist various extensions of TD, including least-squares TD (Bradtke and Barto, 1996; Boyan, 1999; Lazaric et al., 2010; Ghavamzadeh et al., 2010; Tu and Recht, 2017) and gradient TD (Sutton et al., 2009a,b; Bhatnagar et al., 2009; Liu et al., 2015; Du et al., 2017; Wang et al., 2017; Touati et al., 2017). In detail, least-squares TD is based on batch update, which loses the computational and statistical efficiency of the online update in TD. Meanwhile, gradient TD follows unbiased stochastic gradients, but at the cost of introducing another optimization variable. Such a reformulation leads to bilevel optimization, which is less stable in practice when combined with neural networks (Pfau and Vinyals, 2016). As a result, both extensions of TD are less widely used in deep reinforcement learning (Duan et al., 2016; Amiranashvili et al., 2018; Henderson et al., 2018). Moreover, when using neural networks for value function approximation, the convergence to the global optimum of MSPBE remains unclear for both extensions of TD.

Our work is also related to the recent breakthrough in understanding overparametrized neural networks, especially their generalization error (Zhang et al., 2016; Neyshabur et al., 2018; Li and Liang, 2018; Allen-Zhu et al., 2018a,b,c; Zou et al., 2018; Arora et al., 2019; Cao and Gu, 2019a,b). See Fan et al. (2019) for a detailed survey. In particular, Daniely (2017); Chizat and Bach (2018); Jacot et al. (2018); Li and Liang (2018); Allen-Zhu et al. (2018a,b,c); Zou et al. (2018); Arora et al. (2019); Cao and Gu (2019a,b); Lee et al. (2019) characterize the implicit local linearization in the context of supervised learning, where we train an overparametrized neural network by following the stochastic gradients of the mean-squared error. In contrast, neural TD does not follow the stochastic gradients of any objective function, hence leading to possible divergence, which makes the convergence analysis more challenging.

## 2 Background

In Section 2.1, we briefly review policy evaluation in reinforcement learning. In Section 2.2, we introduce the corresponding optimization formulations.

## 2.1 Policy Evaluation

We consider a Markov decision process ( S , A , P , r, γ ) , in which an agent interacts with the environment to learn the optimal policy that maximizes the expected total reward. At the t -th time step, the agent has a state s t ∈ S and takes an action a t ∈ A . Upon taking the action, the agent enters the next state s t +1 ∈ S according to the transition probability P ( · | s t , a t ) and receives a random reward r t = r ( s t , a t ) from the environment. The action that the agent takes at each state is decided by a policy π : S → ∆ , where ∆ is the set of all probability distributions over A . The performance of policy π is measured by the expected total reward, J ( π ) = E [ ∑ ∞ t =0 γ t r t | a t ∼ π ( s t )] , where γ &lt; 1 is the discount factor.

Given policy π , policy evaluation aims to learn the following two functions, the value function V π ( s ) = E [ ∑ ∞ t =0 γ t r t | s 0 = s, a t ∼ π ( s t )] and the action-value function (Q-function) Q π ( s, a ) = E [ ∑ ∞ t =0 γ t r t | s 0 = s, a 0 = a, a t ∼ π ( s t )] . Both functions form the basis for policy improvement. Without loss of generality, we focus on learning the Q-function in this paper. We define the Bellman evaluation operator,

<!-- formula-not-decoded -->

for which Q π is the fixed point, that is, the solution to the Bellman equation Q = T π Q .

## 2.2 Optimization Formulation

Corresponding to (2.1), we aim to learn Q π by minimizing the mean-squared Bellman error (MSBE),

<!-- formula-not-decoded -->

where the Q-function is parametrized by ̂ Q θ with parameter θ . Here µ is the stationary distribution of ( s, a ) corresponding to policy π . Due to Q-function approximation, we focus on minimizing the following surrogate of MSBE, namely the projected mean-squared Bellman error (MSPBE),

<!-- formula-not-decoded -->

Here Π F is the projection onto a function class F . For example, for linear Q-function approximation (Sutton, 1988), F takes the form { ̂ Q θ ′ : θ ′ ∈ Θ } , where ̂ Q θ ′ is linear in θ ′ and Θ is the set of feasible parameters. As another example, for nonlinear Q-function approximation (Bhatnagar et al., 2009), F takes the form { ̂ Q θ + ∇ θ ̂ Q /latticetop θ ( θ ′ -θ ) : θ ′ ∈ Θ } , which consists of the local linearization of Q θ ′ at θ .

̂ Throughout Sections 3-6, we assume that we are able to sample tuples in the form of ( s, a, r, s ′ , a ′ ) from the stationary distribution of policy π in an independent and identically distributed manner. Our analysis is extended to handle temporal dependence in Appendix G using the proof techniques of Bhandari et al. (2018). With a slight abuse of notation, we use µ to denote the stationary distribution of ( s, a, r, s ′ , a ′ ) corresponding to policy π and any of its marginal distributions.

## 3 Neural Temporal-Difference Learning

TD updates the parameter θ of the Q-function by taking the stochastic semigradient descent step (Sutton, 1988; Szepesv´ ari, 2010; Sutton and Barto, 2018),

<!-- formula-not-decoded -->

which corresponds to the MSBE in (2.2). Here ( s, a, r, s ′ , a ′ ) ∼ µ and η &gt; 0 is the stepsize. In a more general context, (3.1) is referred to as TD(0). In this paper, we focus on TD(0), which is abbreviated as TD, and leave the extension to TD( λ ) to future work.

In the sequel, we consider S to be continuous and A to be finite. We represent the state-action pair ( s, a ) ∈ S × A by a vector x = ψ ( s, a ) ∈ X ⊆ R d with d &gt; 2 , where ψ is a given one-to-one feature map. With a slight abuse of notation, we use ( s, a ) and x interchangeably. Without loss of generality, we assume that ‖ x ‖ 2 = 1 and | r ( x ) | is upper bounded by a constant r &gt; 0 for any x ∈ X . We use a two-layer neural network

<!-- formula-not-decoded -->

to parametrize the Q-function, which is extended to a multi-layer neural network in Appendix F. Here σ is the rectified linear unit (ReLU) activation function σ ( y ) = max { 0 , y } and the parameter θ = ( b 1 , . . . , b m , W 1 , . . . , W m ) are initialized as b r ∼ Unif ( {-1 , 1 } ) and W r ∼ N (0 , I d /d ) for any r ∈ [ m ] independently. During training, we only update W = ( W 1 , . . . , W m ) ∈ R md , while keeping b = ( b 1 , . . . , b m ) ∈ R m fixed as the random initialization. To ensure global convergence,

we incorporate an additional projection step with respect to W . See Algorithm 1 for a detailed description.

## Algorithm 1 Neural TD

- 1: Initialization: b r ∼ Unif ( {-1 , 1 } ) , W r (0) ∼ N (0 , I d /d ) ( r ∈ [ m ]) , W = W (0) ,

Initialization:

- 2: For t = 0 to T -2 :

<!-- formula-not-decoded -->

- 3: Sample a tuple ( s, a, r, s ′ , a ′ ) from the stationary distribution µ of policy π
- 4: Let x = ( s, a ) , x ′ = ( s ′ , a ′ )
- 5: Bellman residual calculation: δ ← ̂ Q ( x ; W ( t )) -r -γ ̂ Q ( x ′ ; W ( t )) 6: TD update: ˜ W ( t +1) ← W ( t ) -ηδ · ∇ W ̂ Q ( x ; W ( t )) 7: Projection: W ( t +1) ← argmin W ∈ S B ‖ W -˜ W ( t +1) ‖ 2 8: Averaging: W ← t +1 t +2 · W + 1 t +2 · W ( t +1)
- 9: End For
- 10: Output: ̂ Q out ( · ) ← ̂ Q ( · ; W )

To understand the intuition behind the global convergence of neural TD, note that for the TD update in (3.1), we have from (2.1) that

<!-- formula-not-decoded -->

Here (i) is the Bellman residual at ( s, a ) , while (ii) is the gradient of the first term in (i). Although the TD update in (3.1) resembles the stochastic gradient descent step for minimizing a mean-squared error, it is not an unbiased stochastic gradient of any objective function. However, we show that the TD update yields a descent direction towards the global optimum of the MSPBE in (2.3). Moreover, as the neural network becomes wider, the function class F that Π F projects onto in (2.3) becomes richer. Correspondingly, the MSPBE reduces to the MSBE in (2.2) as the projection becomes closer to identity, which implies the recovery of the desired Q-function Q π such that Q π = T π Q π . See Section 4 for a more rigorous characterization.

## 4 Main Results

In Section 4.1, we characterize the global optimality of the stationary point attained by Algorithm 1 in terms of minimizing the MSPBE in (2.3) and its other properties. In Section 4.2, we establish the nonasymptotic global rates of convergence of neural TD to the global optimum of the MSPBE when following the population semigradients in (3.3) and the stochastic semigradients in (3.1), respectively. Throughout Section 4, we focus on two-layer neural networks. In Appendix F, we present the extension to multi-layer neural networks.

We use the subscript E µ [ · ] to denote the expectation with respect to the randomness of the tuple ( s, a, r, s, a ′ ) (or its concise form ( x, r, x ′ ) ) conditional on all other randomness, e.g., the random initialization and the random current iterate. Meanwhile, we use the subscript E init ,µ [ · ] when we are taking expectation with respect to all randomness, including the random initialization.

## 4.1 Properties of Stationary Point

We consider the population version of the TD update in Line 6 of Algorithm 1,

<!-- formula-not-decoded -->

where µ is the stationary distribution and δ ( x, r, x ′ ; W ( t )) = ̂ Q ( x ; W ( t )) -r -γ ̂ Q ( x ′ ; W ( t )) is the Bellman residual at ( x, r, x ′ ) . The stationary point W † of (4.1) satisfies the following stationarity condition,

<!-- formula-not-decoded -->

Also, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and ∇ W r ̂ Q ( x ; W ) = b r 1 { W /latticetop r x &gt; 0 } x almost everywhere in R md . Meanwhile, recall that S B = { W ∈ R md : ‖ W -W (0) ‖ 2 ≤ B } . We define the function class which consists of the local linearization of ̂ Q ( x ; W ) at W = W † . Then (4.2) takes the following equivalent form

<!-- formula-not-decoded -->

which implies ̂ Q ( · ; W † ) = Π F † B,m T π ̂ Q ( · ; W † ) by the definition of the projection induced by 〈· , ·〉 µ . By (2.3), ̂ Q ( · ; W † ) is the global optimum of the MSPBE that corresponds to the projection onto F † B,m .

Intuitively, when using an overparametrized neural network with width m → ∞ , the average variation in each W r diminishes to zero. Hence, roughly speaking, we have 1 { W r ( t ) /latticetop x &gt; 0 } = 1 { W r (0) /latticetop x &gt; 0 } with high probability for any t ∈ [ T ] . As a result, the function class F † B,m defined in (4.3) approximates

<!-- formula-not-decoded -->

In the sequel, we show that, to characterize the global convergence of Algorithm 1 with a sufficiently large m , it suffices to consider F B,m in place of F † B,m , which simplifies the analysis, since the distribution of W (0) is given. To this end, we define the approximate stationary point W ∗ with respect to the function class F B,m defined in (4.5).

Definition 4.1 (Approximate Stationary Point W ∗ ) . If W ∗ = ( W ∗ 1 , . . . , W ∗ m ) ∈ R md satisfies

<!-- formula-not-decoded -->

where we define

<!-- formula-not-decoded -->

then we say that W ∗ is an approximate stationary point of the population update in (4.1). Here W ∗ depends on the random initialization b = ( b 1 , . . . , b m ) and W (0) = ( W 1 (0) , . . . , W m (0)) .

<!-- formula-not-decoded -->

The next lemma proves that such an approximate stationary point uniquely exists, since it is the fixed point of the operator Π F B,m T π , which is a contraction in the /lscript 2 -norm associated with the stationary distribution µ .

Lemma 4.2 (Existence, Uniqueness, and Optimality of ̂ Q 0 ( · ; W ∗ ) ) . There exists an approximate stationary point W ∗ for any b ∈ R m and W (0) ∈ R md . Also, ̂ Q 0 ( · ; W ∗ ) is unique almost everywhere and is the global optimum of the MSPBE that corresponds to the projection onto F B,m in (4.5).

Proof. See Appendix B.1 for a detailed proof.

## 4.2 Global Convergence

In this section, we establish the main results on the global convergence of neural TD in Algorithm 1. We first lay out the following regularity condition on the stationary distribution µ .

Assumption 4.3 (Regularity of Stationary Distribution µ ) . There exists a constant c 0 &gt; 0 such that for any τ ≥ 0 and w ∈ R d with ‖ w ‖ 2 = 1 , it holds that where x ∼ µ .

Assumption 4.3 regularizes the density of µ in terms of the marginal distribution of x . In particular, it is straightforwardly implied when the marginal distribution of x has a uniformly upper bounded probability density over the unit sphere.

Population Update: The next theorem establishes the nonasymptotic global rate of convergence of neural TD when it follows population semigradients. Recall that the approximate stationary point W ∗ and the corresponding ̂ Q 0 ( · ; W ∗ ) are defined in Definition 4.1. Also, B is the radius of the set of feasible W , which is defined in Algorithm 1, T is the number of iterations, γ is the discount factor, and m is the width of the neural network in (3.2).

Theorem 4.4 (Convergence of Population Update) . We set η = (1 -γ ) / 8 in Algorithm 1 and replace the TD update in Line 6 by the population update in (4.1). Under Assumption 4.3, the output ̂ Q out of Algorithm 1 satisfies where the expectation is taken with respect to all randomness, including the random initialization and the stationary distribution µ .

<!-- formula-not-decoded -->

Proof. The key to the proof of Theorem 4.4 is the one-point monotonicity of the population semigradient g ( t ) , which is established through the local linearization ̂ Q 0 ( x ; W ) of ̂ Q ( x ; W ) . See Appendix C.5 for a detailed proof.

Stochastic Update: To further prove the global convergence of neural TD when it follows stochastic semigradients, we first establish an upper bound of their variance, which affects the choice of the stepsize η . For notational simplicity, we define the stochastic and population semigradients as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 4.5 (Variance Bound) . There exists σ 2 g = O ( B 2 ) such that the variance of the stochastic semigradient is upper bounded as E init ,µ [ ‖ g ( t ) -g ( t ) ‖ 2 2 ] ≤ σ 2 g for any t ∈ [ T ] .

Proof. See Appendix B.2 for a detailed proof.

Based on Theorem 4.4 and Lemma 4.5, we establish the global convergence of neural TD in Algorithm 1.

Theorem 4.6 (Convergence of Stochastic Update) . We set η = min { (1 -γ ) / 8 , 1 / √ T } in Algorithm 1. Under Assumption 4.3, the output ̂ Q out of Algorithm 1 satisfies

Proof. See Appendix C.6 for a detailed proof.

<!-- formula-not-decoded -->

As the width of the neural network m →∞ , Lemma 4.2 implies that ̂ Q 0 ( · ; W ∗ ) is the global optimum of the MSPBE in (2.3) with a richer function class F B, ∞ to project onto. In fact, the function class F B, ∞ -̂ Q ( · ; W (0)) is a subset of an RKHS with H -norm upper bounded by B . Here ̂ Q ( · ; W (0)) is defined in (3.2). See Appendix A.2 for a more detailed discussion on the representation power of F B, ∞ . Therefore, if the desired Q-function Q π ( · ) falls into F B, ∞ , it is the global optimum of the MSPBE. By Lemma 4.2 and Theorem 4.6, we approximately obtain Q π ( · ) = Q 0 ( · ; W ∗ ) through Q out ( · ) .

Proposition 4.7 (Convergence of Stochastic Update to Q π ) . It holds that ‖ ̂ Q 0 ( · ; W ∗ ) -Q π ( · ) ‖ µ ≤ (1 -γ ) -1 · ‖ Π F B,m Q π ( · ) -Q π ( · ) ‖ µ , which by Theorem 4.6 implies

̂ ̂ More generally, the following proposition quantifies the distance between ̂ Q 0 ( · ; W ∗ ) and Q π ( · ) in the case that Q π ( · ) does not fall into the function class F B,m . In particular, it states that the /lscript 2 -norm distance ‖ ̂ Q 0 ( · ; W ∗ ) -Q π ( · ) ‖ µ is upper bounded by the distance between Q π ( · ) and F B,m .

<!-- formula-not-decoded -->

Proof. See Appendix B.3 for a detailed proof.

Proposition 4.7 implies that if Q π ( · ) ∈ F B, ∞ , then ̂ Q out ( · ) → Q π ( · ) as T, m → ∞ . In other words, neural TD converges to the global optimum of the MSPBE in (2.3), or equivalently, the MSBE in (2.2), both of which have objective value zero.

## 5 Proof Sketch

In the sequel, we sketch the proofs of Theorems 4.4 and 4.6 in Section 4.

## 5.1 Implicit Local Linearization via Overparametrization

Recall that as defined in (4.7), ̂ Q 0 ( x ; W ) takes the form

<!-- formula-not-decoded -->

which is linear in the feature map Φ( x ) . In other words, with respect to W , ̂ Q 0 ( x ; W ) linearizes the neural network ̂ Q ( x ; W ) defined in (3.2) locally at W (0) . The following lemma characterizes the difference between ̂ Q ( x ; W ( t )) , which is along the solution path of neural TD in Algorithm 1, and its local linearization ̂ Q 0 ( x ; W ( t )) . In particular, we show that the error of such a local linearization diminishes to zero as m → ∞ . For notational simplicity, we use ̂ Q t ( x ) to denote ̂ Q ( x ; W ( t )) in the sequel. Note that by (4.7) we have ̂ Q 0 ( x ) = ̂ Q ( x ; W (0)) = ̂ Q 0 ( x ; W (0)) . Recall that B is the radius of the set of feasible W in (4.5).

Lemma 5.1 (Local Linearization of Q-Function) . There exists a constant c 1 &gt; 0 such that for any t ∈ [ T ] , it holds that

Proof. See Appendix C.1 for a detailed proof.

<!-- formula-not-decoded -->

As a direct consequence of Lemma 5.1, the next lemma characterizes the effect of local linearization on population semigradients. Recall that g ( t ) is defined in (4.9). We denote by g 0 ( t ) the locally linearized population semigradient, which is defined by replacing ̂ Q t ( x ) in g ( t ) with its local linearization ̂ Q 0 ( x ; W ( t )) . In other words, by (4.9), (4.7), and (4.8), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 5.2 (Local Linearization of Semigradient) . Let r be the upper bound of the reward r ( x ) for any x ∈ X . There exists a constant c 2 &gt; 0 such that for any t ∈ [ T ] , it holds that

<!-- formula-not-decoded -->

Proof. See Appendix C.2 for a detailed proof.

Lemmas 5.1 and 5.2 show that the error of local linearization diminishes as the degree of overparametrization increases along m . As a result, we do not require the explicit local linearization in nonlinear TD (Bhatnagar et al., 2009). Instead, we show that such an implicit local linearization suffices to ensure the global convergence of neural TD.

## 5.2 Proofs for Population Update

The characterization of the locally linearized Q-function in Lemma 5.1 and the locally linearized population semigradients in Lemma 5.2 allows us to establish the following descent lemma, which extends Lemma 3 of Bhandari et al. (2018) for characterizing linear TD.

Lemma 5.3 (Population Descent Lemma) . For { W ( t ) } t ∈ [ T ] in Algorithm 1 with the TD update in Line 6 replaced by the population update in (4.1), it holds that

<!-- formula-not-decoded -->

Proof. See Appendix C.3 for a detailed proof.

Lemma 5.3 shows that, with a sufficiently small stepsize η , ‖ W ( t ) -W ∗ ‖ 2 decays at each iteration up to the error of local linearization, which is characterized by Lemma 5.2. By combining Lemmas 5.2 and 5.3 and further plugging them into a telescoping sum, we establish the convergence of ̂ Q out ( · ) to the global optimum ̂ Q 0 ( · ; W ∗ ) of the MSPBE. See Appendix C.5 for a detailed proof.

## 5.3 Proofs for Stochastic Update

Recall that the stochastic semigradient g ( t ) is defined in (4.9). In parallel with Lemma 5.3, the following lemma additionally characterizes the effect of the variance of g ( t ) , which is induced by the randomness of the current tuple ( x, r, x ′ ) . We use the subscript E W [ · ] to denote the expectation with respect to the randomness of the current iterate W ( t ) conditional on the random initialization b and W (0) . Correspondingly, E W,µ [ · ] is with respect to the randomness of both the current tuple ( x, r, x ′ ) and the current iterate W ( t ) conditional on the random initialization.

Lemma 5.4 (Stochastic Descent Lemma) . For { W ( t ) } t ∈ [ T ] in Algorithm 1, it holds that

Proof. See Appendix C.4 for a detailed proof.

<!-- formula-not-decoded -->

To ensure the global convergence of neural TD in the presence of the variance of g ( t ) , we rescale the stepsize to be of order T -1 / 2 . The rest proof of Theorem 4.6 mirrors that of Theorem 4.4. See Appendix C.6 for a detailed proof.

## 6 Extension to Policy Optimization

With the Q-function learned by TD, policy iteration may be applied to learn the optimal policy. Alternatively, Q-learning more directly learns the optimal policy and its Q-function using temporaldifference update. Compared with TD, Q-learning aims to solve the projected Bellman optimality equation

<!-- formula-not-decoded -->

which replaces the Bellman evaluation operator T π in (2.3) with the Bellman optimality operator T . When Π F is identity, the fixed-point solution to (6.1) is the Q-function Q π ∗ ( s, a ) of the optimal policy π ∗ , which maximizes the expected total reward (Szepesv´ ari, 2010; Sutton and Barto, 2018). Compared with TD, the max operator in T makes the analysis more challenging and hence requires stronger regularity conditions. In the following, we first introduce neural Q-learning and then establish its global convergence. Finally, we discuss the corresponding implication for policy gradient algorithms. Throughout Section 6, we focus on two-layer neural networks. Our analysis can be extended to handle multi-layer neural networks using the proof techniques in Appendix F.

## 6.1 Neural Q-Learning

In parallel with (3.1), we update the parameter θ of the optimal Q-function by

<!-- formula-not-decoded -->

where the tuple ( s, a, r, s ′ ) is sampled from the stationary distribution µ exp of an exploration policy π exp in an independent and identically distributed manner. Our analysis can be extended to handle temporal dependence using the proof techniques in Appendix G. We present the detailed neural Qlearning algorithm in Algorithm 2. Similar to Definition 4.1, we define the approximate stationary point W ∗ of Algorithm 2 by

<!-- formula-not-decoded -->

where the Bellman residual is now δ 0 ( x, r, x ′ ; W ) = ̂ Q 0 ( x ; W ) -r -γ max a ′ ∈A ̂ Q 0 ( s ′ , a ′ ; W ) . Following the same analysis of neural TD in Lemma 4.2, we have that ̂ Q 0 ( · ; W ∗ ) is the unique fixed-point solution to the projected Bellman optimality equation Q = Π F B,m T Q , where the function class F B,m is define in (4.5).

## Algorithm 2 Neural Q-Learning

- 1: Initialization: b r ∼ Unif ( {-1 , 1 } ) , W r (0) ∼ N (0 , I d /d ) ( r ∈ [ m ]) , W = W (0) , Initialization: S B = { W ∈ R md : ‖ W -W (0) ‖ 2 ≤ B } ( B &gt; 0) , Initialization: exploration policy π exp such that π exp ( a | s ) &gt; 0 for any ( s, a ) ∈ S × A 2: For t = 0 to T -2 : 3: Sample a tuple ( s, a, r, s ′ ) from the stationary distribution µ exp of the exploration policy π exp 4: Let x = ( s, a ) , x ′ = ( s ′ , argmax a ′ ∈A ̂ Q ( s ′ , a ′ ; W ( t ))) 5: Bellman residual calculation: δ ← ̂ Q ( x ; W ( t )) -r -γ ̂ Q ( x ′ ; W ( t )) 6: TD update: ˜ W ( t +1) ← W ( t ) -ηδ · ∇ W ̂ Q ( x ; W ( t )) 7: Projection: W ( t +1) ← argmin W ∈ S B ‖ W -˜ W ( t +1) ‖ 2 8: Averaging: W ← t +1 t +2 · W + 1 t +2 · W ( t +1) 9: End For 10: Output: ̂ Q out ( · ) ← ̂ Q ( · ; W )

## 6.2 Global Convergence

To establish the global convergence of neural Q-learning, we lay out an extra regularity condition on the exploration policy π exp, which is not required by neural TD. Such a regularity condition ensures that x ′ = ( s ′ , a ′ ) with the greedy action a ′ in Line 4 of Algorithm 2 follows a similar distribution to that of x = ( s, a ) , which is the stationary distribution µ exp of the exploration policy π exp . Recall that ̂ Q 0 ( x ; W ) is defined in (4.7) and γ is the discount factor.

Assumption 6.1 (Regularity of Exploration Policy π exp ) . There exists a constant ν &gt; 0 such that for any W 1 , W 2 ∈ S B , it holds that

̂ 0 ̂

<!-- formula-not-decoded -->

We remark that Melo et al. (2008); Zou et al. (2019) establish the global convergence of linear Q-learning based on an assumption that implies (6.4). Although Assumption 6.1 is strong, we are not aware of any weaker regularity condition in the literature, even for linear Q-learning. As our focus is to go beyond linear Q-learning to analyze neural Q-learning, we do not attempt to weaken such a regularity condition in this paper.

The following regularity condition on µ exp mirrors Assumption 4.3, but additionally accounts for the max operator in the Bellman optimality operator.

Assumption 6.2 (Regularity of Stationary Distribution µ exp ) . There exists a constant c 3 &gt; 0 such that for any τ ≥ 0 and w ∈ R d with ‖ w ‖ 2 = 1 , it holds that where ( s, a ) ∼ µ exp .

In parallel with Theorem 4.6, the following theorem establishes the global convergence of neural Q-learning in Algorithm 2.

Theorem 6.3 (Convergence of Stochastic Update) . We set η to be of order T -1 / 2 in Algorithm 2. Under Assumptions 6.1 and 6.2, the output ̂ Q out of Algorithm 2 satisfies

Proof. See Appendix D.1 for a detailed proof.

<!-- formula-not-decoded -->

Corresponding to Proposition 4.7, Theorem 6.3 also implies the convergence to Q π ∗ ( s, a ) , which is omitted due to space limitations.

## 6.3 Implication for Policy Gradient

Theorem 6.3 can be further extended to handle neural soft Q-learning, where the max operator in the Bellman optimality operator is replaced by a more general softmax operator (Haarnoja et al.,

<!-- formula-not-decoded -->

2017; Neu et al., 2017). By exploiting the equivalence between soft Q-learning and policy gradient algorithms (Schulman et al., 2017; Haarnoja et al., 2018), we establish the global convergence of a variant of the policy gradient algorithm. Due to space limitations, we defer the discussion to Appendix E, throughout which we focus on two-layer neural networks. Our analysis can be extended to handle multi-layer neural networks using the proof techniques in Appendix F.

## 7 Conclusions

In this paper we prove that neural TD converges at a sublinear rate to the global optimum of the MSPBE for policy evaluation. In particular, we show how such global convergence is enabled by the overparametrization of neural networks. Moreover, we extend the convergence result to policy optimization, including (soft) Q-learning and policy gradient. Our results shed new light on the theoretical understanding of RL with neural networks, which is widely employed in practice.

## References

- Achiam, J., Knight, E. and Abbeel, P. (2019). Towards characterizing divergence in deep Qlearning. arXiv preprint arXiv:1903.08894 .
- Agazzi, A. and Lu, J. (2019). Temporal-difference learning for nonlinear value function approximation in the lazy training regime. arXiv preprint arXiv:1905.10917 .
- Allen-Zhu, Z., Li, Y. and Liang, Y. (2018a). Learning and generalization in overparameterized neural networks, going beyond two layers. arXiv preprint arXiv:1811.04918 .
- Allen-Zhu, Z., Li, Y. and Liang, Y. (2018b). Learning and generalization in overparameterized neural networks, going beyond two layers. arXiv preprint arXiv:1811.04918 .
- Allen-Zhu, Z., Li, Y. and Song, Z. (2018c). A convergence theory for deep learning via overparameterization. arXiv preprint arXiv:1811.03962 .
- Amiranashvili, A., Dosovitskiy, A., Koltun, V. and Brox, T. (2018). TD or not TD: Analyzing the role of temporal differencing in deep reinforcement learning. arXiv preprint arXiv:1806.01175 .

- Arora, S., Du, S. S., Hu, W., Li, Z. and Wang, R. (2019). Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. arXiv preprint arXiv:1901.08584 .
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. In International Conference on Machine Learning .
- Bertsekas, D. P. (2019). Feature-based aggregation and deep reinforcement learning: A survey and some new implementations. IEEE/CAA Journal of Automatica Sinica , 6 1-31.
- Bhandari, J., Russo, D. and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. arXiv preprint arXiv:1806.02450 .
- Bhatnagar, S., Precup, D., Silver, D., Sutton, R. S., Maei, H. R. and Szepesv´ ari, C. (2009). Convergent temporal-difference learning with arbitrary smooth function approximation. In Advances in Neural Information Processing Systems .
- Borkar, V. S. (2009). Stochastic Approximation: A Dynamical Systems Viewpoint , vol. 48. Springer.
- Borkar, V. S. and Meyn, S. P. (2000). The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38 447-469.
- Boyan, J. A. (1999). Least-squares temporal difference learning. In International Conference on Machine Learning .
- Boyan, J. A. and Moore, A. W. (1995). Generalization in reinforcement learning: Safely approximating the value function. In Advances in Neural Information Processing Systems .
- Bradtke, S. J. and Barto, A. G. (1996). Linear least-squares algorithms for temporal difference learning. Machine Learning , 22 33-57.
- Brandfonbrener, D. and Bruna, J. (2019a). Geometric insights into the convergence of nonlinear TD learning. arXiv preprint arXiv:1905.12185 .
- Brandfonbrener, D. and Bruna, J. (2019b). On the expected dynamics of nonlinear TD learning. arXiv preprint arXiv:1905.12185 .

- Cao, Y. and Gu, Q. (2019a). Generalization bounds of stochastic gradient descent for wide and deep neural networks. arXiv preprint arXiv:1905.13210 .
- Cao, Y. and Gu, Q. (2019b). A generalization theory of gradient descent for learning overparameterized deep ReLU networks. arXiv preprint arXiv:1902.01384 .
- Chizat, L. and Bach, F. (2018). A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956 .
- Chung, W., Nath, S., Joseph, A. and White, M. (2019). Two-timescale networks for nonlinear value function approximation. In International Conference on Learning Representations .
- Dalal, G., Sz¨ or´ enyi, B., Thoppe, G. and Mannor, S. (2018). Finite sample analyses for TD(0) with function approximation. In AAAI Conference on Artificial Intelligence .
- Daniely, A. (2017). SGD learns the conjugate kernel class of the network. In Advances in Neural Information Processing Systems .
- Dann, C., Neumann, G. and Peters, J. (2014). Policy evaluation with temporal differences: A survey and comparison. Journal of Machine Learning Research , 15 809-883.
- Du, S. S., Chen, J., Li, L., Xiao, L. and Zhou, D. (2017). Stochastic variance reduction methods for policy evaluation. In International Conference on Machine Learning .
- Duan, Y., Chen, X., Houthooft, R., Schulman, J. and Abbeel, P. (2016). Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning .
- Facchinei, F. and Pang, J.-S. (2007). Finite-Dimensional Variational Inequalities and Complementarity Problems . Springer Science &amp; Business Media.
- Fan, J., Ma, C. and Zhong, Y. (2019). A selective overview of deep learning. arXiv preprint arXiv:1904.05526 .
- Gao, R., Cai, T., Li, H., Wang, L., Hsieh, C.-J. and Lee, J. D. (2019). Convergence of adversarial training in overparametrized networks. arXiv preprint arXiv:1906.07916 .
- Geist, M. and Pietquin, O. (2013). Algorithmic survey of parametric value function approximation. IEEE Transactions on Neural Networks and Learning Systems , 24 845-867.

- Ghavamzadeh, M., Lazaric, A., Maillard, O. and Munos, R. (2010). LSTD with random projections. In Advances in Neural Information Processing Systems .
- Haarnoja, T., Tang, H., Abbeel, P. and Levine, S. (2017). Reinforcement learning with deep energybased policies. In International Conference on Machine Learning .
- Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290 .
- Harker, P. T. and Pang, J.-S. (1990). Finite-dimensional variational inequality and nonlinear complementarity problems: a survey of theory, algorithms and applications. Mathematical Programming , 48 161-220.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). Deep reinforcement learning that matters. In AAAI Conference on Artificial Intelligence .
- Hofmann, T., Sch¨ olkopf, B. and Smola, A. J. (2008). Kernel methods in machine learning. Annals of Statistics 1171-1220.
- Jaakkola, T., Jordan, M. I. and Singh, S. P. (1994). Convergence of stochastic iterative dynamic programming algorithms. In Advances in Neural Information Processing Systems .
- Jacot, A., Gabriel, F. and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems .
- Jain, P. and Kar, P. (2017). Non-convex optimization for machine learning. Foundations and Trends R © in Machine Learning , 10 142-336.
- Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in Neural Information Processing Systems .
- Kushner, H. and Yin, G. G. (2003). Stochastic Approximation and Recursive Algorithms and Applications . Springer Science &amp; Business Media.
- Lakshminarayanan, C. and Szepesvari, C. (2018). Linear stochastic approximation: How far does constant step-size and iterate averaging go? In International Conference on Artificial Intelligence and Statistics .

- Lazaric, A., Ghavamzadeh, M. and Munos, R. (2010). Finite-sample analysis of LSTD. In International Conference on Machine Learning .
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y ., Sohl-Dickstein, J. and Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. arXiv preprint arXiv:1902.06720 .
- Li, Y. and Liang, Y. (2018). Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in Neural Information Processing Systems .
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 .
- Liu, B., Liu, J., Ghavamzadeh, M., Mahadevan, S. and Petrik, M. (2015). Finite-sample analysis of proximal gradient TD algorithms. In Conference on Uncertainty in Artificial Intelligence .
- Melo, F. S., Meyn, S. P. and Ribeiro, M. I. (2008). An analysis of reinforcement learning with function approximation. In International Conference on Machine Learning .
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning .
- Neu, G., Jonsson, A. and G´ omez, V. (2017). A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 .
- Neyshabur, B., Li, Z., Bhojanapalli, S., LeCun, Y. and Srebro, N. (2018). Towards understanding the role of over-parametrization in generalization of neural networks. arXiv preprint arXiv:1805.12076 .
- Pfau, D. and Vinyals, O. (2016). Connecting generative adversarial networks and actor-critic methods. arXiv preprint arXiv:1610.01945 .
- Rahimi, A. and Recht, B. (2008a). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems .
- Rahimi, A. and Recht, B. (2008b). Uniform approximation of functions with random bases. In Annual Allerton Conference on Communication, Control, and Computing .

- Schulman, J., Chen, X. and Abbeel, P. (2017). Equivalence between policy gradients and soft Qlearning. arXiv preprint arXiv:1704.06440 .
- Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning .
- Srikant, R. and Ying, L. (2019). Finite-time error bounds for linear stochastic approximation and TD learning. arXiv preprint arXiv:1902.00923 .
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning , 3 9-44.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction . MIT press.
- Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesv´ ari, C. and Wiewiora, E. (2009a). Fast gradient-descent methods for temporal-difference learning with linear function approximation. In International Conference on Machine Learning .
- Sutton, R. S., Maei, H. R. and Szepesv´ ari, C. (2009b). A convergent o ( n ) temporal-difference algorithm for off-policy learning with linear function approximation. In Advances in Neural Information Processing Systems .
- Szepesv´ ari, C. (2010). Algorithms for reinforcement learning. Synthesis Lectures on Artificial Intelligence and Machine Learning , 4 1-103.
- Touati, A., Bacon, P.-L., Precup, D. and Vincent, P. (2017). Convergent tree-backup and retrace with function approximation. arXiv preprint arXiv:1705.09322 .
- Tsitsiklis, J. N. and Van Roy, B. (1997). Analysis of temporal-diffference learning with function approximation. In Advances in Neural Information Processing Systems .
- Tu, S. and Recht, B. (2017). Least-squares temporal difference learning for the linear quadratic regulator. arXiv preprint arXiv:1712.08642 .
- Wang, Y., Chen, W., Liu, Y., Ma, Z.-M. and Liu, T.-Y. (2017). Finite sample analysis of the GTD policy evaluation algorithms in Markov setting. In Advances in Neural Information Processing Systems .

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8 229-256.
- Zhang, C., Bengio, S., Hardt, M., Recht, B. and Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530 .
- Zou, D., Cao, Y., Zhou, D. and Gu, Q. (2018). Stochastic gradient descent optimizes overparameterized deep ReLU networks. arXiv preprint arXiv:1811.08888 .
- Zou, S., Xu, T. and Liang, Y. (2019). Finite-sample analysis for SARSA and Q-learning with linear function approximation. arXiv preprint arXiv:1902.02234 .

## A Representation Power of F B,m

## A.1 Background on RKHS

We consider the following kernel function

<!-- formula-not-decoded -->

Here φ is a random feature map parametrized by w , which follows a distribution with density p ( · ) (Rahimi and Recht, 2008a). Any function in the RKHS induced by K ( · , · ) takes the form

<!-- formula-not-decoded -->

such that each c ( · ) corresponds to a function f c ( · ) . The following lemma connects the H -norm of f c ( · ) to the /lscript 2 -norm of c ( · ) associated with the density p ( · ) , denoted by ‖ c ‖ p .

Proof. Recall if f ( x ) = ∫ X a ( y ) K ( x, y ) dy , then by the reproducing property (Hofmann et al., 2008), we have

Lemma A.1. It holds that ‖ f c ‖ 2 H = ‖ c ‖ 2 p = ∫ c ( w ) 2 p ( w ) dw .

<!-- formula-not-decoded -->

Now we write f ( · ) in the form of (A.2). By (A.1), we have

Thus, for c ( w ) = ∫ X a ( y ) φ ( y ; w ) dy , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof of Lemma A.1.

## A.2 F B, ∞ as RKHS

We characterize the approximate stationary point W ∗ and the corresponding ̂ Q 0 ( x ; W ∗ ) defined in Definition 4.1, which are attained by Algorithm 1 according to Theorems 4.4 and 4.6. We focus on its representation power when m →∞ . We first write F B,m in (4.5) as

<!-- formula-not-decoded -->

where the feature map { φ r ( x ) } r ∈ [ m ] is defined as

<!-- formula-not-decoded -->

As m → ∞ , the empirical distribution supported on { φ r ( x ) } r ∈ [ m ] , which has sample size m , converges to the corresponding population distribution. Therefore, from (A.3) we obtain

<!-- formula-not-decoded -->

Here p ( w ) is the density of N (0 , I d /d ) and f 0 ( x ) = lim m →∞ ̂ Q ( x ; W (0)) , which by the central limit theorem is a Gaussian process indexed by x . Furthermore, as discussed in Appendix A.1, φ ( x ; W ) induces an RKHS, namely H , which is the completion of the set of all functions that take the form

<!-- formula-not-decoded -->

In particular, H is equipped with the inner product induced by 〈 K ( · , x i ) , K ( · , x j ) 〉 H = K ( x i , x j ) . Rahimi and Recht (2008b) prove that, similar to Lemma A.1, for any f 1 ( · ) = ∫ φ ( · ; w ) /latticetop α 1 ( w ) · p ( w ) dw and f 2 ( · ) = ∫ φ ( · ; w ) /latticetop α 2 ( w ) · p ( w ) dw , we have f 1 , f 2 ∈ H , and moreover, their inner product has the following equivalence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a result, we have

which is known to be a rich function class (Hofmann et al., 2008). As m → ∞ , ̂ Q 0 ( · ; W ∗ ) becomes the fixed-point solution to the projected Bellman equation

<!-- formula-not-decoded -->

which also implies that ̂ Q 0 ( · ; W ∗ ) is the global optimum of the MSPBE

If we further assume that the Bellman evaluation operator T π satisfies T π ̂ Q 0 ( · ; W ∗ ) -f 0 ( · ) ∈ H and B is sufficiently large such that ‖T π ̂ Q 0 ( · ; W ∗ ) -f 0 ( · ) ‖ H ≤ B , then the projection Π F B, ∞ reduces to identity at T π ̂ Q 0 ( · ; W ∗ ) , which implies ̂ Q 0 ( · ; W ∗ ) = Q π ( · ) as they both solve the Bellman equation Q = T π Q . In other words, if the Bellman evaluation operator is closed with respect to F B, ∞ , which up to the intercept of f 0 ( · ) is a ball with radius B in H , ̂ Q 0 ( · ; W ∗ ) is the unique fixed-point solution to the Bellman equation or equivalently the global optimum of the MSBE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Proofs for Section 4

## B.1 Proof of Lemma 4.2

Proof. Following the same argument for W † in (4.4) and the definition of W ∗ in (4.6), we know that ̂ Q 0 ( · ; W ∗ ) is a fixed-point solution to the projected Bellman equation

Meanwhile, the Bellman evaluation operator T π is a γ -contraction in the /lscript 2 -norm ‖·‖ µ with γ &lt; 1 , since

<!-- formula-not-decoded -->

where the second equality follows from H¨ older's inequality and the fact that marginally x ′ and x have the same stationary distribution. Since the projection onto a convex set is nonexpansive, Π F B,m T π is also a γ -contraction. Thus, the projected Bellman equation in (B.1) has a unique fixed-point solution ̂ Q 0 ( · ; W ∗ ) in F B,m , which corresponds to the approximate stationary point W ∗ .

## B.2 Proof of Lemma 4.5

Proof. It suffices to show that E init ,µ [ ‖ g ( t ) ‖ 2 2 ] is upper bounded. By (4.9), we have where the inequality follows from the fact that, for any W ∈ S B ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

almost everywhere. Using the fact that x and x ′ have the same marginal distribution we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (B.3), we know that ̂ Q ( x ; W ) is 1 -Lipschitz continuous with respect to W . Therefore, we have

Plugging (B.5) into (B.4) and using the Cauchy-Schwarz inequality we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that by the initialization of ̂ Q 0 ( x ) as defined in (3.2), we have

Combining (B.2), (B.6), and (B.7) we obtain E init ,µ [ ‖ g ( t ) ‖ 2 2 ] = O ( B 2 ) . Since

<!-- formula-not-decoded -->

we conclude the proof of Lemma 4.5.

## B.3 Proof of Proposition 4.7

Proof. By the triangle inequality, we have

<!-- formula-not-decoded -->

Since Q π ( · ) is the fixed-point solution to the Bellman equation, we replace Q π ( · ) by T π Q π ( · ) and obtain

<!-- formula-not-decoded -->

Meanwhile, by Lemma 4.2, ̂ Q 0 ( · ; W ∗ ) is the solution to the projected Bellman equation, that is,

Combining (B.9) and (B.10), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality follows from the fact that Π F B,m T π is a γ -contraction, as discussed in the proof of Lemma 4.2. Plugging (B.11) into (B.8), we obtain

<!-- formula-not-decoded -->

which completes the proof of Proposition 4.7.

## C Proofs for Section 5

## C.1 Proof of Lemma 5.1

Proof. By the definition that ̂ Q t ( x ) = ̂ Q ( x ; W ( t )) and the definition of ̂ Q 0 ( x ; W ( t )) in (4.7), we have

<!-- formula-not-decoded -->

where we use the fact that ‖ x ‖ 2 = 1 . Note that 1 { W r ( t ) /latticetop x &gt; 0 } /negationslash = 1 { W r (0) /latticetop x &gt; 0 } implies

<!-- formula-not-decoded -->

Thus, we obtain

<!-- formula-not-decoded -->

Plugging (C.2) into (C.1), we obtain the following upper bound,

<!-- formula-not-decoded -->

Here the second inequality follows from the fact that

<!-- formula-not-decoded -->

for any x and y &gt; 0 . To characterize E init ,µ [ | ̂ Q t ( x ) -̂ Q 0 ( x ; W ( t )) | 2 ] , we first invoke the CauchySchwarz inequality and the fact that ‖ W ( t ) -W (0) ‖ 2 ≤ B , which gives

<!-- formula-not-decoded -->

Taking expectation on both sides, by Lemma H.1 we obtain

<!-- formula-not-decoded -->

Thus, we finish the proof of Lemma 5.1.

## C.2 Proof of Lemma 5.2

Proof. By the definition of g ( t ) and g 0 ( t ) in (5.1) and (5.2), respectively, we have

<!-- formula-not-decoded -->

Here to obtain the second inequality, we use the fact that, for any t ∈ [ T ] ,

<!-- formula-not-decoded -->

Taking expectation with respect to the random initialization on the both sides of (C.3), we obtain

<!-- formula-not-decoded -->

In the following, we characterize the three terms on the right-hand side of (C.4).

For (i) in (C.4), note that

<!-- formula-not-decoded -->

Since x and x ′ follow the same stationary distribution µ on the right-hand side of (C.5), by Lemma 5.1 we have

<!-- formula-not-decoded -->

For (ii) in (C.4), we have

<!-- formula-not-decoded -->

where the inequality follows from (C.2) and the fact that ‖ x ‖ 2 = 1 .

For (iii) in (C.4), we have

<!-- formula-not-decoded -->

To obtain an upper bound of the right-hand side of (C.8), we use the fact that

<!-- formula-not-decoded -->

which follows from (4.7), and obtain

<!-- formula-not-decoded -->

Since x and x ′ follow the same stationary distribution µ on the right-hand side of (C.8) and | γ | &lt; 1 , we have

<!-- formula-not-decoded -->

Plugging (C.6), (C.7), and (C.9) into (C.4), we obtain

<!-- formula-not-decoded -->

Invoking Lemmas H.1 and H.2, we obtain

<!-- formula-not-decoded -->

which finishes the proof of Lemma 5.2.

## C.3 Proof of Lemma 5.3

Proof. Recall that

We denote the locally linearized population semigradient g 0 ( t ) evaluated at the approximate stationary point W ∗ by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any W ( t ) ( t ∈ [ T ]) , by the convexity of S B , we have

We decompose the inner product ( g ( t ) -g ∗ 0 ) /latticetop ( W ( t ) -W ∗ ) on the right-hand side of (C.12) into two terms,

<!-- formula-not-decoded -->

It remains to characterize the first term ( g 0 ( t ) -g ∗ 0 ) /latticetop ( W ( t ) -W ∗ ) on the right-hand side of (C.13), since the second term ‖ g ( t ) -g 0 ( t ) ‖ 2 is characterized by Lemma 5.2. Note that by (C.10) and (C.11), we have

<!-- formula-not-decoded -->

where we use the following consequence of (4.7),

<!-- formula-not-decoded -->

Moreover, by (4.8) it holds that

<!-- formula-not-decoded -->

Combining (4.7), (C.14), and (C.15), we have

<!-- formula-not-decoded -->

where the last inequality is from the fact that x and x ′ have the same marginal distribution under µ and therefore by the Cauchy-Schwarz inequality,

<!-- formula-not-decoded -->

The inequality in (C.16) is the key to our convergence result. It shows that the locally linearized population semigradient update g 0 ( t ) is one-point monotone with respect to the approximate stationary point W ∗ .

Also, for ‖ g ( t ) -g ∗ 0 ‖ 2 2 on the right-hand side of (C.12), we have

<!-- formula-not-decoded -->

For the first term on the right-hand side of (C.17), by (C.14), (C.15), and the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

where the first inequality follows from the fact

<!-- formula-not-decoded -->

Plugging (C.16), (C.17), and (C.18) into (C.12), we finish the proof of Lemma 5.3.

## C.4 Proof of Lemma 5.4

Proof. For any W ( t ) ( t ∈ [ T ]) , by the convexity of S B , (4.9), and (C.11), we have

<!-- formula-not-decoded -->

Taking expectation on both sides conditional on W ( t ) , we obtain

<!-- formula-not-decoded -->

For the inner product ( g ( t ) -g ∗ 0 ) /latticetop ( W ( t ) -W ∗ ) on the right-hand side of (C.20), it follows from (C.13) and (C.16) that

<!-- formula-not-decoded -->

Meanwhile, for E µ [ ‖ g ( t ) -g ∗ 0 ‖ 2 2 | W ( t )] on the right-hand side of (C.20), we have the decomposition

<!-- formula-not-decoded -->

where the inequality follows from (C.17) and (C.18). Taking expectation on the both sides of (C.20) with respect to W ( t ) , we complete the proof of Lemma 5.4.

## C.5 Proof of Theorem 4.4

Proof. By Lemma 5.2 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Setting η = (1 -γ ) / 8 in Algorithm 1, by (C.21), (C.22), and Lemma 5.3, we have

<!-- formula-not-decoded -->

Telescoping (C.23) for t = 0 , . . . , T -1 , we obtain

<!-- formula-not-decoded -->

Recall that as define in (4.7), ̂ Q 0 ( · ; W ) is linear in W . By Jensen's inequality, we have

Next we characterize the output ̂ Q out ( · ) = ̂ Q ( · ; W ) of Algorithm 1. Since S B is convex and W ∈ S B , by Lemma 5.1 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the Cauchy-Schwarz inequality we have

<!-- formula-not-decoded -->

Here we plug in (C.24) and (C.25) and obtain

<!-- formula-not-decoded -->

which completes the proof of Theorem 4.4.

## C.6 Proof of Theorem 4.6

Proof. Similar to (C.23), by Lemmas 4.5, 5.2, and 5.4 we have

<!-- formula-not-decoded -->

Telescoping (C.27) for t = 0 , . . . , T -1 , by η 2 ≤ 1 /T we have

<!-- formula-not-decoded -->

Meanwhile, when T &lt; (8 / (1 -γ )) 2 , we have η = (1 -γ ) / 8 and where η = min { 1 / √ T, (1 -γ ) / 8 } . Note that when T ≥ (8 / (1 -γ )) 2 , we have η = 1 / √ T and √ T · ( 2 η (1 -γ ) -8 η 2 ) = 2(1 -γ ) -8 / √ T ≥ 1 -γ.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since | 1 -γ | &lt; 1 , we obtain that for any T ∈ N ,

Similar to (C.24) and (C.26), by combining (C.28) and (C.29) with Lemma 5.1, we obtain

<!-- formula-not-decoded -->

which completes the proof of Theorem 4.6.

## D Proofs for Section 6

Similar to the population semigradient g ( t ) in policy evaluation, we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Our proof extends that of Theorem 2 in Zou et al. (2019) for characterizing linear Q-learning. We additionally incorporate the error of local linearization and also handle soft Q-learning in the next section. The following lemma is analogous to Lemma 4.2.

LemmaD.1. Under Assumption 6.1, there exists an approximate stationary point W ∗ that satisfies (6.3). Also, ̂ Q 0 ( · ; W ∗ ) is unique almost everywhere.

Proof. We prove the lemma by showing that T is a contraction in ‖ · ‖ µ exp for any Q 1 , Q 2 ∈ F B,m . By the definition of the Bellman optimality operator T , we have

<!-- formula-not-decoded -->

Under Assumption 6.1, for any Q 1 , Q 2 ∈ F B,m , we have

<!-- formula-not-decoded -->

Therefore, Π F B,m T is a γ/ ( γ + ν ) -contraction in ‖ · ‖ µ exp , since Π F B,m is nonexpansive. Since the set S B of feasible W is closed and bounded, F B,m is complete under ‖ · ‖ µ exp . Thus, Π F B,m T has a unique fixed point ̂ Q 0 ( · ; W ∗ ) in F B,m , which corresponds to W ∗ .

The following lemma is analogous to Lemma 5.2 with a similar proof.

Lemma D.2. For any t ∈ [ T ] , we have

<!-- formula-not-decoded -->

Proof. By the definitions of z ( t ) and z 0 ( t ) in (D.2) and (D.3), respectively, we have

<!-- formula-not-decoded -->

For notational simplicity, we define ̂ Q /sharp t ( s ) = max a ∈A ̂ Q t ( s, a ) . Recall that ̂ Q /sharp 0 ( s ; W ) is similarly defined in Assumption 6.1. Then on the right-hand side of (D.5), we have

<!-- formula-not-decoded -->

Thus, from (D.5) we obtain

<!-- formula-not-decoded -->

Here we use the fact that, for any t ∈ [ T ] ,

<!-- formula-not-decoded -->

Taking expectation on the both sides of (D.6) with respect to the random initialization, we obtain

<!-- formula-not-decoded -->

Similar to the proof of Lemma 5.2, we characterize the three terms on the right-hand side of (D.7). For (i) in (D.7), recall that Lemma 5.1 gives

<!-- formula-not-decoded -->

We establish a similar upper bound of (ii) in (D.7). Note that

<!-- formula-not-decoded -->

Similar to Lemma 5.1, we have the following lemma for characterizing the right-hand side of (D.8).

Lemma D.3. Under Assumption 6.2, there exists a constant c 4 &gt; 0 such that for any t ∈ [ T ] , it holds that

<!-- formula-not-decoded -->

Proof. See Appendix H.1 for a detailed proof.

By Lemma D.3, (ii) in (D.7) satisfies

<!-- formula-not-decoded -->

For (iii) in (D.7), in the proof of Lemma 5.2, we show that

<!-- formula-not-decoded -->

Meanwhile, by the definition that ̂ Q /sharp 0 ( s ; W ) = max a ∈A ̂ Q 0 ( s, a ; W ) and (4.7), we have where a ′ max = argmax a ′ ∈A ̂ Q 0 ( s ′ , a ′ ; W ( t )) . Then applying Lemmas H.1, H.2, and H.3, we obtain that (iii) in (D.7) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the upper bounds of (i)-(iii) in (D.7), we complete the proof of Lemma D.2.

## D.1 Proof of Theorem 6.3

Proof. Recall that z ( t ) , z ( t ) , z 0 ( t ) , and z ∗ 0 are defined in (D.1)-(D.4), respectively. Similar to the proof of Theorem 4.4, we have

<!-- formula-not-decoded -->

To characterize the inner product on the right-hand side of (D.10), we take conditional expectation and obtain

<!-- formula-not-decoded -->

We establish a lower bound of ( z 0 ( t ) -z ∗ 0 ) /latticetop ( W ( t ) -W ∗ ) as follows. By (D.3) and (D.4), we have

<!-- formula-not-decoded -->

Applying H¨ older's inequality to the second term on the right-hand side of (D.12), we obtain

<!-- formula-not-decoded -->

By Assumption 6.1, we have

<!-- formula-not-decoded -->

which implies that, on the right-hand side of (D.12),

<!-- formula-not-decoded -->

Therefore, from (D.12) we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to the proof of Theorem 4.4, for ‖ z ( t ) -z ∗ 0 ‖ 2 2 on the right-hand side of (D.10), we have where the expectation of ‖ z ( t ) -z 0 ( t ) ‖ 2 2 on the right-hand side is characterized by Lemma D.2, while the expectation of ‖ z 0 ( t ) -z ∗ 0 ‖ 2 2 has the following upper bound,

Here the last inequality follows from Assumption 6.1 as γ/ ( γ + ν ) &lt; 1 .

<!-- formula-not-decoded -->

Plugging (D.11), (D.13), (D.14), and (D.15) into (D.10) yields the following inequality, which parallels Lemma 5.4,

<!-- formula-not-decoded -->

Rearranging terms in (D.16), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In parallel with Lemma 4.5, we establish an upper bound of the variance E W,µ exp [ ‖ z ( t ) -z ( t ) ‖ 2 2 ] in the right-hand side of (D.17), which is independent of t and m . Note that by ‖∇ W ̂ Q t ( s, a ) ‖ 2 ≤ 1 , we have

To characterize ̂ Q t ( s, a ) 2 , we have where the second inequality comes from (B.5). Similarly, to characterize max a ′ ∈A ̂ Q t ( s ′ , a ′ ) in (D.18), we take maximum on the both side of (D.19) and obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging (D.19) and (D.20) into (D.18), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To upper bound the expectation of (D.21), it remains to characterize E init ,µ exp [ ̂ Q 0 ( s, a ) 2 ] and

In fact, for any ( s, a ) , E init [ ̂ Q 0 ( s, a ) 2 ] has the following uniform upper bound,

For E init ,s ′ ∼ µ exp [max a ′ ∈A ̂ Q 0 ( s ′ , a ′ ) 2 ] , we use the inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With the variance term upper bounded, we take expectation on (D.17) with respect to the random initialization and obtain which gives us the same upper bound as in (D.22) but with an additional factor |A| . Therefore, we know that the variance E init ,µ exp [ ‖ z ( t ) -z ( t ) ‖ 2 2 ] in the expectation of (D.17) has an upper bound σ 2 z = O ( B 2 ) , according to the fact that E init ,µ exp [ ‖ z ( t ) -z ( t ) ‖ 2 2 ] ≤ E init ,µ exp [ ‖ z ( t ) ‖ 2 2 ] .

<!-- formula-not-decoded -->

We set η = min { 1 / √ T, ν/ 8( γ + ν ) } and telescope (D.23) for t = 0 , . . . , T -1 . Then the rest proof mirrors that of Theorem 4.6. Thus, we complete the proof of Theorem 6.3.

## E From Neural Soft Q-Learning to Policy Gradient

## E.1 Global Convergence of Neural Soft Q-Learning

We extend the global convergence of neural Q-learning in Section 6 to neural soft Q-learning, where the max operator is replaced by a more general softmax operator. More specifically, we consider the soft Bellman optimality operator

<!-- formula-not-decoded -->

which in parallel with (6.2) corresponds to the update

<!-- formula-not-decoded -->

See Algorithm 3 for a detailed description of such neural soft Q-learning algorithm.

In parallel with Assumption 6.1, we require the following regularity condition on the exploration policy π exp .

Assumption E.1 (Regularity of Exploration Policy π exp ) . There exists a constant ν ′ &gt; 0 such that for any W 1 , W 2 ∈ S B , it holds that

<!-- formula-not-decoded -->

We remark that, when β → ∞ , the softmax operator converges to the max operator, which implies that Assumptions E.1 and 6.1 are equivalent.

The approximate stationary point W ∗ of the projected soft Q-learning satisfies

<!-- formula-not-decoded -->

where ̂ Q 0 ( · ; W ∗ ) uniquely exists by the same proof of Lemma D.1. Under the above regularity condition, we can extend Theorem 6.3 to cover neural soft Q-learning.

Theorem E.2 (Convergence of Stochastic Update) . We set η to be of order T -1 / 2 in Algorithm 3. Under Assumptions E.1 and 6.2, the output ̂ Q out of Algorithm 3 satisfies

Proof. The proof of Theorem E.2 mirrors that of Theorem 6.3 with the max operator replaced by the softmax operator in (E.1). We prove that the same claim of Lemma D.2 holds under Assumption 6.2, for which it suffices to upper bound (i), (ii), and (iii) in (D.7). Note that (i) does not involve the max operator. For (ii), we lay out the following lemma.

<!-- formula-not-decoded -->

Lemma E.3. For any W ∈ S B and the constant c 4 in Lemma D.3, we have

Proof. The softmax operator has the following duality. For any function Q ( s, a ) , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∆ is the set of all probability distributions over A and H ( π ( s )) is the entropy of π ( s ) . Hence, we obtain

<!-- formula-not-decoded -->

By applying Lemma D.3, we complete the proof of Lemma E.3.

Meanwhile, note that the upper bound in (D.9) of (iii) in (D.7) still holds by Lemma H.3. Thus, the claim of Lemma D.2 holds for neural soft Q-learning. Moreover, (D.17) also holds for neural soft Q-learning under Assumption E.1. To further extend the upper bound of E init ,µ exp [ ‖ z ( t ) -z ( t ) ‖ 2 2 ] to neural soft Q-learning, it remains to upper bound softmax a ′ ∈A ̂ Q t ( s ′ , a ′ ) , which replaces max a ′ ∈A ̂ Q t ( s ′ , a ′ ) in (D.21), by where β is the parameter of the softmax operator. Here the inequality follows from (E.2). The additional term β -1 · log |A| is independent of t and m . Thus, we obtain an upper bound of the variance E init ,µ exp [ ‖ z ( t ) -z ( t ) ‖ 2 2 ] , which is independent of t and m . With (D.17) and Lemma D.2, the proof of Theorem E.2 follows from that of Theorem 6.3.

<!-- formula-not-decoded -->

## Algorithm 3 Neural Soft Q-Learning

<!-- formula-not-decoded -->

Initialization:

- 2: For t = 0 to T -2 :

<!-- formula-not-decoded -->

- 3: Sample a tuple ( s, a, r, s ′ ) from the stationary distribution µ exp of the exploration policy π exp
- 4: Bellman residual calculation: δ ← ̂ Q ( s, a ; W ( t )) -r -γ softmax a ′ ∈A ̂ Q ( s ′ , a ′ ; W ( t ))

<!-- formula-not-decoded -->

- 8: End For
- 6: Projection: W ( t +1) ← argmin W ∈ S B ‖ W -˜ W ( t +1) ‖ 2 7: Averaging: W ← t +1 t +2 · W + 1 t +2 · W ( t +1)
- 9: Output: ̂ Q out ( · ) ← ̂ Q ( · ; W )

## E.2 Implication for Policy Gradient

In this section, we briefly summarize the equivalence between policy gradient algorithms and neural soft Q-learning (Schulman et al., 2017; Haarnoja et al., 2018), which implies that our results are extendable to characterize a variant of the policy gradient algorithm.

Wedefine π θ as the Boltzmann policy corresponding to the Q-function Q θ , which is parametrized by θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here π is the uniform policy and V θ is the partition function. In the context of neural soft Qlearning, we use the parametrization

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the sequel, we show that the population semigradient in soft Q-learning, which is defined in (D.2), equals a variant of population policy gradient, given that the exploration policy π exp in (D.2)

From (E.3) we have

is π θ . Recall that the population semigradient in soft Q-learning is given by

<!-- formula-not-decoded -->

where µ θ is the stationary distribution of π θ . For notational simplicity, we define

<!-- formula-not-decoded -->

Plugging (E.5) and (E.7) into (E.6), we obtain

<!-- formula-not-decoded -->

Taking gradient on the both sides of (E.5), we obtain

<!-- formula-not-decoded -->

Plugging (E.9) into the right-hand side of (E.8) yields

<!-- formula-not-decoded -->

By the definition of the KL-divergence, we have

<!-- formula-not-decoded -->

Thus, on the right-hand side of (E.10), we have

<!-- formula-not-decoded -->

Also, since

<!-- formula-not-decoded -->

on the right-hand side of (E.10), we have

<!-- formula-not-decoded -->

Plugging (E.11) and (E.13) into (E.10), we obtain

<!-- formula-not-decoded -->

We characterize (i) and (ii) on the right-hand side of (E.14). For (i), by the definition of ξ in (E.7), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here the operator T ˜ π KL is defined as which is the Bellman evaluation operator for the value function V ˜ π ( s ) associated with the KLregularized reward

<!-- formula-not-decoded -->

By (E.15), E ( s,a,s ′ ) ∼ µ θ [ -∇ θ V θ ( s ) · ξ ] is the population semigradient for the evaluation of policy π θ . Now we characterize (ii) in (E.14). First, by the definition of the KL-divergence, we have

<!-- formula-not-decoded -->

Here the second equality follows from (E.12). Hence, (ii) in (E.14) takes the form

<!-- formula-not-decoded -->

We show that (ii) in (E.14) is the population policy gradient. For (ii).a, note that ξ defined in (E.7) is an unbiased estimator of the advantage function Q π θ ( s, a ) -V π θ ( s ) associated with the KL-regularized reward r KL defined in (E.16). If we denote by J KL ( π θ ) the expected total reward, then (ii).a is an estimator of the population policy gradient ∇ θ J KL ( π θ ) . For (ii).b, it is the gradient of the entropy regularization

<!-- formula-not-decoded -->

Therefore, we recover the policy gradient update in the Q-learning updating scheme. Combining (i) and (ii) in (E.14), we obtain a variant of the policy gradient algorithm, which is connected with the soft actor-critic algorithm (Haarnoja et al., 2018). Hence, our global convergence of neural soft Q-learning extends to a variant of the actor-critic algorithm. See Algorithm 4 for a detailed description of such an algorithm with ̂ Q θ parametrized by a two-layer neural network, which can also be extended to allow for a multi-layer neural network. In parallel with V θ ( s ) in (E.4), we define

<!-- formula-not-decoded -->

## F Extension to Multi-Layer Neural Networks

In this section, we generalize our main results in Section 4 to the setting where the Q-function is parametrized by a multi-layer neural network. Similar to the setting with a two-layer neural network, we represent the state-action pair ( s, a ) ∈ S ×A by a vector x = ψ ( s, a ) ∈ X ⊆ R d with d &gt; 2 , where ψ is a given one-to-one feature map. With a slight abuse of notation, we use ( s, a ) and x interchangeably. Without loss of generality, we assume that ‖ x ‖ 2 = 1 . The Q-function is parametrized by

<!-- formula-not-decoded -->

## Algorithm 4 Neural Soft Actor-Critic

<!-- formula-not-decoded -->

- 3: Policy Update: π t ( a | s ) ∝ π ( a | s ) · exp( β · ̂ Q ( s, a ; W ( t ))) 4: Sample a tuple ( s, a, r, s ′ ) from the stationary distribution µ t of policy π t
- 2: For t = 0 to T -2 :
- 5: Reward regularization: r KL ← r -β -1 · D KL ( π t ( · | s ) ‖ π ( · | s ))
- 6: Bellman residual calculation: ξ ← r KL + γ ̂ V ( s ′ ; W ( t )) -̂ V ( s ; W ( t )) 7: Actor Update:

<!-- formula-not-decoded -->

- 12: End For
- ˜ 9: Critic update: ˜ W ′ ( t +1) ← ˜ W ( t +1) + η · ξ · ∇ W ̂ V ( s ; W ( t )) 10: Projection: W ( t +1) ← argmin W ∈ S B ‖ W -˜ W ′ ( t +1) ‖ 2 11: Averaging: W ← t +1 t +2 · W + 1 t +2 · W ( t +1)
- 13: Output: ̂ Q out ( · ) ← ̂ Q ( · ; W ) , π out ( a | s ) ∝ π ( a | s ) · exp( β · ̂ Q out ( s, a ))

where A ∈ R m × d , W ( h ) ∈ R m × m , and b ∈ R m are the weights. Here x ( h ) corresponds to the ( h +1) -th hidden layer and y gives ̂ Q ( x ; W ) . For notational simplicity, we define

Each entry of A and { W ( h ) } H h =1 is independently initialized by N (0 , 2) , while each entry of b is independently initialized by N (0 , 1) . During training, we only update W using the TD update in (3.1), while keeping A and b fixed as the random initialization (Allen-Zhu et al., 2018c; Gao et al., 2019).

<!-- formula-not-decoded -->

Similar to (4.7), we redefine the locally linearized Q-function as

<!-- formula-not-decoded -->

where W (0) is the random initialization of W . Also, we redefine

<!-- formula-not-decoded -->

Correspondingly, we redefine g ( t ) , g ( t ) , g 0 ( t ) , and g 0 ( t ) by plugging the redefined ̂ Q and ̂ Q 0 into (4.9), (5.1), and (5.2), respectively. Also, we redefine W ∗ and g ∗ 0 by plugging the redefined S B , F B,m , ̂ Q , and ̂ Q 0 into (4.6) and (C.11), respectively.

In the sequel, we establish the global convergence of neural TD with a multi-layer neural network. Note that we abandon Assumption 4.3 at the cost of a slightly worse upper bound of the error of local linearization, which is characterized by the following lemma.

Lemma F.1. Let m = Ω( d 3 / 2 log 3 / 2 ( m 1 / 2 /B ) / ( BH 3 / 2 )) and B = O ( m 1 / 2 H -6 log -3 m ) . With probability at least 1 -e -Ω(log 2 m ) with respect to the random initialization, it holds for any W ∈ S B and x ∈ R d with ‖ x ‖ 2 = 1 that

Proof. See Allen-Zhu et al. (2018c); Gao et al. (2019) for a detailed proof. In detail, following the proofs of Lemmas A.5 and A.6 in Gao et al. (2019), we have that the first and second equalities hold with probability at least 1 -O ( H ) · e -Ω( B 2 / 3 m 2 / 3 H ) . Also, following the proof of Theorem 1 in Allen-Zhu et al. (2018c), we have that the third equality holds with probability at least 1 -e -Ω(log 2 m ) , which concludes the proof of Lemma F.1.

<!-- formula-not-decoded -->

The following lemma replaces Lemmas 5.1, 5.2, and 4.5.

Lemma F.2. Under the same condition of Lemma F.1, with probability at least 1 -e -Ω(log 2 m ) with respect to the random initialization, it holds for any t ∈ [ T ] and x ∈ R d with ‖ x ‖ 2 = 1 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove the three equalities one by one.

Proof of (F.2) : Note that W ∈ S B implies ‖ W -W (0) ‖ 2 ≤ B √ H . Hence, we have

<!-- formula-not-decoded -->

Applying Lemma F.1 and the fact that (1 -s ) W (0) + sW ( t ) ∈ S B for any s ∈ [0 , 1] , we obtain (F.2).

Proof of (F.3) : Similar to (C.3), we have

<!-- formula-not-decoded -->

To upper bound (i), recall the definitions of δ ( x, r, x ′ ; W ) and δ 0 ( x, r, x ′ ; W ) ,

<!-- formula-not-decoded -->

Following from (F.2), which is proved previously, we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

To upper bound (ii), by Lemma F.1 we have

<!-- formula-not-decoded -->

To upper bound (iii), by the triangle inequality we have

<!-- formula-not-decoded -->

By (F.1), we have

<!-- formula-not-decoded -->

Also, the same upper bound holds for | ̂ Q 0 ( x ′ ; W ( t )) | . Thus, from (F.8) we obtain

To upper bound (iv), by Lemma F.1 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Plugging (F.6), (F.7), (F.10), and (F.11) into (F.5), we obtain

<!-- formula-not-decoded -->

Proof of (F.4) : By the redefinition of g ( t ) , we have

<!-- formula-not-decoded -->

By (F.8) and (F.9), we have

<!-- formula-not-decoded -->

Meanwhile, by Lemma F.1 we have

<!-- formula-not-decoded -->

Combining (F.12), (F.13), and (F.14), we obtain ‖ g ( t ) ‖ 2 2 = O ( B 2 H 5 log 2 m ) . Also, by the definition of g ( t ) and Jensen's inequality, we have

<!-- formula-not-decoded -->

Therefore, we conclude the proof of Lemma F.2.

Thus, by the triangle inequality, we obtain ‖ g ( t ) -g ( t ) ‖ 2 2 = O ( B 2 H 5 log 2 m ) .

Now we present the global convergence of neural TD with a multi-layer neural network.

Theorem F.3. Let m = Ω( d 3 / 2 log 3 / 2 ( m 1 / 2 /B ) / ( BH 3 / 2 )) and B = O ( m 1 / 2 H -6 log -3 m ) . We set η = 1 / √ T and H = O ( T 1 / 4 ) in Algorithm 1. With probability at least 1 -e -Ω(log 2 m ) with respect to the random initialization, the output ̂ Q out of Algorithm 1 satisfies where the expectation is taken with respect to the randomness of W in Algorithm 1 and x ∼ µ .

<!-- formula-not-decoded -->

Proof. We first reestablish Lemma 5.4. Similar to (C.19), for any W ( t ) ( t ∈ [ T ]) , the convexity of S B implies

<!-- formula-not-decoded -->

Taking expectation on both sides with respect to the tuple ( x, r, x ′ ) ∼ µ conditional on W ( t ) , we obtain

<!-- formula-not-decoded -->

For the inner product ( g ( t ) -g ∗ 0 ) /latticetop ( W ( t ) -W ∗ ) on the right-hand side of (F.16), following the same proof of (C.13) and (C.16), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Meanwhile, for E µ [ ‖ g ( t ) -g ∗ 0 ‖ 2 2 | W ( t )] on the right-hand side of (F.16), we have the decomposition

Here the inequality follows from (C.17) and (C.18), where we plug ‖∇ W ̂ Q ( x ; W ) ‖ 2 = O ( H ) into (C.18) instead of ‖∇ W ̂ Q ( x ; W ) ‖ 2 ≤ 1 . Combining (F.17) with (F.18) and taking expectation on the both sides of (F.16), we obtain the following inequality, which corresponds to Lemma 5.4,

<!-- formula-not-decoded -->

Here the expectation on the left-hand side is taken with respect to the randomness of W ( t + 1) , which is determined by W ( t ) and the tuple ( x, r, x ′ ) ∼ µ drawn at the current iteration. Applying Lemma F.2 to (F.19), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Telescoping (F.20) for t = 0 , 1 , . . . , T -1 , we obtain

<!-- formula-not-decoded -->

Here the expectation is taken conditional on the random initialization of ̂ Q . Following the same proof of Theorem 4.6, that is, using the triangle inequality and the upper bound of | ̂ Q ( x ; W ( t )) -̂ Q 0 ( x ; W ( t )) | in Lemma F.2, we conclude the proof of Theorem F.3.

## G Extension to Markov Sampling

Previously, we assume that the tuples ( x, r, x ′ ) in Algorithm 1 are independently sampled from the stationary distribution µ of policy π . In this section, we weaken such an assumption by allow-

ing the tuples to be sequentially sampled from the β -mixing Markov chain induced by policy π . Our analysis extends Section 8 in Bhandari et al. (2018), which focuses on the setting with linear function approximation. In contrast, we stick to the same setting as in Appendix F, where the Q-function is parametrized by a multi-layer neural network. For notational simplicity, we omit the conditioning on the random initialization in all the following expectations.

The following assumption states that the Markov chain of states is β -mixing.

Assumption G.1. There exist constants ι &gt; 0 and β ∈ (0 , 1) such that where P t ( · | s 0 = s ) is the conditional distribution of s t given s 0 = s , µ S is the marginal distribution of s under the stationary distribution µ , and d TV denotes the total variation distance.

<!-- formula-not-decoded -->

In the following, we establish the counterpart of Theorem F.3 under Markov sampling. Taking conditional expectation on the both sides of (F.15), we have

<!-- formula-not-decoded -->

Here recall that g ( t ) is the population semigradient defined with respect to the stationary distribution. Rearranging terms in (G.1), we have

<!-- formula-not-decoded -->

Plugging (F.17) and (F.18) into (G.2), and taking expectation with respect to the current iterate W ( t ) and the tuple ( x, r, x ′ ) drawn at the current iteration, similar to (F.19) we obtain

<!-- formula-not-decoded -->

Thus, similar to (F.21) we obtain

<!-- formula-not-decoded -->

To upper bound the left-hand side of (G.3), we upper bound E [( g ( t ) -g ( t )) /latticetop ( W ( t ) -W ∗ )] in the following lemma.

Lemma G.2. Let m = Ω( d 3 / 2 log 3 / 2 ( m 1 / 2 /B ) / ( BH 3 / 2 )) and B = O ( m 1 / 2 H -6 log -3 m ) . We set η = 1 / √ T and H = O ( T 1 / 4 ) in Algorithm 1, where the tuples ( x t , r t , x t +1 ) are sampled from a Markov chain satisfying Assumption G.1. With probability at least 1 -e -Ω(log 2 m ) with respect to the random initialization, for all t = 0 , 1 , . . . , T -1 , it holds that

<!-- formula-not-decoded -->

Proof. For notational simplicity, we define

<!-- formula-not-decoded -->

In the following, we prove that the function ζ t ( W ) = ( ˜ g ( W ) -g ( t, W )) /latticetop ( W -W ∗ ) is bounded and approximately Lipschitz continuous. Then Lemma G.2 is a direct application of Lemmas 10 and 11 in Bhandari et al. (2018).

For any W ∈ S B , by the definition of ζ t ( W ) and the Cauchy-Schwarz inequality, we have

Here we use the fact that ‖ W -W ∗ ‖ ≤ 2 BH 1 / 2 and ‖ ˜ g ( W ) -g ( t, W ) ‖ 2 = O ( BH 5 / 2 log m ) , which follows from the same proof of (F.4) in Lemma F.2.

<!-- formula-not-decoded -->

Also, for any W,W ′ ∈ S B , using the triangle inequality and the Cauchy-Schwarz inequality,

we have where ‖ ˜ g ( W ) -g ( t, W ) ‖ 2 = O ( BH 5 / 2 log m ) following the same proof of (F.4) in Lemma F.2. Meanwhile, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from the triangle inequality and the Cauchy-Schwarz inequality, while the second equality is implied by Lemma F.1. The same upper bound holds for ‖ ˜ g ( W ) -g ( W ′ ) ‖ 2 , plugging which into (G.4) yields

<!-- formula-not-decoded -->

Following the proof of the first inequality of Lemma 11 in Bhandari et al. (2018), we conclude the proof of Lemma G.2. The only difference with Lemma 11 in Bhandari et al. (2018) is that ζ t ( W ) is only approximately Lipschitz continuous rather than exactly Lipschitz continuous, which incurs the additional term O ( B 7 / 3 H 9 / 2 m -1 / 6 log 3 / 2 m ) in Lemma G.2.

Applying Lemma G.2 to (G.3), we obtain the following theorem under Markov sampling.

Theorem G.3 (Convergence of Stochastic Update) . Let m = Ω( d 3 / 2 log 3 / 2 ( m 1 / 2 /B ) / ( BH 3 / 2 )) and B = O ( m 1 / 2 H -6 log -3 m ) . We set η = 1 / √ T and H = O ( T 1 / 4 ) in Algorithm 1, where the tuples ( x t , r t , x t +1 ) are sampled from a Markov chain satisfying Assumption G.1. With probability

at least 1 -e -Ω(log 2 m ) with respect to the random initialization, the output ̂ Q out of Algorithm 1 satisfies

E W,µ [( ̂ Q out ( x ) -̂ Q 0 ( x ; W ∗ ) ) 2 ] = O ( ( B 2 H 5 / √ T + B 8 / 3 m -1 / 6 H 8 ) · log 3 m · log T ) . Proof. By Lemma G.2, we have that the left-hand side of (G.3) satisfies

<!-- formula-not-decoded -->

Hence, similar to the proof of Theorem 4.6, using the triangle inequality and the upper bound of | ̂ Q ( x ; W ) -̂ Q 0 ( x ; W ) | given by (F.2) in Lemma F.2, we conclude the proof of Theorem G.3.

## H Auxiliary Lemmas

Under Assumption 4.3, we establish the following auxiliary lemmas on the random initialization W (0) and the stationary distribution µ , which plays a key role in quantifying the error of local linearization.

Lemma H.1. There exists a constant c 1 &gt; 0 such that for any random vector W with ‖ W -W (0) ‖ 2 ≤ B , it holds that

<!-- formula-not-decoded -->

Proof. By Assumption 4.3, we have

<!-- formula-not-decoded -->

Applying H¨ older's inequality to the right-hand side, we obtain

<!-- formula-not-decoded -->

where the second inequality follows from

<!-- formula-not-decoded -->

Setting c 1 = c 0 · E w ∼ N (0 ,I d /d ) [1 / ‖ w ‖ 2 2 ] 1 / 2 , we complete the proof of Lemma H.1.

Lemma H.2. There exists a constant c 2 &gt; 0 such that for any random vector W with ‖ W -W (0) ‖ 2 ≤ B , it holds that

<!-- formula-not-decoded -->

Proof. By the definition of ̂ Q 0 ( x ) = ̂ Q 0 ( x ; W (0)) in (4.7), we have

/negationslash

<!-- formula-not-decoded -->

Following the same derivation of (H.1) and (H.2), we have

/negationslash

<!-- formula-not-decoded -->

Note that b r and b s are independent of W (0) and E init [ b r b s ] = 0 . Thus, we obtain

<!-- formula-not-decoded -->

By the definition of σ ( W r (0) /latticetop x ) and the fact that ‖ x ‖ 2 = 1 , we have

<!-- formula-not-decoded -->

Hence, it holds that

<!-- formula-not-decoded -->

By (H.3) and the fact that

<!-- formula-not-decoded -->

the right-hand side of (H.4) is O ( Bm -1 / 2 ) . Setting

<!-- formula-not-decoded -->

we complete the proof of Lemma H.2.

Lemma H.3. For any random vector W with ‖ W -W (0) ‖ 2 ≤ B , we have

<!-- formula-not-decoded -->

Proof. The proof mirrors that of Lemma H.2. We utilize the fact that A is finite, so that the expectation of the maximum can be upper bounded by a finite sum of expectations,

<!-- formula-not-decoded -->

For each expectation E s ∼ µ [ ̂ Q 0 ( s, a ) 2 ] , note that the distribution of ( s, a ) is independent of the initialization. Hence, the same proof of Lemma H.2 is applicable, as E s ∼ µ [ ̂ Q 0 ( s, a ) 2 ] plays the same role of E µ [ ̂ Q 0 ( x ) 2 ] . Thus, we obtain the same upper bound in Lemmas H.1 and H.2 except for an extra factor involving |A| , which however does not change the order of m .

## H.1 Proof of Lemma D.3

Proof. In the proof of Lemma 5.1, we show that, for any s ∈ S and a ∈ A ,

<!-- formula-not-decoded -->

Taking maximum over a , we obtain

<!-- formula-not-decoded -->

Taking expectation with respect to the random initialization and the stationary distribution of s , we obtain

<!-- formula-not-decoded -->

By Assumption 6.2, it holds that

<!-- formula-not-decoded -->

Applying H¨ older's inequality to the right-hand side, we obtain

<!-- formula-not-decoded -->

Setting c = 4 c E [1 / w ] , we finish the proof of Lemma D.3.

4 3 · w ∼ N (0 ,I d /d ) ‖ ‖ 2 2 1 / 2