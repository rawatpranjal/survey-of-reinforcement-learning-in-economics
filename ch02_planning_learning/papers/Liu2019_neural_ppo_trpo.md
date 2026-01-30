## Neural Proximal/Trust Region Policy Optimization Attains Globally Optimal Policy

Boyi Liu ∗† Qi Cai ∗‡ Zhuoran Yang § Zhaoran Wang ¶

## Abstract

Proximal policy optimization and trust region policy optimization (PPO and TRPO) with actor and critic parametrized by neural networks achieve significant empirical success in deep reinforcement learning. However, due to nonconvexity, the global convergence of PPO and TRPO remains less understood, which separates theory from practice. In this paper, we prove that a variant of PPO and TRPO equipped with overparametrized neural networks converges to the globally optimal policy at a sublinear rate. The key to our analysis is the global convergence of infinite-dimensional mirror descent under a notion of onepoint monotonicity, where the gradient and iterate are instantiated by neural networks. In particular, the desirable representation power and optimization geometry induced by the overparametrization of such neural networks allow them to accurately approximate the infinite-dimensional gradient and iterate.

## 1 Introduction

Policy optimization aims to find the optimal policy that maximizes the expected total reward through gradient-based updates. Coupled with neural networks, proximal policy optimization (PPO) (Schulman et al., 2017) and trust region policy optimization (TRPO) (Schulman

∗ equal contribution

† Northwestern University; boyiliu2018@u.northwestern.edu

‡ Northwestern University; qicai2022@u.northwestern.edu

§ Princeton University; zy6@princeton.edu

¶ Northwestern University; zhaoranwang@gmail.com

et al., 2015) are among the most important workhorses behind the empirical success of deep reinforcement learning across applications such as games (OpenAI, 2019) and robotics (Duan et al., 2016). However, the global convergence of policy optimization, including PPO and TRPO, remains less understood due to multiple sources of nonconvexity, including (i) the nonconvexity of the expected total reward over the infinite-dimensional policy space and (ii) the parametrization of both policy (actor) and action-value function (critic) using neural networks, which leads to nonconvexity in optimizing their parameters. As a result, PPO and TRPO are only guaranteed to monotonically improve the expected total reward over the infinite-dimensional policy space (Kakade, 2002; Kakade and Langford, 2002; Schulman et al., 2015, 2017), while the global optimality of the attained policy, the rate of convergence, as well as the impact of parametrizing policy and action-value function all remain unclear. Such a gap between theory and practice hinders us from better diagnosing the possible failure of deep reinforcement learning (Rajeswaran et al., 2017; Henderson et al., 2018; Ilyas et al., 2018) and applying it to critical domains such as healthcare (Ling et al., 2017) and autonomous driving (Sallab et al., 2017) in a more principled manner.

Closing such a theory-practice gap boils down to answering three key questions: (i) In the ideal case that allows for infinite-dimensional policy updates based on exact action-value functions, how do PPO and TRPO converge to the optimal policy? (ii) When the actionvalue function is parametrized by a neural network, how does temporal-difference learning (TD) (Sutton, 1988) converge to an approximate action-value function with sufficient accuracy within each iteration of PPO and TRPO? (iii) When the policy is parametrized by another neural network, based on the approximate action-value function attained by TD, how does stochastic gradient descent (SGD) converge to an improved policy that accurately approximates its ideal version within each iteration of PPO and TRPO? However, these questions largely elude the classical optimization framework, as questions (i)-(iii) involve nonconvexity, question (i) involves infinite-dimensionality, and question (ii) involves bias in stochastic (semi)gradients (Szepesv´ ari, 2010; Sutton and Barto, 2018). Moreover, the policy evaluation error arising from question (ii) compounds with the policy improvement error arising from question (iii), and they together propagate through the iterations of PPO and TRPO, making the convergence analysis even more challenging.

Contribution. By answering questions (i)-(iii), we establish the first nonasymptotic global

rate of convergence of a variant of PPO (and TRPO) equipped with neural networks. In detail, we prove that, with policy and action-value function parametrized by randomly initialized and overparametrized two-layer neural networks, PPO converges to the optimal policy at the rate of O (1 / √ K ), where K is the number of iterations. For solving the subproblems of policy evaluation and policy improvement within each iteration of PPO, we establish nonasymptotic upper bounds of the numbers of TD and SGD iterations, respectively. In particular, we prove that, to attain an /epsilon1 accuracy of policy evaluation and policy improvement, which appears in the constant of the O (1 / √ K ) rate of PPO, it suffices to take O (1 //epsilon1 2 ) TD and SGD iterations, respectively.

More specifically, to answer question (i), we cast the infinite-dimensional policy updates in the ideal case as mirror descent iterations. To circumvent the lack of convexity, we prove that the expected total reward satisfies a notation of one-point monotonicity (Facchinei and Pang, 2007), which ensures that the ideal policy sequence evolves towards the optimal policy. In particular, we show that, in the context of infinite-dimensional mirror descent, the exact action-value function plays the role of dual iterate, while the ideal policy plays the role of primal iterate (Nemirovski and Yudin, 1983; Nesterov, 2013; Puterman, 2014). Such a primal-dual perspective allows us to cast the policy evaluation error in question (ii) as the dual error and the policy improvement error in question (iii) as the primal error. More specifically, the dual and primal errors arise from using neural networks to approximate the exact action-value function and the ideal improved policy, respectively. To characterize such errors in questions (ii) and (iii), we unify the convergence analysis of TD for minimizing the mean squared Bellman error (MSBE) (Cai et al., 2019) and SGD for minimizing the mean squared error (MSE) (Jacot et al., 2018; Li and Liang, 2018; Chizat and Bach, 2018; Allen-Zhu et al., 2018; Zou et al., 2018; Cao and Gu, 2019a,b; Lee et al., 2019; Arora et al., 2019), both over neural networks. In particular, we show that the desirable representation power and optimization geometry induced by the overparametrization of neural networks enable the global convergence of both the MSBE and MSE, which correspond to the dual and primal errors, at a sublinear rate to zero. By incorporating such errors into the analysis of infinite-dimensional mirror descent, we establish the global rate of convergence of PPO. As a side product, the proof techniques developed here for handling nonconvexity, infinitedimensionality, semigradient bias, and overparametrization may be of independent interest to the analysis of more general deep reinforcement learning algorithms. In addition, it is

worth mentioning that, when the activation functions of neural networks are linear, our results cover the classical setting with linear function approximation, which encompasses the classical tabular setting as a special case.

More Related Work. PPO (Schulman et al., 2017) and TRPO (Schulman et al., 2015) are proposed to improve the convergence of vanilla policy gradient (Williams, 1992; Sutton et al., 2000) in deep reinforcement learning. Related algorithms based on the idea of KL-regularization include natural policy gradient and actor-critic (Kakade, 2002; Peters and Schaal, 2008), entropy-regularized policy gradient and actor-critic (Mnih et al., 2016), primal-dual actor-critic (Dai et al., 2017; Cho and Wang, 2017), soft Q-learning and actorcritic (Haarnoja et al., 2017, 2018), and dynamic policy programming (Azar et al., 2012). Despite its empirical success, policy optimization generally lacks global convergence guarantees due to nonconvexity. One exception is the recent analysis by Neu et al. (2017), which establishes the global convergence of TRPO to the optimal policy. However, Neu et al. (2017) require infinite-dimensional policy updates based on exact action-value functions and do not provide the nonasymptotic rate of convergence. In contrast, we allow for the parametrization of both policy and action-value function using neural networks and provide the nonasymptotic rate of PPO as well as the iteration complexity of solving the subproblems of policy improvement and policy evaluation. In particular, based on the primal-dual perspective of reinforcement learning (Puterman, 2014), we develop a concise convergence proof of PPO as infinite-dimensional mirror descent under one-point monotonicity, which is of independent interest. In addition, we refer to the closely related concurrent work (Agarwal et al., 2019) for the global convergence analysis of (natural) policy gradient for discrete state and action spaces as well as continuous state space with linear function approximation. See also the concurrent work (Zhang et al., 2019), which studies continuous state space with general function approximation, but only establishes the convergence to a locally optimal policy. In addition, in our companion paper (Wang et al., 2019), we establish the global convergence of neural (natural) policy gradient.

## 2 Background

In this section, we briefly introduce the general setting of reinforcement learning as well as PPO and TRPO.

Markov Decision Process. We consider the Markov decision process ( S , A , P , r, γ ), where S is a compact state space, A is a finite action space, P : S × S × A → R is the transition kernel, r : S × A → R is the reward function, and γ ∈ (0 , 1) is the discount factor. We track the performance of a policy π : A×S → R using its action-value function (Q-function) Q π : S × A → R , which is defined as

Correspondingly, the state-value function V π : S → R of a policy π is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The advantage function A π : S×A → R of a policy π is defined as A π ( s, a ) = Q π ( s, a ) -V π ( s ). We denote by ν π ( s ) and σ π ( s, a ) = π ( a | s ) · ν π ( s ) the stationary state distribution and the stationary state-action distribution associated with a policy π , respectively. Correspondingly, we denote by E σ π [ · ] and E ν π [ · ] the expectations E ( s,a ) ∼ σ π [ · ] = E a ∼ π ( · | s ) ,s ∼ ν π ( · ) [ · ] and E s ∼ ν π [ · ], respectively. Meanwhile, we denote by 〈· , ·〉 the inner product over A , e.g., we have V π ( s ) = E a ∼ π ( · | s ) [ Q π ( s, a )] = 〈 Q π ( s, · ) , π ( · | s ) 〉 .

PPO and TRPO. At the k -th iteration of PPO, the policy parameter θ is updated by

<!-- formula-not-decoded -->

where A k is an estimator of A π θ k and ̂ E [ · ] is taken with respect to the empirical version of σ π θ k , that is, the empirical stationary state-action distribution associated with the current policy π θ k . In practice, the penalty parameter β k is adjusted by line search.

At the k -th iteration of TRPO, the policy parameter θ is updated by

<!-- formula-not-decoded -->

where δ is the radius of the trust region. The PPO update in (2.2) can be viewed as a Lagrangian relaxation of the TRPO update in (2.3) with Lagrangian multiplier β k , which implies their updates are equivalent if β k is properly chosen. Without loss of generality, we focus on PPO hereafter.

It is worth mentioning that, compared with the original versions of PPO (Schulman et al., 2017) and TRPO (Schulman et al., 2015), the variants in (2.2) and (2.3) use KL( π θ ( · | s ) ‖ π θ k ( · | s )) instead of KL( π θ k ( · | s ) ‖ π θ ( · | s )). In Sections 3 and 4, we show that, as the original versions, such variants also allow us to approximately obtain the improved policy π θ k +1 using SGD, and moreover, enjoy global convergence.

## 3 Neural PPO

We present more details of PPO with policy and action-value function parametrized by neural networks. For notational simplicity, we denote by ν k and σ k the stationary state distribution ν π θ k and the stationary state-action distribution σ π θ k , respectively. Also, we define an auxiliary distribution σ k over S × A as σ k = ν k π 0 .

˜ ˜ Neural Network Parametrization. Without loss of generality, we assume that ( s, a ) ∈ R d for all s ∈ S and a ∈ A . We parametrize a function u : S × A → R , e.g., policy π or actionvalue function Q π , by the following two-layer neural network, which is denoted by NN( α ; m ),

<!-- formula-not-decoded -->

Here m is the width of the neural network, b i ∈ {-1 , 1 } ( i ∈ [ m ]) are the output weights, σ ( · ) is the rectified linear unit (ReLU) activation, and α = ([ α ] /latticetop 1 , . . . , [ α ] /latticetop m ) /latticetop ∈ R md with [ α ] i ∈ R d ( i ∈ [ m ]) are the input weights. We consider the random initialization

<!-- formula-not-decoded -->

We restrict the input weights α to an /lscript 2 -ball centered at the initialization α (0) by the projection Π B 0 ( R α ) ( α ′ ) = argmin α ∈B 0 ( R α ) {‖ α -α ′ ‖ 2 } , where B 0 ( R α ) = { α : ‖ α -α (0) ‖ 2 ≤ R α } . Throughout training, we only update α , while keeping b i ( i ∈ [ m ]) fixed at the initialization. Hence, we omit the dependency on b i ( i ∈ [ m ]) in NN( α ; m ) and u α ( s, a ).

Policy Improvement. We consider the population version of the objective function in (2.2),

<!-- formula-not-decoded -->

where Q ω k is an estimator of Q π θ k , that is, the exact action-value function of π θ k . In the following, we convert the subproblem max θ L ( θ ) of policy improvement into a least-squares subproblem. We consider the energy-based policy π ( a | s ) ∝ exp { τ -1 f ( s, a ) } , which is abbreviated as π ∝ exp { τ -1 f } . Here f : S × A → R is the energy function and τ &gt; 0 is the temperature parameter. We have the following closed form of the ideal infinite-dimensional policy update. See also, e.g., Abdolmaleki et al. (2018) for a Bayesian inference perspective.

Proposition 3.1. Let π θ k ∝ exp { τ -1 k f θ k } be an energy-based policy. Given an estimator Q ω k of Q π θ k , the update ̂ π k +1 ← argmax π { E ν k [ 〈 Q ω k ( s, · ) , π ( · | s ) 〉-β k · KL( π ( · | s ) ‖ π θ k ( · | s ))] } gives

<!-- formula-not-decoded -->

Proof. See Appendix C for a detailed proof.

Here we note that the closed form of ideal infinite-dimensional update in (3.4) holds statewise. To represent the ideal improved policy ̂ π k +1 in Proposition 3.1 using the energy-based policy π θ k +1 ∝ exp { τ -1 k +1 f θ k +1 } , we solve the subproblem of minimizing the MSE,

<!-- formula-not-decoded -->

which is justified in Appendix B as a majorization of -L ( θ ) defined in (3.3). Here we use the neural network parametrization f θ = NN( θ ; m f ) defined in (3.1), where θ denotes the input weights and m f is the width. It is worth mentioning that in (3.5) we sample the actions according to ˜ σ k so that π θ k +1 approximates the ideal infinite-dimensional policy update in (3.4) evenly well over all actions. Also note that the subproblem in (3.5) allows for off-policy sampling of both states and actions (Abdolmaleki et al., 2018).

To solve (3.5), we use the SGD update

<!-- formula-not-decoded -->

where ( s, a ) ∼ ˜ σ k and θ ( t +1) ← Π B 0 ( R f ) ( θ ( t +1 / 2)). Here η is the stepsize. See Appendix A for a detailed algorithm.

Policy Evaluation. To obtain the estimator Q ω k of Q π θ k in (3.3), we solve the subproblem of minimizing the MSBE,

<!-- formula-not-decoded -->

Here the Bellman evaluation operator T π of a policy π is defined as

<!-- formula-not-decoded -->

We use the neural network parametrization Q ω = NN( ω ; m Q ) defined in (3.1), where ω denotes the input weights and m Q is the width. To solve (3.7), we use the TD update

<!-- formula-not-decoded -->

where ( s, a ) ∼ σ k , s ′ ∼ P ( · | s, a ), a ′ ∼ π θ k ( · | s ′ ), and ω ( t +1) = Π B 0 ( R Q ) ( ω ( t +1 / 2)). Here η is the stepsize. See Appendix A for a detailed algorithm.

Neural PPO. By assembling the subproblems of policy improvement and policy evaluation, we present neural PPO in Algorithm 1, which is characterized in Section 4.

## Algorithm 1 Neural PPO

Require: MDP ( S , A , P , r, γ ), penalty parameter β , widths m f and m Q , number of SGD and TD iterations T , number of TRPO iterations K , and projection radii R f ≥ R Q

- 2: for k = 0 , . . . , K -1 do
- 1: Initialize with uniform policy: τ 0 ← 1, f θ 0 ← 0, π θ 0 ← π 0 ∝ exp { τ -1 0 f θ 0 }
- 3: Set temperature parameter τ k +1 ← β √ K/ ( k +1) and penalty parameter β k ← β √ K
- 5: Solve for Q ω k = NN( ω k ; m Q ) in (3.7) using the TD update in (3.8) (Algorithm 3)
- 4: Sample { ( s t , a t , a 0 t , s ′ t , a ′ t ) } T t =1 with ( s t , a t ) ∼ σ k , a 0 t ∼ π 0 ( · | s t ), s ′ t ∼ P ( · | s t , a t ) and a ′ t ∼ π θ k ( · | s ′ t )
- 6: Solve for f θ k +1 = NN( θ k +1 ; m f ) in (3.5) using the SGD update in (3.6) (Algorithm 2)
- 7: Update policy: π θ k +1 ∝ exp { τ -1 k +1 f θ k +1 }
- 8: end for

## 4 Main Results

In this section, we establish the global convergence of neural PPO in Algorithm 1 based on characterizing the errors arising from solving the subproblems of policy improvement and policy evaluation in (3.5) and (3.7), respectively.

Our analysis relies on the following regularity condition on the boundedness of reward.

Assumption 4.1 (Bounded Reward) . There exists a constant R max &gt; 0 such that R max = sup ( s,a ) ∈S×A | r ( s, a ) | , which implies | V π ( s ) | ≤ R max and | Q π ( s, a ) | ≤ R max for any policy π .

To ensure the compatibility between the policy and the action-value function (Konda and Tsitsiklis, 2000; Sutton et al., 2000; Kakade, 2002; Peters and Schaal, 2008; Wagner, 2011, 2013), we set m f = m Q and use the following random initialization. In Algorithm 1, we first generate according to (3.2) the random initialization α (0) = θ (0) = ω (0) and b i ( i ∈ [ m ]), and then use it as the fixed initialization of both SGD and TD in Lines 6 and 5 of Algorithm 1 for all k ∈ [ K ], respectively.

## 4.1 Errors of Policy Improvement and Policy Evaluation

We define the following function class, which characterizes the representation power of the neural network defined in (3.1).

Definition 4.2. For any constant R &gt; 0, we define the function class

<!-- formula-not-decoded -->

where [ α (0)] i and b i ( i ∈ [ m ]) are the random initialization defined in (3.2).

As m →∞ , F R,m -NN( α (0); m ) approximates a subset of the reproducing kernel Hilbert space (RKHS) induced by the kernel K ( x, y ) = E z ∼ N (0 ,I d /d ) [ 1 { z /latticetop x &gt; 0 , z /latticetop y &gt; 0 } x /latticetop y ] (Jacot et al., 2018; Li and Liang, 2018; Chizat and Bach, 2018; Allen-Zhu et al., 2018; Zou et al., 2018; Cao and Gu, 2019a,b; Lee et al., 2019; Arora et al., 2019; Cai et al., 2019). Such a subset is a ball with radius R in the corresponding H -norm, which is known to be a rich function class (Hofmann et al., 2008). Correspondingly, for a sufficiently large width m and radius R , F R,m is also a sufficiently rich function class.

Based on Definition 4.2, we lay out the following regularity condition on the action-value function class.

Assumption 4.3 (Action-Value Function Class) . It holds that Q π ( s, a ) ∈ F R Q ,m Q for any π .

Assumption 4.3 states that F R Q ,m Q is closed under the Bellman evaluation operator T π , as Q π is the fixed-point solution of the Bellman equation T π Q π = Q π . Such a regularity condition is commonly used in the literature (Munos and Szepesv´ ari, 2008; Antos et al., 2008; Farahmand et al., 2010, 2016; Tosatto et al., 2017; Yang et al., 2019). In particular, Yang and Wang (2019) define a class of Markov decision processes that satisfy such a regularity condition, which is sufficiently rich due to the representation power of F R Q ,m Q .

In the sequel, we lay out another regularity condition on the stationary state-action distribution σ π .

Assumption 4.4 (Regularity of Stationary Distribution) . There exists a constant c &gt; 0 such that for any vector z ∈ R d and ζ &gt; 0, it holds almost surely that E σ π [ 1 {| z /latticetop ( s, a ) | ≤ ζ } | z ] ≤ c · ζ/ ‖ z ‖ 2 for any π .

Assumption 4.4 states that the density of σ π is sufficiently regular. Such a regularity condition holds as long as the stationary state distribution ν π has upper bounded density.

We are now ready present bounds for errors induced by approximation via two-layer neural networks, with analysis generalizing those of Cai et al. (2019); Arora et al. (2019) included in Appendix D. First, we characterize the policy improvement error, which is induced by solving the subproblem in (3.5) using the SGD update in (3.6), in the following theorem. See Line 6 of Algorithm 1 and Algorithm 2 for a detailed algorithm.

Theorem 4.5 (Policy Improvement Error) . Suppose that Assumptions 4.1, 4.3, and 4.4 hold. We set T ≥ 64 and the stepsize to be η = T -1 / 2 . Within the k -th iteration of Algorithm 1, the output f θ of Algorithm 2 satisfies

<!-- formula-not-decoded -->

Proof. See Appendix D for a detailed proof.

Similarly, we characterize the policy evaluation error, which is induced by solving the subproblem in (3.7) using the TD update in (3.8), in the following theorem. See Line 5 of Algorithm 1 and Algorithm 3 for a detailed algorithm.

Theorem 4.6 (Policy Evaluation Error) . Suppose that Assumptions 4.1, 4.3, and 4.4 hold. We set T ≥ 64 / (1 -γ ) 2 and the stepsize to be η = T -1 / 2 . Within the k -th iteration of Algorithm 1, the output Q ω of Algorithm 3 satisfies

<!-- formula-not-decoded -->

Proof. See Appendix D for a detailed proof.

As we show in Sections 4.3 and 5, Theorems 4.5 and 4.6 characterize the primal and dual errors of the infinite-dimensional mirror descent corresponding to neural PPO. In particular, such errors decay to zero at the rate of 1 / √ T when the width m f = m Q is sufficiently large, where T is the number of TD and SGD iterations in Algorithm 1. For notational simplicity, we omit the dependency on the random initialization in the expectations hereafter.

## 4.2 Error Propagation

We denote by π ∗ the optimal policy with ν ∗ being its stationary state distribution and σ ∗ being its stationary state-action distribution. Recall that, as defined in (3.4), ̂ π k +1 is the ideal improved policy based on Q ω k , which is an estimator of the exact action-value function Q π θ k . Correspondingly, we define the ideal improved policy based on Q π θ k as

<!-- formula-not-decoded -->

By the same proof of Proposition 3.1, we have π k +1 ∝ exp { β -1 k Q π θ k + τ -1 k f θ k } , which is also an energy-based policy.

Let σ ∗ k = π θ k ν ∗ . We define the following quantities related to density ratios between policies or stationary distributions,

<!-- formula-not-decoded -->

where d ν ∗ / d ν k , d σ ∗ / d ˜ σ k , d σ ∗ k / d ˜ σ k , and d σ ∗ / d σ k are the Radon-Nikodym derivatives. A closely related quantity known as the concentrability coefficient is commonly used in the

literature (Munos and Szepesv´ ari, 2008; Antos et al., 2008; Farahmand et al., 2010; Tosatto et al., 2017; Yang et al., 2019). In comparison, as our analysis is based on stationary distributions, our definitions of ϕ ∗ k , φ ∗ k and ψ ∗ k are simpler in that they do not require unrolling the state-action sequence. Then we have the following lemma that quantifies how the errors of policy improvement and policy evaluation propagate into the infinite-dimensional policy space.

Lemma 4.7 (Error Propagation) . Suppose that the policy improvement error in Line 6 of Algorithm 1 satisfies

<!-- formula-not-decoded -->

and the policy evaluation error in Line 5 of Algorithm 1 satisfies

<!-- formula-not-decoded -->

For π k +1 defined in (4.1) and π θ k +1 obtained in Line 7 of Algorithm 1, we have

<!-- formula-not-decoded -->

where ε k = τ -1 k +1 /epsilon1 k +1 · φ ∗ k +1 + β -1 k /epsilon1 ′ k · ψ ∗ k .

Proof. See Appendix E.1 for a detailed proof.

Lemma 4.7 quantifies the difference between the ideal case, where we use the infinitedimensional policy update based on the exact action-value function, and the realistic case, where we use the neural networks defined in (3.1) to approximate the exact action-value function and the ideal improved policy.

Note that we have ‖ π ( · | s ) ‖ 1 = 1 and ‖ Q π ( s, · ) ‖ ∞ ≤ R max for any policy π and any s ∈ S . It is natural that we equip (i) the primal iterate, which is the policy, with the /lscript 1 -norm, and (ii) the dual iterate, which is the action-value function, with the /lscript ∞ -norm. We give the following lemma that characterizes the policy improvement error with respect to the /lscript ∞ -norm under the optimal stationary state distribution ν ∗ .

Lemma 4.8 (Policy Improvement /lscript ∞ -Error) . Under the same conditions of Lemma 4.7, we have

<!-- formula-not-decoded -->

where ε ′ k = 2 |A| · τ k +1 /epsilon1 k +1 · ϕ ∗ k

<!-- formula-not-decoded -->

Proof. See Appendix E.2 for a detailed proof.

The following lemma characterizes the energy increment.

Lemma 4.9 (Stepwise Energy Increment) . Under the same conditions of Lemma 4.7, we have

<!-- formula-not-decoded -->

where M = 2 E ν ∗ [max a ∈A ( Q ω 0 ( s, a )) 2 ] + 2 R 2 f .

Proof. See Appendix E.3 for a detailed proof.

Intuitively, due to the KL-regularization in (3.3), the bound of β -1 k Q ω k ( s, · ) quantified in Lemma 4.9 keeps the updated policy π θ k +1 from being too far away from the current policy π θ k .

Lemmas 4.7-4.9 play key roles in establishing the global convergence of neural PPO.

## 4.3 Global Convergence of Neural PPO

We track the progress of neural PPO in Algorithm 1 using the expected total reward

<!-- formula-not-decoded -->

where ν ∗ is the stationary state distribution of the optimal policy π ∗ . The following theorem characterizes the global convergence of L ( π θ k ) towards L ( π ∗ ). Recall that T f and T Q are the numbers of SGD and TD iterations in Lines 6 and 5 of Algorithm 1, while φ ∗ k and ψ ∗ k are defined in (4.2).

Theorem 4.10 (Global Rate of Convergence of Neural PPO) . Suppose that Assumptions 4.1, 4.3, and 4.4 hold. For the policy sequence { π θ k } K k =1 attained by neural PPO in Algorithm 1, we have

<!-- formula-not-decoded -->

Here M = 2 E ν ∗ [max a ∈A ( Q ω 0 ( s, a )) 2 ] + 2 R 2 f , ε k = τ -1 k +1 /epsilon1 k +1 · φ ∗ k + β -1 k /epsilon1 ′ k · ψ ∗ k , and ε ′ k = 2 |A| · τ -1 k +1 /epsilon1 k +1 · ϕ ∗ k , where

<!-- formula-not-decoded -->

Proof. See Section 5 for a detailed proof of Theorem 4.10. The key to our proof is the global convergence of infinite-dimensional mirror descent with errors under one-point monotonicity, where the primal and dual errors are characterized by Theorems 4.5 and 4.6, respectively.

To understand Theorem 4.10, we consider the infinite-dimensional policy update based on the exact action-value function, that is, /epsilon1 k +1 = /epsilon1 ′ k = 0 for any k + 1 ∈ [ K ]. In such an ideal case, by Theorem 4.10, neural PPO globally converges to the optimal policy π ∗ at the rate of with the optimal choice of the penalty parameter β k = √ MK/ log |A| .

<!-- formula-not-decoded -->

Note that Theorem 4.10 sheds light on the difficulty of choosing the optimal penalty coefficient in practice, which is observed by Schulman et al. (2017). In particular, the optimal choice of β in β k = β √ K is given by

<!-- formula-not-decoded -->

where M and ∑ K -1 k =0 ( ε k + ε ′ k ) may vary across different deep reinforcement learning problems. As a result, line search is often needed in practice.

To better understand Theorem 4.10, the following corollary quantifies the minimum width m f and m Q and the minimum number of SGD and TD iterations T that ensure the O (1 / √ K ) rate of convergence.

Corollary 4.11 (Iteration Complexity of Subproblems and Minimum Widths of Neural Networks) . Suppose that Assumptions 4.1, 4.3, and 4.4 hold. Let m f = R 10 f · Ω( K 18 · φ ∗ k 8 + K 8 · |A| ), m Q = Ω ( K 4 R 10 Q · ψ ∗ k 4 ) and T = Ω( K 4 R 4 f · ϕ ∗ k 4 + K 6 R 4 f · φ ∗ k 4 + K 2 R 4 Q · ψ ∗ k 4 ). We have

<!-- formula-not-decoded -->

Proof. See Appendix F for a detailed proof.

The difference between the requirements on the widths m f and m Q in Corollary 4.11 suggests that the errors of policy improvement and policy evaluation play distinct roles in the global convergence of neural PPO. In fact, Theorem 4.10 depends on the total error

τ -1 k +1 /epsilon1 k +1 · φ ∗ k + β -1 k /epsilon1 ′ k · ψ ∗ k + |A|· τ -2 k +1 /epsilon1 2 k +1 , where the weight τ -1 k +1 of the policy improvement error /epsilon1 k +1 is much larger than the weight β -1 k of the policy evaluation error /epsilon1 ′ k , and |A| · τ -2 k +1 /epsilon1 2 k +1 is a high-order term when /epsilon1 k +1 is sufficiently small. In other words, the policy improvement error plays a more important role.

## 5 Proof Sketch

In this section, we sketch the proof of Theorem 4.10. In detail, we cast neural PPO in Algorithm 1 as infinite-dimensional mirror descent with primal and dual errors and exploit a notion of one-point monotonicity to establish its global convergence.

We first present the performance difference lemma of Kakade and Langford (2002). Recall that the expected total reward L ( π ) is defined in (4.7) and ν ∗ is the stationary state distribution of the optimal policy π ∗ .

Lemma 5.1 (Performance Difference) . For L ( π ) defined in (4.7), we have

<!-- formula-not-decoded -->

Proof. See Appendix G for a detailed proof.

Since the optimal policy π ∗ maximizes the value function V π ( s ) with respect to π for any s ∈ S , we have L ( π ∗ ) = E ν ∗ [ V π ∗ ( s )] ≥ E ν ∗ [ V π ( s )] = L ( π ) for any π . As a result, we have

<!-- formula-not-decoded -->

Under the variational inequality framework (Facchinei and Pang, 2007), (5.1) corresponds to the monotonicity of the mapping Q π evaluated at π ∗ and any π . Note that the classical notion of monotonicity requires the evaluation at any pair π ′ and π , while we restrict π ′ to π ∗ in (5.1). Hence, we refer to (5.1) as one-point monotonicity. In the context of nonconvex optimization, the mapping Q π can be viewed as the gradient of L ( π ) at π , which lives in the dual space, while π lives in the primal space. Another condition related to (5.1) in nonconvex optimization is known as dissipativity (Zhou et al., 2019).

The following lemma establishes the one-step descent of the KL-divergence in the infinitedimensional policy space, which follows from the analysis of mirror descent (Nemirovski and Yudin, 1983; Nesterov, 2013) as well as the fact that given any ν k , the subproblem of policy improvement in (4.1) can be solved for each s ∈ S individually.

Lemma 5.2 (One-Step Descent) . For the ideal improved policy π k +1 defined in (4.1) and the current policy π θ k , we have that, for any s ∈ S ,

<!-- formula-not-decoded -->

Proof. See Appendix G for a detailed proof.

Based on Lemmas 5.1 and 5.2, we prove Theorem 4.10 by casting neural PPO as infinitedimensional mirror descent with primal and dual errors, whose impact is characterized in Lemma 4.7. In particular, we employ the /lscript 1 -/lscript ∞ pair of primal-dual norms.

Proof of Theorem 4.10. Taking expectation with respect to s ∼ ν ∗ and invoking Lemmas 4.7 and 5.2, we have

<!-- formula-not-decoded -->

where the second inequality follows from Lemma 5.1. By the H¨ older's inequality, we have

<!-- formula-not-decoded -->

where in the third inequality we use the fact that ‖ π θ k ( · | s ) -π θ k +1 ( · | s ) ‖ 1 ≤ ‖ π θ k ( · | s ) ‖ 1 + ‖ π θ k +1 ( · | s ) ‖ 1 = 2 and in the last inequality we use Lemma 4.8. Plugging (5.3) into (5.2), we

further have

<!-- formula-not-decoded -->

where in the second inequality we use 2 xy -y 2 ≤ x 2 and in the last inequality we use Lemma 4.9. Rearranging the terms in (5.4), we have

<!-- formula-not-decoded -->

Telescoping (5.5) for k +1 ∈ [ K ], we obtain

<!-- formula-not-decoded -->

Note that we have (i) ∑ K -1 k =0 β -1 k · ( L ( π ∗ ) -L ( π θ k )) ≥ ( ∑ K -1 k =0 β -1 k ) · min 0 ≤ k ≤ K {L ( π ∗ ) -L ( π θ k ) } , (ii) E ν ∗ [KL( π ∗ ( · | s ) ‖ π θ 0 ( · | s ))] ≤ log |A| due to the uniform initialization of policy, and that (iii) the KL-divergence is nonnegative. Hence, we have

<!-- formula-not-decoded -->

Setting the penalty parameter β k = β √ K , we have ∑ K -1 k =0 β -1 k = β -1 √ K and ∑ K -1 k =0 β -2 k = β -2 , which together with (5.6) concludes the proof of Theorem 4.10.

## Acknowledgement

The authors thank Jason D. Lee, Chi Jin, and Yu Bai for enlightening discussions throughout this project.

## References

- Abdolmaleki, A., Springenberg, J. T., Tassa, Y., Munos, R., Heess, N. and Riedmiller, M. (2018). Maximum a posteriori policy optimisation. arXiv preprint arXiv:1806.06920 .
- Agarwal, A., Kakade, S. M., Lee, J. D. and Mahajan, G. (2019). Optimality and approximation with policy gradient methods in Markov decision processes. arXiv preprint arXiv:1908.00261 .
- Allen-Zhu, Z., Li, Y. and Liang, Y. (2018). Learning and generalization in overparameterized neural networks, going beyond two layers. arXiv preprint arXiv:1811.04918 .
- Antos, A., Szepesv´ ari, C. and Munos, R. (2008). Fitted Q-iteration in continuous actionspace MDPs. In Advances in Neural Information Processing Systems .
- Arora, S., Du, S. S., Hu, W., Li, Z. and Wang, R. (2019). Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. arXiv preprint arXiv:1901.08584 .
- Azar, M. G., G´ omez, V. and Kappen, H. J. (2012). Dynamic policy programming. Journal of Machine Learning Research , 13 3207-3245.
- Cai, Q., Yang, Z., Lee, J. D. and Wang, Z. (2019). Neural temporal-difference learning converges to global optima. arXiv preprint arXiv:1905.10027 .
- Cao, Y. and Gu, Q. (2019a). Generalization bounds of stochastic gradient descent for wide and deep neural networks. arXiv preprint arXiv:1905.13210 .
- Cao, Y. and Gu, Q. (2019b). A generalization theory of gradient descent for learning overparameterized deep ReLU networks. arXiv preprint arXiv:1902.01384 .
- Chizat, L. and Bach, F. (2018). A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956 .
- Cho, W. S. and Wang, M. (2017). Deep primal-dual reinforcement learning: Accelerating actor-critic using Bellman duality. arXiv preprint arXiv:1712.02467 .

- Dai, B., Shaw, A., Li, L., Xiao, L., He, N., Liu, Z., Chen, J. and Song, L. (2017). SBEED: Convergent reinforcement learning with nonlinear function approximation. arXiv preprint arXiv:1712.10285 .
- Duan, Y., Chen, X., Houthooft, R., Schulman, J. and Abbeel, P. (2016). Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning .
- Facchinei, F. and Pang, J.-S. (2007). Finite-Dimensional Variational Inequalities and Complementarity Problems . Springer Science &amp; Business Media.
- Farahmand, A.-m., Ghavamzadeh, M., Szepesv´ ari, C. and Mannor, S. (2016). Regularized policy iteration with nonparametric function spaces. Journal of Machine Learning Research , 17 4809-4874.
- Farahmand, A.-m., Szepesv´ ari, C. and Munos, R. (2010). Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems .
- Haarnoja, T., Tang, H., Abbeel, P. and Levine, S. (2017). Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning .
- Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290 .
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. and Meger, D. (2018). Deep reinforcement learning that matters. In AAAI Conference on Artificial Intelligence .
- Hofmann, T., Sch¨ olkopf, B. and Smola, A. J. (2008). Kernel methods in machine learning. Annals of Statistics 1171-1220.
- Ilyas, A., Engstrom, L., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L. and Madry, A. (2018). Are deep policy gradient algorithms truly policy gradient algorithms? arXiv preprint arXiv:1811.02553 .
- Jacot, A., Gabriel, F. and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems .

- Kakade, S. (2002). A natural policy gradient. In Advances in Neural Information Processing Systems .
- Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning .
- Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in Neural Information Processing Systems .
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Sohl-Dickstein, J. and Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. arXiv preprint arXiv:1902.06720 .
- Li, Y. and Liang, Y. (2018). Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in Neural Information Processing Systems .
- Ling, Y., Hasan, S. A., Datla, V., Qadir, A., Lee, K., Liu, J. and Farri, O. (2017). Diagnostic inferencing via improving clinical concept extraction with deep reinforcement learning: A preliminary study. In Machine Learning for Healthcare Conference .
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning .
- Munos, R. and Szepesv´ ari, C. (2008). Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9 815-857.
- Nemirovski, A. S. and Yudin, D. B. (1983). Problem Complexity and Method Efficiency in Optimization . Springer.
- Nesterov, Y. (2013). Introductory Lectures on Convex Optimization: A Basic Course , vol. 87. Springer Science &amp; Business Media.
- Neu, G., Jonsson, A. and G´ omez, V. (2017). A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 .
- OpenAI (2019). OpenAI Five. https://openai.com/five/ .

- Peters, J. and Schaal, S. (2008). Natural actor-critic. Neurocomputing , 71 1180-1190.
- Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons.
- Rajeswaran, A., Lowrey, K., Todorov, E. V. and Kakade, S. M. (2017). Towards generalization and simplicity in continuous control. In Advances in Neural Information Processing Systems .
- Sallab, A. E., Abdou, M., Perot, E. and Yogamani, S. (2017). Deep reinforcement learning framework for autonomous driving. Electronic Imaging , 2017 70-76.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning .
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine Learning , 3 9-44.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction . MIT press.
- Sutton, R. S., McAllester, D. A., Singh, S. P. and Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems .
- Szepesv´ ari, C. (2010). Algorithms for reinforcement learning. Synthesis Lectures on Artificial Intelligence and Machine Learning , 4 1-103.
- Tosatto, S., Pirotta, M., D'Eramo, C. and Restelli, M. (2017). Boosted fitted Q-iteration. In International Conference on Machine Learning .
- Wagner, P. (2011). A reinterpretation of the policy oscillation phenomenon in approximate policy iteration. In Advances in Neural Information Processing Systems .

- Wagner, P. (2013). Optimistic policy iteration and natural actor-critic: A unifying view and a non-optimality result. In Advances in Neural Information Processing Systems .
- Wang, L., Cai, Q., Yang, Z. and Wang, Z. (2019). Neural policy gradient methods: Global optimality and rates of convergence. arXiv preprint arXiv:1909.01150 .
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8 229-256.
- Yang, L. F. and Wang, M. (2019). Sample-optimal parametric Q-learning with linear transition models. arXiv preprint arXiv:1902.04779 .
- Yang, Z., Xie, Y. and Wang, Z. (2019). A theoretical analysis of deep Q-learning. arXiv preprint arXiv:1901.00137 .
- Zhang, K., Koppel, A., Zhu, H. and Ba¸ sar, T. (2019). Global convergence of policy gradient methods to (almost) locally optimal policies. arXiv preprint arXiv:1906.08383 .
- Zhou, M., Liu, T., Li, Y., Lin, D., Zhou, E. and Zhao, T. (2019). Toward understanding the importance of noise in training neural networks. In International Conference on Machine Learning .
- Zou, D., Cao, Y., Zhou, D. and Gu, Q. (2018). Stochastic gradient descent optimizes overparameterized deep ReLU networks. arXiv preprint arXiv:1811.08888 .

## A Algorithms in Section 3

We present the algorithms for solving the subproblems of policy improvement and policy evaluation in Section 3.

## Algorithm 2 Policy Improvement via SGD

- 1: Require: MDP ( S , A , P , r, γ ), current energy function f θ k , initial weights b i , [ θ (0)] i ( i [ m ]), number of iterations T , sample ( s , a 0 ) T

<!-- formula-not-decoded -->

## Algorithm 3 Policy Evaluation via TD

- 1: Require: MDP ( S , A , P , r, γ ), initial weights b i , [ ω (0)] i ( i ∈ [ m Q ]), number of iterations T , sample { ( s t , a t , s ′ t , a ′ t ) } T t =1 2: Set stepsize η ← T -1 / 2 3: for t = 0 , . . . , T -1 do 4: ( s, a, s ′ , a ′ ) ← ( s t +1 , a t +1 , s ′ t +1 , a ′ t +1 ) 5: ω ( t +1 / 2) ← ω ( t ) -η · ( Q ω ( t ) ( s, a ) -(1 -γ ) · r ( s, a ) -γQ ω ( t ) ( s ′ , a ′ ) ) · ∇ ω Q ω ( t ) ( s, a ) 6: ω ( t +1) ← argmin ω ∈B 0 ( R Q ) { ‖ ω -ω ( t +1 / 2) ‖ 2 } 7: end for 8: Average over path ω ← 1 /T · ∑ T -1 t =0 ω ( t ) 9: Output: Q ω

## B Supplementary Lemma in Section 3

The following lemma quantifies the policy improvement error in terms of the distance between polices, which is induced by solving (3.5).

Lemma B.1. Suppose that π θ k +1 ∝ exp { τ -1 k +1 f θ k +1 } satisfies

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

where ̂ π k +1 is defined in (3.4). Proof. Let τ -1 k +1 ̂ f k +1 = β -1 k Q ω k + τ -1 k f θ k . Since an energy-based policy π ∝ exp { τ -1 f } is continuous with respect to f , by the mean value theorem, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ f is a function determined by f θ k +1 and ̂ f k +1 . Furthermore, we have

Therefore, we obtain

<!-- formula-not-decoded -->

Taking expectation E ˜ σ k [ · ] on the both sides of (B.1), we finally obtain

<!-- formula-not-decoded -->

which concludes the proof of Lemma B.1.

Lemma B.1 ensures that if the policy improvement error /epsilon1 k +1 is small, then the corresponding improved policy π θ k +1 is close to the ideal improved policy ̂ π k +1 , which justifies solving the subproblem in (3.5) for policy improvement.

## C Proof of Proposition 3.1

Proof. The subproblem of policy improvement for solving ̂ π k +1 takes the form

<!-- formula-not-decoded -->

The Lagrangian of the above maximization problem takes the form

<!-- formula-not-decoded -->

Plugging in π θ k ( s, a ) = exp { τ -1 k f θ k ( s, a ) } / ∑ a ′ ∈A exp { τ -1 k f θ k ( s, a ′ ) } , we obtain the optimality condition

<!-- formula-not-decoded -->

for any a ∈ A and s ∈ S . Note that log( ∑ a ′ ∈A exp { τ -1 k f θ k ( s, a ′ ) } ) is determined by the state s only. Hence, we have ̂ π k +1 ( a | s ) ∝ exp { β -1 k Q ω k ( s, a ) + τ -1 k f θ k ( s, a ) } for any a ∈ A and s ∈ S , which concludes the proof of Proposition 3.1.

## D Proofs for Section 4.1

The proofs in this section generalizes those of Cai et al. (2019); Arora et al. (2019) under a unified framework, which accounts for both SGD, and TD, which uses stochastic semigradient. In particular, we develop a unified global convergence analysis of a meta-algorithm with the following update,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where µ ∈ [0 , 1) is a constant, ( s, a, s ′ , a ′ ) is sampled from a stationary distribution ρ , and u α is parametrized by the two-layer neural network NN( α ; m ) defined in (3.1). The random initialization of u α is given in (3.2). We denote by E init [ · ] the expectation over such random initialization and E ρ [ · ] the expectation over ( s, a ) conditional on the random initialization.

Such a meta-algorithm recovers SGD for policy improvement in (3.5) when we set ρ = ˜ σ k , u α = f θ , v = τ k +1 · ( β -1 k Q ω k + τ -1 k f θ k ), µ = 0, and R u = R f , and recovers TD for policy evaluation in (3.8) when we set ρ = σ k , u α = Q ω , v = (1 -γ ) · r , µ = γ , and R u = R Q .

To unify our analysis for SGD and TD, we assume that v in (D.1) satisfies

<!-- formula-not-decoded -->

for constants v 1 , v 2 , v 3 ≥ 0. Also, without loss of generality, we assume that ‖ ( s, a ) ‖ 2 ≤ 1 for any s ∈ S and a ∈ A . In Section D.2, we set v 1 = 4, v 2 = 4, and v 3 = 0 for SGD, and v 1 = 0, v 2 = 0, and v 3 = R max for TD, respectively.

For notational simplicity, we define the residual δ α ( s, a, s ′ , a ′ ) = u α ( s, a ) -v ( s, a ) -µ · u α ( s ′ , a ′ ). We denote by

<!-- formula-not-decoded -->

the stochastic update vector at the t -th iteration and its population mean, respectively. For SGD, g α ( t ) ( s, a, s ′ , a ′ ) corresponds to the stochastic gradient, while for TD, g α ( t ) ( s, a, s ′ , a ′ ) corresponds to the stochastic semigradient.

Note that the gradient of u α ( s, a ) with respect to α takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, u α ( s, a ) is 1-Lipschitz continuous with respect to α .

In the following, we first show in Section D.1 that the overparametrization of u α ensures that it behaves similarly as its local linearization at the random initialization α (0) defined in (3.2). Then in Section D.2, we establish the global convergence of the meta-algorithm defined in (D.1) and (D.2), which implies the global convergence of SGD and TD.

## D.1 Local Linearization

In this section, we first define a local linearization of the two-layer neural network u α at its random initialization and then characterize the error induced by local linearization. We

define implies

<!-- formula-not-decoded -->

Next, applying the inequality 1 {| z | ≤ y }| z | ≤ 1 {| z | ≤ y } y to the right-hand side of (D.6), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The linearity of u 0 α with respect to α yields

<!-- formula-not-decoded -->

The following lemma characterizes how far u 0 α ( t ) deviates from u α ( t ) for α ( t ) ∈ B 0 ( R u ).

Lemma D.1. For any α ′ ∈ B 0 ( R u ), we have

<!-- formula-not-decoded -->

Proof. By the definition of u α in (3.1), we have

<!-- formula-not-decoded -->

where the second inequality follows from | b i | = 1 and the fact that

/negationslash

<!-- formula-not-decoded -->

Further applying the Cauchy-Schwarz inequality to (D.7) and invoking the upper bound ‖ α ′ -α (0) ‖ 2 ≤ R u , we obtain

<!-- formula-not-decoded -->

Taking expectation on the both sides and invoking Assumption 4.4, we obtain

<!-- formula-not-decoded -->

By the Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

where the second inequality follows from ∑ m i =1 ‖ [ α ′ ] i -[ α (0)] i ‖ 2 2 = ‖ α ′ -α (0) ‖ 2 2 ≤ R 2 u . Therefore, we have that the right-hand side of (D.9) is O ( R 3 u m -1 / 2 ). Thus, we obtain

<!-- formula-not-decoded -->

which concludes the proof of Lemma D.1.

Corresponding to u 0 α defined in (D.4), let δ 0 α ( s, a, s ′ , a ′ ) = u 0 α ( s, a ) -v ( s, a ) -µ · u 0 α ( s ′ , a ′ ). We define the local linearization of ¯ g α ( t ) , which is defined in (D.3), as

<!-- formula-not-decoded -->

The following lemma characterizes the difference between ¯ g 0 α ( t ) and ¯ g α ( t ) .

Lemma D.2. For any t ∈ [ T ], we have

<!-- formula-not-decoded -->

Proof. By the definition of ¯ g 0 α ( t ) and ¯ g α ( t ) in (D.10) and (D.3), we have

<!-- formula-not-decoded -->

Upper Bounding (i): We have ‖∇ α u α ( t ) ( s, a ) ‖ 2 ≤ 1 as ‖ ( s, a ) ‖ 2 ≤ 1. Note that the difference between δ α ( t ) and δ 0 α ( t ) takes the form

<!-- formula-not-decoded -->

Taking expectation on the both sides, we obtain

<!-- formula-not-decoded -->

where the equality follows from | µ | ≤ 1 and the fact that ( s, a ) and ( s ′ , a ′ ) have the same marginal distribution. Thus, by Lemma D.1, we have that (i) in (D.11) is O ( R 3 u m -1 / 2 ).

Upper Bounding (ii): First, by the H¨ older's inequality, we have

<!-- formula-not-decoded -->

We use | u 0 α ( t ) ( s, a ) -u 0 α (0) ( s, a ) | ≤ ‖ α ( t ) -α (0) ‖ 2 ≤ R u to obtain

<!-- formula-not-decoded -->

Next we characterize ‖∇ α u α ( t ) ( s, a ) -∇ α u 0 α ( t ) ( s, a ) ‖ 2 in (ii). Recall that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

where the inequality follows from the same arguments used to derive (D.6). Plugging (D.12) and (D.13) into (ii) and recalling that

<!-- formula-not-decoded -->

we find that it remains to upper bound the following two terms

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We already show in the proof of Lemma D.1 that (D.14) is O ( R u m -1 / 2 ). We characterize (D.15) in the following. For the random initialization of u α ( s, a ) in (3.2), we have

/negationslash

<!-- formula-not-decoded -->

plugging which into (D.15) gives

<!-- formula-not-decoded -->

/negationslash where we use the same arguments applied to (D.8) in the proof of Lemma D.1. Note that b i , b j are independent of α (0), E init [ b i b j ] = 0, and ∑ m i =1 ‖ [ α ( t )] i -[ α (0)] i ‖ 2 2 = ‖ α ( t ) -α (0) ‖ 2 2 ≤ R 2 u . We further obtain

<!-- formula-not-decoded -->

Finally, by the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

whose right-hand side is O ( m 3 / 2 ). Thus, we obtain that (D.15) is O ( R u m -1 / 2 ) and (ii) in (D.11) is O ( R 3 u m -1 / 2 ), which concludes the proof of Lemma D.2.

## D.2 Global Convergence

In this section, we establish the global convergence of the meta-algorithm defined in (D.1) and (D.2). We first present the following lemma for characterizing the variance of the stochastic update vector g α ( t ) ( s, a, s ′ , a ′ ) defined in (D.3), which later allows us to focus on tracking its mean in the global convergence analysis.

Lemma D.3 (Variance of the Stochastic Update Vector) . There exists a constant ξ 2 g = O ( R 2 u ) independent of t , such that for any t ≤ T , it holds that

<!-- formula-not-decoded -->

Proof. Since we have

<!-- formula-not-decoded -->

it suffices to prove that E [ ‖ g α ( t ) ( s, a, s ′ , a ′ ) ‖ 2 2 ] = O ( R 2 u ). By the definition of E ρ [ ‖ g α ( t ) ( s, a, s ′ , a ′ ) ‖ 2 2 ] in (D.3), using ‖∇ α ( t ) u α ( t ) ( s, a ) ‖ 2 2 ≤ 1, we obtain

<!-- formula-not-decoded -->

Then, by similar arguments used in the derivation of (D.12), we obtain

<!-- formula-not-decoded -->

Note that by ‖ ( s, a ) ‖ 2 ≤ 1, we have

<!-- formula-not-decoded -->

which together with (D.16) and (D.17) implies E init ,ρ [ ‖ g α ( t ) ( s, a, s , a ) ‖ 2 ] = O ( R u ). Thus, we complete the proof of Lemma D.3.

≤ ‖ ‖ ′ ′ 2 2

Before presenting the global convergence result of the meta-algorithm defined in (D.1), we first define u 0 α ∗ , which later become the exact learning target of the meta-algorithm defined in (D.1) and (D.2). In specific, we define the approximate stationary point as α ∗ ∈ B 0 ( R u ) such that

<!-- formula-not-decoded -->

which is equivalent to the condition

<!-- formula-not-decoded -->

Then we establish the uniqueness and existence of u 0 α ∗ with α ∗ defined in D.18. We first define the operator

<!-- formula-not-decoded -->

Then using the definition of T in (D.20) and plugging the definition of ¯ g 0 α ∗ in (D.4) into (D.19), we obtain

<!-- formula-not-decoded -->

which is equivalent to u 0 α ∗ = Π F B,m T u 0 α ∗ . Here the projection Π F B,m is defined with respect to the /lscript 2 -distance under measure ρ . Finally, as we have the following contraction inequality

<!-- formula-not-decoded -->

we know that such fixed-point solution u 0 α ∗ uniquely exists.

Now, with a well-defined learning target u 0 α ∗ , we are ready to prove the the global convergence of the meta-algorithm defined in (D.1) and (D.2) with two-layer neural network approximation.

Theorem D.4. Suppose that we run T ≥ 64 / (1 -µ ) 2 iterations of the meta-algorithm defined in (D.1) and (D.2). Setting the stepsize η = T -1 / 2 , we have

<!-- formula-not-decoded -->

where α = 1 /T · ∑ T -1 t =0 α ( t ) and α ∗ is the approximate stationary point defined in (D.18).

Proof. The proof of the theorem consists of two parts. We first analyze the progress of each step. Then based on such one-step analysis, we establish the error bound of the approximation via two-layer neural network u α .

One-Step Analysis: For any t &lt; T , using the stationarity condition in (D.18) and the convexity of B 0 ( R u ), we obtain

<!-- formula-not-decoded -->

In the following, we upper bound the last two terms in (D.21). First, to upper bound E ρ [ ‖ g α ( t ) ( s, a, s ′ , a ′ ) -¯ g 0 α ∗ ‖ 2 2 | α ( t )], by the Cauchy-Schwarz inequality we have

<!-- formula-not-decoded -->

where the total expectation on the first two terms on the right-hand side are characterized in Lemmas D.3 and D.2, respectively. To characterize ‖ ¯ g 0 α ( t ) -¯ g 0 α ∗ ‖ 2 2 , again using ‖ ( s, a ) ‖ 2 ≤ 1, we have

<!-- formula-not-decoded -->

For the right-hand side of (D.23), we use the Cauchy-Schwarz inequality on the interaction

term and obtain

<!-- formula-not-decoded -->

where in the last line we use the fact that ( s, a ) and ( s ′ , a ′ ) have the same marginal distribution. Thus, we obtain

<!-- formula-not-decoded -->

Next, to upper bound 〈 ¯ g α ( t ) -¯ g 0 α ∗ , α ( t ) -α ∗ 〉 , we use the H¨ older's inequality to obtain

<!-- formula-not-decoded -->

where the second inequality follows from ‖ α ( t ) -α ∗ ‖ 2 ≤ R u . For the term 〈 ¯ g 0 α ( t ) -¯ g 0 α ∗ , α ( t ) -α ∗ 〉 on the right-hand side of (D.26), we have

<!-- formula-not-decoded -->

where the second equality and the first inequality follow from (D.5) and (D.24), respectively.

Therefore, combining (D.21) with (D.22), (E.4), (D.26), and (D.27), we obtain

<!-- formula-not-decoded -->

Error Bound: Rearranging (D.28), we obtain

<!-- formula-not-decoded -->

Taking total expectation on both sides of (D.29) and telescoping for t +1 ∈ [ T ], we further obtain

<!-- formula-not-decoded -->

Let T ≥ 64 / (1 -µ ) 2 and η = T -1 / 2 . It holds that T -1 / 2 · ( η (1 -γ ) -4 η 2 ) -1 ≤ 16(1 -γ ) -1 / 2 and Tη 2 ≤ 1, which together with (D.30) implies

<!-- formula-not-decoded -->

where in the second inequality we use ‖ α (0) -α ∗ ‖ 2 ≤ R u and in the equality we use Lemma D.3. Thus, we conclude the proof of Theorem D.4.

Following the definition of u 0 α in (D.4), we define the local linearization of Q ω at the initialization as

<!-- formula-not-decoded -->

Similarly, for f θ we define

<!-- formula-not-decoded -->

In the sequel, we show that Theorem D.4 implies both Theorems 4.5 and 4.6.

To obtain Theorem 4.5, we set ρ = ˜ σ k , u α = f θ , v = τ k +1 · ( β -1 k Q ω k + τ -1 k f θ k ), µ = 0, and R u = R f . Using τ k +1 , τ k , and β k specified in Algorithm 1, we have

<!-- formula-not-decoded -->

where in the second inequality we use τ 2 k +1 β -2 k + τ 2 k +1 τ -2 k ≤ 1 and the fact that ( Q ω k ( s, a )) 2 ≤ 2( Q ω (0) ( s, a )) 2 +2 R 2 Q and ( f θ k ( s, a )) 2 ≤ 2( f θ (0) ( s, a )) 2 +2 R 2 f , which is a consequence of the 1-Lipschitz continuity of the neural network with respect to the weights. Also note that Q ω (0) ( s, a ) = f θ (0) ( s, a ) due to the fact that Q ω k and f θ k share the same initialization. Thus, we have v 1 = 4, v 2 = 4, and v 3 = 0. Moreover, by f 0 θ ∗ = Π F R f ,m f T f 0 θ ∗ = Π F R f ,m ( τ k +1 · ( β -1 k Q ω k + τ -1 k f θ k )), we have

<!-- formula-not-decoded -->

which together with the fact that τ k +1 · ( β -1 k Q 0 ω k ( s, a ) + τ -1 k f 0 θ k ( s, a )) ∈ F R f ,m f implies

<!-- formula-not-decoded -->

Finally, plugging (D.31) into Theorem D.4 for f θ , we obtain

<!-- formula-not-decoded -->

which, combining with the fact that √ a + b ≤ √ a + √ b for any a, b ≥ 0, gives Theorem 4.5.

To obtain Theorem 4.6, we set ρ = σ k , u α = Q ω , v = (1 -γ ) · r , µ = γ and R u = R Q . Correspondingly, we have v 1 = 0, v 2 = 0, v 3 = R 2 max and u 0 α ∗ = Q 0 ω ∗ . Moreover, by the definition of the operator T in (D.20), we have T = T π θ k , which implies Q π θ k = T Q π θ k . Meanwhile, by Assumption 4.3, we have Q π θ k ∈ F R Q ,m Q , which implies Q π θ k =

Π F R Q ,m Q Q π θ k = Π F R Q ,m Q T Q π θ k . Since we already show that Q 0 ω ∗ is the unique solution to the equation Q = Π F R Q ,m Q T Q , we obtain Q 0 α ∗ = Q π θ k . Therefore, we can substitute Q 0 α ∗ with Q π θ k in Theorem D.4 to obtain Theorem 4.6.

## E Proofs for Section 4.2

## E.1 Proof of Lemma 4.7

Proof. We first have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here Z k +1 ( s ) , Z θ k +1 ( s ) ∈ R are normalization factors, which are defined as

<!-- formula-not-decoded -->

respectively. Thus, we reformulate the inner product in (4.5) as

<!-- formula-not-decoded -->

where we use the fact that

<!-- formula-not-decoded -->

Thus, it remains to upper bound the right-hand side of (E.2). We first decompose it to two terms, namely the error from learning the Q-function and the error from fitting the improved and

policy, that is,

<!-- formula-not-decoded -->

## Upper Bounding (i): We have

<!-- formula-not-decoded -->

Taking expectation with respect to s ∼ ν ∗ on the both sides of (E.4) and using the CauchySchwarz inequality, we obatin

<!-- formula-not-decoded -->

where in the last inequality we use the error bound in (4.3) and the definition of φ ∗ k in (4.2). Upper Bounding (ii): By the Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

where in the last inequality we use the error bound in (4.4) and the definition of ψ ∗ k in (4.2). Combining (E.2), (E.3), (E.5), and (E.6), we have

<!-- formula-not-decoded -->

Therefore, we conclude the proof of Lemma 4.7.

## E.2 Proof of Lemma 4.8

Proof. We have

<!-- formula-not-decoded -->

where the first inequality follows from the definition of /lscript ∞ -norm and the third inequality follows from the Cauchy-Schwartz inequality. Hence, we finish the proof of (4.6). Therefore, we conclude the proof of Lemma 4.8.

## E.3 Proof of Lemma 4.9

Proof. We have

<!-- formula-not-decoded -->

where we use the 1-Lipschitz continuity of Q ω in ω and the constraint ‖ ω k -ω 0 ‖ 2 ≤ R ω . Therefore, we finish the proof of Lemma 4.9.

## F Proof of Corollary 4.11

Proof. By Theorems 4.5 and 4.6, we have /epsilon1 k +1 = O ( R f T -1 / 4 + R 5 / 4 f m -1 / 8 f + R 3 / 2 f m -1 / 4 f ) and /epsilon1 ′ k = O ( R Q T -1 / 4 + R 5 / 4 Q m -1 / 8 Q + R 3 / 2 Q m -1 / 4 Q ), which gives

<!-- formula-not-decoded -->

when m f = Ω( R 2 f ) and m Q = Ω( R 2 Q ).

Next, setting m f = R 10 f · Ω( K 18 · φ ∗ k 8 + K 8 · |A| ), m Q = Ω ( K 4 R 10 Q · ψ ∗ k 4 ) and T = Ω( K 4 R 4 f · ϕ ∗ k 4 + K 6 R 4 f · φ ∗ k 4 + K 2 R 4 Q · ψ ∗ k 4 ), we further have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing up (F.1) and (F.2) for k +1 ∈ [ K ] and plugging it into Theorem 4.10, we obtain

<!-- formula-not-decoded -->

which completes the proof of Corollary 4.11.

## G Proofs of Section 5

Proof of Lemma 5.1. The proof follows that of Lemma 6.1 in Kakade and Langford (2002). By the definition of V π ( s ) in (2.1), we have

<!-- formula-not-decoded -->

where the third inequality is obtained by taking E ν ∗ [ V π ( s 0 )] = E ν ∗ [ V π ( s )] out and, correspondingly, delaying V π ( s t ) by one time step to V π ( s t +1 ) in each term of the summation. Note that for the advantage function, by definition of the action-value function, we have

<!-- formula-not-decoded -->

which together with (G.1) implies

<!-- formula-not-decoded -->

Here the second equality follows from ( P π ∗ ) t ν ∗ = ν ∗ for any t ≥ 0 and σ ∗ = π ∗ ν ∗ . Finally, note that for any given s ∈ S ,

<!-- formula-not-decoded -->

Plugging (G.3) into (G.2) and recalling the definition of L ( π ) in (4.7), we finish the proof of Lemma 5.1.

Proof of Lemma 5.2. First, we have

<!-- formula-not-decoded -->

Recall that π k +1 ∝ exp { τ -1 k f θ k + β -1 k Q π θ k } and Z k +1 ( s ) and Z θ k ( s ) are defined in (E.1). Also recall that we have 〈 log Z θ k ( s ) , π ( · | s ) -π ′ ( · | s ) 〉 = 〈 log Z k ( s ) , π ( · | s ) -π ′ ( · | s ) 〉 = 0 for all k ,

<!-- formula-not-decoded -->

π , and π ′ , which implies that, on the right-hand-side of (G.4),

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Plugging (G.5) and (G.6) into (G.4), we obtain

<!-- formula-not-decoded -->

where in the last inequality we use the Pinsker's inequality. Rearranging the terms in (G.7), we finish the proof of Lemma 5.2.