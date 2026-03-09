## Finite Sample Analysis of Average-Reward TD Learning and Q -Learning

## Sheng Zhang ∗ Zhe Zhang ∗ Siva Theja Maguluri

The H. Milton Stewart School of Industrial and Systems Engineering

Georgia Institute of Technology siva.theja}@gatech.edu

{shengzhang, jimmy\_zhang,

## Abstract

The focus of this paper is on sample complexity guarantees of average-reward reinforcement learning algorithms, which are known to be more challenging to study than their discounted-reward counterparts. To the best of our knowledge, we provide the first known finite sample guarantees using both constant and diminishing step sizes of (i) average-reward TD( λ ) with linear function approximation for policy evaluation and (ii) average-reward Q -learning in the tabular setting to find the optimal policy. A major challenge is that since the value functions are agnostic to an additive constant, the corresponding Bellman operators are no longer contraction mappings under any norm. We obtain the results for TD( λ ) by working in an appropriately defined subspace that ensures uniqueness of the solution. For Q -learning, we exploit the span seminorm contractive property of the Bellman operator, and construct a novel Lyapunov function obtained by infimal convolution of a generalized Moreau envelope and the indicator function of a set.

## 1 Introduction

The average-reward setting is a classical setting for formulating the goal in an infinite-horizon Markov decision process (MDP) [1]. The need to maximize the average reward has been demonstrated in many applications, including scheduling automatic guided vehicles [2], inventory management in supply chains [3], communication system control and routing [4], cooperative multi-robot learning [5] and queuing network control [6]. In these problems, the discounted-reward criterion usually leads to poor long-time performance since the system operates over an extended period of time and the goal is to optimize long-term behavior, whereas the discounted objective biases the optimal policy to choose actions that lead to high near-term performance.

Even though there is a well developed theory of average-reward MDPs [7, 8, 9], the theoretical understanding of average-reward RL methods is still quite limited. Most existing results are focused only on asymptotic convergence [10, 11, 12, 13]. The focus of this paper is to understand the sample efficiency. How much data is required to guarantee a given level of accuracy?

Recent literature obtains finite sample guarantees for discounted-reward TD learning and Q -learning algorithms by developing novel analytical techniques [14, 15, 16, 17, 18]. Such a study of averagereward RL algorithms is not undertaken. Analysis of average-reward RL algorithms is known to be more challenging to study than their discounted-reward counterparts. The key property that is exploited in the study of discounted-reward problems is the contraction property of the underlying Bellman operator. In the average-reward setting, such a contraction property does not hold under any norm, and the Bellman equation is known to have multiple fixed points.

∗ Equal contribution. Correspondence to Sheng Zhang &lt;shengzhang@gatech.edu | shengzhangblog.com&gt;.

In this work, we take the first step toward understanding finite sample guarantees of (i) averagereward TD( λ ) with linear function approximation for policy evaluation, and (ii) average-reward tabular Q -learning in the synchronous setting for the control problem.

## 1.1 Contributions and Summary of Our Techniques

We establish the first finite sample convergence guarantees of average-reward TD( λ ) with linear function approximation and average-reward tabular Q -learning in the literature.

TD( λ ) Results. We study the average-reward TD( λ ) with linear function approximation under a general asynchronous update. We present finite sample bounds under both constant and diminishing step sizes. With a constant step-size, the iterates converge at an exponential rate to a small cylinder around the set of TD fixed points. With properly chosen diminishing step sizes, the mean-square distance of the iterates to the set of TD fixed points converges with an ˜ O ( 1 T ) rate, and this leads to a sample complexity of ˜ O ( 1 glyph[epsilon1] 2 ) . Our sample complexity bound also suggests that an intermediate value of λ yields the best performance. The dependence on the effective horizon plays a key role in the study of discounted-reward RL algorithms [19, 20]. There is no such effective horizon in average-reward problems, and the spectral gap of an appropriately defined matrix plays a key role instead.

TD( λ ) Analysis. A major challenge in the analysis is that the projected Bellman operator is not a contraction under any norm. Moreover, even though the projected Bellman equation can be written as a linear set of equations, they are underdetermined. So existing techniques [14, 15] are not directly applicable. Since the value function is unique up to an additive constant, we have a unique solution of the projected Bellman equation when restricted to an appropriately defined subspace. We exploit this property and work in this subspace, and use a quadratic Lyapunov function to obtain finite sample guarantees.

Q -learning Results. We consider a J -step synchronous Q -learning algorithm. We present finite sample error bounds for both constant and diminishing step sizes. The span of a vector is defined to be the difference between the maximum and minimum element. Since the optimal action-value function, Q ∗ , is agnostic up to an additive constant, we show that, with at most O (1 /glyph[epsilon1] 2 ) samples, the expected span of the error Q T -Q ∗ converges to glyph[epsilon1] for both decreasing and constant step sizes.

Q -learning Analysis. While the corresponding Bellman operator is not a contraction under any norm, it is known to be a contraction under the span seminorm. The span seminorm can be interpreted as the glyph[lscript] ∞ -distance to the subspace spanned by the all-ones vector. Finite sample bounds for stochastic approximation of glyph[lscript] ∞ -norm contractive operators were obtained in [18] by using generalized Moreau envelop as a smooth Lyapunov function. Here, we generalize this approach and introduce a new Lyapunov function to study span seminorm contractive operators. Our Lyapunov function is obtained by applying an infimal convolution with respect to an indicator function to the generalized Moreau envelop used in [18].

## 1.2 Related Literature

Average-Reward MDP. There is an extensive body of literature on average-reward MDPs. Several authors have made early contributions to average-reward problems [21, 7, 8, 22, 23]. There are well known dynamic programming algorithms for finding optimal policies such as policy iteration [7] and value iteration [24]. However, these algorithms require complete knowledge of the MDP, and are also computationally intractable in large state spaces [25].

Average-Reward Policy Evaluation. Tsitsiklis and Van Roy [10] proved the asymptotic convergence of the average-reward TD( λ ) with linear function approximation, and provided approximation error bounds. Yu and Bertsekas [26] proved the asymptotic convergence of the average-reward LSPE( λ ), and provided the rate of convergence for constant step size. Both TD( λ ) with linear function approximation and LSPE( λ ) aim to solve the same projected Bellman equation. However, TD( λ ) is based on stochastic approximation while LSPE( λ ) is based on least squares. In addition, the papers above assumed that the set of basis functions are independent of the all-ones vector, which apparently does not hold in the tabular setting. We do not require such a restrictive assumption in this paper. Recent work has also established the asymptotic convergence of the off-policy average-reward TD learning algorithm in the tabular setting [12], and finite sample guarantees of average-reward gradient

TD algorithms with linear function approximation [13]. Note that, [13] needed either the restrictive assumption aforementioned or the ridge regularization to obtain the finite sample results.

Average-Reward Control. The earliest control algorithms were those introduced by Schwartz [27] and Singh [28] without convergence proofs. The first provably convergent algorithms are RVI Q -learning and SSP Q -learning, introduced by Abounadi, Bertsekas, and Borkar [11]. SSP Q -learning and the algorithm introduced later by Gosavi [29] are limited to MDPs with a special state that is recurrent under all stationary policies, whereas RVI Q -learning is convergent for more general MDPs. Recently, Wan et al.[12] introduced an algorithm without a reference function, which is needed in RVI Q -learning, and proved its asymptotic convergence with the techniques which are a slight generalization of those in [11]. To the best of our knowledge, our paper is the first work in the literature that studies the finite sample guarantees of a general average-reward Q -learning algorithm.

Stochastic Approximation. Many RL algorithms can be viewed through the lens of stochastic approximation (SA). There is a well developed asymptotic theory of SA [30, 31, 32]. The ODE method is a dominant approach used in most asymptotic convergence proofs in RL [33]. However, this is a coarse tool, since it is not able to generate insight into an algorithm's sensitivity to noise in the system and step-size choices. Driven by the interest in finite sample guarantees of RL algorithms, recent years have witnessed a focus shifted from asymptotic analysis to non-asymptotic analysis of SA schemes. For example, a finite-time bound for linear SA was given in [15], which leads to finite-time bounds for asynchronous TD learning. [17] provided a finite-time analysis of asynchronous nonlinear SA, which yields finite-time bounds for asynchronous Q -learning.

Others. There are other related papers which are beyond the scope of the present paper. For instance, there is a line of work [34, 35, 36] on regret guarantees, which is a different focus compared to our work, for learning in average-reward MDPs. In addition, there are RL methods based on linear programming [37, 38], or learning automata [39, 40].

## 2 The Average-Reward Problem Setting

We consider an infinite-horizon average-reward MDP described by ( S , A , R , p ) , where S = { 1 , 2 , · · · , |S|} is a finite state space, A = { 1 , 2 , · · · , |A|} is a finite action space, R : S×A → [0 , 1] is the reward function, and p : S × S × A → [0 , 1] is the transition dynamics of the environment. An agent interacts with the environment according to the following protocol: at each time step t = 0 , 1 , 2 , · · · , the agent is in a state S t ∈ S and selects an action A t ∈ A , then receives from the environment an immediate reward R ( S t , A t ) and the next state S t +1 which is a sample drawn from p ( ·| S t , A t ) . The average reward of a deterministic stationary policy µ : S → A starting from state s ∈ S is defined as

<!-- formula-not-decoded -->

Let r ∗ ( s ) := sup µ ∈M r µ ( s ) , where M is the set of deterministic stationary policies. A policy µ ∗ ∈ M is said to be optimal if it satisfies r µ ∗ ( s ) = r ∗ ( s ) for all s ∈ S .

## 3 Policy Evaluation Algorithm: TD Learning

## 3.1 Problem Formulation

We consider the problem of evaluating a given policy µ ∈ M when the data is generated by applying the policy µ in the MDP. Since the system is an induced Markov reward process (MRP), for simplicity, we employ the notation R ( i ) := R ( i, µ ( i )) for rewards, and P ( i, j ) := p ( j | i, µ ( i )) for transition probabilities. We make the following standard assumption to ensure the existence and uniqueness of a stationary distribution π := [ π 1 , · · · , π |S| ] glyph[latticetop] .

Assumption 1. The Markov chain associated with P is irreducible and aperiodic › .

› It is a standard assumption in studying convergence of TD learning algorithms with linear function approximation [41, 10, 14] and guarantees that all states are visited an infinite number of times during an infinitely long trajectory.

Notice that π satisfies π glyph[latticetop] P = π glyph[latticetop] , with π i &gt; 0 for all i ∈ S . Let E π [ · ] denote expectation with respect to π and define D = diag ( π 1 , · · · , π |S| ) ∈ R |S|×|S| . It is easy to see that 〈 x, y 〉 D := x glyph[latticetop] Dy is a D -weighted inner product and we denote its induced norm by ‖ x ‖ D := √ x glyph[latticetop] Dx .

Under Assumption 1, the average reward in (2.1) satisfies r µ ( s ) = r ( µ ) := π glyph[latticetop] R for all s ∈ S . A differential value function v : S → R for policy µ satisfies the Bellman equation Tv = v , where the Bellman operator T : R |S| → R |S| is defined by Tv := Rr ( µ ) e + Pv . Here, e ∈ R |S| is the all-ones vector. Under Assumption 1, it is known that the set of differential value functions takes the form { v µ + ce | c ∈ R } , where v µ : S → R , known as the basic differential value function, is given by v µ := ∑ ∞ t =0 P t ( Rr ( µ ) e ) .

Most modern applications have large state spaces, so due to the curse of dimensionality, exact value function learning may be intractable. To mitigate this, we consider a linear function approximation V θ ( i ) = φ ( i ) glyph[latticetop] θ to differential value functions, where φ ( i ) := [ φ 1 ( i ) , · · · , φ d ( i )] glyph[latticetop] ∈ R d is the feature vector for state i ∈ S and θ ∈ R d is a tunable parameter vector. Here, { φ k : S → R | k = 1 , 2 , · · · , d } is a set of d basis functions to be viewed as vectors of dimension |S| . With this notation, V θ can be expressed compactly in the form V θ = Φ θ , where Φ is an |S|× d matrix whose k -th column is φ k . We assume that Φ has full column rank; that is, the basis functions { φ k | k = 1 , 2 , · · · , d } are linearly independent. This results in no loss of generality because if some basis function φ k is a linear combination of the others, it can be eliminated without changing the power of the approximation architecture. Additionally, we assume that ‖ φ ( i ) ‖ 2 ≤ 1 for all i ∈ S , which can be ensured through feature normalization.

## 3.2 Average-Reward TD( λ )

## Algorithm 1: TD( λ ) with linear function approximation

Input : initial guess ¯ r 0 and θ 0 , basis functions { φ k } k =1 , step-size sequence { β t } t ∈ N and positive constant c .

```
d α Initialize: z -1 = 0 , λ ∈ [0 , 1) . for t = 0 , 1 , . . . do Observe tuple: O t = ( s t , R ( s t ) , s t +1 ) Get TD error: δ t ( θ t ) = R ( s t ) -¯ r t + φ ( s t +1 ) glyph[latticetop] θ t -φ ( s t ) glyph[latticetop] θ t Update eligibility trace: z t = λz t -1 + φ ( s t ) Update average-reward estimate: ¯ r t +1 = ¯ r t + c α β t ( R ( s t ) -¯ r t ) Update parameter vector: θ t +1 = θ t + β t δ t ( θ t ) z t end
```

We study the average-reward TD learning with eligibility traces [10], denoted by TD( λ ) and parameterized by λ ∈ [0 , 1) . We consider the Markov chain observation model, where the observed tuples used by TD( λ ) are gathered from a single trajectory of the MRP. At every time step t , the algorithm observes one data tuple O t := ( s t , R ( s t ) , s t +1 ) consisting of the current state, the current reward and the next state. Suppose that at some time t , the current value of the parameter vector θ is θ t , and we have a scalar estimate ¯ r t of the average reward r ( µ ) , we define the TD error δ t ( θ t ) corresponding to the transition from s t to s t +1 as δ t ( θ t ) := R ( s t ) -¯ r t + φ ( s t +1 ) glyph[latticetop] θ t -φ ( s t ) glyph[latticetop] θ t . TD( λ ) updates ¯ r t and θ t as follows:

<!-- formula-not-decoded -->

where α t and β t are scalar step sizes, and the vector z t := ∑ t k =0 λ t -k φ ( s k ) is called the eligibility trace.

In this work, we focus on the single time-scale variant of TD( λ ) presented in Algorithm 1, that is, we assume that there exists a constant c α &gt; 0 such that α t = c α β t for all t . In order to represent TD( λ ) in a compact form, we construct a process X t := ( s t , s t +1 , z t ) . It is easy to see that { X t } is a Markov chain with an infinite state space. If we let Θ t := [ ¯ r t θ t ] , the TD( λ ) updates (3.1) can be expressed compactly as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Non-uniqueness of TD( λ ) limit point. For any m = 0 , 1 , · · · , the m -step Bellman operator is given by

<!-- formula-not-decoded -->

The asymptotic properties of TD( λ ) are closely tied to the λ -weighted version of the m -step Bellman operator. Define the averaged Bellman operator

<!-- formula-not-decoded -->

Note that the set of differential value functions is also the fixed points of T ( λ ) ; see Lemma 3 in [10] for a proof.

We denote by A := E π [ A ( X t )] and b := E π [ b ( X t )] the steady-state expectations of A ( X t ) and b ( X t ) , respectively. Stochastic approximation theory [32] shows that the asymptotic behavior of the sequence { Θ t } generated by (3.2) is closely linked with the corresponding ordinary differential equation (ODE) ˙ Θ t = A Θ t + b , and the limit of Θ t , denoted by Θ ∞ = [ ¯ r ∞ θ ∞ ] if exists, is an equilibrium point of the ODE, i.e.,

<!-- formula-not-decoded -->

Therefore, ¯ r ∞ = r ( µ ) , and θ ∞ is a solution of the projected Bellman equation

<!-- formula-not-decoded -->

where Π D,W Φ := Φ ( Φ glyph[latticetop] D Φ ) -1 Φ glyph[latticetop] D is the projection matrix onto the column space W Φ := { Φ θ | θ ∈ R d } of Φ with respect to the norm ‖·‖ D . It is worth noting that if e ∈ W Φ , then Π D,W Φ T ( λ ) has multiple fixed points, since any scalar multiple of e when added to a fixed point of Π D,W Φ T ( λ ) would also be a fixed point. For example, in the tabular case where Φ = I , Eq. (3.4) would become θ = T ( λ ) θ , of which the set of differential value functions are solutions.

## 3.3 Finite-Time Bounds for Average-Reward TD ( λ )

Before we present the finite-time bounds on the performance of TD( λ ) with Markovian observation noise, we illustrate the key ideas, which are inspired by [15]. The detailed proof can be found in Appendix A.3.

We study the drift of an appropriately chosen Lyapunov function to obtain an upper bound on the mean-square error. We define the subspace S Φ ,e as

<!-- formula-not-decoded -->

Let E be the orthogonal complement of S Φ ,e . We can interpret E as the set of equivalent classes with the equivalence relation ∼ on R d defined by θ 1 ∼ θ 2 if and only if θ 1 -θ 2 is in S Φ ,e . The following lemma characterizes the set of TD( λ ) fixed points; see Appendix A.1 for a proof.

Lemma 1. Under Assumption 1, the fixed points of the projected Bellman equation (3.4) are

<!-- formula-not-decoded -->

where θ ∗ ∈ E is a unique solution to the equation Φ θ = Π D,W Φ ,E T ( λ ) Φ θ . Here, Π D,W Φ ,E ( · ) is the projection operator onto the subspace W Φ ,E := { Φ θ | θ ∈ E } with respect to the norm ‖·‖ D

Remark 1. In the case where e glyph[negationslash]∈ W Φ , the projected Bellman equation (3.4) has a unique fixed point θ ∗ . This is why prior work requires that e does not belong to the column space of Φ .

We consider the Lyapunov function Φ(¯ r, θ ) := (¯ r -r ( µ )) 2 + ‖ Π 2 ,E ( θ -θ ∗ ) ‖ 2 2 . Here, Π 2 ,E is the projection onto the subspace E with respect to the 2 -norm ‖·‖ 2 . Note that ‖ Π 2 ,E ( θ -θ ∗ ) ‖ 2 2 measures the distance of θ to the set of TD( λ ) fixed points. The following Lemma establishes that the matrix A in (3.3) is negative definite over the subspace R × E for a sufficiently large c α . The proof is presented in Appendix A.2. With this result, we can show that the Lyapunov function Φ(¯ r, θ ) has a one-time-step negative drift.

Lemma 2. Under Assumption 1, we have

<!-- formula-not-decoded -->

where P ( λ ) = (1 -λ ) ∑ ∞ m =0 λ m P m +1 . Furthermore, when c α ≥ ∆+ √ 1 ∆ 2 (1 -λ ) 4 -1 (1 -λ ) 2 , we have

<!-- formula-not-decoded -->

To handle the Markovian noise, we use the conditioning argument along with the geometric mixing of the underlying Markov chain { X t } . Thus, we consider the following definition of the mixing time of a Markov chain.

Definition 1. Given a positive constant glyph[epsilon1] , we define τ ( glyph[epsilon1] ) ≥ 1 to be the mixing time of a Markov chain { X t } such that

<!-- formula-not-decoded -->

The following lemma establishes that the expectations of A ( X t ) and b ( X t ) converge to their steadystate values at a geometric rate. See Lemma 6.7 in [42] for a proof.

Lemma 3. Under Assumption 1, the Markov chain { X t } has a geometric mixing time, i.e., there exists a constant K ≥ 1 such that given a small positive constant glyph[epsilon1] we have τ ( glyph[epsilon1] ) ≤ K ln ( 1 glyph[epsilon1] ) .

We now state two finite-time bounds on the performance of TD( λ ). Part (a) studies TD( λ ) applied with sufficiently small constant step-size, which is common in practice. In this case, the iterates θ t will never converge to any TD( λ ) fixed point due to the noise variance, but our result shows that the expected distance of θ t to the set of TD( λ ) fixed points decreases at an exponential rate below some level that depends on the choice of step-size. Part (b) attains an ˜ O ( 1 T ) convergence rate to some TD( λ ) fixed point with a carefully chosen decaying step-size sequence.

Theorem 1. Consider iterates { (¯ r , θ ) } generated by Algorithm 1 with Assumption 1 and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(a) Let β t = β for all t , where positive constant β is properly chosen such that ∆ β &lt; 2 and βτ ( β ) ≤ min { 1 4 η , ∆ 228 η 2 } . Then, for all T ≥ τ ( β ) , we have

<!-- formula-not-decoded -->

(b) Let β t = c 1 t + c 2 where positive constants c 1 and c 2 are properly chosen such that 2 &lt; ∆ c 1 &lt; 2 c 2 and there exists a smallest positive integer t ∗ such that ∑ t ∗ -1 k =0 β k ≤ 1 2 η , and for all t ≥ t ∗ , ∑ t -1 k = t -τ ( β t ) β k ≤ min { 1 4 η , ∆ 228 η 2 } . Then, for all T ≥ t ∗ , we have

<!-- formula-not-decoded -->

Therefore, with an appropriate decaying step sizes suggested in Theorem 1(b), the following sample complexity of Algorithm 1 can be obtained.

Corollary 1. To find a pair (¯ r, θ ) with E [ | ¯ r -r ( µ ) | ] ≤ glyph[epsilon1] and E [ ‖ Π 2 ,E ( θ -θ ∗ ) ‖ 2 ] ≤ glyph[epsilon1] , Algorithm 1 requires at most the following number of samples:

<!-- formula-not-decoded -->

where K is the mixing time constant defined in Lemma 3.

Remark 2. Since ∆ defined in (3.5) is a non-decreasing function of λ , the sample complexity in Corollary 1 implies that the optimal λ should be neither too large nor too small.

## 3.4 Approximation Error

As we are satisfied with an approximation of any differential value function, we define the approximation error, inf c ∈ R ‖ Φ θ ∗ -( v µ + ce ) ‖ D , as the infimum of the D -weighted Euclidean distance of Φ θ ∗ to the set of differential value functions. Following the similar arguments from the proof of Theorem 3 in [10], we obtain the following approximation error bound,

<!-- formula-not-decoded -->

where the constant c λ is in [0 , 1) for any λ ∈ [0 , 1) and goes to 0 as λ → 1 . Note that the term inf θ ∈ R d ,c ∈ R ‖ Φ θ -( v µ + ce ) ‖ D is the minimal error possible given our approximation architecture, and becomes zero if our approximation architecture is able to represent exactly some differential value function. In particular, under the tabular setting, since any differential value function has exact representation, we have inf c ∈ R ‖ Φ θ ∗ -( v µ + ce ) ‖ D = 0 .

## 4 Control Algorithm: Q -learning

## 4.1 Problem Formulation

We consider the problem of finding an optimal policy µ ∗ ∈ M under the following unichain assumption (see Section 8.4 in [9] for details).

Assumption 2. An MDP is called unichain if the induced Markov chain consists of a single recurrent class plus a possibly empty set of transient states for any deterministic stationary policy.

Under Assumption 2, standard MDP theory [9] shows that there exist a unique r ∗ ∈ R such that r ∗ ( s ) = r ∗ for all s ∈ S , and a unique Q ∗ : S × A → R up to an additive constant, such that the following Bellman optimality equation holds for all state-action pairs ( s, a ) ∈ S × A :

<!-- formula-not-decoded -->

The optimal policy µ ∗ is then obtained by µ ∗ ( s ) := argmax a ∈A Q ∗ ( s, a ) . If we define ¯ E := { ce | c ∈ R } as the subspace spanned by the all-ones vector e ∈ R |S|×|A| and denote the Bellman operator H by

<!-- formula-not-decoded -->

then (4.1) can be rewritten as a set inclusion condition:

<!-- formula-not-decoded -->

Importantly, by observing that the operator H is indifferent to constant shifting, i.e., H ( Q + ce ) = H ( Q ) + ce, we can view all Q constant shifts, Q ¯ E := { Q + ce : c ∈ R } , as an equivalent class and interpret (4.2) as a fixed-set equation:

<!-- formula-not-decoded -->

Next we propose an SA algorithm to solve (4.3) by iteratively updating some 'representative" of the underlying equivalent class.

Algorithm 2: J -step Synchronous Q -learning

Input : initial guess Q 0 , step-size sequence { η t } t ∈ N , offset function f : R |S|×|A| → R .

<!-- formula-not-decoded -->

```
for t = 0 , 1 , . . . do end
```

## 4.2 Synchronous Q -learning

Given the sample Bellman operator ˆ H defined by

<!-- formula-not-decoded -->

an SA algorithm for solving (4.3) is

<!-- formula-not-decoded -->

It might be necessary, sometimes, to apply H to Q for J steps before updating Q . More specifically, if we denote by µ Q ( s ) := arg max a Q ( s, a ) the Q -improving policy, the J -step Bellman operator and the J -step fixed-set equation are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So the J -step SA algorithm is the same as (4.4) except for a J -step sample Bellman operator,

<!-- formula-not-decoded -->

where s 1 ∼ p ( ·| s, a ) , . . . , s J ∼ p ( ·| s J -1 , µ Q ( s J -1 )) . The complete algorithm is presented in Algorithm 2.

Remark 3. (1) The SA algorithm in (4.4) is a special case of Algorithm 2 with J = 1 . (2) The solution to the J -step fixed-set equation (4.6) is the same for all J ≥ 1 . (3) An extra offset function f ( Q ) is included to ensure numerical stability (see Section 2.2 in [11]). If J = 1 and the offset function f satisfies Assumption 2.2 in [11], Algorithm 2 recovers the RVI Q -learning algorithm.

## 4.3 Finite-Time Analysis

Now we seek to establish finite-time convergence guarantees of Algorithm 2. Since the Q -improving policy and its suboptimality gap are the same for any Q ∈ Q ¯ E , we can measure Algorithm 2's progress by

<!-- formula-not-decoded -->

for some norm ‖·‖ α . For example, the span seminorm is equivalent to ‖·‖ ∞ , sp . Additionally, we need to make a span contraction assumption similar to Theorem 8.5.2 in [9].

Assumption 3. The J -step Bellman operator H J is a span contraction for some J ≥ 1 , i.e., there exists a γ ∈ [0 , 1) such that for any Q 1 and Q 2 defined on S × A ,

<!-- formula-not-decoded -->

Such an assumption is not restrictive. A lower bound for 1 -γ is the minimum probability of any two deterministic stationary Markov policies starting from any pair of states ending in the same state

in J steps [9]. Thus, under an unichain assumption, γ &lt; 1 can be guaranteed for J = | S | if we apply the aperiodic transformation in Section 8.5.4 of [9] to ensure a non-zero probability of all states in a single recurrent class in | S | steps.

We sketch the outline of our Lyaponuv convergence proof and leave the details to Appendix B. First, we construct a Lyaponov function. The key insight is that the span seminorm ‖·‖ ∞ , sp can be interpreted as the infimal convolution of the glyph[lscript] ∞ -norm ‖·‖ ∞ and the indicator function of ¯ E , i.e.,

<!-- formula-not-decoded -->

where δ ¯ E ( x ) := { 0 , x ∈ ¯ E, ∞ , otherwise .

As illustrated by Lemma 4 in Appendix B.1, the inifimal convolution operation has many desirable properties. For example, it is commutative, associative, convexity-preserving and smoothnesspreserving. These nice properties allow us to design a Lyaponov function M ¯ E for the equivalent classes by tweaking the usual Lyaponov function M as follows,

<!-- formula-not-decoded -->

In particular, since Assumption 3 implies only span contraction, we utilize the smoothed Lyaponov function proposed in [18] for glyph[lscript] ∞ -norm contraction, i.e., M ( Q ) := 1 2 ( ‖·‖ 2 ∞ ✷ 1 u ‖·‖ 2 p )( Q ) with p := 4 log( | S || A | ) and u := (1 / 2 + 1 / (2 γ )) 2 -1 . We show M ¯ E ( · ) is a uniform approximation of ‖·‖ 2 ∞ , sp :

<!-- formula-not-decoded -->

and it is smooth with respect to ‖·‖ p, sp :

<!-- formula-not-decoded -->

Next, since the span contraction assumption leads to a negative drift in (4.10) for some constant α 2 defined in Appendix B.3,

<!-- formula-not-decoded -->

we can use the smoothness and uniform approximation properties of M ¯ E to provide a recursive bound of Q t +1 for some α 3 and α 4 defined in Appendix B.3,

<!-- formula-not-decoded -->

Clearly, by taking a small enough step size η t , (4.12) implies the convergence of E [ M ¯ E ( Q t -Q ∗ )] . Now we can state a sample complexity upper bound for Algorithm 2 by choosing a specific step-size sequence.

Theorem 2. If { Q t } is generated by Algorithm 2 with a decaying step-size sequence

<!-- formula-not-decoded -->

then for some universal constant C the following bound holds for all t ≥ 1 ,

<!-- formula-not-decoded -->

If instead a constant step size η ≤ (1 -γ ) 2 288 log( | S || A | ) is employed, then

<!-- formula-not-decoded -->

In both cases, it takes

<!-- formula-not-decoded -->

samples to find a Q t with E [ ‖ Q t -Q ∗ ‖ ∞ , sp ] ≤ glyph[epsilon1] . Taking into account ‖ Q ∗ ‖ ∞ , sp ≤ J 1 -γ from the span contraction assumption, the sample complexity can be simplified further to

<!-- formula-not-decoded -->

which is similar to the sample complexity of γ -discounted Q -learning algorithm [18, 43].

## 5 Numerical Experiments

In this section we present empirical results of the average-reward TD( λ ) with linear function approximation (i.e. Algorithm 1). In our simulation, we consider a randomly generated MRP with |S| = 100 states and a randomly generated feature matrix Φ with d = 20 features and e ∈ W Φ . Experimental details and figures are provided in Appendix C. All the implementations are publicly available fi .

We first show that if the algorithm starts from different initial points, it will converge to different TD fixed points. To demonstrate that, we implement the algorithm using diminishing step sizes for 4 different θ 0 , and then plot E [ ‖ Π 2 ,E ( θ t -θ ∗ ) ‖ 2 ] and E [ ( θ t -θ ∗ ) glyph[latticetop] θ e ‖ θ e ‖ 2 ] as functions of the number of iterations t in Figure 1 and 2, respectively. Recall from Lemma 1 that { θ ∗ + cθ e | c ∈ R } is the set of TD limit points. We observe in Figure 1 that E [ ‖ Π 2 ,E ( θ t -θ ∗ ) ‖ 2 ] converges to 0 for all 4 initial points, which means that the iterates θ t converge to some TD limit point regardless of θ 0 . Moreover, Figure 2 shows that E [ ( θ t -θ ∗ ) glyph[latticetop] θ e ‖ θ e ‖ 2 ] converges to different values for different initial points. This, combined with Figure 1, implies that the algorithm converges to different TD limit points, starting from different θ 0 .

We next numerically verify the finite-time error bounds of Algorithm 1 using decaying step sizes. ,

In Figure 3, we plot E [ (¯ r t -r ∗ ) 2 + ‖ Π 2 ,E ( θ t -θ ∗ ) ‖ 2 2 ] as a function of t for λ ∈ { 0 , 0 . 2 , 0 . 4 , 0 . 8 } where r ∗ denotes the average-reward of the MRP. We see that the iterates { (¯ r t , θ t ) } of the algorithm converge for all values of λ . To further verify the rate of convergence, we plot ln E [ (¯ r t -r ∗ ) 2 + ‖ Π 2 ,E ( θ t -θ ∗ ) ‖ 2 2 ] as a function of ln t in Figure 4 for large t and the slopes of these lines are provided in the legend. The plot shows E [ (¯ r t -r ∗ ) 2 + ‖ Π 2 ,E ( θ t -θ ∗ ) ‖ 2 2 ] ≈ O ( 1 t ) asymptotically, which agrees with Theorem 1(b). In addition, we notice from Figure 3 and 4 that the best λ in terms of sample complexity is 0 . 2 , which confirms our Remark 2 that intermediate value of λ yields the best performance with regard to sample complexity.

## 6 Conclusion

We establish the first finite sample convergence bounds of (i) average-reward TD( λ ) with linear function approximation under Markovian observation noise, and (ii) average-reward tabular Q -learning in the synchronous setting. These RL algorithms can be viewed as SA schemes to solve average-reward Bellman equations. However, the Bellman operators are not contractive under any norm. To resolve this difficulty, we construct Lyapunov functions using projection and infimal convolution to analyze the convergence of equivalent classes generated by these algorithms. Our approach is simple and general, so we expect it to have broader applications in other problems.

When analyzing the average-reward Q -learning algorithm, we made a J -step span contraction assumption (i.e. Assumption 4), which is not needed for the asymptotic convergence [10]. However, it is unclear if such an assumption is necessary for establishing any finite-time convergence bound. Since our results are the first finite sample bounds, a future research direction is on relaxing this assumption. Besides, it would be interesting to experiment our algorithms in practice and see how the empirical performance is compared with our theoretical bounds.

fi https://github.com/xiaojianzhang/Average-Reward-TD-Q-Learning

## Acknowledgment

This work was partially supported by an award from Raytheon Technologies and a seed grant from Georgia Institute of Technology.

## References

- [1] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning , 3(1):9-44, 1988.
- [2] Prasad Tadepalli, DoKyeong Ok, et al. H-learning: A reinforcement learning method to optimize undiscounted average reward. 1994.
- [3] Ilaria Giannoccaro and Pierpaolo Pontrandolfo. Inventory management in supply chains: a reinforcement learning approach. International Journal of Production Economics , 78(2):153161, 2002.
- [4] Peter Marbach, Oliver Mihatsch, and John N Tsitsiklis. Call admission control and routing in integrated services networks using neuro-dynamic programming. IEEE Journal on selected areas in communications , 18(2):197-208, 2000.
- [5] Poj Tangamchit, John M Dolan, and Pradeep K Khosla. The necessity of average rewards in cooperative multirobot learning. In Proceedings 2002 IEEE International Conference on Robotics and Automation (Cat. No. 02CH37292) , volume 2, pages 1296-1301. IEEE, 2002.
- [6] Jim G Dai and Mark Gluzman. Queueing network controls via deep reinforcement learning. arXiv preprint arXiv:2008.01644 , 2020.
- [7] Ronald A Howard. Dynamic programming and markov processes. 1960.
- [8] David Blackwell. Discrete dynamic programming. The Annals of Mathematical Statistics , pages 719-726, 1962.
- [9] Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014.
- [10] John N Tsitsiklis and Benjamin Van Roy. Average cost temporal-difference learning. Automatica , 35(11):1799-1808, 1999.
- [11] Jinane Abounadi, D Bertsekas, and Vivek S Borkar. Learning algorithms for markov decision processes with average cost. SIAM Journal on Control and Optimization , 40(3):681-698, 2001.
- [12] Yi Wan, Abhishek Naik, and Richard S Sutton. Learning and planning in average-reward markov decision processes. arXiv preprint arXiv:2006.16318 , 2020.
- [13] Shangtong Zhang, Yi Wan, Richard S Sutton, and Shimon Whiteson. Average-reward off-policy policy evaluation with function approximation. arXiv preprint arXiv:2101.02808 , 2021.
- [14] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. arXiv preprint arXiv:1806.02450 , 2018.
- [15] Rayadurgam Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation andtd learning. In Conference on Learning Theory , pages 2803-2830. PMLR, 2019.
- [16] Zaiwei Chen, Sheng Zhang, Thinh T Doan, Siva Theja Maguluri, and John-Paul Clarke. Performance of q-learning with linear function approximation: Stability and finite-time analysis. arXiv preprint arXiv:1905.11425 , 2019.
- [17] Guannan Qu and Adam Wierman. Finite-time analysis of asynchronous stochastic approximation and q -learning. In Conference on Learning Theory , pages 3185-3205. PMLR, 2020.
- [18] Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finitesample analysis of stochastic approximation using smooth convex envelopes. arXiv preprint arXiv:2002.00874 , 2020.

- [19] Ashwin Pananjady and Martin J Wainwright. Instance-dependent glyph[lscript] ∞ -bounds for policy evaluation in tabular reinforcement learning. IEEE Transactions on Information Theory , 67(1):566585, 2020.
- [20] Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. A lyapunov theory for finite-sample guarantees of asynchronous q-learning and td-learning variants. arXiv preprint arXiv:2102.01567 , 2021.
- [21] Dean Gillette. 9. stochastic games with zero stop probabilities. In Contributions to the Theory of Games (AM-39), Volume III , pages 179-188. Princeton University Press, 2016.
- [22] Barry W Brown. On the iterative method of dynamic programming on a finite space discrete time markov process. The annals of mathematical statistics , pages 1279-1285, 1965.
- [23] Arthur F Veinott. On finding optimal policies in discrete dynamic programming with no discounting. The Annals of Mathematical Statistics , 37(5):1284-1294, 1966.
- [24] Douglas J White. Dynamic programming, markov chains, and the method of successive approximations. Journal of Mathematical Analysis and Applications , 6(3):373-376, 1963.
- [25] Sridhar Mahadevan. Average reward reinforcement learning: Foundations, algorithms, and empirical results. Machine learning , 22(1):159-195, 1996.
- [26] Huizhen Yu and Dimitri P Bertsekas. Convergence results for some temporal difference methods based on least squares. IEEE Transactions on Automatic Control , 54(7):1515-1531, 2009.
- [27] Anton Schwartz. A reinforcement learning method for maximizing undiscounted rewards. In Proceedings of the tenth international conference on machine learning , volume 298, pages 298-305, 1993.
- [28] Satinder P Singh. Reinforcement learning algorithms for average-payoff markovian decision processes. In AAAI , volume 94, pages 700-705, 1994.
- [29] Abhijit Gosavi. Reinforcement learning for long-run average cost. European Journal of Operational Research , 155(3):654-674, 2004.
- [30] Harold Kushner and G George Yin. Stochastic approximation and recursive algorithms and applications , volume 35. Springer Science &amp; Business Media, 2003.
- [31] Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint , volume 48. Springer, 2009.
- [32] Albert Benveniste, Michel Métivier, and Pierre Priouret. Adaptive algorithms and stochastic approximations , volume 22. Springer Science &amp; Business Media, 2012.
- [33] Vivek S Borkar and Sean P Meyn. The ode method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38(2):447-469, 2000.
- [34] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(4), 2010.
- [35] Yasin Abbasi-Yadkori, Peter Bartlett, Kush Bhatia, Nevena Lazic, Csaba Szepesvari, and Gellért Weisz. Politex: Regret bounds for policy iteration using expert prediction. In International Conference on Machine Learning , pages 3692-3702. PMLR, 2019.
- [36] Chen-Yu Wei, Mehdi Jafarnia Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Modelfree reinforcement learning in infinite-horizon average-reward markov decision processes. In International Conference on Machine Learning , pages 10170-10180. PMLR, 2020.
- [37] Mengdi Wang. Primal-dual π learning: Sample complexity and sublinear run time for ergodic markov decision problems. arXiv preprint arXiv:1710.06100 , 2017.
- [38] Gergely Neu, Anders Jonsson, and Vicenç Gómez. A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.

- [39] R Wheeler and K Narendra. Decentralized learning in finite markov chains. IEEE Transactions on Automatic Control , 31(6):519-526, 1986.
- [40] Hyeong Soo Chang. Decentralized learning in finite markov chains: revisited. IEEE Transactions on Automatic Control , 54(7):1648-1653, 2009.
- [41] John N Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. IEEE transactions on automatic control , 42(5):674-690, 1997.
- [42] Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.
- [43] Martin J Wainwright. Stochastic approximation with cone-contractive operators: Sharp glyph[lscript] ∞ -bounds for q-learning. arXiv preprint arXiv:1905.06265 , 2019.
- [44] Abhishek Gupta, Rahul Jain, and Peter W Glynn. An empirical algorithm for relative value iteration for average-cost mdps. In 2015 54th IEEE Conference on Decision and Control (CDC) , pages 5079-5084. IEEE, 2015.
- [45] Amir Beck. First-order methods in optimization . SIAM, 2017.