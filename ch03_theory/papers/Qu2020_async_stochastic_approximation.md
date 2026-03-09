## Finite-Time Analysis of Asynchronous Stochastic Approximation and Q -Learning

Guannan Qu Adam Wierman

Department of Computing and Mathematical Sciences California Institute of Technology Pasadena, CA 91125, USA

## Abstract

We consider a general asynchronous Stochastic Approximation (SA) scheme featuring a weighted infinity-norm contractive operator, and prove a bound on its finite-time convergence rate on a single trajectory. Additionally, we specialize the result to asynchronous Q -learning. The resulting bound matches the sharpest available bound for synchronous Q -learning, and improves over previous known bounds for asynchronous Q -learning.

Keywords:

Stochastic approximation, Q -learning, finite time analysis.

## 1. Introduction

Reinforcement learning (RL) has received renewed interest recently due to its remarkable successes in diverse areas. Many RL algorithms can be viewed through the lens of Stochastic Approximation (SA) (Robbins and Monro, 1951). SA algorithms are widely used beyond RL in areas such as machine learning, stochastic control, signal processing, and communications and, as a result, there is a broad and deep literature focused on the analysis and applications of SA that has developed a rich class of ODE-based tools for proving convergence of SA schemes, e.g., see the books Borkar (2009); Benveniste et al. (2012). In the context of RL, it has been shown that linear SA captures TD-learning and that the ODE-based SA framework can be used to prove the convergence of TDlearning (Tsitsiklis and Van Roy, 1997). A similar connection can be found in the case of actor-critic methods (Konda and Tsitsiklis, 2000, 2003).

Most of the classical analysis in SA is asymptotic in nature; however this has changed recently. Driven by the interest in finite-time convergence of RL methods, the focus has shifted to nonasymptotic analysis of SA schemes. For example, in just the past year, a finite-time bound for linear SA is given in Srikant and Ying (2019), which leads to finite time error bounds for TD-learning, and a finite-time bound for a linear two time scale SA model is given in Gupta et al. (2019); Doan (2019); Xu et al. (2019), which leads to finite-time error bounds for the gradient TD method. These results can be viewed as extensions of the classical ODE-based SA framework, which requires the SA algorithm to admit a 'limiting' ODE associated with a Lyapunov function that certifies stability.

While ODE-based approaches are powerful, there are popular classes of nonlinear SA schemes featuring a nonlinear operator with infinity-norm contraction that cannot be directly analyzed from the ODE-based SA framework (Tsitsiklis, 1994; Bertsekas and Tsitsiklis, 1996). This class of SA methods captures a particularly important class of RL methods, the Watkin's Q -learning method (Watkins and Dayan, 1992), and so understanding the behavior of this class of SA schemes is impor-

c © 2020 G. Qu &amp; A. Wierman.

GQU@CALTECH.EDU

ADAMW@CALTECH.EDU

tant for understanding the finite-time behavior of Q -learning. Over the past year, progress has been made toward the finite-time analysis of these nonlinear SA schemes. In particular, Shah and Xie (2018) provides a finite-time convergence result for SA with an infinity-norm contractive operator, and Wainwright (2019a) provides sharp convergence rates for SA with a cone-contractive operator. However, both of these works consider the synchronous case, i.e., at each time all entries of the iterate are updated. This is a significant limitation since, in many applications, e.g., Q -learning on a single trajectory, the update is asynchronous , i.e., only one of the entries is updated at a time. This leads to the following question, which is the focus of this paper:

What is the finite-time convergence rate for asynchronous SA/ Q -learning on a single trajectory?

Contribution. In this paper, we provide a finite-time analysis of asynchronous nonlinear SA schemes featuring a weighted infinity norm contraction. We prove an O ( 1 (1 -γ ) 1 . 5 1 √ T ) convergence rate in weighted infinity-norm for the SA scheme, where γ is the contraction coefficient (Theorem 4). Notably, our results are sharper than the result in the synchronous case in Shah and Xie (2018, Thm. 5). 1

As a direct consequence, our result shows a ˜ O ( 1 (1 -γ ) 5 1 /epsilon1 2 ) convergence time to reach an ε -accurate (measured in infinity-norm) estimate of the Q -function for the asynchronous Q -learning method on a single trajectory in the infinite horizon γ -discounted MDP setting (Theorem 7). This result matches the sharpest known bound for synchronous Q -learning (Wainwright, 2019a), and to the best of our knowledge, improves over the best known finite-time bounds on asynchronous Q -learning (Even-Dar and Mansour, 2003) on a single trajectory in terms of its dependence on 1 ε , 1 1 -γ , and the state-action space size. Further, our results clarify a blow-up phenomenon in the asynchronous Q -learning literature where the error can blow up exponentially in 1 1 -γ . We show such a blow-up can be avoided by using a rescaled linear step size. This is consistent with related findings in other settings (Jin et al., 2018; Wainwright, 2019a).

Our proof technique is different from those in the literature, e.g., Even-Dar and Mansour (2003); Shah and Xie (2018); Wainwright (2019a). Specifically, we do not use an epoch-based analysis, as in Even-Dar and Mansour (2003); Shah and Xie (2018), where the error is controlled epoch-byepoch. Instead, we decompose the error in a recursive manner, and this decomposition provides a more transparent approach for analyzing how the stochastic noise impacts the approximation error. This ultimately leads to a sharper bound. Further, our approach for handling asynchronicity is very different from Even-Dar and Mansour (2003) and is partially inspired by the 'drift' analysis in the ODE-based SA literature Srikant and Ying (2019).

Related Work. Our results provide new insights about Q -learning and more generally, SA with an infinity-norm contractive operator. Q -learning was first proposed in Watkins and Dayan (1992). Its asymptotic convergence has been proven in Tsitsiklis (1994); Jaakkola et al. (1994), where its connection to SA with infinity-norm contractive operator was established. The first work on non-asymptotic analysis of Q -learning is Szepesv´ ari (1998), which focused on an i.i.d. setting. A generalization beyond the i.i.d. setting was provided by Even-Dar and Mansour (2003), which proves finite-time bounds for synchronous and asynchronous Q -learning with polynomial and linear step sizes. Both Szepesv´ ari (1998) and Even-Dar and Mansour (2003) discover that, when using a linear step size, there is an exponential blow-up in 1 1 -γ , where γ is the discounting factor;

1. As another related work Wainwright (2019a) does not provide an explicit bound for the synchronous SA scheme, we can only compare with Wainwright (2019a) in the context of Q -learning.

further, in the asynchronous setting, there is at least cubic dependence on the state-action space size (Even-Dar and Mansour, 2003, Thm. 4). Subsequently, Azar et al. (2011) proposes speedy Q -learning, a variant of synchronous Q -learning, by adding a momentum term, and shows it avoids the exponential blow-up with a finite time bound that scales in 1 (1 -γ ) 4 /epsilon1 2 . More recently, Shah and Xie (2018); Wainwright (2019a) provide finite time bounds for general synchronous SA, which indicates that even in the classical Q -learning setup, the exponential blow-up can be avoided by using a rescaled linear step size. Specifically, Wainwright (2019a) shows a finite time bound for synchronous Q -learning that scales in 1 (1 -γ ) 5 /epsilon1 2 . To the best of our knowledge, this is the sharpest known bound for synchronous Q -learning. Compared with the above papers, our result bridges the gap between the understanding of synchronous SA/ Q -learning and asynchronous SA/ Q -learning. Our finite time bounds for asynchronous Q -learning match the sharpest known scaling in 1 (1 -γ ) and 1 ε in synchronous Q -learning. Further, compared with the best known bounds for asynchronous Q -learning (Even-Dar and Mansour, 2003), our result improves the dependence on state-action space size from (at least) cubic to square. Additionally, our work presents a new analytic approach.

Other related work on SA and Q -learning include Lee and He (2019), which combines the ODEbased SA framework with the switch system theory to show the asymptotic convergence of asynchronous Q -learning in an i.i.d. setting; Beck and Srikant (2012), which studies the finite time error bound of constant step size Q -learning; and Melo et al. (2008); Chen et al. (2019), which analyze Q -learning with linear function approximation.

We also mention that there are other lines of work on Q -learning focusing on different models and performance measures. One line of work seeks to propose variants of Q -learning, e.g. recent work Wainwright (2019b) that achieves a minimax optimal rate. Earlier examples include Hasselt (2010); Azar et al. (2013); Sidford et al. (2018a,b); Devraj and Meyn (2017); Kearns and Singh (1999). Compared to these papers, our work focuses on general asynchronous SA and seeks to understand the convergence of the classical form of asynchronous SA/ Q -learning. Another related line of work on Q -learning focuses on proving bounds on regret, e.g. Strehl et al. (2006); Jin et al. (2018); Dong et al. (2019); Wei et al. (2019). Regret is a fundamentally different goal than providing finitetime convergence bounds, and the results and techniques across the two communities are quite different. The reason is that regret bound results need to address the problem of exploration, and the performance metric focuses on the transient performance, without the need to approximate every entry of Q -function to the same accuracy. In contrast, infinity-norm finite-time error bound results typically assume a form of sufficient exploration (e.g. the i.i.d. assumption used in Szepesv´ ari (1998); Lee and He (2019) and the covering time assumption used in Even-Dar and Mansour (2003)) and require every entry of the Q -function to be accurately estimated.

## 2. Finite-Time Analysis of Stochastic Approximation

In this section, we present our results on the finite-time analysis of asynchronous SA with a (weighted) infinity-norm contractive operator. We apply the results in this section to Q -learning in Section 3.

To begin, we formally define the problem setting. Let N = { 1 , . . . , n } , x ∈ R N , and F : R N → R N is an operator. We use F i to denote the i 'th entry of F . We consider the following stochastic approximation scheme that keeps updating x ( t ) ∈ R N starting from x (0) being the all zero vector,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash where i t ∈ N is a stochastic process adapted to a filtration F t , and w ( t ) is some noise that we will discuss later. As we show in Section 3, this stochastic approximation scheme captures the asynchronous Q -learning algorithm.

Given the setting described above, the following assumptions underlie our main result. Similar to Tsitsiklis (1994), the first assumption is concerned with the contraction of F in a weighted infinity norm, which we define in Definition 1. The reason that we consider the weighted infinity norm instead of the standard infinity norm is that its generality will capture not just the discounted case Q -learning, but also the undiscounted case, as shown by Tsitsiklis (1994, Sec. 7).

Definition 1 (Weighted Infinity Norm) Given a positive vector v = [ v 1 , . . . , v n ] /latticetop ∈ R N , the weighted infinity norm ‖ · ‖ v is given by ‖ x ‖ v = sup i ∈N | x i | v i .

Throughout the rest of the section, we fix a positive vector v ∈ R n and all the norms in the section are in ‖ · ‖ v . We also denote v = inf i ∈N v i , the smallest entry of v . We comment that when v is a all one vector, ‖ · ‖ v becomes the standard infinity norm. We use the following result frequently on the induced matrix norm of ‖ · ‖ v , the proof of which can be found in Appendix A.1.

Proposition 2 The induced matrix norm of ‖ · ‖ v for a matrix A = [ a ij ] i,j ∈N is given by ‖ A ‖ v = sup i ∈N ∑ j ∈N v j v i | a ij | . When A is a diagonal matrix, ‖ A ‖ v = sup i ∈N | a ii | .

With these preparations, we are now ready to state Assumption 1 on the contraction property of F . This assumption is standard in the literature, e.g., (Tsitsiklis, 1994; Wainwright, 2019a), 2 and is satisfied by the Q -learning algorithm as will be shown in Section 3. Note that, as a consequence of Assumption 1, F has a unique fixed point x ∗ . We also note that we do not require the monotonicity assumption needed in Wainwright (2019a).

Assumption 1 (Contraction) (a) Operator F is γ contraction in ‖ · ‖ v , i.e. for any x, y ∈ R N , ‖ F ( x ) -F ( y ) ‖ v ≤ γ ‖ x -y ‖ v . (b) There exists some constant C &gt; 0 s.t. ‖ F ( x ) ‖ v ≤ γ ‖ x ‖ v + C, ∀ x ∈ R N .

Assumption 1(a) directly implies Assumption 1(b) with C = (1 + γ ) ‖ x ∗ ‖ v . 3 We write Assumption 1(b) as a separate assumption since, in some applications (e.g. Q -learning), the constant C can be better than (1 + γ ) ‖ x ∗ ‖ v . Our next assumption concerns the noise sequence w ( t ) . It is also standard (Shah and Xie, 2018) and is satisfied by Q -learning.

Assumption 2 (Martingale Difference Sequence) w ( t ) is F t +1 measurable and satisfies E w ( t ) |F t = 0 . Further, | w ( t ) | ≤ ¯ w almost surely for some constant ¯ w .

Lastly, we make an assumption regarding the stochastic process i t .

Assumption 3 (Sufficient Exploration) There exists a σ ∈ (0 , 1) and positive integer, τ , such that, for any i ∈ N and t ≥ τ , P ( i t = i |F t -τ ) ≥ σ .

2. Wainwright (2019a) considers contraction in a gauge norm associated with a cone, which is more general than the weighted infinity norm.

3. To see this, note ‖ F ( x ) ‖ v ≤ ‖ F ( x ) -F ( x ∗ ) ‖ v + ‖ F ( x ∗ ) ‖ v ≤ γ ‖ x -x ∗ ‖ v + ‖ x ∗ ‖ v ≤ γ ‖ x ‖ v +(1 + γ ) ‖ x ∗ ‖ v .

Assumption 3 means that, given the history up to t -τ , the distribution of i t must have positive probability for every i . Its purpose is to ensure every i is visited by i t sufficiently often. We note that Assumption 3 is more general than many typical ergodicity assumptions used in the SA literature, e.g., Srikant and Ying (2019). For example, the following proposition shows that if i t is an ergodic Markov chain on state space N , then Assumption 3 is automatically true with σ and τ depending on the stationary distribution and the mixing time of the Markov chain, where the mixing time refers to the minimum time it takes to reach within 1 / 4 total variation distance of the stationary distribution regardless of the initial state (Levin and Peres, 2017, Sec. 4.5). The proof of Proposition 3 can be found in Appendix A.2.

Proposition 3 If i t is a ergodic Markov chain on state space N with stationary distribution µ and mixing time t MIX , then Assumption 3 holds with σ = 1 2 µ min , where µ min = min i ∈N µ i , and τ = /ceilingleft log 2 ( 2 µ min ) /ceilingright t MIX .

With these assumptions, we are ready to state our main result,

Theorem 4 Suppose Assumptions 1, 2 and 3 hold. Further, assume there exists constant ¯ x ≥ ‖ x ∗ ‖ v s.t. ∀ t, ‖ x ( t ) ‖ v ≤ ¯ x almost surely. Let the step size be α t = h t + t 0 with t 0 ≥ max(4 h, τ ) , and h ≥ 2 σ (1 -γ ) . Then, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The assumption in Theorem 4 that ‖ x ( t ) ‖ v ≤ ¯ x is not necessary. In particular, it can be shown (see Proposition 5 below) that under Assumption 1 and Assumption 2, ‖ x ( t ) ‖ v can be bounded by some constant almost surely. The proof of Proposition 5 can be found in Appendix A.3. We treat the upper bound on ‖ x ( t ) ‖ v as a separate assumption because in the Q -learning case, the constant can be better than what is implied in Proposition 5.

Proposition 5 Suppose Assumptions 1 and 2 hold. Then for all t , ‖ x ( t ) ‖ v ≤ 1 1 -γ ((1 + γ ) ‖ x ∗ ‖ v + ¯ w v ) almost surely.

Theorem 4 shows that, when setting h = Θ( 1 σ (1 -γ ) ) and t 0 = Θ(max( h, τ )) , ‖ x ( T ) -x ∗ ‖ v ≤ ˜ O ( ¯ /epsilon1 √ τ (1 -γ ) 1 . 5 σ 1 √ T ) + ˜ O ( ¯ /epsilon1τ σ 2 (1 -γ ) 2 1 T ) . This means that, to get an approximation error of ε , the number of time steps required is T /greaterorsimilar ¯ /epsilon1 2 τ σ 2 (1 -γ ) 3 1 ε 2 . Compared to Shah and Xie (2018, Thm. 5), our result improves the dependence on 1 ε . Note that Wainwright (2019a) does not provide an explicit approximation bound for the SA scheme, but state the bounds in the context of Q -learning instead. For this reason, we compare to Wainwright (2019a) in the context of Q -learning in Section 3.

We also comment that in the step size h t + t 0 in Theorem 4, it is important for the h constant to scale with Θ( 1 (1 -γ ) σ ) to avoid an exponential blow-up in 1 1 -γ . This fact is not apparent in the some of the earlier work like Even-Dar and Mansour (2003), but has been pointed out recently (Jin et al., 2018; Wainwright, 2019a). Specifically, Wainwright (2019a) shows that h needs to grow with 1 1 -γ in the synchronous SA setting. Our result is consistent with Wainwright (2019a) and further shows

that in the asynchronous setting, h also needs to scale with 1 σ . If we interpret σ as the fraction of times that each state is visited, then such scaling in 1 σ will result in step size of Θ( 1 σt ) , which is similar in spirit to a common practice in asynchronous Q -learning, where the step size is coordinate dependent, α t = Θ( 1 N t i t ) instead of Θ( 1 t ) , where N t i t means the number of times i t has been visited up to time t .

## 3. Application to Q -learning

We now apply the results for SA to the important special case of Q -learning. The setting we study is defined as follows. We consider a γ -discounted infinite horizon Markov Decision Process (MDP) with finite state space S and finite action space A . Our SA result applies to both the discounted ( γ &lt; 1 ) and undiscounted ( γ = 1 ) case. For the connection between the undiscounted case Q -learning and the SA scheme with the weighted infinity norm, see e.g. Tsitsiklis (1994). For ease of presentation, we focus on the discounted case ( γ &lt; 1 ), where we can let the norm be the standard infinity norm ‖ · ‖ ∞ , i.e., v is the all-one vector.

Let the transition probability of the MDP be given by P ( s t +1 = s ′ | s t = s, a t = a ) = P ( s ′ | s, a ) . At time t , conditioned on the current state s t and action a t , the stage reward is a random variable r t independently drawn from some fixed distribution depending on ( s t , a t ) , with its expectation given by r s t ,a t , where r ∈ R S×A is a deterministic vector. A policy π : S → ∆( A ) , s ↦→ π ( ·| s ) maps the state space to the probability simplex on the action space ∆( A ) , and under the policy, a t is drawn from π ( ·| s t ) . Given a policy π , the Q table Q π : R S×A under this policy is,

<!-- formula-not-decoded -->

where E π means the expectation is taken with a t drawn from π ( ·| s t ) . The MDP problem seeks to find an optimal policy π ∗ such that Q π ( s, a ) is maximized simultaneously for all ( s, a ) . Classical MDP theory (Bertsekas and Tsitsiklis, 1996) guarantees that such a π ∗ must exist and, further, the resulting Q -function, which we denote as Q ∗ , is the unique fixed point of the Bellman Operator F : R S×A → R S×A given by,

<!-- formula-not-decoded -->

Once Q ∗ is known, an optimal policy can be easily determined (Bertsekas and Tsitsiklis, 1996).

When the transition probabilities and the rewards are unknown, we cannot directly use (3) to calculate Q ∗ . The Q -learning algorithm is an off-policy learning algorithm that approximates Q ∗ . In the asynchronous version of Q -learning, we sample a trajectory { ( s t , a t , r t ) } ∞ t =0 by taking a behavioral policy π . In this process, we maintain a Q table Q ( t ) , which is initialized with Q (0) being the all-zero table, and is updated upon observing every new state action pair ( s t +1 , a t +1 ) using the following update rule,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash

Our results make the following standard assumptions regarding the MDP. Assumption 4(a) is an upper bound on the reward, and Assumption 4(b) is to ensure the sufficient exploration condition in Assumption 3 holds (cf. Proposition 3). 4 In the asynchronous Q -learning literature, it is common to require some type of sufficient exploration assumption. Assumption 4(b) is more general than the i.i.d. assumption in Szepesv´ ari (1998); Lee and He (2019), and is similar in spirit to the covering time assumption in Even-Dar and Mansour (2003) and another related assumption in Beck and Srikant (2012).

## Assumption 4 The following conditions hold.

- (a) For all t , the stage reward r t is upper bounded, | r t | ≤ ¯ r almost surely.
- (b) Under the behavioral policy π , the induced Markov chain with state ( s t , a t ) is ergodic, has a stationary distribution µ and mixing time t MIX . Further, define µ min = inf s,a µ s,a &gt; 0 .

We now show that under this assumption, the Q -learning updates (4) and (5) can be written in the form of (1) and (2) and meet Assumptions 1, 2, 3. We first identify N = S × N , i t = ( s t , a t ) , and Q ( t ) with x ( t ) . We let F t be the σ -algebra generated by ( s 0 , a 0 , r 0 , . . . , s t -1 , a t -1 , r t -1 , s t , a t ) . Then, clearly ( s t , a t ) is F t measurable. We also define

<!-- formula-not-decoded -->

Then, (4) can be written as,

<!-- formula-not-decoded -->

which shows the Q -learning algorithm (4) and (5) can be written in the form of (1) and (2). We then check Assumptions 1, 2, 3. For Assumption 1, it is known that the Bellman Operator F is a γ -contraction in infinity norm (Tsitsiklis, 1994); further, it easy to check ‖ F ( Q ) ‖ ∞ ≤ ¯ r + γ ‖ Q ‖ ∞ , and hence Assumption 1 is met with C = ¯ r . For Assumption 2, clearly w ( t ) is F t +1 -measurable, and satisfies E w ( t ) |F t = 0 . For the boundedness of w ( t ) , we have the following proposition, which completes the verification of Assumption 2. The proof of Proposition 6 can be found in Appendix A.4.

Proposition 6 Under Assumption 4, the Q -learning update satisfies the following. (a) For all t , ‖ Q ( t ) ‖ ∞ ≤ ¯ x := ¯ r 1 -γ almost surely; also, ‖ Q ∗ ‖ ∞ ≤ ¯ x . (b) For all t , | w ( t ) | ≤ ¯ w := 2¯ r 1 -γ almost surely.

Finally, using Assumption 4(b) and Proposition 3, we have that Assumption 3 holds with σ = 1 2 µ min and τ = /ceilingleft log 2 2 µ min /ceilingright t MIX .

Combining the three assumptions together with the upper bound on ‖ Q ( t ) ‖ ∞ in Proposition 6(a), we can directly apply Theorem 4 and obtain the following finite-time error bounds for Q -learning.

4. Assumption 4(b) is a simple sufficient condition that leads to Assumption 3, but it is not necessary. For example, Assumption 3 does not even require the exploratory policy to be stationary.

Theorem 7 Suppose Assumption 4 holds and the step size is taken to be α t = h t + t 0 with t 0 ≥ max(4 h, /ceilingleft log 2 2 µ min /ceilingright t MIX ) and h ≥ 4 µ min (1 -γ ) . Then, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

From the above theorem, if we take h = Θ( 1 µ min (1 -γ ) ) , t 0 = ˜ Θ(max( 1 µ min (1 -γ ) , t MIX )) , the convergence rate becomes ˜ O ( ¯ r √ t MIX (1 -γ ) 5 / 2 µ min 1 √ T + ¯ rt MIX (1 -γ ) 3 µ 2 min 1 T ) . Therefore, to reach a ε accuracy in infinity norm, it takes T /greaterorsimilar ¯ r 2 t MIX (1 -γ ) 5 µ 2 min 1 ε 2 iterations. This bound matches the best known dependence on 1 1 -γ and 1 ε in synchronous Q -learning (Wainwright, 2019a). The extra factor t MIX µ 2 min is a result of the asynchronous updates. If we interpret 1 µ min to scale with |S| × |A| (the state-action space size), the extra factor becomes t MIX ( |S| × |A| ) 2 . We believe the scaling in t MIX is inevitable. When compared with the results on asynchronous Q -learning, to the best of our knowledge, the best finite-time bound is that of Even-Dar and Mansour (2003, Thm. 4), where the scaling is ( |S||A| ) 5 (1 -γ ) 5 /epsilon1 2 . 5 when ω = 4 / 5 (optimizing dependence on 1 1 -γ ), or ( |S||A| ) 3 . 3 (1 -γ ) 5 . 2 ε 2 . 6 when ω = 0 . 77 (optimizing dependence on |S||A| ). 5 Here ω is a step size parameter in Even-Dar and Mansour (2003). While our result improves the dependence on 1 ε , 1 1 -γ , ( |S||A| ) over that of Even-Dar and Mansour (2003), we believe our square dependence on the state-action space size is not optimal. We leave it as future work to investigate whether this is an intrinsic property of the algorithm or it is an artifact of the proof.

## 4. Convergence Proof

In this section, we prove our main result, Theorem 4. The proof is divided into three steps. In the first step, we manipulate the update equation ((1) and (2)) and decompose the error in a recursive form, which provides a transparent view of how the stochastic noise affects the error. In the second step, we bound the contribution of the noise sequence to the error decomposition. In the third step, we use the error decomposition and the noise sequence bounds to prove the result.

Step 1: Decomposition of Error. Let e i to be the unit vector (the i 'th entry is 1 and others are zero). We let D t = E e i t e /latticetop i t |F t -τ . Then, it is clear D t is a F t -τ -measurable n -byn diagonal random matrix, with its i 'th entry being d t,i = P ( i t = i |F t -τ ) . By Assumption 3, we have

<!-- formula-not-decoded -->

With these definitions, we can rewrite the update equation (1) and (2) as follows,

<!-- formula-not-decoded -->

5. Notably, Even-Dar and Mansour (2003) uses a different assumption on sufficient exploration.

<!-- formula-not-decoded -->

Clearly, x ( t ) is F t measurable and /epsilon1 ( t ) is F t +1 measurable (as /epsilon1 ( t ) depends on w ( t ) , which is F t +1 measurable). Further,

<!-- formula-not-decoded -->

In other words, /epsilon1 ( t ) is like a 'shifted' martingale difference sequence, where here 'shifted' means the conditioning in (8) is with respect to F t -τ instead of F t as would be the case in a standard martingale difference sequence. Property (8) will be useful later in the proof. For now, we focus on (7) and expand it recursively, getting,

<!-- formula-not-decoded -->

where we have defined, B k,t = α k D k ∏ t /lscript = k +1 ( I -α /lscript D /lscript ) , ˜ B k,t = ∏ t /lscript = k +1 ( I -α /lscript D /lscript ) . Clearly, B k,t and ˜ B k,t are n -byn diagonal random matrices, with the i 'th diagonal entry given by b k,t,i and ˜ b k,t,i , where b k,t,i = α k d k,i ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) and ˜ b k,t,i = ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) . So, for any i ,

<!-- formula-not-decoded -->

Also, by (6), we have for any i , almost surely

<!-- formula-not-decoded -->

With these preparations, we are ready to state the following Lemma, which decomposes the error ‖ x ( t ) -x ∗ ‖ v in a recursive form. The proof of Lemma 8 can be found in Appendix B.1.

Lemma 8 Let a t = ‖ x ( t ) -x ∗ ‖ v , we have almost surely,

<!-- formula-not-decoded -->

Step 2: Bounding ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v and ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v . We start with a bound on each individual /epsilon1 ( k ) and φ ( k ) in the following lemma, proven in Appendix B.2.

From Lemma 8, it is clear that to control the error a t , we need to bound ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v and ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v , which will be the focus of the next step.

Lemma 9 The following bounds hold almost surely. (a) ‖ /epsilon1 ( t ) ‖ v ≤ ¯ /epsilon1 := 2¯ x + C + ¯ w v . (b) ‖ φ ( t ) ‖ v ≤ ∑ t k = t -τ +1 2¯ /epsilon1α k -1 .

To bound ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v and ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v , we also need to understand the behavior of α k and ˜ B k,t . Recall that, by (11), each entry of B k,t and ˜ B k,t are upper bounded by β k,t and ˜ β k,t respectively. We now provide the following results on the sequence β k,t , ˜ β k,t which we will frequently use later to control α k ˜ B k,t . The proof of Lemma 10 is provided in Appendix B.3.

Lemma 10 If α t = h t + t 0 , where h &gt; 2 σ and t 0 ≥ max(4 h, τ ) , then β k,t , ˜ β k,t satisfies the following.

- (a) β k,t ≤ h k + t 0 ( k +1+ t 0 t +1+ t 0 ) σh , ˜ β k,t ≤ ( k +1+ t 0 t +1+ t 0 ) σh .
- (c) ∑ t k = τ β k,t ∑ k /lscript = k -τ +1 α /lscript -1 ≤ 8 hτ σ 1 t +1+ t 0 .
- (b) ∑ t k =1 β 2 k,t ≤ 2 h σ 1 ( t +1+ t 0 ) .

We are now ready to bound ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v and ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v . Our bound on ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v is an immediate consequence of Lemma 9 (b) and Lemma 10 (c).

Lemma 11 The following inequality holds almost surely,

<!-- formula-not-decoded -->

Proof Wehave ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v ≤ ∑ t k = τ α k ‖ ˜ B k,t ‖ v ‖ φ ( k ) ‖ v ≤ ∑ t k = τ β k,t ∑ k /lscript = k -τ +1 2¯ /epsilon1α /lscript -1 ≤ 16¯ /epsilon1hτ σ ( t + t 0 +1) . Here we have used by Proposition 2, ‖ ˜ B k,t ‖ v = sup i | ˜ b k,t,i | ≤ ˜ β k,t .

Lemma 12 For each t , with probability at least 1 -δ , we have,

We now focus on proving Lemma 12. Recall /epsilon1 ( t ) is F t +1 measurable is a 'shifted' martingale difference sequence in the sense that E /epsilon1 ( t ) |F t -τ = 0 (cf. (8)). We will use a variant of the AzumaHoeffding bound in Lemma 13 that handles our 'shifted' Martingale difference sequence. The proof of Lemma 13 is postponed to Appendix B.4.

<!-- formula-not-decoded -->

Lemma 13 Let X t be a F t -adapted stochastic process, satisfying E X t |F t -τ = 0 . Further, | X t | ≤ ¯ X t almost surely. Then with probability 1 -δ , we have, | ∑ t k =0 X k | ≤ √ 2 τ ∑ t k =0 ¯ X 2 k log( 2 τ δ ) .

<!-- formula-not-decoded -->

To prove Lemma 12, recall that ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) is a random vector in R N , with its i 'th entry

with d /lscript,i ≥ σ almost surely, cf. (6). Fixing i , as have been shown in (8), /epsilon1 i ( k ) is a F k +1 adapted stochastic process satisfying E /epsilon1 i ( k ) |F k -τ = 0 . However, ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) is not F k -τ -measurable, and as such we cannot directly apply the Azuma-Hoeffding bound in Lemma 13 to (12). To proceed, we need to get rid of the randomness of ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) in the summation (12).This is done in Lemma 14 which shows that the absolute value of quantity (12) can be upper bounded by the sup of another quantity where the randomness caused by ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) is removed through the use of d /lscript,i ≥ σ , and to this new quantity we can directly apply Lemma 13. The proof of Lemma 14 is postponed to Appendix B.5.

Lemma 14 For each i , we have almost surely,

<!-- formula-not-decoded -->

With the help of Lemma 14, we use the Azuma-Hoeffding bound to prove Lemma 12.

Proof of Lemma 12. Fix i and τ ≤ k 0 ≤ t . As have been shown in (8), 1 v i /epsilon1 i ( k ) β k,t is a F k +1 adapted stochastic process satisfying E 1 v i /epsilon1 i ( k ) β k,t |F k -τ = 0 . Also by Lemma 9(a), | 1 v i /epsilon1 i ( k ) β k,t | ≤ ¯ /epsilon1β k,t almost surely. As a result, we can use the Azuma-Hoeffding bound in Lemma 13 to get with probability 1 -δ ,

By a union bound on τ ≤ k 0 ≤ t , we get with probability 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, by Lemma 14, we have with probability 1 -δ ,

<!-- formula-not-decoded -->

where in the third inequality, we have used the bounds on β k,t in Lemma 10. Finally, applying the union bound over i ∈ N will lead to the desired result. /squaresolid

Step 3: Bounding the error sequence. We are now ready to use the error decomposition in Lemma 8 and the bound on ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v and ‖ ∑ t k = τ α k ˜ B k,t φ ( k ) ‖ v in Lemma 12 and Lemma 11 to bound a t = ‖ x ( t ) -x ∗ ‖ v . Recall, we want to show that, with probability 1 -δ ,

<!-- formula-not-decoded -->

where C a = 12¯ /epsilon1 1 -γ √ ( τ +1) h σ log( 2( τ +1) T 2 n δ ) , C ′ a = 4 1 -γ max( C φ , 2¯ x ( τ + t 0 )) . To prove (13), we start by applying Lemma 12 to t ≤ T with δ replaced by δ/T . Then, using a union bound, we get with probability 1 -δ , for any t ≤ T , ‖ ∑ t k = τ α k ˜ B k,t /epsilon1 ( k ) ‖ v ≤ C /epsilon1 1 √ t +1+ t 0 , where C /epsilon1 = 6¯ /epsilon1 √ ( τ +1) h σ log( 2( τ +1) T 2 n δ ) . Combine the above with Lemma 8 and use Lemma 11, we get with probability 1 δ , for all τ t T ,

<!-- formula-not-decoded -->

-≤ ≤

We now condition on (14) and use induction to show (13). Eq. (13) is true for t = τ , as C ′ a τ + t 0 ≥ 8 1 -γ ¯ x ≥ a τ , where we have used a τ = ‖ x ( τ ) -x ∗ ‖ v ≤ ‖ x ( τ ) ‖ v + ‖ x ∗ ‖ v ≤ 2¯ x by the definition of ¯ x . Then, assuming (13) is true for up to k ≤ t , we have by (14),

<!-- formula-not-decoded -->

We use the following auxiliary Lemma, whose proof is provided in Appendix B.6.

Lemma 15 Recall α k = h k + t 0 , and b k,t,i = α k d k,i ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) , here d k,i ≥ σ . If σh (1 - √ γ ) ≥ 1 , t 0 ≥ 1 , and α 0 ≤ 1 2 , then, for any i ∈ N , and any 0 &lt; ω ≤ 1 , we have ∑ t k = τ b k,t,i 1 ( k + t 0 ) ω ≤ 1 √ γ ( t +1+ t 0 ) ω .

<!-- formula-not-decoded -->

With Lemma 15, and using the bound on ˜ β τ -1 ,t in Lemma 10 (a), we have

To finish the induction, it suffices to show F t ≤ C a √ t +1+ t 0 and F ′ t ≤ C ′ a t +1+ t 0 . To see this,

<!-- formula-not-decoded -->

It suffices to show that, C /epsilon1 C a ≤ 1 - √ γ , C φ C ′ a ≤ 1 - √ γ 2 , and a τ ( τ + t 0 ) C ′ a ≤ 1 - √ γ 2 . Using a τ ≤ 2¯ x , one can check that C a and C ′ a satisfy the above three inequalities, which concludes the proof. /squaresolid

## References

- Mohammad Gheshlaghi Azar, R´ emi Munos, Mohammad Ghavamzadeh, and Hilbert Kappen. Speedy Q-learning. In Advances in Neural Information Processing Systems , 2011.
- Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3): 325-349, 2013.
- Carolyn L Beck and Rayadurgam Srikant. Error bounds for constant step-size Q-learning. Systems &amp;control letters , 61(12):1203-1208, 2012.
- Albert Benveniste, Michel M´ etivier, and Pierre Priouret. Adaptive algorithms and stochastic approximations , volume 22. Springer Science &amp; Business Media, 2012.
- Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming , volume 5. Athena Scientific Belmont, MA, 1996.
- Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint , volume 48. Springer, 2009.
- Zaiwei Chen, Sheng Zhang, Thinh T Doan, Siva Theja Maguluri, and John-Paul Clarke. Performance of Q-learning with linear function approximation: Stability and finite-time analysis. arXiv preprint arXiv:1905.11425 , 2019.
- Adithya M Devraj and Sean Meyn. Zap Q-learning. In Advances in Neural Information Processing Systems , pages 2235-2244, 2017.
- Thinh T Doan. Finite-time analysis and restarting scheme for linear two-time-scale stochastic approximation. arXiv preprint arXiv:1912.10583 , 2019.
- Kefan Dong, Yuanhao Wang, Xiaoyu Chen, and Liwei Wang. Q-learning with UCB exploration is sample efficient for infinite-horizon MDP. arXiv preprint arXiv:1901.09311 , 2019.
- Eyal Even-Dar and Yishay Mansour. Learning rates for Q-learning. Journal of machine learning Research , 5(Dec):1-25, 2003.
- Harsh Gupta, R Srikant, and Lei Ying. Finite-time performance bounds and adaptive learning rate selection for two time-scale reinforcement learning. In Advances in Neural Information Processing Systems , pages 4706-4715, 2019.
- Hado V Hasselt. Double Q-learning. In Advances in neural information processing systems , pages 2613-2621, 2010.
- Tommi Jaakkola, Michael I Jordan, and Satinder P Singh. Convergence of stochastic iterative dynamic programming algorithms. In Advances in neural information processing systems , pages 703-710, 1994.
- Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873, 2018.

- Michael J Kearns and Satinder P Singh. Finite-sample convergence rates for Q-learning and indirect algorithms. In Advances in neural information processing systems , pages 996-1002, 1999.
- Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in neural information processing systems , pages 1008-1014, 2000.
- Vijay R Konda and John N Tsitsiklis. Linear stochastic approximation driven by slowly varying markov chains. Systems &amp; control letters , 50(2):95-102, 2003.
- Donghwan Lee and Niao He. A unified switching system perspective and ODE analysis of Qlearning algorithms. arXiv preprint arXiv:1912.02270 , 2019.
- David A Levin and Yuval Peres. Markov chains and mixing times , volume 107. American Mathematical Soc., 2017.
- Francisco S Melo, Sean P Meyn, and M Isabel Ribeiro. An analysis of reinforcement learning with function approximation. In Proceedings of the 25th international conference on Machine learning , pages 664-671, 2008.
- Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics , pages 400-407, 1951.
- Devavrat Shah and Qiaomin Xie. Q-learning with nearest neighbors. In Advances in Neural Information Processing Systems , pages 3111-3121, 2018.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196, 2018a.
- Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM, 2018b.
- RSrikant and Lei Ying. Finite-time error bounds for linear stochastic approximation and td learning. arXiv preprint arXiv:1902.00923 , 2019.
- Alexander L Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L Littman. PAC modelfree reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888, 2006.
- Csaba Szepesv´ ari. The asymptotic convergence-rate of Q-learning. In Advances in Neural Information Processing Systems , pages 1064-1070, 1998.
- John N Tsitsiklis. Asynchronous stochastic approximation and Q-learning. Machine learning , 16 (3):185-202, 1994.
- John N Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function approximation. In Advances in neural information processing systems , pages 1075-1081, 1997.
- Martin J Wainwright. Stochastic approximation with cone-contractive operators: Sharp /lscript i nfty -bounds for Q-learning. arXiv preprint arXiv:1905.06265 , 2019a.

- Martin J Wainwright. Variance-reduced q -learning is minimax optimal. arXiv preprint arXiv:1906.04697 , 2019b.
- Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning , 8(3-4):279-292, 1992.
- Chen-Yu Wei, Mehdi Jafarnia-Jahromi, Haipeng Luo, Hiteshi Sharma, and Rahul Jain. Modelfree reinforcement learning in infinite-horizon average-reward markov decision processes. arXiv preprint arXiv:1910.07072 , 2019.
- Tengyu Xu, Shaofeng Zou, and Yingbin Liang. Two time-scale off-policy td learning: Nonasymptotic analysis over markovian samples. In Advances in Neural Information Processing Systems , pages 10633-10643, 2019.

## Appendix A. Proofs of Auxiliary Propositions in Section 2 and Section 3

## A.1. Proof of Proposition 2

Let x ∈ R N be any vector s.t. ‖ x ‖ v = 1 . Then,

<!-- formula-not-decoded -->

As a result, ‖ A ‖ v ≤ sup i ∈N ∑ j ∈N | a ij | v j v i . On the other hand, let i ∗ = arg max i ∈N ∑ j ∈N | a ij | v j v i (ties broken arbitrarily). And we set x = [ x 1 , . . . , x n ] /latticetop with x j = v j sign ( a i ∗ j ) , where sign ( z ) = 1 when z ≥ 0 , and -1 otherwise. Then, clearly ‖ x ‖ v = 1 , and

<!-- formula-not-decoded -->

## A.2. Proof of Proposition 3

This shows ‖ A ‖ v ≥ sup i ∈N ∑ j ∈N | a ij | v j v i and finishes the proof.

Let d be the distribution of i t conditioned on F t -τ . Then, by Levin and Peres (2017, eq. (4.33)),

<!-- formula-not-decoded -->

where TV means the total-variation distance. As a result, for each i ∈ N , d i ≥ µ i - | µ i -d i | ≥ µ min -TV ( d, µ ) ≥ 1 2 µ min . This shows that for any i , P ( i t = i |F t -τ ) ≥ 1 2 µ min which verifies Assumption 3. /squaresolid

## A.3. Proof of Proposition 5

Note that by Assumption 1(a), we have,

<!-- formula-not-decoded -->

/squaresolid

In other words, Assumption 1(b) holds with C = (1 + γ ) ‖ x ∗ ‖ v . Let ¯ x = 1 1 -γ ((1 + γ ) ‖ x ∗ ‖ v + ¯ w v ) . We prove ‖ x ( t ) ‖ v ≤ ¯ x by induction. The statement is obviously true for t = 0 as x (0) is initialized to be the all-zero vector. Suppose it is true for t , then

<!-- formula-not-decoded -->

Then, notice that,

<!-- formula-not-decoded -->

where in the second inequality, we have used | w ( t ) | ≤ ¯ w almost surely (cf. Assumption 2), and in the last equality, we have used that γ ¯ x + C + ¯ w v = ¯ x . This finishes the induction. /squaresolid

## A.4. Proof of Proposition 6

Weprove ‖ Q ( t ) ‖ ∞ ≤ ¯ r 1 -γ by induction. Firstly, the statement is true for t = 0 as Q (0) is initialized to be the all zero table. Then, assume the statement is true for t . For t +1 , clearly ‖ Q ( t +1) ‖ ∞ ≤ max( ‖ Q ( t ) ‖ ∞ , | Q s t ,a t ( t +1) | ) . Further, notice,

<!-- formula-not-decoded -->

This finishes the induction, and hence ‖ Q ( t ) ‖ ∞ ≤ ¯ r 1 -γ almost surely for all t ≥ 0 . As Q ∗ is the Q -function under an optimal policy π ∗ , we get for any s ∈ S , a ∈ A ,

<!-- formula-not-decoded -->

which concludes the proof of part (a). For part (b), notice,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which finishes the proof of part (b).

## Appendix B. Proofs of Auxiliary Lemmas in Section 4

## B.1. Proof of Lemma 8 (Error Decomposition)

By (9), we have,

<!-- formula-not-decoded -->

Notice that by (10), for each i , ˜ b τ -1 ,t,i + ∑ t k = τ b k,t,i = 1 . Then, for each i , we have

<!-- formula-not-decoded -->

where in the last inequality, we have used that F is γ -contraction in ‖ · ‖ v with fixed point x ∗ . Combining the above with (15), we have,

<!-- formula-not-decoded -->

## B.2. Proof of Lemma 9 (Bounds on ‖ /epsilon1 ( t ) ‖ v and ‖ φ ( t ) ‖ v )

For part (a), we have,

<!-- formula-not-decoded -->

/squaresolid

where we have used by Proposition 2, ‖ e i t e /latticetop i t -D t ‖ v = sup i | 1 ( i t = i ) -d t,i | ≤ 1 (here 1 is the indicator function); and ‖ F ( x ( t -τ )) ‖ v ≤ γ ‖ x ( t -τ ) ‖ v + C ≤ ¯ x + C . For part (b), we have,

<!-- formula-not-decoded -->

Notice that ‖ x ( t ) -x ( t -1) ‖ v ≤ α t -1 ( ‖ F ( x ( t -1)) ‖ v + ‖ x ( t -1) ‖ v + 1 v ¯ w ) ≤ α t -1 (2¯ x + C + 1 v ¯ w ) = α t -1 ¯ /epsilon1 . Summing up, we get

/squaresolid

<!-- formula-not-decoded -->

## B.3. Proof of Lemma 10 (Step Sizes)

For part (a), notice that log(1 -x ) ≤ -x for all x &lt; 1 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, which leads to the bound on β k,t and ˜ β k,t .

For part (b),

<!-- formula-not-decoded -->

where we have used ( k +1+ t 0 ) 2 σh ≤ 2( k + t 0 ) 2 σh , which is true when t 0 ≥ 4 h . Then,

<!-- formula-not-decoded -->

where in the last inequality we have used 2 σh -1 &gt; σh .

For part (c), notice that for k -τ ≤ /lscript ≤ k -1 where k ≥ τ , we have α /lscript ≤ h k -τ + t 0 ≤ 2 h k + t 0 (using t 0 ≥ τ ). Then,

<!-- formula-not-decoded -->

where we have used ( k +1+ t 0 ) σh ≤ 2( k + t 0 ) σh , and σh -1 &gt; 1 2 σh . /squaresolid

## B.4. Proof of Lemma 13 (Azuma Hoeffding)

Let /lscript be an integer between 0 and τ -1 . For each /lscript , define process Y /lscript k = X τk + /lscript , scalar ¯ Y /lscript k = ¯ X kτ + /lscript , and define Filtration ˜ F /lscript k = F τk + /lscript . Then, Y /lscript k is ˜ F /lscript k -adapted, and satisfies

<!-- formula-not-decoded -->

Therefore, applying Azuma-Hoeffding bound on Y /lscript k , we have

<!-- formula-not-decoded -->

i.e. with probability at least 1 -δ τ ,

<!-- formula-not-decoded -->

Using the union bound for /lscript = 0 , . . . , τ -1 , we get that with probability at least 1 -δ , where the last inequality is due to Cauchy-Schwarz.

<!-- formula-not-decoded -->

## B.5. Proof of Lemma 14

Let p k be a scalar sequence defined as follows. Set p τ = 0 , and

<!-- formula-not-decoded -->

Then p t +1 = ∑ t k = τ α k /epsilon1 i ( k ) ∏ t /lscript = k +1 (1 -α /lscript d /lscript,i ) , and to prove Lemma 14 we need to bound | p t +1 | . Let

<!-- formula-not-decoded -->

/squaresolid

We must have k 0 ≥ τ since | p τ | = 0 . With k 0 defined, we now define another scalar sequence ˜ p s.t. ˜ p k 0 +1 = p k 0 +1 and

<!-- formula-not-decoded -->

We claim that for all k ≥ k 0 +1 , p k and ˜ p k have the same sign, and | p k | ≤ | ˜ p k | . This is obviously true for k = k 0 +1 . Suppose it is true for for k -1 . Without loss of generality, suppose both p k -1 and ˜ p k -1 are non-negative. Since k -1 &gt; k 0 and by the definition of k 0 , we must have

<!-- formula-not-decoded -->

Therefore, p k &gt; 0 . Further, since d k -1 ,i ≥ σ , we also have

<!-- formula-not-decoded -->

These imply ˜ p k ≥ p k &gt; 0 . The case where both p k -1 and ˜ p k -1 are negative is similar. This finishes the induction, and as a result, | p t +1 | ≤ | ˜ p t +1 | . Notice,

<!-- formula-not-decoded -->

By the definition of k 0 , we have

<!-- formula-not-decoded -->

where in the last step, we have used the upper bound on ‖ /epsilon1 ( k 0 ) ‖ v in Lemma 9 (a). As a result,

<!-- formula-not-decoded -->

## B.6. Proof of Lemma 15

Throughout the proof, we fix i and will frequently use the property d k,i ≥ σ which holds almost surely. Define the sequence

<!-- formula-not-decoded -->

We use induction to show that e t ≤ 1 √ γ ( t +1+ t 0 ) ω . The statement is clearly true for t = τ , as e τ = b τ,τ,i 1 ( τ + t 0 ) ω = α τ d τ,i 1 ( τ + t 0 ) ω ≤ 1 √ γ ( τ +1+ t 0 ) ω (the last step needs α τ ≤ 1 2 , (1 + 1 t 0 ) ω ≤ 2 √ γ , implied by t 0 ≥ 1 , ω ≤ 1 ). Let the statement be true for t -1 . Then, notice that,

<!-- formula-not-decoded -->

/squaresolid

<!-- formula-not-decoded -->

where the inequality is based on induction assumption. Then, plug in α t = h t + t 0 and use d t,i ≥ σ , we have,

<!-- formula-not-decoded -->

Now using the inequality that for any x &gt; -1 , (1 + x ) ≤ e x , we have,

<!-- formula-not-decoded -->

where in the last inequality, we have used ω ≤ 1 and the condition on h s.t. σh (1 - √ γ ) ≥ 1 . This shows e t ≤ 1 √ γ ( t +1+ t 0 ) ω and finishes the induction. /squaresolid