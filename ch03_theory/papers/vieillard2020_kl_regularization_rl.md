## Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning

## Nino Vieillard

Google Research, Brain Team Université de Lorraine, CNRS, Inria IECL, F-54000 Nancy, France vieillard@google.com

## Bruno Scherrer

Okinawa Institute of Science and Technology tadashi.kozuno@oist.jp

Université de Lorraine, CNRS, Inria IECL, F-54000 Nancy, France bruno.scherrer@inria.fr

Rémi Munos DeepMind munos@google.com

Matthieu Geist Google Research, Brain Team mfgeist@google.com

## Abstract

Recent Reinforcement Learning (RL) algorithms making use of KullbackLeibler (KL) regularization as a core component have shown outstanding performance. Yet, only little is understood theoretically about why KL regularization helps, so far. We study KL regularization within an approximate value iteration scheme and show that it implicitly averages q -values. Leveraging this insight, we provide a very strong performance bound, the very first to combine two desirable aspects: a linear dependency to the horizon (instead of quadratic) and an error propagation term involving an averaging effect of the estimation errors (instead of an accumulation effect). We also study the more general case of an additional entropy regularizer. The resulting abstract scheme encompasses many existing RL algorithms. Some of our assumptions do not hold with neural networks, so we complement this theoretical analysis with an extensive empirical study.

## 1 Introduction

In Reinforcement Learning (RL), Kullback-Leibler (KL) regularization consists in penalizing a new policy from being too far from the previous one, as measured by the KL divergence. It is at the core of efficient deep RL algorithms, such as Trust Region Policy Optimization (TRPO) [37] (motivated by trust region constraints) or Maximum a Posteriori Policy Optimization (MPO) [2] (arising from the view of control as probabilistic inference [26, 16]), but without much theoretical guarantees. Recently, Geist et al. [20] have analyzed algorithms operating in the larger scope of regularization by Bregman divergences. They concluded that regularization doesn't harm in terms of convergence, rate of convergence, and propagation of errors, but these results are not better than the corresponding ones in unregularized approximate dynamic programming (ADP).

∗ Work done while at DeepMind.

## Tadashi Kozuno ∗

## Olivier Pietquin

Google Research, Brain Team pietquin@google.com

Building upon their formalism, we show that using a KL regularization implicitly averages the successive estimates of the q -function in the ADP scheme. Leveraging this insight, we provide a strong performance bound, the very first to combine two desirable aspects: 1) it has a linear dependency to the time horizon (1 -γ ) -1 , 2) it exhibits an error averaging property of the KL regularization. The linear dependency in the time horizon contrasts with the standard quadratic dependency of usual ADP, which is tight [35]. The only approaches achieving a linear dependency we are aware of make use of non-stationary policies [8, 35] and never led to practical deep RL algorithms. More importantly, the bound involves the norm of the average of the errors, instead of a discounted sum of the norms of the errors for classic ADP. This means that, while standard ADP is not guaranteed to converge for the ideal case of independent and centered errors, KL regularization allows convergence to the optimal policy in that case. The sole algorithms that also enjoy this compensation of errors are Dynamic Policy Programming (DPP) [7] and Speedy Q-learning (SQL) [6], that also build (implicitly) on KL regularization, as we will show for SQL. However, their dependency to the horizon is quadratic, and they are not well amenable to a deep learning setting [43].

We also study the case of an additional entropy regularization, usual in practical algorithms, and specifically the interplay between both regularizations. The resulting abstract framework encompasses a wide variety of existing RL algorithms, the connections between some of them being known [20], but many other being new, thanks to the implicit average of q -values. We highlight that, even though our analysis covers the case where only the entropy regularization is considered, it does not explain why it helps without an additional KL term. Some argue that having a higher entropy helps exploration [38], other that it has beneficial effects on the optimization landscape [3], but it also biases the solution of the MDP [20].

Our analysis requires some assumptions, notably that the regularized greedy step is done without approximation. If this is reasonable with discrete actions and a linear parameterization, it does not hold when neural networks are considered. Given their prevalence today, we complement our thorough analysis with an extensive empirical study, that aims at observing the core effect of regularization in a realistic deep RL setting.

## 2 Background and Notations

Let ∆ X be the set of probability distributions over a finite set X and Y X the set of applications from X to the set Y . An MDP is a tuple {S , A , P, r, γ } with S the finite state space, A the finite set of actions, P ∈ ∆ S×A S the Markovian transition kernel, r ∈ R S×A the reward function bounded by r max , and γ ∈ (0 , 1) the discount factor. For τ ≥ 0, we write v τ max = r max + τ ln |A| 1 -γ and simply v max = v 0 max . We write 1 ∈ R S×A the vector whose components are all equal to 1. A policy π ∈ ∆ S A associates a distribution over actions to each state. Its (state-action) value function is defined as q π ( s, a ) = E π [ ∑ ∞ t =0 γ t r ( S t , A t ) | S 0 = s, A 0 = a ], E π being the expectation over trajectories induced by π . Any optimal policy satisfies π ∗ ∈ argmax π ∈ ∆ S A q π (all scalar operators applied on vectors should be understood point-wise), and q ∗ = q π ∗ . The following notations will be useful. For f 1 , f 2 ∈ R S×A , 〈 f 1 , f 2 〉 = ( ∑ a f 1 ( s, a ) f 2 ( s, a )) s ∈ R S . This will be used with q -values and (log) policies. We write P π the stochastic kernel induced by π , and for q ∈ R S×A we have P π q = ( ∑ s ′ P ( s ′ | s, a ) ∑ a ′ π ( a ′ | s ′ ) q ( s ′ , a ′ )) s,a ∈ R S×A . For v ∈ R S , we also define Pv = ( ∑ s ′ P ( s ′ | s, a ) v ( s ′ )) s,a ∈ R S×A , hence P π q = P 〈 π, q 〉 .

The Bellman evaluation operator is T π q = r + γP π q , its unique fixed point being q π . The set of greedy policies w.r.t. q ∈ R S×A is G ( q ) = argmax π ∈ ∆ S A 〈 q, π 〉 . A classical approach to estimate an optimal policy is Approximate Modified Policy Iteration (AMPI) [34, 36],

<!-- formula-not-decoded -->

which reduces to Approximate Value Iteration (AVI, m = 1) and Approximate Policy Iteration (API, m = ∞ ) as special cases. The term /epsilon1 k +1 accounts for errors made when applying the Bellman operator. For example, the classic DQN [27] is encompassed by this abstract ADP scheme, with m = 1 and the error arising from fitting the neural network

(regression step of DQN). The typical use of m -step rollouts in (deep) RL actually corresponds to an AMPI scheme with m&gt; 1. Next, we add regularization to this scheme.

## 3 Regularized MPI

In this work, we consider the entropy H ( π ) = -〈 π, ln π 〉 ∈ R S and the KL divergence KL( π 1 || π 2 ) = 〈 π 1 , ln π 1 -ln π 2 〉 ∈ R S . First, we introduce a slight variation of the Mirror Descent MPI scheme [20] (handling both KL and entropy penalties, based on q -values).

Mirror Descent MPI. For q ∈ R S×A and an associated policy µ ∈ ∆ S A , we define the regularized greedy policy as G λ,τ µ ( q ) = argmax π ∈ ∆ S A ( 〈 π, q 〉 -λ KL( π || µ ) + τ H ( π )). Observe that with λ = τ = 0, we get the usual greediness. Notice also that with λ = 0, the KL term disappears, so does the dependency to µ . In this case we write G 0 ,τ . We also account for the regularization in the Bellman evaluation operator. Recall that the standard operator is T π q = r + γP 〈 π, q 〉 . Given the form of the regularized greediness, it is natural to replace the term 〈 π, q 〉 by the regularized one, giving T λ,τ π | µ q = r + γP ( 〈 π, q 〉 -λ KL( π || µ ) + τ H ( π )). These lead to the following MD-MPI( λ , τ ) scheme. It is initialized with q 0 ∈ R S×A such that ‖ q 0 ‖ ∞ ≤ v max and with π 0 the uniform policy, without much loss of generality (notice that the greedy policy is unique whenever λ &gt; 0 or τ &gt; 0):

<!-- formula-not-decoded -->

Dual Averaging MPI. We provide an equivalent formulation of scheme (1). This will be the basis of our analysis, and it also allows drawing connections to other algorithms, originally not introduced as using a KL regularization. All the technical details are provided in the Appendix, but we give an intuition here, for the case τ = 0 (no entropy). Let π k +1 = G λ, 0 π k ( q k ). This optimization problem can be solved analytically, yielding π k +1 ∝ π k exp q k λ . By direct induction, π 0 being uniform, we have π k +1 ∝ π k exp q k λ ∝ · · · ∝ exp 1 λ ∑ k j =0 q j . This means that penalizing the greedy step with a KL divergence provides a policy being a softmax over the scaled sum of all past q -functions (no matter how they are obtained). This is reminiscent of dual averaging in convex optimization, hence the name.

/negationslash

We now introduce the Dual Averaging MPI (DA-MPI) scheme. Contrary to MD-MPI, we have to distinguish the cases τ = 0 and τ = 0. DA-MPI( λ ,0) and DA-MPI( λ , τ &gt; 0) are

<!-- formula-not-decoded -->

with h 0 = q 0 for τ = 0 and h -1 = 0 for τ &gt; 0. The following result is proven in Appx. C.1. Proposition 1. For any λ &gt; 0 , MD-MPI( λ ,0) and DA-MPI( λ ,0) are equivalent (but not in the limit λ → 0 ). Moreover, for any τ &gt; 0 , MD-MPI( λ , τ ) and DA-MPI( λ , τ ) are equivalent.

Table 1: Algorithms encompassed by MD/DA-MPI (in italic if new compared to [20]).

|                         | only entropy                                                       | only KL                              | both                                          |
|-------------------------|--------------------------------------------------------------------|--------------------------------------|-----------------------------------------------|
| reg. eval. unreg. eval. | Soft Q-learning [17, 21], SAC [22], Mellowmax [5] softmax DQN [41] | DPP [7], SQL [6] TRPO [37], MPO [1], | CVI [25], AL [9, 11] softened LSPI [31], [43] |

Links to existing algorithms. Equivalent schemes (1) and (2) encompass (possibly variations of) many existing RL algorithms (see Tab. 1 and details below). Yet, we think important to highlight that many of them don't consider regularization in the evaluation step (they use T π k +1 instead of T λ,τ π k +1 | π k ), something we abbreviate as ' w/o '. If it does not

preclude convergence in the case τ = 0 [20, Thm. 4], it is known for the case τ &gt; 0 and λ = 0 that the resulting Bellman operator may have multiple fixed points [5], which is not desirable. Therefore, we only consider a regularized evaluation for the analysis, but we will compare both approaches empirically. Now, we present the approaches encompassed by scheme (1) (see also Appx. B.1). Soft Actor Critic (SAC) [22] and soft Q-learning [21] are variations of MD-MPI(0, τ ), as is softmax DQN [41] but w/o . The Mellowmax policy [5] is equivalent to MD-MPI(0, τ ). TRPO and MPO are variations of MD-MPI( λ ,0), w/o . DPP [7] is almost a reparametrization of MD-MPI( λ ,0), and Conservative Value Iteration (CVI) [25] is a reparametrization of MD-MPI 1 ( λ , τ ), which consequently also generalizes Advantage Learning (AL) [9, 11]. Next, we present the approaches encompassed by schemes (2) (see also Appx. B.2). Politex [1] is a PI scheme for the average reward case, building upon prediction with expert advice. In the discounted case, it is DA-MPI( λ ,0), w/o . Momentum Value Iteration (MoVI) [43] is a limit case of DA-MPI( λ ,0), w/o , as λ → 0, and its practical extension to deep RL momentum DQN (MoDQN) is a limit case of DA-MPI( λ , τ ), w/o . SQL [6] is a limit case of DA-MPI( λ , 0) as λ → 0. Softened LSPI [30] deals with zero-sum Markov games, but specialized to single agent RL it is a limit case of DA-MPI( λ , τ ), w/o .

## 4 Theoretical Analysis

Here, we analyze the propagation of errors of MD-MPI, through the equivalent DA-MPI, for the case m = 1 (that is regularized VI, the extension to m&gt; 1 remaining an open question). We provide component-wise bounds that assess the quality of the learned policy, depending on τ = 0 or not. From these, /lscript p -norm bounds could be derived, using [36, Lemma 5].

Analysis of DA-VI( λ ,0). This corresponds to scheme (2), left, with m = 1. The following Thm. is proved in Appx. C.2.

<!-- formula-not-decoded -->

Remark 1. The assumption ‖ q k ‖ ∞ ≤ v max is not strong. It can be enforced by simply clipping the result of the evaluation step in [ -v max , v max ] . See also Appx. C.3.

To ease the discussion, we express an /lscript ∞ -bound as a direct corollary of Thm. 1:

<!-- formula-not-decoded -->

We also recall the typical propagation of errors of AVI without regularization ( e.g. [36], we scale the sum by 1 -γ to make explicit the normalizing factor of a discounted sum):

<!-- formula-not-decoded -->

For each bound, the first term can be decomposed as a factor, the horizon term ((1 -γ ) -1 is the average horizon of the MDP), scaling the error term , that expresses how the errors made at each iteration reflect in the final performance. The second term reflects the influence of the initialization over iterations, without errors it give the rate of convergence of the algorithms. We discuss these three terms.

Rate of convergence. It is slower for DA-VI( λ ,0) than for AVI, γ k = o ( 1 k ). This was to be expected, as the KL term slows down the policy updates. It is not where the benefits of KL regularization arise. However, notice that for k small enough and γ close to 1, we may have 1 k ≤ γ k . This term has also a linear dependency to λ (through v λ max ), suggesting that a lower λ is better. This is intuitive, a larger λ leads to smaller changes of the policy, and thus to a slower convergence.

Horizon term. We have a linear dependency to the horizon, instead of a quadratic one, which is very strong. Indeed, it is known that the square dependency to the horizon is tight

for API and AVI [35]. The only algorithms based on ADP having a linear dependency we are aware of make use of non-stationary policies [35, 8], and have never led to practical (deep) RL algorithms. Minimizing directly the Bellman residual would also lead to a linear dependency ( e.g. , [32, Thm. 1]), but it comes with its own drawbacks [19] ( e.g. , bias problem with stochastic dynamics, and it is not used in deep RL, as far as we know).

Error term. For AVI, the error term is a discounted sum of the norms of the successive estimation errors, while in our case it is the norm of the average of these estimation errors. The difference is fundamental, it means that the KL regularization allows for a compensation of the errors made at each iteration. Assume that the sequence of errors is a martingale difference. AVI would not converge in this case, while DA-VI( λ , 0) converges to the optimal policy ( ‖ 1 k ∑ k j =1 /epsilon1 j ‖ ∞ converges to 0 by the law of large numbers). As far as we know, only SQL and DPP have such an error term, but they have a worse dependency to the horizon.

Thm. 1 is the first result showing that an RL algorithm can benefit from both a linear dependency to the horizon and from an averaging of the errors, and we argue that this explains, at least partially, the beneficial effect of using a KL regularization. Notice that Thm. 4 of Geist et al. [20] applies to DA-VI( λ , 0), as they study more generally MPI regularized by a Bregman divergence. Although they bound a regret rather than q ∗ -q π k , their result is comparable to AVI, with a quadratic dependency to the horizon and a discounted sum of the norms of the errors. Therefore, our result significantly improves previous analyses.

We illustrate the bound with a simple experiment 2 , see Fig. 1, left. We observe that AVI doesn't converge, while DA-VI( λ ,0) does, and that higher values of λ slow down the convergence. Yet, they are also a bit more stable. This is not explained by our bound but is quite intuitive (policies changing less between iterations).

Figure 1: Left : behavior for Thm 1. Middle : function g 2 ( k ). Right : behavior for Thm 2.

<!-- image -->

Analysis of DA-VI( λ , τ ). This is scheme (2), right, with m = 1. Due to the non-vanishing entropy term in the greedy step, it cannot converge to the unregularized optimal q -function. Yet, without errors and with λ = 0, it would converge to the solution of the MDP regularized by the scaled entropy (that is, considering the reward augmented by the scaled entropy). Our bound will show that adding a KL penalty does not change this. To do so, we introduce a few notations. The proofs of the following claims can be found in [20], for example. We already have defined the operator T 0 ,τ π . It has a unique fixed point, that we write q τ π . The unique optimal q -function is q τ ∗ = max π q τ π . We write π τ ∗ = G 0 ,τ ( q τ ∗ ) the associated unique optimal policy, and q τ π τ ∗ = q τ ∗ . The next result is proven in Appx. C.4.

Theorem 2. For a sequence of policies π 0 , . . . , π k , we define P k : j = P π k P π k -1 . . . P π j if j ≤ k , P k : j = I else. We define A 2 k : j = P k -j π τ ∗ +( I -γP π k +1 ) -1 P k : j +1 ( I -γP π j ) . We define g 2 ( k ) = γ k (1 + 1 -β 1 -γ ) ∑ k j =0 ( β γ ) j v τ max , with β as defined in Eq. (2) . Finally, we define E β k = (1 -β ) ∑ k j =1 β k -j /epsilon1 j . With these notations: 0 ≤ q τ ∗ -q τ π k +1 ≤ ∑ k j =1 γ k -j ∣ ∣ ∣ A 2 k : j E β j ∣ ∣ ∣ + g 2 ( k ) 1 .

2 We illustrate the bounds in a simple tabular setting with access to a generative model. Considering random MDPs (called Garnets), at each iteration of DA-VI we sample a single transition for each state-action couple and apply the resulting sampled Bellman operator. The error /epsilon1 k is the difference between the sampled and the exact operators. The sequence of these estimation errors is thus a martingale difference w.r.t. its natural filtration [6] (one can think about bounded, centered and roughly independent errors). More details about this practical setting are provided in Appx. D.

Again, to ease the discussion, we express an /lscript ∞ -bound as a direct corollary of Thm. 2:

<!-- formula-not-decoded -->

There is a square dependency to the horizon, as for AVI. We discuss the other terms.

/negationslash

Rate of convergence. It is given by the function g 2 , defined in Thm. 2. If β = γ , we have g 2 ( k ) = 2( k +1) γ k v τ max . If β = γ , we have g 2 ( k ) = (1 + 1 -β 1 -γ ) β k +1 -γ k +1 β -γ . In all cases, g 2 ( k ) = o ( 1 k ), so it is asymptotically faster than in Thm. 1, but the larger the β , the slower the initial convergence. This is illustrated in Fig. 1, middle (notice that it's a logarithmic plot, except for the upper part of the y -axis).

Error rate. As with AVI, the error term is a discounted sum of the norms of errors. However, contrary to AVI, each error term is not an iteration error, but a moving average of past iteration errors, E β k = βE β k -1 +(1 -β ) /epsilon1 k . In the ideal case where the sequence of these errors is a martingale difference with respect to the natural filtration, this term no longer 1 E

/epsilon1 j E k bounded by 1 -β &lt; 1, that tends toward 0 for β close to 1. Therefore, we advocate that DA-VI 1 ( λ , τ ) allows for a better control of the error term than AVI (retrieved for β = 0). Notice that if asymptotically this error term predominates, the non-asymptotic behavior is also driven by the convergence rate g 2 , which will be faster for β closer to 0. Therefore, there is a trade-off, illustrated in Fig. 1, right (for the same simple experiment 2 ). Higher values of β lead to better asymptotic performance, but at the cost of slower initial convergence rate.

vanishes, contrary to k k . However, it can reduce the variance. For simplicity, assume that the 's are i.i.d. of variance 1. In this case, it is easy to see that the variance of β is

Interplay between the KL and the entropy terms. The l.h.s. of the bound of Thm. 2 solely depends on the entropy scale τ , while the r.h.s. solely depends on the term β = λ λ + τ . DA-VI( λ , τ ) approximates the optimal policy of the regularized MDP, while we are usually interested in the solution of the original one. We have that ‖ q ∗ -q π τ ∗ ‖ ∞ ≤ τ ln |A| 1 -γ [20], this bias can be controlled by setting an (arbitrarily) small τ . This does not affect the r.h.s. of the bound, as long as the scale of the KL term follows (such that λ λ + τ remains fixed to the chosen value). So, Thm. 2 suggests to set τ to a very small value and to choose λ such that we have a given value of β . However, adding an entropy term has been proven efficient empirically, be it with arguments of exploration and robustness [22] or regarding the optimization landscape [3]. Our analysis does not cover this aspect. Indeed, it applies to λ = β = 0 (that is, solely entropy regularization), giving the propagation of errors of SAC, as a special case of [20, Thm. 3]. In this case, we retrieve the bound of AVI ( E 0 j = /epsilon1 j , g 2 ( k ) ∝ γ k ), up to the bounded quantity. Thus, it does not show an advantage of using solely an entropy regularization, but it shows the advantage for considering an additional KL regularization, if the entropy is of interest for other reasons.

We end this discussion with some related works. The bound of Thm. 2 is similar to the one of CVI, despite a quite different proof technique. Notably, both involve a moving average of the errors. This is not surprising, CVI being a reparameterization of DA-VI. The core difference is that by bounding the distance to the regularized optimal q -function (instead of the unregularized one), we indeed show to what the algorithm converges without error. Shani et al. [40] study a variation of TRPO, for which they show a convergence rate of O ( 1 √ k ), improved to O ( 1 k ) when an additional entropy regularizer is considered. This is to be compared to the convergence rate of our variation of TRPO, O ( 1 k ) = o ( 1 √ k ) (Thm. 1) improved to g 2 ( k ) = o ( 1 k ) with an additional entropy term (Thm. 2). Our rates are much better. However, this is only part of the story. We additionally show a compensation of errors in both cases, something not covered by their analysis. They also provide a sample complexity, but it is much worse than the one of SQL, that we would improve (thanks to the improved horizon term). Therefore, our results are stronger and more complete.

Limitations of our analysis. Our analysis provides strong theoretical arguments in favor of considering KL regularization in RL. Yet, it has also some limitations. First, it does not

provide arguments for using only entropy regularization, as already extensively discussed (even though it provides arguments for combining it with a KL regularization). Second, we study how the errors propagate over iterations, and show that KL allows for a compensation of these errors, but we say nothing about how to control these errors. This depends heavily on how the q -functions are approximated and on the data used to approximate them. We could easily adapt the analysis of Azar et al. [6] to provide sample complexity bounds for MD-VI in the case of a tabular representation and with access to a generative model, but providing a more general answer is difficult, and beyond the scope of this paper. Third, we assumed that the greedy step was performed exactly. This assumption would be reasonable with a linear parameterization and discrete actions, but not if the policy and the q -function are approximated with neural networks. In this case, the equivalence between MD-VI and DA-VI no longer holds, suggesting various ways of including the KL regularizer (explicitly, MD-VI, or implicitly, DA-VI). Therefore, we complement our thorough theoretical analysis with an extensive empirical study, to analyse the core effect of regularization in deep RL.

## 5 Empirical study

Before all, we would like to highlight that if regularization is a core component of successful deep RL algorithms (be it with entropy, KL, or both), it is never the sole component. For example, SAC uses a twin critic [18], TRPO uses a KL hard constraint rather than a KL penalty [39], or MPO uses retrace [29] for value function evaluation. All these further refinements play a role in the final performance. On the converse, our goal is to study the core effect of regularization, especially of KL regularization, in a deep RL context. To achieve this, we notice that DA-VI and MD-VI are extensions of AVI. One of the most prevalent VI-based deep RL algorithm being DQN [28], our approach is to start from a reasonably tuned version of it [15] and to provide the minimal modifications to obtain deep versions of MD-VI or DA-VI. Notably, we fixed the meta-parameters to the best values for DQN.

Practical algorithms. We describe briefly the variations we consider, a complementary high-level view is provided in Appx. E.1 and all practical details in Appx. E.2. We modify DQN by adding an actor. For the evaluation step , we keep the DQN loss, modified to account for regularization (that we'll call ' w/ ', and that simply consists in adding the regularization term to the target q -network). Given that many approaches ignore the regularization there, we'll also consider the DQN loss (denoted ' w/o ' before, not covered by our analysis). For the greedy step , MD-VI and DA-VI are no longer equivalent. For MD-VI, there are two ways of approximating the regularized policy. The first one, denoted ' MD direct ', consists in directly solving the optimization problem corresponding to the regularized greediness, the policy being a neural network. This is reminiscent of TRPO (with a penalty rather than a constraint). The second one, denoted ' MD indirect ', consists in computing the analytical solution to the greedy step ( π k +1 ∝ π β k exp( 1 λ βq k )) and to approximate it with a neural network. This is reminiscent of MPO. For DA-VI, we have to distinguish τ &gt; 0 from τ = 0. In the first case, the regularized greedy policy can be computed analytically from an h -network, that can be computed by fitting a moving average of the online q -network and of a target h -network. This is reminiscent of MoDQN. If τ = 0, DA-VI( λ ,0) is not practical in a deep learning setting, as it requires averaging over iterations. Updates of target networks are too fast to consider them as new iterations, and a moving average is more convenient. So, we only consider the limit case λ, τ → 0 with β = λ λ + τ kept constant. This is MoDQN with fixed β , and the evaluation step is necessarily unregularized ( λ = τ = 0). To sum up, we have six variations (three kinds of greediness, evaluation regularized or not), restricted to five variations for τ = 0.

Research questions. Before describing the empirical study, we state the research questions we would like to address. The first is to know if regularization, without further refinements, helps, compared to the baseline DQN. The second one is to know if adding regularization in the evaluation step, something required by our analysis, provides improved empirical results. The third one is to compare the different kinds of regularized greediness, which are no longer equivalent with approximation. The last one is to study the effect of entropy, not covered by our analysis, and its interplay with the KL term.

Je-02

Oe-01

Je-01

Je-01

w/o w/

0e-01

Je-01

0e-01

Se-01

w/o

9e-01

0e-01

Je-02

Je-01

0e-01

Je-01

Je-01

0e-01

5e-01

9e-011

MD direct

MD direct (T = 0)

MD direct

MD direct (T = 0)

MD indirect

MD indirect (т = 0)

MD indirect

MD indirect (T = 0)

DA

DA

Mo-DQN

MO-DQN

Figure 3: Asterix.

<!-- image -->

Environments. We consider two environments here (more are provided in Appx. E). The light Cartpole from Gym [14] allows for a large sweep over the parameters, and to average each result over 10 seeds. We also consider the Asterix Atari game [10], with sticky actions, to assess the effect of regularization on a large-scale problem. The sweep over parameters is smaller, and each result is averaged over 3 seeds.

Visualisation. For each environment, we present results as a table, the rows corresponding to the type of evaluation ( w/ or w/o ), the columns to the kind of greedy step. Each element of this table is a grid, varying β for the rows and τ for the columns. One element of this grid is the average undiscounted return per episode obtained during training, averaged over the number of seeds. On the bottom of this table, we show the limit cases with the same principle, varying with λ for MD-VI and with β for DA-VI (ony w/o , as explained before). The scale of colors is common to all these subplots, and the performance of DQN is indicated on this scale for comparison. Additional visualisations are provided in Appx. E.

Discussion. Results are provided in Fig. 2 and 3. First, we observe that regularization helps. Indeed, the results obtained by all these variations are better than the one of DQN, the baseline, for a large range of the parameters, sometime to a large extent. We also observe that, for a given value of τ , the results are usually better for medium to large values of

β (or λ ), suggesting that KL regularization is beneficial (even though too large KL regularization can be harmful in some case, for example for MD direct, τ = 0, on Asterix).

Then, we study the effect of regularizing the evaluation step, something suggested by our analysis. The effect of this can be observed by comparing the first row to the second row of each table. One can observe that the range of good parameters is larger in the first row (especially for large entropy), suggesting that regularizing the evaluation step helps . Yet, we can also observe that when τ = 0 (no entropy), there is much less difference between the two rows. This suggests that adding the entropy regularization to the evaluation step might be more helpful (but adding the KL term too is costless and never harmful).

Next, we study the effect of the type of greediness. MD-direct shows globally better results than MD-indirect, but MD-indirect provides the best result on both environments (by a small margin), despite being more sensitive to the parameters. DA is more sensitive to parameters than MD for Cartpole, but less for Asterix, its best results being comparable to those of MD. This let us think that the best choice of greediness is problem dependent , something that goes beyond our theoretical analysis.

Last, we discuss the effect of entropy. As already noticed, for a given level of entropy, medium to large values of the KL parameter improve performance, suggesting that entropy works better in conjunction with KL, something appearing in our bound. Now, observing the table corresponding to τ = 0 (no entropy), we observe that we can obtain comparable best performance with solely a KL regularization, especially for MD. This suggests that entropy is better with KL, and KL alone might be sufficient . We already explained that some beneficial aspects of entropy, like exploration or better optimization landscape, are not explained by our analysis. However, we hypothesize that KL might have similar benefits. For examples, entropy enforces stochastic policies, which helps for exploration. KL has the same effect (if the initial policy is uniform), but in an adaptive manner (exploration decreases with training time).

## 6 Conclusion

We provided an explanation of the effect of KL regularization in RL, through the implicit averaging of q -values. We provided a very strong performance bound for KL regularization, the very first RL bound showing both a linear dependency to the horizon and an averaging the estimation errors. We also analyzed the effect of KL regularization with an additional entropy term. The introduced abstract framework encompasses a number of existing approaches, but some assumptions we made do not hold when neural networks are used. Therefore, we complemented our thorough theoretical analysis with an extensive empirical study. It confirms that KL regularization is helpful, and that regularizing the evaluation step is never detrimental. It also suggests that KL regularization alone, without entropy, might be sufficient (and better than entropy alone).

The core issue of our analysis is that it relies heavily on the absence of errors in the greedy step, something we deemed impossible with neural networks. However, Vieillard et al. [42] proposed subsequently a reperameterization of our regularized approximate dynamic scheme. The resulting approach, called 'Munchausen Reinforcement Learning', is simple and general, and provides agents outperforming the state of the art. Crucially, thanks to this reparameterization, there's no error in their greedy step and our bounds apply readily. More details can be found in [42].

Broader impact. Our core contribution is theoretical. We unify a large body of the literature under KL-regularized reinforcement learning, and provide strong performance bounds, among them the first one ever to combine a linear dependency to the horizon and an averaging of the errors. We complement these results with an empirical study. It shows that the insights provided by the theory can still be used in a deep learning context, when some of the assumptions are not satisfied. As such, we think the broader impact of our contribution to be the same as the one of reinforcement learning.

Funding transparency statement. Nothing to disclose.

## References

- [1] Yasin Abbasi-Yadkori, Peter Bartlett, Kush Bhatia, Nevena Lazic, Csaba Szepesvári, and Gellért Weisz. Politex: Regret bounds for policy iteration using expert prediction. In International Conference on Machine Learning (ICML) , 2019.
- [2] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. In International Conference on Learning Representations (ICLR) , 2018.
- [3] Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning (ICML) , 2019.
- [4] TW Archibald, KIM McKinnon, and LC Thomas. On the generation of markov decision processes. Journal of the Operational Research Society , 46(3):354-361, 1995.
- [5] Kavosh Asadi and Michael L Littman. An alternative softmax operator for reinforcement learning. In International Conference on Machine Learning (ICML) , 2017.
- [6] Mohammad G Azar, Rémi Munos, Mohammad Ghavamzadeh, and Hilbert J Kappen. Speedy q-learning. In Advances in neural information processing systems (NeurIPS) , 2011.
- [7] Mohammad Gheshlaghi Azar, Vicenç Gómez, and Hilbert J Kappen. Dynamic policy programming. Journal of Machine Learning Research (JMLR) , 13(Nov):3207-3245, 2012.
- [8] J Andrew Bagnell, Sham M Kakade, Jeff G Schneider, and Andrew Y Ng. Policy search by dynamic programming. In Advances in neural information processing systems , pages 831-838, 2004.
- [9] Leemon C Baird III. Reinforcement Learning Through Gradient Descent . PhD thesis, US Air Force Academy, US, 1999.
- [10] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47:253-279, 2013.
- [11] Marc G Bellemare, Georg Ostrovski, Arthur Guez, Philip S Thomas, and Rémi Munos. Increasing the action gap: New operators for reinforcement learning. In AAAI Conference on Artificial Intelligence (AAAI) , 2016.
- [12] Stephen Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [13] Steven J Bradtke and Andrew G Barto. Linear least-squares algorithms for temporal difference learning. Machine learning , 22(1-3):33-57, 1996.
- [14] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540 , 2016.

- [15] Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G Bellemare. Dopamine: A research framework for deep reinforcement learning. arXiv preprint arXiv:1812.06110 , 2018.
- [16] Matthew Fellows, Anuj Mahajan, Tim GJ Rudner, and Shimon Whiteson. Virel: A variational inference framework for reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS) , pages 7122-7136, 2019.
- [17] Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. In Conference on Uncertainty in Artificial Intelligence (UAI) , 2016.
- [18] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning , pages 1587-1596, 2018.
- [19] Matthieu Geist, Bilal Piot, and Olivier Pietquin. Is the bellman residual a bad proxy? In Advances in Neural Information Processing Systems , pages 3205-3214, 2017.
- [20] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International Conference on Machine Learning (ICML) , 2019.
- [21] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning (ICML) , 2017.
- [22] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning (ICML) , 2018.
- [23] Jean-Baptiste Hiriart-Urruty and Claude Lemaréchal. Fundamentals of convex analysis . Springer Science &amp; Business Media, 2012.
- [24] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning (ICML) , 2002.
- [25] Tadashi Kozuno, Eiji Uchibe, and Kenji Doya. Theoretical analysis of efficiency and robustness of softmax and gap-increasing operators in reinforcement learning. In International Conference on Artificial Intelligence and Statistics (AISTATS) , 2019.
- [26] Sergey Levine. Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review. arXiv preprint arXiv:1805.00909 , 2018.
- [27] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.
- [28] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning (ICML) , 2016.
- [29] Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2016.
- [30] Julien Perolat, Bruno Scherrer, Bilal Piot, and Olivier Pietquin. Approximate dynamic programming for two-player zero-sum markov games. In International Conference on Machine Learning (ICML) , 2015.
- [31] Julien Pérolat, Bilal Piot, Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. Softened approximate policy iteration for markov games. In International Conference on Machine Learning (ICML) , 2016.

- [32] Bilal Piot, Matthieu Geist, and Olivier Pietquin. Difference of convex functions programming for reinforcement learning. In Advances in Neural Information Processing Systems , pages 2519-2527, 2014.
- [33] Martin L Puterman. Markov Decision Processes.: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, 2014.
- [34] Martin L Puterman and Moon Chirl Shin. Modified policy iteration algorithms for discounted markov decision problems. Management Science , 24(11):1127-1137, 1978.
- [35] Bruno Scherrer and Boris Lesner. On the use of non-stationary policies for stationary infinite-horizon markov decision processes. In Advances in Neural Information Processing Systems (NeurIPS) , 2012.
- [36] Bruno Scherrer, Mohammad Ghavamzadeh, Victor Gabillon, Boris Lesner, and Matthieu Geist. Approximate modified policy iteration and its application to the game of tetris. Journal of Machine Learning Research (JMLR) , 16:1629-1676, 2015.
- [37] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning (ICML) , 2015.
- [38] John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft q-learning. arXiv preprint arXiv:1704.06440 , 2017.
- [39] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [40] Lior Shani, Yonathan Efroni, and Shie Mannor. Adaptive trust region policy optimization: Global convergence and faster rates for regularized MDPs. In AAAI Conference on Artificial Intelligence , 2020.
- [41] Zhao Song, Ron Parr, and Lawrence Carin. Revisiting the softmax bellman operator: New benefits and new perspective. In International Conference on Machine Learning (ICML) , 2019.
- [42] Nino Vieillard, Olivier Pietquin, and Matthieu Geist. Munchausen Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2020.
- [43] Nino Vieillard, Bruno Scherrer, Olivier Pietquin, and Matthieu Geist. Momentum in reinforcement learning. In International Conference on Artificial Intelligence and Statistics (AISTATS) , 2020.

Content. These appendices complement the core paper with the following:

- Appx. A is a warm-up that states a few facts about the Legendre-Fenchel transform, useful all along the derivations.
- Appx. C provides the proofs of all stated theoretical results, as well as some necessary lemmata.
- Appx. B justifies the connections drawn in Sec. 3 between MD-MPI or DA-MPI and the literature.
- Appx. D provides details about the experiment used to illustrate the bounds in Sec. 4.
- Appx. E provides additional details regarding the practical algorithms and the experiments, as well as additional experiments and visualisations.

## A Convex Conjugacy for KL and Entropy Regularization

Let q ∈ R S×A and µ ∈ ∆ S A , and consider the general greedy step π ′ ∈ G λ,τ µ , the optimization being understood here state-wise.

The function λ KL( π || µ ) -τ H ( π ) being convex in π , this optimization problem is related to the Legendre-Fenchel transform ( e.g. , Hiriart-Urruty and Lemaréchal [23, Ch. E]), or convex conjugate (which is the maximum rather than the maximizer). First, we consider a simple case, λ = 0 and τ = 1. It is well known in this case that the maximum (the convex conjugate) is the log-sum-exp function and the maximizer (the gradient of the convex conjugate) is the softmax ( e.g. , Boyd and Vandenberghe [12, Ex. 3.25]):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with 1 ∈ R S×A the vector of which all components are equal to 1. We made use of the notations introduced in Sec. 2, and overload v ∈ R S to v ∈ R S×A as v ( s, a ) = v ( s ). To make things clear, it gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice also that a direct consequence of this is that

From this simple case, we can easily handle the general case. We have

From this, we can deduce directly that the maximum of (3) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and that the maximizer of (3) is

<!-- formula-not-decoded -->

Again, the relationship between the maximum and the maximizer gives

<!-- formula-not-decoded -->

## B Connections to existing algorithms

In this section, we justify the connections stated in Sec. 3 between the considered regularized ADP schemes and the literature.

## B.1 Connection of MD-MPI( λ , τ ) to other algorithms

Connection to SAC. We stated that SAC [22] is a variation of MD-MPI(0, τ ). SAC was introduced as PI scheme ( m = ∞ ), while it is practically implemented as VI scheme ( m = 1). We keep the VI viewpoint for this discussion. The MD-VI(0, τ ) scheme is given by

<!-- formula-not-decoded -->

The regularized Bellman operator can be rewritten as follows:

<!-- formula-not-decoded -->

This is exactly the Bellman operator considered in SAC. For the greedy step, we have directly from Eq. (5) that π k +1 ∝ exp q k τ . In SAC, continuous actions are considered, so the policy cannot be computed (due to the partition function). Therefore, it is approximated with a neural network by minimizing a reverse KL divergence (that allows getting rid of the partition function) between the neural policy and the target policy (the solution of the original greedy step):

<!-- formula-not-decoded -->

Connection to Soft Q-learning. We stated that Soft Q-learning [17, 21] is also a variation of MD-MPI(0, τ ). It is indeed a VI scheme, so a variation of MD-VI(0, τ ) depicted in Eq. (7). As a direct consequence of Eq. (6), π k +1 ∝ exp q k τ being the maximizer, we have

<!-- formula-not-decoded -->

This allows rewriting the evaluation step as follows:

<!-- formula-not-decoded -->

Eq. (8) is equivalent to Eq. (7), and it is the Bellman operator upon which Soft Q-learning is built (replacing the hard maximum by the log-sum-exp). Haarnoja et al. [21] additionally handle continuous actions, which requires some refinements.

Connection to Softmax DQN. We stated that Softmax DQN [41] is a variation of MD-MPI(0, τ ), but w/o (without regularization in the evaluation step). Therefore, it is scheme (7), but replacing T 0 ,τ π k +1 by T π k +1 :

<!-- formula-not-decoded -->

.

Given that π k +1 ∝ exp q k τ , this amounts to iterating the following so called softmax operator

<!-- formula-not-decoded -->

which is the core update rule of softmax DQN. Notice that this operator might not be a contraction (depending on the value of τ ), and that it can have multiple fixed points [5].

<!-- formula-not-decoded -->

Connection to the mellowmax policy. Asadi and Littman [5] introduced a so-called mellowmax policy as a convergent alternative to the softmax operator. This can be indeed seen as an alternative way of regularizing the evaluation step. We explain here why. To do so, we reframe the mellowmax idea with our notations. Asadi and Littman [5] introduced the mellowmax operator as

<!-- formula-not-decoded -->

One can easily see that it is indeed the convex conjugate of the KL with respect to the uniform policy (that behaves like the entropy). Indeed, from Eq. (4), we have directly that

<!-- formula-not-decoded -->

with π U the uniform policy. From Geist et al. [20], we know that the following equivalent schemes,

<!-- formula-not-decoded -->

are convergent (MDP regularized with τ KL( ·|| π U ), the equivalence being from Eq. (6)). This is not the viewpoint of Asadi and Littman [5]. They try to find a policy π ′ k +1 such that q k +1 = r + γP mm τ ( q k ) = r + γP 〈 π ′ k +1 , q k 〉 . To account for the possible existence of multiple policies, they look for the one with maximal entropy and solve (numerically) for

<!-- formula-not-decoded -->

Then, they apply q k +1 = r + γP 〈 π ′ k +1 , q k 〉 . If there is no error when computing π ′ k +1 , this is equivalent to adding the regularization to the evaluation step.

Connection to TRPO. We stated that TRPO [37] is a variation of MD-MPI( λ , 0), w/o . More precisely, it is a variation of MD-PI( λ , 0):

<!-- formula-not-decoded -->

In TRPO, the q -function is evaluated using Monte Carlo rollouts. The greedy policy is approximated with a neural network by directly solving the expected greedy step:

<!-- formula-not-decoded -->

TRPO is indeed a bit different, as it uses importance sampling to sample actions according to π k (which is especially useful for continuous actions, but does not change the objective function), it uses a constraint based on the KL rather than a regularization, and it considers the KL in the other direction:

<!-- formula-not-decoded -->

However, from an abstract viewpoint, TRPO is close to scheme (9).

Connection to MPO. We stated that MPO [2] is also a variation of MD-MPI( λ , 0), w/o :

<!-- formula-not-decoded -->

The evaluation step is done by combining a TD approach with eligibility traces (a geometric average of m -step returns), rather than using m -step returns (that amounts to using the T π m operator). For the greedy step, the analytic solution can be computed for any state-action couple, and generalized to the whole state-action space by minimizing a KL between this analytical solution and a neural network:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The greedy step of MPO is indeed a bit different, the algorithm being derived from an expectation-maximization principle based on a probabilistic inference view of RL. The term λ is not fixed but learnt by the minimization of a convex dual function (coming from viewing the KL term as a constraint rather than a regularization), and an additional KL penalty is added (not necessarily redundant with the initial one, as the KL there is in the other direction):

<!-- formula-not-decoded -->

However, from an abstract viewpoint, MPO is close to scheme (10).

Connection to DPP. We stated that DPP [7] is a variation of MD-MPI( λ , 0). More precisely, it is close to be a reparameterization of MD-VI( λ , 0), the difference being mainly the error term:

<!-- formula-not-decoded -->

To derive the DPP update rule from Eq. (11), we consider /epsilon1 k = 0. The greedy policy is, according to (5),

<!-- formula-not-decoded -->

Define v k +1 as (the second equality coming from Eq. (6))

<!-- formula-not-decoded -->

With this, we have

<!-- formula-not-decoded -->

Let us define ψ k +1 ∈ R S×A as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Injecting Eqs. (13) and (14) into (12), we get

<!-- formula-not-decoded -->

Thus, we have

This is how DPP is justified from a DP viewpoint [7, Appx. A]. It is a bit different from the DPP algorithm analyzed by Azar et al. [7], for which ln 〈 1 , ψ k λ 〉 is replaced by 〈 π k , ψ k 〉 (both terms being equal in the limit λ → 0), and that consider an estimation error /epsilon1 ′ k +1 :

<!-- formula-not-decoded -->

We advocate that the error /epsilon1 ′ k is usually harder to control than /epsilon1 k (or equivalently that q k is easier to estimate than ψ k ), because the function ψ ∗ (the optimal ψ -function for the MDP) is equal to -∞ for any suboptimal action [7, Cor. 4].

Connection to CVI. We stated that CVI is a reparametrization of MD-VI( λ , τ ), that we recall (without the error term, to do the reparameterization):

<!-- formula-not-decoded -->

.

We now show how to derive the CVI update rule from this. The regularized greedy policy is, thanks to Eq. (5), and writing β = λ λ + τ :

<!-- formula-not-decoded -->

Similarly to DPP, we can define v k +1 as (still using Eq. (6) for the second equality):

<!-- formula-not-decoded -->

With this, we have

<!-- formula-not-decoded -->

Let us define ψ k +1 ∈ R S×A as

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Injecting Eqs. (16) and (17) into (15), we get

<!-- formula-not-decoded -->

This is exactly the CVI update rule. Notice that setting β = 1, i.e., τ = 0 (no entropy term), we retrieve DPP (which was to be expected). As we obtain CVI, by considering λ + τ → 0 while keeping β = λ λ + τ constant, we retrieve advantage learning in the limit [9, 11], that DA-VI( λ , τ ) thus generalizes.

## B.2 Connection of DA-MPI( λ , τ ) to other algorithms

Connection to Politex. Politex [1] addresses the average reward criterion. It is a PI scheme, up to the fact that the policy, instead of being greedy according to the last q -function, is softmax according to the sum of all past q -function. In the discounted reward case considered here, this is exactly DA-PI( λ ,0), w/o (without regularization in the evaluation step):

<!-- formula-not-decoded -->

Indeed, by definition h k = 1 k +1 ∑ k j =0 q j and the greedy policy is

<!-- formula-not-decoded -->

This is exactly the Politex algorithm, but for the discounted reward case (that changes how the q -function is defined, and thus estimated).

Connection to MoVI. MoVI [43] is a VI scheme, up to the fact that the policy, instead of being greedy according to the last q -function, is greedy according to the average of past q -functions. It is indeed is a limiting case of DA-VI( λ , 0), w/o :

<!-- formula-not-decoded -->

.

It is well known that the limit of a softmax, when the temperatures goes to zero, is the greedy policy: G 0 , λ k +1 ( h k ) →G ( h k ) as λ → 0. So, DA-VI( λ → 0, 0), w/o , is the following scheme,

,

that is exactly MoVI. Notice that it is different from MD-VI( λ → 0, 0), w/o , which is AVI (see also Prop. 1).

<!-- formula-not-decoded -->

Connection to momentum DQN. Momentum DQN [43] was introduced as a practical heuristic to MoVI, changing the exact average by a moving average (more amenable to optimization with deep networks). We show below that it is indeed a limiting case of DA-VI( λ , τ ), w/o (without regularized greedy step), that is:

<!-- formula-not-decoded -->

.

Fix β ∈ (0 , 1), we can consider λ, τ → 0 with β = λ λ + τ kept constant. In this case, the regularized greedy operator tends to the usual greedy one: G 0 ,τ ( h k ) →G ( h k ) as τ → 0. In the limit, we obtain the following scheme,

<!-- formula-not-decoded -->

for a chosen β , which is exactly momentum DQN with fixed β .

Connection to Speedy Q-learning. We stated that Speedy Q-learning [6] is a limiting case of DA-VI( λ ,0), which we recall (without the error term here):

<!-- formula-not-decoded -->

.

As shown in Lemma 2 in Appx. C.2, we have

<!-- formula-not-decoded -->

With this, DA-VI 1 ( λ ,0) can be expressed solely in terms of h k and π k :

<!-- formula-not-decoded -->

As before, as λ → 0, the regularized greedy step tends to the greedy step, G 0 , λ k +1 ( h k ) →G ( h k ). Regarding the evaluation step, we can write, by definition of the regularized Bellman operator and using Eq. (6),

<!-- formula-not-decoded -->

It is a classical result that the convex conjugate of the entropy tends to the hard maximum as the associated temperature goes to zero. For any s ∈ S ,

<!-- formula-not-decoded -->

Writing T ∗ the Bellman optimality operator, defined as T ∗ q = max π T π q , we thus have

<!-- formula-not-decoded -->

Thus, writing the limit of scheme (18) as λ → 0, we obtain

<!-- formula-not-decoded -->

which is exactly the Speedy Q-learning update rule.

Connection to softened LSPI. [31] address the problem of learning a Nash equilibria in zero-sum Markov games. They show that state of the art algorithms can be derived by minimizing the norm of the (projected) Bellman residual using a Newton descent, and propose more stable algorithms by using instead a quasi-Newton descent. Single agent reinforcement learning is a special case of zero-sum Markov games, and in this case the algorithm they propose can be written as follows, in an abstract way 3 :

<!-- formula-not-decoded -->

Using the same arguments as for the connection to momentum DQN, this is a limit case of DA-PI( λ , τ ), w/o , as λ, τ → 0 with β = λ λ + τ kept constant. It is also closely related to Politex (the policy is greedy instead of being softmax, moving average of the q -values instead of an average).

## C Proofs of Theoretical Results

In this section, we prove the results stated in the paper.

## C.1 Proof of Proposition 1

Sketch of proof. As explained in the paper, the optimization problem π k +1 = G λ, 0 π k ( q k ) can be solved analytically, yielding π k +1 ∝ π k exp q k λ . By direct induction, π 0 being uniform, we have π k +1 ∝ π k exp q k λ ∝ · · · ∝ exp 1 λ ∑ k j =0 q j . Thus, the policy is indeed softmax according to the sum of q -values. Defining h k as the average of past q -values basically provides the stated DA-VI( λ ,0). The case with an additionnal entropy term is a bit more involved, but the principle is the same.

3 Specialized to single agent RL, their algorithm adopts a linear parameterization of the q -function and estimate q π k +1 either with LSTD [13] or by minimizing the norm of the Bellman residual.

Proof. We start by proving the equivalence for the case τ = 0. Recall that we assumed, with little loss of generality, that π 0 is the uniform policy. We recall MD-MPI( λ ,0):

<!-- formula-not-decoded -->

Let us define h 0 = q 0 and h k for k ≥ 1 as the average of past q -functions.

<!-- formula-not-decoded -->

As a direct consequence of Eq. (5), we have that π k +1 ∝ π k exp q k λ . By direct induction,

<!-- formula-not-decoded -->

Still thanks to Eq. (5), this means that π k +1 satisfies

<!-- formula-not-decoded -->

This shows that Eq. (19) is equivalent to

<!-- formula-not-decoded -->

which is DA-MPI( λ ,0), and this shows the first part of the result. In the limit λ → 0, the regularized greediness becomes the usual greediness (hard maximum over q -values) and the (regularized) evaluation operator becomes the standard one. However, notice that schemes are not equivalent in the limit: scheme (19) tends to classic VI, while scheme (20) tends to Speedy Q-learning [6] (see the justification of the connection to SQL in Appx. B.2).

Next, we prove the equivalence for the case τ &gt; 0. We recall MD-MPI( λ , τ ):

<!-- formula-not-decoded -->

Thanks to Eq. (5), we have that π k +1 ∝ exp q k + λ ln π k λ + τ . We define β = λ λ + τ (and thus 1 -β = τ λ + τ and β λ = 1 λ + τ ). By induction, we have (writing 'cst' any function depending solely on states, not necessarily the same for different lines):

<!-- formula-not-decoded -->

We now define h k as the moving average of past q -values, with h -1 = 0:

<!-- formula-not-decoded -->

Noticing also that β λ (1 -β ) = 1 τ , this shows that

<!-- formula-not-decoded -->

As before, this means that π k +1 is the solution of an entropy regularized greedy step with respect to h k :

This means that Eq. (21) is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is the DA-MPI( λ , τ ) scheme. This concludes the proof.

## C.2 Proof of Theorem 1

Here, we provide the bound for DA-VI( λ ,0), which we recall:

<!-- formula-not-decoded -->

Sketch of proof. The quantity of interest is q ∗ -q π k +1 , it can be decomposed as q ∗ -q π k +1 = q ∗ -h k + h k -q π k +1 . Lemma 1 allows expressing the quantity of interest essentially as a function of the Bellman residual T π k +1 h k -h k . Controlling this residual is the key to state our bound. To achieve this, we first derive Lemma 2 that expresses the evaluation step (the update of the q -function) as a difference of Bellman operators applied to successive h-functions (the averages of q -values). Thanks to this, we're able to derive a Bellman-like recursion for h k in Lemma 3, using notably Lemma 2 and a telescoping argument. The rest of the proof consists in exploiting this Bellman-like recursion to control the residual and eventually boud the quantity of interest.

Proof. We start by stating a useful lemma.

Lemma 1. For any q ∈ R S×A and π ∈ ∆ S A , we have

<!-- formula-not-decoded -->

Proof. This result is classic, and appears many times in the literature ( e.g. , Kakade and Langford [24]). We provide a one line proof for completeness, relying on basic properties of the Bellman operator:

<!-- formula-not-decoded -->

The aim is to bound the quantity q ∗ -q π k +1 , the difference between the optimal value function and the value function computed by DA-VI( λ ,0). Thanks to Lemma 1, we can decompose this term as

<!-- formula-not-decoded -->

Notice that q ∗ = q π ∗ for any optimal policy π ∗ . There exists an optimal deterministic policy [33], so we will consider a deterministic π ∗ . As for any deterministic policy, H ( π ∗ ) = 0. Using the definition of π k +1 , we have

<!-- formula-not-decoded -->

Injecting this into Eq. (24), we obtain, using the fact that for any π the matrix ( I -γP π ) -1 = ∑ t ≥ 0 γ t P t π is positive,

<!-- formula-not-decoded -->

So, what we have to do is to control the residual T 0 , λ k +1 π k +1 h k -h k .

To do so, the following lemma will be useful.

Lemma 2. For any k ≥ 1 , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For k = 0 , we have

Proof. To prove this result, we will start by working on the optimization problem related to the regularized greedy step G λ, 0 π k q k :

<!-- formula-not-decoded -->

For DA-VI 1 ( λ ,0), π k +1 ∈ G 0 , λ k +1 ( h k ) (see Eq. (23)), so according to Eq. (5), π k +1 ∝ exp ( k +1) h k λ . Therefore, we have, using also the definition of h k

Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The maximizer is π k +1 , obviously. It is also the maximizer of 〈 π, ( k + 1) h k 〉 -λ 〈 π, ln π 〉 (the third term not depending on π ), and the associated maximum is, according to Eq. (4), λ ln 〈 1 , exp ( k +1) h k λ 〉 . This gives

Still from Eq. (4), we know that λ k +1 ln 〈 1 , exp ( k +1) h k λ 〉 is the maximum of 〈 π, h k 〉 + λ k +1 H ( π ), the associated maximizer being again π k +1 , so using Eq. (6), we can conclude that

<!-- formula-not-decoded -->

Noticing that r = ( k +1) r -kr , we have the first part of the result:

<!-- formula-not-decoded -->

This only holds for k ≥ 1. For k = 0, using the fact that h 0 = q 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used in the last line the fact that, π 0 being uniform,

This concludes the proof.

Using this lemma, we can provide a Bellman-like induction on h k .

Lemma 3. Define E k = -∑ k j =1 /epsilon1 j . For any k ≥ 1 , we have that

<!-- formula-not-decoded -->

Proof. Using the definition of h k , Lemma 2, the fact that q k +1 = T λ, 0 π k +1 | π k q k + /epsilon1 k +1 , and the definition E k = -∑ k j =1 /epsilon1 j , we have

<!-- formula-not-decoded -->

We now have the tools to work on the residual of interest. Starting from Lemma 3, and using the fact that ( k +2) h k +1 = ( k +1) h k + q k +1 ,

<!-- formula-not-decoded -->

Injecting this last result into decomposition (25), we get

<!-- formula-not-decoded -->

where we used for the last inequality the fact that -( I -γP π k +1 ) -1 P H ( π 0 ) ≤ 0. Next, using the fact that q ∗ -q π k +1 ≥ 0 and rearranging terms, we have

<!-- formula-not-decoded -->

We assumed that ‖ q k +1 ‖ ∞ ≤ v max ≤ v λ max (see also Rk. 1). When introducing the algorithm, we assumed that ‖ q 0 ‖ ∞ ≤ v max . Therefore, ‖ q 0 -γλP H ( π 0 ) ‖ ∞ ≤ v λ max . Writing 1 the vector whose components are all 1, we get | q k +1 -q 0 + γλP H ( π 0 ) | ≤ 2 v λ max 1 . Notice that for any policy π , we have that P π 1 = 1 . Therefore, we have

<!-- formula-not-decoded -->

With the same arguments, we have that

<!-- formula-not-decoded -->

We finally have

<!-- formula-not-decoded -->

which is the stated result.

## C.3 About Remark 1

We stated in Rk. 1, in the context of DA-VI( λ ,0), that the assumption ‖ q k ‖ ∞ ≤ v max is not strong with approximation, as this just requires clipping the q -values. Indeed, without approximation, it's not even necessary to clip the q -values.

No approximation. We will proceed by induction. Assume that ‖ q k ‖ ∞ ≤ v max . We assumed generally that ‖ q 0 ‖ ∞ ≤ v max . Without error, the considered scheme is

<!-- formula-not-decoded -->

As π k +1 = G λ, 0 ( q k ), we have that

<!-- formula-not-decoded -->

The inequality making use of the induction argument. On the other hand, making use of the positiveness of the KL divergence, we have that

<!-- formula-not-decoded -->

where again the inequality comes from the induction argument. This allows concluding, ‖ q k +1 ‖ ∞ ≤ v max .

With approximation. Knowing a bound of the q -values without approximation, we can clip q k such that it satisfies the bound, the effect of the clipping being part of the error. For example, assume that the evaluation step is approximated with a least-squares problems, a parameterized q -function, the target being a sampling of T λ, 0 π k +1 | π k q k , q k being the previous approximation (for example the target network). We can clip the result of the least-squares in [ -v max , + v max ] and call the resulting function q k +1 . The resulting error is defined as /epsilon1 k +1 = q k +1 -T λ, 0 π k +1 | π k q k .

## C.4 Proof of Theorem 2

In this section, we provide a bound for DA-VI( λ , τ ). First, we recall the scheme:

<!-- formula-not-decoded -->

.

We recall that due to the entropy term, this scheme cannot converge to the unregularized optimal q ∗ function. Yet, without errors and with λ = 0, it would converge to the solution of the MDP regularized by the scaled entropy [20] (optimizing for the reward augmented by the scaled entropy). Our bound will show that adding a KL penalty does not change this. We recall the notations introduced in the main paper. We already have defined the operator T 0 ,τ π . It has a unique fixed point, which we write q τ π . The unique optimal q -function is q τ ∗ = max π q τ π . We write π τ ∗ = G 0 ,τ ( q τ ∗ ) the associated unique optimal policy, and q τ π τ ∗ = q τ ∗ .

Sketch of proof. The proof is similar to the one of Thm. 1, albeit a bit more technical. Thanks to Lemma 4 (that generalizes Lemma 1), we decompose the quantity of interest q τ ∗ -q τ π k +1 as a function of q τ ∗ -h k and of T 0 ,τ π k +1 h k -h k , to be respectively upper-bounded and lower-bounded. To achieve this, we first derive Lemma 5 that expresses the evaluation step as a difference of Bellman operators applied to successive h-functions (similarly to Lemma 2). Thanks to this, we're able to derive a Bellman-like recursion for h k in Lemma 6, using notably Lemma 5 and a telescoping argument (similarly to Lemma 3). The end of the proof is then close to the classic propagation of errors of AVI, involving moving averages of the errors instead of the errors, as well as some additional terms.

Proof. The following lemma, generalizing Lemma 1 to the regularized Bellman operator, will be useful:

Lemma 4. Let τ ≥ 0 . For any q ∈ R S×A and π ∈ ∆ S A , we have

<!-- formula-not-decoded -->

Proof. The proof is the same as the one of Lemma 1, relying on the fact that the regularized Bellman operator has the same properies as the Bellman operator [20]:

<!-- formula-not-decoded -->

We will bound the quantity q τ ∗ -q τ π k +1 , using the following decomposition, based on Lemma 4:

<!-- formula-not-decoded -->

To do so, we will upper-bound q τ ∗ -h k and lower-bound T 0 ,τ π k +1 h k -h k (we recall that the matrix ( I -γP π k +1 ) -1 is non-negative). This requires a Bellman-like induction on h k . For this, the following intermediate lemma, similar to Lemma 2, will be useful.

Lemma 5. For any k ≥ 0 , we have that

<!-- formula-not-decoded -->

Proof. We have that, for any π ,

<!-- formula-not-decoded -->

As π k +1 ∝ exp h k τ , using also the fact that β = λ λ + τ and 1 -β = τ λ + τ , as well as the definition of h k (22), we have

<!-- formula-not-decoded -->

Hence, injecting this in the previous result, we get

<!-- formula-not-decoded -->

Now, as π k +1 ∝ exp h k τ , we have that 〈 π k +1 , h k 〉 + τ H ( π k +1 ) = τ ln 〈 1 , exp h k τ 〉 (again from Eq. (6)), therefore

<!-- formula-not-decoded -->

The result follows by the definition of T λ,τ π k +1 | π k q k = r + γP ( 〈 π k +1 , q k 〉 -λ KL( π k +1 || π k ) + τ H ( π k +1 )), and noticing that r = 1 1 -β ( r -βr ).

This result allows to build the lemma stating a Bellman-like induction for h k .

Lemma 6. Define E β k +1 = -(1 -β ) ∑ k +1 j =1 β k +1 -j /epsilon1 j = βE β k +(1 -β ) /epsilon1 k +1 (with E β 0 = 0 ). For any k ≥ 0 , we have that

<!-- formula-not-decoded -->

Proof. Using the definition of h k , Eq. (22), the relationship between q k +1 and q k , and Lemma 5, we have

<!-- formula-not-decoded -->

Let define E β k +1 as

We also have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice also that h 0 = (1 -β ) q 0 . Putting all these parts together, we obtain

<!-- formula-not-decoded -->

which is the stated result.

Thanks to this result, we can now bound the terms of interest.

Upper-bounding q τ ∗ -h k . Write e k = E β k + β k ( T 0 ,τ π 0 h -1 -h 0 ), we have from Lemma 6 that h k +1 = T 0 ,τ π k +1 h k -e k +1 . Then, we have :

<!-- formula-not-decoded -->

By direct induction, we obtain

<!-- formula-not-decoded -->

This is the desired upper-bound.

Lower-bounding T 0 ,τ π k +1 h k -h k . Using the same notation e k , we have

<!-- formula-not-decoded -->

We define P k : j = P π k P π k -1 . . . P π j +1 P π j for j ≤ k , with the convention P k : k +1 = I . By direct induction, the preceding inequality gives

<!-- formula-not-decoded -->

Putting things together. Plugging Eqs. (27) and (28) into Eq. (26), we obtain

<!-- formula-not-decoded -->

Using the fact that q τ ∗ -q τ π k +1 ≥ 0, rearranging terms, we have

<!-- formula-not-decoded -->

The first term is related to the error, the others to the initialisation. We'll work on each of these other terms.

Recall that we assumed that ‖ q 0 ‖ ∞ ≤ v max = r max 1 -γ . Therefore, ‖ q 0 ‖ ∞ ≤ v τ max = r max + τ ln |A| 1 -γ . As h 0 = (1 -β ) q 0 , we have ‖ h 0 ‖ ∞ ≤ (1 -β ) v τ max . From obvious properties of regularized MDPs [20], we have ‖ q τ ∗ ‖ ∞ ≤ v τ max . Therefore, writing 1 ∈ R S×A the vector with all components equal to 1, we have | q τ ∗ -h 0 | ≤ (2 -β ) v τ max 1 . Notice that for any policy π , we have P π 1 = 1 , thus

We also have that ‖ T 0 ,τ π 1 h 0 ‖ ∞ ≤ r max + τ ln |A| + γ (1 -β ) v τ max = (1 -γβ ) v τ max , so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition h -1 = 0, so we have ‖ T 0 ,τ π 0 h -1 ‖ ∞ = ‖ r + γPτ H ( π 0 ) ‖ ∞ ≤ r max + τ ln |A| = (1 -γ ) v τ max , so ‖ T 0 ,τ π 0 h -1 -h 0 ‖ ∞ ≤ (2 -γ -β ) v τ max . Therefore, we have the following bound:

<!-- formula-not-decoded -->

Similarly, for the last term we have

<!-- formula-not-decoded -->

Summing these four upper bounds, we obtain

<!-- formula-not-decoded -->

Plugging this result into Eq. (29), we obtain the stated result:

<!-- formula-not-decoded -->

## D Empirical illustration of the bounds

We have illustrated the bounds of Sec. 2 (Fig. 1) in a simple tabular setting with acces to a generative model. We provide more details about this setting here.

We consider MDPs with small state and action spaces, such that a tabular representation of the q -function is possible. We also assume to have access to a generative model, allowing us to sample a transition for any state-action couple. We then consider sampled MD-VI( λ , τ ), depicted in Alg. 1. At each iteration of MD-VI, we sample a single transition for each state-action couple and apply the resulting sampled Bellman operator. The error /epsilon1 k is the difference between the sampled and the exact operators. The sequence of these estimation errors is thus a martingale difference w.r.t. its natural filtration [6] (one can think about bounded, centered and roughly i.i.d. errors).

We run this algorithm on randomized MDPs called Garnets. A Garnet [4] is an abstract MDP, built from three parameters ( N S , N A , N B ), with N S and N A respectively the number of states and actions, and N B the branching factor. The principle is to directly build the transition kernel P that represents the MDP. For each ( s, a ) ∈ S ×A , N B states ( s 1 , . . . s N B ) are drawn uniformly from S without replacement. Then, N B -1 numbers are drawn uniformly in (0 , 1) and sorted as ( p 0 = 0 , p 1 , . . . p N B -1 , p N B = 1). The transition kernel is then defined as P ( s k | s, a ) = p k -p k -1 for each 1 ≤ k ≤ N B . The reward function is drawn uniformly in (0 , 1) for 10% of the states, these states being drawn uniformly without replacement.

For the experiments shown in Fig. 1, we set N S = 30, N A = 4, N B = 4 and γ = 0 . 9. We generate 100 Garnets and run MD-VI once for each of these Garnets, for K = 800 iterations. The results in Fig. 1 shows the normalized average performance, ‖ q τ ∗ -q τ π k ‖ 1 ‖ q τ ∗ ‖ 1 . For sampled DA-VI( λ , 0), we show the behavior for various values of λ . For DA-VI( λ , τ ), we fix τ to a small value ( τ = 10 -3 ) and show the behavior for various values of β = λ λ + τ . Notice that considering a large value of τ would not be interesting. In this case, the regularized optimal policy would be close to be uniform, so close to the initial policy.

## E Algorithms and experimental details

This appendix provides additional details about the algorithms and the experiments:

- Appx. E.1 provides a complementary high level view of algorithms sketched in Sec. 5.
- Appx. E.2 provides implementation details of these algorithms, including a pseudocode.

<!-- formula-not-decoded -->

- Appx. E.3 provides all hyperparameters used in our experiments.
- Appx. E.4 provides additionnal experiments (one additional gym environment, Lunar Lander, and two additional Atari games, Breakout and Seaquest), as well as additional visualisations (including all training curves on Atari games).

## E.1 High level view of practical algorithms

DA-VI and MD-VI are extensions of VI. One of the most prevalent VI-based deep RL algorithm is probably DQN [27]. Thus, our approach consists in modifying the DQN algorithm to study regularization. To complement the sketch of Sec. 5, We present the different variations we consider with a high level viewpoint here, all practical details being just after.

DQN maintains a replay buffer and a target network q k , and computes q k +1 by minimizing the loss (recall that ' w/o ' stands for 'without regularization'):

<!-- formula-not-decoded -->

with q a neural network, π k +1 ∈ G ( q k ) the greedy policy computed analytically from q k , [ ˆ T π k +1 q k ]( s, a ) = r ( s, a ) + γ 〈 π k +1 , q k 〉 ( s ′ ) the sampled Bellman operator (with s ′ ∼ P ( ·| s, a )), and where the empirical expectation E s,a is according to the transitions in the buffer. DQN is an optimistic AVI scheme, in the sense that only a few steps of stochastic gradient descent are performed before updating the target network. We modify DQN by adding a policy network and possibly modifying the evaluation step. For the moment, we consider τ &gt; 0.

Greedy step. As explained before, when the greedy step is approximated, MD-VI and DA-VI are no longer equivalent. We start with MD-VI. A natural way to learn the policy network is to optimize directly for the greedy step. Let π k be the target policy network and q k the target q -network, it corresponds to ('dir' stands for direct):

<!-- formula-not-decoded -->

Maximizing this loss over networks gives π k +1 . This is reminiscent of TRPO (see Appx. B.1).

One can also compute analytically the policy π k +1 (see Appx. A), but it would require remembering all past networks. Thus, another solution is to approximate this analytical solution by a neural network ('ind' stands for indirect):

<!-- formula-not-decoded -->

Minimizing this loss over networks gives π k +1 . This is reminiscent of MPO (see Appx. B.1), up to the fact that we consider the KL in the reverse order. Indeed, MPO (or SAC) would optimise for ˆ E s [KL( π || π ∗ k +1 )( s )]. The motivation to do so is to get ride of the partition function. Yet, this is equivalent to what we call the 'direct' approach, writing Z k ∈ R S the partition function:

<!-- formula-not-decoded -->

So, up to the scaling β λ = 1 λ + τ and to the term ln Z k , which is a constant regarding the optimized policy π and can thus safely be ignored, we obtain the loss of Eq. (31).

When considering DA-VI, the policy can be computed analytically, π k +1 = G 0 ,τ ( h k ), but h k has to be approximated (and can be seen as the logits of the policy). With h k -1 and q k the target networks:

<!-- formula-not-decoded -->

Minimizing this loss over networks h gives h k . This is reminiscent of momentum-DQN (see Appx. B.2).

Evaluation step. Given one of the three ways of doing the greedy step, one can choose between regularizing the evaluation step ( w/ , as suggested by the theory) or not ( w/o , as often done empirically). This second case is already depicted in Eq. (30) (changing the considered policy) and the first case is given by

<!-- formula-not-decoded -->

So combining one of the two evaluation steps ( w/ or w/o ) with one of the three greedy steps (MD-dir, MD-ind or DA), we get six variations. We discuss also the limit case without entropy.

When τ = 0 . For MD-VI, one can set τ = 0. However, recall that for DA-VI, the resulting algorithm is different. DA-VI( λ , 0) is not practical in a deep learning setting, as it requires averaging over iterations. Indeed, updates of target networks are too fast to consider them as new iterations, and a moving average is more convenient. Vieillard et al. [43] used a decay on β to mimic this behavior, but this is a heuristic that needs to be tuned. Therefore, for DA-VI we will only consider the limit case λ + τ → 0 with β = λ λ + τ kept constant (that is, momentum-DQN with fixed β ). In this case, type 1 and 2 are equivalent. We offer additional visualisations in Appx. E.4.

## E.2 More on practical algorithms

We now detail the losses presented in the previous section, giving equations that are closer to implementation, and providing a detailed pseudo-code in Algorithm 2. Firts, let us introduce some notations.The q -value is represented by a neural network Q θ of parameters θ , and the policy is represented by a network Π φ of parameters φ . During training, the algorithms interact with an environment, and collect transitions ( s, a, r, s ′ ) that are stored in a FIFO replay buffer B . The parameters of the networks are copied regularly into old versions of themselves, with target weights ¯ θ and ¯ φ . The weights θ are optimized during the evaluation step, and φ during the greedy step.

## E.2.1 Evaluation step

All the actor-critics we consider have the same update rule of their critic - the Q -network. We consider two regressions targets, corresponding to regularizing the evaluation step or not. If not regularized, we define a regression target as

<!-- formula-not-decoded -->

and if regularized,

<!-- formula-not-decoded -->

The weights θ are then updated by minimizing the following regression loss with a variant of SGD

<!-- formula-not-decoded -->

Note that if Π φ was greedy with respect to Q ¯ θ , using L w/o would reduce to Deep q -networks (DQN) [27].

## E.2.2 Greedy step

Let us re-write in detail the three equations from Section E.1 that define three ways of performing the greedy step.

MD-dir. The Direct MD update tackles directly the optimization problem derived from the greedy step. For convenience, we define a loss (the opposite of what we would like to maximize) that we minimize with SGD

<!-- formula-not-decoded -->

MD-ind. The indirect version is based on the analytical result of the optimization problem corresponding to the greedy step. We show in Appendix B.1 that , at iteration k of MDVI( λ, τ ), we have π k +1 = G λ,τ π k ( q k ) ∝ π β k exp q k τ + λ . Hence, we would need to fit a target that approximates this maximizer, by defining ˆ Π( a | s ) as

<!-- formula-not-decoded -->

However, the exponential term can cause numerical problems, so what we optimize during the evaluation step is actually the logarithm of the policy. To work around this, we define a network L φ that represents the log-probabilities of a policy, and we define a regression target

<!-- formula-not-decoded -->

and then we have ˆ Π( a | s ) = exp ( ˆ L ( s, a ) ) and Π φ ( a | s ) = exp ( L φ ( a | s )). We then define a loss on the parameters φ ,

<!-- formula-not-decoded -->

DA. The dual averaging version is inspired by the DA-VI formulation. Instead of representing directly the policy, we estimate a moving average of the q -values, and then compute its softmax. The moving average is estimated via a network H φ , which fits a regression target

<!-- formula-not-decoded -->

and the policy is defined as softmax over H φ ( s, · ),

<!-- formula-not-decoded -->

The weights φ are optimized by minimizng the loss

<!-- formula-not-decoded -->

## E.2.3 Pseudo code

We give a general pseudo-code of the deep RL algorithms we used in Alg. 2. Notice that for a policy π , we define the e -greedy policy with respect to π as the policy that takes a random action (uniformly on A ) with probability e , and follows π with probability 1 -e .

## Algorithm 2 (MD-dir | MD-ind | DA)

```
Require: L q ( θ ) and L π ( φ ), two losses, respectively for the evaluation and the greediness. The choice of these losses determines the algorithm, see Table 2. Require: K ∈ N ∗ the number of steps, C ∈ N ∗ the update period, F ∈ N ∗ the interaction period. set θ , φ at random set Q θ the q -value network, Π φ the policy network, as defined in Sec. E.2. set B = {} set Π φ,e k the policy e k -greedy w.r.t. Π φ ¯ θ = θ, ¯ φ = φ for 1 ≤ k ≤ K do Collect a transition t = ( s, a, r, s ′ ) from Π φ,e k B ← B ∪ { t } if k mod F == 0 then On a random batch of transitions B q,k ⊂ B , update θ with one step of SGD on L q On a random batch of transitions B h,k ⊂ B , update φ with one step of SGD on L π end if if k mod C == 0 then ¯ θ ← θ , ¯ φ ← φ end if end for output Π φ
```

Table 2: Resulting algorithms given the choice of losses in Algorithm 2

|                                  | L π             | L π              | L π             |
|----------------------------------|-----------------|------------------|-----------------|
| L q                              | L dir (Eq.(34)) | L ind (Eq. (33)) | L da (Eq. (35)) |
| L w/ (Eq. (32)) L w/o (Eq. (32)) | MD-dir w/       | MD-ind w/        | DA w/           |
|                                  | MD-dir w/o      | MD-ind w/o       | DA w/o          |

## E.3 Hyperparameters

We provide the hyperparameters used on the Atari environments in Table 3, and on the Gym environments in Table 4. We use the following notations to describe neural networks: FC n is a fully connected layer with n neurons; Conv d a,b c is a 2d convolutional layer with c filters of size a × b and a stride of d . All hyperparameters are the one found in the Dopamine code base. We only tuned the learning rate and the update period of DQN on Lunar Lander (not provided in Dopamine).

Table 3: Parameters used on Atari. Both the Q -network and policy-network have the same structure. n A is the number of actions available in a given game.

| Parameter                                                                                                                                                                                                   | Value                                                                                                                                                                                    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| K (number of steps) C (update period) F (interaction period) γ (discount) |B| (replay buffer size) | B π,k | and | B q,k | (batch size) e k (random actions rate) networks structure activations optimizers | 5 ∗ 10 7 8000 4 0.99 10 6 32 e 0 = 0 . 01, linear decay of period 2 . 5 · 10 5 steps Conv 4 8 , 8 32 - Conv 2 4 , 4 64 - Conv 1 3 , 3 64 - FC512 - FC n A Relu RMSprop ( lr = 0 . 00025) |

Table 4: Parameters used on CartPole and Lunar Lander . Both the Q -network and policynetwork have the same structure. We have n A = 2 on CartPole, and n A = 8 on Lunar Lander.

| Parameter                                                                                                                                                                                                   | Value                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| K (number of steps) C (update period) F (interaction period) γ (discount) |B| (replay buffer size) | B π,k | and | B q,k | (batch size) e k (random actions rate) networks structure activations optimizers | 5 ∗ 10 5 100 (Cartpole), 2500 (Lunar Lander) 4 0.99 5 ∗ 10 4 128 0.01 (constant with k ) FC512 - FC512 - FC n A Relu Adam ( lr = 0 . 001) |

## E.4 Additional results

Additional environment. In addition to the environments considered in Sec. 5, we provide three additional environments: Lunar Lander (from gym), Breakout and Seaquest (from Atari). The comments on these environments are similar to the discussion of Sec. 5

Full tables. We also provide the full results of the experiments (those from Section 5 and the new ones). The same plots are reported, expect that we add the exact value of each grid cell for completeness. Results for Carpole and Lunarlander are provided in Figs. 4 and 5, while results for the considered Atari games (Asterix, Breakout and Seaquest) are reported in Figs. 6, 7 and 8.

Training curves. We also report training curves on Atari. We report training curves of DA, MD-dir and MD-ind in Fig. 9 for Asterix, on Fig. 10 for Breakout, and on Fig. 11 for Seaquest. We report the training curves of the limit cases on these three games on Figs. 12, 13 and 14. In these figures, an iteration corresponds to 250000 training steps, and we report every iteration the undiscounted reward averaged over the last 100 episodes (the averaged score ). The training curves are averaged over 3 random seeds.

The training curves give more hindsights on the performance of the algorithms. Indeed, the metric we used in the tables (the averaged score over all iteration) is partly flawed, because it could give a high score to an algorithm with a performance drop at the end of training. For example, the MD-dir method on Atari seems to benefit from regularizing the evaluation step (as unregularized evaluation suffers from a performance drop), which is less visible from the score tables. In almost all the cases, we do not observe such behaviour, which validates the use of our metric.

Je-02

e-02

Je-01

e-01

e-01

Je-01

w/o w/

w/o w/

Je-01

e-01

se-01

e-01

e-01

e-01

Je-021

e-02

De-01|

e-01

Je-01

e-01

De-01

e-01

se-01

e-01

MD direct

MD direct (T = 0)

MD direct

MD direct (7 = 0)

291

446 446

281 280 282 288 317 358

278

344

24

359

297

-26

14

290

-51

292

-110 -126

402 406

34

-86

-54

-80 -50 -26 -37 -37

6

32

290

-27

271 283 291

-176

-140

351

399

18

-133

-53

-42

279

280

408

295 285 284 306 313

-272

84

294

296

377

288

25

388

407

420

-61

10

-89

-110

-26

-13

303

288

427

-26

429

352

55

-50

280

-77

324

-139

279

•71

287

286

91

289

-71

-114

288

309

-108

e-01

-24

289

-63

357

284

288

-72

293

290

-65

303

-67

368

-30

355

409

37

279

-35

-25

281

277

299

-26

325

412

34

17

405

34/

288

-66

288

-29

310

-23

379

385

439

38

13

429

447

18

261

•24

32

267

301

24

349

-40

351

18

387

34

429

33

444

33

205

24

26

213

229

263

44

270

43

305

32

27

-42

413

23

14

414

-45

421

-32

167

•33

167

-33

172

192

-38

-46

202

293

358

-39

-85

13

362

297

429 435

-38

-87

34

-33

282

360

-95

364

299

-101

284

375

-100 -133

141

-96

-96

141

-96

145

-99

157

-101

166

240

110

-121

-121

111

112

-122

-124

116

119

-127

198

-85

-114 -145

+00

127

le-01

190

457

45

33

192

11

207

37

462

226

26

-16

263

-31

332

-45

200

202

201

239

36

290

-13

330

-11

208

MD indirect

MD indirect

MD indirect (7 = 0)

MD indirect 7 = 0)

202 223

432 461 461

192 205 221 310 315 365

24

-87

-59

233

-37

30

373

21

309

398

242

400

298

29

27

33

4 -20 -72 -8

33 52 38

191

315

-85

-24

198

-27

229

199

-52

206

259

-36

308

347

-23

198

40

21

202

209

-100

310

-27

-52

334

361

-16

22

255

232

15

21

403

401

370

313

60

-13

264

-20

23

-28

23

243

211 249 294 344 359 412 463 468

10

290

265

40

28

322

417

-14

407

376

277

229

376

4

-32

251

-86

14

33

-71 -53

31 48

-29

339

-32

238

392

-35

-33

410

367

42

446

53|

460

24

201 209

•13

206

-40

235

345

-93

342

•64

427

39

5

36

218

-27

-17

372

396

31

451

51

54

419

37|

481

464

-26

218

431

37

41

439

461

46|

228

-92

225

•30

15

267

256

-88

21

350

335

12

392

59

451

49

44

44

382

25

439

25

424

15

429

-52

439

-34

228

-31

227

-36

239

320

-45

-52

361

-88

409

395

-37

-48

402

-108

401

173

-98

340

-108

338

-142

323

112

-96 -122

-96

179

111

-122

-97

188

261

-101

311

339

114

213 213

-25

-32

163

249

218

-101 -132

110

214

21

22

264

229

11

291

241

24

-10

276

-69

185

-206

410

-19

228

18

259

275

DA

DA

246

248

40

18

258

253

57

252

-17

265

•18

44

269

270

43

26

-112|

135

-122|

•64

280

-62

278

166

Mo-DQN

Mo-DON

287 280 | 215

47

-161

281

277

-28

303

276

386

29

-147

98

26

2 16

299 174

-67 -27

-114

-139 -133 -171 -136 -175 -188

314

323

411

434

-182

-201

-267

428

248

10

22

256

277

462

-214

228

38

30

223

17

243

463

-233

455

-150

406

172

56

47

175

191

458

-135

458

62

187

124

•23

126

-25

-28

152

384

52

-160 -177

147

-122

33

21

-165

-90

103

-92

102

-95

138

37

-145

86

-147

109

-123

-19

•121

Figure 4: Cartpole with complete values.

<!-- image -->

Figure 5: Lunar Lander with complete values.

<!-- image -->

80

De-01

De-01

De-01

De-01

wIo

De-01

De-01

WIo

De-01

De-01

De-01

De-01

De-01

De-01

MD direct

MD direct (7 = 0)

MD direct

MD direct (7 = 0)

4086

2276

3848

5052

695

128

28

96

107

44

135

4965

97

147

129

151

5129

163

6550

6484

5518

3714

142

126

114

52

135

6119

148

5939

134

137

158

5711

151

6145

4178

4115

131

123

6059

139

91

121

116

3995

7072

140

2466

26

128

3580

26

2273

1966

28

30

2080

2128

33

2262

74

2510

86

431

62

3151

112

66

2345

89

2553

3201

126

3681

4695

99

MD indirect

MD indirect

MD indirect (T = 0)

MD indirect (T = 0)

2637

2410

92

75

3440

7608

4934

128

122

2561

138

23

4665

2175

4574

83

84

3288

100

2538

77

130

2465

2789

106

4652

123

3344

109

2576

95

132

2918

102

2676

5480

119

103

2162

122

25

3420

116

2419

27

2517

77

3175

92

4336

26

1820

1980

24

2064

31

3760

4092

DA

DA

4560

5120

1581

1986

15

MO-DQN

MO-DQN

104

4080

6978

102

6659

4557

4973

5054

2319

1791

1981

26

1562

5580

146

6605

4523

5673

6186

5734

103

5086

131

6547

5494

6039

4763

116

2740

2159

Figure 6: Asterix with complete values.

<!-- image -->

Figure 7: Breakout with complete values.

<!-- image -->

De-01

De-01

w/o

De-011

De-01|

De-011

De-01

MD direct

MD direct (т = 0)

1669

272

3280

494

473

2495

389

1593

1686

2367

4440

883

2690

1284

1043

2317

1036

1315

2604

1935

1901

7965

1001

2221

2260

3348

847

1677

814

1468

1505

837

1021

413

3300

583

752

2241

979

2470

7627

MD indirect (т = 0)

MD indirect

756

960

1023

658

2243

4892

940

2301

3327

3310

11141

1738

917

5695

887

1249

2765

1905

1502

1115

1404

7058

1968

739

3733

1706

791

1179

1539

1531

2068

6257

3140

4221

1990

7417

7947

1565

4763

4281

DA

2972

1932

1405

1468

322

501

MO-DQN

9260

4708

1144

1388

2375

1813

6120

1269

2597

1870

1538

315

1025

1307

493

Figure 8: Seaquest with complete values.

<!-- image -->

3701

4426

Figure 9: All averaged training scores of MD-dir (top), MD-ind (middle) and DA (bottom), w/ and w/o , on Asterix, for several values of β and τ . Each plot corresponds to one value of β (in the titles). In each plot, a curve corresponds to a value of τ : 1 e -3 (orange), 3 e -3 (green), 1 e -02 (red), 3 e -2 (blue), 1 e -1 (brown). The blue dotted line is DQN.

<!-- image -->

Figure 10: All averaged training scores of MD-dir (top), MD-ind (middle) and DA (bottom), w/ and w/o , on Breakout, for several values of β and τ . Each plot corresponds to one value of β (in the titles). In each plot, a curve corresponds to a value of τ : 1 e -3 (orange), 3 e -3 (green), 1 e -02 (red), 3 e -2 (blue), 1 e -1 (brown). The blue dotted line is DQN.

<!-- image -->

Figure 11: All averaged training scores of MD-dir (top), MD-ind (middle) and DA (bottom), w/ and w/o , on Seaquest, for several values of β and τ . Each plot corresponds to one value of β (in the titles). In each plot, a curve corresponds to a value of τ : 1 e -3 (orange), 3 e -3 (green), 1 e -02 (red), 3 e -2 (blue), 1 e -1 (brown). The blue dotted line is DQN.

<!-- image -->

<!-- image -->

Figure 12: All averaged training scores of limit cases on Asterix, for several values of β and λ . In each plot, a curve corresponds to a value of λ for MD-ind and MD-dir, and to a value of β for Mo-DQN. The blue dotted line is DQN.

Figure 13: All averaged training scores of limit cases on Breakout, for several values of β and λ . In each plot, a curve corresponds to a value of λ for MD-ind and MD-dir, and to a value of β for Mo-DQN. The blue dotted line is DQN.

<!-- image -->

Figure 14: All averaged training scores of limit cases on Seaquest, for several values of β and λ . In each plot, a curve corresponds to a value of λ for MD-ind and MD-dir, and to a value of β for Mo-DQN. The blue dotted line is DQN.

<!-- image -->