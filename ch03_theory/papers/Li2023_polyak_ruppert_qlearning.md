## A Statistical Analysis of Polyak-Ruppert Averaged Q-learning

Xiang Li lx10077@pku.edu.cn Peking University

## Wenhao Yang

Jiadong Liang yangwenhaosms@pku.edu.cn Peking University

Zhihua Zhang zhzhang@math.pku.edu.cn

Peking University

## Abstract

We study Q-learning with Polyak-Ruppert averaging in a discounted Markov decision process in synchronous and tabular settings. Under a Lipschitz condition, we establish a functional central limit theorem for the averaged iteration ¯ Q T and show that its standardized partial-sum process converges weakly to a rescaled Brownian motion. The functional central limit theorem implies a fully online inference method for reinforcement learning. Furthermore, we show that ¯ Q T is the regular asymptotically linear (RAL) estimator for the optimal Q-value function Q ∗ that has the most efficient influence function. We present a nonasymptotic analysis for the glyph[lscript] ∞ error, E ‖ ¯ Q T -Q ∗ ‖ ∞ , showing that it matches the instance-dependent lower bound for polynomial step sizes. Similar results are provided for entropy-regularized Q-learning without the Lipschitz condition.

## 1 INTRODUCTION

Q-learning [Watkins, 1989], as a model-free approach seeking the optimal Q-function of a Markov decision process (MDP), is perhaps the most widely deployed algorithm in reinforcement learning (RL) [Sutton and Barto, 2018]. Unlike policy evaluation where the underlying structure is linear in nature and the goal is essentially to solve a linear system, Q-learning is nonlinear, nonsmooth and nonstationary. Theoretical analysis for Q-learning ranges from asymptotic convergence [Jaakkola et al., 1993, Tsitsiklis, 1994, Borkar

Proceedings of the 26 th International Conference on Artificial Intelligence and Statistics (AISTATS) 2023, Valencia, Spain. PMLR: Volume 206. Copyright 2023 by the author(s).

jdliang@pku.edu.cn Peking University

Michael I. Jordan jordan@cs.berkeley.edu UC Berkeley

and Meyn, 2000, Szepesvári et al., 1998] to nonasymptotic rates [Even-Dar et al., 2003, Beck and Srikant, 2012, Chen et al., 2020b, Li et al., 2021a, 2020b]. Variants of Qlearning [Lattimore and Hutter, 2014, Sidford et al., 2018a,b, Wainwright, 2019c] have been proposed that achieve the minimax lower bound of sample complexity established in [Azar et al., 2013].

On the other hand, Q-learning can be viewed through the lens of stochastic approximation (SA) [Konda and Tsitsiklis, 1999], a general iterative framework for solving root-finding problems [Robbins and Monro, 1951]. It is a particular instance of SA that targets the Bellman fixed-point equation, T Q ∗ = Q ∗ , where T is the population Bellman operator (see Eq. (5) for the definition).

The last-iterate behavior of Q-learning has been analyzed thoroughly within the nonlinear SA framework. In particular, on the asymptotic side, the ODE approach [Kushner and Yin, 2003, Abounadi et al., 2002, Borkar, 2009, Gadat et al., 2018, Borkar et al., 2021] establishes a functional central limit theorem (functional CLT), showing that the interpolated process that connects rescaled last iterates converges weakly to the solution of a specific SDE. From the nonasymptotic side, specific nonlinear SA convergence analyses have been tailored for Q-learning, capturing its nonasymptotic convergence rate [Chen et al., 2020b, 2021, Qu and Wierman, 2020].

An important gap in this literature is the behavior of Qlearning under averaging, specifically Polyak-Ruppert averaging [Polyak and Juditsky, 1992]. Polyak-Ruppert averaging provides a general tool for stabilizing and accelerating SA algorithms. It is known to accelerate policy evaluation [Mou et al., 2020a,b] and exhibits superior empirical performance in various RL problems [Lillicrap et al., 2016, Anschel et al., 2017]. However, a theoretical understanding of Q-learning with Polyak-Ruppert averaging is not yet available.

In this paper, we analyze averaged Q-learning in the setting of a discounted infinite-horizon MDP and in the synchronous setting where a generative model produces independent samples for all state-action pairs in every iteration [Kearns et al., 2002]. We provide both asymptotic and nonasymptotic analyses. On the asymptotic side, we establish an functional CLT for averaged Q-learning, showing that the partial-sum process, φ T ( r ) := 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] t =1 ( Q t -Q ∗ ) , converges weakly to a rescaled Brownian motion, namely Var 1 / 2 Q B D ( r ) , where r ∈ [0 , 1] is the fraction of data used, glyph[floorleft]·glyph[floorright] is the floor function, Var Q (see Eq. (10)) is the asymptotic variance, and B D ( · ) is a standard D -dimensional Brownian motion on [0 , 1] . Such a functional result for partial-sum processes has not been presented previously in the RL literature. This allows us to construct an asymptotically pivotal statistic using information from the whole function φ T ( · ) (see Proposition 3.1). This obviates the need to estimate the asymptotic variance in providing asymptotically valid confidence intervals for Q ∗ , which is required by [Chen et al., 2020a, Zhu et al., 2021, Hao et al., 2021, Shi et al., 2020, Khamaru et al., 2022]. It opens a door to online statistical inference for RL.

As a complementary result, we establish a semiparametric efficiency lower bound for any regular asymptotically linear (RAL) estimator (see Definition 4.2 for details) of the optimal Q-value function Q ∗ . Given the r -th fraction of data, we further show that φ T ( r ) is the most efficient RAL estimator with the smallest asymptotic variance, confirming its optimality in the asymptotic regime.

Onthe nonasymptotic side, we provide the first finite-sample error analysis of E ‖ ¯ Q T -Q ∗ ‖ ∞ in the glyph[lscript] ∞ -norm for both linearly rescaled and polynomial step sizes. The error is dominated by O ( √ ‖ diag(Var Q ) ‖ ∞ √ ln |S×A| T ) for polynomial step sizes given a sufficiently large T , which matches the instance-dependent lower bound established by [Khamaru et al., 2021b]. This, together with the worst-case bound ‖ diag(Var Q ) ‖ ∞ = O ((1 -γ ) -3 ) , implies that averaged Q-learning already achieves the optimal minimax sample complexity ˜ O ( |S×A| (1 -γ ) 3 ε 2 ) established by [Azar et al., 2013]. Those lower bounds have only been shown to hold for a complicated variance-reduced version of Q-learning in this setting [Wainwright, 2019c, Khamaru et al., 2021b].

From a technical perspective, we carefully decompose the partial sum process, φ T ( r ) , into several processes, each of which either has a nice structure (e.g., a sum of i.i.d. variables) or vanishes in the glyph[lscript] ∞ -norm with probability one. In this way, the nonasymptotic analysis reduces to careful examination of these diminishing rates. To underpin the functional CLT, we develop a new lemma that shows that a certain residual error converges to zero in probability (see Lemma D.1). Generalizing an existing result from Lee et al. [2021], Li et al. [2022], this technical lemma may be of independent interest. Finally, while both our asymptotic and nonasymptotic analyses rely on a Lipschitz condition, stated in Assumption 3.2, we find that averaged Q-learning regularized by entropy achieves a similar functional CLT and instance-dependent bound without the Lipschitz assumption.

Paper organization. The remainder of this paper is organized as follows. In Section 2, we introduce our notation and preliminaries on RL. We present the formal functional CLT in Section 3 and the semiparametric efficiency lower bound in Section 4. In Section 5, we show the nonasymptotic convergence bound and contrast it with previous work. We summarize our results and discuss future research directions in Section 7. We provide additional discussion of related work, and all proof details, in the appendix.

## 2 PRELIMINARIES

Discounted infinite-horizon MDPs. An infinite-horizon MDP is represented by a tuple M = ( S , A , γ, P, R, r ) . Here S is the state space, A is the action space, and γ ∈ (0 , 1) is the discount factor. For simplicity, we define D = |S ×A| = SA . We use P : S ×A → ∆( S ) to represent the probability transition kernel with P ( s ′ | s, a ) the probability of transiting to s ′ from a given state-action pair ( s, a ) ∈ S × A . Let R : S × A → [0 , ∞ ) stand for the random reward, i.e., R ( s, a ) is the immediate reward collected in state s ∈ S when action a ∈ A is taken. Unlike previous works [Wainwright, 2019b, Li et al., 2021a] which assume the immediate reward R is deterministic, we consider a general setting where R itself is a random function with r = E R the expected reward. A policy π maps each s ∈ S to a probability over A . In a γ -discounted MDP, a common objective is to maximize the expected long-term reward. For a given policy π : S → ∆( A ) , the expected long-term reward is measured by the Q-function Q π defined as follows

<!-- formula-not-decoded -->

and its companion value function is defined via V π ( s ) = ∑ a ∈A π ( a | s ) Q π ( s, a ) . Here E π ( · ) is taken with respect to the randomness of the trajectory of the MDP induced by the policy π . The optimal value function V ∗ and optimal Q-function Q ∗ are defined as V ∗ ( s ) = max π V π ( s ) and Q ∗ ( s, a ) = max π Q π ( s, a ) . For simplicity, we employ the vectors V π , V ∗ ∈ R S and Q π , Q ∗ , Q t , ¯ Q t ∈ R D to denote evaluations of the functions V π , V ∗ , Q π , Q ∗ , Q t , ¯ Q t .

A generative model is assumed [cf. Kearns and Singh, 1999, Sidford et al., 2018a, Li et al., 2021a]. In iteration t , we collect independent samples of rewards r t ( s, a ) and the next state s t ( s, a ) ∼ P ( ·| s, a ) for every state-action pair ( s, a ) ∈ S × A . We summarize the observations into the reward vector r t = ( r t ( s, a )) ( s,a ) ∈ R D and the empirical

transition matrix P t = ( e s t ( s,a ) ) ( s,a ) ∈ R D × S with each row a one-hot vector. We introduce the transition matrix P ∈ R D × S to represent the probability transition kernel P , whose ( s, a ) -th row P s,a is a probability vector representing P ( ·| s, a ) . The square probability transition matrix P π ∈ R D × D (resp. P π ∈ R S × S ) induced by the deterministic policy π over the state-action pairs (resp. states) is

<!-- formula-not-decoded -->

where Π π ∈ R S × D is a projection matrix associated with a given policy π :

<!-- formula-not-decoded -->

where π ( ·| s ) ∈ R A is the policy vector at state s .

Q-learning. The synchronous Q-learning algorithm maintains a Q-function vector, Q t ∈ R D , for all t ≥ 0 and updates its entries via the following update rule:

<!-- formula-not-decoded -->

where η t ∈ (0 , 1] is the step size in the t -th iteration and ̂ T t : R D → R D is the empirical Bellman operator constructed by samples collected in the t -th iteration:

<!-- formula-not-decoded -->

with r t ( s, a ) ∼ R ( s, a ) and s t = s t ( s, a ) ∼ P ( ·| s, a ) for each state-action pair ( s, a ) ∈ S × A . In matrix form, ̂ T t Q t -1 = P t V t -1 where V t -1 ( s ) = max a Q t -1 ( s, a ) is the greedy value. Clearly, ̂ T t is an unbiased estimate of the Bellman operator T : R D → R D given by

<!-- formula-not-decoded -->

The optimal Q ∗ is the unique fixed point of the Bellman operator, T Q ∗ = Q ∗ . Let π t be the greedy policy w.r.t. Q t ; i.e., π t ( s ) ∈ arg max a ∈A Q t ( s, a ) for s ∈ S and π ∗ the optimal policy.

Averaged Q-learning. Ruppert [1988] and Polyak and Juditsky [1992] showed that averaging the iterates generated by a stochastic approximation (SA) algorithm has favorable asymptotic statistical properties. There is a line of work which has adapted Polyak-Ruppert averaging to the problem of policy evaluation in RL [Bhandari et al., 2018, Khamaru et al., 2021a, Mou et al., 2020a]. Q-learning is different than policy evaluation due to the nonstationarity (i.e., π t changes over time) and the nonlinearity of T . The averaged Q-learning iterate has the form

<!-- formula-not-decoded -->

with { Q t } t ≥ 0 updated as in Eq. (3) and T is the number of iterates. When we conduct inference, we use the average estimate ¯ Q T rather than the last iterative value Q T given an iteration budget T . The application of Polyak-Ruppert averaging in deep RL has been shown empirically to have benefits in terms of error reduction and stability [Lillicrap et al., 2016, Anschel et al., 2017].

Bellman noise. Let Z t ∈ R D be the Bellman noise at the t -th iteration, whose ( s, a ) -th entry is

<!-- formula-not-decoded -->

In matrix form, the Bellman noise at iteration t can be equivalently presented as Z t = ( r t -r ) + γ ( P t -P ) V ∗ . The Bellman noise Z t reflects the noise present in the empirical Bellman operator (4) using samples collected at iteration t as an estimate of the population Bellman operator (5).

In our synchronous setting, r t and P t are independent of each other and the past history. Therefore, { Z t } is an i.i.d. random vector sequence with coordinates that are mean zero and mutually independent. When it is clear from the context, we drop the dependence on t and use Z to denote an independent copy of Z t . We refer to Z as the Bellman noise (vector). Finally, an important quantity in our analysis is the covariance matrix of Z :

<!-- formula-not-decoded -->

where the expectation E r t ,s t ( · ) is taken over the randomness of rewards r t and states s t . Clearly, Var( Z ) is a diagonal matrix with the ( s, a ) -th diagonal entry given by E Z 2 t ( s, a ) .

## 3 FUNCTIONAL CENTRAL LIMIT THEOREM FOR PARTIAL-SUM AVERAGED Q-LEARNING

Our main result is a functional central limit theorem for the partial-sum process of averaged Q-learning. To that end, we make three assumptions. The first is that all random rewards have uniformly bounded fourth moments (Assumption 3.1). Though typical in the SA literature [Borkar, 2009], it is weaker than the uniform boundedness assumption which is often used for nonasymptotic analysis in RL. It is required for a technical reason (that we should ensure a residual error vanishes uniformly in probability, a result which is one of our technical contributions).

The second is a Lipschitz condition (Assumption 3.2) over a specific optimal policy π ∗ ∈ Π ∗ , where Π ∗ collects all optimal policies. The condition is true when | Π ∗ | = 1 (See Lemma B.1 for the reason). Similar assumptions have been adapted for asymptotic analysis for general nonlinear SA [Mokkadem and Pelletier, 2006], and nonasymptotic analysis for both variance reduced Q-learning [Khamaru et al., 2021b] and policy iteration [Puterman and Brumelle, 1979]. The condition implies that when Q t ≈ Q ∗ the asymptotic behavior of averaged Q-learning is captured by

a linear system up to a high-order approximation error. As a result, we can explicitly formulate the asymptotic variance matrix. The approach of approximating a nonlinear SA by a specific linear SA and analyzing the approximation errors is also standard in the SA literature [Polyak and Juditsky, 1992, Mokkadem and Pelletier, 2006, Lee et al., 2021, Li et al., 2022].

The last assumption (Assumption 3.3) requires that the step size decays at a sufficiently slow rate; this is necessary in order to establish asymptotic normality [Polyak and Juditsky, 1992, Su and Zhu, 2018, Chen et al., 2020a, Li et al., 2022]. A typical example satisfying Assumption 3.3 is the polynomial step size, η t = t -α with α ∈ (0 . 5 , 1) .

Assumption 3.1. We assume E | R ( s, a ) | 4 &lt; ∞ for all ( s, a ) ∈ S × A .

Assumption 3.2. There exists π ∗ ∈ Π ∗ such that for any Q -function estimator Q ∈ R D , ‖ ( P π Q -P π ∗ )( Q -Q ∗ ) ‖ ∞ ≤ L ‖ Q -Q ∗ ‖ 2 ∞ where π Q ( s ) := arg max a ∈A Q ( s, a ) is the greedy policy w.r.t. Q .

Assumption 3.3. Assume (i) 0 ≤ sup t η t ≤ 1 , η t ↓ 0 and tη t ↑ ∞ ; (ii) η t -1 -η t η t -1 = o ( η t -1 ) ; (iii) 1 √ T ∑ T t =0 η t → 0 for all t ≥ 1 ; (iv) ∑ T t =0 η t Tη T ≤ C for all T ≥ 1 .

We now present the functional CLT for averaged Q-learning under the same conditions. Define the standardized partialsum processes associated with { Q t } t ≥ 0 as follows:

<!-- formula-not-decoded -->

where r ∈ [0 , 1] is the fraction of the data used to compute the partial-sum process and glyph[floorleft]·glyph[floorright] returns the largest integer smaller than or equal to the input number.

Theorem 3.1. Under Assumptions 3.1, 3.2 and 3.3, we have

<!-- formula-not-decoded -->

where Var Q ∈ R D × D is the asymptotic variance

<!-- formula-not-decoded -->

and B D ( · ) ∈ R D is a standard Brownian motion on [0 , 1] .

The conventional CLT asserts that φ T (1) = √ T ( ¯ Q T -Q ∗ ) converges in distribution to a rescaled Gaussian random variable Var 1 / 2 Q B D (1) as T → ∞ (see Appendix B for more details). The functional CLT in Theorem 3.1 extends this convergence to the whole function φ T = { φ T ( r ) } r ∈ [0 , 1] in the sense that any finitedimensional projections of φ T converge in distribution. That is, for any given integer n ≥ 1 and any 0 ≤ t 1 &lt; · · · &lt; t n ≤ 1 , as T → ∞ , ( φ T ( t 1 ) , · · · , φ T ( t n )) d → Var 1 / 2 Q ( B D ( t 1 ) , · · · , B D ( t n )) . The convergence w → in (9)

Figure 1: Empirical coverage rates (left) and CI lengths (right) of ¯ Q T ( s 0 , a 0 ) against the number of iterations T on a specific ( s 0 , a 0 ) . Both are obtained by averaging over 500 independent Q-learning trajectories. Black dashed line denotes the nominal coverage rate of 95%.

<!-- image -->

also corresponds to the weak convergence of measures in the D -dimensional Skorokhod spaces D ([0 , 1] , R D ) (see Appendix C.1.1 for a short introduction). Here D ([0 , 1] , R D ) = { right continuous with left limits ω ( r ) ∈ R D , r ∈ [0 , 1] } . Eq. (9) is equivalent to the convergence of finite-dimensional projections.

Theorem 3.1 can be viewed as a generalization of Donsker's theorem [Donsker, 1951] to Q-learning iterates. Donsker's theorem shows the partial-sum process of a sequence of independent and identically distributed (i.i.d.) random variables weakly converges to a standard Brownian motion, while subsequent works extend this functional result to weakly dependent stationary sequences [Dudley, 2014]. Since in our case π t and V t might depend on history data arbitrarily, { Q t } t ≥ 0 is neither i.i.d. nor stationary. To prove the functional CLT, we use a particular error decomposition and partial-sum decomposition. We give a proof sketch in Section 3.2.

Comparison with previous (functional) CLTs. Most CLT results consider linear SA which is non-applicable here (see Mou et al. [2020a,b] and references therein). The original result for Polyak-Ruppert averaging [Polyak and Juditsky, 1992, Moulines and Bach, 2011, Durmus et al., 2022] also doesn't apply in our case because it assumes a locally strongly convex Lyapunov function-which is not known to exist for Q-learning. Konda and Tsitsiklis [1999] shows Q T -Q ∗ √ η T d → N ( 0 , Var) with Var = lim T η T E ( Q T -Q ∗ )( Q T -Q ∗ ) glyph[latticetop] when we assume the limit involved exists. Mokkadem and Pelletier [2006] shows φ T (1) d →N ( 0 , Var Q ) under a similar Lipschitz condition Assumption 3.2.

To date, formal functional CLT results for SA are mainly based on the ODE approach [Abounadi et al., 2002, Borkar,

2009, Gadat et al., 2018, Borkar et al., 2021]. These works focus on the asymptotic behavior of the interpolated process connecting properly rescaled last iterates. An example interpolated process ˜ φ T ( · ) satisfies ˜ φ T (0) = Q T -Q ∗ √ η T and ˜ φ T ( t T k ) = Q T + k -Q ∗ √ η T + k for a specific sequence { t T k } k ≥ 0 depending on the step size and satisfying t T 0 = 0 and lim k t T k = ∞ . This functional CLT result implies ˜ φ T ( · ) converges weakly to the solution of a specific SDE. Theorem 3.1 is different because it is concerned with the partialsum process φ T ( · ) and explicitly formulates the asymptotic variance Var Q . Recent work studying statistical inference via SGD variants also provides functional CLTs for a similar partial-sum process [Lee et al., 2021, Li et al., 2022], given the loss function is smooth and strongly convex. However, those results don't apply here since Q-learning doesn't meet the underlying assumptions. Our functional CLT for the partial-sum process of Q-learning is novel.

## 3.1 Online Statistical Inference

The functional CLT opens a path towards statistical inference in RL. While traditional approaches estimate asymptotic variances in RL by batch-mean estimators [Chen et al., 2020a, Zhu et al., 2021] or bootstrapping [Hao et al., 2021], by contrast, the functional CLT allows us to construct an asymptotically pivotal statistic using the whole function φ T . The inference method, known as random scaling, was originally designed for strongly convex optimization [Lee et al., 2021, Li et al., 2022].

Proposition 3.1. The continuous mapping theorem together with Theorem 3.1 yields that with probability approaching one, ∫ 1 0 φ T ( r ) φ T ( r ) glyph[latticetop] dr is invertible and

<!-- formula-not-decoded -->

where ¯ φ T ( r ) := φ T ( r ) -r · φ T (1) and ¯ B D ( r ) := B D ( r ) -r · B D (1) for simplicity.

The left-hand side of (11) is a pivotal quantity involving samples and the unobservable parameter of interest Q ∗ . The pivotal quantity can be constructed in a fully online fashion and thus is computationally efficient. 1 The righthand side of (11) is a known distribution whose quantiles can be computed via simulation [Kiefer et al., 2000, Abadir and Paruolo, 2002]. In this way, we don't need a consistent estimator for the asymptotic variance in order to provide asymptotically valid confidence intervals for Q ∗ , as are required by previous work [Hao et al., 2021, Shi et al., 2020, Khamaru et al., 2022]. As an illustration, Figure 1

1 See Algorithm 1 in [Lee et al., 2021] or Algorithm 2 in [Li et al., 2022] for the online procedure.

shows the empirical coverage rates and confidence interval (CI) lengths on a random MDP with three values of γ . As T increases, the empirical coverage rates increase rapidly, approaching 95% , and the CI lengths decay. More details are placed in Appendix J.

## 3.2 Proof Sketch

In the part, we provide a proof sketch of Theorem 3.1 to highlight our technical contributions. A full proof of Theorem 3.1 is provided in Appendix C.

Step 1: Error decomposition. Let ∆ t = Q t -Q ∗ . Recall that the Q-learning update rule is (3). It follows that

<!-- formula-not-decoded -->

where Z t = ( r t -r ) + γ ( P t -P ) V ∗ is the Bellman noise. Notice that P t ( V t -1 -V ∗ ) = ( P t -P )( V t -1 -V ∗ ) + P ( V t -1 -V ∗ ) . Using PV t -1 = P π t -1 Q t -1 and PV ∗ = P π ∗ Q ∗ , we further have P ( V t -1 -V ∗ ) = P π t -1 Q t -1 -P π ∗ Q ∗ = ( P π t -1 -P π ∗ ) Q t -1 + P π ∗ ∆ t -1 . Putting the pieces together,

<!-- formula-not-decoded -->

where A t = I -η t G , G = I -γ P π ∗ , Z ′ t = ( P t -P )( V t -1 -V ∗ ) , and Z ′′ t = ( P π t -1 -P π ∗ ) Q t -1 . Recursing the last equality gives

<!-- formula-not-decoded -->

In addition, using the general step size in Assumption 3.3, we can show 1 √ T ∑ T t =1 E ‖ ∆ t ‖ 2 ∞ → 0 (in Theorem E.1).

Step 2: Partial-sum decomposition. For simplicity, for any T ≥ j ≥ 0 we denote

<!-- formula-not-decoded -->

Setting ψ 0 ( r ) := 1 η 0 √ T ( A glyph[floorleft] Tr glyph[floorright] 0 -η 0 I ) ∆ 0 and plugging (12) into φ T ( r ) = 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] t =1 ∆ t , yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3: Establish the functional CLT. To measure the distance between random functions, we define ‖ ψ ‖ sup = sup r ∈ [0 , 1] ‖ ψ ( r ) ‖ ∞ . The standard martingale functional CLT [Hall and Heyde, 2014, Jirak, 2017] implies ψ 1 ( · ) w → Var 1 / 2 Q B D ( · ) . To complete the proof, it suffice to show ‖ φ T -ψ 1 ‖ sup = o P (1) which is implied by ‖ ψ i ‖ sup = o P (1) for i = 0 , 2 , 3 , 4 , 5 .

By Lemma 1 in [Polyak and Juditsky, 1992], we know sup T ≥ j ≥ 0 ‖ A T j ‖ ∞ ≤ C 0 and lim T →∞ 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 = 0 . Then it is obvious ‖ ψ 0 ‖ sup = o P (1) . Noting that Z j , Z ′ j are martingale differences, we can show E ‖ ψ i ‖ 2 sup = o (1) for i = 2 , 3 by Doob's inequality.

By definition of greedy policies π ∗ and π t -1 , we know P π ∗ Q t -1 ≤ P π t -1 Q t -1 and P π t -1 Q ∗ ≤ P π ∗ Q ∗ , which implies ‖ Z ′′ t ‖ ∞ = ‖ ( P π t -1 -P π ∗ ) Q t -1 ‖ ∞ ≤ ‖ ( P π t -1 -P π ∗ ) ∆ t -1 ‖ ∞ ≤ L ‖ ∆ t -1 ‖ 2 ∞ from Assumption 3.2. Then E ‖ ψ 5 ‖ sup ≤ LC 0 √ T ∑ T t =1 E ‖ ∆ t ‖ 2 ∞ → 0 .

The most challenging step is to show ‖ ψ 4 ‖ sup = o P (1) . Notice that ψ 4 is a weighted sum of martingale differences, Z j + γ Z ′ j , with the coefficients varying in r such that we can't apply Doob's inequality. To deal with this issue, we relate ψ 4 to an autoregressive sequence indexed by k ∈ [ T ] and analyze the maximum over k directly. More specifically, we can show

<!-- formula-not-decoded -->

Previous results Lee et al. [2021], Li et al. [2022] do not apply here, since they require G = I -γ P π ∗ to be positive semidefinite, which isn't our case. Noticing that all eigenvalues of G have nonnegative real parts, we provide a novel analysis of the right-hand side in Lemma D.1, showing it is indeed o P (1) under Assumption 3.1. This is one of our technical contributions.

Remark 3.1. If we consider policy evaluation (so that π t remains unchanged and ψ 5 disappears), ψ 4 is still present. Showing ‖ ψ 4 ‖ sup = o P (1) is required even for linear SA.

## 4 INFORMATION-THEORETIC LOWER BOUND

The standard CLT implies ¯ Q T is a √ T -consistent estimate for Q ∗ . It is of theoretical interest to investigate whether or not ¯ Q T is asymptotically efficient. In parametric statistics [Lehmann and Casella, 2006], the Cramer-Rao lower bound assesses the hardness of estimating a target parameter β ( θ ) in a parametric model P θ indexed by parameter θ . Any unbiased estimator whose variance achieves the Cramer-Rao lower bound is viewed as optimal and efficient. The concept of Cramer-Rao lower bounds can be extended to possibly biased but asymptotically unbiased estimators and also to nonparametric statistical models where the dimension of the parameter θ is infinity [Van der Vaart, 2000, Tsiatis, 2006].

The semiparametric model. In our case, the transition kernel { P ( ·| s, a ) } s,a is specified by D parametric distributions on D , while the random reward { R ( s, a ) } s,a is fully nonparametric because the R ( s, a ) are not assumed to come from finite-dimensional models. Hence, to derive an extended Cramer-Rao lower bound for Q ∗ estimation, we need to enter the world of semiparametric statistics. In particular, our MDP model M = ( S , A , γ, P, R, r ) has parameter θ = ( P, R ) . Our parameter of interest is β ( θ ) = Q ∗ . At iteration t , we observe the random rewards and empirical transitions for each ( s, a ) and concatenate them into r t ∈ R D and P t ∈ R D × S . The distribution of P t is determined by its expectation P = E P t , which belongs to

<!-- formula-not-decoded -->

while R is nonparametric and belongs to

<!-- formula-not-decoded -->

According to the generative model, the r t and P t are mutually independent and also independent of the historical data. Let D = { ( r t , P t ) } t ∈ [ T ] contain the T samples generated as described above.

Semiparametric efficiency lower bound. Tsiatis [2006] has argued that regular asymptotically linear (RAL) estimators provide a good tradeoff between expressivity and tractability. In RL, RAL estimators are widely considered in off-policy evaluation problems [Kallus and Uehara, 2020]. Definition 4.1 (Regular estimator) . Denote the distribution of r t and P t by L ( r ) and L ( P ) . 2 For any given T , let

2 Given a probability space (Ω , P, F ) , L ( X ) is the law of the random variable X in this probability space. Since r t are i.i.d., they share the same distribution L ( r ) and similarly for L ( P ) .

L T ( r ) and L T ( P ) be the perturbed distributions of L ( r ) and L ( P ) which are consistent in the sense that they converge 3 to L ( r ) and L ( P ) when T goes infinity. Let ̂ Q T be any estimator of Q ∗ computed from D . Let Q ∗ T be the true optimal Q-value function when rewards and transition probabilities are generated i.i.d. from L T ( r ) and L T ( P ) . We say ̂ Q T is a regular estimator of Q ∗ if √ T ( ̂ Q T -Q ∗ T ) weakly converges to a random variable that depends only on L ( r ) and L ( P ) , when samples are distributed according to the probability measure ( L T ( r ) , L T ( P )) .

Remark 4.1. Informally speaking, an estimator is regular if its limiting distribution is unaffected by local changes in the data-generating process. The assumption of regularity excludes super-efficient estimators, whose asymptotic variance can be smaller than the Cramer-Rao lower bound for some parameter values, but which perform poorly in the neighborhood of points of super-efficiency. We refer interested readers to Section 3.1 in [Tsiatis, 2006] for a detailed exposition.

Definition 4.2 (Regular asymptotically linear) . Let ̂ Q T ∈ R D be a measurable random function of D = { ( r t , P t ) } t ∈ [ T ] . We say that ̂ Q T is regular asymptotically linear (RAL) for Q ∗ if it is regular and asymptotically linear with a measurable random function φ ( r t , P t ) ∈ R D such that

<!-- formula-not-decoded -->

Here φ ( · , · ) is referred to as an influence function , and it satisfies E φ ( r t , P t ) = 0 and E φ ( r t , P t ) φ ( r t , P t ) glyph[latticetop] .

Theorem 4.1. Given the dataset D = { ( r t , P t ) } t ∈ [ T ] , for any RAL estimator ̂ Q T of Q ∗ computed from D = { ( r t , P t ) } t ∈ [ T ] , its variance satisfies

<!-- formula-not-decoded -->

where A glyph[followsequal] B means A -B is positive semidefinite and Var Q is given in (10) .

By Definition 4.2, any influence function determines an asymptotic linear estimator for Q ∗ . The semiparametric efficiency bound in Theorem 4.1 gives us a concrete target in the construction of the influence function. If we can find an influence function that achieves the bound, we know that it is the most efficient among all RAL estimators. Fortunately, Theorem 4.2 implies that ¯ Q T is the most efficient estimator among all RAL estimators with the efficient influence function ( I -γ P π ∗ ) -1 Z t . It also implies that for any fixed r ∈ [0 , 1] , φ T ( r ) = √ r · √ glyph[floorleft] Tr glyph[floorright] ( ¯ Q glyph[floorleft] Tr glyph[floorright] -Q ∗ ) has the optimal asymptotic variance (scaled by a factor √ r ). Proofs are provided in Appendix G.

3 L T ( r ) and L T ( P ) are differentiable in quadratic mean at L ( r ) and L ( P ) . See Chapter 25.3 in Van der Vaart [2000].

Theorem 4.2. Under Assumptions 3.1, 3.2 and 3.3, the averaged Q-learning iterate ¯ Q T is a RAL estimator for Q ∗ . In particular, we have the following decomposition

<!-- formula-not-decoded -->

where Z t = ( r t -r ) + γ ( P t -P ) V ∗ is the Bellman noise at iteration t .

## 5 INSTANCE-DEPENDENT NONASYMPTOTIC CONVERGENCE

In the section, we explore the nonasymptotic behavior of averaged Q-learning, i.e., we study the dependence of E ‖ ¯ Q T -Q ∗ ‖ ∞ on finite T and (1 -γ ) -1 .

Theorem 5.1. Let Assumptions 3.2 hold and 0 ≤ R ( s, a ) ≤ 1 for all ( s, a ) ∈ S ×A . 4 When D is larger than a universal constant,

- If η t = t -α with α ∈ (0 . 5 , 1) for t ≥ 1 and η 0 = 1 , it follows that for all T ≥ 1 , E ‖ ¯ Q T -Q ∗ ‖ ∞ =

<!-- formula-not-decoded -->

- If η t = 1 1+(1 -γ ) t , it follows that for all T ≥ 1 , E ‖ ¯ Q T -Q ∗ ‖ ∞ =

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here ˜ O ( · ) hides polynomial dependence on α, L and logarithmic factors (i.e., ln D and ln T ).

Instance-dependent behavior. For the polynomial step size, Theorem 5.1 shows that the instance-dependent term O ( √ ‖ diag(Var Q ) ‖ ∞ √ ln D T ) dominates the glyph[lscript] ∞ error, which matches the instance-dependent lower bound established by Khamaru et al. [2021b] given a sufficiently large T . To the best of our knowledge, this is the first finite-sample analysis of averaged Q-learning in the glyph[lscript] ∞ -norm showing instance-dependent optimality. However, for the linearly

4 To simplify the parameter dependence, we assume rewards are uniformly bounded as in previous work [Wainwright, 2019c, Khamaru et al., 2021b, Li et al., 2021a] . Note that, thanks to the error decomposition in (14), it is possible to provide a nonasymptotic analysis assuming rewards have finite second moments. The consequence is that the dependence on d and δ would change from log D, log 1 δ to D and 1 δ .

rescaled step size, we see that O ( √ ‖ Var( Z ) ‖ ∞ (1 -γ ) 2 √ ln D T ) is the dominant factor, which is larger because we have

<!-- formula-not-decoded -->

where ( a ) uses ‖ diag( AV A glyph[latticetop] ) ‖ ∞ ≤ ‖ V ‖ ∞ ‖ A ‖ 2 ∞ for any diagonal matrix V (see Lemma F.2) and ( b ) uses ‖ ( I -γ P π ∗ ) -1 ‖ ∞ ≤ (1 -γ ) -1 . Hence, the linearly rescaled step size doesn't match the instance-dependent lower bound. It might be true because the linearly rescaled step size doesn't satisfy Assumption 3.3, implying that (22) does not necessarily hold for it.

Comparison with variance-reduced Q-learning. Under the same assumptions, Khamaru et al. [2021b] analyzed a variance-reduced variant of Q-learning that also achieves instance-dependent optimality with the following guarantee:

<!-- formula-not-decoded -->

which has a better nonleading term than averaged Qlearning. This might somewhat explain the finding of Khamaru et al. [2021a] that averaging can be sub-optimal in the nonasymptotic regime with limited samples. However, the dominant terms are equal, implying that averaging is still powerful and efficient in the asymptotic regime. Instance-dependent convergence with a variance structure in the dominant term has also been found for other settings; please see Appendix A.

Worst-case behavior. The instance-dependent bound provides more information about the convergence rate. Previous works [Azar et al., 2013, Li et al., 2020a] imply the worst-case bound ‖ diag(Var Q ) ‖ ∞ = O ((1 -γ ) -3 ) . Such a dependence on (1 -γ ) -1 is tight, because Khamaru et al. [2021b] constructs a family of MDPs parameterized by λ ≥ 0 where ‖ diag(Var Q ) ‖ ∞ = Θ((1 -γ ) -3+ λ ) . When plugging in the worst-case bound, we find that for polynomial step sizes and for sufficiently small ε , averaged Q-learning already achieves the optimal minimax sample complexity ˜ O ( D (1 -γ ) 3 ε 2 ) established by Azar et al. [2013]. Wainwright [2019c] uses a variance-reduced variant of Qlearning to achieve the optimality, but the algorithm requires an additional collection of i.i.d. samples at each outer loop to obtain an Monte Carlo approximation of the population Bellman operator (5). Our results show that a simple average is sufficient to guarantee optimality. Moreover, the computation of ¯ Q T is fully online with no additional samples needed.

Figure 2: Log-log plots of the sample complexity T ( ε, γ ) versus the asymptotic variance ‖ diag(Var Q ) ‖ ∞ .

<!-- image -->

Confirming the theoretical predictions. We provide numerical experiments to illustrate instance-adaptivity as well as the worst-case behavior delineated in Theorem 5.1. We focus on the sample complexity T ( ε, γ ) = inf { T : E ‖ ¯ Q T -Q ∗ ‖ ∞ ≤ ε } for ε = 10 -4 . We conduct 10 3 independent trials in a random MDP to compute T ( ε, γ ) for different values of γ ∈ Γ and two step sizes. We plot the least-squares fits, { (log ‖ diag(Var Q ) ‖ ∞ , log T ( ε, γ )) } γ ∈ Γ , and provide the slopes k of these lines in the legend. Further details are provided in Appendix J. At a high level, we see that averaged Q-learning produces sample complexity that is well predicted by our theory-all the slopes are no larger than the theoretical limit k predicted by our theory.

Proof Sketch. The proof idea of Theorem 5.1 is based on that of Theorem 3.1. Notice that ¯ Q T -Q ∗ = 1 T ∑ T t =1 ∆ t = 1 √ T φ T (1) . From (14), we know that 5

glyph[negationslash]

<!-- formula-not-decoded -->

Bounding the term of i = 0 is easy since it's deterministic. Because ψ i (1)( i = 1 , 2 , 3) is a weighted sum of martingale differences, we use the variance-aware multi-dimensional Freedman's inequality (in Lemma H.1) to analyze its expectation under glyph[lscript] ∞ -norm. The instance-dependent dominant term comes from the variance term for E ‖ ψ 1 (1) ‖ ∞ . Analyzing the variance of ψ 2 (1) is quite challenging since it relies on 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 ∞ with A T j defined in (13). We then bound that quantity in terms of α, 1 -γ and T in Lemma C.4. Finally, due to ‖ ψ 5 (1) ‖ ∞ ≤ L ‖ ∆ t ‖ 2 ∞ , bounding E ‖ ψ 5 (1) ‖ ∞ is reduced to bound E ‖ ∆ t ‖ 2 ∞ for all t ≥ 0 , which can be given by a similar argument from Wainwright [2019b]. Putting all pieces together completes the proof; the detailed proof is in Appendix F.

5 Since r = 1 , ψ 4 doesn't appear in the decomposition.

## 6 RELAXATION OF THE LIPSCHITZ CONDITION

Both our asymptotic and nonasymptotic analysis rely on the Lipschitz condition in Assumption 3.2. That condition is essentially equivalent to assuming a unique optimal policy. It turns out that, once regularized by entropy, the (regularized) optimal policy is naturally unique. In the following, we show that entropy-regularized Q-learning enjoys a similar functional CLT and instance-dependent bounds without Assumption 3.2.

Entropy-regularized Q-learning uses the following matrixform update rule,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is a soft version of the empirical Bellman operator ̂ T . The nonlinear operator L λ ( · ) : R D → R S is a soft version of a hard max, with regularization coefficient λ . It is defined by

<!-- formula-not-decoded -->

Let Q ∗ λ denote the unique fixed point of the regularized Bellman equation Q ∗ λ = r + γ P L λ Q ∗ λ and let π ∗ λ be the unique optimal policy.

Theorem 6.1. Define { ˜ Q t } t ≥ 0 in (16) . The corresponding partial-sum process is ˜ φ T ( r ) := 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] t =1 ( ˜ Q t -Q ∗ λ ) . Under Assumptions 3.1 and 3.3,

<!-- formula-not-decoded -->

where ˜ Var Q is the asymptotic matrix defined by

<!-- formula-not-decoded -->

with ˜ Z d. = ˜ Z t = ( r t -r )+ γ ( P t -P ) L λ Q ∗ λ the regularized Bellman noise.

Theorem 6.2. Under Assumptions 3.1 and 3.3, when the two step sizes are considered, E ‖ 1 T ∑ T t =1 ˜ Q t -Q ∗ λ ‖ ∞ has similar bounds as in Theorem 5.1 except that we replace Var Q , L with ˜ Var Q and 1 λ .

We note that the two theorems in this section can be proved via an almost identical argument as Theorem 3.1 and 5.1, since Assumption 3.2 is naturally satisfied with L = 1 λ for entropy-regularized Q-learning (see Appendix I). Actually, our proof is applicable to a class of nonlinear SAs. 6 Second, due to the bias introduced by entropy, the

6 More specifically, our method can analyze Q t = (1 -η t ) Q t -1 + η t ( r t + γ P t L Q t -1 ) where L is a smooth nonlinear non-expansive operator.

instance-dependent factor changes from Var Q to ˜ Var Q and 1 T ∑ T t =1 ˜ Q t converges to Q ∗ λ instead of Q ∗ in expectation. Finally, note that these results provide a new argument for the benefits of entropy regularization; it smooths the Bellman operator and weakens the assumptions required for asymptotic analysis. It is supplementary to previous efforts that shows entropy regularization aids exploration [Fox et al., 2016], encourages robust optimal policies [Eysenbach and Levine, 2021], induces a smoother landscape [Ahmed et al., 2019], and hastens the convergence of RL algorithms [Cen et al., 2022].

## 7 DISCUSSION

We have studied the asymptotic and nonasymptotic convergence of averaged Q-learning, establishing its statistical efficiency. We first established a functional central limit theorem, showing that the standardized partial-sum process converges weakly to a rescaled Brownian motion, a result which can serve as an underpinning for the development of statistical inference methods for RL. We then established a semiparametric efficiency lower bound for Q ∗ estimation, showing that the averaged iterate ¯ Q T is the most efficient RAL estimator in the sense of having the smallest asymptotic variance. Finally, we presented the first finite-sample error analysis of E ‖ ¯ Q T -Q ∗ ‖ ∞ in the glyph[lscript] ∞ -norm for both linearly rescaled and polynomial step sizes. We showed that averaged Q-learning achieves the same instance-dependent optimality and worst-case optimality as previous variancereduced algorithms [Khamaru et al., 2021b, Wainwright, 2019c] under a Lipschitz condition.

Some open problems remain. On the one hand, with the Lipschitz condition, it's unclear whether averaged Q-learning with linearly rescaled step sizes can match the instancedependent lower bound. Additionally, we suspect that the dependence on (1 -γ ) -1 of the nonleading terms in Theorem 5.1 is loose and speculate it can be improved by finer analysis. On the other hand, without the Lipschitz condition, it is not clear whether averaged Q-learning still achieves the optimal instance-dependent bound. Finally, previous analysis [Kozuno et al., 2022] shows the last-iterate entropyregularized Q-learning is minimax optimal. It is also unknown whether the averaged iterates of entropy-regularized Q-learning achieve the optimal instance-dependent bound.

## Acknowledgement

The authors would like to express their gratitude to Prof. Csaba Szepesvári for his valuable suggestion regarding the relaxation of the Lipschitz condition through entropy regularization. Xiang Li and Zhihua Zhang have been supported by the National Key Research and Development Project of China (No. 2022YFA1004002) and the National Natural Science Foundation of China (No. 12271011).

## References

- Karim M Abadir and Paolo Paruolo. Simple robust testing of regression hypotheses: A comment. Econometrica , 70 (5):2097-2099, 2002.
- Jinane Abounadi, Dimitri P Bertsekas, and Vivek Borkar. Stochastic approximation for nonexpansive maps: Application to Q-learning algorithms. SIAM Journal on Control and Optimization , 41(1):1-22, 2002.
- Alekh Agarwal, Sham Kakade, and Lin F Yang. Modelbased reinforcement learning with a generative model is minimax optimal. In Conference on Learning Theory , pages 67-83. PMLR, 2020.
- Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning , pages 151-160. PMLR, 2019.
- Oron Anschel, Nir Baram, and Nahum Shimkin. Averageddqn: Variance reduction and stabilization for deep reinforcement learning. In International Conference on Machine Learning , pages 176-185. PMLR, 2017.
- Mohammad Gheshlaghi Azar, Rémi Munos, and Hilbert J Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning , 91(3):325-349, 2013.
- Necdet Batir. Inequalities for the gamma function. Archiv der Mathematik , 91(6):554-563, 2008.
- Carolyn L Beck and Rayadurgam Srikant. Error bounds for constant step-size Q-learning. Systems and Control Letters , 61(12):1203-1208, 2012.
- Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Conference on Learning Theory , pages 1691-1692. PMLR, 2018.
- Vivek Borkar, Shuhang Chen, Adithya Devraj, Ioannis Kontoyiannis, and Sean Meyn. The ODE method for asymptotic statistics in stochastic approximation and reinforcement learning. arXiv preprint arXiv:2110.14427 , 2021.
- Vivek S Borkar. Stochastic Approximation: A Dynamical Systems Viewpoint , volume 48. Springer, 2009.
- Vivek S Borkar and Sean P Meyn. The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38 (2):447-469, 2000.
- Donald L Burkholder. Sharp inequalities for martingales and stochastic integrals. Astérisque , 157(158):75-94, 1988.
- Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research , 70(4):2563-2578, 2022.
- Xi Chen, Jason D Lee, Xin T Tong, and Yichen Zhang. Statistical inference for model parameters in stochastic gradient descent. Annals of Statistics , 48(1):251-273, 2020a.
- Zaiwei Chen, Sheng Zhang, Thinh T Doan, John-Paul Clarke, and Siva Theja Maguluri. Finite-sample analysis of nonlinear stochastic approximation with applications in reinforcement learning. arXiv preprint arXiv:1905.11425 , 2019.
- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysis of stochastic approximation using smooth convex envelopes. arXiv e-prints , pages arXiv-2002, 2020b.
- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. A Lyapunov theory for finite-sample guarantees of asynchronous Q-learning and TD-learning variants. arXiv preprint arXiv:2102.01567 , 2021.
- Monroe David Donsker. An invariance principle for certain probability limit theorems. American Mathematical Society , pages 1-10, 1951.
- Richard M Dudley. Uniform Central Limit Theorems , volume 142. Cambridge University Press, 2014.
- Alain Durmus, Eric Moulines, Alexey Naumov, and Sergey Samsonov. Finite-time high-probability bounds for Polyak-Ruppert averaged iterates of linear stochastic approximation. arXiv preprint arXiv:2207.04475 , 2022.
- Eyal Even-Dar, Yishay Mansour, and Peter Bartlett. Learning rates for Q-learning. Journal of Nachine Learning Research , 5(1), 2003.
- Benjamin Eysenbach and Sergey Levine. Maximum entropy RL (provably) solves some robust RL problems. In International Conference on Learning Representations , 2021.
- Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. In Conference on Uncertainty in Artificial Intelligence , pages 202-211, 2016.
- David A Freedman. On tail probabilities for martingales. Annals of Probability , pages 100-118, 1975.
- Sébastien Gadat, Fabien Panloup, and Sofiane Saadane. Stochastic heavy ball. Electronic Journal of Statistics , 12 (1):461-529, 2018.
- Robert M Gower, Mark Schmidt, Francis Bach, and Peter Richtárik. Variance-reduced methods for machine learning. Proceedings of the IEEE , 108(11):1968-1983, 2020.
- Peter Hall and Christopher C Heyde. Martingale Limit Theory and its Application . Academic Press, 2014.
- Botao Hao, Xiang Ji, Yaqi Duan, Hao Lu, Csaba Szepesvári, and Mengdi Wang. Bootstrapping statistical inference for

- off-policy evaluation. arXiv preprint arXiv:2102.03607 , 2021.
- Tommi Jaakkola, Michael I. Jordan, and Satinder P. Singh. Convergence of stochastic iterative dynamic programming algorithms. In Advances in Neural Information Processing Systems , 1993.
- Jean Jacod and Albert N Shiryaev. Skorokhod topology and convergence of processes. In Limit Theorems for Stochastic Processes , pages 324-388. Springer, 2003.
- Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning. In International Conference on Machine Learning , pages 652-661. PMLR, 2016.
- Moritz Jirak. On weak invariance principles for partial sums. Journal of Theoretical Probability , 30(3):703-728, 2017.
- Nathan Kallus and Masatoshi Uehara. Double reinforcement learning for efficient off-policy evaluation in Markov decision processes. J. Mach. Learn. Res. , 21(167):1-63, 2020.
- Michael Kearns and Satinder Singh. Finite-sample convergence rates for Q-learning and indirect algorithms. Advances in Neural Information Processing Systems , pages 996-1002, 1999.
- Michael Kearns, Yishay Mansour, and Andrew Y Ng. A sparse sampling algorithm for near-optimal planning in large Markov decision processes. Machine Learning , 49 (2):193-208, 2002.
- Koulik Khamaru, Ashwin Pananjady, Feng Ruan, Martin J Wainwright, and Michael I Jordan. Is temporal difference learning optimal? An instance-dependent analysis. SIAM Journal on Mathematics of Data Science , 3(4): 1013-1040, 2021a.
- Koulik Khamaru, Eric Xia, Martin J Wainwright, and Michael I Jordan. Instance-optimality in optimal value estimation: Adaptivity via variance-reduced Q-learning. arXiv preprint arXiv:2106.14352 , 2021b.
- Koulik Khamaru, Eric Xia, Martin J Wainwright, and Michael I Jordan. Instance-dependent confidence and early stopping for reinforcement learning. arXiv preprint arXiv:2201.08536 , 2022.
- Nicholas M Kiefer, Timothy J Vogelsang, and Helle Bunzel. Simple robust testing of regression hypotheses. Econometrica , 68(3):695-714, 2000.
- Vijaymohan Konda and John Tsitsiklis. Actor-critic algorithms. Advances in Neural Information Processing Systems, 12 , 1999.
- Tadashi Kozuno, Wenhao Yang, Nino Vieillard, Toshinori Kitamura, Yunhao Tang, Jincheng Mei, Pierre Ménard, Mohammad Gheshlaghi Azar, Michal Valko, Rémi Munos, et al. KL-entropy-regularized RL with a generative model is minimax optimal. arXiv preprint arXiv:2205.14211 , 2022.
- Harold Kushner and G George Yin. Stochastic Approximation and Recursive Algorithms and Applications , volume 35. Springer Science &amp; Business Media, 2003.
- Tor Lattimore and Marcus Hutter. Near-optimal PAC bounds for discounted MDPs. Theoretical Computer Science , 558:125-143, 2014.
- Donghwan Lee and Niao He. Target-based temporaldifference learning. In International Conference on Machine Learning , pages 3713-3722. PMLR, 2019a.
- Donghwan Lee and Niao He. A unified switching system perspective and ODE analysis of Q-learning algorithms. arXiv preprint arXiv:1912.02270 , 2019b.
- Sokbae Lee, Yuan Liao, Myung Hwan Seo, and Youngki Shin. Fast and robust online inference with stochastic gradient descent via random scaling. arXiv preprint arXiv:2106.03156 , 2021.
- Erich L Lehmann and George Casella. Theory of Point Estimation . Springer Science &amp; Business Media, 2006.
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Breaking the sample size barrier in model-based reinforcement learning with a generative model. Advances in Neural Information Processing Systems , 33: 12861-12872, 2020a.
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. arXiv preprint arXiv:2006.03041 , 2020b.
- Gen Li, Changxiao Cai, Yuxin Chen, Yuantao Gu, Yuting Wei, and Yuejie Chi. Is Q-learning minimax optimal? A tight sample complexity analysis. arXiv preprint arXiv:2102.06548 , 2021a.
- Tianjiao Li, Guanghui Lan, and Ashwin Pananjady. Accelerated and instance-optimal policy evaluation with linear function approximation. arXiv preprint arXiv:2112.13109 , 2021b.
- Xiang Li, Jiadong Liang, Xiangyu Chang, and Zhihua Zhang. Statistical estimation and online inference via local SGD. Proceedings of Thirty Fifth Conference on Learning Theory , 178:1613-1661, 2022.
- Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In International Conference on Learning of Representations , 2016.
- Daniel J Luckett, Eric B Laber, Anna R Kahkoska, David M Maahs, Elizabeth Mayer-Davis, and Michael R Kosorok. Estimating dynamic treatment regimes in mobile health using V-learning. Journal of the American Statistical Association , 2019.
- Francisco S Melo, Sean P Meyn, and M Isabel Ribeiro. An analysis of reinforcement learning with function ap-

- proximation. In International Conference on Machine Learning , pages 664-671, 2008.
- Abdelkader Mokkadem and Mariane Pelletier. Convergence rate and averaging of nonlinear two-time-scale stochastic approximation algorithms. The Annals of Applied Probability , 16(3):1671-1702, 2006.
- Terrence Joseph Moore Jr. A theory of Cramér-Rao bounds for constrained parametric models . University of Maryland, College Park, 2010.
- Wenlong Mou, Chris Junchi Li, Martin J Wainwright, Peter L Bartlett, and Michael I Jordan. On linear stochastic approximation: Fine-grained Polyak-Ruppert and nonasymptotic concentration. In Conference on Learning Theory , pages 2947-2997. PMLR, 2020a.
- Wenlong Mou, Ashwin Pananjady, and Martin J Wainwright. Optimal oracle inequalities for solving projected fixedpoint equations. arXiv preprint arXiv:2012.05299 , 2020b.
- Eric Moulines and Francis Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. Advances in Neural Information Processing Systems , 24:451-459, 2011.
- Whitney K Newey. Semiparametric efficiency bounds. Journal of Applied Econometrics , 5(2):99-135, 1990.
- Ashwin Pananjady and Martin J Wainwright. Instancedependent glyph[lscript] ∞ bounds for policy evaluation in tabular reinforcement learning. IEEE Transactions on Information Theory , 67(1):566-585, 2020.
- Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization , 30(4):838-855, 1992.
- Martin L Puterman and Shelby L Brumelle. On the convergence of policy iteration in stationary dynamic programming. Mathematics of Operations Research , 4(1):60-69, 1979.
- Guannan Qu and Adam Wierman. Finite-time analysis of asynchronous stochastic approximation and Q-learning. In Conference on Learning Theory , pages 3185-3205. PMLR, 2020.
- Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics , pages 400-407, 1951.
- David Ruppert. Efficient estimations from a slowly convergent Robbins-Monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.
- Devavrat Shah and Qiaomin Xie. Q-learning with nearest neighbors. In Advances in Neural Information Processing Systems , pages 3115-3125, 2018.
- Chengchun Shi, Sheng Zhang, Wenbin Lu, and Rui Song. Statistical inference of the value function for reinforcement learning in infinite horizon settings. arXiv preprint arXiv:2001.04515 , 2020.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin F Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5192-5202, 2018a.
- Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving Markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM, 2018b.
- Weijie J Su and Yuancheng Zhu. Uncertainty quantification for online learning and stochastic approximation via hierarchical incremental gradient descent. arXiv preprint arXiv:1802.04876 , 2018.
- Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction . MIT Press, 2018.
- Csaba Szepesvári et al. The asymptotic convergence-rate of Q-learning. Advances in Neural Information Processing Systems , pages 1064-1070, 1998.
- Anastasios A Tsiatis. Semiparametric Theory and Missing Data . Springer, 2006.
- John N Tsitsiklis. Asynchronous stochastic approximation and Q-learning. Machine Learning , 16(3):185-202, 1994.
- Masatoshi Uehara, Jiawei Huang, and Nan Jiang. Minimax weight and Q-function learning for off-policy evaluation. In International Conference on Machine Learning , pages 9659-9668. PMLR, 2020.
- Aad W Van der Vaart. Asymptotic Statistics . Cambridge University Press, 2000.
- Karel Vermeulen. Semiparametric Efficiency . Gent Universiteit, 2011.
- Martin J Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge University Press, 2019a.
- Martin J Wainwright. Stochastic approximation with conecontractive operators: Sharp glyph[lscript] ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 , 2019b.
- Martin J Wainwright. Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 , 2019c.
- Christopher Watkins. Learning from delayed rewards . PhD thesis, 1989.
- Wenhao Yang, Xiang Li, and Zhihua Zhang. A regularized approach to sparse optimal policy in reinforcement learning. In Advances in Neural Information Processing Systems , pages 5938-5948, 2019.
- Wenhao Yang, Liangyu Zhang, and Zhihua Zhang. Towards theoretical understandings of robust Markov decision processes: Sample complexity and asymptotics. arXiv preprint arXiv:2105.03863 , 2021.

- Ming Yin and Yu-Xiang Wang. Asymptotically efficient off-policy evaluation for tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 3948-3958. PMLR, 2020.
- Ming Yin and Yu-Xiang Wang. Towards instance-optimal offline reinforcement learning with pessimism. Advances in Neural Information Processing Systems , 34, 2021.
- Wanrong Zhu, Xi Chen, and Wei Biao Wu. Online covariance matrix estimation in stochastic gradient descent. Journal of the American Statistical Association , pages 1-30, 2021.

## A RELATED WORK

Due to the rapidly growing literature on Q-learning, we review only the theoretical results that are most relevant to our work. Interested readers can check references therein for more information.

Asymptotic normality in RL. Establishing asymptotic normality of an estimator permits statistical inference and the quantification of uncertainty. Existing work on statistical inference for Q-learning has focused mainly on the off-policy evaluation (OPE) problem, where one aims to estimate the value function of a given policy using pre-collected data. In this setting, a parametric Cramer-Rao lower bound has been established by Jiang and Li [2016], and asymptotic efficiency has been established for certain estimators using linear approximation [Uehara et al., 2020, Hao et al., 2021, Yin and Wang, 2020, Mou et al., 2020a] or bootstrapping [Hao et al., 2021]. Further inferential work includesthe asymptotic analysis of multi-stage algorithms [Luckett et al., 2019, Shi et al., 2020], asymptotic behavior of robust estimators [Yang et al., 2021], and work by Kallus and Uehara [2020] on a semiparametric doubly robust estimator.

In contradistinction to existing work, we establish a functional central limit theorem that captures the weak convergence of the whole trajectory rather than its endpoint. Such functional results have not been presented previously in the RL literature. Furthermore, we supplement these upper bounds with a semiparametric efficiency lower bound which additionally considers the randomness of rewards. We also show that averaged Q-learning is the most efficient RAL estimator vis-a-vis this lower bound.

Sample complexity for Q-learning. For the goal of obtaining an ε -accurate estimate of the optimal Q-function in a γ -discounted MDP in the presence of a generative model, model-based Q-value-iteration has been shown to achieve optimal minimax sample complexity ˜ O ( D ε 2 (1 -γ ) 3 ) [Azar et al., 2013, Agarwal et al., 2020, Li et al., 2020a]. In the model-free context, Wainwright [2019b] showed empirically that classical Q-learning suffers from at least worst-case fourth-order scaling in (1 -γ ) -1 in sample complexity. A complexity bound of ˜ O ( D ε 2 (1 -γ ) 5 ) has been provided [Wainwright, 2019b, Chen et al., 2020b]; this is far from the optimal though better than previous efforts [Even-Dar et al., 2003, Beck and Srikant, 2012]. Li et al. [2021a] gave a sophisticated analysis showing the complexity of Q-learning is ˜ O ( D ε 2 (1 -γ ) 4 ) and provided a matching lower bound to confirm its sharpness. Wainwright [2019c], Khamaru et al. [2021b] introduced a variance-reduced variant of Q-learning [Gower et al., 2020] that achieves the optimal sample complexity and instance complexity. Our results show that a simple average over all history Q t is sufficient to guarantee the same optimality. The averaged method is fully online without requiring additional samples and storage space.

Instance-dependent convergence in RL. Recent years have witnessed new instance-specific bounds, where an instancedependent functional of a variance structure appears as the dominant term on stochastic errors. Unlike global minimax bounds which are worst-case in nature, instance-specific bounds help identify the difficulty of estimation case by case. Such bounds have been established for policy evaluation in the tabular setting [Pananjady and Wainwright, 2020, Khamaru et al., 2021a, Li et al., 2020a] or with linear function approximation [Li et al., 2021b] and for optimal value function estimation [Yin and Wang, 2021]. The most related work to ours is by Khamaru et al. [2021b], who show that a variance-reduced variant of Q-learning achieves the instance-dependent optimality after identifying an instance-dependent lower bound for Q ∗ estimation. By contrast, our result shows that a simple average is sufficient to yield optimality.

Nonlinear stochastic approximation. Q-learning has also been studied through the lens of nonlinear stochastic approximation. From this general point of view, asymptotic convergence has been provided [Tsitsiklis, 1994, Borkar and Meyn, 2000]. On the nonasymptotic side, Q-learning is studied either in the synchronous setting [Shah and Xie, 2018, Wainwright, 2019b, Chen et al., 2020b] or the asynchronous setting where only one sample from current state-action pair is available at a time [Qu and Wierman, 2020, Li et al., 2020b, Chen et al., 2021]. The sample complexities obtained therein are far from optimal. Others consider Q-learning with linear function approximation in the glyph[lscript] 2 -norm [Melo et al., 2008, Chen et al., 2019]. Asymptotic convergence of averaged Q-learning has been studied by Lee and He [2019a,b] via the ODE (ordinary differential equation) approach. Our results are complementary to these results, including asymptotic statistical properties and finite-sample analysis in the glyph[lscript] ∞ -norm. Though peculiar to averaged Q-learning, we believe our analysis can be extended to nonlinear SA problems.

## B CENTRAL LIMIT THEOREM FOR AVERAGED Q-LEARNING

For completeness, we present a CLT for the averaged Q-learning sequence ¯ Q T := 1 T ∑ T t =1 Q t in this part. This result can be derived not only from our Theorem 3.1 but also from CLT for non-linear SA, e.g., [Mokkadem and Pelletier, 2006].

Theorem B.1 (Asymptotic normality for Q ∗ ) . Under Assumptions 3.1, 3.2 and 3.3, we have

<!-- formula-not-decoded -->

where the asymptotic variance is given by

<!-- formula-not-decoded -->

Here Var( Z ) is the covariance matrix of the Bellman noise Z defined in (7) .

Asymptotic variance. Theorem B.1 implies that the average of the sequence ( Q t ) has an asymptotic normal distribution with Var Q the asymptotic variance. Var Q includes Var( Z ) , the covariance matrix of Bellman noise Z , multiplied with a prefactor ( I -γ P π ∗ ) -1 . By a von Neumann expansion, ( I -γ P π ∗ ) -1 is equivalent to ∑ ∞ t =0 ( γ P π ∗ ) t . As argued by Khamaru et al. [2021b], the sum of the powers of γ P π ∗ accounts for the compounded effect of an initial perturbation when following the MDP induced by π ∗ . The Bellman noise Z reflects the noise present in the empirical Bellman operator (4) as an estimate of the population Bellman operator (5). Note that this implies ‖ ( I -γ P π ∗ ) -1 ‖ ≤ ∑ ∞ t =0 γ t ‖ ( P π ∗ ) t ‖ ∞ = (1 -γ ) -1 . ‖ diag(Var Q ) ‖ ∞ coincides with the instance-dependent functional proposed by Khamaru et al. [2021b] that controls the difficulty of estimating Q ∗ in the glyph[lscript] ∞ -norm.

Asymptotic normality for V ∗ estimation. If the optimal policy is unique, we can obtain a similar result for the optimal value function V ∗ , making use of the asymptotic normality of ¯ Q T . We define an estimator ¯ V T ∈ R S greedily from ¯ Q T ∈ R D : the s -th entry of ¯ V T is ¯ V T ( s ) ∈ arg max a ∈A ¯ Q T ( s, a ) . As a corollary of Theorem B.1, ¯ V T enjoys a similar asymptotic normality with the asymptotic variance defined by Var V . One can check that

<!-- formula-not-decoded -->

where Π π ∗ ∈ { 0 , 1 } S × D is the projection matrix associated with the deterministic optimal policy π ∗ (see (2)). Hence, Var V is formed by selecting entries from Var Q . In particular, Var V ( s, s ′ ) = Var Q (( s, π ∗ ( s )) , ( s ′ , π ∗ ( s ′ ))) for any s, s ′ ∈ S . The proof is deferred to Appendix B.2.

glyph[negationslash]

Lemma B.1. If π ∗ is unique, then we have a positive optimality gap gap := min s min a = π ∗ ( s ) | V ∗ ( s ) -Q ∗ ( s, a ) | &gt; 0 where π ∗ ( s ) is the unique action satisfying V ∗ ( s ) = Q ∗ ( s, a ∗ ( s )) . For any Q -function estimator Q ∈ R D , it follows that { π Q = π ∗ } ⊆ {‖ Q -Q ∗ ‖ ∞ ≥ gap 2 } and

<!-- formula-not-decoded -->

where π Q is the greedy policy with respective to Q defined by π Q ( s ) := arg max a ∈A Q ( s, a ) . If arg max a ∈A Q ( s, a ) has more than one element, we break the tie by randomness.

Corollary B.1 (Asymptotic normality for V ∗ ) . Let ¯ V T ∈ R S be the greedy value function computed from ¯ Q T ∈ R D , i.e., ¯ V T ( s ) ∈ arg max a ∈A ¯ Q T ( s, a ) . Under Assumptions 3.1 and 3.3, if we assume the optimal policy π ∗ is unique, then

<!-- formula-not-decoded -->

glyph[negationslash]

where the asymptotic variance is

<!-- formula-not-decoded -->

and Var( Π π ∗ Z ) is the covariance matrix of the projected Bellman noise Π π ∗ Z .

Insights on sample efficiency. The asymptotic results shed light on the sample efficiency of averaged Q-learning. Under ideal conditions, we have

<!-- formula-not-decoded -->

In this case, roughly speaking, to obtain an ε -accurate estimator of the optimal Q-value function Q ∗ (i.e., E ‖ ¯ Q T -Q ∗ ‖ ∞ ≤ ε ), we require approximately T = O ( ln D ε 2 ‖ diag(Var Q ) ‖ ∞ ) iterations or equivalently DT = O ( D ln D ε 2 ‖ diag(Var Q ) ‖ ∞ ) samples. This explains why Khamaru et al. [2021b] regarded ‖ diag(Var Q ) ‖ ∞ as the difficulty indicator because it affects the sample complexity directly.

## B.1 Proof of Theorem B.1

Proof of Theorem B.1. One can prove Theorem B.1 by applying continuous mapping theorem to Theorem 3.1 with the functional f : D ([0 , 1] , R D ) → R D , f ( w ) = w (1) . Once we can prove f is a continuous functional in ( D ([0 , 1] , R D ) , d 0 ) , an application of (27) would conclude the proof. Recalling the metric (25) defined on D ([0 , 1] , R D ) , we have for any w 1 , w 2 ∈ D ([0 , 1] , R D ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We even show that f is 1 -Lipschitz continuous in ( D ([0 , 1] , R D ) , d 0 ) and thus complete the proof.

## B.2 Proof of Corollary B.1

Proof of Corollary B.1. We first prove

<!-- formula-not-decoded -->

Recall the definition

<!-- formula-not-decoded -->

For one thing, we have Var( Π π ∗ Z ) = Π π ∗ Var( Z )( Π π ∗ ) glyph[latticetop] . For another thing, we have Π π ∗ ( I -γ P π ∗ ) -1 = ( I -γ P π ∗ ) -1 Π π ∗ . This is because

<!-- formula-not-decoded -->

Putting these together, (23) follows from direct verification.

We then prove the asymptotic normality of ¯ V T . Let ¯ π t is the greedy policy with respect to ¯ Q t , i.e., ¯ π t ( s ) ∈ argmax a ∈A ¯ Q t ( s, a ) . From the definition of our estimator,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, it is easy to see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we can prove then the conclusion follows from Slutsky's theorem. We have that

<!-- formula-not-decoded -->

which implies glyph[negationslash]

glyph[negationslash]

where ( a ) uses ‖ ¯ Q T ‖ ∞ ≤ (1 -γ ) -1 , ( b ) uses the fact that both ¯ π T and π ∗ are deterministic policies and thus ‖ Π ¯ π T -Π π ∗ ‖ ∞ = 2 · 1 { ¯ π T = π ∗ } , ( c ) uses the fact { ¯ π t = π ∗ } ⊆ {‖ ¯ Q t -Q ∗ ‖ ∞ ≥ gap 2 } which we derived in Lemma B.1, and finally ( d ) follows from Jensen's inequality.

glyph[negationslash]

From Theorem E.1, we know 1 √ T ∑ T t =1 E ‖ Q t -Q ∗ ‖ 2 ∞ → 0 as T → ∞ . Therefore, we have that √ T E ‖ Π ¯ π T ¯ Q T -Π π ∗ ¯ Q T ‖ ∞ = o (1) which implies (24) is true.

## B.3 Proof of Lemma B.1

Proof of Lemma B.1. Recall that gap = min s min a = π ∗ ( s ) | Q ∗ ( s, π ∗ ( s )) -Q ∗ ( s, a ) | . If gap = 0 , by definition, there must exist some s 0 ∈ S and a 0 ∈ A such that V ∗ ( s 0 ) = Q ∗ ( s 0 , a 0 ) and a 0 = π ∗ ( s 0 ) , which is contradictory with the uniqueness of π ∗ . Hence, a unique π ∗ implies a positive gap .

glyph[negationslash]

glyph[negationslash]

For any Q satisfying ‖ Q -Q ∗ ‖ ∞ &lt; gap 2 , we must have ‖ Q ( s, · ) -Q ∗ ( s, · ) ‖ ∞ &lt; gap 2 for any s ∈ S . In this case, it must be true that π Q ( s ) = π ∗ ( s ) for all s ∈ S . Otherwise, there exists some s ∈ S such that π Q ( s ) = π ∗ ( s ) . We then have

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

where ( a ) follows from the definition of the optimality gap. The result Q ( s, π Q ( s )) &lt; Q ( s, π ∗ ( s )) contradicts with the fact that π Q ( s ) is the greedy policy with respect to Q at state s , which implies Q ( s, π ∗ ( s )) ≤ Q ( s, π Q ( s )) . This implies that the event { π Q = π ∗ } ⊆ {‖ Q -Q ∗ ‖ ∞ ≥ gap 2 } and thus 1 { π Q = π ∗ } ≤ 1 {‖ Q -Q ∗ ‖ ∞ ≥ gap 2 } . Hence, glyph[negationslash]

<!-- formula-not-decoded -->

where the last line uses 1 {‖ Q -Q ∗ ‖ ∞ ≥ gap 2 } ≤ 2 gap ‖ Q -Q ∗ ‖ ∞ .

## B.4 Proof of Proposition 3.1

Proof of Proposition 3.1. Let g : D ([0 , 1] , R D ) → R be a functional defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Once we prove g is continuous in (dom( g ) , d 0 ) , the continuous mapping theorem together with Theorem 3.1 would complete the proof for Proposition 3.1.

In Appendix B.1, we have shown f : D ([0 , 1] , R D ) → R D , f ( w ) = w (1) is 1 -Lipschitz continuous in ( D ([0 , 1] , R D ) , d 0 ) . Let h : D ([0 , 1] , R D ) → R D × D be defined by h ( w ) = ∫ 1 0 w ( r ) w ( r ) glyph[latticetop] dr . Hence, once we prove h is continuous in ( D ([0 , 1] , R D ) , d 0 ) , it follows that g = f glyph[latticetop] h -1 f is also continuous in (dom( g ) , d 0 ) . To that end, we only show each entry of h is continuous in w . This is true because of each entry of h is in form of integration which is a continuous functional on the Skorohod space D ([0 , 1] , R ) .

Finally, by Theorem 3.1 and definition of weak convergence, we know that as T goes to infinity,

<!-- formula-not-decoded -->

Hence, with probability approaching to one, ∫ 1 0 φ T ( r ) φ T ( r ) glyph[latticetop] dr is invertible and thus g ( φ T ) is well defined.

Here the domain of g is glyph[negationslash]

## C PROOF OF THEOREM 3.1

## C.1 Preliminaries and High-level Idea

In this section, we provide a self-contained proof of our functional central limit theorem (FCLT). Let ∆ t = Q t -Q ∗ be the error vector at iteration t . The application of Polyak-Ruppert average [Polyak and Juditsky, 1992] gives an estimator for Q ∗ : ¯ Q T = 1 T ∑ T t =1 Q t . Then its partial sum of the first r -fraction ( r ∈ [0 , 1]) is 1 T ∑ glyph[floorleft] Tr glyph[floorright] t =1 Q t . The associated standardized partial-sum process is defined by

<!-- formula-not-decoded -->

Here φ T ( · ) should be viewed as a D -dimensional random function. For simplicity, we also use φ T = { φ T ( r ) } r ∈ [0 , 1] to denote the whole function.

## C.1.1 Weak convergence of measures in Polish spaces

We will introduce some basic knowledge of weak convergence in metric spaces. See Chapter VI in [Jacod and Shiryaev, 2003] for a detailed introduction.

A Polish space is a topological space that is separable, complete, and metrizable. Let D ([0 , 1] , R d ) = { càdlàg function ω ( r ) ∈ R d , r ∈ [0 , 1] } collect all d -dimensional functions which are right continuous with left limits. Define D ([0 , 1] , R d ) as the σ -field generated by all maps X ↦→ X ( r ) for r ∈ [0 , 1] . The J 1 Skorokhod topology equips D ([0 , 1] , R d ) with a metric d 0 such that ( D ([0 , 1] , R d ) , d 0 ) is a Polish space and D ([0 , 1] , R d ) is its Borel σ -field (the σ -field generated by all open subsets). In particular, for any w 1 , w 2 ∈ D ([0 , 1] , R d ) ,

<!-- formula-not-decoded -->

where Λ denotes the class of strictly increasing continuous mappings λ : [0 , 1] → [0 , 1] with λ (0) = 0 and λ (1) = 1 .

An important subset of D ([0 , 1] , R d ) is C ([0 , 1] , R d ) = { continuous ω ( r ) ∈ R d , r ∈ [0 , 1] } , which collects all d -dimensional continuous functions defined on [0 , 1] . The uniform topology equips C ([0 , 1] , R d ) with the uniform norm

<!-- formula-not-decoded -->

The resulting ( C ([0 , 1] , R d ) , ‖ · ‖ sup ) is a Polish space. Additionally, we have d 0 ( w 1 , w 2 ) ≤ ‖ w 1 -w 2 ‖ sup for any w 1 , w 2 ∈ D ([0 , 1] , R d ) . The J 1 Skorokhod topology is weaker than the uniform topology. However, if X ∈ D ([0 , 1] , R d ) is a continuous function, a sequence { X t } t ≥ 0 ⊆ D ([0 , 1] , R d ) converges to X for the Skorokhod topology if and only if it converges to X under the uniform norm ‖ · ‖ sup . Hence, the Skorokhod topology relativized to C ([0 , 1] , R d ) coincides with the uniform topology there.

Any random element X t ∈ D ([0 , 1] , R d ) introduces a probability measure on D ([0 , 1] , R d ) denoted by L ( X t ) such that ( D ([0 , 1] , R d ) , D ([0 , 1] , R d ) , L ( X t )) becomes a probability space. We say a sequence of random elements { X t } t ≥ 0 ⊆ D ([0 , 1] , R d ) weakly converges to X , if for any bounded continuous function f : D ([0 , 1] , R d ) → R , we have

<!-- formula-not-decoded -->

The condition is equivalent to that any finite-dimensional projections of φ T converge in distribution. We denote the weak convergence by X T w → X .

Theorem C.1 (Slutsky's theorem on Polish spaces) . Suppose S is a Polish space with metric d and { ( X t , Y t ) } t ≥ 0 are random elements of S × S . Suppose X T w → X and d ( X T , Y T ) w → 0 , then Y T w → X .

By Slutsky's theorem in Theorem C.1, if ‖ Y T ‖ sup w → 0 and X T w → X , then X T + Y T w → X . A sufficient condition to ‖ Y T ‖ sup w → 0 is E ‖ Y T ‖ sup → 0 by Markov's inequality.

Proposition C.1. For two random sequences { X t } t ≥ 0 , { Y t } t ≥ 0 ⊆ D ([0 , 1] , R d ) satisfying E ‖ Y T ‖ sup → 0 and X T w → X , we have X T + Y T w → X .

## C.1.2 Proof Idea

In the following, we will show under the three assumptions in the main text, we can establish

<!-- formula-not-decoded -->

where B D ∈ C ([0 , 1] , R D ) is the standard D -dimensional Brownian motion on [0 , 1] . That is the associated measure of φ T weakly converges to the measure introduced by Var 1 / 2 Q B D on D ([0 , 1] , R D ) .

To proceed the proof, we will use two auxiliary sequences { ∆ 1 t } t ≥ 0 and { ∆ 2 t } t ≥ 0 defined in Lemma C.1. The proof of Lemma C.1 can be found in Appendix C.4.1.

Lemma C.1. Denote G = I -γ P π ∗ , A t = I -η t G and W t = ( r t -r ) + γ ( P t -P ) V t -1 for short. The auxiliary sequences { ∆ 1 t } t ≥ 0 and { ∆ 2 t } t ≥ 0 are defined iteratively: ∆ 1 0 = ∆ 2 0 = ∆ 0 and for t ≥ 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As long as sup t η t ≤ 1 , it follows that all t ≥ 0 ,

By the sandwich inequality (31), we have

<!-- formula-not-decoded -->

as T goes to infinity. Proposition C.1 implies φ T weakly converges to a rescaled Brownian motion Var 1 / 2 Q B D , by which we complete the proof.

## C.2 Functional CLT for φ 1 T

We first establish the FLCT of φ 1 T ( r ) = 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] t =1 ∆ 1 t , i.e., lim T →∞ E ‖ φ 1 T -Z‖ sup = 0 for some L ( Z ) = L (Var 1 / 2 Q B D ) The FCLT of φ 2 T ( r ) = 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] t =1 ∆ 2 t can be validated in an almost identical way. We start by rewriting (28) as

.

<!-- formula-not-decoded -->

where A t = I -η t ( I -γ P π ∗ ) , Z t = ( r t -r ) + γ ( P t -P ) V ∗ , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The two sequences form a sandwich bound for ∆ t , producing ∆ 2 t ≤ ∆ t ≤ ∆ 1 t coordinate-wise. We similarly define the error vectors of their first r -fraction partial sums as

<!-- formula-not-decoded -->

Then, it is valid that φ 1 T , φ 2 T ∈ D ([0 , 1] , R D ) and for any r ∈ [0 , 1] ,

<!-- formula-not-decoded -->

In the following subsections, we will show that under Assumption 3.1, 3.2 and 3.3, we can find a random function Z ∈ D ([0 , 1] , R D ) which satisfies

<!-- formula-not-decoded -->

Furthermore, φ 1 T and φ 2 T weakly converge to Z such that

<!-- formula-not-decoded -->

We comment that { Z t } t ≥ 0 collects the i.i.d. noise inherent in the empirical Bellman operator and { D 1 t -1 } t ≥ 1 captures the closeness between the current Q -function estimator Q t -1 and the optimal Q ∗ . Recurring (34) gives

<!-- formula-not-decoded -->

Here we use the convention that ∏ t i = t +1 A i = I for any t ≥ 0 . For any r ∈ [0 , 1] , summing the last equality over t = 1 , · · · , glyph[floorleft] Tr glyph[floorright] and scaling it properly, we have

<!-- formula-not-decoded -->

where the last line uses the following notation:

<!-- formula-not-decoded -->

Define G = I -γ P π ∗ with γ ∈ [0 , 1) , then A i = I -η i G . Typically speaking, A T j approximates G uniformly well (see Lemma C.4). By the observation, we further expand (36) and decompose φ 1 T ( r ) into six terms { ψ i } 5 i =0 which will be analyzed respectively in the following:

<!-- formula-not-decoded -->

Readers should keep in mind that all ψ i 's depend on T , a dependence which we omit for simplicity. In the following, we will show (32) is true by setting Z = ψ 1 . In order to establish (33), we will show that E ‖ ψ i ‖ sup = o (1) for i = 0 , 2 , 3 , 4 , 5 . In this way, based on (38), we have

<!-- formula-not-decoded -->

and validate (33). To that end, we first study the properties of { A T j } 0 ≤ j ≤ T since it appears in many ψ i 's.

## C.2.1 Properties of { A T j } 0 ≤ j ≤ T

First, prior work [Polyak and Juditsky, 1992] considers a general step size { η t } t ≥ 0 satisfying Assumption 3.3 and establishes the following lemma.

Lemma C.2 (Lemma 1 in [Polyak and Juditsky, 1992]) . For { η t } t ≥ 0 satisfying Assumption 3.3,

- Uniform boundedness: ‖ A T j ‖ ∞ ≤ C 0 uniformly for all T ≥ j ≥ 0 for some constant C 0 ≥ 1 ;
- Uniform approximation: lim T →∞ 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 = 0 .

Lemma C.2 shows that when the step size η t decreases at a slow rate, A T j is uniformly bouned (that is sup T ≥ j ≥ 1 ‖ A T j ‖ ∞ &lt; ∞ ) and is a good surrogate of G -1 := ( I -γ P π ∗ ) -1 in the asymptotic sense: lim T →∞ 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 = 0 . 7 It is sufficient to derive our asymptotic result. However, on purpose of non-asymptotic analysis, we should provide a non-asymptotic counterpart capturing the specific decaying rate in the glyph[lscript] ∞ -norm. Therefore, we consider two specific step sizes, namely (S1) the linear rescaled step size and (S2) polynomial step size. Define ˜ η t = (1 -γ ) η t as the rescaled step size for simplicity, we have

- (S1) linear rescaled step size that uses η t = 1 1+(1 -γ ) t (equivalently ˜ η t = 1 -γ 1+(1 -γ ) t );
- (S2) polynomial step size that uses η t = t -α with α ∈ (0 , 1) for t ≥ 1 and η 0 = 1 .

The first is uniform boundedness whose proof is provided in Appendix C.4.2.

Lemma C.3 (Uniform boundedness) . There exists some c &gt; 0 such that

<!-- formula-not-decoded -->

The second is the uniform approximation. The proof is deferred in Appendix C.4.3. We observe that as T grows, 1 T ∑ T j =1 ‖ A T j -G -1 I ‖ 2 ∞ vanishes under (S2), but is only guaranteed to be bounded for (S1). This is not contradictory with Lemma C.2 since (S1) doesn't satisfy Assumption 3.3.

Lemma C.4 (Uniform approximation) . There exists some constant c &gt; 0 such that

<!-- formula-not-decoded -->

## C.2.2 Establishing the Functional CLT

Uniform negligibility of ψ 0 . It is clear that ψ 0 is a deterministic function. Using the uniform boundedness of A T j ( T ≥ j ≥ 0) in Lemma C.2, we have

<!-- formula-not-decoded -->

where we use η 0 ≤ 1 ≤ C 0 and ‖ ∆ 0 ‖ ∞ ≤ 1 1 -γ .

7 The original Lemma 1 in [Polyak and Juditsky, 1992] uses the glyph[lscript] 2 -norm and spectral norm. Due to the equivalence between these norms, we formulate our Lemma C.2.

Partial-sum asymptotic behavior of ψ 1 . Recall that Z j = ( r j -r )+ γ ( P j -P ) V ∗ is the noise inherent in the empirical Bellman operator at iteration j . Since at each iteration the simulator generates rewards r j and produces the empirical transition P j in an i.i.d. fashion, T 1 ( r ) is the scaled partial sum of glyph[floorleft] Tr glyph[floorright] independent copies of the random vector Z j which has zero mean and finite variance denoted by Var( Z j ) = Var( r j + γ P j V ∗ ) = E Z j Z glyph[latticetop] j . Additionally, it is clear that ‖ Z j ‖ ∞ ≤ (1 -γ ) -1 is uniformly bounded and thus its moments of any order is uniformly bounded. By Theorem 4.2 in [Hall and Heyde, 2014] (or Theorem 2.2 in [Jirak, 2017]), we establish the following FCLT for the partial sums of independent random vectors.

Lemma C.5. For any r ∈ [0 , 1] ,

<!-- formula-not-decoded -->

where B D is the D -dimensional standard Brownian motion and the variance matrix Var Q is

<!-- formula-not-decoded -->

Uniform negligibility of ψ 2 . Recall that ψ 2 ( r ) = 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] j =1 ( A T j -G -1 ) Z j . If we define X t = 1 √ T ∑ t j =1 ( A T j -G -1 ) Z j , then ψ 2 ( r ) = X glyph[floorleft] Tr glyph[floorright] . Let F t = σ ( { r j , P j } 0 ≤ j ≤ t ) be the σ -field generated by all randomness before and including iteration t . Then { X t , F t } is a martingale since E [ X t |F t -1 ] = X t -1 . As a result {‖ X t ‖ 2 , F t } is a submartingale since by conditional Jensen's inequality, we have E [ ‖ X t ‖ 2 |F t -1 ] ≥ ‖ E [ X t |F t -1 ] ‖ 2 = ‖ X t -1 ‖ 2 . By Doob's maximum inequality for submartingales (which we use to derive the following ( ∗ ) inequality),

<!-- formula-not-decoded -->

Here, we change to the glyph[lscript] 2 -norm since it will facilitate the analysis. The last inequality follows by using a finite c 1 satisfying E ‖ Z j ‖ 2 2 sup T ≥ j ≥ 1 ‖ A T j -G -1 ‖ 2 ≤ c 1 . Indeed, we can set c 1 = ( 1 1 -γ +sup T ≥ j ‖ A T j ‖ 2 )tr(Var Q ) thanks to Lemma C.2. In addition, Lemma C.2 implies 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 → 0 as T goes to infinity. As a result, E ‖ ψ 2 ‖ sup = E sup r ∈ [0 , 1] ‖ ψ 2 ( r ) ‖ ∞ ≤ E sup r ∈ [0 , 1] ‖ ψ 2 ( r ) ‖ 2 ≤ √ E sup r ∈ [0 , 1] ‖ ψ 2 ( r ) ‖ 2 2 = o (1) .

Uniform negligibility of ψ 3 . Recall that ψ 3 ( r ) = γ √ T ∑ glyph[floorleft] Tr glyph[floorright] j =1 A T j ( P j -P )( V j -1 -V ∗ ) . By a similar argument in the analysis of ψ 2 , we have E sup r ∈ [0 , 1] ‖ ψ 3 ( r ) ‖ 2 2 ≤ 4 E ‖ ψ 3 (1) ‖ 2 2 by Doob's maximum inequality. Therefore,

<!-- formula-not-decoded -->

where ( a ) follows since all cross terms have zero mean due to E [( P j -P )( V j -1 -V ∗ ) |F j -1 ] = 0 , and ( b ) follows by setting c 2 = 16 D (sup T ≥ j ‖ A T j ‖ 2 ) 2 because of the uniform boundedness of ‖ A T j ‖ ∞ from Lemma C.2 and ‖ P j -P ‖ 2 2 ≤ D ‖ P j -P ‖ 2 ∞ = 4 D . By Theorem E.4, we know 1 T ∑ T j =1 E ‖ V j -1 -V ∗ ‖ 2 2 → 0 under the general step size when T →∞ . As a result, E ‖ ψ 3 ( r ) ‖ sup = E sup r ∈ [0 , 1] ‖ ψ 3 ( r ) ‖ ∞ ≤ E sup r ∈ [0 , 1] ‖ ψ 3 ( r ) ‖ 2 ≤ √ E sup r ∈ [0 , 1] ‖ ψ 3 ( r ) ‖ 2 2 = o (1) .

Uniform negligibility of ψ 4 . Recall that ψ 4 ( r ) = 1 √ T ∑ glyph[floorleft] Tr glyph[floorright] j =1 ( A glyph[floorleft] Tr glyph[floorright] j -A T j ) ε j where ε j = Z j + γ ( P j -P )( V j -1 -V ∗ ) . It is clear that we have sup j ≥ 0 E ‖ Q t -Q ∗ ‖ 4 &lt; ∞ as a result of sup j ≥ 0 EE ‖ Q j \_ ∞ 4 &lt; ∞ in Lemma C.6. Notice that the coefficient A glyph[floorleft] Tr glyph[floorright] j -A T j changes as r varies. The analysis of ψ 4 should be more careful and subtle.

Lemma C.6 (Moment bounds) . Under Assumption 3.1, it follows that

<!-- formula-not-decoded -->

Proof of Lemma C.6. By Lemma E.2, ‖ ∆ t ‖ ∞ ≤ a t + b t + ‖ N t ‖ ∞ . It implies that E ‖ ∆ t ‖ 4 ∞ ≤ 3 3 E ( a 4 t + b 4 t + ‖ N t ‖ 4 ∞ ) . Notice that

<!-- formula-not-decoded -->

First, it is easy to find that sup t ≥ 0 a t &lt; ∞ since it is deterministic and decays exponentially fast. Second, we have sup t ≥ 0 ‖ N t ‖ 4 ∞ &lt; ∞ . This is because we have E ‖ N t ‖ 4 ∞ ≤ (1 -η t ) E ‖ N t -1 ‖ 4 ∞ + η t E ‖ Z t ‖ 4 ∞ from Jensen's inequality. It is easy to show sup t ≥ 0 ‖ N t ‖ 4 ∞ &lt; sup t ≥ 0 E ‖ Z t ‖ 4 ∞ &lt; ∞ by this inequality and induction. Finally, iterating the expression of b t , we have b T = γ ∑ T t =1 ∏ T j = t +1 (1 -(1 -γ ) η j ) η t ‖ N t -1 ‖ ∞ = γ 1 -γ ∑ T t =1 ˜ η ( t,T ) ‖ N t -1 ‖ ∞ with ˜ η ( t,T ) a probability defined on [ T ] in (59). The last equation implies b T is a probability weighted sum of N t ( t ∈ [ T ]) . Hence, by Jensen's inequality, we know sup t ≥ 0 E b 4 t &lt; sup t ≥ 0 E ‖ Z t ‖ 4 ∞ &lt; ∞ .

Recall F t = σ ( { r j , P j } 0 ≤ j ≤ t ) is the σ -field generated by all randomness before and including iteration t . { ε t , F t } is a martingale difference since E [ ε t |F t -1 ] = 0 . Furthermore, ε t has finite moments of any order since it is almost surely bounded ‖ ε t ‖ ∞ = O ((1 -γ ) -1 ) . On the other hand, by definition (37), it follows that for any 0 ≤ k ≤ T ,

<!-- formula-not-decoded -->

On one hand, ‖ A T k +1 A k +1 ‖ 2 ≤ c 3 is uniformly bounded with c 3 = (sup T ≥ j ‖ A T j ‖ 2 )(1 + ‖ G ‖ 2 ) for any T ≥ k +1 from Lemma C.2. On the other hand, we define an auxiliary sequence { Y k } k ≥ 1 as following: Y 1 = 0 and Y k +1 = A k Y k + η k ε k for any k ≥ 1 . One can check that Y k +1 = ∑ k j =1 ( ∏ k i = j +1 A i ) η j ε j where we use the convention ∏ k i = k +1 A i = I for any k ≥ 0 . These results imply we can apply Lemma D.1. Putting these pieces together, we have that

<!-- formula-not-decoded -->

where ( ∗ ) follows from Lemma D.1.

Uniform negligibility of ψ 5 . In the following, we will prove ‖ ψ 5 ‖ sup = o P (1) by showing E ‖ ψ 5 ‖ sup = o (1) . It is worth mentioning that ψ 5 arises purely due to the non-stationary nature of Q-learning. If we consider a stationary update process, e.g., policy evaluation [Mou et al., 2020a,b, Khamaru et al., 2021b], π t would remain the same all the time and ψ 5 would disappear in the case. Notice that ψ 5 ( r ) = γ √ T ∑ glyph[floorleft] Tr glyph[floorright] j =1 A glyph[floorleft] Tr glyph[floorright] j ( P π j -1 -P π ∗ ) ∆ j -1 is a sum of correlated random variables (which are even not mean-zero). We need a high-order residual condition Assumption 3.2 to bound E ‖ ψ 5 ‖ sup . With such a

Lipschitz condition, Lemma C.7 shows E ‖ ψ 5 ‖ sup is dominated by 1 √ T ∑ T j =1 E ‖ ∆ j -1 ‖ 2 ∞ , which is o (1) for the general step size as suggested by Theorem E.1. The proof of Lemma C.7 is in Appendix C.4.4.

Lemma C.7. It follows that

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

Putting the pieces together. From (36), φ 1 T = ∑ 5 i =0 ψ i . We have shown ψ 1 w → Var 1 / 2 Q B D in the sense of ( D ([0 , 1] , R D ) , d 0 ) and ‖ ψ i ‖ sup = o P (1) for i = 1 . Using ‖ φ 1 T -ψ 1 ‖ sup ≤ ∑ i =1 ‖ ψ i ‖ sup , we know that ‖ φ 1 T -ψ 1 ‖ sup = o P (1) . Proposition C.1 implies φ 1 T w → Var 1 / 2 Q B D . We then establish the FCLT for φ 1 T ( r ) .

## C.3 Functional CLT for φ 2 T

We can repeat the above analysis for φ 2 T . We rewrite (29) as

<!-- formula-not-decoded -->

where A t = I -η t ( I -γ P π ∗ ) and Z t = ( r t -r ) + γ ( P t -P ) V ∗ are the same as those defined in (34) except that D 1 t -1 (defined in (35)) is replaced by

<!-- formula-not-decoded -->

Since D 2 t -1 is much simpler than D 1 t -1 , the analysis for φ 2 T ( r ) should be easier than φ 1 T ( r ) . Using the notation A T j (see(37)), we decompose φ 2 T ( r ) into five terms:

<!-- formula-not-decoded -->

glyph[negationslash]

Here { ψ i } 4 i =0 are exactly the same as those in (38). Our previous analysis provides us a low-hanging fruit result: ψ 1 w → Var 1 / 2 Q B D in the sense of ( D ([0 , 1] , R D ) , d 0 ) and ‖ ψ i ‖ sup = o P (1) for i = 1 . Then we know that ‖ φ 2 T -T 1 ‖ sup = o (1) and φ 2 T w → Var 1 / 2 Q B D due to Proposition C.1. We thus establish the FCLT for φ 2 T .

## C.4 Proofs of Lemmas

## C.4.1 Proof of Lemma C.1

Proof of Lemma C.1. We use mathematical induction to prove the statement. When t = 0 , the inequality (30) holds by initialization. Assume (30) holds at t -1 , i.e., ∆ 2 t -1 ≤ ∆ t -1 ≤ ∆ 1 t -1 . Let us analyze the case of t . By the Q-learning update rule, it follows that

<!-- formula-not-decoded -->

where ( a ) uses W t = ( r t -r ) + γ ( P t -P ) V t -1 ; ( b ) uses PV t -1 = P π t -1 Q t -1 and PV ∗ = P π ∗ Q ∗ , and ( c ) follows by arrangement and the shorthand A t = I -η t ( I -γ P π ∗ ) . Since all the entries of A t = I -η t ( I -γ P π ∗ ) are non-negative (which results from the assumption sup t η t ≤ 1 ), then A t ∆ 2 t -1 ≤ A t ∆ t -1 ≤ A t ∆ 1 t -1 .

For one hand, based on (42), we have

<!-- formula-not-decoded -->

where the last inequality uses P π t -1 Q t -1 ≥ P π ∗ Q t -1 which results from the fact π t -1 is the greedy policy with respect to Q t -1 . For the other hand, it follows that

<!-- formula-not-decoded -->

where the last inequality uses P π t -1 Q ∗ ≤ P π ∗ Q ∗ which results from the fact π ∗ is the greedy policy with respect to Q ∗ . Hence, we have proved ∆ 2 t ≤ ∆ t ≤ ∆ 1 t holds at iteration t .

## C.4.2 Proof of Lemma C.3

Proof of Lemma C.3. By the definition of (37), we have ‖ A T j ‖ ∞ ≤ η j ∑ T t = j ∏ t i = j +1 (1 -˜ η i ) . Plugging the specific form of { η t } , we have for (S1)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and for (S2)

where ( a ) uses ∑ t i = j i -α ≥ 1 1 -α (( t +1) 1 -α -j 1 -α ) and exp((1 -γ ) j -α ) ≤ e , ( b ) uses the change of variable y = 1 -γ 1 -α ( t 1 -α -j 1 -α ) , ( c ) uses ( a + b ) p ≤ max { 2 p -1 , 1 } ( a p + b p ) for any p &gt; 0 , and ( d ) uses (1 -α ) α 1 -α Γ ( 1 1 -α ) ≤ √ 2 πe 1 / 2 √ 1 -α from (65) and max { 2 α 1 -α , 1 } ≤ 2 1 1 -α .

## C.4.3 Proof of Lemma C.4

Proof of Lemma C.4. For (S1), we have

<!-- formula-not-decoded -->

where ( a ) uses ln 2 (1 + x ) /x ≤ 7 8 for all x ≥ 0 and ∫ 1 0 ln 2 xdx = Γ(3) = 2Γ(1) = 2 . For (S2), based on (37) and G = η -1 j ( I -( I -η j G )) , we have

<!-- formula-not-decoded -->

On the one hand,

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

where ( a ) uses the fact that for η t = t -α , we have

<!-- formula-not-decoded -->

where we use ln(1 + x ) ≥ x/ (1 + x ) in the first inequality and ln(1 + x ) ≤ x in the second inequality. ( b ) uses the notation ˜ m j,t := ∑ t i = j ˜ η i and exp( ˜ η j ) ≤ exp(1) = e . ( c ) uses the following lemma.

Lemma C.8. Let ˜ m j,t := ∑ t i = j ˜ η i and recall ˜ η i = (1 -γ ) i -α . Then T ≥ j ≥ 1 , for some constant c &gt; 1 ,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Proof of Lemma C.8. Clearly we have

<!-- formula-not-decoded -->

Then ˜ m j,t ≤ 1 -γ 1 -α ( t 1 -α -( j -1) 1 -α ) ≤ ˜ m j -1 ,t -1 . Hence,

<!-- formula-not-decoded -->

where ( a ) uses the change of variable y = 1 -γ 1 -α ( t 1 -α -( j -1) 1 -α ) , ( b ) uses ( a + b ) p ≤ max { 2 p -1 , 1 } ( a p + b p ) for any p &gt; 0 , ( c ) uses max { 2 α 1 -α , 2 } ≤ 2 1 1 -α , ( d ) uses Γ ( 1 + 1 1 -α ) = 1 1 -α Γ ( 1 1 -α ) and (1 -α ) α 1 -α Γ ( 1 1 -α ) ≤ √ 2 πe √ 1 -α from (65).

## C.4.4 Proof of Lemma C.7

Proof of Lemma C.7. By Lemma B.1 and Lemma C.3, it follows that

<!-- formula-not-decoded -->

Here we use sup r ∈ [0 , 1] ∥ ∥ ∥ A glyph[floorleft] Tr glyph[floorright] j ∥ ∥ ∥ ∞ ≤ C 0 due to Lemma C.2.

## D UNIFORM NEGLIGIBILITY OF NOISE RECURSION

Definition D.1 (Hurwitz matrix) . We say -G ∈ R d × d is a Hurwitz (or stable) matrix if Re λ i ( G ) &gt; 0 for i ∈ [ d ] . Here λ i ( · ) denotes the i -th eigenvalue.

Lemma D.1 (A generalization of Lemma B.7 in [Li et al., 2022]) . Let { ε t } t ≥ 0 be a martingale difference sequence adapting to the filtration F t . Define an auxiliary sequence { y t } t ≥ 0 as following: y 0 = 0 and for t ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is easy to verify that

Let { η t } t ≥ 0 satisfy Assumption 3.3. If -G ∈ R d × d is Hurwitz, and sup t ≥ 0 E ‖ ε t ‖ 4 &lt; ∞ , then we have that

<!-- formula-not-decoded -->

Proof of Lemma D.1. In the sequel, we denote ˇ y t = y t √ η t -1 . We will also use a glyph[precedesorequal] b to denote a ≤ Cb for unimportant positive constants C with the specific value of C changing according to the context. Then the update rule (45) can be rewritten as

<!-- formula-not-decoded -->

Step 1: Divide the time interval. For a specific λ &gt; 0 , we divide the the time interval [0 , T ] into several disjoint portions with the t k the k -th endpoint such that ∑ t k +1 -1 t k η s ≥ λ . In particular, { t k } k ≥ 0 is defined iteratively by t 0 = 0 and

<!-- formula-not-decoded -->

Clearly, K is the number of portions and we have 0 = t 0 &lt; t 1 &lt; · · · &lt; t K = T . Since ∑ ∞ t =1 η t = ∞ , we know that K →∞ as T →∞ What's more, K is upper bounded by 1 λ T ∑ η t due to t =0

<!-- formula-not-decoded -->

The fact sup t ≤ T ‖ y t ‖ η t -1 ≤ sup t ≤ T ‖ ˇ y t ‖ √ η T implies we have for any glyph[epsilon1] &gt; 0 ,

<!-- formula-not-decoded -->

LemmaD.2. Let { y t } t ≥ 0 be defined in the way of (45) . If -G ∈ R d × d is Hurwitz and sup t ≥ 0 E ‖ ε t ‖ p &lt; ∞ for p ≥ 4 , then the sequence { y t } t ≥ 0 is ( L 4 , √ η t ) -consistency, that is, there exists a universal constant C 4 &gt; 0 such that E ‖ y t ‖ 4 ≤ C 4 η 2 t for all t ≥ 0 .

The proof of Lemma D.2 is deferred in Section D.1. Lemma D.2 implies that sup t ≥ 0 E ‖ ˇ y t ‖ 4 glyph[precedesorequal] 1 . Let B := { sup 1 ≤ k ≤ K ‖ ˇ y t k ‖ ≤ glyph[epsilon1] √ Tη T } be the event where all ‖ ˇ y t k ‖ 's are smaller than glyph[epsilon1] √ Tη T for 1 ≤ k ≤ K . By the union bound and Markov inequality,

<!-- formula-not-decoded -->

Here the last inequality uses (48) and the condition on { η t } t ≥ 0 that ∑ T t =0 η t ( Tη T ) 2 → 0 due to ∑ T t =0 η t Tη T ≤ C and η T T →∞ . The above result implies for given λ, glyph[epsilon1] &gt; 0 , the event B holds with probability approaching one. Hence, we focus our analysis on the event B . Conditioning on the event B , we split our target event into several disjoint events whose probability will be

analyzed latter.

<!-- formula-not-decoded -->

Step 2: Bound each P k . Leveraging (47) recursively implies for given r &lt; t ,

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

In the following, we highlight the dependence on T and λ and use glyph[precedesorequal] to omit universal constants.

We consider to bound P (1) k first. Since η t η t -1 = 1 -o ( η t -1 ) , we have √ η t -1 η t -1 = 1 √ 1 -o ( η t -1 ) -1 = o ( η t -1 ) . Hence, there exists a universal positive C &gt; 0 such that ∥ ∥ ∥ (√ η t -1 η t -1 ) ˇ y t - √ η t -1 η t G ˇ y t ∥ ∥ ∥ ≤ Cη t ‖ ˇ y t ‖ for all t ≥ 0 . As a result,

<!-- formula-not-decoded -->

Let K 0 = max { m ≥ 0 : η m ≥ λ } . Since η t decreases in t and converges to 0, we know K 0 also decreases in λ . If t k ≤ K 0 , we have t k +1 = t k +1 and thus t k +1 -1 ∑ s = t k η s = η t k ≤ η 0 ; otherwise, t k +1 -1 ∑ s = t k η s ≤ 2 λ by definition. Summing over P (1) k from

0 to K -1 and using (52) yield

<!-- formula-not-decoded -->

The last inequality uses ∑ T t =0 η t ≤ CTη T for all T ≥ 1 . For a given λ , letting T →∞ can make the first term go to zero. Then letting λ → 0 make the second term vanish too. Hence, we have

<!-- formula-not-decoded -->

Next, we consider to bound P (2) k . To than end, we will use the Burkholder inequality which relates a martingale with its quadratic variation.

Lemma D.3 (Burkholder's inequality [Burkholder, 1988]) . Fix any p ≥ 2 . For a martingale difference { x t } t ∈ [ T ] in a real (or complex) Hilbert space, each with finite L p -norm, one has

<!-- formula-not-decoded -->

where B p is a universal positive constant depending only on p .

Hence,

<!-- formula-not-decoded -->

where ( a ) uses Lemma D.3; ( b ) uses Jensen's inequality; and ( c ) uses sup t ≥ 0 E ‖ glyph[epsilon1] t ‖ 4 &lt; ∞ . As before, we will discuss two cases depending on whether η t is larger than λ or not. It is equivalent to whether t k is greater than K 0 . Similar to the

argument in bounding K -1 ∑ k =0 P (1) k , we have

<!-- formula-not-decoded -->

where the last inequality uses ∑ T t =0 η t ≤ CTη T for all T ≥ 1 . From the last inequality, letting T →∞ makes these two terms converge to zero. Hence, we have

<!-- formula-not-decoded -->

Step 3: Putting the pieces together. Therefore,

<!-- formula-not-decoded -->

Since the probability of the left-hand side has nothing to do with λ , letting λ → 0 gives

<!-- formula-not-decoded -->

## D.1 Proof of Lemma D.2

For the proof in the section, we will consider random variables (or matrices) in the complex field C . Hence, we will introduce new notations for them. For a vector v ∈ C (or a matrix U ∈ C d × d ), we use v H (or U H ) to denote its Hermitian transpose or conjugate transpose. For any two vectors v , u ∈ C , with a slight abuse of notation, we use 〈 v , u 〉 = v H u to denote the inner product in C . For simplicity, for a complex matrix U ∈ C d × d , we use ‖ U ‖ to denote the its operator norm introduced by the complex inner product 〈· , ·〉 . When U ∈ R d × d , ‖ U ‖ is reduced to the spectrum norm.

Proof of Lemma D.2. By Lemma D.4, G = UDU -1 for two non-singular matrices U , D ∈ C d × d that satisfies 2 µ · I glyph[precedesequal] D + D H with µ := min i ∈ [ d ] λ i ( G ) for simplicity.

Lemma D.4 (Property of Hurwitz matrices, Lemma 1 in [Mou et al., 2020a]) . If -G ∈ R d × d be a Hurwitz matrix (i.e., Re λ i ( G ) &gt; 0 for all i ∈ [ d ] ), there exists a non-degenerate matrix U ∈ C d × d such that G = UDU -1 for some matrix D ∈ C d × d that satisfies

<!-- formula-not-decoded -->

where D H denotes the conjugate transpose or Hermitian transpose.

Notice that

<!-- formula-not-decoded -->

We then bound ‖ I -η t D ‖ as following.

<!-- formula-not-decoded -->

For simplicity, we define

Then we have

<!-- formula-not-decoded -->

where for simplicity we denote

<!-- formula-not-decoded -->

Taking the second-order moment on the both sides of (53), we obtain

<!-- formula-not-decoded -->

Due to η t +1 = (1 -o ( η t )) η t and η t = o (1) , there exists t 0 &gt; 0 so that for any t ≥ t 0 , η t ≤ 2 η t +1 and

<!-- formula-not-decoded -->

By Jensen's inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since E [Re 〈 ( I -η t D ) U -1 y t , U -1 glyph[epsilon1] t 〉|F t ] = 0 , it follows that

<!-- formula-not-decoded -->

where the last inequality follows from Hölder's inequality. Notice that sup t ≥ 0 E ‖ glyph[epsilon1] t ‖ 4 glyph[precedesorequal] 1 by assumption. Putting the pieces together, we have that there exists some c &gt; 0 such that

<!-- formula-not-decoded -->

By induction, one can show that

<!-- formula-not-decoded -->

of which the right hand side is the solution of the quadratic equation µx = c ( √ x + η 0 ) . Since U is non-singular, E h 2 t glyph[precedesorequal] 1 is equivalent to E ‖ y t ‖ 4 η -2 t glyph[precedesorequal] 1 .

## E A CONVERGENCE RESULT

Denote ∆ t = Q t -Q ∗ as the error of the Q-function estimate Q t in the t -th iteration. In this section, we study both asymptotic and non-asymptotic convergence of 1 T ∑ T t =0 E ‖ ∆ t ‖ 2 ∞ .

## E.1 For General Step Sizes

We first show that 1 T ∑ T t =0 E ‖ ∆ t ‖ 2 ∞ = o ( 1 √ T ) when using the general step size in Assumption 3.3.

Theorem E.1. Under Assumption 3.1 and using the general step size in Assumption 3.3, we have

<!-- formula-not-decoded -->

Proof of Theorem E.1. We will make use of the convergence result in [Chen et al., 2020b].

Theorem E.2 (Theorem 2.1 and Corollary 2.1.3 in [Chen et al., 2020b]) . Consider the algorithm x t +1 = x t + η t ( H ( x t ) -x t + ε t ) and x ∗ is the solution of H ( x ) = x . Assume ( i ) ‖H ( x ) - H ( y ) ‖ ∞ ≤ γ ‖ x -y ‖ ∞ for any x , y ∈ R D ; ( ii ) E [ ε t |F t ] = 0 and E [ ‖ ε t ‖ 2 ∞ |F t ] ≤ A + B ‖ x t ‖ 2 ∞ and ( iii ) η t is positive and non-increasing. If η 0 ≤ α 2 α 3 , it follows that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Recall the update rule is Q t = (1 -η t ) Q t -1 + η t ( r t + γ P t V t -1 ) = Q t -1 + η t ( r + γ PV t -1 -Q t -1 + ε t ) where ε t = r t -r + γ ( P t -P ) V t -1 . Let F t = σ ( { ( r τ , P τ ) } 0 ≤ τ&lt;t ) . Hence, E [ ε t |F t ] = 0 and E [ ‖ ε t ‖ 2 ∞ |F t ] ≤ 2 E ‖ r t -r ‖ 2 ∞ + 2 γ 2 E ‖ P t -P ‖ 2 ∞ ‖ V t -1 ‖ 2 ∞ := A + B ‖ Q t -1 ‖ 2 ∞ where the last equation uses A = 2 E ‖ r t -r ‖ 2 ∞ , B = 2 γ 2 E ‖ P t -P ‖ 2 ∞ and ‖ V t -1 ‖ ∞ = ‖ Q t -1 ‖ ∞ . Then setting ˜ η t = (1 -γ ) η t , by Theorem E.2, we have

<!-- formula-not-decoded -->

where

To simplify the notation, we denote

<!-- formula-not-decoded -->

It is clear that we have ∑ T t =0 ˜ η ( t,T ) = 1 . Then it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, it follows that

<!-- formula-not-decoded -->

Recall that Assumption 3.3 requires the step size satisfies

- (C1) 0 ≤ sup t η t ≤ 1 , η t ↓ 0 and tη t ↑ ∞ when t →∞ ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Noticing tη t ↑ ∞ due to (C1), we must have ∑ T t =1 ˜ η t -1 4 ln T → + ∞ and thus implies

<!-- formula-not-decoded -->

which, together with the Stolz-Cesaro theorem, implies 1 √ T ∑ T t =1 ˜ η 2 (0 ,t ) → 0 .

On the other hand, by Lemma E.1 and (C3), it follows that

<!-- formula-not-decoded -->

Lemma E.1. There exists some c &gt; 0 such that ∑ T l = t ˜ η ( t,l ) ≤ c for any T ≥ t ≥ 1 . Here { ˜ η ( t,l ) } l ≥ t ≥ 0 is defined in (56) and { ˜ η t } t ≥ 0 satisfies Assumption 3.3.

Putting all pieces together, we have established (54).

Proof of Lemma E.1. We define ˜ m t,l := ∑ l i = t ˜ η i . Due to t ˜ η t ↑ ∞ , we have t ˜ η t ≤ i ˜ η i for all i ≥ t and thus

<!-- formula-not-decoded -->

Since t ˜ η t ↑ ∞ , there exists some t 0 &gt; 0 such that any t ≥ t 0 , we have t ˜ η t ≥ 2 . Therefore, we have for all l ≥ t ≥ t 0 ,

<!-- formula-not-decoded -->

In the following, we will discuss three cases.

- If T ≥ t ≥ t 0 , by definition, it follows that

<!-- formula-not-decoded -->

where ( a ) follows from (57); ( b ) uses 1 -˜ η t ≥ 1 -˜ η 0 = γ ; and ( c ) uses ∑ T l = t ˜ η l exp ( -˜ m t,l 2 ) ≤ ∫ ∞ 0 exp( -x/ 2) dx = 2 due to ˜ m t,l ↑ ∞ as l →∞ .

- If T ≥ t 0 ≥ t , by definition, ˜ η ( t,l ) = ˜ η ( t,t 0 ) ˜ η ( t 0 ,l ) / ˜ η t 0 ≤ C 2 ˜ η ( t 0 ,l ) where C 2 = sup 0 ≤ t ≤ t 0 ˜ η ( t,t 0 ) / ˜ η t 0 . Then we have ∑ T l = t ˜ η ( t,l ) = ∑ t 0 l = t ˜ η ( t,l ) + ∑ T l = t 0 ˜ η ( t,l ) ≤ t 0 + C 2 ∑ T l = t 0 ˜ η ( t 0 ,l ) ≤ t 0 + C 2 2 √ e γ .
- If t 0 ≥ T ≥ t , we have ∑ T l = t ˜ η ( t,l ) ≤ t 0 .

Putting the three cases together, we can set c = t 0 + 2max { C 0 , 1 } √ e/γ which ensures that ∑ T l = t ˜ η ( t,l ) ≤ c for any T ≥ t ≥ 1 .

## E.2 For Two Specific Step Sizes

To obtain an log D dependence (which implies the rewards are distributed either sub-gaussian or sub-exponential), we use a almost-surely bounded rewards assumption as follows.

Assumption E.1. We assume 0 ≤ R ( s, a ) ≤ 1 for all ( s, a ) ∈ S × A .

Theorem E.3. Under Assumption E.1, there exist some positive constant c &gt; 0 such that

- If η t = 1 1+(1 -γ ) t , it follows that

<!-- formula-not-decoded -->

- If η t = t -α with α ∈ (0 , 1) for t ≥ 1 and η 0 = 1 , it follows that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

## E.3 Proof of Theorem E.3

Our proof is divided into three steps. The first is a upper bound for ‖ ∆ t ‖ ∞ provided by Lemma E.2: ‖ ∆ t ‖ ∞ ≤ a t + b t + ‖ N t ‖ ∞ , As a result, ‖ ∆ t ‖ 2 ∞ ≤ 3( a 2 t + b 2 t + ‖ N t ‖ 2 ∞ ) . Lemma E.2 follows from Theorem 1 in [Wainwright, 2019b] which views Q-learning as a cone-contractive operator and establishes a glyph[lscript] ∞ -norm bound.

Lemma E.2 (Theorem 1 in [Wainwright, 2019b]) . For any sequence of step sizes { η t } t ≥ 0 in the interval (0 , 1) , the iterates { ∆ t } t ≥ 0 satisfies the sandwich relation

<!-- formula-not-decoded -->

where { a t } t ≥ 0 , { b t } t ≥ 0 are non-negative scalars and { N t } t ≥ 0 are random vectors collecting noise terms from empirical Bellman operators. The three sequences are defined in a recursive way: they are initialized as a 0 = ‖ ∆ 0 ‖ ∞ , b 0 = 0 and N 0 = 0 and satisfy the following recursion:

<!-- formula-not-decoded -->

where Z t = ( r t -r ) + γ ( P t -P ) V ∗ is the empirical Bellman error at iteration t .

The second step is to bound E ‖ N T ‖ 2 ∞ which is an autoregressive process of independent Bellman noise terms. One can prove the result following a similar argument of Lemma 2 in [Wainwright, 2019b].

Lemma E.3. Under Assumption E.1 and assuming (1 -η t ) η t -1 ≤ η t for any t ≥ 1 , we have

<!-- formula-not-decoded -->

The final step is to establish the dependence of E ‖ ∆ T ‖ 2 ∞ on { η t } t ≥ 0 . Wainwright [2019b] finds it is crucial to set η t to be proportional to 1 / (1 -γ ) to ensure the sample complexity has polynomial dependence on 1 / (1 -γ ) . We then set ˜ η t = (1 -γ ) η t as the rescaled step size. We first redefine

<!-- formula-not-decoded -->

It is clear that we have ∑ T t =0 ˜ η ( t,T ) = 1 .

Lemma E.4. Under Assumption 3.1, if (1 -η t ) η t -1 ≤ η t for any t ≥ 1 , then we have

<!-- formula-not-decoded -->

where { ˜ η ( t,T ) } T ≥ t ≥ 0 defined in (59) and { N t } t ≥ 0 is defined in Lemma E.2.

Proof of Lemma E.4. By the recursion of { a t } t ≥ 0 and { b t } t ≥ 0 in Lemma E.2, it follows that

<!-- formula-not-decoded -->

Hence, a 2 T = ˜ η 2 (0 ,T ) ‖ ∆ 0 ‖ 2 ∞ and

<!-- formula-not-decoded -->

where ( a ) uses ∑ T t =1 ˜ η ( t,T ) = 1 -˜ η (0 ,T ) ≤ 1 and Jensen's inequality. Therefore,

<!-- formula-not-decoded -->

Given the condition (1 -η t ) η t -1 ≤ η t , we can apply Lemma E.3 which implies

<!-- formula-not-decoded -->

Plugging these bounds into (61) yields (60).

With these lemmas, we are ready to prove the following theorem.

Theorem E.4. Under Assumption 3.1, we have the following bounds for E ‖ ∆ T ‖ 2 ∞ . Here c &gt; 0 is a universal positive constant and might be overwritten (and thus different) in different statements. The specific value of different c 's can be found in our proof.

- If η t = 1 1+(1 -γ ) t , it follows that for all T ≥ 1 ,

<!-- formula-not-decoded -->

- If η t = t -α with α ∈ (0 , 1) for t ≥ 1 and η 0 = 1 , it follows that for all T ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem E.4. We discuss the two cases separately.

(I) Linearly rescaled step size. If we use a linear rescaled step size, i.e., η t = 1 1+(1 -γ ) t (equivalently ˜ η t = 1 -γ 1+(1 -γ ) t ), then we have (i) 1 -η t ≤ 1 -˜ η t = 1+(1 -γ )( t -1) 1+(1 -γ ) t = ˜ η t / ˜ η t -1 = η t /η t -1 for t ≥ 1 and (ii) ˜ η ( t,T ) ≤ ˜ η T . It implies Lemma E.4 is applicable. Notice that ∑ T t =1 ˜ η t -1 ≤ 1 + ∑ T -1 t =1 1 t ≤ 1 + ln( T -1) ≤ ln( eT ) and ln (1 -γ )( T +1) 2 ≤ ln 1+(1 -γ )( T +1) 1+(1 -γ ) = ∫ T +1 1 1 -γ 1+(1 -γ ) t dt ≤ ∑ T t =1 1 -γ 1+(1 -γ ) t = ∑ T t =1 ˜ η t . Hence,

<!-- formula-not-decoded -->

Finally, plugging these inequalities into (60), we have

<!-- formula-not-decoded -->

(II) Polynomial step size. If we choose a polynomial step size, i.e., η t = t -α with α ∈ (0 , 1) for t ≥ 1 and η 0 = 1 , then we again have 1 -η t = 1 -1 t α ≤ ( t -1 t ) α = η t /η t -1 for t ≥ 1 , which implies Lemma E.3 is applicable. Note that

<!-- formula-not-decoded -->

which implies that ∑ T t =1 η t ≥ ∑ T t =1 t -α ≥ 1 1 -α ( ( T +1) 1 -α -1 ) and ( T +1) 1 -α ≤ 1 + T 1 -α . Hence,

<!-- formula-not-decoded -->

Additionally, using η t -1 ≤ 2 η t for all t ≥ 1 and (63), we have,

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

At the the end of this subsection, we will prove that

, where

Lemma E.5. For any α ∈ (0 , 1) and β &gt; 0 , it follows that

<!-- formula-not-decoded -->

By setting β = 2 α , we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Putting together the pieces, we can safely conclude that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof of Lemma E.5. We do this via a similar argument of Lemma 4 in [Wainwright, 2019b]. Let f ( t ) = exp ( 1 -γ 1 -α t 1 -α ) t β . By taking derivatives, we find that f ( t ) is decreasing in t on the interval [0 , t ∗ ] and increasing for [ t ∗ , ∞ ) , where t ∗ = ( β 1 -γ ) 1 1 -α . Hence,

<!-- formula-not-decoded -->

Using integrating by parts, it follows that

<!-- formula-not-decoded -->

where the last equality uses definition of t ∗ and I ∗ . Hence, we have

<!-- formula-not-decoded -->

Putting together the pieces, we have shown that if T &gt; glyph[floorleft] t ∗ glyph[floorright] ,

<!-- formula-not-decoded -->

If T ≤ glyph[floorleft] t ∗ glyph[floorright] , then

<!-- formula-not-decoded -->

Thus we have proved the inequality is true for any choice of T .

Based on Theorem E.4, we now can prove Theorem E.3 by averaging the individual error bounds.

Proof of Theorem E.3. The result directly follows from Theorem E.4.

- For the first item, we already have E ‖ ∆ T ‖ 2 ∞ ≤ 12 ‖ ∆ 0 ‖ 2 ∞ (1 -γ ) 2 1 (1+ T ) 2 + 12 γ 2 ln(2 eD ) (1 -γ ) 5 ln( eT ) T . Using ∑ ∞ t =1 t -2 = π 2 6 and ∑ T t =1 t -1 ≤ 1 + ln T = ln( eT ) , we have for some universal constant c &gt; 0 ,

<!-- formula-not-decoded -->

- For the second item, we have E ‖ ∆ T ‖ 2 ∞ ≤ ∆ 0 exp ( -1 -γ 1 -α ( (1 + T ) 1 -α -1 ) ) + 114 ln(2 eD ) (1 -γ ) 4 1 T α with ∆ 0 = 3 ‖ ∆ 0 ‖ 2 ∞ + 48 γ 2 ln(2 eD ) (1 -γ ) 3 ( 2 α 1 -γ ) 1 1 -α . Notice that

<!-- formula-not-decoded -->

and ∑ T t =1 t -α ≤ ∫ T 0 t -α dt = T 1 -α 1 -α . Here ( a ) uses the change of variable x = 1 -γ 1 -α t 1 -α and ( b ) uses the definition of gamma function Γ( z ) = ∫ ∞ 0 e -x x z -1 dx . Finally ( c ) follows from a numeral inequality about gamma function. Since Γ(1 + x ) &lt; √ 2 π ( x +1 / 2 e ) x +1 / 2 for any x &gt; 0 (see Theorem 1.5 of [Batir, 2008]), then which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

## F PROOF OF THEOREM 5.1

In the section, we provide the proof for our finite-sample analysis of averaged Q-learning in the glyph[lscript] ∞ -norm. Our main idea is similar to Appendix C. The average Q-learning estimator ¯ Q T has the error

<!-- formula-not-decoded -->

Using two auxiliary sequences { ∆ 1 t } t ≥ 0 and { ∆ 2 t } t ≥ 0 defined in Lemma C.1, we similarly define

<!-- formula-not-decoded -->

Because ∆ 2 t ≤ ∆ t ≤ ∆ 1 t coordinate-wise, it is valid that

<!-- formula-not-decoded -->

As a result, E ‖ ¯ ∆ T ‖ ∞ ≤ E max {‖ ¯ ∆ 1 T ‖ ∞ , ‖ ¯ ∆ 2 T ‖ ∞ } . Hence, bounding ‖ ¯ ∆ T ‖ ∞ in expectation is reduced to bound the maximum between ‖ ¯ ∆ 1 T ‖ ∞ and ‖ ¯ ∆ 2 T ‖ ∞ . Given ¯ ∆ 1 T and ¯ ∆ 2 T are defined in a similar way (see Lemma C.1), they share a similar error decomposition.

## F.1 Error Decomposition

Setting r = 1 in (36), we obtain

<!-- formula-not-decoded -->

Similar to (38), we decompose ¯ ∆ 1 T into five separate terms

<!-- formula-not-decoded -->

Here one should distinguish T i with ψ i , the former a random variable and the latter a random function. Comparing (35) and (40), we find that D 1 j -1 = D 2 j -1 +( P π j -1 -P π ∗ ) ∆ j -1 . Repeating the same argument to ¯ ∆ 2 T , we obtain

<!-- formula-not-decoded -->

Here {T i } 3 i =0 are exactly the same as in (68). Putting the pieces together, we have

<!-- formula-not-decoded -->

## F.2 Bounding the Separate Terms

For ‖T 0 ‖ ∞ . Recall that C 0 = sup T ≥ j ≥ 0 ‖ A T j ‖ ∞ . Since η 0 = 1 ≤ C 0 , it is obvious that

<!-- formula-not-decoded -->

For ‖T 1 ‖ ∞ . We apply (85) in Lemma H.1 to bound T 1 := 1 T ∑ T j =1 G -1 Z j . Indeed, by setting B j ≡ I , X j = 1 T G -1 Z j , we have B = 1 , X = 1 (1 -γ ) 2 T and ‖ W T ‖ ∞ ≤ ‖ diag(Var Q ) ‖ ∞ T defined therein. Hence,

<!-- formula-not-decoded -->

For ‖T 2 ‖ ∞ . We also apply (85) in Lemma H.1 to analyze T 2 := 1 T ∑ T j =1 ( A T j -G -1 ) Z j . Indeed, by setting B j = A T j -G -1 , X j = 1 T Z j , we have B = 2 C 0 , X = 1 (1 -γ ) T and ‖ W T ‖ ∞ ≤ 1 T 2 ∑ T j =1 ‖ A T j -G -1 ‖ 2 ∞ ‖ Var( Z ) ‖ ∞ defined therein. Hence,

<!-- formula-not-decoded -->

For ‖T 3 ‖ ∞ . We apply (86) in Lemma H.1 to analyze T 3 := γ T ∑ T j =1 A T j ( P j -P )( V j -1 -V ∗ ) . Because T 3 is more complex than T 1 and T 2 , we defer the detailed proof in Appendix F.5. Lemma F.1.

<!-- formula-not-decoded -->

where C 0 is the uniform bound given in Lemma C.3 and D = |S × A| .

For ‖T 4 ‖ ∞ . We have already analyzed T 4 := γ T ∑ T j =1 A T j ( P π j -1 -P π ∗ ) ∆ j -1 in Lemma C.7. It follows that

<!-- formula-not-decoded -->

Remark F.1. Under Assumption 3.1 3.2 and 3.3, we assert that √ T E ‖T i ‖ = o (1) for i = 0 , 2 , 3 , 4 . It is handy to verify √ T ‖T 0 ‖ = o (1) . Lemma C.2 implies 1 T ∑ T j =1 ‖ A T j -G -1 ‖ 2 ∞ = o (1) , by which we conclude √ T E ‖T 2 ‖ = o (1) . Theorem E.1 shows 1 √ T ∑ T t =0 E ‖ ∆ t ‖ 2 ∞ → 0 when we use the general step size. We then know that both √ T E ‖T 3 ‖ and √ T E ‖T 4 ‖ converge to zero when T goes to infinity.

## F.3 Specific Rates for Two Step Sizes

(I) Linearly rescaled step size. If we use a linear rescaled step size, i.e., η t = 1 1+(1 -γ ) t (equivalently ˜ η t = 1 -γ 1+(1 -γ ) t ), then Lemma C.3 and Lemma C.4 give

<!-- formula-not-decoded -->

Hiding constant factors in c , Theorem E.3 gives

<!-- formula-not-decoded -->

Hence, combining these bounds with (71), (72), (73), (74), and (75), we have

<!-- formula-not-decoded -->

where ˜ O ( · ) hides polynomial dependence on logarithmic terms namely ln D and ln T . Here we use ‖ diag(Var Q ) ‖ ∞ ≤ ‖ Var( Z ) ‖ ∞ (1 -γ ) 2 to simplify the final inequality.

(II) Polynomial step size. If we choose a polynomial step size, i.e., η t = t -α with α ∈ (0 . 5 , 1) for t ≥ 1 and η 0 = 1 , then hiding constant factors in c , Lemma C.3 and Lemma C.4 give

<!-- formula-not-decoded -->

where O ( · ) hides constant factors on α . Theorem E.3 gives

<!-- formula-not-decoded -->

Hence, combining these bounds with (71), (72), (73), (74), and (75), we have

<!-- formula-not-decoded -->

where ˜ O ( · ) hides polynomial dependence on logarithmic terms, namely ln D and ln T . Here we use ‖ Var( Z ) ‖ ∞ ≤ 1 (1 -γ ) 2 , T -1+ α 2 ≤ T -α to simplify the final inequality.

## F.4 A Useful Inequality

The following is a useful inequality which will be used frequently in the subsequent proof.

Lemma F.2. For any matrices A , V with a compatible order, we have

<!-- formula-not-decoded -->

where ‖ V ‖ max = max i,k | V ( i, k ) | .

Proof of Lemma F.2. For any diagonal entry i , it follows that

<!-- formula-not-decoded -->

## F.5 Proof of Lemma F.1

Proof of Lemma F.1. Recall that T 3 = γ T ∑ T j =1 A T j ( P j -P )( V j -1 -V ∗ ) and F j is the σ -field generated by all randomness before (and including) iteration j . We will apply Lemma H.1 to prove our lemma. Using the notation defined therein, we set X j = γ T ( P j -P )( V j -1 -V ∗ ) and B j = A T j . Clearly, { X j } j ≥ 0 is a martingale difference sequence since E [ X j |F j -1 ] = γ T E [ P j -P |F j -1 ]( V j -1 -V ∗ ) = 0 . As a result, X = 4 γ T (1 -γ ) , B = C 0 , D = |S × A| and U j = Var[ X j |F j -1 ] . 8

Recall that W T = diag( ∑ T j =1 B j U j B glyph[latticetop] j ) . To upper bound E ‖ W T ‖ ∞ , we aim to find a upper bound for ‖ W T ‖ ∞ . We first note that

<!-- formula-not-decoded -->

8 To distinguish Var[ X j |F j -1 ] and the value function V j , we use U j to denote the conditional variance.

glyph[negationslash]

Here the last inequality uses (76). To bound ‖ U j ‖ max , we find that for any i = k , U j ( i, k ) = E [ e glyph[latticetop] i X j X glyph[latticetop] j e k |F j -1 ] = 0 due to each coordinate of X j are independent conditioning on F j -1 . Hence,

<!-- formula-not-decoded -->

where ( a ) again uses (76) and ( b ) uses ‖ P j -P ‖ ∞ ≤ ‖ P j ‖ ∞ + ‖ P ‖ ∞ = 2 .

Putting the pieces together, we have

<!-- formula-not-decoded -->

where we use sup j ‖ B j ‖ ∞ ≤ B = C 0 1 -γ . The rest follows from (86) in Lemma H.1 by plugging the corresponding B,X,D and σ 2 and the inequality ‖ V j -1 -V ∗ ‖ ∞ ≤ ‖ Q j -1 -Q ∗ ‖ ∞ = ‖ ∆ j -1 ‖ ∞ .

## G PROOF OF THE INFORMATION-THEORETIC LOWER BOUND

## G.1 Proof of Theorem 4.1

The semiparametric model P θ ∈ P P ×P R described in Section 4 is described through an infinite-dimensional parameter θ = ( P , R ) , which is partitioned into a finite-dimensional parameter P ∈ R D × S and an infinite-dimensional parameter R . The reason why R is infinite dimensional is because we don't specify the probability model of each R ( s, a ) , which is equivalent to considering the class of all p.d.f.'s on the interval [0 , 1] , which is infinite dimensional. The parameter of interest is a smooth function of θ , denoted by β ( θ ) = Q ∗ ∈ R D . To compute the semiparametric Cramer-Rao lower bound (see Definition 4.7 of [Vermeulen, 2011]), we need to compute

<!-- formula-not-decoded -->

where P γ is any parametric submodel containing the truth, i.e., P γ 0 = P θ . Hence, under one kind of parameterization, the true model P θ can be recovered by setting γ = γ 0 in the parametric submodel P γ . Here, Γ ( γ 0 ) = ∂ Q ∗ ∂γ | γ = γ 0 is the score and I ( γ 0 ) is the corresponding Fisher information matrix. Let γ 0 ( R ) (resp. γ 0 ( P ) ) be the finite-dimensional part of γ 0 that relates with R (resp. P ). Due to the (variational) independence between P and R , γ 0 ( P ) doesn't intersect with γ 0 ( R ) . Hence, (77) can be divided into two parts

<!-- formula-not-decoded -->

where P γ ( R ) (resp. P γ ( P ) ) denotes the parametric submodel depending only on R (resp. P ). The equality ( ∗ ) follows because in the case the parametric model P P is the full model and the parametric Cramer-Rao lower bound is not affected by any one-to-one reparameterization. Here, Γ ( P ) = ∂ Q ∗ ∂ P and I ( P ) is the (constrained) information matrix.

In the following, we will first handle the parametric part (i.e., the transition kernel P ) by computing the (constrained) information matrix and then cope with the nonparametric part (i.e., the random reward R ) by using semiparametric tools. Combining the two parts together, we find that the semiparametric efficiency bound is

<!-- formula-not-decoded -->

using the notation Z j = r j + γ P j V ∗ and the independence of r j and P j .

## G.1.1 Parametric Part

We first investigate the Cramer-Rao lower bound for estimating Q ∗ using samples from { P t } t ∈ [ T ] whose distribution is determined by P ∈ P with P defined in (15). Note that P ∈ P is linearly constrained, i.e.,

<!-- formula-not-decoded -->

where h : R D × S → R D with its (˜ s, ˜ a ) -th coordinate of h given by

<!-- formula-not-decoded -->

Hence, we encounter the Cramer-Rao lower bound for constrained parameters. Let C T ( P ) is the inverse Fisher information matrix using T i.i.d. samples under the constraint h ( P ) = 0 . Hence, C T ( P ) = C 1 ( P ) T and the constrained Cramer-Rao lower bound [Moore Jr, 2010] is

<!-- formula-not-decoded -->

where ∂ Q ∗ ∂ P is the partial derivatives computed ignoring the linear constraint h ( P ) = 0 .

To give a precise formulation of the bound (79), we first compute ∂ Q ∗ ∂ P .

Lemma G.1. Under Assumption 3.2, Q ∗ is differentiable w.r.t. P with the partial derivatives given by

<!-- formula-not-decoded -->

We then compute C 1 ( P ) via the following lemma.

Lemma G.2. The ( s, a ) -th row of the random matrix P t is given by P t ( s ′ | s, a ) = 1 { s t ( s,a )= s ′ } where s t ( s, a ) is the generated next-state from ( s, a ) at iteration t with probability given as the ( s, a ) -th row of P . Hence P = E P t and P belongs to the following parametric space

<!-- formula-not-decoded -->

with h defined in (78) . The constrained inverse Fisher information matrix C 1 ( P ) is

<!-- formula-not-decoded -->

By Lemma G.1 and G.2, we have

<!-- formula-not-decoded -->

The Cramer-Rao lower bound is thus equal to

<!-- formula-not-decoded -->

At the end of this part, we provide the deferred proof for Lemma G.1 and G.2.

Proof of Lemma G.1. Notice that Q ∗ = R + γ PV ∗ . Then by the chain rule, we have glyph[negationslash]

<!-- formula-not-decoded -->

Assumption 3.2 implies the optimal policy π ∗ is unique. Hence, using V ∗ ( s 1 ) = max a Q ∗ ( s 1 , a ) = Q ∗ ( s 1 , π ∗ ( s 1 )) , we have

<!-- formula-not-decoded -->

Notice that P ∗ (( s, a ) , (˜ s, ˜ a )) = P (˜ s | s, a ) 1 { ˜ a = π ∗ (˜ s ) } . Putting all the pieces together and solving { ∂Q ∗ ( s,a ) ∂P ( s ′ | ˜ s, ˜ a ) } s,a,s ′ , ˜ s, ˜ a from the linear system, we have

<!-- formula-not-decoded -->

Proof of Lemma G.2. We write our the log-likelihood of sample P t as

<!-- formula-not-decoded -->

which implies ∂ ∂ P log f P ( P t ) ∈ R S 2 A with the ( s, a, s ′ ) -th entry given by

<!-- formula-not-decoded -->

By definition of the Fisher information matrix, we have

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

By definition of h ( P ) , we rearrange h ( P ) into an S 2 A × SA matrix given by

<!-- formula-not-decoded -->

Let U ( P ) ∈ R S 2 A × ( S 2 A -SA ) be the orthogonal matrix whose column space is the orthogonal complement of the column space of H ( P ) , which stands for H ( P ) glyph[latticetop] U ( P ) = 0 and U ( P ) glyph[latticetop] U ( P ) = I . Using results in [Moore Jr, 2010], the constrained CRLB is

<!-- formula-not-decoded -->

We define an auxiliary matrix X ∈ R SA × S 2 A satisfying

<!-- formula-not-decoded -->

which implies

By H ( P ) glyph[latticetop] U ( P ) = 0 , we have

<!-- formula-not-decoded -->

where D ( P )(( s, a, s ′ ) , ( s, a, s ′ )) = 1 /P ( s ′ | s, a ) and takes value 0 elsewhere. Now we reformulate D ( P ) as a block diagonal matrix D ( P ) = diag( { D ( s,a ) } ( s,a ) ) := diag( { 1 /P ( ·| s, a ) } ( s,a ) ) where D ( s,a ) is a diagonal matrix with D ( s,a ) ( s ′ , s ′ ) = 1 /P ( s ′ | s, a ) . Similarly, we have H ( P ) = diag( { 1 S } ( s,a ) ) , where 1 S is an all-1 vector with dimension S , glyph[negationslash]

and U ( P ) = diag( { U ( s,a ) } ( s,a ) ) , where U ( s,a ) ∈ R S × S -1 satisfying U glyph[latticetop] ( s,a ) 1 S = 0 . In this way, C 1 ( P ) has a equivalent block diagonal formulation

<!-- formula-not-decoded -->

For each block ( s, a ) of C 1 ( P ) , the submatrix is exactly the constrained Cramer-Rao bound of a multinomial distribution P s,a = { P ( ·| s, a ) } , which is equal to diag( P s,a ) -P s,a P glyph[latticetop] s,a . Therefore,

<!-- formula-not-decoded -->

## G.1.2 Nonparametric Part

Next, we move on discussing the efficiency on rewards. Unlike P t that is generated according to a parametric model, the generating mechanism of r t can be arbitrary. In other words, a finite dimensional parametric space is not enough to cover the possible distributions of r t . Thus, semiparametric theory is needed here. Fortunately, our interest parameter Q ∗ = ( I -γ P π ∗ ) -1 r is linear in r := E r t , implying only the expectation of r t matters. In semiparametric theory [Van der Vaart, 2000, Tsiatis, 2006], the efficienct influence function for mean estimation is exatly the random variable minus its expectation. Lemma G.3 shows it is still true in our case.

Lemma G.3. Let Assumption 3.2 hold. Given a random sample r t , the most efficient influence function for estimating Q ∗ ( s, a ) for any ( s, a ) is

<!-- formula-not-decoded -->

where r = E r t . Hence, the semiparametric efficiency bound of estimating Q ∗ with { r t } t ∈ [ T ] is

<!-- formula-not-decoded -->

Proof of Lemma G.3. As r t ( s, a ) are independent with different ( s ′ , a ′ ) pairs, we can only consider randomness of one pair ( s, a ) .

Firstly, we consider a submodel family P R ε of P R that is parameterized by ε such that when ε = 0 , we recover the distribution of R ( s, a ) . That is P R ε = { R ε : ε ∈ [ -δ, δ ] and R ( s, a ) = R ε ( s, a ) | ε =0 } . This can be achieved by manipulating density functions of each R ( s, a ) . It is clear that P R ε is a parametric family on rewards and we can make use of results in parametric statistics for our purpose. By definition, we have for ( s, a ) ,

<!-- formula-not-decoded -->

For any (˜ s, ˜ a ) = ( s, a ) , we have glyph[negationslash]

<!-- formula-not-decoded -->

Recursively expanding the above terms like what we have done in Lemma G.1, we have

<!-- formula-not-decoded -->

Let F ε denote the cumulative distribution function of R ε ( s, a ) . Then we have

<!-- formula-not-decoded -->

where r ( s, a ) = E r t ( s, a ) and ∂ ∂ε log dF ε is the score function. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the parametric submodel family R ε is arbitrary, we conclude that the efficient influence function of Q ∗ (˜ s, ˜ a ) is φ (˜ s, ˜ a ) by Theorem 2.2 in [Newey, 1990]. Finally, as r t ( s, a ) is independent with each other r t ( s ′ , a ′ ) 's, our final result is obtained by summing the above equation over all ( s, a ) .

## G.2 Proof of Theorem 4.2

Proof of Theorem 4.2. Recall that ¯ ∆ T = 1 T ∑ T t =1 ( Q T -Q ∗ ) . Combining (67), (68) and (69), we have

<!-- formula-not-decoded -->

where the inequality holds coordinate-wise. In Appendix F.2, we have analyze E ‖T i ‖ ∞ with explicit upper bounds. It is easy to verify that √ T E ‖T i ‖ = o (1) for i = 0 , 2 , 3 , 4 (see Remark F.1). Hence,

<!-- formula-not-decoded -->

where Z t = ( r t -r ) + γ ( P t -P ) V ∗ is the Bellman noise at iteration t . This implies ¯ Q T is asymptotically linear with the influence function φ ( r t , P t ) := ( I -γ P ∗ ) -1 Z t .

The remaining issue is to prove regularity. By definition, a RAL estimator is regular for a semiparametric model P = P P ×P R if it is a RAL estimator for every parametric submodel P γ = P P ×P R ε ⊂ P where γ = ( P , ε ) is the finitedimensional parameter controlling P γ . In a parametric submodel P P ×P R ε , by Theorem 2.2 in [Newey, 1990], for the asymptotically linear estimator ¯ Q T of Q ∗ which has the influence function

<!-- formula-not-decoded -->

its regularity is equivalent to the equality where

glyph[negationslash]

<!-- formula-not-decoded -->

where S γ ( · ) is the score function, γ = ( P ′ , ε ) ∈ P P × [ -δ, δ ] is the finite-dimensional parameter and γ 0 = ( P , 0) is the true underlying parameter. Since P and ε are variationally independent, S γ ( γ 0 ) = ( S P ( γ 0 ) , S ε ( γ 0 )) .

glyph[negationslash]

For the transition kernel P . Since our parametric space P P has a linear constraint, it is not easy to compute the constrained score function. Hence, for P = { P ( s ′ | s, a ) } s,a,s ′ , we regard { P ( s ′ | s, a ) } s,a,s ′ = s 0 as free parameters where s 0 ∈ S is any fixed state and use it as our new parameter. For a fixed ( s, a ) , once P ( s ′ | s, a ) is determined for all s ′ = s 0 , one can recover P ( s 0 | s, a ) by P ( s 0 | s, a ) = 1 -∑ s ′ = s 0 P ( s ′ | s, a ) . In this way, each { P ( s ′ | s, a ) } s ′ = s 0 lies in a open set. We still denote the set collecting all feasible { P ( s ′ | s, a ) } s,a,s ′ = s 0 as P , but readers should remember that current P = { P ( s ′ | s, a ) } s,a,s ′ = s 0 ∈ R SA × ( S -1) . From (80) and under our new notation of P , S P ( γ 0 ) ∈ R SA ( S -1) with entries given by glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

By Lemma G.1 and the chain rule, it follows that ∂ Q ∗ ∂ P ∈ R SA × SA ( S -1) and its (˜ s, ˜ a, s ′ ) -th column is

<!-- formula-not-decoded -->

Since ( I -γ P π ∗ ) -1 has a full rank (i.e., SA ), it is easy to see that ∂ Q ∗ ∂ P also has rank SA by varying (˜ s, ˜ a ) and fixing s ′ , s 0 in (83). On the other hand, the (˜ s, ˜ a, s ′ ) -th column of E φ ( r t , P t ) S P ( θ 0 ) glyph[latticetop] is

<!-- formula-not-decoded -->

glyph[negationslash]

where the last equality uses the following result. By direct calculation, the ( s, a ) -th entry of E ( P t -P ) V ∗ [ 1 { s t ( s,a )= s ′ } P ( s ′ | s,a ) -1 { s t ( s,a )= s 0 } P ( s 0 | s,a ) ] is 0 for all ( s, a ) = (˜ s, ˜ a ) (due to independence) and the (˜ s, ˜ a ) -th entry is V ∗ ( s ′ ) -V ∗ ( s 0 ) . Indeed, the (˜ s, ˜ a ) -th entry of the mentioned matrix is glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Therefore, combining the results for all (˜ s, ˜ a, s ′ )( s ′ = s 0 ) , we have

<!-- formula-not-decoded -->

which implies (82) holds for the P part.

For the random reward R . Using the notation in the proof of Lemma G.3, S ε ( γ 0 ) = ∂ ∂ε log dF ε | ε =0 . By (81), we have

<!-- formula-not-decoded -->

which implies (82) holds for the ε part.

P R ε can be arbitrary, so (82) holds for all parametric submodels. This means ¯ Q T is regular for all parametric submodels and thus is regular for our semiparametric model.

## H A USEFUL CONCENTRATION INEQUALITY

We introduce a useful concentration inequality in this section. It captures the expectation and high probability concentration of a martingale difference sum in terms of ‖ · ‖ ∞ . It uses a similar idea of Theorem 4 in Li et al. [2021a] and is built on Freedman's inequality [Freedman, 1975] and the union bound.

Lemma H.1. Assume { X j } ⊆ R d are martingale differences adapted to the filtration {F j } j ≥ 0 with zero conditional mean E [ X j |F j -1 ] = 0 and finite conditional variance V j = E [ X j X glyph[latticetop] j |F j -1 ] . Moreover, assume { X j } j ≥ 0 is uniformly bounded, i.e., sup j ‖ X j ‖ ∞ ≤ X . For any sequence of deterministic matrices { B j } j ≥ 0 ⊆ R D × d satisfying sup j ‖ B j ‖ ∞ ≤ B , we define the weighted sum as

<!-- formula-not-decoded -->

and let W T = diag( ∑ T j =1 B j V j ( B j ) glyph[latticetop] ) be a diagonal matrix that collects conditional quadratic variations. Then, it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Generally, we have

<!-- formula-not-decoded -->

Proof of Lemma H.1. Fixing any i ∈ [ D ] , we denote the i -th row of B j as b glyph[latticetop] j . For simplicity, we omit the dependence of b j on i . Then the i -th coordinate of Y T is Y T ( i ) = ∑ T j =1 b glyph[latticetop] j X j and W T ( i, i ) = ∑ T j =1 b glyph[latticetop] j V j b j . Clearly { b glyph[latticetop] j X j } is a scalar martingale difference with W T ( i, i ) = ∑ T j =1 E [( b glyph[latticetop] j X j ) 2 |F j -1 ] the quadratic variation and | b glyph[latticetop] j X j | ≤ ‖ b j ‖ 1 ‖ X j ‖ ∞ ≤ ‖ B j ‖ ∞ ‖ X j ‖ ∞ = BX the uniform upper bound. By Freedman's inequality [Freedman, 1975], it follows that

<!-- formula-not-decoded -->

Then by the union bound, we have

<!-- formula-not-decoded -->

Solving for τ such that the right-hand side of (87) is equal to δ gives

<!-- formula-not-decoded -->

Using √ a + b ≤ √ a + √ b gives an upper bound on τ and provides the high probability result.

The tail bound of ‖ Y T ‖ ∞ 1 {‖ W T ‖ ∞ ≤ σ 2 } has already been derived in (87). For the expectation result, we refer to the conclusion of Exercise 2.8 (a) in [Wainwright, 2019a] which implies that

<!-- formula-not-decoded -->

where the last inequality uses √ a + √ b ≤ √ 2( a + b ) .

For the last result, we aim to bound E ‖ Y T ‖ ∞ without the condition ‖ W T ‖ ∞ ≤ σ 2 for some positive number σ . We first assert that there exists a trivial upper bound for ‖ W T ‖ ∞ which is ‖ W T ‖ ∞ ≤ TB 2 X 2 . This is because

<!-- formula-not-decoded -->

where ( a ) uses Lemma F.2 and ( b ) is due to ‖ V j ‖ max ≤ X 2 for all j ∈ [ T ] . However, if we set σ 2 = TB 2 X 2 in (85), the resulting expectation bound of E ‖ Y T ‖ ∞ has a poor dependence on T .

To refine the dependence, we adapt and modify the argument of Theorem 4 in Li et al. [2021a]. For any positive integer K , we define

<!-- formula-not-decoded -->

and claim that we have P ( H K ) ≤ δ . We observe that the event H K is contained within the union of the following K events: H K ⊆ ∪ k ∈ [ K ] B k where for 0 ≤ k &lt; K , B k is defined to be

<!-- formula-not-decoded -->

Invoking (84) with a proper σ 2 = TB 2 X 2 2 k -1 and δ = δ K , we have P ( B k ) ≤ δ K for all k ∈ [ K ] . Taken this result together with the union bound gives P ( H K ) ≤ ∑ k ∈ [ K ] P ( B k ) ≤ δ . Then we have

<!-- formula-not-decoded -->

where ( a ) uses ‖ Y T ‖ ∞ ≤ TBX , ( b ) follows by setting δ = 1 T and K = glyph[ceilingleft] log 2 T glyph[ceilingright] ≤ T , ( c ) uses √ a + b ≤ √ a + √ b , and ( d ) follows from Jensen's inequality and exp( 3 8 ) ≤ 3 2 .

## I PROOF FOR ENTROPY REGULARIZED Q-LEARNING

In this section, we provide the counterpart results for Q-Learning with entropy. Since the proof is almost similar to that of Q-Learning, we just provide a sketch for simplicity. Recall that the matrix-form of the update rule is

<!-- formula-not-decoded -->

It is easy to show L λ is a 1 -contraction with respect to ‖ · ‖ ∞ .

## I.1 Convergence Under the General Step Sizes

Theorem I.1. Under Assumption 3.1 and using the general step size in Assumption 3.3, we have

<!-- formula-not-decoded -->

where Q ∗ λ is the unique fixed point of the regularized Bellman equation Q ∗ λ = r + γ P L λ Q ∗ λ .

Proof of Theorem I.1. Denote ˜ ∆ t = ˜ Q t -Q ∗ λ for simplicity. We will show that lim T →∞ 1 √ T ∑ T t =0 E ‖ ˜ ∆ t ‖ 2 ∞ = 0 for the sequence generated via (16). Similar to Theorem E.1, we notice that the update rule satisfies ˜ Q t = ˜ Q t -1 + η t ( r + γ P L λ ˜ Q t -1 -˜ Q t -1 + ε t ) where ε t = r t -r + γ ( P t -P ) L λ ˜ Q t -1 . Hence, E [ ε t |F t ] = 0 and E [ ‖ ε t ‖ 2 ∞ |F t ] ≤ 2 E ‖ r t -r ‖ 2 ∞ + 2 γ 2 E ‖ P t -P ‖ 2 ∞ ‖L λ ˜ Q t -1 ‖ 2 ∞ := A + B ‖ ˜ Q t -1 ‖ 2 ∞ with A = 2 E ‖ r t -r ‖ 2 ∞ , B = 2 γ 2 E ‖ P t -P ‖ 2 ∞ . By Theorem E.2, we arrive the same inequality as (55). Following the same analysis therein, we can show lim T →∞ 1 √ T ∑ T t =0 E ‖ ˜ ∆ t ‖ 2 ∞ = 0 under the general step size in Assumption 3.3.

## I.2 Establishment of FCLT in Proof of Theorem 6.1

Proof of Theorem 6.1. Since the analysis is almost similar to that in Theorem 3.1, we just specify the differences. The three-step analysis in Section 3.2 still applies here except that we show only modify the first step.

Similar error decomposition. Let ˜ ∆ t = ˜ Q t -Q ∗ λ . By the regularized Bellman equation Q ∗ λ = r + γ P L λ Q ∗ λ , it follows that

<!-- formula-not-decoded -->

where we use ˜ Z t = ( r t -r ) + γ ( P t -P ) L λ Q ∗ λ is the regularized Bellman noise and ˜ Z ′ t = ( P t -P )( L λ ˜ Q t -1 -L λ ˜ Q ∗ λ ) (which is still a martingale difference.)

To analyze L λ ˜ Q t -1 -L λ ˜ Q ∗ λ , we introduce an intermediate linear operator L π λ , which is defined by

<!-- formula-not-decoded -->

for a given policy π and regularization coefficient λ . As a result of notation, ( L λ Q )( · ) = sup π ∈ Π ( L π λ Q )( · ) for all Q ∈ R D . We assume L λ ˜ Q t = L ˜ π t λ ˜ Q t and L λ ˜ Q ∗ λ = L π ∗ λ λ ˜ Q ∗ λ . Hence,

<!-- formula-not-decoded -->

where the last equation uses L π ∗ λ λ ˜ Q t -1 -L π ∗ λ λ ˜ Q ∗ λ = P π ∗ λ ˜ ∆ t -1 by definition. Putting pieces together,

<!-- formula-not-decoded -->

where ˜ A t = I -η t ( I -γ P π ∗ λ ) , ˜ Z ′ t = ( P t -P )( L λ ˜ Q t -1 -L λ ˜ Q ∗ λ ) , and ˜ Z ′′ t = P ( L ˜ π t -1 λ -L π ∗ λ λ ) ˜ Q t -1 . Recurring the last equality gives

<!-- formula-not-decoded -->

Besides, using the general step size in Assumption 3.3, we can show 1 √ T ∑ T t =1 E ‖ ˜ ∆ t ‖ 2 ∞ → 0 (in Theorem I.1).

Satisfied Lipschitz condition. In order to apply the second and third analysis in Section 3.2, we only need to show that ‖ ˜ Z ′′ t ‖ ∞ ≤ L ‖ ˜ ∆ t -1 ‖ 2 ∞ for an appropriate L &gt; 0 . Notice that L π ∗ λ λ ˜ Q t -1 ≤ L ˜ π t -1 λ ˜ Q t -1 and L ˜ π t -1 λ ˜ Q ∗ λ ≤ L π ∗ λ λ ˜ Q ∗ λ coordinately. It implies that ˜ Z ′′ t = P ( L ˜ π t -1 λ -L π ∗ λ λ ) ˜ Q t -1 satisfies

<!-- formula-not-decoded -->

Hence, ‖ ˜ Z ′′ t ‖ ∞ ≤ ‖ ( P ˜ π t -1 -P π ∗ λ ) ˜ ∆ t -1 ‖ ∞ ≤ ‖ P ˜ π t -1 -P π ∗ λ ‖ ∞ ‖ ˜ ∆ t -1 ‖ ∞ ≤ ‖ Π ˜ π t -1 -Π π ∗ λ ‖ ∞ ‖ ˜ ∆ t -1 ‖ ∞ . By definition of Π π , we know that

<!-- formula-not-decoded -->

On the other hand, ˜ π t -1 , π λ has a closed form in terms of ˜ Q t -1 and Q ∗ λ respectively. Actually, we have that ˜ π t -1 ( ·| s ) ∝ exp( ˜ Q t -1 ( s, · ) /λ ) and π ∗ λ ( ·| s ) ∝ exp( Q ∗ λ ( s, · ) /λ ) . By the following lemma, we know that ‖ π t -1 ( ·| s ) -π λ ( ·| s ) ‖ ∞ ≤ 1 λ ‖ Q ∗ λ ( s, · ) -˜ Q t -1 ( s, · ) ‖ ∞ . As a result, we have ‖ ˜ Z ′′ t ‖ ∞ ≤ L ‖ ˜ ∆ t -1 ‖ 2 ∞ with L = 1 λ .

LemmaI.1. For any vector v ∈ R d , let softmax : R d → R d be defined by softmax( v )( i ) = exp( v ( i )) / ∑ j ∈ [ d ] exp( v ( j )) . Then, ‖ softmax( v 1 ) -softmax( v 2 ) ‖ ∞ ≤ ‖ v 1 -v 2 ‖ ∞ .

Proof of Lemma I.1. For any v , it is easy to find that softmax( v ) = ∂L ( v ) ∂ v where L ( v ) = log( ∑ j ∈ [ d ] exp( v ( j ))) . It is easy to show that ∥ ∥ ∥ ∂ 2 L ( v ) ∂ 2 v ∥ ∥ ∥ ∞ ≤ 1 for any v . Hence, the result follows from Taylor's expansion.

The rest proof is almost the same as that in Section 3.2.

log T(e, y)

11

10 -

9

8

7

6

Baseline,k = 3

Linear,k = 2.31

Poly.,a = 0.51,k = 2.43

Poly.,a = 0.55,k = 2.54

Poly.,a = 0.6,k = 2.76

1.0

Figure 3: Left: log-log plots of the sample complexity T ( ε, γ ) versus the discount complexity parameter (1 -γ ) -1 . Right: the coverage rate and the average length of the 95% confidence interval for regularized Q-Learning.

<!-- image -->

## I.3 Non-asymptotic Bounds in Proof of Theorem 6.1

The error decomposition in Appendix F.1 still apply here. Let ∆ t = ˜ Q t -Q ∗ λ for simplicity. Hence, it follows that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Pay attention that the A T j used above depends on π ∗ λ rather than π ∗ now. As argued in last subsection, Assumption 3.2 is satisfied here with L = 1 λ .

The remaining thing are to repeat what we have done in Appendix F.2, analyzing each term ˜ T i 's using non-asymptotic concentration inequalities. There are some important aspects to notice. First, for any j , ‖ ˜ Z j ‖ ∞ ≤ 2(1 + γ ‖L λ Q ∗ λ ‖ ∞ ) ≤ 2(1 + γ ‖ Q ∗ λ ‖ ∞ + γλ Entropy( π ∗ λ )) ≤ 1+ λ log 1 |A| 1 -γ = ˜ O ( 1 1 -γ ) where we use Entropy( π ∗ λ ) ≤ log 1 |A| and ‖ Q ∗ -Q ∗ λ ‖ ∞ ≤ λ 1 -γ log 1 |A| (which is proved in Theorem 5 of [Yang et al., 2019]). Second, the properties of A T j 's in Lemma C.3 and C.4 still hold with the same parameters. Finally, we have a counterpart Theorem E.4 due to Theorem 1 in [Wainwright, 2019b] also holds here. The possible difference is that ‖ ˜ Z j ‖ ∞ is bounded λ 1 -γ log 1 |A| instead of 1 1 -γ , which is equivalent up to log factors. Hence, up to log factors, Theorem E.4 also holds for entropy regularized Q-Learning. Putting pieces together, we complete the proof.

## J DETAILS OF EXPERIMENTS

The setup of MDP. According to Theorem 5.1, for sufficiently small error ε &gt; 0 , we expect the sample complexity T ( ε, γ ) is always upper bounded by ‖ diag(Var Q ) ‖ ∞ and 1 (1 -γ ) 3 at a worst case. To ensure Assumption 3.2, we consider a random MDP. In particular, for each ( s, a ) pair, the random reward R ( s, a ) ∼ U (0 , 1) is the uniformly sampled from (0 , 1) and the transition probability P ( s ′ | s, a ) = u ( s ′ ) / ∑ s u ( s ) , where u ( s ) i.i.d. ∼ U (0 , 1) . The size of the MDP we choose is |S| = 4 , |A| = 3 . We consider 30 different values of γ equispaced between 0 . 6 and 0 . 9 . For a given γ , we run Q-learning algorithm for 10 5 steps (which already ensures convergence) and repeat the process independently for 10 3 times. Finally,

we average the glyph[lscript] ∞ error ‖ ¯ Q T -Q ∗ ‖ ∞ of the 10 3 independent trials as an approximation of E ‖ ¯ Q T -Q ∗ ‖ ∞ and compute T ( ε, γ ) by definition. The polynomial step size η t = t -α uses α ∈ { 0 . 51 , 0 . 55 , 0 . 60 } and the resacled linear step size is η t = (1+(1 -γ ) t ) -1 . In Figure 2, we choose ε = e -4 and plot the results on a log-log scale. We then plot the least-squares fits through these points and the slopes of these lines are also provided in the legend.

Confirming the theoretical predictions. In the body, we show the least-squares fits through the points { (log ‖ diag(Var Q ) ‖ ∞ , log T ( ε, γ )) } γ ∈ Γ . As a complementary, we also show the fits through { (log(1 -γ ) -1 , log T ( ε, γ )) } γ ∈ Γ in Figure 3.

Online inference experiments. We visualize the empirical coverage rate and confidence interval lengths of averaged Q-Learning in Figure 1. We use the random scaling method (Algorithm 1 in [Lee et al., 2021]) to compute the weighting matrix W T ∈ R D × D where W T = ∫ 1 0 ¯ φ T ( r ) ¯ φ T ( r ) glyph[latticetop] d r and ¯ φ T ( r ) = φ T ( r ) -r · φ T (1) . We focus on the inference of the optimal value function on the first state s 0 and the first action a 0 , i.e., Q ∗ ( s 0 , a 0 ) . We use 10 4 steps of value iteration to compute the optimal value function Q ∗ . From [Lee et al., 2021, Li et al., 2022], the asymptotic confidence interval is given by

<!-- formula-not-decoded -->

We set T = 10 4 and discard the first 5% samples as a warm-up. This warm-up is quite important; otherwise W T would change rapidly (as a result of fast convergence of Q T ) and deteriorate the performance. The performance is measured by two statistics: the coverage rate and the average length of the 95% confidence interval. We also provide similar results for regularized Q-Learning in Figure 3.