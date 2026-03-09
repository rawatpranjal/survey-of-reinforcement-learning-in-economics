This article was downloaded by: [73.183.48.82] On: 16 July 2022, At: 07:53 Publisher: Institute for Operations Research and the Management Sciences (INFORMS)

INFORMS is located in Maryland, USA

<!-- image -->

## To cite this article:

Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, Yuejie Chi (2021) Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization. Operations Research

Published online in Articles in Advance 02 Dec 2021

- .  https://doi.org/10.1287/opre.2021.2151

Full terms and conditions of use: https://pubsonline.informs.org/Publications/Librarians-Portal/PubsOnLine-Terms-andConditions

This article may be used only for the purposes of research, teaching, and/or private study. Commercial use or systematic downloading (by robots or other automatic processes) is prohibited without explicit Publisher approval, unless otherwise noted. For more information, contact permissions@informs.org.

The Publisher does not warrant or guarantee the article's accuracy, completeness, merchantability, fitness for a particular purpose, or non-infringement. Descriptions of, or references to, products or publications, or inclusion of an advertisement in this article, neither constitutes nor implies a guarantee, endorsement, or support of claims made of that product, publication, or service.

Copyright © 2021 The Author(s)

Please scroll down for article-it is on subsequent pages

<!-- image -->

With 12,500 members from nearly 90 countries, INFORMS is the largest international association of operations research (O.R.) and analytics professionals and students. INFORMS provides unique networking and learning opportunities for individual professionals, and organizations of all types and sizes, to better understand and use O.R. and analytics tools and methods to transform strategic visions and achieve better outcomes.

For more information on INFORMS, its publications, membership, or meetings visit http://www.informs.org

## Operations Research

Publication details, including instructions for authors and subscription information: http://pubsonline.informs.org

## Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization

Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, Yuejie Chi

Published in Operations Research on December 02, 2021 as DOI: 10.1287/opre.2021.2151.

This article has not been copyedited or formatted. The final version may differ from this version.

<!-- image -->

## Crosscutting Areas

## Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization

Shicong Cen, a Chen Cheng, b Yuxin Chen, c Yuting Wei, d Yuejie Chi a a Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, Pennsylvania 15213; b Department of Statistics, Stanford University, Stanford, California 94305; c Department of Electrical and Computer Engineering, Princeton University, Princeton, NewJersey 08544; d Department of Statistics and Data Science, The Wharton School, University of Pennsylvania, Philadelphia, Pennsylvania 19104

<!-- image -->

Contact: shicongc@andrew.cmu.edu (SC); chencheng@stanford.edu (CC); yuxin.chen@princeton.edu, https:/ /orcid.org/0000-0001-9256-5815 (YC); ytwei@wharton.upenn.edu, https:/ /orcid.org/0000-0002-3041-3434 (YW); yuejiechi@cmu.edu, https:/ /orcid.org/0000-0002-6766-5459 (YC)

<!-- image -->

Received:

August 5, 2020

Revised:

December 21, 2020

Accepted:

April 6, 2021

Published Online in Articles in Advance:

December 2, 2021

OR/MS Subject Classi /uniFB01 cations : Analysis of algorithms: computational complexity; decision analysis: theory Area of Review: Machine Learning and Data Science

https://doi.org/10.1287/opre.2021.2151

Copyright:

© 2021 The Author(s)

Abstract. Natural policy gradient (NPG) methods are among the most widely used policy optimization algorithms in contemporary reinforcement learning. This class of methods is often applied in conjunction with entropy regularization -an algorithmic scheme that encourages exploration -and is closely related to soft policy iteration and trust region policy optimization. Despite the empirical success, the theoretical underpinnings for NPG methods remain limited even for the tabular setting. This paper develops nonasymptotic convergence guarantees for entropy-regularized NPG methods under softmax parameterization, focusing on discounted Markov decision processes (MDPs). Assuming access to exact policy evaluation, we demonstrate that the algorithm converges linearly -even quadratically, once it enters a local region around the optimal policy -when computing optimal value functions of the regularized MDP. Moreover, the algorithm is provably stable vis-` a-vis inexactness of policy evaluation. Our convergence results accommodate a wide range of learning rates and shed light upon the role of entropy regularization in enabling fast convergence.

Open Access Statement: This work is licensed under a Creative Commons Attribution 4.0 International License. You are free to copy, distribute, transmit, and adapt this work, but you must attribute this work as ' Operations Research . Copyright © 2021 The Author(s). https:/ /doi.org/doi/10.1287/opre.2021.2151, used under a Creative Commons Attribution License: https:/ /creativecommons.org/licenses/by/4.0. '

Funding: This work was supported by the National Science Foundation, Air Force Of /uniFB01 ce of Scienti /uniFB01 c Research, Army Research Of /uniFB01 ce, and Of /uniFB01 ce of Naval Research. S. Cen and Y. Chi are supported in part by Grants ONR N00014-18-1-2142 and N00014-19-1-2404, ARO W911NF-18-1-0303, NSF CCF1806154, CCF-1901199, and CCF-2007911. C. Cheng is supported by the William R. Hewlett Stanford graduate fellowship. Y. Wei is supported in part by the National Science Foundation [Grants CCF-2106778, CCF-2007911, and DMS-2015447/2147546]. Y. Chen is supported in part by Grants AFOSR awards FA9550-19-1-0030 and FA9550-22-1-0198, ONR N00014-19-1-2120 and N00014-22-12354, ARO YIP award W911NF-20-1-0097, ARO W911NF-18-1-0303, NSF CCF-2106739/2221009, CCF-1907661, IIS-1900140 and IIS-2100158/2218773, the Google Research Scholar Award, and the Alfred P. Sloan Research Fellowship.

Supplemental Material: The e-companion is available at https:/ /doi.org/10.1287/opre.2021.2151.

Keywords :

reinforcement learning • natural policy gradient methods • entropy regularization • global convergence

## 1. Introduction

Policy gradient (PG) methods and their variants (Williams 1992, Konda and Tsitsiklis 2000, Sutton et al. 2000, Kakade 2002, Peters and Schaal 2008), which aim to optimize (parameterized) policies via gradient-type methods, lie at the heart of recent advances in reinforcement learning (RL) (see, e.g., Mnih et al. (2015), Schulman et al. (2015), Silver et al. (2016), and Schulman et al. (2017b)). Perhaps most appealing is their /uniFB02 exibility in adopting various kinds of policy parameterizations (e.g., a class of policies parameterized via deep neural networks), which makes them remarkably powerful and versatile in contemporary RL.

As an important and widely used extension of PG methods, natural policy gradient (NPG) methods, propose to employ natural policy gradients (Amari 1998) as search directions in order to achieve faster convergence than the update rules based on policy gradients (Kakade 2002, Peters and Schaal 2008, Bhatnagar et al. 2009, Even-Dar et al. 2009). Informally speaking, NPG methods precondition the gradient directions by Fisher information matrices (which are the Hessians of a certain divergence metric) and fall under the category of quasi second-order policy optimization methods. In fact, a variety of mainstream RL algorithms, such as trust region policy optimization

## OPERATIONS RESEARCH

Articles in Advance , pp. 1 -16

ISSN 0030-364X (print), ISSN 1526-5463 (online)

<!-- image -->

(TRPO) (Schulman et al. 2015) and proximal policy optimization (PPO) (Schulman et al. 2017b), can be viewed as generalizations of NPG methods (Shani et al. 2019). In this paper, we pursue in-depth theoretical understanding about this popular class of methods in conjunction with entropy regularization to be introduced momentarily.

## 1.1. Background and Motivation

Despite the enormous empirical success, the theoretical underpinnings of policy gradient type methods have been limited even until recently, primarily because of the intrinsic nonconcavity underlying the value maximization problem of interest (Bhandari and Russo 2019, Agarwal et al. 2020b). To further exacerbate the situation, an abundance of problem instances contain suboptimal policies residing in regions with /uniFB02 at curvatures (namely, vanishingly small gradients and high-order derivatives) (Agarwal et al. 2020b). Such plateaus in the optimization landscape could, in principle, be dif /uniFB01 cult to escape once entered, thereby necessitating a higher degree of exploration in order to accelerate policy optimization.

In practice, a strategy that has been frequently adopted to encourage exploration and improve convergence is to enforce entropy regularization (Williams and Peng 1991, Cen et al. 2021, Peters et al. 2010, Duan et al. 2016, Mnih et al. 2016, Haarnoja et al. 2017, Hazan et al. 2019, Xiao et al. 2019, Vieillard et al. 2020). By inserting an additional penalty term to the objective function, this strategy penalizes policies that are not stochastic/exploratory enough, in the hope of preventing a policy optimization algorithm from being trapped in an undesired local region. Through empirical visualization, Ahmed et al. (2019) suggested that entropy regularization induces a smoother landscape that allows for the use of larger learning rates and hence, faster convergence. However, the theoretical support for regularization-based policy optimization remains highly inadequate.

Motivated by this, a very recent line of works set out to elucidate, in a theoretically sound manner, the ef /uniFB01 -ciency of entropy-regularized policy gradient methods. Assuming access to exact policy gradients, Agarwal et al. (2020b) and Mei et al. (2020) developed convergence guarantees for regularized PG methods (with relative entropy regularization considered in Agarwal et al. 2020b and entropy regularization in Mei et al. 2020). Encouragingly, both papers suggested the positive role of regularization in guaranteeing faster convergence for the tabular setting. However, these works fell short of explaining the role of entropy regularization for other policy optimization algorithms like NPG methods, which we seek to understand in this paper.

## 1.2. This Paper

Inspired by recent theoretical progress toward understanding PG methods (Bhandari and Russo 2019,

Agarwal et al. 2020b, Mei et al. 2020), we aim to develop nonasymptotic convergence guarantees for entropyregularized NPG methods in conjunction with softmax parameterization. We focus attention on studying tabular discounted Markov decision processes (MDPs), which is an important /uniFB01 rst step and a stepping stone toward demystifying the effectiveness of entropy-regularized policy optimization in more complex settings.

- 1.2.1. Settings. Consider a γ -discounted in /uniFB01 nite-horizon MDP with state space S and action space A . Assuming availability of exact policy evaluation, the update rule of entropy-regularized NPG methods with softmax parameterization admits a simple update rule in the policy space (see Section 2 for precise descriptions)

for any ( s , a ) ∈ S × A , where τ &gt; 0 is the regularization parameter, 0 &lt; η ≤ 1 -γ τ is the learning rate (or stepsize), π ( t ) indicates the t -th policy iterate, and Q π τ is the soft Q-function under policy π (to be de /uniFB01 ned in (11a)). The update rule (1) is closely connected to several popular algorithms in practice. For instance, the trust region policy optimization (TRPO) algorithm (Schulman et al. 2015), when instantiated in the tabular setting, can be viewed as implementing (1) with line search. In addition, by setting the learning rate as η /equals 1 -γ τ , the update rule (1) coincides with soft policy iteration (SPI) studied in Haarnoja et al. (2017).

<!-- formula-not-decoded -->

- 1.2.2. Our Contributions. The results of this paper deliver fully nonasymptotic convergence rates of entropy-regularized NPG methods without any hidden constants, which are previewed as follows (in an orderwise manner). The de /uniFB01 nition of /epsilon1 -optimality can be found in Table 1.
- Linear convergence of exact entropy-regularized NPG methods. We establish linear convergence of entropy-regularized NPG methods for /uniFB01 nding the optimal policy of the entropy-regularized MDP, assuming access to exact policy evaluation. To yield an /epsilon1 -optimal policy for the regularized MDP (cf. Table 1), the algorithm (1) with a general learning rate 0 &lt; η ≤ 1 -γ τ needs no more than an order of

<!-- formula-not-decoded -->

iterations, where we hide the dependencies that are logarithmic on salient problem parameters (see Theorem 1). Some highlights of our convergence results are (i) their near dimension-free feature and (ii) their applicability to a wide range of learning rates (including small learning rates).

- Linear convergence of approximate entropyregularized NPG methods. We demonstrate the stability of the regularized NPG method with a general

Table 1. The Iteration Complexities of NPG Methods to Reach /epsilon1 -Accuracy in Terms of Optimization Error, Where the Unregularized (Resp. Regularized) Version is Given by (13) (cf. (15)) with η the Learning Rate

| Paper                     | Iteration complexity upper bound                  | Regularization   | Learning rates           |
|---------------------------|---------------------------------------------------|------------------|--------------------------|
| Agarwal et al. (2020b)    | 2 ( 1 - γ ) 2 /epsilon1 + 2 η /epsilon1           | Unregularized    | constant: ( 0, ∞)        |
| Bhandari and Russo (2020) | 1 ( 1 - γ ) min s ∈ S ρ ( s ) log ( 1 /epsilon1 ) | Unregularized    | exact line search        |
| This work                 | 1 1 - γ log ( 1 /epsilon1 )                       | Regularized      | constant: 1 - γ τ        |
| This work                 | 1 ητ log ( 1 /epsilon1 )                          | Regularized      | constant: ( 0, 1 - γ τ ) |

Notes. We assume exact gradient evaluation and softmax parameterization and hide the dependencies that are logarithmic on problem parameters. Here, /epsilon1 -accuracy or /epsilon1 -optimality for the unregularized (resp. regularized) case means that V ? ( s ) -V π ( t ) ( s ) ≤ /epsilon1 (resp. V ? τ ( s ) -V π ( t ) τ ( s ) ≤ /epsilon1 ) holds simultaneously for all s ∈ S ; ρ denotes the initial state distribution, which clearly obeys 1 min s ∈ S ρ ( s ) ≥ | S | .

learning rate 0 &lt; η ≤ 1 -γ τ even when the soft Q-functions of interest are only available approximately. This paves the way for future investigations that involve /uniFB01 nite-sample analysis. Informally speaking, the algorithm exhibits the same convergence behavior as in the exact gradient case before an error /uniFB02 oor is hit, where the error /uniFB02 oor scales linearly in the entry-wise error of the soft Q-function estimates (see Theorem 2).

· Quadratic convergence in the small/epsilon1 regime . In the high-accuracy regime, where the target level /epsilon1 is very small, the algorithm (1) with η /equals 1 -γ τ converges superlinearly, in the sense that the iteration complexity to reach /epsilon1 -accuracy for the regularized MDP is at most on the order of after entering a small local neighborhood surrounding the optimal policy. Here, we again hide the dependencies that are logarithmic on salient problem parameters (see Theorem 3).

<!-- formula-not-decoded -->

- 1.2.3. Comparisons with Prior Art. Agarwal et al. (2020b) proved that unregularized NPG methods with softmax parameterization attain an /epsilon1 -accuracy within O ( 1 = /epsilon1 ) iterations. In contrast, our results assert that O ( log ( 1 = /epsilon1 )) iterations suf /uniFB01 ce with the assistance of entropy regularization, which hints at the potential bene /uniFB01 t of entropy regularization in accelerating the convergence of NPG methods. Shortly after the initial posting of our paper, Bhandari and Russo (2020) posted a note that proves linear convergence of unregularized NPG methods with exact line search, by exploiting a clever connection to policy iteration. Their convergence rate is governed by a quantity min s ∈ S ρ ( s ) , resulting in an iteration complexity at least | S | times larger than ours. In comparison, our results cover a broad range of /uniFB01 xed learning rates (including small step sizes that are of particular interest in practice) and accommodate the scenario with inexact gradient evaluation. See Table 1 for a quantitative comparison. Moreover, we note that the entropyregularized NPG method with general learning rates

is closely related to TRPO in the tabular setting (see Shani et al. 2019). The recent work by Shani et al. (2019) demonstrated that TRPO converges with an iteration complexity O ( 1 = /epsilon1 ) in entropy-regularized MDPs. The analysis therein is inspired by the mirror descent theory in generic optimization literature, which characterizes sublinear convergence under properly decaying step sizes and accommodates various choices of divergence metrics. In comparison, our analysis strengthens the performance guarantees by carefully exploiting properties speci /uniFB01 c to the current version of the NPG method. In particular, we identify the delicate interplay between the crucial operational quantities Q ? τ -Q ( t ) τ and Q ? τ -τ log ξ ( t ) (to be de /uniFB01 ned later) and invoke the linear system theory to establish appealing contractions, which allow for the use of more aggressive constant step sizes and hence, improved convergence.

It is also helpful to compare our results with the state-of-the-art theory for PG methods with softmax parameterization (Agarwal et al. 2020b, Mei et al. 2020). Speci /uniFB01 cally, Agarwal et al. (2020b) established the asymptotic convergence of unregularized PG methods with softmax parameterization, whereas an iteration complexity of O ( 1 = /epsilon1 ) was recently pinned down by Mei et al. (2020). In the presence of entropy regularization, Agarwal et al. (2020b) showed that PG with relative entropy regularization and softmax parameterization enjoys an iteration complexity of O ( 1 = /epsilon1 2 ) , whereas Mei et al. (2020) showed that the entropy-regularized softmax PG method converges linearly in O ( log ( 1 = /epsilon1 )) iterations. However, the dependencies of the iteration complexity in Mei et al. (2020) on other salient parameters like | S | , | A | and 1 1 -γ are not fully speci /uniFB01 ed. Very recently, Li et al. (2021b) delivered a negative message demonstrating that these dependencies can be highly pessimistic; in fact, one can /uniFB01 nd an MDP instance that takes softmax PG methods (super)-exponential time (in terms of | S | and 1 1 -γ ) to converge. In contrast, the bounds derived in the current paper are fully nonasymptotic, delineating clear dependencies on all salient problem parameters, which clearly demonstrate the

4

algorithmic advantages of NPG methods. Figure 1 depicts the policy paths of PG and NPG methods with entropy regularization for a simple bandit problem with three actions. It is evident from the plots that the NPG method follows a more direct path to the global optimum compared with the PG counterpart and hence, converges faster. In addition, both algorithms converge more rapidly as the regularization parameter τ increases.

## 1.3. Other Related Works

There has been a /uniFB02 urry of recent activities in studying theoretical behaviors of policy optimization methods. For example, Fazel et al. (2018), Jansch-Porto et al. (2020), Tu and Recht (2019), Zhang et al. (2019a), and Mohammadi et al. (2019) established the global convergence of policy optimization methods for a couple of control problems, Bhandari and Russo (2019) identi /uniFB01 ed structural properties that guarantee the global optimality of PG methods without parameterization, Karimi et al. (2019) studied the convergence of PG methods to an approximate /uniFB01 rst-order stationary point, and Zhang et al. (2019b) proposed a variant of

PG methods that converges to locally optimal policies leveraging saddle-point escaping algorithms in nonconvex optimization. Beyond the tabular setting, the convergence of PG methods with function approximations has been studied in Agarwal et al. (2020b), Wang et al. (2019), and Liu et al. (2019). In particular, Cai et al. (2019) developed an optimistic variant of NPG that incorporates linear function approximation. We do not elaborate on this line of works since our focus is on understanding the performance of entropyregularized NPG in the tabular setting; we also do not elaborate on PG methods that involve sample-based estimates, since we primarily consider exact gradients or black-box gradient estimators.

Regarding entropy regularization, Neu et al. (2017) and Geist et al. (2019) provided uni /uniFB01 ed views of entropy-regularized MDPs from an optimization perspective by connecting them to algorithms such as mirror descent (Nemirovsky and Yudin 1983) and dual averaging (Nesterov 2009). The soft policy iteration algorithm has been identi /uniFB01 ed as a special case of entropy-regularized NPG, highlighting again the link between policy gradient methods and soft Q-learning

Figure 1. (Color online) Comparisons of PG and NPG Methods with Entropy Regularization for a Bandit Problem ( γ /equals 0) with Three Actions, Whose Corresponding Rewards are 1.0, 0.9, and 0.1, Respectively

<!-- image -->

Notes. The regularization parameter is set as τ /equals 0.1 for the /uniFB01 rst row and τ /equals 1 for the second row. In (a) and (d), the policy paths of (log π (a1), log π (a2)) following the PG method are plotted in orange, with the blue lines indicating the gradient /uniFB02 ow; in (b) and (e), the policy paths of (log π (a1), log π (a2)) following the NPG method are depicted in red, with the blue lines indicating the natural gradient /uniFB02 ow. The error contractions of both PG and NPG methods with η /equals 0.1 are shown in (c) and (f).

(Schulman et al. 2017a). The asymptotic convergence of soft policy iteration was established in Haarnoja et al. (2017), which fell short of providing explicit convergence rate guarantees. Additionally, Grill et al. (2019) developed planning algorithms for entropyregularized MDPs, and Mei et al. (2020) showed that the suboptimality gap of soft policy iteration is small if the policy improvement is small in consecutive iterations.

## 1.4. Notation

We denote by ∆ ( S ) (resp. ∆ ( A ) ) the probability simplex over the set S (resp. A ). When scalar functions such as | · | , exp (·) and log (·) are applied to vectors, their applications should be understood in an entrywise fashion. For instance, given any vector z /equals [ zi ] 1 ≤ i ≤ n ∈ R n , the notation | · | denotes | z | : /equals [| zi |] 1 ≤ i ≤ n ; other functions are de /uniFB01 ned analogously. For any vectors z /equals [ zi ] 1 ≤ i ≤ n and w /equals [ wi ] 1 ≤ i ≤ n , the notation z ≥ w (resp. z ≤ w ) means zi ≥ wi (resp. zi ≤ wi ) for all 1 ≤ i ≤ n . The softmax function softmax : R n /turnstileleft→ R n is de /uniFB01 ned such that [ softmax ( θ )] i : /equals exp ( θ i ) = ( Σ i exp ( θ i )) for a vector θ /equals [ θ i ] 1 ≤ i ≤ n ∈ R n . Given two probability distributions π 1 and π 2 over A , the Kullback-Leibler (KL) divergence from π 2 to π 1 is de /uniFB01 ned by KL ( π 1 || π 2 ) : /equals Σ a ∈ A π 1 ( a ) log π 1 ( a ) π 2 ( a ) . Given two probability distributions p and q over S , we introduce the notation ‖ p q ‖∞ : /equals max s ∈ S p ( s ) q ( s ) and ‖ 1 q ‖∞ : /equals max s ∈ S 1 q ( s ) .

## 2. Model and Algorithms

## 2.1. Problem Settings

2.1.1. Markov Decision Processes. The current paper studies a discounted Markov decision process (MDP) (Puterman 2014) denoted by M /equals ( S , A , P , r , γ ) , where S is the state space, A is the action space, γ ∈ ( 0, 1 ) indicates the discount factor, P : S × A → ∆ ( S ) is the transition kernel, and r : S × A →[ 0, 1 ] stands for the reward function. 1 To be more speci /uniFB01 c, for each state-action pair ( s , a ) ∈ S × A and any state s ′ ∈ S , we denote by P ( s ′ | s , a ) the transition probability from state s to state s ′ when action a is taken and r ( s , a ) the instantaneous reward received in state s due to action a . A policy π : S → ∆ ( A ) represents a (randomized) action selection rule; namely, π ( a | s ) speci /uniFB01 es the probability of executing action a in state s for each ( s , a ) ∈ S × A .

2.1.2. Value Functions and Q-functions. For any given policy π , we denote by V π : S → R the corresponding value function, namely, the expected discounted cumulative reward with an initial state s 0 /equals s , given by

<!-- formula-not-decoded -->

where the action at ~ π (·| st ) follows the policy π , and st + 1 ~ P (·| st , at ) is generated by the MDP M for all t ≥ 0. Wealso overload the notation V π ( ρ ) to indicate the expected value function of a policy π when the initial state is drawn from a distribution ρ over S , namely,

<!-- formula-not-decoded -->

Additionally, the Q-function Q π : S × A → R of a policy π -namely, the expected discounted cumulative reward with an initial state s 0 /equals s and an initial action a 0 /equals a -is de /uniFB01 ned by

<!-- formula-not-decoded -->

where the action at ~ π (·| st ) follows the policy π for all t ≥ 1, and st + 1 ~ P (·| st , at ) is generated by the MDP M for all t ≥ 0.

2.1.3. Discounted State Visitation Distributions. A type of marginal distributions -commonly dubbed as discounted state visitation distributions -plays an important role in our theoretical development. To be specific, the discounted state visitation distribution d π s 0 of a policy π given the initial state s 0 ∈ S is de /uniFB01 ned by

<!-- formula-not-decoded -->

where the trajectory ( s 0 , s 1 , /uni22EF ) is generated by the MDP M under policy π starting from state s 0 . In words, d π s 0 (·) captures the state occupancy probabilities when each state visitation is properly discounted depending on the time stamp. Further, for any distribution ρ over S , we de /uniFB01 ne the distribution d π ρ as follows

<!-- formula-not-decoded -->

which describes the discounted state visitation distribution when the initial state s 0 is randomly drawn from a prescribed initial distribution ρ .

2.1.4. Softmax Parameterization. It is common practice to parameterize the class of feasible policies in a way that is amenable to policy optimization. The focal point of this paper is softmax parameterization, a widely adopted scheme that naturally ensures that the policy lies in the probability simplex. Speci /uniFB01 cally, for any θ : S × A → R (called ' logic values ' ), the corresponding softmax policy πθ is generated through the softmax transform

<!-- formula-not-decoded -->

In what follows, we shall often abuse the notation to treat πθ and θ as vectors in R | S || A | and suppress the subscript θ from πθ , whenever it is clear from the context.

2.1.5. Entropy-Regularized Value Maximization. To promote exploration and discourage premature convergence to suboptimal policies, a widely used strategy is entropy regularization, which searches for a policy that maximizes the following entropy-regularized value function

<!-- formula-not-decoded -->

Here, the quantity τ ≥ 0 denotes the regularization parameter, and H ( ρ , π ) stands for a sort of discounted entropy de /uniFB01 ned as follows

Equivalently, V π τ can be viewed as the value function of π by adjusting the instantaneous reward to be the policy-dependent regularized version as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also de /uniFB01 ne V π τ ( s ) analogously when the initial state is /uniFB01 xed to be any given state s ∈ S . The regularized Q-function Q π τ of a policy π , also known as the soft Q-function, 2 is related to V π τ as

<!-- formula-not-decoded -->

(11b)

2.1.6. Optimal Policies and Stationary Distributions. Denote by π ? (resp. π ? τ ) the policy that maximizes the value function (resp. regularized value function with regularization parameter τ ), and let V ? (resp. V ? τ ) represent the resulting optimal value function (resp. regularized value function). Importantly, the optimal policies π ? and π ? τ of the MDP do not depend on the initial distribution ρ (Mei et al. 2020). In addition, π ? and π ? τ maximize the Q-function and the soft Q-function, respectively (which is selfevident from (11a)). A simple yet crucial connection between π ? and π ? τ can be demonstrated via the following sandwich bound 3

<!-- formula-not-decoded -->

which holds for all initial distributions ρ . The key takeaway message is that the optimal policy π ? τ of the regularized problem could also be nearly optimal in terms of the unregularized value function, as long as the regularization parameter τ is chosen to be suf /uniFB01 ciently small.

## 2.2. Algorithm: NPG Methods With Entropy Regularization

2.2.1. Natural Policy Gradient Methods. Toward computing the optimal policy (in the parameterized form), perhaps the /uniFB01 rst strategy that comes into mind is to run gradient ascent w.r.t. the parameter θ until convergence, a /uniFB01 rst-order method commonly referred to as the policy gradient (PG) algorithm (see, e.g., Sutton et al. 2000). In comparison, the natural policy gradient (NPG) method (Kakade 2002) adopts a preconditioned gradient update rule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in the hope of searching along a direction independent of the policy parameterization in use. Here, η is the learning rate or step size, F θ ρ denotes the Fisher information matrix given by and we use B † to indicate the Moore-Penrose pseudoinverse of a matrix B . It has been understood that the NPG method essentially attempts to monitor/control the policy changes approximately in terms of the Kullback-Leibler (KL) divergence (see, e.g., Section 7 in Schulman et al. 2015).

2.2.2. NPG Methods With Entropy Regularization. Equipped with entropy regularization, the NPG update rule can be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where F θ ρ is de /uniFB01 ned in (14) and V π τ ( ρ ) is de /uniFB01 ned in (8). Under softmax parameterization, this update rule admits a fairly simple form in the policy space, which, interestingly, is invariant to the choice of ρ . More precisely, if we let θ ( t ) denote the t th iterate and π ( t ) /equals softmax ( θ ( t ) ) the associated policy, then the entropyregularized NPG updates satisfy where Q π ( t ) τ is the soft Q-function of policy π ( t ) , and Z ( t ) ( s ) is some normalization factor. This can alternatively be viewed as an instantiation/variant of the trust region policy optimization (TRPO) algorithm (see

Schulman et al. 2015 and Shani et al. (2019). As an important special case, the update rule (16) reduces to

<!-- formula-not-decoded -->

for some normalization factor Z ( t ) ( s ) . The procedure (17) can be interpreted as a ' soft ' version of the classical policy iteration algorithm (Bertsekas 2017) (as it employs a softmax function to approximate the max operator) w.r.t. the soft Q-function and is often dubbed as soft policy iteration (SPI) (see Section 4.1 in Haarnoja et al. 2018).

To simplify notation, we shall use V ( t ) τ , Q ( t ) τ and d ( t ) ρ throughout to denote V π ( t ) τ , Q π ( t ) τ and d π ( t ) ρ , respectively. The complete procedure is summarized in Algorithm 1.

Algorithm 1 (Entropy-Regularized NPG With Exact Policy Evaluation)

1. Inputs: learning rate η , initialization π ( 0 ) .
2. For t /equals 0, 1, 2, /uni22EF do
3. Compute the regularized Q-function Q ( t ) τ . (de /uniFB01 ned in (11a)) of policy π ( t ) .
4. Update the policy:

<!-- formula-not-decoded -->

## 2.3. A Warm-Up Example: The Bandit Case

Inspired by Schulman et al. (2017a) and Mei et al. (2020), we look at a toy example -the bandit case -before proceeding to general MDPs. To be more precise, this is concerned with an MDP with only a single state and discount factor γ /equals 0. Despite its simplicity, the exposition of this example sheds light upon the convergence behavior of the regularized NPG methods of interest.

In this single-state example with γ /equals 0, the aim reduces to computing a policy πθ : A → ∆ ( A ) that solves the following optimization problem

<!-- formula-not-decoded -->

where r ( a ) is the instantaneous reward of taking action a (i.e., pulling arm a in the bandit language). As demonstrated in Proposition 1 in Mei et al. (2020), this toy case is already nonconcave and hence, nontrivial to solve. As it turns out, direct calculation reveals that the optimal policy of (19) is given by

<!-- formula-not-decoded -->

which is in general a randomized policy. When applied to this example, the entropy-regularized NPG

update rule (18) simpli /uniFB01 es to (up to normalization)

with η the learning rate. The following proposition, whose proof is fairly elementary and can be found in the suppmental material reveals that the above procedure converges (at least) linearly to the optimal policy π ? τ .

<!-- formula-not-decoded -->

Proposition 1 (The Bandit Case). The algorithm (21) converges linearly to π ? τ (cf. (20)) in an entrywise fashion, namely,

Although this result concentrates only on a toy example, it hints at the potential capability of entropyregularized NPG methods in achieving rapid convergence. In particular, by setting the learning rate to be η /equals 1 = τ , the algorithm converges in a single iteration . This special choice corresponds to the SPI update (17), which will be singled out in our general theory due to its appealing convergence properties.

<!-- formula-not-decoded -->

## 3. Main Results

Given its appealing convergence behavior when applied to the preceding warm-up example (the bandit case), it is natural to ask whether the entropyregularized NPG method is fast-convergent for general MDPs. This section answers this question in the af /uniFB01 rmative.

## 3.1. Exact Entropy-Regularized NPG Methods

We /uniFB01 rst study the convergence behavior of entropyregularized NPG methods (18) assuming access to exact policy evaluation in every iteration (namely, we assume that the soft Q-function Q ( t ) τ can be evaluated accurately in all t ). Remarkably, this algorithm converges linearly -in terms of computing both the optimal soft Q-function Q ? τ and the associated log policy log π ? τ -as asserted by the following theorem. The proof of this result is provided in Section 4.2.

Theorem 1 (Linear Convergence of Exact EntropyRegularized NPG). For any learning rate 0 &lt; η ≤ ( 1 -γ ) = τ , the entropy-regularized NPG updates (18) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all t ≥ 0, where

<!-- formula-not-decoded -->

It is worth emphasizing that Theorem 1 is stated in a completely nonasymptotic form containing no hidden constants and that our result covers any learning rate η in the range ( 0, ( 1 -γ ) = τ ] . A few implications of this theorem are in order.

- Linear convergence of soft Q-functions. To reach ‖ Q ? τ -Q ( t ) τ ‖∞ ≤ /epsilon1 , the entropy-regularized NPG method needs at most 1 ητ log ( C 1 γ /epsilon1 ) iterations. Remarkably, the iteration complexity almost does not depend on the dimensions of the MDP (except for some very weak dependency embedded in log C 1 ); this inherits a dimension-free feature of NPG methods that has been highlighted in Agarwal et al. (2020b) for the unregularized case. When the learning rate η is /uniFB01 xed in the admissible range, the iteration complexity scales inverse proportionally with τ , suggesting that a higher level of entropy regularization might accelerate convergence, albeit to the solution of a regularized problem that is further away from the original MDP.
- Linear convergence of log policies. In contrast to the unregularized case, entropy regularization ensures uniqueness of the optimal policy and, therefore, makes it possible to study the convergence of the policy directly. Our theorem reveals that the entropy-regularized NPGmethodneeds at most 1 ητ log ( 2 C 1 /epsilon1 τ ) iterations to yield ‖ log π ? τ -log π ( t + 1 ) ‖∞ ≤ /epsilon1 .
- Linear convergence of soft value functions. As a byproduct, Theorem 1 implies that the iterates of soft value functions also converge linearly, namely,

<!-- formula-not-decoded -->

To see this, we make note of the following relation previously established in Nachum et al. (2017):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, combining this with the de /uniFB01 nition (11b) yields

<!-- formula-not-decoded -->

which together with (22) immediately establishes (24).

- Convergence rate of SPI. The best convergence guarantee is achieved when η /equals ( 1 -γ ) = τ (i.e., the SPI case), where the iteration complexity to reach ‖ Q ? τ -Q ( t ) τ ‖∞ ≤ /epsilon1 reduces to

<!-- formula-not-decoded -->

which is proportional to the effective horizon 1 1 -γ modulo some log factor. This means that the iteration complexity of SPI recovers that of policy iteration (Puterman 2014). Interestingly, the contraction rate in this case (which is γ ) is independent of the choice of the regularization parameter τ . Similarly, the iteration complexity of SPI to reach ‖ log π ? τ -log π ( t + 1 ) ‖∞ ≤ /epsilon1

is again independent of τ .

becomes 1 1 -γ log ( 2 || Q ? τ -Q ( 0 ) τ ||∞ /epsilon1 τ ) , and the contraction rate

- 3.1.1. Comparison With Entropy-Regularized Policy Gradient Methods. Theorem 6 in Mei et al. (2020) proved that the entropy-regularized policy gradient method achieves 4

<!-- formula-not-decoded -->

and they further showed that inf k ≥ 0 min s , a π ( k ) ( a | s ) is nonvanishing in t . It remains unclear, however, how inf t ≥ 0 min s , a π ( t ) ( a | s ) scales with other potentially large salient parameters like (| S | , | A | , 1 1 -γ , 1 τ ) . In truth, existing theory does not rule out the possibility of exponential dependency on these salient parameters. It would thus be of great interest to establish algorithm-dependent lower bounds to uncover the right scaling with these important parameters. In contrast, our convergence guarantees for entropy-regularized NPG methods unveil concrete dependencies on all problem parameters.

## 3.1.2. Computing an e -Optimal Policy for the Original

MDP. Thus far, we have established an intriguing convergence behavior of the entropy-regularized NPG method. However, caution needs to be exercised when interpreting the ef /uniFB01 cacy of this method; the preceding results are concerned with convergence to the optimal regularized value function V ? τ , as opposed to /uniFB01 nding the optimal value function V ? of the original MDP. Fortunately, by choosing the regularization parameter τ to be suf /uniFB01 ciently small (in accordance with the target accuracy level /epsilon1 ), we can guarantee that V ? τ ≈ V ? (cf. (12)), thus ensuring the relevance and applicability of our results for solving the original MDP. To be speci /uniFB01 c, let us adopt the following choice of τ :

<!-- formula-not-decoded -->

and assume the error of the regularized value function satis /uniFB01 es ‖ V ? τ -V ( t ) τ ‖∞ &lt; /epsilon1 = 2. By virtue of Theorem 1, this optimization accuracy can be achieved via no more than 4log | A | ( 1 -γ ) η /epsilon1 log ( 2 C 1 γ /epsilon1 ) iterations of entropyregularized NPG updates with a general learning rate 5 or no more than 1 1 -γ log ( γ ‖ Q ? τ -Q ( 0 ) τ ‖∞ /epsilon1 ) iterations with the

speci /uniFB01 c choice η /equals 1 -γ τ . It then follows that

<!-- formula-not-decoded -->

for any s ∈ S , where we have used our choice of τ in (25). Here, the second inequality arises from (12) as well as the fact that, for any policy π ,

<!-- formula-not-decoded -->

given the elementary entropy bound 0 ≤ H ( s , π ) ≤ 1 = 1 -γ log | A | .

## 3.1.3. Convergence Guarantee for Conservative Policy

Iteration. Our analysis framework also leads to a similar convergence guarantee for a type of policy update adopted in conservative policy iteration (CPI; see Kakade and Langford 2002), where the policy is updated as a convex combination of the previous policy and an improved one. We refer the interested reader to the suppmental material for details.

## 3.2. Approximate Entropy-Regularized NPG Methods

There is no shortage of scenarios where the soft Q-function Q ( t ) τ ( s , a ) is available only in an approximate fashion, e.g., the cases when the value function has to be evaluated using /uniFB01 nite samples. To account for inexactness of policy evaluation, we extend our theory to accommodate the following approximate update rule: for any s ∈ S and any t ≥ 0,

∣ ∣ ∣ Here, δ is some quantity that captures the size of approximation errors. We do not specify the estimator for the soft Q-function (as long as it satis /uniFB01 es the entrywise estimation bound), thus allowing one to plug in both model-based and model-free value function estimators designed for a variety of sampling mechanisms (see, e.g., Azar et al. (2013), Li et al. (2020b)). Encouragingly, the algorithm (26) is robust vis-` a-vis inexactness of value function estimates, as it still converges linearly until an error /uniFB02 oor is hit. This is formalized in the following theorem, with the proof postponed to Section 4.3.

<!-- formula-not-decoded -->

Theorem 2 (Linear Convergence of Approximate Entropy-Regularized NPG). When 0 &lt; η ≤ ( 1 -γ ) = τ , the inexact entropy-regularized NPG updates (26) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ for all t ≥ 0, where C1 is the same as de /uniFB01 ned in (23) and C2 is given by

Apparently, Theorem 2 reduces to Theorem 1 when δ /equals 0. As implied by this theorem, if the ℓ ∞ error of the soft-Q function estimates does not exceed

<!-- formula-not-decoded -->

then the algorithm (26) achieves 2 /epsilon1 -accuracy (i.e., ‖ Q ? τ -Q ( t ) τ ‖∞ ≤ 2 /epsilon1 ) within 1 ητ log C 1 γ /epsilon1 ( ) iterations. In particular, in the case of soft policy iteration (i.e., η /equals 1 -γ τ ), the tolerance level δ can be up to ( 1 -γ ) 2 /epsilon1 2 γ , which matches the theory of approximate policy iteration in Agarwal et al. (2019).

Remark 1. It is straightforward to combine Theorem 2 with known sample complexities for approximate policy evaluation to obtain a crude sample complexity bound. For instance, assuming access to a generative model, Li et al. (2020a) asserts that for any /uniFB01 xed policy π , model-based policy evaluation achieves ‖ ̂ Q π τ -Q π τ ‖∞ ≤ δ with high probability, as long as the number of samples per state-action pair exceeds the order of

<!-- formula-not-decoded -->

up to some logarithmic factor. By employing fresh samples for each policy evaluation, we can set δ /equals( 1 -γ ) 2 /epsilon1 2 γ and invoke the union bound over ˜ O ( 1 1 -γ ) iterations to demonstrate that SPI with model-based policy evaluation needs at most

<!-- formula-not-decoded -->

samples to /uniFB01 nd an /epsilon1 -optimal policy. Here, ˜ O (·) hides any logarithmic factor. We note, however, that the above sample analysis is extremely crude and might be improvable by, say, allowing sample reuses across iterations. It remains an interesting open question as to whether NPG with entropy regularization is minimax optimal with a generative model, where the minimax lower bound is on the order of | S || A | ( 1 -γ ) 3 /epsilon1 2 ( Azar et al.

2013 ) and achievable by model-based plug-in estimators (Agarwal et al. 2020a , Li et al. 2020a ) but not by vanilla Q-learning (Li et al. 2021a ).

## 3.3. Quadratic Convergence in the Smalle Regime

Somewhat remarkably, the regularized NPG method with η /equals 1 -γ τ achieves superlinear convergence in computing V ? τ once the algorithm enters a suf /uniFB01 ciently small local neighborhood surrounding the optimizer.

Before presenting the result, we need to introduce the stationary distribution over S of the MDP M under policy π ? τ , denoted by µ ? τ ∈ ∆ ( S ) . It is straightforward to verify the following basic property

<!-- formula-not-decoded -->

given that the state visitation distribution remains unchanged if the initial state is already in a steady state. Throughout this paper, we assume that min s µ ? τ ( s ) &gt; 0. Our /uniFB01 nding is stated in the following theorem, with the proof deferred to Section 4.4.

Theorem 3 (Quadratic Convergence of Exact Regularized NPG). Suppose that the algorithm (17) with η /equals 1 -γ τ (or SPI) satis /uniFB01 es for all t ≥ 0 , then one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 2. In view of the convergence guarantees in Theorem 2, a suitable initialization of π ( 0 ) and V ( 0 ) τ (such that 4 γ 2 ( 1 -γ ) τ ∣ ∣ ∣ ∣ 1 µ ? τ ∣ ∣ ∣ ∣ ∞ ( V ? τ ( µ ? τ ) -V ( 0 ) τ ( µ ? τ )) &lt; 1) can be obtained by running SPI for suf /uniFB01 ciently many iterations; furthermore, all subsequent iterations are then guaranteed to satisfy (30) according to Theorem 2.

Under the assumptions of Theorem 3, our result indicates that when /epsilon1 is suf /uniFB01 ciently small, the iteration complexity for SPI to yield an /epsilon1 optimization accuracy -that is, V ? τ ( ρ ) -V ( t ) τ ( ρ ) ≤ /epsilon1 -is at most on the order of

This uncovers the faster-than-linear convergence behavior of regularized NPG methods in the high-accuracy regime, accommodating a range of optimization accuracy and all possible choices of the regularization parameter τ . It is worth noting, however, that our quadratic convergence result is stated in terms of the

<!-- formula-not-decoded -->

optimization accuracy (namely, convergence to the soft value function V ? τ ( ρ ) ) as opposed to the accuracy w.r.t. the original unregularized MDP. Thus, interpreting Theorem 3 in practice requires caution, since the approximation error V ? τ ( ρ ) -V ? ( ρ ) might sometimes dominate the optimization error in this regime.

## 4. Analysis

## 4.1. Main Pillars for the Convergence Analysis

Before proceeding, we isolate a few ingredients that provide the main pillars for our theoretical development.

## 4.1.1. Performance Improvement and Monotonicity.

This lemma is a sort of ascent lemma , which quanti /uniFB01 es the progress made over each iteration -measured in terms of the soft value function.

Lemma 1 (Performance Improvement). Suppose that 0 &lt; η ≤ ( 1 -γ ) = τ . For any distribution ρ , one has

<!-- formula-not-decoded -->

## Proof. See the supplemental material. w

In a nutshell, Lemma 1 asserts that each iteration of the entropy-regularized NPG method is guaranteed to improve the estimates of the soft value function, with the improvement depending on the KL divergence between the current policy π ( t ) and the updated one π ( t + 1 ) . In fact, the arbitrary choice of ρ readily reveals a sort of pointwise monotonicity for the above range of learning rates in the sense that V ( t + 1 ) τ ( s ) ≥ V ( t ) τ ( s ) for all s ∈ S . Indeed, this lemma can be viewed as the counterpart of the performance difference lemma in Kakade and Langford (2002) for the unregularized form. Lemma 1 also implies the monotonicity of the soft Q-function in t , since for any ( s , a ) ∈ S × A , one has

<!-- formula-not-decoded -->

where the equalities follow from the de /uniFB01 nition (11a), and the inequality follows since V ( t + 1 ) τ ( s ) ≥ V ( t ) τ ( s ) for all s ∈ S -a consequence of Lemma 1 and the nonnegativity of the KL divergence.

4.1.2. A Key Contraction Operator: The Soft Bellman Optimality Operator. An operator that plays a pivotal role in the theory of dynamic programming (Bellman

1952) is the renowned Bellman optimality operator T : R | S || A | → R | S || A | , de /uniFB01 ned as follows

<!-- formula-not-decoded -->

In order to facilitate analysis for entropyregularized MDPs, we /uniFB01 nd it particularly fruitful to introduce a ' soft ' Bellman optimality operator T τ : R | S || A | → R | S || A | as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which reduces to T when τ /equals 0. To see this, observe that where the last line follows since the optimal policy is exactly the greedy policy w.r.t. Q (Puterman 2014). The operator T τ plays a similar role, as does the Bellman optimality operator for the unregularized case, whose key properties are summarized below. Similar results have been derived in Section 3.1 in Dai et al. (2018).

Lemma 2 (Soft Bellman Optimality Operator). The operator T τ de /uniFB01 ned in (35) satis /uniFB01 es the properties below.

- T τ admits the following closed-form expression:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- The optimal soft Q-function Q ? τ is a /uniFB01 xed point of T τ , namely,

<!-- formula-not-decoded -->

- T τ is a γ -contraction in the ℓ ∞ norm, namely, for any Q1 , Q2 ∈ R | S || A | , one has

<!-- formula-not-decoded -->

## Proof. See the supplemental material. w

For those familiar with dynamic programming, it should become evident that T τ inherits many appealing features of the original Bellman optimality operator T . For example, as an immediate application of the γ -contraction property (38) and the /uniFB01 xed-point property (37), the following soft Q -value iteration is guaranteed to converge linearly to the optimal Q ? τ with a contraction rate γ , a simple observation

<!-- formula-not-decoded -->

consistent with the behavior of value iteration designed for unregularized MDPs.

## 4.2. Analysis of Exact Entropy-Regularized NPG Methods

4.2.1. The SPI Case (i.e. h5 ( 1 2g ) = t ). With the help of the soft Bellman optimality operator, we have

<!-- formula-not-decoded -->

Here, (i) comes from the de /uniFB01 nition (11a) of the soft Q-function, (ii) follows from the relation (11b), (iii) relies on the monotonicity of the soft Q-function (see (33)), and (iv) uses the form of π ( t + 1 ) in (17), whereas (v) makes use of the expression (36). The inequality (39) further leads to 0 ≤ Q ? τ -Q ( t + 1 ) τ ≤ Q ? τ -T τ ( Q ( t + 1 ) τ ) , and hence,

<!-- formula-not-decoded -->

where the /uniFB01 rst equality follows from the /uniFB01 xed-point property (37), and the second inequality is due to the contraction property (38). We have thus established linear convergence of Q ( t ) τ in || · ||∞ for this case.

Turning to the log policies, recall that

<!-- formula-not-decoded -->

where the second relation comes from Equation (12) in Nachum et al. (2017). It then follows from an elementary property of the softmax function that

<!-- formula-not-decoded -->

thus concluding the proof for this case.

4.2.2. The Case With General Learning Rates. We now move to the case with a general learning rate. For the sake of brevity, we shall denote

<!-- formula-not-decoded -->

Additionally, it is helpful to introduce an auxiliary sequence { ξ ( t ) ∈ R | S || A | } constructed recursively by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is easily seen from the construction (42b) that

<!-- formula-not-decoded -->

and consequently,

<!-- formula-not-decoded -->

Step 1: A Linear System that Describes the Error Recursions. In the case with general learning rates, the estimation error ‖ Q ? τ -Q ( t ) τ ‖∞ does not contract in the same form as that of soft policy iteration; instead, it is more succinctly controlled with the aid of an auxiliary quantity ‖ Q ? τ -τ log ξ ( t ) ‖∞ . In what follows, we leverage a simple yet powerful technique by describing the dynamics concerning ‖ Q ? τ -Q ( t ) τ ‖∞ and ‖ Q ? τ -τ log ξ ( t ) ‖∞ via a linear system, whose spectral properties dictate the convergence rate. Toward this, we start with the following key observation, whose proof is deferred to the supplemental material.

Lemma 3. For any learning rate 0 &lt; η ≤ ( 1 -γ ) = τ , the entropy-regularized NPG updates (18) satisfy

<!-- formula-not-decoded -->

where α is de /uniFB01 ned in (41).

If we substitute (43) into (45), it is straightforwardly seen that Lemma 3 is a generalization of the contraction property (40) of soft policy iteration (the case corresponding to α /equals 0). Given that Lemma 3 involves the interaction of more than one quantity, it is convenient to combine (44) and (45) into the following linear system

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We shall make note of the following appealing features of the rank-1 system matrix A :

<!-- formula-not-decoded -->

which relies on the identity ( 1 -α ) γ + α /equals 1 -ητ (according to the de /uniFB01 nition (41) of α ).

Remark 3. By left multiplying both sides of (46) by [ 1 -α , α ] , we obtain

L ( t + 1 ) ≤ ( 1 -ητ ) L ( t ) + γ ( 1 -α ) α t + 1 ‖ Q ( 0 ) τ -τ log ξ ( 0 ) ‖∞ , where L ( t ) : /equals ( 1 -α )‖ Q ? τ -Q ( t ) τ ‖ ∞ + α ‖ Q ? τ -τ log ξ ( t ) ‖∞ can be viewed as a sort of Lyapunov function. This hints at the intimate connection between our proof and the Lyapunov-type analysis used in system theory.

Step 2: Characterizing the Contraction Rate from the Linear System. In view of the recursion Equation (46) and the nonnegativity of ( A , xt , y ) , it is immediate to deduce that

<!-- formula-not-decoded -->

Here, the last line follows from the elementary relation ( α t + 1 I + α t A + /uni22EF + α A t )( α -1 A -I ) /equals A t + 1 -α t + 1 I

and the invertibility of α -1 A -I (since α -1 A is a rank-1 matrix whose nonzero singular value is larger than 1). In addition, the Woodbury matrix inversion formula together with the decomposition (48) yields

        (50) which is a nonnegative vector. Consequently, this taken together with (49) gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third line follows from (48), (50), and the de /uniFB01 nition of xt . Furthermore, observe that

<!-- formula-not-decoded -->

where the inequality comes from the triangle inequality, and the last identity follows from (42a). Substituting this back into (51), we obtain

<!-- formula-not-decoded -->

To /uniFB01 nish up, recall that π ( t ) is related to ξ ( t ) as follows

<!-- formula-not-decoded -->

which can be seen by comparing (42) with (18). Therefore, invoking the elementary property of the softmax function, we arrive at

<!-- formula-not-decoded -->

This combined with (53) as well as the de /uniFB01 nition (47) of xt + 1 immediately establishes Theorem 1.

## 4.3. Analysis of Approximate EntropyRegularized NPG Methods

Wenowturn to the convergence properties of approximate entropy-regularized NPG methods -as claimed in Theorem 2 -when only inexact policy evaluation Q ( t ) τ is available (in the sense of (26)).

̂ Step 1: Performance Difference Accounting for Inexact Policy Evaluation. We /uniFB01 rst bound the quality of the policy updates (26) by examining the difference between V ( t + 1 ) τ and V ( t ) τ and how it is impacted by the imperfectness of policy evaluation. This is made precise by the following lemma.

Lemma 4 (Performance Difference of Approximate Entropy-Regularized NPG). Suppose that 0 &lt; η ≤ ( 1 -γ ) = τ . For any state s0 ∈ S , one has

<!-- formula-not-decoded -->

The careful reader might already realize that the above lemma is a relaxation of Lemma 1; in particular, the last term of (55) quanti /uniFB01 es the effect of the approximation error (i.e., the difference between ̂ Q ( t ) τ and Q ( t ) τ ) upon performance improvement. Under the assumption ∣ ∣ ∣ ∣ ∣ ∣ ̂ Q ( t ) τ -Q ( t ) τ ∣ ∣ ∣ ∣ ∣ ∣ ∞ ≤ δ , repeating the argument of (33) reveals that the soft Q -function estimates are not far from being monotone in t in the sense that

<!-- formula-not-decoded -->

Step 2: A Linear System Accounting for Inexact Policy Evaluation. With the assistance of (56), it is possible to construct a linear system -similar to the one built in Section 4.2 -that takes into account inexact policy evaluation. Toward this end, we adopt a similar approach as in (42) by introducing the following auxiliary sequence ̂ ξ ( t ) de /uniFB01 ned recursively using Q ( t ) τ :

<!-- formula-not-decoded -->

where α : /equals 1 -ητ 1 -γ as before.

We claim that the following linear system tracks the error dynamics of the policy updates:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Here, the system matrix B (in particular its eigenvalues) governs the contraction rate, whereas the term b captures the error introduced by inexact policy evaluation. Theorem 2 then follows by carrying out a similar analysis argument as in Section 4.2 to characterize the error dynamics. Details are postponed to the supplemental material.

## 4.4. Analysis of Local Quadratic Convergence

We now sketch the proof of Theorem 3, which establishes local quadratic convergence of SPI.

Step 1: Characterization of the Suboptimality Gap. Lemma 1 bounds the performance improvement of SPI by the KL divergence between the current policy π ( t ) and the updated policy π ( t + 1 ) . Interestingly, the type of KL divergence can be further employed to bound the suboptimality gap for each iteration.

Lemma 5 (Suboptimality Gap). Suppose that η /equals ( 1 -γ ) = τ . For any distribution ρ , one has

<!-- formula-not-decoded -->

Proof. This result has appeared in Eqn. (486) of Mei et al. (2020). For completeness, we include a proof in the supplemental material. w

In words, Lemma 5 formalizes the connection between the suboptimality gap (w.r.t. the optimal soft value function) and the proximity of the two consecutive policy iterates. As re /uniFB02 ected by this lemma, if the current and the updated policies do not differ by much (which indicates that the algorithm might be close to convergence), then the current estimate of the soft value function is close to optimal.

Step 2: A Contraction Property. The importance of the above two lemmas is made apparent by the following contraction property when η /equals ( 1 -γ ) = τ :

<!-- formula-not-decoded -->

Step 3: Superlinear Convergence in the Smalle Regime. The contraction property (60) implies that V ( t + 1 ) τ ( ρ ) converges superlinearly to V ? τ once π ( t ) gets suf /uniFB01 ciently close to π ? τ . In fact, once the ratio d ( t + 1 ) ρ = d π ? τ ρ becomes suf /uniFB01 ciently close to 1, the contraction factor 1 -∣ ∣ ∣ ∣ d π ? τ ρ = d ( t + 1 ) ρ ∣ ∣ ∣ ∣ -1 ∞ in (60) is approaching 0, thereby accelerating convergence. This observation underlies Theorem 3, whose complete analysis is postponed until the supplemental material.

Here, (i) arises from Lemma 1, and (ii) employs the prefactor ∣ ∣ ∣ ∣ d π ? τ ρ = d ( t + 1 ) ρ ∣ ∣ ∣ ∣ -1 ∞ to accommodate the change of distributions, whereas (iii) follows from Lemma 5.

## 5. Discussions

This paper establishes nonasymptotic convergence of entropy-regularized natural policy gradient methods, providing theoretical footings for the role of entropy regularization in guaranteeing fast convergence. Our analysis opens up several directions for future research; we close the paper by sampling a few of them.

- Extended analysis of policy gradient methods with inexact gradients. It would be of interest to see whether our
- analysis framework can be applied to improve the theory of policy gradient methods (Mei et al. 2020) to accommodate the case with inexact policy gradients.
- Finite-sample analysis in the presence of sample-based policy evaluation. Another natural extension is toward understanding the sample complexity of entropyregularized NPG methods when the value functions are estimated using rollout trajectories (see, e.g., Kakade and Langford 2002, Shani et al. 2019, and Agarwal et al. 2020b) or bootstrapping (see, e.g., Haarnoja et al. 2018, Wuetal. 2020, and Xu et al. 2020).
- Function approximation. The current work has been limited to the tabular setting. It would certainly be interesting and fundamentally important to understand entropy-regularized NPG methods in conjunction with function approximation; see Agarwal et al. 2019, 2020b, and Sutton et al. 2000) for a few representative scenarios.
- Beyond softmax parameterization. The current paper has been devoted to softmax parameterization, which enables a concise and NPG update rule. A couple of other parameterization schemes have been proposed for (vanilla) PG methods as well (Agarwal et al. 2019, 2020b; Bhandari and Russo 2019, 2020), e.g. vanilla parameterization (paired with proper projection onto the probability simplex in each iteration), log-linear parameterization, and neural softmax parameterization. Unfortunately, the analysis in our paper relies heavily on the softmax NPG update rule and does not immediately extend to other parameterization. It would be of great importance to establish convergence guarantees that accommodate other parameterizations of practical interest.

## Endnotes

1 For the sake of simplicity, we assume throughout that the reward resides within [ 0, 1 ] . Our results can be generalized in a straightforward manner to other ranges of bounded rewards.

- 2 In this paper, we use the terms ' regularized ' value (resp. Q) functions and ' soft ' value (resp. Q) functions interchangeably.

3 To see this, invoke the optimality of π ? τ and the elementary entropy bound 0 ≤ H ( ρ , π ) ≤ 1 1 -γ log | A | to obtain

<!-- formula-not-decoded -->

- 4 Here, we have assumed that the exact policy gradient is computed with respect to V ( t ) τ ( ρ ) .
- 5 This result is in fact better than the iteration complexity 2 ( 1 -γ ) 2 /epsilon1 of the unregularized NPG method established in Agarwal et al. (2020b) as soon as η ≥ 2 ( 1 -γ ) log | A | log 2 C 1 γ /epsilon1 ( ) . Consequently, our finding hints at the potential advantage of entropy-regularized NPG methods over the unregularized counterpart even when solving the original MDP.

## References

Agarwal A, Jiang N, Kakade SM (2019) Reinforcement Learn.: Theory and algorithms. Technical report, University of Washington Seattle, Seattle, WA.

Agarwal A, Kakade S, Yang LF (2020a) Model-based reinforcement Learn. with a generative model is minimax optimal. Proc. 33rd Conf. Learn. Theory , 67 -83.

- Agarwal A, Kakade SM, Lee JD, Mahajan G (2020b) Optimality and approximation with policy gradient methods in Markov decision processes. Proc. 33rd Conf. Learn. Theory, 64 -66.
- Ahmed Z, Le Roux N, Norouzi M, Schuurmans D (2019) Understanding the impact of entropy on policy optimization. Proc. 36th Internat. Conf. Machine Learn. , 151 -160.
- Amari SI (1998) Natural gradient works ef /uniFB01 ciently in Learn.. Neural Comput. 10(2):251 -276.
- Azar MG, Munos R, Kappen HJ (2013) Minimax PAC bounds on the sample complexity of reinforcement Learn. with a generative model. Machine Learn. 91(3):325 -349.
- Bellman R (1952) On the theory of dynamic programming. Proc. Natl. Acad. Sci. USA 38(8):716.
- Bertsekas DP (2017) Dynamic Programming and Optimal Control , 4th ed. (Athena Scienti /uniFB01 c, Belmont, MA).
- Bhandari J, Russo D (2019) Global optimality guarantees for policy gradient methods. Preprint, submitted June 5, https:/ /arxiv. org/abs/1906.01786.
- Bhandari J, Russo D (2020) A note on the linear convergence of policy gradient methods. Preprint, submitted July 21, https:/ /arxiv. org/abs/2007.11120.
- Bhatnagar S, Sutton RS, Ghavamzadeh M, Lee M (2009) Natural actor-critic algorithms. Automatica J. IFAC . 45(11):2471 -2482.
- Cai Q, Yang Z, Jin C, Wang Z (2019) Provably ef /uniFB01 cient exploration in policy optimization. Proc. 37th Conf. Machine Learn. , PMLR 119:1283 -1294.
- Cen S, Wei Y, Chi Y (2021) Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization. arXiv preprint arXiv:2105.15186.
- Dai B, Shaw A, Li L, Xiao L, He N, Liu Z, Chen J, Song L (2018) SBEED: Convergent reinforcement Learn. with nonlinear function approximation. Proc. 35th Internat. Conf. Machine Learn. , PMLR, 80:1125 -1134.
- Duan Y, Chen X, Houthooft R, Schulman J, Abbeel P (2016) Benchmarking deep reinforcement Learn. for continuous control. Proc. 33rd Internat. Conf. Machine Learn. , PMLR, 48:1329 -1338.
- Even-Dar E, Kakade SM, Mansour Y (2009) Online Markov decision processes. Math. Oper. Res. 34(3):726 -736.
- Fazel M, Ge R, Kakade S, Mesbahi M (2018) Global convergence of policy gradient methods for the linear quadratic regulator. Proc. 35th Internat. Conf. Machine Learn. , PMLR, 80:1467 -1476.
- Geist M, Scherrer B, Pietquin O (2019) A theory of regularized Markov decision processes. Internat. Conf. Machine Learn., 2160 -2169.
- Grill JB, Darwiche Domingues O, Menard P, Munos R, Valko M (2019) Planning in entropy-regularized markov decision processes and games. Advances in Neural Information Processing Systems 32:12404 -12413.
- Haarnoja T, Tang H, Abbeel P, Levine S (2017) Reinforcement Learn. with deep energy-based policies. Proc. 34th Internat. Conf. Machine Learn., PMLR, 70:1352 -1361.
- Haarnoja T, Zhou A, Abbeel P, Levine S (2018) Soft actor-critic: offpolicy maximum entropy deep reinforcement Learn. with a stochastic actor. Proc. 35th Internat. Conf. Machine Learn. , PMLR, 80:1861 -1870.
- Hazan E, Kakade S, Singh K, Van Soest A (2019) Provably ef /uniFB01 cient maximum entropy exploration. Proc. 36th Internat. Conf. Machine Learn. , PMLR, 97:2681 -2691.
- Jansch-Porto JP, Hu B, Dullerud G (2020) Convergence guarantees of policy optimization methods for Markovian jump linear systems. arXiv preprint arXiv:2002.04090 .
- Kakade SM (2002) A natural policy gradient. Advances in Neural Information Processing Systems (MIT Press, Cambridge, MA), 1531 -1538.
- Kakade S, Langford J (2002) Approximately optimal approximate reinforcement Learn. Proc. 19th Internat. Conf. Machine Learn., 267 -274.
- Karimi B, Miasojedow B, Moulines ´ E, Wai HT (2019) Nonasymptotic analysis of biased stochastic approximation scheme. Proc. Thirty-Second Conf. Learn. Theory, 99:1944 -1974.
- Konda VR, Tsitsiklis JN (2000) Actor-critic algorithms. Advances in Neural Information Processing Systems (MIT Press, Cambridge, MA) 1008 -1014.
- Li G, Wei Y, Chi Y, Gu Y, Chen Y (2020a) Breaking the sample size barrier in model-based reinforcement Learn. with a generative model. 34th Conf. Neural Information Processing Systems (NeurIPS 2020) , Vancouver, BC, Canada, 12861 -12872.
- Li G, Wei Y, Chi Y, Gu Y, Chen Y (2020b) Sample complexity of asynchronous Q-Learn.: Sharper analysis and variance reduction. Preprint, submitted Jun 4, https:/ /arxiv.org/abs/2006. 03041.
- Li G, Wei Y, Chi Y, Gu Y, Chen Y (2021b) Softmax policy gradient methods can take exponential time to converge. Belkin M, Kpotufe S, eds. Proc. 34th Conf. Learning Theory (PMLR), 134:3107 -3110.
- Li G, Cai C, Chen Y, Gu Y, Wei Y, Chi Y (2021a) Is Q-Learn. minimax optimal? a tight sample complexity analysis. arXiv preprint arXiv:2102.06548 .
- Liu B, Cai Q, Yang Z, Wang Z (2019). Neural proximal trust region policy optimization attains globally optimal policy. 33rd Conf. Neural Information Processing Systems (NeurIPS 2019) , Vancouver, BC, Canada.
- Mei J, Xiao C, Szepesvari C, Schuurmans D, (2020) On the global convergence rates of softmaz policy gradient methods. Proc. 37th Internat. Conf. Machine Learn. , PMLR, 119:6820 -6829.
- Mnih V, Badia AP, Mirza M, Graves A, Lillicrap T, Harley T, Silver D, Kavukcuoglu K (2016). Asynchronous methods for deep reinforcement Learn. Proc. 33rd Internat. Conf. Machine Learn. , PMLR, 48:1928 -1937.
- Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, Graves A, et al (2015) Human-level control through deep reinforcement Learn. Nature 518(7540):529 -533.
- Mohammadi H, Zare A, Soltanolkotabi M, Jovanovi ´ c MR (2019). Convergence and sample complexity of gradient methods for the model-free linear quadratic regulator problem. Preprint, submitted December 26, https:/ /arxiv.org/abs/1912.11899.
- Nachum O, Norouzi M, Xu K, Schuurmans D (2017). Bridging the gap between value and policy based reinforcement Learn.. Preprint, submitted November 22, https:/ /arxiv.org/abs/1702. 08892.
- Nemirovsky AS, Yudin DB (1983) Problem complexity and method ef /uniFB01 ciency in optimization. (J. Wiley &amp; Sons).
- Nesterov Y (2009) Primal-dual subgradient methods for convex problems. Math. Programming 120(1):221 -259.
- Neu G, Jonsson A, G ´ omez V (2017). A uni /uniFB01 ed view of entropyregularized Markov decision processes. Preprint, submitted May 22, https:/ /arxiv.org/abs/1705.07798.
- Peters J, Schaal S (2008) Natural actor-critic. Neurocomputing 71(7-9): 1180 -1190.
- Peters J, Mulling K, Altun Y (2010) Relative entropy policy search. Proc. AAAI Conf. Arti /uniFB01 cial Intelligence, 24(1):1607 -1612.
- Puterman ML (2014) Markov Decision Processes: Discrete Stochastic Dynamic Programming . (John Wiley &amp; Sons, Hoboken, NJ).
- Schulman J, Chen X, Abbeel P (2017a). Equivalence between policy gradients and soft Q-Learn. Preprint, submitted April 21, https:/ /arxiv.org/abs/1704.06440.
- Schulman J, Levine S, Abbeel P, Jordan M, Moritz P (2015) Trust region policy optimization. Proc. 32nd Conf. Machine Learn. , PMLR, 37:1889 -1897.
- Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O (2017b) Proximal policy optimization algorithms. Preprint, submitted July 20, https:/ /arxiv.org/abs/1707.06347.

Downloaded from informs.org by [73.183.48.82] on 16 July 2022, at 07:53 . For personal use only, all rights reserved.

Shani L, Efroni Y, Mannor S (2019) Adaptive trust region policy optimization: global convergence and faster rates for regularized MDPs. Proc. AAAI Conf. Arti /uniFB01 cial Intelligence 34(4):5668 -5675.

Silver D, Huang A, Maddison CJ, Guez A, Sifre L, Van Den Driessche G, Schrittwieser J, Antonoglou I, Panneershelvam V, Lanctot M, et al. (2016) Mastering the game of Go with deep neural networks and tree search. Nature 529(7587):484 -489.

Sutton RS, McAllester DA, Singh SP, Mansour Y (2000) Policy gradient methods for reinforcement Learn. with function approximation. NIPS ' 99, 1057 -1063.

Tu S, Recht B (2019) The gap between model-based and model-free methods on the linear quadratic regulator: an asymptotic viewpoint. Proc. Thirty-Second Conf. Learn. Theory , PMLR, 99: 3036 -3083.

Vieillard N, Kozuno T, Scherrer B, Pietquin O, Munos R, Geist M (2020) Leverage the average: an analysis of KL regularization in RL. Preprint, submitted March 31, https:/ /arxiv.org/abs/ 2003.14089.

Wang L, Cai Q, Yang Z, Wang Z (2019) Neural policy gradient methods: Global optimality and rates of convergence. Preprint, submitted August 29, https:/ /arxiv.org/abs/1909.01150.

Williams RJ (1992) Simple statistical gradient-following algorithms for connectionist reinforcement Learn. Machine Learn . 8(3-4): 229 -256.

Williams RJ, Peng J (1991) Function optimization using connectionist reinforcement Learn. algorithms. Connect. Sci. 3(3):241 -268.

Wu Y, Zhang W, Xu P, Gu Q (2020) A /uniFB01 nite time analysis of two time-scale actor critic methods. Preprint, submitted May 4, https:/ /arxiv.org/abs/2005.01350.

Xiao C, Huang R, Mei J, Schuurmans D, M ¨ uller M (2019) Maximum entropy Monte-Carlo planning. Advances in Neural Information Processing Systems (Curran Associates, Inc., Red Hook, NY), 9520 -9528.

Xu T, Wang Z, Liang Y (2020) Non-asymptotic convergence analysis of two time-scale (natural) actor-critic algorithms. Preprint, submitted May 7, https:/ /arxiv.org/abs/2005.03557.

Zhang K, Hu B, Basar T (2019a) Policy optimization for h 2 linear control with h ∞ robustness guarantee: implicit regularization and global convergence. Proc. 2nd Conf. Learn. Dynam. Control , PMLR, 120:179 -190.

Zhang K, Koppel A, Zhu H, Bas ¸ar T (2019b). Global convergence of policy gradient methods to (almost) locally optimal policies. SIAM J. Control Optim. 58(6):3586 -3612.

Shicong Cen is a second-year PhD student in the Department of Electrical and Computer Engineering of Carnegie

Mellon University, advised by Professor Yuejie Chi. He received his bachelor ' s degree from School of Mathematical Sciences, Peking University. His research interests lie in the theoretical foundations of optimization methods in machine learning and reinforcement learning.

Chen Cheng is a second-year PhD student in the Department of Statistics at Stanford University, jointly advised by Professor John Duchi and Professor Andrea Montanari. He received his bachelor ' s degree from the School of Mathematical Sciences, Peking University. His research interests lie in statistical theory and algorithms for high-dimensional data, random matrix theory, and reinforcement learning.

Yuxin Chen is currently an assistant professor of electrical and computer engineering at Princeton University. His research interests include high-dimensional statistics, mathematical optimization, and reinforcement learning. He has received the AFOSR and ARO Young Investigator Awards, the Princeton graduate mentoring award, and the 2020 ICCM best paper award (gold medal), and was selected as a finalist for the Best Paper Prize for Young Researchers in Continuous Optimization, 2019.

Yuting Wei is currently an assistant professor in the Statistics and Data Science Department at the Wharton School, University of Pennsylvania. Prior to this, she spent two years at Carnegie Mellon University as an assistant professor, and one year at Stanford University as a Stein Fellow. She obtained her PhD in statistics at University of California at Berkeley, receiving the 2018 Erich L. Lehmann Citation for her PhD dissertation. Her research interests include high-dimensional statistics, machine learning, and reinforcement learning.

Yuejie Chi is a professor in the Department of Electrical and Computer Engineering at Carnegie Mellon University. Her research interests lie in the theoretical and algorithmic foundations of data science, signal processing, machine learning and inverse problems. Among others, Dr. Chi received the Presidential Early Career Award for Scientists and Engineers (PECASE), and the inaugural IEEE Signal Processing Society Early Career Technical Achievement Award for contributions to high-dimensional structured signal processing.