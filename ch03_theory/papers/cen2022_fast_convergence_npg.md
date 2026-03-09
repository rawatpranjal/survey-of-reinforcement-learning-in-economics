## Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization

Shicong Cen ∗ CMU

Chen Cheng † Stanford

Yuxin Chen ‡ Princeton

Yuting Wei § CMU

July 13, 2020;

Revised April 12, 2021

## Abstract

Natural policy gradient (NPG) methods are among the most widely used policy optimization algorithms in contemporary reinforcement learning. This class of methods is often applied in conjunction with entropy regularization - an algorithmic scheme that encourages exploration - and is closely related to soft policy iteration and trust region policy optimization. Despite the empirical success, the theoretical underpinnings for NPG methods remain limited even for the tabular setting.

This paper develops non-asymptotic convergence guarantees for entropy-regularized NPG methods under softmax parameterization, focusing on discounted Markov decision processes (MDPs). Assuming access to exact policy evaluation, we demonstrate that the algorithm converges linearly - even quadratically once it enters a local region around the optimal policy - when computing optimal value functions of the regularized MDP. Moreover, the algorithm is provably stable vis-` a-vis inexactness of policy evaluation. Our convergence results accommodate a wide range of learning rates, and shed light upon the role of entropy regularization in enabling fast convergence.

Keywords: natural policy gradient methods, entropy regularization, global convergence, soft policy iteration, conservative policy iteration, trust region policy optimization

## Contents

| 1 Introduction   | 1 Introduction       | 1 Introduction                                                   |   2 |
|------------------|----------------------|------------------------------------------------------------------|-----|
|                  | 1.1                  | Background and motivation . . . . . . . . . . . . . . .          |   3 |
|                  | 1.2                  | This paper . . . . . . . . . . . . . . . . . . . . . . . .       |   3 |
|                  | 1.3                  | Other related works . . . . . . . . . . . . . . . . . . .        |   5 |
|                  | 1.4                  | Notation . . . . . . . . . . . . . . . . . . . . . . . . . .     |   6 |
| 2                | Model and algorithms | Model and algorithms                                             |   6 |
|                  | 2.1                  | Problem settings . . . . . . . . . . . . . . . . . . . . .       |   6 |
|                  | 2.2                  | Algorithm: NPG methods with entropy regularization               |   8 |
|                  | 2.3                  | A warm-up example: the bandit case . . . . . . . . . .           |   9 |
| 3                | Main results         | Main results                                                     |  10 |
|                  | 3.1                  | Exact entropy-regularized NPG methods . . . . . . . .            |  10 |
|                  | 3.2                  | Approximate entropy-regularized NPG methods . . . .              |  12 |
|                  | 3.3                  | Quadratic convergence in the small- /epsilon1 regime . . . . . . |  13 |

∗ Department of Electrical and Computer Engineering, Carnegie Mellon University; email: shicongc@andrew.cmu.edu .

† Department of Statistics, Stanford University; email: chencheng@stanford.edu .

‡ Department of Electrical and Computer Engineering, Princeton University; email: yuxin.chen@princeton.edu .

§ Department of Statistics and Data Science, Carnegie Mellon University; email: ytwei@cmu.edu .

¶ Department of Electrical and Computer Engineering, Carnegie Mellon University; email: yuejiechi@cmu.edu .

Yuejie Chi ¶ CMU

| 4 Analysis   | 4 Analysis                                                   | 4 Analysis                                                     |   14 |
|--------------|--------------------------------------------------------------|----------------------------------------------------------------|------|
|              | 4.1                                                          | Main pillars for the convergence analysis . . . . . . . . . .  |   14 |
|              | 4.2                                                          | Analysis of exact entropy-regularized NPG methods . . .        |   15 |
|              |                                                              | 4.2.1 The SPI case (i.e. η = (1 - γ ) /τ ) . . . . . . . . . . |   15 |
|              |                                                              | 4.2.2 The case with general learning rates . . . . . . . .     |   16 |
|              | 4.3                                                          | Analysis of approximate entropy-regularized NPG methods        |   18 |
|              | 4.4                                                          | Analysis of local quadratic convergence . . . . . . . . . . .  |   19 |
| 5            | Discussions                                                  | Discussions                                                    |   20 |
| A            | Preliminaries                                                | Preliminaries                                                  |   24 |
|              | A.1 Derivation of entropy-regularized NPG methods .          | . . . . .                                                      |   24 |
|              | A.2 Basic facts about the function log( ‖ exp( θ ) ‖ 1 ) . . | . . . . .                                                      |   24 |
| B            | Proof for the bandit case (Proposition 1)                    | Proof for the bandit case (Proposition 1)                      |   25 |
| C            | Proof for key lemmas                                         | Proof for key lemmas                                           |   25 |
| C.1          | Proof of Lemma 1                                             | . . . . . . . . . . . . . . . . . . . . . .                    |   25 |
| C.2          | Proof of Lemma 2 . . . . . . . . . . . .                     | . . . . . . . . . .                                            |   27 |
| C.3          | Proof of Lemma 3                                             | . . . . . . . . . . . . . . . . . . . . . .                    |   28 |
| C.4          | Proof of Lemma 4                                             | . . . . . . . . . . . . . . . . . . . . . .                    |   29 |
| C.5          | Proof of Lemma 5                                             | . . . . . . . . . . . . . . . . . . . . . .                    |   30 |
| C.6          | Proof of Lemma 6                                             | . . . . . . . . . . . . . . . . . . . . . .                    |   31 |
| D            | Convergence guarantees for CPI-style policy updates          | Convergence guarantees for CPI-style policy updates            |   33 |
|              | D.1 Proof of Lemma 7 . .                                     | . . . . . . . . . . . . . . . . . . . .                        |   34 |
| E            | Proof for approximate entropy-regularized NPG (Theorem 2)    | Proof for approximate entropy-regularized NPG (Theorem 2)      |   35 |
| F            | Proof for local quadratic convergence                        | (Theorem 3)                                                    |   38 |

F.1

Proof of Lemma 8

## 1 Introduction

Policy gradient (PG) methods and their variants (Williams, 1992; Sutton et al., 2000; Kakade, 2002; Peters and Schaal, 2008; Konda and Tsitsiklis, 2000), which aim to optimize (parameterized) policies via gradient-type methods, lie at the heart of recent advances in reinforcement learning (RL) (e.g. Mnih et al. (2015); Schulman et al. (2015); Silver et al. (2016); Schulman et al. (2017b)). Perhaps most appealing is their flexibility in adopting various kinds of policy parameterizations (e.g. a class of policies parameterized via deep neural networks), which makes them remarkably powerful and versatile in contemporary RL.

As an important and widely used extension of PG methods, natural policy gradient (NPG) methods propose to employ natural policy gradients (Amari, 1998) as search directions, in order to achieve faster convergence than the update rules based on policy gradients (Kakade, 2002; Peters and Schaal, 2008; Bhatnagar et al., 2009; Even-Dar et al., 2009). Informally speaking, NPG methods precondition the gradient directions by Fisher information matrices (which are the Hessians of a certain divergence metric), and fall under the category of quasi second-order policy optimization methods. In fact, a variety of mainstream RL algorithms, such as trust region policy optimization (TRPO) (Schulman et al., 2015) and proximal policy optimization (PPO) (Schulman et al., 2017b), can be viewed as generalizations of NPG methods (Shani et al., 2019). In this paper, we pursue in-depth theoretical understanding about this popular class of methods in conjunction with entropy regularization to be introduced momentarily.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

38

## 1.1 Background and motivation

Despite the enormous empirical success, the theoretical underpinnings of policy gradient type methods have been limited even until recently, primarily due to the intrinsic non-concavity underlying the value maximization problem of interest (Bhandari and Russo, 2019; Agarwal et al., 2020b). To further exacerbate the situation, an abundance of problem instances contain suboptimal policies residing in regions with flat curvatures (namely, vanishingly small gradients and high-order derivatives) (Agarwal et al., 2020b). Such plateaus in the optimization landscape could, in principle, be difficult to escape once entered, thereby necessitating a higher degree of exploration in order to accelerate policy optimization.

In practice, a strategy that has been frequently adopted to encourage exploration and improve convergence is to enforce entropy regularization (Williams and Peng, 1991; Peters et al., 2010; Mnih et al., 2016; Duan et al., 2016; Haarnoja et al., 2017; Hazan et al., 2019; Vieillard et al., 2020; Xiao et al., 2019). By inserting an additional penalty term to the objective function, this strategy penalizes policies that are not stochastic/exploratory enough, in the hope of preventing a policy optimization algorithm from being trapped in an undesired local region. Through empirical visualization, Ahmed et al. (2019) suggested that entropy regularization induces a smoother landscape that allows for the use of larger learning rates, and hence, faster convergence. However, the theoretical support for regularization-based policy optimization remains highly inadequate.

Motivated by this, a very recent line of works set out to elucidate, in a theoretically sound manner, the efficiency of entropy-regularized policy gradient methods. Assuming access to exact policy gradients, Agarwal et al. (2020b) and Mei et al. (2020) developed convergence guarantees for regularized PG methods (with relative entropy regularization considered in Agarwal et al. (2020b) and entropy regularization in Mei et al. (2020)). Encouragingly, both papers suggested the positive role of regularization in guaranteeing faster convergence for the tabular setting. However, these works fell short of explaining the role of entropy regularization for other policy optimization algorithms like NPG methods, which we seek to understand in this paper.

## 1.2 This paper

Inspired by recent theoretical progress towards understanding PG methods (Agarwal et al., 2020b; Bhandari and Russo, 2019; Mei et al., 2020), we aim to develop non-asymptotic convergence guarantees for entropy-regularized NPG methods in conjunction with softmax parameterization. We focus attention on studying tabular discounted Markov decision processes (MDPs), which is an important first step and a stepping stone towards demystifying the effectiveness of entropy-regularized policy optimization in more complex settings.

Settings. Consider a γ -discounted infinite-horizon MDP with state space S and action space A . Assuming availability of exact policy evaluation, the update rule of entropy-regularized NPG methods with softmax parameterization admits a simple update rule in the policy space (see Section 2 for precise descriptions)

<!-- formula-not-decoded -->

for any ( s, a ) ∈ S × A , where τ &gt; 0 is the regularization parameter, 0 &lt; η ≤ 1 -γ τ is the learning rate (or stepsize), π ( t ) indicates the t -th policy iterate, and Q π τ is the soft Q-function under policy π (to be defined in (11a)). The update rule (1) is closely connected to several popular algorithms in practice. For instance, the trust region policy optimization (TRPO) algorithm (Schulman et al., 2015), when instantiated in the tabular setting, can be viewed as implementing (1) with line search. In addition, by setting the learning rate as η = 1 -γ τ , the update rule (1) coincides with soft policy iteration (SPI) studied in Haarnoja et al. (2017).

Our contributions. The results of this paper deliver fully non-asymptotic convergence rates of entropyregularized NPG methods without any hidden constants, which are previewed as follows (in an orderwise manner). The definition of /epsilon1 -optimality can be found in Table 1.

- Linear convergence of exact entropy-regularized NPG methods. We establish linear convergence of entropy-regularized NPG methods for finding the optimal policy of the entropy-regularized

MDP, assuming access to exact policy evaluation. To yield an /epsilon1 -optimal policy for the regularized MDP (cf. Table 1), the algorithm (1) with a general learning rate 0 &lt; η ≤ 1 -γ τ needs no more than an order of

<!-- formula-not-decoded -->

iterations, where we hide the dependencies that are logarithmic on salient problem parameters (see Theorem 1). Some highlights of our convergence results are (i) their near dimension-free feature and (ii) their applicability to a wide range of learning rates (including small learning rates).

- Linear convergence of approximate entropy-regularized NPG methods. We demonstrate the stability of the regularized NPG method with a general learning rate 0 &lt; η ≤ 1 -γ τ even when the soft Q-functions of interest are only available approximately. This paves the way for future investigations that involve finite-sample analysis. Informally speaking, the algorithm exhibits the same convergence behavior as in the exact gradient case before an error floor is hit, where the error floor scales linearly in the entrywise error of the soft Q-function estimates (see Theorem 2).
- Quadratic convergence in the small/epsilon1 regime. In the high-accuracy regime where the target level /epsilon1 is very small , the algorithm (1) with η = 1 -γ τ converges super-linearly, in the sense that the iteration complexity to reach /epsilon1 -accuracy for the regularized MDP is at most on the order of

<!-- formula-not-decoded -->

after entering a small local neighborhood surrounding the optimal policy. Here, we again hide the dependencies that are logarithmic on salient problem parameters (see Theorem 3).

Comparisons with prior art. Agarwal et al. (2020b) proved that unregularized NPG methods with softmax parameterization attain an /epsilon1 -accuracy within O (1 //epsilon1 ) iterations. In contrast, our results assert that O (log(1 //epsilon1 )) iterations suffice with the assistance of entropy regularization, which hints at the potential benefit of entropy regularization in accelerating the convergence of NPG methods. Shortly after the initial posting of our paper, Bhandari and Russo (2020) posted a note that proves linear convergence of unregularized NPG methods with exact line search, by exploiting a clever connection to policy iteration. Their convergence rate is governed by a quantity min s ∈S ρ ( s ), resulting in an iteration complexity at least |S| times larger than ours. In comparison, our results cover a broad range of fixed learning rates (including small stepsizes that are of particular interest in practice), and accommodate the scenario with inexact gradient evaluation. See Table 1 for a quantitative comparison. Moreover, we note that the entropy-regularized NPG method with general learning rates is closely related to TRPO in the tabular setting (see Shani et al. (2019)). The recent work Shani et al. (2019) demonstrated that TRPO converges with an iteration complexity O (1 //epsilon1 ) in entropyregularized MDPs. The analysis therein is inspired by the mirror descent theory in generic optimization literature, which characterizes sublinear convergence under properly decaying stepsizes and accommodates various choices of divergence metrics. In comparison, our analysis strengthens the performance guarantees by carefully exploiting properties specific to the current version of the NPG method. In particular, we identify the delicate interplay between the crucial operational quantities Q /star τ -Q ( t ) τ and Q /star τ -τ log ξ ( t ) (to be defined later), and invoke the linear system theory to establish appealing contraction, which allow for the use of more aggressive constant stepsizes and hence improved convergence.

It is also helpful to compare our results with the state-of-the-art theory for PG methods with softmax parameterization (Agarwal et al., 2020b; Mei et al., 2020). Specifically, Agarwal et al. (2020b) established the asymptotic convergence of unregularized PG methods with softmax parameterization, while an iteration complexity of O (1 //epsilon1 ) was recently pinned down by Mei et al. (2020). In the presence of entropy regularization, Agarwal et al. (2020b) showed that PG with relative entropy regularization and softmax parameterization enjoys an iteration complexity of O (1 //epsilon1 2 ), while Mei et al. (2020) showed that the entropy-regularized softmax PG method converges linearly in O (log(1 //epsilon1 )) iterations. However, the dependencies of the iteration complexity in Mei et al. (2020) on other salient parameters like |S| , |A| and 1 1 -γ are not fully specified. Very recently, Li et al. (2021b) delivered a negative message demonstrating that these dependencies can be highly pessimistic; in fact, one can find an MDP instance which takes softmax PG methods (super)-exponential

| paper                     | iteration complexity                           | regularization   | learning rates        |
|---------------------------|------------------------------------------------|------------------|-----------------------|
| Agarwal et al. (2020b)    | 2 (1 - γ ) 2 /epsilon1 + 2 η/epsilon1          | unregularized    | constant: (0 , ∞ )    |
| Bhandari and Russo (2020) | 1 (1 - γ )min s ∈S ρ ( s ) log ( 1 /epsilon1 ) | unregularized    | exact line search     |
| this work                 | 1 1 - γ log ( 1 /epsilon1 )                    | regularized      | constant: 1 - γ τ     |
| this work                 | 1 ητ log 1 /epsilon1                           | regularized      | constant: 0 , 1 - γ τ |

(

)

(

)

Table 1: The iteration complexities of NPG methods to reach /epsilon1 -accuracy in terms of optimization error , where the unregularized (resp. regularized) version is given by (13) (cf. (15)) with η the learning rate. We assume exact gradient evaluation and softmax parameterization, and hide the dependencies that are logarithmic on problem parameters. Here, /epsilon1 -accuracy or /epsilon1 -optimality for the unregularized (resp. regularized) case mean V /star ( s ) -V π ( t ) ( s ) ≤ /epsilon1 (resp. V /star τ ( s ) -V π ( t ) τ ( s ) ≤ /epsilon1 ) holds simultaneously for all s ∈ S ; ρ denotes the initial state distribution, which clearly obeys 1 min s ∈S ρ ( s ) ≥ |S| .

time (in terms of |S| and 1 1 -γ ) to converge. In contrast, the bounds derived in the current paper are fully non-asymptotic, delineating clear dependencies on all salient problem parameters, which clearly demonstrate the algorithmic advantages of NPG methods. Fig. 1 depicts the policy paths of PG and NPG methods with entropy regularization for a simple bandit problem with three actions. It is evident from the plots that the NPG method follows a more direct path to the global optimum compared to the PG counterpart and hence converges faster. In addition, both algorithms converge more rapidly as the regularization parameter τ increases.

## 1.3 Other related works

There has been a flurry of recent activities in studying theoretical behaviors of policy optimization methods. For example, Fazel et al. (2018); Jansch-Porto et al. (2020); Tu and Recht (2019); Zhang et al. (2019a); Mohammadi et al. (2019) established the global convergence of policy optimization methods for a couple of control problems; Bhandari and Russo (2019) identified structural properties that guarantee the global optimality of PG methods without parameterization; Karimi et al. (2019) studied the convergence of PG methods to an approximate first-order stationary point, and Zhang et al. (2019b) proposed a variant of PG methods that converges to locally optimal policies leveraging saddle-point escaping algorithms in nonconvex optimization. Beyond the tabular setting, the convergence of PG methods with function approximations has been studied in Agarwal et al. (2020b); Wang et al. (2019); Liu et al. (2019). In particular, Cai et al. (2019) developed an optimistic variant of NPG that incorporates linear function approximation. We do not elaborate on this line of works since our focus is on understanding the performance of entropy-regularized NPG in the tabular setting; we also do not elaborate on PG methods that involve sample-based estimates, since we primarily consider exact gradients or black-box gradient estimators.

Regarding entropy regularization, Neu et al. (2017); Geist et al. (2019) provided unified views of entropyregularized MDPs from an optimization perspective by connecting them to algorithms such as mirror descent (Nemirovsky and Yudin, 1983) and dual averaging (Nesterov, 2009). The soft policy iteration algorithm has been identified as a special case of entropy-regularized NPG, highlighting again the link between policy gradient methods and soft Q-learning (Schulman et al., 2017a). The asymptotic convergence of soft policy iteration was established in Haarnoja et al. (2017), which fell short of providing explicit convergence rate guarantees. Additionally, Grill et al. (2019) developed planning algorithms for entropy-regularized MDPs, and Mei et al. (2020) showed that the sub-optimality gap of soft policy iteration is small if the policy improvement is small in consecutive iterations.

1

Figure 1: Comparisons of PG and NPG methods with entropy regularization for a bandit problem ( γ = 0) with 3 actions, whose corresponding rewards are 1 . 0, 0 . 9 and 0 . 1, respectively. The regularization parameter is set as τ = 0 . 1 for the first row and τ = 1 for the second row. In (a) and (d), the policy paths of (log π ( a 1 ) , log π ( a 2 )) following the PG method are plotted in orange, with the blue lines indicating the gradient flow; in (b) and (e), the policy paths of (log π ( a 1 ) , log π ( a 2 )) following the NPG method are depicted in red, with the blue lines indicating the natural gradient flow. The error contractions of both PG and NPG methods with η = 0 . 1 are shown in (c) and (f).

<!-- image -->

## 1.4 Notation

We denote by ∆( S ) (resp. ∆( A )) the probability simplex over the set S (resp. A ). When scalar functions such as | · | , exp( · ) and log( · ) are applied to vectors, their applications should be understood in an entry-wise fashion. For instance, given any vector z = [ z i ] 1 ≤ i ≤ n ∈ R n , the notation | · | denotes | z | := [ | z i | ] 1 ≤ i ≤ n ; other functions are defined analogously. For any vectors z = [ z i ] 1 ≤ i ≤ n and w = [ w i ] 1 ≤ i ≤ n , the notation z ≥ w (resp. z ≤ w ) means z i ≥ w i (resp. z i ≤ w i ) for all 1 ≤ i ≤ n . The softmax function softmax : R n ↦→ R n is defined such that [ softmax ( θ )] i := exp( θ i ) / ( ∑ i exp( θ i ) ) for a vector θ = [ θ i ] 1 ≤ i ≤ n ∈ R n . Given two probability distributions π 1 and π 2 over A , the Kullback-Leibler (KL) divergence from π 2 to π 1 is defined by KL ( π 1 ‖ π 2 ) := ∑ a ∈A π 1 ( a ) log π 1 ( a ) π 2 ( a ) . Given two probability distributions p and q over S , we introduce the notation ∥ ∥ p q ∥ ∥ ∞ := max s ∈S p ( s ) q ( s ) and ∥ ∥ 1 q ∥ ∥ ∞ := max s ∈S 1 q ( s ) .

## 2 Model and algorithms

## 2.1 Problem settings

Markov decision processes. The current paper studies a discounted Markov decision process (MDP) (Puterman, 2014) denoted by M = ( S , A , P, r, γ ), where S is the state space, A is the action space, γ ∈ (0 , 1) indicates the discount factor, P : S × A → ∆( S ) is the transition kernel, and r : S × A → [0 , 1] stands for

the reward function. 1 To be more specific, for each state-action pair ( s, a ) ∈ S × A and any state s ′ ∈ S , we denote by P ( s ′ | s, a ) the transition probability from state s to state s ′ when action a is taken, and r ( s, a ) the instantaneous reward received in state s due to action a . A policy π : S → ∆( A ) represents a (randomized) action selection rule, namely, π ( a | s ) specifies the probability of executing action a in state s for each ( s, a ) ∈ S × A .

Value functions and Q-functions. For any given policy π , we denote by V π : S → R the corresponding value function, namely, the expected discounted cumulative reward with an initial state s 0 = s , given by

<!-- formula-not-decoded -->

where the action a t ∼ π ( ·| s t ) follows the policy π and s t +1 ∼ P ( ·| s t , a t ) is generated by the MDP M for all t ≥ 0. We also overload the notation V π ( ρ ) to indicate the expected value function of a policy π when the initial state is drawn from a distribution ρ over S , namely,

<!-- formula-not-decoded -->

Additionally, the Q-function Q π : S × A → R of a policy π - namely, the expected discounted cumulative reward with an initial state s 0 = s and an initial action a 0 = a - is defined by

<!-- formula-not-decoded -->

where the action a t ∼ π ( ·| s t ) follows the policy π for all t ≥ 1, and s t +1 ∼ P ( ·| s t , a t ) is generated by the MDP M for all t ≥ 0.

Discounted state visitation distributions. A type of marginal distributions - commonly dubbed as discounted state visitation distributions - plays an important role in our theoretical development. To be specific, the discounted state visitation distribution d π s 0 of a policy π given the initial state s 0 ∈ S is defined by where the trajectory ( s 0 , s 1 , · · · ) is generated by the MDP M under policy π starting from state s 0 . In words, d π s 0 ( · ) captures the state occupancy probabilities when each state visitation is properly discounted depending on the time stamp. Further, for any distribution ρ over S , we define the distribution d π ρ as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which describes the discounted state visitation distribution when the initial state s 0 is randomly drawn from a prescribed initial distribution ρ .

Softmax parameterization. It is common practice to parameterize the class of feasible policies in a way that is amenable to policy optimization. The focal point of this paper is softmax parameterization - a widely adopted scheme which naturally ensures that the policy lies in the probability simplex. Specifically, for any θ : S × A → R (called 'logic values'), the corresponding softmax policy π θ is generated through the softmax transform

<!-- formula-not-decoded -->

In what follows, we shall often abuse the notation to treat π θ and θ as vectors in R |S||A| , and suppress the subscript θ from π θ , whenever it is clear from the context.

1 For the sake of simplicity, we assume throughout that the reward resides within [0 , 1]. Our results can be generalized in a straightforward manner to other ranges of bounded rewards.

Entropy-regularized value maximization. To promote exploration and discourage premature convergence to suboptimal policies, a widely used strategy is entropy regularization, which searches for a policy that maximizes the following entropy-regularized value function

<!-- formula-not-decoded -->

Here, the quantity τ ≥ 0 denotes the regularization parameter, and H ( ρ, π ) stands for a sort of discounted entropy defined as follows

<!-- formula-not-decoded -->

Equivalently, V π τ can be viewed as the value function of π by adjusting the instantaneous reward to be policy-dependent regularized version as follows

<!-- formula-not-decoded -->

We also define V π τ ( s ) analogously when the initial state is fixed to be any given state s ∈ S . The regularized Q-function Q π τ of a policy π , also known as the soft Q-function, 2 is related to V π τ as

<!-- formula-not-decoded -->

Optimal policies and stationary distributions. Denote by π /star (resp. π /star τ ) the policy that maximizes the value function (resp. regularized value function with regularization parameter τ ), and let V /star (resp. V /star τ ) represent the resulting optimal value function (resp. regularized value function). Importantly, the optimal policies π /star and π /star τ of the MDP do not depend on the initial distribution ρ (Mei et al., 2020). In addition, π /star and π /star τ maximize the Q-function and the soft Q-function, respectively (which is self-evident from (11a)). A simple yet crucial connection between π /star and π /star τ can be demonstrated via the following sandwich bound 3

<!-- formula-not-decoded -->

which holds for all initial distributions ρ . The key takeaway message is that: the optimal policy π /star τ of the regularized problem could also be nearly optimal in terms of the unregularized value function, as long as the regularization parameter τ is chosen to be sufficiently small.

## 2.2 Algorithm: NPG methods with entropy regularization

Natural policy gradient methods. Towards computing the optimal policy (in the parameterized form), perhaps the first strategy that comes into mind is to run gradient ascent w.r.t. the parameter θ until convergence - a first-order method commonly referred to as the policy gradient (PG) algorithm (e.g. Sutton et al. (2000)). In comparison, the natural policy gradient (NPG) method (Kakade, 2002) adopts a pre-conditioned gradient update rule

<!-- formula-not-decoded -->

in the hope of searching along a direction independent of the policy parameterization in use. Here, η is the learning rate or stepsize, F θ ρ denotes the Fisher information matrix given by

<!-- formula-not-decoded -->

2 In this paper, we use the terms 'regularized' value (resp. Q) functions and 'soft' value (resp. Q) functions interchangeably. 3 To see this, invoke the optimality of π /star τ and the elementary entropy bound 0 ≤ H ( ρ, π ) ≤ 1 1 -γ log |A| to obtain

<!-- formula-not-decoded -->

and we use B † to indicate the Moore-Penrose pseudoinverse of a matrix B . It has been understood that the NPG method essentially attempts to monitor/control the policy changes approximately in terms of the Kullback-Leibler (KL) divergence (see e.g. Schulman et al. (2015, Section 7)).

NPG methods with entropy regularization. Equipped with entropy regularization, the NPG update rule can be written as

<!-- formula-not-decoded -->

where F θ ρ is defined in (14) and V π τ ( ρ ) is defined in (8). Under softmax parameterization, this update rule admits a fairly simple form in the policy space (see Appendix A.1 for detailed derivations), which, interestingly, is invariant to the choice of ρ . More precisely, if we let θ ( t ) denote the t -th iterate and π ( t ) = softmax ( θ ( t ) ) the associated policy, then the entropy-regularized NPG updates satisfy

<!-- formula-not-decoded -->

where Q π ( t ) τ is the soft Q-function of policy π ( t ) , and Z ( t ) ( s ) is some normalization factor. This can alternatively be viewed as an instantiation/variant of the trust region policy optimization (TRPO) algorithm (see Schulman et al. (2015); Shani et al. (2019)). As an important special case, the update rule (16) reduces to

<!-- formula-not-decoded -->

for some normalization factor Z ( t ) ( s ). The procedure (17) can be interpreted as a 'soft' version of the classical policy iteration algorithm (Bertsekas, 2017) (as it employs a softmax function to approximate the max operator) w.r.t. the soft Q-function, and is often dubbed as soft policy iteration (SPI) (see Haarnoja et al. (2018, Section 4.1)).

To simplify notation, we shall use V ( t ) τ , Q ( t ) τ and d ( t ) ρ throughout to denote V π ( t ) τ , Q π ( t ) τ and d π ( t ) ρ , respectively. The complete procedure is summarized in Algorithm 1.

## Algorithm 1: Entropy-regularized NPG with exact policy evaluation

- 1 inputs: learning rate η , initialization π (0) .
- 2 for t = 0 , 1 , 2 , · · · do

3 Compute the regularized Q-function Q ( t ) τ (defined in (11a)) of policy π ( t ) . 4 Update the policy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 2.3 A warm-up example: the bandit case

Inspired by Schulman et al. (2017a); Mei et al. (2020), we look at a toy example - the bandit case - before proceeding to general MDPs. To be more precise, this is concerned with an MDP with only a single state and discount factor γ = 0. Despite its simplicity, the exposition of this example sheds light upon the convergence behavior of the regularized NPG methods of interest.

In this single-state example with γ = 0, the aim reduces to computing a policy π θ : A → ∆( A ) that solves the following optimization problem

<!-- formula-not-decoded -->

where r ( a ) is the instantaneous reward of taking action a (i.e. pulling arm a in the bandit language). As demonstrated in Mei et al. (2020, Proposition 1), this toy case is already non-concave and hence nontrivial to solve. As it turns out, direct calculation reveals that the optimal policy of (19) is given by

<!-- formula-not-decoded -->

which is in general a randomized policy. When applied to this example, the entropy-regularized NPG update rule (18) simplifies to (up to normalization)

<!-- formula-not-decoded -->

with η the learning rate. The following proposition, whose proof is fairly elementary and can be found in Appendix B, reveals that the above procedure converges (at least) linearly to the optimal policy π /star τ .

Proposition 1 (The bandit case) . The algorithm (21) converges linearly to π /star τ (cf. (20) ) in an entrywise fashion, namely,

∥ ∥ log π ( t ) -log π /star τ ∥ ∥ ∞ ≤ 2(1 -τη ) t ∥ ∥ log π (0) -log π /star τ ∥ ∥ ∞ . While this result concentrates only on a toy example, it hints at the potential capability of entropyregularized NPG methods in achieving rapid convergence. In particular, by setting the learning rate to be η = 1 /τ , the algorithm converges in a single iteration . This special choice corresponds to the SPI update (17), which will be singled out in our general theory due to its appealing convergence properties.

## 3 Main results

Given its appealing convergence behavior when applied to the preceding warm-up example (the bandit case), it is natural to ask whether the entropy-regularized NPG method is fast-convergent for general MDPs. This section answers this question in the affirmative.

## 3.1 Exact entropy-regularized NPG methods

We first study the convergence behavior of entropy-regularized NPG methods (18) assuming access to exact policy evaluation in every iteration (namely, we assume the soft Q-function Q ( t ) τ can be evaluated accurately in all t ). Remarkably, this algorithm converges linearly - in terms of computing both the optimal soft Q-function Q /star τ and the associated log policy log π /star τ - as asserted by the following theorem. The proof of this result is provided in Section 4.2.

Theorem 1 (Linear convergence of exact entropy-regularized NPG) . For any learning rate 0 &lt; η ≤ (1 -γ ) /τ , the entropy-regularized NPG updates (18) satisfy for all t ≥ 0 , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is worth emphasizing that Theorem 1 is stated in a completely non-asymptotic form containing no hidden constants, and that our result covers any learning rate η in the range (0 , (1 -γ ) /τ ]. A few implications of this theorem are in order.

- Linear convergence of soft Q-functions. To reach ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ ≤ /epsilon1 , the entropy-regularized NPG method needs at most 1 ητ log ( C 1 γ /epsilon1 ) iterations. Remarkably, the iteration complexity almost does not depend on the dimensions of the MDP (except for some very weak dependency embedded

in log C 1 ) - this inherits a dimension-free feature of NPG methods that has been highlighted in Agarwal et al. (2020b) for the unregularized case. When the learning rate η is fixed in the admissible range, the iteration complexity scales inverse proportionally with τ , suggesting a higher level of entropy regularization might accelerate convergence, albeit to the solution of a regularized problem that is further away from the original MDP.

- Linear convergence of log policies. In contrast to the unregularized case, entropy regularization ensures uniqueness of the optimal policy and, therefore, makes it possible to study the convergence of the policy directly. Our theorem reveals that the entropy-regularized NPG method needs at most 1 ητ log ( 2 C 1 /epsilon1τ ) iterations to yield ∥ log π /star τ -log π ( t +1) ∥ ∞ ≤ /epsilon1 .
- ∥ ∥ · Linear convergence of soft value functions. As a byproduct, Theorem 1 implies that the iterates of soft value functions also converge linearly, namely,

To see this, we make note of the following relation previously established in Nachum et al. (2017):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, combining this with the definition (11b) yields

<!-- formula-not-decoded -->

- Convergence rate of SPI. The best convergence guarantee is achieved when η = (1 -γ ) /τ (i.e. the SPI case), where the iteration complexity to reach ∥ Q /star τ -Q ( t ) τ ∥ ∞ ≤ /epsilon1 reduces to

<!-- formula-not-decoded -->

which is proportional to the effective horizon 1 1 -γ modulo some log factor. This means the iteration complexity of SPI recovers that of policy iteration (Puterman, 2014). Interestingly, the contraction rate in this case (which is γ ) is independent of the choice of the regularization parameter τ . Similarly, the iteration complexity of SPI to reach ∥ ∥ log π /star τ -log π ( t +1) ∥ ∥ ∞ ≤ /epsilon1 becomes 1 1 -γ log ( 2 ‖ Q /star τ -Q (0) τ ‖ ∞ /epsilon1τ ) , and the contraction rate is again independent of τ .

Comparison with entropy-regularized policy gradient methods. Mei et al. (2020, Theorem 6) proved that the entropy-regularized policy gradient method achieves 4

<!-- formula-not-decoded -->

and they further showed that inf k ≥ 0 min s,a π ( k ) ( a | s ) is non-vanishing in t . It remains unclear, however, how inf t ≥ 0 min s,a π ( t ) ( a | s ) scales with other potentially large salient parameters like ( |S| , |A| , 1 1 -γ , 1 τ ). In truth, existing theory does not rule out the possibility of exponential dependency on these salient parameters. It would thus be of great interest to establish algorithm-dependent lower bounds to uncover the right scaling with these important parameters. In contrast, our convergence guarantees for entropy-regularized NPG methods unveil concrete dependencies on all problem parameters.

4 Here, we have assumed the exact policy gradient is computed with respect to V ( t ) τ ( ρ ).

Computing an /epsilon1 -optimal policy for the original MDP. Thus far, we have established an intriguing convergence behavior of the entropy-regularized NPG method. However, caution needs to be exercised when interpreting the efficacy of this method: the preceding results are concerned with convergence to the optimal regularized value function V /star τ , as opposed to finding the optimal value function V /star of the original MDP. Fortunately, by choosing the regularization parameter τ to be sufficiently small (in accordance with the target accuracy level /epsilon1 ), we can guarantee that V /star τ ≈ V /star (cf. (12)), thus ensuring the relevance and applicability of our results for solving the original MDP. To be specific, let us adopt the following choice of τ :

<!-- formula-not-decoded -->

and assume the error of the regularized value function satisfies ∥ ∥ V /star τ -V ( t ) τ ∥ ∥ ∞ &lt; /epsilon1/ 2. By virtue of Theorem 1, this optimization accuracy can be achieved via no more than 4 log |A| (1 -γ ) η/epsilon1 log ( 2 C 1 γ /epsilon1 ) iterations of entropyregularized NPG updates with a general learning rate, 5 or no more than 1 1 -γ log ( γ ‖ Q /star τ -Q (0) τ ‖ ∞ /epsilon1 ) iterations with the specific choice η = 1 -γ τ . It then follows that

<!-- formula-not-decoded -->

for any s ∈ S , where we have used our choice of τ in (25). Here, the second inequality arises from (12) as well as the fact that for any policy π ,

<!-- formula-not-decoded -->

given the elementary entropy bound 0 ≤ H ( s, π ) ≤ 1 1 -γ log |A| .

Convergence guarantee for conservative policy iteration (CPI). Our analysis framework also leads to a similar convergence guarantee for a type of policy updates adopted in conservative policy iteration (Kakade and Langford, 2002), where the policy is updated as a convex combination of the previous policy and an improved one. We refer the interested reader to Appendix D for details.

## 3.2 Approximate entropy-regularized NPG methods

There is no shortage of scenarios where the soft Q-function Q ( t ) τ ( s, a ) is available only in an approximate fashion, e.g. the cases when the value function has to be evaluated using finite samples. To account for inexactness of policy evaluation, we extend our theory to accommodate the following approximate update rule: for any s ∈ S and any t ≥ 0,

<!-- formula-not-decoded -->

Here, δ is some quantity that captures the size of approximation errors. We do not specify the estimator for the soft Q-function (as long as it satisfies the entrywise estimation bound), thus allowing one to plug in both model-based and model-free value function estimators designed for a variety of sampling mechanisms (e.g. Azar et al. (2013); Li et al. (2020b)). Encouragingly, the algorithm (26) is robust vis-` a-vis inexactness of value function estimates, as it still converges linearly until an error floor is hit. This is formalized in the following theorem, with the proof postponed to Section 4.3.

5 This result is in fact better than the iteration complexity 2 (1 -γ ) 2 /epsilon1 of the unregularized NPG method established in Agarwal et al. (2020b) as soon as η ≥ 2(1 -γ ) log |A| log ( 2 C 1 γ /epsilon1 ) . Consequently, our finding hints at the potential advantage of entropy-regularized NPG methods over the unregularized counterpart even when solving the original MDP.

Theorem 2 (Linear convergence of approximate entropy-regularized NPG) . When 0 &lt; η ≤ (1 -γ ) /τ , the inexact entropy-regularized NPG updates (26) satisfy

<!-- formula-not-decoded -->

for all t ≥ 0 , where C 1 is the same as defined in (23) and C 2 is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Apparently, Theorem 2 reduces to Theorem 1 when δ = 0. As implied by this theorem, if the /lscript ∞ error of the soft-Q function estimates does not exceed

<!-- formula-not-decoded -->

then the algorithm (26) achieves 2 /epsilon1 -accuracy (i.e. ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ ≤ 2 /epsilon1 ) within 1 ητ log ( C 1 γ /epsilon1 ) iterations. In particular, in the case of soft policy iteration (i.e. η = 1 -γ τ ), the tolerance level δ can be up to (1 -γ ) 2 /epsilon1 2 γ , which matches the theory of approximate policy iteration in Agarwal et al. (2019).

Remark 1. It is straightforward to combine Theorem 2 with known sample complexities for approximate policy evaluation to obtain a crude sample complexity bound. For instance, assuming access to a generative model, Li et al. (2020a) asserts that for any fixed policy π , model-based policy evaluation achieves ∥ ∥ ̂ Q π τ -Q π τ ∥ ∥ ∞ ≤ δ with high probability, as long as the number of samples per state-action pair exceeds the order of

<!-- formula-not-decoded -->

up to some logarithmic factor. By employing fresh samples for each policy evaluation, we can set δ = (1 -γ ) 2 /epsilon1 2 γ and invoke the union bound over ˜ O ( 1 1 -γ ) iterations to demonstrate that: SPI with model-based policy evaluation needs at most samples to find an /epsilon1 -optimal policy. Here, ˜ O ( · ) hides any logarithmic factor. We note, however, that the above sample analysis is extremely crude and might be improvable by, say, allowing sample reuses across iterations. It remains an interesting open question as to whether NPG with entropy regularization is minimax-optimal with a generative model, where the minimax lower bound is on the order of |S| |A| (1 -γ ) 3 /epsilon1 2 (Azar et al., 2013) and achievable by model-based plug-in estimators (Agarwal et al., 2020a; Li et al., 2020a) but not by vanilla Q-learning (Li et al., 2021a).

<!-- formula-not-decoded -->

## 3.3 Quadratic convergence in the small/epsilon1 regime

Somewhat remarkably, the regularized NPG method with η = 1 -γ τ achieves super-linear convergence in computing V /star τ , once the algorithm enters a sufficiently small local neighborhood surrounding the optimizer.

Before presenting the result, we need to introduce the stationary distribution over S of the MDP M under policy π /star τ , denoted by µ /star τ ∈ ∆( S ). It is straightforward to verify the following basic property

<!-- formula-not-decoded -->

given that the state visitation distribution remains unchanged if the initial state is already in a steady state. Throughout this paper, we assume that min s µ /star τ ( s ) &gt; 0. Our finding is stated in the following theorem, with the proof deferred to Section 4.4.

Theorem 3 (Quadratic convergence of exact regularized NPG) . Suppose that the algorithm (17) with η = 1 -γ τ (or SPI) satifies for all t ≥ 0 , then one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 2. In view of the convergence guarantees in Theorem 2, a suitable initialization of π (0) and V (0) τ (such that 4 γ 2 (1 -γ ) τ ∥ ∥ 1 µ /star τ ∥ ∥ ∞ ( V /star τ ( µ /star τ ) -V (0) τ ( µ /star τ )) &lt; 1 ) can be obtained by running SPI for sufficiently many iterations; further, all subsequent iterations are then guaranteed to satisfy (30) according to Theorem 2.

<!-- formula-not-decoded -->

Under the assumptions of Theorem 3, our result indicates that: when /epsilon1 is sufficiently small, the iteration complexity for SPI to yield an /epsilon1 optimization accuracy - that is, V /star τ ( ρ ) -V ( t ) τ ( ρ ) ≤ /epsilon1 - is at most on the order of

This uncovers the faster-than-linear convergence behavior of regularized NPG methods in the high-accuracy regime, accommodating a range of optimization accuracy and all possible choices of the regularization parameter τ . It is worth noting, however, that our quadratic convergence result is stated in terms of the optimization accuracy (namely, convergence to the soft value function V /star τ ( ρ )) as opposed to the accuracy w.r.t. the original unregularized MDP. Thus, interpreting Theorem 3 in practice requires caution, since the approximation error V /star τ ( ρ ) -V /star ( ρ ) might sometimes dominate the optimization error in this regime.

## 4 Analysis

## 4.1 Main pillars for the convergence analysis

Before proceeding, we isolate a few ingredients that provide the main pillars for our theoretical development.

Performance improvement and monotonicity. This lemma is a sort of ascent lemma , which quantifies the progress made over each iteration - measured in terms of the soft value function.

Lemma 1 (Performance improvement) . Suppose that 0 &lt; η ≤ (1 -γ ) /τ . For any distribution ρ , one has

<!-- formula-not-decoded -->

Proof. See Appendix C.1.

In a nutshell, Lemma 1 asserts that each iteration of the entropy-regularized NPG method is guaranteed to improve the estimates of the soft value function, with the improvement depending on the KL divergence between the current policy π ( t ) and the updated one π ( t +1) . In fact, the arbitrary choice of ρ readily reveals a sort of pointwise monotoncity for the above range of learning rates, in the sense that V ( t +1) τ ( s ) ≥ V ( t ) τ ( s ) for all s ∈ S . Indeed, this lemma can be viewed as the counterpart of the performance difference lemma in Kakade and Langford (2002) for the unregularized form. Lemma 1 also implies the monotonicity of the soft Q-function in t , since for any ( s, a ) ∈ S × A one has

<!-- formula-not-decoded -->

where the equalities follow from the definition (11a), and the inequality follows since V ( t +1) τ ( s ) ≥ V ( t ) τ ( s ) for all s ∈ S - a consequence of Lemma 1 and the non-negativity of the KL divergence.

A key contraction operator: the soft Bellman optimality operator. An operator that plays a pivotal role in the theory of dynamic programming (Bellman, 1952) is the renowned Bellman optimality operator T : R |S||A| → R |S||A| , defined as follows

<!-- formula-not-decoded -->

In order to facilitate analysis for entropy-regularized MDPs, we find it particularly fruitful to introduce a 'soft' Bellman optimality operator T τ : R |S||A| → R |S||A| as follows

<!-- formula-not-decoded -->

which reduces to T when τ = 0. To see this, observe that

<!-- formula-not-decoded -->

where the last line follows since the optimal policy is exactly the greedy policy w.r.t. Q (Puterman, 2014). The operator T τ plays a similar role as does the Bellman optimality operator for the unregularized case, whose key properties are summarized below. Similar results have been derived in Dai et al. (2018, Section 3.1).

Lemma 2 (Soft Bellman optimality operator) . The operator T τ defined in (35) satisfies the properties below.

- T τ admits the following closed-form expression:

<!-- formula-not-decoded -->

- The optimal soft Q-function Q /star τ is a fixed point of T τ , namely,

<!-- formula-not-decoded -->

- T τ is a γ -contraction in the /lscript ∞ norm, namely, for any Q 1 , Q 2 ∈ R |S||A| one has

Proof. See Appendix C.2.

<!-- formula-not-decoded -->

For those familiar with dynamic programming, it should become evident that T τ inherits many appealing features of the original Bellman optimality operator T . For example, as an immediate application of the γ -contraction property (38) and the fixed-point property (37), the following soft Q -value iteration

<!-- formula-not-decoded -->

is guaranteed to converge linearly to the optimal Q /star τ with a contraction rate γ - a simple observation consistent with the behavior of value iteration designed for unregularized MDPs.

## 4.2 Analysis of exact entropy-regularized NPG methods

## 4.2.1 The SPI case (i.e. η = (1 -γ ) /τ )

With the help of the soft Bellman optimality operator, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, (i) comes from the definition (11a) of the soft Q-function, (ii) follows from the relation (11b), (iii) relies on the monotonicity of the soft Q-function (see (33)), (iv) uses the form of π ( t +1) in (17), whereas (v) makes use of the expression (36). The inequality (39) further leads to 0 ≤ Q /star τ -Q ( t +1) τ ≤ Q /star τ -T τ ( Q ( t +1) τ ), and hence where the first equality follows from the fixed-point property (37), and the second inequality is due to the contraction property (38). We have thus established linear convergence of Q ( t ) τ in ‖ · ‖ ∞ for this case.

Turning to the log policies, recall that

<!-- formula-not-decoded -->

where the second relation comes from Nachum et al. (2017, Eqn. (12)). It then follows from an elementary property of the softmax function (see (66) in Appendix A.2) that

<!-- formula-not-decoded -->

thus concluding the proof for this case.

## 4.2.2 The case with general learning rates

We now move to the case with a general learning rate. For the sake of brevity, we shall denote

<!-- formula-not-decoded -->

Additionally, it is helpful to introduce an auxiliary sequence { ξ ( t ) ∈ R |S||A| } constructed recursively by

<!-- formula-not-decoded -->

It is easily seen from the construction (42b) that and, consequently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 1: a linear system that describes the error recursions. In the case with general learning rates, the estimation error ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ does not contract in the same form as that of soft policy iteration; instead, it is more succinctly controlled with the aid of an auxiliary quantity ∥ ∥ Q /star τ -τ log ξ ( t ) ∥ ∥ ∞ . In what follows, we leverage a simple yet powerful technique by describing the dynamics concerning ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ and ∥ ∥ Q /star τ -τ log ξ ( t ) ∥ ∥ ∞ via a linear system, whose spectral properties dictate the convergence rate. Towards this, we start with the following key observation, whose proof is deferred to Appendix C.3.

Lemma 3. For any learning rate 0 &lt; η ≤ (1 -γ ) /τ , the entropy-regularized NPG updates (18) satisfy where α is defined in (41) .

<!-- formula-not-decoded -->

If we substitute (43) into (45), it is straightforwardly seen that Lemma 3 is a generalization of the contraction property (40) of soft policy iteration (the case corresponding to α = 0). Given that Lemma 3 involves the interaction of more than one quantities, it is convenient to combine (44) and (45) into the following linear system

<!-- formula-not-decoded -->

where

A := [ γ (1 -α ) γα 1 -α α ] , x t := [ ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ ∥ ∥ Q /star τ -τ log ξ ( t ) ∥ ∥ ∞ ] and y := [ ∥ ∥ Q (0) τ -τ log ξ (0) ∥ ∥ ∞ 0 ] . (47) We shall make note of the following appealing features of the rank-1 system matrix A :

<!-- formula-not-decoded -->

which relies on the identity (1 -α ) γ + α = 1 -ητ (according to the definition (41) of α ).

Remark 3. By left multiplying both sides of (46) by [1 -α, α ] , we obtain

<!-- formula-not-decoded -->

where L ( t ) := (1 -α ) ∥ ∥ Q /star τ -Q ( t ) τ ∥ ∥ ∞ + α ∥ ∥ Q /star τ -τ log ξ ( t ) ∥ ∥ ∞ can be viewed as a sort of Lyapunov function. This hints at the intimate connection between our proof and the Lyapunov-type analysis used in system theory.

Step 2: characterizing the contraction rate from the linear system. In view of the recursion formula (46) and the non-negativity of ( A,x t , y ), it is immediate to deduce that

<!-- formula-not-decoded -->

Here, the last line follows from the elementary relation

<!-- formula-not-decoded -->

and the invertibility of α -1 A -I (since α -1 A is a rank-1 matrix whose non-zero singular value is larger than 1). In addition, the Woodbury matrix inversion formula together with the decomposition (48) yields

<!-- formula-not-decoded -->

which is a non-negative vector. Consequently, this taken together with (49) gives

<!-- formula-not-decoded -->

where the third line follows from (48), (50) and the definition of x t . Further, observe that

<!-- formula-not-decoded -->

where the inequality comes from the triangle inequality, and the last identity follows from (42a). Substituting this back into (51), we obtain

<!-- formula-not-decoded -->

To finish up, recall that π ( t ) is related to ξ ( t ) as follows

<!-- formula-not-decoded -->

which can be seen by comparing (42) with (18). Therefore, invoking the elementary property of the softmax function (see (66) in Appendix A.2), we arrive at

This combined with (53) as well as the definition (47) of x t +1 immediately establishes Theorem 1.

<!-- formula-not-decoded -->

## 4.3 Analysis of approximate entropy-regularized NPG methods

We now turn to the convergence properties of approximate entropy-regularized NPG methods - as claimed in Theorem 2 - when only inexact policy evaluation ̂ Q ( t ) τ is available (in the sense of (26)).

Step 1: performance difference accounting for inexact policy evaluation. We first bound the quality of the policy updates (26) by examining the difference between V ( t +1) τ and V ( t ) τ and how it is impacted by the imperfectness of policy evaluation. This is made precise by the following lemma.

Lemma 4 (Performance difference of approximate entropy-regularized NPG) . Suppose that 0 &lt; η ≤ (1 -γ ) /τ . For any state s 0 ∈ S , one has

Proof. See Appendix C.4.

<!-- formula-not-decoded -->

The careful reader might already realize that the above lemma is a relaxation of Lemma 1; in particular, the last term of (55) quantifies the effect of the approximation error (i.e. the difference between ̂ Q ( t ) τ and Q ( t ) τ ) upon performance improvement. Under the assumption ∥ ∥ ̂ Q ( t ) τ -Q ( t ) τ ∥ ∥ ∞ ≤ δ , repeating the argument of (33) reveals that the soft Q -function estimates are not far from being monotone in t , in the sense that

<!-- formula-not-decoded -->

Step 2: a linear system accounting for inexact policy evaluation. With the assistance of (56), it is possible to construct a linear system - similar to the one built in Section 4.2 - that takes into account inexact policy evaluation. Towards this end, we adopt a similar approach as in (42) by introducing the following auxiliary sequence ξ ( t ) defined recursively using Q ( t ) τ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α := 1 -ητ 1 -γ as before.

where

<!-- formula-not-decoded -->

Here, the system matrix B (in particular its eigenvalues) governs the contraction rate, while the term b captures the error introduced by inexact policy evaluation. Theorem 2 then follows by carrying out a similar analysis argument as in Section 4.2 to characterize the error dynamics. Details are postponed to Appendix E.

## 4.4 Analysis of local quadratic convergence

We now sketch the proof of Theorem 3, which establishes local quadratic convergence of SPI.

Step 1: characterization of the sub-optimality gap. Lemma 1 bounds the performance improvement of SPI by the KL divergence between the current policy π ( t ) and the updated policy π ( t +1) . Interestingly, the type of KL divergence can be further employed to bound the sub-optimality gap for each iteration.

Lemma 5 (Sub-optimality gap) . Suppose that η = (1 -γ ) /τ . For any distribution ρ , one has

<!-- formula-not-decoded -->

Proof. This result has appeared in Mei et al. (2020, Eqn. (486)). For completeness we include a proof in Appendix C.5.

In words, Lemma 5 formalizes the connection between the sub-optimality gap (w.r.t. the optimal soft value function) and the proximity of the two consecutive policy iterates. As reflected by this lemma, if the current and the updated policies do not differ by much (which indicates that the algorithm might be close to convergence), then the current estimate of the soft value function is close to optimal.

Step 2: a contraction property. The importance of the above two lemmas is made apparent by the following contraction property when η = (1 -γ ) /τ :

<!-- formula-not-decoded -->

We claim that the following linear system tracks the error dynamics of the policy updates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, (i) arises from Lemma 1, (ii) employs the pre-factor ∥ ∥ d π /star τ ρ /d ( t +1) ρ ∥ ∥ -1 ∞ to accommodate the change of distributions, whereas (iii) follows from Lemma 5.

Step 3: super-linear convergence in the small/epsilon1 regime. The contraction property (60) implies that V ( t +1) τ ( ρ ) converges super-linearly to V /star τ , once π ( t ) gets sufficiently close to π /star τ . In fact, once the ratio d ( t +1) ρ /d π /star τ ρ becomes sufficiently close to 1, the contraction factor 1 -∥ ∥ d π /star τ ρ /d ( t +1) ρ ∥ ∥ -1 ∞ in (60) is approaching 0, thereby accelerating convergence. This observation underlies Theorem 3, whose complete analysis is postponed to Appendix F.

## 5 Discussions

This paper establishes non-asymptotic convergence of entropy-regularized natural policy gradient methods, providing theoretical footings for the role of entropy regularization in guaranteeing fast convergence. Our analysis opens up several directions for future research; we close the paper by sampling a few of them.

- Extended analysis of policy gradient methods with inexact gradients. It would be of interest to see whether our analysis framework can be applied to improve the theory of policy gradient methods (Mei et al., 2020) to accommodate the case with inexact policy gradients.
- Finite-sample analysis in the presence of sample-based policy evaluation. Another natural extension is towards understanding the sample complexity of entropy-regularized NPG methods when the value functions are estimated using rollout trajectories (see e.g. Kakade and Langford (2002); Agarwal et al. (2020b); Shani et al. (2019)), or using bootstrapping (see e.g. Xu et al. (2020); Haarnoja et al. (2018); Wu et al. (2020)).
- Function approximation. The current work has been limited to the tabular setting. It would certainly be interesting, and fundamentally important, to understand entropy-regularized NPG methods in conjunction with function approximation; see Sutton et al. (2000); Agarwal et al. (2019, 2020b) for a few representative scenarios.
- Beyond softmax parameterization. The current paper has been devoted to softmax parameterization, which enables a concise and NPG update rule. A couple of other parameterization schemes have been proposed for (vanilla) PG methods as well (Agarwal et al., 2019, 2020b; Bhandari and Russo, 2019, 2020), e.g. vanilla parameterization (paired with proper projection onto the probability simplex in each iteration), log-linear parameterization, and neural softmax parameterization. Unfortunately, the analysis in our paper relies heavily on the softmax NPG update rule, and does not immediately extend to other parameterization. It would be of great importance to establish convergence guarantees that accommodate other parameterizations of practical interest.

## Acknowledgments

The authors are grateful to anonymous reviewers for helpful suggestions, particularly for bringing Dai et al. (2018) to our attention. S. Cen and Y. Chi are supported in part by the grants ONR N00014-18-1-2142 and N00014-19-1-2404, ARO W911NF-18-1-0303, NSF CCF-1806154, CCF-1901199 and CCF-2007911. C. Cheng is supported by the William R. Hewlett Stanford graduate fellowship. Y. Wei is supported in part by the NSF grants CCF-2007911 and DMS-2015447. Y. Chen is supported in part by the grants AFOSR YIP award FA9550-19-1-0030, ONR N00014-19-1-2120, ARO YIP award W911NF-20-1-0097, ARO W911NF-18-1-0303, NSF CCF-1907661, IIS-1900140 and DMS-2014279, and the Princeton SEAS Innovation Award.

## References

- Agarwal, A., Jiang, N., and Kakade, S. M. (2019). Reinforcement learning: Theory and algorithms. Technical report.
- Agarwal, A., Kakade, S., and Yang, L. F. (2020a). Model-based reinforcement learning with a generative model is minimax optimal. In Conference on Learning Theory , pages 67-83. PMLR.
- Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. (2020b). Optimality and approximation with policy gradient methods in Markov decision processes. In Conference on Learning Theory , pages 64-66. PMLR.
- Ahmed, Z., Le Roux, N., Norouzi, M., and Schuurmans, D. (2019). Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning , pages 151-160.
- Amari, S.-I. (1998). Natural gradient works efficiently in learning. Neural computation , 10(2):251-276.
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Bellman, R. (1952). On the theory of dynamic programming. Proceedings of the National Academy of Sciences of the United States of America , 38(8):716.
- Bertsekas, D. P. (2017). Dynamic programming and optimal control (4th edition) . Athena Scientific.
- Bhandari, J. and Russo, D. (2019). Global optimality guarantees for policy gradient methods. arXiv preprint arXiv:1906.01786 .
- Bhandari, J. and Russo, D. (2020). A note on the linear convergence of policy gradient methods. arXiv preprint arXiv:2007.11120 .
- Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., and Lee, M. (2009). Natural actor-critic algorithms. Automatica , 45(11):2471-2482.
- Cai, Q., Yang, Z., Jin, C., and Wang, Z. (2019). Provably efficient exploration in policy optimization. arXiv preprint arXiv:1912.05830 .
- Cover, T. M. (1999). Elements of information theory . John Wiley &amp; Sons.
- Dai, B., Shaw, A., Li, L., Xiao, L., He, N., Liu, Z., Chen, J., and Song, L. (2018). SBEED: Convergent reinforcement learning with nonlinear function approximation. In International Conference on Machine Learning , pages 1125-1134. PMLR.
- Duan, Y., Chen, X., Houthooft, R., Schulman, J., and Abbeel, P. (2016). Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning , pages 1329-1338.
- Even-Dar, E., Kakade, S. M., and Mansour, Y. (2009). Online Markov decision processes. Mathematics of Operations Research , 34(3):726-736.
- Fazel, M., Ge, R., Kakade, S., and Mesbahi, M. (2018). Global convergence of policy gradient methods for the linear quadratic regulator. In International Conference on Machine Learning , pages 1467-1476.
- Geist, M., Scherrer, B., and Pietquin, O. (2019). A theory of regularized Markov decision processes. In International Conference on Machine Learning , pages 2160-2169.
- Grill, J.-B., Darwiche Domingues, O., Menard, P., Munos, R., and Valko, M. (2019). Planning in entropyregularized markov decision processes and games. In Advances in Neural Information Processing Systems , volume 32.
- Haarnoja, T., Tang, H., Abbeel, P., and Levine, S. (2017). Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning , pages 1352-1361.

- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290 .
- Hazan, E., Kakade, S., Singh, K., and Van Soest, A. (2019). Provably efficient maximum entropy exploration. In International Conference on Machine Learning , pages 2681-2691.
- Jansch-Porto, J. P., Hu, B., and Dullerud, G. (2020). Convergence guarantees of policy optimization methods for Markovian jump linear systems. arXiv preprint arXiv:2002.04090 .
- Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning , pages 267-274.
- Kakade, S. M. (2002). A natural policy gradient. In Advances in neural information processing systems , pages 1531-1538.
- Karimi, B., Miasojedow, B., Moulines, ´ E., and Wai, H.-T. (2019). Non-asymptotic analysis of biased stochastic approximation scheme. arXiv preprint arXiv:1902.00629 .
- Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in neural information processing systems , pages 1008-1014.
- Li, G., Cai, C., Chen, Y., Gu, Y., Wei, Y., and Chi, Y. (2021a). Is Q-learning minimax optimal? a tight sample complexity analysis. arXiv preprint arXiv:2102.06548 .
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020a). Breaking the sample size barrier in model-based reinforcement learning with a generative model. arXiv preprint arXiv:2005.12900 .
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020b). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. arXiv preprint arXiv:2006.03041 .
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2021b). Softmax policy gradient methods can take exponential time to converge. arXiv preprint arXiv:2102.11270 .
- Liu, B., Cai, Q., Yang, Z., and Wang, Z. (2019). Neural trust region/proximal policy optimization attains globally optimal policy. In Advances in Neural Information Processing Systems , pages 10565-10576.
- Mei, J., Xiao, C., Szepesvari, C., and Schuurmans, D. (2020). On the global convergence rates of softmax policy gradient methods. arXiv preprint arXiv:2005.06392 .
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-level control through deep reinforcement learning. Nature , 518(7540):529-533.
- Mohammadi, H., Zare, A., Soltanolkotabi, M., and Jovanovi´ c, M. R. (2019). Convergence and sample complexity of gradient methods for the model-free linear quadratic regulator problem. arXiv preprint arXiv:1912.11899 .
- Nachum, O., Norouzi, M., Xu, K., and Schuurmans, D. (2017). Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems , pages 2775-2785.
- Nemirovsky, A. S. and Yudin, D. B. (1983). Problem complexity and method efficiency in optimization.
- Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. Mathematical programming , 120(1):221-259.
- Neu, G., Jonsson, A., and G´ omez, V. (2017). A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 .

- Peters, J., Mulling, K., and Altun, Y. (2010). Relative entropy policy search. In Twenty-Fourth AAAI Conference on Artificial Intelligence .
- Peters, J. and Schaal, S. (2008). Natural actor-critic. Neurocomputing , 71(7-9):1180-1190.
- Puterman, M. L. (2014). Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons.
- Schulman, J., Chen, X., and Abbeel, P. (2017a). Equivalence between policy gradients and soft Q-learning. arXiv preprint arXiv:1704.06440 .
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International conference on machine learning , pages 1889-1897.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017b). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
- Shani, L., Efroni, Y., and Mannor, S. (2019). Adaptive trust region policy optimization: Global convergence and faster rates for regularized MDPs. arXiv preprint arXiv:1909.02769 .
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. (2016). Mastering the game of Go with deep neural networks and tree search. nature , 529(7587):484-489.
- Sutton, R. S., McAllester, D. A., Singh, S. P., and Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , pages 1057-1063.
- Tu, S. and Recht, B. (2019). The gap between model-based and model-free methods on the linear quadratic regulator: An asymptotic viewpoint. In Conference on Learning Theory , pages 3036-3083.
- Vieillard, N., Kozuno, T., Scherrer, B., Pietquin, O., Munos, R., and Geist, M. (2020). Leverage the average: an analysis of regularization in RL. arXiv preprint arXiv:2003.14089 .
- Wang, L., Cai, Q., Yang, Z., and Wang, Z. (2019). Neural policy gradient methods: Global optimality and rates of convergence. arXiv preprint arXiv:1909.01150 .
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256.
- Williams, R. J. and Peng, J. (1991). Function optimization using connectionist reinforcement learning algorithms. Connection Science , 3(3):241-268.
- Wu, Y., Zhang, W., Xu, P., and Gu, Q. (2020). A finite time analysis of two time-scale actor critic methods.
- Xiao, C., Huang, R., Mei, J., Schuurmans, D., and M¨ uller, M. (2019). Maximum entropy Monte-Carlo planning. In Advances in Neural Information Processing Systems , pages 9520-9528.
- Xu, T., Wang, Z., and Liang, Y. (2020). Non-asymptotic convergence analysis of two time-scale (natural) actor-critic algorithms. arXiv preprint arXiv:2005.03557 .
- Zhang, K., Hu, B., and Basar, T. (2019a). Policy optimization for H 2 linear control with H ∞ robustness guarantee: Implicit regularization and global convergence. arXiv preprint arXiv:1910.09496 .
- Zhang, K., Koppel, A., Zhu, H., and Ba¸ sar, T. (2019b). Global convergence of policy gradient methods to (almost) locally optimal policies. arXiv preprint arXiv:1906.08383 .

## A Preliminaries

## A.1 Derivation of entropy-regularized NPG methods

This subsection establishes the equivalence between the update rules (15) and (18). Such derivations are inherently similar to the ones for the NPG update rule (without entropy regularization) (see, e.g., Agarwal et al. (2019)); we provide the proof here for pedagogical reasons.

First of all, let us follow the convention to introduce the advantage function A π τ : S × A → R of a policy π w.r.t. the entropy-regularized value function:

<!-- formula-not-decoded -->

with Q π τ defined in (11a), which reflects the gain one can harvest by executing action a instead of following the policy π in state s . This advantage function plays a crucial role in the calculation of policy gradients, due to the following fundamental relation (see Appendix C.6 for the proof):

Lemma 6. Under softmax parameterization (7) , the gradient of the regularized value function satisfies

<!-- formula-not-decoded -->

for any ( s, a ) ∈ S × A , where c ( s ) := ∑ a π θ ( a | s ) w s,a is some function depending only on s .

<!-- formula-not-decoded -->

It is worth highlighting that the search direction of NPG, given in (62b), is invariant to the choice of ρ . With the above calculations in place, it is seen that for any s ∈ S , the regularized NPG update rule (15) results in a policy update as follows

<!-- formula-not-decoded -->

where we use A ( t ) τ to abbreviate A π ( t ) τ . Here, (i) uses the definition of the softmax policy, (ii) comes from the update rule (15), (iii) is a consequence of (62b) (since c ( · ) does not depend on a ), whereas (iv) results from the definition (61) and the fact that V π τ ( · ) is not dependent on a . This validates the equivalence between (15) and (18).

## A.2 Basic facts about the function log( ‖ exp( θ ) ‖ 1 )

In the current paper, we often encounter the function log ( ‖ exp( θ ) ‖ 1 ) := log ( ∑ 1 ≤ a ≤|A| exp( θ a ) ) for any vector θ = [ θ a ] 1 ≤ a ≤|A| ∈ R |A| . To facilitate analysis, we single out several basic properties concerning this function, which will be used multiple times when establishing our main results. For notational convenience, we denote by π θ ∈ R |A| the softmax transform of θ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By straightforward calculations, the gradient of the function log ( ‖ exp( θ ) ‖ 1 ) is given by

Difference of log policies. In the analysis, we often need to control the difference of two policies, towards which the following bounds prove useful. To begin with, the mean value theorem reveals a Lipschitz continuity property (w.r.t. the /lscript ∞ norm): for any θ 1 , θ 2 ∈ R |A| ,

<!-- formula-not-decoded -->

where θ c is a certain convex combination of θ 1 and θ 2 , and the second line relies on (64). In addition, for any two vectors π θ 1 and π θ 2 defined w.r.t. θ 1 , θ 2 ∈ R |A| (see (63)), one has

<!-- formula-not-decoded -->

where log( · ) denotes entrywise operation. To justify (66), we observe from the definition (63) that

‖ log π θ 1 -log π θ 2 ‖ ∞ ≤ ‖ θ 1 -θ 2 ‖ ∞ + ∣ ∣ ∣ log ( ‖ exp( θ 1 ) ‖ 1 ) -log ( ‖ exp( θ 2 ) ‖ 1 ) ∣ ∣ ∣ ≤ 2 ‖ θ 1 -θ 2 ‖ ∞ , where the last inequality is a consequence of (65).

## B Proof for the bandit case (Proposition 1)

We start by defining an auxiliary sequence ξ ( t ) ∈ R |A| ( t ≥ 0) recursively as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When combined with (21), it is easily seen that π ( t ) ( · ) ∝ ξ ( t ) ( · ) and, as a result, π ( t ) = ξ ( t ) / ∥ ∥ ξ ( t ) ∥ ∥ 1 . By construction, the auxiliary sequence satisfies the following property thus indicating that

This taken together with the optimal policy π /star τ = softmax ( r/τ ) ∝ exp( r/τ ) leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first line follows from the inequality (66), the second line follows from the expression (67), whereas the last line follows from the form of π /star τ . We have thus completed the proof of Proposition 1.

## C Proof for key lemmas

## C.1 Proof of Lemma 1

To begin with, the regularized NPG update rule (see (18) in Algorithm 1) indicates that

<!-- formula-not-decoded -->

where Z ( t ) is some quantity depending only on the state s (but not the action a ). Rearranging terms gives

<!-- formula-not-decoded -->

This in turn allows us to express V ( t ) τ ( s 0 ) for any s 0 ∈ S as follows

<!-- formula-not-decoded -->

where the first identity makes use of the definitions (8) and (11a), the second line follows from (69), the third line relies on the definition of the KL divergence, and the last line follows since Z ( t ) ( s ) does not depend on a . Invoking (69) again to rewrite log Z ( t ) ( s 0 ) appearing in the first term of (70), we reach

<!-- formula-not-decoded -->

where the second line uses the definition of the KL divergence, and the third line expands Q ( t ) τ using the definition (11a).

To finish up, applying the above relation (71) recursively to expand V ( t ) τ ( s i ) ( i ≥ 1), we arrive at

<!-- formula-not-decoded -->

where the second line follows since the regularized value function V ( t +1) τ can be viewed as the value function of π ( t +1) with adjusted rewards r ( t +1) τ ( s, a ) := r ( s, a ) -τ log π ( t +1) ( a | s ). Averaging the initial state s 0 over the distribution ρ concludes the proof.

## C.2 Proof of Lemma 2

In the sequel, we prove each claim in Lemma 2 in order.

Proof of Eqn. (36) . Jensen's inequality tells us that: for any s ∈ S one has

<!-- formula-not-decoded -->

where in the second line, equality is attained if π ( ·| s ) ∝ exp( Q ( s, · ) /τ ). This immediately gives rise to

<!-- formula-not-decoded -->

Proof of Eqn. (37) . Recall the characterization of π /star τ and V /star τ established in Nachum et al. (2017):

<!-- formula-not-decoded -->

Substitution into the expression (36) tells us that for any ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second line results from (74b), and the last line follows from the definition of the soft Q-function.

Proof of Eqn. (38) . Invoking again the expression (36), we can demonstrate that for any Q 1 and Q 2 ,

<!-- formula-not-decoded -->

holds for all ( s, a ) ∈ S × A , where the inequality follows from the Lipschitz property (65).

## C.3 Proof of Lemma 3

For any state-action pair ( s, a ) ∈ S × A , we observe that

<!-- formula-not-decoded -->

where the first step invokes the definition (11a) of Q τ , and the second step is due to the expression (74b) of V /star τ . To continue, recall that π ( t ) is related to ξ ( t ) as

<!-- formula-not-decoded -->

which can be seen by comparing (42) with (18). This in turn leads to where the second line comes from (42b). By plugging (77) into (75) we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- In view of the property (65), the first term on the right-hand side of (78) can be bounded by τ log (∥ ∥ exp ( Q /star τ ( s ′ , · ) /τ )∥ ∥ 1 ) -τ log (∥ ∥ ξ ( t +1) ( s ′ , · ) ∥ ∥ 1 ) ≤ ∥ ∥ Q /star τ -τ log ξ ( t +1) ∥ ∥ ∞ . · Regarding the second term, the monotonicity (33) of the soft Q-function allows us to derive

∥ ∥ for any ( s, a ) ∈ S × A . Here, (i) follows by construction (42b), (ii) invokes the monotonicity property (33) (so that Q ( t ) τ ≥ Q ( t -1) τ ), and (iii) follows by repeating the arguments (i) and (ii) recursively.

<!-- formula-not-decoded -->

Combining the preceding two bounds with the expression (78), we conclude that for any ( s, a ) ∈ S × A , thus concluding the proof.

<!-- formula-not-decoded -->

## C.4 Proof of Lemma 4

Recall that, in this scenario, the policies are updated using inexact policy evaluation via (26), namely,

<!-- formula-not-decoded -->

̂ where ̂ Z ( t ) ( s ) := ∑ a ′ π ( t ) ( a ′ | s ) 1 -ητ 1 -γ exp ( η 1 -γ ̂ Q ( t ) τ ( s, a ′ ) ) . To facilitate analysis, we further introduce another auxiliary policy sequence { ˘ π ( t ) } , which corresponds to the policy update as if we had access to exact soft Q-function of π ( t ) in the t -th iteration; this is defined as

<!-- formula-not-decoded -->

where we abuse the notation by letting Z ( t ) ( s ) := ∑ a ′ π ( t ) ( a ′ | s ) 1 -ητ 1 -γ exp ( η 1 -γ Q ( t ) τ ( s, a ′ ) ) . It is worth emphasizing that ˘ π ( t +1) is produced on the basis of π ( t ) as opposed to ˘ π ( t ) ; it should be viewed as a one-step perfect update from a given policy π ( t ) .

We first make note of the following fact: for any step size 0 &lt; η ≤ (1 -γ ) /τ , it follows from (66) together with the construction (80) and (81) - that

<!-- formula-not-decoded -->

Next, let us recall the inequality (70) in the proof of Lemma 1 under exact policy evaluation ˘ π ( t +1) ( ·| s ); when applied to the current setting, it essentially indicates that

<!-- formula-not-decoded -->

where the last step follows since the quantity Z ( t ) ( s ) does not depend on a at all. In order to control the first term of (83), we invoke the definition of ˘ π ( t +1) ( ·| s ) to show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last identity makes use of the relation Q ( t ) τ ( s 0 , a 0 ) = r ( s 0 , a 0 ) + γ E s 1 ∼ P ( ·| s 0 ,a 0 ) [ V ( t ) τ ( s 1 ) ] . Invoking the above inequality recursively as in the expression (72) (see Lemma 1), we can expand it to establish

## C.5 Proof of Lemma 5

First of all, we follow the definition (8) of the entropy-regularized value function to deduce that

Here, (i) is due to the definition V ( t ) τ ( ρ ) = E s 0 ∼ ρ [ V ( t ) τ ( s 0 ) ] , (ii) follows by aggregating terms corresponding to the same state-action pair and the definition of d π /star τ ρ (cf. (5)), whereas (iii) results from the definition (11a) of the regularized Q-function.

<!-- formula-not-decoded -->

To continue, we shall attempt to control each part of (85) separately. To begin with, observe that the first part of (85) can be bounded by Jensen's inequality, namely,

With regards to the second part of (85), it is seen from the definition of π ( t +1) (cf. (17)) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus allowing one to derive

<!-- formula-not-decoded -->

where (i) relies on the identity (87). Substituting the inequalities (86) and (88) into the expression (85), we can demonstrate with a little algebra that

<!-- formula-not-decoded -->

## C.6 Proof of Lemma 6

The results of this lemma, or some similar versions, have appeared in prior work (e.g. Mei et al. (2020, Lemma 10) and Agarwal et al. (2020b, Lemma 5.6)). We include the proof here primarily for the sake of self-completeness.

Proof of Eqn. (62a) . The policy gradient of the unregularized value function V π θ ( s 0 ) is well-known as the policy gradient theorem (Sutton et al., 2000). Here, we deal with a slightly different variant - an entropy-regularized value function V π θ τ ( s 0 ) in the expression (2) with the softmax policy parameterization in (7). Invoking the Bellman equation and recognizing that V π θ τ ( s 0 ) can be viewed as an unregularized value function with instantaneous rewards r ( s, a ) -τ log π θ ( a | s ) for any ( s, a ), we obtain

<!-- formula-not-decoded -->

where (i) relies on the definition (11a) of Q π θ τ , and (ii) makes use of the identity

<!-- formula-not-decoded -->

as well as the definition (11a) of Q π θ τ . Given that

<!-- formula-not-decoded -->

and that r ( s, a ) is independent of θ , one can continue the above derivative to reach

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Repeating the above calculations recursively, we arrive at

<!-- formula-not-decoded -->

where the second line follows by aggregating the terms corresponding to the same state-action pair, and the third line invokes the definition (61) of A π θ τ . To see why the last line holds, invoke (89) to reach

<!-- formula-not-decoded -->

Further, it is easily seen that under the softmax parametrization in (7),

<!-- formula-not-decoded -->

for any ( s, a ) , ( s ′ , a ′ ) ∈ S × A . Combining with (90), it further implies that

<!-- formula-not-decoded -->

where (i) follows from E a ′ ∼ π θ ( ·| s ′ ) A π θ τ ( s ′ , a ′ ) = ∑ a ′ π θ ( a ′ | s ′ ) A π θ τ ( s ′ , a ′ ) = 0 due to the definition (61). The proof regarding V π θ τ ( ρ ) can be obtained by averaging the initial state s 0 over the distribution ρ .

Proof of Eqn. (62b) . In order to establish (62b), a crucial observation is that w θ := ( F θ ρ ) † ∇ θ V π θ τ ( ρ ) is exactly the solution to the following least-squares problem

From the definition (14) of the Fisher information matrix, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any fixed vector w = [ w s,a ] ( s,a ) ∈S×A . As a result, for any ( s, a ) ∈ S × A one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) makes use of the derivative calculation (91), and we define c ( s ) := ∑ a π θ ( a | s ) w s,a . Consequently, the objective function of (92) can be written as which is minimized by choosing w s,a = 1 1 -γ A π θ τ ( s, a ) + c ( s ) for all ( s, a ) ∈ S × A . This concludes the proof.

## D Convergence guarantees for CPI-style policy updates

Employing the SPI update as the improved policy, we arrive at the following CPI-style update

<!-- formula-not-decoded -->

Here, π ( t +1) corresponds to a one-step SPI update from π ( t ) , namely,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we denote as usual. Here, β ∈ (0 , 1] is a parameter that controls the 'conservatism' of the updates. We characterize the convergence rate of this update rule (93) in the following theorem.

Theorem 4 (Linear convergence of CPI-style updates) . For any 0 &lt; β ≤ 1 , the update rule (93) satisfies where µ /star τ is the stationary distribution defined in (29) .

<!-- formula-not-decoded -->

According to Theorem 4, it takes the CPI-style policy update (93) at most

<!-- formula-not-decoded -->

iterations to reach V /star τ ( ρ ) -V ( t ) τ ( ρ ) ≤ /epsilon1 . As it turns out, the CPI-style update rule can be analyzed using our framework through the following performance improvement lemma, which is an adaptation of Lemma 1. In what follows, we use Q ( t +1) τ and V ( t +1) τ to abbreviate Q π ( t +1) τ and V π ( t +1) τ , respectively.

Lemma 7 (Performance improvement of CPI-style updates) . Consider the policy update rule (93a) with any β ∈ (0 , 1] . For any distribution ρ , one has

<!-- formula-not-decoded -->

Proof. See Appendix D.1.

Combining the above result with Lemma 5 and following a similar approach to (60) give

<!-- formula-not-decoded -->

Here, (i) arises from Lemma 7, (ii) employs the pre-factor ∥ ∥ d π /star τ ρ /d ( t +1) ρ ∥ ∥ -1 ∞ to accommodate the change of distributions, whereas (iii) follows from Lemma 5 and the constraint that 0 ≤ η ≤ 1 -γ τ . By taking ρ to be the stationary distribution µ /star τ (cf. (29)), one has

<!-- formula-not-decoded -->

where we have used d π /star τ µ /star τ = µ /star τ (cf. (29)) and d ( t +1) µ /star τ ≥ (1 -γ ) µ /star τ in the second step. This immediately concludes the proof.

## D.1 Proof of Lemma 7

First of all, we claim that

<!-- formula-not-decoded -->

which we shall establish momentarily. Since the KL divergence KL ( π ( ·| s ) ‖ π ( t +1) ( ·| s ) ) is convex in π ( ·| s ) (Cover, 1999), the update rule (93a) together with Jensen's inequality necessarily implies that

Substituting the above inequality into (96) allows us to conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The rest of this proof is then dedicated to establishing the claim (96), which is similar to the proof of Lemma 1. To begin with, we express V ( t ) τ ( s 0 ) as follows

<!-- formula-not-decoded -->

where the first line makes use of the definitions (8) and (11a), the second line follows from (93), the third line uses the definition of the KL divergence, and the last line follows since Z ( t ) ( s 0 ) does not depend on a . To continue, we subtract and add τ KL ( π ( t +1) ( ·| s 0 ) ‖ π ( t +1) ( ·| s 0 ) ) to obtain

<!-- formula-not-decoded -->

Here, the first step relies on the definition of KL divergence, the second step comes from (93), while the last step is obtained by using the relation Q ( t ) τ ( s 0 , a 0 ) = r ( s 0 , a 0 ) + γ E s 1 ∼ P ( ·| s 0 ,a 0 ) [ V ( t ) τ ( s 1 ) ] and then invoking the above equality recursively as in the expression (72) (see Lemma 1). Averaging the equality over the initial state distribution s 0 ∼ ρ thus establishes the claim (96).

## E Proof for approximate entropy-regularized NPG (Theorem 2)

In this section, we complete the proofs of Theorem 2 in Section 4.3, which consists of (i) establishing the linear system in (58) and (ii) extracting the convergence rate from (58).

Step 1: establishing the linear system (58) . In what follows, we shall justify the linear system relation by checking each row separately.

<!-- formula-not-decoded -->

Taken together with the triangle inequality and the assumption ∥ Q ( t ) τ -Q ( t ) τ ∥ ∞ ≤ δ , this gives

(2) Bounding -min s,a ( Q ( t +1) τ ( s, a ) -τ log ̂ ξ ( t +1) ( s, a ) ) . Invoking the definition (57b) of ̂ ξ ( t +1) again implies that for any ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from ∥ ∥ Q ( t ) τ -̂ Q ( t ) τ ∥ ∥ ∞ ≤ δ and (56). Taking the maximum over ( s, a ) ∈ S ×A on both sides and using the definition α = 1 -ητ 1 -γ yield

(3) Bounding ∥ ∥ Q /star τ -Q ( t +1) τ ∥ ∥ ∞ . Following the same arguments as for (78), we obtain where the last line follows from (65). By plugging (97) and (98) into the above inequality, we arrive at the claimed bound regarding this term.

<!-- formula-not-decoded -->

Step 2: deducing convergence guarantees from the linear system (58) . We start by pinning down the eigenvalues and eigenvectors of the matrix B . Specifically, the three eigenvalues can be calculated as

<!-- formula-not-decoded -->

whose corresponding eigenvectors are given respectively by

<!-- formula-not-decoded -->

With some elementary computation, one can show that z 0 and b introduced in (59) can be related to the eigenvectors of B in the following way:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c z is some scalar whose value is immaterial since the eigenvalue corresponding to v 3 is λ 3 = 0, and the last line follows from the same reasoning for (52). Another userful identity is:

<!-- formula-not-decoded -->

With these preparations in place, we can now invoke the recursion relationship (58) and the non-negativity of B to obtain

<!-- formula-not-decoded -->

where the eigenvalues and eigenvectors of B are given in (99) and (100), respectively, and the second inequality relies on (101) and (102). Note that we are only interested in the first two entries of the vector z t . Since the first two entries of the eigenvector v 2 are non-positive, we can safely drop the term involving v 2 in the above inequality to obtain

<!-- formula-not-decoded -->

When it comes to the log policies, we recall again the fact that π ( t ) is related to ξ ( t ) as

Invoking the elementary property (66), we reach

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This together with the bound on ∥ ∥ Q /star τ -τ log ̂ ξ ( t +1) ∥ ∥ ∞ in (103) establishes our claim for ∥ ∥ log π /star τ -log π ( t +1) ∥ ∥ ∞ .

## F Proof for local quadratic convergence (Theorem 3)

Assuming that the policy π ( t ) obeys Condition (30), we can control the difference of the corresponding discounted state visitation probabilities in terms of the sub-optimality gap w.r.t. the log policy. This is stated in the following lemma, whose proof is deferred to Section F.1.

Lemma 8. Consider any policy π satisfying ‖ log π -log π /star τ ‖ ∞ ≤ 1 . It follows that

In particular, by taking ρ = µ /star τ one has

<!-- formula-not-decoded -->

First, by virtue of the SPI update rule (17) and the inequality (66), it is guaranteed that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality comes from a change of distributions argument. Armed with Lemma 8 and the inequality (105), we arrive at

Substitution into (60) gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.1 Proof of Lemma 8

For any policy π , denote by P π ∈ R |S|×|S| the state transition matrix induced by π as follows

<!-- formula-not-decoded -->

For any policy π satisfying ‖ log π -log π /star τ ‖ ∞ ≤ 1 , we develop an upper bound on ∣ ∣ ∣ [ P π -P π /star τ ] s,s ′ ∣ ∣ ∣ as follows where (i) uses the assumption ‖ log π /star -log π ‖ ∞ ≤ 1 together with the elementary inequality | x | ≤ ( e -1) | log(1 + x ) | when -1 &lt; x ≤ e -1. With the preceding bound in mind, we can demonstrate that

<!-- formula-not-decoded -->

Here and throughout, we overload the notation | z | for any vector z ∈ R |S| to denote [ | z i | ] 1 ≤ i ≤|S| .

<!-- formula-not-decoded -->

In addition, the definitions of d π ρ and d π τ ρ admit the following matrix-vector representation:

/star

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus allowing one to derive

<!-- formula-not-decoded -->

This together with the non-negativity of the matrix ( I -γP π ) -1 (Li et al., 2020b, Lemma 7) enables the following bound where the last inequality results from (108).

Furthermore, we make the observation that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line arises from the expression (109). As a result, we establish the claimed bound

<!-- formula-not-decoded -->