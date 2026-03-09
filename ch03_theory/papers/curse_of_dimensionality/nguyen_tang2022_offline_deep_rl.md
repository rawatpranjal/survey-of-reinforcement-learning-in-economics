## On Sample Complexity of Offline Reinforcement Learning with Deep ReLU Networks in Besov Spaces

Thanh Nguyen-Tang

Department of Computer Science, Johns Hopkins University

Sunil Gupta &amp; Hung Tran-The &amp; Svetha Venkatesh

Applied AI Institute, Deakin University

Reviewed on OpenReview:

https: // openreview. net/ forum? id= LdEm0umNcv

## Abstract

Offline reinforcement learning (RL) leverages previously collected data for policy optimization without any further active exploration. Despite the recent interest in this problem, its theoretical results in neural network function approximation settings remain elusive. In this paper, we study the statistical theory of offline RL with deep ReLU network function approximation. In particular, we establish the sample complexity of n = ˜ O ( H 4+4 d α κ 1+ d α µ /epsilon1 -2 -2 d α ) for offline RL with deep ReLU networks, where κ µ is a measure of distributional shift, H = (1 -γ ) -1 is the effective horizon length, d is the dimension of the state-action space, α is a (possibly fractional) smoothness parameter of the underlying Markov decision process (MDP), and /epsilon1 is a user-specified error. Notably, our sample complexity holds under two novel considerations: the Besov dynamic closure and the correlated structure. While the Besov dynamic closure subsumes the dynamic conditions for offline RL in the prior works, the correlated structure renders the prior works of offline RL with general/neural network function approximation improper or inefficient in long (effective) horizon problems. To the best of our knowledge, this is the first theoretical characterization of the sample complexity of offline RL with deep neural network function approximation under the general Besov regularity condition that goes beyond the linearity regime in the traditional Reproducing Hilbert kernel spaces and Neural Tangent Kernels.

## 1 Introduction

Offline reinforcement learning (Lange et al., 2012; Levine et al., 2020) is a practical paradigm of reinforcement learning (RL) where logged experiences are abundant but a new interaction with the environment is limited or even prohibited. The fundamental offline RL problems concern with how well previous experiences could be used to evaluate a new target policy, known as off-policy evaluation (OPE) problem, or to learn the optimal policy, known as off-policy learning (OPL) problem. We study these offline RL problems with infinitely large state spaces, where the agent must rely on function approximation such as deep neural networks to generalize across states from an offline dataset. Such problems form the core of modern RL in practical settings (Levine et al., 2020; Kumar et al., 2020; Singh et al., 2020; Zhang et al., 2022).

Prior sample-efficient results in offline RL mostly focus on tabular environments with small finite state spaces (Yin &amp; Wang, 2020; Yin et al., 2021; Yin &amp; Wang, 2021a), but as these methods scale with the number of states, they are infeasible for the settings with infinitely large state spaces. While this tabular setting has been extended to large state spaces via linear environments (Duan &amp; Wang, 2020; Jin et al., 2020b; Xiong et al., 2022; Yin et al., 2022; Nguyen-Tang et al., 2022b), the linearity assumption often does not hold for many RL problems in practice. Theoretical guarantees for offline RL with general and deep neural network function approximations have also been derived, but these results are either inadequate or relatively disconnected from the regularity structure of the underlying MDP. In particular, while the finite-sample results for offline thnguyentang@gmail.com

RL with general function approximation (Munos &amp; Szepesvári, 2008; Le et al., 2019) depend on an inherent Bellman error which could be uncontrollable in practice, the other analysis in the neural network function approximation in Yang et al. (2019) relies on a data splitting technique to deal with the correlated structures arisen in value regression for offline RL and use a relatively strong dynamic assumption. Recent works have studied offline RL with function approximations in Reproducing Hilbert kernel spaces (RHKS) (Jin et al., 2020b) and Neural Tangent Kernels (NTK) (Nguyen-Tang et al., 2022a). However, these function classes also have (approximately) linear structures (in terms of the underlying features) which make their analysis similar to the linear case. Moreover, the smoothness assumption imposed by the RKHS is often strong for several practical cases while the NTK analysis requires a extremely wide neural net (the network width scales with n 10 for the NTK case in Nguyen-Tang et al. (2022a) versus only n 2 / 5 (Proposition 5.1) in the current work). Recent works (Xie et al., 2021a; Zhan et al., 2022; Chen &amp; Jiang, 2022; Uehara &amp; Sun, 2021) have considered offline RL with general function approximation with weaker data coverage assumption. However, they assumed the function class is finite and did not consider the (Besov) regularity of the underlying MDP. Thus, to our knowledge, no prior work has dedicated to study a comprehensive and adequate analysis of the statistical efficiency for offline RL with neural network function approximation in Besov spaces. Thus, it is natural to ask:

Is offline RL sample-efficient with deep ReLU network function approximation beyond the (approximate-) linear regime imposed by RKHS and NTK?

Our contributions. In this paper, we provide a statistical theory of both OPE and OPL with neural network function approximation in a broad generality that is beyond the (approximate-) linear regime imposed by RKHS and NTK. In particular, our contributions, which are summarized in Table 1 and will be discussed in details in Section 5, are:

- First, we achieve a generality for the guarantees of offline RL with neural network function approximation via two novel considerations: (i) We introduce a new structural condition namely Besov dynamic closure which subsumes the existing dynamic conditions for offline RL with neural network function approximation and even includes MDPs that need not be continuous, differentiable or spatially homogeneous in smoothness; (ii) We take into account the correlated structure of the value estimate produced by a regression-based algorithm from the offline data. This correlated structure plays a central role in the statistical efficiency of an offline algorithm; yet the prior works improperly ignore this structure (Le et al., 2019) or avoid it using an data splitting approach (Yang et al., 2019).
- Second, we prove that an offline RL algorithm based on fitted-Q iteration (FQI) can achieve the sample complexity of n = ˜ O ( H 4+4 d α κ 1+ d α µ /epsilon1 -2 -2 d α ) where κ measures the distributional shift in the offline data, H = (1 -γ ) -1 is the effective horizon length, d is the input dimension, α is a smoothness parameter of the underlying MDP, and /epsilon1 is a user-specified error. Notably, our guarantee holds under our two novel considerations above that generalize the condition in Yang et al. (2019) and do not require data splitting in Yang et al. (2019). Moreover, our analysis also corrects the technical mistake in Le et al. (2019) that ignores the correlated structure of offline value estimate.

Problem scope. We emphasize that the present work focuses on statistical theory of offline RL with neural network function approximation in Besov spaces where we analyze a relatively standard algorithm, FQI. Regarding the empirical effectiveness of FQI with neural network function approximation for offline RL, we refer the readers to the empirical study in Voloshin et al. (2019). Finally, this work is an extension of our workshop paper (Nguyen-Tang et al., 2021).

Notations. Denote ‖ f ‖ p,µ := (∫ X | f | p dµ ) 1 /p , and for simplicity, we write ‖ · ‖ µ for ‖ · ‖ p,µ when p = 2 and write ‖ · ‖ p for ‖ · ‖ p,µ if µ is the Lebesgue measure. Let L p ( X ) be the space of measurable functions for which the p -th power of the absolute value is Lebesgue integrable, i.e. L p ( X ) = { f : X → R |‖ f ‖ p &lt; ∞} . Denote by ‖ · ‖ 0 the 0-norm, i.e., the number of non-zero elements, and a ∨ b = max { a, b } . For any two

Table 1: Comparison among existing representative works of FQI estimators for offline RL with function approximation under a uniform data coverage assumption. Here S and A are the cardinalities of the state and action space when they are finite, κ is a measure of distribution shift (which can be defined slightly different in different works), /epsilon1 is the user-specified precision, d is the dimension of the input space, α is the smoothness parameter of the underlying MDP, and H := (1 -γ ) -1 is the effective horizon length.

| Work               | Function   | Regularity   | Tasks   | Sample complexity                                                      | Remark            |
|--------------------|------------|--------------|---------|------------------------------------------------------------------------|-------------------|
| Yin & Wang (2020)  | Tabular    | Tabular      | OPE     | ˜ O ( κ · H 4 · /epsilon1 - 2 · ( SA ) 2 )                             | -                 |
| Duan & Wang (2020) | Linear     | Linear       | OPE     | ˜ O ( κ · H 4 · /epsilon1 - 2 · d )                                    | -                 |
| Le et al. (2019)   | General    | General      | OPE/OPL | N/A                                                                    | improper analysis |
| Yang et al. (2019) | ReLU nets  | Hölder       | OPL     | ˜ O ( κ 2+ d α · H 5+2 d α · /epsilon1 - 2 - d α · log( H 2 //epsilon1 | data splitting    |
| This work          | ReLU nets  | Besov        | OPE/OPL | ˜ O κ 1+ d α · H 4+4 d α · /epsilon1 - 2 - 2 d α                       | data reuse        |

(

)

real-valued functions f and g , we write f ( · ) /lessorsimilar g ( · ) if there is an absolute constant c independent of the function parameters ( · ) such that f ( · ) ≤ c · g ( · ). We write f ( · ) /equivasymptotic g ( · ) if f ( · ) /lessorsimilar g ( · ) and g ( · ) /lessorsimilar f ( · ). We write f ( · ) /similarequal g ( · ) if there exists an absolute constant c such that f ( · ) = c · g ( · ). We denote H := (1 -γ ) -1 which is the effective horizon length in the discounted MDP and is equivalent to the horizon (episode) length in finite-horizon MDPs.

## 2 Related Work

Offline RL with tabular representation. The majority of the theoretical results for offline RL focus on tabular MDP where the state space is finite and an importance sampling -related approach is possible (Precup et al., 2000; Dudík et al., 2011; Jiang &amp; Li, 2015; Thomas &amp; Brunskill, 2016; Farajtabar et al., 2018; Kallus &amp; Uehara, 2019). The main drawback of the importance sampling-based approach is that it suffers high variance in long horizon problems. The high variance problem can be mitigated by direct methods where we employ models to estimate the value functions or the transition kernels. We focus on direct methods in this work. For tabular MDPs with some uniform data-visitation measure d m &gt; 0, a nearoptimal sample complexity bound of O ( H 3 d m //epsilon1 2 ) and O ( H 2 d m //epsilon1 2 ) were obtained for time-inhomogeneous tabular MDP (Yin et al., 2021) and for time-homogeneous tabular MDP (Yin &amp; Wang, 2021b; Ren et al., 2021), respectively. With the single-concentrability assumption, a tight bound of O ( H 3 SC ∗ //epsilon1 2 ) was achieved (Xie et al., 2021b; Rashidinejad et al., 2021), where H ≈ 1 / (1 -γ ) is the episode length. Yin &amp; Wang (2021c) introduced intrinsic offline bound that further incorporates instance-dependent quantities. Shi et al. (2022) obtained the minimax rate with model-free methods. Wang et al. (2022) derived gap-dependent bounds for offline RL in tabular MDPs.

Offline RL with linear function approximation. Offline RL with function approximation often follow two algorithmic approaches: Fitted Q-iteration (FQI) (Bertsekas &amp; Tsitsiklis, 1995; Jong &amp; Stone, 2007; Lagoudakis &amp; Parr, 2003; Grünewälder et al., 2012; Munos, 2003; Munos &amp; Szepesvári, 2008; Antos et al., 2008; Tosatto et al., 2017; Le et al., 2019), and pessimism principle (Buckman et al., 2020), where the former requires a uniform data coverage and the latter only needs a sufficient coverage over the target policy. Duan &amp; Wang (2020) studied fitted-Q iteration algorithm in linear MDPs. Wang et al. (2020) highlighted the necessity of strong structural assumptions (e.g., on low distributional shift or strong dynamic condition beyond realizability) for sample-efficient offline RL with linear function approximation suggesting that only realizability and strong uniform data coverage are not sufficient for sample-efficient offline RL. Jin et al. (2020b) brought pessimism principle into offline linear MDPs. Nguyen-Tang et al. (2022b) derived a minimax rate of 1 / √ n for offline linear MDPs under a partial data coverage assumption and obtained the instancedependent rate of 1 /n when the gap information is available. Xiong et al. (2022); Yin et al. (2022) used variance reduction and data splitting to tighten the bound of Jin et al. (2020b). Xie et al. (2021a) proposed

Bellman-consistent condition with general function approximation which improves the bound of Jin et al. (2020b) by a factor of √ d when realized to finite action space and linear MDPs. Chen et al. (2021) studied sample complexity of FQI in linear MDPs and derive a lower bound for this setting.

Offline RL with non-linear function approximation. Beyond linearity, some works study offline RL in general or nonparametric function approximation, either with FQI estimators (Munos &amp; Szepesvári, 2008; Le et al., 2019; Duan et al., 2021a;b; Hu et al., 2021), pessimistic estimators (Uehara &amp; Sun, 2021; Nguyen-Tang et al., 2022a; Jin et al., 2020b), or minimax estimators (Uehara et al., 2021), where Uehara et al. (2021) also realized their minimax estimators to the neural network function approximation, Nguyen-Tang et al. (2022a) considered offline contextual bandits with Neural Tangent Kernels (NTK), and Jin et al. (2020b) considered the pessimistic value iteration algorithm with Reproducing Kernel Hilbert Space (RKHS) in their extended version. Our work is different from these aforementioned works in that we analyze the fundamental FQI estimators with neural network function approximation under the Besov regularity condition that is much more general than RKHS and NTK. We also further emphasize that even that RKHS and NTK spaces are non-linear function approximation, the functions in those spaces are linear in terms of an underlying feature space, making the analysis for these spaces akin to the case of linear function approximation. Yang et al. (2019) also considered deep neural network approximation. In particular, Yang et al. (2019) focused on analyzing deep Q-learning using a disjoint fold of offline data for each iteration. Such approach is considerably sample-inefficient for offline RL with long (effective) horizon. In addition, they rely on a relatively restricted smoothness assumption of the underlying MDPs that hinders their results from being widely applicable in more general settings. Recently, other works (Xie et al., 2021a; Zhan et al., 2022; Chen &amp; Jiang, 2022; Uehara &amp; Sun, 2021) considered offline RL with general function approximation and imposed weaker data coverage assumption by using pessimistic algorithms. Their algorithms are more involved than FQI but did not study the effect of the regularity of the underlying MDP on the sample complexity of offline RL. They also assume that the function class is finite which is not applicable to neural network function approximation. Since the first version of our paper appeared online, there have been several other works establishing sample complexity of reinforcement learning in Besov spaces for various problem settings, including /epsilon1 -greedy exploration for online setting with Markovian data (Liu et al., 2022) and off-policy evaluation on low-dimensional manifolds (Ji et al., 2022).

## 3 Preliminaries

We consider a discounted Markov decision process (MDP) with possibly infinitely large state space S , continuous action space A , initial state distribution ρ ∈ P ( S ), transition operator P : S×A → P ( S ), reward distribution R : S × A → P ([0 , 1]), and a discount factor γ ∈ [0 , 1). For notational simplicity, we assume that X := S × A ⊆ [0 , 1] d .

A policy π : S → P ( A ) induces a distribution over the action space conditioned on states. The Q -value function for policy π at state-action pair ( s, a ), denoted by Q π ( s, a ) ∈ [0 , 1], is the expected discounted total reward the policy collects if it initially starts in the state-action pair, i.e.,

<!-- formula-not-decoded -->

where r t ∼ R ( s t , a t ) , a t ∼ π ( ·| s t ), and s t ∼ P ( ·| s t -1 , a t -1 ). The value of policy π is V π = E s ∼ ρ,a ∼ π ( ·| s ) [ Q π ( s, a )], and the optimal value is V ∗ = max π V π where the maximization is taken over all stationary policies. Alternatively, the optimal value V ∗ can be obtained via the optimal Q -function Q ∗ = max π Q π as V ∗ = E s ∼ ρ [max a Q ∗ ( s, a )]. Denote by T π and T ∗ the Bellman operator and the optimality Bellman operator, respectively, i.e., for any f : S × A → R

<!-- formula-not-decoded -->

we have T π Q π = Q π and T ∗ Q ∗ = Q ∗ .

Offline regime. We consider the offline RL setting where a learner cannot explore the environment but has access to a fixed logged data D = { ( s i , a i , s ′ i , r i ) } n i =1 collected a priori by certain behaviour policy η . For simplicity, we assume that { s i } n i =1 are independent and η is stationary. Equivalently, { ( s i , a i ) } n i =1 are i.i.d. samples from the normalized discounted stationary distribution over state-actions with respect to η , i.e., ( s i , a i ) i.i.d. ∼ µ ( · , · ) := (1 -γ ) ∑ ∞ t =0 γ t P ( s t = · , a t = ·| ρ, η ) where s ′ i ∼ P ( ·| s i , a i ) and a i ∼ η ( ·| s i ). This assumption is relatively standard in the offline RL setting (Munos &amp; Szepesvári, 2008; Chen &amp; Jiang, 2019a; Yang et al., 2019). The goals of Off-Policy Evaluation (OPE) and Off-Policy Learning (OPL) are to estimate V π and V ∗ , respectively from D . The performance of OPE and OPL estimates are measured via sub-optimality gaps defined as follows.

For OPE Task. Given a fixed target policy π , for any value estimate ˆ V computed from the offline data D , the sub-optimality of OPE is defined as

<!-- formula-not-decoded -->

For OPL Task. For any estimate ˆ π of the optimal policy π ∗ that is learned from the offline data D , we define the sup-optimality of OPL as

<!-- formula-not-decoded -->

## 3.1 Deep ReLU Networks as Function Approximation

In practice, the state space is often very large and complex, and thus function approximation is required to ensure generalization across different states. Deep neural networks with the ReLU activation offer a rich class of parameterized functions with differentiable parameters. Deep ReLU networks are state-of-the-art in many applications, e.g., Krizhevsky et al. (2012); Mnih et al. (2015), including offline RL with deep ReLU networks that can yield superior empirical performance (Voloshin et al., 2019). In this section, we describe the architecture of deep ReLU networks and the associated function space which we use throughout this paper. Specifically, a L -height, m -width ReLU network on R d takes the form of

<!-- formula-not-decoded -->

where W ( L ) ∈ R 1 × m , b ( L ) ∈ R , W (1) ∈ R m × d , b (1) ∈ R m , W ( l ) ∈ R m × m , b ( l ) ∈ R m , ∀ 1 &lt; l &lt; L , θ = { W ( l ) , b ( l ) } 1 ≤ l ≤ L , and σ ( x ) = max { x, 0 } is the (element-wise) ReLU activation. We define Φ( L, m, S, B ) as the space of L -height, m -width ReLU functions f L,m θ ( x ) with sparsity constraint S , and norm constraint B , i.e., ∑ L l =1 ( ‖ W ( l ) ‖ 0 + ‖ b ( l ) ‖ 0 ) ≤ S, max 1 ≤ l ≤ L ‖ W ( l ) ‖ ∞ ∨ ‖ b ( l ) ‖ ∞ ≤ B . Finally, for some L, m ∈ N and S, B ∈ (0 , ∞ ), we define the unit ball of ReLU network function space F NN as

<!-- formula-not-decoded -->

In nonparametric regressions, Suzuki (2018) showed that deep ReLU networks outperform any non-adaptive linear estimator due to their higher adaptivity to spatial inhomogeneity.

## 3.2 Besov spaces

Our new dynamic condition relies on the regularity of Besov spaces. There are several ways to characterize the smoothness in Besov spaces. Here, we pursue a characterization via multivariate moduli of smoothness as it is more intuitive, following Giné &amp; Nickl (2016).

Definition 3.1 ( Multivariate moduli of smoothness ) . For any t &gt; 0 and r ∈ N , the r -th multivariate modulus of smoothness of any function f ∈ L p ( X ) , p ∈ [1 , ∞ ] is defined as

<!-- formula-not-decoded -->

where ∆ r h ( f ) is the r -th order translation-difference operator defined as

<!-- formula-not-decoded -->

Remark 3.1 . The quantity ∆ r h ( f ) captures the local oscillation of f which is not necessarily differentiable. In the case the r -th order weak derivative D r f exists and is locally integrable, we have

<!-- formula-not-decoded -->

It also follows from Minkowski's inequality that

<!-- formula-not-decoded -->

Definition 3.2 (Besov space B α p,q ( X )) . For 1 ≤ p, q ≤ ∞ and α &gt; 0 , we define the norm ‖ · ‖ B α p,q of the Besov space B α p,q ( X ) as ‖ f ‖ B α p,q := ‖ f ‖ p + | f | B α p,q where

<!-- formula-not-decoded -->

is the Besov seminorm. Then, B α p,q := { f ∈ L p ( X ) : ‖ f ‖ B α p,q &lt; ∞} .

Intuitively, the Besov seminorm | f | B α p,q roughly describes the L q -norm of the l p -norm of the α -order smoothness of f . Besov spaces are considerably general that subsume Hölder spaces and Sobolev spaces as well as functions with spatially inhomogeneous smoothness (Triebel, 1983; Sawano, 2018; Suzuki, 2018; Cohen, 2009; Nickl &amp; Pötscher, 2007). In particular, the Besov space B α p,q reduces into the Hölder space C α when p = q = ∞ and α is positive and non-integer while it reduces into the Sobolev space W α 2 when p = q = 2 and α is a positive integer. We further consider the unit ball of B α p,q ( X ) as ¯ B α p,q ( X ) := { g ∈ B α p,q : ‖ g ‖ B α p,q ≤ 1 and ‖ g ‖ ∞ ≤ 1 } . When the context is clear, we drop X from ¯ B α p,q ( X ).

## 4 Fitted Q-Iteration for Offline Reinforcement Learning

## Algorithm 1 Fitted Q-Iteration with Neural Network Function Approximation

- 1: Input: Offline data D = { ( s i , a i , s ′ i , r i ) } n i =1 , number of iterations K , function family F NN , target policy π (for OPE Task only)
- 2: Initialize Q 0 ∈ F NN
- 3: for k = 1 , . . . , K do
- 4: Compute the estimated state-action value Q k as
- 5: end for

<!-- formula-not-decoded -->

- 6: Output: Return the following estimates

<!-- formula-not-decoded -->

In this work, we study a variant of fitted Q-iteration (FQI) algorithm for offline RL, presented in Algorithm 1. This algorithm is appealingly simple as it iteratively constructs Q -estimate from the offline data and the previous Q -estimate, as in Algorithm 1. This FQI-style algorithm has been largely studied for offline RL, such as Munos &amp; Szepesvári (2008); Chen &amp; Jiang (2019a); Duan et al. (2021a) to name a few; yet there has been no work studying this algorithm in offline RL with neural network function approximation except Yang et al. (2019). However, Yang et al. (2019) use data splitting and rely on a more limited dynamic condition than ours. Thus, the notable difference in Algorithm 1 is the use of neural network to approximate Q -functions and we estimate each Q k using the entire offline data set, instead of splitting the data into disjoint sets as in Yang et al. (2019). In particular, Yang et al. (2019) split the offline data into K disjoint sets, resulting in the sample complexity linearly scaled with K , which is highly inefficient in long (effective) horizon problems where the effective horizon length H = 1 / (1 -γ ) is large.

/negationslash

As we do not split the data into disjoint sets, a correlated structure is induced. Specifically, at each iteration k in Algorithm 1, Q k -1 also depends on ( s i , a i ) which makes E [ r i + γ max a Q k -1 ( s ′ i , a )] = [ T ∗ Q k -1 ]( s i , a i ) in OPL Task (and E [ r i + γ E a ∼ π ( ·| s ′ i ) [ Q k -1 ( s ′ i , a )] ] = [ T π Q k -1 ]( s i , a i ) in OPE Task, respectively). This correlated structure hinders a direct use of the standard concentration inequalities such as Bernstein's inequality that require a sequence of random variables to adapt to certain filtration. We overcome this technical difficulty using uniform convergence argument.

/negationslash

In our analysis, we assume access to the minimizer of the optimization in Algorithm 1. In practice, we can use (stochastic) gradient descent to effectively solve this optimization with L 0 regularization (Louizos et al., 2017). If the L 0 constraint is relaxed in practice, (stochastic) gradient descent is guaranteed to converge to a global minimum under certain structural assumptions (Du et al., 2019a;b; Allen-Zhu et al., 2019; Nguyen, 2021).

## 5 Main Result

To obtain a non-trivial guarantee, certain assumptions on the distribution shift and the MDP regularity are necessary. We introduce the assumptions about the data generation in Assumption 5.1 and the regularity of the underlying MDP 5.2.

Assumption 5.1 ( Uniform concentrability coefficient (Munos &amp; Szepesvári, 2008) ) . ∃ κ µ &lt; ∞ such that ∥ ∥ ∥ dν dµ ∥ ∥ ∥ ∞ ≤ κ µ for any admissible distribution ν . 1

∥ ∥ The finite κ µ in Assumption 5.1 asserts that the sampling distribution µ is not too far away from any admissible distribution, which holds for a reasonably large class of MDPs, e.g., for any finite MDP, any MDP with bounded transition kernel density, and equivalently any MDP whose top-Lyapunov exponent is negative. We present a simple (though stronger than necessary) example for which Assumption 5.1 holds.

Example 5.1. If there exist absolute constants c 1 , c 2 &gt; 0 such that for any s, s ′ ∈ S , there exists an action a ∈ A such that P ( s ′ | s, a ) ≥ 1 /c 1 and η ( a | s ) ≥ 1 /c 2 , ∀ s, a , then we can choose κ µ = c 1 c 2 .

Chen &amp; Jiang (2019a) further provided natural problems with rich observations generated from hidden states that has a low concentrability coefficient. These suggest that low concentrability coefficients can be found in fairly many interesting problems in practice.

We now state the assumption about the regularity of the underlying MDP.

Assumption 5.2 ( Besov dynamic closure ) . Consider some fixed p, q ∈ [1 , ∞ ] and α &gt; d min { p, 2 } .

- For OPE Task: For a target policy π , and for some ( L, m, S, B ) ∈ N × N × N × R + (which will be specified later) we assume that: ∀ f ∈ F NN ( L, m, S, B ) = ⇒ T π f ∈ ¯ B α p,q .
- For OPL Task: For some ( L, m, S, B ) ∈ N × N × N × R + (which will be specified later) we assume that: ∀ f ∈ F NN ( L, m, S, B ) = ⇒ T ∗ f ∈ ¯ B α p,q .

1 ν is said to be admissible if there exist t ≥ 0 and policy ¯ π such that ν ( s, a ) = P ( s t = s, a t = a | s 1 ∼ ρ, ¯ π ) , ∀ s, a .

Assumption 5.2 signifies that for OPL task (for OPE task with target policy π , respectively) the Bellman operator T ∗ ( T π , respectively) applied on any ReLU network function in F NN ( L, m, S, B ) results in a new function that sits in ¯ B α p,q ( X ). The smoothness constraint α &gt; d min { p, 2 } is necessary to guarantee the compactness and the finite (local) Rademacher complexity of the Besov space, and α -d/p is called the differential dimension of the Besov space. Note that when p &lt; 2 (thus the condition above becomes α &gt; d/p ), a function in the corresponding Besov space contains both spiky parts and smooth parts, i.e., the Besov space has inhomogeneous smoothness (Suzuki, 2018).

Our Besov dynamic closure is sufficiently general that subsumes almost all the previous completeness assumptions in the literature. For example, a simple (yet considerably stronger than necessary) sufficient condition for Assumption 5.2 is that the expected reward function r ( s, a ) and the transition density P ( s ′ | s, a ) for each fixed s ′ are functions in ¯ B α p,q , regardless of any input function f and any target policy π . 2 Such a condition on the transition dynamic is common in the RL literature; for example, linear MDPs (Jin et al., 2020a) posit a linear structure on the expected reward and the transition density as r ( s, a ) = 〈 φ ( s, a ) , θ 〉 and P ( s ′ | s, a ) = 〈 φ ( s, a ) , λ ( s ′ ) 〉 for some feature map φ : X → R d 0 and signed measures λ ( s ′ ) = ( λ ( s ′ ) 1 , . . . , λ ( s ′ ) d 0 ). To make it even more concrete, we present the following examples for Assumption 5.2.

Example 5.2 ( Reproducing kernel Hilbert space (RKHS) ) . Define k h,l the Matérn kernel with smoothness parameter h &gt; 0 and length scale l &gt; 0 . If both r ( · ) and g s ′ ( · ) := P ( s ′ |· ) at any s ′ ∈ S are functions in the RKHS of Matérn kernel k h,l where h = α -d/ 2 &gt; 0 and l &gt; 0 , then Assumption 5.2 holds for p = q = 2 . 3 Moreover, this particular case is equivalent to the dynamic condition considered in Yang et al. (2019).

Example 5.3 (Reduction to linear MDPs) . Linear MDPs (Jin et al., 2020a) correspond to Assumption 5.2 with α = 1 and p = q on a p -norm bounded domain. 4

Note that Assumption 5.2 even allows the expected rewards r ( · ) and the transition densities g s ′ ( · ) := P ( s ′ |· ) to contain both spiky parts and smooth parts, i.e., inhomogeneous smoothness, as long as p &lt; 2 (thus the constraint condition becomes α &gt; d/p ).

We are now ready to present our main result.

Theorem 5.1. Under Assumption 5.1 and Assumption 5.2 for some ( L, m, S, B ) satisfying (1), for any /epsilon1 &gt; 0 , δ ∈ (0 , 1] , K &gt; 0 , if n satisfies that n /greaterorsimilar ( 1 /epsilon1 2 ) 1+ d α log 6 n + 1 /epsilon1 2 (log(1 /δ ) + log log n ) , then with probability at least 1 -δ , the sup-optimality of Algorithm 1 is

In addition, the optimal deep ReLU network Φ( L, m, S, B ) that obtains such sample complexity (for both OPE and OPL) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As the complete form of Theorem 5.1 is quite involved, we interpret and disentangle this result to understand FQI algorithms with neural network function approximation for offline RL tasks. The sub-optimality in both

2 This sufficient condition imposes the smoothness constraint solely on the underlying MDP regardless of the input function f . Thus, the 'max' over the input function f ( s, a ) does not affect the smoothness of the resulting function after f is passed through the Bellman operator. This holds regardless of whether f is in the Besov space.

3 This is due to the norm-equivalence between the above RKHS and the Sobolev space W α 2 ( Ξ ) (Kanagawa et al., 2018) and the degeneration from Besov spaces to Sobolev spaces as B α 2 , 2 ( Ξ ) = W α 2 ( Ξ ).

4 However, linear MDPs do not require the smoothness constraint α &gt; d min { p, 2 } to ensure a finite Rademacher complexity of linear models. Of course, our analysis addresses significantly more complex and general settings than linear MDPs which we believe is more important than recovering this particular condition of linear MDPs.

OPE and OPL consists of the statistical error (the first term) and the algorithmic error (the second term). While the algorithmic error enjoys the fast linear convergence to 0 as K gets large, the statistical error reflects the fundamental difficulty of our problems. To make it more interpretable, we present a simplified version of Theorem 5.1 where we state the sample complexity required to obtain a sub-optimality within /epsilon1 .

Proposition 5.1 (Simplified version of Theorem 5.1) . For any K /greaterorsimilar H log(1 //epsilon1 ) , the sample complexity of Algorithm 1 for OPE Task and OPL Task is n = ˜ O ( H 2+2 d α κ 1+ d α µ /epsilon1 -2 -2 d α ) and n = ˜ O ( H 4+4 d α κ 1+ d α µ /epsilon1 -2 -2 d α ) , respectively. Moreover, the optimal deep ReLU network Φ( L, m, S, B ) for both OPE and OPL Tasks that obtains such sample complexity is L = O (log n ) , m = O ( n 2 / 5 log n ) , S = O ( n 2 / 5 ) , and log B = O ( n 2 / 5 d ) .

To discuss our result, we compare it with other existing works in Table 1. As the literature of offline RL is vast, we only compare with representative works of FQI estimators for offline RL with function approximation under a uniform data coverage assumption, as they are directly relevant to our work that uses FQI estimators with neural network function approximation under uniform data coverage. Here, our sample complexity does not scale with the number of states as in tabular MDPs (Yin &amp; Wang, 2020; Yin et al., 2021; Yin &amp; Wang, 2021a) or the inherent Bellman error as in the general function approximation (Munos &amp; Szepesvári, 2008; Le et al., 2019; Duan et al., 2021a). Instead, it explicitly scales with the (possible fractional) smoothness α of the underlying MDP, the dimension d of the input space, the distributional shift measure κ µ and the effective episode length H = (1 -γ ) -1 . Importantly, this guarantee is established under the Besov dynamic closure that subsumes the dynamic conditions of the prior results. Compared to Yang et al. (2019), our sample complexity has a strong advantage in long (effective) horizon problems where H &gt; d α -2 d log(1 //epsilon1 ) 5 and improves it by a factor of H 1 -2 d/α /epsilon1 -d/α log( H 2 //epsilon1 2 ). It also suggests that the data splitting in Yang et al. (2019) should be preferred for short (effective) horizon problems. Though our bound has a tighter dependence on H in the long horizon setting, the dependence on /epsilon1 in our bound is compromised and does not match the minimax rate in the regression setting. We leave as future direction to construct the lower bound for the data-reuse setting of offline RL.

On the role of deep ReLU networks in offline RL. We make several remarks about the role of deep networks in offline RL. The role of deep ReLU networks in offline RL is to guarantee a maximal adaptivity to the (spatial) regularity of the functions in Besov space and obtain an optimal approximation error rate that otherwise were not possible with other function approximation such as kernel methods (Suzuki, 2018). Moreover, by the equivalence in the functions that a neural architecture can compute (Yarotsky, 2017), Theorem 5.1 also readily holds for any other continuous piece-wise linear activation functions with finitely many line segments M where the optimal network architecture only increases the number of units and weights by constant factors depending only on M . Moreover, we observe that the optimal ReLU network is relatively 'thinner' than overparameterized neural networks that have been recently studied in the literature (Arora et al., 2019; Allen-Zhu et al., 2019; Hanin &amp; Nica, 2019; Cao &amp; Gu, 2019; Belkin, 2021) where the width m is a high-order polynomial of n . As overparameterization is a key feature for such overparameterized neural networks to obtain a good generalization, it is natural to ask why a thinner neural network in Theorem 5.1 also guarantees a strong generalization for offline RL? Intuitively, the optimal ReLU network in Theorem 5.1 is regularized by a strong sparsity which resonates with our practical wisdom that a sparsity-based regularization prevents over-fitting and achieve a better generalization. Indeed, as the total number of parameters in the considered neural network is p = md + m + m 2 ( L -2) = O ( N 2 log 3 N ) while the number of non-zeros parameters S only scales with N , the optimal ReLU network in Theorem 5.1 is relatively sparse.

## 6 Technical Review

In this section, we highlight the key technical challenges in our analysis. In summary, two key technical challenges in our analysis are rooted in the consideration of the correlated structure in value regression in Algorithm 1, and the use of deep neural network as function approximation (and their combination). To address these challenges, we devise the so-called double uniform convergence argument and leverage a

5 This condition is often easily satisfied as in practice we commonly set γ = 0 . 99 and /epsilon1 = 0 . 001, thus we have H = 100 and log(1 //epsilon1 ) = 3.

localization argument via sub-root functions for local Rademacher complexities. In what follows, we briefly discuss these technical challenges and our analysis approach.

/negationslash

The analysis and technical proofs of Yang et al. (2019); Le et al. (2019) heavily rely on the equation E [ r i + γ E a ′ ∼ π ( ·| s ′ i ) [ Q k -1 ( s ′ i , a )] ] = [ T ∗ Q k -1 ]( s i , a i ) to leverage the standard nonparametric regression techniques (in a supervised learning setting). However, the correlated structure in Algorithm 1 implies E [ r i + γ E a ′ ∼ π ( ·| s ′ i ) [ Q k -1 ( s ′ i , a )] ] = [ T ∗ Q k -1 ]( s i , a i ) as Q k -1 also depends on ( s i , a i ). Thus, the techniques in these prior works could not be used here and we require a new analysis. It is worth noting that Le et al. (2019) also re-use the data as in Algorithm 1 (instead of data splitting as in Yang et al. (2019)) but mistakenly assume that E [ r i + γ E a ′ ∼ π ( ·| s ′ i ) [ Q k -1 ( s ′ i , a )] ] = [ T ∗ Q k -1 ]( s i , a i ). To deal with the correlated structure, we devise a double uniform convergence argument. The double uniform convergence argument is appealingly intuitive: while in a standard regression problem, the (single) uniform convergence argument seeks the generalization guarantee uniformly over an entire hypothesis space of a data-dependent empirical risk minimizer, in the value regression problem of Algorithm 1, we additionally guarantee generalization uniformly over the hypothesis space of the data-dependent regression target T ∗ Q k -1 . To make it concrete, we highlight a key equality in our proof where the double uniform convergence argument is used:

<!-- formula-not-decoded -->

where f Q ∗ ( x ) = E [ r + γ max a ′ Q ( s ′ , a ′ ) | x ], and f Q ⊥ := arg inf f ∈F NN ‖ f -f Q ∗ ‖ 2 ,µ , and l f Q ⊥ := ( f Q ⊥ ( x 1 ) -r 1 -γ max a ′ Q ( s ′ 1 , a ′ )) 2 and l f Q ∗ := ( f Q ∗ ( x 1 ) -r 1 -γ max a ′ Q ( s ′ 1 , a ′ )) 2 are random variables with respect to the randomness of ( x 1 , s ′ 1 , r 1 ) . We have learned that a similar general idea of the double uniform convergence argument has been leveraged in Chen &amp; Jiang (2019b) for general function classes. We remark they use finite function classes, and in our case, the double uniform convergence argument is particularly helpful in dealing with local Rademacher complexities under a data-dependent structure as local Rademacher complexities already involve the supremum operator which can be naturally incorporated with the double uniform convergence argument.

The double uniform convergence argument also requires a different technique to control an empirical process term I 1 as it now requires a uniform convergence over the regression target. We leverage local Rademacher complexities to derive a bound on I 1 :

<!-- formula-not-decoded -->

where R n is the local Rademacher complexity (Bartlett et al., 2005). An explicit bound is then derived via a localization argument and the fixed point of a sub-root function.

The use of neural networks pose a new challenge mainly in bounding the bias term I 2 . We derive this bound using the adaptivity of deep ReLU network to the regularity in Besov spaces, leveraging our Besov dynamic condition in Assumption 5.2. Bounding the bias term also requires the use of a concentration inequality. While Le et al. (2019) use Bernstein's inequality, our bias term I 2 requires a uniform convergence version of Bernstein's inequality as I 2 requires a guarantee uniformly over F NN . We omit a detailed proof for Theorem 5.1 to Section A.

## 7 Conclusion and Discussion

We presented the sample complexity of FQI estimators for offline RL with deep ReLU network function approximation under a uniform data coverage assumption. We proved that the FQI-type algorithm achieved the sample complexity of n = ˜ O ( H 4+4 d α κ 1+ d α µ /epsilon1 -2 -2 d α ) under a correlated structure and a general dynamic condition namely the Besov dynamic closure. In addition, we corrected the mistake in ignoring the correlated

structure when reusing data with FQI estimators in Le et al. (2019), avoided the possibly inefficient data splitting technique in Yang et al. (2019) for long (effective) horizon problems, and proposed a general dynamic condition that subsumes all the previous Bellmen completeness assumptions. In the following, we discuss future directions.

Relaxing the assumption about uniform data coverage. For a future work, we can include the pessimistic approach in Jin et al. (2020b); Rashidinejad et al. (2021); Uehara &amp; Sun (2021); Nguyen-Tang et al. (2022a) to the current work with a more involved analysis of uncertainty quantifiers under non-linear function approximation to relax the strictness of the uniform data coverage assumption.

Relaxing the assumption about optimization oracle. The present work assumes access to the optimization oracle when fitting a neural network to the offline data. It is desirable to understand how optimization and generalization of a trained neural network can contribute to offline RL with neural function approximation. A promising approach to obtain a tight trajectory-dependent sub-optimality bound of offline RL with neural function approximation is to characterize the SGD-based optimization via a stochastic differential equation by allowing the stochastic noises to follow the fractional Brownian motion Tan et al. (2022); Tong et al. (2022).

## Acknowledgements

We thank our anonymous reviewers and our action editor Yu-Xiang Wang (UC Santa Barbara) at TMLR for the constructive comments and feedback. Thanh Nguyen-Tang thank Le Minh Khue Nguyen (University of Rochester) for the support of this project during the COVID times.

## References

- Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning , pp. 242-252. PMLR, 2019. 7, 9
- András Antos, Csaba Szepesvári, and Rémi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Mach. Learn. , 71(1):89-129, April 2008. ISSN 0885-6125. doi: 10.1007/s10994-007-5038-2. URL https://doi.org/10.1007/s10994-007-5038-2 . 3
- Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955 , 2019. 9
- Peter L. Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. Ann. Statist. , 33(4):1497-1537, 08 2005. doi: 10.1214/009053605000000282. URL https://doi.org/10.1214/009053605000000282 . 10, 19, 21, 25
- Mikhail Belkin. Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. arXiv preprint arXiv:2105.14368 , 2021. 9
- Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming: an overview. In Proceedings of 1995 34th IEEE Conference on Decision and Control , volume 1, pp. 560-564. IEEE, 1995. 3
- Jacob Buckman, Carles Gelada, and Marc G Bellemare. The importance of pessimism in fixed-dataset policy optimization. arXiv preprint arXiv:2009.06799 , 2020. 3
- Yuan Cao and Quanquan Gu. Generalization bounds of stochastic gradient descent for wide and deep neural networks. Advances in Neural Information Processing Systems , 32:10836-10846, 2019. 9
- Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In ICML , volume 97 of Proceedings of Machine Learning Research , pp. 1042-1051. PMLR, 2019a. 5, 7

- Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. arXiv preprint arXiv:1905.00360 , 2019b. 10
- Jinglin Chen and Nan Jiang. Offline reinforcement learning under value and density-ratio realizability: the power of gaps. 2022. 2, 4
- Lin Chen, Bruno Scherrer, and Peter L Bartlett. Infinite-horizon offline reinforcement learning with linear function approximation: Curse of dimensionality and algorithm. arXiv preprint arXiv:2103.09847 , 2021. 4
- Albert Cohen. A primer on besov spaces, 2009. URL http://cnx.org/content/col10679/1.2/&gt; . 6
- Simon S. Du, Jason D. Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In ICML , volume 97 of Proceedings of Machine Learning Research , pp. 1675-1685. PMLR, 2019a. 7
- Simon S. Du, Xiyu Zhai, Barnabás Póczos, and Aarti Singh. Gradient descent provably optimizes overparameterized neural networks. In ICLR (Poster) . OpenReview.net, 2019b. 7
- Yaqi Duan and Mengdi Wang. Minimax-optimal off-policy evaluation with linear function approximation. CoRR , abs/2002.09516, 2020. 1, 3
- Yaqi Duan, Chi Jin, and Zhiyuan Li. Risk bounds and rademacher complexity in batch reinforcement learning. arXiv preprint arXiv:2103.13883 , 2021a. 4, 7, 9
- Yaqi Duan, Mengdi Wang, and Martin J Wainwright. Optimal policy evaluation using kernel-based temporal difference methods. arXiv preprint arXiv:2109.12002 , 2021b. 4
- Miroslav Dudík, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601 , 2011. 3
- Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh. More robust doubly robust off-policy evaluation. arXiv preprint arXiv:1802.03493 , 2018. 3
- Max H Farrell, Tengyuan Liang, and Sanjog Misra. Deep neural networks for estimation and inference: Application to causal effects and other semiparametric estimands. arXiv preprint arXiv:1809.09953 , 2018. 20
- Evarist Giné and Richard Nickl. Mathematical foundations of infinite-dimensional statistical models , volume 40. Cambridge University Press, 2016. 5
- Steffen Grünewälder, Guy Lever, Luca Baldassarre, Massimiliano Pontil, and Arthur Gretton. Modelling transition dynamics in mdps with RKHS embeddings. In ICML . icml.cc / Omnipress, 2012. 3
- László Györfi, Michael Kohler, Adam Krzyzak, and Harro Walk. A Distribution-Free Theory of Nonparametric Regression . Springer series in statistics. Springer, 2002. 25
- Boris Hanin and Mihai Nica. Finite depth and width corrections to the neural tangent kernel. arXiv preprint arXiv:1909.05989 , 2019. 9
- Yichun Hu, Nathan Kallus, and Masatoshi Uehara. Fast rates for the regret of offline reinforcement learning. arXiv preprint arXiv:2102.00479 , 2021. 4
- Xiang Ji, Minshuo Chen, Mengdi Wang, and Tuo Zhao. Sample complexity of nonparametric off-policy evaluation on low-dimensional manifolds using deep networks. arXiv preprint arXiv:2206.02887 , 2022. 4
- Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning, 2015. 3
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pp. 2137-2143. PMLR, 2020a. 8

- Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? arXiv preprint arXiv:2012.15085 , 2020b. 1, 2, 3, 4, 11
- Nicholas K. Jong and Peter Stone. Model-based function approximation in reinforcement learning. In AAMAS , pp. 95. IFAAMAS, 2007. 3
- Nathan Kallus and Masatoshi Uehara. Double reinforcement learning for efficient off-policy evaluation in markov decision processes, 2019. 3
- Motonobu Kanagawa, Philipp Hennig, Dino Sejdinovic, and Bharath K Sriperumbudur. Gaussian processes and kernel methods: A review on connections and equivalences. arXiv preprint arXiv:1807.02582 , 2018. 8
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems , pp. 1097-1105, 2012. 5
- Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. arXiv preprint arXiv:2006.04779 , 2020. 1
- Michail G. Lagoudakis and Ronald Parr. Least-squares policy iteration. J. Mach. Learn. Res. , 4:1107-1149, 2003. 3
- Sascha Lange, Thomas Gabel, and Martin Riedmiller. Batch reinforcement learning. In Reinforcement learning , pp. 45-73. Springer, 2012. 1
- Hoang Minh Le, Cameron Voloshin, and Yisong Yue. Batch policy learning under constraints. In ICML , volume 97 of Proceedings of Machine Learning Research , pp. 3703-3712. PMLR, 2019. 2, 3, 4, 9, 10, 11, 16, 19, 20
- Yunwen Lei, Lixin Ding, and Yingzhou Bi. Local rademacher complexity bounds based on covering numbers. Neurocomputing , 218:320-330, 2016. 25
- Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 , 2020. 1
- Fanghui Liu, Luca Viano, and Volkan Cevher. Understanding deep neural function approximation in reinforcement learning via $\epsilon$-greedy exploration. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems , 2022. URL https://openreview.net/forum?id=o8vYKDWMnq1 . 4
- Christos Louizos, Max Welling, and Diederik P. Kingma. Learning sparse neural networks through l 0 regularization. CoRR , abs/1712.01312, 2017. URL http://arxiv.org/abs/1712.01312 . 7
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature , 518(7540):529-533, 2015. 5
- Rémi Munos. Error bounds for approximate policy iteration. In ICML , pp. 560-567. AAAI Press, 2003. 3
- Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. J. Mach. Learn. Res. , 9: 815-857, 2008. 2, 3, 4, 5, 7, 9, 16, 19, 20
- Quynh Nguyen. On the proof of global convergence of gradient descent for deep relu networks with linear widths. arXiv preprint arXiv:2101.09612 , 2021. 7
- Thanh Nguyen-Tang, Sunil Gupta, Hung Tran-The, and Svetha Venkatesh. Sample complexity of offline reinforcement learning with deep relu networks, 2021. 2
- Thanh Nguyen-Tang, Sunil Gupta, A. Tuan Nguyen, and Svetha Venkatesh. Offline neural contextual bandits: Pessimism, optimization and generalization. In International Conference on Learning Representations , 2022a. URL https://openreview.net/forum?id=sPIFuucA3F . 2, 4, 11

- Thanh Nguyen-Tang, Ming Yin, Sunil Gupta, Svetha Venkatesh, and Raman Arora. On instancedependent bounds for offline reinforcement learning with linear function approximation. arXiv preprint arXiv:2211.13208 , 2022b. 1, 3
- R. Nickl and B. M. Pötscher. Bracketing metric entropy rates and empirical central limit theorems for function classes of besov- and sobolev-type. Journal of Theoretical Probability , 20:177-199, 2007. 6, 26
- Doina Precup, Richard S. Sutton, and Satinder P. Singh. Eligibility traces for off-policy policy evaluation. In Proceedings of the Seventeenth International Conference on Machine Learning , ICML '00, pp. 759-766, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. ISBN 1558607072. 3
- Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging offline reinforcement learning and imitation learning: A tale of pessimism. arXiv preprint arXiv:2103.12021 , 2021. 3, 11
- Patrick Rebeschini. Oxford Algorithmic Foundations of Learning, Lecture Notes: Maximal Inequalities and Rademacher Complexity, 2019. URL: http://www.stats.ox.ac.uk/~rebeschi/teaching/AFoL/20/material/lecture02.pdf . Last visited on Sep. 14, 2020. 25
- Tongzheng Ren, Jialian Li, Bo Dai, Simon S Du, and Sujay Sanghavi. Nearly horizon-free offline reinforcement learning. Advances in neural information processing systems , 34:15621-15634, 2021. 3
- Yoshihiro Sawano. Theory of Besov Spaces , volume 56. Springer, 2018. ISBN 978-981-13-0835-2. 6
- Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Pessimistic q-learning for offline reinforcement learning: Towards optimal sample complexity. arXiv preprint arXiv:2202.13890 , 2022. 3
- Avi Singh, Albert Yu, Jonathan Yang, Jesse Zhang, Aviral Kumar, and Sergey Levine. Cog: Connecting new skills to past experience with offline reinforcement learning. arXiv preprint arXiv:2010.14500 , 2020. 1
- Taiji Suzuki. Adaptivity of deep relu network for learning in besov and mixed smooth besov spaces: optimal rate and curse of dimensionality. arXiv preprint arXiv:1810.08033 , 2018. 5, 6, 8, 9, 26
- Chengli Tan, Jiangshe Zhang, and Junmin Liu. Trajectory-dependent generalization bounds for deep neural networks via fractional brownian motion. arXiv preprint arXiv:2206.04359 , 2022. 11
- Philip Thomas and Emma Brunskill. Data-efficient off-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning , pp. 2139-2148, 2016. 3
- Anh Tong, Thanh Nguyen-Tang, Toan Tran, and Jaesik Choi. Learning fractional white noises in neural stochastic differential equations. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems , 2022. URL https://openreview.net/forum?id=lTZBRxm2q5 . 11
- Samuele Tosatto, Matteo Pirotta, Carlo D'Eramo, and Marcello Restelli. Boosted fitted q-iteration. In ICML , volume 70 of Proceedings of Machine Learning Research , pp. 3434-3443. PMLR, 2017. 3
- H. Triebel. Theory of function spaces. 1983. 6
- Masatoshi Uehara and Wen Sun. Pessimistic model-based offline reinforcement learning under partial coverage. arXiv preprint arXiv:2107.06226 , 2021. 2, 4, 11
- Masatoshi Uehara, Masaaki Imaizumi, Nan Jiang, Nathan Kallus, Wen Sun, and Tengyang Xie. Finite sample analysis of minimax offline reinforcement learning: Completeness, fast rates and first-order efficiency. arXiv preprint arXiv:2102.02981 , 2021. 4
- Cameron Voloshin, Hoang M Le, Nan Jiang, and Yisong Yue. Empirical study of off-policy policy evaluation for reinforcement learning. arXiv preprint arXiv:1911.06854 , 2019. 2, 5

- Ruosong Wang, Dean P Foster, and Sham M Kakade. What are the statistical limits of offline rl with linear function approximation? arXiv preprint arXiv:2010.11895 , 2020. 3
- Xinqi Wang, Qiwen Cui, and Simon S Du. On gap-dependent bounds for offline reinforcement learning. arXiv preprint arXiv:2206.00177 , 2022. 3
- Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for offline reinforcement learning. Advances in neural information processing systems , 34, 2021a. 2, 3, 4
- Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, and Yu Bai. Policy finetuning: Bridging sampleefficient offline and online reinforcement learning. Advances in neural information processing systems , 34: 27395-27407, 2021b. 3
- Wei Xiong, Han Zhong, Chengshuai Shi, Cong Shen, Liwei Wang, and T. Zhang. Nearly minimax optimal offline reinforcement learning with linear function approximation: Single-agent mdp and markov game. ArXiv , abs/2205.15512, 2022. 1, 3
- Zhuoran Yang, Yuchen Xie, and Zhaoran Wang. A theoretical analysis of deep q-learning. CoRR , abs/1901.00137, 2019. 2, 3, 4, 5, 7, 8, 9, 10, 11, 19
- Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural Networks , 94:103-114, 2017. 9
- Ming Yin and Yu-Xiang Wang. Asymptotically efficient off-policy evaluation for tabular reinforcement learning. In AISTATS , volume 108 of Proceedings of Machine Learning Research , pp. 3948-3958. PMLR, 2020. 1, 3, 9
- Ming Yin and Yu-Xiang Wang. Characterizing uniform convergence in offline policy evaluation via modelbased approach: Offline learning, task-agnostic and reward-free, 2021a. 1, 9
- Ming Yin and Yu-Xiang Wang. Optimal uniform ope and model-based offline reinforcement learning in timehomogeneous, reward-free and task-agnostic settings. Advances in neural information processing systems , 34:12890-12903, 2021b. 3
- Ming Yin and Yu-Xiang Wang. Towards instance-optimal offline reinforcement learning with pessimism. Advances in neural information processing systems , 34:4065-4078, 2021c. 3
- Ming Yin, Yu Bai, and Yu-Xiang Wang. Near-optimal provable uniform convergence in offline policy evaluation for reinforcement learning. In Arindam Banerjee and Kenji Fukumizu (eds.), The 24th International Conference on Artificial Intelligence and Statistics, AISTATS 2021, April 13-15, 2021, Virtual Event , volume 130 of Proceedings of Machine Learning Research , pp. 1567-1575. PMLR, 2021. URL http://proceedings.mlr.press/v130/yin21a.html . 1, 3, 9
- Ming Yin, Yaqi Duan, Mengdi Wang, and Yu-Xiang Wang. Near-optimal offline reinforcement learning with linear representation: Leveraging variance information with pessimism. arXiv preprint arXiv:2203.05804 , 2022. 1, 3
- Wenhao Zhan, Baihe Huang, Audrey Huang, Nan Jiang, and Jason D Lee. Offline reinforcement learning with realizability and single-policy concentrability. arXiv preprint arXiv:2202.04634 , 2022. 2, 4
- Mengyan Zhang, Thanh Nguyen-Tang, Fangzhao Wu, Zhenyu He, Xing Xie, and Cheng Soon Ong. Twostage neural contextual bandits for personalised news recommendation. arXiv preprint arXiv:2206.14648 , 2022. 1

## A Appendix

## A Proof of Theorem 5.1

We now provide a complete proof of Theorem 5.1. The proof has four main components: a sub-optimality decomposition for error propagation across iterations, a Bellman error decomposition using a uniform convergence argument, a deviation analysis for least squares with deep ReLU networks using local Rademacher complexities and a localization argument, and a upper bound minimization step to obtain an optimal deep ReLU architecture.

## Step 1: A sub-optimality decomposition

The first step of the proof is a sub-optimality decomposition, stated in Lemma A.1, that applies generally to any least-squares Q-iteration methods.

Lemma A.1 ( A sub-optimality decomposition ) . Under Assumption 5.1, the sub-optimality of V K returned by Algorithm 1 is bounded as

<!-- formula-not-decoded -->

The lemma states that the sub-optimality decomposes into a statistical error (the first term) and an algorithmic error (the second term). While the algorithmic error enjoys the fast linear convergence rate, the statistical error arises from the distributional shift in the offline data and the estimation error of the target Q -value functions due to finite data. Crucially, the contraction of the (optimality) Bellman operators T π and T ∗ allows the sup-optimality error at the final iteration K to propagate across all iterations k ∈ [0 , K -1]. Note that this result is agnostic to any function approximation form and does not require Assumption 5.2. The result uses a relatively standard argument that appears in a number of works on offline RL (Munos &amp; Szepesvári, 2008; Le et al., 2019).

Proof of Lemma A.1. We will prove the sup-optimality decomposition for both settings: OPE and OPL.

(i) For OPE. We denote the right-linear operator by P π · : {X → R } → {X → R } where

<!-- formula-not-decoded -->

for any f ∈ {X → R } . Denote Denote ρ π ( dsda ) = ρ ( ds ) π ( da | s ). Let /epsilon1 k := Q k +1 -T π Q k , ∀ k ∈ [0 , K -1] and /epsilon1 K = Q 0 -Q π . Since Q π is the (unique) fixed point of T π , we have

<!-- formula-not-decoded -->

By recursion, we have

<!-- formula-not-decoded -->

where α k := (1 -γ ) γ k 1 -γ K +1 , ∀ k ∈ [ K ] and A k := ( P π ) k , ∀ k ∈ [ K ]. Note that ∑ K k =0 α k = 1 and A k 's are probability kernels. Denoting by | f | the point-wise absolute value | f ( s, a ) | , we have that the following inequality holds point-wise:

<!-- formula-not-decoded -->

We have

The inequalities ( a ) and ( b ) follow from Jensen's inequality, ( c ) follows from ‖ Q 0 ‖ ∞ , ‖ Q π ‖ ∞ ≤ 1, and ( d ) follows from Assumption 5.1 that ρ π A k = ρ π ( P π ) k ≤ κ µ µ . Thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(ii) For OPL. The sup-optimality for the OPL setting is more complex than the OPE setting but the technical steps are relatively similar. In particular, let /epsilon1 k -1 = T ∗ Q k -1 -Q k , ∀ k and π ∗ ( s ) = arg max a Q ∗ ( s, a ) , ∀ s , we have

<!-- formula-not-decoded -->

Now, let π k be the greedy policy w.r.t. Q k , we have

<!-- formula-not-decoded -->

Now, we turn to decompose Q ∗ -Q π K as

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Note that the operator ( I -γP π K ) -1 = ∑ ∞ i =0 ( γP π K ) i is monotone, thus

<!-- formula-not-decoded -->

Combining Equation (4) with Equations (2) and (3), we have

<!-- formula-not-decoded -->

Using the triangle inequality, the above inequality becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that A k is a probability kernel for all k and ∑ k α k = 1. Thus, similar to the steps in the OPE setting, for any policy π , we have

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Step 2: A Bellman error decomposition

The next step of the proof is to decompose the Bellman errors ‖ Q k +1 -T π Q k ‖ µ for OPE and ‖ Q k +1 -T ∗ Q k ‖ µ for OPL. Since these errors can be decomposed and bounded similarly, we only focus on OPL here.

The difficulty in controlling the estimation error ‖ Q k +1 -T ∗ Q k ‖ 2 ,µ is that Q k itself is a random variable that depends on the offline data D . In particular, at any fixed k with Bellman targets { y i } n i =1 where y i = r i + γ max a ′ Q k ( s ′ i , a ′ ), it is not immediate that E [[ T ∗ Q k ]( x i ) -y i | x i ] = 0 for each covariate x i := ( s i , a i ) as Q k itself depends on x i (thus the tower law cannot apply here). A naive and simple approach to break such data dependency of Q k is to split the original data D into K disjoint subsets and estimate each Q k using a separate subset. This naive approach is equivalent to the setting in Yang et al. (2019) where a fresh batch of data is generated for different iterations. This approach is however not efficient as it uses only n/K samples to estimate each Q k . This is problematic in high-dimensional offline RL when the number of iterations K can be very large as it is often the case in practical settings. We instead prefer to use all n samples to estimate each Q k . This requires a different approach to handle the complicated data dependency of each Q k . To circumvent this issue, we leverage a uniform convergence argument by introducing a deterministic covering of T ∗ F NN . Each element of the deterministic covering induces a different regression target { r i + γ max a ′ ˜ Q ( s ′ i , a ′ ) } n i =1 where ˜ Q is a deterministic function from the covering which ensures that E [ r i + γ max a ′ ˜ Q ( s ′ i , a ′ ) -[ T ∗ ˜ Q ]( x i ) | x i ] = 0. In particular, we denote

<!-- formula-not-decoded -->

where l ( x, y ) = ( x -y ) 2 is the squared loss function. Note that for any deterministic Q ∈ F NN , we have f Q ∗ ( x 1 ) = E [ y Q 1 | x 1 ] , ∀ x 1 , thus

<!-- formula-not-decoded -->

where l f denotes the random variable ( f ( x 1 ) -y Q 1 ) 2 for a given fixed Q . Now letting f Q ⊥ := arg inf f ∈F NN ‖ f -f Q ∗ ‖ 2 ,µ be the projection of f Q ∗ onto the function class F NN , we have

<!-- formula-not-decoded -->

where (a) follows from that Q k ∈ F NN , (b) follows from Equation (5), and (c) follows from that E n [ l ˆ f Q ] ≤ E n [ l f Q ] , ∀ f, Q ∈ F NN . That is, the error is decomposed into two terms: the first term I 1 resembles the empirical process in statistical learning theory and the second term I 2 specifies the bias caused by the regression target f Q ∗ not being in the function space F NN .

## Step 3: A deviation analysis

The next step is to bound the empirical process term and the bias term via an intricate concentration, local Rademacher complexities and a localization argument. First, the bias term in Equation (6) is taken uniformly over the function space, thus standard concentration arguments such as Bernstein's inequality and Pollard's inequality used in Munos &amp; Szepesvári (2008); Le et al. (2019) do not apply here. Second, local Rademacher complexities (Bartlett et al., 2005) are data-dependent complexity measures that exploit the fact that only a small subset of the function class will be used. Leveraging a localization argument for

local Rademacher complexities (Farrell et al., 2018), we localize an empirical Rademacher ball into smaller balls by which we can handle their complexities more effectively. Moreover, we explicitly use the sub-root function argument to derive our bound and extend the technique to the uniform convergence case. That is, reasoning over the sub-root function argument makes our proof more modular and easier to incorporate the uniform convergence argument.

Localization is particularly useful to handle the complicated approximation errors induced by deep ReLU network function approximation.

## Step 3.a: Bounding the bias term via a uniform convergence concentration inequality

Before delving into our proof, we introduce relevant notations. Let F - G := { f -g : f ∈ F , g ∈ G} , let N ( /epsilon1, F , ‖·‖ ) be the /epsilon1 -covering number of F w.r.t. ‖·‖ norm, H ( /epsilon1, F , ‖·‖ ) := log N ( /epsilon1, F , ‖·‖ ) be the entropic number, let N [] ( /epsilon1, F , ‖ · ‖ ) be the bracketing number of F , i.e., the minimum number of brackets of ‖ · ‖ -size less than or equal to /epsilon1 , necessary to cover F , let H [] ( /epsilon1, F , ‖ · ‖ ) = log N [] ( /epsilon1, F , ‖ · ‖ ) be the ‖ · ‖ -bracketing metric entropy of F ,let F|{ x i } n i =1 = { ( f ( x 1 ) , ..., f ( x n )) ∈ R n | f ∈ F} , and let T ∗ F = { T ∗ f : f ∈ F} . Finally, for sample set { x i } n i =1 , we define the empirical norm ‖ f ‖ n := √ 1 n ∑ n i =1 f ( x i ) 2 .

We define the inherent Bellman error as d F NN := sup Q ∈F NN inf f ∈F NN ‖ f -T ∗ Q ‖ µ . This implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have

We have

<!-- formula-not-decoded -->

For any /epsilon1 ′ &gt; 0 and δ ′ ∈ (0 , 1), it follows from Lemma B.2 with /epsilon1 = 1 / 2 and α = /epsilon1 ′ 2 , with probability at least 1 -δ ′ , for any Q ∈ F NN , we have

<!-- formula-not-decoded -->

given that

<!-- formula-not-decoded -->

Note that if we use Pollard's inequality (Munos &amp; Szepesvári, 2008) in the place of Lemma B.2, the RHS of Equation (8) is bounded by /epsilon1 ′ instead of /epsilon1 ′ 2 (i.e., n scales with O (1 //epsilon1 ′ 4 ) instead of O (1 //epsilon1 ′ 2 )). In addition, unlike Le et al. (2019), the uniform convergence argument hinders the application of Bernstein's inequality. We remark that Le et al. 2019 makes a mistake in their proof by ignoring the data-dependent structure in the algorithm (i.e., they wrongly assume that Q k in Algorithm 1 is fixed and independent of { s i , a i } n i =1 ). Thus, the uniform convergence argument in our proof is necessary.

## Step 3.b: Bounding the empirical process term via local Rademacher complexities

For any Q ∈ F NN , we have

<!-- formula-not-decoded -->

Thus, it follows from Lemma 1 (with α = 1 / 2) that with any r &gt; 0 , δ ∈ (0 , 1), with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

## Step 3.c: Bounding ‖ Q k +1 -T ∗ Q k ‖ µ using localization argument via sub-root functions

We bound ‖ Q k +1 -T ∗ Q k ‖ µ using the localization argument, breaking down the Rademacher complexities into local balls and then build up the original function space from the local balls. Let ψ be a sub-root function (Bartlett et al., 2005, Definition 3.1) with the fixed point r ∗ and assume that for any r ≥ r ∗ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We recall that a function ψ : [0 , ∞ ) → [0 , ∞ ) is sub-root if it is non-negative, non-decreasing and r ↦→ ψ ( r ) / √ r is non-increasing for r &gt; 0. Consequently, a sub-root function ψ has a unique fixed point r ∗ where r ∗ = ψ ( r ∗ ). In addition, ψ ( r ) ≤ √ rr ∗ , ∀ r ≥ r ∗ . In the next step, we will find a sub-root function ψ that satisfies the inequality above, but for this step we just assume that we have such ψ at hand. Combining Equations (6), (8), and (9), we have: for any r ≥ r ∗ and any δ ∈ (0 , 1), if ‖ ˆ f Q k -1 -f Q k -1 ∗ ‖ 2 2 ,µ ≤ r , with probability at least 1 -δ , where

<!-- formula-not-decoded -->

Consider r 0 ≥ r ∗ (to be chosen later) and denote the events

<!-- formula-not-decoded -->

where l = log 2 ( 1 r 0 ) ≤ log 2 ( 1 r ∗ ). We have B 0 ⊆ B 1 ⊆ ... ⊆ B l and since ‖ f -g ‖ 2 µ ≤ 1 , ∀| f | ∞ , | g | ∞ ≤ 1, we have P ( B l ) = 1. If ‖ ˆ f Q k -1 -f Q k -1 ∗ ‖ 2 µ ≤ 2 i r 0 for some i ≤ l , then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

if the following inequalities hold where

<!-- formula-not-decoded -->

## Step 3.d: Finding a sub-root function and its fixed point

It remains to find a sub-root function ψ ( r ) that satisfies Equation (9) and thus its fixed point. The main idea is to bound the RHS, the local Rademacher complexity, of Equation (9) by its empirical counterpart as the latter can then be further bounded by a sub-root function represented by a measure of compactness of the function spaces F NN and T ∗ F NN .

For any /epsilon1 &gt; 0, we have the following inequalities for entropic numbers:

<!-- formula-not-decoded -->

where N is a hyperparameter of the deep ReLU network described in Lemma B.9, (a) follows from Lemma B.9, and (b) follows from Assumption 5.2, and (c) follows from Lemma B.8. Let H := F NN -T ∗ F NN , it

<!-- formula-not-decoded -->

We choose r 0 ≥ r ∗ such that the inequalities above hold for all 0 ≤ i ≤ l . This can be done by simply setting

<!-- formula-not-decoded -->

Since { B i } is a sequence of increasing events, we have

<!-- formula-not-decoded -->

Thus, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

follows from Lemma B.5 with { ξ k := /epsilon1/ 2 k } k ∈ N for any /epsilon1 &gt; 0 that where we use √ a + b ≤ √ a + √ b, ∀ a, b ≥ 0, ∑ ∞ k =1 √ k 2 k -1 &lt; ∞ , and ∑ ∞ k =1 ( 1 2 1 -d 2 α ) k -1 &lt; ∞ . It now follows from Lemma B.4 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that V [( f -g ) 2 ] ≤ E [( f -g ) 4 ] ≤ E [( f -g ) 2 ] for any f ∈ F NN , g ∈ T ∗ F NN . Thus, for any r ≥ r ∗ , it follows from Lemma B.1 that with probability at least 1 -1 n , we have the following inequality for any f ∈ F NN , g ∈ T ∗ F NN such that ‖ f -g ‖ 2 µ ≤ r , where β ∈ (0 , α d ) is an absolute constant to be chosen later.

<!-- formula-not-decoded -->

if r ≥ r ∗ ∨ 2 logn n ∨ 56 logn 3 n . For such r , denote E r = {‖ f -g ‖ 2 n ≤ 4 r }∩{‖ f -f ∗ ‖ 2 µ ≤ r } , we have P ( E r ) ≥ 1 -1 /n and

<!-- formula-not-decoded -->

It is easy to verify that ψ ( r ) defined above is a sub-root function. The fixed point r ∗ of ψ ( r ) can be solved analytically via the simple quadratic equation r ∗ = ψ ( r ∗ ). In particular, we have

<!-- formula-not-decoded -->

It follows from Equation (10) (where l /lessorsimilar log(1 /r ∗ )), the definition of d F NN , Lemma B.9, and Equation (13) that for any /epsilon1 ′ &gt; 0 and δ ∈ (0 , 1), with probability at least 1 -δ , we have where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Step 4: Minimizing the upper bound

The final step for the proof is to minimize the upper error bound obtained in the previous steps w.r.t. two free parameters β ∈ (0 , α d ) and N ∈ N . Note that N parameterizes the deep ReLU architecture Φ( L, m, S, B ) given Lemma B.9. In particular, we optimize over β ∈ (0 , α d ) and N ∈ N to minimize the upper bound in the RHS of Equation (14). The RHS of Equation (14) is minimized (up to log n -factor) by choosing

<!-- formula-not-decoded -->

which results in N /equivasymptotic n 1 2 (2 β +1) d 2 α + d . At these optimal values, Equation (14) becomes

<!-- formula-not-decoded -->

Now, for any /epsilon1 &gt; 0, we set /epsilon1 ′ = /epsilon1/ 3 and let where we use inequalities n -β 2 (1 -d 2 α ) -1 2 ≤ n -1 2 (1 -βd α ) /equivasymptotic N -α/d = n -1 2 ( 2 α 2 α + d + d α ) -1 .

<!-- formula-not-decoded -->

It then follows from Equation (17) that with probability at least 1 -δ , we have max k ‖ Q k +1 -T ∗ Q k ‖ µ ≤ /epsilon1 if n simultaneously satisfies Equation (15) with /epsilon1 ′ = /epsilon1/ 3 and

<!-- formula-not-decoded -->

Next, we derive an explicit formula of the sample complexity satisfying Equation (15). Using Equations (14), (18), and (16), we have that n satisfies Equation (15) if

<!-- formula-not-decoded -->

Note that β ≤ 1 / 2 and d α ≤ 2; thus, we have

<!-- formula-not-decoded -->

Hence, n satisfies Equations (18) and (19) if

<!-- formula-not-decoded -->

## B Technical Lemmas

Lemma B.1 (Bartlett et al. (2005)) . Let r &gt; 0 and let

<!-- formula-not-decoded -->

1. For any λ &gt; 0 , we have with probability at least 1 -e -λ ,

<!-- formula-not-decoded -->

2. With probability at least 1 -2 e -λ ,

<!-- formula-not-decoded -->

Moreover, the same results hold for sup f ∈F ( E n f -E f ) .

Lemma B.2 (Györfi et al. (2002, Theorem 11.6)) . Let B ≥ 1 and F be a set of functions f : R d → [0 , B ] . Let Z 1 , ..., Z n be i.i.d. R d -valued random variables. For any α &gt; 0 , 0 &lt; /epsilon1 &lt; 1 , and n ≥ 1 , we have

<!-- formula-not-decoded -->

Lemma B.3 ( Contraction property (Rebeschini, 2019)) . Let φ : R → R be a L -Lipschitz, then

<!-- formula-not-decoded -->

Lemma B.4 (Lei et al. (2016, Lemma 1)) . Let F be a function class and P n be the empirical measure supported on X 1 , ..., X n ∼ µ , then for any r &gt; 0 (which can be stochastic w.r.t X i ), we have

<!-- formula-not-decoded -->

Lemma B.5 (Lei et al. (2016, modification)) . Let X 1 , ..., X n be a sequence of samples and P n be the associated empirical measure. For any function class F and any monotone sequence { ξ k } ∞ k =0 decreasing to 0 , we have the following inequality for any non-negative integer N

<!-- formula-not-decoded -->

Lemma B.6 ( Pollard's inequality ) . Let F be a set of measurable functions f : X → [0 , K ] and let /epsilon1 &gt; 0 , N arbitrary. If { X i } N i =1 is an i.i.d. sequence of random variables taking values in X , then

<!-- formula-not-decoded -->

Lemma B.7 ( Properties of (bracketing) entropic numbers ) . Let /epsilon1 ∈ (0 , ∞ ) . We have

1. H ( /epsilon1, F , ‖ · ‖ ) ≤ H [] (2 /epsilon1, F , ‖ · ‖ ) ;
2. H ( /epsilon1, F|{ x i } n i =1 , n -1 /p · ‖ · ‖ p ) = H ( /epsilon1, F , ‖ · ‖ p,n ) ≤ H ( /epsilon1, F|{ x i } n i =1 , ‖ · ‖ ∞ ) ≤ H ( /epsilon1, F , ‖ · ‖ ∞ ) for all { x i } n i =1 ⊂ dom ( F ) .

<!-- formula-not-decoded -->

Lemma B.8 ( Entropic number of bounded Besov spaces (Nickl &amp; Pötscher, 2007, Corollary 2.2)) . For 1 ≤ p, q ≤ ∞ and α &gt; d/p , we have

<!-- formula-not-decoded -->

Lemma B.9 ( Approximation power of deep ReLU networks for Besov spaces (Suzuki, 2018, a modified version)) . Let 1 ≤ p, q ≤ ∞ and α ∈ ( d p ∧ 2 , ∞ ) . For sufficiently large N ∈ N , there exists a neural network architecture Φ( L, m, S, B ) with

<!-- formula-not-decoded -->

where ν := α -δ 2 δ and δ := d ( p -1 -(1 + /floorleft α /floorright ) -1 ) + such that

<!-- formula-not-decoded -->