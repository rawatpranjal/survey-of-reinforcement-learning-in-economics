## An Analysis of Quantile Temporal-Difference Learning

## Mark Rowland

Google DeepMind, London, UK

R´ emi Munos

Google DeepMind, Paris, France

Mohammad Gheshlaghi Azar

Google DeepMind, Seattle, USA

Yunhao Tang

Google DeepMind, London, UK

Georg Ostrovski

Google DeepMind, London, UK

Anna Harutyunyan

Google DeepMind, London, UK

Karl Tuyls

Google DeepMind, Paris, France

Marc G. Bellemare

Reliant AI &amp; McGill University, Montr´ eal, Canada

Will Dabney

Google DeepMind, Seattle, USA

Editor: Alexandre Proutiere

markrowland@google.com

## Abstract

We analyse quantile temporal-difference learning (QTD), a distributional reinforcement learning algorithm that has proven to be a key component in several successful large-scale applications of reinforcement learning. Despite these empirical successes, a theoretical understanding of QTD has proven elusive until now. Unlike classical TD learning, which can be analysed with standard stochastic approximation tools, QTD updates do not approximate contraction mappings, are highly non-linear, and may have multiple fixed points. The core result of this paper is a proof of convergence to the fixed points of a related family of dynamic programming procedures with probability 1, putting QTD on firm theoretical footing. The proof establishes connections between QTD and non-linear differential inclusions through stochastic approximation theory and non-smooth analysis.

Keywords: Reinforcement learning, temporal-difference learning, distributional reinforcement learning, stochastic approximation, differential inclusion.

© 2024 Mark Rowland, R´ emi Munos, Mohammad Gheshlaghi Azar, Yunhao Tang, Georg Ostrovski, Anna Harutyunyan, Karl Tuyls, Marc G. Bellemare, Will Dabney.

## 1 Introduction

In distributional reinforcement learning, an agent aims to predict the full probability distribution over future returns it will encounter (Morimura et al., 2010b,a; Bellemare et al., 2017, 2023), in contrast to predicting just the mean return, as in classical reinforcement learning (Sutton and Barto, 2018). A widely-used family of algorithms for distributional reinforcement learning is based on the notion of learning quantiles of the return distribution, an approach that originated with Dabney et al. (2018b), who introduced the quantile temporal-difference (QTD) learning algorithm. This approach has been particularly successful in combination with deep reinforcement learning, and has been a central component in several recent real-world applications, including sim-to-real stratospheric balloon navigation (Bellemare et al., 2020), robotic manipulation (Bodnar et al., 2020), and algorithm discovery (Fawzi et al., 2022), as well as on benchmark simulated domains such as the Arcade Learning Environment (Bellemare et al., 2013; Machado et al., 2018; Dabney et al., 2018b,a; Yang et al., 2019) and racing simulation (Wurman et al., 2022).

Despite these empirical successes of QTD, little is known about its behaviour from a theoretical viewpoint. In particular, questions regarding the asymptotic behaviour of the algorithm (Do its predictions converge? Under what conditions? What is the qualitative character of the predictions when they do converge?) were left open. A core reason for this is that unlike classical TD, and other distributional reinforcement learning algorithms such as categorical temporal-difference learning (Rowland et al., 2018; Bellemare et al., 2023), the updates of QTD rely on asymmetric L 1 losses. As a result, these updates do not approximate the application of a contraction mapping, are highly non-linear (even in the tabular setting), and also may have multiple fixed points (depending on the exact structure of the reward distributions of the environment), and their analysis requires a distinct set of tools to those typically used to analyse temporal-difference learning algorithms.

In this paper, we prove the convergence of QTD-notably under weaker assumptions than are required in typical proofs of convergence for classical TD learning-establishing it as a sound algorithm with theoretical convergence guarantees, and paving the way for further analysis and investigation. The more general conditions stem from the structure of the QTD updates (namely, their boundedness), and the proof is obtained through the use of stochastic approximation theory with differential inclusions.

We begin by providing background on Markov decision processes, classical TD learning, and quantile regression in Section 2. After motivating the QTD algorithm in Section 3, we describe the related family of quantile dynamic programming (QDP) algorithms, and provide a convergence analysis of these algorithms in Section 4. We then present the main result, a convergence analysis of QTD, in Section 5. The proof relies on the stochastic approximation framework set out by Bena¨ ım et al. (2005), arguing that the QTD algorithm approximates a continuous-time differential inclusion, and then constructing a Lyapunov function to demonstrate that the limiting behaviour of trajectories of the differential inclusion matches that of the QDP algorithms introduced earlier. Finally, in Section 6, we analyse the limit points of QTD, bounding their approximation error to the true return distributions of interest, and investigating the kinds of approximation artefacts that arise empirically.

## 2 Background

We first introduce background concepts and notation.

## 2.1 Markov Decision Processes

We consider a Markov decision process specified by finite state and action spaces X and A , transition kernel P X : X ×A → P ( X ), reward distribution function P R : X ×A → P 1 ( R ), and discount factor γ ∈ [0 , 1). Here, P ( X ) is the set of probability distributions over the finite set X , and P 1 ( R ) is the set of probability distributions over R (with its usual Borel σ -algebra) with finite mean.

Given a policy π : X → P ( A ) and an initial state x 0 ∈ X , an agent interacting with the environment using the policy π generates a sequence of states, actions and rewards ( X t , A t , R t ) ∞ t =0 , called a trajectory , the joint distribution of which is determined by the transition dynamics and reward distributions of the environment, and the policy of the agent. More precisely, we have

- X 0 = x 0 , and for each t ≥ 0:
- A t | X 0: t , A 0: t -1 , R 0: t -1 ∼ π ( ·| X t );
- R t | X 0: t , A 0: t , R 0: t -1 ∼ P R ( ·| X t , A t );
- X t +1 | X 0: t , A 0: t , R 0: t ∼ P X ( ·| X t , A t ).

The distribution of the trajectory is thus parametrised by the initial state x 0 , and the policy π . To illustrate this dependency, we use the notation P π x 0 and E π x 0 to denote the probability distribution and expectation operator corresponding to this distribution, and will write P π ( ·| x ) for the joint distribution over a reward-next-state pair when the current state is x .

## 2.2 Predicting Expected Returns and the Return Distribution

The quality of the agent's performance on the trajectory is quantified by the discounted return , or simply the return , given by

<!-- formula-not-decoded -->

The return is a random variable, whose sources of randomness are the random selections of actions made according to π , the randomness in state transitions, and the randomness in rewards observed. Typically in reinforcement learning, a single scalar summary of performance is given by the expectation of this return over all these sources of randomness. For a given policy, this is summarised across each possible starting state via the value function V π : X → R , defined by

<!-- formula-not-decoded -->

Learning the value function of a policy π from sampled trajectories generated through interaction with the environment is a central problem in reinforcement learning, referred to as the policy evaluation task .

Each expected return is a scalar summary of a much more rich, complex object: the probability distributions of the random return in Equation (1) itself. Distributional reinforcement learning (Bellemare et al., 2023) is concerned with the problem of learning to predict the probability distribution over returns, in contrast to just their expected value. Mathematically, the goal is to learn the return-distribution function η π : X → P ( R ); for each state x ∈ X , η π ( x ) is the probability distribution of the random return in Expression (1) when the trajectory begins at state x , and the agent acts using policy π . Mathematically, we have

<!-- formula-not-decoded -->

where D π x extract the probability distribution of a random variable under P π x .

There are several distinct motivations for aiming to learn these more complex objects. First, the richness of the distribution provides an abundance of signal for an agent to learn from, in contrast to a single scalar expectation. The strong performance of deep reinforcement learning agents that incorporate distributional predictions is hypothesised to be related to this fact (Dabney et al., 2018b; Barth-Maron et al., 2018; Dabney et al., 2018a; Yang et al., 2019). Second, learning about the full probability distribution of returns makes possible the use of risk-sensitive performance criteria; one may be interested in not only the expected return under a policy, but also the variance of the return, or the probability of the return being under a certain threshold.

Unlike the value function V π , which is an element of R X , and can therefore be straightforwardly represented on a computer (up to floating-point precision), the return-distribution function η π is not representable. Each object η π ( x ) is a probability distribution over the real numbers, and, informally speaking, probability distributions have infinitely many degrees of freedom. Distributional reinforcement learning algorithms therefore typically work with a subset of distributions that are amenable to parametrisation on a computer (Bellemare et al., 2023). Common choices of subsets include categorical distributions (Bellemare et al., 2017), exponential families (Morimura et al., 2010b), and mixtures of Gaussian distributions (Barth-Maron et al., 2018). Quantile temporal-difference learning, the core algorithm of study in this paper, aims to learn a particular set of quantiles of the return distribution, as described in Section 3.

## 2.3 Monte Carlo and Temporal-Difference Learning

To foreshadow our description and motivation of quantile temporal-difference learning, we recall a line of thinking that interprets the classical TD learning update rule as an approximation to Monte Carlo learning; this material is common to many introductory texts on reinforcement learning (Sutton and Barto, 2018), and we present it here to make a direct analogy with QTD. First, we may observe that, under the condition that all reward distributions have finite variance, V π ( x ) is the unique minimiser of the following loss function

over u ∈ R , the prediction of mean return at x :

<!-- formula-not-decoded -->

This well-known characterisation of the expectation of a random variable is readily verified by, for example, observing that the loss is convex and differentiable in u , and solving the equation ∂ u L π x ( u ) = 0. This motivates an approach to learning V π ( x ) based on stochastic gradient descent on the loss function L π x . We maintain an estimate V ∈ R X of the value function, and each time a trajectory ( X t , A t , R t ) t ≥ 0 beginning at state x is observed, we can obtain an unbiased estimator of the negative gradient of L π x ( V ( x )) as

<!-- formula-not-decoded -->

and update V ( x ) by taking a step in the direction of this negative gradient, with some step size α :

<!-- formula-not-decoded -->

This is a Monte Carlo algorithm, so called because it uses Monte Carlo samples of the random return to update the estimate V .

A popular alternative to this Monte Carlo algorithm is temporal-difference learning, which replaces samples from the random return with a bootstrapped approximation to the return, obtained from a transition ( x, A, R, X ′ ) by combining the immediate reward R with the current estimate of the expected return obtained at X ′ , resulting in the return estimate

<!-- formula-not-decoded -->

and the corresponding update rule

<!-- formula-not-decoded -->

While the mean-return estimator in Expression (4) is generally biased , since V ( X ′ ) is not generally equal to the true expected return V π ( X ′ ), it is often a lower-variance estimate, since we are replacing the random return from X ′ with an estimate of its expectation (Sutton, 1988; Sutton and Barto, 2018; Kearns and Singh, 2000).

This motivates the TD learning rule in Expression (5) based on the Monte Carlo update rule in Expression (3), with the understanding that this algorithm can be applied more generally, with access only to sampled transitions (rather than full trajectories), and may result in more accurate estimates of the value function, due to lower-variance updates, and the propensity of TD algorithms to 'share information' across states. Note however that this does not prove anything about the behaviour of temporal-difference learning, and a fully rigorous theory of the asymptotic behaviour emerged several years after TD methods were formally introduced (Sutton, 1984, 1988; Watkins, 1989; Watkins and Dayan, 1992; Dayan, 1992; Dayan and Sejnowski, 1994; Jaakkola et al., 1994; Tsitsiklis, 1994).

## 3 Quantile Temporal-Difference Learning and Quantile Dynamic Programming

We now present the main algorithms of study in this paper.

## 3.1 Quantile Regression

To motivate QTD, we begin by considering how we might adapt a Monte Carlo algorithm such as that in Expression (3) to learn about the distribution of returns, rather than just their expected value. We cannot learn the return distribution in its entirety with a finite collection of parameters; the space of return distributions is infinite-dimensional, so we must instead be satisfied with learning an approximation of the return distribution by selecting a probability distribution representation (Bellemare et al., 2023, Chapter 5): a subset of probability distributions parametrised by a finite-dimensional set of parameters. The approach of quantile temporal-difference learning is to learn an approximation of the form

<!-- formula-not-decoded -->

an equally-weighted mixture of Dirac deltas, for each state x ∈ X . The quantile-based approach to distributional reinforcement learning aims to have the particle locations ( θ ( x, i )) m i =1 approximate certain quantiles of η π ( x ).

Definition 1 For a probability distribution ν ∈ P ( R ) and parameter τ ∈ (0 , 1) , the set of τ -quantiles of ν is given by the set

<!-- formula-not-decoded -->

where F ν : R → [0 , 1] is the CDF of ν , defined by F ν ( t ) = P Z ∼ ν ( Z ≤ t ) for all t ∈ R .

Expanding on this definition, if the set { z ∈ R : F ν ( z ) = τ } is non-empty, then the τ -quantiles are precisely the values z such that P Z ∼ ν ( Z ≤ z ) = τ . If however this set is empty (which may arise when F ν has points of discontinuity), then the quantile is the smallest value y such that P Z ∼ ν ( Z ≤ y ) &gt; τ . Note also that if F ν is strictly increasing, this guarantees uniqueness of each τ -quantile for τ ∈ (0 , 1); this is often a useful property in the analysis we consider later. These different cases are illustrated in Figure 1. The generalised inverse CDF of ν , F -1 ν : (0 , 1) → R , is defined by

<!-- formula-not-decoded -->

and provides a way of uniquely specifying a quantile for each level τ . In cases where there is not a unique τ -quantile (see Figure 1), F -1 ν ( τ ) corresponds to the left-most or least valid τ -quantile. We also introduce the notation

<!-- formula-not-decoded -->

which corresponds to the right-most or greatest τ -quantile; notice the strict inequality that appears in the definition, in contrast to that of F -1 ν ( τ ). If F -1 ν is continuous at τ , then

Figure 1: The three distinct scenarios that arise in defining quantiles. Firstly, there is a value z 1 for which F ν ( z 1 ) = τ 1 and at which F ν is strictly increasing. Therefore z 1 is the unique τ 1 -quantile of ν . Next, there is an interval [ z 2 , z ′ 2 ] on which F ν equals τ 2 , therefore all elements in this interval are τ 2 -quantiles of ν . Finally, there is no value z such that F ν ( z ) = τ 3 , and the unique τ 3 -quantile is therefore defined by the infimum part of the definition.

<!-- image -->

F -1 ν ( τ ) = ¯ F -1 ν ( τ ), as is the case for τ = τ 1 and τ = τ 3 in Figure 1. However, if F ν has a flat region for the value τ (as is the case for τ = τ 2 in Figure 1), then F -1 ν ( τ ) and ¯ F -1 ν ( τ ) are distinct, and correspond to the boundary points of this flat region.

Algorithmically, we aim for θ ( x, i ) to approximate a τ i -quantile of η π ( x ), where τ i = 2 i -1 / 2 m . To build learning algorithms that achieve this, we require an incremental algorithm that updates θ ( x, i ) in response to samples from the target distribution η π ( x ), which converges to a 2 i -1 / 2 m -quantile of η π ( x ).

Such an approach is available by using the quantile regression loss. We define the quantile regression loss associated with distribution ν ∈ P ( R ) and quantile level τ ∈ (0 , 1) as a function of v by

<!-- formula-not-decoded -->

This loss is the expectation of an asymmetric absolute value loss, in which positive and negative errors are weighted according to the parameters τ and 1 -τ respectively. Just as the expected squared loss encountered above encodes the mean as its unique minimiser, the quantile regression loss encodes the τ -quantiles of ν as the unique minimisers; see, for example, Koenker (2005) for further background. Thus, applying the quantile regression loss to the problem of estimating τ -quantiles of the return distribution, we arrive at the loss

<!-- formula-not-decoded -->

Given an observed return ∑ t ≥ 0 γ t R t from the state x , we therefore have that an unbiased estimator of the negative gradient 1 of this loss is

<!-- formula-not-decoded -->

which motivates an update rule of the form

<!-- formula-not-decoded -->

This can be rewritten as

<!-- formula-not-decoded -->

This is essentially the application of the stochastic gradient descent method for quantile regression to learning quantiles of the return distribution.

## 3.2 Quantile Temporal-Difference Learning

We can motivate and describe the quantile temporal-difference learning algorithm (Dabney et al., 2018b; Bellemare et al., 2023) by modifying the Monte Carlo algorithm in Expression (8) in a similar manner to the modification that led to the TD algorithm in Expression (5). We replace the Monte Carlo return

<!-- formula-not-decoded -->

based on a full trajectory, with an approximate sample from the return distribution derived from an observed transition ( x, R, X ′ ), and the estimate η ( X ′ ) of the return distribution at state X ′ . If the return distribution estimate η ( X ′ ) takes the form given in Equation (6), as is the case for the probability distribution representation considered here, then such a sample return is obtained as

<!-- formula-not-decoded -->

with J sampled uniformly from { 1 , . . . , m } . This yields the update rule

<!-- formula-not-decoded -->

✶ We can consider also a variance-reduced version of this update, in which we average over updates performed under different realisations of J , leading to the update

<!-- formula-not-decoded -->

1. Technically speaking, we are assuming that differentiation and expectation can be interchanged here. Further, under certain circumstances the loss is only sub -differentiable. As our principal goal in this section is to provide intuition for QTD, we do not comment further on these technical details here. The convergence results later in the paper deal with these issues carefully.

This is precisely the quantile temporal-difference learning update, presented in Algorithm 1 below, which underlies many recent successful applications of reinforcement learning at scale (Dabney et al., 2018b,a; Yang et al., 2019; Bellemare et al., 2020; Wurman et al., 2022; Fawzi et al., 2022). Similar to other temporal-difference learning algorithms, QTD updates its parameters (( θ ( x, i )) m i =1 : x ∈ X ) on the basis of sample transitions ( x, r, x ′ ) generated through interaction with the environment via the policy π , comprising a state, reward, and next state.

## Algorithm 1 QTD update

```
Require: Quantile estimates θ ∈ R X× [ m ] , Observed transition ( x, r, x ′ ) , Learning rate α . 1: Set τ i = 2 i -1 2 m for each i = 1 , . . . , m . 2: for i = 1 , . . . , m do 3: Set θ ′ ( x, i ) ← θ ( x, i ) + α 1 m ∑ m j =1 [ τ i -✶ { r + γθ ( x ′ , j ) -θ ( x, i ) < 0 }] 4: end for 5: for i = 1 , . . . , m do 6: Set θ ( x, i ) ← θ ′ ( x, i ) 7: end for 8: return (( θ ′ ( x, i )) m i =1 : x ∈ X )
```

Whilst the QTD update makes use of temporal-difference errors r + γθ ( x ′ , j ) -θ ( x, i ), there are two key differences to the use of analogous quantities in classical TD learning. First, the TD errors influence the update only through their sign, not their magnitude. Second, the predictions at each state ( θ ( x, i )) m i =1 are indexed by i , and each update includes a distinct term τ i (equal to 2 i -1 / 2 m ). The presence of these terms causes the learnt parameters to make distinct predictions, as described in Section 3.1. Practical implementations of QTD use these precise values for τ i , equally spaced out on [0 , 1], as proposed by Dabney et al. (2018b). Much of the analysis in this paper goes through straightforwardly for other values of τ i , though we will see in Section 6 that this choice is well motivated in that it provides the best bounds on distribution approximation. The tabular QTD algorithm as described in Algorithm 1 uses a factor O ( m ) times more memory than an analogous classical TD algorithm, owing to the need to store multiple predictions at each state, though the scaling with the size of the state space is the same as for classical TD. For further discussion of the computational complexity of QTD, see Rowland et al. (Appendix A.3; 2023). Much of the analysis in this paper goes through straightforwardly for other values of τ i , though we will see in Section 6 that this choice is well motivated in that it provides the best bounds on distribution approximation.

The discussion above provides motivation for the form of the QTD update given in Algorithm 1, and intuition as to why this algorithm might perform reasonably, and learn a sensible approximation to the return distribution. However, it stops short of providing an explanation of how the algorithm should be expected to behave, or providing any theoretical guarantees as to what the algorithm will in fact converge to. A core goal of the sections that follow is to answer these questions, and put QTD on firm theoretical footing.

## 3.3 Motivating Examples

Before undertaking an analysis of QTD, we pause to provide several numerical examples of its behaviour in small environments. These examples provide further intuition for the characteristics of the algorithm, illustrate the breadth of qualitative behaviours it can exhibit, and provide motivation for the kinds of theoretical questions we might hope to answer.

Example 2 Consider the chain MDP illustrated at the top of Figure 2. The random return at each state is a sum of independent Gaussian random variables, and hence the return distribution at each state is Gaussian. The centre plot in Figure 2 illustrates the evolution of m = 5 quantile estimates learnt by QTD, using a constant learning rate of 0.01, and updating all states at each update. The estimated quantile values eventually settle after around 6,000 updates, with small oscillations around this point. The bottom of Figure 2 compares the true return distribution at each state (in blue), with the approximation learnt by QTD (in black), and the approximation obtained with the true value of the five quantiles of interest (grey). The behaviour of QTD in this case raises several questions: Can it be shown that QTD is guaranteed to stabilise/converge around a certain point? Can a guarantee be given on the quality of the approximate distributions learnt by QTD?

Example 3 For a different perspective on the behaviour of QTD, consider a two-state MDP with transition dynamics as illustrated in the top-left of Figure 3, and discount factor γ = 0 . 5 . The reward obtained when transitioning from state x 1 is distributed as N(2, 1), and the reward obtained when transitioning from state x 2 is distributed as N(-1, 1); here, we write N ( µ, σ 2 ) for the normal distribution with mean µ and variance σ 2 . We consider the case of learning a single quantile (the median) at each of these two states, taking m = 1 ; this allows us to plot the full phase space of the QTD algorithm in a two-dimensional plot.

The top-right of Figure 3 shows a path taken by QTD under this MDP. In addition, the streamplot illustrates the direction of the expected update that QTD undertakes at each point in phase space. We empirically observe convergence of the algorithm to a point. Additionally, the expected update direction changes smoothly; the result is a vector field that appears to point towards the point of convergence from all directions.

The bottom-left of Figure 3 shows a path taken by QTD under a modified version of the MDP, in which the reward distributions N(2, 1) and N(-1, 1) are replaced with δ 2 and δ -1 , respectively. We observe that the algorithm still converges to a point, although the vector field of expected update directions is now piecewise constant, with discontinuities along several lines. This behaviour is typical of QTD; the less 'smooth' the reward distributions in the MDP, the more abrupt the changes in behaviour we typically observe with QTD.

Finally, we consider a modified version of the MDP in which all transition probabilities are 1 / 2 , rewards from state x 1 are always 2 , and rewards from state x 2 are always -1 . In this case, QTD no longer appears to converge to a point, but instead converges to the set bounded by the four grey lines appearing in the bottom-right of Figure 3, and subsequently performing a random walk over this set. This collection of examples illustrates that QTD can exhibit a fairly wide family of behaviours depending on the characteristics of the environment. In particular, non-uniqueness of quantiles in reward distributions (corresponding to flat regions in reward distribution CDFs) can lead to multiple possible limit points, and

Figure 2: Top: A chain MDP with four states. Each transition yields a normally-distributed reward; from x 3 , the episode ends. The discount factor is γ = 0 . 9. Centre-top: The progress of QTD, run with m = 5 quantiles, over the course of 10,000 updates. The vertical axis corresponds to the predicted quantile values. Centre-bottom: The true CDF of the return distribution (blue) at each state, along with the final estimate produced by QTD (black), and the approximation produced by the quantiles of the return distribution (grey). Bottom: The PDF of the return distribution (blue) at each state, along with the final quantile approximation produced by QTD (black).

<!-- image -->

Figure 3: Top left: The example Markov decision process described in Example 3. Top right: Example dynamics of QTD with m = 1 in this environment, when reward distributions are Gaussian. Also included are the directions of expected update, in blue. Bottom left: Example dynamics and expected update directions when reward distributions are Dirac deltas. Bottom right: Example dynamics and expected updates with modified environment transition probabilities.

<!-- image -->

discontinuities in reward distributions can lead to discontinuous changes in expected updates; by contrast, reward distributions that are absolutely continuous lead to smooth changes in expected dynamics.

## 3.4 Quantile Dynamic Programming

Recall the QTD update given in Equation (10). As described in Section 3.2, this update serves, on average, to move θ ( x, i ) in the direction of the τ i -quantiles of the distribution of the random variable R + θ ( X ′ , J ), where ( x, R, X ′ ) is a random transition generated by interacting with the environment using π , and J ∼ Unif( { 1 , . . . , m } ).

Suppose we were able to update θ ( x, i ) not just with a single gradient step in this direction, but instead were able to update it to take on exactly this quantile value. This motivates a

dynamic programming alternative to QTD, quantile dynamic programming (QDP), which directly calculates these quantiles iteratively, in a similar manner to iterative policy evaluation in classical reinforcement learning (Bertsekas and Tsitsiklis, 1996).

The mathematical structure of such an algorithm is given in Algorithm 2. This stops short of being an implementable algorithm, since we do not describe in what format the transition probabilities and reward distributions are available, which are required to evaluate the inverse CDFs that arise in the algorithm. However, for MDPs in which transition probabilities and reward distributions are available, QDP is an algorithmic framework of interest in its own right, and to this end we provide several concrete implementations in Appendix B.

The QDP template in Algorithm 2 is parametrised by the interpolation parameters λ ∈ [0 , 1] X× [ m ] . These parameters control exactly which quantile is chosen when the desired quantile level τ i corresponds to a flat region of the CDF for the distribution ν (the second case in Figure 1). QDP was originally presented by Bellemare et al. (2023) in the case λ ( x, i ) ≡ 0; the presentation here generalises QDP to a family of algorithms, parametrised by λ .

Our interest in QDP stems from the fact that QTD can be viewed as approximating the behaviour of the QDP algorithms, without requiring access to the transition structure and reward distributions of the environment. In particular, we will show that under appropriate conditions, the asymptotic behaviour of QTD and QDP are equivalent: they both converge to the same limiting points. Figure 4 illustrates the behaviour of the QDP algorithm in the environment described in Example 3; since the reward distributions in this example have strictly increasing CDFs, QDP behaves identically for all choices of interpolation parameters λ . QTD and QDP appear to have the same asymptotic behaviour, converging to the same limiting point. In cases where QTD appears to converge to a set, such as in the bottom-right plot of Figure 3, the relationship is slightly more complicated, and there is a correspondence between the asymptotic behaviour of QTD and the family of dynamic programming algorithms parametrised by λ , as illustrated at the bottom of Figure 4. Thus, to understand the asymptotic behaviour of QTD, we begin by analysing the asymptotic behaviour of QDP.

## Algorithm 2 Quantile dynamic programming

```
Require: Quantile estimates (( θ ( x, i )) m i =1 : x ∈ X ), Interpolation parameters λ ∈ [0 , 1] X× [ m ] . 1: for x ∈ X do 2: Let ( x, R, X ′ ) be a random transition under π , and J ∼ Unif( { 1 , . . . , m } ). 3: Set ν to be the distribution of R + γθ ( X ′ , J ). 4: for i = 1 , . . . , m do 5: Set θ ( x, i ) ← (1 -λ ( x, i )) F -1 ν ( τ i ) + λ ( x, i ) ¯ F -1 ν ( τ i ). 6: end for 7: end for 8: return (( θ ( x, i )) m i =1 : x ∈ X )
```

Figure 4: Top left: Illustration of QDP (dashed purple) and QTD (solid red) on the first MDP from Example 3, with Gaussian rewards. Top right: Illustration of QDP and QTD on the second MDP from Example 3, with deterministic rewards. Bottom: Values of λ and corresponding fixed points of QDP in the final MDP from Example 3.

<!-- image -->

## 4 Convergence of Quantile Dynamic Programming

We can decompose the update QDP performs into the composition of several operators. Algorithm 2 manipulates tables of the form (( θ ( x, i )) m i =1 : x ∈ X ). For a given state x , the vector ( θ ( x, i )) m i =1 represents the estimated 2 i -1 / 2 m -quantiles of the return distribution at state x , for i = 1 , . . . , m . In mathematically analysing the algorithm, it is useful to be able to refer to the distribution encoded by these quantiles:

<!-- formula-not-decoded -->

and reason about the transformations undertaken by Algorithm 2 directly in terms of distributions. To this end, if we write η ( x ) ∈ P ( R ) for the probability distribution associated with the quantile estimates ( θ ( x, i )) m i =1 , we can interpret the transformation performed by Algorithm 2 as comprising two parts, which we now describe in turn.

First, the variable η ( x ) is assigned the distribution of R + γG ( X ′ ), where R,X ′ are the random reward and next-state encountered from the initial state x with policy π , and ( G ( y ) : y ∈ X ) is an independent collection of random variables, with each G ( y ) distributed according to η ( y ).

We write T π : P ( R ) X → P ( R ) X for this transformation. The function T π is known as the distributional Bellman operator (Bellemare et al., 2017; Rowland et al., 2018; Bellemare et al., 2023). In terms of the above definition via distributions of random variables, T π can be written

<!-- formula-not-decoded -->

where ( x, R, X ′ ) is a random environment transition beginning at x , independent of ( G ( y ) : y ∈ X ), and D π extracts the distribution of its argument when ( x, R, X ′ ) is generated by sampling an action from π . See Bellemare et al. (2023) for further background on the distributional Bellman operator.

In general, T π η may comprise much more complicated distributions than η itself, with many more atoms, or possibly infinite support, if reward distributions are infinitely-supported. Algorithm 2 does not return these full transformed distributions, but rather approximations, or projections , of these distributions, obtained by keeping only information about certain quantiles (in the inner for-loop of Algorithm 2); this is the second distribution transformation the algorithm undertakes. Each choice of interpolation parameters λ corresponds to a different projection operator, denoted Π λ : P ( R ) X → P ( R ) X , and defined by

<!-- formula-not-decoded -->

Thus, the composition Π λ T π , the projected distributional Bellman operator, is a transformation on the space of return-distribution functions P ( R ) X . We will also find it useful to abuse notation slightly and consider Π λ T π as an operator on the space R X× [ m ] of parameters that QDP and QTD operate over. The understanding is that an input θ ∈ R X× [ m ] is

first re-interpreted as a collection of distributions as in Expression (11), with Π λ T π applied as defined above to this collection of probability distributions, and then finally extracting the support of the output distributions, which take the form

<!-- formula-not-decoded -->

to return an element of R X× [ m ] . We will also write T π θ for the element of P ( R ) X obtained by applying T π to the distributions ( η ( x ) : x ∈ X ) defined by

<!-- formula-not-decoded -->

Remark 4 This convention highlights that there are two complementary views of distributional reinforcement learning algorithms, through finite-dimensional sets of parameters, and through probability distributions. The view in terms of probability distributions is often useful in contraction analysis, and in measuring approximation error, while we will see that the parameter view is key to the stochastic approximation analysis that follows, and is ultimately the way in which these algorithms are implemented.

With this convention, Π λ T π θ is precisely the table θ ′ output by Algorithm 2 on input θ , and so the QDP algorithm is mathematically equivalent to repeated application of the operator Π λ T π to an initial collection of quantile estimates. To understand the long-term behaviour of QDP, we can therefore seek to understand this projected operator Π λ T π .

## 4.1 Convergence Analysis

We will show that Π λ T π is a contraction mapping with respect to an appropriate metric over return-distribution functions. Building on the analysis in the case of λ ≡ 0 carried out by Dabney et al. (2018b) and Bellemare et al. (2023), we use the Wasserstein-∞ metric w ∞ : P ( R ) × P ( R ) → [0 , ∞ ], defined by

<!-- formula-not-decoded -->

and its extension to return-distribution functions, ¯ w ∞ : P ( R ) X × P ( R ) X → [0 , ∞ ], given by

<!-- formula-not-decoded -->

Both w ∞ and ¯ w ∞ fulfil all the requirements of a metric, except that they may assign infinite distances (Villani, 2009; see also Bellemare et al., 2023 for a detailed discussion specifically in the context of reinforcement learning). We must therefore take some care as to when distances are finite. The following is established by Bellemare et al. (2023, Proposition 4.15).

Proposition 5 The distributional Bellman operator T π : P ( R ) X → P ( R ) X is a γ -contraction with respect to ¯ w ∞ . That is,

<!-- formula-not-decoded -->

for all η, η ′ ∈ P ( R ) X .

Next, we show that the projection operator Π λ cannot expand distances as measured by ¯ w ∞ , generalising the proof given by Bellemare et al. (2023) in the case λ ≡ 0; the proof is given in Appendix A.1.

Proposition 6 The projection operator Π λ : P ( R ) X → P ( R ) X is a non-expansion with respect to ¯ w ∞ . That is, for any η, η ′ ∈ P ( R ) X , we have

<!-- formula-not-decoded -->

Finally, we put these two results together to obtain our desired conclusion. In stating this result, it is useful here to introduce the notation

<!-- formula-not-decoded -->

for the set of probability distributions representable with m quantile locations.

Proposition 7 The projected operator Π λ T π : F X Q ,m → F X Q ,m is a γ -contraction with respect to ¯ w ∞ . Hence, Π λ T π has a unique fixed point in F X Q ,m , which we denote ˆ η π λ . Further, given any initial η 0 ∈ F X Q ,m , the sequence ( η k ) ∞ k =0 defined iteratively by η k +1 = Π λ T π η k for k ≥ 0 satisfies ¯ w ∞ ( η k , ˆ η π λ ) ≤ γ k ¯ w ∞ ( η 0 , ˆ η π λ ) → 0 .

Proof That Π λ T π : F X Q ,m → F X Q ,m is a γ -contraction with respect to ¯ w ∞ follows directly from Propositions 5 and 6:

<!-- formula-not-decoded -->

Next, observe that ¯ w ∞ assigns finite distance to all pairs of return-distribution functions in F X Q ,m , and further, this set is complete with respect to ¯ w ∞ . Hence, we may apply Banach's fixed point theorem to obtain the existence of the unique fixed point ˆ η π λ in F X Q ,m . The final claim follows by induction, and the contraction property established for Π λ T π .

Note that the fixed point ˆ η π λ depends on λ , and therefore implicitly on m . We also introduce the notation ˆ θ π λ ∈ R X× [ m ] for the parameters of this collection of distributions, which is what the QDP algorithm really operates over, so that we have

<!-- formula-not-decoded -->

Note that the convergence result of Proposition 7 also implies convergence of the estimated quantile locations to ˆ θ π λ . In Section 6, we will analyse the fixed point ˆ η π λ , and understand how closely it approximates the true return-distribution function η π . For now, having established convergence of QDP through contraction mapping theory, we can return to QTD and demonstrate its own convergence to the same fixed points.

## 5 Convergence of Quantile Temporal-Difference Learning

We now present the convergence analysis of QTD. We will consider a synchronous version of QTD, in which all states are updated using independent transitions at each algorithm step, given by:

<!-- formula-not-decoded -->

where given x and k , we have ( R k ( x ) , X ′ k ( x )) ∼ P π ( ·| x ), independently of the transitions used at all other states/time steps, and ( α k ) ∞ k =0 is a sequence of step sizes. The assumption of synchronous updates makes the analysis easier to present, and means that our results follow classical approaches to stochastic approximation with differential inclusions (Bena¨ ım et al., 2005). It is also possible to extend the analysis to the asynchronous case, where a single state is updated at each algorithm time step (as would be the case in fully online QTD, or an implementation using a replay buffer); see Section 5.7. We now state the main convergence result of the paper.

Theorem 8 Consider the sequence ( θ k ) ∞ k =0 defined by an initial point θ 0 ∈ R X× [ m ] , the iterative update in Equation (13) , and non-negative step sizes satisfying the condition

<!-- formula-not-decoded -->

Then ( θ k ) ∞ k =0 converges almost surely to the set of fixed points of the projected distributional Bellman operators { Π λ T π : λ ∈ [0 , 1] X× [ m ] } ; that is,

<!-- formula-not-decoded -->

with probability 1.

Of particular note is the generality of this result. It does not require finite-variance conditions on rewards (as is typically the case with convergence results for classical TD); it holds for any collection of reward distributions with the finite mean property set out at the beginning of the paper. Some intuition as to why this is the case is that the finite-variance conditions typically encountered are to ensure that the updates performed in classical TD learning cannot grow in magnitude too rapidly. Since the updates performed in QTD are bounded, this is not a concern, meaning that the proof does not rely on such conditions. We note also that the step size conditions are weaker than the typical Robbins-Monro conditions used in classical TD analyses (see, for example, Bertsekas and Tsitsiklis, 1996), which enforce square-summability, also to avoid the possibility of divergence due to unbounded noise in the classical TD learning.

The proof is based on the ODE method for stochastic approximation; in particular we use the framework set out by Bena¨ ım (1999) and Bena¨ ım et al. (2005). This involves interpreting the QTD update as a noisy Euler discretisation of a differential equation (or more generally, a differential inclusion). The broad steps are then to argue that the trajectories of the

differential equation/inclusion converge to some set of fixed points in a suitable way (that is, in such a way that is robust to small perturbations), and that the asymptotic behaviour of QTD, forming a noisy Euler discretisation, matches the asymptotic behaviour of the true trajectories. This then allows us to deduce that the QTD iterates converge to the same set of fixed points as the true trajectories. We begin by elucidating the connection to differential equations and differential inclusions.

## 5.1 The QTD Differential Equation

Taking the expectation over the random variables R k ( x ) and X ′ k ( x ) in Equation (13) conditional on the algorithm history up to time k yields an expected increment of

<!-- formula-not-decoded -->

We now briefly introduce an assumption on the MDP reward structure that simplifies the analysis that follows. This assumption guarantees that the two 'difficult' cases of flat and vertical regions of CDFs (see Figure 1) do not arise; note that this assumptions removes the possibility of multiple fixed points or discontinuous expected dynamics, as described in Example 3. We will lift this assumption later.

Assumption 9 For each state x ∈ X , the reward distribution at x has a CDF which is strictly increasing, and Lipschitz continuous.

As described in Section 4, the distribution of R + θ k ( X ′ , J ) given the initial state x is in fact equal to the application of the distributional Bellman operator T π applied to the return-distribution function η k ∈ P ( R ) X given by

<!-- formula-not-decoded -->

Under Assumption 9, and in particular the assumption of continuous reward CDFs, this yields a concise rewriting of the increment as

<!-- formula-not-decoded -->

We may therefore intuitively interpret Equation (13) as a noisy discretisation of the differential equation

<!-- formula-not-decoded -->

which we refer to as the QTD differential equation (or QTD ODE). Note also that Assumption 9 guarantees the global existence and uniqueness of solutions to this differential equation, by the Cauchy-Lipschitz theorem.

Remark 10 Calling back to Figure 3, the trajectories of the QTD ODE are obtained precisely by integrating the vector fields that appear in these plots. In contrast to the ODE that emerges when analysing classical TD learning (both in tabular and linear function approximation settings) (Tsitsiklis and Van Roy, 1997), the right-hand side of Equation (16) is non-linear in the parameters ϑ t , meaning that we are outside the domain of linear stochastic approximation methods.

## 5.2 The QTD Differential Inclusion

In lifting Assumption 9, a few complications arise. Firstly, if F ( T π θ )( x ) is not continuous at θ ( x, i ), then the right-hand side of the QTD ODE in Equation (16) is modified to

<!-- formula-not-decoded -->

the difference is the strict inequality. Now the right-hand side of the differential equation itself is not continuous; in general, solutions may not even exist for this differential equation. The situation is illustrated in the bottom-left panel of Figure 3; the lines in this plot illustrate points of discontinuity of the vector field to be integrated, and there are instances where the vector field either side of such a line of discontinuity 'pushes' back into the discontinuity. In such cases, the differential equation has no solution in the usual sense. This phenomenon is known as sliding, or sticking, from cases when it arises in the modelling of physical systems with potentially discontinuous forces (such as static friction models in mechanics).

Filippov (1960) proposed a method to deal with such non-existence issues, by relaxing the definition of the dynamics at points of discontinuity. Technically, Filippov's proposal is to allow the derivative to take on any value in the convex hull of possible limiting values as we approach the point of discontinuity. In our case, we consider redefining the dynamics at points of discontinuity as follows:

<!-- formula-not-decoded -->

where F ( T π ϑ t )( x ) ( ϑ t ( x, i ) -) denotes lim s ↑ ϑ t ( x,i ) F ( T π ϑ t )( x ) ( s ). This refines the dynamics so that for each coordinate ( x, i ), the derivative may take on either the left or right limit around ϑ t ( x, i ), or any value in between; this is a looser relaxation than Filippov's proposal, and is easier to work with in our analysis.

Equation (17) is a differential inclusion , as opposed to a differential equation; the derivative is constrained to a set at each instant, rather than constrained to a single value. We refer to Equation (17) specifically as the QTD differential inclusion (or QTD DI). Note that if F ( T π θ )( x ) is continuous at θ ( x, i ), then the right-hand side of Equation (17) reduces to the singleton { τ i -F ( T π θ )( x ) ( θ ( x, i )) } , and we thus obtain the ODE dynamics considered previously.

## 5.3 Solutions of Differential Inclusions

We briefly recall some key concepts regarding solutions of differential inclusions; a full review of the theory of differential inclusions is beyond the scope of this article, and we refer the reader to the standard references by Aubin and Cellina (1984), Clarke et al. (1998), and Smirnov (2002).

Definition 11 Let H : R n ⇒ R n be a set-valued map. The path ( z t ) t ≥ 0 is a solution to the differential inclusion ∂ t z t ∈ H ( z t ) if there exists an integrable function g : [0 , ∞ ) → R n such that

<!-- formula-not-decoded -->

for all t ≥ 0 , and g t ∈ H ( z t ) for almost all t ≥ 0 .

Note that Definition 11 does not require that z t is differentiable with derivative g t , but only the weaker integration condition in Equation (18). We then have the following existence result (see, for example, Smirnov, 2002 for a proof).

Proposition 12 Consider a set-valued map H : R n ⇒ R n , and suppose that H is a Marchaud map : that is,

- the set { ( z, h ) : z ∈ R n , h ∈ H ( z ) } is closed.
- For all z ∈ R n , H ( z ) is non-empty, compact, and convex.
- There exists a constant C &gt; 0 such that for all z ∈ R n ,

<!-- formula-not-decoded -->

Then the differential inclusion ∂ t z t ∈ H ( z t ) has a global solution, for any initial condition.

It is readily verified that the QTD DI satisfies the requirements of this result, and we are therefore guaranteed global solutions to this differential inclusion, under any initial conditions.

## 5.4 Asymptotic Behaviour of Differential Inclusion Trajectories

Recall that our goal is to show that the trajectories of the QTD differential inclusion must approach the fixed points of QDP. A key tool in doing so is the notion of a Lyapunov function; the following definition is based on Bena¨ ım et al. (2005).

Definition 13 Consider a Marchaud map H : R n ⇒ R n , and a subset Λ ⊆ R n . A continuous function L : R n → [0 , ∞ ) is said to be a Lyapunov function for the differential inclusion ∂ t z t ∈ H ( z t ) and subset Λ if for any solution ( z t ) t ≥ 0 of the differential inclusion and 0 ≤ s &lt; t , we have L ( z t ) &lt; L ( z s ) for all z s ̸∈ Λ and L ( z ) = 0 for all z ∈ Λ .

Intuitively, L is a Lyapunov function if it decreases along trajectories of the differential inclusion, and is minimal precisely on Λ. Lyapunov functions are a central tool in dynamical systems for demonstrating convergence, and in the sections that follow, we will consider the QTD differential inclusion, and take Λ to be the set of fixed points of the family of QDP algorithms.

## 5.5 QTD as a Stochastic Approximation to the QTD Differential Inclusion

We can now give the proof of our core result, Theorem 8. The abstract stochastic approximation result at the heart of the convergence proof of QTD is presented below. It is a special case of the general framework described by Bena¨ ım et al. (2005), the proof of which is given in Appendix A.2.

Theorem 14 Consider a Marchaud map H : R n ⇒ R n , and the corresponding differential inclusion ∂ t z t ∈ H ( z t ) . Suppose there exists a Lyapunov function L for this differential inclusion and a subset Λ ⊆ R n . Suppose also that we have a sequence ( θ k ) k ≥ 0 satisfying

<!-- formula-not-decoded -->

where:

- ( α k ) ∞ k =0 satisfy the conditions ∑ ∞ k =0 α k = ∞ , α k = o (1 / log( k )) ;
- g k ∈ H ( θ k ) for all k ≥ 0 ;
- ( w k ) ∞ k =0 is a bounded martingale difference sequence with respect to the natural filtration generated by ( θ k ) ∞ k =0 ; that is, there is an absolute constant C such that ∥ w k ∥ ∞ &lt; C almost surely, and E [ w k | θ 0 , . . . , θ k ] = 0 .

If further ( θ k ) ∞ k =0 is bounded almost surely (that is, sup k ≥ 0 ∥ θ k ∥ ∞ &lt; ∞ almost surely), then θ k → Λ almost surely.

The intuition behind the conditions of the theorem are as follows. The Marchaud map condition ensures the differential inclusion of interest has global solutions. The existence of the Lyapunov function guarantees that trajectories of the differential inclusion converge in a suitably stable sense to Λ. The step size conditions, martingale difference condition, and boundedness conditions mean that the iterates ( θ k ) ∞ k =0 will closely track the differential inclusion trajectories, and hence exhibit the same asymptotic behaviour. We can now give the proof of Theorem 8, first requiring the following proposition, which is proven in Appendix A.3.

Proposition 15 Under the conditions of Theorem 8, the iterates ( θ k ) ∞ k =0 are bounded almost surely.

Proof (Proof of Theorem 8) We see that for the QTD sequence ( θ k ) ∞ k =0 and the QTD DI and QDP invariant set Λ = { ˆ θ π λ : λ ∈ [0 , 1] X× [ m ] } , the conditions of Theorem 14 are satisfied, except perhaps for the boundedness of ( θ k ) ∞ k =0 , and the existence of the Lyapunov function. The fact that the sequence ( θ k ) ∞ k =0 is bounded almost surely is Proposition 15; its proof is somewhat technical, and given in the appendix. The construction of a valid Lyapunov function is given in Proposition 18 below, which completes the proof.

Remark 16 What makes the relaxation to the differential inclusion work in this analysis? We have already seen that some kind of relaxation of the dynamics is required in order to define a valid continuous-time dynamical system; the original ODE may not have solutions in general. If we relax the dynamics too much (an extreme example would be the differential inclusion ϑ t ( x, i ) ∈ R ), what goes wrong? The answer is that there are too many resulting solutions, which do not exhibit the desired asymptotic behaviour. Thus, the differential inclusion in Equation (17) is in some sense just the right level of relaxation of the differential equation we started with, since trajectories of the QTD DI are still guaranteed to converge to the QDP fixed points.

## 5.6 A Lyapunov Function for the QDP Fixed Points

In this section, we prove the existence of a Lyapunov function required in order to use Theorem 14 to prove Theorem 8. We treat the case when Assumption 9 holds separately as the proof is instructive, and considerably simpler than the general case. Under this assumption, note that all projections Π λ behave identically on the image of T π , since all

resulting CDFs are strictly increasing. We therefore introduce the notation Π to refer to any such projection in this case, and the notation ˆ θ π m to refer to the unique fixed point of Π T π .

Proposition 17 Consider the ODE in Equation (16) , and suppose Assumption 9 holds. A Lyapunov function for the equilibrium point ˆ θ π m is given by

<!-- formula-not-decoded -->

Proof We immediately observe that L is continuous, non-negative, and takes on the value 0 only at ˆ θ π m . To show that L ( ϑ t ) is decreasing, where ( ϑ t ) t ≥ 0 is an ODE trajectory, suppose ( x, i ) is a state-index pair attaining the maximum in L ( ϑ t ). It is sufficient to show that ϑ t ( x, i ) is moving towards ˆ θ π m ( x, i ), or expressed mathematically,

<!-- formula-not-decoded -->

where we use a S = b as shorthand for equality of signs sign( a ) = sign( b ), where

<!-- formula-not-decoded -->

Now note that

<!-- formula-not-decoded -->

where the sign equality follows from Assumption 9; since all reward CDFs are strictly increasing, so too is F ( T π ϑ t )( x ) , and so F -1 ( T π ϑ t )( x ) is strictly monotonic. Additionally, from the contractivity of Π T π with respect to w ∞ (see Proposition 7), we have

<!-- formula-not-decoded -->

the equality follows since we selected ( x, i ) attain the maximum in the definition of L ( ϑ t ). From this, we deduce

<!-- formula-not-decoded -->

which follows by considering the three cases for the sign of ˆ θ π m ( x, i ) -ϑ t ( x, i ). If the sign equals zero, then since ( x, i ) was chosen to be maximal in the definition of L ( ϑ t ), we have

ϑ t = ˆ θ π m , and hence Π T π ϑ t = ˆ θ π m , and the claim follows; both sides are equal to 0. For the case ˆ θ π m ( x, i ) -ϑ t ( x, i ) &lt; 0, then note we have

<!-- formula-not-decoded -->

as required, with the inequality above following from Equation (19). The case ˆ θ π m ( x, i ) -ϑ t ( x, i ) &gt; 0 follow similarly. We therefore have

<!-- formula-not-decoded -->

We therefore have that L ( ϑ t ) is decreasing at t , strictly so if ϑ t = ˆ θ π m , as required to establish the result.

̸

The proof of Proposition 17 also sheds further light on the mechanisms underlying the QTD algorithm. A key step in the argument is to show that for the state-index pairs ( x, i ) such that ϑ t ( x, i ) is maximally distant from the fixed point θ π m ( x, i ), the expected update under QTD moves this coordinate of the estimate in the same direction as gradient descent on a squared loss from the fixed point. However, the fact that it is only the sign of the update that has this property, and not its magnitude, means that the empirical rate of convergence and stability of QTD can be expected to be somewhat different from methods based on an L 2 loss, such as classical TD.

We now state the Lyapunov result in the general case; the proof is somewhat more involved, and is given in Appendix A.4.

## Proposition 18 The function

<!-- formula-not-decoded -->

is a Lyapunov function for the differential inclusion in Equation (17) and the set of fixed points { ˆ θ π λ : λ ∈ [0 , 1] X× [ m ] } .

## 5.7 Extension to Asynchronous QTD

Our convergence results have focused on the synchronous case of QTD. However, in practice, it is often of interest to implement asynchronous versions of TD algorithms, in which only a single state is updated at a time. More formally, an asynchronous version of QTD computes the sequence ( θ k ) k ≥ 0 defined by an initial estimate θ ∈ R X× [ m ] , a sequence of transitions ( X k , R k , X ′ k ) k ≥ 0 , and the update rule

<!-- formula-not-decoded -->

for x = X k , and θ k +1 ( x, i ) = θ k ( x, i ) otherwise. Here, the step size β x,k depends on both x and k , and is typically selected so that each state individually makes use of a fixed step size

sequence ( α k ) ∞ k =0 , by taking β x,k = α ∑ k l =0 ✶ { X l = x } . This models the online situation where a stream of experience ( X k , R k ) k ≥ 0 is generated by interacting with the environment using the policy π , and updates are performed setting X ′ k = X k +1 , and also the setting in which the tuples ( X k , R k , X ′ k ) k ≥ 0 are sampled i.i.d. from a replay buffer, among others.

Convergence of QTD in such asynchronous settings can also be proven; Perkins and Leslie (2013) extend the analysis of Bena¨ ım (1999) and Bena¨ ım et al. (2005), incorporating the approach of Borkar (1998), to obtain convergence guarantees for asynchronous stochastic approximation algorithms approximating differential inclusions. In the interest of space, we do not provide the full details of the proof here, but instead sketch the key differences that arise in the analysis in Appendix C.

## 6 Analysis of the QTD Limit Points

In general, the limiting points ˆ η π λ for QTD/QDP will not be the same as the true returndistribution function η π . On the one hand, this is clear; each return-distribution function ˆ η π λ is in the image of the projection Π λ , so each constituent probability distribution must be of the form 1 m ∑ m i =1 δ z i , whereas the true return distributions need not take on this form. In addition, the magnitude of this approximation error is not immediately clear. Each application of the projection Π λ in the dynamic programming process causes some loss of information, and the quality of the fixed point ˆ η π λ is affected by the build up of these approximations over time.

Measuring approximation error in ¯ w ∞ typically turns out to be uninformative, as ¯ w ∞ is a particularly strict notion of distance between probability distributions, as discussed in the context of distributional RL by Rowland et al. (2019) and Bellemare et al. (2023). In particular, fixed points ˆ η π λ that intuitively provide a good approximation to η π may have high ¯ w ∞ -distance, and the ¯ w ∞ -distance generally does not decrease with m (Bellemare et al., 2023). Instead, we use the Wasserstein-1 metric, and its extension to return-distribution functions, defined by

<!-- formula-not-decoded -->

for all ν, ν ′ ∈ P ( R ), and η, η ′ ∈ P ( R ) X . The following result improves on the analysis given by Bellemare et al. (2023) for the case of λ ≡ 0, establishing an upper bound on the w 1 distance between ˆ η π λ and η π for any λ , essentially by showing that the errors accumulated in dynamic programming can be made arbitrarily small by increasing m , which controls the richness of the distribution representation.

Proposition 19 For any λ ∈ [0 , 1] X× [ m ] , if all reward distributions are supported on [ R min , R max ] , then we have

<!-- formula-not-decoded -->

where V max = R max / (1 -γ ) , and similarly V min = R min / (1 -γ ) .

Remark 20 This bound also provides motivation for the specific values of ( τ i ) m i =1 that QTD uses. A similar convergence analysis and fixed-point analysis can be straightforwardly carried out for a version of the QTD algorithm with other values for ( τ i ) m i =1 ; by tracing through the proof of Proposition 19, it can be seen that the bound is proportional to max( τ 1 , max(( τ i +1 -τ i ) / 2 : i = 2 , . . . , m -1) , 1 -τ m ) , which is minimised precisely by the choice of ( τ i ) m i =1 used by QTD.

## 6.1 Instance-Dependent Bounds

The result above implicitly assumes the worst-case projection error is incurred at all states with each application of the Bellman operator. In environments where this is not the case, the fixed point can be shown to be of considerably better quality. We describe an example of an instance-dependent quality bound here.

Proposition 21 Consider an MDP such that for any trajectory, after k time steps all encountered transition distributions and reward distributions are Dirac deltas. If all reward distributions in the MDP are supported on [ R min , R max ] , then for any λ ∈ [0 , 1] X× [ m ] , we have

<!-- formula-not-decoded -->

Remark 22 One particular upshot of this bound for practitioners is that for agents in near-deterministic environments using near-deterministic policies, it may be possible to use m = o ((1 -γ ) -1 ) quantiles and still obtain accurate approximations to the return-distribution function via QTD and/or QDP. It is interesting to contrast this result for quantile-based distributional reinforcement learning against the case when using categorical distribution representations (Bellemare et al., 2017; Rowland et al., 2018; Bellemare et al., 2023). In this latter case, fixed point error continues to be accumulated even when the environment has solely deterministic transitions and rewards, due to the well-documented phenomenon of the approximate distribution 'spreading its mass out' under the Cram´ er projection (Bellemare et al., 2017; Rowland et al., 2018; Bellemare et al., 2023). Our observation here leads to immediate practical advice for practitioners (in environments with mostly deterministic transitions, a quantile representation may be preferred to a categorical representation, leading to less approximation error), and raises a general question that warrants further study: how can we use prior knowledge about the structure of the environment to select a good distribution representation?

We conclude this section by noting that many variants of Proposition 21 are possible; one can for example modify the assumption that rewards are deterministic to an assumption that rewards distributions are supported on a 'small' interval, and still obtain a fixed-point bound that improves over the instance-independent bound of Proposition 19. There are a wide variety of such modifications that could be imagined, and we believe this to be an interesting direction for future research and applications.

## 6.2 Qualitative Analysis of QDP Fixed Points

The analysis in the previous section establishes quantitative upper bounds on the quality of the fixed point learnt by QTD, and guarantees that with enough atoms an arbitrarily accurate approximation of the return-distribution function (as measured by w 1 ) can be learnt. We now take a closer look at the way in which approximation errors may manifest in QTD and QDP.

Example 23 Consider the two-state Markov decision process (with a single action) whose transition probabilities are specified by the left-hand side of Figure 5, such that a deterministic reward of 2 is obtained in state x 1 , and -1 in state x 2 ; further, let us take a discount factor γ = 0 . 9 . The centre panel of this figure shows various estimates of the CDF for the return distribution at state x 1 . The ground truth estimate in black is obtained from Monte Carlo sampling. The CDFs in purple, blue, green, and orange are the points of convergence for QDP with m = 2 , 5 , 10 , 100 , respectively. For m = 100 , a very close fit to the true return distribution is obtained. However, for small m in particular, the distribution is heavily skewed to the right. In the case of m = 2 , half of the probability mass is placed on the greatest possible return in this MDP-namely 20-even though with probability 1 the true return is less than this value. What is the cause of this behaviour from QDP? This question is answered by investigating the dynamic programming update itself in more detail.

In this MDP, the result of the QDP operator applied to the fixed point θ is to update each particle location with a 'backed-up' particle appearing in the distributions T π θ . When such settings arise, tracking which backed-up particles are allocated to which other particles helps us to understand the behaviour of QDP, and the nature of the approximation incurred. We also gain intuition about the situation, since the QDP operator is behaving like a an affine policy evaluation operator on X × [ m ] locally around the fixed point. We can visualise which particles are assigned to one another by a QDP operator application through local quantile back-up diagrams ; the right-hand side of Figure 5 shows the local quantile back-up diagram for particular MDP. We observe that θ ( x 1 , 2) backs up from itself, and hence learns a value that corresponds to observing a self-transition at every state, with a reward of 2; under the discount factor of 0 . 9 , this corresponds to a return of 20. This is the source of the drastic over-estimation of returns in the approximation obtained with m = 2 , and the fact that all other state-quantile pairs implicitly bootstrap from this estimate leads to the overestimation leaking out into all quantiles estimated in this case. As m increases, we observe from the CDF plot that there is always one particle that learns this maximal return of 20, but that this has less effect on the other quantiles; indeed in the orange curve, we obtain a very good approximation (in w 1 ) to the true return distribution despite this particle with a maximal value of 20 remaining present. We can interpret the increase in m as preventing pathological self-loops/small cycles in the quantile backup diagram from 'leaking out' and degrading the quality of other quantile estimates; this provides a complementary perspective on the approximation artefacts that occur in QDP/QTD fixed points to the quantitative upper bounds in the previous section.

We expect the local quantile back-up diagram introduced in Example 23 to be a useful tool for developing intuition, as well as further analysis, of QDP and QTD. As described in the example itself, being able to define the local back-up diagram depends on the structure

Figure 5: Left: An example MDP. Centre: The fixed point return distribution estimates for state x 1 obtained by QDP for m = 2 , 5 , 20 , 100 (solid purple, dotted blue, dashed green, and dash-dotted orange, respectively) compared to ground truth in solid black. Right: The corresponding local quantile backup diagram at the fixed point for m = 2, illustrating potential approximation artefacts in QDP fixed points.

<!-- image -->

of the MDP being such that the QDP operator obtains each new coordinate value from a single backed-up particle location. It is an interesting question as to how the definition of such local back-up diagrams could be generalised to apply in situations where this does not hold, such as with absolutely continuous reward distributions.

## 7 Related Work

Stochastic approximation theory with differential inclusions. The ODE method was introduced by Ljung (1977) as a means of analysing stochastic approximation algorithms, and was subsequently extended and refined by Kushner and Clark (1978); standard references on the subject include Kushner and Yin (2003); Borkar (2008); Benveniste et al. (2012); see also Meyn (2022) for an overview in the context of reinforcement learning. The framework we follow in this paper is set out by Bena¨ ım (1999), and was extended by Bena¨ ım et al. (2005) to allow for differential inclusions. Perkins and Leslie (2013) later extended this analysis further to allow for asynchronous algorithms, building on the approach introduced by Borkar (1998), and extended, with particular application to reinforcement learning, by Borkar and Meyn (2000).

Differential inclusion theory. Differential inclusions have found application across a wide variety of fields, including control theory (Wazewski, 1961), economics (Aubin, 1991) differential game theory (Krasovskii and Subbotin, 1988), and mechanics (Monteiro Marques, 2013). The approach to modelling differential equations with discontinuous right-hand sides via differential inclusions was introduced by Filippov (1960). Standard references on the theory of differential inclusions include Aubin and Cellina (1984); Clarke et al. (1998); Smirnov (2002); see also Bernardo et al. (2008) on the related field of piecewise-smooth dynamical systems. Joseph and Bhatnagar (2019) also use tools combining stochastic approximation and differential inclusions from Bena¨ ım et al. (2005) to analyse (sub-)gradient descent as a means of estimating quantiles of fixed distributions. Within reinforcement learning and related fields more specifically, differential inclusions have played a key role in

the analysis of game-theoretic algorithms based on fictitious play (Brown, 1951; Robinson, 1951); see Bena¨ ım et al. (2006); Leslie and Collins (2006); Bena¨ ım and Faure (2013) for examples. More recently, Gopalan and Thoppe (2023) used differential inclusions to analyse TD algorithms for control with linear function approximation.

Quantile regression. Quantile regression as a methodology for statistical inference was introduced by Koenker and Bassett (1978). Koenker (2005) and Koenker et al. (2017) provide detailed surveys of the field. Quantile temporal-difference learning may be viewed as fusing quantile regression with the bootstrapping approach ( learning a guess from a guess , as Sutton and Barto (2018) express it) that is core to much of the reinforcement learning methodology.

Quantiles in reinforcement learning. The approach to distributional reinforcement learning based on quantiles was introduced by Dabney et al. (2018b). A variety of modifications and extensions were then considered in the deep reinforcement learning setting (Dabney et al., 2018a; Yang et al., 2019; Zhou et al., 2020; Luo et al., 2021), as well as further developments on the theoretical side (Lh´ eritier and Bondoux, 2022). A summary of the approach is presented by Bellemare et al. (2023). Gilbert and Weng (2016) study the problem of optimising quantile criteria in end-state MDPs. Li et al. (2022) consider the risk-sensitive control problem of optimising particular quantiles of the return distribution, and derive a dynamic programming algorithm that maintains a value function over state and the 'target quantile-to-go'.

## 8 Conclusion

We have provided the first convergence analysis for QTD, a popular and effective distributional reinforcement learning algorithm. In contrast to the analysis of many classical temporal-difference learning algorithms, this has required the use of tools from the field of differential inclusions and branches of stochastic approximation theory that deal with the associated dynamical systems. Due to the structure of the QTD algorithm, such as its bounded-magnitude updates, these convergence guarantees hold under weaker conditions than are generally used in the analysis of TD algorithms. These results establish the soundness of QTD, representing an important step towards understanding its efficacy and practical successes, and we expect the theoretical tools used here to be useful in further analyses of (distributional) reinforcement learning algorithms.

There are several natural directions for further work building on this analysis. One such direction is to establish finite-sample bounds for the convergence of QTD predictions to the set of QDP fixed points. This is a central theoretical question for developing our understanding of QTD, and may also shed further light on the recently observed empirical phenomenon in which tabular QTD can outperform TD in stochastic environments as a means of value estimation (Rowland et al., 2023). Related to this point, the Lyapunov analysis conducted in this paper provides further intuition for why QTD works in general, and we expect this to inform the design of further variants of QTD, for example incorporating multi-step bootstrapping (Watkins, 1989), or Ruppert-Polyak averaging (Ruppert, 1988; Polyak and Juditsky, 1992). Another important direction is to analyse more complex

variants of the QTD algorithm, incorporating more aspects of the large-scale systems in which it has found application. Examples include incorporating function approximation, or control variants of the algorithm based on Q-learning. We believe further research into the theory, practice and applications of QTD, at a variety of scales, are important directions for foundational reinforcement learning research.

## Acknowledgments

We thank the anonymous reviewers for helpful suggestions and feedback. We also thank Tor Lattimore for detailed comments on an earlier draft, and David Abel, Bernardo Avila Pires, Diana Borsa, Yash Chandak, Daniel Guo, Clare Lyle, and Shantanu Thakoor for helpful discussions. Marc G. Bellemare was supported by Canada CIFAR AI Chair funding. The simulations in this paper were generated using the Python 3 language, and made use of the NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), and Matplotlib (Hunter, 2007) libraries.

## Appendix A. Proofs

In this section, we provide proofs for results which are not proven in the main text.

## A.1 Proof of Proposition 6

Let η, η ′ ∈ P ( R ). We have

<!-- formula-not-decoded -->

Clearly

Additionally, we have

<!-- formula-not-decoded -->

Putting this together, we obtain

<!-- formula-not-decoded -->

as required.

<!-- formula-not-decoded -->

## A.2 Proof of Theorem 14

Theorem 14 is essentially a special case of the general results presented in Bena¨ ım et al. (2005), in the form needed for the proof of convergence of QTD. To explain how to obtain Theorem 14 from the results of Bena¨ ım et al. (2005), first, we associate a continuoustime path ( ¯ θ ( t )) t ≥ 0 with the iterates ( θ k ) ∞ k =0 by linear interpolation, in particular defining ¯ θ ( ∑ s k =0 α k ) = θ s , and linearly interpolating in between. The continuous-time path ( ¯ θ ( t )) t ≥ 0 satisfies the definition of a perturbed solution of the Marchaud differential inclusion with probability 1, as defined by Definition II of Bena¨ ım et al. (2005), since: (i) ¯ θ is piecewise linear, hence absolutely continuous; (ii) the difference ∥ θ k +1 -θ k ∥ ∞ is O ( α k ), due to the growth condition on H and since ¯ θ is bounded by assumption; and (iii) the lim-sup condition holds with probability 1 thanks to the boundedness of the martingale difference sequence ( w k ) ∞ k =0 and Proposition 1.4 of Bena¨ ım et al. (2005); see also Theorem 5.3.3 of Kushner and Yin (2003).

Next, since we assume ¯ θ is bounded, Theorem 4.2 of Bena¨ ım et al. (2005) applies so that we deduce that it is an asymptotic pseudotrajectory of the differential inclusion (w.p.1). We then have that ( ¯ θ ( t )) t ≥ 0 is a bounded asymptotic pseudotrajectory (w.p.1), so Theorem 4.3 of Bena¨ ım et al. (2005) applies, and we deduce that the set of limit points of ( ¯ θ ( t )) t ≥ 0 is internally chain transitive (w.p.1). But now by Proposition 3.27 of Bena¨ ım et al. (2005) applied to the Lyapunov function L and the set Λ, all internally chain transitive sets are contained within Λ. Since ( ¯ θ ( t )) t ≥ 0 is bounded, we deduce that it converges to Λ (w.p.1). It therefore follows that the discrete sequence ( θ k ) ∞ k =0 converges to Λ with probability 1, as required.

## A.3 Proof of Proposition 15

Roughly, the intuition of the proof is that the structure of the QTD differential inclusion means that when ∥ θ k ∥ ∞ is sufficiently large, the coordinates of θ k furthest from the origin are moved back towards the origin by the differential inclusion. We then argue that the martingale noise cannot cause divergence, which completes the argument.

Differential inclusion update direction. To begin with the analysis of the differential inclusion, fix δ &gt; 0 such that 1 -δ &gt; γ , and let M &gt; 0 be such that for all ( x, a ) ∈ X × A , we have

<!-- formula-not-decoded -->

We then introduce the events

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, roughly speaking, hold when θ k has at least one large coordinate (in absolute value), and θ k ( x, i ) is a positive (respectively, negative) coordinate close to the maximum value.

When I + k ( x, i ) holds, we have

<!-- formula-not-decoded -->

and hence the differential inclusion moves θ k ( x, i ) towards the origin. Inequality (a) follows since on I + k ( x, i ), we have

<!-- formula-not-decoded -->

Analogously, we conclude that on I -k ( x, i ), we have

<!-- formula-not-decoded -->

and so the differential inclusion moves θ k ( x, i ) towards the origin in this case too.

Chaining updates and reasoning about noise. To describe the relationship between successive iterates in the sequence ( θ k ) k ≥ 0 , we introduce the notation θ k +1 = θ k + α k g k + α k w k , where w j is martingale difference noise, and hence g j is an expected update direction, from the right-hand side of the QTD differential inclusion. By boundedness of the update noise and the step size assumptions, we have from Proposition 1.4 of Bena¨ ım et al. (2005) (see also Theorem 5.3.3 of Kushner and Yin (2003)) that

<!-- formula-not-decoded -->

almost surely. In particular, letting ε ∈ (0 , 1), there almost-surely exists K (which depends on the realisation of the martingale noise) such that

<!-- formula-not-decoded -->

for all k ≥ K , and further such that α k &lt; 1 for all k ≥ K .

Let us additionally take ¯ M ≥ M such that δ ¯ M ≥ 4(8 m + 1). Suppose that for some k ≥ K , ∥ θ k ∥ ∞ ≥ ¯ M +(8 m +1). Let l be minimal such that ∑ k + l j = k α j &gt; 8 m . Then we have ∥ θ k + j -θ k ∥ ∞ ≤ 8 m +1 for all 0 ≤ j ≤ l , and so ∥ θ k + j ∥ ∞ ≥ ¯ M for all 0 ≤ j ≤ l . Further, if

θ k ( x, i ) satisfies θ k ( x, i ) &gt; ∥ θ k ∥ ∞ (1 -δ ) + 2(8 m +1), then we have

<!-- formula-not-decoded -->

so I + k + j ( x, i ) holds for all 0 ≤ j ≤ l , and hence

<!-- formula-not-decoded -->

Similarly, if θ k ( x, i ) &lt; -∥ θ k ∥ ∞ (1 -δ ) -2(8 m +1), I -k + j ( x, i ) holds for all 0 ≤ j ≤ l , and we reach the same conclusion as in Equation (23). Finally, if | θ k ( x, i ) | ≤ ∥ θ k ∥ ∞ (1 -δ ) + 2(8 m + 1), then since δ ∥ θ k ∥ ∞ &gt; δ ¯ M , we have | θ k ( x, i ) | ≤ ∥ θ k ∥ ∞ -2(8 m + 1), and hence | θ k + l +1 ( x, i ) | ≤ ∥ θ k ∥ ∞ -(8 m +1). Putting these components together, we have

<!-- formula-not-decoded -->

as required to establish boundedness.

## A.4 Proof of Proposition 18

We first state and prove a useful lemma that allows us to compare QDP fixed points for different values of λ . Throughout this section, we will adopt the shorthand θ λ for ˆ θ π λ .

Lemma 24 Let λ, λ ′ ∈ [0 , 1] X× [ m ] . Then we have

<!-- formula-not-decoded -->

where C is a constant depending only on the reward distributions of the MDP and γ .

Proof By the triangle inequality, we have

<!-- formula-not-decoded -->

Now we aim to bound ∥ θ λ ∥ ∞ , and hence the term on the right-hand side above. Note that in general for a mixture distribution ν = ∑ n i =1 p i ν i , we have F -1 ν ( τ ) ≥ min { F -1 ν i ( τ ) : i =

1 , . . . , n } , since

<!-- formula-not-decoded -->

Thus, it follows that the 1 / 2 m quantile of T π θ λ ( x ) is at least as great as

<!-- formula-not-decoded -->

By analogous reasoning, we obtain that ¯ F -1 ( T π θ λ )( x ) ( 2 m -1 / 2 m ) is no greater than

<!-- formula-not-decoded -->

From these facts, it follows that

<!-- formula-not-decoded -->

and hence

<!-- formula-not-decoded -->

as required for the statement of the result.

We now turn to the proof of Proposition 18. First, we observe that the infimum over λ in Equation (20) is attained, since Lemma 24 establishes that λ ↦→ θ λ is continuous (in fact Lipschitz), and [0 , 1] X× [ m ] is compact. We therefore have that L is continuous, non-negative, and takes on the value 0 only on the set of fixed points { θ λ : λ ∈ [0 , 1] X× [ m ] } .

For the decreasing property, let ( ϑ t ) t ≥ 0 be a solution to the differential inclusion in Equation (17), and as in Definition 11, let g : [0 , ∞ ) → R X× [ m ] satisfy

<!-- formula-not-decoded -->

with g t ( x, i ) ∈ H π x,i ( ϑ t ) for all ( x, i ), and for almost all t ≥ 0, where we have introduced the notation

<!-- formula-not-decoded -->

As in the proof of Proposition 17, we will show that L ( ϑ t ) is locally decreasing outside of the fixed point set, which is enough for the global decreasing property. Further, by continuity of L ( ϑ t ), it is enough to show this property for almost all t ≥ 0. We will therefore consider a value of t ≥ 0 at which the above inclusion for g t holds.

Let λ attain the minimum in the definition of L ( ϑ t ). Write θ λ for the corresponding fixed point for conciseness, and let ( x, i ) be a λ -argmax index with respect to ϑ t ; a state-particle pair achieving the maximum in the definition of the norm ∥ ϑ t -θ λ ∥ ∞ . First, we consider the cases where H π x,i ( ϑ t ) is not a singleton. Now, if 0 ∈ H π x,i ( ϑ t ), then we have (Π λ T π ϑ t )( x, i ) = ϑ t ( x, i ), and with the same logic as above, we have ϑ t = θ λ , and hence ϑ t is in the fixed point set, and L ( ϑ t ) is constant. If 0 ̸∈ H π x,i ( ϑ t ), then as in the proof of Proposition 17, it can be shown that any element of H π x,i ( ϑ t ) has the same sign as

<!-- formula-not-decoded -->

In the case of Proposition 17, continuity of the derivative then allowed us to deduce that | ϑ t ( x, i ) -θ λ ( x, i ) | is locally decreasing. Here, we require a related concept of continuity for the set-valued map θ ↦→ H π x,i ( θ ), namely that it is upper semicontinuous (see, for example, Smirnov, 2002); for a given θ ∈ R X× [ m ] and any given ε &gt; 0, there exists δ &gt; 0 such that if ∥ θ ′ -θ ∥ ∞ &lt; δ , then H π x,i ( θ ′ ) ⊆ { h + v : h ∈ H π x,i ( θ ) , | v | &lt; ε } . From this, it follows that any element of H π x,i ( ϑ t + s ), for sufficiently small positive s , has the same sign as the expression in Equation (25), and so from Equation (24), we have that | ϑ t ( x, i ) -θ λ ( x, i ) | is locally decreasing, as required.

Now, when H π x,i ( ϑ t ) is a singleton, if it is non-zero, then by the same argument as in the proof of Proposition 17, the corresponding element has the same sign as the expression in Equation (25), and so as above, we conclude that | ϑ t ( x, i ) -θ λ ( x, i ) | is locally decreasing.

Finally, the case where there exists an argmax index ( x, i ) with H π x,i ( ϑ t ) = { 0 } requires more care, and we will need to reason about the effects of perturbing λ to show that the Lyapunov function is decreasing. For some intuition as to what the problem is, if H π x,i ( ϑ t + s ) = { 0 } for small positive s , then the coordinate ϑ t + s ( x, i ) is static, as it lies on the flat region of the CDF F ( T π ϑ t + s )( x ) at level τ i , and so the distance | ϑ t + s ( x, i ) -θ λ ( x, i ) | is not decreasing. We explain how to deal with this case below.

## A.4.1 Perturbative Argument

We introduce the notation J 0 ⊆ X × [ m ] for the set of λ -argmax indices with respect to ϑ t . Assuming that ∥ ϑ t -θ λ ∥ ∞ is not locally decreasing, it must be locally constant (it cannot increase, by the arguments above). Now consider s &gt; 0 sufficiently small so that (i) no coordinates not in J 0 can be a λ -argmax index with respect to ϑ t + s , so that J , the set of λ -argmax indices with respect to ϑ t + s , satisfies J ⊆ J 0 , (ii) all indices ( x, i ) ∈ J satisfy H π x,i ( ϑ t + u ) = { 0 } for all u ∈ [0 , 2 s ].

We will now demonstrate the existence of a parameter λ ′ ∈ [0 , 1] X× [ m ] such that ∥ ϑ t + s -θ λ ′ ∥ ∞ &lt; ∥ ϑ t -θ λ ∥ ∞ , which establishes the locally decreasing property of the Lyapnuov function, as required. To do so, we introduce a modification of the fixed point map λ ↦→ θ λ . Letting µ ∈ R J , and defining λ [ µ ] ∈ R X× [ m ] to be the replacement of the J coordinates of λ with the corresponding coordinates of µ , we consider the map

<!-- formula-not-decoded -->

where P J : R X× [ m ] → R J extracts the J coordinates. At an intuitive level, this map allows us to study the effect of perturbing the J coordinates of λ on the corresponding coordinates of the fixed point.

## A.4.2 Case 1: λ J is in the Interior of [0 , 1] J

̸

We now first consider the case where λ J , the J coordinates of λ , lies in the interior of [0 , 1] J , that is (0 , 1) J . By Lemma 24, h λ is continuous, since it is the composition of the continuous maps µ ↦→ λ [ µ ], λ ↦→ θ λ , and θ ↦→ P J θ . It is also injective in a neighbourhood of λ J . This can be seen by noting first that the fixed points θ λ [ µ ] are distinct for distinct values of µ sufficiently close to λ J ; if µ = µ ′ are each sufficiently close to λ J , then we have

̸

<!-- formula-not-decoded -->

where the inequality follows from the fact that since θ λ [ µ ] is continuous in µ , for µ sufficiently close to J there is a flat region of F ( T π θ λ [ µ ] )( x ) at level τ i , for any ( x, i ) ∈ J . To complete the injectivity argument, we cannot have P J θ λ [ µ ′ ] = P J θ λ [ µ ] if θ λ [ µ ′ ] = θ λ [ µ ] , as the contraction maps Π λ [ µ ] T π and Π λ [ µ ] T π are equal on coordinates not in J , and these two maps would therefore have the same fixed point, a contradiction.

̸

We may now appeal to the invariance of domain theorem (Brouwer, 1912) to deduce that since h λ is a continuous injective map between an open subset of [0 , 1] J containing λ J (here we are using the assumption that λ J lies in the interior of [0 , 1] J ) and the Euclidean space R J of equal dimension, it is an open map on this domain; that is, it maps open sets to open sets. Hence, we can perturb θ λ in the J coordinates in any direction we want by locally modifying the J coordinates of λ . In particular, we can move all J coordinates of θ λ closer to those of ( ϑ t + s ( x, i ) : ( x, i ) ∈ J ). Let λ ′ ∈ (0 , 1) X× [ m ] be such a modification of λ , taken to be close enough to λ so that all coordinates outside J have sufficiently small perturbations so that they cannot be λ ′ -argmax indices with respect to ϑ t + s . We then have that ∥ ϑ t + s -θ λ ′ ∥ ∞ &lt; ∥ ϑ t -θ λ ∥ ∞ , as required.

## A.4.3 Case 2: λ J is on the Boundary of [0 , 1] J

In the more general case when λ J may lie on the boundary of [0 , 1] J , we can apply the same argument to an extension of the function h λ , by increasing its domain from [0 , 1] J to an open neighbourhood of this domain in R J . We define this extension simply by extending the definition of Π λ in Equation (12) to allow coordinates of λ to lie outside the range [0 , 1]. We lose the non-expansiveness of Π λ (in L ∞ ) under this extension, but if λ min , λ max are the minimum and maximum coordinates of λ , respectively, it is easy verified (by modifying the proof of Proposition 6) that Π λ is max(1 -λ min , λ max )-Lipschitz, and so if we extend the function to a domain where λ max , 1 -λ min ≤ γ -1 / 2 , the composition Π λ T π is a γ 1 / 2 -contraction in L ∞ , and hence has a unique fixed point θ λ .

By the same arguments as above, the extended map h λ is continuous and injective in a neighbourhood of λ J on this extended domain, and hence we may again apply the invariance of domain theorem to obtain that h λ \ J is locally surjective around λ J . However, since λ J lies on the boundary of the original domain, we must additionally check that we can perturb

λ J to obtain µ in such a way that we obtain the desired perturbation of θ λ , without the parameters µ leaving the set [0 , 1] J . To do this, we first rule out λ J lying on certain parts of the boundary.

Lemma 25 If ( x, i ) ∈ J and ϑ t + s ( x, i ) &lt; θ λ ( x, i ) , then λ ( x, i ) &gt; 0 . Similarly, if ϑ t + s ( x, i ) &gt; θ λ ( x, i ) , then λ ( x, i ) &lt; 1 .

Proof We prove the claim when ϑ t + s ( x, i ) &lt; θ λ ( x, i ); the other case follows analogously. If λ ( x, i ) = 0, then since ϑ t + s ( x, i ) corresponds to the flat region at level τ i of the CDF F ( T π ϑ t + s )( x ) , we must have (Π λ T π ϑ t + s )( x, i ) ≤ ϑ t + s ( x, i ) since λ ( x, i ) = 0, and so the chosen quantile at level τ i by the projection Π λ is the left-most point of this flat region. We therefore have

<!-- formula-not-decoded -->

contradicting contractivity of Π λ T π around θ λ

.

We write v = sign(( ϑ t + s ) J -θ λ J ) ∈ R J , where the sign mapping is applied elementwise, and introduce the notation N( v ) = { α ⊙ v : α ∈ R n &gt; 0 } for the (open) orthant containing the vector v . We are therefore seeking a perturbation µ of λ J such that θ λ [ µ ] J lies in a direction in N( v ) from θ λ J , and further such that the perturbation to θ λ is sufficiently small that no index that was not an argmax in ∥ ϑ t + s -θ λ ∥ ∞ can become one in ∥ ϑ t + s -θ λ [ µ ] ∥ ∞ ; under these conditions, we have ∥ ϑ t + s -θ λ [ µ ] ∥ ∞ &lt; ∥ ϑ t + s -θ λ ∥ ∞ , as required. Lemma 25 then guarantees that a (sufficiently small) perturbation of λ J in any direction in N( v ) remains within [0 , 1] J , so it is sufficient to show that a perturbation in such a direction achieves the desired perturbation of θ λ .

Differentiability. Now, if the extended map λ ↦→ θ λ is differentiable at λ , then differentiating through the fixed-point equation θ λ = G ( λ, θ λ ) (where we write G ( λ, θ ) = Π λ T π θ for conciseness) yields

<!-- formula-not-decoded -->

differentiability of G in θ results from differentiability of the map λ ↦→ G ( λ, θ λ ), and continuous differentiability of G in λ . Since θ ↦→ G ( λ, θ ) is contractive in L ∞ with factor γ 1 / 2 (on the extended domain), and by coordinatewise monotonicity of θ ↦→ G ( λ, θ ), it follows that ∂ θ G ( λ, θ λ ) is non-negative and strictly substochastic, with row L 1 norms bounded by γ 1 / 2 , the contraction factor for the extended set of contraction mappings. We remark as a point of independent interest that this is a kind of Bellman equation for ∇ λ θ λ , with ∂ θ G ( λ, θ λ ) taking the role of the transition matrix, and ∂ λ G ( λ, θ λ ) taking the role of a collection of cumulants; in fact, the structure of ∂ θ G ( λ, θ λ ) coincides with the local quantile back-up diagrams described in Example 23. We therefore have

<!-- formula-not-decoded -->

By extracting the principal submatrix on the J coordinates, we obtain a derivative for h λ ( λ J ). The following lemma is useful in reasoning about the structure of this principal submatrix.

Lemma 26 Let Q 1 ∈ R n × n be strictly substochastic, and let K ⊆ [ n ] . Then the principal submatrix on the K coordinates of ( I -Q 1 ) -1 can be expressed as ( I -Q 2 ) -1 , with Q 2 ∈ R K × K strictly substochastic.

Proof We interpret Q 1 as the transition matrix of a Markov chain ( Z t ) t ≥ 0 that includes a non-zero probability of termination at each state. Each row of the matrix ( I -Q 1 ) -1 is then the pre-termination visitation measure associated with a particular initial state in the Markov chain. Now let Q 2 be the strictly substochastic matrix defined by

<!-- formula-not-decoded -->

K,

By construction, the pre-termination visitation distribution ( I -Q 2 ) -1 is identical to the principal submatrix of ( I -Q 1 ) -1 on the K coordinates, as required.

From Lemma 26, we therefore obtain that ∇ h λ ( λ J ) has the form

<!-- formula-not-decoded -->

with D ∈ R J × J diagonal, with positive elements on the diagonal (from monotonicity of λ ↦→ G ( λ, θ )), with Q ∈ R J × J strictly substochastic. The derivative is therefore invertible, and we obtain the derivative of the inverse of the form

<!-- formula-not-decoded -->

From strict substochasticity of Q , and since v ∈ {± 1 } J , it follows that for the desired perturbation direction v , we have

<!-- formula-not-decoded -->

and so ∇ h -1 λ ( θ λ J ) v ∈ N( v ), where the equality of signs applies elementwise. Therefore, a perturbation of λ J in a direction in N( v ) is achieved by a sufficiently small perturbation of λ J in a direction in N( v ), as required.

Non-differentiability. If λ ↦→ θ λ is not differentiable at λ , we instead use techniques from non-smooth analysis to complete the argument. First, since λ ↦→ θ λ is Lipschitz (by Lemma 24), it is differentiable almost everywhere by Rademacher's theorem (Rademacher, 1919). By adapting the argument made by Clarke (1976, Lemma 3), by Fubini's theorem, for almost all λ \ J ∈ R X× [ m ] \ J , the map λ ↦→ θ λ is differentiable at ( λ \ J , µ ) for almost all µ with ( λ \ J , µ ) in the extended domain. The map ( λ \ J , µ ) ↦→ ( λ \ J , h λ \ J ( µ )) is Lipschitz and locally injective around λ , and hence maps sufficiently small open neighbourhoods of λ to open neighbourhoods of ( λ \ J , h λ \ J ( λ J )). Further, since each h λ \ J is Lipschitz, and so

absolutely continuous, the inverse map h -1 λ \ J is almost-everywhere differentiable within such a neighbourhood. Following the analysis of the differentiable case, we therefore deduce

<!-- formula-not-decoded -->

for almost all λ \ J in a ball B around λ \ J , and (for each such λ \ J ) for almost all θ in the L ∞ ball B ′ with centre h λ \ J ( λ J ) and radius ρ , for some radius ρ &gt; 0. We further take B and B ′ to be of small enough radii so that this directional derivative is bounded on this set, so that h -1 λ is locally Lipschitz on B ′ for each λ \ J ∈ B (and hence absolutely continuous), and so that for any θ ∈ B ′ , we have sign( θ -( ϑ t + s ) J ) = sign( θ λ J -( ϑ t + s ) J ), and so that for no µ in the preimage of B ′ under h λ \ J can have that ∥ θ ( λ \ J ,µ ) -ϑ t + s ∥ ∞ has new argmax coordinates outside of J .

Let us consider ˜ λ ∈ B at which the almost-everywhere differentiability condition holds. By applying the same argument with Fubini's theorem, for almost all ¯ θ in B ( h ˜ λ ( λ J ) , ρ/ 4), the inverse h -1 ˜ λ is differentiable almost everywhere on { ¯ θ + uv : u ∈ [0 , ρ/ 2] } .

Now, defining µ τ = h -1 ˜ λ ( ¯ θ + τv ) for τ ∈ [0 , ρ/ 2], we have

<!-- formula-not-decoded -->

for almost all τ , and by absolute continuity of h -1 ˜ λ , it follows that

<!-- formula-not-decoded -->

Hence, µ ε -µ 0 ∈ N( v ), and by construction h ˜ λ ( µ ρ/ 2 ) = ¯ θ + ρv/ 2. By continuity of λ ↦→ θ λ and its inverse, and since ˜ λ and ¯ θ can be chosen above to be arbitrarily close to λ \ J and h λ \ J ( λ J ) respectively, we may consider a sequence of these parameters converging to λ \ J and h λ \ J ( λ J ), such that the values of µ ρ/ 2 as constructed above also converge (by compactness), and thus conclude the existence of ¯ µ ρ/ 2 such that ¯ µ ρ/ 2 -λ J ∈ N( v ), and h λ (¯ µ ρ/ 2 ) = h λ ( λ J ) + ρv/ 2, as required.

## A.5 Proof of Proposition 19

We begin with the observation that for any return-distribution function η ∈ P ([ V min , V max ]), for the projection Π λ onto F Q ,m (for any λ ∈ [0 , 1] X× [ m ] ), we have

<!-- formula-not-decoded -->

Using this observation, we have

<!-- formula-not-decoded -->

Here, (a) follows from the triangle inequality, (b) follows as ˆ η π λ , η π are fixed points of Π λ T π , T π , respectively, and (c) follows from the application of the inequality at the beginning of the proof and contractivity of T π . Rearranging then gives the desired result.

## A.6 Proof of Proposition 21

From the assumptions of the proposition, we have ˆ η π λ = (Π λ T π ) k η π . Then observe that following the argument for the proof of Proposition 19, we have, for any l ∈ { 1 , . . . , k } ,

<!-- formula-not-decoded -->

Chaining these inequalities yields the required statement.

## Appendix B. Implementations of Quantile Dynamic Programming

Here, we describe two concrete implementations of QDP, which may be of independent interest to the reader. Algorithm 3 (Bellemare et al., 2023) describes an implementation when the reward distributions are available as input to the algorithm as a list of outcomes and probabilities.

```
Require: Quantile estimates (( θ ( x, i ) m i =1 : x ∈ X ), Transition and reward probabilities ( P π ( x ′ , r | x ) : x, x ′ ∈ X ), Interpolation parameters λ ∈ [0 , 1] X× [ m ] . 1: for x ∈ X do 2: Set Targets as empty list { List of outcome/probability pairs } 3: for x ′ ∈ X do 4: for r ∈ R do 5: for j = 1 , . . . , m do 6: Append ( r + γθ ( x ′ , j ) , P π ( x ′ , r | x ) /m ) to Targets 7: end for 8: end for 9: end for 10: Sort Targets ascending according to outcomes. 11: for i = 1 , . . . , m do 12: Find minimal outcome q ′ such that cumulative probability is ≥ 2 i -1 / 2 m . 13: Set θ ′ ( x, i ) ← q ′ . 14: end for 15: end for 16: return (( θ ′ ( x, i ) m i =1 : x ∈ X )
```

Algorithm 3 Quantile dynamic programming (finitely-supported rewards)

Algorithm 4 makes use of a root-finding subroutine (such as scipy.optimize.root scalar ), and can be used when the CDFs of the reward distributions are available as input, and can be queried at individual points. A common use case for this implementation is the case of Gaussian rewards. Note that the root-finding subroutine is called on a monotonic scalar

function, and therefore strong guarantees can be given on the approximate solution returned when the reward CDFs of the MDP are continuous. Nevertheless, note that Algorithm 4 does not exactly implement the operator Π λ T π due to this root-finding approximation error. For simplicity, we present the algorithm in the case where the reward and next state in a transition are conditionally independent given the current state, though the algorithm can be straightforwardly extended to the general case, by working with CDFs of reward distributions conditioned on the next state.

## Algorithm 4 Quantile dynamic programming (reward CDFs)

Require: Quantile estimates (( θ ( x, i ) m i =1 : x ∈ X ),

Transition probabilities ( P π ( x ′ | x ) : x, x ′ ∈ X ),

Reward CDFs ( F R π ( x ) : x ∈ X ).

1: for x ∈ X do

2: Construct function

- 5: end for
- 6: end for
- 7: return (( θ ′ ( x, i ) m i =1 : x ∈ X )

## Appendix C. Convergence of Asynchronous QTD Updates

Here, we describe the key considerations in extending our analysis to a proof of convergence for asynchronous versions of QTD; our discussion follows the approach of Perkins and Leslie (2013).

Step size restrictions. Typically, more restrictive assumptions on step sizes, beyond the Robbins-Monro conditions, are required for asynchronous convergence guarantees. See, for example, Assumption A2 of Perkins and Leslie (2013); note that the typical Robbins-Monro step size schedule of α k ∝ 1 /k ρ for ρ ∈ (1 / 2 , 1] satisfies these requirements.

Conditions on the sequence of states ( X k ) k ≥ 0 to be updated. Additionally, different states are required to be updated 'comparably often'; assuming that ( X k ) k ≥ 0 forms an aperiodic irreducible time-homogeneous Markov chain is sufficient, and this conditions holds when either (i) π generates such a Markov chain over the state space of the MDP of interest, or (ii) when the states to be updated are sampled i.i.d. from a fixed distribution supported on the entirety of the state space, amongst other settings. See Assumption A4 of Perkins and Leslie (2013) for further details.

<!-- formula-not-decoded -->

- 3: for i = 1 , . . . , m do

4: Use a scalar root-finding subroutine to find θ ′ ( x, i ) approximately satisfying

<!-- formula-not-decoded -->

Modified differential inclusion. The QTD differential inclusion in Equation (17) must be broadened to account for the possibility of different states being updated with different frequencies, leading to a differential inclusion of the form

<!-- formula-not-decoded -->

where δ represents a minimum relative update frequency for the state x , derived from the conditions on ( X k ) k ≥ 0 described above. Because of the structure of the Lyapunov function for the QTD DI in Equation (20), it is readily verified that this remains a valid Lyapunov function for this broader differential inclusion, for the same invariant set of QDP fixed points.

## References

Jean-Pierre Aubin. Viability theory . Springer Birkhauser, 1991.

- Jean-Pierre Aubin and Arrigo Cellina. Differential inclusions: Set-valued maps and viability theory . Springer Science &amp; Business Media, 1984.
- Gabriel Barth-Maron, Matthew W. Hoffman, David Budden, Will Dabney, Dan Horgan, Dhruva TB, Alistair Muldal, Nicolas Heess, and Timothy Lillicrap. Distributed distributional deterministic policy gradients. In Proceedings of the International Conference on Learning Representations , 2018.
- Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The Arcade Learning Environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47:253-279, June 2013.
- Marc G. Bellemare, Will Dabney, and R´ emi Munos. A distributional perspective on reinforcement learning. In Proceedings of the International Conference on Machine Learning , 2017.
- Marc G. Bellemare, Salvatore Candido, Pablo Samuel Castro, Jun Gong, Marlos C. Machado, Subhodeep Moitra, Sameera S. Ponda, and Ziyu Wang. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature , 588(7836):77-82, 2020.
- Marc G. Bellemare, Will Dabney, and Mark Rowland. Distributional reinforcement learning . MIT Press, 2023.
- Michel Bena¨ ım. Dynamics of stochastic approximation algorithms. In Seminaire de Probabilites XXXIII , pages 1-68. Springer, 1999.
- Michel Bena¨ ım and Mathieu Faure. Consistency of vanishingly smooth fictitious play. Mathematics of Operations Research , 38(3):437-450, 2013.
- Michel Bena¨ ım, Josef Hofbauer, and Sylvain Sorin. Stochastic approximations and differential inclusions. SIAM Journal on Control and Optimization , 44(1):328-348, 2005.

- Michel Bena¨ ım, Josef Hofbauer, and Sylvain Sorin. Stochastic approximations and differential inclusions, Part II: Applications. Mathematics of Operations Research , 31(4):673-695, 2006.
- Albert Benveniste, Michel M´ etivier, and Pierre Priouret. Adaptive algorithms and stochastic approximations . Springer Science &amp; Business Media, 2012.
- Mario Bernardo, Chris Budd, Alan Richard Champneys, and Piotr Kowalczyk. Piecewisesmooth dynamical systems: Theory and applications . Springer Science &amp; Business Media, 2008.
- Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.
- Cristian Bodnar, Adrian Li, Karol Hausman, Peter Pastor, and Mrinal Kalakrishnan. Quantile QT-Opt for risk-aware vision-based robotic grasping. In Robotics: Science and Systems , 2020.
- Vivek S. Borkar. Asynchronous stochastic approximations. SIAM Journal on Control and Optimization , 36(3):840-851, 1998.
- Vivek S. Borkar. Stochastic approximation: A dynamical systems viewpoint . Springer, 2008.
- Vivek S. Borkar and Sean P. Meyn. The ODE method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization , 38 (2):447-469, 2000.
- Luitzen E. J. Brouwer. Beweis der Invarianz des n -dimensionalen Gebiets. Mathematische Annalen , 71(3):305-313, 1912.
- George W. Brown. Iterative solution of games by fictitious play. Act. Anal. Prod Allocation , 13(1):374, 1951.
- Francis H. Clarke. On the inverse function theorem. Pacific Journal of Mathematics , 64 (1):97-102, 1976.
- Francis H. Clarke, Yuri S. Ledyaev, Ronald J. Stern, and Peter R. Wolenski. Nonsmooth analysis and control theory . Springer Science &amp; Business Media, 1998.
- Will Dabney, Georg Ostrovski, David Silver, and R´ emi Munos. Implicit quantile networks for distributional reinforcement learning. In Proceedings of the International Conference on Machine Learning , 2018a.
- Will Dabney, Mark Rowland, Marc G. Bellemare, and R´ emi Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence , 2018b.
- Peter Dayan. The convergence of TD( λ ) for general λ . Machine Learning , 8(3-4):341-362, 1992.

- Peter Dayan and Terrence J. Sejnowski. TD( λ ) converges with probability 1. Machine Learning , 14(3):295-301, 1994.
- Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Francisco J. R. Ruiz, Julian Schrittwieser, Grzegorz Swirszcz, David Silver, Demis Hassabis, and Pushmeet Kohli. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature , 610 (7930):47-53, 2022.
- A. F. Filippov. Differential equations with discontinuous right-hand side. Mat. Sb. (N.S.) , 51(93):99-128, 1960.
- Hugo Gilbert and Paul Weng. Quantile reinforcement learning. In Proceedings of the Asian Workshop on Reinforcement Leanring , 2016.
- Aditya Gopalan and Gugan Thoppe. Demystifying approximate value-based RL with ϵ -greedy exploration: A differential inclusion analysis. arXiv , 2023.
- Charles R. Harris, K. Jarrod Millman, St´ efan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern´ andez del R´ ıo, Mark Wiebe, Pearu Peterson, Pierre G´ erardMarchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature , 585(7825): 357-362, September 2020.
- John D. Hunter. Matplotlib: A 2d graphics environment. Computing in Science &amp; Engineering , 9(3):90-95, 2007.
- Tommi Jaakkola, Michael I. Jordan, and Satinder P. Singh. On the convergence of stochastic iterative dynamic programming algorithms. Neural Computation , 6(6):1185-1201, 1994.
- Ajin George Joseph and Shalabh Bhatnagar. An adaptive and incremental approach to quantile estimation. In IEEE Conference on Decision and Control , 2019.
- Michael J. Kearns and Satinder Singh. Bias-variance error bounds for temporal difference updates. In Proceedings of the Conference on Learning Theory , 2000.
- Roger Koenker. Quantile regression . Econometric Society Monographs. Cambridge University Press, 2005.
- Roger Koenker and Gilbert Bassett. Regression quantiles. Econometrica: Journal of the Econometric Society , pages 33-50, 1978.
- Roger Koenker, Victor Chernozhukov, Xuming He, and Limin Peng. Handbook of quantile regression . CRC press, 2017.
- Nikolai Nikolaevich Krasovskii and Andrej I. Subbotin. Game-theoretical control problems . Springer, 1988.

- Harold Kushner and Dean Clark. Stochastic approximation methods for constrained and unconstrained systems . Springer, 1978.
- Harold J. Kushner and G. George Yin. Stochastic approximation and recursive algorithms and applications . Springer Science &amp; Business Media, 2003.
- David S. Leslie and Edmund J. Collins. Generalised weakened fictitious play. Games and Economic Behavior , 56(2):285-298, 2006.
- Alix Lh´ eritier and Nicolas Bondoux. A Cram´ er distance perspective on quantile regression based distributional reinforcement learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics , 2022.
- Xiaocheng Li, Huaiyang Zhong, and Margaret L Brandeau. Quantile Markov decision processes. Operations Research , 70(3):1428-1447, 2022.
- Lennart Ljung. Analysis of recursive stochastic algorithms. IEEE Transactions on Automatic Control , 22(4):551-575, 1977.
- Yudong Luo, Guiliang Liu, Haonan Duan, Oliver Schulte, and Pascal Poupart. Distributional reinforcement learning with monotonic splines. In Proceedings of the International Conference on Learning Representations , 2021.
- Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and Michael Bowling. Revisiting the Arcade Learning Environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research , 61:523562, 2018.
- Sean Meyn. Control systems and reinforcement learning . Cambridge University Press, 2022.
- Manuel D. P. Monteiro Marques. Differential inclusions in nonsmooth mechanical problems: Shocks and dry friction . Birkh¨ auser, 2013.
- Tetsuro Morimura, Masashi Sugiyama, Hisashi Kashima, Hirotaka Hachiya, and Toshiyuki Tanaka. Nonparametric return density estimation for reinforcement learning. In Proceedings of the International Conference on Machine Learning , 2010a.
- Tetsuro Morimura, Masashi Sugiyama, Hisashi Kashima, Hirotaka Hachiya, and Toshiyuki Tanaka. Parametric return density estimation for reinforcement learning. In Proceedings of the Conference on Uncertainty in Artificial Intelligence , 2010b.
- Steven Perkins and David S. Leslie. Asynchronous stochastic approximation with differential inclusions. Stochastic Systems , 2(2):409-446, 2013.
- Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization , 30(4):838-855, 1992.
- Hans Rademacher. ¨ Uber partielle und totale Differenzierbarkeit von Funktionen mehrerer Variabeln und ¨ uber die Transformation der Doppelintegrale. Mathematische Annalen , 79(4):340-359, 1919.

- Julia Robinson. An iterative method of solving a game. Annals of Mathematics , 54(2): 296-301, 1951.
- Mark Rowland, Marc G. Bellemare, Will Dabney, R´ emi Munos, and Yee Whye Teh. An analysis of categorical distributional reinforcement learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics , 2018.
- Mark Rowland, Robert Dadashi, Saurabh Kumar, R´ emi Munos, Marc G. Bellemare, and Will Dabney. Statistics and samples in distributional reinforcement learning. In Proceedings of the International Conference on Machine Learning , 2019.
- Mark Rowland, Yunhao Tang, Clare Lyle, R´ emi Munos, Marc G Bellemare, and Will Dabney. The statistical benefits of quantile temporal-difference learning for value estimation. In Proceedings of the International Conference on Machine Learning , 2023.
- David Ruppert. Efficient estimations from a slowly convergent Robbins-Monro process. Technical report, Cornell University, 1988.
- Georgi V. Smirnov. Introduction to the theory of differential inclusions . American Mathematical Society, 2002.
- Richard S. Sutton. Temporal credit assignment in reinforcement learning . PhD thesis, University of Massachusetts Amherst, 1984.
- Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning , 3(1):9-44, 1988.
- Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction . MIT press, 2018.
- John N. Tsitsiklis. Asynchronous stochastic approximation and Q-learning. Machine Learning , 16(3):185-202, 1994.
- John N. Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690, 1997.
- C´ edric Villani. Optimal transport: Old and new . Springer, 2009.
- Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, St´ efan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙ Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antˆ onio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods , 17:261272, 2020.
- Christopher J. C. H. Watkins. Learning from delayed rewards . PhD thesis, University of Cambridge, 1989.

- Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine Learning , 8(3-4): 279-292, 1992.
- Tadeusz Wazewski. Systemes de commande et equations au contingent. Bulletin de l'Academie Polonaise des Sciences. Serie des Sciences Mathematiques, Astronomiques et Physiques , 9:151-155, 1961.
- Peter R. Wurman, Samuel Barrett, Kenta Kawamoto, James MacGlashan, Kaushik Subramanian, Thomas J. Walsh, Roberto Capobianco, Alisa Devlic, Franziska Eckert, Florian Fuchs, Leilani Gilpin, Piyush Khandelwal, Varun Kompella, HaoChih Lin, Patrick MacAlpine, Declan Oller, Takuma Seno, Craig Sherstan, Michael D. Thomure, Houmehr Aghabozorgi, Leon Barrett, Rory Douglas, Dion Whitehead, Peter D¨ urr, Peter Stone, Michael Spranger, and Hiroaki Kitano. Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature , 602(7896):223-228, 2022.
- Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, and Tie-Yan Liu. Fully parameterized quantile function for distributional reinforcement learning. In Advances in Neural Information Processing Systems , 2019.
- Fan Zhou, Jianing Wang, and Xingdong Feng. Non-crossing quantile regression for distributional reinforcement learning. In Advances in Neural Information Processing Systems , 2020.