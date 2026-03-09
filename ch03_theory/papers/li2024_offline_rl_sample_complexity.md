## Settling the Sample Complexity of Model-Based Offline Reinforcement Learning

Gen Li ∗ UPenn

Laixi Shi † CMU

Yuxin Chen ∗‡ UPenn

Yuejie Chi † CMU

April 2022;

Revised: February 2024

## Abstract

This paper is concerned with offline reinforcement learning (RL), which learns using pre-collected data without further exploration. Effective offline RL would be able to accommodate distribution shift and limited data coverage. However, prior algorithms or analyses either suffer from suboptimal sample complexities or incur high burn-in cost to reach sample optimality, thus posing an impediment to efficient offline RL in sample-starved applications.

We demonstrate that the model-based (or 'plug-in') approach achieves minimax-optimal sample complexity without burn-in cost for tabular Markov decision processes (MDPs). Concretely, consider a γ -discounted infinite-horizon (resp. finite-horizon) MDP with S states and effective horizon 1 1 -γ (resp. horizon H ), and suppose the distribution shift of data is reflected by some single-policy clipped concentrability coefficient C ⋆ clipped . We prove that model-based offline RL yields ε -accuracy with a sample complexity of

<!-- formula-not-decoded -->

up to log factor, which is minimax optimal for the entire ε -range . The proposed algorithms are 'pessimistic' variants of value iteration with Bernstein-style penalties, and do not require sophisticated variance reduction. Our analysis framework is established upon delicate leave-one-out decoupling arguments in conjunction with careful self-bounding techniques tailored to MDPs.

Keywords: offline reinforcement learning, model-based approach, minimax lower bounds, distribution shift, pessimism in the face of uncertainty

## Contents

| 1   | Introduction                                           | Introduction                                                                  |   2 |
|-----|--------------------------------------------------------|-------------------------------------------------------------------------------|-----|
|     | 1.1                                                    | Challenges: distribution shift and limited data coverage                      |   3 |
|     | 1.2                                                    | Inadequacy of prior works . . . . . . . . . . . . . . . . .                   |   3 |
|     | 1.3                                                    | Main contributions . . . . . . . . . . . . . . . . . . . . .                  |   5 |
|     | 1.4                                                    | Notation . . . . . . . . . . . . . . . . . . . . . . . . . . .                |   6 |
| 2   | Algorithm and theory: discounted infinite-horizon MDPs | Algorithm and theory: discounted infinite-horizon MDPs                        |   6 |
|     | 2.1                                                    | Models and assumptions . . . . . . .                                          |   7 |
|     | 2.2                                                    | . . . . . . . . . . . Algorithm: VI-LCB for infinite-horizon MDPs . . . . . . |  10 |
|     | 2.3                                                    | Performance guarantees . . . . . . . . . . . . . . . . . .                    |  12 |

∗ Department of Statistics and Data Science, Wharton School, University of Pennsylvania, Philadelphia, PA 19104, USA.

† Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA 15213, USA.

‡ Department of Electrical and Systems Engineering, University of Pennsylvania, Philadelphia, PA 19104, USA.

Yuting Wei ∗ UPenn

| 3 Algorithm and theory: episodic finite-horizon   | 3 Algorithm and theory: episodic finite-horizon            | 3 Algorithm and theory: episodic finite-horizon            |   14 |
|---------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------|
|                                                   | 3.1                                                        | Models and assumptions . . . . . . . . . . . . .           |   14 |
|                                                   | 3.2                                                        | A model-based offline RL algorithm: VI-LCB . . . .         |   16 |
|                                                   | 3.3                                                        | VI-LCB with two-fold subsampling . . . .                   |   17 |
|                                                   | 3.4                                                        | Performance guarantees . . . . . . . . . . . . .           |   18 |
| 4                                                 | Numerical experiments                                      | Numerical experiments                                      |   21 |
| 5                                                 | Related works                                              | Related works                                              |   22 |
| 6                                                 | Analysis: discounted infinite-horizon MDPs                 | Analysis: discounted infinite-horizon MDPs                 |   23 |
|                                                   | 6.1                                                        | Preliminary facts . . . . . . . . . . . . . . . . .        |   24 |
|                                                   | 6.2                                                        | Proof of Theorem 5 . . . . . . . . . . . . . . . .         |   25 |
| 7                                                 | Analysis: episodic finite-horizon MDPs                     | Analysis: episodic finite-horizon MDPs                     |   31 |
|                                                   | 7.1                                                        | Preliminary facts and notation . . . . . . . . .           |   31 |
|                                                   | 7.2                                                        | A crucial statistical independence property . .            |   32 |
|                                                   | 7.3                                                        | Proof of Theorem 3 . . . . . . . . . . . . . . . .         |   33 |
| 8                                                 | Discussion                                                 | Discussion                                                 |   38 |
| A                                                 | Proof of auxiliary lemmas: infinite-horizon MDPs           | Proof of auxiliary lemmas: infinite-horizon MDPs           |   38 |
|                                                   | A.1                                                        | Proof of Lemma 1 . . . . . . . . . . . . . . . .           |   38 |
|                                                   | A.2                                                        | Proof of Lemma 2 . . . . . . . . . . . . . . . .           |   41 |
|                                                   | A.3                                                        | Proof of Lemma 4 . . . . . . . . . . . . . . . .           |   42 |
|                                                   | A.4                                                        | Proof of Lemma 5 . . . . . . . . . . . . . . . .           |   42 |
| B                                                 | Proof of auxiliary lemmas: episodic finite-horizon MDPs    | Proof of auxiliary lemmas: episodic finite-horizon MDPs    |   49 |
|                                                   | B.1 Proof of Lemma 3                                       | . . . . . . . . . . . . . . . .                            |   49 |
|                                                   | B.2 Proof of the instance-dependent statistical bound (65) | B.2 Proof of the instance-dependent statistical bound (65) |   51 |
| C                                                 | Proof of minimax lower bounds                              | Proof of minimax lower bounds                              |   52 |
|                                                   | C.1 Preliminary facts                                      | . . . . . . . . . . . . . . . .                            |   52 |
|                                                   | . C.2 Proof of Theorem 2                                   | . . . . . . . . . . . . . . . .                            |   53 |
|                                                   | C.3 Proof of Theorem 4                                     | . . . . . . . . . . . . . . . .                            |   59 |

## 1 Introduction

Reinforcement learning (RL) has recently achieved superhuman performance in the gaming frontier, such as the game of Go (Silver et al., 2017), under the premise that vast amounts of training data can be obtained. However, limited capability of online data collection in other real-world applications - e.g., clinical trials and online advertising, where real-time data acquisition is expensive, high-stakes, and/or time-consuming, -presents a fundamental bottleneck for carrying such RL success over to broader scenarios. To circumvent this bottleneck, one plausible strategy is to make more effective use of data collected previously, given that such historical data might contain useful information that readily transfers to new tasks (for instance, the state transitions in a historical task might sometimes resemble what happen in new tasks). The potential of this data-driven approach has been explored and recognized in a diverse array of contexts including but not limited to robotic manipulation (Ebert et al., 2018), autonomous driving (Diehl et al., 2021), and healthcare (Tang and Wiens, 2021); see Levine et al. (2020); Prudencio et al. (2022) for overviews of recent development. Nowadays, the subfield of reinforcement learning using historical data, without further exploration of the environment, is commonly referred to as offline RL or batch RL (Lange et al., 2012; Levine et al., 2020). A desired offline RL algorithm would achieve the target statistical accuracy using as few samples as possible.

## 1.1 Challenges: distribution shift and limited data coverage

In contrast to online exploratory RL, offline RL has to deal with several critical issues resulting from the absence of active exploration. Below we single out two representative issues surrounding offline RL.

- Distribution shift. For the most part, the historical data is generated by a certain behavior policy that departs from the optimal one. A key challenge in offline RL thus stems from the shift of data distributions: how to leverage past data to the most effect, even though the distribution induced by the target policy differs from what we have available?
- Limited data coverage. Ideally, if the dataset contained sufficiently many data samples for every stateaction pair, then there would be hope to simultaneously learn the performance of every policy. Such a uniform coverage requirement, however, is oftentimes not only unrealistic (given that we can no longer change the past data) but also unnecessary (given that we might only be interested in identifying a single optimal policy).

Whether one can effectively cope with distribution shift and insufficient data coverage becomes a major factor that governs the feasibility and statistical efficiency of offline RL.

In order to address the aforementioned issues, a recent strand of works put forward the principle of pessimism or conservatism (e.g., Buckman et al. (2020); Chen et al. (2021a); Cui and Du (2022); Jin et al. (2021); Kumar et al. (2020); Liu et al. (2020); Rashidinejad et al. (2022); Shi et al. (2022b); Uehara and Sun (2021); Xie et al. (2021); Yan et al. (2023); Yin and Wang (2021); Zanette et al. (2021); Zhong et al. (2022)). This is reminiscent of the optimism principle in the face of uncertainty for online exploration (Azar et al., 2017; Bourel et al., 2020; Jaksch et al., 2010; Jin et al., 2018; Lai and Robbins, 1985), but works for drastically different reasons (as we shall elucidate momentarily). One plausible idea of the pessimism principle, which has been incorporated into both model-based and model-free approaches, is to penalize value estimation of those state-action pairs that have been poorly covered. Informally speaking, insufficient coverage of a state-action pair inevitably results in low confidence and high uncertainty in the associated value estimation, and it is hence advisable to act cautiously by tuning down the corresponding value estimate. Proper use of pessimism amid uncertainty brings about several provable benefits (Rashidinejad et al., 2022; Xie et al., 2021): (i) it allows for a reduced sample size that adapts to the degree of distribution shift; (ii) as opposed to uniform data coverage, it only requires coverage of the part of the state-action space reachable by the target policy. Details to follow momentarily.

## 1.2 Inadequacy of prior works

In the present paper, we evaluate and compare the statistical performance of offline RL algorithms mainly through the lens of sample complexity - namely, the number of samples needed for an algorithm to output, with probability approaching one, a policy whose resultant value function is at most ε away from optimal (called ' ε -accuracy' throughout). An ultimate goal is to design an algorithm to achieve the smallest possible sample complexity.

Despite extensive recent activities, however, existing statistical guarantees for the above paradigm remain inadequate, as we shall elaborate on below. For concreteness, our discussions focus on two widely studied Markov decision processes (MDPs) with S states and A actions (Bertsekas, 2017): (a) γ -discounted infinitehorizon MDPs, with effective horizon 1 1 -γ ; (b) finite-horizon MDPs with horizon length H and nonstationary transition kernels. We shall bear in mind that all of these salient problem parameters (i.e., S , A , 1 1 -γ , H ) could be enormous in modern RL applications. In addition, previous works have isolated an important parameter C ⋆ ≥ 1 -called the single-policy concentrability coefficient (Rashidinejad et al., 2022; Xie et al., 2021) - that measures the mismatch of distributions induced by the target policy against the behavior policy; see Sections 3.1 and 2.1 for precise definitions. Naturally, the statistical performance of desirable algorithms would degrade gracefully as the distribution mismatch worsens (i.e., as C ⋆ increases). In the sequel, we shall discuss two dinstinctive RL paradigms - model-based RL and model-free RL - separately. Throughout this paper, the standard notation ˜ O ( · ) indicates the order of a function with all log terms in S, A, 1 1 -γ , H, 1 ε , and 1 δ (with 1 -δ the target success probability) hidden.

:

Prior ny samples are needed to ensure

how many samples are needed to ensure paper

Q

Question:

sample complexity

&amp; Mansour'03

sample complexity r '03

cover

(1

SC

1

(

t

/star r '03

-

|S||A|

t

&amp; Mansour'03

1+3

ω

SC

γ

(1

2

1

γ

cover

/star

(1

Srikant '12

-

γ

)

(1

-

γ

(

2

γ

-

)

3

4

-

ε

3

(1

-

0

)

(

2

|S||A|

(1

)

t

cover

SC

/star

1

-

2

µ

γ

γ

(1

3

)

min

-

-

|S||A|

Wierman'20

ω

Q

/star

Question:

[Kearns and Singh, 1999]

/star how many samples are needed to ensure

?

‖

≤

Q

Q

∞

ε

‖

‖

-

∞

how many samples are needed to ensure

(not necessarily convex)

, . . . , m

R

p

?

,

×

≤

,

SC

Prior ar

Q

Lagrangian

‖

Q

Lagrangian

Q

/star

[Kearns and Singh, 1999]

‖

H

̂

̂

ε

p

R

ε

≤

5

∞

/star

)

‖

SC

R

R

p

,

×

m

-

/star

,

Lagrangian

R

D×

p

×

x

)

i

(

,

Lagrangian

i

ν

h

=1

)

∑

i

1

i

5

6-2

6-2

= 1

, . . . , p

=

H

2

.

≤

) = 0

,

= 1

,

(not necessarily convex)

0

i

p

γ

(not necessarily convex)

SC

p

, . . . , m

/star

, . . . , m

/star

i

= 1

0

,

3

,

) = 0

, with do

i

= 1

p

, optimal value

, . . . , p

p

, optimal value

R

/star

(1

×

R

m

R

SC

→

i

/star

f

, with dom

x

-

(

x

i

λ

γ

) +

L

5

p

λ

∑

(

(

0

f

f

SC

i

i

D×

-

x

) +

/star

R

H

γ

9

)

(

) =

i

=1

ν

i

D×

h

i

(

)

x

R

×

/star

i

=1

p

,

6-2

∑

m

5

) =

SC

SC

<!-- image -->

5

̂

-

‖

/star

SC

H

2

.

Duality Duality H 9 SC /star ε = 1 H 2 . 5 Figure 1: An illustration of prior works, where (a) is about discounted infinite-horizon MDPs and (b) is about finite-horizon MDPs. To facilitate comparisons, we replace C ⋆ clipped with C ⋆ in our results when drawing the plots given that C ⋆ clipped ≤ C ⋆ . The shaded regions indicate the state-of-the-art achievability results. Our work manages to close the gaps between the prior achievable regions and the minimax lower bounds.

/star

Model-based offline RL. Model-based algorithms - which can be interpreted as a 'plug-in' statistical approach - start by computing an empirical model for the unknown MDP, and output a policy that is (near)-optimal in accordance with the empirical MDP. When coupled with the pessimism principle, the model-based approach has been shown to enjoy the following sample complexity bounds.

- 1 · By incorporating Hoeffding-style lower confidence bounds into value iteration, Rashidinejad et al. (2022); Xie et al. (2021) demonstrated that a sample complexity of

1

<!-- formula-not-decoded -->

1 1 suffices to yield ε -accuracy. Such a sample complexity bound, however, is a large factor of 1 (1 -γ ) 2 (resp. H 2 ) above the minimax lower limit derived for infinite-horizon MDPs (resp. finite-horizon MDPs) (Rashidinejad et al., 2022; Xie et al., 2021; Yin and Wang, 2021).

- 1 1 1 · In an attempt to optimize the sample complexity, Xie et al. (2021) leveraged the idea of variance reduction - a powerful strategy originating from the stochastic optimization literature (Johnson and Zhang, 2013) - in model-based RL and obtained a strengthened sample complexity of

1

<!-- formula-not-decoded -->

for finite-horizon MDPs. This sample complexity bound approaches the minimax lower limit (i.e., the order of H 4 SC ⋆ ε 2 ) once the sample size exceeds the order of

<!-- formula-not-decoded -->

in other words, an enormous burn-in sample size is needed in order to attain sample optimality.

1

Model-free offline RL. The model-free approach forms a contrastingly different class of RL algorithms, which bypasses the model estimation stage and directly learns the optimal values. Noteworthily, Q-learning

1

)

/star

R

∞

R

/star

m

6-2

D×

SC

/star

, . . . , p learning rate

= 1

6-2

Lagrangian

̂

-

Lagrangian

1

‖

Lagrangian

SC

/star

-

≤

/star

Q

ε

?

?

and its variants (Watkins and Dayan, 1992), which apply stochastic approximation (Robbins and Monro, 1951) based on the Bellman optimality condition, are among the most widely used model-free paradigms. The principle of pessimism amid uncertainty has recently been integrated into model-free algorithms as well, with the state-of-the-art statistical guarantees listed below (Shi et al., 2022b; Yan et al., 2023).

- When Q-learning is implemented in conjunction with Hoeffding-style lower confidence bounds, it has been shown to achieve the same sample complexity as (1), which is suboptimal by a factor of either 1 (1 -γ ) 2 or H 2 .
- A variance-reduced variant of pessimistic Q-learning allows for further sample size benefits, achieving a sample complexity of

<!-- formula-not-decoded -->

for any target accuracy level ε . This means that the algorithm is guaranteed to be sample-optimal only after the total sample size exceeds the order of

<!-- formula-not-decoded -->

which again manifests itself as a significant burn-in cost for long-horizon problems.

Summary. As elucidated above, existing algorithms either suffer from suboptimal sample complexities, or require sophisticated techniques like variance reduction to approach minimax optimality. Even when variance reduction is employed, prior algorithms incur an enormous burn-in cost in order to work optimally, thus posing an impediment to achieving sample efficiency in data-starved applications. Table 1 summarizes quantitatively the previous results, whereas Figure 1 illustrates the gaps between the state-of-the-art upper bounds and the minimax lower bounds (as derived by Rashidinejad et al. (2022); Xie et al. (2021)). All this motivates the studies of the following natural questions:

Can we develop an offline RL algorithm that achieves near-optimal sample complexity without burn-in cost? If so, can we accomplish this goal by means of a simple algorithm without resorting to sophisticated schemes like variance reduction?

The current paper answers these questions affirmatively by studying the model-based approach.

## 1.3 Main contributions

In this paper, we settle the sample complexity of model-based offline RL by studying a pessimistic variant of value iteration - called VI-LCB - applied to some empirical MDP. Encouragingly, for both discounted infinite-horizon and finite-horizon MDPs, the model-based algorithms provably achieve minimax-optimal sample complexities for any given target accuracy level ε -namely, any ε ∈ ( 0 , 1 1 -γ ] for discounted infinitehorizon MDPs and ε ∈ (0 , H ] for finite-horizon MDPs.

To be more precise, we introduce a slightly modified version C ⋆ clipped of the concentrability coefficient C ⋆ , which always satisfies C ⋆ clipped ≤ C ⋆ and shall be termed the single-policy clipped concentrability coefficient (see Sections 2.1 and 3.1 for more details as well as the advantages of this coefficient). The introduction of this new parameter leads to slightly improved sample complexity compared to the one based on C ⋆ . The main contributions are summarized as follows.

- For γ -discounted infinite-horizon MDPs, we demonstrate that with high probability, the VI-LCB algorithm with Bernstein-style penalty finds an ε -optimal policy with a sample complexity of

<!-- formula-not-decoded -->

for any given accuracy level ε ∈ ( 0 , 1 1 -γ ] (see Theorem 1). Our algorithm reuses all samples across all iterations in order to achieve data efficiency, and our analysis builds upon a novel leave-one-out argument to decouple complicated statistical dependency across iterations. Moreover, the above sample complexity (6) remains valid if C ⋆ clipped is replaced by C ⋆ .

- For finite-horizon MDPs with nonstationary transition kernels, we propose a variant of VI-LCB that adopts the Bernstein-style penalty to enforce pessimism in the face of uncertainty. We prove that for any given ε ∈ (0 , H ] , the proposed algorithm yields an ε -optimal policy using

<!-- formula-not-decoded -->

samples with high probability (see Theorem 3). A key ingredient in the algorithm design is a two-fold subsampling trick that helps decouple the statistical dependency along the sample rollouts. Note that the above sample complexity result (7) continues to hold if one replaces C ⋆ clipped with C ⋆ .

- To assess the tightness and optimality of our results, we further develop minimax lower bounds in Theorems 2 and 4, which match the above upper bounds modulo some logarithmic factors.

Remarkably, our algorithms do not require sophisticated variance reduction schemes, as long as suitable confidence bounds are adopted. Detailed theoretical comparisons with prior art can be found in Table 1 and are also illustrated in Figure 1. Finally, we have conducted a series of numerical experiments to evaluate the performance of the proposed algorithms in Section 4.

Statistical contributions: solving the most sample-hungry regime. The offline RL problem considered herein is statistical in nature, in that it seeks to learn from pre-collected data in the face of uncertainty. As far we know, our theory is the first to identify an offline algorithm that provably attains minimax-optimal statistical efficiency for the entire ε -range, which in turn makes clear that no burn-in phase is needed to achieve optimal statistical accuracy. Achieving this requires developing a new suite of statistical theory that works all the way to the most data-hungry regime . It is noteworthy that the existing statistical toolbox not merely for offline RL, but for online exploratory RL as well (as we shall detail in Section 5) - is only guaranteed to work when the total sample size already exceeds a fairly large threshold, a (often unnecessary) requirement that substantially simplifies statistical analysis. In this sense, the regime we aim to solve is reminiscent of the subfield of high-dimensional statistics (Donoho et al., 2000; Wainwright, 2019a) that helps extend the frontier of classical statistics to the sample-starved regime, for which an enriched statistical toolbox is critically needed.

## 1.4 Notation

Throughout this paper, we adopt the convention that 0 / 0 = 0 . We use ∆( S ) to indicate the probability simplex over the set S , and denote by [ H ] the set { 1 , · · · , H } for any positive integer H . We use 1 ( · ) to represent the indicator function. For any vector x = [ x ( s, a )] ( s,a ) ∈S×A ∈ R SA , we overload the notation by letting x 2 = [ x ( s, a ) 2 ] ( s,a ) ∈S×A . For two vectors a = [ a i ] 1 ≤ i ≤ n and b = [ b i ] 1 ≤ i ≤ n , a ◦ b = [ a i b i ] 1 ≤ i ≤ n denotes their Hadamard product, and a ≥ b (resp. a ≤ b ) means a i ≥ b i (resp. a i ≤ b i ) for all i . Following the convention in RL (Agarwal et al., 2021), the norm ∥ · ∥ 1 of a matrix P = [ P ij ] is defined to be ∥ P ∥ 1 := max i ∑ j | P ij | . For any probability vector q ∈ R 1 × S (which is a row vector) and any vector V ∈ R S , define

<!-- formula-not-decoded -->

with qV = ∑ i q i V i , which corresponds to the variance of V w.r.t. the distribution q . The standard notation O ( · ) is adopted to represent the orderwise scaling of a function.

## 2 Algorithm and theory: discounted infinite-horizon MDPs

We begin by studying offline RL in discounted infinite-horizon Markov decision processes. In the following, we shall first introduce the models and assumptions for discounted infinite-horizon MDPs, followed by algorithm design and main results.

Table 1: Comparisons with prior results (up to log terms) regarding finding an ε -optimal policy in offline RL. The ε -range stands for the range of accuracy level ε for which the derived sample complexity is optimal. Here, one always has C ⋆ clipped ≤ C ⋆ ; and the parameter d b min := 1 min s,a,h { d b h ( s,a ): d b h ( s,a ) &gt; 0 } employed in Yin and Wang (2021) could be exceedingly small, with d b h the occupancy distribution of the dataset. While multiple algorithms are referred to as VI-LCB in the table, they correspond to different variants of VI-LCB. Our results are the first to achieve sample optimality for the full ε -range.

| horizon   | algorithm                                                          | sample complexity                                                                 | ε -range to attain sample optimality   | type           |
|-----------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------|----------------|
| infinite  | VI-LCB (Rashidinejad et al., 2022)                                 | SC ⋆ (1 - γ ) 5 ε 2                                                               | --                                     | model-based    |
| infinite  | Q-LCB (Yan et al., 2023)                                           | SC ⋆ (1 - γ ) 5 ε 2                                                               | --                                     | model-free     |
| infinite  | VR-Q-LCB (Yan et al., 2023)                                        | SC ⋆ (1 - γ ) 3 ε 2 + SC ⋆ (1 - γ ) 4 ε                                           | (0 , 1 - γ ]                           | model-free     |
| infinite  | VI-LCB (this paper: Theorem 1) lower bound (this paper: Theorem 2) | SC ⋆ clipped (1 - γ ) 3 ε 2 ( ≤ SC ⋆ (1 - γ ) 3 ε 2 ) SC ⋆ clipped (1 - γ ) 3 ε 2 | ( 0 , 1 1 - γ ] --                     | model-based -- |
| finite    | VI-LCB (Xie et al., 2021)                                          | H 6 SC ⋆ ε 2                                                                      | --                                     | model-based    |
| finite    | VPVI (Yin and Wang, 2021)                                          | H 5 SC ⋆ ε 2                                                                      | --                                     | model-based    |
| finite    | PEVI-Adv (Xie et al., 2021)                                        | H 4 SC ⋆ ε 2 + H 6 . 5 SC ⋆ ε                                                     | ( 0 , 1 H 2 . 5 ]                      | model-based    |
| finite    | LCB-Q-Advantage (Shi et al., 2022b)                                | H 4 SC ⋆ ε 2 + H 5 SC ⋆ ε                                                         | ( 0 , 1 H ]                            | model-free     |
| finite    | APVI/LCBVI (Yin and Wang, 2021)                                    | H 4 SC ⋆ ε 2 + H 4 d b min ε                                                      | (0 ,SC ⋆ d b min ]                     | model-based    |
| finite    | VI-LCB (this paper: Theorem 3)                                     | H 4 SC ⋆ clipped ε 2 ( ≤ H 4 SC ⋆ ε 2 )                                           | (0 ,H ]                                | model-based    |
| finite    | lower bound (this paper: Theorem 4)                                | H 4 SC ⋆ clipped ε 2                                                              | --                                     | --             |

## 2.1 Models and assumptions

Let us begin with some preliminary concepts and notation of discounted infinite-horizon MDPs, followed by a concrete setting specific to offline RL. A more detailed introduction of discounted infinite-horizon MDPs can be found in classical textbooks like Bertsekas (2017).

Basics of discounted infinite-horizon MDPs. Consider a discounted infinite-horizon MDP (Bertsekas, 2017) represented by a tuple M = {S , A , P, γ, r } . The key components of M are: (i) S = { 1 , 2 , · · · , S } : a finite state space of size S ; (ii) A = { 1 , 2 , · · · , A } : an action space of size A ; (iii) P : S × A → ∆( S ) : the transition probability kernel of the MDP (i.e., P ( · | s, a ) denotes the transition probability from state s when action a is executed); (iv) γ ∈ [0 , 1) : the discount factor, so that 1 1 -γ represents the effective horizon; (v) r : S × A → [0 , 1] : the deterministic reward function (namely, r ( s, a ) indicates the immediate reward received when the current state-action pair is ( s, a ) ). Without loss of generality, the immediate rewards are normalized so that they are contained within the interval [0 , 1] . Throughout this section, we introduce the convenient notation

<!-- formula-not-decoded -->

Policy, value function and Q-function. A stationary policy π : S → ∆( A ) is a possibly randomized action selection rule; that is, π ( a | s ) represents the probability of choosing a in state s . When π is a deterministic policy, we abuse the notation by letting π ( s ) represent the action chosen by the policy π in state s . A sample trajectory induced by the MDP under policy π can be written as { ( s t , a t ) } t ≥ 0 , with s t (resp. a t ) denoting the state (resp. action) of the trajectory at time t . To proceed, we shall also introduce the value function V π and Q-value function Q π associated with policy π . Specifically, the value function V π : S → R of policy π is defined as the expected discounted cumulative reward as follows:

<!-- formula-not-decoded -->

where the expectation is taken over the sample trajectory { ( s t , a t ) } t ≥ 0 generated in a way that a t ∼ π ( · | s t ) and s t +1 ∼ P ( · | s t , a t ) for all t ≥ 0 . Given that all immediate rewards lie within [0 , 1] , it is easily verified that 0 ≤ V π ( s ) ≤ 1 1 -γ for any policy π . The Q-function (or action-state function) of policy π can be defined analogously as follows:

<!-- formula-not-decoded -->

which differs from (10) in that it is also conditioned on a 0 = a .

Let ρ ∈ ∆( S ) be a given state distribution. If the initial state is randomly drawn from ρ , then we can define the following weighted value function of policy π :

<!-- formula-not-decoded -->

In addition, we introduce the discounted occupancy distributions associated with policy π as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we consider the randomness over a sample trajectory that starts from an initial state s 0 ∼ ρ and that follows policy π (i.e., a t ∼ π ( · | s t ) and s t +1 ∼ P ( · | s t , a t ) for all t ≥ 0 ).

It is known that there exists at least one deterministic policy - denoted by π ⋆ - that simultaneously maximizes V π ( s ) and Q π ( s, a ) for all state-action pairs ( s, a ) ∈ S×A (Bertsekas, 2017). We use the following shorthand notation to represent respectively the resulting optimal value and optimal Q-function:

<!-- formula-not-decoded -->

Correspondingly, we adopt the notation of the discounted occupancy distributions associated with π ⋆ as:

<!-- formula-not-decoded -->

where the last equality is valid since π ⋆ is assumed to be deterministic.

Offline/batch data. Let us work with an independent sampling model as studied in the prior work Rashidinejad et al. (2022). To be precise, imagine that we observe a batch dataset D = { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N containing N sample transitions. These samples are independently generated based on a distribution d b ∈ ∆( S × A ) and the transition kernel P of the MDP, namely,

<!-- formula-not-decoded -->

In addition, it is assumed that the learner is aware of the reward function.

In order to capture the distribution shift between the desired occupancy measure and the data distribution, we introduce a key quantity previously introduced in Rashidinejad et al. (2022).

Definition 1 (Single-policy concentrability for infinite-horizon MDPs) . The single-policy concentrability coefficient of a batch dataset D is defined as

<!-- formula-not-decoded -->

Clearly, one necessarily has C ⋆ ≥ 1 .

In words, C ⋆ measures the distribution mismatch in terms of the maximum density ratio. The batch dataset can be viewed as expert data when C ⋆ approaches 1, meaning that the batch dataset is close to the target policy in terms of the induced distributions. Moreover, this coefficient C ⋆ is referred to as the 'single-policy' concentrability coefficient since it is concerned with a single policy π ⋆ ; this is clearly a much weaker assumption compared to the all-policy concentrability assumption (as adopted in, e.g., Chen and Jiang (2019); Fan et al. (2019); Farahmand et al. (2010); Munos (2007); Ren et al. (2021); Xie and Jiang (2021)), the latter of which assumes a uniform density-ratio bound over all policies and requires the dataset to be highly exploratory.

In the current paper, we also introduce a slightly improved version of C ⋆ as follows.

Definition 2 (Single-policy clipped concentrability for infinite-horizon MDPs) . The single-policy clipped concentrability coefficient of a batch dataset D is defined as

<!-- formula-not-decoded -->

Remark 1 . A direct comparison of Conditions (18) and (19) implies that for a given batch dataset D ,

<!-- formula-not-decoded -->

As we shall see later, while our sample complexity upper bounds will be mainly stated in terms of C ⋆ clipped , all of them remain valid if C ⋆ clipped is replaced with C ⋆ . Additionally, in contrast to C ⋆ that is always lower bounded by 1, we have a smaller lower bound as follows (directly from the definition (19))

<!-- formula-not-decoded -->

which is nearly tight. 1 This attribute could lead to sample size saving in some cases, to be detailed shortly.

Let us take a moment to further interpret the coefficient in Definition 2, which says that

<!-- formula-not-decoded -->

holds for any pair ( s, a ) . Consider, for instance, the case where C ⋆ clipped = O (1) : if a state-action pair is infrequently (or rarely) visited by the optimal policy, then it is fine for the associated density in the batch data to be very small (e.g., a density proportional to that of the optimal policy); by contrast, if a state-action pair is visited fairly often by the optimal policy, then Definition 2 might only require d b ( s, a ) to exceed the order of 1 /S . In other words, the required level of d b ( s, a ) is clipped at the level 1 C ⋆ clipped S regardless of the value of d ⋆ ( s, a ) .

̸

1 As a concrete example, suppose that d ⋆ ( s ) = { 1 -S -1 S 3 if s = 1 1 S 3 else and d b ( s, a ) =      1 -S -1 S 2 if a = π ⋆ ( s ) and s = 1 , 1 S 2 if a = π ⋆ ( s ) and s = 1 , 0 , else . Then

it can be easily verified that C ⋆ clipped = 1 S -1+ 1 S . Nonetheless, caution should be exercised that an exceedingly small C ⋆ clipped requires highly compressible structure of d ⋆ , and the real-world data often do not fall within this benign range of C ⋆ clipped .

Goal. Armed with the batch dataset D , the objective of offline RL in this case is to find a policy ̂ π that attains near-optimal value functions - with respect to a given test state distribution ρ ∈ ∆( S ) - in a sample-efficient manner. To be precise, for a prescribed accuracy level ε , we seek to identify an ε -optimal policy ̂ π satisfying

<!-- formula-not-decoded -->

with high probability, using a batch dataset D (cf. (17)) containing as few samples as possible. Particular emphasis is placed on achieving minimal sample complexity for the entire range of accuracy levels (namely, for any ε ∈ ( 0 , 1 1 -γ ] ).

## 2.2 Algorithm: VI-LCB for infinite-horizon MDPs

In this subsection, we introduce a model-based offline RL algorithm that incorporates lower concentration bounds in value estimation. The algorithm, called VI-LCB, applies value iteration (based on some pessimistic Bellman operator) to the empirical MDP, with the key ingredients described below.

The empirical MDP. Recall that we are given N independent sample transitions { ( s i , a i , s ′ i ) } N i =1 in the dataset D . For any given state-action pair ( s, a ) , we denote by

<!-- formula-not-decoded -->

the number of samples transitions from ( s, a ) . We then construct an empirical transition matrix ̂ P such that

<!-- formula-not-decoded -->

for each ( s, a, s ′ ) ∈ S × A × S .

The pessimistic Bellman operator. Our offline algorithm is developed based on finding the fixed point of some variant of the classical Bellman operator. Let us first introduce this key operator and eludicate how the pessimism principle is enforced. Recall that the Bellman operator T ( · ) : R SA → R SA w.r.t. the transition kernel P is defined such that for any vector Q ∈ R SA ,

<!-- formula-not-decoded -->

where V = [ V ( s )] s ∈S with V ( s ) := max a Q ( s, a ) . We propose to penalize the original Bellman operator w.r.t. the empirical kernel ̂ P as follows:

<!-- formula-not-decoded -->

where b ( s, a ; V ) denotes the penalty term employed to enforce pessimism amid uncertainty. As one can anticipate, the properties of the fixed point of ̂ T pe ( · ) relies heavily upon the choice of the penalty terms { b h ( s, a ; V ) } , often derived based on certain concentration bounds. In this paper, we focus on the following Bernstein-style penalty to exploit the importance of certain variance statistics:

<!-- formula-not-decoded -->

for every ( s, a ) ∈ S × A , where c b &gt; 0 is some numerical constant (e.g., c b = 144 ), and δ ∈ (0 , 1) is some given quantity (in fact, 1 -δ is the target success probability). Here, for any vector V ∈ R S , we recall that Var ̂ P s,a ( V ) is the variance of V w.r.t. the distribution ̂ P s,a (see (8)).

We immediately isolate several useful properties as follows, whose proof is postponed to Appendix A.1.

Algorithm 1: Offline value iteration with LCB (VI-LCB) for discounted infinite-horizon MDPs

- 1 input: dataset D ; reward function r ; target success probability 1 -δ ; max iteration number τ max . 2 initialization: ̂ Q 0 = 0 , ̂ V 0 = 0 .
- 3 construct the empirical transition kernel ̂ P according to (25).
- 4 for τ = 1 , 2 , · · · , τ max do

6

- 5 for s ∈ S , a ∈ A do

compute the penalty term

7

set

̂

Q

τ

(

s, a

) = max

8

9

b

(

s, a

{

r

s

set

∈ S

τ

̂

V

do

(

s

) = max

a

̂

Q

τ

(

(

s, a s, a

)

) +

.

10 output: ̂ π s.t. ̂ π ( s ) ∈ arg max a ̂ Q τ max ( s, a ) for any s ∈ S .

Lemma 1. For any γ ∈ [ 1 2 , 1) , the operator ̂ T pe ( · ) (cf. (27) ) with the Bernstein-style penalty (28) is a γ -contraction w.r.t. ∥ · ∥ ∞ , that is,

<!-- formula-not-decoded -->

for any Q 1 , Q 2 ∈ R S × A obeying Q 1 ( s, a ) , Q 2 ( s, a ) ∈ [ 0 , 1 1 -γ ] for all ( s, a ) ∈ S × A . In addition, there exists a unique fixed point ̂ Q ⋆ pe of the operator ̂ T pe ( · ) , which also obeys 0 ≤ ̂ Q ⋆ pe ( s, a ) ≤ 1 1 -γ for all ( s, a ) ∈ S × A .

In words, even though ̂ T pe ( · ) integrates the penalty terms, it still preserves the γ -contraction property and admits a unique fixed point, thereby resembling the classical Bellman operator (26).

The VI-LCB algorithm. We are now positioned to introduce the VI-LCB algorithm, which can be regarded as classical value iteration applied in conjunction with pessimism. Specifically, the algorithm applies the Bernstein-style pessimistic operator ̂ T pe (cf. (27)) iteratively in order to find its fixed point:

<!-- formula-not-decoded -->

We shall initialize it to ̂ Q 0 = 0 , implement (30) for τ max iterations, and output ̂ Q = ̂ Q τ max as the final Q-estimate. The final policy estimate ̂ π is chosen on the basis of ̂ Q as follows:

<!-- formula-not-decoded -->

with the whole algorithm summarized in Algorithm 1.

Interestingly, Algorithm 1 is guaranteed to converge rapidly. In view of the γ -contraction property in Lemma 1, the iterates { ̂ Q τ } τ ≥ 0 converge linearly to the fixed point ̂ Q ⋆ pe , as asserted below.

Let us pause to explain the rationale of the pessimism principle on a high level. If a pair ( s, a ) has been insufficiently visited in D (i.e., N ( s, a ) is small), then the resulting Q-estimate ̂ Q τ ( s, a ) could suffer from high uncertainty and become unreliable, which might in turn mislead value estimation. By enforcing suitable penalization b ( s, a ; ̂ V τ -1 ) based on certain lower confidence bounds, we can suppress the negative influence of such poorly visited state-action pairs. Fortunately, suppressing these state-action pairs might not result in significant bias in value estimation when C ⋆ clipped is small; for instance, when the behavior policy π b resembles π ⋆ , the poorly visited state-action pairs correspond primarily to suboptimal actions (as they are not selected by π ⋆ ), making it acceptable to neglect these pairs.

Lemma 2. Suppose ̂ Q 0 = 0 . Then the iterates of Algorithm 1 obey

<!-- formula-not-decoded -->

;

γ

P

̂

V

τ

s,a

̂

-

V

1

τ

̂

)

-

1

for according to (28).

-

(

b

s, a

;

V

τ

̂

-

1

)

,

0

}

.

where ̂ Q ⋆ pe is the unique fixed point of ̂ T pe . As a consequence, by choosing τ max ≥ log N 1 -γ log(1 /γ ) one fulfills

<!-- formula-not-decoded -->

The proof of this lemma is deferred to Appendix A.2.

Algorithmic comparison with Rashidinejad et al. (2022). VI-LCB has been studied in the prior work Rashidinejad et al. (2022). The difference between our algorithm and the version therein is two-fold:

- Sample reuse vs. ˜ O ( 1 1 -γ ) -fold sample splitting. Our algorithm reuses the same set of samples across all iterations, which is in sharp contrast to Rashidinejad et al. (2022) that employs fresh samples in each of the ˜ O ( 1 1 -γ ) iterations. This results in considerably better usage of available information.
- Bernstein-style vs. Hoeffding-style penalty. Our algorithm adopts the Bernstein-type penalty, as opposed to the Hoeffding-style penalty in Rashidinejad et al. (2022). This choice leads to more effective exploitation of the variance structure across time.

Pessimism vs. optimism in the face of uncertainty. The careful reader might also notice the similarity between the pessimism principle and the optimism principle utilized in online RL. A well-developed paradigm that balances exploration and exploitation in online RL is optimistic exploration based on uncertainty quantification (Lai and Robbins, 1985). The earlier work Jaksch et al. (2010) put forward an algorithm called UCRL2 that computes an optimistic policy with the aid of Hoeffding-style confidence regions for the probability transition kernel. Later on, Azar et al. (2017) proposed to build upper confidence bounds (UCB) for the optimal values instead, which leads to significantly improved sample complexity; see, e.g., He et al. (2021); Wang et al. (2019) for the application of this strategy to discounted infinite-horizon MDPs. Note, however, that the rationales behind optimism and pessimism are remarkably different. In offline RL (which does not allow further data collection), the uncertainty estimates are employed to identify, and then rule out, poorly-visited actions; this stands in sharp contrast to the online counterpart where poorly-visited actions might be more favored during exploration.

## 2.3 Performance guarantees

When the Bernstein-style concentration bound (28) is adopted, the VI-LCB algorithm in Algorithm 1 yields ε -accuracy with a near-minimal number of samples, as stated below.

Theorem 1. Suppose γ ∈ [ 1 2 , 1) , and consider any 0 &lt; δ &lt; 1 and ε ∈ ( 0 , 1 1 -γ ] . Suppose that the total number of iterations exceeds τ max ≥ 1 1 -γ log N 1 -γ . With probability at least 1 -2 δ, the policy ̂ π returned by Algorithm 1 obeys

<!-- formula-not-decoded -->

provided that c b (cf. the Bernstein-style penalty term in (28) ) is some sufficiently large numerical constant and the total sample size exceeds

<!-- formula-not-decoded -->

for some large enough numerical constant c 1 &gt; 0 , where C ⋆ clipped is introduced in Definition 2. In addition, the above result continues to hold if C ⋆ clipped is replaced with C ⋆ (introduced in Definition 1).

Remark 2 . Regarding the numerical constants in Theorem 1, a conservative yet concrete sufficient condition is that c b ≥ 144 and c 1 = 21000 c b , which we shall rigorize in the proof.

Before discussing key implications of Theorem 1, we develop matching minimax lower bounds that help confirm the efficacy of the proposed model-based algorithm, whose proof can be found in Appendix C.2.

The proof of this theorem is postponed to Section 6. In general, the total sample size characterized by Theorem 1 could be far smaller than the ambient dimension (i.e., S 2 A ) of the transition kernel P , thus precluding one from estimating P in a reliable fashion. As a crucial insight from Theorem 1, the model-based (or plug-in) approach enables reliable offline learning even when model estimation is completely off.

Theorem 2. For any ( γ, S, C ⋆ clipped , ε ) obeying γ ∈ [ 2 3 , 1 ) , S ≥ 2 , C ⋆ clipped ≥ 8 γ S , and ε ≤ 1 42(1 -γ ) , one can construct two MDPs M 0 , M 1 , an initial state distribution ρ , and a batch dataset with N independent samples and single-policy clipped concentrability coefficient C ⋆ clipped such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

provided that for some numerical constant c 2 &gt; 0 . Here, the infimum is over all estimator ̂ π , and P 0 (resp. P 1 ) denotes the probability when the MDP is M 0 (resp. M 1 ).

Remark 3 . As a more concrete (yet conservative) condition for c 2 , Theorem 2 is valid when c 2 = 1 / 25088 .

Implications. In the following, we take a moment to interpret the above two theorems and single out several key implications about the proposed model-based algorithm.

- Optimal sample complexities. In the presence of the Bernstein-style penalty, the total number of samples needed for our algorithm to yield ε -accuracy is

<!-- formula-not-decoded -->

This taken together with the minimax lower bound asserted in Theorem 2 confirms the optimality of the proposed model-based approach (up to some logarithmic factor). In comparison, the sample complexity derived in Rashidinejad et al. (2022) exhibits a worse dependency on the effective horizon (i.e., 1 (1 -γ ) 5 ). Theorem 2 also enhances the lower bound developed in Rashidinejad et al. (2022) to accommodate the scenario where C ⋆ clipped can be much smaller than C ⋆ , i.e., C ⋆ clipped = O (1 /S ) .

- No burn-in cost. The fact that the sample size bound (35) holds for the full ε -range (i.e., any given ε ∈ ( 0 , 1 1 -γ ] ) means that there is no burn-in cost required to achieve sample optimality. This not only drastically improves upon, but in fact eliminates, the burn-in cost of the best-known sample-optimal result (cf. (5)), the latter of which required a burn-in cost at least on the order of SC ⋆ (1 -γ ) 5 . Accomplishing this requires one to tackle the sample-hungry regime, which is statistically challenging to cope with.
- No need of sample splitting. It is noteworthy that prior works typically required sample splitting. For instance, Rashidinejad et al. (2022) analyzed the VI-LCB algorithm with fresh samples employed in each iteration, which effectively split the data into ˜ O ( 1 1 -γ ) disjoint subsets. In contrast, the algorithm studied herein permits the reuse of all samples across all iterations. This is an important feature in sample-starved applications to effectively maximize information utilization, and is a crucial factor that assists in improving the sample complexity compared to Rashidinejad et al. (2022).
- Sample size saving when C ⋆ clipped &lt; 1 . In view of Theorem 1, the sample complexity of the proposed algorithm can be as low as

when C ⋆ clipped is on the order of 1 /S . This might seem somewhat surprising at first glance, given that the minimax sample complexity for policy evaluation is at least ˜ O ( S (1 -γ ) 3 ε 2 ) even in the presence of a simulator (Azar et al., 2013). To elucidate this, we note that the condition C ⋆ clipped = O (1 /S ) implicitly imposes special - in fact, highly compressible - structure on the MDP that enables sample size reduction. As we shall see from the lower bound construction in Theorem 2, the case with C ⋆ clipped = O (1 /S ) might require d ⋆ ( s, a ) to concentrate on one or a small number of important states, with exceedingly small probability assigned to the remaining ones. If this occurs, then it often suffices to focus on what happens on these important states, thus requiring much fewer samples.

<!-- formula-not-decoded -->

Comparisons with prior statistical analysis. Before concluding this section, we highlight the innovations of our statistical analysis compared to past theory when it comes to discounted infinite-horizon MDPs. To begin with, our sample size improvement over Rashidinejad et al. (2022) stems from the two algorithmic differences mentioned in Section 2.2: the sample-reuse feature allows one to improve a factor of 1 1 -γ , while the use of Bernstein-style penalty yields an additional gain of 1 1 -γ . In addition, while the design of data-driven Bernstein-style bounds has been extensively studied in online RL in discounted MDPs (e.g., He et al. (2021); Zhang et al. (2021b)), all of these past results were either sample-suboptimal, or required a huge burn-in sample size (e.g., S 3 A 2 (1 -γ ) 4 in He et al. (2021)). In other words, sample optimality was not previously achieved in the most data-hungry regime. In comparison, our theory ensures optimality of our algorithm even for the most sample-constrained scenario, which relies on much more delicate statistical tools. In a nutshell, our statistical analysis is built upon at least two ideas: (i) a leave-one-out analysis framework that allows to decouple complicated statistical dependency across iterations without losing statistical tightness; (ii) a delicate self-bounding trick that allows us to simultaneously control multiple crucial statistical quantities (e.g., empirical variance) in the most sample-starved regime.

## 3 Algorithm and theory: episodic finite-horizon MDPs

In this section, we turn attention to the studies of offline RL for episodic finite-horizon MDPs.

## 3.1 Models and assumptions

As before, we briefly state some preliminaries about finite-horizon MDPs, before moving on to the sampling model and the goal. The readers can consult Bertsekas (2017) for more details about finite-horizon MDPs.

Basics of finite-horizon MDPs. Consider the setting of a finite-horizon Markov decision process, as denoted by M = {S , A , H, P, r } . It consists of the following key components: (i) S = { 1 , · · · , S } : a state space of size S ; (ii) A = { 1 , · · · , A } : an action space of size A ; (iii) H : the horizon length; (iv) P = { P h } 1 ≤ h ≤ H , with P h : S × A → ∆( S ) denoting the probability transition kernel at step h (namely, P h ( · | s, a ) stands for the transition probability of the MDP at step h when the current state-action pair is ( s, a ) ); (v) r = { r h } 1 ≤ h ≤ H , with r h : S ×A → [0 , 1] denoting the reward function at step h (namely, r h ( s, a ) indicates the immediate reward gained at step h when the current state-action pair is ( s, a ) ). It is assumed without loss of generality that the immediate rewards fall within the interval [0 , 1] and are deterministic. Conveniently, we introduce the following S -dimensional row vector for any ( s, a, h ) ∈ S × A × [ H ] .

<!-- formula-not-decoded -->

A (possibly randomized) policy π = { π h } 1 ≤ h ≤ H with π h : S → ∆( A ) is an action selection rule, such that π h ( a | s ) specifies the probability of choosing action a when in state s and step h . When π is a deterministic policy, we overload the notation and let π h ( s ) represent the action selected by π in state s at step h . We can generate a sample trajectory { ( s h , a h ) } 1 ≤ h ≤ H by implementing policy π in the MDP M , where s h and a h denote the state and the action in step h , respectively. We then introduce the value function V π = { V π h } 1 ≤ h ≤ H and the Q-function Q π = { Q π h } 1 ≤ h ≤ H associated with policy π ; specifically, the value function V h : S → R of policy π at step h is defined to the be the expected cumulative reward from step h on as a result of policy π , namely,

<!-- formula-not-decoded -->

where the expectation is taken over the randomness over the sample trajectory { ( s t , a t ) } H t = h when policy π is implemented (i.e., a t ∼ π t ( · | s t ) and s t +1 ∼ P t ( · | s t , a t ) for all t ≥ h ). Correspondingly, the Q-function of policy π at step h is defined to be

<!-- formula-not-decoded -->

when conditioned on the state-action pair ( s, a ) at step h . If the initial state is drawn from a distribution ρ ∈ ∆( S ) , we find it convenient to define the following weighted value function of policy π :

<!-- formula-not-decoded -->

Additionally, we introduce the following occupancy distributions associated with policy π at step h :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which are conditioned on the initial state distribution s 1 ∼ ρ and the event that all actions are selected according to π . In particular, it is self-evident that

<!-- formula-not-decoded -->

It is well known that there exists at least one deterministic policy that simultaneously maximizes the value function and the Q-function for all ( s, a, h ) ∈ S × A × [ H ] (Bertsekas, 2017). In light of this, we shall denote by π ⋆ = { π ⋆ h } 1 ≤ h ≤ H an optimal deterministic policy throughout this paper; this allows us to employ π ⋆ h ( s ) to indicate the corresponding optimal action chosen in state s at step h . The resulting optimal value function and optimal Q-function are denoted respectively by V ⋆ = { V ⋆ h } 1 ≤ h ≤ H and Q ⋆ = { Q ⋆ h } 1 ≤ h ≤ H :

<!-- formula-not-decoded -->

Furthermore, we adopt the following notation for convenience:

<!-- formula-not-decoded -->

where the last identity holds given that π ⋆ is assumed to be deterministic.

Offline/batch data. Suppose that we have access to a batch dataset (or historical dataset) D , which comprises a collection of K i.i.d. sample trajectories generated by a behavior policy π b = { π b h } 1 ≤ h ≤ H . More specifically, the k -th sample trajectory ( 1 ≤ k ≤ K ) consists of a data sequence

<!-- formula-not-decoded -->

which is generated by the MDP M under the behavior policy π b in the following manner:

<!-- formula-not-decoded -->

Here and throughout, ρ b stands for some predetermined initial state distribution associated with the batch dataset. In addition to the above dataset (cf. (44) for all 1 ≤ k ≤ K ), the learner also has access to the reward function. For notational simplicity, we introduce the following short-hand notation for the occupancy distribution w.r.t. the behavior policy π b :

<!-- formula-not-decoded -->

In particular, it is easily seen that d b 1 ( s ) = ρ b ( s ) for all s ∈ S . Note that the initial state distribution ρ b of the batch dataset might not coincide with the test state distribution ρ .

Akin to Definition 1, prior works (e.g., Xie et al. (2021)) have introduced the following concentrability coefficient to capture the distribution shift between the desired distribution and the one induced by the behavior policy.

Definition 3 (Single-policy concentrability for finite-horizon MDPs) . The single-policy concentrability coefficient of a batch dataset D is defined as

<!-- formula-not-decoded -->

which clearly satisfies C ⋆ ≥ 1 .

Similar to the discounted infinite-horizon counterpart, C ⋆ employs the largest density ratio (using the occupancy distributions defined above) to measure the distribution mismatch; it concerns the behavior policy vs. a single policy π ⋆ , and does not require uniform coverage of the state-action space (namely, it suffices to cover the part reachable by π ⋆ ). As before, we further introduce a slightly modified version of C ⋆ as follows.

Definition 4 (Single-policy clipped concentrability for finite-horizon MDPs) . The single-policy clipped concentrability coefficient of a batch dataset D is defined as

<!-- formula-not-decoded -->

From the definition above, it holds trivially that

<!-- formula-not-decoded -->

As we shall see shortly, while all sample complexity upper bounds developed herein remain valid if we replace C ⋆ clipped with C ⋆ , the use of C ⋆ clipped might yield some sample size reduction when C ⋆ clipped drops below 1.

Goal. With the above batch dataset D in hand, our aim is to compute, in a sample-efficient fashion, a policy ̂ π that results in near-optimal values w.r.t. a given test state distribution ρ ∈ ∆( S ) . Formally speaking, the current paper focuses on achieving with high probability using as few samples as possible, where ε stands for the target accuracy level. We seek to achieve sample optimality for the full ε -range, i.e., for any ε ∈ (0 , H ] .

<!-- formula-not-decoded -->

## 3.2 A model-based offline RL algorithm: VI-LCB

Suppose for the moment that we have access to a dataset D 0 containing N sample transitions { ( s i , a i , h i , s ′ i ) } N i =1 , where ( s i , a i , h i , s ′ i ) denotes the transition from state s i at step h i to state s ′ i in the next step when action a i is taken. We now describe a pessimistic variant of the model-based approach on the basis of D 0 .

Empirical MDP. For each ( s, a, h ) ∈ S × A × [ H ] , we denote by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the total number of sample transitions at step h that transition from ( s, a ) and from s , respectively. We can then compute the empirical estimate ̂ P = { ̂ P h } 1 ≤ h ≤ H of the transition kernel P as follows:

<!-- formula-not-decoded -->

for each ( s, a, h, s ′ ) ∈ S × A × [ H ] ×S .

The VI-LCB algorithm. With this estimated model in place, the VI-LCB algorithm (i.e., value iteration with lower confidence bounds) maintains the value function estimate { ̂ V h } and Q-function estimate { ̂ Q h } , and works backward from h = H to h = 1 as in classical dynamic programming with the terminal value ̂ V H +1 = 0 (Jin et al., 2021; Xie et al., 2021). Specifically, the algorithm adopts the following update rule:

<!-- formula-not-decoded -->

Algorithm 2: Offline value iteration with LCB (VI-LCB) for finite-horizon MDPs.

```
1 input: dataset D 0 ; reward function r ; target success probability 1 -δ . 2 initialization: ̂ V H +1 = 0 . 3 for h = H, · · · , 1 do 4 compute the empirical transition kernel ̂ P h according to (51). 5 for s ∈ S , a ∈ A do 6 compute the penalty term b h ( s, a ) according to (55). 7 set ̂ Q h ( s, a ) = max { r h ( s, a ) + ̂ P h,s,a ̂ V h +1 -b h ( s, a ) , 0 } . 8 for s ∈ S do 9 set ̂ V h ( s ) = max a ̂ Q h ( s, a ) and ̂ π h ( s ) ∈ arg max a ̂ Q h ( s, a ) . .
```

10 output: ̂ π = { ̂ π h } 1 ≤ h ≤ H

where ̂ P h,s,a is the empirical estimate of P h,s,a (cf. (37)),

<!-- formula-not-decoded -->

and b h ( s, a ) ≥ 0 denotes some penalty term that is a decreasing function in N h ( s, a ) (as we shall specify momentarily). In addition, the policy ̂ π is selected greedily in accordance to the Q-estimate:

<!-- formula-not-decoded -->

In a nutshell, the VI-LCB algorithm - as summarized in Algorithm 2 - applies the classical value iteration approach to the empirical model ̂ P , and in addition, implements the principle of pessimism via certain lower confidence penalty terms { b h ( s, a ) } .

The Bernstein-style penalty terms. As before, we adopt Bernstein-style penalty in order to better capture the variance structure over time; that is,

<!-- formula-not-decoded -->

for some universal constant c b &gt; 0 (e.g., c b = 16 ). Here, Var ̂ P h,s,a ( ̂ V h +1 ) corresponds to the variance of ̂ V h +1 w.r.t. the distribution ̂ P h,s,a (see the definition (8)). Note that we choose ̂ P as opposed to P (i.e., Var P h,s,a ( ̂ V h +1 ) ) in the variance term, mainly because we have no access to the true transition kernel P .

Finally, it is worth noting that the Bernstein-style uncertainty estimates have been widely studied when performing online exploration in episodic finite-horizon MDPs (e.g., Azar et al. (2017); Fruit et al. (2020); Jin et al. (2018); Li et al. (2021); Talebi and Maillard (2018); Zhang et al. (2020)). Once again, the main purpose therein is to encourage exploration of the insufficiently visited states/actions, a mechanism that is not applicable to offline RL due to the absence of further data collection.

## 3.3 VI-LCB with two-fold subsampling

Given that the batch dataset D is composed of several sample trajectories each of length H , the sample transitions in D cannot be viewed as being independently generated (as the sample transitions at step h might influence the sample transitions in the subsequent steps). As one can imagine, the presence of such temporal statistical dependency considerably complicates analysis.

In order to circumvent this technical difficulty, we propose a two-fold subsampling trick that allows one to exploit the desired statistical independence. Informally, we propose the following steps:

- First of all, we randomly split the dataset into two halves D main and D aux , where D main consists of N main h ( s ) sample transitions from state s at step h .

## Algorithm 3: Subsampled VI-LCB for episodic finite-horizon MDPs

- 1 input: r

2 subsampling:

- 1) Data splitting. Split D into two halves: D main (which contains the first K/ 2 trajectories), and D aux (which contains the remaining K/ 2 trajectories); we let N main h ( s ) (resp. N aux h ( s ) ) denote the number of sample transitions in D main (resp. D aux ) that transition from state s at step h .
2. a dataset D ; reward function . run the following procedure to generate the subsampled dataset D trim .
- 2) Lower bounding { N main h ( s ) } using D aux . For each s ∈ S and 1 ≤ h ≤ H , compute

<!-- formula-not-decoded -->

- 3) Random subsampling. Let D main ′ be the set of all sample transitions (i.e., the quadruples taking the form ( s, a, h, s ′ ) ) from D main . Subsample D main ′ to obtain D trim , such that for each ( s, h ) ∈ S × [ H ] , D trim contains min { N trim h ( s ) , N main h ( s ) } sample transitions randomly drawn from D main ′ .

3 run VI-LCB: set D 0 = D trim ; run Algorithm 2 to compute a policy ̂ π .

- For each ( s, h ) ∈ S × [ H ] , we use the dataset D aux to construct a high-probability lower bound N trim h ( s ) on N main h ( s ) , and then subsample N trim h ( s ) sample transitions w.r.t. ( s, h ) from D main ; this results in a new subsampled dataset D trim .
- Run VI-LCB on the subsampled dataset D trim (i.e., Algorithm 2).

The whole procedure is detailed in Algorithm 3. A few important features are worth highlighting, under the assumption that the sample trajectories in D are independently generated from the same distribution.

- Given that { N trim h ( s ) } are computed on the basis of the dataset D aux and that D trim is subsampled from another dataset D main , one can clearly see that { N trim h ( s ) } are statistically independent from the sample transitions in D trim .
- As we shall justify in the analysis (i.e., Section 7.2), the samples in D trim can almost be treated as being statistically independent, a key attribute resulting from the subsampling trick.
- The proposed algorithm only splits the data into two subsets, which is in stark contrast to prior variants of VI-LCB that perform H -fold sample splitting (e.g., Xie et al. (2021)). Eliminating the H -fold splitting requirement plays a crucial role in enabling optimal sample complexity.

Before proceeding, we formally justify that N trim h ( s ) - as computed in (56) - is a valid lower bound on N main h ( s ) . Here and below, we denote by N trim h ( s, a ) the number of sample transitions in D trim that are associated with the state-action pair ( s, a ) at step h . The proof of this lemma can be found in Appendix B.1.

Lemma 3. Suppose that the K trajectories in D are generated in an i.i.d. fashion (see Section 3.1). With probability at least 1 -8 δ , the quantities constructed in (56) obey

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

simultaneously for all 1 ≤ h ≤ H and all ( s, a ) ∈ S × A .

## 3.4 Performance guarantees

In what follows, we characterize the sample complexity of Algorithm 3, as formalized below.

Theorem 3. Consider any ε ∈ (0 , H ] and any 0 &lt; δ &lt; 1 . With probability exceeding 1 -12 δ, the policy ̂ π returned by Algorithm 3 obeys

<!-- formula-not-decoded -->

as long as the penalty terms are chosen according to the Bernstein-style quantity (55) for some large enough numerical constant c b &gt; 0 , and the total number of sample trajectories exceeds

<!-- formula-not-decoded -->

for some sufficiently large numerical constant c k &gt; 0 , where C ⋆ clipped is introduced in Definition 4. Additionally, the above result continues to hold if C ⋆ clipped is replaced with C ⋆ (introduced in Definition 3).

Remark 4 . One concrete yet conservative requirement on c b and c k for Theorem 3 to hold is: c b ≥ 16 and c k = 12800 c b , as we shall solidify in the proof of Theorem 3.

The proof of this result is postponed to Section 7. In general, the total sample size characterized by Theorem 3 could be far smaller than the ambient dimension (i.e., S 2 AH ) of the probability transition kernel P , thus precluding one from estimating P in a reliable fashion. As a crucial insight from Theorem 3, the model-based (or plug-in) approach enables reliable policy learning even when model estimation is completely off. Our analysis of Theorem 3 relies heavily on (i) suitable decoupling of complicated statistical dependency via subsampling, and (ii) careful control of the variance terms in the presence of Bernstein-style penalty.

In order to help assess the tightness and optimality of Theoerem 3, we further develop a minimax lower bound as follows; the proof can be found in Appendix C.3.

Theorem 4. For any ( H,S,C ⋆ clipped , ε ) obeying H ≥ 12 , C ⋆ clipped ≥ 8 /S and ε ≤ c 3 H , one can construct a collection of MDPs {M θ | θ ∈ Θ } , an initial state distribution ρ , and a batch dataset with K independent sample trajectories each of length H , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, c 3 , c 4 &gt; 0 are some small enough numerical constants, the infimum is over all estimator ̂ π , and P θ denotes the probability when the MDP is M θ .

Remark 5 . More concretely, one (conservative) condition regarding c 3 and c 4 that is sufficient for the validity of Theorem 4 is: c 3 = 1 / 2 14 and c 4 = 1 / 2 36 , as we shall see in the proof.

Implications. In what follows, let us take a moment to discuss several other key implications of Theorem 3.

- Near-optimal sample complexities. In the presence of the Bernstein-style penalty, the total number of samples (i.e., KH ) needed for our algorithm to yield ε -accuracy is

<!-- formula-not-decoded -->

This confirms the optimality of the proposed model-based approach (up to some logarithmic term) when Bernstein-style penalty is employed, since Theorem 4 reveals that at least H 4 SC ⋆ clipped ε 2 samples are needed regardless of the algorithm in use.

- Full ε -range and no burn-in cost. The sample complexity bound (59) stated in Theorem 3 holds for an arbitrary ε ∈ (0 , H ] . In other words, no burn-in cost is needed for the algorithm to work sampleoptimally. This improves substantially upon the state-of-the-art results for model-based and model-free offline algorithms, both of which require a significant level of burn-in sample size ( H 9 SC ⋆ and H 6 SC ⋆ , respectively).

provided that the total sample size

- Sample reduction and model compressibility when C ⋆ clipped &lt; 1 . Given that C ⋆ clipped might drop below 1, the sample complexity of our algorithm might be as low as ˜ O ( H 4 S ε 2 ) . In fact, recognizing that C ⋆ clipped can be as small as 1+ o (1) S , we see that the sample complexity can sometimes be reduced to

<!-- formula-not-decoded -->

resulting in significant sample size saving compared to prior works. Caution needs to be exercised, however, that this sample size improvement is made possible as a result of certain model compressibility implied by a small C ⋆ clipped . For instance, C ⋆ clipped = O (1 /S ) might happen when a small number of states accounts for a dominant fraction of probability mass in d ⋆ h ( s ) , with the remaining states exhibiting vanishingly small occupancy probability (see also the lower bound construction in the proof of Theorem 4); if this happens, then it often suffices to focus on learning those dominant states.

(In)-feasibility of estimating C ⋆ clipped . With the sample complexity (62) in mind, one natural question arises as to whether it is possible to estimate C ⋆ clipped from the batch dataset. Unfortunately, this is in general infeasible, as demonstrated by the following example.

- (A hard example) Consider an MDP with horizon H = 2 . In step h = 1 , we have a singleton state space S 1 = { 0 } and an action space A 1 = { 0 , 1 } , whereas in step h = 2 , we have a state space S 2 = { 0 , 1 } and a singleton action space A 2 = { 0 } . The reward function and the transition kernel are given by:

<!-- formula-not-decoded -->

for some unknown parameter p ∈ (0 , 1) . We have K independent trajectories as usual, and let

<!-- formula-not-decoded -->

Elementary calculation then reveals that: C ⋆ clipped = K when p &lt; 1 2 , and C ⋆ clipped = 1+ 1 K -1 when p &gt; 1 2 . Such a remarkable difference in C ⋆ clipped depends on the value of p , which is only reflected in ( s, a ) = (0 , 1) at step 1. However, by construction, there is nonvanishing probability (i.e., ( 1 -d b 1 (0 , 1) ) K ≈ 1 /e for large K ) such that the dataset does not visit ( s, a ) = (0 , 1) in step h = 1 at all, which in turn precludes one from distinguishing C ⋆ clipped = 1 + 1 K -1 from C ⋆ clipped = K given only the available dataset.

Fortunately, implementing our algorithm does not require prior knowledge of C ⋆ clipped at all, and the algorithm succeeds once the task becomes feasible. On the other hand, we won't be able to tell how large a sample size is enough a priori , but this is in general information-theoretically infeasible as illustrated by the above example.

Towards instance optimality. While the primary focus of the current paper is minimax-optimal algorithm design, the theoretical framework developed herein enables instance-dependent analysis as well. Take episodic finite-horizon MDPs for example: our analysis framework directly leads to the following instancedependent guarantee for Algorithm 3:

<!-- formula-not-decoded -->

with the proviso that K ≥ 100 c b HSC ⋆ log NH δ . Encouragingly, the dominate term (i.e., the first term in the bound (65)) matches the instance-dependent lower bound established in (Yin and Wang, 2021, Theorem 4.3), thus confirming the instance optimality of the proposed algorithm for a large enough sample size. The proof of (65) can be found in Appendix B.2.

Figure 2: The performances of the proposed method VI-LCB and the baseline value iteration (VI) in the gambler's problem. It shows that VI-LCB outperforms VI by taking advantage of the pessimism principle and achieves approximately 1 / √ N sample complexity dependency w.r.t. the sample size N .

<!-- image -->

Comparisons with prior statistical analysis. We now briefly discuss the novelty of our statistical analysis compared with past theory. Perhaps the most related prior work is Xie et al. (2021), which proposed two algorithms. The first algorithm therein is VI-LCB with H -fold sample splitting and Hoeffding-style penalty, and each of these two features adds an H factor to the total sample complexity. The second algorithm therein combines VI-LCB with variance reduction, which leads to optimal sample complexity for sufficiently small ε (i.e., a large burn-in cost is required). Note, however, that none of the existing statistical tools for variance reduction is able to work without imposing a large burn-in cost, regardless of the sampling mechanism in use (e.g., generative model, offline RL, online RL) (Li et al., 2021; Sidford et al., 2018; Xie et al., 2021; Zhang et al., 2020). In contrast, our theory makes apparent that variance reduction is unnecessary, which leads to both simpler algorithm and tighter analysis. Additionally, while Bernstein-style confidence bounds have been deployed in online RL for finite-horizon MDPs (Azar et al., 2017; Fruit et al., 2020; Jin et al., 2018; Zhang et al., 2020), none of these works was able to yield optimal sample complexity without a large burn-in cost (e.g., Azar et al. (2017) incurred a burn-in cost as large as S 3 AH 6 ). This in turn underscores the power of our statistical analysis when coping with the most data-hungry regime.

## 4 Numerical experiments

To confirm the practical applicability of the proposed VI-LCB algorithm, we evaluate its performance in the gambler's problem (Panaganti and Kalathil, 2022; Shi and Chi, 2022; Sutton and Barto, 2018; Zhou et al., 2021). The code can be accessed at:

https://github.com/Laixishi/Model-based-VI-LCB .

Gambler's problem. We start by introducing the formulation of the gambler's problem and its underlying MDP. An agent plays a gambling game in which she bets on a sequence of random coin flips, winning when the coins are heads and losing when they are tails. To bet on each random clip, the agent's policy chooses an integer number of dollars based on an initial balance. If the number of bets hits the maximum length H , or if the agent reaches 50 dollars (win) or 0 dollars (lose), the game ends. Without loss of generality, the problem can be formulated as an episodic finite-horizon MDP. Here, S is the state space { 0 , 1 , · · · , 50 } and the associated accessible actions obey a ∈ { 0 , 1 , · · · , min { s, 50 -s } } , H = 100 is the horizon length, the reward is set to 0 for all other states unless s = 50 . For the transition kernel, we fix the probability of heads as p head = 0 . 45 at all steps h ∈ [ H ] in the episode. Moreover, the initial state/balance distribution of the agent ρ is taken as a uniform distribution over S . The offline historical dataset is constructed by collecting N independent samples drawn randomly over each state-action pair and time step.

Evaluation results. First, we evaluate the performance of our proposed method VI-LCB (cf. Algorithm 2) with comparisons to the well-known value iteration (VI) method without the pessimism principle. To begin with, Fig. 2(a) shows the average and standard derivations of the performance gap V ⋆ 1 ( s ) -V ̂ π 1 ( s ) over

all states s ∈ S , over 10 independent experiments with a fixed sample size N = 50 . The results indicate that the proposed VI-LCB method outperforms the baseline VI method uniformly over the entire state space, showing that pessimism brings significant advantages in this sample-scarce regime. Secondly, we evaluate the performance gap V ⋆ 1 ( ρ ) -V ̂ π 1 ( ρ ) with varying sample size N ∈ { 54 , 90 , 148 , · · · , 22026 } ≈ { e 4 , e 4 . 5 , e 5 , · · · , e 10 } , over 10 independent trials. Note that throughout the experiments, we fix the parameter c b = 0 . 05 , which determines the level of the pessimism penalty of VI-LCB (cf. (55)). Fig. 2(b) shows the average and standard derivations of the performance gap V ⋆ 1 ( ρ ) -V ̂ π 1 ( ρ ) with respect to the sample size N . Clearly, as the sample size increases, both our method VI-LCB and the baseline VI method perform better. Moreover, our VI-LCB method consistently outperforms the baseline VI method over the entire range of the sample size N , especially in the sample-starved regime. In addition, to corroborate the scaling of the sample size on the performance gap, we plot the sub-optimality performance gap of VI-LCB w.r.t. the sample size on a log-log scale in Fig. 2(c). Fitting using linear regression leads to a slope estimate of -0 . 502 , with the corresponding fitted line plotted in Fig. 2(c) as well. This nicely matches the finding of Theorem 3, which says the performance gap of VI-LCB scales as N -1 / 2 .

## 5 Related works

In this section, we provide further discussions about prior art, with an emphasis on settings that are most relevant to the current paper.

Off-policy evaluation and offline RL. Broadly speaking, at least two families of problems have been investigated in the literature that tackle offline batch data: off-policy evaluation, where the goal is to estimate the value function of a target policy that deviates from the behavior policy used in data collection; and offline policy learning, where the goal is to identify a near-optimal policy (or at least an improved one compared to the behavior policy). Our work falls under the second category. A topic of its own interest, off-policy evaluation has been extensively studied in the recent literature; we excuse ourselves from enumerating the works in that space but only provide pointers to a few examples including Duan et al. (2020, 2021); Jiang and Huang (2020); Jiang and Li (2016); Kallus and Uehara (2020); Li et al. (2014); Ren et al. (2021); Thomas and Brunskill (2016); Uehara et al. (2020); Xu et al. (2021); Yang et al. (2020).

Offline RL with the pessimism principle. The prior works that are the most relevant to this paper are Jin et al. (2021); Rashidinejad et al. (2022); Shi et al. (2022b); Xie et al. (2021); Yan et al. (2023); Yin and Wang (2021), which incorporated lower confidence bounds into value estimation in order to avoid overly uncertain regions not covered by the target policy. In addition to the ones discussed in Section 1.2 that focus on minimax performance, the recent works Yin et al. (2022); Yin and Wang (2021) further developed instance-dependent statistical guarantees for the pessimistic model-based approach. The results in Yin and Wang (2021), however, required a large burn-in sample size H 4 SC ⋆ ( d b min ) 2 (since d b min could be exceedingly small), thus preventing it from attaining minimax optimality for the entire ε -range. It is noteworthy that the principle of pessimism has been incorporated into policy optimization and actor-critic methods as well by searching for some least-favorable models (e.g., Uehara and Sun (2021); Zanette et al. (2021)), which is quite different from the approach studied herein. On the empirical side, model-based algorithms (Kidambi et al., 2020; Yu et al., 2020) have been shown to achieve superior performance than their model-free counterpart for offline RL. In addition, a number of recent works studied offline RL under various function approximation assumptions, e.g., Jin et al. (2021); Nguyen-Tang et al. (2021); Uehara and Sun (2021); Uehara et al. (2022); Yin et al. (2022); Zanette et al. (2021); Zhan et al. (2022), which are beyond the scope of the current paper. Recently, the insights gleaned from the studies of offline RL have inspired improved algorithm designs for online and hybrid RL as well (Li et al., 2023, 2024c).

Online RL and the optimism principle. The optimism principle in the face of uncertainty has received widespread adoption from bandits to online RL (Agarwal et al., 2021; Lai and Robbins, 1985; Lattimore and Szepesvári, 2020). In the context of online RL, Jaksch et al. (2010) constructed confidence regions for the probability transition kernel to help select optimistic policies in the setting of weakly communicating MDPs, based on a variant (called UCRL2) of the UCRL algorithm originally proposed in Auer and Ortner

(2006); see also Bourel et al. (2020); Filippi et al. (2010); Talebi and Maillard (2018) for other variants of UCRL. When applied to episodic finite-horizon MDPs, the regret bound in Jaksch et al. (2010) was suboptimal by a factor of at least √ H 2 S ; see discussion in Azar et al. (2017); Jin et al. (2018). Fruit et al. (2020) developed an improved regret bound for UCRL2 by using empirical Bernstein-style bounds, which however was still suboptimal by a factor of at least √ HS when specialized to episodic finite-horizon MDPs. In comparison, a more sample-efficient paradigm is to build Bernstein-style UCBs for the optimal values to help select exploration policies, which has been recently adopted in both model-based (Azar et al., 2017; Zhang et al., 2023) and model-free algorithms (Jin et al., 2018). Note that Bernstein-style uncertainty estimation alone is not enough to ensure regret optimality in model-free algorithms, thereby motivating the design of more sophisticated variance reduction strategies (Li et al., 2021; Zhang et al., 2020). Finally, the optimism principle has been studied in undiscounted infinite-horizon MDPs too (e.g., Qian et al. (2019)), which is beyond the scope of this paper.

Model-based RL. The algorithms studied herein fall under the category of model-based RL, which decouples the model estimation and the planning. This popular paradigm has been deployed and studied under various data collection mechanisms beyond offline RL, including but not limited to the generative model (or simulator) setting (Agarwal et al., 2020; Azar et al., 2013; Li et al., 2024b, 2020) and the online exploratory setting (Azar et al., 2017; Jin et al., 2020; Zhang et al., 2023, 2021a). The leave-one-out analysis (and the construction of absorbing MDPs) adopted in the proof of Theorem 1 has been inspired by several recent works Agarwal et al. (2020); Cui and Yang (2021); Li et al. (2024b); Pananjady and Wainwright (2020), and has recently been shown to be effective for multi-agent offline RL as well (Yan et al., 2022).

Model-free RL. Another widely used paradigm is model-free RL, which attempts to learn the optimal value function without explicit construction of the model. Arguably the most famous example of modelfree RL is Q-learning, which applies the stochastic approximation paradigm to find the fixed point of the Bellman operator (Beck and Srikant, 2012; Chen et al., 2020; Even-Dar and Mansour, 2003; Li et al., 2024a, 2022; Murphy, 2005; Qu and Wierman, 2020; Shi et al., 2022a; Szepesvári, 1998; Watkins and Dayan, 1992; Xiong et al., 2020). It is worth noting that the asynchronous Q-learning, which aims to learn the optimal Q-function from a data trajectory collected by following a certain behavior policy, shares some similarity with offline RL; note that prior results on vanilla asynchronous Q-learning require a strong uniform coverage requirement (Chen et al., 2021b; Li et al., 2024a; Qu and Wierman, 2020), which is stronger than the singlepolicy concentrability considered herein. Moreover, Q-learning alone is known to be sub-optimal in terms of the sample complexity in various settings (Bai et al., 2019; Jin et al., 2018; Li et al., 2024a; Shi et al., 2022b; Wainwright, 2019b). This motivates the incorporation of the variance reduction in order to further improve the sample complexity (Du et al., 2017; Li et al., 2021, 2022; Shi et al., 2022b; Wainwright, 2019c; Yan et al., 2023; Zhang et al., 2020, 2021b). Note, however, variance-reduced model-free RL typically requires a large burn-in cost in order to operate in a sample-optimal fashion, and is hence outperformed by the model-based approach under multiple sampling mechanisms.

## 6 Analysis: discounted infinite-horizon MDPs

This section is devoted to establishing Theorem 1. Towards this end, we claim that it is sufficient to prove the following theorem.

Theorem 5. Consider any 0 &lt; δ &lt; 1 and any γ ∈ [ 1 2 , 1) . Suppose that the penalty terms are set to be (28) for any numerical constant c b ≥ 144 . Then with probability exceeding 1 -2 δ, for any estimate ̂ Q obeying ∥ ∥ ̂ Q -̂ Q ⋆ pe ∥ ∥ ∞ ≤ 1 /N one has

<!-- formula-not-decoded -->

where ̂ π ( s ) ∈ arg max a ̂ Q ( s, a ) for any s ∈ S .

As we have demonstrated in Lemma 2, the output of Algorithm 1 satisfies ∥ ∥ ̂ Q -̂ Q ⋆ pe ∥ ∥ ∞ ≤ 1 /N once the iteration number exceeds τ max ≥ log N 1 -γ log(1 /γ ) , thus making Theorem 5 applicable. Taking the right-hand side of (66) to be no larger than ε reveals that: V ⋆ ( ρ ) -V ̂ π ( ρ ) ≤ ε holds as long as N exceeds

<!-- formula-not-decoded -->

The remainder of this section is thus dedicated to establishing Theorem 5. Throughout the proof, it suffices to focus on the case where

<!-- formula-not-decoded -->

for some large constant c 3 ≥ 2880000 ; otherwise the claim (66) follows directly since V ⋆ ( ρ ) -V ̂ π ( ρ ) ≤ 1 1 -γ .

## 6.1 Preliminary facts

Before embarking on the proof, we collect a couple of preliminary facts that will be used multiple times.

Properties of N ( s, a ) . To begin with, the quantity N ( s, a ) -the total number of sample transitions from ( s, a ) - can be bounded through the following lemma; the proof is provided in Appendix A.3.

Lemma 4. Consider any δ ∈ (0 , 1) . With probability at least 1 -δ , the quantities { N ( s, a ) } in (24) obey

<!-- formula-not-decoded -->

simultaneously for all ( s, a ) ∈ S × A .

Properties about ̂ V and ̂ V ⋆ pe . First of all, note that the assumption

<!-- formula-not-decoded -->

given that ε ∈ ( 0 , 1 1 -γ ] .

has the following direct consequence:

<!-- formula-not-decoded -->

Given the proximity of ̂ V and ̂ V ⋆ pe , we can bound the difference of the corresponding variance terms as follows:

<!-- formula-not-decoded -->

Here, (i) follows from the definition (8), the penultimate inequality follows from (71) and the basic facts ∥ ̂ P s,a ∥ 1 = 1 and ∥ ̂ V ∥ ∞ ≤ 1 1 -γ , while the last line relies on (68).

Armed with (72), one can further control the difference of the associated penalty terms. Note that the definition of b ( s, a ; V ) in (28) tells us that

<!-- formula-not-decoded -->

If at least one of the variance terms is not too small in the sense that

<!-- formula-not-decoded -->

then (73) implies that

<!-- formula-not-decoded -->

where (i) results from (74), and (ii) holds due to (72). On the other hand, if (74) is not satisfied, then one clearly has b ( s, a ; ̂ V ⋆ pe ) = b ( s, a ; ̂ V ) . In conclusion, in all cases we have

<!-- formula-not-decoded -->

## 6.2 Proof of Theorem 5

Armed with the preceding preliminary facts, we can readily turn to the proof of Theorem 5. By virtue of Lemma 4, our proof shall - unless otherwise noted - operate on the high-probability event that

<!-- formula-not-decoded -->

In addition, from the sampling model (17), the sample transitions employed to form ̂ P are statistically independent conditional on { N ( s, a ) } . Our proof consists of four steps as detailed below.

Step 1: Bernstein-style inequalities and leave-one-out decoupling argument. We are in need of tight control of the size of ( ̂ P s,a -P s,a ) ̂ V . However, this becomes challenging due to the statistical dependency between ̂ P and the value estimate ̂ V (given that we reuse samples in all iterations of Algorithm 1). In order to circumvent this difficulty, we resort to a leave-one-out argument to decouple the statistical dependency, as motivated by Agarwal et al. (2020); Li et al. (2024b). The result stated below establishes Bernstein-style inequalities despite the complicated dependency.

Lemma 5. Suppose that γ ∈ [ 1 2 , 1) , and consider any δ ∈ (0 , 1) . With probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

simultaneously for all ( s, a ) ∈ S × A and all ˜ V with ∥ ∥˜ V -̂ V ⋆ pe ∥ ∥ ∞ ≤ 1 N and ∥ ˜ V ∥ ∞ ≤ 1 1 -γ .

High-level proof ideas. In short, the proof consists of contructing a finite collection of auxiliary MDPs { ̂ M s,u } for each state s obeying the following properties: (i) each ̂ M s,u is constructed without using any sample transition that comes from state s , and is hence statistically independent from ̂ P s,a for all a ∈ A (instead, the useful information is embedded into the corresponding immediate reward, which is a low-dimensional object and easier to control); (ii) at least one of the MDPs in { ̂ M s,u } is extremely close to the true MDP in terms of the resulting value function. With the aid of these leave-one-out auxiliary MDPs, one can control ( ̂ P s,a -P s,a ) ˜ V by first exploiting the statistical independence between ̂ P s,a and { ̂ M s,u } and then transferring the concentration bound back to the original MDP using the proximity property (ii). The construction of these auxiliary MDPs and the proof details can be found in Appendix A.4.

Note that (78a) has been derived only for those pairs ( s, a ) with N ( s, a ) &gt; 0 . For every ( s, a ) with N ( s, a ) = 0 , one can directly obtain

<!-- formula-not-decoded -->

Putting these bounds together with the definition (28) of b ( s, a ; V ) reveals that

<!-- formula-not-decoded -->

for all ˜ V obeying ∥ ∥˜ V -̂ V ⋆ pe ∥ ∥ ∞ ≤ 1 N and ∥ ˜ V ∥ ∞ ≤ 1 1 -γ , provided that the constant c b is sufficiently large. The remainder of the proof should then also operate on the high-probability events (79) and (78b), in addition to assuming that the event (77) occurs.

Step 2: showing that ̂ Q ( s, a ) is a lower bound on Q ̂ π ( s, a ) . We now justify that ̂ Q ( s, a ) (resp. ̂ V ( s ) ) is a 'pessimistic' estimate of Q ̂ π ( s, a ) (resp. V ̂ π ( s ) ); this is enabled by the pessimism principle (so that the algorithm effectively seeks lower estimates of the value iteration) and the Bernstein-style bounds in Lemma 5 (so that the penalty term always dominates the uncertainty incurred by using the empirical MDP).

To begin with, recall that ̂ Q ⋆ pe ( s, a ) is the unique fixed point of the pessimistic Bellman operator that obeys

<!-- formula-not-decoded -->

In the sequel, we divide the set of state-action pairs ( s, a ) into two types.

- Case 1: ̂ Q ⋆ pe ( s, a ) = 0 . Given that ̂ Q 0 = 0 , Lemma 2 tells us that

<!-- formula-not-decoded -->

This combined with the basic fact Q ̂ π ≥ 0 immediately yields 0 = ̂ Q ( s, a ) ≤ Q ̂ π ( s, a ) .

- Case 2: ̂ Q ⋆ pe ( s, a ) = r ( s, a ) + γ ̂ P s,a ̂ V ⋆ pe -b ( s, a ; ̂ V ⋆ pe ) &gt; 0 . It is first observed that

<!-- formula-not-decoded -->

Here, (i) and (iii) arise from the assumption (70), (ii) relies on the fact that ̂ Q ⋆ pe is the fixed point of the operator ̂ T pe , whereas (iv) takes advantage of (76) and (79). Combining (81) with the Bellman equation Q ̂ π = r + γPV ̂ π results in

<!-- formula-not-decoded -->

Suppose for the moment that there exists some ( s, a ) obeying Q ̂ π ( s, a ) -̂ Q ( s, a ) &lt; 0 (which clearly cannot happen in Case 1), then arg min s,a [ Q ̂ π ( s, a ) -̂ Q ( s, a ) ] must belong to Case 2. Thus, taking the minimum over ( s, a ) and using the above inequality (82) give

<!-- formula-not-decoded -->

where (i) holds since P s,a ∈ ∆( S ) . Given that 1 &gt; γ &gt; 0 , inequality (83) holds only when min s,a [ Q ̂ π ( s, a ) -̂ Q ( s, a ) ] ≥ 0 . We therefore conclude that in this case, one also has Q ̂ π ( s, a ) ≥ ̂ Q ( s, a ) .

With the arguments for the above two cases in place, we arrive at

<!-- formula-not-decoded -->

and evidently,

<!-- formula-not-decoded -->

Step 3: bounding V ⋆ ( s ) -V ̂ π ( s ) . Recall that the Bellman optimality equation gives

<!-- formula-not-decoded -->

Before continuing, we make note of the following lower bound on ̂ V :

<!-- formula-not-decoded -->

Here, (i) results from the assumption (70), (ii) relies on (80), (iii) is valid since ̂ P s,π ⋆ ( s ) ( ̂ V -̂ V ⋆ pe ) ≤ ∥ ∥ ̂ P s,π ⋆ ( s ) ∥ ∥ 1 ∥ ∥̂ V -̂ V ⋆ pe ∥ ∥ ∞ ≤ 1 /N , whereas (iv) holds by virtue of (76) and (79). Armed with the results in (86) and (87), we can readily show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For notational convenience, let us introduce a matrix P ⋆ ∈ R S × S and a vector b ⋆ ∈ R S × 1 whose s -th row are given respectively by

<!-- formula-not-decoded -->

This allows us to rewrite (88) in the following matrix/vector form:

<!-- formula-not-decoded -->

Note that this relation holds for any arbitrary ρ . Apply it recursively to arrive at

<!-- formula-not-decoded -->

where (i) holds since lim i →∞ γ i ρ ⊤ ( P ⋆ ) i ( V ⋆ -̂ V ) = 0 (given that lim i →∞ γ i = 0 and ∥ ρ ⊤ ( P ⋆ ) i ∥ 1 = 1 for any i ≥ 0 ), and the last equality results from the definition of d ⋆ (see (16)) expressed in the following matrix/vector form:

<!-- formula-not-decoded -->

Combine the above inequality with (85) to reach

<!-- formula-not-decoded -->

Step 4: using concentrability to control 〈 d ⋆ , b ⋆ 〉 . We shall control 〈 d ⋆ , b ⋆ 〉 by dividing the state set S into the following two disjoint subsets:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- To begin with, consider any state s ∈ S small . Applying Definition 2 and the definition of S small yields

<!-- formula-not-decoded -->

provided that N &gt; 8 SC ⋆ clipped log NS (1 -γ ) δ (see (68)). This inequality necessarily implies that

<!-- formula-not-decoded -->

Combining the preceding inequality with the following fact (see the definition (28))

<!-- formula-not-decoded -->

we arrive at

<!-- formula-not-decoded -->

- Next, we turn to any state s ∈ S large . Using the definition (28) of b ( s, a ; V ) , we obtain

<!-- formula-not-decoded -->

where (i) arises from Lemma 5 and (71), (ii) applies the elementary inequality √ x + y ≤ √ x + √ y for any x, y ≥ 0 and the fact N ≥ N ( s, a ) , in addition to assuming that c b is large enough. To continue, we observe that

<!-- formula-not-decoded -->

where (i) follows from the assumption (77) and the definition of S large , and (ii) results from Assumption 2. Substitution into (99) yields

<!-- formula-not-decoded -->

where the last line comes from the elementary inequality √ x + y ≤ √ x + √ y for any x, y ≥ 0 . To proceed, observe that the sum of the first terms in (101) satisfies

<!-- formula-not-decoded -->

where (i) arises from the Cauchy-Schwarz inequality and the fact ∑ s d ⋆ ( s ) = 1 . In addition, it is easily verified that the sum of the second terms in (101) obeys

<!-- formula-not-decoded -->

which also makes use of the identity ∑ s d ⋆ ( s ) = 1 . Combining (102) and (103) with (101) gives

<!-- formula-not-decoded -->

The above results (98) and (104) taken collectively give

<!-- formula-not-decoded -->

Here, (i) follows when c b is sufficiently large and C ⋆ clipped ≥ 1 /S (see (21)), (ii) would hold as long as the following inequality could be established:

<!-- formula-not-decoded -->

(iii) is valid since γ ∈ [ 1 2 , 1) , and (iv) follows from the elementary inequality 2 xy ≤ x 2 + y 2 . Rearranging terms, we are left with

<!-- formula-not-decoded -->

which combined with (93) yields

<!-- formula-not-decoded -->

This concludes the proof, as long as the inequality (105) can be established.

Proof of inequality (105) . To begin with, we make the observation that

<!-- formula-not-decoded -->

where (i) holds since b ⋆ ≥ 0 and ̂ V + γP ⋆ ̂ V ≥ 0 , (ii) follows from the basic property ∥ ̂ V + γP ⋆ ̂ V ∥ ∞ ≤ 2 ∥ ̂ V ∥ ∞ ≤ 2 1 -γ and the fact ̂ V -γP ⋆ ̂ V +2 b ⋆ ≥ 0 , the latter of which has been verified in (87). Armed with this fact, one can deduce that

<!-- formula-not-decoded -->

Here, (i) follows by invoking the definition (8), (ii) holds due to (108), (iii) is valid since γ &lt; 1 , (iv) is a direct consequence of (92), while (v) comes from the basic facts ∥ ρ ⊤ ∥ 1 = 1 and ∥ ̂ V ∥ ∞ ≤ 1 1 -γ .

## 7 Analysis: episodic finite-horizon MDPs

## 7.1 Preliminary facts and notation

We first collect a few preliminary facts that are useful for the analysis. The first fact determines the range of our estimates ̂ Q h and ̂ V h .

Lemma 6. The iterates of Algorithm 2 obey

<!-- formula-not-decoded -->

Proof. The non-negativity of ̂ Q h (and hence ̂ V h ) follows directly from the update rule (52). Regarding the upper bound, we suppose for the moment that ̂ V h +1 ( s ) ≤ H -h for step h +1 . Then (52) tells us that

<!-- formula-not-decoded -->

which together with ̂ V h ( s ) = max a ̂ Q h ( s, a ) justifies the claim (109) for step h as well. Taking this together with the base case ̂ V H +1 = 0 and the standard induction argument concludes the proof.

The second fact is concerned with the vector d ⋆ h := [ d ⋆ h ( s )] s ∈S ∈ R S . For any h ∈ [ H ] , denote by P ⋆ h ∈ R S × S a matrix whose s -th row is given by P h ( · | s, π ⋆ h ( s ) ) . Then the Markovian property of the MDP indicates that: for any j &gt; h , one has

<!-- formula-not-decoded -->

Notation. We remind the reader that P h,s,a ∈ R 1 × S represents the probability transition vector P h ( · | s, a ) , and the associated variance parameter Var P h,s,a ( V ) is defined to be the ( h, s, a ) -th row of Var P ( V ) (cf. (8)), namely,

<!-- formula-not-decoded -->

for any given vector V ∈ R S . The vector ̂ P h,s,a ∈ R 1 × S and the variance parameter Var ̂ P h,s,a ( V ) are defined analogously.

## 7.2 A crucial statistical independence property

This subsection demonstrates that the subsampling trick introduced in Section 3.3 leads to some crucial statistical independence property. To be precise, let us consider the following two data-generating mechanisms; here and below, a sample transition refers to a quadruple ( s, a, h, s ′ ) that indicates a transition from state s to state s ′ when action a is taken at step h .

- Model 1 (augmented dataset). Augment D trim to yield a dataset D trim , aug via the following steps. For every ( s, h ) ∈ S × [ H ] :
- 1) Add to D trim , aug all N trim h ( s ) sample transitions in D trim that transition from state s at step h ;
- 2) If N trim h ( s ) &gt; N main h ( s ) , then we further add to D trim , aug another set of N trim h ( s ) -N main h ( s ) independent sample transitions {( s, a ( i ) h,s , h, s ′ ( i ) h,s )} obeying

<!-- formula-not-decoded -->

- Model 2 (independent dataset). For every ( s, h ) ∈ S × [ H ] , generate N trim h ( s ) independent sample transitions {( s, a ( i ) h,s , h, s ′ ( i ) h,s )} as follows:

<!-- formula-not-decoded -->

This forms the following dataset:

<!-- formula-not-decoded -->

In words, the dataset D trim , aug generated in Model 1 differs from D trim only if N trim h ( s ) &gt; N main h ( s ) occurs; this data generating mechanism ensures that D trim , aug comprises exactly N trim h ( s ) sample transitions from state s at step h . Two key features are: (a) the samples in D trim , aug are statistically independent, and (b) D trim , aug is essentially equivalent to D trim with high probability, as asserted below.

Lemma 7. The above two datasets D trim , aug and D i . i . d . have the same distributions. In addition, with probability exceeding 1 -8 δ , D trim , aug = D trim .

Proof. Both D trim , aug and D i . i . d . contain exactly N trim h ( s ) sample transitions from state s at step h . where { N trim h ( s ) } are statistically independent from the randomness of the samples. It is easily seen that: given { N trim h ( s ) } , the sample transitions in D trim , aug across different steps are statistically independent. As a result, D trim and D i . i . d . both consist of independent samples and are of the same distribution.

Furthermore, Lemma 3 tells us that with probability at least 1 -8 δ , N trim h ( s ) ≤ N main h ( s ) holds for all ( s, h ) ∈ S × [ H ] , implying that that all data in D trim , aug come from D main and hence D trim , aug = D trim .

## 7.3 Proof of Theorem 3

We first demonstrate that Theorem 3 is valid as long as the following theorem can be established.

Theorem 6. Consider the dataset D 0 described in Section 3.2, and any 0 &lt; δ &lt; 1 . Suppose that D 0 contains N sample transitions, and that the non-negative integers { N h ( s, a ) } defined in (50) obey

<!-- formula-not-decoded -->

with K some quantity obeying K ≥ 3872 HSC ⋆ clipped log NH δ . Assume that conditional on { N h ( s, a ) } , the sample transitions { ( s, a, h, s ′ ( i ) ) | 1 ≤ i ≤ N h ( s, a ) , ( s, a, h ) ∈ S ×A× [ H ] } are statistically independent. The penalty terms are taken to be (55) , where c b ≥ 16 is chosen to be some constant. Then with probability at least 1 -4 δ , one has

<!-- formula-not-decoded -->

By construction, { N trim h ( s, a ) } are computed using D aux , and hence are independent from the empirical model ̂ P h generated based on D trim . Additionally, Lemma 7 permits us to treat the samples in D trim as being statistically independent. Recalling that the lower bound (57b) holds with probability at least 1 -8 δ , we can readily invoke Theorem 6 by taking N h ( s, a ) = N trim h ( s, a ) and the property (42) to show that

<!-- formula-not-decoded -->

with probability at least 1 -12 δ , provided that K ≥ 3872 HSC ⋆ clipped log KH δ . Setting the right-hand side of (117) to be smaller than ε immediately concludes the proof of Theorem 3, where we have used the fact that N ≤ KH in D 0 . As a consequence, it suffices to establish Theorem 6. In the sequel, we shall assume without loss of generality that we are working on the high-probability event (57).

## 7.3.1 Proof of Theorem 6

Step 1: showing that ̂ Q h ( s, a ) ≤ Q ̂ π h ( s, a ) . This part relies crucially on the following lemma.

Lemma 8. Consider any 1 ≤ h ≤ H , and any vector V ∈ R S independent of ̂ P h obeying ∥ V ∥ ∞ ≤ H . With probability at least 1 -4 δ/H , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

simultaneously for all ( s, a ) ∈ S × A .

Proof. The proof follows from exactly the same argument as that of Lemma 9, except that the assumed upper bound on ∥ V ∥ ∞ is now H (as opposed to 1 1 -γ ) and δ is replaced with δ/H . We thus omit the proof details for brevity.

Additionally, we make note of the crude bound ∣ ∣ ( ̂ P h,s,a -P h,s,a ) ̂ V h +1 ∣ ∣ ≤ ∥ ̂ V h +1 ∥ ∞ ≤ H . Also, given that Algorithm 2 works backwards, the iterate ̂ V h +1 does not use ̂ P h , and is hence statistically independent from ̂ P h . Thus, we can readily apply Lemma 8 to obtain

<!-- formula-not-decoded -->

in the presence of the Bernstein-style penalty (55), provided that the constant c b &gt; 0 is sufficiently large.

In the sequel, we shall work with the high-probability events (120) and (119), in addition to (57). We intend to prove the following relation

<!-- formula-not-decoded -->

hold with probability exceeding 1 -4 δ . Note that the latter assertion concerning ̂ V h is implied by the former, according to the following relation:

<!-- formula-not-decoded -->

Therefore, we focus on the first assertion and will show it by induction. First of all, the claim (121) holds trivially for the base case with h = H + 1 , given that ̂ Q H +1 ( s, a ) = Q ̂ π H +1 ( s, a ) = 0 . Next, suppose that ̂ Q h +1 ( s, a ) ≤ Q ̂ π h +1 ( s, a ) holds for all ( s, a ) ∈ S × A and some step h +1 . We would like to show that the claimed inequality holds for step h as well. If ̂ Q h ( s, a ) = 0 , then the claim holds trivially; otherwise, our update rule (52) reveals that

<!-- formula-not-decoded -->

with probability at least 1 -δ/ 2 , where (i) results from (120) and (122) (i.e., ̂ V h +1 ( s ) ≤ V ̂ π h +1 ( s ) ), and (ii) arises from the Bellman equation. We have thus established (121) via a standard induction argument.

Step 2: bounding V ⋆ h ( s ) -V ̂ π h ( s ) . In view of (122), we make the observation that

<!-- formula-not-decoded -->

where the last inequality holds true since V ⋆ h ( s ) = Q ⋆ h ( s, π ⋆ h ( s )) and ̂ V h ( s ) = max a ̂ Q h ( s, a ) ≥ ̂ Q h ( s, π ⋆ h ( s )) . Recognizing that

<!-- formula-not-decoded -->

we can continue the derivation of (123) to obtain

<!-- formula-not-decoded -->

with probability at least 1 -δ , where the last inequality is valid due to (120). For notational convenience, let us introduce a sequence of matrices P ⋆ h ∈ R S × S ( 1 ≤ h ≤ H ) and vectors b ⋆ h ∈ R S ( 1 ≤ h ≤ H ), with their s -th rows given by

<!-- formula-not-decoded -->

This allows us to rewrite (124) in matrix/vector form as follows:

<!-- formula-not-decoded -->

The inequality (126) plays a key role in the analysis since it establishes a connection between the value estimation errors in step h and step h +1 .

Given that b ⋆ h , P ⋆ h and V ⋆ h -̂ V h are all non-negative, applying (126) recursively with the boundary condition V ⋆ H +1 = ̂ V H +1 = 0 leads to

<!-- formula-not-decoded -->

where we adopt the following notation for convenience (note the order of the product)

<!-- formula-not-decoded -->

With this inequality in mind, we can let d ⋆ h := [ d ⋆ h ( s )] s ∈S be a S -dimensional vector and derive

<!-- formula-not-decoded -->

where we have made use of (123) and the elementary identity (110).

Step 3: using concentrability to bound ⟨ d ⋆ j , b ⋆ j ⟩ . To finish up, we need to make use of the concentrability coefficient. In what follows, we look at two cases separately.

- Case 1: Kd b j ( s, π ⋆ j ( s ) ) ≤ 4 c b log NH δ . Given that b h ( s, a ) ≤ H (cf. (55)), we necessarily have

<!-- formula-not-decoded -->

in this case, where the last inequality arises from Definition 4.

- Case 2: Kd b j ( s, π ⋆ j ( s ) ) &gt; 4 c b log NH δ . It follows from the assumption (115) that

<!-- formula-not-decoded -->

as long as c b &gt; 0 is sufficiently large. Here, the last line results from Definition 4 and the assumption that π ⋆ is a deterministic policy (so that d ⋆ j ( s ) = d ⋆ j ( s, π ⋆ j ( s ) ) ). This further leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, (i) comes from (119) and the elementary inequality √ x + y ≤ √ x + √ y for any x, y ≥ 0 , provided that c b is large enough; and (ii) relies on (129).

Putting the above two cases together, we arrive at

<!-- formula-not-decoded -->

where the last inequality holds since

<!-- formula-not-decoded -->

In addition, we make the observation that

<!-- formula-not-decoded -->

where the third line makes use of the Cauchy-Schwarz inequality, and the last line would hold as long as we could establish the following inequality

<!-- formula-not-decoded -->

for all h ∈ [ H ] with probability exceeding 1 -4 δ . Substitution into (130) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from the elementary inequality 2 xy ≤ x 2 + y 2 . Rearranging terms, we are left with

<!-- formula-not-decoded -->

provided that K ≥ 3872 HSC ⋆ clipped log NH δ . This taken collectively with (127) completes the proof of Theorem 6, as long as the inequality (131) can be validated.

Proof of inequality (131) . First of all, we observe that

<!-- formula-not-decoded -->

for any s ∈ S , where (i) is a consequence of (120), and the last line arises from (52) and (53). This implies the non-negativity of the vector ̂ V j +2 b ⋆ j -P ⋆ j ̂ V j +1 , which in turn allows one to deduce that

<!-- formula-not-decoded -->

where the last line relies on Lemma 6. Consequently, we can demonstrate that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) arises from (132) as well as the basic property ( d ⋆ j ) ⊤ P ⋆ j = ( d ⋆ j +1 ) ⊤ , (ii) follows by rearranging terms and using the property ( d ⋆ j ) ⊤ P ⋆ j = ( d ⋆ j +1 ) ⊤ once again, and (iii) holds due to the fact that ∥ ̂ V h ∥ ∞ ≤ H and ∥ d ⋆ h ∥ 1 = 1 . This concludes the proof of (131).

## 8 Discussion

Our primary contribution has been to pin down the sample complexity of model-based offline RL for the tabular settings, by establishing its (near) minimax optimality for both infinite- and finite-horizon MDPs. While reliable estimation of the transition kernel is often infeasible in the sample-starved regime, it does not preclude the success of this 'plug-in' approach in learning the optimal policy. Encouragingly, the sample complexity characterization we have derived holds for the entire range of target accuracy level ε , thus revealing that sample optimality comes into effect without incurring any burn-in cost. This is in stark contrast to all prior results, which either suffered from sample sub-optimality or required a large burnin sample size in order to yield optimal efficiency. We have demonstrated that sophisticated techniques like variance reduction are not necessary, as long as Bernstein-style lower confidence bounds are carefully employed to capture the variance of the estimates in each iteration.

Turning to future directions, we first note that the two-fold subsampling adopted in Algorithm 3 is likely unnecessary; it would be of interest to develop sharp analysis for the VI-LCB algorithm without sample splitting, which would call for more refined analysis in order to handle the complicated statistical dependency between different time steps. Notably, while avoiding sample splitting cannot improve the sample complexity in an order-wise sense, the potential gain in terms of the pre-constants as well as the algoritmic simplicity might be of practical interest. Moreover, given the appealing memory efficiency of model-free algorithms, understanding whether one can design sample-optimal model-free offline algorithms with minimal burn-in periods is another open direction. Moving beyond tabular settings, it would be of great interest to extend our analysis to accommodate model-based offline RL in more general scenarios; examples include MDPs with low-complexity linear representations, and offline RL involving multiple agents.

## Acknowledgements

Y. Wei is supported in part by the NSF grants CCF-2106778, DMS-2147546/2015447, the NSF CAREER award DMS-2143215, and the Google Research Scholar Award. Y. Chen is supported in part by the Alfred P. Sloan Research Fellowship, the Google Research Scholar Award, the AFOSR grants FA9550-19-1-0030 and FA9550-22-1-0198, the ONR grant N00014-22-1-2354, and the NSF grants CCF-2221009, CCF-1907661, DMS-2014279, IIS-2218713 and IIS-2218773. L. Shi and Y. Chi are supported in part by the grants ONR N00014-19-1-2404, NSF CCF-2106778 and DMS-2134080, and CAREER award ECCS-1818571. L. Shi is also gratefully supported by the Leo Finzi Memorial Fellowship, Wei Shen and Xuehong Zhang Presidential Fellowship, and Liang Ji-Dian Graduate Fellowship at Carnegie Mellon University. Part of this work was done while G. Li, Y. Chen and Y. Wei were visiting the Simons Institute for the Theory of Computing.

## A Proof of auxiliary lemmas: infinite-horizon MDPs

## A.1 Proof of Lemma 1

Before embarking on the proof, we introduce several notation. To make explicit the dependency on V , we shall express the penalty term using the following notation throughout this subsection:

<!-- formula-not-decoded -->

For any Q,Q 1 , Q 2 ∈ R SA , we write

<!-- formula-not-decoded -->

for all s ∈ S . Unless otherwise noted, we assume that

<!-- formula-not-decoded -->

throughout this subsection. In addition, let us define another operator ˜ T pe obeying

<!-- formula-not-decoded -->

for any Q ∈ R SA . It is self-evident that

<!-- formula-not-decoded -->

γ -contraction. The main step of the proof lies in showing the monotonicity of the operator ˜ T pe in the sense that

<!-- formula-not-decoded -->

Suppose that this claim is valid for the moment, then one can demonstrate that: for any Q 1 , Q 2 ∈ R SA ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with 1 denoting the all-one vector. Additionally, observe that

<!-- formula-not-decoded -->

for any constant c , which together with the identity ̂ P 1 = 1 immediately leads to

<!-- formula-not-decoded -->

Taking this together with (138) yields

<!-- formula-not-decoded -->

which combined with the basic property ∥ ∥̂ T pe ( Q 1 ) -̂ T pe ( Q 2 ) ∥ ∥ ∞ ≤ ∥ ∥˜ T pe ( Q 1 ) -˜ T pe ( Q 2 ) ∥ ∥ ∞ (as a result of (136)) justifies that

<!-- formula-not-decoded -->

The remainder of the proof is thus devoted to establishing the monotonicity property (137).

Proof of the monotonicity property (137) . Consider any point Q ∈ R SA , and we would like to examine the derivative of ˜ T pe at point Q . Towards this end, we consider any ( s, a ) ∈ S × A and divide into several cases.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the derivative of ˜ T pe ( Q )( s, a ) w.r.t. the s ′ -th component of V leads to

<!-- formula-not-decoded -->

for any s ′ ∈ S .

- Case 2: √ c b log N (1 -γ ) δ N ( s,a ) Var ̂ P s,a ( V ) &lt; 2 c b log N (1 -γ ) δ (1 -γ ) N ( s,a ) &lt; 1 1 -γ . The penalty (133) in this case reduces to

<!-- formula-not-decoded -->

an expression that is independent of V . As a result, repeating the argument for Case 1 indicates that (140) continues to hold for this case.

- Case 3: 2 c b log N (1 -γ ) δ (1 -γ ) N ( s,a ) &lt; √ c b log N (1 -γ ) δ N ( s,a ) Var ̂ P s,a ( V ) &lt; 1 1 -γ . In this case, the penalty term is given by

<!-- formula-not-decoded -->

Note that in this case, we necessarily have

<!-- formula-not-decoded -->

which together with the definition in (8) indicates that

<!-- formula-not-decoded -->

As a result, for any s ′ ∈ S , taking the derivative of b ( s, a ; V ) w.r.t. the s ′ -th component of V gives

<!-- formula-not-decoded -->

where the penultimate inequality relies on (141), and the last inequality is valid since V ( s ′ ) = max a Q ( s ′ , a ) ≤ 1 1 -γ and γ ≥ 1 / 2 . In turn, the preceding relation allows one to derive

<!-- formula-not-decoded -->

for any s ′ ∈ S .

Putting the above cases together reveals that

<!-- formula-not-decoded -->

holds almost everywhere (except for the boundary points of these cases). Recognizing that ˜ T pe ( Q ) is continuous in Q and that V is non-decreasing in Q , one can immediately conclude that

<!-- formula-not-decoded -->

Existence and uniqueness of fixed points. To begin with, note that for any 0 ≤ Q ≤ 1 1 -γ · 1 , one has 0 ≤ ̂ T pe ( Q ) ≤ 1 1 -γ · 1 . If we produce the following sequence recursively:

<!-- formula-not-decoded -->

then the standard proof for the Banach fixed-point theorem (e.g., Agarwal et al. (2001, Theorem 1)) tells us that Q ( t ) converges to some point Q ( ∞ ) as t → ∞ . Clearly, Q ( ∞ ) is a fixed point of ̂ T pe obeying 0 ≤ Q ( ∞ ) ≤ 1 1 -γ · 1 .

We then turn to justifying the uniqueness of fixed points of ̂ T pe . Suppose that there exists another point ˜ Q obeying ˜ Q = ̂ T pe ( ˜ Q ) , which clearly satisfies ˜ Q ≥ 0 . If ∥ ˜ Q ∥ ∞ &gt; 1 1 -γ , then

<!-- formula-not-decoded -->

resulting in contradiction. Consequently, one necessarily has 0 ≤ ˜ Q ≤ 1 1 -γ · 1 . Further, the γ -contraction property (139) implies that

<!-- formula-not-decoded -->

Given that γ &lt; 1 , this inequality cannot happen unless ˜ Q = Q ∞ , thus confirming the uniqueness of Q ∞ .

## A.2 Proof of Lemma 2

Let us first recall the monotone non-decreasing property (137) of the operator ˜ T pe defined in (135), which taken together with the property (136) readily yields

<!-- formula-not-decoded -->

for any Q and ˜ Q obeying Q ≤ ˜ Q , 0 ≤ Q ≤ 1 1 -γ · 1 and 0 ≤ ˜ Q ≤ 1 1 -γ · 1 (with 1 the all-one vector). Given that ̂ Q 0 = 0 ≤ ̂ Q ⋆ pe , we can apply (143) to obtain

<!-- formula-not-decoded -->

Repeat this argument recursively to arrive at

<!-- formula-not-decoded -->

In addition, it comes directly from Lemma 1 that

<!-- formula-not-decoded -->

for any τ ≥ 0 , where the last inequality is valid since ̂ Q 0 = 0 and ∥ ̂ Q ⋆ pe ∥ ∞ ≤ 1 1 -γ (see Lemma 1). The other claim (33) also follows immediately by taking the right-hand side of (144) to be no larger than 1 /N .

## A.3 Proof of Lemma 4

For any ( s, a ) ∈ S × A , if Nd b ( s,a ) 12 &lt; 2 3 log SN δ , then it is self-evident that this pair satisfies (69). As a consequence, it suffices to focus attention on the following set of state-action pairs:

<!-- formula-not-decoded -->

To bound the cardinality of N large , we make the observation that

<!-- formula-not-decoded -->

thus leading to the crude bound

<!-- formula-not-decoded -->

Let us now look at any ( s, a ) ∈ N large . Given that N ( s, a ) can be viewed as the sum of N independent Bernoulli random variables each with mean d b ( s, a ) , we can apply the Bernstein inequality to yield

<!-- formula-not-decoded -->

for any τ ≥ 0 , where we define

<!-- formula-not-decoded -->

A little algebra then yields that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Combining this result with the union bound over ( s, a ) ∈ N large and making use of (146) give: with probability at least 1 -δ ,

<!-- formula-not-decoded -->

holds simultaneously for all ( s, a ) ∈ N large . Recalling that Nd b ( s, a ) ≥ 8 log NS δ holds for any ( s, a ) ∈ N large , we can easily verify that

<!-- formula-not-decoded -->

thereby establishing (69) for any ( s, a ) ∈ N large . This concludes the proof.

## A.4 Proof of Lemma 5

If N ( s, a ) = 0 , then the inequalities hold trivially. Hence, it is sufficient to focus on the case where N ( s, a ) &gt; 0 . Before proceeding, we make note of a key Bernstein-style result; the proof is deferred to Appendix A.4.1.

Lemma 9. Consider any given pair ( s, a ) ∈ S × A with N ( s, a ) &gt; 0 . Let V ∈ R S be any vector independent of ̂ P s,a obeying ∥ V ∥ ∞ ≤ 1 1 -γ . With probability at least 1 -4 δ , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 6 . In words, Lemma 9 develops a Bernstein bound (150a) on ∣ ∣ ( ̂ P s,a -P s,a ) V ∣ ∣ that makes clear the importance of the variance parameter. Lemma 9 (cf. (150b)) also ascertains that the variance w.r.t. the empirical distribution ̂ P s,a does not deviate much from the variance w.r.t. the true distribution P s,a .

Equipped with this result, we are now ready to present the proof of Lemma 5, which is built upon a leave-one-out decoupling argument and consists of the following steps.

Step 1: construction of auxiliary state-absorbing MDPs. Recall that ̂ M is the empirical MDP. For each state s ∈ S and each scalar u ≥ 0 , we construct an auxiliary state-absorbing MDP ̂ M s,u in a way that makes it identical to the empirical MDP ̂ M except for state s . More specifically, the transition kernel of the auxiliary MDP ̂ M s,u - denoted by P s,u - is chosen such that

<!-- formula-not-decoded -->

and the reward function of ̂ M s,u - denoted by r s,u - is set to be

<!-- formula-not-decoded -->

̸

In words, the probability transition kernel of ̂ M s,u is obtained by dropping all randomness of ̂ P s,a ( a ∈ A ) that concerns state s and making s an absorbing state. In addition, let us define the pessimistic Bellman operator ̂ T s,u pe based on the auxiliary MDP ̂ M s,u such that

<!-- formula-not-decoded -->

for any ( s, a ) ∈ S × A , where the penalty term is taken to be

<!-- formula-not-decoded -->

Step 2: the correspondence between the empirical MDP and auxiliary MDP. Taking

<!-- formula-not-decoded -->

we claim that there exists a fixed point ̂ Q ⋆ s,u ⋆ of ̂ T s,u ⋆ pe whose corresponding value function ̂ V ⋆ s,u ⋆ coincides with ̂ V ⋆ pe . To justify this, it suffices to verify the following properties:

- Consider any a ∈ A . Given that P s,u ( · | s, a ) only has a single non-zero entry (equal to 1), it is easily seen that Var P s,u ( · | s,a ) ( V ) = 0 holds for any V and any u , thus indicating that

<!-- formula-not-decoded -->

Consequently, for state s , one has

<!-- formula-not-decoded -->

where the third identity makes use of our choice (153) of u ⋆ and (154).

̸

- Next, consider any s ′ = s and any a ∈ A . We make the observation that

̸

<!-- formula-not-decoded -->

where the last relation holds since ̂ Q ⋆ pe is a fixed point of ̂ T pe .

Armed with (155) and (156), we see that ̂ Q ⋆ s,u ⋆ = ̂ T s,u ⋆ pe ( ̂ Q ⋆ s,u ⋆ ) by taking

<!-- formula-not-decoded -->

̸

This readily confirms the existence of a fixed point of ̂ T s,u ⋆ pe whose corresponding value coincides with ̂ V ⋆ pe .

Step 3: building an ϵ -net. Consider any ( s, a ) ∈ S × A with N ( s, a ) &gt; 0 . Construct a set U cover as follows

<!-- formula-not-decoded -->

with u max = min { 2 c b log N (1 -γ ) δ (1 -γ ) N ( s,a ) , 1 1 -γ } + 5 N + 1 . This can be viewed as the ϵ -net (Vershynin, 2018) of the range [0 , u max ] ⊆ [ 0 , 2 1 -γ ] with ϵ = 1 /N . Let us construct an auxiliary MDP ̂ M s,u as in Step 1 for each u ∈ U cover . Repeating the argument in the proof of Lemma 1 (see Section A.1), we can easily show that there exists a unique fixed point ̂ Q ⋆ s,u of ̂ M s,u , which also obeys 0 ≤ ̂ Q ⋆ s,u ≤ 1 1 -γ · 1 . In what follows, we denote by ̂ V ⋆ s,u the corresponding value function of ̂ Q ⋆ s,u .

Recognizing that ̂ M s,u is statistically independent from ̂ P s,a for any u ∈ U cover (by construction), we can apply Lemma 9 in conjunction with the union bound (over all u ∈ U cover ) to show that, with probability exceeding 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold simultaneously for all u ∈ U cover . Clearly, the total number of ( s, a ) pairs with N ( s, a ) &gt; 0 cannot exceed N . Thus, taking the union bound over all these pairs yield that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold simultaneously for all ( s, a, u ) ∈ S × A × U cover obeying N ( s, a ) &gt; 0 .

Step 4: a covering argument. In this step, we shall work on the high-probability event (158) that holds simultaneously for all u ∈ U cover . Given that ̂ V ⋆ pe satisfies the trivial bound 0 ≤ ̂ V ⋆ pe ( s ) ≤ 1 1 -γ for all s ∈ S , one can find some u 0 ∈ U cover such that | u 0 -u ⋆ | ≤ 1 /N , where we recall the choice of u ⋆ in (153). From the definition of the MDP ̂ M s,u and the operator (151), it is readily seen that

<!-- formula-not-decoded -->

holds for any Q ∈ R SA . Consequently, we can use γ -contraction of the operator to obtain

<!-- formula-not-decoded -->

which implies that and therefore

<!-- formula-not-decoded -->

This in turn allows us to demonstrate that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third line comes from the fact that E [ X ] = arg min c E [( X -c ) 2 ] , and the last line relies on the property 0 ≤ ̂ V ⋆ s,u 0 , ̂ V ⋆ s,u ⋆ ≤ 1 1 -γ . In addition, by swapping ̂ V ⋆ s,u 0 and ̂ V ⋆ s,u ⋆ , we can derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly, this bound (160) continues to be valid if we replace P s,a with ̂ P s,a .

With the above perturbation bounds in mind, we can invoke the triangle inequality and (159a) to reach

<!-- formula-not-decoded -->

where the second line holds since

<!-- formula-not-decoded -->

and then

the penultimate line is valid due to (160), and the last line holds true under the conditions that T ≥ N ( s, a ) and that T is sufficiently large. Moreover, apply (159b) and the triangle inequality to arrive at

<!-- formula-not-decoded -->

where (i) arise from (159b) and (160), (ii) follows from (160), and the last line holds true since N ≥ N ( s, a ) .

Step 5: extending the bounds to ˜ V . Consider any ˜ V obeying ∥ ˜ V -̂ V ⋆ pe ∥ ∞ ≤ 1 N and ∥ ˜ V ∥ ∞ ≤ 1 1 -γ . Invoke (161) and the triangle inequality to arrive at

<!-- formula-not-decoded -->

where the penultimate inequality relies on N ≥ N ( s, a ) , and the second line holds since

<!-- formula-not-decoded -->

Given that ∥ ∥˜ V -̂ V ⋆ pe ∥ ∥ ∞ ≤ 1 /N , we can repeat the argument for (160) allows one to demonstrate that

<!-- formula-not-decoded -->

which taken together with (163) and the basic inequality √ x + y ≤ √ x + √ y gives

<!-- formula-not-decoded -->

Additionally, repeating the argument for (162) leads to another desired inequality:

<!-- formula-not-decoded -->

## A.4.1 Proof of Lemma 9

In this proof, we shalle often use Var s,a to abbreviate Var P s,a for notational simplicity. Before proceeding, let us define the following vector

<!-- formula-not-decoded -->

with 1 denoting the all-one vector. It is clearly seen that

<!-- formula-not-decoded -->

In addition, we make note of the following basic facts that will prove useful:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of inequality (150a) . If 0 &lt; N ( s, a ) &lt; 48 log N δ , then we can immediately see that

<!-- formula-not-decoded -->

and hence the claim (150a) is valid. As a result, it suffices to focus on the case where

<!-- formula-not-decoded -->

Note that the total number of pairs ( s, a ) with nonzero N ( s, a ) cannot exceed N . Akin to (148), taking the Bernstein inequality together with (166) and invoking the union bound, we can demonstrate that with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

hold simultaneously over all ( s, a ) with N ( s, a ) &gt; 0 . Note, however, that the Bernstein bounds in (169) involve the variance Var s,a ( V ) ; we still need to connect Var s,a ( V ) with its empirical estimate Var ̂ P s,a ( V ) . In the sequel, let us look at two cases separately.

- Case 1: Var s,a ( V ) ≤ 9 log N δ (1 -γ ) 2 N ( s,a ) . In this case, our bound (169a) immediately leads to

<!-- formula-not-decoded -->

- Case 2: Var s,a ( V ) &gt; 9 log N δ (1 -γ ) 2 N ( s,a ) . We first single out the following useful identity:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (171) with (169b) then implies that, with probability exceeding 1 -4 δ ,

<!-- formula-not-decoded -->

where the second line arises from the identity (171), the penultimate inequality results from (169b), and the last inequality holds true due to the assumption Var s,a ( V ) &gt; 9 log N δ (1 -γ ) 2 N ( s,a ) in this case. Rearranging terms of the above inequality, we are left with

<!-- formula-not-decoded -->

Taking this upper bound on Var s,a ( V ) collectively with (169a) and using a little algebra lead to

<!-- formula-not-decoded -->

with probability at least 1 -4 δ . When N ( s, a ) ≥ 48 log N δ (cf. (168)), one has √ 12 log N δ N ( s,a ) ≤ 1 / 2 . Substituting this into (174) and rearranging terms, we arrive at

<!-- formula-not-decoded -->

with probability at least 1 -4 δ .

Putting the above two cases together establishes the advertised bound (150a).

Proof of inequality (150b) . It follows from (172) and (169a) that with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

Invoke the elementary inequality 2 xy ≤ x 2 + y 2 to establish the claimed bound:

<!-- formula-not-decoded -->

## B Proof of auxiliary lemmas: episodic finite-horizon MDPs

## B.1 Proof of Lemma 3

(a) Let us begin by proving the claim (57a). Recall from our construction that D aux is composed of the second half of the sample trajectories, and hence for each s ∈ S and 1 ≤ h ≤ H ,

<!-- formula-not-decoded -->

can be viewed as the sum of K/ 2 independent Bernoulli random variables, each with mean d b h ( s ) . According to the union bound and the Bernstein inequality, we obtain

<!-- formula-not-decoded -->

for any τ ≥ 0 , where

<!-- formula-not-decoded -->

A little algebra then yields that with probability at least 1 -2 δ , one has

<!-- formula-not-decoded -->

simultaneously for all s ∈ S and all 1 ≤ h ≤ H . The same argument also reveals that with probability exceeding 1 -2 δ ,

<!-- formula-not-decoded -->

holds simultaneously for all s ∈ S and all 1 ≤ h ≤ H . Combine (175) and (176) to show that

<!-- formula-not-decoded -->

for all s ∈ S and all 1 ≤ h ≤ H .

To establish the claimed result (57a), we divide into two cases.

- Case 1: N aux h ( s ) ≤ 100 log HS δ . By construction, it is easily seen that

<!-- formula-not-decoded -->

- Case 2: N aux h ( s ) &gt; 100 log HS δ . In this case, invoking (175) reveals that

<!-- formula-not-decoded -->

and hence one necessarily has

<!-- formula-not-decoded -->

In turn, this property (179) taken collectively with (148) ensures that

<!-- formula-not-decoded -->

Therefore, in the case with N aux h ( s ) &gt; 100 log HS δ , we can demonstrate that

<!-- formula-not-decoded -->

where (i) comes from Condition (180), (ii) is valid under the condition (179), and (iii) holds true with probability at least 1 -2 δ due to the inequality (177).

Putting the above two cases together establishes the claim (57a).

(b) We now turn to the second claim (57b). Towards this, we first claim that the following bound holds simultaneously for all ( s, a, h ) ∈ S × A × [ H ] with probability exceeding 1 -2 δ :

<!-- formula-not-decoded -->

Let us take this claim as given for the moment, and return to establish it towards the end of this section. We shall discuss the following two cases separately.

- If Kd b h ( s, a ) = Kd b h ( s ) π b h ( a | s ) &gt; 1600 log KH δ , then it follows from (180) (with slight modification) that

<!-- formula-not-decoded -->

This property together with the definition of N trim h ( s ) in turn allows us to derive

<!-- formula-not-decoded -->

and as a result,

<!-- formula-not-decoded -->

where the last inequality arises from the assumption of this case. Taking this lower bound with (182) implies that

<!-- formula-not-decoded -->

- If Kd b h ( s, a ) ≤ 1600 log KH δ , then one can easily verify that

<!-- formula-not-decoded -->

Putting these two cases together concludes the proof, provided that the claim (182) is valid.

Proof of inequality (182) . Let us look at two cases separately.

- If N trim h ( s ) π b h ( a | s ) ≤ 4 log KH δ , then the right-hand side of (182) is negative, and hence the claim (182) holds trivially.
- We then turn attention to the following set:

<!-- formula-not-decoded -->

Recognizing that

<!-- formula-not-decoded -->

we can immediately bound the cardinality of A large as follows:

<!-- formula-not-decoded -->

Additionally, it follows from our construction that: conditional on N trim h ( s ) , N main h ( s ) and the highprobability event (57a), N trim h ( s, a ) can be viewed as the sum of min { N trim h ( s ) , N main h ( s ) } = N trim h ( s ) independent Bernoulli random variables each with mean π h ( a | s ) . As a result, repeating the Bernsteintype argument in (148) on the event (57a) reveals that, with probability at least 1 -2 δ/ ( KH ) ,

<!-- formula-not-decoded -->

for any fixed triple ( s, a, h ) . Taking the union bound over all ( s, a, h ) ∈ A large and using the bound (185) imply that with probability exceeding 1 -δ , (186) holds simultaneously for all ( s, a, h ) ∈ A large .

Combining the above two cases allows one to conclude that with probability at least 1 -δ , the advertised property (182) holds simultaneously for all ( s, a, h ) ∈ S × A × [ H ] .

## B.2 Proof of the instance-dependent statistical bound (65)

To establish relation (65), we make use of relation (171) as follows: for any 1 ≤ h ≤ H ,

<!-- formula-not-decoded -->

provided that K ≥ 100 c b HSC ⋆ log NH δ . To see why the last inequality in (187) holds, it suffices to observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) holds due to the elementary inequality √ Var ( X + Y ) ≤ √ Var ( X ) + √ Var ( Y ) ; (ii) follows since

<!-- formula-not-decoded -->

which comes from the fact that V ⋆ j +1 ≥ ̂ V j +1 ≥ 0 and ∥ V ⋆ j +1 ∥ ∞ ≤ H ; (iii) invokes the Cauchy-Schwarz inequality; (iv) makes use of the definition of C ⋆ clipped ; and (v) is obtained by applying (116) of Theorem 6.

## C Proof of minimax lower bounds

## C.1 Preliminary facts

For any two distributions P and Q , we denote by KL ( P ∥ Q ) the Kullback-Leibler (KL) divergence of P and Q . Letting Ber ( p ) be the Bernoulli distribution with mean p , we also introduce

<!-- formula-not-decoded -->

which represent respectively the KL divergence and the chi-square divergence of Ber ( p ) from Ber ( q ) (Tsybakov, 2009). We make note of the following useful properties about the KL divergence.

Lemma 10. For any p, q ∈ [ 1 2 , 1 ) and p &gt; q , it holds that

<!-- formula-not-decoded -->

Proof. The second inequality in (190) is a well-known relation between KL divergence and chi-square divergence; see Tsybakov (2009, Lemma 2.7). As a result, it suffices to justify the first inequality. Towards this end, let us introduce a = p + q 2 ∈ [ 1 2 , 1 ] and b = p -q 2 ∈ [ 0 , 1 4 ] , which allow us to re-parameterize ( p, q ) as p = a + b and q = a -b . The definition (189) together with a little algebra gives

<!-- formula-not-decoded -->

Taking the derivative w.r.t. b yields

<!-- formula-not-decoded -->

with f ( x ) := 2 x x + b + 2 x x -b (for x &gt; b ). Here, the last inequality follows since f ( · ) is a decreasing function and that a ≥ 1 -a . This implies that g ( a, b ) is non-increasing in b ≥ 0 for any given a , which in turn leads to as claimed.

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 2

We now construct some hard problem instances and use them to establish the minimax lower bounds claimed in Theorem 2. It is assumed throughout this subsection that

<!-- formula-not-decoded -->

## C.2.1 Construction of hard problem instances

Construction of the hard MDPs. Let us introduce two MDPs {M θ = ( S , A , P θ , r, γ ) | θ ∈ { 0 , 1 }} parameterized by θ , which involve S states and 2 actions as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We single out a crucial state distribution (supported on the state subset { 0 , 1 } ) as follows:

for some quantity C &gt; 0 obeying

<!-- formula-not-decoded -->

We shall make clear the relation between C and the concentrability coefficient C ⋆ clipped shortly (see (203)). Armed with this distribution, we are ready to define the transition kernel P θ of the MDP M θ as follows:

where the parameters p and q are chosen to be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In view of the assumptions (191), one has

<!-- formula-not-decoded -->

As can be clearly seen from the construction, if the MDP is initialized to either state 0 or state 1, then it will never leave the state subset { 0 , 1 } . In addition, the reward function for any MDP M θ is chosen to be

<!-- formula-not-decoded -->

where the reward gained in state 0 is clearly higher than that in other states.

Value functions and optimal policies. Next, let us take a moment to compute the value functions of the constructed MDPs and identify the optimal policies. For notational clarity, for the MDP M θ with θ ∈ { 0 , 1 } , we denote by π ⋆ θ the optimal policy, and let V π θ (resp. V ⋆ θ ) represent the value function of policy π (resp. π ⋆ θ ). The lemma below collects several useful properties about the value functions and the optimal policies; the proof is deferred to Appendix C.2.3.

Lemma 11. Consider any θ ∈ { 0 , 1 } and any policy π . One has

<!-- formula-not-decoded -->

where we define

<!-- formula-not-decoded -->

In addition, the optimal policy π ⋆ θ and the optimal value function obey

<!-- formula-not-decoded -->

Construction of the batch dataset. Given any constructed MDP M θ , we generate a dataset containing N i.i.d. samples { ( s i , a i , s ′ i ) } 1 ≤ i ≤ N according to (17), where the initial state distribution ρ b and behavior policy π b are chosen to be:

<!-- formula-not-decoded -->

with µ denoting the distribution defined in (192). Interestingly, the occupancy state distribution of this dataset coincides with µ , in the sense that

<!-- formula-not-decoded -->

Moreover, letting us choose the test distribution ρ in a way that

<!-- formula-not-decoded -->

we can also characterize the single-policy clipped concentrability coefficient C ⋆ clipped of the dataset w.r.t. the constructed MDP M θ as follows

<!-- formula-not-decoded -->

The proof of the claims (201) and (203) can be found in Appendix C.2.4.

## C.2.2 Establishing the minimax lower bound

Equipped with the above construction, we are ready to develop our lower bounds. We remind the reader of the test distribution ρ chosen in (202), and hence we need to control ⟨ ρ, V ⋆ θ -V ̂ π θ ⟩ = V ⋆ θ (0) -V ̂ π θ (0) with ̂ π representing a policy estimate (computed based on the batch dataset).

Step 1: converting ̂ π into an estimate ̂ θ of θ . Consider first an arbitrary policy π . By combining the definition (199) with the properties (200), we see that x π ⋆ θ ,θ = p , which together with (198) gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the second line holds since V π θ ≤ V ⋆ θ , and the last inequality will be established in Appendix C.2.4.

<!-- formula-not-decoded -->

Denoting by P θ the probability distribution when the MDP is M θ , suppose for the moment that the policy estimate ̂ π achieves then in view of (204), one necessarily has ̂ π ( θ | 0) ≥ 13 21 with probability at least 7 / 8 . If this were true, then we could then construct the following estimate ̂ θ for θ :

<!-- formula-not-decoded -->

which would necessarily satisfy

<!-- formula-not-decoded -->

In what follows, we would like to show that (206) cannot happen - i.e., one cannot possibly find such a good estimator for θ - without a sufficient number of samples.

Step 2: probability of error in testing two hypotheses. The next step lies in studying the feasibility of differentiating two hypotheses θ = 0 and θ = 1 . Define the minimax probability of error as follows

<!-- formula-not-decoded -->

̸

̸

where the infimum is taken over all possible tests ψ (based on the batch dataset in hand). Letting µ b θ denote the distribution of a sample ( s i , a i , s ′ i ) under the MDP M θ and recalling that the samples are independently generated, one can demonstrate that

<!-- formula-not-decoded -->

Here, the first inequality results from Tsybakov (2009, Theorem 2.2) and the additivity property of the KL divergence (cf. Tsybakov (2009, Page 85)), and the second line holds true since

<!-- formula-not-decoded -->

where the second line is valid since P 0 ( · | s, a ) and P 1 ( · | s, a ) differ only when s = 0 .

Next, we turn attention to the KL divergence of interest. Recall that

<!-- formula-not-decoded -->

Given that p ≥ q ≥ 1 / 2 (see (196)), we can apply Lemma 10 to arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) arises from Lemma 10, (ii) follows from the definitions of p and q (195), (iii) holds true as long as 14(1 -γ ) 2 ε γ ≤ 1 -γ 2 , and (iv) results from the assumption γ ∈ [ 1 2 , 1) . Evidently, the same upper bound holds for KL ( P 0 ( · | 0 , 1) ∥ P 1 ( · | 0 , 1) ) as well. Substitution back into (208) reveals that: if the sample size does not exceed then one necessarily has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3: putting all this together. To finish up, suppose that there exists an estimator ̂ π such that

<!-- formula-not-decoded -->

Then in view of our arguments in Step 1, the estimator ̂ θ defined in (205) must satisfy

<!-- formula-not-decoded -->

̸

̸

This, however, cannot possibly happen under the sample size condition (209); otherwise it contradicts the lower bound (210).

## C.2.3 Proof of Lemma 11

To begin with, for any policy π , the value function of state 0 obeys

<!-- formula-not-decoded -->

where in (i) we have defined the following quantity

<!-- formula-not-decoded -->

and (ii) relies on the fact that µ (0) + µ (1) = 1 . Rearranging terms in (211), we are left with

<!-- formula-not-decoded -->

Additionally, the value function of state 1 can be calculated as

<!-- formula-not-decoded -->

where (i) arises from the elementary property 0 ≤ V π θ ( s ) ≤ 1 1 -γ for any π and s ∈ S , and (ii) comes from the assumption (193). The above observation reveals several facts:

- If we take π (0 | 1) = 1 , then (214) tells us that

<!-- formula-not-decoded -->

- It also follows from (215) that for any policy π , one has

<!-- formula-not-decoded -->

These two facts taken collectively imply that the optimal policy and the optimal value function obey

<!-- formula-not-decoded -->

Next, we have learned from (213) that

<!-- formula-not-decoded -->

Note that 1 -(1 -γ ) V ⋆ θ (1) ≥ 1 -(1 -γ ) 1 1 -γ = 0 . Since the function

<!-- formula-not-decoded -->

is increasing in x and that x π,θ (cf. (212)) is increasing in π ( θ | 0) (given that p ≥ q ), one can easily see that the optimal policy obeys

<!-- formula-not-decoded -->

## C.2.4 Proof of auxiliary properties

Proof of claim (201) . We begin by proving the property (201). Towards this, let us abuse the notation by considering a MDP trajectory denoted by { ( s t , a t ) } t ≥ 0 , and suppose that it starts from s 0 ∼ ρ b = µ . It can be straightforwardly calculated that

<!-- formula-not-decoded -->

we can obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last identity holds since µ (0) + µ (1) = 1 . Similarly, one can derive P { s 1 = 1 } = µ (1) , thus indicating that s 1 ∼ µ . Repeating this analysis reveals that s t ∼ µ for any t ≥ 0 . Consequently, one has

<!-- formula-not-decoded -->

Additionally, it it observed that

<!-- formula-not-decoded -->

Proof of claim (203) . Consider the MDP M θ , whose optimal policy π ⋆ θ satisfies π ⋆ θ ( θ | 0) = 1 (see Lemma 11). Let us generate a MDP trajectory denoted by { ( s t , a t ) } t ≥ 0 with a t ∼ π ⋆ θ ( · | s t ) , where we have again abused notation as long as it is clear from the context. In this case, we can deduce that

<!-- formula-not-decoded -->

where in (i) we compute, for each t , the probability of a special trajectory with s 1 = · · · = s t = 0 and a 0 = · · · = a t -1 = θ , and (ii) holds true since P θ (0 | 0 , θ ) ≥ p ≥ γ . Taking this together with (220) yields

<!-- formula-not-decoded -->

In addition, it is easily seen that d ⋆ ( s, a ) = 0 for any s &gt; 1 , and that

<!-- formula-not-decoded -->

where the first inequality comes from (220), the first identity uses the definition (192), and the last two inequalities result from an immediate consequence of (193) and γ ≥ 1 / 2 , i.e.,

<!-- formula-not-decoded -->

As a result, putting the above relations together leads to

<!-- formula-not-decoded -->

Proof of inequality (204) . Observing the basic identity (using µ (0) + µ (1) = 1 )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last relation arises from the fact (200).

The remainder of the proof boils down to controlling α . Making use of the definition of p (cf. (195)), µ ( s ) (cf. (192)) and x π (cf. (199)), we can demonstrate that

<!-- formula-not-decoded -->

where (i) holds true owing to the trivial fact that x π,θ ≥ q for any policy π (as long as p ≥ q ), and (ii) is a consequence of the AM-GM inequality. Substituting it into (224) and using the definition (199) give

<!-- formula-not-decoded -->

## C.3 Proof of Theorem 4

To establish Theorem 4, we shall first generate a collection of hard problem instances (including MDPs and the associated batch datasets), and then conduct sample complexity analyses over these hard instances.

## C.3.1 Construction of hard problem instances

Construction of the hard MDPs. To begin with, for any integer H ≥ 32 , let us consider a set Θ ⊆ { 0 , 1 } H of H -dimensional vectors, which we shall construct shortly. We then generate a collection of MDPs

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

To define the transition kernel of these MDPs, we find it convenient to introduce the following state distribution supported on the state subset { 0 , 1 } :

<!-- formula-not-decoded -->

where 1 ( · ) is the indicator function, and C &gt; 0 is some constant that will determine the concentrability coefficient C ⋆ clipped (as we shall detail momentarily). It is assumed that

<!-- formula-not-decoded -->

With this distribution in mind, we can specify the transition kernel P θ = { P θ h h } H h =1 of the MDP M θ as follows:

<!-- formula-not-decoded -->

for any ( s, a, s ′ , h ) ∈ S × A × S × [ H ] , where p and q are set to be

<!-- formula-not-decoded -->

for c 1 = 1 / 4 and c 2 = 4096 such that

<!-- formula-not-decoded -->

It is readily seen from the above assumption that

<!-- formula-not-decoded -->

In view of the transition kernel (229), the MDP will never leave the state subset { 0 , 1 } if its initial state belongs to { 0 , 1 } . The reward function of all these MDPs is chosen to be

<!-- formula-not-decoded -->

Finally, let us choose the set Θ ⊆ { 0 , 1 } H . By virtue of the Gilbert-Varshamov lemma (Gilbert, 1952), one can construct Θ ⊆ { 0 , 1 } H in a way that

̸

for any ( s, a, h ) ∈ S × A × [ H ] .

<!-- formula-not-decoded -->

In other words, the set Θ we construct contains an exponentially large number of vectors that are sufficiently separated. This property plays an important role in the ensuing analysis.

Value functions and optimal policies. Next, we look at the value functions of the constructed MDPs and identify the optimal policies. For the sake of notational clarity, for the MDP M θ , we denote by π ⋆,θ = { π ⋆,θ h } H h =1 the optimal policy, and let V π,θ h (resp. V ⋆,θ h ) indicate the value function of policy π (resp. π ⋆,θ ) at time step h . The following lemma collects a couple of useful properties concerning the value functions and optimal policies; the proof can be found in Appendix C.3.3.

Lemma 12. Consider any θ ∈ Θ and any policy π . Then it holds that

<!-- formula-not-decoded -->

for any h ∈ [ H ] , where

<!-- formula-not-decoded -->

In addition, for any h ∈ [ H ] , the optimal policies and the optimal value functions obey

<!-- formula-not-decoded -->

provided that 0 &lt; c 1 ≤ 1 / 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Construction of the batch dataset. A batch dataset is then generated, which consists of K independent sample trajectories each of length H . The initial state distribution ρ b and the behavior policy π b = { π b h } H h =1 (according to (45)) are chosen as follows:

<!-- formula-not-decoded -->

where µ has been defined in (227). As it turns out, for any MDP M θ , the occupancy distributions of the above batch dataset admit the following simple characterization:

<!-- formula-not-decoded -->

Additionally, we shall choose the initial state distribution ρ as follows

<!-- formula-not-decoded -->

With this choice of ρ , the single-policy clipped concentrability coefficient C ⋆ clipped and the quantity C are intimately connected as follows:

<!-- formula-not-decoded -->

The proof of the claims (238) and (240) can be found in Appendix C.3.4.

## C.3.2 Establishing the minimax lower bound

We are now positioned to establish our sample complexity lower bounds. Recalling our choice of ρ in (239), our proof seeks to control the quantity

<!-- formula-not-decoded -->

where ̂ π is any policy estimator computed based on the batch dataset.

Step 1: converting ̂ π into an estimate ̂ θ of θ . Towards this, we first make the following claim: for an arbitrary policy π obeying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We shall postpone the proof of this claim to Appendix C.3.4. Suppose for the moment that there exists a policy estimate ̂ π that achieves

<!-- formula-not-decoded -->

then in view of (242), we necessarily have

<!-- formula-not-decoded -->

With the above observation in mind, we are motivated to construct the following estimate ̂ θ for θ ∈ Θ :

<!-- formula-not-decoded -->

one has

If ∑ h ∥ ∥ ̂ π ( · | 0) -π ⋆,θ ( · | 0) ∥ ∥ 1 &lt; H/ 8 holds for some θ ∈ Θ , then for any ˜ θ ∈ Θ with ˜ θ = θ one has

̸

<!-- formula-not-decoded -->

where the first inequality holds by the triangle inequality, the second line arises from the fact π ⋆,θ h ( θ h | 0) = 1 for all 1 ≤ h ≤ H (see (237)), and the last line comes from the properties (234) about Θ . Putting (245) and (246) together implies that ̂ θ = θ if

<!-- formula-not-decoded -->

is valid for all ˜ θ ∈ Θ with ˜ θ = θ . As a consequence,

̸

<!-- formula-not-decoded -->

In the sequel, we aim to demonstrate that (247) cannot possibly happen without enough samples, which would in turn contradict (243).

Step 2: probability of error in testing multiple hypotheses. Next, we turn attention to a | Θ | -ary hypothesis testing problem. For any θ ∈ Θ , denote by P θ the probability distribution when the MDP is M θ . We will then study the minimax probability of error defined as follows:

<!-- formula-not-decoded -->

̸

where the infimum is taken over all possible tests ψ (constructed based on the batch dataset available).

Let µ b ,θ (resp. µ b ,θ h h ( s h ) ) represent the distribution of a sample trajectory { s 1 , a 1 , s 2 , a 2 , · · · , s H , a H } (resp. a sample ( a h , s h +1 ) conditional on s h ) for the MDP M θ . Recalling that the K trajectories in the batch dataset are independently generated, one obtains

̸

<!-- formula-not-decoded -->

̸

where (i) arises from Fano's inequality (cf. (Tsybakov, 2009, Corollary 2.6)) and the additivity property of the KL divergence (cf. Tsybakov (2009, Page 85)), (ii) holds since | Θ | ≥ e H/ 8 (according to our construction (234)), and (iii) is valid when H ≥ 16 log 2 . Recalling that the occupancy state distribution d b h is the same for any MDP M θ with θ ∈ Θ (see (238)), one can invoke the chain rule of the KL divergence (Duchi, 2018, Lemma 5.2.8) and the Markovian nature of the sample trajectories to obtain

<!-- formula-not-decoded -->

̸

where the last identity holds true since (by construction and (238))

<!-- formula-not-decoded -->

Substitution into (249) yields

̸

<!-- formula-not-decoded -->

It then boils down to bounding the KL divergence terms in (250). If θ h = ˜ θ h , then it is self-evident that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Consider now the case that θ h = ˜ θ h , and suppose without loss of generality that θ h = 0 and ˜ θ h = 1 . It is seen that

<!-- formula-not-decoded -->

Given that p ≥ q ≥ 1 / 2 (see (232)), we can apply Lemma 10 to arrive at

<!-- formula-not-decoded -->

where (i) and (ii) make use of the definition (230) of ( p, q ) , and the last line follows as long as c 2 ε H 2 ≤ c 1 2 H ≤ 1 4 . Similarly, it can be easily verified that KL ( P θ h h (0 | 0 , 1) ∥ P ˜ θ h h (0 | 0 , 1) ) can be upper bounded in the same way. Substituting (252) and (251) back into (250) indicates that: if the sample size obeys

<!-- formula-not-decoded -->

then one necessarily has

<!-- formula-not-decoded -->

̸

Step 3: combining the above results. Suppose that there exists an estimator ̂ π satisfying

<!-- formula-not-decoded -->

where P θ denotes the probability when the MDP is M θ . Then in view of the analysis in Step 1, we must have

<!-- formula-not-decoded -->

and as a consequence of (247), the estimator ̂ θ defined in (245) must satisfy

<!-- formula-not-decoded -->

̸

Nevertheless, this cannot possibly happen under the sample size condition (253); otherwise it is contradictory to the result in (254). This concludes the proof by inserting c 1 = 1 / 4 and c 2 = 4096 .

## C.3.3 Proof of Lemma 12

To start with, for any policy π , it is observed that the value function of state s = 0 at step h is

<!-- formula-not-decoded -->

where (i) is valid due to the choice and (ii) holds since µ (0) + µ (1) = 1 .

<!-- formula-not-decoded -->

Additionally, the value function of state 1 at any step h obeys

<!-- formula-not-decoded -->

where (i) arises from the basic fact 0 ≤ V π,θ h ( s ) ≤ H -h +1 for any policy π and all ( s, h ) ∈ S × [ H ] , and (ii) holds since 2 c 1 HCS ( H -h ) ≤ 1 2 for c 1 small enough. The above results lead to several immediate facts.

- If we choose π such that π h (0 | 1) = 1 for all h ∈ [ H ] , then (259) tells us that

<!-- formula-not-decoded -->

A recursive application of this relation reveals that

<!-- formula-not-decoded -->

- For any policy π , applying (260) recursively tells us that

<!-- formula-not-decoded -->

The above two facts taken collectively imply that the optimal policy and optimal value function obey

<!-- formula-not-decoded -->

We then return to state 0 . By taking π such that π h ( θ h | 0) = 1 (and hence x π,θ h = p ) for all h ∈ [ H ] , one can invoke (257) to derive

<!-- formula-not-decoded -->

To see that why the last inequality holds, it suffices to observe that

<!-- formula-not-decoded -->

as long as c 1 ≤ 0 . 5 , which follows due to the elementary inequalities 1 -x ≤ exp( -x ) for any x ≥ 0 and exp( -x ) ≤ 1 -2 x/ 3 for any 0 ≤ x ≤ 1 / 2 . Combine (265) with (264) to reach

<!-- formula-not-decoded -->

Moreover, it follows from (257) that

<!-- formula-not-decoded -->

Observing that the function

<!-- formula-not-decoded -->

is increasing in x (as a result of (266)) and that x π,θ h is increasing in π h ( θ h | 0) (since p ≥ q ), we can readily conclude that the optimal policy in state 0 obeys

<!-- formula-not-decoded -->

## C.3.4 Proof of auxiliary properties

Throughout this section, we shall suppress the dependency on θ in the notation d ⋆ h whenever it is clear from the context.

Proof of claim (238) . For any MDP M θ , from the definition of d b h ( s, a ) in (46) and the Markov property, it is clearly seen that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recalling that d b 1 ( s ) = ρ b ( s ) = µ ( s ) for all s ∈ S , one can then show that where the last inequality holds since µ (1) + µ (0) = 1 . Similarly, it can be verified that d b 1 (1) = µ (1) , thereby implying that d b 2 = µ . Repeating this argument recursively for steps h = 2 , . . . , H confirms that

<!-- formula-not-decoded -->

This further allows one to demonstrate that

<!-- formula-not-decoded -->

Proof of claim (240) . Consider any MDP M θ , for which we have shown in Lemma 12 that π ⋆,θ h ( θ h | 0) = 1 for all h ∈ [ H ] . It is observed that

<!-- formula-not-decoded -->

where the last line makes use of the properties p ≥ 1 -c 1 /H , ρ (0) = 1 , and

<!-- formula-not-decoded -->

provided that 0 &lt; c 1 &lt; 1 / 2 . Combining this with (238), we arrive at

<!-- formula-not-decoded -->

where (i) arises from (238), (ii) relies on the definition in (227), and the final two inequalities come from the assumption in (228). Taking this together with the straightforward condition d ⋆ h ( s ) = 0 ( s &gt; 1 ) yields

<!-- formula-not-decoded -->

Proof of inequality (242) . By virtue of (236) and (237), we see that x π ⋆,θ ,θ h = p for all h ∈ [ H ] , which combined with (235) gives

<!-- formula-not-decoded -->

where (i) holds since V π,θ h +1 (1) ≤ V ⋆,θ h +1 (1) , (ii) follows from the fact that x π h ≥ q for any π and h ∈ [ H ] , and (iv) arises from the facts (237) and the choice (230) of ( p, q ) . To see why (iii) is valid, it suffices to note that µ (1) = 1 -1 CS ≥ 3 4 (as a consequence of (227) and (228)) and

<!-- formula-not-decoded -->

To continue, under the condition

<!-- formula-not-decoded -->

applying the relation in (275) recursively yields

<!-- formula-not-decoded -->

Here, (i) follows since

<!-- formula-not-decoded -->

holds as long as 0 &lt; c 1 ≤ 1 / 4 and c 2 ε/H ≤ c 1 . To see why (ii) is valid, we note that for any 0 ≤ x 1 , · · · , x H ≤ x max obeying ∑ H i =1 x i ≥ x sum , the following elementary inequality holds:

<!-- formula-not-decoded -->

this together with ∥ ∥ π ⋆,θ h ( · | 0) -π h ( · | 0) ∥ ∥ 1 ≤ 2 and (276) reveals that (by taking a h = h and x h = ∥ ∥ π ⋆,θ H +1 -h ( · | 0) -π H +1 -h ( · | 0) ∥ ∥ 1 )

<!-- formula-not-decoded -->

thus validating inequality (ii). As a result, we can continue the derivation to obtain

<!-- formula-not-decoded -->

provided that c 2 ≥ 4096 .

## References

- Agarwal, A., Jiang, N., Kakade, S. M., and Sun, W. (2021). Reinforcement learning: Theory and algorithms. Technical report .
- Agarwal, A., Kakade, S., and Yang, L. F. (2020). Model-based reinforcement learning with a generative model is minimax optimal. Conference on Learning Theory , pages 67-83.
- Agarwal, R. P., Meehan, M., and O'regan, D. (2001). Fixed point theory and applications , volume 141. Cambridge university press.
- Auer, P. and Ortner, R. (2006). Logarithmic online regret bounds for undiscounted reinforcement learning. Advances in neural information processing systems , 19.
- Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- Azar, M. G., Osband, I., and Munos, R. (2017). Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272. PMLR.
- Bai, Y., Xie, T., Jiang, N., and Wang, Y.-X. (2019). Provably efficient Q-learning with low switching cost. In Advances in Neural Information Processing Systems , pages 8002-8011.
- Beck, C. L. and Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems &amp; control letters , 61(12):1203-1208.
- Bertsekas, D. P. (2017). Dynamic programming and optimal control (4th edition) . Athena Scientific.
- Bourel, H., Maillard, O., and Talebi, M. S. (2020). Tightening exploration in upper confidence reinforcement learning. In International Conference on Machine Learning , pages 1056-1066. PMLR.
- Buckman, J., Gelada, C., and Bellemare, M. G. (2020). The importance of pessimism in fixed-dataset policy optimization. In International Conference on Learning Representations .
- Chen, J. and Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051. PMLR.
- Chen, M., Li, Y., Wang, E., Yang, Z., Wang, Z., and Zhao, T. (2021a). Pessimism meets invariance: Provably efficient offline mean-field multi-agent RL. Advances in Neural Information Processing Systems , 34.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-sample analysis of contractive stochastic approximation using smooth convex envelopes. Advances in Neural Information Processing Systems , 33:8223-8234.
- Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2021b). A Lyapunov theory for finite-sample guarantees of asynchronous Q-learning and TD-learning variants. arXiv preprint arXiv:2102.01567 .

- Cui, Q. and Du, S. S. (2022). When is offline two-player zero-sum Markov game solvable? arXiv preprint arXiv:2201.03522 .
- Cui, Q. and Yang, L. F. (2021). Minimax sample complexity for turn-based stochastic game. In Uncertainty in Artificial Intelligence , pages 1496-1504. PMLR.
- Diehl, C., Sievernich, T., Krüger, M., Hoffmann, F., and Bertran, T. (2021). Umbrella: Uncertainty-aware model-based offline reinforcement learning leveraging planning. arXiv preprint arXiv:2111.11097 .
- Donoho, D. L. et al. (2000). High-dimensional data analysis: The curses and blessings of dimensionality. AMS math challenges lecture , 1(2000):32.
- Du, S. S., Chen, J., Li, L., Xiao, L., and Zhou, D. (2017). Stochastic variance reduction methods for policy evaluation. In International Conference on Machine Learning , pages 1049-1058. PMLR.
- Duan, Y., Jia, Z., and Wang, M. (2020). Minimax-optimal off-policy evaluation with linear function approximation. In International Conference on Machine Learning , pages 2701-2709. PMLR.
- Duan, Y., Wang, M., and Wainwright, M. J. (2021). Optimal policy evaluation using kernel-based temporal difference methods. arXiv preprint arXiv:2109.12002 .
- Duchi, J. C. (2018). Introductory lectures on stochastic optimization. The mathematics of data , 25:99-186.
- Ebert, F., Finn, C., Dasari, S., Xie, A., Lee, A., and Levine, S. (2018). Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. arXiv preprint arXiv:1812.00568 .
- Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of machine learning Research , 5(Dec):1-25.
- Fan, J., Wang, Z., Xie, Y., and Yang, Z. (2019). A theoretical analysis of deep Q-learning. arXiv e-prints , pages arXiv-1901.
- Farahmand, A.-m., Szepesvári, C., and Munos, R. (2010). Error propagation for approximate policy and value iteration. Advances in Neural Information Processing Systems , 23.
- Filippi, S., Cappé, O., and Garivier, A. (2010). Optimism in reinforcement learning and kullback-leibler divergence. In 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton) , pages 115-122. IEEE.
- Fruit, R., Pirotta, M., and Lazaric, A. (2020). Improved analysis of UCRL2 with empirical Bernstein inequality. arXiv preprint arXiv:2007.05456 .
- Gilbert, E. N. (1952). A comparison of signalling alphabets. The Bell system technical journal , 31(3):504-522.
- He, J., Zhou, D., and Gu, Q. (2021). Nearly minimax optimal reinforcement learning for discounted MDPs. Advances in Neural Information Processing Systems , 34:22288-22300.
- Jaksch, T., Ortner, R., and Auer, P. (2010). Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11:1563-1600.
- Jiang, N. and Huang, J. (2020). Minimax value interval for off-policy evaluation and policy optimization. Advances in Neural Information Processing Systems , 33:2747-2758.
- Jiang, N. and Li, L. (2016). Doubly robust off-policy value evaluation for reinforcement learning. In International Conference on Machine Learning , pages 652-661. PMLR.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018). Is Q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2020). Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 2137-2143. PMLR.

- Jin, Y., Yang, Z., and Wang, Z. (2021). Is pessimism provably efficient for offline RL? In International Conference on Machine Learning , pages 5084-5096.
- Johnson, R. and Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance reduction. In Advances in neural information processing systems , pages 315-323.
- Kallus, N. and Uehara, M. (2020). Double reinforcement learning for efficient off-policy evaluation in markov decision processes. Journal of Machine Learning Research , 21(167):1-63.
- Kidambi, R., Rajeswaran, A., Netrapalli, P., and Joachims, T. (2020). MOReL: Model-based offline reinforcement learning. Advances in neural information processing systems , 33:21810-21823.
- Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020). Conservative Q-learning for offline reinforcement learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H., editors, Advances in Neural Information Processing Systems , volume 33, pages 1179-1191. Curran Associates, Inc.
- Lai, T. L. and Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics , 6(1):4-22.
- Lange, S., Gabel, T., and Riedmiller, M. (2012). Batch reinforcement learning. In Reinforcement learning , pages 45-73. Springer.
- Lattimore, T. and Szepesvári, C. (2020). Bandit algorithms . Cambridge University Press.
- Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 .
- Li, G., Cai, C., Chen, Y., Wei, Y., and Chi, Y. (2024a). Is Q-learning minimax optimal? a tight sample complexity analysis. Operations Research , 72(1):203-221.
- Li, G., Shi, L., Chen, Y., Gu, Y., and Chi, Y. (2021). Breaking the sample complexity barrier to regretoptimal model-free reinforcement learning. Advances in Neural Information Processing Systems , 34.
- Li, G., Wei, Y., Chi, Y., and Chen, Y. (2024b). Breaking the sample size barrier in model-based reinforcement learning with a generative model. Operations Research , 72(1):222-236.
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020). Breaking the sample size barrier in model-based reinforcement learning with a generative model. In Advances in Neural Information Processing Systems , volume 33.
- Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2022). Sample complexity of asynchronous Q-learning: Sharper analysis and variance reduction. IEEE Transactions on Information Theory , 68(1):448-473.
- Li, G., Yan, Y., Chen, Y., and Fan, J. (2023). Minimax-optimal reward-agnostic exploration in reinforcement learning. arXiv preprint arXiv:2304.07278 .
- Li, G., Zhan, W., Lee, J. D., Chi, Y., and Chen, Y. (2024c). Reward-agnostic fine-tuning: Provable statistical benefits of hybrid reinforcement learning. Advances in Neural Information Processing Systems , 36.
- Li, L., Munos, R., and Szepesvári, C. (2014). On minimax optimal offline policy evaluation. arXiv preprint arXiv:1409.3653 .
- Liu, Y., Swaminathan, A., Agarwal, A., and Brunskill, E. (2020). Provably good batch reinforcement learning without great exploration. arXiv preprint arXiv:2007.08202 .
- Munos, R. (2007). Performance bounds in l p -norm for approximate value iteration. SIAM journal on control and optimization , 46(2):541-561.
- Murphy, S. (2005). A generalization error for Q-learning. Journal of Machine Learning Research , 6:10731097.

- Nguyen-Tang, T., Gupta, S., and Venkatesh, S. (2021). Sample complexity of offline reinforcement learning with deep ReLU networks. arXiv preprint arXiv:2103.06671 .
- Panaganti, K. and Kalathil, D. (2022). Sample complexity of robust reinforcement learning with a generative model. In International Conference on Artificial Intelligence and Statistics , pages 9582-9602. PMLR.
- Pananjady, A. and Wainwright, M. J. (2020). Instance-dependent ℓ ∞ -bounds for policy evaluation in tabular reinforcement learning. IEEE Transactions on Information Theory , 67(1):566-585.
- Prudencio, R. F., Maximo, M. R., and Colombini, E. L. (2022). A survey on offline reinforcement learning: Taxonomy, review, and open problems. arXiv preprint arXiv:2203.01387 .
- Qian, J., Fruit, R., Pirotta, M., and Lazaric, A. (2019). Exploration bonus for regret minimization in discrete and continuous average reward MDPs. Advances in Neural Information Processing Systems , 32.
- Qu, G. and Wierman, A. (2020). Finite-time analysis of asynchronous stochastic approximation and Qlearning. Conference on Learning Theory , pages 3185-3205.
- Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., and Russell, S. (2022). Bridging offline reinforcement learning and imitation learning: A tale of pessimism. IEEE Transactions on Information Theory , 68(12):8156-8196.
- Ren, T., Li, J., Dai, B., Du, S. S., and Sanghavi, S. (2021). Nearly horizon-free offline reinforcement learning. Advances in neural information processing systems , 34.
- Robbins, H. and Monro, S. (1951). A stochastic approximation method. The annals of mathematical statistics , pages 400-407.
- Shi, C., Luo, S., Zhu, H., and Song, R. (2022a). Statistically efficient advantage learning for offline reinforcement learning in infinite horizons. arXiv preprint arXiv:2202.13163 .
- Shi, L. and Chi, Y. (2022). Distributionally robust model-based offline reinforcement learning with nearoptimal sample complexity. arXiv preprint arXiv:2208.05767 .
- Shi, L., Li, G., Wei, Y., Chen, Y., and Chi, Y. (2022b). Pessimistic Q-learning for offline reinforcement learning: Towards optimal sample complexity. International Conference on Machine Learning , pages 19967-20025.
- Sidford, A., Wang, M., Wu, X., Yang, L., and Ye, Y. (2018). Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of Go without human knowledge. Nature , 550(7676):354359.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- Szepesvári, C. (1998). The asymptotic convergence-rate of Q-learning. In Advances in Neural Information Processing Systems , pages 1064-1070.
- Talebi, M. S. and Maillard, O.-A. (2018). Variance-aware regret bounds for undiscounted reinforcement learning in mdps. In Algorithmic Learning Theory , pages 770-805. PMLR.
- Tang, S. and Wiens, J. (2021). Model selection for offline reinforcement learning: Practical considerations for healthcare settings. In Machine Learning for Healthcare Conference , pages 2-35. PMLR.
- Thomas, P. and Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning , pages 2139-2148. PMLR.
- Tsybakov, A. B. (2009). Introduction to nonparametric estimation , volume 11. Springer.

- Uehara, M., Huang, J., and Jiang, N. (2020). Minimax weight and Q-function learning for off-policy evaluation. In International Conference on Machine Learning , pages 9659-9668. PMLR.
- Uehara, M. and Sun, W. (2021). Pessimistic model-based offline reinforcement learning under partial coverage. arXiv preprint arXiv:2107.06226 .
- Uehara, M., Zhang, X., and Sun, W. (2022). Representation learning for online and offline RL in low-rank MDPs. In International Conference on Learning Representations .
- Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press.
- Wainwright, M. (2019a). High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press.
- Wainwright, M. J. (2019b). Stochastic approximation with cone-contractive operators: Sharp ℓ ∞ -bounds for Q-learning. arXiv preprint arXiv:1905.06265 .
- Wainwright, M. J. (2019c). Variance-reduced Q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 .
- Wang, Y., Dong, K., Chen, X., and Wang, L. (2019). Q-learning with UCB exploration is sample efficient for infinite-horizon MDP. In International Conference on Learning Representations .
- Watkins, C. J. and Dayan, P. (1992). Q-learning. Machine learning , 8(3-4):279-292.
- Xie, T. and Jiang, N. (2021). Batch value-function approximation with only realizability. In International Conference on Machine Learning , pages 11404-11413. PMLR.
- Xie, T., Jiang, N., Wang, H., Xiong, C., and Bai, Y. (2021). Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. arXiv preprint arXiv:2106.04895 .
- Xiong, H., Zhao, L., Liang, Y., and Zhang, W. (2020). Finite-time analysis for double Q-learning. Advances in Neural Information Processing Systems , 33.
- Xu, T., Yang, Z., Wang, Z., and Liang, Y. (2021). A unified off-policy evaluation approach for general value function. arXiv preprint arXiv:2107.02711 .
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2022). Model-based reinforcement learning is minimax-optimal for offline zero-sum Markov games. arXiv preprint arXiv:2206.04044 .
- Yan, Y., Li, G., Chen, Y., and Fan, J. (2023). The efficacy of pessimism in asynchronous Q-learning. IEEE Transactions on Information Theory , 69(11):7185-7219.
- Yang, M., Nachum, O., Dai, B., Li, L., and Schuurmans, D. (2020). Off-policy evaluation via the regularized Lagrangian. Advances in Neural Information Processing Systems , 33:6551-6561.
- Yin, M., Duan, Y., Wang, M., and Wang, Y.-X. (2022). Near-optimal offline reinforcement learning with linear representation: Leveraging variance information with pessimism. In International Conference on Learning Representations .
- Yin, M. and Wang, Y.-X. (2021). Towards instance-optimal offline reinforcement learning with pessimism. Advances in neural information processing systems , 34.
- Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., Finn, C., and Ma, T. (2020). MOPO: Modelbased offline policy optimization. Advances in Neural Information Processing Systems , 33:14129-14142.
- Zanette, A., Wainwright, M. J., and Brunskill, E. (2021). Provable benefits of actor-critic methods for offline reinforcement learning. Advances in neural information processing systems , 34.
- Zhan, W., Huang, B., Huang, A., Jiang, N., and Lee, J. D. (2022). Offline reinforcement learning with realizability and single-policy concentrability. arXiv preprint arXiv:2202.04634 .

- Zhang, Z., Chen, Y., Lee, J. D., and Du, S. S. (2023). Settling the sample complexity of online reinforcement learning. arXiv preprint arXiv:2307.13586 .
- Zhang, Z., Ji, X., and Du, S. (2021a). Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. In Conference on Learning Theory , pages 4528-4531. PMLR.
- Zhang, Z., Zhou, Y., and Ji, X. (2020). Almost optimal model-free reinforcement learning via referenceadvantage decomposition. Advances in Neural Information Processing Systems , 33.
- Zhang, Z., Zhou, Y., and Ji, X. (2021b). Model-free reinforcement learning: from clipped pseudo-regret to sample complexity. In International Conference on Machine Learning , pages 12653-12662. PMLR.
- Zhong, H., Xiong, W., Tan, J., Wang, L., Zhang, T., Wang, Z., and Yang, Z. (2022). Pessimistic minimax value iteration: Provably efficient equilibrium learning from offline datasets. arXiv preprint arXiv:2202.07511 .
- Zhou, Z., Zhou, Z., Bai, Q., Qiu, L., Blanchet, J., and Glynn, P. (2021). Finite-sample regret bound for distributionally robust offline tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 3331-3339. PMLR.