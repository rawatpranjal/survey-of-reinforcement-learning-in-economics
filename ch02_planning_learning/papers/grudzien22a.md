TRL

Mirror Learning Space

MDPO

PPO

TRPO

GPI

## Mirror Learning: A Unifying Framework of Policy Optimisation

¡aura 1 Known DI fromarole ond olanrithme ne nointe in

## Jakub Grudzien Kuba 1 Christian Schroeder de Witt 1 Jakob Foerster 1

## Abstract

Modern deep reinforcement learning (RL) algorithms are motivated by either the generalised policy iteration (GPI) or trust-region learning (TRL) frameworks. However, algorithms that strictly respect these theoretical frameworks have proven unscalable. Surprisingly, the only known scalable algorithms violate the GPI/TRL assumptions, e.g. due to required regularisation or other heuristics. The current explanation of their empirical success is essentially 'by analogy': they are deemed approximate adaptations of theoretically sound methods. Unfortunately, studies have shown that in practice these algorithms differ greatly from their conceptual ancestors. In contrast, in this paper we introduce a novel theoretical framework, named Mirror Learning , which provides theoretical guarantees to a large class of algorithms, including TRPO and PPO. While the latter two exploit the flexibility of our framework, GPI and TRL fit in merely as pathologically restrictive corner cases thereof. This suggests that the empirical performance of state-of-the-art methods is a direct consequence of their theoretical properties, rather than of aforementioned approximate analogies. Mirror learning sets us free to boldly explore novel, theoretically sound RL algorithms, a thus far uncharted wonderland.

## 1. Introduction

The generalised policy iteration (Sutton &amp; Barto, 2018, GPI) and trust-region learning (Schulman et al., 2015, TRL) frameworks lay the foundations for the design of the most commonly used reinforcement learning (RL) algorithms. At each GPI iteration, an RL agent first evaluates its policy by computing a scalar value function , and then updates its policy so as to maximise the value function

1 University of Oxford. Correspondence to: Jakub Grudzien Kuba &lt; jakub.grudzien@new.ox.ac.uk &gt; .

Proceedings of the 39 th International Conference on Machine Learning , Baltimore, Maryland, USA, PMLR 162, 2022. Copyright 2022 by the author(s).

at every environment state. This procedure serves well for Markov Decision Problems (Bellman, 1957, MDPs) with small state spaces, as it is guaranteed to produce a new policy whose value at every state improves monotonically over the old one. However, the sizes of state spaces that many practical problem settings constitute are intractable to exact implementations of GPI. Instead, large scale settings employ function approximation and sample based learning. Unfortunately, such an adoption of GPI has proven unstable (Mnih et al., 2015), which has necessitated a number of adjustments, such as replay buffers, target networks, etc., to stabilise learning (Van Hasselt et al., 2016). In their days, these heuristics have been empirically sucessful but were not backed up by any theory and required extensive hyperparameter tuning (Mnih et al., 2016).

Figure 1. Known RL frameworks and algorithms as points in the infinite space of theoretically sound mirror learning algorithms.

<!-- image -->

Instead, TRL improves the robustness of deep RL methods by optimising a surrogate objective, which restricts the policy update size at each learning iteration while preserving monotonic improvement guarantees (Schulman et al., 2015). To this end it introduces a notion of distance between policies, e.g. through evaluating the maximal KLdivergence. Unfortunately, these types of measures do not scale to large MDPs that TRL was meant to tackle in the first place (since small problems can be solved by GPI).

Nevertheless, TRL's theoretical promises and intuitive accessibility inspired a number of algorithms based on heuristic approximations thereof. This paradigm shift led to substantial empirical success: First, in Trust-Region Policy Optimization (Schulman et al., 2015, TRPO), a hard constraint mechanism on the policy update size was introduced. Second, Proximal Policy Optimization (Schulman et al., 2017, PPO) replaced the hard constraint with the clipping objec-

АЗС

tive which, intuitively, disincentivises large policy updates. Since they were published, these algorithms have both been applied widely, resulting in state-of-the-art (SOTA) performance on a variety of benchmark, and even real-world, tasks (Schulman et al., 2015; 2017; Berner et al., 2019).

There is a stark contrast between the empirical success and the lack of theoretical understanding, which is largely limited to 'proof by analogy': For example, PPO is regarded as sound since it approximates an algorithm compliant with TRL. However, recent studies concluded that PPO arbitrarily exceeds policy update size constraints, thus fundamentally violating TRL principles (Wang et al., 2020; Engstrom et al., 2020), which shows that PPO's empirical success does not follow from TRL. On a higher level, this reveals a concerning lack of theoretical understanding of the perhaps most widely used RL algorithm.

To reconnect theory with practice, in this paper we introduce a novel theoretical framework, named Mirror Learning , which provides theoretical guarantees to a large class of known algorithms, including TRPO (Schulman et al., 2015), PPO (Schulman et al., 2017), and MDPO (Tomar et al., 2020), to name a few, as well as to myriads of algorithms that are yet to be discovered. Mirror learning is a general, principled policy-learning framework that possesses monotonic improvement and optimal-policy convergence guarantees, under arbitrary update-size constraints. Intuitively, a mirror-learning policy update maximises the current value function while keeping the, arbitrarily specified, update cost (called drift ) small. The update is further constrained within a neighbourhood of the current policy, that can also be specified arbitrarily.

Since TRL and GPI were the only anchor points for theoretically sound policy optimisation, prior unsuccessful attempts to proving the soundness of PPO and other algorithms tried to shoe-horn them into the overly narrow confines of these frameworks (Liu et al., 2019; Wang et al., 2020; Queeney et al., 2021). Mirror learning shows that this was a flawed approach: Rather than shoe-horning the algorithms, we radically expand the space of theoretically sound methods, naturally covering PPO et al in their current, practical formulations. Our work suggests that the empirical performance of these state-of-the-art methods is a direct consequence of their theoretical properties, rather than of aforementioned approximate analogies. Rather than being limited by the confines of two singular anchor points (GPI/TRL), mirror learning sets us free to explore an endless space of possible algorithms, each of which is already endowed with theoretical guarantees.

We also illustrate the explanatory power of mirror learning and use it to explain a number of thus far unexplained observations concerning the performance of other algorithms in the literature.

Finally, we show that mirror learning allows us to view policy optimisation as a search on a directed graph, where the total path-weight between any two nodes in an optimisation path is upper bounded by the optimality gap between them, which we confirm experimentally. We also analyse proof-of-concept instantiations of mirror learning that use a variety of different neighbourhood and drift functions.

## 2. Background

In this section, we introduce the RL problem formulation and briefly survey state-of-the-art learning protocols.

## 2.1. Preliminaries

We consider a Markov decision process (MDP) defined by a tuple x S , A , r, P, γ, d y . Here, S is a discrete state space, A is a discrete action space 1 , r : S ˆ A Ñr´ R max , R max s is the bounded reward function, P : S ˆ A ˆ S Ñ r 0 , 1 s is the probabilistic transition function, γ P r 0 , 1 q is the discount factor, and d P P p S q (here P p X q denotes the set of probability distributions over a set X ) is the initial state distribution. At time step t P N , the agent is at a state s t , takes an action a t according to its stationary policy π p¨| s t q P P p A q , receives a reward r p s t , a t q , and moves to the state s t ` 1 , whose probability distribution is P p¨| s t , a t q . The whole experience is evaluated by the return , which is the discounted sum of all rewards

<!-- formula-not-decoded -->

The state-action and state value functions that evaluate the quality of states and actions, by providing a proxy to the expected return, are given by

<!-- formula-not-decoded -->

respectively. The advantage function , defined as

<!-- formula-not-decoded -->

estimates the advantage of selecting one action over another. The goal of the agent is to learn a policy that maximises the expected return , defined as a function of π ,

<!-- formula-not-decoded -->

Here, ρ π p s q fi 8 ř t ' 0 γ t Pr p s t ' s | π q is the (improper) marginal state distribution. Interestingly, the set of solutions to this problem always contains the optimal policies -policies π ˚ , for which Q π ˚ p s, a q fi Q ˚ p s, a q ě Q π p s, a q holds for any π P Π fi Ś s P S P p A q , s P S , a P

1 Our results extend to any compact state and action spaces. However, we work in the discrete setting in this paper for clarity.

A . Furthermore, the optimal policies are the only ones that satisfy the optimality equation (Sutton &amp; Barto, 2018)

<!-- formula-not-decoded -->

In the next subsections, we survey the most fundamental and popular approaches of finding these optimal policies.

## 2.2. Generalised Policy Iteration

One key advantage of the generalised policy iteration (GPI) framework is its simplicity. Even though the policy influences both the rewards and state visitations, GPI guarantees that simply reacting greedily to the value function,

<!-- formula-not-decoded -->

guarantees that the new policy obtains higher expected returns at every state, i.e., V π new p s q ě V π old p s q , @ s P S . Moreover, this procedure converges to the set of optimal policies (Sutton &amp; Barto, 2018), which can be seen intuitively by substituting a fixed-point policy into Equation (1).

In settings with small, discrete action spaces GPI can be executed approximately without storing the policy variable π by responding greedily to the state-action value function. This gives rise to value-based learning (Sutton &amp; Barto, 2018), where the agent learns a Q-function with the Bellman-max update

<!-- formula-not-decoded -->

which is known to converge to Q ˚ (Watkins &amp; Dayan, 1992), and has inspired design of a number of methods (Mnih et al., 2015; Van Hasselt et al., 2016).

Another, approximate, implementation of GPI is through the policy gradient (PG) algorithms (Sutton et al., 2000). These are methods which optimise the policy π θ by parameterising it with θ , and updating the parameter in the direction of the gradient of the expectd return, given by

<!-- formula-not-decoded -->

which is the gradient of the optimisation objective of GPI from Equation (1) weighted by ρ π θ old . An analogous result holds for (continuous) deterministic policies (Silver et al., 2014; Lillicrap et al., 2015). Thus, PG based algorithms approximately solve the GPI step, with a policy in the neighbourhood of π θ old , provided the step-size α ą 0 in the update θ new ' θ ` α ∇ θ η p π θ q| θ ' θ old is sufficiently small. PG methods have played a major role in applications of RL to the real-world settings, and especially those that involve continuous actions, where some value-based algorithms, like DQN, are intractable (Williams, 1992; Baxter &amp;Bartlett, 2001; Mnih et al., 2016).

## 2.3. Trust-Region Learning

In practice, policy gradient methods may suffer from the high variance of the PG estimates and training instability (Kakade &amp; Langford, 2002; Zhao et al., 2011). Trustregion learning (TRL) is a framework that aims to solve these issues. At its core lies the following policy update

<!-- formula-not-decoded -->

It was shown by Schulman et al. (2015) that this update guarantees the monotonic improvement of the return, i.e., η p π new q ě η p π old q . Furthermore, the KL-penalty in the above objective ensures that the new policy stays within the neighbourhood of π old, referred to as the trust region . This is particularly important with regard to the instability issue of the PG-based algorithms (Duan et al., 2016).

Although the exact calculation of the max-KL penalty is intractable in settings with large/continuous state spaces, the algorithm can be heuristically approximated, e.g. through Trust Region Policy Optimization (Schulman et al., 2015, TRPO), which performs a constrained optimisation update

<!-- formula-not-decoded -->

where δ ą 0 . Despite its deviation from the original theory, empirical results suggest that TRPO approximately maintains the TRL properties. However, in order to achieve a simpler TRL-based heuristic, Schulman et al. (2017) introduced Proximal Policy Optimization (PPO) which updates the policy by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where for given s, a , r p ¯ π q fi ¯ π p a | s q{ π p a | s q . The clip operator truncates r p ¯ π q to 1 ´ ϵ (or 1 ` ϵ ), if it is below (or above) the threshold interval. Despite being motivated from a TRL perspective, PPO violates its very core principles by failing to constrain the update size-whether measured either by KL divergence, or by the likelihood ratios (Wang et al., 2020; Engstrom et al., 2020). Nevertheless, PPO's ability to stabilise policy training has been demonstrated on a wide variety of tasks, frequently resulting in SOTA performance (Henderson et al., 2018; Berner et al., 2019). This begs the question how PPO's empirical performance can be justified theoretically. Mirror learning, which we introduce in the next section, addresses this issue.

## 3. Mirror Learning

In this section we introduce Mirror Learning , state its theoretical properties, and explore its connection to prior RL theory and methodology.

## 3.1. The Framework

We start from the following definition.

Definition 3.1. A drift functional D is a map

<!-- formula-not-decoded -->

such that for all s P S , and π, ¯ π P Π , writing D π p ¯ π | s q for D π ` ¯ π p¨| s q| s ˘ , the following conditions are met

1. D π p ¯ π | s q ě D π p π | s q ' 0 (nonnegativity) ,
2. D π p ¯ π | s q has zero gradient 2 with respect to ¯ π p¨| s q , evaluated at ¯ π p¨| s q ' π p¨| s q (zero gradient) .

Let ν ¯ π π P P p S q be a state distribution that can depend on π and ¯ π . The drift D ν of ¯ π from π is defined as

<!-- formula-not-decoded -->

ν π ¯ π is required to be such that the above expectation is continuous in π and ¯ π . The drift is positive if D ν π p ¯ π q ' 0 implies ¯ π ' π , and trivial if D ν π p ¯ π q ' 0 , @ π, ¯ π P Π .

We would like to highlight that drift is not the Bregman distance, associated with mirror descent (Nemirovskij &amp; Yudin, 1983; Beck &amp; Teboulle, 2003), as we do not require it to be (strongly) convex, or differentiable everywhere. All we require are, the much milder, continuity and Gˆ ateaux-differentiability (Gateaux, 1922; Hamilton &amp; Nashed, 1982) of the drift functional at ¯ π p¨| s q ' π p¨| s q .

We introduce one more concept whose role is to generally account for explicit update-size constraints. Such constraints reflect a learner's risk-aversity towards subtantial changes to its behaviour, like in TRPO, or are dictated by an algorithm design, like the learning rate in PG methods.

Definition 3.2. We say that N : Π Ñ P p Π q is a neighbourhood operator , where P p Π q is the power set of Π , if

1. It is a continuous map (continuity) ,
2. Every N p π q is a compact set (compactness) ,
3. There exists a metric χ : Π ˆ Π Ñ R , such that @ π P Π , there exists ζ ą 0 , such that χ p π, ¯ π q ď ζ implies ¯ π P N p π q (closed ball) .

The trivial neighbourhood operator is N ' Π .

Let D ν be a drift, and π, ¯ π be policies. Suppose that β π P P p S q is a state distribution, referred to as a sampling distribution , such that β π p s q ą 0 , @ s P S . Then, the mirror operator transforms the value function of π into the following functional of ¯ π ,

<!-- formula-not-decoded -->

2 More precisely, all its Gˆ ateaux derivatives are zero.

Note that, in the above definition, Q π can be equivalently replaced by A π , as this only subtracts a constant V π p s q independent of ¯ π . We will use this fact later. As it turns out, despite the appearance of the drift penalty, simply acting to increase the mirror operator suffices to guarantee the policy improvement, as summarised by the following lemma, proved in Appendix B.

Lemma 3.3. Let π old and π new be policies. Suppose that

<!-- formula-not-decoded -->

Then, π new is better than π old, so that for every state s ,

<!-- formula-not-decoded -->

Of course, the monotonic improvement property of the expected return is a natural corollary of this lemma, as

<!-- formula-not-decoded -->

Hence, optimisation of the mirror operator guarantees the improvement of the new policy's performance. Condition (4), however, is represented by | S | inequalities and solving them may be intractable in large state spaces. Hence, we shall design a proxy objective whose solution simply satisfies Condition (4), and admits Monte-Carlo estimation (for practical reasons). For this purpose, we define the update rule of mirror learning.

Definition 3.4. Let π old be a policy. Then, mirror learning updates the policy by

<!-- formula-not-decoded -->

This, much simpler and clearer, problem formulation, is also robust to settings with large state spaces and parameterised policies. Being an expectation, it can be approximated by sampling with the unbiased 'batch' estimator

<!-- formula-not-decoded -->

The distribution ν ¯ π π old can be chosen equal to β π old , in which case the fraction from the front of the drift disappears, simplifying the estimator. Note also that one may choose β π old independently of π old . For example, it can be some fixed distribution over S , like the uniform distribution. Hence, the above update supports both on-policy and off-policy learning. In the latter case, the fraction ¯ π p a | s q{ π old p a | s q is replaced by ¯ π p a | s q{ π hist p a | s q , where π hist is the policy used to sample the pair p s , a q and insert it to a replay buffer (for more details see Appendix F). Such a Monte-Carlo objective can be optimised with respect to ¯ π parameters by, for example, a few steps of gradient ascent. Most importantly, in its exact form, the update in Equation (5) guarantees that the resulting policy satisfies the desired Condition (4).

Lemma3.5. Let π old be a policy and π new be obtained from the mirror update of Equation (5). Then,

<!-- formula-not-decoded -->

Hence, π new attains the properties provided by Lemma 3.3.

For proof, see Appendix B. Next, we use the above results to characterise the properties of mirror learning. The improvement and convegence properties that it exhibits not only motivate its usage, but may also contribute to explaining the empirical success of various widely-used algorithms. We provide a proof sketch below, and a detailed proof in Appendix C.

Theorem 3.6 (The Fundamental Theorem of Mirror Learning) . Let D ν be a drift, N be a neighbourhood operator, and the sampling distribution β π depend continuously on π . Let π 0 P Π , and the sequence of policies p π n q 8 n ' 0 be obtained by mirror learning induced by D ν , N , and β π . Then, the learned policies

1. Attain the strict monotonic improvement property,

<!-- formula-not-decoded -->

2. Their value functions converge to the optimal one,

<!-- formula-not-decoded -->

3. Their expected returns converge to the optimal return,

<!-- formula-not-decoded -->

4. Their ω -limit set consists of the optimal policies.

Proof sketch. We split the proof into four steps. In Step 1 , we use Lemmas 3.3 &amp; 3.5 to show that the sequence of value functions p V π k q k P N converges. In Step 2 , we show the existance of limit points ¯ π of p π k q k P N , and show that they are fixed points of the mirror learning update, which we do by contradiction. The most challenging Step 3 , is where we prove (by contradiction) that ¯ π is a fixed point of GPI. In the proof, we supposed that ¯ π is not a fixed point of GPI, and use the drift's zero-gradient property (Definition 3.1) to show that one could slightly perturb ¯ π to obtain a policy π 1 , which corresponds to a higher mirror learning objective. This contradicts ¯ π simultaneously being a fixed point of mirror learning. Step 4 finalises the proof, by recalling that fixed points of GPI are optimal policies.

Hence, any algorithm whose update takes the form of Equation (5) improves the return monotonically, and converges to the set of optimal policies. This result provides

RL algorithm designers with a template. New instances of it can be obtained by altering the drift D ν , the neighbourhood operator N , and the sampling distribution function β π . Indeed, in the next section, we show how some of the most well-known RL algorithms fit this template.

## 4. Mirror Learning View of RL Phenomena

We provide a list of RL algorithms in their mirror learning representation in Appendix E.

Generalised Policy Iteration For the trivial drift D ν ' 0 , and the trivial neighbourhood operator N ' Π , the mirror learning update from Equation (5) is equivalent to

<!-- formula-not-decoded -->

As in this case the maximisation is unconstrained, and the expectation over s ' β π old is monotonically increasing in the individual conditional E a ' ¯ π ' Q π old p s, a q ‰ , the maximisation distributes across states

<!-- formula-not-decoded -->

which is exactly the GPI update, as in Equation (1). Hence, all instances of GPI, e.g., policy iteration (Sutton &amp; Barto, 2018), are special cases of mirror learning, and thus inherit its improvement and convergence properties. Furthermore, neural implementations, like A3C (Mnih et al., 2016), maintain these qualities approximately.

Trust-Region Learning Let us choose the drift operator D to be the scaled KL-divergence, so that D π p ¯ π | s q ' C π DKL ` π p¨| s q , ¯ π p¨| s q ˘ , where C π is a constant that depends on π . Further, in the construction of the drift D ν π p ¯ π q , let us set ν ¯ π π p s q ' δ p s ´ s max q , where δ is the Diracdelta distribution, and s max is the state at which the KLdivergence between π and ¯ π is largest. For the neighbourhood operator, we choose the trivial one. Lastly, for the sampling distribution, we choose β π ' ¯ ρ π , the normalised version of ρ π . Then, a mirror learning step maximises

<!-- formula-not-decoded -->

which, for appropriately chosen C π old , is proportional (and thus equivalent) to the trust-region learning update from Equation (2). Hence, the monotonic improvement of trustregion learning follows from Theorem 3.6, which also implies its convergence to the set of optimal policies.

TRPO The algorithm is designed to approximate trustregion learning, and so its mirror-learning representation is similar. We make the following changes: set the drift operator to D ν ' 0 , and choose the neighbourhood operator to

the average-KL ball . Precisely,

<!-- formula-not-decoded -->

The resulting mirror learning update is the learning rule of the TRPO algorithm. As a result, even though TRPO was supposed to only approximate the monotonic trust-region learning update (Schulman et al., 2015), this analysis shows that TRPO has monontonic convergence guarantees.

PPO We analyse PPO through unfolding the clipping objective L clip (Equation (3)). For a given s P S , it equals

<!-- formula-not-decoded -->

By recalling that r p ¯ π q ' ¯ π p a | s q{ π old p a | s q , we use importance sampling (Sutton &amp; Barto, 2018), and the trick of 'adding and subtracting', to rewrite L clip as

<!-- formula-not-decoded -->

Going forward, we focus on the expectation that is subtracted. First, we can replace the min operator with max , with the identity min f p x q ' max ' ´ f p x q ‰ , as follows

<!-- formula-not-decoded -->

Then, we move r p ¯ π q A π old p s, a q inside the max , and obtain

<!-- formula-not-decoded -->

which can be simplified as

<!-- formula-not-decoded -->

Notice now that this expression is always non-negative, due to the presence of the ReLU function (Fukushima &amp; Miyake, 1982). Furthermore, for ¯ π sufficently close to π old , i.e., so that r p ¯ π q P r 1 ´ ϵ, 1 ` ϵ s , the clip operator reduces to the identity function, and so Equation (7) is constant and zero. Hence, it is zero at ¯ π ' π old, and has a zero gradient. Therefore, the expression in Equation (7) is a drift functional of ¯ π p¨| s q . This, together with taking β π old ' ν ¯ π π old ' ¯ ρ π old , and the trivial neighbourhood operator N ' Π , shows that PPO, while it was supposed to be a heuristic approximation of trust-region learning, is by itself a rigorous instance of mirror learning. Thus, it inherits the monotonic improvement and convergence properties, which helps explain its great performance.

Interestingly, during the developmnent of PPO, a variant more closely related to TRL was considered (Schulman et al., 2017). Known as PPO-KL, the algorithm updates the policy to ¯ π ' π new to maximise

<!-- formula-not-decoded -->

and then scales τ up or down by a constant (typically 1 . 5 ), depending on whether the KL-divergence induced by the update exceeded or subceeded some target level. Intriguingly, according to the authors this approach, although more closely-related to TRL, failed in practice. Mirror learning explains this apparent paradox. Namely, the rescaling scheme of τ causes discontinuity of the penalty term, when viewed as a function of π old and ¯ π . This prevents the penalty from being a valid drift, so unlike PPO this version, PPO-KL, is not an instance of mirror learning. Recently, Hsu et al. (2020) introduced PPO-KL with a fixed τ , allowing for continuity of the KL-penalty. Such an algorithm is an instance of mirror learning, independently of whether the forward or backward KL-divergence is employed. And indeed, the authors find that both versions of the algorithms result in strong empirical performance, further validating our theory.

## 5. Related Work

The RL community has long worked on the development of theory to aid the development of theoretically-sound techniques. Perhaps, one of the greatest achievements along this thread is the development of trust-region learning (Schulman et al., 2015, TRL)-a framework that allows for stable training of monotonically improving policies. Unfortunately, despite its great theoeretical guarantees, TRL is intractable in most practical settings. Hence, the RL community focused on the development of heuristics, like TRPO (Schulman et al., 2015) and PPO (Schulman et al., 2017), that approximate it while trading off theoretical guarantees of TRL for practicality. As these methods established new state-of-the-art performance on a variety of tasks (Duan et al., 2016; Berner et al., 2019), the conceptual connection to TRL has been considered the key to success for many algorithms (Arjona-Medina et al., 2018; Queeney et al., 2021). However, recent works have shown that PPO is prone to breaking the core TRL principles of constraining the update size (Wang et al., 2020; Engstrom et al., 2020), and thus revealed an existing chasm between RL practice and understanding-a problem that mirror learning resolves.

Trust-region algorithms, although original, inspire connections between RL and optimisation, where the method of mirror descent has been studied (Nemirovskij &amp; Yudin, 1983). This idea has recently been extended upon in (Tomar et al., 2020), where Mirror Descent Policy Optimization (MDPO) was proposed-the algorithm optimises the policy with mirror ascent, where the role of the Bregman distance is played by the KL-divergence. The method

Figure 2. An intuitive view on the policy DAG and initial steps of mirror learning. A policy vertex has a neighbour, within its neighbourhood, which improves the return.

<!-- image -->

has been shown to achieve great empirical performance (which is also implied by mirror learning). Notably, a new stream of research in regularised RL is arising, where the regulariser is subsumed into the Bregman distance. For example, Neu et al. (2017) have shown that, in the averagereward setting, a variant of TRPO is an instance of mirror descent in a regularised MDP, and converges to the optimal policy. Shani et al. (2020) have generalised TRPO to handle regularised problems, and derived convergence rates for their methods in the tabular case. Lan (2021) and Zhan et al. (2021) have proposed mirror-descent algorithms that solve regularised MDPs with convex regularisers, and also provided their convergence properties. Here, we would like to highlight that mirror learning is not a method of solving regularised problems through mirror descent. Instead, it is a very general class of algorithms that solve the classical MDP. The term mirror , however, is inspired by the intuition behind mirror descent, which solves the image of the original problem under the mirror map (Nemirovskij &amp; Yudin, 1983)-similarly, we defined the mirror operator.

Mirror learning is also related to Probability Functional Descent (Chu et al., 2019, PFD). Although PFD is a general framework of probability distribution optimisation in machine learning problems, in the case of RL, it is an instance of mirror learning-the one recovering GPI. Lastly, the concepts of mirror descent and functional policy representation are connected in a concurrent work of Vaswani et al. (2021), who show how the technique of mirror descent implements the functional descent of parameterised policies. This approach, although powerful, fails to capture some algorithms, including PPO, which we prove to be an instance of mirror learning. The key to the generalisation power of our theory is its abstractness, as well as simplicity.

## 6. Graph-theoretical Interpretation

In this section we use mirror learning to make a surprising connection between RL and graph theory. We begin by introducing the following definition of a particular directed acyclic graph (DAG).

Definition 6.1 (Policy DAG) . Let D ν be a positive drift, and N be a neighbourhood operator. Then, the policy DAG

G p Π , D ν , N q is a graph where

- The vertex set is the policy space Π ,
- p π 1 , π 2 q is an edge if η p π 1 q ă η p π 2 q and π 2 P N p π 1 q ,
- The weight of an edge p π 1 , π 2 q is D ν π 1 p π 2 q .

This graph is a valid DAG because the transitive, asymmetric ' ă ' relation that builds edges prevents the graph from having cycles. We also know that, for every non-optimal policy π , there in an outgoing edge p π, π 1 q from π , as π 1 can be computed with a step of mirror learning (see Figure 2 for an intuitive picture).

The above definition allows us to cast mirror learning as a graph search problem. Namely, let π 0 P Π be a vertex policy at which we initialise the search, which further induces a sequence p π n q 8 n ' 0 . Let us define U β ' min π P Π ,s P S d p s q{ β p s q . Theorem 3.6 lets us upper-bound the weight of any traversed edge as

<!-- formula-not-decoded -->

Notice that η p π n q ´ η p π 0 q ' ř n i ' 1 ' η p π i q ´ η p π i ´ 1 q ‰ converges to η ˚ ´ η p π 0 q . Hence, the following series expansion

<!-- formula-not-decoded -->

is valid. Combining it with Inequality (8), we obtain a bound on the total weight of the path induced by the search

<!-- formula-not-decoded -->

Hence, mirror learning finds a path from π 0 to the set of graph sinks (the optimal policies), whose total weight is finite, and does not depend on the policies that appeared during the search. Note that one could also estimate the lefthand side from above by a bound p η ˚ ` V max q{ U β which is uniform for all initial policices π 0 . This finding may be considered counterintuitive: While we can be decreasing the number of edges in the graph by shrinking the neighbourhood operator N p π q , a path to π ˚ still exists, and despite (perhaps) containing more edges, its total weight remains bounded.

Lastly, we believe that this inequality can be exploited for practical purposes: the drift functional D is an abstract hyper-parameter, and the choice of it is a part of algorithm design. Practicioners can choose D to describe a cost that they want to limit throughout training. For example, one can set D ' risk , where risk π p ¯ π | s q quantifies some notion of risk of updating π p¨| s q to ¯ π p¨| s q , or D ' memory , to only make updates at a reasonable memory expense. Inequality (9) guarantees that the total expense will remain finite, and provides an upper bound for it. Thereby, we encourage employing mirror learning with drifts designed to satisfy constraints of interest.

Average Returr

Average Return

7.6

6.0

-60

-100

Single-Step with Different Drifts and Neighbourhoods

Ty-squared

Single-Step with Different Drifts and KL-Neighbourhood

TV-squared

Averaoe Returni

## 7. Numerical Experiments

********************•..•

**************•*•••••|

We verify the correctness of our theory with numerical experiments. Their purpose is not to establish a new stateof-the-art performance in the most challenging deep RL benchmarks. It is, instead, to demonstrate that algorithms that fall into the mirror learning framework obey Theorem 3.6 and Inequality (9). Hence, to enable a close connection between the theory and experiments we choose simple environments and for drift functionals we selected: KLdivergence, squared L2 distance, squared total variation distance, and the trivial (zero) drift. For the neighbourhood operator, in one suite of experiments, we use the expected drift ball, i.e. , the set of policies within some distance from the old policy, measured by the corresponding drift; in the second suite, we use the KL ball neighbourhood, like in TRPO. We represent the policies with n -dimensional action spaces as softmax distributions, with n ´ 1 parameters. For an exact verification of the theoretical results, we test each algorithm over only one random seed. In all experiments, we set the initial-state and sampling distributions to uniform. The code is available at https: //github.com/znowu/mirror-learning .

Single-step Game. In this game, the agent chooses one of 5 actions, and receives a reward corresponding to it. These rewards are 10 , 0 , 1 , 0 , 5 , respectively for each action. The optimal return in this game equals 10 .

Tabular Game. This game has 5 states, lined up next to each other. At each state, an agent can choose to go left, and receive a reward of ` 0 . 1 , stay at the current state and receive 0 reward, or go right and receive ´ 0 . 1 reward. However, if the agent goes left at the left-most state, it receives ´ 10 reward, and if it goes right at the right-most state, it recieves ` 10 reward. In these two cases, the game terminates. We set γ ' 0 . 999 . Clearly, the optimal policy here ignores the small intermediate rewards and always chooses to go right. The corresponding optimal expected return is approximately 9 . 7 .

GridWorld. We consider a 5 ˆ 5 grid, with a barrier that limits the agent's movement. The goal of the agent is to reach the top-right corner as quick as possible, which terminates the episode, and to avoid the bottom-left corner with a bomb. It receives ´ 1 reward for every step taken, and ´ 100 for stepping onto the bomb and terminating the game. For γ ' 0 . 999 , the optimal expected return is approximately ´ 7 .

Figure 3 confirms that the resulting algorithms achieve the monotonic improvement property, and converges to the optimal returns-this confirms Theorem 3.6. In these simple environments, the learning curves are influenced by the choice of the drift functional more than by the neighbourhood operator, although not significantly. An exception oc-

Figure 3. Mirror learning algorithms with different drifts and neighbourhood operators tested on simple environments. The solid lines represent the return, and the dotted ones represent the total drift. Algorithms in the left column use the drift ball neighbourhood, while those in the right use the KL ball. In both columns there are algorithms with each of the aforementioned drifts. Results are for one seed per environment and algorithm.

<!-- image -->

currs in the single-step game where, of course, one step of GPI is sufficient to solve the game. We also see that the value of the total drift (which we shift by V π 0 on the plots for comparison) remains well below the return in all environments, confirming Inequality (9).

## 8. Conclusion

In this paper, we introduced mirror learning -aframework which unifies existing policy iteration algorithms. We have proved Theorem 3.6, which states that any mirror learning algorithm solves the reinforcement learning problem. As a corollary to this theorem, we obtained convergence guarantees of state-of-the-art methods, including PPO and TRPO. More importantly, it provides a framework for the future development of theoretically-sound algorithms. We also proposed an interesting, graph-theoretical perspective on mirror learning, which establishes a connection between graph theory and RL. Lastly, we verifed the correctness of our theoretical results through numerical experiments on a diverse family of mirror learning instances in three simple toy settings. Designing and analysing the properties of the myriads of other possible mirror learning instances is an exciting avenue for future research.

## Acknowledgements

I dedicate this work to my little brother Michał , and thank him for persistently bringing light to my life.

I have to go now. We won't see each other for a while, but one day we'll get together again. I promise.

Kuba

## References

- Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., and Hochreiter, S. Rudder: Return decomposition for delayed rewards. arXiv preprint arXiv:1806.07857 , 2018.
- Ausubel, L. M. and Deneckere, R. J. A generalized theorem of the maximum. Economic Theory , 3(1):99-107, 1993.
- Baxter, J. and Bartlett, P. L. Infinite-horizon policygradient estimation. Journal of Artificial Intelligence Research , 15:319-350, 2001.
- Beck, A. and Teboulle, M. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31(3):167-175, 2003.
- Bellman, R. A markov decision process. journal of mathematical mechanics. 1957.
- Berner, C., Brockman, G., Chan, B., Cheung, V., Debiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680 , 2019.
- Chu, C., Blanchet, J., and Glynn, P. Probability functional descent: A unifying perspective on gans, variational inference, and reinforcement learning. In International Conference on Machine Learning , pp. 1213-1222. PMLR, 2019.
- Duan, Y., Chen, X., Houthooft, R., Schulman, J., and Abbeel, P. Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning , pp. 1329-1338. PMLR, 2016.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., and Madry, A. Implementation matters in deep policy gradients: A case study on ppo and trpo. In International Conference on Learning Representations , 2020.
- Fukushima, K. and Miyake, S. Neocognitron: A selforganizing neural network model for a mechanism of visual pattern recognition. In Competition and cooperation in neural nets , pp. 267-285. Springer, 1982.
- Gateaux, R. Sur diverses questions de calcul fonctionnel. Bulletin de la Soci´ et´ e Math´ ematique de France , 50:1-37, 1922.
- Hamilton, E. and Nashed, M. Global and local variational derivatives and integral representations of gˆ ateaux differentials. Journal of Functional Analysis , 49(1):128-144, 1982.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. Deep reinforcement learning that matters. In Proceedings of the AAAI conference on artificial intelligence , volume 32, 2018.
- Hsu, C. C.-Y., Mendler-D¨ unner, C., and Hardt, M. Revisiting design choices in proximal policy optimization. arXiv preprint arXiv:2009.10897 , 2020.
- Kakade, S. and Langford, J. Approximately optimal approximate reinforcement learning. In In Proc. 19th International Conference on Machine Learning . Citeseer, 2002.
- Kuba, J. G., Chen, R., Wen, M., Wen, Y., Sun, F., Wang, J., and Yang, Y. Trust region policy optimisation in multi-agent reinforcement learning. arXiv preprint arXiv:2109.11251 , 2021.
- Lan, G. Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. arXiv preprint arXiv:2102.00135 , 2021.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 , 2015.
- Liu, B., Cai, Q., Yang, Z., and Wang, Z. Neural proximal/trust region policy optimization attains globally optimal policy. arXiv preprint arXiv:1906.10306 , 2019.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. nature , 518 (7540):529-533, 2015.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pp. 1928-1937. PMLR, 2016.
- Nemirovskij, A. S. and Yudin, D. B. Problem complexity and method efficiency in optimization. 1983.
- Neu, G., Jonsson, A., and G´ omez, V. A unified view of entropy-regularized markov decision processes. CoRR , abs/1705.07798, 2017. URL http://arxiv.org/ abs/1705.07798 .

- Queeney, J., Paschalidis, I., and Cassandras, C. Generalized proximal policy optimization with sample reuse. Advances in Neural Information Processing Systems , 34, 2021.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International conference on machine learning , pp. 1889-1897. PMLR, 2015.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. ArXiv , abs/1707.06347, 2017.
- Shani, L., Efroni, Y., and Mannor, S. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pp. 56685675, 2020.
- Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. Deterministic policy gradient algorithms. In International conference on machine learning , pp. 387-395. PMLR, 2014.
- Sutton, R. S. and Barto, A. G. Reinforcement learning: An introduction . 2018.
- Sutton, R. S., Mcallester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems 12 , volume 12, pp. 1057-1063. MIT Press, 2000.
- Tomar, M., Shani, L., Efroni, Y., and Ghavamzadeh, M. Mirror descent policy optimization. arXiv preprint arXiv:2005.09814 , 2020.
- Van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence , volume 30, 2016.
- Vaswani, S., Bachem, O., Totaro, S., Mueller, R., Geist, M., Machado, M. C., Castro, P. S., and Roux, N. L. A functional mirror ascent view of policy gradient methods with function approximation. arXiv preprint arXiv:2108.05828 , 2021.
- Wang, Y., He, H., and Tan, X. Truly proximal policy optimization. In Uncertainty in Artificial Intelligence , pp. 113-122. PMLR, 2020.
- Watkins, C. J. C. H. and Dayan, P. Q-learning. Machine Learning , 8(3):279-292, May 1992. ISSN 15730565. doi: 10.1007/BF00992698. URL https:// doi.org/10.1007/BF00992698 .
- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8(3-4):229-256, 1992.
- Zhan, W., Cen, S., Huang, B., Chen, Y., Lee, J. D., and Chi, Y. Policy mirror descent for regularized reinforcement learning: A generalized framework with linear convergence. arXiv preprint arXiv:2105.11066 , 2021.
- Zhao, T., Hachiya, H., Niu, G., and Sugiyama, M. Analysis and improvement of policy gradient estimation. In NIPS , pp. 262-270. Citeseer, 2011.

## A. Definition Details

Definition 3.1. A drift functional D is a map

<!-- formula-not-decoded -->

such that for all s P S , and π, ¯ π P Π , writing D π p ¯ π | s q for D π ` ¯ π p¨| s q| s ˘ , the following conditions are met

1. D π p ¯ π | s q ě D π p π | s q ' 0 (nonnegativity) ,
2. D π p ¯ π | s q has zero gradient 3 with respect to ¯ π p¨| s q , evaluated at ¯ π p¨| s q ' π p¨| s q (zero gradient) .

Let ν ¯ π π P P p S q be a state distribution that can depend on π and ¯ π . The drift D ν of ¯ π from π is defined as

<!-- formula-not-decoded -->

ν π ¯ π is required to be such that the above expectation is continuous in π and ¯ π . The drift is positive if D ν π p ¯ π q ' 0 implies ¯ π ' π , and trivial if D ν π p ¯ π q ' 0 , @ π, ¯ π P Π .

In this definition, the notion of the gradient of D π p ¯ π | s q with respect to ¯ π p¨| s q P P p A q is rather intuitive. As the set P p A q is not a subset of R n , the gradient is not necessarily defined-however, we do not require its existence. Instead, as P p A q is a convex statistical manifold, we consider its Gˆ ateaux derivatives (Gateaux, 1922; Hamilton &amp; Nashed, 1982). That is, for any p P P p A q , writing v ' p ´ π p¨| s q , we consider the Gˆ ateaux derivative, given as the limit

<!-- formula-not-decoded -->

and require them to be zero at ¯ π p¨| s q ' π p¨| s q . That is, we require that δ D π p ¯ π | s qr v, π p¨| s qs ' 0 .

Definition 3.2. We say that N : Π Ñ P p Π q is a neighbourhood operator , where P p Π q is the power set of Π , if

1. It is a continuous map (continuity) ,
2. Every N p π q is a compact set (compactness) ,
3. There exists a metric χ : Π ˆ Π Ñ R , such that @ π P Π , there exists ζ ą 0 , such that χ p π, ¯ π q ď ζ implies ¯ π P N p π q (closed ball) .

The trivial neighbourhood operator is N ' Π .

We specify that a metric χ : Π ˆ Π Ñ R is a metric between elements of a product of statistical manifolds, i.e., Π ' Ś s P S P p A q . Therefore, we carefully require that it is a non-negative, monotonically non-decreasing function of individual statistical divergence metrics χ s ` π p¨| s q , ¯ π p¨| s q ˘ , @ s P S , and that χ ` π, ¯ π p¨| s q ˘ ' 0 only if χ s ` π p¨| s q , ¯ π p¨| s q ˘ ' 0 , @ s P S .

3 More precisely, all its Gˆ ateaux derivatives are zero.

## B. Proofs of Lemmas

Lemma 3.3. Let π old and π new be policies. Suppose that

<!-- formula-not-decoded -->

Then, π new is better than π old, so that for every state s ,

<!-- formula-not-decoded -->

Proof. Let us, for brevity, write V new ' V π new , V old ' V π old , β ' β π old , ν ' ν π new π old , and E π r¨s ' E a ' π, s 1 ' P r¨s . For every state s P S , we have

<!-- formula-not-decoded -->

by Inequality (4). Taking infimum within the expectation,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking infimum over s

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude that

<!-- formula-not-decoded -->

which is non-negative, and concludes the proof.

Lemma 3.5. Let π old be a policy and π new be obtained from the mirror update of Equation (5). Then,

<!-- formula-not-decoded -->

Hence, π new attains the properties provided by Lemma 3.3.

Proof. We will prove the statement by contradiction. Suppose that

<!-- formula-not-decoded -->

and that there exists a state s 0 , such that

Hence,

<!-- formula-not-decoded -->

Let us define a policy ˆ π , so that ˆ π p¨| s 0 q ' π old p¨| s 0 q , and ˆ π p¨| s q ' π new p¨| s q for s ‰ s 0 . As the distance between of ˆ π from π old is the same as of π new at s ‰ s 0 , and possibly smaller at s 0 , we have that ¯ π is not further from π old than π new. Hence, ˆ π P N p π old q . Furthermore, Inequality (12) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is a contradiction with Equation (11), which finishes the proof.

## C. The Proof of Theorem 3.6

Theorem 3.6 (The Fundamental Theorem of Mirror Learning) . Let D ν be a drift, N be a neighbourhood operator, and the sampling distribution β π depend continuously on π . Let π 0 P Π , and the sequence of policies p π n q 8 n ' 0 be obtained by mirror learning induced by D ν , N , and β π . Then, the learned policies

1. Attain the strict monotonic improvement property,

<!-- formula-not-decoded -->

2. Their value functions converge to the optimal one,
4. Their ω -limit set consists of the optimal policies.

Proof. We split the proof of the whole theorem into two parts, each of which proves different groups of properties stated in the theorem.

## Strict monotonic improvement (Property 1)

By Lemma 3.3, we have that @ n P N , s P S ,

<!-- formula-not-decoded -->

3. Their expected returns converge to the optimal return,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us write β ' β π n and ν ' ν π n ` 1 π n . With this inequality, as well as Lemma 3.5, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality holds by Lemma 3.5. To summarise, we proved that

<!-- formula-not-decoded -->

Taking the expectation over s ' d , we obtain

<!-- formula-not-decoded -->

which finishes this part of the proof.

## Optimality (Properties 2, 3, &amp; 4)

## Step 1 (convergence of value functions).

We consider the sequence p V π n q 8 n ' 0 of associated value functions. By Lemma 3.5, we know that @ n P N , V π n ` 1 ě V π n uniformly. However, as for every n P N and s P S , the uniform bound V π n p s q ď V max holds, the sequence of value functions must converge. We denote its limit by V .

## Step 2 (characterisation of policy limit points).

Let L Π be the set of all limit points of p π n q 8 n ' 0 . As the sequence p π n q 8 n ' 0 is bounded, Bolzano-Weierstrass guarantees that there exists ¯ π P L Π , and a subsequence, say p π n i q 8 i ' 1 , which converges to it. Writing β k ' β π k , we consider the optimisation problem that mirror learning induces at every element π n i of this subsequence,

<!-- formula-not-decoded -->

Note that by the continuity of the value function (Kuba et al., 2021, Appendix A), as well as the drift, the neighbourhood operator, and the sampling distribution, we obtain that the above expression is continuous in π n i . Hence, by Berge's Maximum Theorem (Ausubel &amp; Deneckere, 1993), writing ¯ β ' β ¯ π , as i Ñ 8 , the above expression converges to the following,

<!-- formula-not-decoded -->

Note that V ¯ π ' V , as the sequence of value functions converges (has a unique limit point V ). Furthermore, for all i P N , π n i ` 1 is the argmax (precisely, is an element of the upper hemicontinuous argmax correspondence) of Equation (15). Hence, there exists a subsequence p π n i k ` 1 q of p π n i ` 1 q i P N that converges to a policy π 1 , which is a solution to Equation (16).

Claim : The solution to Equation (16) is π 1 ' ¯ π .

We prove the above claim by contradiction. Suppose that π 1 ‰ ¯ π . As π 1 can be obtained from ¯ π via mirror learning, Inequality (13) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we have then for some state s ,

which can be written as

Using Inequality (17), and non-negativity of drifts,

<!-- formula-not-decoded -->

However, as V π 1 is the limit of V π n i k ` 1 , we have V π 1 ' V (uniqueness of the value limit) which yields a contradiction, proving the claim. Hence, the limit point ¯ π of the sequence p π n q 8 n ' 0 satisfies

<!-- formula-not-decoded -->

## Step 3 (dropping the drift).

Let ¯ π be a limit point of p π n q 8 n ' 0 . From Equation (18), and the definition of the mirror operator, we know that

<!-- formula-not-decoded -->

Suppose that there exists a policy π 1 , and a state s , such that

<!-- formula-not-decoded -->

For any policy π , consider the canonical parametrisation π p¨| s q ' p x 1 , . . . , x m ´ 1 , 1 ´ ř m ´ 1 i ' 1 x i q , where m is the size of the action space. We have that

<!-- formula-not-decoded -->

This means that E a ' π ' A ¯ π p s, a q ‰ is an affine function of π p¨| s q , and thus, its Gˆ ateaux derivatives are constant in P p A q for fixed directions. Together with Inequality (20), this implies that the Gˆ ateaux derivative, in the direction from ¯ π to π 1 , is strictly positive. Furthermore, the Gˆ ateaux derivatives of ν π ¯ π p s q ¯ β p s q D ¯ π p π | s q are zero at π p¨| s q ' ¯ π p¨| s q , as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and both the lower and upper of the above bounds have zero derivatives. Hence, the Gˆ ateaux derivative of E a ' π ' A ¯ π p s, a q ‰ ´ ν π ¯ π p s q ¯ β p s q D ¯ π p π | s q is strictly positive. Therefore, for conditional policies ˆ π p¨| s q sufficiently close to ¯ π p¨| s q in the direction towards π 1 p¨| s q , we have (with a slight abuse of notation for ν )

<!-- formula-not-decoded -->

Let us construct a policy r π as follows. For all states y ‰ s , we set r π p¨| y q ' ¯ π p¨| y q . Moreover, for r π p¨| s q we choose ˆ π p¨| s q as in Inequality (21), sufficiently close to ¯ π p¨| s q , so that r π P N p ¯ π q . Then, we have

<!-- formula-not-decoded -->

The above contradicts Equation (19). Hence, the assumption made in Inequality (20) was false. Thus, we have proved that, for every state s ,

<!-- formula-not-decoded -->

## Step 4 (optimality).

Equation (23) implies that ¯ π is an optimal policy (Sutton &amp; Barto, 2018), and so the value function V ' V ¯ π is the optimal value function V ˚ (Property 2). Consequently, the expected return that the policies converge to, η ' E s ' d ' V p s q ‰ ' E s ' d ' V ˚ p s q ‰ ' η ˚ is optimal (Property 3). Lastly, as ¯ π was an arbitrary limit point, any element of the ω -limit set is an optimal policy (Property 4). This finishes the proof.

## D. Extension to Continuous State and Action Spaces

The results on Mirror Learning extend to continuous state and action spaces through general versions of our claims. These require a little more care in their formulation, but their validity holds as a corollary to our proofs.

In general, the state and the action spaces S and A must be assumed to be compact and measurable. For the state space, we introduce a reference probability measure µ S : P p S q Ñ R that is strictly positive, i.e , µ S p s q ą 0 , @ s P S . Under such setting, a policy π ˚ is optimal if it satisfies the Bellman optimality equation

<!-- formula-not-decoded -->

at states that form a set of measure 1 with respect to µ S . In other words, a policy is optimal if it obeys the Bellman optimality equation almost surely.

As for the results, the inequality provided by Lemma 3.3 (the state value function improvement) holds almost surely with respect to µ S as long as the policy π new satisfies Inequality (4), also almost surely with respect to this measure. Of course, the corollary on the monotonic improvement remains valid. Next, the inequality introduced in Lemma 3.5 must now be stated almost surely, again with respect to µ S . Lastly, the entire statement of Theorem 3.6 remains unchanged-it has the same formulation for all compact .

## E. The Listing of RL Approaches as Instances of Mirror Learning

## Generalised Policy Iteration (Sutton &amp; Barto, 2018)

<!-- formula-not-decoded -->

- Drift functional: trivial D ' 0 .
- Neighbourhood operator: trivial N ' Π .
- Sampling distribution: arbitrary.

## Trust-Region Learning (Schulman et al., 2015)

<!-- formula-not-decoded -->

- Drift functional:

<!-- formula-not-decoded -->

- Neighbourhood operator: trivial N ' Π .
- Sampling distribution: β π ' ¯ ρ π , the normalised marginal discounted state distribution.

## TRPO (Schulman et al., 2015)

- Drift functional: trivial D ' 0 .
- Neighbourhood operator:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Sampling distribution: β π ' ¯ ρ π .

## PPO (Schulman et al., 2017)

<!-- formula-not-decoded -->

- Drift functional:

<!-- formula-not-decoded -->

- Neighbourhood operator: trivial N ' Π . Note, however, that in practical implementations the policy is updated with ϵ PPO steps of gradient ascent with gradient-clipping threshold M . This corresponds to a neighbourhood of an L2-ball of radius Mϵ PPO in the policy parameter space.
- Sampling distribution: β π ' ¯ ρ π .

## PPO-KL (Hsu et al., 2020)

- Drift functional:
- Drift functional:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Neighbourhood operator: the same as in PPO.
- Sampling distribution: β ¯ ρ

π ' π .

## MDPO(Tomar et al., 2020)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Neighbourhood operator: trivial N ' Π .
- Sampling distribution: β π ' ¯ ρ π for on-policy MDPO, and β π n p s q ' 1 n ` 1 ř n i ' 0 ¯ ρ π i p s q for off-policy MDPO.

## F. Instructions for Implementation of Off-Policy Mirror Learning

In the case of off-policy learning, estimating E a ' ¯ π ' Q π old p s, a q ‰ is not as straighforward as in Equation (6), since sampling actions from the replay buffer is not equivalent to sampling actions from π old p¨| s q anymore. The reason for this is that while sampling an action from the buffer, we also sample a past policy, π hist, which was used to insert the action to the buffer. To formalise it, we draw a past policy from some distribution dictated by the buffer, π hist ' h P P p Π q , and then draw an action a ' π hist. To account for this, we use the following estimator,

<!-- formula-not-decoded -->

Note that this requires that the value π hist p a | s q has also been inserted in the buffer. The expectation of the new estimator can be computed as

<!-- formula-not-decoded -->

Hence, the estimator has the desired mean.