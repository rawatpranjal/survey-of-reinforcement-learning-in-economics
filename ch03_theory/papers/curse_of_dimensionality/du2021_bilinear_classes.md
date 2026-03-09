## Bilinear Classes: A Structural Framework for Provable Generalization in RL

Simon S. Du * Sham M. Kakade † Jason D. Lee ‡ Shachar Lovett § Gaurav Mahajan ¶ Wen Sun || Ruosong Wang **

## Abstract

This work introduces Bilinear Classes, a new structural framework, which permit generalization in reinforcement learning in a wide variety of settings through the use of function approximation. The framework incorporates nearly all existing models in which a polynomial sample complexity is achievable, and, notably, also includes new models, such as the Linear Q ∗ /V ∗ model in which both the optimal Q -function and the optimal V -function are linear in some known feature space. Our main result provides an RL algorithm which has polynomial sample complexity for Bilinear Classes; notably, this sample complexity is stated in terms of a reduction to the generalization error of an underlying supervised learning sub-problem. These bounds nearly match the best known sample complexity bounds for existing models. Furthermore, this framework also extends to the infinite dimensional (RKHS) setting: for the the Linear Q ∗ /V ∗ model, linear MDPs, and linear mixture MDPs, we provide sample complexities that have no explicit dependence on the explicit feature dimension (which could be infinite), but instead depends only on information theoretic quantities.

## 1 Introduction

Tackling large state-action spaces is a central challenge in reinforcement learning (RL). Here, function approximation and supervised learning schemes are often employed for

* University of Washington. Email: ssdu@cs.washington.edu

† University of Washington and Microsoft Research. Email: sham@cs.washington.edu

‡ Princeton University. Email: jasonlee@princeton.edu

§ University of California, San Diego. Email: slovett@cs.ucsd.edu

¶ University of California, San Diego. Email: gmahajan@eng.ucsd.edu

|| Cornell University. Email: ws455@cornell.edu

** Carnegie Mellon University. Email: ruosongw@andrew.cmu.edu

Table 1: Relations between frameworks. /check : the column framework contains the row framework. ✗ : the column framework does not contains the row framework. B-Rank: Bellman Rank [Jiang et al., 2017], which is defined in terms of the roll-in distribution and the function approximation class for Q ∗ . B-Complete: Bellman Complete [Munos, 2005] (Zanette et al. [2020] proposed a sample efficient algorithm), which assumes the function class is closed under the Bellman operator. W-Rank: Witness Rank [Sun et al., 2019]: a model-based analogue of Bellman Rank. Bilinear Class: our proposed framework.

| Framework                  | B-Rank   | B-Complete   | W-Rank   | Bilinear Class (this work)   |
|----------------------------|----------|--------------|----------|------------------------------|
| B-Rank                     | /check   | ✗            | /check   | /check                       |
| B-Complete                 | ✗        | /check       | ✗        | /check                       |
| W-Rank                     | ✗        | ✗            | /check   | /check                       |
| Bilinear Class (this work) | ✗        | ✗            | ✗        | /check                       |

generalization across large state-action spaces. While there have been a number of successful applications [Mnih et al., 2013, Kober et al., 2013, Silver et al., 2017, Wu et al., 2017]. there is also a realization that practical RL approaches are quite sample inefficient.

Theoretically, there is a growing body of results showing how sample efficiency is possible in RL for particular model classes (often with restrictions on the model dynamics though in some cases on the class of value functions), e.g. State Aggregation [Li, 2009, Dong et al., 2020c], Linear MDPs [Yang and Wang, 2019, Jin et al., 2020], Linear Mixture MDPs [Modi et al., 2020a, Ayoub et al., 2020], Reactive POMDPs [Krishnamurthy et al., 2016], Block MDPs [Du et al., 2019a], FLAMBE [Agarwal et al., 2020b], Reactive PSRs [Littman et al., 2001], Linear Bellman Complete [Munos, 2005, Zanette et al., 2020].

More generally, there are also a few lines of work which propose more general frameworks, consisting of structural conditions which permit sample efficient RL; these include the low-rankness structure (e.g. the Bellman rank [Jiang et al., 2017] and Witness rank [Sun et al., 2019]) or under a complete condition [Munos, 2005, Zanette et al., 2020]. The goal in these latter works is to develop a unified theory of generalization in RL, analogous to more classical notions of statistical complexity (e.g. VC-theory and Rademacher complexity) relevant for supervised learning. These latter frameworks are not contained in each other (see Table 1), and, furthermore, there are a number of natural RL models that cannot be incorporated into each of these frameworks (see Table 2).

Motivated by this latter line of work, we aim to understand if there are simple and natural structural conditions which capture the learnability in a general class of RL models.

Our Contributions. This work provides a simple structural condition on the hypothesis class (which may be either model-based or value-based), where the Bellman error has a

Table 2: Whether a framework includes a model that permits a sample efficient algorithm. /check means the framework includes the model, ✗ means not, and /check ! means the sample complexity using that framework needs to scale with the number of action (which is not necessary). 'Sample efficient is not possible' means the sample complexity needs to scale exponentially with at least one problem parameter. See Section 2, Section 4.3, Section 6 and Appendix A for detailed descriptions of the models.

|                                                         | B-Rank                            | B-Complete                        | W-Rank                            | Bilinear Class (this work)        |
|---------------------------------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| Tabular MDP                                             | /check                            | /check                            | /check                            | /check                            |
| Reactive POMDP [Krishnamurthy et al., 2016]             | /check                            | ✗                                 | /check                            | /check                            |
| Block MDP [Du et al., 2019a]                            | /check                            | ✗                                 | /check                            | /check                            |
| Flambe / Feature Selection [Agarwal et al., 2020b]      | /check                            | ✗                                 | /check                            | /check                            |
| Reactive PSR [Littman and Sutton, 2002]                 | /check                            | ✗                                 | /check                            | /check                            |
| Linear Bellman Complete [Munos, 2005]                   | ✗                                 | /check                            | ✗                                 | /check                            |
| Generalized Linear Bellman Complete [Wang et al., 2019] | ✗                                 | ✗                                 | ✗                                 | /check                            |
| Linear MDPs [Yang and Wang, 2019, Jin et al., 2020]     | /check !                          | /check                            | /check !                          | /check                            |
| Linear Mixture Model [Modi et al., 2020b]               | ✗                                 | ✗                                 | ✗                                 | /check                            |
| Linear Quadratic Regulator                              | ✗                                 | /check                            | ✗                                 | /check                            |
| Kernelized Nonlinear Regulator [Kakade et al., 2020]    | ✗                                 | ✗                                 | /check                            | /check                            |
| Factored MDP [Kearns and Koller, 1999]                  | ✗                                 | ✗                                 | ✗                                 | /check                            |
| Q /star 'irrelevant' State Aggregation [Li, 2009]       | /check                            | ✗                                 | ✗                                 | /check                            |
| Linear Q /star / V /star (this work)                    | ✗                                 | ✗                                 | ✗                                 | /check                            |
| RKHS Linear MDP (this work)                             | ✗                                 | ✗                                 | ✗                                 | /check                            |
| RKHS Linear MixtureMDP (this work)                      | ✗                                 | ✗                                 | ✗                                 | /check                            |
| Low Occupancy Complexity (this work)                    | ✗                                 | ✗                                 | ✗                                 | /check                            |
| Q /star State-action Aggregation [Dong et al., 2020c]   | ✗                                 | ✗                                 | ✗                                 | ✗                                 |
| Deterministic linear Q /star [Wen and Van Roy, 2013]    | ✗                                 | ✗                                 | ✗                                 | ✗                                 |
| Linear Q /star [Weisz et al., 2020]                     | Sample efficiency is not possible | Sample efficiency is not possible | Sample efficiency is not possible | Sample efficiency is not possible |

particular bilinear form, under which sample efficient learning is possible; we refer such a framework as a Bilinear Class. This structural assumption can be seen as generalizing the Bellman rank [Jiang et al., 2017]; furthermore, it not only contains existing frameworks, it also covers a number of new settings that are not easily incorporated in previous frameworks (see Tables 1 and 2).

Our main result presents an optimization-based algorithm, BiLin-UCB, which provably enjoys a polynomial sample complexity guarantee for Bilinear Classes (cf. Theorem 5.2). Although our framework is more general than existing ones, our proof is substantially simpler - we give a unified analysis based on the elliptical potential lemma, developed for the theory of linear bandits [Dani et al., 2008, Srinivas et al., 2009].

Furthermore, as a point of emphasis, our results are non-parametric in nature (stated in terms of an information gain quantity [Srinivas et al., 2009]), as opposed to finite dimensional as in prior work. From a technical point of view, it is not evident how to extend prior approaches to this non-parametric setting. Notably, the non-parametric regime is particularly relevant to RL due to that, in RL, performance bounds do not degrade gracefully with

approximation error or model mis-specification (e.g. see Du et al. [2020a] for discussion of these issues); the relevance of the non-parametric regime is that it may provide additional flexibility to avoid the catastrophic quality degradation due to approximation error or model mis-specification.

A few further notable contributions are:

- Definition of Bilinear Class: Our key conceptual contribution is the definition of the Bilinear Class, which isolates two key critical properties. The first property is that the Bellman error can be upper bounded by a bilinear form depending on the hypothesis. The second property is that the corresponding bilinear form for all hypothesis in the hypothesis class can be estimated with the same dataset. Analogous to supervised learning, this allows for efficient data reuse to estimate the Bellman error for all hypothesis simultaneously and eliminate those with high error.
- A reduction to supervised learning: One appealing aspect of this framework is that the our main sample complexity result for RL is quantified via a reduction to the generalization error of a supervised learning problem, where we have a far better understanding of the latter. This is particularly important due to that we make no explicit assumptions on the hypothesis class H itself, thus allowing for neural hypothesis classes in some cases (the Bilinear Class posits an implicit relationship between H and the underlying MDP M ).
- New models: We show our Bilinear Class framework incorporates new natural models, that are not easily incorporated into existing frameworks, e.g. linear Q ∗ /V ∗ , Low Occupancy Complexity, along with (infinite-dimensional) RKHS versions of linear MDPs and linear mixture MDPs. The linear Q ∗ /V ∗ result is particularly notable due to a recent and remarkable lower bound which showed that if we only assume Q ∗ is linear in some given set of features, then sample efficient learning is information theoretically not possible [Weisz et al., 2020]. In perhaps a surprising contrast, our works shows that if we assume that both Q /star and V /star are linear in some given features then sample efficient learning is in fact possible.
- Non-parametric rates: Our work is applicable to the non-parametric setting, where we develop new analysis tools to handle a number of technical challenges. This is notable as non-parametric rates for RL are few and far between. Our results are stated in terms of the critical information gain which can viewed as an analogous quantity to the critical radius , a quantity which is used to obtain sharp rates in non-parametric statistical settings [Wainwright, 2019].
- Flexible Framework: The Bilinear Class framework is easily modified to include cases that do not strictly fit the definition. We show several examples of this in Section 6,

where we show simple modifications of Bilinear Class framework include Witness Rank and Kernelized Nonlinear Regulator.

Organization Section 2 provides further related work. Section 3 introduce some technical background and notation. Section 4 introduces our Bilinear Class framework, where we instantiate it on the several RL models, and Section 5 describes our algorithm and provides our main theoretical results. In Section 6, we introduce further extensions of Bilinear Classes. We conclude in Section 7. Appendix A provides additional examples of the Bilinear Class including the feature selection model Agarwal et al. [2020b], Q ∗ state aggregation, LQR, Linear MDP, and Block MDP. Appendix B provides missing proofs of Section 5. Appendix C provides a key technical theorem to attain non-parametric convergence rates in terms of the information gain, and Appendix D uses this to show concentration inequalities for all the models in a unified approach. Appendix E provides proofs for Section 6. Finally, Appendix G shows that low information gain is necessary in both Bellman Complete and Linear MDP by showing that small RKHS norm is not sufficient for sample-efficient reinforcement learning.

## 2 Related Work: Frameworks and Models

Relations Among Frameworks. We first review existing frameworks and the relations among them. See Table 1 for a summary.

Jiang et al. [2017] defines a notion, Bellman Rank (B-Rank in Tables), in terms of the roll-in distribution and the function approximation class for Q ∗ , and give an algorithm with a polynomial sample complexity in terms of the Bellman Rank. They also showed a class of models, including tabular MDP, LQR, Reactive POMDP [Krishnamurthy et al., 2016], and Reactive PSR [Littman and Sutton, 2002] admit a low Bellman Rank, and thus they can be solved efficiently. Some recently proposed models, such as Block MDP [Du et al., 2019a], linear MDP [Yang and Wang, 2019, Jin et al., 2020] can also be shown to have a low Bellman rank. One caveat is that their algorithm requires a finite number of actions, so cannot be directly applied to (infinite-action) linear MDP and LQR. Subsequently, Sun et al. [2019] proposed a new framework, Witness Rank (W-Rank in tables), which generalizes Bellman Rank to model-based setting.

Bellman Complete (B-Complete in tables) is a framework of another style, which assumes that the class used for approximating the Q -function is closed under the Bellman operator. As shown in Table 1, neither the low-rank-style framework (Bellman Rank and Witness Rank) nor the complete-style framework (B-Complete) contains the other (See e.g., Zanette et al. [2020]).

Eluder dimension [Russo and Van Roy, 2014] is another structural condition which directly assumes the function class allows for strong extrapolation after observing dimension number of samples. With appropriate representation conditions (stronger than Bellman Complete), there is an efficient algorithm for function classes with small eluder dimension [Wang et al., 2020]. However due to Eluder dimension requiring extrapolation, there are few examples of function classes with small eluder dimension beyond linear functions and monotone transformations of linear functions both of which are captured by the bilinear class.

Comparison to Bellman Eluder Concurrently, Jin et al. [2021] proposes a new structural model called Bellman Eluder dimension (BE dimension) which takes both the MDP structure and the function class into consideration. We note that neither BE nor Bilinear Class capture each other. Notably, Bilinear Classes, via use of flexible Bellman error estimators, naturally captures model-based settings including linear mixture MDPs, KNRs, and factored MDPs, which are hard for model-free algorithms and frameworks to capture since the value functions of these models could be arbitrarily complicated. Specifically, Sun et al. [2019] shows that for factored MDPs, model-free algorithms such as OLIVE Jiang et al. [2017] suffer exponential sample complexity in worst case which implies that both BE dimension and Bellman rank are large for factored MDPs. However, Bilinear Class and Witness rank Sun et al. [2019] properly capture the complexity of factored MDPs. Similar situation may also apply to KNRs. For instance, Dong et al. [2020a] showed that for a simple piecewise linear dynamics (thus captured by KNRs) and piecewise reward functions, the optimal policy could contain exponentially many linear pieces and the optimal Q and V functions are fractals which are not differentiable anywhere and cannot be approximated by any neural networks with a polynomial width. It is unclear if such models have low BE dimension.

The primary difference is that the two complexity measures are applied to different structural aspects of the MDP: Bellman eluder framework is applied to the Bellman error and the bilinear class is applied to any loss estimator of the Bellman error. The actual complexity measures of eluder dimension and information gain are very similar and in fact equivalent for RKHS [Huang et al., Jin et al., 2021]. As these two complexity measures are different in general, an interesting direction for further work is to understand how eluder dimension can address new settings of practical interest beyond (generalized) linear models and whether Bellman eluder dimension can be broadened to capture model-based approaches (like the linear mixture model). Finally, we comment that there are models (e.g., deterministic linear Q /star and Q /star state-action aggregation) that are captured by neither frameworks; we leave to future work to propose a framework that can capture these models that do not have error amplification.

With an additional Bellman completeness assumption on the function class, Jin et al. [2021] gives an algorithm which extends Eleanor from Zanette et al. [2020] to nonlinear function approximation that achieves a regret guarantee with faster rates than our algorithm. We note that our algorithm and OLIVE (as shown by Jin et al. [2021]) does not require Bellman completeness which is a much stronger assumption than realizability. As examples, the low occupancy complexity, feature selection model, linear mixture model, and many other model-based models are not Bellman complete. While our work focuses on PAC bounds, we conjecture that the techniques from Dong et al. [2020b] can be used for deriving regret bounds without completeness.

Reinforcement Learning Models. Nowwe discuss existing RL models. A summary on whether a model can be incorporated into a framework is provided in Table 2.

Tabular MDP is the most basic model, which has a finite number of states and actions, and all frameworks incorporate this model. When the state-action space is large, different RL models have been proposed to study when one can generalize across the state-action pairs.

Reactive POMDP [Krishnamurthy et al., 2016] assumes there is a small number of hidden states and the Q ∗ -function belongs to a pre-specified function class. Block MDP [Du et al., 2019a] also assumes there is a small number of hidden states and further assumes the hidden states are decodable. Reactive PSR [Littman et al., 2001] considers partial observable systems whose parameters are grounded in observable quantities. FLAMBE [Agarwal et al., 2020b] considers the feature selection and removes the assumption of known feature in linear MDP. These models all admit a low-rank structure, and thus can be incorporated into the Bellman Rank or Witness Rank and our Bilinear Classes.

The Linear Bellman Complete model [Munos, 2005] uses linear functions to approximate the Q -function, and assumes the linear function class is closed under the Bellman operator. Zanette et al. [2020] presented a statistically efficient algorithm for this model. This model does not have a low Bellman Rank or Witness Rank but can be incorporated into the Bellman Complete framework and ours.

Linear MDP [Yang and Wang, 2019, Jin et al., 2020] assumes the transition probability and the reward are linear in given features. This model not only admits a low-rank structure, but also satisfies the complete condition. Therefore, this model belongs in all frameworks. However, when the number of action is infinite, the algorithms for Bellman Rank and Witness Rank are not applicable because their sample complexity scales with the number of actions. Linear mixture MDP [Modi et al., 2020a, Ayoub et al., 2020] assumes the transition probability is a linear mixture of some base models. This model cannot be included in Bellman Rank, Witness Rank, or Bellman Complete, but our Bilinear Classes includes this model.

LQRisafundamental model for continuous control that can be efficiently solvable [Dean et al., 2019]. While LQR has a low Bellman Rank and low Witness Rank, since the algorithms for Bellman Rank and Witness Rank scale with the number of actions and LQR's action set is uncountable, these two frameworks cannot incorporate LQR.

There is a line of work on state-action aggregation. Q ∗ 'irrelevance' state aggregation assumes one can aggregate states to a meta-state if these states share the same Q ∗ value, and the number of meta-states is small [Li, 2009, Jiang et al., 2015]. Q ∗ state-action aggregation aggregates state-action pairs to a meta-state-action pair if these pairs have the same Q ∗ -value [Dong et al., 2020c, Li, 2009].

Lastly, when only assuming Q ∗ is linear, there exists an exponential lower bound [Weisz et al., 2020], but with the additional assumption that the MDP is (nearly) deterministic and has large sub-optimality gap, there exists sample efficient algorithms [Wen and Van Roy, 2013, Du et al., 2019b, 2020b].

## 3 Setting

We denote an episodic finite horizon, non-stationary MDP with horizon H , by M = { S , A , r, H, { P h } H -1 h =0 , s 0 } , where S is the state space, A is the action space, r : S ×A ↦→ [0 , 1] is the expected reward function with the corresponding random variable R ( s, a ) , P h : S×A ↦→ /triangle ( S ) (where /triangle ( S ) denotes the probability simplex over S ) is the transition kernel for all h , H ∈ Z + is the planning horizon and s 0 is a fixed initial state 1 . For ease of exposition, we use the notation o h for 'observed transition info at timestep h ' i.e. o h = ( r h , s h , a h , s h +1 ) where r h is the observed reward r h = R ( s h , a h ) and s h , a h , s h +1 is the observed state transition at timestep h .

A deterministic, stationary policy π : S ↦→ A specifies a decision-making strategy in which the agent chooses actions adaptively based on the current state, i.e. a h ∼ π ( s h ) . We denote a non-stationary policy π = { π 0 , . . . , π H -1 } as a sequence of stationary policies where π h : S ↦→ A .

<!-- formula-not-decoded -->

Given a policy π and a state-action pair ( s, a ) ∈ S × A , the Q -function at time step h is defined as and, similarly, a value function time step h of a given state s under a policy π is defined as

<!-- formula-not-decoded -->

1 Our results generalizes to any fixed initial state distribution

where both expectations are with respect to s 0 , a 0 , . . . s H -1 , a H -1 ∼ d π . We use Q /star h and V /star h to denote the Q and V -functions of the optimal policy.

Sample Efficient Algorithms. Throughout the paper, we will consider an algorithm as sample-efficient, if it uses number of trajectories polynomial in the problem horizon H , inherent dimension d , accuracy parameter 1 //epsilon1 and poly-logarithmic in the number of candidate value-functions.

Notation. For any two vectors x, y , we denote [ x, y ] as the vector that concatenates x, y , i.e., [ x, y ] := [ x /latticetop , y /latticetop ] /latticetop . For any set S , we write /triangle ( S ) to denote the probability simplex. We often use U ( S ) as the uniform distribution over set S . We will let V denote a Hilbert space (which we assume is either finite dimensional or separable).

We let [ H ] denote the set { 0 , . . . H -1 } . We slightly abuse notation (overloading d π with its marginal distributions), where s h ∼ d π , ( s h , a h ) ∼ d π , ( r h , s h , a h , s h +1 ) ∼ d π and most frequently o h ∼ d π denotes the marginal distributions at timestep h . We also use the shorthand notation s 0 , a 0 , . . . s H -1 , a H -1 ∼ π , s h , a h ∼ π for s 0 , a 0 , . . . s H -1 , a H -1 ∼ d π , s h , a h ∼ d π .

## 4 Bilinear Classes

Before, we define our structural framework - Bilinear Class, we first define our hypothesis class.

Hypothesis Classes. We assume access to a hypothesis class H = H 0 × . . . × H H -1 , which can be abstract sets that permit for both model-based and value-based hypotheses. The only restriction we make is that for all f ∈ H , we have an associated state-action value function Q h,f and a value function V h,f . We next provide some examples:

1. An example of value-based hypothesis class H is an explicit set of state-action value Q and value functions V i.e.

<!-- formula-not-decoded -->

Note that in this case, for any hypothesis f := (( Q 0 , V 0 ) , ( Q 1 , V 1 ) , . . . , ( Q H -1 , V H -1 )) ∈ H , we can take the associated Q h,f = Q h and associated V h,f = V h .

2. Another example of value-based hypothesis class H is when H is just a set of stateaction value Q functions i.e.

<!-- formula-not-decoded -->

In this case, for any hypothesis f := ( Q 0 , Q 1 , . . . , Q H -1 ) ∈ H , we can take the associated Q h,f = Q h and the associated V h,f function to be greedy with respect to the Q h,f function i.e. V h,f ( · ) = max a ∈A Q h,f ( · , a ) .

3. An example of model-based hypothesis class is when H h is a set of models/transition kernels P h and reward functions R h i.e.

<!-- formula-not-decoded -->

In this case, for any hypothesis f := (( P 0 , R 0 ) , ( P 1 , R 1 ) , . . . , ( P H -1 , R H -1 )) ∈ H , we can take the associated Q h,f and V h,f functions to be the optimal value functions corresponding to the transition kernels { P h } H -1 h =0 and reward functions { R h } H -1 h =0 .

Furthermore, we assume the hypothesis class is constrained so that V h,f ( s ) = max a Q h,f ( s, a ) for all f ∈ H , h ∈ [ H ] , and s ∈ S , which is always possible as we can remove hypothesis for which this is not true. We let π h,f be the greedy policy with respect to Q h,f , i.e., π h,f ( s ) = argmax a ∈A Q h,f ( s, a ) , and π f as the sequence of time-dependent policies { π h,f } H -1 h =0 .

## 4.1 Warmup: Bellman rank, the Q and V versions.

As a motivation for our structural framework, we next discuss Bellman rank framework considered in Jiang et al. [2017]. In this case, the hypothesis class H h contains Q value functions, i.e.,

<!-- formula-not-decoded -->

In this case, for any hypothesis f := ( Q 0 , Q 1 , . . . , Q H -1 ) ∈ H , we take the associated state-action value function Q h,f = Q h and the associated state value V h,f function to be greedy with respect to the Q h,f function i.e. V h,f ( · ) = max a ∈A Q h,f ( · , a ) .

Definition 4.1 ( V -Bellman Rank). A MDP has a V -Bellman rank of dimension d if for all h ∈ [ H ] , there exist functions W h : H → R d and X h : H → R d , such that for all

f, g ∈ H :

<!-- formula-not-decoded -->

Even though Jiang et al. [2017] only considered V -Bellman Rank, as a natural extension of this definition, we can also consider the Q -Bellman Rank.

Definition 4.2 ( Q -Bellman Rank). For a given MDP M , we say that our state-action value hypothesis class H has a Q -Bellman rank of dimension d if for all h ∈ [ H ] , there exist functions W h : H → R d and X h : H → R d , such that for all f, g ∈ H

Let us interpret how the two definitions differ in the usage of functions V h,f vs Q h,f (along with the usage of the 'estimation' policies a 0: h ∼ π f vs a 0: h -1 ∼ π f and a h ∼ π g ). Recall that the Bellman equations can be written in terms of the value functions or the state-action values; here, the intuition is that the former definition corresponds to enforcing Bellman consistency of the value functions while the latter definition corresponds to enforcing Bellman consistency of the state-action value functions. Our more general structural framework, Bilinear Classes, will cover both these definitions for infinite dimensional hypothesis class (note that Jiang et al. [2017] only considered finite dimensional hypothesis class).

<!-- formula-not-decoded -->

## 4.2 Bilinear Classes

We now introduce a new structural framework - the Bilinear Class.

Realizability. We say that H is realizable for an MDP M if, for all h ∈ [ H ] , there exists a hypothesis f /star ∈ H such that Q /star h ( s, a ) = Q h,f /star ( s, a ) , where Q /star h is the optimal stateaction value at time step h in the ground truth MDP M . For instance, for the model-based perspective, the realizability assumption is implied if the ground truth transition P belongs to our hypothesis class H .

Now we are ready to introduce the Bilinear Class.

Definition 4.3 (Bilinear Class). Consider an MDP M , a hypothesis class H , a discrepancy function /lscript f : ( R × S × A × S ) × H → R (defined for each f ∈ H ), and a set of estimation policies Π est = { π est ( f ) : f ∈ H} . We say ( H , /lscript f , Π est , M ) is ( implicitly )

a Bilinear Class if H is realizable in M and if there exist functions W h : H → V and X h : H → V for some Hilbert space V , such that the following two properties hold for all f ∈ H and h ∈ [ H ] :

## 1. We have:

<!-- formula-not-decoded -->

2. The policy π est ( f ) and discrepancy measure /lscript f ( o h , g ) can be used for estimation in the following sense: for any g ∈ H , we have that (here o h = ( r h , s h , a h , s h +1 ) is the 'observed transition info')

∣ ∣ E a 0: h -1 ∼ π f E a h ∼ π est ( f ) [ /lscript f ( o h , g ) ]∣ ∣ = |〈 W h ( g ) -W h ( f /star ) , X h ( f ) 〉| . (2) Typically, π est ( f ) will be either the uniform distribution on A or π f itself; in the latter case, we refer to the estimation strategy as being on-policy.

<!-- formula-not-decoded -->

We emphasize the above definition only assumes the existence of W and X functions. Particularly, our algorithm only uses the discrepancy function /lscript f , and does not need to know W or X . A typical example of discrepancy function /lscript f ( o h , g ) would be the bellman error Q h,g ( s h , a h ) -r h -V h +1 ,g ( s h +1 ) , but we would often need to use a different discrepancy function see for e.g. Linear Mixture Models (Section 4.3.1).

We now provide some intuition for definition of Bilinear Class. The first part of the definition (Equation (1)) basically relates the Bellman error for hypothesis f (and hence sub-optimality) to the sum of bilinear forms |〈 W h ( f ) -W h ( f /star ) , X h ( f ) 〉| (see for example proof of Lemma 5.5). Crucially, the second part of the definition (Equation (2)), allows us to 'reuse' data from hypothesis f to estimate the bilinear form |〈 W h ( g ) -W h ( f /star ) , X h ( f ) 〉| for all hypothesis g in our hypothesis class! This is reminiscent of uniform convergence guarantees in supervised learning, where data can be reused to simultaneously estimate the loss for all hypothesis and eliminate those with high loss.

## 4.2.1 Finite Bellman rank = ⇒ Bilinear Class

Here we show our framework naturally generalizes the Bellman rank framework (Section 4.1). For Q -bellman rank case, we define the discrepancy function /lscript f for observed transition info o h = ( r h , s h , a h , s h +1 ) as:

<!-- formula-not-decoded -->

Lemma 4.1 (Finite Q -Bellman Rank = ⇒ Bilinear Class). For given MDP M , suppose our hypothesis class H has a Q -Bellman rank of dimension d . Then, for on-policy estimation policies π est = π f , and the discrepancy function /lscript f defined above, ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Its straightforward to see that in this case, both Equation (1) and Equation (2) are satisfied.

In the V -Bellman rank setting, we define the discrepancy function /lscript f for observed transition info o h = ( r h , s h , a h , s h +1 ) as:

<!-- formula-not-decoded -->

Lemma 4.2 (Finite V -Bellman Rank = ⇒ Bilinear Class). For given MDP M , suppose our hypothesis class H has a V -Bellman rank of dimension d . Then, for uniform estimation policies π est = U ( A ) , and the discrepancy function /lscript f defined above, ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Note that for g = f , we have that for observed transition info o h = ( r h , s h , a h , s h +1 )

<!-- formula-not-decoded -->

Therefore, to prove that this is a Bilinear Class, we will show that a stronger 'equality' version of Equation (2) holds (which will also prove Equation (1) holds). Observe that for any h ,

<!-- formula-not-decoded -->

This completes the proof.

## 4.3 Examples

We now provide examples of Bilinear Classes: two known models (Linear Bellman Complete and Linear Mixture Models) and two new models that we propose (Linear Q /star /V /star and Low Occupancy Complexity). We return to these examples to give non-parametric sample complexities in Section 5.3. See Appendix A for additional examples of Bilinear Classes.

## 4.3.1 Linear Mixture MDP.

First, we show our definition naturally captures model-based hypothesis class.

Definition 4.4 (Linear Mixture Model). We say that a MDP M is a Linear Mixture Model if there exists (known) features φ : S × A × S ↦→ V and ψ : S × A ↦→ V ; and (unknown) θ /star ∈ V for some Hilbert space V such that for all h ∈ [ H ] and ( s, a, s ′ ) ∈ S × A × S

<!-- formula-not-decoded -->

We denote hypothesis in our hypothesis class H as tuples ( θ 0 , . . . θ H -1 ) , where θ h ∈ V . Recall that given a model f ∈ H (i.e. f is the time-dependent transitions, i.e., f h : S × A ↦→ ∆( S ) ), we denote V h,f as the optimal value function under model f and corresponding reward function (in this case defined by ψ ). Specifically, for any hypothesis g = { θ 0 , . . . , θ H -1 } ∈ H , V h,g and Q h,g satisfy the following Bellman optimality equation:

Note that in this example, discrepancy function will explicitly depend on f . For hypothesis g = { θ 0 , . . . , θ H -1 } ∈ H and observed transition info o h = ( r h , s h , a h , s h +1 ) , we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 4.3 (Linear Mixture Model = ⇒ Bilinear Class). Consider a MDP M which is a Linear Mixture Model. Then, for the hypothesis class H , discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Observe that for g = f , using Equation (3), for observed transition info o h = ( r h , s h , a h , s h +1 ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and therefore

We consider on-policy estimation π est = π f . To prove that linear mixture MDP is a Bilinear Class, we only need to show that an 'equality' version of Equation (2) holds

(which implies Equation (1) holds by the frame above). For g = { θ 0 , . . . , θ H -1 } ∈ H , observe:

where we defined the W h , X h functions as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes that Linear Mixture Model also forms a Bilinear Class.

## 4.3.2 Linear Q /star /V /star (new model)

We introduce a new model: linear Q /star /V /star where we assume both the optimal Q /star and V /star are linear functions in features that lie in (possibly infinite dimensional) Hilbert space.

Definition 4.5 (Linear Q /star /V /star ). We say that a MDP M is a linear Q /star /V /star model if there exist (known) features φ : S × A ↦→ V 1 , ψ : S ↦→ V 2 and (unknown) ( w /star , θ /star ) ∈ V 1 ×V 2 for some Hilbert spaces V 1 , V 2 such that for all h ∈ [ H ] and for all ( s, a, s ′ ) ∈ S ×A×S ,

<!-- formula-not-decoded -->

Here, our hypothesis class H = H 0 × . . . , H H -1 is a set of linear functions i.e. for all h ∈ [ H ] , the set H h is defined as:

We define the following discrepancy function /lscript f (in this case the discrepancy function does not depend on f ), for hypothesis g = { ( w h , θ h ) } H -1 h =0 and observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 4.4 (Linear Q /star /V /star = ⇒ Bilinear Class). Consider a MDP M which is a linear Q /star /V /star model. Then, for the hypothesis class H , the discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Note that we will show that a stronger 'equality' version of Equation (2) holds, which will also prove Equation (1) holds since for observed transition info o h = ( r h , s h , a h , s h +1 ) ,

<!-- formula-not-decoded -->

Observe that for any h

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

This concludes the proof.

## 4.3.3 Bellman Complete and Linear MDPs

We now consider Bellman Complete which captures the linear MDP model (see Appendix A.4 for more detail on linear MDP model). Here, our hypothesis class H is set of linear functions with respect to some (known) feature φ : S × A ↦→ V , where V is a Hilbert space. We denote hypothesis in our hypothesis class H as tuples ( θ 0 , . . . θ H -1 ) , where θ h ∈ V .

Definition 4.6 (Linear Bellman Complete). We say our hypothesis class H is Linear Bellman Complete with respect to M if H is realizable and there exists T h : V → V such that for all ( θ 0 , . . . θ H -1 ) ∈ H and h ∈ [ H ] ,

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A .

We define the following discrepancy function /lscript f (in this case the discrepancy function does not depend on f ), for hypothesis g = ( θ 0 , . . . , θ H -1 ) and observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

Lemma 4.5 (Linear Bellman Complete = ⇒ Bilinear Class). Consider an MDP M and hypothesis class H such that H is Linear Bellman Complete with respect to M . Then, for on-policy estimation policies π est ( f ) = π f and the discrepancy function /lscript f defined above, ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Note that in this case, we will show that a stronger version of Equation (2) holds i.e with equality instead of ≤ inequality, which will also prove Equation (1) holds since for observed transition info o h = ( r h , s h , a h , s h +1 ) ,

<!-- formula-not-decoded -->

Observe that for any h

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Observe that W h ( f /star ) = 0 for all h .

## 4.3.4 Low Occupancy Complexity (new model).

We introduce another new model: Low Occupancy Complexity .

Definition 4.7 (Low Occupancy Complexity). We say that a MDP M and hypothesis class H has low occupancy complexity with respect to a (possibly unknown) feature mapping φ h : S × A → V (where V is a Hilbert space) if H is realizable and there exists a (possibly unknown) β h : H ↦→ V for h ∈ [ H ] such that for all f ∈ H and ( s h , a h ) ∈ S ×A we have that:

<!-- formula-not-decoded -->

It is important to emphasize that for this hypothesis class, we are only assuming realizability, but it is otherwise arbitrary (e.g. it could be a neural state-action value class) and the algorithm does not need to know the features φ h nor β h . It is straight forward to see that such a class is Bilinear Class with discrepancy function /lscript f defined for hypothesis g ∈ H and observed transition info o h = ( r h , s h , a h , s h +1 ) as,

<!-- formula-not-decoded -->

Lemma 4.6 (Low Occupancy Complexity = ⇒ Bilinear Class). Consider a MDP M and hypothesis class H which has low occupancy complexity. Then, for the the discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. To see why this is a Bilinear Class, as in previous proofs, we will show that an 'equality' version of Equation (2) holds, which will also prove Equation (1) holds since

<!-- formula-not-decoded -->

Observe that for any h (here observed transition info o h = ( r h , s h , a h , s h +1 ) ):

<!-- formula-not-decoded -->

where the notation E [ V ( s h +1 ) | s h , a h ] is shorthand for E s h +1 ∼ P h ( s h ,a h ) [ V ( s h +1 )] and we defined the W h , X h functions as follows:

<!-- formula-not-decoded -->

Note that W h ( f /star ) = 0 . This completes the proof.

Note that as such the hypothesis class H could be arbitrary and unlike other models where we assume linearity, here it could be a neural state-action value class. Our model can also capture the setting where the state-only occupancy has low complexity, i.e., d π f ( s h ) = β h ( f ) µ h ( s h ) , for some µ h : S → V . In this case, we will use π est = U ( A ) .

## Algorithm 1: BiLin-UCB

- 1: Input : number of iterations T , estimator function /lscript , batch size m , confidence radius R
- 2: for iteration t = 0 , 1 , 2 , . . . , T -1 do
- 3: Set f t as the solution of the following program:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 4: For all h ∈ [ H ] , create batch datasets D t ; h = { ( r i h , s i h , a i h , s i h +1 ) } m -1 i =0 sampled from distribution induced by a 0: h -1 ∼ d π f t and a h ∼ π est .
- 5: end for
- 6: return max t ∈ [ T ] V π f t .

## 5 The Algorithm and Theory

Our algorithm, BiLin-UCB, is described in Algorithm 1, which takes three parameters as inputs, the number of iterations T , the trajectory batch size m per iteration and a confidence radius R . The key component of the algorithm is a constrained optimization in Line 3. For each time step h , we use all previously collected data to form a single constraint using /lscript f . The constraint refines the original version space H to be a restricted version space containing only hypothesis that are consistent with the current batch data. We then perform an optimistic optimization: we search for a feasible hypothesis g that achieves the maximum total reward V g ( s 0 ) .

There are two ways to collect batch samples. For the case where π est = π f t , then for data collection in Line 4, we can generate m length-H trajectories by executing π f t starting from s 0 . For the general case (e.g. consider setting π est to be a uniform distribution over A ), we gather the data for each h ∈ [ H ] independently. For h ∈ [ H ] , we first roll-in with π f t to generate s h ; then execute a h ∼ π est ; and then continue to generate s h +1 ∼ P h ( ·| s h , a h ) and r h ∼ R ( ·| s h , a h ) . Repeating this process for all h , we need Hm trajectories to form the batch datasets {D t ; h } H -1 h =0 .

## 5.1 Main Theory: Generalization in Bilinear Classes

Wenowpresent our main result. We first define some notations. We denote the expectation of the function /lscript f ( · , g ) under distribution µ over R ×S ×A×S by

<!-- formula-not-decoded -->

For a set D ⊂ S × A × S , we will also use D to represent the uniform distribution over this set.

Assumption 5.1 (Ability to Generalize). We assume there exists functions ε gen ( m, H ) and conf ( δ ) such that for any distribution µ over R ×S×A×S and for any δ ∈ (0 , 1 / 2) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m ,

<!-- formula-not-decoded -->

Remark 5.1. It is helpful to separate the dependence of generalization error on failure probability δ and number of samples m in order to state Theorem 5.2 concisely. ε gen ( m, H ) is related to uniform convergence and measures the generalization error of hypothesis class H and for the hypothesis classes discussed in this paper, ε gen ( m, H ) → 0 as m → ∞ . One example is when π est = π f , and H is a discrete function class, then we have ε gen ( m, H ) = O ( √ (1 + ln( |H| )) /m. ) . In Appendix D, we also discuss uniform convergence via a novel covering argument for infinite dimensional RKHS.

Recall the definitions X h := { X h ( f ): f ∈ H} and X := {X h : h ∈ [ H ] } . We first present our main theorem for the finite dimensional case i.e. when X h ⊂ R d for all timesteps h .

Theorem 5.1. (Finite-dimensional case) Suppose ( H , /lscript, Π est , M ) is a Bilinear Class with X h ⊂ R d for all timesteps h and Assumption 5.1 holds. Assume sup f ∈H ,h ∈ [ H ] ‖ W h ( f ) ‖ 2 ≤ B W and sup f ∈H ,h ∈ [ H ] ‖ X h ( f ) ‖ 2 ≤ B X . Fix δ ∈ (0 , 1 / 3) and batch sample size m and define:

Set the parameters as: number of iterations T = ˜ d m and confidence radius R = √ Tε gen ( m, H ) · conf ( δ/ ( TH )) . With probability at least 1 -δ , Algorithm 1 uses at most mHT trajectories and returns a hypothesis f such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As discussed in the Remark 5.1, ε gen ( m, H ) and conf ( δ ) measure the uniform convergence of discrepancy functions /lscript f for the hypothesis class H . Therefore, if ε gen ( m, H ) decays at least as fast as m -α for any constant α , we will get efficient reinforcement learning. In fact, we will see in our examples (Section 5.3), that this is true for all known models where efficient reinforcement learning is possible. One such example is finite hypothesis classes where we immediately get the following sample complexity bound showing only a logarithmic dependence on the size of the hypothesis space.

Corollary 5.1. (Finite-dimensional, Finite Hypothesis Case) Suppose ( H , /lscript, Π est , M ) is a Bilinear Class with X h ⊂ R d for all timesteps h , |H| &gt; 1 and Assumption 5.1 holds. Assume sup f ∈H ,h ∈ [ H ] ‖ W h ( f ) ‖ 2 ≤ B W and sup f ∈H ,h ∈ [ H ] ‖ X h ( f ) ‖ 2 ≤ B X for some B X , B W ≥ 1 . Assume the discrepancy function /lscript f is bounded i.e. sup f ∈H | /lscript f ( · ) | ≤ H + 1 . Fix δ ∈ (0 , 1 / 3) and /epsilon1 ∈ (0 , 1) . Then there exists absolute constants c 1 , c 2 , c 3 , c 4 such that setting the parameters: batch sample size

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

number of iterations T = c 2 dH ln ( B X B W m ) and confidence radius R = c 3 √ T · H √ ln( |H| ) /m · ln( TH/δ ) , with probability at least 1 -δ , Algorithm 1 returns a hypothesis f such that V /star ( s 0 ) -V π f ( s 0 ) ≤ /epsilon1 using at most trajectories.

The proof for this corollary follows from bounds on ε gen ( m, H ) and conf ( δ ) using Hoeffding's inequality (Lemma F.1). We present the complete proof in Appendix B.

Our next results will be non-parametric in nature and therefore it is helpful to introduce the maximum information gain [Srinivas et al., 2009], which captures an important notion of the effective dimension of a set. Let X ⊂ V , where V is a Hilbert space. For λ &gt; 0 and integer n &gt; 0 , the maximum information gain γ n ( λ ; X ) is defined as:

<!-- formula-not-decoded -->

If X is of the form X = {X h : h ∈ [ H ] } , we use the notation

<!-- formula-not-decoded -->

Define critical information gain , denoted by ˜ γ ( λ ; X ) , as the smallest integer k &gt; 0 s.t. k ≥ γ k ( λ ; X ) , i.e.

(where k is an integer). Note that such a ˜ γ ( λ ; X ) exists provided that the information gain γ n ( λ ; X ) has a sufficiently mild growth condition in both n and 1 /λ . The critical information gain can viewed as an analogous quantity to the critical radius , a quantity which arises in non-parametric statistics [Wainwright, 2019].

<!-- formula-not-decoded -->

Remark 5.2. For finite dimension setting where X ⊂ R d and ‖ x ‖ ≤ B X for any x ∈ X , we have: γ n ( λ ; X ) ≤ d ln (1 + nB 2 X /dλ ) and ˜ γ ( λ ; X ) ≤ 3 d ln (1 + 3 B 2 X /λ ) (see Lemma F.3 for a proof). Note that 1 /λ , n, and the norm bound B X only appear inside the log. Furthermore, it is possible that γ n ( λ ; X ) is much smaller than the dimension of X (or V ), when the eigenspectrum of the covariance matrices concentrates in a low-dimension subspace. In fact when X belongs to some infinite dimensional RKHS, γ n ( λ ; X ) could still be small [Srinivas et al., 2009].

We now present our main theorem. Recall the definitions X h := { X h ( f ): f ∈ H} and X := {X h : h ∈ [ H ] } .

Theorem 5.2. (RKHS case) Suppose ( H , /lscript, Π est , M ) is a Bilinear Class and Assumption 5.1 holds. Assume sup f ∈H ,h ∈ [ H ] ‖ W h ( f ) ‖ 2 ≤ B W . Fix δ ∈ (0 , 1 / 3) , batch sample size m , and define:

˜ Set the parameters as: number of iterations T = ˜ d m and confidence radius R = √ ˜ d m ε gen ( m, H ) · conf ( δ/ ( ˜ d m H )) . With probability at least 1 -δ , Algorithm 1 uses at most mH ˜ d m trajectories and returns a hypothesis f such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Next, we provide an elementary and detailed proof for our main theorem using an elliptical potential argument.

## 5.2 Proof of Theorem 5.1 and Theorem 5.2

In this subsection, we prove our main theorems - Theorem 5.1 and Theorem 5.2.

Notation To simplify notation, we denote by µ t ; h the distribution induced over S×A×S by a 0: h -1 ∼ d π f t and a h ∼ π est ; D t ; h the batch dataset collected from distribution µ t ; h ; ε gen the generalization error ε gen ( m, H ) · conf ( δ/ ( TH )) . Also, recall that for any distribution µ over R ×S × A × S and hypothesis f, g ∈ H

<!-- formula-not-decoded -->

Note that throughout the proof unless specified, the statements are true for any fixed δ ∈ (0 , 1) , integer m &gt; 0 and integer T &gt; 0 . Also, we set R = √ Tε gen throughout the proof. To simplify the proof, we will condition on the event that uniform convergence of /lscript holds throughout our algorithm, which we first show holds with high probability.

Lemma 5.1 (Uniform Convergence). For all t ∈ [ T ] and g ∈ H and h ∈ [ H ] , with probability at least 1 -δ , we have:

Proof. This follows from the uniform convergence (Assumption 5.1) and then union bounding over all t ∈ [ T ] and h ∈ [ H ] .

<!-- formula-not-decoded -->

We start by presenting our main lemma which shows if uniform convergence of /lscript holds throughout our algorithm, our algorithm finds a near-optimal policy. This lemma will be enough to prove our main results.

Lemma 5.2 (Existence of high quality policy). Suppose we run the algorithm for T iterations. Set R = √ Tε gen. Assume the event in Lemma 5.1 holds and sup f ∈H ‖ W h ( f ) ‖ 2 ≤ B W for all h ∈ [ H ] . Then, for all λ ∈ R + , there exists t ∈ [ T ] such that the following is true for hypothesis f t :

<!-- formula-not-decoded -->

Wenowcompletethe proof of Theorem 5.1 and Theorem 5.2 using Lemma 5.1, Lemma 5.2 and setting the parameters using the definition of critical information gain.

Proof of Theorem 5.1 and Theorem 5.2. Fix λ = ε 2 gen ( m, H ) /B 2 W . From definition of critical information gain (Equation (6)), it follows that for T = ˜ γ ( λ, X ) , T ≥ γ T ( λ, X )

Using Lemma 5.2, we get that

<!-- formula-not-decoded -->

Observing that for our choice of T , γ T ( λ ; X ) /T ≤ 1 and e -1 &lt; 2 , we get

Moreover, each iteration of the algorithm, takes only mH trajectories, this gives the total trajectories as mHT = mH ˜ γ ( λ, X ) . This proves Theorem 5.2. Theorem 5.1 follows from the upper bound on γ ( λ, X ) for finite dimensional X h using Lemma F.3.

<!-- formula-not-decoded -->

˜ In the rest of the section, we will prove our main lemma - Lemma 5.2. The first step shows that under Assumption 5.1, our R is set properly so that f /star is always a feasible solution of the constrained optimization program in Algorithm 1.

Lemma5.3 (Feasibility of f /star ). Assume the event in Lemma 5.1 holds. Then for all t ∈ [ T ] , we have that f /star is always a feasible solution.

Proof. Note that L µ i ; h ,f i ( f ∗ ) = 0 (Equation (2)). Thus using Lemma 5.1, we have:

<!-- formula-not-decoded -->

Noting that t ≤ T and in our parameter setup R = √ Tε gen completes the proof.

The feasibility result immediately leads to optimism.

Lemma 5.4 (Optimism). Assume the event in Lemma 5.1 holds. Then for all t ∈ [ T ] , we have V /star ≤ V f t ;0 ( s 0 ) .

Proof. Lemma 5.3 implies f /star is a feasible solution for the optimization program for all t ∈ [ T ] . This proves the claim.

The following lemma relates the sub-optimality to a sum of bilinear forms. Using the performance difference lemma, we first show that sub-optimality is upper bounded by the Bellman errors of Q h,f t , which are further upper bounded by sum of bilinear forms via our assumption (Equation (1)).

Lemma 5.5 (Bilinear Regret Lemma). Assume the event in Lemma 5.1 holds. Then, the following holds for all t ∈ [ T ] :

<!-- formula-not-decoded -->

Proof. We can upper bound the regret

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step follows Equation (1) in the Bilinear Class definition.

The following is a variant of the Elliptical Potential Lemma, central in the analysis of linear bandits [Dani et al., 2008, Srinivas et al., 2009, Abbasi-Yadkori et al., 2011].

Lemma5.6(Elliptical potential). Consider any sequence of vectors { x 0 , . . . , x T -1 } where x i ∈ V for some Hilbert space V . Let λ ∈ R + . Denote Σ 0 = λI and Σ t = Σ 0 + ∑ t -1 i =0 x i x /latticetop i . We have that:

Proof. By definition of Σ t and matrix determinant lemma, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using recursion completes the proof.

Now, we will finish the proof of Lemma 5.2 by showing that the sum of bilinear forms in Lemma 5.5 is small for at least for one t ∈ [ T ] . More precisely, using Equation (2) together with elliptical potential argument (Lemma 5.6), we can show that after ˜ d m many iterations, we must have found a policy π f t such that |〈 W h ( f t ) -W h ( f /star ) , X h ( f t ) 〉| is small for all h .

Proof of Lemma 5.2. Our goal (as per Lemma 5.5 and Equation (1)) is to find t ∈ [ T ] such that

<!-- formula-not-decoded -->

To that end, we will show that

<!-- formula-not-decoded -->

for appropriately chosen A . We will show existence of such X h ( f t ) and A (Equation (7)) using the potential argument (Lemma 5.6) and conditions on W h ( f t ) -W h ( f /star ) follow from our optimization program. We now show this in more detail.

Let the hypothesis used by our algorithm at i th iteration be f i . Consider the corresponding sequence of representations { X h ( f i ) } i,h . Then, by Lemma 5.6, we have that for all h ∈ [ H ] and λ ∈ R +

where we have used definition of maximum information gain γ T ( λ ; X h ) (Equation (4)) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summing these inequalities over all h ∈ [ H ] , we have that for all λ ∈ R +

<!-- formula-not-decoded -->

where the last equality follows from Equation (5). Since, each of these terms is ≥ 0 , we get that there exists t ∈ [ T ] such that

<!-- formula-not-decoded -->

Again, since each of these terms is ≥ 0 , we get that for all h ∈ [ H ]

and simplifying, we get that for all h ∈ [ H ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also, by construction of our program, for all iterations and in particular for t , it holds that for all h ∈ [ H ]

and by Lemma 5.1, for all h ∈ [ H ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from ( a + b ) 2 ≤ 2 a 2 +2 b 2 and the last step follows from the frame above and t ∈ [ T ] . Using the definition of Bilinear Class (Equation (2)), for all h ∈ [ H ]

<!-- formula-not-decoded -->

Using this, we get for all h ∈ [ H ]

<!-- formula-not-decoded -->

where the first inequality follows from the frame above and definition of Σ t ; h . Using Equation (7) and the frame above, this immediately shows that for all h ∈ [ H ]

<!-- formula-not-decoded -->

Summing over all h ∈ [ H ] , this gives

Using Lemma 5.5, this gives the desired result.

<!-- formula-not-decoded -->

## 5.3 Corollaries for Particular Models

In this section, we apply our main theorem to special models: linear Q /star /V /star , RKHS bellman complete, RKHS linear mixture model, and low occupancy complexity model. While linear bellman complete and linear mixture model have been studied, our results extends to infinite dimensional RKHS setting.

## 5.3.1 Linear Q /star /V /star

In this subsection, we provide the sample complexity result for the linear Q /star /V /star model (Definition 4.5). To state our results for linear Q /star /V /star , we define the following sets:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and define the concatenation set 2

We first provide the result for the finite dimensional case i.e. when Φ ◦ Ψ ⊂ R d .

2 For infinite dimensional Φ and Ψ , we consider the natural inner product space where 〈 [ x 1 , y 1 ] , [ x 2 , y 2 ] 〉 = 〈 x 1 , x 2 〉 + 〈 y 1 , y 2 〉 .

Corollary 5.2 (Finite Dimensional Linear Q /star /V /star ). Suppose MDP M is a linear Q /star /V /star model with Φ ◦ Ψ ⊂ R d . Assume sup ( w,θ ) ∈H h ,h ∈ [ H ] ‖ [ w, θ ] ‖ 2 ≤ B W and sup x ∈ Φ ◦ Ψ ‖ x ‖ 2 ≤ B X for some B X , B W ≥ 1 . Fix δ ∈ (0 , 1 / 3) and /epsilon1 ∈ (0 , H ) . There exists an appropriate setting of batch sample size m , number of iteration T and confidence radius R such that with probability at least 1 -δ , Algorithm 1 returns a hypothesis f such that V /star ( s 0 ) -V π f ( s 0 ) ≤ /epsilon1 using at most

<!-- formula-not-decoded -->

trajectories for some absolute constant c 1 , c 2 .

To prove this, we will prove a more general sample complexity result for the infinite dimensional RKHS case.

Corollary 5.3 (RKHS Linear Q /star /V /star ). Suppose MDP M is a linear Q /star /V /star model. Assume sup ( w,θ ) ∈H h ,h ∈ [ H ] ‖ [ w, θ ] ‖ 2 ≤ B W and sup x ∈ Φ ◦ Ψ ‖ x ‖ 2 ≤ B X . Fix δ ∈ (0 , 1 / 3) , batch sample size m , and define:

<!-- formula-not-decoded -->

where ν := ln ( 1 + 3 B X B W √ m γ ( 1 8 B 2 W m ; Φ ◦ Ψ ) ) .

<!-- formula-not-decoded -->

˜ Set the parameters as: R = (12 H/ √ m ) √ ˜ d m ( X ) · ˜ d m (Φ ◦ Ψ) · √ ln ( ( ˜ d m ( X ) H ) /δ ) and T = ˜ d m ( X ) . With probability greater than 1 -δ , Algorithm 1 uses at most mH ˜ d m ( X ) trajectories and returns a hypothesis f :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First, using Corollary D.3, we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m

of size m , for all g = ([ w 0 , θ 0 ] , . . . , [ w H -1 , θ H -1 ]) ∈ H (note that L µ ( g ) only depends on [ w h , θ h ] for distribution µ over observed transitions o h = ( r h , s h , a h , s h +1 ) at timestep h .)

<!-- formula-not-decoded -->

where we have used that ln(1 /δ ) &gt; 1 and ˜ γ m = ˜ γ (1 / (8 B 2 W m ); Φ ◦ Ψ) (as defined in Equation (6)). Define

This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting this in Theorem 5.2 gives the result

<!-- formula-not-decoded -->

Next, we complete the proof of Corollary 5.2. Note that both ˜ d m (Φ ◦ Ψ) and ˜ d m ( X ) (related to critical information gain under Φ and X respectively) scale as ˜ O ( d ) if Φ ◦ Ψ ⊂ R d .

Proof of Corollary 5.2. First, from Lemma F.3, we have that and substituting this in Equation (9)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, as sup z ∈X ‖ z ‖ ≤ sup x ∈ Φ ◦ Ψ ‖ x ‖ , using Lemma F.3 and similar analysis as above (and 144 H 2 ˜ d m (Φ ◦ Ψ) ≥ 1 ), we get and substituting this in Equation (10)

<!-- formula-not-decoded -->

To get /epsilon1 -optimal policy (from Equation (11)), we have to set

<!-- formula-not-decoded -->

Further upper bounding the right hand side of the above inequality by substituting in upper bounds for ˜ d m ( X ) and d m (Φ ◦ Ψ) from frames above, we can set m to be as large as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Lemma F.2 for α = 4 , a = 32 · (72) 2 d 2 H 5 ln(1 /δ ) //epsilon1 2 , b = 25 B 2 X B 2 W dH 2 and c = 5 4 , we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting this in the expression above for ˜ d m ( X ) and setting this upper bound to T , we get

Since, we use on policy estimation, i.e., π est = π f t for all t , the trajectory complexity is mT which completes the proof.

## 5.3.2 RKHS Bellman Complete.

In this subsection, we provide the sample complexity result for the Linear Bellman Complete model (Definition 4.6). To state our results, we define

<!-- formula-not-decoded -->

Corollary 5.4 (Finite Dimensional Linear Bellman Complete). Suppose H is Bellman Complete with respect to MDP M for some Hilbert space V ⊂ R d . Assume sup θ ∈H h ,h ∈ [ H ] ‖ θ ‖ 2 ≤ B W and sup x ∈ Φ ‖ x ‖ 2 ≤ B X for some B X , B W ≥ 1 . Fix δ ∈ (0 , 1 / 3) and /epsilon1 ∈ (0 , H ) . There exists an appropriate setting of batch sample size m , number of iteration T and confidence radius R such that with probability at least 1 -δ , Algorithm 1 returns a hypothesis f such that V /star ( s 0 ) -V π f ( s 0 ) ≤ /epsilon1 using at most

We first provide the result for the finite dimensional case i.e. when Φ ⊂ V ⊂ R d .

<!-- formula-not-decoded -->

trajectories for some absolute constant c 1 , c 2 .

In comparison, Jin et al. [2020] has sample complexity ˜ O ( d 3 H 3 //epsilon1 2 log(1 /δ )) and Zanette et al. [2020] has ˜ O ( d 2 H 3 //epsilon1 2 log(1 /δ )) . To prove this, we will prove a more general sample complexity result for the infinite dimensional RKHS case. Note that RKHS Linear MDP is a special instance of RKHS Bellman Complete. Prior works that studied RKHS Linear MDP either achieves worse rate [Agarwal et al., 2020a] or further assumes finite covering dimension of the space of all possible upper confidence bound Q functions which are algorithm dependent quantities [Yang et al., 2020].

Corollary 5.5 (RKHS Bellman Complete). Suppose H is Bellman Complete with respect to MDP M for some Hilbert space V . Assume sup h ∈ [ H ] ,θ ∈H h ‖ θ ‖ 2 ≤ B W and sup x ∈ Φ ‖ x ‖ 2 ≤ B X . Fix δ ∈ (0 , 1 / 3) , batch sample size m , and define:

<!-- formula-not-decoded -->

˜ Set the parameters as: R = (12 H/ √ m ) √ ˜ d m ( X ) · ˜ d m (Φ) · √ ln ( ( ˜ d m ( X ) H ) /δ ) and T = ˜ d m ( X ) . With probability at least 1 -δ , Algorithm 1 uses at most mH ˜ d m ( X ) trajectories and returns a hypothesis f :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Proof. First, using Corollary D.2, we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all g = ( θ 0 , . . . , θ H -1 ) ∈ H (note that L µ ( g ) only depends on θ h for distribution µ over observed transitions o h = ( r h , s h , a h , s h +1 ) at timestep h .)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used that ln(1 /δ ) &gt; 1 and ˜ γ m = ˜ γ (1 / (8 B 2 W m ); Φ) (as defined in Equation (6)). Define

This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

Substituting this in Theorem 5.2 gives the result

<!-- formula-not-decoded -->

Wenowcomplete the proof of Corollary 5.4. Note that both ˜ d m (Φ) and ˜ d m ( X ) (related to critical information gain under Φ and X respectively) scale as O ( d ) if Φ ⊂ R d .

˜ Proof of Corollary 5.4. Since the proof follows similar to proof of Corollary 5.2, we will only provide a proof sketch here. First, from Lemma F.3, we have that and therefore

<!-- formula-not-decoded -->

˜ Similarly, as sup z ∈X ‖ z ‖ ≤ sup x ∈ Φ ‖ x ‖ , using Lemma F.3 (and since 400 H 2 ˜ d m (Φ) ≥ 1 ), we get and therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To get /epsilon1 -optimal policy, we have to set

<!-- formula-not-decoded -->

The rest of the proof follows similarly to proof of Corollary 5.2.

## 5.3.3 RKHS linear mixture model

In this subsection, we provide the sample complexity result for the Linear Mixture model (Definition 4.4). To present our sample complexity results, we define:

<!-- formula-not-decoded -->

Corollary 5.6 (Finite Dimensional Linear Mixture Model). Suppose MDP M is a linear Mixture Model for some Hilbert space V ⊂ R d . Assume sup θ ∈H h ,h ∈ [ H ] ‖ θ ‖ 2 ≤ B W and sup x ∈ Φ h ,h ∈ [ H ] ‖ x ‖ 2 ≤ B X for some B X , B W ≥ 1 . Fix δ ∈ (0 , 1 / 3) and /epsilon1 ∈ (0 , H ) . There exists an appropriate setting of batch sample size m , number of iteration T and confidence radius R such that with probability at least 1 -δ , Algorithm 1 returns a hypothesis f such that V /star ( s 0 ) -V π f ( s 0 ) ≤ /epsilon1 using at most

We first provide the result for the finite dimensional case i.e. when Φ h ⊂ V ⊂ R d for all h ∈ [ H ] .

<!-- formula-not-decoded -->

trajectories for some absolute constant c 1 , c 2 .

In comparison, Modi et al. [2020a] has sample complexity ˜ O ( d 2 H 2 //epsilon1 2 log(1 /δ )) . To prove this, we will prove a more general sample complexity result for the infinite dimensional RKHS case. We omit proof of Corollary 5.6 since it follows same as proof of Corollary 5.2.

Corollary 5.7 (RKHS linear mixture model). Suppose MDP M is a linear Mixture Model . Assume sup θ ∈H h ,h ∈ [ H ] ‖ θ ‖ 2 ≤ B W and sup x ∈ Φ h ,h ∈ [ H ] ‖ x ‖ 2 ≤ B X . Fix δ ∈ (0 , 1 / 3) , batch sample size m , and define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ ˜ ˜ T = ˜ d m ( X ) . With probability greater than 1 -δ , Algorithm 1 uses at most mH ˜ d m ( X ) trajectories and returns a hypothesis f

<!-- formula-not-decoded -->

Proof. First, using Corollary D.3 and Lemma F.1, we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all g = ( θ 0 , . . . , θ H -1 ) ∈ H (note that L µ ( g ) only depends on θ h for distribution µ over observed transitions o h = ( r h , s h , a h , s h +1 ) at timestep h .)

<!-- formula-not-decoded -->

where we have used that ln(1 /δ ) &gt; 1 and ˜ γ m = max h ∈ [ H ] ˜ γ (1 / (8 B 2 W m ); Φ h ) (as defined in Equation (6)). Define

This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting this in Theorem 5.2 gives the result

<!-- formula-not-decoded -->

## 5.3.4 Low Occupancy Complexity

Recall the low occupancy complexity model in Definition 4.7.

Corollary 5.8 (Low Occupancy Complexity). Suppose H has low occupancy complexity . Assume sup f ∈H h ,h ∈ [ H ] ‖ W h ( f ) ‖ 2 ≤ B W . Fix δ ∈ (0 , 1 / 3) , batch sample size m , and define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ ˜ ˜ With probability greater than 1 -δ , Algorithm 1 uses at most mH ˜ d m ( X ) trajectories and returns a hypothesis f such that:

<!-- formula-not-decoded -->

Proof. First, using Lemma F.1, we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of

size m , for all g ∈ H

This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting this in Theorem 5.2 gives the result

<!-- formula-not-decoded -->

## 5.3.5 Finite Bellman Rank

In this section, we will prove sample complexity bounds for MDPs with finite Bellman Rank introduced in Jiang et al. [2016] (also defined as V -Bellman rank in Section 4.1).

Corollary 5.9 (Bellman Rank). For a given MDP M , suppose a hypothesis class H has Bellman rank d . Assume sup f ∈H h ,h ∈ [ H ] ‖ W h ( f ) ‖ 2 ≤ B W and sup f ∈H ,h ∈ [ H ] ‖ X h ( f ) ‖ ≤ B X for some B W , B X ≥ 1 . Fix δ ∈ (0 , 1 / 3) and /epsilon1 ∈ (0 , H ) . There exists an appropriate setting of batch sample size m , number of iteration T and confidence radius R such that

with probability at least 1 -δ , Algorithm 1 returns a hypothesis f such that V /star ( s 0 ) -V π f ( s 0 ) ≤ /epsilon1 using at most trajectories for some absolute constant c 1 , c 2 .

<!-- formula-not-decoded -->

Note that in comparison, Jiang et al. [2016] has sample complexity ˜ O ( d 2 H 5 |A| //epsilon1 2 log(1 /δ )) . We now present the proof.

<!-- formula-not-decoded -->

Proof. First, as observed in Jiang et al. [2016][Lemma 14], we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all g ∈ H

where the second inequality holds as long as m &gt; 2 H |A| ln( |H| /δ ) . This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

Substituting this in Theorem 5.2 gives the result

<!-- formula-not-decoded -->

where the second last step follows from Lemma F.3. Substituting ε gen and conf in Theorem 5.2 also gives

<!-- formula-not-decoded -->

To get /epsilon1 -optimal policy, we have to set

<!-- formula-not-decoded -->

Further simplifying the RHS, we can write it as

<!-- formula-not-decoded -->

Using Lemma F.2 for α = 2 , a = 4608 dH 5 |A| (1 + ln( |H| )) //epsilon1 2 , b = 16 dH 2 B 2 W B 2 X /δ and c = 9 , we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting this in the expression above for ˜ d m ( X ) and setting this upper bound to T , we get

Since, we use on policy estimation, i.e., π est = U ( A ) for all t , the trajectory complexity is mTH which completes the proof.

## 6 Extended Bilinear Classes

While Bilinear Classes captures most existing models, in this section, we discuss several straightforward extensions of it to incorporate additional models such as Kernelized Nonlinear Regulator (KNR), generalized linear Bellman complete model, and Witness Rank.

Consider two nonlinear monotone transformations ξ : R ↦→ R , ζ : R ↦→ R , and a set of discriminator classes {F h } H -1 h =0 where F h ⊂ S × A × S ↦→ R . Denote F as the union of all discriminators F h from h = 0 to H -1 . We extend Bilinear Class to the following new definition, Generalized Bilinear Class .

Definition 6.1 (Generalized Bilinear Class). Consider an MDP M , a hypothesis class H , a discrepancy function /lscript f : R × S × A × S × H × F → R (defined for f ∈ H ), a set of estimation policies Π est = { π est ( f ) : f ∈ H} , and two non-decreasing functions ξ, ζ : R ↦→ R with ξ (0) = 0 , ζ (0) = 0 , and discriminator classes {F h } H -1 h =0 .

We say ( H , /lscript f , Π , M ) is ( implicitly ) a Generalized Bilinear Class if H is realizable in M and if there exist functions W h : H×H→V and X h : H → V for some Hilbert space V , such that the following two properties hold for all f ∈ H and h ∈ [ H ] :

## 1. We have:

<!-- formula-not-decoded -->

2. The policy π est ( f ) and discrepancy measure /lscript f ( o h , g, v ) can be used for estimation in the following sense: for any g ∈ H , we have that (here o h = ( r h , s h , a h , s h +1 ) is the 'observed transition info')

<!-- formula-not-decoded -->

∣ ∣ Typically, π est ( f ) will be either the uniform distribution on A or π f itself; in the latter case, we refer to the estimation strategy as being on-policy.

<!-- formula-not-decoded -->

We also define X h := { X h ( f ): f ∈ H} and X := {X h : h ∈ [ H ] } .

Below we dive into the details of the the new definition and the examples it captures, we first see how this new definition generalizes Bilinear Class. To see that, note that we just need to set ξ and ζ to be identity function, and set the discriminator classes F h = ∅ for all h ∈ [ H ] (i.e. ignore ν in the discrepancy measure /lscript f ).

We make the following assumptions on the two nonlinear transformations. We assume the slope of ζ is lower bounded, and ξ is non-decreasing and concave. Similar assumption has been used in generalized linear bandit model (e.g, Russo and Van Roy [2014]).

Assumption 6.1. For ζ , we assume ζ (0) = 0 and ζ is continuously differentiable, and

<!-- formula-not-decoded -->

For ξ , we assume ξ (0) = 0 , and ξ is concave and non-decreasing.

We again rely on a reduction to supervised learning style generalization error by extending assumption 5.1 to the following new assumption such that it now includes the additional function class F h .

<!-- formula-not-decoded -->

We denote the expectation of the function /lscript f ( · , g, ν ) under distribution µ over R ×S× A×S by

For a set D ⊂ R ×S × A × S , we will also use D to represent the uniform distribution over this set.

Assumption 6.2 (Ability to Generalize). We assume there exists functions ε gen ( m, H , F ) and conf ( δ ) such that for any distribution µ over R ×S×A×S and for any δ ∈ (0 , 1 / 2) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m ,

One simple example of the ε gen ( m, H , F ) is when H and F are both discrete, ε gen ( m, H , F ) will scale in the order of ˜ O ( √ ln( |H||F| ) /m ) via standard uniform convergence analysis. With the above assumptions, we can show that our algorithm achieves the following regret.

<!-- formula-not-decoded -->

Theorem 6.1. For Generalized Bilinear Class under Assumption 6.1, setting parameters properly, we have that with probability at least 1 -δ :

˜ Furthermore, if ξ is differentiable and has slope being upper bounded, i.e., ∃ α ∈ R + such that max f,g,h ξ ′ ( 〈 W h ( g ) -W h ( f /star ) , X h ( f ) 〉 ) ≤ α , then we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ The proof of the above theorem largely follows the proof of Theorem 5.2, and is deferred to Appendix E.

## 6.1 Kernelized Nonlinear Regulator (KNR)

In this section, we show how the above definition captures KNR [Kakade et al., 2020] which we define next. We note that neither Bellman rank nor Witness rank could capture KNR directly. Specifically, since φ ( s, a ) could be nonlinear transformation and reward could be arbitrary (except being bounded in [0 , 1] ), it is not possible to leverage modelfree approaches to solve KNR as the value functions and Q functions of a KNR could be too complicated to be captured by function classes with bounded complexity.

Definition 6.2 (Kernelized Nonlinear Regulator). Given features φ : S ×A → V with V being some Hilbert space, we say a MDP M is a Kernelized Nonlinear Regulator (KNR) if it admits the following transition function:

<!-- formula-not-decoded -->

where U /star h is a linear operator V ↦→ R d s .

While Kakade et al. [2020] considered arbitrary unbounded reward function, for analysis simplicity, we assume bounded reward, i.e., r ( s, a ) ∈ [0 , 1] for all s, a , but otherwise it could be arbitrary. We assume S ⊂ R d s and ‖ U /star h ‖ 2 := sup x ∈V : ‖ x ‖ 2 ≤ 1 ‖ U /star h x ‖ 2 ≤ B U . We can define the hypothesis class H h as follows:

<!-- formula-not-decoded -->

for all h ∈ [ H ] . Wedefine the discrepancy function /lscript f as follows, for g := { U 0 , U 1 , . . . , U H -1 } with U h ∈ H h and observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

where c = E x ∼N (0 ,σ 2 I ) ‖ x ‖ 2 2 . Note that in this example we set F h = ∅ for all h ∈ [ H ] , thus for notation simplicity, we drop the discriminator notation from the discrepancy function.

Lemma 6.1 (KNR = ⇒ Bilinear Class). Consider a MDP M which is a Kernelized Nonlinear Regulator. Then, for the hypothesis class H , discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Generalized Bilinear Class .

Proof. We follow on-policy strategy and set discriminator classes to be empty, i.e., we set π est = π f , and F h = ∅ for all h ∈ [ H ] . Thus, we have for observed transition info

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, for Bellman error, use the fact that one step immediate reward is bounded in [0 , 1] , Q h,f ( s h , a h ) = r ( s h , a h )+ E s ′ ∼ P h,f ( ·| s h ,a h ) V h +1 ,f ( s ′ ) (since Q h,f and V h,f are the corresponding optimal Q and V functions for model f ∈ H ), we immediately have:

where we use the fact that E s ′ ∼ P h ( ·| s h ,a h ) s ′ = U /star h φ ( s h , a h ) , and we use vec to represent the operator of vectorizing a matrix by stacking its columns into a long vector. Also using the definition of c , it is easy to verify that E a 0: h -1 ∼ π f E a h ∼ π est [ /lscript f ( o h , g ) ] = 0 .

<!-- formula-not-decoded -->

To this end, we can verify that the generalized Bilinear Class captures KNR as follows. We set ζ ( x ) = x , i.e., ζ being identity and β = 1 , ξ ( x ) = H √ x/σ where we see that ξ ( x ) is a concave and non-decreasing function with ξ (0) = 0 , W h ( f ) = vec ( ( U h -U /star h ) /latticetop ( U h -U /star h ) ) (note W h ( f /star ) = 0 ), and X h ( f ) = vec ( E s h ,a h ∼ π f φ ( s h , a h ) φ ( s h , a h ) /latticetop ) .

## 6.2 Generalized Linear Bellman Complete

We first introduce the generalized linear Bellman complete model, and then we show how our framework captures it.

Definition 6.3 (Generalized Linear Bellman Complete). Given a hypothesis class H with H h := { σ ( θ /latticetop h φ ( s, a )) : ‖ θ h ‖ 2 ≤ W } where σ : R ↦→ R + is some inverse link function, we call it generalized linear Bellman complete model is if we have Bellman Completeness for H , i.e., there exists T h : V ↦→ V , such that for all ( θ 0 , . . . , θ H -1 ) and h ∈ [ H ] , we have:

and σ ( T h ( θ h +1 ) /latticetop φ ( s, a )) ∈ H h .

<!-- formula-not-decoded -->

Let us define discriminators F h := { f -f ′ : f ∈ H h , f ′ ∈ H h } . Note that the Bellman complete assumption indicates the following. For any f := { θ 0 , . . . , θ H -1 } , we have σ ( T h ( θ h +1 ) /latticetop φ ( · , · )) -σ ( θ /latticetop h φ ( · , · )) ∈ F h and σ ( θ /latticetop h φ ( · , · )) -σ ( T h ( θ h +1 ) /latticetop φ ( · , · )) ∈ F h .

Assumption 6.3. We assume that inverse link function σ is non-decreasing and the slope of σ is bounded. I.e., for all x ∈ R , σ ′ ( x ) ∈ [ a, b ] for some 0 ≤ a ≤ b .

Under this assumption (also used in Wang et al. [2019]), we can show that Definition 6.1 captures the generalized linear Bellman complete model.

First we will define the discrepancy function /lscript f as follows. For g := { θ 0 , . . . , θ H -1 } and ν ∈ H , and the observed transition info o h = ( r h , s h , a h , s h +1 ) , define /lscript f ( o h , g, ν ) as:

Note that E a 0: h -1 ∼ π f E a h ∼ π est /lscript f ( o h , f /star , ν ) = 0 for all ν ∈ F due to the Bellman complete assumption.

<!-- formula-not-decoded -->

Lemma 6.2 (Generalized Linear Bellman Complete = ⇒ Bilinear Class). Consider a MDP M and hypothesis class H which is a Generalized Linear Bellman Complete model. Then, for discrepancy function /lscript f , discriminator class F h defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Generalized Bilinear Class .

Proof. Setting π est = π f , adding expectation with respect to s h , a h , r h , s h +1 under the

roll-in policy π f , we get:

<!-- formula-not-decoded -->

where the third equality uses the generalized linear Bellman complete assumption, and the first inequality uses the fact that σ ( θ /latticetop h φ ( s h , a h )) -σ ( T h ( θ h +1 ) /latticetop φ ( s h , a h )) ∈ F . Now we continue with the property of the inverse link function as follows.

<!-- formula-not-decoded -->

where the first inequality above uses mean value theorem and σ ′ ( x ) ≥ a, ∀ x (Assumption 6.3). Thus, we can conclude that:

<!-- formula-not-decoded -->

The above is captured by Equation (13) with ζ ( x ) = ax being a linear function.

Now we consider upper bounding the Bellman error. Denote f := { θ 0 , . . . , θ H -1 } . We

have

<!-- formula-not-decoded -->

where the first inequality above uses Jensen's inequality and the second inequality uses mean value theorem and the fact that σ ′ ( x ) ≤ b, ∀ x (Assumption 6.3). Thus, we see that the condition in Equation (12) captures this case with ξ ( x ) = b √ x , X h ( f ) = vec ( E s h ,a h ∼ π f φ ( s h , a h ) φ ( s h , a h ) and W h ( f ) = vec ( ( θ h -T h ( θ h +1 )) ( θ h -T h ( θ h +1 )) /latticetop ) (note that by the Bellman completeness condition, we have W h ( f /star ) = 0 ).

Thus, we have shown that generalized linear MDP is captured by Definition 6.1 with ζ ( x ) = ax and ξ ( x ) = b √ x . Note that ζ and ξ satisfies Assumption 6.1.

## 6.3 Witness Rank

Witness rank [Sun et al., 2019] is a structural complexity that captures model-based RL with H h being the hypothesis space containing transitions P h . Witness rank uses a discriminator class F h ⊂ S × A × S ↦→ R (with F h being symmetric and rich enough to capture V h +1 ,f for all f ∈ H ) to capture the discrepancy between models. Here we focus on model-based setting and H h contains possible transitions g h : S × A ↦→ ∆( S ) and the realizability assumption implies that P h ∈ H h . For simplicity here, we assume reward function is known.

Definition 6.4. We say a MDP M has witness rank d if given two models f ∈ H and g ∈ H , there exists X h : H ↦→ R d and W h : H ↦→ R d such that:

<!-- formula-not-decoded -->

where κ ∈ (0 , 1] .

Similar to Bellman rank, the algorithm and analysis from Sun et al. [2019] rely on d being finite. Below we show how definition 6.1 naturally captures witness rank.

We define /lscript f as follows:

<!-- formula-not-decoded -->

Lemma 6.3 (Finite Witness Rank = ⇒ Bilinear Class). Consider a MDP M which has finite Witness Rank. Then, for the hypothesis class H , discrepancy function /lscript f defined above and uniform estimation policies π est ( f ) = U ( A ) , ( H , /lscript f , Π est , M ) is a Bilinear Class with Discrepancy Family.

Proof. Recall that we denote f /star as the ground truth which in this case means the ground truth transition P . This implies that 〈 W h ( f /star ) , X h ( f ) 〉 = 0 for any f ∈ H . This allows us to write the above formulation as:

<!-- formula-not-decoded -->

For the Bellman error part, since this is the model-based setting, we have Q h,f ( s h , a h ) = r ( s h , a h ) + E s ′ ∼ g h ( s h ,a h ) V h +1 ,f ( s ′ ) . Thus, we have:

<!-- formula-not-decoded -->

∣ Therefore, it is a Bilinear Class with Discrepancy Family with ζ ( x ) = x and ξ ( x ) = 1 κ x .

Here we also give an example for /epsilon1 gen . For F and H with bounded complexity (e.g., discrete F and discrete H ), we still achieve the generalization error, i.e., for all f , for all

g ∈ H , with probability at least 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where s i ∼ d π f h , a i ∼ U ( A ) , s ′ i ∼ P h ( ·| s, a ) , and the inequality assumes that ln(1 /δ ) ≥ 1 (see Lemma 12 from Sun et al. [2019] for derivation).

## 6.3.1 Factored MDP

For completeness, we consider factored MDP as a special example here. We refer readers to Sun et al. [2019] for a detailed treatment of how witness rank capturing factored MDP.

We consider state space S ⊂ O d where O is a discrete set and we denote s [ i ] as the i-th entry of the state s . For each dimension i , we denote pa i ⊂ [ d ] as the set of state dimensions that directly influences state dimension i (we call them the parent set of the i-th dimension). In factored MDP, the transition is governed by the following factorized transition:

<!-- formula-not-decoded -->

where P ( i ) is the condition distribution that governs the transition from s [ pa i ] , a to s ′ [ i ] . Here, we do not assume any structure on reward function.

Note that the complexity of the problem is captured by the number of parameters in the transition operator, which in this case is equal to ∑ d i =1 HA |O| 1+ | pa i | . Note that when the parent set pa i is not too big (e.g., a constant that is independent of d ), this complexity could be exponentially smaller than |O| d for a MDP that does not have factorized structure.

The hypothesis class H contains possible transitions. In factored MDP, we design the following discrepancy function /lscript f ( o h , g, v ) at h for observed transition info o h = ( r h , s h , a h , s h +1 ) ,

<!-- formula-not-decoded -->

With π est = U ( A ) , and discriminators F h = { w 1 + w 2 · · · + w d : w i ∈ W i } where W i = { O | pa i |×A×O ↦→{-1 , 1 } } , Sun et al. [2019] (Proposition 24) shows that there exists X h : H ↦→ R L and W h : H ↦→ R L with L = ∑ d i =1 K |O| | pa i | , such that:

∣ ∣ where we use the fact that 〈 W h ( f /star ) , X h ( f ) 〉 = 0 for all f ∈ H due to the design of the discrepancy function /lscript f . Moreover, Sun et al. [2019] (Lemma 26) also proved that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ Thus factored MDP is captured by Definition 6.1 where ζ ( x ) = x , and ξ ( s ) = AHx . Sun et al. [2019] shows that value function based approaches including Olive Jiang et al. [2017] in worst case requires 2 H many samples to solve factored MDPs, which in turn indicates that the prior structural complexity such as Bellman rank and Bellman Eluder [Jin et al., 2021] must be exponential in H.

## 7 Conclusion

We presented a new framework, Bilinear Classes, together with a new sample efficient algorithm, BiLin-UCB. A key emphasis of the new class and algorithm is that many learnable RL models can be analyzed with the same algorithm and proof.

Our framework is more general than existing ones, and incorporates a large number of RL models with function approximation. Along with the general framework, our work also introduces several important new models including linear Q /star /V /star , RKHS Bellman complete, RKHS linear mixture models and low occupancy complexity. Our rates are non-parametric and depend on a new information theoretic quantity-critical information gain, which is an analog to the critical radius from non-parametric statistics. With this new quantity, our results extend prior finite-dimension results to infinite dimensional RKHS setting.

The Bilinear Classes can also be flexibly extended to cover many other examples including Witness Rank and Kernelized Nonlinear Regulator. We believe many other models (potentially even those proposed in the future) can be analyzed via extensions of the Bilinear Classes.

## Acknowledgements

We thank Chi Jin and Qinghua Liu for discussions on Section 6 including the generalized linear bellman complete model. We thank Akshay Krishnamurthy for a discussion regarding Q/V -Bellman rank.

## References

- Yasin Abbasi-Yadkori, D´ avid P´ al, and Csaba Szepesv´ ari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , 2011.
- Naoki Abe, Alan W Biermann, and Philip M Long. Reinforcement learning with immediate rewards and linear hypotheses. Algorithmica , 37(4):263-293, 2003.
- Alekh Agarwal, Mikael Henaff, Sham Kakade, and Wen Sun. PC-PG: Policy cover directed exploration for provable policy gradient learning. In Advances in Neural Information Processing Systems , 2020a.
- Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank mdps. arXiv preprint arXiv:2006.10814 , 2020b.
- Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin F Yang. Model-based reinforcement learning with value-targeted regression. arXiv:2006.01107 , 2020.
- Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. In Conference on Learning Theory , 2008.
- Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu. On the sample complexity of the linear quadratic regulator. Foundations of Computational Mathematics , pages 1-47, 2019.
- Kefan Dong, Yuping Luo, Tianhe Yu, Chelsea Finn, and Tengyu Ma. On the expressivity of neural networks for deep reinforcement learning. In International Conference on Machine Learning , pages 2627-2637. PMLR, 2020a.
- Kefan Dong, Jian Peng, Yining Wang, and Yuan Zhou. Root-n-regret for learning in markov decision processes with function approximation and low bellman rank. In Conference on Learning Theory , pages 1554-1557. PMLR, 2020b.

- Shi Dong, Benjamin Van Roy, and Zhengyuan Zhou. Provably efficient reinforcement learning with aggregated states, 2020c.
- Simon S Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dud´ ık, and John Langford. Provably efficient RL with rich observations via latent state decoding. In International Conference on Machine Learning , 2019a.
- Simon S Du, Yuping Luo, Ruosong Wang, and Hanrui Zhang. Provably efficient Qlearning with function approximation via distribution shift error checking oracle. In Advances in Neural Information Processing Systems , 2019b.
- Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations , 2020a.
- Simon S Du, Jason D Lee, Gaurav Mahajan, and Ruosong Wang. Agnostic Q-learning with function approximation in deterministic systems: Tight bounds on approximation error and sample complexity. In Advances in Neural Information Processing Systems , 2020b.
- Dylan Foster and Alexander Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning , pages 3199-3210. PMLR, 2020.
- Kaixuan Huang, Sham Kakade, Jason D. Lee, and Qi Lei. A short note on the relationship of information gain and eluder dimension. Unpublished Note .
- Nan Jiang, Alex Kulesza, and Satinder Singh. Abstraction selection in model-based reinforcement learning. In International Conference on Machine Learning , 2015.
- Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E. Schapire. Contextual decision processes with low bellman rank are pac-learnable, 2016.
- Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In International Conference on Machine Learning , 2017.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , 2020.

- Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. arXiv preprint arXiv:2102.00815 , 2021.
- Sham Kakade, Akshay Krishnamurthy, Kendall Lowrey, Motoya Ohnishi, and Wen Sun. Information theoretic regret bounds for online nonlinear control. arXiv preprint arXiv:2006.12466 , 2020.
- Michael Kearns and Daphne Koller. Efficient reinforcement learning in factored mdps. In IJCAI , volume 16, pages 740-747, 1999.
- Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013.
- Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Pac reinforcement learning with rich observations. In Proceedings of the 30th International Conference on Neural Information Processing Systems , pages 1848-1856, 2016.
- Lihong Li. A Unifying Framework for Computational Reinforcement Learning Theory . PhD thesis, USA, 2009. AAI3386797.
- Michael L Littman and Richard S Sutton. Predictive representations of state. In Advances in Neural Information Processing Systems , 2002.
- Michael L Littman, Richard S Sutton, and Satinder P Singh. Predictive representations of state. In NIPS , volume 14, page 30, 2001.
- Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, and John Langford. Kinematic state abstraction and provably efficient rich-observation reinforcement learning. In International Conference on Machine Learning , 2020.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.
- Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. In Conference on Artificial Intelligence and Statistics , 2020a.
- Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. In International Conference on Artificial Intelligence and Statistics , pages 2010-2020. PMLR, 2020b.

- R´ emi Munos. Error bounds for approximate value iteration. In Proceedings of the National Conference on Artificial Intelligence , volume 20, page 1006. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2005.
- Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research , 39(4):1221-1243, 2014.
- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature , 550(7676):354-359, 2017.
- Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995 , 2009.
- Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Modelbased RL in contextual decision processes: PAC bounds and exponential improvements over model-free approaches. In Conference on Learning Theory , 2019.
- Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge University Press, 2019.
- Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems , 33, 2020.
- Yining Wang, Ruosong Wang, Simon S Du, and Akshay Krishnamurthy. Optimism in reinforcement learning with generalized linear function approximation. arXiv:1912.04136 , 2019.
- Gellert Weisz, Philip Amortila, and Csaba Szepesv´ ari. Exponential lower bounds for planning in mdps with linearly-realizable optimal action-value functions, 2020.
- Zheng Wen and Benjamin Van Roy. Efficient exploration and value function generalization in deterministic systems. In Advances in Neural Information Processing Systems , 2013.
- Cathy Wu, Kanaad Parvate, Nishant Kheterpal, Leah Dickstein, Ankur Mehta, Eugene Vinitsky, and Alexandre M Bayen. Framework for control and deep reinforcement learning in traffic. In 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC) , pages 1-8. IEEE, 2017.

- Lin Yang and Mengdi Wang. Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning , 2019.
- Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, and Michael I Jordan. Bridging exploration and general function approximation in reinforcement learning: Provably efficient kernel and neural value iterations. arXiv preprint arXiv:2011.04622 , 2020.
- Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error, 2020.

## Contents

| 1   | Introduction   |
|-----|----------------|

| A.3   | Linear Quadratic Regulator . . . . . . . . . . . . . . . . . . . . . . . . .   | 60   |
|-------|--------------------------------------------------------------------------------|------|
| A.4   | Linear MDP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 62   |
| A.5   | Block MDPand Reactive POMDP . . . . . . . . . . . . . . . . . . . . .          | 62   |
| B     | Proofs for Section 5                                                           | 64   |
| C     | An Elliptical Cover for Hilbert Spaces                                         | 65   |
| D     | Concentration Arguments for Special Cases                                      | 67   |
| E     | Generalized Bilinear Classes                                                   | 69   |
| F     | Auxiliary Lemmas                                                               | 72   |
| G     | Sample Complexity Lower Bound for RHKS Bellman Complete and Linear             |      |

MDP

73

## A Additional Examples of Bilinear Classes

We now include some other examples of Bilinear Classes in addition to ones discussed in Section 4.3.

## A.1 FLAMBE / Feature Selection

We consider the feature selection setting introduced by Agarwal et al. [2020b].

Definition A.1 (Feature Selection). We say a MDP M is low rank feature selection model if there exists (unknown) functions µ /star h : S ↦→ V and (unknown) features φ /star : S × A ↦→ V , ψ /star : S × A for some Hilbert space V such that for all h ∈ [ H ] and ( s, a, s ′ ) ∈ S × A × S

<!-- formula-not-decoded -->

Note that unlike linear MDP model where φ /star is assumed to be known, here φ /star is unknown to the learner. We use a function class Φ ⊂ S × A ↦→ V to capture φ /star , i.e., we assume realizability φ /star ∈ Φ .

We can define our function class H = H 0 × . . . , H H -1 as follows

<!-- formula-not-decoded -->

to capture the optimal value Q /star . Note that since φ /star ∈ Φ , and the optimal Q function is linear with respect to feature φ /star ( s, a ) , we immediately have f /star := { Q /star 0 , . . . , Q /star H -1 } ∈ H . Wedefine the following discrepancy function /lscript f (in this case the discrepancy function does not depend on f ) for any g ∈ H and for observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

Lemma A.1. Consider a MDP M which is a low rank feature selection model. Then, for the hypothesis class H , discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = U ( A ) , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Note that for g = f , we have that (here observed transition info o h = ( r h , s h , a h , s h +1 ) )

<!-- formula-not-decoded -->

Therefore, to prove that this is a Bilinear Class, we will show that a stronger 'equality' version of Equation (2) holds (which will also prove Equation (1) holds). Observe that for

any h ,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Observe that W h ( f /star ) = 0 due to Bellman optimality condition for V /star and π /star .

## A.2 Q /star irrelevance Aggregation / Q /star state Aggregation

We now consider the Q /star irrelevance aggregation model introduced in Li [2009].

Definition A.2 ( Q /star irrelevance aggregation model). We say a MDP M is the Q /star irrelevance aggregation model if there exists known function ζ : S ↦→ V such that for all states s 1 , s 2 ∈ S

<!-- formula-not-decoded -->

Let Z = { ζ ( s ) : s ∈ S} . Here, our hypothesis class H = H 0 × . . . , H H -1 is a set of linear functions i.e. for all h ∈ [ H ] , the set H h is defined as:

We also define the following discrepancy function /lscript f (in this case the discrepancy function does not depend on f ), for hypothesis g = { ( w h , θ h ) } H -1 h =0 and observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.2. Consider a MDP M which is the Q /star irrelevance aggregation model. Then, for the hypothesis class H , discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. To prove that this is implicitly a Bilinear Class, we will reduce this into linear Q /star /V /star model (Definition 4.5). Let Z = { ζ ( s ) : s ∈ S} . Now, we construct one hot representation functions φ : S × A ↦→ { 0 , 1 } |Z|×|A| and ψ : S ↦→ { 0 , 1 } |Z| where

Then, it clear that we can construct w /star ∈ R |Z|×|A| and θ /star ∈ R |Z| as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

such that the following holds:

<!-- formula-not-decoded -->

This is linear Q /star /V /star model (Definition 4.5) and therefore is a Bilinear Class.

## A.3 Linear Quadratic Regulator

In this subsection, we prove that Linear Quadratic Regulators (LQR) forms a Bilinear Class. Note that even though LQR has small bellman rank, the corresponding algorithm in Jiang et al. [2017] has action dependence in sample complexity unlike our algorithm which does not have a dependence on number of actions. Here we consider S ⊂ R d and A ⊂ R K .

Definition A.3 (Linear Quadratic Regulator). We say a MDP M is a finite-horizon discrete-time Linear Quadratic Regulator if there exists (unknown) A ∈ R d × d , (unknown) B ∈ R d × K and (unknown) Q ∈ R d × d such that we can write the transition function and reward function as follows

<!-- formula-not-decoded -->

where noise variables /epsilon1 h , τ h are zero centered with E [ /epsilon1 h /epsilon1 /latticetop h ] = Σ and E [ τ 2 h ] = σ 2 .

To maintain notation of fixed starting state, without loss of generality, we also assume s 0 = 0 and a 0 = 0 . An important property of LQR is that for linear non stationary policies π , the value function V π induced is quadratic (see for e.g. Jiang et al. [2017][Lemma 7] for a proof).

Lemma A.3. If π is a non stationary linear policy π h ( s h ) = C π,h x for some C π,h ∈ R K × d , then V π h ( s h ) = s /latticetop h Λ π,h s h + O π,h for some Λ π,h ∈ R d × d and O π,h ∈ R .

This allows us to define out hypothesis class H = H 0 , . . . , H H -1 as

<!-- formula-not-decoded -->

with for any f ∈ H

<!-- formula-not-decoded -->

We define the following discrepancy function /lscript f for any hypothesis g ∈ H and observed transition info o h = ( r h , s h , a h , s h +1 ) :

<!-- formula-not-decoded -->

LemmaA.4. Consider a MDP M which is a Linear Quadratic Regulator. Then, for the hypothesis class H , discrepancy function /lscript f defined above and on-policy estimation policies π est ( f ) = π f for f ∈ H , ( H , /lscript f , Π est , M ) is ( implicitly ) a Bilinear Class .

Proof. Note that for g = f , we have that (here observed transition info o h = ( r h , s h , a h , s h +1 ) )

<!-- formula-not-decoded -->

Therefore, to prove that this is a Bilinear Class, we will show that a stronger 'equality' version of Equation (2) holds (which will also prove Equation (1) holds). Observe that for any h ,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that we used 〈 W h ( f /star ) , X h ( f ) 〉 = 0 which follows from the bellman conditions i.e. for a h = C h,f /star s h

<!-- formula-not-decoded -->

Taking expectation over s h ∼ d π f proves the claim.

## A.4 Linear MDP

We consider the Linear MDP setting from Yang and Wang [2019], Jin et al. [2020].

Definition A.4 (Linear MDP). We say a MDP M is a Linear MDP with features φ : S × A ↦→ V , where V is a Hilbert space if for all h ∈ [ H ] , there exists (unknown) measures µ h over S and (unknown) θ h ∈ V , such that for any ( s, a ) ∈ S × A , we have

<!-- formula-not-decoded -->

Here, our hypothesis class H is set of linear functions with respect to φ . We denote hypothesis in our hypothesis class H as tuples ( θ 0 , . . . θ H -1 ) , where θ h ∈ V . As observed in Jin et al. [2020][Proposition 2.3], this satisfies the conditions of Bellman Complete model (Definition 4.6) and therefore is also a Bilinear Class.

## A.5 Block MDP and Reactive POMDP

Both Block MDP [Du et al., 2019a, Misra et al., 2020] and a Reactive POMDP [Krishnamurthy et al., 2016] are partially observable MDPs (POMDPs) which can be described by a finite (unobservable) latent state space S , a finite action space A , and a possibly infinite but observable

context space X . The transitions can be described by two conditional probabilities. One is the latent state transition p : S × A ↦→ /triangle ( S ) , and the other is the context-emission function q : S ↦→ /triangle ( X ) .

The key differences among Block MDP and Reactive POMDP are in the assumptions which we define below.

Definition A.5 (Block MDP). For Block MDPs, the context space X can be partitioned into disjoint blocks X s for s ∈ S , each containing the support of the conditional distributiion q ( ·| s ) .

This assumption implies there exists a perfect decoding function f ∗ : X → S , which maps contexts to their generating states. Therefore, we have that the transition of contexts satisfies

<!-- formula-not-decoded -->

where e f ∗ ( x ′ ) ∈ R |S| is a one-hot vector where only the entry that corresponds to f ∗ ( x ′ ) is 1 . Note one can define µ ∗ ( x ′ ) /defines e f ∗ ( x ′ ) and φ ∗ ( x, a ) /defines p ( ·| f ∗ ( x ) , a ) as in the FLAMBE setting. Thus, Block MDP is a subclass of FLAMBE with the Hilbert space V being the | S | -dimensional Euclidean space. Since FLAMBE is within our Bilinear Class, Block MDP is also within our framework.

For POMDP, assume reward is known and is a deterministic function over observations and actions and r ( x, a ) ∈ [0 , 1] . let us define belief b h ( ·| h h ) ∈ ∆( S ) as the posterior distribution of state s at time step h given history h h := x 0 , a 0 , . . . , x h -1 , a h -1 , x h , i.e., given any state s , we have b h ( s | h h ) = P ( s | x 0 , a 0 , . . . , x h -1 , a h -1 , x h ) . Given a h and conditioned on x h +1 being observed at h +1 , the belief is updated based on the Bayes rule, deterministically,

<!-- formula-not-decoded -->

with b 0 ( s | x 0 ) ∝ µ 0 ( s ) q ( x 0 | s ) , where µ 0 ∈ ∆( S ) is the initial state distribution (in the simplified case where we have a fixed s 0 , then µ 0 is a delta distribution with all probability mass on s 0 ).

Note that given a h , x h +1 , the above update is deterministic, and b h ( s | h h ) is a function of history h h . Denote the deterministic Belief update procedure as b h +1 = Γ( b h , a h , x h +1 ) . For POMDP, the optimal policy π /star is a mapping from ∆( S ) to A . Given a belief b , and an action a , we can define Q /star h ( b, a ) backward as follows. Start with V /star H ( b ) = 0 for all b ∈ ∆( S ) ,

<!-- formula-not-decoded -->

where V /star h ( b ) = argmax a Q /star h ( b, a ) , π /star h ( b ) = argmax a Q /star h ( b, a ) .

Definition A.6 (Reactive POMDP). For Reactive POMDPs, the optimal Q function Q /star h is only dependent on latest observation and action, i.e., for all h , there exists g /star h : X × A ↦→ [0 , H ] , such that, for any given history h h := x 0 , a 0 , . . . , x h -1 , a h -1 , x h , we have:

<!-- formula-not-decoded -->

Note that in this case, the optimal policy π /star h only depends on the latest observation x h , i.e., π /star h ( b ( ·| h h )) = argmax a ∈A Q /star h ( b ( ·| h h ) , a ) = argmax a ∈A g /star h ( x h , a ) . As shown in Jiang et al. [2017], Reactive POMDPs have bellman rank bounded by |S| which implies (see Section 4.1 for more detail) that Reactive POMDPs are a Bilinear Class.

## B Proofs for Section 5

Proof of Corollary 5.1. First, using Lemma F.1, we get that for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all g ∈ H

This satisfies our Assumption 5.1 with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using this in Theorem 5.1, we set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we get /epsilon1 -optimal policy by setting

<!-- formula-not-decoded -->

or equivalently by setting m at least as large as

<!-- formula-not-decoded -->

Using Lemma F.2, we get a solution for m

<!-- formula-not-decoded -->

This gives the total trajectory complexity

<!-- formula-not-decoded -->

for some absolute constants c .

## C An Elliptical Cover for Hilbert Spaces

The following theorem is a key technical contribution which allows us to obtain a number of non-parametric convergence rates.

Theorem C.1. Let X ⊂ V , where V is a Hilbert space. Suppose T ∈ N + , /epsilon1 ∈ R + ; define W ⊆ { w ∈ V : ‖ w ‖ ≤ B W } for some real number B W ; and suppose for all x ∈ X that ‖ x ‖ 2 ≤ B X . Set λ = /epsilon1 2 / (8 B 2 W ) .There exists a set C ⊂ W (a cover of W ) such that: (i) log |C| ≤ T log(1 + 3 B W B X √ T//epsilon1 ) and (ii) for all w ∈ W , there exists a w ′ ∈ C , such that:

<!-- formula-not-decoded -->

Proof. Let us suppose that X is closed, in order for certain maximizers (and arg-maximizers) over X to exist. If X is not closed, then let us replace X with the closure of X , which is possible since X is a bounded set. Consider the process: Set Σ 0 = λI with λ ∈ R + .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Via Lemma 5.6, we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that there must exist a t ∈ 0 , . . . , T -1 , such that:

which means that:

<!-- formula-not-decoded -->

Note that x t = argmax x ∈X ‖ x ‖ Σ -1 t . Thus, we have that:

Note that the above derivation holds for any λ ∈ R + .

<!-- formula-not-decoded -->

Define M T = ∑ T i =0 x t x /latticetop t . Note that the range of M T , Range ( M T ) is a T + 1 -dimensional object. For an /epsilon1 ′ -net, C , in /lscript 2 distance over B W -norm ball on Range ( M T ) , i.e., { v ∈ W : v ∈ Range ( M T ) } . With a standard covering number bound, we have that ln( |C| ) ≤ 2 T ln (1 + 2 B W //epsilon1 ′ ) (e.g. see Lemma D.1).

Fix some w ∈ W . Denote the projection of w on the the range of M T by w . Let w ′ ∈ C being the closest point to w in /lscript 2 distance. Note that ‖ w -w ′ ‖ 2 ≤ /epsilon1 ′ . For any x ∈ X , we have:

<!-- formula-not-decoded -->

where the equality in the third step uses that ( w -w ′ ) /latticetop x i = ( w -w ′ ) /latticetop x i for all i ∈ 0 , . . . , T . The proof is completed choosing λ = /epsilon1 2 / (8 B 2 W ) and ( /epsilon1 ′ ) 2 = /epsilon1 2 / (2 TB 2 X ) .

## D Concentration Arguments for Special Cases

An application to RKHS Linear MDPs. Consider the RKHS linear MDP, where φ : S × A ↦→ H with H being some Hilbert space. Define Φ = { φ ( s, a ) : s ∈ S , a ∈ A} .

Corollary D.1. Suppose T ∈ N + and /epsilon1 ∈ R + ; define W ⊆ { w ∈ H : ‖ w ‖ ≤ B W } for some real number B W ; and suppose for all φ ( s, a ) ∈ Φ that ‖ φ ( s, a ) ‖ 2 ≤ B φ . There exists a set C ⊂ W such that: (i) log |C| ≤ T log(1 + 3 B φ B W √ T//epsilon1 ) and (ii) for all w ∈ W , there exists a w ′ ∈ C such that for all distributions d over S × A × S , we have:

<!-- formula-not-decoded -->

Proof. For any distribution d , we seek to bound:

<!-- formula-not-decoded -->

where the last step follows using that | sup x f ( x ) -sup x g ( x ) | ≤ sup x | f ( x ) -g ( x ) | (which can be verified by considering both case of the sign inside the absolute value). The proof is completed by choose w ′ to be closest point C to w and applying Theorem C.1.

Corollary D.2. Define W =: { w ∈ H : ‖ w ‖ ≤ B W , w /latticetop φ ( s, a ) ∈ [0 , H ] ∀ s, a ∈ S × A} for some real number B W ; and suppose for all φ ( s, a ) ∈ Φ that ‖ φ ( s, a ) ‖ 2 ≤ B φ . Let

<!-- formula-not-decoded -->

with r ∈ [0 , 1] . Then, for any distribution µ over R ×S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all w ∈ H

where γ m = γ (1 / (8 B 2 W m ); Φ) (as defined in Equation (6) ).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ ˜ Proof. First note that for any w ∈ W , we must have:

since we eliminate all w such that w /latticetop φ ( s, a ) /negationslash∈ [0 , H ] for some s, a .

<!-- formula-not-decoded -->

Consider the cover C from Corollary D.1. From Lemma F.1 and a union bound over all w ′ ∈ C , for all w ′ ∈ C , we have that with probability at least 1 -δ :

Now consider any w ∈ W , via Corollary D.1, we know that there exists a w ′ ∈ C such that:

<!-- formula-not-decoded -->

Thus, together with the fact that Corollary D.1 holds for both µ and the uniform distribution over D , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us set /epsilon1 = 1 / √ m and rearrange terms, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote ˜ γ m = T where T is the smallest integer that satisfies T ≥ γ T (1 / (8 B 2 W m )) . Thus, we have:

where in the inequality we use exp ( γ T (1 / (8 B 2 W m )) T ) -1 ≤ e -1 ≤ 2 .

<!-- formula-not-decoded -->

An application to RKHS linear functions Consider features ζ : S × A × S ↦→ V with V being some Hilbert space. Define Z = { ζ ( s, a, s ′ ): ( s, a, s ′ ) ∈ S × A × S} .

Corollary D.3. Define W =: { w ∈ V : ‖ w ‖ ≤ B W , w /latticetop ζ ( s, a, s ′ ) ∈ [0 , H ] ∀ s, a, s ′ ∈ S × A×S} for some real number B W ; and suppose for all ζ ( s, a, s ′ ) ∈ Z that ‖ ζ ( s, a, s ′ ) ‖ 2 ≤ B ζ . Let

<!-- formula-not-decoded -->

Then, for any distribution µ over S × A × S and for any δ ∈ (0 , 1) , with probability of at least 1 -δ over choice of an i.i.d. sample D ∼ µ m of size m , for all w ∈ H

where ˜ γ m = ˜ γ (1 / (8 B 2 W m ); Z ) (as defined in Equation (6) ). Proof. The proof follows exactly as proof of Corollary D.2.

<!-- formula-not-decoded -->

Lemma D.1 (Covering number). For any /epsilon1 &gt; 0 , the /epsilon1 -covering number of the Euclidean ball in R d with radius R ∈ R + , i.e., B = { x ∈ R d : ‖ x ‖ 2 ≤ R } , is upper bounded by (1 + 2 R//epsilon1 ) d .

## E Generalized Bilinear Classes

Recall Definition 6.1 for Generalized Bilinear Class. We next complete the proof of Theorem 6.1.

Proof of Theorem 6.1. First notice that a uniform convergence result similar to Lemma 5.1 still holds:

where ε gen := ε gen ( m, H , F ) · conf ( δ/ ( TH )) .

<!-- formula-not-decoded -->

Also it is easy to verify that the feasibility claim similar to Lemma 5.3 holds as well since max ν ∈F h L µ t ; h ,f t ( f /star , ν ) = 0 . The feasibility result immediately implies the optimism claimed in Lemma 5.4. While the derivation of Lemma 5.5 mostly follows, we use Equation (12) rather than Equation (1), which gives us the following:

<!-- formula-not-decoded -->

where the last step follows from concavity of ξ (Assumption 6.1) and Jensen's inequality.

To show the existence of a high quality policy, we also mainly follow the steps in the proof of Lemma 5.2. First we can verify Equation (7) holds due to the elliptical potential argument. This implies that for all h ,

<!-- formula-not-decoded -->

Thus together with Equation (13), we have:

<!-- formula-not-decoded -->

Note that by Assumption 6.1 and an application of mean-value theorem, we have:

<!-- formula-not-decoded -->

Thus, we have:

<!-- formula-not-decoded -->

Together, we arrive:

<!-- formula-not-decoded -->

Sum over all h, we have:

<!-- formula-not-decoded -->

Apply ξ on both sides and use the assumption that ξ is non-decreasing, we have:

<!-- formula-not-decoded -->

This means that there exists a t :

<!-- formula-not-decoded -->

Now set λ = ε 2 gen ( m, H ) /B 2 W , and T ≥ γ ( λ, X ) , we get:

This concludes the first part of the theorem.

When ξ is continuously differentiable, ξ (0) = 0 , and max f,g,h ξ ′ ( 〈 W h ( g, f /star ) , X h ( f ) 〉 ) ≤ α , we simply have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F Auxiliary Lemmas

Lemma F.1 (Azuma-Hoeffding). Let X 1 , . . . , X m be independent random variables with mean µ such that | X i | ≤ B for some B &gt; 0 almost surely for all i ∈ [ m ] . Then, with probability 1 -δ ,

∣ ∣ Lemma F.2. (Log Dominance Rule) Suppose α, a, b ≥ 0 and c ≥ (1 + α ) α . Then, m = ca ln α ( abc ) is a solution to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma F.3. Let X ⊂ R d and sup x ∈X ‖ x ‖ 2 ≤ B X . Then, the maximum information gain

<!-- formula-not-decoded -->

Furthermore, the critical information gain

Proof.

We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First note that

Therefore, using the Determinant-Trace inequality, we get the first result

<!-- formula-not-decoded -->

To get the second result, first note that for n = cd ln(1 + cB 2 X /λ ) and c = 3 ,

<!-- formula-not-decoded -->

where the third last step follows from ln(1 + cB 2 X /λ ) ≥ 0 and ln(1 + cB 2 X /λ ) ≥ ln(ln(1 + cB 2 X /λ )) and last step follows from c = 3 &gt; 2 .

## G Sample Complexity Lower Bound for RHKS Bellman Complete and Linear MDP

Recall that in Section 5.3.2, we show that under the assumption that sup h ∈ [ H ] ,θ ∈H h ‖ θ ‖ 2 and sup x ∈ Φ ‖ x ‖ 2 are both bounded, and the assumption that the maximum information gain is bounded, then our algorithm finds a near-optimal policy using polynomial number of samples for RHKS Bellman Complete and Linear MDP. One may wonder if the assumption on the maximum information gain can be removed as in the case of contextual bandits [Abe et al., 2003, Foster and Rakhlin, 2020]. Here we show that for the case of reinforcement learning, without the maximum information gain assumption, there is an

exponential sample complexity lower bound (in the problem horizon H ). Therefore, our hardness result justifies the necessity of assuming bounded maximum information gain for the case of RHKS Bellman Complete and Linear MDP.

Our hard instance is based on the binary tree instance (see Du et al. [2020a], Krishnamurthy et al. [2016] for previous hardness results that use such a construction). In this construction, there are H levels of states, and level h ∈ [ H ] contains 2 h distinct states. Thus we have |S| = 2 H -1 . We use s 0 , s 1 , . . . , s 2 H -2 to name these states. Here, s 0 is the unique state in level h = 0 , s 1 and s 2 are the two states in level h = 1 , s 3 , s 4 , s 5 and s 6 are the four states in level h = 2 , etc. There are two different actions, a 1 and a 2 , in the MDPs. For a state s i in level h with h &lt; H -1 , playing action a 1 transits state s i to state s 2 i +1 and playing action a 2 transits state s i to state s 2 i +2 , where s 2 i +1 and s 2 i +2 are both states in level h +1 . In the hard instances, r ( s, a ) = 0 for all ( s, a ) pairs except for a special state s in level H -1 and a special action a ∈ { a 1 , a 2 } . For the special state s and the special action a , we have r ( s, a ) = 1 . It is known that for such hard instances, any algorithm requires Ω(2 H ) to find a policy π with V /star ( s 0 ) -V π ( s 0 ) ≤ 0 . 5 with probability at least 0 . 9 (see Du et al. [2020a]). Now we construct a set of uninformative features and the hypothesis class H so that sup h ∈ [ H ] ,θ ∈H h ‖ θ ‖ 2 and sup x ∈ Φ ‖ x ‖ 2 are both bounded.

Recall that the feature mapping φ maps S × A to a Hilbert space V . In our case, we set V = R d with d = 2 |S| . For each i ∈ [ |S| ] , we define φ ( s i , a 1 ) = e 2 i +1 and φ ( s i , a 2 ) = e 2 i +2 . Here, for an integer k ∈ [ d ] , e k is the k -th standard basis vector. For each h ∈ [ H ] , we have H h = { e 1 , e 2 , . . . , e 2 |S| } . Clearly, no matter which state-action pair ( s, a ) ∈ S × A is chosen as the special state-action pair, we always have Q /star ∈ H , i.e., the realizability assumption is satisfied. Moreover, both sup h ∈ [ H ] ,θ ∈H h ‖ θ ‖ 2 and sup x ∈ Φ ‖ x ‖ 2 are bounded by 1 . Formally, we have the following theorem.

Theorem G.1. For any H &gt; 0 , there exists a class of MDPs M where the number of states is 2 H -1 and the number of actions is 2 , together with a hypothesis class H that is Bellman Complete with respect to MDPs in M . Moreover, sup h ∈ [ H ] ,θ ∈H h ‖ θ ‖ 2 ≤ 1 and sup x ∈ Φ ‖ x ‖ 2 are bounded by 1 , and the transitions and rewards of MDPs in M are all deterministic. Any algorithm that finds a policy π with V /star ( s 0 ) -V π ( s 0 ) ≤ 0 . 5 with probability at least 0 . 9 for MDPs in M requires Ω(2 H ) samples.