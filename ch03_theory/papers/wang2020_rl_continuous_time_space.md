## Reinforcement Learning in Continuous Time and Space: A Stochastic Control Approach

## Haoran Wang

hrwang2718@gmail.com

CAI Data Science and Machine Learning The Vanguard Group, Inc. Malvern, PA 19355, USA

## Thaleia Zariphopoulou

zariphop@math.utexas.edu

Department of Mathematics and IROM The University of Texas at Austin Austin, TX 78712, USA Oxford-Man Institute University of Oxford Oxford, UK

## Xun Yu Zhou

xz2574@columbia.edu

Department of Industrial Engineering and Operations Research The Data Science Institute Columbia University New York, NY 10027, USA

Editor:

Shie Mannor

## Abstract

We consider reinforcement learning (RL) in continuous time with continuous feature and action spaces. We motivate and devise an exploratory formulation for the feature dynamics that captures learning under exploration, with the resulting optimization problem being a revitalization of the classical relaxed stochastic control. We then study the problem of achieving the best trade-off between exploration and exploitation by considering an entropy-regularized reward function. We carry out a complete analysis of the problem in the linear-quadratic (LQ) setting and deduce that the optimal feedback control distribution for balancing exploitation and exploration is Gaussian. This in turn interprets the widely adopted Gaussian exploration in RL, beyond its simplicity for sampling. Moreover, the exploitation and exploration are captured respectively by the mean and variance of the Gaussian distribution. We characterize the cost of exploration, which, for the LQ case, is shown to be proportional to the entropy regularization weight and inversely proportional to the discount rate. Finally, as the weight of exploration decays to zero, we prove the convergence of the solution of the entropy-regularized LQ problem to the one of the classical LQ problem.

Keywords: Reinforcement learning, entropy regularization, stochastic control, relaxed control, linear-quadratic, Gaussian distribution

## 1. Introduction

Reinforcement learning (RL) is currently one of the most active and fast developing subareas in machine learning. In recent years, it has been successfully applied to solve large scale

© 2020 Haoran Wang, Thaleia Zariphopoulou and Xun Yu Zhou.

real world, complex decision making problems, including playing perfect-information board games such as Go (AlphaGo/AlphaGo Zero, Silver et al., 2016; Silver et al., 2017), achieving human-level performance in video games (Mnih et al., 2015), and driving autonomously (Levine et al., 2016; Mirowski et al., 2017). An RL agent does not pre-specify a structural model or a family of models but, instead, gradually learns the best (or near-best) strategies based on trial and error, through interactions with the random (black box) environment and incorporation of the responses of these interactions, in order to improve the overall performance. This is a case of 'kill two birds with one stone': the agent's actions (controls) serve both as a means to explore (learn) and a way to exploit (optimize) .

Since exploration is inherently costly in terms of resources, time and opportunity, a natural and crucial question in RL is to address the dichotomy between exploration of uncharted territory and exploitation of existing knowledge. Such question exists in both the stateless RL settings (e.g., the multi-armed bandit problem) and the more general multistate RL settings (e.g., Sutton and Barto, 2018; Kaelbling et al., 1996). Specifically, the agent must balance between greedily exploiting what has been learned so far to choose actions that yield near-term higher rewards, and continuously exploring the environment to acquire more information to potentially achieve long-term benefits.

Extensive studies have been conducted to find strategies for the best trade-off between exploitation and exploration. For the classical multi-armed bandit problem, well known strategies include the Gittins index approach (Gittins, 1974), Thompson sampling (Thompson, 1933), and upper confidence bound algorithm (Auer et al., 2002), whereas theoretical optimality is established, for example, in Russo and Van Roy (2013, 2014). For general RL problems, various efficient exploration methods have been proposed that aim to improve learning efficiency and yield low sample complexity, among other goals. Most of these works seem to mainly focus on the algorithmic aspect of the RL for discrete-time, Markov decision processes (MDPs). The learning efficiency of these algorithms are typically analyzed in the PAC (Probably Approximately Correct) framework in order to minimize regret and/or sample complexity, with notable examples including Brafman and Tennenholtz (2002), Strehl and Littman (2008), Strehl et al. (2009) that develop PAC-MDP exploration algorithms for small and finite MDPs. For linear-quadratic-regulator (LQR) problems in discrete time, methods of adaptive control and Thompson sampling have been shown to be effective in leading to low regret (Abbasi-Yadkori and Szepesv´ ari, 2011; Abeille and Lazaric, 2017; Abeille and Lazaric, 2018; Fazel et al., 2018). A more unified approach to characterize the complexity of general model-based RL problems can be found in Osband and Van Roy (2014), which includes linear control problems as a special case.

In a different direction, discrete-time entropy-regularized (also termed as 'entropyaugmented' or 'softmax') RL formulation has been recently proposed which explicitly incorporates exploration into the optimization objective as a regularization term, with a trade-off weight imposed on the entropy of the exploration strategy (Ziebart et al., 2008; Nachum et al., 2017; Fox et al., 2016; see also Neu et al., 2017 and the references therein). An exploratory distribution with a greater entropy signifies a higher level of exploration, reflecting a bigger weight on the exploration front. On the other hand, having the minimal entropy, the extreme case of Dirac measure implies no exploration, reducing to the case of classical optimization with a complete knowledge about the underlying model. Recent works have been devoted to designing various algorithms to solve the entropy-regularized

RL problem, where numerical experiments have demonstrated remarkable robustness and multi-modal policy learning (Haarnoja et al., 2017; Haarnoja et al., 2018). Neu et al. (2017) provides a more general framework of entropy-regularized RL with a focus on duality and convergence properties of the corresponding algorithms. In particular, the paper shows that a regularization using the conditional entropy of the joint state-action distributions leads to a dual problem similar to the dynamic programming equation.

In this paper, we study RL in a continuous-time setting with both continuous control (action) and state (feature) spaces. 1 Such a continuous-time formulation is appealing, and indeed necessary, if the agent can interact with the environment at ultra-high frequency, examples including high frequency stock trading, autonomous driving and snowboard riding. More importantly, once cast in continuous time and space, it is possible, thanks in no small measure to the tools of stochastic calculus and differential equations, to derive analytical results which, in turn, may lead to theoretical understanding of some of the important issues in RL, give guidance to algorithm design and provide interpretability to the underlying learning technologies.

The objective of this paper is not to develop any new, efficient RL algorithm (like most existing works do) but, rather, to propose and provide a theoretical framework-that of stochastic control-for studying RL problems in continuous time and space. 2 Our main contribution is to motivate and devise an 'exploratory formulation' for the state dynamics that captures repetitive learning under exploration in the continuous time limit. In RL, the notion of exploration is captured by randomizing actions. This randomization can be naturally and easily formulated as what is known as 'stochastic policies' and be carried out only at discrete time epochs, one at a time, for an MDP. The extension to the continuous-time setting is highly non-trivial as one needs to continuously randomize actions, and there has been little understanding (if any) of how to appropriately incorporate stochastic policies into the standard stochastic control problems. Indeed, exploration substantially enriches the space of control strategies, from that of Dirac measures to that of all probability distributions. This, in turn, is poised to change both the underlying state transitions and the system dynamics. We show that our exploratory formulation can account for the effects of learning in the state transitions observed from the interactions with the environment.

Intriguingly, the proposed formulation of the state dynamics coincides with that in the relaxed control framework in classical control theory (see, e.g., Fleming and Nisio, 1984; El Karoui et al., 1987; Zhou, 1992; Kurtz and Stockbridge, 1998, 2001), which was motivated by entirely different reasons. Specifically, relaxed controls were introduced to mainly deal with the theoretical question of whether an optimal control exists. The approach essentially entails randomization to convexify the universe of control strategies. To the best of our knowledge, the present paper is the first to bring back the formulation of relaxed control, guided by a practical motivation: exploration and learning. This, in turn, represents a main innovation of our RL formulation compared with the existing entropy-regularized

1. The terms 'feature' and 'action' are typically used in the RL literature, whose counterparts in the control literature are 'state' and 'control', respectively. Since this paper uses the control approach to study RL problems, we will interchangeably use these terms whenever there is no confusion.

2. Within our framework, specific algorithms could indeed be developed in various application domains; see a follow-up work Wang and Zhou (2020) for an application in mean-variance portfolio selection.

study in discrete time-in the latter, there is no analogous 'relaxed' formulation arising from exploration.

The proposed exploratory dynamics lay a foundation upon which one can study various dynamic optimization problems associated with different RL objectives in continuous time and spaces. As an illustration, in this paper we study the exploration-exploitation trade-off by considering an entropy-regularized objective function, which has been widely studied in the discrete-time setting as discussed earlier. We carry out a complete analysis of the continuous-time entropy-regularized RL problem, assuming that the original system dynamics is linear in both the control and the state, and that the original reward function is quadratic in both of them. This type of linear-quadratic (LQ) problems has occupied the center stage for research in classical control theory for its elegant solutions and its ability to approximate more general nonlinear problems. An important, conceptual contribution of this paper along this line is to link entropy-regularization and Gaussian explorationtwo extensively studied topics in the current RL literature (albeit mostly for discrete-time MDPs). We accomplish this by showing that the optimal feedback control distribution for balancing exploitation and exploration is Gaussian . Precisely speaking, if, at any given state, the agent sets out to engage in exploration, then she needs to look no further than Gaussian distributions. As is well known, a pure exploitation optimal distribution is Dirac and a pure exploration optimal distribution is uniform. Our results reveal that Gaussian is the correct choice if one seeks a balance between those two extremes. Moreover, we find that the mean of this optimal exploratory distribution is a function of the current state independent of the intended exploration level, whereas the variance is a linear function of the entropy regularizing weight (also called the 'temperature parameter' or 'exploration weight') irrespective of the current state. This result highlights a separation between exploitation and exploration: the former is reflected in the mean and the latter in the variance of the optimal Gaussian distribution. It is worth noting that Gaussian exploration and the related results just described have not been obtained theoretically and endogenously in any discrete-time setting, even with entropy regularization and the LQ structure.

Moreover, we establish a direct connection between the solvability of the exploratory LQ problem and that of the classical LQ problem. We prove that as the exploration weight in the former decays to zero, the optimal Gaussian control distribution and its value function converge respectively to the optimal Dirac measure and the value function of the classical LQ problem, a desirable result for practical learning purposes.

We also observe that, beyond the LQ problems and under proper conditions, the Gaussian distribution remains optimal for a much larger class of control problems, namely, problems with drift and volatility linear in control and reward functions linear or quadratic in control, even if the dependence on state is nonlinear. Such a family of problems can be seen as the locally linear-quadratic approximation to more general stochastic control problems whose state dynamics are linearized in the control variables and the reward functions are locally approximated by quadratic control functions (Todorov and Li, 2005; Li and Todorov, 2007). Note also that although such iterative LQ approximation generally has different parameters at different local state-action pairs, our result on the optimality of Gaussian distribution under the exploratory LQ framework still holds at any local point, and therefore justifies, from a stochastic control perspective, why Gaussian distribution is commonly

used in the RL practice for exploration (see, among others, Haarnoja et al., 2017; Haarnoja et al., 2018; Nachum et al., 2018), beyond its simplicity for sampling.

Finally, we need to stress that entropy regularization is just one approach to generate exploration, and it may not be effective or efficient for many problems. For example, it fails for the so-called combination lock problem which is a very simple MDP (see, e.g., Leffler et al., 2007). In general, simple randomization for exploration may not work effectively for various problems; see Agarwal et al. (2020); Matheron et al. (2019); Osband et al. (2017). All these cited references are for discrete time/space, but the underlying reason why entropy regularization may not work properly carries over to the continuous setting. By no means is this paper to advocate entropy regularization as a superior means for exploration. Rather, the main objective is to set up a theoretical framework for continuous RL problems, and entropy regularization is used as an example to demonstrate the usefulness of the framework.

The rest of the paper is organized as follows. In Section 2, we motivate and propose the relaxed stochastic control formulation involving an exploratory state dynamics and an entropy-regularized reward function for our RL problem. We then present the associated Hamilton-Jacobi-Bellman (HJB) equation and the optimal control distribution for general entropy-regularized stochastic control problems in Section 3. In Section 4, we study the special LQ problem in both the state-independent and state-dependent reward cases, corresponding respectively to the multi-armed bandit problem and the general RL problem in discrete time, and derive the optimality of Gaussian exploration. We discuss the connections between the exploratory LQ problem and the classical LQ problem in Section 5, establish the solvability equivalence of the two and the convergence result for vanishing exploration, and finally characterize the cost of exploration. We conclude in Section 6. Some technical contents and proofs are relegated to Appendices.

## 2. RL Formulation in Continuous Time and Spaces

In this section we introduce an exploratory stochastic control problem and provide its motivation in the context of RL.

## 2.1 Exploratory Formulation

Consider a filtered probability space (Ω , F , P ; {F t } t ≥ 0 ) in which we define an {F t } t ≥ 0 -adapted Brownian motion W = { W t , t ≥ 0 } . An 'action space' U is given, representing the constraints on an agent's decisions ('controls' or 'actions'). An admissible ( open-loop ) control u = { u t , t ≥ 0 } is an {F t } t ≥ 0 -adapted measurable process taking values in U .

The classical stochastic control problem is to control the state (or 'feature') dynamics 3

<!-- formula-not-decoded -->

where (and throughout this paper) x is a generic variable representing a current state of the system dynamics. The aim of the control is to achieve the maximum expected total

3. We assume that both the state and the control are scalar-valued, only for notational simplicity. There is no essential difficulty to carry out our analysis with these being vector-valued.

discounted reward represented by the value function

<!-- formula-not-decoded -->

where r is the reward function, ρ &gt; 0 is the discount rate, and A cl ( x ) denotes the set of all admissible controls which in general may depend on x .

In the classical setting, where the model is fully known (namely, when the functions b, σ and r are fully specified) and dynamic programming is applicable, the optimal control can be derived and represented as a deterministic mapping from the current state to the action space U , u ∗ t = u ∗ ( x ∗ t ). The mapping u ∗ is called an optimal feedback control (or 'policy' or 'law'); this feedback control is derived at t = 0 and will be carried out through [0 , ∞ ). 4

In contrast, in the RL setting, where the underlying model is not known and therefore dynamic learning is needed, the agent employs exploration to interact with and learn the unknown environment through trial and error. The key idea is to model exploration by a distribution of controls π = { π t ( u ) , t ≥ 0 } over the control space U from which each 'trial' is sampled. 5 We can therefore extend the notion of controls to distributions. 6 The agent executes a control for N rounds over the same time horizon, while at each round, a classical control is sampled from the distribution π . The reward of such a policy becomes accurate enough when N is large. This procedure, known as policy evaluation , is considered as a fundamental element of most RL algorithms in practice (Sutton and Barto, 2018). Hence, for evaluating such a policy distribution in our continuous time setting, it is necessary to consider the limiting situation as N →∞ .

In order to capture the essential idea for doing this, let us first examine the special case when the reward depends only on the control, namely, r ( x u t , u t ) = r ( u t ) . One then considers N identical independent copies of the control problem in the following way: at round i , i = 1 , 2 , . . . , N, a control u i is sampled under the (possibly random) control distribution π , and executed for its corresponding copy of the control problem (1)-(2). Then, at each fixed time t , it follows, from the law of large numbers (and under certain mild technical conditions), that the average reward over [ t, t +∆ t ], with ∆ t small enough, should satisfy, as N →∞ ,

<!-- formula-not-decoded -->

For a general reward r ( x u t , u t ) which also depends on the state, we first need to describe how exploration might alter the state dynamics (1) by defining appropriately its 'exploratory' version. For this, we look at the effect of repetitive learning under a given

4. In general, feedback controls are easier to implement as they respond directly to the current state of the controlled dynamics.

5. As will be evident in the sequel, rigorously speaking, π t ( · ) is a probability density function for each t ≥ 0. With a slight abuse of terminology, we will not distinguish a density function from its corresponding probability distribution or probability measure and thus will use these terms interchangeably in this paper . Such nomenclature is common in the RL literature.

6. A classical control u = { u t , t ≥ 0 } can be regarded as a Dirac distribution (or 'measure') π = { π t ( u ) , t ≥ 0 } where π t ( · ) = δ u t ( · ). In a similar fashion, a feedback policy u t = u ( x u t ) can be embedded as a Dirac measure π t ( · ) = δ u ( x u t ) ( · ), parameterized by the current state x u t .

control distribution, say π , for N rounds. Let W i t , i = 1 , 2 , . . . , N , be N independent sample paths of the Brownian motion W t , and x i t , i = 1 , 2 , . . . , N , be the copies of the state process respectively under the controls u i , i = 1 , 2 , . . . , N , each sampled from π . Then, the increments of these state process copies are, for i = 1 , 2 , . . . , N ,

<!-- formula-not-decoded -->

Each such process x i , i = 1 , 2 , . . . , N , can be viewed as an independent sample from the exploratory state dynamics X π . The superscript π of X π indicates that each x i is generated according to the classical dynamics (3), with the corresponding u i sampled independently under this policy π.

It then follows from (3) and the law of large numbers that, as N →∞ ,

<!-- formula-not-decoded -->

In the above, we have implicitly applied the (reasonable) assumption that both π t and X π t are independent of the increments of the Brownian motion sample paths, which are identically distributed over [ t, t +∆ t ].

Similarly, as N →∞ ,

<!-- formula-not-decoded -->

As we see, not only ∆ x i t but also (∆ x i t ) 2 are affected by repetitive learning under the given policy π .

Finally, as the individual state x i t is an independent sample from X π t , we have that ∆ x i t and (∆ x i t ) 2 , i = 1 , 2 , . . . , N , are the independent samples from ∆ X π t and (∆ X π t ) 2 , respectively. As a result, the law of large numbers gives that as N →∞ ,

<!-- formula-not-decoded -->

This interpretation, together with (4) and (5), motivates us to propose the exploratory version of the state dynamics, namely,

<!-- formula-not-decoded -->

where the coefficients ˜ b ( · , · ) and ˜ σ ( · , · ) are defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

with P ( U ) being the set of density functions of probability measures on U that are absolutely continuous with respect to the Lebesgue measure.

We will call (6) the exploratory formulation of the controlled state dynamics, and ˜ b ( · , · ) and ˜ σ ( · , · ) in (7) and (8), respectively, the exploratory drift and the exploratory volatility . 7 In a similar fashion, as N →∞ ,

<!-- formula-not-decoded -->

Hence, the reward function r in (2) needs to be modified to the exploratory reward

<!-- formula-not-decoded -->

## 2.2 Entropy Regularization

We have introduced a relaxed stochastic control formulation to model exploration and learning in RL. If, however, the model is fully known, exploration and learning would not be needed at all and the control distributions would all degenerate to the Dirac measures, and we would then be in the realm of the classical stochastic control. Thus, in the RL context, we need to add a 'regularization term' to account for model uncertainty and to encourage exploration. We use Shanon's differential entropy to measure the level of exploration:

<!-- formula-not-decoded -->

We therefore introduce the following entropy-regularized relaxed stochastic control problem

<!-- formula-not-decoded -->

where λ &gt; 0 is an exogenous exploration weight parameter capturing the trade-off between exploitation (the original reward function) and exploration (the entropy), A ( x ) is the set

7. The exploratory formulation (6), inspired by repetitive learning, is consistent with the notion of relaxed control in the control literature (see, e.g., Fleming and Nisio, 1984; El Karoui et al., 1987; Zhou, 1992; Kurtz and Stockbridge, 1998, 2001). Indeed, let f : R ↦→ R be a bounded and twice continuously differentiable function, and consider the infinitesimal generator associated to the classical controlled process (1),

<!-- formula-not-decoded -->

In the classical relaxed control framework, the controlled dynamics is replaced by the six-tuple (Ω , F , F = {F t } t ≥ 0 , P , X π , π ), such that X π 0 = x and

<!-- formula-not-decoded -->

It is easy to verify that our proposed exploratory formulation (6) agrees with the above martingale formulation. However, even though the mathematical formulations are equivalent, the motivations of the two are entirely different. Relaxed control was introduced to mainly deal with the existence of optimal controls, whereas the exploratory formulation here is motivated by learning and exploration in RL.

of the admissible control distributions (which may in general depend on x ), and V is the value function. 8

The precise definition of A ( x ) depends on the specific dynamic model under consideration and the specific problems one wants to solve, which may vary from case to case. Here, we first provide some of the 'minimal' requirements for A ( x ). Denote by B ( U ) the Borel algebra on U . An admissible control distribution is a measure-valued (or precisely a density-function-valued) process π = { π t , t ≥ 0 } satisfying at least the following properties:

- (i) for each t ≥ 0, π t ∈ P ( U ) a.s.;
- (ii) for each A ∈ B ( U ), { ∫ A π t ( u ) du, t ≥ 0 } is F t -progressively measurable;

(iii) the stochastic differential equation (SDE) (6) has a unique strong solution X π = { X π t , t ≥ 0 } if π is applied;

(iv) the expectation on the right hand side of (12) is finite.

Naturally, there could be additional requirements depending on specific problems. For the linear-quadratic control case, which will be the main focus of the paper, we define A ( x ) precisely in Section 4.

Finally, analogous to the classical control formulation, A ( x ) contains open-loop control distributions that are measure-valued stochastic processes . We will also consider feedback control distributions. Specifically, a deterministic mapping π ( · ; · ) is called a feedback control (distribution) if i) π ( · ; x ) is a density function for each x ∈ R ; ii) the following SDE (which is the system dynamics after the feedback law π ( · ; · ) is applied)

<!-- formula-not-decoded -->

has a unique strong solution { X t ; t ≥ 0 } ; and iii) the open-loop control π = { π t , t ≥ 0 } ∈ A ( x ) where π t := π ( · ; X t ). In this case, the open-loop control π is said to be generated from the feedback control law π ( · ; · ) with respect to x .

## 3. HJB Equation and Optimal Control Distributions

We present the general procedure for solving the optimization problem (12). The arguments are informal and a rigorous analysis will be carried out in the next section.

To this end, applying the classical Bellman's principle of optimality, we have

<!-- formula-not-decoded -->

Proceeding with standard arguments, we deduce that V satisfies the Hamilton-JacobiBellman (HJB) equation

<!-- formula-not-decoded -->

8. In the RL community, λ is also known as the temperature parameter, which we will be using occasionally.

or

<!-- formula-not-decoded -->

where v denotes the generic unknown solution of the equation.

Recalling that π ∈ P ( U ) if and only if

<!-- formula-not-decoded -->

we can solve the (constrained) maximization problem on the right hand side of (15) to get a feedback control:

<!-- formula-not-decoded -->

For each given initial state x ∈ R , this feedback control in turn generates an optimal openloop control

<!-- formula-not-decoded -->

where { X ∗ t , t ≥ 0 } solves (6) when the feedback control law π ∗ ( · ; · ) is applied and assuming that { π ∗ t , t ≥ 0 } ∈ A ( x ) . 9

Formula (17) above elicits qualitative understanding about optimal explorations. We further investigate this in the next section.

## 4. The Linear-Quadratic Case

We now focus on the family of entropy-regularized (relaxed) stochastic control problems with linear state dynamics and quadratic rewards, in which

<!-- formula-not-decoded -->

where A,B,C,D ∈ R , and

<!-- formula-not-decoded -->

where M ≥ 0, N &gt; 0 , R,P,Q ∈ R .

In the classical control literature, this type of linear-quadratic (LQ) control problems is one of the most important, not only because it admits elegant and simple solutions but also because more complex, nonlinear problems can be approximated by LQ problems. As is standard with LQ control, we assume that the control set is unconstrained, namely, U = R .

9. We stress that the procedure described in this section, while standard, is informal. A rigorous treatment requires a precise definition of A ( x ) and a verification that indeed { π ∗ t , t ≥ 0 } ∈ A ( x ) . This will be carried out in the study of the linear-quadratic case in the following sections.

Fix an initial state x ∈ R . For each open-loop control π ∈ A ( x ) , denote its mean and variance processes µ t , σ 2 t , t ≥ 0 , by

<!-- formula-not-decoded -->

Then, the state SDE (6) becomes

<!-- formula-not-decoded -->

Further, denote

<!-- formula-not-decoded -->

Next, we specify the associated set of admissible controls A ( x ): π ∈ A ( x ), if

- (i) for each t ≥ 0, π t ∈ P ( R ) a.s.;
- (ii) for each A ∈ B ( R ), { ∫ A π t ( u ) du, t ≥ 0 } is F t -progressively measurable;
- (iii) for each t ≥ 0, E [ ∫ t 0 ( µ 2 s + σ 2 s ) ds ] &lt; ∞ ;
- (iv) with { X π t , t ≥ 0 } solving (22), lim inf T →∞ e -ρT E [ ( X π T ) 2 ] = 0;
- (v) with { X π t , t ≥ 0 } solving (22), E [∫ ∞ 0 e -ρt | L ( X π t , π t ) | dt ] &lt; ∞ .

In the above, condition (iii) is to ensure that for any π ∈ A ( x ), both the drift and volatility terms of (22) satisfy a global Lipschitz condition and a type of linear growth condition in the state variable and, hence, the SDE (22) admits a unique strong solution X π . Condition (iv) will be used to ensure that dynamic programming and verification are applicable for this model, as will be evident in the sequel. Finally, the reward is finite under condition (v).

We are now ready to introduce the entropy-regularized relaxed stochastic LQ problem

<!-- formula-not-decoded -->

with r as in (20) and X π as in (22).

In the following two subsections, we derive explicit solutions for both cases of stateindependent and state-dependent rewards.

## 4.1 The Case of State-Independent Reward

We start with the technically less challenging case r ( x, u ) = -( N 2 u 2 + Qu ) , namely, the reward is state (feature) independent. In this case, the system dynamics becomes irrelevant. However, the problem is still interesting in its own right as it corresponds to the stateindependent RL problem, which is known as the continuous-armed bandit problem in the continuous time setting (Mandelbaum, 1987; Kaspi and Mandelbaum, 1998).

Following the derivation in the previous section, the optimal feedback control in (17) reduces to

<!-- formula-not-decoded -->

Therefore, the optimal feedback control distribution appears to be Gaussian . More specifically, at any present state x , the agent should embark on exploration according to the Gaussian distribution with mean and variance given, respectively, by CDxv ′′ ( x )+ Bv ′ ( x ) -Q N -D 2 v ′′ ( x ) and λ N -D 2 v ′′ ( x ) . Note that in deriving the above, we have used that N -D 2 v ′′ ( x ) &gt; 0, x ∈ R , a condition that will be justified and discussed later on.

Remark 1 If we examine the derivation of (24) more closely, we easily see that the optimality of the Gaussian distribution still holds as long as the state dynamics is linear in control and the reward is quadratic in control, whereas the dependence of both on the state can be generally nonlinear.

Substituting (24) back to (14), the HJB equation becomes, after straightforward calculations,

<!-- formula-not-decoded -->

In general, this nonlinear equation has multiple smooth solutions, even among quadratic polynomials that satisfy N -D 2 v ′′ ( x ) &gt; 0. One such solution is a constant, given by

<!-- formula-not-decoded -->

with the corresponding optimal feedback control distribution (24) being

<!-- formula-not-decoded -->

It turns out that the right hand side of the above is independent of the current state x . So the optimal feedback control distribution is the same across different states. Note that the classical LQ problem with the state-independent reward function r ( x, u ) = -( N 2 u 2 + Qu ) clearly has the optimal control u ∗ = -Q N , which is also state-independent and is nothing else than the mean of the optimal Gaussian feedback control π ∗ .

The following result establishes that the constant v is indeed the value function V and that the feedback control π ∗ defined by (27) is optimal. Henceforth, we denote, for notational convenience, by N ( ·| µ, σ 2 ) the density function of a Gaussian random variable with mean µ and variance σ 2 .

Theorem 2 If r ( x, u ) = -( N 2 u 2 + Qu ) , then the value function in (23) is given by

<!-- formula-not-decoded -->

and the optimal feedback control distribution is Gaussian, with

<!-- formula-not-decoded -->

Moreover, the associated optimal state process, { X ∗ t , t ≥ 0 } , under π ∗ ( · ; · ) is the unique solution of the SDE

<!-- formula-not-decoded -->

Proof Let v ( x ) ≡ v be the constant solution to the HJB equation (25) defined by (26). Then, the corresponding feedback optimizer π ∗ ( u ; x ) = N ( u ∣ ∣ -Q N , λ N ) follows immediately from (24). Let π ∗ = { π ∗ t , t ≥ 0 } be the open-loop control generated from π ∗ ( · ; · ). It is straightforward to verify that π ∗ ∈ A ( x ). 10

Now, for any π ∈ A ( x ) and T ≥ 0, it follows from the HJB equation (14) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since π ∈ A ( x ), the dominated convergence theorem yields that, as T →∞ ,

<!-- formula-not-decoded -->

and, thus, v ≥ V ( x ), for ∀ x ∈ R . On the other hand, π ∗ has been derived as the maximizer for the right hand side of (14); hence

<!-- formula-not-decoded -->

Replacing the inequalities by equalities in the above argument and sending T to infinity, we conclude that for x ∈ R .

Finally, the exploratory dynamics equation (28) follows readily from substituting µ ∗ t = -Q N and ( σ ∗ t ) 2 = λ N , t ≥ 0, into (22).

10. Since the state process is irrelevant in the current case, it is not necessary to verify the admissibility condition (iv).

<!-- formula-not-decoded -->

It is possible to obtain explicit solutions to the SDE (28) for most cases, which may be useful in designing exploration algorithms based on the theoretical results derived in this paper. We relegate this discussion about solving (28) explicitly to Appendix A.

The above solution suggests that when the reward is independent of the state, so is the optimal feedback control distribution with density N ( · | -Q N , λ N ). This is intuitive since objective (12) in this case does not explicitly distinguish between states. 11

A remarkable feature of the derived optimal distribution N ( · | -Q N , λ N ) is that its mean coincides with the optimal control of the original, non-exploratory LQ problem, whereas the variance is determined by the temperature parameter λ . In the context of continuousarmed bandit problem, this result stipulates that the mean is concentrated on the current incumbent of the best arm and the variance is determined by the temperature parameter. The more weight put on the level of exploration, the more spread out the exploration becomes around the current best arm. This type of exploration/exploitation strategies is clearly intuitive and, in turn, gives a guidance on how to actually choose the temperature parameter in practice: it is nothing else than the variance of the exploration the agent wishes to engage in (up to a scaling factor being the quadratic coefficient of the control in the reward function).

However, we shall see in the next section that when the reward depends on the local state, the optimal feedback control distribution genuinely depends on the state.

## 4.2 The Case of State-Dependent Reward

We now consider the general case with the reward depending on both the control and the state, namely,

<!-- formula-not-decoded -->

We will be working with the following assumption.

<!-- formula-not-decoded -->

11. Similar observation can be made for the (state-independent) pure entropy maximization formulation, where the goal is to solve

<!-- formula-not-decoded -->

This problem becomes relevant when λ → ∞ in the entropy-regularized objective (23), corresponding to the extreme case of pure exploration without considering exploitation (i.e., without maximizing any reward). To solve problem (29), we can pointwisely maximize its integrand, leading to the stateindependent optimization problem

<!-- formula-not-decoded -->

It is then straightforward that the optimal control distribution π ∗ is, for all t ≥ 0, the uniform distribution. This is in accordance with the traditional static setting where uniform distribution achieves maximum entropy (Shannon, 2001).

This assumption requires a sufficiently large discount rate. It reduces to the more familiar condition ρ &gt; 2 A + C 2 when the cross term in the quadratic reward R = 0, which in turn ensures that lim inf T →∞ e -ρT E [ ( X π T ) 2 ] = 0 for any admissible π and, hence, the corresponding reward value is finite.

Following an analogous argument as for (24), we deduce that a candidate optimal feedback control is given by

<!-- formula-not-decoded -->

In turn, denoting by µ ∗ ( x ) and ( σ ∗ ( x )) 2 the mean and variance of π ∗ ( · ; x ) given above, the HJB equation (14) becomes

<!-- formula-not-decoded -->

Reorganizing, the above reduces to

<!-- formula-not-decoded -->

Under Assumption 3 and the additional condition R 2 &lt; MN (which holds automatically if R = 0, M &gt; 0 and N &gt; 0, a standard case in the classical LQ problems), one smooth solution to the HJB equation (32) is given by

<!-- formula-not-decoded -->

where 12

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For this particular solution, given by v ( x ) above, we can verify that k 2 &lt; 0, due to Assumption 3 and R 2 &lt; MN . Hence, v is concave, a property that is essential in proving that it is actually the value function. 13 On the other hand, N -D 2 v ′′ ( x ) = N -k 2 D 2 &gt; 0, ensuring that k 0 is well defined.

Next, we state one of the main results of this paper.

Theorem 4 Suppose the reward function is given by

<!-- formula-not-decoded -->

with M ≥ 0 , N &gt; 0 , R,Q,P ∈ R and R 2 &lt; MN . Furthermore, suppose that Assumption 3 holds. Then, the value function in (23) is given by

<!-- formula-not-decoded -->

where k 2 , k 1 and k 0 are as in (36), (37) and (38), respectively. Moreover, the optimal feedback control is Gaussian, with its density function given by

<!-- formula-not-decoded -->

12. In general, there are multiple solutions to (32). Indeed, applying, for example, a generic quadratic function ansatz v ( x ) = 1 2 a 2 x 2 + a 1 x + a 0 , x ∈ R , in (32) yields the system of algebraic equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This system has two sets of solutions (as the quadratic equation (33) has, in general, two roots), leading to two quadratic solutions to the HJB equation (32). The one given through (36)-(38) is one of the two solutions.

13. Under Assumption 3 and R 2 &lt; MN , the HJB equation has an additional quadratic solution, which however is convex .

Finally, the associated optimal state process { X ∗ t , t ≥ 0 } under π ∗ ( · ; · ) is the unique solution of the SDE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A proof of this theorem follows essentially the same idea as that of Theorem 2, but it is more technically involved, mainly for verifying the admissibility of the candidate optimal control. To ease the presentation, we defer it to Appendix B.

glyph[negationslash]

Remark 5 As in the state-independent case (see Appendix A), the solution to the SDE (41) can be expressed through the Doss-Saussman transformation if D = 0 .

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the function F is given by

<!-- formula-not-decoded -->

and the process Y t , t ≥ 0 , is the unique pathwise solution to the random ODE

<!-- formula-not-decoded -->

with ˜ A := A + B ( k 2 ( B + CD ) -R ) N -k 2 D 2 , ˜ B := B ( k 1 B -Q ) N -k 2 D 2 , ˜ C 1 := C + D ( k 2 ( B + CD ) -R ) N -k 2 D 2 ,

˜ C 2 := D ( k 1 B -Q ) N -k 2 D 2 and ˜ D := λD 2 N -k 2 D 2 .

If C + D ( k 2 ( B + CD ) -R ) N -k 2 D 2 = 0 and ˜ A = 0 , then it follows from direct computation that glyph[negationslash]

<!-- formula-not-decoded -->

We leave the detailed derivations to the interested readers.

glyph[negationslash]

The above results demonstrate that, for the general state and control dependent reward case, the optimal actions over R also depend on the current state x , which are selected according to a state-dependent Gaussian distribution (40) with a state-independent variance λ N -k 2 D 2 . Note that if D = 0, then λ N -k 2 D 2 &lt; λ N (since k 2 &lt; 0). Therefore, the exploration variance in the general state-dependent case is strictly smaller than λ N , the one in the stateindependent case. Recall that D is the coefficient of the control in the diffusion term of

the state dynamics, generally representing the level of randomness of the environment. 14 Therefore, volatility impacting actions reduce the need for exploration. This is because while an exploration strategy with larger variance could lead the RL agent to explore more states, it is also more costly with respect to the objective function. In the state-dependent reward case, reward depends on state whose dynamics is in turn affected by D . The agent can therefore leverage on D to explore equally broad area of the state space with a smaller exploration variance.

On the other hand, the mean of the Gaussian distribution does not explicitly depend on λ . The implication is that the agent should concentrate on the most promising region in the action space while randomly selecting actions to interact with the unknown environment. It is intriguing that the entropy-regularized RL formulation separates the exploitation from exploration, respectively through the mean and variance of the resulting optimal Gaussian distribution.

Remark 6 It should be noted that it is the optimal feedback control distribution, not the open-loop control generated from the feedback control, that has the Gaussian distribution. More precisely, π ∗ ( · ; x ) defined by (40) is Gaussian for each and every x , but the measurevalued process with the density function

<!-- formula-not-decoded -->

where { X ∗ t , t ≥ 0 } is the solution of the exploratory dynamics under the feedback control π ∗ ( · ; · ) with any fixed initial state, say, X ∗ 0 = x 0 , is in general not Gaussian for any t &gt; 0 . The reason is that, for each t &gt; 0 , the right hand side of (42) is a composition of the Gaussian density function and a random variable X ∗ t whose distribution is, in general, unknown. We stress that the Gaussian property of the optimal feedback control is more important and relevant in the RL context, as it stipulates that, at any given state, if one undertakes exploration then she should follow Gaussian. The open-loop control { π ∗ t , t ≥ 0 } , generated from the Gaussian feedback control, is just what the agent would end up with if she follows Gaussian exploration at every state.

Remark 7 In direct analogy with the classical stochastic LQ control theory (e.g., Yong and Zhou, 1999, Chapter 6), a generalization to high dimensions (both of state and control) is rather straightforward (save for the notational complexity), thanks to the multidimensional Itˆ o formula. Specifically, one can derive a similar HJB equation (15) in the multi-dimensional case. The Hamiltonian can, then, be similarly maximized and the resulting distribution remains to be multivaiate Gaussian. The calculation of the entropy of a multivaiate Gaussian is also simliar; for details see Wang (2019) in which a multidimensional action space is involved.

Remark 8 As noted earlier (see Remark 1), the optimality of the Gaussian distribution is still valid for problems with dynamics

<!-- formula-not-decoded -->

14. For example, in the Black-Scholes model, D is the volatility parameter of the underlying stock.

and reward function of the form r ( x, u ) = -1 2 r 2 ( x ) u 2 -r 1 ( x ) u -r 0 ( x ) , where the functions A,B,C,D,r 2 , r 1 and r 0 are possibly nonlinear. Naturally, we need proper assumptions to ensure: 1) the existence and uniqueness of the solution to the state equation and the regularity of the value function, 2) the validity of the inequality r 2 ( x ) -D ( x ) 2 v ′′ ( x ) &gt; 0 , leading to a legitimate Gaussian distribution (note this condition is weaker than v ′′ ( x ) &lt; 0 since the former may still hold even if the latter does not as long as r 2 ( x ) is sufficiently large), and 3) the admissibility of the constructed Gaussian distribution and the validity of the verification argument (this may require a generalization of Assumption 3). To derive precise results along this line remains an interesting research problem.

## 5. The Cost and Effect of Exploration

Motivated by the necessity of exploration facing the typically unknown environment in an RL setting, we have formulated and analyzed a new class of stochastic control problems that combine entropy-regularized criteria and relaxed controls. We have also derived closedform solutions and presented verification results for the important class of LQ problems. A natural question arises, namely, how to quantify the cost and effect of the exploration. This can be done by comparing our results to the ones for the classical stochastic LQ problems, which have neither entropy regularization nor control relaxation.

We carry out this comparison analysis next.

## 5.1 The Classical LQ Problem

We first briefly recall the classical stochastic LQ control problem in an infinite horizon with discounted reward. Let { W t , t ≥ 0 } be a standard Brownian motion defined on the filtered probability space (Ω , F , {F t } t ≥ 0 , P ) that satisfies the usual conditions. The controlled state process { x u t , t ≥ 0 } solves

<!-- formula-not-decoded -->

with given constants A,B,C and D, and the process { u t , t ≥ 0 } being a (classical, nonrelaxed) control.

The value function is defined as in (2),

<!-- formula-not-decoded -->

for x ∈ R , where the reward function r ( · , · ) is given by (20). Here, the admissible set A cl ( x ) is defined as follows: u ∈ A cl ( x ) if

- (i) { u t , t ≥ 0 } is F t -progressively measurable;
- (ii) for each t ≥ 0, E [ ∫ t 0 ( u s ) 2 ds ] &lt; ∞ ;
- (iii) with { x u t , t ≥ 0 } solving (43), lim inf T →∞ e -ρT E [ ( x u T ) 2 ] = 0;
- (iv) with { x u t , t ≥ 0 } solving (43), E [ ∫ ∞ 0 e -ρt | r ( x u t , u t ) | dt ] &lt; ∞ .

The associated HJB equation is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the maximizer being, provided that N -D 2 w ′′ ( x ) &gt; 0,

<!-- formula-not-decoded -->

Standard verification arguments then yield that u is the optimal feedback control.

In the next section, we will establish a solvability equivalence between the entropyregularized relaxed LQ problem and the classical one.

## 5.2 Solvability Equivalence of Classical and Exploratory Problems

Given a reward function r ( · , · ) and a classical controlled process (1), the relaxed formulation (6) under the entropy-regularized objective is, naturally, a technically more challenging problem, compared to its classical counterpart.

In this section, we show that there is actually a solvability equivalence between the exploratory and the classical stochastic LQ problems, in the sense that the value function and optimal control of one problem lead directly to those of the other. Such equivalence enables us to readily establish the convergence result as the exploration weight λ decays to zero. Furthermore, it makes it possible to quantify the exploration cost, which we introduce in the sequel.

Theorem 9 The following two statements (a) and (b) are equivalent.

- (a) The function v ( x ) = 1 2 α 2 x 2 + α 1 x + α 0 + λ 2 ρ ( ln ( 2 πeλ N -α 2 D 2 ) -1 ) , x ∈ R , with α 0 , α 1 ∈ R and α 2 &lt; 0 , is the value function of the exploratory problem (23) and the corresponding optimal feedback control is

<!-- formula-not-decoded -->

- (b) The function w ( x ) = 1 2 α 2 x 2 + α 1 x + α 0 , x ∈ R , with α 0 , α 1 ∈ R and α 2 &lt; 0 , is the value function of the classical problem (44) and the corresponding optimal feedback control is

<!-- formula-not-decoded -->

Proof See Appendix C.

The above equivalence between statements (a) and (b) yields that if one problem is solvable, so is the other; and conversely, if one is not solvable, neither is the other.

## 5.3 Cost of Exploration

We define the exploration cost for a general RL problem to be the difference between the discounted accumulated rewards following the corresponding optimal open-loop controls under the classical objective (2) and the exploratory objective (12), net of the value of the entropy. Note that the solvability equivalence established in the previous subsection is important for this definition, not least because the cost is well defined only if both the classical and the exploratory problems are solvable.

Specifically, let the classical maximization problem (2) with the state dynamics (1) have the value function V cl ( · ) and optimal strategy { u ∗ t , t ≥ 0 } , and the corresponding exploratory problem have the value function V ( · ) and optimal control distribution { π ∗ t , t ≥ 0 } . Then, we define the exploration cost as

<!-- formula-not-decoded -->

for x ∈ R .

The first term of the right hand side above, V cl ( x ), is the optimal value of the original objective without exploration should the model be a priori fully known, while the second term is the value of the original objective under the solution that maximizes the regularized objective. 15 Hence, the exploration cost measures the loss in the original (i.e., non-regularized) objective due to exploration. 16

We next compute the exploration cost for the LQ case. As we show, this cost is surprisingly simple: it depends only on two 'agent-specific' parameters: the temperature parameter λ and the discounting parameter ρ .

Theorem 10 Assume that statement (a) (or equivalently, (b)) of Theorem 9 holds. Then, the exploration cost for the stochastic LQ problem is

<!-- formula-not-decoded -->

Proof Let { π ∗ t , t ≥ 0 } be the open-loop control generated by the feedback control π ∗ given in statement (a) with respect to the initial state x , namely,

<!-- formula-not-decoded -->

where { X ∗ t , t ≥ 0 } is the associated state process of the exploratory problem, starting from the state x , when π ∗ is applied. Then, we easily deduce that

<!-- formula-not-decoded -->

15. Here, the original objective under the control distribution { π ∗ t , t ≥ 0 } should be understood to be the average after large number of controls sampled from { π ∗ t , t ≥ 0 } ; see the explanations leading to (10).

16. This definition resembles that of average loss (Definition 3) introduced in Strehl and Littman (2008).

The desired result now follows immediately from the general definition in (47) and the expressions of V ( · ) in (a) and V cl ( · ) in (b).

In other words, the exploration cost for stochastic LQ problems can be completely predetermined by the learning agent through choosing her individual parameters λ and ρ , since the cost relies neither on the specific (unknown) linear state dynamics, nor on the quadratic reward structure.

Moreover, the exploration cost (48) depends on λ and ρ in a rather intuitive way: it increases as λ increases, due to more emphasis placed on exploration, or as ρ decreases, indicating an effectively longer horizon for exploration. 17

## 5.4 Vanishing Exploration

Herein, the exploration weight λ has been taken as an exogenous parameter reflecting the level of exploration desired by the learning agent. The smaller this parameter is, the more emphasis is placed on exploitation. When this parameter is sufficiently close to zero, the exploratory formulation is sufficiently close to the problem without exploration. Naturally, a desirable result is that if the exploration weight λ goes to zero, then the entropy-regularized LQ problem would converge to its classical counterpart. The following result makes this precise.

Theorem 11 Assume that statement (a) (or equivalently, (b)) of Theorem 9 holds. Then, for each x ∈ R ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof The weak convergence of the feedback controls follows from the explicit forms of π ∗ and u ∗ in statements (a) and (b), and the fact that α 1 , α 2 are independent of λ . The pointwise convergence of the value functions follows easily from the forms of V ( · ) and V cl ( · ), together with the fact that

<!-- formula-not-decoded -->

17. The connection between a discounting parameter and an effective length of time horizon is well known in the discrete time discounted reward formulation E [ ∑ t ≥ 0 γ t R t ] for classical Markov Decision Processes (MDP) (see, among others, Derman, 1970). This infinite horizon discounted problem can be viewed as an undiscounted, finite horizon problem with a random termination time T that is geometrically distributed with parameter 1 -γ . Hence, an effectively longer horizon with mean 1 1 -γ is applied to the optimization problem as γ increases. Since a smaller ρ in the continuous time objective (2) or (12) corresponds to a larger γ in the discrete time objective, we can see the similar effect of a decreasing ρ on the effective horizon of continuous time problems.

Moreover, for each x ∈ R ,

## 6. Conclusions

This paper approaches RL from a stochastic control perspective. Indeed, control and RL both deal with the problem of managing dynamic and stochastic systems by making the best use of available information. However, as a recent survey paper Recht (2019) points out, '... That the RL and control communities remain practically disjoint has led to the co-development of vastly different approaches to the same problems ....' It is our view that communication and exchange of ideas between the two fields are of paramount importance to the progress of both fields, for an old idea from one field may well be a fresh one to the other. The continuous-time relaxed stochastic control formulation employed in this paper exemplifies such a vision.

The main contributions of this paper are conceptual rather than algorithmic : we propose a stochastic relaxed control framework for studying continuous-time RL problems and, with the aid of stochastic control and stochastic calculus, we link entropy regularization and Gaussian exploration, two widespread research subjects in the current RL literature. This linkage is independent of the specific parameters of the underlying dynamics and reward function structure, as long as the dependence on actions is linear in the former and quadratic in the latter. The same can be said about other results of the paper, such as the separation between exploration and exploitation in the mean and variance of the resulting Gaussian distribution, and the cost of exploration. The explicit forms of the derived optimal Gaussian distributions do indeed depend on the model specifications which are unknown in the RL context. With regards to implementing RL algorithms based on our results for LQ problems, we can either do it in continuous time and space directly following, for example, Doya (2000), or modify the problem into an MDP one by discretizing the time, and then learn the parameters of the optimal Gaussian distribution following standard RL procedures (e.g., the so-called Q -learning). For that, our results may again be useful: they suggest that we only need to learn among the class of simpler Gaussian policies, i.e., π = N ( · | θ 1 x + θ 2 , φ ) (cf. (40)), rather than generic (nonlinear) parametrized Gaussian policy π θ,φ = N ( · | θ ( x ) , φ ( x )). We expect that this simpler functional form can considerably increase the learning speed. An application to mean-variance portfolio choice model is provided in Wang and Zhou (2020).

## Acknowledgments

We are grateful for comments from the seminar participants at UC Berkeley, Stanford, Fields Institute, Renyi Institute Budapest and Soochow University, and from the participants at the Columbia Engineering for Humanity Research Forum 'Business Analytics; Financial Services and Technologies' in New York, The 2018 Quantitative Methods in Finance Conference in Sydney, The 2019 Workshop on Frontier Areas in Financial Analytics in Toronto, The 2019 SIAM Conference on Financial Mathematics and Engineering in Toronto, The Thirteenth Annual Risk Management Conference in Singapore, The 2019 International Workshop on Probability, Uncertainty, and Quantitative Risk in Weihai, and The 2019 Annual Conference of the Institute of Financial Econometrics and Risk Management of Chinese Society of Management Science and Engineering in Dalian. We thank Jose Blanchet, Wendell Fleming, Kay Giesecke, Xin Guo, Miklos Rasonyi, Josef Teichmann, Ruodu Wang and

Renyuan Xu for helpful discussions and comments on the paper. We are also indebted to the Associate Editor and two anonymous referees for constructive comments which have led to an improved version of the paper. Wang gratefully acknowledges a postdoctoral fellowship through the Nie Center for Intelligent Asset Management at Columbia University. Zhou gratefully acknowledges financial support through a start-up grant at Columbia University and through the Nie Center for Intelligent Asset Management.

## Appendix A: Explicit Solutions to (28)

For a range of parameters, we derive explicit solutions to SDE (28) satisfied by the optimal state process { X ∗ t , t ≥ 0 } .

If D = 0, the SDE (28) reduces to

<!-- formula-not-decoded -->

If x ≥ 0 and BQ ≤ 0, the above equation has a nonnegative solution given by

<!-- formula-not-decoded -->

If x ≤ 0 and BQ ≥ 0, it has a nonpositive solution

<!-- formula-not-decoded -->

These two cases cover the special case when Q = 0 which is standard in the LQ control formulation. We are unsure if there is an explicit solution when neither of these assumptions is satisfied (e.g., when x ≥ 0 and BQ &gt; 0).

If C = 0, the SDE (28) becomes

<!-- formula-not-decoded -->

and its unique solution is given by

<!-- formula-not-decoded -->

glyph[negationslash]

if A = 0, and by if A = 0.

glyph[negationslash]

If C = 0 and D = 0, then the diffusion coefficient of SDE (28) is C 2 in the unknown, with the first and second order derivatives being bounded. Hence, (28) can be solved explicitly using the Doss-Saussman transformation (see, e.g., Karatzas and Shreve, 1991, pp 295-297). This transformation uses the ansatz

<!-- formula-not-decoded -->

for some deterministic function F and an adapted process Y t , t ≥ 0, solving a random ODE. Applying Itˆ o's formula to (49) and using the dynamics in (28), we deduce that F solves, for each fixed y , the ODE

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

Moreover, Y t , t ≥ 0, is the unique pathwise solution to the random ODE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

It is easy to verify that both equations (50) and (51) have a unique solution. Solving (50), we obtain

<!-- formula-not-decoded -->

This, in turn, leads to the explicit expression of the function G ( z, y ).

## Appendix B: Proof of Theorem 4

Recall that the function v , where v ( x ) = 1 2 k 2 x 2 + k 1 x + k 0 , x ∈ R , where k 2 , k 1 and k 0 are defined by (36), (37) and (38), respectively, satisfies the HJB equation (14).

Throughout this proof we fix the initial state x ∈ R . Let π ∈ A ( x ) and X π be the associated state process solving (22) with π being used. Let T &gt; 0 be arbitrary. Define the stopping times τ π n := { t ≥ 0 : ∫ t 0 ( e -ρt v ′ ( X π t )˜ σ ( X π t , π t ) ) 2 dt ≥ n } , for n ≥ 1. Then, Itˆ o's formula yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking expectations, using that v solves the HJB equation (14) and that π is in general suboptimal yield

<!-- formula-not-decoded -->

Classical results yield that E [ sup 0 ≤ t ≤ T | X π t | 2 ] ≤ K (1+ x 2 ) e KT , for some constant K &gt; 0 independent of n (but dependent on T and the model coefficients). Sending n → ∞ , we deduce that

<!-- formula-not-decoded -->

where we have used the dominated convergence theorem and that π ∈ A ( x ).

Next, we recall the admissibility condition lim inf T →∞ e -ρT E [ ( X π T ) 2 ] = 0 . This, together with the fact that k 2 &lt; 0, lead to lim sup T →∞ E [ e -ρT v ( X π T ) ] = 0. Applying the dominated convergence theorem once more yields

<!-- formula-not-decoded -->

for each x ∈ R and π ∈ A ( x ). Hence, v ( x ) ≥ V ( x ), for all x ∈ R .

On the other hand, we deduce that the right hand side of (14) is maximized at

<!-- formula-not-decoded -->

Let π ∗ = { π ∗ t , t ≥ 0 } be the open-loop control distribution generated from the above feedback law along with the corresponding state process { X ∗ t , t ≥ 0 } with X ∗ 0 = x , and assume for now that π ∗ ∈ A ( x ). Then

<!-- formula-not-decoded -->

Noting that lim inf T →∞ E [ e -ρT v ( X ∗ T ) ] ≤ lim sup T →∞ E [ e -ρT v ( X ∗ T ) ] = 0, and applying the dominated convergence theorem yield

<!-- formula-not-decoded -->

for any x ∈ R . This proves that v is indeed the value function, namely v ≡ V .

It remains to show that π ∗ ∈ A ( x ). First, we verify that

<!-- formula-not-decoded -->

where { X ∗ t , t ≥ 0 } solves the SDE (41). To this end, Itˆ o's formula yields, for any T ≥ 0 ,

<!-- formula-not-decoded -->

Following similar arguments as in the proof of Lemma 12 in Appendix C, we can show that E [( X ∗ T ) 2 ] contains the terms e (2 ˜ A + ˜ C 1 2 ) T and e ˜ AT .

If 2 ˜ A + ˜ C 1 2 ≤ ˜ A , then ˜ A ≤ 0, in which case (52) easily follows. Therefore, to show (52), it remains to consider the case in which the term e (2 ˜ A + ˜ C 1 2 ) T dominates e ˜ AT , as T →∞ . In turn, using that k 2 solves the equation (33), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that the first fraction is nonpositive due to k 2 &lt; 0, while the second fraction is bounded for any k 2 &lt; 0. Using Assumption 3 on the range of ρ , we then easily deduce (52). Next, we establish the admissibility constraint

<!-- formula-not-decoded -->

The definition of L and the form of r ( x, u ) yield

<!-- formula-not-decoded -->

where we have applied similar computations as in the proof of Theorem 10.

Recall that

<!-- formula-not-decoded -->

It is then clear that it suffices to prove E [∫ ∞ 0 e -ρt ( X ∗ t ) 2 dt ] &lt; ∞ , which follows easily since, as shown in (54), we obtained that ρ &gt; 2 ˜ A + ˜ C 1 2 under Assumption 3. The remaining admissibility conditions for π ∗ can be easily verified.

## Appendix C: Proof of Theorem 7

We first note that when (a) holds, the function v solves the HJB equation (32) of the exploratory LQ problem. Similarly for w of the classical LQ problem when (b) holds.

Next, we prove the equivalence between (a) and (b). First, a comparison between the two HJB equations (32) and (45) yields that if v in (a) solves the former, then w in (b) solves the latter, and vice versa.

Throughout this proof, we let x be fixed, being the initial state of both the exploratory problem in statement (a) and the classical problem in statement (b). Let π ∗ = { π ∗ t , t ≥ 0 } and u ∗ = { u ∗ t , t ≥ 0 } be respectively the open-loop controls generated by the feedback controls π ∗ and u ∗ of the two problems, and X ∗ = { X ∗ t , t ≥ 0 } and x ∗ = { x ∗ t , t ≥ 0 } be respectively the corresponding state processes, both starting from x . It remains to establish

the equivalence between the admissibility of π ∗ for the exploratory problem and that of u ∗ for the classical problem. To this end, we first compute E [( X ∗ T ) 2 ] and E [( x ∗ T ) 2 ].

To ease the presentation, we rewrite the exploratory dynamics of X ∗ under π ∗ as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A 1 := A + B ( α 2 ( B + CD ) -R ) N -α 2 D 2 , A 2 := B ( α 1 B -Q ) N -α 2 D 2 , B 1 := C + D ( α 2 ( B + CD ) -R ) N -α 2 D 2 ,

<!-- formula-not-decoded -->

Similarly, the classical dynamics of x ∗ under u ∗ solves

<!-- formula-not-decoded -->

The desired equivalence of the admissibility then follows from the following lemma.

<!-- formula-not-decoded -->

Proof Denote n ( t ) := E [ X ∗ t ], for t ≥ 0. Then, a standard argument involving a series of stopping times and the dominated convergence theorem yields the ODE

<!-- formula-not-decoded -->

glyph[negationslash]

whose solution is n ( t ) = ( x + A 2 A 1 ) e A 1 t -A 2 A 1 , if A 1 = 0, and n ( t ) = x + A 2 t , if A 1 = 0. Similarly, the function m ( t ) := E [ ( X ∗ t ) 2 ] , t ≥ 0, solves the ODE

<!-- formula-not-decoded -->

We can also show that n ( t ) = E [ x ∗ t ] , and deduce that ˆ m ( t ) := E [ ( x ∗ t ) 2 ] , t ≥ 0, satisfies

<!-- formula-not-decoded -->

Next, we find explicit solutions to the above ODEs corresponding to various conditions on the parameters.

(a) If A 1 = B 2 1 = 0, then direct computation gives n ( t ) = x + A 2 t , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

(b) If A 1 = 0 and B 2 = 0, we have n ( t ) = x + A 2 t

<!-- formula-not-decoded -->

glyph[negationslash]

(c) If A 1 = 0 and A 1 + B 2 1 = 0, then n ( t ) = ( x + A 2 A 1 ) e A 1 t -A 2 A 1 . Further calculations yield glyph[negationslash]

m ( t ) = ( x 2 + A 1 ( B 2 2 + C 1 ) -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 ) e A 1 t + 2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 1 te A 1 t -A 1 ( B 2 2 + C 1 ) -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 , ˆ m ( t ) = ( x 2 + A 1 B 2 2 -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 ) e A 1 t + 2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 1 te A 1 t -A 1 B 2 2 -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 . (d) If A 1 = 0 and 2 A 1 + B 2 1 = 0, we have n ( t ) = ( x + A 2 A 1 ) e A 1 t -A 2 A 1 , and m ( t ) = 2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 2 1 e A 1 t + A 1 ( B 2 2 + C 1 ) -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 t + x 2 -2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 2 1 , ˆ m ( t ) = 2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 2 1 e A 1 t + A 1 B 2 2 -2 A 2 ( A 2 + B 1 B 2 ) A 2 1 t + x 2 -2( A 2 + B 1 B 2 )( A 1 x + A 2 ) A 2 1 .

glyph[negationslash]

(e) If A 1 = 0, A 1 + B 2 1 = 0 and 2 A 1 + B 2 1 = 0, then we arrive at n ( t ) = ( x + A 2 A 1 ) e A 1 t -A 2 A 1 , and glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is easy to see that for all cases (a)-(e), the assertions in the Lemma follow and we conclude.

## References

- Yasin Abbasi-Yadkori and Csaba Szepesv´ ari. Regret bounds for the adaptive control of linear quadratic systems. In Conference on Learning Theory , pages 1-26, 2011.
- Marc Abeille and Alessandro Lazaric. Thompson sampling for linear-quadratic control problems. In International Conference on Artificial Intelligence and Statistics , 2017.
- Marc Abeille and Alessandro Lazaric. Improved regret bounds for Thompson sampling in linear quadratic control problems. In International Conference on Machine Learning , 2018.
- Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. Optimality and approximation with policy gradient methods in Markov decision processes. In Conference on Learning Theory , 2020.
- Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multi-armed bandit problem. Machine Learning , 47(2-3):235-256, 2002.
- Ronen I Brafman and Moshe Tennenholtz. R-max-A general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research , 3:213-231, 2002.
- Cyrus Derman. Finite State Markovian Decision Processes . Acedemic Press, New York, New York, 1970.
- Kenji Doya. Reinforcement learning in continuous time and space. Neural Computation , 12 (1):219-245, 2000.
- Nicole El Karoui, Nguyen Du Huu, and Monique Jeanblanc-Picqu´ e. Compactification methods in the control of degenerate diffusions: Existence of an optimal control. Stochastics , 20(3):169-219, 1987.
- Maryam Fazel, Rong Ge, Sham Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. In International Conference on Machine Learning , pages 1467-1476, 2018.

- Wendell H Fleming and Makiko Nisio. On stochastic relaxed control for partially observed diffusions. Nagoya Mathematical Journal , 93:71-108, 1984.
- Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. In Conference on Uncertainty in Artificial Intelligence , 2016.
- John Gittins. A dynamic allocation index for the sequential design of experiments. Progress in Statistics , pages 241-266, 1974.
- Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning , 2017.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning , pages 1856-1865, 2018.
- Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of Artificial Intelligence Research , 4:237-285, 1996.
- Ioannis Karatzas and Steven E Shreve. Brownian Motion and Stochastic Calculus . SpringerVerlag, New York, New York, 2nd edition, 1991.
- Haya Kaspi and Avishai Mandelbaum. Multi-armed bandits in discrete and continuous time. Annals of Applied Probability , pages 1270-1290, 1998.
- Thomas Kurtz and Richard Stockbridge. Existence of Markov controls and characterization of optimal Markov controls. SIAM Journal on Control and Optimization , 36(2):609-653, 1998.
- Thomas Kurtz and Richard Stockbridge. Stationary solutions and forward equations for controlled and singular martingale problems. Electronic Journal of Probability , 6, 2001.
- Bethany Leffler, Michael Littman, and Timothy Edmunds. Efficient reinforcement learning with relocatable action models. In AAAI Conference on Artificial Intelligence , 2007.
- Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research , 17(1):1334-1373, 2016.
- Weiwei Li and Emanuel Todorov. Iterative linearization methods for approximately optimal control and estimation of non-linear stochastic system. International Journal of Control , 80(9):1439-1453, 2007.
- Avi Mandelbaum. Continuous multi-armed bandits and multi-parameter processes. Annals of Probability , pages 1527-1556, 1987.
- Guillaume Matheron, Nicolas Perrin, and Olivier Sigaud. The problem with DDPG: understanding failures in deterministic environments with sparse rewards. arXiv preprint arXiv:1911.11679 , 2019.

- Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andrew J Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, and Koray Kavukcuoglu. Learning to navigate in complex environments. In International Conference on Learning Representations , 2017.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, and Georg Ostrovski. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.
- Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems , pages 2775-2785, 2017.
- Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Trust-PCL: An off-policy trust region method for continuous control. In International Conference on Learning Representations , 2018.
- Gergely Neu, Anders Jonsson, and Vicen¸ c G´ omez. A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.
- Ian Osband and Benjamin Van Roy. Model-based reinforcement learning and the eluder dimension. In Advances in Neural Information Processing Systems , pages 1466-1474, 2014.
- Ian Osband, Daniel Russo, Zheng Wen, and Benjamin Van Roy. Deep exploration via randomized value functions. Journal of Machine Learning Research , 20(124):1-62, 2017.
- Benjamin Recht. A tour of reinforcement learning: The view from continuous control. Annual Review of Control, Robotics, and Autonomous Systems , 2:253-279, 2019.
- Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In Advances in Neural Information Processing Systems , pages 2256-2264, 2013.
- Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research , 39(4):1221-1243, 2014.
- Claude Elwood Shannon. A mathematical theory of communication. ACM SIGMOBILE Mobile Computing and Communications Review , 5(1):3-55, 2001.
- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, and Marc Lanctot. Mastering the game of Go with deep neural networks and tree search. Nature , 529(7587):484, 2016.
- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, and Adrian Bolton. Mastering the game of Go without human knowledge. Nature , 550(7676):354, 2017.

- Alexander L Strehl and Michael L Littman. An analysis of model-based interval estimation for Markov decision processes. Journal of Computer and System Sciences , 74(8):13091331, 2008.
- Alexander L Strehl, Lihong Li, and Michael L Littman. Reinforcement learning in finite MDPs: PAC analysis. Journal of Machine Learning Research , 10(84):2413-2444, 2009.
- Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction . MIT press, 2018.
- William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3/4):285-294, 1933.
- Emanuel Todorov and Weiwei Li. A generalized iterative LQG method for locally-optimal feedback control of constrained nonlinear stochastic systems. In American Control Conference , pages 300-306. IEEE, 2005.
- Haoran Wang. Large scale continuous-time mean-variance portfolio allocation via reinforcement learning. arXiv preprint arXiv:1907.11718 , 2019.
- Haoran Wang and Xun Yu Zhou. Continuous-time mean-variance portfolio selection: A reinforcement learning framework. Mathematical Finance , 2020. doi: 10.1111/mafi.12281.
- Jiongmin Yong and Xun Yu Zhou. Stochastic controls: Hamiltonian systems and HJB equations , volume 43. Springer Science &amp; Business Media, 1999.
- Xun Yu Zhou. On the existence of optimal relaxed controls of stochastic partial differential equations. SIAM Journal on Control and Optimization , 30(2):247-261, 1992.
- Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In AAAI Conference on Artificial Intelligence , volume 8, pages 1433-1438. Chicago, IL, USA, 2008.