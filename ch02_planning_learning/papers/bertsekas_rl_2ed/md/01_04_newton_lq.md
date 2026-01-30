# 1.5: Newton's Method & LQ Problems

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 56-78
**Topics:** Newton's method, linear quadratic, LQ problems, region of stability, rollout, policy iteration, error bounds

---

Parametric approximation Neural nets Discretization

Multiagent Q-factor minimization

Approximate Min Approximate

Simple choices Parametric approximation Problem approximation

Certainty equivalence Monte Carlo tree search

Figure 1.3.2 Schematic illustration of approximation in value space for stochastic problems, and the three approximations involved in its design. Typically the approximations can be designed independently of each other, and with a variety of approaches. There are also multistep lookahead versions of approximation in value space, which will be discussed later.

<!-- image -->

any policy can be used on-line as base policy, including policies obtained by a sophisticated o ff -line procedure, using for example neural networks and training data. The rollout algorithm has a cost improvement property, whereby it yields an improved cost relative to its underlying base policy. We will discuss this property and some conditions under which it is guaranteed to hold in Chapter 2.

A major variant of rollout is truncated rollout , which combines the use of one-step optimization, simulation of the base policy for a certain number of steps m , and then adds an approximate cost ˜ J k + m +1 ( x k + m +1 ) to the cost of the simulation, which depends on the state x k + m +1 obtained at the end of the rollout. Note that if one foregoes the use of a base policy (i.e., m = 0), one recovers as a special case the general approximation in value space scheme (1.21); see Fig. 1.3.3. Thus rollout provides an extra layer of lookahead to the one-step minimization, but this lookahead need not extend to the end of the horizon.

Note also that versions of truncated rollout with multistep lookahead minimization are possible. They will be discussed later. The terminal cost approximation is necessary in infinite horizon problems, since an infinite number of stages of the base policy rollout is impossible. However, even for finite horizon problems it may be necessary and/or beneficial to artificially truncate the rollout horizon. Generally, a large combined number of multistep lookahead minimization and rollout steps is likely to be beneficial.

## Cost Versus Q-Factor Approximations - Robustness and OnLine Replanning

Similar to the deterministic case, Q-learning involves the calculation of either the optimal Q-factors (1.20) or approximations ˜ Q k ( x k ↪ u k ). The

Possible States

Multiagent Q-factor minimization for Stages Beyond Truncation

Figure 1.3.3 Schematic illustration of truncated rollout. One-step lookahead is followed by simulation of the base policy for m steps, and an approximate cost ˜ J k + m +1 ( x k + m +1 ) is added to the cost of the simulation, which depends on the state x k + m +1 obtained at the end of the rollout. If the base policy simulation is omitted (i.e., m = 0), one recovers the general approximation in value space scheme (1.21). Truncated rollout with multistep lookahead is also possible and is discussed in some detail in Chapter 2.

<!-- image -->

approximate Q-factors may be obtained using approximation in value space schemes, and can be used to obtain approximately optimal policies through the Q-factor minimization

<!-- formula-not-decoded -->

Since it is possible to implement approximation in value space by using cost function approximations [cf. Eq. (1.21)] or by using Q-factor approximations [cf. Eq. (1.22)], the question arises which one to use in a given practical situation. One important consideration is the facility of obtaining suitable cost or Q-factor approximations. This depends largely on the problem and also on the availability of data on which the approximations can be based. However, there are some other major considerations.

In particular, the cost function approximation scheme

<!-- formula-not-decoded -->

has an important disadvantage: the expected value above needs to be computed on-line for all u k ∈ U k ( x k ) , and this may involve substantial computation . It also has an important advantage in situations where the system function f k , the cost per stage g k , or the control constraint set U k ( x k ) can change as the system is operating. Assuming that the new f k , g k , or U k ( x k ) become known to the controller at time k , on-line replanning may be used, and this may improve substantially the robustness of the approximation in

value space scheme . By comparison, the Q-factor function approximation scheme (1.22) does not allow for on-line replanning. On the other hand, for problems where there is no need for on-line replanning, the Q-factor approximation scheme may not require the on-line computation of expected values and may allow a much faster on-line computation of the minimizing control ˜ θ k ( x k ) via Eq. (1.22).

One more disadvantage of using Q-factors will emerge later, as we discuss the synergy between o ff -line training and on-line play based on Newton's method; see Section 1.5. In particular, we will interpret the cost function of the lookahead minimization policy ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ as the result of one step of Newton's method for solving the Bellman equation that underlies the DP problem, starting from the terminal cost function approximations ¶ ˜ J 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J N ♦ . This synergy tends to be negatively a ff ected when Q-factor (rather than cost) approximations are used.

## 1.3.3 Approximation in Policy Space

The major alternative to approximation in value space is approximation in policy space , whereby we select the policy from a suitably restricted class of policies, usually a parametric class of some form. In particular, we can introduce a parametric family of policies (or approximation architecture, as we will call it in Chapter 3),

<!-- formula-not-decoded -->

where r k is a parameter, and then estimate the parameters r k using some type of training process or optimization; cf. Fig. 1.3.4.

In this section and throughout this book, we focus on selecting a policy o ff -line , possibly through training with o ff -line-collected data. There are algorithms that aim to improve parametric policies by using data that is collected on-line, but this subject is beyond our scope (see also the relevant discussion on policy gradient methods in Chapter 3).

Neural networks, described in Chapter 3, are often used to generate the parametric class of policies, in which case r k is the vector of weights/parameters of the neural network. In Chapter 3, we will also discuss methods for obtaining the training data required for obtaining the parameters r k , and we will consider several other classes of approximation architectures.

A general scheme for parametric approximation in policy space is to somehow obtain a training set, consisting of a large number of sample state-control pairs

<!-- formula-not-decoded -->

such that for each s , u s k is a 'good' control at state x s k . We can then choose the parameter r k by solving the least squares/regression problem

<!-- formula-not-decoded -->

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

(

) Approximate Q-Factor

Figure 1.3.4 Schematic illustration of parametric approximation in policy space.

<!-- image -->

A policy

<!-- formula-not-decoded -->

from a parametric class is computed o ff -line based on data, and it is used to generate the control u k = ˜ θ k ( x k ↪ r k ) on-line, when at state x k .

(possibly modified to add regularization). In particular, we may determine u s k using a human or a software 'expert' that can choose 'near-optimal' controls at given states, so ˜ θ k is trained to match the behavior of the expert. Methods of this type are commonly referred to as supervised learning in artificial intelligence.

An important approach for generating the training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , for the least squares training problem (1.24) is based on approximation in value space. In particular, we may use a one-step lookahead minimization of the form

<!-- formula-not-decoded -->

Here ‖ · ‖ denotes the standard quadratic Euclidean norm. It is implicitly assumed here (and in similar situations later) that the controls are members of a Euclidean space (i.e., the space of finite dimensional vectors with real-valued components) so that the distance between two controls can be measured by their normed di ff erence (randomized controls, i.e., probabilities that a particular action will be used, fall in this category). Regression problems of this type arise in the training of parametric classifiers based on data, including the use of neural networks (see Section 3.4). Assuming a finite control space, the classifier is trained using the data ( x s k ↪ u s k ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ which are viewed as state-category pairs, and then a state x k is classified as being of 'category' ˜ θ k ( x k ↪ r k ). Parametric approximation architectures, and their training through the use of classification and regression techniques are described in Chapter 3. An important modification is to use regularized regression where a quadratic regularization term is added to the least squares objective. This term is a positive multiple of the squared deviation ‖ r -ˆ r ‖ 2 of r from some initial guess ˆ r .

where ˜ J k +1 is a suitable (separately obtained) approximation in value space. Alternatively, we may use an approximate Q-factor based minimization

<!-- formula-not-decoded -->

where ˜ Q k is a (separately obtained) Q-factor approximation. We may view this as approximation in policy space built on top of approximation in value space .

There is a significant advantage of the least squares training procedure of Eq. (1.24), and more generally approximation in policy space: once the parametrized policy ˜ θ k is obtained, the computation of controls

<!-- formula-not-decoded -->

during on-line operation of the system is often much easier compared with the lookahead minimization (1.23). For this reason, one of the major uses of approximation in policy space is to provide an approximate implementation of a known policy (no matter how obtained) for the purpose of convenient on-line use. On the negative side, such an implementation is less well suited for on-line replanning.

## Model-Free Approximation in Policy Space

There are also alternative optimization-based approaches for policy space approximation. The main idea is that once we use a vector ( r 0 ↪ r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r N -1 ) to parametrize the policies π , the expected cost J π ( x 0 ) is parametrized as well, and can be viewed as a function of ( r 0 ↪ r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r N -1 ). We can then optimize this cost by using a gradient-like or random search method. This is a widely used approach for optimization in policy space, which will be discussed somewhat briefly in this book (see Section 3.5, and the RL book [Ber19a], Section 5.7).

An interesting feature of this approach is that in principle it does not require a mathematical model of the system and the cost function; a computer simulator (or availability of the real system for experimentation) su ffi ces instead. This is sometimes called a model-free implementation . The advisability of implementations of this type, particularly when they rely exclusively on simulation (i.e., without the use of any prior mathematical model knowledge), is a hotly debated and much contested issue; see for example the review paper by Alamir [Ala22].

The term 'model-free' can be confusing. In reality, there is always a model in DP/RL problem formulations . It is just a question of whether it is a mathematical model (i.e., based on equations), or a computer model (i.e., based on computer simulation or a trained neural network), or a hybrid model, (i.e., one that relies both on mathematical equations and computer software).

Target Cost Function

Figure 1.3.5 The general structure for parametric cost approximation. We approximate the target cost function J ( x ) with a member from a parametric class ˜ J ( x↪ r ) that depend on a parameter vector r . We use training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and a form of optimization that aims to find a parameter ˆ r that 'minimizes' the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q .

<!-- image -->

We finally note an important conceptual di ff erence between approximation in value space and approximation in policy space. The former is primarily an on-line method (with o ff -line training used optionally to construct cost function approximations for one-step or multistep lookahead). The latter is primarily an o ff -line training method (which may be used without modification for on-line play or optionally to provide a policy for on-line rollout).

## 1.3.4 Training of Cost Function and Policy Approximations

When it comes to o ff -line constructed approximations, a major approach is based on the use of parametric approximation. Feature-based architectures and neural networks are very useful within our RL context, and will be discussed in Chapter 3, together with methods that can be used for training them.

A general structure for parametric cost function approximation is illustrated in Fig. 1.3.5. We have a target function J ( x ) that we want to

The principal role of neural networks within the context of this book is to provide the means for approximating various target functions from input-output data. This includes cost functions and Q-factors of given policies, and optimal cost-to-go functions and Q-factors; in this case the neural network is referred to as a value network (sometimes the alternative term critic network is also used). In other cases the neural network represents a policy viewed as a function from state to control, in which case it is called a policy network (the alternative term actor network is also used). The training methods for constructing the cost function, Q-factor, and policy approximations from data are mostly based on optimization and regression, and will be reviewed in Chapter 3. Further DPoriented discussions are found in many sources, including the RL books [Ber19a], [Ber20a], and the neuro-dynamic programming book [BeT96]. Machine learning books, including those describing at length neural network architectures and training are also recommended; see e.g., the recent book by Bishop and Bishop [BiB24], and the references quoted therein.

Approximating Function

approximate with a member of a parametric class of functions ˜ J ( x↪ r ) that depend on a parameter vector r (to simplify, we drop the time index, using J in place of J k ). To this end, we collect training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , which we use to determine a parameter ˆ r that leads to a good 'fit' between the data J ( x s ) and the predictions ˜ J ( x s ↪ ˆ r ) of the parametrized function. This is usually done through some form of optimization that aims to minimize in some sense the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ).

The methodological ideas for parametric cost approximation can also be used for approximation of a target policy θ with a policy from a parametric class ˜ θ ( x↪ r ). The training data may be obtained, for example, from rollout control calculations, thus enabling the construction of both value and policy networks that can be combined for use in a perpetual rollout scheme. However, there is an important di ff erence: the approximate cost values ˜ J ( x↪ r ) are real numbers, whereas the approximate policy values ˜ θ ( x↪ r ) are elements of a control space U . Thus if U consists of m dimensional vectors, ˜ θ ( x↪ r ) consists of m numerical components. In this case the parametric approximation problems for cost functions and for policies are fairly similar, and both involve continuous space approximations.

On the other hand, the case where the control space is finite, U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ . is markedly di ff erent. In this case, for any x , ˜ θ ( x↪ r ) consists of one of the m possible controls u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m . This ushers a connection with traditional classification schemes, whereby objects x are classified as belonging to one of the categories u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m , so that θ ( x ) defines the category of x , and can be viewed as a classifier . Some of the most prominent classification schemes actually produce randomized outcomes, i.e., x is associated with a probability distribution

<!-- formula-not-decoded -->

which is a randomized policy in our policy approximation context; see Fig. 1.3.6. This is done usually for reasons of algorithmic convenience, since many optimization methods, including least squares regression, require that the optimization variables are continuous. In this case, the randomized policy (1.25) can be converted to a nonrandomized policy using a maximization operation: associate x with the control of maximum probability (cf. Fig. 1.3.6),

<!-- formula-not-decoded -->

The use of classification methods for approximation in policy space will be discussed in Chapter 3 (Section 3.4).

## 1.4 INFINITE HORIZON PROBLEMS - AN OVERVIEW

We will now provide an outline of infinite horizon stochastic DP with an emphasis on its aspects that relate to our RL/approximation methods. We

Figure 1.3.6 A general structure for parametric policy approximation for the case where the control space is finite, U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ , and its relation to a classification scheme. It produces a randomized policy of the form (1.25), which is converted to a nonrandomized policy through the maximization operation (1.26).

<!-- image -->

will deal primarily with infinite horizon stochastic problems, where we aim to minimize the total cost over an infinite number of stages, given by

<!-- formula-not-decoded -->

see Fig. 1.4.1. Here, J π ( x 0 ) denotes the cost associated with an initial state x 0 and a policy π = ¶ θ 0 ↪ θ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , and α is a scalar in the interval (0 ↪ 1]. The functions g and f that define the cost per stage and the system equation

<!-- formula-not-decoded -->

do not change from one stage to the next. The stochastic disturbances, w 0 ↪ w 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] , have a common probability distribution P ( · ♣ x k ↪ u k ).

When α is strictly less that 1, it has the meaning of a discount factor , and its e ff ect is that future costs matter to us less than the same costs incurred at the present time. Among others, a discount factor guarantees that the limit defining J π ( x 0 ) exists and is finite (assuming that the range of values of the stage cost g is bounded). This is a nice mathematical property that makes discounted problems analytically and algorithmically tractable.

Thus, by definition, the infinite horizon cost of a policy is the limit of its finite horizon costs as the horizon tends to infinity. The three types of problems that we will focus on are:

- (a) Stochastic shortest path problems (SSP for short). Here, α = 1 but there is a special cost-free termination state; once the system reaches that state it remains there at no further cost. In some types of problems, the termination state may represent a goal state that we are trying to reach at minimum cost, while in others it may be a state that we are trying to avoid for as long as possible. We will mostly assume a problem structure such that termination is inevitable under all policies. Thus the horizon is in e ff ect finite, but its length is random and may be a ff ected by the policy being used. A significantly

Figure 1.4.1 Illustration of an infinite horizon problem. The system and cost per stage are stationary, except for the use of a discount factor α . If α = 1, there is typically a special cost-free termination state that we aim to reach.

<!-- image -->

more complicated type of SSP problems, which we will discuss selectively, arises when termination can be guaranteed only for a subset of policies, which includes all optimal policies. Some common types of SSP belong to this category, including deterministic shortest path problems that involve graphs with cycles.

- (b) Discounted problems . Here, α &lt; 1 and there need not be a termination state. However, we will see that a discounted problem with a finite number of states can be readily converted to an SSP problem. This can be done by introducing an artificial termination state to which the system moves with probability 1 -α at every state and stage, thus making termination inevitable. As a result, algorithms and analysis for SSP problems can be easily adapted to discounted problems; the DP textbook [Ber17a] provides a detailed account of this conversion, and an accessible introduction to discounted and SSP problems with a finite number of states.
- (c) Deterministic nonnegative cost problems . Here, the disturbance w k takes a single known value. Equivalently, there is no disturbance in the system equation and the cost expression, which now take the form

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We assume further that there is a cost-free and absorbing termination state t , and that we have

<!-- formula-not-decoded -->

/negationslash and g ( t↪ u ) = 0 for all u ∈ U ( t ). This type of structure expresses the objective to reach or approach t at minimum cost, a classical control problem. An extensive analysis of the undiscounted version of this problem was given in the author's paper [Ber17b].

Discounted stochastic problems with a finite number of states [also referred to as discounted MDP (abbreviation for Markovian Decision Problem) ] are very common in the DP/RL literature, particularly because of

their benign analytical and computational nature. Moreover, there is a widespread belief that discounted MDP can be used as a universal model, i.e., that in practice any other kind of problem (e.g., undiscounted problems with a termination state and/or a continuous state space) can be painlessly converted to a discounted MDP with a discount factor that is close enough to 1. This is questionable, however, for a number of reasons:

- (a) Deterministic models are common as well as natural in many practical contexts (including discrete optimization/integer programming problems), so to convert them to MDP does not make sense.
- (b) The conversion of a continuous-state problem to a finite-state problem through some kind of discretization involves mathematical subtleties that can lead to serious practical/algorithmic complications. In particular, the character of the optimal solution may be seriously distorted by converting to a discounted MDP through some form of discretization, regardless of how fine the discretization is.
- (c) For some practical shortest path contexts it is essential that the termination state is ultimately reached. However, when a discount factor α is introduced in such a problem, the character of the problem may be fundamentally altered. In particular, the threshold for an appropriate value of α may be very close to 1 and may be unknown in practice. For a simple example consider a shortest path problem with states 1 and 2 plus a termination state t . From state 1 we can go to state 2 at cost 0, from state 2 we can go to either state 1 at a small cost /epsilon1 &gt; 0 or to the termination state at a substantial cost C &gt; 0. The optimal policy over an infinite horizon is to go from 1 to 2 and from 2 to t . Suppose now that we approximate the problem by introducing a discount factor α ∈ (0 ↪ 1). Then it can be shown that if α &lt; 1 -/epsilon1 glyph[triangleleft]C , it is optimal to move indefinitely around the cycle 1 → 2 → 1 → 2 and never reach t , while for α &gt; 1 -/epsilon1 glyph[triangleleft]C the shortest path 2 → 1 → t will be obtained. Thus the solution of the discounted problem varies discontinuously with α : it changes radically at some threshold, which in general may be unknown.

An important class of problems that we will consider in some detail in this book is finite-state deterministic problems with a large number of states. Finite horizon versions of these problems include challenging discrete optimization problems, whose exact solution is practically impossible. An important fact to keep in mind is that we can transform such problems to infinite horizon SSP problems with a termination state at the end of the horizon, so that the conceptual framework of the present section applies. The approximate solution of discrete optimization problems by RL methods, and particularly by rollout, will be considered in Chapter 2, and has been discussed at length in the books [Ber19a] and [Ber20a].

## 1.4.1 Infinite Horizon Methodology

There are several analytical and computational issues regarding our infinite horizon problems. Many of them revolve around the relation between the optimal cost function J * of the infinite horizon problem and the optimal cost functions of the corresponding N -stage problems.

In particular, let J N ( x ) denote the optimal cost of the problem involving N stages, initial state x , cost per stage g ( x↪ u↪ w ), and zero terminal cost. This cost is generated after N iterations of the algorithm

<!-- formula-not-decoded -->

starting from J 0 ( x ) ≡ 0. The algorithm (1.31) is known as the value iteration algorithm (VI for short). Since the infinite horizon cost of a given policy is, by definition, the limit of the corresponding N -stage costs as N →∞ , it is natural to speculate that:

- (a) The optimal infinite horizon cost is the limit of the corresponding N -stage optimal costs as N →∞ ; i.e.,

<!-- formula-not-decoded -->

for all states x .

- (b) The following equation should hold for all states x ,

<!-- formula-not-decoded -->

This is obtained by taking the limit as N →∞ in the VI algorithm (1.31) using Eq. (1.32). The preceding equation, called Bellman's equation , is really a system of equations (one equation per state x ), which has as solution the optimal costs-to-go of all the states.

- (c) If θ ( x ) attains the minimum in the right-hand side of the Bellman equation (1.33) for each x , then the policy ¶ θ↪ θ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ should be optimal. This type of policy is called stationary , and for simplicity it is denoted by θ .

This is just the finite horizon DP algorithm of Section 1.3.1, except that we have reversed the time indexing to suit our infinite horizon context. In particular, consider the N -stages problem and let V N -k ( x ) be the optimal cost-to-go starting at x with k stages to go, and with terminal cost equal to 0. Applying DP, we have for all x ,

<!-- formula-not-decoded -->

By defining J k ( x ) = V N -k ( x ) glyph[triangleleft] α N -k , we obtain the VI algorithm (1.31).

- (d) The cost function J θ of a stationary policy θ satisfies

<!-- formula-not-decoded -->

We can view this as just the Bellman equation (1.33) for a di ff erent problem, where for each x , the control constraint set U ( x ) consists of just one control, namely θ ( x ). Moreover, we expect that J θ is obtained in the limit by the VI algorithm:

<!-- formula-not-decoded -->

where J θ↪N is the N -stage cost function of θ generated by

<!-- formula-not-decoded -->

starting from J θ↪ 0 ( x ) ≡ 0 or some other initial condition; cf. Eqs. (1.31)-(1.32).

All four of the preceding results can be shown to hold for finitestate discounted problems, and also for finite-state SSP problems under reasonable assumptions. The results also hold for infinite-state discounted problems, provided the cost per stage function g is bounded over the set of possible values of ( x↪ u↪ w ), in which case we additionally can show that J * is the unique solution of Bellman's equation. The VI algorithm is also valid under these conditions, in the sense that J k → J * , even if the initial function J 0 is nonzero. The motivation for a di ff erent choice of J 0 is faster convergence to J * ; generally the convergence is faster as J 0 is chosen closer to J * . The associated mathematical proofs can be found in several sources, e.g., [Ber12], Chapter 1, or [Ber19a], Chapter 4.

It is important to note that for infinite horizon problems, there are additional important algorithms that are amenable to approximation in value space. Approximate policy iteration, Q-learning, temporal di ff erence methods, linear programming, and their variants are some of these; see the RL books [Ber19a], [Ber20a]. For this reason, in the infinite horizon case, there is a richer set of algorithmic options for approximation in value space, despite the fact that the associated mathematical theory is more complex. In this book, we will only discuss approximate forms and variations of the policy iteration algorithm, which we describe next.

For undiscounted problems and discounted problems with unbounded cost per stage, we may still adopt the four preceding results as a working hypothesis. However, we should also be aware that exceptional behavior is possible under unfavorable circumstances, including nonuniqueness of solution of Bellman's equation, and nonconvergence of the VI algorithm to J ∗ from some initial conditions; see the books [Ber12], [Ber22b].

Policy Evaluation Policy Improvement Rollout Policy ˜

<!-- image -->

Policy Evaluation Policy Improvement Rollout Policy ˜

Figure 1.4.2 Schematic illustration of PI as repeated rollout. It generates a sequence of policies, with each policy θ in the sequence being the base policy that generates the next policy ˜ θ in the sequence as the corresponding rollout policy. This rollout policy is used as the base policy in the subsequent iteration.

## Policy Iteration

A major infinite horizon algorithm is policy iteration (PI for short). We will argue that PI, together with its variations, forms the foundation for self-learning in RL, i.e., learning from data that is self-generated (from the system itself as it operates) rather than from data supplied from an external source. Figure 1.4.2 describes the method as repeated rollout, and indicates that each of its iterations consists of two phases:

- (a) Policy evaluation , which computes the cost function J θ of the current (or base) policy θ . One possibility is to solve the corresponding Bellman equation

<!-- formula-not-decoded -->

cf. Eq. (1.34). However, the value J θ ( x ) for any x can also be computed by Monte Carlo simulation, by averaging over many randomly generated trajectories the cost of the policy starting from x .

- (b) Policy improvement , which computes the 'improved' (or rollout) policy ˜ θ using the one-step lookahead minimization

<!-- formula-not-decoded -->

We call ˜ θ 'improved policy' because we can generally prove that

<!-- formula-not-decoded -->

This cost improvement property will be shown in Chapter 2, Section 2.7, and can be used to show that PI produces an optimal policy in a finite number of iterations under favorable conditions (for example for finitestate discounted problems; see the DP books [Ber12], [Ber17a], or the RL book [Ber19a]).

The rollout algorithm in its pure form is just a single iteration of the PI algorithm . It starts from a given base policy θ and produces the rollout policy ˜ θ . It may be viewed as approximation in value space with one-step lookahead that uses J θ as terminal cost function approximation. It has the advantage that it can be applied on-line by computing the needed values of J θ ( x ) by simulation. By contrast, approximate forms of PI for challenging problems, involving for example neural network training, can only be implemented o ff -line.

## 1.4.2 Approximation in Value Space - Infinite Horizon

The approximation in value space approach that we discussed in connection with finite horizon problems can be extended in a natural way to infinite horizon problems. Here in place of J * , we use an approximation ˜ J , and generate at any state x , a control ˜ θ ( x ) by the one-step lookahead minimization

<!-- formula-not-decoded -->

This minimization yields a stationary policy ¶ ˜ θ↪ ˜ θ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ♦ , with cost function denoted J ˜ θ [i.e., J ˜ θ ( x ) is the total infinite horizon discounted cost obtained when using ˜ θ starting at state x ]; see Fig. 1.4.3. Note that when ˜ J = J * , the one-step lookahead policy attains the minimum in the Bellman equation (1.33) and is expected to be optimal. This suggests that one should try to use ˜ J as close as possible to J * , which is generally true as we will argue later.

Naturally an important goal to strive for is that J ˜ θ is close to J * in some sense. However, for classical control problems, which involve steering and maintaining the state near a desired reference state (e.g., problems with a cost-free and absorbing terminal state, and positive cost for all other states), stability of ˜ θ may be a principal objective . In this book, we will discuss stability issues primarily for this one class of problems, and we will consider the policy ˜ θ to be stable if J ˜ θ is real-valued , i.e.,

<!-- formula-not-decoded -->

Selecting ˜ J so that ˜ θ is stable is a question of major interest for some application contexts, such as model predictive and adaptive control, and will be discussed in the next section within the limited context of linear quadratic problems.

## /lscript -Step Lookahead

An important extension of one-step lookahead minimization is /lscript -step lookahead , whereby at a state x k we minimize the cost of the first /lscript &gt; 1 stages

¿ An lins olor citl multiaton 1od

At x

First Step

"Future"

minuEU(x) E{g(x, u, w) + aJ (f(x, 2, w)) }

One-Step Lookahead

First l Steps

Multistep Lookahead

"Future"

Figure 1.4.3 Schematic illustration of approximation in value space with one-step and /lscript -step lookahead minimization for infinite horizon problems. In the former case, the minimization yields at state x a control ˜ u , which defines the one-step lookahead policy ˜ θ via

<!-- image -->

<!-- formula-not-decoded -->

In the latter case, the minimization yields a control ˜ u k policies ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 . The control ˜ u k is applied at x k while the remaining sequence ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 is discarded. The control ˜ u k defines the /lscript -step lookahead policy ˜ θ .

with the future costs approximated by a function ˜ J (see the bottom half of Fig. 1.4.3). This minimization yields a control ˜ u k and a sequence ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 . The control ˜ u k is applied at x k , and defines the /lscript -step lookahead policy ˜ θ via ˜ θ ( x k ) = ˜ u k , while ˜ θ k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ k + /lscript -1 are discarded. Actually, we may view /lscript -step lookahead minimization as the special case of its one-step counterpart where the lookahead function is the optimal cost function of an ( /lscript -1)-stage DP problem with a terminal cost ˜ J ( x k + /lscript ) on the state x k + /lscript obtained after /lscript -1 stages.

The motivation for /lscript -step lookahead minimization is that by increasing the value of /lscript , we may require a less accurate approximation ˜ J to obtain good performance . Otherwise expressed, for the same quality of cost function approximation, better performance may be obtained as /lscript becomes larger. This will be explained visually later, using the formalism of Newton's method in Section 1.5. In particular, for AlphaZero chess, long multistep lookahead is critical for good on-line performance. Another motivation for multistep lookahead is to enhance the stability properties of the gener-

On-line play with multistep lookahead minimization (and possibly truncated rollout) is referred to by a number of di ff erent names in the RL literature, such as on-line search , predictive learning , learning from prediction , etc; in the model predictive control literature the combined interval of lookahead minimization and truncated rollout is referred as the prediction interval .

At Xk min

Uk, Hk+1,., Hk+e-1

Min Approximation

Figure 1.4.4 Approximation in value space with one-step lookahead for infinite horizon problems. There are three potential areas of approximation, which can be considered independently of each other: optimal cost approximation, expected value approximation, and minimization approximation.

<!-- image -->

ated on-line policy , as we will discuss later in Section 1.5. On the other hand, solving the multistep lookahead minimization problem, instead of the one-step lookahead counterpart of Eq. (1.36), is more time consuming.

## The Three Approximations: Optimal Cost, Expected Value, and Lookahead Minimization Approximations

There are three potential areas of approximation for infinite horizon problems: optimal cost approximation, expected value approximation, and minimization approximation; cf. Fig. 1.4.4. They are similar to their finite horizon counterparts that we discussed in Section 1.3.2. In particular, we have potentially:

- (a) A terminal cost approximation ˜ J of the optimal cost function J * : A major advantage of the infinite horizon context is that only one approximate cost function ˜ J is needed, rather than the N functions ˜ J 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ J N of the N -step horizon case.
- (b) An approximation of the expected value operation : This operation can be very time consuming. It may be simplified in various ways. For example some of the random quantities w k ↪ w k +1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w k + /lscript -1 appearing in the /lscript -step lookahead minimization may be replaced by deterministic quantities; this is another example of the certainty equivalence approach , which we discussed in Section 1.3.2.
- (c) A simplification of the minimization operation : For example in multiagent problems the control consists of multiple components,

<!-- formula-not-decoded -->

with each component u i chosen by a di ff erent agent/decision maker. In this case the size of the control space can be enormous, but it

can be simplified in ways that will be discussed later (e.g., choosing components sequentially, one-agent-at-a-time). This will form the core of our approach to multiagent problems; see Section 1.6.7 and Chapter 2, Section 2.9.

We will next describe briefly various approaches for selecting the terminal cost function approximation.

## Constructing Terminal Cost Approximations for On-Line Play

A major issue in value space approximation is the construction of a suitable approximate cost function ˜ J . This can be done in many di ff erent ways, giving rise to some of the principal RL methods.

For example, ˜ J may be constructed with sophisticated o ff -line training methods. Alternatively, the approximate values ˜ J ( x ) may be obtained online as needed with truncated rollout, by running an o ff -line obtained policy for a suitably large number of steps, starting from x , and supplementing it with a suitable, perhaps primitive, terminal cost approximation.

For orientation purposes, let us describe briefly four broad types of approximation. We will return to these approaches later, and we also refer to the RL and approximate DP literature for more detailed discussions.

- (a) O ff -line problem approximation : Here the function ˜ J is computed o ff -line as the optimal or nearly optimal cost function of a simplified optimization problem, which is more convenient for computation. Simplifications may include exploiting decomposable structure, reducing the size of the state space, neglecting some of the constraints, and ignoring various types of uncertainties. For example we may consider using as ˜ J the cost function of a related deterministic problem, obtained through some form of certainty equivalence approximation, thus allowing computation of ˜ J by gradient-based optimal control methods or shortest path-type methods.

A major type of problem approximation method is aggregation , described in Section 3.6, and in the books [Ber12], [Ber19a] and papers [Ber18a], [Ber18b]. Aggregation provides a systematic procedure to simplify a given problem. A principal example is to group states together into a relatively small number of subsets, called aggregate states. The optimal cost function of the simpler aggregate problem is computed by exact DP methods, possibly involving the use of simulation. This cost function is then used to provide an approximation ˜ J to the optimal cost function J * of the original problem, using some form of interpolation.

- (b) On-line simulation : This possibility arises in rollout algorithms for stochastic problems, where we use Monte-Carlo simulation and some suboptimal policy θ (the base policy) to compute (whenever needed) values ˜ J ( x ) that are exactly or approximately equal to J θ ( x ). The

policy θ may be obtained by any method, e.g., one based on heuristic reasoning (such as in the case of the traveling salesman Example 1.2.3), or o ff -line training based on a more principled approach, such as approximate policy iteration or approximation in policy space. Note that while simulation is time-consuming, it is uniquely wellsuited for the use of parallel computation. Moreover, it can be simplified through the use of certainty equivalence approximations.

- (c) On-line approximate optimization . This approach involves the solution of a suitably constructed shorter horizon version of the problem, with a simple terminal cost approximation. It can be viewed as either approximation in value space with multistep lookahead, or as a form of rollout algorithm. It is often used in model predictive control (MPC).
- (d) Parametric cost approximation , where ˜ J is obtained from a given parametric class of functions J ( x↪ r ), where r is a parameter vector, selected by a suitable algorithm. The parametric class typically involves prominent characteristics of x called features , which can be obtained either through insight into the problem at hand, or by using training data and some form of neural network (see Chapter 3).

Such methods include approximate forms of PI, as discussed in Section 1.1 in connection with chess and backgammon. The policy evaluation portion of the PI algorithm can be done by approximating the cost function of the current policy using an approximation architecture such as a neural network (see Chapter 3). It can also be done with stochastic iterative algorithms such as TD( λ ), LSPE( λ ), and LSTD( λ ), which are described in the DP book [Ber12] and the RL book [Ber19a]. These methods are somewhat peripheral to our course, and will not be discussed at any length. We note, however, that approximate PI methods do not just yield a parametric approximate cost function J ( x↪ r ), but also a suboptimal policy, which can be improved on-line by using (possibly truncated) rollout.

Aside from approximate PI, parametric approximate cost functions J ( x↪ r ) may be obtained o ff -line with methods such as Q-learning, linear programming, and aggregation methods, which are also discussed in the books [Ber12] and [Ber19a].

Let us also mention that for problems with special structure, ˜ J may be chosen so that the one-step lookahead minimization (1.36) is facilitated. In fact, under favorable circumstances, the lookahead minimization may be carried out in closed form. An example is when the system is nonlinear, but the control enters linearly in the system equation and quadratically in the cost function, while the terminal cost approximation is quadratic. Then the one-step lookahead minimization can be carried out analytically, because it involves a function that is quadratic in u .

## From O ff -Line Training to On-Line Play

Generally o ff -line training will produce either just a cost approximation (as in the case of TD-Gammon), or just a policy (as for example by some approximation in policy space/policy gradient approach), or both (as in the case of AlphaZero). We have already discussed in this section one-step lookahead and multistep lookahead schemes to implement on-line approximation in value space using ˜ J ; cf. Fig. 1.4.3. Let us now consider some additional possibilities, which involve the use of a policy θ that has been obtained o ff -line (possibly in addition to a terminal cost approximation). Here are some of the main possibilities:

- (a) Given a policy θ that has been obtained o ff -line, we may use as terminal cost approximation ˜ J the cost function J θ of the policy. For the case of one-step lookahead, this requires a policy evaluation operation, and can be done on-line, by computing (possibly by simulation) just the values of

<!-- formula-not-decoded -->

that are needed [cf. Eq. (1.36)]. For the case of /lscript -step lookahead, the values

<!-- formula-not-decoded -->

for all states x k + /lscript that are reachable in /lscript steps starting from x k are needed. This is the simplest form of rollout, and only requires the o ff -line construction of the policy θ .

- (b) Given a terminal cost approximation ˜ J that has been obtained o ff -line, we may use it on-line to compute fast when needed the controls of a corresponding one-step or multistep lookahead policy ˜ θ . The policy ˜ θ can in turn be used for rollout as in (a) above. In a truncated variation of this scheme, we may also use ˜ J to approximate the tail end of the rollout process (an example of this is the rollout-based TD-Gammon algorithm).
- (c) Given a policy θ and a terminal cost approximation ˜ J , we may use them together in a truncated rollout scheme, whereby the tail end of the rollout with θ is approximated using the cost approximation ˜ J . This is similar to the truncated rollout scheme noted in (b) above, except that the policy θ is computed o ff -line rather than on-line using ˜ J and one-step or multistep lookahead.

The preceding three possibilities are the principal ones for using the results of o ff -line training within on-line play schemes. Naturally, there are variations where additional information is computed o ff -line to facilitate and/or expedite the on-line play algorithm. As an example, in MPC, in addition to a terminal cost approximation, a target tube may need to be computed o ff -line in order to guarantee that some state constraints can

be satisfied on-line; see the discussion of MPC in Section 1.6.9. Other examples of this type will be noted in the context of specific applications.

Finally, let us note that while we have emphasized approximation in value space with cost function approximation, our discussion applies to Q-factor approximation, involving functions

<!-- formula-not-decoded -->

The corresponding one-step lookahead scheme has the form

<!-- formula-not-decoded -->

cf. Eq. (1.36). The second term on the right in the above equation represents the cost function approximation

<!-- formula-not-decoded -->

The use of Q-factors is common in the 'model-free' case where a computer simulator is used to generate samples of w , and corresponding values of g and f . Then, having obtained ˜ Q through o ff -line training, the one-step lookahead minimization in Eq. (1.37) must be performed on-line with the use of the simulator.

## 1.4.3 Understanding Approximation in Value Space

We will now discuss some of our aims as we try to get insight into the process of approximation in value space. Clearly, it makes sense to approximate J * with a function ˜ J that is as close as possible to J * . However, we should also try to understand quantitatively the relation between ˜ J and J ˜ θ , the cost function of the resulting one-step lookahead (or multistep lookahead) policy ˜ θ . Interesting questions in this regard are the following:

- (a) How is the quality of the lookahead policy ˜ θ a ff ected by the quality of the o ff -line training? A related question is how much should we care about improving ˜ J through a longer and more sophisticated training process, for a given approximation architecture? A fundamental fact that provides a lot of insight in this respect is that J ˜ θ is the result of a step of Newton's method that starts at ˜ J and is applied to the Bellman Eq. (1.33) . This will be the focus of our discussion in the next section, and has been a major point in the narrative of the author's books, [Ber20a] and [Ber22a].

A related fact is that in approximation in value space with multistep lookahead, J ˜ θ is the result of a step of Newton's method that starts at the function obtained by applying multiple value iterations to ˜ J .

- (b) How do simplifications in the multistep lookahead implementation affect J ˜ θ ? The Newton step interpretation of approximation in value space leads to an important insight into the special character of the initial step of the multistep lookahead. In particular, it is only the first step that acts as the Newton step, and needs to be implemented with precision . The subsequent steps are value iterations, which only serve to enhance the quality of the starting point of the Newton step, and hence their precise implementation is not critical .

This idea suggests that simplifications of the lookahead steps after the first can be implemented with relatively small (if any) performance loss for the multistep lookahead policy. Important examples of such simplifications are the use of certainty equivalence (Sections 1.6.9, 2.7.2, 2.8.3), and forms of pruning of the lookahead tree (Section 2.4). In practical terms, simplifications after the first step of the multistep lookahead can save a lot of on-line computation, which can be fruitfully invested in extending the length of the lookahead.

- (c) When is ˜ θ stable? The question of stability is very important in many control applications where the objective is to keep the state near some reference point or trajectory. Indeed, in such applications, stability is the dominant concern, and optimality is secondary by comparison. Among others, here we are interested to characterize the set of terminal cost approximations ˜ J that lead to a stable ˜ θ .
- (d) How does the length of lookahead minimization or the length of the truncated rollout a ff ect the stability and quality of the multistep lookahead policy ˜ θ ? While it is generally true that the length of lookahead has a beneficial e ff ect on quality, it turns out that it also has a beneficial e ff ect on the stability properties of the multistep lookahead policy, and we are interested in the mechanism by which this occurs.

In what follows we will be keeping in mind these questions. In particular, in the next section, we will discuss them in the context of the simple and convenient linear quadratic problem. Our conclusions, however, hold within a far more general context with the aid of the abstract DP formalism; see the author's books [Ber20a] and [Ber22a] for a broader presentation and analysis, which address these questions in greater detail and generality.

## 1.5 NEWTON'SMETHOD-LINEARQUADRATICPROBLEMS

We will now aim to understand the character of the Bellman equation, approximation in value space, and the VI and PI algorithms within the context of an important deterministic problem. This is the classical continuous-spaces problem where the system is linear, with no control constraints, and the cost function is nonnegative quadratic. While this prob-

lem can be solved analytically, it provides a uniquely insightful context for understanding visually the Bellman equation and its algorithmic solution, both exactly and approximately.

In its general form, the problem deals with the system

<!-- formula-not-decoded -->

where x k and u k are elements of the Euclidean spaces /Rfractur n and /Rfractur m , respectively, A is an n × n matrix, and B is an n × m matrix. It is assumed that there are no control constraints. The cost per stage is quadratic of the form

<!-- formula-not-decoded -->

where Q and R are positive definite symmetric matrices of dimensions n × n and m × m , respectively (all finite-dimensional vectors in this work are viewed as column vectors, and a prime denotes transposition). The analysis of this problem is well known and is given with proofs in several control theory texts, including the author's DP books [Ber17a] and [Ber12].

In what follows, we will focus for simplicity only on the one-dimensional version of the problem, where the system has the form

<!-- formula-not-decoded -->

/negationslash cf. Example 1.3.1. Here the state x k and the control u k are scalars, and the coe ffi cients a and b are also scalars, with b = 0. The cost function is undiscounted and has the form

<!-- formula-not-decoded -->

where q and r are positive scalars. The one-dimensional case allows a convenient and insightful analysis of the algorithmic issues that are central for our purposes. This analysis generalizes to multidimensional linear quadratic problems and beyond, but requires a more demanding mathematical treatment.

## The Riccati Equation and its Justification

The analytical results for our problem may be obtained by taking the limit in the results derived in the finite horizon Example 1.3.1, as the horizon length tends to infinity. In particular, we can show that the optimal cost function is expected to be quadratic of the form

<!-- formula-not-decoded -->

where the scalar K ∗ solves the equation

<!-- formula-not-decoded -->

with F defined by where

<!-- formula-not-decoded -->

This is the limiting form of Eq. (1.19).

Moreover, the optimal policy is linear of the form

<!-- formula-not-decoded -->

where L ∗ is the scalar given by

<!-- formula-not-decoded -->

To justify Eqs. (1.41)-(1.44), we show that J * as given by Eq. (1.40), satisfies the Bellman equation

<!-- formula-not-decoded -->

and that θ ∗ ( x ), as given by Eqs. (1.43)-(1.44), attains the minimum above for every x when J = J * . Indeed for any quadratic cost function J ( x ) = Kx 2 with K ≥ 0, the minimization in Bellman's equation (1.45) is written as

<!-- formula-not-decoded -->

Thus it involves minimization of a positive definite quadratic in u and can be done analytically. By setting to 0 the derivative with respect to u of the expression in braces in Eq. (1.46), we obtain

<!-- formula-not-decoded -->

so the minimizing control and corresponding policy are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By substituting this control, the minimized expression (1.46) takes the form

<!-- formula-not-decoded -->

After straightforward algebra, using Eq. (1.48) for L K , it can be verified that this expression is written as F ( K ) x 2 , with F given by Eq. (1.42). Thus when J ( x ) = Kx 2 , the Bellman equation (1.45) takes the form

<!-- formula-not-decoded -->