# 1.6.7-1.6.8: Multiagent & Adaptive Control

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 102-117
**Topics:** multiagent problems, multiagent rollout, unknown model, adaptive control

---

Still, in either case, the stochastic DP algorithm applies, with the su ffi cient statistic [whether I k or P k ( x k ♣ I k )] playing the role of the state.

## A Few Words about the Choice of an RL Method

An attractive aspect of the current RL methodology, inherited by the generality of our DP formulation, is that it can address a very broad range of challenging problems, deterministic as well as stochastic, discrete as well as continuous, etc. However, in the practical application of RL one has to contend with limited theoretical guarantees. In particular, several of the RL methods that have been successful in practice have less than solid performance properties, and may not work on a given problem, even one of the type for which they are designed.

This is a reflection of the state of the art in the field: there are no methods that are guaranteed to work for all or even most DP problems . However, there are enough methods to try on a given problem with a reasonable chance of success in the end (after some heuristic and problem specific tuning). For this reason, it is important to develop insight into the inner workings of various methods, as a means of selecting the proper type of methodology to try on a given problem.

A related consideration is the context within which a method is applied. In particular, is it a single problem that is being addressed, such as chess that has fixed rules and a fixed initial condition, or is it a family of related problems that must be periodically be solved with small variations in its data or its initial conditions? Also, are the problem data fixed or may they change over time as the system is being controlled?

Generally, convenient but relatively unreliable methods, which can be tuned to the problem at hand, may be tried with a reasonable chance of success if a single problem is addressed. Similarly, RL methods that require extensive tuning of parameters, including ones that involve approximation in policy space and the use of neural networks, may be well suited for a stable problem environment and a single problem solution. However, they are not well suited for problems with a variable environment and/or realtime changes of model parameters. For such problems, RL methods based on approximation in value space and on-line play, possibly involving on-line replanning, are much better suited.

Note also that even when on-line replanning is not needed, on-line play may improve substantially the performance of o ff -line trained policies, so we may wish to use it in conjunction with o ff -line training. This is

Aside from insight and intuition, it is also important to have a foundational understanding of the analytical principles of the field and of the mechanisms underlying the central computational methods. The role of the theory in this respect is to structure mathematically the methodology, guide the art, and delineate the sound from the flawed ideas.

due to the Newton step that is implicit in one-step or multistep lookahead minimization, cf. our discussion of the AlphaZero and TD-Gammon architectures in Section 1.1. Of course the computational requirements of an on-line play method may be substantial and have to be taken into account when assessing its suitability for a particular application. In this connection, deterministic problems are better suited than stochastic problems for on-line play. Moreover, methods that are well-suited for parallel computation, and/or involve the use of certainty equivalence approximations are generally better suited for a stochastic control environment.

## 1.6.2 Problems with a Termination State

Many DP problems of interest involve a termination state , i.e., a state t that is cost-free and absorbing in the sense that for all k ,

<!-- formula-not-decoded -->

Thus the control process essentially terminates upon reaching t , even if this happens before the end of the horizon. One may reach t by choice if a special stopping decision is available, or by means of a random transition from another state. Problems involving games, such as chess, Go, backgammon, and others include a termination state (the end of the game) and have played an important role in the development of the RL methodology.

Generally, when it is known that an optimal policy will reach the termination state with certainty within at most some given number of stages N , the DP problem can be formulated as an N -stage horizon problem, with a very large termination cost for the nontermination states. ‡ The reason is that even if the termination state t is reached at a time k &lt; N , we can extend our stay at t for an additional N -k stages at no additional cost, so the optimal policy will still be optimal, since it will not incur the large termination cost at the end of the horizon.

## Example 1.6.1 (Parking)

A driver is looking for inexpensive parking on the way to his destination. The parking area contains N spaces, numbered 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, and a garage following space N -1. The driver starts at space 0 and traverses the parking

Games often involve two players/decision makers, in which case they can be addressed by suitably modified exact or approximate DP algorithms. The DP algorithm that we have discussed in this chapter involves a single decision maker, but can be used to find an optimal policy for one player against a fixed and known policy of the other player.

‡ When an upper bound on the number of stages to termination is not known, the problem may be formulated as an infinite horizon problem of the stochastic shortest path problem.

have k beir

he dec

(F)|

F, с

es F a

**(F):

dF)

*(F):

c(0)

1

2

c(1)

k + 1•••

c(k)

N

c(k + 1)

c(N - 1)

Parking Spaces

Termination State tial State 15 1 5 18 4 19 9 21 25 8 12 13

Figure 1.6.1 Cost structure of the parking problem. The driver may park at space k = 0 ↪ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 at cost c ( k ), if the space is free, or continue to the next space k +1 at no cost. At space N (the garage) the driver must park at cost C .

<!-- image -->

spaces sequentially, i.e., from space k he goes next to space k +1, etc. Each parking space k costs c ( k ) and is free with probability p ( k ) independently of whether other parking spaces are free or not. If the driver reaches the last parking space N -1 and does not park there, he must park at the garage, which costs C . The driver can observe whether a parking space is free only when he reaches it, and then, if it is free, he makes a decision to park in that space or not to park and check the next space. The problem is to find the minimum expected cost parking policy.

We formulate the problem as a DP problem with N stages, corresponding to the parking spaces, and an artificial termination state t that corresponds to having parked; see Fig. 1.6.1. At each stage k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we have three states: the artificial termination state t , and the two states F and F , corresponding to space k being free or taken, respectively. At stage 0, we have only two states, F and F , and at the final stage there is only one state, the termination state t . The decision/control is to park or continue at state F [there is no choice at states F and state t ]. From location k , the termination state t is reached at cost c ( k ) when a parking decision is made (assuming location k is free). Otherwise, the driver continues to the next state at no cost. At stage N , the driver must park at cost C .

Let us now derive the form of the DP algorithm, denoting:

J ∗ k ( F ): The optimal cost-to-go upon arrival at a space k that is free.

J ∗ k ( F ): The optimal cost-to-go upon arrival at a space k that is taken.

J ∗ k ( t ): The cost-to-go of the 'parked'/termination state t .

The DP algorithm for k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1 takes the form

<!-- formula-not-decoded -->

for the states other than the termination state t , while for t we have

<!-- formula-not-decoded -->

N

The minimization above corresponds to the two choices (park or not park) at the states F that correspond to a free parking space.

While this algorithm is easily executed, it can be written in a simpler and equivalent form. This can be done by introducing the scalars

<!-- formula-not-decoded -->

which can be viewed as the optimal expected cost-to-go upon arriving at space k but before verifying its free or taken status . Indeed, from the preceding DP algorithm, we have

ˆ J N -1 = p ( N -1) min [ c ( N -1) ↪ C ] + ( 1 -p ( N -1) ) C↪ ˆ J k = p ( k ) min [ c ( k ) ↪ ˆ J k +1 ] + ( 1 -p ( k ) ) ˆ J k +1 ↪ k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -2 glyph[triangleright] From this algorithm we can also obtain the optimal parking policy:

<!-- formula-not-decoded -->

This is an example of DP simplification that occurs when the state involves components that are not a ff ected by the choice of control, and will be addressed in the next section.

## 1.6.3 General Discrete Optimization Problems

Discrete deterministic optimization problems, including challenging combinatorial problems, can be typically formulated as DP problems by breaking down each feasible solution into a sequence of decisions/controls, similar to the preceding four queens example, the scheduling Example 1.2.1, and the traveling salesman Examples 1.2.2 and 1.2.3. This formulation often leads to an intractable exact DP computation because of an exponential explosion of the number of states as time progresses. However, a reformulation to a discrete optimal control problem brings to bear approximate DP methods, such as rollout and others, to be discussed later, which can deal with the exponentially increasing size of the state space.

Let us now extend the ideas of the examples just noted to the general discrete optimization problem:

<!-- formula-not-decoded -->

where U is a finite set of feasible solutions and G ( u ) is a cost function.

We assume that each solution u has N components; i.e., it has the form

<!-- formula-not-decoded -->

where N is a positive integer. We can then view the problem as a sequential decision problem, where the components u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 are selected one-ata-time. A k -tuple ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) consisting of the first k components of a

., UN

le Uk+

..., UN.

k, Uk+1

Artificial

Initial State

Stage 1

•

U1

States

(no)

Stage 2

States

Stage 3

Stage N

) Approximate ..

<!-- image -->

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

) Approximate ..

Figure 1.6.2 Formulation of a discrete optimization problem as a DP problem with N stages. There is a cost G ( u ) only at the terminal stage on the arc connecting an N -solution u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ) upon reaching the terminal state. Note that there is only one incoming arc at each node.

solution is called a k -solution . We associate k -solutions with the k th stage of the finite horizon discrete optimal control problem shown in Fig. 1.6.2. In particular, for k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , we view as the states of the k th stage all the k -tuples ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ). For stage k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, we view u k as the control. The initial state is an artificial state denoted s . From this state, by applying u 0 , we may move to any 'state' ( u 0 ), with u 0 belonging to the set

<!-- formula-not-decoded -->

Thus U 0 is the set of choices of u 0 that are consistent with feasibility.

More generally, from a state ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) ↪ we may move to any state of the form ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ↪ u k ) ↪ upon choosing a control u k that belongs to the set

<!-- formula-not-decoded -->

These are the choices of u k that are consistent with the preceding choices u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 , and are also consistent with feasibility [we do not exclude the possibility that the set (1.63) is empty]. The last stage corresponds to the N -solutions u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ), and the terminal cost is G ( u ); see Fig. 1.6.2. All other transitions in this DP problem formulation have cost 0.

Let J * k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ) denote the optimal cost starting from the k -solution ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 ), i.e., the optimal cost of the problem over solutions whose first k components are constrained to be equal to u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 . The

DP algorithm is described by the equation

<!-- formula-not-decoded -->

with the terminal condition

<!-- formula-not-decoded -->

This algorithm executes backwards in time: starting with the known function J * N = G , we compute J * N -1 , then J * N -2 , and so on up to computing J * 0 . An optimal solution ( u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ N -1 ) is then constructed by going forward through the algorithm

<!-- formula-not-decoded -->

where U 0 is given by Eq. (1.62), and U k is given by Eq. (1.63): first compute u ∗ 0 , then u ∗ 1 , and so on up to u ∗ N -1 ; cf. Eq. (1.8).

Of course here the number of states typically grows exponentially with N , but we can use the DP minimization (1.64) as a starting point for approximation methods. For example we may try to use approximation in value space, whereby we replace J * k +1 with some suboptimal ˜ J k +1 in Eq. (1.64). One possibility is to use as

<!-- formula-not-decoded -->

the cost generated by a heuristic method that solves the problem suboptimally with the values of the first k + 1 decision components fixed at u ∗ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u ∗ k -1 ↪ u k . This is the rollout algorithm , which turns out to be a very simple and e ff ective approach for approximate combinatorial optimization.

Let us finally note that while we have used a general cost function G and constraint set U in our discrete optimization model of this section, in many problems G and/or U may have a special (e.g., additive) structure, which is consistent with a sequential decision making process and may be computationally exploited. The traveling salesman Example 1.2.2 is a case in point, where G consists of the sum of N components (the intercity travel costs), one per stage. Our next example deals with a problem of great current interest.

## Example 1.6.2 (A Large Language Model Based on N -Grams)

Let us consider an N -gram model, whereby a text string consisting of N words is transformed into another string of N words by adding a word at the front of the string and deleting the word at the back of the string. We view the text strings as states of a dynamic system a ff ected by the added word choice, which we view as the control. We denote by x k the string obtained at

indow Next Text Window Next Word Prompt

Figure 1.6.3 Schematic visualization of an LLM problem based on N -grams.

<!-- image -->

time k , and by u k the word added at time k . We assume that u k is chosen from a given set U ( x k ). Thus we have a controlled dynamic system, which is deterministic and is described by an equation of the form

<!-- formula-not-decoded -->

where f specifies the operation of adding u k at the front of x k and removing the word at the back of x k . The initial string x 0 is assumed given.

If we have a cost function G by which to evaluate a text string, we can pose a DP problem with either a finite or an infinite horizon. For example if the string evolution terminates after exactly N steps, we obtain the finite horizon problem of minimizing the function G ( x N ) of the final text string x N . In this case, x N is obtained after we have a chance to change successively all the words of the initial string x 0 , subject to the constraints u k ∈ U ( x k ).

Another possibility is to introduce a termination action, whereby addition/deletion of words is optionally stopped at some time and the final text string x is obtained with cost G ( x ). In such a problem formulation, we may also include an additive stage cost that depends on u . This is an infinite horizon formulation that involves an additional termination state t in the manner of Section 1.6.2.

Note that in both the finite and the infinite horizon formulation of the problem, the initial string x 0 may include a 'prompt,' which may be subject to optimization through some kind of 'prompt engineering.' Depending on the context, this may include the use of another optimization or heuristic algorithm, perhaps unrelated to DP, which searches for a favorable prompt from within a given set of choices.

Interesting policies for the preceding problem formulation may be provided by a neural network, such as a Generative Pre-trained Transformer (GPT). In our terms, the GPT can be viewed simply as a policy that generates next words. This policy may be either deterministic, i.e., u k = θ ( x k ) for some function θ , or it may be a 'randomized' policy, which generates u k according to a probability distribution that depends on x k . Our DP formulation can also form the basis for policy improvement algorithms such as rollout, which aim to improve the quality of the output generated by the GPT. Another, more ambitious, possibility is to consider an approximate, neural network-based, policy iteration/self-training scheme, such as the ones discussed earlier, based on the AlphaZero/TD-Gammon architecture. Such a scheme generates a sequence of GPTs, with each GPT trained with data provided by the preceding GPT, a form of self-learning in the spirit of the AlphaZero and TD-Gammon policy iteration algorithms, cf. Section 1.1.

It is also possible to provide an infinite horizon formulation of the general discrete optimization problem

<!-- formula-not-decoded -->

where U is a finite set of feasible solutions, G ( u ) is a cost function, and u consists of N components, u = ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u N -1 ); cf. Eq. (1.61). To this end, we introduce a termination state t that the system enters after N steps. At step k , the component u k is selected subject to u k ∈ U k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 , where the constraint set U k ( u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 is given by Eq. (1.63). This is a special case of a general finite to infinite horizon stochastic DP problem reformulation, which we describe in the next section.

## 1.6.4 General Finite to Infinite Horizon Reformulation

There is a conceptually important reformulation that transforms a finite horizon problem, possibly involving a nonstationary system and cost per stage, to an equivalent infinite horizon problem. It is based on introducing an expanded state space, which is the union of the state spaces of the finite horizon problem plus an artificial cost-free termination state that the system moves into at the end of the horizon. This reformulation is of great conceptual value, as it provides a mechanism to bring to bear ideas that can be most conveniently understood within an infinite horizon context. For example, it helps to understand the synergy of o ff -line training and on-line play based on Newton's method, and the related insights that explain the good performance of rollout algorithms in practice.

To define the reformulation, let us consider the N -stage horizon stochastic problem of Section 1.3.1, whose system has the form

<!-- formula-not-decoded -->

and let us denote by X k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N , and U k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, the corresponding state spaces and control spaces, respectively. We introduce an artificial termination state t , and we consider an infinite horizon problem with state and control spaces X and U given by see Fig. 1.6.4.

<!-- formula-not-decoded -->

The system equation and the control constraints of this problem are also reformulated so that states in X k , k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, are mapped to states in X k +1 , according to Eq. (1.66), while states x N ∈ X N are mapped to the termination state t at cost g N ( x N ). Upon reaching t , the state stays at t at no cost. Thus the policies of the infinite horizon problem map states

9N (XN)

JN -1 (XN-1)

MN - 1 (XN-1)

Jj (x2)

1º (x2)

Ji (x1) .

H] (х1)

J, (x0)

Но (хо)

Optimal cost and policy

XN

X2

Х1

XN

X2

X1

Xo

Optimal cost and policy

Optimal cost and policy

XN

N

Figure 1.6.4 Illustration of the infinite horizon equivalent of a finite horizon problem. The state space is X = ( ∪ N k =0 X k ) ∪ ¶ t ♦ ↪ and the control space is U = ∪ N -1 k =0 U k glyph[triangleright] Transitions from states x k ∈ X k lead to states in x k +1 ∈ X k +1 according to the system equation x k +1 = f k ( x k ↪ u k ↪ w k ), and they are stochastic when they involve the random disturbance w k . The transition from states x N ∈ X N lead deterministically to the termination state at cost g N ( x N ). The termination state t is cost-free and absorbing.

<!-- image -->

The infinite horizon optimal cost J ∗ ( x k ) and optimal policy θ ∗ ( x k ) at state x k ∈ X k of the infinite horizon problem are equal to optimal cost-to-go J ∗ k ( x k ) and optimal policy θ ∗ k ( x k ) of the finite horizon problem.

x k ∈ X k to controls in U k ( x k ) ⊂ U k , and consist of functions θ k ( x k ) that are policies of the finite horizon problem. Moreover, the Bellman equation for the infinite horizon problem is identical to the DP algorithm for the finite horizon problem.

It can be seen that the optimal cost and optimal control, J * ( x k ) and θ ∗ ( x k ), at a state x k ∈ X k in the infinite horizon problem are equal to the optimal cost-to-go J ∗ k ( x k ) and optimal control θ ∗ k ( x k ) of the original finite horizon problem, respectively; cf. Fig. 1.6.4. Moreover approximation in value space and rollout in the finite horizon problem translate to infinite horizon counterparts, and can be understood as Newton steps for solving the Bellman equation of the infinite horizon problem (or equivalently the DP algorithm of the finite horizon problem).

XN

XN

In summary, finite horizon problems can be viewed as infinite horizon problems with a special structure that involves a termination state t , and the state and control spaces of Eq. (1.67), as illustrated in Fig. 1.6.4. The Bellman equation of the infinite horizon problem coincides with the DP algorithm of the finite horizon problem. The PI algorithm for the infinite horizon problem can be translated directly to a PI algorithm for the finite horizon problem, involving repeated policy evaluations and policy improvements. Finally, the Newton step interpretations for approximation in value space and rollout schemes for the infinite horizon problem have straightforward analogs for finite horizon problems, and explain the powerful cost improvement mechanism that underlies the rollout algorithm and its variations.

## 1.6.5 State Augmentation, Time Delays, Forecasts, and Uncontrollable State Components

In practice, we are often faced with situations where some of the assumptions of our stochastic optimal control problem formulation are violated. For example, the disturbances may involve a complex probabilistic description that may create correlations that extend across stages, or the system equation may include dependences on controls applied in earlier stages, which a ff ect the state with some delay.

Generally, in such cases the problem can be reformulated into our DP problem format through a technique, which is called state augmentation because it typically involves the enlargement of the state space. The general intuitive guideline in state augmentation is to include in the enlarged state at time k all the information that is known to the controller at time k and can be used with advantage in selecting u k . State augmentation allows the treatment of time delays in the e ff ects of control on future states, correlated disturbances, forecasts of probability distributions of future disturbances, and many other complications. We note, however, that state augmentation often comes at a price: the reformulated problem may have a very complex state space. We provide some examples.

## Time Delays

In some applications the system state x k +1 depends not only on the preceding state x k and control u k , but also on earlier states and controls. Such situations can be handled by expanding the state to include an appropriate number of earlier states and controls.

As an example, assume that there is at most a single stage delay in the state and control; i.e., the system equation has the form

<!-- formula-not-decoded -->

If we introduce additional state variables y k and s k , and we make the identifications y k = x k -1 , z k = u k -1 , the system equation (1.68) yields

<!-- formula-not-decoded -->

By defining ˜ x k = ( x k ↪ y k ↪ z k ) as the new state, we have

<!-- formula-not-decoded -->

where the system function ˜ f k is defined from Eq. (1.69).

By using the preceding equation as the system equation and by expressing the cost function in terms of the new state, the problem is reduced to a problem without time delays. Naturally, the control u k should now depend on the new state ˜ x k , or equivalently a policy should consist of functions θ k of the current state x k , as well as the preceding state x k -1 and the preceding control u k -1 .

When the DP algorithm for the reformulated problem is translated in terms of the variables of the original problem, it takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar reformulations are possible when time delays appear in the cost or the control constraints; for example, in the case where the cost is

<!-- formula-not-decoded -->

The extreme case of time delays in the cost arises in the nonadditive form

<!-- formula-not-decoded -->

Then, the problem can be reduced to the standard problem format, by using as augmented state

<!-- formula-not-decoded -->

and E { g N (˜ x N ) } as reformulated cost. Policies consist of functions θ k of the present and past states x k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x 0 , the past controls u k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u 0 , and the past disturbances w k -1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w 0 . Naturally, we must assume that the past disturbances are known to the controller. Otherwise, we are faced with a problem where the state is imprecisely known to the controller, which will be discussed in the next section.

## Forecasts

Consider a situation where at time k the controller has access to a forecast y k that results in a reassessment of the probability distribution of the subsequent disturbance w k and, possibly, future disturbances. For example, y k may be an exact prediction of w k or an exact prediction that the probability distribution of w k is a specific one out of a finite collection of distributions. Forecasts of interest in practice are, for example, probabilistic predictions on the state of the weather, the interest rate for money, and the demand for inventory. Generally, forecasts can be handled by introducing additional state variables corresponding to the information that the forecasts provide. We will illustrate the process with a simple example.

Assume that at the beginning of each stage k , the controller receives an accurate prediction that the next disturbance w k will be selected according to a particular probability distribution out of a given collection of distributions ¶ P 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ P m ♦ ; i.e., if the forecast is i , then w k is selected according to P i . The a priori probability that the forecast will be i is denoted by p i and is given.

The forecasting process can be represented by means of the equation

<!-- formula-not-decoded -->

where y k +1 can take the values 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , corresponding to the m possible forecasts, and ξ k is a random variable taking the value i with probability p i . The interpretation here is that when ξ k takes the value i , then w k +1 will occur according to the distribution P i .

By combining the system equation with the forecast equation y k +1 = ξ k , we obtain an augmented system given by

<!-- formula-not-decoded -->

The new state and disturbance are

<!-- formula-not-decoded -->

The probability distribution of ˜ w k is determined by the distributions P i and the probabilities p i , and depends explicitly on ˜ x k (via y k ) but not on the prior disturbances.

Thus, by suitable reformulation of the cost, the problem can be cast as a stochastic DP problem. Note that the control applied depends on both the current state and the current forecast. The DP algorithm takes the form

<!-- formula-not-decoded -->

where y k may take the values 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and the expectation over w k is taken with respect to the distribution P y k .

Note that the preceding formulation admits several extensions. One example is the case where forecasts can be influenced by the control action (e.g., pay extra for a more accurate forecast), and may involve several future disturbances. However, the price for these extensions is increased complexity of the corresponding DP algorithm.

## Problems with Uncontrollable State Components

In many problems of interest the natural state of the problem consists of several components, some of which cannot be a ff ected by the choice of control. In such cases the DP algorithm can be simplified considerably, and be executed over the controllable components of the state.

As an example, let the state of the system be a composite ( x k ↪ y k ) of two components x k and y k . The evolution of the main component, x k , is a ff ected by the control u k according to the equation

<!-- formula-not-decoded -->

where the distribution P k ( w k ♣ x k ↪ y k ↪ u k ) is given. The evolution of the other component, y k , is governed by a given conditional distribution P k ( y k ♣ x k ) and cannot be a ff ected by the control, except indirectly through x k . One is tempted to view y k as a disturbance, but there is a di ff erence: y k is observed by the controller before applying u k , while w k occurs after u k is applied, and indeed w k may probabilistically depend on u k .

It turns out that we can formulate a DP algorithm that is executed over the controllable component of the state, with the dependence on the uncontrollable component being 'averaged out' (see also the parking Example 1.6.1). In particular, let J * k ( x k ↪ y k ) denote the optimal cost-to-go at

stage k and state ( x k ↪ y k ), and define

<!-- formula-not-decoded -->

Note that the preceding expression can be interpreted as an 'average costto-go' at x k (averaged over the values of the uncontrollable component y k ). Then, similar to the parking Example 1.6.1, a DP algorithm that generates ˆ J k ( x k ) can be obtained, and has the following form:

<!-- formula-not-decoded -->

This is a consequence of the calculation

<!-- formula-not-decoded -->

Note that the minimization in the right-hand side of the preceding equation must still be performed for all values of the full state ( x k ↪ y k ) in order to yield an optimal control law as a function of ( x k ↪ y k ). Nonetheless, the equivalent DP algorithm (1.71) has the advantage that it is executed over a significantly reduced state space. Later, when we consider approximation in value space, we will find that it is often more convenient to approximate ˆ J k ( x k ) than to approximate J * k ( x k ↪ y k ); see the following discussions of forecasts and of the game of tetris.

As an example, consider the augmented state resulting from the incorporation of forecasts, as described earlier. Then, the forecast y k represents an uncontrolled state component, so that the DP algorithm can be simplified as in Eq. (1.71). In particular, assume that the forecast y k can take values i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m with probability p i . Then, by defining

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1.6.5 Illustration of a tetris board.

<!-- image -->

which is executed over the space of x k rather than x k and y k . Note that this is a simpler algorithm to approximate than the one of Eq. (1.70).

Uncontrollable state components often occur in arrival systems, such as queueing, where action must be taken in response to a random event (such as a customer arrival) that cannot be influenced by the choice of control. Then the state of the arrival system must be augmented to include the random event, but the DP algorithm can be executed over a smaller space, as per Eq. (1.71). Here is an example of this type.

## Example 1.6.3 (Tetris)

Tetris is a popular video game played on a two-dimensional grid. Each square in the grid can be full or empty, making up a 'wall of bricks' with 'holes' and a 'jagged top' (see Fig. 1.6.5). The squares fill up as blocks of di ff erent shapes fall from the top of the grid and are added to the top of the wall. As a given block falls, the player can move horizontally and rotate the block in all possible ways, subject to the constraints imposed by the sides of the grid and the top of the wall. The falling blocks are generated independently according to some probability distribution, defined over a finite set of standard shapes. The game starts with an empty grid and ends when a square in the top row becomes full and the top of the wall reaches the top of the grid. When a row of full squares is created, this row is removed, the bricks lying above this row move one row downward, and the player scores a point. The player's objective is to maximize the score attained (total number of rows removed) up to termination of the game, whichever occurs first.

We can model the problem of finding an optimal tetris playing strategy as a finite horizon stochastic DP problem, with very long horizon. The state consists of two components:

- (1) The board position, i.e., a binary description of the full/empty status of each square, denoted by x .

## (2) The shape of the current falling block, denoted by y .

The control, denoted by u , is the horizontal positioning and rotation applied to the falling block. There is also an additional termination state which is cost-free. Once the state reaches the termination state, it stays there with no change in score. Moreover there is a very large amount added to the score when the end of the horizon is reached without the game having terminated.

The shape y is generated according to a probability distribution p ( y ), independently of the control, so it can be viewed as an uncontrollable state component. The DP algorithm (1.71) is executed over the space of board positions x and has the intuitive form

<!-- formula-not-decoded -->

where g ( x↪ y↪ u ) is the number of points scored (rows removed),

f ( x↪ y↪ u ) is the next board position (or termination state), when the state is ( x↪ y ) and control u is applied, respectively. The DP algorithm (1.72) assumes a finite horizon formulation of the problem.

Alternatively, we may consider an undiscounted infinite horizon formulation, involving a termination state (i.e., a stochastic shortest path problem). The 'reduced' form of Bellman's equation, which corresponds to the DP algorithm (1.72), has the form

<!-- formula-not-decoded -->

The value ˆ J ( x ) can be interpreted as an 'average score' at x (averaged over the values of the uncontrollable block shapes y ).

Finally, let us note that despite the simplification achieved by eliminating the uncontrollable portion of the state, the number of states x is still enormous, and the problem can only be addressed by suboptimal methods.

## 1.6.6 Partial State Information and Belief States

We have assumed so far that the controller has access to the exact value of the current state x k , so a policy consists of a sequence of functions of x k . However, in many practical settings, this assumption is unrealistic

Tetris is generally considered to be an interesting and challenging stochastic testbed for RL algorithms, and has received a lot of attention over a period spanning 20 years (1995-2015), starting with the paper [TsV96], the subsequent paper [BeI96] and the neuro-dynamic programming book [BeT96], and ending with the papers [GGS13], [SGG15], which contain many references to related works in the intervening years. All of these works are based on approximation in value space and various forms of approximate policy iteration.