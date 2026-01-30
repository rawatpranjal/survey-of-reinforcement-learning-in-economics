## Variance-reduced Q -learning is minimax optimal

Martin J. Wainwright UC Berkeley

Departments of Statistics and EECS Voleon Group, Berkeley, CA

wainwrig@berkeley.edu

## Abstract

We introduce and analyze a form of variance-reduced Q -learning. For γ -discounted MDPs with finite state space X and action space U , we prove that it yields an glyph[epsilon1] -accurate estimate of the optimal Q -function in the glyph[lscript] ∞ -norm using O (( D glyph[epsilon1] 2 (1 -γ ) 3 ) log ( D (1 -γ ) )) samples, where D = |X|×|U| . This guarantee matches known minimax lower bounds up to a logarithmic factor in the discount complexity. By contrast, our past work shows that ordinary Q -learning has worst-case quartic scaling in the discount complexity.

## 1 Introduction

Markov decision processes and reinforcement learning algorithms provide a flexible framework for decision-making in dynamic settings, and have been studied for decades (e.g., [24, 30, 8, 9, 32]). Given the explosion in the amount of available data and computing power, recent years have witnessed dramatic success of reinforcement learning (RL) techniques in various application domains (e.g., [33, 20, 29, 23, 30]). Providing a firm theoretical foundation to the trade-offs intrinsic to different classes of methods, as characterized by their access to the underlying Markov decision process, is a major open question in RL.

Such performance trade-offs have been studied in some detail for both MDPs with finite stateaction spaces (e.g., [18, 31, 13, 5, 4, 6, 7, 13, 31, 19, 16, 37]), as well as for linear state space models with quadratic rewards, known as the linear quadratic regulator (LQR) problem (e.g., [2, 1, 3, 10, 22, 35, 14, 21]). While both classes of problems are relatively idealized, gaining a precise understanding of methods in these settings provides a firm foundation for the analysis and improvement of RL algorithms in more complex settings. To provide some flavor for the quantitative trade-offs that arise, in context of the d -dimensional linear quadratic regulator, Tu and Recht [35] studied the LSTD algorithm, a model-free method for policy evaluation, and proved that it has sample complexity larger by factor d than a model-based approach that directly estimates the linear dynamics and then applies a robust solver for the Ricatti equation. As another example, in our own past work [36] on γ -discounted MDPs with finite state-action spaces, we have shown that the usual Q -learning suffers from at least worst-case fourth-order scaling in the discount complexity 1 / (1 -γ ), as opposed to the third-order scaling that is achievable by empirical Q -value-iteration [6].

In this paper, we revisit the classical problem of Q -learning in MDPs with finite state-action spaces. Our main contribution is to introduce a simple variant of Q -learning based on an appropriate form of variance reduction, and to prove that up to a logarithmic factor in discount complexity, it achieves the minimax optimal sample complexity [6] for estimating Q -functions in glyph[lscript] ∞ -norm.

Related work and our contributions In this paper, we study γ -discounted Markov decision processes with finite state space X and action space U . Throughout, we adopt the shorthand D := |X| × |U| for the total number of state-action pairs. Our main focus is the performance of iterative algorithms for estimating the optimal Q -function in the glyph[lscript] ∞ -norm, and our brief overview of the past literature is accordingly targeted. The Q -learning algorithm itself is classical [38], and there is a long line of work on its analysis (e.g., [34, 15, 31, 18, 9, 13, 37]). Moreover, a number of extensions to Q -learning have been proposed over the years (e.g., [18, 5, 19, 16]). For the Q -learning algorithm itself, our own recent work [37] established sharp upper bounds on the number of samples required to achieve an glyph[epsilon1] -accurate estimate of the optimal Q -function in glyph[lscript] ∞ -norm. Consider in particular, the synchronous or generative setting of Q -learning, in which at each iteration, we observe a new state drawn from the transition probability distribution indexed by each state-action pair. In this setting, the sample complexity of an algorithm corresponds to the total number of state-action samples required to achieve an error of glyph[epsilon1] ; to be clear, in the generative setting, the sample complexity is a factor of D larger than the number of iterations, since each iteration involves drawing D samples. Various algorithms can be compared based on their sample complexity. For ordinary Q -learning, the best known upper bound on the sample complexity required to achieve glyph[epsilon1] -accuracy in the glyph[lscript] ∞ -norm scales as O ( r 2 max glyph[epsilon1] 2 D log( D/δ ) (1 -γ ) 5 ) , as shown in the paper [37]. Earlier work by Azar et al. [5] had introduced an extension of Q -learning known as the speedy Q -learning algorithm, and shown that it has sample complexity O ( r 2 max glyph[epsilon1] 2 D log( D/δ ) (1 -γ ) 4 ) . In another piece of earlier work, Azar et al. [6] studied the sample complexity of model-based Q -value-iteration-that is, in which the transition probability matrices are estimated using a collection of data, and then we perform Q -value iteration using the fitted model. Note that this can be viewed as a fully batched method, since it uses all the data at once to form the empirical Bellman operator. Under the same assumptions as above, they proved that this batched form of Q -value-iteration yields an glyph[epsilon1] -accurate estimate with probability at least 1 -δ using a total of O ( r 2 max glyph[epsilon1] 2 D log( D/δ ) (1 -γ ) 3 ) samples. Moreover, they proved that this sample complexity is minimax optimal. In a more recent line of work, brought to our attention after initial posting of this work, Sidford et al. [28, 27] substantially strengthened the results of Azar et al. [6], in particular by showing that a mini-batched form of value iteration, together with a form of variance reduction applied to the batch operators, is not only minimax optimal in estimating the value functions, but also can be used to return a policy whose value function is glyph[epsilon1] -close to the true value function. This strengthening requires an algorithm that carefully maintains certain monotonicity relations at each iterate, along with some delicate analysis. In other earlier work, Lattimore and Hutter [19], working in the more challenging on-line setting, studied an extension of the UCLR algorithm, and proved that it achieves the optimal 1 / (1 -γ ) 3 scaling in the discount complexity parameter. However, their sample complexity bound either requires restrictions on the state transition matrices, or has quadratic scaling in the number of states (as opposed to the optimal linear scaling). To the best of our knowledge, it remains unresolved as to whether this minor gap is intrinsic to the method or an artifact of the analysis.

With this past work in context, a natural question is whether there is a simple extension of the standard Q -learning algorithm that is minimax optimal. The main contribution of this paper is to answer this question in the affirmative, up to a logarithmic factor. In order to do so, we introduce an extension of Q -learning based on an appropriate form of variance reduction. To be clear, variance reduction in stochastic approximation is a well-known idea, shown to be especially fruitful in accelerating stochastic gradient methods for optimization (e.g., [25, 26, 11, 17]); in the

context of reinforcement learning, it has also been applied independently in the context of variancereduced value iteration [28, 27] as well as policy iteration [12].

The form of variance-reduced Q -learning that we study, to be specified in Section 3, is relatively simple to describe and implement, and can be seen to be using the same variance-reduction device as the SVRG algorithm in stochastic optimization [17]. Our main result is a sharp analysis of this procedure, showing that it has minimax optimal sample complexity [6] up to a logarithmic factor in the discount complexity 1 / (1 -γ ). Analysis of variance-reduced Q -learning requires techniques different from those used in stochastic optimization, in particular building off the non-asymptotic bounds for cone-contractive operators introduced in our past work [37], as well as recent work [6, 19] in reinforcement learning that provides control on the variance of the empirical Bellman operator and related quantities.

The remainder of this paper is organized as follows. We begin in Section 2 with basic background on Markov decision processes and the Q -learning algorithm. In Section 3, we introduce the variance-reduced Q -learning algorithm studied in this paper, and state our main results (Theorem 1, Corollary 1 and Proposition 1) on its convergence guarantees. Section 4 is devoted to the proof of our main results, with the proofs of some auxiliary results provided in the appendix.

Notation: Throughout the paper, we use notation such as c , c ′ etc. to denote universal constants that do not depend on any parameters of the MDP, including the discount factor γ , size of state and action spaces and so on. A warning to the reader: the values of these universal constants may change from line to line within an argument.

## 2 Background

We begin by providing some standard background on Markov decision processes and the Q -learning algorithm, before discussing the effective variance in Q -learning. Our treatment is very brief; we refer the reader to various books (e.g., [24, 30, 8, 9, 32]) for more background on MDPs and reinforcement learning.

## 2.1 Markov decision processes and Q -functions

In this paper, we study Markov decision process (MDP) with a finite set of possible states X , and a finite set of possible actions U . The states evolve dynamically in time, with the evolution being influenced by the actions. More precisely, we define a collection of probability transition functions { P u ( · | x ) | ( x, u ) ∈ X × U} , indexed by state-action pairs ( x, u ). When in state x , performing an action u causes a transition to the next state drawn randomly from the transition function P u ( · | x ). The next ingredient of an MDP is a reward function r ; it maps state-action pairs to real numbers, so that r ( x, u ) is the reward received upon executing action u while in state x . A deterministic policy π is a mapping from the state space to the action space, so that action π ( x ) is taken when in state x .

The quality of a policy is measured by the expected sum of discounted rewards over all stateaction pairs in an infinite sample path. Of central interest to this paper is the Q -value-function or state-action function associated with a given policy π . For a given discount factor γ ∈ (0 , 1), it is

given by

<!-- formula-not-decoded -->

That is, it measures the expected sum of discounted rewards, conditioned on starting in state-action pair ( x, u ), and following the policy π in all subsequent transitions.

## 2.2 Bellman operators and Q -learning

Naturally, we would like to choose the policy π so as to optimize the values of the Q -function. From the classical theory of finite Markov decision processes [24, 30, 9], it is known that there exists an optimal deterministic policy, and it can be found by computing the unique fixed point of the Bellman operator. The Bellman operator is a mapping from R |X|×|U| to itself, whose ( x, u )-entry is given by

<!-- formula-not-decoded -->

It is well-known that T is γ -contractive with respect to the glyph[lscript] ∞ -norm

<!-- formula-not-decoded -->

This property ensures the existence and uniqueness of a fixed point θ ∗ , and any optimal policy takes the form π ∗ ( x ) ∈ arg max u ∈U θ ∗ ( x, u ).

In the learning context, the transition dynamics { P u ( · | x ) , ( x, u ) ∈ X × U} are unknown, so that it is not possible to exactly evaluate the Bellman operator. Instead, we assume some form of access to a simulation engine that generates samples. In this paper, we study the synchronous or generative setting , in which at each time k = 1 , 2 , . . . and for each state-action pair ( x, u ), we observe a sample x k ( x, u ) drawn according to the transition function P u ( · | x ). We note that guarantees for the sychronous setting can be transferred to guarantees for the on-line setting via notions of cover times of Markov chains; we refer the reader to the papers [13, 5] for conversions of this type.

The synchronous form of Q -learning algorithm generates a sequence of iterates { θ k } k ≥ 1 according to the recursion

<!-- formula-not-decoded -->

Here λ k ∈ (0 , 1) is a stepsize to be chosen by the user. The operator ̂ T k is a mapping from R |X|×|U| to itself, and is known as the empirical Bellman operator : its ( x, u )-entry is given by

<!-- formula-not-decoded -->

Here x k ∈ R |X|×|U| is a random matrix indexed by state-action pairs ( x, u ); entry x k ( x, u ) is drawn according to the probability distribution P u ( · | x ). By construction, for any fixed θ , we have E [ ̂ T k ( θ )] = T ( θ ), so that the empirical Bellman operator (5) is an unbiased estimate of the population Bellman operator (2). Thus, we recognize Q -learning as a particular form of stochastic approximation.

For future reference, it is worth noting that ̂ T k is also γ -contractive with respect to the glyph[lscript] ∞ -norm; in particular, we have

<!-- formula-not-decoded -->

for any pair of Q -functions θ and θ ′ , as can be verified by direct calculation.

## 2.3 The effective variance in ordinary Q -learning

As is well known from the theory of stochastic approximation, the accuracy of iterative procedures like Q -learning (4) is partly controlled by the variance of the updates. In order to make this intuition clear, let us introduce the error matrix ∆ k = θ k -θ ∗ , and rewrite the Q -learning updates (4) in the recentered form

<!-- formula-not-decoded -->

Here V k := ̂ T k ( θ ∗ ) -T ( θ ∗ ) is a zero-mean random matrix, in which entry ( x, u ) has variance

<!-- formula-not-decoded -->

The matrix of variances σ 2 ( θ ∗ ) controls the asymptotic behavior of the algorithm, and it plays a central role in our non-asymptotic analysis in the sequel.

## 3 Variance-reduced Q -learning

In this section, we give a precise specification of the variance-reduced Q -learning algorithm studied in this paper. Before doing so in Section 3.2, we begin in Section 3.1 with some intuition from an oracle form of variance reduction. In Section 3.3, we state our main theoretical result (Theorem 1) on variance-reduced Q -learning, along with a follow-up result (Proposition 1) that shows its minimax optimality up to a logarithmic factor.

## 3.1 Q -learning with oracle variance reduction

We begin with a thought experiment about an algorithm that, while neither implementable nor sensible-because it assumes access to the quantity θ ∗ that we are trying to compute-nonetheless provides helpful intuition. More precisely, suppose that we could compute both an empirical Bellman update ̂ T k ( θ ∗ ) and the population 1 Bellman update T ( θ ∗ ). In this case, we could implement the recentered 'algorithm'

<!-- formula-not-decoded -->

What is the effective variance of these updates? Again defining the error matrix ∆ k := θ k -θ ∗ , we find that it evolves according to the recursion:

<!-- formula-not-decoded -->

1 Of course, we have T ( θ ∗ ) = θ ∗ by definition, but we write the update in this way to build a natural bridge to our form of variance-reduced Q -learning.

By construction, this is entirely analogous to the evolution of the error matrix in ordinary Q -learning (7), but without the additional additive noise term. Moreover, from the γ -contractivity of the empirical Bellman update (6), each of updates ∆ ↦→ ̂ T k ( θ ∗ +∆) -̂ T k ( θ ∗ ), for k = 1 , 2 , . . . , is γ -contractive in glyph[lscript] ∞ -norm. Consequently, if we were to run this idealized algorithm (9a) with a constant step size, the error matrix ∆ k from the idealized update (9b) would vanish at a geometric rate.

While the idealized update is not implementable, it gives intuition for the form of variance reduction that we study. Given an algorithm that converges to θ ∗ , we can use one of its iterates θ as a proxy for θ ∗ , and then recenter the ordinary Q -learning updates by the quantity -̂ T k ( θ ) + T ( θ ). Note that even this recentering is not implementable, since we cannot compute the population Bellman update T ( θ ) exactly. However, we can use a set of samples to generate an unbiased approximation of it. In a nutshell, this is the form of variance-reduced Q -learning that we study.

## 3.2 An implementable form of variance reduction

With this intuition in hand, let us now describe the form of variance-reduced Q -learning that we study in this paper. At the core of the algorithm is a variance-reduced form of Q -learning, which we describe Section 3.2.1. The algorithm itself consists of a sequence of epochs, and we specify the form of each epoch in Section 3.2.2. We combine these ingredients to specify the overall algorithm in Section 3.2.3.

## 3.2.1 The basic variance-reduced update

We begin by defining a sequence of operators {V k } k ≥ 1 that define the variance-reduced Q -learning algorithm. Recall from our previous discussion that the method uses a matrix θ as a surrogate to θ ∗ , and requires an approximation to the Bellman update T ( θ ). In particular, for a given integer N ≥ 1, the parameters of the algorithm, we define the Monte Carlo approximation

<!-- formula-not-decoded -->

where D N is a collection of N i.i.d. samples (i.e., matrices with samples for each state-action pair ( x, u )). By construction, the random matrix ˜ T N ( θ ) is an unbiased approximation of population Bellman update T ( θ ). Each of its entries has variance proportional to 1 /N , so that we can control the approximation error by a suitable choice of N .

Given the pair ( θ, ˜ T N ( θ )) and a stepsize parameter λ ∈ (0 , 1), we define an operator V k on R |X|×|U| via

<!-- formula-not-decoded -->

Here ̂ T k is a version of the empirical Bellman operator constructed using a sample not in D N . Thus, the random operators ̂ T k and ˜ T N are independent. By construction, we have

<!-- formula-not-decoded -->

so that this update is unbiased as an estimate of the population Bellman update. As noted earlier, the device used to construct the variance-reduced operator (11) is the same as that used in the SVRG algorithm for stochastic optimization [17].

## 3.2.2 A single epoch

Having defined the basic variance-reduced update (11), we now describe how to exploit in a sequence of epochs. The input to each epoch is a matrix θ , corresponding to our current best guess of the optimal Q -function θ ∗ . Epochs are parameterized by their length K , corresponding to the number of iterations of the variance reduced update, and a second integer N , corresponding to the number of samples used to compute the Monte Carlo approximation ˜ T N . We summarize the operation of an epoch in terms of the following function RunEpoch :

<!-- formula-not-decoded -->

The choice of stepsize λ k = 1 1+(1 -γ ) k is motivated by our previous work on ordinary Q -learning [37], where we proved sharp non-asymptotic bounds with this choice. We use this same approach in analyzing the behavior of the variance-reduced updates (Step (3) in RunEpoch ) within each epoch. (It is worth noting that past work [31, 13] shows that the stepsize choice λ k = 1 /k leads to very poor behavior with ordinary Q -learning-in particular, a convergence rate that is exponentially slow in terms of the discount complexity parameter-and the same statement would apply to our variance-reduced updates.)

## 3.2.3 Overall algorithm

We now have the necessary ingredients to specify the variance-reduced Q -learning algorithm. The overall algorithm is parameterized by three choices: the total number of epochs M ≥ 1 to be run; the length K of each epoch; and the sequence of recentering samples { N m } M m =1 used in the M epochs. Each epoch is based on a single call to the function RunEpoch . Over all the epochs, the total number of matrix samples used in any run of the algorithm is given by KM + ∑ M m =1 N m . Given any choice of the triple ( M,K, { N m } M m =1 ), the overall algorithm takes the following form:

```
Algorithm: Variance-reduced Q -learning Inputs: ( a ) Number of epochs M ( b ) Epoch length K ( c ) Recentering sizes { N m } M m =1 (1) Initialize θ 0 = 0. (2) For epochs m = 1 , . . . , M : θ m = RunEpoch ( θ m -1 ; K,N m ).
```

## Output: Return Q -function estimate θ M

For a given tolerance parameter δ ∈ (0 , 1), we we choose the epoch length K and recentering sizes { N m } M m =1 so as to ensure that our final guarantees hold with probability at least 1 -δ . The dependence on the failure probability δ scales as log(1 /δ ). For the purposes of our analysis, we choose these parameters in the following way:

<!-- formula-not-decoded -->

and in epochs m = 1 , . . . , M , we use the

<!-- formula-not-decoded -->

A few comments about these choices are in order. First, our choice of the epoch length (13a) serves to make the error decrease by a factor of 1 / 2 in each epoch. A larger choice K is not helpful (and in fact, wastes samples), since the effective noise in variance-reduced Q -learning includes an additional bias term that persists independently of the number of iterations within the epoch. On the other hand, note that the number of samples N m used in epoch m follows the geometric progression 4 m as a function of the epoch number. This increase is needed in order to ensure that the bias introduced in our estimate of the Bellman operator ˜ T N ( θ m -1 ) decreases geometrically as a function of m . Our particular choice of the factor 4 in the geometric progression was for concreteness, and is not essential; as shown in Figure 1(b), the algorithm has qualitatively similarly convergence behavior for other choices of this parameter.

## 3.3 Theoretical guarantees

In this section, we state our main theoretical guarantees on variance-reduced Q -learning, beginning with its geometric convergence rate as a function of the epoch number (Theorem 1), followed by an upper bound on the total number of samples used (Corollary 1).

## 3.4 Geometric convergence over epochs

Our main result guarantees that variance-reduced Q -learning exhibits geometric convergence over the epochs with high probability. More precisely, we have:

Theorem 1. Given a γ -discounted MDP with optimal Q -function θ ∗ and a given error probability δ ∈ (0 , 1) , suppose that we run variance-reduced Q -learning from θ 0 = 0 for M epochs using parameters K and { N m } m ≥ 1 chosen according to the criteria (13) . Then we have

<!-- formula-not-decoded -->

An immediate consequence of Theorem 1 is that for any glyph[epsilon1] &gt; 0, running the algorithm with the number of epochs

<!-- formula-not-decoded -->

yields an output that is glyph[epsilon1] -accurate in glyph[lscript] ∞ -norm, with probability at least 1 -δ .

## 3.4.1 Illustrations of qualitative behavior

In Figure 1, we provide some plots that illustrate the qualitative behavior of variance-reduced Q -learning. In panel (a), we plot the log glyph[lscript] ∞ -error versus the number of samples for both VRQ-learning (red dashed curves), and ordinary Q -learning (blue solid curves). Due to the epoch structure of VR-Q-learning, note how the error decreases in distinct quanta. 2 For small values of the discount factor γ , the convergence rate of VR-Q-learning is very similar to that of ordinary Q -learning. On the other hand, as γ increases towards 1, we start to see the benefits of variance reduction, as predicted by our theory.

Figure 1. (a) Comparison of the convergence behavior of variance-reduced Q -learning and ordinary Q -learning with rescaled linear stepsize. For each algorithm, the figure plots the log glyph[lscript] ∞ -error ‖ θ n -θ ∗ ‖ ∞ versus the number of samples n for two different values of the discount factor γ ∈ { 0 . 50 , 0 . 85 } . As predicted by our theory, the gains from variance-reduction become significant as γ increases towards 1. (b) Behavior of variance-reduced Q -learning for different choices of the epoch reduction factor (base). In our proof, we established the result for the choice 2 . 0, but other choices are also valid, modulo slightly different choices of the parameters K and { N m } m ≥ 1 .

<!-- image -->

In Theorem 1, we proved that the algorithm converges at a geometric rate, with contraction factor 1 / 2 in terms of the number of epochs. The factor of 2 is a consequence of the term 4 m in our choice (13b) of the recentering sample sizes. More generally, by replacing the factor of 4 m with a term of the form ( C 2 ) m for any C &gt; 1, we can prove geometric convergence with contraction factor 1 /C . Panel (b) illustrates the qualitative effects of varying the choice of the base parameter C on the convergence behavior of the algorithm.

## 3.4.2 Total number of samples used

We now state a corollary that provides an explicit bound on the number of samples required to return an glyph[epsilon1] -accurate solution with high probability, as a function of the instance θ ∗ . We then specialize this result to the worst-case setting. In stating this result, we introduce the complexity

2 We have interpolated the error so as to avoid sharp jumps while conveying the qualitative behavior.

parameter,

<!-- formula-not-decoded -->

and recall that the number of epochs is given by M = ⌈ log 2 ( b 0 glyph[epsilon1] )⌉ .

Corollary 1. Consider a γ -discounted MDP with optimal Q -function θ ∗ , a given error probability δ ∈ (0 , 1) and glyph[lscript] ∞ -error level glyph[epsilon1] &gt; 0 . Then there are universal constants c, c ′ such that a total of

<!-- formula-not-decoded -->

matrix samples in the generative model is sufficient to obtain an glyph[epsilon1] -accurate estimate with probability at least 1 -δ .

See Section 4.2 for the proof of this claim.

Note that the bound (16) depends on the instance via the quantities ‖ σ ( θ ∗ ) ‖ and ‖ θ ∗ ‖ ∞ , both of which can vary substantially as a function of θ ∗ . In order to obtain worst-case bounds, we study the the class M ( γ, r max ) of all optimal Q -functions that can be obtained from a γ -discounted MDP with an r max -uniformly bounded reward function. (The reward function is r max -uniformly bounded means that max ( x,u ) ∈X×U | r ( x, u ) | ≤ r max .) Over this class, it can be shown that

<!-- formula-not-decoded -->

Applying this upper bound to equation (16) and simplifying, we find the uniform upper bound

<!-- formula-not-decoded -->

This worst-case upper bound improves upon the best known bounds for ordinary Q -learning [37], but does not match the cubic scaling in 1 / (1 -γ ) of the minimax optimal sample complexity [6]. In the following section, we show that a slightly refined analysis of our variance-reduced updates allows us to achieve the minimax optimal sample complexity.

## 3.4.3 Refining the worst-case guarantee

Suppose that we run the algorithm analyzed in Theorem 1 with glyph[epsilon1] = r max √ 1 -γ . From the bound (17), doing so requires 1 (1 -γ ) 3 samples, up to the logarithmic factor corrections. Moreover, the output of this procedure-call it θ 0 -then satisfies the bound ‖ θ 0 -θ ∗ ‖ ∞ ≤ r max √ 1 -γ with high probability. We claim that running the variance-reduced updates from this initialization for a further logarithmic number of steps yields an algorithm with minimax-optimal sample complexity.

Proposition 1 (Minimax optimality) . Consider a γ -discounted MDP with optimal Q -function θ ∗ , a given error probability δ ∈ (0 , 1) , and a given error tolerance. Then running variance-reduced Q -learning from an initial point θ 0 such that ‖ θ 0 -θ ∗ ‖ ∞ ≤ r max √ 1 -γ for a total of M = c log( r max (1 -γ ) glyph[epsilon1] ) epochs using K and { N m } m ≥ 1 chosen according to the criteria (13) , yields a solution θ M such that

<!-- formula-not-decoded -->

The total number of matrix samples, counting both the initial iterations required to obtain the initialization θ 0 and all later iterations, used to obtain this glyph[epsilon1] -accurate solution is at most

<!-- formula-not-decoded -->

In this way, we have recovered the worst-case optimal cubic scaling in 1 / (1 -γ ), matching the lower bound due to Azar at al. [6].

## 4 Proofs

We now turn to the proof of Theorem 1, as well as Corollary 1 and Proposition 1. In all cases, we defer proofs of various auxiliary lemmas to the appendices.

## 4.1 Proof of Theorem 1

We begin with the proof of Theorem 1 on the geometric convergence of variance-reduced Q -learning over epochs.

## 4.1.1 High-level roadmap

At a high-level, we prove Theorem 1 via an inductive argument. To set up the induction, we say that epoch m terminates successfully-or is 'good' for short-if its output θ m satisfies the bound

<!-- formula-not-decoded -->

Our strategy is to show that, with the specified choices of K and { N m } m ≥ 1 , the bounds (20) hold uniformly with probability at least 1 -δ , and we do so via induction on m . The inductive argument consists of two steps.

Base case m = 1 : Given the initialization θ 0 = 0, we prove that θ 1 satisfies the bound (20) with probability at least 1 -δ M .

Inductive step: In this step, we suppose that the input θ m to epoch m satisfies the bound (20). We then prove that θ m +1 satisfies the bound (20) with probability at least 1 -δ M .

Union bound: Finally, by taking a union bound over all M epochs of the algorithm, we are guaranteed that the claim (20) holds uniformly for all m = 1 , . . . , M , with probability at least 1 -δ . This implies the claimed bound (14) in the theorem statement.

## 4.1.2 Proof of the base case

For the given initialization θ 0 = 0, note that we have ̂ T k ( θ 0 ) = r and ˜ T N ( θ 0 ) = r . Consequently, we have ̂ T k ( θ 0 ) -˜ T N ( θ 0 ) = 0, so that the variance-reduced updates (11) reduce to the case of ordinary Q -learning with stepsize λ k = 1 1+(1 -γ ) k . It follows from analysis in our past work [37] that there is a universal constant c ′ &gt; 0 such that, after M iterations, we have

<!-- formula-not-decoded -->

with probability at least 1 -δ M , where the matrix of variances σ 2 ( θ ∗ ) = var ( ̂ T ( θ ∗ ) ) was defined previously (8).

Consequently, choosing K = c log ( 8 MD δ (1 -γ ) ) (1 -γ ) 3 for a sufficiently large constant c suffices to ensure that

<!-- formula-not-decoded -->

Since θ 1 = θ K +1 by definition, this bound is equivalent to the claim (20) for the base case m = 1, as desired.

## 4.1.3 Proof of inductive step

For this step, we assume that the input θ m to epoch m satisfies the bound

<!-- formula-not-decoded -->

and our goal is to show that ‖ θ m +1 -θ ∗ ‖ ∞ ≤ b m 2 . Recall that θ m +1 is equivalent to the output θ K of running K rounds of variance-reduced Q -learning from the initialization θ 0 = θ m , using the parameter N = N m for the operator ˜ T N . In this section, we prove that there is a universal constant c &gt; 0 such that

<!-- formula-not-decoded -->

with probability at least 1 -δ M . From this bound, we see that the choices of K and N m given in equation (13), with sufficiently large constants of the pre-factors c 1 and c 2 , are sufficient to ensure that ‖ θ K -θ ∗ ‖ ∞ ≤ b m 2 with probability at least 1 -δ M , as desired. Accordingly, the remainder of this section is devoted to proving the bound (23).

Throughout the remainder of this proof, we drop the subscript m as it can be implicitly understood; in particular, we use N , b and θ as shorthands for N m , b m and θ m , respectively.

Rewriting the update: Recall the form of the variance-reduced Q -learning updates (11). We begin by re-writing these updates in a form suitable for analysis using the general results on stochastic approximation from Wainwright [37]. Define the recentered operators

<!-- formula-not-decoded -->

By recentering the updates (11) around the optimal Q -function θ ∗ , we can write

<!-- formula-not-decoded -->

where W k := -̂ H k ( θ ) -T ( θ ∗ ) + ˜ T N ( θ ) is a random noise sequence. We use this noise sequence to define an auxiliary stochastic process

<!-- formula-not-decoded -->

Note that the operator H k ( θ ) = ̂ T k ( θ ) -̂ T k ( θ ∗ ) is monotonic with respect to the orthant ordering, and γ -contractive with respect to the glyph[lscript] ∞ -norm. Consequently, from past results, we have:

Corollary 2 (Adapted from the paper [37]) . For all iterations k = 1 , 2 , . . . , we have

<!-- formula-not-decoded -->

In order to derive a concrete result based on this bound, we need to obtain high-probability upper bounds on the terms ‖ P glyph[lscript] ‖ ∞ . We begin by decomposing the effective noise into a sum of three terms that can be controlled nicely. Recalling the definition (24) of ̂ H k and ˜ H N , we have

<!-- formula-not-decoded -->

Thus, if we define another recentered operator H ( θ ) = T ( θ ) -T ( θ ∗ ), then we have

<!-- formula-not-decoded -->

From the linearity of the recursion (26) and the fact that W ◦ and W † are independent of k , we can write

<!-- formula-not-decoded -->

where the stochastic process { P ′ k } k ≥ 1 evolves according to a recursion of the form (26) with W k replaced by W ′ k . Applying the bound (27) at iteration K and using the fact that ‖ θ 1 -θ ∗ ‖ ∞ ≤ b by assumption, we find that

<!-- formula-not-decoded -->

Thus, it remains to bound the two terms on the right-hand side, involving the noise terms W ◦ and W † , as well as the stochastic process { P ′ k } k ≥ 1 .

## 4.1.4 Bounding the recentering terms

We begin by bounding the noise terms W ◦ and W † , which arise from differences between the population Bellman operator T and the randomized approximation ˜ T N used to recenter the iterates throughout the given epoch. Note that both W ◦ and W † are zero mean random variables, formed of sums of N i.i.d. terms, so that we can control their magnitudes by increasing N . The following lemma makes this intuition precise:

Lemma 1 (High probability bounds on recentering terms) . Fix an arbitrary δ ∈ (0 , 1) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(b) There is a universal constant c such that

<!-- formula-not-decoded -->

The proof is a relatively straightforward application of Hoeffding's inequality (for the bound (30a)) and Bernstein's inequality (for the bound (30b)). See Appendix A.1 for the details.

## 4.1.5 Bounding the process { P ′ k } k ≥ 1

Our next step is to control the terms in the bound (29) that depend on the stochastic process { P ′ k } k ≥ 1 .

Lemma 2 (High probability bound on noise) . There is a universal constant c &gt; 0 such that for any δ ∈ (0 , 1)

<!-- formula-not-decoded -->

with probability at least 1 -δ 3 M .

The proof of this lemma is more involved, in particular involving an inductive argument to control the MGF of the process { P ′ k } k ≥ 1 . See Appendix A.2 for the details.

## 4.1.6 Putting together the pieces

We now put together the pieces, in particular using the bounds (30) and (31) to control the terms in the inequality (29), with the ultimate goal of proving the claim (23). Doing so yields that there are universal constants such that

<!-- formula-not-decoded -->

By union bound over the three different bounds that we have applied (each holding with probability at least 1 -δ 3 M ), the overall bound holds with probability at least 1 -δ M . Recalling that b = ‖ σ ( θ ∗ ) ‖ ∞ + ‖ θ ∗ ‖ ∞ (1 -γ ) 2 m , we have

<!-- formula-not-decoded -->

Putting together the pieces yields that, with probability at least 1 -δ M , we have

<!-- formula-not-decoded -->

for some universal constant c . Thus, we have proved the desired claim (23), which completes the proof of Theorem 1.

## 4.2 Proof of Corollary 1

Now let us prove the bound on the total number of samples used, as stated in Corollary 1. Recall that we use K samples for the Q -learning updates within each epoch, and N m samples to compute the recentering operator ˜ T N m in epoch m . Defining b 0 := ‖ σ ( θ ∗ ) ‖ ∞ + ‖ θ ∗ ‖ ∞ (1 -γ ), the total number of samples is used

<!-- formula-not-decoded -->

as claimed in equation (16).

## 4.3 Proof of Proposition 1

We now turn to the proof of Proposition 1, which applies a more refined analysis to guarantee the worst-case optimal sample complexity. At a high level, our proof is based on showing that under the stated conditions, the epoch iterates { θ m } M m =1 satisfy

<!-- formula-not-decoded -->

with probability at least 1 -δ .

We follow the same inductive argument as before. The base case is trivial, so that it only remains to establish the inductive step. In the epoch that moves from θ m to θ m +1 , variance-reduced Q -learning uses θ ≡ θ m in its recentering process. By the induction hypothesis, we have

<!-- formula-not-decoded -->

In this case, our analysis of the epoch is based on the two operators

<!-- formula-not-decoded -->

Note that the variance-reduced Q -learning updates can be written as

<!-- formula-not-decoded -->

Moreover, note that J is γ -contractive, it has a unique fixed point, which we denote by ̂ θ . Since J ( θ ) = E [ ̂ J k ( θ )] by construction, it is natural to analyze the convergence of θ k to ̂ θ .

From the initialization θ , suppose that we run the variance-reduced updates with recentering point θ for K steps, thereby obtaining the estimate θ K +1 . Given a bound on ‖ θ K +1 -̂ θ ‖ ∞ , we can then bound the distance to θ ∗ via the triangle inequality-viz.

<!-- formula-not-decoded -->

With this decomposition, our proof of Proposition 1 hinges on the following two lemmas. The first lemma bounds the error of θ K +1 as an estimate of the fixed point ̂ θ :

Lemma 3. After K = c 1 log ( 8 MD (1 -γ ) δ ) (1 -γ ) 3 iterations, we are guaranteed that

<!-- formula-not-decoded -->

See Appendix B.1 for the proof of this claim.

Note that ̂ θ is a fixed point of the operator J from equation (34a), which can be seen as a perturbed version of the original Bellman operator T , for which θ ∗ is the fixed point. Our second lemma uses this fact to bound the difference ̂ θ -θ ∗ :

Lemma 4. Given a sample size N m = c 2 4 m log( MD/δ ) (1 -γ ) 2 , we have

<!-- formula-not-decoded -->

See Appendix B.2 for the proof of this claim.

Completing the inductive step is straightforward given these two lemmas. Combining with our earlier bound, we have

<!-- formula-not-decoded -->

where steps (i) and (ii) follow from Lemmas 3 and 4, respectively. This sequence of inequalities holds with probability at least 1 -δ M as claimed.

## 5 Discussion

In this paper, we have proposed a variance-reduced form of Q -learning, and shown that it has sample complexity that achieves the minimax optimal sample complexity, up to a logarithmic factor in the discount complexity 1 / (1 -γ ). Although our result can be summarized succinctly in this way, in fact, our analysis is instance specific, and we have proved bounds that depend on the optimal Q -function θ ∗ via its supremum norm ‖ θ ∗ ‖ ∞ , and the variance σ 2 ( θ ∗ ) of the associated empirical Bellman update. Although the analysis of this paper focuses purely on the tabular setting, the variance-reduced Q -learning algorithm itself can be applied in more generality. It would be interesting to explore the uses of this algorithm in more general settings.

## Acknowledgements

This work was partially supported by Office of Naval Research Grant ONR-N00014-18-1-2640 and National Science Foundation Grant NSF-DMS-1612948. We thank A. Pananjady and K. Khamaru for careful reading and comments on an earlier draft.

## A Auxiliary lemmas for Theorem 1

In this appendix, we collect the proofs of Lemmas 1 and 2, used in the proof of Theorem 1.

## A.1 Proof of Lemma 1

The lemma consists of the two separate bounds (30a) and (30b), and we split our proof accordingly.

Proof of bound (30a) : By definition, the random operator ˜ H N ( θ ) = 1 N ∑ N i =1 { ̂ T i ( θ ) -̂ T i ( θ ∗ ) } is the sum of N i.i.d. terms. Each random operator ̂ T i is γ -contractive, so that for each state-action pair ( x, u ), we have

<!-- formula-not-decoded -->

Consequently, each entry of the noise matrix W ◦ is zero-mean, and the i.i.d. sum of N random variables bounded in absolute value by b . Therefore, the claim follows from Hoeffding's inequality for bounded random variables [36].

Proof of bound (30b) : Note that ˜ T N ( θ ∗ ) -T ( θ ∗ ) is a sum of N i.i.d. terms, each bounded in absolute value by ‖ θ ∗ ‖ ∞ , and with variance matrix σ 2 ( θ ∗ ), as was previously defined in equation (8). Consequently, by a combination of the union bound (over state-action pairs) and Bernstein's inequality, there is a universal constant c such that, with probability at least 1 -δ 3 M , we have

<!-- formula-not-decoded -->

where the final inequality follows since N ≥ c 4 m log(8 MD/δ ) (1 -γ ) 2 .

## A.2 Proof of Lemma 2

Our proof consists of two steps. First, we prove by induction that the moment-generating function of P ′ k ( x, u ) is upper bounded as

<!-- formula-not-decoded -->

uniformly over all state-action pairs ( x, u ). Combining with the Chernoff bounding technique and the union bound, we find that there is a universal constant c such that

<!-- formula-not-decoded -->

Taking union bound over all K iterations, we find that

<!-- formula-not-decoded -->

with probability at least 1 -δ 3 M . From the proof of Corollary 3 in the paper [37], we have

<!-- formula-not-decoded -->

again for some universal constant c . Putting together the pieces yields the claimed bound (31).

## A.2.1 Proof of the claim (37)

It remains to prove the bound (37). Recall that the stochastic process { P ′ k } k ≥ 1 evolves according to the recursion P ′ k +1 = (1 -λ k ) P ′ k + λ k W ′ k , where

<!-- formula-not-decoded -->

Since the operator ̂ T k is γ -contractive, we have

<!-- formula-not-decoded -->

where the final step uses the assumption that ‖ θ -θ ∗ ‖ ∞ ≤ b . Moreover, we have E [ ̂ H k ( θ )] = H ( θ ), so that each W ′ k is a zero-mean random matrix, with its entries bounded in absolute value by b . Consequently, by standard results on sub-Gaussian variables (cf. Chapter 2, [36]), we have

<!-- formula-not-decoded -->

valid for each state-action pair ( x, u ).

We use this auxiliary result to prove the claim (37) by induction.

Base case: The case k = 1 is trivial, since P ′ 1 = 0. Turning to the case k = 2, we have P ′ 2 = λ 1 W ′ 1 , and hence

<!-- formula-not-decoded -->

where the final bound follows from the fact that λ 1 = 1 1+(1 -γ ) ≤ 1.

Induction step: Next we assume that the claim (37) holds for some iteration k ≥ 1, and we verify that it holds at iteration k +1. By definition of P ′ k +1 and the independence of P ′ k and W ′ k , we have

<!-- formula-not-decoded -->

where the inequality makes use of the inductive assumption, and the bound (39). Recalling that λ k = 1 1+(1 -γ ) k , we have

<!-- formula-not-decoded -->

Consequently, we have

<!-- formula-not-decoded -->

which verifies the claim (37) for k +1.

## B Auxiliary lemmas for Proposition 1

In this appendix, we collect the proofs of Lemmas 3 and 4, used in the proof of Proposition 1.

## B.1 Proof of Lemma 3

We begin by re-writing the recursion (34b) in a form suitable for application of our results from past work [37]. Subtracting off the fixed point ̂ θ of J , we find that

<!-- formula-not-decoded -->

Note that the operator θ ↦→ ̂ J k ( θ ) -̂ J k ( ̂ θ ) is γ -contractive and monotonic so that Corollary 1 from Wainwright [37] can be applied. In applying this corollary, the effective noise term is given by

<!-- formula-not-decoded -->

Consequently, we have ‖ E k ‖ ∞ ≤ 2 ‖ ̂ θ -θ ‖ ∞ , a fact that is useful in bounding the effect of these noise terms on the evolution. By adapting Corollary 1 from the paper [37], we have

<!-- formula-not-decoded -->

where the auxiliary stochastic process evolves as P k = (1 -λ k -1 ) P k -1 + λ k -1 E k -1 . Following the same line of argument as in the proof of Lemma 2 (see Section A.2), we find that

<!-- formula-not-decoded -->

with probability at least 1 -δ 2 M . Here we have used the fact that log(8 DM/δ ) K ≤ 1 by assumption.

Consequently, with the choice K = c 1 log ( 8 MD (1 -γ ) δ ) (1 -γ ) 3 , we are guaranteed that

<!-- formula-not-decoded -->

with probability at least 1 -δ 2 M .

## B.2 Proof of Lemma 4

Note that ̂ θ is the fixed point of the operator J ( θ ) := T ( θ ) - T ( θ ) + ˜ T N ( θ ), and hence can be seen as a fixed point of the population Bellman operator defined with perturbed reward function ˜ r with entries ˜ r ( x, u ) = r ( x, u ) + [ ˜ T N ( θ ) -T ( θ ) ] ( x, u ). The following lemma guarantees that this perturbation is relatively small, where the reader should recall the standard deviation σ ( θ ∗ ) that was previously defined (8).

Lemma 5 (Bounds on perturbed rewards) . For any matrix θ such that ‖ θ -θ ∗ ‖ ∞ ≤ b m , we have

<!-- formula-not-decoded -->

with probability at least 1 -δ 8 M .

We also require a lemma that provides elementwise upper bounds on the absolute difference | θ ∗ -̂ θ | in terms of the absolute difference | ˜ r -r | . In order to state these bounds, we follow the notation of Azar et al. [6], letting P π ∗ denote the linear operator defined by the policy π ∗ that is optimal with respect to θ ∗ , and similarly letting P ˆ π denote the linear operator defined by the policy ˆ π that is optimal with respect to ̂ θ .

Lemma 6 (Elementwise bounds) . We have the elementwise upper bound:

<!-- formula-not-decoded -->

Equipped with these lemmas, we now proceed to prove the claim. From the inequality (41), it suffices to bound the elements of the two vectors ( I -γ P π ∗ ) -1 | ˜ r -r | and ( I -γ P ˆ π ) -1 | ˜ r -r | .

Upper bounding ( I -γ P π ∗ ) -1 | ˜ r -r | : On one hand, from Lemma 5 and the fact that the matrix ( I -γ P π ∗ ) -1 has non-negative entries, we have

<!-- formula-not-decoded -->

where we have also used the fact that ‖ ( I -γ P π ∗ ) -1 u ‖ ∞ ≤ ‖ u ‖ ∞ 1 -γ for any vector u . Now we have

<!-- formula-not-decoded -->

where step (i) follows from Lemma 8 of Azar et al. [6], and step (ii) follows since b m = 1 2 m 1 √ 1 -γ . Similarly, we have ‖ θ ∗ ‖ ∞ 1 -γ ≤ 1 (1 -γ ) 2 ≤ 2 m b m (1 -γ ) 3 / 2 . Putting together the pieces yields the elementwise bound

<!-- formula-not-decoded -->

where we define the non-negative scalar

<!-- formula-not-decoded -->

for a sufficiently large but universal constant c ′ .

Upper bounding ( I -γ P ˆ π ) -1 | ˜ r -r | : The only term that needs to be handled differently is the one involving σ ( θ ∗ ). Let σ ( ̂ θ ) denote the variance under the transition function P of the Q -function ̂ θ . Again, by the results of Azar et al. [6], we are guaranteed that ‖ ( I -γ P ˆ π ) -1 σ ( ̂ θ ) ‖ ∞ ≤ 4 (1 -γ ) 3 / 2 . Moreover, we have σ ( θ ∗ ) glyph[precedesequal] σ ( ̂ θ ) + | ̂ θ -θ ∗ | . Combining the pieces, we are guaranteed to have the elementwise bound

<!-- formula-not-decoded -->

where the vector Φ was previously defined in equation (42b).

Putting together the pieces: By combining the bounds (42a) and (43) with Lemma 6, we find that

<!-- formula-not-decoded -->

Our choice of N ensures that c 1 -γ √ log(8 MD/δ ) N ≤ 1 2 , so that we have established the upper bound ‖ ̂ θ -θ ∗ ‖ ∞ ≤ 2 b m Φ( N,m,γ ). Finally, returning to the definition (42b) of Φ, we see that our choice of N ensures that ‖ Φ( N,m,γ ) ‖ ∞ ≤ 1 10 , so that the claim follows.

## B.2.1 Proof of Lemma 5

Starting with the definition of ˜ r and adding and subtracting terms, we obtain the bound

<!-- formula-not-decoded -->

By definition, the random matrix ˜ T N ( θ ) -˜ T N ( θ ∗ ) is the sum of N i.i.d. terms. The entries in each term are uniformly bounded by γ ‖ θ -θ ∗ ‖ ∞ ≤ b m . Consequently, by a combination of Hoeffding's inequality and the union bound, we find that

<!-- formula-not-decoded -->

with probability at least 1 -δ 4 M . Turning to the term | ˜ T N ( θ ∗ ) -T ( θ ∗ ) | , by a Bernstein inequality, we have

<!-- formula-not-decoded -->

Combining the pieces yields the claim.

## B.2.2 Proof of Lemma 6

In this proof, we make use of the function | u | + = max { u, 0 } , applied elementwise to a vector u . Note that we have | u | = max {| u | + , | -u | + } by definition. Using this fact, it suffices to prove the two elementwise bounds:

<!-- formula-not-decoded -->

Recall that θ ∗ and ̂ θ are the optimal Q -functions for the reward functions r and ˜ r , respectively, with corresponding optimal policies π ∗ and ˆ π , respectively. By this optimality, we have

<!-- formula-not-decoded -->

Proof of inequality (44) (i): Using these relations, we can write

<!-- formula-not-decoded -->

where we have used the non-negativity of the entries of γ P π ∗ , and the fact that θ ∗ -̂ θ glyph[precedesequal] | θ ∗ -̂ θ | + . Since the RHS is non-negative, this inequality implies that

<!-- formula-not-decoded -->

Re-arranging and using the non-negativity of the entries of the matrix ( I -γ P π ∗ ) -1 , we find that | θ ∗ -̂ θ | + glyph[precedesequal] ( I -γ P π ∗ ) -1 | ˜ r -r | , as claimed in inequality (44)(i).

Proof of inequality (44) (ii): In the other direction, similar reasoning yields

<!-- formula-not-decoded -->

and hence | ̂ θ -θ ∗ | + glyph[precedesequal] ( I -γ P ˆ π ) -1 | ˜ r -r | , as claimed in equation (44)(ii).

## References

- [1] Y. Abbasi-Yadkori, N. Lazic, and C. Szepesv´ ari. Regret bounds for model-free linear quadratic control. arXiv preprint arXiv:1804.06021 , 2018.
- [2] Y. Abbasi-Yadkori and C. Szepesv´ ari. Regret bounds for the adaptive control of linear quadratic systems. In Conference on Learning Theory , pages 1-26, 2011.
- [3] M. Abeille and A. Lazaric. Improved regret bounds for Thompson sampling in linear quadratic control problems. In International Confernce on Machine Learning , pages 1-9, 2018.
- [4] S. Agrawal and R. Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Advances in Neural Information Processing Systems , pages 1184-1194, 2017.
- [5] M. G. Azar, R. Munos, M. Ghavamzadeh, and H. J. Kappen. Speedy Q -learning. In Neural Information Processing Systems , pages 2411-2419, 2011.
- [6] M. G. Azar, R. Munos, and H. J. Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning , 91:325-349, 2013.
- [7] M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , 2017.
- [8] D. P. Bertsekas. Dynamic programming and stochastic control , volume 1. Athena Scientific, Belmont, MA, 1995.
- [9] D. P. Bertsekas and J. N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1st edition, 1996.
- [10] A. Cohen, T. Koren, and Y. Mansour. Learning linear-quadratic regulators efficiently with only √ T -regret. Technical Report arXiv:1902.06223, arXiv, February 2019.
- [11] A. Defazio, F. Bach, and S. Lacoste Julien. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. In NIPS Conference , 2014.
- [12] S. S. Du, J. Chen, L. Li, L. Xiao, and D. Zhou. Stochastic variance reduction methods for policy evaluation. Technical Report arxiv:1702.07944, Microsoft Research, February 2017.
- [13] E. Even-Dar and Y. Mansour. Learning rates for Q -learning. Journal of Machine Learning Research , 5:1-25, 2003.
- [14] M. Fazel, R. Ge, S. Kakade, and M. Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. In International Conference on Machine Learning , pages 1466-1475, 2018.

- [15] T. Jaakkola, M. I. Jordan, and S. P. Singh. On the convergence of stochastic iterative dynamic programming algorithms. Neural Computation , 6(6), November 1994.
- [16] C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is Q -learning provably efficient? Technical report, arxiv, July 2018.
- [17] R. Johnson and T. Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In NIPS Conference , 2013.
- [18] M. Kearns and S. Singh. Finite-sample convergence rates for Q -learning and indirect algorithms. In NIPS Conference , 1999.
- [19] T. Lattimore and M. Hutter. Near-optimal PAC bounds for discounted MDPs. Theoretical Computer Science , 558:125-143, 2014.
- [20] S. Levine, C. Finn, T. Darrell, and P. Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research , 17(1):1334-1373, 2016.
- [21] D. Malik, A. Panajady, K. Bhatia, K. Khamaru, P. L. Bartlett, and M. J. Wainwright. Derivative-free methods for policy optimization: Guarantees for linear-quadratic systems. In AISTATS: Conference on AI and Statistics , 2019.
- [22] H. Mania, S. Tu, and B. Recht. Certainty equivalent control of LQR is efficient. Technical Report arXiv:1902.07826, arXiv, February 2019.
- [23] V. Mnih et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, February 2015.
- [24] M. L. Puterman. Markov decision processes: Discrete stochastic dynamic programming . Wiley, 2005.
- [25] M. Schmidt, N. Le Roux, and F. Bach. Minimizing finite sums with the stochastic average gradient. Mathematical Programming , 162:83-112, March 2017.
- [26] S. Shalev-Shwartz and T. Zhang. Stochastic dual coordinate ascent methods for regularized loss minimization. Journal of Machine Learning Research , 14:567-599, 2013.
- [27] A. Sidford, M. Wang, C. Wu, L. Yang, and Y. Ye. Near-optimal time and sample complexities for solving Markov decision processes with a generative model. In NeurIPS: Advances in Neural Information Processing Systems , 2018.
- [28] A. Sidford, M. Wang, X. Wu, and Y. Ye. Variance reduced value iteration and faster algorithms for solving Markov decision processes. In Symposium on Discrete Algorithms (SODA) , 2018.
- [29] D. Silver et al. Mastering the game of Go with deep neural networks and tree search. Nature , 529(7587):484-489, January 2016.
- [30] R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction . MIT Press, Cambridge, MA, 2nd edition, 2018.
- [31] C. Szepesv´ ari. The asymptotic convergence rate of Q -learning. In NIPS 10 , pages 1064-1070, 1997.

- [32] C. Szepesv´ ari. Algorithms for reinforcement learning . Morgan-Claypool, 2009.
- [33] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In Intelligent Robots and Systems (IROS) , pages 23-30. IEEE, 2017.
- [34] J. N. Tsitsiklis. Asynchronous stochastic approximation and Q -learning. Machine Learning , 16:185-202, 1994.
- [35] S. Tu and B. Recht. The gap between model-based and model-free methods on the linear quadratic regulator: An asymptotic viewpoint. Technical report, UC Berkeley, February 2019.
- [36] M. J. Wainwright. High-dimensional statistics: A non-asymptotic viewpoint . Cambridge University Press, Cambridge, UK, 2019.
- [37] M. J. Wainwright. Stochastic approximation with cone-contractive operators: Sharp glyph[lscript] ∞ -bounds for Q-learning. Technical report, UC Berkeley, May 2019. arxiv:1905.06265.
- [38] C. Watkins and P. Dayan. Q -learning. Machine Learning , 8:279-292, 1992.