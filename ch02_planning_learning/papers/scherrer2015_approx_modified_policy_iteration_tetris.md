## Approximate Modified Policy Iteration and its Application to the Game of Tetris

## Bruno Scherrer

Bruno.Scherrer@inria.fr

INRIA Nancy - Grand Est, Team Maia, 615 rue du Jardin Botanique, 54600 Vandœuvre-ls-Nancy, France

## Mohammad Ghavamzadeh

Mohammad.Ghavamzadeh@inria.fr

Adobe Research &amp; INRIA Lille 321 Park Avenue San Jose, CA 95110, USA

## Victor Gabillon

Victor.Gabillon@inria.fr

INRIA Lille - Nord Europe, Team SequeL, 40 avenue Halley, 59650 Villeneuve d'Ascq, France

## Boris Lesner

lesnerboris@gmail.com

INRIA Nancy - Grand Est, Team Maia, 615 rue du Jardin Botanique, 54600 Vandœuvre-ls-Nancy, France

## Matthieu Geist

matthieu.geist@centralesupelec.fr

CentraleSup´ elec, IMS-MaLIS Research Group &amp; UMI 2958 (GeorgiaTech-CNRS), 2 rue Edouard Belin, 57070 Metz, France

Editor:

Shie Mannor

## Abstract

Modified policy iteration (MPI) is a dynamic programming (DP) algorithm that contains the two celebrated policy and value iteration methods. Despite its generality, MPI has not been thoroughly studied, especially its approximation form which is used when the state and/or action spaces are large or infinite. In this paper, we propose three implementations of approximate MPI (AMPI) that are extensions of the well-known approximate DP algorithms: fitted-value iteration, fitted-Q iteration, and classification-based policy iteration. We provide error propagation analysis that unify those for approximate policy and value iteration. We develop the finite-sample analysis of these algorithms, which highlights the influence of their parameters. In the classification-based version of the algorithm (CBMPI), the analysis shows that MPI's main parameter controls the balance between the estimation error of the classifier and the overall value function approximation. We illustrate and evaluate the behavior of these new algorithms in the Mountain Car and Tetris problems. Remarkably, in Tetris, CBMPI outperforms the existing DP approaches by a large margin, and competes with the current state-of-the-art methods while using fewer samples. 1

1. This paper is a significant extension of two conference papers by the authors (Scherrer et al., 2012; Gabillon et al., 2013). Here we discuss better the relation of the AMPI algorithms with other approximate DP methods, and provide more detailed description of the algorithms, proofs of the theorems, and report

Keywords: approximate dynamic programming, reinforcement learning, Markov decision processes, finite-sample analysis, performance bounds, game of tetris

## 1. Introduction

Modified Policy Iteration (MPI) (Puterman, 1994, Chapter 6, and references therein for a detailed historical account) is an iterative algorithm to compute the optimal policy and value function of a Markov Decision Process (MDP). Starting from an arbitrary value function v 0 , it generates a sequence of value-policy pairs

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where G v k is a greedy policy w.r.t. (with respect to) v k , T π k is the Bellman operator associated to the policy π k , and m ≥ 1 is a parameter. MPI generalizes the well-known dynamic programming algorithms: Value Iteration (VI) and Policy Iteration (PI) for the values m = 1 and m = ∞ , respectively. MPI has less computation per iteration than PI (in a way similar to VI), while enjoys the faster convergence (in terms of the number of iterations) of the PI algorithm (Puterman, 1994). In problems with large state and/or action spaces, approximate versions of VI (AVI) and PI (API) have been the focus of a rich literature (see e.g. , Bertsekas and Tsitsiklis 1996; Szepesv´ ari 2010). Approximate VI (AVI) generates the next value function as the approximation of the application of the Bellman optimality operator to the current value (Singh and Yee, 1994; Gordon, 1995; Bertsekas and Tsitsiklis, 1996; Munos, 2007; Ernst et al., 2005; Antos et al., 2007; Munos and Szepesv´ ari, 2008). On the other hand, approximate PI (API) first finds an approximation of the value of the current policy and then generates the next policy as greedy w.r.t. this approximation (Bertsekas and Tsitsiklis, 1996; Munos, 2003; Lagoudakis and Parr, 2003a; Lazaric et al., 2010b, 2012). Another related algorithm is λ -policy iteration (Bertsekas and Ioffe, 1996), which is a rather complicated variation of MPI. It involves computing a fixed-point at each iteration, and thus, suffers from some of the drawbacks of the PI algorithms. This algorithm has been analyzed in its approximate form by Thiery and Scherrer (2010a); Scherrer (2013). The aim of this paper is to show that, similarly to its exact form, approximate MPI (AMPI) may represent an interesting alternative to AVI and API algorithms.

In this paper, we propose three implementations of AMPI (Section 3) that generalize the AVI implementations of Ernst et al. (2005); Antos et al. (2007); Munos and Szepesv´ ari (2008) and the classification-based API algorithms of Lagoudakis and Parr (2003b); Fern et al. (2006); Lazaric et al. (2010c); Gabillon et al. (2011). We then provide an error propagation analysis of AMPI (Section 4), which shows how the L p -norm of its performance loss

<!-- formula-not-decoded -->

of using the policy π k computed at some iteration k instead of the optimal policy π ∗ can be controlled through the errors at each iteration of the algorithm. We show that the error

of the experimental results, especially in the game of Tetris. Moreover, we report new results in the game Tetris that were obtained after the publication of our paper on this topic (Gabillon et al., 2013).

propagation analysis of AMPI is more involved than that of AVI and API. This is due to the fact that neither the contraction nor monotonicity arguments, that the error propagation analysis of these two algorithms rely on, hold for AMPI. The analysis of this section unifies those for AVI and API and is applied to the AMPI implementations presented in Section 3. We then detail the analysis of the three algorithms of Section 3 by providing their finitesample analysis in Section 5. Interestingly, for the classification-based implementation of MPI (CBMPI), our analysis indicates that the parameter m allows us to balance the estimation error of the classifier with the overall quality of the value approximation. Finally, we evaluate the proposed algorithms and compare them with several existing methods in the Mountain Car and Tetris problems in Section 6. The game of Tetris is particularly challenging as the DP methods that are only based on approximating the value function have performed poorly in this domain. An important contribution of this work is to show that the classification-based AMPI algorithm (CBMPI) outperforms the existing DP approaches by a large margin, and competes with the current state-of-the-art methods while using fewer samples.

## 2. Background

We consider a discounted MDP 〈S , A , P, r, γ 〉 , where S is a state space, A is a finite action space, P ( ds ′ | s, a ), for all state-action pairs ( s, a ), is a probability kernel on S , the reward function r : S × A → R is bounded by R max , and γ ∈ (0 , 1) is a discount factor. A deterministic stationary policy (for short thereafter: a policy) is defined as a mapping π : S → A . For a policy π , we may write r π ( s ) = r ( s, π ( s ) ) and P π ( ds ′ | s ) = P ( ds ′ | s, π ( s ) ) . The value of the policy π in a state s is defined as the expected discounted sum of rewards received by starting at state s and then following the policy π , i.e.,

<!-- formula-not-decoded -->

Similarly, the action-value function of a policy π at a state-action pair ( s, a ), Q π ( s, a ), is the expected discounted sum of rewards received by starting at state s , taking action a , and then following the policy π , i.e.,

<!-- formula-not-decoded -->

Since the rewards are bounded by R max , the values and action-values are bounded by V max = Q max = R max / (1 -γ ).

The Bellman operator T π of policy π takes an integrable function f on S as input and returns the function T π f defined as

For any distribution µ on S , µP π is a distribution given by ( µP π )( ds ′ ) = ∫ P π ( ds ′ | ds ) µ ( ds ). For any integrable function v on S , P π v is a function defined as ( P π v )( s ) = ∫ v ( s ′ ) P π ( ds ′ | s ). The product of two kernels is naturally defined as ( P π ′ P π )( ds ′′ | s ) = ∫ P π ′ ( ds ′′ | s ′ ) P π ( ds ′ | s ). In analogy with the discrete space case, we write ( I -γP π ) -1 to denote the kernel that is defined as ∑ ∞ t =0 ( γP π ) t .

<!-- formula-not-decoded -->

or in compact form, T π f = r π + γP π f . It is known that v π = ( I -γP π ) -1 r π is the unique fixed-point of T π . Given an integrable function f on S , we say that a policy π is greedy w.r.t. f , and write π = G f , if

<!-- formula-not-decoded -->

or equivalently T π f = max π ′ [ T π ′ f ]. We denote by v ∗ the optimal value function. It is also known that v ∗ is the unique fixed-point of the Bellman optimality operator T : v → max π T π v = T G ( v ) v , and that a policy π ∗ that is greedy w.r.t. v ∗ is optimal and its value satisfies v π ∗ = v ∗ .

Wenow define the concentrability coefficients (Munos, 2003, 2007; Munos and Szepesv´ ari, 2008; Farahmand et al., 2010; Scherrer, 2013) that measure the stochasticity of an MDP, and will later appear in our analysis. For any integrable function f : S → R and any distribution µ on S , the µ -weighted L p norm of f is defined as

<!-- formula-not-decoded -->

Given some distributions µ and ρ that will be clear in the context of the paper, for all integers j and q , we shall consider the following Radon-Nikodym derivative based quantities

<!-- formula-not-decoded -->

where π 1 , . . . , π j is any set of policies defined in the MDP, and with the understanding that if ρP π 1 P π 2 · · · P π j is not absolutely continuous with respect to µ , then we take c q ( j ) = ∞ . These coefficients measure the mismatch between some reference measure µ and the distribution ρP π 1 P π 2 · · · P π j obtained by starting the process from distribution ρ and then making j steps according to π 1 , π 2 , ... π j , respectively. Since the bounds we shall derive will be based on these coefficients, they will be informative only if these coefficients are finite. We refer the reader to Munos (2007); Munos and Szepesv´ ari (2008); Farahmand et al. (2010) for more discussion on this topic. In particular, the interested reader may find a simple MDP example for which these coefficients are reasonably small in Munos (2007, Section 5.5 and 7).

## 3. Approximate MPI Algorithms

In this section, we describe three approximate MPI (AMPI) algorithms. These algorithms rely on a function space F to approximate value functions, and in the third algorithm, also on a policy space Π to represent greedy policies. In what follows, we describe the iteration k of these iterative algorithms.

## 3.1 AMPI-V

The first and most natural AMPI algorithm presented in the paper, called AMPI-V, is described in Figure 1. In AMPI-V, we assume that the values v k are represented in a

function space F ⊆ R S . In any state s , the action π k +1 ( s ) that is greedy w.r.t. v k can be estimated as follows:

<!-- formula-not-decoded -->

where for all a ∈ A and 1 ≤ j ≤ M , r ( j ) a and s ( j ) a are samples of rewards and next states when action a is taken in state s . Thus, approximating the greedy action in a state s requires M |A| samples. The algorithm works as follows. We sample N states from a distribution µ on S , and build a rollout set D k = { s ( i ) } N i =1 , s ( i ) ∼ µ . We denote by ̂ µ the empirical distribution corresponding to µ . From each state s ( i ) ∈ D k , we generate a rollout of size m , i.e. , ( s ( i ) , a ( i ) 0 , r ( i ) 0 , s ( i ) 1 , . . . , a ( i ) m -1 , r ( i ) m -1 , s ( i ) m ) , where a ( i ) t is the action suggested by π k +1 in state s ( i ) t , computed using Equation 4, and r ( i ) t and s ( i ) t +1 are sampled reward and next state induced by this choice of action. For each s ( i ) , we then compute a rollout estimate

<!-- formula-not-decoded -->

which is an unbiased estimate of [ ( T π k +1 ) m v k ] ( s ( i ) ). Finally, v k +1 is computed as the best fit in F to these estimates, i.e. , it is a function v ∈ F that minimizes the empirical error

<!-- formula-not-decoded -->

with the goal of minimizing the true error

<!-- formula-not-decoded -->

Each iteration of AMPI-V requires N rollouts of size m , and in each rollout, each of the |A| actions needs M samples to compute Equation 4. This gives a total of Nm ( M |A| +1) transition samples. Note that the fitted value iteration algorithm (Munos and Szepesv´ ari, 2008) is a special case of AMPI-V when m = 1.

## 3.2 AMPI-Q

In AMPI-Q, we replace the value function v : S → R with the action-value function Q : S ×A → R . Figure 2 contains the pseudocode of this algorithm. The Bellman operator for a policy π at a state-action pair ( s, a ) can then be written as

<!-- formula-not-decoded -->

and the greedy operator is defined as

<!-- formula-not-decoded -->

Input: Value function space F , state distribution µ Initialize: Let v 0 ∈ F be an arbitrary value function

for k = 0 , 1 , . . . do

- Perform rollouts:

Construct the rollout set = s ( i ) N , s ( i ) iid

- D k { } i =1 ∼ µ for all states s ( i ) ∈ D k do Perform a rollout (using Equation 4 for each action) ̂ v k +1 ( s ( i ) ) = ∑ m -1 t =0 γ t r ( i ) t + γ m v k ( s ( i ) m ) end for · Approximate value function:

end for v k +1 ∈ argmin v ∈F ̂ L F k ( ̂ µ ; v )

Figure 1: The pseudo-code of the AMPI-V algorithm.

In AMPI-Q, action-value functions Q k are represented in a function space F ⊆ R S×A , and the greedy action w.r.t. Q k at a state s , i.e. , π k +1 ( s ), is computed as

<!-- formula-not-decoded -->

The evaluation step is similar to that of AMPI-V, with the difference that now we work with state-action pairs. We sample N state-action pairs from a distribution µ on S × A and build a rollout set D k = { ( s ( i ) , a ( i ) ) } N i =1 , ( s ( i ) , a ( i ) ) ∼ µ . We denote by ̂ µ the empirical distribution corresponding to µ . For each ( s ( i ) , a ( i ) ) ∈ D k , we generate a rollout of size m , i.e. , ( s ( i ) , a ( i ) , r ( i ) 0 , s ( i ) 1 , a ( i ) 1 , · · · , s ( i ) m , a ( i ) m ) , where the first action is a ( i ) , a ( i ) t for t ≥ 1 is the action suggested by π k +1 in state s ( i ) t computed using Equation 7, and r ( i ) t and s ( i ) t +1 are sampled reward and next state induced by this choice of action. For each ( s ( i ) , a ( i ) ) ∈ D k , we then compute the rollout estimate

<!-- formula-not-decoded -->

which is an unbiased estimate of [ ( T π k +1 ) m Q k ] ( s ( i ) , a ( i ) ). Finally, Q k +1 is the best fit to these estimates in F , i.e. , it is a function Q ∈ F that minimizes the empirical error

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the goal of minimizing the true error

Each iteration of AMPI-Q requires Nm samples, which is less than that for AMPI-V. However, it uses a hypothesis space on state-action pairs instead of states (a larger space than that used by AMPI-V). Note that the fitted-Q iteration algorithm (Ernst et al., 2005; Antos et al., 2007) is a special case of AMPI-Q when m = 1.

(regression) (see Equation 6)

- Input: Value function space F , state distribution µ Initialize: Let Q 0 ∈ F be an arbitrary value function for k = 0 , 1 , . . . do · Perform rollouts: Construct the rollout set D k = { ( s ( i ) , a ( i ) } N i =1 , ( s ( i ) , a ( i ) ) iid ∼ µ for all states ( s ( i ) , a ( i ) ) ∈ D k do Perform a rollout (using Equation 7 for each action) ̂ Q k +1 ( s ( i ) , a ( i ) ) = ∑ m -1 t =0 γ t r ( i ) t + γ m Q k ( s ( i ) m , a ( i ) m ) , end for · Approximate action-value function: Q k +1 ∈ argmin Q ∈F ̂ L F k ( ̂ µ ; Q ) (regression) (see Equation 8) end for

Figure 2: The pseudo-code of the AMPI-Q algorithm.

## 3.3 Classification-Based MPI

The third AMPI algorithm presented in this paper, called classification-based MPI (CBMPI), uses an explicit representation for the policies π k , in addition to the one used for the value functions v k . The idea is similar to the classification-based PI algorithms (Lagoudakis and Parr, 2003b; Fern et al., 2006; Lazaric et al., 2010c; Gabillon et al., 2011) in which we search for the greedy policy in a policy space Π (defined by a classifier) instead of computing it from the estimated value or action-value function (similar to AMPI-V and AMPI-Q).

In order to describe CBMPI, we first rewrite the MPI formulation (Equations 1 and 2) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that in this equivalent formulation both v k and π k +1 are functions of ( T π k ) m v k -1 . CBMPI is an approximate version of this new formulation. As described in Figure 3, CBMPI begins with arbitrary initial policy π 1 ∈ Π and value function v 0 ∈ F . 2 At each iteration k , a new value function v k is built as the best approximation of the m -step Bellman operator ( T π k ) m v k -1 in F ( evaluation step ). This is done by solving a regression problem whose target function is ( T π k ) m v k -1 . To set up the regression problem, we build a rollout set D k by sampling N states i.i.d. from a distribution µ . 3 We denote by ̂ µ the empirical distribution corresponding to µ . For each state s ( i ) ∈ D k , we generate a rollout ( s ( i ) , a ( i ) 0 , r ( i ) 0 , s ( i ) 1 , . . . , a ( i ) m -1 , r ( i ) m -1 , s ( i ) m ) of size m , where a ( i ) t = π k ( s ( i ) t ), and r ( i ) t and s ( i ) t +1 are sampled reward and next state induced by this choice of action. From this rollout, we compute an unbiased estimate v k ( s ( i ) ) of [ ( T π k ) m v k -1 ] ( s ( i ) ) as in Equation 5:

2. Note that the function space F and policy space Π are automatically defined by the choice of the regressor and classifier, respectively.

3. Here we used the same sampling distribution µ for both regressor and classifier, but in general different distributions may be used for these two components of the algorithm.

and use it to build a training set {( s ( i ) , ̂ v k ( s ( i ) ) )} N i =1 . This training set is then used by the regressor to compute v k as an estimate of ( T π k ) m v k -1 . Similar to the AMPI-V algorithm, the regressor here finds a function v ∈ F that minimizes the empirical error

<!-- formula-not-decoded -->

with the goal of minimizing the true error

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The greedy step at iteration k computes the policy π k +1 as the best approximation of G [ ( T π k ) m v k -1 ] by solving a cost-sensitive classification problem. From the definition of a greedy policy, if π = G [ ( T π k ) m v k -1 ] , for each s ∈ S , we have

By defining Q k ( s, a ) = [ T a ( T π k ) m v k -1 ] ( s ), we may rewrite Equation 13 as

The cost-sensitive error function used by CBMPI is of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To simplify the notation we use L Π k instead of L Π π k ,v k -1 . To set up this cost-sensitive classification problem, we build a rollout set D ′ k by sampling N ′ states i.i.d. from a distribution µ . For each state s ( i ) ∈ D ′ k and each action a ∈ A , we build M independent rollouts of size m +1, i.e. , 4

where for t ≥ 1, a ( i,j ) t = π k ( s ( i,j ) t ), and r ( i,j ) t and s ( i,j ) t +1 are sampled reward and next state induced by this choice of action. From these rollouts, we compute an unbiased estimate of Q k ( s ( i ) , a ) as ̂ Q k ( s ( i ) , a ) = 1 M ∑ M j =1 R j k ( s ( i ) , a ) where

<!-- formula-not-decoded -->

Given the outcome of the rollouts, CBMPI uses a cost-sensitive classifier to return a policy π k +1 that minimizes the following empirical error

<!-- formula-not-decoded -->

4. In practice, one may implement CBMPI in more sample-efficient way by reusing the rollouts generated for the greedy step in the evaluation step, but we do not consider this here because it makes the forthcoming analysis more complicated.

Figure 3: The pseudo-code of the CBMPI algorithm.

<!-- image -->

with the goal of minimizing the true error L Π k ( µ ; π ) defined by Equation 15.

Each iteration of CBMPI requires Nm + M |A| N ′ ( m +1) (or M |A| N ′ ( m +1) in case we reuse the rollouts, see Footnote 4) transition samples. Note that when m tends to ∞ , we recover the DPI algorithm proposed and analyzed by Lazaric et al. (2010c).

## 3.4 Possible Approaches to Reuse the Samples

In all the proposed AMPI algorithms, we generate fresh samples for the rollouts, and even for the starting states, at each iteration. This may result in relatively high sample complexity for these algorithms. In this section, we propose two possible approaches to circumvent this problem and to keep the number of samples independent of the number of iterations.

One approach would be to use a fixed set of starting samples ( s ( i ) ) or ( s ( i ) , a ( i ) ) for all iterations, and think of a tree of depth m that contains all the possible outcomes of m -steps choices of actions (this tree contains | A | m leaves). Using this tree, all the trajectories with the same actions share the same samples. In practice, it is not necessarily to build the entire depth m tree, it is only needed to add a branch when the desired action does not belong to the tree. Using this approach, that is reminiscent of the work by Kearns et al. (2000), the sample complexity of the algorithm no longer depends on the number of iterations. For example, we may only need NM | A | m transitions for the CBMPI algorithm.

We may also consider the case where we do not have access to a generative model of the system, and all we have is a set of trajectories of size m generated by a behavior policy π b that is assumed to choose all actions a in each state s with a positive probability ( i.e. , π b ( a | s ) &gt;

0 , ∀ s, ∀ a ) (Precup et al., 2000, 2001; Geist and Scherrer, 2014). In this case, one may still compute an unbiased estimate of the application of ( T π ) m operator to value and actionvalue functions. For instance, given a m -step sample trajectory ( s, a 0 , r 0 , s 1 , . . . , s m , a m ) generated by π b , an unbiased estimate of [( T π ) m v ]( s ) may be computed as (assuming that the distribution µ has the following factored form p ( s, a 0 | µ ) = p ( s ) π b ( a 0 | s ) at state s )

<!-- formula-not-decoded -->

is an importance sampling correction factor that can be computed along the trajectory. Note that this process may increase the variance of such an estimate, and thus, requires many more samples to be accurate-the price to pay for the absence of a generative model.

## 4. Error Propagation

In this section, we derive a general formulation for propagation of errors through the iterations of an AMPI algorithm. The line of analysis for error propagation is different in VI and PI algorithms. VI analysis is based on the fact that this algorithm computes the fixed point of the Bellman optimality operator, and this operator is a γ -contraction in maxnorm (Bertsekas and Tsitsiklis, 1996; Munos, 2007). On the other hand, it can be shown that the operator by which PI updates the value from one iteration to the next is not a contraction in max-norm in general. Unfortunately, we can show that the same property holds for MPI when it does not reduce to VI ( i.e. , for m&gt; 1).

Proposition 1 If m &gt; 1 , there exists no norm for which the operator that MPI uses to update the values from one iteration to the next is a contraction.

Proof We consider the MDP with two states { s 1 , s 2 } , two actions { change, stay } , rewards r ( s 1 ) = 0 , r ( s 2 ) = 1, and transitions P ch ( s 2 | s 1 ) = P ch ( s 1 | s 2 ) = P st ( s 1 | s 1 ) = P st ( s 2 | s 2 ) = 1. Consider two value functions v = ( glyph[epsilon1], 0) and v ′ = (0 , glyph[epsilon1] ) with glyph[epsilon1] &gt; 0. Their corresponding greedy policies are π = ( st, ch ) and π ′ = ( ch, st ), and the next iterates of v and

<!-- formula-not-decoded -->

( ) ( ) Thus, the norm of ( T π ′ ) m v ′ -( T π ) m v can be arbitrarily larger than the norm of v -v ′ as long as m&gt; 1.

<!-- formula-not-decoded -->

glyph[negationslash]

We also know that the analysis of PI usually relies on the fact that the sequence of the generated values is non-decreasing (Bertsekas and Tsitsiklis, 1996; Munos, 2003). Unfortunately, it can be easily shown that for m finite, the value functions generated by MPI may decrease (it suffices to take a very high initial value). It can be seen from what we just described and Proposition 1 that for m = 1 and ∞ , MPI is neither contracting nor non-decreasing, and thus, a new proof is needed for the propagation of errors in this algorithm.

To study error propagation in AMPI, we introduce an abstract algorithmic model that accounts for potential errors. AMPI starts with an arbitrary value v 0 and at each iteration

k ≥ 1 computes the greedy policy w.r.t. v k -1 with some error glyph[epsilon1] ′ k , called the greedy step error . Thus, we write the new policy π k as

<!-- formula-not-decoded -->

Equation 18 means that for any policy π ′ , we have T π ′ v k -1 ≤ T π k v k -1 + glyph[epsilon1] ′ k . AMPI then generates the new value function v k with some error glyph[epsilon1] k , called the evaluation step error

<!-- formula-not-decoded -->

Before showing how these two errors are propagated through the iterations of AMPI, let us first define them in the context of each of the algorithms presented in Section 3 separately.

AMPI-V: The term glyph[epsilon1] k is the error when fitting the value function v k . This error can be further decomposed into two parts: the one related to the approximation power of F and the one due to the finite number of samples/rollouts. The term glyph[epsilon1] ′ k is the error due to using a finite number of samples M for estimating the greedy actions.

AMPI-Q: In this case glyph[epsilon1] ′ k = 0 and glyph[epsilon1] k is the error in fitting the state-action value function Q k .

CBMPI: This algorithm iterates as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Unfortunately, this does not exactly match the model described in Equations 18 and 19. By introducing the auxiliary variable w k ∆ = ( T π k ) m v k -1 , we have v k = w k + glyph[epsilon1] k , and thus, we may write

Using v k -1 = w k -1 + glyph[epsilon1] k -1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, Equations 20 and 21 exactly match Equations 18 and 19 by replacing v k with w k and glyph[epsilon1] k with ( γP π k ) m glyph[epsilon1] k -1 .

The rest of this section is devoted to show how the errors glyph[epsilon1] k and glyph[epsilon1] ′ k propagate through the iterations of an AMPI algorithm. We only outline the main arguments that will lead to the performance bounds of Theorems 7 and 8 and report most technical details of the proof in Appendices A to C. To do this, we follow the line of analysis developed by Scherrer and Thi´ ery (2010), and consider the following three quantities:

- 1) The distance between the optimal value function and the value before approximation at the k th iteration:
- 2) The shift between the value before approximation and the value of the policy at the k th iteration:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 3) The (approximate) Bellman residual at the k th iteration:

<!-- formula-not-decoded -->

We are interested in finding an upper bound on the loss

<!-- formula-not-decoded -->

To do so, we will upper bound d k and s k , which requires a bound on the Bellman residual b k . More precisely, the core of our analysis is to prove the following point-wise inequalities for our three quantities of interest.

Lemma 2 Let k 1 , x k ∆ = ( I γP π k ) glyph[epsilon1] k + glyph[epsilon1] ′ and y k ∆ = γP π ∗ glyph[epsilon1] k + glyph[epsilon1] ′ . We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof See Appendix A.

Since the stochastic kernels are non-negative, the bounds in Lemma 2 indicate that the loss l k will be bounded if the errors glyph[epsilon1] k and glyph[epsilon1] ′ k are controlled. In fact, if we define glyph[epsilon1] as a uniform upper-bound on the pointwise absolute value of the errors, | glyph[epsilon1] k | and | glyph[epsilon1] ′ k | , the first inequality in Lemma 2 implies that b k ≤ O ( glyph[epsilon1] ), and as a result, the second and third inequalities gives us d k ≤ O ( glyph[epsilon1] ) and s k ≤ O ( glyph[epsilon1] ). This means that the loss will also satisfy l k ≤ O ( glyph[epsilon1] ).

Our bound for the loss l k is the result of careful expansion and combination of the three inequalities in Lemma 2. Before we state this result, we introduce some notations that will ease our formulation and significantly simplify our proofs compared to those in the similar existing work (Munos, 2003, 2007; Scherrer, 2013).

Definition 3 For a positive integer n , we define P n as the smallest set of discounted transition kernels that are defined as follows:

- 1) for any set of n policies { π 1 , . . . , π n } , ( γP π 1 )( γP π 2 ) . . . ( γP π n ) ∈ P n ,

2) for any α ∈ (0 , 1) and ( P 1 , P 2 ) ∈ P n × P n , αP 1 +(1 -α ) P 2 ∈ P n . Furthermore, we use the somewhat abusive notation Γ n for denoting any element of P n . For example, if we write a transition kernel P as P = α 1 Γ i + α 2 Γ j Γ k = α 1 Γ i + α 2 Γ j + k , it should be read as: 'there exist P 1 ∈ P i , P 2 ∈ P j , P 3 ∈ P k , and P 4 ∈ P k + j such that P = α 1 P 1 + α 2 P 2 P 3 = α 1 P 1 + α 2 P 4 .'

Using the notation in Definition 3, we now derive a point-wise bound on the loss.

Lemma 4 After k iterations, the losses of AMPI-V and AMPI-Q satisfy

<!-- formula-not-decoded -->

while the loss of CBMPI satisfies

<!-- formula-not-decoded -->

where h ( k ) ∆ = 2 ∑ ∞ j = k Γ j | d 0 | or h ( k ) ∆ = 2 ∑ ∞ j = k Γ j | b 0 | . Proof See Appendix B.

Remark 5 A close look at the existing point-wise error bounds for AVI (Munos, 2007, Lemma 4.1) and API (Munos, 2003, Corollary 10) shows that they do not consider error in the greedy step ( i.e. , glyph[epsilon1] ′ k = 0 ) and have the following form:

<!-- formula-not-decoded -->

This indicates that the bound in Lemma 4 not only unifies the analysis of AVI and API, but it generalizes them to the case of error in the greedy step and to a finite number of iterations k . Moreover, our bound suggests that the way the errors are propagated in the whole family of algorithms, VI/PI/MPI, is independent of m at the level of abstraction suggested by Definition 3. 5

An important immediate consequence of the point-wise bound of Lemma 4 is a simple guarantee on the performance of the algorithms. Let us define glyph[epsilon1] = sup j ≥ 1 ‖ glyph[epsilon1] j ‖ ∞ and glyph[epsilon1] ′ = sup j ≥ 1 ‖ glyph[epsilon1] ′ j ‖ ∞ as uniform bounds on the evaluation and greedy step errors. Now by taking the max-norm (using the fact that for all i , ‖ Γ i ‖ ∞ = γ i ) and limsup when k tends to infinity, we obtain

<!-- formula-not-decoded -->

Such a bound is a generalization of the bounds for AVI ( m = 1 and glyph[epsilon1] ′ = 0) and API ( m = ∞ ) in Bertsekas and Tsitsiklis (1996). This bound can be read as follows: if we can control the max-norm of the evaluation and greedy errors at all iterations, then we can control the loss of the policy returned by the algorithm w.r.t. the optimal policy. Conversely, another interpretation of the above bound is that errors should not be too big if we want to have a performance guarantee. Since the loss is always bounded by 2 V max , the bound stops to be informative as soon as 2 γglyph[epsilon1] + glyph[epsilon1] ′ &gt; 2(1 -γ ) 2 V max = 2(1 -γ ) R max .

Assume we use (max-norm) regression and classification for the evaluation and greedy steps. Then, the above result means that one can make a reduction from the RL problem to these regression and classification problems. Furthermore, if any significant breakthrough is made in the literature for these (more standard problems), the RL setting can automatically benefit from it. The error terms glyph[epsilon1] and glyph[epsilon1] ′ in the above bound are expressed in terms of the

5. Note however that the dependence on m will reappear if we make explicit what is hidden in Γ j terms.

max-norm. Since most regressors and classifiers, including those we have described in the algorithms, control some weighted quadratic norm, the practical range of a result like Equation 22 is limited. The rest of this section addresses this specific issue, by developing a somewhat more complicated but more useful error analysis in L p -norm.

We now turn the point-wise bound of Lemma 4 into a bound in weighted L p -norm, which we recall, for any function f : S → R and any distribution µ on S is defined as ‖ f ‖ p,µ ∆ = [ ∫ | f ( x ) | p µ ( dx ) ] 1 /p . Munos (2003, 2007); Munos and Szepesv´ ari (2008), and the recent work of Farahmand et al. (2010), which provides the most refined bounds for API and AVI, show how to do this process through quantities, called concentrability coefficients . These coefficients use the Radon-Nikodym coefficients introduced in Section 2 and measure how a distribution over states may concentrate through the dynamics of the MDP. We now state a technical lemma that allows to convert componentwise bounds to L p -norm bounds, and that generalizes the analysis of Farahmand et al. (2010) to a larger class of concentrability coefficients.

Lemma 6 Let I and ( J i ) i ∈I be sets of non-negative integers, {I 1 , . . . , I n } be a partition of I , and f and ( g i ) i ∈I be functions satisfying

<!-- formula-not-decoded -->

Then for all p , q and q ′ such that 1 q + 1 q ′ = 1 , and for all distributions ρ and µ , we have

<!-- formula-not-decoded -->

with the following concentrability coefficients

<!-- formula-not-decoded -->

where c q ( j ) is defined by Equation 3.

Proof See Appendix C.

We now derive an L p -norm bound for the loss of the AMPI algorithm by applying Lemma 6 to the point-wise bound of Lemma 4.

Theorem 7 For all q , l , k and d , define the following concentrability coefficients:

<!-- formula-not-decoded -->

with c q ( j ) given by Equation 3. Let ρ and µ be distributions over states. Let p , q , and q ′ be such that 1 q + 1 q ′ = 1 . After k iterations, the loss of AMPI satisfies

<!-- formula-not-decoded -->

while the loss of CBMPI satisfies

<!-- formula-not-decoded -->

Proof We only detail the proof for AMPI, the proof is similar for CBMPI. We define I = { 1 , 2 , . . . , 2 k } and the (trivial) partition I = {I 1 , I 2 , . . . , I 2 k } , where I i = { i } , i ∈ { 1 , . . . , 2 k } . For each i ∈ I , we also define where g ( k ) ∆ = 2 γ k 1 -γ ( C k,k +1 , 0 q ) 1 p min ( ‖ d 0 ‖ pq ′ ,µ , ‖ b 0 ‖ pq ′ ,µ ) .

<!-- formula-not-decoded -->

With the above definitions and the fact that the loss l k is non-negative, Lemma 4 may be rewritten as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The result follows by applying Lemma 6 and noticing that ∑ k -1 i = i 0 ∑ ∞ j = i γ j = γ i 0 -γ k (1 -γ ) 2 .

Similar to the results of Farahmand et al. (2010), this bound shows that the last iterations have the highest influence on the loss and the influence decreases at the exponential rate γ towards the initial iterations. This phenomenon is related to the fact that the DP algorithms progressively forget about the past iterations. This is similar to the fact that exact VI and PI converge to the optimal limit independently of their initialization.

We can group the terms differently and derive an alternative L p -norm bound for the loss of AMPI and CBMPI. This also shows the flexibility of Lemma 6 for turning the point-wise bound of Lemma 4 into L p -norm bounds.

Theorem 8 With the notations of Theorem 7, and writing glyph[epsilon1] = sup 1 ≤ j ≤ k -1 ‖ glyph[epsilon1] j ‖ pq ′ ,µ and glyph[epsilon1] ′ = sup 1 ≤ j ≤ k ‖ glyph[epsilon1] ′ j ‖ pq ′ ,µ , the loss of AMPI satisfies

<!-- formula-not-decoded -->

while the loss of CBMPI satisfies

<!-- formula-not-decoded -->

Proof We only give the details of the proof for AMPI, the proof is similar for CBMPI. Defining I = { 1 , 2 , · · · , 2 k } and g i as in the proof of Theorem 7, we now consider the partition I = {I 1 , I 2 , I 3 } as I 1 = { 1 , . . . , k -1 } , I 2 = { k, . . . , 2 k -1 } , and I 3 = { 2 k } , where for each i ∈ I

The proof ends similar to that of Theorem 7.

<!-- formula-not-decoded -->

By sending the iteration number k to infinity, one obtains the following bound for AMPI:

<!-- formula-not-decoded -->

Compared to the simple max-norm bound of Equation 22, we can see that the price that we must pay to have an error bound in L p -norm is the appearance of the concentrability coefficients C 1 , ∞ , 0 q and C 0 , ∞ , 0 q . Furthermore, it is easy to see that the above bound is more general, i.e., by sending p to infinity, we recover the max-norm bound of Equation 22.

Remark 9 We can balance the influence of the concentrability coefficients (the bigger the q , the higher the influence) and the difficulty of controlling the errors (the bigger the q ′ , the greater the difficulty in controlling the L pq ′ -norms) by tuning the parameters q and q ′ , given that 1 q + 1 q ′ = 1 . This potential leverage is an improvement over the existing bounds and concentrability results that only consider specific values of these two parameters: q = ∞ and q ′ = 1 in Munos (2007) and Munos and Szepesv´ ari (2008), and q = q ′ = 2 in Farahmand et al. (2010).

Remark 10 It is important to note that our loss bound for AMPI does not 'directly' depend on m (although as we will discuss in the next section, it 'indirectly' does through glyph[epsilon1] k ). For CBMPI, the parameter m controls the influence of the value function approximator, cancelling it out in the limit when m tends to infinity (see Equation 24). Assuming a fixed budget of sample transitions, increasing m reduces the number of rollouts used by the classifier, and thus, worsens its quality. In such a situation, m allows making a trade-off between the estimation error of the classifier and the overall value function approximation.

The arguments we developed globally follow those originally developed for λ -policy iteration (Scherrer, 2013). With respect to that work, our proof is significantly simpler thanks to the use of the Γ n notation (Definition 3) and the fact that the AMPI scheme is itself much simpler than λ -policy iteration. Moreover, the results are deeper since we consider a possible error in the greedy step and more general concentration coefficients. Canbolat and Rothblum (2012) recently (and independently) developed an analysis of an approximate form of MPI. While Canbolat and Rothblum (2012) only consider the error in the greedy step, our work is more general since it takes into account both this error and the error in the value update. Note that it is required to consider both sources of error for the analysis of CBMPI. Moreover, Canbolat and Rothblum (2012) provide bounds when the errors are

controlled in max-norm, while we consider the more general L p -norm. At a more technical level, Theorem 2 in Canbolat and Rothblum (2012) bounds the norm of the distance v ∗ -v k , while we bound the loss v ∗ -v π k . Finally, if we derive a bound on the loss (using e.g. , Theorem 1 in Canbolat and Rothblum 2012), this leads to a bound on the loss that is looser than ours. In particular, this does not allow to recover the standard bounds for AVI and API, as we may obtain here (in Equation 22).

The results that we just stated (Theorem 7 and 8) can be read as follows: if one can control the errors glyph[epsilon1] k and glyph[epsilon1] ′ k in L p -norm, then the performance loss is also controlled. The main limitation of this result is that in general, even if there is no sampling noise ( i.e. , N = ∞ for all the algorithms and M = ∞ for AMPI-V), the error glyph[epsilon1] k of the evaluation step may grow arbitrarily and make the algorithm diverge. The fundamental reason is that the composition of the approximation and the Bellman operator T π is not necessarily contracting. Since the former is contracting with respect to the µ -norm, another reason for this issue is that T π is in general not contracting for that norm. A simple well-known pathological example is due to Tsitsiklis and Van Roy (1997) and involves a two-state uncontrolled MDP and a linear projection onto a 1-dimensional space (that contains the real value function). Increasing the parameter m of the algorithm makes the operator ( T π ) m used in Equation 19 more contracting and can in principle address this issue. For instance, if we consider that we have a state space of finite size |S| , and take the uniform distribution µ , it can be easily seen that for any v and v ′ , we have

<!-- formula-not-decoded -->

In other words, ( T π ) m is contracting w.r.t. the µ -weighted norm as soon as m &gt; log |S| 2 log 1 γ . In particular, it is sufficient for m to be exponentially smaller than the size of the state space in order to solve this potential divergence problem.

## 5. Finite-Sample Analysis of the Algorithms

In this section, we first show how the error terms glyph[epsilon1] k and glyph[epsilon1] ′ k appeared in Theorem 8 (Equations 23 and 24) can be bounded in each of the three proposed algorithms, and then use the obtained results and derive finite-sample performance bounds for these algorithms. We first bound the evaluation step error glyph[epsilon1] k . In AMPI-V and CBMPI, the evaluation step at each iteration k is a regression problem with the target ( T π k ) m v k -1 and a training set of the form {( s ( i ) , ̂ v k ( s ( i ) ) )} N i =1 in which the states s ( i ) are i.i.d. samples from the distribution µ and ̂ v k ( s ( i ) )'s are unbiased estimates of the target computed using Equation 5. The situation is the same for AMPI-Q, except everything is in terms of action-value function Q k instead of value function v k . Therefore, in the following we only show how to bound glyph[epsilon1] k in AMPI-V and CBMPI, the extension to AMPI-Q is straightforward.

We may use linear or non-linear function space F to approximate ( T π k ) m v k -1 . Here we consider a linear architecture with parameters α ∈ R d and bounded (by L ) basis functions { ϕ j } d j =1 , ‖ ϕ j ‖ ∞ ≤ L . We denote by φ : X → R d , φ ( · ) = ( ϕ 1 ( · ) , . . . , ϕ d ( · ) ) glyph[latticetop] the feature

vector, and by F the linear function space spanned by the features ϕ j , i.e. , F = { f α ( · ) = φ ( · ) glyph[latticetop] α : α ∈ R d } . Now if we define v k as the truncation (by V max ) of the solution of the above linear regression problem, we may bound the evaluation step error glyph[epsilon1] k using the following lemma.

Lemma 11 (Evaluation step error) Consider the linear regression setting described above, then we have

<!-- formula-not-decoded -->

with probability at least 1 -δ , where and α ∗ is such that f α ∗ is the best approximation (w.r.t. µ ) of the target function ( T π k ) m v k -1 in F .

<!-- formula-not-decoded -->

Proof See Appendix D.

After we showed how to bound the evaluation step error glyph[epsilon1] k for the proposed algorithms, we now turn our attention to bounding the greedy step error glyph[epsilon1] ′ k , that contrary to the evaluation step error, varies more significantly across the algorithms. While the greedy step error equals to zero in AMPI-Q, it is based on sampling in AMPI-V, and depends on a classifier in CBMPI. To bound the greedy step error in AMPI-V and CBMPI, we assume that the action space A contains only two actions, i.e., |A| = 2. The extension to more than two actions is straightforward along the same line of analysis as in Section 6 of Lazaric et al. (2010a). The main difference w.r.t. the two action case is that the VC-dimension of the policy space is replaced with its Natarajan dimension. We begin with AMPI-V.

Lemma 12 (Greedy step error of AMPI-V) Let µ be a distribution over the state space S and N be the number of states in the rollout set D k drawn i.i.d. from µ . For each state s ∈ D k and each action a ∈ A , we sample M states resulted from taking action a in state s . Let h be the VC-dimension of the policy space obtained by Equation 4 from the truncation (by V max ) of the function space F . For any δ &gt; 0 , the greedy step error glyph[epsilon1] ′ k in the AMPI-V algorithm is bounded as

<!-- formula-not-decoded -->

with probability at least 1 -δ , with

<!-- formula-not-decoded -->

Proof See Appendix E.

We now show how to bound glyph[epsilon1] ′ k in CBMPI. From the definitions of glyph[epsilon1] ′ k (Equation 20) and L Π k ( µ ; π ) (Equation 15), it is easy to see that ‖ glyph[epsilon1] ′ k ‖ 1 ,µ = L Π k -1 ( µ ; π k ). This is because

<!-- formula-not-decoded -->

Lemma 13 (Greedy step error of CBMPI) Let the policy space Π defined by the classifier have finite VC-dimension h = V C (Π) &lt; ∞ , and µ be a distribution over the state space S . Let N ′ be the number of states in D ′ k -1 drawn i.i.d. from µ , M be the number of rollouts per state-action pair used in the estimation of ̂ Q k -1 , and π k = argmin π ∈ Π ̂ L Π k -1 ( ̂ µ, π ) be the policy computed at iteration k -1 of CBMPI. Then, for any δ &gt; 0 , we have with probability at least 1 -δ , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof See Appendix F.

From Lemma 11, we have a bound on ‖ glyph[epsilon1] k ‖ 2 ,µ for all the three algorithms. Since ‖ glyph[epsilon1] k ‖ 1 ,µ ≤ ‖ glyph[epsilon1] k ‖ 2 ,µ , we also have a bound on ‖ glyph[epsilon1] k ‖ 1 ,µ for all the algorithms. On the other hand, from Lemmas 12 and 13, we have a bound on ‖ glyph[epsilon1] ′ k ‖ 1 ,µ for the AMPI-V and CMBPI algorithms. This means that for AMPI-V, AMPI-Q ( glyph[epsilon1] ′ k = 0 for this algorithm), and CBMPI, we can control the right hand side of Equations 23 and 24 in L 1 -norm, which in the context of Theorem 8 means p = 1, q ′ = 1, and q = ∞ . This leads to the main result of this section, finite-sample performance bounds for the three proposed algorithms.

## Theorem 14 Let

<!-- formula-not-decoded -->

where F is the function space used by the algorithms and Π is the policy space used by CBMPI with the VC-dimension h . With the notations of Theorem 8 and Lemmas 11-13, after k iterations, and with probability 1 -δ , the expected losses E ρ [ l k ] = ‖ l k ‖ 1 ,ρ of the proposed AMPI algorithms satisfy: 6

6. Note that the bounds of AMPI-V and AMPI-Q may also be written with ( p = 2 , q ′ = 1 , q = ∞ ), and ( p = 1 , q ′ = 2 , q = 2).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 15 Assume that we run AMPI-Q with a total fixed budget B that is equally divided between the K iterations. 7 Recall from Theorem 8 that g ( k ) = γ k C k,k +1 , 0 q C 0 , where C 0 = min ( ‖ d 0 ‖ pq ′ ,µ , ‖ b 0 ‖ pq ′ ,µ ) ≤ V max measures the quality of the initial value/policy pair. Then, up to constants and logarithmic factors, one can see that the bound has the form

<!-- formula-not-decoded -->

We deduce that the best choice for the number of iterations K can be obtained as a compromise between the quality of the initial value/policy pair and the estimation errors of the value estimation step.

Remark 16 The CBMPI bound in Theorem 14 allows to turn the qualitative Remark 10 into a quantitative one. Assume that we have a fixed budget per iteration B = Nm + N ′ M |A| ( m + 1) that is equally divided over the classifier and regressor. Note that the budget is measured in terms of the number of calls to the generative model. Then up to constants and logarithmic factors, the bound has the form

<!-- formula-not-decoded -->

This shows a trade-off in tuning the parameter m : a large value of m makes the influence (in the final error) of the regressor's error (both approximation and estimation errors) smaller, and at the same time the influence of the estimation error of the classifier larger.

## 6. Experimental Results

The main objective of this section is to present experiments for the new algorithm that we think is the most interesting, CBMPI, but we shall also illustrate AMPI-Q (we do not

7. Similar reasoning can be done for AMPI-V and CBMPI, we selected AMPI-Q for the sake of simplicity. Furthermore, one could easily relax the assumption that the budget is equally divided by using Theorem 7.

illustrate AMPI-V that is close to AMPI-Q but significantly less efficient to implement). We consider two different domains: 1) the mountain car problem and 2) the more challenging game of Tetris . In several experiments, we compare the performance of CBMPI with the DPI algorithm (Lazaric et al., 2010c), which is basically CBMPI without value function approximation. 8 Note that comparing DPI and CBMPI allows us to highlight the role of the value function approximation.

As discussed in Remark 10, the parameter m in CBMPI balances between the errors in evaluating the value function and the policy. The value function approximation error tends to zero for large values of m . Although this would suggest to have large values for m , as mentioned in Remark 16, the size of the rollout sets D and D ′ would correspondingly decreases as N = O ( B/m ) and N ′ = O ( B/m ), thus decreasing the accuracy of both the regressor and classifier. This leads to a trade-off between long rollouts and the number of states in the rollout sets. The solution to this trade-off strictly depends on the capacity of the value function space F . A rich value function space would lead to solve the trade-off for small values of m . On the other hand, when the value function space is poor, or, as in the case of DPI, when there is no value function, m should be selected in a way to guarantee large enough rollout sets (parameters N and N ′ ), and at the same time, a sufficient number of rollouts (parameter M ).

One of the objectives of our experiments is to show the role of these parameters in the performance of CBMPI. However, since we almost always obtained our best results with M = 1, we only focus on the parameters m and N in our experiments. Moreover, as mentioned in Footnote 3, we implement a more sample-efficient version of CBMPI by reusing the rollouts generated for the classifier in the regressor. More precisely, at each iteration k , for each state s ( i ) ∈ D ′ k and each action a ∈ A , we generate one rollout of length m +1, i.e., ( s ( i ) , a, r ( i ) 0 , s ( i ) 1 , a ( i ) 1 , . . . , a ( i ) m , r ( i ) m , s ( i ) m +1 ) . We then take the rollout of action π k ( s ( i ) ), select its last m steps, i.e., ( s ( i ) 1 , a ( i ) 1 , . . . , a ( i ) m , r ( i ) m , s ( i ) m +1 ) (note that all the actions here have been taken according to the current policy π k ), use it to estimate the value function ̂ v k ( s ( i ) 1 ), and add it to the training set of the regressor. This process guarantees to have N = N ′ .

In each experiment, we run the algorithms with the same budget B per iteration. The budget B is the number of next state samples generated by the generative model of the system at each iteration. In DPI and CBMPI, we generate a rollout of length m + 1 for each state in D ′ and each action in A , so, B = ( m +1) N |A| . In AMPI-Q, we generate one rollout of length m for each state-action pair in D , and thus, B = mN .

## 6.1 Mountain Car

Mountain Car (MC) is the problem of driving a car up to the top of a one-dimensional hill (see Figure 4). The car is not powerful enough to accelerate directly up the hill, and thus, it must learn to oscillate back and forth to build up enough inertia. There are three possible actions: forward (+1), reverse ( -1), and stay (0). The reward is -1 for all the states but the goal state at the top of the hill, where the episode ends with a reward 0. The discount factor is set to γ = 0 . 99. Each state s consists of the pair ( x s , ˙ x s ), where x s is the

8. DPI, as it is presented by Lazaric et al. (2010c), uses infinitely long rollouts and is thus equivalent to CBMPI with m = ∞ . In practice, implementations of DPI use rollouts that are truncated after some horizon H , and is then equivalent to CBMPI with m = H and v k = 0 for all the iterations k .

Figure 4: (Left) The Mountain Car (MC) problem in which the car needs to learn to oscillate back and forth in order to build up enough inertia to reach the top of the one-dimensional hill. (Right) A screen-shot of the game of Tetris and the seven pieces (shapes) used in the game.

<!-- image -->

position of the car and ˙ x s is its velocity. We use the formulation described in Dimitrakakis and Lagoudakis (2008) with uniform noise in [ -0 . 2 , 0 . 2] added to the actions.

In this section, we report the empirical evaluation of CBMPI and AMPI-Q and compare it to DPI and LSPI (Lagoudakis and Parr, 2003a) in the MC problem. In our experiments, we show that CBMPI, by combining policy and value function approximation, can improve over AMPI-Q, DPI, and LSPI.

## 6.1.1 Experimental Setup

The value function is approximated using a linear space spanned by a set of radial basis functions (RBFs) evenly distributed over the state space. More precisely, we uniformly divide the 2-dimensional state space into a number of regions and place a Gaussian function at the center of each of them. We set the standard deviation of the Gaussian functions to the width of a region. The function space to approximate the action-value function in LSPI is obtained by replicating the state-features for each action. We run LSPI off-policy (i.e., samples are collected once and reused through the iterations of the algorithm).

The policy space Π (classifier) is defined by a regularized support vector classifier (CSVC) using the LIBSVM implementation by Chang and Lin (2011). We use the RBF kernel exp( -| u -v | 2 ) and set the cost parameter C = 1000. We minimize the classification error instead of directly solving the cost-sensitive multi-class classification step as in Figure 3. In fact, the classification error is an upper-bound on the empirical error defined by Equation 17. Finally, the rollout set is sampled uniformly over the state space.

In our MC experiments, the policies learned by the algorithms are evaluated by the number of steps-to-go (average number of steps to reach the goal with a maximum of 300) averaged over 4 , 000 independent trials. More precisely, we define the possible starting configurations (positions and velocities) of the car by placing a 20 × 20 uniform grid over the state space, and run the policy 6 times from each possible initial configuration. The performance of each algorithm is represented by a learning curve whose value at each iteration is the average number of steps-to-go of the policies learned by the algorithm at that iteration in 1 , 000 separate runs of the algorithm.

We tested the performance of DPI, CBMPI, and AMPI-Q on a wide range of parameters ( m,M,N ), but only report their performance for the best choice of M (as mentioned earlier, M = 1 was the best choice in all the experiments) and different values of m .

## 6.1.2 Experimental Results

<!-- image -->

- (a) Performance of DPI (for different values of m and LSPI.
- ) (b) Performance of CBMPI for different values of m .
- (c) Performance of AMPI-Q for different values of m .

<!-- image -->

Figure 5: Performance of the policies learned by (a) DPI and LSPI, (b) CBMPI, and (c) AMPI-Q algorithms in the Mountain Car (MC) problem, when we use a 3 × 3 RBF grid to approximate the value function. The results are averaged over 1 , 000 runs. The total budget B is set to 4 , 000 per iteration.

Figure 5 shows the learning curves of DPI, CBMPI, AMPI-Q, and LSPI algorithms with budget B = 4 , 000 per iteration and the function space F composed of a 3 × 3 RBF grid.

We notice from the results that this space is rich enough to provide a good approximation for the value function components (e.g., in CBMPI, for ( T π ) m v k -1 defined by Equation 19). Therefore, LSPI and DPI obtain the best and worst results with about 50 and 160 steps to reach the goal, respectively. The best DPI results are obtained with the large value of m = 20. DPI performs better for large values of m because the reward function is constant everywhere except at the goal, and thus, a DPI rollout is only informative if it reaches the goal. We also report the performance of CBMPI and AMPI-Q for different values of m . The value function approximation is very accurate, and thus, CBMPI and AMPI-Q achieve performance similar to LSPI for m &lt; 20. However when m is large ( m = 20), the performance of these algorithms is worse, because in this case, the rollout set does not have enough elements ( N small) to learn the greedy policy and value function well. Note that as we increase m (up to m = 10), CBMPI and AMPI-Q converge faster to a good policy.

<!-- image -->

- (a) Performance of CBMPI (for different values of m ) and LSPI.

Rollout size m of AMPI-Q

1

6

2

20

30

Iterations

- (b) Performance of AMPI-Q for different values of m .

Figure 6: Performance of the policies learned by (a) CBMPI and LSPI and (b) AMPI-Q algorithms in the Mountain Car (MC) problem, when we use a 2 × 2 RBF grid to approximate the value function. The results are averaged over 1 , 000 runs. The total budget B is set to 4 , 000 per iteration.

Although this experiment shows that the use of a critic in CBMPI compensates for the truncation of the rollouts (CBMPI performs better than DPI), most of this advantage is due to the richness of the function space F (LSPI and AMPI-Q perform as well as CBMPILSPI even converges faster). Therefore, it seems that it would be more efficient to use LSPI instead of CBMPI in this case.

In the next experiment, we study the performance of these algorithms when the function space F is less rich, composed of a 2 × 2 RBF grid. The results are reported in Figure 6. Now, the performance of LSPI and AMPI-Q (for the best value of m = 1) degrades to 75 and 70 steps, respectively. Although F is not rich, it still helps CBMPI to outperform DPI. We notice the effect of (a weaker) F in CBMPI when we observe that it no longer converges to its best performance (about 50 steps) for small values of m = 1 and m = 2. Note that

10

Averaged steps to the goal

200

150

100

50

0

0

10

40

20

50

CMBPI outperforms all the other algorithms for m = 10 (and even for m = 6), while still has a sub-optimal performance for m = 20, mainly due to the fact that the rollout set would be too small in this case.

## 6.2 Tetris

Tetris is a popular video game created by Alexey Pajitnov in 1985. The game is played on a grid originally composed of 20 rows and 10 columns, where pieces of 7 different shapes fall from the top (see Figure 4). The player has to choose where to place each falling piece by moving it horizontally and rotating it. When a row is filled, it is removed and all the cells above it move one line down. The goal is to remove as many rows as possible before the game is over, i.e., when there is no space available at the top of the grid for the new piece. This game constitutes an interesting optimization benchmark in which the goal is to find a controller (policy) that maximizes the average (over multiple games) number of lines removed in a game (score). 9 This optimization problem is known to be computationally hard. It contains a huge number of board configurations (about 2 200 glyph[similarequal] 1 . 6 × 10 60 ), and even in the case that the sequence of pieces is known in advance, finding the strategy to maximize the score is a NP hard problem (Demaine et al., 2003). Here, we consider the variation of the game in which the player only knows the current falling piece and none of the next several coming pieces.

Approximate dynamic programming (ADP) and reinforcement learning (RL) algorithms including approximate value iteration (Tsitsiklis and Van Roy, 1996), λ -policy iteration ( λ -PI) (Bertsekas and Ioffe, 1996; Scherrer, 2013), linear programming (Farias and Van Roy, 2006), and natural policy gradient (Kakade, 2002; Furmston and Barber, 2012) have been applied to this very setting. These methods formulate Tetris as a MDP (with discount factor γ = 1) in which the state is defined by the current board configuration plus the falling piece, the actions are the possible orientations of the piece and the possible locations that it can be placed on the board, 10 and the reward is defined such that maximizing the expected sum of rewards from each state coincides with maximizing the score from that state. Since the state space is large in Tetris, these methods use value function approximation schemes (often linear approximation) and try to tune the value function parameters (weights) from game simulations. Despite a long history, ADP/RL algorithms, that have been (almost) entirely based on approximating the value function, have not been successful in finding good policies in Tetris. On the other hand, methods that search directly in the space of policies by learning the policy parameters using black-box optimization, such as the cross entropy (CE) method (Rubinstein and Kroese, 2004), have achieved the best reported results in this game (see e.g., Szita and L˝ orincz 2006; Thiery and Scherrer 2009b). This makes us conjecture that Tetris is a game in which good policies are easier to represent, and thus to learn, than their corresponding value functions. So, in order to obtain a good performance with ADP in Tetris, we should use those ADP algorithms that search in a policy space, like CBMPI and DPI, instead of the more traditional ones that search in a value function space.

9. Note that this number is finite because it was shown that Tetris is a game that ends with probability one (Burgiel, 1997).

10. The total number of actions at a state depends on the shape of the falling piece, with the maximum of 32 actions in a state, i.e., |A| ≤ 32.

In this section, we evaluate the performance of CBMPI in Tetris and compare it with DPI, λ -PI, and CE. In these experiments, we show that CBMPI improves over all the previously reported ADP results. Moreover, it obtains the best results reported in the literature for Tetris in both small 10 × 10 and large 10 × 20 boards. Although the CBMPI's results are similar to those achieved by the CE method in the large board, it uses considerably fewer samples (call to the generative model of the game) than CE.

## 6.2.1 Experimental Setup

In this section, we briefly describe the algorithms used in our experiments: the cross entropy (CE) method, our particular implementation of CBMPI, and its slight variation DPI. We refer the readers to Scherrer (2013) for λ -PI. We begin by defining some terms and notations. A state s in Tetris consists of two components: the description of the board b and the type of the falling piece p . All controllers rely on an evaluation function that gives a value to each possible action at a given state. Then, the controller chooses the action with the highest value. In ADP, algorithms aim at tuning the weights such that the evaluation function approximates well the value function, which coincides with the optimal expected future score from each state. Since the total number of states is large in Tetris, the evaluation function f is usually defined as a linear combination of a set of features φ , i.e., f ( · ) = φ ( · ) glyph[latticetop] θ . Alternatively, we can think of the parameter vector θ as a policy (controller) whose performance is specified by the corresponding evaluation function f ( · ) = φ ( · ) glyph[latticetop] θ . The features used in Tetris for a state-action pair ( s, a ) may depend on the description of the board b ′ resulting from taking action a in state s , e.g., the maximum height of b ′ . Computing such features requires to exploit the knowledge of the game's dynamics (this dynamics is indeed known for tetris). We consider the following sets of features, plus a constant offset feature: 11

- (i) Bertsekas Features: First introduced by Bertsekas and Tsitsiklis (1996), this set of 22 features has been mainly used in the ADP/RL community and consists of: the number of holes in the board , the height of each column , the difference in height between two consecutive columns , and the maximum height of the board .
- (ii) Dellacherie-Thiery (D-T) Features: This set consists of the six features of Dellacherie (Fahey, 2003), i.e., the landing height of the falling piece , the number of eroded piece cells , the row transitions , the column transitions , the number of holes , and the number of board wells ; plus 3 additional features proposed in Thiery and Scherrer (2009b), i.e., the hole depth , the number of rows with holes , and the pattern diversity feature . Note that the best policies reported in the literature have been learned using this set of features.
- (iii) RBF Height Features: These new 5 features are defined as exp( -| c -ih/ 4 | 2 2( h/ 5) 2 ) , i = 0 , . . . , 4, where c is the average height of the columns and h = 10 or 20 is the total number of rows in the board.

11. For a precise definition of the features, see Thiery and Scherrer (2009a) or the documentation of their code (Thiery and Scherrer, 2010b). Note that the constant offset feature only plays a role in value function approximation, and has no effect in modeling polices.

```
Input: parameter space Θ, number of parameter vectors n , proportion ζ ≤ 1, noise η Initialize: Set the mean and variance parameters µ = (0 , 0 , . . . , 0) and σ 2 = (100 , 100 , . . . , 100) for k = 1 , 2 , . . . do Generate a random sample of n parameter vectors { θ i } n i =1 ∼ N ( µ , diag( σ 2 )) For each θ i , play G games and calculate the average number of rows removed (score) by the controller Select glyph[floorleft] ζn glyph[floorright] parameters with the highest score θ ′ 1 , . . . , θ ′ glyph[floorleft] ζn glyph[floorright] Update µ and σ : µ ( j ) = 1 glyph[floorleft] ζn glyph[floorright] ∑ glyph[floorleft] ζn glyph[floorright] i =1 θ ′ i ( j ) and σ 2 ( j ) = 1 glyph[floorleft] ζn glyph[floorright] ∑ glyph[floorleft] ζn glyph[floorright] i =1 [ θ ′ i ( j ) -µ ( j )] 2 + η end for
```

Figure 7: The pseudo-code of the cross-entropy (CE) method used in our experiments.

The Cross Entropy (CE) Method: CE (Rubinstein and Kroese, 2004) is an iterative method whose goal is to optimize a function f parameterized by a vector θ ∈ Θ by direct search in the parameter space Θ. Figure 7 contains the pseudo-code of the CE algorithm used in our experiments (Szita and L˝ orincz, 2006; Thiery and Scherrer, 2009b). At each iteration k , we sample n parameter vectors { θ i } n i =1 from a multivariate Gaussian distribution with diagonal covariance matrix N ( µ , diag( σ 2 )). At the beginning, the parameters of this Gaussian have been set to cover a wide region of Θ. For each parameter θ i , we play G games and calculate the average number of rows removed by this controller (an estimate of the expected score). We then select glyph[floorleft] ζn glyph[floorright] of these parameters with the highest score, θ ′ 1 , . . . , θ ′ glyph[floorleft] ζn glyph[floorright] , and use them to update the mean µ and variance diag( σ 2 ) of the Gaussian distribution, as shown in Figure 7. This updated Gaussian is used to sample the n parameters at the next iteration. The goal of this update is to sample more parameters from the promising parts of Θ at the next iteration, and hopefully converge to a good maximum of f . In our experiments, in the pseudo-code of Figure 7, we set ζ = 0 . 1 and η = 4, the best parameters reported in Thiery and Scherrer (2009b). We also set n = 1 , 000 and G = 10 in the small board (10 × 10) and n = 100 and G = 1 in the large board (10 × 20).

Our Implementation of CBMPI (DPI): We use the algorithm whose pseudo-code is shown in Figure 3. We sampled states from the trajectories generated by a good policy for Tetris, namely the DU controller obtained by Thiery and Scherrer (2009b). Since this policy is good, this set is biased towards boards with small height. The rollout set is then obtained by subsampling this set so that the board height distribution is more uniform. We noticed from our experiments that this subsampling significantly improves the performance. We now describe how we implement the regressor and the classifier.

- Regressor: Weuse linear function approximation for the value function, i.e., ̂ v k ( s ( i ) ) = φ ( s ( i ) ) α , where φ ( · ) and α are the feature and weight vectors, and minimize the empirical error L F k ( µ ; v ) using the standard least-squares method.
- ̂ ̂ · Classifier: The training set of the classifier is of size N with s ( i ) ∈ D ′ k as input and ( max a ̂ Q k ( s ( i ) , a ) -̂ Q k ( s ( i ) , a 1 ) , . . . , max a ̂ Q k ( s ( i ) , a ) -̂ Q k ( s ( i ) , a |A| ) ) as output. We use the policies of the form π β ( s ) = argmax a ψ ( s, a ) glyph[latticetop] β , where ψ is the policy

feature vector (possibly different from the value function feature vector φ ) and β ∈ B is the policy parameter vector. We compute the next policy π k +1 by minimizing the empirical error ̂ L Π k ( ̂ µ ; π β ), defined by (17), using the covariance matrix adaptation evolution strategy (CMA-ES) algorithm (Hansen and Ostermeier, 2001). In order to evaluate a policy β ∈ B in CMA-ES, we only need to compute ̂ L Π k ( ̂ µ ; π β ), and given the training set, this procedure does not require further simulation of the game.

We set the initial value function parameter to α = (0 , 0 , . . . , 0) and select the initial policy π 1 (policy parameter β ) randomly. We also set the CMA-ES parameters (classifier parameters) to ζ = 0 . 5, η = 0, and n equal to 15 times the number of features.

## 6.2.2 Experiments

In our Tetris experiments, the policies learned by the algorithms are evaluated by their score (average number of rows removed in a game started with an empty board) averaged over 200 games in the small 10 × 10 board and over 20 games in the large 10 × 20 board (since the game takes much more time to complete in the large board). The performance of each algorithm is represented by a learning curve whose value at each iteration is the average score of the policies learned by the algorithm at that iteration in 100 separate runs of the algorithm. The curves are wrapped in their confidence intervals that are computed as three time the standard deviation of the estimation of the performance at each iteration. In addition to their score, we also evaluate the algorithms by the number of samples they use. In particular, we show that CBMPI/DPI use 6 times fewer samples than CE in the large board. As discussed in Section 6.2.1, this is due the fact that although the classifier in CBMPI/DPI uses a direct search in the space of policies (for the greedy policy), it evaluates each candidate policy using the empirical error of Equation 17, and thus, does not require any simulation of the game (other than those used to estimate the ̂ Q k 's in its training set). In fact, the budget B of CBMPI/DPI is fixed in advance by the number of rollouts NM and the rollout's length m as B = ( m +1) NM |A| . On the contrary, CE evaluates a candidate policy by playing several games, a process that can be extremely costly (sample-wise), especially for good policies in the large board.

We first run the algorithms on the small board to study the role of their parameters and to select the best features and parameters, and then use the selected features and parameters and apply the algorithms to the large board. Finally, we compare the best policies found in our experiments with the best controllers reported in the literature (Tables 1 and 2).

## 6.2.2.1 Small ( 10 × 10 ) Board

Here we run the algorithms with two different feature sets: Dellacherie-Thiery (D-T) and Bertsekas , and report their results.

D-T Features: Figure 8 shows the learning curves of CE, λ -PI, DPI, and CBMPI. Here we use the D-T features for the evaluation function in CE, the policy in DPI and CBMPI, and the value function in λ -PI (in the last case we also add the constant offset feature). For the value function of CBMPI, we tried different choices of features and 'D-T plus the 5 RBF features and constant offset' achieved the best performance (see Figure 8(d)). The budget

<!-- image -->

- (c) DPI with budget B = 8 , 000 , 000 per iteration and m = { 1 , 2 , 5 , 10 , 20 } .

(d) CBMPI with budget and

m

=

{

1

,

2

,

5

,

10

,

20

}

.

Figure 8: Learning curves of CE, λ -PI, DPI, and CBMPI using the 9 Dellacherie-Thiery (D-T) features on the small 10 × 10 board. The results are averaged over 100 runs of the algorithms.

of CBMPI and DPI is set to B = 8 , 000 , 000 per iteration. The CE method reaches the score 3 , 000 after 10 iterations using an average budget B = 65 , 000 , 000. λ -PI with the best value of λ only manages to score 400. In Figure 8(c), we report the performance of DPI for different values of m . DPI achieves its best performance for m = 5 and m = 10 by removing 3 , 400 lines on average. As explained in Section 6.1, having short rollouts ( m = 1) in DPI leads to poor action-value estimates ̂ Q , while having too long rollouts ( m = 20) decreases the size of the training set of the classifier N . CBMPI outperforms the other algorithms, including CE, by reaching the score of 4 , 200 for m = 5. This value of m = 5 corresponds to N = 8000000 (5+1) × 32 ≈ 42 , 000. Note that unlike DPI, CBMPI achieves good performance with very short rollouts m = 1. This indicates that CBMPI is able to approximate the value

B

= 8

,

000

,

000 per iteration

Figure 9: (a)-(c) Learning curves of CE, λ -PI, DPI, and CBMPI algorithms using the 22 Bertsekas features on the small 10 × 10 board.

<!-- image -->

function well, and as a result, build a more accurate training set for its classifier than DPI. Despite this improvement, the good results obtained by DPI in Tetris indicate that with small rollout horizons like m = 5, one already has fairly accurate action-value estimates in order to detect greedy actions accurately (at each iteration).

Overall, the results of Figure 8 show that an ADP algorithm, namely CBMPI, outperforms the CE method using a similar budget (80 vs. 65 millions after 10 iterations). Note that CBMPI takes less iterations to converge than CE. More generally, Figure 8 confirms the superiority of the policy search and classification-based PI methods to value-function based ADP algorithms ( λ -PI). This suggests that the D-T features are more suitable to represent policies than value functions in Tetris.

Figure 10: Learning curves of CBMPI, DPI and CE (left) using the 9 features listed in Table 2, and λ -PI (right) using the Bertsekas features (those for which λ -PI achieves here its best performance) on the large 10 × 20 board. The total budget B of CBMPI and DPI is set to 16,000,000 per iteration.

<!-- image -->

Bertsekas Features: Figures 9(a)-(c) show the performance of CE, λ -PI, DPI, and CBMPI. Here all the approximations in the algorithms are with the Bertsekas features plus constant offset. CE achieves the score 500 after about 60 iterations and outperforms λ -PI with score 350. It is clear that the Bertsekas features lead to much weaker results than those obtained by the D-T features (Figure 8) for all the algorithms. We may conclude then that the D-T features are more suitable than the Bertsekas features to represent both value functions and policies in Tetris. In DPI and CBMPI, we managed to obtain results similar to CE, only after multiplying the per iteration budget B used in the D-T experiments by 10. Indeed, CBMPI and DPI need more samples to solve the classification and regression problems in this 22-dimensional weight vector space than with the 9 D-T features. Moreover, in the classifier, the minimization of the empirical error through the CMA-ES method (see Equation 12) was converging most of the times to a local minimum. To solve this issue, we run multiple times the minimization problem with different starting points and small initial covariance matrices for the Gaussian distribution in order to force local exploration of different parts of the weight vector areas. However, CBMPI and CE require the same number of samples, 150 , 000 , 000, to reach their best performance, after 2 and 60 iterations, respectively (see Figure 9). Note that DPI and CBMPI obtain the same performance, which means that the use of a value function approximation by CBMPI does not lead to a significant performance improvement over DPI. We tried several values of m in this setting among which m = 10 achieved the best performance for both DPI and CBMPI.

## 6.2.2.2 Large ( 10 × 20 ) Board

We now use the best parameters and features in the small board experiments, run CE, DPI, and CBMPI in the large board, and report their results in Figure 10 (left). We also report

the results of λ -PI in the large board in Figure 10 (right). The per iteration budget of DPI and CBMPI is set to B = 32 , 000 , 000. While λ -PI with per iteration budget 100 , 000, at its best, achieves the score of 2 , 500, DPI and CBMPI, with m = 5 and m = 10, reach the scores of 12 , 000 , 000 and 20 , 000 , 000 after 3 and 8 iterations, respectively. CE matches the performances of CBMPI with the score of 20 , 000 , 000 after 8 iterations, this is achieved with almost 6 times more samples: after 8 iterations, CBMPI and CE use 256 , 000 , 000 and 1 , 700 , 000 , 000 samples, respectively.

## 6.2.2.3 Comparison of the Best Policies

So far the reported scores for each algorithm was averaged over the policies learned in 100 separate runs. Here we select the best policies observed in all our experiments and compute their scores more accurately by averaging over 10 , 000 games. We then compare these results with the best policies reported in the literature, i.e., DU and BDU (Thiery and Scherrer, 2009b) in both small and large boards in Table 1. The DT-10 and DT-20 policies, whose weights and features are given in Table 2, are policies learned by CBMPI with D-T features in the small and large boards, respectively. 12 As shown in Table 1, DT-10 removes 5 , 000 lines and outperforms DU, BDU, and DT-20 in the small board. Note that DT-10 is the only policy among these four that has been learned in the small board. In the large board, DT-20 obtains the score of 51 , 000 , 000 and not only outperforms the other three policies, but also achieves the best reported result in the literature (to the best of our knowledge). We observed in our experiments that the learning process in CBMPI has more variance in its performance than the one of CE. We believe this is why in the large board, although the policies learned by CE have similar performance to CBMPI (see Figure 10 (left)), the best policy learned by CBMPI outperforms BDU, the best one learned by CE (see Table 1).

| Boards \ Policies     | DU             | BDU            | DT-10          | DT-20          |
|-----------------------|----------------|----------------|----------------|----------------|
| Small (10 × 10) board | 3800           | 4200           | 5000           | 4300           |
| Large (10 20) board   | 31 , 000 , 000 | 36 , 000 , 000 | 29 , 000 , 000 | 51 , 000 , 000 |

×

Table 1: Average (over 10 , 000 games) score of DU, BDU, DT-10, and DT-20 policies.

| feature            |   weight |   weight | feature            |   weight |   weight | feature       |   weight |   weight |
|--------------------|----------|----------|--------------------|----------|----------|---------------|----------|----------|
| landing height     |    -2.18 |    -2.68 | column transitions |    -3.31 |    -6.32 | hole depth    |    -0.81 |    -0.43 |
| eroded piece cells |     2.42 |     1.38 | holes              |     0.95 |     2.03 | rows w/ holes |    -9.65 |    -9.48 |
| row transitions    |    -2.17 |    -2.41 | board wells        |    -2.22 |    -2.71 | diversity     |     1.27 |     0.89 |

Table 2: The weights of the 9 D-T features in the DT-10 (left) and DT-20 (right) policies.

12. Note that in the standard code by Thiery and Scherrer (2010b), there exist two versions of the feature 'board wells' numbered 6 and -6. In our experiments, we used the feature -6 as it is the more computationally efficient of the two.

## 7. Conclusion

In this paper, we considered a dynamic programming (DP) scheme for Markov decision processes, known as modified policy iteration (MPI). We proposed three original approximate MPI (AMPI) algorithms that are extensions of the existing approximate DP (ADP) algorithms: fitted-value iteration, fitted-Q iteration, and classification-based policy iteration. We reported a general error propagation analysis for AMPI that unifies those for approximate policy and value iteration. We instantiated this analysis for the three algorithms that we introduced, which led to a finite-sample analysis of their guaranteed performance. For the last introduced algorithm, CBMPI, our analysis indicated that the main parameter of MPI controls the balance of errors (between value function approximation and estimation of the greedy policy). The role of this parameter was illustrated for all the algorithms on two benchmark problems: Mountain Car and Tetris. Remarkably, in the game of Tetris, CBMPI showed advantages over all previous approaches: it significantly outperforms previous ADP approaches, and is competitive with black-box optimization techniques-the current state of the art for this domain-while using fewer samples. In particular, CBMPI led to what is to our knowledge the currently best Tetris controller, removing 51 , 000 , 000 lines on average. Interesting future work includes 1) the adaptation and precise analysis of our three algorithms to the computation of non-stationary policies-we recently showed that considering a variation of AMPI for computing non-stationary policies allows improving the 1 (1 -γ ) 2 constant (Lesner and Scherrer, 2013)-and 2) considering problems with large action spaces, for which the methods we have proposed here are likely to have limitation.

## Acknowledgments

The experiments were conducted using Grid5000 ( https://www.grid5000.fr ).

## Appendix A. Proof of Lemma 2

Before we start, we recall the following definitions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.1 Bounding b k

Using the definition of x k , i.e., we may write Equation 25 as

<!-- formula-not-decoded -->

- (a) From the definition of glyph[epsilon1] ′ k +1 , we have ∀ π ′ T π ′ v k ≤ T π k +1 v k + glyph[epsilon1] ′ k +1 , thus this inequality holds also for π ′ = π k .
- (b) This step is due to the fact that for every v and v ′ , we have T π k ( v + v ′ ) = T π k v + γP π k v ′ .
- (d) This step is due to the fact that for every v and v ′ , any m , we have ( T π k ) m v -( T π k ) m v ′ = ( γP π k ) m ( v -v ′ ).
- (c) This is from the definition of glyph[epsilon1] k , i.e., v k = ( T π k ) m v k -1 + glyph[epsilon1] k .

## A.2 Bounding d k

Define

Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (a) This step is from the definition of glyph[epsilon1] ′ k +1 (see step (a) in bounding b k ) and that of g k +1 in Equation 28.
- (b) This is from the definition of y k , i.e.,

<!-- formula-not-decoded -->

- (c) This step comes from rewriting g k +1 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.3 Bounding s k

With some slight abuse of notation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(a) For any v , we have v π k = ( T π k ) ∞ v . This step follows by setting v = v k -1 , i.e., v π k = ( T π k ) ∞ v k -1 .

## Appendix B. Proof of Lemma 4

We begin by focusing our analysis on AMPI. Here we are interested in bounding the loss l k = v ∗ -v π k = d k + s k .

and thus:

<!-- formula-not-decoded -->

By induction, from Equations 27 and 29, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in which we have used the notation introduced in Definition 3. In Equation 34, we also used the fact that from Equation 31, we may write g k +1 = ∑ m -1 j =1 Γ j b k . Moreover, we may rewrite Equation 32 as

<!-- formula-not-decoded -->

## B.1 Bounding l k

From Equations 33 and 34, we may write

<!-- formula-not-decoded -->

where we used the following definition

<!-- formula-not-decoded -->

The triple sum involved in Equation 36 may be written as

<!-- formula-not-decoded -->

Using Equation 37, we may write Equation 36 as

<!-- formula-not-decoded -->

Similarly, from Equations 35 and 33, we have

<!-- formula-not-decoded -->

where we used the following definition

<!-- formula-not-decoded -->

Finally, using the bounds in Equations 38 and 39, we obtain the following bound on the loss

<!-- formula-not-decoded -->

where we used the following definition

<!-- formula-not-decoded -->

Note that we have the following relation between b 0 and d 0

<!-- formula-not-decoded -->

In Equation 42, we used the fact that v ∗ = T π ∗ v ∗ , glyph[epsilon1] 0 = 0, and T π ∗ v 0 -T π 1 v 0 ≤ glyph[epsilon1] ′ 1 (this is because the policy π 1 is glyph[epsilon1] ′ 1 -greedy w.r.t. v 0 ). As a result, we may write | η k | either as

<!-- formula-not-decoded -->

or using the fact that from Equation 42, we have d 0 ≤ ( I -γP π ∗ ) -1 ( -b 0 + glyph[epsilon1] ′ 1 ), as

<!-- formula-not-decoded -->

Now, using the definitions of x k and y k in Equations 26 and 30, the bound on | η k | in Equation 43 or 44, and the fact that glyph[epsilon1] 0 = 0, we obtain

<!-- formula-not-decoded -->

where we used the following definition

<!-- formula-not-decoded -->

depending on whether one uses Equation 43 or Equation 44.

We end this proof by adapting the error propagation to CBMPI. As expressed by Equations 20 and 21 in Section 4, an analysis of CBMPI can be deduced from that we have just done by replacing v k with the auxiliary variable w k = ( T π k ) m v k -1 and glyph[epsilon1] k with ( γP π k ) m glyph[epsilon1] k -1 = Γ m glyph[epsilon1] k -1 . Therefore, using the fact that glyph[epsilon1] 0 = 0, we can rewrite the bound of Equation 46 for CBMPI as follows:

<!-- formula-not-decoded -->

## Appendix C. Proof of Lemma 6

For any integer t and vector z , the definition of Γ t and H¨ older's inequality imply that

We define

<!-- formula-not-decoded -->

where { ξ l } n l =1 is a set of non-negative numbers that we will specify later. We now have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

y

̂

<!-- image -->

̂

Figure 11: The notations used in the proof.

̂

̂

where (a) results from Jensen's inequality, (b) from Equation 48, and (c) from the definition of C q ( l ). Now, by setting ξ l = ( C q ( l ) ) 1 /p sup i ∈I l ‖ g i ‖ pq ′ ,µ , we obtain

<!-- formula-not-decoded -->

where the last step follows from the definition of K .

## Appendix D. Proof of Lemma 11

Let ̂ µ be the empirical distribution corresponding to states s (1) , . . . , s ( n ) . Let us define two N -dimensional vectors z = ( [ ( T π k ) m v k -1 ] ( s (1) ) , . . . , [ ( T π k ) m v k -1 ] ( s ( N ) ) ) glyph[latticetop] and y = ( ̂ v k ( s (1) ) , . . . , ̂ v k ( s ( N ) ) ) glyph[latticetop] and their orthogonal projections onto the vector space F N as ̂ z = ̂ Π z and ̂ y = ̂ Π y = ( ˜ v k ( s (1) ) , . . . , ˜ v k ( s ( N ) ) ) glyph[latticetop] , where ˜ v k is the result of linear regression and its truncation (by V max ) is v k , i.e., v k = T ( ˜ v k ) (see Figure 11). What we are interested in is to find a bound on the regression error ‖ z -̂ y ‖ (the difference between the target function z and the result of the regression y ). We may decompose this error as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ ̂ ̂ ̂ ̂ where ̂ ξ = ̂ z -̂ y is the projected noise (estimation error) ̂ ξ = ̂ Π ξ , with the noise vector ξ = z -y defined as ξ i = [ ( T π k ) m v k -1 ] ( s ( i ) ) -̂ v k ( s ( i ) ). It is easy to see that noise is zero mean, i.e., E [ ξ i ] = 0 and is bounded by 2 V max , i.e., | ξ i | ≤ 2 V max . We may write the estimation error as

- ̂ 13. We should discriminate between the linear function space F = { f α | α ∈ R d and f α ( · ) = φ ( · ) glyph[latticetop] α } , where φ ( · ) = ( ϕ 1 ( · ) , . . . , ϕ d ( · ) ) glyph[latticetop] , and its corresponding linear vector space F N = { Φ α, α ∈ R d } ⊂ R N , where Φ = [ φ ( s (1) ) glyph[latticetop] ; . . . ; φ ( s ( N ) ) glyph[latticetop] ] .

equals to { ̂ ξ i } N i =1 . By application of a variation of Pollard's inequality (Gy¨ orfi et al., 2002), we obtain with probability at least 1 -δ ′ . Thus, we have

<!-- formula-not-decoded -->

From Equations 49 and 50, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now in order to obtain a random design bound, we first define f ̂ α ∗ ∈ F as f ̂ α ∗ ( s ( i ) ) = [ ̂ Π( T π k ) m v k -1 ] ( s ( i ) ), and then define f α ∗ = Π( T π k ) m v k -1 that is the best approximation (w.r.t. µ ) of the target function ( T π k ) m v k -1 in F . Since f ̂ α ∗ is the minimizer of the empirical loss, any function in F different than f ̂ α ∗ has a bigger empirical loss, thus we have

<!-- formula-not-decoded -->

with probability at least 1 -δ ′ , where the second inequality is the application of a variation of Theorem 11.2 in Gy¨ orfi et al. (2002) with ‖ f α ∗ -( T π k ) m v k -1 ‖ ∞ ≤ V max + ‖ α ∗ ‖ 2 sup x ‖ φ ( x ) ‖ 2 . Similarly, we can write the left-hand-side of Equation 51 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ ′ , where Λ( N,d,δ ′ ) = 2( d +1)log N +log e δ ′ +log ( 9(12 e ) 2( d +1) ) . Putting together Equations 51, 52, and 53 and using the fact that T ( v k ) = v k , we obtain

The result follows by setting δ = 3 δ ′ and some simplifications.

## Appendix E. Proof of Lemma 12

Proof We prove the following series of inequalities:

<!-- formula-not-decoded -->

- (a) This step is the result of the following lemma.

Lemma 17 Let Π be the policy space of the policies obtained by Equation 4 from the truncation (by V max ) of the function space F , with finite VC-dimension h = V C (Π) &lt; ∞ . Let N &gt; 0 be the number of states in the rollout set D k , drawn i.i.d. from the state distribution µ . Then, we have

<!-- formula-not-decoded -->

Proof The proof is similar to the proof of Lemma 1 in Lazaric et al. (2010c).

with e ′ 3 ( N,δ ′ ) = 16 V max √ 2 N ( h log eN h +log 8 δ ′ ) .

- (b) This is from the definition of || glyph[epsilon1] ′ k || 1 , ̂ µ .
- (c) This step is the result of bounding

<!-- formula-not-decoded -->

by e ′ 4 ( N,M,δ ′ ). The supremum over all the policies in the policy space Π is due to the fact that π k is a random object whose randomness comes from all the randomly generated samples at the k 'th iteration (i.e., the states in the rollout set and all the generated rollouts). We bound this term using the following lemma.

Lemma 18 Let Π be the policy space of the policies obtained by Equation 4 from the truncation (by V max ) of the function space F , with finite VC-dimension h = V C (Π) &lt; ∞ . Let { s ( i ) } N i =1 be N states sampled i.i.d. from the distribution µ . For each sampled state s ( i ) ,

we take the action suggested by policy π , M times, and observe the next states { s ( i,j ) } j M =1 . Then, we have

<!-- formula-not-decoded -->

Proof The proof is similar to the proof of Lemma 4 in Lazaric et al. (2010a).

- (d) This step is from the definition of π k in the AMPI-V algorithm (Equation 4).
- (e) This step is algebra, replacing two maximums with one.
- (f) This step follows from applying Chernoff-Hoeffding to bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for each i = 1 , . . . , N , by e ′ 5 ( M,δ ′′ ) = V max √ 2 log(1 /δ ′′ ) M , followed by a union bound, which gives us e ′ 5 ( M,N,δ ′ ) = V max √ 2 log( N/δ ′ ) M . Note that the fixed action a ( i ) ∗ is defined as

The final statement of the theorem follows by setting δ = 3 δ

## Appendix F. Proof of Lemma 13

The proof of this lemma is similar to the proof of Theorem 1 in Lazaric et al. (2010c). Before stating the proof, we report the following two lemmas that are used in the proof.

Lemma 19 Let Π be a policy space with finite VC-dimension h = V C (Π) &lt; ∞ and N ′ be the number of states in the rollout set D ′ k -1 drawn i.i.d. from the state distribution µ . Then we have with glyph[epsilon1] = 16 Q max √ 2 N ′ ( h log eN ′ h +log 8 δ ) .

<!-- formula-not-decoded -->

Proof This is a restatement of Lemma 1 in Lazaric et al. (2010c).

′ .

Lemma 20 Let Π be a policy space with finite VC-dimension h = V C (Π) &lt; ∞ and s (1) , . . . , s ( N ′ ) be an arbitrary sequence of states. Assume that at each state, we simulate M independent rollouts. We have

 ∣ ∣ ∣ ∑ ∑ ( ) ∑ ( ) ∣ ∣ ∣  δ , with glyph[epsilon1] = 8 Q max √ 2 MN ′ ( h log eMN ′ h +log 8 δ ) . Proof The proof is similar to the one for Lemma 19.

<!-- formula-not-decoded -->

Proof (Lemma 13) Let a ∗ ( · ) ∈ argmax a ∈A Q k -1 ( · , a ) be a greedy action. To simplify the notation, we remove the dependency of a ∗ on states and use a ∗ instead of a ∗ ( s ( i ) ) in the following. We prove the following series of inequalities:

<!-- formula-not-decoded -->

The statement of the theorem is obtained by setting δ ′ = δ/ 4.

- (a) This follows from Lemma 19.
- (b) Here we introduce the estimated action-value function Q k -1 by bounding

using Lemma 20.

<!-- formula-not-decoded -->

- (c) From the definition of π k in CBMPI, we have

<!-- formula-not-decoded -->

thus, -1 /N ′ ∑ N ′ i =1 ̂ Q k -1 ( s ( i ) , π k ( s ( i ) ) ) can be maximized by replacing π k with any other policy, particularly with

<!-- formula-not-decoded -->

## References

- A. Antos, R. Munos, and Cs. Szepesv´ ari. Fitted q-iteration in continuous action-space MDPs. In Proceedings of the Advances in Neural Information Processing Systems 19 , pages 9-16, 2007.
- D. Bertsekas and S. Ioffe. Temporal differences-based policy iteration and applications in neuro-dynamic programming. Technical report, MIT, 1996.
- D. Bertsekas and J. Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.
- H. Burgiel. How to lose at tetris. Mathematical Gazette , 81:194-200, 1997.
5. Pelin Canbolat and Uriel Rothblum. (Approximate) iterated successive approximations algorithm for sequential decision processes. Annals of Operations Research , pages 1-12, 2012. ISSN 0254-5330.
- C. Chang and C. Lin. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology , 2(3):27:1-27:27, May 2011. ISSN 2157-6904. doi: 10.1145/1961189.1961199. URL http://doi.acm.org/10.1145/1961189.1961199 .
- E. Demaine, S. Hohenberger, and D. Liben-Nowell. Tetris is hard, even to approximate. In Proceedings of the Ninth International Computing and Combinatorics Conference , pages 351-363, 2003.
- C. Dimitrakakis and M. Lagoudakis. Rollout sampling approximate policy iteration. Machine Learning Journal , 72(3):157-171, 2008.
- D. Ernst, P. Geurts, and L. Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6:503-556, 2005.
- C. Fahey. Tetris AI, computer plays tetris, 2003. http://colinfahey.com/tetris/ tetris.html .
- A. Farahmand, R. Munos, and Cs. Szepesv´ ari. Error propagation for approximate policy and value iteration. In Proceedings of the Advances in Neural Information Processing Systems 22 , pages 568-576, 2010.
- V. Farias and B. Van Roy. Tetris: A study of randomized constraint sampling . SpringerVerlag, 2006.

<!-- formula-not-decoded -->

- A. Fern, S. Yoon, and R. Givan. Approximate policy iteration with a policy language bias: Solving relational Markov decision processes. Journal of Artificial Intelligence Research , 25:75-118, 2006.
- T. Furmston and D. Barber. A unifying perspective of parametric policy search methods for Markov decision processes. In Proceedings of the Advances in Neural Information Processing Systems 24 , pages 2726-2734, 2012.
- V. Gabillon, A. Lazaric, M. Ghavamzadeh, and B. Scherrer. Classification-based policy iteration with a critic. In Proceedings of the Twenty-Eighth International Conference on Machine Learning , pages 1049-1056, 2011.
- V. Gabillon, M. Ghavamzadeh, and B. Scherrer. Approximate dynamic programming finally performs well in the game of tetris. In Proceedings of Advances in Neural Information Processing Systems 26 , pages 1754-1762, 2013.
- M. Geist and B. Scherrer. Off-policy learning with eligibility traces: A survey. Journal of Machine Learning Research , 14, April 2014.
6. G.J. Gordon. Stable function approximation in dynamic programming. In ICML , pages 261-268, 1995.
- L. Gy¨ orfi, M. Kolher, M. Krzy˙ zak, and H. Walk. A distribution-free theory of nonparametric regression . Springer-Verlag, 2002.
- N. Hansen and A. Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation , 9:159-195, 2001.
- S. Kakade. A natural policy gradient. In Proceedings of the Advances in Neural Information Processing Systems 14 , pages 1531-1538, 2002.
- M. Kearns, Y. Mansour, and A. Ng. Approximate planning in large pomdps via reusable trajectories. In Proceedings of the Advances in Neural Information Processing Systems 12 , pages 1001-1007. MIT Press, 2000.
- M. Lagoudakis and R. Parr. Least-squares policy iteration. Journal of Machine Learning Research , 4:1107-1149, 2003a.
- M. Lagoudakis and R. Parr. Reinforcement learning as classification: Leveraging modern classifiers. In Proceedings of the Twentieth International Conference on Machine Learning , pages 424-431, 2003b.
- A. Lazaric, M. Ghavamzadeh, and R. Munos. Analysis of a classification-based policy iteration algorithm. Technical Report inria-00482065, INRIA, 2010a.
- A. Lazaric, M. Ghavamzadeh, and R. Munos. Finite-sample analysis of LSTD. In Proceedings of the Twenty-Seventh International Conference on Machine Learning , pages 615-622, 2010b.

- A. Lazaric, M. Ghavamzadeh, and R. Munos. Analysis of a classification-based policy iteration algorithm. In Proceedings of the Twenty-Seventh International Conference on Machine Learning , pages 607-614, 2010c.
- A. Lazaric, M. Ghavamzadeh, and R. Munos. Finite-sample analysis of least-squares policy iteration. Journal of Machine Learning Research , 13:3041-3074, 2012.
3. Boris Lesner and Bruno Scherrer. Tight performance bounds for approximate modified policy iteration with non-stationary policies. CoRR , abs/1304.5610, 2013.
- R. Munos. Error bounds for approximate policy iteration. In Proceedings of the Twentieth International Conference on Machine Learning , pages 560-567, 2003.
- R. Munos. Performance bounds in glyph[lscript] p -norm for approximate value iteration. SIAM J. Control and Optimization , 46(2):541-561, 2007.
- R. Munos and Cs. Szepesv´ ari. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9:815-857, 2008.
- D. Precup, R. Sutton, and S. Singh. Eligibility traces for off-policy policy evaluation. In Proceedings of the Seventeenth International Conference on Machine Learning , pages 759-766, 2000.
- D. Precup, R. Sutton, and S. Dasgupta. Off-policy temporal difference learning with function approximation. In Proceedings of the Eighteenth International Conference on Machine Learning , pages 417-424, 2001.
- M. Puterman. Markov decision processes . Wiley, New York, 1994.
- R. Rubinstein and D. Kroese. The cross-entropy method: A unified approach to combinatorial optimization, Monte-Carlo simulation, and machine learning . Springer-Verlag, 2004.
- B. Scherrer. Performance bounds for λ -policy iteration and application to the game of tetris. Journal of Machine Learning Research , 14:1175-1221, 2013.
- B. Scherrer and C. Thi´ ery. Performance bound for approximate optimistic policy iteration. Technical report, INRIA, 2010.
- B. Scherrer, M. Ghavamzadeh, V. Gabillon, and M. Geist. Approximate modified policy iteration. In Proceedings of the Twenty Ninth International Conference on Machine Learning , pages 1207-1214, 2012.
- S. Singh and R. Yee. An upper bound on the loss from approximate optimal-value functions. Machine Learning , 16-3:227-233, 1994.
15. Cs. Szepesv´ ari. Reinforcement learning algorithms for mdps. In Wiley Encyclopedia of Operations Research . Wiley, 2010.
- I. Szita and A. L˝ orincz. Learning tetris using the noisy cross-entropy method. Neural Computation , 18(12):2936-2941, 2006.

- C. Thiery and B. Scherrer. Building controllers for tetris. International Computer Games Association Journal , 32:3-11, 2009a. URL http://hal.inria.fr/inria-00418954 .
- C. Thiery and B. Scherrer. Improvements on learning tetris with cross entropy. International Computer Games Association Journal , 32, 2009b. URL http://hal.inria.fr/ inria-00418930 .
- C. Thiery and B. Scherrer. Least-squares λ -policy iteration: bias-variance trade-off in control problems. In Proceedings of the Twenty-Seventh International Conference on Machine Learning , pages 1071-1078, 2010a.
- C. Thiery and B. Scherrer. MDPTetris features documentation, 2010b. http://mdptetris. gforge.inria.fr/doc/feature\_functions\_8h.html .
- J. Tsitsiklis and B Van Roy. Feature-based methods for large scale dynamic programming. Machine Learning , 22:59-94, 1996.
- J. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control , 42(5):674-690, 1997.