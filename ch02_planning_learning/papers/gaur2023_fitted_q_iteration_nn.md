## On the Global Convergence of Fitted Q-Iteration with Two-layer Neural Network Parametrization

Mudit Gaur Mridul Agarwal Vaneet Aggarwal

Purdue University, West Lafayette IN 47907, USA

## Editor:

mgaur@purdue.eduu agarw180@purdue.edu vaneet@purdue.edu

## Abstract

Deep Q-learning based algorithms have been applied successfully in many decision making problems, while their theoretical foundations are not as well understood. In this paper, we study a Fitted Q-Iteration with two-layer ReLU neural network parameterization, and find the sample complexity guarantees for the algorithm. Our approach estimates the Qfunction in each iteration using a convex optimization problem. We show that this approach achieves a sample complexity of ˜ O (1 //epsilon1 2 ), which is order-optimal. This result holds for a countable state-spaces and does not require any assumptions such as a linear or low rank structure on the MDP.

## 1. Introduction

Reinforcement learning aims to maximize the cumulative rewards wherein an agent interacts with the system in a sequential manner, and has been used in many applications including games Silver et al. (2017); Vinyals et al. (2017); Bonjour et al. (2022), robotics Maes et al. (2003), autonomous driving Kiran et al. (2022), ride-sharing Al-Abbasi et al. (2019), networking Geng et al. (2020), recommender systems WARLOP et al. (2018), etc. One of the key class of algorithms used in reinforcement learning are those based on Q-learning. Due to large state space (possibly infinite), such algorithms are parameterized Van Hasselt et al. (2016). However, with general parametrization (e.g., neural network parametrization), sample complexity guarantees to achieve an /epsilon1 gap to the optimal Q-values have not been fully understood. In this paper, we study this problem to provide the first sample complexity results for Q-learning based algorithm with neural network parametrization.

We note that most of the theoretical results for Q-learning based algorithms have been in tabular setups which assume finite states such as in Li et al. (2020); Jin et al. (2018), or assume a linear function approximation Carvalho et al. (2020); Wang et al. (2021a); Chen et al. (2022); Zhou et al. (2021) where the linear representation is assumed to be known. Although practically large state spaces are handled using deep neural networks due to the universal approximation theorem, the theoretical analysis with neural network parametrization of Q-learning is challenging as the training of neural network involves solving a non-convex optimization problem. Recently, the authors of Fan et al. (2020) studied the problem of neural network parametrization for Q-learning based algorithms, and provided the asymptotic global convergence guarantees when the parameters of the neural

network used to approximate the Q function are large. In this paper, we study the first results on sample complexity to achieve the global convergence of Q-learning based algorithm with neural network parametrization.

In this paper, we propose a fitted Q-Iteration based algorithm, wherein the Q-function is parametrized by a two-layer ReLU network. The key for the two-layer ReLU network is that finding the optimal parameters for estimating the Q-function could be converted to a convex problem. In order to reduce the problem from non-convex to convex, one can replace the ReLU function with an equivalent diagonal matrix. Further, the minima of both the non-convex ReLU optimization and the equivalent convex optimization are same Wang et al. (2021b); Pilanci and Ergen (2020).

We find the sample complexity for fitted Q-Iteration based algorithm with neural network parametrization. The gap between the learnt Q-function and the optimal Q-function arises from the parametrization error due to the 2-layer neural network, the error incurred due to imperfect reformulation of the neural network as a convex optimization problem as well as the error due to the random nature of the underlying MDP. This error reduces with increasing the number of iterations of the fitted Q-iteration algorithm, the number of iterations of the convex optimization step, the number of data points sampled at each step. We achieve an overall sample complexity of ˜ O ( 1 (1 -γ ) 4 /epsilon1 2 ) . Our proof consists of expressing the error in estimation of the Q-function at a fixed iteration in terms of the Bellman error incurred at each successive iteration upto that point. We then express the Bellman error in terms of the previously mentioned components for the statistical error. We then upper bound the expectation of these error components with respect to a distribution of the state action space.

## 2. Related Works

## 2.1 Fitted Q-Iteration

The analysis of Q-learning based algorithms has been studied since the early 1990's Watkins and Dayan (1992); Singh and Yee (1994), where it was shown that small errors in the approximation of a task's optimal value function cannot produce arbitrarily bad performance when actions are selected by a greedy policy. Value Iteration algorithms, and its analogue Approximate Value iteration for large state spaces has been shown to have finite error bounds as Puterman (2014). Similar analysis for finite state space has been studied in Bertsekas and Tsitsiklis (1996) and Munos (2007).

When the state space is large (possibly infinite), the value function can not be updated at each state action pair at every iteration. Thus, approximate value iteration algorithms are used that obtains samples from state action space at each iteration and estimates the action value function by minimizing a loss which is a function of the sampled state action pairs. This is the basis of the Fitted Q-Iteration, first explained in Boyan and Moore (1994). Instead of updating the Q-function at each step of the trajectory, it collects a batch of sample transitions and updates the Q-functions based on the collected samples and the existing estimate of the Q-function. The obtained samples could be using a generative model Munos (2003); Ghavamzadeh et al. (2008), or using the buffer memory Kozakowski et al. (2022); Wang et al. (2020).

## 2.2 Deep Neural Networks in Reinforcement Learning

Parametrization of Q-network is required for scaling the Q-learning algorithms to a large state space. Neural networks have been used previously to parameterize the Q-function Tesauro et al. (1995); Xu and Gu (2020); Fujimoto et al. (2019). This approach, also called Deep Q-learning has been widely used in many reinforcement learning applications Yang et al. (2020); Damjanovi´ c et al. (2022); Gao et al. (2020). However, fundamental guarantees of Qlearning with such function approximation are limited, because the non-convexity of neural network makes the parameter optimization problem non-convex.

Even though the sample complexity of Q-learning based algorithms have not been widely studied for general parametrization, we note that such results have been studied widely for policy gradient based approaches Agarwal et al. (2020); Wang et al. (2019); Zhang et al. (2022). The policy gradient approaches directly optimize the policy, while still having the challenge that the parametrization makes the optimization of parameters non-convex. This is resolved by assuming that the class of parametrization (e.g., neural networks) is rich in the sense that arbitrary policy could be approximated by the parametrization. However, note that for systems with infinite state spaces, results for policy gradient methods consist of upper bounds on the gradient of the estimate of the average reward function, such as in Yuan et al. (2022). For upper bounds on the error of estimation of the value function, linear structure on the MDP has to be assumed, as is done in Chen and Zhao (2022).

In the case of analysis of Fitted Q-Iteration (FQI) algorithms such as Fan et al. (2020), upper bounds on the error of estimation of the Q-function are obtained by considering sparse Neural networks with ReLU functions and Holder smooth assumptions on the Neural networks. At each iteration of the FQI algorithm, an estimate of the Q-function is obtained by optimizing a square loss error, which is non-convex in the parameters of the neural network used to represent the Q-function. Due to this non-convexity, the upper bounds are asymptotic in terms of the parameters of the neural network and certain unknown constants. Xu and Gu (2020) improves upon this result by demonstrating finite time error bounds for Q-learning with neural network parametrization. However the error bounds obtained can go unbounded as the number of iterations increase (See Appendix A), hence they do not give any sample complexity results. Our result establishes the first sample complexity results for a (possibly) infinite state space without the need for a linear structure on the MDP.

## 2.3 Neural Networks Parameter Estimation

The global optimization of neural networks has shown to be NP hard Blum and Rivest (1992). Even though Stochastic Gradient Descent algorithms can be tuned to give highly accurate results as in Bengio (2012), convergence analysis of such methods requires assumptions such as infinite width limit such as in Zhu and Xu (2021). Recently, it has been shown that the parameter optimization for two-layer ReLU neural networks can be converted to an equivalent convex program which is exactly solvable and computationally tractable Pilanci and Ergen (2020). Convex formulations for convolutions and deeper models have also been studied Sahiner et al. (2020a,b). In this paper, we will use these approaches for estimating the parameterized Q-function.

## 3. Problem Setup

We study a discounted Markov Decision Process (MDP), which is described by the tuple M = ( S , A , P, R, γ ), where S is a bounded measurable state space, A is the finite set of actions, we represent set of state action pairs as [0 , 1] d (a d dimensional tuple with all elements in [0,1]), where d is a positive integer greater than 1. P : S × A → P ( S ) is the probability transition kernel 1 . R : S × A → P ([0 , R max ]) is the reward kernel on the state action space with R max being the absolute value of the maximum reward and 0 &lt; γ &lt; 1 is the discount factor. A policy π : S → P ( A ) is a mapping that maps state to a probability distribution over the action space. Here, we denote by P ( S ) , P ( A ) , P ([ a, b ]), the set of all probability distributions over the state space, the action space, and a closed interval [ a, b ], respectively. We define the action value function for a given policy π respectively as follows.

<!-- formula-not-decoded -->

where r ′ ( s t , a t ) ∼ R ( ·| s t , a t ), a t +1 ∼ π ( ·| s t +1 ) and s t +1 ∼ P ( ·| s t , a t ) for t = { 0 , · · · , ∞} . For a discounted MDP, we define the optimal action value functions as follows:

<!-- formula-not-decoded -->

A policy that achieves the optimal action value functions is known as the optimal policy and is denoted as π ∗ . It can be shown that π ∗ is the greedy policy with respect to Q ∗ Bertsekas and Shreve (2007). Hence finding Q ∗ is sufficient to obtain the optimal policy. We define the Bellman operator for a policy π as follows

<!-- formula-not-decoded -->

where r ( s, a ) = E ( r ′ ( s, a ) | ( s, a )) Similarly we define the Bellman Optimality Operator as

<!-- formula-not-decoded -->

Further, operator P π is defined as

<!-- formula-not-decoded -->

which is the one step Markov transition operator for policy π for the Markov chain defined on S × A with the transition dynamics given by S t +1 ∼ P ( ·| S t , A t ) and A t +1 ∼ π ( ·| S t +1 ). It defines a distribution on the state action space after one transition from the initial state. Similarly, P π 1 P π 2 · · · P π m is the m -step Markov transition operator following policy π t at steps 1 ≤ t ≤ m . It defines a distribution on the state action space after m transitions from the initial state. We have the relation

<!-- formula-not-decoded -->

1. For a measurable set X , let P ( X ) denote the set of all probability measures over X .

Which defines P ∗ as

<!-- formula-not-decoded -->

in other words, P ∗ is the one step Markov transition operator with respect to the greedy policy of the function on which it is acting.

This gives us the relation

<!-- formula-not-decoded -->

For any measurable function f : S × A : → R , we also define for any distribution ν ∈ P ( S × A ).

<!-- formula-not-decoded -->

For representing the action value function, we will use a 2 layer ReLU neural network. A 2-layer ReLU Neural Network with input x ∈ R d is defined as

<!-- formula-not-decoded -->

where m ≥ 1 is the number of neurons in the neural network, the parameter space is Θ m = R d × m × R m and θ = ( U, α ) is an element of the parameter space, where u i is the i th column of U , and α i is the i th coefficient of α . The function σ ′ : R → R ≥ 0 is the ReLU or restricted linear function defined as σ ′ ( x ) /defines max( x, 0). In order to obtain parameter θ for a given set of data X ∈ R n × d and the corresponding response values y ∈ R n × 1 , we desire the parameter that minimizes the squared loss (with a regularization parameter β ∈ [0 , 1]), given by

<!-- formula-not-decoded -->

Here, we have the term σ ( Xu i ) which is a vector { σ ′ (( x j ) T u i ) } j ∈{ 1 , ··· ,n } where x j is the j th row of X . It is the ReLU function applied to each element of the vector Xu i . We note that the optimization in Equation (11) is non-convex in θ due to the presence of the ReLU activation function. In Wang et al. (2021b), it is shown that this optimization problem has an equivalent convex form, provided that the number of neurons m goes above a certain threshold value. This convex problem is obtained by replacing the ReLU functions in the optimization problem with equivalent diagonal operators. The convex problem is given as

<!-- formula-not-decoded -->

where p ∈ R d ×| D X | .

D X is the set of diagonal matrices D i which depend on the dataset X . Except for cases of X being low rank it is not computationally feasible to obtain the set D X . We instead use ˜ D ∈ D X to solve the convex problem

<!-- formula-not-decoded -->

where p ∈ R d ×| ˜ D | .

## 4. Proposed Algorithm

## Algorithm 1 Iterative algorithm to estimate Q function

Input: S , A , γ, Time Horizon K ∈ Z , sampling distribution ν , one step transition operator κ , T k : k ∈{ 1 , ··· ,K }

Initialize: ˜

Q ( s, a ) = 0 ∀ ( s, a ) ∈ S × A

- 1: for k ∈ { 1 , · · · , K } do
- 2: sample n i.i.d ( s i , a i ) with s i , a i drawn from the sampling distribution ν
- 3: obtain { s ′ i , r ′ i } from κ ( s i , a i ) = { s ′ i , r ′ i }
- 5: Set X k , Y k as the matrix of the sampled state action pairs and vector of estimated Q values respectively
- 4: Set y i = r ′ i + γ max a ′ ∈A ˜ Q ( s ′ i , a ′ ), where i ∈ { 1 , · · · , n }
- 6: Call Algorithm 2 with input ( X = X k , y = Y k , T = T k ) and return parameter θ 7: ˜ Q = Q θ
- 8: end for

Define π as the greedy policy with respect to ˜ Q

Q of Q and π k as it's greedy policy

K Output: An estimator ˜ ∗

In this section, we describe our Neural Network Fitted Q-iteration algorithm. The key in the algorithm is the use of convex optimization for the update of parameters of the Qfunction. The algorithm, at each iteration k , updates the estimate of the Q function, here denoted as Q k . The update at each step in the ideal case is to be done by applying the Bellman optimality operator defined in Equation (3). However, there are two issues which prevent us from doing that. First, we do not know the transition kernel P . Second for the

The relevant details of the formulation and the definition of the diagonal matrices D i are provided in Appendix B. For a set of parameters θ = ( u, α ) ∈ Θ, we denote neural network represented by these parameters as

<!-- formula-not-decoded -->

case of an infinite state space, we cannot update the estimate at each iteration of the state space. Therefore, we apply an approximation of the Bellman optimality operator defined as

<!-- formula-not-decoded -->

where r ′ ( s, a ) ∼ R ( . | s, a ). Since we cannot perform even this approximated optimality operator for all state action pairs due to the possibly infinite state space, we instead update our estimate of the Q function at iteration k as the 2 layer ReLU Neural Network which minimizes the following loss function

<!-- formula-not-decoded -->

Here Q k -1 is the estimate of the Q function from iteration k -1 and the state action pairs have been sampled from some distribution, ν , over the state action pairs. Note that this is a problem of the form given in Equation (11) with y i = ( r ′ ( s i , a i ) + max a ′ ∈A γQ k -1 ( s ′ , a ′ )) where i ∈ (1 , · · · , n ) and Q θ represented as in Equation (14).

We define the Bellman error at iteration k as

<!-- formula-not-decoded -->

The main algorithm, Algorithm 1, iteratively samples from the state action space at every iteration, and the corresponding reward is observed. For each of the sampled state action pairs, the approximate Bellman optimality operator given in Equation (15) is applied to obtain the corresponding output value y . We have access to these values under Assumption 3. The sampled set of state action pairs and the corresponding y values are then passed to Algorithm 2, which returns the estimates of the neural network which minimizes the loss given in Equation (16). The algorithm updates the estimate of the action value function as the neural network corresponding to the estimated parameters.

Algorithm 2 optimizes the parameters for the neural network at each step of Algorithm 1. This is performed by reducing the problem to an equivalent convex problem as described in Appendix B. The algorithm first samples a set of diagonal matrices denoted by ˜ D in line 2 of Algorithm 2. The elements of ˜ D act as the diagonal matrix replacement of the ReLU function. Algorithm 2 then solves an optimization of the form given in Equation (13) with the regularization parameter β = 0. This convex optimization is solved in Algorithm 2 using the projected gradient descent algorithm. After obtaining the optima for this convex program, denoted by u ∗ = { u ∗ i } i ∈{ 1 , ··· , | ˜ D |} , in line 10, we transform them to the parameters of a neural network of the form given in Equation (14) which are then passed back to Algorithm 1. The procedure is described in detail along with the relevant definitions in Appendix B.

## Algorithm 2 Neural Network Parameter Estimation

- 1:
- Input:
- 2: Sample: ˜ D = diag (1( Xg i &gt; 0)) : g i ∼ N (0 , I ) , i ∈ [ | ˜ D | ]

data ( X,y,T )

- 3: Initialize y 1 = 0 , u 1 = 0 Initialize g ( u ) = || ∑ D i ∈ ˜ D D i Xu i -y || 2 2 4: for k ∈ { 0 , · · · , T } do
- 6: y k +1 = arg min y : | y | 1 ≤ Rmax 1 -γ || u k +1 -y || 2 2
- 5: u k +1 = y k -α k ∇ g ( y k )
- 7: end for
- 8: Set u T +1 = u ∗
- 9: Solve Cone Decomposition:

¯ v, ¯ w ∈ u ∗ i = v i -w i , i ∈ [ d ] } such that v i , w i ∈ K i and at-least one v i , w i is zero.

- 10: Construct ( θ = { u i , α i } ) using the transformation

for all i ∈ { 1 , · · · , m }

<!-- formula-not-decoded -->

- 11: Return θ

## 5. Error Characterization

We now characterize the errors which can result in the gap between the point of convergence and the optimal Q-function. To define the errors, we first define the various possible Q -functions which we can approximate in decreasing order of the accuracy.

We start by defining the best possible Q-function, Q k 1 for episode k &gt; 1. Q k 1 is the best approximation of the function TQ k -1 possible from the class of two layer ReLU neural networks, with respect to the expected square from the true ground truth TQ k -1 .

Definition 1 For a given iteration k of Algorithm 1, we define

<!-- formula-not-decoded -->

The expectation is with respect to the sampling distribution of the state action pairs denoted by ν . TQ k -1 is the bellman optimality operator applied to Q k -1 .

Note that we do not have access to the transition probability kernel P , hence we cannot calculate TQ k -1 . To alleviate this, we use the observed next states to estimate the Q-value function. Using this, we define Q k 2 as,.

Definition 2 For a given function Q : 0 , R max 1 γ , we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Compared to Q k 1 , in Q k 2 , we are minimizing the expected square loss from target function r ′ ( s, a ) + γ max a ′ Q ( s ′ , a ′ ) or the expected loss function.

( ) To obtain Q k 2 , we still need to compute the true expected value in Equation 20. However, we still do not know the transition function P . To remove this limitation, we use sampling. Consider a set, X , of state-action pairs sampled using distribution ν . We now define Q k 3 as,

Definition 3 For a given set of state action pairs X and a given function Q : S × A → [ 0 , R max 1 -γ ] we define where r ′ ( s i , a i ) , and s ′ i are the observes reward and the observed next state for state action pair s i , a i respectively.

<!-- formula-not-decoded -->

Q k 3 is the best possible approximation for Q -value function which minimizes the sample average of the square loss functions with the target values as ( r ′ ( s i , a i )+ γ max a ′ ∈A Q ( s ′ , a ′ ) ) 2 or the empirical loss function.

After defining the possible solutions for the Q -values using different loss functions, we define the errors.

We first define approximation error which represents the difference between TQ k -1 and its best approximation possible from the class of 2 layer ReLU neural networks. We have

Definition 4 (Approximation Error) For a given iteration k of Algorithm 1, /epsilon1 k 1 = TQ k -1 -Q k 1 , where Q k -1 is the estimate of the Q function at the iteration k -1 .

We also define Estimation Error which denotes the error between the best approximation of TQ k -1 possible from a 2 layer ReLU neural network and Q k 2 . We demonstrate that these two terms are the same and this error is zero.

Definition 5 (Estimation Error) For a given iteration k of Algorithm 1, /epsilon1 k 2 = Q k 1 -Q k 2 .

We now define Sampling error which denotes the difference between the minimizer of expected loss function Q k 2 and the minimizer of the empirical loss function using samples, Q k 3 . We will use Rademacher complexity results to upper bound this error.

Definition 6 (Sampling Error) For a given iteration k of Algorithm 1, /epsilon1 k 3 = Q k 2 -Q k 3 . Here X k is the set of state action pairs sampled at the k th iteration of Algorithm 1.

Lastly, we define optimization error which denotes the difference between the minimizer of the empirical square loss function, Q k 3 , and our estimate of this minimizer that is obtained from the projected gradient descent algorithm.

Definition 7 (Optimization Error) For a given iteration k of Algorithm 1, /epsilon1 k 4 = Q k 3 -Q k . Here Q k is our estimate of the Q function at iteration k of Algorithm 1.

## 6. Assumptions

In this section, we formally describe the assumptions that will be used in the results.

Assumption 1 Let θ ∗ /defines arg min θ ∈ Θ L ( θ ) , where L ( θ ) is defined in (11) and we denote Q θ ∗ ( · ) as Q θ ( · ) as defined in (14) for θ = θ ∗ . Also, let θ ∗ ˜ D /defines arg min θ ∈ Θ L | ˜ D | ( θ ) , where L ˜ D ( θ ) is defined in (52) . Further, we denote Q θ ∗ | ˜ D | ( · ) as Q θ ( · ) as defined in (14) for θ = θ ∗ | ˜ D | . Then we assume

<!-- formula-not-decoded -->

for any ν ∈ P ( S × A )

L ˜ D ( θ ) is the non-convex problem equivalent to the convex problem in (13). Thus, /epsilon1 | ˜ D | is a measure of the error incurred due to taking a sample of diagonal matrices ˜ D and not the full set D X . In practice, setting | ˜ D | to be the same order of magnitude as d (dimension of the data) gives us a sufficient number of diagonal matrices to get a reformulation of the non convex optimization problem which performs comparably or better than existing gradient descent algorithms, therefore /epsilon1 | ˜ D | is only included for theoretical completeness and will be negligible in practice. This has been practically demonstrated in Mishkin et al. (2022); Bartan and Pilanci (2022); Sahiner et al. (2022). Refer to Appendix B for details of D X , ˜ D and L | ˜ D | ( θ ).

Assumption 2 We assume that for all functions Q : S × A → [ 0 , ( R max 1 -γ )] , there exists a function Q θ where θ ∈ Θ such that

<!-- formula-not-decoded -->

for any ν ∈ P ( S × A ) .

/epsilon1 bias reflects the error that is incurred due to the inherent lack of expressiveness of the neural network function class. In the analysis of Fan et al. (2020), this error is assumed to be zero. We account for this error with an assumption similar to the one used in Liu et al. (2020).

Assumption 3 We assume that for the MDP M , we have access to a one step transition operator κ : S × A → S × ([0 , R max ]) defined as where s ′ ∼ P ( ·| ( s, a )) and r ′ ( s, a ) ∼ R ( . | s, a )

<!-- formula-not-decoded -->

One step sampling distributions offer a useful tool for finding next state and rewards as i.i.d samples. In practice such an operator may not be available and we only have access to these samples as a Markov chain. This can be overcome be storing the transitions and then sampling the stored transitions independently in a technique known as experience replay. Examples of experience replay being used are Mnih et al. (2013); Agarwal et al. (2021); Andrychowicz et al. (2017).

Assumption 4 Let ν 1 be a probability measure on S×A which is absolutely continuous with respect to the Lebesgue measure. Let { π t } be a sequence of policies and suppose that the state action pair has an initial distribution of ν 1 . Then we assume that for all ν 1 , ν 2 ∈ P ( S ×A ) there exists a constant φ ν 1 ,ν 2 ≤ ∞ such that

<!-- formula-not-decoded -->

for all m ∈ { 1 , · · · , ∞} , where d ( P π 1 P π 2 ··· P πm ν 2 ) dν 1 denotes the Radon Nikodym derivative of the state action distribution P π 1 P π 2 · · · P π m ν 2 with respect to the distribution ν 1 .

This assumption puts an upper bound on the difference between the state action distribution ν 1 and the state action distribution induced by sampling a state action pair from the distribution µ 2 followed by any possible policy for the next m steps for any finite value of m . Similar assumptions have been made in Fan et al. (2020); Lazaric et al. (2016).

## 7. Supporting Lemmas

We will now state the key lemmas that will be used for finding the sample complexity of the proposed algorithm.

Lemma 8 For any given iteration k ∈ { 1 , · · · , K } for the approximation error denoted by /epsilon1 k 1 in Definition 4, we have

<!-- formula-not-decoded -->

Proof Sketch: We use Assumption 2 and the definition of the variance of a random variable to obtain the required result. The detailed proof is given in Appendix E.1.

Lemma 9 For any given iteration k ∈ { 1 , · · · , K } , Q k 1 = Q k 2 , or equivalently /epsilon1 k 2 = 0

Proof Sketch: We use Lemma 14 in Appendix C and use the definitions of Q k 1 and Q k 2 to prove this result. The detailed proof is given in Appendix E.2.

Lemma 10 For any given iteration k ∈ { 1 , · · · , K } , if the number of samples of the state action pairs sampled by Algorithm 1 at iteration k , denoted by n k , satisfies

<!-- formula-not-decoded -->

for some constants C k , η and β k , then the error /epsilon1 k 3 defined in Definition 6 is upper bounded as

<!-- formula-not-decoded -->

Proof Sketch: First we note that for a fixed iteration k of Algorithm 1, E ( R X,Q k -1 ( θ )) = L Q k -1 ( θ ) where R X,Q k -1 ( θ ) and L Q k -1 ( θ ) are defined in Appendix E.3. We use this to get a probabilistic bound on the expected value of | ( Q k 2 ) -( Q k 3 ) | using Rademacher complexity theory. The detailed proof is given in Appendix E.3.

Lemma 11 For any given iteration k ∈ { 1 , · · · , K } of Algorithm 1, let the number of steps of the projected gradient descent performed by Algorithm 2, denoted by T k , and the gradient descent step size α k satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where /epsilon1 &lt; C ′ k , for some constants C ′ k , l k , L k and || ( u ∗ k ) || 2 . Then the error /epsilon1 k 4 defined in Definition 7 is upper bounded as

<!-- formula-not-decoded -->

Proof Sketch: We use the number of iterations T k required to get an /epsilon1 bound on the difference between the minimum objective value and the objective value corresponding to the estimated parameter at iteration T k . We use the convexity of the objective and the Lipschitz property of the neural network to get a bound on the Q functions corresponding to the estimated parameters. The detailed proof is given in Appendix E.4.

## 8. Main Result

In this section, we provide the guarantees for the proposed algorithm, which is given in the following theorem.

Theorem 1 Suppose Assumptions 1-4 hold. Let Algorithm 1 run for K iterations with n k state-action pairs sampled at iteration k ∈ { 1 , · · · , K } , T k be the number of steps used and step size α k in the projected gradient descent in Algorithm 2 at iteration k ∈ { 1 , · · · , K } and | ˜ D | be the number of diagonal matrices sampled in Algorithm 2 for all iterations. Let ν ∈ P ( S × A ) be the state action distribution used to sample the state action pairs in Algorithm 1. Further, let /epsilon1 ∈ (0 , 1) , C k , C ′ k , l k , , L k , φ ν,µ , β k , η, ( || u ∗ k || 2 ) be constants. If we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and /epsilon1 &lt; ( C ′ k ) for all k ∈ (1 , · · · , K ) , then we obtain

<!-- formula-not-decoded -->

where µ ∈ P ( S × A ) . Thus we get an overall sample complexity given by ∑ K k =1 n k = ˜ O ( /epsilon1 -2 (1 -γ ) -4 ) .

The above algorithm achieves a sample complexity of ˜ O (1 //epsilon1 2 ), which is the first such result for general parametrized large state space reinforcement learning. Further, we note that ˜ O (1 //epsilon1 2 ) is the best order-optimal result even in tabular setup Zhang et al. (2020). In the following, we provide an outline of the proof, with the detailed proof provided in the Appendix D.

The expectation of the difference between our estimated Q function denoted by Q π K and the optimal Q function denoted by Q ∗ (where π k is the policy obtained at the final step K of algorithm 1) is first expressed as a function of the Bellman errors (defined in Equation (17)) incurred at each step of Algorithm 1. The Bellman errors are then split into different components which are analysed separately. The proof is thus split into two stages. In the first stage, we demonstrate how the expectation of the error of estimation ( Q π K -Q ∗ ) of the Q function is upper bounded by a function of the Bellman errors incurred till the final step K . The second part is to upper bound the expectation of the Bellman error.

Upper Bounding Q Error In Terms Of Bellman Error: Since we only have access to the approximate Bellman optimality operator defined in Equation (15), we will rely upon the analysis laid out in Farahmand et al. (2010) and instead of the iteration of the value functions, we will apply a similar analysis to the action value function to get the desired result. We recreate the result for the value function from Lemmas 2 and 3 of Munos (2003) for the action value function Q to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where /epsilon1 k = TQ k -1 -Q k .

We use the results in Equation (36) and (37) to obtain

<!-- formula-not-decoded -->

and

The first term on the right hand side is called as the algorithmic error, which depends on how good our approximation of the Bellman error is. The second term on the right hand side is called as the statistical error, which is the error incurred due to the random nature of the system and depends only on the parameters of the MDP as well as the number of iterations of the FQI algorithm.

The expectation of the error of estimation is taken with respect to any arbitrary distribution on the state action space denoted as µ . The dependence of the expected value of the error of estimation on this distribution is expressed through the constant φ ν,µ , which is a measure of the similarity between the distributions ν and µ .

Upper Bounding Expectation of Bellman Error: The upper bound on E ( Q ∗ -Q π K ) µ in Equation (38) is in terms of E ( | /epsilon1 k | ) µ , where /epsilon1 k = TQ k -1 -Q k is a measure of how closely our estimate of the Q function at iteration k approximates the function obtained by applying the bellman optimality operator applied to the estimate of the Q function at iteration k -1. Intuitively, this error depends on how much data is collected at each iteration, how efficient our solution to the optimization step is to the true solution, and how well our function class can approximate the true Bellman optimality operator applied to the estimate at the end of the previous iteration. Building upon this intuition, we split /epsilon1 k into four different components as follows.

<!-- formula-not-decoded -->

We use the Lemmas 8, 9, 10, and 11 to bound the error terms in Equation (39). Before plugging these in Equation (38), we replace /epsilon1 in the Lemma results with /epsilon1 ′ = /epsilon1 (1 -γ ) 3 6 γ as these error terms are in a summation in Equation (38). We also bound the last term on the right hand side of Equation (38) by solving for a value of K that makes the term smaller than /epsilon1 3 .

## 9. Conclusion and Future Work

In this paper, we study a Fitted Q-Iteration with two-layer ReLU neural network parametrization, and find the sample complexity guarantees for the algorithm. Using the convex approach for estimating the Q-function, we show that our approach achieves a sample complexity of ˜ O (1 //epsilon1 2 ), which is order-optimal. This demonstrates the first approach for achieving sample complexity beyond linear MDP assumptions for large state space.

This study raises multiple future problems. First is whether we can remove the assumption on the generative model, while estimating Q-function by efficient sampling of past samples. Further, whether the convexity in training that was needed for the results could be extended to more than two layers. Finally, efficient analysis for the error incurred when a sample of cones are chosen rather than the complete set of cones and how to efficiently choose this subset will help with a complete analysis by which Assumption 1 can be relaxed.

## References

- Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. Optimality and approximation with policy gradient methods in markov decision processes. In Conference on Learning Theory , pages 64-66. PMLR, 2020.
- Naman Agarwal, Syomantak Chaudhuri, Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. Online target q-learning with reverse experience replay: Efficiently finding the optimal policy for linear mdps. arXiv preprint arXiv:2110.08440 , 2021.
- Abubakr O. Al-Abbasi, Arnob Ghosh, and Vaneet Aggarwal. Deeppool: Distributed modelfree algorithm for ride-sharing using deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems , 20(12):4714-4727, 2019. doi: 10.1109/TITS.2019. 2931830.
- Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf
- Burak Bartan and Mert Pilanci. Neural Fisher discriminant analysis: Optimal neural network embeddings in polynomial time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 1647-1663. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/bartan22a.html .
- Yoshua Bengio. Practical recommendations for gradient-based training of deep architectures. In Neural networks: Tricks of the trade , pages 437-478. Springer, 2012.
- Dimitri P. Bertsekas and Steven E. Shreve. Stochastic Optimal Control: The Discrete-Time Case . Athena Scientific, 2007. ISBN 1886529035.
- Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-dynamic programming. , volume 3 of Optimization and neural computation series . Athena Scientific, 1996. ISBN 1886529108.
- A. Blum and R. Rivest. Training a 3-node neural network is NP-complete. Neural Networks , 5:117-127, 1992.
- Trevor Bonjour, Marina Haliem, Aala Alsalem, Shilpa Thomas, Hongyu Li, Vaneet Aggarwal, Mayank Kejriwal, and Bharat Bhargava. Decision making in monopoly using a hybrid deep reinforcement learning approach. IEEE Transactions on Emerging Topics in Computational Intelligence , 2022.
- Justin Boyan and Andrew Moore. Generalization in reinforcement learning: Safely approximating the value function. In G. Tesauro, D. Touretzky, and T. Leen, editors, Advances in Neural Information Processing Systems , volume 7. MIT Press, 1994. URL https://proceedings.neurips.cc/paper/1994/file/ef50c335cca9f340bde656363ebd02fd-Paper.pdf

- D. Carvalho, Francisco S. Melo, and P. Santos. A new convergent variant of Q -learning with linear function approximation. In Adv. Neural Information Proc. Systems 33 , pages 19412-19421, 2020.
2. Xuyang Chen and Lin Zhao. Finite-time analysis of single-timescale actor-critic. arXiv preprint arXiv:2210.09921 , 2022.
3. Yuanzhou Chen, Jiafan He, and Quanquan Gu. On the sample complexity of learning infinite-horizon discounted linear kernel MDPs. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 3149-3183. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/chen22f.html .
4. Ivana Damjanovi´ c, Ivica Pavi´ c, Mate Puljiz, and Mario Brcic. Deep reinforcement learning-based approach for autonomous power flow control using only topology changes. Energies , 15(19), 2022. ISSN 1996-1073. doi: 10.3390/en15196920. URL https://www.mdpi.com/1996-1073/15/19/6920 .
5. Jianqing Fan, Zhaoran Wang, Yuchen Xie, and Zhuoran Yang. A theoretical analysis of deep q-learning. In Alexandre M. Bayen, Ali Jadbabaie, George Pappas, Pablo A. Parrilo, Benjamin Recht, Claire Tomlin, and Melanie Zeilinger, editors, Proceedings of the 2nd Conference on Learning for Dynamics and Control , volume 120 of Proceedings of Machine Learning Research , pages 486-489. PMLR, 10-11 Jun 2020. URL https://proceedings.mlr.press/v120/yang20a.html .
6. Amir-massoud Farahmand, Csaba Szepesv´ ari, and R´ emi Munos. Error propagation for approximate policy and value iteration. Advances in Neural Information Processing Systems , 23, 2010.
7. Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International conference on machine learning , pages 2052-2062. PMLR, 2019.
8. Ziming Gao, Yuan Gao, Yi Hu, Zhengyong Jiang, and Jionglong Su. Application of deep q-network in portfolio management. In 2020 5th IEEE International Conference on Big Data Analytics (ICBDA) , pages 268-275, 2020. doi: 10.1109/ICBDA49040.2020.9101333.
9. Nan Geng, Tian Lan, Vaneet Aggarwal, Yuan Yang, and Mingwei Xu. A multi-agent reinforcement learning perspective on distributed traffic engineering. In 2020 IEEE 28th International Conference on Network Protocols (ICNP) , pages 1-11. IEEE, 2020.
10. Mohammad Ghavamzadeh, Csaba Szepesv´ ari, Shie Mannor, et al. Regularized fitted qiteration: Application to planning. In European Workshop on Reinforcement Learning , pages 55-68. Springer, 2008.
11. Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is qlearning provably efficient? In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Infor-

mation Processing Systems , volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/file/d3b1fb02964aa64e257f9f26a31f72cf-Paper.pdf

- B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A. Al Sallab, Senthil Yogamani, and Patrick P´ erez. Deep reinforcement learning for autonomous driving: A survey. IEEE Transactions on Intelligent Transportation Systems , 23(6):49094926, 2022. doi: 10.1109/TITS.2021.3054625.
- Piotr Kozakowski, Lukasz Kaiser, Henryk Michalewski, Afroz Mohiuddin, and Katarzyna Ka´ nska. Q-value weighted regression: Reinforcement learning with limited data. In 2022 International Joint Conference on Neural Networks (IJCNN) , pages 1-8. IEEE, 2022.
- Alessandro Lazaric, Mohammad Ghavamzadeh, and R´ emi Munos. Analysis of classificationbased policy iteration algorithms. Journal of Machine Learning Research , 17(19):1-30, 2016. URL http://jmlr.org/papers/v17/10-364.html .
- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Sample complexity of asynchronous q-learning: Sharper analysis and variance reduction. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 7031-7043. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/4eab60e55fe4c7dd567a0be28016bff3-Paper.pdf
- Yanli Liu, Kaiqing Zhang, Tamer Basar, and Wotao Yin. An improved analysis of (variance-reduced) policy gradient and natural policy gradient methods. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 7624-7636. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/56577889b3c1cd083b6d7b32d32f99d5-Paper.pdf
- Sam Maes, Karl Tuyls, and Bernard Manderick. Reinforcement learning in large state spaces: Simulated robotic soccer as a testbed, 2003.
- Aaron Mishkin, Arda Sahiner, and Mert Pilanci. Fast convex optimization for two-layer ReLU networks: Equivalent model classes and cone decompositions. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 15770-15816. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/mishkin22a.html .
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.
- R´ emi Munos. Error bounds for approximate policy iteration. In ICML , volume 3, pages 560-567, 2003.
- R´ emi Munos. Performance Bounds in L p norm for Approximate Value Iteration. SIAM Journal on Control and Optimization , 46(2):541-561, 2007. doi: 10.1137/040614384. URL https://hal.inria.fr/inria-00124685 .

- Mert Pilanci and Tolga Ergen. Neural networks are convex regularizers: Exact polynomialtime convex optimization formulations for two-layer networks. In Hal Daum´ e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 7695-7705. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/pilanci20a.html .
- Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014.
- Arda Sahiner, Tolga Ergen, John Pauly, and Mert Pilanci. Vector-output relu neural network problems are copositive programs: Convex analysis of two layer networks and polynomial-time algorithms. arXiv preprint arXiv:2012.13329 , 2020a.
- Arda Sahiner, Morteza Mardani, Batu Ozturkler, Mert Pilanci, and John Pauly. Convex regularization behind neural reconstruction. arXiv preprint arXiv:2012.05169 , 2020b.
- Arda Sahiner, Tolga Ergen, Batu Ozturkler, John Pauly, Morteza Mardani, and Mert Pilanci. Unraveling attention via convex duality: Analysis and interpretations of vision transformers. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 19050-19088. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/sahiner22a.html .
- Kevin Scaman and Aladin Virmaux. Lipschitz regularity of deep neural networks: Analysis and efficient estimation. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , NIPS'18, page 3839-3848, Red Hook, NY, USA, 2018. Curran Associates Inc.
- David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature , 550:354-, October 2017. URL http://dx.doi.org/10.1038/nature24270 .
- Satinder P. Singh and Richard C. Yee. An upper bound on the loss from approximate optimal-value functions. Mach. Learn. , 16(3):227-233, sep 1994. ISSN 0885-6125. doi: 10.1023/A:1022693225949. URL https://doi.org/10.1023/A:1022693225949 .
- Gerald Tesauro et al. Temporal difference learning and td-gammon. Communications of the ACM , 38(3):58-68, 1995.
- Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence , volume 30, 2016.
- Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich K¨ uttler, John P. Agapiou, Julian Schrittwieser, John Quan, Stephen Gaffney, Stig Petersen, Karen Simonyan, Tom Schaul, Hado

van Hasselt, David Silver, Timothy P. Lillicrap, Kevin Calderone, Paul Keet, Anthony Brunasso, David Lawrence, Anders Ekermo, Jacob Repp, and Rodney Tsing. Starcraft II: A new challenge for reinforcement learning. CoRR , abs/1708.04782, 2017. URL http://arxiv.org/abs/1708.04782 .

- Chengwei Wang, Tengfei Zhou, Chen Chen, Tianlei Hu, and Gang Chen. Off-policy recommendation system without exploration. In Pacific-Asia Conference on Knowledge Discovery and Data Mining , pages 16-27. Springer, 2020.
- Lingxiao Wang, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural policy gradient methods: Global optimality and rates of convergence. arXiv preprint arXiv:1909.01150 , 2019.
- Tianhao Wang, Dongruo Zhou, and Quanquan Gu. Provably efficient reinforcement learning with linear function approximation under adaptivity constraints. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 13524-13536. Curran Associates, Inc., 2021a. URL https://proceedings.neurips.cc/paper/2021/file/70a32110fff0f26d301e58ebbca9cb9f-Paper.pdf
- Yifei Wang, Jonathan Lacotte, and Mert Pilanci. The hidden convex optimization landscape of regularized two-layer relu networks: an exact characterization of optimal solutions. In International Conference on Learning Representations , 2021b.
- Romain WARLOP, Alessandro Lazaric, and J´ er´ emie Mary. Fighting boredom in recommender systems with linear reinforcement learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/file/210f760a89db30aa72ca258a3483cc7f-Paper.pdf
- Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine Learning , 8(3):279-292, May 1992. ISSN 1573-0565. doi: 10.1007/BF00992698. URL https://doi.org/10.1007/BF00992698 .
- Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. In International Conference on Machine Learning , pages 10555-10565. PMLR, 2020.
- Ting Yang, Liyuan Zhao, Wei Li, and Albert Y. Zomaya. Reinforcement learning in sustainable energy and electric systems: a survey. Annu. Rev. Control. , 49:145-163, 2020.
- Rui Yuan, Robert M Gower, and Alessandro Lazaric. A general sample complexity analysis of vanilla policy gradient. In International Conference on Artificial Intelligence and Statistics , pages 3332-3380. PMLR, 2022.
- Matthew S Zhang, Murat A Erdogdu, and Animesh Garg. Convergence and optimality of policy gradient methods in weakly smooth settings. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 9066-9073, 2022.

- Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia reference-advantage decomposition. Advances in Neural Information Processing Systems , 33:15198-15207, 2020.
- Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted mdps with feature mapping. In International Conference on Machine Learning , pages 12793-12802. PMLR, 2021.
- Hanjing Zhu and Jiaming Xu. One-pass stochastic gradient descent in overparametrized two-layer neural networks. In International Conference on Artificial Intelligence and Statistics , pages 3673-3681. PMLR, 2021.

## A. Comparison of Result with Xu and Gu (2020)

Analysis of Q learning algorithms with neural network function approximation was carried out in Xu and Gu (2020), where finite time error bounds were studied for estimation of the Q function for MDP's with countable state spaces. The final result in Xu and Gu (2020) is of the form

<!-- formula-not-decoded -->

Here, Q ( s, a, θ t ) is the estimate of the Q function obtained using the neural network with the parameters obtained at step t of the Q learning algorithm represented by θ t and Q ∗ ( s, a ) is the optimal Q function. T is the total number of iterations of the Q learning algorithm.

The first term on the right hand side of (40) is monotonically decreasing with respect to T . The second term on the right hand side monotonically increases with respect to T . This term is present due to the error incurred by the linear approximation of the neural network representing the Q function. Our approach does not require this approximation due to the convex representation of the neural network. Due to the increasing function of T , it is not possible to obtain the number of iteration of the Q learning required to make the error of estimation smaller than some fixed value. Thus, this result cannot be said to be a sample complexity result. However, in our results, this is not the case.

## B. Convex Reformulation with Two-Layer Neural Networks

In order to understand the convex reformulation of the squared loss optimization problem, consider the vector σ ( Xu i )

<!-- formula-not-decoded -->

Now for a fixed X ∈ R n × d , different u i ∈ R d × 1 will have different components of σ ( Xu i ) that are non zero. For example, if we take the set of all u i such that only the first element of σ ( Xu i ) are non zero (i.e, only ( x 1 ) T u i ≥ 0 and ( x j ) T u i &lt; 0 ∀ j ∈ [2 , · · · , n ] ) and denote it by the set K 1 , then we have

<!-- formula-not-decoded -->

where D 1 is the n × n diagonal matrix with only the first diagonal element equal to 1 and the rest 0. Similarly, there exist a set of u ′ s which result in σ ( Xu ) having certain components to be non-zero and the rest zero. For each such combination of zero and nonzero components, we will have a corresponding set of u ′ i s and a corresponding n × n Diagonal matrix D i . We define the possible set of such diagonal matrices possible for a given matrix

X as

<!-- formula-not-decoded -->

where diag ( 1 ( Xu ≥ 0)) represents a matrix given by

<!-- formula-not-decoded -->

/negationslash where 1 ( x ) = 1 if x &gt; 0 and 1 ( x ) = 0 if x ≤ 0 . Corresponding to each such matrix D i , there exists a set of u i given by

<!-- formula-not-decoded -->

where I is the n × n identity matrix. The number of these matrices D i is upper bounded by 2 n . From Wang et al. (2021b) the upper bound is O ( r ( n r ) r ) where r = rank ( X ). Also, note that the sets K i form a partition of the space R d × 1 . Using these definitions, we define the equivalent convex problem to the one in Equation (11) as

<!-- formula-not-decoded -->

where v = { v i } i ∈ 1 , ··· , | D X | , w = { w i } i ∈ 1 , ··· , | D X | , v i , w i ∈ K i , note that by definition, for any fixed i ∈ { 1 , · · · , | D X |} at-least one of v i or w i are zero. If v ∗ , w ∗ are the optimal solutions to Equation (45), the number of neurons m of the original problem in Equation (11) should be greater than the number of elements of v ∗ , w ∗ , which have at-least one of v ∗ i or w ∗ i non-zero. We denote this value as m ∗ X,y , with the subscript X denoting that this quantity depends upon the data matrix X and response y .

We convert v ∗ , w ∗ to optimal values of Equation (11), denoted by θ ∗ = ( U ∗ , α ∗ ), using a function ψ : R d × R d → R d × R defined as follows

<!-- formula-not-decoded -->

Since D X is hard to obtain computationally unless X is of low rank, we can construct a subset ˜ D ∈ D X and perform the optimization in Equation (45) by replacing D X with ˜ D to get where according to Pilanci and Ergen (2020) we have ( u ∗ i , α ∗ i ) = ψ ( v ∗ i , w ∗ i ), for all i ∈ { 1 , · · · , | D X |} where u ∗ i , α ∗ i are the elements of θ ∗ . Note that restriction of α i to { 1 , -1 , 0 } is shown to be valid in Mishkin et al. (2022). For i ∈ {| D X | +1 , · · · , m } we set ( u ∗ i , α ∗ i ) = (0 , 0).

<!-- formula-not-decoded -->

where v = { v i } i ∈ 1 , ··· , | ˜ D | , w = { w i } i ∈ 1 , ··· , | ˜ D | , v i , w i ∈ K i , by definition, for any fixed i ∈ { 1 , · · · , | ˜ D |} at-least one of v i or w i are zero.

The required condition for ˜ D to be a sufficient replacement for D X is as follows. Suppose ( v, w ) = (¯ v i , ¯ w i ) i ∈ (1 , ··· , | ˜ D | ) denote the optimal solutions of Equation (47). Then we require

/negationslash

<!-- formula-not-decoded -->

/negationslash

Or, the number of neurons in the neural network are greater than the number of indices i for which at-least one of v ∗ i or w ∗ i is non-zero. Further,

<!-- formula-not-decoded -->

In other words, the diagonal matrices induced by the optimal u ∗ i 's of Equation (11) must be included in our sample of diagonal matrices. This is proved in Theorem 2.1 of Mishkin et al. (2022).

A computationally efficient method for obtaining ˜ D and obtaining the optimal values of the Equation (11), is laid out in Mishkin et al. (2022). In this method we first get our sample of diagonal matrices ˜ D by first sampling a fixed number of vectors from a d dimensional standard multivariate distribution, multiplying the vectors with the data matrix X and then forming the diagonal matrices based of which co-ordinates are positive. Then we solve an optimization similar to the one in Equation (45), without the constraints, that its parameters belong to sets of the form K i as follows.

<!-- formula-not-decoded -->

where p ∈ R d ×| ˜ D | . In order to satisfy the constraints of the form given in Equation (45), this step is followed by a cone decomposition step. This is implemented through a function { ψ ′ i } i ∈{ 1 , ··· , | ˜ D |} . Let p ∗ = { p ∗ i } i ∈{ 1 , ··· , | ˜ D |} be the optimal solution of Equation (50). For each i we define a function ψ ′ i : R d → R d × R d as

<!-- formula-not-decoded -->

Then we obtain ψ ( p ∗ i ) = (¯ v i , ¯ w i ). As before, at-least one of v i , w i is 0. Note that in practice we do not know if the conditions in Equation (48) and (49) are satisfied for a given sampled ˜ D . We express this as follows. If ˜ D was the full set of Diagonal matrices then we would have (¯ v i , ¯ w i ) = v ∗ i , w ∗ i and ψ (¯ v i , ¯ w i ) = ( u ∗ i , α ∗ i ) for all i ∈ (1 , · · · , | D X | ). However, since that is not the case and ˜ D ∈ D X , this means that { ψ (¯ v i , ¯ w i ) } i ∈ (1 , ··· , | ˜ D | ) is an optimal solution of a non-convex optimization different from the one in Equation (11). We denote this non-convex optimization as L | ˜ D | ( θ ) defined as

<!-- formula-not-decoded -->

where m ′ = | ˜ D | or the size of the sampled diagonal matrix set. In order to quantify the error incurred due to taking a subset of D X , we assume that the expectation of the absolute value of the difference between the neural networks corresponding to the optimal solutions of the non-convex optimizations given in Equations (52) and (11) is upper bounded by a constant depending on the size of ˜ D . The formal assumption and its justification is given in Assumption 1.

## C. Supplementary lemmas and Definitions

Here we provide some definitions and results that will be used to prove the lemmas stated in the paper.

Definition 12 For a given set Z ∈ R n , we define the Rademacher complexity of the set Z as where Ω i is random variable such that P (Ω i = 1) = 1 2 , P (Ω i = -1) = 1 2 and z i are the co-ordinates of z which is an element of the set Z

<!-- formula-not-decoded -->

Lemma 13 Consider a set of observed data denoted by z = { z 1 , z 2 , · · · z n } ∈ R n , a parameter space Θ , a loss function { l : R × Θ → R } where 0 ≤ l ( θ, z ) ≤ 1 ∀ ( θ, z ) ∈ Θ × R . The empirical risk for a set of observed data as R ( θ ) = 1 n ∑ n i =1 l ( θ, z i ) and the population risk as r ( θ ) = E l ( θ, ˜ z i ) , where ˜ z i is a co-ordinate of ˜ z sampled from some distribution over Z .

We define a set of functions denoted by L as

<!-- formula-not-decoded -->

Given z = { z 1 , z 2 , z 3 · · · , z n } we further define a set L◦ z as

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

If the data is of the form z i = ( x i , y i ) , x ∈ X,y ∈ Y and the loss function is of the form l ( a θ ( x ) , y ) , is L lipschitz and a θ : Θ × X → R , then we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The detailed proof of the above statement is given in (Rebeschini, P. (2022). Algorithmic Foundations of Learning [Lecture Notes]. https://www.stats.ox.ac.uk/ rebeschi/teaching/AFoL/20/material/. The upper bound for E sup θ ∈ Θ ( { r ( θ ) -R ( θ ) } ) is proved in the aformentioned reference. However, without loss of generality the same proof holds for the upper bound for E sup θ ∈ Θ ( { R ( θ ) -r ( θ ) } ). Hence the upper bound for E sup θ ∈ Θ |{ r ( θ ) -R ( θ ) }| can be established.

Lemma 14 Consider two random random variable x ∈ X and y, y ′ ∈ Y . Let E x,y , E x and E y | x , E y ′ | x denote the expectation with respect to the joint distribution of ( x, y ) , the marginal distribution of x , the conditional distribution of y given x and the conditional distribution of y ′ given x respectively . Let f θ ( x ) denote a bounded measurable function of x parameterised by some parameter θ and g ( x, y ) be bounded measurable function of both x and y .

Then we have

<!-- formula-not-decoded -->

Proof Denote the left hand side of Equation (59) as X θ , then add and subtract E y | x ( g ( x, y ) | x ) to it to get

<!-- formula-not-decoded -->

Consider the third term on the right hand side of Equation (61)

<!-- formula-not-decoded -->

Equation (62) is obtained by writing E x,y = E x E y | x from the law of total expectation. Equation (63) is obtained from (62) as the term f θ ( x ) -E y ′ | x ( g ( x, y ′ ) | x ) is not a function of y . Equation (64) is obtained from (63) as E y | x ( E y ′ | x ( g ( x, y ′ ) | x ) ) = E y ′ | x ( g ( x, y ′ ) | x ) because E y ′ | x ( g ( x, y ′ ) | x ) is not a function of y hence is constant with respect to the expectation operator E y | x .

Thus plugging in value of 2 E x,y ( f θ ( x ) -E y ′ | x ( g ( x, y ′ ) | x ) )( g ( x, y ) -E y ′ | x ( g ( x, y ′ ) | x ) ) in Equation (61) we get

<!-- formula-not-decoded -->

Note that the second term on the right hand side of Equation (67) des not depend on f θ ( x ) therefore we can write Equation (67) as

<!-- formula-not-decoded -->

Since the right hand side of Equation (68) is not a function of y we can replace E x,y with E x to get

<!-- formula-not-decoded -->

Lemma 15 Consider an optimization of the form given in Equation (47) with the regularization term β = 0 denoted by L | ˜ D | and it's convex equivalent denoted by L 0 . Then the value of these two loss functions evaluated at ( v, w ) = ( v i , w i ) i ∈{ 1 , ··· , | ˜ D |} and θ = ψ ( v i , w i ) i ∈{ 1 , ··· , | ˜ D |} respectively are equal and thus we have

<!-- formula-not-decoded -->

Proof Consider the loss functions in Equations (45), (50) with β = 0 are as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ψ ( v i , w i ) 1 , ψ ( v i , w i ) 2 represent the first and second coordinates of ψ ( v i , w i ) respectively.

For any fixed i ∈ { 1 , · · · , | ˜ D |} consider the two terms

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a fixed i either v i or w i is zero. In case both are zero, both of the terms in Equations (73) and (74) are zero as ψ (0 , 0) = (0 , 0). Assume that for a given i w i = 0. Then we have ψ ( v i , w i ) = ( v i , 1). Then equations (73), (74) are.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

But by definition of v i we have D i ( X ( v i ) = σ ( X ( v i )), therefore Equations (75), (76) are equal. Alternatively if for a given i v i = 0, then ψ ( v i , w i ) = ( w i , -1), then the terms in (73), (74) become.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition of w i we have D i ( X ( w i ) = σ ( X ( w i )), then the terms in (77), (77) are equal. Since this is true for all i , we have

<!-- formula-not-decoded -->

Lemma 16 The function Q θ ( x ) defined in equation (14) is Lipschitz continuous in θ , where θ is considered a vector in R ( d +1) m with the assumption that the set of all possible θ belong to the set B = { θ : | θ ∗ -θ | 1 &lt; 1 } , where θ ∗ is some fixed value.

## Proof

Note that

First we show that for all θ 1 = { u i , α i } , θ 2 = { u ′ i , α ′ i } ∈ B we have α i = α ′ i for all i ∈ (1 , · · · , m )

<!-- formula-not-decoded -->

/negationslash

By construction α i , α ′ i can only be 1, -1 or 0. Therefore if α i = α ′ i then | α i -α ′ i | = 2 if both non zero or | α i -α ′ i | = 1 if one is zero. Therefore | θ 1 -θ 2 | 1 ≥ 1. Which leads to a contradiction.

where | u i -u ′ i | 1 = ∑ d j =1 | u i j -u ′ i j | with u i j , u ′ i j denote the j th component of u i , u ′ i respectively.

Therefore α i = α ′ i for all i and we also have

Q θ ( x ) is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Proposition 1 in Scaman and Virmaux (2018) the function Q θ ( x ) is Lipschitz continuous in x , therefore there exist l &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we consider a single neuron of Q θ , for example i = 1, we have l 1 &gt; 0 such that

<!-- formula-not-decoded -->

Now consider Equation (85), but instead of considering the left hand side a a function of x, y consider it a function of u where we consider the difference between σ ′ ( x T u ) α i evaluated at u 1 and u ′ 1 such that

<!-- formula-not-decoded -->

Similarly, for all other i if we change u i to u ′ i to be unchanged we have for some l x 1 &gt; 0.

Therefore we obtain

<!-- formula-not-decoded -->

for all x if both θ 1 , θ 2 ∈ B .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This result for a fixed x . If we take the supremum over x on both sides we get

<!-- formula-not-decoded -->

Denoting (sup i,x l x i ) = l , we get

<!-- formula-not-decoded -->

## D. Proof of Theorem 1

## Proof

For ease of notations, let Q 1 , Q 2 be two real valued functions on the state action space. The expression Q 1 ≥ Q 2 implies Q 1 ( s, a ) ≥ Q 2 ( s, a ) ∀ ( s, a ) ∈ S × A .

Q k denotes our estimate of the action value function at step k of Algorithm 1 and Q π k denotes the action value function induced by the policy π k which is the greedy policy with respect to Q k .

Consider /epsilon1 k +1 = TQ k -Q k +1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This follows from the definition of T π ∗ and T in Equation (3) and (4), respectively.

Thus we get,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Right hand side of Equation (96) is obtained by writing Q ∗ = T π ∗ Q ∗ . This is because the function Q ∗ is a stationary point with respect to the operator T π ∗ . Equation (97) is obtained from (96) by adding and subtracting T π ∗ Q k . Equation (99) is obtained from (98) as P π ∗ Q k ≤ P ∗ Q k and P ∗ is the operator with respect to the greedy policy of Q k .

By recursion on k , we get,

<!-- formula-not-decoded -->

using TQ K ≥ T π ∗ Q K (from definition of T π ∗ ) and TQ K = T π K Q K as π k is the greedy policy with respect to Q k hence T π K acts on it the same way T does.

Similarly we write,

<!-- formula-not-decoded -->

The right hand side of Equation (102) is obtained by adding and subtracting T π ∗ Q K and TQ K to the right hand side of Equation (101). Equation (103) is obtained from (102) by noting that the term T π ∗ Q K -TQ K is non-positive for all ( s, a ) as T is the greedy operator on a Q function and results in a higher or equal value than any other operator. Equation (105) is obtained from (104) by writing P π k Q π k = P ∗ Q π k . This is true as π k is the greedy policy with respect to Q π k , hence the operator P ∗ acts on Q π k in the same way as P π k . Equation (106) is obtained from (105) by adding and subtracting Q ∗ to the second term on the right hand side.

By rearranging the terms in Equation (106) we get

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Equation (110) is obtained from (109) by taking the inverse of the operator ( I -γP π K ) on both sides of (109)

Plugging in value of ( Q ∗ -Q K ) from Equation (100) in Equation (110), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (113) is obtained from Equation (112) by noting that the operator ( P π ∗ + P π K ) acting on | /epsilon1 k | will produce a larger value than ( P π ∗ -P π K ) acting on | /epsilon1 k | .

We know Q ∗ ≤ R max 1 -γ for all ( s, a ) ∈ S × A and Q 0 = 0 by initialization. Therefore, we have

<!-- formula-not-decoded -->

For simplicity, in stating our results, we also define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then substitute the value α k and A k from Equation (118) and (119) respectively to get,

<!-- formula-not-decoded -->

Taking an expectation on both sides with respect to µ ∈ P ( S × A ) we get

<!-- formula-not-decoded -->

Consider the term A in Equation (121) Plugging in the value of α k A k in the term E ( α k A k ( /epsilon1 k )) ν , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (123) is obtained from (122) by writing ( I -γP π k ) -1 = ∑ ∞ m =1 ( γP π k ) m using the binomial expansion formula.

Consider the term A 1 in Equation (124), We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, Equation (130) is obtained from (129) from Assumption 4. In the same manner, we have

<!-- formula-not-decoded -->

Denote the term in Equation (128) as

<!-- formula-not-decoded -->

Where ( P π k ) m ( P π ∗ ) K -k µ is the marginal distribution of the state action pair at step m + K -k + 1 denoted by ˜ ν for notational simplicity. It is the state action distribution at step m + K -k +1 obtained by starting the state action pair sampled form µ and the following the policies π ∗ for K -k steps and then π k for m steps. We then get Equation (129) from (128) from the definition of the Radon Nikodym derivative as follows.

<!-- formula-not-decoded -->

Plugging upper bound on A 1 and A 2 into Equation (125), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now consider B in Equation (121). We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (140) is obtained from the fact that the transition operator A k acting on the constant R max 1 -γ is equivalent to multiplying it by 1 -γ . Equation (141) is obtained by plugging in the value of α K from Equation (118). Plugging upper bound on A and value of B into Equation (121), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Equation (39) we have that

<!-- formula-not-decoded -->

Thus, equation (143) becomes

<!-- formula-not-decoded -->

Consider T 1, from Lemma 8, we have E ( | /epsilon1 k 1 | ) ν ≤ √ /epsilon1 bias . Thus, for term T 1, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider T 3, from Lemma 10, we have that if the number of samples of state action pairs at iteration k denoted by n k satisfy

then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging in /epsilon1 = /epsilon1 (1 -γ ) 2 6 φ ν,µ γ in Equation (152), we get that if number of samples of state action pairs at iteration k denoted by n k and the step size α satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (159) is obtained from (158) by using the inequality ∑ K -1 k =1 γ K -k ≤ 1 1 -γ . Consider T 4. From Lemma 11, we have that if the number of iterations of the projected gradient descent algorithm at iteration k denoted by T k and the step size denoted by α k satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(164)

then we have

and /epsilon1 &lt; C ′ k , then we have

<!-- formula-not-decoded -->

Plugging in /epsilon1 = /epsilon1 (1 -γ ) 2 6 φ ν,µ γ in Equation (162) we get that if the number of iterations of the projected gradient descent algorithm at iteration k denoted by T k and the step size denoted by α k satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and if /epsilon1 &lt; C ′ k for all k ∈ (1 , · · · , K ), then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (169), (172) are obtained from (168), (171), respectively, by using the inequality ∑ K -1 k =1 γ K -k ≤ 1 1 -γ . Consider T 5. Assume K is large enough such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (177) is obtained from (176) by multiplying on -1 on both sides and noting that log( x ) = -log 1 x

( ) Thus, we obtain that if the number of iterations of Algorithm 1, number of state actions pairs sampled at iteration k denoted by n k and the number of iterations of the projected gradient descent on iteration k denoted by T k and the step size denoted by α k satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and if /epsilon1 &lt; ( C ′ k ) for all k ∈ (1 , · · · , K ), We have

<!-- formula-not-decoded -->

## E. Proof of Supporting Lemmas

## E.1 Proof Of Lemma 8

## Proof

Using Assumption 2 and the definition of Q k 1 for some iteration k of Algorithm 1 we have

<!-- formula-not-decoded -->

Since | a | 2 = a 2 we obtain

We have for a random variable x , V ar ( x ) = E ( x 2 ) -( E ( x )) 2 hence E ( x ) = √ E ( x 2 ) -V ar ( x ), Therefore replacing x with | TQ k -1 -Q k 1 | we get

<!-- formula-not-decoded -->

using the definition of the variance of a random variable we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since /epsilon1 k 1 = TQ k -1 -Q k 1 we have

/squaresolid

## E.2 Proof Of Lemma 9

Proof From Lemma 14, we have

<!-- formula-not-decoded -->

The function f θ ( x ) to be Q θ ( s, a ) and g ( x, y ) to be the function r ′ ( s, a )+max a ′ ∈A γQ k -1 ( s ′ , a ′ ). We also have y as the two dimensional random variable ( r ′ ( s, a ) , s ′ ). We now have ( s, a ) ∼ ν and s ′ | ( s, a ) ∼ P ( . | ( s, a )) and r ′ ( s, a ) ∼ R ( . | s, a ).

Then the loss function in (59) becomes

<!-- formula-not-decoded -->

Therefore by Lemma 14, we have that the function Q θ ( s, a ) which minimizes Equation (191) it will be minimizing

<!-- formula-not-decoded -->

But we have from Equation (4) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we get

Combining Equation (191) and (193) we get

<!-- formula-not-decoded -->

The left hand side of Equation (194) is Q k 2 as defined in Definition 2 and the right hand side is Q k 1 as defined in Definition 1, which gives us

<!-- formula-not-decoded -->

## E.3 Proof Of Lemma 10

## Proof

We define R X,Q k -1 ( θ ) as

<!-- formula-not-decoded -->

Here, X = { s i , a i } i = { 1 , ··· , | X |} , where s i , a i ∼ ν ∈ P ( S × A ), r ( s i , a i ) ∼ R ( . | s i , a i ) and s ′ i ∼ P ( . | s i , a i ). θ ∈ Θ, Q θ is as defined in Equation (14) and Q k -1 is the estimate of the Q function obtained at iteration k -1 of Algorithm 1.

We also define the term

<!-- formula-not-decoded -->

We denote by θ k 2 , θ k 3 the parameters of the neural networks Q k 2 , Q k 3 respectively. Q k 2 , Q k 3 are defined in Definition 2 and 3 respectively.

We then obtain,

<!-- formula-not-decoded -->

We get the inequality in Equation (196) because L Q k -1 ( θ k 3 ) -L Q k -1 ( θ k 2 ) &gt; 0 as Q k 2 is the minimizer of the loss function L Q k -1 ( Q θ ).

Consider Lemma 13. The loss function R X k ,Q k -1 ( θ k 3 ) can be written as the mean of loss functions of the form l ( a θ ( s i , a i ) , y i ) where l is the square function. a θ ( s i , a i ) = Q θ ( s i , a i ) and y i = ( r ′ ( s i , a i ) + γ max a ′ ∈A Q k -1 ( s ′ , a ′ ) ) . Thus we have

<!-- formula-not-decoded -->

Where n k = | X | , ( A◦{ ( s 1 , a 1 ) , ( s 2 , a 2 ) , ( s 3 , a 3 ) , · · · , ( s n , a n ) } = { Q θ ( s 1 , a 1 ) , Q θ ( s 2 , a 2 ) , · · · , Q θ ( s n , a n ) } and η is the Lipschitz constant for the square function over the state action space [0 , 1] d . The expectation is with respect to ( s i , a i ) ∼ ν, s ′ i ∼ P ( s ′ | s, a ) , r i ∼ R ( . | s i , a i ) i ∈ (1 , ··· ,n k ) , .

From (Ma,Tengyu.(2018). Statistical Learning Theory [Lecture Notes]. https://web.stanford.edu/class/cs2 we have that

<!-- formula-not-decoded -->

Where {|| θ ‖| 2 ≤ β k ; ∀ θ ∈ Θ } . Since we only have to demonstrate the inequality for θ k 2 and θ k 3 , we set β k as a constant greater than max( || θ k 2 || 2 , || θ k 3 || 2 ), which gives us

<!-- formula-not-decoded -->

The same argument can be applied for Q k 3 to get

<!-- formula-not-decoded -->

Thus, if we have

Then we have

Plugging in the definition of R X,Q k -1 ( θ k 2 ) , R X,Q k -1 ( θ k 3 ) in equation (204) we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now since all ( s i , a i ) are independent (205) becomes

<!-- formula-not-decoded -->

Where the expectation is now over a single ( s, a ) drawn from ν , r ( s, a ) ∼ R ( . | s, a ) and s ′ ∼ P ( . | s, a ). We re-write Equation (206) as

<!-- formula-not-decoded -->

′ ′

Where ν , µ 2 , µ 3 are the measures with respect to ( s, a ), r and s respectively Now for the integral in Equation (207) we split the integral into four different integrals. Each integral is over the set of ( s, a ) , r ′ , s ′ corresponding to the 4 different combinations of signs of A 1 , A 2.

<!-- formula-not-decoded -->

Now note that the first 2 terms are non-negative and the last two terms are non-positive. We then write the first two terms as

<!-- formula-not-decoded -->

We write the last two terms as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here C k 1 , C k 2 , C k 4 and C k 4 are positive constants. Plugging Equations (209), (210), (211), (212) into Equation (207).

<!-- formula-not-decoded -->

(214)

which implies then we get

<!-- formula-not-decoded -->

Now define ( 1+ C k 3 + C k 4 C k 1 + C k 2 ) = C k to get

<!-- formula-not-decoded -->

(216)

(218)

Therefore, we have that if the number of samples of the data set X of ( s i , a i ) pairs drawn independently from ν satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Replacing /epsilon1 with /epsilon1 ( C k ) in Equation (219), we obtain that if we get

which implies

## E.4 Proof Of Lemma 4

Proof For a given iteration k of Algorithm 1 the optimization problem to be solved in Algorithm 2 is the following

<!-- formula-not-decoded -->

Here, Q k -1 is the estimate of the Q function from the iteration k -1 and the state action pairs ( s i , a i ) i = { 1 , ··· ,n } have been sampled from a distribution over the state action

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

pairs denoted by ν . Since min θ L ( θ ) is a non convex optimization problem we instead solve the equivalent convex problem given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, X k ∈ R n k × d is the matrix of sampled state action pairs at iteration k , y k ∈ R n k × 1 is the vector of target values at iteration k . ˜ D is the set of diagonal matrices obtained from line 2 of Algorithm 2 and u ∈ R | ˜ Dd |× 1 (Note that we are treating u as a vector here for notational convenience instead of a matrix as was done in Section 4).

The constraint in Equation (226) ensures that the all the co-ordinates of the vector ∑ D i ∈ ˜ D D i X k u i are upper bounded by R max 1 -γ (since all elements of X k are between 0 and 1). This ensures that the corresponding neural network represented by Equation (14) is also upper bounded by R max 1 -γ . We use the a projected gradient descent to solve the constrained convex optimization problem which can be written as.

<!-- formula-not-decoded -->

From Ang, Andersen(2017). 'Continuous Optimization' [Notes]. https://angms.science/doc/CVX we have that if the step size α k = || u ∗ k || 2 L k √ T k +1 , after K iterations of the projected gradient descent algorithm we obtain

<!-- formula-not-decoded -->

Where L k is the lipschitz constant of g k ( u ) and u T k is the parameter estimate at step T k .

Therefore if the number of iteration of the projected gradient descent algorithm T k and the step-size α satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have

Let ( v ∗ i , w ∗ i ) i ∈ (1 , ··· , | ˜ D | ) , ( v T k i , w T k i ) i ∈ (1 , ··· , | ˜ D | ) be defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ψ ′ is defined in Equation (51). Further, we define θ ∗ | ˜ D | and θ T k as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ψ is defined in Equation (46), θ ∗ | ˜ D | = arg min θ L | ˜ D | ( θ ) for L | ˜ D | ( θ ) defined in Appendix B.

Since ( g ( u T k ) -g ( u ∗ )) ≤ /epsilon1 , then by Lemma 15, we have

<!-- formula-not-decoded -->

Note that L | ˜ D | ( θ T k ) -L | ˜ D | ( θ ∗ | ˜ D | ) is a constant value. Thus we can always find constant C ′ k such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which according to Equation (238) implies that

<!-- formula-not-decoded -->

Dividing Equation (242) by C ′ k we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assuming /epsilon1 is small enough such that /epsilon1 C ′ k &lt; 1 from lemma 16, this implies that there exists an l k &gt; 0 such that

Therefore if we have then we have

Which implies

then we have we have

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A . Equation (246) implies that if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition in section D Q k is our estimate of the Q function at the k th iteration of Algorithm 1 and thus we have Q θ T k = Q k which implies that

<!-- formula-not-decoded -->

If we replace /epsilon1 by C ′ k /epsilon1 l k in Equation (249), we get that if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Assumption 1, we have that

<!-- formula-not-decoded -->

where θ ∗ = arg min θ ∈ Θ L ( θ ) and by definition of Q k 3 in Definition 6, we have that Q k 3 = Q θ ∗ . Therefore if we have

we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->