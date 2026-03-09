## A Theoretical Analysis of Deep Q-Learning

Jianqing Fan ∗

Zhaoran Wang †

Yuchen Xie †

February 25, 2020

## Abstract

Despite the great empirical success of deep reinforcement learning, its theoretical foundation is less well understood. In this work, we make the first attempt to theoretically understand the deep Qnetwork (DQN) algorithm (Mnih et al., 2015) from both algorithmic and statistical perspectives. In specific, we focus on a slight simplification of DQN that fully captures its key features. Under mild assumptions, we establish the algorithmic and statistical rates of convergence for the actionvalue functions of the iterative policy sequence obtained by DQN. In particular, the statistical error characterizes the bias and variance that arise from approximating the action-value function using deep neural network, while the algorithmic error converges to zero at a geometric rate. As a byproduct, our analysis provides justifications for the techniques of experience replay and target network, which are crucial to the empirical success of DQN. Furthermore, as a simple extension of DQN, we propose the Minimax-DQN algorithm for zero-sum Markov game with two players. Borrowing the analysis of DQN, we also quantify the difference between the policies obtained by Minimax-DQN and the Nash equilibrium of the Markov game in terms of both the algorithmic and statistical rates of convergence.

## 1 Introduction

Reinforcement learning (RL) attacks the multi-stage decision-making problems by interacting with the environment and learning from the experiences. With the breakthrough in deep learning, deep reinforcement learning (DRL) demonstrates tremendous success in solving highly challenging problems, such as the game of Go (Silver et al., 2016, 2017), computer games (Vinyals et al., 2019), robotics (Kober and Peters, 2012), dialogue systems (Chen et al., 2017). In DRL, the value or policy functions are often represented as deep neural networks and the related deep learning techniques can be readily applied. For example, deep Q-network (DQN) (Mnih et al., 2015), asynchronous advantage actor-critic (A3C) (Mnih et al., 2016), trust region policy optimization (TRPO) (Schulman et al., 2015), proximal policy optimization (PPO) (Schulman et al., 2017) build upon

∗ Department of Operations Research and Financial Engineering, Princeton University. Research supported by the NSF grant DMS-1662139 and DMS-1712591, the ONR grant N00014-19-1-2120, and the NIH grant 2R01-GM07261114.

† Department of Industrial Engineering and Management Sciences, Northwestern University

Zhuoran Yang ∗

classical RL methods (Watkins and Dayan, 1992; Sutton et al., 2000; Konda and Tsitsiklis, 2000) and have become benchmark algorithms for artificial intelligence.

Despite its great empirical success, there exists a substantial gap between the theory and practice of DRL. In particular, most existing theoretical work on reinforcement learning focuses on the tabular case where the state and action spaces are finite, or the case where the value function is linear. Under these restrictive settings, the algorithmic and statistical perspectives of reinforcement learning are well-understood via the tools developed for convex optimization and linear regression. However, in presence of nonlinear function approximators such as deep neural network, the theoretical analysis of reinforcement learning becomes intractable as it involves solving a highly nonconvex statistical optimization problem.

To bridge such a gap in DRL, we make the first attempt to theoretically understand DQN, which can be cast as an extension of the classical Q-learning algorithm (Watkins and Dayan, 1992) that uses deep neural network to approximate the action-value function. Although the algorithmic and statistical properties of the classical Q-learning algorithm are well-studied, theoretical analysis of DQN is highly challenging due to its differences in the following two aspects.

First, in online gradient-based temporal-difference reinforcement learning algorithms, approximating the action-value function often leads to instability. Baird (1995) proves that this is the case even with linear function approximation. The key technique to achieve stability in DQN is experience replay (Lin, 1992; Mnih et al., 2015). In specific, a replay memory is used to store the trajectory of the Markov decision process (MDP). At each iteration of DQN, a mini-batch of states, actions, rewards, and next states are sampled from the replay memory as observations to train the Q-network, which approximates the action-value function. The intuition behind experience replay is to achieve stability by breaking the temporal dependency among the observations used in training the deep neural network.

Second, in addition to the aforementioned Q-network, DQN uses another neural network named the target network to obtain an unbiased estimator of the mean-squared Bellman error used in training the Q-network. The target network is synchronized with the Q-network after each period of iterations, which leads to a coupling between the two networks. Moreover, even if we fix the target network and focus on updating the Q-network, the subproblem of training a neural network still remains less well-understood in theory.

In this paper, we focus on a slight simplification of DQN, which is amenable to theoretical analysis while fully capturing the above two aspects. In specific, we simplify the technique of experience replay with an independence assumption, and focus on deep neural networks with rectified linear units (ReLU) (Nair and Hinton, 2010) and large batch size. Under this setting, DQN is reduced to the neural fitted Q-iteration (FQI) algorithm (Riedmiller, 2005) and the technique of target network can be cast as the value iteration. More importantly, by adapting the approximation results for ReLU networks to the analysis of Bellman operator, we establish the algorithmic and statistical rates of convergence for the iterative policy sequence obtained by DQN. As shown in the main results in § 3, the statistical error characterizes the bias and variance that arise from approximating the action-value function using neural network, while the algorithmic error geometrically decays to zero as the number of iteration goes to infinity.

Furthermore, we extend DQN to two-player zero-sum Markov games (Shapley, 1953). The proposed algorithm, named Minimax-DQN, can be viewed as a combination of the Minimax-Q learning algorithm for tabular zero-sum Markov games (Littman, 1994) and deep neural networks for function approximation. Compared with DQN, the main difference lies in the approaches to compute the target values. In DQN, the target is computed via maximization over the action space. In contrast, the target obtained computed by solving the Nash equilibrium of a zero-sum matrix game in Minimax-DQN, which can be efficiently attained via linear programming. Despite such a difference, both these two methods can be viewed as approximately applying the Bellman operator to the Q-network. Thus, borrowing the analysis of DQN, we also establish theoretical results for Minimax-DQN. Specifically, we quantify the suboptimality of policy returned by the algorithm by the difference between the action-value functions associated with this policy and with the Nash equilibrium policy of the Markov game. For this notion of suboptimality, we establish the both algorithmic and statistical rates of convergence, which implies that the action-value function converges to the optimal counterpart up to an unimprovable statistical error in geometric rate.

Our contribution is three-fold. First, we establish the algorithmic and statistical errors of the neural FQI algorithm, which can be viewed as a slight simplification of DQN. Under mild assumptions, our results show that the proposed algorithm obtains a sequence of Q-networks that geometrically converges to the optimal action-value function up to an intrinsic statistical error induced by the approximation bias of ReLU network and finite sample size. Second, as a byproduct, our analysis justifies the techniques of experience replay and target network used in DQN, where the latter can be viewed as a single step of the value iteration. Third, we propose the Minimax-DQN algorithm that extends DQN to two-player zero-sum Markov games. Borrowing the analysis for DQN, we establish the algorithmic and statistical convergence rates of the action-value functions associated with the sequence of policies returned by the Minimax-DQN algorithm.

## 1.1 Related Works

There is a huge body of literature on deep reinforcement learning, where these algorithms are based on Q-learning or policy gradient (Sutton et al., 2000). We refer the reader to Arulkumaran et al. (2017) for a survey of the recent developments of DRL. In addition, the DQN algorithm is first proposed in Mnih et al. (2015), which applies DQN to Artari 2600 games (Bellemare et al., 2013). The extensions of DQN include double DQN (van Hasselt et al., 2016), dueling DQN (Wang et al., 2016), deep recurrent Q-network (Hausknecht and Stone, 2015), asynchronous DQN (Mnih et al., 2016), and variants designed for distributional reinforcement learning (Bellemare et al., 2017; Dabney et al., 2018b,a). All of these algorithms are corroborated only by numerical experiments, without theoretical guarantees. Moreover, these algorithms not only inherit the tricks of experience replay and the target network proposed in the original DQN, but develop even more tricks to enhance the performance. Furthermore, recent works such as Schaul et al. (2016); Andrychowicz et al. (2017); Liu and Zou (2018); Zhang and Sutton (2017); Novati and Koumoutsakos (2019) study the effect of experience replay and propose various modifications.

In addition, our work is closely related to the literature on batch reinforcement learning (Lange et al., 2012), where the goal is to estimate the value function given transition data. These problems

are usually formulated into least-squares regression, for which various algorithms are proposed with finite-sample analysis. However, most existing works focus on the settings where the value function are approximated by linear functions. See Bradtke and Barto (1996); Boyan (2002); Lagoudakis and Parr (2003); Lazaric et al. (2016); Farahmand et al. (2010); Lazaric et al. (2012); Tagorti and Scherrer (2015) and the references therein for results of the least-squares policy iteration (LSPI) and Bellman residue minimization (BRM) algorithms. Beyond linear function approximation, a recent work (Farahmand et al., 2016) studies the performance of LSPI and BRM when the value function belongs to a reproducing kernel Hilbert space. However, we study the fitted Qiteration algorithm, which is a batch RL counterpart of DQN. The fitted Q-iteration algorithm is proposed in Ernst et al. (2005), and Riedmiller (2005) proposes the neural FQI algorithm. Finitesample bounds for FQI have been established in Murphy (2005); Munos and Szepesv´ ari (2008) for large classes of regressors. However, their results are not applicable to ReLU networks due to the huge capacity of deep neural networks. Furthermore, various extensions of FQI are studied in Antos et al. (2008a); Farahmand et al. (2009); Tosatto et al. (2017); Geist et al. (2019) to handle continuous actions space, ensemble learning, and entropy regularization. The empirical performances of various batch RL methods have been examined in Levine et al. (2017); Agarwal et al. (2019); Fujimoto et al. (2019).

Moreover, Q-learning, and reinforcement learning methods in general, have been widely applied to dynamic treatment regimes (DTR) (Chakraborty, 2013; Laber et al., 2014; Tsiatis, 2019), where the goal is to find sequential decision rules for individual patients that adapt to time-evolving illnesses. There is a huge body of literature on this line of research. See, e.g., Murphy (2003); Zhao et al. (2009); Qian and Murphy (2011); Zhao et al. (2011); Zhang et al. (2012); Zhao et al. (2012); Goldberg and Kosorok (2012); Nahum-Shani et al. (2012); Goldberg et al. (2013); Schulte et al. (2014); Song et al. (2015); Zhao et al. (2015); Linn et al. (2017); Zhou et al. (2017); Shi et al. (2018); Zhu et al. (2019) and the references therein. Our work provides a theoretical underpinning for the application of DQN to DTR (Liu et al., 2019b) and motivates the principled usage of DRL methods in healthcare applications (Yu et al., 2019).

Furthermore, our work is also related to works that apply reinforcement learning to zero-sum Markov games. The Minimax-Q learning is proposed by Littman (1994), which is an online algorithm that is an extension Q-learning. Subsequently, for Markov games, various online algorithms are also proposed with theoretical guarantees. These works consider either the tabular case or linear function approximation. See, e.g., Bowling (2001); Conitzer and Sandholm (2007); Prasad et al. (2015); Wei et al. (2017); P´ erolat et al. (2018); Srinivasan et al. (2018); Wei et al. (2017) and the references therein. In addition, batch reinforcement learning is also applied to zero-sum Markov games by Lagoudakis and Parr (2002); P´ erolat et al. (2015, 2016a,b); Zhang et al. (2018), which are closely related to our work. All of these works consider either linear function approximation or a general function class with bounded pseudo-dimension (Anthony and Bartlett, 2009). However, there results cannot directly imply finite-sample bounds for Minimax-DQN due to the huge capacity of deep neural networks.

Finally, our work is also related a line of research on the model capacity of ReLU deep neural networks, which leads to understanding the generalization property of deep learning (Mohri et al.,

2012; Kawaguchi et al., 2017). Specifically, Bartlett (1998); Neyshabur et al. (2015b,a); Bartlett et al. (2017); Golowich et al. (2018); Liang et al. (2019) propose various norms computed from the networks parameters and establish capacity bounds based upon these norms. In addition, Maass (1994); Bartlett et al. (1999); Schmidt-Hieber (2020+); Bartlett et al. (2019); Klusowski and Barron (2016); Barron and Klusowski (2018); Suzuki (2019); Bauer et al. (2019) study the Vapnik-Chervonenkis (VC) dimension of neural networks and Dziugaite and Roy (2017); Neyshabur et al. (2018) establish the PAC-Bayes bounds for neural networks. Among these works, our work is more related to Schmidt-Hieber (2020+); Suzuki (2019), which relate the VC dimension of the ReLU networks to a set of hyperparameters used to define the networks. Based on the VC dimension, they study the statistical error of nonparametric regression using ReLU networks. In sum, theoretical understanding of deep learning is pertinent to the study of DRL algorithms. See Kawaguchi et al. (2017); Neyshabur et al. (2017); Fan et al. (2019) and the references therein for recent developments on theoretical analysis of the generalization property of deep learning.

## 1.2 Notation

For a measurable space with domain S , we denote by B ( S , V ) the set of measurable functions on S that are bounded by V in absolute value. Let P ( S ) be the set of all probability measures over S . For any ν ∈ P ( S ) and any measurable function f : S → R , we denote by ‖ f ‖ ν,p the /lscript p -norm of f with respect to measure ν for p ≥ 1. In addition, for simplicity, we write ‖ f ‖ ν for ‖ f ‖ 2 ,ν . In addition, let { f ( n ) , g ( n ) } n ≥ 1 be two positive series. We write f ( n ) /lessorsimilar g ( n ) if there exists a constant C such that f ( n ) ≤ C · g ( n ) for all n larger than some n 0 ∈ N . In addition, we write f ( n ) /equivasymptotic g ( n ) if f ( n ) /lessorsimilar g ( n ) and g ( n ) /lessorsimilar f ( n ).

## 2 Background

In this section, we introduce the background. We first lay out the formulation of the reinforcement learning problem, and then define the family of ReLU neural networks.

## 2.1 Reinforcement Learning

A discounted Markov decision process is defined by a tuple ( S , A , P, R, γ ). Here S is the set of all states, which can be countable or uncountable, A is the set of all actions, P : S × A → P ( S ) is the Markov transition kernel, R : S × A → P ( R ) is the distribution of the immediate reward, and γ ∈ (0 , 1) is the discount factor. In specific, upon taking any action a ∈ A at any state s ∈ S , P ( · | s, a ) defines the probability distribution of the next state and R ( · | s, a ) is the distribution of the immediate reward. Moreover, for regularity, we further assume that S is a compact subset of R r which can be infinite, A = { a 1 , a 2 , . . . , a M } has finite cardinality M , and the rewards are uniformly bounded by R max , i.e., R ( · | s, a ) has a range on [ -R max , R max ] for any s ∈ S and a ∈ A .

A policy π : S → P ( A ) for the MDP maps any state s ∈ S to a probability distribution π ( · | s ) over A . For a given policy π , starting from the initial state S 0 = s , the actions, rewards, and states

evolve according to the law as follows:

<!-- formula-not-decoded -->

and the corresponding value function V π : S → R is defined as the cumulative discounted reward obtained by taking the actions according to π when starting from a fixed state, that is,

<!-- formula-not-decoded -->

The policy π can be controlled by decision makers, yet the functions P and R are given by the nature or the system that are unknown to decision makers.

By the law of iterative expectation, for any policy π , where Q π ( s, a ), called an action value function, is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with r ( s, a ) = ∫ rR (d r | s, a ) is the expected reward at state s given action a . Moreover, we define an operator P π by and define the Bellman operator T π by ( T π Q )( s, a ) = r ( s, a ) + γ · ( P π Q )( s, a ) . Then Q π in (2.3) is the unique fixed point of T π .

The goal of reinforcement learning is to find the optimal policy, which achieves the largest cumulative reward via dynamically learning from the acquired data. To characterize optimality, by (2.2), we naturally define the optimal action-value function Q ∗ as

<!-- formula-not-decoded -->

where the supremum is taken over all policies. In addition, for any given action-value function Q : S×A → R , we define the greedy policy π Q as any policy that selects the action with the largest Q -value, that is, for any s ∈ S , π Q ( · | s ) satisfies

/negationslash

<!-- formula-not-decoded -->

Based on Q ∗ , we define the optimal policy π ∗ as any policy that is greedy with respect to Q ∗ . It can be shown that Q ∗ = Q π ∗ . Finally, we define the Bellman optimality operator T via

<!-- formula-not-decoded -->

Then we have the Bellman optimality equation TQ ∗ = Q ∗ .

Furthermore, it can be verified that the Bellman operator T is γ -contractive with respect to the supremum norm over S × A . That is, for any two action-value functions Q,Q ′ : S × A → R , it holds that ‖ TQ -TQ ′ ‖ ∞ ≤ γ · ‖ Q -Q ′ ‖ ∞ . Such a contraction property yields the celebrated value iteration algorithm (Sutton and Barto, 2011), which constructs a sequence of action-value functions { Q k } k ≥ 0 by letting Q k = TQ k -1 for all k ≥ 1, where the initialization function Q 0 is arbitrary. Then it holds that ‖ Q k -Q ∗ ‖ ∞ ≤ γ k ·‖ Q 0 -Q ∗ ‖ ∞ , i.e., { Q k } k ≥ 0 converges to the optimal value function at a linear rate. This approach forms the basis of the neural FQI algorithm, where the Bellman operator is empirically learned from a batch of data dynamically and the action-value functions are approximated by deep neural networks.

## 2.2 Deep Neural Network

We study the performance of DQN with rectified linear unit (ReLU) activation function σ ( u ) = max( u, 0). For any positive integer L and { d j } L +1 i =0 ⊆ N , a ReLU network f : R d 0 → R d L +1 with L hidden layers and width { d j } L +1 i =0 is of form

<!-- formula-not-decoded -->

where W /lscript ∈ R d /lscript × d /lscript -1 and v /lscript ∈ R d /lscript are the weight matrix and the shift (bias) vector in the /lscript -th layer, respectively. Here we apply σ to to each entry of its argument in (2.8). In deep learning, the network structure is fixed, and the goal is to learn the network parameters (weights) { W /lscript , v /lscript } /lscript ∈ [ L +1] with the convention that v L +1 = 0. For deep neural networks, the number of parameters greatly exceeds the input dimension d 0 . To restrict the model class, we focus on the class of ReLU networks where most parameters are zero.

Definition 2.1 (Sparse ReLU Network) . For any L, s ∈ N , { d j } L +1 i =0 ⊆ N , and V &gt; 0, the family of sparse ReLU networks bounded by V with L hidden layers, network width d , and weight sparsity s is defined as

<!-- formula-not-decoded -->

where we denote ( W /lscript , v /lscript ) by ˜ W /lscript . Moreover, f in (2.9) is expressed as in (2.8), and f j is the j -th component of f .

Here we focus on functions that are uniformly bounded because the value functions in (2.1) and (2.3) are always bounded by V max = R max / (1 -γ ). We also assume that the network weights are uniformly bounded and bounded by one without loss of generality. In the sequel, we write F ( L, { d j } L +1 j =0 , s, V max ) as F ( L, { d j } L +1 j =0 , s ) to simplify the notation. In addition, we restrict the networks weights to be sparse, i.e., s is much smaller compared with the total number of parameters. Such an assumption implies that the network has sparse connections, which are useful for applying deep learning in memory-constrained situations such as mobile devices (Han et al., 2016; Liu et al., 2015). Empirically, sparse neural networks are realized via various regularization techniques such as Dropout (Srivastava et al., 2014), which randomly sets a fixed portion of the

network weights to zero. Moreover, sparse network architectures have recently been advocated by the intriguing lottery ticket hypothesis (Frankle and Carbin, 2019), which states that each dense network has a subnetwork with the sparse connections, when trained in isolation, achieves comparable performance as the original network. Thus, focusing on the class of sparse ReLU networks does not sacrifice the statistical accuracy.

Moreover, we introduce the notion of H¨ older smoothness as follows, which is a generalization of Lipschitz continuity, and is widely used to characterize the regularity of functions.

Definition 2.2 (H¨ older Smooth Function) . Let D be a compact subset of R r , where r ∈ N . We define the set of H¨ older smooth functions on D as

/negationslash

<!-- formula-not-decoded -->

where β &gt; 0 and H &gt; 0 are parameters and /floorleft β /floorright is the largest integer no greater than β . In addition, here we use the multi-index notation by letting α = ( α 1 , . . . , α r ) /latticetop ∈ N r , and ∂ α = ∂ α 1 . . . ∂ α r .

Finally, we conclude this section by defining functions that can be written as a composition of multiple H¨ older functions, which captures complex mappings in real-world applications such as multi-level feature extraction.

Definition 2.3 (Composition of H¨ older Functions) . Let q ∈ N and { p j } j ∈ [ q ] ⊆ N be integers, and let { a j , b j } j ∈ [ q ] ⊆ R such that a j &lt; b j j ∈ [ q ]. Moreover, let g j : [ a j , b j ] p j → [ a j +1 , b j +1 ] p j +1 be a function, ∀ j ∈ [ q ]. Let ( g jk ) k ∈ [ p j +1 ] be the components of g j , and we assume that each g jk is H¨ older smooth, and depends on at most t j of its input variables, where t j could be much smaller than p j , i.e., g jk ∈ C t j ([ a j , b j ] t j , β j , H j ). Finally, we denote by G ( { p j , t j , β j , H j } j ∈ [ q ] ) the family of functions that can be written as compositions of { g j } j ∈ [ q ] , with the convention that p q +1 = 1. That is, for any f ∈ G ( { p j , t j , β j , H j } j ∈ [ q ] ), we can write

<!-- formula-not-decoded -->

with g jk ∈ C t j ([ a j , b j ] t j , β j , H j ) for each k ∈ [ p j +1 ] and j ∈ [ q ].

Here f in (2.10) is a composition of q vector-valued mappings { g j } j ∈ [ q ] where each g j has p j +1 components and its k -th component, g jk , ∀ k ∈ [ p j +1 ], is a H¨ older smooth function defined on [ a j , b j ] p j . Moreover, it is well-known the statistical rate for estimating a H¨ older smooth function depends on the input dimension (Tsybakov, 2008). Here we assume that g jk only depends on t j of its inputs, where t j ∈ [ p j ] can be much smaller than p j , which enables us to obtain a more refined analysis that adapts to the effective smoothness of f . In particular, Definition 2.3 covers the family of H¨ older smooth functions and the additive model (Friedman and Stuetzle, 1981) on [0 , 1] r as two special cases, where the former suffers from the curse of dimensionality whereas the latter does not.

## 3 Understanding Deep Q-Network

In the DQN algorithm, a deep neural network Q θ : S×A→ R is used to approximate Q ∗ , where θ is the parameter. For completeness, we state DQN as Algorithm 3 in § A. As shown in the experiments in (Mnih et al., 2015), two tricks are pivotal for the empirical success of DQN.

First, DQN use the trick of experience replay (Lin, 1992). Specifically, at each time t , we store the transition ( S t , A t , R t , S t +1 ) into the replay memory M , and then sample a minibatch of independent samples from M to train the neural network via stochastic gradient descent. Since the trajectory of MDP has strong temporal correlation, the goal of experience replay is to obtain uncorrelated samples, which yields accurate gradient estimation for the stochastic optimization problem.

Another trick is to use a target network Q θ /star with parameter θ /star (current estimate of parameter). With independent samples { ( s i , a i , r i , s ′ i ) } i ∈ [ n ] from the replay memory (we use s ′ i instead of s i +1 for the next state right after s i and a i to avoid notation crash with next independent sample s i +1 in the state space), to update the parameter θ of the Q-network, we compute the target

<!-- formula-not-decoded -->

(compare with Bellman optimality operator (2.7)), and update θ by the gradient of

<!-- formula-not-decoded -->

Whereas parameter θ /star is updated once every T target steps by letting θ /star = θ . That is, the target network is hold fixed for T target steps and then updated it by the current weights of the Q-network.

To demystify DQN, it is crucial to understand the role played by these two tricks. For experience replay, in practice, the replay memory size is usually very large. For example, the replay memory size is 10 6 in Mnih et al. (2015). Moreover, DQN use the /epsilon1 -greedy policy, which enables exploration over S × A . Thus, when the replay memory is large, experience replay is close to sampling independent transitions from an explorative policy. This reduces the variance of the ∇ L ( θ ), which is used to update θ . Thus, experience replay stabilizes the training of DQN, which benefits the algorithm in terms of computation.

To understand the statistical property of DQN, we replace the experience replay by sampling independent transitions from a given distribution σ ∈ P ( S ×A ). That is, instead of sampling from the replay memory, we sample i.i.d. observations { ( S i , A i ) } i ∈ [ n ] from σ . Moreover, for any i ∈ [ n ], let R i and S ′ i be the immediate reward and the next state when taking action A i at state S i . Under this setting, we have E ( Y i | S i , A i ) = ( TQ θ /star )( S i , A i ) , where T is the Bellman optimality operator in (2.7) and Q θ /star is the target network.

Furthermore, to further understand the necessity of the target network, let us first neglect the target network and set θ /star = θ . Using bias-variance decomposition, the the expected value of L ( θ ) in (3.1) is

<!-- formula-not-decoded -->

Here the first term in (3.2) is known as the mean-squared Bellman error (MSBE), and the second term is the variance of Y 1 . Whereas L ( θ ) can be viewed as the empirical version of the MSBE, which has bias E { [ Y 1 -( TQ θ )( S 1 , A 1 )] 2 } that also depends on θ . Thus, without the target network, minimizing L ( θ ) can be drastically different from minimizing the MSBE.

To resolve this problem, we use a target network in (3.1), which has expectation

<!-- formula-not-decoded -->

where the variance of Y 1 does not depend on θ . Thus, minimizing L ( θ ) is close to solving

<!-- formula-not-decoded -->

where Θ is the parameter space. Note that in DQN we hold θ /star still and update θ for T target steps. When T target is sufficiently large and we neglect the fact that the objective in (3.3) is nonconvex, we would update θ by the minimizer of (3.3) for fixed θ /star .

Therefore, in the ideal case, DQN aims to solve the minimization problem (3.3) with θ /star fixed, and then update θ /star by the minimizer θ . Interestingly, this view of DQN offers a statistical interpretation of the target network. In specific, if { Q θ : θ ∈ Θ } is sufficiently large such that it contains TQ θ /star , then (3.3) has solution Q θ = TQ θ /star , which can be viewed as one-step of value iteration (Sutton and Barto, 2011) for neural networks. In addition, in the sample setting, Q θ /star is used to construct { Y i } i ∈ [ n ] , which serve as the response in the regression problem defined in (3.1), with ( TQ θ /star ) being the regression function.

Furthermore, turning the above discussion into a realizable algorithm, we obtain the neural fitted Q-iteration (FQI) algorithm, which generates a sequence of value functions. Specifically, let F be a class of function defined on S × A . In the k -th iteration of FQI, let ˜ Q k be current estimate of Q ∗ . Similar to (3.1) and (3.3), we define Y i = R i + γ · max a ∈A Q k ( S ′ i , a ), and update Q k by

This gives the fitted-Q iteration algorithm, which is stated in Algorithm 1.

<!-- formula-not-decoded -->

The step of minimization problem in (3.4) essentially finds ˜ Q k +1 in F such that ˜ Q k +1 ≈ T ˜ Q k . Let us denote ˜ Q k +1 = ̂ T k ˜ Q k where ̂ T k is an approximation of the Bellman optimality operator T learned from the training data in the k -th iteration. With the above notation, we can now understand our Algorithm 1 as follows. Starting from the initial estimator ˜ Q 0 , collect the data { ( S i , A i , R i , S ′ i ) } i ∈ [ n ] and learn the map ̂ T 1 via (3.4) and get ˜ Q 1 = ̂ T 1 ˜ Q 0 . Then, get a new batch of sample and learn the map ̂ T 2 and get ˜ Q 2 = ̂ T 2 ˜ Q 1 , and so on. Our final estimator of the action value is ˜ Q K = ̂ T K · · · ̂ T 1 ˜ Q 0 , which resembles the updates of the value iteration algorithm at the population level.

When F is the family of neural networks, Algorithm 1 is known as the neural FQI algorithm, which is proposed in Riedmiller (2005). Thus, we can view neural FQI as a modification of DQN, where we replace experience replay with sampling from a fixed distribution σ , so as to understand the statistical property. As a byproduct, such a modification naturally justifies the trick of target network in DQN. In addition, note that the optimization problem in (3.4) appears in each iteration of FQI, which is nonconvex when neural networks are used. However, since we focus solely on the statistical aspect, we make the assumption that the global optima of (3.4) can be reached, which is also contained F . Interestingly, a recent line of research on deep learning (Du et al., 2019b,a;

Zou et al., 2018; Chizat et al., 2019; Allen-Zhu et al., 2019a,b; Jacot et al., 2018; Cao and Gu, 2019; Arora et al., 2019; Weinan et al., 2019; Mei et al., 2019; Yehudai and Shamir, 2019) has established global convergence of gradient-based algorithms for empirical risk minimization when the neural networks are overparametrized. We provide more discussions on the computation aspect in § B. Furthermore, we make the i.i.d. assumption in Algorithm 1 to simplify the analysis. Antos et al. (2008b) study the performance of fitted value iteration with fixed data used in the regression sub-problems repeatedly, where the data is sampled from a single trajectory based on a fixed policy such that the induced Markov chain satisfies certain conditions on the mixing time. Using similar analysis as in Antos et al. (2008b), our algorithm can also be extended to handled fixed data that is collected beforehand.

## Algorithm 1 Fitted Q-Iteration Algorithm

Input: MDP ( S , A , P, R, γ ), function class F , sampling distribution σ , number of iterations K , number of samples n , the initial estimator ˜ Q 0 . for k = 0 , 1 , 2 , . . . , K -1 do

Sample i.i.d. observations { ( S i , A i , R i , S ′ i ) } i ∈ [ n ] with ( S i , A i ) drawn from distribution σ . Compute Y i = R i + γ · max a ∈A ˜ Q k ( S ′ i , a ). Update the action-value function:

end for

<!-- formula-not-decoded -->

Define policy π K as the greedy policy with respect to ˜ Q K .

Output: An estimator ˜ Q K of Q ∗ and policy π K .

## 4 Theoretical Results

We establish statistical guarantees for DQN with ReLU networks. Specifically, let Q π K be the action-value function corresponding to π K , which is returned by Algorithm 1. In the following, we obtain an upper bound for ‖ Q π K -Q ∗ ‖ 1 ,µ , where µ ∈ P ( S × A ) is allowed to be different from ν . In addition, we assume that the state space S is a compact subset in R r and the action space A is finite. Without loss of generality, we let S = [0 , 1] r hereafter, where r is a fixed integer. To begin with, we first specify the function class F in Algorithm 1.

<!-- formula-not-decoded -->

Definition 4.1 (Function Classes) . Following Definition 2.1, let F ( L, { d j } L +1 j =0 , s ) be the family of sparse ReLU networks defined on S with d 0 = r and d L +1 = 1. Then we define F 0 by

In addition, let G ( { p j , t j , β j , H j } j ∈ [ q ] ) be set of composition of H¨ older smooth functions defined on S ⊆ R r . Similar to F 0 , we define a function class G 0 as

<!-- formula-not-decoded -->

By this definition, for any function f ∈ F 0 and any action a ∈ A , f ( · , a ) is a ReLU network defined on S , which is standard for Q-networks. Moreover, G 0 contains a broad family of smooth functions on S × A . In the following, we make a mild assumption on F 0 and G 0 .

Assumption 4.2. We assume that for any f ∈ F 0 , we have Tf ∈ G 0 , where T is the Bellman optimality operator defined in (2.7). That is, for any f ∈ F and any a ∈ A , ( Tf )( s, a ) can be written as compositions of H¨ older smooth functions as a function of s ∈ S .

This assumption specifies that the target function T ˜ Q k in each FQI step stays in function class G 0 . When G 0 can be approximated by functions in F 0 accurately, this assumption essentially implies that Q ∗ is close to F 0 and that F 0 is approximately closed under Bellman operator T . Such an completeness assumption is commonly made in the literature on batch reinforcement learning under various forms and is conjectured to be indispensable in Chen and Jiang (2019).

We remark that this Assumption (4.2) holds when the MDP satisfies some smoothness conditions. For any state-action pair ( s, a ) ∈ S × A , let P ( · | s, a ) be the density of the next state. By the definition of the Bellman optimality operator in (2.7), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any s ′ ∈ S and a ∈ A , we define functions g 1 , g 2 by letting g 1 ( s ) = r ( s, a ) and g 2 ( s ) = P ( s ′ | s, a ). Suppose both g 1 and g 2 are H¨ older smooth functions on S = [0 , 1] r with parameters β and H . Since ‖ f ‖ ∞ ≤ V max , by changing the order of integration and differentiation with respect to s in (4.3), we obtain that function s → ( Tf )( s, a ) belongs to the H¨ older class C r ( S , β, H ′ ) with H ′ = H (1+ V max ). Furthermore, in the more general case, suppose for any fixed a ∈ A , we can write P ( s ′ | s, a ) as h 1 [ h 2 ( s, a ) , h 3 ( s ′ )], where h 2 : S → R r 1 , and h 3 : S → R r 2 can be viewed as feature mappings, and h 1 : R r 1 + r 2 → R is a bivariate function. We define function h 4 : R r 1 → R by

Then by (4.3) we have ( Tf )( s, a ) = g 1 ( s ) + h 4 ◦ h 2 ( s, a ). Then Assumption 4.2 holds if h 4 is H¨ older smooth and both g 1 and h 2 can be represented as compositions of H¨ older functions. Thus, Assumption 4.2 holds if both the reward function and the transition density of the MDP are sufficiently smooth.

Moreover, even when the transition density is not smooth, we could also expect Assumption 4.2 to hold. Consider the extreme case where the MDP has deterministic transitions, that is, the next state s ′ is a function of s and a , which is denoted by s ′ = h ( s, a ). In this case, for any ReLU network f , we have ( Tf )( s, a ) = r ( s, a ) + γ · max a ′ ∈A f [ h ( s, a ) , a ′ ] . Since for any s 1 , s 2 ∈ S , and network f ( · , a ) is Lipschitz continuous for any fixed a ∈ A , function m 1 ( s ) = max a ′ f ( s, a ′ ) is Lipschitz on S . Thus, for any fixed a ∈ A , if both g 1 ( s ) = r ( s, a ) and m 2 ( s ) = h ( s, a ) are compositions of H¨ older functions, so is ( Tf )( s, a ) = g 1 ( s ) + m 1 ◦ m 2 ( s ).

<!-- formula-not-decoded -->

Therefore, even if the MDP has deterministic dynamics, when both the reward function r ( s, a ) and the transition function h ( s, a ) are sufficiently nice, Assumption 4.2 still holds true.

In the following, we define the concentration coefficients, which measures the similarity between two probability distributions under the MDP.

Assumption 4.3 (Concentration Coefficients) . Let ν 1 , ν 2 ∈ P ( S×A ) be two probability measures that are absolutely continuous with respect to the Lebesgue measure on S × A . Let { π t } t ≥ 1 be a sequence of policies. Suppose the initial state-action pair ( S 0 , A 0 ) of the MDP has distribution ν 1 , and we take action A t according to policy π t . For any integer m , we denote by P π m P π m -1 · · · P π 1 ν 1 the distribution of { ( S t , A t ) } m t =0 . Then we define the m -th concentration coefficient as where the supremum is taken over all possible policies.

Furthermore, let σ be the sampling distribution in Algorithm 1 and let µ be a fixed distribution on S × A . We assume that there exists a constant φ µ,σ &lt; ∞ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition, concentration coefficients in (4.4) quantifies the similarity between ν 2 and the distribution of the future states of the MDP when starting from ν 1 . Moreover, (4.5) is a standard assumption in the literature. See, e.g., Munos and Szepesv´ ari (2008); Lazaric et al. (2016); Scherrer et al. (2015); Farahmand et al. (2010, 2016). This assumption holds for large class of systems MDPs and specifically for MDPs whose top-Lyapunov exponent is finite. Moreover, this assumption essentially requires that the sampling distribution σ has sufficient coverage over S ×A and is shown to be necessary for the success of batch RL methods (Chen and Jiang, 2019). See Munos and Szepesv´ ari (2008); Antos et al. (2007); Chen and Jiang (2019) for more detailed discussions on this assumption.

where (1 -γ ) 2 in (4.5) is a normalization term, since ∑ m ≥ 1 γ m -1 · m = (1 -γ ) -2 .

Now we are ready to present the main theorem.

Theorem 4.4. Under Assumptions 4.2 and 4.3, let F 0 be defined in (4.1) based on the family of sparse ReLU networks F ( L ∗ , { d ∗ j } L ∗ +1 j =0 , s ∗ ) and let G 0 be given in (4.2) with { H j } j ∈ [ q ] being absolute constants. Moreover, for any j ∈ [ q -1], we define β ∗ j = β j · ∏ /lscript = j +1 min( β /lscript , 1); let β ∗ q = 1. In addition, let α ∗ = max j ∈ [ q ] t j / (2 β ∗ j + t j ) . For the parameters of G 0 , we assume that the sample size n is sufficiently large such that there exists a constant ξ &gt; 0 satisfying

<!-- formula-not-decoded -->

For the hyperparameters L ∗ , { d ∗ j } L ∗ +1 j =0 , and s ∗ of the ReLU network, we set d ∗ 0 = 0 and d ∗ L ∗ +1 = 1. Moreover, we set

<!-- formula-not-decoded -->

for some constant ξ ∗ &gt; 1 + 2 ξ . For any K ∈ N , let Q π K be the action-value function corresponding to policy π K , which is returned by Algorithm 1 based on function class F 0 . Then there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

This theorem implies that the statistical rate of convergence is the sum of a statistical error and an algorithmic error. The algorithmic error converges to zero in linear rate as the algorithm proceeds, whereas the statistical error reflects the fundamental difficulty of the problem. Thus, when the number of iterations satisfy

<!-- formula-not-decoded -->

iterations, where C ′ is a sufficiently large constant, the algorithmic error is dominated by the statistical error. In this case, if we view both γ and φ µ,σ as constants and ignore the polylogarithmic term, Algorithm 1 achieves error rate

<!-- formula-not-decoded -->

which scales linearly with the capacity of the action space, and decays to zero when the n goes to infinity. Furthermore, the rates { n -β ∗ j / (2 β ∗ j + t j ) } j ∈ [ q ] in (4.9) recovers the statistical rate of nonparametric regression in /lscript 2 -norm, whereas our statistical rate n ( α ∗ -1) / 2 in (4.9) is the fastest among these nonparametric rates, which illustrates the benefit of compositional structure of G 0 .

Furthermore, as a concrete example, we assume that both the reward function and the Markov transition kernel are H¨ older smooth with smoothness parameter β . As stated below Assumption 4.2, for any f ∈ F 0 , we have ( Tf )( · , a ) ∈ C r ( S , β, H ′ ). Then Theorem 4.4 implies that Algorithm 1 achieves error rate |A|· n -β/ (2 β + r ) when K is sufficiently large. Since |A| is finite, this rate achieves the minimax-optimal statistical rate of convergence within the class of H¨ older smooth functions defined on [0 , 1] d (Stone, 1982) and thus cannot be further improved. As another example, when ( Tf )( · , a ) ∈ C r ( S , β, H ′ ) can be represented as an additive model over [0 , 1] r where each component has smoothness parameter β , (4.9) reduces to |A|· n -β/ (2 β +1) , which does not depends on the input dimension r explicitly. Thus, by having a composite structure in G 0 , Theorem 4.4 yields more refined statistical rates that adapt to the intrinsic difficulty of solving each iteration of Algorithm 1.

In the sequel, we conclude this section by sketching the proof of Theorem 4.4; the detailed proof is deferred to § 6.

Proof Sketch of Theorem 4.4. Recall that π k is the greedy policy with respect to ˜ Q k and Q π K is the action-value function associated with π K , whose definition is given in (2.3). Since { ˜ Q k } k ∈ [ K ] is constructed by a iterative algorithm, it is crucial to relate ‖ Q ∗ -Q π K ‖ 1 ,µ , the quantity of interest, to the errors incurred in the previous steps, namely { ˜ Q k -T ˜ Q k -1 } k ∈ [ K ] . Thus, in the first step of the proof, we establish Theorem 6.1, also known as the error propagation (Munos and Szepesv´ ari, 2008; Lazaric et al., 2016; Scherrer et al., 2015; Farahmand et al., 2010, 2016) in the batch reinforcement

learning literature, which provides an upper bound on ‖ Q ∗ -Q π K ‖ 1 ,µ using {‖ ˜ Q k -T ˜ Q k -1 ‖ σ } k ∈ [ K ] . In particular, Theorem 6.1 asserts that where φ µ,σ , given in (4.5), is a constant that only depends on distributions µ and σ .

<!-- formula-not-decoded -->

The upper bound in (4.10) shows that the total error of Algorithm 1 can be viewed as a sum of a statistical error and an algorithmic error, where max k ∈ [ K ] ‖ ˜ Q k -T ˜ Q k -1 ‖ σ is essentially the statistical error and the second term on the right-hand side of (4.10) corresponds to the algorithmic error. Here, the statistical error diminishes as the sample size n in each iteration grows, whereas the algorithmic error decays to zero geometrically as the number of iterations K increases. This implies that the fundamental difficulty of DQN is captured by the error incurred in a single step. Moreover, the proof of this theorem depends on bounding ˜ Q /lscript -T ˜ Q /lscript -1 using ˜ Q k -T ˜ Q k -1 for any k &lt; /lscript , which characterizes how the one-step error ˜ Q k -T ˜ Q k -1 propagates as the algorithm proceeds. See § C.1 for a detailed proof.

It remains to bound ‖ ˜ Q k -T ˜ Q k -1 ‖ σ for any k ∈ [ K ]. We achieve such a goal using tools from nonparametric regression. Specifically, as we will show in Theorem 6.2, under Assumption 4.2, for any k ∈ [ K ] we have for any δ &gt; 0, where C &gt; 0 is an absolute constant,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is the /lscript ∞ -error of approximating functions in G 0 using functions in F 0 , and N δ is the minimum number of cardinality of the balls required to cover F 0 with respect to /lscript ∞ -norm.

In the sequel, we fix δ = 1 /n in (4.11), which implies that

Furthermore, (4.11) characterizes the bias and variance that arise in estimating the action-value functions using deep ReLU networks. Specifically, [dist ∞ ( F 0 , G 0 )] 2 corresponds to the bias incurred by approximating functions in G 0 using ReLU networks. We note that such a bias is measured in the /lscript ∞ -norm. In addition, V 2 max /n · log N δ + V max · δ controls the variance of the estimator.

<!-- formula-not-decoded -->

where C ′ &gt; 0 is an absolute constant.

In the subsequent proof, we establish upper bounds for dist( F 0 , G 0 ) defined in (4.12) and log N δ , respectively. Recall that the family of composite H¨ older smooth functions G 0 is defined in (4.2).

By the definition of G 0 in (4.2), for any f ∈ G 0 and any a ∈ A , f ( · , a ) ∈ G ( { ( p j , t j , β j , H j ) } j ∈ [ q ] ) is a composition of H¨ older smooth functions, that is, f ( · , a ) = g q ◦· · · ◦ g 1 . Recall that, as defined in Definition 2.3, g jk is the k -th entry of the vector-valued function g j . Here g jk ∈ C t j ([ a j , b j ] t j , β j , H j ) for each k ∈ [ p j +1 ] and j ∈ [ q ]. To construct a ReLU network that is f ( · , a ), we first show that f ( · , a )

can be reformulated as a composition of H¨ older functions defined on a hypercube. Specifically, let h 1 = g 1 / (2 H 1 ) + 1 / 2, h q ( u ) = g q (2 H q -1 u -H q -1 ), and

<!-- formula-not-decoded -->

for all j ∈ { 2 , . . . , q -1 } . Then we immediately have

<!-- formula-not-decoded -->

Furthermore, by the definition of H¨ older smooth functions in Definition 2.2, for any j ∈ [ q ] and k ∈ [ p j +1 ], it is not hard to verify that h jk ∈ C t j ( [0 , 1] t j , W ) , where we define W &gt; 0 by

<!-- formula-not-decoded -->

Now we employ Lemma 6.3, obtained from Schmidt-Hieber (2020+), to construct a ReLU network that approximates each h jk , which, combined with (4.14), yields a ReLU network that is close to f ( · , a ) in the /lscript ∞ -norm.

<!-- formula-not-decoded -->

To apply Lemma 6.3 we set m = η · /ceilingleft log 2 n /ceilingright for a sufficiently large constant η &gt; 1, and set N to be a sufficiently large integer that depends on n , which will be specified later. In addition, we set L j = 8 + ( m +5) · (1 + /ceilingleft log 2 ( t j + β j ) /ceilingright ) . Then, by Lemma 6.3, there exists a ReLU network ˜ h jk such that ‖ ˜ h jk -h jk ‖ ∞ /lessorsimilar N -β j /t j . Furthermore, we have ˜ h jk ∈ F ( L j , { t j , ˜ d j , . . . , ˜ d j , 1 } , s j ) , with

<!-- formula-not-decoded -->

˜ Now we define ˜ f : S → R by ˜ f = ˜ h q ◦ · · · ◦ h 1 and set

For this choice of N , we show that ˜ f belongs to function class F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ). Moreover, we define λ j = ∏ q /lscript = j +1 ( β /lscript ∧ 1) for any j ∈ [ q -1], and set λ q = 1. Then we have β j · λ j = β ∗ j for all j ∈ [ q ]. Furthermore, we show that f is a good approximation of f ( · , a ). Specifically, we have

Combining this with (4.17) and the fact that ‖ h jk -h jk ‖ ∞ /lessorsimilar N -β j /t j , we obtain that

<!-- formula-not-decoded -->

Moreover, using classical results on the covering number of neural networks (Anthony and Bartlett, 2009), we further show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where δ = 1 /n . Therefore, combining (4.11), (4.13), (4.18), and (4.19), we conclude the proof.

## 5 Extension to Two-Player Zero-Sum Markov Games

In this section, we propose the Minimax-DQN algorithm, which combines DQN and the MinimaxQ learning for two-player zero-sum Markov games. We first present the background of zero-sum Markov games and introduce the the algorithm in § 5.1. Borrowing the analysis for DQN in the previous section, we provide theoretical guarantees for the proposed algorithm in § 5.2.

## 5.1 Minimax-DQN Algorithm

As one of the simplistic extension of MDP to the multi-agent setting, two-player zero-sum Markov game is denoted by ( S , A , B , P, R, γ ), where S is state space, A and B are the action spaces of the first and second player, respectively. In addition, P : S × A × B → P ( S ) is the Markov transition kernel, and R : S × A × B → P ( R ) is the distribution of immediate reward received by the first player. At any time t , the two players simultaneously take actions A t ∈ A and B t ∈ B at state S t ∈ S , then the first player receives reward R t ∼ R ( S t , A t , B t ) and the second player obtains -R t . The goal of each agent is to maximize its own cumulative discounted return.

Furthermore, let π : S → P ( A ) and ν : S → P ( B ) be policies of the first and second players, respectively. Then, we similarly define the action-value function Q π,ν : S × A × B → R as and define the state-value function V π,ν : S → R as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that these two value functions are defined by the rewards of the first player. Thus, at any stateaction tuple ( s, a, b ), the two players aim to solve max π min ν Q π,ν ( s, a, b ) and min ν max π Q π,ν ( s, a, b ) , respectively. By the von Neumann's minimax theorem (Von Neumann and Morgenstern, 1947; Patek, 1997), there exists a minimax function of the game, Q ∗ : S × A × B → R , such that

<!-- formula-not-decoded -->

Moreover, for joint policy ( π, ν ) of two players, we define the Bellman operators T π,ν and T by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r ( s, a, b ) = ∫ rR (d r | s, a, b ), and we define operators P π,ν and P ∗ by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that P ∗ is defined by solving a zero-sum matrix game based on Q ( s ′ , · , · ) ∈ R |A|×|B| , which could be achieved via linear programming. It can be shown that both T π,ν and T are γ -contractive,

with Q π,ν defined in (5.1) and Q ∗ defined in (5.3) being the unique fixed points, respectively. Furthermore, similar to (2.6), in zero-sum Markov games, for any action-value function Q , the equilibrium joint policy with respect to Q is defined as

<!-- formula-not-decoded -->

That is, π Q ( · | s ) and ν Q ( · | s ) solves the zero-sum matrix game based on Q ( s, · , · ) for all s ∈ S . By this definition, we obtain that the equilibrium joint policy with respect to the minimax function Q ∗ defined in (5.3) achieves the Nash equilibrium of the Markov game.

Therefore, to learn the Nash equilibrium, it suffices to estimate Q ∗ , which is the unique fixed point of the Bellman operator T . Similar to the standard Q-learning for MDP, Littman (1994) proposes the Minimax-Q learning algorithm, which constructs a sequence of action-value functions that converges to Q ∗ . Specifically, in each iteration, based on a transition ( s, a, b, s ′ ), Minimax-Q learning updates the current estimator of Q ∗ , denoted by Q , via where α ∈ (0 , 1) is the stepsize.

<!-- formula-not-decoded -->

Motivated by this algorithm, we propose the Minimax-DQN algorithm which extend DQN to two-player zero-sum Markov games. Specifically, we parametrize the action-value function using a deep neural network Q θ : S × A × B → R and store the transition ( S t , A t , B t , R t , S t +1 ) into the replay memory M at each time-step. Parameter θ of the Q-network is updated as follows. Let Q θ ∗ be the target network. With n independent samples { ( s i , a i , b i , r i , s ′ i ) } i ∈ [ n ] from M , for all i ∈ [ n ], we compute the target

<!-- formula-not-decoded -->

To understand the theoretical aspects of this algorithm, we similarly utilize the framework of batch reinforcement learning for statistical analysis. With the insights gained in § 3, we consider a modification of Minimax-DQN based on neural fitted Q-iteration, whose details are stated in Algorithm 2. As in the MDP setting, we replace sampling from the replay memory by sampling i.i.d. state-action tuples from a fixed distribution σ ∈ P ( S × A × B ), and estimate Q ∗ in (5.3) by solving a sequence of least-squares regression problems specified by (5.8). Intuitively, this algorithm approximates the value iteration algorithm for zero-sum Markov games (Littman, 1994) by constructing a sequence of value functions { ˜ Q k } k ≥ 0 such that ˜ Q k +1 ≈ T ˜ Q k for all k , where T defined in (5.5) is the Bellman operator.

which can be attained via linear programming. Then we update θ in the direction of ∇ θ L ( θ ), where L ( θ ) = n -1 ∑ i ∈ [ n ] [ Y i -Q θ ( s i , a i , b i )] 2 . Finally, the target network Q θ ∗ is updated every T target steps by letting θ ∗ = θ . For brevity, we defer the details of Minimax-DQN to Algorithm 4 in § A.

## 5.2 Theoretical Results for Minimax-FQI

Following the theoretical results established in § 4, in this subsection, we provide statistical guarantees for the Minimax-FQI algorithm with F being a family of deep neural networks with ReLU

## Algorithm 2 Fitted Q-Iteration Algorithm for Zero-Sum Markov Games (Minimax-FQI)

Input: Two-player zero-sum Markov game ( S , A , B , P, R, γ ), function class F , distribution σ ∈ P ( S × A × B ), number of iterations K , number of samples n , the initial estimator ˜ Q 0 ∈ F . for k = 0 , 1 , 2 , . . . , K -1 do

<!-- formula-not-decoded -->

Sample n i.i.d. observations { ( S i , A i , B i ) } i ∈ [ n ] from σ , obtain R i ∼ R ( · | S i , A i , B i ) and S ′ i ∼ P ( · | S i , A i , B i ).

end for

<!-- formula-not-decoded -->

Let ( π K , ν K ) be the equilibrium joint policy with respect to ˜ Q K , which is defined in (5.6). Output: An estimator ˜ Q K of Q ∗ and joint policy ( π K , ν K ).

activation. Hereafter, without loss of generality, we assume S = [0 , 1] r with r being a fixed integer, and the action spaces A and B are both finite. To evaluate the performance of the algorithm, we first introduce the best-response policy as follows.

Definition 5.1. For any policy π : S → P ( A ) of player one, the best-response policy against π , denoted by ν ∗ π , is defined as the optimal policy of second player when the first player follows π . In other words, for all s ∈ S , we have ν ∗ π ( · | s ) = argmin ν V π,ν ( s ) , where V π,ν is defined in (5.2).

Note that when the first player adopt a fixed policy π , from the perspective of the second player, the Markov game becomes a MDP. Thus, ν ∗ π is the optimal policy of the MDP induced by π . Moreover, it can be shown that, for any policy π , Q ∗ ( s, a, b ) ≥ Q π,ν ∗ π ( s, a, b ) holds for every stateaction tuple ( s, a, b ). Thus, by considering the adversarial case where the opponent always plays the best-response policy, the difference between Q π.ν ∗ π and Q ∗ servers as a characterization of the suboptimality of π . Hence, to quantify the performance of Algorithm 2, we consider the closeness between Q ∗ and Q π K ,ν ∗ π K , which will be denoted by Q ∗ K hereafter for simplicity. Specifically, in the following we establish an upper bound for ‖ Q ∗ -Q ∗ K ‖ 1 ,µ for some distribution µ ∈ P ( S × A × B ).

We first specify the function class F in Algorithm 2 as follows.

Assumption 5.2 (Function Classes) . Following Definition 4.1, let F ( L, { d j } L +1 j =0 , s ) and G ( { p j , t j , β j , H j } j ∈ [ q ] ) be the family of sparse ReLU networks and the set of composition of H¨ older smooth functions defined on S , respectively. Similar to (4.1), we define F 1 by

For the Bellman operator T defined in (5.5), we assume that for any f ∈ F 1 and any state-action tuple ( s, a, b ), we have ( Tf )( · , a, b ) ∈ G ( { p j , t j , β j , H j } j ∈ [ q ] ).

<!-- formula-not-decoded -->

We remark that this Assumption is in the same flavor as Assumption 4.2. As discussed in § 4, this assumption holds if both the reward function and the transition density of the Markov game are sufficiently smooth.

In the following, we define the concentration coefficients for Markov games.

Assumption 5.3 (Concentration Coefficient for Zero-Sum Markov Games) . Let { τ t : S → P ( A× B ) } be a sequence of joint policies for the two players in the zero-sum Markov game. Let ν 1 , ν 2 ∈ P ( S × A × B ) be two absolutely continuous probability measures. Suppose the initial state-action pair ( S 0 , A 0 , B 0 ) has distribution ν 1 , the future states are sampled according to the Markov transition kernel, and the action ( A t , B t ) is sampled from policy τ t . For any integer m , we denote by P τ m P τ m -1 · · · P τ 1 ν 1 the distribution of { ( S t , A t , B t ) } m t =0 . Then, the m -th concentration coefficient is defined as

<!-- formula-not-decoded -->

where the supremum is taken over all possible joint policy sequences { τ t } t ∈ [ m ] .

Furthermore, for some µ ∈ P ( S × A × B ), we assume that there exists a finite constant φ µ,σ such that (1 -γ ) 2 · ∑ m ≥ 1 γ m -1 · m · κ ( m ; µ, σ ) ≤ φ µ,σ , where σ is the sampling distribution in Algorithm 2 and κ ( m ; µ, σ ) is the m -th concentration coefficient defined in (5.10).

We remark that the definition of the m -th concentration coefficient is the same as in (4.4) if we replace the action space A of the MDP by A×B of the Markov game. Thus, Assumptions 4.3 and 5.3 are of the same nature, which are standard in the literature.

Now we are ready to present the main theorem.

Theorem 5.4. Under Assumptions 5.2 and 5.3, consider the Minimax-FQI algorithm with the function class F being F 1 defined in (5.9) based on the family of sparse ReLU networks F ( L ∗ , { d ∗ j } L ∗ +1 j =0 , s ∗ ). We make the same assumptions on F ( L ∗ , { d ∗ j } L ∗ +1 j =0 , s ∗ ) and G ( { p j , t j , β j , H j } j ∈ [ q ] ) as in (4.6) and (4.7). Then for any K ∈ N , let ( π K , ν K ) be the policy returned by the algorithm and let Q ∗ K be the action-value function corresponding to ( π K , ν ∗ π K ). Then there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

where ξ ∗ appears in (4.7), α ∗ = max j ∈ [ q ] t j / (2 β ∗ j + t j ) and φ µ,σ is specified in Assumption 5.3.

Similar to Theorem 4.4, the bound in (5.11) shows that closeness between ( π K , ν K ) returned by Algorithm 2 and the Nash equilibrium policy ( π Q ∗ , ν Q ∗ ), measured by ‖ Q ∗ -Q ∗ K ‖ 1 ,µ , is bounded by the sum of statistical error and an algorithmic error. Specifically, the statistical error balances the bias and variance of estimating the value functions using the family of deep ReLU neural networks, which exhibits the fundamental difficulty of the problem. Whereas the algorithmic error decay to zero geometrically as K increases. Thus, when K is sufficiently large, both γ and φ µ,σ are constants, and the polylogarithmic term is ignored, Algorithm 2 achieves error rate

<!-- formula-not-decoded -->

which scales linearly with the capacity of joint action space. Besides, if |B| = 1, the minimax-FQI algorithm reduces to Algorithm 1. In this case, (5.12) also recovers the error rate of Algorithm 1. Furthermore, the statistical rate n ( α ∗ -1) / 2 achieves the optimal /lscript 2 -norm error of regression for nonparametric regression with a compositional structure, which indicates that the statistical error in (5.11) can not be further improved.

Proof. See § D for a detailed proof.

## 6 Proof of the Main Theorem

In this section, we present a detailed proof of Theorem 4.4.

Proof. The proof requires two key ingredients. First in Theorem 6.1 we quantify how the error of action-value function approximation propagates through each iteration of Algorithm 1. Then in Theorem 6.2 we analyze such one-step approximation error for ReLU networks.

Theorem 6.1 (Error Propagation) . Recall that { ˜ Q k } 0 ≤ k ≤ K are the iterates of Algorithm 1. Let π K be the one-step greedy policy with respect to ˜ Q K , and let Q π K be the action-value function corresponding to π K . Under Assumption 4.3, we have

<!-- formula-not-decoded -->

where we define the maximum one-step approximation error as ε max = max k ∈ [ K ] ‖ T ˜ Q k -1 -˜ Q k ‖ σ . Here φ µ,σ is a constant that only depends on the probability distributions µ and σ .

Proof. See § C.1 for a detailed proof.

We remark that similar error propagation result is established for the state-value function in Munos and Szepesv´ ari (2008) for studying the fitted value iteration algorithm, which is further extended by Lazaric et al. (2016); Scherrer et al. (2015); Farahmand et al. (2010, 2016) for other batch reinforcement learning methods.

In the sequel, we establish an upper bound for the one-step approximation error ‖ T ˜ Q k -1 -˜ Q k ‖ σ for each k ∈ [ K ].

Theorem 6.2 (One-step Approximation Error) . Let F ⊆ B ( S ×A , V max ) be a class of measurable functions on S×A that are bounded by V max = R max / (1 -γ ), and let σ be a probability distribution on S × A . Also, let { ( S i , A i ) } i ∈ [ n ] be n i.i.d. random variables in S × A following σ . For each i ∈ [ n ], let R i and S ′ i be the reward and the next state corresponding to ( S i , A i ). In addition, for any fixed Q ∈ F , we define Y i = R i + γ · max a ∈A Q ( S ′ i , a ). Based on { ( X i , A i , Y i ) } i ∈ [ n ] , we define ̂ Q as the solution to the least-squares problem

<!-- formula-not-decoded -->

Meanwhile, for any δ &gt; 0, let N ( δ, F , ‖ · ‖ ∞ ) be the minimal δ -covering set of F with respect to /lscript ∞ -norm, and we denote by N δ its cardinality. Then for any /epsilon1 ∈ (0 , 1] and any δ &gt; 0, we have

<!-- formula-not-decoded -->

where C and C ′ are two positive absolute constants and ω ( F ) is defined as

<!-- formula-not-decoded -->

Proof. See § C.2 for a detailed proof.

This theorem characterizes the bias and variance that arise in estimating the action-value functions using deep ReLU networks. Specifically, ω ( F ) in (6.4) corresponds to the bias incurred by approximating the target function Tf using ReLU neural networks. It can be viewed as a measure of completeness of F with respect to the Bellman operator T . In addition, V 2 max /n · log N δ + V max · δ controls the variance of the estimator, where the covering number N δ is used to obtain a uniform bound over F 0 .

To obtain an upper bound for ‖ T ˜ Q k -1 -˜ Q k ‖ σ as required in Theorem 6.1, we set Q = ˜ Q k -1 in Theorem 6.2. Then according to Algorithm 1, ̂ Q defined in (6.2) becomes ˜ Q k . We set the function class F in Theorem 6.2 to be the family of ReLU Q-networks F 0 defined in (4.1). By setting /epsilon1 = 1 and δ = 1 /n in Theorem 6.2, we obtain where C is a positive absolute constant and

<!-- formula-not-decoded -->

N 0 = ∣ ∣ N (1 /n, F 0 , ‖ · ‖ ∞ ) ∣ ∣ (6.6) is the 1 /n -covering number of F 0 . In the subsequent proof, we establish upper bounds for ω ( F 0 ) defined in (6.4) and log N 0 , respectively. Recall that the family of composite H¨ older smooth functions G 0 is defined in (4.2). By Assumption 4.2, we have Tg ∈ G 0 for any g ∈ F 0 . Hence, we have

<!-- formula-not-decoded -->

where the right-hand side is the /lscript ∞ -error of approximating the functions in G 0 using the family of ReLU networks F 0 .

By the definition of G 0 in (4.2), for any f ∈ G 0 and any a ∈ A , f ( · , a ) ∈ G ( { ( p j , t j , β j , H j ) } j ∈ [ q ] ) is a composition of H¨ older smooth functions, that is, f ( · , a ) = g q ◦· · · ◦ g 1 . Recall that, as defined in Definition 2.3, g jk is the k -th entry of the vector-valued function g j . Here g jk ∈ C t j ([ a j , b j ] t j , β j , H j ) for each k ∈ [ p j +1 ] and j ∈ [ q ]. In the sequel, we construct a ReLU network to approximate f ( · , a ) and establish an upper bound of the approximation error on the right-hand side of (6.7). We first show that f ( · , a ) can be reformulated as a composition of H¨ older functions defined on a hypercube. We define h 1 = g 1 / (2 H 1 ) + 1 / 2,

<!-- formula-not-decoded -->

and h q ( u ) = g q (2 H q -1 u -H q -1 ). Then we immediately have

<!-- formula-not-decoded -->

Furthermore, by the definition of H¨ older smooth functions in Definition 2.2, for any k ∈ [ p 2 ], we have that h 1 k takes value in [0 , 1] and h 1 k ∈ C t 1 ([0 , 1] t 1 , β 1 , 1). Similarly, for any j ∈ { 2 , . . . , q -1 } and k ∈ [ p j +1 ], h jk also takes value in [0 , 1] and

Finally, recall that we use the convention that p q +1 = 1, that is, h q is a scalar-valued function that satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following, we show that the composition function in (6.8) can be approximated by an element in F ( L ∗ , { d ∗ j } L +1 j =1 , s ∗ ) when the network hyperparameters are properly chosen. Our proof consists of three steps. In the first step, we construct a ReLU network ˜ h jk that approximates each h jk in (6.9). Then, in the second step, we approximate f ( · , a ) by the composition of { ˜ h j } j ∈ [ q ] and quantify the architecture of this network. Finally, in the last step, we prove that this network can be embedded into class F ( L ∗ , { d ∗ j } L +1 j =1 , s ∗ ) and characterize the final approximation error.

Step (i). Now we employ the following lemma, obtained from Schmidt-Hieber (2020+), to construct a ReLU network that approximates each h jk , which combined with (6.8) yields a ReLU network that is close to f ( · , a ). Recall that, as defined in Definition 2.2, we denote by C r ( D , β, H ) the family of H¨ older smooth functions with parameters β and H on D ⊆ R r .

Lemma 6.3 (Theorem 5 in Schmidt-Hieber (2020+)) . For any integers m ≥ 1 and N ≥ max { ( β + 1) r , ( H +1) e r } , let L = 8 + ( m +5) · (1 + /ceilingleft log 2 ( r + β ) /ceilingright ), d 0 = r , d j = 6( r + /ceilingleft β /ceilingright ) N for each j ∈ [ L ], and d L +1 = 1. For any g ∈ C r ([0 , 1] r , β, H ), there exists a ReLU network f ∈ F ( L, { d j } L +1 j =0 , s, V max ) as defined in Definition 2.1 such that

<!-- formula-not-decoded -->

where the parameter s satisfies s ≤ 141 · ( r + β +1) 3+ r · N · ( m +6).

Proof. See Appendix B in Schmidt-Hieber (2020+) for a detailed proof. The idea is to first approximate the H¨ older smooth function by polynomials via local Taylor expansion. Then, neural networks are constructed explicitly to approximate each monomial terms in these local polynomials.

We apply Lemma 6.3 to h jk : [0 , 1] t j → [0 , 1] for any j ∈ [ q ] and k ∈ [ p j +1 ]. We set m = η ·/ceilingleft log 2 n /ceilingright for a sufficiently large constant η &gt; 1, and set N to be a sufficiently large integer depending on n , which will be specified later. In addition, we set

<!-- formula-not-decoded -->

and define

We will later verify that N ≥ max { ( β +1) t j , ( W +1) e t j } for all j ∈ [ q ]. Then by Lemma 6.3, there exists a ReLU network h jk such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ Furthermore, we have h jk ∈ F ( L j , { t j , d j , . . . , d j , 1 } , s j ) with

˜ Meanwhile, since h j +1 = ( h ( j +1) k ) k ∈ [ p j +2 ] takes input from [0 , 1] t j +1 , we need to further transform ̂ h jk so that it takes value in [0 , 1]. In particular, we define σ ( u ) = 1 -(1 -u ) + = min { max { u, 0 } , 1 } for any u ∈ R . Note that σ can be represented by a two-layer ReLU network with four nonzero weights. Then we define ˜ h jk = σ ◦ ̂ h jk and ˜ h j = ( ˜ h jk ) k ∈ [ p j +1 ] . Note that by the definition of ˜ h jk , we have ˜ h jk ∈ F ( L j +2 , { t j , d j , . . . , d j , 1 } , s j +4) , which yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Moreover, since both h jk and h jk take value in [0 , 1], by (6.12) we have where the constant W is defined in (6.11). Since we can set the constant η in (6.15) to be sufficiently large, the second term on the right-hand side of (6.15) is the leading term asymptotically, that is,

‖ ˜ h jk -h jk ‖ ∞ /lessorsimilar N -β j /t j . (6.16) Thus, in the first step, we have shown that there exists ˜ h jk ∈ F ( L j +2 , { t j , ˜ d j , . . . , ˜ d j , 1 } , ˜ s j +4) satisfying (6.16).

˜ ˜ ˜ ˜ where we define ˜ L = ∑ q j =1 ( L j +2), ˜ d = max j ∈ [ q ] ˜ d j · p j +1 , and ˜ s = ∑ q j =1 ( ˜ s j +4) · p j +1 . Recall that L j is defined in (6.10). Then when n is sufficiently large, we have

Step (ii). In the second step, we stack ˜ h j defined in (6.14) to approximate f ( · , a ) in (6.8). Specifically, we define ˜ f : S → R as ˜ f = ˜ h q ◦ · · · ◦ ˜ h 1 , which falls in the function class F ( L, { r, d, . . . , d, 1 } , s ) , (6.17)

<!-- formula-not-decoded -->

where ξ &gt; 0 is an absolute constant. Here the last inequality follows from (4.6). Moreover, for ˜ d defined in (6.17), by (4.6) we have

In addition, combining (6.13), (4.6), and the fact that t j ≤ p j , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step (iii). In the last step, we show that the function class in (6.17) can be embedded in F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ) and characterize the final approximation bias, where L ∗ , { d ∗ j } L ∗ +1 j =1 , and s ∗ are specified in (4.7). To this end, we set

<!-- formula-not-decoded -->

where the absolute constant C &gt; 0 is sufficiently large. Note that we define α ∗ = max j ∈ [ q ] t j / (2 β ∗ j + t j ). Then (6.21) implies that N /equivasymptotic n α ∗ . When n is sufficiently large, it holds that N ≥ max { ( β + 1) t j , ( W +1) e t j } for all j ∈ [ q ]. When ξ ∗ in (4.7) satisfies ξ ∗ ≥ 1 + 2 ξ , by (6.18) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, (6.19) and (4.7) implies that we can set d ∗ j ≥ ˜ d for all j ∈ [ L ∗ ]. Finally, by (6.20) and (6.21), we have ˜ s /lessorsimilar n α ∗ · (log n ) ξ ∗ , which implies ˜ s +( L ∗ -˜ L ) · r ≤ s ∗ . For an ˜ L -layer ReLU network in (6.17), we can make it an L ∗ -layer ReLU network by inserting L ∗ -˜ L identity layers, since the inputs of each layer are nonnegative. Thus, ReLU networks in (6.17) can be embedded in which is a subset of F ( L ∗ , { d ∗ j } L +1 j =1 , s ∗ ) by (4.7).

<!-- formula-not-decoded -->

To obtain the approximation error ‖ ˜ f -f ( · , a ) ‖ ∞ , we define G j = h j ◦· · ·◦ h 1 and ˜ G j = ˜ h j ◦· · ·◦ ˜ h 1 for any j ∈ [ q ]. By triangle inequality, for any j &gt; 1 we have where the second inequality holds since h j is H¨ older smooth. To simplify the notation, we define λ j = ∏ q /lscript = j +1 ( β /lscript ∧ 1) for any j ∈ [ q -1], and set λ q = 1. By applying recursion to (6.22), we obtain

<!-- formula-not-decoded -->

where the constant W is defined in (6.11). Here in (6.23) we use the fact that ( a + b ) α ≤ a α + b α for all α ∈ [0 , 1] and a, b &gt; 0.

In the sequel, we combine (6.7), (6.15), (6.23), and (6.21) to obtain the final bound on ω ( F 0 ). Also note that β ∗ j = β j · ∏ q /lscript = j +1 ( β /lscript ∧ 1) = β j · λ j for all j ∈ [ q -1]. Thus we have β ∗ j = β j · λ j for all j ∈ [ q ]. Combining (6.23) and (6.16), we have

<!-- formula-not-decoded -->

Thus, we combine (6.7), (6.21), and (6.24) to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As the final step of the proof, it remains to control the covering number of F 0 defined in (4.1). By definition, for any f ∈ F 0 , we have f ( · , a ) ∈ F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ) for any a ∈ A . For notational simplicity, we denote by N δ the δ -covering of F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ), that is, we define

By the definition of covering, for any f ∈ F 0 and any a ∈ A , there exists g a ∈ N δ such that ‖ f ( · , a ) -g a ‖ ∞ ≤ δ . Then we define a function g : S×A → R by g ( s, a ) = g a ( s ) for any ( s, a ) ∈ S×A . By the definition of g , it holds that ‖ f -g ‖ ∞ ≤ δ . Therefore, the cardinality of N ( δ, F 0 , ‖ · ‖ ∞ ) satisfies

<!-- formula-not-decoded -->

Now we utilize the following lemma in Anthony and Bartlett (2009) to obtain an upper bound of the cardinality of N δ .

Lemma 6.4 (Covering Number of ReLU Network) . Recall that the family of ReLU networks F ( L, { d j } L +1 j =0 , s, V max ) is given in Definition 2.1. Let D = ∏ L +1 /lscript =1 ( d /lscript +1). For any δ &gt; 0, we have

<!-- formula-not-decoded -->

Recall that we denote N (1 /n, F 0 , ‖ · ‖ ∞ ) by N 0 in (6.6). By combining (6.26) with Lemma 6.4 and setting δ = 1 /n , we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D = ∏ L ∗ +1 /lscript =1 ( d ∗ /lscript +1). By the choice of L ∗ , s ∗ , and { d ∗ j } L ∗ +1 j =0 in (4.7), we conclude that

Finally, combining (6.1), (6.5), (6.25), and (6.27), we conclude the proof of Theorem 4.4.

## 7 Conclusion

We study deep Q-network from the statistical perspective. Specifically, by neglecting the computational issues, we consider the fitted Q-iteration algorithm with ReLU networks, which can be viewed as a modification of DQN that fully captures its key features. Under mild assumptions, we show that DQN creates a sequence of policies whose corresponding value functions converge to the optimal value function, when both the sample size and the number of iteration go to infinity. Moreover, we establish a precise characterization of both the statistical and the algorithmic rates of convergence. As a byproduct, our results provide theoretical justification for the trick of using a target network in DQN. Furthermore, we extend DQN to two-player zero-sum Markov games by proposing the Minimax-DQN algorithm. Utilizing the analysis of DQN, we establish theoretical guarantees for Minimax-DQN. To further extend this work, one future direction is to analyze reinforcement learning methods targeting at MDP with continuous action spaces, e.g., example, soft Q-learning (Haarnoja et al., 2017) and deep deterministic policy gradient (DDPG) (Lillicrap et al., 2016). Another promising direction is to combine results on optimization for deep learning with our statistical analysis to gain a unified understanding of the statistical and computational aspects of DQN.

## A Deep Q-Network

We first present the DQN algorithm for MDP in details, which is proposed by Mnih et al. (2015) and adapted here to discounted MDP. As shown in Algorithm 3 below, DQN features two key tricks that lead to its empirical success, namely, experience replay and target network.

## Algorithm 3 Deep Q-Network (DQN)

Input: MDP ( S , A , P, R, γ ), replay memory M , number of iterations T , minibatch size n , exploration probability /epsilon1 ∈ (0 , 1), a family of deep Q-networks Q θ : S ×A → R , an integer T target for updating the target network, and a sequence of stepsizes { α t } t ≥ 0 .

Initialize the Q-network with random weights θ .

Initialize the replay memory M to be empty.

Initialize the weights of the target network with θ /star = θ .

Initialize the initial state S 0 .

<!-- formula-not-decoded -->

With probability /epsilon1 , choose A t uniformly at random from A , and with probability 1 -/epsilon1 , choose A t such that Q θ ( S t , A t ) = max a ∈A Q θ ( S t , a ).

Execute A t and observe reward R t and the next state S t +1 .

Store transition ( S t , A t , R t , S t +1 ) in M .

For each i ∈ [ n ], compute the target Y i = r i + γ · max a ∈A Q θ /star ( s ′ i , a ).

Experience replay: Sample random minibatch of transitions { ( s i , a i , r i , s ′ i ) } i ∈ [ n ] from M .

Update the Q-network: Perform a gradient descent step

<!-- formula-not-decoded -->

Update the target network: Update θ /star ← θ every T target steps.

## end for

Define policy π as the greedy policy with respect to Q θ .

Output: Action-value function Q θ and policy π .

Furthermore, in the following, we present the details of the Minimax-DQN algorithm that extends DQN to two-player zero-sum Markov games introduced in § 5. Similar to DQN, this algorithm also utilizes the experience replay and target networks. The main difference is that here the target Y i in (5.7) is obtained by solving a zero-sum matrix game. In Algorithm 4 we present the algorithm for the second player, which can be easily modified for the first player. We note that for the second player, similar to (5.6),the equilibrium joint policy is defined as

<!-- formula-not-decoded -->

## Algorithm 4 Minimax Deep Q-Network (Minimax-DQN) for the second player

Input: Zero-Sum Markov game ( S , A , B , P, R, γ ), replay memory M , number of iterations T , minibatch size n , exploration probability /epsilon1 ∈ (0 , 1), a family of deep Q-networks Q θ : S×A×B → R , an integer T target for updating the target network, and a sequence of stepsizes { α t } t ≥ 0 . Initialize the replay memory M to be empty.

Initialize the Q-network with random weights θ .

Initialize the weights of the target network by letting θ /star = θ .

Initialize the initial state S 0 .

<!-- formula-not-decoded -->

With probability /epsilon1 , choose B t uniformly at random from B , and with probability 1 -/epsilon1 , sample B t according to the equilibrium policy ν Q θ ( · | S t ) defined in (A.1).

Store transition ( S t , A t , B t , R t , S t +1 ) in M .

˜ Execute B t and observe the first player's action A t , reward R t satisfying -R t ∼ R ( S t , A t , B t ), and the next state S t +1 ∼ P ( · | S t , A t , B t ).

Experience replay: Sample random minibatch of transitions { ( s i , a i , b i , r i , s ′ i ) } i ∈ [ n ] from M . For each i ∈ [ n ], compute the target

<!-- formula-not-decoded -->

Update the Q-network: Perform a gradient descent step

<!-- formula-not-decoded -->

Update the target network: Update θ /star ← θ every T target steps. end for

Output: Q-network Q θ and equilibrium joint policy with respect to Q θ .

## B Computational Aspect of DQN

Recall that in Algorithm 1 we assume the global optima of the nonlinear least-squares problem in (3.1) is obtained in each iteration. We make such an assumption as our focus is on the statistical analysis. In terms of optimization, it has been shown recently that, when the neural network is overparametrized, (stochastic) gradient descent converges to the global minima of the empirical function. Moreover, the generalization error of the obtained neural network can also be established. The intuition behind these results is that, when the neural network is overparametrized, it behaves similar to the random feature model (Rahimi and Recht, 2008, 2009). See, e.g., Du et al. (2019b,a); Zou et al. (2018); Chizat et al. (2019); Allen-Zhu et al. (2019a,b); Jacot et al. (2018); Cao and Gu (2019); Arora et al. (2019); Weinan et al. (2019); Mei et al. (2019); Yehudai and Shamir (2019); Bietti and Mairal (2019); Yang and Salman (2019); Yang (2019); Gao et al. (2019); Bai and Lee (2019); Huang et al. (2020) and the references therein. Also see Fan et al. (2019) for a detailed sur-

vey. In this section, we make an initial attempt in providing a unified statistical and computational analysis of DQN.

In the sequel, we consider the reinforcement learning problem with the state space S = [0 , 1] r and a finite action space A . To simplify the notation, we represent action a using one-hot embedding and thus identify it as an element in { 0 , 1 } |A| ⊆ R |A| . In practice, categorical actions are often embedded into the Euclidean space (Dulac-Arnold et al., 2015). Thus, we can pack the state s and the action a together and obtain a vector ( s, a ) in R d , where we denote r + |A| by d . Moreover, without loss of generality, we assume that ‖ ( s, a ) ‖ 2 ≤ 1.

We represent the Q-network by the family of two-layer neural networks

<!-- formula-not-decoded -->

Here 2 m is the number of neurons, b j ∈ R and W j ∈ R d for all j ∈ [2 m ], and σ ( u ) = max { u, 0 } is the ReLU activation function. Here b = ( b 1 , . . . , b 2 m ) /latticetop ∈ R 2 m and W = ( W 1 , . . . , W 2 m ) ∈ R d × 2 m are the weights of the neural network.

For such class of neural networks, for any k ≥ 1, in k -th iteration of the neural FQI algorithm, the optimization problem in (3.1) becomes

<!-- formula-not-decoded -->

where Y i = R i + γ · max a ∈A ˜ Q k -1 ( S ′ i , a ) is the target and ˜ Q k -1 is the Q-network computed in the previous iteration. Notice that this problem is a least-squares regression with overparameterized neural networks. For computational efficiency, we propose to solve (B.2) via stochastic gradient descent (SGD). Specifically, in each iteration of SGD, we sample a fresh observation ( S, A, R, S ′ ) with ( S, A ) drawn from the sampling distribution σ , R ∼ R ( · | S, A ), and S ′ ∼ P ( · | S, A ). Then an estimator of the gradient is computed based on ( S, A, R, S ′ ), which is used to update the network parameters. We run the SGD updates for a total of n iterations and denote the output by Q k .

˜ Besides, in each FQI-step, we initialize the parameters via the symmetric initialization scheme (Gao et al., 2019; Bai and Lee, 2019) as follows. For any j ∈ [ m ], we set b j i.i.d. ∼ Unif( {-1 , 1 } ) and W j i.i.d. ∼ N (0 , I d /d ), where I d is the identity matrix in R d . For any j ∈ { m + 1 , . . . , 2 m } , we set b j = -b j -m and W j = W j -m . We remark that such initialization implies that the initial Q -network is a zero function, which is used only to simply the theoretical analysis. Besides, for ease of presentation, during training we fix the value of b at its initial value and only optimize over W . We initialize b and W at the very beginning of our algorithm and in each FQI subproblem, we update the Q-network starting from the same initialization. Hereafter, we denote the initial value of W and b by W (0) ∈ R d × 2 m and b (0) ∈ R 2 m , respectively, and let Q ( · , · ; W ) denote Q ( · , · ; b (0) , W ). In order to have bounded functions, we further restrict the weight W to a Frobenius norm ball centered at W (0) with radius B &gt; 0, i.e., we define

<!-- formula-not-decoded -->

where B is a sufficiently large constant. Thus, the population version of the k -th iteration of the FQI algorithm becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( S, A ) ∼ σ and Y is computed using ˜ Q k -1 . We solve this optimization problem via projected SGD, which generates a sequence of weight matrices { W ( t ) } t ≥ 0 ⊆ B B satisfying where Π B B is the projection operator onto B B with respect to the Frobenius norm, η &gt; 0 is the step size, and ( S t , A t , Y t ) is a random observation. We present the details of fitted Q-iteration method with projected SGD in Algorithm 5.

## Algorithm 5 Fitted Q-Iteration Algorithm with Projected SGD Updates

Input: MDP ( S , A , P, R, γ ), function class F , sampling distribution σ , number of FQI iterations K , number of SGD iterations T , the initial estimator ˜ Q 0 . Initialize the weights b (0) and W (0) of Q-network via the symmetric initialization scheme.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Draw an independent sample ( S t , A t , R t , S ′ t ) with ( S t , A t ) drawn from distribution σ .

Compute Y t = R t + γ · max a ∈A ˜ Q k ( S ′ t , a ). Perform projected SGD update

## end for

<!-- formula-not-decoded -->

Update the action-value function ˜ Q k +1 ( · , · ) ← Q ( · , · ; W k +1 ) where W k +1 = T -1 ∑ T t =1 W ( t ) . end for

Define policy π K as the greedy policy with respect to ˜ Q K .

To understand the convergence of the projected SGD updates in (B.5), we utilize the fact that the dynamics of training overparametrized neural networks is captured by the neural tangent kernel (Jacot et al., 2018) when the width is sufficiently large. Specifically, since σ ( u ) = u · 1 { u &gt; 0 } , the gradient of the Q-network in (B.1) is given by

Output: An estimator ˜ Q K of Q ∗ and policy π K .

<!-- formula-not-decoded -->

Recall that we initialize parameters b and W as b (0) and W (0) and that we only update W during training. We define a function class F ( t ) B,m as

<!-- formula-not-decoded -->

By (B.6), for each function ̂ Q ( · , · ; W ) ∈ F ( t ) B,m , we can write it as which is the first-order linearization of Q ( · , · ; W ( t ) ) at W ( t ) . Furthermore, since B in (B.3) is a constant, for each weight matrix W in B B , when m goes to infinity, ‖ W j -W (0) j ‖ 2 would be small for almost all j ∈ [2 m ], which implies that 1 { W /latticetop j ( s, a ) &gt; 0 } = 1 { ( W (0) j ) /latticetop ( s, a ) &gt; 0 } holds with high probability for all j ∈ [2 m ] and ( s, a ) ∈ S × A . As a result, when m is sufficiently large, F ( t ) B,m defined in (B.7) is close to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where b (0) and W (0) are the initial parameters. Specifically, as proved in Lemma A.2 in Wang et al. (2019), when the sampling distribution σ is regular in the sense that Assumption (B.2) specified below is satisfied, for any W 1 , W 2 ∈ B B , we have where E init denotes that the expectation is taken with respect to the initialization of the network parameters. Thus, when the network width 2 m is sufficiently large such that B 3 · m -1 / 2 = o (1), the linearized function classes {F ( t ) B,m } t ∈ [ T ] are all close to F (0) B,m .

To simplify the notation, for b ∈ {-1 , 1 } and W ∈ R d , we define feature mapping φ ( · , · ; b, W ): S× A → R d as

<!-- formula-not-decoded -->

Besides, for all j ∈ [2 m ], we let φ j denote φ ( · , · ; b (0) j , W (0) j ). Due to the symmetric initialization scheme, { φ j } j ∈ [ m ] are i.i.d. random feature functions and φ j = -φ j + m for all j ∈ [ m ]. Thus, each ̂ Q ( · , · ; W ) in (B.8) can be equivalently written as

Let W ′ j = ( W j -W j + m ) / √ 2. Since W ∈ B B , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the fact that W (0) j = W (0) j + m for all j ∈ [ m ]. Thus, combining (B.10) and (B.11), we conclude that F (0) B,m in (B.8) is a subset of F B,m defined as

Notice that each function in F B,m is a linear combination of m i.i.d. random features. In particular, let β ∈ Unif( {-1 , 1 } ) and ω ∼ N ( I d /d ) be two independent random variables and let µ denote their joint distribution. Then the random feature φ ( · , · ; β, ω ) induces a reproducing kernel Hilbert space H (Rahimi and Recht, 2008, 2009; Bach, 2017) with kernel K : ( S × A ) × ( S × A ) → R given by

Each function in H can be represented by a mapping α : {-1 , 1 } × R d → R d as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For two functions f α 1 and f α 2 in H represented by α 1 and α 2 , respectively, their inner product is given by

<!-- formula-not-decoded -->

Therefore, from the perspective of neural tangent kernel, when the Q-network is represented by the class of overparametrized neural networks given in (B.1) with a sufficiently large number of neurons, each population problem associated with each FQI iteration in (B.4) becomes

We let ‖·‖ H denote the RKHS norm of H . Then, when m goes to infinity, F B,m in (B.12) converges to the RKHS norm ball H B = { f ∈ H : ‖ f ‖ H ≤ B } .

<!-- formula-not-decoded -->

Utilizing the connection between neural network training and RKHS, in the sequel, we provide a jointly statistical and computational analysis of Algorithm 5. To this end, we define a function class G B as where the minimization is over a subset of H B as F (0) B,m is a subset of F B,m .

<!-- formula-not-decoded -->

That is, each function in G B is represented by a feature mapping α : {-1 , 1 } × R d → R d which is almost surely bounded in the /lscript ∞ -norm. Thus, G B is a strict subset of the RKHS-norm ball H B . When B is sufficiently large, G B is known to be a rich function class (Hofmann et al., 2008). Similar to Assumption 4.2 in § 4, we impose the following assumption on the Bellman optimality operator.

Assumption B.1 (Completeness of G B ) . We assume that that TQ ( · , · ; W ) ∈ G B for all for all W ∈ B B , where T is the Bellman optimality operator and Q ( · , · ; W ) is given in (B.1).

This assumption specifies that T maps any neural network with weight matrix W in B B to a subset G B of the RKHS H . When m is sufficiently large, this assumption is similar to stating that G B is approximately closed under T .

We also impose the following regularity condition on the sampling distribution σ .

Assumption B.2 (Regularity Condition on σ ) . We assume that there exists an absolute constant C &gt; 0 such that

Assumption B.2 states that the density of σ is sufficiently regular, which holds when the density is upper bounded.

<!-- formula-not-decoded -->

Now we are ready to present the main result of this section, which characterizes the performance of π K returned by Algorithm 5.

Theorem B.3. In Algorithm 5, we assume that each step of the fitted-Q iteration is solved by T steps of projected SGD updates with a constant stepsize η &gt; 0. We set T = C 1 m and η = C 2 / √ T , where C 1 , C 2 are absolute constants that are properly chosen. Then, under Assumptions 4.3, B.1, and B.2, we have

<!-- formula-not-decoded -->

where E init denotes that the expectation is taken with respect to the randomness of the initialization.

As shown in (B.15), the error E init [ ‖ Q ∗ -Q π K ‖ 1 ,µ ] can be similarly written as the sum of a statistical error and an algorithmic error, where the algorithmic error converges to zero at a linear rate as K goes to infinity. The statistical error corresponds to the error incurred in solving each FQI step via T projected SGD steps. As shown in (B.15), when B is regarded as a constant, with T /equivasymptotic m projected SGD steps, we obtain an estimator with error O ( m -1 / 8 ). Hence, Algorithm 5 finds the globally optimal policy when both m and K goes to infinity. Therefore, when using overparametrized neural networks, our fitted Q-iteration algorithm provably attains both statistical accuracy and computational efficiency.

Finally, we remark that focus on the class of two-layer overparametrized ReLU neural networks only for the simplicity of presentation. The theory of neural tangent kernel can be extended to feedforward neural networks with multiple layers and neural networks with more complicated architectures (Gao et al., 2019; Frei et al., 2019; Yang and Salman, 2019; Yang, 2019; Huang et al., 2020).

## B.1 Proof of Theorem B.3

Proof. Our proof is similar to that of Theorem 4.4. For any k ∈ [ K ], we define the maximum one-step approximation error as ε max = max k ∈ [ K ] E init [ ‖ T ˜ Q k -1 -˜ Q k ‖ σ ], where E init denotes that the expectation is taken with respect to the randomness in the initialization of network weights, namely b (0) and W (0) . By Theorem 6.1, we have

<!-- formula-not-decoded -->

where φ µ,σ , specified in Assumption 4.3, is a constant that only depends on the concentration coefficients. Thus, it remains to characterize ‖ T ˜ Q k -1 -˜ Q k ‖ σ for each k , which corresponds to the prediction risk of the estimator constructed by T projected SGD steps.

In the sequel, we characterize the prediction risk of the projected SGD method via the framework of neural tangent kernel. Our proof technique is motivated by recent work (Gao et al., 2019; Cai et al., 2019; Liu et al., 2019a; Wang et al., 2019; Xu and Gu, 2019) which analyze the training of overparametrized neural networks via projected SGD for adversarial training and reinforcement learning. We focus on the case where the target network is Q k -1 and bound ‖ T Q k -1 -Q k ‖ σ .

˜ ˜ ˜ To begin with, recall that we define function classes F ( t ) B,m , F B,m , and G B in (B.7), (B.12), and (B.14), respectively. Notice that ˜ Q k -1 = Q ( · , · ; W k ) is a neural network where W k ∈ B B due to projection. Then, by Assumption B.1, T ˜ Q k -1 belongs to G B , which is a subset of the RKHS H . Thus, it suffices to study the least-squares regression problem where the target function is in G B and the neural network is trained via projected SGD.

In the following, we use function class F B,m to connect G B and ˜ Q k . Specifically, we show that any function g ∈ G B as well as ˜ Q k can be well approximated by functions in F B,m and quantify the approximation errors. Finally, we focus on the projected SGD algorithm within the linearized function class F B,m and establish the statistical and computational error. The proof is divided into three steps as follows.

Step (i). In the first step, we quantify the difference between ˜ Q k and functions in F B,m . For any j ∈ [2 m ], let [ W k +1 ] j ∈ R d be the weights of ˜ Q k corresponding to the j -th neuron. We define a function ̂ Q k : S × A → R as ̂ Q k ( s, a ) = 1 / √ 2 m · ∑ 2 m j =1 φ j ( s, a ) /latticetop [ W k +1 ] j , which belongs to F (0) B,m defined in (B.8), a subset of F B,m . The following following lemma, obtained from Wang et al. (2019), prove that Q ( · , · ; W ) is close to a linearized function when m is sufficiently large.

Lemma B.4 (Linearization Error) . Let B B be defined in (B.3). Under Assumption B.2, for any W (1) , W (2) ∈ B B , we have

Proof. See Lemma A.2 in Wang et al. (2019) for a detailed proof.

<!-- formula-not-decoded -->

Notice that we have ˜ Q k ( · , · ) = 〈∇ W Q ( · , · ; W k +1 ) , W k +1 〉 and ̂ Q k ( · , · ) = 〈∇ W Q ( · , · ; W (0) ) , W k +1 〉 . Applying Lemma B.4 with W (1) = W (2) = W k +1 , we obtain that

Thus, we have constructed a function in F B,m that is close to ˜ Q k when m is sufficiently large, which completes the first step of the proof.

<!-- formula-not-decoded -->

Step (2). In the second step, we show that each function in G B can also be well approximated by functions in F B,m . To this end, similar to the definition of G B in (B.14), we define a function class F B,m as

<!-- formula-not-decoded -->

which is a subset of F B,m by definition. Intuitively, as m goes to infinity, F B,m becomes G B . The following Lemma, obtained from (Rahimi and Recht, 2009), provides a rigorous characterization of this argument.

Lemma B.5 (Approximation Error of F B,m (Rahimi and Recht, 2009)) . Let Q be any fixed function in G B and define Π B,m Q ∈ F R,m as the solution to

<!-- formula-not-decoded -->

Then, there exists a constant C &gt; 0 such that, for any t &gt; B/ √ m , we have

Proof. See Rahimi and Recht (2009) for a detailed proof.

<!-- formula-not-decoded -->

Now we integrate the tail probability in (B.19) to obtain a bound on E init [ ‖ Π B,m Q -Q ‖ σ ]. Specifically, by direct computation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the second equality we let u = t · √ m/B -1. Thus, by (B.20) we have where ˜ T ˜ Q k -1 = Π B,m T ˜ Q k -1 . Thus, we conclude the second step. Step (3). Finally, in the last step, we utilize the existing analysis of projected SGD over function class F B,m to obtain an upper bound on ‖ T Q k -1 -Q k ‖ .

˜ ˜ ̂ Theorem B.6 (Convergence of Projected SGD (Liu et al., 2019a)) . In Algorithm 5, let T be the number of iterations of projected SGD steps for solving each iteration of the FQI update and we set η = O (1 / √ T ). Under Assumption B.2, it holds that

Proof. See Theorem 4.5 in Liu et al. (2019a) for a detailed proof.

<!-- formula-not-decoded -->

Finally, combining (B.17), (B.21), and (B.22) we obtain that

<!-- formula-not-decoded -->

Setting T /equivasymptotic m in (B.23), we obtain that

Combining (B.16) and (B.24), we conclude the proof of Theorem B.3.

<!-- formula-not-decoded -->

## C Proofs of Auxiliary Results

In this section, we present the proofs for Theorems 6.1 and 6.2, which are used in the § 6 to establish our main theorem.

## C.1 Proof of Theorem 6.1

Proof. Before we present the proof, we introduce some notation. For any k ∈ { 0 , . . . , K -1 } , we denote T ˜ Q k by Q k +1 and define

Also, we denote by π k the greedy policy with respect to ˜ Q k . In addition, throughout the proof, for two functions Q 1 , Q 2 : S × A → R , we use the notation Q 1 ≥ Q 2 if Q 1 ( s, a ) ≥ Q 2 ( s, a ) for any s ∈ S and any a ∈ A , and define Q 1 ≤ Q 2 similarly. Furthermore, for any policy π , recall that in (2.4) we define the operator P π by

In addition, we define the operator T π by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we denote R max / (1 -γ ) by V max . Now we are ready to present the proof, which consists of three key steps.

Step (i): In the first step, we establish a recursion that relates Q ∗ -˜ Q k +1 with Q ∗ -˜ Q k to measure the sub-optimality of the value function ˜ Q k . In the following, we first establish an upper bound for Q ∗ -˜ Q k +1 as follows. For each k ∈ { 0 , . . . , K -1 } , by the definition of /rho1 k +1 in (C.1), we have where π ∗ is the greedy policy with respect to Q ∗ . Now we leverage the following lemma to show T π ∗ ˜ Q k ≤ T ˜ Q k .

<!-- formula-not-decoded -->

Lemma C.1. For any action-value function Q : S × A → R and any policy π , it holds that

<!-- formula-not-decoded -->

Proof. Note that we have max a ′ Q ( s ′ , a ′ ) ≥ Q ( s ′ , a ′ ) for any s ′ ∈ S and a ′ ∈ A . Thus, it holds that

( TQ )( s, a ) = r ( s, a ) + γ · E [ max a ′ Q ( S ′ , a ′ ) ∣ ∣ S ′ ∼ P ( · | s, a ) ] ≥ r ( s, a ) + γ · E [ Q ( S ′ , A ′ ) ∣ ∣ S ′ ∼ P ( · | s, a ) , A ′ ∼ π ( · | S ′ ) ] = ( T π Q )( s, a ) . Recall that π Q is the greedy policy with respect to Q such that which implies

Consequently, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which concludes the proof of Lemma C.1.

<!-- formula-not-decoded -->

By Lemma C.1, we have T ˜ Q k ≥ T π ∗ ˜ Q k . Also note that Q ∗ is the unique fixed point of T π ∗ . Thus, by (C.3) we have

In the following, we establish a lower bound for Q ∗ -˜ Q k +1 based on ˜ Q ∗ -˜ Q k . Note that, by Lemma C.1, we have T π k ˜ Q k = T ˜ Q k and TQ ∗ ≥ T π k Q ∗ . Similar to (C.3), since Q ∗ is the unique fixed point of T , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, combining (C.4) and (C.5) we obtain that, for any k ∈ { 0 , . . . , K -1 } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The inequalities in (C.6) show that the error Q ∗ -˜ Q k +1 can be sandwiched by the summation of a term involving Q ∗ -˜ Q k and the error /rho1 k +1 , which is defined in (C.1) and induced by approximating the action-value function. Using P π defined in (C.2), we can write (C.6) in a more compact form,

Meanwhile, note that P π defined in (C.2) is a linear operator. In fact, P π is the Markov transition operator for the Markov chain on S × A with transition dynamics

<!-- formula-not-decoded -->

By the linearity of the operator P π and the one-step error bound in (C.6), we have the following characterization of the multi-step error.

Lemma C.2 (Error Propagation) . For any k, /lscript ∈ { 0 , 1 , . . . , K -1 } with k &lt; /lscript , we have

<!-- formula-not-decoded -->

π π ′ π k

<!-- formula-not-decoded -->

Here /rho1 i +1 is defined in (C.1) and we use P P and ( P ) to denote the composition of operators. Proof. Note that P π is a linear operator for any policy π . We obtain (C.8) and (C.9) by iteratively applying the inequalities in (C.7).

Lemma C.2 gives the upper and lower bounds for the propagation of error through multiple iterations of Algorithm 1, which concludes the first step of our proof.

Step (ii): The results in the first step only concern the propagation of error Q ∗ -˜ Q k . In contrast, the output of Algorithm 1 is the greedy policy π k with respect to ˜ Q k . In the second step, our goal is to quantify the suboptimality of Q π k , which is the action-value function corresponding to π k . In the following, we establish an upper bound for Q ∗ -Q π k .

To begin with, we have Q ∗ ≥ Q π k by the definition of Q ∗ in (2.5). Note that we have Q ∗ = T π ∗ Q ∗ and Q π k = T π k Q π k . Hence, it holds that

<!-- formula-not-decoded -->

Now we quantify the three terms on the right-hand side of (C.10) respectively. First, by Lemma C.1, we have

Meanwhile, by the definition of the operator P π in (C.2), we have

<!-- formula-not-decoded -->

T π ∗ Q ∗ -T π ∗ ˜ Q k = γ · P π ∗ ( Q ∗ -˜ Q k ) , T π k ˜ Q k -T π k Q π k = γ · P π k ( ˜ Q k -Q π k ) . (C.12) Plugging (C.11) and (C.12) into (C.10), we obtain which further implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here I is the identity operator. Since T π is a γ -contractive operator for any policy π , I -γ · P π is invertible. Thus, we obtain

<!-- formula-not-decoded -->

which relates Q ∗ -Q π k with Q ∗ -˜ Q k . In the following, we plug Lemma C.2 into (C.13) to obtain the multiple-step error bounds for Q π k . First note that, by the definition of P π in (C.2), for any functions f 1 , f 2 : S × A → R satisfying f 1 ≥ f 2 , we have P π f 1 ≥ P π f 2 . Combining this inequality with the upper bound in (C.8) and the lower bound in (C.9), we have that, for any k &lt; /lscript ,

<!-- formula-not-decoded -->

Then we plug (C.14) and (C.15) into (C.13) and obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any k &lt; /lscript . To quantify the error of Q π K , we set /lscript = K and k = 0 in (C.16) to obtain

<!-- formula-not-decoded -->

For notational simplicity, we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One can show that ∑ K i =0 α i = 1. Meanwhile, we define K +1 linear operators { O k } K k =0 by

Using this notation, for any ( s, a ) ∈ S × A , by (C.17) we have where both O i | /rho1 i +1 | and O K | Q ∗ -˜ Q 0 | are functions defined on S ×A . Here (C.19) gives a uniform upper bound for Q ∗ -Q π K , which concludes the second step.

<!-- formula-not-decoded -->

Step (iii): In this step, we conclude the proof by establishing an upper bound for ‖ Q ∗ -Q π K ‖ 1 ,µ based on (C.19). Here µ ∈ P ( S × A ) is a fixed probability distribution. To simplify the notation, for any measurable function f : S × A → R , we denote µ ( f ) to be the expectation of f under µ , that is, µ ( f ) = ∫ S×A f ( s, a )d µ ( s, a ) . Using this notation, by (C.19) we bound ‖ Q ∗ -Q π /lscript ‖ 1 ,µ by

By the linearity of expectation, (C.20) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, since both Q ∗ and ˜ Q 0 are bounded by V max = R max / (1 -γ ) in /lscript ∞ -norm, we have

Moreover, for any i ∈ { 0 , . . . , K -1 } , by expanding (1 -γP π K ) -1 into a infinite series, we have

<!-- formula-not-decoded -->

To upper bound the right-hand side of (C.23), we consider the following quantity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here τ 1 , . . . , τ m are m policies. Recall that P π is the transition operator of a Markov process defined on S × A for any policy π . Then the integral on the right-hand side of (C.24) corresponds to the expectation of the function f ( X t ), where { X t } t ≥ 0 is a Markov process defined on S × A . Such a Markov process has initial distribution X 0 ∼ µ . The first m transition operators are { P τ j } j ∈ [ m ] , followed by j identical transition operators P π K . Hence, ( P π K ) j ( P τ m P τ m -1 · · · P τ 1 ) µ is the marginal distribution of X j + m , which we denote by µ j for notational simplicity. Hence, (C.24) takes the form for any measurable function f on S × A . By Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

in which d ˜ µ j / d σ : S × A → R is the Radon-Nikodym derivative. Recall that the ( m + j )-th order concentration coefficient κ ( m + j ; µ, σ ) is defined in (4.4). Combining (C.25) and (C.26), we obtain

Thus, by (C.23) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we combine (C.21), (C.22), and (C.27) to obtain

<!-- formula-not-decoded -->

Recall that in Theorem 6.1 and (C.1) we define ε max = max i ∈ [ K ] ‖ /rho1 i ‖ σ . We have that ‖ Q ∗ -Q π K ‖ 1 ,µ is further upper bounded by

<!-- formula-not-decoded -->

where the last equality follows from the definition of { α i } 0 ≤ i ≤ K in (C.18). We simplify the summation on the right-hand side of (C.28) and use Assumption 4.3 to obtain

<!-- formula-not-decoded -->

where the last inequality follows from (4.5) in Assumption 4.3. Finally, combining (C.28) and (C.29), we obtain

<!-- formula-not-decoded -->

which concludes the third step and hence the proof of Theorem 6.1.

## C.2 Proof of Theorem 6.2

Proof. Recall that in Algorithm 1 we define Y i = R i + γ · max a ∈A Q ( S i +1 , a ), where Q is any function in F . By definition, we have E ( Y i | S i = s, A i = a ) = ( TQ )( s, a ) for any ( s, a ) ∈ S × A . Thus, TQ can be viewed as the underlying truth of the regression problem defined in (6.2), where the covariates and responses are { ( S i , A i ) } i ∈ [ n ] and { Y i } i ∈ [ n ] , respectively. Moreover, note that TQ is not necessarily in function class F . We denote by Q ∗ the best approximation of TQ in F , which is the solution to

<!-- formula-not-decoded -->

̂ In the following, we prove (6.3) in two steps, which are bridged by E [ ‖ Q -TQ ‖ 2 n ].

For notational simplicity, in the sequel we denote ( S i , A i ) by X i for all i ∈ [ n ]. For any f ∈ F , we define ‖ f ‖ 2 n = 1 /n · ∑ n i =1 [ f ( X i )] 2 . Since both ̂ Q and TQ are bounded by V max = R max / (1 -γ ), we only need to consider the case where log N δ ≤ n . Here N δ is the cardinality of N ( δ, F , ‖ · ‖ ∞ ). Moreover, let f 1 , . . . , f N δ be the centers of the minimal δ -covering of F . Then by the definition of δ -covering, there exists k ∗ ∈ [ N δ ] such that ‖ ̂ Q -f k ∗ ‖ ∞ ≤ δ . It is worth mentioning that k ∗ is a random variable since Q is obtained from data.

<!-- formula-not-decoded -->

̂ Step (i): We relate E [ ‖ ̂ Q -TQ ‖ 2 n ] with its empirical counterpart ‖ ̂ Q -TQ ‖ 2 n . Recall that we define Y i = R i + γ · max a ∈A Q ( S i +1 , a ) for each i ∈ [ n ]. By the definition of Q , for any f ∈ F we have

For each i ∈ [ n ], we define ξ i = Y i -( TQ )( X i ). Then (C.31) can be written as

<!-- formula-not-decoded -->

Since both f and Q are deterministic, we have E ( ‖ f -TQ ‖ 2 n ) = ‖ f -TQ ‖ 2 σ . Moreover, since E ( ξ i | X i ) = 0 by definition, we have E [ ξ i · g ( X i )] = 0 for any bounded and measurable function g . Thus, it holds that

<!-- formula-not-decoded -->

In addition, by triangle inequality and (C.33), we have

<!-- formula-not-decoded -->

where f k ∗ satisfies ‖ f k ∗ -̂ Q ‖ ∞ ≤ δ . In the following, we upper bound the two terms on the righthand side of (C.34) respectively. For the first term, by applying Cauchy-Schwarz inequality twice, we have where we use the fact that { ξ i } i ∈ [ n ] have the same marginal distributions and ‖ ̂ Q -f k ∗ ‖ n ≤ δ . Since both Y i and TQ are bounded by V max , ξ i is a bounded random variable by its definition. Thus, there exists a constant C ξ &gt; 0 depending on ξ such that E ( ξ 2 i ) ≤ C 2 ξ · V 2 max . Then (C.35) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It remains to upper bound the second term on the right-hand side of (C.34). We first define N δ self-normalized random variables

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all j ∈ [ N δ ]. Here recall that { f j } j ∈ [ N δ ] are the centers of the minimal δ -covering of F . Then we have where the first inequality follows from triangle inequality and the second inequality follows from the fact that ‖ ̂ Q -f k ∗ ‖ ∞ ≤ δ . Then applying Cauchy-Schwarz inequality to the last term on the right-hand side of (C.38), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, since ξ i is centered conditioning on { X i } i ∈ [ n ] and is bounded by 2 V max , ξ i is a subGaussian random variable. In specific, there exists an absolute constant H ξ &gt; 0 such that ‖ ξ i ‖ ψ 2 ≤ H ξ · V max for each i ∈ [ n ]. Here the ψ 2 -norm of a random variable W ∈ R is defined as

By the definition of Z j in (C.37), conditioning on { X i } i ∈ [ n ] , ξ i · [ f j ( X i ) -( TQ )( X i )] is a centered and sub-Gaussian random variable with

<!-- formula-not-decoded -->

Moreover, since Z j is a summation of independent sub-Gaussian random variables, by Lemma 5.9 of Vershynin (2010), the ψ 2 -norm of Z j satisfies

<!-- formula-not-decoded -->

where C &gt; 0 is an absolute constant. Furthermore, by Lemmas 5.14 and 5.15 of Vershynin (2010), Z 2 j is a sub-exponential random variable, and its the moment-generating function is bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any t satisfying C ′ · | t | · H 2 ξ · V 2 max ≤ 1, where C and C ′ are two positive absolute constants. Moreover, by Jensen's inequality, we bound the moment-generating function of max j ∈ [ N δ ] Z 2 j by

Combining (C.40) and (C.41), we have

<!-- formula-not-decoded -->

where C &gt; 0 is an absolute constant. Hence, plugging (C.42) into (C.38) and (C.39), we upper bound the second term of the right-hand side of (C.33) by

<!-- formula-not-decoded -->

Finally, combining (C.32), (C.36) and (C.43), we obtain the following inequality

<!-- formula-not-decoded -->

where C and C ′ are two positive absolute constants. Here in the first inequality we take the infimum over F because (C.31) holds for any f ∈ F , and the second inequality holds because log N δ ≤ n .

Now we invoke a simple fact to obtain the final bound for E [ ‖ ̂ Q -TQ ‖ 2 n ] from (C.44). Let a, b, and c be positive numbers satisfying a 2 ≤ 2 ab + c . For any /epsilon1 ∈ (0 , 1] , since 2 ab ≤ /epsilon1 · a 2 / (1+ /epsilon1 )+(1+ /epsilon1 ) · b 2 //epsilon1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, applying (C.45) to (C.44) with a 2 = E [ ‖ ̂ Q -TQ ‖ 2 n ], b = C · V max · √ log N δ /n , and c = inf f ∈F E [ ‖ f -TQ ‖ 2 n ] + C ′ · V max · δ , we obtain

Step (ii). In this step, we relate the population risk ‖ ̂ Q -TQ ‖ 2 σ with E [ ‖ ̂ Q -TQ ‖ 2 n ], which is characterized in the first step. To begin with, we generate n i.i.d. random variables { ˜ X i = ( ˜ S i , ˜ A i ) } i ∈ [ n ] following σ , which are independent of { ( S i , A i , R i , S ′ i ) } i ∈ [ n ] . Since ‖ ̂ Q -f k ∗ ‖ ∞ ≤ δ , for any x ∈ S × A , we have

∣ ̂ ∣ ∣ ̂ ∣ where the last inquality follows from the fact that ‖ TQ ‖ ∞ ≤ V max and ‖ f ‖ ∞ ≤ V max for any f ∈ F . Then by the definition of ‖ Q -TQ ‖ 2 σ and (C.47), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we apply (C.47) to obtain the first inequality, and in the last equality we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any ( x, y ) ∈ S ×A and any j ∈ [ N δ ]. Note that h k ∗ is a random function since k ∗ is random. By the definition of h j in (C.49), we have | h j ( x, y ) | ≤ 4 V 2 max for any ( x, y ) ∈ S×A and E [ h j ( X i , ˜ X i )] = 0 for any i ∈ [ n ]. Moreover, the variance of h j ( X i , X i ) is upper bounded by where we define Υ by letting

<!-- formula-not-decoded -->

Furthermore, we define

Combining (C.48) and (C.51), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the sequel, we utilize Bernstein's inequality to establish an upper bound for E ( T ), which is stated as follows for completeness.

Lemma C.3 (Bernstein's Inequality) . Let U 1 , . . . U n be n independent random variables satisfying E ( U i ) = 0 and | U i | ≤ M for all i ∈ [ n ]. Then for any t &gt; 0, we have where σ 2 = ∑ n i =1 Var( U i ) is the variance of ∑ n i =1 U i .

<!-- formula-not-decoded -->

We first apply Bernstein's inequality by setting U i = h j ( X i , ˜ X i ) / Υ for each i ∈ [ n ]. Then we take a union bound for all j ∈ [ N δ ] to obtain

Since T is nonnegative, we have E ( T ) = ∫ ∞ 0 P ( T ≥ t )d t . Thus, for any u ∈ (0 , 3Υ · n ), by (C.53) it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the second inequality we use the fact that ∫ ∞ s exp( -t 2 / 2)d t ≤ 1 /s · exp( -s 2 / 2) for all s &gt; 0. Now we set u = 4 V max √ n · log N δ in (C.54) and plug in the definition of Υ in (C.50) to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality holds when log N δ ≥ 4 . Moreover, the definition of Υ in (C.50) implies that Υ ≤ max[2 V max √ log N δ /n, ‖ ̂ Q -TQ ‖ σ + δ ] . In the following, we only need to consider the case where Υ ≤ ‖ ̂ Q -TQ ‖ σ + δ , since we already have (6.3) if ‖ ̂ Q -TQ ‖ σ + δ ≤ 2 V max √ log N δ /n , which concludes the proof.

We apply the inequality in (C.45) to (C.56) with a = ‖ ̂ Q -TQ ‖ σ , b = 8 V max √ log N δ /n , and c = E [ ‖ ̂ Q -TQ ‖ 2 n ] + 16 V max · δ . Hence we finally obtain that

<!-- formula-not-decoded -->

which concludes the second step of the proof.

Finally, combining these two steps together, namely, (C.46) and (C.57), we conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C 1 and C 2 are two absolute constants. Moreover, since Q ∈ F , we have which concludes the proof of Theorem 6.2.

## D Proof of Theorem 5.4

In this section, we present the proof of Theorem 5.4. The proof is similar to that of Theorem 4.4, which is presented in § 6 in details. In the following, we follow the proof in § 6 and only highlight the differences for brevity.

Proof. The proof requires two key ingredients, namely the error propagation and the statistical error incurred by a single step of Minimax-FQI. We note that P´ erolat et al. (2015) establish error propagation for the state-value functions in the approximate modified policy iteration algorithm, which is more general than the FQI algorithm.

Theorem D.1 (Error Propagation) . Recall that { ˜ Q k } 0 ≤ k ≤ K are the iterates of Algorithm 2 and ( π K , ν K ) is the equilibrium policy with respect to ˜ Q K . Let Q ∗ K be the action-value function corresponding to ( π K , ν ∗ π K ), where ν ∗ π K is the best-response policy of the second player against π K . Then under Assumption 5.3, we have

<!-- formula-not-decoded -->

where we define the maximum one-step approximation error ε max = max k ∈ [ K ] ‖ T ˜ Q k -1 -˜ Q k ‖ σ , and constant φ µ,ν is specified in Assumption 5.3.

Proof. We note that the proof of Theorem 6.1 cannot be directly applied to prove this theorem. The main reason is that here we also need to consider the role played by the opponent, namely player two. Different from the MDP setting, here Q ∗ K is a fixed point of a nonlinear operator due to the fact that player two adopts the optimal policy against π K . Thus, we need to conduct a more refined analysis. See § D.1 for a detailed proof.

By this theorem, we need to derive an upper bound of ε max . We achieve such a goal by studying the one-step approximation error ‖ T ˜ Q k -1 -˜ Q k ‖ σ for each k ∈ [ K ].

<!-- formula-not-decoded -->

Theorem D.2 (One-step Approximation Error) . Let F ⊆ B ( S ×A×B , V max ) be a family of measurable functions on S×A×B that are bounded by V max = R max / (1 -γ ). Also, let { ( S i , A i , B i ) } i ∈ [ n ] be n i.i.d. random variables following distribution σ ∈ P ( S ×A×B ). . For each i ∈ [ n ], let R i and S ′ i be the reward obtained by the first player and the next state following ( S i , A i , B i ). In addition, for any fixed Q ∈ F , we define the response variable as

Based on { ( X i , A i , Y i ) } i ∈ [ n ] , we define ̂ Q as the solution to the least-squares problem

<!-- formula-not-decoded -->

Then for any /epsilon1 ∈ (0 , 1] and any δ &gt; 0, we have

<!-- formula-not-decoded -->

where C and C ′ are two positive absolute constants, T is the Bellman operator defined in (5.5), N δ is the cardinality of the minimal δ -covering of F with respect to /lscript ∞ -norm.

Proof. By the definition of Y i in (D.2), for any ( s, a, b ) ∈ S × A × min ν ′ ∈P ( B ) , we have

<!-- formula-not-decoded -->

Thus, TQ can be viewed as the ground truth of the nonlinear least-squares regression problem in (D.3). Therefore, following the same proof of Theorem 6.2, we obtain the desired result.

Now we let F be the family of ReLU Q-networks F 1 defined in (5.9) and set Q = ˜ Q k -1 in Theorem D.2. In addition, setting /epsilon1 = 1 and δ = 1 /n in (D.4), we obtain

<!-- formula-not-decoded -->

where C is a positive absolute constant, N 1 is the 1 /n -covering number of F 1 , and function class G 1 is defined as

Here the second inequality follows from Assumption 5.2.

<!-- formula-not-decoded -->

Thus, it remains to bound the /lscript ∞ -error of approximating functions in G 1 using ReLU Q-networks

˜ F { j } j =1

in F 1 and the 1 /n -covering number of F 1 . In the sequel, obtain upper bounds for these two terms. By the definition of G 1 in (D.6), for any f ∈ G 1 and any ( a, b ) ∈ A × B , we have f ( · , a, b ) ∈ G ( { ( p j , t j , β j , H j ) } j ∈ [ q ] ). Following the same construction as in § C.2, we can find a function f in ( L ∗ , d ∗ L ∗ +1 , s ∗ ) such that which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, for any f ∈ F 1 and any ( a, b ) ∈ A × B , we have f ( · , a, b ) ∈ F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ). Let N δ be the δ -covering of F ( L ∗ , { d ∗ j } L ∗ +1 j =1 , s ∗ ) in the /lscript ∞ -norm. Then for any f ∈ F 1 and any ( a, b ) ∈ A × B , there exists g ab ∈ N δ such that ‖ f ( · , a, b ) -g a,b ‖ ∞ ≤ δ . Thus, the cardinality of the N ( δ, F 1 , ‖ · ‖ ∞ ) satisfies

Combining (D.8) with Lemma 6.4 and setting δ = 1 /n , we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, combining (D.1), (D.5), (D.7), and (D.9), we conclude the proof of Theorem 5.4.

where D = ∏ L ∗ +1 /lscript =1 ( d ∗ /lscript +1) and the second inequality follows from (4.7).

## D.1 Proof of Theorem D.1

Proof. The proof is similar to the that of Theorem D.1. Before presenting the proof, we first introduce the following notation for simplicity. For any k ∈ { 0 , . . . , K -1 } , we denote T ˜ Q k by Q k +1 and define /rho1 k = Q k -˜ Q k . In addition, throughout the proof, for two action-value functions Q 1 and Q 2 , we write Q 1 ≤ Q 2 if Q 1 ( s, a, b ) ≥ Q 2 ( s, a, b ) for any ( s, a, b ) ∈ S × A × B , and define Q 1 ≥ Q 2 similarly. Furthermore, we denote by ( π k , ν k ) and ( π ∗ , ν ∗ ) the equilibrium policies with respect to ˜ Q k by Q ∗ , respectively. Besides, in addition to the Bellman operators T π,ν and T defined in (5.4) and (5.5), for any policy π of the first player, we define corresponds to the case where the first player follows policy π and player 2 adopts the best policy in response to π . By this definition, it holds that Q ∗ = T π ∗ Q ∗ . Unlike the MDP setting, here T π

<!-- formula-not-decoded -->

is a nonlinear operator due to the minimization in (D.10). Furthermore, for any fixed action-value function Q , we define the best-response policy against π with respect to Q , denote by ν ( π, Q ), as

<!-- formula-not-decoded -->

Using this notation, we can write (D.10) equivalently as

<!-- formula-not-decoded -->

Notice that P π,ν ( π,Q ) is a linear operator and that ν Q = ν ( π Q , Q ) by definition.

Now we are ready to present the proof, which can be decomposed into three key steps.

Step (i): In the first step, we establish recursive upper and lower bounds for { Q ∗ -˜ Q k } 0 ≤ k ≤ K . For each k ∈ { 0 , . . . , K -1 } , similar to the decomposition in (C.3), we have where π ∗ is part of the equilibrium policy with respect to Q ∗ and T π ∗ is defined in (D.10).

Similar to Lemma C.1, we utilize the following lemma to show T π ∗ ˜ Q k ≥ T ˜ Q k . Lemma D.3. For any action-value function Q : S × A × B → R , let ( π Q , ν Q ) be the equilibrium policy with respect to Q . Then for and any policy π of the first player, it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, for any policy π : S → P ( A ) of player one and any action-value function Q , we have

<!-- formula-not-decoded -->

for any policy ν : S → P ( B ), where ν ( π, Q ) is the best-response policy defined in (D.11).

Proof. Note that for any s ′ ∈ S , by the definition of equilibrium policy, we have

Thus, for any state-action tuple ( s, a, b ), taking conditional expectations of s with respect to P ( · | s, a, b ) on both ends of this equation, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves T π Q Q = TQ . Moreover, for any policy π of the first player, it holds that

<!-- formula-not-decoded -->

Taking expectations with respect to s ′ ∼ P ( · | s, a, b ) on both ends, we establish TQ ≥ T π Q .

It remains to show the second part of Lemma D.3. By the definition of ν ( π, Q ), we have

<!-- formula-not-decoded -->

which, combined with the definition of T π in (D.10), implies that T π,ν ( π,Q ) Q = T π Q . Finally, for any policy ν of player two, we have

<!-- formula-not-decoded -->

which yields T π Q ≤ T π,ν Q . Thus, we conclude the proof of this lemma.

Hereafter, for notational simplicity, for each k , let ( π k , ν k ) be the equilibrium joint policy with respect to ˜ Q k , and we denote ν ( π ∗ , ˜ Q k ) and ν ( π k , Q ∗ ) by ˜ ν k and ¯ ν k , respectively. Applying Lemma D.3 to (D.12) and utilizing the fact that Q ∗ = T π ∗ Q ∗ , we have where the last inequality follows from (D.13). Furthermore, for a lower bound of Q ∗ -˜ Q k +1 , similar to (C.5), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any k ∈ { 0 , . . . , K -1 } . Similar to the proof of Lemma C.2 , by applying recursion to (D.16), we obtain the following upper and lower bounds for the error propagation of Algorithm 2.

Lemma D.4 (Error Propagation) . For any k, /lscript ∈ { 0 , 1 , . . . , K -1 } with k &lt; /lscript , we have

<!-- formula-not-decoded -->

Proof. The desired results follows from applying the inequalities in (D.16) multiple times and the linearity of the operator P π,ν for any joint policy ( π, ν ).

<!-- formula-not-decoded -->

The above lemma establishes recursive upper and lower bounds for the error terms { Q ∗ -Q k } 0 ≤ k ≤ K -1 , which completes the first step of the proof.

˜ Step (ii): In the second step, we characterize the suboptimality of the equilibrium policies constructed by Algorithm 2. Specifically, for each π k , we denote by Q ∗ k the action-value function obtained when agent one follows π k while agent two adopt the best-response policy against π k . In other words, Q ∗ k is the fixed point of Bellman operator T π k defined in (D.10). In the following, we obtain an upper bound of Q ∗ -Q ∗ k , which establishes the a notion of suboptimality of policy ( π k , ν k ) from the perspective of the first player.

To begin with, for any k , we first decompose Q ∗ -Q ∗ k by

Since π k is the equilibrium policy with respect to ˜ Q k , by Lemma D.3, we have T π ∗ ˜ Q k ≤ T π k ˜ Q k . Recall that ( π ∗ , ν ∗ ) is the joint equilibrium policy with respect to Q ∗ . The second argument of Lemma D.3 implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ ν k = ν ( π ∗ , ˜ Q k ) and we define ̂ ν k = ν ( π k , Q ∗ k ). Thus, combining (D.19) and (D.20) yields that

Furthermore, since I -γ · P π k , ̂ ν k is invertible, by (D.21) we have

Now we apply Lemma D.4 to the right-hand side of (D.22). Then for any k ≤ /lscript , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, setting /lscript = K and k = 0 in (D.23) and (D.24), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To simplify the notation, we define { α i } K i =0 as in (C.18). Note that we have ∑ K i =0 α i = 1 by definition. Moreover, we define K +1 linear operators { O k } K k =0 as follows. For any i ≤ K -1, let

Moreover, we define O K by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, taking absolute values on both sides of (D.25), we obtain that

<!-- formula-not-decoded -->

for any ( s, a, b ) ∈ S × A × B , which concludes the second step of the proof.

Step (iii): We note that (D.26) is nearly the same as (C.19) for the MDP setting. Thus, in the last step, we follow the same proof strategy as in Step (iii) in § C.1. For notational simplicity, for any function f : S × A × B → R and any probability distribution µ ∈ P ( S × A × B ), we denote the expectation of f under µ by µ ( f ). By taking expectation with respect to µ in (D.26), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the definition of O i , we can write µ ( O i | /rho1 i +1 | ) as

To upper bound the right-hand side of (D.28), we consider the following quantity

<!-- formula-not-decoded -->

where { τ t : S → P ( A × B ) } t ∈ [ m ] are m joint policies of the two-players. By Cauchy-Schwarz inequality, it holds that

<!-- formula-not-decoded -->

where κ ( m ; µ, σ ) is the m -th concentration parameter defined in (5.10). Thus, by (D.28)we have

<!-- formula-not-decoded -->

Besides, since both Q ∗ and ˜ Q 0 are bounded by R max / (1 -γ ) in /lscript ∞ -norm, we have

Finally, combining (D.27), (D.29), and (D.30), we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from the fact that ε max = max i ∈ [ K ] ‖ /rho1 i ‖ σ . Note that in (C.29) we show that it holds under Assumption 5.3 that

<!-- formula-not-decoded -->

Hence, we obtain (D.1) and thus conclude the proof of Theorem D.1.

## References

- Agarwal, R., Schuurmans, D. and Norouzi, M. (2019). Striving for simplicity in off-policy deep reinforcement learning. arXiv preprint arXiv:1907.04543 .
- Allen-Zhu, Z., Li, Y. and Liang, Y. (2019a). Learning and generalization in overparameterized neural networks, going beyond two layers. In Advances in Neural Information Processing Systems .
- Allen-Zhu, Z., Li, Y. and Song, Z. (2019b). A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning .
- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, O. P. and Zaremba, W. (2017). Hindsight experience replay. In Advances in Neural Information Processing Systems .
- Anthony, M. and Bartlett, P. L. (2009). Neural network learning: Theoretical foundations . Cambridge University Press.
- Antos, A., Szepesv´ ari, C. and Munos, R. (2007). Value-iteration based fitted policy iteration: Learning with a single trajectory. In IEEE International Symposium on Approximate Dynamic Programming and Reinforcement Learning .
- Antos, A., Szepesv´ ari, C. and Munos, R. (2008a). Fitted Q-iteration in continuous action-space mdps. In Advances in Neural Information Processing Systems .
- Antos, A., Szepesv´ ari, C. and Munos, R. (2008b). Learning near-optimal policies with Bellmanresidual minimization based fitted policy iteration and a single sample path. Machine Learning , 71 89-129.
- Arora, S., Du, S., Hu, W., Li, Z. and Wang, R. (2019). Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning .
- Arulkumaran, K., Deisenroth, M. P., Brundage, M. and Bharath, A. A. (2017). A brief survey of deep reinforcement learning. arXiv preprint arXiv:1708.05866 .
- Bach, F. (2017). On the equivalence between kernel quadrature rules and random feature expansions. The Journal of Machine Learning Research , 18 714-751.
- Bai, Y. and Lee, J. D. (2019). Beyond linearization: On quadratic and higher-order approximation of wide neural networks. arXiv preprint arXiv:1910.01619 .
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. In Machine Learning Proceedings 1995 . 30-37.
- Barron, A. R. and Klusowski, J. M. (2018). Approximation and estimation for high-dimensional deep learning networks. arXiv preprint arXiv:1809.03090 .

- Bartlett, P. L. (1998). The sample complexity of pattern classification with neural networks: The size of the weights is more important than the size of the network. IEEE Transactions on Information Theory , 44 525-536.
- Bartlett, P. L., Foster, D. J. and Telgarsky, M. J. (2017). Spectrally-normalized margin bounds for neural networks. In Advances in Neural Information Processing Systems .
- Bartlett, P. L., Harvey, N., Liaw, C. and Mehrabian, A. (2019). Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research , 20 1-17.
- Bartlett, P. L., Maiorov, V. and Meir, R. (1999). Almost linear VC dimension bounds for piecewise polynomial networks. In Advances in Neural Information Processing Systems .
- Bauer, B., Kohler, M. et al. (2019). On deep learning as a remedy for the curse of dimensionality in nonparametric regression. The Annals of Statistics , 47 2261-2285.
- Bellemare, M. G., Dabney, W. and Munos, R. (2017). A distributional perspective on reinforcement learning. In International Conference on Machine Learning .
- Bellemare, M. G., Naddaf, Y., Veness, J. and Bowling, M. (2013). The Arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47 253-279.
- Bietti, A. and Mairal, J. (2019). On the inductive bias of neural tangent kernels. In Advances in Neural Information Processing Systems .
- Bowling, M. (2001). Rational and convergent learning in stochastic games. In International Conference on Artificial Intelligence .
- Boyan, J. A. (2002). Technical update: Least-squares temporal difference learning. Machine Learning , 49 233-246.
- Bradtke, S. J. and Barto, A. G. (1996). Linear least-squares algorithms for temporal difference learning. Machine learning , 22 33-57.
- Cai, Q., Yang, Z., Lee, J. D. and Wang, Z. (2019). Neural temporal-difference learning converges to global optima. In Advances in Neural Information Processing Systems .
- Cao, Y. and Gu, Q. (2019). A generalization theory of gradient descent for learning overparameterized deep ReLU networks. arXiv preprint arXiv:1902.01384 .
- Chakraborty, B. (2013). Statistical methods for dynamic treatment regimes . Springer.
- Chen, H., Liu, X., Yin, D. and Tang, J. (2017). A survey on dialogue systems: Recent advances and new frontiers. ACM SIGKDD Explorations Newsletter , 19 25-35.

- Chen, J. and Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning .
- Chizat, L., Oyallon, E. and Bach, F. (2019). On lazy training in differentiable programming. In Advances in Neural Information Processing Systems .
- Conitzer, V. and Sandholm, T. (2007). AWESOME: A general multiagent learning algorithm that converges in self-play and learns a best response against stationary opponents. Machine Learning , 67 23-43.
- Dabney, W., Ostrovski, G., Silver, D. and Munos, R. (2018a). Implicit quantile networks for distributional reinforcement learning. In International Conference on Machine Learning .
- Dabney, W., Rowland, M., Bellemare, M. G. and Munos, R. (2018b). Distributional reinforcement learning with quantile regression. In AAAI Conference on Artificial Intelligence .
- Du, S., Lee, J., Li, H., Wang, L. and Zhai, X. (2019a). Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning .
- Du, S. S., Zhai, X., Poczos, B. and Singh, A. (2019b). Gradient descent provably optimizes overparameterized neural networks. In International Conference on Learning Representations .
- Dulac-Arnold, G., Evans, R., Sunehag, P. and Coppin, B. (2015). Reinforcement learning in large discrete action spaces. arXiv preprint arXiv:1512.07679 .
- Dziugaite, G. K. and Roy, D. M. (2017). Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008 .
- Ernst, D., Geurts, P. and Wehenkel, L. (2005). Tree-based batch mode reinforcement learning. Journal of Machine Learning Research , 6 503-556.
- Fan, J., Ma, C. and Zhong, Y. (2019). A selective overview of deep learning. arXiv preprint arXiv:1904.05526 .
- Farahmand, A.-m., Ghavamzadeh, M., Szepesv´ ari, C. and Mannor, S. (2009). Regularized fitted Q-iteration for planning in continuous-space Markovian decision problems. In American Control Conference .
- Farahmand, A.-m., Ghavamzadeh, M., Szepesv´ ari, C. and Mannor, S. (2016). Regularized policy iteration with nonparametric function spaces. The Journal of Machine Learning Research , 17 4809-4874.
- Farahmand, A.-m., Szepesv´ ari, C. and Munos, R. (2010). Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems .
- Frankle, J. and Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations .

- Frei, S., Cao, Y. and Gu, Q. (2019). Algorithm-dependent generalization bounds for overparameterized deep residual networks. In Advances in Neural Information Processing Systems .
- Friedman, J. H. and Stuetzle, W. (1981). Projection pursuit regression. Journal of the American Statistical Association , 76 817-823.
- Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019). Benchmarking batch deep reinforcement learning algorithms. arXiv preprint arXiv:1910.01708 .
- Gao, R., Cai, T., Li, H., Wang, L., Hsieh, C.-J. and Lee, J. D. (2019). Convergence of adversarial training in overparametrized networks. In Advances in Neural Information Processing Systems .
- Geist, M., Scherrer, B. and Pietquin, O. (2019). A theory of regularized Markov decision processes. In International Conference on Machine Learning .
- Goldberg, Y. and Kosorok, M. R. (2012). Q-learning with censored data. Annals of statistics , 40 529.
- Goldberg, Y., Song, R. and Kosorok, M. R. (2013). Adaptive Q-learning. In From Probability to Statistics and Back: High-Dimensional Models and Processes-A Festschrift in Honor of Jon A. Wellner . Institute of Mathematical Statistics, 150-162.
- Golowich, N., Rakhlin, A. and Shamir, O. (2018). Size-independent sample complexity of neural networks. In Conference on Learning Theory .
- Haarnoja, T., Tang, H., Abbeel, P. and Levine, S. (2017). Reinforcement learning with deep energybased policies. In International Conference on Machine Learning .
- Han, S., Mao, H. and Dally, W. J. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations .
- Hausknecht, M. and Stone, P. (2015). Deep recurrent Q-learning for partially observable MDPs. In AAAI Conference on Artificial Intelligence .
- Hofmann, T., Sch¨ olkopf, B. and Smola, A. J. (2008). Kernel methods in machine learning. Annals of Statistics 1171-1220.
- Huang, K., Wang, Y., Tao, M. and Zhao, T. (2020). Why do deep residual networks generalize better than deep feedforward networks?-A neural tangent kernel perspective. arXiv preprint arXiv:2002.06262 .
- Jacot, A., Gabriel, F. and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems .
- Kawaguchi, K., Kaelbling, L. P. and Bengio, Y. (2017). Generalization in deep learning. arXiv preprint arXiv:1710.05468 .

- Klusowski, J. M. and Barron, A. R. (2016). Risk bounds for high-dimensional ridge function combinations including neural networks. arXiv preprint arXiv:1607.01434 .
- Kober, J. and Peters, J. (2012). Reinforcement learning in robotics: A survey. In Reinforcement Learning . Springer, 579-610.
- Konda, V. R. and Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in Neural Information Processing Systems .
- Laber, E. B., Lizotte, D. J., Qian, M., Pelham, W. E. and Murphy, S. A. (2014). Dynamic treatment regimes: Technical challenges and applications. Electronic Journal of Statistics , 8 1225.
- Lagoudakis, M. G. and Parr, R. (2002). Value function approximation in zero-sum Markov games. In Uncertainty in Artificial Intelligence .
- Lagoudakis, M. G. and Parr, R. (2003). Least-squares policy iteration. Journal of machine learning research , 4 1107-1149.
- Lange, S., Gabel, T. and Riedmiller, M. (2012). Batch reinforcement learning. In Reinforcement learning . Springer, 45-73.
- Lazaric, A., Ghavamzadeh, M. and Munos, R. (2012). Finite-sample analysis of least-squares policy iteration. Journal of Machine Learning Research , 13 3041-3074.
- Lazaric, A., Ghavamzadeh, M. and Munos, R. (2016). Analysis of classification-based policy iteration algorithms. The Journal of Machine Learning Research , 17 583-612.
- Levine, N., Zahavy, T., Mankowitz, D. J., Tamar, A. and Mannor, S. (2017). Shallow updates for deep reinforcement learning. In Advances in Neural Information Processing Systems .
- Liang, T., Poggio, T., Rakhlin, A. and Stokes, J. (2019). Fisher-Rao metric, geometry, and complexity of neural networks. In International Conference on Artificial Intelligence and Statistics .
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D. (2016). Continuous control with deep reinforcement learning. In International Conference on Learning Representations .
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning , 8 293-321.
- Linn, K. A., Laber, E. B. and Stefanski, L. A. (2017). Interactive q-learning for quantiles. Journal of the American Statistical Association , 112 638-649.
- Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. In Machine Learning Proceedings 1994 . Elsevier, 157-163.
- Liu, B., Cai, Q., Yang, Z. and Wang, Z. (2019a). Neural proximal/trust region policy optimization attains globally optimal policy. In Advances in Neural Information Processing Systems .

- Liu, B., Wang, M., Foroosh, H., Tappen, M. and Pensky, M. (2015). Sparse convolutional neural networks. In Conference on Computer Vision and Pattern Recognition .
- Liu, N., Liu, Y., Logan, B., Xu, Z., Tang, J. and Wang, Y. (2019b). Learning the dynamic treatment regimes from medical registry data through deep Q-network. Scientific Reports , 9 1-10.
- Liu, R. and Zou, J. (2018). The effects of memory replay in reinforcement learning. In Allerton Conference on Communication, Control, and Computing .
- Maass, W. (1994). Neural nets with superlinear VC-dimension. Neural Computation , 6 877-884.
- Mei, S., Misiakiewicz, T. and Montanari, A. (2019). Mean-field theory of two-layers neural networks: Dimension-free bounds and kernel limit. In Conference on Learning Theory .
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning .
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G. et al. (2015). Human-level control through deep reinforcement learning. Nature , 518 529-533.
- Mohri, M., Rostamizadeh, A. and Talwalkar, A. (2012). Foundations of machine learning . MIT Press.
- Munos, R. and Szepesv´ ari, C. (2008). Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9 815-857.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. Journal of the Royal Statistical Society, Series B , 65 331-355.
- Murphy, S. A. (2005). A generalization error for Q-learning. Journal of Machine Learning Research , 6 1073-1097.
- Nahum-Shani, I., Qian, M., Almirall, D., Pelham, W. E., Gnagy, B., Fabiano, G. A., Waxmonsky, J. G., Yu, J. and Murphy, S. A. (2012). Q-learning: A data analysis method for constructing adaptive interventions. Psychological methods , 17 478.
- Nair, V. and Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In International Conference on Machine Learning .
- Neyshabur, B., Bhojanapalli, S., McAllester, D. and Srebro, N. (2017). Exploring generalization in deep learning. In Advances in Neural Information Processing Systems .
- Neyshabur, B., Bhojanapalli, S., McAllester, D. and Srebro, N. (2018). A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations .

- Neyshabur, B., Salakhutdinov, R. R. and Srebro, N. (2015a). Path-SGD: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems .
- Neyshabur, B., Tomioka, R. and Srebro, N. (2015b). Norm-based capacity control in neural networks. In Conference on Learning Theory .
- Novati, G. and Koumoutsakos, P. (2019). Remember and forget for experience replay. In International Conference on Machine Learning .
- Patek, S. D. (1997). Stochastic and shortest path games: Theory and algorithms . Ph.D. thesis, Massachusetts Institute of Technology.
- P´ erolat, J., Piot, B., Geist, M., Scherrer, B. and Pietquin, O. (2016a). Softened approximate policy iteration for Markov games. In International Conference on Machine Learning .
- P´ erolat, J., Piot, B. and Pietquin, O. (2018). Actor-critic fictitious play in simultaneous move multistage games. In International Conference on Artificial Intelligence and Statistics .
- P´ erolat, J., Piot, B., Scherrer, B. and Pietquin, O. (2016b). On the use of non-stationary strategies for solving two-player zero-sum Markov games. In International Conference on Artificial Intelligence and Statistics .
- P´ erolat, J., Scherrer, B., Piot, B. and Pietquin, O. (2015). Approximate dynamic programming for two-player zero-sum Markov games. In International Conference on Machine Learning .
- Prasad, H., LA, P. and Bhatnagar, S. (2015). Two-timescale algorithms for learning Nash equilibria in general-sum stochastic games. In International Conference on Autonomous Agents and Multiagent Systems .
- Qian, M. and Murphy, S. A. (2011). Performance guarantees for individualized treatment rules. Annals of Statistics , 39 1180.
- Rahimi, A. and Recht, B. (2008). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems .
- Rahimi, A. and Recht, B. (2009). Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. In Advances in Neural Information Processing Systems .
- Riedmiller, M. (2005). Neural fitted Q iteration - First experiences with a data efficient neural reinforcement learning method. In European Conference on Machine Learning .
- Schaul, T., Quan, J., Antonoglou, I. and Silver, D. (2016). Prioritized experience replay. International Conference on Learning Representations .
- Scherrer, B., Ghavamzadeh, M., Gabillon, V., Lesner, B. and Geist, M. (2015). Approximate modified policy iteration and its application to the game of Tetris. Journal of Machine Learning Research , 16 1629-1676.

- Schmidt-Hieber, J. (2020+). Nonparametric regression using deep neural networks with ReLU activation function. Annals of Statistics To appear.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning .
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .
- Schulte, P. J., Tsiatis, A. A., Laber, E. B. and Davidian, M. (2014). Q-and A-learning methods for estimating optimal dynamic treatment regimes. Statistical Science , 29 640.
- Shapley, L. S. (1953). Stochastic games. Proceedings of the national academy of sciences , 39 10951100.
- Shi, C., Fan, A., Song, R. and Lu, W. (2018). High-dimensional A-learning for optimal dynamic treatment regimes. Annals of Statistics , 46 925.
- Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M. et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature , 529 484-489.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A. et al. (2017). Mastering the game of Go without human knowledge. Nature , 550 354.
- Song, R., Wang, W., Zeng, D. and Kosorok, M. R. (2015). Penalized Q-learning for dynamic treatment regimens. Statistica Sinica , 25 901.
- Srinivasan, S., Lanctot, M., Zambaldi, V., P´ erolat, J., Tuyls, K., Munos, R. and Bowling, M. (2018). Actor-critic policy optimization in partially observable multiagent environments. In Advances in Neural Information Processing Systems .
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research , 15 1929-1958.
- Stone, C. J. (1982). Optimal global rates of convergence for nonparametric regression. Annals of Statistics , 10 1040-1053.
- Sutton, R. S. and Barto, A. G. (2011). Reinforcement learning: An introduction . MIT Press.
- Sutton, R. S., McAllester, D. A., Singh, S. P. and Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems .

- Suzuki, T. (2019). Adaptivity of deep relu network for learning in Besov and mixed smooth Besov spaces: Optimal rate and curse of dimensionality. In International Conference on Learning Representations .
- Tagorti, M. and Scherrer, B. (2015). On the rate of convergence and error bounds for LSTD ( λ ). In International Conference on Machine Learning .
- Tosatto, S., Pirotta, M., D'Eramo, C. and Restelli, M. (2017). Boosted fitted Q-iteration. In International Conference on Machine Learning .
- Tsiatis, A. A. (2019). Dynamic Treatment Regimes: Statistical Methods for Precision Medicine . CRC Press.
- Tsybakov, A. B. (2008). Introduction to nonparametric estimation . Springer.
- van Hasselt, H., Guez, A. and Silver, D. (2016). Deep reinforcement learning with double Qlearning. In AAAI Conference on Artificial Intelligence .
- Vershynin, R. (2010). Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 .
- Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P. et al. (2019). Grandmaster level in StarCraft II using multiagent reinforcement learning. Nature , 575 350-354.
- Von Neumann, J. and Morgenstern, O. (1947). Theory of games and economic behavior . Princeton University Press.
- Wang, L., Cai, Q., Yang, Z. and Wang, Z. (2019). Neural policy gradient methods: Global optimality and rates of convergence. In International Conference on Learning Representations .
- Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M. and Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In International Conference on Machine Learning .
- Watkins, C. J. and Dayan, P. (1992). Q-learning. Machine learning , 8 279-292.
- Wei, C.-Y., Hong, Y.-T. and Lu, C.-J. (2017). Online reinforcement learning in stochastic games. In Advances in Neural Information Processing Systems .
- Weinan, E., Ma, C. and Wu, L. (2019). A comparative analysis of optimization and generalization properties of two-layer neural network and random feature models under gradient descent dynamics. Science China Mathematics 1-24.
- Xu, P. and Gu, Q. (2019). A finite-time analysis of Q-learning with neural network function approximation. arXiv preprint arXiv:1912.04511 .

- Yang, G. (2019). Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation. arXiv preprint arXiv:1902.04760 .
- Yang, G. and Salman, H. (2019). A fine-grained spectral perspective on neural networks. arXiv preprint arXiv:1907.10599 .
- Yehudai, G. and Shamir, O. (2019). On the power and limitations of random features for understanding neural networks. In Advances in Neural Information Processing Systems .
- Yu, C., Liu, J. and Nemati, S. (2019). Reinforcement learning in healthcare: A survey. arXiv preprint arXiv:1908.08796 .
- Zhang, B., Tsiatis, A. A., Laber, E. B. and Davidian, M. (2012). A robust method for estimating optimal treatment regimes. Biometrics , 68 1010-1018.
- Zhang, K., Yang, Z., Liu, H., Zhang, T. and Ba¸ sar, T. (2018). Finite-sample analyses for fully decentralized multi-agent reinforcement learning. arXiv preprint arXiv:1812.02783 .
- Zhang, S. and Sutton, R. S. (2017). A deeper look at experience replay. arXiv preprint arXiv:1712.01275 .
- Zhao, Y., Kosorok, M. R. and Zeng, D. (2009). Reinforcement learning design for cancer clinical trials. Statistics in Medicine , 28 3294-3315.
- Zhao, Y., Zeng, D., Rush, A. J. and Kosorok, M. R. (2012). Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association , 107 1106-1118.
- Zhao, Y., Zeng, D., Socinski, M. A. and Kosorok, M. R. (2011). Reinforcement learning strategies for clinical trials in nonsmall cell lung cancer. Biometrics , 67 1422-1433.
- Zhao, Y.-Q., Zeng, D., Laber, E. B. and Kosorok, M. R. (2015). New statistical learning methods for estimating optimal dynamic treatment regimes. Journal of the American Statistical Association , 110 583-598.
- Zhou, X., Mayer-Hamblett, N., Khan, U. and Kosorok, M. R. (2017). Residual weighted learning for estimating individualized treatment rules. Journal of the American Statistical Association , 112 169-187.
- Zhu, W., Zeng, D. and Song, R. (2019). Proper inference for value function in high-dimensional Q-learning for dynamic treatment regimes. Journal of the American Statistical Association , 114 1404-1417.
- Zou, D., Cao, Y., Zhou, D. and Gu, Q. (2018). Stochastic gradient descent optimizes overparameterized deep ReLU networks. arXiv preprint arXiv:1811.08888 .