## Nearly Minimax Optimal Reinforcement Learning for Discounted MDPs

## Jiafan He

Department of Computer Science University of California, Los Angeles CA 90095, USA jiafanhe19@ucla.edu

## Dongruo Zhou

Department of Computer Science University of California, Los Angeles CA 90095, USA

drzhou@cs.ucla.edu

## Quanquan Gu

Department of Computer Science University of California, Los Angeles CA 90095, USA qgu@cs.ucla.edu

## Abstract

We study the reinforcement learning problem for discounted Markov Decision Processes (MDPs) under the tabular setting. We propose a model-based algorithm named UCBVIγ , which is based on the optimism in the face of uncertainty principle and the Bernstein-type bonus. We show that UCBVIγ achieves an ˜ O ( √ SAT/ (1 -γ ) 1 . 5 ) regret, where S is the number of states, A is the number of actions, γ is the discount factor and T is the number of steps. In addition, we construct a class of hard MDPs and show that for any algorithm, the expected regret is at least ˜ Ω ( √ SAT/ (1 -γ ) 1 . 5 ) . Our upper bound matches the minimax lower bound up to logarithmic factors, which suggests that UCBVIγ is nearly minimax optimal for discounted MDPs.

## 1 Introduction

The goal of reinforcement learning (RL) is designing algorithms to learn the optimal policy through interactions with the unknown dynamic environment. Markov decision process (MDPs) plays a central role in reinforcement learning due to their ability to describe the time-independent state transition property. More specifically, the discounted MDP is one of the standard MDPs in reinforcement learning to describe sequential tasks without interruption or restart. For discounted MDPs, with a generative model [12], several algorithms with near-optimal sample complexity have been proposed. More specifically, Azar et al. [3] proposed an Empirical QVI algorithm which achieves the optimal sample complexity to find the optimal value function. Sidford et al. [22] proposed a sublinear randomized value iteration algorithm that achieves a near-optimal sample complexity to find the optimal policy, and Sidford et al. [23] further improved it to reach the optimal sample complexity. Since generative model is a powerful oracle that allows the algorithm to query the reward function and the next state for any state-action pair ( s, a ) , it is natural to ask whether there exist online RL algorithms (without generative model) that achieve optimality.

To measure an online RL algorithm, a widely used notion is regret , which is defined as the summation of sub-optimality gaps over time steps. The regret is firstly introduced for episodic and infinite-horizon average-reward MDPs and later extended to discounted MDPs by [15, 30, 35, 35]. Liu and Su [15] proposed a double Q-learning algorithm with the UCB exploration (Double Q-

learning), which enjoys ˜ O ( √ SAT/ (1 -γ ) 2 . 5 ) regret, where S is the number of states, A is the number of actions, γ is the discount factor and T is the number of steps. While Double Q-learning enjoys a standard √ T -regret, it still does not match the lower bound proved in [15] in terms of the dependence on S, A and 1 / (1 -γ ) . Recently, Zhou et al. [34] proposed a UCLK + algorithm for discounted MDPs under the linear mixture MDP assumption and achieved ˜ O ( d √ T/ (1 -γ ) 1 . 5 ) regret, where d is the dimension of the feature mapping. However, directly applying their algorithm to our setting would yield an ˜ O ( S 2 A √ T/ (1 -γ ) 1 . 5 ) regret 1 , which is even worse that of double Q-learning [15] in terms of the dependence on S, A .

In this paper, we aim to close this gap by designing a practical algorithm with a nearly optimal regret. In particular, we propose a model-based algorithm named UCBVIγ for discounted MDPs without using the generative model. At the core of our algorithm is to use a 'refined' Bernstein-type bonus and the law of total variance [3, 4], which together can provide tighter upper confidence bound (UCB). Our contributions are summarized as follows:

- We propose a model-based algorithm UCBVIγ to learn the optimal value function under the discounted MDP setting. We show that the regret of UCBVIγ in first T steps is upper bounded by ˜ O ( √ SAT/ (1 -γ ) 1 . 5 ) . Our regret bound strictly improves the best existing regret O ( √ SAT/ (1 -γ ) 2 . 5 ) 2 in [15] by a factor of (1 -γ ) -1 .
- The nearly matching upper and the lower bounds together suggest that the proposed UCBVIγ algorithm is minimax-optimal up to logarithmic factors.
- ˜ · We also prove a lower bound of the regret by constructing a class of hard-to-learn discounted MDPs, which can be regarded as a chain of the hard MDPs considered in [15]. We show that for any algorithm, its regret in the first T steps can not be lower than ˜ Ω( √ SAT/ (1 -γ ) 1 . 5 ) on the constructed MDP. This lower bound also strictly improves the lower bound Ω( √ SAT/ (1 -γ ) + √ AT/ (1 -γ ) 1 . 5 ) proved by [15].

We compare the regret of UCBVIγ with previous online algorithms for learning discounted MDPs in Table 1.

Notation For any positive integer n , we denote by [ n ] the set { 1 , . . . , n } . For any two numbers a and b , we denote by a ∨ b as the shorthand for max( a, b ) . For two sequences { a n } and { b n } , we write a n = O ( b n ) if there exists an absolute constant C such that a n ≤ Cb n , and we write a n = Ω( b n ) if there exists an absolute constant C such that a n ≥ Cb n . We use ˜ O ( · ) and ˜ Ω( · ) to further hide the logarithmic factors.

## 2 Related Work

Model-free Algorithms for Discounted MDPs. A large amount of reinforcement learning algorithms like Q-learning can be regarded as model-free algorithms. These algorithms directly learn the action-value function by updating the values of each state-action pair. Kearns and Singh [13] firstly proposed a phased Q-Learning which learns an /epsilon1 -optimal policy with ˜ O ( SA/ ((1 -γ ) 7 /epsilon1 2 )) sample complexity for /epsilon1 ≤ 1 / (1 -γ ) . Later on, Strehl et al. [25] proposed a delay-Q-learning algorithm, which achieves ˜ O ( SA/ ((1 -γ ) 8 /epsilon1 4 )) sample complexity of exploration. Wang [29] proposed a randomized primal-dual method algorithm, which improves the sample complexity to ˜ O ( SA/ ((1 -γ ) 4 /epsilon1 2 )) for /epsilon1 ≤ 1 / (1 -γ ) under the ergodicity assumption. Later, Sidford et al. [23] proposed a sublinear randomized value iteration algorithm and achieved ˜ O ( SA/ ((1 -γ ) 4 /epsilon1 2 )) sample complexity for /epsilon1 ≤ 1 . Sidford et al. [22] further improved the empirical QVI algorithm and proposed a variance-reduced QVI algorithm, which improves the sample complexity to ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) for /epsilon1 ≤ 1 . Wainwright [28] proposed a variance-reduced Q-learning algorithm, which is an extension of the Q-learning algorithm and achieves O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) sam-

2 The regret definition in [15] differs from our definition by a factor of (1 -γ ) -1 . Here we translate their regret from their definition to our definition for a fair comparison. A detailed comparison can be found in Appendix.

˜ 1 Linear mixture MDP assumes that there exists a feature mapping φ ( s ′ | s, a ) ∈ R d and a vector θ ∈ R d such that P ( s ′ | s, a ) = 〈 φ ( s ′ | s, a ) , θ 〉 . It can be verified that any MDP is automatically a linear mixture MDP with a S A -dimensional feature mapping [2, 35].

Table 1: Comparison of RL algorithms for discounted MDPs in terms of sample complexity and regret. Note that the regret bounds for all the compared algorithms except Double Q-learning [15] are derived from their sample complexity results. See Appendix A.1 for more details.

|            | Algorithm                                                                                                        | Sample complexity                                                                                                                                 | Regret                                                                                                                                                                                                                |
|------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model-free | Delay-Q-learning [25] Q-learning with UCB [9] UCB-multistage [33] UCB-multistage-adv [33] Double Q-learning [15] | ˜ O ( SA (1 - γ ) 8 /epsilon1 4 ) ˜ O ( SA (1 - γ ) 7 /epsilon1 2 ) ˜ O ( SA (1 - γ ) 5 . 5 /epsilon1 2 ) ˜ O ( SA (1 - γ ) 3 /epsilon1 2 ) 3 N/A | ˜ O ( S 1 / 5 A 1 / 5 T 4 / 5 (1 - γ ) 9 / 5 ) ˜ O ( S 1 / 3 A 1 / 3 T 2 / 3 (1 - γ ) 8 / 3 ) ˜ O ( S 1 / 3 A 1 / 3 T 2 / 3 (1 - γ ) 13 / 6 ) ˜ O ( S 1 / 3 A 1 / 3 T 2 / 3 (1 - γ ) 4 / 3 ) O ( √ SAT (1 γ ) 2 . 5 ) |
| Model-free |                                                                                                                  |                                                                                                                                                   | ˜ -                                                                                                                                                                                                                   |
| Model-free | R-max [5]                                                                                                        | O ( S 2 A (1 - γ ) 6 /epsilon1 3 )                                                                                                                | O ( S 1 / 2 A 1 / 4 T 3 / 4 (1 - γ ) 7 / 4 )                                                                                                                                                                          |
| Model-free | MoRmax [27]                                                                                                      | ˜ O ( SA (1 - γ ) 6 /epsilon1 2 )                                                                                                                 | ˜ O ( S 1 / 3 A 1 / 3 T 2 / 3 (1 - γ ) 7 / 3 )                                                                                                                                                                        |
| Model-free | UCRL [14]                                                                                                        | ˜ ˜ O ( S 2 A (1 - γ ) 3 /epsilon1 2 )                                                                                                            | ˜ ˜ O ( S 2 / 3 A 1 / 3 T 2 / 3 (1 - γ ) 4 / 3 )                                                                                                                                                                      |
|            | UCBVI- γ ( Our work )                                                                                            | N/A                                                                                                                                               | ˜ O ( √ SAT (1 - γ ) 1 . 5 )                                                                                                                                                                                          |
|            |                                                                                                                  | ( )                                                                                                                                               | ( )                                                                                                                                                                                                                   |

2. It holds when /epsilon1 ≤ 1 / poly ( S, A, 1 / (1 -γ )) .

ple complexity. In addition, Dong et al. [9] proposed an infinite Q-learning with UCB and improved the sample complexity of exploration to ˜ O ( SA/ ((1 -γ ) 7 /epsilon1 2 )) . Zhang et al. [33] proposed a UCB-multistage algorithm which attains the ˜ O ( SA/ ((1 -γ ) 5 . 5 /epsilon1 2 )) sample complexity of exploration, and proposed a UCB-multistage-adv algorithm which attains a better sample complexity ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) in the high accuracy regime. Recently, Liu and Su [15] focused on regret minimization for the infinite-horizon discounted MDP and showed the connection between regret and sample complexity of exploration. Liu and Su [15] proposed a Double Q-Learning algorithm, which achieves ˜ O ( √ SAT/ (1 -γ ) 2 . 5 ) regret within T steps. Furthermore, Liu and Su [15] constructed a series of hard MDPs and showed that the expected regret for any algorithm is lower bounder by ˜ Ω ( √ SAT/ (1 -γ ) + √ AT/ (1 -γ ) 1 . 5 ) . There still exists a 1 / (1 -γ ) -gap between the upper and lower regret bounds. In contrast to the aforementioned model-free algorithms, our proposed algorithm is model-based.

Model-based Algorithms for Discounted MDP. Our UCBVIγ falls into the category of modelbased reinforcement learning algorithms. Model-based algorithms maintain a model of the environment and update it based on the observed data. They will form the policy based on the learnt model. More specifically, to learn the /epsilon1 -optimal value function, Azar et al. [3] proposed an empirical QVI algorithm which achieves ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) sample complexity. Azar et al. [3] proposed an empirical QVI algorithm which improves the sample complexity to ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) for /epsilon1 ≤ 1 / √ (1 -γ ) S . Szita and Szepesvári [27] proposed an MoRmax algorithm, which achieves ˜ O ( SA/ ((1 -γ ) 6 /epsilon1 2 )) sample complexity. Later, Lattimore and Hutter [14] proposed a UCRL algorithm, which achieves ˜ O ( S 2 A/ ((1 -γ ) 3 /epsilon1 2 )) sample complexity in general and ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) sample complexity with a strong assumption on the state transition. Recently, Agarwal et al. [1] proposed a refined analysis for the empirical QVI algorithm which achieves ˜ O ( SA/ ((1 -γ ) 3 /epsilon1 2 )) sample complexity when /epsilon1 ≤ 1 / √ 1 -γ .

Upper and Lower Bounds for Episodic MDPs. There is a line of work which aims at proving sample complexity or regret for episodic MDPs (MDPs which consist of restarting episodes) [7, 18, 4, 19, 11, 8, 24, 21, 31, 32, 17, 20]. Compared with the episodic MDP, discounted MDPs involve only one infinite-horizon sample trajectory, suggesting that any two states or actions on the trajectory are dependent. Such a dependence makes the learning of discounted MDPs more challenging.

## 3 Preliminaries

We consider infinite-horizon discounted Markov Decision Processes (MDP) which are defined by a tuple ( S , A , γ, r, P ) . Here S is the state space with |S| = S , A is the action space with |A| = A , γ ∈ (0 , 1) is the discount factor, r : S×A → [0 , 1] is the reward function, P ( s ′ | s, a ) is the transition probability function, which denotes the probability that state s transfers to state s ′ with action a . For simplicity, we assume the reward function is deterministic and known . A non-stationary policies π is a collection of function { π t } ∞ t =1 , where each function π t : {S × A} t -1 × S → A maps history { s 1 , a 1 , ..., s t -1 , a t -1 , s t = s } to an action. For any non-stationary policy π , we denote π t ( s ) = π t ( s ; s 1 , a 1 , ..., s t -1 , a t -1 ) for simplicity. We define the action-value function and value function at step t as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a t + i = π t + i ( s t + i ) , and s t + i +1 ∼ P ( · | s t + i , π t + i ( s t + i ) ) . In addition, we denote the optimal action-value function and the optimal value function as Q ∗ ( s, a ) = sup π Q π 1 ( s, a ) and V ∗ ( s ) = sup π V π 1 ( s ) respectively. Note that the optimal action-value function and the optimal value function are independent of the step t . For simplicity, for any function V : S → R , we denote [ P V ]( s, a ) = E s ′ ∼ P ( ·| s,a ) V ( s ′ ) . According to the definition of the value function, we have the following non-stationary Bellman equation and Bellman optimality equation for non-stationary policy π and optimal policy π ∗ :

<!-- formula-not-decoded -->

## 4 Main Results

## 4.1 Algorithm

In this subsection, we propose the Upper Confidence Bound Value Iterationγ (UCBVIγ ) algorithm, which is illustrated in Algorithm 1. The algorithm framework of UCBVIγ follows the UCBVI algorithm proposed in Azar et al. [4], which can be regarded as the counterpart of UCBVIγ in the episodic MDP setting.

UCBVIγ is a model-based algorithm that maintains an empirical measure P t at each step t . At the beginning of the t -th iteration, UCBVIγ takes action a t based on the greedy policy induced by Q t ( s t , a ) and transits to the next state s t +1 . After receiving the next state s t +1 , UCBVIγ computes the empirical transition probability function P t ( s ′ | s, a ) in (4.1). Based on empirical transition probability function P t ( s ′ | s, a ) , UCBVIγ updates Q t +1 ( s, a ) by performing one-step value iteration on Q t ( s, a ) with an additional upper confidence bound (UCB) term UCB t ( s, a ) defined in (4.3). Here the UCB bonus term is used to measure the uncertainty of the expectation of the value function V t ( s ) . Unlike previous work, which adapts a Hoeffding-type bonus [15], our UCBVIγ uses a Bernstein-type bonus which brings a tighter upper bound by accessing the variance of V t ( s ) , denoted by Var s ′ ∼ P ( ·| ,s,a ) V t ( s ′ ) . However, since the probability transition P ( ·| s, a ) is unknown, it is impossible to calculate the exact variance of V t . Instead, UCBVIγ estimates the variance by considering the variance of V t over the empirical probability transition function P t ( ·| s, a ) defined in (4.1). Therefore, the final UCB bonus term in (4.3) can be regarded as a standard Bernstein-type bonus on the empirical measure P t ( ·| s, a ) with an additional error term.

Compared with UCBVI algorithm in Azar et al. [4], the action-value function Q t ( s, a ) in UCBVIγ is updated in a forward way from step 1 to step T with the initial value Q 1 ( s, a ) = 1 / (1 -γ ) for all s ∈ S , a ∈ A , while UCBVI updates its action-value function in a backward way from Q t,H to Q t, 1 with initial value Q t,H ( s, a ) = 0 . Compared with UCRL in Lattimore and Hutter [14], UCBVIγ

## Algorithm 1 Upper Confidence Value-iteration UCBVIγ

- 1: Receive state s 1 and set initial value function Q 1 ( s, a ) ← 1 / (1 -γ ) , N 0 ( s, a ) = N 0 ( s, a, s ′ ) = N 0 ( s ) ← 0 for all s ∈ S , a ∈ A , s ′ ∈ S
- 3: Let π t ( · ) ← argmax a ∈A Q t ( · , a ) , take action a t ← π t ( s t ) and receive next state s t +1 ∼ P ( ·| s t , a t )
- 2: for step t = 1 , . . . do
- 4: Set N t ( s ) ← N t -1 ( s ) , N t ( s, a ) ← N t -1 ( s, a ) and N t ( s, a, s ′ ) ← N t -1 ( s, a, s ′ ) for all s ∈ S , a ∈ A , s ′ ∈ S
- 6: For all s ∈ S , a ∈ A , set
- 5: Update N t ( s t ) ← N t ( s t ) + 1 , N t ( s t , a t ) ← N t ( s t , a t ) + 1 and N t ( s t , a t , s t +1 ) ← N t ( s t , a t , s t +1 ) + 1

<!-- formula-not-decoded -->

- 7: Update new value function Q t +1 ( s, a ) and V t +1 ( s ) by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

does not need to call an additional extended value iteration sub-procedure [10, 26], which is not easy to implement even with infinite computation [14].

Computational complexity In each step t , Algorithm 1 needs to first compute the empirical transition P t and update the value function V t +1 by one-step value iteration, which will cost O ( S 2 A ) time complexity for each update. However, the number of updates can be reduced by using the 'batch' update scheme adapted in [10, 7] and in this case Algorithm 1 only needs to update the value function V t +1 when the number of visits N t ( s, a ) doubles. With this update scheme, the number of updates is upper bounded by O ( SA log T ) and the total cost for updating the value function is O ( S 3 A 2 log T ) . In addition, the Algorithm 1 still needs to choose the action with respect to the value function V t and it costs O ( AT ) time complexity. Thus, the total computation complexity of the 'batch' version of Algorithm 1 is O ( AT + S 3 A 2 log T ) .

## 4.2 Regret Analysis

In this subsection, we provide the regret bound of UCBVIγ . We first give the formal definition of the regret for the discounted MDP setting.

Definition 4.1. For a given non-stationary policy π , we define the regret Regret ( T ) as follow:

<!-- formula-not-decoded -->

The same regret has been used in prior work [30, 35, 34] on discounted MDPs. It is related to the 'sample complexity of exploration' [12, 14, 9]. For more details about the connection between the regret and the sample complexity, please refer to Appendix A.

Remark 4.2. Without the use of generative model [12], an agent may enter bad states at the first few steps in discounted MDPs and there is no 'restarting' mechanism as in episodic MDPs that can

prevent the agent from being stuck in those bad states. Due to this limitation, both the regret and the sample complexity of exploration guarantees are not sufficient to ensure a good policy being learned. We think this is the fundamental limitation in the online learning of discounted MDPs.

With Definition 4.1, we introduce our main theorem, which gives an upper bound on the regret for UCBVIγ .

Theorem 4.3. Let U = log(40 SAT 3 log 2 T/ ( δ (1 -γ ) 2 )) . If we set β = S 2 A 2 U 5 in UCBVIγ , then with probability at least 1 -δ , the regret of UCBVIγ in Algorithm 1 is bounded by

<!-- formula-not-decoded -->

Remark 4.4. Notice that when T = ˜ Ω( S 3 A 2 / (1 -γ ) 4 ) and SA = Ω(1 / (1 -γ )) , the regret is bounded by ˜ O ( √ SAT/ (1 -γ ) 1 . 5 ) . In addition, since Regret ( T ) ≤ T/ (1 -γ ) holds for any T , we have E [ Regret ( T )] = ˜ O ( √ SAT/ (1 -γ ) 1 . 5 + Tδ/ 1 -γ ) . When choosing δ = 1 /T , we have E [ Regret ( T )] = ˜ O ( √ SAT/ (1 -γ ) 1 . 5 ) . We also provide a regret lower bound, which suggests that our UCBVIγ is nearly minimax optimal. Theorem 4.5. Suppose γ ≥ 2 / 3 , A ≥ 30 and T ≥ 100 SAL/ (1 -γ ) 4 , then for any algorithm, there exists an MDP such that where L = log (300 S 4 T 2 / (1 -γ )) log(10 ST ) .

<!-- formula-not-decoded -->

Remark 4.6. When T is large enough and A = ˜ Ω(1) , Theorem 4.5 suggests that the lower bound of regret is ˜ Ω( √ SAT/ (1 -γ ) 1 . 5 ) . It can be seen that the regret of UCBVIγ in Theorem 4.3 matches this lower bound up to logarithmic factors. Therefore, UCBVIγ is nearly minimax optimal.

## 5 Proof of the Main Results

In this section, we provide the proofs of Theorems 4.3 and 4.5. The missing proofs are deferred to the appendix.

## 5.1 Proof of Theorem 4.3

In this subsection, we prove Theorem 4.3. For simplicity, let δ ′ = (1 -γ ) 2 δ/ (80 T log 2 T ) , then U = log( SAT 2 /δ ′ ) . We first present the following key lemma, which shows that the optimal value functions V ∗ and Q ∗ can be upper bounded by the estimated functions V t and Q t with high probability:

Lemma 5.1. With probability at least 1 -64 Tδ log 2 T/ (1 -γ ) 2 , for all t ∈ [ T ] , s ∈ S , a ∈ A , we have Q t ( s, a ) ≥ Q ∗ ( s, a ) , V t ( s ) ≥ V ∗ ( s ) .

Equipped with Lemma 5.1, we can decompose the regret of UCBVIγ as follows:

<!-- formula-not-decoded -->

where the inequality holds due to Lemma 5.1. Therefore, it suffices to bound Regret ′ ( T ) . We have

<!-- formula-not-decoded -->

where the inequality holds due to the update rule (4.2) and the Bellman equation Q π t ( s t , a t ) = r ( s t , a t ) + γ [ P V π t +1 ]( s t , a t ) . We further have

<!-- formula-not-decoded -->

In the remaining of the proof, it suffices to bound terms I 1 to I 5 separately.

In the above decomposition, term I 1 controls the estimation error between the value functions V t -1 and V π t +1 , terms I 2 and I 3 measure the estimation error between the transition probability function P and the estimated transition probability function P t -1 , term I 4 comes from the exploration bonus in Algorithm 1, and term I 5 accounts for the randomness in the stochastic transition process, which can be controlled by the third term O ( √ TU/ (1 -γ ) 2 ) in Theorem 4.3.

First, I 1 can be regarded as the difference between the estimated V t -1 and the value function V π t +1 of policy π , and it can be bounded by the following lemma.

<!-- formula-not-decoded -->

Next, I 2 can be regarded as the 'correction" term between the estimated V t -1 and the optimal value function V ∗ . It can be bounded by the following lemma.

Lemma 5.3. With probability at least 1 -64 Tδ log 2 T/ (1 -γ ) 2 -3 δ , we have

In addition, I 3 can be regarded as the error between the empirical probability distribution P t -1 and the true transition probability P . Note that V ∗ is a fixed value function that does not have any randomness. Therefore, I 3 can be bounded through the standard concentration inequalities, and its upper bound is presented in the following lemma.

<!-- formula-not-decoded -->

Lemma 5.4. With probability at least 1 -2 δ -δ/ (1 -γ ) , we have

<!-- formula-not-decoded -->

Furthermore, I 4 can be regarded as the summation of the UCB terms, which is also the dominating term of the total regret. It can be bounded by the following lemma.

Lemma 5.5. With probability at least 1 -4 δ -δ/ (1 -γ ) , we have

<!-- formula-not-decoded -->

Finally, I 5 is the summation of a martingale difference sequence. By Azuma-Hoeffding inequality, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Substituting the upper bounds of terms I 1 to I 5 from Lemma 5.2 to Lemma 5.5, as well as (5.2), into (5.1), and taking a union bound to let all the events introduced in Lemma 5.2 to Lemma 5.5 and (5.2) hold, we have with probability at least 1 -20 TU 2 δ/ (1 -γ ) 2 , the following inequality holds:

<!-- formula-not-decoded -->

Using the fact that x ≤ a + b √ x ⇒ x ≤ 1 . 1 a +4 b 2 , (5.3) can be further bounded as follows

This completes our proof.

<!-- formula-not-decoded -->

## 5.2 Proof of Theorem 4.5

<!-- image -->

In this subsection, we provide the proof of Theorem 4.5. The proof of the lower bound is based on constructing a class of hard MDPs. Specifically, the state space S consists of 2 S states { s i, 0 , s i, 1 } i ∈ [ S ] and the action space A contains A actions. The reward function r satisfies that r ( s i, 0 , a ) = 0 and r ( s i, 1 , a ) = 1 for any a ∈ A , i ∈ [ S ] . The probability transition function P is defined as follows.

Figure 1: A class of hard-to-learn MDPs considered in Theorem 4.5. The MDP can be regarded as a combination of S two-state MDPs, each of which is an MDP illustrated on the top-left corner. In addition, the i -th two-state MDP has the a ∗ i -th action as its optimal action. The blue arrows represent the optimal actions in different states. /epsilon1 = √ A (1 -γ ) /K/ 24 .

<!-- formula-not-decoded -->

where we assume s S +1 , 0 = s 1 , 0 for simplicity and a ∗ i is the optimal action for state s i, 0 . The MDP is illustrated in Figure 1, which can be regarded as S copies of the 'single" two-state MDP arranged in a circle. The two-state MDP is the same as that proposed in [15]. Each of the two-state MDP has two states and one 'optimal" action a ∗ i satisfied P ( s i, 1 | s i, 0 , a ∗ i ) = 1 -γ + /epsilon1 . Compared with the MDP instance in [10], both instances use S copies of a single MDP. However, unlike the MDP in [10] which only has one 'optimal" action among all SA actions, our MDP which has in total S 'optimal" actions, which makes it harder to analyze.

Now we begin to prove our lower bound. Let E a ∗ [ · ] denote the expectation conditioned on one fixed selection of a ∗ = ( a ∗ 1 , . . . , a ∗ S ) . We introduce a shorthand notation E ∗ to denote E ∗ [ · ] =

1 /A S · ∑ a ∗ ∈A S E a ∗ [ · ] . Here E ∗ is the average value of expectation over the randomness from MDP defined by different optimal actions. From now on, we aim to lower bound E ∗ [ Regret ( T )] , since once E ∗ [ Regret ( T )] is lower bounded, E [ Regret ( T )] can be lower bounded by selecting a ∗ 1 , . . . , a ∗ S which maximizes E [ Regret ( T )] . We set T = 10 SK in the following proof. Based on the definition of E ∗ , we have the following lemma.

Lemma 5.6. The expectated regret E ∗ [ Regret ( T )] can be lower bounded as follows:

<!-- formula-not-decoded -->

By Lemma 5.6, it suffices to lower bound ∑ T t =1 [ V ∗ ( s t ) -r ( s t , a t ) / (1 -γ )] , which is Regret Liu ( T ) defined in [15]. When an agent visits the state set { s j, 0 , s j, 1 } for the i -th time, we denote the state in { s j, 0 , s j, 1 } it visited as X j,i , and the following action selected by the agent as A j,i . Let T j be the number of steps for the agent staying in { s j, 0 , s j, 1 } in the total T steps. Then the regret can be further decomposed as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that I 1 essentially represents the regret over S two-state MDPs in their first K steps, and it can be lower bounded through the following lemma.

Lemma 5.7. If K ≥ 10 SA/ (1 -γ ) 4 , then for each j ∈ [ S ] , we have

<!-- formula-not-decoded -->

This lemma shows that the expected regret of first K steps on states s j, 0 and s j, 1 is at least ˜ Ω ( √ AK/ (1 -γ ) 0 . 5 -1 / (1 -γ ) ) . Therefore by Lemma 5.7, we have

To bound I 2 , we need the following lemma.

<!-- formula-not-decoded -->

Lemma 5.8. With probability at least 1 -2 STδ log T/ (1 -γ ) , for each j ∈ [ S ] and K +1 ≤ t ≤ T , we have

<!-- formula-not-decoded -->

Lemma 5.8 gives a crude lower bound of I 2 . Taking expectation over Lemma 5.8 and taking summation over all states, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 5.9. When K ≥ 10 A log(1 /δ ) / (1 -γ ) 4 , with probability at least 1 -2 Sδ , for all i ∈ [ S ] , we have T i &gt; K .

where the first inequality holds due to Lemma 5.8, the second inequality holds since 1 -2 STδ log T/ (1 -γ ) ≤ 1 and E [ -X | Y ] P ( Y ) ≥ E [ -X ] when X ≥ 0 , the third inequality holds due to Jensen's inequality and the fact that √ x is a concave function, and the last inequality holds due to Jensen's inequality and the fact that ∑ S j =1 E ∗ [ T j ] = T . To bound I 3 , we need the following lemma, which suggests that when K is large enough, T i &gt; K happens with high probability:

Notice that the difference of transition probability between the optimal action and suboptimal actions is √ A (1 -γ ) / 24 K . In this case, when T is large enough, T i is close to T/S = 10 K . Thus I 3 can be lower bounded as follows:

where the first inequality holds due to 0 ≤ r ( X j,i , A j,i ) ≤ 1 and the second inequality holds due to Lemma 5.9. Finally, setting δ = 1 / ( 4 ST 2 (1 -γ ) 2 log T ) , we can verify that the requirements of K in Lemma 5.7 and Lemma 5.9 hold when T satisfies T ≥ 100 SAL/ (1 -γ ) 4 , and L = log (300 S 4 T 2 / ((1 -γ ) 2 δ )) log T . Therefore, substituting δ = 1 / ( 4 ST 2 (1 -γ ) 2 log T ) into (5.5) and (5.6), and combining (5.4), (5.5), (5.6) and Lemma 5.6, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which completes the proof of Theorem 4.5.

## 6 Conclusions and Future Work

We proposed UCBVIγ , an online RL algorithm for discounted tabular MDPs. We show that the regret of UCBVIγ can be upper bounded by ˜ O ( √ SAT/ (1 -γ ) 1 . 5 ) and we prove a matching lower bound on the expected regret ˜ Ω( √ SAT/ (1 -γ ) 1 . 5 ) . There is still a gap between the upper and lower bounds when T ≤ max { S 3 A 2 / (1 -γ ) 4 , SA/ (1 -γ ) 4 } , and we leave it as an open problem for future work.

## Acknowledgments and Disclosure of Funding

We thank Csaba Szepesvári for a valuable suggestion on improving the presentation of the proof. We thank the anonymous reviewers for their helpful comments. JH, DZ and QG are partially supported by the National Science Foundation CAREER Award 1906169, IIS-1904183, BIGDATA IIS1855099 and AWS Machine Learning Research Award. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

## References

- [1] AGARWAL, A., KAKADE, S. and YANG, L. F. (2019). Model-based reinforcement learning with a generative model is minimax optimal. arXiv preprint arXiv:1906.03804 .
- [2] AYOUB, A., JIA, Z., SZEPESVARI, C., WANG, M. and YANG, L. F. (2020). Model-based reinforcement learning with value-targeted regression. arXiv preprint arXiv:2006.01107 .

- [3] AZAR, M. G., MUNOS, R. and KAPPEN, H. J. (2013). Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning 91 325-349.
- [4] AZAR, M. G., OSBAND, I. and MUNOS, R. (2017). Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 . JMLR. org.
- [5] BRAFMAN, R. I. and TENNENHOLTZ, M. (2002). R-max-a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research 3 213-231.
- [6] CESA-BIANCHI, N. and LUGOSI, G. (2006). Prediction, learning, and games . Cambridge university press.
- [7] DANN, C. and BRUNSKILL, E. (2015). Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems .
- [8] DANN, C., LI, L., WEI, W. and BRUNSKILL, E. (2019). Policy certificates: Towards accountable reinforcement learning. In International Conference on Machine Learning . PMLR.
- [9] DONG, K., WANG, Y., CHEN, X. and WANG, L. (2019). Q-learning with ucb exploration is sample efficient for infinite-horizon mdp. arXiv preprint arXiv:1901.09311 .
- [10] JAKSCH, T., ORTNER, R. and AUER, P. (2010). Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research 11 1563-1600.
- [11] JIN, C., ALLEN-ZHU, Z., BUBECK, S. and JORDAN, M. I. (2018). Is q-learning provably efficient? In Advances in Neural Information Processing Systems .
- [12] KAKADE, S. M. ET AL. (2003). On the sample complexity of reinforcement learning . Ph.D. thesis, University of London London, England.
- [13] KEARNS, M. J. and SINGH, S. P. (1999). Finite-sample convergence rates for q-learning and indirect algorithms. In Advances in neural information processing systems .
- [14] LATTIMORE, T. and HUTTER, M. (2012). Pac bounds for discounted mdps. In International Conference on Algorithmic Learning Theory . Springer.
- [15] LIU, S. and SU, H. (2020). Regret bounds for discounted mdps.
- [16] MAURER, A. and PONTIL, M. (2009). Empirical bernstein bounds and sample variance penalization. arXiv preprint arXiv:0907.3740 .
- [17] NEU, G. and PIKE-BURKE, C. (2020). A unifying view of optimism in episodic reinforcement learning. arXiv preprint arXiv:2007.01891 .
- [18] OSBAND, I. and VAN ROY, B. (2016). On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 .
- [19] OSBAND, I. and VAN ROY, B. (2017). Why is posterior sampling better than optimism for reinforcement learning? In International Conference on Machine Learning .
- [20] PACCHIANO, A., BALL, P., PARKER-HOLDER, J., CHOROMANSKI, K. and ROBERTS, S. (2020). On optimism in model-based reinforcement learning. arXiv preprint arXiv:2006.11911 .
- [21] RUSSO, D. (2019). Worst-case regret bounds for exploration via randomized value functions. In Advances in Neural Information Processing Systems .
- [22] SIDFORD, A., WANG, M., WU, X., YANG, L. F. and YE, Y. (2018). Near-optimal time and sample complexities for for solving discounted markov decision process with a generative model. arXiv preprint arXiv:1806.01492 .
- [23] SIDFORD, A., WANG, M., WU, X. and YE, Y. (2018). Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms . SIAM.

- [24] SIMCHOWITZ, M. and JAMIESON, K. G. (2019). Non-asymptotic gap-dependent regret bounds for tabular mdps. In Advances in Neural Information Processing Systems .
- [25] STREHL, A. L., LI, L., WIEWIORA, E., LANGFORD, J. and LITTMAN, M. L. (2006). Pac model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning .
- [26] STREHL, A. L. and LITTMAN, M. L. (2008). An analysis of model-based interval estimation for markov decision processes. Journal of Computer and System Sciences 74 1309-1331.
- [27] SZITA, I. and SZEPESVÁRI, C. (2010). Model-based reinforcement learning with nearly tight exploration complexity bounds .
- [28] WAINWRIGHT, M. J. (2019). Variance-reduced q -learning is minimax optimal. arXiv preprint arXiv:1906.04697 .
- [29] WANG, M. (2017). Randomized linear programming solves the discounted markov decision problem in nearly-linear running time. arXiv preprint arXiv:1704.01869 .
- [30] YANG, K., YANG, L. and DU, S. (2021). Q-learning with logarithmic regret 1576-1584.
- [31] ZANETTE, A. and BRUNSKILL, E. (2019). Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. arXiv preprint arXiv:1901.00210 .
- [32] ZHANG, Z., ZHOU, Y. and JI, X. (2020). Almost optimal model-free reinforcement learning via reference-advantage decomposition. arXiv preprint arXiv:2004.10019 .
- [33] ZHANG, Z., ZHOU, Y. and JI, X. (2020). Model-free reinforcement learning: from clipped pseudo-regret to sample complexity. arXiv preprint arXiv:2006.03864 .
- [34] ZHOU, D., GU, Q. and SZEPESVARI, C. (2021). Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In COLT .
- [35] ZHOU, D., HE, J. and GU, Q. (2021). Provably efficient reinforcement learning for discounted mdps with feature mapping. In International Conference on Machine Learning . PMLR.

## Checklist

1. For all authors...
2. (a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]
3. (b) Did you describe the limitations of your work? [Yes] We discuss the limitations and potential future works in Section 6
4. (c) Did you discuss any potential negative societal impacts of your work? [N/A] Our work provides a theoretical analysis of discounted MDPs, and there is no potential negative social impact.
5. (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]
2. If you are including theoretical results...
7. (a) Did you state the full set of assumptions of all theoretical results? [Yes]
8. (b) Did you include complete proofs of all theoretical results? [Yes]
3. If you ran experiments...
10. (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [N/A]
11. (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [N/A]
12. (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [N/A]

- (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [N/A]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
- (a) If your work uses existing assets, did you cite the creators? [N/A]
- (b) Did you mention the license of the assets? [N/A]
- (c) Did you include any new assets either in the supplemental material or as a URL? [N/A]
- (d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]
- (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]
5. If you used crowdsourcing or conducted research with human subjects...
- (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]
- (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]
- (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

## A More Discussions on the Regret and Sample Complexity

## A.1 Converting Sample Complexity of Exploration to Regret

In this subsection, we shows the relationship between the sample complexity of exploration and the regret.

The definition of regret in Defintion 4.1 is related to the 'sample complexity of exploration' N ( /epsilon1, δ ) [12, 14, 9], which is the upper bound on the number of steps t such that V ∗ ( s t ) -V π t ( s t ) ≥ /epsilon1 with probability at least 1 -δ . Compared with the regret, sample complexity of exploration focuses on the sub-optimalities at all steps t , rather than the first T steps, and ignores the small sub-optimalities. Though both metrics have been used to describe the performance of an algorithm, these two metrics are not directly comparable. More specifically, algorithms with fewer but larger sub-optimalities will have a small sample complexity of exploration but a high regret. In contrast, algorithms with a lot of moderate sub-optimalities will have a high sample complexity of exploration but a low regret.

By the definition of the sample complexity exploration N ( /epsilon1, δ ) , with probability at least 1 -δ , the number of steps t where V ∗ ( s t ) -V π t ( s t ) ≥ /epsilon1 is upper bounded by N ( /epsilon1, δ ) . Thus, for the regret within T steps, we have following inequality:

<!-- formula-not-decoded -->

where the inequality holds due to the definition of N ( /epsilon1, δ ) . Furthermore,if an algorithm achieve sample complexity N ( /epsilon1, δ ) = O ( B/epsilon1 -α ) , then we can choose /epsilon1 = T -1 / ( α +1) (1 -γ ) 1 / ( α +1) B -1 / ( α +1) to minimize the (A.1). Thus, we have

<!-- formula-not-decoded -->

## A.2 Comparison with the Regret in [15]

Furthermore, the best result in sample complexity of exploration [33] achieves ˜ O ( SA/ ( (1 -γ ) 3 /epsilon1 2 ) ) sample complexity and this result implies ˜ O ( S 1 / 3 A 1 / 3 (1 -γ ) -4 / 3 T 2 / 3 ) regret, which is worse than our result by a T 1 / 6 factor.

Our definition is similar to that of Liu and Su [15]. Note that Liu and Su [15] define the regret as Regret Liu ( T ) = ∑ T t =1 ∆ t , where ∆ t = (1 -γ ) V ∗ ( s t ) -r ( s t , a t ) . Comparing the definition in Liu and Su [15] with our definition, we can show that (1 -γ ) Regret ( T ) ≈ Regret Liu ( T ) since where the first approximate equality holds due to Azuma-Hoeffding inequality and the second approximate equality holds due to 0 ≤ r ( s, a ) ≤ 1 . Therefore, our regret definition is equivalent to that in [15] up to a 1 -γ factor.

<!-- formula-not-decoded -->

## B Proof of Lemmas in Section 5.1

In this section, we prove Lemma 5.1 to Lemma 5.5. For simplicity, we introduce the following shorthand notations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We start with a list of technical lemmas that will be used to prove Lemma 5.1 to Lemma 5.5. We first provide the Azuma-Hoeffding and Bernstein inequalities.

Lemma B.1 (Azuma-Hoeffding inequality, Cesa-Bianchi and Lugosi 6) . Let { x i } n i =1 be a martingale difference sequence with respect to a filtration {G i } satisfying | x i | ≤ M for some constant M , x i is G i +1 -measurable, E [ x i |G i ] = 0 . Then for any 0 &lt; δ &lt; 1 , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.2 (Bernstein inequality, Cesa-Bianchi and Lugosi 6) . Let { x i } n i =1 be a martingale difference sequence with respect to a filtration {G i } satisfying | x i | ≤ M for some constant M , x i is G i +1 -measurable, E [ x i |G i ] = 0 . Suppose that for some constant v . Then for any δ &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The following first lemma provides basic inequalities for the summations of counted numbers N i ( s i , a i ) and N i ( s i ) .

Lemma B.3. For all t ∈ [ T ] and subset C ⊆ [ T ] , we have

<!-- formula-not-decoded -->

Next lemma upper bounds the difference between the empirical measure P t -1 and P , with respect to the true variance of the optimal value function V ∗ ( s, a ) .

<!-- formula-not-decoded -->

Lemma B.4. If 0 ≤ V ∗ ( s ) ≤ 1 / (1 -γ ) for all s ∈ S , then with probability at least 1 -δ , for all t ∈ [ T ] , s ∈ S , a ∈ A , we have

Similar to Lemma B.4, the following lemmas also upper bounds the difference between the empirical measure P t -1 and P , but with respect to the estimated variance.

<!-- formula-not-decoded -->

Lemma B.5 (Theorem 4 in Maurer and Pontil 16) . Let Z, Z 1 , .., Z n be i.i.d random variable with value in [0 , M ] and let δ &gt; 0 , then with probability at least 1 -δ , we have where V n Z is the estimated variance V n Z = ∑ 1 ≤ i&lt;j ≤ n ( Z i -Z j ) 2 /n ( n -1) .

<!-- formula-not-decoded -->

Lemma B.6. If 0 ≤ V ∗ ( s ) ≤ 1 / (1 -γ ) for all s ∈ S , then with probability at least 1 -δ , for all t ∈ [ T ] , s ∈ S , a ∈ A , we have

The next lemma shows that the total variance of the nonstationary policy π can be upper bounded by O ( T/ (1 -γ )) . It is worth noting that a trivial bound which bounds V π i ( s i , a i ) by 1 / (1 -γ ) 2 only gives an O ( T/ (1 -γ ) 2 ) bound.

<!-- formula-not-decoded -->

Lemma B.7. With probability at least 1 -δ/ (1 -γ ) , we have

<!-- formula-not-decoded -->

Based on previous concentration Lemma, we define the following high probability events and our proof of Lemma 5.2 to Lemma 5.5 relies on these high probability events. Let E denote the event when the conclusion of Lemma 5.1 holds. Then by Lemma 5.1, we have Pr( E ) ≥ 1 -64 Tδ log 2 T/ (1 -γ ) 2 . We also define the following event:

<!-- formula-not-decoded -->

where U = log(40 SAT 3 log 2 T/ ( δ (1 -γ ) 2 )) . For these high probability events, according to the Lemma B.1, we have Pr( E 4 ) ≥ 1 -δ, Pr( E 6 ) ≥ 1 -δ, Pr( E 7 ) ≥ 1 -δ, Pr( E 8 ) ≥ 1 -δ, Pr( E 9 ) ≥ 1 -δ. According to the Lemma B.2, we have Pr( E 3 ) ≥ 1 -δ . According to the Lemma B.4, we have Pr( E 1 ) ≥ 1 -δ . According to the Lemma B.6, we have Pr( E 2 ) ≥ 1 -δ . According to the Lemma B.7, we have Pr( E 5 ) ≥ 1 -δ/ (1 -γ ) .

The next lemma shows that the total difference between the optimal variance and the variance induced by π can be bounded in terms of Regret ′ ( T ) .

Lemma B.8. On the event E 7 , we have

<!-- formula-not-decoded -->

Similar to Lemma B.8, the next lemma shows that the total difference between the estimated variance and the variance induced by π can be upper-bounded in terms of Regret ′ ( T ) .

Lemma B.9. On the event E 6 ∩ E 8 , we have

<!-- formula-not-decoded -->

## B.1 Proof of Lemma 5.1

For simplicity, we denote U = log( SAT 2 /δ ) and H = /floorleft 2 log T/ (1 -γ ) /floorright +1 and for h ∈ [ H ] , we define

Then we have the following lemma.

<!-- formula-not-decoded -->

Lemma B.10. For each t ∈ [ T ] , with probability at least 1 -4 H 2 δ , for all s ∈ S , h ∈ [ H ] , we have

In addition, if N t ( s ) &gt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we start the proof of Lemma 5.1,

Proof of Lemma 5.1. We prove this lemma by induction. At the first step t = 1 , for all s ∈ S , we have V 1 ( s ) = 1 / (1 -γ ) ≥ V ∗ ( s ) . When Lemma 5.1 holds for the first t steps, we consider for each s ∈ S , a ∈ A , then by the update rule (4.2), we have

If Q t +1 ( s, a ) = Q t ( s, a ) , then by induction, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to (4.2) in Algorithm 1 and the second inequality holds due to 0 ≤ V ∗ ( s ) ≤ 1 / (1 -γ ) . Otherwise, if N t ( s, a ) = 0 , then we have

When N t ( s, a ) &gt; 0 , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to V t ( s ) ≥ V ∗ ( s ) , the second inequality holds due to Lemma B.6 and the third inequality holds due to the definition of UCB t in (4.3). For the term V ∗ t ( s, a ) , we have where the first inequality holds due to ( x + y ) 2 ≤ 2 x 2 +2 y 2 and the second inequality holds due to E [ ( X -E [ X ]) 2 ] ≤ E [ X 2 ] . Substituting (B.2) into (B.1), with probability at least 1 -4( t +1) H 2 δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to (B.1), the second inequality holds due to (B.2), the third inequality holds due to √ a + b ≤ √ a + √ b , the last inequality holds due to Lemma B.10 with probability at least 1 -4 H 2 δ and induction hypothesis with probability at least 1 -4 tH 2 δ . In addition, for all s ∈ S , we have

Thus, by induction, we complete the proof of Lemma 5.1.

## B.2 Proof of Lemma 5.2

Proof of Lemma 5.2. We have

<!-- formula-not-decoded -->

For the term I 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to V t -1 ( s ) ≥ V t +1 ( s ) by (4.2) in Algorithm 1, and the second inequality holds due to 0 ≤ V t ( s ) ≤ 1 / (1 -γ ) . For the term I 2 , we have

<!-- formula-not-decoded -->

where the inequality holds due to 0 ≤ V t ( s ) , V π t ( s ) ≤ 1 / (1 -γ ) . Combining (B.3) and (B.4), we complete the proof of Lemma 5.2.

## B.3 Proof of Lemma 5.3

Proof of Lemma 5.3. On the event E , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where first inequality holds due to the definition of E 2 and the second inequality holds due to 0 ≤ V t +1 ( s ′ ) -V ∗ ( s ′ ) ≤ 1 / (1 -γ ) . To bound term I 1 , we separate S into two subsets S 1 t ∪S 2 t , where

Then on the event E 4 , we have

<!-- formula-not-decoded -->

where the first inequality holds due to separate condition of P ( s ′ ) , the second inequality holds due to Lemma B.3, the third inequality holds due to V t -1 ( s ′ ) ≥ V ∗ ( s ′ ) , the fourth inequality holds due to the definition of event E 4 , the fifth inequality holds due to V ∗ ≥ V π t +1 , and the last inequality holds due to Lemma 5.2. For the term I 2 , according to Lemma B.3, we have

<!-- formula-not-decoded -->

Substituting (B.6),(B.7) into (B.5), we complete the proof of Lemma 5.3.

## B.4 Proof of Lemma 5.4

Proof of Lemma 5.4. On the event E 1 ∩ E 5 ∩ E 7 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to the definition of event E 1 , the second inequality holds due to Cauchy-Schwarz inequality, the third inequality holds due to Lemma B.3 and the definition of U , and the last inequality holds due to Lemma B.8 and the definition of event E 5 . Thus, we complete the proof of Lemma 5.4.

## B.5 Proof of Lemma 5.5

Proof of Lemma 5.5. For the term UCB t -1 ( s t , a t ) , we have

<!-- formula-not-decoded -->

For the term I 1 , on the event E 5 ∩ E 6 ∩ E 8 , we have

<!-- formula-not-decoded -->

where the first inequality holds due to Cauchy-Schwarz inequality, the second inequality holds due to Lemma B.3, the last inequality holds due to the definition of event E 5 and Lemma B.9. For the term I 2 , by Lemma B.3, we have

<!-- formula-not-decoded -->

For the term I 3 , on the event E 8 ∩ E 9 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to Cauchy-Schwarz inequality, the second inequality holds due to Lemma B.3, the third inequality holds due to the definition of event E 8 , the forth inequality holds due to the definition of event E 9 and the last inequality holds due to Lemma B.3. Substituting (B.10), (B.11) and (B.12) into (B.9), we complete the proof of Lemma 5.5.

## C Proof of Lemmas in Section 5.2

## C.1 Proof of Lemma 5.6

Proof of Lemma 5.6. We have

<!-- formula-not-decoded -->

where the first inequality holds due to 0 ≤ r ( s t , a t ) ≤ 1 and the last inequality holds due to ∑ ∞ k =0 γ k = 1 / (1 -γ ) . Thus, we finish the proof of Lemma 5.6.

## C.2 Proof of Lemma 5.7

Proof of Lemma 5.7. In this proof, we follow the proof technique in [15] and [10]. For simplicity, we denote /epsilon1 = √ A (1 -γ ) /K/ 24 and we first determine the optimal policy in these hard-to-learn MDPs. According to (3.1), for optimal policy π ∗ , we have

<!-- formula-not-decoded -->

For each j ∈ [ S ] and state s = s j, 1 , the choice of action a will not effect the reward r ( s, a ) and the probability transition function P ( ·| s, a ) . For optimal action a ∗ at state s = s j, 0 , we have

<!-- formula-not-decoded -->

Since P ( s j, 0 | s j, 0 , a ∗ ) + P ( s j, 1 | s j, 0 , a ∗ ) = 1 , we have

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

and it implies that V ∗ ( s j, 1 ) ≥ V ∗ ( s j, 0 ) . Therefore, for all action a = a ∗ j , we have Q ∗ ( s j, 0 , a ∗ j ) ≥ Q ∗ ( s j, 0 , a ) and it further implies that the optimal action at state s = s j, 0 is a ∗ j . Thus, according to the optimal bellman equation 3.1, for each j ∈ [ S ] , we have and it implies that the optimal value function V ∗ is

<!-- formula-not-decoded -->

When an agent visits the state set { s j, 0 , s j, 1 } for the i -th time, we denote the state in { s j, 0 , s j, 1 } it visited as X j,i , and the following action selected by the agent as A j,i . For each j ∈ [ S ] , by the definition of X j,i , we have

<!-- formula-not-decoded -->

where the third equality holds because when X j,i -1 leave state s j, 0 , s j, 1 , the next state in s j, 0 , s j, 1 must be s j, 0 . Similar to the proof of Theorem 5 in [10], we focus on the first K visits to the state set { s j, 0 , s j, 1 } and let random variable N 0 , N 1 and N ∗ 0 denote the total number of visit state s j, 0 , the total number of visit state s j, 1 and the total number of visit state s j, 0 with action a ∗ j . By the same argument as the proof of Theorem 5 in [10], for the random variable N 1 and N ∗ 0 , we have following property:

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, the regret can be upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality holds due to the fact that E [ N 0 ] + E [ N 1 ] = K , the third inequality holds due to (C.2) and the last inequality holds due to (C.3). Since K ≥ 10 SA/ (1 -γ ) 4 , γ &gt; 2 / 3 and A ≥ 30 , (C.4) can be further bounded by

<!-- formula-not-decoded -->

where the second inequality holds to /epsilon1 = √ A (1 -γ ) /K/ 24 ≤ 1 -γ with K ≥ 10 SA/ (1 -γ ) 4 , the third inequality holds due to /epsilon1 = √ A (1 -γ ) /K/ 24 with A ≥ 30 and the last inequality holds due to γ ≥ 2 / 3 and /epsilon1 = √ A (1 -γ ) /K/ 24 ≤ 1 -γ . Therefore, we finish the proof of Lemma 5.7.

## C.3 Proof of Lemma 5.8

Proof of Lemma 5.8. For each j ∈ [ S ] and t ∈ [ T ] , we denote H = /floorleft log T/ (1 -γ ) /floorright +1 , random variable

<!-- formula-not-decoded -->

and filtration F j,i contain all random variable before X j,i + H . For simplicity, we ignore the subscript j and only focus on the subscript i .

Since Y i is F i -measurable and 0 ≤ Y i ≤ 1 / (1 -γ ) , for each k ∈ [ H ] , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

where the first inequality holds due to Lemma B.1 and the second inequality holds due to the definition of optimal value function V ∗ . Taking summation of (C.6), for all k ∈ [ H ] , with probability at least 1 -Hδ , we have

<!-- formula-not-decoded -->

where the second inequality holds due to 0 ≤ r ( s, a ) ≤ 1 . Finally, taking union for all j ∈ [ S ] and t ∈ [ T ] , we complete the proof.

## C.4 Proof of Lemma 5.9

Proof of Lemma 5.9. Let Y j,i be an indicator random variables which denote whether the agent at state X j,i with action A j,i goes to the different state. Y j,i = 1 if the agent goes to the different state and Y j,i = 0 if the agent stay at the same state. Let filtration F j,i contain all random variables before X j,i . Then, for each j ∈ [ S ] , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

where the first inequality holds due to Lemma B.1, the second inequality holds due to the definition of our MDPs and the last one holds due to the selection of K . Similarly, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

where the first inequality holds due to Lemma B.1, the second inequality holds due to the definition of our MDPs and the last one holds due to the selection of K . Taking a union bound (C.7) and (C.8) for all j ∈ [ S ] , then we have (C.7) and (C.8) hold with probability at least 1 -2 Sδ . Let Z j,i be the number of times for the agent to start from state s j,i and travel the next different state in the first T steps. By definition, we have

<!-- formula-not-decoded -->

By Pigeonhole principle, there exist a j ∗ such that T j ∗ ≥ T/S = 10 K &gt; 5 K . Therefore, we have

<!-- formula-not-decoded -->

Furthermore, after leaving the state s j ∗ , 0 , the agent will visit all other states before arrive the state s j ∗ , 0 again. Thus, for any k ∈ [ S ] , the difference between Z j ∗ , 0 and Z k, 0 is at most 1, so do Z j ∗ , 1 and Z k, 1 . Therefore, for any k ∈ [ S ] , we have

<!-- formula-not-decoded -->

where the second inequality holds due to (C.10), the third inequality holds since K &gt; 2 / (1 -γ ) and the last one holds due to (C.7). Finally, by (C.9) we have Z k, 0 + Z k, 1 = ∑ T k i =1 Y k,i . Combining it with (C.11), we have ∑ T k i =1 Y k,i &gt; ∑ K i =1 Y k,i , which suggests that T k &gt; k . Thus, we complete the proof.

## D Proof of Lemmas in Appendix B

## D.1 Proof of Lemma B.3

Proof of Lemma B.3. We have

<!-- formula-not-decoded -->

We also have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to (D.1), for a subset C ⊆ [ T ] , we have where the first inequality holds due to Cauchy-Schwarz inequality and the second inequality holds due to (D.1). Thus, we complete the proof.

## D.2 Proof of Lemma B.4

Proof of Lemma B.4. For each s ∈ S , a ∈ A , we denote t 0 = 0 and

Here, t i is the time which state-action pair ( s, a ) appear for the i th time and the random variable t i is a stopping time. Beside, the random variable V ∗ ( s t i +1 )( i = 1 , 2 ., , ) are random variable with value in [ 0 , 1 / (1 -γ ) ] and variance V ∗ ( s, a ) . By Lemma B.2 and a union bound, with probability at least 1 -δ , for all s ∈ S , a ∈ A , τ ∈ [ T ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for all τ ∈ [ T ] , we have

In addition, for τ = 0 , we have

<!-- formula-not-decoded -->

where the first inequality holds due to 0 ≤ V ∗ ( s ) ≤ 1 / (1 -γ ) and the second inequality holds due to N t τ ( s, a ) = 0 . Since P t and N t -1 ( s, a ) changed only when t = t τ +1 , we complete the proof by combining (D.3) and (D.4).

## D.3 Proof of Lemma B.6

Proof of Lemma B.6. For each s ∈ S , a ∈ A , we denote t 0 = 0 and denote

Here, t i is the time which state-action pair ( s, a ) appear for the i th time and the random variable t i is a stopping time. Beside, the random variable V ∗ ( s t i +1 )( i = 1 , 2 ., , ) are random variable with value in [ 0 , 1 / (1 -γ ) ] and variance V ∗ ( s, a ) . By Lemma B.5 and a union bound, with probability at least 1 -δ , for all s ∈ S , a ∈ A , τ ∈ [ T ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for all τ ∈ [ T ] , we have

In addition, for τ = 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to 0 ≤ V ∗ ( s ) ≤ 1 / (1 -γ ) and the second inequality holds due to N t τ ( s, a ) = 0 . Since P t , V ∗ t -1 and N t -1 ( s, a ) changed only when t = t τ +1 , we complete the proof by combining (D.6) and (D.7).

## D.4 Proof of Lemma B.7

Proof of Lemma B.7. For simplicity, we denote H = /floorleft 1 / (1 -γ ) /floorright +1 , T ′ = /floorleft T/H /floorright +1 and filtration F t contained all random variables before first t + H steps. Then for every t ∈ [ T ] , we have where the first inequality holds due to 0 ≤ r ( s, a ) ≤ 1 , 0 ≤ V π t ( s ) ≤ 1 / (1 -γ ) and the second inequality holds due to V π t + i ( s t + i , a t + i ) ≥ 0 . For the random variable X t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since X t is F t -measurable and E [ X t |F t -H ] ≤ 1 / (1 -γ ) 2 , for each i ∈ [ H ] , by Lemma B.2, with probability at least 1 -δ , we have

Taking summation for (D.9) with all i ∈ [ H ] , with probability at least 1 -Hδ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to (D.9), the second inequality holds due to T ′ = /floorleft T/H /floorright + 1 and the third inequality holds due to x 2 + y 2 ≥ 2 xy . By the definition of X t , we have

<!-- formula-not-decoded -->

where the first inequality holds due to V π t ( s t , a t ) ≥ 0 and the second inequality holds due to V π t ( s t , a t ) ≤ 1 / (1 -γ ) 2 . To further bound (D.11), we have

<!-- formula-not-decoded -->

where the first inequality holds since 2 H +2 = 2 /floorleft 1 / (1 -γ ) /floorright +2 ≥ 2 / (1 -γ ) , the second inequality holds since 0 ≤ γ 1 / (1 -γ ) ≤ 0 . 4 when 0 ≤ γ ≤ 1 , the last one holds since 1 + γ ≤ 2 . We also have

<!-- formula-not-decoded -->

Substituting (D.12) and (D.13) into (D.11), we have

<!-- formula-not-decoded -->

Finally, substituting (D.14) into (D.10), we have

<!-- formula-not-decoded -->

Thus, we complete the proof.

## D.5 Proof of Lemma B.8

Proof of Lemma B.8. On the event E 7 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds because of Lemma 5.1, the second inequality holds due to 0 ≤ V ∗ ( s ) , V π i +1 ( s ) ≤ 1 1 -γ , the third inequality holds due to the definition of E 7 and the last inequality holds due to 0 ≤ V ∗ ( s ) ≤ V i ( s ) ≤ 1 / 1 -γ . Thus, we complete the proof.

## D.6 Proof of Lemma B.9

Proof of Lemma B.9.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality holds due to V i -1 ( s ′ ) ≥ V ∗ ( s ′ ) ≥ V π i +1 ( s ′ ) . By the definition of event E 8 , we have

Thus, for the term I 1 , since 0 ≤ V 2 i -1 ( s ′ ) ≤ 1 / (1 -γ ) 2 , we have where the first inequality holds due to (D.15) and the second inequality holds due to Lemma B.3. For the term I 2 , on the event E 6 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the term I 3 , since 0 ≤ V ∗ ( s ′ ) 2 ≤ 1 / (1 -γ ) 2 , on the event E 8 , we have where the first inequality holds due to V i -1 ( s ′ ) ≥ V ∗ ( s ′ ) ≥ V π i +1 ( s ′ ) , the second inequality holds due to 0 ≤ V i -1 ( s ′ ) , V π i +1 ( s ′ ) ≤ 1 / (1 -γ ) , the third inequality holds due to the definition of event E 6 and the forth inequality holds due to V i -1 ( s ′ ) ≥ V i +1 ( s ′ ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to (D.15) and the second inequality holds due to Lemma B.3. Taking an union bound for (D.16), (D.17) and (D.18), with probability at least 1 -3 δ , we have

## D.7 Proof of Lemma B.10

Proof of Lemma B.10. For each i ∈ [ H ] , s ∈ S and t ∈ [ T ] , if N t ( s ) = 0 , the we have

Otherwise, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to definition update rule (4.2). I 1 , . . . , I 4 are defined as follows.

<!-- formula-not-decoded -->

For the term I 1 , we have

<!-- formula-not-decoded -->

For the term I 2 , with probability at least 1 -δ , we have where the first inequality holds due to V i + h -1 ( s ′ ) ≥ V i + h +1 ( s ′ ) and the second inequality holds due to 0 ≤ V t ( s ) ≤ 1 / (1 -γ ) .

<!-- formula-not-decoded -->

where the first inequality holds due to Lemma B.1 and the definition of U , the second inequality holds due to Cauchy-Schwarz inequality and the third inequality holds due to Lemma B.3.

For the term I 3 , Since the random process s i + h +1 ∼ P ( ·| s i + h , a i + h ) is dependent with whether s i +1 , .., s i + h +1 = s , we cannot directly use Lemma B.1 to bound this term. However, we can use the same technique in the proof of Lemme B.7, which divide the time horizon into H sub-horizon and use Lemma B.1 for each sub-horizon. Compared with the upper bound of I 3 in proof of Theorem 4.5, this technique will lead to a gap of √ H and we have where the second inequality holds due to the definition of U . For the term I 4 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to Cauchy-Schwarz inequality, the second inequality holds due to Lemma B.3, the last inequality holds due to 0 ≤ V i + h -1 ( s i + h , a i + h ) ≤ 1 / (1 -γ ) 2 . For the term I 42 , by Lemma B.3, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the term I 43 , with probability at least 1 -2 δ , we have where the first inequality holds due to Cauchy-Schwarz inequality, the second inequality holds due to Lemma B.3, the third inequality holds due to Lemma B.1, the forth inequality holds due to Lemma B.1 and the last inequality holds due to Lemma B.3. Substituting (D.20), (D.21), (D.22), (D.23) into (D.19), with probability at least 1 -4 Hδ , we have

Notice that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality holds due to V i + H ( s i + H ) -V π i + H ( s i + H ) ≤ 1 / (1 -γ ) and the second inequality holds due to definition of H . Thus, taking summation of (D.27) with all h ∈ [ H ] , with probability at least 1 -H 2 δ , we have

In addition, if N t ( s ) &gt; 0 , we have

<!-- formula-not-decoded -->

where the first inequality holds due to V i ( s ) is decreasing, the second inequality holds due to V ∗ ( s ) ≥ V π i ( s ) and the third inequality holds due to (D.28). Notice that when N t ( s ) ≥ S 2 AU 3 / (1 -γ ) 2 , we have

Otherwise, we have

<!-- formula-not-decoded -->

Thus, we complete the proof of Lemma B.10.

<!-- formula-not-decoded -->