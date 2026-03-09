## Is Q-learning Provably Efficient?

Chi Jin ∗ University of California, Berkeley chijin@cs.berkeley.edu

Sebastien Bubeck Microsoft Research, Redmond sebubeck@microsoft.com

Zeyuan Allen-Zhu ∗ Microsoft Research, Redmond zeyuan@csail.mit.edu

Michael I. Jordan University of California, Berkeley jordan@cs.berkeley.edu

## Abstract

Model-free reinforcement learning (RL) algorithms, such as Q-learning, directly parameterize and update value functions or policies without explicitly modeling the environment. They are typically simpler, more flexible to use, and thus more prevalent in modern deep RL than model-based approaches. However, empirical work has suggested that model-free algorithms may require more samples to learn [7, 22]. The theoretical question of 'whether model-free algorithms can be made sample efficient ' is one of the most fundamental questions in RL, and remains unsolved even in the basic scenario with finitely many states and actions.

We prove that, in an episodic MDP setting, Q-learning with UCB exploration achieves regret ˜ O ( √ H 3 SAT ), where S and A are the numbers of states and actions, H is the number of steps per episode, and T is the total number of steps. This sample efficiency matches the optimal regret that can be achieved by any model-based approach, up to a single √ H factor. To the best of our knowledge, this is the first analysis in the model-free setting that establishes √ T regret without requiring access to a 'simulator.'

## 1 Introduction

Reinforcement Learning (RL) is a control-theoretic problem in which an agent tries to maximize its cumulative rewards via interacting with an unknown environment through time [26]. There are two main approaches to RL: model-based and model-free. Model-based algorithms make use of a model for the environment, forming a control policy based on this learned model. Model-free approaches dispense with the model and directly update the value function -the expected reward starting from each state, or the policy -the mapping from states to their subsequent actions. There has been a long debate on the relative pros and cons of the two approaches [7].

From the classical Q-learning algorithm [27] to modern DQN [17], A3C [18], TRPO [22], and others, most state-of-the-art RL has been in the model-free paradigm. Its pros-model-free algorithms are online, require less space, and, most importantly, are more expressive since specifying the value functions or policies is often more flexible than specifying the model for the environmentarguably outweigh its cons relative to model-based approaches. These relative advantages underly the significant successes of model-free algorithms in deep RL applications [17, 24].

On the other hand it is believed that model-free algorithms suffer from a higher sample complexity compared to model-based approaches. This has been evidenced empirically in [7, 22], and

∗ The first two authors contributed equally.

recent work has tried to improve the sample efficiency of model-free algorithms by combining them with model-based approaches [19, 21]. There is, however, little theory to support such blending, which requires a more quantitative understanding of relative sample complexities. Indeed, the following basic theoretical questions remain open:

## Can we design model-free algorithms that are sample efficient? In particular, is Q-learning provably efficient?

The answers remain elusive even in the basic tabular setting where the number of states and actions are finite. In this paper, we attack this problem head-on in the setting of the episodic Markov Decision Process (MDP) formalism (see Section 2 for a formal definition). In this setting, an episode consists of a run of MDP dynamics for H steps, where the agent aims to maximize total reward over multiple episodes. We do not assume access to a 'simulator' (which would allow us to query arbitrary state-action pairs of the MDP) and the agent is not allowed to 'reset' within each episode. This makes our setting sufficiently challenging and realistic. In this setting, the standard Q-learning heuristic of incorporating ε -greedy exploration appears to take exponentially many episodes to learn [14].

As seen in the literature on bandits, the key to achieving good sample efficiency generally lies in managing the tradeoff between exploration and exploitation . One needs an efficient strategy to explore the uncertain environment while maximizing reward. In the model-based setting, a recent line of research has imported ideas from the bandit literature-including the use of upper confidence bounds (UCB) and improved design of exploration bonuses-and has obtained asymptotically optimal sample efficiency [1, 5, 10, 12]. In contrast, the understanding of model-free algorithms is still very limited. To the best of our knowledge, the only existing theoretical result on model-free RL that applies to the episodic setting is for delayed Q-learning ; however, this algorithm is quite sample-inefficient compared to model-based approaches [25].

In this paper, we answer the two aforementioned questions affirmatively. We show that Qlearning, when equipped with a UCB exploration policy that incorporates estimates of the confidence of Q values and assign exploration bonuses, achieves total regret ˜ O ( √ H 3 SAT ). Here, S and A are the numbers of states and actions, H is the number of steps per episode, and T is the total number of steps. Up to a √ H factor, our regret matches the information-theoretic optimum, which can be achieved by model-based algorithms [5, 12]. Since our algorithm is just Q-learning, it is online and does not store additional data besides the table of Q values (and a few integers per entry of this table). Thus, it also enjoys a significant advantage over model-based algorithms in terms of time and space complexities. To our best knowledge, this is the first sharp analysis for model-free algorithms-featuring √ T regret or equivalently O (1 /ε 2 ) samples for ε -optimal policywithout requiring access to a 'simulator.'

For practitioners, there are two key takeaways from our theoretical analysis:

1. The use of UCB exploration instead of ε -greedy exploration in the model-free setting allows for better treatment of uncertainties for different states and actions.
2. It is essential to use a learning rate which is α t = O ( H/t ), instead of 1 /t , when a state-action pair is being updated for the t -th time. The former learning rate assigns more weight to updates that are more recent, as opposed to assigning uniform weights to all previous updates. This delicate choice of reweighting leads to the crucial difference between our sample-efficient guarantee versus earlier highly inefficient results that require exponentially many samples in H .

Table 1: Regret comparisons for RL algorithms on episodic MDP. T = KH is totally number of steps, H is the number of steps per episode, S is the number of states, and A is the number of actions. For clarity, this table is presented for T ≥ poly( S, A, H ), omitting low order terms.

|             | Algorithm                                       | Regret                        | Time           | Space        |
|-------------|-------------------------------------------------|-------------------------------|----------------|--------------|
| Model-based | UCRL2 [10] 1                                    | at least ˜ O ( √ H 4 S 2 AT ) | Ω( TS 2 A )    | O ( S 2 AH ) |
| Model-based | Agrawal and Jia [1] 1                           | at least ˜ O ( √ H 3 S 2 AT ) | Ω( TS 2 A )    | O ( S 2 AH ) |
| Model-based | UCBVI [5] 2                                     | ˜ O ( √ H 2 SAT )             | ˜ O ( TS 2 A ) | O ( S 2 AH ) |
| Model-based | vUCQ [12] 2                                     | ˜ O ( √ H 2 SAT )             | ˜ O ( TS 2 A ) | O ( S 2 AH ) |
| Model-free  | Q-learning ( ε -greedy) [14] (if 0 initialized) | Ω(min { T,A H/ 2 } )          | O ( T )        | O ( SAH )    |
| Model-free  | Delayed Q-learning [25] 3                       | ˜ O S,A,H ( T 4 / 5 )         | O ( T )        | O ( SAH )    |
| Model-free  | Q-learning (UCB-H)                              | ˜ O ( √ H 4 SAT )             | O ( T )        | O ( SAH )    |
| Model-free  | Q-learning (UCB-B)                              | ˜ O ( √ H 3 SAT )             | O ( T )        | O ( SAH )    |
|             | lower bound                                     | Ω( √ H 2 SAT )                | -              | -            |

## 1.1 Related Work

In this section, we focus our attention on theoretical results for the tabular MDP setting, where the numbers of states and actions are finite. We acknowledge that there has been much recent work in RL for continuous state spaces [see, e.g., 9, 11], but this setting is beyond our scope.

With simulator. Some results assume access to a simulator [15] (a.k.a., a generative model [3]), which is a strong oracle that allows the algorithm to query arbitrary state-action pairs and return the reward and the next state. The majority of these results focus on an infinite-horizon MDP with discounted reward [e.g., 2, 3, 8, 16, 23]. When a simulator is available, model-free algorithms [2] (variants of Q-learning) are known to be almost as sample efficient as the best model-based algorithms [3]. However, the simulator setting is considered to much easier than standard RL, as it 'does not require exploration' [2]. Indeed, a naive exploration strategy which queries all stateaction pairs uniformly at random already leads to the most efficient algorithm for finding optimal policy [3].

Without simulator. Reinforcement learning becomes much more challenging without the presence of a simulator, and the choice of exploration policy can now determine the behavior of the learning algorithm. For instance, Q-learning with ε -greedy may take exponentially many episodes to learn the optimal policy [14] (for the sake of completeness, we present this result in our episodic language in Appendix A).

1 Jaksch et al. [10] and Agrawal and Jia [1] apply to the more general setting of weakly communicating MDPs with S ′ states and diameter D ; our episodic MDP is a special case obtained by augmenting the state space so that S ′ = SH and D ≥ H .

3 Strehl et al. [25] applies to MDPs with S ′ states and discount factor γ ; our episodic MDP can be converted to that case by setting S ′ = SH and 1 -γ = 1 /H . Their result only applies to the stochastic setting where initial states x k 1 come from a fixed distribution, and only gives a PAC guarantee. We have translated it to a regret guarantee (see Section 3.1).

2 Azar et al. [5] and Kakade et al. [12] assume equal transition matrices P 1 = · · · = P H ; in the setting of this paper P 1 , · · · , P H can be entirely different. This adds a factor of √ H to their total regret.

In the model-based setting, UCRL2 [10] and Agrawal and Jia [1] form estimates of the transition probabilities of the MDP using past samples, and add upper-confidence bounds (UCB) to the estimated transition matrix. When applying their results to the episodic MDP scenario, their total regret is at least ˜ O ( √ H 4 S 2 AT ) and ˜ O ( √ H 3 S 2 AT ) respectively. 1 In contrast, the informationtheoretic lower bound is ˜ O ( √ H 2 SAT ). The additional √ S and √ H factors were later removed by the UCBVI algorithm [5] which adds a UCB bonus directly to the Q values instead of the estimated transition matrix. 2 The vUCQ algorithm [12] is similar to UCBVI but improves lower-order regret terms using variance reduction.

We note that despite the sharp regret guarantees, all of the results in this line of research require estimating and storing the entire transition matrix and thus suffer from unfavorable time and space complexities compared to model-free algorithms.

In the model-free setting, Strehl et al. [25] introduced delayed Q-learning, where, to find an ε -optimal policy, the Q value for each state-action pair is updated only once every m = ˜ O (1 /ε 2 ) times this pair is visited. In contrast to the incremental update of Q-learning, delayed Q-learning always replaces old Q values with the average of the most recent m experiences. When translated to the setting of this paper, this gives ˜ O ( T 4 / 5 ) total regret, ignoring factors in S, A and H . 3 This is quite suboptimal compared to the ˜ O ( √ T ) regret achieved by model-based algorithm.

## 2 Preliminary

We consider the setting of a tabular episodic Markov decision process, MDP( S , A , H , P , r), where S is the set of states with |S| = S , A is the set of actions with |A| = A , H is the number of steps in each episode, P is the transition matrix so that P h ( ·| x, a ) gives the distribution over states if action a is taken for state x at step h ∈ [ H ], and r h : S × A → [0 , 1] is the deterministic reward function at step h . 4

In each episode of this MDP, an initial state x 1 is picked arbitrarily by an adversary. Then, at each step h ∈ [ H ], the agent observes state x h ∈ S , picks an action a h ∈ A , receives reward r h ( x h , a h ), and then transitions to a next state, x h +1 , that is drawn from the distribution P h ( ·| x h , a h ). The episode ends when x H +1 is reached.

<!-- formula-not-decoded -->

A policy π of an agent is a collection of H functions { π h : S → A } h ∈ [ H ] . We use V π h : S → R to denote the value function at step h under policy π , so that V π h ( x ) gives the expected sum of remaining rewards received under policy π , starting from x h = x , until the end of the episode. In symbols:

Accordingly, we also define Q π h : S × A → R to denote Q -value function at step h so that Q π h ( x, a ) gives the expected sum of remaining rewards received under policy π , starting from x h = x, a h = a , till the end of the episode. In symbols:

<!-- formula-not-decoded -->

Since the state and action spaces, and the horizon, are all finite, there always exists (see, e.g., [5]) an optimal policy π /star which gives the optimal value V /star h ( x ) = sup π V π h ( x ) for all x ∈ S and h ∈ [ H ]. For simplicity, we denote [ P h V h +1 ]( x, a ) := E x ′ ∼ P ( ·| x,a ) V h +1 ( x ′ ). Recall the Bellman equation and

4 While we study deterministic reward functions for notational simplicity, our results generalize to randomized reward functions. Also, we assume the reward is in [0 , 1] without loss of generality.

## Algorithm 1 Q-learning with UCB-Hoeffding

```
1: initialize Q h ( x, a ) ← H and N h ( x, a ) ← 0 for all ( x, a, h ) ∈ S × A × [ H ]. 2: for episode k = 1 , . . . , K do 3: receive x 1 . 4: for step h = 1 , . . . , H do 5: Take action a h ← argmax a ′ Q h ( x h , a ′ ), and observe x h +1 . 6: t = N h ( x h , a h ) ← N h ( x h , a h ) + 1; b t ← c √ H 3 ι/t . 7: Q h ( x h , a h ) ← (1 -α t ) Q h ( x h , a h ) + α t [ r h ( x h , a h ) + V h +1 ( x h +1 ) + b t ]. 8: V h ( x h ) ← min { H, max a ′ ∈A Q h ( x h , a ′ ) } .
```

the Bellman optimality equation:

The agent plays the game for K episodes k = 1 , 2 , . . . , K , and we let the adversary pick a starting state x k 1 for each episode k , and let the agent choose a policy π k before starting the k -th episode. The total (expected) regret is then

## 3 Main Results

In this section, we present our main theoretical result-a sample complexity result for a variant of Q-learning that incorporates UCB exploration. We also present a theorem that establishes an information-theoretic lower bound for episodic MDP.

As seen in the bandit setting, the choice of exploration policy plays an essential role in the efficiency of a learning algorithm. In episodic MDP, Q-learning with the commonly used ε -greedy exploration strategy can be very inefficient: it can take exponentially many episodes to learn [14] (see also Appendix A). In contrast, our algorithm (Algorithm 1), which is Q-learning with an upper-confidence bound (UCB) exploration strategy, will be seen to be efficient. This algorithm maintains Q values, Q h ( x, a ), for all ( x, a, h ) ∈ S × A × [ H ] and the corresponding V values V h ( x ) ← min { H, max a ′ ∈A Q h ( x, a ′ ) } . If, at time step h ∈ [ H ], the state is x ∈ S , the algorithm takes the action a ∈ A that maximizes the current estimate Q h ( x, a ), and is apprised of the next state x ′ ∈ S . The algorithm then updates the Q values:

<!-- formula-not-decoded -->

where t is the counter for how many times the algorithm has visited the state-action pair ( x, a ) at step h , b t is the confidence bonus indicating how certain the algorithm is about current state-action pair, and α t is a learning rate defined as follows:

<!-- formula-not-decoded -->

As mentioned in the introduction, our choice of learning rate α t scales as O ( H/t ) instead of O (1 /t )-this is crucial to obtain regret that is not exponential in H .

We present analyses for two different specifications of the upper confidence bonus b t in this

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

paper:

Q-learning with Hoeffding-style bonus. The first (and simpler) choice is b t = O ( √ H 3 ι/t ). (Here, and throughout this paper, we use ι := log( SAT/p ) to denote a log factor.) This choice of bonus makes sense intuitively because: (1) Q-values are upper-bounded by H , and, accordingly, (2) Hoeffding-type martingale concentration inequalities imply that if we have visited ( x, a ) for t times, then a confidence bound for the Q value scales as 1 / √ t . For this reason, we call this choice UCB-Hoeffding (UCB-H). See Algorithm 1.

Theorem 1 (Hoeffding) . There exists an absolute constant c &gt; 0 such that, for any p ∈ (0 , 1) , if we choose b t = c √ H 3 ι/t , then with probability 1 -p , the total regret of Q-learning with UCB-Hoeffding (see Algorithm 1) is at most O ( √ H 4 SATι ) , where ι := log( SAT/p ) .

Theorem 1 shows, under a rather simple choice of exploration bonus, Q-learning can be made very efficient, enjoying a ˜ O ( √ T ) regret which is optimal in terms of dependence on T . To the best of our knowledge, this is the first analysis of a model-free procedure that features a √ T regret without requiring access to a 'simulator.'

Compared to the previous model-based results, Theorem 1 shows that the regret (or equivalently the sample complexity; see discussion in Section 3.1) of this version of Q-learning is as good as the best model-based one in terms of the dependency on the number of states S , actions A and the total number of steps T . Although our regret slightly increases the dependency on H , the algorithm is online and does not store additional data besides the table of Q values (and a few integers per entry of this table). Thus, it enjoys an advantage over model-based algorithms in time and space complexities, especially when the number of states S is large.

Q-learning with Bernstein-style bonus. Our second specification of b t makes use of a Bernsteinstyle upper confidence bound. The key observation is that, although in the worst case the value function is at most H for any state-action pair, if we sum up the 'total variance of the value function' for an entire episode, we obtain a factor of only O ( H 2 ) as opposed to the naive O ( H 3 ) bound (see Lemma C.5). This implies that the use of a Bernstein-type martingale concentration result could be sharper than the Hoeffding-type bound by an additional factor of H . 5 (The idea of using Bernstein instead of Hoeffding for reinforcement learning applications has appeared in previous work; see, e.g., [3, 4, 16].)

Using Bernstein concentration requires us to design the bonus term b t more carefully, as it now depends on the empirical variance of V h +1 ( x ′ ) where x ′ is the next state over the previous t visits of current state-action ( x, a ). This empirical variance can be computed in an online fashion without increasing the space complexity of Q-learning. We defer the full specification of b t to Algorithm 2 in Appendix C. We now state the regret theorem for this approach.

Theorem 2 (Bernstein) . For any p ∈ (0 , 1) , one can specify b t so that with probability 1 -p , the total regret of Q-learning with UCB-Bernstein (see Algorithm 2) is at most O ( √ H 3 SATι + √ H 9 S 3 A 3 · ι 2 ) .

Theorem 2 shows that for Q-learning with UCB-B exploration, the leading term in regret (which scales as √ T ) improves by a factor of √ H over UCB-H exploration, at the price of using a more complicated exploration bonus design. The asymptotic regret of UCB-B is now only one √ H factor worse than the best regret achieved by model-based algorithms.

5 Recall that for independent zero-mean random variables X 1 , . . . , X T satisfying | X i | ≤ M , their summation does not exceed ˜ O ( M √ T ) with high probability using Hoeffding concentration. If we have in hand a better variance bound, this can be improved to ˜ O ( M + √∑ i E [ X i ] 2 ) using Bernstein concentration.

We also note that Theorem 2 has an additive term O ( √ H 9 S 3 A 3 · ι 2 ) in its regret, which dominates the total regret when T is not very large compared with S, A and H . It is not clear whether this lower-order term is essential, or is due to technical aspects of the current analysis.

Information-theoretical limit. To demonstrate the sharpness of our results, we also note an information-theoretic lower bound for the episodic MDP setting studied in this paper:

Theorem 3. For the episodic MDP problem studied in this paper, the expected regret for any algorithm must be at least Ω( √ H 2 SAT ) .

Theorem 3 (see Appendix D for details) shows that both variants of our algorithm are nearly optimal, in the sense they differ from the optimal regret by a factor of H and √ H , respectively.

## 3.1 From Regret to PAC Guarantee

Recall that the probably approximately correct (PAC) learning setting for RL provides sample complexity guarantee to find a near-optimal policy [13]. In this setting, the initial state x 1 ∈ S is sampled from a fixed initial distribution, rather than being chosen adversarially. Without loss of generality, we only discuss here the case in which x 1 is fixed; the general case reduces to this case by adding an additional time step at the beginning of each episode. The PAC-learning question is 'how many samples are needed to find an ε -optimal policy π satisfying V /star 1 ( x 1 ) -V π 1 ( x 1 ) ≤ ε ?'

Conversely, any algorithm with finite sample complexity in the PAC setting translates to sublinear total regret in non-adversarial case (assuming x 1 is chosen from a fixed distribution). Suppose the algorithm finds ε -optimal policy π using T 1 = C · ε -β samples where β ≥ 1 is a constant. Then, we can use this π to play the game for another T -T 1 steps, giving total regret T 1 + ε ( T -T 1 ) /H . After balancing T and T 1 optimally, this gives ˜ O ( C 1+ β · ( T/H ) β/ (1+ β ) ) total regret. For instance, Strehl et al. [25] gives sampling complexity ∝ 1 /ε 4 in the PAC setting, and this translates to ∝ T 4 / 5 total regret.

Any algorithm with total regret sublinear in T yields a finite sample complexity in the PAC setting. Indeed, suppose we have total regret ∑ K k =1 [ V /star 1 ( x 1 ) -V π k 1 ( x 1 )] ≤ C · T 1 -α , where α ∈ (0 , 1) is a absolute constant, and C is independent of T . Then, by randomly selecting π = π k for k = 1 , 2 , . . . , K , we have V /star 1 ( x 1 ) -V π 1 ( x 1 ) ≤ 3 CH · T -α with probability at least 2 / 3. Therefore, for every ε ∈ (0 , H ], our Theorem 1 (for UCB-H) and Theorem 2 (for UCB-B) also find ε -optimal policies in the PAC setting using ˜ O ( H 5 SA/ε 2 ) and ˜ O ( H 4 SA/ε 2 ) samples respectively.

## 4 Proof for Q-learning with UCB-Hoeffding

In this section, we provide the full proof of Theorem 1. Intuitively, the episodic MDP with H steps per epsiode can be viewed as a contextual bandit of H 'layers.' The key challenge here is to control the way error and confidence propagate through different 'layers' in an online fashion, where our specific choice of exploration bonus and learning rate make the regret as sharp as possible.

Notation. We denote by I [ A ] the indicator function for event A . We denote by ( x k h , a k h ) the actual state-action pair observed and chosen at step h of episode k . We also denote by Q k h , V k h , N k h respectively the Q h , V h , N h functions at the beginning of episode k . Using this notation, the update equation at episode k can be rewritten as follows, for every h ∈ [ H ]:

<!-- formula-not-decoded -->

Figure 1: Illustration of { α i 1000 } 1000 i =1 for learning rates α t = H +1 H + t , 1 t and 1 √ t when H = 10.

<!-- image -->

Accordingly,

Recall that we have [ P h V h +1 ]( x, a ) := E x ′ ∼ P h ( ·| x,a ) V h +1 ( x ′ ). We also denote its empirical counterpart of episode k as [ ˆ P k h V h +1 ]( x, a ) := V h +1 ( x k h +1 ), which is defined only for ( x, a ) = ( x k h , a k h ).

<!-- formula-not-decoded -->

Recall that we have chosen the learning rate as α t := H +1 H + t . For notational convenience, we also introduce the following related quantities:

<!-- formula-not-decoded -->

Favoring Later Updates. At any ( x, a, h, k ) ∈ S × A × [ H ] × [ K ], let t = N k h ( x, a ) and suppose ( x, a ) was previously taken at step h of episodes k 1 , . . . , k t &lt; k . By the update equation (4.1) and the definition of α i t in (4.2), we have:

It is easy to verify that (1) ∑ t i =1 α i t = 1 and α 0 t = 0 for t ≥ 1; (2) ∑ t i =1 α i t = 0 and α 0 t = 1 for t = 0.

<!-- formula-not-decoded -->

According to (4.3), the Q value at episode k equals a weighted average of the V values of the 'next states' with weights α 1 t , . . . , α t t . As one can see from Figure 1, our choice of the learning rate α t = H +1 H + t ensures that, approximately speaking, the last 1 /H fraction of the indices i is given non-negligible weights, whereas the first 1 -1 /H fraction is forgotten. This ensures that the information accumulates smoothly across the H layers of the MDP. If one were to use α t = 1 t instead, the weights α 1 t , . . . , α t t would all equal 1 /t , and using those V values from earlier episodes would hurt the accuracy of the Q function. In contrast, if one were to use α t = 1 / √ t instead, the weights α 1 t , . . . , α t t would concentrate too much on the most recent episodes, which would incur high variance.

## 4.1 Proof Details

We first present an auxiliary lemma which exhibits some important properties that result from our choice of learning rate. The proof is based on simple manipulations on the definition of α t , and is provided in Appendix B.

Lemma 4.1. The following properties hold for α i t :

<!-- formula-not-decoded -->

- (b) max i ∈ [ t ] α i t ≤ 2 H t and ∑ t i =1 ( α i t ) 2 ≤ 2 H t for every t ≥ 1 .

We note that property ( c ) is especially important-as we will show later, each step in one episode can blow up the regret by a multiplicative factor of ∑ ∞ t = i α i t . With our choice of learning rate, we ensure that this blow-up is at most (1 + 1 /H ) H , which is a constant factor.

<!-- formula-not-decoded -->

We nnow proceed to the formal proof. We start with a lemma that gives a recursive formula for Q -Q /star , as a weighted average of previous updates.

Lemma 4.2 (recursion on Q ) . For any ( x, a, h ) ∈ S×A× [ H ] and episode k ∈ [ K ] , let t = N k h ( x, a ) and suppose ( x, a ) was previously taken at step h of episodes k 1 , . . . , k t &lt; k . Then:

<!-- formula-not-decoded -->

Proof of Lemma 4.2. From the Bellman optimality equation, Q /star h ( x, a ) = ( r h + P h V /star h +1 )( x, a ), our notation [ ˆ P k i h V h +1 ]( x, a ) := V h +1 ( x k i h +1 ), and the fact that ∑ t i =0 α i t = 1, we have

Subtracting the formula (4.3) from this equation, we obtain Lemma 4.2.

<!-- formula-not-decoded -->

/square

Next, using Lemma 4.2 and the Azuma-Hoeffding concentration bound, our next lemma shows that Q k is always an upper bound on Q /star at any episode k , and the difference between Q k and Q /star can be bounded by quantities from the next step.

Lemma 4.3 (bound on Q k -Q /star ) . There exists an absolute constant c &gt; 0 such that, for any p ∈ (0 , 1) , letting b t = c √ H 3 ι/t , we have β t = 2 ∑ t i =1 α i t b i ≤ 4 c √ H 3 ι/t and, with probability at least 1 -p , the following holds simultaneously for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] :

where t = N k h ( x, a ) and k 1 , . . . , k t &lt; k are the episodes where ( x, a ) was taken at step h .

<!-- formula-not-decoded -->

Proof of Lemma 4.3. For each fixed ( x, a, h ) ∈ S × A × [ H ], let us denote k 0 = 0, and denote

That is, k i is the episode of which ( x, a ) was taken at step h for the i th time (or k i = K + 1 if it is taken for fewer than i times). The random variable k i is clearly a stopping time. Let F i be the σ -field generated by all the random variables until episode k i , step h . Then, ( I [ k i ≤ K ] · [( ˆ P k i h -P h ) V /star h +1 ]( x, a ) ) τ i =1 is a martingale difference sequence w.r.t the filtration {F i } i ≥ 0 . By Azuma-Hoeffding and a union bound, we have that with probability at least 1 -p/ ( SAH ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some absolute constant c . Because inequality (4.4) holds for all fixed τ ∈ [ K ] uniformly, it also holds for τ = t = N k h ( x, a ) ≤ K , which is a random variable, where k ∈ [ K ]. Also note I [ k i ≤ K ] = 1

for all i ≤ N k h ( x, a ). Putting everything together, and using a union bound, we see that with least 1 -p probability, the following holds simultaneously for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ]:

∣ ∣ On the other hand, if we choose b t = c √ H 3 ι/t for the same constant c in Eq. (4.4), we have β t / 2 = ∑ t i =1 α i t b i ∈ [ c √ H 3 ι/t, 2 c √ H 3 ι/t ] according to Lemma 4.1.a. Then the right-hand side of Lemma 4.3 follows immediately from Lemma 4.2 and inequality (4.5). The left-hand side also follows from Lemma 4.2 and Eq. (4.5) and induction on h = H,H -1 , . . . , 1. /square

<!-- formula-not-decoded -->

We are now ready to prove Theorem 1. The proof decomposes the regret in a recursive form, and carefully controls the error propagation with repeated usage of Lemma 4.3.

Proof of Theorem 1. Denote by

<!-- formula-not-decoded -->

By Lemma 4.3, we have that with 1 -p probability, Q k h ≥ Q /star h and thus V k h ≥ V /star h . Thus, the total regret can be upper bounded:

<!-- formula-not-decoded -->

For any fixed ( k, h ) ∈ [ K ] × [ H ], let t = N k h ( x k h , a k h ), and suppose ( x k h , a k h ) was previously taken at step h of episodes k 1 , . . . , k t &lt; k . Then we have:

The main idea of the rest of the proof is to upper bound ∑ K k =1 δ k h by the next step ∑ K k =1 δ k h +1 , thus giving a recursive formula to calculate total regret. We can obtain such a recursive formula by relating ∑ K k =1 δ k h to ∑ K k =1 φ k h .

<!-- formula-not-decoded -->

We turn to computing the summation ∑ K k =1 δ k h . Denoting by n k h = N k h ( x k h , a k h ), we have:

where β t = 2 ∑ α i t b i ≤ O (1) √ H 3 ι/t and ξ k h +1 := [( P h -ˆ P k h )( V /star h +1 -V k h +1 )]( x k h , a k h ) is a martingale difference sequence. Inequality ① holds because V k h ( x k h ) ≤ max a ′ ∈A Q k h ( x k h , a ′ ) = Q k h ( x k h , a k h ), and inequality ② holds by Lemma 4.3 and the Bellman equation (2.1). Finally, equality ③ holds by definition δ k h +1 -φ k h +1 = ( V /star h +1 -V π k h +1 )( x k h +1 ).

<!-- formula-not-decoded -->

The key step is to upper bound the second term in (4.6), which is:

<!-- formula-not-decoded -->

where k i ( x k h , a k h ) is the episode in which ( x k h , a k h ) was taken at step h for the i th time. We regroup the summands in a different way. For every k ′ ∈ [ K ], the term φ k ′ h +1 appears in the summand with

k &gt; k ′ if and only if ( x k h , s k h ) = ( x k ′ h , s k ′ h ). The first time it appears we have n k h = n k ′ h +1, the second time it appears we have n k h = n k ′ h +2, and so on. Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the final inequality uses ∑ ∞ t = i α i t = 1+ 1 H from Lemma 4.1.c. Plugging these back into (4.6), we have:

where the final inequality uses φ k h +1 ≤ δ k h +1 (owing to the fact that V /star ≥ V π k ). Recursing the result for h = 1 , 2 , . . . , H , and using the fact δ K H +1 ≡ 0, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, by the pigeonhole principle, for any h ∈ [ H ]:

where inequality ① is true because ∑ x,a N K h ( x, a ) = K and the left-hand side of ① is maximized when N K h ( x, a ) = K/SA for all x, a . Also, by the AzumaHoeffding inequality, with probability 1 -p , we have:

This establishes ∑ K k =1 δ k 1 ≤ O ( H 2 SA + √ H 4 SATι ) . We note that when T ≥ √ H 4 SATι , we have √ H 4 SATι ≥ H 2 SA , and when T ≤ √ H 4 SATι , we have ∑ K k =1 δ k 1 ≤ HK = T ≤ √ H 4 SATι . Therefore, we can remove the H 2 SA term in the regret upper bound.

<!-- formula-not-decoded -->

In sum, we have ∑ K k =1 δ k 1 ≤ O ( H 2 SA + √ H 4 SATι ) , with probability at least 1 -2 p . Rescaling p to p/ 2 finishes the proof. /square

## Acknowledgements

We thank Nan Jiang, Sham M. Kakade, Greg Yang and Chicheng Zhang for valuable discussions. This work was supported in part by the DARPA program on Lifelong Learning Machines.

## References

- [1] Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In NIPS , pages 1184-1194, 2017.

- [2] Mohammad Gheshlaghi Azar, Remi Munos, Mohammad Ghavamzadeh, and Hilbert J Kappen. Speedy q-learning. In Proceedings of the 24th International Conference on Neural Information Processing Systems , pages 2411-2419. Curran Associates Inc., 2011.
- [3] Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J. Kappen. On the sample complexity of reinforcement learning with a generative model. In ICML , 2012.
- [4] Mohammad Gheshlaghi Azar, R´ emi Munos, and Hilbert J. Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning , 91(3):325-349, 2013.
- [5] Mohammad Gheshlaghi Azar, Ian Osband, and R´ emi Munos. Minimax regret bounds for reinforcement learning. In ICML , pages 263-272, 2017.
- [6] S´ ebastien Bubeck, Nicolo Cesa-Bianchi, et al. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends R © in Machine Learning , 5(1):1-122, 2012.
- [7] Marc Deisenroth and Carl E Rasmussen. Pilco: A model-based and data-efficient approach to policy search. In Proceedings of the 28th International Conference on machine learning (ICML-11) , pages 465-472, 2011.
- [8] Eyal Even-Dar and Yishay Mansour. Learning rates for q-learning. Journal of Machine Learning Research , 5(Dec):1-25, 2003.
- [9] Maryam Fazel, Rong Ge, Sham M Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for linearized control problems. arXiv preprint arXiv:1801.05039 , 2018.
- [10] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11:1563-1600, 2010.
- [11] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. arXiv preprint arXiv:1610.09512 , 2016.
- [12] Sham Kakade, Mengdi Wang, and Lin F Yang. Variance reduction methods for sublinear reinforcement learning. ArXiv e-prints , abs/1802.09184, April 2018.
- [13] Sham M. Kakade. On the sample complexity of reinforcement learning . PhD thesis, University of London London, England, 2003.
- [14] Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning , 49(2-3):209-232, 2002.
- [15] Sven Koenig and Reid G Simmons. Complexity analysis of real-time reinforcement learning. In AAAI , pages 99-105, 1993.
- [16] Tor Lattimore and Marcus Hutter. PAC bounds for discounted MDPs. In ALT , pages 320-334, 2012.
- [17] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.
- [18] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning , pages 1928-1937, 2016.
- [19] Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. arXiv preprint arXiv:1708.02596 , 2017.

- [20] Ian Osband and Benjamin Van Roy. On lower bounds for regret in reinforcement learning. ArXiv e-prints , abs/1608.02732, April 2016.
- [21] Vitchyr Pong, Shixiang Gu, Murtaza Dalal, and Sergey Levine. Temporal difference models: Model-free deep rl for model-based control. arXiv preprint arXiv:1802.09081 , 2018.
- [22] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning , pages 18891897, 2015.
- [23] Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM, 2018.
- [24] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature , 529 (7587):484-489, 2016.
- [25] Alexander L Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L Littman. PAC model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888. ACM, 2006.
- [26] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press Cambridge, 1998.
- [27] Christopher John Cornish Hellaby Watkins. Learning from delayed rewards . PhD thesis, King's College, Cambridge, 1989.

## Appendix

## A Explanation for Q-Learning with ε -Greedy

We recall a construction of a hard instance for Q-learning, known as a 'combination lock,' and tracing back at least to Koenig and Simmons [15]. In our context of our episodic MDP, this instance corresponds to the following MDP.

Consider a special state s /star ∈ S where the adversary always picks x 1 = s /star . For steps h = 1 , 2 , . . . , H/ 2, there is one special action a /star ∈ A where the distribution P h ( ·| s /star , a /star ) is a singleton and always leads to a next state x h +1 = s /star . For any other state s ∈ S \ { s /star } , or any other action a ∈ A \ { a /star } , the distribution P h ( ·| s, a ) is uniform over S \ { s /star } . For steps h = H/ 2 + 1 , . . . , H , P h ( ·| s, a ) is always a singleton and leads to the next state x h +1 = s . Finally, the reward function r h ( s, a ) = 0 for all s, a, h , except when s = s /star and h &gt; H/ 2, we have r H ( s /star , a /star ) = 1. It is clear that the optimal policy gives reward H/ 2 (by always selecting action a /star ).

For this MDP, for the Q-learning algorithm (or its Sarsa variant) with zero initialization, unless the algorithm picks a path with prefix ( x 1 , a 1 , x 2 , a 2 , . . . , x H/ 2 , a H/ 2 ) = ( s /star , a /star , . . . , s /star , a /star ), the reward value of the path is always zero and thus the algorithm will not change Q h ( s, a ) for any s, a, h . In other words, all Q values remain at zero until the first time ( s /star , a /star , . . . , s /star , a /star ) is visited. Unfortunately, this can happen with probability at most A -H/ 2 , and therefore the algorithm must suffer H/ 2 regret per round unless K ≥ Ω( A H/ 2 ).

## B Proof of Lemma 4.1

In this section, we derive three important properties implied by our choice of the learning rate. Recall the notation from (3.1) and (4.2):

<!-- formula-not-decoded -->

Lemma 4.1. The following properties hold for α i t :

- (a) 1 √ t ≤ ∑ t i =1 α i t √ i ≤ 2 √ t for every t ≥ 1 .
- (c) ∑ ∞ t = i α i t = 1 + 1 H for every i ≥ 1 .
- (b) max i ∈ [ t ] α i t ≤ 2 H t and ∑ t i =1 ( α i t ) 2 ≤ 2 H t for every t ≥ 1 .

Proof of Lemma 4.1.

- (a) The proof is by induction on t . For the base case t = 1 we have ∑ t i =1 α i t √ i = α 1 1 = 1 so the statement holds. For t ≥ 2, by the relationship α i t = (1 -α t ) α i t -1 for i = 1 , 2 , . . . , t -1 we have:

<!-- formula-not-decoded -->

On the one hand, by induction we have:

<!-- formula-not-decoded -->

On the other hand, by induction we have:

<!-- formula-not-decoded -->

where the final inequality holds because H ≥ 1.

- (b) We have:

<!-- formula-not-decoded -->

- (c) We first note the following identity, which holds for all positive integers n and k with n ≥ k :

Therefore, we have proved max i ∈ [ t ] α i t ≤ 2 H/t . The second inequality, ∑ t i =1 ( α i t ) 2 ≤ 2 H/t , follows directly since ∑ t i =1 ( α i t ) 2 ≤ [max i ∈ [ t ] α i t ] · ∑ t i =1 α i t and ∑ t i =1 α i t = 1.

<!-- formula-not-decoded -->

To verify (B.1), we write the terms of its right-hand side as x 0 = 1 , x 1 = n -k n +1 , . . . . It is easy to verify by induction that n k -∑ t i =0 x i = n -k k ∏ t i =1 n -k + i n + i . This means lim t →∞ n k -∑ t i =0 x i = 0

and this proves that (B.1) holds. Now, using (B.1) with n = i + H and k = H , we have:

<!-- formula-not-decoded -->

## C Proof for Q-learning with UCB-Bernstein

In this section, we prove Theorem 2.

Notation. In addition to the notation of Section 4, we define a variance operator V h :

We also consider an empirical version of variance that can be computed by the algorithm: when ( x, a ) was taken at step h for t times at k 1 , · · · , k t episodes respectively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this section, we choose two constants c 1 , c 2 &gt; 0 and define and accordingly,

<!-- formula-not-decoded -->

It is easy to verify that β t = 2 ∑ t i =1 α i t b i for every t ≥ 1. We include in Algorithm 2 the efficient implementation for calculating b t ( x, a, h ) in O (1) time per time step. Now we restate Theorem 2.

Theorem 2 (Bernstein, restated) . There exist absolute constants c 1 , c 2 &gt; 0 such that, for any p ∈ (0 , 1) , if we choose b t according to (C.3), then with probability 1 -p , the total regret of Qlearning with UCB-Bernstein (see Algorithm 2) is at most O ( √ H 3 SATι + √ H 9 S 3 A 3 · ι 2 ) .

## C.1 Proof

We first note that the following recursion, obtained in the proof for the Hoeffding case (see Lemma 4.2), still holds here:

Lemma C.1 (recursion on Q) . For any ( x, a, h ) ∈ S×A× [ H ] and episode k ∈ [ K ] , let t = N k h ( x, a ) and suppose ( x, a ) was previously taken at step h of episodes k 1 , . . . , k t &lt; k , then

<!-- formula-not-decoded -->

Parallel to the Hoeffding case, we aim at proving an equivalent version of Lemma 4.3 that shows that Q k -Q /star is (1) nonnegative and (2) bounded from above. However, unlike the Hoeffding case, this new proof becomes very delicate.

## Algorithm 2 Q-learning with UCB-Bernstein

```
1: for all ( x, a, h ) ∈ S × A × [ H ] do 2: Q h ( x, a ) ← H ; N h ( x, a ) ← 0; µ h ( x, a ) ← 0; σ h ( x, a ) ← 0; β 0 ( x, a, h ) ← 0. 3: for episode k = 1 , . . . , K do 4: receive x 1 . 5: for step h = 1 , . . . , H do 6: Take action a h ← argmax a ′ Q h ( x h , a ′ ), and observe x h +1 . 7: t = N h ( x h , a h ) ← N h ( x h , a h ) + 1. 8: µ h ( x h , a h ) ← µ h ( x h , a h ) + V h +1 ( x h +1 ). 9: σ h ( x h , a h ) ← σ h ( x h , a h ) + ( V h +1 ( x h +1 ) ) 2 . 10: β t ( x h , a h , h ) ← min { c 1 (√ H t σ h ( x h ,a h ) -( µ h ( x h ,a h )) 2 t + H ) ι + √ H 7 SA · ι t ) , c 2 √ H 3 ι t } . 11: b t ← β t ( x h ,a h ,h ) -(1 -α t ) β t -1 ( x h ,a h ,h ) 2 α t . 12: Q h ( x h , a h ) ← (1 -α t ) Q h ( x h , a h ) + α t [ r h ( x h , a h ) + V h +1 ( x h +1 ) + b t ]. 13: V h ( x h ) ← min { H, max a ′ ∈A Q h ( x h , a ′ ) } .
```

We first provide a coarse upper bound on Q k -Q /star that does not assert whether Q k -Q /star is nonnegative or not. This coarse upper bound only makes use of the fact that β t is at most O ( √ H 3 ι/t ), which was precisely how we have chosen β t in the Hoeffding case and in Lemma 4.3.

<!-- formula-not-decoded -->

Lemma C.2 (coarse bound on Q k -Q /star ) . There exists absolute constant c 2 &gt; 0 such that, if β t ( x, a, h ) ≤ c 2 √ H 3 ι t in (C.2), then, with probability at least 1 -p , the following holds

<!-- formula-not-decoded -->

where t = N k h ( x, a ) and k 1 , . . . , k t &lt; k are the episodes in which ( x, a ) was taken at step h .

Proof of Lemma C.2. The result follows from Lemma C.1 and the proof of Lemma 4.3. /square

In order to apply the Bernstein concentration inequality to the recursive formula in Lemma C.1, we need to estimate the variance of V /star . Unfortunately, V /star is unknown as its variance. At the k th episode, we are only able to compute the 'empirical' version of the variance using V k , which is W t as defined in (C.1).

Our next lemma shows that, if Q k ′ -Q /star is nonnegative for all episodes k ′ &lt; k , the variance of V /star (i.e., V h V /star h +1 ( x, a )) and the 'empirical' variance of V k are sufficiently close.

Lemma C.3. There exists an absolute constant c &gt; 0 such that for any p ∈ (0 , 1) and k ∈ [ K ] , with probability at least 1 -p/K , if then for all ( x, a, h ) ∈ S × A × [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma C.3. For each fixed ( x, a, h ) ∈ S × A × [ H ], let us denote k 0 = 0, and:

<!-- formula-not-decoded -->

That is, k i is the episode if which ( x, a ) was taken at step h for the i th time, and it is clearly a stopping time. Let F i be the σ -field generated by all the random variables until episode k i , step h . We also denote t = N k h ( x, a ).

<!-- formula-not-decoded -->

To bridge the gap between V h V /star h +1 ( x, a ) and W t ( x, a, h ), we consider following four quantities:

We shall bound the difference | P 1 -P 4 | by | P 1 -P 2 | + | P 2 -P 3 | + | P 3 -P 4 | via the triangle inequality.

Bounding | P 1 -P 2 | : We notice that for any fixed τ ∈ [ k ], by the Azuma-Hoeffding inequality, there exists a sufficiently large constant c &gt; 0 such that, with probability at least 1 -p/ (2 SAT ):

since LHS is a martingale sequence with respect to the filtration {F i } . Because Eq. (C.5) holds for all fixed τ ∈ [ k ] uniformly, it also holds for τ = t = N k h ( x, a ) ≤ k which is a random variable. Also note I [ k i ≤ k ] = 1 for all i ≤ N k h ( x, a ). Therefore, we can conclude | P 1 -P 2 | ≤ cH 2 √ ι/t .

<!-- formula-not-decoded -->

Bounding | P 2 -P 3 | : We calculate

Again, for any fixed τ ∈ [ k ], by the Azuma-Hoeffding inequality, with probability 1 -p/ (2 SAT ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding | P 3 -P 4 | : We calculate that

By the same argument as above, we also know that Eq. (C.5) holds for the random variable τ = t = N k h ( x, a ) ≤ k , which implies | P 2 -P 3 | ≤ 2 cH 2 √ ι/t .

<!-- formula-not-decoded -->

where the last inequality uses V k ′ h +1 ( x ) ≥ V /star h +1 ( x ) for all x ∈ S and k ′ &lt; k , which follows from our assumption ( Q k ′ h +1 -Q /star h +1 )( x, a ) ≥ 0 for all k ′ &lt; k .

We apply Lemma C.7 (see Section C.3 later) with a weight vector w such that w k i = 1 t for all i ∈ [ t ], but w k ′ = 0 for all k ′ /negationslash∈ { k 1 , . . . , k t } (so ‖ w ‖ 1 = 1 and ‖ w ‖ ∞ = 1 /t ). This tells us that

Finally, by the triangle inequality ∣ ∣ [ V h V /star h +1 -W k h ]( x, a ) ∣ ∣ ≤ | P 1 -P 2 | + | P 2 -P 3 | + | P 3 -P 4 | , and a union bound over ( x, a, h ) ∈ S × A × [ H ], we finish the proof. /square

<!-- formula-not-decoded -->

Now, equipped with Lemma C.2 and Lemma C.3, we can use induction and an Azuma-Bernstein concentration argument to prove that Q k -Q /star is nonnegative and upper bounded by β . This gives an analog of Lemma 4.3 that we state here.

Lemma C.4 (fine bound on Q k -Q /star ) . For every p ∈ (0 , 1) , there exists an absolute constant c 1 , c 2 &gt; 0 such that, under the choice of β t ( x, a, h ) in (C.2), with probability at least 1 -2 p , the following holds simultaneously for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ] :

where t = N k h ( x, a ) and k 1 , . . . , k t &lt; k are the episodes in which ( x, a ) was taken at step h .

<!-- formula-not-decoded -->

Proof of Lemma C.4. Wefirst choose c 2 &gt; 0 large enough so that Lemma C.2 holds with probability at least 1 -p .

<!-- formula-not-decoded -->

For each fixed ( x, a, h ) ∈ S × A × [ H ], let us denote k 0 = 0, and:

By the Azuma-Bernstein inequality, with probability at least 1 -p/ ( SAT ), we have for all τ ∈ [ K ]:

<!-- formula-not-decoded -->

where the last inequality is by Lemma 4.1.b. Since the inequality (C.8) holds for all fixed τ ∈ [ K ] uniformly, it also holds for the random variable τ = t = N k h ( x, a ) ≤ K . By a union bound, with probability at least 1 -p , we have that for all ( x, a, h, k ) ∈ S × A × [ H ] × [ K ]

We are now ready to prove (C.7). We do so by induction over k ∈ [ K ]. Clearly, the statement is true for k = 1, so in the rest of the proof we assume (C.7) holds for all k ′ &lt; k . We denote by k 1 , k 2 , . . . , k t &lt; k all indices of previous episodes where ( x, a ) is taken at step h . By Lemma C.3, with probability 1 -p/K , we have for all ( x, a, h ) ∈ S × A × [ H ]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, putting this into (C.9), we have

∣ ∣ where inequality ① uses √ H 7 SAι t ≤ H + H 6 SAι t , and inequality ② is due to our choice of β t in (C.2) and the sufficiently large choice of c 1 &gt; 0.

<!-- formula-not-decoded -->

Finally, applying the above inequality to Lemma C.1, we have for all ( x, a, h ) ∈ S × A × [ H ]

This proves that (C.7) holds for k with probability at least 1 -p/K . By induction, we know (C.7) holds for all k ∈ [ K ] with probability at least 1 -p . Combining this with the 1 -p probability event for (C.9), we finish the proof that Lemma C.4 holds with probability at least 1 -2 p . /square

<!-- formula-not-decoded -->

As mentioned in Section 3, the key reason why a Bernstein approach can improve by a factor of √ H is that, although the value function at each step is at most H , the 'total variance of the value function' for an entire episode is at most O ( H 2 ). Or more simply, the total variance for all steps is at most O ( HT ). This is captured directly in the following lemma.

Lemma C.5. There exists an absolute constant c , such that with probability at least 1 -p :

<!-- formula-not-decoded -->

Proof of Lemma C.5. First, we note for any fixed policy π and initial state x 1 , suppose ( x 2 , · · · , x h ) is a sequence generated by following policy π starting at x 1 , then where equality ① is because V π H +1 = 0, and equality ② uses the independence due to the Markov property. Therefore, letting F k -1 be the σ -field generated by all the random variables over the first k -1 episodes, at the k th episode we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ Also, note that | X k | ≤ H 3 and Var[ X k | F k -1 ] ≤ H 3 E [ X k | F k -1 ] ≤ H 5 . Therefore, by an AzumaBernstein inequality on X 1 + · · · + X K with respect to filtration {F k } k ≥ 0 , we have with probability at least 1 -p , where the last step is by ab ≤ a 2 + b 2 .

/square

Our last lemma shows that the 'empirical' variance of V k (i.e., W t ( x, a, h )) is also upper bounded by the variance V h V π k h +1 ( x, a ) (which appeared in Lemma C.5) plus some small terms.

Lemma C.6. There exist absolute constants c 1 , c 2 , c &gt; 0 such that, letting ( x, a ) = ( x k h , a k h ) and t = n k h = N k h ( x, a ) , we have that for all ( k, h ) ∈ [ K ] × [ H ] , with probability at least 1 -4 p ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma C.6. We first assume that Lemma C.4 holds (which happens with probability at least 1 -2 p ) and Lemma C.2 holds (which happens with probability at least 1 -p ). As a consequence, with probability at least 1 -p , Lemma C.3 also holds for all k ∈ [ K ]. By the triangle inequality, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 2

We are now ready to prove Theorem 2. Again, the proof decomposes the regret in a recursive form, and carefully controls the error propagation via repeated usage of Lemma C.4 and Lemma C.6.

Proof of Theorem 2. We first assume that Lemma C.5 holds (which happens with probability at least 1 -4 p ) and Lemma C.6 holds (which happens with probability at least 1 -p ).

By the same argument as in the proof of Theorem 1 (in particular, inequality (4.7)) we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ξ k h +1 := [( P h -ˆ P k h )( V /star h +1 -V k h +1 )]( x k h , a k h ) and δ k h +1 := ( V /star h +1 -V k h +1 )( x k h +1 ). As a result, for any h ∈ H , by recursing the above formula for h, h +1 , . . . , H , we have:

By the Azuma-Hoeffding inequality, with probability 1 -p , we have:

Also, recall β t ( x, a, h ) ≤ c √ H 3 ι/t so ∑ K k =1 β n k h ≤ O ( √ H 2 SATι ) according to (4.8). Putting these into (C.11), we derive that ∑ K k =1 δ k h ≤ O ( SAH 2 + √ H 4 SATι ) . Note when T ≥ √ H 4 SATι , we have √ H 4 SATι ≥ H 2 SA ; when T ≤ √ H 4 SATι , we have ∑ K k =1 δ k h ≤ HK = T ≤ √ H 4 SATι . Therefore, we can simply write

<!-- formula-not-decoded -->

By our choice of β t , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The summation of the second term in (C.14) is upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

because 1 + 1 2 + 1 3 + · · · ≤ ι . The summation of the first term in (C.14) can be upper bounded by

We calculate

Here, inequality ① uses Lemma C.6; inequality ② uses ∑ K k =1 ( n k h ) -1 ≤ SAι and ∑ K k =1 ( √ n k h ) -1 / 2 ≤ O ( √ KSA ); inequality ③ uses Lemma C.5; and inequality ④ uses (C.12) and (C.13).

<!-- formula-not-decoded -->

Putting (C.16) and (C.15) back to (C.14), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, putting this and (C.12) back to (C.11), we finish the proof that with probability at least 1 -6 p , for every h ∈ [ H ]

Since we also have Regret( K ) ≤ ∑ K k =1 δ k 1 as in the proof of Theorem 1, rescaling p to p/ 6 finishes the proof. /square

## C.3 Proof of Auxiliary Lemma

The next lemma shows how the weighted sum over ( V k h -V /star h )( x k h ) is upper bounded by the infinity norm and the one-norm of the weights w . This lemma provides the key to prove Lemma C.3.

Lemma C.7. Suppose (C.4) in Lemma C.2 holds. For any h ∈ [ H ] , let φ k h := ( V k h -V /star h )( x k h ) , and

letting w = ( w 1 , . . . , w k ) be a nonnegative weight vector, we have:

<!-- formula-not-decoded -->

where φ k h := ( V k h -V /star h )( x k h ) .

Proof of Lemma C.7. For any fixed ( k, h ) ∈ [ K ] × [ H ], let t = N k h ( x k h , a k h ), and suppose ( x k h , a k h ) was previously taken at step h of episodes k 1 , . . . , k t &lt; k . We then have, for some absolute constant c :

Here, inequality ① holds from V k h ( x k h ) ≤ max a ′ ∈A Q k h ( x k h , a ′ ) = Q k h ( x k h , a k h ) and the Bellman optimality equation V /star h ( x k h ) = max a ′ ∈A Q /star h ( x k h , a ′ ) ≥ Q /star h ( x k h , a k h ). Inequality ② holds by the assumption that (C.4) in Lemma C.2 holds.

<!-- formula-not-decoded -->

Next, let us compute the summation ∑ K k =1 w k δ k h . Denoting n k h = N k h ( x k h , a k h ), we have:

Above,

<!-- formula-not-decoded -->

- Equality ① is by reordering the indices k ∈ [ K ] so that the ones with the same ( x, a ) = ( x k h , a k h ) are grouped together; and we denote by k i ( x, a ) = k where k is the i th episode where ( x, a ) is taken at step h .
- Inequality ② is because ∑ x,a ∑ N K h ( x,a ) i =1 w k i ( x,a ) = ‖ w ‖ 1 . Therefore, the left-hand side of ② is maximized when the weights are distributed to those indices i that have smaller values:

<!-- formula-not-decoded -->

To bound the second term in (C.18), which is

<!-- formula-not-decoded -->

we regroup the summands in (C.21) in a different way. For every k ′ ∈ [ K ], we group all terms φ k ′ h +1 that appear in the inner summand of (C.21)-denoting their total weight by w ′ k ′ -and write:

<!-- formula-not-decoded -->

We make two key observations

- We have ‖ w ′ ‖ 1 ≤ ‖ w ‖ 1 because ∑ t i =1 α i t ≤ 1.

- For every k ′ ∈ [ K ], we note that the term φ k ′ h +1 only appears on the left-hand side of (C.22) in episode k ≥ k ′ , where ( x k h , s k h ) = ( x k ′ h , s k ′ h ). Suppose it appears in episodes k ′ 1 , k ′ 2 , . . . . Then, letting τ = n k ′ h , we have corresponding weight is w k ′ α τ τ , w k ′ 1 α τ τ +1 , w k ′ 2 α τ τ +2 · · · . Therefore, the total weight satisfies

<!-- formula-not-decoded -->

where the final inequality uses ∑ ∞ t = i α i t = 1 + 1 H from Lemma 4.1.c. Plugging (C.19), (C.20), and (C.22) back into (C.18), we have:

<!-- formula-not-decoded -->

with ‖ w ′ ‖ ∞ ≤ (1 + 1 H ) ‖ w ‖ ∞ and ‖ w ′ ‖ 1 ≤ ‖ w ‖ ∞ . Recursing this for h, h +1 , . . . , H , we conclude that

<!-- formula-not-decoded -->

## D Proof of Lower Bound

Recall that Jaksch et al. [10] showed that for any algorithm, there is an MDP with diameter D , S states and A actions, such that the algorithm's regret must be at least Ω( √ DSAT ). The natural analogous notion of the diameter in the episodic setting is H , and thus this suggests a lower bound in Ω( √ HSAT ), as presented in [5, 20].

We show that, in our episodic setting of this paper, one can obtain a stronger lower bound:

Theorem 3. For any algorithm there exists an H -episodic MDP with S states and A actions such that for any T , the algorithm's regret is Ω( H √ SAT ) .

This result seemingly contradicts the O ( √ HSAT ) regret bound of Azar et al. [5]. There is no contradiction, however, because Azar et al. [5] assumes that the transition matrix P h is the same at each step h ∈ [ H ]. On the contrary, in this paper we consider the more general setting where the transition matrices P 1 , . . . , P H are distinct for each step. Our setting can be viewed as a special case of the non-episodic MDP studied by Jaksch et al. [10], obtained by augmenting the state space to S ′ = S × [ H ].

Rather than providing a formal proof of Theorem 3 we give the intuition behind the construction and its analysis. The formalization itself is an easy exercise following well-known lower-bound techniques from the multi-armed bandit literature; see, e.g., [6]. For the sake of simplicity, we consider A = 2 and S = 2 (again the generalization to arbitrary A and S is routine).

We start by recalling the construction from Jaksch et al. [10], which we will refer to as the 'JAO MDP.' The reward does not depend on actions: state 1 always has reward 1 and state 0 always has reward 0. From state 1, any action takes the agent to state 0 with probability δ , and to state 1 with probability 1 -δ . In state 0, there is one action a /star takes the agent to state 1 with probability δ + ε , and the other action a takes the agent to 1 with probability δ . A standard Markov chain exercise shows that the stationary distribution of the optimal policy (that is, the one that in state

0 takes action a /star ) has a probability of being in state 1 of

<!-- formula-not-decoded -->

In contrast, acting sub-optimally (that is, taking action a in state 0) leads to a uniform distribution over the two states, or equivalently a regret per time step of order ε/δ . Moreover, in order to identify the two actions a, a /star (each with probability δ and δ + ε ), the number of observations in state 0 needs to be at least Ω( δ/ε 2 ). Thus, taking the latter quantity to be T , one obtains the following lower bound on total regret:

In the JAO MDP, the diameter is D = Θ(1 /δ ). This proves the √ DT lower bound from Jaksch et al. [10].

<!-- formula-not-decoded -->

The natural analogue of the JAO MDP for the episodic setting is to put the JAO MDP in 'series' for H steps (in other words, one takes H steps in the JAO MDP and then restarts, say starting in state 0). The main difference with the non-episodic version is that, in H steps, one may not have time to mix , i.e., to reach the stationary distribution over the two states. Using standard theory of Markov chains, one can show that the optimal policy on this episodic MDP has a mixing time of Θ(1 /δ ). By choosing H to be slightly larger than Θ(1 /δ ), we have a sufficient number of steps (in each episode) to mix, and thus the previous non-episodic argument remains valid for the episodic case. This leads to a lower bound Ω( √ HT ) for the episodic case, as illustrated by [5, 20].

Finally, recall that in our episodic setting, the transition matrices P 1 , . . . , P H may not necessarily be the same. Therefore, we can further strengthen this lower bound to Ω( H √ T ) in the following way.

Let us use H distinct JAO MDPs, each with a different optimal action a /star h , when putting them in series. In other words, for at least half of the steps h ∈ H , one has to identify the correct action a /star h for that specific step. (If not, the per-iteration regret will again be Ω( ε/δ ).) However the number of observations in that specific step h is only T/H , and thus one now needs to take T/H = O ( δ/ε 2 ) (instead of T = Ω( δ/ε 2 ) previously). This gives the claimed Ω ( H √ T ) lower bound.