## Model-Free Non-Stationary RL: Near-Optimal Regret and Applications in Multi-Agent RL and Inventory Control

Weichao Mao

Department of Electrical and Computer Engineering &amp; Coordinated Science Laboratory, University of Illinois Urbana-Champaign, Urbana, IL 61801, weichao2@illinois.edu

## Kaiqing Zhang

Laboratory for Information &amp; Decision Systems, Massachusetts Institute of Technology, Cambridge, MA 02139, kaiqing@mit.edu

## Ruihao Zhu

Cornell SC Johnson College of Business, Ithaca, NY 14853, ruihao.zhu@cornell.edu

## David Simchi-Levi

Institute for Data, Systems, and Society, Department of Civil and Environmental Engineering, Operations Research Center, Massachusetts Institute of Technology, Cambridge, MA 02139, dslevi@mit.edu

## Tamer Ba¸ sar

Department of Electrical and Computer Engineering &amp; Coordinated Science Laboratory, University of Illinois Urbana-Champaign, Urbana, IL 61801, basar1@illinois.edu

We consider model-free reinforcement learning (RL) in non-stationary Markov decision processes. Both the reward functions and the state transition functions are allowed to vary arbitrarily over time as long as their cumulative variations do not exceed certain variation budgets. We propose Restarted Q-Learning with Upper Confidence Bounds (RestartQ-UCB), the first model-free algorithm for non-stationary RL, and show that it outperforms existing solutions in terms of dynamic regret. Specifically, RestartQ-UCB with Freedman-type bonus terms achieves a dynamic regret bound of ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ), where S and A are the numbers of states and actions, respectively, ∆ &gt; 0 is the variation budget, H is the number of time steps per episode, and T is the total number of time steps. We further present a parameter-free algorithm named Double-Restart Q-UCB that does not require prior knowledge of the variation budget. We show that our algorithms are nearly optimal by establishing an information-theoretical lower bound of Ω( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 ), the first lower bound in non-stationary RL. Numerical experiments validate the advantages of RestartQ-UCB in terms of both cumulative rewards and computational efficiency. We demonstrate the power of our results in examples of multi-agent RL and inventory control across related products.

Key words : reinforcement learning, data-driven decision making, non-stationarity, multi-agent learning, inventory control

## 1. Introduction

Reinforcement learning (RL) focuses on the class of problems where an agent maximizes its cumulative reward through sequential interactions with an initially unknown but fixed environment, usually modeled by a Markov Decision Process (MDP). In classical RL problems, the state transition functions and the reward functions are assumed to be time-invariant, i.e., stationary. However, stationary models cannot capture the time-varying environments in a wide range of sequential decision-making problems, such as online advertisement auctions (Cai et al. 2017, Lu et al. 2019), dynamic pricing (Chawla et al. 2016), traffic management (Chen et al. 2020a), healthcare operations (Shortreed et al. 2011), multi-agent RL (Littman 1994), and inventory control (Agrawal and Jia 2019, Cheung et al. 2020b). Among the many intriguing applications, in the following, we specifically elaborate on two research areas, namely multi-agent RL and inventory control, that can significantly benefit from progresses on non-stationary RL. In Appendix A, we further discuss the potential application scenarios of non-stationary RL in other important areas, such as sequential transfer RL and multi-task RL.

- Multi-agent RL: In multi-agent RL, a set of agents either collaborate or compete by taking actions in a shared environment. This commonly occurs in many operational scenarios when multiple decision-makers interact with each other, such as ads auctions (Balseiro and Gur 2019) and dynamic pricing (Birge et al. 2021). In such scenarios, each agent faces a non-stationary environment, especially when the agents learn and update their policies simultaneously, as the actions of the other agents can alter the environment. We discuss this connection with more details in Section 8 through a concrete example, where we show that our non-stationary RL solution can be readily applied to a multi-agent RL problem against a slowly-changing opponent.
- Inventory control across related but different products : In conventional inventory control (Huh and Rusmevichientong 2009, Zhang et al. 2019, Agrawal and Jia 2019), the retailer typically focuses on managing the stock level of a single product. Nevertheless, the sequential launch of new related products (e.g., the line of iPhone) provides us with the opportunity to leverage experience from past products to inform inventory management for future products. In Section 9, we discuss how one can apply our non-stationary RL solutions to guide the inventory management not only for a single product but also across a sequence of related, but different products.

RL in a non-stationary MDP is highly non-trivial due to the following challenges. First, similar to stationary RL, the agent faces the exploration vs. exploitation dilemma: it needs to explore the uncertain environment efficiently while maximizing its rewards along the way. In Jaksch et al. (2010), the authors proposed to leverage the 'optimism in the face of uncertain' principle to guide exploration. Another challenge, which is unique to non-stationary RL, is the trade-off between remembering and forgetting . On the one hand, since the underlying MDP varies over time, data

Table 1 Dynamic regret comparisons for RL in non-stationary MDPs. S and A are the numbers of states and actions, L is the number of abrupt changes, D is the maximum diameter, d is the dimension of the feature space for linear MDPs, H is the number of steps per episode, and T is the total number of steps. All upper bounds listed in the table are high-probability results that hold with probability at least 1 -δ for some δ ∈ (0 , 1) , and ˜ O ( · ) suppresses logarithmic dependence on S,A,T and 1 δ . Gray cells denote results from this paper.

| Setting        | Algorithm                                                                                        | Regret                                                                                                                                                                                     | Model- Free   | Comment                                                                              |
|----------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------|
| Undis- counted | Jaksch et al. (2010) Gajane et al. (2018) Ortner et al. (2019) Cheung et al. (2020b) Lower bound | ˜ O ( S 1 1 A 1 2 L 1 3 D 1 1 T 2 3 ) ˜ O ( S 2 3 A 1 3 L 1 3 D 2 3 T 2 3 ) ˜ O ( S 2 3 A 1 2 ∆ 1 3 D 1 1 T 2 3 ) ˜ O ( S 2 3 A 1 2 ∆ 1 4 D 1 1 T 3 4 ) Ω( S 1 3 A 1 3 ∆ 1 3 D 2 3 T 2 3 ) | 7 7 7 7       | only abrupt changes only abrupt changes requires local variations does not require ∆ |
| Episodic       | Domingues et al. (2021) RestartQ-UCB Double-Restart Q-UCB Lower bound                            | ˜ O ( S 1 1 A 1 2 ∆ 1 3 H 4 3 T 2 3 ) ˜ O ( S 1 3 A 1 3 ∆ 1 3 H 1 1 T 2 3 ) ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 + H 3 4 T 3 4 ) Ω( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 )                              | 7 3 3         | also metric spaces does not require ∆                                                |
| Linear MDPs    | Zhou et al. (2020a) Touati and Vincent (2020)                                                    | ˜ O ( d 4 3 ∆ 1 3 H 4 3 T 2 3 ) ˜ O ( d 5 4 ∆ 1 4 H 5 4 T 3 4 )                                                                                                                            | 3 3           |                                                                                      |

samples collected in prior interactions can become obsolete. In fact, it has been shown that a standard stationary RL algorithm might incur a linear regret if the non-stationarity is not handled properly (Ortner et al. 2019). On the other hand, the agent needs to extract a sufficient amount of information from historical data to inform future decision-making.

To resolve the aforementioned challenges, Ortner et al. (2019) and Cheung et al. (2020b) have proposed algorithms to guide learning in non-stationary MDPs. Although both model-based and model-free algorithms have been proposed for stationary RL, existing solutions for non-stationary RL are often built upon model-based methods. Nevertheless, it has been observed that model-based solutions often suffer from the following shortcomings:

- Time- and space-inefficiency: Model-based methods are in general more time- and spaceconsuming, and are less compatible with the design of modern deep RL architectures (Jin et al. 2018, Zhang et al. 2020).
- Inefficient exploration: In Cheung et al. (2020b,a), an example was given to show that under non-stationarity, the estimated model can incorrectly indicate that transitioning between states is very unlikely. This suggests that model-based methods, which try to estimate the latent model, might suffer 'The Perils of Drift' (Cheung et al. 2020b).

- Limited applicability: In an important application of nonstationary RL decentralized multi-agent RL, the agents cannot observe the actions taken by the other agents. This information structure precludes model-based methods, as the explicit estimation of the state transition functions is hardly possible without observing all the agents' actions.

These observations have thus motivated us to turn our attention to model-free methods, which, instead of maintaining estimates of the unknown underlying model, directly learn the Q-values.

Main Contributions. In this paper, we focus on the problem of designing model-free algorithms with near-optimal performances for non-stationary RL. Our contributions are as follows:

1. We introduce an algorithm named Restarted Q-Learning with Upper Confidence Bounds (RestartQ-UCB), which is the first model-free algorithm in the general setting of non-stationary RL. Our algorithm adopts a simple but effective restarting strategy (Jaksch et al. 2010, Besbes et al. 2014) that resets the memory of the agent according to a calculated schedule. The restarting strategy ensures that our algorithm only refers to the most up-to-date experience for decision-making. RestartQ-UCB also utilizes an extra optimism term (in addition to the standard Hoeffding/Freedman-based bonus) for exploration to counteract the non-stationarity of the MDP. This additional bonus term, depending on the local variation budgets (i.e., the environmental variation in each restarting interval), guarantees that our optimistic Q -value is still an upper bound of the optimal Q glyph[star] -value even when the environment changes. Our analysis shows that RestartQ-UCB achieves the lowest dynamic regret bound when compared to existing works in the literature;
2. We conduct simulations showing that RestartQ-UCB achieves highly competitive cumulative rewards against a state-of-the-art solution (Zhou et al. 2020a), while only taking 0 . 18% of its computation time;
3. We establish the first lower bounds in non-stationary RL, which suggest that our algorithm is optimal in all parameter dependences except for an H 1 3 factor, where H is the episode length;
4. To further showcase the flexibility and potential of non-stationary RL, we illustrate how it can be utilized to address the non-stationarity issue inherent in multi-agent RL. Specifically, we show that RestartQ-UCB can be readily applied to a multi-agent RL example against a slowly-changing opponent (Radanovic et al. 2019, Lee et al. 2020). The setting we consider is a more practical and general decentralized learning setting, which entails model-free solutions;
5. A preliminary version of this paper (Anonymous 2021) has appeared in the Proceedings of the 38th International Conference on Machine Learning (ICML 2021). In the current version, 1) We have proved a new result of the Freedman-based RestartQ-UCB that no longer requires knowledge of the local variation budget (Theorem 3); 2) We further show that our algorithm can easily remove the dependence on prior knowledge of the variation budget, a critical assumption

commonly made in the literature (Ortner et al. 2019, Zhou et al. 2020a). To do that, we propose a parameter-free algorithm that leverages a 'double restart' strategy to adaptively learn the variation budget; 3) In addition, we discuss the application of our non-stationary RL algorithm in inventory control. Specifically, we demonstrate how to implement our RestartQ-UCB algorithm for the problem of inventory control across related, but different products with time-varying demands; 4) We also provide detailed proofs that were missing in the conference version.

Related Works. Dynamic regret of non-stationary RL has been mostly studied using modelbased solutions. Jaksch et al. (2010) consider the setting where the MDP is allowed to change abruptly for L times. A sliding window approach is proposed in Gajane et al. (2018) under the same setting. Ortner et al. (2019) generalize the previous setting by allowing the MDP to vary either abruptly or gradually at every step, subject to a total variation budget. Cheung et al. (2020b) consider the same setting and introduce a Bandit-over-RL technique that adaptively tunes the algorithm without knowing the variation budget. Directly applying their method to our episodic setting will lead to a dynamic regret of ˜ O ( S 2 3 A 1 2 ∆ 1 4 HT 3 4 ) . Although it may be possible to further obtain an improved dependence on T , this is sub-optimal in terms of S and A . We remark that a recent (but later than ours) version of this paper develops a lower bound tailored to the infinite horizon undiscounted non-stationary RL, this is not directly applicable to our episodic non-stationary RL setting.

In a setting most similar to ours, Domingues et al. (2021) investigate non-stationary RL in the episodic setting, and propose a kernel-based approach when the state-action set forms a metric space. Their results can be reduced to an ˜ O ( SA 1 2 ∆ 1 3 H 4 3 T 2 3 ) regret in the tabular case. Fei et al. (2020) assume stationary transitions and adversarial full-information rewards, and their setting is not directly comparable with ours. Two concurrent works Zhou et al. (2020a) and Touati and Vincent (2020) consider non-stationary RL in linear MDPs, but their regret bounds, ˜ O ( S 4 3 A 4 3 ∆ 1 3 H 4 3 T 2 3 ) and ˜ O ( S 5 4 A 5 4 ∆ 1 4 H 5 4 T 3 4 ) when reduced to the tabular RL setting, respectively, are less competitive than ours. After an earlier version of this paper was made publicly available, Wei and Luo (2021) have proposed a black-box reduction procedure that turns an RL algorithm in a (nearly-)stationary environment to a non-stationary RL algorithm. In the episodic setting, Wei and Luo (2021) have achieved a strong dynamic regret bound of ˜ O ( S 1 3 A 1 3 ∆ 1 3 H 5 3 T 2 3 ) (with or without knowledge of the degree of non-stationarity). However, their regret bound has a worse dependence on H when compared to ours, and it has been pointed out in Wei and Luo (2021) that such a sub-optimality cannot be improved upon by using a Freedman-style confidence bound as we do. Their compelling theoretical guarantees also come at the cost of a rather sophisticated and memory-inefficient algorithmic design, which needs to maintain many instances of the stationary subroutine, and constantly switch among them. Interested readers are referred to Padakandla (2020)

for a comprehensive survey on RL in non-stationary environments. Table 1 compares our regret bounds with existing results that tackle similar settings as ours. It can be seen that our result is the first one that achieves the optimal dependence on S and A , and also establishes the tightest dependence on H/D and T among existing solutions in the literature, without relying on their assumptions.

Another related line of research studies online/adversarial MDPs (Yu and Mannor 2009, Neu et al. 2010, Arora et al. 2012, Yadkori et al. 2013, Dick et al. 2014, Wang et al. 2020, Lykouris et al. 2019, Jin et al. 2019), but they mostly only allow variations in reward functions, and use the static regret as a performance metric. In addition, RL with low switching cost (Bai et al. 2019) also shares a similar spirit as our restarting strategy since it also periodically forgets previous experiences. However, such algorithms do not address the non-stationarity of the environment, and their dynamic regret in terms of the variation budget is unclear.

Non-stationarity has also been considered in bandit problems (Besbes et al. 2019). Within different non-stationary multi-armed bandit (MAB) settings, various methods have been proposed, including decaying memory and sliding windows (Garivier and Moulines 2011, Keskin and Zeevi 2017), as well as restart-based strategies (Auer et al. 2002, Besbes et al. 2014, Allesiardo et al. 2017). These methods largely inspired later research on non-stationary RL. A more recent line of work developed methods that do not require prior knowledge of the variation budget (Karnin and Anava 2016, Cheung et al. 2019a) or the number of abrupt changes (Auer et al. 2019). Other related settings considered in the literature include Markovian bandits (Tekin and Liu 2010, Ma 2018, Zhou et al. 2020b), non-stationary contextual bandits (Luo et al. 2018, Chen et al. 2019), linear bandits (Cheung et al. 2019b, Zhao et al. 2020), continuous-armed bandits (Mao et al. 2020), and learning with seasonal patterns (Chen et al. 2020b).

Outline. The rest of the paper is organized as follows: In Sections 2, we introduce the mathematical model of our problem and necessary preliminaries. In Section 3, we present our RestartQ-UCB algorithm. A dynamic regret analysis of RestartQ-UCB is provided in Section 4. In Section 5, we further propose a parameter-free algorithm that does not require prior knowledge of the variation budget. In Section 6, we establish information-theoretical lower bounds. Simulation results are presented in Section 7. In Sections 8 and 9, we discuss the applications of our method to two important scenarios: multi-agent RL and inventory control, respectively. Finally, we conclude the main part of the paper in Section 10. Some supplementary material and proofs of all the results are included in eleven appendices at the end of the paper.

## 2. Preliminaries

Model: We consider an episodic RL setting where an agent interacts with a non-stationary MDP for M episodes, with each episode containing H steps. We use a pair of integers ( m,h ) as a time

index to denote the h -th step of the m -th episode. The environment can be denoted by a tuple ( S , A , H, P, r ), where S is the finite set of states with |S| = S , A is the finite set of actions with |A| = A , H is the number of steps in one episode, P = { P m h } m ∈ [ M ] ,h ∈ [ H ] is the set of transition kernels, and r = { r m h } m ∈ [ M ] ,h ∈ [ H ] is the set of mean reward functions. Specifically, when the agent takes action a m h ∈A in state s m h ∈S at the time ( m,h ), it will receive a random reward R m h ( s m h , a m h ) ∈ [0 , 1] with expected value r m h ( s m h , a m h ), and the environment transitions to a next state s m h +1 following the distribution P m h ( · | s m h , a m h ). It is worth emphasizing that the transition kernel and the mean reward function depend both on m and h , and hence the environment is non-stationary over time. The episode ends when s m H +1 is reached. We further denote T = MH as the total number of steps.

A deterministic policy π : [ M ] × [ H ] ×S →A is a mapping from the time index and state space to the action space, and we let π m h ( s ) denote the action chosen in state s at time ( m,h ). Define V m,π h : S → R to be the value function under policy π at time ( m,h ), i.e.,

<!-- formula-not-decoded -->

where s h ′ +1 ∼ P m h ′ ( · | s h ′ , a h ′ ). Accordingly, the state-action value function Q m,π h : S × A → R is defined as:

<!-- formula-not-decoded -->

For simplicity of notation, we let P m h V h +1 ( s, a ) def = E s ′ ∼ P h m ( ·| s,a ) [ V h +1 ( s ′ )]. Then, the Bellman equation gives V m,π h ( s ) = Q m,π h ( s, π m h ( s )) and Q m,π h ( s, a ) = ( r m h + P m h V m,π h +1 )( s, a ), and we also have V m,π H +1 ( s ) = 0 , ∀ s ∈ S by definition. Since the state space, the action space, and the length of each episode are all finite, there always exists an optimal policy π glyph[star] that gives the optimal value V m,glyph[star] h ( s ) def = V m,π glyph[star] h ( s ) = sup π V m,π h ( s ) , ∀ s ∈S , m ∈ [ M ] , h ∈ [ H ]. From the Bellman optimality equation, we have V m,glyph[star] h ( s ) = max a ∈A Q m,glyph[star] h ( s, a ), where Q m,glyph[star] h ( s, a ) def =( r m h + P m h V m,glyph[star] h +1 )( s, a ), and V m,glyph[star] H +1 ( s ) = 0 , ∀ s ∈S .

Dynamic Regret: The agent aims to maximize the cumulative expected reward over the entire M episodes, by adopting some policy π . We measure the optimality of the policy π in terms of its dynamic regret (Cheung et al. 2020b, Domingues et al. 2021), which compares the agent's policy with the optimal policy of each individual episode in hindsight:

<!-- formula-not-decoded -->

where the initial state s m 1 of each episode is chosen by an oblivious adversary (Zhang et al. 2020). Dynamic regret is a stronger measure than the standard (static) regret, which only considers the single policy that is optimal over all episodes combined.

## Algorithm 1: RestartQ-UCB (Hoeffding/Freedman)

<!-- formula-not-decoded -->

Variation: We measure the non-stationarity of the MDP in terms of its variation budget in the mean reward function and transition kernels:

<!-- formula-not-decoded -->

where ‖·‖ 1 is the L 1 -norm. Note that our definition of variation budgets only imposes restrictions on the summation of non-stationarity across two different episodes, and does not put any restriction on the difference between two consecutive steps in the same episode; that is, P m h ( · | s, a ) and P m h +1 ( · | s, a ) are allowed to be arbitrarily different. We further let ∆ = ∆ r +∆ p , and assume ∆ &gt; 0.

## 3. Algorithm: RestartQ-UCB

We present our algorithm Restarted Q-Learning with Hoeffding/Freedman Upper Confidence Bounds (RestartQ-UCB Hoeffding/Freedman) in Algorithm 1. For illustrative purposes, we start

with a simpler RestartQ-UCB algorithm with Hoeffding-style bonus terms, which only executes the pseudocode colored in black in Algorithm 1. Further incorporating the gray parts in Algorithm 1 leads to the RestartQ-UCB algorithm with Freedman-style bonus terms and reference-advantage decomposition (Zhang et al. 2020), which achieves a sharper dynamic regret bound at the cost of a slightly more involved analysis.

Common to both the Hoeffding and the Freedman bonus terms, RestartQ-UCB breaks the M episodes into D epochs , with each epoch containing K = glyph[ceilingleft] M D glyph[ceilingright] episodes (except for the last epoch which possibly has less than K episodes). With a large value of D , Algorithm 1 restarts more frequently to adjust to the potential variations of the environment, at the cost of spending more time searching for new optimal policies. On the contrary, a small value of D would lead to running stable policies for long periods of time with less frequent restarts, but the resulting algorithm might not be able to adjust to the environmental variations rapidly enough. To strike a balance, we set the number of epochs to be D = S -1 3 A -1 3 ∆ 2 3 H -2 3 T 1 3 so as to achieve the optimal dynamic regret bound, and such a choice will be justified later in our analysis. RestartQ-UCB periodically restarts a Q-learning algorithm with UCB exploration at the beginning of each epoch, thereby addressing the non-stationarity of the environment. For each d ∈ [ D ], define ∆ ( d ) r to be the local variation budget of the mean reward function within epoch d . By definition, we have ∑ D d =1 ∆ ( d ) r ≤ ∆ r . Define the local variation budget of transitions ∆ ( d ) p analogously.

Since our algorithm essentially invokes the same procedure for every epoch, in the following, we focus our analysis on what happens inside one epoch only (and without loss of generality, we focus on epoch 1, which contains episodes 1 , 2 , . . . , K ). At the end of our analysis, we will merge the results across all epochs.

For each triple ( s, a, h ) ∈S ×A× [ H ], we divide the visitations (within epoch 1) to the triple into multiple stages , where the length of the stages increases exponentially at a rate of (1 + 1 H ). Specifically, let e 1 = H , and e i +1 = glyph[floorleft] (1 + 1 H ) e i glyph[floorright] , i ≥ 1 denote the lengths of the stages. Further, let the partial sums L def = { ∑ j i =1 e i | j =1 , 2 , 3 , . . . } denote the set of the ending times of the stages. We remark that the stages are defined for each individual triple ( s, a, h ), and for different triples the starting and ending times of their stages do not necessarily align in time. Such a definition of stages is mostly motivated by the design of the learning rate α t = H +1 H + t in Jin et al. (2018). It ensures that only the last O (1 /H ) fraction of samples is given non-negligible weights when used to estimate the optimistic Q h ( s, a ) values, while the first 1 -O (1 /H ) fraction is forgotten (Zhang et al. 2020). We set ι def =log ( 2 δ ) , where δ is an input parameter that can be set by us.

Recall that the time index ( k, h ) represents the h -th step of the k -th episode. At each step ( k, h ), we take the optimal action with respect to the optimistic Q h ( s, a ) value (Line 6 in Algorithm 1), which is designed as an optimistic estimate of the optimal Q k,glyph[star] h ( s, a ) value of the corresponding episode. For

each triple ( s, a, h ), we update the optimistic Q h ( s, a ) value at the end of each stage, using samples only from this latest stage that is about to end (Line 16 in Algorithm 1). The optimism in Q h ( s, a ) comes from two bonus terms b h /b h and b ∆ , where b h /b h is a standard Hoeffding/Freedman-based optimism that is commonly used in upper confidence bounds (Jin et al. 2018, Zhang et al. 2020), and b ∆ is the extra optimism that we need to take into account because of the non-stationarity of the environment. The definition of b ∆ requires knowledge of the local variation budget in each epoch, which is a rather strong assumption in practice. However, we can further show (later in Theorems 2 and 3) that if we simply replace Equation ( ∗ ) in Algorithm 1 with the following update rule:

<!-- formula-not-decoded -->

then our algorithm can achieve the same regret without assumptions on the local variation budget.

Compared with the Hoeffding-based algorithm, there are two major improvements in the Freedmanbased one. The first improvement is the replacement of the Hoeffding-based bonus term b k h with a tighter term b k h . The latter term takes into account the second moment information of the random variables, which allows sharper tail bounds that rely on second moments to come into use (in our case, the Freedman's inequality). The second improvement is a variance reduction technique, or more specifically, the reference-advantage decomposition as coined in Zhang et al. (2020). The intuition is to first learn a reference value function V ref that serves as a roughly accurate estimate of the optimal value function V glyph[star] in each epoch. The goal of learning the optimal value function V glyph[star] = V ref +( V ∗ -V ref ) can hence be decomposed into estimating the two terms V ref and V ∗ -V ref . The reference value V ref is a fixed term, and can be accurately estimated using a large number of samples (in Algorithm 1, we estimate V ref only when we have N 0 = cSAH 6 ι samples for a large constant c ). The advantage term V ∗ -V ref can also be estimated more accurately due to the reduced variance.

## 4. Analysis

In this section, we present our main result-a dynamic regret analysis of the RestartQ-UCB algorithm. Our first result on RestartQ-UCB with Hoeffding-style bonus terms is summarized in the following theorem. Complete proofs of its supporting lemmas are given in Appendix B.

Theorem 1. (Hoeffding) For T =Ω( SA ∆ H 2 ) , and for any δ ∈ (0 , 1) , with probability at least 1 -δ , the dynamic regret of RestartQ-UCB with Hoeffding bonuses is bounded by ˜ O ( S 1 3 A 1 3 ∆ 1 3 H 5 3 T 2 3 ) , where ˜ O ( · ) hides poly-logarithmic factors of S,A,T and 1 /δ .

Our proof relies on the following technical lemma, stating that for any triple ( s, a, h ), the difference of their optimal Q -values at two different episodes 1 ≤ k 1 &lt;k 2 ≤ K is bounded by the variation of this epoch.

Lemma 1. For any triple ( s, a, h ) and any 1 ≤ k 1 &lt;k 2 ≤ K , it holds that | Q k 1 ,glyph[star] h ( s, a ) -Q k 2 ,glyph[star] h ( s, a ) | ≤ ∆ (1) r + H ∆ (1) p .

Let Q k h ( s, a ) denote the value of Q h ( s, a ) at the beginning of the k -th episode in RestartQ-UCB Hoeffding. The following lemma states that the optimistic Q -value Q k h ( s, a ) is an upper bound of the optimal Q -value Q k,glyph[star] h ( s, a ) with high probability. Note that we only need to show that the event holds with probability 1 -poly ( S,A,K,H ) δ , because we can replace δ with δ/ poly ( S,A,K,H ) in the end to get the desired high probability bound without affecting the polynomial part of the regret bound.

Lemma 2. (Hoeffding) For δ ∈ (0 , 1) , with probability at least 1 -2 KHδ , it holds that Q k,glyph[star] h ( s, a ) ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ) , ∀ ( s, a, h, k ) ∈S ×A× [ H ] × [ K ] .

Building upon Lemmas 1 and 2, a complete proof of Theorem 1 is given in Appendix C. We remark that Algorithm 1 relies on the assumption that the local variations b ∆ are known a priori, which is a strong but commonly made assumption in the literature on non-stationary RL (Ortner et al. 2019, Zhou et al. 2020a). To the best of our knowledge, existing restart-based solutions either crucially rely on this local variation assumption (Ortner et al. 2019), or suffer a severe regret degeneration after removing this assumption (Zhou et al. 2020a). Interestingly, in the following theorem, we show that this assumption can be safely removed in our approach without affecting the regret bound. The only modification to the algorithm is to replace the Q -value update rule in Equation ( ∗ ) of Algorithm 1 with the new update rule in Equation (1).

Theorem 2. (Hoeffding, no local budgets) For T = Ω( SA ∆ H 2 ) , and for any δ ∈ (0 , 1) , with probability at least 1 -δ , the dynamic regret of RestartQ-UCB with Hoeffding bonuses and no knowledge of local budgets is bounded by ˜ O ( S 1 3 A 1 3 ∆ 1 3 H 5 3 T 2 3 ) , where ˜ O ( · ) hides poly-logarithmic factors of S,A,T and 1 /δ .

To understand why this simple modification works, notice that in ( ∗ ) we add exactly the same value 2 b ∆ to the upper confidence bounds of all ( s, a ) pairs in the same epoch. Subtracting the same value from all optimistic Q -values simultaneously should not change the choice of actions in future steps. The only difference is that the new 'optimistic' Q k h ( s h , a h ) values would no longer be strict upper bounds of the optimal Q k,glyph[star] h ( s h , a h ) anymore, but instead 'upper bounds' subject to some error term induced by b ∆ . Specifically, since Q h ( s h , a h ) is updated using V h +1 ( s h +1 ), which in turn contains some error in terms of b ∆ , the error will propagate across the steps. By properly tracking such error terms, we can see that there are in total H -h +1 copies of the 2 b ∆ error accumulated from step H back to step h . This leads to the following variant of Lemma 2 that quantifies the error terms in the new 'optimistic' bounds.

Lemma 3. (Hoeffding, no local budgets) Suppose that we have no prior knowledge of the local variations and replace the update rule ( ∗ ) in RestartQ-UCB Hoeffding with Equation (1) . For

δ ∈ (0 , 1) , with probability at least 1 -2 KHδ , it holds that Q k,glyph[star] h ( s, a ) -2( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ) , ∀ ( s, a, h, k ) ∈S ×A× [ H ] × [ K ] .

Remark 1. The easy removal of the local budget assumption is non-trivial in the design of the algorithm, and to the best of our knowledge is absent in the non-stationary RL literature with restarts. In fact, it has been shown in a concurrent work (Zhou et al. 2020a) that removing this assumption could lead to a much worse regret bound (cf. Corollary 2 and Corollary 3 therein).

Replacing the Hoeffding-based upper confidence bound with a Freedman-style one will lead to a tighter regret bound, summarized in Theorem 3 below. To remove the local budget assumption, we also need to replace the update rule ( ∗ ) in Algorithm 1 with Equation (1). The proof of the theorem follows a similar procedure as in the proof of Theorem 2, and is given in Appendix E. It relies on a reference-advantage decomposition technique for variance reduction as in Zhang et al. (2020).

Theorem 3. (Freedman, no local budgets) For T greater than some polynomial of S,A, ∆ and H , and for any δ ∈ (0 , 1) , with probability at least 1 -δ , the dynamic regret of RestartQ-UCB with Freedman bonuses (Algorithm 1 including the gray parts) is upper bounded by ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ) , where ˜ O ( · ) hides poly-logarithmic factors of S,A,T and 1 /δ .

Remark 2 (From High Probability Regret Bound to Expected Regret Bound) . We note that δ is an input parameter, and our high probability regret bounds can immediately imply expected regret bounds. In all the above theorems presented in this section, the dynamic regret depends on 1 /δ through logarithmic terms. Since the regret can at most be O ( T ) , by setting δ =1 /T , one can retain the same regret bound in an expectation sense. For instance, in Theorem 3, by setting δ =1 /T, we have that with probability at least 1 -δ, the regret is ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ) , while with probability at most δ, the regret is O ( T ) . Hence, the expected regret of the algorithm is (1 -δ ) ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ) + δO ( T ) = ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 )

## 5. Unknown Variation Budgets

In Theorem 3, we have removed the assumption on the knowledge of 'local' variation budgets ∆ ( d ) r and ∆ ( d ) p for d ∈ [ D ], but the design of the algorithm still relies on knowledge of the 'total' variation budget ∆. Specifically, to achieve the dynamic regret bound presented in Theorem 3, we need to set the number of epochs to D glyph[star] = S -1 3 A -1 3 ∆ 2 3 T 1 3 , which clearly requires to know ∆ in advance. To further overcome such a limitation, in this section, we propose a parameter-free algorithm that adaptively learns the variation budget ∆ when it is unknown a priori, while still achieving sublinear dynamic regret in T .

Our new algorithm, Double-Restart Q-UCB, for the unknown variation budget setting is presented in Algorithm 2. Inspired by the Bandit-over-Bandit algorithm (Cheung et al. 2019a, 2020b) that adaptively tunes the algorithm parameters in a linear bandit problem, we also use a multi-armed

## Algorithm 2: Double-Restart Q-UCB

- 1 Input: Parameters W, J , α , and γ as given in Equation (2) and (3).
- 2 Initialize: Weights of the bandit arms s 1 ( j ) = exp ( αγ 3 √ glyph[ceilingleft] M/W glyph[ceilingright] J +1 ) for j =0 , 1 , . . . , glyph[ceilingleft] ln W glyph[ceilingright] .
- 3 for phase i ← 1 to ⌈ M W ⌉ do
- 4 p i ( j ) ← (1 -γ ) s i ( j ) ∑ J j ′ =0 s i ( j ′ ) + γ J +1 , ∀ j =0 , 1 , . . . , J ;
- 5 Draw an arm A i from { 0 , . . . , J } randomly according to the probabilities p i (0) , . . . , p i ( J );
- 6 Set the estimated number of epochs D i ← ⌊ TW A i J SAH 2 W ⌋ ;
- 7 Run a new instance of Algorithm 1 (including gray parts) for W episodes with parameter value D ← D i ;
- 8 Observe the cumulative reward R i from the last W episodes;
- 9 for arm j ← 0 , 1 , . . . , J do 10 ˆ R i ( j ) ← R i I { j = A i } / ( WHp i ( j )); ;
- 11 s i +1 ( j ) ← s i ( j ) exp ( γ 3( J +1) ( ˆ R i ( j ) + α p i ( j ) √ ( J +1) glyph[ceilingleft] M/W glyph[ceilingright] ))

bandit algorithm as a master procedure to learn the optimal value D glyph[star] of D . Given a set J of candidate values for D , the idea of our algorithm is to first divide the time horizon T into multiple phases , and then in each phase we experiment with one candidate value from the set J . If we choose values from J properly using a bandit algorithm, the cumulative reward we obtain through this experimentation procedure should be close to the performance of using the best fixed candidate from J in hindsight. Since the underlying environment need not drift according to any statistical pattern, we use an adversarial bandit algorithm Exp3.P (Auer et al. 2002) to defend against the possibly adversarial changes of the best D value in each phase.

Figure 1 Structure of the Double-Restart Q-UCB algorithm.

<!-- image -->

We sketch the high-level structure of the Double-Restart Q-UCB algorithm in Figure 1 to help clarify any possible confusion regarding our definitions of 'phases', 'epochs', and 'episodes'.

Concretely, we divide the overall M episodes into ⌈ M W ⌉ phases, each phase containing W ∈ N + episodes (except that the last phase could have less than W episodes). At the beginning of each phase i , we start a new instance of Algorithm 1 (including gray parts) with a candidate value of D i ∈ J to be experimented in this phase. Since Algorithm 1 itself is a restart-based process, it further sub-divides the W episodes in phase i into ⌈ D i W M ⌉ epochs. To understand this value, suppose D i is an appropriate value for D , such that dividing the overall horizon into D i epochs leads to near-optimal dynamic regret. Then, since the overall horizon contains M episodes while each phase only contains W episodes, we should only divide each phase into ⌈ D i W M ⌉ epochs to reflect the corresponding consequence of choosing D i as the overall number of epochs. Since we restart Algorithm 1 in each phase and Algorithm 1 in turn restarts an optimistic Q-learning sub-routine in each epoch, our overall algorithm exhibits a double-loop restarting behavior, and hence the name Double-Restart Q-UCB.

In the following, we instantiate the choices of the set J , the phase length W , as well as the parameter values used in the Exp3.P bandit algorithm. First, we define

<!-- formula-not-decoded -->

where J is the set of candidate values for D and we can see that |J | = glyph[ceilingleft] ln W glyph[ceilingright] +1= J +1. Each candidate value in J is also called an 'arm' in the language of bandits, and we use 'arm j ' to refer to the candidate value ⌊ TW j J SAH 2 W ⌋ for j =0 , 1 , . . . , J . We initialize the weights of the bandit arms by s 1 ( j ) = exp ( αγ 3 √ glyph[ceilingleft] M/W glyph[ceilingright] J +1 ) for j =0 , 1 , . . . , J , where as specified in Auer et al. (2002),

<!-- formula-not-decoded -->

for some failure probability δ &gt; 0. At the beginning of each phase i ∈{ 1 , 2 , . . . , ⌈ M W ⌉ } , we randomly draw an arm j with probability p i ( j ) that is calculated from the weights

<!-- formula-not-decoded -->

We set our estimated parameter D i to be the value associated with the selected arm j in the set J . We then run Algorithm 1 for W episodes by setting the number of epochs to be D = D i . To put it in another way, we execute a new instance of Algorithm 1 for ⌈ D i W M ⌉ epochs, where each epoch contains K i = ⌊ M D i ⌋ episodes. We collect the cumulative reward R i from the aforementioned W episodes. The normalized value R i / ( WH ) ∈ [0 , 1] hence corresponds to the reward of playing the

selected arm in time step i of the bandit problem. Finally, we update the weights of the bandit arms based on the observed reward, using the following update rule specified in the Exp3.P algorithm:

<!-- formula-not-decoded -->

where ˆ R i ( j ) = R i I { j = A i } / ( WHp i ( j )) , ∀ j =0 , 1 , . . . , J , and A i denotes the arm selected at phase i .

The following result states that our Double-Restart Q-UCB algorithm achieves a sublinear dynamic regret in T , without requiring knowledge of the (total) variation budget ∆.

Theorem 4. (Freedman, no total budgets) For T greater than some polynomial of S,A, ∆ and H , and for any δ ∈ (0 , 1) , with probability at least 1 -δ , the dynamic regret of Double-Restart Q-UCB with Freedman bonuses and no prior knowledge of the total variation budget ∆ is bounded by ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 + H 3 4 T 3 4 ) , where ˜ O ( · ) hides poly-logarithmic factors.

The regret bound in Theorem 4 consists of two terms: The first term is the dynamic regret of using the optimal candidate value D † ∈J of the number of epochs. This term is in the same order as the known-variation case (Theorem 3), because we have discretized the candidate value set J at a proper granularity such that the optimal candidate value D † ∈J approximates the actual optimal value D glyph[star] . The second regret term in Theorem 4 is caused by the regret of learning the optimal candidate value inside J using the Exp3.P algorithm. Due to the additional step of estimating the unknown variation budget, the overall dynamic regret bound becomes slightly worse in terms of its dependence on T (from ˜ O ( T 2 3 ) in Theorem 3 to ˜ O ( T 3 4 )). Such a degradation seems unavoidable under the current framework as it has also appeared in a similar bandit scenario (Cheung et al. 2019b).

Remark 3 (Comparison with Cheung et al. (2020a)) . We follow the Bandit-over-RL technique to utilize a separate bandit algorithm to select the key parameters for our algorithm. But we have to emphasize that the resulting algorithm is simpler and more practical for implementation. This is because our Double-Restart Q-UCB algorithm is essentially running a stationary Q-UCB algorithm in between restarts. In contrast, the algorithm in Cheung et al. (2020a)) relies on a carefully tuned sliding-window update schedule. More importantly, we point out that such a design (together with our new analysis) can lead to an improved dynamic regret bound in terms of S and A (even with the Hoeffding-style bonus terms similar to Cheung et al. (2020a)). This exhibits the advantage of our design compared to that of Cheung et al. (2020a), which combines restart and sliding-window.

## 6. Lower Bounds

In this section, we provide information-theoretical lower bounds of the dynamic regret to characterize the fundamental limits of any algorithm in non-stationary RL.

Theorem 5. For any algorithm, there exists an episodic non-stationary MDP such that the dynamic regret of the algorithm is at least Ω( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 ) .

Proof sketch. The proof of our lower bound relies on the construction of a 'hard instance' of non-stationary MDPs. The instance we construct is essentially an MDP with piecewise constant dynamics on each segment of the horizon, and its dynamics experience an abrupt change at the beginning of each new segment. Specifically, we divide the horizon T into L segments 1 , where each segment has T 0 def = ⌊ T L ⌋ steps and contains M 0 def = ⌊ M L ⌋ episodes. Within each segment, the system dynamics of the MDP do not vary, and we construct the dynamics for each segment in a way such that the instance is a hard instance of stationary MDPs on its own. The MDP within each segment is essentially similar to the hard instances constructed in Osband and Van Roy (2016), Jin et al. (2018). Between two consecutive segments, the dynamics of the MDP change abruptly, and we let the dynamics vary in a way such that no information learned from previous interactions with the MDP can be used in the new segment. In this sense, the agent needs to learn a new hard MDP in each segment. Finally, optimizing the value of L and the variation magnitude between consecutive segments (subject to the constraints of the total variation budget) leads to our lower bound.

Remark 4. We emphasize that in our construction of the worst-case non-stationary MDP, we only let the state transition kernel vary over time but keep the reward functions fixed. By doing so, we are able to provide a lower bound of order Ω( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 ) . Recall that the upper bound stated in Theorem 3 is O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ) , and hence our upper and lower bounds match in terms of ∆ (=∆ r +∆ p ) .

Remark 5 (Tightness of Our Results) . For our setting, we conjecture that the lower bound can be improved. Our current construction of the lower bound relies on a chain of H copies of 'JAO MDPs' Jaksch et al. (2010). The non-stationarity is achieved by changing the transitions abruptly after a fixed time period, and such a change applies simultaneously across all H copies of JAO MDPs. One possible direction is to construct the lower bound instances such that the state transition kernel is allowed to vary within the same episode, which we have not taken advantage of. Including this extra ingredient into the construction could potentially lead to a sharper lower bound, and we leave this as future work.

A useful side result of our proof is the following lower bound for non-stationary RL in the un-discounted setting, which is the same setting as studied in Gajane et al. (2018), Ortner et al. (2019) and Cheung et al. (2020b).

1 The definition of segments is irrelevant to, and should not be confused with, the notion of epochs we previously defined.

| Algorithm        | Time per episode   |
|------------------|--------------------|
| RestartQ-UCB     | 0.102 ms           |
| LSVI-UCB-Restart | 57.65 ms           |
| Q-Learning UCB   | 0.098 ms           |
| Epsilon-Greedy   | 0.123 ms           |

Figure 2 Cumulative rewards of the four algorithms under (a) abrupt variations, and (b) gradual variations, respectively, as well as their (c) time usage. Shaded areas denote the standard deviations of rewards. Note that RestartQ-UCB significantly outperforms Q-Learning UCB and Epsilon-Greedy, and matches LSVI-UCB-Restart while being much more time-efficient.

<!-- image -->

Proposition 1. Consider a reinforcement learning problem in un-discounted non-stationary MDPs with horizon length T , total variation budget ∆ , and maximum MDP diameter D (Cheung et al. 2020b). For any learning algorithm, there exists a non-stationary MDP such that the dynamic regret of the algorithm is at least Ω( S 1 3 A 1 3 ∆ 1 3 D 2 3 T 2 3 ) .

## 7. Simulations

In this section, we empirically evaluate RestartQ-UCB on reinforcement learning tasks with various types of non-stationarity. We compare RestartQ-UCB with three baseline algorithms: LSVI-UCB-Restart (Zhou et al. 2020a), Q-Learning UCB, and Epsilon-Greedy (Watkins 1989). LSVI-UCB-Restart is a state-of-the-art non-stationary RL algorithm that combines optimistic least-squares value iteration with periodic restarts. Q-Learning UCB is simply our RestartQ-UCB algorithm with no restart. It is a Q-learning based algorithm that uses upper confidence bounds to guide the exploration. Epsilon-Greedy is a restart-based algorithm that uses an epsilon-greedy strategy for action selection.

We evaluate the cumulative rewards of the four algorithms on a variant of a reinforcement learning task named Bidirectional Diabolical Combination Lock (Agarwal et al. 2020, Misra et al. 2020). This task is designed to be particularly difficult for exploration . We introduce two types of non-stationarity to the task, namely abrupt variations and gradual variations. A detailed discussion on the task settings as well as the configuration of the hyper-parameters is deferred to Appendix J. The cumulative rewards of the four algorithms in the abruptly-changing and gradually-changing environments are shown in Figures 2(a) and 2(b), respectively. All results are averaged over 30 runs.

As we can see, RestartQ-UCB outperforms Q-Learning UCB and Epsilon-Greedy under both types of environment variations. For the abruptly-changing environment as an example, RestartQ-UCB achieves 1 . 36 and 2 . 52 times of the cumulative rewards of Q-Learning UCB and Epsilon-Greedy,

respectively. This demonstrates the importance of both addressing the environment variations (using restarts) and actively exploring the environment (using UCB-based bonus terms) in non-stationary RL. LSVI-UCB-Restart nearly matches the performance of RestartQ-UCB, which is unsurprising because both of them use the restarting strategy and optimistic exploration. Nevertheless, LSVIUCB-Restart requires a higher time and space complexity. It needs to store all the history information in one epoch and solve a regularized least-squares minimization problem at every time step. This is indeed evidenced by our simulation results (shown in Figure 2(c)) that RestartQ-UCB only takes 0 . 18% of the computation time of LSVI-UCB-Restart.

Remark 6. The heavy computation in LSVI-UCB-Restart mostly comes from the usage of a high-dimensional feature. In our simulations, we followed Example 2.1 in Jin et al. (2019) to convert a linear MDP algorithm to a tabular one, which results in a feature dimension of d = S × A . This is essentially the most efficient feature encoding when no special structure is imposed on the tabular MDP. We believe that designing low-dimensional features for specific MDP instances can possibly reduce the computations for LSVI-UCB-Restart by a large amount, and is an interesting future direction for learning in linear MDPs per se.

## 8. Application to Multi-Agent RL

In this section, we discuss an application of our non-stationary RL method to multi-agent RL in episodic stochastic teams, which by nature leads to a non-stationary RL problem from each agent's perspective. Such a decentralized multi-agent scenario also helps us demonstrate the significance and flexibility of the model-free approach.

## 8.1. Problem Setup

In general, an N -player episodic stochastic team is defined by a tuple ( N , H, S , {A ( i ) } N i =1 , r, P ), where (1) N = { 1 , 2 , . . . , N } is the set of agents; (2) H ∈ N + is the number of time steps in each episode; (3) S is the finite state space; (4) A ( i ) is the finite action space for agent i ∈ N ; (5) r h : S × A → [0 , 1] is the team reward function at step h ∈ [ H ] common to all the agents i ∈ N , where A = × N i =1 A ( i ) ; and (6) P h : S × A → ∆( S ) is the transition kernel at step h ∈ [ H ], where the next state depends on the current state and the joint actions of all the agents. The game lasts for M episodes, and we let T = MH be the total number of time steps. At each time step ( m,h ), the agents observe the state s m h ∈S , and take actions a ( i ) ,m h ∈A ( i ) , i ∈N simultaneously. 2 We let a m h =( a (1) ,m h , . . . , a ( N ) ,m h ). All the agents receive a common reward with an expected value of r h ( s m h , a m h ), and the environment transitions to the next state s m h +1 ∼ P h ( ·| s m h , a m h ). For each agent i , a policy is a mapping from the time index and state space to (possibly a distribution over) the

2 Note that we use superscripts in parentheses to index the agents, while a superscript with no parenthesis denotes the index of an episode.

action space. We denote the set of policies for agent i by Π ( i ) = { π ( i ) : [ M ] × [ H ] ×S → ∆( A ( i ) ) } . The set of joint policies are denoted by Π = × N i =1 Π ( i ) . Each agent seeks to find a policy such that the collective choice maximizes the cumulative team reward.

For notational convenience, and without much loss of conceptual generality, we consider two-player teams, i.e., N =2. We consider the problem where we can control the policy of agent 1, while agent 2 is an opponent that is adapting its own policy in an unknown way. Achieving sublinear regret in the face of an arbitrarily changing opponent is known to be computationally hard (Radanovic et al. 2019). Therefore, existing works (Radanovic et al. 2019, Lee et al. 2020) often focus on a setting where the opponent is only 'slowly changing' its policy over time. One such example is when the opponent is using a relatively stable learning algorithm. We also focus on the decentralized setting, a more practical multi-agent RL paradigm, where an agent cannot observe the actions and rewards of the other agent.

A joint policy induces a probability measure on the sequence of states and joint actions. For a joint policy π =( π (1) , π (2) ) ∈ Π, and for each time step ( m,h ) ∈ [ M ] × [ H ], state s ∈S , we define the state value function for the agents as follows:

<!-- formula-not-decoded -->

For a joint policy ( π (1) , π (2) ), we again evaluate the optimality of agent 1's policy π (1) in terms of its dynamic regret , which compares the agent's policy with the optimal policy of each individual episode in hindsight:

<!-- formula-not-decoded -->

The initial state of each episode s m 1 is again chosen by an oblivious adversary.

We model the slowly-changing behavior of agent 2 by requiring it to have a low switching cost (Bai et al. 2019, Gao et al. 2021) defined as follows.

Definition 1. The switching cost between any pair of policies ( π,π ′ ) is the number of ( h,s ) pairs on which π and π ′ act differently:

glyph[negationslash]

<!-- formula-not-decoded -->

For a policy trajectory ( π 1 , . . . , π M ) across M episodes, its switching cost is defined as N switch def = ∑ M m =1 n switch ( π m , π m +1 ) .

We characterize the behavior of agent 2 by assuming that the switching cost of its policy trajectory is upper bounded by O ( T β ) for some 0 &lt;β &lt; 1. Many existing RL algorithms (Bai et al. 2019, Zhang et al. 2020) satisfy this upper bound.

## 8.2. Learning Team-Optimality

Our non-stationary RL algorithm can be readily applied to learning team-optimal policies in 'smooth games', which is the setting considered in Radanovic et al. (2019). This corresponds to the setting where a team of agents learn to collaborate. We define team-optimality as the joint policy of the agents that induces the highest possible accumulated reward.

Definition 2. In a two-player team, a joint policy π glyph[star] =( π (1) glyph[star] , π (2) glyph[star] ) ∈ Π is called team-optimal if

<!-- formula-not-decoded -->

where V ( π (1) ,π (2) ) h ( s ) def = E [ ∑ H h ′ = h r h ′ ( s h ′ , π (1) h ′ ( s h ′ ) , π (2) h ′ ( s h ′ )) | s h = s ] is the value function.

Since we cannot control the behavior of agent 2, its behavior might be sub-optimal and drive us away from team-optimality. To avoid such scenarios, we impose a structural assumption that allows us to quantify the distance from optimality. In particular, we assume that the team is ( λ,µ )-smooth, following the definition in Radanovic et al. (2019).

Definition 3. (Adapted from Definition 1 in Radanovic et al. (2019)) A two-player stochastic team is ( λ,µ ) -smooth if there exists a pair of policies ( π (1) glyph[star] , π (2) glyph[star] ) such that for every policy pair ( π (1) , π (2) ) and every h ∈ [ H ] , s ∈S :

<!-- formula-not-decoded -->

The ( λ,µ )-smoothness ensures that agent 2's sub-optimal behavior only has a bounded negative impact on the joint value. This notion of smoothness is motivated by the definition of smooth games in Roughgarden (2009) and Syrgkanis et al. (2015), as stated in Radanovic et al. (2019). Applying our RestartQ-UCB algorithm for agent 1 would lead to the following theorem, which implies that the time-average return of the agents converges to a λ 1+ µ factor of the team-optimal value as T grows.

Theorem 6. Let π (2) denote the policy of agent 2, and suppose that the switching cost of agent 2 satisfies N switch = O ( T β ) for 0 &lt;β&lt; 1 . Assume that the team problem is ( λ,µ ) -smooth. Let agent 1 run the RestartQ-UCB algorithm, and let π (1) denote its induced policy. For T large enough, the return of the algorithm is lower bounded by:

<!-- formula-not-decoded -->

Remark 7. (Significance of model-freeness.) Decentralized multi-agent RL is generally only possible with model-free approaches (see, e.g., Arslan and Y¨ uksel (2016), Tian et al. (2021), Daskalakis et al. (2020)); model-based methods proceed by explicitly estimating the transition and reward functions,

which crucially relies on observing the other agents' actions. This further demonstrates the flexibility and significance of model-free methods, when one addresses the non-stationarity issues in multi-agent RL through the lens of non-stationary RL.

## 9. Application to Inventory Control Across Related Products

In this section, we discuss the application of our non-stationary RL algorithm to the problem of inventory control across related products. Different from conventional inventory control problems (e.g., Huh and Rusmevichientong (2009)) that only consider one product, we investigate the case where a sequence of related products are being sold, and the products share similar but different demand distributions. This is motivated by the sequential launch of related products (e.g., the line of iPhone) that allows us to leverage experience from past products to inform inventory management for future ones. Following Yuan et al. (2021), Cheung et al. (2020a) (who only consider a single product being sold), we focus on the setting of zero lead time, fixed cost, and lost sales.

## 9.1. Problem Setup

The inventory control problem has M episodes, representing M different but related products. Each episode/product lasts for H time steps. 3 For each time step h ∈ [ H ] of an episode m ∈ [ M ], the following sequence of events happens in order:

1. The seller observes her stock level s m h ≥ 0 for product m at the beginning of time step h , and decides on the quantity a m h ≥ 0 to order.
2. If a m h &gt; 0, the order arrives immediately, and the seller's stock level becomes s m h + a m h . The seller pays a fixed cost f and a c per-unit ordering cost.
3. The random demand X m h is realized. The seller only observes the actual sales quantity, or censored demand Y h m =min { X m h , s m h + a m h } . She will not know the actual demand if X m h ≥ s m h + a m h . Following prior works Cheung et al. (2020b), Agarwal et al. (2020) and Yuan et al. (2021), we assume that the demands X m h are independent random variables over m , but they do not necessarily follow identical distributions since we consider different products across the episodes.
4. All unfulfilled demands are permanently lost and incur a per-unit lost sales cost p . Excess inventory incurs a per-unit holding cost q . The total cost at step h can be expressed as

<!-- formula-not-decoded -->

5. The inventory carried over to the next step h +1 is s m h +1 =[ s m h + a m h -X m h ] + .

3 We assume for simplicity that the life cycle of each product is of the same length.

Following Cheung et al. (2020a), Yuan et al. (2021), we assume that the seller has a finite storage capacity S , in the sense that she can hold at most S -1 units of inventory at any time. The seller's objective is to minimize her cumulative cost ∑ M m =1 ∑ H h =1 C m h ( s m h , a m h ). At the end of each episode, as a product is reaching the end of its life cycle, we assume for simplicity that the storage is emptied at no cost. Such an inventory control problem can be easily formulated as an instance of the non-stationary RL model that we defined in Section 2. Concretely, we treat the stock level s m h at the beginning of each time step as the state of the environment, and regard the order quantity a m h as the action at the corresponding time step. Consequently, we define the state space of the problem as S = { 0 , 1 , . . . , S -1 } , and the state-dependent action space as A s = { 0 , 1 , . . . , S -1 -s } . One can verify that Algorithm 1 and its analysis easily generalize to state-dependent action spaces.

The reward function of the non-stationary MDP is defined as R m h ( s m h , a m h ) = -C m h ( s m h , a m h ), and we let r m h ( s m h , a m h ) = E [ R m h ( s m h , a m h )] be the expected value of the reward. For any s m h , s m h +1 ∈S and a m h ∈A s , we define the state transition function as

<!-- formula-not-decoded -->

Our definitions of the policy π , the value function V m,π h , the state-action value function Q m,π h , as well as the optimal policy π glyph[star] and its corresponding value functions V m,glyph[star] h , Q m,glyph[star] h directly carry over from Section 2 to this problem instance, and we do not repeat such definitions here for simplicity. The variation budget ∆ is also defined in the same way as in Section 2, which captures the differences in the products' demand distributions for this problem. The dynamic regret of the agent's policy is defined analogously as

<!-- formula-not-decoded -->

## 9.2. Implementation of RestartQ-UCB

Notably, one major difference between the inventory control problem we considered in Section 9.1 and our non-stationary MDP formulation in Section 2 is that due to demand censoring, the seller cannot calculate the actual cost C m h ( s m h , a m h ), and hence the immediate reward R m h ( s m h , a m h ) is also not observable. Nevertheless, we will show that one can bypass such an issue by using a pseudoreward technique, which was originally introduced for a stationary problem (Agrawal and Jia 2019). Specifically, for every time step h ∈ [ H ] in episode m ∈ [ M ], and for every state s ∈S and action a ∈A s , we define the pseudo-reward as

<!-- formula-not-decoded -->

where we recall that the censored demand Y h m =min { X m h , s + a } is perfectly observable. Similarly, we can also define the mean pseudo-reward as

<!-- formula-not-decoded -->

Therefore, the mean pseudo-reward can be considered as shifting the mean reward function uniformly by an amount of p · E [ X m h ]. Without loss of generality, we normalize the pseudoreward to the range [0 , 1]. We use the tuple M = { S , A , H, { P m h } m ∈ [ M ] ,h ∈ [ H ] , { r m h } m ∈ [ M ] ,h ∈ [ H ] } to denote the non-stationary MDP with respect to the original reward function, and let M pseudo = { S , A , H, { P m h } m ∈ [ M ] ,h ∈ [ H ] , { r m, pseudo h } m ∈ [ M ] ,h ∈ [ H ] } be the one corresponding to the pseudo-reward. We further define π glyph[star], pseudo to be the (episode-wise) optimal policy for M pseudo , and let V m,glyph[star], pseudo h and Q m,glyph[star], pseudo h , respectively, be the corresponding value function and state-action value function.

Since only the pseudo-reward is observable, we can only apply our RestartQ-UCB algorithm to M pseudo rather than M . A natural question then is whether we can generalize the performance guarantee from M pseudo to M . Interestingly, the following result (adapted from Agrawal and Jia (2019)) shows that, for any (possibly non-Markovian) policy π induced by Algorithm 1, the dynamic regret on M pseudo and M are equal.

Lemma 4. (Adapted from Lemma 3.1 in Agrawal and Jia (2019)). Let F m h be the set of all historical information collected up to the beginning of time step h of episode m . Let π be the (possibly non-Markovian) policy induced by Algorithm 1, such that π m h ( s m h , F m h ) maps the state and history to a distribution over the action space. Then, π incurs the same dynamic regret on M and M pseudo :

<!-- formula-not-decoded -->

Together with Theorem 1, we obtain the following dynamic regret bound for running Algorithm 1 on the inventory control problem across related products.

Theorem 7. For T =Ω( SA ∆ H 2 ) , and for any δ ∈ (0 , 1) , with probability at least 1 -δ , the dynamic regret of running Algorithm 1 on the inventory control problem formulated in Section 9.1 with pseudorewards and Freedman bonuses is bounded by ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ) , where ˜ O ( · ) hides poly-logarithmic factors of S,A,T and 1 /δ .

Remark 8 (Comparison with Cheung et al. (2020b)) . Although both our work and Cheung et al. (2020b) consider applications in inventory control and utilizes techniques from Agrawal and Jia (2019), the foci are quite different. In Cheung et al. (2020b), the authors only study single product inventory, whereas in contrast, our work studies the setting where there is a sequence of related, but different products.

Specifically, in Cheung et al. (2020b), variation budget has been defined with respect to demand changes within a single product selling horizon, and the corresponding regret upper bound scales with this budget, whereas in ours no constraint has been put to limit the demand changes within a single product's selling horizon (an episode), and the variation budget captures the difference across products. This is similar to a meta/transfer learning setting where the goal is to leverage

data obtained from inventory learning for similar products to accelerate inventory learning for the new product.

Moreover, as discussed in Section 1, a direct application of the results in Cheung et al. (2020b) to this setting may lead to a worse regret upper bound.

Remark 9. Our results can be extended to a multi-product inventory control problem with a warehouse-capacity constraint, similar to the setting studied in Shi et al. (2016). Specifically, we have an episodic setting with n products and M episodes, where each episode lasts for H time steps. For each time step h ∈ [ H ] of an episode m ∈ [ M ] , a demand is specified for every product i ∈ [ n ] . In our non-stationary formulation, the demands need not follow identical distributions over time. An overall warehouse capacity constraint is also imposed on the total number of products simultaneously in the inventory. At each time step, the seller observes the stock level and decides on the quantity to order for each product at a certain per-unit ordering cost. Unfulfilled demands are permanently lost and incur a per-unit lost sales cost. Excess inventory also incurs a per-unit holding cost. The seller's objective is to minimize the cumulative cost. Such a multi-product problem can also be cast as an MDP, where we define the stock levels of all products to be the state of the environment, and define the joint ordering quantity across all products as the action of each step. We also let the action space be state-dependent to handle the joint capacity constraint; in particular, an action is considered invalid at a certain state if the corresponding ordering quantity causes the stock levels to exceed the warehouse capacity. Applying Algorithm 1 to such a non-stationary multi-product problem leads to the same dynamic regret bound as in Theorem 7, though the state space should now be interpreted as the possible combinations of stock levels across all products that do not exceed the warehouse capacity, which is significantly larger than the single-product case. The same is true for the action space. A final remark is that the multi-product formulation above does not consider upgrading (Yu et al. 2015), the situation where a high-quality product is used to serve the demand of a lower-quality one that has been sold out. Upgrading adds an additional element of difficulty to the decision-making process, as the seller now needs to consider the ordering and upgrading decisions jointly. We leave the treatment of such a more intricate scenario to our future work.

## 10. Concluding Remarks

In this paper, we have considered model-free reinforcement learning in non-stationary episodic MDPs. We have proposed an algorithm named RestartQ-UCB that adopts a simple restarting strategy. RestartQ-UCB with Freedman-type bonus terms achieves a dynamic regret of ˜ O ( S 1 3 A 1 3 ∆ 1 3 HT 2 3 ), which nearly matches the information-theoretical lower bound Ω( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 ). We have further presented a parameter-free algorithm named Double-Restart Q-UCB that removes the assumption on knowing the variation budget. Numerical experiments have validated the advantages of RestartQUCB in terms of both cumulative rewards and computational efficiency. Examples in multi-agent

RL and inventory control have been discussed as applications to illustrate the power of our method. An interesting future direction would be to close the ˜ O ( H 1 3 ) factor gap between the upper and lower bounds that we have established for the non-stationary RL problem. It would also be interesting to explore if non-stationary RL can be helpful in other multi-agent RL or inventory control scenarios.

## References

- Abel, David, Yuu Jinnai, Sophie Yue Guo, George Konidaris, Michael Littman. 2018. Policy and value transfer in lifelong reinforcement learning. International Conference on Machine Learning . 20-29.
- Agarwal, Alekh, Mikael Henaff, Sham Kakade, Wen Sun. 2020. PC-PG: Policy cover directed exploration for provable policy gradient learning. arXiv preprint arXiv:2007.08459 .
- Agrawal, Shipra, Randy Jia. 2019. Learning in structured MDPs with convex cost functions: Improved regret bounds for inventory management. ACM Conference on Economics and Computation . 743-744.
- Allesiardo, Robin, Rapha¨ el F´ eraud, Odalric-Ambrym Maillard. 2017. The non-stationary stochastic multiarmed bandit problem. International Journal of Data Science and Analytics 3 (4) 267-283.
- Anonymous. 2021. Near-optimal model-free reinforcement learning in non-stationary episodic MDPs. International Conference on Machine Learning . 7447-7458.
- Arora, Raman, Ofer Dekel, Ambuj Tewari. 2012. Deterministic MDPs with adversarial rewards and bandit feedback. Conference on Uncertainty in Artificial Intelligence . 93-101.
- Arslan, G¨ urdal, Serdar Y¨ uksel. 2016. Decentralized Q-learning for stochastic teams and games. IEEE Transactions on Automatic Control 62 (4) 1545-1558.
- Auer, Peter, Nicolo Cesa-Bianchi, Yoav Freund, Robert E Schapire. 2002. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing 32 (1) 48-77.
- Auer, Peter, Pratik Gajane, Ronald Ortner. 2019. Adaptively tracking the best bandit arm with an unknown number of distribution changes. Conference on Learning Theory . 138-158.
- Azar, Mohammad Gheshlaghi, Ian Osband, R´ emi Munos. 2017. Minimax regret bounds for reinforcement learning. International Conference on Machine Learning . 263-272.
- Bai, Yu, Tengyang Xie, Nan Jiang, Yu-Xiang Wang. 2019. Provably efficient Q-learning with low switching cost. Advances in Neural Information Processing Systems . 8004-8013.
- Balseiro, Santiago R., Yonatan Gur. 2019. Learning in repeated auctions with budgets: Regret minimization and equilibrium. Management Science 65 (9) 3952-3968.
- Bastani, Hamsa, David Simchi-Levi, Ruihao Zhu. 2021. Meta dynamic pricing: Transfer learning across experiments. Management Science (Forthcoming) .
- Besbes, Omar, Yonatan Gur, Assaf Zeevi. 2014. Stochastic multi-armed-bandit problem with non-stationary rewards. Advances in Neural Information Processing Systems . 199-207.

- Besbes, Omar, Yonatan Gur, Assaf Zeevi. 2019. Optimal exploration-exploitation in a multi-armed bandit problem with non-stationary rewards. Stochastic Systems 9 (4) 319-337.
- Birge, John R., Hongfan Chen, N. Bora Keskin, Amy Ward. 2021. To interfere or not to interfere: Information revelation and price-setting incentives in a multiagent learning environment. SSRN 3864227 .
- Brunskill, Emma, Lihong Li. 2013. Sample complexity of multi-task reinforcement learning. Uncertainty in Artificial Intelligence . 122.
- Cai, Han, Kan Ren, Weinan Zhang, Kleanthis Malialis, Jun Wang, Yong Yu, Defeng Guo. 2017. Real-time bidding by reinforcement learning in display advertising. International Conference on Web Search and Data Mining . 661-670.
- Chawla, Shuchi, Nikhil R Devanur, Anna R Karlin, Balasubramanian Sivan. 2016. Simple pricing schemes for consumers with evolving values. ACM-SIAM Symposium on Discrete Algorithms . 1476-1490.
- Chen, Chacha, Hua Wei, Nan Xu, Guanjie Zheng, Ming Yang, Yuanhao Xiong, Kai Xu, Zhenhui Li. 2020a. Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control. AAAI Conference on Artificial Intelligence . 3414-3421.
- Chen, Ningyuan, Chun Wang, Longlin Wang. 2020b. Learning and optimization with seasonal patterns. arXiv preprint arXiv:2005.08088 .
- Chen, Yifang, Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei. 2019. A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free. Conference on Learning Theory . PMLR, 696-726.
- Cheung, Wang Chi, David Simchi-Levi, Ruihao Zhu. 2019a. Hedging the drift: Learning to optimize under non-stationarity. arXiv preprint arXiv:1903.01461 .
- Cheung, Wang Chi, David Simchi-Levi, Ruihao Zhu. 2019b. Learning to optimize under non-stationarity. International Conference on Artificial Intelligence and Statistics . 1079-1087.
- Cheung, Wang Chi, David Simchi-Levi, Ruihao Zhu. 2020a. Non-stationary reinforcement learning: The blessing of (more) optimism. SSRN Preprint 3397818 .
- Cheung, Wang Chi, David Simchi-Levi, Ruihao Zhu. 2020b. Reinforcement learning for non-stationary Markov decision processes: The blessing of (more) optimism. arXiv preprint arXiv:2006.14389 .
- Daskalakis, Constantinos, Dylan J Foster, Noah Golowich. 2020. Independent policy gradient methods for competitive reinforcement learning. Advances in Neural Information Processing Systems 33 .
- Dick, Travis, Andras Gyorgy, Csaba Szepesvari. 2014. Online learning in Markov decision processes with changing cost sequences. International Conference on Machine Learning . 512-520.
- Domingues, Omar Darwiche, Pierre M´ enard, Matteo Pirotta, Emilie Kaufmann, Michal Valko. 2021. A kernelbased approach to non-stationary reinforcement learning in metric spaces. International Conference on Artificial Intelligence and Statistics . 3538-3546.

- Fei, Yingjie, Zhuoran Yang, Zhaoran Wang, Qiaomin Xie. 2020. Dynamic regret of policy optimization in non-stationary environments. Advances in Neural Information Processing Systems 33 .
- Freedman, David A. 1975. On tail probabilities for martingales. The Annals of Probability 100-118.
- Gajane, Pratik, Ronald Ortner, Peter Auer. 2018. A sliding-window algorithm for Markov decision processes with arbitrarily changing rewards and transitions. arXiv preprint arXiv:1805.10066 .
- Gao, Minbo, Tianle Xie, Simon S Du, Lin F Yang. 2021. A provably efficient algorithm for linear Markov decision process with low switching cost. arXiv preprint arXiv:2101.00494 .
- Garivier, Aur´ elien, Eric Moulines. 2011. On upper-confidence bound policies for switching bandit problems. International Conference on Algorithmic Learning Theory . 174-188.
- Huh, Woonghee Tim, Paat Rusmevichientong. 2009. A nonparametric asymptotic analysis of inventory planning with censored demand. Mathematics of Operations Research 34 (1) 103-123.
- Jaksch, Thomas, Ronald Ortner, Peter Auer. 2010. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research 11 1563-1600.
- Jin, Chi, Zeyuan Allen-Zhu, Sebastien Bubeck, Michael I Jordan. 2018. Is Q-learning provably efficient? Advances in Neural Information Processing Systems . 4863-4873.
- Jin, Chi, Tiancheng Jin, Haipeng Luo, Suvrit Sra, Tiancheng Yu. 2019. Learning adversarial MDPs with bandit feedback and unknown transition. arXiv preprint arXiv:1912.01192 .
- Kaplanis, Christos, Murray Shanahan, Claudia Clopath. 2018. Continual reinforcement learning with complex synapses. International Conference on Machine Learning . 2497-2506.
- Karnin, Zohar S, Oren Anava. 2016. Multi-armed bandits: Competing with optimal sequences. Advances in Neural Information Processing Systems . 199-207.
- Keskin, N Bora, Assaf Zeevi. 2017. Chasing demand: Learning and earning in a changing environment. Mathematics of Operations Research 42 (2) 277-307.
- Lee, Chung-Wei, Haipeng Luo, Chen-Yu Wei, Mengxiao Zhang. 2020. Linear last-iterate convergence for matrix games and stochastic games. arXiv preprint arXiv:2006.09517v1 Available at https://arxiv.org/pdf/2006.09517v1.pdf.
- Littman, Michael L. 1994. Markov games as a framework for multi-agent reinforcement learning. International Conference on Machine Learning . 157-163.
- Lu, Junwei, Chaoqi Yang, Xiaofeng Gao, Liubin Wang, Changcheng Li, Guihai Chen. 2019. Reinforcement learning with sequential information clustering in real-time bidding. International Conference on Information and Knowledge Management . 1633-1641.
- Luo, Haipeng, Chen-Yu Wei, Alekh Agarwal, John Langford. 2018. Efficient contextual bandits in nonstationary worlds. Conference On Learning Theory . 1739-1776.

- Lykouris, Thodoris, Max Simchowitz, Aleksandrs Slivkins, Wen Sun. 2019. Corruption robust exploration in episodic reinforcement learning. arXiv preprint arXiv:1911.08689 .
- Ma, Will. 2018. Improvements and generalizations of stochastic knapsack and Markovian bandits approximation algorithms. Mathematics of Operations Research 43 (3) 789-812.
- Mao, Weichao, Kaiqing Zhang, Qiaomin Xie, Tamer Ba¸ sar. 2020. POLY-HOOT: Monte-Carlo planning in continuous space MDPs with non-asymptotic analysis. Advances in Neural Information Processing Systems 33 .
- Misra, Dipendra, Mikael Henaff, Akshay Krishnamurthy, John Langford. 2020. Kinematic state abstraction and provably efficient rich-observation reinforcement learning. International Conference on Machine Learning . 6961-6971.
- Neu, Gergely, Andras Antos, Andr´ as Gy¨ orgy, Csaba Szepesv´ ari. 2010. Online Markov decision processes under bandit feedback. Advances in Neural Information Processing Systems . 1804-1812.
- Ortner, Ronald, Pratik Gajane, Peter Auer. 2019. Variational regret bounds for reinforcement learning. Uncertainty in Artificial Intelligence . 81-90.
- Osband, Ian, Benjamin Van Roy. 2016. On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 .
- Padakandla, Sindhu. 2020. A survey of reinforcement learning algorithms for dynamically varying environments. arXiv preprint arXiv:2005.10619 .
- Radanovic, Goran, Rati Devidze, David Parkes, Adish Singla. 2019. Learning to collaborate in Markov decision processes. International Conference on Machine Learning . 5261-5270.
- Roughgarden, Tim. 2009. Intrinsic robustness of the price of anarchy. Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing . 513-522.
- Shi, Cong, Weidong Chen, Izak Duenyas. 2016. Nonparametric data-driven algorithms for multiproduct inventory systems with censored demand. Operations Research 64 (2) 362-370.
- Shortreed, Susan M, Eric Laber, Daniel J Lizotte, T Scott Stroup, Joelle Pineau, Susan A Murphy. 2011. Informing sequential clinical decision-making through reinforcement learning: An empirical study. Machine Learning 84 (1-2) 109-136.
- Sun, Yanchao, Xiangyu Yin, Furong Huang. 2020. Temple: Learning template of transitions for sample efficient multi-task RL. arXiv preprint arXiv:2002.06659 .
- Syrgkanis, Vasilis, Alekh Agarwal, Haipeng Luo, Robert E Schapire. 2015. Fast convergence of regularized learning in games. Advances in Neural Information Processing Systems 28 2989-2997.
- Tekin, Cem, Mingyan Liu. 2010. Online algorithms for the multi-armed bandit problem with Markovian rewards. 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton) . 1675-1682.

- Tian, Yi, Yuanhao Wang, Tiancheng Yu, Suvrit Sra. 2021. Online learning in unknown Markov games. International Conference on Machine Learning . 10279-10288.
- Tirinzoni, Andrea, Riccardo Poiani, Marcello Restelli. 2020. Sequential transfer in reinforcement learning with a generative model. arXiv preprint arXiv:2007.00722 .
- Touati, Ahmed, Pascal Vincent. 2020. Efficient learning in non-stationary linear Markov decision processes. arXiv preprint arXiv:2010.12870 .
- Wang, Jingkang, Yang Liu, Bo Li. 2020. Reinforcement learning with perturbed rewards. Proceedings of the AAAI Conference on Artificial Intelligence , vol. 34. 6202-6209.
- Watkins, Christopher John Cornish Hellaby. 1989. Learning from delayed rewards. PhD thesis, King's College, University of Cambridge .
- Wei, Chen-Yu, Haipeng Luo. 2021. Non-stationary reinforcement learning without prior knowledge: An optimal black-box approach. arXiv preprint arXiv:2102.05406 .
- Yadkori, Yasin Abbasi, Peter L Bartlett, Varun Kanade, Yevgeny Seldin, Csaba Szepesv´ ari. 2013. Online learning in Markov decision processes with adversarially chosen transition probability distributions. Advances in Neural Information Processing Systems . 2508-2516.
- Yu, Jia Yuan, Shie Mannor. 2009. Online learning in Markov decision processes with arbitrarily changing rewards and transitions. International Conference on Game Theory for Networks . IEEE, 314-322.
- Yu, Yueshan, Xin Chen, Fuqiang Zhang. 2015. Dynamic capacity management with general upgrading. Operations Research 63 (6) 1372-1389.
- Yuan, Hao, Qi Luo, Cong Shi. 2021. Marrying stochastic gradient descent with bandits: Learning algorithms for inventory systems with fixed costs. Management Science .
- Zhang, Huanan, Xiuli Chao, Cong Shi. 2019. Closing the gap: A learning algorithm for the lost-sales inventory system with lead times. Management Science 66 (5) 1962-1980.
- Zhang, Zihan, Yuan Zhou, Xiangyang Ji. 2020. Almost optimal model-free reinforcement learning via reference-advantage decomposition. Advances in Neural Information Processing Systems 33 .
- Zhao, Peng, Lijun Zhang, Yuan Jiang, Zhi-Hua Zhou. 2020. A simple approach for non-stationary linear bandits. International Conference on Artificial Intelligence and Statistics , vol. 2020.
- Zhou, Huozhi, Jinglin Chen, Lav R Varshney, Ashish Jagmohan. 2020a. Nonstationary reinforcement learning with linear function approximation. arXiv preprint arXiv:2010.04244 .
- Zhou, Xiang, Yi Xiong, Ningyuan Chen, Xuefeng Gao. 2020b. Regime switching bandits. arXiv preprint arXiv:2001.09390 .

## Appendix. Supplementary

## A. Applications to Sequential Transfer and Multi-Task RL

Other areas that could benefit from non-stationary RL include sequential transfer in bandit (Bastani et al. 2021) and RL (Tirinzoni et al. 2020) and multi-task RL (Brunskill and Li 2013), which in turn are conceptually related to continual RL (Kaplanis et al. 2018) and life-long RL (Abel et al. 2018). In the setting of sequential transfer/multi-task RL, the agent encounters a sequence of tasks over time with different system dynamics, and seeks to bootstrap learning by transferring knowledge from previously-solved tasks. Typical solutions in this area (Brunskill and Li 2013, Tirinzoni et al. 2020, Sun et al. 2020) need to assume that there are finitely many candidate tasks, and every task should be sufficiently different from the others 4 . Only under this assumption can the agent quickly identify the current task it is operating on, by essentially comparing the system dynamics it observes with the dynamics it has memorized for each candidate task. After identifying the current task with high confidence, the agent then invokes the policy that it learned through previous interactions with this specific task. This transfer learning paradigm in turn causes another problem-it 'cold switches' between policies that are most likely very different, which might lead to unstable and inconsistent behaviors of the agent over time. Fortunately, non-stationary RL can help alleviate both the finite-task assumption and the cold-switching problem. First, non-stationary RL algorithms do not need the candidate tasks to be sufficiently different in order to correctly identify each of them, because the algorithm itself can tolerate some variations in the task environment. There will also be no need to assume the finiteness of the candidate task set anymore, and the candidate tasks can be drawn from a continuous space. Second, since we are running the same non-stationary RL algorithm for a series of tasks, it improves its policy gradually over time, instead of cold-switching to a completely independent policy for each task. This could largely help with the unstable behavior issues.

## B. Proofs of the Technical Lemmas

## B.1. Proof of Lemma 1

Proof. For each d ∈ [ D ], define ∆ ( d ) r to be the local variation of the mean reward function within epoch d . By definition, we have ∑ D d =1 ∆ ( d ) r ≤ ∆ r . Further, for each d ∈ [ D ] and h ∈ [ H ], define ∆ ( d ) r,h to be the variation of the mean reward at step h in epoch d , i.e.,

<!-- formula-not-decoded -->

It also holds that ∑ H h =1 ∆ ( d ) r,h =∆ ( d ) r by definition. Define ∆ ( d ) p and ∆ ( d ) p,h analogously.

In the following, we will prove a stronger statement: ∣ ∣ Q k 1 ,glyph[star] h ( s, a ) -Q k 2 ,glyph[star] h ( s, a ) ∣ ∣ ≤ ∑ H h ′ = h ∆ (1) r,h ′ + H ∑ H h ′ = h ∆ (1) p,h ′ , which implies the statement of the lemma because ∑ H h ′ = h ∆ (1) r,h ′ ≤ ∆ (1) r and ∑ H h ′ = h ∆ (1) p,h ′ ≤ ∆ (1) p by definition. Our proof relies on backward induction on h . First, the statement holds for h = H because for any ( s, a ), by definition

<!-- formula-not-decoded -->

4 Needless to say, this assumption itself also to some extent contradicts the primary motivation of transfer learning. After all, we only want to transfer knowledge among tasks that are essentially similar to each other.

<!-- formula-not-decoded -->

where we have used the triangle inequality. Now suppose the statement holds for h +1; by the Bellman optimality equation,

<!-- formula-not-decoded -->

where inequality (5) holds due to a similar reasoning as in (4), and in (6) π k 1 ,glyph[star] and π k 2 ,glyph[star] denote the optimal policy in episodes k 1 and k 2 , respectively. Then by our induction hypothesis on h +1, for any s ′ ∈S ,

<!-- formula-not-decoded -->

where inequality (7) is due to the optimality of the policy π k 2 ,glyph[star] in episode k 2 over π k 1 ,glyph[star] . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (8) is by H¨ older's inequality, and (9) is by the definition of ∆ (1) p,h and by the definition of optimal Q -values that Q k 2 ,glyph[star] h +1 ( s, a ) ≤ H -h, ∀ ( s, a ) ∈S×A . Repeating a similar process gives us Q k 2 ,glyph[star] h ( s, a ) -Q k 1 ,glyph[star] h ( s, a ) ≤ ∑ H h ′ = h ∆ (1) r,h ′ + H ∑ H h ′ = h ∆ (1) p,h ′ . This completes our proof.

## B.2. Proof of Lemma 2

Proof. It should be clear from the way we update Q h ( s, a ) that Q k h ( s, a ) is monotonically decreasing in k . We now prove Q k,glyph[star] h ( s, a ) ≤ Q k +1 h ( s, a ) for all s, a, h, k by induction on k . First, it holds for k =1 by our initialization of Q h ( s, a ). For k ≥ 2, now suppose Q j,glyph[star] h ( s, a ) ≤ Q j +1 h ( s, a ) ≤ Q j h ( s, a ) for all s, a, h and 1 ≤ j ≤ k . For a fixed triple ( s, a, h ), we consider the following two cases.

Case 1: Q h ( s, a ) is updated in episode k . Then with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Inequality (10) is by the induction hypothesis that Q ˇ l i h +1 ( s ˇ l i h +1 , a ) ≥ Q ˇ l i ,glyph[star] h +1 ( s ˇ l i h +1 , a ) , ∀ a ∈ A , and hence V ˇ l i h +1 ( s ˇ l i h +1 ) ≥ V ˇ l i ,glyph[star] h +1 ( s ˇ l i h +1 ). Inequality (11) follows from the Azuma-Hoeffding inequality. (12) uses the Bellman optimality equation. Inequality (13) is by the Hoeffding's inequality that 1 ˇ n ( ∑ ˇ n i =1 r ˇ l i h ( s, a ) -ˇ r h ( s, a ) ) ≤ √ ι ˇ n with high probability, and by Lemma 1 that Q ˇ l i ,glyph[star] h ( s, a ) + b ∆ ≥ Q k,glyph[star] h ( s, a ). According to the monotonicity of Q k h ( s, a ), we know that Q k,glyph[star] h ( s, a ) ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ). In fact, we have proved the stronger statement Q k +1 h ( s, a ) ≥ Q k,glyph[star] h ( s, a ) + b ∆ that will be useful in Case 2 below.

Case 2: Q h ( s, a ) is not updated in episode k . Then there are two possibilities:

1. If Q h ( s, a ) has never been updated from episode 1 to episode k : It is easy to see that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q 1 h ( s, a ) = H -h +1 ≥ Q k,glyph[star] h ( s, a ) holds.
2. If Q h ( s, a ) has been updated at least once from episode 1 to episode k : Let j be the index of the latest episode that Q h ( s, a ) was updated. Then, from our induction hypothesis and Case 1, we know that Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) + b ∆ . Since Q h ( s, a ) has not been updated from episode j +1 to episode k , we know that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) + b ∆ ≥ Q k,glyph[star] h ( s, a ), where the last inequality holds because of Lemma 1.

A union bound over all time steps completes our proof.

## B.3. Proof of Lemma 3

Proof. This proof follows a similar structure as the proof of Lemma 2. It should be clear from the way we update Q h ( s, a ) that Q k h ( s, a ) is monotonically decreasing in k . We now prove Q k,glyph[star] h ( s, a ) -2( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) for all s, a, h, k by induction on k . First, it holds for k =1 by our initialization of Q h ( s, a ). For k ≥ 2, now suppose that Q j,glyph[star] h ( s, a ) -2( H -h +1) b ∆ ≤ Q j +1 h ( s, a ) ≤ Q j h ( s, a ) for all s, a, h and 1 ≤ j ≤ k . For a fixed triple ( s, a, h ), we consider the following two cases.

Case 1: Q h ( s, a ) is updated in episode k . Then, with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Inequality (14) is by the induction hypothesis that Q ˇ l i h +1 ( s ˇ l i h +1 , a ) ≥ Q ˇ l i ,glyph[star] h +1 ( s ˇ l i h +1 , a ) -2( H -h ) b ∆ , ∀ a ∈ A , and hence V ˇ l i h +1 ( s ˇ l i h +1 ) ≥ V ˇ l i ,glyph[star] h +1 ( s ˇ l i h +1 ) -2( H -h ) b ∆ . Inequality (15) follows from the Azuma-Hoeffding inequality. (16) uses the Bellman optimality equation. Inequality (17) is by the Hoeffding's inequality that 1 ˇ n ( ∑ ˇ n i =1 r ˇ l i h ( s, a ) -ˇ r h ( s, a ) ) ≤ √ ι ˇ n with high probability, and by Lemma 1 that Q ˇ l i ,glyph[star] h ( s, a ) ≥ Q k,glyph[star] h ( s, a ) -b ∆ . According to the monotonicity of Q k h ( s, a ), we know that Q k,glyph[star] h ( s, a ) -2( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ). In fact, we have proved the stronger statement Q k +1 h ( s, a ) ≥ Q k,glyph[star] h ( s, a ) -b ∆ -2( H -h ) b ∆ that will be useful in Case 2 below.

Case 2: Q h ( s, a ) is not updated in episode k . Then there are two possibilities:

1. If Q h ( s, a ) has never been updated from episode 1 to episode k : It is easy to see that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q 1 h ( s, a ) = H -h +1 ≥ Q k,glyph[star] h ( s, a ) -2( H -h +1) b ∆ holds.
2. If Q h ( s, a ) has been updated at least once from episode 1 to episode k : Let j be the index of the latest episode that Q h ( s, a ) was updated. Then, from our induction hypothesis and Case 1, we know that Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) -b ∆ -2( H -h ) b ∆ . Since Q h ( s, a ) has not been updated from episode j +1 to episode k , we know that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) -b ∆ -2( H -h ) b ∆ ≥ Q k,glyph[star] h ( s, a ) -2( H -h +1) b ∆ , where the last inequality holds because of Lemma 1.

A union bound over all time steps completes our proof.

## B.4. Proof of Proposition 2

In the following, we will bound each term in Λ k h +1 separately in a series of lemmas.

Lemma 5. With probability 1 , we have that

<!-- formula-not-decoded -->

Proof. First, by the definition of b ∆ , it is easy to see that ∑ H h =1 ∑ K k =1 (1 + 1 H ) h -1 5 b ∆ ≤ ∑ H h =1 ∑ K k =1 O (∆ (1) r + H ∆ (1) p ) ≤ O ( KH ∆ (1) r + KH 2 ∆ (1) p ). Recall our definition that e 1 = H and e i +1 = ⌊ (1 + 1 H ) e i ⌋ , i ≥ 1. For a fixed h ∈ [ H ], since H 2 ≥ 1,

<!-- formula-not-decoded -->

where w ( s, a, j ) def = ∑ K k =1 ✶ [ ( s k h , a k h ) = ( s, a ) , ˇ N k h ( s k h , a k h ) = e j ] , and w ( s, a ) def = ∑ j ≥ 1 w ( s, a, j ). We then know that ∑ s,a w ( s, a ) = K . For a fixed ( s, a ), let us now find an upper bound of j , denoted as J . Since each stage is (1 + 1 H ) times longer than the previous stage, we know for 1 ≤ j ≤ J , w ( s, a, j ) = ∑ K k =1 ✶ [ ( s k h , a k h ) = ( s, a ) , ˇ N k h ( s k h , a k h ) = e j ] = ⌊ (1 + 1 H ) e j ⌋ . From ∑ J j =1 w ( s, a, j ) = w ( s, a ), we get e J ≤ (1 + 1 H ) J -1 ≤ 10 1+ 1 H w ( s,a ) H . Therefore,

<!-- formula-not-decoded -->

Finally, by the Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

Combining the bounds for b k h and b ∆ completes the proof.

Lemma 6. With probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

where the last inequality follows from Lemma 1 and the definition of b ∆ . From the proof of Lemma 5, we know that the first term can be bounded as

<!-- formula-not-decoded -->

Further, the second term is bounded by the Azuma-Hoeffding inequality as

<!-- formula-not-decoded -->

Combining the two terms completes the proof.

Lemma 7. With probability at least 1 -( KH +1) δ , it holds that

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

where the last step is by the fact that V ˇ l i h +1 ( s k h , a k h ) ≥ V ˇ l i ,glyph[star] h +1 ( s k h , a k h ) from Lemma 2, and then by H¨ older's inequality and the triangle inequality. The following proof is analogous to the proof of Lemma 15 in Zhang et al. (2020). For completeness we reproduce it here. We have

<!-- formula-not-decoded -->

where (19) holds because ˇ l k h,i ( s k h , a k h ) = j if and only if j is in the previous stage of k and ( s k h , a k h ) = ( s j h , a j h ). For simplicity of notations, we define θ k h +1 def =(1+ 1 H ) h -1 ∑ K j =1 1 ˇ n j h ∑ ˇ n j h i =1 ✶ [ ˇ l j h,i = k ] . Then, we further have (note that we have swapped the notation of j and k )

<!-- formula-not-decoded -->

For ( k, h ) ∈ [ K ] × [ H ], let x k h denote the number of occurrences of the triple ( s k h , a k h , h ) in the current stage. Define also ˜ θ k h +1 def =(1+ 1 H ) h -1 glyph[floorleft] (1+ 1 H ) x k h glyph[floorright] x k h ≤ 3. Define K def = { ( k, h ) : θ k h +1 = ˜ θ k h +1 } , and ¯ K def = { ( k, h ) ∈ [ K ] × [ H ] : θ k h +1 = ˜ θ k h +1 } . Then, we have that glyph[negationslash]

<!-- formula-not-decoded -->

Since ˜ θ k h +1 is independent of s k h +1 , by the Azuma-Hoeffding inequality, it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

It is easy to see that if k is in a stage that is before the second last stage of the triple ( s k h , a k h , h ), then ( k, h ) ∈K . For a triple ( s, a, h ), define K ⊥ h ( s, a ) def = { k ∈ [ K ] : k is in the second last stage of the triple ( s, a, h ) , ( s k h , a k h ) = ( s, a ) } . We have that

<!-- formula-not-decoded -->

where for a fixed triple ( s, a, h ), we have defined θ h +1 ( s, a ) def = θ k h +1 , for any k ∈K ⊥ h ( s, a ). Note that θ h +1 ( s, a ) is well-defined, because θ k 1 h +1 = θ k 2 h +1 , ∀ k 1 , k 2 ∈K ⊥ h ( s, a ). Similarly, let ˜ θ h +1 ( s, a ) def = ˜ θ k h +1 for any k ∈K ⊥ h ( s, a ),

and ˜ θ h +1 ( s, a ) is also well-defined. By the Azuma-Hoeffding inequality and a union bound, it holds with probability at least 1 -KHδ that

<!-- formula-not-decoded -->

where ˇ N K +1 h ( s, a ) is defined to be the total number of visitations to the triple ( s, a, h ) over the entire K episodes. (22) is by the Cauchy-Schwartz inequality. (23) holds because, by the way stages are defined, for each triple ( s, a, h ), the length of its last two stages is at most an O (1 /H ) fraction of the total number of visitations.

Combining (18), (20) and (23) completes the proof.

## C. Proof of Theorem 1

We introduce a few terms to facilitate the analysis. Denote by s k h and a k h respectively the state and action taken at step h of episode k . Let N k h ( s, a ) , ˇ N k h ( s, a ) , Q k h ( s, a ) and V k h ( s ) denote, respectively, the values of N h ( s, a ) , ˇ N h ( s, a ) , Q h ( s, a ) and V h ( s ) at the beginning of the k -th episode in Algorithm 1. Further, for the triple ( s k h , a k h , h ), let n k h be the total number of episodes that this triple has been visited prior to the current stage, and let l k h,i denote the index of the episode that this triple was visited the i -th time among the total n k h times. Similarly, let ˇ n k h denote the number of visits to the triple ( s k h , a k h , h ) in the stage right before the current stage, and let ˇ l k h,i be the i -th episode among the ˇ n k h episodes right before the current stage. For simplicity, we use l i and ˇ l i to denote l k h,i and ˇ l k h,i , and ˇ n to denote ˇ n k h , when h and k are clear from the context. We also use ˇ r h ( s, a ) and ˇ v h ( s, a ) to denote the values of ˇ r h ( s k h , a k h ) and ˇ v h ( s k h , a k h ) when updating the Q h ( s k h , a k h ) value in Line 16 of Algorithm 1.

We now proceed to analyze the dynamic regret in one epoch, and at the very end of Appendix C, we will see how to combine the dynamic regret over all the epochs to prove Theorem 1. The following analysis will be conditioned on the successful event of Lemma 2.

The dynamic regret of Algorithm 1 in epoch d =1 can hence be expressed as

<!-- formula-not-decoded -->

From the update rules of the value functions in Algorithm 1, we have

<!-- formula-not-decoded -->

For ease of exposition, we define the following terms:

<!-- formula-not-decoded -->

We further define ˜ r k h ( s k h , a k h ) def = ˇ r h ( s k h ,a k h ) ˇ N k h ( s k h ,a k h ) -r k h ( s k h , a k h ). Then by the Hoeffding's inequality, it holds with high probability that

<!-- formula-not-decoded -->

By the Bellman equation V k,π h ( s k h ) = Q k,π h ( s k h , π ( s k h )) = r k h ( s k h , a k h ) + P k h V k,π h +1 ( s k h , a k h ), we have

<!-- formula-not-decoded -->

where (27) is by the Azuma-Hoeffding inequality and by (26). In the following, we bound each term in (28) separately. First, by H¨ older's inequality, we have

<!-- formula-not-decoded -->

Let e j denote a standard basis vector of proper dimensions that has a 1 at the j -th entry and 0s at the others, in the form of (0 , . . . , 0 , 1 , 0 , . . . , 0). Recall the definition of δ k h in (25), and we have

<!-- formula-not-decoded -->

Finally, recalling the definition of ζ k h in (25), we have that

<!-- formula-not-decoded -->

where inequality (31) is by Lemma 1. Combining (28), (29), (30), and (31) leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To find an upper bound of ∑ K k =1 ζ k h , we proceed to upper bound each term on the RHS of (32) separately. First, notice that ∑ K k =1 ✶ [ n k h =0] ≤ SAH , because each fixed triple ( s, a, h ) contributes at most 1 to ∑ K k =1 ✶ [ n k h =0]. In the following, we upper bound the second term in (32). Notice that

For a fixed episode j , notice that ∑ ˇ n k h i =1 ✶ [ ˇ l k h,i = j ] ≤ 1, and that ∑ ˇ n k h i =1 ✶ [ ˇ l k h,i = j ] = 1 happens if and only if ( s k h , a k h ) = ( s j h , a j h ) and ( j, h ) lies in the previous stage of ( k, h ) with respect to the triple ( s k h , a k h , h ). Let K def = { k ∈ [ K ] : ∑ ˇ n k h i =1 ✶ [ ˇ l k h,i = j ] = 1 } ; then, we know that every element k ∈K has the same value of ˇ n k h , i.e., there exists an integer N j &gt; 0, such that ˇ n k h = N j , ∀ k ∈K . Further, by our definition of the stages, we know that |K| ≤ (1 + 1 H ) N j , because the current stage is at most (1 + 1 H ) times longer than the previous stage. Therefore, for every j , we know that

Substituting it back into (33) leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (32) and (35), we now have that

<!-- formula-not-decoded -->

where in (36) we have used the fact that δ k h +1 ≤ ζ k h +1 , which in turn is due to the optimality that V k,glyph[star] h ( s k h ) ≥ V k,π h ( s k h ). Notice that we have ζ k h on the LHS of (36) and ζ k h +1 on the RHS. By iterating (36) over h = H,H -1 , . . . , 1, we conclude that

<!-- formula-not-decoded -->

We bound ∑ H h =1 ∑ K k =1 (1 + 1 H ) h -1 Λ k h +1 in the proposition below. Its proof relies on a series of lemmas in Appendix B that upper bound each term in Λ k h +1 separately.

Proposition 2. With probability at least 1 -( KH +2) δ , it holds that

<!-- formula-not-decoded -->

Now we are ready to prove Theorem 1.

Proof. (of Theorem 1) By (24) and (37), and by replacing δ with δ KH +2 in Proposition 2, we know that the dynamic regret in epoch d =1 can be upper bounded with probability at least 1 -δ by:

<!-- formula-not-decoded -->

and this holds for every epoch d ∈ [ D ]. Suppose T =Ω( SA ∆ H 2 ); summing up the dynamic regret over all the D epochs gives us an upper bound of ˜ O ( D √ SAKH 5 + ∑ D d =1 KH ∆ ( d ) r + ∑ D d =1 KH 2 ∆ ( d ) p ). Recall the definition that ∑ D d =1 ∆ ( d ) r ≤ ∆ r , ∑ D d =1 ∆ ( d ) p ≤ ∆ p , ∆=∆ r +∆ p , and that K =Θ( T DH ). By setting D = S -1 3 A -1 3 ∆ 2 3 H -2 3 T 1 3 , the dynamic regret over the entire T steps is bounded by R ( π,M ) ≤ ˜ O ( S 1 3 A 1 3 ∆ 1 3 H 5 3 T 2 3 ) , which completes the proof.

## D. Proof Sketch of Theorem 2

Proof sketch. We only outline the difference with respect to the proof of Theorem 1. The reader should have no difficulty recovering the complete proof by following the same routine as in the proof of Theorem 1. Specifically, it suffices to investigate the steps that are involved with Lemma 2.

The dynamic regret of the new algorithm in epoch d =1 can now be expressed as

<!-- formula-not-decoded -->

where we applied the results of Lemma 3 instead of Lemma 2. The reader should bear in mind that from the new update rules of the value functions, we now have

<!-- formula-not-decoded -->

where the RHS no longer has the additional bonus term b ∆ . If we define ζ k h , ξ k h +1 , and φ k h +1 in the same way as before, the reader can easily verify that all the derivations until Equation (37) still holds, although the value of Λ k h +1 should be re-defined as Λ k h +1 def = ξ k h +1 + φ k h +1 +3 b k h +3 b ∆ due to the new upper bound in (40) that is independent of b ∆ . Proposition 2 also follows analogously though some additional attention should be paid to the proof of Lemma 7 where the results of Lemma 2 have been utilized. Finally, we obtain the dynamic regret upper bound in epoch d =1 as follows:

<!-- formula-not-decoded -->

where the additional term 2 KHb ∆ comes from (39). From our definition of b ∆ , we can easily see that 2 KHb ∆ ≤ O ( KH ∆ (1) r + KH 2 ∆ (1) p ). Therefore, we can conclude that the dynamic regret upper bound in one epoch remains the same order, which leaves the dynamic regret over the entire horizon also unchanged.

## E. Proof of Theorem 3

Similar to the proofs of Theorems 1 and 2, we start with the dynamic regret in one epoch, and then extend to all epochs in the end. The proof follows the same routine as in the proofs of Theorems 1 and 2. Given that a rigorous analysis on the Freedman-based bonus with variance reduction is present in Zhang et al. (2020), one should not find it difficult to extend our Hoeffding-based algorithm to a Freedman-based one. Therefore,

rather than providing a complete proof of Theorem 3, in the following, we sketch the differences and highlight the additional analysis needed that is not covered by the proof of Theorem 2 and Zhang et al. (2020).

To facilitate the analysis, first recall a few notations N k h , ˇ N k h , Q k h ( s, a ) , V k h ( s ) , n k h , l k h,i , ˇ n k h , ˇ l k h,i , l i and ˇ l i that we have defined in Appendix C. In addition, when ( h,k ) is clear from the context, we drop the time indices and simply use ˇ µ, ˇ σ,µ ref , σ ref to denote their corresponding values in the computation of the Q h ( s k h , a k h ) value in Line 16 of Algorithm 1.

We start with the following lemma, which is an analogue of Lemma 3 but requires a more careful treatment of variations accumulated in µ ref and ˇ µ h . It states that the optimistic Q k h ( s, a ) is an 'upper bound' of the optimal Q k,glyph[star] h ( s, a ) subject to an error term of the order 2( H -h +1) b ∆ with high probability.

Lemma 8. (Freedman, no local budgets) For δ ∈ (0 , 1) , with probability at least 1 -2 KHδ , it holds that Q k,glyph[star] h ( s, a ) -4( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ) , ∀ ( s, a, h, k ) ∈S ×A× [ H ] × [ K ] .

Proof. It should be clear from the way we update Q h ( s, a ) that Q k h ( s, a ) is monotonically decreasing in k . We now prove Q k,glyph[star] h ( s, a ) -4( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) for all s, a, h, k by induction on k . First, it holds for k =1 by our initialization of Q h ( s, a ). For k ≥ 2, now suppose Q j,glyph[star] h ( s, a ) -4( H -h +1) b ∆ ≤ Q j h ( s, a ) for all s, a, h and 1 ≤ j ≤ k . For a fixed triple ( s, a, h ), we consider the following two cases.

Case 1: Q h ( s, a ) is updated in episode k . Notice that it suffices to analyze the case where Q h ( s, a ) is updated using b k h , because the other case of b k h would be exactly the same as in Lemma 3. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

In the following, we will bound each term in (42) separately. First, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (43) ≥-b ∆ and (44) ≥-b ∆ by H¨ older's inequality and the definition of b ∆ . In (45), we have that 1 n ∑ n i =1 P k h V ref ,l i h +1 ( s, a ) -1 ˇ n ∑ ˇ n i =1 P k h V ref , ˇ l i h +1 ( s, a ) ≥ 0, because V ref ,k h +1 ( s ) is non-increasing in k .

Following a similar procedure as in Lemma 10, Lemma 12, and Lemma 13 in Zhang et al. (2020), we can further bound | χ 1 | and | χ 2 | as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ν ref def = σ ref n -( µ ref n ) 2 and ˇ ν def = ˇ σ ˇ n -( ˇ µ ˇ n ) 2 . These are the steps where Freedman's inequality (Freedman 1975) come into use, and we omit these steps since they are essentially the same as the derivations in Zhang et al. (2020). We can see from (47), (48), and the definition of b k h that | χ 1 | + | χ 2 | ≤ b k h .

Substituting the results on χ 1 , χ 2 and χ 3 back to (42), it holds that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in (49) we used (46), (47), (48), and the definition of b k h in Algorithm 1. (50) is by the induction hypothesis that Q ˇ l i h +1 ( s ˇ l i h +1 , a ) ≥ Q ˇ l i ,glyph[star] h +1 ( s ˇ l i h +1 , a ) -2( H -h ) b ∆ , ∀ a ∈A , 1 ≤ ˇ l i ≤ k . The second to last inequality holds due to the Hoeffding's inequality that 1 ˇ n ( ∑ ˇ n i =1 r ˇ l i h ( s, a ) -ˇ r h ( s, a ) ) ≤ √ ι ˇ n ≤ b k h with high probability. Finally, the last inequality follows from Lemma 1.

According to the monotonicity of Q k h ( s, a ), we can conclude from (51) that Q k,glyph[star] h ( s, a ) -4( H -h +1) b ∆ ≤ Q k +1 h ( s, a ) ≤ Q k h ( s, a ). In fact, we have proved the stronger statement Q k +1 h ( s, a ) ≥ Q k,glyph[star] h ( s, a ) -4( H -h + 1) b ∆ + b ∆ that will be useful in Case 2 below.

Case 2: Q h ( s, a ) is not updated in episode k . Then, there are two possibilities:

1. If Q h ( s, a ) has never been updated from episode 1 to episode k : It is easy to see that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q 1 h ( s, a ) = H -h +1 ≥ Q k,glyph[star] h ( s, a ) holds.
2. If Q h ( s, a ) has been updated at least once from episode 1 to episode k : Let j be the index of the latest episode that Q h ( s, a ) was updated. Then, from our induction hypothesis and Case 1, we know that Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) -4( H -h +1) b ∆ + b ∆ . Since Q h ( s, a ) has not been updated from episode j +1 to episode k , we know that Q k +1 h ( s, a ) = Q k h ( s, a ) = · · · = Q j +1 h ( s, a ) ≥ Q j,glyph[star] h ( s, a ) -4( H -h +1) b ∆ + b ∆ ≥ Q k,glyph[star] h ( s, a ) -4( H -h +1) b ∆ , where the last inequality holds because of Lemma 1.

A union bound over all time steps completes our proof.

Conditional on the successful event of Lemma 8, the dynamic regret of RestartQ-UCB Freedman in epoch d =1 can hence be expressed as

<!-- formula-not-decoded -->

From the update rules of the value functions in Algorithm 1, we have

<!-- formula-not-decoded -->

If we again define ζ k h def = V k h ( s k h ) -V k,π h ( s k h ), we can follow a similar routine as in the proof of Theorem 1 (details can be found in Zhang et al. (2020)) and obtain

<!-- formula-not-decoded -->

where Λ k h +1 def = ψ k h +1 + ξ k h +1 + φ k h +1 +4 b k h +4 b ∆ with the following definitions:

<!-- formula-not-decoded -->

An upper bound on the first four terms in Λ k h +1 is derived in the proof of Lemma 7 in Zhang et al. (2020) (There is an extra term of √ 1 ˇ n ι in our defnition of b k h compared to theirs, but it does not affect the leading term in the upper bound). By further recalling the definition of b ∆ , we can obtain the following lemma. Lemma 9. (Lemma 7 in Zhang et al. (2020)) With probability at least (1 -O ( H 2 T 4 δ )) , it holds that

<!-- formula-not-decoded -->

Combined with (52) and the definition of ζ k h , we obtain the dynamic regret bound in a single epoch:

<!-- formula-not-decoded -->

From our definition of b ∆ , we can easily see that KHb ∆ ≤ O ( KH ∆ (1) r + KH 2 ∆ (1) p ). Finally, suppose T is greater than a polynomial of S,A, ∆ and H , √ SAH 3 Kι would be the leading term of the dynamic regret in a single epoch. In this case, summing up the dynamic regret over all the D epochs gives us an upper bound of

<!-- formula-not-decoded -->

Recall that ∑ D d =1 ∆ ( d ) r ≤ ∆ r , ∑ D d =1 ∆ ( d ) p ≤ ∆ p , ∆ = ∆ r + ∆ p , and that K = Θ( T DH ). By setting D = S -1 3 A -1 3 ∆ 2 3 T 1 3 , the dynamic regret over the entire T steps is bounded by

<!-- formula-not-decoded -->

This completes the proof of Theorem 3.

## F. Proof of Theorem 4

First, we define D † to be the optimal candidate value in J that leads to the lowest dynamic regret. Recall that since J is a discretized set and only covers values in the range of [⌊ T SAH 2 W ⌋ , ⌊ T SAH 2 ⌋] , it might not contain the actual optimal value D glyph[star] = S -1 3 A -1 3 ∆ 2 3 T 1 3 for the number of epochs D . Further, let R i ( D ) be the cumulative reward collected in phase i due to choosing the value D for the number of total epochs. Then, the dynamic regret of Algorithm 2 can be decomposed into two parts:

<!-- formula-not-decoded -->

where the first term is the dynamic regret of using the optimal candidate value D † of the number of epochs, and the second term is caused by the regret of learning the optimal candidate value using the Exp3.P algorithm. Applying the regret bound of the Exp3.P algorithm (Auer et al. (2002)), for any choice of D † , the second term in (54) is upper bounded by

<!-- formula-not-decoded -->

where in the last step we used that W = √ HT .

From the proof of Theorem 3 (e.g., Equation (53) with the fact that K =Θ( T DH )), and applying the Azuma-Hoeffding inequality and a union bound, we can upper bound the first term in (54) by

<!-- formula-not-decoded -->

To derive a further upper bound of (56), we need to distinguish between two cases: Whether D glyph[star] is covered in the range of J or not. Since we have assumed that the horizon is sufficiently long, i.e., T is greater than some polynomial of S,A, ∆ and H , it holds that D glyph[star] = S -1 3 A -1 3 ∆ 2 3 T 1 3 ≤ ⌊ T SAH 2 ⌋ . Therefore, to determine whether D glyph[star] is covered in the range of J , we only need to compare D glyph[star] with the lower bound ⌊ T SAH 2 W ⌋ in J .

- If D glyph[star] is covered in the range of J , i.e., D glyph[star] ≥ ⌊ T SAH 2 W ⌋ : Since J is discretized in a way that two consecutive values differ from each other by a factor of at most W 1 /J , we know that there exists a value D † ∈J , such that D glyph[star] ≤ D † ≤ W 1 /J D glyph[star] . In this case, we can upper bound the RHS of (56) by

<!-- formula-not-decoded -->

where in the last step we used the facts that D glyph[star] = S -1 3 A -1 3 ∆ 2 3 T 1 3 and that W 1 /J = W 1 / glyph[ceilingleft] ln W glyph[ceilingright] ≤ exp(1).

- If D glyph[star] is not covered in the range of J , i.e., D glyph[star] &lt; ⌊ T SAH 2 W ⌋ : Since D glyph[star] = S -1 3 A -1 3 ∆ 2 3 T 1 3 &lt; ⌊ T SAH 2 W ⌋ , it implies that ∆ &lt;S -1 A -1 H -15 4 T 1 4 . The optimal candidate value in J would be the smallest one, and hence D † = ⌊ T SAH 2 W ⌋ . In this case, we can upper bound the RHS of (56) by

<!-- formula-not-decoded -->

where in the last step we used that ∆ &lt;S -1 A -1 H -15 4 T 1 4 and W = HT .

Combining the above two cases with (54), (55), and (56), we can conclude that the dynamic regret of Algorithm 2 is upper bounded by

<!-- formula-not-decoded -->

This completes the proof of Theorem 4.

## G. Proof of Theorem 5

The proof of our lower bound relies on the construction of a 'hard instance' of non-stationary MDPs. The instance we construct is essentially a switching-MDP: an MDP with piecewise constant dynamics on each segment of the horizon, and its dynamics experience an abrupt change at the beginning of each new segment. More specifically, we divide the horizon T into L segments 5 , where each segment has T 0 def = ⌊ T L ⌋ steps and contains M 0 def = ⌊ M L ⌋ episodes, each episode having a length of H . Within each such segment, the system dynamics of the MDP do not vary, and we construct the dynamics for each segment in a way such that the instance is a hard instance of stationary MDPs on its own. The MDP within each segment is essentially similar to the hard instances constructed in stationary RL problems (Osband and Van Roy 2016, Jin et al. 2018). Between two consecutive segments, the dynamics of the MDP change abruptly, and we let the dynamics vary in a way such that no information learned from previous interactions with the MDP can be used in the new segment. In this sense, the agent needs to learn a new hard stationary MDP in each segment. Finally, optimizing the value of L and the variation magnitude between consecutive segments (subject to the constraints of the total variation budget) leads to our lower bound.

We start with a simplified episodic setting where the transition kernels and reward functions are held constant within each episode, i.e., P m 1 = · · · = P m h = . . . P m H and r m 1 = · · · = r m h = . . . r m H , ∀ m ∈ [ M ]. This is a popular but less challenging episodic setting, and its stationary counterpart has been studied in Azar et al. (2017). We further require that when the environment varies due to the non-stationarity, all steps in one episode should vary simultaneously in the same way. This simplified setting is easier to analyze, and its analysis conveniently leads to a lower bound for the un-discounted setting as a side result along the way. Later we will show how the analysis can be naturally extended to the more general setting we introduced in Section 2, using techniques that have also been utilized in Jin et al. (2018). For simplicity of notations, we temporarily drop the h indices and use P m and r m to denote the transition kernel and reward function whenever there is no ambiguity.

Consider a two-state MDP as depicted in Figure 3. This MDP was initially proposed in Jaksch et al. (2010) as a hard instance of stationary MDPs, and following Jin et al. (2018) we will refer to this construction as the 'JAO MDP'. This MDP has 2 states S = { s ◦ , s } and SA actions A = { 1 , 2 , . . . , SA } . The reward does not depend on actions: state s always gives reward 1 whatever action is taken, and state s ◦ always gives reward 0. Any action taken at state s takes the agent to state s ◦ with probability δ , and to state s with probability 1 -δ . At state s ◦ , for all but a single 'good' action a glyph[star] , the agent is taken to state s with probability δ , and for the good action a glyph[star] , the agent is taken to state s with probability δ + ε for some 0 &lt;ε&lt;δ . The exact values of

Figure 3 The 'JAO MDP' constructed in Jaksch et al. (2010). Dashed lines denote transitions related to the good action a glyph[star] .

<!-- image -->

Figure 4 A chain with H copies of JAO MDPs correlated in time. At the end of an episode, the state should deterministically transition from any state in the last copy to the s ◦ state in the first copy of the chain, the arrows of which are not shown in the figure. Also, the s state in the first copy is actually never reached and is redundant.

<!-- image -->

δ and ε will be chosen later. Note that this is not an MDP with S states and A actions as we desire, but the extension to an MDP with S states and A actions is routine (Jaksch et al. 2010), and is hence omitted here.

To apply the JAO MDP to the simplified episodic setting, we 'concatenate' H copies of exactly the same JAO MDP into a chain as depicted in Figure 4, denoting the H steps in an episode. The initial state of this MDP is the s ◦ state in the first copy of the chain, and after each episode the state is 'reset' to the initial state. In the following, we first show that the constructed MDP is a hard instance of stationary MDPs, without worrying about the evolution of the system dynamics. The techniques that we will be using are essentially the same as in the proofs of the lower bound in the multi-armed bandit problem (Auer et al. 2002) or the reinforcement learning problem in the un-discounted setting (Jaksch et al. 2010).

The good action a glyph[star] is chosen uniformly at random from the action space A , and we use E glyph[star] [ · ] to denote the expectation with respect to the random choice of a glyph[star] . We write E a [ · ] for the expectation conditioned on action a being the good action a glyph[star] . Finally, we use E unif [ · ] to denote the expectation when there is no good action in the MDP, i.e., every action in A takes the agent from state s ◦ to s with probability δ . Define the probability notations P glyph[star] ( · ) , P a ( · ), and P unif ( · ) analogously.

Consider running a reinforcement learning algorithm on the constructed MDP for T 0 steps, where T 0 = M 0 H . It has been shown in Auer et al. (2002) and Jaksch et al. (2010) that it is sufficient to consider deterministic

5 The definition of segments is irrelevant to, and should not be confused with, the notion of epochs we previously defined.

policies. Therefore, we assume that the algorithm maps deterministically from a sequence of observations to an action a t at time t . Define the random variables N,N ◦ and N glyph[star] ◦ to be the total number of visits to state s , the total number of visits to s ◦ , and the total number of times that a glyph[star] is taken at state s ◦ , respectively. Let s t denote the state observed at time t , and a t the action taken at time t . When there is no chance of ambiguity, we sometimes also use s m h to denote the state at step h of episode m , which should be interpreted as the state s t observed at time t =( m -1) × H + h . The notation a m h is used analogously. Since s ◦ is assumed to be the initial state, we have that glyph[negationslash]

<!-- formula-not-decoded -->

and rearranging the last inequality gives us E a [ N ] ≤ E a [ N ◦ -N glyph[star] ◦ ] + (1 + ε δ ) E a [ N glyph[star] ◦ ].

For this proof only, define the random variable W ( T 0 ) to be the total reward of the algorithm over the horizon T 0 , and define G ( T 0 ) to be the (static) regret with respect to the optimal policy. Since for any algorithm, the probability of staying in state s ◦ under P a ( · ) is no larger than under P unif ( · ), it follows that

<!-- formula-not-decoded -->

Let τ m ◦ denote the first step that the state transits from state s ◦ to s in the m -th episode; then

<!-- formula-not-decoded -->

Since the algorithm is a deterministic mapping from the observation sequence to an action, the random variable N glyph[star] ◦ is also a function of the observations up to time T . In addition, since the immediate reward only depends on the current state, N glyph[star] ◦ can further be considered as a function of just the state sequence up to T . Therefore, the following lemma from Jaksch et al. (2010), which in turn was adapted from Lemma A.1 in Auer et al. (2002), also applies in our setting.

Lemma 10. (Lemma 13 in Jaksch et al. (2010)) For any finite constant B , let f : { s ◦ , s } T 0 +1 → [0 , B ] be any function defined on the state sequence s ∈{ s ◦ , s } T 0 +1 . Then, for any 0 &lt;δ ≤ 1 2 , any 0 &lt;ε ≤ 1 -2 δ , and any a ∈A , it holds that

<!-- formula-not-decoded -->

Since N glyph[star] ◦ itself is a function from the state sequence to [0 , T 0 ], we can apply Lemma 10 and arrive at

<!-- formula-not-decoded -->

From (58), we have that ∑ SA a =1 E unif [ N glyph[star] ◦ ] = T 0 -E unif [ N ] ≤ T 0 2 + M 0 2 δ . By the Cauchy-Schwarz inequality, we further have that ∑ SA a =1 √ 2 E unif [ N glyph[star] ◦ ] ≤ √ SA ( T 0 + M 0 δ ). Therefore, from (59), we obtain

<!-- formula-not-decoded -->

Together with (57) and (58), it holds that

<!-- formula-not-decoded -->

## G.1. The Un-discounted Setting

Let us now momentarily deviate from the episodic setting and consider the un-discounted setting (with M 0 =1). This is the case of the JAO MDP in Figure 3 where there is not reset. We could calculate the stationary distribution and find that the optimal average reward for the JAO MDP is δ + ε 2 δ + ε . It is also easy to calculate that the diameter of the JAO MDP is D = 1 δ . Therefore, the expected (static) regret with respect to the randomness of a ∗ can be lower bounded by

<!-- formula-not-decoded -->

By assuming T 0 ≥ DSA (which in turn suggests D ≤ √ T 0 D SA ) and setting ε = c √ SA T 0 D for c = 3 40 , we further have that

<!-- formula-not-decoded -->

It is easy to verify that our choice of δ and ε satisfies our assumption that 0 &lt;ε&lt;δ . So far, we have recovered the (static) regret lower bound of Ω( √ SAT 0 D ) in the un-discounted setting, which was originally proved in Jaksch et al. (2010).

Based on this result, let us now incorporate the non-stationarity of the MDP and derive a lower bound for the dynamic regret R ( T ). Recall that we are constructing the non-stationary environment as a switching-MDP. For each segment of length T 0 , the environment is held constant, and the regret lower bound for each segment is Ω( √ SAT 0 D ). At the beginning of each new segment, we uniformly sample a new action a ∗ at random from the action space A to be the good action for the new segment. In this case, the learning algorithm cannot use the information it learned during its previous interactions with the environment, even if it knows the switching structure of the environment. Therefore, the algorithm needs to learn a new (static) MDP in each segment, which leads to a dynamic regret lower bound of Ω( L √ SAT 0 D ) = Ω( √ SATLD ), where let us recall

that L is the number of segments. Every time the good action a ∗ varies, it will cause a variation of magnitude 2 ε in the transition kernel. The constraint of the overall variation budget requires that 2 εL = 3 20 √ SA T 0 D L ≤ ∆, which in turn requires L ≤ 4∆ 2 3 T 1 3 D 1 3 S -1 3 A -1 3 . Finally, by assigning the largest possible value to L subject to the variation budget, we obtain a dynamic regret lower bound of Ω ( S 1 3 A 1 3 ∆ 1 3 D 2 3 T 2 3 ) . This completes the proof of Proposition 1.

## G.2. The Episodic Settings

Now let us go back to our simplified episodic setting, as depicted in Figure 4. One major difference with the previous un-discounted setting is that we might not have time to mix between s ◦ and s in H steps. (Note that we only need to reach the stationary distribution over the ( s ◦ , s ) pair in each step h , rather than the stationary distribution over the entire MDP. In fact, the latter case is never possible because the entire MDP is not aperiodic.) It can be shown that the optimal policy on this MDP has a mixing time of Θ ( 1 δ ) (Jin et al. 2018), and hence we can choose δ to be slightly larger than Θ( 1 H ) to guarantee sufficient time to mix. All the analysis up to inequality (60) carries over to the episodic setting, and essentially we can set δ to be Θ ( 1 H ) to get a (static) regret lower bound of Ω( √ SAT 0 H ) in each segment. Another difference with the previous setting lies in the usage of the variation budget. Since we require that all the steps in the same episode should vary simultaneously, it now takes a variation budget of 2 εH each time we switch to a new action a ∗ at the beginning of a new segment. Therefore, the overall variation budget now puts a constraint of 2 εHL ≤ O (∆) on the magnitude of each switch. Again, by choosing ε =Θ (√ SA T 0 H ) and optimizing over possible values of L subject to the budget constraint, we obtain a dynamic regret lower bound of Ω ( S 1 3 A 1 3 ∆ 1 3 H 1 3 T 2 3 ) in the simplified episodic setting.

Finally, we consider the standard episodic setting as introduced in Section 2. In this setting, we essentially will be concatenating H distinct JAO MDPs, each with an independent good action a ∗ , into a chain like Figure 4. The transition kernels in these JAO MDPs are also allowed to vary asynchronously in each step h , although our construction of the lower bound does not make use of this property. As argued similarly in Jin et al. (2018), the number of observations for each specific JAO MDP is only T 0 /H , instead of T 0 . Therefore, we can assign a slightly larger value to ε and the learning algorithm would still not be able to identify the good action given the fewer observations. Setting δ =Θ ( 1 H ) and ε =Θ (√ SA T 0 ) leads to a (static) regret lower bound of Ω( H √ SAT 0 ) in the stationary RL problem. Again, the transition kernels in all the H JAO MDPs vary simultaneously at the beginning of each new segment. By optimizing L subject to the overall budget constraint 2 εHL ≤ O (∆), we obtain a dynamic regret lower bound of Ω ( S 1 3 A 1 3 ∆ 1 3 H 2 3 T 2 3 ) in the episodic setting. This completes our proof of Theorem 5.

## H. Proof of Theorem 6

Proof. We first show that when the switching cost of agent 2 satisfies N switch = O ( T β ) for 0 &lt;β &lt; 1, the dynamic regret of agent 1 is upper bounded by ˜ O ( T β +2 3 ). To see this, notice that from the perspective of agent 1, the environment is non-stationary due to the fact that agent 2 is changing its policy over time. Since the switching cost of agent 2 is upper bounded by O ( T β ), by the definitions of ∆ r and ∆ p in Section 2, we know that the variation of the environment from the perspective of agent 1 is upper bounded by O ( T β ). Substituting the value of ∆ with O ( T β ) in Theorem 2 or Theorem 3 leads to the desired result.

From the ( λ,µ )-smoothness of the MDP, it follows that

<!-- formula-not-decoded -->

Therefore, it holds that

<!-- formula-not-decoded -->

where the last step follows from the ˜ O ( T β +2 3 ) dynamic regret bound of agent 1 as we discussed above. Rearranging the terms leads to the desired result.

## I. Proofs for Section 9

## I.1. Proof of Lemma 4

First, we show that

<!-- formula-not-decoded -->

Recall that the pseudo-reward is constructed by uniformly shifting the reward function by an amount of p · E [ X m h ]. Since the difference between the reward and the pseudo-reward does not depend on the action taken, for any realization of the demands { X m h } m ∈ [ M ] ,h ∈ [ H ] , the optimal policies π glyph[star] and π glyph[star], pseudo induce the same distribution over the action space, which in turn leads to the same distribution over state-action trajectories. We can hence conclude that (61) holds.

Similarly, one can show by induction that for any realization of the demands { X m h } m ∈ [ M ] ,h ∈ [ H ] , Algorithm 1 also induces the same distribution of action sequences on M and M pseudo . This leads us to

<!-- formula-not-decoded -->

Combining (61) and (62) yields the desired result.

## J. Simulations Setup

We compare RestartQ-UCB with three baseline algorithms: LSVI-UCB-Restart (Zhou et al. 2020a), QLearning UCB, and Epsilon-Greedy (Watkins 1989). LSVI-UCB-Restart is a state-of-the-art non-stationary RL algorithm that combines optimistic least-squares value iteration with periodic restarts. It is originally designed for non-stationary RL in linear MDPs, but in our simulations we reduce it to the tabular case by setting the feature map to be essentially an identity mapping, i.e., the feature dimension is set to be d = S × A . Q-Learning UCB is simply our RestartQ-UCB algorithm with no restart. It is a Q-learning based algorithm that uses upper confidence bounds to guide the exploration. Epsilon-Greedy is also a Q-learning based algorithm with restarts. Compared with RestartQ-UCB, Epsilon-Greedy does not employ a UCB-based

bonus term to explicitly force exploration. Instead, it takes the greedy action according to the estimated Q function with a high probability 1 -ε , and explores an action from the action set uniformly at random with probability ε .

We evaluate the cumulative rewards of the four algorithms on a variant of a reinforcement learning task named Bidirectional Diabolical Combination Lock (Agarwal et al. 2020, Misra et al. 2020). This task is designed to be particularly difficult for exploration . At the beginning of each episode, the agent starts at a fixed state. According to its first action, the agent transitions to one of the two paths, or 'combination locks', each of length H . Each path is a chain of H states, where the state at the endpoint of each path gives a high reward. At each step on the path, there is only one 'correct' action that leads the agent to the next state on the path, while the other A -1 actions lead it to a sinking state that yields a small per-step reward of 1 8 H ever since. Since we are considering a non-deterministic MDP, each intended transition 'succeeds' with probability 0 . 98; that is, even if the agent takes the correct action at a certain step, there is still a 0 . 02 probability that it will end in the sinking state. The agent obtains a 0 reward when taking a correct action, and gets a 1 8 H reward at the step when it transitions to the sinking state. Finally, the endpoint state of one path gives a reward of 1, while the other endpoint only gives a reward of 0 . 25. As argued in Agarwal et al. (2020), the following properties make this task especially challenging: First, it has sparse high rewards, and uniform exploration only has a A -H probability of reaching a high reward endpoint. Second, it has dense low rewards, and a locally optimal policy will lead to the sinking state quickly. Third, there is no indication which path has the globally optimal reward, and the agent must remember to still visit the other one. Interested readers can refer to Section 5.1 of Agarwal et al. (2020) for detailed descriptions of the task.

We introduce two types of non-stationarity to the Bidirectional Diabolical Combination Lock task, namely abrupt variations and gradual variations. For abrupt variations, we periodically switch the two high-reward endpoints: One high-reward endpoint gives a reward of 1 at the beginning, and abruptly changes to a reward of 0 . 25 after a certain number of episodes, and then switches back to the reward of 1 after the same number of episodes. The other high-reward endpoint goes the other way around. For gradual changes, we gradually vary the transition probability at the starting state: At the first episode, one action leads to the first path with 0 . 98 probability, and to the second path with 0 . 02 probability. We linearly decrease its probability of leading to the first path, and increase its probability to the second path. As a result, at the last episode, this action would lead to the first path with 0 . 02 probability, and to the second path with 0 . 98 probability instead. The same is true for the other actions.

For simplicity, we use Hoeffding-based bonus terms in the simulations for RestartQ-UCB. We set M = 5000 , H =5 , S =10, and A =2. For abrupt variations, we switch the two high-reward endpoints after every 1000 episodes. The hyper-parameters for each algorithm are optimized individually. For RestartQ-UCB, LSVI-UCB-Restart, and Epsilon-Greedy, we restart the algorithms after every 1000 episodes both for abrupt variations and gradual variations. This is the same frequency as the abrupt variation of the environment (because the restart frequency is optimized as a hyper-parameter), although it turns out that other restart frequencies lead to very similar results. For Epsilon-Greedy, we set the exploration probability to be ε =0 . 05. All results are averaged over 30 runs on a laptop with an Intel Core i5-9300H CPU and 16 GB memory.