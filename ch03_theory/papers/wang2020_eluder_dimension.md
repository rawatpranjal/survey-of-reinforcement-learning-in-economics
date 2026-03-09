## Reinforcement Learning with General Value Function Approximation: Provably Efficient Approach via Bounded Eluder Dimension

Ruosong Wang Carnegie Mellon University ruosongw@andrew.cmu.edu

Ruslan Salakhutdinov Carnegie Mellon University rsalakhu@cs.cmu.edu

Lin F. Yang University of California, Los Angles linyang@ee.ucla.edu

## Abstract

Value function approximation has demonstrated phenomenal empirical success in reinforcement learning (RL). Nevertheless, despite a handful of recent progress on developing theory for RL with linear function approximation, the understanding of general function approximation schemes largely remains missing. In this paper, we establish a provably efficient RL algorithm with general value function approximation. We show that if the value functions admit an approximation with a function class F , our algorithm achieves a regret bound of ˜ O (poly( dH ) √ T ) where d is a complexity measure of F that depends on the eluder dimension [Russo and Van Roy, 2013] and log-covering numbers, H is the planning horizon, and T is the number interactions with the environment. Our theory generalizes recent progress on RL with linear value function approximation and does not make explicit assumptions on the model of the environment. Moreover, our algorithm is model-free and provides a framework to justify the effectiveness of algorithms used in practice.

## 1 Introduction

In reinforcement learning (RL), we study how an agent maximizes the cumulative reward by interacting with an unknown environment. RL finds enormous applications in a wide variety of domains, e.g., robotics [Kober et al., 2013], education [Koedinger et al., 2013], gaming-AI [Shalev-Shwartz et al., 2016], etc. The unknown environment in RL is often modeled as a Markov decision process (MDP), in which there is a set of states S that describes all possible status of the environment. At a state s ∈ S , an agent interacts with the environment by taking an action a from an action space A . The environment then transits to another state s ′ ∈ S which is drawn from some unknown transition distribution, and the agent also receives an immediate reward. The agent interacts with the environment episodically, where each episode consists of H steps. The goal of the agent is to interact with the environment strategically such that after a certain number of interactions, sufficient information is collected so that the agent can act nearly optimally afterward. The performance of an agent is measured by the regret , which is defined as the difference between the total rewards collected by the agent and those a best possible agent would collect.

Without additional assumptions on the structure of the MDP, the best possible algorithm

achieves a regret bound of ˜ Θ( √ H |S||A| T ) 1 [Azar et al., 2017], where T is the total number of steps the agent interacts with the environment. In other words, the algorithm learns to interact with the environment nearly as well as an optimal agent after roughly H |S||A| steps. This regret bound, however, can be unacceptably large in practice. E.g., the game of Go has a state space with size 3 361 , and the state space of certain robotics applications can even be continuous. Practitioners apply function approximation schemes to tackle this issue, i.e., the value of a state-action pair is approximated by a function which is able to predict the value of unseen state-action pairs given a few training samples. The most commonly used function approximators are deep neural networks (DNN) which have achieved remarkable success in playing video games [Mnih et al., 2015], the game of Go [Silver et al., 2017], and controlling robots [Akkaya et al., 2019]. Nevertheless, despite the outstanding achievements in solving real-world problems, no convincing theoretical guarantees were known about RL with general value function approximators like DNNs.

Recently, there is a line of research trying to understand RL with simple function approximators, e.g. linear functions. For instance, given a feature extractor which maps state-action pairs to d -dimensional feature vectors, [Yang and Wang, 2019, 2020, Jin et al., 2019, Cai et al., 2020, Modi et al., 2019, Jia et al., 2019, Zanette et al., 2019, Du et al., 2019, Wang et al., 2019, Zanette et al., 2020, Du et al., 2020b] developed algorithms with regret bound proportional to poly( dH ) √ T which is independent of the size of S × A . Although being much more efficient than algorithms for the tabular setting, these algorithms require a well-designed feature extractor and also make restricted assumptions on the transition model. This severely limits the scope that these approaches can be applied to, since obtaining a good feature extractor is by no means easy and successful algorithms used in practice usually specify a function class (e.g. DNNs with a specific architecture) rather than a feature extractor. To our knowledge, the following fundamental question about RL with general function approximation remains unanswered at large:

## Does RL with general function approximation learn to interact with an unknown environment provably efficiently?

In this paper, we address the above question by developing a provably efficient (both computationally and statistically) Q -learning algorithm that works with general value function approximators. To run the algorithm, we are only required to specify a value function class, without the need for feature extractors or explicit assumptions on the transition model. Since this is the same requirement as algorithms used in practice like deep Q -learning [Mnih et al., 2015], our theoretical guarantees on the algorithm provide a justification of why practical algorithms work so well. Furthermore, we show that our algorithm enjoys a regret bound of ˜ O (poly( dH ) √ T ) where d is a complexity measure of the function class that depends on the eluder dimension [Russo and Van Roy, 2013] and log-covering numbers. Our theory also generalizes existing algorithms for linear and generalized linear function approximation and provides comparable regret bounds when applied to those settings.

## 1.1 Related Work

Tabular RL. There is a long line of research on the sample complexity and regret bound for RL in the tabular setting. See, e.g., [Kearns and Singh, 2002, Kakade, 2003, Strehl et al., 2006, 2009, Jaksch et al., 2010, Szita and Szepesv´ ari, 2010, Azar et al., 2013, Lattimore and Hutter, 2014,

1 Throughout the paper, we use ˜ O ( · ) to suppress logarithmic factors.

Dann and Brunskill, 2015, Osband and Van Roy, 2016, Osband et al., 2019, Agrawal and Jia, 2017, Azar et al., 2017, Sidford et al., 2018, Dann et al., 2019, Jin et al., 2018, Zanette and Brunskill, 2019, Zhang et al., 2020, Wang et al., 2020] and references therein. In particular, Jaksch et al. [2010] proved a tight regret lower bound Ω( √ H |S||A| T ) and Azar et al. [2017] showed the first asymptotically tight regret upper bound ˜ O ( √ H |S||A| T ). Although these algorithms achieve asymptotically tight regret bounds, they can not be applied in problems with huge state space due to the linear dependency on √ |S| in the regret bound. Moreover, the regret lower bound Ω( √ H |S||A| T ) demonstrates that without further assumptions, RL with huge state space is information-theoretically hard to solve. In this paper, we exploit the structure that the value functions lie in a function class with bounded complexity and devise an algorithm whose regret bound scales polynomially in the complexity of the function class instead of the number of states.

Bandits. Another line of research studies bandits problems with linear function approximation [Dani et al., 2008, Abbasi-Yadkori et al., 2011, Li et al., 2019]. These algorithms are later generalized to the generalized linear model [Filippi et al., 2010, Li et al., 2017]. A novel work of Russo and Van Roy [2013] studies bandits problems with general function approximation and proves that UCB-type algorithms and Thompson sampling achieve a regret bound of ˜ O ( √ dim E · log( N where dim E is the eluder dimension of the function class and N is the covering number of the function class. In this paper we study the RL setting with general value function approximation, and the regret bound of our algorithm also depends on the eluder dimension and the log-covering number of the function class. However, we would like to stress that the RL setting is much more complicated than the bandits setting, since the bandits setting is a special case of the RL setting with planning horizon H = 1 and thus there is no state transition in the bandits setting.

RL with Linear Function Approximation. Recently there has been great interest in designing and analyzing algorithms for RL with linear function approximation. See, e.g., [Yang and Wang, 2019, 2020, Jin et al., 2019, Cai et al., 2020, Modi et al., 2019, Jia et al., 2019, Zanette et al., 2019, Du et al., 2019, Wang et al., 2019, Du et al., 2020a, Zanette et al., 2020, Du et al., 2020b]. These papers design provably efficient algorithms under the assumption that there is a well-designed feature extractor available to the agent and the value function or the model can be approximated by a linear function or a generalized linear function of the feature vectors. Moreover, the algorithm in [Zanette et al., 2020] requires solving the Planning Optimization Program which could be computationally intractable. In this paper, we study RL with general function approximation in which case a feature extractor may not even be available, and our goal is to develop an efficient (both computationally and statistically) algorithm with provable regret bounds without making explicit assumptions on the model.

RL with General Function Approximation. It has been shown empirically that combining RL algorithms with neural network function approximators could lead to superior performance on various tasks [Mnih et al., 2013, Schaul et al., 2016, Wang et al., 2016, Van Hasselt et al., 2016, Silver et al., 2017, Akkaya et al., 2019]. Theoretically, Osband and Van Roy [2014] analyzed the regret bound of Thompson sampling when applied to RL with general function approximation. Compared to our result, Osband and Van Roy [2014] makes explicit model-based assumptions (the transition operator and the reward function lie in a function class) and their regret bound depends on the global Lipschitz constant. In contrast, in this paper we focus on UCB-type algorithms with

)

T

)

value-based assumptions, and our regret bound does not depend on the global Lipschitz constant. Recently, Ayoub et al. [2020] proposed an algorithm for model-based RL with general function approximation based on value-targeted regression, and the regret bound of their algorithm also depends on the eluder dimension. On the contrary, in this paper we focus on value-based RL algorithms.

Recent theoretical progress has produced provably sample efficient algorithms for RL with general value function approximation, but many of these algorithms are relatively impractical. In particular, [Jiang et al., 2017, Sun et al., 2019, Dong et al., 2020] devised algorithms whose sample complexity or regret bound can be upper bounded in terms of the Bellman rank or the witness rank. However, these algorithms are not computationally efficient. The algorithm in [Du et al., 2020b] can also be applied in RL with general function approximation. However, their algorithms require the transition of the MDP to be deterministic. There is also a line of research analyzing Approximate Dynamic Programming (ADP) in RL with general function approximation [Bertsekas and Tsitsiklis, 1996, Munos, 2003, Szepesv´ ari and Munos, 2005, Antos et al., 2008, Munos and Szepesv´ ari, 2008, Chen and Jiang, 2019]. These papers focus on the batch RL setting, and there is no exploration components in the algorithms. The sample complexity of these algorithms usually depends on the concentrability coefficient and is thus incomparable to our results.

## 2 Preliminaries

Throughout the paper, for a positive integer N , we use [ N ] to denote the set { 1 , 2 , . . . , N } .

Episodic Markov Decision Process. Let M = ( S , A , P, r, H, µ ) be a Markov decision process (MDP) where S is the state space, A is the action space with bounded size, P : S × A → ∆( S ) is the transition operator which takes a state-action pair and returns a distribution over states, r : S × A → [0 , 1] is the deterministic reward function 2 , H ∈ Z + is the planning horizon (episode length), and µ ∈ ∆( S ) is the initial state distribution.

A policy π chooses an action a ∈ A based on the current state s ∈ S and the time step h ∈ [ H ]. Formally, π = { π h } H h =1 where for each h ∈ [ H ], π h : S → A maps a given state to an action. The policy π induces a trajectory

<!-- formula-not-decoded -->

where s 1 ∼ µ , a 1 = π 1 ( s 1 ), r 1 = r ( s 1 , a 1 ), s 2 ∼ P ( s 1 , a 1 ), a 2 = π 2 ( s 2 ), etc.

<!-- formula-not-decoded -->

An important concept in RL is the Q -function. Given a policy π , a level h ∈ [ H ] and a state-action pair ( s, a ) ∈ S × A , the Q -function is defined as

Similarly, the value function of a given state s ∈ S is defined as

<!-- formula-not-decoded -->

2 We assume the reward function is deterministic only for notational convience. Our results can be readily generalized to the case that rewards are stochastic.

We use π ∗ to denote an optimal policy, i.e., π ∗ is a policy that maximizes

<!-- formula-not-decoded -->

We also denote Q ∗ h ( s, a ) = Q π ∗ h ( s, a ) and V ∗ h ( s ) = V π ∗ h ( s ).

In the episodic MDP setting, the agent aims to learn the optimal policy by interacting with the environment during a number of episodes. For each k ∈ [ K ], at the beginning of the k -th episode, the agent chooses a policy π k which induces a trajectory, based on which the agent chooses policies for later episodes. We assume K is fixed and known to the agent, though our algorithm and analysis can be readily generalized to the case that K is unknown in advance. Throughout the paper, we define T := KH to be the total number of steps that the agent interacts with the environment.

We adopt the following regret definition in this paper.

Definition 1. The regret of an algorithm A after K episodes is defined as

<!-- formula-not-decoded -->

where π k is the policy played by algorithm A at the k -th episode.

Additional Notations. For a function f : S × A → R , define

<!-- formula-not-decoded -->

Similarly, for a function v : S → R , define

<!-- formula-not-decoded -->

Given a dataset D = { ( s i , a i , q i ) } |D| i =1 ⊆ S × A × R , for a function f : S × A → R , define

For a set of state-action pairs Z ⊆ S × A , for a function f : S × A → R , define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a set of functions F ⊆ { f : S × A → R } , we define the width function of a state-action pair ( s, a ) as

<!-- formula-not-decoded -->

Our Assumptions. Our algorithm (Algorithm 1) receives a function class F ⊆ { f : S × A → [0 , H +1] } as input. We make the following assumption on the Q -functions throughout the paper.

Assumption 1. For any V : S → [0 , H ] , there exists f V ∈ F which satisfies

<!-- formula-not-decoded -->

Intuitively, Assumption 1 requires that for any V : S → [0 , H ], after applying the Bellman backup operator, the resulting function lies in the function class F . We note that Assumption 1 is very general and includes many previous assumptions as special cases. For instance, for the tabular RL setting, F can be the entire function space of S × A → [0 , H + 1]. For linear MDPs [Yang and Wang, 2019, 2020, Jin et al., 2019, Wang et al., 2019] where both the reward function r : S × A → [0 , 1] and the transition operator P : S × A → ∆( S ) are linear functions of a given feature extractor φ : S × A → R d , F can be defined as the class of linear functions with respect to φ . In practice, when F is a function class with sufficient expressive power (e.g. deep neural networks), Assumption 1 (approximately) holds. In Section 5, we consider a misspecified setting where (1) only holds approximately, and we show that our algorithm still achieves provable regret bounds in the misspecified setting.

The complexity of F determines the learning complexity of the RL problem under consideration. To characterize the complexity of F , we use the following definition of eluder dimension which was first introduced in [Russo and Van Roy, 2013] to characterize the complexity of different function classes in bandits problems.

Definition 2 (Eluder dimension) . Let ε ≥ 0 and Z = { ( s i , a i ) } n i =1 ⊆ S × A be a sequence of state-action pairs.

- A state-action pair ( s, a ) ∈ S × A is ε -dependent on Z with respect to F if any f, f ′ ∈ F satisfying ‖ f -f ′ ‖ Z ≤ ε also satisfies | f ( s, a ) -f ′ ( s, a ) | ≤ ε .
- An ( s, a ) is ε -independent of Z with respect to F if ( s, a ) is not ε -dependent on Z .
- The ε -eluder dimension dim E ( F , ε ) of a function class F is the length of the longest sequence of elements in S × A such that, for some ε ′ ≥ ε , every element is ε ′ -independent of its predecessors.

It has been shown in [Russo and Van Roy, 2013] that dim E ( F , ε ) ≤ |S||A| when S and A are finite. When F is the class of linear functions, i.e., f θ ( s, a ) = θ /latticetop φ ( s, a ) for a given feature extractor φ : S × A → R d , dim E ( F , ε ) = O ( d log(1 /ε )). When F is the class generalized linear functions of the form f θ ( s, a ) = g ( θ /latticetop φ ( s, a )) where g is an increasing continuously differentiable function, dim E ( F , ε ) = O ( dr 2 log( h/ε )) where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

In [Osband and Van Roy, 2014], it has been shown that when F is the class of quadratic functions, i.e., f Λ ( s, a ) = φ ( s, a ) /latticetop Λ φ ( s, a ) where Λ ∈ R d × d , dim E ( F , ε ) = O ( d 2 log(1 /ε )).

We further assume the function class F and the state-action pairs S × A have bounded complexity in the following sense.

Assumption 2. For any ε &gt; 0 , the following holds:

1. there exists an ε -cover C ( F , ε ) ⊆ F with size |C ( F , ε ) | ≤ N ( F , ε ) , such that for any f ∈ F , there exists f ′ ∈ C ( F , ε ) with ‖ f -f ′ ‖ ∞ ≤ ε ;
2. there exists an ε -cover C ( S × A , ε ) with size |C ( S × A , ε ) | ≤ N ( S × A , ε ) , such that for any ( s, a ) ∈ S × A , there exists ( s ′ , a ′ ) ∈ C ( S × A , ε ) with max f ∈F | f ( s, a ) -f ( s ′ , a ′ ) | ≤ ε .

Assumption 2 requires both the function class F and the state-action pairs S×A have bounded covering numbers. Since our regret bound depends logarithmically on N ( F , · ) and N ( S × A , · ), it is acceptable for the covers to have exponential size. In particular, when S and A are finite, it is clear that log N ( F , ε ) = ˜ O ( |S||A| ) and log N ( S×A , ε ) = log( |S||A| ). For the case of d -dimensional linear functions and generalized linear functions, log N ( F , ε ) = ˜ O ( d ) and log N ( S × A , ε ) = ˜ O ( d ). For quadratic functions, log N ( F , ε ) = ˜ O ( d 2 ) and log N ( S × A , ε ) = ˜ O ( d ).

## 3 Algorithm

Overview. The full algorithm is formally presented in Algorithm 1. From a high-level point of view, our algorithm resembles least-square value iteration (LSVI) and falls in a similar framework as the algorithm in [Jin et al., 2019, Wang et al., 2019]. At the beginning of each episode k ∈ [ K ], we maintain a replay buffer { ( s τ h , a τ h , r τ h ) } ( h,τ ) ∈ [ H ] × [ k -1] which contains all existing samples. We set Q k H +1 = 0, and calculate Q k H , Q k H -1 , . . . , Q k 1 iteratively as follows. For each h = H,H -1 , . . . , 1,

<!-- formula-not-decoded -->

and define

Here, b k h ( · , · ) is a bonus function to be defined shortly. The above equation optimizes a least squares objective to estimate the next step value. We then play the greedy policy with respect to Q k h to collect data for the k -th episode. The above procedure is repeated until all the K episodes are completed.

<!-- formula-not-decoded -->

Stable Upper-Confidence Bonus Function. With more collected data, the least squares predictor is expected to return a better approximate the true Q -function. To encourage exploration, we carefully design a bonus function b k h ( · , · ) which guarantees that, with high probability, Q k h +1 ( s, a ) is an overestimate of the one-step backup. The bonus function b k h ( · , · ) is guaranteed to tightly characterize the estimation error of the one-step backup

<!-- formula-not-decoded -->

## Algorithm 1 F -LSVI ( δ )

- 1: Input : failure probability δ ∈ (0 , 1) and number of episodes K 2: for episode k = 1 , 2 , . . . , K do 3: Receive initial state s k 1 ∼ µ 4: Q k H +1 ( · , · ) ← 0 and V k H +1 ( · ) ← 0 5: Z k ← { ( s τ h ′ , a τ h ′ ) } ( τ,h ′ ) ∈ [ k -1] × [ H ] 6: for h = H,... , 1 do 7: D k h ← {( s τ h ′ , a τ h ′ , r τ h ′ + V k h +1 ( s τ h ′ +1 , a ) )} ( τ,h ′ ) ∈ [ k -1] × [ H ] 8: f k h ← arg min f ∈F ‖ f ‖ 2 D k h 9: b k h ( · , · ) ← Bonus ( F , f k h , Z k , δ ) (Algorithm 3) 10: Q k h ( · , · ) ← min { f k h ( · , · ) + b k h ( · , · ) , H } and V k h ( · ) = max a ∈A Q k h ( · , a ) 11: π k h ( · ) ← arg max a ∈A Q k h ( · , a ) 12: for h = 1 , 2 , . . . , H do 13: Take action a k h ← π k h ( s k h ) and observe s k h +1 ∼ P ( · | s k h , a k h ) and r k h = r ( s k h , a k h )

where

<!-- formula-not-decoded -->

is the value function of the next step. The bonus function b k h ( · , · ) is designed by carefully prioritizing important data and hence is stable even when the replay buffer has large cardinality. A detailed explanation and implementation of b k h ( · , · ) is provided in Section 3.1.

## 3.1 Stable UCB via Importance Sampling

In this section, we formally define the bonus function b k h ( · , · ) used in Algorithm 1. The bonus function is designed to estimate the confidence interval of our estimate of the Q -function. In our algorithm, we define the bonus function to be the width function b k h ( · , · ) = w ( F k h , · , · ) where the confidence region F k h is defined so that

<!-- formula-not-decoded -->

with high probability. By definition of the width function, b k h ( · , · ) gives an upper bound on the confidence interval of the estimate of the Q -function, since the width function maximizes the difference between all pairs of Q -functions that lie in the confidence region. We note that similar ideas have been applied in the bandit literature [Russo and Van Roy, 2013], in reinforcement learning with linear function approximation [Du et al., 2019] and in reinforcement learning with general function apprximation in deterministic systems [Du et al., 2020b].

To define the confidence region F k h , a natural definition would be where β is defined so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 Sensitivity-Sampling ( F , Z , λ , ε , δ )

- 2: Initialize Z ′ ←{}
- 1: Input : function class F , set of state-action pairs Z ⊆ S ×A , accuracy parameters λ, ε &gt; 0 and failure probability δ ∈ (0 , 1)
- 3: For each z ∈ Z , let p z to be smallest real number such that 1 /p z is an integer and
- 4: For each z ∈ Z , independently add 1 /p z copies of z into Z ′ with probability p z

<!-- formula-not-decoded -->

- 5: return Z ′

with high probability, and recall that Z k = { ( s τ h ′ , a τ h ′ ) } ( τ,h ′ ) ∈ [ k -1] × [ H ] is the set of state-action pairs defined in Line 5. However, as one can observe, the complexity of such a bonus function could be extremely high as it is defined by a dataset Z k whose size can be as large as T = KH . A high-complexity bonus function could potentially introduce instability issues in the algorithm. Technically, we require a stable bonus function to allow for highly concentrated estimate of the one-step backup so that the confidence region F k h is accurate even for bounded β . Our strategy to 'stabilize' the bonus function is to reduce the size of the dataset by importance sampling, so that only important state-action pairs are kept and those unimportant ones (which potentially induce instability) are ignored. Another benefit of reducing the size of the dataset is that it leads to superior computational complexity when evaluating the bonus function in practice. In later part of this section, we introduce an approach to estimate the importance of each state-action pair and a corresponding sampling method based on that. Finally, we note that importance sampling has also been applied in practical RL systems. For instance, in prioritized experience replay [Schaul et al., 2016], the importance is measured by the TD error.

Sensitivity Sampling. Here we present a framework to subsample a given dataset, so that the confidence region is approximately preserved while the size of the dataset is greatly reduced. Our framework is built upon the sensitivity sampling technique introduced in [Langberg and Schulman, 2010, Feldman and Langberg, 2011, Feldman et al., 2013].

Definition 3. For a given set of state-action pairs Z ⊆ S × A and a function class F , for each z ∈ Z , define the λ -sensitivity of ( s, a ) with respect to Z and F to be

<!-- formula-not-decoded -->

Sensitivity measures the importance of each data point z in Z by considering the pair of functions f, f ′ ∈ F such that z contributes the most to ‖ f -f ′ ‖ 2 Z . In Algorithm 2, we define a procedure to sample each state-action pair with sampling probability proportional to the sensitivity. In this analysis, we show that after applying Algorithm 2 on the input dataset Z , with high probability, the confidence region { f ∈ F | ‖ f -f k h ‖ 2 Z ≤ β } is approximately preserved, while the size of the subsampled dataset is upper bounded by the eluder dimension of F times the log-covering number of F .

## Algorithm 3 Bonus ( F , ¯ f , Z , δ )

- 2: Z ′ ← Sensitivity-Sampling ( F , Z , δ/ (16 T ) , 1 / 2 , δ ) /triangleright Subsample the dataset 3: Z ′ ←{} if |Z ′ | ≥ 4 T/δ or the number of distinct elements in Z ′ exceeds
- 1: Input : function class F , reference function ¯ f ∈ F , state-action pairs Z ⊆ S × A and failure probability δ ∈ (0 , 1)

<!-- formula-not-decoded -->

/triangleright

Round

¯

f

/triangleright Round state-action pairs

<!-- formula-not-decoded -->

- ̂
- 6: for z ∈ Z ′ do

<!-- formula-not-decoded -->

- 7: Let ̂ z ∈ C ( S × A , 1 / (8 √ 4 T/δ )) be such that sup f,f ′ ∈F | f ( z ) -f ′ ( z )) | ≤ 1 / (8 √ 4 T/δ ) 8: ̂ Z ← ̂ Z ∪ { ̂ z } 9: return ̂ w ( · , · ) := w ( ̂ F , · , · ), where ̂ F = { f ∈ F | ‖ f -̂ f ‖ 2 ̂ Z ≤ 3 β ( F , δ ) + 2 } and

for some absolute constants c ′ &gt; 0.

The Stable Bonus Function. With the above sampling procedure, we are now ready to obtain a stable bonus function which is formally defined in Algorithm 3. In Algorithm 3, we first subsample the given dataset Z and then round the reference function ¯ f and all data points in the subsampled dataset Z to their nearest neighbors in a 1 / (8 √ 4 T/δ )-cover. We discard the subsampled dataset if its size is too large (which happens with low probability as guaranteed by our analysis), and then define the confidence region using the new dataset and the rounded reference function.

We remark that in Algorithm 3, we round the reference function ¯ f and the state-action pairs in Z mainly for the purpose of theoretical analysis. In practice, the reference function and the state-action pairs are always stored with bounded precision, in which case explicit rounding is unnecessary. Moreover, when applying Algorithm 3 in practice, if the eluder dimension of the function class is unknown in advance, one may treat β ( F , δ ) in (4) as a tunable parameter.

## 3.2 Computational Efficiency

Finally, we discuss how to implement our algorithm computationally efficiently. To implement Algorithm 1, in Line 8, one needs to solve an empirical risk minimization (ERM) problem which can often be efficiently solved using appropriate optimization methods. To implement Algorithm 3, one needs to evaluate the width function w ( F , · , · ) for a confidence region ̂ F of the form which is a constrained optimization problem. When F is the class of linear functions, there is a closed-form formula for the width function and thus the width function can be efficiently evaluated in this case. To implement Algorithm 2, one needs to efficiently estimate λ -sensitivity of all state-action pairs in a given set Z . When F is the class of linear functions, sensitivity is equivalent to leverage score [Drineas et al., 2006] which can be efficiently estimated [Cohen et al., 2015,

<!-- formula-not-decoded -->

Clarkson and Woodruff, 2017]. For a general function class F , we give an algorithm to estimate the λ -sensitivity in Appendix A which is computationally efficient if one can efficiently judge whether a given state-action pair z is ε -independent of a set of state-actions pairs Z with respect to F for a given ε &gt; 0.

## 4 Theoretical Guarantee

In this section we provide the theoretical guarantee of Algorithm 1, which is stated in Theorem 1.

Theorem 1. Under Assumption 1, after interacting with the environment for T = KH steps, with probability 1 -δ , Algorithm 1 achieves a a regret bound of where

<!-- formula-not-decoded -->

for some constant C &gt; 0 .

Remark 1. For the tabular setting, we may set F to be the entire function space of S × A → [0 , H +1] . Recall that when S and A are finite, for any ε &gt; 0 , dim E ( F , ε ) ≤ |S||A| , log( N ( F , ε )) = ˜ O ( |S||A| ) and log( N ( S × A , ε )) = O (log( |S||A| )) , and thus the regret bound in Theorem 1 is ˜ O ( √ |S| 3 |A| 3 H 2 T ) which is worse than the near-optimal bound in [Azar et al., 2017]. However, when applied to the tabular setting, our algorithm is similar to the algorithm in [Azar et al., 2017]. By a more refined analysis specialized to the tabular setting, the regret bound of our algorithm can be improved using techniques in [Azar et al., 2017]. We would like to stress that our algorithm and analysis tackle a much more general setting and recovering the optimal regret bound for the tabular setting is not the focus of the current paper.

Remark 2. When F is the class of d -dimensional linear functions, we have dim E ( F , ε ) = ˜ O ( d ) , log( N ( F , ε )) = ˜ O ( d ) and log( N ( S × A , ε )) = ˜ O ( d ) and thus the regret bound in Theorem 1 is ˜ O ( √ d 4 H 2 T ) , which is worse by a ˜ O ( √ d ) factor when compared to the bound in [Jin et al., 2019, Wang et al., 2019], and is worse by a ˜ O ( d ) factor when compared to the bound in [Zanette et al., 2020]. Note that for our algorithm, a regret bound of ˜ O ( √ d 3 H 2 T ) is achievable using a more refined analysis (see Remark 3). Moreover, unlike our algorithm, the algorithm in [Zanette et al., 2020] requires solving the Planning Optimization Program and is thus computationally intractable. Finally, we would like to stress that our algorithm and analysis tackle the case that F is a general function class which contains the linear case studied in [Jin et al., 2019, Wang et al., 2019, Zanette et al., 2020] as a special case.

Here we provide an overview of the proof to highlight the technical novelties in the analysis.

The Stable Bonus Function. Similar to the analysis in [Jin et al., 2019, Wang et al., 2019], to account for the dependency structure in the data sequence, we need to bound the complexity of the bonus function b k h ( · , · ). When F is the class of d -dimensional linear functions (as in [Jin et al., 2019, Wang et al., 2019]), b ( · , · ) = ‖ φ ( · , · ) ‖ Λ -1 for a covariance matrix Λ ∈ R d × d , whose complexity is upper bounded by d 2 which is the number of entries in the covariance matrix Λ. However, such

<!-- formula-not-decoded -->

simple complexity upper bound is no longer available for the class of general functions considered in this paper. Instead, we bound the complexity of the bonus function by relying on the fact that the subsampled dataset has bounded size. Scrutinizing the sampling algorithm (Algorithm 2), it can be seen that the size of the subsampled dataset is upper bounded by the sum of the sensitivity of the data points in the given dataset times the log-convering number of the function class F . To upper bound the sum of the sensitivity of the data points in the given dataset, we rely on a novel combinatorial argument which establishes a surprising connection between the sum of the sensitivity and the eluder dimension of the function class F . We show that the sum of the sensitivity of data points is upper bounded by the eluder dimension of the dataset up to logarithm factors. Hence, the complexity of the subsampled dataset, and therefore, the complexity of the bonus function, is upper bound by the log-covering number of S × A (the complexity of each state-action pair) times the product of the eluder dimension of the function class and the log-covering number of the function class (the number of data points in the subsampled dataset).

In order to show that the confidence region is approximately preserved when using the subsampled dataset Z ′ , we show that for any f, f ′ ∈ F , ‖ f -f ′ ‖ 2 Z ′ is a good approximation to ‖ f -f ′ ‖ 2 Z . To show this, we apply a union bound over all pairs of functions on the cover of F which allows us to consider fixed f, f ′ ∈ F . For fixed f, f ′ ∈ F , note that ‖ f -f ′ ‖ 2 Z ′ is an unbiased estimate of ‖ f -f ′ ‖ 2 Z , and importance sampling proportinal to the sensitivity implies an upper bound on the variance of the estimator which allows us to apply concentration bounds to prove the desired result. We note that the sensitivity sampling framework used here is very crucial to the theoreical guarantee of the algorithm. If one replaces sensitivity sampling with more na¨ ıve sampling approaches (e.g. uniform sampling), then the required sampling size would be much larger, which does not give any meaningful reduction on the size of the dataset and also leads to a high complexity bonus function.

Remark 3. When F is the class of d -dimensional linear functions, our upper bound on the size of the subsampled dataset is ˜ O ( d 2 ) . However, in this case, our sampling algorithm (Algorithm 2) is equivalent to the leverage score sampling [Drineas et al., 2006] and therefore the sample complexity can be further improved to ˜ O ( d ) using a more refined analysis [Spielman and Srivastava, 2011]. Therefore, our regret bound can be improved to ˜ O ( √ d 3 H 2 T ) , which matches the bounds in [Jin et al., 2019, Wang et al., 2019]. However, the ˜ O ( d ) sample bound is specialized to the linear case and heavily relies on the matrix Chernoff bound which is unavailable for the class of general functions considered in this paper. This also explains why our regret bound in Theorem 1, when applied to the linear case, is larger by a √ d factor when compared to those in [Jin et al., 2019, Wang et al., 2019]. We leave it as an open question to obtain more refined bound on the size of the subsampled dataset and improve the overall regret bound of our algorithm.

The Confidence Region. Our algorithm applies the principle of optimism in the face of uncertainty (OFU) to balance exploration and exploitation. Note that V k h +1 is the value function estimated at step h + 1. In our analysis, we require the Q -function Q k h estimated at level h to satisfy with high probability. To achieve this, we optimize the least squares objective to find a solution f k h ∈ F using collected data. We then show that f k h is close to r ( · , · )+ ∑ s ′ ∈S P ( s ′ |· , · ) V k h +1 ( s ′ ). This would follow from standard analysis if the collected samples were independent of V k h +1 . However,

<!-- formula-not-decoded -->

V k h +1 is calculated using the collected samples and thus they are subtly dependent on each other. To tackle this issue, we notice that V k h +1 is computed by using f k h +1 and the bonus function b k h +1 , and both f k h +1 and the bonus function b k h +1 have bounded complexity, thanks to the design of bonus function. Hence, we can construct a 1 /T -cover to approximate V k h +1 . By doing so, we can now bound the fitting error of f k h by replacing V k h +1 with its closest neighbor in the 1 /T -cover which is independent of the dataset. By a union bound over all functions in the 1 /T -cover, it follows that with high probability,

<!-- formula-not-decoded -->

for some β that depends only on the complexity of the bonus function and the function class F .

Regret Decomposition and the Eluder Dimension. By standard regret decomposition for optimistic algorithms, the total regret is upper bounded by the summation of the bonus function ∑ K k =1 ∑ H h =1 b k h ( s k h , a k h ) . To bound the summation of the bonus function, we use an argument similar to that in [Russo and Van Roy, 2013], which shows that the summation of the bonus function can be upper bounded in terms of the eluder dimension of the function class F , if the confidence region is defined using the original dataset. In the formal analysis, we adapt the argument in [Russo and Van Roy, 2013] to show that even if the confidence region is defined using the subsampled dataset, the summation of the bonus function can be bounded in a similar manner.

## 4.1 Analysis of the Stable Bonus Function

Our first lemma gives an upper bound on the sum of the sensitivity in terms of the eluder dimension of the function class F .

Lemma 1. For a given set of state-action pairs Z ,

<!-- formula-not-decoded -->

Proof. For each z ∈ Z , let f, f ′ ∈ F be an arbitrary pair of functions such that ‖ f -f ′ ‖ 2 Z ≥ λ and

<!-- formula-not-decoded -->

is maximized, and we define L ( z ) = ( f ( z ) -f ′ ( z )) 2 for such f and f ′ . Note that 0 ≤ L ( z ) ≤ ( H +1) 2 . Let Z = ⋃ log(( H +1) 2 |Z| /λ ) -1 α =0 Z α ∪ Z ∞ be a dyadic decomposition with respect to L ( · ), where for each 0 ≤ α &lt; log(( H +1) 2 |Z| /λ ), define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Clearly, for any z ∈ Z ∞ , sensitivity Z , F ,λ ( z ) ≤ 1 / |Z| and thus

Now we bound ∑ z ∈Z α sensitivity Z , F ,λ ( z ) for each 0 ≤ α &lt; log(( H +1) 2 |Z| /λ ) separately. For each α , let and we decompose Z α into N α +1 disjoint subsets, i.e., Z α = ⋃ N α +1 j =1 Z α j , by using the following procedure. Let Z α = { z 1 , z 2 , . . . , z |Z α | } and we consider each z i sequentially. Initially Z α j = {} for all j . Then, for each z i , we find the largest 1 ≤ j ≤ N α such that z i is ( H +1) 2 · 2 -α -1 -independent of Z α j with respect to F . We set j = N α + 1 if such j does not exist, and use j ( z i ) ∈ [ N α + 1] to denote the choice of j for z i . By the design of the algorithm, for each z i , it is clear that z i is dependent on each of Z α 1 , Z α 2 , . . . , Z α j ( z i ) -1 .

<!-- formula-not-decoded -->

Now we show that for each z i ∈ Z α ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any z i ∈ Z α , we use f, f ′ ∈ F to denote the pair of functions in F such that ‖ f -f ′ ‖ 2 Z ≥ λ and is maximized. Since z i ∈ Z α , we must have ( f ( z i ) -f ′ ( z i )) 2 &gt; ( H + 1) 2 · 2 -α -1 . Since z i is dependent on each of Z α 1 , Z α 2 , . . . , Z α j ( z i ) -1 , for each 1 ≤ k &lt; j ( z i ), we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, by the definition of ( H +1) 2 · 2 -α -1 -independence, we have |Z α j | ≤ dim E ( F , ( H +1) 2 · 2 -α -1 ) for all 1 ≤ j ≤ N α . Therefore,

By the monotonicity of eluder dimension, it follows that

<!-- formula-not-decoded -->

Using Lemma 1, we can prove an upper bound on the number of distinct elements in Z ′ returned by the sampling algorithm (Algorithm 2).

Lemma 2. With probability at least 1 -δ/ 4 , the number of distinct elements in Z ′ returned by Algorithm 2 is at most

<!-- formula-not-decoded -->

Proof. Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since for any real number x &lt; 1, there always exists ̂ x ∈ [ x, 2 x ] such that 1 / ̂ x is an integer. Let X z be a random variable defined as

Clearly, the number of distinct elements in Z ′ is upper bounded by ∑ z ∈Z X z and E [ X z ] = p z . By Lemma 1,

<!-- formula-not-decoded -->

By Chernoff bound, with probability at least 1 -δ/ 4, we have

<!-- formula-not-decoded -->

Our second lemma upper bounds the number of elements in Z ′ returned by Algorithm 2.

Lemma 3. With probability at least 1 -δ/ 4 , |Z ′ | ≤ 4 |Z| /δ .

Proof. Let X z be the random variable which is defined as

<!-- formula-not-decoded -->

Note that |Z ′ | = ∑ z ∈Z X z and E [ X z ] = 1. By Markov inequality, with probability 1 -δ/ 4, |Z ′ | ≤ 4 |Z| /δ .

Our third lemma shows that for the given set of state-action pairs Z and function class F , Algorithm 2 returns a set of state-action pairs Z ′ so that ‖ f -f ′ ‖ 2 Z is approximately preserved for all f, f ′ ∈ F .

Lemma 4. With probability at least 1 -δ/ 2 , for any f, f ′ ∈ F ,

<!-- formula-not-decoded -->

Proof. In our proof, we separately consider two cases: ‖ f -f ′ ‖ 2 Z &lt; 2 λ and ‖ f -f ′ ‖ 2 Z ≥ 2 λ .

Case I: ‖ f -f ′ ‖ 2 Z &lt; 2 λ . Consider f, f ′ ∈ F with ‖ f -f ′ ‖ 2 Z &lt; 2 λ . Conditioned on the event defined in Lemma 3 which holds with probability at least 1 -δ/ 4, we have ‖ f -f ′ ‖ 2 Z ′ ≤ |Z ′ | · ‖ f -f ′ ‖ 2 Z ≤ 8 |Z| λ/δ . Moreover, we always have ‖ f -f ′ ‖ Z ′ ≥ 0. In summary, we have

<!-- formula-not-decoded -->

Case II: ‖ f -f ′ ‖ 2 Z ≥ 2 λ . We first show that for any fixed f, f ′ ∈ F with ‖ f -f ′ ‖ 2 Z ≥ λ , with probability at least 1 -δ/ (4 N ( F , ε/ 72 · √ λδ/ ( |Z| ))), we have

<!-- formula-not-decoded -->

To prove this, for each z ∈ Z , define

<!-- formula-not-decoded -->

.

Clearly, ‖ f -f ′ ‖ Z ′ = ∑ z ∈Z X z and E [ X z ] = ( f ( z ) -f ′ ( z )) 2 . Moreover, since ‖ f -f ′ ‖ 2 Z ≥ λ , by (3) and Definition 3, we have

Moreover, E [ X 2 z ] ≤ ( f ( z ) -f ′ ( z )) 4 /p z . Therefore, by H¨ older's inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, by Bernstein inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By union bound, the above inequality implies that with probability at least 1 -δ/ 4, for any ( f, f ′ ) ∈ C ( F, ε/ 72 · √ λδ/ ( |Z| )) ×C ( F, ε/ 72 · √ λδ/ ( |Z| )) with ‖ f -f ′ ‖ 2 Z ≥ λ ,

Now we condition on the event defined above and the event defined in Lemma 3. Consider f, f ′ ∈ F with ‖ f -f ′ ‖ 2 Z ≥ 2 λ . Recall that there exists

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, conditioned on the event defined above, we have

(1 -ε/ 4) ‖ ̂ f -̂ f ′ ‖ 2 Z ≤ ‖ ̂ f -̂ f ′ ‖ 2 Z ′ ≤ (1 + ε/ 4) ‖ ̂ f -̂ f ′ ‖ 2 Z ′ . Conditioned on the event defined in Lemma 3 which holds with probability at least 1 -δ/ 4, we have

<!-- formula-not-decoded -->

where the last inequality holds since ‖ f -f ‖ Z ≥ √ λ . Similarly,

<!-- formula-not-decoded -->

Combining Lemma 2, Lemma 3 and Lemma 4 with a union bound, we have the following proposition.

Proposition 1. With probability at least 1 -δ , the size of Z ′ returned by Algorithm 2 satisfies |Z ′ | ≤ 4 |Z| /δ , the number of distinct elements in Z is at most

<!-- formula-not-decoded -->

and for any f, f ′ ∈ F ,

<!-- formula-not-decoded -->

Proposition 2. For Algorithm 3, suppose |Z| ≤ KH = T , the following holds.

1. With probability at least 1 -δ/ (16 T ) ,

<!-- formula-not-decoded -->

2. ̂ w ( · , · ) ∈ W for a function set W with

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 if T is sufficiently large.

Proof. For the first part, conditioned on the event defined in Proposition 1, for any f ∈ F , we have

<!-- formula-not-decoded -->

Therefore, we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second part, note that ̂ w ( · , · ) is uniquely defined by ̂ F . When |Z| ≥ 4 T/δ or the number of distinct elements in Z exceeds

Therefore, for any f ∈ F , we have ‖ f -¯ f ‖ 2 Z ≤ β ( F , δ ), which implies ‖ f -̂ f ‖ 2 ̂ Z ≤ 3 β ( F , δ ) + 2 and thus f ∈ ̂ F . Moreover, for any f ∈ ̂ F , we have ‖ f -̂ f ‖ 2 ̂ Z ≤ 3 β ( F , δ ) + 2, which implies ‖ f -¯ f ‖ 2 Z ≤ 9 β ( F , δ ) + 12.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have | ̂ Z| = 0 and thus ̂ F = F . Otherwise, ̂ F is defined by ̂ f and ̂ Z . Since ̂ f ∈ C ( F , 1 / (8 √ 4 T/δ )), the total number of distinct f is upper bounded by N ( F , 1 / (8 √ 4 T/δ )). Since there are at most

distinct elements in ̂ Z , while each of them belongs to C ( S × A , 1 / (8 √ 4 T/δ )) and | ̂ Z| ≤ 4 T/δ , the total number of distinct Z is upper bounded by

<!-- formula-not-decoded -->

## 4.2 Analysis of the Algorithm

We are now ready to prove the regret bound of Algorithm 1. The next lemma establishes a bound on the estimate of a single backup.

Lemma 5 (Single Step Optimization Error) . Consider a fixed k ∈ [ K ] . Let

<!-- formula-not-decoded -->

as defined in Line 5 in Algorithm 1. For any V : S → [0 , H ] , define and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any V : S → [0 , H ] and δ ∈ (0 , 1) , there is an event E V,δ which holds with probability at least 1 -δ , such that conditioned on E V,δ , for any V ′ : S → [0 , H ] with ‖ V ′ -V ‖ ∞ ≤ 1 /T , we have for some absolute constant c ′ &gt; 0 .

Proof. In our proof, we consider a fixed V : S → [0 , H ], and define

For any f ∈ F , we consider ∑ ( τ,h ) ∈ [ k -1] × [ H ] ξ τ h ( f ) where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any ( τ, h ) ∈ [ k -1] × [ H ], define F τ h as the filtration induced by the sequence

<!-- formula-not-decoded -->

Then E [ ξ τ h ( f ) | F τ h ] = 0 and

<!-- formula-not-decoded -->

By Azuma-Hoeffding inequality, we have

Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have, with probability at least 1 -δ , for all f ∈ C ( F , 1 /T ),

We define the above event to be E V,δ , and we condition on this event for the rest of the proof. For all f ∈ F , there exists g ∈ C ( F , 1 /T ), such that ‖ f -g ‖ ∞ ≤ 1 /T , and we have

Consider V ′ : S → [0 , H ] with ‖ V ′ -V ‖ ∞ ≤ 1 /T . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any f ∈ F ,

For the second term, we have,

<!-- formula-not-decoded -->

Recall that ̂ f V ′ = arg min f ∈F ‖ f ‖ 2 D k V ′ . We have ‖ ̂ f V ′ ‖ 2 D k V ′ -‖ f V ′ ‖ 2 D k V ′ ≤ 0, which implies,

<!-- formula-not-decoded -->

for an absolute constant c &gt; 0.

<!-- formula-not-decoded -->

Lemma 6 (Confidence Region) . In Algorithm 1, let F k h be a confidence region defined as

Then with probability at least 1 -δ/ 8 , for all k, h ∈ [ K ] × [ H ] , provided

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For all ( k, h ) ∈ [ K ] × [ H ], the bonus function b k h ( · , · ) ∈ W . Note that for some absolute constant c ′ &gt; 0 . Here W is given as in Propostion 2.

<!-- formula-not-decoded -->

is a (1 /T )-cover of

<!-- formula-not-decoded -->

I.e., there exists q ∈ Q such that ‖ q -Q k h +1 ‖ ∞ ≤ 1 /T . This implies

<!-- formula-not-decoded -->

is a (1 /T )-cover of V k h +1 with log( |V| ) ≤ log |W| +log N ( F , 1 /T )+1. For each V ∈ V , let E V,δ/ (8 |V| T ) be the event defined in Lemma 5. By Lemma 5, we have Pr [⋂ V ∈V E V,δ/ (8 |V| T ) ] ≥ 1 -δ/ (8 T ). We condition on ⋂ V ∈V E V,δ/ (8 |V| T ) in the rest part of the proof.

<!-- formula-not-decoded -->

Recall that f k h is the solution of the optimization problem in Line 8 of Algorithm 1, i.e., f k h = arg min f ∈F ‖ f ‖ 2 D k h . Let V ∈ V such that ‖ V -V k h +1 ‖ ∞ ≤ 1 /T . Thus, by Lemma 5, we have

∥ ∥ for some absolute constant c ′ . Therefore, by a union bound, for all ( k, h ) ∈ [ K ] × [ H ], we have f k h ( · , · ) -( r ( · , · ) + ∑ s ′ ∈S P ( s ′ | · , · ) V k h +1 ( s ′ ) ) ∈ F k h with probability at least 1 -δ/ 8.

The above lemma guarantees that, with high probability, r ( · , · ) + ∑ s ′ ∈S P ( s ′ | · , · ) V k h +1 ( · , · ) lies in the confidence region. With this, it is guaranteed that { Q k h } ( h,k ) ∈ [ H ] × [ K ] are all optimistic, with high probability. This is formally presented in the next lemma.

Lemma 7. With probability at least 1 -δ/ 4 , for all ( k, h ) ∈ [ K ] × [ H ] , for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

Proof. For each ( k, h ) ∈ [ K ] × [ H ], define

Let E be the event that for all ( k, h ) ∈ [ K ] × [ H ], r ( · , · ) + ∑ s ′ ∈S P ( s ′ | · , · ) V k h +1 ( s ′ ) ∈ F k h . By Lemma 6, Pr[ E ] ≥ 1 -δ/ 8. Let E ′ be the event that for all ( k, h ) ∈ [ K ] × [ H ] and ( s, a ) ∈ S × A , b k h ( s, a ) ≥ w ( F k h , s, a ). By Proposition 2 and union bound, E ′ holds failure probability at most δ/ 8. In the rest part of the proof we condition on E and E ′ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since for any ( s, a ) ∈ S × A we have

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we prove Q ∗ h ( s, a ) ≤ Q k h ( s, a ) by induction on h . When h = H +1, the desired inequality clearly holds. Now we assume Q ∗ h +1 ( · , · ) ≤ Q k h +1 ( · , · ) for some h ∈ [ H ]. Clearly we have V ∗ h +1 ( · ) ≤ V k h +1 ( · ). Therefore, for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next lemma upper bounds the regret of the algorithm by the sum of b k h ( · , · ).

Note that

Lemma 8. With probability at least 1 -δ/ 2 ,

<!-- formula-not-decoded -->

Proof. In our proof, for any ( k, h ) ∈ [ K ] × [ H -1] define

<!-- formula-not-decoded -->

and define F k h as the filtration induced by the sequence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then

By Azuma-Hoeffding inequality, with probability at least 1 -δ/ 4,

<!-- formula-not-decoded -->

We condition on the above event in the rest of the proof. We also condition on the event defined in Lemma 7 which holds with probability 1 -δ/ 4.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that

We have

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It remains to bound ∑ K k =1 ∑ H h =1 b k h ( s k h , a k h ), for which we will exploit fact that F has bounded eluder dimension.

Lemma 9. With probability at least 1 -δ/ 4 , for any ε &gt; 0 ,

<!-- formula-not-decoded -->

for some absolute constant c &gt; 0 . Here β ( F , δ ) is as defined in (4) .

Proof. Let E be the event that or all ( k, h ) ∈ [ K ] × [ H ],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

By Proposition 2, E holds with probability at least 1 -δ/ 4. In the rest of the proof, we condition on E .

Let L = { ( s k h , a k h ) | b k h ( s k h , a k h ) &gt; ε } with |L| = L . We show that there exists ( s k h , a k h ) ∈ L such that ( s k h , a k h ) is ε -dependent on at least L/ dim E ( F , ε ) -H disjoint subsequences in Z k ∩ L . We demonstrate this by using the following procedure. Let L 1 , L 2 , . . . , L L/ dim E ( F ,ε ) -1 be L/ dim E ( F , ε ) -1 disjoint subsequences of L which are initially empty. We consider

<!-- formula-not-decoded -->

for each k ∈ [ K ] sequentially. For each k ∈ [ K ], for each z ∈ { ( s k 1 , a k 1 ) , ( s k 2 , a k 2 ) , . . . , ( s k H , a k H ) } ∩ L , we find j ∈ [ L/ dim E ( F , ε ) -1] such that z is ε -independent of L j and then add z into L j . By the definition of ε -independence, |L j | ≤ dim E ( F , ε ) for all j and thus we will eventually find some ( s k h , a k h ) ∈ L such that ( s k h , a k h ) is ε -dependent on each of L 1 , L 2 , . . . , L L/ dim E ( F ,ε ) -1 . Among L 1 , L 2 , . . . , L L/ dim E ( F ,ε ) -1 , there are at most H -1 of them that contain an element in

<!-- formula-not-decoded -->

and all other subsequences only contain elements in Z k ∩ L . Therefore, ( s k h , a k h ) is ε -dependent on at least L/ dim E ( F , ε ) -H disjoint subsequences in Z k ∩ L .

On the other hand, since ( s k h , a k h ) ∈ L , we have b k h ( s k h , a k h ) &gt; ε , which implies there exists f, f ′ ∈ F with ‖ f -f k h ‖ 2 Z k ≤ 9 β +12 and ‖ f ′ -f k h ‖ 2 Z k ≤ 9 β +12 such that f ( z ) -f ′ ( z ) &gt; ε . By triangle inequality, we have ‖ f -f ′ ‖ 2 Z k ≤ 36 β +48. On the other hand, since ( s k h , a k h ) is ε -dependent on at least L/ dim E ( F , ε ) -H disjoint subsequences in Z k ∩ L , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

Lastly, we apply the above lemma to bound the overall regret.

Lemma 10. With probability at least 1 -δ/ 4 ,

<!-- formula-not-decoded -->

for some absolute constant c &gt; 0 . Here β ( F , δ ) is as defined in (4) .

<!-- formula-not-decoded -->

Proof. In the proof we condition on the event defined in Lemma 9. We define w k h := b k h ( s k h , a k h ) . Let w 1 ≥ w 2 ≥ . . . ≥ w T be a permutation of { w k h } ( k,h ) ∈ [ K ] × [ H ] . By the event defined in Lemma 9, for any w t ≥ 1 /T , we have which implies

Moreover, we have w t ≤ 4 H . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are now ready to prove our main theorem.

Proof of Theorem 1. By Lemma 8 and Lemma 10, with probability at least 1 -δ , for some absolute constants c &gt; 0. Substituting the value of β ( F , δ ) completes the proof.

<!-- formula-not-decoded -->

## 5 Model Misspecification

In this section, we study the case when there is a misspecification error. Formally, we consider the following assumption.

Assumption 3. There exists a set of functions F ⊆ { f : S × A → [0 , H +1] } and a real number ζ &gt; 0 , such that for any V : S → [0 , H ] , there exists f V ∈ F which satisfies

We call ζ the misspecification error .

<!-- formula-not-decoded -->

Our algorithm for the misspecification case is identical the original algorithm except for the change of β ( F , δ ). In particular, we change the definition of β ( F , δ ) (defined in (4)) as follows.

<!-- formula-not-decoded -->

for some absolute constant c ′ &gt; 0. With this, we can now reprove Lemma 5 in the misspecified case.

Lemma 11 (Misspecified Single Step Optimization Error) . Suppose F satisfies Assumption 3. Consider a fixed k ∈ [ K ] . Let

<!-- formula-not-decoded -->

as defined in Line 5 in Algorithm 1. For any V : S → [0 , H ] , define and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any V : S → [0 , H ] and δ ∈ (0 , 1) , there is an event E V,δ which holds with probability at least 1 -δ , such that conditioned on E V,δ , for any V ′ : S → [0 , H ] with ‖ V ′ -V ‖ ∞ ≤ 1 /T , we have

Proof. In our proof, we consider a fixed V : S → [0 , H ], and define

<!-- formula-not-decoded -->

Note that unlike Lemma 5, we may have f V /negationslash∈ F . By Assumption 3, we immediately have

<!-- formula-not-decoded -->

For any f ∈ F , we consider ∑ ( τ,h ) ∈ [ k -1] × [ H ] ξ τ h ( f ) where

<!-- formula-not-decoded -->

Similar to Lemma 5, we still have, with probability at least 1 -δ , for all f ∈ C ( F , 1 /T ),

<!-- formula-not-decoded -->

We define the above event to be E V,δ , and we condition on this event for the rest of the proof. Similarly, we have, for all f ∈ F ,

Consider V ′ : S → [0 , H ] with ‖ V ′ -V ‖ ∞ ≤ 1 /T . We still have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again by the same argument as in the proof of Lemma 5, we have for any f ∈ F ,

Let ˜ f V ′ = arg min f ∈F ‖ f -f V ′ ‖ 2 Z k . Recall that ̂ f V ′ = arg min f ∈F ‖ f ‖ 2 D k V ′ . We have

<!-- formula-not-decoded -->

which implies,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ‖ ̂ f V ′ ‖ D k V ′ + ‖ f V ′ ‖ D k V ′ ≤ 4 H √ T , solving the above inequality, we have, for an absolute constant c ′ &gt; 0.

<!-- formula-not-decoded -->

Similar to Lemma 6, we have the following lemma.

Lemma 12 (Misspecified Confidence Region) . Suppose F satisfies Assumption 3. In Algorithm 1, let F k h be a confidence region defined as

Then with probability at least 1 -δ/ 8 , for all k, h ∈ [ K ] × [ H ] , provided

for some absolute constant c ′ &gt; 0 . Here W is given as in Propostion 2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof is nearly identical to that of Lemma 6.

Combining Lemma 12 with Lemma 7-10, we obtain the following theorem.

Theorem 2. Under Assumption 3, after interacting with the environment for T = KH steps, with probability at least 1 -δ , Algorithm 1 achieves a regret bound of where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some absolute constants C &gt; 0 .

## 6 Conclusion

In this paper, we give the first provably efficient value-based RL algorithm with general function approximation. Our algorithm achieves a regret bound of ˜ O (poly( dH ) √ T ) where d is a complexity measure that depends on the eluder dimension and log-covering numbers of the function class. One interesting future direction is to extend our results to policy-based methods, by combining our techniques with, e.g., those in [Cai et al., 2020].

## Acknowledgments

The authors would like to thank Jiantao Jiao, Sham M. Kakade and Csaba Szepesv´ ari for insightful comments on an earlier version of this paper. RW and RS were supported in part by NSF IIS1763562, AFRL CogDeCON FA875018C0014, and DARPA SAGAMORE HR00111990016.

## References

- Y. Abbasi-Yadkori, D. P´ al, and C. Szepesv´ ari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , pages 2312-2320, 2011.
- S. Agrawal and R. Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Advances in Neural Information Processing Systems , pages 1184-1194, 2017.
- I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113 , 2019.
- A. Antos, C. Szepesv´ ari, and R. Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning , 71(1): 89-129, 2008.
- A. Ayoub, Z. Jia, C. Szepesvari, M. Wang, and L. F. Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , 2020.

- M. G. Azar, R. Munos, and H. J. Kappen. Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349, 2013.
- M. G. Azar, I. Osband, and R. Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272, 2017.
- D. P. Bertsekas and J. N. Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.
- Q. Cai, Z. Yang, C. Jin, and Z. Wang. Provably efficient exploration in policy optimization. In International Conference on Machine Learning , 2020.
- J. Chen and N. Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051, 2019.
- K. L. Clarkson and D. P. Woodruff. Low-rank approximation and regression in input sparsity time. Journal of the ACM , 63(6):1-45, 2017.
- M. B. Cohen, Y. T. Lee, C. Musco, C. Musco, R. Peng, and A. Sidford. Uniform sampling for matrix approximation. In Proceedings of the 2015 Conference on Innovations in Theoretical Computer Science , pages 181-190, 2015.
- V. Dani, T. P. Hayes, and S. M. Kakade. Stochastic linear optimization under bandit feedback. In Conference on Learning Theory , 2008.
- C. Dann and E. Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , pages 2818-2826, 2015.
- C. Dann, L. Li, W. Wei, and E. Brunskill. Policy certificates: Towards accountable reinforcement learning. In International Conference on Machine Learning , pages 1507-1516, 2019.
- K. Dong, J. Peng, Y. Wang, and Y. Zhou. √ n -regret for learning in markov decision processes with function approximation and low bellman rank. In Conference on Learning Theory , 2020.
- P. Drineas, M. W. Mahoney, and S. Muthukrishnan. Sampling algorithms for l 2 regression and applications. In Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm , pages 1127-1136. Society for Industrial and Applied Mathematics, 2006.
- S. S. Du, Y. Luo, R. Wang, and H. Zhang. Provably efficient q-learning with function approximation via distribution shift error checking oracle. In Advances in Neural Information Processing Systems , pages 8058-8068, 2019.
- S. S. Du, S. M. Kakade, R. Wang, and L. F. Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations , 2020a.
- S. S. Du, J. D. Lee, G. Mahajan, and R. Wang. Agnostic q-learning with function approximation in deterministic systems: Tight bounds on approximation error and sample complexity. arXiv preprint arXiv:2002.07125 , 2020b.
- D. Feldman and M. Langberg. A unified framework for approximating and clustering data. In Proceedings of the forty-third annual ACM symposium on Theory of computing , pages 569-578, 2011.

- D. Feldman, M. Schmidt, and C. Sohler. Turning big data into tiny data: Constant-size coresets for k-means, pca and projective clustering. In Proceedings of the twenty-fourth annual ACM-SIAM symposium on Discrete algorithms , pages 1434-1453. SIAM, 2013.
- S. Filippi, O. Cappe, A. Garivier, and C. Szepesv´ ari. Parametric bandits: The generalized linear case. In Advances in Neural Information Processing Systems , pages 586-594, 2010.
- T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600, 2010.
- Z. Jia, L. F. Yang, and M. Wang. Feature-based q-learning for two-player stochastic games. arXiv preprint arXiv:1906.00423 , 2019.
- N. Jiang, A. Krishnamurthy, A. Agarwal, J. Langford, and R. E. Schapire. Contextual decision processes with low bellman rank are PAC-learnable. In International Conference on Machine Learning , pages 1704-1713, 2017.
- C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873, 2018.
- C. Jin, Z. Yang, Z. Wang, and M. I. Jordan. Provably efficient reinforcement learning with linear function approximation. arXiv preprint arXiv:1907.05388 , 2019.
- S. M. Kakade. On the sample complexity of reinforcement learning . PhD thesis, University of London London, England, 2003.
- M. Kearns and S. Singh. Near-optimal reinforcement learning in polynomial time. Machine learning , 49(2-3):209-232, 2002.
- J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013.
- K. R. Koedinger, E. Brunskill, R. S. Baker, E. A. McLaughlin, and J. Stamper. New potentials for data-driven intelligent tutoring system development and optimization. AI Magazine , 34(3): 27-41, 2013.
- M. Langberg and L. J. Schulman. Universal ε -approximators for integrals. In Proceedings of the twenty-first annual ACM-SIAM symposium on Discrete Algorithms , pages 598-607. SIAM, 2010.
- T. Lattimore and M. Hutter. Near-optimal pac bounds for discounted mdps. Theoretical Computer Science , 558:125-143, 2014.
- L. Li, Y. Lu, and D. Zhou. Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning , pages 2071-2080. JMLR. org, 2017.
- Y. Li, Y. Wang, and Y. Zhou. Nearly minimax-optimal regret for linearly parameterized bandits. In Conference on Learning Theory , pages 2173-2174, 2019.
- V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 , 2013.

- V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529, 2015.
- A. Modi, N. Jiang, A. Tewari, and S. Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. arXiv preprint arXiv:1910.10597 , 2019.
- R. Munos. Error bounds for approximate policy iteration. In ICML , volume 3, pages 560-567, 2003.
- R. Munos and C. Szepesv´ ari. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9(May):815-857, 2008.
- I. Osband and B. Van Roy. Model-based reinforcement learning and the eluder dimension. In Advances in Neural Information Processing Systems , pages 1466-1474, 2014.
- I. Osband and B. Van Roy. On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 , 2016.
- I. Osband, B. Van Roy, D. J. Russo, and Z. Wen. Deep exploration via randomized value functions. Journal of Machine Learning Research , 20:1-62, 2019.
- D. Russo and B. Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In Advances in Neural Information Processing Systems , pages 2256-2264, 2013.
- T. Schaul, J. Quan, I. Antonoglou, and D. Silver. Prioritized experience replay. In International Conference on Learning Representations , 2016.
- S. Shalev-Shwartz, S. Shammah, and A. Shashua. Safe, multi-agent, reinforcement learning for autonomous driving. arXiv preprint arXiv:1610.03295 , 2016.
- A. Sidford, M. Wang, X. Wu, L. Yang, and Y. Ye. Near-optimal time and sample complexities for solving markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196, 2018.
- D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, et al. Mastering the game of Go without human knowledge. Nature , 550 (7676):354, 2017.
- D. A. Spielman and N. Srivastava. Graph sparsification by effective resistances. SIAM Journal on Computing , 40(6):1913-1926, 2011.
- A. L. Strehl, L. Li, E. Wiewiora, J. Langford, and M. L. Littman. PAC model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888. ACM, 2006.
- A. L. Strehl, L. Li, and M. L. Littman. Reinforcement learning in finite MDPs: PAC analysis. Journal of Machine Learning Research , 10(Nov):2413-2444, 2009.
- W. Sun, N. Jiang, A. Krishnamurthy, A. Agarwal, and J. Langford. Model-based rl in contextual decision processes: PAC bounds and exponential improvements over model-free approaches. In Conference on Learning Theory , pages 2898-2933, 2019.

- C. Szepesv´ ari and R. Munos. Finite time bounds for sampling based fitted value iteration. In International Conference on Machine Learning , pages 880-887, 2005.
- I. Szita and C. Szepesv´ ari. Model-based reinforcement learning with nearly tight exploration complexity bounds. In International Conference on Machine Learning , pages 1031-1038, 2010.
- H. Van Hasselt, A. Guez, and D. Silver. Deep reinforcement learning with double Q-learning. In Thirtieth AAAI conference on artificial intelligence , 2016.
- R. Wang, S. S. Du, L. F. Yang, and S. M. Kakade. Is long horizon reinforcement learning more difficult than short horizon reinforcement learning? arXiv preprint arXiv:2005.00527 , 2020.
- Y. Wang, R. Wang, S. S. Du, and A. Krishnamurthy. Optimism in reinforcement learning with generalized linear function approximation. arXiv preprint arXiv:1912.04136 , 2019.
- Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, and N. Freitas. Dueling network architectures for deep reinforcement learning. In International Conference on Machine Learning , pages 1995-2003, 2016.
- L. Yang and M. Wang. Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004, 2019.
- L. F. Yang and M. Wang. Reinforcement leaning in feature space: Matrix bandit, kernels, and regret bound. In International Conference on Machine Learning , 2020.
- A. Zanette and E. Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning , pages 7304-7312, 2019.
- A. Zanette, A. Lazaric, M. J. Kochenderfer, and E. Brunskill. Limiting extrapolation in linear approximate value iteration. In Advances in Neural Information Processing Systems , pages 56165625, 2019.
- A. Zanette, A. Lazaric, M. Kochenderfer, and E. Brunskill. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning , 2020.
- Z. Zhang, Y. Zhou, and X. Ji. Almost optimal model-free reinforcement learning via referenceadvantage decomposition. arXiv preprint arXiv:2004.10019 , 2020.

## A Estimating the Sensitivity

In this section, we present a computationally efficient algorithm to estimate the λ -sensitivity of all state-action pairs in a give set Z ⊆ S × A with respect to a given function class F . The algorithm is formally described in Algorithm 4.

## Algorithm 4 Estimate ( F , Z , λ )

```
1: Input : function class F , set of state-action pairs Z ⊆ S × A , accuracy parameter λ 2: Initialize sensitivity est Z , F ,λ ( z ) ← 0 for all z ∈ Z 3: for α ∈ { 0 , 1 , . . . , log(( H +1) 2 |Z| /λ ) -1 } do 4: Set N α ←|Z| / dim E ( F , ( H +1) 2 · 2 -α -1 ) 5: Initialize Z α j ←{} for each j ∈ [ N α ] 6: for z ∈ Z do 7: if z is dependent on Z α j for all j ∈ [ N α ] then 8: j α ( z ) ← N α +1 9: else 10: j α ( z ) ← min { j ∈ [ N α ] | z is independent of Z α j } 11: sensitivity α Z , F ,λ ( z ) ← 2 j α ( z ) 12: for z ∈ Z do 13: sensitivity est Z , F ,λ ( z ) ← 1 |Z| + ∑ 0 ≤ α< log(( H +1) 2 |Z| /λ ) sensitivity α Z , F ,λ ( z ) 14: return { sensitivity est Z , F ,λ ( z ) } z ∈Z
```

Given a function class F , a set of state-action pairs Z and an accuracy parameter λ , Algorithm 4 returns an estimate of the λ -sensitivity for each z ∈ Z . With Algorithm 4, we can now implement Algorithm 2 computationally efficiently by replacing (3) in Algorithm 2 with

<!-- formula-not-decoded -->

where for each z ∈ Z , sensitivity est Z , F ,λ ( z ) is the estimated λ -sensitivity returned by Algorithm 4. According to the analysis in Section 4.1, to prove the correctness of Algorithm 2 after this modification, it suffices to prove that

<!-- formula-not-decoded -->

for each z ∈ Z and

<!-- formula-not-decoded -->

which we prove in the remaining part of this section.

Lemma 13. For each z ∈ Z , sensitivity est Z , F ,λ ( z ) ≥ sensitivity Z , F ,λ ( z ) .

Proof. In our proof we consider a fixed z ∈ Z . Let f, f ′ ∈ F be an arbitrary pair of functions such that ‖ f -f ′ ‖ 2 Z ≥ λ and

<!-- formula-not-decoded -->

is maximized and we define L ( z ) = ( f ( z ) -f ′ ( z )) 2 for such f and f ′ . If L ( z ) ≤ λ/ |Z| , then we have sensitivity est Z , F ,λ ( z ) ≥ 1 / |Z| ≥ sensitivity Z , F ,λ ( z ). Otherwise, there exists 0 ≤ α &lt; log(( H +1) 2 |Z| /λ ) such that L ( z ) ∈ (( H +1) 2 · 2 -α -1 , ( H +1) 2 · 2 -α ]. Since L ( z ) &gt; ( H +1) 2 · 2 -α -1 and z is dependent on each of Z α 1 , Z α 2 , . . . , Z α j α ( z ) -1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 14. ∑ z ∈Z sensitivity est Z , F ,λ ( z ) ≤ 4dim E ( F , λ/ |Z| ) log(( H +1) 2 |Z| /λ ) ln |Z| .

Proof. For each 0 ≤ α &lt; log(( H +1) 2 |Z| /λ ), by the definition of ( H + 1) 2 · 2 -α -1 -independence, we have

<!-- formula-not-decoded -->

for each j ∈ [ N α ]. Therefore,

<!-- formula-not-decoded -->

≤ E F |Z| |Z| E F · ≤ E F |Z| |Z|

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->