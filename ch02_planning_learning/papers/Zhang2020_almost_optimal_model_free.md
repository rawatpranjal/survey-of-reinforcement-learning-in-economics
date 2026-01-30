## Almost Optimal Model-Free Reinforcement Learning via Reference-Advantage Decomposition

## Zihan Zhang

Department of Automation Tsinghua University zihan-zh17@mails.tsinghua.edu.cn

## Yuan Zhou

Department of ISE University of Illinois at Urbana-Champaign yuanz@illinois.edu

## Abstract

We study the reinforcement learning problem in the setting of finite-horizon episodic Markov Decision Processes (MDPs) with S states, A actions, and episode length H . We propose a model-free algorithm UCB-ADVANTAGE and prove that it achieves ˜ O p ? H 2 SAT q regret where T ' KH and K is the number of episodes to play. Our regret bound improves upon the results of [Jin et al., 2018] and matches the best known model-based algorithms as well as the information theoretic lower bound up to logarithmic factors. We also show that UCB-ADVANTAGE achieves low local switching cost and applies to concurrent reinforcement learning, improving upon the recent results of [Bai et al., 2019].

## 1 Introduction

Reinforcement learning (RL) [Burnetas and Katehakis, 1997] studies the problem where an agent aims to maximize its accumulative rewards through sequential decision making in an unknown environment modeled by Markov Decision Processes (MDPs). At each time step, the agent observes the current state s and interacts with the environment by taking an action a and transits to next state s 1 following the underlying transition model.

There are mainly two types of algorithms to approach reinforcement learning: model-based and model-free learning. Model-based algorithms learn a model from the past experience and make decision based on this model while model-free algorithms only maintain a group of value functions and take the induced optimal actions. Because of these differences, model-free algorithms are usually more space- and time-efficient compared to model-based algorithms. Moreover, because of their simplicity and flexibility, model-free algorithms are popular in a wide range of practical tasks (e.g., DQN [Mnih et al., 2015], A3C [Mnih et al., 2016], TRPO [Schulman et al., 2015a], and PPO [Schulman et al., 2017]). On the other hand, however, it is believed that model-based algorithms may be able to take the advantage of the learned model and achieve better learning performance in terms of regret or sample complexity, which has been empirically evidenced by Deisenroth and Rasmussen [2011] and Schulman et al. [2015a]. Much experimental research has been done for both types of the algorithms, and given that there has been a long debate on their pros and cons that dates back to [Deisenroth and Rasmussen, 2011], a natural and intriguing theoretical question to study about reinforcement learning algorithms is that -

Question 1. Is it possible that model-free algorithms achieve as competitive learning efficiency as model-based algorithms, while still maintaining low time and space complexities?

Preprint. Under review.

## Xiangyang Ji

Department of Automation Tsinghua University xyji@tsinghua.edu.cn

Towards answering this question, the recent work by Jin et al. [2018] formally defines that an RL algorithm is model-free if its space complexity is always sublinear relative to the space required to store the MDP parameters, and then proposes a model-free algorithm (which is a variant of the Q -learning algorithm [Watkins, 1989]) that achieves the first ? T -type regret bound for finite-horizon episodic MDPs in the tabular setting (i.e., discrete state spaces). However, there is still a gap of factor ? H between the regret of their algorithm and the best model-based algorithms. In this work, we close this gap by proposing a novel model-free algorithm, whose regret matches the optimal model-based algorithms, as well as the information theoretic lower bound. The results suggest that model-free algorithms can learn as efficiently as model-based ones, giving an affirmative answer to Question 1 in the setting of episodic tabular MDPs.

## 1.1 Our Results

Main Theorem. We propose a novel variant of the Q -learning algorithm, UCB-ADVANTAGE. We then prove the following main theorem of the paper.

Theorem 1. For T greater than some polynomial of S , A , and H , and for any p P p 0 , 1 q , with probability p 1 ´ p q , the regret of UCB-ADVANTAGE is bounded by Regret p T q ď ˜ O p ? H 2 SAT q , where poly-logarithmic factors of T and 1 { p are hidden in the ˜ O p¨q notation.

One of the main technical ingredients of UCB-ADVANTAGE is to incorporate a novel update rule for the Q -function based on the proposed reference-advantage decomposition . More specifically, we propose to view the optimal value function V ˚ as V ˚ ' V ref `p V ˚ ´ V ref q , where V ref , the reference component, is a comparably easier learned approximate of V ˚ and the other component p V ˚ ´ V ref q is referred to as the advantage part. Based on this decomposition, the new update rule learns the corresponding parts of the Q -function using carefully designed (and different) subsets of the collected data, so as to minimize the deviation, maximize the data utilization, and reduce the estimation variance.

Compared to the ˜ O p ? H 3 SAT q regret bound of the UCB-Bernstein algorithm in [Jin et al., 2018], UCB-ADVANTAGE saves a factor of ? H , and matches the information theoretic lower bound of Ω p ? H 2 SAT q in [Jin et al., 2018] up to logarithmic factors. The regret of UCB-ADVANTAGE is at the same order of the best model-based algorithms such as UCBVI [Azar et al., 2017] and vUCQ [Kakade et al., 2018]. 1 However, the time complexity before time step T is O p T q and the space complexity is O p SAH q for UCB-ADVANTAGE. In contrast, both UCBVI and vUCQ uses ˜ O p TS 2 A q time and O p S 2 AH q space.

Another highlight of UCB-ADVANTAGE is the use of the stage-based update framework which enables an easy integration of the new update rule (as above) and the standard update rule. In such a framework, the visits to each state-action pair are partitioned into stages , which are used to design the trigger and subsets of data for each update.

Implications. An extra benefit of the stage-based update framework is to ensure the low frequency of policy switches of UCB-ADVANTAGE, stated as follows.

Theorem 2. The local switching cost of UCB-ADVANTAGE is bounded by O p SAH 2 log T q .

While one may refer to Appendix C for the details of the theorem, the notion of local switching cost for RL is recently introduced and studied by Bai et al. [2019], where the authors integrate a lazy update scheme with the UCB-Bernstein algorithm [Jin et al., 2018] and achieve ˜ O p ? H 3 SAT q regret and O p SAH 3 log T q local switching cost. In contrast, our result improves in both metrics of regret and switching cost.

Our results also apply to concurrent RL, a research direction closely related to batched learning and learning with low switching costs, stated as follows.

Corollary 3. Given M parallel machines, the concurrent and pure exploration version of UCBADVANTAGE can compute an /epsilon1 -optimal policy in ˜ O p H 2 SA ` H 3 SA {p /epsilon1 2 M qq concurrent episodes.

1 Both Azar et al. [2017] and Kakade et al. [2018] assume equal transition matrices P 1 ' P 2 ' ¨ ¨ ¨ ' P H . In this work, we adopt the same setting as in, e.g., [Jin et al., 2018] and [Bai et al., 2019], where P 1 , P 2 , . . . , P H can be different. This adds a factor of ? H to the regret analysis in [Azar et al., 2017] and [Kakade et al., 2018].

In contrast, the state-of-the-art result [Bai et al., 2019] uses ˜ O p H 3 SA ` H 4 SA {p /epsilon1 2 M qq concurrent episodes. When M ' 1 , Corollary 3 implies that the single-threaded exploration version of UCBADVANTAGE uses ˜ O p H 3 SA { /epsilon1 2 q episodes to learn an /epsilon1 -optimal policy. In Appendix C, we provide a simple Ω p H 3 SA { /epsilon1 2 q -episode lower bound for the sample complexity, showing the optimality up to logarithmic factors.

## 1.2 Additional Related Works

Regret Analysis for RL. Since our results focus on the tabular case, we will not mention most of the results on RL for continuous state spaces. For the tabular setting, there are plenty of recent works on model-based algorithms under various settings (e.g., [Jaksch et al., 2010, Agrawal and Jia, 2017, Azar et al., 2017, Ouyang et al., 2017, Fruit et al., 2019, Simchowitz and Jamieson, 2019, Zanette and Brunskill, 2019, Zhang and Ji, 2019]). The readers may refer to [Jin et al., 2018] for more detailed review and comparison. In contrast, fewer model-free algorithms are proposed. Besides [Jin et al., 2018], an earlier work [Strehl et al., 2006] implies that T 4 { 5 -type regret can be achieved by a model-free algorithm.

Variance Reduction and Advantage Functions. Variance reduction techniques via referenceadvantage decomposition is used for faster optimization algorithms [Johnson and Zhang, 2013]. The technique is also recently applied to pure exploration in learning discounted MDPs [Sidford et al., 2018b,a]. However, since Sidford et al. [2018b,a] assume the access to a simulator and UCBADVANTAGE is completely online, our update rule and data partition design is very different. Our work is also the first for regret analysis in RL.

The use of advantage functions have also witnessed much success for RL in practice. For example, in A3C[Mnih et al., 2016], the advantage function is defined to be Adv p s, a q : ' Q π p s, a q´ V π p s q , and helps to reduce the estimation variance of the policy gradient. Similar definitions can also be found in other works such as [Sutton et al., 2000], Generalized Advantage Estimation [Schulman et al., 2015b] and Dueling DQN [Wang et al., 2015]. In comparison, our advantage function is defined on the states instead of the state-action pairs.

## 2 Preliminaries

We study the setting of episodic MDPs where an MDP is described by p S , A , H, P, r q . Here, S ˆ A is the state-action space, H is the length of each episode, P is the transition probability matrix and r is the deterministic reward function 2 . Without loss of generality, we assume that r h p s, a q P r 0 , 1 s for all s, a, h . During each episode, the agent observes the initial state s 1 which may be chosen by an oblivious adversary (i.e., the adversary may have the access to the algorithm description used by the agent but does not observe the execution trajectories of the agent 3 ).

During each step within the episode, the agent takes an action a h and transits to s h ` 1 according to P h p¨| s h , a h q . The agent keeps running for H steps and then the episode terminates.

<!-- formula-not-decoded -->

A policy 4 π is a mapping from S ˆ r H s to A . Given a policy π , we define its value function and Q -function as

<!-- formula-not-decoded -->

As boundary conditions, we define V π H ` 1 p s q ' Q π H ` 1 p s, a q ' 0 for any π, s, a . Also note that, for simplicity, throughout the paper, we use xy to denote x T y for two vectors of the same dimension and use P s,a,h to denote P h p¨| s, a q .

2 Our results generalize to stochastic reward functions easily.

3 Another adversary model is the the stronger adaptive adversary who may observe the execution trajectories and select the initial states based on the observation. While it is possible that a more careful analysis of our algorithm also works for the adaptive adversary, we do not make any effort verifying this statement. We also note that previous works such as [Jin et al., 2018, Bai et al., 2019] do not explicitly define their adversary models and it is not clear whether their analysis works for the adaptive adversary.

4 In this work, we mainly consider deterministic policies since the optimal value function can be achieved by a deterministic policy.

The optimal value function is then given by V ˚ h p s q ' sup π V π h p s q and Q ˚ h p s, a q ' r h p s, a q ` P s,a,h V ˚ h ` 1 for any p s, a q P S ˆ A , h P r H s .

The learning problem consists of K episodes, i.e, T ' KH steps. Let s k 1 be the state given to the agent at the beginning of the k -th episode, and let π k be the policy adopted by the agent during the k -th episode. To goal is to minimize the total regret at time step T which is defined as follows,

<!-- formula-not-decoded -->

## 3 The UCB-ADVANTAGE Algorithm

In this section, we introduce the UCB-ADVANTAGE algorithm. We start by reviewing the Q -learning algorithms proposed in [Jin et al., 2018]. Recall that Jin et al. [2018] selects the learning rate α t ' H ` 1 H ` t , and sets the weights α i t ' α i Π t j ' i ` 1 p 1 ´ α j q for the i -th samples out of the a total of t data points, for any state-action pair. Note that α i t is roughly Θ p H { t q for the indices i P r H ´ 1 H ¨ t, t s and vanishes quickly when i ! H ´ 1 H ¨ t . As a result, their update process is roughly equivalent to using the latest 1 H fraction of samples to update the value function for any state-action pair. Next, we introduce our stage-based update framework, which shares much similarity with the process discussed above. However, our framework enjoys simpler analysis and enables easier integration of the two update rules which will be explained afterwards.

Stages and Stage-Based Update Framework. For any triple p s, a, h q , we divide the samples received for the triple into consecutive stages . The length of each stage roughly increases exponentially with the growth rate p 1 ` 1 { H q . More specifically, we define e 1 ' H and e i ` 1 ' X p 1 ` 1 H q e i \ for all i ě 1 , standing for the length of the stages. We also let L : ' t ř j i ' 1 e i | j ' 1 , 2 , 3 , . . . u be the set of indices marking the ends of the stages.

Now we introduce the stage-based update framework . For any p s, a, h q triple, we update Q h p s, a q when the total visit number of p s, a, h q the end of the current stage (in other word, the total visit number occurs in L ). Only the samples in the latest stage will be used in this update. Using the language of [Jin et al., 2018], for any total visit number t in the p j ` 1 q -th stage, our update framework is equivalent to setting the weight distribution to be α i t ' e ´ 1 j ¨ I r i in the j -th stage s .

We note that the definition of stages is with respect to the triple p s, a, h q . For any fixed pair of k and h , let p s k h , a k h q be the state-action pair at the h -th step during the k -th episode of the algorithm. We say that p k, h q falls in the j -th stage of p s, a, h q if and only if p s, a q ' p s k h , a k h q and the total visit number of p s k h , a k h q after the k -th episode is in p j ´ 1 i ' 1 e i , j i ' 1 e i s .

ř ř One benefit of our stage-based update framework is that it helps to reduce the number of the updates to the Q -function, leading to less local switching costs, which is recently also studied by Bai et al. [2019], where the authors propose to apply a lazy update scheme to the algorithms in Jin et al. [2018]. The lazy update scheme uses an exponential triggering sequence with a growth rate of p 1 ` 1 {p 2 H p H ` 1 qqq , which is more conservative than the growth rate of stage lengths in our work. As a result, our algorithm saves an H factor in the switching cost compared to [Bai et al., 2019].

More importantly, our stage-based update framework, compared to the algorithms in [Jin et al., 2018], (in our opinion) simplifies the analysis, makes it easier to integrate the standard update rule and the one based on the reference-advantage decomposition. Both update rules are used in our algorithm, and we now discuss them separately.

The Standard Update Rule and its Limitation. The algorithms in [Jin et al., 2018] uses the following standard update rule,

<!-- formula-not-decoded -->

where b is the exploration bonus, and P s,a,h V h ` 1 Ź is the empirical estimate of P s,a,h V h ` 1 . We also adopt this update rule in our algorithm. However, a crucial restriction is that the earlier samples collected, the more deviation one would expect between the V h ` 1 learned at that moment and the

true value. To ensure that these deviations do not ruin the whole estimate, we have to require that P s,a,h V h ` 1 Ź only uses the samples acquired from the last stage. This means that we can only estimate the P s,a,h V h ` 1 term using about 1 { H fraction of the obtained data, and we note that this is also the reason of the extra ? H occurred in the UCB-Bernstein algorithm by Jin et al. [2018].

Reference-Advantage Decomposition and the Advantage-Based Update Rule. We now introduce the reference-advantage decomposition, which is the key to reducing the extra ? H factor. At a high level, we aim at first learning a quite accurate estimation of the optimal value function V ˚ and denote it by the reference value function V ref . The accuracy is controlled by an error parameter β which is quite small but independent of T or K . In other words, we wish to have V ˚ h p s q ď V ref h p s q ď V ˚ h p s q ` β for all s and h , and for the purpose of simple explanation, we set β ' 1 { H at this moment; in our algorithm, β can be any value that is less than a 1 { H while independent of T or K .

For starters, let us first assume that we have the access to the dreamed V ref reference function as stated above. Now we write V ˚ ' V ref ` p V ˚ ´ V ref q , and refer to the second term as the advantage compared to the reference values 5 . Now the Q -function can be updated using the following advantage-based rule,

<!-- formula-not-decoded -->

where b is the exploration bonus, and both P s,a,h V ref h ` 1 Ź and P s,a,h p V h ` 1 ´ V ref h ` 1 q Ź are empirical estimates of P s,a,h V ref h ` 1 and P s,a,h p V h ` 1 ´ V ref h ` 1 q (respectively) based on the observed samples. We still have to require that P s,a,h p V h ` 1 ´ V ref h ` 1 q Ź uses the samples only from the last stage so as to limit the deviation error due to V h ` 1 in the earlier samples.

Fortunately, thanks to the reference-advantage decomposition, and since that V is learned based on V ref and approximates V ˚ even better than V ref , we have that } V h ` 1 ´ V ref h ` 1 } 8 ď β ' 1 { H holds for all samples, which suffices to offset the weakness of using only 1 { H of the total data, and helps to learn an accurate estimation of the second term. On the other hand, for the first term in the RightHand-Side of (3), since V ref is fixed and never changes, we are able to use all the samples collected to conduct the estimation, without suffering any deviation. This means that the first term can also be estimated with high accuracy.

The discussion till now has assumed that the reference value vector V ref is known. To remove this assumption, we note that β is independent of T , therefore a natural hope is to learn V ref using sample complexity also almost independent of T , incurring regret only in the lower order terms. However, since it is not always possible to learn the value function of every state (especially the ones almost not reachable), we need to integrate the learning for reference vector into the main algorithm, and much technical effort is made to enable the analysis for the integrated algorithm.

Description of the Algorithm. UCB-ADVANTAGE is described in Algorithm 1, where c 1 , c 2 , and c 3 are large enough positive universal constants so that concentration inequalities may be applied in the analysis. Besides the standard quantities such as Q h p s, a q , V h p s q , and the reference value function V ref h , the algorithm keeps seven types of accumulators to facilitate the update to the Q - and value functions: accumulators N h p s, a q and ˇ N h p s, a q are used to keep the total visit number and the number of visits only counting the current stage to p s, a, h q , respectively. Three types of intra-stage accumulators are used for the samples in the latest stage; they are reset at the beginning of each stage and updated at every time step as follows (note that short-hands are defined for succinct presentation of the Q -function update rule in (9)):

<!-- formula-not-decoded -->

Finally, the following two types of global accumulators are used for the samples in all stages,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

5 Interestingly, one might argue that the term should rather be called 'disadvantage' as it is always nonpositive. We choose the name 'advantage' to highlight the similarity between our algorithm and many empirical algorithms in literature. See Section 1.2 for more discussion.

## Algorithm 1 UCB-ADVANTAGE

Initialize: set all accumulators to 0 ; for all p s, a, h q P S ˆ A ˆ r H s , set V h p s q Ð H ´ h ` 1 ; Q h p s, a q Ð H ´ h ` 1 ; V ref h p s, a q Ð H ;

for episodes

k

1

,

observe for

Update the accumulators via n : ' N h p s h , a h q ` Ð 1 , ˇ n : ' ˇ N h p s h , a h q ` Ð 1 , and (4), (5), (6).

Ð

s

1

;

h Ð 1 , 2 , . . . , H do Take action a h Ð arg max a Q h p s h , a q , and observe s h ` 1 .

if n P L { Reaching the end of the stage and update triggered } then

{ Set the exploration bonuses, update the Q -function and the value function }

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ˇ N h p s h , a h q , ˇ µ h p s h , a h q , ˇ υ h p s h , a h q , ˇ σ h p s h , a h q Ð 0 ; { Reset intra-stage accumulators } end if if ř a N h p s h , a q ' N 0 then V ref h p s h q Ð V h p s h q ; { Learn the reference value function } end for end for

All accumulators are initialized to 0 at the beginning of the algorithm.

The algorithm sets ι Ð log p 2 p q (where p is the parameter for the failure probability) and β Ð 1 ? H . We also set N 0 : ' c 4 SAH 5 ι β 2 for a large enough universal constant c 4 ą 0 , denoting the number of visits needed for each state to learn a β -accurate reference value.

By the definition of the accumulators, the first two expressions in min t¨u in (9) respectively correspond to update rules (2) and (3), where b and ¯ b are the respective exploration bonuses. The bonuses are set in a way that both expressions can be shown to upper bound Q ˚ in the desired event. The update (9) also makes sure that the learned Q -function is non-increasing as the algorithm proceeds.

## 4 The Analysis (Proof of Theorem 1)

Let N k h p s, a q , ˇ N k h p s, a q , Q k h p s, a q , V k h p s q and V ref ,k h p s q respectively denote the values of N h p s, a q , ˇ N h p s, a q , Q h p s, a q , V h p s q and V ref h p s q at the beginning of k -th episode. In particular, N K ` 1 h p s, a q denotes the number of visits of p s, a, h q after all K episodes are done.

Recall that the value function Q h p s, a q is non-increasing as the algorithm proceeds. On the other hand, we claim in the following proposition that Q h p s, a q upper bounds Q ˚ h p s, a q with high probability.

To facilitate the proof, we need a few more notations. For each k and h , let n k h be the total number of visits to p s k h , a k h , h q prior to the current stage with respect to the same triple. Let ˇ n k h be the number of visits to p s k h , a k h , h q during the stage immediately before the current stage. We let l k h,i denote the index of the i -th episode among the n k h episodes defined above. Also let ˇ l k h,i be the index of the i -th episode among the ˇ n k h episodes defined above. When h and k are clear from the context, we omit the two letters and use l i and ˇ l i for short. We use µ ref ,k h , ˇ µ k h , ˇ ν k h , σ ref ,k h , ˇ σ k h , b k h and b k h to denote respectively the values of µ ref , ˇ µ , ˇ υ , σ ref , ˇ σ , b and b in the computation of Q k h p s k h , a k h q in (9).

Proposition 4. Let p P p 0 , 1 q . With probability at least p 1 ´ 4 T p H 2 T 3 ` 3 qq p , it holds that Q ˚ h p s, a q ď Q k ` 1 h p s, a q ď Q k h p s, a q for any s, a, h, k .

2

, . . . , K

do

The proof of Proposition 4 involves some careful application of the concentration inequalities for martingales and is deferred to Appendix B.

## 4.1 Learning the Reference Value Function

As mentioned before, we hope to get an accurate estimate of V ˚ as the reference value function. Similar to the proof of Lemma 2 in [Dong et al., 2019], we show in the following lemma (the proof of which deferred to Appendix B) that we can learn a good reference value for each state with bounded sample complexity. Also note that while it is possible to improve the upper bound in Lemma 5 via more refined analysis, the current form is sufficient to prove our main theorem.

Lemma 5. Conditioned on the successful events of Proposition 4, for any /epsilon1 P p 0 , H s , with probability p 1 ´ Tp q it holds that for any h P r H s , ř K k ' 1 I ' V k h p s k h q ´ V ˚ h p s k h q ě /epsilon1 ‰ ď O p SAH 5 ι { /epsilon1 2 q .

By Lemma 5 with /epsilon1 set to β , the fact that V k is non-increasing in k and the definition of N 0 , we have the following corollary.

Corollary 6. Conditioned on the successful events of Proposition 4 and Lemma 5, for every state s we have that n k h p s q ě N 0 ùñ V ˚ h p s q ď V ref ,k h p s q ď V ˚ h p s q ` β .

## 4.2 Regret Analysis with Reference-Advantage Decomposition

We now prove Theorem 1. We start by replacing p by p { poly p H,T q so that we only need to show the desired regret bound with probability p 1 ´ poly p H,T q ¨ p q . The proof in this subsection will also be conditioned on the successful events in Proposition 4 and Lemma 5, so that the regret can be expressed as

<!-- formula-not-decoded -->

Define δ k h : ' V k h p s k h q ´ V ˚ h p s k h q and ζ k h : ' V k h p s k h q ´ V π k h p s k h q . Note that when N k h p s k h , a k h q P L , we have that n k h ' N k h p s k h , a k h q and ˇ n k h ' ˇ N k h p s k h , a k h q . Following the update rules (9) and (10), we have that 6

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together with the Bellman equation V π k h p s k h q ' r h p s k h , a k h q ` P s k h ,a k h ,h V π k h ` 1 , we have that

6 Here we define 0 { 0 to be 0 so that forms such as 1 n k h ř n k h i ' 1 X i are treated as 0 if n k h ' 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where letting V REF be the final reference vector (i.e., V REF : ' V ref ,K ` 1 ), and 1 j be the j -th canonical basis vector (i.e., p 0 , . . . , 0 , 1 , 0 , . . . , 0 q where the only 1 is located at the j -th entry), we define

<!-- formula-not-decoded -->

Here at Inequality (12) is implied by the successful event of martingale concentration (which is implied by the successful event in the proof of Proposition 4, in particular, Inequality (45)). Inequality (13) holds by the fact that V ref ,k h ` 1 ě V REF h ` 1 for any k, h . Now we turn to bound ř K k ' 1 ζ k h . Note that

The first term in the RHS of p 14 q is bounded by ř K k ' 1 I r n k h ' 0 s ď SAH because n k h ě H when N k h p s k h , a k h q ě H . We rewrite the second term as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let j ě 1 be a fixed episode. Note that ř ˇ n k h i ' 1 I r j ' ˇ l k h,i s ' 1 if and only if p s j h , a j h q ' p s k h , a k h q , and p j, h q falls in the previous stage that p k, h q falls in. As a result, every k such that ř ˇ n k h i ' 1 I r j ' ˇ l k h,i s ' 1 has the same ˇ n k h which we denote by Z j , and the set t k : ř ˇ n k h i ' 1 I r j ' ˇ l k h,i s ' 1 u has at most p 1 ` 1 H q Z j elements. Therefore, for every j we have that

<!-- formula-not-decoded -->

Because δ k h ` 1 ď ζ k h ` 1 , combining (14), (15), and (16), we have that

<!-- formula-not-decoded -->

Iterating the derivation above for h ' 1 , 2 , ¨ ¨ ¨ , H and we have that

<!-- formula-not-decoded -->

We bound ř H h ' 1 ř K k ' 1 p 1 ` 1 H q h ´ 1 Λ k h ` 1 in the lemma below. The detailed proof is deferred to Appendix B due to space constraints.

Lemma 7. With probability at least p 1 ´ O p H 2 T 4 p qq , it holds that

<!-- formula-not-decoded -->

Combining Proposition 4, Lemma 5, (18) and Lemma 7, we conclude that with probability at least p 1 ´ O p H 2 T 4 p qq ,

<!-- formula-not-decoded -->

## References

- Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worstcase regret bounds. In Advances in Neural Information Processing Systems , pages 1184-1194, 2017.
- Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. arXiv preprint arXiv:1703.05449 , 2017.
- Yu Bai, Tengyang Xie, Nan Jiang, and Yu-Xiang Wang. Provably efficient q-learning with low switching cost. In Advances in Neural Information Processing Systems , pages 8002-8011, 2019.
- A. N. Burnetas and M. N. Katehakis. Optimal Adaptive Policies for Markov Decision Processes . 1997.
- Marc Deisenroth and Carl E Rasmussen. Pilco: A model-based and data-efficient approach to policy search. In Proceedings of the 28th International Conference on machine learning (ICML-11) , pages 465-472, 2011.
- Kefan Dong, Yuanhao Wang, Xiaoyu Chen, and Liwei Wang. Q-learning with ucb exploration is sample efficient for infinite-horizon mdp. arXiv preprint arXiv:1901.09311 , 2019.
- David A Freedman et al. On tail probabilities for martingales. the Annals of Probability , 3(1): 100-118, 1975.
- Ronan Fruit, Matteo Pirotta, and Alessandro Lazaric. Improved analysis of ucrl2b. 2019.
- Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(Apr):1563-1600, 2010.
- Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Advances in Neural Information Processing Systems , pages 4863-4873, 2018.
- Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in neural information processing systems , pages 315-323, 2013.
- Sham Kakade, Mengdi Wang, and Lin F Yang. Variance reduction methods for sublinear reinforcement learning. arXiv preprint arXiv:1802.09184 , 2018.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, 2015.
- Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning , pages 1928-1937, 2016.
- Yi Ouyang, Mukul Gagrani, Ashutosh Nayyar, and Rahul Jain. Learning unknown markov decision processes: A thompson sampling approach. In Advances in Neural Information Processing Systems , pages 1333-1342, 2017.

- John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning , pages 1889-1897, 2015a.
- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438 , 2015b.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving markov decision processes with a generative model. In Advances in Neural Information Processing Systems , pages 5186-5196, 2018a.
- Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 770-787. SIAM, 2018b.
- Max Simchowitz and Kevin G Jamieson. Non-asymptotic gap-dependent regret bounds for tabular mdps. In Advances in Neural Information Processing Systems , pages 1151-1160, 2019.
- Alexander L Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L Littman. Pac modelfree reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , pages 881-888, 2006.
- Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems , pages 1057-1063, 2000.
- Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Van Hasselt, Marc Lanctot, and Nando De Freitas. Dueling network architectures for deep reinforcement learning. arXiv preprint arXiv:1511.06581 , 2015.
- Christopher Watkins. Learning from delayed rewards. 1989. Ph.D. thesis.
- Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. arXiv preprint arXiv:1901.00210 , 2019.
- Zihan Zhang and Xiangyang Ji. Regret minimization for reinforcement learning by evaluating the optimal bias function. arXiv preprint arXiv:1906.05110 , 2019.

## Appendices

## A Basic Lemmas

Lemma 8 (Azuma-Hoeffding Inequality) . Suppose t X k u k ' 0 , 1 , 2 ,... is a martingale and | X k ´ X k ´ 1 | ď c k almost surely. Then for all positive integers N and all positive reals /epsilon1 , it holds that

Lemma 9 (Freedman's Inequality, Theorem 1.6 of [Freedman et al., 1975]) . Let p M n q n ě 0 be a martingale such that M 0 ' 0 and | M n ´ M n ´ 1 | ď c . Let Var n ' ř n k ' 1 E rp M k ´ M k ´ 1 q 2 | F k ´ 1 s for n ě 0 , where F k ' σ p M 0 , M 1 , M 2 , ..., M k q . Then, for any positive x and for any positive y ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 10. Let p M n q n ě 0 be a martingale such that M 0 ' 0 and | M n ´ M n ´ 1 | ď c for some c ą 0 and any n ě 1 . Let Var n ' ř n k ' 1 E rp M k ´ M k ´ 1 q 2 | F k ´ 1 s for n ě 0 , where F k ' σ p M 1 , M 2 , ..., M k q . Then for any positive integer n , and any /epsilon1, p ą 0 , we have that

<!-- formula-not-decoded -->

Proof. For any fixed n , we apply Lemma 9 with y ' i/epsilon1 and x ' ˘p 2 b y log p 1 p q ` 2 c log p 1 p qq . For each i ' 1 , 2 , . . . , r nc 2 /epsilon1 s , we get that

<!-- formula-not-decoded -->

Then via a union bound, we have that

<!-- formula-not-decoded -->

Lemma 11. For any non-negative weights t w h p s, a qu s P S ,a P A ,h Pr H s and α P p 0 , 1 q , it holds that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

In the case α ' 1 , it holds that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. By the definition of L , for any h, k such that n k h ą 0 , there exists j such that ˇ n k h ' e j and n k h ' ř j i ' 1 e i . Therefore, 1 2 H n k h ď ˇ n k h ď 3 H n k h . So it suffices to prove (24) and (25). By basic calculus, for two positive numbers x, y such that y { 2 ď x ď y and any α P p 0 , 1 q , we have that and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By applying (26) and p 27 q with y ' ř j ` 1 i ' 1 e i and x ' ř j i ' 1 e i for j ' 1 , 2 , ... and taking sum, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## B Missing Proofs in the Regret Analysis

## B.1 Proof of Proposition 4

We prove Q ˚ h p s, a q ď Q k h p s, a q for all k, h, s, a, by induction on k . Firstly, the conclusion holds when k ' 1 . For k ě 2 , assume Q ˚ h p s, a q ď Q u h p s, a q for any h, s, a and 1 ď u ď k . Let p s, a, h q be fixed. If we do not update Q h p s, a q in the k -th episode, then Q k ` 1 h p s, a q ' Q k h p s, a q ě Q ˚ h p s, a q . Otherwise, we have

<!-- formula-not-decoded -->

( ii )).

where µ ref , ˇ µ , σ ref , ˇ σ , n , ˇ n , b and b are given by respectively the values of µ ref , ˇ µ , σ ref , σ , n , ˇ n , b and b to compute Q k ` 1 h p s, a q in (9). We use l i to denote the episode index of the i -th sample and ˇ l i to denote the episode index of the i -th sample of the last stage with respect to the triple p s, a, h q . Besides the last Q k h p s, a q term, there are two non-trivial cases to discuss (corresponding to ( i ) and

For the first case, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, Inequality (29) holds because V ref ,u h ` 1 is non-increasing in u , Inequality (30) is by the induction V u ě V ˚ for any 1 ď u ď k .

Define V p x, y q : ' x J p y 2 q ´ p x J y q 2 for two vectors x, y of the same dimension, where y 2 is obtained by squaring each entry of y . By Lemma 10 with /epsilon1 ' 1 T 2 , we have that with probability p 1 ´ 2 p H 2 T 3 ` 1 q p q it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now bound ř ˇ n i ' 1 V p P s,a,h , V ref ,l i h ` 1 q in order to upper bound | χ 1 | . Define

We claim that,

Lemma 12. With probability p 1 ´ 2 p q , it holds that

Proof. We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Azuma's inequality, we have | χ 3 | ď H 2 ? 2 nι with probability at least p 1 ´ p q . We apply Azuma's inequality again to obtain that with probability at least p 1 ´ p q , it holds that a Onthe other hand, we have that χ 5 ď 0 by Cauchy-Schwartz inequality. The proof then is completed by (37).

Combing (34) with (36) we have

<!-- formula-not-decoded -->

We now bound ř ˇ n i ' 1 V p P s,a,h , W ˇ l i h ` 1 q for | χ 2 | . Define

Similarly to Lemma 12, we have that

<!-- formula-not-decoded -->

Lemma 13. With probability p 1 ´ 2 p q , it holds that

Therefore, given (35), it holds with probability p 1 ´ 2 p q that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, combining (42), (44), and the definition of b with r c 1 , c 2 , c 3 s ' r 2 , 2 , 5 s , and collecting probabilities, we have that with probability at least p 1 ´ 2 p H 2 T 3 ` 3 qq p , it holds that which means that Q k ` 1 h p s, a q ě Q ˚ h p s, a q .

<!-- formula-not-decoded -->

For the second case, by Hoeffding's inequality, with probability p 1 ´ p q it holds that

Combining the two cases, and via a union bound over all time steps, we prove the proposition.

## B.2 Proof of Lemma 5

First, by Hoeffding's inequality, for every k and h , we have that

<!-- formula-not-decoded -->

ˇ ˇ Now the whole proof will be conditioned on that (47) holds for every k and h , which happens with probability at least p 1 ´ Tp q . For every k and h , we let δ k h : ' V k h p s k h q ´ V ˚ h p s k h q (which aligns with the definition for δ k h in the proof of Theorem 1).

For any weight sequence t w k u K k ' 1 such that w k ě 0 , let } w } 8 ' max K k ' 1 w k and } w } 1 ' ř K k ' 1 w k . We will prove that

Once we have established (48), we let w k ' I r δ k h ě /epsilon1 s and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that } w } 8 is either 0 or 1 . In either cases, we are able to derive that

<!-- formula-not-decoded -->

and concludes the proof of the lemma. Therefore, we only need to prove (48), and the rest of the proof is devoted to establishing (48).

By the update rule (9) and (10), that V k always upper bounds V ˚ (conditioned on the successful event of Proposition 4), and that we have conditioned on (47), we have that

<!-- formula-not-decoded -->

Using the similar trick we do for (15) and (16), we have

<!-- formula-not-decoded -->

where if we let

<!-- formula-not-decoded -->

we have that

<!-- formula-not-decoded -->

Therefore, combining (49), (50), and (51), and plugging them into k w k δ k h , we have that

<!-- formula-not-decoded -->

We now bound the first term of (53). Define w p s, a, j q : ' ř K k ' 1 w k I r ˇ n k h ' e j , p s k h , a k h q ' p s, a qs and w p s, a q : ' ř j ě 1 w p s, a, j q . We have w p s, a, j q ď } w } 8 p 1 ` 1 H q e j and ř s,a w p s, a q ' ř k w k . We then have

<!-- formula-not-decoded -->

We fix p s, a q and consider the sum ř j ě 1 w p s, a, j q b 1 e j . Notice that a 1 { e j is monotonically decreasing in j . Given that ř j ě 0 w p s, a, j q ' w p s, a q is fixed, by rearrangement inequality we have that

Therefore, by Cauchy-Schwartz, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (53) and (54), we have that

<!-- formula-not-decoded -->

With (55) and (52) in hand, applying induction on h with the base case that h ' H , one may deduce that

<!-- formula-not-decoded -->

## B.3 Proof of Lemma 7

The entire proof is conditioned on the successful events of Proposition 4 and Lemma 5 which happen with probability at least p 1 ´ 2 T p H 2 T 3 ` 5 q p q . For convenience, we define λ k h as λ k h p s q ' I ' n k h p s q ă N 0 ‰ for all state s and all k and h .

By the definition of Λ k h ` 1 , we have that ř H h ' 1 ř K k ' 1 p 1 ` 1 H q h ´ 1 Λ k h ` 1 by the definition that

We will bound the four terms separately.

<!-- formula-not-decoded -->

## B.3.1 The ψ k h ` 1 Term

Lemma 14. With probability at least p 1 ´ p q , it holds that

<!-- formula-not-decoded -->

Proof. Because ψ k h ` 1 is always non-negative, we have that with probability p 1 ´ p q it holds that

<!-- formula-not-decoded -->

Here, Inequality (57) is because 1 n k h ř n k h i ' 1 I r l k h,i ' j s ‰ 0 only if p s k h , a k h q ' p s j h , a j h q . Inequality (58) is because

<!-- formula-not-decoded -->

Inequality (59) holds with probability p 1 ´ p q due to Azuma's inequality.

## B.3.2 The ξ k h ` 1 Term

Proof. We have that

<!-- formula-not-decoded -->

Note that in the expression above ˇ l k h,i ' j if and only if p s k h , a k h q ' p s j h , s j h q . Therefore, we have

<!-- formula-not-decoded -->

For p j, h q P r K s ˆ r H s , let x j h be the number of elements in current stage with respect to p s j h , a j h , h q and ˜ θ j h ` 1 : ' p 1 ` 1 H q h ´ 1 t p 1 ` 1 H q x j h u x j h ď 3 . Define K ' tp k, h q : θ k h ` 1 ' ˜ θ k h ` 1 u . Note that if k is before the second last stage (before the final episode K ) of the triple p s k h , a k h , h q , then we have θ k h ` 1 ' ˜ θ k h ` 1 and p k, h q P K . Given that p k, h q P K , s k h ` 1 still follows the transition distribution P s k h ,a k h ,h .

where we define θ j h ` 1 : ' p 1 ` 1 H q h ´ 1 ř K k ' 1 ` 1 ˇ n k h ř ˇ n k h i ' 1 I r ˇ l k h,i ' j s ˘ .

Let K K h p s, a q ' t k : p s k h , a k h q ' p s, a q , k is in the second last stage of p s, a, h qu . Note that for two different episodes j, k , if p s k h , a k h q ' p s j h , a j h q and j, k are in the same stage of p s k h , a k h , h q , then θ k h ` 1 ' θ j h ` 1 and ˜ θ k h ` 1 ' ˜ θ j h ` 1 . Let θ h ` 1 p s, a q and ˜ θ h ` 1 p s, a q to denote θ k h ` 1 and ˜ θ k h ` 1 respectively for some k P K K h p s, a q .

We rewrite as

<!-- formula-not-decoded -->

Because ˜ θ k h ` 1 is independent from s k h ` 1 , by Azuma's inequality, we have with probability p 1 ´ p q , it holds that

<!-- formula-not-decoded -->

Lemma 15. With probability at least p 1 ´p T ` 1 q p q , it holds that

<!-- formula-not-decoded -->

For the second term in (61), we have that

<!-- formula-not-decoded -->

Combining (61), (62), (65), and collecting probabilities, we prove the desired result.

## B.3.3 The φ k h ` 1 Term

<!-- formula-not-decoded -->

Lemma 16. With probability p 1 ´ p q , it holds that

Proof. The lemma follows easily from Azuma's inequality.

## B.3.4 The b k h Term

Lemma 17. With probability p 1 ´ 9 p q , it holds that

Proof. Define ν ref ,k h ' σ ref ,k n k h ´p µ ref ,k h n k h q 2 and ˇ ν k h ' ˇ σ k h ˇ n k h ´p ˇ µ k h ˇ n k h q 2 . Since b k h is non-negative, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Inequality (67) is due to Lemma 11 with α ' 3 4 and α ' 1 . Now we only need to analyze the first term in (67).

We first present an upper bound for ν ref ,k h . Recall that V p x, y q ' x J p y 2 q ´ p x J y q 2 .

Lemma 18. With probability p 1 ´ 4 p q , it holds that

<!-- formula-not-decoded -->

Proof. We prove by first bounding ν ref ,k h ´ 1 n k h ř n k h i ' 1 V p P s k h ,a k h ,h , V ref ,l i h ` 1 q . Recall that by (37), where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Azuma's inequality, with probability p 1 ´ 2 p q it holds that

<!-- formula-not-decoded -->

It left us to handle ´ χ 8 . By Azuma's inequality and the fact that V ref ,k ě V REF for any k , with probability p 1 ´ p q it holds that

<!-- formula-not-decoded -->

Then we obtain that

<!-- formula-not-decoded -->

When (72) holds, we have that with probability p 1 ´ p q ,

<!-- formula-not-decoded -->

where Inequality (73) holds with probability p 1 ´ p q by Azuma's inequality and (74) holds by Corollary 6 (and note that the whole proof is conditioned on the successful events of Proposition 4 and Lemma 5).

We will also prove the following bound of the total variance.

Lemma 19. With probability p 1 ´ 2 p q , it holds that

<!-- formula-not-decoded -->

Proof. By direct calculation, with probability p 1 ´ 2 p q , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Inequality (76) holds with probability p 1 ´ p q by Azuma's inequality, Equation (77) holds with the fact that V ˚ h p s q ´ P s,a,h V ˚ h ` 1 ě V ˚ h p s q ´ Q ˚ h p s, a q ě 0 for any s, a, h and Inequality (78) holds with probability p 1 ´ p q by Azuma's inequality.

Combining Lemma 11, Lemma 18, and Lemma 19, we have that with probability p 1 ´ 7 p q ,

We now bound ˇ ν k h . By Corollary 6 (and that the whole proof is conditioned on the successful events of Proposition 4 and Lemma 5), we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Lemma 11, we obtain that

<!-- formula-not-decoded -->

The proof is completed by combining (67), (79), and (81).

## B.3.5 Putting Everything Together

Recall that β ' 1 ? H , and N 0 ' c 4 SAH 5 ι β 2 ' O p SAH 6 ι q . Combining (56), Lemma 14, Lemma 15, Lemma 16 and Lemma 17, we conclude that with probability at least p 1 ´ O p H 2 T 4 p qq ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Other Results

## C.1 Local Switching Cost Analysis

The notion of local switching cost for RL is introduced in [Bai et al., 2019] to quantify the adaptivity of the learning algorithms. With a slight abuse of notations, we use π k,h to denote the policy at the h -th step of the k -th episode. We first recall formal definition of the local switching cost.

Definition 1. The local switching cost at p s, h q is defined as

<!-- formula-not-decoded -->

The total local switching cost is then defined as

<!-- formula-not-decoded -->

Now we prove Theorem 2.

Proof of Theorem 2. By the definition of e i , it is easy to verify that e i ` 1 ě p 1 ` 1 2 H q e i for any i ě 1 . Then the number of stages of p s, a, h q is at most

<!-- formula-not-decoded -->

Because π k,h p s q ' arg max a Q k h p s, a q , we have that

Now, by definition, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, by the concavity of log p x q in x , the total local switching cost of UCB-ADVANTAGE is bounded by

<!-- formula-not-decoded -->

## Algorithm 2 Concurrent UCB-ADVANTAGE

Initialize: Q h p s, a q Ð H ´ h ` 1 , k Ð 1 , K /epsilon1 Ð c 5 SAH 3 log p SAH /epsilon1 q /epsilon1 2 M ( c 5 is a large enough universal constant).

for concurrent episodes

k

1

,

2

,

3

, . . .

All agents follow the same policy for

i

1

,

'

2

,

3

, . . . , M

do

Collect the trajectory of the if

'

an update is triggered

Update

Q

do where

q '

i

p

p

q

-th agent and feed it to UCB-ADVANTAGE

then

-value function following UCB-ADVANTAGE;

break end if

end for if

The number of trajectories use is greater than or equal to break

end if end for

## C.2 Application to Concurrent RL

In concurrent RL, multiple agents act in parallel and shares the experience in a limited way to accelerate the learning process. In this subsection, we follow the setting in [Bai et al., 2019] to introduce the problem.

Suppose there are M parallel agents, where each agent interacts with the environments independently. In the concurrent RL problem, each agent finishes an episode simultaneously, so that there are M episodes done per concurrent round. The agents can only exchange experience and update their policies at the end of each round. The goal is to find an /epsilon1 -optimal policy using the minimum number of rounds, which we also refer to as the number of concurrent episodes.

In Algorithm 2, we present the details of the concurrent UCB-ADVANTAGE algorithm. The idea is to simulate the single-agent UCB-ADVANTAGE by treating the M episodes finished in a single round as M consecutive episodes (without policy change) in the single-agent setting. We collect the trajectories and feed them to the single-agent UCB-ADVANTAGE. When an update is triggered in the single-agent UCB-ADVANTAGE during an episode, we update the Q -function (as well as the value function) and discard the trajectories left in the round.

We now prove Corollary 3 that shows the performance of the concurrent UCB-ADVANTAGE.

Proof of Corollary 3. The proof follows the similar lines in the proof of Theorem 5 in [Bai et al., 2019]. By Theorem 2, the switching cost is at most O p H 2 SA log p K /epsilon1 SAH qq , so there are at most

<!-- formula-not-decoded -->

concurrent episodes. On the other hand, the regret incurred in the episodes corresponding to K /epsilon1 is at most ˜ O p ? SAH 3 K /epsilon1 q ď K /epsilon1 /epsilon1 , so by randomly choosing an episode index k and selecting π ' π k we achieve a policy with expected performance at most /epsilon1 below the optimum.

## C.3 Lower Bound of the Sample Complexity

Theorem 20. For any H , S , and A greater than a universal constant, and all /epsilon1 P p 0 , 8 H s , for any algorithm with input parameter /epsilon1 , there exists an episodic MDP with S states, A actions, horizon H such that, with probability at least 1 { 2 , among the execution history of the algorithm, there are at least Ω p SAH 3 { /epsilon1 2 q episodes in which the corresponding policy π k satisfies that V ˚ 1 p s k 1 q ´ V π k 1 p s k 1 q ą /epsilon1 .

Proof Sketch. Instead of presenting a concrete proof of Theorem 20, we provide the high-level intuition in the construction and analysis.

Like the regret lower bound analysis in [Jin et al., 2018], we consider the special case where S ' A ' 2 . It does not require too much difficulty to generalize to arbitrary S and A . Also, we will use almost the same hard instance as constructed in the proof of Theorem 3 in [Jin et al., 2018].

π

k

π

k,h

s

arg max

K

/epsilon1

a

Q

h

then s, a

.

We recall the structure of 'JAO MDP' in [Jaksch et al., 2010]. There are two states in the MDP, named s 0 and s 1 . The rewards are defined as r p s 0 , a q ' 0 and r p s 1 , a q ' 1 for any a and the transition probabilities are defined as P p¨| s 1 , a q ' r δ, 1 ´ δ s J , @ a , P p¨| s 0 , a q ' r 1 ´ δ, δ s J , @ a ‰ a ˚ and P p¨| s 0 , a ˚ q ' r 1 ´ δ ´ /epsilon1, δ ` /epsilon1 s J . Clearly the optimal action for state s 0 is a ˚ . Let δ ă 1 2 be fixed. By the lower bound of [Jaksch et al., 2010], there exists a constant c 5 ą 0 , such that for any /epsilon1 P p 0 , δ 2 q , it costs at least c 5 ¨ δ /epsilon1 2 observations to identify a ˚ with non-trivial probability.

By connecting H JAO MDPs with different optimal actions layer by layer, we get an episodic MDP with horizon H . We choose δ ' 16 H to ensure that the MDP is well-mixed for h ě H 2 . For any /epsilon1 ď 8 H ' δ 2 and h ě H 2 , the agent reaches s 0 in the h -th layer with at least constant probability. If there are at least 7 H 8 layers in which the agent can not identify a ˚ , then the agent makes Ω p H q mistakes in the range h P r H 2 , 3 H 4 s . Because each mistake for h P r H 2 , 3 H 4 s leads to Ω p /epsilon1H q regret , the expected regret incurred during one episode is Ω p /epsilon1H 2 q . As a result, if the total number of observations is less than c 5 H 8 ¨ δ /epsilon1 2 (i.e., number of episodes less than c 5 8 ¨ δ /epsilon1 2 ), the expected regret per episode is Ω p /epsilon1H 2 q . Replacing /epsilon1 by /epsilon1H 2 , we have that for the first Θ p δH 4 { /epsilon1 2 q ' Θ p H 3 { /epsilon1 2 q episodes, the expected regret per episode is Ω p /epsilon1 q . The proof is then completed by applying Markov's inequality.