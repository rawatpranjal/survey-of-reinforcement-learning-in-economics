HAL

open science

<!-- image -->

## Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model

Mohammad Gheshlaghi Azar, Rémi Munos, Hilbert Kappen

## To cite this version:

Mohammad Gheshlaghi Azar, Rémi Munos, Hilbert Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine Learning, 2013, 91 (3), pp.325349. ￿10.1007/s10994-013-5368-1￿. ￿hal-00831875￿

## HAL Id: hal-00831875 https://hal.science/hal-00831875v1

Submitted on 7 Jun 2013

HAL is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire HAL , est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

## Minimax PAC Bounds on the Sample Complexity of Reinforcement Learning with a Generative Model

- Mohammad Gheshlaghi Azar · R´ emi Munos · Hilbert J. Kappen

Received: date / Accepted: date

Abstract We consider the problem of learning the optimal action-value function in discounted-reward Markov decision processes (MDPs). We prove new PAC bounds on the sample-complexity of two well-known model-based reinforcement learning (RL) algorithms in the presence of a generative model of the MDP: value iteration and policy iteration. The first result indicates that for an MDP with N state-action pairs and the discount factor γ ∈ [0 , 1) only O ( N log( N/δ ) / ( (1 -γ ) 3 ε 2 )) state-transition samples are required to find an ε -optimal estimation of the action-value function with the probability (w.p.) 1 -δ . Further, we prove that, for small values of ε , an order of O ( N log( N/δ ) / ( (1 -γ ) 3 ε 2 )) samples is required to find an ε -optimal policy w.p. 1 -δ . We also prove a matching lower bound of Θ ( N log( N/δ ) / ( (1 -γ ) 3 ε 2 )) on the sample complexity of estimating the optimal action-value function. To the best of our knowledge, this is the first minimax result on the sample complexity of RL: The upper bound matches the lower bound in terms of N , ε , δ and 1 / (1 -γ ) up to a constant factor. Also, both our lower bound and upper bound improve on the state-of-the-art in terms of their dependence on 1 / (1 -γ ).

An extended abstract of this paper appeared in Proceedings of International Conference on Machine Learning (ICML 2012).

M. Gheshlaghi Azar Department of Biophysics Radboud University Nijmegen 6525 EZ Nijmegen, The Netherlands E-mail: m.azar@science.ru.nl

R. Munos INRIA Lille, SequeL Project 40 avenue Halley 59650 Villeneuve d'Ascq, France E-mail: remi.munos@inria.fr

H. J. Kappen Department of Biophysics Radboud University Nijmegen 6525 EZ Nijmegen, The Netherlands E-mail: b.kappen@science.ru.nl

Keywords sample complexity · Markov decision processes · reinforcement learning · learning theory

## 1 Introduction

An important problem in the field of reinforcement learning (RL) is to estimate the optimal policy (or the optimal value function) from the observed rewards and the transition samples (Sutton and Barto, 1998; Szepesv´ ari, 2010). To estimate the optimal policy one may use model-free or model-based approaches. In modelbased RL, we first learn a model of the MDP using a batch of state-transition samples and then use this model to estimate the optimal policy or the optimal action-value function using the Bellman recursion, whereas model-free methods directly aim at estimating the optimal value function without resorting to learning an explicit model of the dynamical system. The fact that the model-based RL methods decouple the model-estimation problem from the value (policy) iteration problem may be useful in problems with a limited budget of sampling. This is because the model-based RL algorithms, after learning the model, can perform many Bellman recursion steps without any need to make new transition samples, whilst the model-free RL algorithms usually need to generate fresh samples at each step of value (policy) iteration process.

The focus of this article is on model-based RL algorithms for finite stateaction problems, where we have access to a generative model (simulator) of the MDP. Especially, we derive tight sample-complexity upper bounds for two wellknown model-based RL algorithms, the model-based value iteration and the modelbased policy iteration (Wiering and van Otterlo, 2012), It has been shown (Kearns and Singh, 1999; Kakade, 2004, chap. 9.1) that an action-value based variant of model-based value iteration algorithm, Q-value iteration (QVI), finds an ε -optimal estimate of the action-value function with high probability (w.h.p.) using only ˜ O ( N/ ( (1 -γ ) 4 ε 2 ) ) samples, where N and γ denote the size of state-action space and the discount factor, respectively. 1 One can also prove, using the result of Singh and Yee (1994), that QVI w.h.p. finds an ε -optimal policy using an order of ˜ O ( N/ ( (1 -γ ) 6 ε 2 ) ) samples. An upper-bound of a same order can be proven for model-based PI. These results match the best upper-bound currently known (Azar et al, 2011b) for the sample complexity of RL. However, there exist gaps with polynomial dependency on 1 / (1 -γ ) between these upper bounds and the state-of-the-art lower bound, which is of order ˜ Ω ( N/ ((1 -γ ) 2 ε 2 ) ) (Azar et al, 2011a; Even-Dar et al, 2006). 2 It has not been clear, so far, whether the upper bounds or the lower bound can be improved or both.

In this paper, we prove new bounds on the performance of QVI and PI which indicate that for both algorithms with the probability (w.p) 1 -δ an order of O ( N log( N/δ ) / ( (1 -γ ) 3 ε 2 )) samples suffice to achieve an ε -optimal estimate of action-value function as well as to find an ε -optimal policy. The new upper bound improves on the previous result of AVI and API by an order of 1 / (1 -γ ). We also

1 The notation g = ˜ O ( f ) implies that there are constants c 1 and c 2 such that g ≤ c 1 f log c 2 ( f ).

2 The notation g = ˜ Ω ( f ) implies that there are constants c 1 and c 2 such that g ≥ c 1 f log c 2 ( f ).

present a new minimax lower bound of Θ ( N log( N/δ ) / ( (1 -γ ) 3 ε 2 )) , which also improves on the best existing lower bound of RL by an order of 1 / (1 -γ ). The new results, which close the above-mentioned gap between the lower bound and the upper bound, guarantee that no learning method, given the generative model of the MDP, can be significantly more efficient than QVI and PI in terms of the sample complexity of estimating the optimal action-value function or the optimal policy.

The main idea to improve the upper bound of the above-mentioned RL algorithms is to express the performance loss Q ∗ -Q k , where Q k is the estimate of the action-value function after k iteration of QVI or PI, in terms of Σ π ∗ , the variance of the sum of discounted rewards under the optimal policy π ∗ , as opposed to the maximum V max = R max / (1 -γ ) as was used before. For this we make use of the Bernstein's concentration inequality (Cesa-Bianchi and Lugosi, 2006, appendix, pg. 361), which is expressed in terms of the variance of the random variables. We also rely on the fact that the variance of the sum of discounted rewards, like the expected value of the sum (value function), satisfies a Bellman-like equation, in which the variance of the value function plays the role of the instant reward in the standard Bellman equation (Munos and Moore, 1999; Sobel, 1982). These results allow us to prove a high-probability bound of order ˜ O ( √ Σ π ∗ / ( n (1 -γ ))) on the performance loss of both algorithms, where n is the number of samples per state-action. This leads to a tight PAC upper-bound of ˜ O ( N/ ( ε 2 (1 -γ ) 3 )) on the sample complexity of these methods.

In the case of lower bound, we introduce a new class of 'hard' MDPs, which adds some structure to the bandit-like class of MDP used previously by Azar et al (2011a); Even-Dar et al (2006): In the new model, there exist states with high probability of transition to themselves. This adds to the difficulty of estimating the value function, since even a small modeling error may cause a large error in the estimate of the optimal value function, especially when the discount factor γ is close to 1.

The rest of the paper is organized as follows. After introducing the notations used in the paper in Section 2, we describe the model-based Q-value iteration (QVI) algorithm and the model-based policy iteration (PI) in Subsection 2.1. We then state our main theoretical results, which are in the form of PAC sample complexity bounds in Section 3. Section 4 contains the detailed proofs of the results of Sections 3, i.e., sample complexity bound of QVI and a matching lower bound for RL. Finally, we conclude the paper and propose some directions for the future work in Section 5.

## 2 Background

In this section, we review some standard concepts and definitions from the theory of Markov decision processes (MDPs). We then present two model-based RL algorithms which make use of generative model for sampling: the model-based Q-value iteration and the model-based policy iteration (Wiering and van Otterlo, 2012; Kearns and Singh, 1999).

We consider the standard reinforcement learning (RL) framework (Bertsekas and Tsitsiklis, 1996; Sutton and Barto, 1998), where an RL agent interacts with

a stochastic environment and this interaction is modeled as a discrete-time discounted MDP. A discounted MDP is a quintuple ( X , A , P, R , γ ), where X and A are the set of states and actions, P is the state transition distribution, R is the reward function, and γ ∈ [0 , 1) is a discount factor. 3 We denote by P ( ·| x, a ) and r ( x, a ) the probability distribution over the next state and the immediate reward of taking action a at state x , respectively.

To keep the representation succinct, in the sequel, we use the notation Z for the joint state-action space X × A . We also make use of the shorthand notations z and β for the state-action pair ( x, a ) and 1 / (1 -γ ), respectively.

Assumption 1 (MDP Regularity) We assume Z and, subsequently, X and A are finite sets with cardinalities N , | X | and | A | , respectively. We also assume that the immediate reward r ( x, a ) is taken from the interval [0 , 1] . 4

A mapping π : X → A is called a stationary and deterministic Markovian policy, or just a policy in short. Following a policy π in an MDP means that at each time step t the control action A t ∈ A is given by A t = π ( X t ), where X t ∈ X . The value and the action-value functions of a policy π , denoted respectively by V π : X → R and Q π : Z → R , are defined as the expected sum of discounted rewards that are encountered when the policy π is executed. Given an MDP, the goal is to find a policy that attains the best possible values, V ∗ ( x ) glyph[defines] sup π V π ( x ) , ∀ x ∈ X . The function V ∗ is called the optimal value function . Similarly the optimal actionvalue function is defined as Q ∗ ( x, a ) = sup π Q π ( x, a ). We say that a policy π ∗ is optimal if it attains the optimal V ∗ ( x ) for all x ∈ X . The policy π defines the state transition kernel P π as P π ( y | x ) glyph[defines] P ( y | x, π ( x )) for all x ∈ X . The right-linear operators P π · , P · and P π · are also defined as ( P π Q )( z ) glyph[defines] ∑ y ∈ X P ( y | z ) Q ( y, π ( y )), ( PV )( z ) glyph[defines] ∑ y ∈ X P ( y | z ) V ( y ) for all z ∈ Z and ( P π V )( x ) glyph[defines] ∑ y ∈ X P π ( y | x ) V ( y ) for all x ∈ X , respectively. The optimal action-value function Q ∗ is the unique fixed-point of the Bellman optimality operator defined as

<!-- formula-not-decoded -->

Also, for the policy π , the action-value function Q π is the unique fixed-point of the Bellman operator T π which is defined as ( T π Q )( z ) glyph[defines] r ( z ) + γ ( P π Q )( z ) for all z ∈ Z . One can also define the Bellman optimality operator and the Bellman operator on the value function as ( T V )( x ) glyph[defines] r ( x, π ∗ ( x )) + γ ( P π ∗ V )( x ) and ( T π V )( x ) glyph[defines] r ( x, π ( x )) + γ ( P π V )( x ) for all x ∈ X , respectively.

It is important to note that T and T π are γ -contractions, i.e., for any pair of value functions V and V ′ and any policy π , we have ‖ T V -T V ′ ‖ ≤ γ ‖ V -V ′ ‖ and ‖ T π V -T π V ′ ‖ ≤ γ ‖ V -V ′ ‖ (Bertsekas, 2007, Chap. 1). ‖ · ‖ shall denote the supremum ( glyph[lscript] ∞ ) norm, defined as ‖ g ‖ glyph[defines] max y ∈ Y | g ( y ) | , where Y is a finite set and g : Y → R is a real-valued function. We also define the glyph[lscript] 1 -norm on the function g as ‖ g ‖ 1 = ∑ y ∈ Y | g ( y ) | .

For ease of exposition, in the sequel, we remove the dependence on z and x , e.g., writing Q for Q ( z ) and V for V ( x ), when there is no possible confusion.

3 For simplicity, here we assume that the reward r ( x, a ) is a deterministic function of stateaction pairs ( x, a ). Nevertheless, It is straightforward to extend our results to the case of stochastic rewards under some mild assumption, e.g., boundedness of the absolute value of the rewards.

4 Our results also hold if the rewards are taken from some interval [ r min , r max ] instead of [0 , 1], in which case the bounds scale with the factor r max -r min .

## 2.1 Algorithms

We begin by describing the procedure which is used by both PI and QVI to make an empirical estimate of the state-transition distributions.

The model estimator makes n transition samples for each state-action pair z ∈ Z for which it makes n calls to the generative model, i.e., the total number of calls to the generative model is T = nN . It then builds an empirical model of the transition probabilities as ̂ P ( y | z ) glyph[defines] m ( y, z ) /n , where m ( y, z ) denotes the number of times that the state y ∈ X has been reached from the state-action pair z ∈ Z (see Algorithm 3). Based on the empirical model ̂ P the operator ̂ T is defined on the action-value function Q , for all z ∈ Z , by ̂ T Q ( z ) = r ( z )+ γ ( ̂ PV )( z ), with V ( x ) = max a ∈ A ( Q ( x, a )) for all x ∈ X . Also, the empirical operator ̂ T π is defined on the action-value function Q , for every policy π and all z ∈ Z , by ̂ T π Q ( z ) = r ( z ) + γ ̂ P π Q ( z ). Likewise, one can also define the empirical Bellman operator ̂ T and ̂ T π for the value function V . The fixed points of the operator ̂ T in Z and X domains are denoted by ̂ Q ∗ and ̂ V ∗ , respectively. Also, the fixed points of the operator ̂ T π in Z and X domains are denoted by ̂ Q π and ̂ V π , respectively. The empirical optimal policy ̂ π ∗ is the policy which attains ̂ V ∗ under the model ̂ P .

Having the empirical model ̂ P estimated, QVI and PI rely on standard value iteration and policy iteration schemes to estimate the optimal action-value function: QVI iterates some action-value function Q j , with the initial value of Q 0 , through the empirical Bellman optimality operator ̂ T until Q j admits some convergence criteria. PI, in contrast, relies on iterating some policy π j with the initial value π 0 : At each iteration j &gt; 0, the algorithm solves the dynamic programming problem for a fixed policy π j using the empirical model ̂ P . The next policy π j +1 is then determined as the greedy policy w.r.t. the action-value function ̂ Q π j , that is, π j +1 ( x ) = arg max a ∈ A ̂ Q π j ( x, a ) for all x ∈ X . Note that Q k , as defined by PI and QVI are deferent, but nevertheless we use a same notation for both action-functions since we will show in the next section that they enjoy the same performance guarantees. The pseudo codes of both algorithms are provided in Algorithm 1 and Algorithm 2.

## Algorithm 1 Model-based Q-value Iteration (QVI)

```
Require: reward function r , discount factor γ , initial action-value function Q 0 , samples per state-action n , number of iterations k ̂ P = EstimateModel ( n ) glyph[triangleright] Estimate the model (defined in Algorithm 3) for j := 0 , 1 , . . . , k -1 do for each x ∈ X do π j ( x ) = arg max a ∈ A Q j ( x, a ) glyph[triangleright] greedy policy w.r.t. the latest estimation of Q ∗ for each a ∈ A do ̂ T Q j ( x, a ) = r ( x, a ) + γ ( ̂ P π j Q j )( x, a ) glyph[triangleright] empirical Bellman operator Q j +1 ( x, a ) = ̂ T Q j ( x, a ) glyph[triangleright] Iterate the action-value function Q j end for end for end for return Q k
```

## Algorithm 2 Model-based Policy Iteration (PI)

Require: reward function r , discount factor γ , initial action-value function Q 0 , samples per

```
state-action n , number of iterations k ̂ P = EstimateModel ( n ) glyph[triangleright] Estimate the model (defined in Algorithm 3) for j := 0 , 1 , . . . , k -1 do for each x ∈ X do π j ( x ) = arg max a ∈ A Q j ( x, a ) glyph[triangleright] greedy policy w.r.t. the latest estimation of Q ∗ end for ̂ Q π j = SolveDP ( ̂ P,π j ) glyph[triangleright] Find the fixed point of the Bellman operator for the policy π j Q j +1 = ̂ Q π j glyph[triangleright] Iterate the action-value function Q j end for return Q k function SolveDP ( P, π ) Q = ( I -γP π ) -1 r return Q end function
```

## Algorithm 3 Function: EstimateModel

```
Require: The generative model (simulator) P function EstimateModel ( n ) glyph[triangleright] Estimating the transition model using n samples ∀ ( y, z ) ∈ X × Z : m ( y, z ) = 0 glyph[triangleright] initialization for each z ∈ Z do for i := 1 , 2 , . . . , n do y ∼ P ( ·| z ) glyph[triangleright] Generate a state-transition sample m ( y, z ) := m ( y, z ) + 1 glyph[triangleright] Count the transition samples end for ∀ y ∈ X : ̂ P ( y | z ) = m ( y,z ) n glyph[triangleright] Normalize by n end for return ̂ P glyph[triangleright] Return the empirical model end function
```

## 3 Main Results

Our main results are in the form of PAC (probably approximately correct) sample complexity bounds on the total number of samples required to attain a near-optimal estimate of the action-value function:

## Theorem 1 (PAC-bound on Q ∗ -Q k )

Let Assumption 1 hold and T be a positive integer. Then, there exist some constants c , c 0 , d and d 0 such that for all ε ∈ (0 , 1) and δ ∈ (0 , 1) , a total sampling budget of

<!-- formula-not-decoded -->

suffices for the uniform approximation error ‖ Q ∗ -Q k ‖ ≤ ε , w.p. at least 1 -δ , after k = glyph[ceilingleft] d log( d 0 β/ε ) / log(1 /γ ) glyph[ceilingright] iteration of QVI or PI algorithm. 5

We also prove a similar bound on the sample-complexity of finding a nearoptimal policy for small values of ε :

Theorem 2 (PAC-bound on Q ∗ -Q π k ) Let Assumption 1 hold and T be a positive integer. Define π k as the greedy policy w.r.t. Q k at iteration k of PI or QVI. Then, there exist some constants c ′ , c ′ 0 , c ′ 1 , d ′ and d ′ 0 such that for all ε ∈ (0 , c ′ 1 √ β/ ( γ | X ) | ) and δ ∈ (0 , 1) , a total sampling budget of

<!-- formula-not-decoded -->

suffices for the uniform approximation error ‖ Q ∗ -Q π k ‖ ≤ ε , w.p. at least 1 -δ , after k = d ′ glyph[ceilingleft] log( d ′ 0 β/ε ) / log(1 /γ ) glyph[ceilingright] iteration of QVI or PI algorithm.

The following general result provides a tight lower bound on the number of transitions T for every RL algorithm to find a near optimal solution w.p. 1 -δ , under the assumption that the algorithm is ( ε, δ, T )-correct:

Definition 1 ( ( ε, δ ) -correct algorithm) Let Q A : Z → R be the output of some RL Algorithm A .We say that A is ( ε, δ )-correct on the class of MDPs M = { M 1 , M 2 , . . . , M m } if ∥ ∥ Q ∗ -Q A ∥ ∥ ≤ ε with probability at least 1 -δ for all M ∈ M . 6

## Theorem 3 (Lower bound on the sample complexity of RL)

Let Assumption 1 hold and T be a positive integer. There exist some constants ε 0 , δ 0 , c 1 , c 2 , and a class of MDPs M , such that for all ε ∈ (0 , ε 0 ) , δ ∈ (0 , δ 0 /N ) , and every ( ε, δ ) -correct RL Algorithm A on the class of MDPs M the total number of state-transition samples (sampling budget) needs to be at least

<!-- formula-not-decoded -->

## 4 Analysis

In this section, we first provide the full proof of the finite-time PAC bound of QVI and PI, reported in Theorem 1 and Theorem 2, in Subsection 4.1. We then prove Theorem 3, a new RL lower bound, in Subsection 4.2.

## 4.1 Proofs of Theorem 1 and Theorem 2 - The Upper Bounds

We begin by introducing some new notation. For the stationary policy π , we define Σ π ( z ) glyph[defines] E [ | ∑ t ≥ 0 γ t r ( Z t ) -Q π ( z ) | 2 | Z 0 = z ] as the variance of the sum of discounted rewards starting from z ∈ Z under the policy π . We also make use of the

5 For every real number u , glyph[ceilingleft] u glyph[ceilingright] is defined as the smallest integer number not less than u .

6 Algorithm A , unlike QVI and PI, does not require a same number of transition samples for every state-action pair and can generate samples arbitrarily.

following definition of the variance of a function: For any real-valued function f : Y → R , where Y is a finite set, we define V y ∼ ρ ( f ( y )) glyph[defines] E y ∼ ρ | f ( y ) -E y ∼ ρ ( f ( y )) | 2 as the variance of f under the probability distribution ρ , where Y is a finite set and ρ is a probability distribution on Y . We then define σ Q π ( z ) glyph[defines] γ 2 V y ∼ P ( ·| z ) [ Q π ( y, π ( y ))] as the discounted variance of Q π at z ∈ Z . Also, we shall denote σ V π and σ V ∗ as the discounted variance of the value function V π and V ∗ defined as σ V π ( z ) glyph[defines] γ 2 V y ∼ P ( ·| z ) [ V π ( y )] and σ V ∗ ( z ) glyph[defines] γ 2 V y ∼ P ( ·| z ) [ V ∗ ( y )], for all z ∈ Z , respectively. For each of these variances we define the corresponding empirical variance ̂ σ Q π ( z ) glyph[defines] γ 2 V y ∼ ̂ P ( ·| z ) [ Q π ( y, π ( y ))], ̂ σ V π ( z ) glyph[defines] γ 2 V y ∼ ̂ P ( ·| z ) [ V π ( y )] and ̂ σ V ∗ ( z ) glyph[defines] γ 2 V y ∼ ̂ P ( ·| z ) [ V ∗ ( y )], respectively, for all z ∈ Z under the model ̂ P . We also define ̂ σ Q ∗ ( z ) glyph[defines] γ 2 V y ∼ ̂ P ( ·| z ) [ Q ∗ ( y, π ∗ ( y ))]. We also notice that σ Q π and σ V π can be written as follows: σ Q π ( z ) = γ 2 P π [ | Q π -P π Q π | 2 ]( z ) and σ V π ( z ) = γ 2 P [ | V π -PV π | 2 ]( z ) for all z ∈ Z .

We now prove our first result which shows that Q k , for both QVI and PI, is very close to ̂ Q ∗ up to an order of O ( γ k ). Therefore, to prove bound on ‖ Q ∗ -Q k ‖ , one only needs to bound ‖ Q ∗ -̂ Q ∗ ‖ in high probability.

Lemma 1 Let Assumption 1 hold and Q 0 ( z ) be in the interval [0 , β ] for all z ∈ Z . Then, for both QVI and PI, we have

<!-- formula-not-decoded -->

## Proof

We begin by proving the result for QVI. For all k ≥ 0, we have

<!-- formula-not-decoded -->

Thus by an immediate recursion

<!-- formula-not-decoded -->

In the case of PI, we notice that Q k = ̂ Q π k -1 ≥ ̂ Q π k -2 = Q k -1 , which implies that

<!-- formula-not-decoded -->

where in the last line we rely on the fact hat π k -1 is the greedy policy w.r.t. Q k -1 . This implies the component-wise inequality ̂ P π k -1 Q k -1 ≥ ̂ P ̂ π ∗ Q k -1 . The result then follows by taking the glyph[lscript] ∞ -norm on both sides of the inequality and then recursively expand the resulted bound.

One can easily prove the following corollary, which bounds the difference between ̂ Q ∗ and ̂ Q π k , based on the result of Lemma 1 and the main result of Singh and Yee (1994). Corollary 1 is required for the proof of Theorem 2.

Corollary 1 Let Assumption 1 hold and π k be the greedy policy induced by the k th iterate of QVI and PI. Also, let Q 0 ( z ) takes value in the interval [0 , β ] for all z ∈ Z . Then we have

<!-- formula-not-decoded -->

We notice that the tight bound on ‖ ̂ Q π k -̂ Q ∗ ‖ for PI is of order γ k +1 β since ̂ Q π k = Q k +1 . However, for ease of exposition we make use of the bound of Corollary 1 for both QVI and PI.

In the rest of this subsection, we focus on proving a high probability bound on ‖ Q ∗ -̂ Q ∗ ‖ . One can prove a crude bound of ˜ O ( β 2 / √ n ) on ‖ Q ∗ -̂ Q ∗ ‖ by first proving that ‖ Q ∗ -̂ Q ∗ ‖ ≤ β ‖ ( P -̂ P ) V ∗ ‖ and then using the Hoeffding's tail inequality (Cesa-Bianchi and Lugosi, 2006, appendix, pg. 359) to bound the random variable ‖ ( P -̂ P ) V ∗ ‖ in high probability. Here, we follow a different and more subtle approach to bound ‖ Q ∗ -̂ Q ∗ ‖ , which leads to our desired result of ˜ O ( β 1 . 5 / √ n ): (i) We prove in Lemma 2 component-wise upper and lower bounds on the error Q ∗ -̂ Q ∗ which are expressed in terms of ( I -γ ̂ P π ∗ ) -1 [ P -̂ P ] V ∗ and ( I -γ ̂ P ̂ π ∗ ) -1 [ P -̂ P ] V ∗ , respectively. (ii) We make use of of Bernstein's inequality to bound [ P -̂ P ] V ∗ in terms of the squared root of the variance of V ∗ in high probability. (iii) We prove the key result of this subsection (Lemma 6) which shows that the variance of the sum of discounted rewards satisfies a Bellman-like recursion, in which the instant reward r ( z ) is replaced by σ Q π ( z ). Based on this result we prove an upper-bound of order O ( β 1 . 5 ) on ( I -γP π ) -1 √ σ Q π for every policy π , which combined with the previous steps leads to an upper bound of ˜ O ( β 1 . 5 / √ n ) on ‖ Q ∗ -̂ Q ∗ ‖ . A similar approach leads to a bound of ˜ O ( β 1 . 5 / √ n ) on ‖ Q ∗ -Q π k ‖ under the assumption that there exist constants c 1 &gt; 0 and c 2 &gt; 0 such that n &gt; c 1 γ 2 β 2 | X | log( c 2 N/δ )).

The following component-wise results bound Q ∗ -̂ Q ∗ from above and below:

## Lemma 2 (Component-wise bounds on Q ∗ -̂ Q ∗ )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof

We have that ̂ Q ∗ ≥ ̂ Q π ∗ . Thus:

<!-- formula-not-decoded -->

In the case of Ineq. (2) we have

<!-- formula-not-decoded -->

in which we make use of the following component-wise inequalities:

<!-- formula-not-decoded -->

where 0 is a function which assigns 0 to all ( z 1 , z 2 ) ∈ Z × Z .

We now concentrate on bounding the RHS (right hand sides) of (1) and (2) in high probability, for that we need the following technical lemmas (Lemma 3 and Lemma 4).

Lemma 3 Let Assumption 1 hold. Then, for any 0 &lt; δ &lt; 1 w.p at least 1 -δ

<!-- formula-not-decoded -->

where c v glyph[defines] γβ 2 √ 2 log(2 | X | /δ ) /n .

## Proof

We begin by proving bound on ‖ V ∗ -̂ V π ∗ ‖ :

<!-- formula-not-decoded -->

By solving this inequality w.r.t. ‖ V ∗ -̂ V π ∗ ‖ we deduce

<!-- formula-not-decoded -->

By using a similar argument the same bound can be proven on ‖ V ∗ -̂ V ∗ ‖ :

<!-- formula-not-decoded -->

We then make use of Hoeffding's inequality (Cesa-Bianchi and Lugosi, 2006, Appendix A, pg. 359) to bound | ( P π ∗ -̂ P π ∗ ) V ∗ ( x ) | for all x ∈ X in high probability:

<!-- formula-not-decoded -->

By applying the union bound we deduce

<!-- formula-not-decoded -->

We then define the probability of failure δ as

where b v is defined as

<!-- formula-not-decoded -->

and 1 is a function which assigns 1 to all z ∈ Z .

## Proof

Here, we only prove (8). One can prove (9) following similar lines.

<!-- formula-not-decoded -->

It is not difficult to show that V Y ∼ ̂ P ( ·| z ) ( V ∗ ( Y ) -̂ V π ∗ ( Y )) ≤ ‖ V ∗ -̂ V π ∗ ‖ 2 , which implies that

<!-- formula-not-decoded -->

The following inequality then holds w.p. at least 1 -δ :

<!-- formula-not-decoded -->

in which we make use of Hoeffding's inequality as well as Lemma 3 and a union bound to prove the bound on σ V ∗ in high probability. It is not then difficult to show that for every policy π and for all z ∈ Z : σ V π ( z ) ≤ σ Q π ( z ). This combined with a union bound on all state-action pairs in Eq.(10) completes the proof.

<!-- formula-not-decoded -->

By plugging (6) into (5) we deduce

<!-- formula-not-decoded -->

The results then follow by plugging (7) into (3) and (4).

We now state Lemma 4 which relates σ V ∗ to ̂ σ ̂ Q π ∗ and ̂ σ ̂ Q ∗ . Later, we make use of this result in the proof of Lemma 5.

Lemma 4 Let Assumption 1 hold and 0 &lt; δ &lt; 1 . Then, w.p. at least 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following result proves a bound on γ ( P -̂ P ) V ∗ , for which we make use of the Bernstein's inequality (Cesa-Bianchi and Lugosi, 2006, appendix, pg. 361) as well as Lemma 4.

Lemma 5 Let Assumption 1 hold and 0 &lt; δ &lt; 1 . Define c pv glyph[defines] 2 log(2 N/δ ) and b pv as

<!-- formula-not-decoded -->

Then w.p. at least 1 -δ we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof

For all z ∈ Z and all 0 &lt; δ &lt; 1, Bernstein's inequality implies that w.p. at least 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We deduce (using a union bound)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c ′ pv glyph[defines] 2 log( N/δ ) and b ′ pv glyph[defines] 2 γβ log( N/δ ) / 3 n . The result then follows by plugging (8) and (9) into (13) and (14), respectively, and then taking a union bound.

We now state the key lemma of this section which shows that for any policy π the variance Σ π satisfies the following Bellman-like recursion. Later, we use this result, in Lemma 7, to bound ( I -γP π ) -1 σ Q π .

Lemma 6 Σ π satisfies the Bellman equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof

For all z ∈ Z we have

<!-- formula-not-decoded -->

in which we rely on E ( ∑ t ≥ 1 γ t r ( Z t ) -γQ π ( Z 1 ) | Z 1 ) = 0.

Based on Lemma 6, one can prove the following result on the discounted variance.

## Lemma 7

## Proof

The first inequality follows from Lemma 6 by solving (15) in terms of Σ π and taking the sup-norm over both sides of the resulted equation. In the case of Eq. (17) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in which we write k = tl + j with t any positive integer. 7 We now prove a bound on ∥ ∥ ∑ t -1 j =0 ( γP π ) j √ σ Q π ∥ ∥ by making use of Jensen's inequality, Cauchy-Schwarz inequality and Eq. 16:

<!-- formula-not-decoded -->

The result then follows by plugging (19) into (18) and optimizing the bound in terms of t to achieve the best dependency on β .

Now, we make use of Lemma 7 and Lemma 5 to bound ‖ Q ∗ -̂ Q ∗ ‖ in high probability.

Lemma 8 Let Assumption 1 hold. Then, for any 0 &lt; δ &lt; 1 :

<!-- formula-not-decoded -->

w.p. at least 1 -δ , where ε ′ is defined as

<!-- formula-not-decoded -->

## Proof

By incorporating the result of Lemma 5 and Lemma 7 into Lemma 2 and taking in to account that ( I -γ ̂ P π ∗ ) -1 1 = β 1 , we deduce

<!-- formula-not-decoded -->

w.p. at least 1 -δ . The scalar b is given by

<!-- formula-not-decoded -->

The result then follows by combining these two bounds using a union bound and taking the glyph[lscript] ∞ norm.

7 For any real-valued function f , √ f is defined as a component wise squared-root operator on f . Also, for any policy π and k ≥ 1: ( P π ) k ( · ) glyph[defines] P π · · · P π ( · ).

︸

︸

## Proof of Theorem 1

We define the total error ε = ε ′ + γ k β which bounds ‖ Q ∗ -Q k ‖ ≤ ‖ Q ∗ -̂ Q ∗ ‖ + ‖ ̂ Q ∗ -Q k ‖ in high probability ( ε ′ is defined in Lemma 8). The results then follows by solving this bound w.r.t. n and k and then quantifying the total number of samples by T = nk .

We now draw our attention to the proof of Theorem 2, for which we need the following component-wise bound on Q ∗ -Q π k .

Lemma 9 Let Assumption 1 hold. Then w.p. at least 1 -δ

<!-- formula-not-decoded -->

where b is defined by (22) .

## Proof

We make use of Corollary 1 and Lemma 8 to prove the result:

<!-- formula-not-decoded -->

Lemma 9 states that w.h.p. Q ∗ -Q π k is close to ̂ Q π k -Q π k for large values of k and n . Therefore, to prove the result of Theorem 2 we only need to bound ̂ Q π k -Q π k in high probability:

Lemma 10 (Component-wise upper bound on ̂ Q π k -Q π k )

<!-- formula-not-decoded -->

## Proof

We prove this result using a similar argument as in the proof of Lemma 2:

<!-- formula-not-decoded -->

Now we bound the terms in the RHS of Eq. (23) in high probability. We begin by bounding γ ( I -γ ̂ P π k ) -1 ( P -̂ P ) V ∗ :

Lemma 11 Let Assumption 1 hold. Then, w.p. at least 1 -δ we have

<!-- formula-not-decoded -->

## Proof

From Lemma 5, w.p. at least 1 -δ , we have

<!-- formula-not-decoded -->

where in the last line we rely on Corollary 1. The result then follows by combining (24) with the result of Lemma 7.

We now prove bound on ‖ ( P -̂ P )( V ∗ -̂ V π k ) ‖ in high probability, for which we require the following technical result:

## Lemma 12 (Weissman et. al. 2003)

Let ρ be a probability distribution on the finite set X . Let { X 1 , X 2 , · · · , X n } be a set of i.i.d. samples distributed according to ρ and ̂ ρ be the empirical estimation of ρ using this set of samples. Define π ρ glyph[defines] max X ⊆ X min( P ρ ( X ) , 1 -P ρ ( X )) , where P ρ ( X ) is the probability of X under the distribution ρ and ϕ ( p ) glyph[defines] 1 / (1 -2 p ) log((1 -p ) /p ) for all p ∈ [0 , 1 / 2) with the convention ϕ (1 / 2) = 2 , then w.p. at least 1 -δ we have

<!-- formula-not-decoded -->

Lemma 13 Let Assumption 1 hold. Then, w.p. at least 1 -δ we have

<!-- formula-not-decoded -->

## Proof

From the H¨ older's inequality for all z ∈ Z we have

<!-- formula-not-decoded -->

This combined with Lemma 12 implies that

<!-- formula-not-decoded -->

The result then follows by taking union bound on all z ∈ Z .

We now make use of the results of Lemma 13 and Lemma 11 to bound ‖ Q ∗ -Q π k ‖ in high probability:

Lemma 14 Let Assumption 1 hold. Assume that

<!-- formula-not-decoded -->

Then, w.p. at least 1 -δ we have

<!-- formula-not-decoded -->

where ε ′ is defined by Eq. (20) .

## Proof

By incorporating the result of Lemma 13 and Lemma 11 into Lemma 10 we deduce

<!-- formula-not-decoded -->

w.p. 1 -δ . Eq. (26) combined with the result of Lemma 9 and a union bound implies that

<!-- formula-not-decoded -->

By taking the glyph[lscript] ∞ -norm and solving the resulted bound in terms of ‖ Q ∗ -Q π k ‖ we deduce

<!-- formula-not-decoded -->

The choice of n &gt; 8 β 2 γ 2 | X | log 4 N δ deduce the result.

## Proof of Theorem 2

The result follows by solving the bound of Lemma 14 w.r.t. n and k , in that we also need to assume that ε ≤ c √ β γ | X | for some c &gt; 0 in order to reconcile the bound of Theorem 2 with Eq. (25).

## 4.2 Proof of Theorem 3 - The Lower-Bound

In this section, we provide the proof of Theorem 3. In our analysis, we rely on the likelihood-ratio method, which has been previously used to prove a lower bound for multi-armed bandits (Mannor and Tsitsiklis, 2004), and extend this approach to RL and MDPs.

We begin by defining a class of MDPs for which the proposed lower bound will be obtained (see Figure 1). We define the class of MDPs M as the set of all MDPs with the state-action space of cardinality N = 3 KL , where K and L are positive integers. Also, we assume that for all M ∈ M , the state space X consists of three smaller subsets S , Y 1 and Y 2 . The set S includes K states, each of those states corresponds with the set of actions A = { a 1 , a 2 , . . . , a L } , whereas the states in Y 1 and Y 2 are single-action states. By taking the action a ∈ A from every

X1

X2

XK

al

12

aL

aL

a1

aL

yl

Fig. 1 The class of MDPs considered in the proof of Theorem 3. Nodes represent states and arrows show transitions between the states (see the text for details).

<!-- image -->

state x ∈ S , we move to the next state y ( z ) ∈ Y 1 with the probability 1, where z = ( x, a ). The transition probability from Y 1 is characterized by the transition probability p M from every y ( z ) ∈ Y 1 to itself and with the probability 1 -p M to the corresponding y ( z ) ∈ Y 2 . We notice that every state y ∈ Y 2 is only connected to one state in Y 1 and S , i.e., there is no overlapping path in the MDP. Further, for all M ∈ M , Y 2 consists of only absorbing states, i.e., for all y ∈ Y 2 , P ( y | y ) = 1. The instant reward r is set to 1 for every state in Y 1 and 0 elsewhere. For this class of MDPs, the optimal action-value function Q ∗ M can be solved in closed form from the Bellman equation. For all M ∈ M

<!-- formula-not-decoded -->

Now, let us consider two MDPs M 0 and M 1 in M with the transition probabilities

<!-- formula-not-decoded -->

where α and p are some positive numbers such that 0 &lt; p &lt; p + α ≤ 1, to be quantified later in this section. We denote the set { M 0 , M 1 } ⊂ M with M ∗ .

In the rest of this section, we concentrate on proving the lower bound on ‖ Q ∗ M -Q A T ‖ for all M ∈ M ∗ , where Q A T is the output of Algorithm A after observing T state-transition samples. It turns out that a lower-bound on the sample complexity of M ∗ also bounds the sample complexity of M from below. In the sequel, we make use of the notation E m ad P m for the expectation and the probability under the model M m : m ∈ { 0 , 1 } , respectively.

We follow the following steps in the proof: (i) we prove a lower bound on the sample-complexity of learning the action-value function for every state-action pair z ∈ S × A on the class of MDP M ∗ (ii) we then make use of the fact that the estimates of Q ∗ ( z ) for different z ∈ S × A are independent of each others to combine the bounds for all z ∈ S × A and prove the tight result of Theorem 3.

Webegin our analysis of the lower bound by proving a lower-bound on the probability of failure of any RL algorithm to estimate a near-optimal action-value function for every state-action pair z ∈ S × A . In order to prove this result (Lemma 16) we need to introduce some new notation: We define Q A t ( z ) as the output of Algorithm A using t &gt; 0 transition samples from the state y ( z ) ∈ Y 1 for all z ∈ S × A . We also define the event E 1 ( z ) glyph[defines] {| Q ∗ M 0 ( z ) -Q A t ( z ) | ≤ ε } for all z ∈ S × A . We then define k glyph[defines] r 1 + r 2 + · · · + r t as the sum of rewards of making t transitions from y ( z ) ∈ Y 1 . We also introduce the event E 2 ( z ), for all z ∈ S × A as

<!-- formula-not-decoded -->

where we have defined θ glyph[defines] exp ( -c ′ 1 α 2 t/ ( p (1 -p )) ) . Further, we define E ( z ) glyph[defines] E 1 ( z ) ∩ E 2 ( z ).

We also make use of the following technical lemma which bounds the probability of the event E 2 ( z ) from below:

Lemma 15 For all p &gt; 1 2 and every z ∈ S × A , we have

<!-- formula-not-decoded -->

## Proof

We make use of the Chernoff-Hoeffding bound for Bernoulli's (Hagerup and R¨ ub, 1990) to prove the result: For p &gt; 1 2 , define ε = √ 2 p (1 -p ) t log c ′ 2 2 θ , we then have

<!-- formula-not-decoded -->

where KL( p || q ) glyph[defines] p log( p/q ) + (1 -p ) log((1 -p ) / (1 -q )) denotes the KullbackLeibler divergence between p and q .

We now state the key result of this section:

Lemma 16 For every RL Algorithm A and every z ∈ S × A , there exists an MDP M m ∈ M ∗ and constants c ′ 1 &gt; 0 and c ′ 2 &gt; 0 such that

<!-- formula-not-decoded -->

by the choice of α = 2(1 -γp ) 2 ε/ ( γ 2 ) .

## Proof

To prove this result we make use of a contradiction argument, i.e., we assume that there exists an algorithm A for which:

<!-- formula-not-decoded -->

for all M m ∈ M ∗ and show that this assumption leads to a contradiction.

By the assumption that P m ( | Q ∗ M m ( z ) -Q A t ( z ) | ) &gt; ε ) ≤ θ/c ′ 2 for all M m ∈ M ∗ , we have P 0 ( E 1 ( z )) ≥ 1 -θ/c ′ 2 ≥ 1 -1 /c ′ 2 . This combined with Lemma 15 and by the choice of c ′ 2 = 6 implies that, for all z ∈ S × A , P 0 ( E ( z )) &gt; 1 / 2. Based on this result we now prove a bound from below on P 1 ( E 1 ( z )).

We define W as the history of all the outcomes of trying z for t times and the likelihood function L m ( w ) for all M m ∈ M ∗ as

<!-- formula-not-decoded -->

for every possible history w and M m ∈ M ∗ . This function can be used to define a random variable L m ( W ), where W is the sample path of the random process (the sequence of observed transitions). The likelihood ratio of the event W between two MDPs M 1 and M 0 can then be written as

<!-- formula-not-decoded -->

Now, by making use of log(1 -u ) ≥ -u -u 2 for 0 ≤ u ≤ 1 / 2, and exp ( -u ) ≥ 1 -u for 0 ≤ u ≤ 1, we have

<!-- formula-not-decoded -->

for α ≤ (1 -p ) / 2. Thus

<!-- formula-not-decoded -->

since k ≤ t .

Using log(1 -u ) ≥ -2 u for 0 ≤ u ≤ 1 / 2, we have for α 2 ≤ p (1 -p ),

<!-- formula-not-decoded -->

and for α 2 ≤ p 2 / 2, we have

<!-- formula-not-decoded -->

on E 2 . Further, we have t -k/p ≤ √ 2 1 -p p t log( c 2 / (2 θ )), thus for α ≤ (1 -p ) / 2:

<!-- formula-not-decoded -->

We then deduce that

<!-- formula-not-decoded -->

for the choice of c ′ 1 = 8. Thus

✶ ✶ where ✶ E is the indicator function of the event E ( z ). Then by a change of measure we deduce

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the choice of α = 2(1 -γp ) 2 ε/ ( γ 2 ), we have α ≤ (1 -p ) / 2 ≤ p (1 -p ) ≤ p/ √ 2 whenever ε ≤ 1 -p 4 γ 2 (1 -γp ) 2 . For this choice of α , we have that Q ∗ M 1 ( z ) -Q ∗ M 0 ( z ) = γ 1 -γ ( p + α ) -γ 1 -γp &gt; 2 ε , thus Q ∗ M 0 ( z )+ ε &lt; Q ∗ M 1 ( z ) -ε . In words, the random event {| Q ∗ M 0 ( z ) -Q ( z ) | ≤ ε } does not overlap with the event {| Q ∗ M 1 ( z ) -Q ( z ) | ≤ ε } .

✶ where we make use of the fact that P 0 ( Q ( z )) &gt; 1 2 .

Now let us return to the assumption of Eq. (28), which states that for all M m ∈ M ∗ , P m ( | Q ∗ M m ( z ) -Q A t ( z ) | ) ≤ ε ) ≥ 1 -θ/c ′ 2 under Algorithm A . Based

on Eq. (29), we have P 1 ( | Q ∗ M 0 ( z ) -Q A t ( z ) | ≤ ε ) &gt; θ/c ′ 2 . This combined with the fact that {| Q ∗ M 0 ( z ) -Q A t ( z ) |} and {| Q ∗ M 1 ( z ) -Q A t ( z ) |} do not overlap implies that P 1 ( | Q ∗ M 1 ( z ) -Q A t ( z ) | ) ≤ ε ) ≤ 1 -θ/c ′ 2 , which violates the assumption of Eq. (28). Therefore, the lower bound of Eq. (27) shall hold.

Based on the result of Lemma 16 and by the choice of p = 4 γ -1 3 γ and c 1 = 8100, we have that for every ε ∈ (0 , 3] and for all 0 . 4 = γ 0 ≤ γ &lt; 1 there exists an MDP M m ∈ M ∗ such that

<!-- formula-not-decoded -->

This result implies that for any state-action pair z ∈ S × A :

<!-- formula-not-decoded -->

on M 0 or M 1 whenever the number of transition samples t is less than ξ ( ε, δ ) glyph[defines] 6 β 3 c 1 ε 2 log 1 c ′ 2 δ .

Based on this result, we prove a lower bound on the number of samples T foe which ‖ Q ∗ M m -Q A T ‖ &gt; ε on either M 0 or M 1 :

Lemma 17 For any δ ′ ∈ (0 , 1 / 2) and any Algorithm A using a total number of transition samples less than T = N 6 ξ ( ε, 12 δ ′ N ) , there exists an MDP M m ∈ M ∗ such that

<!-- formula-not-decoded -->

## Proof

First, we note that if the total number of observed transitions is less than ( KL/ 2) ξ ( ε, δ ) = ( N/ 6) ξ ( ε, δ ), then there exists at least KL/ 2 = N/ 6 state-action pairs that are sampled at most ξ ( ε, δ ) times. Indeed, if this was not the case, then the total number of transitions would be strictly larger than N/ 6 ξ ( ε, δ ), which implies a contradiction). Now let us denote those states as z (1) , . . . , z ( N/ 6) .

In order to prove that (31) holds for every RL algorithm, it is sufficient to prove it for the class of algorithms that return an estimate Q A T z ( z ), where T z is the number of samples collected from z , for each state-action z based on the transition samples observed from z only. 8 This is due to the fact that the samples from z and z ′ are independent. Therefore, the samples collected from z ′ do not bring more information about Q ∗ M ( z ) than the information brought by the samples collected from z . Thus, by defining Q ( z ) glyph[defines] {| Q ∗ M ( z ) -Q A T z ( z ) | &gt; ε } for all M ∈ M ∗ we have that for such algorithms, the events Q ( z ) and Q ( z ′ ) are conditionally independent given T z and T z ′ . Thus, there exists an MDP M m ∈ M ∗ such that

8 We let T z to be random.

<!-- formula-not-decoded -->

from Eq. (30), thus

<!-- formula-not-decoded -->

We finally deduce that if the total number of transition samples is less than N 6 ξ ( ε, δ ), then

<!-- formula-not-decoded -->

whenever δN ≤ 1. Setting δ ′ = δN

6 12 , we obtain the desired result.

Lemma 17 implies that if the total number of samples T is less than β 3 N/ ( c 1 ε 2 ) log( N/ ( c 2 δ )), with the choice of c 1 = 8100 and c 2 = 72, then the probability of ‖ Q ∗ M -Q A T ‖ ≤ ε is at maximum 1 -δ on either M 0 or M 1 . This is equivalent to the argument that for every RL algorithm A to be ( ε, δ )-correct on the set M ∗ , and subsequently on the class of MDPs M , the total number of transitions T needs to satisfy the inequality T ≥ β 3 N/ ( c 1 ε 2 ) log( N/ ( c 2 δ )), which concludes the proof of Theorem 3.

## 5 Conclusion and Future Works

In this paper, we have presented the first minimax bound on the sample complexity of estimating the optimal action-value function in discounted reward MDPs. We have proven that both model-based Q-value iteration (QVI) and modelbased policy iteration (PI), in the presence of the generative model of the MDP,

are optimal in the sense that the dependency of their performances on 1 /ε , N , δ and 1 / (1 -γ ) matches the lower bound of RL. Also, our results have significantly improved on the state-of-the-art in terms of dependency on 1 / (1 -γ ).

Overall, we conclude that both QVI and PI are efficient RL algorithms in terms of the number of samples required to attain a near optimal solution as the upper bounds on the performance loss of both algorithms completely match the lower bound of RL up to a multiplicative factor.

In this work, we only consider the problem of estimating the optimal actionvalue function when a generative model of the MDP is available. This allows us to make an accurate estimate of the state-transition distribution for all stateaction pairs and then estimate the optimal control policy based on this empirical model. This is in contrast to the online RL setup in which the choice of the exploration policy has an influence on the behavior of the learning algorithm and vise-versa. Therefore, we do not compare our results with those of online RL algorithms such as PAC-MDP (Szita and Szepesv´ ari, 2010; Strehl et al, 2009), upper-confidence-bound reinforcement learning (UCRL) (Jaksch et al, 2010) and REGAL of Bartlett and Tewari (2009). However, we believe that it would be possible to improve on the state-of-the-art in PAC-MDP, based on the results of this paper. This is mainly due to the fact that most PAC-MDP algorithms rely on an extended variant of model-based Q-value iteration to estimate the action-value function. However, those results bound the estimation error in terms of V max rather than the total variance of discounted reward which leads to a non-tight sample complexity bound. One can improve on those results, in terms of dependency on 1 / (1 -γ ), using the improved analysis of this paper which makes use of the sharp result of Bernstein's inequality to bound the estimation error in terms of the variance of sum of discounted rewards. It must be pointed out that, almost contemporaneously to our work, Lattimore and Hutter (2012) have independently proven a similar upper-bound of order ˜ O ( N/ ( ε 2 (1 -γ ) 3 )) for UCRL algorithm under the assumption that only two states are accessible form any state-action pair. Their work also includes a similar lower bound of ˜ Ω ( N/ ( ε 2 (1 -γ ) 3 )) for any RL algorithm which matches, up to a logarithmic factor, the result of Theorem 3.

## References

- Azar MG, Munos R, Ghavamzadeh M, Kappen HJ (2011a) Reinforcement Learning with a Near Optimal Rate of Convergence. Tech. rep., URL http://hal.inria.fr/inria-00636615
- Azar MG, Munos R, Ghavamzadeh M, Kappen HJ (2011b) Speedy q-learning. In: Advances in Neural Information Processing Systems 24, pp 2411-2419
- Bartlett PL, Tewari A (2009) REGAL: A regularization based algorithm for reinforcement learning in weakly communicating MDPs. In: Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence
- Bertsekas DP (2007) Dynamic Programming and Optimal Control, vol II, 3rd edn. Athena Scientific, Belmount, Massachusetts
- Bertsekas DP, Tsitsiklis JN (1996) Neuro-Dynamic Programming. Athena Scientific, Belmont, Massachusetts
- Cesa-Bianchi N, Lugosi G (2006) Prediction, Learning, and Games. Cambridge University Press, New York, NY, USA

- Even-Dar E, Mannor S, Mansour Y (2006) Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of Machine Learning Research 7:1079-1105
- Hagerup L, R¨ ub C (1990) A guided tour of chernoff bounds. Information Processing Letters 33:305-308
- Jaksch T, Ortner R, Auer P (2010) Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research 11:1563-1600
- Kakade SM (2004) On the sample complexity of reinforcement learning. PhD thesis, Gatsby Computational Neuroscience Unit
- Kearns M, Singh S (1999) Finite-sample convergence rates for Q-learning and indirect algorithms. In: Advances in Neural Information Processing Systems 12, MIT Press, pp 996-1002
- Lattimore T, Hutter M (2012) Pac bounds for discounted mdps. CoRR abs/1202.3890
- Mannor S, Tsitsiklis JN (2004) The sample complexity of exploration in the multiarmed bandit problem. Journal of Machine Learning Research 5:623-648
- Munos R, Moore A (1999) Influence and variance of a Markov chain : Application to adaptive discretizations in optimal control. In: Proceedings of the 38th IEEE Conference on Decision and Control
- Singh SP, Yee RC (1994) An upper bound on the loss from approximate optimalvalue functions. Machine Learning 16(3):227-233
- Sobel MJ (1982) The variance of discounted markov decision processes. Journal of Applied Probability 19:794-802
- Strehl AL, Li L, Littman ML (2009) Reinforcement learning in finite MDPs: PAC analysis. Journal of Machine Learning Research 10:2413-2444
- Sutton RS, Barto AG (1998) Reinforcement Learning: An Introduction. MIT Press, Cambridge, Massachusetts
- Szepesv´ ari C (2010) Algorithms for Reinforcement Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan &amp; Claypool Publishers
- Szita I, Szepesv´ ari C (2010) Model-based reinforcement learning with nearly tight exploration complexity bounds. In: Proceedings of the 27th International Conference on Machine Learning, Omnipress, pp 1031-1038
- Wiering M, van Otterlo M (2012) Reinforcement Learning: State-of-the-Art, Springer, chap 1, pp 3-39