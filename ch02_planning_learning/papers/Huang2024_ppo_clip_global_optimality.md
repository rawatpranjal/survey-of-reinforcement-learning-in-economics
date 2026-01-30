## PPO-Clip Attains Global Optimality: Towards Deeper Understandings of Clipping

Nai-Chieh Huang, Ping-Chun Hsieh, Kuo-Hao Ho, I-Chen Wu

Department of Computer Science, National Yang Ming Chiao Tung University, Hsinchu, Taiwan

{

naich.cs09, pinghsieh } @nycu.edu.tw

## Abstract

Proximal Policy Optimization algorithm employing a clipped surrogate objective (PPO-Clip) is a prominent exemplar of the policy optimization methods. However, despite its remarkable empirical success, PPO-Clip lacks theoretical substantiation to date. In this paper, we contribute to the field by establishing the first global convergence results of a PPO-Clip variant in both tabular and neural function approximation settings. Our findings highlight the O (1 / √ T ) min-iterate convergence rate specifically in the context of neural function approximation. We tackle the inherent challenges in analyzing PPO-Clip through three central concepts: (i) We introduce a generalized version of the PPO-Clip objective, illuminated by its connection with the hinge loss. (ii) Employing entropic mirror descent, we establish asymptotic convergence for tabular PPO-Clip with direct policy parameterization. (iii) Inspired by the tabular analysis, we streamline convergence analysis by introducing a two-step policy improvement approach. This decouples policy search from complex neural policy parameterization using a regression-based update scheme. Furthermore, we gain deeper insights into the efficacy of PPO-Clip by interpreting these generalized objectives. Our theoretical findings also mark the first characterization of the influence of the clipping mechanism on PPO-Clip convergence. Importantly, the clipping range affects only the pre-constant of the convergence rate.

## 1 Introduction

Policy optimization is a prevalent method for solving reinforcement learning problems, involving iterative parameter updates to maximize objectives. Policy gradient methods, a prominent subset of this approach, were introduced as a direct solution using gradient descent. Their primary aim is to identify an optimal policy that maximizes the total expected reward through interactions with the environment. The selection of an appropriate step size is crucial as it significantly influences policy gradient algorithm performance. Addressing this challenge, Trust Region Policy Optimization (TRPO) was created (Schulman et al. 2015). Utilizing a trust-region approach with a second-order approximation, TRPO guarantees substantial policy improvement. Unlike computationally intensive TRPO, Proximal Policy Optimization (PPO) (Schulman et al. 2017) leverages first-order

Copyright © 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

derivatives for policy improvement. PPO encompasses two main variants: PPO-KL and PPO-Clip, each with distinct characteristics. PPO-KL adds a Kullback-Leibler divergence penalty to the objective, while PPO-Clip integrates probability ratio clipping. These variants showcase remarkable performance across various environments, with PPO standing out for its computational efficiency (Chen, Peng, and Zhang 2018; Ye et al. 2020; Byun, Kim, and Wang 2020).

Given the empirical success of these policy optimization algorithms, recent works have made significant strides in enhancing their theoretical guarantees. In particular, (Agarwal et al. 2020; Bhandari and Russo 2019) prove the global convergence result of the policy gradient algorithm under different settings. Additionally, (Mei et al. 2020) establishes the convergence rates of the softmax policy gradient in both the standard and the entropy-regularized settings. Furthermore, it has been shown that various policy gradient algorithms also enjoy global convergence (Fazel et al. 2018; Liu et al. 2020; Wang et al. 2021). In the context of TRPO and PPO, (Shani, Efroni, and Mannor 2020) have utilized the mirror descent method to establish the convergence rate of adaptive TRPO under both the standard and entropy-regularized settings. Furthermore, (Liu et al. 2019) have provided the convergence rate of PPO-KL and TRPO under neural function approximation. 1 By contrast, despite that PPO-Clip is computationally efficient and empirically successful, the following question about the theory of PPO-Clip remains largely open: Does PPO-Clip enjoy provable global convergence or have any convergence rate guarantee?

In this paper, we answer the above question affirmatively. To begin with, we generalize the PPO-Clip objective to encompass a wider range of variants, enhancing our comprehension of its efficacy. Accordingly, we present the first-ever global convergence guarantee for a PPO-Clip variant under both tabular and neural function approximation. Notably, through convergence analysis, we offer two pivotal insights into the clipping mechanism: (i) Under PPO-Clip, the policy updates scale with advantage magnitudes, while the sign dictates whether to increase or decrease the action probabilities. Notably, given the representation power of neural networks, incorrect signs typically emerge when the advan-

1 For the detailed discussion about related work, please refer to Appendix H.

tage magnitudes are nearly zero. In such cases, these values insignificantly contribute to the objective, preserving the objective accuracy despite the incorrect signs. This perspective illuminates the robustness and empirical success of PPOClip. (ii) Through our convergence analysis, we demonstrate that the clipping range merely affects the pre-constant of the convergence rate, not the asymptotic behavior. All the code is available at https://github.com/NYCU-RL-BanditsLab/Neural-PPO-Clip

Our Contributions. We summarize the main contributions of this paper as follows:

- To establish the global convergence of PPO-Clip, we leverage the connection between PPO-Clip and the hinge loss, leading to the formulation of generalized PPO-Clip objectives. Additionally, we harness the power of the entropic mirror descent (EMDA) (Beck and Teboulle 2003) for tabular PPO-Clip under direct policy parameterization, thereby demonstrating its asymptotic convergence.
- Inspired by the tabular analysis, we present a two-step policy improvement framework based on EMDA for Neural PPO-Clip. This framework enhances the manageability of the analysis by effectively separating policy search from policy parameterization. Accordingly, we establish the first global convergence result and explicitly characterize the O (1 / √ T ) min-iterate convergence rate for the generalized PPO-Clip and hence provide an affirmative answer to one critical open question about PPO-Clip.
- We gain deeper insights into the PPO-Clip performance. Our theoretical findings yield two key insights into the clipping mechanism, as mentioned earlier. Furthermore, our analysis extends seamlessly to various Neural PPOClip variants with different classifiers, guided by the provided sufficient conditions.

## 2 Preliminaries

Markov Decision Processes. Consider a discounted Markov Decision Process ( S , A , P , R, γ, µ ) , where S is the state space (possibly infinite ), A is a finite action space, P : S × A × S → [0 , 1] is the transition dynamic of the environment, R : S × A → [0 , R max ] is the bounded reward function, γ ∈ (0 , 1) is the discount factor, and µ is the initial state distribution. Given a policy π : S → ∆( A ) , where ∆( A ) is the unit simplex over A , we define the state-action value function Q π ( · , · ) := E a t ∼ π ( ·| s t ) ,s t +1 ∼P ( ·| s t ,a t ) [ ∑ ∞ t =0 γ t R ( s t , a t ) | s 0 = s, a 0 = a ] . Moreover, we define V π ( s ) := E a ∼ π ( ·| s ) [ Q π ( s, a )] and A π ( s, a ) := Q π ( s, a ) -V π ( s ) . Also, we denote π ∗ as an optimal policy that attains the maximum total expected reward and denote π 0 as the uniform policy. We introduce ν π ( s ) = (1 -γ ) ∑ ∞ t =0 γ t P ( s t = s | s 0 ∼ µ, π ) as the discounted state visitation distribution induced by π and σ π ( s, a ) = ν π ( s ) · π ( a | s ) as the state-action visitation distribution induced by π . In addition, we define the distribution ν ∗ and σ ∗ as the discounted state visitation distribution and the state-action visitation distribution induced by the optimal policy π ∗ , respectively. Moreover, we define ˜ σ π = ν π π 0 as the state-action distribution induced by interactions with the environment through π , sampling actions from the uniform policy π 0 . We use E ν π [ · ] and E σ π [ · ] as the shorthand notations of E s ∼ ν π [ · ] and E ( s,a ) ∼ σ π [ · ] , respectively.

For the convergence property, we define the total expected reward over the state distribution ν ∗ as

<!-- formula-not-decoded -->

Here, a maximizer of (1) is equivalent to the original definition of the optimal policy π ∗ . We will prove the global convergence by analyzing the difference in L between our policy and the optimal policy and show that the total expected reward monotonically increases.

Proximal Policy Optimization (PPO). PPO is an empirically successful algorithm that achieves policy improvement by maximizing a surrogate lower bound of the original objective, either through the Kullback-Leibler penalty (termed PPO-KL) or the clipped probability ratio (termed PPO-Clip). PPO-KL and PPO-Clip represent the two main branches of PPO, both aiming to enforce policy constraints during updates for policy improvement. It is crucial to emphasize that PPO-Clip represents a conceptual approach, utilizing the clipping mechanism to achieve policy constraints, rather than being a precise algorithm itself. In this paper, our focus is PPO-Clip. Let ρ s,a ( θ ) denote the probability ratio π θ ( a | s ) π θ t ( a | s ) . PPO-Clip avoids large policy updates by applying a simple heuristic that clips the probability ratio by the clipping range ϵ and thereby removes the incentive for moving ρ s,a ( θ ) away from 1. Specifically, the PPO-Clip objective is

<!-- formula-not-decoded -->

Neural Networks. We introduce the notations and assumptions relevant to neural networks. It is important to highlight that our analysis of neural networks draws inspiration from (Liu et al. 2019), and we adopt their notations to ensure compatibility. Specifically, this paper centers around the analysis of two-layer neural networks. For simplicity, let us consider ( s, a ) ∈ R d for all ( s, a ) ∈ S × A . We represent the two-layer neural network as NN ( α ; m ) , where α denotes the network input weights and m represents the network width. These neural networks act as the parameterization for both our policy π θ and the Q function. The parameterized function associated with NN ( α ; m ) is depicted as follows:

<!-- formula-not-decoded -->

where α = ([ α ] ⊤ 1 , . . . , [ α ] ⊤ m ) ⊤ ∈ R md is the input weights, with [ α ] i ∈ R d , b i ∈ {-1 , 1 } are the weights of the output, and σ ( · ) refers to the Rectified Linear Unit (ReLU) activation function. The initializations for the input weights α 0 and b i are provided as follows:

<!-- formula-not-decoded -->

where both b i and [ α 0 ] i are i.i.d. for each i ∈ [ m ] and I d is the d × d identity matrix. The values of b i remain fixed following initialization, with the training exclusively focused on adjusting the weights α . To uphold the local linearization characteristics, we employ a projection mechanism that confines the training weights α within an ℓ 2 -ball centered at α 0 , which is represented as B f = { α : ∥ α -α 0 ∥ 2 ≤ R f } , where f is the canonical name of the networks (It will be f for the policy network and Q for the Q function network in the following section).

Our examination of neural networks is grounded in the subsequent assumptions, which are widely adopted regularity conditions for neural networks in the reinforcement learning literature (Liu et al. 2019; Antos, Szepesv´ ari, and Munos 2007; Farahmand et al. 2016):

Assumption 1 (Q Function Class) . We assume that the our neural network class possesses sufficient representational capacity to model the Q function of any given policy π . Specifically, for any R &gt; 0 , define a function class

<!-- formula-not-decoded -->

for all α satisfying ∥ α -α 0 ∥ 2 ≤ R , where b i and α 0 are initialized as (4). We assume that Q π ( s, a ) ∈ F R Q ,m Q for any policy π , where R Q and m Q are the projection radius and width of the neural network for Q function.

Given that T π Q π remains a Q function, Assumption 1 affords us the property of completeness within our function class under the Bellman operator T π .

Notations: Weuse ⟨ a, b ⟩ and a ◦ b to denote the inner product and the Hadamard product, respectively.

## 3 Generalized PPO-Clip Objectives

Connecting PPO-Clip and Hinge Loss. According to (Hu et al. 2020; Pi et al. 2020), the original PPO-Clip objective could be connected with the hinge loss. Specifically, the gradient of the clipped objective is indeed the negative of the gradient of hinge loss objective, i.e.,

<!-- formula-not-decoded -->

where ℓ ( y i , f θ ( x i ) , ϵ ) is the hinge loss defined as max { 0 , ϵ -y i · f θ ( x i ) } , ϵ is the margin, y i ∈ {-1 , 1 } the label corresponding to the data x i , and f θ ( x i ) serves as the binary classifier. For completeness, please see Appendix I for a detailed comparison of the two objectives. From the above, maximizing the objective in (2) can be rewritten as minimizing the following loss:

<!-- formula-not-decoded -->

In practice, we draw a batch of state-action pairs and use the sample average to approximately minimize the loss in (7).

Generalized PPO-Clip Objectives. Based on the above reinterpretation of PPO-Clip, we provide a general form of the PPO-Clip loss function from a hinge loss perspective as follows,

<!-- formula-not-decoded -->

(8)

Different combinations of classifiers, margins, and weights lead to different loss functions, thereby representing diverse algorithms. PPO-Clip is a special case of (8) with a specific classifier ρ s,a ( θ ) -1 . Another variant, termed PPOClip-sub in this paper, can be obtained by employing a subtraction classifier, i.e., π θ ( a | s ) -π θ t ( a | s ) . There are several other variants under this generalized objective by employing distinct classifiers, e.g., log( π θ ( a | s )) -log( π θ t ( a | s )) and √ ρ s,a ( θ ) -1 . We demonstrate the empirical evaluation of these variants in Section 6. Given the above examples, the proposed objective provides to generalizing PPOClip via various classifiers, thereby expanding the objective choices within the context of PPO-Clip. This generalization also connects the PPO-Clip with the classifier selection paradigm. Additionally, this generalized objective provide an intution to understand more about the clipping mechanism. Please refer to Section 5.4.

## 4 Tabular PPO-Clip

## 4.1 Direct Policy Parameterization

In this section, we study the global convergence of PPO-Clip with direct parameterization, i.e., policies are parameterized by π ( a | s ) = θ s,a , where θ s ∈ ∆( A ) denotes the vector θ s, · and θ ∈ ∆( A ) |S| . We use V ( t ) ( s ) and A ( t ) ( s, a ) as the shorthands for V π ( t ) ( s ) and A π ( t ) ( s, a ) , respectively.

For the sake of clarity, we focus our discussion on the original PPO-Clip rather than delving into the broader scope of the generalized objective (8). Furthermore, we also provide additional analysis for other PPO-Clip variants with different classifiers in Appendix F. Note that by choosing the weight as | A ( t ) ( s, a ) | , the classifier as ρ ( t ) s,a ( θ ) -1 , and the margin as ϵ in (8) at the t -th iteration, the generalized objective would recover the form of the objective of PPO-Clip, which denoted as ˆ L ( t ) ( θ ) . The detailed algorithm is shown in Appendix A as Algorithm 7.

In each iteration, PPO-Clip updates the policy by minimizing the loss ˆ L ( t ) ( θ ) via the EMDA (Beck and Teboulle 2003). While there are alternative ways to minimize the loss ˆ L ( t ) ( θ ) over ∆( A ) |S| (e.g., the projected subgradient method), we leverage EMDA for the following two reasons: (i) PPO-Clip achieves policy improvement by increasing or decreasing the probability of those state-action pairs in D ( t ) based on the sign of A ( t ) ( s, a ) as well as properly reallocating the probabilities of those state-action pairs not contained in the batch (to ensure the probability sum is one). Using EMDA enforces a proper reallocation in PPO-Clip, as shown later in the proof of Theorem 1 in Appendix E; (ii) The exponentiated gradient scheme of EMDA guarantees π ( t ) remains strictly positive for all state-action pairs in each iteration t , ensuring the well-defined nature of the probability ratio ρ s,a ( θ ) used in PPO-Clip. In this section, we consider the stylized setting with tabular policy and true advantage mainly for motivating the PPO-Clip method and its analysis.

## 4.2 Global Convergence of PPO-Clip with Direct Parameterization

We first make the following assumptions. Note that we only consider these assumptions in the tabular case.

Assumption 2 (Infinite Visitation to Each State-Action Pair) . Each state-action pair ( s, a ) appears infinitely often in {D ( τ ) } , i.e., lim t →∞ ∑ t τ =0 1 { ( s, a ) ∈ D ( τ ) } = ∞ , with probability one.

Assumption 3. In each iteration t , the state-action pairs in D ( t ) have distinct states.

Assumption 2 resembles the standard infinite-exploration condition commonly used in the temporal-difference methods, such as Sarsa (Singh et al. 2000). Assumption 3 is rather mild: (i) This can be met by post-processing the mini-batch of state-action pairs via an additional sub-sampling step; (ii) In most RL problems with discrete actions, the state space is typically much larger than the action space.

Theorem 1 (Global Convergence of PPO-Clip) . Under PPO-Clip, we have V ( t ) ( s ) → V π ∗ ( s ) as t →∞ , ∀ s ∈ S , with probability one.

The proof of Theorem 1 is provided in Appendix E. We highlight the main ideas behind the proof of Theorem 1: (i) State-wise policy improvement: Through the lens of generalized objective, we show that PPO-Clip enjoys state-wise policy improvement in every iteration with the help of the EMDA subroutine. This property greatly facilitates the rest of the convergence analysis. (ii) Quantifying the probabilities of those actions with positive or negative advantages in the limit : By (i), we know the limits of the value functions and the advantage function all exist. Then, we proceed to show that the actions with positive advantages in the limit cannot exist by establishing a contradiction. The above also manifests how reinterpreting PPO-Clip helps with establishing the convergence guarantee.

## 5 Neural PPO-Clip

In this section, we begin by illustrating the process of decoupling policy search and policy parameterization, drawing inspiration from the tabular case. Subsequently, we provide a comprehensive overview of the neural PPO-Clip algorithm. Weproceed to delineate the intricacies posed by our analysis and present our results on the min-iterate convergence rate, both for the generalized PPO-Clip. In particular, the convergence rate of PPO-Clip can be view as a special case of our general results. Lastly, we offer a profound insight into the understanding of the clipping mechanism.

## 5.1 EMDA-Based Policy Search

Drawing inspiration from the tabular case, we proceed to present our two-step policy improvement scheme based on EMDA, and we call it EMDA-based Policy Search. Specifically, this scheme consists of two subroutines:

- Direct policy search : In this step, we directly search for an improved policy in the policy space by EMDA. More specifically, in each iteration t , we do a policy search by applying EMDA with direct parameterization to minimize the generalized PPO-Clip objective in (8) for finitely many iterations K and thereby obtain an improved policy ̂ π t +1 as the target policy. The pseudo code of EMDA is provided in Algorithm 2. Notably, under EMDA, we can obtain an explicit expression of the target policy ̂ π t +1 .
- Neural approximation for the target policy : Given the target policy ̂ π t +1 obtained by EMDA, we then approximate it in the parameter space by utilizing the representation power of neural networks via a regression-based policy update scheme (e.g., by using the mean-squared error loss). The detailed neural parameterization will be described in the next subsection.

While the decision to employ EMDA is inspired by the tabular case, there are two primary motivations and benefits for integrating EMDA with direct parameterization:

- Decoupling improvement and approximation: One major goal of this paper is to provide rigorous theoretical guarantees for PPO-Clip under neural function approximation. To make the analysis tractable and general, we would like to decouple policy improvement and function approximation of the policy. To achieve this, we adopt the EMDA-based two-step approach outlined previously.
- EMDA-induced closed-form expression of the target policy: For policy optimization analysis, the goal is often to derive a closed-form optimal solution for the policy improvement objective as the ideal target policy. However, such a closed-form optimal solution of an arbitrary objective function does not always exist. A case in point is the loss function of PPO-Clip. From this view, EMDA, which enjoys closed-form updates, substantially facilitates the convergence analysis, as can be observed in Proposition 1 presented in the subsequent subsection 5.2.

## 5.2 Neural PPO-Clip

Parameterization Setting. At each iteration t , we parameterize our policy as an energy-based policy π θ t ( a | s ) ∝ exp { τ -1 t f θ t ( s, a ) } , where τ t denotes the temperature parameter and f θ t ( s, a ) = NN ( θ t ; m f ) corresponds to the energy functions. The width of the neural network f θ is denoted as m f , as defined in Section 2. Likewise, we parameterize our state-action value function as Q ω ( s, a ) = NN ( ω ; m Q ) , with width m Q of the neural network Q ω . Concurrently, we define V ω ( s ) as the value function derived from the Bellman Expectation Equation. Also, we define A ω ( s, a ) := Q ω ( s, a ) -V ω ( s ) to be the advantage function. Policy Improvement. According to the EMDA-based Policy Search framework presented above, we first give the closed-form of the obtained target policy of Neural PPOClip as follows. The detailed proof is in Appendix B.

Proposition 1 (EMDA Target Policy) . For the target policy obtained by the EMDA subroutine at the t -th iteration, we have

<!-- formula-not-decoded -->

where C t ( s, a ) A ω t ( s, a ) = -∑ K -1 k =0 ηg ( k ) s,a as given in Algorithm 2.

Recall that the target policy ̂ π is the direct parameterization in the policy space, but our policy π θ is an energybased (softmax) policy that is proportional to the exponentiated energy function. This explains why we consider the log ̂ π t +1 ( a | s ) in Proposition 1. Another benefit of using EMDA is that it closely matches the energy-based policies considered in Neural PPO-Clip due to the inherent exponentiated gradient update.

Then, we discuss the details of the neural function approximation of our policy. After obtaining the target policy by Proposition 1, we solve the Mean Squared Error (MSE) subproblem with respect to θ to approximate the target policy as follows:

<!-- formula-not-decoded -->

Notice that we consider the state-action distribution ˜ σ t sampling the action through a uniform policy π 0 . In this manner, we use more exploratory data to improve our current policy. In particular, we use the SGD to tackle the above subproblem, and the pseudo code is provided in Appendix A.

Policy Evaluation. To evaluate Q , we use a neural network to approximate the true state-action value function Q π θ t by solving the Mean Square Bellman Error (MSBE) subproblem. The MSBE subproblem is to minimize the following objective with respect to ω at each iteration t :

<!-- formula-not-decoded -->

where T π θ t is the Bellman operator of policy π θ t such that

<!-- formula-not-decoded -->

The pseudo code of neural TD update for state-action value function Q ω is in Appendix A. It is worth mentioning that this variant of Neural PPO-Clip is not a fully on-policy algorithm. Although we interact with the environment by our current policy, we sample the actions by the uniform policy π 0 for policy improvement. We provide the pseudo code of Neural PPO-Clip as the following Algorithm 1 (please refer

## Algorithm 1: Neural PPO-Clip

Input : L Hinge ( θ ) , T , ϵ , EMDA step size η , number of EMDA iterations K , number of SGD, TD update iterations T upd Initialization : uniform policy π θ 0

- 1: for t = 1 , · · · , T -1 do
- 2: Set temperature parameter τ t +1
- 3: Sample the tuple { s i , a i , a 0 i , s ′ i , a ′ i } T upd i =1
- 4: Run TD as Algorithm 5: Q ω t = NN ( ω t ; m Q )
- 5: Calculate V ω t and the advantage A ω t = Q ω t -V ω t
- 6: Run EMDA as Algorithm 2 with L Hinge ( θ )
- 7: Run SGD as Algorithm 6: f θ t +1 = NN ( θ t +1 ; m f )
- 8: Update the policy π θ t +1 ∝ exp { τ -1 t +1 f θ t +1 }
- 9: end for

## Algorithm 2: EMDA

Input : L Hinge ( θ ) , EMDA step size η , number of EMDA iterations K , initial policy π θ t , sample batch { s i } T upd i =1 Initialization : ˜ θ (0) = π θ t , C t ( s, a ) = 0 , for all s, a Output : ̂ π t +1 and C t

- 1: for k = 0 , · · · , K -1 do
- 2: for each state s in the batch do
- 4: Let w s = ( e -ηg s, 1 , . . . , e -ηg s, |A| )
- 3: Find g ( k ) s,a = ∂L Hinge ( θ ) ∂θ s,a ∣ ∣ ∣ θ =˜ θ ( k ) , for each a
- 5: ˜ θ ( k +1) = 1 ⟨ w s , ˜ θ ( k ) ⟩ ( w s ◦ ˜ θ ( k ) )

̸

- 6: C t ( s, a ) ← C t ( s, a ) -ηg ( k ) s,a /A ω t ( s, a ) , for each a with A ω t ( s, a ) = 0
- 7: end for
- 8: end for
- 9: ̂ π t +1 = ˜ θ ( K )

to Algorithm 3 in Appendix A for the complete version) and the pseudo code of EMDA as Algorithm 2. The pseudo code of Algorithms 5-6 used by Algorithm 1 is in Appendix A.

Regarding our analyses, we need assumptions about distribution density. Assumption 4 states that the distribution σ π is sufficiently regular, which is required to analyze the neural network error. Additionally, the common theory works (Antos, Szepesv´ ari, and Munos 2007; Farahmand, Szepesv´ ari, and Munos 2010; Farahmand et al. 2016; Chen and Jiang 2019; Liu et al. 2019) have the concentrability assumption, we also have this common regularity condition.

Assumption 4 (Regularity of Stationary Distribution) . Given any state-action visitation distribution σ π , there exists a universal upper bounding constant c &gt; 0 for any weight vector z ∈ R d and ζ &gt; 0 , such that E σ π [ 1 {| z ⊤ ( s, a ) | ≤ ζ }| z ] ≤ c · ζ/ ∥ z ∥ 2 holds almost surely.

Assumption 5 (Concentrability Coefficient and Ratio) . Define the density ratio between the policy-induced distributions and the policies,

<!-- formula-not-decoded -->

where the above fractions are the Radon-Nikodym Derivatives. We assume that there exist 0 &lt; ϕ ∗ , ψ ∗ &lt; ∞ such that ϕ ∗ t &lt; ϕ ∗ and ψ ∗ t &lt; ψ ∗ , for all t . Also, let C ∞ &lt; ∞ be the concentrability coefficient. We assume that the density ratio between the optimal state distribution and any state distribution, i.e. ∥ ν ∗ /ν ∥ ∞ &lt; C ∞ for any ν .

## 5.3 Convergence Guarantee of Neural PPO-Clip

In this subsection, we present the convergence analysis of Neural PPO-Clip. Inspired by the analysis of (Liu et al. 2019), we analyze the convergence behavior of Neural PPOClip based on the neural networks analysis technique. Nevertheless, the analysis presents several unique technical challenges in establishing its convergence: (i) Tight coupling between function approximation error and the clipping behavior : The clipping mechanism can be viewed as an indicator function. The function approximation for advantage would significantly influence the value of the indicator function in a highly complex manner. As a result, handling the error between the neural approximated advantage and the true advantage serves as one major challenge in the analysis (please refer to the proof of Lemma 5 in Appendix C for more details); (ii) Lack of a closed-form expression of policy update : Due to the clipping function in the hinge loss objective and the iterative updates in the EMDA subroutine, the new policy does not have a simple closed-form expression. This is one salient difference between the analysis of Neural PPOClip and other neural algorithms (cf. (Liu et al. 2019)); (iii) Neural networks analysis on advantage function : Another technicality is that the advantage function requires the neural network projection and linearization properties to characterize the approximation error. However, since we use the neural network to approximate the state-action value function instead of the advantage function, it requires additional effort to establish the error bound of the advantage function (please refer to the proof of Lemma 3).

Given that we need to analyze the error between our approximation and the true function, we further define the target policy under the true advantage function A π θ t as π t +1 ( a | s ) := ¯ C t ( s, a ) A π θ t ( s, a ) + τ -1 t f θ t ( s, a ) , where ¯ C t ( s, a ) is the C t ( s, a ) obtained under A π θ t . Moreover, all the expectations about A ω throughout the analysis are with respect to the randomness of the neural network initialization. Below we state the min-iterate convergence rate and the sufficient condition of Neural PPO-Clip, which is also the main theorem of our paper. Throughout this section, we solely suppose Assumptions 1, 4, and 5 hold.

The central result of this paper is Theorem 2. In this theorem, L C ( T ) and U C ( T ) are functions influenced by T and determined by ¯ C t , a classifier-specific attribute. For detailed supporting lemmas and proofs, see Appendix C.

Theorem 2 (General Convergence Rate of Neural PPOClip) . Consider the Neural PPO-Clip with the classifier satisfying the following conditions for all t ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the policy sequence { π θ t } T t =0 obtained by Neural PPO-Clip satisfies

<!-- formula-not-decoded -->

where ε t = C ∞ τ -1 t +1 ϕ ∗ ϵ 1 / 2 t +1 + Y 1 / 2 ψ ∗ ϵ ′ 1 / 2 t , ε ′ t = |A| · C ∞ τ -2 t +1 ϵ t +1 , M = 4 E ν t [max a ( Q ω 0 ( s, a )) 2 ] + 4 R 2 f , and Y = 2 M +2( R max / (1 -γ )) 2 .

To demonstrate that our convergence analysis is general for Neural PPO-Clip with various classifiers, we choose to state Theorem 2 in a general form utilizing the condition

(14) and (15). Indeed, we show that (14) and (15) can be naturally satisfied by using the standard PPO-Clip classifier described in (7) in the following Corollary 1. Importantly, these conditions are not technical assumptions for our theorem. Notably, we also establish that PPO-Clip-sub (a variant of generalized PPO-Clip utilizing a distinct classifier) aligns with the result presented in Theorem 2. For a comprehensive statement and analysis, please refer to Appendix D.

Corollary 1 (Global Convergence of Neural PPO-Clip, Informal) . Consider Neural PPO-Clip with the standard PPO-Clip classifier ρ s,a ( θ ) -1 and the objective function L ( t ) ( θ ) in each iteration t as

<!-- formula-not-decoded -->

(i) If we specify the EMDA step size η = T -α where α ∈ [1 / 2 , 1) and the temperature parameter τ t = T α / ( Kt ) . Recall that K is the number of EMDA iterations. Let the neural networks' widths be m f , m Q , and the SGD and TD updates T upd be configured as in Appendix D, we have

<!-- formula-not-decoded -->

Hence, Neural PPO-Clip has O ( T -α ) convergence rate. (ii) Furthermore, let the α = 1 / 2 , we obtain the fastest convergence rate, which is O (1 / √ T ) .

Notably, the min-iterate convergence rates presented in (16) and (18) are commonly observed in the realms of nonconvex optimization and neural network theory (LacosteJulien 2016; Ghadimi and Lan 2016; Liu et al. 2019), and they do not constitute stringent results. Furthermore, it is worth pointing out that in (16), the terms ε t and ε ′ t correspond to the errors introduced by policy improvement and policy evaluation, respectively. These errors can be controlled by adjusting neural network widths and the number of TD and SGD iterations T upd, and they can be made arbitrarily small. Further details can be found in Appendix C.

Consequently, the convergence rate obtained by our analysis is determined by U C ( T ) 2 /L C ( T ) . After a brief calculation, it becomes evident that under conditions (14) and (15), the most optimal convergence rate achievable through (16) is O (1 / √ T ) . This scenario arises when L C ( T ) = U C ( T ) = O ( T -1 / 2 ) . This insight underscores that within our analysis, the original PPO-Clip stands as the algorithm that achieves the most favorable bound.

## 5.4 Understanding the Clipping Mechanism

In this subsection, we delve into the more profound understanding of the clipping mechanism.

Rationale Behind the PPO-Clip Convergence. As outlined in Section 3, the clipping mechanism establishes a connection to the hinge loss, consequently shaping the objective as (8). Notably, in the context of the original PPO-Clip, we

160

140

100

2 120

60

40

20

1M

20

15

10

300

200

100

-100

-200

-300

500

400

$ 200

\_-----—==-----

Neural PPO-clip

Neural PPO-clip-sub

Neural PPO-clip-log

Neural PPO-clip-root

Figure 1: Evaluation of PPO-Clip with different classifiers and popular benchmark methods in MinAtar and OpenAI Gym.

<!-- image -->

specify the objective as follows:

<!-- formula-not-decoded -->

We delve more deeply into this objective (19). It is important to note that if the signs of the advantages are incorrect, it can lead to significant errors in computing the objective value during learning. However, due to the impressive empirical performance of neural networks in approximating values, erroneous signs of advantages tend to occur mainly when | A π ( s, a ) | is close to zero. Moreover, when | A π ( s, a ) | is near zero, its contribution to the objective remains relatively insignificant. Consequently, despite incorrect signs, the objective value remains reasonably accurate. This perspective offers an explanation for the robustness and impressive empirical performance of PPO-Clip. Additionally, this notion supports the potential of PPO-Clip to achieve convergence. Furthermore, this concept is essential to comprehend the novel proof technique introduced in Lemma 5. This lemma forms the cornerstone for bounding the errors in policy improvement and evaluation. For more detailed insights, please refer to Appendix C.

Characterization of the Clipping Mechanism. Our convergence analysis reveals that clipping mechanisms solely impact the pre-constant of convergence rates. Surprisingly, our analysis and results show that the clipping range ϵ only influences the pre-constant of the Neural PPO-Clip convergence rate. This is unexpected since, intuitively, ϵ is considered analogous to the penalty parameter of PPO-KL (Liu et al. 2019), which directly affects convergence rates. Contrary to expectations, we discover that the EMDA step size η plays a crucial role in determining convergence rates, rather than the clipping range ϵ . This result is illustrated by the involvement of the clipping mechanism in the EMDA subroutine through the indicator functions in the gradients. Moreover, as the clipping range ϵ is contained inside the indicator function, it only influences the number of effective EMDA updates but not the magnitude of each EMDA update . Since we know that the convergence rate is determined by the magnitude of the gradient updates (i.e., U C ( T ) , L C ( T ) , which is η -dependent and η is T -dependent), the clipping range can only affect the pre-constant of the convergence rate and the rate would still be O (1 / √ T ) . For a more comprehensive understanding, please refer to Appendices C and D.

## 6 Experiments

Experimental Setup. Given the convergence guarantees in Section 5.3, to better understand the empirical behavior of the generalized PPO-Clip objective, we further conduct experiments to evaluate Neural PPO-Clip with different classifiers. Specifically, we evaluate Neural PPOClip, Neural PPO-Clip-sub (as introduced in Section 3), and two additional classifiers, log( π θ ( a | s )) -log( π θ t ( a | s )) and √ ρ s,a ( θ ) -1 (termed as Neural PPO-Clip-log and Neural PPO-Clip-root), against benchmark approaches in several RL benchmark environments. Our implementations of Neural PPO-Clip are based on the RL Baseline3 Zoo framework (Raffin 2020). We test the algorithms in both MinAtar (Young and Tian 2019) and OpenAI Gym environments (Brockman et al. 2016). In addition, the algorithms are compared with popular baselines, including A2C and Rainbow. A2C follows the implementation and default settings from RL Baseline3 Zoo. For Rainbow, we adopt the configuration from (Ceron and Castro 2021). Please refer to Appendix G for more details about our experiment settings.

Variants of Neural PPO-Clip Achieves Comparable Empirical Performance. Figure 1 shows the training curves of Neural PPO-Clip with various classifiers and the benchmark methods. Notably, we observe that Neural PPO-Clip with various classifiers can achieve comparable or better performance than the baseline methods in both RL environments. To be mentioned, the performance of Rainbow is consistent with the results reported by (Ceron and Castro 2021). In summary, the outcomes depicted above underscore the practicality of the hinge loss reinterpretation of PPO-Clip within standard RL tasks. Furthermore, this approach positions classifier selection as a potential hyperparameter for the future deployment of PPO-Clip.

## 7 Concluding Remarks

The convergence behavior of PPO-Clip, a longstanding open problem, is addressed in this paper, providing the first convergence result and deeper insights. Our limitations are (i) analysis under discrete action space and (ii) reliance on NN error analysis, typically requiring large NN width. Despite the empirical success of PPO-Clip without this, our twolayer NN exploration suggests our results hold if approximation errors are well-managed. We anticipate this work will spark a deeper understanding of PPO-Clip within the RL community.

E 300.

€100

Rewards

Rewards

## Acknowledgment

We thank Hsuan-Yu Yao, Kai-Chun Hu, and Liang-Chun Ouyang for the helpful discussion and for providing insightful advice regarding the experiment. This material is based upon work partially supported by the National Science and Technology Council (NSTC), Taiwan under Contract No. NSTC 112-2628-E-A49-023 and Contract No. NSTC 1122634-F-A49-001-MBK and based upon work partially supported by the Higher Education Sprout Project of the National Yang Ming Chiao Tung University and Ministry of Education (MOE), Taiwan.

## References

Agarwal, A.; Kakade, S. M.; Lee, J. D.; and Mahajan, G. 2019. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. arXiv preprint arXiv:1908.00261 .

Agarwal, A.; Kakade, S. M.; Lee, J. D.; and Mahajan, G. 2020. Optimality and approximation with policy gradient methods in markov decision processes. In Conference on Learning Theory , 64-66. PMLR.

Antos, A.; Szepesv´ ari, C.; and Munos, R. 2007. Fitted Qiteration in continuous action-space MDPs. Advances in neural information processing systems , 20.

Beck, A.; and Teboulle, M. 2003. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31(3): 167-175.

Bhandari, J.; and Russo, D. 2019. Global optimality guarantees for policy gradient methods. arXiv preprint arXiv:1906.01786 .

Brockman, G.; Cheung, V.; Pettersson, L.; Schneider, J.; Schulman, J.; Tang, J.; and Zaremba, W. 2016. Openai gym. arXiv preprint arXiv:1606.01540 .

Byun, J.-S.; Kim, B.; and Wang, H. 2020. Proximal Policy Gradient: PPO with Policy Gradient. arXiv preprint arXiv:2010.09933 .

Ceron, J. S. O.; and Castro, P. S. 2021. Revisiting rainbow: Promoting more insightful and inclusive deep reinforcement learning research. In International Conference on Machine Learning , 1373-1383. PMLR.

Chen, G.; Peng, Y.; and Zhang, M. 2018. An adaptive clipping approach for proximal policy optimization. arXiv preprint arXiv:1804.06461 .

Chen, J.; and Jiang, N. 2019. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , 1042-1051. PMLR.

Farahmand, A.-m.; Ghavamzadeh, M.; Szepesv´ ari, C.; and Mannor, S. 2016. Regularized policy iteration with nonparametric function spaces. The Journal of Machine Learning Research , 17(1): 4809-4874.

Farahmand, A.-m.; Precup, D.; Barreto, A.; and Ghavamzadeh, M. 2014. Classification-based approximate policy iteration: Experiments and extended discussions. arXiv preprint arXiv:1407.0449 .

Farahmand, A.-m.; Szepesv´ ari, C.; and Munos, R. 2010. Error propagation for approximate policy and value iteration. Advances in Neural Information Processing Systems , 23.

Fazel, M.; Ge, R.; Kakade, S.; and Mesbahi, M. 2018. Global convergence of policy gradient methods for the linear quadratic regulator. In International Conference on Machine Learning , 1467-1476. PMLR.

Ghadimi, S.; and Lan, G. 2016. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming , 156(1-2): 59-99.

Hu, K.-C.; Hsieh, P.-C.; Wei, T. H.; and Wu, I.-C. 2020. Rethinking Deep Policy Gradients via State-Wise Policy Improvement. In 'I Can't Believe It's Not Better!'NeurIPS 2020 workshop .

Kakade, S. M.; and Langford, J. 2002. Approximately Optimal Approximate Reinforcement Learning. In International Conference on Machine Learning , 267-274.

Lacoste-Julien, S. 2016. Convergence rate of frank-wolfe for non-convex objectives. arXiv preprint arXiv:1607.00345 .

Lagoudakis, M. G.; and Parr, R. 2003. Reinforcement learning as classification: Leveraging modern classifiers. In International Conference on Machine Learning , 424-431.

Lazaric, A.; Ghavamzadeh, M.; and Munos, R. 2010. Analysis of a classification-based policy iteration algorithm. In International Conference on Machine Learning , 607-614.

Liu, B.; Cai, Q.; Yang, Z.; and Wang, Z. 2019. Neural trust region/proximal policy optimization attains globally optimal policy. Advances in Neural Information Processing Systems , 32: 10565-10576.

Liu, Y.; Zhang, K.; Basar, T.; and Yin, W. 2020. An improved analysis of (variance-reduced) policy gradient and natural policy gradient methods. Advances in Neural Information Processing Systems , 33: 7624-7636.

Mei, J.; Xiao, C.; Szepesvari, C.; and Schuurmans, D. 2020. On the global convergence rates of softmax policy gradient methods. In International Conference on Machine Learning , 6820-6829.

Pi, C.-H.; Hu, K.-C.; Cheng, S.; and Wu, I.-C. 2020. Lowlevel autonomous control and tracking of quadrotor using reinforcement learning. Control Engineering Practice , 95: 104222.

Raffin, A. 2020. RL Baselines 3 Zoo. Available at https: //github.com/DLR-RM/rl-baselines3-zoo.

Schulman, J.; Levine, S.; Abbeel, P.; Jordan, M.; and Moritz, P. 2015. Trust region policy optimization. In International conference on machine learning , 1889-1897. PMLR.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .

Shani, L.; Efroni, Y.; and Mannor, S. 2020. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In AAAI Conference on Artificial Intelligence , volume 34, 5668-5675.

Singh, S.; Jaakkola, T.; Littman, M. L.; and Szepesv´ ari, C. 2000. Convergence results for single-step on-policy reinforcement-learning algorithms. Machine learning , 38(3): 287-308.

Wang, L.; Cai, Q.; Yang, Z.; and Wang, Z. 2019. Neural Policy Gradient Methods: Global Optimality and Rates of Convergence. In International Conference on Learning Representations .

Wang, W.; Han, J.; Yang, Z.; and Wang, Z. 2021. Global convergence of policy gradient for linear-quadratic meanfield control/game in continuous time. In International Conference on Machine Learning , 10772-10782. PMLR.

Ye, D.; Liu, Z.; Sun, M.; Shi, B.; Zhao, P.; Wu, H.; Yu, H.; Yang, S.; Wu, X.; Guo, Q.; et al. 2020. Mastering complex control in moba games with deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, 6672-6679.

Young, K.; and Tian, T. 2019. MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments. arXiv preprint arXiv:1903.03176 .

- 7: end for
- 8: end for
- 9: ̂ π t +1 = ˜ θ ( K )

For consistency in notation, we present the EMDA utilized in Tabular PPO-Clip as Algorithm 8.

## Appendix

## A Pseudo Code of Algorithms

Algorithm 3: Neural PPO-Clip (A More Detailed Version of Algorithm 1)

Input : MDP ( S , A , P , r, γ, µ ) , Objective function L Hinge, EMDA step size η , number of EMDA iterations K , number of SGD and TD update iterations T upd, number of Neural PPO-Clip iterations T , the clipping range ϵ

Initialization : the policy π θ 0 as a uniform policy

- 1: for t = 1 , · · · , T -1 do
- 2: Set temperature parameter τ t +1
- 3: Sample the tuple { s i , a i , a 0 i , s ′ i , a ′ i } T upd i =1 , where ( s i , a i ) ∼ σ t , a 0 i ∼ π 0 ( ·| s i ) , s ′ i ∼ P ( ·| s i , a i ) and a ′ i ∼ π θ t ( ·| s ′ i )
- 4: Solve for Q ω t = NN ( ω t ; m Q ) by using TD update as Algorithm 5
- 5: Calculate V ω t by Bellman expectation equation and the advantage A ω t = Q ω t -V ω t
- 6: Use the states with nonzero advantage as the batch { s i } T upd i =1 for L Hinge ( θ ) and obtain target policy ̂ π t +1 and C t by using EMDA in Algorithm 2
- 7: Solve for f θ t +1 = NN ( θ t +1 ; m f ) by using SGD as Algorithm 6 based on the EMDA result
- 8: Update the policy π θ t +1 ∝ exp { τ -1 t +1 f θ t +1 }
- 9: end for

Remark A.1. In Neural PPO-Clip, there are various types of classifiers, the choices of the EMDA step size η and the temperature parameters { τ t } of the neural networks are important factors to the convergence rate and hence shall be configured properly according to the properties of different classifiers. As a result, we do not specify the specific choices of η and { τ t } in the following pseudo code of the generic Neural PPO-Clip. Please refer to Corollaries 1-2 in Appendix D for the choices of η

and { τ t } for Neural PPO-Clip with several classifiers including the standard PPO-Clip classifier ρ s,a ( θ ) -1 = π θ ( a | s ) π θ t ( a | s ) -1 .

For better readability, we restate EMDA (Algorithm 2) here as Algorithm 4.

## Algorithm 4: EMDA

Input : L Hinge ( θ ) , EMDA step size η , number of EMDA iterations K , initial policy π θ t , sample batch { s i } T upd i =1 Initialization : ˜ θ (0) = π θ t , C t ( s, a ) = 0 , for all s, a

Output :

π

t

+1

and

C

t

- ̂ 1: for k = 0 , · · · , K -1 do
- 2: for each state s in the batch do
- θ θ 4: Let w s = ( e -ηg s, 1 , . . . , e -ηg s, |A| )
- 3: Find g ( k ) s,a = ∂L Hinge ( θ ) ∂θ s,a ∣ ∣ ∣ =˜ ( k ) , for each a
- 5: ˜ θ ( k +1) = 1 ⟨ w s , ˜ θ ( k ) ⟩ ( w s ◦ ˜ θ ( k ) )
- 6: C t ( s, a ) ← C t ( s, a ) -ηg ( k ) s,a /A ω t ( s, a ) , for

each

a

with

A

ω

t

(

s, a

)

̸

= 0

## Algorithm 5: Policy Evaluation via TD

Input : MDP ( S , A , P , r, γ ) , initial weights b i , [ ω (0)] i ( i ∈ [ m Q ]) , number of iterations T upd, sample { s i , a i , s ′ i , a i } T upd i =1 Output : Q ¯ ω

- 1: Set the step size η upd ← T -1 / 2 upd
- 2: for t = 0 , · · · , T upd -1 do
- 4: ω ( t +1 / 2) ← ω ( t ) -η upd · ( Q ω ( t ) ( s, a ) -r ( s, a ) -γQ ω ( t ) ( s ′ , a ′ )) · ∇ ω Q ω ( t ) ( s, a )
- 3: ( s, a, s ′ , a ′ ) ← ( s i , a i , s ′ i , a ′ i )
- 5: ω ( t +1) ← arg min ω ∈ B Q {|| ω -ω ( t +1 / 2) || 2 }
- 6: end for
- 7: Take the average over path ¯ ω ← 1 /T upd · ∑ T upd -1 t =0 ω ( t )

## Algorithm 6: Policy Improvement via SGD

Input : MDP ( S , A , P , r, γ ) , the current energy function f θ t , initial weights b i , [ θ (0)] i ( i ∈ [ m f ]) , number of iterations T upd, sample { s i , a 0 i } T upd i =1

## f ¯ θ

Output :

- 1: Set the step size η upd ← T -1 / 2 upd
- 2: for t = 0 , · · · , T upd -1 do
- 3: ( s, a ) ← ( s i , a 0 i )

4:

+1

θ

(

t

/

2)

θ

(

t

)

η

-

←

- t 5: θ ( t +1) ← arg min θ ∈ B f {|| θ -θ ( t +1 / 2) || 2 }

upd

- 6: end for

-

(

f

(

s, a

)

τ

(

C

(

s, a

·

·

+1

θ

t

t

- 7: Take the average over path ¯ θ ← 1 /T upd · ∑ T upd -1 t =0 θ ( t )

## Algorithm 7: Tabular PPO-Clip

Initialization : policy π (0) = π ( θ (0) ) , initial state distribution µ , step size of EMDA η , number of EMDA iterations K ( t ) Output : Learned policy π ( ∞ )

- 1: for t = 0 , 1 , · · · do
- 2: Collect a set of trajectories τ ∈ D ( t ) under policy π ( t ) = π ( θ ( t ) )
- 3: Find A ( t ) by a policy evaluation method
- 4: Compute ˆ L ( t ) ( θ ) based on A ( t ) and the collected samples in D ( t )
- 5: Update the policy by θ ( t +1) = EMDA-tabular ( ˆ L ( t ) ( θ ) , η, K ( t ) , D ( t ) , θ ( t ) )
- 6: end for

## Algorithm 8: EMDA-tabular ( L ( θ ) , η, K, D , θ init )

Input : Objective L ( θ ) , step size η , number of iteration K , dataset D , and initial parameter θ init

Initialization :

˜

˜

init

,

θ

θ

θ

=

=

(0)

: Learned parameter ˜ θ

Output

- 1: for k = 0 , · · · , K -1 do
- 2: for each state s in D do
- 3: Find g ( k ) s,a = ∂L ( θ ) ∂θ s,a | θ = ˜ θ ( k ) , for each a
- 4: Let w s = ( e -ηg ( k ) s, 1 , · · · , e -ηg ( k ) s, |A| )

<!-- formula-not-decoded -->

- 6: end for
- 7: end for

θ

init

)

·

A

ω

t

(

s, a

) +

τ

-

1

t

f

θ

t

(

s, a

)))

· ∇

θ

f

θ

t

(

s, a

)

## B Proof of Proposition 1

For completeness, we restate Proposition 1 as follows.

Proposition (EMDA Target Policy) . For the target policy obtained by the EMDA subroutine at the t -th iteration, we have

<!-- formula-not-decoded -->

where C t ( s, a ) A ω t ( s, a ) = -∑ K -1 k =0 ηg ( k ) s,a as given in Algorithm 2.

Proof of Proposition 1. We expand the closed-form of the log of the EMDA target policy,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Z t ( s ) is the normalizing factor of the policy at step t . Since both the ∑ K ( t ) -1 k =0 log( ⟨ w s , ˜ θ ( k ) ⟩ ) and log( Z t ( s )) are statedependent, we can cancel it under softmax policy. We obtain C t ( s, a ) from Algorithm 2 and complete the proof.

## C Proof of the Supporting Lemmas for Theorem 2

In the following, we slightly abuse the notations E ˜ σ t , E σ t , and E ν ∗ to denote the expectations (over the respective distribution) conditioned on the policy π θ t .

## C.1 Additional Supporting Lemmas

Throughout this section, we slightly abuse the notation that we use E init [ · ] to denote the expectation over the initialization of neural networks. Also, we assume that Assumptions 1, 4, and 5 hold in the following proofs.

Lemma 1 (Policy Evaluation Error) . The output A ¯ ω = Q ¯ ω -V ¯ ω of Algorithm 5 and Bellman expectation equation satisfies

<!-- formula-not-decoded -->

To prove Lemma 1, we start by stating a bound on the error of the estimated state-action value function.

Lemma 2 (Theorem 4.6 in (Liu et al. 2019)) . The output Q ¯ ω of Algorithm 5 satisfies

<!-- formula-not-decoded -->

Proof of Lemma 1. We are ready to show the policy evaluation error of the advantage function. First, we find the bound of | A ω t ( s, a ) -A π θ t ( s, a ) | . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, we can derive the bound of ( A π θ t ( s, a ) -A ω t ( s, a )) 2 as follows,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (32) holds by Jensen's inequality. By taking the expectation of (31)-(32) over the state-action distribution σ t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality in (35) is obtained by the actions are directly sampled by π θ t so we can ignore it in the latter term. Last, we leverage Lemma 2 to obtain the result of Lemma 1.

Lemma 3 (Policy Improvement Error) . The output f ¯ θ of Algorithm 6 satisfies

<!-- formula-not-decoded -->

To prove Lemma 3, we first state the following useful result noindently proposed by (Liu et al. 2019).

Theorem 3 ((Liu et al. 2019), Meta-Algorithm of Neural Networks) . Consider a meta-algorithm with the following update:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where µ ∈ [0 , 1) is a constant, ( s, a, s ′ , a ′ ) is sampled from some stationary distribution d , u α is parameterized as a two-layer neural network NN ( α ; m ) , and v ( s, a ) satisfies

<!-- formula-not-decoded -->

for some constants ¯ v 1 , ¯ v 2 , ¯ v 3 ≥ 0 . We define the update operator T u ( s, a ) = E [ v ( s, a ) + µ · u ( s ′ , a ′ ) | s ′ ∼ P ( ·| s, a ) , a ′ ∼ π ( ·| s )] , and define α ∗ as the approximate stationary point (cf. (D.18) in (Liu et al. 2019)), which inherently have the property u 0 α ∗ = ∏ F Ru,m T u 0 α ∗ , where u 0 α ∗ is the linearization of u at α ∗ . Suppose we run the above meta-algorithm in (37)-(38) for T iterations with T ≥ 64 / (1 -µ ) 2 and set the step size η upd = T -1 / 2 . Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ α := 1 /T · ( ∑ T -1 t =0 α ( t )) and α ′ is a parameter in B α .

Proof of Lemma 3. Now we are ready to prove Lemma 3 as follows. To begin with, (37)-(38) match the policy improvement update of Neural PPO-Clip if we put u ( s, a ) = f ( s, a ) , v ( s, a ) = τ t +1 ( C t ( s, a ) · A ω t ( s, a ) + τ -1 t f θ t ( s, a )) , µ = 0 , d = ˜ σ t , and R u = R f . For E ˜ σ t [( v ( s, a )) 2 ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, since C t and ¯ C t are dependent only on the EMDA step size η and the indicator function that depends on the sign of the advantage (either under the true advantage A π θ t or the approximated advantage A ω t ), one can always find one common upper bound U C ( T ) for both C t and ¯ C t . In particular, as shown in Corollary 1, we set U C = ∑ K -1 k =0 η for PPO-Clip, which is independent from the advantage function. The inequality in (43) holds by the condition that τ 2 t +1 ( U 2 C + τ -2 t ) ≤ 1 , ( a + b ) 2 ≤ 2 a 2 +2 b 2 , E ˜ σ t [( A ω t ( s, a )) 2 ] ≤ 4 E ˜ σ t [( Q ω t ( s, a )) 2 ] , and E ˜ σ t [( u α t ( s, a )) 2 ] ≤ 2 E ˜ σ t [( u α 0 ( s, a )) 2 ] + 2 R 2 f which holds by using the Lipschitz property of neural networks where u α = f θ , A ω . The condition τ 2 t +1 ( U 2 C + τ -2 t ) ≤ 1 can be satisfied by configuring proper { τ t } , as described momentarily in Appendix D. We also use that E ˜ σ t [ Q ω (0) ] = E ˜ σ t [ f θ (0) ] because they share the same initialization. Thus, we have ¯ v 1 = ¯ v 2 = 20 and ¯ v 3 = 0 in (39).

Due to that θ ∗ is the approximate stationary point, we have f 0 θ ∗ = ∏ F R f ,m f T f 0 θ ∗ = ∏ F R f ,m f τ t +1 ( C t ◦ A ω t + τ -1 t f θ t ) . Thus,

<!-- formula-not-decoded -->

where ∥·∥ 2 , ˜ σ t = E init, ˜ σ t [ ∥·∥ 2 ] 1 / 2 is the ˜ σ t -weighted ℓ 2 -norm. Then, by the fact that τ t +1 ( C t ( s, a ) · A 0 ω t ( s, a ) + τ -1 t f 0 θ t ( s, a )) ∈ F R f ,m f and that A 0 ω t ( s, a ) = Q 0 ω t ( s, a ) -∑ a ∈A π ( a | s ) Q 0 ω t ( s, a ) , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We obtain (48) as the same reason in (31)-(35) in the proof of Lemma 1. The terms in (48) are both the designated form as the (41), we leverage the (41) in Theorem 3 and obtain the result in (49).

Last, we bound the error of our policy improvement, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (51) is bounded as O ( R 2 f T -1 / 2 upd + R 5 / 2 f m -1 / 4 f + R 3 f m -1 / 2 f ) by (40) of Theorem 3, and (52) is bounded as O ( R 3 f m -1 / 2 f ) by the derivation of (49). Thus, we obtain (53) and complete the proof.

Lemma4 (Error Probability of Advantage) . Given the policy π θ t , the probability of the event that the advantage error is greater than ϵ err can be bounded as

<!-- formula-not-decoded -->

Proof of Lemma 4. By applying Markov's inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that the randomness of the above event in (54) comes from the state-action visitation distribution σ t and the initialization of the neural networks.

Lemma 5 (Error Propagation) . Let π t +1 be the target policy obtained by EMDA with the true advantage. Suppose the policy improvement error satisfies

<!-- formula-not-decoded -->

and the policy evaluation error satisfies

<!-- formula-not-decoded -->

Then, the following holds,

<!-- formula-not-decoded -->

where ε t = C ∞ τ -1 t +1 ϕ ∗ ϵ 1 / 2 t +1 + U C X 1 / 2 ψ ∗ ϵ ′ 1 / 2 t and ε err = √ 2 U C ϵ err ψ ∗ , and X = [ (2 /ϵ 2 err )( M ′ +( R max / (1 -γ )) 2 -ϵ ′ t / 2) ] , and M ′ = 4 E ν t [max a ( Q ω 0 ( s, a )) 2 ] + 4 R 2 f .

Remark C.1. Notice that ϵ t +1 in (57) and ϵ ′ t in (58) can be controlled by the width of neural networks and the number of iteration for each SGD and TD updates based on Lemma 1 and 3. Therefore, ε t could be made sufficiently small per our requirement.

Proof of Lemma 5. For ease of exposition, let us first fix a policy π θ t . Through the analysis, we will show that one can derive an upper bound (in the form of (59)) that holds regardless of the policy π θ t . Recall that C t ( s, a ) = -∑ K ( t ) -1 k =0 ηg ( k ) s,a , where g ( k ) s,a is obtained in the EMDA subroutine and depends on the sign of the estimated advantage A ω t . Similarly, we define ¯ C t ( s, a ) as the counterpart of C t ( s, a ) by replacing A ω t with the true advantage A π θ t . We first simplify ⟨ log π θ t +1 ( ·| s ) -log π t +1 ( ·| s ) , π ∗ ( ·| s ) -π θ t ( ·| s ) ⟩ . The normalizing factor Z of the policies π θ t +1 and π t +1 is state-dependent, and the inner product between any state-dependent function and the policy difference π ∗ ( ·| s ) -π θ t ( ·| s ) is always zero. Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, we decompose the above equation into two terms: (i) the error in the policy improvement and (ii) the error between the true advantage and the approximated advantage, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first bound the expectation of (i) over ν ∗ as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (67) follows from the definition of ˜ σ t , (68) is obtained by Cauchy-Schwarz inequality and Assumption 5, and the last inequality in (69) holds by the condition in (57) and that ∥ ν ∗ /ν ∥ ∞ &lt; C ∞ . Similarly, we consider the expectation of (ii) over ν ∗ as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (74) holds by the Cauchy-Schwarz inequality. Next, we bound for the term E σ t [( C t ( s, a ) A ω t ( s, a ) -¯ C t ( s, a ) A π θ t ( s, a )) 2 ] . For ease of notation, let D = ( C t ( s, a ) A ω t ( s, a ) -¯ C t ( s, a ) A π θ t ( s, a )) 2 and simply write E init, σ t as E . Also, we slightly abuse the notation by using A ω t as the random variable A ω t ( s, a ) , whose randomness results from the state-action pairs sampled from σ t and the initialization of neural networks, and using A π θ t as the random variable A π θ t ( s, a ) , whose randomness comes from the state-action pairs sampled from σ t . To establish the bound of E [ D ] , we consider two different cases for E [ D ] : one is that the error is greater than ϵ err, and the other is that the error is less than or equal to ϵ err. Specifically,

<!-- formula-not-decoded -->

Then, we upper bound the two terms in (75) separately. Regarding the first term in (75), we have

<!-- formula-not-decoded -->

where (76) holds by that ( a + b ) 2 ≤ 2 a 2 +2 b 2 . Next, regarding the second term in (75), we further consider two cases based on whether the absolute value of A π θ t is greater than ϵ err or not. Specifically,

<!-- formula-not-decoded -->

where (77) holds by the fact that we fix a policy π θ t and hence A π θ t is determined, (78) holds by that the indicator function is no larger than 1, the first term in (79) holds by the fact that A ω t and A π θ t have the same sign and hence C t is equal to ¯ C t , and the second term in (79) follows from that ( a + b ) 2 ≤ 2 a 2 +2 b 2 . Then, by combining the above terms, we have

<!-- formula-not-decoded -->

Recall that ϵ ′ t = E [( A ω t ( s, a ) -A π θ t ( s, a )) 2 ] . As we could choose an ϵ err small enough and use the neural network power to make ϵ ′ t is also small by Lemma 1 such that we have 2 U 2 C ( E ν t [ ∥ A ω t ( s, · ) ∥ 2 ∞ ] + A π θ t max ) &gt; U 2 C ϵ ′ t +4 U 2 C ϵ 2 err , then by Lemma 4 we have

<!-- formula-not-decoded -->

Rearranging the terms in (82), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ′ := 4 E ν t [max a ( Q ω 0 ( s, a )) 2 ] + 4 R 2 f . By introducing the notation X = [ (2 /ϵ 2 err )( M ′ +( A π θ t max ) 2 -ϵ ′ t / 2) ] and combining all the above results, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (87) follows from the inequality √ a + b ≤ √ a + √ b and that ε t = ϵ 1 / 2 t +1 C ∞ τ -1 t +1 ϕ ∗ + ϵ ′ 1 / 2 t U C X 1 / 2 ψ ∗ and ε err = 2 U C ϵ err ψ ∗ . The proof is complete.

Lemma 6 (Stepwise Energy ℓ ∞ -Difference) .

<!-- formula-not-decoded -->

where ε ′ t = |A| · C ∞ τ -2 t +1 ϵ t +1 and M = 4 E ν ∗ [max a ( Q ω 0 ( s, a )) 2 ] + 4 R 2 f .

Remark C.2. As described in Remark C.1, ϵ t +1 can be sufficiently small due to Lemma 3. Similarly, ε ′ t can also be made arbitrarily small.

Proof of Lemma 6. We first find an explicit bound for ∥ τ -1 t +1 f θ t +1 ( s, · ) -τ -1 t f θ t ( s, · ) ∥ 2 ∞ . Note that

<!-- formula-not-decoded -->

Next, we consider the expectation of (90) over ν ∗ : For the first term in (90), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (94) holds by the condition in (57), the definition of the concentrability coefficient, and the fact that π 0 is a uniform policy. Furthermore, we bound E ν ∗ [ ∥ C t ( s, · ) ◦ A ω t ( s, · ) ∥ 2 ∞ ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (99) holds by using Jensen's inequality and leveraging the ℓ ∞ -norm instead of the expectation E a ∼ π θ t [ · ] , and the last inequality in (101) holds by the 1-Lipschitz property of neural networks with respect to the weights. By setting ε ′ t = |A| · C ∞ τ -2 t +1 ϵ t +1 and M = 4 E ν ∗ [max a ( Q ω 0 ( s, a )) 2 ] + 4 R 2 f , we complete the proof of Lemma 6.

Lemma 7 (Stepwise KL Difference) . The KL difference is as follows,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 7. We directly expand the one-step KL divergence difference as

<!-- formula-not-decoded -->

Then, by Pinsker's inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, by Proposition 1, we have log π t +1 ( ·| s ) = log π θ t ( ·| s ) + ¯ C t ( s, · ) ◦ A π θ t ( s, · ) and then apply this to the first term in (109). The proof is complete.

Lemma 8 (Performance Difference Using Advantage) . Recall that L ( π ) = E ν ∗ [ V π ( s )] . We have

<!-- formula-not-decoded -->

Before proving Lemma 8, we first state the following property.

Lemma 9 ((Liu et al. 2019), Lemma 5.1) .

<!-- formula-not-decoded -->

Proof of Lemma 8. As the value function V π ( · ) is state-dependent, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, by (113) and Lemma 9, we have

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 2

By taking expectation of the KL difference in Lemma 7 over ν ∗ , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from Lemma 7 and Lemma 5, the second inequality holds by the H¨ older's inequality, and the last inequality holds by the fact that 2 xy -x 2 ≤ y 2 and merging the last two terms. Then, by Lemma 6 and rearranging the terms, we obtain that

<!-- formula-not-decoded -->

By the first condition of (14), we have L C E ν ∗ [ ⟨ A π θ t ( s, · ) , π ∗ ( ·| s ) -π θ t ( ·| s ) ⟩ ] ≤ E ν ∗ [ ⟨ ¯ C t ( s, · ) ◦ A π θ t ( s, · ) , π ∗ ( ·| s ) -π θ t ( ·| s ) ⟩ ] . By obtaining the performance difference via Lemma 8, we have

<!-- formula-not-decoded -->

Then, by taking the telescoping sum of (121) from t = 0 to T -1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the facts that (i) E ν ∗ [ KL ( π ∗ ( ·| s ) ∥ π θ 0 ( ·| s ))] ≤ log |A| , (ii) KL divergence is nonnegative, (iii) ∑ T -1 t =0 ( L ( π ∗ ) -L ( π θ t )) ≥ T · min 0 ≤ t ≤ T {L ( π ∗ ) -L ( π θ t ) } , we have

<!-- formula-not-decoded -->

Since we have ε err = 2 U C ϵ err ψ ∗ and the condition of (15), we know that if we set ϵ err = U C ( T ) and T to be sufficiently large, ϵ err shall be sufficiently small and hence satisfy the condition required by (82). Thus, by plugging ϵ err = U C ( T ) into (124), we have ε err = 2 U C ( T ) 2 ψ ∗ and ε t = ϵ 1 / 2 t +1 C ∞ τ -1 t +1 ϕ ∗ + ϵ ′ 1 / 2 t U C [[ (2 /U C ( T ) 2 )( M +( A π θ t max ) 2 -ϵ ′ t / 2) ]] 1 / 2 ψ ∗ = ϵ 1 / 2 t +1 C ∞ τ -1 t +1 ϕ ∗ + ϵ ′ 1 / 2 t U C Y 1 / 2 ψ ∗ , where Y = 2 M +2( R max / (1 -γ )) 2 -ϵ ′ t ≤ 2 M +2( R max / (1 -γ )) 2 . Finally, we have

<!-- formula-not-decoded -->

By the condition (15), U C ( T ) 2 can always cancel out T in the numerator of (125). Moreover, in the denominator of (125), L C ( T ) = ω ( T -1 ) is large enough to attain convergence, and we complete the proof.

Remark C.3. As mentioned in Remark A.1, the choices of η and { τ t } would affect the convergence rate and need to be configured properly for Neural PPO-Clip with different classifiers. As will be shown in Appendix D, this fact can be further explained through the bounds U C ( T ) and L C ( T ) obtained in (131) and (143).

## D.1 Proof of Corollary 1

For ease of exposition, we restate the corollary as follows.

Corollary (Global Convergence of Neural PPO-Clip with Convergence Rate) . Consider Neural PPO-Clip with the standard PPO-Clip classifier ρ s,a ( θ ) -1 and the objective function L ( t ) ( θ ) in each iteration t as

<!-- formula-not-decoded -->

(i) If we specify the EMDA step size η = T -α where α ∈ [1 / 2 , 1) and the temperature parameter τ t = T α / ( Kt ) . Recall that K is the maximum number of EMDA iterations. Let the neural networks' widths m f = Ω( R 10 f ϕ ∗ 8 K 8 C 8 ∞ T 12 + R 10 f K 8 T 8 C 4 ∞ |A| 4 ) , m Q = Ω( R 10 Q ψ ∗ 8 Y 4 T 8 ) , and the SGD and TD updates T upd = Ω( R 4 f ϕ ∗ 4 K 4 C 4 ∞ T 6 + R 4 Q ψ ∗ 4 Y 2 T 4 + R 4 f T 4 K 4 C 2 ∞ |A| 2 ) , we have

<!-- formula-not-decoded -->

Hence, Neural PPO-Clip has O ( T -α ) convergence rate. (ii) Furthermore, let the α = 1 / 2 , we obtain the fastest convergence rate, which is O (1 / √ T ) .

Proof of Corollary 1. We find the lower and upper bounds L C ( T ) , U C ( T ) for PPO-Clip. We first consider the derivative g s,a of the objective with the true advantage function A π θ t .

<!-- formula-not-decoded -->

Then, we check the sufficient conditions (14) and (15). Recall that K is the maximum number of EMDA iteration for each t . We sum up the gradients with η and rearrange the terms into ¯ C t ( s, a ) . Then, we have the upper bound as

<!-- formula-not-decoded -->

Regarding the lower bound, as we know that under PPO-Clip, the first step of EMDA shall always make an update, i.e., it will never be clipped, and hence we have

<!-- formula-not-decoded -->

Lastly, by setting η = T -α and selecting the temperature as τ t = T α / ( Kt ) to satisfy the condition τ 2 t +1 ( U 2 C + τ -2 t ) ≤ 1 that we use in (43), we obtain

<!-- formula-not-decoded -->

We have checked the sufficient conditions of Theorem 2. Thus, we obtain,

<!-- formula-not-decoded -->

Then, we show the minimum widths and the number of iterations of SGD and TD updates to attain convergence. We must force the summation of errors ε t , ε ′ t to be O (1) . By Lemma 1, 3, where ϵ t +1 = O ( R 2 f T -1 / 2 upd + R 5 / 2 f m -1 / 4 f + R 3 f m -1 / 2 f ) , ϵ ′ t = O ( R 2 Q T -1 / 2 upd + R 5 / 2 Q m -1 / 4 Q + R 3 Q m -1 / 2 Q ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

when m f = Ω( R 2 f ) and m Q = Ω( R 2 Q ) . Then, by taking m f = Ω( R 10 f ϕ ∗ 8 K 8 C 8 ∞ T 12 ) , m Q = Ω( R 10 Q ψ ∗ 8 Y 4 T 8 ) , and T upd = Ω( R 4 f ϕ ∗ 4 K 4 C 4 ∞ T 6 + R 4 Q ψ ∗ 4 Y 2 T 4 ) , we have

<!-- formula-not-decoded -->

## D Additional Corollaries and Proofs

Moreover, we further put m f = Ω( R 10 f T 8 K 8 C 4 ∞ |A| 4 ) and T upd = Ω( R 4 f T 4 K 4 C 2 ∞ |A| 2 ) , we have

<!-- formula-not-decoded -->

Last, we add up the lower bound of each term of m f , m Q , and T upd, and then sum the errors in (136) and (137) for all t from 0 to T -1 , we obtain

<!-- formula-not-decoded -->

which completes the proof and obtains the O ( T -α ) convergence rate. √

Furthermore, if we set α = 1 / 2 , η will be 1 / T , and we plug into the result above, we have the O (1 / √ T ) convergence rate.

## D.2 Convergence Rate of Neural PPO-Clip With an Alternative Classifier

Corollary 2 (Global Convergence of Neural PPO-Clip with subtraction classifier with Convergence Rate) . Consider Neural PPO-Clip with the subtraction classifier π θ ( a | s ) -π θ t ( a | s ) (termed Neural PPO-Clip-sub) and the objective function L ( t ) ( θ ) in each iteration t as

<!-- formula-not-decoded -->

We specify the EMDA step size η = 1 / √ T and the temperature parameter τ t = √ T/ ( Kt ) . Recall that K is the maximum number of EMDA iterations. Let the neural networks' widths m f = Ω( R 10 f ϕ ∗ 8 K 8 C 8 ∞ T 12 + R 10 f K 8 T 8 C 4 ∞ |A| 4 ) , m Q = Ω( R 10 Q ψ ∗ 8 Y 4 T 8 ) , and the SGD and TD updates T upd = Ω( R 4 f ϕ ∗ 4 K 4 C 4 ∞ T 6 + R 4 Q ψ ∗ 4 Y 2 T 4 + R 4 f T 4 K 4 C 2 ∞ |A| 2 ) , we have

<!-- formula-not-decoded -->

Hence, we provide the O (1 / √ T ) convergence rate of Neural PPO-Clip-sub.

Proof of Corollary 2. Similar to Corollary 1, we derive the gradient of our objective with the true advantage function A π θ t ( s, a ) . Specifically, we have

<!-- formula-not-decoded -->

Thus, similar to D.1, we have

<!-- formula-not-decoded -->

We also set η = 1 / √ T and pick τ t = √ T/ ( Kt ) to satisfy the condition τ 2 t +1 ( U 2 C + τ -2 t ) ≤ 1 that we use in (43). Accordingly, we obtain

<!-- formula-not-decoded -->

We have checked the sufficient condition of Theorem 2. Therefore, by plugging in L C ( T ) and U C ( T ) , we obtain

<!-- formula-not-decoded -->

Similar to the proof of Corollary D.1, we set the same minimum widths and number of iterations to attain convergence, which directly implies

<!-- formula-not-decoded -->

Then, we complete the proof and obtain the O (1 / √ T ) convergence rate of PPO-Clip with a subtraction classifier.

## E Tabular PPO-Clip and Proof

## E.1 Supporting Lemmas for the Proof of Theorem 1

For completeness, we state the state-wise policy improvement Lemma in (Kakade and Langford 2002) and provide the proof. Lemma 10. Given policies π 1 and π 2 , V π 1 ( s ) ≥ V π 2 ( s ) for all s ∈ S if the following holds:

<!-- formula-not-decoded -->

Proof of Lemma 10. By the performance difference lemma (Kakade and Langford 2002), we have

<!-- formula-not-decoded -->

Also, since we have ∑ a ∈A π 2 ( a | s ) A π 2 ( s, a ) = 0 holds for any s ∈ S , if ∑ a ∈A ( π 1 ( a | s ) -π 2 ( a | s )) A π 2 ( s, a ) ≥ 0 holds for any ( s, a ) ∈ S × A , then ∑ a ∈A π 1 ( a | s ) A π 2 ( s, a ) ≥ 0 . Hence, we will obtain V π 1 ( s ) ≥ V π 2 ( s ) for all s ∈ S .

Notably, Lemma 10 offers a useful insight that policy improvement can be achieved by simply adjusting the action distribution based solely on the sign of the advantage of the state-action pairs, regardless of their magnitude. We provide the proof in Appendix E.1. Interestingly, one can draw an analogy between (146) in Lemma 10 and learning a linear binary classifier: (i) Features : The state-action representation can be viewed as the feature vector of a training sample; (ii) Labels : The sign of A π 2 ( s, a ) resembles a binary label; (iii) Classifiers : π 1 ( a | s ) -π 2 ( a | s ) serves as the prediction of a linear classifier. We provide the intuition behind using π 1 ( a | s ) -π 2 ( a | s ) as a classifier. Let's fix π 2 and let π 1 be the improved policy. If the sign of A π 2 ( s, a ) ≥ 0 , which implies that the action a has a positive effect on the total return, it is desired to slightly tune up the probability of acting in action a . Thus, the update π 1 must have a greater probability on action a in order to obtain the sufficient condition of the state-wise policy improvement, i.e., ( π 1 ( a | s ) -π 2 ( a | s )) A π 2 ( s, a ) ≥ 0 . Next, we substantiate this insight and rethink PPO-Clip via hinge loss.

As described in Section 3, one major component of the proof of Theorem 1 is the state-wise policy improvement property of PPO-Clip. For ease of exposition, we introduce the following definition regarding the partial ordering over policies.

Definition 1 (Partial ordering over policies) . Let π 1 and π 2 be two policies. Then, π 1 ≥ π 2 , called π 1 improves upon π 2 , if and only if V π 1 ( s ) ≥ V π 2 ( s ) , ∀ s ∈ S . Moreover, we say π 1 &gt; π 2 , called π 1 strictly improves upon π 2 , if and only if π 1 ≥ π 2 and there exists at least one state s such that V π 1 ( s ) &gt; V π 2 ( s ) .

Lemma 11 (Sufficient condition of state-wise policy improvement) . Given any two policies π 1 and π 2 , we have π 1 ≥ π 2 if the following condition holds:

<!-- formula-not-decoded -->

Proof of Lemma 11. This is the same result of the proof of Lemma 10.

Next, we present two critical properties that hold under PPO-Clip for every sample path.

Lemma 12 (Strict improvement and strict positivity of policy under PPO-Clip with direct tabular parameterization) . In any iteration t , suppose π ( t ) is strictly positive in all state-action pairs, i.e., π ( t ) ( a | s ) &gt; 0 , for all ( s, a ) . Under PPO-Clip in Algorithm 7, π ( t +1) satisfies that (i) π ( t +1) &gt; π ( t ) and (ii) π ( t +1) ( a | s ) &gt; 0 , for all ( s, a ) .

Proof of Lemma 12. Consider the t -th iteration of PPO-Clip (cf. Algorithm 7) and the corresponding update from π ( t ) to π ( t +1) . Regarding (ii), recall from Algorithm 8 that K ( t ) denotes the number of iterations undergone by the EMDA subroutine for the update from π ( t ) to π ( t +1) and that K ( t ) is designed to be finite. Therefore, it is easy to verify that π ( t +1) ( a | s ) &gt; 0 for all ( s, a ) by the exponentiated gradient update scheme of EMDA and the strict positivity of π ( t ) .

Next, for ease of exposition, for each k ∈ { 0 , 1 , · · · , K ( t ) } and for each state-action pair ( s, a ) , let ˜ θ ( k ) s,a denote the policy parameter after k EMDA iterations. Regarding (i), recall that we define g ( k ) s,a := ∂ L ( θ ) ∂θ s,a ∣ ∣ θ = ˜ θ ( k ) s and w ( k ) s := ( e -ηg ( k ) s, 1 , · · · , e -ηg ( k ) s, |A| ) . Note that as the weights in the loss function only affects the effective step sizes of EMDA, we simply set the weights of PPO-Clip to be one, without loss of generality. By EMDA in Algorithm 8, for every ( s, a ) ∈ D ( t ) , we have

<!-- formula-not-decoded -->

Note that g ( k ) s,a can be written as

<!-- formula-not-decoded -->

By (149)-(150), it is easy to verify that for those ( s, a ) ∈ D ( t ) with positive advantage, we must have ∏ K ( t ) -1 k =0 exp( -ηg ( k ) s,a ) &gt; 1 . Similarly, for those ( s, a ) ∈ D ( t ) with negative advantage, we have ∏ K ( t ) -1 k =0 exp( -ηg ( k ) s,a ) &lt; 1 . Now we are ready to check the condition of strict policy improvement given by Lemma 11: For each s ∈ S , we have

<!-- formula-not-decoded -->

Note that Lemma 12 implies that the limits V ( ∞ ) ( s ) , Q ( ∞ ) ( s, a ) , A ( ∞ ) ( s, a ) exist, for every sample path: By the strict policy improvement shown in Lemma 12, we know that the sequence of state values is point-wise monotonically increasing, i.e., V ( t +1) ( s ) ≥ V ( t ) ( s ) , ∀ s ∈ S . Moreover, by the bounded reward and the discounted setting, we have -R max 1 -γ ≤ V ( t ) ( s ) ≤ R max 1 -γ . The above monotone increasing property and boundedness imply convergence, i.e., V ( t ) ( s ) → V ( ∞ ) ( s ) , for each sample path. Similarly, we also know that Q ( t ) ( s, a ) → Q ( ∞ ) ( s, a ) , and thus A ( t ) ( s, a ) → A ( ∞ ) ( s, a ) . As a result, we can define the three sets I + s , I 0 s and I -s as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that for each sample path, the sets I + s , I 0 s and I -s are well-defined, based on the limit A ( ∞ ) ( s, a ) .

Lemma 13. Conditioned on the event that each state-action pair occurs infinitely often in {D ( t ) } , if I + s is not an empty set, then we have ∑ a ∈ I -s π ( t ) ( a | s ) → 0 , as t →∞ .

Proof of Lemma 13. We discuss each state separately as it is sufficient to show that for each state s , given some fixed a ′ ∈ I + s , for any a ′′ ∈ I -s , we have π ( t ) ( a ′′ | s ) π ( t ) ( a ′ | s ) → 0 , as t → ∞ . For ease of exposition, we reuse some of the notations from the proof of Lemma 12. Recall that we let K ( t ) denote the number of iterations undergone by the EMDA subroutine for the update from π ( t ) to π ( t +1) , and K ( t ) is designed to be finite. For each k ∈ { 0 , 1 , · · · , K ( t ) } and for each state-action pair ( s, a ) , let ˜ θ ( k ) s,a denote the policy parameter after k EMDA iterations. Recall from Algorithm 8 that g ( k ) s,a := ∂ L ( θ ) ∂θ s,a ∣ ∣ θ = ˜ θ ( k ) s and w ( k ) s := ( e -ηg ( k ) s, 1 , · · · , e -ηg ( k ) s, |A| ) . Define ∆ ∗ := min a ∈ I + s ∪ I -s | A ( ∞ ) ( s, a ) | &gt; 0 (and here ∆ ∗ is a random variable as A ( ∞ ) ( s, a ) is defined with respect to each sample path). By the definition of I + s , I -s and ∆ ∗ , we know that for each sample path, there must exist finite T + and T -such that: (i) for every a ∈ I + s , A ( t ) ( s, a ) ≥ ∆ ∗ 2 , for all t &gt; T + , and (ii) for every a ∈ I -s , A ( t ) ( s, a ) ≤ -∆ ∗ 2 , for all t &gt; T -. Under Assumption 3, at each iteration t with t &gt; max { T + , T -} , there are three possible cases regarding the state-action pairs ( s, a ′ ) and ( s, a ′′ ) :

- Case 1: ( s, a ′ ) ∈ D ( t ) , ( s, a ′′ ) / ∈ D ( t ) By the EMDA subroutine and (149), we have

<!-- formula-not-decoded -->

where the last inequality holds by (150), a ′ ∈ I + s , and π ( t ) ( a ′ | s ) ≤ 1 .

- Case 2: ( s, a ′ ) / ∈ D ( t ) , ( s, a ′′ ) ∈ D ( t )

By the EMDA subroutine, we have -g (0) s,a ′′ &lt; 0 and -g ( k ) s,a ′′ ≤ 0 for all k ∈ { 1 , · · · , K ( t ) } . Therefore, we have

<!-- formula-not-decoded -->

̸

- Case 3: ( s, a ′ ) / ∈ D ( t ) , ( s, a ′′ ) / ∈ D ( t )

̸

Under EMDA, as neither ( s, a ′ ) nor ( s, a ′′ ) is in / ∈ D ( t ) , the action probability ratio between these two actions remains unchanged (despite that the values of π ( t ) ( a ′′ | s ) and π ( t ) ( a ′′ | s ) can still change if there is an action a ′′′ such that a ′′′ = a ′ , a ′′′ = a ′′ , and ( s, a ′′′ ) ∈ D ( t ) ), i.e.,

<!-- formula-not-decoded -->

Conditioned on the event that each state-action pair occurs infinitely often in {D ( t ) } , we know Case 1 and (157) must occur infinitely often. By (155)-(157), we conclude that π ( t ) ( a ′′ | s ) π ( t ) ( a ′ | s ) → 0 , as t →∞ , for every a ′′ ∈ I -s .

Lemma 14. Conditioned on the event that each state-action pair occurs infinitely often in {D ( t ) } , if I + s is not an empty set, then there exists some constant c &gt; 0 such that ∑ a ∈ I -s π ( t ) ( a | s ) ≥ c , for infinitely many t .

Proof of Lemma 14. For each ( s, a ) , define T s,a := { t : ( s, a ) ∈ D ( t ) } to be the index set that collects the time indices at which ( s, a ) is contained in the mini-batch. Given that each state-action pair occurs infinitely often, we know T s,a is a (countably) infinite set.

For ease of exposition, define a positive constant χ as

<!-- formula-not-decoded -->

Define ∆ := min a ∈ I + s A ( ∞ ) ( s, a ) &gt; 0 (and here ∆ is a random variable as A ( ∞ ) ( s, a ) is defined with respect to each sample path). By the definition of I + s and ∆ , we know that there must exist a finite T (+) such that for every a ∈ I + s , A ( t ) ( s, a ) ≥ 3∆ 4 , for all t &gt; T (+) . Similarly, by the definition of I 0 s , there must exist a finite T (0) such that for every a ∈ I 0 s , | A ( t ) ( s, a ) | ≤ χ ∆ 4 , for all t &gt; T (0) . We also define T ∗ := max { T (+) , T (0) } .

We reuse some of the notations from the proof of Lemma 12. Recall that we let K ( t ) denote the number of iterations undergone by the EMDA subroutine for the update from π ( t ) to π ( t +1) , and K ( t ) is a finite positive integer. For ease of exposition, for each k ∈ { 0 , 1 , · · · , K ( t ) } and for each state-action pair ( s, a ) , let ˜ θ ( k ) s,a denote the policy parameter after k EMDA iterations. Recall that we define g ( k ) s,a := ∂ L ( θ ) ∂θ s,a ∣ ∣ θ = ˜ θ ( k ) s and w ( k ) s := ( e -ηg ( k ) s, 1 , · · · , e -ηg ( k ) s, |A| ) . If I + s is not an empty set, then we can select an arbitrary action a ′ ∈ I + s . For any t with t &gt; T (+) and t ∈ T s,a ′ , by (149) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (160) holds due to the fact that ˜ θ ( k ) s,a is non-decreasing with k under Assumption 3 and that K ( t ) ≥ 1 , (161) follows from (150) and that a ′ ∈ I + s , and (162) holds by that the function q ( z ) = z · exp( η/z ) has a unique minimizer at z = η with minimum value e · η . For all t that satisfies ( t -1) ∈ T s,a and t &gt; T ∗ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (163) follows from that ∑ a ∈A π ( t ) ( a | s ) = 0 and A ( t ) ( s, a ) &lt; 0 for all a ∈ I -s , and (164) follows from the definition of T (+) , T (0) as well as the boundedness of rewards. Since T s,a is a countably infinite set, we know ∑ a ∈ I -s π ( t ) ( a | s ) ≥ χ ∆ 4 R max 1 -γ , for infinitely many t .

## E.2 Proof of Theorem 1

Now we are ready to show Theorem 1. For ease of exposition, we restate Theorem 1 as follows.

Theorem (Global Convergence of PPO-Clip) . Under PPO-Clip, we have V ( t ) ( s ) → V π ∗ ( s ) as t → ∞ , ∀ s ∈ S , with probability one.

Proof. We establish that π ( t ) converges to an optimal policy by showing that I + s is an empty set for all s . Under Assumption 2, the analysis below is presumed to be conditioned on the event that each state-action pair occurs infinitely often in {D ( t ) } . The proof proceeds by contradiction as follows: Suppose I + s is non-empty. From Lemma 13, we have that ∑ a ∈ I -s π ( t ) ( a | s ) → 0 , as t →∞ . However, Lemma 14 suggests that there exists some constant c &gt; 0 such that ∑ a ∈ I -s π ( t ) ( a | s ) ≥ c infinitely often. This leads to a contraction, and thus completes the proof.

## F Global Convergence of Tabular PPO-Clip With Alternative Classifiers

Theorem 4. Theorem 1 also holds under the following algorithms: (i) PPO-Clip with the classifier log( π θ ( a | s )) -log( π ( a | s )) (termed PPO-Clip-log); (ii) PPO-Clip with the classifier √ ρ s,a ( θ ) -1 (termed PPO-Clip-root).

Proof of Theorem 4. We show that Theorem 1 can be extended to these two alternative classifiers by following the proof procedure of Theorem 1. Specifically, we extend the supporting lemmas (cf. Lemma 12, Lemma 13, and Lemma 14) as follows:

- To extend Lemma 12 to the alternative classifiers, we can reuse (149) and rewrite (166) for each classifier. That is, for PPO-Clip-log, we have

<!-- formula-not-decoded -->

On the other hand, for PPO-Clip-root, we have

<!-- formula-not-decoded -->

As the sign of g ( k ) s,a depends only on the sign of the advantage, it is easy to verify that (151) still goes through and hence the sufficient condition of Lemma 11 is satisfied under these two alternative classifiers. Moreover, by using the same argument of EMDA as that in Lemma 12, it is easy to verify that π ( t +1) ( a | s ) &gt; 0 for all ( s, a ) .

- Regarding Lemma 13, we can extend this result again by considering the three cases as in Lemma 13. For Case 1, given the g ( k ) s,a in (166) and (167), we have: For PPO-Clip-log,

<!-- formula-not-decoded -->

Similarly, for PPO-Clip-root, we have

<!-- formula-not-decoded -->

Moreover, it is easy to verify that the arguments in Case 2 and Case 3 still hold under these two alternative classifiers. Hence, Lemma 13 still holds.

- Regarding Lemma 14, we can reuse all the setup and slightly revise (159)-(162) for the two alternative classifiers: For PPO-Clip-log, by (166), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, for PPO-Clip-root, by (167), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Accordingly, (163)-(165) still go through and hence Lemma 14 indeed holds under PPO-Clip-log and PPO-Clip-root.

In summary, since all the supporting lemmas hold for these alternative classifiers, we complete this part of the proof by obtaining a contradiction similar to that in Theorem 1.

## G Experiments and Detailed Configuration

## G.1 Experimental Settings

For our experiments, we implement Neural PPO-Clip with different classifiers on the open-source RL baseline3-zoo framework (Raffin 2020). Specifically, we consider four different classifiers as follows: (i) ρ s,a ( θ ) -1 (the standard PPO-Clip classifier); (ii) π θ ( a | s ) -π θ t ( a | s ) (PPO-Clip-sub); (iii) √ ρ s,a ( θ ) -1 (PPO-Clip-root); (iv) log( π θ ( a | s )) -log( π θ t ( a | s )) (PPO-Clip-log). We test these variants in the MinAtar environments (Young and Tian 2019) such as Breakout and Space Invaders. On the other hand, we evaluate them in OpenAI Gym environments (Brockman et al. 2016), which are LunarLander, Acrobot, and CartPole, as well. For the comparison with other benchmark methods, we consider A2C and Rainbow. The training curves are drawn by the averages over 5 random seeds. For the computing resources we use to run the experiment, we use (i) CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz; (ii) GPU: NVIDIA GeForce GTX 1080.

## G.2 Model Parameters

The neural networks architecture of policy and value function in the experiments share two full-connected layers and connect to respective output layers. We provide the parameters of the algorithms for each environment in the following tables 1-4. Notice that lin 5e-4 means that the learning rate decays linearly from 5 × 10 -4 to 0. Also, the vf coef is the weight of the value loss and temperature lambda is the pre-constant of the adaptive temperature parameter for energy-based neural networks. We also give the parameters searching range in table 6.

Table 1: Parameters for MinAtar Breakout experiments.

| Hyperparameters    | PPO-Clip   | PPO-Clip-sub   | PPO-Clip-root   | PPO-Clip-log   | A2C   |
|--------------------|------------|----------------|-----------------|----------------|-------|
| batch size         | 256        | 256            | 256             | 256            | 80    |
| learning rate      | lin 1e-3   | lin 1e-3       | lin 1e-3        | lin 1e-3       | 7e-4  |
| vf coef            | 0.00075    | 0.00075        | 0.00075         | 0.00075        | 0.25  |
| EMDA step size     | 0.005      | 0.005          | 0.005           | 0.005          | -     |
| EMDA iteration     | 2          | 2              | 2               | 2              | -     |
| clipping range     | 0.3        | 0.3            | 0.3             | 0.3            | -     |
| temperature lambda | 25         | 25             | 25              | 25             | -     |

## G.3 Additional Experimental Validation

Ablation study of EMDA iterations. As shown in Algorithm 2, the number of EMDA iteration K is one of the hyperparameters of the algorithm. We conduct ablation studies on it, specifically for K = 2 , 5 , 10 . In the LunarLander environment, their scores are 247, 253, and 237, respectively. This shows empirically that the performance is not sensitive to K .

Empirical comparison between SGD-based PPO and EMDA-based PPO. We report the results under Breakout and 5 seeds. After 5M steps, the conventional PPO has a mean 21.48 with std. dev. 19.41. On the other hand, EMDA-based PPO has a mean 18.24 with std. dev. 3.97. Also in LunarLander, we show that EMDA-based PPO achieves comparable or better performance than conventional PPO in these RL benchmark environments.

Table 2: Parameters for MinAtar Space Invaders experiments.

| Hyperparameters    | PPO-Clip   | PPO-Clip-sub   | PPO-Clip-root   | PPO-Clip-log   | A2C   |
|--------------------|------------|----------------|-----------------|----------------|-------|
| batch size         | 256        | 256            | 256             | 256            | 80    |
| learning rate      | lin 1e-3   | lin 1e-3       | lin 1e-3        | lin 1e-3       | 7e-4  |
| vf coef            | 0.00075    | 0.00075        | 0.00075         | 0.00075        | 0.25  |
| EMDA step size     | 0.005      | 0.005          | 0.005           | 0.005          | -     |
| EMDA iteration     | 5          | 5              | 2               | 5              | -     |
| clipping range     | 0.5        | 0.5            | 0.5             | 0.5            | -     |
| temperature lambda | 10         | 10             | 10              | 10             | -     |

Table 3: Parameters for OpenAI Gym LunarLander-v2 experiments.

| Hyperparameters    | PPO-Clip   | PPO-Clip-sub   | PPO-Clip-root   | PPO-Clip-log   | A2C        |
|--------------------|------------|----------------|-----------------|----------------|------------|
| batch size         | 64         | 8              | 64              | 64             | 40         |
| learning rate      | lin 5e-4   | lin 5e-4       | lin 5e-4        | lin 5e-4       | lin 8.3e-4 |
| vf coef            | 0.75       | 0.75           | 0.75            | 0.75           | 0.5        |
| EMDA step size     | 0.01       | 0.002          | 0.01            | 0.01           | -          |
| EMDA iteration     | 5          | 5              | 5               | 5              | -          |
| clipping range     | 0.3        | 0.5            | 0.3             | 0.3            | -          |
| temperature lambda | 10         | 10             | 10              | 10             | -          |

Table 4: Parameters for OpenAI Gym Acrobot-v1 experiments.

| Hyperparameters    | PPO-Clip   | PPO-Clip-sub   | PPO-Clip-root   | PPO-Clip-log   | A2C        |
|--------------------|------------|----------------|-----------------|----------------|------------|
| batch size         | 64         | 64             | 64              | 64             | 40         |
| learning rate      | lin 7.5e-4 | lin 7.5e-4     | lin 7.5e-4      | lin 7.5e-4     | lin 8.3e-4 |
| vf coef            | 0.5        | 0.5            | 0.5             | 0.5            | 0.5        |
| EMDA step size     | 0.01       | 0.01           | 0.01            | 0.01           | -          |
| EMDA iteration     | 5          | 5              | 5               | 5              | -          |
| clipping range     | 0.3        | 0.3            | 0.3             | 0.3            | -          |
| temperature lambda | 10         | 10             | 10              | 10             | -          |

Table 5: Parameters for OpenAI Gym CartPole-v1 experiments.

| Hyperparameters    | PPO-Clip   | PPO-Clip-sub   | PPO-Clip-root   | PPO-Clip-log   | A2C        |
|--------------------|------------|----------------|-----------------|----------------|------------|
| batch size         | 64         | 64             | 64              | 64             | 40         |
| learning rate      | lin 7.5e-4 | lin 7.5e-4     | lin 7.5e-4      | lin 7.5e-4     | lin 8.3e-4 |
| vf coef            | 0.5        | 0.5            | 0.5             | 0.5            | 0.5        |
| EMDA step size     | 0.01       | 0.01           | 0.01            | 0.01           | -          |
| EMDA iteration     | 5          | 5              | 5               | 5              | -          |
| clipping range     | 0.3        | 0.3            | 0.3             | 0.3            | -          |
| temperature lambda | 10         | 10             | 10              | 10             | -          |

Table 6: Parameters searching range for the experiments.

| Hyperparameters    | Searching Range                            |
|--------------------|--------------------------------------------|
| batch size         | 64, 128, 256                               |
| learning rate      | lin 1e-3, lin 7.5e-4, lin 5e-4, lin 2.5e-4 |
| vf coef            | 0.00075, 0.0005, 0.3, 0.5, 0.75, 0.8       |
| EMDA step size     | 0.001, 0.005, 0.075, 0.02, 0.05, 0.01, 0.1 |
| EMDA iteration     | 2, 5, 10                                   |
| clipping range     | 0.3, 0.5, 0.7                              |
| temperature lambda | 0.1, 0.5, 1, 5, 10, 25, 40, 60, 75         |

Experiments of the generalized objective using different classifiers for SGD-based PPO. Experiments of the generalized objective using different classifiers: We conduct the experiments for the generalized objective under the conventional PPOClip approach. In Breakout with 5 seeds, the mean scores of the root-, log-, and sub-classifiers are 18.08, 12.20, and 17.09, respectively. Also, the standard deviations are 8.83, 0.99, and 7.42, respectively. Moreover, our experiment results show that other classifiers outperform the original objective in some environments, which implies that each of them has its own advantage.

## H Supplementary Related Works

Global Convergence of Policy Gradient Methods. One related line of recent research is on the global convergence of policy gradient methods. (Agarwal et al. 2019, 2020) establishes global convergence results of various policy gradient approaches, including the vanilla policy gradient (with both tabular and softmax policy parametrizations) and the natural policy gradient method (with a softmax policy parametrization). Concurrently, (Bhandari and Russo 2019) provides an alternative analysis of global optimality of the policy gradient method. (Wang et al. 2019) provides the global optimality guarantees for both the vanilla policy gradient and natural policy gradient methods under the overparameterized two-layer neural parameterization. (Mei et al. 2020) establishes the convergence rates of both vanilla softmax policy gradient and the entropy-regularized policy gradient. (Liu et al. 2020) further establishes the global convergence rates of various variance-reduced policy gradient methods. Inspired by the analysis of (Agarwal et al. 2019), we establish the global convergence of the proposed HPO-AM.

Global Convergence of TRPO and PPO. Regarding TRPO, (Shani, Efroni, and Mannor 2020) presents the global convergence rates of an adaptive TRPO, which is established by connecting TRPO and the mirror descent method. (Liu et al. 2019) proves global convergence in expected total reward for a neural variant of PPO with adaptive Kullback-Leibler penalty (PPO-KL). To the best of our knowledge, (Liu et al. 2019) appears to be the only global convergence result for PPO-KL. By contrast, our focus is PPO-clip. Given the salient algorithmic difference between PPO-KL and PPO-clip, there remains no proof of global convergence to an optimal policy for PPO with a clipped objective. In this paper, we rigorously provide the first global convergence guarantee for a variant of PPO-clip.

RL as Classification. Regarding the general idea of casting RL as a classification problem, it has been investigated by the existing literature (Lagoudakis and Parr 2003; Lazaric, Ghavamzadeh, and Munos 2010; Farahmand et al. 2014), which view the one-step greedy update (e.g. in Q-learning) as a binary classification problem. However, a major difference is the labeling: classification-based approximate policy iteration labels the action with the largest Q value as positive; Generalized PPO-Clip labels the actions with positive advantage as positive. Despite the high-level resemblance, our paper is fundamentally different from the prior works (Lagoudakis and Parr 2003; Lazaric, Ghavamzadeh, and Munos 2010; Farahmand et al. 2014) as our paper is meant to study the theoretical foundation of PPO-Clip, from the perspective of hinge loss.

## I Comparison of the Clipped Objective and the Generalized PPO-Clip Objective

Recall that the original objective of PPO-Clip is

<!-- formula-not-decoded -->

where ρ s,a ( θ ) = π θ ( a | s ) π ( a | s ) . In practice, L clip ( θ ) is approximated by the sample average as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that H clip s,a ( θ ) can be further written as

<!-- formula-not-decoded -->

Recall that the generalized objective of PPO-Clip with hinge loss takes the form as

<!-- formula-not-decoded -->

Similarly, H s,a ( θ ) can be further written as

<!-- image -->

Therefore, it is easy to verify that ˆ L clip ( θ ) and -ˆ L ( θ ) only differ by a constant with respect to θ . This also implies that ∇ θ ˆ L clip ( θ ) = -∇ θ ˆ L ( θ ) .