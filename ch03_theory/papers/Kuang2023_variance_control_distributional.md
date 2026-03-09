## Variance Control for Distributional Reinforcement Learning

Qi Kuang * 1 Zhoufan Zhu * 1 Liwen Zhang 1 Fan Zhou 1

## Abstract

Although distributional reinforcement learning (DRL) has been widely examined in the past few years, very few studies investigate the validity of the obtained Q-function estimator in the distributional setting. To fully understand how the approximation errors of the Q-function affect the whole training process, we do some error analysis and theoretically show how to reduce both the bias and the variance of the error terms. With this new understanding, we construct a new estimator Quantiled Expansion Mean (QEM) and introduce a new DRL algorithm (QEMRL) from the statistical perspective. We extensively evaluate our QEMRL algorithm on a variety of Atari and Mujoco benchmark tasks and demonstrate that QEMRL achieves significant improvement over baseline algorithms in terms of sample efficiency and convergence performance.

## 1. Introduction

Distributional Reinforcement Learning (DRL) algorithms have been shown to achieve state-of-art performance in RL benchmark tasks (Bellemare et al., 2017; Dabney et al., 2018b;a; Yang et al., 2019; Zhou et al., 2020; 2021). The core idea of DRL is to estimate the entire distribution of the future return instead of its expectation value, i.e. the Q-function, which captures the intrinsic uncertainty of the whole process in three folds: (i) the stochasticity of rewards, (ii) the indeterminacy of the policy, and (iii) the inherent randomness of transition dynamics. Existing DRL algorithms parameterize the return distribution in different ways, including categorical return atoms (Bellemare et al., 2017), expectiles (Rowland et al., 2019), particles (Nguyen-Tang et al., 2021), and quantiles (Dabney et al., 2018b;a). Among these works, the quantile-based algorithm is widely used

* Equal contribution 1 School of Statistics and Management, Shanghai University of Finance and Economics, Shanghai, China. Correspondence to: Fan Zhou &lt; zhoufan@mail.shufe.edu.cn &gt; .

Proceedings of the 40 th International Conference on Machine Learning , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s).

due to its simplicity, efficiency of training, and flexibility in modeling the return distribution.

Although the existing quantile-based algorithms achieve remarkable empirical success, the approximated distribution still requires further understanding and investigation. One aspect is the crossing issue, namely, a violation of the monotonicity of the obtained quantile estimations. Zhou et al. (2020; 2021) solves this issue by enforcing the monotonicity of the estimated quantiles using some well-designed neural networks. However, these methods may suffer from some underestimation or overestimation issues. In other words, the estimated quantiles tend to be higher or lower than their true values. Considering this shortcoming, Luo et al. (2021) applies monotonic rational-quadratic splines to ensure monotonicity, but their algorithm is computationally expensive and hard to implement in large-scale tasks.

Another aspect is regard to the tail behavior of the return distribution. It is widely acknowledged that the precision of tail estimation highly depends on the frequency of tail observations (Koenker, 2005). Due to data sparsity, the quantile estimation is often unstable at the tails. To alleviate this instability, Kuznetsov et al. (2020) proposes to truncate the right tail of the approximated return distribution by discarding some topmost atoms. However, this approach lacks theoretical support and ignores the potentially useful information hidden in the tail.

The crossing issue and tail unrealization illustrate that there is a substantial gap between the quantile estimation and its true value. This finding reduces the reliability of the Q-function estimator obtained by quantile-based algorithms and inspires us to further minimize the difference between the estimated Q-function and its true value. In particular, the error associated with Q-function approximation can be decomposed into three parts:

<!-- formula-not-decoded -->

where Q π ( · ) is the true Q-function, Q π θ ( · ) is the approximated Q-function, Z π is the random variable with the true return distribution, Z π θ is the random variable with the approximated quantile function parameterized by a set of quantiles θ , D is the replay buffer, and P is the transition kernel. These errors can be attributed to different kinds of approximations in DRL (Rowland et al., 2018), including (i) parameterization and its associated projection operators, (ii) stochastic approximation of the Bellman operator, and (iii) gradient updates through quantile loss.

We elaborate on the properties of the three error terms in (1). E 1 is derived from the target approximation in quantile loss. E 2 is caused by the stochastic approximation of the Bellman operator. E 3 results from the parametrization of quantiles and the corresponding projection operator. Among the three, E 3 can be theoretically eliminated if the representation size is large enough, whereas E 1 + E 2 is inevitable in practice due to the batch-based optimization procedure. Therefore, controlling the variance Var( E 1 + E 2 ) can significantly speed up the training convergence (see an illustrating example in Figure 1). Thus, one main target of this work is to reduce the two inevitable errors E 1 and E 2 , and subsequently improve the existing DRL algorithms.

Figure 1. Error decay during training. (a) The parameterizationinduced error E 3 (grey areas) remains constant over time with a fixed representation size. The approximation errors E 1 and E 2 (blue areas) decrease slowly with time steps. (b) Increase the size of the representation (i.e., the number of quantiles), E 3 can be theoretically eliminated. By applying the variance reduction technique QEM estimator, E 1 + E 2 can be quickly decreased, resulting in faster convergence of algorithms.

<!-- image -->

The contributions of this work are summarized as follows,

- We offer a rigorous investigation on the three error terms E 1 , E 2 , and E 3 in DRL, and find that the approximation errors result from the heteroskedasticity of quantile estimates, especially tail estimates.
- We borrow the idea from the Cornish-Fisher Expansion (Cornish &amp; Fisher, 1938), and propose a statistically robust DRL algorithm, called QEMRL, to reduce the variance of the estimated Q-function.
- We show that QEMRL achieves a higher stability and a faster convergence rate from both theoretical and empirical perspectives.

## 2. Background

## 2.1. Reinforcement Learning

Consider a finite Markov Decision Process (MDP) ( X , A , P, γ, R ) , with a finite set of states X , a finite set of actions A , the transition kernel P : X × A → P ( X ) , the discounted factor γ ∈ [0 , 1) , and the bounded reward function R : X × A → P ([ -R max , R max ]) . At each timestep, an agent observes state X t ∈ X , takes an action A t ∈ A , transfers to the next state X t +1 ∼ P ( · | X t , A t ) , and receives a reward R t ∼ R ( X t , A t ) . The state-action value function Q π : X×A → R of a policy π : X → P ( A ) is the expected discounted sum of rewards starting from x , taking an action a and following a policy π . P ( X ) denotes the set of probability distributions on a space X .

The classic Bellman equation (Bellman, 1966) relates expected return at each state-action pair ( x, a ) to the expected returns at possible next states by:

<!-- formula-not-decoded -->

In the learning task, Q-Learning (Watkins, 1989) employs a common way to obtain π ∗ , which is to find the unique fixed point Q ∗ = Q π ∗ of the Bellman optimality equation:

<!-- formula-not-decoded -->

## 2.2. Distributional Reinforcement Learning

Instead of directly estimating the expectation Q π ( x, a ) , DRL focuses on estimating the distribution of the sum of discounted rewards η π ( x, a ) = D ( ∑ ∞ t =0 γ t R t | X 0 = x, A 0 = a ) to sufficiently capture the intrinsic randomness, where D extract the probability distribution of a random variable. In analogy with Equation (2), η π satisfies the distributional Bellman equation (Bellemare et al., 2017) as follows,

<!-- formula-not-decoded -->

where f γ,r : R → R is defined by f γ,r ( x ) = r + γx, and ( f γ,r ) # η is the pushforward measure of η by f γ,r . Note that η π is the fixed point of distributional Bellman operator T π : P ( R ) X×A → P ( R ) X×A , i.e., T π η π = η π .

In general, the return distribution supports a wide range of possible returns and its shape can be quite complex. Moreover, the transition dynamics are usually unknown in practice, and thus the full computation of the distributional Bellman operator is usually either impossible or computationally infeasible. In the following subsections, we review two main categories of DRL algorithms relying on parametric approximations and projection operators.

## 2.2.1. CATEGORICAL DISTRIBUTIONAL RL

Categorical distributional RL (CDRL, Bellemare et al., 2017) represents the return distribution η with a categorical form η ( x, a ) = ∑ N i =1 p i ( x, a ) δ z i , where δ z denotes the Dirac distribution at z . z 1 ≤ z 2 ≤ . . . ≤ z N are evenly spaced locations, and { p i } N i =1 are the corresponding probabilities learned using the Bellman update,

<!-- formula-not-decoded -->

where Π C : P ( R ) → P ( { z 1 , z 2 . . . z N } ) is a categorical projection operator which ensures the return distribution supported only on { z 1 , . . . , z N } . In practice, CDRL with N = 51 has been shown to achieve significant improvement in Atari games.

## 2.2.2. QUANTILED DISTRIBUTIONAL RL

Quantiled distributional RL (QDRL, Dabney et al., 2018b) represents the return distribution with a mixture of Diracs η ( x, a ) = 1 N ∑ N i =1 δ θ i ( x,a ) , where { θ i ( x, a ) } N i =1 are learnable parameters. The Bellman operator moves each atom location θ i towards τ i -th quantile of the target distribution η ′ ( x, a ) := T π η ( x, a ) , where τ i = 2 i -1 2 N . The corresponding Bellman update form is:

<!-- formula-not-decoded -->

where Π W 1 : P ( R ) → P ( R ) is a quantile projection operator defined by Π W 1 µ = 1 N ∑ N i =1 δ F -1 µ ( τ i ) , and F µ is the cumulative distribution function (CDF) of µ . F -1 η ′ ( τ ) can be characterized as the minimizer of the quantile regression loss, while the atom locations θ can be updated by minimizing the following loss function

<!-- formula-not-decoded -->

## 3. Error Analysis of Distributional RL

As mentioned in Section 1, the parametrization induced error E 3 in Equation (1) comes from quantile representation and its projection operator, which can be eliminated as N → ∞ . However, as illustrated in Figure 1, the approximation errors E 1 and E 2 are unavoidable in practice and a high variance Var( E 1 + E 2 ) may lead to unstable performance of DRL algorithms. Thus, in this section, we further study the three error terms E 1 , E 2 and E 3 , and show why it is important to control them in practice.

## 3.1. Parametrization Induced Error

We first show the convergence of both the expectation and the variance of the distributional Bellman operator T π . Then, we take parametric representation and projection operator into consideration.

Proposition 3.1 (Sobel, 1982; Bellemare et al., 2017) . Suppose there are two value distributions ν 1 , ν 2 ∈ P ( R ) , and random variables Z k +1 i ∼ T π ν i , Z k i ∼ ν i . Then, we have

<!-- formula-not-decoded -->

Based on the fact that T π is a γ -contraction in ¯ d p metric (Bellemare et al., 2017), where ¯ d p is the maximal form of the Wasserstein metric, Proposition 3.1 implies that T π is a contraction for both the expectation and the variance. The two converge exponentially to their true values by iteratively applying the distributional Bellman operator.

However, in practice, employing parametric representation for the return distribution leaves a theory-practice gap, which makes neither the expectation nor the variance converge to the true values. To better understand the bias in the Q-function approximation caused by the parametric representation, we introduce the concept of mean-preserving 1 to describe the relationship between the expectations of the original distribution and the projected distribution:

Definition 3.2 ( Mean-preserving ) . Let Π F : P ( R ) → F be a projection operator that maps the space of probability distributions to the desired representation. Suppose there is a representation F ∈ P ( R ) and its associated projection operator Π F are mean-preserving if for any distribution ν ∈ F , the expectation of Π F ν is the same as that of ν .

For CDRL, a discussion of the mean-preserving property is given by Lyle et al. (2019) and Rowland et al. (2019). It can be shown that for any ν ∈ F C , where F C is a N -categorical representation, the projection Π C preserves the distribution's expectation when its support is contained in the interval [ z 1 , z N ] . However, these practitioners usually employ a wide predefined interval for return which makes the projection operator typically overestimate the variance.

For QDRL, Π W 1 is not mean-preserving (Bellemare et al., 2023). Given any distribution ν ∈ F W 1 , where F W 1 is a N -quantile representation, there is no unique N -quantile distribution Π W 1 ν in most cases, as the projection operator Π W 1 is not a non-expansion in 1-Wasserstein distance (See Appendix B for details). This means that the expectation, variance, and higher-order moments are not preserved. To make this concrete, a simple MDP example is used to illustrate the bias in the learned quantile estimates.

In Figure 2 (a), rewards R 1 and R 2 are randomly sampled from Unif(0 , 1) and Unif(1 /N, 1 + 1 /N ) at states x 1 and x 2 respectively, and no rewards are received at x 0 . Clearly, the true return distribution at state x 0 is the mix-

1 This property has been thoroughly discussed in previous work. Based on Section 5.11 of Bellemare et al. (2023), we conclude this definition.

1.0

0.-0.2

R1 ~ Unif (0,1)

X1

50%

1.2

X1

X3

X2

X4

ture γ 2 ( R 1 + R 2 ) , hence the 1 2 N -th quantile is γ N . When using the QDRL algorithm with N quantile estimates, the approximated return distribution ˆ η ( x 1 , a ) = 1 N ∑ N i =1 δ 2 i -1 2 N and ˆ η ( x 2 , a ) = 1 N ∑ N i =1 δ 2 i +1 2 N . In this case, the 1 2 N -th quantile of the approximated return distribution at state x 0 is 3 γ 2 N , whereas the true value is γ N . Moreover, for each i = 1 , . . . , N , the 2 i -1 2 N -th quantile estimate at state x 0 is not equal to the true value. QEMRL mean

Figure 2. (a) Example MDP, with a single action, equal transition probability, an initial state x 0 , and two terminal states x 1 , x 2 where rewards are drawn from uniform. (b) 5-state MDP, with two actions at initial state x 0 , deterministic transition, and stochastic rewards are exponential at terminal states x 3 , x 4 . (c) We show the true return distributions η ( x 0 , a 1 ) and η ( x 0 , a 2 ) , and the expected returns estimated by QDRL and QEMRL.

<!-- image -->

These biased quantile estimates illustrated in Figure 2 (a) are caused by the use of quantile representation and its projection operator Π W 1 . This undesirable property in turn affects the QDRL update, as the combined operator Π W 1 T π is in general not a non-expansion in ¯ d p , for p ∈ [1 , ∞ ) (Dabney et al., 2018b), which means that the learned quantile estimates may not converge to the true quantiles of the return distribution 2 . The projection operator Π W 1 is not meanpreserving which inevitably leads to bias in the expectation of return distribution when iteratively applying the projected Bellman operator Π W 1 T π during the training process, resulting in a deviation between the estimate and the true value of the Q-function in the end. We now derive an upper bound to quantify this deviation, i.e. E 3 .

Theorem 3.3 ( Parameterization induced error bound ) . Let Π W 1 be a projection operator onto evenly spaced quantiles τ i 's where each τ i = 2 i -1 2 N for i = 1 , . . . , N , and

2 A recent study (Rowland et al., 2023a) proves that QDRL update may have multiple fixed points, indicating quantiles may not converge to the truth. Despite this, Proposition 2 (Dabney et al., 2018b) concludes that the projected Bellman operator Π W 1 T π remains a contraction in ¯ d ∞ . This implies that quantile convergence is guaranteed for all p ∈ [1 , ∞ ] .

R2 ~ Unif (. 1 + N

50%

Xo a1

a2

η k ∈ P ( R ) be the return distribution of k -th iteration. Let random variables Z k θ ∼ Π W 1 T π η k and Z k ∼ T π η k . Assume that the distribution of the immediate reward is supported on [ -R max , R max ] , then we have

<!-- formula-not-decoded -->

where E k 3 is parametrization induced error at k -th iteration.

Theorem 3.3 implies that the convergence of expectation with projected Bellman operator Π W 1 T π cannot be guaranteed after quantile representation and its projection operator are applied 3 . Note that the bound will tend to zero with N → ∞ , thus it is reasonable to use a relatively large representation size N to reduct E 3 in practice.

## 3.2. Approximation Error

The other two types of errors E 1 and E 2 , which determine the variance of the Q-function estimate, are accumulated during the training process by keeping encountering unseen state-action pairs. The target approximation error E 1 affects action selections, while the Bellman operator approximation error E 2 leads to the accumulated error of the Q-function estimate, which can be amplified by using the temporal difference updates (Sutton, 1988). The accumulated errors of the Q-function estimate with high uncertainty can make some certain states to be incorrectly estimated, leading to suboptimal policies and potentially divergent behaviors.

As depicted in Figure 2 (b), we utilize this toy example to illustrate how QDRL fails to learn an optimal policy due to a high variance of the approximation error. This 5-state MDP example is originally introduced in Figure 7 of Rowland et al. (2019). In this case, η ( x 0 , a 1 ) and η ( x 0 , a 2 ) follow exponential distributions, and the expectations of them are 1.2 and 1, respectively. We consider a tabular setting, which uniquely represents the approximated return distribution at each state-action pair. Figure 2 (c) demonstrates that in policy evaluation, QDRL inaccurately approximates the Qfunction, as it underestimates the expectation of η ( x 0 , a 1 ) and overestimates the other. This is caused by the poor capture of tail events, which results in high uncertainty in the Q-function estimate. Due to the high variance, QDRL fails to learn the optimal policy and chooses a non-optimal action a 2 at the initial state x 0 . On the contrary, our proposed algorithm, QEMRL, employs a statistically robust estimator of the Q-function to reduce its variance, relieves the underestimation and overestimation issues, and ultimately allows

3 Note that this bound has a limitation, which only considers the one-step effect of applying the projection operator Π W 1 . Therefore, it becomes irrelevant with the iteration number k . However, Proposition 4.1 of Rowland et al. (2023b) provides a more compelling bound considering the cumulative effect of iteratively applying Π W 1 .

for more efficient policy learning.

Different from previous QDRL studies that focus on exploiting the distribution information to further improve the model performance, this work highlights the importance of controlling the variance of the approximation error to obtain a more accurate estimate of the Q-function. More discussion about this is given in the following section.

## 4. Quantiled Expansion Mean

This section introduces a novel variance reduction technique to estimate the Q-function. In traditional statistics, estimators with lower variance are considered to be more efficient. In RL, variance reduction is also an effective technique for achieving fast convergence in both policy-based and value-based RL algorithms, especially for large-scale tasks (Greensmith et al., 2004; Anschel et al., 2017). Motivated by these findings, we introduce QEM as an estimator that is more robust and has a lower variance than that of QDRL under the heteroskedasticity assumption. Furthermore, we demonstrate the potential benefits of QEM for the distribution approximation in DRL.

## 4.1. Heteroskedasticity of quantiles

In the context of quantile-based DRL, Q-function is the integral of the quantiles. To approximate this, QDRL employs a simple empirical mean (EM) estimator 1 N Σ i ˆ q ( τ i ) , and it is natural to assume that the estimated quantile satisfies

<!-- formula-not-decoded -->

where ε ( τ ) is a zero-mean error. In this case, considering the crossing issue and the biased tail estimates, we assume that the variance of ε ( τ ) is non-constant and depends on τ , which is usually called heteroskedasticity in statistics.

For a direct understanding, we conduct a simple simulation using a Chain MDP to illustrate how QDRL can fail to fit the quantile function. As shown in Figure 3(b), QDRL fits well in the peak area but struggles at the bottom and the tail. Moreover, the non-monotonicity of the quantile estimates in the poorly fitted areas is more severe than the others. As the deviations of the quantile estimates from the truths is significantly larger in the low probability region and the tail, we can make the heteroskedasticity assumption in this case. This phenomenon can be explained since samples near the bottom and the tail are less likely to be drawn. In real-world situations, multimodal distributions are commonly encountered and the heteroskedasticity problem may result in imprecise distribution approximations and consequently poor Q-function approximations. In the next part, we will discuss how to enhance the stability of the Q-function estimate.

## 4.2. Cornish-Fisher Expansion

It is well-known that quantile can be expressed by the Cornish-Fisher Expansion (CFE, Cornish &amp; Fisher, 1938):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where z τ is the τ -th quantile of the standard normal distribution, µ is the mean, σ is the standard deviation, s and k are the skewness and kurtosis of the interested distribution, and the remaining terms in the ellipsis are higher-order moments (See Appendix C for more details). The CFE theoretically determines the distribution with known moments and is widely used in financial studies. Recently, Zhang &amp; Zhu (2023) employ CFE to estimate higher-order moments of financial time series data, which are not directly observable. Our method utilizes a truncated version of CFE framework and employs a linear regression model to construct efficient estimators for distribution moments based on known quantiles. Consequently, we apply this approach within the context of quantile-based DRL.

To be more specific, we plug in the estimate ˆ q ( τ ) of the the τ -th quantile to Equation (5) and expand it by the first order:

<!-- formula-not-decoded -->

where m 1 is the mean (say, 1 -th moment) of the return distribution, i.e., the Q-function, and ω 1 ( τ ) is the remaining term associated with the higher-order ( &gt; 1 -th) moments. If ω 1 ( τ ) is negligible, m 1 can be estimated by averaging the N quantile estimates in QDRL.

When the estimated quantile is expanded to the second order, we particularly have the following representation:

<!-- formula-not-decoded -->

where ω 2 ( τ ) is the remaining term associated with the higher-order ( &gt; 2 -th) moments. Assume that ω 2 ( τ ) is negligible, we can derive a regression model by plugging in the N quantile estimates, such that

<!-- formula-not-decoded -->

The higher-order expansions can be conducted in the same manner. Note that the remaining term is omitted for constructing a regression model, and a more in-depth analysis of the remaining term is available in Appendix C.2.

For notation simplicity, we rewrite (8) in a matrix form,

<!-- formula-not-decoded -->

X3

X4)

Х5

R

(a)

returns

2

where ˆ Q ∈ R N is the vector of estimated quantiles, X 2 ∈ R N × 2 and M 2 ∈ R 2 are the design matrix and the moments respectively, and E is the vector of error terms. -2

estimated quantile

Figure 3. (a) Chain MDP, with six states, one action, γ = 0 . 99 and gaussian mixture reward distribution at terminal state x 5 . (b) True quantile function (top) and QDRL quantile function at state x 0 after 10K steps iterate. Scatter diagram (bottom) of approximated quantile from training process.

<!-- image -->

For this bivariate regression model (9), the traditional ordinary least squares method (OLS) can be used to estimate M 2 = ( m 1 , √ m 2 ) ′ when the variances of the errors are invariant across different quantile locations, also known as the homoscedasticity assumption. The estimator ˆ m 1 is denoted as Quantiled Expansion Mean (QEM) in this work. However, since the homoscedasticity assumption required by OLS is always violated in real cases, we may consider using the weighted ordinary least squares method (WLS) instead. Under the normality assumption, the following results tell that the WLS estimator ˆ m 1 has a lower variance than the direct empirical mean.

Lemma 4.1. Consider the linear regression model ˆ Q = X 2 M 2 + E , E is distributed on N ( 0 , σ 2 V ) , where V = diag ( v 1 , v 2 , · · · , v N ) , v i ≥ 1 , i = 1 , · · · , N , and we set noise variance σ 2 = 1 without loss of generality. The WLS estimator is

<!-- formula-not-decoded -->

and the QEM estimator ˆ m 1 is the first component of ̂ M 2 .

Remark: Note that it is impossible to determine the weight matrix V for each state-action pair in practice. Hence, we focus on capturing the relatively high variance in the tail, specifically in the range of τ ∈ (0 , 0 . 1] ∪ [0 . 9 , 1) . To achieve this, we use a constant v i , which is set to a value greater than 1 in the tail and equal to 1 in the rest. v i is treated as a hyperparameter to be tuned in practice (See Appendix E).

Ground truth probability

With Lemma 4.1, the reduction of variance can be guaranteed by the following Proposition 4.2. Throughout the training process, heteroskedasticity is inevitable, and thus the QEM estimator always exhibits a lower variance than the standard EM estimator ˆ m ∗ 1 = 1 N ∑ N i =1 ˆ q ( τ i ) .

Proposition 4.2. Suppose the noise ε i independently follows N (0 , v i ) where v i ≥ 1 for i = 1 , · · · , N , then,

(i) In the homoskedastic case where v i = 1 for i = 1 , . . . N , the empirical mean estimator ˆ m ∗ 1 has a lower variance, Var( ˆ m ∗ 1 ) &lt; Var( ˆ m 1 ) ;

(ii) In the heteroskedastic case where v i 's are not eaqul, the QEM estimator ˆ m 1 achieves a lower variance, i.e. Var( ˆ m 1 ) &lt; Var( ˆ m ∗ 1 ) , if and only if ¯ v 2 -1 -1 / ( ( ∑ i v i ∑ i v i z 2 τ i ) ( ∑ i v i z τ i ) 2 -1) &gt; 0 , where ¯ v = 1 N ∑ i v i . This inequality holds when z τ i = -z τ N -i , which can be guaranteed in QDRL.

We also try to explore the potential benefits of the variance reduction technique QEM in improving the approximation accuracy. The Q-function estimate with higher variance can lead to noisy policy gradients in policy-based algorithms (Fujimoto et al., 2018) and prevent selection optimal actions in value-based algorithms (Anschel et al., 2017). These issues can slow down the learning process and negatively impact the algorithm performance. By the following theorem, we are able to show that QEM can reduce the variance and thus improve the approximation performance.

Theorem 4.3. Consider the policy ˆ π that is learned policy, and denote the optimal policy to be π opt , α = max x ′ D TV (ˆ π ( · | x ′ ) ∥ π opt ( · | x ′ )) , and n ( x, a ) = |D| . For all δ ∈ R , with probability at least 1 -δ , for any η ( x, a ) ∈ P ( R ) , and all ( x, a ) ∈ D ,

<!-- formula-not-decoded -->

Theorem 4.3 indicates that a lower concentration bound can be obtained with a smaller α value. The decrease in α can be attributed to the benefits of QEM. Specifically, QEM helps to decrease the perturbations on the Q-function and reduce the variance of the policy gradients, which allows for faster convergence of the policy training and a more accurate distribution approximation. To conclude, QEM relieves the error accumulation within the Q-function update, improves the estimation accuracy, reduces the risk of underestimation and overestimation, and thus ultimately enhances the stability of the whole training process.

## 5. Experimental Results

In this section, we do some empirical studies to demonstrate the advantage of our QEMRL method. First, a simple tabular experiment is conducted to validate some of the theo-

moves

0.14.

÷ 0.12 1

з 0.10 -

₫ 0.08 1

0.06-

•0, 0.02 -

0.00

1

1

→

1

G

1.0

0.8

0.6

0.4

0.2

0.0

ODRL

Ground \_truth

## Algorithm 1 QEMRL update algorithm

- 1: Require: Quantile estimates ˆ q i ( x, a ) for each ( x, a )
- 2: Collect sample ( x, a, r, x ′ )

QEMRL

QORL

- 3: # Compute distributional Bellman target
- 4: Compute Q ( x ′ , a ) using Equation (10)
- 5: if policy evaluation then
- 6: a ∗ ∼ π ( ·| x ′ ) 100k

50k

100k

- 7: else if Q-Learning then
- 8: a ∗ ← arg max a Q ( x ′ , a )
- 9: end if
- 10: Scale samples ˆ q ∗ i ( x ′ , a ∗ ) ← r + γ ˆ q i ( x ′ , a ∗ ) , ∀ i .
- 11: # Compute quantile loss
- 12: Update estimated quantiles ˆ q i ( x, a ) by computing the gradients for each i = 1 , . . . , N , ∇ ˆ q i ( x,a ) ∑ N i =1 L QR (ˆ q i ( x, a ); 1 N ∑ N j =1 δ ˆ q ∗ j ( x ′ ,a ∗ ) , τ i ) .

retical results presented in Sections 3 and 4. Then we apply the proposed QEMRL update strategy in Algorithm 1 to both the DQN-style and SAC-style DRL algorithms, which are evaluated on the Atari and MuJoCo environments. The detailed architectures of these methods and the hyperparameter selections can be found in Appendix D, and the additional experimental results are included in Appendix E.

In this work, we implement QEM using a 4 -th order expansion that includes mean, variance, skewness, and kurtosis in this work. The effects of a higher-order expansion on model estimation are discussed in Appendix C.1. Intuitively, including more terms in the expansion improves the estimation accuracy of quantiles, but the overfitting risk and the computational cost are also increased. Hence, there is a trade-off between explainability and learning efficiency. We evaluate different expansion orders using the R 2 statistic, which measures the goodness of model fitting. The simulation results (Figure 9) show that a 4 -th order expansion seems to be the optimal choice while a higher-order ( &gt; 4 -th) expansion does not show a significant increase in R 2 .

## 5.1. A Tabular Example

FrozenLake (Brockman et al., 2016) is a classic benchmark problem for Q-learning control with high stochasticity and sparse rewards, in which an agent controls the movement of a character in an n × n grid world. As shown in Figure 4 with a FrozenLake4 × 4 task, 'S' is the starting point, 'H' is the hole that terminates the game, 'G' is the goal state with a reward of 1. All the blue grids stand for the frozen surface where the agent can slide to adjacent grids based on some underlying unknown probabilities when taking a certain movement direction. The reward received by the agent is always zero unless the goal state is reached.

We first approximate the return distribution under the optimal policy π ∗ , which can be realized using the value itera-

150k

50k

1

Quantile curve tion approach. To be specific, we start from the 'S' state and perform 1K Monte-Carlo (MC) rollouts. An empirical distribution can be obtained by summarizing all these recording trajectories. With the approximation of the distribution, we can draw a curve of quantile estimates shown in Figure 5. Both QEMRL and QDRL were run for 150K training steps and the ϵ -greedy exploration strategy is applied in the first 1K steps. For both methods, we set the total number of quantiles to be N = 128 .

Figure 4. (a) The optimal direction of movement at each grid. (b) Quantile estimates by MC, QDRL, and QEMRL at the start state. (c) Approximation errors of Q-function estimate and distribution approximation error of QEMRL and QDRL (results are averaged over 10 random seeds).

<!-- image -->

Although both QEMRL and QDRL can eventually find the optimal movement at the start state, their approximations of the return distribution are quite different. Figure 4 (b) visualizes the approximation errors of the Q-function and the distribution for QEMRL and QDRL with respect to the number of training steps. The Q-function estimates of QEMRL converge correctly in average, whereas the estimates of QDRL do not converge exactly to the truth. A similar pattern can also be found when it comes to the distribution approximation error. Besides, the reduction of variance by using QEM can be verified by the fact that the curves of QEMRL are more stable and decline faster. In Figure 4 (c), we show that the distribution at the start state estimated by QEMRL is eventually closer to the ground truth.

## 5.2. Evaluation on MuJoCo and Atari 2600

We do some experiments using the MuJoCo benchmark to further verify the analysis results in Section 4. Our implementation is based on the Distributional Soft Actor-Critic (DSAC, Ma et al., 2020) algorithm, which is a distributional version of SAC. Figure 5 demonstrate that both DSAC and QEM-DSAC significantly outperform the baseline SAC. Among the two, QEM-DSAC performs better than DSAC and the learning curves are more stable, which demonstrates

7000

8000

5000

6000

1000

that QEM-DSAC can achieve a higher sample efficiency.

Average Return

2000

6000

Average Return

5000

3000

2000

1000

Average

Average Return

Soco

Average Return

Average Return

5000

1000

4000)

3000

8000

2000

1000

1000

6000

2000

Average Return

25000

20000

15000

5000

Average Return

350000

300000

250000-

150000

200000

150000

12000

100000|

50000

Average Return

50000

10000

8000|

6000

Jamesbond

Jamesbond hopper

100

150

Millions of frames

Millions of frames

DEM OSAG

100

150

Millions of frames

Millions of frames

Asterix

Asterix halfcheetah

100

150

50

150

Millions of frames

25000

10000

= 20000 -

6000

Retu

15000|

4000|

10000

5000

Average Return

3000|

Average

2000

100

Millions of frames

## that QEM-DSAC can achieve a higher sample efficiency.

: 25000

25000|

20000|

20000

<!-- image -->

Time Steps humanoid

Millions of frames

Hero

Millions of frames

0.2

Time Steps

Millions of frames

Figure 5. Learning curves of SAC, DSAC, and QEM-DSAC across six MuJoCo games. Each curve is averaged over 5 random seeds and shaded by their confidence intervals.

Wealso do some comparison between QEM and the baseline method QR-DQN on the Atari 2600 platform. Figure 8 plots the final results of these two algorithms in six Atari games. At the early training stage, QEM-DQN exhibits significant gain in sampling efficiency, resulting in faster convergence and better performance.

Extension to IQN. Some great efforts have been made by the community of DRL to more precisely parameterize the entire distribution with a limited number of quantile locations. One notable example is the introduction of Implicit Quantile Networks (IQN, Dabney et al., 2018a), which tries to recover the continuous map of the entire quantile curve by sampling a different set of quantile values from a uniform distribution Unif(0 , 1) each time.

Our method can also be applied to IQN as it uses the EM approach to estimate the Q-function. It is noted that the design matrix X must be updated after re-sampling all the quantile fractions at each training step. Moreover, one important sufficient condition z τ i = -z τ N -i which ensures the reduction of variance does not hold in the IQN case as τ 's are sampled from a uniform distribution. However, according to the simulation results in Table 4, the variance reduction still remains valid in practice. In this case, all the baseline methods are modified to the IQN version. As Figure 6 and Figure 7 demonstrate, QEM can achieve some performance gain in most scenarios and the convergence speeds can be slightly increased.

## 5.3. Exploration

Since QEM also provides an estimate of the variance, we may consider using it to develop an efficient exploration strategy. In some recent study studies, to more sufficiently utilize the distribution information, Mavrin et al. (2019) proposes a novel exploration strategy, Decaying Left Truncated Variance (DLTV) by using the left truncated variance of the estimated distribution as a bonus term to encourage exploration in unknown states. The op-

Return

Return

100000|

Average Return

0.21

Alien

Alien ant

Time Steps

200

150000

Time Steps

200

6000

5000

ant

Time Steps humanoid

Time Steps

Figure 6. Learning curves of SAC, DSAC (IQN), and QEM-DSAC

(IQN) across six MuJoCo games. Each curve is averaged over 5

condom code ond chodad he thair confidands intomola hopper

Time Steps halfcheetah

OCK-OSACION

0.2

Time Steps

<!-- image -->

Figure 6. Learning curves of SAC, DSAC (IQN), and QEM-DSAC (IQN) across six MuJoCo games. Each curve is averaged over 5 random seeds and shaded by their confidence intervals.

Figure 7. Learning curves of IQN and IQEM-DQN across six Atari games. Each curve is averaged over 3 random seeds and shaded by their confidence intervals.

<!-- image -->

Figure 8. Learning curves (top and middle) of QR-DQN and QEMDQN across six Atari games. Learning curves (bottom) of QRDQN and QEM-DQN with exploration across three games.

<!-- image -->

timal action a ∗ at state x is selected according to a ∗ = arg max a ′ ( Q ( x, a ′ ) + c t √ σ 2 + ) , where c t is a decay factor to suppress the intrinsic uncertainty, and σ 2 + denotes the estimation of variance. Although DLTV is effective, the validity of the computed truncation lacks a theoretical guarantee. In this work, we follow the idea of DLTV and examine the model performance by using either the variance estimate obtained by QEM or the original DLTV estimation

0.8

6000|

Average Return

3000

in some hard-explored games. As Figure 8 shows, by using QEM, the exploration efficiency is significantly improved compared to QR-DQN+DLTV since QEM enhances the accuracy of the quantile estimates and thus the accuracy of the distribution variance.

## 6. Conclusion and Discussion

In this work, we systematically study the three error terms associated with the Q-function estimate and propose a novel DRL algorithm QEMRL, which can be applied to any quantile-based DRL algorithm regardless of whether the quantile locations are fixed or not. We found that a more robust estimate of the Q-function can improve the distribution approximation and speed up the algorithm convergence. We can also utilize the more precise estimate of the distribution variance to optimize the existing exploration strategy.

Finally, there are some open questions we would like to have further discussions here.

Improving the estimation of weight matrix V . The challenge of estimating the weight matrix V was recognized from the outset of the method proposal since it is unlikely to know the exact value of V in practice. In this work, we treat V as a predefined value that can be tuned, taking into account the computational cost of estimating it across all state-action pairs and time steps. As for future work, we believe a robust and easy-to-implement estimation of weight matrix V is necessary. Given that the variance of quantile estimation errors varies with state-action pairs and algorithm iterations, we consider two approaches for future investigation. The first approach considers a decay value of v i instead of the constant. It is worth noting that the variance of poorly estimated quantiles tends to decrease gradually as the number of training samples increases, which motivates us to decrease the value of v i as training epochs increase. The second approach involves assigning different values of v i to different state-action pairs. Ideas from the exploration field, specifically the count-based method (Ostrovski et al., 2017), can be borrowed to measure the novelty of stateaction pairs. Accordingly, for familiar state-action pairs, a smaller value of v i should be assigned, while unfamiliar pairs should be assigned a larger value of v i .

Statistical variance reduction. Our variance reduction method is based on a statistical modeling perspective, and the core insight of our method is that performance might be improved through more careful use of the quantiles to construct a Q-function estimator. While alternative ensembling methods can be directly applied to DRL to reduce the uncertainty in Q-function estimator, commonly used in existing works (Osband et al., 2016; Anschel et al., 2017), it undoubtedly increases model complexity. In this work, we transform the Q value estimation into a linear regres- sion problem, where the Q value is the coefficient of the regression model. In this way, we can leverage the weighted least squares (WLS) method to effectively capture the heteroscedasticity of quantiles and obtain a more efficient and robust Q-function estimator.

## Acknowledgements

We thank anonymous reviewers for valuable and constructive feedback on an early version of this manuscript. This work is supported by National Social Science Foundation of China (Grant No.22BTJ031 ) and Postgraduate Innovation Foundation of SUFE. Dr. Fan Zhou's work is supported by National Natural Science Foundation of China (12001356), Shanghai Sailing Program (20YF1412300), 'Chenguang Program' supported by Shanghai Education Development Foundation and Shanghai Municipal Education Commission, Open Research Projects of Zhejiang Lab (NO.2022RC0AB06), Shanghai Research Center for Data Science and Decision Technology, Innovative Research Team of Shanghai University of Finance and Economics.

## References

Anschel, O., Baram, N., and Shimkin, N. Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning. In International Conference on Machine Learning , pp. 176-185. PMLR, 2017.

Bellemare, M. G., Dabney, W., and Munos, R. A distributional perspective on reinforcement learning. In International Conference on Machine Learning , pp. 449-458. PMLR, 2017.

Bellemare, M. G., Dabney, W., and Rowland, M. Distributional Reinforcement Learning . MIT Press, 2023. http://www.distributional-rl.org .

Bellman, R. Dynamic programming. Science , 153(3731): 34-37, 1966.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym. arXiv preprint arXiv:1606.01540 , 2016.

Cornish, E. A. and Fisher, R. A. Moments and cumulants in the specification of distributions. Revue de l'Institut international de Statistique , pp. 307-320, 1938.

Dabney, W., Ostrovski, G., Silver, D., and Munos, R. Implicit quantile networks for distributional reinforcement learning. In International Conference on Machine Learning , pp. 1096-1105, 2018a.

Dabney, W., Rowland, M., Bellemare, M. G., and Munos, R. Distributional reinforcement learning with quantile

- regression. In Proceedings of the AAAI Conference on Artificial Intelligence , 2018b.
- Fujimoto, S., Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning , pp. 1587-1596. PMLR, 2018.
- Greensmith, E., Bartlett, P. L., and Baxter, J. Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research , 5(9), 2004.
- Hsu, D., Kakade, S. M., and Zhang, T. An analysis of random design linear regression. arXiv preprint arXiv:1106.2363 , 2011.
- Koenker. Quantile regression . Cambridge University Press, 2005.
- Kuznetsov, A., Shvechikov, P., Grishin, A., and Vetrov, D. Controlling overestimation bias with truncated mixture of continuous distributional quantile critics. In International Conference on Machine Learning , pp. 5556-5566. PMLR, 2020.
- Luo, Y., Liu, G., Duan, H., Schulte, O., and Poupart, P. Distributional reinforcement learning with monotonic splines. In International Conference on Learning Representations , 2021.
- Lyle, C., Bellemare, M. G., and Castro, P. S. A comparative analysis of expected and distributional reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pp. 4504-4511, 2019.
- Ma, X., Xia, L., Zhou, Z., Yang, J., and Zhao, Q. Dsac: distributional soft actor critic for risk-sensitive reinforcement learning. arXiv preprint arXiv:2004.14547 , 2020.
- Mavrin, B., Yao, H., Kong, L., Wu, K., and Yu, Y. Distributional reinforcement learning for efficient exploration. In International Conference on Machine Learning , pp. 4424-4434, 2019.
- Nguyen-Tang, T., Gupta, S., and Venkatesh, S. Distributional reinforcement learning via moment matching. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pp. 9144-9152, 2021.
- Osband, I., Blundell, C., Pritzel, A., and Van Roy, B. Deep exploration via bootstrapped dqn. Advances in Neural Information Processing Systems , 29, 2016.
- Ostrovski, G., Bellemare, M. G., Oord, A., and Munos, R. Count-based exploration with neural density models. In International Conference on Machine Learning , pp. 2721-2730. PMLR, 2017.
- Rowland, M., Bellemare, M., Dabney, W., Munos, R., and Teh, Y. W. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pp. 29-37. PMLR, 2018.
- Rowland, M., Dadashi, R., Kumar, S., Munos, R., Bellemare, M. G., and Dabney, W. Statistics and samples in distributional reinforcement learning. In International Conference on Machine Learning , pp. 5528-5536. PMLR, 2019.
- Rowland, M., Munos, R., Azar, M. G., Tang, Y., Ostrovski, G., Harutyunyan, A., Tuyls, K., Bellemare, M. G., and Dabney, W. An analysis of quantile temporal-difference learning. arXiv preprint arXiv:2301.04462 , 2023a.
- Rowland, M., Tang, Y., Lyle, C., Munos, R., Bellemare, M. G., and Dabney, W. The statistical benefits of quantile temporal-difference learning for value estimation. arXiv preprint arXiv:2305.18388 , 2023b.
- Sobel, M. J. The variance of discounted markov decision processes. Journal of Applied Probability , 19(4):794-802, 1982.
- Sutton, R. S. Learning to predict by the methods of temporal differences. Machine Learning , 3:9-44, 1988.
- Villani, C. Optimal transport: old and new , volume 338. Springer, 2009.
- Watkins, C. J. C. H. Learning from delayed rewards. PhD thesis , 1989.
- Yang, D., Zhao, L., Lin, Z., Qin, T., Bian, J., and Liu, T.-Y . Fully parameterized quantile function for distributional reinforcement learning. In Advances in Neural Information Processing Systems , pp. 6193-6202, 2019.
- Zhang, N. and Zhu, K. Quantiled conditional variance, skewness, and kurtosis by cornish-fisher expansion. arXiv preprint arXiv:2302.06799 , 2023.
- Zhou, F., Wang, J., and Feng, X. Non-crossing quantile regression for distributional reinforcement learning. Advances in Neural Information Processing Systems , 33: 15909-15919, 2020.
- Zhou, F., Zhu, Z., Kuang, Q., and Zhang, L. Non-decreasing quantile function network with efficient exploration for distributional reinforcement learning. International Joint Conference on Artificial Intelligence , pp. 3455-3461, 2021.

## A. Projection Operator

## A.1. Categorical projection operator

CDRL algorithm uses a categorical projection operator Π C : P ( R ) → P ( { z 1 , . . . , z N } ) to restrict approximated distributions to the parametric family of the form F C := { ∑ N i =1 p i δ z i | ∑ N i =1 p i = 1 , p i ≥ 0 } ⊆ P ( R ) , where z 1 &lt; · · · &lt; z N are evenly spaced, fixed supports. The operator Π C is defined for a single Dirac delta as

<!-- formula-not-decoded -->

## A.2. Quantile projection operator

QDRL algorithm uses a quantile projection operator Π W 1 : P ( R ) → P ( R ) to restrict approximated distributions to the parametric family of the form F W 1 := { 1 N ∑ N i =1 δ z i | z 1: N ∈ R N } ⊆ P ( R ) . The operator Π W 1 is defined as

<!-- formula-not-decoded -->

where τ i = 2 i -1 2 N , and F µ is the CDF of µ . The midpoint 2 i -1 2 N of the interval [ i -1 N , i N ] minimizes the 1-Wasserstein distance W 1 ( µ, Π W 1 µ ) between the distribution, µ , and its projection Π W 1 µ (a N -quantile distribution with evenly spaced τ i ), as demonstrated in Lemma 2 (Dabney et al., 2018b).

## B. Proofs

In this section, we provide the proofs of the theorems discussed in the main manuscript.

## B.1. Proof of Section 3

Proposition B.1 (Sobel, 1982; Bellemare et al., 2017) . Suppose there are value distributions ν 1 , ν 2 ∈ P ( R ) , and random variables Z k +1 i ∼ T π ν i , Z k i ∼ ν i . Then, we have

<!-- formula-not-decoded -->

Proof. This proof follows directly from Bellemare et al. (2017). The first statement can be proved using the exchange of E T π = T π E . By independence of R and P π Z i , where P π is the transition operator, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we have

Lemma B.2 (Lemma B.2 of Rowland et al. (2019)) . Let τ k = 2 k -1 2 K , for k = 1 , . . . , K . Consider the corresponding 1-Wasserstein projection operator Π W 1 : P ( R ) → P ( R ) , defined by

<!-- formula-not-decoded -->

for all µ i ∈ P ( R ) , where F -1 µ i is the inverse CDF of µ i . Let random variable X ∼ µ 1 , X 2 ∼ µ 2 , and η 1 , η 2 ∈ P ( R ) . Suppose immediate reward distributions supported on [ -R max , R max ] . Then, we have: (i) W 1 (Π W 1 µ 1 , µ 1 ) ≤ 2 R max ;

K

(1

-

γ

)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This proof follows directly from Lemma B.2 of Rowland et al. (2019). For proving (i), let F -1 µ 1 be the inverse CDF of µ 1 . We have

<!-- formula-not-decoded -->

For proving (ii), using the triangle inequality and statement (i):

<!-- formula-not-decoded -->

(ii) implies the fact that the quantile projection operator Π W 1 is not a non-expansion under 1-Wasserstein distance, which is important for the uniqueness of the fixed point and the convergence of the algorithm.

The proof of (iii) is similar to (i), using the fact that the return distribution µ 2 is bounded on [0 , R 2 max 1 -γ ] to obtain the following inequality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem B.3 ( Parameterization induced error bound ) . Let Π W 1 be a projection operator onto evenly spaced quantiles τ i 's where each τ i = 2 i -1 2 N for i = 1 , . . . , N , and η k ∈ P ( R ) be the return distribution of k -th iteration. Let random variables Z k θ ∼ Π W 1 T π η k and Z k ∼ T π η k . Assume that the distribution of the immediate reward is supported on [ -R max , R max ] , then we have

<!-- formula-not-decoded -->

where E k 3 is parametrization induced error at k -th iteration.

Proof. Using the dual representation of the Wasserstein distance (Villani, 2009) and Lemma B.2, ∀ ( x, a ) , we have

<!-- formula-not-decoded -->

By taking the limitation over ( x, a ) and iteration k on the left-hand side, we obtain

<!-- formula-not-decoded -->

In a similar way, the second-order moment can be bounded by,

<!-- formula-not-decoded -->

It suggests that higher-order moments are not preserved after quantile representation is applied.

## B.2. Proof of Section 4

Lemma B.4 (expectation by quantiles) . . Let Z ∼ ν be a random variable with CDF F ν and quantile function F -1 ν . Then,

<!-- formula-not-decoded -->

Proof. As any CDF is non-decreasing and right continuous, we have for all ( τ, z ) ∈ (0 , 1) × R :

<!-- formula-not-decoded -->

Then, denoting U by a uniformly distributed random variable over [0 , 1] ,

<!-- formula-not-decoded -->

which shows that the random variable F -1 ν ( U ) has the same distribution as Z . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.5. Consider the linear regression model ˆ Q = X 2 M 2 + E , E is distributed on N ( 0 , σ 2 V ) , where V = diag ( v 1 , v 2 , · · · , v N ) , v i ≥ 1 , i = 1 , · · · , N , and we set noise variance σ 2 = 1 without loss of generality. The WLS estimator is

<!-- formula-not-decoded -->

and the distribution of mean estimator takes the form,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Premultiplying by V -1 / 2 , we get the transformed model

<!-- formula-not-decoded -->

Now, set ˆ Q ∗ = V -1 / 2 Q , X ∗ 2 = V -1 / 2 X 2 , and E ∗ = V -1 / 2 E , so that the transformed model can be written as ˆ Q ∗ = X ∗ 2 M 2 + E ∗ . The transformed model is a Gaussian-Markov model, satisfying OLS assumptions. Thus, the unique OLS solution is ̂ M 2 = ( X ⊤ 2 V -1 X 2 ) -1 X ⊤ 2 V -1 ˆ Q , and ̂ M 2 ∼ N ( M 2 , σ 2 ( X ⊤ 2 V -1 X 2 ) -1 ) . By computing ( X ⊤ 2 V -1 X 2 ) -1 ,

<!-- formula-not-decoded -->

When V equals identity matrix I ,

Proposition B.6. Suppose the noise ε i independently follows N (0 , v i ) where v i ≥ 1 for i = 1 , · · · , N , then,

(i) In the homoskedastic case where v i = 1 for i = 1 , . . . N , the empirical mean estimator ˆ m ∗ 1 has a lower variance, Var( ˆ m ∗ 1 ) &lt; Var( ˆ m 1 ) ;

(ii) In the heteroskedastic case where v i 's are not equal, the QEM estimator ˆ m 1 achieves a lower variance, i.e. Var( ˆ m 1 ) &lt; Var( ˆ m ∗ 1 ) , if and only if ¯ v 2 -1 -1 / ( ( ∑ i v i ∑ i v i z 2 τ i ) ( ∑ i v i z τ i ) 2 -1) &gt; 0 , where ¯ v = 1 N ∑ i v i . This inequality holds when z τ i = -z τ N -i , which can be guaranteed in QDRL.

Proof. The proof of (i) comes directly from the comparison of variances, i.e. Var( ˆ m 1 ) = 1 N &lt; 1 N + ¯ z 2 ∑ i ( z τ i -¯ z ) 2 = Var( ˆ m ∗ 1 ) . Next, we prove that (ii) holds under a sufficient condition z τ i = -z τ N -i . In QDRL, the quantile levels τ i = 2 i -1 2 N are equally spaced around 0.5. Under this setup, the condition z τ i = -z τ N -i indeed holds, where z τ i is the τ i -th quantile of standard normal distribution. For N = 2 , we need to validate the inequality ¯ v 2 -1 -1 / ( ( ∑ i v i ∑ i v i z 2 τ i ) ( ∑ i v i z τ i ) 2 -1) &gt; 0 . This can be transformed into a multivariate extreme value problem. By analyzing the function f ( v 1 , v 2 ) = ( v 1 + v 2 ) 2 4 -1 -1 ( v 1 + v 2 ) 2 ( v 1 -v 2 ) 2 -1 , the infimum of f ( v 1 , v 2 ) is 0 when v 1 , v 2 &gt; 1 , and f ( v 1 , v 2 ) reaches 0 at the limit lim ( v 1 ,v 2 ) → (1 , 1) f ( v 1 , v 2 ) = 0 . For N = 3 , this case is identical to N = 2 since z 0 . 5 = 0 . For N = 4 , f ( v 1 , v 2 , v 3 , v 4 ) = ( v 1 + v 2 + v 3 + v 4 ) 2 N 2 -1 -1 ( v 1 + v 2 + v 3 + v 4 )( k 2 v 1 + v 2 + v 3 + k 2 v 4 ) ( kv 1 + v 2 -v 3 -kv 4 ) 2 -1 , and this expression can be factored as, f ( v 1 , v 2 , v 3 , v 4 ) = v 1 + v 2 + v 3 + v 4 N 2 C ( ( v 1 + v 2 + v 3 + v 4 ) C -N 2 ( k 2 v 1 + v 2 + v 3 + k 2 v 4 ) ) , where C = ( k -1) 2 v 1 v 2 + ( k + 1) 2 v 1 v 3 + 4 k 2 v 1 v 4 + 4 v 2 v 3 + ( k + 1) 2 v 2 v 4 + ( k + 1) 2 v 3 v 4 , and k = Φ -1 (7 / 8) Φ -1 (5 / 8) &gt; 3 . By comparing the coefficient corresponding to the same terms, we can verify that f ( v 1 , v 2 , v 3 , v 4 ) &gt; 0 when v i &gt; 1 . Finally, the remaining cases can be proven in the same manner.

Theorem B.7. Consider the policy ˆ π that is learned policy, and denote the optimal policy to be π opt , α = max x ′ D TV (ˆ π ( · | x ′ ) ∥ π opt ( · | x ′ )) , and n ( x, a ) = |D| . For all δ ∈ R , with probability at least 1 -δ , for any η ( x, a ) ∈ P ( R ) , and all ( x, a ) ∈ D ,

<!-- formula-not-decoded -->

Proof. We give this proof in a tabular MDP. Directly following from the definition of the distributional Bellman operator applied to the CDF, we have that

<!-- formula-not-decoded -->

For notation convenience, we use random variables instead of measures. ˆ P and ˆ R are the maximum likelihood estimates of the transition and the reward functions, respectively. Adding and subtracting ∑ x ′ ,a ′ ˆ P ( x ′ | x, a ) π opt ( a ′ | x ′ ) F γZ ( x ′ ,a ′ )+ R ( x,a ) ( u ) , then we have

<!-- formula-not-decoded -->

For the first term, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second term can be bounded as follows:

<!-- formula-not-decoded -->

Next, we show the two norms can be bounded. By the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality, the following inequality holds with probability at least 1 -δ/ 2 , for all ( x, a ) ∈ D ,

<!-- formula-not-decoded -->

By Hoeffding's inequality and an l 1 concentration bound for multinomial distribution 4 , the following inequality holds with probability at least 1 -δ/ 2 ,

<!-- formula-not-decoded -->

4 see https://nanjiang.cs.illinois.edu/files/cs598/note3.pdf.

, and

Consequently, the claim follows from combining the two inequalities,

<!-- formula-not-decoded -->

## C. Cornish-Fisher Expansion

The Cornish-Fisher Expansion (Cornish &amp; Fisher, 1938) is an asymptotic expansion used to approximate the quantiles of a probability distribution based on its cumulants. To be more explicit, let X ∗ be a non-gaussian variable with mean 0 and variance 1. Then, the Cornish-Fisher Expansion can be represented as a polynomial expansion:

<!-- formula-not-decoded -->

where the parameters a i depend on the cumulants of the X ∗ and Φ is the standard normal distribution function. To use this expansion in practice, we need to truncate the series. According to Cornish &amp; Fisher (1938), the highest power of i must be odd, and the fourth order ( i = 3 ) approximation is commonly used in practice. The parameters for the fourth order expansion are a 2 = a 0 = κ 3 6 , a 1 = 1 + 5( κ 3 6 ) 2 -3 κ 4 24 and a 3 = κ 4 24 -2( κ 3 6 ) 2 , where κ i denotes i -th cumulant. Therefore, the fourth order expansion is

<!-- formula-not-decoded -->

Now, simply define the X ∗ as the normalization of X , X = µ + σX ∗ , with mean µ and variance σ 2 . F -1 X ( τ ) can be approximated by

<!-- formula-not-decoded -->

Denote skewness s = κ 3 σ 3 , kurtosis k = κ 4 σ 4 and normal distribution quantile z τ = Φ -1 ( τ ) . Then, we can rewrite the above equation

<!-- formula-not-decoded -->

## C.1. Regression model selection

We use the R-Squared ( R 2 ) statistic to determine the number of terms in Equation (12) that should be included in the regression model. R 2 , also known as the coefficient of determination, is a statistical measure that shows how well the independent variables explain the variance in the dependent variable. In other words, it is a measure of how well the data fit the regression model.

Consider the linear regression model,

<!-- formula-not-decoded -->

The dependent variable Y = ( F -1 X ( τ 1 ) , . . . , F -1 X ( τ N )) T is composed of the quantiles from distribution of X , and E is the noise vector sampled from N (0 , 0 . 25) . When the design matrix X 1 = (1 , · · · , 1) ′ , this regression model reduces to a one-sample problem, and β 1 can be directly estimated by 1 N ∑ N n =1 F -1 X ( τ n ) . We then investigate the following four types of regression models,

Model 1:

Model 2:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Y \_fitted

Model 1 fitted quantile (R*=0.947)|

Model 2 fitted quantile (R==0.948)|

Model 3 fitted quantile (R*=0.950) |

Model 4 fitted quantile (R?=0.950)

True quantile curve itted

Model 1 fitted quantile (R*=0.916)

Model 2 fitted quantile (R^=0.963)

Model 3 fitted quantile (R? -0.964)

Model 4 fitted quantile (R'-0.964)

True quantile curve

## Variance Control for Distributional Reinforcement Learning

Figure 9. Fitted quantile plot. (a) Normal, N (0 , 1) . (b) Mixture Gaussian, 0 . 7 N ( -2 , 1) + 0 . 3 N (3 , 1) . (c) Exponential, Exp (1) = e -x . (d) Gumbel, G (0 , 1) = e -( x + e -x ) .

<!-- image -->

Model 3:

Model 4:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 9 shows that the regression fitted values and corresponding R 2 across several distributions of X . As the number of independent variables increases, more variance in the error can be explained. However, having too many independent variables increases the risk of multicollinearity and overfitting. Based on practical considerations, we choose Model 3 as our regression model due to its satisfactory level of explainability. In the subsequent section, we will give a more in-depth interpretation of this regression model.

## C.2. Interpretation of the remaining term ω ( τ )

In this section, we explore the role of the remaining term ω ( τ ) in the context of random design regression. As discussed in Section 4, we present a decomposition of the estimate ˆ q ( τ ) of the τ -th quantile, which includes contributions from the mean, noise error, and misspecified error. Specifically, we expressed the estimate as follows:

<!-- formula-not-decoded -->

where µ can be estimated using the mean estimator 1 N ∑ q ( τ i ) , which is commonly used in QDRL and IQN settings. However, this simple model fails to capture important information in the ω 1 ( τ ) . To address this limitation, we employ the Cornish-Fisher Expansion to expand the equation, resulting in the following expression:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where µ can be estimated by linear regression estimator given multiple quantile levels { τ i } , which can be sampled from a uniform distribution or predefined to be evenly spaced in (0 , 1) . In theory, higher-order expansions can capture more misspecified information in ω ( τ ) , leading to a more accurate representation of the quantile. However, as discussed before, expansions are typically limited to the fourth order in practice to balance the trade-off between model complexity and estimation accuracy.

Model 1 fitted quantile (R*=0.941)

Model 2 fitted quantile (R*=0.941)

Model 3 fitted quantile (R' -0.942)

Model 4 fitted quantile (R? -0.942)

Model 2 fitted quantile (R*=0.937)

Model 4 fitted quantile (R? »0.941)

To gain a better understanding of the remaining term ω ( τ ) and its impact on the regression estimator, consider the linear model,

<!-- formula-not-decoded -->

where τ can be generally considered a uniform, x τ = (1 , z τ , z 2 τ -1 , ... ) ′ ∈ R d , and β = ( µ, σ, σ s 6 , ... ) ′ ∈ R d . In particular, define the random variables,

<!-- formula-not-decoded -->

where ε corresponds to the noise with zero mean, σ 2 noise variance and independent across different level of τ , and ω τ corresponds to the misspecified error of β . Under the following conditions, we can derive a bound for the regression estimator in the misspecified model.

Condition 1 (Subgaussian noise). There exist a finite constant σ noise ≥ 0 such that for all λ ∈ R , almost surely:

<!-- formula-not-decoded -->

Condition 2 (Bounded approximation error). There exist a finite constant C bias ≥ 0 , almost surely:

<!-- formula-not-decoded -->

where Σ = E [ x τ x τ ′ ] .

Condition 3 (Subgaussian projections). There exists a finite constant ρ ≥ 1 such that:

<!-- formula-not-decoded -->

Theorem C.1. Suppose that Conditions 1, 2, and 3 hold. Then for any δ ∈ (0 , 1) and with probability at least 1 -3 δ , the following holds:

<!-- formula-not-decoded -->

Noise error contribution

where K ρ,δ,N is a constant depending on ρ , δ and N .

Proof. The proof of the above theorem can be easily adapted from Theorem 2 in Hsu et al. (2011).

The first term on the right-hand side represents the error due to model misspecification, which occurs when the true model differs from the assumed model. Intuitively, incorporating more relevant information in ω ( τ ) into explanation variables could decrease the quantity of E ∥ ∥ Σ -1 / 2 x τ ω τ ∥ ∥ 2 2 and C bias . Therefore, the accuracy of the estimator may be potentially improved by reducing the magnitude of the misspecified error. The second term represents the noise error contribution, which is inevitable and can only be controlled by increasing the sample size N .

## D. Experimental Details

## D.1. Tabular experiment

The parameter settings used for tabular control are presented in Table 1. In the QEMRL case, the weight matrix V is set as shown in the table based on domain knowledge indicating that the distribution has low probability support around its median. The greedy parameter decreases exponentially every 100 steps, and the learning rate decrease in segments every 50K steps.

Table 1. The (hyper-)parameters of QEMRL and QDRL used in the tabular control experiment.

| Hyperparameter                                                                                                                                                                   | Value                                                                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Learning rate schedule Discount factor Quantile initialization Number of quantiles Number of training steps ϵ -greedy schedule Number of MCrollouts Weight matrix V (QEMRL only) | { 0.05,0.025,0.0125 } 0.999 Unif( - 0 . 5 , 0 . 5) 128 150K 0 . 9 ⌊ t/ 100 ⌋ 10000 diag { 1 , 1 , · · · , 1 . 5 , · · · , 1 . 5 ︸ ︷︷ ︸ τ ∈ [0 . 45 , 0 . 55] , · · · , 1 , 1 } |

## D.2. Atari experiment

We extend QEMRL to a DQN-like architecture, and we use the same architecture as QR-DQN, which we refer to as QEMDQN 5 . Our hyperparameter settings (Table 2) are aligned with Dabney et al. (2018b) for a fair comparison. Additionally, we extend QEMRL to the unfixed quantile fraction algorithm IQN, which embeds quantile fraction τ into the quantile value network on the top of QR-DQN. In Atari, it is infeasible to determine the low probability supports for every state-action pair, therefore we only consider the heteroskedasticity that occurs in the tail and treat V as a tuning parameter to select an appropriate value. For exploration experiments, we follow the settings of Mavrin et al. (2019) and set the decay factor c t = c √ log t t , where c = 50 .

Table 2. The hyperparameters of QEM-DQN and QR-DQN used in the Atari experiments.

| Hyperparameter                                                                            | Value                                                                                                                                |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Learning rate Discount factor Optimizer Bath size Number of quantiles Number of quantiles | 0.00005 0.99 Adam 32                                                                                                                 |
|                                                                                           | 200                                                                                                                                  |
| (IQN)                                                                                     | 32                                                                                                                                   |
| Weight matrix V (QEM-DQN only)                                                            | diag { 1 . 5 , · · · , 1 . 5 ︸ ︷︷ ︸ τ ∈ [0 . 9 , 1) , · · · , 1 , 1 , · · · , 1 . 5 , · · · , 1 . 5 ︸ ︷︷ ︸ τ ∈ (0 , 0 . 1] } |

## D.3. MuJoCo experiment

We extend QEMRL to a SAC-like architecture, and we use the same architecture of DSAC, named QEM-DSAC. Similarly, we extend QEMRL to an IQN version of DSAC. Hyperparameters and environment-specific parameters are listed in Table 3. In addition, SAC has a variant that introduces a mechanism of fine-tuning α to achieve target entropy adaptively. While this adaptive mechanism performs well, we follow the use of fixed α suggested in the original SAC paper to reduce irrelevant factors.

5 Code is available at https://github.com/Kuangqi927/QEM

Table 3. The hyperparameters of QEM-DSAC and DSAC used in the MuJoCo experiments.

| Hyperparameter                         | Value                                                                                                                                |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Policy network learning rate           | 0 . 0003                                                                                                                             |
| Quantile Value network learning rate   | 0 . 0003                                                                                                                             |
| Discount factor                        | 0.99                                                                                                                                 |
| Optimization                           | Adam                                                                                                                                 |
| Target smoothing                       | 0 . 005                                                                                                                              |
| Batch size                             | 256                                                                                                                                  |
| Minimum steps before training          | 10000                                                                                                                                |
| Number of quantiles                    | 32                                                                                                                                   |
| Quantile fraction embedding size (IQN) | 64                                                                                                                                   |
| Weight matrix V (QEM-DSAC only)        | diag { 1 . 2 , · · · , 1 . 2 ︸ ︷︷ ︸ τ ∈ [0 . 9 , 1) , · · · , 1 , 1 , · · · , 1 . 2 , · · · , 1 . 2 ︸ ︷︷ ︸ τ ∈ (0 , 0 . 1] } |

| Environment    |   Temperature Parameter |
|----------------|-------------------------|
| Ant-v2         |                    0.2  |
| HalfCheetah-v2 |                    0.2  |
| Hopper-v2      |                    0.2  |
| Walker2d-v2    |                    0.2  |
| Swimmer-v2     |                    0.2  |
| Humanoid-v2    |                    0.05 |

## E. Additional Experimental Results

## E.1. Variance reduction for IQN

IQN does not satisfy the sufficient condition z τ i = -z τ N -i since τ is sampled from a uniform distribution, rather than evenly spaced as in QDRL. To examine the impact of this on the inequality ( ∑ i v i N ) 2 -1 -1 / ( ( ∑ i v i ∑ i v i z 2 i ) ( ∑ i v i z i ) 2 -1) &gt; 0 in Proposition 4.2, simulation experiments are conducted. We use the function f ( v 1 , · · · , v N ) = ( ∑ i v i N ) 2 -1 -1 / ( ( ∑ i v i ∑ i v i z 2 τ i ) ( ∑ i v i z τ i ) 2 -1) to examine this inequality, where v i &gt; 1 and τ i are sampled uniformly. In every trial, v i are randomly sampled from [1 , M ] , repeating the process 100,000 times. The minimum values of f ( v 1 , · · · , v N ) are shown in the following Table 4 for varying values of N and M . The results indicate that the minimum of f is always greater than 0, which demonstrates that the inequality holds in practice.

| Table 4. Minimum of f .   | Table 4. Minimum of f .   | Table 4. Minimum of f .   |
|---------------------------|---------------------------|---------------------------|
| Minimum of f              | M                         | N                         |
| 0.614                     | 2                         | 32                        |
| 4.778                     | 5                         | 32                        |
| 43.143                    | 20                        | 32                        |
| 0.932                     | 2                         | 128                       |
| 7.707                     | 5                         | 128                       |
| 76.489                    | 20                        | 128                       |
| 1.082                     | 2                         | 500                       |
| 9.357                     | 5                         | 500                       |
| 96.473                    | 20                        | 500                       |

Average Return

8000

60

7000

50

Average Return

6000

5000

4000

20

3000

10

2000

1000

5

5000

-5

-10

4000 -

-15

-20

Average Return

3000

2000

1000

Average Return

25000

0.0

20000

15000

10000 -

5000

Average Return v=1.0

v=1.25

v=1.5

v=1.75|

25

v=1.0

v=1.2

v=1.4

v=1.6

v=1.8

v-2.0

0.2

Alien

Bowling

MAN

12000

5000

10000

8000

: 4000|

60001

3000

4000

## E.2. Weight V tuning experiments

QR-DON

50

100

150

Millions of frames

50

125

100

150

75

DoubleDunk

Millions of frames walker2d

50

100

150

Millions of frames

Qbert

0.4

0.6

Time Steps

50

100

150

Millions of frames

200

70000

3500

60000

2500 -

40000

1500 -

1000|

30000

era

500 -

20000

10000

50

150

v=1.0

v=1.25

v=1.5

v=1.75

v=2.01

200

50

100

150

v=1.0

ChopperCommand v= 1.25

v=1.75

v=2.0

QEM-DQN

QR-DON

200

Figure 10. Comparison of different weight v in QEM-DSAC and QEM-DQN experiments

<!-- image -->

## E.3. Additional Atari results

Figure 11. Comparison of QEM-DQN and QR-DQN across 9 Atari games

<!-- image -->

0.8

1000-

ChopperCommand

Centipede

QEM-DQN|

QR-DON

100

Av

KungFuMaster

40000|

1000 -

Average Return ge Return

800

30000

20000

6001

400 -

10000

200 -

0

20000

15000

5000

20000

15000

10000

5000

Avera

Return

Average Return

BattleZone

BankHeist

IQEM-DQN

QEM-DON+QEM

IQN

QEM-DQN+ DLTV

OR-DON +DLTV

50

100

150

Millions of frames

NameThisGame

100

50

150

Millions of frames

IQEM-DON/

IQN

50

100

150

Millions of frames

Riverraid

IQEM-DON

IQN

50

100

150

Millions of frames

3000 -

15000- era

7500 -

Av

1000-

5000

500

2500

60000

Berzerk

Seaquest

IQEM-DQN

QEM-DON+QEM

IQN

QEM-DQN+ DLTV

OR-DON+DLTV

50

100

150

Millions of frames

KungFuMaster

50

100

150

Millions of frames

IOEM-DON

IQN

Figure 12. Comparison of IQEM-DQN and IQN across 9 Atari games

<!-- image -->

Figure 13. Comparison of QEM and DLTV across 3 hard-explored Atari games

<!-- image -->

Average

200

200

1400

80

1200

Average Return

60

1000-

800-

40

600 -

400

20

200

Average Return

-200

10000

Bowling

PrivateEye

IQEM-DQN

QEM-DON+QEM

IQN

QEM-DON+ DLTV

QR-DON+ DLTV

50

100

150

200

Millions of frames

Krull

50

100

150

Millions of frames

200

200

200