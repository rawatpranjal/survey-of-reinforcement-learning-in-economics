## TEMPORAL-DIFFERENCE ESTIMATION OF DYNAMIC DISCRETE CHOICE MODELS

KARUN ADUSUMILLI + AND DITA ECKARDT *

Abstract. We study the use of Temporal-Difference learning for estimating the structural parameters in dynamic discrete choice models. Our algorithms are based on the conditional choice probability approach but use functional approximations to estimate various terms in the pseudo-likelihood function. We suggest two approaches: The first -linear semi-gradient- provides approximations to the recursive terms using basis functions. The second -Approximate Value Iteration- builds a sequence of approximations to the recursive terms by solving non-parametric estimation problems. Our approaches are fast and naturally allow for continuous and/or high-dimensional state spaces. Furthermore, they do not require specification of transition densities. In dynamic games, they avoid integrating over other players' actions, further heightening the computational advantage. Our proposals can be paired with popular existing methods such as pseudomaximum-likelihood, and we propose locally robust corrections for the latter to achieve parametric rates of convergence. Monte Carlo simulations confirm the properties of our algorithms in practice.

Key words and phrases. Dynamic discrete choice models, dynamic discrete games, Temporal-Difference learning, Reinforcement learning.

This version : December 23, 2022.

+ Department of Economics, University of Pennsylvania; akarun@sas.upenn.edu.

* Department of Economics, University of Warwick; dita.eckardt@warwick.ac.uk.

We would like to thank Xiaohong Chen, Frank Diebold, Aviv Nevo, Whitney Newey and Frank Schorfheide for helpful discussions.

## 1. Introduction

Recent years have seen a number of important developments in the field of Reinforcement Learning (RL) for computation of value functions. The goal of this paper is to study the use of a popular RL technique, Temporal-Difference (TD) learning, for estimation and inference in Dynamic Discrete Choice (DDC) models.

DDC models are frequently used to describe the inter-temporal choices of forward-looking individuals in a variety of contexts. In these models, agents maximize their expected future payoff through repeated choice amongst a set of discrete alternatives. Based on a revealed preference argument, structural estimation proceeds by using microdata on choices and outcomes to recover the underlying model parameters. 1 A key challenge in this literature is the complexity of estimation. Uncovering the structural parameters originally required an explicit solution to a dynamic programming problem in addition to the optimization of an estimation criterion. A key advance has been Hotz and Miller's (1993) Conditional Choice Probability (CCP) algorithm which avoids the repeated solution of the inter-temporal optimization problem by taking advantage of a mapping between value function differences and conditional choice probabilities.

Unfortunately, the standard CCP algorithm is computationally infeasible when the underlying states are continuous and/or the state space is high-dimensional. Such state spaces are common in applications. One frequently employed approach to tackle continuous state spaces is through state space discretization, e.g., Kalouptsidi (2014) and Almagro and Domınguez-Iino (2019) use aggregation and clustering methods to do this. However, it is not always clear how to perform such a discretization in practice, and moreover, it introduces bias into the parameter estimates. An alternative is to employ functional approximations for the value functions. For instance, Barwick and Pathak (2015) and Kalouptsidi (2018) use estimated transition densities and numerical/analytical integration to approximate the value functions using linear sieves and LASSO, respectively. However, the theoretical properties of these methods when using machine learning methods (such as LASSO) are as yet unknown, and they still require estimation of transition densities, which is not straightforward, along with numerical integration, which can be computationally expensive. 2

1 See Aguirregabiria and Mira (2010) for a detailed survey of the literature on the estimation of DDC models.

2 Yet another alternative is to use forward Monte Carlo simulations (Bajari et al., 2007, Hotz et al., 1994), but this again becomes very involved as the number of continuous state variables or players increases. The use of a finite number of Monte Carlo simulations also adds bias to the estimates.

The aim of this paper is to develop tractable algorithms for CCP estimation when the state variables are continuous and/or the state space is large. Such algorithms should possess three properties: First, they should be fast to compute even under high-dimensional state spaces. Second, they should avoid state space discretization, and instead rely on functional approximation of value functions. Third, they should avoid estimation of transition densities which are difficult to parameterize and estimate under continuous states.

In this paper, we suggest two methods, based on the idea of TD learning, that satisfy all the above properties. The methods involve two different techniques for estimating various recursive terms (which are akin to value functions) that arise in CCP estimation. The first approach, the linear semi-gradient method, provides functional approximations to the recursive terms using basis functions. This approach simply involves inverting a low-dimensional matrix, where the dimension is the number of basis functions being used. Thus the computational cost is trivial in most settings. Furthermore, it only requires the observed sequences of current and future state-action pairs as input and estimation of transition densities is not needed. The second approach, Approximate Value Iteration (AVI), builds a sequence of approximations to the value terms by solving a non-parametric estimation problem in each step. Almost any machine learning (ML) method devised for prediction can be used for approximation under this method, including (but not limited to) LASSO, random forests and neural networks. To the best of our knowledge, the AVI estimator is the first estimator for DDC models that can be applied with any ML method. Hence, it naturally allows for very high-dimensional state spaces. Again, no estimation of transition densities is required. We derive the non-parametric rates of convergence for estimation of the value terms under both these methods. With the estimates of these functions at hand, estimation of the structural parameters can proceed with standard methods such as pseudo-maximum-likelihood estimation (PMLE, Aguirregabiria and Mira (2002)) or minimum distance estimation.

The focus of this paper is on the estimation of structural parameters. To this end, our procedures allow one to avoid modeling state transitions. Performing counterfactual analysis may still require estimating the transition density, but we argue that our techniques remain useful, even for this purpose, for two reasons: First, counterfactuals often involve new transition densities which are different from the ones that enter the estimation of the structural parameters, see Kalouptsidi (2018) for an example. Our methods therefore avoid estimation of the original transition densities that may not be

needed for counterfactual analysis. Second, with continuous states, decoupling the estimation of structural parameters and transition densities may be beneficial for robustness and efficiency reasons. For instance, it is common to employ AR (e.g., Aguirregabiria and Mira, 2007; Kalouptsidi, 2014) or VAR (Barwick and Pathak, 2015) specifications for transition densities. However, even within these specifications there are a number of choices to be made (e.g., dimension of VARs, distribution of error terms etc) and the estimates of the structural parameters may not be robust to these choices. More importantly, even when non-parametric estimates of transition densities are available, simply plugging them into the second-stage PMLE criterion would seriously degrade the rate of convergence of structural parameters. One would need to adjust the PMLE to account for the non-parametric first stage, but it is not known what form this adjustment takes. By contrast, our proposals employ non-parametric estimates of value functions, and as described below, we derive the necessary adjustments to account for this. To carry out counterfactual analysis, we therefore suggest plugging in our estimates of the structural parameters, which do not rely on non-parametric estimates of transition densities and are robust to mis-specification, together with non-parametric estimates of the transition densities.

The above discussion highlights that in continuous state spaces, estimation of structural parameters is inherently a problem of semi-parametric estimation. In fact, even under discrete states, Aguirregabiria and Mira (2002) show that estimation of transition densities affects the variance of the structural parameter estimates. If the state variables are continuous, existing two-step CCP methods, such as the PMLE will no longer converge at parametric rates. We therefore derive a locally robust estimator by adding a correction term to the PMLE criterion function that accounts for the non-parametric estimation of value function terms using either of our TD methods. We emphasize that this construction is novel and does not directly follow from existing results, e.g., in Chernozhukov et al. (2022). Its computation is particularly straightforward under the linear semi-gradient approach. The resulting estimator converges at parametric rates under continuous states and unrestricted transition densities.

Our TD estimators are thus consistent, computationally very cheap, and they converge at parametric rates. Importantly, they provide a feasible estimation method when the states are continuous and/or the state space is large. This is particularly important for the estimation of dynamic discrete games. Even with discrete states, existing methods

for the estimation of dynamic games (Bajari et al., 2007; Aguirregabiria and Mira, 2007; Pesendorfer and Schmidt-Dengler, 2008) require integrating out other players' actions. With many players, or under continuous states, this can get quite cumbersome. By contrast, our procedure works directly with the joint empirical distribution of the states and their sample successors. Thus the 'integrating out' is done implicitly within the sample expectations.

Finally, we also incorporate permanent unobserved heterogeneity into our methods by combining the TD estimation with an Expectation-Maximization (EM) algorithm (Dempster et al., 1977).

A range of Monte Carlo studies confirm the workings of our algorithms. First, we present simulations based on the dynamic firm entry problem described in Aguirregabiria and Magesan (2018). The model has seven structural parameters of interest and five continuous state variables. Existing methods usually struggle at this dimensionality; certainly, discretization of the state space would not work too well. We show simulations using both the linear semi-gradient method and the AVI method where we estimate the value function terms using random forests, and derive results both with and without locally robust correction. Our findings suggest that while the linear semi-gradient has very little bias even without the locally robust correction, the AVI method has slightly higher small sample bias which is substantially reduced when using our locally robust estimator. Overall, our estimators perform very well in this setting and they outperform CCP estimators that employ discretization, leading to a 4 to 11-fold reduction in average mean squared error across the seven structural parameters of interest.

Second, we test our algorithms for dynamic discrete games based on a firm entry game similar to that outlined in Aguirregabiria and Mira (2007). We use the linear semigradient method here and, as before, our estimates are closely centered around the true parameters. Since this approach requires the selection of a set of basis functions for the functional approximations, we present results for different sets of basis functions (a second, third and fourth order polynomial) in this model. Our findings suggest that the choice of basis functions has only a small effect on the performance of the estimator. Moreover, we show that a simple cross-validation procedure may be used to select the preferred set of functions.

1.1. Related literature. The paper contributes to the literature on estimation of DDC models. Rust (1987) is the seminal work in this literature. Motivated by computational

considerations, Hotz and Miller (1993) propose the CCP algorithm. The CCP idea has subsequently been refined by Hotz et al. (1994) who suggest a simulation-based method, and Aguirregabiria and Mira (2002) who develop a pseudo-likelihood estimator. Arcidiacono and Miller (2011) introduce and exploit the property of finite dependence to speed up CCP estimation. Despite these advances, the estimation of DDC models remains constrained by its computational complexity, particularly in the large class of models where finite dependence does not hold. Estimation of dynamic discrete games is particularly affected by these issues as the strategic interaction of agents means that the state space increases exponentially with the number of players. It is also uncommon for finite dependence to hold in dynamic games.

The standard CCP algorithm is a two-step method, and is known to suffer from severe bias in finite samples. Aguirregabiria and Mira (2002; 2007) address this issue by presenting a recursive CCP algorithm, the nested pseudo-likelihood (NPL) algorithm. Under discrete states, the first-step estimation does not affect the rate of convergence, but shows up in form of higher-order bias for the structural parameters. The NPL algorithm can ameliorate this, see Kasahara and Shimotsu (2008). With continuous states, however, estimation of the transition density introduces bias that is the dominant term in determining the rate of convergence. This motivates the construction of our locally robust estimator which gets rid of this bias and restores parametric rates of convergence.

Ackerberg et al. (2014) and Chernozhukov et al. (2022) consider semi-parametric estimation using ML methods when either finite dependence or a 'terminal action' property holds (Hotz and Miller, 1993). Chernozhukov et al. (2022) also derive locally robust corrections for this setting. In both cases, the PMLE criterion can be written as a function of choice probabilities only (transition densities are not required); the authors employ non-parametric estimates for choice probabilities and correct for this estimation in the second stage. Computation and estimation is thus relatively simpler under finite dependence. By contrast, our methods are applicable to the more general and difficult setting where finite dependence may not apply. Nevertheless, the computational speed of our linear semi-gradient procedure is comparable to methods that exploit finite dependence. For dynamic games, Semenova (2018) allows for high-dimensional state spaces, but the parameters are only partially identified.

In making use of TD learning, our methods relate to the literature on RL, particularly batch RL. Batch RL describes learning about how to map states into actions so

as to maximize an expected payoff, using a fixed set of data (also called a batch); see Lange et al. (2012) for a survey. 3 It is closely related to the idea of 'experience replay', an important ingredient of RL algorithms that achieve human level play in Atari games (Mnih et al., 2015). A key step in RL, including batch RL, is the estimation of value functions. TD learning methods, first formulated by Sutton (1988), are the most commonly used set of algorithms for this purpose. We study non-parametric estimation of value functions using two TD methods: semi-gradients and AVI. Our analysis builds on the techniques developed by Tsitsiklis and Van Roy (1997) for linear semi-gradients, and Munos and Szepesvári (2008) for AVI. While Tsitsiklis and Van Roy (1997) focus on online learning (i.e., where collection of data and estimation of value functions is conducted simultaneously), we translate their methods to batch learning. With regards to Munos and Szepesvári (2008), we differ in employing assumptions that are more common to econometrics and our characterization of the rates is also different (compare Theorem 2 in their paper with our Theorem 3).

TD methods are distinct from other value function approximation methods developed in economics, e.g., parametric policy iteration (Hall et al., 2000), simulation and interpolation (Keane and Wolpin 1994), and sieve value function iteration (Arcidiacono et al., 2013). The last of these is similar in spirit to AVI with linear functional approximations. However, our semi-gradient method provides a linear approximation in a single step without any need for iterations, and we analyze AVI under generic machine learning methods. Our approximation results, and the technical arguments leading to them, are thus different from Arcidiacono et al. (2013); in fact, their setting is different too as the authors focus on estimating the 'optimal' value function, while the recursive terms in our setting are more akin to a value function under a fixed policy.

The remainder of this paper is organized as follows. Section 2 outlines the setup of the DDC model and fixes notation. Section 3 describes our TD estimation method for the functional approximations of the value functions. Section 4 proves its theoretical properties and describes the second-step estimation of the structural parameters under discrete and continuous state variables. Section 5 discusses the estimation of dynamic discrete games. Section 6 provides Monte Carlo simulations for our algorithm. Section 7 concludes. The Online Appendix discusses extensions to permanent unobserved heterogeneity.

3 See Sutton and Barto (2018) for a detailed treatment of RL in genreal.

## 2. Setup

We start with a single agent DDC model. In particular, we consider an infinite horizon model in discrete time with i = 1 , . . . , n agents. We assume that the individuals are homogeneous, relegating extensions for unobserved heterogeneity to Online Appendix B.6. In each period, an agent chooses among A mutually exclusive actions, each of which is denoted by a . The payoff from the action depends on the current state x . Choosing action a when the state is x gives the agent an instantaneous utility of z ( a, x ) ᵀ θ + e , where z ( a, x ) is some known vector valued function of a, x and e is an idiosyncratic error term. We denote the realization of the state of an individual i at time t by x it , and her corresponding action and error terms by a it and e it . We assume that e it is an iid draw from some known distribution g e ( · ). Let ( a ′ , x ′ ) denote the one-period ahead random variables immediately following the actions and states ( a, x ), where x ′ ∼ K ( ·| a, x ), with K ( ·| a, x ) denoting the transition density given a, x (more precisely, it is the Markov kernel). We do not make any parametric assumptions about K ( ·| a, x ). The utility from future periods is discounted by β .

Agent i chooses actions a i = ( a i 1 , a i 2 , . . . ) to sequentially maximize the discounted sum of payoffs

<!-- formula-not-decoded -->

The econometrician observes a panel consisting of state-action pairs for all individuals, ( x i , a i ) = { ( x i 1 , a i 1 ) , . . . , ( x iT , a iT ) } , for T periods (note, however, that the agent maximizes an infinite horizon objective, not a fixed T one). Typically T /lessmuch n in applications, so we work within an asymptotic regime where n →∞ but T is fixed. Using this data, the econometrician aims to recover the structural parameters θ ∗ .

In this paper, we study the CCP approach for estimating θ ∗ (Hotz and Miller, 1993). CCP methods are based on the conditional choice probabilities of choosing action a given state x . We denote these by P t ( a | x ) for a given period t but henceforth drop the subscript t with the idea that it can be made a part of the state variable x , if needed (we should also add that some of our theoretical results are based on assuming stationarity, i.e P t ( a | x ) is independent of t ). Denote e ( a, x ) as expected value of the idiosyncratic error term e given that action a was chosen. Hotz and Miller (1993) show that if the distribution of e follows a Generalized Extreme Value (GEV) distribution, it is possible to express e ( a, x ) as a function of the choice probabilities P ( a | x ), i.e e ( a, x ) = G ( P ( a | x )). We assume that

e follows a Type I Extreme Value distribution, which is perhaps the most common choice in applications. In this case e ( a, x ) = γ -ln P ( a | x ), where γ is the Euler constant.

The standard procedure in the CCP approach is as follows: Under the given distributional assumptions, the parameters are obtained as the maximizers of the pseudolikelihood function

<!-- formula-not-decoded -->

where for any ( a, x ), h ( . ) and g ( . ) solve the following recursive expressions:

<!-- formula-not-decoded -->

Here, E [ . ] denotes the expectation over the distribution of ( a ′ , x ′ ) conditional on ( a, x ). Note that E [ h ( a ′ , x ′ ) | a, x ] is a function of K ( ·| a, x ) , P ( . | x ). Both h ( a, x ) and g ( a, x ) have a 'value-function' form, which turns out to be useful for our approach.

Clearly, h ( . ) and g ( . ) are functions of K ( ·|· ) and P ( . | . ). Since the latter are unknown, the current literature proceeds by first estimating these as ( ˆ K, ˆ P ). Typically, ˆ K is obtained by MLE based on a parametric form of K ( x ′ | a, x ; θ f ), while ˆ P is estimated nonparametrically using either a blocking scheme or kernel regression. Then, given ( ˆ K, ˆ P ), the values of h ( . ) and g ( . ) can be estimated by solving the recursive equations (2.1). In the next section, we propose an alternative algorithm for maximizing Q ( θ ) that directly estimates h ( · ) and g ( · ) in a single step without requiring any knowledge about or estimation of K ( ·|· ).

Notation. We assume that the distribution of ( a it , x it ) is time stationary. This greatly simplifies our notation. It is not necessary for our results on the approximation properties of our TD methods, see Appendix A, but we do require it for deriving a locally robust estimator. Let P denote the stationary population (i.e, in the limit as n → ∞ ) distribution of ( a, x, a ′ , x ′ ), and E [ · ] the corresponding expectation over P . Define E n [ · ] as the expectation over the empirical distribution, P n , of ( a, x, a ′ , x ′ ). In particular, E n [ f ( a, x, a ′ , x ′ )] := ( n ( T -1)) -1 ∑ n i =1 ∑ T -1 t =1 f ( a it , x it , a it +1 , x it +1 ), i.e we always drop the last time period in the summation index even if f ( · ) does not depend on a ′ , x ′ .

Let H denote the space of all square integrable functions over the domain A×X of ( a, x ). Define the pseudo-norm ‖·‖ 2 over H as ‖ f ‖ 2 := E [ | f ( a, x ) | 2 ] 1 / 2 for all f ∈ H . We use |·| to denote the usual Euclidean norm on a Euclidean space.

## 3. Temporal-difference estimation

This section presents our TD method for estimating h ( . ) and g ( . ). Note that h ( · ) is a vector, of the same dimension as θ . Our methods provide functional approximations separately for each component h ( j ) of h . To simplify notation, we drop the superscript j indexing the elements of h ( . ) and proceed as if the latter, and therefore θ ∗ , is a scalar. However, it should be taken as implicit that all our results hold for general h ( . ), as long as each of its elements are treated separately.

For any candidate function, f ( a, x ), for h ( a, x ), denote the TD error by

<!-- formula-not-decoded -->

and the dynamic programming operator by

<!-- formula-not-decoded -->

Clearly, h ( a, x ) is the unique fixed point of Γ z [ · ]. TD estimation involves approximating h ( a, x ) using a functional class F , where each element h ( · ; ω ) of F is indexed by a finitedimensional vector ω . The aim in TD estimation is to ostensibly minimize the meansquared TD error

<!-- formula-not-decoded -->

However, this minimization problem is neither computationally feasible nor is it proven to converge when h / ∈ F . Instead, two approaches are commonly used.

The first approach, the semi-gradient method, involves updating ω using the semigradients

<!-- formula-not-decoded -->

for some small value of γ . As the name suggests, the above is not a complete gradient as the derivative does not take into account how ω affects the 'target', i.e., the future value h ( a ′ , x ′ ; ω ). Nevertheless, for linear functional classes F , it is possible to explicitly characterize the limit point of the updates, ω ∗ , and compute it directly. Section 3.1 describes this in greater detail. In the RL literature, it is popular to employ semigradients with neural networks as the functional class F , but it appears difficult to extend our theoretical analysis to this setting.

The second approach, Approximate Value Iteration (AVI; Munos and Szepesvári, 2008), employs the idea of 'target networks'. In this approach, the parameters in the future value of h , i.e h ( a ′ , x ′ ; ω ), are fixed at the current ω , and the functional parameters iteratively updated as

<!-- formula-not-decoded -->

Clearly, the semi-gradient method and AVI are closely related: if one were to solve the minimization problem in (3.2) using gradient descent, the updates (within each iteration) would look similar to (3.1) except for fixing the value of ω in h ( a ′ , x ′ ; ω ) at the past values. After the gradient updates converge, i.e at the end of the iteration, h ( a ′ , x ′ ; ω ) is revised with the new ω . The semi-gradient approach can thus be considered a one-step variant of AVI. Section 3.2 describes AVI in greater detail. We characterize the theoretical properties of AVI under general functional classes F including neural networks, random forests, LASSO etc.

The approximation to g follows similarly after replacing δ z ( · ; f ) , Γ z [ · ] by

<!-- formula-not-decoded -->

3.1. Semi-gradients. The semi-gradient approach with linear F is particularly well suited for computation. Let φ ( a, x ) consist of a set of basis functions over the domain ( a, x ). Then the linear approximation class is F ≡ { φ ( a, x ) ᵀ ω : ω ∈ R k } . Denote the projection operator onto F by P φ :

<!-- formula-not-decoded -->

For linear basis functions, it can be shown, e.g Tsitsiklis and Van Roy (1997), that the sequence of functional approximations h ( a, x ; ω j ) := φ ( a, x ) ᵀ ω j converges to h ∗ := φ ( a, x ) ᵀ ω ∗ , defined as the fixed point of the projected dynamic programming operator P φ Γ z [ · ] (i.e., P φ Γ z [ h ∗ ] = h ∗ ). Based on this characterization, we show in Lemma 1 in Appendix A that h ∗ ( a, x ) = φ ( a, x ) ᵀ ω ∗ , where

<!-- formula-not-decoded -->

Lemma 2 in Appendix A assures that E [ φ ( a, x ) ( φ ( a, x ) -βφ ( a ′ , x ′ )) ᵀ ] is indeed nonsingular as long as β &lt; 1 and E [ φ ( a, x ) φ ( a, x ) ᵀ ] is non-singular. As defined above, ω ∗

cannot be computed directly, since it is a function of the true expectation E [ · ]. We can however obtain an estimator, ˆ ω , after replacing E [ · ] with the sample expectation E n [ · ]:

<!-- formula-not-decoded -->

Using ˆ ω , we obtain an estimate of h ( · ) as ˆ h ( a, x ) = φ ( a, x ) ᵀ ˆ ω .

We now turn to the estimation of g ( · ). As with h ( · ), we approximate g ( · ) using basis functions r ( a, x ), which may generally be different from φ ( a, x ). Let P r denote the projection operator onto the space { r ( a, x ) ᵀ ξ : ξ ∈ R k } . The limit of the semi-gradient iterations is g ∗ ( a, x ) := r ( a, x ) ᵀ ξ ∗ , defined as the fixed point of P r Γ e [ · ]. Assuming e ( a, x ) is known, we obtain the following characterization of ξ ∗ in analogy with (3.3):

<!-- formula-not-decoded -->

In the above, e ( a, x ) = γ -ln P ( a | x ) is a function of choice probabilities, which are unknown. Denote η ( a, x ) := P ( a | x ). Suppose that we have access to a non-parametric estimator ˆ η of η . This can be obtained, e.g., through series or kernel regression. We can then plug in this estimate to obtain e ( a, x ; ˆ η ) := γ -ln ˆ η ( a, x ). This in turn enables us to estimate ξ ∗ using ˆ ξ , computed as

<!-- formula-not-decoded -->

Using the above, we obtain an estimate of g ( · ) as ˆ g ( a, x ) = r ( a, x ) ᵀ ˆ ξ .

Interestingly, estimation of ξ ∗ is unaffected to a first order by the estimation of ˆ η , even though the latter converges to the true η at non-parametric rates (see Section 4 for a formal statement). This is because an orthogonality property holds for the estimation of ξ , in that

<!-- formula-not-decoded -->

where ∂ η · denotes the Fréchet derivative with respect to η . To show (3.7), expand

<!-- formula-not-decoded -->

where the first equality follows from the Markov property. Consider the functional M (˜ η ) := E [ ln ˜ η ( a ′ , x ′ ) | x ′ ] at different candidate values ˜ η ( · , · ). At the true conditional choice probability, η , M (˜ η ) becomes the conditional entropy of P ( a | x ′ ) and attains its

maximum. Hence, ∂ η E [ ln η ( a ′ , x ′ ) | x ′ ] = 0 and (3.7) follows from (3.8). Consequently, ˆ ξ is a locally robust estimator for ξ .

Note that computation of ˆ ω and ˆ ξ only involves solving linear equations of dimension dim( φ ) and dim( r ), respectively. This is computationally very cheap. Using ˆ h ( a, x ) and ˆ g ( a, x ), we can in turn estimate θ ∗ in many different ways. For instance, we can plug them into the PMLE estimator

<!-- formula-not-decoded -->

However, such plug-in estimates are sub-optimal. In Section 4.2, we suggest a locally robust version of (3.9).

Suppose that the underlying states and actions are discrete, and that our algorithm uses basis functions comprised of the set of all discrete elements of x, a . Then, we show in Online Appendix B.1 that the resulting estimate of h ( a, x ) is exactly the same as that obtained from the standard CCP estimators, if both the choice and transition probabilities were estimated using cell values.

3.2. Approximate Value Iteration (AVI). For a feasible estimation procedure using AVI, we can replace E [ · ] by E n [ · ] in (3.2). The procedure builds a sequence of approximations { ˆ h j := h ( a, x ; ˆ ω j ) , j = 1 , . . . , J } for h , where

<!-- formula-not-decoded -->

The process can be started with an arbitrary initialization, e.g., ˆ h 1 ( a, x ) = z ( a, x ). The maximum number of iterations, J , is only limited by computational feasibility.

The minimization problem in (3.10) is equivalent to a prediction problem using the functional class F , where the outcomes are z ( a, x ) + β ˆ h j ( a ′ , x ′ ) evaluated at the various sample draws of ( a, x, a ′ , x ′ ). Hence, the estimation target for ˆ h j +1 is the conditional expectation E [ z ( a, x ) + β ˆ h j ( a ′ , x ′ ) | a, x ] ≡ Γ z [ ˆ h j ]( a, x ). In other words, each ˆ h j +1 is a nonparametric approximation to Γ z [ ˆ h j ], and in this manner AVI builds a series of approximate value function iterations (as its name suggests).

The interpretation of (3.10) as a prediction problem also enables us to employ any machine learning method devised for prediction, including (but not limited to) LASSO, random forests and neural networks. Our theoretical results show that it is possible to

estimate h at suitably fast rates under very weak assumptions on the non-parametric estimation rates of machine learning methods.

The estimation procedure for g ( · ) is similar: we construct a sequence of approximations { ˆ g j , j = 1 , . . . , J } for g as

<!-- formula-not-decoded -->

As in Section 3.1, it will be shown that the estimation error of η is first-order ignorable for the estimation of g . Using ˆ h ( a, x ) and ˆ g ( a, x ), we can, as before, estimate θ ∗ in many different ways, including the PMLE estimator 3.9.

Compared to the semi-gradient approach, AVI is computationally more expensive as it requires solving J prediction problems (in Section 4.1 we show that in the worst case J ≈ ln n , but this can be substantially reduced through good initializations). However, semi-gradient methods require differentiable classes of functions (e.g., random forests are not allowed) and it appears difficult to characterize their theoretical properties in the case of nonlinear basis functions.

3.3. Tuning parameters. Both the semi-gradient and AVI methods will require choosing tuning parameters. For AVI this is straightforward: as each iteration is a nonparametric estimation problem, the tuning parameters can be chosen in the usual manner, e.g., using cross-validation. In the case of linear semi-gradient methods, the tuning parameters are the dimensions k φ = dim( φ ) and k r = dim( r ) of the basis functions. In analogy with AVI, we propose selecting both through a procedure akin to cross-validation. The value of ω is estimated using a training sample and its performance evaluated on a hold-out or test sample, where the performance is measured in terms of the empirical mean-squared TD error E n, test [ δ 2 h ( a, x, a ′ , x ′ ; ˆ h )] on the test dataset. The values of k φ , k r are chosen to minimize the mean squared TD error.

3.4. Nonlinear utility functions. We take the utility function to be linear in θ for simplicity - and because it is the most common choice in practice - but this is easily relaxed. Denote the nonlinear utility by z ( a, x ; θ ). Then, θ ∗ is the maximizer of the pseudo-likelihood criterion

<!-- formula-not-decoded -->

where, for each θ , h ( . ; θ ) is defined analogously to (2.1) with z ( a, x ; θ ) replacing z ( a, x ).

We can use our TD procedures to obtain a functional approximation ˆ h ( a, x ; θ ) for h ( a, x ; θ ) at each θ (estimation of g ( · ) remains unchanged as it does not depend on θ ). As argued earlier, computation of ˆ h ( · ; θ ) can be very fast. Appealingly, the matrix E n [ φ ( a, x ) ( φ ( a, x ) -βφ ( a ′ , x ′ )) ᵀ ] employed in the linear semi-gradient estimate (3.4) does not feature z ( a, x ; θ ), and therefore only has to be inverted once. We can then plug in the values of ˆ h ( . ; θ ) and ˆ g ( · ) to estimate θ ∗ as

<!-- formula-not-decoded -->

A locally robust counterpart to ˆ Q ( θ ) can be derived in the same manner as in Section 4.2. However, computation of ˆ θ is more involved as it is no longer convex optimization.

3.5. Unobserved heterogeneity. In Online Appendix B.6 we incorporate permanent unobserved heterogeneity into our models by pairing our TD methods with the sequential Expectation-Maximization (EM) algorithm (Arcidiacono and Jones, 2003). The resulting algorithm can handle discrete heterogeneity in both individual utilities and transition densities. Online Appendix B.7.2 provides Monte-Carlo evidence suggesting that the algorithm works well in practice.

## 4. Theoretical Properties of TD estimators

4.1. Estimation of non-parametric terms. We characterize rates of convergence for estimation of h ( · ) and g ( · ) under both semi-gradients and AVI.

4.1.1. Linear semi-gradients. We impose the following assumptions for estimation of h ( · ).

Assumption 1. (i) The basis vector φ ( a, x ) is linearly independent (i.e. φ ( a, x ) ᵀ ω = 0 for all ( a, x ) if and only if ω = 0 ). Additionally, the eigenvalues of E [ φ ( a, x ) φ ( a, x ) ᵀ ] are uniformly bounded away from zero for all k φ .

(ii) | φ ( a, x ) | ∞ ≤ M for some M &lt; ∞ .

- (iii) There exists C &lt; ∞ and α &gt; 0 such that ‖ h -P φ [ h ] ‖ 2 ≤ Ck -α φ .
- (iv) The domain of ( a, x ) is a compact set, and | z ( a, x ) | ∞ ≤ L for some L &lt; ∞ .
- (v) k φ →∞ and k 2 φ /n → 0 as n →∞ .

Assumption 1(i) rules out multi-collinearity in the basis functions. This is easily satisfied. Assumption 1(ii) ensures that the basis functions are bounded. This is again a mild requirement and is easily satisfied if either the domain of ( a, x ) is compact, or the

basis functions are chosen appropriately (e.g., a Fourier basis). Assumption 1(iii) is a standard condition on the rate of approximation of h ( a, x ) using a basis approximation. The value of α is related to the smoothness of h ( · ). Newey (1997) shows that for splines and power series, α = r/d , where r is the number of continuous derivatives of h ( a, · ) and d is the dimension of x . Similar results can also be derived for other approximating functions such as Fourier series, wavelets and Bernstein polynomials. The smoothness properties of h ( a, · ) are discussed in Online Appendix B.2, where we provide primitive conditions on z ( a, x ) , K ( x ′ | a, x ) that ensure existence of r continuous derivatives of h ( a, · ) for each a ∈ A . Assumption 1(iv) requires z ( a, x ) to be bounded. Finally, Assumption 1(v) specifies the rate at which the dimension of the basis functions are allowed to grow. The rate requirements are mild, and are the same as those employed for standard series estimation. For the theoretical properties, the exact rate of k φ is not relevant up to a first order since we propose estimators of θ ∗ that are locally robust to estimation of h ( · ).

We then have the following theorem on the estimation of h ( a, x ):

Theorem 1. Under Assumptions 1(i) - 1(v), the following hold:

- (i) Both ω ∗ and ˆ ω exist, the latter with probability approaching one.

<!-- formula-not-decoded -->

- (iii) The L 2 error for the difference between h ( a, x ) and φ ( a, x ) ᵀ ˆ ω is bounded as

<!-- formula-not-decoded -->

We prove Theorem 1 in Appendix A.1 by adapting the results of Tsitsiklis and Van Roy (1997). The first part of Theorem 1 assures that both population and empirical TD fixed points exist. The second and third parts of Theorem 1 imply that the approximation bias and MSE of linear semi-gradients are analogous to those of standard series estimation apart from a (1 -β ) -1 factor.

For the estimation of ˆ ξ we make use of cross-fitting as a technical device to obtain easyto-verify assumptions for the estimation of ˆ η . This entails the following: we randomly partition the data into two folds. We estimate ˆ ξ separately for each fold using ˆ η estimated from the opposite fold. The final estimate of ξ ∗ is the weighted average of ˆ ξ from both the folds. We think of cross-fitting in this context as a convenient assumption for the proofs, and do not believe it is necessary in practice.

We impose the following assumptions for the estimation of g ( a, x ). Let k r denote the dimension of r ( a, x ).

Assumption 2. (i) The basis vector r ( a, x ) is linearly independent, and the eigenvalues of E [ r ( a, x ) r ( a, x ) ᵀ ] are uniformly bounded away from zero for all k r .

- (ii) | r ( a, x ) | ∞ ≤ M for some M &lt; ∞ .
- (iii) There exists C &lt; ∞ and α &gt; 0 such that ‖ g ( a, x ) -P r [ g ( a, x )] ‖ 2 ≤ Ck -α r .
- (iv) The domain of ( a, x ) is a compact set, and | e ( a, x ) | ∞ ≤ L &lt; ∞ .
- (v) k r →∞ and k 2 r /n → 0 as n →∞ .

(vi) ˆ ξ is estimated from a cross-fitting procedure described above. The conditional choice probability function satisfies η ( a, x ) ≥ δ &gt; 0 , where δ is independent of a, x . Additionally, ‖ η -ˆ η ‖ ∞ = o p (1) and ‖ η -ˆ η ‖ 2 2 = o p ( n -1 / 2 ) .

Assumption 2 is a direct analogue of Assumption 1, except for the last part which provides regularity conditions when η ( · ) is estimated. These conditions are typical for locally robust estimates and only require the non-parametric function η ( a, x ) to be estimable at faster than n -1 / 4 rates. This is easily verified for most non-parametric estimation methods such as kernel or series regression. Under these assumptions, we have the following analogue of Theorem 1, which we prove in Appendix A.2 .

Theorem 2. Under Assumptions 2(i) to 2(vi), the following hold:

- (i) Both ξ ∗ and ˆ ξ exist, the latter with probability approaching one.

<!-- formula-not-decoded -->

(iii) The L 2 error for the difference between g ( a, x ) and r ( a, x ) ᵀ ˆ ξ is bounded as

<!-- formula-not-decoded -->

4.1.2. Approximate Value Iteration. We can expand the estimation error ∥ ∥ ∥ ˆ h J -h ∥ ∥ ∥ 2 in terms of the non-parametric estimation errors ∥ ∥ ∥ Γ z [ ˆ h j -1 ] -h j ∥ ∥ ∥ 2 for j = 1 , . . . , J . In particular, since Γ z [ h ] = h and Γ z [ · ] is a β -contraction, we have

<!-- formula-not-decoded -->

Iterating the above gives

<!-- formula-not-decoded -->

Equation (4.1) can be considered a special case of error propagation (Munos and Szepesvári, 2008).

Recall that ˆ h 1 is an arbitrary initialization. It is thus straightforward to provide conditions under which ∥ ∥ ∥ h -ˆ h 1 ∥ ∥ ∥ 2 is bounded by some constant M 1 . As for the second term in (4.1), recall from the discussion in Section 3.2 that the minimization problem (3.10) corresponds to non-parametric estimation of Γ z [ ˆ h j -1 ] using the functional class F . Most machine learning methods come with guarantees on the non-parametric estimation rate ∥ ∥ ∥ Γ z [ ˆ h j -1 ] -ˆ h j ∥ ∥ ∥ 2 .

We now describe our assumptions for AVI. Let X denote the d -dimensional space of x , and define W γ, ∞ ( X ) as the Hölder ball with smoothness parameter γ :

<!-- formula-not-decoded -->

Assumption 3. (i) The domain, X , of x is compact and there exist M 0 , M &lt; ∞ such that | h | ∞ ≤ M 0 and h ( · , a ) ∈ W γ, ∞ ( X ) for each a .

(ii) ∣ ∣ ∣ ˆ h 1 ∣ ∣ ∣ ∞ ≤ M 0 and ∥ ∥ ∥ h -ˆ h 1 ∥ ∥ ∥ 2 ≤ M 1 for some M 1 &lt; ∞ .

(iii) | Γ z [ f ] | ∞ ≤ M 0 and Γ z [ f ]( · , a ) ∈ W γ, ∞ ( X ) for all a ∈ A and { f : | f | ∞ ≤ M 0 } .

(iv) The candidate class of functions F is such that | f | ∞ ≤ M 0 for all f ∈ F . Additionally, consider the non-parametric estimation problem

ˆ f = arg min ˜ f ∈F n -1 ∑ n i =1 ( y i -˜ f ( a i , x i )) 2 , where y i is compactly supported and E [ y i | a i , x i ] = f ( a i , x i ) for some f ∈ W γ, ∞ ( X ) . Then, uniformly over all f ∈ W γ, ∞ ( X ) , E [∥ ∥ ∥ f -ˆ f ∥ ∥ ∥ 2 ] ≤ Cn -c for constants C &lt; ∞ , c &gt; 0 independent of n , but which may depend on M,M 0 , γ .

Assumption 3(i) is a standard requirement in non-parametric estimation. The assumption of γ -Hölder continuity is taken from Farrell et al. (2021). Assumption 3(ii) is a mild condition on the initialization ˆ h 1 . Assumption 3(iii), which is novel to this paper, is a crucial smoothness condition requiring the operator Γ z [ · ]( · , a ) to map all bounded f onto W γ, ∞ ( X ). In Online Appendix B.2, we show that this is satisfied if z ( a, · ) and K ( x ′ | a, · ) are γ -Hölder continuous.

Assumption 3(iv) is a high-level condition on the machine learning (ML) method F . The requirement of bounded f implies that the ML method cannot diverge in the l ∞ sense, see Farrell et al. (2021) for a discussion of this in the context of multi-layer perceptrons (MLPs). The second part of Assumption 3(iv) implies that the ML method is able to non-parametrically approximate all functions in W γ, ∞ ( X ) at the rate of at least n -c . Most ML methods are proven to satisfy this. Consider, for instance, the class F of MLPs of width W and depth L ; MLPs and, more generally, Neural Networks are widely used

in RL. The results of Farrell et al. (2021) imply that for W /equivasymptotic n d 2( γ + d ) ln 2 n and L /equivasymptotic ln n ,

<!-- formula-not-decoded -->

Thus, Assumption 3(iv) is satisfied for MLPs. See Biau (2012) for related results on random forests.

Assumptions 3(iii) and 3(iv) imply that one can estimate Γ[ f ] for any | f | ∞ ≤ M 0 at the n -c rate, i.e., sup j E [∥ ∥ ∥ Γ z [ ˆ h j -1 ] -ˆ h j ∥ ∥ ∥ 2 ] ≤ Cn -c . Combined with (4.1), this proves:

Theorem 3. Suppose Assumptions 3(i) to 3(iv) hold. Then, for all n large enough,

<!-- formula-not-decoded -->

The first term in the expression for E [∥ ∥ ∥ h -ˆ h J ∥ ∥ ∥ 2 ] from Theorem 3 is the statistical rate of estimation of h . The second term is the numerical error, which is seen to decline exponentially with the number of iterations J . Setting J /equivasymptotic ln n will ensure the numerical error is smaller than the statistical rate of convergence. The number of iterations can be further reduced using a good initialization that makes M 1 small. For instance, ˆ h 1 can be the linear semi-gradient estimator. This is fast to compute and ensures M 1 = o p (1). Incidentally, Theorem 3 justifies the use of neural networks for batch RL; to the best of our knowledge this appears to be new even in the RL literature.

Turning to estimation of ˆ g , we again assume cross-fitting is employed as in Theorem 2, i.e., ˆ η is computed from one half of the data, and ˆ g is computed using AVI on the other half, taking ˆ η as given.

Assumption 4. (i) Assumptions 3(i) - 3(iv) hold after replacing ( h, ˆ h 1 , Γ z [ · ]) with ( g, ˆ g 1 , Γ e [ · ]) . (ii) ˆ g is estimated from a cross-fitting procedure. The conditional choice probability function satisfies η ( a, · ) ∈ W γ, ∞ ( X ) for all a , and η ( a, x ) ≥ δ &gt; 0 , where δ is independent of a, x . Additionally, ‖ η -ˆ η ‖ 2 2 = o p ( n -1 / 2 ) and with probability approaching one, ˆ η ( a, · ) ∈ W γ, ∞ ( X ) for all a .

Assumption 4(ii) is similar to Assumption 2(vi) with the additional requirement that η and ˆ η be α -Hölder continuous, the latter with probability approaching one. Note that ˆ η would be α -Hölder continuous if the non-parametric estimator consistently estimates not only η but also its first α derivatives.

Let ˜ g denote the fixed point of ˜ Γ e [ f ]( a, x ) := β E [ e ( a ′ , x ′ ; ˆ η ) + f ( a ′ , x ′ ) | a, x ]. Decompose

<!-- formula-not-decoded -->

Since ˆ η ( a, · ) ∈ W γ, ∞ ( X ), the first term, ‖ ˜ g -ˆ g J ‖ 2 , can be bounded by a similar rate as in Theorem 3 (recall that we take ˆ η as given under cross-fitting). As for the second term, ‖ g -˜ g ‖ 2 , observe that under Assumption 4,

<!-- formula-not-decoded -->

where the ' glyph[lessorsimilar] ' holds with probability approaching one, and uses ∂ η E [ ln η ( a ′ , x ′ ) | x ′ ] = 0, see the discussion following (3.7). Since ‖ η -ˆ η ‖ 2 2 = o p ( n -1 / 2 ), the above proves:

Theorem 4. Suppose Assumptions 4(i) to 4(ii) hold. Then, with probability approaching one,

<!-- formula-not-decoded -->

4.2. Estimation of structural parameters. Estimation of h ( a, x ) and g ( a, x ) is inherently non-parametric. This is because h ( a, x ) and g ( a, x ) are functions of two nonparametric terms: the choice probabilities η ( a, x ), and the transition densities K ( x ′ | a, x ). The TD estimators implicitly take both into account. Under discrete states, the firststep estimation error does not affect the rates of convergence of structural parameters (see Online Appendix B.1). With continuous states, however, the first-step non-parametric estimation error does affect the estimation of θ ∗ to a first order when using the PMLE criterion (3.9). This is because the estimates for K ( x ′ | a, x ) and θ ∗ are not orthogonal under a PMLE, which extends to the lack of orthogonality between the estimates for h ( a, x ) , g ( a, x ) and θ ∗ . Consequently, the PMLE estimator with plug-in values of ˆ h and ˆ g will converge at slower than parametric rates.

We can recover √ n -consistent estimation by adjusting the PMLE criterion to account for the first-stage estimation of h and g . Denote (˜ a , ˜ x ) := ( a, x, a ′ , x ′ ) and m ( a, x ; θ, h, g ) := ∂ θ ln π ( a, x ; θ, h, g ), where

<!-- formula-not-decoded -->

The PMLE estimator with plug-in estimates solves E n [ m ( a, x ; θ, ˆ h, ˆ g )] = 0, but this is not robust to estimation of h, g . Let

<!-- formula-not-decoded -->

denote the continuation value given ( a, x ). Also, define λ ( a, x ; θ ) as the fixed point of the 'backward' dynamic programming operator

<!-- formula-not-decoded -->

where ( a -′ , x -′ ) denotes the past actions and states preceding ( a, x ), and

<!-- formula-not-decoded -->

In Online Appendix B.3, we show that the locally robust moment corresponding to m ( a, x ; θ, h, g ) is given by

<!-- formula-not-decoded -->

The construction of the locally robust moment (4.4) is new. But it is infeasible since λ ( · ) , h ( · ) and g ( · ) are unknown. However, we can replace these quantities with consistent estimates. We have already described how to estimate h ( · ) , g ( · ). Let ˜ θ denote the plug-in estimator of θ ∗ using (3.9); note that ˜ θ consistently estimates θ ∗ but is not efficient. An estimator, ˆ λ ( · ), of λ ( · ) can then be obtained by applying either of our TD estimation methods on (4.2), with ˜ θ, ˆ h, ˆ g plugged in in place of θ, h, g . For instance, using AVI, we could obtain iterative approximations { ˆ λ ( j ) , j = 1 , . . . , J } for λ ( · ) using

<!-- formula-not-decoded -->

Plugging in ˆ λ ( · ) , ˆ h, ˆ g into (4.4), we obtain the feasible locally robust moment

<!-- formula-not-decoded -->

Based on the above, we can obtain a locally robust estimator, ˆ θ , as the solution to E n [ ζ n (˜ a , ˜ x ; θ, ˆ h, ˆ g )] = 0. We recommend obtaining this estimate using cross-fitting, see Section 4.2.1 for more details. Compared to the plug-in estimate (3.9), our locally robust estimator requires computation of λ ( · ), but even this can be avoided when linear semi-gradients are used for estimating h, g (see Online Appendix B.4). Solving E n [ ζ n (˜ a , ˜ x ; θ, ˆ h, ˆ g )] = 0 is also computationally easy; the correction term is a constant, and ∂ θ ζ n (˜ a , ˜ x ; θ, ˆ h, ˆ g ) = ∂ θ m ( a, x ; θ, ˆ h, ˆ g ) is negative definite (as the PMLE criterion is

concave), so solving this is no harder than solving the original moment condition without a correction term.

4.2.1. √ n -consistent estimation. We focus on the general construction of the locally robust estimator, ˆ θ , using (4.5). As mentioned in the previous sub-section, we advocate cross-fitting to obtain this estimator. This entails computing ˜ θ, ˆ λ, ˆ h, ˆ g using one half of the sample, say N 2 , and plugging them into the locally robust moment to compute ˆ θ as the solution to E (1) n [ ζ n (˜ a , ˜ x ; θ, ˆ h, ˆ g )] = 0, where E (1) n [ · ] is the empirical expectation using only observations from the other half of the sample N 1 . Following the analysis of Chernozhukov et al. (2022), it can be shown that this estimator has the same limiting distribution as the one based on (4.4). In particular, it achieves parametric rates of convergence. We state the regularity conditions below (for the remainder of this section we allow θ ∗ to be vector valued):

Assumption 5. (i) θ ∗ ∈ Θ , a compact set, and E [ ζ (˜ a , ˜ x ; θ, h, g )] = 0 ⇐⇒ θ = θ ∗ .

(ii) There exists a neighborhood, N , of θ ∗ such that uniformly over θ ∈ N and for ‖ ˜ h -h ‖ , ‖ ˜ g -g ‖ sufficiently small, ‖ ∂ θ ζ (˜ a , ˜ x ; θ, ˜ h, ˜ g ) -∂ θ ζ (˜ a , ˜ x ; θ ∗ , ˜ h, ˜ g ) ‖≤ d (˜ a , ˜ x ) ‖ θ -θ ∗ ‖ , where E [ d (˜ a , ˜ x )] &lt; ∞ . Furthermore, G := E [ ∂ θ ζ (˜ a , ˜ x ; θ ∗ , h, g )] is invertible.

<!-- formula-not-decoded -->

Assumption 5(i) implies θ ∗ is identified. Assumption 5(ii) is a mild regularity condition that is similar to Assumption 5 in Chernozhukov et al. (2022). Assumption 5(iii) is satisfied if ψ ( · ) is uniformly Lipschitz continuous in θ . Assumption 5(iv) follows from Theorems 1-4 under suitable conditions on the degree of smoothness of h, g . For instance, it is satisfied for AVI with neural networks if γ ≥ d . Assumption 5(v) requires λ to be estimable at faster than n -1 / 4 rates as well. If h, g are known, it is straightforward to derive n -1 / 4 rates as in Theorems 1-4. For plug-in estimation, we would need additional assumptions. For instance, we could employ three-way sample splitting (the first third of the sample is used to compute ˆ h, ˆ g , which are then plugged into the second third of the sample to estimate λ ), in which case Assumption 5(v) holds under the regularity condition

<!-- formula-not-decoded -->

We refer to Chernozhukov et al. (2018) for more details on three-way splitting.

We are now ready to state the main result of this section.

Theorem 5. Suppose that either Assumptions 1, 2 &amp; 5 or 3-5 hold. Then the estimator, ˆ θ of θ ∗ , based on (4.5) is √ n -consistent, and satisfies

<!-- formula-not-decoded -->

where V = ( G ᵀ Ω -1 G ) -1 , with Ω := E [ ζ (˜ a , ˜ x ; θ ∗ , h, g ) ζ (˜ a , ˜ x ; θ ∗ , h, g ) ᵀ ] .

The proof of the above theorem follows by verifying the regularity conditions of Chernozhukov et al. (2022, Theorem 9). Since these are more or less straightforward to verify given our previous results, we omit the details. For inference on ˆ θ , the covariance matrix V can be estimated as ˆ V = ( ˆ G ᵀ ˆ Ω -1 ˆ G ) -1 , where

<!-- formula-not-decoded -->

Chernozhukov et al. (2022) provide conditions under which ˆ V is consistent for V ; these are straightforward to translate to our context but we omit them for brevity. Alternatively, one could employ the bootstrap.

## 5. Estimation of dynamic discrete games

So far we have considered applications of our algorithm to single-agent models, where we have argued that there are substantial computational and statistical gains from using our procedure. These gains are magnified when extended to the estimation of dynamic discrete games.

Our setup is based on Aguirregabiria and Mira (2010). We assume a single MarkovPerfect-Equilibrium setup where multiple players i = 1 , 2 , . . . , N play against each other in M different markets. Each player chooses among A mutually exclusive actions to maximize an infinite horizon objective. We observe the state of play for T time periods, where both T and the number of players N are assumed fixed while M →∞ . Utility of the players in any time period is affected by the actions of all the others, and a set of states x that are observed by all players. The per-period utility is denoted by z i ( a i , a -i , x ) ᵀ θ ∗ + e i for each player i , for some finite-dimensional parameter θ ∗ , where a i denotes player i 's

action, a -i denotes the actions of all other players and e i is an idiosyncratic error term. As in Section 3, we take θ ∗ to be scalar to simplify the notation; all our results continue to hold for vector valued θ , as long as each dimension is treated separately. Evolution of the states in the next period is determined by the transition density K ( x ′ | a, x ) where a := ( a 1 , . . . , a N ) denotes the actions of all the players. We denote by x tm the state at market m in time period t, by a tm the vector of actions by all players at time t in market m , and by a itm the action of player i at time t in market m . We also let P i ( a i | x t ) denote the choice probability of player i taking action a i when the state is x t , and define e ( a i , x ) := γ -ln P i ( a i | x ).

As in the single agent case, the parameters θ ∗ can be obtained as solutions to the pseudo-likelihood function:

<!-- formula-not-decoded -->

where h i ( . ) and g i ( . ) are now player-specific, and given by

<!-- formula-not-decoded -->

In contrast to (2.1) in the single agent case, the expectation now averages over the actions of the other players as well.

Previous literature estimates θ ∗ using a two-step procedure: In the first step, the conditional choice probabilities P i ( a i | x t ) are calculated non-parametrically. These, along with estimates of K ( . ) are then used to recursively solve for h i ( . ) and g i ( . ) using equation (5.2). This step requires integrating over the actions of all the other players. Finally, given the estimated values of h i ( . ) and g i ( . ), the parameter θ is estimated through either pseudolikelihood (Aguirregabiria and Mira, 2007) or minimum distance estimation (Pesendorfer and Schmidt-Dengler, 2008).

By contrast, our algorithm is a straightforward extension of those suggested in earlier sections for single-agent models. Let ˆ η i ( a i , x ) denote a non-parametric estimate of the choice probabilities for player i and denote e ( a i , x ; ˆ η i ) = γ -ln ˆ η i ( a i , x ). We apply our TD methods on the recursion (5.2), separately for each player. The linear semi-gradient

estimates are given by ˆ h i ( a i , x ) = φ ( a i , x ) ᵀ ˆ ω i and ˆ g i ( a i , x ) = r ( a i , x ) ᵀ ˆ ξ , where

<!-- formula-not-decoded -->

and for any function f ( · ), we define

<!-- formula-not-decoded -->

Similarly, the AVI iterations for h i ( · ) , g i ( · ) are given by

<!-- formula-not-decoded -->

If the players are symmetric ( z i ( a i , a -i , x ) does not depend on player i ) we can obtain computationally faster and more precise estimates by pooling across players.

Importantly, neither of the estimation strategies (5.3) nor (5.5) require partialling out other players' actions, leading to a tremendous reduction of computation. Intuitively, the procedures take expectations over other players' actions 'internally' using the empirical distribution. The non-parametric estimates ˆ h i , ˆ g i can be plugged into the PMLE criterion (5.1) to obtain an estimate for θ as

<!-- formula-not-decoded -->

It is straightforward to construct a locally robust estimator for θ in analogy with that for single-agent models. We describe this in Online Appendix B.5.

By the same reasoning as in Online Appendix B.1, it possible to show that with discrete states, h i ( . ) and g i ( . ) are numerically identical to the estimates obtained by plugging in cell estimates ˆ P j ( ·| x ) and ˆ K ( . ) in (5.2). This implies the psuedo-likelihood with plugin estimates for h ( . ) and g ( . ) is not efficient even with discrete states, as discussed by Aguirregabiria and Mira (2007). However the values of h ( . ) and g ( . ) can be plugged into other, more efficient objectives, such as our locally robust estimator or the minimum distance estimator of Pesendorfer and Schmidt-Dengler (2008). With continuous states, one would need to employ locally robust corrections even for the minimum distance

estimator to recover parametric rates of convergence for θ . The locally robust correction term can be constructed in a similar way as that for the PMLE criterion.

## 6. Simulations

Werun Monte Carlo simulations to test our estimation methods. We start by presenting results for a DDC model using both the linear semi-gradient approach, and the AVI approach with random forests as the prediction method. For each of these methods, we compare findings from the locally robust version of our estimators to those obtained without correction. Our simulations for this part are based on the firm entry problem described in Aguirregabiria and Magesan (2018).

In a second set of Monte Carlo simulations, we test our estimation method for dynamic discrete games. For this part, we present results from the linear semi-gradient approach using different sets of basis functions to approximate the value function terms, and employ the cross-validation method described in Section 3.3 to select the preferred set of basis functions. Our simulations are based on the dynamic firm entry game used in Aguirregabiria and Mira (2007).

Finally, we run additional simulations based on the famous Rust (1987) bus engine replacement problem (see Online Appendix B.7.2). Using this model, we also provide simulation results for a case with permanent unobserved heterogeneity.

6.1. Firm entry problem. Consider the following dynamic firm entry problem described in Aguirregabiria and Magesan (2018). A firm decides whether to enter ( a t = 1) or not enter ( a t = 0) in a market for t = 1 , ..., T time periods. The payoff when entering is given by Π t = V P t -FC t -EC t + ε t , where V P t , FC t and EC t denote the firm's variable profit, fixed cost and entry cost, and ε t is a transitory shock that follows a logistic distribution. Variable profit is given by V P t = ( θ V P 0 + θ V P 1 z 1 t + θ V P 2 z 2 t ) exp( ω t ), where ω t denotes the firm's productivity shock, and z 1 t , z 2 t are exogenous state variables affecting the price-cost margin in the market. The fixed cost is given by FC t = θ FC 0 + θ FC 1 z 3 t , and the entry cost is given by EC t = ( θ EC 0 + θ EC 1 z 4 t )(1 -a t -1 ), where z 3 t , z 4 t are further exogenous state variables, and a t -1 denotes the entry decision in period t -1 which is an endogenous state variable. The payoff of not entering is normalized to zero. The parameters θ ∗ ≡ { θ V P 0 , θ V P 1 , θ V P 2 , θ FC 0 , θ FC 1 , θ EC 0 , θ EC 1 } are the structural parameters of interest. The exogenous state variables z jt and ω t are continuous and follow AR(1) processes,

where z jt = γ j 0 + γ j 1 z jt -1 + e jt , and ω t = γ ω 0 + γ ω 1 ω t -1 + e ωt . The error terms e jt , e ωt follow normal N (0 , 1) distributions. The discount factor β is 0 . 95.

To carry out the simulations, we choose values for the structural parameters θ ∗ ( θ V P 0 = 0 . 5, θ V P 1 = 1 . 0, θ V P 2 = -1 . 0, θ FC 0 = 1 . 5, θ FC 1 = 1 . 0, θ EC 0 = 1 . 0, θ EC 1 = 1 . 0) and for the autoregressive processes of z jt and ω t ( γ j 0 = 0 . 0, γ j 1 = 0 . 6, γ ω 0 = 0 . 2, γ ω 1 = 0 . 6), and discretize the exogenous state variables to obtain a transition matrix with a 6-point support following Tauchen (1986). The resulting dimension of the state space is 2 × 6 5 = 15 , 552 . The discretization of the support is for simulations only; our estimation algorithms treat these variables as continuous and do not require any prior knowledge of how they evolve (the knowledge of AR(1) dynamics is also not used). We iterate on the value function to obtain the vector of choice probabilities for each combination of the states, and use these to derive the steady-state distribution of the state variables. Using the steady-state distributions, we generate data for 3 , 000 firms, with T = 2 time periods.

6.1.1. Simulation results - firm entry problem. Panel A of Table 1 shows results for 1000 simulations using the linear semi-gradient method. Panel B of Table 1 presents results of 250 simulations using the AVI method. Each round of the simulations begins by generating new data, where the first-period state variables are drawn from the steady-state distribution. For the results in Panel A, we parameterize h ( a, x ) and g ( a, x ) using a second order polynomial in the state variables. 4 For the results in Panel B, we approximate h ( a, x ) and g ( a, x ) using a random forest, where we iterate the AVI procedure 70 times for each round of the simulations. For both the linear semi-gradient and the AVI methods, we estimate the choice probabilities η that enter e ( a ′ , x ′ ; η ) using a logit model where the explanatory variables are the same as those used as basis functions in Panel A.

We present results generated with and without the locally robust correction. For the results without correction, we obtain estimates for θ ∗ using (3.9). To generate the locally robust estimates, we use moment equation (B.11) for the linear semi-gradient method, and moment equation (4.5) for the AVI method where we employ a random forest to derive an estimate for the λ ( a, x, ˜ θ ) term contained in the locally robust moment. As before, the AVI method for estimation of λ ( · ) iterated 70 times. We also use the sample

4 For the ω 's relating to parameters θ V P 0 , θ V P 1 , θ V P 2 , θ FC 0 , θ FC 1 , and for ξ , the terms include a constant, the exogenous state variables and their interactions up to a second order, the player's binary choice a t and the interactions of a t with all terms in the exogenous states. Given the set-up of the model, we treat the interactions z 1 t exp( ω t ) and z 2 t exp( ω t ) instead of z 1 t and z 2 t themselves as state variables. In addition to the terms included above, the ω 's relating to parameters θ EC 0 and θ EC 1 also contain the terms (1 -a t -1 ) and (1 -a t -1 ) z 4 t , respectively. The total number of terms included is 42 (43 for θ EC 0 and θ EC 1 ).

splitting method described in Section 4.2.1 for the locally robust estimators, and we obtain the final ˆ θ as weighted average of the θ ∗ estimates from the two samples. 5

Panel A of Table 1 shows that our linear semi-gradient results are closely centered around the true values. While the locally robust estimator should in theory be preferable, we find that it produces results which are similar and if anything have slightly higher mean squared error than the non-robust version of our algorithm. In fact, there is very little bias, and the distribution of the estimates under the non-robust version is already very close to normal, see Online Appendix B.7 for the plots of the finite sample distributions. On the other hand, locally robust methods are associated with higher variance due to sample splitting. So there appears to be no gain from the locally robust method here.

Panel B of Table 1 shows that the AVI method produces estimates with slightly higher small sample bias than the linear semi-gradient. In this case, the locally robust method is more useful as it decreases the bias significantly, especially for estimation of the entry cost parameters.

6.1.2. Comparison with existing methods. We compare our findings from Section 6.1 to those that would be obtained with a standard CCP estimator where the state variables are discretized and the transition and choice probabilities are estimated using cell values.

We start our simulations by generating data as outlined in Section 6.1. We then discretize the state space by creating dummy variables for each state variable z 1 t , z 2 t , z 3 t , z 4 t and exp( ω t ). To make the estimation feasible, the state space needs to be restricted further. A common approach is to use K-means clustering here, but this is not appropriate in the given simulation setting where the state variables are independent by construction. We therefore restrict the state space grid by combining variables z 1 t and z 2 t into a binary variable taking value one whenever both individual dummies take value one. The resulting state space consists of four binary variables, implying 16 cells in the state space grid. We try alternative feasible ways of discretizing the state space, but find that these do not lead to improvements over the chosen method. We run 1000 simulations, and the results are shown in Table 2.

It can be seen that, compared to the results from Table 1, the CCP estimator leads to substantially larger bias in some of the estimated parameters. Column (4) shows that the

5 We run our simulations on a MacBook Pro with an M1 chip and 16 GB of RAM. The computation time for one estimation round is about 4 seconds for the linear semi-gradient method without locally robust correction, and 14 seconds with locally robust correction. For the AVI method, the computation time is about 90 seconds without locally robust correction, and 315 seconds with locally robust correction.

Table 1. Simulations: Firm entry problem

|                         |         | not locally robust   | not locally robust   | locally robust   | locally robust   |
|-------------------------|---------|----------------------|----------------------|------------------|------------------|
|                         | DGP (1) | TDL (2)              | MSE (3)              | TDL (4)          | MSE (5)          |
| A. Linear semi-gradient |         |                      |                      |                  |                  |
| θ V P 0                 | 0.5     | 0.4898 (0.0781)      | 0.0062               | 0.5376 (0.0983)  | 0.0111           |
| θ V P 1                 | 1.0     | 0.9883 (0.0793)      | 0.0064               | 1.0650 (0.1060)  | 0.0154           |
| θ V P 2                 | -1.0    | -0.9908 (0.0831)     | 0.0070               | -1.0675 (0.1107) | 0.0168           |
| θ FC 0                  | 1.5     | 1.4905 (0.1521)      | 0.0232               | 1.5745 (0.1862)  | 0.0402           |
| θ FC 1                  | 1.0     | 0.9877 (0.1348)      | 0.0183               | 1.0446 (0.1703)  | 0.0309           |
| θ EC 0                  | 1.0     | 0.9949 (0.1002)      | 0.0101               | 1.0253 (0.1141)  | 0.0137           |
| θ EC 1                  | 1.0     | 0.9978 (0.1654)      | 0.0273               | 1.0529 (0.1917)  | 0.0395           |
| B. AVI                  |         |                      |                      |                  |                  |
| θ V P 0                 | 0.5     | 0.4183 (0.0823)      | 0.0134               | 0.3903 (0.0896)  | 0.0200           |
| θ V P 1                 | 1.0     | 1.1043 (0.0810)      | 0.0174               | 1.0608 (0.0914)  | 0.0120           |
| θ V P 2                 | -1.0    | -1.1080 (0.0856)     | 0.0189               | -1.0663 (0.1006) | 0.0145           |
| θ FC 0                  | 1.5     | 1.5453 (0.1614)      | 0.0280               | 1.4471 (0.1751)  | 0.0334           |
| θ FC 1                  | 1.0     | 1.1209 (0.1384)      | 0.0337               | 1.0707 (0.1645)  | 0.0319           |
| θ EC 0                  | 1.0     | 1.1388 (0.1104)      | 0.0314               | 1.0430 (0.1495)  | 0.0241           |
| θ EC 1                  | 1.0     | 1.2700 (0.2043)      | 0.1145               | 1.1466 (0.2519)  | 0.0847           |

Notes: The table reports results with 3000 firms. Panel A is based on 1000 simulations, Panel B on 250 simulations. Column (1) shows the true parameter values in the model. Columns (2) and (4) report the empirical mean and standard deviation for the estimated parameters. Columns (2)-(3) are based on the estimation method without correction function, columns (4)-(5) report results using the locally robust estimator. The mean squared errors are reported in columns (3) and (5), respectively.

corresponding mean squared errors are large and generally exceed those obtained using our estimators. This is particularly true for parameters θ V P 1 and θ V P 2 , highlighting the challenges inherent in the discretization of continuous state spaces. Overall, the average

Table 2. Simulations: Firm entry problem - Comparison with standard CCP

|          | DGP (1)   | TDL (2)          | bias (3)   | MSE (4)   |
|----------|-----------|------------------|------------|-----------|
| CCP with |           |                  |            |           |
| θ V P 0  | 0.5       | 0.1518 (0.2303)  | -0.3482    | 0.1742    |
| θ V P 1  | 1.0       | 0.7642 (0.5613)  | -0.2358    | 0.3704    |
| θ V P 2  | -1.0      | -0.3826 (0.2428) | 0.6174     | 0.4401    |
| θ FC 0   | 1.5       | 1.4139 (0.1366)  | -0.0861    | 0.0261    |
| θ FC 1   | 1.0       | 0.8587 (0.1431)  | -0.1413    | 0.0404    |
| θ EC 0   | 1.0       | 0.7788 (0.0894)  | -0.2212    | 0.0569    |
| θ EC 1   | 1.0       | 0.9148 (0.1854)  | -0.0852    | 0.0416    |

Notes: The table reports results of 1000 simulations with 3000 firms. Column (1) shows the true parameter values in the model. Column (2) reports the empirical mean and standard deviation for the estimated parameters. Column (3) reports the average bias in the estimated parameters. The mean squared errors are reported in column (4).

mean squared error increases from 0 . 01 -0 . 04 across all parameters in Table 1 to 0 . 16 in Table 2. These results show that our estimation methods lead to important improvements over existing methods in models with as few as five continuous state variables.

6.2. Firm entry game. Consider the following firm market entry game, which is similar to that described in Aguirregabiria and Mira (2007). There are i = 1 , ..., 5 firms (players), and we observe their decision to enter ( a itm = 1) or not enter ( a itm = 0) in m = 1 , ..., M different markets for t = 1 , ..., T time periods. Denote a firm's action by j ∈ { 1 , 0 } . The payoff of each firm i is affected by the decision of all the other firms whether to enter, as well as firm i 's previous-period entry decision. Current-period profits when entering are given by

/negationslash

<!-- formula-not-decoded -->

where ln( S tm ) is a measure of consumer market size of market m in period t , and ε itm is a transitory shock that follows a logistic distribution. We assume that ln( S tm ) is

continuous and follows an AR(1) process, where the parameters are the same across markets: ln( S tm ) = α + λ ln( S ( t -1) m ) + u tm . The error term u tm is assumed to follow a normal N (0 , 1) distribution. The profit of not entering is normalized to zero, and the discount factor β is 0 . 95. The parameters θ ∗ ≡ { θ RS , θ RN , θ FC , θ EC } are the structural parameters of interest. The state variables in this setting are given by the current market demand variable S tm , as well as the vector of all firms' previous entry decisions a ( t -1) m = { a i ( t -1) m : i = 1 , ..., 5 } .

To carry out the simulations, we choose values for the structural parameters θ ∗ ( θ RS = 1 , θ RN = 1 , θ FC = 1 . 7 , θ EC = 1), and for the autoregressive process for log market size ( α = 1 . 5, λ = 0 . 5). Wediscretize ln( S tm ) and obtain a transition matrix for the discretized variable with a 10-point support following the method by Tauchen (1986). As in the Monte Carlo experiments for the firm entry problem in Section 6.1, the discretization is for simulations of the data only and we treat the state variables as continuous in our estimations. We then solve for the Markov-Perfect-Equilibrium of the game. This is done by finding the firms' conditional value functions ν j ( S tm , a ( t -1) m ) for each of the 2 5 × 10 = 320 possible combinations of the state variables through repeated iteration, and using these to derive the equilibrium choice probabilities p ( S tm , a ( t -1) m ). Based on the equilibrium probabilities, we compute the equilibrium distribution of state variables. Using the equilibrium distributions, we generate data for 1000 and for 3000 markets, with T = 2 time periods.

6.2.1. Simulation results - firm entry game. We present the results of 1000 simulations based on the linear semi-gradient method, without employing the locally robust correction. Each round of the simulations begins by generating new data, where the first-period state variables are drawn from the steady-state distribution. In order to assess the sensitivity of our algorithm to different specifications for the basis functions, we parameterize h ( a, x ) and g ( a, x ) using different sets of polynomials in the state variables. In particular, we show results where h ( a, x ) and g ( a, x ) are approximated using a second, third or fourth order polynomial. 6 For all simulations, the choice probabilities η that enter e ( a ′ , x ′ ; η ) are estimated using individual logit models for each firm, where we use a third order polynomial in the state variables as explanatory variables. We then estimate the parameters

/negationslash

6 For the ω ′ s relating to parameters θ RS , θ RN , θ FC and for ξ , the terms include a constant, terms up to the second/third/fourth order in the state variables ln( S tm ) and ln(1 + ∑ j = i a j ( t -1) m ), the player's binary choice a itm and the interactions of a itm with all terms in the state variables. The total number of terms is 12 / 20 / 30. In addition to these terms, the ω ′ s relating to parameter θ EC also contain the term (1 -a i ( t -1) m ) a itm .

ω and ξ using equation 5.3. 7 Finally, we obtain estimates for the θ ∗ parameters as the solutions to the pseudo-likelihood function 5.1. 8

The results are shown in Table 3. Panels A, B and C present simulations for the same dataset using different basis functions to parameterize the value function terms h ( a, x ) and g ( a, x ). Column (2) shows that even with 1000 markets our algorithm produces parameter estimates that are closely centered around the true values. The results are generally similar across Panels A to C, although the bias and mean squared error tends to be lowest for the second order polynomial, and highest for the fourth order polynomial. This is especially the case for the parameter on the number of market entrants, θ RN . To assess these differences formally, we use the cross-validation procedure described in Section 3.3. The procedure is applied to ten random samples of market size 1000, and we find that the TD error criterion consistently selects the second order polynomial as the optimal set of basis functions. This suggests that the proposed cross-validation method provides useful guidance when choosing a set of basis functions in practice.

In a similar version of the firm entry game, Aguirregabiria and Mira (2007) use the NPL algorithm and derive results comparable to ours. Our simulations rely on slightly larger sample sizes, but note that these differences are likely due to the fact that our algorithm implicitly estimates the transition densities for market size and therefore does not rely on the true transition density to be used in the estimation. For a direct comparison of our results with those obtained using the NPL algorithm, one would need to obtain a non-parametric estimate of the transition density which is not trivial in practice.

As expected, columns (4) and (5) show that increasing the market size generally reduces the small sample bias in the estimated parameters, and leads to a fall in the empirical standard deviations. In addition to being smaller, the mean squared error across Panels A to C is also more similar across the three sets of basis functions. As before, we employ the cross-validation method described above to compare these specifications more formally. In line with the estimation results, we find that all three polynomials now produce very similar sets of mean squared TD error, even though the second order polynomial continues to be the one that is selected by the criterion. 9 While we view this as further evidence that

7 Given the symmetric set-up of the game, we pool the data across players in this application.

8 We run our simulations on a MacBook Pro with an M1 chip and 16 GB of RAM. The approximate computation times for one estimation round are: 1.5 seconds (second order polynomial, 1000 markets);

3.7 seconds (second order polynomial, 3000 markets); 2.5 seconds (third order polynomial, 1000 markets);

6.7 seconds (third order polynomial, 3000 markets); 4 seconds (fourth order polynomial, 1000 markets); 12 seconds (fourth order polynomial, 3000 markets).

9 As before, we compute the TD criterion for ten random samples.

Table 3. Simulations: Firm entry game - Linear semi-gradient

|                         | DGP (1)   | TDL (2)         | MSE (3)      | TDL (4)         | MSE (5)      |
|-------------------------|-----------|-----------------|--------------|-----------------|--------------|
| A. 2nd order polynomial |           | 1000 markets    | 1000 markets | 3000 markets    | 3000 markets |
| θ RS (market size)      | 1.0       | 0.9718 (0.1598) | 0.0263       | 0.9847 (0.0895) | 0.0082       |
| θ RN (n. of entrants)   | 1.0       | 0.8962 (0.5301) | 0.2915       | 0.9586 (0.2899) | 0.0857       |
| θ FC (fixed cost)       | 1.7       | 1.7225 (0.2942) | 0.0870       | 1.6920 (0.1617) | 0.0262       |
| θ EC (entry cost)       | 1.0       | 1.0191 (0.0621) | 0.0042       | 1.0226 (0.0353) | 0.0018       |
| B. 3rd order polynomial |           | 1000 markets    | 1000 markets | 3000 markets    | 3000 markets |
| θ RS (market size)      | 1.0       | 0.9149 (0.1466) | 0.0287       | 0.9648 (0.0867) | 0.0088       |
| θ RN (n. of entrants)   | 1.0       | 0.6905 (0.4792) | 0.3251       | 0.8862 (0.2794) | 0.0909       |
| θ FC (fixed cost)       | 1.7       | 1.7815 (0.2721) | 0.0806       | 1.7135 (0.1573) | 0.0249       |
| θ EC (entry cost)       | 1.0       | 1.0173 (0.0622) | 0.0042       | 1.0219 (0.0353) | 0.0017       |
| C. 4th order polynomial |           | 1000 markets    | 1000 markets | 3000 markets    | 3000 markets |
| θ RS (market size)      | 1.0       | 0.8642 (0.1318) | 0.0358       | 0.9457 (0.0845) | 0.0101       |
| θ RN (n. of entrants)   | 1.0       | 0.5075 (0.4222) | 0.4206       | 0.8166 (0.2705) | 0.1067       |
| θ FC (fixed cost)       | 1.7       | 1.8338 (0.2513) | 0.0810       | 1.7337 (0.1530) | 0.0245       |
| θ EC (entry cost)       | 1.0       | 1.0159 (0.0620) | 0.0041       | 1.0213 (0.0352) | 0.0017       |

Notes: The table reports results for 1000 simulations. Panels A, B and C use different sets of basis functions to parameterize h ( a, x ) and g ( a, x ). Column (1) shows the true parameter values in the model. Columns (2) and (4) report the empirical mean and standard deviations for the estimated parameters, based on a sample of 1000 and 3000 markets, respectively. The mean squared errors are reported in columns (3) and (5). All results are based on the estimation method without correction function.

the proposed cross-validation method can provide useful guidance to choose a suitable set of basis functions, more importantly, the small differences in the results across panels A to C also suggest that our methods prove fairly robust to this choice in practice.

## 7. Conclusions

We propose two new estimators for DDC models which overcome previous computational and statistical limitations by combining traditional CCP estimation approaches with the idea of TD learning from the RL literature. The first approach, linear semigradient, makes use of simple matrix inversion techniques, is computationally very cheap and therefore fast. The second approach, Approximate Value Iteration, can be easily combined with any machine learning method devised for prediction. Unlike previous estimation methods, our methods are able to easily handle continuous and/or high-dimensional state spaces in settings where a finite dependence property does not hold. This is of particular importance for the estimation of dynamic discrete games. We also propose a locally robust estimator to account for the non-parametric estimation in the first stage. We prove the statistical properties of our estimator and show that it is consistent and converges at parametric rates. A range of Monte Carlo simulations using a dynamic firm entry problem, a dynamic firm entry game and a version of the famous Rust (1987) engine replacement problem show that the proposed algorithms work well in practice.

## References

- D. Ackerberg, X. Chen, J. Hahn, and Z. Liao, 'Asymptotic efficiency of semiparametric two-step gmm,' Review of Economic Studies , vol. 81, no. 3, pp. 919-943, 2014.
- V. Aguirregabiria and A. Magesan, 'Solution and estimation of dynamic discrete choice structural models using euler equations,' Working paper , 2018.
- V. Aguirregabiria and P. Mira, 'Swapping the nested fixed point algorithm: A class of estimators for discrete markov decision models,' Econometrica , vol. 70, no. 4, pp. 1519-1543, 2002.
4. --, 'Sequential estimation of dynamic discrete games,' Econometrica , vol. 75, no. 1, pp. 1-53, 2007.
5. --, 'Dynamic discrete choice structural models: A survey,' Journal of Econometrics , vol. 156, no. 1, pp. 38-67, 2010.
- M. Almagro and T. Domınguez-Iino, 'Location sorting and endogenous amenities: Evidence from amsterdam,' NYU, mimeograph , 2019.
- P. Arcidiacono and J. B. Jones, 'Finite mixture distributions, sequential likelihood and the em algorithm,' Econometrica , vol. 71, no. 3, pp. 933-946, 2003.

- P. Arcidiacono and R. A. Miller, 'Conditional choice probability estimation of dynamic discrete choice models with unobserved heterogeneity,' Econometrica , vol. 79, no. 6, pp. 1823-1867, 2011.
- P. Arcidiacono, P. Bayer, F. A. Bugni, and J. James, 'Approximating high-dimensional dynamic models: Sieve value function iteration,' in Structural Econometric Models . Emerald Group Publishing Limited, 2013, pp. 45-95.
- P. Bajari, C. L. Bankard, and J. Levin, 'Estimating dynamic models of imperfect competition,' Econometrica , vol. 75, no. 5, pp. 1331-1370, 2007.
- P. J. Barwick and P. A. Pathak, 'The costs of free entry: an empirical study of real estate agents in greater boston,' The RAND Journal of Economics , vol. 46, no. 1, pp. 103-145, 2015.
- G. Biau, 'Analysis of a random forests model,' The Journal of Machine Learning Research , vol. 13, pp. 1063-1095, 2012.
- V. Chernozhukov, W. K. Newey, and R. Singh, 'Learning l2 continuous regression functionals via regularized riesz representers,' arXiv preprint arXiv:1809.05224 , vol. 8, 2018.
- V. Chernozhukov, J. C. Escanciano, H. Ichimura, W. K. Newey, and J. M. Robins, 'Locally robust semiparametric estimation,' Econometrica , vol. 90, no. 4, pp. 15011535, 2022.
- A. P. Dempster, N. M. Laird, and D. B. Rubin, 'Maximum likelihood from incomplete data via the em algorithm,' Journal of the Royal Statistical Society. Series B (Methodological) , vol. 39, no. 1, pp. 1-38, 1977.
- M. H. Farrell, T. Liang, and S. Misra, 'Deep neural networks for estimation and inference,' Econometrica , vol. 89, no. 1, pp. 181-213, 2021.
- G. Hall, G. J. Hitsch, G. Pauletto, and J. Rust, 'A comparison of discrete and parametric approximation methods for continuous-state dynamic programming problems,' 2000.
- V. J. Hotz and R. A. Miller, 'Conditional choice probabilities and the estimation of dynamic models,' The Review of Economic Studies , vol. 60, no. 3, pp. 497-529, 1993.
- V. J. Hotz, R. A. Miller, S. Sanders, and J. Smith, 'A simulation estimator for dynamic models of discrete choice,' The Review of Economic Studies , vol. 61, no. 2, pp. 265-289, 1994.
- C. Johnson, 'Positive definite matrices,' The American Mathematical Monthly , vol. 77, no. 3, pp. 259-264, 1970.

- M. Kalouptsidi, 'Time to build and fluctuations in bulk shipping,' American Economic Review , vol. 104, no. 2, pp. 564-608, 2014.
2. --, 'Detection and impact of industrial subsidies: The case of chinese shipbuilding,' The Review of Economic Studies , vol. 85, no. 2, pp. 1111-1158, 2018.
- H. Kasahara and K. Shimotsu, 'Pseudo-likelihood estimation and bootstrap inference for structural discrete marvok decision models,' Journal of Econometrics , vol. 146, no. 1, pp. 92-106, 2008.
- M. P. Keane and K. I. Wolpin, 'The solution and estimation of discrete choice dynamic programming models by simulation and interpolation: Monte carlo evidence,' the Review of economics and statistics , pp. 648-672, 1994.
- S. Lange, T. Gabel, and M. Riedmiller, 'Batch reinforcement learning,' in Reinforcement learning . Springer, 2012, pp. 45-73.
- V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al. , 'Human-level control through deep reinforcement learning,' nature , vol. 518, no. 7540, pp. 529-533, 2015.
- R. Munos and C. Szepesvári, 'Finite-time bounds for fitted value iteration.' Journal of Machine Learning Research , vol. 9, no. 5, 2008.
- W. K. Newey, 'The asymptotic variance of semiparametric estimators,' Econometrica , vol. 62, no. 6, pp. 1349-1382, 1994.
9. --, 'Convergence rates and asymptotic normality for series estimators,' Journal of econometrics , vol. 79, no. 1, pp. 147-168, 1997.
- M. Pesendorfer and P. Schmidt-Dengler, 'Asymptotic least squares estimators for dynamic games,' Review of Economic Studies , vol. 75, no. 3, pp. 901-928, 2008.
- J. Rust, 'Optimal replacement of gmc bus engines: An empirical model of harold zurcher,' Econometrica , vol. 55, no. 5, pp. 999-1033, 1987.
- V. Semenova, 'Machine learning for dynamic models of imperfect information and semiparametric moment inequalities,' arXiv preprint arXiv:1808.02569 , 2018.
- R. S. Sutton, 'Learning to predict by the methods of temporal differences,' Machine learning , vol. 3, no. 1, pp. 9-44, 1988.
- R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction , 2nd ed. MIT Press, Cambridge, MA, 2018.
- G. Tauchen, 'Finite state markov-chain approimations to univariate and vector autoregressions,' Economics Letters , vol. 20, no. 2, pp. 177-181, 1986.

- J. N. Tsitsiklis and B. Van Roy, 'An analysis of temporal-difference learning with function approximation,' IEEE Transactions on Automatic Control , vol. 42, no. 5, pp. 674-690, 1997.

## Appendix A. Proofs of main results

In what follows we drop the functional argument ( a, x ) when the context is clear and denote f ′ ≡ f ( a ′ , x ′ ) for different functions f .

Lemma 1. There exists a unique fixed point to the operator P φ Γ z . If Assumption 1(i) holds, this fixed point is given by φ ᵀ ω ∗ , where ω ∗ is such that

<!-- formula-not-decoded -->

Proof. First, note that Γ z , and therefore P φ Γ z , are both contraction maps with the contraction factor β . This implies P φ Γ z has a unique fixed point. Clearly, this fixed point must lie in the space L φ . Let us denote this as φ ᵀ ω ∗ . Now, for any function f ∈ L φ ,

<!-- formula-not-decoded -->

Since φ ᵀ ω ∗ is the fixed point, we must have

<!-- formula-not-decoded -->

But φ is linearly independent and E [ φφ ᵀ ] -1 is non-singular, by Assumption 1(i). Hence it must be the case that E [ φ ( z + βφ ′ ᵀ ω ∗ -φ ᵀ ω ∗ )] = 0 . glyph[square]

For the next lemma, we use the following definition: a square, possibly asymmetric, matrix A is said to be negative definite with the coefficient ¯ λ ( A ) if

<!-- formula-not-decoded -->

For a symmetric negative-definite matrix, ¯ λ ( A ) = maxeig( A ), where maxeig( · ) is the maximal eigenvalue. We can similarly define a positive-definite matrix with the coefficient λ ( A ). If the latter is symmetric, then λ ( A ) = mineig( A ). Note that under our definition, if A is negative definite, it is also invertible. This holds even if the matrix is asymmetric, see e.g Johnson (1970).

Lemma 2. Under Assumption 1(i), the matrix A := E [ φ ( βφ ′ -φ ) ᵀ ] is negative definite with ¯ λ ( A ) ≤ -(1 -β ) λ ( E [ φφ ᵀ ]) , and is therefore invertible.

Proof. The proof is adapted from Tsitsiklis and van Roy (1997). Recall the definition of φ ᵀ ω ∗ as the fixed point to P φ Γ z [ · ] from Lemma 1. We now show that

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

since the second expression in the first equation is 0. Now,

<!-- formula-not-decoded -->

where the last equality holds since E [ φ ( I -P φ )[ f ]] = 0 for all f . We thus have

<!-- formula-not-decoded -->

Since P φ Γ z [ · ] is a contraction mapping with contraction factor β , it follows

<!-- formula-not-decoded -->

In view of the above,

<!-- formula-not-decoded -->

This completes the proof of the lemma.

Lemma 3. Suppose that Assumption 1(i) holds. Then,

<!-- formula-not-decoded -->

glyph[square]

Proof. Recall that h ( · , · ) is the unique fixed point of Γ z , and φ ᵀ ω ∗ is the unique fixed point of P φ Γ z . The operator Γ z is a contraction mapping with contraction factor β . Furthermore, the projection operator P φ is linear, and ‖ P φ [ f ] ‖ 2 ≤ ‖ f ‖ 2 for any function f . Thus

<!-- formula-not-decoded -->

Rearranging the above expression proves the desired claim.

glyph[square]

For the proofs of Theorems 1-2, we work within a more general setting than in the main text, by letting the distribution of ( a it , x it ) be time-varying. Let P t denote the population distribution of ( a, x ) at time t . Also, let P denote the probability distribution of the process { ( a 1 , x 1 ) , . . . , ( a T , x T ) } . Note that P ≡ P 1 × · · · × P T . We will denote E [ · ] as the expectation over P . Furthermore, we use the o p ( · ) and O p ( · ) notations to denote convergence in probability, and bounded in probability, respectively, under the probability distribution P .

We also need to extend the definitions of P and E [ · ]: Let P denote the relative frequency of occurrence of ( a, x, a ′ , x ′ ) in the data as n → ∞ . Let E [ · ] denote the corresponding expectation over P . Note that P is different from P since the latter is the distribution of ( a, x, a, x ′ ) after dropping the time index. However, the two are related since for any function f , we have E [ f ( a, x, a ′ , x ′ )] = ( T -1) -1 ∑ T -1 t =1 E [ f ( a it , x it , a it +1 , x it +1 )]. These updated definitions of P and E [ · ] are applicable wherever these notations are used in the main text.

Note that due to the Markov process assumption, the conditional distribution P ( a t +1 , x t +1 | a t , x t ) is always independent of t (indeed, one could always consider t as also a part of x ). Hence, P ( a ′ , x ′ | a, x ) ≡ P ( a t +1 , x t +1 | a t , x t ) and E [ f ( a ′ , x ′ ) | a, x ] ≡ E [ f ( a t +1 , x t +1 ) | a t , x t ] for all t . Also note that time stationarity of ( a it , x it ), if it holds, implies P t ≡ P and E t [ · ] ≡ E [ · ] for all t .

A.1. Proof of Theorem 1. Lemma 1 implies ω ∗ exists. To prove that ˆ ω exists, it suffices to show that ˆ A := E n [ φ ( βφ ′ -φ ) ᵀ ] is invertible with probability approaching one. Recall that using our notation, ˆ A = ( n ( T -1)) -1 ∑ i ∑ T -1 t =1 φ it ( βφ it +1 -φ it ) ᵀ , while A =

( T -1) -1 ∑ T -1 t =1 E [ φ it ( βφ it +1 -φ it ) ᵀ . Wecan thus write ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ ≤ ( T -1) -1 ∑ T -1 t =1 ∣ ∣ ∣ ˆ A t -A t ∣ ∣ ∣ , where ˆ A t := n -1 ∑ i φ it ( βφ it +1 -φ it ) ᵀ and A t := E [ φ it ( βφ it +1 -φ it ) ᵀ ]. By Assumption 1(ii), | φ ( a, x ) | ∞ ≤ M independent of k φ , so

<!-- formula-not-decoded -->

This proves ∣ ∣ ∣ ˆ A t -A t ∣ ∣ ∣ = O p ( k φ / √ n ). But T is fixed, which implies that ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ = O p ( k φ / √ n ) as well. We thus obtain ¯ λ ( ˆ A ) ≤ ¯ λ ( A ) + ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ ≤ ¯ λ ( A ) + o p (1). Since ¯ λ ( A ) &lt; 0, this proves that ¯ λ ( ˆ A ) &lt; 0 with probability approaching one, and subsequently, that ˆ A is invertible. This completes the proof of the first claim.

The second claim follows directly from Lemma 3 and Assumption 1(iii).

To prove the last claim, we first show that with probability approaching one,

<!-- formula-not-decoded -->

for some C &lt; ∞ . Define b = E [ φz ] and ˆ b = E n [ φz ]. We then have Aω ∗ = b and ˆ A ˆ ω = ˆ b . We can combine the two equations to get

<!-- formula-not-decoded -->

The above implies

<!-- formula-not-decoded -->

We earlier showed ∣ ∣ ∣ ˆ A -A ∣ ∣ ∣ = O p ( k φ / √ n ). Hence, λ ( -ˆ A ) ≥ λ ( -A ) + o p (1), so

<!-- formula-not-decoded -->

with probability approaching one, for any constant c ∈ (0 , 1). Given (A.3) and (A.4),

<!-- formula-not-decoded -->

with probability approaching one.

It remains to bound ∣ ∣ ∣ ˆ b -b ∣ ∣ ∣ and ∣ ∣ ∣ ˆ Aω ∗ -Aω ∗ ∣ ∣ ∣ . As before, we can define ˆ b t = n -1 ∑ i φ it z it and b t = E [ φ it z it ] to obtain

<!-- formula-not-decoded -->

This proves

<!-- formula-not-decoded -->

In a similar vein,

<!-- formula-not-decoded -->

as long as E [ | φ ( βφ -φ ) ᵀ ω ∗ | 2 ] = O ( k φ ) . But the latter is true under Assumptions 1(ii)(iv) since

<!-- formula-not-decoded -->

where the second inequality uses ‖ φ ᵀ ω ∗ -h ‖ 2 = O ( k -α φ ) (as shown above), and | h ( · , · ) | ∞ ≤ (1 -β ) -1 | z ( · , · ) | ∞ &lt; (1 -β ) -1 L (which can be easily verified using (2.1) and Assumption 1(iv)). Combining the above, there exists C &lt; ∞ such that | ˆ ω -ω ∗ | ≤ C √ k φ /n, with probability approaching one. We have thus shown (A.2).

Now observe that

<!-- formula-not-decoded -->

where the final inequality follows from the second claim of this theorem and (A.2). The last claim then follows from the above along with the fact that, by Assumption 1(iv),

<!-- formula-not-decoded -->

A.2. Proof of Theorem 2. The first two claims follow from steps analogous to those in Theorem 1. We thus need to show that with probability approaching one,

<!-- formula-not-decoded -->

for some C &lt; ∞ . The third claim is a straightforward consequence of this.

Recall that we use a cross-fitting procedure to estimate ξ ∗ . Let n 1 , n 2 denote the sample sizes, and ˆ η 1 , ˆ ξ 1 and ˆ η 2 , ˆ ξ 2 the estimates of η and ξ ∗ in the two folds. We shall show that | ˆ ξ 1 -ξ | = O p ( √ k r /n ) (and similarly | ˆ ξ 2 -ξ | = O p ( √ k r /n )), and therefore | ˆ ξ -ξ | = O p ( √ k r /n ). To this end, let A r := E [ rr ᵀ ], b r := E [ r ( a, x ) e ( a ′ , x ′ )], ˆ A (1) r := E (1) n [ rr ᵀ ] and ˆ b (1) r := E (1) n [ r ( a, x ) e ( a ′ , x ′ ; ˆ η 2 )], where E (1) n [ · ] denotes the empirical expectation using only the first block. We also employ the notation ψ ( a, x, a ′ , x ′ ; η ) := r ( a, x ) e ( a ′ , x ′ ; η ) and ψ it ( η ) := r ( a it , x it ) e ( a it +1 , x it +1 ; η ).

Based on the above definitions, we have ˆ A (1) r ˆ ξ 1 = ˆ b (1) r , and A r ξ ∗ = b r . Comparing with the proof of Theorem 1, the only difference is in the treatment of | ˆ b (1) r -b r | . As before, define ˆ b (1) rt := n -1 ∑ i ψ it (ˆ η 2 ) and b rt := E [ ψ it ( η )]. We then have | ˆ b (1) r -b r | = ( T -1) -1 ∑ T -1 t =1 | ˆ b (1) rt -b rt | . Since T is finite, it suffices to bound | ˆ b (1) rt -b rt | for some arbitrary t . Now, by similar arguments as in the proof of Theorem 1, we have

<!-- formula-not-decoded -->

Hence (A.5) follows once we show

<!-- formula-not-decoded -->

We now prove (A.6). Denoting N 2 the set of observations in the second fold:

<!-- formula-not-decoded -->

Define the above as R 1 nt + R 2 nt . First consider the term R 1 nt . Define

<!-- formula-not-decoded -->

Clearly, E [ δ it |N 2 ] = 0. We then have

<!-- formula-not-decoded -->

Now for any ( a, x, a ′ , x ′ ), we can note from the definition of ψ ( · ) that with probability approaching one,

<!-- formula-not-decoded -->

where the second inequality follows from Assumption 2(iii), and the third follows from 2(v). 10 In view of (A.7) and (A.8), there exists C &lt; ∞ such that

<!-- formula-not-decoded -->

where the last equality follows by Assumption 2(v). This proves

<!-- formula-not-decoded -->

Next consider the term R 2 nt . We note that E [ ψ it ( η )] is twice Fréchet differentiable. In the main text we have shown that ∂ η E [ ψ it ( η )] = 0 (c.f equation (3.7)). Furthermore, following some straightforward algebra it is possible to show | ∂ 2 η E [ ψ it ( η )] | ≤ C 1 √ k r , for some C 1 &lt; ∞ , as long as η is bounded away from 0 (as assured by Assumption 2(v)). Hence

<!-- formula-not-decoded -->

(A.9) and (A.10) imply (A.6), which concludes the proof of the theorem.

10 In particular, we have used the fact ˆ η 2 &gt; δ + o p (1) which follows from η &gt; δ and | ˆ η 2 -η | = o p (1).

## Appendix B. Online Appendix

B.1. Discrete states. Following discretization, CCP methods (see e.g Aguirregabiria and Mira, 2010), estimate h ( a, x ) by solving the recursive equations

<!-- formula-not-decoded -->

where ˆ K, ˆ P are estimates of K,P obtained as cell estimates. Now, by Tsitsiklis and Van Roy (1997), when the functional approximation saturates all the states, the TD estimate from (3.4) satisfies

<!-- formula-not-decoded -->

where E n [ ˆ h ( a ′ , x ′ ) | a, x ] denotes the conditional expectation of ˆ h ( a ′ , x ′ ) given a and x under the empirical distribution P n (the conditional distribution exists because of the discrete number of states). But for discrete data, E n [ ˆ h ( a ′ , x ′ ) | a, x ] is simply

<!-- formula-not-decoded -->

and the values of ˆ h ( a, x ) and ˘ h ( a, x ) coincide exactly. Thus, the two algorithms give identical results (a similar property also holds for g ( a, x )). Since our estimates ˆ h ( a, x ) coincide with those from the standard CCP estimators, the resulting estimate ˆ θ is also exactly the same.

When the states are discrete, Aguirregabiria and Mira (2002) show that the estimation of η is orthogonal to the estimation of θ ∗ . It is important to note, however, that the estimation of the Markov kernel K ( x ′ | a, x ) is not orthogonal to the estimation of θ ∗ . Now, with discrete states, any estimate, ˆ K ( x ′ | a, x ), of K ( x ′ | a, x ) converges at parametric rates, so √ n -consistent estimation of θ is still possible. However, as we show in Section 4.2, this creates issues with continuous states.

## B.2. Smoothness properties of dynamic-programming operators and fixed points. The following result provides a sufficient condition for Assumption 3(iii).

Lemma 4. Suppose that max | p |≤ γ sup a,x | ∂ p x z ( a, x ) | &lt; C and max | p |≤ γ sup a,x ∫ | ∂ p x K ( x ′ | a, x ) | dx ′ &lt; C for some C &lt; ∞ . Then, there exists M 0 &lt; ∞ such that | Γ z [ f ] | ∞ ≤ M 0 for all { f : | f | ∞ ≤ M 0 } . Furthermore, for each M 0 , there exists M &lt; ∞ such that Γ z [ f ]( · , a ) ∈ W γ, ∞ for all | f | ∞ ≤ M 0 and a ∈ A .

Proof. For the first claim, note that

<!-- formula-not-decoded -->

Hence, the claim is satisfied for M 0 ≥ C/β .

We now turn to the second claim. For any | p | ≤ γ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now provide sufficient conditions for h ( · ) to be γ -Hölder continuous. The result provides a sufficient condition for Assumption 3(i). It can also be used as a justification for Assumption 1(iii); see the discussion following Assumption 1.

Lemma 5. Under the assumptions of Lemma 4, there exist M 0 , M &lt; ∞ such that | h | ∞ ≤ M 0 and h ( a, · ) ∈ W γ, ∞ for all a ∈ A .

Proof. We start by showing h ( a, x ) is uniformly bounded. Define M 0 to be any positive real number such that | z ( a, x ) | ∞ &lt; βM 0 (such a number exists under the stated assumptions). Now, for any f such that | f | ∞ ≤ M 0 ,

<!-- formula-not-decoded -->

In other words, Γ z [ · ] maps the space S 0 ≡ { f : | f | ∞ ≤ M 0 } onto itself. Hence, by the properties of contraction mappings, the fixed point of Γ z [ · ] must lie in S 0 , i.e | h ( a, x ) | ∞ ≤ M 0 .

We now show h ( a, · ) ∈ W γ, ∞ for all a . For any f ∈ S 0 , and | p | ≤ γ ,

<!-- formula-not-decoded -->

where M := max { M 0 , C +(1 -β ) M 0 C } .

Defining S 1 ≡ { f : max 0 &lt; | p |≤ γ | ∂ p x f | ∞ ≤ M } , we have thus shown Γ z [ S 0 ] ⊆ S 1 . But the first part of this proof implies Γ z [ S 0 ] ⊆ S 0 . Hence, Γ z [ S 0 ] ⊆ S 0 ∩ S 1 . Consequently, by the properties of contraction mappings, h ( a, · ) ∈ S 0 ∩ S 1 . glyph[square]

It is straightforward to write down analogous versions of Lemmas 4 and 5 for Γ g [ · ] and g ( a, · ). We omit the details.

B.3. Verification of the locally robust moment function. The locally robust correction needs to satisfy two properties. First, it must be mean zero. This is easily verified. Second, it must satisfy the zero Gâteaux derivative requirement, i.e., after adding the correction term, the moment condition should satisfy

<!-- formula-not-decoded -->

for all γ ∈ H , where H is the set of all square integrable functions over the domain A×X . We now verify this below:

Let ψ h ( · ) denote the solution of the functional equation

<!-- formula-not-decoded -->

and similarly, ψ g ( · ), the solution of the functional equation

<!-- formula-not-decoded -->

Following some tedious but straightforward algebra, we can show ψ g ( · ) = ψ ( · ) and ψ h ( · ) = θψ ( · ), where ψ ( · ) is defined in (4.3) in the main text.

Now, the doubly robust moment condition in (4.4) can be expanded as

<!-- formula-not-decoded -->

where we have defined λ h ( a, x ; θ ) = θλ ( a, x ; θ ) and λ g ( a, x, ; θ ) = λ ( a, x ; θ ) for the second equality. Using the definitions of ψ ( · ) , ψ h ( · ) , ψ g ( · ) and λ ( · ), we observe that λ h ( a, x ; θ ) is the fixed point of the 'backward' dynamic programming operator

<!-- formula-not-decoded -->

and λ g ( a, x ; θ ) is the fixed point of

<!-- formula-not-decoded -->

We now verify (B.2). To this end, observe that for any square integrable γ ,

<!-- formula-not-decoded -->

where the second equality follows from the definition of ψ h ( · ) in (B.3). Since λ h ( · ) is the fixed point of Γ † h,θ [ · ], we can expand the third term in (B.7) as

<!-- formula-not-decoded -->

where the second equality uses the fact that E [ · ] corresponds to a stationary distribution. We thus conclude ∂ τ E [ ζ (˜ a , ˜ x ; θ, h + τγ, g )] | τ =0 = 0 for all γ , or ∂ h E [ ζ (˜ a , ˜ x ; θ, h, g )] = 0, as required. In fact, by similar arguments, we can also show the stronger statement that ∂ h E [ ζ (˜ a , ˜ x ; θ, h, g )] = 0 and ∂ g E [ ζ (˜ a , ˜ x ; θ, h, g )] = 0 in a Fréchet sense. The argument for showing ∂ τ E [ ζ (˜ a , ˜ x ; θ, h, g + τγ )] | τ =0 = 0 is similar.

B.4. The locally robust estimator for linear functional classes. Suppose h ( x, a ) and g ( x, a ) were truly finite-dimensional, i.e h ( x, a ) ≡ φ ( x, a ) ᵀ ω ∗ and g ( x, a ) ≡ r ( x, a ) ᵀ ξ ∗ . Denote (˜ a , ˜ x ) := ( a, x, a ′ , x ′ ), v := ( ω, ξ ), v ∗ := ( ω ∗ , ξ ∗ ) and

<!-- formula-not-decoded -->

The true value θ ∗ solves

<!-- formula-not-decoded -->

Now, (3.3) and (3.5) imply ω ∗ and ξ ∗ are identified by the auxiliary moments

<!-- formula-not-decoded -->

We make use of (B.8) and (B.9) to construct a locally robust moment for θ ∗ . Following Newey (1994) and Chernozhukov et al. (2022), this is given by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We can now construct a locally robust estimator for θ ∗ based on (B.10). Following Chernozhukov et al. (2022), we employ a cross-fitting procedure by randomly splitting the data into two samples N 1 and N 2 . We compute ˆ ω and ˆ ξ using one of the samples, say N 2 . We also compute, ˜ θ , a preliminary consistent estimator of θ ∗ by applying the plug-in estimator (3.9) on observations in N 2 . Denote by E (1) n [ · ] the empirical expectation using only the observations in the first sample, N 1 . We then obtain ˆ θ as the solution to the moment equation

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The use of cross-fitting is critical. If we had used the entire sample to estimate all of θ ∗ , ω ∗ and ξ ∗ , we would have E n [ ϕ h (˜ a , ˜ x , ˆ ω )] = 0 and E n [ ϕ g (˜ a , ˜ x , ˆ ξ )] = 0, which implies E n [ ζ n (˜ a , ˜ x , θ, ˆ ω, ˆ ξ ) ] = E n [ m ( a, x, θ, ˆ ω, ˆ ξ ) ] . As noted by Chernozhukov et al. (2022), cross-fitting gets rid of the 'own observation bias' that is the source of the degeneracy here. The solution, ˆ θ , of (B.11) is the locally robust pseduo-MLE estimator of θ ∗ .

Even though the estimator in (B.11) is predicated on h ( x, a ) and g ( x, a ) being truly finite dimensional, the work of Chernozhukov et al. (2022) suggests that ζ n (˜ a , ˜ x ; θ, v ) should have the same form as ζ n (˜ a , ˜ x ; θ, h, g ) from (4.5). Suppose that the same vector of basis functions is used for both h and g , i.e., φ ( a, x ) ≡ r ( a, x ). Then, substituting the expressions for ϕ h (˜ a , ˜ x , ω ) , ϕ g (˜ a , ˜ x , ξ ), we obtain (after some algebra)

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Comparing with (4.4), we observe that ˆ λ ( a, x ; θ ) is simply the linear semi-gradient estimator of λ ( a, x ; θ ). The benefit of (B.11) over (4.5) is that we no longer need to estimate λ ( · ) separately.

B.5. Locally robust estimators for dynamic games. The locally robust estimator for dynamic games is similar to that for single-agent models. To describe this, we recast the PMLE criterion function in the form Q ( a, x ; θ, { h i } , { g i } ) = ∑ i ln Q i ( a i , x ; θ, h i , g i ) , where

<!-- formula-not-decoded -->

Denote m i ( a i , x ; θ, h i , g i ) := ∂ θ Q i ( a i , x ; θ, h i , g i ), and

<!-- formula-not-decoded -->

The locally robust moment for θ is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Here, λ i ( a i , x ; θ ) is the fixed point to

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

For computation, we employ cross-fitting as in the single-agent setting and randomly split the markets into two samples N 1 and N 2 . We compute ˆ h i , ˆ g i , ˆ λ i for all players using one of the samples, say N 2 . Let ˜ θ denote the plug-in estimator of θ ∗ obtained using N 2 . Also, denote by E (1) n [ · ] the empirical expectation as in (5.4), but constructed only from observations in N 1 . The feasible locally robust moment for θ is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that ˆ θ can be computed player-by-player, via E (1) n [ ζ ( i ) n (˜ a i , ˜ x ; θ, ˆ h i , ˆ g i ) ] = 0, if there were no common parameters θ across players, i.e if we could partition θ ≡ ( θ 1 , . . . , θ N ). Alternatively, if the players were symmetric, i.e, z i ( · ) did not depend on i , we can pool all the players and compute a common ˆ h, ˆ g, ˆ λ .

The locally robust estimator (B.12) has the same form as (B.11), except for there being separate correction terms for the estimates ˆ h i , ˆ g i of each player i . Its theoretical properties are thus equivalent to, and can be derived in the same manner, as those for single-agent models.

B.6. Incorporating permanent unobserved heterogeneity. We incorporate permanent unobserved heterogeneity into our models by pairing the techniques from Section 3 with the sequential Expectation-Maximization (EM) algorithm (Arcidiacono and Jones, 2003). The use of the sequential EM algorithm in CCP estimation under unobserved heterogeneity was first advocated by Arcidiacono and Miller (2011), and we employ a similar approach.

Suppose that in addition to the observed state x , and the choice-specific shock e , individuals also base their choice decisions on a random state variable s which is known to the individual, but unobserved to the econometrician. As is common in the literature, we assume a finite set of unobserved states indexed by { 1 , 2 , . . . , k, ...K } . The number of states is also assumed to be known a priori. Let π k denote the population probability P ( s = k ). The value of s for an individual is assumed to be permanent and not change with time. However, we do not place any restrictions on the transition density K ( x ′ | a, x, s ), which is allowed to change with s .

To simplify the exposition, we only describe the basic version of the algorithm without local robustness corrections as in Section 4.2. It is straightforward to incorporate the correction term into the algorithm, but it comes at the expense of higher computational times.

Suppose that the per-period utility is given by z ( a, x, s ) θ . For each k , define h k ( a, x ) and g k ( a, x ) as the solutions to

<!-- formula-not-decoded -->

To simplify notation, let h itk := h k ( a it , x it ) and g itk := g k ( a it , x it ). If these quantities were known, one can estimate ( θ, π ) by maximizing the integrated pseudo-likelihood

<!-- formula-not-decoded -->

As before, we have chosen to make h ( · ) uni-dimensional to simplify the notation. We emphasize that the methods suggested here could be employed even if there was no heterogeneity in individual utilities, but the transition density were heterogenous across individuals. This is because even just the latter would result in heterogenous values for h ( a, x ), as it is a function of transition densities.

To make (B.13) usable, h k ( a, x ) , g k ( a, x ) would have to be estimated. Similar to the motivation of TD methods in Section 2, the heuristic we employ is to approximate h k (at each k ) by minimizing the mean-squared TD error

<!-- formula-not-decoded -->

where ¯ E [ · ] differs from E [ · ] in also taking the expectation over the distribution of the unobserved state s , and

<!-- formula-not-decoded -->

If P ( s = k | a , x ) were known, we could use them as weights in the semi-gradient and AVI approaches. In particular, for linear semi-gradient methods, the estimates for h k , g k would be ˆ h k ( a, x ) = φ ( a, x ) ᵀ ˆ ω k and ˆ g k ( a, x ) = r ( a, x ) ᵀ ˆ ξ k , where

<!-- formula-not-decoded -->

and we have used the notation p ik = P ( s = k | a i , x i ), z itk := z ( a it , x it , k ), φ it := φ ( a it , x it ), r it := r ( a it , x it ), and ˙ e it +1 k is the current estimate (described below) of e it +1 k := γ -ln P ( a it | x it , s i = k ). Similarly, for AVI, we could obtain iterative approximations { ˆ h ( j ) k , j = 1 , . . . , J } for h k ( · ) using

<!-- formula-not-decoded -->

The above involves solving a sequence of weighted non-parametric estimation problems, which most machine learning methods can easily handle. The estimates ˆ g ( j ) k ( · ) of g k ( · ) can be obtained in a similar fashion.

Estimation of θ ∗ , h k , g k using (B.13) and (B.14) or (B.15) requires knowledge of the unknown quantities π k and p ik along with ˙ e it +1 k . Furthermore, even if π k were known, maximizing the integrated likelihood function (B.13) is computationally very expensive. The sequential EM algorithm of Arcidiacono and Jones (2003) solves both issues. 11 To describe the procedure, let

<!-- formula-not-decoded -->

Denote by ˆ π k and ˆ p ik the estimates for π k and p ik . The algorithm consists of two steps: the M-step and the E-step. We first describe the M-step. Here, we update the estimates for h k , g k and θ ∗ based on the current estimates for π k , p ik and e it +1 k . To this end, note that we can estimate h k , g k using either (B.14) or (B.15). From these we can in-turn update ˆ θ as

<!-- formula-not-decoded -->

Next, given ( ˆ θ, ˆ h k , ˆ g k ), we update ˆ π k , ˆ p ik and ˙ e it +1 k for all i, k . This is the E-step of the EM algorithm. This step consists of three parts. In the first part, we use the current ˆ θ, ˆ h k , ˆ g k and ˆ π k to update ˆ p ik for each i, k using Bayes' rule:

<!-- formula-not-decoded -->

11 Maximizing (B.13) is not equivalent to Full Information Maximum Likelihood (FIML). As in Arcidiacono and Jones (2003), the identification and asymptotic properties of θ, π are in fact determined by constructing moment conditions that correspond to the first order conditions from maximizing Q ( θ, π ), augmented with additional moments identifying h k , g k . Together, these moment conditions, which motivate the sequential EM algorithm, can in turn be related to the identification properties of FIML.

<!-- image -->

Note: Histograms denote the finite sample distribution, and blue line is normal density

Figure B.1. Finite sample distributions under linear semi-gradients without locally robust corrections.

In the second part, we update ˆ π k , for each k , as

<!-- formula-not-decoded -->

Finally, we also update ˙ e it +1 k for all i, t, k as

<!-- formula-not-decoded -->

The E and M steps are iterated until convergence.

It is also possible to extend our methods to allow for Markovian unobserved heterogeneity, by employing a variant of the classical Baum-Welch algorithm. The computational and statistical details of such a procedure are however more involved and will be described elsewhere.

## B.7. Additional simulation results.

B.7.1. Firm entry problem - finite sample distributions. Figures B.1 and B.2 plot the finite sample distribution of the estimates for linear semi-gradients under non-locally robust and locally robust methods. The distributions are very close to normal even without a locally robust correction.

Figures B.3 and B.4 present similar results for AVI.

## B.7.2. Bus engine replacement problem.

<!-- image -->

Note: Histograms denote the finite sample distribution, and blue line is normal density

Figure B.2. Finite sample distributions under linear semi-gradients with locally robust corrections.

<!-- image -->

Note: Histograms denote the finite sample distribution, and blue line is normal density

Figure B.3. Finite sample distributions under AVI without locally robust corrections.

Consider the following version of the Rust (1987) bus engine replacement problem which is adapted from Arcidiacono and Miller (2011). Each period t = 1 , ..., T ; T &lt; ∞ , Harold Zurcher decides whether to replace the engine of a bus ( a t = 0), or keep it ( a t = 1). Denote his action by j ∈ { 0 , 1 } . Each bus is characterized by a permanent type s ∈ { 1 , 2 } , and the mileage accumulated since the last engine replacement x t ∈ { 1 , 2 , ... } . Harold Zurcher observes both s and x t . The econometrician observes mileage x t , and we make different assumptions on the observability of bus type s .

<!-- image -->

Note: Histograms denote the finite sample distribution, and blue line is normal density

Figure B.4. Finite sample distributions under AVI with locally robust corrections.

Mileage increases by one unit if the engine is kept in period t and is set to zero if the engine is replaced. The current-period payoff for keeping the engine is given by θ 0 + θ 1 x t + θ 2 s + e 1 t , where θ ∗ ≡ { θ 0 , θ 1 , θ 2 } are the structural parameters of interest, and e jt is a choice-specific transitory shock that follows a Type 1 Extreme Value distribution. The current-period payoff of replacing the engine is normalized to e 0 t , and the discount factor is set to 0.9 in this application.

To carry out the simulations, we choose values for the structural parameters ( θ 0 = 2 , θ 1 = -0 . 15 , θ 2 = 1), and recursively derive the value functions for each possible combination of x , s and t . We then use these to compute the conditional replacement probabilities for the same set of combinations of variables.

We provide results using the linear semi-gradient method to approximate h ( a, x ) and g ( a, x ). Our first set of results treats the bus type s as known to the econometrician. We then provide findings for a setting with permanent unobserved heterogeneity. All results are based on 1000 simulations with 1000 buses and T = 30 time periods each. Each round of the simulations begins by generating data for 1000 buses and 2000 time periods. The mileage of each bus is set to zero in t = 0. We then simulate the choices a t using the conditional replacement probabilities. Finally, we restrict the generated data to 30 time periods between t = 1000 and t = 1030. This is to ensure that our data is close to being drawn from a stationary model. In all simulations, we parameterize h ( a, x ) and

Table B.1. Simulations: Bus engine replacement problem

|                      |         | not locally robust   | not locally robust   | locally robust   | locally robust   |
|----------------------|---------|----------------------|----------------------|------------------|------------------|
|                      | DGP (1) | TDL (2)              | MSE (3)              | TDL (4)          | MSE (5)          |
| Linear semi-gradient |         |                      |                      |                  |                  |
| θ 0 (intercept)      | 2.0     | 1.9788 (0.0868)      | 0.0080               | 1.9778 (0.0870)  | 0.0081           |
| θ 1 (mileage)        | -0.15   | -0.1492 (0.0033)     | 1.2e-05              | -0.1489 (0.0034) | 1.3e-05          |
| θ 2 (bus type)       | 1.0     | 1.0044 (0.0583)      | 0.0034               | 1.0032 (0.0584)  | 0.0034           |

Notes: The table reports results for 1000 simulations. Column (1) shows the true parameter values in the model. Columns (2) and (4) report the empirical mean and standard deviations for the estimated parameters. Columns (2)-(3) are based on the estimation method without correction function, columns (4)-(5) report results for the locally robust estimator. The mean squared error are reported in columns (3) and (5), respectively.

g ( a, x ) using a third order polynomial in s , x t and a t . 12 The choice probabilities η are estimated using a logit model that is a function of the state variables s and x t , where the same polynomial is used as before. 13

Table B.1 shows the results treating bus type s as known, with and without locally robust correction. It can be seen that our estimator produces parameter estimates that are closely centered around the true values. These results are comparable to those found by Arcidiacono and Miller (2011) in a similar version of the bus engine replacement problem. However, in contrast to their CCP method, our estimator does not exploit a finite dependence property. When comparing the results from our locally robust estimator in column (4) to the results from the suboptimal estimator in column (2), it can be seen that the absolute bias is smaller for all three parameter estimates. However the variance of the locally robust estimator is higher due to the sample splitting employed in the locally robust procedure (we used two-fold cross-fitting). Overall, while in theory the locally robust estimator is preferable to the non-robust version, we find that in practice there is very little difference between the two versions of the algorithm.

12 In total, there are k φ = k r = 16 terms. These include a constant, the binary variables s and a t , all x t terms up to a third order, and pairwise and triple interactions between the x t terms and the binary variables.

13 We run our simulations on a MacBook Pro with an M1 chip and 16 GB of RAM. The approximate computation times for one estimation round is 4 seconds without locally robust correction, and 14 seconds with locally robust correction.

Table B.2. Simulations: Bus engine replacement problem with unobserved heterogeneity

|                      | DGP (1)   | TDL (2)          | MSE (3)   |
|----------------------|-----------|------------------|-----------|
| Linear semi-gradient |           |                  |           |
| θ 0 (intercept)      | 2.0       | 1.9750 (0.1255)  | 0.0164    |
| θ 1 (mileage)        | -0.15     | -0.1492 (0.0039) | 1.5e-05   |
| θ 2 (bus type)       | 1.0       | 1.0035 (0.1018)  | 0.0104    |

Notes: The table reports results for 1000 simulations. Column (1) shows the true parameter values in the model. Column (2) reports the empirical mean and standard deviations for the estimated parameters. The mean squared errors are reported in column (3). The results are based on the estimation method without correction function.

In a final set of simulations, we introduce permanent unobserved heterogeneity into our setting by assuming that the permanent bus type s ∈ { 1 , 2 } is unknown to the researcher. To generate results for these simulations, we follow the steps outlined in Section B.6 where we pair our techniques with a sequential EM algorithm (Arcidiacono and Jones, 2003), without using a locally robust correction. 14 The results are shown in Table B.2. Once again, our algorithm produces parameter estimates that are closely centered around the true values. Compared to the results without permanent unobserved heterogeneity, the standard deviation of our estimates is slightly higher due to the uncertainty around the bus type s .

14 We run our simulations on a MacBook Pro with an M1 chip and 16 GB of RAM. The approximate computation times is 305 seconds for one estimation round.