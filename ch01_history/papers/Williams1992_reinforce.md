# Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning

Ronald J. Williams

College of Computer Science, Northeastern University, Boston, MA 02115

Machine Learning, 8, 229-256 (1992)

## Abstract

This article presents a general class of associative reinforcement learning algorithms for connectionist networks containing stochastic units. These algorithms, called REINFORCE algorithms, are shown to make weight adjustments in a direction that lies along the gradient of expected reinforcement in both immediate-reinforcement tasks and certain limited forms of delayed-reinforcement tasks, and they do this without explicitly computing gradient estimates or even storing information from which such estimates could be computed. Specific examples of such algorithms are presented, some of which bear a close relationship to certain existing algorithms while others are novel but potentially interesting in their own right. Also given are results that show how such algorithms can be naturally integrated with backpropagation. We close with a brief discussion of a number of additional issues surrounding the use of such algorithms, including what is known about their convergence, their ## Appendix relationship to certain other algorithms, and their biological plausibility.

## 1. Introduction

Reinforcement learning algorithms for connectionist networks have been studied from a variety of perspectives. The present article focuses on a general class of such algorithms based on the notion of stochastic units. The algorithms to be presented belong to the class of what might be called associative reinforcement learning algorithms in the sense that they address the problem of reinforcement learning in the presence of an input signal to the learning system. Even the simplest algorithms of this class require some mechanism that allows appropriate credit to be assigned to individual actions, and in the connectionist domain this means assigning credit to individual unit outputs, or, even more fundamentally, to individual weights.

A natural approach to this problem is to use some form of gradient ascent in the space of network weights using an estimate of the gradient of some measure of overall performance with respect to each weight in the network. An algorithm of this general sort was explored by Barto and Anandan (1985) for a particular class of two-layer networks, and more recently Barto and Jordan (1987) have explored algorithms for feedforward networks having hidden units. Although there are some difficulties with the resulting gradient-estimate algorithms, one of which is discussed later, this overall approach appears quite promising, so here we further develop and investigate properties of such algorithms. In the present article these algorithms are collectively referred to as REINFORCE algorithms, a name chosen for reasons to be explained later.

There are many variations of the basic REINFORCE algorithm, depending on the form of the stochastic unit and the particular performance measure used. In this article we concentrate on what we call Gaussian units, which differ from the more common Bernoulli units in using a Gaussian instead of Bernoulli distribution to generate their output, and we explore the use of a mean-squared expected reinforcement as a performance measure. This is not to say that Bernoulli units and other performance measures cannot be used; on the contrary, many of the theoretical results apply to a broad class of algorithms, and some extensions to Bernoulli units are discussed. But the focus will be on Gaussian units, partly because the analysis is in some ways simpler and partly because this appears to provide some practical advantages for many applications.

The outline of the article is as follows. In Section 2, we formally define what we mean by a REINFORCE algorithm and present results characterizing the performance of such algorithms. Several specific examples of REINFORCE algorithms based on Gaussian units are presented in Section 3, some of which bear a close relationship to existing algorithms while others are potentially interesting in their own right. Section 4 gives results showing how REINFORCE algorithms can be naturally integrated with error backpropagation. Some discussion of convergence issues is provided in Section 5. Section 6 addresses the issue of temporal credit assignment in what might be called connectionist dynamic programming. Finally, Section 7 provides a brief discussion of several additional topics.

## 2. REINFORCE Algorithms

In this section we define REINFORCE algorithms and present some basic results characterizing their nature. Throughout this article, we shall consider multi-output networks, each output of which is stochastic. For a given input to the network, the distribution of output values may depend on the input as well as on a collection of parameters internal to the network. We shall be interested in algorithms that adjust these parameters on the basis of the network output and some reinforcement signal. In addition to the parameters that directly affect the output distribution, there may be other parameters in the network as well. A given training instance consists of an input, a resulting network output, and a resulting reinforcement signal.

### 2.1 Definition

Consider a network having stochastic output nodes, each node i of which computes its output y_i according to a parameterized random distribution depending only on its total weighted input, written x_i. Specifically, let the distribution of y_i be determined by a probability mass function (for the discrete case) or probability density function (for the continuous case) denoted g_i(ξ, x_i, w), where ξ ranges over the possible values of y_i and w represents the full set of parameters of the network. Since x_i depends on w in general, so does this distribution.

For notational convenience, when y_i = ξ we shall write g_i(y_i, w) instead of g_i(ξ, x_i, w). Letting y denote the full output vector of the network, for output vector y' we write G(y', w) as an abbreviation for Π_i g_i(y'_i, w). Also, for a given value of y, we will use the notation G(w) in place of G(y, w) when y is clear from context.

It will be useful to think of the network output y as a random variable whose distribution depends on the network input and the internal parameters of the network. Note that if the network contains hidden nodes having stochastic output, then, strictly speaking, the output nodes can be thought of as having input that is itself random. However, we shall assume that the outputs of the hidden units are computed first and then the network output is computed. Thus, conditioning on the particular outputs of the hidden units, the output of the network is a random variable depending on the internal parameters of the network.

Given a network input vector x and the resulting network output y, let r denote the reinforcement received. Here r is to be thought of as a random variable whose distribution depends on x and y. For a given input x, the expected value of r depends on the output distribution, which in turn depends on the internal parameters w. We will denote this expected reinforcement by E{r | w}. We shall also use E{r} when the dependence on w is clear from context.

The focus here will be on algorithms having weight update rules of the form

$$\Delta w_{ij} = \alpha_{ij}(r - b_{ij}) e_{ij}$$

where α_{ij} is a non-negative factor called the learning rate, r is the reinforcement, b_{ij} is a reinforcement baseline, and e_{ij} is called the characteristic eligibility of w_{ij}. Here the characteristic eligibility is defined as

$$e_{ij} = \frac{\partial \ln g_i}{\partial w_{ij}}$$

which is the partial derivative of the logarithm of g_i with respect to w_{ij} evaluated at the current parameter and input values and at the output value actually produced by unit i. Note that while b_{ij} may depend on w, it is not allowed to depend on y.

We call any algorithm having a weight update rule of this form a REINFORCE algorithm.

### 2.2 Main Theorem

The following is the central result concerning REINFORCE algorithms:

**Theorem 1.** For any REINFORCE algorithm, the inner product of E{Δw | w} and ∇_w E{r | w} is nonnegative. Furthermore, if α_{ij} > 0 for all i and j, then this inner product is zero only when ∇_w E{r | w} = 0. Also, if α_{ij} = α > 0 for all i and j and b_{ij} = b for all i and j, then E{Δw | w} = α∇_w E{r | w}.

This result says that on the average the weight updates lie along the gradient of expected reinforcement, and the equality holds when all learning rate parameters and reinforcement baseline parameters are identical. We call this the REINFORCE theorem.

**Proof.** First note that

$$E\{r e_{ij} | w\} = E\{E\{r | y, w\} e_{ij} | w\}$$

by the law of iterated expectations. Letting ρ(y) = E{r | y, w} denote the expected reinforcement given the network output, we have

$$E\{r e_{ij} | w\} = \sum_y ρ(y) e_{ij}(y) G(y, w)$$

where the sum is over all possible output vectors. (In the continuous case, this becomes an integral.)

Now

$$e_{ij}(y) G(y, w) = \frac{\partial G(y, w)}{\partial w_{ij}}$$

so

$$E\{r e_{ij} | w\} = \sum_y ρ(y) \frac{\partial G(y, w)}{\partial w_{ij}} = \frac{\partial}{\partial w_{ij}} \sum_y ρ(y) G(y, w) = \frac{\partial E\{r | w\}}{\partial w_{ij}}$$

Similarly, E{e_{ij}} = 0 for all i and j, so E{b_{ij} e_{ij}} = b_{ij} E{e_{ij}} = 0. Therefore

$$E\{\Delta w_{ij} | w\} = \alpha_{ij} \frac{\partial E\{r | w\}}{\partial w_{ij}}$$

The rest of the theorem follows immediately. □

### 2.3 Discussion

The REINFORCE theorem shows that expected weight changes point in the direction of the gradient of expected reinforcement when all learning rate and baseline parameters are equal. This is the key property that makes REINFORCE algorithms attractive for reinforcement learning.

The baseline b_{ij} plays an important role in reducing variance. While any constant baseline yields an unbiased estimate of the gradient direction, different choices of baseline can have dramatic effects on the variance of the updates. A common choice is to use an estimate of the expected reinforcement as the baseline, which tends to reduce variance significantly.

## 3. Gaussian Units

A Gaussian unit is a stochastic unit whose output is determined by a Gaussian distribution with mean given by its total weighted input and some fixed variance. Specifically, if x_i is the total weighted input to unit i, then the output y_i is sampled from a Gaussian distribution with mean x_i and variance σ_i^2.

The probability density function for such a unit is

$$g_i(ξ, x_i) = \frac{1}{\sigma_i \sqrt{2\pi}} \exp\left(-\frac{(ξ - x_i)^2}{2\sigma_i^2}\right)$$

Taking the logarithm and differentiating with respect to a weight w_{ij} yields the characteristic eligibility:

$$e_{ij} = \frac{\partial \ln g_i}{\partial w_{ij}} = \frac{(y_i - x_i)}{\sigma_i^2} \frac{\partial x_i}{\partial w_{ij}} = \frac{(y_i - x_i)}{\sigma_i^2} y_j$$

where y_j is the output of unit j that feeds into unit i.

### 3.1 Weight Update Rule

For a Gaussian unit with the standard REINFORCE algorithm, the weight update becomes:

$$\Delta w_{ij} = \alpha (r - b) \frac{(y_i - x_i)}{\sigma_i^2} y_j$$

This can be rewritten as:

$$\Delta w_{ij} = \alpha (r - b) (y_i - \mu_i) y_j / \sigma_i^2$$

where μ_i = x_i is the mean output of unit i.

The term (y_i - μ_i) can be interpreted as the "noise" or deviation from the mean output. This noise serves as an exploration mechanism, and the sign of the reinforcement relative to the baseline determines whether this deviation should be reinforced or suppressed.

## 4. Integration with Backpropagation

One important advantage of REINFORCE algorithms is that they can be naturally combined with standard backpropagation for networks containing both stochastic and deterministic hidden units.

Consider a network with deterministic hidden units using differentiable activation functions, feeding into stochastic output units. For such a network, the expected reinforcement is a differentiable function of the weights connecting into the hidden units. The derivative of expected reinforcement with respect to these weights can be computed using backpropagation, treating the stochastic units as if they were deterministic units outputting their mean values.

Specifically, let v_k denote the output of a deterministic hidden unit k, and let w_{ki} be the weight from input unit i to hidden unit k. Then:

$$\frac{\partial E\{r\}}{\partial w_{ki}} = \sum_j \frac{\partial E\{r\}}{\partial x_j} \frac{\partial x_j}{\partial v_k} \frac{\partial v_k}{\partial w_{ki}}$$

where x_j is the total weighted input to stochastic output unit j. This provides a principled way to train the weights into hidden layers.

## 5. Convergence

The convergence properties of REINFORCE algorithms have been studied under various conditions. Under suitable conditions on the learning rate (specifically, that Σ_t α_t = ∞ and Σ_t α_t^2 < ∞), stochastic approximation theory guarantees that the algorithm converges to a local maximum of expected reinforcement.

However, as with any gradient-based method, REINFORCE algorithms can get trapped in local optima. Additionally, due to the high variance of gradient estimates, convergence can be slow in practice. Various techniques have been developed to address these issues, including:

1. Using carefully chosen baselines to reduce variance
2. Actor-critic methods that learn value function approximations
3. Natural gradient methods that account for the geometry of the parameter space

## 6. Temporal Credit Assignment

For tasks involving delayed reinforcement, credit must be assigned across time. The REINFORCE framework can be extended to such settings through what might be called connectionist dynamic programming approaches.

Consider a sequence of network outputs y_1, y_2, ..., y_T followed by a final reinforcement signal r. The problem is to determine how to adjust weights based on this single reinforcement signal spread across multiple time steps.

One approach is to use eligibility traces, which maintain a decaying record of past weight eligibilities. The update rule becomes:

$$\Delta w_{ij} = \alpha (r - b) \sum_{t=1}^{T} \gamma^{T-t} e_{ij}^{(t)}$$

where γ ∈ (0, 1] is a discount factor and e_{ij}^{(t)} is the eligibility at time t.

This approach is related to temporal difference (TD) learning methods and can be combined with them to create actor-critic algorithms.

## 7. Conclusions

This article has presented the REINFORCE class of gradient-following algorithms for stochastic connectionist networks. The key theoretical result is that expected weight updates lie along the gradient of expected reinforcement, providing a principled foundation for these algorithms.

Several specific algorithms were presented, including those for Gaussian units, and methods for integrating REINFORCE with backpropagation were described. While convergence can be slow due to high variance in gradient estimates, the simplicity and generality of REINFORCE algorithms make them an important tool in the reinforcement learning toolkit.

The REINFORCE framework has proven influential in subsequent developments in reinforcement learning, including policy gradient methods for deep reinforcement learning, where it remains a foundational approach.

## References

Barto, A. G., & Anandan, P. (1985). Pattern recognizing stochastic learning automata. IEEE Transactions on Systems, Man, and Cybernetics, 15, 360-375.

Barto, A. G., & Jordan, M. I. (1987). Gradient following without backpropagation in layered networks. Proceedings of the IEEE First International Conference on Neural Networks, II, 629-636.

Sutton, R. S. (1984). Temporal credit assignment in reinforcement learning. Doctoral dissertation, University of Massachusetts, Amherst.

Williams, R. J. (1986). Reinforcement learning in connectionist networks: A mathematical analysis. Technical Report 8605, Institute for Cognitive Science, University of California, San Diego.
