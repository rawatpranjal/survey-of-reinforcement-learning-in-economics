# 3.2: Neural Networks

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 370-376
**Topics:** neural networks, training neural networks, multilayer, deep neural networks

---

| 3.5. 3.6.                                              | Policy Gradient and Related Methods . . . . . . . . p. 408 3.5.1. Gradient Methods for Parametric Cost . . . . . . . Optimization . . . . . . . . . . . . . . . . p. 3.5.2. Policy Gradient-Like Methods . . . . . . . . . p. 412 3.5.3. Scaling and Proximal Policy Optimization . . . . . . Methods . . . . . . . . . . . . . . . . . . p. 3.5.4. Random Direction Methods . . . . . . . . . . p. 424 3.5.5. Random Search and Cross-Entropy Methods . .   |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aggregation . . . . . . . . . . . . . . .              | 409                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|                                                        | 417                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|                                                        | p. 426                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 3.5.6. Refining and Retraining Parametric Policies . . | . . p. 428                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|                                                        | . . p. 429                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 3.6.1. Aggregation with Representative States          | . . . . p. 430                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.6.2. Continuous Control Space Discretization .       | . . . p. 437                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 3.6.3. Continuous State Space - POMDP Discretization   | p. 438                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 3.6.4. General Aggregation .                           | . . . . . . . . . . . . p. 440                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.6.5. Types of Aggregation and Error Bounds           | . . . . p. 443                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.6.6. Aggregation Using Features                      | . . . . . . . . . . p. 445                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 3.6.7. Biased Aggregation .                            | . . . . . . . . . . . . p. 446                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 3.6.8. Asynchronous Distributed Aggregation            | . . . . . p. 449                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 3.7. Notes, Sources, and Exercises . . . . . . . . .   | . . . p. 451                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

In this chapter, we will discuss the methods and objectives of o ff -line training through the use of parametric approximation architectures such as neural networks. We begin with a general discussion of parametric architectures and their training in Section 3.1. We then consider the training of neural networks in Section 3.2, and their use in the context of finite horizon approximate DP in Section 3.3. In Sections 3.4 and 3.5, we discuss the training of policies. Finally, in Section 3.6, we discuss aggregation methods.

## 3.1 PARAMETRIC APPROXIMATION ARCHITECTURES

For the success of approximation in value space, it is important to select a class of lookahead function approximations ˜ J k that is suitable for the problem at hand. In the preceding two chapters we discussed several methods for choosing ˜ J k , based mostly on some form of rollout. We will now discuss how ˜ J k can be obtained by o ff -line training from a parametric class of functions, possibly involving a neural network, with the parameters 'optimized' with the use of some algorithm.

Approximating Function

<!-- image -->

Target Cost Function

Figure 3.1.1 The general structure for parametric cost approximation. We approximate the target cost function J ( x ) with a member from a parametric class ˜ J ( x↪ r ) that depends on a parameter r . We use training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and a form of optimization that aims to find a parameter ˆ r that 'minimizes' the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q .

As we have noted in Chapter 1, the most popular structure for parametric cost function approximation involves a target function J ( x ) that we want to approximate with a member of a parametric class of functions ˜ J ( x↪ r ) that depends on a parameter r (see Fig. 3.1.1). In particular, we collect training data ( x s ↪ J ( x s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , which we use to determine a parameter ˆ r that leads to a good 'fit' between the data J ( x s ) and the predictions ˜ J ( x s ↪ ˆ r ) of the parametrized function. This is usually done through an optimization approach, aiming to minimize the size of the errors J ( x s ) -˜ J ( x s ↪ ˆ r ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q .

Approximation of a target policy θ with a policy from a parametric class ˜ θ ( x↪ r ) is largely similar. Here, the training data may be obtained, for example, from rollout control calculations, thus enabling the construction of both value and policy networks that can be combined for use in a perpetual rollout scheme. An important di ff erence, however, is that the

Figure 3.1.2 A general structure for parametric policy approximation for the case where the control space is finite, U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ , and its relation to a classification scheme. It produces a randomized policy of the form (3.1), which is converted to a nonrandomized policy through the maximization operation (3.2).

<!-- image -->

approximate cost values ˜ J ( x↪ r ) are real numbers, whereas the approximate policy values ˜ θ ( x↪ r ) are elements of a control space U . Thus if U consists of m dimensional vectors, ˜ θ ( x↪ r ) consists of m numerical components. In this case the parametric approximation problems for cost functions and for policies are fairly similar, and both involve continuous space approximations.

On the other hand, the case where the control space is finite U = ¶ u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m ♦ is quite di ff erent. In this case, for any x , ˜ θ ( x↪ r ) consists of one of the m possible controls u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m . This connects policy space approximation with traditional classification schemes, whereby objects x are classified as belonging to one of the categories u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u m . In particular, ˜ θ ( x↪ r ) provides the category of x , and ˜ θ ( · ↪ r ) can be viewed as a classifier . Some of the most prominent classification schemes actually produce randomized outcomes, i.e., x is associated with a probability distribution

<!-- formula-not-decoded -->

which is a randomized policy in our policy approximation context; see Fig. 3.1.2. This is often done for algorithmic convenience, since many optimization methods, including least squares regression, require that the optimization variables are continuous. Then, the randomized policy (3.1) can be converted to a nonrandomized policy using a maximization operation: associate x with the control of maximum probability (cf. Fig. 3.1.2),

<!-- formula-not-decoded -->

We will discuss the use of classification methods for approximation in policy space in Section 3.4, following our discussion of parametric approximation in value space.

## 3.1.1 Cost Function Approximation

For the remainder of this section, as well as Sections 3.2 and 3.3, we will focus on approximation in value space schemes, where the approximate cost

functions are selected from a parametric class of functions ˜ J k ( x k ↪ r k ) that for each k , depend on the current state x k and a vector r k = ( r 1 ↪k ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r m k ↪k ) of m k 'tunable' scalar parameters. By adjusting these parameters, the 'shape' of ˜ J k can be adjusted to closely approximate a target function, typically the true optimal cost-to-go function J * k or the cost-to-go function Jk↪ π of a given policy π . The class of functions ˜ J k ( x k ↪ r k ) is called an approximation architecture , and the process of choosing the parameter vectors r k is known as training or tuning the architecture. We will focus initially on approximation of cost functions, hence the use of the ˜ J k notation. In Section 3.4 we will consider the other major use of parametric approximation architectures, of the form ˜ θ k ( x k ↪ r k ), where the target function is a control function θ k that is part of some policy.

The simplest training approach for parametric architectures is to do some form of semi-exhaustive or semi-random search in the space of parameter vectors and adopt the parameters that result in best performance of the associated one-step lookahead controller (according to some criterion). Such methods are mainly practical when the number of parameters is relatively small.

Random search and Bayesian optimization methods have also been used to tune hyperparameters of an approximation architecture. Examples include determining the number of layers in a neural network or selecting the number of clusters when partitioning discrete spaces. For further details on these methods, we refer to the research literature.

More systematic training methods are based on numerical optimization, such as a least squares fit that seeks to match the architecture's cost approximation to a 'training set,' i.e., a large number of state and cost value pairs obtained through some form of sampling. In Sections 3.1-3.3 we will focus primarily on this approach.

## 3.1.2 Feature-Based Architectures

There is a large variety of approximation architectures, including those based on polynomials, wavelets, radial basis functions, discretization/interpolation schemes, neural networks, and others. A particularly interesting type of cost approximation involves feature extraction , a process that maps the state x k into some vector φ k ( x k ), called the feature vector associated with x k at time k . The vector φ k ( x k ) consists of scalar components

<!-- formula-not-decoded -->

called features . A feature-based cost approximation has the form

<!-- formula-not-decoded -->

where r k is a parameter vector and ˆ J k is some function. Thus, the cost approximation depends on the state x k through its feature vector φ k ( x k ).

Figure 3.1.3 The structure of a linear feature-based architecture. At time k , we use a feature extraction mapping to generate an input φ k ( x k ) to a linear mapping defined by a parameter vector r k .

<!-- image -->

Note that we are allowing for di ff erent features φ k ( x k ) and di ff erent parameter vectors r k for each stage k . This is necessary for nonstationary problems (e.g., if the state space changes over time), and also to capture the e ff ect of proximity to the end of the horizon. In contrast, for stationary problems with long or infinite horizons where the state space remains constant over k , it is common to use the same features and parameter vectors across all stages. The following discussion can be easily extended to accommodate infinite horizon methods, which we will explore in detail later.

Features are often designed manually, leveraging human intelligence, insights, and experience to capture the most significant characteristics of the current state. There are also systematic ways to construct features, including the use of data and neural networks, which we will discuss shortly. In this section, we provide a brief and selective presentation of architectures.

The rationale behind using features is that optimal cost-to-go functions J * k can be complex, nonlinear mappings. By decomposing their complexity into smaller, more manageable components, it becomes feasible to approximate these functions e ff ectively. In particular, if the features encode much of the nonlinearity of J * k , we may be able to use a relatively simple architecture ˆ J k to approximate J * k . For example, with a well-chosen feature vector φ k ( x k ), a good approximation to the cost-to-go is often provided by linearly weighting the features, i.e.,

<!-- formula-not-decoded -->

where r /lscript ↪k and φ /lscript ↪k ( x k ) are the /lscript th components of r k and φ k ( x k ), respectively, and r ′ k φ k ( x k ) denotes the inner product of r k and φ k ( x k ), viewed as column vectors of /Rfractur m k (a prime denotes transposition, so r ′ k is a row vector); see Fig. 3.1.3.

This is called a linear feature-based architecture , and the scalar parameters r /lscript ↪k are also called weights . Among other advantages, these architectures admit simpler training algorithms than their nonlinear counterparts; see the neuro-dynamic programming (NDP) book [BeT96]. Mathematically, the approximating function ˜ J k ( x k ↪ r k ) can be viewed as a member of the subspace spanned by the features φ /lscript ↪k ( x k ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m k , which

glyph[triangleright]

<!-- image -->

glyph[triangleright]

Figure 3.1.4 Illustration of a piecewise constant architecture. The state space is partitioned into subsets S 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ S m , with each subset S /lscript defining the feature

<!-- formula-not-decoded -->

with its own weight r /lscript .

for this reason are also referred to as basis functions . We provide a few examples, where for simplicity we drop the index k .

## Example 3.1.1 (Piecewise Constant Approximation)

Suppose that the state space is partitioned into subsets S 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ S m , so that every state belongs to one and only one subset. Let the /lscript th feature be defined by membership to the set S /lscript , i.e.,

<!-- formula-not-decoded -->

Consider the architecture

<!-- formula-not-decoded -->

where r is the vector consists of the m scalar parameters r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r m . It can be seen that ˜ J ( x↪ r ) is the piecewise constant function that has value r /lscript for all states within the set S /lscript ; see Fig. 3.1.4.

The piecewise constant approximation is an example of a linear feature-based architecture that involves exclusively local features . These are

features that take a nonzero value only for a relatively small subset of states. Thus a change of a single weight causes a change of the value of ˜ J ( x↪ r ) for relatively few states x . At the opposite end we have linear feature-based architectures that involve global features . These are features that take nonzero values for a large number of states. The following is a common example.

## Example 3.1.2 (Polynomial Approximation)

An important case of linear architecture is one that uses polynomial basis functions. Suppose that the state consists of n components x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x n , each taking values within some range of integers. For example, in a queueing system, x i may represent the number of customers in the i th queue, where i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . Suppose that we want to use an approximating function that is quadratic in the components x i . Then we can define a total of 1 + n + n 2 basis functions that depend on the state x = ( x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x n ) via

<!-- formula-not-decoded -->

A linear approximation architecture that uses these functions is given by

<!-- formula-not-decoded -->

where the parameter vector r has components r 0 , r i , and r ij , with i↪ j = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . Indeed, any kind of approximating function that is polynomial in the components x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x n can be constructed similarly.

A more general polynomial approximation may be based on some other known features of the state. For example, we may start with a feature vector

<!-- formula-not-decoded -->

and transform it with a quadratic polynomial mapping. In this way we obtain approximating functions of the form

<!-- formula-not-decoded -->

where the parameter r has components r 0 , r i , and r ij , with i↪ j = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . This can also be viewed as a linear architecture that uses the basis functions

<!-- formula-not-decoded -->

The preceding example architectures are generic in the sense that they can be applied to many di ff erent types of problems. Other architectures rely on problem-specific insight to construct features, which are then combined into a relatively simple architecture. We present two examples involving games.