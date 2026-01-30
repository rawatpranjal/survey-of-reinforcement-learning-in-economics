## Learning Values and Policies

| Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3.1. Parametric Approximation Architectures . . . . . . . p. 351 3.1.1. Cost Function Approximation . . . . . . . . . p. 352 3.1.2. Feature-Based Architectures . . . . . . . . . p. 353 3.1.3. Training of Linear and Nonlinear Architectures . p. 364 3.2. Neural Networks . . . . . . . . . . . . . . . . . p. 372 3.2.1. Training of Neural Networks . . . . . . . . . p. 377 3.2.2. Multilayer and Deep Neural Networks . . . . . p. 378 3.3. Learning Cost Functions in Approximate DP . . . . . p. 379 3.3.1. Fitted Value Iteration . . . . . . . . . . . . p. 380 3.3.2. Q-Factor Parametric Approximation - Model-Free . . Implementation . . . . . . . . . . . . . . . p. 382 3.3.3. Parametric Approximation in Infinite Horizon . . . . Problems - Approximate Policy Iteration . . . . p. 384 3.3.4. Optimistic Policy Iteration with Parametric Q-Factor . Approximation - SARSA and DQN . . . . . . p. 388 3.3.5. Approximate Policy Iteration for Infinite Horizon . . . POMDP . . . . . . . . . . . . . . . . . . p. 390 3.3.6. Advantage Updating - Approximating Q-Factor . . . Di ff erences . . . . . . . . . . . . . . . . . p. 395 3.3.7. Di ff erential Training of Cost Di ff erences for Rollout p. 397 3.4. Learning a Policy in Approximate DP . . . . . . . . p. 399 3.4.1. The Use of Classifiers for Approximation in Policy . . Space . . . . . . . . . . . . . . . . . . . p. 400 3.4.2. Policy Iteration with Value and Policy Networks p. 404 3.4.3. Why Use On-Line Play and not Just Train a Policy . . Network to Emulate the Lookahead Minimization? p. 406 |

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

¿ Mbo con of foton Loan]

Figure 3.1.5 The board of the tetris game. The squares fill up as blocks of di ff erent shapes fall from the top of the grid and are added to the top of the wall. The shapes are generated according to some stochastic process. As a given block falls, the player can move horizontally and rotate the block in all possible ways, subject to the constraints imposed by the sides of the grid and the top of the wall. When a row of full squares is created, this row is removed, the bricks lying above this row move one row downward, and the player scores a point. The player's objective is to maximize the score attained (total number of rows removed) within N steps or up to termination of the game, whichever occurs first.

<!-- image -->

## Example 3.1.3 (Tetris)

Let us consider the game of tetris, which we formulated in Example 1.6.2 as a stochastic shortest path problem with the termination state being the end of the game (see Fig. 3.1.5). The state is the pair of the board position x and the shape of the current falling block y . We viewed as control, the horizontal positioning and rotation applied to the falling block. The optimal cost-togo function is a vector of huge dimension (there are 2 200 board positions in a 'standard' tetris board of width 10 and height 20). However, it has been successfully approximated in practice by low-dimensional linear architectures.

In particular, the following features have been proposed in the paper by Bertsekas and Io ff e [BeI96]: the heights of the columns, the height di ff erentials of adjacent columns, the wall height (the maximum column height), the number of holes of the board, and the constant 1 (the unit is often included as a feature in cost approximation architectures, as it allows for a constant shift in the approximating function). These features are readily recognized by tetris players as capturing important aspects of the board position. There

The use of feature-based approximate DP methods for the game of tetris was first suggested in the paper by Tsitsiklis and Van Roy [TsV96], which introduced just two features (in addition to the constant 1): the wall height and the number of holes of the board. Most studies have used the set of features of [BeI96] described here, but other sets of features have also been used; see [ThS09] and the discussion in [GGS13].

Mobility, Safety, etc Weighting of Features Score Position Evaluator

Mobility, Safety, etc Weighting of Features Score Position Evaluator

<!-- image -->

Material Balance, Mobility, Safety, etc

Mobility, Safety, etc Weighting of Features Score Position Evaluator

Figure 3.1.6 A feature-based architecture for computer chess.

are a total of 22 features for a 'standard' board with 10 columns. Of course the 2 200 × 22 matrix of feature values cannot be stored in a computer, but for any board position, the corresponding row of features can be easily generated, and this is su ffi cient for implementation of the associated approximate DP algorithms. For recent works involving approximate DP methods and the preceding 22 features, see [Sch13], [GGS13], and [SGG15], which reference several other related papers.

In the works mentioned above the shapes of the falling blocks are stochastically independent. In a more challenging version of the problem, which has not been considered in the literature thus far, successive shapes are correlated. Then the state of the problem would become more complex, since past shapes would be useful in predicting future shapes. As a result, we may need to introduce state estimation and additional features in order to properly deal with the e ff ects of correlations.

## Example 3.1.4 (Computer Chess)

Computer chess programs that involve feature-based architectures have existed for many years, and are still used widely (they have been upstaged in the mid-2010s by alternative types of chess programs, which use neural network techniques that will be discussed later). These programs are based on approximate DP for minimax problems, a feature-based parametric architecture, and multistep lookahead.

The fundamental principles on which all computer chess programs (as well as most two-person game programs) are based were laid out by Shannon [Sha50], before Bellman started his work on DP. Shannon proposed multistep lookahead and evaluation of the end positions by means of a 'scoring function' (in our terminology this plays the role of a cost function approximation). This function may involve for example the calculation of a numerical value for each of a set of major features of a position that chess players easily recognize (such as material balance, mobility, pawn structure, and other positional factors), together with a method to combine these numerical values into a single score. Shannon then went on to describe various strategies of exhaustive and selective search over a multistep lookahead tree of moves.

We may view the scoring function as a feature-based architecture for evaluating a chess position/state (cf. Fig. 3.1.6). In most computer chess programs, the features are weighted linearly, i.e., the architecture ˜ J ( x↪ r ) that is used for multistep lookahead is linear [cf. Eq. (3.3)]. In many cases, the weights have been determined manually, by trial and error based on experi-

ence. However, in some programs, the weights have been determined with supervised learning techniques that use examples of grandmaster play, i.e., by adjustment to bring the play of the program as close as possible to the play of chess grandmasters. This is a technique that applies more broadly in artificial intelligence; see Tesauro [Tes89b], [Tes01].

In a recent computer chess breakthrough, the entire idea of extracting features of a position through human expertise was abandoned in favor of feature discovery through self-play and the use of neural networks. The first program of this type to attain supremacy over humans, as well as over the best computer programs that use human expertise-based features, was AlphaZero (Silver et al. [SHS17]). This program, described in Section 1.1, is based on DP principles of approximate policy iteration and multistep lookahead based on Monte Carlo tree search.

Our next example relates to a methodology for feature construction, where the number of features may increase as more data is collected. For a simple example, consider the piecewise constant approximation of Example 3.1.1, where more pieces are progressively added based on new data, possibly using some form of exploration-exploitation tradeo ff .

## Example 3.1.5 (Feature Extraction from Data)

We have viewed so far feature vectors φ ( x ) as functions of x , obtained through some unspecified process that is based on prior knowledge about the cost function being approximated. On the other hand, features may also be extracted from data. For example suppose that with some preliminary calculation using data, we have identified some suitable states x ( /lscript ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , that can serve as 'anchors' for the construction of Gaussian basis functions of the form

<!-- formula-not-decoded -->

where σ is a scalar 'variance' parameter, and ‖ · ‖ denotes the standard Euclidean norm. This type of function is known as a radial basis function . It is concentrated around the state x ( /lscript ), and it is weighed with a scalar weight r /lscript to form a parametric linear feature-based architecture, which can be trained using additional data. Several other types of data-dependent basis functions, such as support vector machines, are used in machine learning, where they are often referred to as kernels .

While it is possible to use a preliminary calculation to obtain the anchors x ( /lscript ) in Eq. (3.4), and then use additional data for training, one may also consider enrichment of the set of basis functions simultaneously with training. In this case the number of the basis functions increases as the training data is collected. A motivation here is that the quality of the approximation may increase with additional basis functions. This idea underlies a field of machine learning, known as kernel methods or sometimes nonparametric methods .

A further discussion is outside our scope. We refer to the literature; see e.g., books such as Cristianini and Shawe-Taylor [ChS00], [ShC04], Scholkopf and Smola [ScS02], Bishop [Bis06], Kung [Kun14], surveys such as Hofmann,

Scholkopf, and Smola [HSS08], Pillonetto et al. [PDC14], RL-related discussions such as Dietterich and Wang [DiW02], Ormoneit and Sen [OrS02], Engel, Mannor, and Meir [EMM05], Jung and Polani [JuP07], Reisinger, Stone, and Miikkulainen [RSM08], Busoniu et al. [BBD10a], Bethke [Bet10], and recent developments such as Tu et al. [TRV16], Rudi, Carratino, and Rosasco [RCR17], Belkin, Ma, and Mandal [BMM18]. In what follows, for the sake of simplicity, we will focus on parametric architectures with a fixed and given feature vector, since the choice of approximation architecture is somewhat peripheral to our main focus.

The next example considers a feature extraction strategy that is particularly relevant to problems of partial state information.

## Example 3.1.6 (Feature Extraction from Su ffi cient Statistics)

The concept of a su ffi cient statistic, which originated in inference methodologies, plays an important role in DP. As discussed in Section 1.6, it refers to quantities that summarize all the essential content of the state x k for optimal control selection at time k .

In particular, consider a partial information context where at time k we have accumulated the information vector (also called the past history )

<!-- formula-not-decoded -->

which consists of the past controls u 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u k -1 and the state-related measurements z 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ z k obtained at the times 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ k . The control u k is allowed to depend only on I k , and the optimal policy is a sequence of the form { θ ∗ 0 ( I 0 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ ∗ N -1 ( I N -1 ) } . We say that a function S k ( I k ) is a su ffi -cient statistic at time k if the control function θ ∗ k depends on I k only through S k ( I k ), i.e., for some function ˆ θ k , we have where θ ∗ k is optimal.

<!-- formula-not-decoded -->

There are several examples of su ffi cient statistics, and they are typically problem-dependent. A trivial possibility is to view I k itself as a su ffi cient statistic, and a more sophisticated possibility is to view the belief state b k as a su ffi cient statistic (this is the conditional probability distribution of x k given I k ; cf. Section 1.6.6). For a proof that b k is indeed a su ffi cient statistic and for a more detailed discussion of other possible su ffi cient statistics, see [Ber17a], Chapter 4. For a mathematically more advanced discussion, see [BeS78], Chapter 10.

Since a su ffi cient statistic contains all the relevant information for optimal control purposes, it is natural to introduce features of a given su ffi cient statistic and to train a corresponding approximation architecture accordingly. Potentially e ff ective features could include some special characteristic of I k (such as whether some alarm-like 'special' event has been observed), or a partial history (such as the last m measurements and controls in I k , or more sophisticated versions based on the concept of a finite-state controller proposed

by White [Whi91], and White and Scherer [WhS94], and further discussed by Hansen [Han98], Kaelbling, Littman, and Cassandra [KLC98], Meuleau et al. [MPK99], Poupart and Boutilier [PoB04], Yu and Bertsekas [YuB08], Saldi, Yuksel, and Linder [SYL17]). In the case where the belief state b k is used as a su ffi cient statistic, examples of good features may be a point estimate based on b k , the variance of this estimate, and other quantities that can be simply extracted from b k .

The paper by Bhattacharya et al. [BBW20] considers another type of feature vector that is related to the belief state. This is a su ffi cient statistic, denoted by y k , which subsumes the belief state b k , in the sense that b k can be computed exactly knowing y k . One possibility is for y k to be the union of b k and some identifiable characteristics of the belief state, or some compact representation of the measurement history up to the current time (such as a number of most recent measurements, or the state of a finite-state controller). Even though the information content of y k is no di ff erent than the information content of b k for the purposes of exact optimization, a su ffi cient statistic y k that is specially designed for the problem at hand may lead to improved performance in the presence of cost and policy approximations.

We finally note a related idea, which is to supplement a su ffi cient statistic with features of other su ffi cient statistics, and thus obtain an enlarged/richer su ffi cient statistic. In problem-specific contexts, and in the presence of approximations, this may yield improved results.

## Example 3.1.7 (Feature-Based Dimensionality Reduction by Aggregation)

The use of a feature vector φ ( x ) to represent the state x in an approximation architecture of the form ˜ J ( φ ( x ) ↪ r ) implicitly involves state aggregation , i.e., the grouping of states into subsets. We will discuss aggregation in some detail in Section 3.6. Here we will give a summary of a special type of aggregation architecture.

In particular, let us assume that the feature vector can take only a finite number of values, and define for each possible value v , the subset of states S v whose feature vector is equal to v :

<!-- formula-not-decoded -->

We refer to the sets S v as the aggregate states induced by the feature vector. These sets form a partition of the state space. An approximate cost-to-go function of the form ˜ J ( φ ( x ) ↪ r ) is piecewise constant with respect to this partition; that is, it assigns the same cost-to-go value ˜ J ( v↪ r ) to all states in the set S v .

An often useful approach to deal with problem complexity in DP is to introduce an 'aggregate' DP problem, whose states are some suitably defined feature vectors φ ( x ) of the original problem. The precise form of the aggregate problem may depend on intuition and/or heuristic reasoning, based on our understanding of the original problem. Suppose now that the aggregate problem is simple enough to be solved exactly by DP, and let ˆ J ( v )

Sy

State Space

Feature

Extraction

<!-- image -->

Feature Space

Figure 3.1.7 Feature-based state partitioning using a partition of the space of features. Each set Y of the feature space partition induces a set S Y of the state space partition that consists of states with 'similar' features, i.e., states that map into the same subset of the feature-space partition.

be its optimal cost-to-go when the initial value of the feature vector is v . Then ˆ J ( φ ( x ) ) provides an approximation architecture for the original problem, i.e., the architecture that assigns to state x the (exactly) optimal cost-togo ˆ J ( φ ( x ) ) of the feature vector φ ( x ) in the aggregate problem. There is considerable freedom on how one formulates and solves aggregate problems. Werefer to the DP textbooks [Ber12], [Ber17a], and the RL textbook [Ber19a], Chapter 6, for a detailed treatment; see also the discussion of Section 3.6.

The next example relates to an architecture that is particularly useful when parallel computation is available.

## Example 3.1.8 (Feature-Based State Space Partitioning)

A simple method to construct complex and sophisticated approximation architectures, is to partition the state space into several subsets and construct a separate approximation for each subset. For example, by using a separate linear or quadratic polynomial approximation for each subset of the partition, we can construct piecewise linear or piecewise quadratic approximations over the entire state space. Similarly, we may use a separate neural network architecture for each set of the partition. An important issue here is the choice of the method for partitioning the state space. Regular partitions (e.g., grid partitions) may be used, but they often lead to a large number of subsets and very time-consuming computations.

Generally speaking, each subset of the partition should contain 'similar' states so that the variation of the optimal cost-to-go over the states of the subset is relatively smooth and can be approximated with smooth functions. An interesting possibility is to use features as the basis for partition. In particular, one may use a more or less regular partition of the space of features, which induces a possibly irregular meaningful partition of the original state space. In this way, each subset of the irregular partition contains states with 'similar features;' see Fig. 3.1.7.

As an illustration consider the game of chess. The state here consists of the board position, but the nature of the position evolves through distinct phases: opening, middlegame, and endgame. Each of these phases

may be a ff ected di ff erently by special features of the position. For example there are several di ff erent types of endgames (rook endgames, king-and-pawn endgames, minor-piece endgames, etc), which are characterized by identifiable features and call for di ff erent playing strategies. It would thus make sense to partition the set of chess positions according to their features, and use a separate strategy on each set of the partition. Indeed this is done to some extent in a number of chess programs.

A potential challenge with partitioned architectures is the discontinuity that can occur at the boundaries between subsets. To address this, a variant known as soft partitioning , is sometimes used, whereby the subsets of the partition are allowed to overlap and the discontinuity is smoothed out over their intersection. Once function approximations are computed for each subset, the approximate cost-to-go in the overlapping regions is represented as a smoothly varying linear combination of the corresponding subset approximations.

Partitioning and local approximations can also be used to enhance the quality of approximation in parts of the space where the target function has some special character. The next example addresses one such possibility.

## Example 3.1.9 (Local-Global Architectures)

Suppose that the state space is partitioned in subsets S 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ S M and consider approximations of the form

<!-- formula-not-decoded -->

where each φ k↪m ( x ) is a basis function which is local, in the sense that it contributes to the approximation only on the set S m ; that is, it takes the value 0 for x glyph[triangleleft] ∈ S m . Here ˆ J ( x↪ ˆ r ) is an architecture of the type discussed earlier. The parameter vector r consists of ˆ r and the coe ffi cients r m ( k ) of the basis functions. Thus the portion ˆ J ( x↪ ˆ r ) of the architecture is used to capture 'global' aspects of the target function, while each portion

<!-- formula-not-decoded -->

is used to capture aspects of the target function that are 'local' to the subset S m .

Note that the training of local-global approximation architectures may potentially be enhanced by using specialized algorithms. The NDP book [BeT96] (Section 3.1.3) discusses computational methods that are tailored to the local-global structure.

## Architectures with Automatic Feature Construction

Unfortunately, identifying an adequate set of features can be challenging, so it is important to have methods that construct features automatically, to

supplement any existing ones. Indeed, some architectures are specifically designed to function without pre-existing knowledge of suitable features. The kernel methods of Example 3.1.5 are one such approach. Another very popular possibility is neural networks , which we will describe in Section 3.2. Some of these architectures involve training that constructs simultaneously both the feature vectors φ ( x ) and the parameter vectors r that weigh them.

Importantly, architectures that generate features automatically can still incorporate additional features based on prior knowledge or problemspecific insights. For example, these architectures can take as input not only the state x , but also supplementary hand-crafted features deemed relevant for the problem. Another possibility is to combine automatically constructed features with other a priori known good features into a (mixed) linear architecture that involves both types of features. The weights of the latter linear architecture may be obtained with a separate second stage training process, following the first stage training process that constructs automatically suitable features using a nonlinear architecture such as a neural network. This approach can enhance the overall e ff ectiveness of the approximation by merging the strengths of automatic feature construction with domain-specific expertise.

## 3.1.3 Training of Linear and Nonlinear Architectures

In this section, we discuss briefly the training process of choosing the parameter vector r of a parametric architecture ˜ J ( x↪ r ), focusing primarily on incremental gradient methods. The most common type of training is based on a least squares optimization, also known as least squares regression . Here a set of state-cost training pairs

<!-- formula-not-decoded -->

called the training set , is collected and r is determined by solving the problem

<!-- formula-not-decoded -->

Thus r is chosen to minimize the sum of squared errors between the sample costs β s and the architecture-predicted costs ˜ J ( x s ↪ r ). Here there is some target cost function J that we aim to approximate with ˜ J ( · ↪ r ), and the sample cost β s is the value J ( x s ) plus perhaps some error or 'noise.'

The cost function of the training problem (3.6) is generally nonconvex, and can be quite complicated. This may pose challenges, since there may exist multiple local minima. However, for a linear architecture the cost function is convex quadratic, and the training problem admits a closedform solution. In particular, for the linear architecture

<!-- formula-not-decoded -->

the problem becomes or

<!-- formula-not-decoded -->

By setting the gradient of the quadratic objective to 0, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus by matrix inversion we obtain the minimizing parameter vector

<!-- formula-not-decoded -->

If the inverse above does not exist, an additional quadratic in r , called a regularization function, is added to the least squares objective to deal with this, and also to help with other issues to be discussed later. A singular value decomposition approach may also be used to deal with the matrix inversion issue; see the NDP book [BeT96], Section 3.2.2.

Thus a linear architecture has the important advantage that the training problem can be solved exactly and conveniently with the formula (3.7) (of course it may be solved by any other algorithm that is suitable for linear least squares problems, including iterative algorithms). By contrast, if we use a nonlinear architecture, such as a neural network, the associated least squares problem is nonquadratic and also nonconvex, so it is hard to solve in principle. Despite this fact, through a combination of sophisticated implementation of special gradient algorithms, called incremental , and powerful computational resources, neural network methods have been successful in practice.

## Incremental Gradient Methods

We will now discuss briefly special methods for solution of the nonlinear least squares training problem (3.6), assuming a parametric architecture that is di ff erentiable in the parameter vector. This methodology can be properly viewed as a subject in nonlinear programming and iterative algorithms, and as such it can be studied independently of the approximate DP methods of this book. Thus the reader who has already some exposure to the subject may skip to the next section. The author's nonlinear programming textbook [Ber16] and the RL book [Ber19a] provide more detailed presentations.

We view the training problem (3.6) as a special case of the minimization of a sum of component functions

<!-- formula-not-decoded -->

where each f i is a di ff erentiable scalar function of the n -dimensional column vector y (this is the parameter vector). Thus we use the more common symbols y and m in place of r and q , respectively, and we replace the squared error terms

<!-- formula-not-decoded -->

in the training problem (3.6) with the generic terms f i ( y ).

The (ordinary) gradient method for problem (3.8) generates a sequence ¶ y k ♦ of iterates, starting from some initial guess y 0 for the minimum of the cost function f . It has the form

<!-- formula-not-decoded -->

where γ k is a positive stepsize parameter. The incremental gradient method is similar to the ordinary gradient method, but uses the gradient of a single component of f at each iteration. It has the general form

<!-- formula-not-decoded -->

where i k is some index from the set ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright]↪ m ♦ , chosen by some deterministic or randomized rule. Thus a single component function f i k is used at iteration k , with great economies in gradient calculation cost over the ordinary gradient method (3.9), particularly when m is large. This is of course a radical simplification, which involves a large approximation error, yet it performs surprisingly well! The idea is to attain faster convergence when far from the solution as we will explain shortly; see the author's books [BeT96], [Ber16], and [Ber19a] for a more detailed discussion.

The method for selecting the index i k of the component to be iterated on at iteration k is important for the performance of the method. We describe three common rules , the last two of which involve randomization: ‡

- (1) A cyclic order , the simplest rule, whereby the indexes are taken up in the fixed deterministic order 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , so that i k is equal to ( k modulo

We use standard calculus notation for gradients; see, e.g., [Ber16], Appendix A. In particular, ∇ f ( y ) denotes the n -dimensional column vector whose components are the first partial derivatives ∂ f ( y ) glyph[triangleleft] ∂ y i of f with respect to the components y 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ y n of the column vector y .

‡ With these stepsize rules, the incremental gradient method is often called stochastic gradient or stochastic gradient descent method.

m ) plus 1. A contiguous block of iterations involving the components f 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ f m in this order and exactly once is called a cycle .

- (2) A uniform random order , whereby the index i k chosen randomly by sampling over all indexes with a uniform distribution, independently of the past history of the algorithm. This rule may perform better than the cyclic rule in some circumstances.
- (3) A cyclic order with random reshu ffl ing , whereby the indexes are taken up one by one within each cycle, but their order after each cycle is reshu ffl ed randomly (and independently of the past). This rule is used widely in practice, particularly when the number of components m is modest, for reasons to be discussed later.

Note that in the cyclic cases, it is essential to include all components in a cycle; otherwise some components will be sampled more often than others, leading to a bias in the convergence process. Similarly, it is necessary to sample according to the uniform distribution in the random order case.

Focusing for the moment on the cyclic rule (with or without reshuffling), we note that the motivation for the incremental gradient method is faster convergence: we hope that far from the solution, a single cycle of the method will be as e ff ective as several (as many as m ) iterations of the ordinary gradient method (think of the case where the components f i are similar in structure). Near a solution, however, the incremental method may not be as e ff ective.

In particular, we note that there are two complementary performance issues to consider in comparing incremental and nonincremental methods:

- (a) Progress when far from convergence . Here the incremental method can be much faster. For an extreme case take m large and all components f i identical to each other. Then an incremental iteration requires m times less computation than a classical gradient iteration, but gives exactly the same result, when the stepsize is scaled to be m times larger. While somewhat extreme, this example reflects the essential mechanism by which incremental methods can be much superior: far from the minimum a single component gradient will point to 'more or less' the right direction, at least most of the time; see the following example.
- (b) Progress when close to convergence . Here the incremental method can be inferior. In particular, the ordinary gradient method (3.9) is convergent with a constant stepsize under reasonable assumptions, see e.g., [Ber16]. However, the incremental method requires a diminishing stepsize, and its ultimate rate of convergence can be much slower.

This behavior is illustrated in the following example, first given in the 1995 first edition of the author's nonlinear programming book [Ber16], and the NDP book [BeT96].

## Example 3.1.10

Assume that y is a scalar, and that the problem is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash where c i and b i are given scalars with c i = 0 for all i . The minimum of each of the components f i ( y ) = 1 2 ( c i y -b i ) 2 is

<!-- formula-not-decoded -->

while the minimum of the least squares cost function f is

<!-- formula-not-decoded -->

It can be seen that y ∗ lies within the range of the component minima

<!-- formula-not-decoded -->

and that for all y outside the range R , the gradient

<!-- formula-not-decoded -->

has the same sign as ∇ f ( y ) (see Fig. 3.1.8). As a result, when outside the region R , the incremental gradient method

<!-- formula-not-decoded -->

approaches y ∗ at each step, provided the stepsize γ k is small enough. In fact it can be verified that it is su ffi cient that

<!-- formula-not-decoded -->

However, for y inside the region R , the i th step of a cycle of the incremental gradient method need not make progress. It will approach y ∗ (for small enough stepsize γ k ) only if the current point y k does not lie in the interval connecting y ∗ i and y ∗ . This induces an oscillatory behavior within the region R , and as a result, the incremental gradient method will typically not converge to y ∗ unless γ k → 0. By contrast, the ordinary gradient method, which takes the form

<!-- formula-not-decoded -->

Сі

min

i

FAROUT REGION

(сіх — bi) ?

REGION OF CONFUSION

Figure 3.1.8. Illustrating the advantage of incrementalism when far from the optimal solution. The region of component minima

<!-- image -->

<!-- formula-not-decoded -->

is labeled as the 'region of confusion.' It is the region where the method does not have a clear direction towards the optimum. The i th step in an incremental gradient cycle is a gradient step for minimizing ( c i y -b i ) 2 , so if y lies outside the region of component minima R = [ min i y ∗ i ↪ max i y ∗ i ] ↪ (labeled as the 'farout region') and the stepsize is small enough, progress towards the solution y ∗ is made.

can be verified to converge to y ∗ for any constant stepsize γ with

<!-- formula-not-decoded -->

However, for y outside the region R , a full iteration of the ordinary gradient method need not make more progress towards the solution than a single step of the incremental gradient method. In other words, with comparably intelligent stepsize choices, far from the solution (outside R ), a single pass through the entire set of cost components by incremental gradient is roughly as e ff ective as m passes by ordinary gradient .

The preceding example assumes that each component function f i has a minimum, so that the range of component minima is defined. In cases where the components f i have no minima, a similar phenomenon may occur, as illustrated by the following example (the idea here is that we may combine several components into a single component that has a minimum).

## Example 3.1.11:

Consider the case where f is the sum of increasing and decreasing convex exponentials, i.e.,

<!-- formula-not-decoded -->

where a i and b i are scalars with a i &gt; 0 and b i = 0. Let

/negationslash

<!-- formula-not-decoded -->

and assume that I + and I -have roughly equal numbers of components. Let also y ∗ be the minimum of m i =1 f i .

∑ Consider the incremental gradient method that given the current point, call it y k , chooses some component f i k and iterates according to the incremental gradient iteration

<!-- formula-not-decoded -->

Then it can be seen that if y k &gt;&gt; y ∗ , y k +1 will be substantially closer to y ∗ if i ∈ I + , and negligibly further away than y ∗ if i ∈ I -. The net e ff ect, averaged over many incremental iterations, is that if y k &gt;&gt; y ∗ , an incremental gradient iteration makes roughly one half the progress of a full gradient iteration, with m times less overhead for calculating gradients. The same is true if y k &lt;&lt; y ∗ . On the other hand as y k gets closer to y ∗ the advantage of incrementalism is reduced, similar to the preceding example. In fact in order for the incremental method to converge, a diminishing stepsize is necessary, which will ultimately make the convergence slower than the one of the nonincremental gradient method with a constant stepsize.

The discussion of the preceding examples relies on y being one-dimensional, but in many multidimensional problems the same qualitative behavior can be observed. In particular, a pass through the i th component f i by the incremental gradient method can make progress towards the solution in the region where the component gradient ∇ f i k ( y k ) makes an angle less than 90 degrees with the cost function gradient ∇ f ( y k ). If the components f i are not 'too dissimilar,' this is likely to happen in a region of points that are far from the optimal solution set. This behavior has been validated in numerous practical contexts, including the training of neural networks (see the following section), where incremental gradient methods, often referred to as backpropagation methods , have been widely employed.

## Stepsize Choice

The choice of the stepsize γ k (often referred to as the learning rate ) plays an important role in the performance of incremental gradient methods. In practice, it is common to use a constant stepsize for a (possibly prespecified) number of iterations, then decrease the stepsize by a certain factor, and

repeat, up to the point where the stepsize reaches a prespecified floor value. An alternative possibility is to use a diminishing stepsize rule of the form

<!-- formula-not-decoded -->

where γ , β 1 , and β 2 are some positive scalars. There are also variants of the method that use a constant stepsize throughout, and can be shown to converge to a stationary point of f under reasonable assumptions. In one type of such method the degree of incrementalism gradually diminishes as the method progresses (see the author's paper [Ber97a]). Another incremental approach with similar aims, is the aggregated gradient method, which is discussed in the author's textbooks [Ber15a], [Ber16], [Ber19a].

Regardless of whether a constant or a diminishing stepsize is ultimately used, the incremental method must use a much larger stepsize than the corresponding nonincremental gradient method (as much as m times larger, so that the size of the incremental gradient step is comparable to the size of the nonincremental gradient step).

One possibility is to use an adaptive stepsize rule, whereby, roughly speaking, the stepsize is reduced (or increased) when the progress of the method indicates that the algorithm is (or is not) oscillating. Formal ways to implement such stepsize rules with sound convergence properties have been proposed by Tseng [Tse98], and Moriyama, Yamashita, and Fukushima [MYF03].

## Diagonal Scaling

The di ffi culty with stepsize selection may also be addressed with diagonal scaling , i.e., using a stepsize γ k j that is di ff erent for each of the components y j of y . Second derivatives can be very useful for this purpose. A time-honored method in generic nonlinear programming problems of unconstrained minimization of a function f , is to use diagonal scaling with stepsizes

<!-- formula-not-decoded -->

where γ is a constant that is nearly equal 1 (the second derivatives may also be approximated by gradient di ff erence approximations). However, in least squares training problems [cf. Eq. (3.6)], this type of scaling is inconvenient because of the additive form of f as a sum of a large number of component functions:

cf. Eq. (3.8).

<!-- formula-not-decoded -->

Thus incremental versions of diagonal scaling have been suggested in the neural network literature, and have been incorporated in publicly and commercially available software. Two influential papers are Duchi, Hazan, and Singer [DHS11], and Kingman and Ba [KiB14]. The later paper has introduced the ADAM (Adaptive Moment Estimation) method, which is widely used at present for training neural networks. It adapts the stepsize for each component of the parameter vector r , based on estimates of the first and second moments of the gradient, and incorporates several other ideas that have been proposed in the incremental methods literature over the years.

There are also alternative approaches for scaling the direction of incremental methods to steer it closer to the Newton direction. The RL book [Ber19a] (Section 3.1.3) describes one such method that involves second derivatives, whereby the idea here is to write Newton's method in a format that is well suited to the additive character of the cost function f , and involves low order matrix inversion. One can then implement diagonal scaling by setting to zero the o ff -diagonal terms of the inverted matrices, so that the algorithm involves no matrix inversion. There is also another related algorithm, which is based on the Gauss-Newton method and the extended Kalman filter; see the author's paper [Ber96], and the books [BeT96] and [Ber16]. Finally, we note an important scaling approach for stochastic parametric optimization problems, which will be discussed in Section 3.5.2.

## 3.2 NEURAL NETWORKS

There are several di ff erent types of neural networks that can be used for a variety of tasks, such as pattern recognition, classification, image and speech recognition, natural language processing, and others. In this section, we focus on our finite horizon DP context, and the role that neural networks can play in approximating the optimal cost-to-go functions J * k . As an example within this context, we may first use a neural network to construct an approximation to J * N -1 . Then we may use this approximation to approximate J * N -2 , and continue this process backwards in time, to obtain approximations to all the optimal cost-to-go functions J * k , k = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1, as we will discuss in more detail in Section 3.3.

Throughout this section, we will focus on the type of neural network, known as a multilayer perceptron , which is the one most used at present in the RL applications discussed in this book. Naturally, there are variations that are adapted to the problem at hand. For example AlphaZero uses a specialized neural network that takes advantage of the board-like structure of chess and Go to facilitate and expedite the associated computations.

To describe the use of neural networks in finite horizon DP, let us consider the typical stage k , and for convenience drop the index k ; the subsequent discussion applies to each value of k separately. We consider

parametric architectures ˜ J ( x↪ v↪ r ) of the form

<!-- formula-not-decoded -->

that depend on two parameter vectors v and r . Our objective is to select v and r so that ˜ J ( x↪ v↪ r ) approximates some target cost function that can be sampled (possibly with some error). The process is to collect a training set that consists of a large number of state-cost pairs ( x s ↪ β s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and to find a function ˜ J ( x↪ v↪ r ) of the form (3.11) that matches the training set in a least squares sense, i.e., ( v↪ r ) minimizes

<!-- formula-not-decoded -->

We postpone for later the question of how the training pairs ( x s ↪ β s ) are generated. Notice the di ff erent roles of the two parameter vectors here: v parametrizes φ ( x↪ v ), which in some interpretation may be viewed as a feature vector, and r is a vector of linear weighting parameters for the components of φ ( x↪ v ).

## Single Layer Perceptron

A neural network architecture provides a parametric class of functions ˜ J ( x↪ v↪ r ) of the form (3.11) that can be used in the optimization framework just described. The simplest type of neural network is the single layer perceptron ; see Fig. 3.2.1. Here the state x is encoded as a vector of numerical values y ( x ) with components y 1 ( x ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ y n ( x ), which is then transformed linearly as

<!-- formula-not-decoded -->

where A is an m × n matrix and b is a vector in /Rfractur m . ‡ This transformation is called the linear layer of the neural network. We view the components of A and b as parameters to be determined, and we group them together into the parameter vector v = ( A↪b ).

Each of the m scalar output components of the linear layer,

<!-- formula-not-decoded -->

The least squares training problem used here is based on nonlinear regression . This is a classical method for approximating the expected value of a function with a parametric architecture, and involves a least squares fit of the architecture to simulation-generated samples of the expected value. We refer to machine learning and statistics textbooks for more discussion.

‡ The method of encoding x into the numerical vector y ( x ) is generally problem-dependent, but it can be critical for the success of the training process. We should note also that some of the components of y ( x ) could be known interesting features of x that can be designed based on problem-specific knowledge.

(May Include 'Problem-Specific' Features)

(May Include 'Problem-Specific' Features)

Figure 3.2.1 Schematic illustration of a single layer perceptron, a neural network consisting of a linear layer and a nonlinear layer. It provides a way to compute features of the state, which can be used for cost function approximation. The state x is encoded as a vector of numerical values y ( x ), which is then transformed linearly as Ay ( x ) + b in the linear layer. The m scalar output components of the linear layer, become the inputs to nonlinear one-dimensional functions σ : /Rfractur ↦→/Rfractur , thus producing the m scalars

<!-- image -->

<!-- formula-not-decoded -->

which can be viewed as features that are in turn linearly weighted with parameters r /lscript .

becomes the input to a nonlinear di ff erentiable and monotonically increasing function σ that maps scalars to scalars. A simple and popular possibility is the rectified linear unit (ReLU for short), which is simply the function max ¶ 0 ↪ ξ ♦ , approximated by a di ff erentiable function σ by some form of smoothing operation; for example σ ( ξ ) = ln(1 + e ξ ), which is illustrated in Fig. 3.2.2. Other functions, used since the early days of neural networks, have the property

<!-- formula-not-decoded -->

see Fig. 3.2.3. Such functions are called sigmoids , and some common choices are the hyperbolic tangent function

<!-- formula-not-decoded -->

and the logistic function

<!-- formula-not-decoded -->

In what follows, we will ignore the character of the function σ (except for di ff erentiability), and simply refer to it as a 'nonlinear unit' and to the corresponding layer as a 'nonlinear layer.'

Linear Layer Parameter

Cost Approximation

1 0 -1

Figure 3.2.2 The rectified linear unit σ ( ξ ) = ln(1 + e ξ ). It is the function max ¶ 0 ↪ ξ ♦ with its corner 'smoothed out.' Its derivative is σ ′ ( ξ ) = e ξ glyph[triangleleft] (1 + e ξ ), and approaches 0 and 1 as ξ →-∞ and ξ →∞ , respectively.

<!-- image -->

Selective Depth Lookahead Tree

<!-- image -->

1 0 -1

Figure 3.2.3 Some examples of sigmoid functions. The hyperbolic tangent function is on the left, while the logistic function is on the right.

At the outputs of the nonlinear units, we obtain the scalars

<!-- formula-not-decoded -->

One possible interpretation is to view φ /lscript ( x↪ v ) as features of x , which are linearly combined using weights r /lscript , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , to produce the final output

<!-- formula-not-decoded -->

Note that each value φ /lscript ( x↪ v ) depends on just the /lscript th row of A and the /lscript th component of b , not on the entire vector v . In some cases this motivates placing some constraints on individual components of A and b to achieve special problem-dependent 'handcrafted' e ff ects.

## State Encoding and Direct Feature Extraction

The state encoding operation that transforms x into the neural network input y ( x ) can be instrumental in the success of the approximation scheme.

Cost Approximation

Figure 3.2.4 A nonlinear architecture with a view of the state encoding process as a feature extraction mapping preceding the neural network. The state encoder may also contain tunable parameters.

<!-- image -->

Examples of state encodings are components of the state x , numerical representations of qualitative characteristics of x , and more generally features of x , i.e., functions of x that aim to capture 'important nonlinearities' of the optimal cost-to-go function. With the latter view of state encoding, we may consider the approximation process as consisting of a feature extraction mapping, followed by a neural network with input the extracted features of x , and output the cost-to-go approximation; see Fig. 3.2.4.

The idea here is that with a well-designed feature extraction mapping, the neural network need not be very complicated and may be trained more easily. This intuition is borne out by simple examples and practical experience. However, as is often the case with neural networks, it is hard to support it with a quantitative analysis.

## Universal Approximation Property of Neural Networks

An important question is how well we can approximate the target function J * k with a neural network architecture, assuming we can choose the number of the nonlinear units m to be as large as we want. The answer to this question is quite favorable and is provided by the so-called universal approximation theorem .

Roughly, the theorem says that assuming that x is an element of a Euclidean space X and y ( x ) ≡ x , a neural network of the form described can approximate arbitrarily closely (in an appropriate mathematical sense), over a compact subset S ⊂ X , any piecewise continuous function J : S ↦→ /Rfractur , provided the number m of nonlinear units is su ffi ciently large. For proofs of the theorem, we refer to Cybenko [Cyb89], Funahashi [Fun89], Hornik, Stinchcombe, and White [HSW89], and Leshno et al. [LLP93]. For additional sources and intuitive explanations we refer to Bishop ([Bis95], pp. 129-130), Jones [Jon90], and the RL textbook [Ber19a], Section 3.2.1.

While the universal approximation theorem provides some reassurance about the adequacy of the neural network structure, it does not predict how many nonlinear units we may need for 'good' performance in a given problem. Unfortunately, this is a di ffi cult question to even pose

precisely, let alone to answer adequately. In practice, one is often reduced to trying increasingly larger values of m until one is convinced that satisfactory performance has been obtained for the task at hand. One may improve on trial-and-error schemes with more systematic hyperparameter search methods, such as Bayesian optimization, and in fact this idea has been used to tune the parameters of the deep network used by AlphaZero.

Experience has shown that in many cases the number of required nonlinear units and corresponding dimension of the parameter space can be very large, adding significantly to the di ffi culty of solving the training problem. This has given rise to many suggestions for modifications of the perceptron structure. An important possibility is to concatenate multiple single layer perceptrons so that the output of the nonlinear layer of one perceptron becomes the input to the linear layer of the next, giving rise to deep neural networks, which we will discuss later.

## 3.2.1 Training of Neural Networks

Given a set of state-cost training pairs ( x s ↪ β s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , the parameters of the neural network A , b , and r are obtained by solving the problem

<!-- formula-not-decoded -->

Note that the cost function of this problem is generally nonconvex, so there may exist multiple local minima.

In practice it is common to augment the cost function of this problem with a regularization function, such as a quadratic in the parameters A , b , and r . This is customary in least squares problems in order to make the problem easier to solve algorithmically. However, in the context of neural network training, regularization is primarily important for a di ff erent reason: it helps to avoid overfitting , which occurs when the number of parameters of the neural network is relatively large (comparable to the size of the training set). In this case a neural network model matches the training data very well but may not do as well on new data. This is a well-known di ffi culty, which has been researched extensively in machine learning. It is also the subject of much current research, particularly in the context of deep neural networks, as we will discuss later.

An important issue is to select a method to solve the training problem (3.12). While we can use any unconstrained optimization method that is based on gradients, in practice it is important to take into account the cost function structure of problem (3.12). The salient characteristic of this cost function is that it is the sum of a potentially very large number q of component functions. This makes the computation of the cost function value of the training problem and/or its gradient very costly. For this reason

the incremental methods of Section 3.1.3 are universally used for training. Experience has shown that these methods can be vastly superior to their nonincremental counterparts in the context of neural network training.

The implementation of the training process has benefited from experience that has been accumulated over time, and has provided guidelines for scaling, regularization, initial parameter selection, and other practical issues; we refer to books on neural networks such as Bishop [Bis95], Bishop and Bishop [BiB24], Goodfellow, Bengio, and Courville [GBC16], and Haykin [Hay08], as well as to the overview paper on deep neural network training by Sun [Sun19], and the references quoted therein. Still, incremental methods can be quite slow, and training may be a time-consuming process. Fortunately, the training is ordinarily done o ff -line, possibly using parallel computation, in which case computation time may not be a serious issue. Moreover, in practice the neural network training problem typically need not be solved with great accuracy. This is also supported by the Newton step view of approximation in value space, which suggests that great accuracy in the o ff -line training of the terminal cost approximation is not critically important for good performance of the on-line play algorithm.

## 3.2.2 Multilayer and Deep Neural Networks

An important generalization of the single layer perceptron architecture involves a concatenation of multiple layers of linear and nonlinear functions; see Fig. 3.2.5. In particular the outputs of each nonlinear layer become the inputs of the next linear layer. In some cases it may make sense to add as additional inputs some of the components of the state x or the state encoding y ( x ).

In the early days of neural networks practitioners tended to use few nonlinear layers (say one to three). For example, Tesauro's backgammon program and its descendants have performed well with one or two nonlinear layers [PaR12]. However, more recently a lot of success in certain problem domains (including image and speech processing, large language models, as well as approximate DP) has been achieved with deep neural networks , which involve a considerably larger number of layers.

There are a few questions to consider here. The first has to do with the reason for having multiple nonlinear layers, when a single one is su ffi cient to guarantee the universal approximation property. Here are some qualitative (and somewhat speculative) explanations:

The incremental methods are valid for an arbitrary order of component selection within the cycle, but it is common to randomize the order at the beginning of each cycle. Also, in a variation of the basic method, we may operate on a batch of several components at each iteration, called a minibatch , rather than a single component. This has an averaging e ff ect, which reduces the tendency of the method to oscillate and allows for the use of a larger stepsize; see the end-of-chapter references.

1 0 -1 Encoding

Figure 3.2.5 A deep neural network, with multiple layers. Each nonlinear layer constructs the inputs of the next linear layer.

<!-- image -->

- (a) We may consider the possibility of using linear layers with a particular sparsity pattern, or other structure that embodies special linear operations such as convolution, which may be well-matched to the training problem at hand. Moreover, when such structures are used, the training problem often becomes easier, because the number of parameters in the linear layers is drastically decreased.
- (b If we view the outputs of each nonlinear layer as features, we see that the multilayer network produces a hierarchy of features, where each set of features is a function of the preceding set of features [except for the first set of features, which is a function of the encoding y ( x ) of the state x ]. In the context of specific applications, this hierarchical structure can be exploited to specialize the role of some of the layers and to enhance some characteristics of the state.
- (c) Overparametrization (more weights than data, as in a deep neural network) helps to mitigate the detrimental e ff ects of overfitting, and the attendant need for regularization. The explanation for this fascinating phenomenon (observed as early as the late 90s) is the subject of much recent research; see [ZBH16], [BMM18], [BRT18], [SJL18], [ADH19], [BLL19], [HMR19], [MVS19], [SuY19], [Sun19], [DMB21], [HaR21], [VLK21], [ZBH21] for representative works.

We finally note that the use of deep neural networks has been an important factor for the success of the AlphaGo and AlphaZero programs, as well as for large language models. New developments in hardware, software, architectural structures, training methodology, and integration with other types of neural networks such as transformers, are likely to improve the already enormous power of deep neural networks, and to allow the use of ever larger datasets, which are becoming increasingly available.

## 3.3 LEARNING COST FUNCTIONS IN APPROXIMATE DP

In the context of approximate DP/RL, architectures are mainly used to

approximate either cost functions or policies. When a neural network is involved, the terms value network and policy network are commonly used, respectively. In this section we will illustrate the use of value networks in finite horizon DP, while in the next section we will discuss the use of policy networks. We will also illustrate in Section 3.3.3 the combined use of policy and value networks within an approximate policy iteration context, whereby the policies and their cost functions are approximated by a policy and a value network, respectively, to generate a sequence of (approximately) improved policies. Finally, in Sections 3.3.6 and 3.3.7, we will describe how approximating Q-factor di ff erences or cost di ff erences (rather than Q-factors or costs) can be beneficial within our context of approximation in value space.

## 3.3.1 Fitted Value Iteration

Let us describe a popular approach for training an approximation architecture ˜ J k ( x k ↪ r k ) for a finite horizon DP problem. The parameter vectors r k are determined sequentially, starting from the end of the horizon, and proceeding backwards as in the DP algorithm: first r N -1 then r N -2 , and so on. The algorithm samples the state space for each stage k , and generates a large number of states x s k , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q . It then determines sequentially the parameter vectors r k to obtain a good 'least squares fit' to the DP algorithm. The method can also be used in the infinite horizon case, in essentially identical form, and it is commonly called fitted value iteration .

In particular, each r k is determined by generating a large number of sample states and solving a least squares problem that aims to minimize the error in satisfying the DP equation for these states at time k . At the typical stage k , having obtained r k +1 , we determine r k from the least squares problem

<!-- formula-not-decoded -->

where x s k , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ are the sample states that have been generated for the k th stage. Since r k +1 is assumed to be already known, the complicated minimization term in the right side of this equation is the known scalar

<!-- formula-not-decoded -->

The alternative terms critic network and actor network are also used often. In this book, we will use the terms 'value network' and 'policy network.'

so that r k is obtained as

<!-- formula-not-decoded -->

The algorithm starts at stage N -1 with the minimization

<!-- formula-not-decoded -->

and ends with the calculation of r 0 at k = 0.

In the case of a linear architecture, where the approximate cost-to-go functions are

<!-- formula-not-decoded -->

the least squares problem (3.14) greatly simplifies, and admits the closed form solution

<!-- formula-not-decoded -->

cf. Eq. (3.7). For a nonlinear architecture such as a neural network, incremental gradient algorithms may be used.

An important implementation issue is how to select the sample states x s k , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ k = 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ N -1. In practice, they are typically obtained by some form of Monte Carlo simulation, but the distribution by which they are generated is important for the success of the method. In particular, it is important that the sample states are 'representative' in the sense that they are visited often under a nearly optimal policy. More precisely, the frequencies with which various states appear in the sample should be roughly proportional to the probabilities of their occurrence under an optimal policy.

Aside from the issue of selection of the sampling distribution that we have just described, a di ffi culty with fitted value iteration arises when the horizon N is very long, since then the total number of parameters over the N stages may become excessive. In this case, however, the problem is often stationary, in the sense that the system and cost per stage do not change as time progresses. Then it may be possible to treat the problem as one with an infinite horizon and bring to bear additional methods for training approximation architectures; see the relevant discussions in Chapter 5 of the book [Ber19a].

We finally note an important di ffi culty with the training method of this section: the calculation of each sample β s k of Eq. (3.13) requires a minimization of an expected value, which can be very time consuming. In the next section, we describe an alternative type of fitted value iteration, which uses Q-factors, and involves a simpler minimization, whereby the order of the minimization and expectation operations in Eq. (3.13) is reversed.

## 3.3.2 Q-Factor Parametric Approximation - Model-Free Implementation

We will now consider an alternative form of approximation in value space and fitted value iteration, which involves approximation of the optimal Q-factors of state-control pairs ( x k ↪ u k ) at time k , with no intermediate approximation of cost-to-go functions. An important characteristic of this algorithm is that it allows for a model-free computation (i.e., the use of a computer model in place of a mathematical model).

We recall that the optimal Q-factors are defined by

<!-- formula-not-decoded -->

where J * k +1 is the optimal cost-to-go function for stage k +1. Thus Q * k ( x k ↪ u k ) is the cost attained by using u k at state x k , and subsequently using an optimal policy.

As noted in Section 1.3, the DP algorithm can be written as

<!-- formula-not-decoded -->

and by using this equation, we can write Eq. (3.15) in the following equivalent form that relates Q * k with Q * k +1 :

<!-- formula-not-decoded -->

This suggests that in place of the Q-factors Q * k ( x k ↪ u k ), we may use Q-factor approximations as the basis for suboptimal control.

We can obtain such approximations by using methods that are similar to the ones we have considered so far. Parametric Q-factor approximations ˜ Q k ( x k ↪ u k ↪ r k ) may involve a neural network, or a feature-based linear architecture. The feature vector may depend on just the state, or on both the state and the control. In the former case, the architecture has the form

<!-- formula-not-decoded -->

where r k ( u k ) is a separate weight vector for each control u k . In the latter case, the architecture has the form

<!-- formula-not-decoded -->

where r k is a weight vector that is independent of u k . The architecture (3.17) is suitable for problems with a relatively small number of control options at each stage. In what follows, we will focus on the architecture (3.18), but the discussion, with few modifications, also applies to the architecture (3.17) and to nonlinear architectures as well.

We may adapt the fitted value iteration approach of the preceding section to compute sequentially the parameter vectors r k in Q-factor parametric approximations, starting from k = N -1. This algorithm is based on Eq. (3.16), with r k obtained by solving least squares problems similar to the ones of the cost function approximation case [cf. Eq. (3.14)]. As an example, the parameters r k of the architecture (3.18) are computed sequentially by collecting sample state-control pairs ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and solving the linear least squares problems

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Thus, having obtained r k +1 , we obtain r k through a least squares fit that aims to minimize the sum of the squared errors in satisfying Eq. (3.16). Note that the solution of the least squares problem (3.19) can be obtained in closed form as

<!-- formula-not-decoded -->

[cf. Eq. (3.7)]. Once r k has been computed, the one-step lookahead control ˜ θ k ( x k ) is obtained on-line as

<!-- formula-not-decoded -->

without the need to calculate any expected value. This latter property is a primary incentive for using Q-factors in approximate DP, particularly when there are tight constraints on the amount of on-line computation that is possible in the given practical setting.

The samples β s k of Eq. (3.20) involve the exact computation of an expected value. In an alternative implementation, we may replace β s k with an average of just a few samples (even a single sample) of the random variable

<!-- formula-not-decoded -->

collected according to the probability distribution of w k . This distribution may either be known explicitly, or in a model-free situation, through a computer simulator. In particular, to implement this scheme, we only need a simulator that for any pair ( x k ↪ u k ) generates samples of the next stage cost and state

<!-- formula-not-decoded -->

according to the distribution of w k .

Note that the samples of the random variable (3.22) do not require the computation of an expected value like the samples (3.13) in the cost approximation method of the preceding chapter. Moreover the samples of (3.22) involve a simpler minimization than the samples (3.13). This is an important advantage of working with Q-factors rather than state costs.

Having obtained the weight vectors r 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r N -1 , and hence the onestep lookahead policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ through Eq. (3.21), a further possibility is to approximate this policy with a parametric architecture. This is approximation in policy space built on top of approximation in value space . The idea here is to simplify even further the on-line computation of the suboptimal controls by avoiding the minimization of Eq. (3.21); see the discussion in Section 1.3.3.

## 3.3.3 Parametric Approximation in Infinite Horizon Problems - Approximate Policy Iteration

In this section we will briefly discuss parametric approximation methods for infinite horizon problems, based on policy iteration (PI). We will focus on the finite-state version of the α -discounted problem of Section 1.4.1, using notation that is more convenient for such problems. In particular, states and successor states will be denoted by i and j , respectively. Moreover the system equation will be represented by control-dependent transition probabilities p ij ( u ) (the probability that the system will move to state j , given that it starts at state i and control u is applied). For a state-control pair ( i↪ u ), the average cost per stage is denoted by g ( i↪ u↪ j ).

We recall that the PI algorithm in its exact form produces a sequence of stationary policies whose cost functions are progressively improving and converge in a finite number of iterations to the optimal. The convergence proof relies on the fundamental cost improvement property of PI, and depends on the finiteness of the state and control spaces. This proof, together

with other PI-related convergence proofs, can be found in the author's textbooks [Ber17a] or [Ber19a].

Let us state the exact form of the PI algorithm in terms of Q -factors, and in a way that facilitates approximations and simulation-based implementations. Given any policy θ , it generates the next policy ˜ θ with a two-step process as follows (cf. Section 1.4.1):

- (a) Policy evaluation : We compute the cost function J θ of θ and its associated Q -factors, which are given by

<!-- formula-not-decoded -->

Thus Q θ ( i↪ u ) is the cost of starting at i , using u at the first stage, and then using θ for the remaining stages.

- (b) Policy improvement : We compute the new policy ˜ θ according to

<!-- formula-not-decoded -->

Let us now describe one approach to approximating the two steps of the PI process:.

- (a) Approximate policy evaluation : Here we introduce a parametric architecture ˜ Q θ ( i↪ u↪ r ) for the Q -factors of θ . We determine the value of the parameter vector r by generating (using a simulator of the system) a large number of training triplets ( i s ↪ u s ↪ β s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and by using a least squares fit:

<!-- formula-not-decoded -->

In particular, for a given pair ( i s ↪ u s ), the scalar β s is generated by starting at i s , using u s at the first stage, and simulating a trajectory of states and controls using θ for some number k of subsequent stages. Thus, β s is a sample of Q k θ ( i s ↪ u s ), the k -stage Q -factor of θ , which in the limit as k →∞ yields the infinite horizon Q -factor of θ . The number of stages k may be either large, or fairly small. However, in the latter case some terminal cost function approximation should be added at the end of the k -stage trajectory, to compensate for the di ff erence ∣ ∣ Q θ ( i↪ u ) -Q k θ ( i↪ u ) ∣ ∣ , which decreases in proportion to α k , and may be large when k is small. Such a function may be obtained with additional training or from a previous iteration.

- (b) Approximate policy improvement : Here we compute the new policy ˜ θ according to

<!-- formula-not-decoded -->

where r is the parameter vector obtained from the policy evaluation formula (3.23).

An important alternative for approximate policy improvement, is to compute a set of pairs ( i s ↪ ˜ θ ( i s ) ) , s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , using Eq. (3.24), and fit these pairs with a policy approximation architecture (see the next section on approximation in policy space). The overall scheme then becomes a form of PI that is based on approximation in both value and policy spaces.

At the end of the last policy evaluation step of PI, we have obtained a final Q-factor approximation ˜ Q ( i↪ u↪ ˜ r ). Then, in on-line play mode, we may apply the policy

<!-- formula-not-decoded -->

i.e., use the (would be) next policy iterate. Alternatively, we may apply the one-step lookahead policy

<!-- formula-not-decoded -->

or its multistep lookahead version. The latter alternative implements a Newton step and will likely result in substantially better performance. However, it is more time consuming, particularly if it is implemented by using a computer model and model-free simulation. Still another possibility, which also implements a Newton step, is to replace the function

<!-- formula-not-decoded -->

in the preceding Eq. (3.25) with an o ff -line trained approximation.

## Challenges Relating to Approximate Policy Iteration

Approximate PI in its various forms has been the subject of extensive theoretical and applied research. A comprehensive discussion is beyond our scope, and we refer to the literature, for detailed accounts, including the DP textbook [Ber12] or the RL textbook [Ber19a]. Let us provide a few comments relating to the challenges of approximate PI implementation.

- (a) Architectural issues : The architecture ˜ Q θ ( i↪ u↪ r ) may involve the use of features, and it could be linear, or it could be nonlinear such as a neural network. A major advantage of a linear feature-based architecture is that the policy evaluation (3.23) is a linear least squares problem, which admits a closed-form solution. Moreover, when linear architectures are used, there is a broader variety of approximate policy evaluation methods with solid theoretical performance guarantees,

such as TD( λ ), LSTD( λ ), and LSPE( λ ), which are not described in this book, but are discussed extensively in the literature. Generally, identifying an architecture that fits the problem well and training it e ff ectively can be a challenging and time-intensive task.

- (b) Exploration issues : Generating an appropriate set of training triplets ( i s ↪ u s ↪ β s ) at the policy evaluation step poses considerable challenges. A major di ffi culty has to do with inadequate exploration , the Achilles' heel of approximate PI. In particular, straightforward ways to evaluate a policy θ , typically rely on Q -factor samples of θ starting from states frequently visited by θ . Unfortunately, this may bias the simulation by underrepresenting states that are unlikely to occur under θ . As a result, the Q -factor estimates of these underrepresented states may be highly inaccurate, potentially causing serious errors in the calculation of the improved control policy ˜ θ via the policy improvement Eq. (3.24).

One way to address this issue is to use a large number of initial states to form a rich and representative subset of the state space. To keep simulation costs manageable, it may be necessary to use relatively short trajectories. However, when using short trajectories it may be important to introduce a terminal cost function approximation in the policy evaluation step in order to make the cost sample β s more accurate. Other approaches to enhance exploration include the use of a so-called o ff -policy , i.e., a policy θ ′ other than the currently evaluated policy θ , which tends to visit states that are unlikely to be visited using θ . For further discussion, see Section 6.4 of the DP textbook [Ber12].

- (c) Oscillation issues : Contrary to exact PI, which is guaranteed to yield an optimal policy, approximate PI produces a sequence of policies, which are only guaranteed to lie asymptotically within a certain error bound from the optimal; see the books [BeT96], Section 6.2.2, [Ber12], Section 2.5, and [Ber19a], Section 5.3.5. Moreover, the generated policies may oscillate. By this we mean that after a few iterations, policies tend to repeat in cycles.

The associated parameter vectors r may also tend to oscillate, although it is possible that there is convergence in parameter space and oscillation in policy space. This phenomenon, known as chattering , is explained in the author's survey papers [Ber10c], [Ber11b], and book [Ber12] (Section 6.4.3), and can be quite problematic, because there is no guarantee that the policies involved in the oscillation are 'good' policies, and it is often di ffi cult to assess their performance relative to the optimal. We note, however, that oscillations can be avoided and approximate PI can be shown to converge under special conditions, which arise in particular when an aggregation approach is

used; see the approximate PI survey [Ber11b].

We refer to the literature for further discussion of the preceding issues, as well as a variety of other approximate PI methods.

## 3.3.4 Optimistic Policy Iteration with Parametric Q-Factor Approximation - SARSA and DQN

There are also 'optimistic' approximate PI methods with Q-factor approximation, and/or a few samples in between policy updates. We view these primarily as o ff -line training methods, but because of the limited number of samples between policy updates, they have the potential of on-line implementation. In this case, however, a number of di ffi culties must be overcome, as we will explain later in this section.

As an example, let us consider an extreme version of Q-factor parametric approximation that uses a single sample between policy updates. At the start of iteration k , we have the current parameter vector r k , we are at some state i k , and we have chosen a control u k . Then:

- (1) We simulate the next transition ( i k ↪ i k +1 ) using the transition probabilities p i k j ( u k ) glyph[triangleright]
- (2) We generate the control u k +1 with the minimization

<!-- formula-not-decoded -->

[In some schemes, to enhance exploration, u k +1 is chosen with a small probability to be a random element of U ( i k +1 ) or one that is ' /epsilon1 -greedy,' i.e., attains within some /epsilon1 the minimum above. This is commonly referred to as the use of an o ff -policy .]

- (3) We update the parameter vector via

<!-- formula-not-decoded -->

where γ k is a positive stepsize, and ∇ ( · ) denotes gradient with respect to r evaluated at the current parameter vector r k . In the special case where ˜ Q is a linear feature-based architecture, ˜ Q ( i↪ u↪ r ) = φ ( i↪ u ) ′ r , the gradient ∇ ˜ Q ( i k ↪ u k ↪ r k ) is just the feature vector φ ( i k ↪ u k ), and iteration (3.27) becomes

<!-- formula-not-decoded -->

Thus r k is changed in an incremental gradient direction : the one opposite to the gradient (with respect to r ) of the incremental error

<!-- formula-not-decoded -->

evaluated at the current iterate r k .

The process is now repeated with r k +1 , i k +1 , and u k +1 replacing r k , i k , and u k , respectively.

Extreme optimistic schemes of the type just described have received a lot of attention, in part because they admit a model-free implementation [i.e., the use of a simulator, which provides for each pair ( i k ↪ u k ), the next state i k +1 and corresponding cost g ( i k ↪ u k ↪ i k +1 ) that are needed in Eq. (3.27)]. They are often referred to as SARSA (State-Action-Reward-StateAction); see e.g., the books [BeT96], [BBD10], [SuB18]. When Q-factor approximation is used, their behavior is very complex, their theoretical convergence properties are unclear, and there are no associated performance bounds in the literature. In practice, SARSA is often implemented in a less extreme/optimistic form, whereby several (perhaps many) statecontrol-transition cost-next state samples are batched together and suitably averaged before updating the vector r k .

Other variants of the method attempt to reduce sampling e ff ort by storing the generated samples in a bu ff er for reuse in subsequent iterations through some randomized process (cf. our earlier discussion of exploration). This is also called sometimes experience replay , an idea that has been used since the early days of RL, both to save in sampling e ff ort and to enhance exploration. The DQN (Deep Q Network) scheme, championed by DeepMind (see Mnih et al. [MKS15]), is based on this idea (the term 'Deep' is a reference to DeepMind's a ffi nity for deep neural networks, but the idea of experience replay does not depend on the use of a deep neural network architecture).

Another interesting idea from DeepMind [MRM16] is to introduce asynchronous parallel computation into the algorithm, based on the theory of distributed asynchronous methods in DP, gradient optimization, and RL, by Bertsekas and Tsitsiklis [Ber82a], [TBA86], [BeT89], [Tsi94], [BeT00], [Ber19a].

## Q-Learning Algorithms and On-Line Play

Algorithms that approximate Q-factors, including SARSA and DQN, are fundamentally o ff -line training algorithms. This is because their training process is long and requires the collection of many samples before reaching a stage that resembles parameter convergence. It can thus be unreliable to use the interim approximate Q-factors for on-line decision making, particularly in an adaptive context that involves changing system parameters, thereby requiring on-line replanning.

On the other hand, compared to the approximate PI method of Section 3.3.3, SARSA and DQN are far better suited for on-line implementation, because the control generation process of Eq. (3.26) can also be used to select controls on-line, thereby facilitating the combination of training and on-line control selection. To this end, it is important, among others,

to make sure that the parameters r k stay at 'safe' levels during the on-line control process, which can be a challenge. Still, even if this di ffi culty can be overcome in the context of a given problem, there are a number of other di ffi culties that SARSA and DQN can encounter during on-line play.

- (a) On-line exploration issues : There is a need to occasionally select controls using an o ff -policy in order to enhance exploration, and finding an adequate o ff -line policy in a given practical context can be a challenge. Moreover, the o ff -policy controls may improve exploration, but may be of poor quality, and in some contexts, may induce instability.
- (b) Robustness and replanning issues : In an adaptive control context where the problem parameters are changing, the algorithm may be too slow to adapt to the changes.
- (c) Performance degradation issues : Similar to our earlier discussion [cf. the comparison of Eqs. (3.24) and (3.25)], the minimization of Eq. (3.26) does not implement a Newton step, thereby resulting in performance loss. The alternative implementation

<!-- formula-not-decoded -->

which is patterned after Eq. (3.26), is better in this regard, but is computationally more costly, and thus less suitable for on-line implementation.

Generally speaking, the combination of o ff -line training and on-line play with the use of SARSA and DQN poses serious challenges. Nevertheless, encouraging results have been achieved in specific contexts, often with 'manual tuning,' i.e., tuning tailored to the problem at hand. Moreover, the popularity of these methods has been bolstered by the availability of open-source software that allow model-free implementations.

## 3.3.5 Approximate Policy Iteration for Infinite Horizon POMDP

In this section, we consider partial observation Markovian decision problems (POMDP) with a finite number of states and controls, and discounted additive cost over an infinite horizon. As discussed in Section 1.6.6, the optimal solution is typically intractable, so approximate DP/RL approaches must be used. In this section we focus on PI methods that are based on rollout, and approximations in policy and value space. They update a policy by using truncated rollout with that policy and a terminal cost function approximation. We focus on cost function approximation schemes, but Q-factor approximation is also possible.

Due to its simulation-based rollout character, the methodology of this section relies critically on the finiteness of the control space. It can be extended to POMDP with infinite state space but finite control space, although we will not consider this possibility in this section.

We assume that there are n states denoted by i ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ and a finite set of controls U at each state. We denote by p ij ( u ) and g ( i↪ u↪ j ) the transition probabilities and corresponding transition costs, from i to j under u ∈ U . The cost is accumulated over an infinite horizon and is discounted by α ∈ (0 ↪ 1). At each new generated state j , an observation z from a finite set Z is obtained with known probability p ( z ♣ j↪ u ) that depends on j and the control u that was applied prior to the generation of j . The objective is to select each control optimally as a function of the prior history of observations and controls.

A classical approach to this problem is to convert it to a perfect state information problem whose state is the current belief b = ( b 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ b n ), where b i is the conditional distribution of the state i given the prior history. As noted in Section 1.6.6, b is a su ffi cient statistic, which can serve as a substitute for the set of available observations, in the sense that optimal control can be achieved with knowledge of just b .

In this section, we consider a generalized form of su ffi cient statistic, which we call the feature state and we denote by y . We require that the feature state y subsumes the belief state b . By this we mean that b can be computed exactly knowing y . One possibility is for y to be the union of b and some distinguishable characteristics of the belief state, or some compact representation of the measurement history up to the current time (such as a number of most recent measurements, or the state of a finitestate controller).

We further assume that y can be sequentially generated using a known feature estimator F ( y↪ u↪ z ). By this we mean that given that the current feature state is y , control u is applied, and observation z is obtained, the next feature can be exactly predicted as F ( y↪ u↪ z ).

Clearly, since b is a su ffi cient statistic, the same is true for y . Thus the optimal costs achievable by the policies that depend on y and on b are the same. However, specific suboptimal schemes may become more e ff ective with the use of the feature state y instead of just the belief state b .

The optimal cost J * ( y ), as a function of the su ffi cient statistic/feature state y , is the unique solution of the corresponding Bellman equation

<!-- formula-not-decoded -->

Here we use the following notation:

b y is the belief state that corresponds to feature state y , with components denoted by b y↪i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n .

Transition Cost ik+1

Original

Observer

2k+1

Feature Estimator

F (y, u, 2)

System

Policy yk+1

4k+1

44+1

Figure 3.3.1 Composite system simulator for POMDP for a given policy. The starting state i k at stage k of a trajectory is generated randomly using the belief state b k , which is in turn computed from the feature state y k .

<!-- image -->

ˆ g ( y↪ u ) is the expected cost per stage

<!-- formula-not-decoded -->

ˆ p ( z ♣ b y ↪ u ) is the conditional probability that the next observation will be z given the current belief state b y and control u

F is the feature state estimator. In particular, F ( y↪ u↪ z ) is the next feature vector, when the current feature state is y , control u is applied, and observation z is obtained.

The feature space reformulation of the problem can serve as the basis for approximation in value space, whereby J * is replaced in Bellman's equation by some function ˜ J after one-step or multistep lookahead. For example a one-step lookahead scheme yields the suboptimal policy ˜ θ given by

<!-- formula-not-decoded -->

In /lscript -step lookahead schemes, ˜ J is used as terminal cost function in an /lscript -step horizon version of the original infinite horizon problem. In the standard form of a rollout algorithm, ˜ J is the cost function of some base policy. We will next discuss a rollout scheme with /lscript -step lookahead, which involves rollout truncation and terminal cost approximation.

## Truncated Rollout with Terminal Cost Function Approximation

In the pure form of the rollout algorithm, the cost function approximation ˜ J is the cost function J θ of a known base policy θ , and its value ˜ J ( y ) = J θ ( y )

"k+1

Transition Cost ix+1

Original

System

Uk +1

Observer

24 +2

Feature Estimator

1k +2

Policy ik +2

4k+2

at any y is obtained by first extracting b from y , and then running a simulator starting from b , and using the system model, the feature generator, and θ . In the truncated form of rollout, ˜ J ( y ) is obtained by running the simulator of θ for a given number of steps m , and then adding a terminal cost approximation ˆ J (¯ y ) for each terminal feature state ¯ y that is obtained at the end of the m steps of the simulation with θ (see Fig. 3.3.1).

Thus the rollout policy is defined by the base policy θ , the terminal cost function approximation ˆ J , the number of steps m after which the simulated trajectory with θ is truncated, and the lookahead size /lscript . The choices of m and /lscript are typically made by trial and error, based on computational tractability among other considerations, while ˆ J may be chosen on the basis of problem-dependent insight or through the use of some o ff -line approximation method. In some variants of the method, the multistep lookahead may be implemented approximately using a Monte Carlo tree search or adaptive sampling scheme.

Using m -step rollout between the /lscript -step lookahead and the terminal cost approximation gives the method the character of a single PI. We will use repeated truncated rollout as the basis for constructing a PI algorithm, which we will discuss next.

## Supervised Learning of Rollout Policies and Cost Functions Approximate Policy Iteration

The rollout algorithm uses multistep lookahead and on-line simulation of the base policy to generate the rollout control at any feature state of interest. To avoid the cost of on-line simulation, we can approximate the rollout policy o ff -line by using some approximation architecture, potentially involving a neural network. This is policy approximation built on top of the rollout scheme.

To this end, we may introduce a parametric family/architecture of policies of the form ˆ θ ( y↪ r ), where r is a parameter vector. We then construct a training set that consists of a large number of sample feature state-control pairs ( y s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , such that for each s , u s is the rollout control at feature state y s . We use this data set to obtain a parameter ¯ r by solving a corresponding classification problem, which associates each feature state y with a control ˆ θ ( y↪ ¯ r ). The parameter ¯ r defines a classifier, which given a feature state y , classifies y as requiring control ˆ θ ( y↪ ¯ r ) (see Section 3.4).

We can also apply the rollout policy approximation to the context of PI. The idea is to view rollout as a single policy improvement, and to view the PI algorithm as a perpetual rollout process , which performs multiple policy improvements, using at each iteration the current policy as the base policy, and the next policy as the corresponding rollout policy.

In particular, we consider a PI algorithm where at the typical iteration we have a policy θ , which we use as the base policy to generate

many feature state-control sample pairs ( y s ↪ u s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , where u s is the rollout control corresponding to feature state y s . We then obtain an 'improved' policy ˆ θ ( y↪ r ) with an approximation architecture and a classification algorithm, as described above. The 'improved' policy is then used as a base policy to generate samples of the corresponding rollout policy, which is approximated in policy space, etc.

To use truncated rollout in this PI scheme, a terminal cost approximation is required, which can take a variety of forms. Using zero is a simple possibility, which may work well if either the size /lscript of multistep lookahead or the length m of the rollout is relatively large. Another possibility is to use as terminal cost in the truncated rollout an approximation of the cost function of some base policy, which may be obtained with a neural network-based approximation architecture.

In particular, at any policy iteration with a given base policy, once the rollout data is collected, one or two neural networks are constructed: A policy network that approximates the rollout policy, and (in the case of rollout with truncation) a value network that constructs a cost function approximation for that rollout policy. Thus, we may consider two types of methods:

- (a) Approximate rollout and PI with truncation , where each generated policy as well as its cost function are approximated by a policy and a value network, respectively. The cost function approximation of the current policy is used to truncate the rollout trajectories that are used to train the next policy.
- (b) Approximate rollout and PI without truncation , where each generated policy is approximated using a policy network, but the rollout trajectories are continued up to a large maximum number of stages (enough to make the cost of the remaining stages insignificant due to discounting) or upon reaching a termination state. The advantage of this scheme is that only a policy network is needed; a value network is unnecessary since there is no rollout truncation with cost function approximation at the end.

Note that as in all approximate PI schemes, the sampling of feature states used for training is subject to exploration concerns. In particular, for each policy approximation, it is important to include in the sample set ¶ y s ♣ s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q ♦ , a subset of feature states that are 'favored' by the rollout trajectories; e.g., start from some initial subset of feature states y s and selectively add to this subset feature states that are encountered along the rollout trajectories. This is a challenging issue, which must be approached with care.

An extensive case study of the methodology of this section was given in the paper by Bhattacharya et al. [BBW20], for the case of a pipeline repair problem. The implementation used there also includes the use of

a partitioned state space architecture and an asynchronous distributed algorithm for o ff -line training; see Section 3.4.2. See also the paper by Bhattacharya et al. [BKB23], which deals with large multiagent POMDP problems, involving multiple robots operating on a network.

## 3.3.6 Advantage Updating - Approximating Q-Factor Di ff erences

Let us now explore an important alternative to computing Q-factor approximations. It is motivated by the potential benefit of approximating Q-factor di ff erences rather than Q-factors. In this method, called advantage updating , instead of computing and comparing Q * k ( x k ↪ u k ) for all u k ∈ U k ( x k ), we compute

<!-- formula-not-decoded -->

The function A k ( x k ↪ u k ) can serve to compare controls, i.e., at state x k select

<!-- formula-not-decoded -->

and this can also be done when A k ( x k ↪ u k ) is approximated with a value network.

Note that in the absence of approximations, selecting controls by advantage updating is equivalent to selecting controls by comparing their Q-factors. By contrast, when approximation is involved, comparing advantages instead of Q-factors can be important, because the former may have a much smaller range of values than the latter. In particular, Q * k may embody sizable quantities that depend on x k but are independent of u k , and which may interfere with algorithms such as the fitted value iteration (3.19)-(3.20). Thus, when training an architecture to approximate Q * k , the training algorithm may naturally try to capture the large scale behavior of Q * k , which may be irrelevant because it may not be reflected in the Q-factor di ff erences A k . However, with advantage updating, we may instead focus the training process on finer scale variations of Q * k , which may be all that matters. Here is an example (first given in the book [BeT96]) of what can happen when trained approximations of Q-factors are used.

## Example 3.3.1

Consider the deterministic scalar linear system

<!-- formula-not-decoded -->

and the quadratic cost per stage

<!-- formula-not-decoded -->

where δ is a very small positive constant [think of δ -discretization of a continuoustime problem involving the di ff erential equation dx ( t ) glyph[triangleleft]dt = u ( t )]. Let us focus on the stationary policy π , which applies at state x the control

<!-- formula-not-decoded -->

and view it as the base policy of a rollout algorithm. The Q-factors of π over an infinite number of stages can be calculated to be

<!-- formula-not-decoded -->

(We omit the details of this calculation, which is based on the classical analysis of linear-quadratic optimal control problems; see e.g., Section 1.5, or [Ber17a], Section 3.1.) Thus the important part of Q π ( x↪ u ) for the purpose of rollout policy computation is

<!-- formula-not-decoded -->

However, when a value network is trained to approximate Q π ( x↪ u ), the approximation will be dominated by 5 x 2 4 , and the important part (3.28) will be 'lost' when δ is very small. By contract, the advantage function can be calculated to be

<!-- formula-not-decoded -->

and when approximated with a value network, the approximation will be essentially una ff ected by δ .

## The Use of a Baseline

The idea of advantage updating is also related to the useful technique of subtracting a suitable constant (often called a baseline ) from a quantity that is estimated; see Fig. 3.3.2 (in the case of advantage updating, the baselines depend on x k , but the same general idea applies). This idea can also be used in the context of the fitted value iteration method given earlier, as well as in conjunction with other simulation-based methods in RL.

Example 3.1.1 also points to the connection between the ideas underlying advantage updating and the rollout methods for small stage costs relative to the cost function approximation, which we discussed in Section 2.6. In both cases it is necessary to avoid including terms of disproportionate size in the target function that is being approximated. The remedy in both cases is to subtract from the target function a suitable state-dependent baseline.

u u

Figure 3.3.2 Illustration of the idea of subtracting a baseline constant from a cost or Q-factor approximation. Here we have samples h ( u 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ h ( u q ) of a scalar function h ( u ) at sample points u 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ u q , and we want to approximate h ( u ) with a linear function ˜ h ( u↪ r ) = ru , where r is a scalar tunable weight. We subtract a baseline constant b from the samples, and we solve the problem

<!-- image -->

<!-- formula-not-decoded -->

By properly adjusting b , we can improve the quality of the approximation, which after subtracting b from all the sample values, takes the form ˜ h ( u↪ b↪ r ) = b + ruglyph[triangleright] Conceptually, b serves as an additional weight (multiplying the basis function 1), which enriches the approximation architecture.

## 3.3.7 Di ff erential Training of Cost Di ff erences for Rollout

Let us now consider ways to approximate Q-factor di ff erences (cf. our advantage updating discussion of the preceding section) by approximating cost function di ff erences first. We recall here that given a base policy π = ¶ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ θ N -1 ♦ , the o ff -line computation of an approximate rollout policy ˜ π = ¶ ˜ θ 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ˜ θ N -1 ♦ consists of two steps:

- (1) In a preliminary phase, we compute approximations ˜ J k to the cost functions J k↪ π of the base policy π , possibly using simulation and a least squares fit from a parametrized class of functions.
- (2) Given ˜ J k and a state x k at time k , we compute the approximate Q -factor

<!-- formula-not-decoded -->

for all u ∈ U k ( x k ), and we obtain the (approximate) rollout control ˜ θ k ( x k ) from the minimization

<!-- formula-not-decoded -->

Unfortunately, this method also su ff ers from the error magnification inherent in the Q -factor di ff erencing operation. This motivates an alternative approach, called di ff erential training , which is based on cost-to-go di ff erence approximations. To this end, we note that to compute the rollout control ˜ θ k ( x k ), it is su ffi cient to have the di ff erences of costs-to-go

<!-- formula-not-decoded -->

where θ k ( x k ) is the control applied by the base policy at x k .

We thus consider a function approximation approach, whereby given any two states x k +1 and ˆ x k +1 , we obtain an approximation ˜ G k +1 ( x k +1 ↪ ˆ x k +1 ) of the cost di ff erence (3.29). We then compute the rollout control by

<!-- formula-not-decoded -->

where θ k ( x k ) is the control applied by the base policy at x k . Note that the minimization (3.30) aims to simply subtract the approximate Q-factor of the base policy control θ k ( x k ) from the approximate Q-factor of every other control u ∈ U k ( x k ).

An important point here is that the training of an approximation architecture to obtain ˜ G k +1 can be done using any of the standard training methods, and a 'di ff erential' system, whose 'states' are pairs ( x k ↪ ˆ x k ) and will be described shortly. To see this, let us denote for all k and pair of states ( x k ↪ ˆ x k )

<!-- formula-not-decoded -->

the cost function di ff erences corresponding to the base policy π . We consider the DP equations corresponding to π , and to x k and ˆ x k :

<!-- formula-not-decoded -->

and we subtract these equations to obtain

<!-- formula-not-decoded -->

for all ( x k ↪ ˆ x k ) and k . Therefore, G k can be viewed as the cost-to-go function for a problem involving a fixed policy (the base policy), the state ( x k ↪ ˆ x k ), the cost per stage

<!-- formula-not-decoded -->

and the system equation

<!-- formula-not-decoded -->

Thus, it can be seen that any of the standard methods that can be used to train architectures that approximate J k↪ π , can also be used for training architectures that approximate G k . For example, one may use simulationbased methods that generate pairs of trajectories starting at the pair of initial states ( x k ↪ ˆ x k ), and generated according to Eq. (3.32) by using the base policy π . Note that a single random sequence ¶ w 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ w N -1 ♦ may be used to simultaneously generate samples of G k ( x k ↪ ˆ x k ) for several triples ( x k ↪ ˆ x k ↪ k ), and in fact this may have a substantial beneficial e ff ect.

A special case of interest arises when a linear, feature-based architecture is used for the approximator ˜ G k . In particular, let φ k be a feature extraction mapping that associates a feature vector φ k ( x k ) with state x k and time k , and let ˜ G k be of the form

<!-- formula-not-decoded -->

where r k is a tunable weight vector of the same dimension as φ k ( x k ) and prime denotes transposition. The rollout policy is generated by

<!-- formula-not-decoded -->

which corresponds to using r ′ k +1 φ k +1 ( x k +1 ) (plus an unknown inconsequential constant) as an approximation to J k +1 ↪ π ( x k +1 ). Thus, in this approach, we essentially use a linear feature-based architecture to approximate the cost functions J k↪ π of the base policy, but we train this architecture using the di ff erential system (3.32) and the di ff erential cost per stage of Eq. (3.31) . This is done by selecting pairs of initial states, running in parallel the corresponding trajectories using the base policy, and subtracting the resulting trajectory costs from each other.

## 3.4 LEARNING A POLICY IN APPROXIMATE DP

We have focused so far on approximation in value space using parametric architectures. In this section we will discuss how the cost function approximation methods discussed earlier this chapter can be adapted for

the purpose of approximation in policy space, whereby we approximate a given policy by using optimization over a parametric family of some form. Throughout this section we focus on a fixed policy and we focus on the o ff -line training of that policy.

In particular, suppose that for a given stage k , we have access to a dataset of sample state-control pairs ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ obtained through some unspecified control process, such as rollout or problem approximation. We may then wish to 'learn' this process by training the parameter vector r k of a parametric family of policies ˜ θ k ( x k ↪ r k ) ↪ using least squares minimization/regression:

<!-- formula-not-decoded -->

cf. our discussion of approximation in policy space in Section 1.3.3.

## 3.4.1 The Use of Classifiers for Approximation in Policy Space

As we have noted in Section 3.1, in the case of a continuous control space, training of a parametric architecture for policy approximation is similar to training for a cost approximation. In the case where the control space is finite, however, it is useful to make the connection of approximation in policy space with classification ; cf. Fig. 3.1.2 and the discussion of Section 3.1.

Classification is an important subject in machine learning. The objective is to construct an algorithm, called a classifier , which assigns a given 'object' to one of a finite number of 'categories' based on its 'characteristics.' Here we use the term 'object' generically. In some cases, the classification may relate to persons or situations. In other cases, an object may represent a hypothesis, and the problem is to decide which of the hypotheses is true, based on some data. In the context of approximation in policy space, objects correspond to states, and categories correspond to controls to be applied at the di ff erent states . Thus in this case, we view each sample x s k ↪ u s k as an object-category pair.

( ) Generally, in classification we assume that we have a population of objects, each belonging to one of m categories c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m . We want to be able to assign a category to any object that is presented to us. Mathematically, we represent an object with a vector x (e.g., some raw description or a vector of features of the object), and we aim to construct a rule that assigns to every possible object x a unique category c .

To illustrate a popular classification method, let us assume that if we draw an object x at random from this population, the conditional probability of the object being of category c is p ( c ♣ x ). If we know the probabilities p ( c ♣ x ), we can use a classical statistical approach, whereby we assign x to

the category c ∗ ( x ) that has maximal posterior probability, i.e.,

<!-- formula-not-decoded -->

This is called the Maximum a Posteriori rule (or MAP rule for short; see for example the book [BeT08], Section 8.2, for a discussion).

When the probabilities p ( c ♣ x ) are unknown, we may try to estimate them using a least squares optimization, based on the following property, whose proof is outlined in Exercise 3.1.

Proposition 3.4.1: (Least Squares Property of Conditional Probabilities) Let ξ ( x ) be any prior distribution of x , so that the joint distribution of ( c↪ x ) is

<!-- formula-not-decoded -->

For a pair of classes ( c↪ c ′ ), define z ( c↪ c ′ ) by

<!-- formula-not-decoded -->

and for a fixed class c and any function h of ( c↪ x ), consider

<!-- formula-not-decoded -->

the expected value with respect to the distribution ζ ( c ′ ↪ x ) of the random variable ( z ( c↪ c ′ ) -h ( c↪ x ) ) 2 . Then p ( c ♣ x ) minimizes this expected value over all functions h ( c↪ x ), i.e., for all functions h and all classes c , we have

<!-- formula-not-decoded -->

The proposition states that p ( c ♣ x ) is the function of ( c↪ x ) that minimizes

<!-- formula-not-decoded -->

over all functions h of ( c↪ x ), for any prior distribution of x and class c . This suggests that we can obtain approximations to the probabilities p ( c ♣ x ), c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , by minimizing an empirical/simulation-based approximation of the expected value (3.36).

More specifically, let us assume that we have a training set consisting of q object-category pairs ( x s ↪ c s ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q , and corresponding vectors

<!-- formula-not-decoded -->

and let us adopt a parametric approach. In particular, for each category c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , we approximate the probability p ( c ♣ x ) with a function ˜ h ( c↪ x↪ r ) that is parametrized by a vector r , and optimize over r the empirical approximation to the expected squared error of Eq. (3.36). Thus we can obtain r by the least squares regression:

<!-- formula-not-decoded -->

perhaps with some quadratic regularization added. The functions ˜ h ( c↪ x↪ r ) may be provided for example by a feature-based architecture or a neural network.

/negationslash

Note that each training pair ( x s ↪ c s ) is used to generate m examples for use in the regression problem (3.37): m -1 'negative' examples of the form ( x s ↪ 0), corresponding to the m -1 categories c = c s , and one 'positive' example of the form ( x s ↪ 1), corresponding to c = c s . Note also that the incremental gradient method can be applied to the solution of this problem.

The regression problem (3.37) approximates the minimization of the expected value (3.36), so we conclude that its solution ˜ h ( c↪ x↪ ¯ r ), c = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , approximates the probabilities p ( c ♣ x ). Once this solution is obtained, we may use it to classify a new object x according to the rule

<!-- formula-not-decoded -->

which approximates the MAP rule (3.34); cf. Fig. 3.4.1.

Returning to approximation in policy space, for a given training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ the classifier just described provides (approximations to) the 'probabilities' of using the controls u k ∈ U k ( x k ) at the states x k , so it yields a 'randomized' policy ˜ h ( u↪ x k ↪ r k ) for stage k [once the values ˜ h ( u↪ x k ↪ r k ) are normalized so that, for any given x k , they add to 1]; cf. Fig. 3.4.2. In practice, this policy is usually approximated by the deterministic policy ˜ θ k ( x k ↪ r k ) that uses at state x k the control of maximal probability at that state; cf. Eq. (3.38).

For the simpler case of a classification problem with just two categories, say A and B , a similar formulation is to hypothesize a relation of the following form between object x and its category:

<!-- formula-not-decoded -->

k)

Data-Trained

Maxu

Classifier

Idealized

Maxc

Maxc

Data-Trained

MAP Classifier

(e.g., a NN)

Classifier

2 Illustration of classifica

τ

τ

<!-- image -->

Next Partial Tours, MAP Classifier Data-Trained Max MAX max

Figure 3.4.1 Illustration of the MAP classifier c ∗ ( x ) for the case where the probabilities p ( c ♣ x ) are known [cf. Eq. (3.34)], and its data-trained version ˜ c ( x↪ ¯ r ) [cf. Eq. (3.38)]. The classifier may be obtained by using the data set ( x s k ↪ u s k ) ↪ s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q↪ and an approximation architecture such as a feature-based architecture or a neural network.

Next Partial Tours, MAP Classifier Data-Trained Max MAX max

Figure 3.4.2 Illustration of classification-based approximation in policy space. The classifier, defined by the parameter r k , is constructed by using the training set ( x s k ↪ u s k ), s = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ q . It yields a randomized policy that consists of the probability ˜ h ( u↪ x k ↪ r k ) of using control u ∈ U k ( x k ) at state x k . This policy is approximated by the deterministic policy ˜ θ k ( x k ↪ r k ) that uses at state x k the control that maximizes over u ∈ U k ( x k ) the probability ˜ h ( u↪ x k ↪ r k ) [cf. Eq. (3.38)].

<!-- image -->

where ˜ h is a given function and r is the unknown parameter vector. Given a set of q object-category pairs ( x 1 ↪ z 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( x q ↪ z q ) where

<!-- formula-not-decoded -->

we obtain r by the least squares regression:

<!-- formula-not-decoded -->

) System PID Controller

The optimal parameter vector ¯ r is used to classify a new object with data vector x according to the rule

<!-- formula-not-decoded -->

In the context of DP and approximation in policy space, this classifier may be used, among others, in stopping problems where there are just two controls available at each state: stopping (i.e., moving to a termination state) and continuing (i.e., moving to some nontermination state).

There are several variations of the preceding classification schemes, for which we refer to the specialized literature. Moreover, there are several commercially and publicly available software packages for solving the associated regression problems and their variants. They can be brought to bear on the problem of parametric approximation in policy space using any training set of state-control pairs, regardless of how it was obtained.

## 3.4.2 Policy Iteration with Value and Policy Networks

We noted earlier that contrary to rollout, approximate policy iteration (PI) is fundamentally an o ff -line training algorithm, because for a large scale problem, it is necessary to represent the cost functions or Q-factors of the successively generated policies with an approximation architecture. Thus, in a typical implementation, approximate PI involves the successive use of value networks to represent the cost functions of the generated policies, and one-step or multistep lookahead minimization to implement policy improvement.

On the other hand, it is also possible to use policy networks to approximate the results of policy improvement. In particular, we can start with a base policy and a terminal cost approximation, and generate state-control samples of the corresponding truncated rollout policy. These samples can be used with an approximation in policy space scheme to train a policy network that approximates the truncated rollout policy.

Then the cost function of the policy network can be approximated with a value network using the cost approximation methodology that we have discussed in this chapter. This value network can be used in turn as a terminal cost approximation in a truncated rollout algorithm where the previously obtained policy network can be used as a base policy. A new policy network can then be trained using samples of this rollout policy, etc. Thus a perpetual rollout scheme is obtained, which involves a sequence of value and policy networks.

One may also consider approximate PI algorithms that do not use a value network at all. Indeed the value network is only used to provide the approximate cost function values of the current policy, which are needed to calculate samples of the improved policy and train the corresponding

State Space Partition

Initial State Truncated Rollout Using Local Policy Network

Terminal Cost Supplied bu Local Value Network

Terminal Cost Supplied by Local Value Network

Figure 3.4.3 Illustration of a truncated rollout scheme with a partitioned architecture. A local value network is used for terminal cost function approximation for each subset of the partition.

<!-- image -->

policy network. On the other hand the samples of the improved policy can also be computed by rollout, using simulation-generated cost function values of the current policy. If the rollout can be suitably implemented with simulation, the training of a value network may be unnecessary.

## Multiprocessor Parallelization

We have noted earlier that parallelization and distributed computation can be used in several di ff erent ways in rollout and PI schemes, including Qfactor, Monte Carlo, and multiagent parallelization. It is also possible to consider the use of multiple neural networks in the implementation of rollout or approximate PI. For example, when feature-based partitioning of the state space is used (cf. Example 3.1.8), we may consider a multiprocessor parallelization scheme, which involves multiple local value and/or policy networks, which operate locally within a subset of the state space partition; see Figs. 3.4.3 and 3.4.4.

Let us finally note that multiprocessor parallelization leads to the idea of an approximation architecture that involves a graph. Each node of the graph consists of a neural network and each arc connecting a pair of nodes corresponds to data transfer between the corresponding neural networks. The question of how to train such an architecture is quite complex and one may think of several alternative possibilities. For example the training may be collaborative with the exchange of training results and/or training data communicated periodically or asynchronously; see the book [Ber20a], Section 5.8.

and a Local Policy Network

Each Set Has a Local Value Network and a Local Policy Network

Each Set Has a Local Value Network and a Local Policy Network

State Space Partition

<!-- image -->

Initial State Truncated Rollout Using Local Policy Network

Terminal Cost Supplied bu Local Value Network

Terminal Cost Supplied by Local Value Network

Figure 3.4.4 Illustration of a perpetual truncated rollout scheme with a partitioned architecture. A local value network and a local policy network are used for each subset of the partition. The policy network is used as the base policy and the value network is used to provide a terminal cost function approximation.

State-control training pairs for the corresponding rollout policy are obtained by starting at an initial state within some subset of the partition, generating rollout trajectories using the local policy network, which are truncated once the state enters a di ff erent subset of the partition, with the corresponding terminal cost function approximation supplied by the value network of that subset.

When a separate processor is used for each subset of partition, the corresponding value networks are communicated between processors. This can be done asynchronously, with each processor sharing its value network as it becomes available. In a variation of this scheme, the local policy networks may also be shared selectively among processors for selective use in the truncated rollout process.

## 3.4.3 Why Use On-Line Play and not Just Train a Policy Network to Emulate the Lookahead Minimization?

This is a sensible and common question, which stems from the mindset that neural networks have extraordinary function approximation properties. In other words, why go through the arduous and time-consuming process of on-line lookahead minimization, if we can do the same thing o ff -line and represent the lookahead policy with a trained policy network? In particular, we can select the policy from a suitably restricted class of policies, such as a parametric class of the form θ ( x↪ r ) ↪ where r is a parameter vector. We may then estimate r using some type of o ff -line training. Then the on-line computation of controls θ ( x↪ r ) can be much faster compared with on-line lookahead minimization.

On the negative side, because parametrized approximations often involve substantial calculations, they are not well suited for on-line replan-

Cost function difference

12

10

8

6

4

2

-0.8

-0.6

-0.4

-0.2

Linear policy parameter

Without the Newton Step

Figure 3.4.5 Illustration of the performance enhancement obtained when the Newton step/rollout is used in conjunction with an o ff -line trained (suboptimal) base policy for the linear quadratic problem; cf. Example 3.4.1. The figure shows the quadratic cost coe ffi cient di ff erences K L -K ∗ and K ˜ L -K ∗ as a function of L , where K L is the quadratic cost coe ffi cient of θ (without one-step lookahead/Newton step) and K L is the quadratic cost coe ffi cient of ˜ θ (with one-step lookahead/Newton step). The optimal performance is obtained for L ∗ ≈ -0 glyph[triangleright] 4.

<!-- image -->

ning. In other words, when the problem parameters change, as in adaptive control, there may not be enough time to retrain the parametric policy to deal with the changes . From our point of view in this book, there is also another important reason why approximation in value space is needed on top of approximation in policy space: the o ff -line trained policy may not perform nearly as well as the corresponding one-step or multistep lookahead/rollout policy , because it lacks the extra power of the associated exact Newton step (cf. our discussion of AlphaZero and TD-Gammon in Section 1.1, and linear quadratic problems in Section 1.5).

## Example 3.4.1 (A Linear-Quadratic Example)

Let us illustrate the benefit of the Newton step associated with one-step lookahead in an adaptive control setting. We consider a one-dimensional linear-quadratic problem, and we compare the performance of a linear policy with its corresponding one-step lookahead/rollout policy; cf. Fig. 3.4.5. In this example the system equation has the form

<!-- formula-not-decoded -->

and the quadratic cost function parameters are q = 1, r = 0 glyph[triangleright] 5. The optimal policy can be calculated to be

<!-- formula-not-decoded -->

with L ∗ ≈ -0 glyph[triangleright] 4, and the optimal cost function is

<!-- formula-not-decoded -->

where K ∗ ≈ 1 glyph[triangleright] 1. We want to to explore what happens when we use a policy of the form

<!-- formula-not-decoded -->

where L = L ∗ (e.g., a policy that is optimal for another system equation or cost function parameters). The cost function of θ L has the form

<!-- formula-not-decoded -->

where K L is obtained by using the formulas given in Section 1.5. When J θ is used as cost function approximation in a one-step lookahead/Newton step scheme, the cost function obtained is K ˜ L x 2 , where ˜ θ ( x ) = ˜ Lx is the corresponding one-step lookahead policy. As the figure shows, when the change L -L ∗ is substantial, the performance degradation K ˜ L -K ∗ is negligible, while the performance degradation K L -K ∗ is substantial.

## 3.5 POLICY GRADIENT AND RELATED METHODS

In this section we focus on infinite horizon problems and we discuss an alternative training approach for approximation in policy space, which is based on controller parameter optimization: we parametrize the policies by a vector r , and we optimize the corresponding expected cost over r . Thus, in contrast with the preceding section, we directly aim to approximate an optimal policy, rather than approximate a fixed policy.

For the most part in this section, we will assume no constraints on the parameter r , although we will occasionally comment regarding the presence of constraints. In particular, we determine r through the minimization

<!-- formula-not-decoded -->

where J ˜ θ ( r ) ( i 0 ) is the cost of the policy ˜ θ ( r ) starting from the initial state i 0 , and the expected value above is taken with respect to a suitable probability distribution of the initial state i 0 (cf. Fig. 3.5.1). This is to be contrasted with the classification approach of the preceding section, whereby we aim to learn a fixed policy, possibly as part of a policy iteration process. Here, we instead aim directly for an (approximately) optimal policy.

Note that in the case where the initial state i 0 is known and fixed, the method involves just minimization of J ˜ θ ( r ) ( i 0 ) over r . This simplifies a great deal the minimization, particularly when the problem is deterministic.

Similar to the preceding section, we focus on o ff -line training of policies . The on-line training of a policy involves a controller that interacts with the environment, applies controls, observes in real-time the e ff ects of these controls, and updates the policy parameters based on the observations. This challenging context is of considerable interest in practice, particularly in robotics, but is beyond our scope in this book.

/negationslash

Uncertainty System Environment Cost Control Current State certainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

Uncertainty System Environment Cost Control Current State

<!-- image -->

Uncertainty System Environment Cost Control Current State

Figure 3.5.1 Illustration of the policy optimization framework for an infinite horizon problem. Policies are parametrized with a parameter vector r and denoted by ˜ θ ( r ), with components ˜ θ ( i↪ r ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . Each parameter value r determines a policy ˜ θ ( r ), and a cost J ˜ θ ( r ) ( i 0 ) for each initial state i 0 , as indicated in the figure. The optimization determines r through the minimization

<!-- formula-not-decoded -->

where the expected value above is taken with respect to a suitable probability distribution of i 0 .

Before delving into the details, it is worth reminding the reader that using an o ff -line trained policy without combining it with approximation in value space has the fundamental shortcoming, which we noted frequently in this book: the o ff -line trained policy will not perform nearly as well as a scheme that uses this policy in conjunction with lookahead minimization , because it lacks the extra power of the associated exact Newton step.

## 3.5.1 Gradient Methods for Parametric Cost Optimization

Let us first consider methods that perform the minimization (3.39) by using a gradient method, and for simplicity let us assume that the initial condition i 0 is known [otherwise an expected value over i 0 must be introduced in the cost function; cf. Eq. (3.39)]. Thus the aim is to minimize J ˜ θ ( r ) ( i 0 ) over r by using the gradient method

<!-- formula-not-decoded -->

assuming that J ˜ θ ( r ) ( i 0 ) is di ff erentiable with respect to r . Here γ k is a positive stepsize parameter, and ∇ ( · ) denotes gradient with respect to r evaluated at the current iterate r k .

An important concern in this method is that the gradients ∇ J ˜ θ ( r k ) ( i 0 ) may not be explicitly available. In this case, the gradients can be approximated by finite di ff erences of cost function values J ˜ θ ( r k ) ( i 0 ). Unfortunately,

when the problem is stochastic, the cost function values may be computable only through Monte Carlo simulation. This may introduce a large amount of noise, so it is likely that we will need to average many samples in order to obtain su ffi ciently accurate gradients, thereby making the method ine ffi -cient. On the other hand, when the problem is deterministic, this di ffi culty does not arise, and the use of the gradient method (3.40) or other methods that do not rely on the use of gradients (such as coordinate descent) is facilitated.

In what follows in this section we will focus on alternative and typically more e ffi cient gradient-like methods for stochastic problems, which are based on sampling, rather than the exact calculation of the gradient, as in Eq. (3.40). Before doing so, we address briefly the issue of nondi ff erentiabilities and constraints in gradient-based optimization

## Constraints and Nondi ff erentiablities - Soft-Max Approximation

The gradient method (3.40) assumes that the cost function J ˜ θ ( r ) is di ff erentiable with respect to r . Frequently, however, this is not so, and to deal with nondi ff erentiabilities, a common practice has been to use the so-called soft-max approximation to smooth the di ff erentiabilities. As an example of this approach, nondi ff erentiable terms of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c is a positive parameter, and λ i are scalars satisfying

<!-- formula-not-decoded -->

The parameter c controls the quality of the approximation (as c increases the approximation becomes better, but also the smoothed problem becomes more ill-conditioned and harder to solve by gradient methods). The parameters λ i can be interpreted as Lagrange multipliers, which when chosen properly, improve the quality of the approximation of the smoothed function in the neighborhood of an optimal solution; see the book [Ber82a] for a detailed discussion.

The soft-max approximation is discussed within the context of a broad class of smoothing schemes in the author's book [Ber82a], Chapters 3 and 5. It was

can be approximated by

Another di ffi culty associated with the gradient method (3.40) is that there may be constraints on the parameter vector r . Gradient methods can be modified to deal with constraints, particularly in simple cases, such as nonnegativity or simplex constraints; see the discussion of Section 3.5.3 on gradient projection and proximal algorithms. However, constraints can also be dealt with by reparametrization of the policies in a way that eliminates the constraints. An important special case arises when the policies of a finite state and control spaces problem are replaced by randomized policies, which are then reparametrized to eliminate the associated probability distribution constraints by using the soft-max operation.

In particular, suppose that the set of policies is enlarged to include all randomized policies. Then, for a problem with states i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , controls u = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and constraints u ∈ U ( i ), an ordinary policy θ , which applies control θ ( i ) ∈ U ( i ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , is replaced by a randomized policy that applies at state i the control u i with probability r u ( i ). This randomized policy consists of a probability distribution { r u ( i ) ♣ u ∈ U ( i ) } for each state i , where the probabilities/parameters r u ( i ) must satisfy

<!-- formula-not-decoded -->

The policy optimization aims to find an optimal probability distribution for each state i .

The important point here is that the set of randomized policies is continuous and thus far better suited for the application of the gradient optimization (3.40). On the other hand the parameters r u ( i ) must satisfy the simplex constraints (3.41), thus requiring a gradient method that can deal with constraints. The corresponding policy optimization problem can be solved by a gradient-like method that can deal with the probability constraints (3.41). Such methods will be briefly discussed in the next section. However, it is also possible to reparametrize the space of probability distributions with a transformation that approximates probabilities as follows:

<!-- formula-not-decoded -->

first proposed in the author's papers [Ber75], [Ber77]; see also the paper by Poljak [Pol79], and the paper by Nesterov [Nes05], who used the smoothing approach for the complexity analysis of gradient methods. The soft-max approximation also bears a close connection to the exponential method of multipliers, first proposed by Kort and Bertsekas [KoB72]; see the books [Ber82a], Chapter 5, and [Ber17], Chapter 6, which also discuss a duality connection with the entropy minimization algorithm, a proximal minimization algorithm that is based on an exponential penalty function.

where θ u ( i ) are the transformed parameters and φ ( i ) are some scalars that depend on i . In this way the policy optimization problem becomes unconstrained. We note, however, that there are some serious di ffi culties associated with this reparametrization. In particular, policy controls r u ( i ) that are applied with probability 1, are mapped to parameters θ u ( i ) that are equal to ∞ in the reparametrized problem; see Exercise 3.2(b). Still the reparametrization r u ( i ) ↦→ θ u ( i ) of Eq. (3.42) is used widely in practice.

## 3.5.2 Policy Gradient-Like Methods

To get a sense of the general principle underlying the incremental gradient approach that uses randomization and sampling, let us digress from the DP context of this chapter, and consider the generic optimization problem

<!-- formula-not-decoded -->

where Z is some constraint set, and F is some real-valued function. Note that F may be nondi ff erentiable, and indeed the constraint set Z may be discrete (e.g., Z may be the finite set of stationary policies in a finitespaces Markovian decision problem; see Exercise 3.2). Hence gradient methods would seem like an unlikely candidate for solution. However, we will transform the problem to one that is amenable to the use of the gradient methodology through an unusual procedure that we describe next.

We will convert the problem (3.43) to the stochastic problem

<!-- formula-not-decoded -->

where z is viewed as a random variable, P Z is the set of probability distributions over Z , p denotes the generic distribution in P Z , and E p ¶·♦ denotes expected value with respect to p . Of course this enlarges the search space from Z to P Z , but it enhances the use of randomization schemes, simulation-based methods, even if the original problem is deterministic. Moreover, the cost function of the stochastic problem (3.44) may have some nice di ff erentiability properties that are lacking in the original deterministic version (3.43). This is true in general, although under special assumptions on F , such as convexity, these di ff erentiability properties are enhanced (see the paper [Ber73]). The reader may be able to appreciate the striking di ff erences between the deterministic problem (3.43) and the stochastic problem (3.44) by working out simple examples; see Exercise 3.2.

At this point it is not clear how the stochastic optimization problem (3.44) relates to our stochastic DP context of this chapter. We will return to this question later, but for the purpose of orientation, we note that to obtain a problem of the form (3.44), we must enlarge the set of policies to include randomized policies , mapping a state i into a probability distribution over the set of controls U ( i ).

Suppose now that we restrict attention to a subset ˜ P Z ⊂ P Z of probability distributions p ( z ; r ) that are parametrized by some continuous parameter r , e.g., a vector in some Euclidean space. In other words, we approximate the stochastic optimization problem (3.44) with the restricted problem

<!-- formula-not-decoded -->

Then we may use a gradient method for solving this problem, such as

<!-- formula-not-decoded -->

where ∇ ( · ) denotes gradient with respect to r of the function in parentheses, evaluated at the current iterate r k .

## Likelihood-Ratio Policy Gradient Methods

We will first consider an incremental version of the gradient method (3.46). This method requires that p ( z ; r ) is di ff erentiable with respect to r . It relies on a convenient gradient formula, sometimes referred to as the log-likelihood trick , which involves the natural logarithm of the sampling distribution.

This formula is obtained by the following calculation, which is based on interchanging gradient and expected value, and using the gradient formula ∇ (log p ) = ∇ pglyph[triangleleft]p . We have

<!-- formula-not-decoded -->

and finally

<!-- formula-not-decoded -->

The parametrization p ( z ; r ) must be di ff erentiable with respect to r in order for the gradient to exist. One possibility, noted earlier, is to use a soft-max parametrization such as p ( z ; r ) = e r ′ φ ( z ) ∑ y ∈ Z e r ′ φ ( y ) for all z ∈ Z , where φ ( z ) denotes a vector that suitably depends on z . This may involve some pitfalls, which are

illustrated in Exercise 3.2(b).

where for any given z , ∇ ( log ( p ( z ; r ) ) ) is the gradient with respect to r of the function log p ( z ; · ) , evaluated at r (the gradient is assumed to exist).

( ) The preceding formula suggests an incremental implementation of the gradient iteration (3.46) that approximates the expected value in the right side in Eq. (3.47) with a single sample (cf. Section 3.1.3). The typical iteration of this method is as follows.

## Sample-Based Gradient Method for Parametric Approximation of min z ∈ Z F ( z )

Let r k be the current parameter vector.

- (a) Obtain a sample z k according to the distribution p ( z ; r k ).
- (b) Compute the sampled gradient ∇ ( log ( p ( z k ; r k ) ) ) glyph[triangleright]
- (c) Iterate according to

<!-- formula-not-decoded -->

The advantage of the preceding sample-based method is its simplicity and generality. It allows the use of parametric approximation for any minimization problem (well beyond DP), as long as the logarithm of the sampling distribution p ( z ; r ) can be conveniently di ff erentiated with respect to r , and samples of z can be obtained using the distribution p ( z ; r ).

Note that in iteration (3.48) r is adjusted along a random direction. This direction does not involve at all the gradient of F (even if it exists), only the gradient of the logarithm of the sampling distribution! As a result the iteration has a model-free character : we don't need to know the mathematical form of the function F as long as we have a simulator that produces the cost function value F ( z ) for any given z . This is also a major advantage o ff ered by many random search methods.

An important issue is the e ffi cient computation of the sampled gradient ∇ ( log ( p ( z k ; r k ) )) glyph[triangleright] In the context of DP, including the SSP and discounted problems that we have been dealing with, there are some specialized procedures and corresponding parametrizations to approximate this gradient conveniently. The following is an example.

## Example 3.5.1 (Policy Gradient Method for Discounted DP)

Consider the α -discounted problem and denote by z the infinite horizon statecontrol trajectory:

<!-- formula-not-decoded -->

We consider a parametrization of randomized policies with parameter r , so the control at state i is generated according to a distribution p ( u ♣ i ; r ) over U ( i ). Then for a given r , the state-control trajectory z is a random vector with probability distribution denoted p ( z ; r ). The cost corresponding to the trajectory z is

<!-- formula-not-decoded -->

and the problem is to minimize over r

<!-- formula-not-decoded -->

To apply the sample-based gradient method (3.48), given the current iterate r k , we must generate the sample state-control trajectory

<!-- formula-not-decoded -->

according to the distribution p ( z ; r k ), we compute the corresponding cost F ( z k ), and also calculate the sampled gradient

<!-- formula-not-decoded -->

Let us assume that the logarithm of the randomized policy distribution p ( u ♣ i ; r ) is di ff erentiable with respect to r (a soft-min policy parametrization is often recommended for this purpose). Then the logarithm that is di ff erentiated in Eq. (3.50) can be written as

<!-- formula-not-decoded -->

and its gradient (3.50), which is needed in the iteration (3.48), is given by

<!-- formula-not-decoded -->

This gradient involves the current randomized policy, but does not involve the system's transition probabilities and the costs per stage.

The policy gradient method (3.48) can now be implemented with a finite horizon approximation whereby r k is changed after a finite number N of time steps [so the infinite cost and gradient sums (3.49) and (3.51) are replaced by finite sums]. The method takes the form

<!-- formula-not-decoded -->

where z k N = ( i k 0 ↪ u k 0 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ i k N -1 ↪ u k N -1 ) is the generated N -step trajectory, F N ( z k N ) is the corresponding cost, and γ k is the stepsize. The initial state i k 0 of the trajectory is chosen randomly, with due regard to exploration issues.

Policy gradient methods for other types of DP problems can be similarly developed, as well as variations involving a combination of policy and cost function parametrizations [e.g., replacing F ( z ) of Eq. (3.49) by a parametrized estimate that has smaller variance, and possibly subtracting a suitable baseline, cf. the following discussion]. This leads to a class of actor-critic methods that di ff er from the PI-type methods that we discussed earlier in this chapter; we refer to the end-of-chapter literature for a variety of specific schemes. The policy gradient methods that we discussed in this section are sometimes called actor-only methods, as they involve only policy parametrizations.

## Implementation Issues

There are several additional issues to consider in the implementation of the sample-based gradient method (3.48). The first of these is that the problem solved is a randomized version of the original. If the method produces a parameter r in the limit and the distribution p ( z ; r ) is not atomic (i.e., it is not concentrated at a single point), then a solution z ∈ Z must be extracted from p ( z ; r ). In the SSP and discounted problems that we have been dealing with, the subset ˜ P Z of parametric distributions typically contains the atomic distributions, while it can be shown that minimization over the set of all distributions P Z produces the same optimal value as minimization over Z (the use of randomized policies does not improve the optimal cost of the problem), so this di ffi culty does not arise.

Another issue is how to improve sampling e ffi ciency. To this end, let us note a simple generalization of the gradient method (3.48), which can often improve its performance. It is based on the gradient formula

<!-- formula-not-decoded -->

where b is any scalar. This formula generalizes Eq. (3.47), where b = 0, and holds in view of the following calculation, which shows that the term multiplying b in Eq. (3.52) is equal to 0:

<!-- formula-not-decoded -->

where the last equality holds because ∑ z ∈ Z p ( z ; r ) is identically equal to 1 and hence does not depend on r .

Based on the gradient formula (3.52), we can modify the sampled gradient iteration (3.48) to read as follows:

<!-- formula-not-decoded -->

where b is some fixed scalar, called the baseline . Whereas the choice of b does not a ff ect the gradient ∇ ( E p ( z ; r ) { F ( z ) } ) [cf. Eq. (3.52)], it a ff ects the incremental gradient

<!-- formula-not-decoded -->

which is used in the iteration (3.53). Thus, by optimizing the baseline b , empirically or through a calculation (see e.g., [DNP11]), we can improve the performance of the algorithm. Moreover, in the context of discounted and SSP problems, state-dependent baseline functions have been used. Ideas of cost shaping are useful within this context; we refer to the RL textbook [Ber19a] and specialized literature for further discussion.

## 3.5.3 Scaling and Proximal Policy Optimization Methods

Gradient-like methods are notorious for their slow convergence, which is exacerbated by the detrimental e ff ects of noise (when applied in a stochastic environment). A principal technique to deal with this di ffi culty is to scale the gradient by multiplication with a positive definite matrix.

In particular, scaled gradient methods for solving general nonlinear unconstrained optimization problems of the form

<!-- formula-not-decoded -->

where f is twice continuously di ff erentiable objective function, have the form

<!-- formula-not-decoded -->

where D k is a positive definite symmetric m × m matrix (see nonlinear programming books, such as [Ber16]). Two extreme cases are:

- (a) Select D k to be the Hessian matrix of f evaluated at x k , in which case we obtain Newton's method . This method converges superlinearly when started su ffi ciently close to a strong local minimum (one where the Hessian is positive definite). An approximation to Newton's method for least squares problems where f ( x ) = ∑ N i =1 ‖ g i ( x ) ‖ 2 , is the Gauss-Newton method , which neglects the second derivatives of g i (see e.g., [Ber16] for discussions of the Gauss-Newton method).

- (b) Select D k to be a diagonal approximation to the Hessian matrix of f evaluated at x k . This method converges to a strong local minimum under appropriate conditions on D k (it should not be too 'small'). Its convergence rate is linear, and typically much slower than the superlinear convergence rate of Newton's method. On the other hand, a gradient method with diagonal scaling is much simpler than Newton's method, and more readily admits an incremental implementation of the type that we have discussed. Moreover, incremental versions of diagonally scaled gradient methods have worked well for the training of neural networks and other approximation architectures, particularly when supplemented with e ff ective heuristics; cf. the discussion of Section 3.1.3 in connection with the training of neural networks and the ADAM algorithm.

The scaled gradient iteration (3.54) can also be written in an alternative and equivalent form in terms of a quadratic optimization:

<!-- formula-not-decoded -->

This is easily verified by setting to 0 the derivative of the quadratic cost above, thereby obtaining

<!-- formula-not-decoded -->

which is equivalent to Eq. (3.55).

The form (3.55) of the scaled gradient iteration is in turn related to two other important optimization methods (see nonlinear programming books such as [Ber16]). The first of these is the gradient projection method , which applies to constrained optimization problems of the form

<!-- formula-not-decoded -->

where C is some closed convex set. It takes the form

<!-- formula-not-decoded -->

This iteration can be equivalently written as

<!-- formula-not-decoded -->

where P D k ( · ) denotes projection onto C with respect to a norm that is defined by the scaling matrix D k (hence the name 'gradient projection'). We can see this by writing

<!-- formula-not-decoded -->

and expanding the quadratic form on the right-hand side, to verify that the iterations of Eqs. (3.57) and (3.58) are equivalent. When the problem is unconstrained, i.e., C = /Rfractur m , the gradient projection iteration (3.57) is equivalent to the scaled gradient iterations (3.54) and (3.55).

Let us also mention the proximal minimization algorithm , which applies to the constrained optimization problem (3.56), and has the form

<!-- formula-not-decoded -->

It can be seen that the gradient projection method (3.57) and the proximal algorithm (3.59) have strong similarities, and that they coincide when f is linear. Note also that the two algorithms can be suitably combined in ways that fit the structure of specific problems, and admit incremental and randomized/simulation-based implementations (see the author's survey paper [Ber10d]).

A second way to describe the gradient iteration (3.55) involves the 'trust region' optimization

<!-- formula-not-decoded -->

where β k is a positive scalar. It can be shown that the methods (3.55) and (3.60) are equivalent for an appropriate value of β k (see[Ber16]). The role of β k is to regulate the size of the constraint and ensure that the increment r k +1 -r k is appropriately small. This guards against oscillations of the iterates r k , which may lead to divergence.

## Natural Gradient Methods

While the scaling techniques discussed above have worked well for the general nonlinear optimization problem min r ∈/Rfractur m f ( r ), alternative scaling techniques have been suggested for stochastic gradient methods that involve optimization over parametrized probability distributions. In place of an inverse Hessian matrix, these methods use the Fisher Information Matrix (FIM), which accounts for the curvature of the cost function induced by the parametrization.

For the case of the parametric optimization problem

<!-- formula-not-decoded -->

that we have discussed earlier in this section [cf. Eq. (3.45)], the FIM is given by

<!-- formula-not-decoded -->

where the gradient ∇ ( log ( p ( z ; r ) ) (a column vector) is computed with the formulas given earlier (cf. Example 3.5.1, for the case of discounted DP problems). The scaled version of the corresponding gradient method of Eq. (3.46) is given by

<!-- formula-not-decoded -->

with the gradient in the preceding iteration given by the earlier derived formula (3.47):

<!-- formula-not-decoded -->

The scaling induced by the FIM (3.62) in the scaled gradient method (3.63) is conceptually related to the Newton and the Gauss-Newton scaling noted earlier in connection to the scaled gradient method (3.54); see the relevant literature for more details, including the scholarly and comprehensive book by Amari [Ama16]. The scaled gradient

<!-- formula-not-decoded -->

is also known as the natural gradient for the problem (3.61). Incremental/sampled versions of the scaled gradient method (3.63) are possible along the lines discussed earlier; cf. Eq. (3.48). In particular, for a discounted DP problem of Example 3.5.1, we can use the FIM formula (3.62) (or an approximation thereof), where

<!-- formula-not-decoded -->

cf. Eq. (3.51), and z k is the infinite horizon k th sample state-control trajectory

<!-- formula-not-decoded -->

The motivation for using the FIM in parametric optimization problems of the form (3.61), is that it defines an alternative norm in the space of probability distributions, which captures the sensitivity of the distribution p ( · ; r ) with respect to the parameter r more e ff ectively than the norm defined the inverse Hessian approximations D k discussed earlier. In this regard, it should be noted that the FIM is the Hessian matrix of the Kullback-Leibler (KL) divergence, an important metric for measuring distance between two probability distributions. We refer to the papers by Amari [Ama98] and Kakade [Kak02] for further discussion. For more recent works that provide many additional references, see Deisenroth, Neumann,

and Peters [DNP11], Grondman et al. [GBL12], Fazel et al. [FGK18], Cen et al. [CCC21], Bhandari and Russo [BhR24], and Muller and Montufar [MuM24].

Finally let us note that the trust region analog of the scaled iteration (3.63) takes the form

<!-- formula-not-decoded -->

where β k is a positive scalar, in analogy with Eq. (3.60). Implementations of the policy gradient method with FIM scaling and with forms of trust region are provided by the proximal policy optimization (PPO) method, suggested by Shulman et al. [SWD17], which has been used widely, including for the training of large language models (see Section 3.5.3).

## Constrained Natural Gradient Methods

Natural gradient methods can be extended to constrained optimization problems of the form

<!-- formula-not-decoded -->

where C ⊂ /Rfractur m is a closed convex constraint set, in analogy with problem (3.56). Important special cases of constrained problems arise often in practice, such as when C consists of nonnegativity constraints:

<!-- formula-not-decoded -->

or has some other simple form, such as upper and lower bounds on the components of r , a simplex constraint, or a Cartesian product of simplexes. In what follows, we will provide e ffi cient variants of natural gradient methods for simple constraint sets, focusing primarily on the orthant constraint (3.66).

To simplify notation, let us denote by r 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r m the m components of r , and let us use an abbreviated notation for the gradient and partial derivatives of the cost function:

<!-- formula-not-decoded -->

With this notation, the unconstrained natural gradient method is given by

<!-- formula-not-decoded -->

and its projected analog [cf. Eq. (3.57)] is given by

<!-- formula-not-decoded -->

This is a valid method with strong convergence guarantees, which follow from well-established analysis of gradient projection methods for general constrained nonlinear programming problems (see e.g., [Ber16]). Unfortunately, however, the practical implementation of this method runs into some serious di ffi culties. Chief among these is that the quadratic minimization in Eq. (3.68) may be much harder than the matrix inversion of Eq. (3.67). This is true even if C is the simple orthant constraint (3.66).

To address this issue we may consider a method that involves a simpler projection. For the case of the orthant constraint (3.66), one possibility that comes to mind is

<!-- formula-not-decoded -->

where [ · ] + is the 'clipping' operation, whereby the i th component [ r ] + i of [ r ] + is obtained from the i th component r i of r according to

<!-- formula-not-decoded -->

and ˜ M ( r k ) is either the FIM M ( r k ) or a suitable approximation thereof.

Unfortunately when ˜ M ( r k ) is taken to be equal to the FIM M ( r k ), the algorithm (3.69) does not work. The reason is illustrated in Fig. 3.5.2, and has to do with the fact that when r k lies at the boundary of the orthant, the iteration may not lead to cost reduction regardless of how small the stepsize γ k is . In particular, it is possible that the unprojected iterate r k -γ k ˜ M ( r k ) -1 g ( r k ) improves the cost function for g k su ffi ciently small, but after projection onto the orthant it does not [see Fig. 3.5.2(a)]. In fact it is possible that r k is optimal, but the iteration (3.69) moves r k away from the optimal [see Fig. 3.5.2(b)].

It turns out, however, that there is a class of approximate FIM matrices ˜ M ( r k ) for which cost reduction can be guaranteed. This class is su ffi ciently wide to allow fast convergence when ˜ M ( r k ) properly embodies the essential content of the true FIM M ( r k ).

We say that a symmetric m × m matrix D with elements d ij is diagonal with respect to a subset of indices I ⊂ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ , if

/negationslash

<!-- formula-not-decoded -->

Let us denote for all r ≥ 0

<!-- formula-not-decoded -->

∣ see Fig. 3.5.2 for an example. The following is a key proposition, due to the author's works [Ber82b], [Ber82c], which shows how to select an approximation to the FIM in order to guarantee the validity of the projected iteration (3.69).

Figure 3.5.2. Two-dimensional examples where I + ( r k ) = ¶ 2 ♦ [we have r k 1 &gt; 0, while r k 2 = 0 and g 2 ( r k ) &gt; 0]. (a) A case where the iteration

<!-- image -->

<!-- formula-not-decoded -->

fails to make progress when ˜ M ( r k ) is not properly diagonalized. (b) A case where the iteration fails to stop at an optimal solution r ∗ when ˜ M ( r k ) is not properly diagonalized.

Proposition 3.5.1: Let r ≥ 0 and let ˜ M ( r ) be a positive definite symmetric matrix that is diagonal with respect to I + ( r ). Denote by r ( γ ) the iterate of the projected natural gradient iteration (3.69), starting from r , as a function of the stepsize γ :

<!-- formula-not-decoded -->

Consider a vector r that is nonstationary, in the sense that it violates the first order optimality condition

<!-- formula-not-decoded -->

Then there exists a scalar γ &gt; 0 such that the following cost improvement property holds:

<!-- formula-not-decoded -->

Based on Prop. 3.5.1 we conclude that to guarantee cost improvement, the matrix ˜ M ( r k ) in the iteration (3.69) should be chosen to be diagonal

with respect to a subset of indices that contains

<!-- formula-not-decoded -->

∣ and the stepsize g k should be chosen su ffi ciently small to guarantee cost improvement. Actually, it turns out that to ensure convergence one should implement the iteration more carefully. The reason is that the set I + ( r k ) exhibits an undesirable discontinuity at the boundary of the constraint set, whereby given a sequence ¶ r k ♦ of interior points that converges to a boundary point r the set I + ( r k ) may be strictly smaller than the set I + ( r ). This causes di ffi culties in proving convergence of the algorithm and may have an adverse e ff ect on its rate of convergence. To bypass these di ffi culties one may add to the set I + ( x k ) the indices of those variables x k i that satisfy g i ( r k ) &gt; 0 and are 'near' zero (i.e., 0 ≤ r k i ≤ /epsilon1 , where /epsilon1 is a small fixed scalar).

The theoretical properties of the projection method (3.69), with the preceding modification and a properly defined stepsize rule are solid. Moreover, extensive computational practice has established its superior performance over alternative methods of the gradient type. We refer to the author's constrained optimization book [Ber82b] (Section 1.5) and the paper [Ber82c], where the method (3.69) was first proposed under the name 'two-metric projection method;' see also Gafni and Bertsekas [GaB84]. These references, together with the book [Ber16] (Section 3.4), also discuss extensions to broader classes of problems involving linear constraints, such as upper and lower bounds on the parameters r i , and simplex constraints. For more recent works that are focused on machine learning applications, see Schmidt, Kim, and Sra [SKS12], Xie and Wright [XiW21], [XiW24], and Wu and Xie [WuX24].

An important property of the projected natural gradient method (3.69) is that it can preserve the beneficial scaling properties of the Fisher information matrix. The reason is that the method tends to quickly identify the subspace of active constraints (the ones for which r i = 0 at an optimal solution) and then reduces to the unconstrained natural gradient method on that subspace. For the same reason, the method is also well-suited for an incremental implementation.

## 3.5.4 Random Direction Methods

In this section, we will consider an alternative class of incremental versions of the policy gradient method (3.46), repeated here for convenience:

<!-- formula-not-decoded -->

These methods are based on the use of a random search direction and only two sample function values per iteration . They are generally faster than

z r r

Figure 3.5.3 The distribution p ( z ; r ) used in the gradient iteration (3.72).

<!-- image -->

methods that use a finite di ff erence approximation of the entire cost function gradient; see the book by Spall [Spa03] for a detailed discussion, and the paper by Nesterov and Spokoiny [NeS17] for a more theoretical view. Moreover, these methods do not require the derivative of the sampling distribution or its logarithm. On the other hand, these methods have not been applied widely to the training of policies in the RL field.

For simplicity we first consider the case where z and r are scalars, and later discuss the multidimensional case. In particular, we assume that p ( z ; r ) is symmetric and is concentrated with probabilities p i at the points r + /epsilon1 i and r -/epsilon1 i , where /epsilon1 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ /epsilon1 m are some small positive scalars. Thus we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The gradient iteration (3.71) becomes

<!-- formula-not-decoded -->

Let us now consider approximation of the gradient by finite di ff erences:

<!-- formula-not-decoded -->

We approximate the gradient iteration (3.72) by

<!-- formula-not-decoded -->

One possible sample-based/incremental version of this iteration is

<!-- formula-not-decoded -->

where i k is an index generated with probabilities that are proportional to p i k . This algorithm uses one out of the m terms of the gradient in Eq. (3.73).

The extension to the case, where z and r are multidimensional, is straightforward. Here p ( z ; r ) is a probability distribution, whereby z takes values of the form r + /epsilon1 d , where d is a random vector that lies on the surface of the unit sphere, and /epsilon1 (independently of d ) takes scalar values according to a distribution that is symmetric around 0. The idea is that at r k , we first choose randomly a direction d k on the surface of the unit sphere, and then change r k along d k or along -d k , depending on the sign of the corresponding directional derivative. For a finite di ff erence approximation of this iteration, we sample z k along the line ¶ r k + /epsilon1 d k ♣ /epsilon1 ∈ /Rfractur ♦ , and similar to the iteration (3.74), we set

<!-- formula-not-decoded -->

where /epsilon1 k is the sampled value of /epsilon1 .

Let us also discuss the case where p ( z ; r ) is a discrete but nonsymmetric distribution, i.e., z takes values of the form r + /epsilon1 d , where d is a random vector that lies on the surface of the unit sphere, and /epsilon1 is a zero mean scalar. Then the analog of iteration (3.75) is

<!-- formula-not-decoded -->

where /epsilon1 k is the sampled value of /epsilon1 . Thus in this case, we still require two function values per iteration. Generally, for a symmetric sampling distribution, iteration (3.75) tends to be more accurate than iteration (3.76), and is often preferred.

Algorithms of the form (3.75) and (3.76) are known as random direction methods . They use only two cost function values per iteration, and a direction d k that need not be related to the gradient of F in any way. There is some freedom in selecting d k , which could potentially be exploited in specific schemes. Selection of the stepsize γ k and the sampling distribution for /epsilon1 can be tricky, particularly when the values of F are noisy.

## 3.5.5 Random Search and Cross-Entropy Methods

The main drawback of the policy gradient methods that we have considered so far is the risk of unreliability due to the stochastic uncertainty corrupting the calculation of the gradients, the slow convergence that is typical of

Figure 3.5.4 Schematic illustration of the cross-entropy method. At the current iterate r k , we construct an ellipsoid E k centered at r k . We generate a number of random samples within E k , and we 'accept' a subset of the samples that have 'low' cost. We then choose r k +1 to be the sample mean of the accepted samples, and construct a sample 'covariance' matrix of the accepted samples. We then form the new ellipsoid E k +1 using this matrix and a suitably enlarged radius, and continue. Notice the resemblance with a policy gradient method: we move from r k to r k +1 in a direction of cost improvement.

<!-- image -->

gradient methods in many settings, and the presence of local minima. For this reason, alternative methods based on random search are potentially more reliable alternatives. Viewed from a high level, random search methods are similar to policy gradient methods in that they aim at iterative cost improvement through sampling. However, they need not involve randomized policies, they are not subject to cost di ff erentiability restrictions, and they o ff er some global convergence guarantees, so in principle they are not a ff ected much by local minima.

Let us consider a parametric policy optimization approach based on solving the problem

<!-- formula-not-decoded -->

cf. Eq. (3.39). Random search methods for this problem explore the space of the parameter vector r in some randomized but intelligent fashion. There are several types of such methods for general optimization, and some of them have been suggested for approximate DP. We will briefly describe the cross-entropy method , which has gained considerable attention.

The method, when adapted to the approximate DP context, bears resemblance to policy gradient methods, in that it generates a parameter sequence ¶ r k ♦ by changing r k to r k +1 along a direction of 'improvement.'

This direction is obtained by using the policy ˜ θ ( r k ) to generate randomly cost samples corresponding to a set of sample parameter values that are concentrated around r k . The current set of sample parameters are then screened: some are accepted and the rest are rejected, based on a cost improvement criterion. Then r k +1 is determined as a 'central point' or as the 'sample mean' in the set of accepted sample parameters, some more samples are generated randomly around r k +1 , and the process is repeated; see Fig. 3.5.4. Thus successive iterates r k are 'central points' of successively better groups of samples, so in some broad sense, the random sample generation process is guided by cost improvement. This is a general idea that is shared with other popular classes of random search methods.

The cross-entropy method is very simple to implement, does not suffer from the fragility of gradient-based optimization, does not involve randomized policies, and relies on some supportive theory. Importantly, the method does not require the calculation of gradients, and it does not require di ff erentiability of the cost function. Moreover, it does not need a model to compute the required costs of di ff erent policies; a simulator is su ffi cient.

Like all random search methods, the convergence rate guarantees of the cross-entropy method are limited, and its success depends on domainspecific insights and the skilled use of heuristics. However, the method relies on solid ideas and has gained a favorable reputation. In particular, it was used with impressive success in the context of the game of tetris; see Szita and Lorinz [SzL06], and Thiery and Scherrer [ThS09]. There have also been reports of domain-specific successes with related random search methods; see Salimans et al. [SHC17]. We refer to the end-of-chapter literature for details and examples of implementation.

## 3.5.6 Refining and Retraining Parametric Policies

Suppose that we already have a pre-trained parametric policy, such as one implemented through a neural network, and we want to refine it. A possible motivation for this may be to correct some deficiencies of the policy, such as a bias towards undesirable behaviors. For example the policy may produce poor or unsafe responses at some sensitive states. We note that refining a pre-trained parametric policy has become the subject of intensive interest following the availability of highly capable pre-trained transformer-based software such as ChatGPT and DeepSeek.

A common way to retrain a policy is to modify the cost function of the problem in a way that penalizes the undesirable behaviors. To this end we need a special purpose/domain-specific dataset consisting of input-output pairs that are generated using human or machine feedback. The dataset may be supplemented with specific instructions on how to choose controls in special or critical situations.

Once the dataset has been chosen, we need a method that uses this dataset to retrain the cost function. One possibility is to use a policy approximation method in combination with a parametric comparison training scheme, such as the one described in Section 2.3.6 in the context of learning how to imitate an expert. Another possibility is adversarial training , which involves exposing the policy to adversarial examples (involving states designed to trigger undesirable behaviors) and fine-tuning it to handle them correctly, using a policy gradient or random search method.

Another interesting retraining context, known as knowledge distillation , arises in situations where a policy must be simplified to be deployed in an environment where limited computational resources are available. This involves training a smaller, more computationally e ffi cient policy (the 'student') to mimic the behavior of a larger, more complex policy (the 'teacher'). One possibility is to use the teacher as an expert that generates a dataset for training the student. We refer to Hinton, Vinyals, and Dean [HVD15], and the survey by Xu et al. [XLT24] for further discussion.

## Mixture-of-Experts Techniques

A situation that often arises in practice is that the system may undergo significant structural changes following transition to some special states. For example significant parts of the system may be disabled or repurposed at random or scheduled times, in which case radical changes in the control policy may be needed. This situation comes under the general subject of adaptive control, but in cases where the structural changes are significant, it may be best handled by specialized mixture-of-expects techniques, which we will now explain briefly.

The main idea in a mixture-of-experts scheme is to have multiple policies (or experts) available, and to switch from one policy to another whenever the need arises. For example, when a structural change in the system occurs or when the environment goes through a significant change, simply adopt the policy that is best suited for the change that has occurred. The system change must be detected on line with a suitable algorithm, and then a gating mechanism must be used, which dynamically enables the most relevant policy. The di ff erent policies may be trained o ff line, possible by refining some pre-trained policy.

As an example, we note the technique of multiple model control design that was mentioned in Section 1.6.8 in the context of adaptive control. Another more recent example is the mixture-of-experts scheme that was implemented with success in DeepSeek; see Dai et al. [DDZ24].

## 3.6 AGGREGATION

In this section we consider approximation in value space using a problem approximation approach that is based on aggregation. More specifically,

we construct a simpler and more tractable 'aggregate' problem by creating special subsets of states, which we view as 'aggregate states.' We then solve the aggregate problem exactly by DP. This is the o ff -line training part of the aggregation approach, and it may be carried out with a variety of DP methods, including simulation-based value and policy iteration; we refer to the RL book [Ber19a] for a detailed account. Finally, we use the optimal cost-to-go function of the aggregate problem (or an approximation thereof) to construct a terminal cost approximation in a one-step or multistep lookahead approximation scheme for the original problem. Additionally, we may also use the optimal policy of the aggregate problem as a base policy for a truncated rollout scheme.

In addition to problem approximation, aggregation is related to feature-based parametric approximation. More specifically, it often produces a piecewise constant cost function approximation, which may be viewed as a linear feature-based parametrization, where the features are 0-1 membership functions; see Example 3.1.1. Aggregation can also be combined with other approximation schemes, which can produce a cost function approximation, possibly through the use of a neural network; see our subsequent discussion of biased aggregation.

Aggregation can be applied to both finite horizon and infinite horizon problems. In our discussion, we will focus primarily on the discounted infinite horizon problem with a finite number of states, although the ideas apply more broadly. In particular, we will focus on the standard discounted infinite horizon problem with the n states 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n . States and successor states will be denoted by i and j . State transitions ( i↪ j ) under control u occur at discrete times according to transition probabilities p ij ( u ), and generate a cost α k g ( i↪ u↪ j ) at time k , where α ∈ (0 ↪ 1) is the discount factor.

We consider deterministic stationary policies θ such that for each i , θ ( i ) is a control that belongs to a finite constraint set U ( i ). We denote by J θ ( i ) the total discounted expected cost of θ over an infinite number of stages starting from state i , by J * ( i ) the minimal value of J θ ( i ) over all θ , and by J θ and J * the n -dimensional vectors that have components J θ ( i ) and J * ( i ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , respectively.

We will introduce aggregation in a simple intuitive form in Section 3.6.1, and generalize later to a more sophisticated form of feature-based aggregation, which we also discussed briefly in Example 3.1.7. Our coverage of aggregation in this section is somewhat abbreviated, and we refer to the books [Ber12], Section 6.5, and [Ber19a], Chapter 6, for a more detailed presentation.

## 3.6.1 Aggregation with Representative States

We will first focus on a relatively simple form of aggregation, which involves a special subset of states, called representative . Our approach is to view these states as the states of a smaller optimal control problem, the aggre-

Figure 3.6.1 Illustration of aggregation with representative states; cf. Example 3.6.1. A relatively small number of states are viewed as representative. We define transition probabilities between pairs of aggregate states and we also define the associated expected transition costs. These specify a smaller DP problem, called the aggregate problem, which is solved exactly. The optimal cost function J ∗ of the original problem is approximated by interpolation from the optimal costs of the representative states r ∗ y in the aggregate problem:

<!-- image -->

<!-- formula-not-decoded -->

and is used in a one-step or multistep lookahead scheme.

gate problem, which we will formulate and solve exactly in place of the original. We will then use the optimal aggregate costs of the representative states to approximate the optimal costs of the original problem states by interpolation. Let us describe a classical example.

## Example 3.6.1 (Coarse Grid Approximation)

Consider a discounted problem where the state space is a grid of points i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n on the plane. We introduce a coarser grid that consists of a subset of the states/points, which we call representative and denote by x ; see Fig. 3.6.1. We now wish to formulate a lower-dimensional DP problem just on the coarse grid of states. The di ffi culty here is that there may be positive transition probabilities p xj ( u ) from some representative states x to some nonrepresentative states j . To deal with this di ffi culty, we introduce artificial transition probabilities φ jy from non-representative states j to representative states y , which we call aggregation probabilities . In particular, a transition from representative state x to a nonrepresentative state j , is followed by a transition from j to some other representative state y with probability φ jy ; see Fig. 3.6.2.

Aggregate States Cost

Representative States

States (Fine Grid)

Original State Space

Figure 3.6.2 Illustration of the use of aggregation probabilities φ jy from nonrepresentative states j to representative states y in Example 3.6.1. A transition from a state x to a nonrepresentative state j is followed by a transition to aggregate state y with probability φ jy . In this figure, from representative state x , there are three possible transitions, to states j 1 , j 2 , and j 3 , according to p xj 1 ( u ) ↪ p xj 2 ( u ) ↪ p xj 3 ( u ) ↪ and each of these states is associated with a convex combination of representative states using the aggregation probabilities. For example, the state j 1 is associated with the aggregation probabilities φ j 1 y 1 ↪ φ j 1 y 2 ↪ φ j 1 y 3 ↪ and the cost of j 1 is approximated by the corresponding convex combination of the costs of y 1 ↪ y 2 ↪ y 3 ; [cf. Eq. (3.78)].

<!-- image -->

This process involves approximation but constructs a transition mechanism for an aggregate problem whose states are just the representative ones. The transition probabilities between representative states x↪ y under control u ∈ U ( x ) and the corresponding expected transition costs can be computed as

<!-- formula-not-decoded -->

We can solve the aggregate problem by any suitable exact DP method. Let A denote the set of representative states and let r ∗ x denote the corresponding optimal cost of representative state x . We can then approximate the optimal cost function of the original problem with the interpolation

<!-- formula-not-decoded -->

This function may in turn be used in a one-step or multistep lookahead scheme for approximation in value space of the original problem.

/negationslash

Note that there is a lot of freedom in selecting the aggregation probabilities φ jy . Intuitively, φ jy should express a measure of proximity between j and y , e.g., φ jy should be relatively large when y is geometrically close to j . For example, we could set φ jy j = 1 for the representative state y j that is 'closest' to j , and φ jy j = 0 for all other representative states y = y j . In this case, Eq. (3.78) yields a piecewise constant cost function approximation ˜ J (the constant values are the scalars r ∗ y of the representative states y ).

Aggregation Probabilities

We will now formalize our framework for aggregation with representative states by generalizing the preceding example; see Fig. 3.6.3. We first consider the n -state version of the α -discounted problem of Section 1.4.1. We refer to this problem as the 'original problem,' to distinguish from the 'aggregate problem,' which we define next.

## Aggregation Framework with Representative States

We introduce a finite subset A of the original system states, which we call representative states , and we denote them by symbols such as x and y . We construct an aggregate problem , with state space A , and transition probabilities and transition costs defined as follows:

- (a) We relate the original system states j to representative states y ∈ A with aggregation probabilities φ jy ; these are scalar 'weights' satisfying for all j and y ∈ A ,

<!-- formula-not-decoded -->

- (b) We define the transition probabilities between representative states x and y under control u ∈ U ( x ) by

<!-- formula-not-decoded -->

- (c) We define the expected transition costs at representative states x under control u ∈ U ( x ) by

<!-- formula-not-decoded -->

The optimal costs of the representative states y ∈ A in the aggregate problem are denoted by r ∗ y , and they define approximate costs for the original problem through the interpolation formula

<!-- formula-not-decoded -->

Aside from the selection of representative states, an important consideration is the choice of the aggregation probabilities. These probabilities

Representative States

Aggregate Problem

n

j=1

n

§(x, u) = &gt; Pxj (2)g(x, U, j)

j=1

One-step Lookahead with

)

Range of Weighted Projections Original States

<!-- image -->

Aggregation Probabilities

Figure 3.6.3 Illustration of the aggregate problem in the representative states framework. The transition probabilities ˆ p xy ( u ) and transition costs ˆ g ( x↪ u ) are shown in the bottom part of the figure. Once the aggregate problem is solved (exactly) for its optimal costs r ∗ y , we define approximate costs

<!-- formula-not-decoded -->

which are used for one-step lookahead approximation of the original problem.

express 'similarity' or 'proximity' of original to representative states (as in the case of the coarse grid Example 3.6.1), but in principle they can be arbitrary (as long as they are nonnegative and sum to 1 over y ). Intuitively, φ jy may be interpreted as some measure of 'strength of relation' of j to y . The vectors ¶ φ jy ♣ j = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ may also be viewed as basis functions for a linear cost function approximation via Eq. (3.81).

## Hard Aggregation and Error Bound

A special case of interest, called hard aggregation , is when for every state j , we have φ jy = 0 for all representative states y , except a single one, denoted y j , for which we have φ jy j = 1 (we also require φ yy = 1 for all representative

= 0 or 1 for all connects to a single

Figure 3.6.4 Illustration of the piecewise constant cost approximation

<!-- image -->

<!-- formula-not-decoded -->

in the hard aggregation case where we have φ jy = 0 for all representative states y , except a single one. Here ˜ J is constant and equal to r ∗ y for all j in the footprint set

<!-- formula-not-decoded -->

states y ). In this case, the one-step lookahead approximation

<!-- formula-not-decoded -->

is piecewise constant ; it is constant and equal to r ∗ y for all j in the set

<!-- formula-not-decoded -->

called the footprint of representative state y ; see Fig. 3.6.4. Moreover the footprints of all the representative states are disjoint and form a partition of the state space, i.e.,

<!-- formula-not-decoded -->

The footprint sets can be used to define a bound for the error ( J * -˜ J ). In particular, it can be shown that

<!-- formula-not-decoded -->

where is the maximum variation of J * within the footprint sets S y . This error bound result can be extended to the more general aggregation framework that will be given in the next section. Note the primary intuition derived from the bound: the error due to hard aggregation is small if J * varies little within each set S y .

<!-- formula-not-decoded -->

For a special hard aggregation case of interest, consider the geometrical context of Example 3.6.1. There, aggregation probabilities are often based on a nearest neighbor approximation scheme, whereby each nonrepresentative state j takes the cost value of the 'closest' representative state y , i.e.,

<!-- formula-not-decoded -->

Then all states j for which a given representative state y is the closest to j (the footprint of y ) are assigned equal approximate cost ˜ J ( j ) = r ∗ y .

## Methods for Solving the Aggregate Problem

The most straightforward way to solve the aggregate problem is to compute the aggregate problem transition probabilities ˆ p xy ( u ) [cf. Eq. (3.79)] and transition costs ˆ g ( x↪ u ) [cf. Eq. (3.80)] by either an algebraic calculation or by simulation. The aggregate problem may then be solved by any one of the standard methods, such as VI or PI. This exact calculation is plausible if the number of representative states is relatively small. An alternative possibility is to use a VI or PI method that is based on simulation. We refer to a discussion of these methods in the author's books [Ber12], Section 6.5.2, and [Ber19a], Section 6.3. The idea is that a simulator for the original problem can be used to construct a simulator for the aggregate problem; cf. Fig. 3.6.3.

An important observation is that if the original problem is deterministic and hard aggregation is used, the aggregate problem is also deterministic, and can be solved by shortest-path like methods. This is true for both discounted problems and for undiscounted shortest path-type problems. In the latter case, the termination state of the original problem must be included as a representative state in the aggregate problem. However, if hard aggregation is not used, the aggregate problem will be stochastic, because of the introduction of the aggregation probabilities. Of course, once the aggregate problem is solved and the lookahead approximation ˜ J is obtained, a deterministic structure in the original problem can be exploited to facilitate the on-line lookahead minimizations.

A

Travel speed

1 m/sec

1000 m

B

Figure 3.6.5 Illustration of discretization issues for problems with infinite state and control spaces.

<!-- image -->

## 3.6.2 Continuous Control Space Discretization

Aggregation with representative states extends without di ffi culty to problems with a continuous state space, as long as the control space is finite. Then once the representative states and the aggregation probabilities have been defined, the corresponding aggregate problem is a discounted problem with finite state and control spaces, which can be solved with the standard methods. The only potential di ffi culty arises when the disturbance space is also infinite, in which case the calculation of the transition probabilities and expected stage costs of the aggregate problem must be obtained by some form of integration process.

The case where both the state and the control spaces are continuous is somewhat more complicated, because both of these spaces must be discretized using representative state-control pairs, instead of just representative states. The following example illustrates what may happen if we use representative state discretization only.

## Example 3.6.2 (Continuous Shortest Path Discretization)

Suppose that we want to find the fastest route for a car to travel between two points A and B located at the opposite ends of a square with side 1000 meters, while avoiding some known obstacles. We assume a constant car speed of 1 meter per second and that the car can drive in any direction; cf. Fig. 3.6.5.

Let us consider discretizing the space with a square grid (a set of representative states), and restrict the directions of motion to horizontal and vertical, so that at each stage the car moves from a grid point to one of the four closest grid points. Thus in the discretized version of the problem the car travels with a sequence of horizontal and vertical moves as indicated in the right side of Fig. 3.6.5. Is it possible to approximate the fastest route arbi-

trarily closely with the optimal solution of the discretized problem, assuming a su ffi ciently fine grid?

The answer is no! To see this note that in the discretized problem the optimal travel time is 2000 secs, regardless of how fine the discretization is. On the other hand, in the continuous space/nondiscretized problem the optimal travel time can be as little as √ 2 · 1000 secs (this corresponds to the favorable case where the straight line from A to B does not meet an obstacle).

The di ffi culty in the preceding example is that the state space is discretized finely but the control space is not . What is needed is to introduce a fine discretization of the control space as well, through some set of 'representative controls.' We can deal with this situation with a suitable form of discretized aggregate problem, which when solved provides an appropriate form of cost function approximation for use with one-step lookahead. The discretized problem is a stochastic infinite horizon problem, even if the original problem is deterministic. Further discussion of this approach is outside our scope, and we refer to the sources cited at the end of the chapter. Under reasonable assumptions we can show consistency, i.e., that the optimal cost function of the discretized problem converges to the optimal cost function of the original continuous spaces problem as the discretization of both the state and the control spaces becomes increasingly fine.

## 3.6.3 Continuous State Space - POMDP Discretization

Aggregation with representative states is very well suited to problems with continuous state space and a finite control space because it results in an aggregate problem with finite state and control spaces, so the control discretization issue discussed in the preceding section does not arise. This situation arises prominently in POMDP problems, as we will explain in this section.

Let us consider an α -discounted DP problem, where the state space is a bounded convex subset B of a Euclidean space, such as the unit simplex, but the control space U is finite. We use b to denote the states, to emphasize the connection with belief states in POMDP and to distinguish them from x , which we will use to denote representative states. Bellman's equation has the form J = TJ , with the Bellman operator T defined by

<!-- formula-not-decoded -->

We introduce a set of representative states ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x m ♦ ⊂ B . We assume that the convex hull of ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x m ♦ is equal to B , so each state b ∈ B can be expressed as

<!-- formula-not-decoded -->

where ¶ φ bx i ♣ i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m ♦ is a probability distribution:

<!-- formula-not-decoded -->

We view φ bx i as aggregation probabilities.

Consider the operator ˆ T that transforms a vector r = ( r x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ r x m ) into the vector ˆ Tr with components ( ˆ Tr )( x 1 ) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ ( ˆ Tr )( x m ) defined by

<!-- formula-not-decoded -->

  where φ f ( x i ↪u↪w ) x j are the aggregation probabilities of the state f ( x i ↪ u↪ w ). It can then be shown that ˆ T is a contraction mapping with respect to the maximum norm (we give the proof for a similar result in the next section). Bellman's equation for an aggregate finite-state discounted DP problem whose states are x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x m has the form

<!-- formula-not-decoded -->

and has a unique solution.

The transitions in this problem occur as follows: from state x i under control u , we first move to f ( x i ↪ u↪ w ) at cost g ( x i ↪ u↪ w ), and then we move to a state x j , j = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , according to the probabilities

<!-- formula-not-decoded -->

The optimal costs r ∗ x i , i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , of this problem can often be obtained by standard VI and PI methods that may or may not use simulation. We can then approximate the optimal cost function of the original problem by

<!-- formula-not-decoded -->

and reasonably expect that the optimal discretized solution converges to the optimal as the number of representative states increases.

In the case where B is the belief space of an α -discounted POMDP, the representative states/beliefs and the aggregation probabilities define an aggregate problem, which is a finite-state α -discounted problem with a perfect state information structure. This problem can be solved with exact DP methods if either the aggregate transition probabilities and transition costs can be obtained analytically (in favorable cases) or if the number of representative states is small enough to allow their calculation by simulation. The aggregate problem can also be addressed with the approximate DP methods that we have discussed earlier, such as problem approximation/certainty equivalence approaches. Another possibility is the use of a rollout method, which is well-suited for an on-line implementation. See also the paper by Li, Hammar, and Bertsekas [LHB25], which develops a sophisticated POMDP aggregation methodology, based on belief features, and provides extensive computational results.

## 3.6.4 General Aggregation

We will now discuss a more general aggregation framework for the infinite horizon n -state α -discounted problem. We essentially replace the representative states x with subsets I x ⊂ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ of the original state space.

## General Aggregation Framework

We introduce a finite subset A of aggregate states, which we denote by symbols such as x and y . We define:

- (a) A collection of subsets I x ⊂ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ , x ∈ A .
- (b) A probability distribution over ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ for each x ∈ A , denoted by ¶ d xi ♣ i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ , and referred to the disaggregation probabilities of x . We require that the distribution corresponding to x is concentrated on the subset I x :

<!-- formula-not-decoded -->

- (c) For each original system state j ∈ ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ , a probability distribution over A , denoted by ¶ φ jy ♣ y ∈ A♦ , and referred to as the aggregation probabilities of j .

The aggregation and disaggregation probabilities specify a dynamic system involving both aggregate and original system states; cf. Fig. 3.6.6. In this system:

- (i) From aggregate state x , we generate an original system state i ∈ I x according to d xi .
- (ii) We generate transitions between original system states i and j according to p ij ( u ), with cost g ( i↪ u↪ j ).
- (iii) From original system state j , we generate aggregate state y according to φ jy .

The optimal costs of the aggregate states y ∈ A in the aggregate problem are denoted by r ∗ y , and they define approximate costs for the original problem through the interpolation formula

<!-- formula-not-decoded -->

Our general aggregation framework is illustrated in Fig. 3.6.6. While the sets I x are often constructed by using features, we will formulate our aggregation framework in a general form, and introduce features later. Note

Disaggregation

Probabilities doi

Aggregation Probabilities

Disaggregation Probabilities

Aggregation

Probabilities

Фіз

Figure 3.6.6 Illustration of the aggregate system, and the transition mechanism and the costs per stage of the aggregate problem.

<!-- image -->

that if each set I x consists of a single state, we obtain the representative states framework of the preceding section. In this case the disaggregation distribution ¶ d xi ♣ i ∈ I x ♦ is just the atomic distribution that assigns probability 1 to the unique state in I x . Consistent with the special case of representative states, the disaggregation probability d xi may be interpreted as a 'measure of the relation of x and i .'

The aggregate problem is a DP problem with an enlarged state space that consists of two copies of the original state space ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ plus the set of aggregate states A . We introduce the corresponding optimal vectors ˜ J 0 , ˜ J 1 , and r ∗ = ¶ r ∗ x ♣ x ∈ A♦ where:

r ∗ x is the optimal cost-to-go from aggregate state x .

- ˜ J 0 ( i ) is the optimal cost-to-go from original system state i that has just been generated from an aggregate state (left side of Fig. 3.6.6).
- ˜ J 1 ( j ) is the optimal cost-to-go from original system state j that has just been generated from an original system state (right side of Fig. 3.6.6).

Note that because of the intermediate transitions to aggregate states, ˜ J 0 and ˜ J 1 are di ff erent.

These three vectors satisfy the following three Bellman equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Original

System States

Рід (и), 9(i, и, j)

Aggregation Probabilities

<!-- formula-not-decoded -->

The objective is to solve for the optimal costs r ∗ x of the aggregate states in order to obtain approximate costs for the original problem through the interpolation formula

<!-- formula-not-decoded -->

cf. Eq. (3.83).

By combining the three Bellman equations (3.84)-(3.86), we see that r ∗ satisfies

<!-- formula-not-decoded -->

or equivalently r ∗ = Hr ∗ , where H is the operator that maps the vector r to the vector Hr with components

<!-- formula-not-decoded -->

It can be shown that H is a contraction mapping with respect to the maximum norm , and thus the composite Bellman equation (3.87) has r ∗ as its unique solution. To see this, we note that for any vectors r and r ′ , we have

<!-- formula-not-decoded -->

where ‖ · ‖ is the maximum norm, and the equality follows from the definition of ( Hr ′ )( x ), and the fact that d xi , p ij ( u ), and φ jy are probabilities. It follows that

<!-- formula-not-decoded -->

By reversing the roles of r and r ′ , we also have

<!-- formula-not-decoded -->

so that

∣ ∣ By taking the maximum over x ∈ A , we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we conclude that H is a maximum norm contraction.

Note that the composite Bellman equation (3.87) has dimension equal to the number of aggregate states, which is potentially much smaller than n . To apply the aggregation framework of this section, we may solve exactly this equation for the optimal aggregate costs r ∗ x , x ∈ A , by simulation-based analogs of the VI and PI methods, and obtain a cost function approximation for the original problem through the interpolation formula (3.83); see [Ber12], Section 6.5.2 and [Ber19a], Section 6.3. These methods have strong convergence properties thanks to the contraction property (3.89).

## 3.6.5 Types of Aggregation and Error Bounds

Let us consider the set

<!-- formula-not-decoded -->

called the footprint of aggregate state y (extending our earlier terminology of Section 3.6.1). It consists of all states j that we view as 'related' to aggregate state y . Since we have viewed I x as the set of states i that are 'related' to aggregate state x , it makes sense to assume that for every aggregate state x we have

<!-- formula-not-decoded -->

Intuitively, this means that if a state i is 'related' to aggregate state x in the disaggregation process, then i is also 'related' to x in the aggregation process.

The case of hard aggregation is of special interest. Here, for each state j , we have φ jy = 0 for all aggregate states y , except a single one, denoted y j , for which φ jy j = 1. In this case, the footprints of all the aggregate states are disjoint and form a partition of the state space, i.e.,

<!-- formula-not-decoded -->

cf. our discussion of Section 3.6.1. Then, the one-step lookahead approximation

<!-- formula-not-decoded -->

is piecewise constant; it is constant and equal to r ∗ y for all j in the footprint set S y of y . The condition (3.91) states that φ ix = 1 if d xi &gt; 0, or in words, that the set I x into which the aggregate state x disaggregates is a subset of the footprint set of x .

Aside from hard aggregation and aggregation with representative states, there are several other special cases of aggregation that have received attention in the literature:

- (a) Soft aggregation : This is an extension of hard aggregation, where there is a 'soft' boundary between the sets of the state space partition, i.e., the footprint sets overlap partially. The aggregation probabilities are chosen to be positive for the states of overlap, so that the cost approximation ˜ J is piecewise constant, except along the states of footprint overlap, where ˜ J changes 'smoothly;' see Singh, Jaakkola, and Jordan [SJJ95].
- (b) Aggregation with representative features : Here the aggregate states are characterized by nonempty subsets of original system states, which, however, may not form a partition of the original state space. In an important example of this scheme, we choose a collection of distinct representative feature vectors, and we associate each one of them with an aggregate state consisting of the subset of original system states that share the corresponding feature value (see [Ber12], Section 6.5, or [Ber19a], Section 6.2). The author's paper [Ber18a] provides an overview of feature-based aggregation, and discusses ways to combine the methodology with the use of deep neural networks.

We can show the following error bound, first given by Tsitsiklis and Van Roy [Van95], [TsV96], for the case of hard aggregation, and extended to the more general case, where just the condition (3.91) holds, by Li and Bertsekas [LiB25b].

## (Error Bound for General Aggregation)

Proposition 3.6.1: Let the condition (3.91) hold. Then, we have

<!-- formula-not-decoded -->

where /epsilon1 is the maximum variation of the optimal cost function J * over the footprint sets S y , y ∈ A :

<!-- formula-not-decoded -->

The meaning of the preceding proposition is that if the optimal cost

function J * varies by at most /epsilon1 within each footprint set S y , the aggregation scheme yields a piecewise constant approximation to the optimal cost function that is within /epsilon1 glyph[triangleleft] (1 -α ) of the optimal. The paper [LiB25b] also provides an example where the bound fails to hold because the condition (3.91) is violated; see Exercise 3.3.

## Selecting the Aggregate States

Generally, the method to select the aggregate states is an important issue, for which there is no mathematical theory at present. In practice, intuition and problem-specific insights often suggest reasonable choices, which can then be fine-tuned through experimentation. For example, suppose that the optimal cost function J * is piecewise constant over a partition ¶ S y ♣ y ∈ A♦ of the state space ¶ 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ . By this we mean that for some vector we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then from Prop. 3.6.1 it follows that the hard aggregation scheme with I x = S x for all x ∈ A is exact, meaning that r ∗ x are the optimal costs of the aggregate states x in the aggregate problem. This suggests that the states in the footprint set S y corresponding to an aggregate state y should have roughly equal optimal cost , in line with the error bound of Prop. 3.6.1.

Expanding on this idea, suppose that through some special insight into the problem's structure or preliminary calculations, we know some features of the system's state that can 'predict well' its optimal cost when combined through some approximation architecture, e.g., one that is linear. Then it seems reasonable to form the set of aggregate states A of a hard aggregation scheme so that the sets I y and S y consist of states with 'similar features' for every y ∈ A . This approach, suggested in the book [BeT96], Section 3.1.2, is known as feature-based aggregation . The next section considers this possibility, and provides a way to introduce features and nonlinearities into the aggregation architecture, without compromising its other favorable aspects.

## 3.6.6 Aggregation Using Features

Let us focus on the guideline for hard aggregation discussed above: states i that belong to the same footprint set S y should have nearly equal optimal costs , i.e.,

<!-- formula-not-decoded -->

This raises the question of how to select the sets S y according to this guideline.

armoration fromorrorl, nf

Feature

Extraction

Footprint Sy

State Space

Feature Space

• У

Aggregate States

<!-- image -->

Aggregate States Features

Figure 3.6.7 Feature-based hard aggregation using a partition of the space of features. Each aggregate state y has a footprint S y that consists of states with 'similar' features, i.e., states that map into the same subset of a partition in the space of features.

One approach is to use a feature mapping , i.e., a function F that maps a state i into an m -dimensional feature vector F ( i ); cf. Example 3.1.7. In particular, suppose that F has the property that states i with nearly equal feature vector have nearly equal optimal cost J * ( i ). Then we can form the sets S y by grouping together states with nearly equal feature vectors. Specifically, given F , we introduce a more or less regular partition of the feature space [the subset of /Rfractur m that consists of all possible feature vectors F ( i )]. This partition induces a possibly irregular collection of subsets of the original state space. Each of these subsets can then be used as the footprint of a distinct aggregate state; see Fig. 3.6.7.

Note that in the resulting aggregation scheme the number of aggregate states may become very large. However, this approach o ff ers a significant advantage over the linear feature-based architectures of Section 3.1, where each feature is assigned a single weight: in feature-based hard aggregation, a weight is assigned to each subset of the feature space partition (possibly a weight to every possible feature value, in the extreme case where each feature value is viewed by itself as a distinct set of the partition). In e ff ect we use aggregation to construct a nonlinear (piecewise constant) feature-based architecture, which may be much more powerful than the corresponding linear architecture of Section 3.1.

A question that now arises is how to obtain a suitable feature vector when there is no obvious choice, based on problem-specific considerations. One option, proposed in the author's paper [Ber18a] and also discussed in the book [Ber19a] (Section 6.4), is to obtain 'good' features by using a neural network. More generally, any method that automatically generates features from data may be used.

## 3.6.7 Biased Aggregation

In this section we will introduce an extension of the preceding aggregation frameworks, called biased aggregation . It involves a vector V =

The aggregation framework of this section was proposed in the author's paper [Ber18b], which contains much additional material. It is related to a clas-

( V (1) ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ V ( n ) ) ↪ called the bias vector or bias function , which a ff ects the cost structure of the aggregate problem, and biases the values of its optimal cost function towards their correct levels. For practical purposes the values V ( i ) at various states should be readily accessible through simple computation or precomputation. Generally, the bias function V is obtained with some method that approximates J * , such as for example neural networkbased approximate PI, rollout, or problem approximation.

/negationslash

When V = 0, biased aggregation is identical to the aggregation scheme of Section 3.6.4. When V = 0, biased aggregation yields an approximation to J * that is equal to V plus a local correction ˆ J ; see Fig. 3.6.8. The method relies on a simple cost function change, as indicated below and explained later.

## Biased Aggregation Method

We modify the original problem by replacing the cost per stage,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then apply the aggregation method of Section 3.6.4 to the modified cost problem to obtain a cost approximation ˆ J . The function

<!-- formula-not-decoded -->

forms an approximation to the optimal cost function J * ( i ) of the original problem.

To justify the biased aggregation method, consider the optimal cost function ˆ J of the modified cost problem, i.e., the one with cost per stage given by Eq. (3.92). It satisfies the corresponding Bellman equation:

<!-- formula-not-decoded -->

or equivalently

<!-- formula-not-decoded -->

sical DP scheme, known as cost shaping in the RL literature; see e.g., [Ber19a], the references quoted there.

with

3.6.10.

Figure 3.6.8 Schematic illustration of biased aggregation. It provides an approximation ˜ J to J ∗ that is equal to the bias function V plus a local correction ˆ J , which is obtained by solving a modified cost problem with the aggregation method of Section 3.6.4.

<!-- image -->

By comparing this equation with the Bellman equation for the original problem, we see that the optimal cost functions of the modified and the original problems are related by

<!-- formula-not-decoded -->

and that the two problems have the same optimal policies. This of course assumes that the modified cost problem is solved exactly. If instead it is solved approximately using (unbiased) aggregation, the choice of V and the approximation architecture may a ff ect substantially the character of the resulting approximation in value space scheme and the quality of suboptimal policies obtained.

Figure 3.6.9 provides an interpretation of biased aggregation, which is consistent to the one of the (unbiased) aggregation scheme of Section 3.6.4. It involves three sets of states: two copies of the original state space, as shown in the figure, as well as a finite set A of aggregate states. The state transitions go from a state in A to a state in the left state space copy, according to disaggregation probabilities, then to a state in the right state space copy, and then back to a state in A , according to aggregation probabilities, and the process is repeated. At a state i in the left state space copy we must choose a control u ∈ U ( i ), and then transition to a state j in the right state space copy at a cost g ( i↪ u↪ j ) according to the original system transition probabilities p ij ( u ).

The key insight here is that biased aggregation can be viewed as unbiased aggregation applied to a modified DP problem , with cost per stage given by Eq. (3.92), which is equivalent to the original DP problem in the sense that it has the same optimal policies. Thus any unbiased aggregation scheme and algorithm, when applied to the modified DP problem, yields a biased aggregation scheme and algorithm for the original DP problem. As

Correction (piecewise constant or piecewise linear)

Aggregation Probabilities

Disaggregation Probabilities

Figure 3.6.9 Illustration of the transition mechanism and the costs per stage of the aggregate problem in biased aggregation. When the bias function V is identically zero, we obtain the aggregation framework of Section 3.6.4.

<!-- image -->

a result, we can straightforwardly transfer results, algorithms, and intuition from our earlier unbiased aggregation analysis to the biased aggregation framework, by applying them to the unbiased aggregation framework that corresponds to the modified stage cost (3.92). Moreover, we may use simulation-based algorithms for policy evaluation, policy improvement, and Q-learning for the aggregate problem, with the only requirement that the value V ( i ) for any state i is available when needed.

Regarding the choice of V , consistent with our earlier analysis, the general principle is that V should capture a fair amount of the nonlinearity or 'shape' of J * . In the case of hard aggregation, based on the error bound of Prop. 3.6.1, the function V should be chosen so that the variation of J * ( i ) -V ( i ) is as small as possible within the corresponding footprint sets. This suggests that V should be chosen as a good approximation to J * (within a constant value).

## 3.6.8 Asynchronous Distributed Aggregation

Let us now discuss the distributed solution of large-scale discounted DP problems using cost function approximation, multiple agents/processors, and hard aggregation. Here we partition the original system states into aggregate states/subsets x ∈ A = ¶ x 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ x m ♦ , and we employ a network of processors/agents. Each processor updates asynchronously a detailed/exact local cost function, defined on a single aggregate state/subset. Each processor also maintains an aggregate cost for its aggregate state, which is a weighted average of the detailed cost of the (original system) states in the processor's subset, weighted by the corresponding disaggregation probabilities. These aggregate costs are communicated between processors and are used to perform the local updates.

Aggregation Probabilities

In a synchronous VI method of this type, each processor /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , maintains and updates a (local) cost J ( i ) for every original system state i ∈ x /lscript , and an aggregate cost

<!-- formula-not-decoded -->

where d x /lscript i are the corresponding disaggregation probabilities. We generically denote by J and R the vectors with components J ( i ), i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n , and R ( /lscript ), /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , respectively. These components are updated according to

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

where the mapping H /lscript is defined for all /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , i ∈ x /lscript , u ∈ U ( i ), and J ∈ /Rfractur n , R ∈ /Rfractur m , by

<!-- formula-not-decoded -->

and where for each original system state j , we denote by x ( j ) the subset to which j belongs [i.e., j ∈ x ( j )]. Thus the iteration (3.93) is the same as ordinary VI, except that instead of J ( j ), we use the aggregate costs R x ( j ) for the states j whose costs are updated by other processors.

( ) It is possible to show that the iteration (3.93)-(3.94) involves a supnorm contraction mapping of modulus α , so it converges to the unique solution of the system of equations in ( J↪ R )

<!-- formula-not-decoded -->

This follows from the fact that ¶ d x /lscript i ♣ i = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ n ♦ is a probability distribution. We may view the equations (3.96) as a set of Bellman equations for an 'aggregate' DP problem, which similar to our earlier discussion, involves both the original and the aggregate system states. The di ff erence from the Bellman equations (3.84)-(3.86) is that the mapping (3.95) involves J ( j ) rather than R x ( j ) for j ∈ x /lscript .

( ) In the algorithm (3.93)-(3.94), all processors /lscript must be updating their local costs J ( i ) and aggregate costs R ( /lscript ) synchronously, and communicate the aggregate costs to the other processors before a new iteration may begin. This is often impractical and time-wasting. In a more practical asynchronous version of the method, the aggregate costs R ( /lscript ) may be outdated

to account for communication 'delays' between processors. Moreover, the costs J ( i ) need not be updated for all i ; it is su ffi cient that they are updated by each processor /lscript only for a (possibly empty) subset of I /lscript ↪k of the aggregate state/set x /lscript . In this case, the iteration (3.93)-(3.94) is modified to take the form

<!-- formula-not-decoded -->

with 0 ≤ τ /lscript ↪k ≤ k for /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , and

<!-- formula-not-decoded -->

The di ff erences k -τ /lscript ↪k , /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m , in Eq. (3.97) may be viewed as 'delays' between the current time k and the times τ /lscript ↪k when the corresponding aggregate costs were computed at other processors. For convergence, it is of course essential that every i ∈ x /lscript belongs to I /lscript ↪k for infinitely many k (so each cost component is updated infinitely often), and lim k →∞ τ /lscript ↪k = ∞ for all /lscript = 1 ↪ glyph[triangleright] glyph[triangleright] glyph[triangleright] ↪ m (so that processors eventually communicate more recently computed aggregate costs to other processors).

Convergence of this type of method can be established using the asynchronous convergence theory for DP developed by the author in the paper [Ber82a] (see the books [BeT89], [Ber12], [Ber22b] for a more detailed treatment). The proof is based on the sup-norm contraction property of the mapping underlying Eq. (3.96). The monotonicity property is also su ffi -cient to establish asynchronous convergence, and this is useful in the convergence analysis of related aggregation algorithms for undiscounted DP models (see the paper by Bertsekas and Yu [BeY10]).

## 3.7 NOTES AND SOURCES

Section 3.1 : Our discussion of approximation architectures, neural networks, and training has been limited, and aimed just to provide the connection with approximate DP. The literature on the subject is vast, and some of the textbooks mentioned in the references to Chapter 1 provide detailed accounts and many sources, in addition to the ones given in Sections 3.1 and 3.2.

There are two broad directions of inquiry in parametric architectures:

- (1) The design of architectures, either in a general or a problem-specific context.
- (2) The training of neural networks, as well as other linear and nonlinear architectures.

Research along both of these directions has been extensive and is continuing.

Methods for selection of basis functions have received much attention, particularly in the context of neural network research and deep reinforcement learning (see e.g., the book by Goodfellow, Bengio, and Courville [GBC16]). For discussions that are focused outside the neural network area, see Bertsekas and Tsitsiklis [BeT96], Keller, Mannor, and Precup [KMP06], Jung and Polani [JuP07], Bertsekas and Yu [BeY09], and Bhatnagar, Borkar, and Prashanth [BBP13]. Moreover, there has been considerable research on optimal feature selection within given parametric classes (see Menache, Mannor, and Shimkin [MMS05], Yu and Bertsekas [YuB09], Busoniu et al. [BBD10a], and Di Castro and Mannor [DiM10]).

Incremental algorithms are the principal methods for training approximation architectures. They are supported by substantial theoretical analysis, which addresses issues of convergence, rate of convergence, stepsize selection, and component order selection. Moreover, incremental algorithms have been extended to constrained optimization settings, where the constraints are also treated incrementally, first by Nedi­ c [Ned11], and then by several other authors: Bertsekas [Ber11a], Wang and Bertsekas [WaB15], [WaB16], Bianchi [Bia16], Iusem, Jofre, and Thompson [IJT18]. It is beyond our scope to cover this analysis. The author's surveys [Ber10a] and [Ber15b], and convex optimization and nonlinear programming textbooks [Ber15a], [Ber16], collectively contain an extensive account of incremental methods, including the Kaczmarz, incremental gradient, subgradient, aggregated gradient, Newton, Gauss-Newton, and extended Kalman filtering methods, and give many references. The book [BeT96] and paper [BeT00] by Bertsekas and Tsitsiklis, and the survey by Bottou, Curtis, and Nocedal [BCN18] provide theoretically oriented treatments.

Linear feature-based architectures can also be trained with temporal di ff erences methods, such as TD( λ ), LSTD( λ ), and LSPE( λ ), which are not discussed in this book; see [BeT96], [Ber12], [Ber19a], [SuB18]. These methods can be viewed as algorithms for solving general linear systems of equations by Monte-Carlo simulation, a subject that is of interest beyond approximate DP and entails much subtlety. We refer to the work of H. Yu, M. Wang, and the author for an in-depth analysis and discussion [BeY07], [BeY09], [YuB10], [Yu10], [Ber11c], [WaB13a], [WaB13b]. Section 7.3 of the book [Ber12] provides an extensive textbook treatment and additional references.

Section 3.2 : The publicly and commercially available neural network training programs incorporate heuristics for scaling and preprocessing data, stepsize selection, initialization, etc, which can be very e ff ective in specialized problem domains. We refer to books on neural networks such as Bishop [Bis95], Goodfellow, Bengio, and Courville [GBC16], Haykin [Hay08]. The recent book by Bishop and Bishop [BiB24] includes discussions of deep

neural networks and transformers.

Deep neural networks have created a lot of excitement in the machine learning field, in view of some high profile successes in image and speech recognition, and in RL with the AlphaGo and AlphaZero programs. One question is whether and for what classes of target functions we can enhance approximation power by increasing the number of layers while keeping the number of weights constant. For discussion, analysis, and speculation around this question, see Bengio [Ben09], Liang and Srikant [LiS16], Yarotsky [Yar17], and Daubechies et al. [DDF19].

Another important research question relates to the role of overparametrization in the success of deep neural networks. With more weights than training data, the training problem has infinitely many solutions, each providing an architecture that fits the training data perfectly. The question then is how to select a solution that works well on test data (i.e., data outside the training set); see Zhang et al. [ZBH16], [ZBH21], Belkin, Ma, and Mandal [BMM18], Belkin, Rakhlin, and Tsybakov [BRT18], Soltanolkotabi, Javanmard, and Lee [SJL18], Bartlett et al. [BLL19], Hastie et al. [HMR19], Muthukumar, Vodrahalli, and Sahai [MVS19], Su and Yang [SuY19], Sun [Sun19], Vaswani et al. [VLK21], Zhang et al. [ZBH21], and the discussions in the machine learning books by Hardt and Recht [HaR21], and Bishop and Bishop [BiB24].

Section 3.3 : Fitted value iteration has a long history; it was mentioned by Bellman among others. It has interesting properties, and at times exhibits pathological/unstable behavior due to accumulation of errors over a long horizon (see [Ber19a], Section 5.2).

The approximate policy iteration method of Section 3.3.3 has been proposed by Fern, Yoon, and Givan [FYG06], and variants have also been discussed and analyzed by several other authors. The method (with some variations) has been used to train a tetris playing computer program that performs impressively better than programs that are based on other variants of approximate policy iteration; see Scherrer [Sch13], Scherrer et al. [SGG15], and also Gabillon, Ghavamzadeh, and Scherrer [GGS13], who also provide an analysis of the method. The RL and approximate DP books collectively describe several alternative simulation-based methods for policy evaluation; see e.g., [BeT96], [SuB18], [Ber12], Chapters 6 and 7. These include temporal di ff erence methods, which enjoyed much popularity in the early days of RL. They are stochastic iterative algorithms that are closely related to Galerkin approximation, a major computational approach for solving large scale equations, as first observed by Yu and Bertsekas [YuB10], and Bertsekas [Ber11c]; see also Szepesvari [Sze11].

Simulation-based methods for approximate DP can benefit significantly from the use of parallel and distributed computation. A simple possibility is to parallelize the collection of Monte Carlo simulation samples. The book [Ber20a] describes distributed versions of approximate policy it-

eration, which are based on partitioning of the state space.

The original proposal of SARSA (Section 3.3.4) is attributed to Rummery and Niranjan [RuN94], with related work presented in the papers by Peng and Williams [PeW96], and Wiering and Schmidhuber [WiS98]. The ideas of the DQN algorithm attracted much attention following the paper by Mnih et al. [MKS15], which reported impressive test results on a suite of 49 classic Atari 2600 games.

The rollout and approximate PI methodology for POMDP of Section 3.3.5 was described in the author's RL book [Ber19a]. It was extended and tested in the paper by Bhattacharya et al. [BBW20] in the context of a challenging pipeline repair problem. A skillful application of approximate PI in combination with deep neural networks was given by Rybicki and Nelson [RyN25].

Advantage updating (Section 3.3.6) was proposed by Baird [Bai93], [Bai94], and is discussed further in Section 6.6 of the neuro-dynamic programming book [BeT96]. The di ff erential training methodology (Section 3.3.7) was proposed by the author in the paper [Ber97b], and followup work was presented by Weaver and Baxter [WeB99].

Generally, the challenges of implementing successfully approximate value and policy iteration schemes are quite formidable, and tend to be underestimated, because the literature naturally tends to place emphasis on success stories, and tends to underreport failures. In practice, the training di ffi culties, particularly exploration, must often be addressed on a case-bycase basis, and may require long and tricky parameter tuning, with little guarantee of ultimate success or even a diagnosis of the causes of failure. By contrast, approximation in value space with long multistep lookahead and simple terminal cost function approximation, and rollout (a single policy iteration starting from a base policy), while less ambitious, are typically much easier to implement, and often attain considerable success reliably. An intermediate approach that often works well is to use truncated rollout with a terminal cost function approximation that is trained with data.

Section 3.4 : Classification (sometimes called 'pattern classification' or 'pattern recognition') is a major subject in machine learning, for which there are many approaches, an extensive literature, and an abundance of public domain and commercial software; see e.g. the textbooks by Bishop [Bis95], [Bis06], Duda, Hart, and Stork [DHS12], and Hardt and Recht [HaR21]. Approximation in policy space was formulated as a classification problem in the context of DP by Lagoudakis and Parr [LaP03], and was followed up by several other authors (see e.g., Dimitrakakis and Lagoudakis [DiL08], Lazaric, Ghavamzadeh, and Munos [LGM10], Gabillon et al. [GLG11], Liu and Wei [LiW14], Farahmand et al. [FPB15], and the references quoted there). While we have focused on a classification approach that makes use of least squares regression and a parametric architecture, other classification methods may also be used. For example

the paper [LaP03] discusses the use of nearest neighbor schemes, support vector machines, as well as neural networks.

Section 3.5 : Our coverage of policy gradient and random search methods has aimed to provide an entry point into the field, and has been restricted to the o ff -line training of policies. For a detailed discussion and references on policy gradient methods, we refer to the book by Sutton and Barto [SuB18], the monograph by Deisenroth, Neumann, and Peters [DNP11], and the survey by Grondman et al. [GBL12]. An influential paper in this context by Williams [Wil92] proposed among others the likelihood-ratio policy gradient method given here. The methods of [Wil92] are commonly referred to as REINFORCE in the literature (see e.g., [SuB18], Ch. 13). For recent work on these and related methods, which give many additional references, see Furmston, Lever, and Barber [FLB16], Zhang et al. [ZKZ20], Bhatnagar [Bha23], Bhandari and Russo [BhR24], Maniyar et al. [MPM24], and Muller and Montufar [MuM24]. For general references on stochastic optimization, including stochastic gradient methods, see the books by Amari [Ama16], and Bertsekas and Tsitsiklis [BeT96], and the lecture notes by Duchi [Duc18].

There are several early works on search along randomly chosen directions (Rastrigin [Ras63], Matyas [Mat65], Aleksandrov, Sysoyev, and Shemeneva [ASS68], Rubinstein [Rub69]). For some more modern works, see Spall [Spa92], [Spa03], Duchi at al. [DJW12], [DJW15], and Nesterov and Spokoiny [NeS17]. For early works on simulation-based policy gradient schemes for various DP problems, see Glynn [Gly87], [Gly90], L'Ecuyer [L'Ec91], Fu and Hu [FuH94], Jaakkola, Singh, and Jordan [JSJ95], Cao and Chen [CaC97], Cao and Wan [CaW98]. More recent works have focused on the use of natural gradient scaling and a trust region; see the discussion and the references in Section 3.5.2.

Policy gradient-like methods : The main challenge in the successful implementation of policy gradient methods is twofold:

- (a) The di ffi culties with slow convergence. The detrimental e ff ects of simulation noise contribute further to slow convergence. Much work has been directed towards variations that address these di ffi culties, including the use of a baseline and variance reduction methods (Greensmith, Bartlett, and Baxter [GBB04], Greensmith [Gre05]), or second order information (Wang and Paschalidis [WaP17], and the references quoted there); cf. Section 3.5.2.
- (b) The presence of local minima. In RL applications, the cost function is typically nonconvex, particularly when neural networks are involved, giving rise to many local minima of dubious quality (see Exercise 3.2). This is a serious concern, which in practice, has been mitigated by trying many starting points, and other more or less heuristic randomization devices.

A further concern is that the method may yield a randomized policy [see Exercise 3.2(c)], which has to be converted to a nonrandomized policy for on-line implementation. Finally, a general issue to contend with arises in practical contexts that call for on-line replanning and possibly on-line policy retraining, e.g., in adaptive control.

The natural gradient approach has been used extensively, and have been e ff ective in improving the convergence rate of unscaled policy gradient methods; see the references given in Section 3.5.2. Its origins lie with concepts of information geometry, developed principally by Amari and described in his book [Ama16]. The constrained version of the natural gradient method, described in Section 3.5.2, is presented here for the first time, and has not been tested extensively. An alternative possibility to treat constraints directly, without a softmax reparametrization is the mirror descent algorithm, described in the context of policy gradient methods by Xiao [Xia22].

We have not covered actor-critic methods within the policy gradient context. These methods were introduced in the paper by Barto, Sutton, and Anderson [BSA83]. The more recent works of Sutton et al. [SMS99], Baxter and Bartlett [BaB01], Konda and Tsitsiklis [KoT99], [KoT03], Marbach and Tsitsiklis [MaT01], [MaT03], Peters and Schaal [PeS08], and Bhatnagar et al. [BSG09] have been influential; see also the survey by Grondman et al. [GBL12]. Actor-critic algorithms that are suitable for POMDP and involve gradient estimation have been given by H. Yu [Yu05], and Estanjini, Li, and Paschalidis [ELP12].

Random search methods : The cross-entropy method was initially developed in the context of rare event simulation and was later adapted for use in optimization. For textbook accounts, see Rubinstein and Kroese [RuK04], [RuK13], [RuK16], and Busoniu et al. [BBD10a], and for surveys see de Boer et al. [BKM05], and Kroese et al. [KRC13]. The method was proposed for policy search in an approximate DP context by Mannor, Rubinstein, and Gat [MRG03]. For recent analysis, see Joseph and Bhatnagar [JoB16], [JoB18].

It is generally thought that the cross entropy method, while applicable to problems with unrestricted parameter dimsion, is e ff ective primarily for a low-dimensional parameter space. One such context where the method was successfully applied is the game of tetris; see the papers by Szita and Lorinz [SzL06], and Thiery and Scherrer [ThS09]. These papers report much superior results to the ones obtained earlier by Bertsekas and Io ff e [BeI96], using approximate PI methods, and by Kakade [Kak02] using policy gradient methods (comparable results were obtained later with approximate PI methods by Gabillon, Ghavamzadeh, and Scherrer [GGS13]).

Are the policy gradient and random search approaches related? At first glance the answer seems negative, in view of the fundamental conceptual di ff erences between these two types of methods: while gradient-like

methods are guided by the gradient at the current iterate to move towards an improved iterate, random search methods involve stochastic exploration of the parameter space, with no strict requirement for cost improvement. In the context of RL applications, however, there is substantial commonality of ideas between the two types of methods. Policy gradient methods rely on randomization of the optimization variables (e.g., randomized policies) and on randomization of the starting parameter to escape from local minima (cf. the 2-state problem of Exercise 3.2). Conversely, random search methods often bias their random exploration towards lower cost solutions (cf. the cross entropy method and Fig. 3.5.4). Thus, while their mechanisms di ff er, policy gradient and random search methods are quite similar in RL practice, through shared principles of stochastic exploration and cost-driven refinement. In fact, methods that combine random search for global exploration with gradient-based local optimization have been proposed since the 1990s. Application contexts, have included hybrid systems (e.g., processes with discrete/continuous dynamics), and integer programming (e.g., combinations of branch-and-bound with gradient methods).

Section 3.6 : The aggregation approach has a long history in scientific computation and operations research (see for example Bean, Birge, and Smith [BBS87], Chatelin and Miranker [ChM82], Douglas and Douglas [DoD93], and Rogers et al. [RPW91]). It was introduced in the simulation-based approximate DP context, mostly in the form of VI; see Singh, Jaakkola, and Jordan [SJJ95], Gordon [Gor95], and Tsitsiklis and Van Roy [Van95], [TsV96]. It was further discussed in the neuro-dynamic programming book [BeT96], Sections 3.1.2 and 6.7.

In the RL literature, aggregation, as described here, is sometimes referred to as 'state abstraction.' Another scheme, called options , has been introduced by Sutton, Precup, and Singh [SPS99], and can be viewed as a formalization of temporal abstraction. It introduces additional multi-step 'macro-actions' that are available at some states. Macro-actions transfer the state of the system to one of a selected set of states through a sequence of actions at a given cost. Options can be viewed as a problem approximation approach: the set of actions available at a state is augmented with the additional macro-actions. The possibility of a synergistic combination of the options formalism with state aggregation has been discussed by Ciosek and Silver [Cio15], [CiS15].

The material on POMDP discretization (cf. Section 3.6.3) is based on the paper by Yu and Bertsekas [YuB04]. This paper provides also a similar discretization scheme for the average cost case where the Bellman equation need not have a solution, but the scheme nevertheless provides lower bounds to optimal average cost functions which are di ffi cult to compute. Here we have focused on the simpler discounted case. The paper by Li, Hammar, and Bertsekas [LHB25], which develops a more sophisticated POMDP aggregation methodology, and provides extensive computational

results.

The aggregation framework with representative features was introduced in the author's DP book [Ber12], was discussed in detail in the RL textbook [Ber19a] (Chapter 6), and was further developed in the survey paper [Ber18a], which provides an expanded view of the methodology. These sources also provide several VI and PI simulation-based methods for solving the aggregate problem. The paper by Yu and Bertsekas [YuB12] and the book [Ber12] discuss a view of aggregated equations as projected equations that involve a Euclidean norm or seminorm projection. This view provides a connection with the temporal di ff erence and Galerkin approximation methodologies, both of which also involve Euclidean projections.

Biased aggregation (Section 3.6.7) was introduced in the author's paper [Ber18b], which contains further analysis, discusses connections with rollout algorithms, and suggests additional methods. A noteworthy result given in that paper is that the bound of Prop. 3.6.1 admits an extension whereby the optimal cost function J * is replaced by J * -V . This shows that if the variation of J * -V is small, the performance of the aggregation method improves accordingly.

Distributed asynchronous aggregation (Section 3.6.8) was first proposed in the paper by Bertsekas and Yu [BeY10] (Example 2.5); see also the discussions in the author's DP books [Ber12] (Section 6.5.4) and [Ber22b] (Example 1.2.11). A recent computational study, related to distributed tra ffi c routing, was given by Vertovec and Margellos [VeM23].

Aggregation may also be used as a policy evaluation method in the context of policy iteration with linear feature-based cost function approximations. Within this context, the aggregation approach provides an alternative to the temporal di ff erence approach. These two approaches are described and compared in the author's approximate policy iteration survey paper [Ber11b]. Generally speaking, aggregation methods are characterized by stronger theoretical properties, such as Bellman operator monotonicity, resilience to policy oscillations, and better error bounds. On the other hand, they are more restrictive in their use of linear approximation architectures, compared with temporal di ff erence methods (see [Ber11b], [Ber18a]).

## E X E R C I S E S

## 3.1 (Proof of Prop. 3.4.1)

Complete the details of the following proof of Prop. 3.4.1. Fix c , and for any scalar y , consider for a given x the conditional expected value E { ( z ( c↪ c ′ ) -y ) 2 ♣ x } . Here the random variable z ( c↪ c ′ ) takes the value 1 with probability p ( c ♣ x ) and the value 0 with probability 1 -p ( c ♣ x ), so we have

<!-- formula-not-decoded -->

We minimize this expression with respect to y , by setting to 0 its derivative, i.e.,

<!-- formula-not-decoded -->

We thus obtain the minimizing value of y , namely y ∗ = p ( c ♣ x ) ↪ so that

<!-- formula-not-decoded -->

We set y = h ( c↪ x ) in the above expression and obtain

<!-- formula-not-decoded -->

Since this is true for all x , we also have

<!-- formula-not-decoded -->

showing that Eq. (3.35) holds for all functions h and all classes c .

## 3.2 (Local Minima in Policy Gradient Optimization)

This exercise explores some of the pitfalls of policy gradient optimization. Consider a deterministic α -discounted DP problem with two states, Left and Right, denoted by L and R , respectively, and the two controls Move-to-the-Left and Move-to-the-Right, denoted by u L and u R , respectively. The transition probabilities are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The transition cost for L → R is 2, and for R → L it is 2. The transition cost for L → L is 1, and for R → R it is 0. Starting from R , the optimal policy is to

• rIot Of Ja(r) lor various values or a; cr. Exercise s.-(a).

a = 0.80

a = 0.90

••- a = 0.95

25

20

15

10

-20

-15

-10

-5

T2

10

15

20

20

10

T1

Parametrized Policy

<!-- image -->

r

Figure 3.7.1 Markov chain and parametrized policy in Exercise 3.2.

<!-- image -->

Figure 3.7.2 Plot of J α ( r ) for various values of α ; cf. Exercise 3.2(a).

Figure 3.7.3 Two-dimensional plot of J α ( r ) for α = glyph[triangleright] 95, where r = ( r 1 ↪ r 2 ) are the parameters of the soft-max distribution that specifies the randomized policy; cf. Exercise 3.2(b).

<!-- image -->

stay in R . Starting from L , the optimal policy is to stay in L if α &lt; 1 glyph[triangleleft] 2 and to move to R if α &gt; 1 glyph[triangleleft] 2.

- (a) Consider policies that move to R with probability r and to L with probability 1 -r , starting from each of the two states; cf. Fig. 3.7.1. The

costs J α ( L ; r ) and J α ( R ; r ) of such a policy starting from L and from R , respectively, can be computed from the corresponding Bellman equation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider also the expected cost of the policy corresponding to r , assuming the initial states are L or R with equal probability 1/2:

<!-- formula-not-decoded -->

Compute and plot J α ( r ) for r ∈ [0 ↪ 1] and several values of α ∈ (0 ↪ 1). Verify that J α ( r ) is concave as a function of r , and that the values r = 0 and r = 1 are both local minima of J α ( r ). Moreover, a policy gradient method converges to either one of these local minima if started close enough to it; see Fig. 3.7.2. (This exercise is adapted from an example given in the paper [BhR24].)

- (b) Consider an equivalent soft-max parametrization of policies. Here, we replace the single parameter r of part (a), which is constrained by 0 ≤ r ≤ 1, with two parameters r 1 and r 2 , which are unconstrained. In particular, we move to state R with probability

<!-- formula-not-decoded -->

starting from each of the two states, and we move to L with probability

<!-- formula-not-decoded -->

starting from each of the two states. Compute and plot J α ( r 1 ↪ r 2 ) for several values of α ∈ (0 ↪ 1). Verify that J α ( r 1 ↪ r 2 ) has no local minima if started close enough to it; see Fig. 3.7.3. Explain this observation in light of Fig. 3.7.2.

- (c) Suppose that we change the problem from minimization to maximization, i.e., we view the 'costs' in Fig. 3.7.1 as 'rewards' to be maximized over an infinite horizon. Argue that the plots of Fig. 3.7.2, give the reward J α ( r ) of Eq. (3.98) correctly, so that for the values of α used in Fig. 3.7.2, there is a unique value of r that maximizes J α ( r ), and it lies strictly between 0 and 1. Thus a policy gradient method will tend to produce a randomized policy.

## 3.3 (Counterexample to the Aggregation Error Bound [LiB25b])

This exercise provides an example where the error bound of Prop. 3.6.1 fails to hold when the condition (3.91) is violated. Consider a system involving two absorbing states, 1 and 2, i.e.,

<!-- formula-not-decoded -->

with self transition costs

/negationslash

<!-- formula-not-decoded -->

Thus the infinite horizon costs (without aggregation) are

<!-- formula-not-decoded -->

Assume that there are two aggregate states x 1 and x 2 that disaggregate into states 1 and 2, respectively, but aggregate states 2 and 1, respectively, i.e.,

<!-- formula-not-decoded -->

Then /epsilon1 = 0 since the footprint sets S x 1 and S x 2 consist of a single state. Show that the true aggregation error is positive, i.e., the aggregation process is not exact. Hint : Verify that the sequence of generated costs starting from aggregate state x 1 is

<!-- formula-not-decoded -->

while the sequence of generated costs starting from aggregate state x 2 is

<!-- formula-not-decoded -->

so we have ˜ J ( i ) = J ∗ ( i ) for both states i = 1 ↪ 2.

## References

[ABB19] Agrawal, A., Barratt, S., Boyd, S., and Stellato, B., 2019. 'Learning Convex Optimization Control Policies,' arXiv:1912.09529; also in Learning for Dynamics and Control, pp. 361-373, 2020.

[ACD77] Athans, M., Casta˜ non, D., Dunn, K. P., Greene, C., Lee, W., Sandell, N., and Willsky, A., 1977. 'The Stochastic Control of the F-8C Aircraft Using a Multiple Model Adaptive Control (MMAC) Method - Part I: Equilibrium Flight,' IEEE Trans. on Automatic Control, Vol. 22, pp. 768-780.

[ACF02] Auer, P., Cesa-Bianchi, N., and Fischer, P., 2002. 'Finite Time Analysis of the Multiarmed Bandit Problem,' Machine Learning, Vol. 47, pp. 235-256.

[ADH19] Arora, S., Du, S. S., Hu, W., Li, Z., and Wang, R., 2019. 'Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks,' arXiv:1901.08584.

[AHZ19] Arcari, E., Hewing, L., and Zeilinger, M. N., 2019. 'An Approximate Dynamic Programming Approach for Dual Stochastic Model Predictive Control,' arXiv:1911.03728; also IFAC-Papers OnLine, 2020.

[AKFJ95] Abou-Kandil, H., Freiling, Gerhard, and Jank, G., 1995. 'On the Solution of Discrete-Time Markovian Jump Linear Quadratic Control Problems,' Automatica, Vol. 31, pp. 765768.

[ALZ08] Asmuth, J., Littman, M. L., and Zinkov, R., 2008. 'Potential-Based Shaping in Model-Based Reinforcement Learning,' Proc. of 23rd AAAI Conference, pp. 604-609.

[AMS09] Audibert, J.Y., Munos, R., and Szepesvari, C., 2009. 'Exploration-Exploitation Tradeo ff Using Variance Estimates in Multi-Armed Bandits,' Theoretical Computer Science, Vol. 410, pp. 1876-1902.

[AMS19] Agostinelli, F., McAleer, S., Shmakov, A., and Baldi, P., 2019. Solving the Rubik's Cube with Deep Reinforcement Learning and Search,' Nature Machine Intelligence, Vol. 1, pp. 356-363.

[ASP08] An, W., Singh, S., Pattipati, K. R., Kleinman, D. L., and Gokhale, S. S., 2008. 'Dynamic Scheduling of Multiple Hidden Markov Model-Based Sensors,' J. Advanced Info. Fusion, Vol. 3, pp. 33-49.

[ASR20] Andersen, A. R., Stidsen, T. J. R., and Reinhardt, L. B., 2020. 'SimulationBased Rolling Horizon Scheduling for Operating Theatres,' in SN Operations Research Forum, Vol. 1, pp. 1-26.

[ASS68] Aleksandrov, V. M., Sysoyev, V. I, and Shemeneva, V. V., 1968. 'Stochastic Optimization of Systems,' Engineering Cybernetics, Vol. 5, pp. 11-16.

[AXG16] Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P., 2016. 'Control Barrier Function Based Quadratic Programs for Safety Critical Systems,' IEEE Transactions on Automatic Control, Vol. 62, pp. 3861-3876.

[Abr90] Abramson, B., 1990. 'Expected-Outcome: A General Model of Static Evaluation,' IEEE Trans. on Pattern Analysis and Machine Intelligence, Vol. 12, pp. 182-193.

[Agr95] Agrawal, R., 1995. 'Sample Mean Based Index Policies with O (log n ) Regret for the Multiarmed Bandit Problem,' Advances in Applied Probability, Vol. 27, pp. 1054-1078.

[Ala22] Alamir, M., 2022. 'Learning Against Uncertainty in Control Engineering,' Annual Reviews in Control.

[Ama98] Amari, S. I., 1998. 'Natural Gradient Works E ffi ciently in Learning,' Neural Computation, Vol. 10, pp. 251276.

[Ama16] Amari, S. I., 2016. Information Geometry and its Applications, Springer.

[AnH14] Antunes, D., and Heemels, W.P.M.H., 2014. 'Rollout Event-Triggered Control: Beyond Periodic Control Performance,' IEEE Transactions on Automatic Control, Vol. 59, pp. 3296-3311.

[AnM79] Anderson, B. D. O., and Moore, J. B., 1979. Optimal Filtering, Prentice-Hall, Englewood Cli ff s, NJ.

[ArD21] Arora, S., and Doshi, P., 2021. 'A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress,' Artificial Intelligence, Vol. 297.

[AsH06] Astr om, K. J., and Hagglund, T., 2006. Advanced PID Control, Instrument Society of America, Research Triangle Park, N. C.

[AsW94] Astr  om, K. J., and Wittenmark, B., 1994. Adaptive Control, 2nd Edition, Prentice-Hall, Englewood Cli ff s, NJ.

[Ast83] Astr om, K. J., 1983. 'Theory and Applications of Adaptive Control - A Survey,' Automatica, Vol. 19, pp. 471-486.

[AtF66] Athans, M., and Falb, P., 1966. Optimal Control, McGraw-Hill, NY.

[AvB20] Avrachenkov, K., and Borkar, V. S., 2020. 'Whittle Index Based Q-Learning for Restless Bandits with Average Reward,' arXiv:2004.14427; also Automatica, Vol. 139, 2022.

[BAP08] Berger, C.R., Areta, J., Pattipati, K., and Willett, P., 2008. 'Compressed Sensing - A Look Beyond Linear Programming.' 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 3857-3860.

[BBB22] Bhambri, S., Bhattacharjee, A., and Bertsekas, D. P., 2022. 'Reinforcement Learning Methods for Wordle: A POMDP/Adaptive Control Approach,' arXiv:2211.10298.

[BBB23] Bhambri, S., Bhattacharjee, A., and Bertsekas, D. P., 2023. 'Playing Wordle Using an Online Rollout Algorithm for Deterministic POMDPs,' 2023 IEEE Conference on Games, Boston, MA.

[BBD08] Busoniu, L., Babuska, R., and De Schutter, B., 2008. 'A Comprehensive Survey of Multiagent Reinforcement Learning,' IEEE Transactions on Systems, Man, and Cybernetics, Part C, Vol. 38, pp. 156-172.

[BBD10a] Busoniu, L., Babuska, R., De Schutter, B., and Ernst, D., 2010. Reinforcement Learning and Dynamic Programming Using Function Approximators, CRC Press, NY.

[BBD10b] Busoniu, L., Babuska, R., and De Schutter, B., 2010. 'Multi-Agent Reinforce-

ment Learning: An Overview,' in Innovations in Multi-Agent Systems and Applications, Springer, pp. 183-221.

[BBG13] Bertazzi, L., Bosco, A., Guerriero, F., and Lagana, D., 2013. 'A Stochastic Inventory Routing Problem with Stock-Out,' Transportation Research, Part C, Vol. 27, pp. 89-107.

[BBM17] Borrelli, F., Bemporad, A., and Morari, M., 2017. Predictive Control for Linear and Hybrid Systems, Cambridge Univ. Press, Cambridge, UK.

[BBP13] Bhatnagar, S., Borkar, V. S., and Prashanth, L. A., 2013. 'Adaptive Feature Pursuit: Online Adaptation of Features in Reinforcement Learning,' in Reinforcement Learning and Approximate Dynamic Programming for Feedback Control , by F. Lewis and D. Liu (eds.), IEEE Press, Piscataway, NJ., pp. 517-534.

[BBS87] Bean, J. C., Birge, J. R., and Smith, R. L., 1987. 'Aggregation in Dynamic Programming,' Operations Research, Vol. 35, pp. 215-220.

[BBW20] Bhattacharya, S., Badyal, S., Wheeler, T., Gil, S., and Bertsekas, D. P., 2020. 'Reinforcement Learning for POMDP: Partitioned Rollout and Policy Iteration with Application to Autonomous Sequential Repair Problems,' IEEE Robotics and Automation Letters, Vol. 5, pp. 3967-3974.

[BCD10] Brochu, E., Cora, V. M., and De Freitas, N., 2010. 'A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,' arXiv:1012.2599.

[BCN18] Bottou, L., Curtis, F. E., and Nocedal, J., 2018. 'Optimization Methods for Large-Scale Machine Learning,' SIAM Review, Vol. 60, pp. 223-311.

[BFA22] Bouguila, N., Fan, W., and Amayri, M., eds., 2022. Hidden Markov Models and Applications. Springer, NY.

[BFH86] Breton, M., Filar, J. A., Haurie, A., and Schultz, T. A., 1986. 'On the Computation of Equilibria in Discounted Stochastic Dynamic Games,' in Dynamic Games and Applications in Economics, Springer, pp. 64-87.

[BGH22] Brunke, L., Gree ff , M., Hall, A. W., Yuan, Z., Zhou, S., Panerati, J., and Schoellig, A. P., 2022. 'Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 5, pp. 411-444.

[BKB23] Bhattacharya, S., Kailas, S., Badyal, S., Gil, S., and Bertsekas, D., 2023. 'Multiagent Reinforcement Learning: Rollout and Policy Iteration for POMDP with Application to Multi-Robot Problems,' IEEE Transactions on Robotics, Vol. 40, pp. 2003-2023.

[BKM05] de Boer, P. T., Kroese, D. P., Mannor, S., and Rubinstein, R. Y. 2005. 'A Tutorial on the Cross-Entropy Method,' Annals of Operations Research, Vol. 134, pp. 19-67.

[BLJ23] Bai, T., Li, Y., Johansson, K. H., and Martensson, J., 2023. 'Rollout-Based Charging Strategy for Electric Trucks with Hours-of-Service Regulations,' arXiv:2303. 08895; also IEEE Control Systems Letters, Vol. 7, 2023, pp. 2167-2172.

[BLL19] Bartlett, P. L., Long, P. M., Lugosi, G., and Tsigler, A., 2019. 'Benign Overfitting in Linear Regression,' arXiv:1906.11300; also Proc. of the National Academy of Sciences, Vol. 117, 2020.

[BLW91] Bittanti, S., Laub, A. J., and Willems, J. C., eds., 2012. The Riccati Equation, Springer.

[BMM18] Belkin, M., Ma, S., and Mandal, S., 2018. 'To Understand Deep Learning we Need to Understand Kernel Learning,' arXiv:1802.01396.

[BMM24] Bhusal, G., Miller, K., and Merkurjev, E., 2024. 'MALADY: Multiclass Active Learning with Auction Dynamics on Graphs,' arXiv:2409.09475.

[BPW12] Browne, C., Powley, E., Whitehouse, D., Lucas, L., Cowling, P. I., Rohlfshagen, P., Tavener, S., Perez, D., Samothrakis, S., and Colton, S., 2012. 'A Survey of Monte Carlo Tree Search Methods,' IEEE Trans. on Computational Intelligence and AI in Games, Vol. 4, pp. 1-43.

[BRT18] Belkin, M., Rakhlin, A., and Tsybakov, A. B., 2018. 'Does Data Interpolation Contradict Statistical Optimality?' arXiv:1806.09471.

[BSA83] Barto, A. G., Sutton, R. S., and Anderson, C. W., 1983. 'Neuronlike Elements that Can Solve Di ffi cult Learning Control Problems,' IEEE Trans. on Systems, Man, and Cybernetics, Vol. 13, pp. 835-846.

[BSG09] Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., and Lee, M., 2009. 'Natural ActorCritic Algorithms,' Automatica, Vol. 45, pp. 2471-2482.

[BTW97] Bertsekas, D. P., Tsitsiklis, J. N., and Wu, C., 1997. 'Rollout Algorithms for Combinatorial Optimization,' Heuristics, Vol. 3, pp. 245-262.

[BWL19] Beuchat, P. N., Warrington, J., and Lygeros, J., 2019. 'Accelerated PointWise Maximum Approach to Approximate Dynamic Programming,' arXiv:1901.03619; also IEEE Trans. on Automatic Control, Vol. 67, 2021, pp. 251-266.

[BYB94] Bradtke, S. J., Ydstie, B. E., and Barto, A. G., 1994. 'Adaptive Linear Quadratic Control Using Policy Iteration,' Proc. IEEE American Control Conference, Vol. 3, pp. 3475-3479.

[BaB01] Baxter, J., and Bartlett, P. L., 2001. 'Infinite-Horizon Policy-Gradient Estimation,' Journal of Artificial Intelligence Research, Vol. 15, pp. 319-350.

[BaF88] Bar-Shalom, Y., and Fortman, T. E., 1988. Tracking and Data Association, Academic Press, NY.

[BaL19] Banjac, G., and Lygeros, J., 2019. 'A Data-Driven Policy Iteration Scheme Based on Linear Programming,' Proc. 2019 IEEE CDC, pp. 816-821.

[BaP12] Bauso, D., and Pesenti, R., 2012. 'Team Theory and Person-by-Person Optimization with Binary Decisions,' SIAM Journal on Control and Optimization, Vol. 50, pp. 3011-3028.

[Bai93] Baird, L. C., 1993. 'Advantage Updating,' Report WL-TR-93-1146, Wright Patterson AFB, OH.

[Bai94] Baird, L. C., 1994. 'Reinforcement Learning in Continuous Time: Advantage Updating,' International Conf. on Neural Networks, Orlando, Fla.

[Bar90] Bar-Shalom, Y., 1990. Multitarget-Multisensor Tracking: Advanced Applications, Artech House, Norwood, MA.

[BeA24] Berberich, J., and Allgower, F., 2024. 'An Overview of Systems-Theoretic Guarantees in Data-Driven Model Predictive Control,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 8.

[BeC89] Bertsekas, D. P., and Casta˜ non, D. A., 1989. 'The Auction Algorithm for Transportation Problems,' Annals of Operations Research, Vol. 20, pp. 67-96.

[BeC99] Bertsekas, D. P., and Casta˜ non, D. A., 1999. 'Rollout Algorithms for Stochastic Scheduling Problems,' Heuristics, Vol. 5, pp. 89-108.

[BeC02] Ben-Gal, I., and Caramanis, M., 2002. 'Sequential DOE via Dynamic Programming,' IIE Transactions, Vol. 34, pp. 1087-1100.

[BeC08] Besse, C., and Chaib-draa, B., 2008. 'Parallel Rollout for Online Solution of DEC-POMDPs,' Proc. of 21st International FLAIRS Conference, pp. 619-624.

[BeK65] Bellman, R., and Kalaba, R. E., 1965. Quasilinearization and Nonlinear BoundaryValue Problems, Elsevier, N.Y.

[BeL14] Beyme, S., and Leung, C., 2014. 'Rollout Algorithm for Target Search in a Wireless Sensor Network,' 80th Vehicular Technology Conference (VTC2014), IEEE, pp. 1-5.

[BeI96] Bertsekas, D. P., and Io ff e, S., 1996. 'Temporal Di ff erences-Based Policy Iteration and Applications in Neuro-Dynamic Programming,' Lab. for Info. and Decision Systems Report LIDS-P-2349, Massachusetts Institute of Technology.

[BeP03] Bertsimas, D., and Popescu, I., 2003. 'Revenue Management in a Dynamic Network Environment,' Transportation Science, Vol. 37, pp. 257-277.

[BeR71a] Bertsekas, D. P., and Rhodes, I. B., 1971. 'On the Minimax Reachability of Target Sets and Target Tubes,' Automatica, Vol. 7, pp. 233-247.

[BeR71b] Bertsekas, D. P., and Rhodes, I. B., 1971. 'Recursive State Estimation for a Set-Membership Description of the Uncertainty,' IEEE Trans. Automatic Control, Vol. AC-16, pp. 117-128.

[BeR73] Bertsekas, D. P., and Rhodes, I. B., 1973. 'Su ffi ciently Informative Functions and the Minimax Feedback Control of Uncertain Dynamic Systems,' IEEE Trans. Automatic Control, Vol. AC-18, pp. 117-124.

[BeS78] Bertsekas, D. P., and Shreve, S. E., 1978. Stochastic Optimal Control: The Discrete Time Case, Academic Press, NY; republished by Athena Scientific, Belmont, MA, 1996 (can be downloaded in from the author's website).

[BeS18] Bertazzi, L., and Secomandi, N., 2018. 'Faster Rollout Search for the Vehicle Routing Problem with Stochastic Demands and Restocking,' European J. of Operational Research, Vol. 270, pp.487-497.

[BeT89] Bertsekas, D. P., and Tsitsiklis, J. N., 1989. Parallel and Distributed Computation: Numerical Methods, Prentice-Hall, Englewood Cli ff s, NJ.; republished by Athena Scientific, Belmont, MA, 1997 (can be downloaded from the author's website).

[BeT91] Bertsekas, D. P., and Tsitsiklis, J. N., 1991. 'An Analysis of Stochastic Shortest Path Problems,' Math. Operations Res., Vol. 16, pp. 580-595.

[BeT96] Bertsekas, D. P., and Tsitsiklis, J. N., 1996. Neuro-Dynamic Programming, Athena Scientific, Belmont, MA.

[BeT97] Bertsimas, D., and Tsitsiklis, J. N., 1997. Introduction to Linear Optimization, Athena Scientific, Belmont, MA.

[BeT00] Bertsekas, D. P., and Tsitsiklis, J. N., 2000. 'Gradient Convergence of Gradient Methods with Errors,' SIAM J. on Optimization, Vol. 36, pp. 627-642.

[BeT08] Bertsekas, D. P., and Tsitsiklis, J. N., 2008. Introduction to Probability, 2nd Edition, Athena Scientific, Belmont, MA.

[BeY07] Bertsekas, D. P., and Yu, H., 2007. 'Solution of Large Systems of Equations Using Approximate Dynamic Programming Methods,' Lab. for Information and Decision Systems Report LIDS-P-2754, MIT.

[BeY09] Bertsekas, D. P., and Yu, H., 2009. 'Projected Equation Methods for Approxi-

mate Solution of Large Linear Systems,' J. of Computational and Applied Math., Vol. 227, pp. 27-50.

[BeY10] Bertsekas, D. P., and Yu, H., 2010. 'Distributed Asynchronous Policy Iteration in Dynamic Programming,' Proc. of Allerton Conf. on Communication, Control and Computing, Allerton Park, Ill, pp. 1368-1374.

[BeY12] Bertsekas, D. P., and Yu, H., 2012. 'Q-Learning and Enhanced Policy Iteration in Discounted Dynamic Programming,' Math. of Operations Research, Vol. 37, pp. 6694.

[BeY16] Bertsekas, D. P., and Yu, H., 2016. 'Stochastic Shortest Path Problems Under Weak Conditions,' Lab. for Information and Decision Systems Report LIDS-2909, MIT.

[Bel56] Bellman, R., 1956. 'A Problem in the Sequential Design of Experiments,' Sankhya: The Indian Journal of Statistics, Vol. 16, pp. 221-229.

[Bel57] Bellman, R., 1957. Dynamic Programming, Princeton University Press, Princeton, NJ.

[Bel84] Bellman, R., 1984. Eye of the Hurricane, World Scientific Publishing, Singapore.

[Bel87] Bellman, R., 1987. Introduction to the Mathematical Theory of Control Processes, Academic Press, Vols. I and II, New York, NY.

[Ben09] Bengio, Y., 2009. 'Learning Deep Architectures for AI,' Foundations and Trends in Machine Learning, Vol. 2, pp. 1-127.

[Ber71] Bertsekas, D. P., 1971. 'Control of Uncertain Systems With a Set-Membership Description of the Uncertainty,' Ph.D. Dissertation, Massachusetts Institute of Technology, Cambridge, MA (can be downloaded from the author's website).

[Ber72a] Bertsekas, D. P., 1972. 'Infinite Time Reachability of State Space Regions by Using Feedback Control,' IEEE Trans. Automatic Control, Vol. AC-17, pp. 604-613.

[Ber72b] Bertsekas, D. P., 1972. 'On the Solution of Some Minimax Control Problems,' Proc. 1972 IEEE Decision and Control Conf., New Orleans, LA.

[Ber73] Bertsekas, D. P., 1973. 'Linear Convex Stochastic Control Problems over an Infinite Horizon,' IEEE Trans. Automatic Control, Vol. AC-18, pp. 314-315.

[Ber75] Bertsekas, D. P., 1975. 'Nondi ff erentiable Optimization Via Approximation,' Math. Programming Study 3, Balinski, M., and Wolfe, P., (Eds.), North-Holland, Amsterdam, pp. 1-25.

[Ber76] Bertsekas, D. P., 1976. Dynamic Programming and Stochastic Control, Academic Press, NY (can be downloaded fro the author's website).

[Ber77] Bertsekas, D. P., 1977. 'Approximation Procedures Based on the Method of Multipliers,' J. Opt. Th. and Appl., Vol. 23, pp. 487-510.

[Ber79] Bertsekas, D. P., 1979. 'A Distributed Algorithm for the Assignment Problem,' Lab. for Information and Decision Systems Report, MIT, May 1979.

[Ber82a] Bertsekas, D. P., 1982. 'Distributed Dynamic Programming,' IEEE Trans. Automatic Control, Vol. AC-27, pp. 610-616.

[Ber82b] Bertsekas, D. P., 1982. Constrained Optimization and Lagrange Multiplier Methods, Academic Press, N. Y.; republished by Athena Scientific, Belmont, MA, 1997.

[Ber82c] Bertsekas, D. P., 1982. 'Projected Newton Methods for Optimization Problems with Simple Constraints,' SIAM J. on Control and Optimization, Vol. 20, pp. 221-246.

[Ber83] Bertsekas, D. P., 1983. 'Asynchronous Distributed Computation of Fixed Points,' Math. Programming, Vol. 27, pp. 107-120.

[Ber91] Bertsekas, D. P., 1991. Linear Network Optimization: Algorithms and Codes, MIT Press, Cambridge, MA (can be downloaded from the author's website).

[Ber96] Bertsekas, D. P., 1996. 'Incremental Least Squares Methods and the Extended Kalman Filter,' SIAM J. on Optimization, Vol. 6, pp. 807-822.

[Ber97a] Bertsekas, D. P., 1997. 'A New Class of Incremental Gradient Methods for Least Squares Problems,' SIAM J. on Optimization, Vol. 7, pp. 913-926.

[Ber97b] Bertsekas, D. P., 1997. 'Di ff erential Training of Rollout Policies,' Proc. of the 35th Allerton Conference on Communication, Control, and Computing, Allerton Park, Ill.

[Ber98] Bertsekas, D. P., 1998. Network Optimization: Continuous and Discrete Models, Athena Scientific, Belmont, MA (can be downloaded from the author's website).

[Ber05a] Bertsekas, D. P., 2005. 'Dynamic Programming and Suboptimal Control: A Survey from ADP to MPC,' European J. of Control, Vol. 11, pp. 310-334.

[Ber05b] Bertsekas, D. P., 2005. 'Rollout Algorithms for Constrained Dynamic Programming,' Lab. for Information and Decision Systems Report LIDS-P-2646, MIT.

[Ber07] Bertsekas, D. P., 2007. 'Separable Dynamic Programming and Approximate Decomposition Methods,' IEEE Trans. on Aut. Control, Vol. 52, pp. 911-916.

[Ber10a] Bertsekas, D. P., 2010. 'Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey,' Lab. for Information and Decision Systems Report LIDS-P-2848, MIT; a condensed version with the same title appears in Optimization for Machine Learning, by S. Sra, S. Nowozin, and S. J. Wright, (eds.), MIT Press, Cambridge, MA, 2012, pp. 85-119.

[Ber10b] Bertsekas, D. P., 2010. 'Williams-Baird Counterexample for Q-Factor Asynchronous Policy Iteration,'

http://web.mit.edu/dimitrib/www/Williams-Baird Counterexample.pdf.

[Ber10c] Bertsekas, D. P., 2010. 'Pathologies of Temporal Di ff erence Methods in Approximate Dynamic Programming,' Proc. 2010 IEEE Conference on Decision and Control, Atlanta, GA, Dec. 2010.

[Ber10d] Bertsekas, D. P., 2010. 'Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey,' Lab. for Information and Decision Systems Report LIDS-P-2848, MIT; this is an extended version of a paper in the edited volume Optimization for Machine Learning, by S. Sra, S. Nowozin, and S. J. Wright, MIT Press, Cambridge, MA, 2012, pp. 85-119.

[Ber11a] Bertsekas, D. P., 2011. 'Incremental Proximal Methods for Large Scale Convex Optimization,' Math. Programming, Vol. 129, pp. 163-195.

[Ber11b] Bertsekas, D. P., 2011. 'Approximate Policy Iteration: A Survey and Some New Methods,' J. of Control Theory and Applications, Vol. 9, pp. 310-335.

[Ber11c] Bertsekas, D. P., 2011. 'Temporal Di ff erence Methods for General Projected Equations,' IEEE Trans. on Automatic Control, Vol. 56, pp. 2128-2139.

[Ber12] Bertsekas, D. P., 2012. Dynamic Programming and Optimal Control, Vol. II, 4th Edition, Athena Scientific, Belmont, MA.

[Ber13a] Bertsekas, D. P., 2013. 'Rollout Algorithms for Discrete Optimization: A Survey,' Handbook of Combinatorial Optimization, Springer.

[Ber13b] Bertsekas, D. P., 2013. ' λ -Policy Iteration: A Review and a New Implementation,' in Reinforcement Learning and Approximate Dynamic Programming for Feedback Control, by F. Lewis and D. Liu (eds.), IEEE Press, Piscataway, NJ., pp. 381-409.

[Ber15a] Bertsekas, D. P., 2015. Convex Optimization Algorithms, Athena Scientific, Belmont, MA.

[Ber15b] Bertsekas, D. P., 2015. 'Incremental Aggregated Proximal and Augmented Lagrangian Algorithms,' Lab. for Information and Decision Systems Report LIDS-P3176, MIT; arXiv:1507.1365936.

[Ber16] Bertsekas, D. P., 2016. Nonlinear Programming, 3rd Edition, Athena Scientific, Belmont, MA.

[Ber17a] Bertsekas, D. P., 2017. Dynamic Programming and Optimal Control, Vol. I, 4th Edition, Athena Scientific, Belmont, MA.

[Ber17b] Bertsekas, D. P., 2017. 'Value and Policy Iteration in Deterministic Optimal Control and Adaptive Dynamic Programming,' IEEE Transactions on Neural Networks and Learning Systems, Vol. 28, pp. 500-509.

[Ber18a] Bertsekas, D. P., 2018. 'Feature-Based Aggregation and Deep Reinforcement Learning: A Survey and Some New Implementations,' Lab. for Information and Decision Systems Report, MIT; arXiv:1804.04577; IEEE/CAA Journal of Automatica Sinica, Vol. 6, 2018, pp. 1-31.

[Ber18b] Bertsekas, D. P., 2018. 'Biased Aggregation, Rollout, and Enhanced Policy Improvement for Reinforcement Learning,' Lab. for Information and Decision Systems Report, MIT; arXiv:1910.02426.

[Ber18c] Bertsekas, D. P., 2018. 'Proximal Algorithms and Temporal Di ff erence Methods for Solving Fixed Point Problems,' Computational Optim. Appl., Vol. 70, pp. 709-736.

[Ber19a] Bertsekas, D. P., 2019. Reinforcement Learning and Optimal Control, Athena Scientific, Belmont, MA.

[Ber19b] Bertsekas, D. P., 2019. 'Robust Shortest Path Planning and Semicontractive Dynamic Programming,' Naval Research Logistics, Vol. 66, pp. 15-37.

[Ber19c] Bertsekas, D. P., 2019. 'Multiagent Rollout Algorithms and Reinforcement Learning,' arXiv:1910.00120.

[Ber19d] Bertsekas, D. P., 2019. 'Constrained Multiagent Rollout and Multidimensional Assignment with the Auction Algorithm,' arxiv:2002.07407.

[Ber20a] Bertsekas, D. P., 2020. Rollout, Policy Iteration, and Distributed Reinforcement Learning, Athena Scientific, Belmont, MA.

[Ber20b] Bertsekas, D. P., 2020. 'Multiagent Value Iteration Algorithms in Dynamic Programming and Reinforcement Learning,' arxiv.org/abs/2005.01627; also Results in Control and Optimization Journal, Vol. 1, 2020.

[Ber21a] Bertsekas, D. P., 2021. 'Multiagent Reinforcement Learning: Rollout and Policy Iteration,' IEEE/CAA Journal of Automatica Sinica, Vol. 8, pp. 249-271.

[Ber21b] Bertsekas, D. P., 2021. 'Distributed Asynchronous Policy Iteration for Sequential Zero-Sum Games and Minimax Control,' arXiv:2107.10406

[Ber22a] Bertsekas, D. P., 2022. Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control, Athena Scientific, Belmont, MA.

[Ber22b] Bertsekas, D. P., 2022. Abstract Dynamic Programming, 3rd Edition, Athena Scientific, Belmont, MA (can be downloaded from the author's website).

[Ber22c] Bertsekas, D. P., 2022. 'Newton's Method for Reinforcement Learning and Model Predictive Control,' Results in Control and Optimization, Vol. 7, pp. 100-121.

[Ber22d] Bertsekas, D. P., 2022. 'Rollout Algorithms and Approximate Dynamic Programming for Bayesian Optimization and Sequential Estimation,' arXiv:2212.07998.

[Ber24] Bertsekas, D. P., 2024. 'Model Predictive Control, and Reinforcement Learning: AUnified Framework Based on Dynamic Programming,' arXiv preprint arXiv:2406.00592; Proc. IFAC NMPC.

[Bet10] Bethke, B. M., 2010. Kernel-Based Approximate Dynamic Programming Using Bellman Residual Elimination, Ph.D. Thesis, MIT.

[BhR24] Bhandari, J., and Russo, D., 2024. 'Global Optimality Guarantees for Policy Gradient Methods,' Operations Research, Vol. 72, pp. 1906-1927.

[Bha23] Bhatnagar, S., 2023. 'The Reinforce Policy Gradient Algorithm Revisited,' in 2023 Ninth Indian Control Conference, pp. 177-177.

[BiB24] Bishop, C. M, and Bishop, H., 2024. Deep Learning: Foundations and Concepts, Springer, New York, NY.

[BiL97] Birge, J. R., and Louveaux, 1997. Introduction to Stochastic Programming, Springer, New York, NY.

[Bia16] Bianchi, P., 2016. 'Ergodic Convergence of a Stochastic Proximal Point Algorithm,' SIAM J. on Optimization, Vol. 26, pp. 2235-2260.

[Bis95] Bishop, C. M, 1995. Neural Networks for Pattern Recognition, Oxford University Press, NY.

[Bis06] Bishop, C. M, 2006. Pattern Recognition and Machine Learning, Springer, NY.

[Bit91] Bittanti, S., 1991. 'Count Riccati and the Early Days of the Riccati Equation,' in The Riccati Equation (pp. 1-10), Springer.

[BlG54] Blackwell, D., and Girshick, M. A., 1954. Theory of Games and Statistical Decisions, Wiley, NY.

[BlM08] Blanchini, F., and Miani, S., 2008. Set-Theoretic Methods in Control, Birkhauser, Boston.

[Bla65] Blackwell, D., 1965. 'Discounted Dynamic Programming,' The Annals of Mathematical Statistics, Vol. 36, pp. 226-235.

[Bla67] Blackwell, D., 1967. 'Positive Dynamic Programming,' in Proc. of the 5th Berkeley Symp. on Mathematical Statistics and Probability, Vol. 1, pp. 415-418.

[Bla86] Blackman, S. S., 1986. Multi-Target Tracking with Radar Applications, Artech House, Dehdam, MA.

[Bla99] Blanchini, F., 1999. 'Set Invariance in Control - A Survey,' Automatica, Vol. 35, pp. 1747-1768.

[BoV79] Borkar, V., and Varaiya, P., 1979. 'Adaptive Control of Markov Chains, I: Finite Parameter Set,' IEEE Trans. on Automatic Control, Vol. 24, pp. 953-957.

[Bod20] Bodson, M., 2020. Adaptive Estimation and Control, Independently Published.

[Bor08] Borkar, V. S., 2008. Stochastic Approximation: A Dynamical Systems Viewpoint, Cambridge Univ. Press.

[BrH75] Bryson, A., and Ho, Y. C., 1975. Applied Optimal Control: Optimization, Estimation, and Control, (revised edition), Taylor and Francis, Levittown, Penn.