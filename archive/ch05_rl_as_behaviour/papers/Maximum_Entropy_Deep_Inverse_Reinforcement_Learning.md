Feature

Representation

Hidden

## Maximum Entropy Deep Inverse Reinforcement Learning

Representations

Markus Wulfmeier Peter Ondr´ uˇ ska Ingmar Posner

Mobile Robotics Group, Department of Engineering Science, University of Oxford

## Abstract

This paper presents a general framework for exploiting the representational capacity of neural networks to approximate complex, nonlinear reward functions in the context of solving the inverse reinforcement learning (IRL) problem. We show in this context that the Maximum Entropy paradigm for IRL lends itself naturally to the efficient training of deep architectures. At test time, the approach leads to a computational complexity independent of the number of demonstrations, which makes it especially well-suited for applications in life-long learning scenarios. Our approach achieves performance commensurate to the state-of-the-art on existing benchmarks while exceeding on an alternative benchmark based on highly varying reward structures.Finally, we extend the basic architecture - which is equivalent to a simplified subclass of Fully Convolutional Neural Networks (FCNNs) with width one - to include larger convolutions in order to eliminate dependency on precomputed spatial features and work on raw input representations.

## 1. Introduction

Recent successes in machine learning, vision and robotics have lead to widespread expectations that machines will increasingly succeed in applications of real value to the public domain. A central tenet of any vision delivering on this promise revolves around learning from user interactions. Inverse reinforcement learning (IRL) is playing a pivotal role in these developments and commonly finds applications in robotics (Argall et al., 2009) where it allows robot to learn complex behaviour from human demonstrations and also in fields of cognition (Baker et al., 2009) and preference learning (Ziebart et al., 2008) where it serves

Figure 1: Fully Convolutional Neural Network for reward approximation in the IRL setting. The network serves to model the relationship between input features and final reward map.

<!-- image -->

as a tool to better understand human decisions or medicine (Asoh et al., 2013) to predict patient response to treatment. The objective of inverse reinforcement learning (IRL) is to infer the underlying reward structure guiding an agent's behaviour based on observations as well as a model of the environment. This may be done either to learn the reward structure for modelling purposes or to provide a method to allow the agent to imitate a demonstrator's specific behaviour (Ramachandran &amp; Amir, 2007). While for small problems the complete set of rewards can be learned explicitly, many problems of realistic size require the application of generalisable function approximations.

Much of the prior art in this domain relies on parametrisation of the reward function based on pre-determined features. In addition to better generalisation performance than direct state-to-reward mapping, this approach enables the transfer of learned reward functions between different scenarios with the same feature representation. A number of early works from (Ziebart et al., 2008), (Abbeel &amp; Ng, 2004), (Lopes et al., 2009) and (Ratliff et al., 2006), ex-

MARKUS@ROBOTS.OX.AC.UK ONDRUSKA@ROBOTS.OX.AC.UK INGMAR@ROBOTS.OX.AC.UK

press the reward function as a weighted linear combination of hand selected features. To overcome the inherent limitations of linear models, (Choi &amp; Kim, 2013) and (Levine et al., 2010) extend this approach to a limited set of nonlinear rewards by learning a set of composites of logical conjunctions of atomic features. Non-parametric methods such as Gaussian Processes (GPs) have also been employed to cater for potentially complex, nonlinear reward functions (Levine et al., 2011). While in principle this extends the IRL paradigm to the flexibility of nonlinear reward approximation, the use of a kernel machine makes this approach scale badly with higher numbers of training data and prone to requiring a large number of reward samples in order to approximate highly varying reward functions (Bengio et al., 2007). Even sparse GP approximations as used in (Levine et al., 2011) lead to a query complexity time in dependency of the size of the active set or the number of experienced state-reward pairs. Situations with increasingly complex reward function leading to higher requirements regarding the number of inducing points can quickly render this nonparametric approach computationally impracticable. Furthermore, in comparison to (Babes et al., 2011), we focus on a singular expert in what finally leads to an an end-to-end learning scenario in section 4.3 from raw input to reward without compression or preprocessing on the input representation. To our knowledge the only other work considering the use of deep networks is given by (Levine et al., 2015), who focus on directly approximating policies with neural networks but shortly refer to the possibility of extension for cost function learning with neural networks.

In contrast to prior art, we explore the use of neural networks to approximate the reward function. Neural Networks already achieve state-of-the-art performance across a variety of domains such as computer vision, natural language processing, speech recognition (Bengio et al., 2012) and reinforcement learning (Mnih et al., 2013). Their application in IRL suggests itself due to their compact representation of highly nonlinear functions through the composition and reuse of the results of many nonlinearities in the layered structure (Bengio et al., 2007). In addition, NNs provide favourable computational complexity ( O (1) ) at query time with respect to observed demonstrations, which provides for scaling to problems with large state spaces and complex reward structures - circumstances which might render the application of existing prior methods intractable or ineffective. With the approach represented in Figure 1, a state's reward can be determined either solely based on its own feature representation or - in using wider convolutional layers - analysed in combination with its spatial context. The applied architectures are Fully Convolutional Neural Networks, which - by skipping the fully connected final layers common in classification tasks - preserve spatial information and can create an output of the same spa- tial dimension and size as the input. Recent examples for the application of FCNNs focus on dense prediction: including pixel-wise semantic segmentation by (Long et al., 2014), sliding window detection and prediction of object boundaries (Sermanet et al., 2013), depth estimation with single monocular images (Liu et al., 2015) and human pose estimation in monocular images (Tompson et al., 2014).

Our principal contribution is a framework for Maximum Entropy Deep Inverse Reinforcement Learning (DeepIRL) based on the Maximum Entropy paradigm for IRL (Ziebart et al., 2008), which lends itself naturally for training deep architectures by leading to an objective that is - without approximations - fully differentiable with respect to the network weights. Furthermore, we demonstrate performance commensurate to state-of-the-art methods on a publicly available benchmark, while outperforming the state-of-theart on a new benchmark where the true underlying reward has complex interacting structure over the feature representation. In addition, we emphasise the flexibility of the approach and eliminate the requirement of preprocessing and precomputed features by applying wider convolutional layers to learn spatial features of relevance to the IRL task. This enables the application without manually crafted feature design as long as the state space is constrained to a regularly gridded representation allowing for convolutions.

We argue that these properties are important for practical large-scale applications of IRL as can be seen in life-long learning approaches with often complex reward functions and increasing scale of demonstrations requiring high capacity models and fast computational speeds.

## 2. Inverse Reinforcement Learning

This section presents a brief overview of IRL. Let a Markov Decision Process (MDP) be defined as M = {S , A , T , r } , where S denotes the state space, A denotes the set of possible actions, T denotes the transition model and r denotes the reward structure. Given an MDP, an optimal policy π ∗ is one which, when adhered to, maximizes the expected cumulative reward. Furthermore, an additional factor γ ∈ [0 , 1] may be considered in order to discount future rewards.

IRL considers the case where a MDP specification is available but the reward structure is unknown. Instead, a set of expert demonstrations D = { ς 1 , ς 2 , ..., ς N } is provided which are sampled from a user policy π , i.e. provided by a demonstrator. Each demonstration consists of a set of state-action pairs such that ς i = { ( s 0 , a 0 ) , ( s 1 , a 1 ) , ..., ( s K , a K ) } . The goal of IRL is to uncover the hidden reward r from the demonstrations.

A number of approaches have been proposed to tackle the IRL problem (see, for example, (Abbeel &amp; Ng, 2004),

(Neu &amp; Szepesv´ ari, 2012), (Ratliff et al., 2006), (Syed &amp; Schapire, 2007)). An increasingly popular formulation is Maximum Entropy IRL (Ziebart et al., 2008), which was used to effectively model large-scale user driving behaviour. In this formulation the probability of user preference for any given trajectory between specified start and goal states is proportional to the exponential of the reward along the path

<!-- formula-not-decoded -->

As shown in Ziebart's work, principal benefits of the Maximum Entropy paradigm include the ability to handle expert suboptimality as well as stochasticity by operating on the distribution over possible trajectories. Moreover, the Maximum Entropy based objective function given in Equation 10 enables backpropagation of the objective gradients to the network's weights. The training procedure is then straightforwardly framed as an optimisation task computable e.g. via conjugate gradient or stochastic gradient descent.

## 2.1. Approximating the Reward Structure

Due to the dimensionality and size of the state space in many real world applications, the reward structure can not be observed explicitly for every state. In these cases state rewards are not modelled directly per state, but the reward structure is restricted by imposing that states with similar features, x , should have similar rewards. To this end, function approximation is used in order to regress the feature representation onto a real valued reward using a mapping g : R N → R , with N being the dimensionality of the feature space such that

<!-- formula-not-decoded -->

A feature representation, f , is usually hand-crafted based on preprocessing such as segmentation and manually defined distance metrics, but can be learned based on the proposed framework - as shown in section 4.3. Furthermore, the application of feature based function approximation enables easier generalisation and transfer of models.

The choice of model used for function approximation has a dramatic impact on the ability of the algorithm to capture relationship between the state feature vector f and user preference. Commonly, the mapping from state to reward is simply a weighted linear combination of feature values

<!-- formula-not-decoded -->

This choice, while appropriate in some scenarios, is suboptimal if the true reward can not be accurately approximated by a linear model. In order to alleviate this limitation

(Choi &amp; Kim, 2013) extend the linear model by introducing a mapping Φ : R N →{ 0 , 1 } N such that

<!-- formula-not-decoded -->

Here Φ denotes a set of composite features which are jointly learned as part of the objective function. These composites are assumed to be the logical conjunctions of the predefined, atomic features f . Due to the nature of the features used the representational power of this approach is limited to the family of piecewise constant functions.

In contrast, (Levine et al., 2011) employ a Gaussian Processes (GP) framework to capture the potentially unbounded complexity of any underlying reward structure. The set of expert demonstrations D is used in this context to identify an active set of GP support points, X u , and associated rewards u . The mean function is then used to represent the individual reward at a state described by f

<!-- formula-not-decoded -->

Here K f,u denotes the covariance of the reward at f with the active set reward values u located at X u and K u,u denotes the covariance matrix of the rewards in the active set computed via a covariance function k θ ( f i , f j ) with hyperparameters θ .

Nevertheless, a significant drawback of the GPIRL approach is a computational complexity proportional to the number of demonstrations and the size of the active set of inducing points, which in turn depends on the reward complexity. While the modelling of complex, nonlinear reward structures in problems with large state spaces is theoretically feasible for the GPIRL approach, the cardinality of the active set will quickly become unwieldy, putting GPIRL at a significant computational disadvantage or, worse, rendering it entirely intractable. These shortcomings are remedied when using deep parametric architectures for reward function approximation while keeping the accuracy of nonlinear function approximation, as outlined in the next section.

## 3. Reward Function Approximation with Deep Architectures

We argue that IRL algorithms scalable to MDPs with large feature spaces require models, which are able to efficiently represent complex, nonlinear reward structures. In this context, deep architectures are a natural choice as they explicitly exploit the depth-breadth trade-off (Bengio et al., 2007) and increase representational capacity by reusing the computations of earlier nodes in the following layers.

For the remainder of the paper, we consider a network architecture which accepts as input state features x , maps

S1

S5

S2

S4

S6

Network r1

Figure 2: Schema for Neural Network based reward function approximation based on the feature representation of MDP states

<!-- image -->

these to state reward r and is governed by the network parameters θ 1 , 2 ,..n . In the context of Section 2.1, the state reward is therefore obtained as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

While many choices exist for the individual building blocks of a deep architecture, it has been shown that a sufficiently large NN with as little as two layers and sigmoid activation functions can represent any binary function (Hassoun, 1995) or any piecewise-linear function (Hornik et al., 1989) and can therefore be regarded as a universal approximator . While this holds true in theory, it can be far more computationally practicable to extend the depth of the network structure and reduce the number of required computations in doing so (Bengio, 2009).

Importantly, in applying backpropagation, NNs also lend themselves naturally to training in the maximum entropy IRL framework and the network structure can be adapted to suit individual tasks without complicating or even invalidating the main IRL learning mechanism. In the DeepIRL framework proposed here the full range of architecture choices thus becomes available. Different problem domains can utilise different network architectures as e.g. convolutional layers can remove the dependency on handcrafted spatial features. Furthermore, it is straightforward to show that the linear maximum entropy IRL approach proposed in (Ziebart et al., 2008) can be seen as a simplification of the more general deep approach and can be created by applying the rules of back-propagation to a network with a single linear output connected to all inputs with zero bias term.

While the common NN architectures for whole-image classification regress to fixed size outputs, the applied FCNNs result in an output with equivalent spatial dimensionality and by padding data correspondingly we realise reward maps of the same size as our input. It is to note here that padding is not the only possibility and deconvolutions as applied by (Long et al., 2014) can also transform and reshape to create equally sized model output.

## 3.1. Training Procedure

The task of solving the IRL problem can be framed in the context of Bayesian inference as MAP estimation, maximizing the joint posterior distribution of observing expert demonstrations, D , under a given reward structure and of the model parameters θ .

<!-- formula-not-decoded -->

This joint log likelihood is differentiable with respect to the parameters θ of a linear reward model, which allows the application of gradient descent methods (Snyman, 2005). We extend this benefit with the adaptation of Maximum Entropy for neural networks as presented in L D of Equation 10 by separating into the gradient of the loss with respect to the rewards r and the gradient of the reward with respect to the network's weights obtained via backpropagation.

The complete gradient is given by the sum of the gradients with respect to θ of the data term L D and a weight decay term as model regulariser L θ

<!-- formula-not-decoded -->

The earlier mentioned separation of derivatives in the gradient of the data term is shown in equation 10

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r = g ( f, θ )) . As shown in (Ziebart et al., 2008), the gradient of the expert demonstration term L D with respect to the model parameters of a linear function is equal to the difference in feature counts along the trajectories. For higher level models this gradient can be split into the derivative with respect to the reward r and the derivative of the reward with respect to the model parameters which in case of a neural network is obtained via backpropagation. The derivative of the Maximum Entropy objective with respect to the reward equals the difference in state visitation counts between solutions given by the expert demonstrations and the expected visitation counts for the learned systems trajectory distribution in 12.

<!-- formula-not-decoded -->

## Algorithm 1 Maximum Entropy Deep IRL

Input:

µ a D , f, S, A, T, γ

Output:

optimal weights θ ∗

- 1: θ 1 = initialise weights()

## Iterative model refinement

- 2: for n = 1 : N do
- 3: r n = nn forward( f, θ n )

## Solution of MDP with current reward

- 4: π n = approx value iteration( r n , S, A, T, γ )
- 5: E [ µ n ] = propagate policy( π n , S, A, T )

## Determine Maximum Entropy loss and gradients

- 6: L n D = log( π n ) × µ a D
- 7: ∂ L n D ∂r n = µ D -E [ µ n ]

## Compute network gradients

- 8: ∂ L n D ∂θ n D = nn backprop( f, θ n , ∂ L n D ∂r n )
- 9: θ n +1 = update weights( θ n , ∂ L n D ∂θ n D )
- 10: end for

Computation of E [ µ ] usually involves summation over exponentially many possible trajectories. A more effective algorithm based on dynamic programming which computes this quantity in polynomial-time can be found in (Ziebart et al., 2008; Kitani et al., 2012). Subsequently, the effective computation of the gradient ∂ L D ∂θ involves first computing the difference in visitation counts using this algorithm and then passing this as an error signal through the network using back-propagation.

The complete proposed method is described by Algorithm 1, with the loss and gradient derivation in lines 6 and 7 given by the linear Maximum Entropy formulation. The expert's state action frequencies µ a D , which are needed for the calculation of the loss are summed over the actions to compute the expert state frequencies µ D = A ∑ µ a D .

<!-- formula-not-decoded -->

Lines 4 and 5 are explained in detail in the algorithms 2 and 3 respectively, and are adapted from (Kitani et al., 2012). Algorithm 2 determines the policy given the current reward model via iterative update of the state-action value function, while algorithm 3 determines the expected state visiting frequencies by probabilistically traversing the MDP

## Algorithm 2 Approximate Value Iteration

- 1: V ( s ) = -∞
- 2: repeat
- 3: V t = V ; V ( s goal ) = 0
- 4: Q ( s, a ) = r ( s, a ) + E T ( s,a,s ′ ) [ V ( s ′ )]
- 5: V = softmax a Q i ( s, a )
- 6: until max s ( V ( s ) -V t ( s )) &lt; glyph[epsilon1]
- 7: π ( a | s ) = e Q ( s,a ) -V ( s )

## Algorithm 3 Policy Propagation

- 1: E 1 [ µ ( s start )] = 1
- 2: for i = 1 : N do
- 3: E i [ µ ( s goal )] = 0
- 4: E i +1 [ µ ( s )] = ∑ s ′ ,a T ( s, a, s ′ ) π ( a | s ′ ) E i [ µ ( s ′ )]
- 5: end for
- 6: E [ µ ( s )] = ∑ i E i [ µ ( s )]

given the current policy. Additional indices representing the iteration of the main algorithm were omitted in these subscripts in favour of readability.

The presented algorithm is applied to train FCNNs based on the loss derivatives for all states at once. As each of the final state-wise rewards is influenced by its corresponding area in the original state space - its receptive field, training with the summed loss over the whole scene is equivalent to a stochastic gradient formulation with all receptive fields addressed in a minibatch. This formulation is computationally more efficient than separate computation per field, since these fields overlap as soon as the width of our convolutional filters exceeds one (Long et al., 2014).

## 4. Experiments

We assess the performance of DeepIRL two benchmark tasks against current state-of-the-art approaches : GPIRL (Levine et al., 2011), NPB-FIRL (Choi &amp; Kim, 2013) and the original MaxEnt (Ziebart et al., 2008) to illustrate the necessity of non-linear function approximation.

All tests are run multiple times on training and transfer scenarios for the different settings, while learning is performed based on synthetically generated stochastic demonstrations based on the optimal policy to evaluate performance on suboptimal example sets. This is achieved by providing a number of demonstrations sampled from the optimal pol-

vard.

ditinnel tacta

A

icy based on the true reward structure, but including 30% of random actions.

In our experiments, we employ a FCNN with two hidden layers and rectified linear units as function approximator between state feature representation and reward. This rather shallow networks structure suffices for the application based on strongly simplified toy benchmarks. However, the whole framework can be utilised for training networks of arbitrary capacity. All experiments except for the spatial feature learning in section 4.3 are based on filters of width one to focus on direct evaluation against the other algorithms, which are in their current form limited to the features of each state for reward approximation. Wider filters as applied for spatial feature learning are used to evaluate the performance on raw inputs without manual feature design. For these benchmarks, we apply AdaGrad (Duchi et al., 2011), an approach for stochastic gradient descent with per parameter adaptive learning rates. Significant parts of the neural network implementation are based on MatConvNet (Vedaldi &amp; Lenc, 2014).

In line with related works, we use expected value difference as principal metric of evaluation. It is a measure of the suboptimality of the learned policy under the true reward. The score represents the difference between the value function obtained for the optimal policy given the true reward structure and the value function obtained for the optimal policy based on the learned reward model. Additionally to the evaluation on each specific training scenario, the trained models are evaluated on a number of randomly generated test environments. The test on these transfer examples serves to analyse each algorithm's ability to generalise to the true reward structure without over-fitting.

## 4.1. Objectworld Benchmark

The Objectworld scenario (Levine et al., 2011) consists of a map of M × M states for M = 32 where possible actions include motions in all four directions as well as staying in place. Two different sets of state features are implemented based on randomly placed colours to evaluate the algorithms. For the continuous set x ∈ R C . Each feature dimension describes the minimum distance to an object of one of C colours. Building on the continuous representation the discrete set includes C × M binary features, where each dimension indicates whether an object of a given colour is closer than a threshold d ∈ { 1 , ..., M } .

The reward is positive for cells which are both within the distance 3 of color 1 and distance 2 of color 2, negative if only within distance 3 of color 1 and zero otherwise. This is illustrated for a small subset of the state space in Figure 3.

In line with common benchmarking procedures, we evalu-

Example Rev

DED

Exam:

Figure 3: Objectworld benchmark. The true reward is displayed by the brightness of each cell and based on the surrounding object configuration. Only a subset of colors influences the reward, while the others serve as distracting features.

<!-- image -->

ated the algorithms with a set number of features and increasing demonstrations. Additionally, the learned reward functions are deployed on randomly generated transfer scenarios to uncover any overfitting to the training data.

While the original MaxEnt is unable to capture the nonlinear reward structure well, both DeepIRL and GPIRL provide significantly better approximations as represented in Figure 4. Evaluation of NPB-FIRL on this benchmark was done in (Choi &amp; Kim, 2013) where it showed a similar level of performance as GPIRL. GPIRL generates a good model already with few data points whereas DeepIRL achieves commensurate performance when increasing the number of available expert demonstrations. The same behaviour is exhibited when using both continuous and discrete state features (Fig. 5). The requirement for more training data will be rendered unimportant in robot applications based on autonomous data acquisition, while enforcing the lower algorithmic complexity as dominant advantage of the parametric approach.

Figure 4: Reward reconstruction sample in Objectworld benchmark provided N = 64 examples and C = 2 colours with continuous features. White - high reward; black - low reward.

<!-- image -->

Additional tests are performed with increased number of

distractor features to evaluate each approach's overfitting tendency. The corresponding figures are left out due to limited space. Both DeepIRL and GPIRL show robustness to distractor variables, though DeepIRL shows minimally bigger signs of overfitting as the number of distractor variables is increased. This is due to the NN's capacity being brought to bear on the increasing noise introduced by the distractors and will be addressed in future work with additional regularisation methods, such as Dropout (Hinton et al., 2012) and ensemble methods.

Figure 5: Objectworld benchmark. From top left to bottom right: expected value difference (EVD) with C = 2 colours and varying number of demonstrations N for training a) and transfer case b) with continuous and subsequently with discrete features in c) &amp; d) ; As the number of demonstrations grows DeepIRL is able to quickly match performance of GPIRL on the task.

<!-- image -->

## 4.2. Binaryworld Benchmark

In order to test the ability of all approaches to successfully approximate more complex reward structures, the Binaryworld benchmark is presented. This test scenario is similar to Objectworld , but in this problem every state is randomly assigned one of two colours (blue or red). The feature vector for each state consequently consists of a binary vector of length 9, encoding the colour of each cell in its 3x3 neighbourhood. The true reward structure for a particular state

Figure 6: Value differences observed in the Binaryworld benchmark for GPIRL, MaxEnt and DeepIRL for the training scenario (left) and the transfer task (right).

<!-- image -->

is fully determined by the number of blue states in its local neighbourhood. It is positive if exactly four out of nine neighbouring states are blue, negative if exactly five are blue and zero otherwise. The main difference compared to the Objectworld scenario is that a single feature value does not carry much weight, but rather that higher-order relationships amongst the features determine the reward.

Since the reward depends on a higher representation for the basic features - that is to say the number of specific features - such case is arguably more challenging than the original Objectworld experiment and a good performance on this benchmark implies the algorithm's ability to learn and capture this complex relationship.

The performance of DeepIRL compared to GPIRL, linear MaxEnt and NPB-FIRL is depicted in Fig. 6. In this increasingly more complex scenario, DeepIRL is able to learn the higher-order dependencies between features, whereas GPIRL struggles as the inherent kernel measure can not correctly relate the reward of different examples with similarity in their state features. GPIRL needs a larger number of demonstrations to achieve good performance and to determine an accurate estimate on the reward for all 2 9 possible feature combinations.

Perhaps surprising is the comparatively low performance of the NPB-FIRL algorithm. This can be explained by the limitations of this framework. The true reward in this scenario can not be efficiently described by the logical conjunctions used. In fact, it would require 2 9 different logical conjunctions, each capturing all possible combinations of features, to accurately model the reward in this framework.

Fig. 7 shows the reconstruction of the reward structures estimated by DeepIRL, MaxEnt and GPIRL. While GPIRL was able to reconstruct the correct reward for some of the states having features it has encountered before it provides inaccurate rewards for states which were never encoun-

Groundtruth

GPIRL

ure 7: Reward reconstruction sample for the Bina- vorld benchmark provided N = 128 demonstrations.

lite - high reward; black - low reward.

₫91

D:

• •

2000

••00

•00000

Figure 7: Reward reconstruction sample for the Binaryworld benchmark provided N = 128 demonstrations. White - high reward; black - low reward.

<!-- image -->

tered. It produces an overall too smooth reward function due to assumptions and priors in the GP approximation. On the other hand, DeepIRL is able to reconstruct it with high accuracy demonstrating the ability to effectively learn the highly-varying structure of the underlying function.

## 4.3. Spatial Feature Learning

While the earlier benchmarks visualise performance compared to current algorithms in the context of precomputed features, the approach can be extended via the use of wider filters to eliminate the requirement of preprocessing or manual design of features. Figure 8 represents the results for both earlier benchmarks, but instead of using the earlier described feature representations, the FCNN builds the reward based on the raw input representation, which for each state only includes the availability of each specific object at that specific state. All spatial information is derived based on the convolutional filters. Based on the simplicity of the benchmarks, we employed a five layer approach with 3x3 convolutional kernels in the first two layers. By increasing the depth of the network and include convolutional filters, we add enough capacity to enable the learning of features as well as their combination into the reward function in the same architecture and process.

Due to the increasing number of parameters, the approach requires additional training data to perform at equal accuracy but with increasing number of expert samples converges towards the performance with predefined features. Since the given features in these simplified toy problems are optimal and the true reward is directly calculated on their basis, automatically learned features cannot exceed the performance. However, in real-world scenarios, the compression of raw data - such as images - to feature representations leads to information loss and the learning of task-relevant features gains even more importance.

•191991

Figure 8: Application of convolutional layers for spatial feature learning. Spatial feature learning quickly converges to performance with optimally designed features.

<!-- image -->

## 5. Conclusion and Future Work

This paper presents Maximum Entropy Deep IRL, a framework exploiting FCNNs for reward structure approximation in Inverse Reinforcement Learning. Neural networks lend themselves naturally to this task as they combine representational power with computational efficiency compared to state-of-the-art methods. Unlike prior art in this domain DeepIRL can therefore be applied in cases where complex reward structures need to be modelled for large state spaces. Moreover, training can be achieved effectively and efficiently within the popular Maximum Entropy IRL framework. A further advantage of DeepIRL lies in its versatility. Custom network architectures and types can be developed for any given task while exploiting the same cost function in training. This is expressed in section 4.3, where convolutional filters are applied to eliminate the need of manual feature design.

Our experiments show that DeepIRL's performance is commensurate to the state-of-the-art on a common benchmark. While exhibiting slightly increased requirements regarding training data in this benchmark, a principal strength of the approach lies in its algorithmic complexity independent of the number of demonstrations samples. Therefore, it is particularly well-suited for life-long learning scenarios in the context of robotics, which inherently provide sufficient amounts of training data. We also provide an alternative evaluation on a new benchmark with a significantly more complex reward structure, where DeepIRL significantly outperforms the current state-of-the-art and proves its strong capability in modeling the interaction between features. Furthermore, we extend the approach to wider filters in order to eliminate the dependency on precomputed features and to emphasise the adaptability of framing IRL in the context of deep learning.

In future work we will explore the benefits of autoencoder-

style pretraining to reduce the increased demand of expert demonstrations when employing wider convolutional filters. Especially when based on more complex inputs such as raw image data, the easily available unsupervised training data will help to learn features which then only need to be refined during the supervised IRL-based training phase. Due to the variety of existing work on FCNN architectures mentioned in section 1, we expect to be able to benefit from applying more complex networks for real life problems, such as the skipping architecture by (Long et al., 2014), which enables the concatenation of fine structural information alongside with coarser higher level features in the last regression layer to improve overall performance in evaluating features of multiple scales. Furthermore, other methods for optimising demonstration data likelihood such as given by (Babes et al., 2011) will be evaluated.

## References

- Abbeel, Pieter and Ng, Andrew Y. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning , pp. 1. ACM, 2004.
- Argall, Brenna D, Chernova, Sonia, Veloso, Manuela, and Browning, Brett. A survey of robot learning from demonstration. Robotics and autonomous systems , 57 (5):469-483, 2009.
- Asoh, Hideki, Akaho, Masanori Shiro1 Shotaro, Kamishima, Toshihiro, Hasida, Koiti, Aramaki, Eiji, and Kohro, Takahide. An application of inverse reinforcement learning to medical records of diabetes treatment. 2013.
- Babes, Monica, Marivate, Vukosi, Subramanian, Kaushik, and Littman, Michael L. Apprenticeship learning about multiple intentions. In Proceedings of the 28th International Conference on Machine Learning (ICML-11) , pp. 897-904, 2011.
- Baker, Chris L, Saxe, Rebecca, and Tenenbaum, Joshua B. Action understanding as inverse planning. Cognition , 113(3):329-349, 2009.
- Bengio, Yoshua. Learning deep architectures for ai. Foundations and trends R © in Machine Learning , 2(1):1-127, 2009.
- Bengio, Yoshua, LeCun, Yann, et al. Scaling learning algorithms towards AI. Large-scale kernel machines , 34(5), 2007.
- Bengio, Yoshua, Courville, Aaron C., and Vincent, Pascal. Unsupervised feature learning and deep learning: A review and new perspectives. CoRR , abs/1206.5538, 2012. URL http://arxiv.org/abs/1206.5538 .
- Choi, Jaedeug and Kim, Kee-Eung. Bayesian nonparametric feature construction for inverse reinforcement learning. In Proceedings of the Twenty-Third international joint conference on Artificial Intelligence , pp. 12871293. AAAI Press, 2013.
- Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic optimization. The Journal of Machine Learning Research , 12:2121-2159, 2011.
- Hassoun, Mohamad H. Fundamentals of artificial neural networks . MIT press, 1995.
- Hinton, Geoffrey E., Srivastava, Nitish, Krizhevsky, Alex, Sutskever, Ilya, and Salakhutdinov, Ruslan. Improving neural networks by preventing co-adaptation of feature detectors. CoRR , abs/1207.0580, 2012. URL http: //arxiv.org/abs/1207.0580 .
- Hornik, Kurt, Stinchcombe, Maxwell, and White, Halbert. Multilayer feedforward networks are universal approximators. Neural networks , 2(5):359-366, 1989.
- Kitani, Kris, Ziebart, Brian, Bagnell, James, and Hebert, Martial. Activity forecasting. Computer Vision-ECCV 2012 , pp. 201-214, 2012.
- Levine, Sergey, Popovic, Zoran, and Koltun, Vladlen. Feature construction for inverse reinforcement learning. In Advances in Neural Information Processing Systems , pp. 1342-1350, 2010.
- Levine, Sergey, Popovic, Zoran, and Koltun, Vladlen. Nonlinear inverse reinforcement learning with gaussian processes. In Advances in Neural Information Processing Systems , pp. 19-27, 2011.
- Levine, Sergey, Finn, Chelsea, Darrell, Trevor, and Abbeel, Pieter. Learning deep vision-based costs and policies. Robotics: Science and Systems. WS: Learning from Demonstration , 2015.
- Liu, Fayao, Shen, Chunhua, Lin, Guosheng, and Reid, Ian D. Learning depth from single monocular images using deep convolutional neural fields. CoRR , abs/1502.07411, 2015. URL http://arxiv.org/ abs/1502.07411 .
- Long, Jonathan, Shelhamer, Evan, and Darrell, Trevor. Fully convolutional networks for semantic segmentation. CoRR , abs/1411.4038, 2014. URL http://arxiv. org/abs/1411.4038 .
- Lopes, Manuel, Melo, Francisco, and Montesano, Luis. Active learning for reward estimation in inverse reinforcement learning. In Machine Learning and Knowledge Discovery in Databases , pp. 31-46. Springer, 2009.

- Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Graves, Alex, Antonoglou, Ioannis, Wierstra, Daan, and Riedmiller, Martin. Playing atari with deep reinforcement learning. CoRR , abs/1312.5602, 2013.
- Neu, Gergely and Szepesv´ ari, Csaba. Apprenticeship learning using inverse reinforcement learning and gradient methods. CoRR , abs/1206.5264, 2012.
- Ramachandran, Deepak and Amir, Eyal. Bayesian inverse reinforcement learning. Urbana , 51:61801, 2007.
- Ratliff, Nathan D, Bagnell, J Andrew, and Zinkevich, Martin A. Maximum margin planning. In Proceedings of the 23rd international conference on Machine learning , pp. 729-736. ACM, 2006.
- Sermanet, Pierre, Eigen, David, Zhang, Xiang, Mathieu, Micha¨ el, Fergus, Rob, and LeCun, Yann. Overfeat: Integrated recognition, localization and detection using convolutional networks. CoRR , abs/1312.6229, 2013. URL http://arxiv.org/abs/1312.6229 .
- Snyman, Jan. Practical mathematical optimization: an introduction to basic optimization theory and classical and new gradient-based algorithms , volume 97. Springer Science &amp; Business Media, 2005.
- Syed, Umar and Schapire, Robert E. A game-theoretic approach to apprenticeship learning. In Advances in neural information processing systems , pp. 1449-1456, 2007.
- Tompson, Jonathan, Jain, Arjun, LeCun, Yann, and Bregler, Christoph. Joint training of a convolutional network and a graphical model for human pose estimation. CoRR , abs/1406.2984, 2014. URL http://arxiv. org/abs/1406.2984 .
- Vedaldi, A. and Lenc, K. Matconvnet - convolutional neural networks for matlab. CoRR , abs/1412.4564, 2014.
- Ziebart, Brian D, Maas, Andrew L, Bagnell, J Andrew, and Dey, Anind K. Maximum entropy inverse reinforcement learning. In AAAI , pp. 1433-1438, 2008.