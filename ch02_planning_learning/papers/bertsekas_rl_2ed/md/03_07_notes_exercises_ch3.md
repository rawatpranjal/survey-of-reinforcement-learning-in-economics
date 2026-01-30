# 3.7: Notes, Sources, and Exercises Ch3

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 449-454
**Topics:** exercises, notes, sources, chapter 3

---

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