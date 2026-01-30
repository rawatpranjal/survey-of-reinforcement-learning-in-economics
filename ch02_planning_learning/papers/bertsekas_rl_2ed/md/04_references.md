# References

**Source:** Bertsekas, "A Course in Reinforcement Learning" (2nd ed., 2025)
**Pages:** 455-521
**Topics:** references, bibliography, citations

---

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

[Bra21] Brandimarte, P., 2021. From Shortest Paths to Reinforcement Learning: A MATLAB-Based Tutorial on Dynamic Programming, Springer.

[BuK97] Burnetas, A. N., and Katehakis, M. N., 1997. 'Optimal Adaptive Policies for Markov Decision Processes,' Math. of Operations Research, Vol. 22, pp. 222-255.

[CBH09] Choi, H. L., Brunet, L., and How, J. P., 2009. 'Consensus-Based Decentralized Auctions for Robust Task Allocation,' IEEE Transactions on Robotics, Vol. 25, pp. 912-926.

[CCC21] Cen, S., Cheng, C., Chen, Y., Wei, Y., and Chi, Y., 2022. 'Fast Global Convergence of Natural Policy Gradient Methods with Entropy Regularization,' Operations Research, Vol. 70, pp. 2563-2578.

[CFH05] Chang, H. S., Hu, J., Fu, M. C., and Marcus, S. I., 2005. 'An Adaptive Sampling Algorithm for Solving Markov Decision Processes,' Operations Research, Vol. 53, pp. 126-139.

[CFH13] Chang, H. S., Hu, J., Fu, M. C., and Marcus, S. I., 2013. Simulation-Based Algorithms for Markov Decision Processes, 2nd Edition, Springer, NY.

[CFM05] Costa, O. L. V., Fragoso, M. D., and Marques, R. P., 2005. Discrete-Time Markov Jump Linear Systems, Springer Science and Business Media.

[CLD19] Coulson, J., Lygeros, J., and Dorfler, F., 2019. 'Data-Enabled Predictive Control: In the Shallows of the DeePC,' 18th European Control Conference, pp. 307-312.

[CLR21] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I., 2021. 'Decision Transformer: Reinforcement Learning via Sequence Modeling,' Advances in Neural Information Processing Systems, Vol. 34, pp. 15084-15097.

[CLT19] Chapman, M. P., Lacotte, J., Tamar, A., Lee, D., Smith, K. M., Cheng, V., Fisac, J. F., Jha, S., Pavone, M., and Tomlin, C. J., 2019. 'A Risk-Sensitive Finite-Time Reachability Approach for Safety of Stochastic Dynamic Systems,' arXiv:1902.11277.

[CMT87a] Clarke, D. W., Mohtadi, C., and Tu ff s, P. S., 1987. 'Generalized Predictive Control - Part I. The Basic Algorithm,' Automatica, Vol. 23, pp. 137-148.

[CMT87b] Clarke, D. W., Mohtadi, C., and Tu ff s, P. S., 1987. 'Generalized Predictive Control - Part II,' Automatica, Vol. 23, pp. 149-160.

[CRV06] Cogill, R., Rotkowitz, M., Van Roy, B., and Lall, S., 2006. 'An Approximate Dynamic Programming Approach to Decentralized Control of Stochastic Systems,' in Control of Uncertain Systems: Modelling, Approximation, and Design, Springer, Berlin, pp. 243-256.

[CWA22] Chen, S. W., Wang, T., Atanasov, N., Kumar, V., and Morari, M., 2022. 'Large Scale Model Predictive Control with Neural Networks and Primal Active Sets,' Automatica, Vol. 135.

[CWC86] Chizeck, H. J., Willsky, A. S., and Casta˜ non, D., 'Discrete- Time MarkovianJump Linear Quadratic Optimal Control,' International Journal of Control, Vol. 43, pp. 213231.

[CXL19] Chu, Z., Xu, Z., and Li, H., 2019. 'New Heuristics for the RCPSP with Multiple Overlapping Modes,' Computers and Industrial Engineering, Vol. 131, pp. 146-156.

[CaB07] Camacho, E. F., and Bordons, C., 2007. Model Predictive Control, 2nd Edition, Springer, New York, NY.

[CaC97] Cao, X. R., and Chen, H. F., 1997. 'Perturbation Realization Potentials and

Sensitivity Analysis of Markov Processes,' IEEE Trans. on Aut. Control, Vol. 32, pp. 1382-1393.

[CaW98] Cao, X. R., and Wan, Y. W., 1998. 'Algorithms for Sensitivity Analysis of Markov Systems Through Potentials and Perturbation Realization,' IEEE Trans. Control Systems Technology, Vol. 6, pp. 482-494.

[Can16] Candy, J. V., 2016. Bayesian Signal Processing: Classical, Modern, and Particle Filtering Methods, Wiley-IEEE Press.

[Cao07] Cao, X. R., 2007. Stochastic Learning and Optimization: A Sensitivity-Based Approach, Springer, NY.

[ChC17] Chui, C. K., and Chen, G., 2017. Kalman Filtering, Springer International Publishing.

[ChM82] Chatelin, F., and Miranker, W. L., 1982. 'Acceleration by Aggregation of Successive Approximation Methods,' Linear Algebra and its Applications, Vol. 43, pp. 17-47.

[ChP20] Chen R., and Paschalidis, I. C., 2020. Distributionally Robust Learning, Foundations and Trends in Optimization.

[Cha24] Chang, H. S., 2024. 'On the Convergence Rate of MCTS for the Optimal Value Estimation in Markov Decision Processes,' arXiv:2402.07063.

[Che72] Cherno ff , H., 1972. 'Sequential Analysis and Optimal Design,' Regional Conference Series in Applied Mathematics, SIAM, Philadelphia, PA.

[Chr97] Christodouleas, J. D., 1997. 'Solution Methods for Multiprocessor Network Scheduling Problems with Application to Railroad Operations,' Ph.D. Thesis, Operations Research Center, Massachusetts Institute of Technology.

[CiS15] Ciosek, K., and Silver, D., 2015. 'Value Iteration with Options and State Aggregation,' arXiv:1501.03959.

[Cio15] Ciosek, K. A., 2015. Linear Reinforcement Learning with Options, Doctoral Dissertation, University College London.

[Cou06] Coulom, R., 2006. 'E ffi cient Selectivity and Backup Operators in Monte-Carlo Tree Search,' International Conference on Computers and Games, Springer, pp. 72-83.

[CrS00] Cristianini, N., and Shawe-Taylor, J., 2000. An Introduction to Support Vector Machines and Other Kernel-Based Learning Methods, Cambridge Univ. Press.

[CuR80] Cutler, C. R. and Ramaker, B. L., 1980. 'Dynamic Matrix Control: A Computer Control Algorithm,' Proc. Joint Automatic Control Conf., p. 17.

[Cyb89] Cybenko, 1989. 'Approximation by Superpositions of a Sigmoidal Function,' Math. of Control, Signals, and Systems, Vol. 2, pp. 303-314.

[DDF19] Daubechies, I., DeVore, R., Foucart, S., Hanin, B., and Petrova, G., 2019. 'Nonlinear Approximation and (Deep) ReLU Networks,' arXiv:1905.02199; also Constructive Approximation, Vol. 55, 2022, pp. 127-172.

[CDV02] Costa, E. F., and Do Val, J. B. R., 2002. 'Weak Detectability and the LinearQuadratic Control Problem of Discrete-Time Markov Jump Linear Systems,' International J. of Control, Vol. 75, pp. 1282-1292.

[DBF22] Di Gennaro, G., Buonanno, A., Fioretti, G., Verolla, F., Pattipati, K. R., and Palmieri, F. A., 2022. 'Probabilistic Inference and Dynamic Programming: A Unified Approach to Multi-Agent Autonomous Coordination in Complex and Uncertain Environments,' Frontiers in Physics, Vol. 10, Article 944157.

[DBP23] Di Gennaro, G., Buonanno, A., Palmieri, F. A., Pattipati, K. R., and Merola, M., 2023. 'Path Planning of Multiple Agents Through Probability Flow,' in 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing, pp. 1-6.

[DDZ24] Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., and Xie, Z., 2024. 'DeepseekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,' arXiv:2401.06066.

[DEK98] Durbin, R., Eddy, S. R., Krogh, A., and Mitchison, G., 1998. Biological Sequence Analysis, Cambridge Univ. Press, Cambridge.

[DFM12] Desai, V. V., Farias, V. F., and Moallemi, C. C., 2012. 'Aproximate Dynamic Programming via a Smoothed Approximate Linear Program,' Operations Research, Vol. 60, pp. 655-674.

[DFM13] Desai, V. V., Farias, V. F., and Moallemi, C. C., 2013. 'Bounds for Markov Decision Processes,' in Reinforcement Learning and Approximate Dynamic Programming for Feedback Control, by F. Lewis and D. Liu (eds.), IEEE Press, Piscataway, NJ., pp. 452-473.

[DFV03] De Farias, D. P., and Van Roy, B., 2003. 'The Linear Programming Approach to Approximate Dynamic Programming,' Operations Research, Vol. 51, pp. 850-865.

[DFV04] De Farias, D. P., and Van Roy, B., 2004. 'On Constraint Sampling in the Linear Programming Approach to Approximate Dynamic Programming,' Mathematics of Operations Research, Vol. 29, pp. 462-478.

[DHS11] Duchi, J., Hazan, E., and Singer, Y., 2011. 'Adaptive Subgradient Methods for Online Learning and Stochastic Optimization,' J. of Machine Learning Research, Vol. 12, pp. 2121-2159.

[DHS12] Duda, R. O., Hart, P. E., and Stork, D. G., 2012. Pattern Classification, J. Wiley, NY.

[DJW12] Duchi, J., Jordan, M. I., Wainwright, M. J., and Wibisono, A., 2012. 'Finite Sample Convergence Rate of Zero-Order Stochastic Optimization Methods,' NIPS, pp. 1448-1456.

[DJW15] Duchi, J., Jordan, M. I., Wainwright, M. J., and Wibisono, A., 2015. 'Optimal Rates for Zero-Order Convex Optimization: The Power of Two Function Evaluations,' IEEE Trans. on Information Theory, Vol. 61, pp. 2788-2806.

[DMB21] Dar, Y., Muthukumar, V., and Baraniuk, R. G., 2021. 'A Farewell to the Bias-Variance Tradeo ff ? An Overview of the Theory of Overparameterized Machine Learning, arXiv:2109.02355.

[DNP11] Deisenroth, M. P., Neumann, G., and Peters, J., 2011. 'A Survey on Policy Search for Robotics,' Foundations and Trends in Robotics, Vol. 2, pp. 1-142.

[DNW16] David, O. E., Netanyahu, N. S., and Wolf, L., 2016. 'Deepchess: End-to-End Deep Neural Network for Automatic Learning in Chess,' in International Conference on Artificial Neural Networks, pp. 88-96.

[DeF04] De Farias, D. P., 2004. 'The Linear Programming Approach to Approximate Dynamic Programming,' in Learning and Approximate Dynamic Programming, by J. Si, A. Barto, W. Powell, and D. Wunsch, (Eds.), IEEE Press, NY.

[DeG70] DeGroot, M. H., 1970. Optimal Statistical Decisions, McGraw-Hill, NY.

[DeK11] Devlin, S., and Kudenko, D., 2011. 'Theoretical Considerations of PotentialBased Reward Shaping for Multi-Agent Systems,' in Proceedings of AAMAS.

[Den67] Denardo, E. V., 1967. 'Contraction Mappings in the Theory Underlying Dynamic Programming,' SIAM Review, Vol. 9, pp. 165-177.

[DiL08] Dimitrakakis, C., and Lagoudakis, M. G., 2008. 'Rollout Sampling Approximate Policy Iteration,' Machine Learning, Vol. 72, pp. 157-171.

[DiM10] Di Castro, D., and Mannor, S., 2010. 'Adaptive Bases for Reinforcement Learning,' Machine Learning and Knowledge Discovery in Databases, Vol. 6321, pp. 312-327.

[DiW02] Dietterich, T. G., and Wang, X., 2002. 'Batch Value Function Approximation via Support Vectors,' in Advances in Neural Information Processing Systems, pp. 14911498.

[DoD93] Douglas, C. C., and Douglas, J., 1993. 'A Unified Convergence Theory for Abstract Multigrid or Multilevel Algorithms, Serial and Parallel,' SIAM J. Num. Anal., Vol. 30, pp. 136-158.

[DoJ09] Doucet, A., and Johansen, A. M., 2009. 'A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later,' Handbook of Nonlinear Filtering, Oxford University Press, Vol. 12, p. 3.

[DrH01] Drezner, Z., and Hamacher, H. W. eds., 2001. Facility Location: Applications and Theory, Springer Science and Business Media.

[Dre65] Dreyfus, S. D., 1965. Dynamic Programming and the Calculus of Variations, Academic Press, NY.

[DuV99] Duin, C., and Voss, S., 1999. 'The Pilot Method: A Strategy for Heuristic Repetition with Application to the Steiner Problem in Graphs,' Networks: An International Journal, Vol. 34, pp. 181-191.

[Duc18] Duchi, J. C., 2018. Introductory lectures on stochastic optimization. The mathematics of data, Vol. 25, pp. 99-186.

[EDS18] Efroni, Y., Dalal, G., Scherrer, B., and Mannor, S., 2018. 'Beyond the One-Step Greedy Approach in Reinforcement Learning,' in Proc. International Conf. on Machine Learning, pp. 1387-1396.

[ELP12] Estanjini, R. M., Li, K., and Paschalidis, I. C., 2012. 'A Least Squares Temporal Di ff erence Actor-Critic Algorithm with Applications to Warehouse Management,' Naval Research Logistics, Vol. 59, pp. 197-211.

[EMM05] Engel, Y., Mannor, S., and Meir, R., 2005. 'Reinforcement Learning with Gaussian Processes,' in Proc. of the 22nd ICML, pp. 201-208.

[EPE20] Emami, P., Pardalos, P. M., Elefteriadou, L., and Ranka, S., 2020. 'Machine Learning Methods for Data Association in Multi-Object Tracking,' ACM Computing Surveys (CSUR), Vol. 53, pp. 1-34.

[Edd96] Eddy, S. R., 1996. 'Hidden Markov Models,' Current Opinion in Structural Biology, Vol. 6, pp. 361-365.

[EpM02] Ephraim, Y., and Merhav, N., 2002. 'Hidden Markov Processes,' IEEE Trans. on Information Theory, Vol. 48, pp. 1518-1569.

[FGK18] Fazel, M., Ge, R., Kakade, S., and Mesbahi, M., 2018. 'Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator,' in Proc. of ICML, pp. 1467-1476.

[FHS09] Feitzinger, F., Hylla, T., and Sachs, E. W., 2009. 'Inexact Kleinman-Newton Method for Riccati Equations,' SIAM Journal on Matrix Analysis and Applications, Vol. 3, pp. 272-288.

[FIA03] Findeisen, R., Imsland, L., Allgower, F., and Foss, B.A., 2003. 'State and Output Feedback Nonlinear Model Predictive Control: An Overview,' European Journal of Control, Vol. 9, pp. 190-206.

[FPB15] Farahmand, A. M., Precup, D., Barreto, A. M., and Ghavamzadeh, M., 2015. 'Classification-Based Approximate Policy Iteration,' IEEE Trans. on Automatic Control, Vol. 60, pp. 2989-2993.

[FeV02] Ferris, M. C., and Voelker, M. M., 2002. 'Neuro-Dynamic Programming for Radiation Treatment Planning,' Numerical Analysis Group Research Report NA-02/06, Oxford University Computing Laboratory, Oxford University.

[FeV04] Ferris, M. C., and Voelker, M. M., 2004. 'Fractionation in Radiation Treatment Planning,' Mathematical Programming B, Vol. 102, pp. 387-413.

[Fel60] Feldbaum, A. A., 1960. 'Dual Control Theory,' Automation and Remote Control, Vol. 21, pp. 874-1039.

[Fel63] Feldbaum, A. A., 1963. 'Dual Control Theory Problems,' IFAC Proceedings, pp. 541-550.

[FiT91] Filar, J. A., and Tolwinski, B., 1991. 'On the Algorithm of Pollatschek and Avi-ltzhak,' in Stochastic Games and Related Topics, Theory and Decision Library, Springer, Vol. 7, pp. 59-70.

[FiV96] Filar, J., and Vrieze, K., 1996. Competitive Markov Decision Processes, Springer.

[FoK09] Forrester, A. I., and Keane, A. J., 2009. 'Recent Advances in Surrogate-Based Optimization. Progress in Aerospace Sciences,' Vol. 45, pp. 50-79.

[For73] Forney, G. D., 1973. 'The Viterbi Algorithm,' Proc. IEEE, Vol. 61, pp. 268-278.

[FrW76] Francis, B. A., and Wonham, W. M., 1976. 'The Internal Model Principle of Control Theory,' Automatica, Vol. 12, pp. 457-465.

[Fra18] Frazier, P. I., 2018. 'A Tutorial on Bayesian Optimization,' arXiv:1807.02811.

[Fu17] Fu, M. C., 2017. 'Markov Decision Processes, AlphaGo, and Monte Carlo Tree Search: Back to the Future,' Leading Developments from INFORMS Communities, INFORMS, pp. 68-88.

[FuH94] Fu, M. C., and Hu, J.-Q., 1994. 'Smoothed Perturbation Analysis Derivative Estimation for Markov Chains,' Oper. Res. Letters, Vol. 41, pp. 241-251.

[Fun89] Funahashi, K., 1989. 'On the Approximate Realization of Continuous Mappings by Neural Networks,' Neural Networks, Vol. 2, pp. 183-192.

[GBB04] Greensmith, E., Bartlett, P. L., and Baxter, J., 2004. 'Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning,' Journal of Machine Learning Research, Vol. 5, pp. 1471-1530.

[GBC16] Goodfellow, I., Bengio, J., and Courville, A., Deep Learning, MIT Press, Cambridge, MA.

[GBL12] Grondman, I., Busoniu, L., Lopes, G. A. D., and Babuska, R., 2012. 'A Survey of Actor-Critic Reinforcement Learning: Standard and Natural Policy Gradients,' IEEE Trans. on Systems, Man, and Cybernetics, Part C, Vol. 42, pp. 1291-1307.

[GBL19] Goodson, J. C., Bertazzi, L., and Levary, R. R., 2019. 'Robust Dynamic Media Selection with Yield Uncertainty: Max-Min Policies and Dual Bounds,' Report.

[GDM19] Guerriero, F., Di Puglia Pugliese, L., and Macrina, G., 2019. 'A Rollout Algorithm for the Resource Constrained Elementary Shortest Path Problem,' Optimization

Methods and Software, Vol. 34, pp. 1056-1074.

[GFB24] Gioia, D. G., Fadda, E., and Brandimarte, P., 2024. 'Rolling Horizon Policies for Multi-Stage Stochastic Assemble-to-Order Problems,' International Journal of Production Research, Vol. 62, pp. 5108-5126.

[GGS13] Gabillon, V., Ghavamzadeh, M., and Scherrer, B., 2013. 'Approximate Dynamic Programming Finally Performs Well in the Game of Tetris,' in NIPS, pp. 17541762.

[GGW11] Gittins, J., Glazebrook, K., and Weber, R., 2011. Multi-Armed Bandit Allocation Indices, J. Wiley, NY.

[GHC21] Gerlach, T., Ho ff mann, F., and Charlish, A., 2021. 'Policy Rollout Action Selection with Knowledge Gradient for Sensor Path Planning,' 2021 IEEE 24th International Conference on Information Fusion, pp. 1-8.

[GLB24] Gundawar, A., Li, Y., and Bertsekas, D., 2024. 'Playing Superior Computer Chess with Model Predictive Control and Rollout,' arXiv:2409.06477.

[GLG11] Gabillon, V., Lazaric, A., Ghavamzadeh, M., and Scherrer, B., 2011. 'Classification-Based Policy Iteration with a Critic,' in Proc. of ICML.

[GMP15] Ghavamzadeh, M., Mannor, S., Pineau, J., and Tamar, A., 2015. 'Bayesian Reinforcement Learning: A Survey,' Foundations and Trends in Machine Learning, Vol. 8, pp. 359-483.

[GPG22] Garces, D., Bhattacharya, S., Gil, G., and Bertsekas, D., 'Multiagent Reinforcement Learning for Autonomous Routing and Pickup Problem with Adaptation to Variable Demand,' arXiv:2211.14983.

[GSD06] Goodwin, G., Seron, M. M., and De Dona, J. A., 2006. Constrained Control and Estimation: An Optimisation Approach, Springer, NY.

[GSS93] Gordon, NJ., Salmond, D. J., and Smith, A. F., 1993. 'Novel Approach to Nonlinear/Non-Gaussian Bayesian State Estimation,' in IEE Proceedings, Vol. 140, pp. 107-113.

[GTA17] Gommans, T. M. P., Theunisse, T. A. F., Antunes, D. J., and Heemels, W. P. M. H., 2017. 'Resource-Aware MPC for Constrained Linear Systems: Two Rollout Approaches,' Journal of Process Control, Vol. 51, pp. 68-83.

[GTO15] Goodson, J. C., Thomas, B. W., and Ohlmann, J. W., 2015. 'RestockingBased Rollout Policies for the Vehicle Routing Problem with Stochastic Demand and Duration Limits,' Transportation Science, Vol. 50, pp. 591-607.

[GTO17] Goodson, J. C., Thomas, B. W., and Ohlmann, J. W., 2017. 'A Rollout Algorithm Framework for Heuristic Solutions to Finite-Horizon Stochastic Dynamic Programs,' European Journal of Operational Research, Vol. 258, pp. 216-229.

[GaB84] Gafni, E. M., and Bertsekas, D. P., 1984. 'Two-Metric Projection Methods for Constrained Optimization,' SIAM J. on Control and Optimization, Vol. 22, pp. 936-964.

[GeB13] Ge ff ner, H., and Bonet, B., 2013. A Concise Introduction to Models and Methods for Automated Planning, Morgan and Claypool Publishers.

[GeP24] Gerlach, T., and Piatkowski, N., 2024. 'Dynamic Range Reduction via Branchand-Bound,' arXiv:2409.10863.

[Gly87] Glynn, P. W., 1987. 'Likelihood Ratio Gradient Estimation: An Overview,' Proc. of the 1987 Winter Simulation Conference, pp. 366-375.

[Gly90] Glynn, P. W., 1990. 'Likelihood Ratio Gradient Estimation for Stochastic Systems,' Communications of the ACM, Vol. 33, pp. 75-84.

[GoS84] Goodwin, G. C., and Sin, K. S. S., 1984. Adaptive Filtering, Prediction, and Control, Prentice-Hall, Englewood Cli ff s, NJ.

[Gor95] Gordon, G. J., 1995. 'Stable Function Approximation in Dynamic Programming,' in Machine Learning: Proceedings of the Twelfth International Conference, Morgan Kaufmann, San Francisco, CA.

[Gos15] Gosavi, A., 2015. Simulation-Based Optimization: Parametric Optimization Techniques and Reinforcement Learning, 2nd Edition, Springer, NY.

[GrZ19] Gros, S., and Zanon, M., 2019. 'Data-Driven Economic NMPC Using Reinforcement Learning,' IEEE Trans. on Aut. Control, Vol. 65, pp. 636-648.

[GrZ22] Gros, S., and Zanon, M., 2022. 'Learning for MPC with Stability and Safety Guarantees,' Automatica, Vol. 146, p. 110598.

[Grz17] Grzes, M., 2017. 'Reward Shaping in Episodic Reinforcement Learning,' in Proc. of the 16th Conference on Autonomous Agents and MultiAgent Systems, pp. 565-573.

[GuM01] Guerriero, F., and Musmanno, R., 2001. 'Label Correcting Methods to Solve Multicriteria Shortest Path Problems,' J. Optimization Theory Appl., Vol. 111, pp. 589-613.

[GuM03] Guerriero, F., and Mancini, M., 2003. 'A Cooperative Parallel Rollout Algorithm for the Sequential Ordering Problem,' Parallel Computing, Vol. 29, pp. 663-677.

[Gup20] Gupta, A., 2020. 'Existence of Team-Optimal Solutions in Static Teams with Common Information: A Topology of Information Approach,' SIAM J. on Control and Optimization, Vol. 58, pp. 998-1021.

[HCR21] Ho ff mann, F., Charlish, A., Ritchie, M., and Gri ffi ths, H., 2021. 'Policy Rollout Action Selection in Continuous Domains for Sensor Path Planning,' IEEE Trans. on Aerospace and Electronic Systems.

[HJG16] Huang, Q., Jia, Q. S., and Guan, X., 2016. 'Robust Scheduling of EV Charging Load with Uncertain Wind Power Integration,' IEEE Trans. on Smart Grid, Vol. 9, pp. 1043-1054.

[HLB19] Hafner, D., Lillicrap, T., Ba, J. and Norouzi, M., 2019. 'Dream to Control: Learning Behaviors by Latent Imagination,' arXiv:1912.01603.

[HLS06] Han, J., Lai, T. L., and Spivakovsky, V., 2006. 'Approximate Policy Optimization and Adaptive Control in Regression Models,' Computational Economics, Vol. 27, pp. 433-452.

[HLZ19] Ho, T. Y., Liu, S., and Zabinsky, Z. B., 2019. 'A Multi-Fidelity Rollout Algorithm for Dynamic Resource Allocation in Population Disease Management,' Health Care Management Science, Vol. 22, pp. 727-755.

[HMR19] Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J., 2019. 'Surprises in High-Dimensional Ridgeless Least Squares Interpolation,' arXiv:1903.08560; also Annals of Statistics, Vol. 50, 2022.

[HMV24] Houska, B., Muller, M. A., and Villanueva, M. E., 2024. 'Polyhedral Control Design: Theory and Methods,' arXiv:2412.13082.

[HSS08] Hofmann, T., Scholkopf, B., and Smola, A. J., 2008. 'Kernel Methods in Machine Learning,' The Annals of Statistics, Vol. 36, pp. 1171-1220.

[HSW89] Hornick, K., Stinchcombe, M., and White, H., 1989. 'Multilayer Feedforward Networks are Universal Approximators,' Neural Networks, Vol. 2, pp. 359-159.

[HVD15] Hinton, G., Vinyals, O., and Dean, J., 2015. 'Distilling the Knowledge in a Neural Network,' arXiv:1503.02531.

[HWM20] Hewing, L., Wabersich, K. P., Menner, M., and Zeilinger, M. N., 2020. 'Learning-Based Model Predictive Control: Toward Safe Learning in Control,' Annual Review of Control, Robotics, and Autonomous Systems, Vol. 3, pp. 269-296.

[HWP22] Hu, J., Wang, Y., Pang, Y., and Liu, Y., 2022. 'Optimal Maintenance Scheduling under Uncertainties using Linear Programming-Enhanced Reinforcement Learning,' Engineering Applications of Artificial Intelligence, Vol. 109.

[HaR21] Hardt, M., and Recht, B., 2021. Patterns, Predictions, and Actions: A Story About Machine Learning, arXiv:2102.05242; published by Princeton Univ. Press, 2022.

[HaS18] Ha, D., and Schmidhuber, J., 2018. 'World Models,' arXiv:1803.10122.

[Han98] Hansen, E. A., 1998. 'Solving POMDPs by Searching in Policy Space,' in Proc. of the 14th Conf. on Uncertainty in Artificial Intelligence, pp. 211-219.

[Hay08] Haykin, S., 2008. Neural Networks and Learning Machines, 3rd Edition, PrenticeHall, Englewood-Cli ff s, NJ.

[HeZ19] Hewing, L., and Zeilinger, M. N., 2019. 'Scenario-Based Probabilistic Reachable Sets for Recursively Feasible Stochastic Model Predictive Control,' IEEE Control Systems Letters, Vol. 4, pp. 450-455.

[Hew71] Hewer, G., 1971. 'An Iterative Technique for the Computation of the Steady State Gains for the Discrete Optimal Regulator,' IEEE Trans. on Automatic Control, Vol. 16, pp. 382-384.

[Ho80] Ho, Y. C., 1980. 'Team Decision Theory and Information Structures,' Proceedings of the IEEE, Vol. 68, pp. 644-654.

[HuM16] Huan, X., and Marzouk, Y. M., 2016. 'Sequential Bayesian Optimal Experimental Design via Approximate Dynamic Programming,' arXiv:1604.08320.

[Hua15] Huan, X., 2015. Numerical Approaches for Sequential Bayesian Optimal Experimental Design, Ph.D. Thesis, MIT.

[Hyl11] Hylla, T., 2011. Extension of Inexact Kleinman-Newton Methods to a General Monotonicity Preserving Convergence Theory, PhD Thesis, Univ. of Trier.

[IFT19] Issakkimuthu, M., Fern, A., and Tadepalli, P., 2019. 'The Choice Function Framework for Online Policy Improvement,' arXiv:1910.00614; also Proc. of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020.

[IJT18] Iusem, A., Jofre, A., and Thompson, P., 2018. 'Incremental Constraint Projection Methods for Monotone Stochastic Variational Inequalities,' Math. of Operations Research, Vol. 44, pp. 236-263.

[IoS96] Ioannou, P. A., and Sun, J., 1996. Robust Adaptive Control, Prentice-Hall, Englewood Cli ff s, NJ.

[JCG20] Jiang, S., Chai, H., Gonzalez, J., and Garnett, R., 2020. 'BINOCULARS for E ffi cient, Nonmyopic Sequential Experimental Design,' in Proc. Intern. Conference on Machine Learning, pp. 4794-4803.

[JGJ18] Jones, M., Goldstein, M., Jonathan, P., and Randell, D., 2018. 'Bayes Linear Analysis of Risks in Sequential Optimal Design Problems,' Electronic Journal of Statistics, Vol. 12, pp. 4002-4031.

[JJB20] Jiang, S., Jiang, D. R., Balandat, M., Karrer, B., Gardner, J. R., and Garnett, R., 2020. 'E ffi cient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees,' arXiv:2006.15779.

[JSJ95] Jaakkola, T., Singh, S. P., and Jordan, M. I., 1995. 'Reinforcement Learning Algorithm for Partially Observable Markov Decision Problems,' NIPS, Vol. 7, pp. 345352.

[JSW98] Jones, D. R., Schonlau, M., and Welch, W. J., 1998. 'E ffi cient Global Optimization of Expensive Black-Box Functions,' J. of Global Optimization, Vol. 13, pp. 455-492.

[JiJ17] Jiang, Y., and Jiang, Z. P., 2017. Robust Adaptive Dynamic Programming, J. Wiley, NY.

[JoB16] Joseph, A. G., and Bhatnagar, S., 2016. 'Revisiting the Cross Entropy Method with Applications in Stochastic Global Optimization and Reinforcement Learning,' in Proc. of the 22nd European Conference on Artificial Intelligence, pp. 1026-1034.

[JoB18] Joseph, A. G., and Bhatnagar, S., 2018. 'A Cross Entropy Based Optimization Algorithm with Global Convergence Guarantees,' arXiv:1801.10291.

[Jon90] Jones, L. K., 1990. 'Constructive Approximations for Neural Networks by Sigmoidal Functions,' Proceedings of the IEEE, Vol. 78, pp. 1586-1589.

[JuM23] Jurafsky, D., and Martin, J. H., 2023. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, draft 3rd edition (on-line).

[JuP07] Jung, T., and Polani, D., 2007. 'Kernelizing LSPE( λ ),' Proc. 2007 IEEE Symposium on Approximate Dynamic Programming and Reinforcement Learning, Honolulu, Ha., pp. 338-345.

[KAC15] Kochenderfer, M. J., with Amato, C., Chowdhary, G., How, J. P., Davison Reynolds, H. J., Thornton, J. R., Torres-Carrasquillo, P. A., Ore, N. K., Vian, J., 2015. Decision Making under Uncertainty: Theory and Application, MIT Press, Cambridge, MA.

[KAH15] Khashooei, B. A., Antunes, D. J., and Heemels, W. P. M. H., 2015. 'Rollout Strategies for Output-Based Event-Triggered Control,' in Proc. 2015 European Control Conference, pp. 2168-2173.

[KGB82] Kimemia, J., Gershwin, S. B., and Bertsekas, D. P., 1982. 'Computation of Production Control Policies by a Dynamic Programming Technique,' in Analysis and Optimization of Systems, A. Bensoussan and J. L. Lions (eds.), Springer, NY., pp. 243-269.

[KKK95] Krstic, M., Kanellakopoulos, I., Kokotovic, P., 1995. Nonlinear and Adaptive Control Design, J. Wiley, NY.

[KLC98] Kaelbling, L. P., Littman, M. L., and Cassandra, A. R., 1998. 'Planning and Acting in Partially Observable Stochastic Domains,' Artificial Intelligence, Vol. 101, pp. 99-134.

[KLM82a] Krainak, J. L. S. J. C., Speyer, J., and Marcus, S., 1982. 'Static Team Problems - Part I: Su ffi cient Conditions and the Exponential Cost Criterion,' IEEE Transactions on Automatic Control, Vol. 27, pp. 839-848.

[KLM82b] Krainak, J. L. S. J. C., Speyer, J., and Marcus, S., 1982. 'Static Team Problems - Part II: A ffi ne Control Laws, Projections, Algorithms, and the LEGT Problem,' IEEE Transactions on Automatic Control, Vol. 27, pp. 848-859.

[KLM96] Kaelbling, L. P., Littman, M. L., and Moore, A. W., 1996. 'Reinforcement Learning: A Survey,' J. of Artificial Intelligence Res., Vol. 4, pp. 237-285.

[KMP06] Keller, P. W., Mannor, S., and Precup, D., 2006. 'Automatic Basis Function Construction for Approximate Dynamic Programming and Reinforcement Learning,' Proc. of the 23rd ICML, Pittsburgh, Penn.

[KRC13] Kroese, D. P., Rubinstein, R. Y., Cohen, I., Porotsky, S., and Taimre, T., 2013. 'Cross-Entropy Method,' in Encyclopedia of Operations Research and Management Science, Springer, Boston, MA, pp. 326-333.

[KRW21] Kumar, P., Rawlings, J. B., and Wright, S. J., 2021. 'Industrial, Large-Scale Model Predictive Control with Structured Neural Networks,' Computers and Chemical Engineering, Vol. 150.

[KaW94] Kall, P., and Wallace, S. W., 1994. Stochastic Programming, Wiley, Chichester, UK.

[Kak02] Kakade, S. A., 2002. 'Natural Policy Gradient,' NIPS, Vol. 14, pp. 1531-1538.

[KeG88] Keerthi, S. S., and Gilbert, E. G., 1988. 'Optimal, Infinite Horizon Feedback Laws for a General Class of Constrained Discrete Time Systems: Stability and MovingHorizon Approximations,' J. Optimization Theory Appl., Vo. 57, pp. 265-293.

[KiB14] Kingma, D. P., and Ba, J., 2014. 'Adam: A Method for Stochastic Optimization,' arXiv:1412.6980.

[Kir04] Kirk, D. E., 2004. Optimal Control Theory: An Introduction, Courier Corporation.

[Kim82] Kimemia, J., 1982. 'Hierarchical Control of Production in Flexible Manufacturing Systems,' Ph.D. Thesis, Dep. of Electrical Engineering and Computer Science, Massachusetts Institute of Technology.

[Kle09] Kleijnen, J. P., 2009. 'Kriging Metamodeling in Simulation: A Review,' European Journal of Operational Research, Vol. 192, pp. 707-716.

[Kle68] Kleinman, D. L., 1968. 'On an Iterative Technique for Riccati Equation Computations,' IEEE Trans. Aut. Control, Vol. AC-13, pp. 114-115.

[KoC16] Kouvaritakis, B., and Cannon, M., 2016. Model Predictive Control: Classical, Robust and Stochastic, Springer, NY.

[KoG98] Kolmanovsky, I., and Gilbert, E. G., 1998. 'Theory and Computation of Disturbance Invariant Sets for Discrete-Time Linear Systems,' Math. Problems in Engineering, Vol. 4, pp. 317-367.

[KoS06] Kocsis, L., and Szepesvari, C., 2006. 'Bandit Based Monte-Carlo Planning,' Proc. of 17th European Conference on Machine Learning, Berlin, pp. 282-293.

[KoT99] Konda, V. R., and Tsitsiklis, J. N., 1999. 'Actor-Critic Algorithms,' NIPS, Denver, Colorado, pp. 1008-1014.

[KoT03] Konda, V. R., and Tsitsiklis, J. N., 2003. 'Actor-Critic Algorithms,' SIAM J. on Control and Optimization, Vol. 42, pp. 1143-1166.

[Kre19] Krener, A. J., 2019. 'Adaptive Horizon Model Predictive Control and Al'brekht's Method,' arXiv:1904.00053; also Encyclopedia of Systems and Control, 2021, pp. 27-40.

[Kri16] Krishnamurthy, V., 2016. Partially Observed Markov Decision Processes, Cambridge Univ. Press.

[KuV86] Kumar, P. R., and Varaiya, P. P., 1986. Stochastic Systems: Estimation, Identification, and Adaptive Control, Prentice-Hall, Englewood Cli ff s, NJ.

[Kun14] Kung, S. Y., 2014. Kernel Methods and Machine Learning, Cambridge Univ. Press.

[L'Ec91] L'Ecuyer, P., 1991. 'An Overview of Derivative Estimation,' Proceedings of the 1991 Winter Simulation Conference, pp. 207-217.

[LEC20] Lee, E. H., Eriksson, D., Cheng, B., McCourt, M., and Bindel, D., 2020. 'E ffi cient Rollout Strategies for Bayesian Optimization,' arXiv:2002.10539.

[LEP21] Lee, E. H., Eriksson, D., Perrone, V., and Seeger, M., 2021. 'A Nonmyopic Approach to Cost-Constrained Bayesian Optimization,' in Uncertainty in Artificial Intelligence Proceedings, pp. 568-577.

[LGM10] Lazaric, A., Ghavamzadeh, M., and Munos, R., 2010. 'Analysis of a Classification-Based Policy Iteration Algorithm,' INRIA Report.

[LGW16] Lan, Y., Guan, X., and Wu, J., 2016. 'Rollout Strategies for Real-Time MultiEnergy Scheduling in Microgrid with Storage System,' IET Generation, Transmission and Distribution, Vol. 10, pp. 688-696.

[LHB25] Li, Y., Hammar, K., and Bertsekas, D. P., 2025. 'Feature-Based Belief Aggregation for Partially Observable Markov Decision Problems,' arXiv:2507.04646.

[LJM19] Li, Y., Johansson, K. H., and Martensson, J., 2019. 'Lambda-Policy Iteration with Randomization for Contractive Models with Infinite Policies: Well Posedness and Convergence,' arXiv:1912.08504; also Learning for Dynamics and Control, 2020, pp. 540-549.

[LJM21] Li, Y., Johansson, K. H., Martensson, J., and Bertsekas, D. P., 2021. 'DataDriven Rollout for Deterministic Optimal Control,' 2021 60th IEEE Conference on Decision and Control, pp. 2169-2176.

[LKG21] Li, T., Krakow, L. W., and Gopalswamy, S., 2021. 'Optimizing ConsensusBased Multi-Target Tracking with Multiagent Rollout Control Policies,' in 2021 IEEE Conference on Control Technology and Applications, pp. 131-137.

[LLL19] Liu, Z., Lu, J., Liu, Z., Liao, G., Zhang, H. H., and Dong, J., 2019. 'Patient Scheduling in Hemodialysis Service,' J. of Combinatorial Optimization, Vol. 37, pp. 337-362.

[LLP93] Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S., 1993. 'Multilayer Feedforward Networks with a Nonpolynomial Activation Function can Approximate any Function,' Neural Networks, Vol. 6, pp. 861-867.

[LPS22] Liu, M., Pedrielli, G., Sulc, P., Poppleton, E., Bertsekas, D. P., 2022. 'ExpertRNA: A New Framework for RNA Structure Prediction,' INFORMS Journal on Computing.

[LRD23] Laidlaw, C., Russell, S., and Dragan, A., 2023. 'Bridging RL Theory and Practice with the E ff ective Horizon,' arXiv:2304.09853; also Advances in Neural Information Processing Systems, Vol. 36, 2023.

[LSG07] Lee, H., Singh, S., An, W., Gokhale, S. S., Pattipati, K., and Kleinman, D. L., 2007. Rollout Strategy for Hidden Markov Model (HMM)-Based Dynamic Sensor Scheduling, in 2007 IEEE International Conference on Systems, Man and Cybernetics, pp. 553-558.

[LTZ19] Li, Y., Tang, Y., Zhang, R., and Li, N., 2019. 'Distributed Reinforcement Learning for Decentralized Linear Quadratic Control: A Derivative-Free Policy Opti-

mization Approach,' arXiv:1912.09135; also IEEE Transactions on Automatic Control, Vol. 67, 2021, pp. 6429-6444.

[LWT17] Lowe, L., Wu, Y., Tamar, A., Harb, J., Abbeel, P., Mordatch, I., 2017. 'MultiAgent Actor-Critic for Mixed Cooperative-Competitive Environments,' in Advances in Neural Information Processing Systems, pp. 6379-6390.

[LWW16] Lam, R., Willcox, K., and Wolpert, D. H., 2016. 'Bayesian Optimization with a Finite Budget: An Approximate Dynamic Programming Approach,' in Advances in Neural Information Processing Systems, pp. 883-891.

[LWW17] Liu, D., Wei, Q., Wang, D., Yang, X., and Li, H., 2017. Adaptive Dynamic Programming with Applications in Optimal Control, Springer, Berlin.

[LZS20] Li, H., Zhang, X., Sun, J., and Dong, X., 2020. 'Dynamic Resource Levelling in Projects under Uncertainty,' International J. of Production Research.

[LaP03] Lagoudakis, M. G., and Parr, R., 2003. 'Reinforcement Learning as Classification: Leveraging Modern Classifiers,' in Proc. of ICML, pp. 424-431.

[LaR85] Lai, T., and Robbins, H., 1985. 'Asymptotically E ffi cient Adaptive Allocation Rules,' Advances in Applied Math., Vol. 6, pp. 4-22.

[LaS20] Lattimore, T., and Szepesvari, C., 2020. Bandit Algorithms, Cambridge University Press.

[LaW13] Lavretsky, E., and Wise, K., 2013. Robust and Adaptive Control with Aerospace Applications, Springer.

[LaW17] Lam, R., and Willcox, K., 2017. 'Lookahead Bayesian Optimization with Inequality Constraints,' in Advances in Neural Information Processing Systems, pp. 18901900.

[Lee20] Lee, E. H., 2020. 'Budget-Constrained Bayesian Optimization, Doctoral dissertation, Cornell University.

[LiB24] Li, Y., and Bertsekas, D. P., 2024. 'Most Likely Sequence Generation for n -Grams, Transformers, HMMs, and Markov Chains, by Using Rollout Algorithms,' arXiv:2403.15465.

[LiB25a] Li, Y., and Bertsekas, D. P., 2025. 'Semilinear Dynamic Programming: Analysis, Algorithms, and Certainty Equivalence Properties,' arXiv:2501.04668.

[LiB25b] Li, Y., and Bertsekas, D. P., 2025. 'An Error Bound for Aggregation in Approximate Dynamic Programming,' arXiv:2507.01324.

[LiS16] Liang, S., and Srikant, R., 2016. 'Why Deep Neural Networks for Function Approximation?' arXiv:1610.04161.

[LiW14] Liu, D., and Wei, Q., 2014. 'Policy Iteration Adaptive Dynamic Programming Algorithm for Discrete-Time Nonlinear Systems,' IEEE Trans. on Neural Networks and Learning Systems, Vol. 25, pp. 621-634.

[LiW15] Li, H., and Womer, N. K., 2015. 'Solving Stochastic Resource-Constrained Project Scheduling Problems by Closed-Loop Approximate Dynamic Programming,' European J. of Operational Research, Vol. 246, pp. 20-33.

[Lib11] Liberzon, D., 2011. Calculus of Variations and Optimal Control Theory: A Concise Introduction, Princeton Univ. Press.

[LoC23] Loxley, P. N., and Cheung, K. W., 2023. 'A Dynamic Programming Algorithm for Finding an Optimal Sequence of Informative Measurements,' Entropy, Vol. 25, p. 251.

[MaD21] Markovsky, I., and Dorfler, F., 2021. 'Behavioral Systems Theory in DataDriven Analysis, Signal Processing, and Control,' Annual Reviews in Control, Vol. 52, pp.42-64.

[MAS19] McAleer, S., Agostinelli, F., Shmakov, A. K., and Baldi, P., 2019. 'Solving the Rubik's Cube with Approximate Policy Iteration,' in International Conference on Learning Representations.

[MBP23] Moerland, T. M., Broekens, J., Plaat, A., and Jonker, C. M., 2023. ModelBased Reinforcement Learning: A Survey, Foundations and Trends in Machine Learning, Vol. 16, pp. 1-118.

[MCT10] Mishra, N., Choudhary, A. K., Tiwari, M. K., and Shankar, R., 2010. 'Rollout Strategy-Based Probabilistic Causal Model Approach for the Multiple Fault Diagnosis,' Robotics and Computer-Integrated Manufacturing, Vol. 26, pp. 325-332.

[MDM01] Magni, L., De Nicolao, G., Magnani, L., and Scattolini, R., 2001. 'A Stabilizing Model-Based Predictive Control Algorithm for Nonlinear Systems,' Automatica, Vol. 37, pp. 1351-1362.

[MHD23] Markovsky, I., Huang, L., and Drfler, F., 2023. 'Data-Driven Control Based on the Behavioral Approach: From Theory to Applications in Power Systems,' IEEE Control Systems Magazine, Vol. 43, pp. 28-68.

[MJR22] Mania, H., Jordan, M. I., and Recht, B., 2022. 'Active Learning for Nonlinear System Identification with Guarantees,' J. of Machine Learning Research, Vol. 23, pp. 1-30.

[MKS15] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., and Petersen, S., 2015. 'Human-Level Control Through Deep Reinforcement Learning,' Nature, Vol. 518, p. 529.

[MLB25] Mandal, L., Lakshminarayanan, C., and Bhatnagar, S., 2025. 'Approximate Linear Programming for Decentralized Policy Iteration in Cooperative Multi-Agent Markov Decision Processes,' Systems and Control Letters, Vol. 196, p. 106003.

[MLM20] Montenegro, M., Lopez, R., Menchaca-Mendez, R., Becerra, E., and MenchacaMendez, R., 2020. 'A Parallel Rollout Algorithm for Wildfire Suppression,' in Proc. Intern. Congress of Telematics and Computing, pp. 244-255.

[MLW24] Musunuru, P., Li, Y., Weber, J., and Bertsekas, D., 2024. 'An Approximate Dynamic Programming Framework for Occlusion-Robust Multi-Object Tracking,' arXiv:2405.15137.

[MMB02] McGovern, A., Moss, E., and Barto, A., 2002. 'Building a Basic Building Block Scheduler Using Reinforcement Learning and Rollouts,' Machine Learning, Vol. 49, pp. 141-160.

[MMK23] Marchesoni-Acland, F., Morel, J. M., Kherroubi, J., and Facciolo, G., 2023. 'Optimal and E ffi cient Binary Questioning for Human-in-the-Loop Annotation,' arXiv: 2307.01578.

[MMS05] Menache, I., Mannor, S., and Shimkin, N., 2005. 'Basis Function Adaptation in Temporal Di ff erence Reinforcement Learning,' Ann. Oper. Res., Vol. 134, pp. 215-238.

[MPK99] Meuleau, N., Peshkin, L., Kim, K. E., and Kaelbling, L. P., 1999. 'Learning Finite-State Controllers for Partially Observable Environments,' in Proc. of the 15th Conference on Uncertainty in Artificial Intelligence, pp. 427-436.

[MPL23] Macesker, M., Pattipati, K. R., Licht, S., and Gilboa, R., 2023. A Computationa-

lly-E ffi cient Rollout-Based Approach for Bathymetric Mapping with Multiple Low-Cost Unmanned Surface Vehicles,' in 2023 IEEE International Conference on Systems, Man, and Cybernetics, pp. 3559-3564.

[MPM24] Maniyar, M. P., Prashanth, L. A., Mondal, A., and Bhatnagar, S., 2024. 'A Cubic-Regularized Policy Newton Algorithm for Reinforcement Learning, in International Conference on Artificial Intelligence and Statistics, pp. 4708-4716.

[MPP04] Meloni, C., Pacciarelli, D., and Pranzo, M., 2004. 'A Rollout Metaheuristic for Job Shop Scheduling Problems,' Annals of Operations Research, Vol. 131, pp. 215-235.

[MRG03] Mannor, S., Rubinstein, R. Y., and Gat, Y., 2003. 'The Cross Entropy Method for Fast Policy Search,' in Proc. of the 20th International Conference on Machine Learning (ICML-03), pp. 512-519.

[MRM16] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K., 2016. 'Asynchronous Methods for Deep Reinforcement Learning,' in Proc. of International Conference on Machine Learning, pp. 1928-1937.

[MRR00] Mayne, D., Rawlings, J. B., Rao, C. V., and Scokaert, P. O. M., 2000. 'Constrained Model Predictive Control: Stability and Optimality,' Automatica, Vol. 36, pp. 789-814.

[MVS19] Muthukumar, V., Vodrahalli, K., and Sahai, A., 2019. 'Harmless Interpolation of Noisy Data in Regression,' arXiv:1903.09139; also IEEE Journal on Selected Areas in Information Theory, Vol. 1, 2020, pp. 67-83.

[MYF03] Moriyama, H., Yamashita, N., and Fukushima, M., 2003. 'The Incremental Gauss-Newton Algorithm with Adaptive Stepsize Rule,' Computational Optimization and Applications, Vol. 26, pp. 107-141.

[MaE07] Mamon, R. S., and Elliott, R. J. eds., 2007. Hidden Markov Models in Finance, Springer, NY.

[MaE14] Mamon, R. S., and Elliott, R. J. eds., 2014. Hidden Markov Models in Finance: Further Developments and Applications, Springer, NY.

[MaJ15] Mastin, A., and Jaillet, P., 2015. 'Average-Case Performance of Rollout Algorithms for Knapsack Problems,' J. of Optimization Theory and Applications, Vol. 165, pp. 964-984.

[MaM88] Mayne, D. Q. and Michalska, H., 1988. 'Receding Horizon Control of Nonlinear Systems,' in Proc. of the 27th IEEE Conference on Decision and Control, pp. 464-465.

[MaS99] Manning, C., and Schutze, H., 1999. Foundations of Statistical Natural Language Processing, MIT Press, Cambridge, MA.

[MaS02] Martinez, L., and Soares, S., 2002. 'Comparison Between Closed-Loop and Partial Open-Loop Feedback Control Policies in Long Term Hydrothermal Scheduling,' IEEE Transactions on Power Systems, Vol. 17, pp. 330-336.

[MaT01] Marbach, P., and Tsitsiklis, J. N., 2001. 'Simulation-Based Optimization of Markov Reward Processes,' IEEE Trans. on Aut. Control, Vol. 46, pp. 191-209.

[MaT03] Marbach, P., and Tsitsiklis, J. N., 2003. 'Approximate Gradient Methods in Policy-Space Optimization of Markov Reward Processes,' J. Discrete Event Dynamic Systems, Vol. 13, pp. 111-148.

[Mac02] Maciejowski, J. M., 2002. Predictive Control with Constraints, Addison-Wesley, Reading, MA.

[Mal10] Malikopoulos, A. A., 2010. 'A Rollout Control Algorithm for Discrete-Time

Stochastic Systems,' in Dynamic Systems and Control Conference, Vol. 44182, pp. 711717.

[Mar55] Marschak, J., 1955. 'Elements for a Theory of Teams,' Management Science, Vol. 1, pp. 127-137.

[Mar84] Martins, E. Q. V., 1984. 'On a Multicriteria Shortest Path Problem,' European J. of Operational Research, Vol. 16, pp. 236-245.

[Mar90] Mariton, M., 1990. Jump Linear Systems in Automatic Control, CRC Press.

[Mat65] J. Matyas, J., 1965. 'Random Optimization,' Automation and Remote Control, Vol. 26, pp. 246-253.

[May14] Mayne, D. Q., 2014. 'Model Predictive Control: Recent Developments and Future Promise,' Automatica, Vol. 50, pp. 2967-2986.

[MeB99] Meuleau, N., and Bourgine, P., 1999. 'Exploration of Multi-State Environments: Local Measures and Back-Propagation of Uncertainty,' Machine Learning, Vol. 35, pp. 117-154.

[MeK20] Meshram, R., and Kaza, K., 2020. 'Simulation Based Algorithms for Markov Decision Processes and Multi-Action Restless Bandits,' arXiv:2007.12933.

[Mey07] Meyn, S., 2007. Control Techniques for Complex Networks, Cambridge Univ. Press, NY.

[Mey22] Meyn, S., 2022. Control Systems and Reinforcement Learning, Cambridge Univ. Press, NY.

[Min22] Minorsky, N., 1922. 'Directional Stability of Automatically Steered Bodies,' J. Amer. Soc. Naval Eng.,Vol. 34, pp. 280-309.

[MoL99] Morari, M., and Lee, J. H., 1999. 'Model Predictive Control: Past, Present, and Future,' Computers and Chemical Engineering, Vol. 23, pp. 667-682.

[Mon17] Montgomery, D. C., 2017. Design and Analysis of Experiments, J. Wiley.

[Mor25] Morari, M., 2025. 'Model Predictive Control: The Genesis of an Idea,' IEEE Control Systems [Histories of Control], Vol. 45, pp. 86-88.

[MuM24] Muller, J., and Montufar, G., 2024. 'Geometry and Convergence of Natural Policy Gradient Methods,' Information Geometry, Vol. 7, pp. 485-523.

[Mun14] Munos, R., 2014. 'From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning,' Foundations and Trends in Machine Learning, Vol. 7, pp. 1-129.

[NHR99] Ng, A. Y., Harada, D., and Russell, S. J., 1999. 'Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping,' in Proc. of the 16th International Conference on Machine Learning, pp. 278-287.

[NMT13] Nayyar, A., Mahajan, A., and Teneketzis, D., 2013. 'Decentralized Stochastic Control with Partial History Sharing: A Common Information Approach,' IEEE Transactions on Automatic Control, Vol. 58, pp. 1644-1658.

[NSE19] Nozhati, S., Sarkale, Y., Ellingwood, B., Chong, E. K., and Mahmoud, H., 2019. 'Near-Optimal Planning Using Approximate Dynamic Programming to Enhance Post-Hazard Community Resilience Management,' Reliability Engineering and System Safety, Vol. 181, pp. 116-126.

[NSW05] Nedich, A., Schneider, M. K., and Washburn, R. B., 2005. 'Farsighted Sensor Management Strategies for Move/Stop Tracking,' in Proc. of 7th International Confer-

ence on Information Fusion, Vol. 1, pp. 566-573.

[NaA12] Narendra, K. S., and Annaswamy, A. M., 2012. Stable Adaptive Systems, Courier Corporation.

[NaT19] Nayyar, A., and Teneketzis, D., 2019. 'Common Knowledge and Sequential Team Problems,' IEEE Trans. on Automatic Control, Vol. 64, pp. 5108-5115.

[Nas50] Nash, J., 1950. Non-cooperative Games, Unpublished PhD Dissertation, Princeton University.

[NeS17] Nesterov, Y., and Spokoiny, V., 2017. 'Random Gradient-Free Minimization of Convex Functions,' Foundations of Computational Mathematics, Vol. 17, pp. 527-566.

[Ned11] Nedi­ c, A., 2011. 'Random Algorithms for Convex Minimization Problems,' Math. Programming, Ser. B, Vol. 129, pp. 225-253.

[Nes05] Nesterov, Y., 2005. 'Smooth Minimization of Nonsmooth Functions,' Math. Programming, Vol. 103, pp. 127-152.

[NoS09] Novoa, C., and Storer, R., 2009. 'An Approximate Dynamic Programming Approach for the Vehicle Routing Problem with Stochastic Demands,' European J. of Operational Research, Vol. 196, pp. 509-515.

[OrS02] Ormoneit, D., and Sen, S., 2002. 'Kernel-Based Reinforcement Learning,' Machine Learning, Vol. 49, pp. 161-178.

[PBK08] Patek, S. D., Breton, M., and Kovatchev, B. P., 2008. 'Rollout Policies for Control of Blood Glucose,' XIV Latin Ibero-American Congress on Operations Research - Book of Extended Abstracts.

[PDB92] Pattipati, K. R., Deb, S., Bar-Shalom, Y., and Washburn, R. B., 1992. 'A New Relaxation Algorithm and Passive Sensor Data Association,' IEEE Trans. Automatic Control, Vol. 37, pp. 198-213.

[PDC14] Pillonetto, G., Dinuzzo, F., Chen, T., De Nicolao, G., and Ljung, L., 2014. 'Kernel Methods in System Identification, Machine Learning and Function Estimation: A Survey,' Automatica, Vol. 50, pp. 657-682.

[PPB01] Popp, R. L., Pattipati, K. R., and Bar-Shalom, Y., 2001. ' m -Best SD Assignment Algorithm with Application to Multitarget Tracking,' IEEE Transactions on Aerospace and Electronic Systems, Vol. 37, pp. 22-39.

[PSC22] Paulson, J. A., Sonouifar, F., and Chakrabarty, A., 2022. 'E ffi cient MultiStep Lookahead Bayesian Optimization with Local Search Constraints,' IEEE Conf. on Decision and Control, pp. 123-129.

[PPG16] Perolat, J., Piot, B., Geist, M., Scherrer, B., and Pietquin, O., 2016. 'Softened Approximate Policy Iteration for Markov Games,' in Proc. International Conference on Machine Learning, pp. 1860-1868.

[PSP15] Perolat, J., Scherrer, B., Piot, B., and Pietquin, O., 2015. 'Approximate Dynamic Programming for Two-Player Zero-Sum Markov Games,' in Proc. International Conference on Machine Learning, pp. 1321-1329.

[PaB99] Patek, S. D., and Bertsekas, D. P., 1999. 'Stochastic Shortest Path Games,' SIAM J. on Control and Optimization, Vol. 37, pp. 804-824.

[PaR12] Papahristou, N., and Refanidis, I., 2012. 'On the Design and Training of Bots to Play Backgammon Variants,' in IFIP International Conference on Artificial Intelligence Applications and Innovations, pp. 78-87.

[PaT00] Paschalidis, I. C., and Tsitsiklis, J. N., 2000. 'Congestion-Dependent Pricing of Network Services,' IEEE/ACM Trans. on Networking, Vol. 8, pp. 171-184.

[PeG04] Peret, L., and Garcia, F., 2004. 'On-Line Search for Solving Markov Decision Processes via Heuristic Sampling,' in Proc. of the 16th European Conference on Artificial Intelligence, pp. 530-534.

[PeS08] Peters, J., and Schaal, S., 2008. 'Natural Actor-Critic,' Neurocomputing, Vol. 71, pp.1180-1190.

[PeW96] Peng, J., and Williams, R., 1996. 'Incremental Multi-Step Q-Learning,' Machine Learning, Vol. 22, pp. 283-290.

[PoA69] Pollatschek, M. A., and Avi-Itzhak, B., 1969. 'Algorithms for Stochastic Games with Geometrical Interpretation,' Management Science, Vol. 15, pp. 399-415.

[PoB04] Poupart, P., and Boutilier, C., 2004. 'Bounded Finite State Controllers,' in Advances in Neural Information Processing Systems, pp. 823-830.

[PoF08] Powell, W. B., and Frazier, P., 2008. 'Optimal Learning,' in State-of-the-Art Decision-Making Tools in the Information-Intensive Age, INFORMS, pp. 213-246.

[PoR97] Poore, A. B., and Robertson, A. J. A., 1997. 'New Lagrangian Relaxation Based Algorithm for a Class of Multidimensional Assignment Problems,' Computational Optimization and Applications, Vol. 8, pp. 129-150.

[PoR12] Powell, W. B., and Ryzhov, I. O., 2012. Optimal Learning, J. Wiley, NY.

[Pol79] Poljak, B. T., 1979. 'On Bertsekas' Method for Minimization of Composite Functions,' Internat. Symp. Systems Opt. Analysis, Bensoussan, A., and Lions, J. L., (Eds.), pp. 179-186, Springer-Verlag, Berlin and N. Y.

[Poo94] Poore, A. B., 1994. 'Multidimensional Assignment Formulation of Data Association Problems Arising from Multitarget Tracking and Multisensor Data Fusion,' Computational Optimization and Applications, Vol. 3, pp. 27-57.

[Pow11] Powell, W. B., 2011. Approximate Dynamic Programming: Solving the Curses of Dimensionality, 2nd Edition, J. Wiley and Sons, Hoboken, NJ.

[PrS01] Proakis, J. G., and Salehi, M., 2001. Communication Systems Engineering, Prentice-Hall, Englewood Cli ff s, NJ.

[PrS08] Proakis, J. G., and Salehi, M., 2008. Digital Communications, McGraw-Hill, NY.

[Pre95] Prekopa, A., 1995. Stochastic Programming, Kluwer, Boston.

[PuB78] Puterman, M. L., and Brumelle, S. L., 1978. 'The Analytic Theory of Policy Iteration,' in Dynamic Programming and Its Applications, M. L. Puterman (ed.), Academic Press, NY.

[PuB79] Puterman, M. L., and Brumelle, S. L., 1979. 'On the Convergence of Policy Iteration in Stationary Dynamic Programming,' Mathematics of Operations Research, Vol. 4, pp. 60-69.

[PuS78] Puterman, M. L., and Shin, M. C., 1978. 'Modified Policy Iteration Algorithms for Discounted Markov Decision Problems,' Management Sci., Vol. 24, pp. 1127-1137.

[PuS82] Puterman, M. L., and Shin, M. C., 1982. 'Action Elimination Procedures for Modified Policy Iteration Algorithms,' Operations Research, Vol. 30, pp. 301-318.

[Put94] Puterman, M. L., 1994. Markovian Decision Problems, J. Wiley, NY.

[QHS05] Queipo, N. V., Haftka, R. T., Shyy, W., Goel, T., Vaidyanathan, R., and

Tucker, P. K., 2005. 'Surrogate-Based Analysis and Optimization,' Progress in Aerospace Sciences, Vol. 41, pp. 1-28.

[QuL19] Qu, G., and Li, N., 'Exploiting Fast Decaying and Locality in Multi-Agent MDP with Tree Dependence Structure,' Proc. of 2019 CDC, Nice, France.

[RCR17] Rudi, A., Carratino, L., and Rosasco, L., 2017. 'Falkon: An Optimal Large Scale Kernel Method,' in Advances in Neural Information Processing Systems, pp. 38883898.

[RDM24] Ruoss, A., Del­ etang, G., Medapati, S., Grau-Moya, J., Wenliang, L. K., Catt, E., Reid, J., and Genewein, T., 2024. 'Grandmaster-Level Chess Without Search,' arXiv:2402.04494.

[RGG21] Rim­ el­ e, A., Grangier, P., Gamache, M., Gendreau, M., and Rousseau, L. M., 2021. 'E-Commerce Warehousing: Learning a Storage Policy,' arXiv:2101.08828.

[RMD17] Rawlings, J. B., Mayne, D. Q., and Diehl, M. M., 2017. Model Predictive Control: Theory, Computation, and Design, 2nd Ed., Nob Hill Publishing.

[RPF12] Ryzhov, I. O., Powell, W. B., and Frazier, P. I., 2012. 'The Knowledge Gradient Algorithm for a General Class of Online Learning Problems,' Operations Research, Vol. 60, pp. 180-195.

[RPW91] Rogers, D. F., Plante, R. D., Wong, R. T., and Evans, J. R., 1991. 'Aggregation and Disaggregation Techniques and Methodology in Optimization,' Operations Research, Vol. 39, pp. 553-582.

[RRT78] Richalet, J., Rault, A., Testud, J. L., and Papon, J., 1978. 'Model Predictive Heuristic Control,' Automatica, Vol. 14, pp. 413-428.

[RSM08] Reisinger, J., Stone, P., and Miikkulainen, R., 2008. 'Online Kernel Selection for Bayesian Reinforcement Learning,' in Proc. of the 25th International Conference on Machine Learning, pp. 816-823.

[RST23] Rusmevichientong, P., Sumida, M., Topaloglu, H., and Bai, Y., 2023. 'Revenue Management with Heterogeneous Resources: Unit Resource Capacities, Advance Bookings, and Itineraries over Time Intervals,' Operations Research, Articles in Advance.

[RaF91] Raghavan, T. E. S., and Filar, J. A., 1991. 'Algorithms for Stochastic Games - A Survey,' Zeitschrift fur Operations Research, Vol. 35, pp. 437-472.

[RaM93] Rawlings, J. B., and Muske, K. R., 1993. 'Stability of Constrained Receding Horizon Control,' IEEE Trans. Automatic Control, Vol. 38, pp. 1512-1516.

[RaR17] Rawlings, J. B., and Risbeck, M. J., 2017. 'Model Predictive Control with Discrete Actuators: Theory and Application,' Automatica, Vol. 78, pp. 258-265.

[RaW06] Rasmussen, C. E., and Williams, C. K., 2006. Gaussian Processes for Machine Learning, MIT Press, Cambridge, MA.

[Rab89] Rabiner, L. R., 1989. 'A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition,' Proc. of the IEEE, Vol. 77, pp. 257-286.

[Rad62] Radner, R., 1962. 'Team Decision Problems,' Ann. Math. Statist., Vol. 33, pp. 857-881.

[Ras63] Rastrigin, R. A., 1963. 'About Convergence of Random Search Method in Extremal Control of Multi-Parameter Systems,' Avtomat. i Telemekh., Vol. 24, pp. 14671473.

[RoB17] Rosolia, U., and Borrelli, F., 2017. 'Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework,' IEEE Trans. on Automatic Control,

Vol. 63, pp. 1883-1896.

[RoB19] Rosolia, U., and Borrelli, F., 2019. 'Sample-Based Learning Model Predictive Control for Linear Uncertain Systems,' 58th Conference on Decision and Control (CDC), pp. 2702-2707.

[RoM82] Rouhani, R., and Mehra, R. K., 1982. 'Model Algorithmic Control (MAC); Basic Theoretical Properties,' Automatica, Vol. 18, pp. 401-414.

[Rob52] Robbins, H., 1952. 'Some Aspects of the Sequential Design of Experiments,' Bulletin of the American Mathematical Society, Vol. 58, pp. 527-535.

[Ros70] Ross, S. M., 1970. Applied Probability Models with Optimization Applications, Holden-Day, San Francisco, CA.

[Ros12] Ross, S. M., 2012. Simulation, 5th Edition, Academic Press, Orlando, Fla.

[Rot79] Rothblum, U. G., 1979. 'Iterated Successive Approximation for Sequential Decision Processes,' in Stochastic Control and Optimization, by J. W. B. van Overhagen and H. C. Tijms (eds), Vrije University, Amsterdam.

[RuK04] Rubinstein, R. Y., and Kroese, D. P., 2004. The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Springer, NY.

[RuK13] Rubinstein, R. Y., and Kroese, D. P., 2013. The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning, Springer Science and Business Media.

[RuK16] Rubinstein, R. Y., and Kroese, D. P., 2016. Simulation and the Monte Carlo Method, 3rd Edition, J. Wiley, NY.

[RuN94] Rummery, G. A., and Niranjan, M., 1994. 'On-Line Q-Learning Using Connectionist Systems,' University of Cambridge, England, Department of Engineering, TR-166.

[RuN16] Russell, S. J., and Norvig, P., 2016. Artificial Intelligence: A Modern Approach, Pearson Education Limited, Malaysia.

[RuS03] Ruszczynski, A., and Shapiro, A., 2003. 'Stochastic Programming Models,' in Handbooks in Operations Research and Management Science, Vol. 10, pp. 1-64.

[Rub69] Rubinstein, R. Y., 1969. Some Problems in Monte Carlo Optimization, Ph.D. Thesis.

[Rus88] Rust, J., 1988. 'Maximum Likelihood Estimation of Discrete Control Processes,' SIAM J. on Control and Optimization, Vol. 26, pp. 1006-1024.

[RyN25] Rybicki, B. W., and Nelson, J. K., 2025. 'Train O ffl ine, Refine Online: Improving Cognitive Tracking Radar Performance with Approximate Policy Iteration and Deep Neural Networks,' IEEE Transactions on Radar Systems, Vol. 3, pp. 57-70.

[SGC02] Savagaonkar, U., Givan, R., and Chong, E. K. P., 2002. 'Sampling Techniques for Zero-Sum, Discounted Markov Games,' in Proc. 40th Allerton Conference on Communication, Control and Computing, Monticello, Ill.

[SGG15] Scherrer, B., Ghavamzadeh, M., Gabillon, V., Lesner, B., and Geist, M., 2015. 'Approximate Modified Policy Iteration and its Application to the Game of Tetris,' J. of Machine Learning Research, Vol. 16, pp. 1629-1676.

[SHB15] Simroth, A., Holfeld, D., and Brunsch, R., 2015. 'Job Shop Production Planning under Uncertainty: A Monte Carlo Rollout Approach,' Proc. of the International Scientific and Practical Conference, Vol. 3, pp. 175-179.

[SHM16] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., and Dieleman, S., 2016. 'Mastering the Game of Go with Deep Neural Networks and Tree Search,' Nature, Vol. 529, pp. 484-489.

[SHS17] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., and Lillicrap, T., 2017. 'Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm,' arXiv:1712.01815.

[SHS24] Samani, F. S., Hammar, K., and Stadler, R., 2024. 'Online Policy Adaptation for Networked Systems using Rollout,' in NOMS 2024-2024 IEEE Network Operations and Management Symposium, pp. 1-9.

[SJJ95] Singh, S. P., Jaakkola, T., and Jordan, M. I., 1995. 'Reinforcement Learning with Soft State Aggregation,' in Advances in Neural Information Processing Systems 7, MIT Press, Cambridge, MA.

[SJL18] Soltanolkotabi, M., Javanmard, A., and Lee, J. D., 2018. 'Theoretical Insights into the Optimization Landscape of Over-Parameterized Shallow Neural Networks,' IEEE Trans. on Information Theory, Vol. 65, pp. 742-769.

[SKS12] Schmidt, M., Kim, D., and Sra, S., 2012. 'Projected Newton-type Methods in Machine Learning,' Optimization for Machine Learning, by S. Sra, S. Nowozin, and S. J. Wright, (eds.), MIT Press, Cambridge, MA.

[SLA12] Snoek, J., Larochelle, H., and Adams, R. P., 2012. 'Practical Bayesian Optimization of Machine Learning Algorithms,' in Advances in Neural Information Processing Systems, pp. 2951-2959.

[SLJ13] Sun, B., Luh, P. B., Jia, Q. S., Jiang, Z., Wang, F., and Song, C., 2013. 'Building Energy Management: Integrated Control of Active and Passive Heating, Cooling, Lighting, Shading, and Ventilation Systems,' IEEE Trans. on Automation Science and Engineering, Vol. 10, pp. 588-602.

[SLD24] Samani, F. S., Larsson, H., Damberg, S., Johnsson, A., and Stadler, R., 2024. 'Comparing Transfer Learning and Rollout for Policy Adaptation in a Changing Network Environment,' in 2024 IEEE Network Operations and Management Symposium.

[SMS99] Sutton, R. S., McAllester, D., Singh, S. P., and Mansour, Y., 1999. 'Policy Gradient Methods for Reinforcement Learning with Function Approximation,' NIPS, Denver, Colorado.

[SNC18] Sarkale, Y., Nozhati, S., Chong, E. K., Ellingwood, B. R., and Mahmoud, H., 2018. 'Solving Markov Decision Processes for Network-Level Post-Hazard Recovery via Simulation Optimization and Rollout,' in 2018 IEEE 14th International Conference on Automation Science and Engineering, pp. 906-912.

[SPS99] Sutton, R. S., Precup, D., and Singh, S., 1999. 'Between MDPs and SemiMDPs: A Framework for Temporal Abstraction in Reinforcement Learning,' Artificial Intelligence, Vol. 112, pp. 181-211.

[SSS17] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., and Chen, Y., 2017. 'Mastering the Game of Go Without Human Knowledge,' Nature, Vol. 550, pp. 354-359.

[SSW16] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and De Freitas, N., 2015. 'Taking the Human Out of the Loop: A Review of Bayesian Optimization,' Proc. of IEEE, Vol. 104, pp. 148-175.

[SWD17] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O., 2017. 'Proximal Policy Optimization Algorithms,' arXiv:1707.06347.

[SWM89] Sacks, J., Welch, W. J., Mitchell, T. J., and Wynn, H. P., 1989. 'Design and Analysis of Computer Experiments,' Statistical Science, pp. 409-423.

[SXX22] Shah, D., Xie, Q., and Xu, Z., 2022. 'Nonasymptotic Analysis of Monte Carlo Tree Search,' Operations Research, vol. 70, pp. 3234-3260.

[SYL17] Saldi, N., Yuksel, S., and Linder, T., 2017. 'Finite Model Approximations for Partially Observed Markov Decision Processes with Discounted Cost,' arXiv:1710.07009.

[SZL08] Sun, T., Zhao, Q., Lun, P., and Tomastik, R., 2008. 'Optimization of Joint Replacement Policies for Multipart Systems by a Rollout Framework,' IEEE Trans. on Automation Science and Engineering, Vol. 5, pp. 609-619.

[SaB11] Sastry, S., and Bodson, M., 2011. Adaptive Control: Stability, Convergence and Robustness, Courier Corporation.

[Sal21] Saldi, N., 2021. 'Regularized Stochastic Team Problems,' Systems and Control Letters, Vol. 149.

[Sas02] Sasena, M. J., 2002. Flexibility and E ffi ciency Enhancements for Constrained Global Design Optimization with Kriging Approximations, PhD Thesis, Univ. of Michigan.

[ScS02] Scholkopf, B., and Smola, A. J., 2002. Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond, MIT Press, Cambridge, MA.

[Sch13] Scherrer, B., 2013. 'Performance Bounds for Lambda Policy Iteration and Application to the Game of Tetris,' J. of Machine Learning Research, Vol. 14, pp. 1181-1227.

[Sco10] Scott, S. L., 2010. 'A Modern Bayesian Look at the Multi-Armed Bandit,' Applied Stochastic Models in Business and Industry, Vol. 26, pp. 639-658.

[Sec00] Secomandi, N., 2000. 'Comparing Neuro-Dynamic Programming Algorithms for the Vehicle Routing Problem with Stochastic Demands,' Computers and Operations Research, Vol. 27, pp. 1201-1225.

[Sec01] Secomandi, N., 2001. 'A Rollout Policy for the Vehicle Routing Problem with Stochastic Demands,' Operations Research, Vol. 49, pp. 796-802.

[Sec03] Secomandi, N., 2003. 'Analysis of a Rollout Approach to Sequencing Problems with Stochastic Routing Applications,' J. of Heuristics, Vol. 9, pp. 321-352.

[ShC04] Shawe-Taylor, J., and Cristianini, N., 2004. Kernel Methods for Pattern Analysis, Cambridge Univ. Press.

[Sha50] Shannon, C., 1950. 'Programming a Digital Computer for Playing Chess,' Phil. Mag., Vol. 41, pp. 356-375.

[Sha53] Shapley, L. S., 1953. 'Stochastic Games,' Proc. of the National Academy of Sciences, Vol. 39, pp. 1095-1100.

[SiB22] Silver, D., and Barreto, A., 2022. 'Simulation-Based Search,' in Proc. Int. Cong. Math, Vol. 6, pp. 4800-4819.

[SiK19] Singh, R., and Kumar, P. R., 2019. 'Optimal Decentralized Dynamic Policies for Video Streaming over Wireless Channels,' arXiv:1902.07418.

[SlL91] Slotine, J.-J. E., and Li, W., Applied Nonlinear Control, Prentice-Hall, Englewood Cli ff s, NJ.

[Spa92] Spall, J. C., 'Multivariate Stochastic Approximation Using a Simultaneous Per-

turbation Gradient Approximation,' IEEE Trans. on Automatic Control, Vol. pp. 332341.

[Spa03] Spall, J. C., 2003. Introduction to Stochastic Search and Optimization: Estimation, Simulation, and Control, J. Wiley, Hoboken, NJ.

[StW91] Stewart, B. S., and White, C. C., 1991. 'Multiobjective A ∗ ,' J. ACM, Vol. 38, pp. 775-814.

[Ste84] Stephanopoulos, G., 1984. Chemical Process Control, Prentice-Hall, Englewood Cli ff s, NJ.

[Ste94] Stengel, R. F., 1994. Optimal Control and Estimation, Courier Corporation.

[Ste25] Stephanopoulos, G., 2025. Chemical and Biological Process Dynamics and Control, Tetractys Editions, Winchester, MA.

[SuB18] Sutton, R., and Barto, A. G., 2018. Reinforcement Learning, 2nd Edition, MIT Press, Cambridge, MA.

[SuY19] Su, L., and Yang, P., 2019. 'On Learning Over-Parameterized Neural Networks: A Functional Approximation Perspective,' in Advances in Neural Information Processing Systems, pp. 2637-2646.

[Sun19] Sun, R., 2019. 'Optimization for Deep Learning: Theory and Algorithms,' arXiv:1912.08957.

[Swo69] Sworder, D., 1969. 'Feedback Control of a Class of Linear Systems with Jump Parameters,' IEEE Trans. on Automatic Control, Vol. 14, pp. 914.

[SzL06] Szita, I., and Lorinz, A., 2006. 'Learning Tetris Using the Noisy Cross-Entropy Method,' Neural Computation, Vol. 18, pp. 2936-2941.

[Sze10] Szepesvari, C., 2010. Algorithms for Reinforcement Learning, Morgan and Claypool Publishers, San Franscisco, CA.

[Sze11] Szepesvari, C., 2011. 'Least Squares Temporal Di ff erence Learning and Galerkin's Method,' Presentation at the Mini-Workshop: Mathematics of Machine Learning, Mathematisches Forschungsinstitut Oberwolfach.

[TBA86] Tsitsiklis, J. N., Bertsekas, D. P., and Athans, M., 1986. 'Distributed Asynchronous Deterministic and Stochastic Gradient Optimization Algorithms,' IEEE Trans. on Aut. Control, Vol. AC-31, pp. 803-812.

[TBP08] Tian, X., Bar-Shalom, Y., and Pattipati, K. R., 2008. 'Multi-Step Look-Ahead Policy for Autonomous Cooperative Surveillance by UAVs in Hostile Environments,' in 2008 47th IEEE Conference on Decision and Control, pp. 2438-2443.

[TBP21] Tuncel, Y., Bhat, G., Park, J., and Ogras, U., 2021. 'ECO: Enabling EnergyNeutral IoT Devices Through Runtime Allocation of Harvested Energy,' arXiv:2102.13605; also IEEE Internet of Things Journal, Vol. 9, 2021.

[TCW19] Tseng, W. J., Chen, J. C., Wu, I. C., and Wei, T. H., 2019. 'Comparison Training for Computer Chinese Chess,' IEEE Trans. on Games, Vol. 12, pp. 169-176.

[TGL13] Tesauro, G., Gondek, D. C., Lenchner, J., Fan, J., and Prager, J. M., 2013. 'Analysis of Watson's Strategies for Playing Jeopardy!,' J. of Artificial Intelligence Research, Vol. 47, pp. 205-251.

[TRV16] Tu, S., Roelofs, R., Venkataraman, S., and Recht, B., 2016. 'Large Scale Kernel Learning Using Block Coordinate Descent,' arXiv:1602.05310.

[TaL20] Tanzanakis, A., and Lygeros, J., 2020. 'Data-Driven Control of Unknown Systems: A Linear Programming Approach,' arXiv:2003.00779.

[TeG96] Tesauro, G., and Galperin, G. R., 1996. 'On-Line Policy Improvement Using Monte Carlo Search,' NIPS, Denver, CO.

[Tes89a] Tesauro, G. J., 1989. 'Neurogammon Wins Computer Olympiad,' Neural Computation, Vol. 1, pp. 321-323.

[Tes89b] Tesauro, G. J., 1989. 'Connectionist Learning of Expert Preferences by Comparison Training,' in Advances in Neural Information Processing Systems, pp. 99-106.

[Tes92] Tesauro, G. J., 1992. 'Practical Issues in Temporal Di ff erence Learning,' Machine Learning, Vol. 8, pp. 257-277.

[Tes94] Tesauro, G. J., 1994. 'TD-Gammon, a Self-Teaching Backgammon Program, Achieves Master-Level Play,' Neural Computation, Vol. 6, pp. 215-219.

[Tes95] Tesauro, G. J., 1995. 'Temporal Di ff erence Learning and TD-Gammon,' Communications of the ACM, Vol. 38, pp. 58-68.

[Tes01] Tesauro, G. J., 2001. 'Comparison Training of Chess Evaluation Functions,' in Machines that Learn to Play Games, Nova Science Publishers, pp. 117-130.

[Tes02] Tesauro, G. J., 2002. 'Programming Backgammon Using Self-Teaching Neural Nets,' Artificial Intelligence, Vol. 134, pp. 181-199.

[ThS09] Thiery, C., and Scherrer, B., 2009. 'Improvements on Learning Tetris with Cross-Entropy,' International Computer Games Association J., Vol. 32, pp. 23-33.

[Tol89] Tolwinski, B., 1989. 'Newton-Type Methods for Stochastic Games,' in Basar T. S., and Bernhard P. (eds), Di ff erential Games and Applications, Lecture Notes in Control and Information Sciences, Vol. 119, Springer, pp. 128-144.

[TsV96] Tsitsiklis, J. N., and Van Roy, B., 1996. 'Feature-Based Methods for Large-Scale Dynamic Programming,' Machine Learning, Vol. 22, pp. 59-94.

[Tse98] Tseng, P., 1998. 'Incremental Gradient(-Projection) Method with Momentum Term and Adaptive Stepsize Rule,' SIAM J. on Optimization, Vol. 8, pp. 506-531.

[Tsi94] Tsitsiklis, J. N., 1994. 'Asynchronous Stochastic Approximation and Q-Learning,' Machine Learning, Vol. 16, pp. 185-202.

[TuP03] Tu, F., and Pattipati, K. R., 2003. 'Rollout Strategies for Sequential Fault Diagnosis,' IEEE Trans. on Systems, Man and Cybernetics, Part A, pp. 86-99.

[UGM18] Ulmer, M. W., Goodson, J. C., Mattfeld, D. C., and Hennig, M., 2018. 'O ffl ineOnline Approximate Dynamic Programming for Dynamic Vehicle Routing with Stochastic Requests,' Transportation Science, Vol. 53, pp. 185-202.

[Ulm17] Ulmer, M. W., 2017. Approximate Dynamic Programming for Dynamic Vehicle Routing, Springer, Berlin.

[VBC19] Vinyals, O., Babuschkin, I., Czarnecki, W. M., and thirty nine more authors, 2019. 'Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning,' Nature, Vol. 575, p. 350.

[VLK21] Vaswani, S., Laradji, I. H., Kunstner, F., Meng, S. Y., Schmidt, M., and Lacoste-Julien, S., 2021. 'Adaptive Gradient Methods Converge Faster with OverParameterization (but you should do a line search),' arXiv:2006.06835.

[VPA09] Vrabie, D., Pastravanu, O., Abu-Khalaf, M., and Lewis, F. L., 2009. 'Adaptive Optimal Control for Continuous-Time Linear Systems Based on Policy Iteration,'

Automatica, Vol. 45, pp. 477-484.

[VVL13] Vrabie, D., Vamvoudakis, K. G., and Lewis, F. L., 2013. Optimal Adaptive Control and Di ff erential Games by Reinforcement Learning Principles, The Institution of Engineering and Technology, London.

[VaH20] Van Engelen, J. E., and Hoos, H. H., 2020. 'A survey on Semi-Supervised Learning,' Machine Learning, Vol. 109, pp. 373-440.

[Van76] Van Nunen, J. A., 1976. Contracting Markov Decision Processes, Mathematical Centre Report, Amsterdam.

[Van78] van der Wal, J., 1978. 'Discounted Markov Games: Generalized Policy Iteration Method,' J. of Optimization Theory and Applications, Vol. 25, pp. 125-138.

[Van95] van Roy, B. Feature-Based Methods for Large Scale Dynamic Programming, MS thesis, Massachusetts Institute of Technology.

[VeM23] Vertovec, N., and Margellos, K., 2023. 'State Aggregation for Distributed Value Iteration in Dynamic Programming,' arXiv:2303.10675; also IEEE Control Systems Letters, Vol. 7, pp. 2269-2274.

[Vit67] Viterbi, A. J., 1967. 'Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm,' IEEE Trans. on Info. Theory, Vol. IT-13, pp. 260-269.

[WCG02] Wu, G., Chong, E. K. P., and Givan, R. L., 2002. 'Burst-Level Congestion Control Using Hindsight Optimization,' IEEE Transactions on Aut. Control, Vol. 47, pp. 979-991.

[WCG03] Wu, G., Chong, E. K. P., and Givan, R. L., 2003. 'Congestion Control Using Policy Rollout,' Proc. 2nd IEEE CDC, Maui, Hawaii, pp. 4825-4830.

[WGB22] Wang, T. T., Gleave, A., Belrose, N., Tseng, T., Miller, J., Dennis, M. D., Duan, Y., Pogrebniak, V., Levine, S., and Russell, S., 2022. 'Adversarial Policies Beat Professional-Level Go AIs,' arXiv:2211.00241.

[WGP23] Weber, J., Giriyan, D., Parkar, D., Richa, A., and Bertsekas, D. P., 'Distributed Online Rollout for Multivehicle Routing in Unmapped Environments,' arXiv: 2305.11596v1.

[WOB15] Wang, Y., O'Donoghue, B., and Boyd, S., 2015. 'Approximate Dynamic Programming via Iterated Bellman Inequalities,' International J. of Robust and Nonlinear Control, Vol. 25, pp. 1472-1496.

[WTL25] Wang, Q., Tong, X., Li, Y., Wang, C. and Zhang, C., 2025. 'Integrated Scheduling Optimization for Automated Container Terminal: A Reinforcement LearningBased Approach,' IEEE Trans. on Intelligent Transportation Systems.

[WaB13a] Wang, M., and Bertsekas, D. P., 2013. 'Stabilization of Stochastic Iterative Methods for Singular and Nearly Singular Linear Systems,' Mathematics of Operations Research, Vol. 39, pp. 1-30.

[WaB13b] Wang, M., and Bertsekas, D. P., 2013. 'Convergence of Iterative SimulationBased Methods for Singular Linear Systems,' Stochastic Systems, Vol. 3, pp. 39-96.

[WaB14] Wang, M., and Bertsekas, D. P., 2014. 'Incremental Constraint Projection Methods for Variational Inequalities,' Mathematical Programming, pp. 1-43.

[WaB16] Wang, M., and Bertsekas, D. P., 2016. 'Stochastic First-Order Methods with Random Constraint Projection,' SIAM Journal on Optimization, Vol. 26, pp. 681-717.

[WaP17] Wang, J., and Paschalidis, I. C., 2017. 'An Actor-Critic Algorithm with SecondOrder Actor and Critic,' IEEE Trans. on Automatic Control, Vol. 62, pp. 2689-2703.

[WaS00] de Waal, P. R., and van Schuppen, J. H., 2000. 'A Class of Team Problems with Discrete Action Spaces: Optimality Conditions Based on Multimodularity,' SIAM J. on Control and Optimization, Vol. 38, pp. 875-892.

[Wat89] Watkins, C. J. C. H., Learning from Delayed Rewards, Ph.D. Thesis, Cambridge Univ., England.

[WeB99] Weaver, L., and Baxter, J., 1999. 'Learning from State Di ff erences: STD( λ ),' Tech. Report, Dept. of Computer Science, Australian National University.

[WeV17] Westhead, D. R., and Vijayabaskar, M. S., 2017. Hidden Markov Models, Springer, Berlin.

[WhS94] White, C. C., and Scherer, W. T., 1994. 'Finite-Memory Suboptimal Design for Partially Observed Markov Decision Processes,' Operations Research, Vol. 42, pp. 439-455.

[Whi82] Whittle, P., 1982. Optimization Over Time, Wiley, NY, Vol. 1, 1982, Vol. 2, 1983.

[Whi88] Whittle, P., 1988. 'Restless Bandits: Activity Allocation in a Changing World,' J. of Applied Probability, pp. 287-298.

[Whi91] White, C. C., 1991. 'A Survey of Solution Techniques for the Partially Observed Markov Decision Process,' Annals of Operations Research, Vol. 32, pp. 215-230.

[WiB93] Williams, R. J., and Baird, L. C., 1993. 'Analysis of Some Incremental Variants of Policy Iteration: First Steps Toward Understanding Actor-Critic Learning Systems,' Report NU-CCS-93-11, College of Computer Science, Northeastern University, Boston, MA.

[WiS98] Wiering, M., and Schmidhuber, J., 1998. 'Fast Online Q( λ ),' Machine Learning, Vol. 33, pp. 105-115.

[Wie03] Wiewiora, E., 2003. 'Potential-Based Shaping and Q-Value Initialization are Equivalent,' J. of Artificial Intelligence Research, Vol. 19, pp. 205-208.

[Wil92] Williams, R. J., 1992. 'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning,' Machine Learning, Vol. 8, pp. 229-256.

[Wit66] Witsenhausen, H. S., 1966. Minimax Control of Uncertain Systems, Ph.D. thesis, MIT.

[Wit68] Witsenhausen, H. S., 1968. 'A Counterexample in Stochastic Optimum Control,' SIAM Journal on Control, Vol. 6, pp. 131-147.

[Wit71a] Witsenhausen, H. S., 1971. 'On Information Structures, Feedback and Causality,' SIAM J. Control, Vol. 9, pp. 149-160.

[Wit71b] Witsenhausen, H., 1971. 'Separation of Estimation and Control for Discrete Time Systems,' Proceedings of the IEEE, Vol. 59, pp. 1557-1566.

[Won70] Wonham, W. M., 1970. 'Random Di ff erential Equations in Control Theory,' Probabilistic Analysis and Related Topics, Vol. 2, pp. 131212.

[WuX24] Wu, H., and Xie, Y., 2024. 'A Study on Two-Metric Projection Methods,' arXiv preprint arXiv:2409.05321.

[WuZ23] Wu, Y., and Zeng, B., 2023. 'Dynamic Parcel Pick-Up Routing Problem with Prioritized Customers and Constrained Capacity via Lower-Bound-Based Rollout Ap-

proach,' Computers and Operations Research, Vol. 154.

[XCB23] Xiao, W., Cassandras, C. G., and Belta, C., 2023. Safe Autonomy with Control Barrier Functions: Theory and Applications, Springer.

[XLT24] Xu, X., Li, M., Tao, C., Shen, T., Cheng, R., Li, J., Xu, C., Tao, D., and Zhou, T., 2024. 'A Survey on Knowledge Distillation of Large Language Models,' arXiv:2402.13116.

[XiW21] Xie, Y., and Wright, S. J., 2021. 'Complexity of Projected Newton Methods for Bound-Constrained Optimization,' arXiv preprint arXiv:2103.15989.

[XiW24] Xie, Y., and Wright, S. J., 2024. 'Complexity of a Projected Newton-CG Method for Optimization with Bounds,' Mathematical Programming, Vol. 207, pp. 107144.

[Xia22] Xiao, L., 2022. 'On the Convergence Rates of Policy Gradient Methods,' J. of Machine Learning Research, Vol. 23, pp. 1-36.

[YDR04] Yan, X., Diaconis, P., Rusmevichientong, P., and Van Roy, B., 2004. 'Solitaire: Man Versus Machine,' Advances in Neural Information Processing Systems, Vol. 17, pp. 1553-1560.

[YXK24] Yilmaz, M. B., Xiang, L., and Klein, A., 2024. 'Joint Beamforming and Trajectory Optimization for UAV-Aided ISAC with Dipole Antenna Array,' Report.

[YYM20] Yu, L., Yang, H., Miao, L., and Zhang, C., 2019. 'Rollout Algorithms for Resource Allocation in Humanitarian Logistics,' IISE Transactions, Vol. 51, pp. 887909.

[Yar17] Yarotsky, D., 2017. 'Error Bounds for Approximations with Deep ReLU Networks,' Neural Networks, Vol. 94, pp. 103-114.

[YuB04] Yu, H., and Bertsekas, D. P., 2004. 'Discretized Approximations for POMDP with Average Cost,' Proc. of the 20th Conference on Uncertainty in Artificial Intelligence, Ban ff , Canada.

[YuB08] Yu, H., and Bertsekas, D. P., 2008. 'On Near-Optimality of the Set of FiniteState Controllers for Average Cost POMDP,' Math. of Operations Research, Vol. 33, pp. 1-11.

[YuB09] Yu, H., and Bertsekas, D. P., 2009. 'Basis Function Adaptation Methods for Cost Approximation in MDP,' Proceedings of 2009 IEEE Symposium on Approximate Dynamic Programming and Reinforcement Learning (ADPRL 2009), Nashville, Tenn.

[YuB10] Yu, H., and Bertsekas, D. P., 2010. 'Error Bounds for Approximations from Projected Linear Equations,' Math. of Operations Research, Vol. 35, pp. 306-329.

[YuB12] Yu, H., and Bertsekas, D. P., 2012. 'Weighted Bellman Equations and their Applications in Dynamic Programming,' Lab. for Information and Decision Systems Report LIDS-P-2876, MIT.

[YuB13] Yu, H., and Bertsekas, D. P., 2013. 'Q-Learning and Policy Iteration Algorithms for Stochastic Shortest Path Problems,' Annals of Operations Research, Vol. 208, pp. 95-132.

[YuB15] Yu, H., and Bertsekas, D. P., 2015. 'A Mixed Value and Policy Iteration Method for Stochastic Control with Universally Measurable Policies,' Math. of OR, Vol. 40, pp. 926-968.

[YuK20] Yue, X., and Kontar, R. A., 2020. 'Lookahead Bayesian Optimization via Rollout: Guarantees and Sequential Rolling Horizons,' arXiv:1911.01004.

[Yu05] Yu, H., 2005. 'A Function Approximation Approach to Estimation of Policy Gradient for POMDP with Structured Policies,' Proc. of the 21st Conference on Uncertainty in Artificial Intelligence, Edinburgh, Scotland; also arXiv:1207.1421, 2012.

[Yu14] Yu, H., 2014. 'Stochastic Shortest Path Games and Q-Learning,' arXiv:1412.8570.

[Yua19] Yuanhong, L. I. U., 2019. 'Optimal Selection of Tests for Fault Detection and Isolation in Multi-Operating Mode System,' Journal of Systems Engineering and Electronics, Vol. 30, pp. 425-434.

[ZBH16] Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O., 2016. 'Understanding Deep Learning Requires Rethinking Generalization,' arXiv:1611.03530.

[ZBH21] Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O., 2021. 'Understanding Deep Learning (Still) Requires Rethinking Generalization,' Communications of the ACM, Vol. 64, pp. 107-115.

[ZHU25] Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., and Thakker, U., 2025. 'Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models,' arXiv:2510.04618.

[ZKZ20] Zhang, K., Koppel, A., Zhu, H., and Basar, T., 2020. 'Global Convergence of Policy Gradient Methods to (Almost) Locally Optimal Policies,' SIAM J. on Control and Optimization, Vol. 58, pp. 3586-3612.

[ZLZ24] Zhang, Q., Liu, Y., Zhang, B., and Huang, H. Z., 2024. 'Selective Maintenance Optimization Under Limited Maintenance Capacities: A Machine Learning-Enhanced Approximate Dynamic Programming,' IEEE Transactions on Reliability.

[ZOT18] Zhang, S., Ohlmann, J. W., and Thomas, B. W., 2018. 'Dynamic Orienteering on a Network of Queues,' Transportation Science, Vol. 52, pp. 691-706.

[ZSG20] Zoppoli, R., Sanguineti, M., Gnecco, G., and Parisini, T., 2020. Neural Approximations for Optimal Control and Decision, Springer.

[ZYB21] Zhang, K., Yang, Z., and Basar, T., 2021. 'Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms,' Handbook of Reinforcement Learning and Control, pp. 321-384.

[ZhG22] Zhu, X., and Goldberg, A. B., 2022. Introduction to Semi-Supervised Learning, Springer.

[ZuS81] Zuker, M., and Stiegler, P., 1981. 'Optimal Computer Folding of Larger RNA Sequences Using Thermodynamics and Auxiliary Information,' Nucleic Acids Res., Vol. 9, pp. 133-148.

## Neuro-Dynamic Programming Dimitri P. Bertsekas and John N. Tsitsiklis

## Athena Scientific, 1996 512 pp., hardcover, ISBN 1-886529-10-8

This is the first textbook that fully explains the neuro-dynamic programming/reinforcement learning methodology, a breakthrough in the practical application of neural networks and dynamic programming to complex problems of planning, optimal decision making, and intelligent control.

From the review by George Cybenko for IEEE Computational Science and Engineering, May 1998:

'Neuro-Dynamic Programming is a remarkable monograph that integrates a sweeping mathematical and computational landscape into a coherent body of rigorous knowledge. The topics are current, the writing is clear and to the point, the examples are comprehensive and the historical notes and comments are scholarly.'

'In this monograph, Bertsekas and Tsitsiklis have performed a Herculean task that will be studied and appreciated by generations to come. I strongly recommend it to scientists and engineers eager to seriously understand the mathematics and computations behind modern behavioral machine learning.'

Among its special features, the book:

- ÷ Describes and unifies a large number of NDP methods, including several that are new
- ÷ Describes new approaches to formulation and solution of important problems in stochastic optimal control, sequential decision making, and discrete optimization
- ÷ Rigorously explains the mathematical principles behind NDP
- ÷ Illustrates through examples and case studies the practical application of NDP to complex problems from optimal resource allocation, optimal feedback control, data communications, game playing, and combinatorial optimization
- ÷ Presents extensive background and new research material on dynamic programming and neural network training

Neuro-Dynamic Programming is the winner of the 1997 INFORMS CSTS prize for research excellence in the interface between Operations Research and Computer Science

## Reinforcement Learning and Optimal Control Dimitri P. Bertsekas

Athena Scientific, 2019

388 pp., hardcover, ISBN 978-1-886529-39-7

This book explores the common boundary between optimal control and artificial intelligence, as it relates to reinforcement learning and simulation-based neural network methods. These are popular fields with many applications, which can provide approximate solutions to challenging sequential decision problems and large-scale dynamic programming (DP). The aim of the book is to organize coherently the broad mosaic of methods in these fields, which have a solid analytical and logical foundation, and have also proved successful in practice.

The book discusses both approximation in value space and approximation in policy space. It adopts a gradual expository approach, which proceeds along four directions:

- ÷ From exact DP to approximate DP: We first discuss exact DP algorithms, explain why they may be di ffi cult to implement, and then use them as the basis for approximations.
- ÷ From finite horizon to infinite horizon problems: We first discuss finite horizon exact and approximate DP methodologies, which are intuitive and mathematically simple, and then progress to infinite horizon problems.
- ÷ From model-based to model-free implementations: We first discuss model-based implementations, and then we identify schemes that can be appropriately modified to work with a simulator.

The mathematical style of this book is somewhat di ff erent from the one of the author's DP books, and the 1996 neuro-dynamic programming (NDP) research monograph, written jointly with John Tsitsiklis. While we provide a rigorous, albeit short, mathematical account of the theory of finite and infinite horizon DP, and some fundamental approximation methods, we rely more on intuitive explanations and less on proof-based insights. Moreover, our mathematical requirements are quite modest: calculus, a minimal use of matrix-vector algebra, and elementary probability (mathematically complicated arguments involving laws of large numbers and stochastic convergence are bypassed in favor of intuitive explanations).

The book is supported by on-line video lectures and slides, as well as new research material, some of which has been covered in the present monograph.

## Rollout, Policy Iteration, and Distributed Reinforcement Learning

## Dimitri P. Bertsekas

## Athena Scientific, 2020

480 pp., hardcover, ISBN 978-1-886529-07-6

This book develops in greater depth some of the methods from the author's Reinforcement Learning and Optimal Control textbook (Athena Scientific, 2019). It presents new research, relating to rollout algorithms, policy iteration, multiagent systems, partitioned architectures, and distributed asynchronous computation.

The application of the methodology to challenging discrete optimization problems, such as routing, scheduling, assignment, and mixed integer programming, including the use of neural network approximations within these contexts, is also discussed.

Much of the new research is inspired by the remarkable AlphaZero chess program, where policy iteration, value and policy networks, approximate lookahead minimization, and parallel computation all play an important role.

Among its special features, the book:

- ÷ Presents new research relating to distributed asynchronous computation, partitioned architectures, and multiagent systems, with application to challenging large scale optimization problems, such as combinatorial/discrete optimization, as well as partially observed Markov decision problems.
- ÷ Describes variants of rollout and policy iteration for problems with a multiagent structure, which allow the dramatic reduction of the computational requirements for lookahead minimization.
- ÷ Establishes connections of rollout algorithms and model predictive control, one of the most prominent control system design methodology.
- ÷ Expands the coverage of some research areas discussed in the author's 2019 textbook Reinforcement Learning and Optimal Control.
- ÷ Provides the mathematical analysis that supports the Newton step interpretations and the conclusions of the present book.

The book is supported by on-line video lectures and slides, as well as new research material, some of which has been covered in the present monograph.