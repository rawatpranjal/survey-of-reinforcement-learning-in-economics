<!-- image -->

©

## A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes

## MICHAEL KEARNS ∗

mkearns@cis.upenn.edu

Department of Computer and Information Science, University of Pennsylvania, Moore School Building, 200 South 33rd Street, Philadelphia, PA 19104-6389, USA

## YISHAY MANSOUR

mansour@math.tau.ac.il

Department of Computer Science, Tel Aviv University, 69978 Tel Aviv, Israel

ANDREW Y. NG

ang@cs.berkeley.edu

Department of Computer Science, University of Berkeley, Berkeley, CA 94704, USA

Editor:

Leslie Kaelbling

Abstract. Acritical issue for the application of Markov decision processes (MDPs) to realistic problems is how the complexity of planning scales with the size of the MDP. In stochastic environments with very large or infinite state spaces, traditional planning and reinforcement learning algorithms may be inapplicable, since their running time typically grows linearly with the state space size in the worst case. In this paper we present a new algorithm that, given only a generative model (a natural and common type of simulator) for an arbitrary MDP, performs on-line, near-optimal planning with a per-state running time that has no dependence on the number of states. The running time is exponential in the horizon time (which depends only on the discount factor γ and the desired degree of approximation to the optimal policy). Our algorithm thus provides a different complexity trade-off than classical algorithms such as value iteration-rather than scaling linearly in both horizon time and state space size, our running time trades an exponential dependence on the former in exchange for no dependence on the latter.

Our algorithm is based on the idea of sparse sampling . We prove that a randomly sampled look-ahead tree that covers only a vanishing fraction of the full look-ahead tree nevertheless suffices to compute near-optimal actions from any state of an MDP. Practical implementations of the algorithm are discussed, and we draw ties to our related recent results on finding a near-best strategy from a given class of strategies in very large partially observable MDPs (Kearns, Mansour, &amp; Ng. Neural information processing systems 13, to appear).

Keywords:

reinforcement learning, Markov decision processes, planning

## 1. Introduction

In the past decade, Markov decision processes (MDPs) and reinforcement learning have become a standard framework for planning and learning under uncertainty within the artificial intelligence literature. The desire to attack problems of increasing complexity with this formalism has recently led researchers to focus particular attention on the case of (exponentially or even infinitely) large state spaces. A number of interesting algorithmic and representational suggestions have been made for coping with such large MDPs. Function

∗ This research was conducted while the author was at AT&amp;T Labs.

approximation (Sutton &amp; Barto, 1998) is a well-studied approach to learning value functions in large state spaces, and many authors have recently begun to study the properties of large MDPs that enjoy compact representations, such as MDPs in which the state transition probabilities factor into a small number of components (Boutilier, Dearden, &amp; Goldszmidt, 1995; Meuleau et al., 1998; Koller &amp; Parr, 1999).

In this paper, we are interested in the problem of computing a near-optimal policy in a large or infinite MDP that is given-that is, we are interested in planning . It should be clear that as we consider very large MDPs, the classical planning assumption that the MDP is given explicitly by tables of rewards and transition probabilities becomes infeasible. One approach to this representational difficulty is to assume that the MDP has some special structure that permits compact representation (such as the factored transition probabilities mentioned above), and to design special-purpose planning algorithms that exploit this structure.

Here we take a slightly different approach. We consider a setting in which our planning algorithm is given access to a generative model , or simulator, of the MDP. Informally, this is a 'black box' to which we can give any state-action pair ( s , a ) , and receive in return a randomly sampled next state and reward from the distributions associated with ( s , a ) . Generative models have been used in conjunction with some function approximation schemes (Sutton &amp; Barto, 1998), and are a natural way in which a large MDP might be specified. Moreover, they are more general than most structured representations, in the sense that many structured representations (such as factored models (Boutilier, Dearden, &amp; Goldszmidt, 1995; Meuleau et al., 1998; Koller &amp; Parr, 1999)) usually provide an efficient way of implementing a generative model. Note also that generative models also provide less information than explicit tables of probabilities, but more information than a single continuous trajectory of experience generated according to some exploration policy, and so we view results obtained via generative models as blurring the distinction between what is typically called 'planning' and 'learning' in MDPs.

Our main result is a new algorithm that accesses the given generative model to perform near-optimal planning in an on-line fashion. By 'on-line,' we mean that, similar to real-time search methods (Korf, 1990; Barto, Bradtke, &amp; Singh, 1995; Koenig &amp; Simmons, 1998), our algorithm's computation at any time is focused on computing an actions for a single 'current state,' and planning is interleaved with taking actions. More precisely, given any state s , the algorithm uses the generative model to draw samples for many state-action pairs, and uses these samples to compute a near-optimal action from s , which is then executed. The amount of time required to compute a near-optimal action from any particular state s has no dependence on the number of states in the MDP, even though the next-state distributions from s may be very diffuse (that is, have large support). The key to our analysis is in showing that appropriate sparse sampling suffices to construct enough information about the environment near s to compute a near-optimal action. The analysis relies on a combination of Bellman equation calculations, which are standard in reinforcement learning, and uniform convergence arguments, which are standard in supervised learning; this combination of techniques was first applied in Kearns and Singh (1999). As mentioned, the running time required at each state does have an exponential dependence on the horizon time, which we show to be unavoidable without further assumptions. However, our results leave open the

possiblity of an algorithm that runs in time polynomial in the accuracy parameter, which remains an important open problem.

Note that one can view our planning algorithm as simply implementing a (stochastic) policy-a policy that happens to use a generative model as a subroutine. In this sense, if we view the generative model as providing a 'compact' representation of the MDP, our algorithm provides a correspondingly 'compact' representation of a near-optimal policy. We view our result as complementary to work that proposes and exploits particular compact representations of MDPs (Meuleau et al., 1998), with both lines of work beginning to demonstrate the potential feasibility of planning and learning in very large environments.

The remainder of this paper is structured as follows: In Section 2, we give the formal definitions needed in this paper. Section 3 then gives our main result, an algorithm for planning in large or infinite MDPs, whose per-state running time does not depend on the size of the state space. Finally, Section 4 describes related results and open problems.

## 2. Preliminaries

We begin with the definition of a Markov decision process on a set of N =| S | states, explicitly allowing the possibility of the number of states being (countably or uncountably) infinite.

Definition 1 . A Markov decision process M on a set of states S and with actions { a 1 , . . . , ak } consists of:

- Transition probabilities : For each state-action pair ( s , a ) , a next-state distribution Psa ( s ′ ) that specifies the probability of transition to each state s ′ upon execution of action a from state s . 1
- Reward distributions : For each state-action pair ( s , a ) , a distribution Rsa on real-valued rewards for executing action a from state s . We assume rewards are bounded in absolute value by R max.

For simplicity, we shall assume in this paper that all rewards are in fact deterministic-that is, the reward distributions have zero variance, and thus the reward received for executing a from s is always exactly Rsa . However, all of our results have easy generalizations for the case of stochastic rewards, with an appropriate and necessary dependence on the variance of the reward distributions.

Throughout the paper, we will primarily be interested in MDPs with a very large (or even infinite) number of states, thus precluding approaches that compute directly on the full next-state distributions. Instead, we will assume that our planning algorithms are given M in the form of the ability to sample the behavior of M . Thus, the model given is simulative rather than explicit. We call this ability to sample the behavior of M a generative model .

Definition 2 . A generative model for a Markov decision process M is a randomized algorithm that, on input of a state-action pair ( s , a ) , outputs Rsa and a state s ′ , where s ′ is randomly drawn according to the transition probabilities Psa ( · ) .

We think of a generative model as falling somewhere in between being given explicit next-state distributions, and being given only 'irreversible' experience in the MDP (in which the agent follows a single, continuous trajectory, with no ability to reset to any desired state). On the one hand, a generative model may often be available when explicit next-state distributions are not; on the other, a generative model obviates the important issue of exploration that arises in a setting where we only have irreversible experience. In this sense, planning results using generative models blur the distinction between what is typically called 'planning' and what is typically called 'learning'.

Following standard terminology, we define a (stochastic) policy to be any mapping π : S ↦→{ a 1 , . . . , ak } . Thus π( s ) may be a random variable, but depends only on the current state s . We will be primarily concerned with discounted MDPs, 2 so we assume we are given a number 0 ≤ γ &lt; 1 called the discount factor , with which we then define the value function V π for any policy π :

<!-- formula-not-decoded -->

We also define the Q -function for a given policy π as

∣ where ri is the reward received on the i th step of executing the policy π from state s , and the expectation is over the transition probabilities and any randomization in π . Note that for any s and any π , | V π ( s ) | ≤ V max, where we define V max = R max /( 1 -γ) .

<!-- formula-not-decoded -->

(where the notation s ′ ∼ Psa ( · ) means that s ′ is drawn according to the distribution Psa ( · ) ). We will later describe an algorithm A that takes as input any state s and (stochastically) outputs an action a , and which therefore implements a policy. When we have such an algorithm, we will also write V A and Q A to denote the value function and Q -function of the policy implemented by A . Finally, we define the optimal value function and the optimal Q -function as V ∗ ( s ) = sup π V π ( s ) and Q ∗ ( s , a ) = sup π Q π ( s , a ) , and the optimal policy π ∗ , π ∗ ( s ) = arg max a Q ∗ ( s , a ) for all s ∈ S .

## 3. Planning in large or infinite MDPs

Usually, one considers the planning problem in MDPs to be that of computing a good policy, given as input the transition probabilities Psa ( · ) and the rewards Rsa (for instance, by solving the MDP for the optimal policy). Thus, the input is a complete and exact model, and the output is a total mapping from states to actions. Without additional assumptions about the structure of the MDP, such an approach is clearly infeasible in very large state spaces, where even reading all of the input can take N 2 time, and even specifying a general policy requires space on the order of N . In such MDPs, a more fruitful way of thinking about planning might be an on-line view, in which we examine the per-state complexity of planning. Thus, the input to a planning algorithm would be a single state, and the output would be which

single action to take from that state. In this on-line view, a planning algorithm is itself simply a policy (but one that may need to perform some nontrivial computation at each state).

Our main result is the description and analysis of an algorithm A that, given access to a generative model for an arbitrary MDP M , takes any state of M as input and produces an action as output, and meets the following performance criteria:

- The policy implemented by A is near-optimal in M ;
- The running time of A (that is, the time required to compute an action at any state) has no dependence on the number of states of M .

This result is obtained under the assumption that there is an O ( 1 ) time and space way to refer to the states, a standard assumption known as the uniform cost model (Aho, Hopcroft, &amp; Ullman, 1974), that is typically adopted to allow analysis of algorithms that operate on real numbers (such as we require to allow infinite state spaces). The uniform cost model essentially posits the availability of infinite-precision registers (and constant-size circuitry for performing the basic arithmetic operations on these registers). If one is unhappy with this model, then algorithm A will suffer a dependence on the number of states only equal to the space required to name the states (at worst log ( N ) for N states).

## 3.1. A sparse sampling planner

Here is our main result:

Theorem 1. There is a randomized algorithm A that , given access to a generative model for any k-action MDP M , takes as input any state s ∈ S and any value ε &gt; 0 , outputs an action , and satisfies the following two conditions :

- ( Efficiency ) The running time of A is O (( kC ) H ) , where

<!-- formula-not-decoded -->

In particular , the running time depends only on R max , γ , and ε , and does not depend on N =| S | . If we view R max as a constant , the running time bound can also be written

<!-- formula-not-decoded -->

- ( Near-Optimality ) The value function of the stochastic policy implemented by A satisfies

<!-- formula-not-decoded -->

simultaneously for all states s ∈ S.

Function: EstimateQ(h,C,%, G,s)

Input: depth h, width C, discount 7, generative model G, state s.

Output: A list (Q}(s, a1), @h (s, a2),..., Qi(s, ak)), of estimates of the Q*(s, ai).

1. If h = 0, return (0,..., 0).

2. For each a € A, use G to generate C samples from the next-state distribution Psa (•). Let

Sa be a set containing these C next-states.

3. For each a € A and let our estimate of Q*(s,a) be

Qi (s, a) = Rs, 0) + 7} EstimateV(k - 1,0,7, G, 5).

s'ESa

4. Return (Q}(s, a1), @h(s, a2),..., Qn(s, ak)).

As we have already suggested, it will be helpful to think of algorithm A in two different ways. On the one hand, A is an algorithm that takes a state as input and has access to a generative model, and as such we shall be interested in its resource complexity-its running time, and the number of calls it needs to make to the generative model (both per state input). On the other hand, A produces an action as output in response to each state given as input, and thus implements a (possibly stochastic) policy .

The proof of Theorem 1 is given in Appendix A, and detailed pseudo-code for the algorithm is provided in figure 1. We now give some high-level intuition for the algorithm and its analysis.

Given as input a state s , the algorithm must use the generative model to find a nearoptimal action to perform from state s . The basic idea of the algorithm is to sample the generative model from states in the 'neighborhood' of s . This allows us to construct a small 'sub-MDP' M ′ of M such that the optimal action in M ′ from s is a near-optimal action from s in M . 3 There will be no guarantee that M ′ will contain enough information to compute a good action from any state other than s . However, in exchange for this limited applicability, the MDP M ′ will have a number of states that does not depend on the number of states in M .

Figure 1 . Algorithm A for planning in large or infinite state spaces. EstimateV finds the ˆ V ∗ h described in the text, and EstimateQ finds analogously defined ˆ Q ∗ h . Algorithm A implements the policy.

Output: An action a.

(5)

a2

a2

a2

al a2

a2

a2

a2

a

a2

Depth

H

The graphical structure of M ′ will be given by a directed tree in which each node is labeled by a state, and each directed edge to a child is labeled by an action and a reward. For the sake of simplicity, let us consider only the two-action case here, with actions a 1 and a 2. Each node will have C children in which the edge to the child is labeled a 1, and C children in which the edge to the child is labeled a 2 .

We can also think of M ′ as an MDP in which the start state is s , and in which taking an action from a node in the tree causes a transition to a (uniformly) random child of that node with the corresponding action label; the childless leaf nodes are considered absorbing states. Under this interpretation, we can compute the optimal action to take from the root s in M ′ . Figure 2 shows a conceptual picture of this tree for a run of the algorithm from an input state s 0 , for C = 3. ( C will typically be much larger). From the root s 0, we try action a 1 three times and action a 2 three times. From each of the resulting states, we also try each action C times, and so on down to depth H in the tree. Zero values assigned to the leaves then correspond to our estimates of ˆ V ∗ 0 , which are 'backed-up' to find estimates of ˆ V ∗ 1 for their parents, which are in turn backed-up to their parents, and so on, up to the root to find an estimate of ˆ V ∗ H ( s 0 ) .

The root node of M ′ is labeled by the state of interest s , and we generate the 2 C children of s in the obvious way: we call the generative model C times on the state-action pair ( s , a 1 ) to get the a 1-children, and on C times on ( s , a 2 ) to get the a 2-children. The edges to these children are also labeled by the rewards returned by the generative model, and the child nodes themselves are labeled by the states returned. We will build this ( 2 C ) -ary tree to some depth to be determined. Note that M ′ is essentially a sparse look-ahead tree .

The central claim we establish about M ′ is that its size can be independent of the number of states in M , yet still result in our choosing near-optimal actions at the root. We do this by establishing bounds on the required depth H of the tree and the required degree C .

Recall that the optimal policy at s is given by π ∗ ( s ) = arg max a Q ∗ ( s , a ) , and therefore is completely determined by, and easily calculated from, Q ∗ ( s , · ) . Estimating the Q -values

Figure 2 . Sparse look-ahead tree of states constructed by the algorithm (shown with C = 3, actions a 1, a 2).

<!-- image -->

is a common way of planning in MDPs. ¿ From the standard duality between Q -functions and value functions, the task of estimating Q -functions is very similar to that of estimating value functions. So while the algorithm uses the Q -function, we will, purely for expository purposes, actually describe here how we estimate V ∗ ( s ) .

There are two parts to the approximation we use. First, rather than estimating V ∗ , we will actually estimate, for a value of H to be specified later, the H -step expected discounted reward V ∗ H ( s ) , given by

<!-- formula-not-decoded -->

∣ where ri is the reward received on the i th time step upon executing the optimal policy π ∗ from s . Moreover, we see that the V ∗ h ( s ) , for h ≥ 1, are recursively given by

<!-- formula-not-decoded -->

where a ∗ is the action taken by the optimal policy from state s , and V ∗ 0 ( s ) = 0. The quality of the approximation in Eq. (7) becomes better for larger values of h , and is controllably tight for the largest value h = H we eventually choose. One of the main efforts in the proof is establishing that the error incurred by the recursive application of this approximation can be made controllably small by choosing H sufficiently large.

Thus,ifweareabletoobtainanestimate ˆ V ∗ h -1 ( s ′ ) of V ∗ h -1 ( s ′ ) for any s ′ , we can inductively define an algorithm for finding an estimate ˆ V ∗ h ( s ) of V ∗ h ( s ) by making use of Eq. (7). Our algorithm will approximate the expectation in Eq. (7) by a sample of C random next states from the generative model, where C is a parameter to be determined (and which, for reasons that will become clear later, we call the 'width'). Recursively, given a way of finding the estimator ˆ V ∗ h -1 ( s ′ ) for any s ′ , we find our estimate ˆ V ∗ h ( s ) of V ∗ h ( s ) as follows:

1. For each action a , use the generative model to get Rsa and to sample a set Sa of C independently sampled states from the next-state distribution Psa ( · ) .
3. Following Eq. (7), our estimate of V ∗ h ( s ) is then given by
2. Use our procedure for finding ˆ V ∗ h -1 to estimate ˆ V ∗ h -1 ( s ′ ) for each state s ′ in any of the sets Sa .

<!-- formula-not-decoded -->

To complete the description of the algorithm, all that remains is to choose the depth H and the parameter C , which controls the width of the tree. Bounding the required depth H is the easy and standard part. It is not hard to see that if we choose depth H = log γ /epsilon1( 1 -γ)/ R max (the so-called /epsilon1 -horizon time ), then the discounted sum of the rewards that is obtained by considering rewards beyond this horizon is bounded by /epsilon1 .

The central claim we establish about C is that it can be chosen independent of the number of states in M , yet still result in choosing near-optimal actions at the root. The key to the argument is that even though small samples may give very poor approximations to the next-state distribution at each state in the tree, they will, nevertheless, give good estimates of the expectation terms of Eq. (7), and that is really all we need. For this we apply a careful combination of uniform convergence methods and inductive arguments on the tree depth. Again, the technical details of the proof are in Appendix A.

In general, the resulting tree may represent only a vanishing fraction of all of the H -step paths starting from s 0 that have non-zero probability in the MDP-that is, the sparse look-ahead tree covers only a vanishing part of the full look-ahead tree. In this sense, our algorithm is clearly related to and inspired by classical look-ahead search techniques (Russell &amp; Norvig, 1995) including various real-time search algorithms (Korf, 1990; Barto, Bradtke, &amp; Singh, 1995; Bonet, Loerincs, &amp; Geffner, 1997; Koenig &amp; Simmons, 1998) and receding horizon controllers. Most of these classical search algorithms, however, run into difficulties in very large or infinite MDPs with diffuse transitions, since their search trees can have arbitrarily large (or even infinite) branching factors. Our main contribution is showing that in large stochastic environments, clever random sampling suffices to reconstruct nearly all of the information available in the (exponentially or infinitely) large full look-ahead tree. Note that in the case of deterministic environments, where from each state-action pair we can reach only a single next state, the sparse and full trees coincide (assuming a memoization trick described below), and our algorithm reduces to classical deterministic look-ahead search.

## 3.2. Practical issues and lower bounds

Even though the running time of algorithm A does not depend on the size of the MDP, it still runs in time exponential in the /epsilon1 -horizon time H , and therefore exponential in 1 /( 1 -γ) . It would seem that the algorithm would be practical only if γ is not too close to 1. In a moment, we will give a lower bound showing it is not possible to do much better without further assumptions on the MDP. Nevertheless, there are a couple of simple tricks that may help to reduce the running time in certain cases, and we describe these tricks first.

The first idea is to allow different amounts of sampling at each level of the tree. The intuition is that the further we are from the root, the less influence our estimates will have on the Q -values at the root (due to the discounting). Thus, we can sample more sparsely at deeper levels of the tree without having too adverse an impact on our approximation.

Wehaveanalyzed various schemes for letting the amount of sampling at a node depend on its depth. None of the methods we investigated result in a running time which is polynomial in 1 //epsilon1 . However, one specific scheme that reduces the running time significantly is to let the number of samples per action at depth i be Ci = γ 2 i C , where the parameter C now controls the amount of sampling done at the root. The error in the Q -values using such a scheme does not increase by much, and the running time is the square root of our original running time. Beyond this and analogous to how classical search trees can often be pruned in ways that significantly reduce running time, a number of standard tree pruning methods may also be applied to our algorithm's trees (Russell &amp; Norvig, 1995) (see also Dearden

&amp; Boutilier, 1994), and we anticipate that this may significantly speed up the algorithm in practice.

Another way in which significant savings might be achieved is through the use of memoization in our subroutines for calculating the ˆ V ∗ h ( s ) 's. In figure 2, this means that whenever there are two nodes at the same level of the tree that correspond to the same state, we collapse them into one node (keeping just one of their subtrees). While it is straightforward to show the correctness of such memoization procedures for deterministic procedures, one must be careful when addressing randomized procedures. We can show that the important properties of our algorithm are maintained under this optimization. Indeed, this optimization is particularly nice when the domain is actually deterministic: if each action deterministically causes a transition to a fixed next-state, then the tree would grow only as k H (where k is the number of actions). If the domain is 'nearly deterministic,' then we have behavior somewhere in between. Similarly, if there are only some N 0 /lessmuch | S | states reachable from s 0 , then the tree would also never grow wider than N 0, giving it a size of O ( N 0 H ) .

In implementing the algorithm, one may wish not to specify a targeted accuracy /epsilon1 in advance, but rather to try to do as well as is possible with the computational resources available. In this case, an 'iterative-deepening' approach may be taken. This would entail simultaneously increasing C and H by decreasing the target /epsilon1 . Also, as studied in Davies, Ng, and Moore (1998), if we have access to an initial estimate of the value function, we can replace our estimates ˆ V ∗ 0 ( s ) = 0attheleaveswiththeestimatedvaluefunctionatthosestates. Though we shall not do so here, it is again easy to make formal performance guarantees depending on C , H and the supremum error of the value function estimate we are using.

Unfortunately, despite these tricks, it is not difficult to prove a lower bound that shows that any planning algorithm with access only to a generative model, and which implements a policy that is /epsilon1 -close to optimal in a general MDP, must have running time at least exponential in the /epsilon1 -horizon time. We now describe this lower bound.

Theorem 2. Let A be any algorithm that is given access only to a generative model for an MDP M , and inputs s ( a state in M ) and /epsilon1 . Let the stochastic policy implemented by A satisfy

<!-- formula-not-decoded -->

simultaneously for all states s ∈ S. Then there exists an MDP M on which A makes at least /Omega1 ( 2 H ) = /Omega1 (( 1 //epsilon1 ) ( 1 / log ( 1 /γ)) ) calls to the generative model.

Proof: Let H = log γ /epsilon1 = log ( 1 //epsilon1 )/ log ( 1 /γ) . Consider a binary tree T of depth H . We use T to define an MDP in the following way. The states of the MDP are the nodes of the tree. The actions of the MDP are { 0 , 1 } . When we are in state s and perform an action b we reach (deterministically) state sb , where sb is the b -child of s in T . If s is a leaf of T then we move to an absorbing state. We choose a random leaf v in the tree. The reward function for v and any action is R max, and the reward at any other state and action is zero.

Algorithm A is given s 0 , the root of T . For algorithm A to compute a near optimal policy, it has to 'find' the node v , and therefore has to perform at least /Omega1 ( 2 H ) calls to the generative model. ✷

## 4. Summary and related work

Wehavedescribed an algorithm for near-optimal planning from a generative model, that has a per-state running time that does not depend on the size of the state space, but which is still exponential in the /epsilon1 -horizon time. An important open problem is to close the gap between our lower and upper bound. Our lower bound shows that the number of steps has to grow polynomially in 1 //epsilon1 while in the upper bound the number of steps grows sub-exponentially in 1 //epsilon1 , more precisely ( 1 //epsilon1 ) O ( log ( 1 //epsilon1 )) . Closing this gap, either by giving an algorithm that would be polynomial in 1 //epsilon1 or by proving a better lower bound, is an interesting open problem.

Two interesting directions for improvement are to allow partially observable MDPs (POMDPs), and to find more efficient algorithms that do not have exponential dependence on the horizon time. As a first step towards both of these goals, in a separate paper (Kearns, Mansour, &amp; Ng, to appear) we investigate a framework in which the goal is to use a generative model to find a near-best strategy within a restricted class of strategies for a POMDP. Typical examples of such restricted strategy classes include limited-memory strategies in POMDPs, or policies in large MDPs that implement a linear mapping from state vectors to actions. Our main result in this framework says that as long as the restricted class of strategies is not too 'complex' (where this is formalized using appropriate generalizations of standard notions like VC dimension from supervised learning), then it is possible to find a near-best strategy from within the class, in time that again has no dependence on the size of the state space. If the restricted class of strategies is smoothly parameterized, then this further leads to a number of fast, practical algorithms for doing gradient descent to find the near-best strategy within the class, where the running time of each gradient descent step now has only linear rather than exponential dependence on the horizon time.

Another approach to planning in POMDPs that is based on the algorithm presented here is investigated by McAllester and Singh (1999), who show how the approximate belief-state tracking methods of Boyen and Koller (1998) can be combined with our algorithm.

## Appendix A: Proof sketch of Theorem 1

In this appendix, we give the proof of Theorem 1.

Theorem 1. There is a randomized algorithm A that , given access to a generative model for any k-action MDP M , takes as input any state s ∈ S and any value ε &gt; 0 , outputs an action , and satisfies the following two conditions :

- ( Efficiency ) The running time of A is O (( kC ) H ), where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular , the running time depends only on R max , γ, and ε, and does not depend on N =| S | . If we view R max as a constant , the running time bound can also be written

<!-- formula-not-decoded -->

- ( Near-Optimality ) The value function of the stochastic policy implemented by A satisfies

<!-- formula-not-decoded -->

simultaneously for all states s ∈ S.

Throughout the analysis we will rely on the pseudo-code provided for algorithm A given in figure 1.

The claim on the running time is immediate from the definition of algorithm A . Each call to EstimateQ generates kC calls to EstimateV , C calls for each action. Each recursive call also reduces the depth parameter h by one, so the depth of the recursion is at most H . Therefore the running time is O (( kC ) H ) .

The main effort is in showing that the values of EstimateQ are indeed good estimates of Q ∗ for the chosen values of C and H . There are two sources of inaccuracy in these estimates. The first is that we use only a finite sample to approximate an expectation-we draw only C states from the next-state distributions. The second source of inaccuracy is that in computing EstimateQ , we are not actually using the values of V ∗ ( · ) but rather values returned by EstimateV , which are themselves only estimates. The crucial step in the proof is to show that as h increases, the overall inaccuracy decreases .

Let us first define an intermediate random variable that will capture the inaccuracy due to the limited sampling. Define U ∗ ( s , a ) as follows:

<!-- formula-not-decoded -->

where the si are drawn according to Psa ( · ) . Note that U ∗ ( s , a ) is averaging values of V ∗ ( · ) , the unknown value function. Since U ∗ ( s , a ) is used only for the proof and not in the algorithm, there is no problem in defining it this way. The next lemma shows that with high probability, the difference between U ∗ ( s , a ) and Q ∗ ( s , a ) is at most λ .

Lemma 3. For any state s and action a , with probability at least 1 -e -λ 2 C / V 2 max we have

<!-- formula-not-decoded -->

∣ where the probability is taken over the draw of the si from Psa ( · ) .

Proof: Note that Q ∗ ( s , a ) = Rsa + γ E s ∼ Psa ( · ) [ V ∗ ( s ) ]. The proof is immediate from the Chernoff bound.

Now that we have quantified the error due to finite sampling, we can bound the error from our using values returned by EstimateV rather than V ∗ ( · ) . We bound this error as the difference between U ∗ ( s , a ) and EstimateV . In order to make our notation simpler, let V n ( s ) be the value returned by EstimateV ( n , C , γ, G , s ) , and let Q n ( s , a ) be the component in the output of EstimateQ ( n , C , γ, G , s ) that corresponds to action a . Using this notation, our algorithm computes

<!-- formula-not-decoded -->

where V n -1 ( s ) = max a { Q n -1 ( s , a ) } , and Q 0 ( s , a ) = 0 for every state s and action a .

Wenowdefine a parameter α n that will eventually bound the difference between Q ∗ ( s , a ) and Q n ( s , a ) . We define α n recursively:

<!-- formula-not-decoded -->

where α 0 = V max. Solving for α H we obtain

<!-- formula-not-decoded -->

The next lemma bounds the error in the estimation, at level n , by α n . Intuitively, the error due to finite sampling contributes λ , while the errors in estimation contribute α n . The combined error is λ + α n , but since we are discounting, the effective error is only γ(λ + α n ) , which by definition is α n + 1 . ✷

Lemma 4. With probability at least 1 -( kC ) n e -λ 2 C / V 2 max we have that

<!-- formula-not-decoded -->

Proof: The proof is by induction on n . It clearly holds for n = 0. Now

<!-- formula-not-decoded -->

Werequire that all of the C child estimates be good, for each of the k actions. This means that the probability of a bad estimate increases by a factor of kC , for each n . By Lemma 3 the probability of a single bad estimate is bounded by e -λ 2 C / V 2 max . Therefore the probability of some bad estimate is bounded by 1 -( kC ) n e -λ 2 C / V 2 max . ✷

From α H ≤ γ H V max + λ/( 1 -γ) , we also see that for H = log γ (λ/ V max ) , with probability 1 -( kC ) H e -λ 2 C / V 2 max all the final estimates Q H ( s 0 , a ) are within 2 λ/( 1 -γ) from the true Q -values. The next step is to choose C such that δ = λ/ R max ≥ ( kC ) H e -λ 2 C / V 2 max will bound the probability of a bad estimate during the entire computation. Specifically,

<!-- formula-not-decoded -->

is sufficient to ensure that with probability 1 -δ all the estimates are accurate.

At this point we have shown that with high probability, algorithm A computes a good estimate of Q ∗ ( s 0 , a ) for all a , where s 0 is the input state. To complete the proof, we need to relate this to the expected value of a stochastic policy. We give a fairly general result about MDPs, which does not depend on our specific algorithm. (A similar result appears in Singh &amp; Yee, 1994).

Lemma 5. Assume that π is a stochastic policy , so that π( s ) is a random variable. If for each state s , the probability that Q ∗ ( s , π ∗ ( s )) -Q ∗ ( s , π( s )) &lt; λ is at least 1 -δ, then the discounted infinite horizon return of π is at most (λ + 2 δ V max )/( 1 -γ) from the optimal return , i.e. , for any state sV ∗ ( s ) -V π ( s ) ≤ (λ + 2 δ V max )/( 1 -γ) .

Proof: Since we assume that the rewards are bounded by R max, it implies that the expected return of π at each state s is at least

<!-- formula-not-decoded -->

Now we show that if π has the property that at each state s the difference between E [ Q ∗ ( s , π( s )) ] and Q ∗ ( s , π ∗ ( s )) is at most β , then V ∗ ( s ) -V π ( s ) ≤ β/( 1 -γ) . (A similar result was proved by Singh and Yee (1994), for the case that each action chosen has Q ∗ ( s , π ∗ ( s )) -Q ( s , π( s )) ≤ β . It is easy to extend their proof to handle the case here, and we sketch a proof only for completeness).

The assumption on the Q ∗ values immediately implies | E [ R ( s , π ∗ ( s )) ] -E [ R ( s , π( s )) ] | ≤ β . Consider a policy π j that executes π for the first j + 1 steps and then executes π ∗ . We can show by induction on j that for every state s , V ∗ ( s ) -V π j ( s ) ≤ ∑ j i = 0 βγ i . This implies that V ∗ ( s ) -V π ( s ) ≤ ∞ i 0 βγ i = β/( 1 -γ) .

<!-- formula-not-decoded -->

By setting β = λ + 2 δ V max

Now we can combine all the lemmas to prove our main theorem.

Proof of Theorem 1: As discussed before, the running time is immediate from the algorithm, and the main work is showing that we compute a near-optimal policy. By Lemma 4 we have that the error in the estimation of Q ∗ is at most α H , with probability 1 -( kC ) H e -λ 2 C / V 2 max . Using the values we chose for C and H we have that with probability 1 -δ the error is at most 2 λ/( 1 -γ) . By Lemma 5 this implies that such a policy π has the property that from every state s ,

<!-- formula-not-decoded -->

Substituting back the values of δ = λ/ R max and λ = /epsilon1( 1 -γ) 2 / 4 that we had chosen, it follows that

<!-- formula-not-decoded -->

## Acknowledgments

We give warm thanks to Satinder Singh for many enlightening discussions and numerous insights on the ideas presented here.

## Notes

1. Henceforth, everything that needs to be measurable is assumed to be measurable.
2. However, our results can be generalized to the undiscounted finite-horizon case for any fixed horizon H (McAllester &amp; Sing, 1999).
3. M ′ will not literally be a sub-MDP of M , in the sense of being strictly embedded in M , due to the variations of random sampling. But it will be very 'near' such an embedded MDP.

## References

- Aho, A. V., Hopcroft, J. E., &amp; Ullman, J. D. (1974). The design and analysis of computer algorithms. Reading MA: Addison-Wesley.
- Barto, A. G., Bradtke, S. J., &amp; Singh, S. P. (1995) Learning to act using real-time dynamic programming. Artificial Intelligence, 72 , 81-138.
- Boutilier, C., Dearden, R., &amp; Goldszmidt, M. (1995). Exploiting structure in policy construction. In Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (pp. 1104-1111).
- Boyen, X., &amp; Koller, D. (1998). Tractable inference for complex stochastic processes. In Proceedings of the 1998 Conference on Uncertainty in Artificial Intelligence . San Mateo, CA: Morgan Kauffmann.
- Bonet, B., Loerincs, G., &amp; Geffner, H. (1997). A robust and fast action selection mechanism for planning. In Proceedings of the Fourteenth National Conference on Artifial Intelligence .
- Dearden, R., &amp; Boutilier, C. (1994). Integrating planning and execution in stochastic domains. In Proceedings of the Tenth Annual Conference on Uncertainty in Artificial Intelligence .
- Davies, S., Ng, A. Y., &amp; Moore, A. (1998). Applying online-search to reinforcement learning. In Proceedings of AAAI-98 (pp. 753-760). Menlo Park, CA: AAAI Press.

✷

- Kearns, M., Mansour, Y., &amp; Ng, A. Y. Approximate planning in large POMDPs via reusable trajectories. In neural information processing systems 13 , to appear.
- Korf, R. E. (1990). Real-time heuristic search. Artificial Intelligence, 42 , 189-211.
- Koller, D., &amp; Parr, R. (1999). Computing factored value functions for policies in structured MDPs. In Proceedings of the Sixteenth International Joint Conference on Artificial Intelligence.
- Koenig, S., &amp; Simmons, R. (1998). Solving robot navigation problems with initial pose uncertainty using realtime heuristic search. In Proceedings of the Fourth International Conference on Artificial Intelligence Planning Systems.
- Kearns, M., &amp; Singh, S. (1999). Finite-sample convergence rates for Q-learning and indirect algorithms. In Neural Information Processing systems 12 . Cambridge, MA: MIT Press.
- Meuleau, N., Hauskrecht, M., Kim, K.-E., Peshkin, L., Kaelbling, L. P., Dean, T., &amp; Boutilier, C. (1998). Solving very large weakly coupled Markov decision processes. In Proceedings of AAAI (pp. 165-172).

McAllester, D., &amp; Singh, S. (1999). Personal Communication.

McAllester, D., &amp; Singh, S. Approximate planning for factored POMDPs using belief state simplification. Preprint. Russell, S., &amp; Norvig, P. (1995). Artificial Intelligence: A modern approach . Englewood cliffs, NJ: Prentice-Hall. Sutton, R. S., &amp; Barto, A. G. (1998). Reinforcement learning . Cambridge, MA: MIT Press.

- Singh, S., &amp; Yee, R. (1994). An upper bound on the loss from approximate optimal-value functions. Machine Learning, 16 , 227-233.

Received March 17, 1999 Revised October 4, 2001 Accepted October 4, 2001 Final manuscript October 4, 2001