## The Value Equivalence Principle for Model-Based Reinforcement Learning

## Christopher Grimm

Computer Science &amp; Engineering University of Michigan crgrimm@umich.edu

André Barreto, Satinder Singh, David Silver DeepMind

{andrebarreto,baveja,davidsilver}@google.com

## Abstract

Learning models of the environment from data is often viewed as an essential component to building intelligent reinforcement learning (RL) agents. The common practice is to separate the learning of the model from its use, by constructing a model of the environment's dynamics that correctly predicts the observed state transitions. In this paper we argue that the limited representational resources of model-based RL agents are better used to build models that are directly useful for value-based planning. As our main contribution, we introduce the principle of value equivalence: two models are value equivalent with respect to a set of functions and policies if they yield the same Bellman updates. We propose a formulation of the model learning problem based on the value equivalence principle and analyze how the set of feasible solutions is impacted by the choice of policies and functions. Specifically, we show that, as we augment the set of policies and functions considered, the class of value equivalent models shrinks, until eventually collapsing to a single point corresponding to a model that perfectly describes the environment. In many problems, directly modelling state-to-state transitions may be both difficult and unnecessary. By leveraging the value-equivalence principle one may find simpler models without compromising performance, saving computation and memory. We illustrate the benefits of value-equivalent model learning with experiments comparing it against more traditional counterparts like maximum likelihood estimation. More generally, we argue that the principle of value equivalence underlies a number of recent empirical successes in RL, such as Value Iteration Networks, the Predictron, Value Prediction Networks, TreeQN, and MuZero, and provides a first theoretical underpinning of those results.

## 1 Introduction

Reinforcement learning (RL) provides a conceptual framework to tackle a fundamental challenge in artificial intelligence: how to design agents that learn while interacting with the environment [36]. It has been argued that truly general agents should be able to learn a model of the environment that allows for fast re-planning and counterfactual reasoning [32]. Although this is not a particularly contentious statement, the question of how to learn such a model is far from being resolved. The common practice in model-based RL is to conceptually separate the learning of the model from its use. In this paper we argue that the limited representational capacity of model-based RL agents is better allocated if the future use of the model ( e.g. , value-based planning) is also taken into account during its construction [22, 15, 13].

Our primary contribution is to formalize and analyze a clear principle that underlies this new approach to model-based RL. Specifically, we show that, when the model is to be used for value-based planning, requirements on the model can be naturally captured by an equivalence relation induced by a set

of policies and functions. This leads to the principle of value equivalence : two models are value equivalent with respect to a set of functions and a set of policies if they yield the same updates under corresponding Bellman operators. The policies and functions then become the mechanism through which one incorporates information about the intended use of the model during its construction. We propose a formulation of the model learning problem based on the value equivalence principle and analyze how the set of feasible solutions is impacted by the choice of policies and functions. Specifically, we show that, as we augment the set of policies and functions considered, the class of value equivalent models shrinks, until eventually collapsing to a single point corresponding to a model that perfectly describes the environment.

We also discuss cases in which one can meaningfully restrict the class of policies and functions used to tailor the model. One common case is when the construction of an optimal policy through value-based planning only requires that a model predicts a subset of value functions. We show that in this case the resulting value equivalent models can perform well under much more restrictive conditions than their traditional counterparts. Another common case is when the agent has limited representational capacity. We show that in this scenario it suffices for a model to be value equivalent with respect to appropriately-defined bases of the spaces of representable policies and functions. This allows models to be found with less memory or computation than conventional model-based approaches that aim at predicting all state transitions, such as maximum likelihood estimation. We illustrate the benefits of value-equivalent model learning in experiments that compare it against more conventional counterparts. More generally, we argue that the principle of value equivalence underlies a number of recent empirical successes in RL and provides a first theoretical underpinning of those results [40, 34, 26, 16, 33].

## 2 Background

As usual, we will model the agent's interaction with the environment using a Markov Decision Process (MDP) M≡ 〈S , A , r, p, γ 〉 where S is the state space, A is the action space, r ( s, a, s ′ ) is the reward associated with a transition to state s ′ following the execution of action a in state s , p ( s ′ | s, a ) is the transition kernel and γ ∈ [0 , 1) is a discount factor [30]. For convenience we also define r ( s, a ) = E S ′ ∼ p ( ·| s,a ) [ r ( s, a, S ′ )] .

A policy is a mapping π : S ↦→ P ( A ) , where P ( A ) is the space of probability distributions over A . We define GLYPH&lt;5&gt; ≡ { π | π : S ↦→ P ( A ) } as the set of all possible policies. The agent's goal is to find a policy π ∈ GLYPH&lt;5&gt; that maximizes the value of every state, defined as

<!-- formula-not-decoded -->

where S t and A t are random variables indicating the state occupied and the action selected by the agent at time step t and E π [ · ] denotes expectation over the trajectories induced by π .

Many methods are available to carry out the search for a good policy [36, 39]. Typically, a crucial step in these methods is the computation of the value function of candidate policies-a process usually referred to as policy evaluation . One way to evaluate a policy π is through its Bellman operator :

<!-- formula-not-decoded -->

where v is any function in the space V ≡ { f | f : S ↦→ R } . It is known that lim n →∞ ( T π ) n v = v π , that is, starting from any v ∈ V , the repeated application of T π will eventually converge to v π [30].

In RL it is generally assumed that the agent does not know p and r , and thus cannot directly compute (2). In model-free RL this is resolved by replacing v π with an action-value function and estimating the expectation on the right-hand-side of (2) through sampling [35]. In model-based RL , the focus of this paper, the agent learns approximations ˜ r ≈ r and ˜ p ≈ p and use them to compute (2) with p and r replaced by ˜ p and ˜ r [36].

## 3 Value equivalence

Given a state space S and an action space A , we call the tuple m ≡ ( r, p ) a model . Note that a model plus a discount factor γ induces a Bellman operator (2) for every policy π ∈ GLYPH&lt;5&gt; . In this paper we

are interested in computing an approximate model ˜ m = (˜ r, ˜ p ) such that the induced operators ˜ T π , defined analogously to (2), are good approximations of the true T π . Our main argument is that models should only be distinguished with respect to the policies and functions they will actually be applied to. This leads to the following definition:

Definition 1 (Value equivalence) . Let Π ⊆ GLYPH&lt;5&gt; be a set of policies and let V ⊆ V be a set of functions. We say that models m and ˜ m are value equivalent with respect to Π and V if and only if

<!-- formula-not-decoded -->

where T π and ˜ T π are the Bellman operators induced by m and ˜ m , respectively.

Two models are value equivalent with respect to Π and V if the effect of the Bellman operator induced by any policy π ∈ Π on any function v ∈ V is the same for both models. Thus, if we are only interested in Π and V , value-equivalent models are functionally identical. This can be seen as an equivalence relation that partitions the space of models conditioned on Π and V :

Definition 2 (Space of value-equivalent models) . Let Π and V be defined as above and let M be a space of models. Given a model m , the space of value-equivalent models M m (Π , V ) ⊆ M is the set of all models ˜ m ∈ M that are value equivalent to m with respect to Π and V .

Let M be a space of models containing at least one model m ∗ which perfectly describes the interaction of the agent with the environment. More formally, m ∗ induces the true Bellman operators T π defined in (2). Given a space of models M⊆ M , often one is interested in models m ∈ M that are value equivalent to m ∗ . We will thus simplify the notation by defining M (Π , V ) ≡ M m ∗ (Π , V ) .

## 3.1 The topology of the space of value-equivalent models

The space M (Π , V ) contains all the models in M that are value equivalent to the true model m ∗ with respect to Π and V . Since any two models m,m ′ ∈ M (Π , V ) are equally suitable for value-based planning using Π and V , we are free to use other criteria to choose between them. For example, if m is much simpler to represent or learn than m ′ , it can be preferred without compromises.

Clearly, the principle of value equivalence can be useful if leveraged in the appropriate way. In order for that to happen, it is important to understand the space of value-equivalent models M (Π , V ) . We now provide intuition for this space by analyzing some of its core properties. We refer the reader to Figure 1 for an illustration of the concepts to be discussed in this section. We start with a property that follows directly from Definitions 1 and 2:

Property 1. Given M ′ ⊆ M , we have that M ′ (Π , V ) ⊆ M (Π , V ) .

The proofs of all theoretical results are in Appendix A.1. Property 1 states that, given a set of policies Π and a set of functions V , reducing the size of the space of models M also reduces the space of value-equivalent models M (Π , V ) . One immediate consequence of this property is that, if we consider the space of all policies GLYPH&lt;5&gt; and the space of all functions V , we have one of two possibilities: either we end up with a perfect model or we end up with no model at all. Or, more formally:

Property 2. M ( GLYPH&lt;5&gt; , V ) either contains m ∗ or is the empty set.

Property 1 describes what happens to M (Π , V ) when we vary M with fixed Π and V . It is also interesting to ask what happens when we fix the former and vary the latter. This leads to the next property:

Figure 1: Understanding the space of valueequivalent models for a fixed Π , M ′ ⊂ M and V ′ ⊂ V . Denote M ( V ) ≡ M ( V , Π) . Property 1 : M ′ ( V ) ⊂ M ( V ) and M ′ ( V ′ ) ⊂ M ( V ′ ) . Property 3 : M ( V ) ⊂ M ( V ′ ) and M ′ ( V ) ⊂ M ′ ( V ′ ) . Property 4 : if m ∗ ∈ M , then m ∗ ∈ M ( V ) .

<!-- image -->

<!-- formula-not-decoded -->

According to Property 3, as we increase the size of Π or V the size of M (Π , V ) decreases . Although this makes intuitive sense, it is reassuring to know that value equivalence is a sound principle for

model selection, since by adding more policies to Π or more values to V we can progressively restrict the set of feasible solutions. Thus, if M contains the true model, we eventually pin it down. Indeed, in this case the true model belongs to all spaces of value equivalent models, as formalized below:

Property 4. If m ∗ ∈ M , then m ∗ ∈ M (Π , V ) for all Π and all V .

## 3.2 A basis for the space of value-equivalent models

As discussed, it is possible to use the sets Π and V to control the size of M (Π , V ) . But what exactly is the effect of Π and V on M (Π , V ) ? How much does M (Π , V ) decrease in size when we, say, add one function to V ? In this section we address this and similar questions.

We start by showing that, whenever a model is value equivalent to m ∗ with respect to discrete Π and V , it is automatically value equivalent to m ∗ with respect to much larger sets. In order to state this fact more concretely we will need two definitions. Given a discrete set H , we define span( H ) as the set formed by all linear combinations of the elements in H . Similarly, given a discrete set H in which each element is a function defined over a domain X , we define the pointwise span of H as

<!-- formula-not-decoded -->

where h i ∈ H . Pointwise span can alternatively be characterized by considering each element in the domain separately: g ∈ p -span( H ) ⇐⇒ g ( x ) ∈ span { h ( x ) : h ∈ H} for all x ∈ X . Equipped with these concepts we present the following result:

Proposition 1. For discrete Π and V , we have that M (Π , V ) = M ( p -span(Π) ∩ GLYPH&lt;5&gt; , span( V )) .

Proposition 1 provides one possible answer to the question posed at the beginning of this section: the contraction of M (Π , V ) resulting from the addition of one policy to Π or one function to V depends on their effect on p -span(Π) and span( V ) . For instance, if a function v can be obtained as a linear combination of the functions in V , adding it to this set will have no effect on the space of equivalent models M (Π , V ) . More generally, Proposition 1 suggests a strategy to build the set V : one should find a set of functions that form a basis for the space of interest. When S is finite, for example, having V be a basis for R |S| means that the value equivalence principle will apply to every function v ∈ R | S | . The same reasoning applies to Π . In fact, because p -span(Π) grows independently pointwise, it is relatively simple to build a set Π that covers the space of policies one is interested in. In particular, when A is finite, it is easy to define a set Π for which p -span(Π) ⊇ GLYPH&lt;5&gt; : it suffices to have for every state-action pair ( s, a ) ∈ S × A at least one policy π ∈ Π such that π ( a | s ) = 1 . This means that we can apply the value equivalence principle to the entire set GLYPH&lt;5&gt; using | A | policies only.

Combining Proposition 1 and Property 2 we see that by defining Π and V appropriately we can focus on the subset of M whose models perfectly describe the environment:

Remark 1. If GLYPH&lt;5&gt; ⊆ p -span(Π) and V = span( V ) , then M (Π , V ) = m ∗ or M (Π , V ) = ∅ .

We have shown how Π and V have an impact on the number of value equivalent models in M (Π , V ) ; to make the discussion more concrete, we now focus on a specific model space M and analyze the rate at which this space shrinks as we add more elements to Π and V . Before proceeding we define a set of functions H as pointwise linearly independent if h / ∈ p -span( H\{ h } ) for all h ∈ H .

Suppose both S and A are finite. In this case a model can be defined as m = ( r , P ) , where r ∈ R |S||A| and P ∈ R |S|×|S|×|A| . A policy can then be thought of as a vector π ∈ R |S||A| . We denote the set of all transition matrices induced by transition kernels as P . To simplify the analysis we will consider that r is known and we are interested in finding a model ˜ P ∈ P . In this setting, we write P (Π , V ) to denote the set of transition matrices that are value equivalent to the true transition matrix P ∗ . We define the dimension of a set X as the lowest possible Hamel dimension of a vector-space enclosing some translated version of it: dim[ X ] = min W ,c ∈ W ( X ) H -dim[ W ] where W ( X ) = { ( W , c ) : X + c ⊆ W} , W is a vector-space, c is an offset and H -dim[ · ] denotes the Hamel dimension. Recall that the Hamel dimension of a vector-space is the size of the smallest set of mutually linearly independent vectors that spans the space (this corresponds to the usual notion of dimension, that is, the minimal number of coordinates required to uniquely specify each point). So, under no restrictions imposed by Π and V , we have that dim[ P ] = ( |S| 1) |S||A| . We now show how fast the size of P (Π , V ) decreases as we extend the ranges of Π and V :

Proposition 2. Let Π be a set of m pointwise linearly independent policies π i ∈ R |S||A| and let V be a set of k linearly independent vectors v i ∈ R |S| . Then,

<!-- formula-not-decoded -->

Interestingly, Proposition 2 shows that the elements of Π and V interact in a multiplicative way: when there are m pointwise linearly independent policies, enlarging V with a single function v that is linearly independent of its counterparts will decrease the bound on the dimension of P (Π , V ) by a factor of m . This makes intuitive sense if we note that by definition m ≤ |A| : for an expressive enough Π , each v ∈ V will provide information about the effect of all actions in a ∈ A . Conversely, because span( V ) = k ≤ |S| , we can only go so far in pinning down the model when m &lt; |A| -which also makes sense, since in this case we cannot possibly know about the effect of all actions, no matter how big V is. Note that when m = |A| and k = |S| the space P (Π , V ) reduces to { P ∗ } .

## 4 Model learning based on the value-equivalence principle

We now discuss how the principle of value equivalence can be incorporated into model-based RL. Often in model-based RL one learns a model ˜ m = (˜ r, ˜ p ) without taking the space M (Π , V ) into account. The usual practice is to cast the approximations ˜ r ≈ r and ˜ p ≈ p as optimization problems over a model-space M that do not involve the sets Π and V . Given a space R of possible approximations ˜ r , we can formulate the approximation of the rewards as argmin ˜ r ∈R glyph[lscript] r ( r, ˜ r ) , where glyph[lscript] r is a loss function that measures the dissimilarity between r and ˜ r . The approximation of the transition dynamics can be formalized in an analogous way: argmin ˜ p ∈P glyph[lscript] p ( p, ˜ p ) , where P is the space of possible approximations ˜ p .

A common choice for glyph[lscript] r is

<!-- formula-not-decoded -->

where D is a distribution over S × A . The loss glyph[lscript] p is usually defined based on the principle of maximum likelihood estimation (MLE):

<!-- formula-not-decoded -->

where D KL is the Kullback-Leibler (KL) divergence. Since we normally do not have access to r and p , the losses (4) and (5) are usually minimized using transitions sampled from the environment [38]. There exist several other criteria to approximate p based on state transitions, such as maximum a posteriori estimation, maximum entropy estimation, and Bayesian posterior inference [13]. Although we focus on MLE for simplicity, our arguments should extend to these other criteria as well.

Both (4) and (5) have desirable properties that justify their widespread adoption [24]. However, we argue that ignoring the future use of ˜ r and ˜ p may not always be the best choice [22, 15]. To illustrate this point, we now show that, by doing so, one might end up with an approximate model when an exact one were possible. Let P (Π , V ) be the set of value equivalent transition kernels in P . Then, glyph[negationslash]

Proposition 3. The maximum-likelihood estimate of p ∗ in P may not belong to a P (Π , V ) = ∅ .

Proposition 3 states that, even when there exist models in P that are value equivalent to p ∗ with respect to Π and V , the minimizer of (5) may not be in P (Π , V ) . In other words, even when it is possible to perfectly handle the policies in Π and the values in V , the model that achieves the smallest MLE loss will do so only approximately. This is unsurprising since the loss (5) is agnostic of Π and V , providing instead a model that represents a compromise across all policies GLYPH&lt;5&gt; and all functions V .

We now define a value-equivalence loss that explicitly takes into account the sets Π and V :

<!-- formula-not-decoded -->

where ˜ T π are Bellman operators induced by ˜ m and || · || is a norm. Given (6), the problem of learning a model based on the value equivalence principle can be formulated as argmin ˜ m ∈M glyph[lscript] Π , V ( m ∗ , ˜ m ) .

As noted above, we usually do not have access to T π , and thus the loss (6) will normally be minimized based on sample transitions. Let S π ≡ { ( s π i , a π i , r π i , ˆ s π i ) | i = 1 , 2 , ..., n π } be n π sample transitions associated with policy π ∈ Π . We assume that the initial states s π i were sampled according to some

vial shown i

Ull. we opriate

x A. 1.2 for a concrete example).

even wh

^єМ+

ition 3,

I model

М'

M(V)

mvElo mVE.

distribution D ′ over S and the actions were sampled according to the policy π , a π i ∼ π ( ·| s π i ) (note that D ′ can be the distribution resulting from a direct interaction of the agent with the environment). When ‖ · ‖ appearing in (6) is a p -norm , we can write its empirical version as

<!-- formula-not-decoded -->

✶ where S ′ π is a set containing only the initial states s π i ∈ S π and ✶ {·} is the indicator function. We argue that, when we know policies Π and functions V that are sufficient for planning, the appropriate goal for model-learning is to minimize the value-equivalence loss (6). As shown in Proposition 3, the model ˜ m that minimizes (4) and (5) may not achieve zero loss on (6) even when such a model exists in M . In general, though, we should not expect there to be a model ˜ m ∈ M that leads to zero value-equivalence loss. Even then, value equivalence may lead to a better model than conventional counterparts (see Figure 2 for intuition and Appendix A.1.2 for a concrete example).

## 4.1 Restricting the sets of policies and functions

The main argument of this paper is that, rather than learning a model that suits all policies GLYPH&lt;5&gt; and all functions V , we should instead focus on the sets of policies Π and functions V that are necessary for planning. But how can we know these sets a priori ? We now show that it is possible to exploit structure on both the problem and the solution sides.

First, we consider structure in the problem . Suppose we had access to the true model m ∗ . Then, given an initial function v , a value-based planning algorithm that makes use of m ∗ will generate a sequence of functions glyph[vector] V v ≡ { v 1 , v 2 , ... } [10]. Clearly, if we replace m ∗ with any model in M ( GLYPH&lt;5&gt; , glyph[vector] V v ) , the behavior of the algorithm starting from v remains unaltered. This allows us to state the following:

Proposition 4. Suppose v ∈ V ′ = ⇒ T π v ∈ V ′ for all π ∈ GLYPH&lt;5&gt; . Let p -span(Π) ⊇ GLYPH&lt;5&gt; and span( V ) = V ′ . Then, starting from any v ′ ∈ V ′ , any ˜ m ∈ M (Π , V ) yields the same solution as m ∗ .

Figure 2: When the hypothesis space M ′ and the space of value-equivalent models M ( V ) intersect, the resulting model ˜ m VE has zero loss (6), while the corresponding MLE model ˜ m MLE may not (Proposition 3). But even when M ′ and M ( V ′ ) do not intersect, the resulting ˜ m ′ VE can outperform ˜ m MLE with the appropriate choices of Π and V , as we illustrate in the experiments of Section 5.

<!-- image -->

Because T π are contraction mappings, it is always possible to define a V ′ ⊂ V such that the condition of Proposition 4 holds: we only need to make V ′ sufficiently large to encompass v and the operators' fixed points. But in some cases there exist more structured V ′ : in Appendix A.1 we give an example of a finite state-space MDP in which a sequence v 1 , v 2 = T π v 1 , v 3 = T π ′ v 2 , ... that reaches a specific k -dimensional subspace of R | S | stays there forever. The value equivalence principle provides a mechanism to exploit this type of structure, while conventional model-learning approaches, like MLE, are oblivious to this fact. Although in general we do not have access to V ′ , in some cases this set will be revealed through the very process of enforcing value equivalence. For example, if ˜ m is being learned online based on a sequence v 1 , v 2 = T π v 1 , v 3 = T π ′ v 2 , ... , as long as the sequence reaches a v i ∈ V ′ we should expect ˜ m to eventually specialize to V ′ [13, 33].

Another possibility is to exploit geometric properties of the value functions glyph[vector] V v . It is known that the set of all value functions of a given MDP forms a polytope ¨ V ⊂ V [11]. Even though the sequence glyph[vector] V v an algorithm generates may not be strictly inside the polytope ¨ V , this set can still serve as a reference in the definition of V . For example, based on Proposition 1, we may want to define a V that spans as much of the polytope ¨ V as possible [5]. This suggests that the functions in V should be actual value functions v π associated with policies π ∈ GLYPH&lt;5&gt; . In Section 5 we show experiments that explore this idea.

We now consider structure in the solution . Most large-scale applications of model-based RL use function approximation. Suppose the agent can only represent policies π ∈ ˜ Π and value functions v ∈ ˜ V . Then, a value equivalent model ˜ m ∈ M ( ˜ Π , ˜ V ) is as good as any model. To build intuition, suppose the agent uses state aggregation to approximate the value function. In this case two models

with the same transition probabilities between clusters of states are indistinguishable from the agent's point of view. It thus makes sense to build V using piecewise-constant functions that belong to the space of function representable by the agent, v ∈ ˜ V . The following remark generalises this intuition:

Remark 2. Suppose the agent represents the value function using a linear function approximation: ˜ V = { ˜ v | ˜ v ( s ) = ∑ d i =1 φ i ( s ) w i } , where φ i : S ↦→ R are fixed features and w ∈ R d are learnable parameters. In addition, suppose the agent can only represent policies π ∈ ˜ Π . Then, Proposition 1 implies that if we use the features themselves as the functions adopted with value equivalence, V = { φ i } d i =1 , we have that M ( ˜ Π , { φ i } d i =1 ) = M ( ˜ Π , ˜ V ) . In other words, models that are value equivalent with respect to the features are indiscernible to the agent.

According to the remark above, when using linear function approximation, a model that is value equivalent with respect to the approximator's features will perform no worse than any other model. This prescribes a concrete way to leverage the value equivalence principle in practice, since the set of functions V is automatically defined by the choice of function approximator. Note that, although the remark is specific to linear value function approximation, it applies equally to linear and non-linear models (this is in contrast with previous work showing the equivalence between model-free RL using linear function approximation and model-based RL with a linear model for expected features [27, 38]). The principle of finding a basis for ˜ V also extends to non-linear value function approximation, though in this case it is less clear how to define a set V that spans ˜ V . One strategy is to sample the functions to be included in V from the set ˜ V of (non-linear) functions the agent can represent. Despite its simplicity, this strategy can lead to good performance in practice, as we show next.

## 5 Experiments

We now present experiments illustrating the usefulness of the value equivalence principle in practice. Specifically, we compare models computed based on value equivalence (VE) with models resulting from maximum likelihood estimation (MLE). All our experiments followed the same protocol: ( i ) we collected sample transitions from the environment using a policy that picks actions uniformly at random, ( ii ) we used this data to learn an approximation ˜ r using (4) as well as approximations ˜ p using either MLE (5) or VE (7), ( iii ) we learned a policy ˜ π based on ˜ m = (˜ r, ˜ p ) , and ( iv ) we evaluated ˜ π on the actual environment. The specific way each step was carried out varied according to the characteristics of the environment and function approximation used; see App. A.2 for details.

One of the central arguments of this paper is that the value equivalence principle can yield a better allocation of the limited resources of model-based agents. In order to verify this claim, we varied the representational capacity of the agent's models ˜ m and assessed how well MLE and VE performed under different constraints. As discussed, VE requires the definition of two sets: Π and V . It is usually easy to define a set of policies Π such that p -span(Π) ⊇ GLYPH&lt;5&gt; ; since all the environments used in our experiments have a finite action space A , we accomplished that by defining Π = { π a } a ∈A where π a ( a | s ) = 1 for all s ∈ S . We will thus restrict our attention to the impact of the set of functions V .

As discussed, one possible strategy to define V is to use actual value functions in an attempt to span as much as possible of the value polytope ¨ V [5]. Figure 3 shows results of VE when using this strategy. Specifically, we compare VE's performance with MLE's on two well known domains: 'four rooms' [37] and 'catch' [25]. For each domain, we show two types of results: we either fix the capacity of the model ˜ p and vary the size of V or vice-versa (in the Appendix we show results with all possible combinations of model capacities and sizes of V ). Note how the models produced by VE outperform MLE's counterparts across all scenarios, and especially so under stricter restrictions on the model. This corroborates our hypothesis that VE yields models that are tailored to future use.

Another strategy to define V is to use functions from ˜ V , the space of functions representable by the agent, in order to capture as much as possible of this space. In Figure 4 we compare VE using this strategy with MLE. Here we use as domains catch and 'cart-pole' [4] (but see Appendix for the same type of result on the four-rooms environment). As before, VE largely outperforms MLE, in some cases with a significant improvement in performance. We call attention to the fact that in cart-pole we used neural networks to represent both the transition models ˜ p and the value functions ˜ v , which indicates that VE can be naturally applied with nonlinear function approximation.

It is important to note the broader significance of our experiments. While our theoretical analysis of value equivalence focused on the case where M contained a value equivalent model, this is not

Figure 3: Results with V composed of true value functions of randomly-generated policies. The models ˜ p are rank-constrained transition matrices ˜ P = DK , with D ∈ R |S|× k , K ∈ R k ×|S| , and k &lt; |S| . Error bars are one standard deviation over 30 runs.

<!-- image -->

Figure 4: Results with V composed of functions sampled from the agent's representational space ˜ V . (a-b) Functions in V are the features of the linear function approximation (state aggregation), as per Remark 2. Models ˜ p are rank-constrained transition matrices ( cf. Figure 3). (c-d) Functions in V are randomly-generated neural networks. Models ˜ p are neural networks with rank-constrained linear transformations between layers (Appendix A.2). Error bars are one standard deviation over 30 runs.

<!-- image -->

guaranteed in practice. Our experiments illustrate that, in spite of lacking such a guarantee, we see a considerable gap in performance between VE and MLE, indicating that VE models still offer a strong benefit. Our goal here was to provide insight into the value equivalence principle; in the next section we point to prior work to demonstrate the utility of value equivalence in large-scale settings.

## 6 The value-equivalence principle in practice

Recently, there have been several successful empirical works that can potentially be understood as applications of the value-equivalence principle, like Silver et al.'s [34] Predictron , Oh et al.'s [26] Value Prediction Networks , Farquhar et al.'s [16] TreeQN , and Schrittwieser et al.'s [33] MuZero . Specifically, the model-learning aspect of these prior methods can be understood, with some abuse of notation, as a value equivalence principle of the form T v = ˜ T v , where T is a Bellman operator applied with the true model m ∗ and ˜ T is a Bellman operator applied with an approximate model ˜ m .

There are many possible forms for the operators T and ˜ T . First, value equivalence can be applied to an uncontrolled Markov reward process; the resulting operator T π is analogous to having a single policy in Π . Second, it can be applied over n steps, using a Bellman operator T n π that rolls the model forward n steps: T n π [ v ]( s ) = E π [ R t +1 + ... + γ n -1 R t + n + γ n v π ( S t + n ) | S t = s ] , or a λ -weighted average T λ π [6]. Third, a special case of the n -step operator T a 1 ...a n can be applied to an open-loop action sequence { a 1 , ..., a n } . Fourth, it can be applied to the Bellman optimality operator, T G v , where G v is the 'greedy' policy induced by v defined as G v ( a | s ) = ✶ { a = argmax a ′ E [ R + γv ( S ′ ) | s, a ′ ] } . This idea can also be extended to an n -step greedy search operator, T G n v [ v ]( s ) = max a 1 ,...,a n E [ R t +1 + ... + γ n -1 R t + n + γ n v ( S t + n ) | S t = s, A t = a 1 , ..., A t + n = a n ] . Finally, instead of applying value equivalence over a fixed set of value functions V , we can have a set V t that varies over time-for example, V t can be a singleton with an estimate of the value function of the current greedy policy.

The two operators T and ˜ T can also differ. For example, on the environment side we can use the optimal value function, which can be interpreted as T ∞ v = v ∗ [40, 34], while the approximate operator can be ˜ T λ π [34] or ˜ T n G v t [40]. We can also use approximate values ˜ T v ≈ T v ′ where v ′ ≈ v ,

for example by applying n -step operators to approximate value functions, ˜ T n v ≈ T n v ′ = T n T k v = T n + k v [26, 33] or ˜ T n v ≈ T n v ′ = T n ˜ T k v [16], or even to approximate policies, ˜ T n v a ≈ T n v ′ a where v a = π ( a | s ) ≈ π ′ ( a | s ) = v ′ a for all a ∈ A [33]. The table below characterises the type of value equivalence principle used in prior work. We conjecture that this captures the essential idea underlying each method for model-learning, acknowledging that we ignore many important details.

| Algorithm       | Operator ˜ T         | Policies Π              | Functions V                                  |
|-----------------|----------------------|-------------------------|----------------------------------------------|
| Predictron [34] | ˜ T λ v t            | None                    | Value functions for pseudo-rewards           |
| VIN [40]        | ˜ T n G v t v t      | G v t                   | Value function                               |
| TreeQN [16]     | ˜ T n G n v t v t    | G n v t                 | Value function                               |
| VPN [26]        | ˜ T n a 1 ..a n v t  | { a 1 , ...,a n } ∼ π t | Value function                               |
| MuZero [33]     | ˜ T n a 1 ...a n v t | { a 1 , ...,a n } ∼ π t | Distributional value bins, policy components |

All of these methods, with the exception of VIN, sample the Bellman operator, rather than computing full expectations ( c.f. (7)). In addition, all of the above methods jointly learn the state representation alongside a value-equivalent model based upon that representation. Only MuZero includes both many policies and many functions, which may be sufficient to approximately span the policy and function space required to plan in complex environments; this perhaps explains its stronger performance.

## 7 Related work

Farahmand et al.'s [14, 15] value-aware model learning (VAML) is based on a premise similar to ours. They study a robust variant of (6) that considers the worst-case choice of v ∈ V and provide the gradient when the value-function approximation ˜ v is linear and the model ˜ p belongs to the exponential family. Later, Farahmand [13] also considered the case where the model is learned iteratively. Both versions of VAML come with finite sample-error upper bound guarantees [14, 15, 13]. More recently, Asadi et al. [2] showed that minimizing the V AML objective is equivalent to minimizing the Wasserstein metric. Abachi et al. [1] applied the V AML principle to policy gradient methods. The theory of VAML is complementary to ours: we characterise the space of value-equivalent models, while VAML focuses on the solution and analysis of the induced optimization problem.

Joseph et al. [22] note that minimizing prediction error is not the same as maximizing the performance of the resulting policy, and propose an algorithm that optimizes the parameters of the model rather than the policy's. Ayoub et al. [3] proposes an algorithm that keeps a set of models that are consistent with the most recent value function estimate. They derive regret bounds for the algorithm which suggest that value-targeted regression estimation is both sufficient and efficient for model-based RL.

More broadly, other notions of equivalence between MDPs have been proposed in the literature [12, 28, 20, 31, 17, 23, 41, 29, 8, 42]. Any notion of equivalence over states can be recast as a form of state aggregation; in this case the functions mapping states to clusters can (and probably should) be used to enforce value equivalence (Remark 2). But the principle of value equivalence is more general: it can be applied with function approximations other than state aggregation and can be used to exploit structure in the problem even when there is no clear notion of state abstraction (Appendix A.1.2).

In this paper we have assumed that the agent has access to a well-defined notion of state s ∈ S . More generally, the agent only receives observations from the environment and must construct its own state function-that is, a mapping from histories of observations to features representing states. This is an instantiation of the problem known as representation learning [43, 21, 9, 18, 45, 44, 19, 7]. An intriguing question which arises in this context is whether a model learned through value equivalence induces a space of 'compatible' state representations, which would suggest that the loss (6) could also be used for representation learning. This may be an interesting direction for future investigations.

## 8 Conclusion

We introduced the principle of value equivalence: two models are value equivalent with respect to a set of functions and a set of policies if they yield the same updates of the former on the latter. Value equivalence formalizes the notion that models should be tailored to their future use and provides a mechanism to incorporate such knowledge into the model learning process. It also unifies some important recent work in the literature, shedding light on their empirical success. Besides helping to explain some past initiatives, we believe the concept of value equivalence may also give rise to theoretical and algorithmic innovations that leverage the insights presented.

## Broader impact

The bulk of the research presented in this paper consists of foundational theoretical results about the learning of models for model-based reinforcement learning agents. While applications of these agents can have social impacts depending upon their use, our results merely serve to illuminate desirable properties of models and facilitate the subsequent training of agents using them. In short, this work is largely theoretical and does not present any foreseeable societal impact, except in the general concerns over progress in artificial intelligence.

## Acknowledgements

We would like to thank Gregory Farquhar and Eszter Vertes for the great discussions regarding the value equivalence principle. We also thank the anonymous reviewers for their comments and suggestions to improve the paper.

## References

- [1] Romina Abachi, Mohammad Ghavamzadeh, and Amir massoud Farahmand. Policy-Aware Model Learning for Policy Gradient Methods. arXiv preprint cs.AI:2003.00030 , 2020.
- [2] Kavosh Asadi, Evan Cater, Dipendra Misra, and Michael L. Littman. Equivalence Between Wasserstein and Value-Aware Model-Based Reinforcement Learning. In FAIM Workshop on Prediction and Generative Modeling in Reinforcement Learning , 2018.
- [3] Alex Ayoub, Zeyu Jia, Csaba Szepesvári, Mengdi Wang, and Lin Yang. Model-Based Reinforcement Learning with Value-Targeted Regression. In Proceedings of the International Conference on Machine Learning (ICML) , 2020.
- [4] Andrew G. Barto, Richard S. Sutton, and Charles W Anderson. Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. IEEE Transactions on Systems, Man, and Cybernetics , pages 834-846, 1983.
- [5] Marc Bellemare, Will Dabney, Robert Dadashi, Adrien Ali Taiga, Pablo Samuel Castro, Nicolas Le Roux, Dale Schuurmans, Tor Lattimore, and Clare Lyle. A Geometric Perspective on Optimal Representations for Reinforcement Learning. In Advances in Neural Information Processing Systems , pages 4360-4371, 2019.
- [6] Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- [7] Ondrej Biza, Robert Platt, Jan-Willem van de Meent, and Lawson LS Wong. Learning Discrete State Abstractions with Deep Variational Inference. arXiv preprint arXiv:2003.04300 , 2020.
- [8] Pablo Samuel Castro. Scalable Methods for Computing State Similarity in Deterministic Markov Decision Processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 10069-10076, 2020.
- [9] Dane Corneil, Wulfram Gerstner, and Johanni Brea. Efficient Model-Based Deep Reinforcement Learning with Variational State Tabulation. arXiv preprint arXiv:1802.04325 , 2018.
- [10] Will Dabney, André Barreto, Mark Rowland, Robert Dadashi, John Quan, Marc G. Bellemare, and David Silver. The Value-Improvement Path: Towards Better Representations for Reinforcement Learning, 2020.
- [11] Robert Dadashi, Adrien Ali Taiga, Nicolas Le Roux, Dale Schuurmans, and Marc G. Bellemare. The Value Function Polytope in Reinforcement Learning. In Proceedings of the International Conference on Machine Learning (ICML) , volume 97, pages 1486-1495, 2019.
- [12] Thomas Dean and Robert Givan. Model Minimization in Markov Decision Processes. In AAAI/IAAI , pages 106-111, 1997.

- [13] Amir-massoud Farahmand. Iterative Value-Aware Model Learning. In Advances in Neural Information Processing Systems (NeurIPS) , pages 9090-9101, 2018.
- [14] Amir-Massoud Farahmand, André Barreto, and Daniel Nikovski. Value-Aware Loss Function for Model Learning in Reinforcement Learning. In Proceedings of the European Workshop on Reinforcement Learning (EWRL) , 2013.
- [15] Amir-Massoud Farahmand, André Barreto, and Daniel Nikovski. Value-Aware Loss Function for Model-Based Reinforcement Learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 54, pages 1486-1494, 2017.
- [16] G Farquhar, T Rocktäschel, M Igl, and S Whiteson. TreeQN and ATreeC: Differentiable Tree-Structured Models for Deep Reinforcement Learning. In 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings , volume 6. ICLR, 2018.
- [17] Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for Finite Markov Decision Processes. In UAI , volume 4, pages 162-169, 2004.
- [18] Vincent François-Lavet, Yoshua Bengio, Doina Precup, and Joelle Pineau. Combined Reinforcement Learning via Abstract Representations. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 3582-3589, 2019.
- [19] Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G Bellemare. DeepMDP: Learning Continuous Latent Space Models for Representation Learning. In International Conference on Machine Learning , pages 2170-2179, 2019.
- [20] Robert Givan, Thomas Dean, and Matthew Greig. Equivalence Notions and Model Minimization in Markov Decision Processes. Artificial Intelligence , 147(1-2):163-223, 2003.
- [21] Maximillian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, and Shimon Whiteson. Deep Variational Reinforcement Learning for POMDPs. In ICML 2018: Proceedings of the ThirtyFifth International Conference on Machine Learning , July 2018. URL http://www.cs.ox. ac.uk/people/shimon.whiteson/pubs/iglicml18.pdf .
- [22] Joshua Joseph, Alborz Geramifard, John W Roberts, Jonathan P How, and Nicholas Roy. Reinforcement Learning with Misspecified Model Classes. In 2013 IEEE International Conference on Robotics and Automation , pages 939-946. IEEE, 2013.
- [23] Lihong Li, Thomas J. Walsh, and Michael L. Littman. Towards a Unified Theory of State Abstraction for MDPs. In Proceedings of the International Symposium on Artificial Intelligence and Mathematics , pages 531-539, 2006.
- [24] Russell B. Millar. Maximum Likelihood Estimation and Inference . Hoboken: Wiley, 2011.
- [25] Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu. Recurrent Models of Visual Attention. In Advances in Neural Information Processing Systems (NIPS) , pages 2204-2212, 2014.
- [26] Junhyuk Oh, Satinder Singh, and Honglak Lee. Value Prediction Networks. In Advances in Neural Information Processing Systems , pages 6118-6128, 2017.
- [27] Ronald Parr, Lihong Li, Gavin Taylor, Christopher Painter-Wakefield, and Michael L. Littman. An Analysis of Linear Models, Linear Value-Function Approximation, and Feature Selection for Reinforcement Learning. In Proceedings of the International Conference on Machine Learning (ICML) , pages 752-759, 2008.
- [28] Pascal Poupart and Craig Boutilier. Value-Directed Compression of POMDPs. In Advances in Neural Information Processing Systems (NIPS) , pages 1547-1554. MIT Press, 2002.
- [29] Pascal Poupart and Craig Boutilier. Value-Directed Belief State Approximation for POMDPs. CoRR , abs/1301.3887, 2013. URL http://arxiv.org/abs/1301.3887 .
- [30] Martin L. Puterman. Markov Decision Processes-Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, Inc., 1994.

- [31] Balaraman Ravindran and Andrew G Barto. Approximate Homomorphisms: A Framework for Non-Exact Minimization in Markov Decision Processes. 2004.
- [32] Stuart J. Russell and Peter Norvig. Artificial Intelligence: A Modern Approach . Pearson Education, 3 edition, 2003.
- [33] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. arXiv preprint arXiv:1911.08265 , 2019.
- [34] David Silver, Hado van Hasselt, Matteo Hessel, Tom Schaul, Arthur Guez, Tim Harley, Gabriel Dulac-Arnold, David Reichert, Neil Rabinowitz, Andre Barreto, et al. The Predictron: End-toEnd Learning and Planning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 3191-3199. JMLR. org, 2017.
- [35] Richard S. Sutton. Learning to Predict by the Methods of Temporal Differences. Machine Learning , 3:9-44, 1988.
- [36] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . MIT Press, 2018. URL https://mitpress.mit.edu/books/ reinforcement-learning-second-edition . 2nd edition.
- [37] Richard S Sutton, Doina Precup, and Satinder Singh. Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. Artificial intelligence , 112 (1-2):181-211, 1999.
- [38] Richard S. Sutton, Csaba Szepesvári, Alborz Geramifard, and Michael Bowling. Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping. In Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI) , page 528-536, 2008.
- [39] Csaba Szepesvári. Algorithms for Reinforcement Learning . Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan &amp; Claypool Publishers, 2010.
- [40] Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value Iteration Networks. In Advances in Neural Information Processing Systems , pages 2154-2162, 2016.
- [41] Jonathan Taylor, Doina Precup, and Prakash Panagaden. Bounding Performance Loss in Approximate MDP Homomorphisms. In Advances in Neural Information Processing Systems (NIPS) , pages 1649-1656, 2009.
- [42] Elise van der Pol, Thomas Kipf, Frans A Oliehoek, and Max Welling. Plannable Approximations to MDP Homomorphisms: Equivariance under Actions. In International Conference on Autonomous Agents and MultiAgent Systems , 2020.
- [43] Manuel Watter, Jost Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images. In Advances in Neural Information Processing Systems (NIPS) , pages 2746-2754, 2015.
- [44] Amy Zhang, Zachary C Lipton, Luis Pineda, Kamyar Azizzadenesheli, Anima Anandkumar, Laurent Itti, Joelle Pineau, and Tommaso Furlanello. Learning Causal State Representations of Partially Observable Environments. arXiv preprint arXiv:1906.10437 , 2019.
- [45] Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew Johnson, and Sergey Levine. SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning. In International Conference on Machine Learning (ICML) , pages 7444-7453. PMLR, 2019.

## The Value-Equivalence Principle for Model-Based Reinforcement Learning

## Supplementary Material

## Christopher Grimm

## André Barreto, Satinder Singh, David Silver

Computer Science &amp; Engineering University of Michigan crgrimm@umich.edu

DeepMind

{andrebarreto,baveja,davidsilver}@google.com

In this supplement we give details of our theoretical results and experiments that had to be left out of the main paper due to space constraints. We prove our theoretical results and provide a detailed description of our experimental procedure. Importantly, we present an illustrative example showing how value equivalence (VE) may lead to a better solution for a Markov decision process (MDP) than maximum-likelihood estimate (MLE). We show this to be true both in the exact case, when there exist a value-equivalent model in the model class considered, and in the approximate case, when such a model does not exist in the model class. Our appendix is organized as follows:

- Section A.1.1 contains derivations of the properties and propositions presented in the main text.
- Section A.1.2 contains a sequence of examples using a toy MDP that illustrate points made in the discussion surrounding Propositions 3 and 4. Moreover, we include an additional result which illustrates a situation in which approximate VE models can outperform the MLE model.
- Section A.2 provides a detailed outline of the pipeline used across our experiments in the main text. We also report several additional results that had to be left out of the main paper due to space constraints.

The numbering of equations, figures and citations resume from what is used in the main paper.

## A Appendix

## A.1 Proofs of theoretical results and illustrative examples

## A.1.1 Proofs

Property 1. Given M ′ ⊆ M , we have that M ′ (Π , V ) ⊆ M (Π , V ) .

Proof.

<!-- formula-not-decoded -->

Property 2. M ( GLYPH&lt;5&gt; , V ) either contains m ∗ or is the empty set.

<!-- formula-not-decoded -->

Property 3. Given Π ′ ⊆ Π and V ′ ⊆ V , we have that M (Π , V ) ⊆ M (Π ′ , V ′ ) .

glyph[negationslash]

Proof. We will show the result by contradiction. Suppose there is a model ˜ m ∈ M (Π , V ) such that ˜ m / ∈ M (Π ′ , V ′ ) . This means that there exists a π ∈ Π ′ and a v ∈ V ′ for which ˜ T π v = T π v . But since Π ′ ⊆ Π and V ′ ⊆ V , it must be the case that π ∈ Π and v ∈ V , which contradicts the claim that ˜ m ∈ M (Π , V ) .

Property 4. If m ∗ ∈ M , then m ∗ ∈ M (Π , V ) for all Π and all V .

<!-- formula-not-decoded -->

Proposition 1. For discrete Π and V , we have that M (Π , V ) = M ( p -span(Π) ∩ GLYPH&lt;5&gt; , span( V )) .

Proof. Let π ∈ p -span(Π) ∩ GLYPH&lt;5&gt; . Based on (3), we know that there exists an α s ∈ R | Π | such that π ( ·| s ) = ∑ i α si π i ( ·| s ) , where π i ∈ Π . Thus, for ˜ m ∈ M (Π , V ) , we can write

<!-- formula-not-decoded -->

Let v ∈ span( V ) . We know there is a β ∈ R |V| such that v = ∑ i β i v i , with v i ∈ V .

<!-- formula-not-decoded -->

In order to prove Proposition 2 we will need four lemmas which we state and prove below.

Lemma 1. For arbitrary matrices A ∈ R k × n , C ∈ R m × glyph[lscript] , we can construct a vector-space B = { B ∈ R n × m : ABC = 0 } where 0 denotes a k × glyph[lscript] matrix of zeros. It follows that

<!-- formula-not-decoded -->

Proof. We begin by converting the condition ABC = 0 into a matrix-vector product. Let a i and c j denote the i'th row of A and j'th column of C respectively. Observe that ( ABC ) ij = a i Bc j = ∑ x,y a i x c j y B xy , which implies that

<!-- formula-not-decoded -->

where [ k ] denotes { 1 , . . . , k } .

For each ( i, j ) pair, the above expression is suggestive of a dot-product between two n × m vectors: a combination of a i and c j , and a 'flattened' version of B . Define the former combination of vectors as d ij = [ a i 1 c j 1 , a i 1 c j 2 , · · · , a i n c j m ] glyph[latticetop] ∈ R nm × 1 , and stack them as rows as: D = [ d 11 , d 12 , · · · , d nm ] glyph[latticetop] ∈ R kglyph[lscript] × nm . To flatten B , simply define b = [ B 11 , B 12 , · · · , B nm ] glyph[latticetop] ∈ R nm × 1 .

We now have that ABC = 0 ⇐⇒ Db = 0 . Moreover, unravelling the matrices in B does not change the dimension of the space, thus:

<!-- formula-not-decoded -->

where the last equality comes from a application of the rank-nullity theorem.

Finally notice that the construction of d ij can be thought of as vertically stacking n copies of c j each scaled by a different entry in a i . We can also find scaled copies of a i by c j k in d ij by selecting indices from the combined vector at regular intervals of m : d ij k +( glyph[lscript] -1) m = c j k · a i glyph[lscript] for glyph[lscript] ∈ { 1 , . . . n } .

This means that scaled copies of both a i and c j can be found by selecting specific groups of indices in d ij . It follows that if a 1 , . . . , a n are linearly independent then so are d 1 j , . . . , d nj for any j . And similarly, if c 1 , . . . , c m are linearly independent then so are d i 1 , . . . , d im for any i . Hence if a 1 , . . . a n and c 1 , . . . , c m are both linearly independent sets, then so is d 11 , d 12 , . . . , d nm . Since these a i and c j vectors form the rows and columns of rank n and m matrices: A and C , their corresponding sets of row and column vectors are linearly independent. Thus we have that rank( D ) = rank( A ) · rank( C ) , completing the proof.

Lemma 2. For any c and Y + c = { y + c : y ∈ Y} it follows that dim[ Y + c ] = dim[ Y ] .

Proof.

<!-- formula-not-decoded -->

Lemma 3. If Y is a vector-space then H -dim[ Y ] = dim[ Y ] .

Proof. Recall the definition of dim[ Y ] :

<!-- formula-not-decoded -->

where W is a vector-space. By choosing W = Y and c = 0 we see that dim[ Y ] ≤ H -dim[ Y ] .

Suppose then that dim[ Y ] &lt; H -dim[ Y ] . This implies that there is a vector space W and offset c with d = H -dim[ W ] &lt; H -dim[ Y ] and Y + c ⊆ W . This means that for every y ∈ Y : y + c = ∑ d i =1 α y i w i for some α y 1: d where w 1: d are a basis of W . Since Y is a vector space it must contain the 0 vector, hence c = ∑ d i =1 α 0 i w i . Accordingly any y ∈ Y can be written as y = ∑ d i =1 ( α y i -α 0 i ) w i . However, this is a contradiction since H -dim[ W ] &lt; H -dim[ Y ] . Hence dim[ Y ] = H -dim[ Y ] .

Lemma 4. If X ⊆ Y then dim[ X ] ≤ dim[ Y ] .

Proof. If X ⊆ Y then for any c , X + c ⊆ Y + c . Because of the above, for any vector-space W : W ⊇ Y + c = ⇒ W ⊇ X + c , hence: { ( W , c ) : X + c ⊂ W} ⊇ { ( W , c ) : Y + c ⊂ W} . Notice that this last set-relation corresponds the set of vector-spaces that dim[ · ] is minimizing over for X and Y respectively. Hence dim[ X ] ≤ dim[ Y ] .

Proposition 2. Let Π be a set of m pointwise linearly independent policies π i ∈ R |S||A| and let V be a set of k linearly independent vectors v i ∈ R |S| . Then,

<!-- formula-not-decoded -->

Proof. First note that if π i / ∈ p -span(Π \ { π i } ) then π i / ∈ span(Π \ { π i } ) . Hence, pointwise linear independence implies linear independence.

Since |S| and |A| are finite, we can assume that A = { 1 , . . . , |A||} and S = { 1 , . . . , |S|} . For any transition probability kernel ˜ p ( s ′ | s, a ) we can construct matrix ˜ P ∈ R |S||A|×|S| with ˜ P ( a -1) |S| + s,s ′ = ˜ p ( s ′ | s, a ) . Denote the constructed matrix corresponding to the true dynamics as P . For any π i we can construct a matrix Π i ∈ R | S |×| S || A | with ( Π i ) s, ( a -1) |S| + s = π i ( a | s ) . Vertically stack these m Π i matrices to construct Π ∈ R m | S |×| S || A | . Additionally we construct V ∈ R | S |× k with V j,glyph[lscript] = ( v glyph[lscript] ) j . Note that P (Π , V ) = { ˜ P ∈ P : Π ( ˜ P -P ) V = 0 } . Define the sets X = { X ∈ R | S || A |×| S | : PXV = 0 } and Y = { ˜ P ∈ R | S || A |×| S | : Π ( ˜ P -P ) V = 0 } .

Note the following three facts:

1. dim[ X ] = dim[ Y ] since our notion of dimension is translation-invariant (Lemma 2).
2. dim[ X ] = H -dim[ X ] since X is a vector-space (Lemma 3).
3. P (Π , V ) ⊆ Y which implies that dim[ P (Π , V )] ≤ dim[ Y ] (Lemma 4).

Taken together this gives us that

<!-- formula-not-decoded -->

We can now apply Lemma 1 to obtain dim[ X ] = | S | 2 | A | -k · rank( Π ) . Notice that rank( Π ) = min {| S || A | , m | S |} . Thus dim[ P (Π , V )] ≤ | S | ( | S || A | -mk ) as needed.

glyph[negationslash]

Proposition 3. The maximum-likelihood estimate of p ∗ in P may not belong to a P (Π , V ) = ∅ .

Proof. Suppose we are trying to estimate a transition matrix P ∈ R n × n and choose to use one parameter θ i ∈ R per row. Specifically, we parametrize the distribution on the i -th row as glyph[negationslash]

<!-- formula-not-decoded -->

where p ij = p ( s j | s i ) . We can then write the expected likelihood function for θ ∈ R n as glyph[negationslash]

glyph[negationslash]

<!-- formula-not-decoded -->

which leads to the likelihood equation

<!-- formula-not-decoded -->

The MLE solution is thus to have θ i = p ii for i = 1 , 2 , ..., n . This means that the solution provided by MLE will not be exact if and only if glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

Now, suppose we have V = { v } with v i = 1 for some i and v j = 0 for j = i . In this case it is possible to get an exact value-equivalent solution-that is, Pv = ˜ Pv -by making θ i = p ii and θ j = 1 -( n -1) p ii for j = i , regardless of whether (11) is true or not.

glyph[negationslash]

Proposition 4. Suppose v ∈ V ′ = ⇒ T π v ∈ V ′ for all π ∈ GLYPH&lt;5&gt; . Let p -span(Π) ⊇ GLYPH&lt;5&gt; and span( V ) = V ′ . Then, starting from any v ′ ∈ V ′ , any ˜ m ∈ M (Π , V ) yields the same solution as m ∗ .

Proof. Denote the Bellman operator under a policy that always selects action a as T a , the greedy Bellman operator as T v = max a T a v and the Bellman operator under a policy π as T π , as before. Let T ( n ) v represent n successive applications of operator T on value v .

Note that for any v ∈ V we can construct a π v ( s ) = argmax a ( T a v )( s ) such that T v = max a T a v = T π v v . This implies that the greedy Bellman operator is included in the assumption of our proposition:

<!-- formula-not-decoded -->

We now begin by showing that:

<!-- formula-not-decoded -->

for any v ∈ V and any n &gt; 0 . Assume that T ( n ) v = ˜ T ( n ) v ∈ V ′ . Since T ( n ) v ∈ V ′ and V ′ = span( V ) , we can use use value equivalence to obtain:

<!-- formula-not-decoded -->

for any a ∈ A . Next, since T ( n ) v = ˜ T ( n ) v we can write:

<!-- formula-not-decoded -->

Since (14) holds for any a ∈ A , we can write:

<!-- formula-not-decoded -->

We know from (12) that the fact that T ( n ) v ∈ V ′ implies that T ( n +1) v ∈ V ′ . Thus we have shown that (13) is true.

Finally, by choosing v ∈ V ′ and using analogous reasoning as as above, we can show that T a v = ˜ T a v and T v = max a T a v = max a ˜ T a v = ˜ T v , and since v ∈ V ′ , ˜ T v = T v ∈ V ′ . Thus T ( n ) v = T ( n ) v for all n ∈ N . This is sufficient to conclude that

<!-- formula-not-decoded -->

as needed.

## A.1.2 Examples with a simple MDP

Consider the 3 state MDP with states s 1 , s 2 , s 3 and actions A = { a 1 , a 2 } . Transitioning to state s 1 always incurs a reward of 1 , taking any action in states s 2 and s 3 always results in transitioning to s 1 and taking action a ∈ A from s 1 transitions among the other states according to action-dependent distribution ( p a 11 , p a 12 , p a 13 ) . This MDP is depicted in Figure 5. We now use this MDP to illustrate several points made in the main text.

Closure under Bellman updates We now address the discussion surrounding Proposition 4 in the main text.

Consider a the following two-dimensional subspace of value functions R = { [ x, y, y ] glyph[latticetop] : x, y ∈ R } . We now show that, for the MDP described above, R exhibits closure under arbitrary Bellman updates.

For an arbitrary policy π : S ↦→ P ( A ) the Bellman update for a value function v ∈ R 3 is given by T π v = R π + γ P π v where

<!-- formula-not-decoded -->

Suppose v ∈ R , then v = [ a, b, b ] glyph[latticetop] for some a, b ∈ R . Notice that for such a value function the following holds:

<!-- formula-not-decoded -->

thus we have illustrated that the two-dimensional subspace R is closed under arbitrary Bellman updates in our 3 state MDP. This means that, once a sequence v 1 , v 2 = T π v 1 , v 3 = T π ′ v 2 ... reaches a v i ∈ R , it stays in R . We can then exploit this property finding value-equivalent models with respect to R , as we show next.

Amodel class for which exact VEoutperforms MLE Wenowprovide an example of the scenario discussed around Proposition 3 in the main text by examining the setting where a model, from a restricted class, must be learned to approximate the dynamics of our MDP. We restrict our model class by requiring that for each action a ∈ A we represent ( p a 11 , p a 12 , p a 13 ) as ((1 -θ a ) / 2 , θ a , (1 -θ a ) / 2) . Before continuing we note a few properties of value functions of our MDP. Notice that for any v π we can write:

<!-- formula-not-decoded -->

which illustrates that v π exclusively depends on the value of P π 11 ≡ ∑ a ∈A π ( a | s 1 ) p a 11 .

First we consider the MLE solution to this problem: it can be easily shown (see the proof of Proposition 3) that, for the model class defined above, θ a = p a 12 for all a ∈ A maximizes the likelihood. However notice that this implies that our approximation of p a 11 equals (1 -p a 12 ) / 2 which is clearly not true in general. Thus, there are settings of ( p a 11 , p a 12 , p a 13 ) and policies for which the value function produced by MLE, ˜ v π , is not equivalent to the true value function v π .

Next we consider learning a value-equivalent model with the same restricted model class. Suppose we wish our model to be value equivalent to value v = [1 , 0 , 0] glyph[latticetop] and all policies.

Note that any VE model with respect to V = { v } : { ˜ P a } a ∈A , must satisfy ˜ P a v = P a v . By requiring value equivalence with just v we have:

<!-- formula-not-decoded -->

which implies that ˜ p a 11 = p 11 , ˜ p a 21 = ˜ p a 31 = 1 and ˜ p a 22 = ˜ p a 23 = ˜ p a 32 = ˜ p a 33 = 0 for all a ∈ A .

Figure 5

<!-- image -->

Taking these constraints together restricts the class of VE models to those of the form:

<!-- formula-not-decoded -->

where ˜ p a 1 i are 'free variables' for all i = 2 , 3 and a ∈ A .

Notice that when p a 11 ≤ 0 . 5 for all a ∈ A , we can find a value equivalent model by setting: (1 -θ a ) / 2 = p a 11 . This means that the values produced by these value equivalent models exactly match those of the environment: ˜ v π = v π for all π (and thus the solution of this model also coincides with the optimal value function, ˜ v ∗ = v ∗ ).

Amodel class for which approximate VEoutperforms MLE In the previous example we showed that it is possible to have an MDP and a restricted model class such that VE models are able to perfectly estimate v ∗ while MLE models fail to do so. Notice that in this example a value equivalent model actually existed , which is not guaranteed in general. We now show a related example where, in spite of an exactly value equivalent model not existing, an agent trained using an approximate value equivalent model will outperform its MLE counterpart.

We use our example MDP from before, shown in Figure 5, and denote its actions A = { a, b } for later notational convenience. We set our environment's transition dynamics accordingly: p a ≡ ( p a 11 , p a 12 , p a 13 ) = (0 . 6 , 0 . 4 , 0 . 0) and p b ≡ ( p b 11 , p b 12 , p b 13 ) = (0 . 4 , 0 . 2 , 0 . 4) . We also use the same model class as above: (˜ p i 11 , ˜ p i 12 , ˜ p i 13 ) = (0 . 5(1 -θ i ) , θ i , 0 . 5(1 -θ i )) for each i ∈ A , being mindful of the boundary conditions θ i ∈ [0 , 1] .

As we saw in the previous example, the MLE estimator for this problem will produce the following approximations: p a MLE = (0 . 3 , 0 . 4 , 0 . 3) , p b MLE = (0 . 4 , 0 . 2 , 0 . 4) .

We now consider what an approximate VE model will produce using the same value as before: v = [1 , 0 , 0] glyph[latticetop] and all policies. Recall that we're optimizing the following loss:

<!-- formula-not-decoded -->

The form of this loss indicates that VE will attempt to minimize the MSE of ˜ p a 11 and ˜ p b 11 separately. Notice that for action a , we cannot perfectly estimate p 11 due to the boundary conditions on θ a . However, VE will still find the closest possible ˜ p 11 that respects the boundary condition, giving: ˜ p a VE = (0 . 5 , 0 . 0 , 0 . 5) , ˜ p b VE = (0 . 4 , 0 . 2 , 0 . 4) .

We now display these models together in the following table:

|     |   ˜ p a 11 |   ˜ p a 12 |   ˜ p a 13 |   ˜ p b 11 |   ˜ p b 12 |   ˜ p b 13 |
|-----|------------|------------|------------|------------|------------|------------|
| MDP |        0.6 |        0.4 |        0   |        0.4 |        0.2 |        0.4 |
| MLE |        0.3 |        0.4 |        0.3 |        0.4 |        0.2 |        0.4 |
| VE  |        0.5 |        0   |        0.5 |        0.4 |        0.2 |        0.4 |

Notice that when optimally planning on this MDP, an agent can obtain the most reward by transitioning from s 1 to s 1 as often as possible. The agent can do this taking the action among { a, b } that is mostly likely to induce a self-transition each time it is at s 1 . In the true environment and the VE model this action is a . However, notice that the MLE model would instead prefer the sub-optimal action b , since (˜ p b MLE ) 11 &gt; (˜ p a MLE ) 11 .

This is a concrete example where VE outperforms MLE even though there is no value-equivalent models in the model class considered (that is, VE can be enforced only approximately).

## A.2 Experimental details

(a) Catch

<!-- image -->

(c) Cart-pole

Figure 6: (a) Catch: the agent has three actions corresponding to moving a paddle (orange) left, right and staying in place. Upon initialization, a ball (blue) is placed at a random square at the top of the environment and at each step it descends by one unit. Upon reaching the bottom of the environment the ball is returned to a random square at the top. The agent receives a reward of 1 . 0 if it moves its paddle and intercepts the ball. (b) Four Rooms: the agent (orange) has four actions corresponding to up, down, left and right movement. When the agent takes an action, it moves in its intended direction with 90% of the time and in an random other direction otherwise. There is a rewarding square in the right top corner (green). If the agent transitions into this square it receives a reward of 1 . 0 . (c) Cart-pole: In Cart-pole, the agent may choose between three actions: pushing the cart to the left, right or not pushing the cart. There is a pole balanced on top of the cart that is at risk of tipping over. The agent is incentivized to keep the pole up-right through a reward of cos( θ ) at each step where θ is the angle of the pole ( θ = 0 implies the pole is perfectly up-right). If the pole's height drops below a threshold, the episode terminates and the agent receives a reward of 0 . 0 . The cart itself is resting on a table; if it falls off the table, the episode similarly terminates with a reward of 0 . 0 .

## A.2.1 Environment description

The environments used in our experiments are described in depth in Figure 6. In both Catch and Four Rooms a tabular representation is employed in which each of the environment's finitely many states (250 and 68, respectively) is represented by an index. In Cart-pole we have a continuous state space S ⊂ R 5 (so |S| = ∞ ). Each state s ∈ R 5 consists of the cart position, cart velocity, sine / cosine of pole angle, and pole's angular velocity.

## A.2.2 Experimental pipeline

As mentioned in the main text, a common experimental pipeline is used across all of our results, with slight variations depending upon the experiment type and environment. This pipeline is described at a high-level below:

- (i) Data collection: Data is collected using a policy which selects actions uniformly at random.
- (ii) Model training: The collected data is used to train a model.
- (iii) Policy construction: The model is used to produce a policy.
- (iv) Policy evaluation: The policy is evaluated to assess the quality of the model.

We now discuss steps (ii), (iii) and (iv) in detail.

(ii) Model training All of our experiments involve restricting the capacity of the class of models that the agent can represent: M . In general we restrict the rank of the models in M , but, depending upon the nature of the model, this restriction is carried in different ways.

1. Tabular models: On domains with | S | &lt; ∞ , we employ tabular models. In what follows, n × m matrices referred to as 'row-stochastic' are ensured to be as such by the following parameterization:
2. (a) A matrix F ∈ R n × m is sampled with entries F ij ∼ Uniform ([ -1 , 1]) .

- (b) A new matrix P F is produced by applying row-wise softmax operations with temperature τ = 1 to F :

<!-- formula-not-decoded -->

Here, F can be thought of as the parameters of P F , which often will suppress as ˜ P for clarity.

That is, a model is represented by | A | | S | × | S | row-stochastic matrices: ˜ P 1 , . . . , ˜ P | A | . We ensure that each of these matrices has rank k by factoring it as follows: ˜ P a = D a K a where D a ∈ R | S |× k , K a ∈ R k ×| S | and both are row-stochastic as well.

2. Neural network models: On domains with | S | = ∞ we instead use a neural network parameterized by θ : f θ : ( S , A ) ↦→ ( S , R ) . f θ takes a state and action as input and outputs an approximation of the expected next state and next reward. As an analogue to the rank restriction applied in the tabular case, we restrict the rank of weight matrices in all fullyconnected layers in f θ . Denote a fully-connected layer in f θ as L ( x ) = σ ( Wx + b ) where σ ( · ) is an activation function, W is a weight matrix and b is a bias term. We restrict f θ by replacing each L ( x ) with L k ( x ) = σ (( DK ) x + b ) where D,K ∈ R | S |× k , R k ×| S | .

The models with the restrictions above are trained based on data collected by a policy that selects actions uniformly at random. With a small abuse of notation, denote the collected data as D = ( s i , a i , r i , s ′ i ) N i =1 . We will now describe how this data is used to train models in different contexts.

1. Tabular models: When training a tabular model with capacity restricted to rank k , we use the following expressions:
2. (a) Reward : In our experiments rewards are represented in the same way for both VE and MLE models:
3. ✶ where ✶ {·} is the indicator function. (b) Transition dynamics (MLE) : To learn the transition dynamics we first parameterize ˜ P a = D a K a for all a ∈ A , where D a and K a are row-stochastic matrices (see item 1 in the section 'Restricting Model Capacity' above). Because we are assuming S to be finite, we can identify each state s ∈ S by an index. Let δ ( s ) ∈ { 1 , ..., | S |} be an index that uniquely identifies state s . We then compute ˜ P a = D a K a by minimizing the following loss with respect to D a and K a :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( D a K a ) ij is the element in the i -th row and j -th column of matrix D a K a . Note that the expression above is the empirical version of expression (5) in the paper [15].

- (c) Transition dynamics (VE) : In the VE setting we have a set of value functions and policies: V and Π . We have one transition matrix ˜ P π associated with each policy π ∈ Π . As discussed in Section 5, in our experiments we used Π = { π a } a ∈A , where π a ( a | s ) = 1 for all s ∈ S . Thus, we end up with the same parameterized probability matrices as above: ˜ P a = D a K a . Let D ia ⊆ D be the sample transitions starting in state i where action a was taken, that is, ( s j , a j , r j , s ′ j ) ∈ D ia if and only if δ ( s j ) = i and a j = a . We computed ˜ P a = D a K a by minimizing the following loss with respect to D a and K a :

<!-- formula-not-decoded -->

Note that the expression above corresponds to equation (7) when learning transition matrices associated with policies { π a } a ∈A in an environment with finite state space S (where states s can be associated with an index i ) and p = 2 .

2. Neural network models : When training a neural network model with capacity restrictions construct a network f θ : ( S , A ) ↦→ ( S , R ) . The network is fully connected and takes the concatenation of S with the one-hot representation of A as input. For a given ( s, a ) pair we denote it's output as ˜ s ′ s,a , ˜ r ′ s,a = f θ ( s, a ) . In all cases we train the neural network model by sampling mini-batches uniformly from D . It is important to note that we only use these neural network models on deterministic domains (e.g., Cart-pole) meaning that the output of the model, ˜ s ′ represents a single state rather than an expectation over states.
2. (a) Reward: For both VE and MLE models we train our neural network models to accurately predict the reward associated with each state action transition:

<!-- formula-not-decoded -->

- (b) Transition dynamics (MSE): Welearn models by encouraging f θ to accurately predict the next state:

<!-- formula-not-decoded -->

- (c) Transition dynamics (VE): For VE models use (7), disregarding reward terms to give:

<!-- formula-not-decoded -->

(iii) Policy construction In each experiment we present, after a model is constructed, we subsequently use it to construct a policy. The manner in which we do this varies based upon the type of the experiment and the nature of the environment. There are three mechanisms for constructing policies from models:

1. Value iteration: For experiments with V = V (which are performed only with tabular models), we use the learned model ˜ m = (˜ r, ˜ p ) to perform value iteration until convergence, yielding ˜ v ∗ [30]. Here ˜ v ∗ represents the optimal value function of the model ˜ m . We then produce a policy according to π ( s ) = argmax a (˜ r ( s, a ) + γ ∑ s ′ ˜ p ( s ′ | s, a )˜ v ∗ ( s ′ )) .
2. Approximate policy iteration with least squares temporal-difference learning (LSTD): For experiments on environments with finite S and V = ˜ V we used policy iteration combined with least square policy evaluation using basis { φ i } d i =1 . Specifically, each iteration of policy iteration involved the following steps:
3. (a) Collect experience tuples using the previous policy, π , leading to D = ( s i , a i , r i , s ′ i ) n i =1 .
4. (b) Replace the reward and next-states with those predicted by the model: ˜ r i , ˜ s ′ i = f θ ( s i , a i ) , leading to D ′ = ( s i , a i , ˜ r i , ˜ s ′ i ) n i =1 .
5. (c) Learn v w ( s ) = ∑ d i =1 w i φ i ( s ) ≈ v π using LSTD with D ′ .
6. (d) Construct a new policy π ( s ) = argmax a (˜ r s,a + γv w (˜ s ′ s,a )) where ˜ r s,a , ˜ s ′ s,a are sampled from the trained model conditioned on state s and action a .

This procedure is repeated for a fixed number of iterations.

3. Deep Q-networks (DQN): For experiments with V = ˜ V and infinite S we use Double QLearning to produce policies. We incorporate our learned model, f θ , by replacing elements in the replay buffer ( s, a, r, s ′ ) with ( s, a, ˜ r s,a , ˜ s ′ s,a ) where ˜ r s,a , ˜ s ′ s,a = f θ ( s, a ) .

(iv) Policy evaluation There are two methods to evaluate the policies resulting from the policy construction stage described above:

1. For policies produced using value iteration or policy iteration plus LSTD the ensuing policy, π , is exactly evaluated on the true environment, yielding v π ( s ) . Then the average value of v π ( s ) over all states is reported.
2. For policies produced using DQN, the average return over the last 100 episodes of training is reported.

## A.2.3 Classes of experiments

In addition to varying the capacity of M , there are two primary classes of experiments that were run in our paper that assess different choices of V . We distinguish between these two classes below:

span( V ) ≈ ¨ V , ˜ V = V , Π = GLYPH&lt;5&gt; : In these experiments we consider that there is no limitation on the agent's ability to represent value functions, and focus on achieving value equivalence with respect to the polytope of value functions ¨ V induced by the environment. We enable the agent to represent arbitrary functions in V by restricting ourselves to tabular environments and using dynamic programming to perform exact value iteration in our Policy Construction step. We approximate the value polytope by randomly sampling deterministic policies: { π 1 , . . . , π n } and evaluating them (again using dynamic programming) to produce { v π 1 , . . . , v π n } . We then choose V = { v π 1 , . . . , v π n } . In this setting we vary the number of policies generated.

Corresponding experiments: the experiments in this class vary two dimensions: (1) the rank of the model and (2) the number of policies generated. In Figures 3(a) and 3(b) we depict plots for the Four Rooms environment that fix the number of policies while varying the rank of the model and plots that fix the rank of the model while varying the number of policies, respectively. Figures 3(c) and 3(d) are analogous plots for the Catch environment.

span( V ) ≈ ˜ V , Π = GLYPH&lt;5&gt; : In these experiments we explore the setting described in Remark 2. We assume that the agent has variable ability to represent value functions, ˜ V , and attempt to learn a model in M ( ˜ V , GLYPH&lt;5&gt; ) . From Proposition 1 we only need to find V such that span( V ) ⊇ ˜ V . Experiments in this class can further be broken down into two settings based upon the nature of ˜ V :

- (a) Linear function approximation: In certain experiments our agent uses a class of linear function approximators to represent value functions: ˜ V = { ˜ v : ˜ v ( s ) = ∑ d i =1 φ i ( s ) w i } where φ i ( s ) : S ↦→ R and w ∈ R d . In this setting achieving span( V ) ⊇ ˜ V can be satisfied by choosing V = { φ i } d i =1 . For experiments using linear function approximation, we select our features { φ i } d i =1 to correspond to state aggregations. This entails the following procedure:
- (i) Collect data using a policy that selects actions uniformly at random.
- (ii) For tabular domains (e.g., Catch, Four Rooms), convert tabular state representations into coordinate-based representations. For Catch we convert each tabular state into the positions of both the paddle and the ball: ( x paddle , y paddle , x ball , y ball ) . For Four Rooms we use the position of the agent: ( x agent , y agent ) . Denote the function that performs this conversion as: f : S ↦→ R n where n = 2 and n = 4 for Four Rooms and Catch respectively.
- (iii) Perform k-means clustering on these converted states to produce d centers c 1: d .
- (iv) Define φ i ( s ) = ✶ { argmin j ‖ f ( s ) -c j ‖ 2 = i } , which corresponds to aggregating states according to their proximity to the previously calculated centers.

Corresponding experiments: the experiments in this class vary two dimensions: (1) the rank of the model and (2) the number of basis functions in { φ i } d i =1 . In Figures 4(a) and 4(b) we depict plots of 'slices' of this two-dimensional set of results on the Catch domain: 4(a) depicts fixing the number of basis functions while varying model-rank and 4(b) depicts fixing the model-rank while varying the number of basis functions.

- (b) Neural network function approximation: WhenNeural Networks are used to approximate the agent's value functions we have ˜ V = { ˜ v : ˜ v ( s ) = g θ ( s ) } where g θ represents a neural network with a particular architecture parameterized by θ . In our experiments we choose the architecture of g θ to be a 2 layer neural network with a tanh activation for its hidden layer. Unlike the linear function approximation setting, it is less obvious how to choose V such that span( V ) ⊇ ˜ V . One option is to use randomly initialized neural networks in ˜ V as our basis. To randomly initialize a given layer in some network g θ , we select weights from a truncated normal distribution where µ = 0 and σ = 1 / √ layer-input-size and initialize biases to 0 .

However, we found in practice that a large number of these randomly initialized networks were required to achieve reasonable performance. Instead of maintaining a large set of

initializations in V , we allow the elements of V themselves to be stochastic. Every time we apply an update of gradient descent we sample a new set of randomly initialized neural networks to function as V . This is equivalent to minimizing E V [ glyph[lscript] Π , V , D ′ ( m ∗ , ˜ m )] where glyph[lscript] Π , V ,D ′ is defined in 7. We find that having more random elements in V decreases the variance in the performance of VE models; |V| = 5 in our experiments.

Corresponding experiments: the experiments in this class vary two dimensions: (1) the rank of the model and (2) the width of the neural networks in ˜ V . In Figures 4(c) and 4(d) we depict plots of 'slices' of this two-dimensional set of results on the Catch domain: 4(c) depicts fixing the network width while varying model-rank and 4(d) depicts fixing the model-rank while the network width varies.

## A.2.4 Additional results

In the experimental section of the main text we showed that our theoretical claims about the value equivalence principle hold in practice through a series of bivariate experiments (e.g., varying modelrank and number of bases, varying model-rank and number of policies, varying model-rank and network width). We displayed our results as 'slices' of these bivariate experiments, where one variable would be held fixed and the other would be allowed to vary. To keep the paper concise, we only displayed a subset of these slices where the 'fixed' variable was selected as the median value over full set we experimented with. In what follows, we present the complete set of the experimental results we acquired. We indicate that a plot was included in the main text by printing its caption in bold font.

Figure 7: All Catch results with fixed V and span( V ) ≈ ˜ V .

<!-- image -->

Figure 8: All Catch results with fixed ˜ m and span( V ) ≈ ˜ V .

<!-- image -->

Figure 9: All Catch results with fixed V and V = { v π 1 , . . . , v π n } .

<!-- image -->

<!-- image -->

Figure 10: All Catch results with fixed ˜ m and V = { v π 1 , . . . , v π n } .

<!-- image -->

(d) Four Rooms (fixed

V

)

(e) Four Rooms (fixed

V

)

(f) Four Rooms (fixed

Figure 11: All Four Rooms results with fixed V and V = ˜ V .

V

)

<!-- image -->

(d) Four Rooms (fixed

˜

m

)

(e) Four Rooms (fixed

˜

m

)

(f) Four Rooms (fixed

˜

m

)

Figure 12: All Four Rooms results with fixed ˜ m and V = ˜ V .

<!-- image -->

(e) Four Rooms (fixed V ) (f) Four Rooms (fixed V ) (g) Four Rooms (fixed V ) (h) Four Rooms (fixed V ) Figure 13: All Four Rooms results with fixed V and V = { v π 1 , . . . , v π n } .

<!-- image -->

(d) Four Rooms (fixed

˜

m

)

(e) Four Rooms (fixed

˜

m

)

(f) Four Rooms (fixed

˜

m

)

Figure 14: All Four Rooms results with fixed ˜ m and V = { v π 1 , . . . , v π n } .

<!-- image -->

Figure 15: All Cart-pole results results with fixed V and span( V ) ≈ ˜ V .

## A.2.5 Hyperparameters

Table 1 provides a list detailing the different hyperparameters used throughout our pipeline.

Figure 16: All Cart-pole results results with fixed ˜ m and span( V ) ≈ ˜ V .

<!-- image -->

Table 1: List of hyperparameters used in the experiments.

| Hyperparameter           | Value     | Description                                                                               |
|--------------------------|-----------|-------------------------------------------------------------------------------------------|
| minibatch size           | 32        | Number of samples passed at a time during a training step of any learning method.         |
| model learning rate      | 5e-5      | Learning rate used to train all models.                                                   |
| # model samples          | 1,000,000 | Number of transitions sampled by a random policy in the Data Collection phase.            |
| model depth              | 2         | Number of hidden layers in the model architecture.                                        |
| model width              | 256       | Number of units per hidden layer.                                                         |
| model activation         | tanh      | Activation function following a hidden layer.                                             |
| model learning max steps | 1,000,000 | Maximum number of training iterations.                                                    |
| γ                        | 0.99      | Discount factor used across environments.                                                 |
| LSTD samples / policy    | 10,000    | Number of samples collected for each phase of policy evaluation using LSTD.               |
| # policy iteration steps | 40        | Number of steps of policy iteration in the policy con- struction phase, when applicable.  |
| DQN learning rate        | 5e-4      | Learning rate for DQN.                                                                    |
| DQN # environment steps  | 2,500,000 | Number of environment steps that DQN learns over.                                         |
| DQN learning frequency   | 4         | A learning update is applied after this many environ- ment steps.                         |
| DQN depth                | 1         | Number of hidden layers in the DQN.                                                       |
| DQN activation           | tanh      | Activation function following a hidden layer.                                             |
| DQN target update        | 2000      | Number of environment steps before the target network in the DQN is updated.              |
| Tabular # eval episodes  | 20        | Number of episodes to average performance over to assess a policy in the tabular setting. |
| DQN # eval episodes      | 100       | Number of episodes to average DQN policy perfor- mance over at the end of training.       |
| DQN glyph[epsilon1]      | 0.05      | Chance of picking a random action during training.                                        |
| Optimizer                | Adam      | Optimizer used for all learning operations. Default Adam parameters were used.            |