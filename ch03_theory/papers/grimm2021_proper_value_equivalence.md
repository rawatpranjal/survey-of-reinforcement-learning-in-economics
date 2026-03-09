## Proper Value Equivalence

## Christopher Grimm

Computer Science &amp; Engineering University of Michigan crgrimm@umich.edu

André Barreto, Gregory Farquhar, David Silver, Satinder Singh

DeepMind {andrebarreto,gregfar, davidsilver,baveja}@google.com

## Abstract

One of the main challenges in model-based reinforcement learning (RL) is to decide which aspects of the environment should be modeled. The value-equivalence (VE) principle proposes a simple answer to this question: a model should capture the aspects of the environment that are relevant for value-based planning. Technically, VEdistinguishes models based on a set of policies and a set of functions: a model is said to be VE to the environment if the Bellman operators it induces for the policies yield the correct result when applied to the functions. As the number of policies and functions increase, the set of VE models shrinks, eventually collapsing to a single point corresponding to a perfect model. A fundamental question underlying the VE principle is thus how to select the smallest sets of policies and functions that are sufficient for planning. In this paper we take an important step towards answering this question. We start by generalizing the concept of VE to orderk counterparts defined with respect to k applications of the Bellman operator. This leads to a family of VE classes that increase in size as k → ∞ . In the limit, all functions become value functions, and we have a special instantiation of VE which we call proper VE or simply PVE. Unlike VE, the PVE class may contain multiple models even in the limit when all value functions are used. Crucially, all these models are sufficient for planning, meaning that they will yield an optimal policy despite the fact that they may ignore many aspects of the environment. We construct a loss function for learning PVE models and argue that popular algorithms such as MuZero can be understood as minimizing an upper bound for this loss. We leverage this connection to propose a modification to MuZero and show that it can lead to improved performance in practice.

## 1 Introduction

It has long been argued that, in order for reinforcement learning (RL) agents to solve truly complex tasks, they must build a model of the environment that allows for counterfactual reasoning [29]. Since representing the world in all its complexity is a hopeless endeavor, especially under capacity constraints, the agent must be able to ignore aspects of the environment that are irrelevant for its purposes. This is the premise behind the value equivalence (VE) principle, which provides a formalism for focusing on the aspects of the environment that are crucial for value-based planning [17].

VE distinguishes models based on a set of policies and a set of real-valued scalar functions of state (henceforth, just functions). Roughly, a model is said to be VE to the environment if the Bellman operators it induces for the policies yield the same result as the environment's Bellman operators when applied to the functions. The policies and functions thus become a 'language' to specify which parts of the environment a model should capture. As the number of policies and functions increase the requirements on the model become more stringent, which is to say that the class of VE models shrinks. In the limit, the VE class collapses to a single point corresponding to a perfect model.

Although this result is reassuring, in practice we want to stop short of collapsing-after all, at this point the agent is no longer ignoring irrelevant aspects of the environment.

A fundamental question is thus how to select the smallest sets of policies and functions such that a resulting VE model is sufficient for planning. In this paper we take an important additional step in this direction: we show that the VE principle can be formulated with respect to value functions only. This result drastically reduces the space of functions that must be considered by VE, as in general only a small fraction of the set of all functions will qualify as value functions in a given environment. Since every policy has an associated value function, this new formulation of VE removes the need for selecting functions, only requiring policies. We name our new formulation proper value equivalence (PVE) to emphasize its explicit use of value functions.

PVE has several desirable properties. Unlike with VE, the class of PVE models does not collapse to a singleton in the limit. This means that, even if all value functions are used, we generally end up with multiple PVE models-which can be beneficial if some of these are easier to learn or represent than others. Crucially, all of these models are sufficient for planning, meaning that they will yield an optimal policy despite the fact that they may ignore many aspects of the environment .

Finally, we make more precise Grimm et al.'s [17] suggestion that the VE principle may help explain the good empirical performance of several modern algorithms [38, 33, 24, 12, 30]. Specifically, we show that, with mild assumptions, minimizing the loss of the MuZero algorithm [31] can be understood as minimizing a PVE error. We then leverage this connection to suggest a modification to MuZero and show a small but significant improvement in the Atari Learning Environment [3].

## 2 Background

The agent's interaction with the environment will be modeled as a Markov decision process (MDP) M≡〈S , A , r, p, γ 〉 , where S and A are the state and action spaces, r ( s, a ) is the expected reward following taking a from s , p ( s ′ | s, a ) is the transition kernel and γ ∈ [0 , 1) is a discount factor [27]. A policy is a mapping π : S ↦→ P ( A ) , where P ( A ) is the space of probability distributions over A ; we define GLYPH&lt;5&gt; ≡ { π | π : S ↦→ P ( A ) } as the set of all possible policies. A policy π is deterministic if π ( a | s ) &gt; 0 for only one action a per state s . A policy's value function is defined as

<!-- formula-not-decoded -->

where E π [ · ] denotes expectation over the trajectories induced by π and the random variables S t and A t indicate the state occupied and the action selected by the agent at time step t .

The agent's goal is to find a policy π ∈ GLYPH&lt;5&gt; that maximizes the value of every state [36, 37]. Usually, a crucial step to carry out this search is to compute the value function of candidate policies. This process can be cast in terms of the policy's Bellman operator :

<!-- formula-not-decoded -->

where v is any function in the space V ≡ { f | f : S ↦→ R } . It is known that lim n →∞ T n π v = v π , that is, starting from any v ∈ V , the repeated application of T π will eventually converge to v π . Since in RL the agent does not know p and r , it cannot apply (2) directly. One solution is to learn a model ˜ m ≡ (˜ r, ˜ p ) and use it to compute (2) with p and r replaced by ˜ p and ˜ r [36]. We denote the set of all models as M .

The value equivalence principle defines a model as value equivalent (VE) to the environment m ∗ ≡ ( r, p ) with respect to a set of policies Π and a set of functions V if it produces the same Bellman updates as m ∗ when using Π and V [17]. Classes of such models are expressed as follows:

<!-- formula-not-decoded -->

where M⊆ M is a class of models, ˜ T π denotes one application of the Bellman operator induced by model ˜ m and policy π to function v , and T π is environment's Bellman operator for π .

Grimm et al. [17] showed that the VE principle can be used to learn models that disregard aspects of the environment which are not related to the task of interest. 1 Classical approaches to model learning

1 A related approach is taken in value-aware model learning [11] which minimizes the discrepancy between the Bellman optimality operators induced by the model and the environment.

do not take the eventual use of the model into account, potentially modeling irrelevant aspects of the environment. Accordingly, Grimm et al. [17] have shown that, under the same capacity constraints, models learned using VE can outperform their classical counterparts.

## 3 Proper value equivalence

One can define a spectrum of VE classes corresponding to different numbers of applications of the Bellman operator. We define an orderk VE class as:

M

(Π

,

V

)

≡ {

˜

m

∈ M

:

˜

T

k

π

v

=

T

k

π

v

∀

π

∈

Π

, v

∈ V}

(4)

where ˜ T k π v denotes k applications of ˜ T π to v . Under our generalized definition of VE, Grimm et al. [17] studied order-one VE classes of the form M 1 (Π , V ) . They have shown that M 1 ( GLYPH&lt;5&gt; , V ) either contains only the environment or is empty. This is not generally true for k &gt; 1 . The limiting behavior of orderk value equivalent classes can be described as follows

Proposition 1. Let V be a set of functions such that if v ∈ V then T π v ∈ V for all π ∈ Π . Then, for k, K ∈ Z + such that k divides K , it follows that:

- (i) For any M⊆ M and any Π ⊆ GLYPH&lt;5&gt; , we have that M k (Π , V ) ⊆ M K (Π , V ) .
- (ii) If Π is non-empty and V contains at least one constant function, then there exist environments such that M k (Π , V ) ⊂ M K (Π , V ) .

We defer all proofs of theoretical results to Appendix A.2. Based on Proposition 1 we can relate different VE model classes according to the greatest common divisor of their respective orders; specifically, two classes M k (Π , V ) and M K (Π , V ) will intersect at M gcd( k,K ) (Π , V ) (Figure 1). Proposition 1 also implies that, in contrast to order-one VE classes, higher order VE classes potentially include multiple models, even if VE is defined with respect to all policies GLYPH&lt;5&gt; and all functions V . In addition, the size of a VE class cannot decrease as we increase its order from k to a multiple of k (and in some cases it will strictly increase). This invites the question of what happens in the limit as we keep increasing the VE order. To answer this question, we introduce a crucial concept for this paper:

Definition 1. (Proper value equivalence). Given a set of policies Π ⊆ GLYPH&lt;5&gt; , let

<!-- formula-not-decoded -->

where ˜ v π and v π are the value functions of π induced by model ˜ m and the environment. We say that each ˜ m ∈ M ∞ (Π) is a proper value equivalent model to the environment with respect to Π .

Because the process of repeatedly applying a policy's Bellman operator to a function converges to the same fixed point regardless of the function, in an order-∞ VE class the set Π uniquely determines the set V . This reduces the problem of defining Π and V to defining the former only. Also, since all functions in an order-∞ VE are value functions, we call it proper VE or PVE.

It is easy to show that Proposition 1 is valid for any k ∈ Z + when K = ∞ (Corollary 2 in Appendix A.2). Thus, in some sense, M ∞ is the 'biggest' VE class. It is also possible to define this special VE class in terms of any other:

Proposition 2. For any Π ⊆ GLYPH&lt;5&gt; and any k ∈ Z + it follows that

<!-- formula-not-decoded -->

where v π is the value of policy π in the environment.

We thus have two equivalent ways to describe the class of models which are PVE with respect to a set of policies Π .

Figure 1: Topology of the space of orderk VE classes. Given a set of policies Π , a set of functions V closed under Bellman updates, and k, K ∈ Z + such that k divides K , we have that M k (Π , V ) ⊆ M K (Π , V ) .

<!-- image -->

k

The first, given in (5), is the order-∞ limit of value equivalence with respect to Π and the set of all functions V . The second, given in (6), is the intersection of the classes of models that are orderk VE with respect to the singleton policies in Π and their respective value functions. This latter form is valid for any k , and will underpin our practical algorithmic instantiations of PVE.

Setting k = 1 in Proposition 2 we see that PVE can be written in terms of order-one VE. This means that M ∞ inherits many of the topological properties of M 1 shown by Grimm et al. [17]. Specifically, we know that M ′∞ (Π) ⊆ M ∞ (Π) if M ′ ⊆ M and also that M ∞ (Π ′ ) ⊆ M ∞ (Π) when Π ⊆ Π ′ (these directly follow from Grimm et al.'s [17] Properties 1 and 3 respectively).

Proposition 2 also sheds further light into the relation between PVE and orderk VE more generally. Let Π be a set of policies and V π their value functions. Then, for any k ∈ Z + , we have that

<!-- formula-not-decoded -->

which is another way to say that M ∞ is, in some sense, the largest among all the VE classes. The reason why the size of VE classes is important is that it directly reflects the main motivation behind the VE principle. VE's premise is that models should be constructed taking into account their eventual use: if some aspects of the environment are irrelevant for value-based planning, it should not matter whether a model captures them or not. This means that all models that only differ with respect to these irrelevant aspects but are otherwise correct qualify as valid VE solutions. A larger VE class generally means that more irrelevant aspects of the environment are being ignored by the agent. We now make this intuition more concrete by showing how irrelevant aspects of the environment that are eventually captured by order-one VE are always ignored by PVE:

Proposition 3. Let Π ⊆ GLYPH&lt;5&gt; . If the environment state can be factored as S = X × Y where |Y| &gt; 1 and v π ( s ) = v π (( x, y )) = v π ( x ) for all π ∈ Π , then M 1 (Π , V ) ⊂ M ∞ (Π) .

Note that the subset relation appearing in Proposition 3 is strict. We can think of the variable ' y ' appearing in Proposition 3 as superfluous features that do not influence the RL task, like the background of an image or any other sensory data that is irrelevant to the agent's goal. A model is free to assign arbitrary dynamics to such irrelevant aspects of the state without affecting planning performance. Since order-one VE eventually pins down a model that describes everything about the environment, one would expect the size of M ∞ relative to M 1 to increase as more superfluous features are added. Indeed, in our proof of Proposition 3 we construct a set of models in M ∞ ( GLYPH&lt;5&gt; ) which are in one-to-one correspondence with Y , confirming this intuition (see Appendix A.2).

## Proper value equivalence yields models that are sufficient for optimal planning

In general PVE does not collapse to a single model even in the limit of Π = GLYPH&lt;5&gt; . At first this may cause the impression that one is left with the extra burden of selecting one among the PVE models. However, it can be shown that no such choice needs to be made:

Proposition 4. An optimal policy for any ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; ) is also an optimal policy in the environment.

According to Proposition 1 any model ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; ) used for planning will yield an optimal policy for the environment. In fact, in the spirit of ignoring as many aspects of the environment as possible, we can define an even larger PVE class by focusing on deterministic policies only:

Corollary 1. Let GLYPH&lt;5&gt; det be the set of all deterministic policies. An optimal policy for any ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; det ) is also optimal in the environment.

Given that both M ∞ ( GLYPH&lt;5&gt; ) and M ∞ ( GLYPH&lt;5&gt; det ) are sufficient for optimal planning, one may wonder if these classes are in fact the same. The following result states that the class of PVE models with respect to deterministic policies can be strictly larger than its counterpart defined with respect to all policies:

Proposition 5. There exist environments and model classes for which M ∞ ( GLYPH&lt;5&gt; ) ⊂ M ∞ ( GLYPH&lt;5&gt; det ) .

Figure 2 illustrates Proposition 5 with an example of environment and a model ˜ m such that ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; det ) but ˜ m / ∈ M ∞ ( GLYPH&lt;5&gt; ) .

To conclude our discussion on models that are sufficient for optimal planning, we argue that, in the absence of additional information about the environment or the agent, M ∞ ( GLYPH&lt;5&gt; det ) is in fact the largest possible VE class that is guaranteed to yield optimal performance. To see why this is so, suppose we

Figure 2: An environment / model pair with the same values for all deterministic policies but not all stochastic policies. The environment has three states and two actions: A = { L , R } . The percentages in the figure indicate the probability of a given transition and the corresponding tuples ( r, a ) indicate the reward associated with a given action. A deterministic policy cannot dither between s 1 and s 3 but a stochastic policy can. Note that the dynamics between the pair differs when taking action R from s 2 . This difference will affect the dithering behavior of such a stochastic policy in a way that results in different model and environment values.

<!-- image -->

remove a single deterministic policy from GLYPH&lt;5&gt; det and pick an arbitrary model ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; det -{ π } ) . Let ˜ v π be the value function of π according to the model ˜ m . Because π is not included in the set of policies used to enforce PVE, ˜ v π may not coincide with v π , the actual value function of π according to the environment. Now, if π happens to be the only optimal policy in the environment and ˜ v π is not the optimal value function of ˜ m , the policy returned by this model will clearly be sub-optimal.

## 4 Learning a proper value-equivalent model

Having established that we want to find a model in M ∞ ( GLYPH&lt;5&gt; det ) , we now turn our attention to how this can be done in practice. Following Grimm et al. [17], given a finite set of policies Π and a finite set of functions V , we cast the search for a model ˜ m ∈ M k (Π , V ) as the minimization of deviations from (4):

<!-- formula-not-decoded -->

where ˜ T π are Bellman operators induced by ˜ m and ‖·‖ is a norm. 2 Note that setting k = ∞ in (8) yields a loss that requires computing ˜ m 's value function-which is impractical to do if ˜ m is being repeatedly updated. Thankfully, by leveraging the connection between orderk VE and PVE given in Proposition 2, we can derive a practical PVE loss:

<!-- formula-not-decoded -->

Interestingly, given a set of policies Π , minimizing (9) for any k will result in a model ˜ m ∈ M ∞ (Π) ( cf. Proposition 2). As we will discuss shortly, this property can be exploited to generate multiple loss functions that provide a richer learning signal in practical scenarios.

Contrasting loss functions (8) and (9) we observe an important fact: unlike with other orderk VE classes, PVE requires actual value functions to be enforced in practice. Since value functions require data and compute to be obtained, it is reasonable to ask whether the benefits of PVE justify the associated additional burden. Concretely, one may ask whether the sample transitions and computational effort spent in computing the value functions to be used with PVE would not be better invested in enforcing other forms of VE over arbitrary functions that can be readily obtained.

We argue that in many cases one does not have to choose between orderk VE and PVE. Valuebased RL algorithms usually compute value functions iteratively, generating a sequence of functions v 1 , v 2 , ... which will eventually converge to ˜ v π for some π . A model-based algorithm that computes ˜ v π in this way has to somehow interleave this process with the refinement of the model ˜ m . When it comes to VE, one extreme solution is to only use the final approximation ˜ v π ≈ v π in an attempt to enforce PVE through (9). It turns out that, as long as the sequence v 1 , v 2 , ... is approaching v π , one can use all the functions in the sequence to enforce PVE with respect to π . Our argument is based on the following result:

2 We can also impose VE with infinite sets of functions and policies by replacing the respective sums with integrals; in this case one may consider taking a supremum over VE terms to avoid situations where VE is not necessarily satisfied on measure 0 sets.

Proposition 6. For any π ∈ GLYPH&lt;5&gt; , v ∈ V and k, n ∈ Z + , we have that

<!-- formula-not-decoded -->

Note that the left-hand side of (10) corresponds to one of the terms of the PVE loss (9) associated with a given π . This means that, instead of minimizing this quantity directly, one can minimize the upper-bound on the right-hand side of (10). The first term in this upper bound, glyph[epsilon1] v , is the conventional value-function approximation error that most value-based methods aim to minimize (either directly or indirectly). The second term, glyph[epsilon1] ve , is similar to the terms appearing in the orderk VE loss (8), except that here the number of applications of T π and of its approximation ˜ T π do not have to coincide.

All the quantities appearing in glyph[epsilon1] ve are readily available or can be easily approximated using sample transitions [17]. Thus, glyph[epsilon1] ve can be used to refine the model ˜ m using functions v that are not necessarily value functions. As v → v π , two things happen. First, glyph[epsilon1] ve approaches one of the terms of the PVE loss (9) associated with policy π . Second, glyph[epsilon1] v vanishes. Interestingly, the importance of glyph[epsilon1] v also decreases with n and k , the number of times T π and ˜ T π are applied in glyph[epsilon1] ve , respectively. This makes sense: since T n π v → v π as n →∞ and, by definition, VE approaches PVE as k →∞ , we have that glyph[epsilon1] ve approaches the left-hand side of (10) as both n and k grow.

## An extended example: MuZero through the lens of value equivalence

Grimm et al. [17] suggested that the VE principle might help to explain the empirical success of recent RL algorithms like Value Iteration Networks, the Predictron, Value Prediction Networks, TreeQN, and MuZero [38, 33, 24, 12, 31]. In this section we investigate this hypothesis further and describe a possible way to interpret one of these algorithms, MuZero, through the lens of VE. We acknowledge that the derivation that follows abstracts away many details of MuZero and involves a few approximations of its mechanics, but we believe it captures and explains the algorithm's essence.

MuZero is a model-based RL algorithm that achieved state-of-the-art performance across both board games, such as Chess and Go, and Atari 2600 games [31]. The model ˜ m in MuZero is trained on sequences of states, actions and rewards resulting from executing a 'behavior policy' in the environment: s t : t + n + K , a t : t + n + K , r t : t + n + K where n and K are hyperparameters of the agent which will be explained shortly. The agent produces an 'agent state' z 0 t from s t and subsequently generates z 1: K t by using its model to predict the next K agent states following actions a t : t + K . The agent also maintains reward and value function estimates as a function of agent states, which we denote ˜ r ( z ) and v ( z ) respectively. A variant 3 of MuZero's per-state model loss can thus be expressed as:

<!-- formula-not-decoded -->

where V t + k = r t + k + · · · + γ n -1 r t + k + n -1 + γ n v targ ( z 0 t + k + n ) . The term v targ is a value target produced by Monte-Carlo tree search (MCTS, [7]). Because the behavior policy is itself computed via MCTS, we have that v targ ≈ v ; for simplicity we will assume that v targ = v and only use v .

In what follows we show, subject to a modest smoothness assumption, that minimizing MuZero's loss with respect to its behavior policy, π , also minimizes a corresponding PVE loss. Put precisely:

<!-- formula-not-decoded -->

for some C &gt; 0 , where d π is a stationary distribution. We proceed by combining two derivations: a lower-bound on E d π [ glyph[lscript] µ ( S t )] in (15), and an upper-bound on ( glyph[lscript] K { π } , ∞ ( m ∗ , ˜ m )) 2 in (17).

As a preliminary step we note that glyph[lscript] µ ( s t ) and glyph[lscript] K { π } , ∞ ( m ∗ , ˜ m ) are expressed in terms of samples and expectations respectively. We note the following connection between these quantities:

<!-- formula-not-decoded -->

3 In reality MuZero uses a categorical representation for its value and reward functions and minimizes them using a cross-entropy objective. We argue that this choice is not essential to its underlying ideas and use scalar representations with a squared loss to simplify our analysis.

where P k π is the k -step environment transition operator under policy π : P k π [ x ]( s t ) = E [ x ( S t + k ) | s t , m ∗ , π ] , r π ( s ) = E A ∼ π [ r ( s, A )] and ˜ P k π and ˜ r π are the corresponding quantities using the model instead of the environment. The above expectations are taken with respect to the environment or model and π as appropriate. We now derive our lower-bounds on E d π [ glyph[lscript] µ ( S t )] :

<!-- formula-not-decoded -->

where we apply the tower-property, Jensen's inequality and the identities in (13). We write the expression using norms and drop all terms except k ∈ { 0 , K } in the first sum to obtain:

<!-- formula-not-decoded -->

recalling that ‖ x -y ‖ 2 d π = E d π [( x ( S t ) -y ( S t )) 2 ] . To derive an upper-bound for ( glyph[lscript] K { π } , ∞ ( m ∗ , ˜ m )) 2 we assume that the error in value estimation is smooth in the sense that there is some g &gt; 0 (independent of v ) such that ‖ v -v π ‖ ∞ &lt; g · ‖ v -v π ‖ d π . We can then use a modified version of (10) for the d π -weighted glyph[lscript] 2 -norm (see Appendix A.2), plugging in n + K and K :

<!-- formula-not-decoded -->

from here we can square both sides and apply Jensen's inequality,

<!-- formula-not-decoded -->

where a = γ K ( g + γ n )(1 -γ n ) -1 and b = a + K +2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus minimizing MuZero's loss minimizes a squared PVE loss with respect to a single policy.

## 5 Experiments

We first provide results from tabular experiments on a stochastic version of the Four Rooms domain which serve to corroborate our theoretical claims. Then, we present results from experiments across the full Atari 57 benchmark [3] showcasing that the insights from studying PVE and its relationship to MuZero can provide a benefit in practice at scale. See Appendix A.3 for a full account of our experimental procedure.

In Section 3 we described the topological relationships between orderk and PVE classes. This is summarized by Proposition 1, which shows that, for appropriately defined V and Π , M k ⊆ M K if K is a multiple of k . We illustrate this property empirically by randomly initializing a set of models and then using (8) (or (9) for the limiting case of k = ∞ ) to iteratively update them towards M k ( GLYPH&lt;5&gt; , V ) , with k ∈ { 1 , 30 , 40 , 50 , 60 , ∞} . We take the vectors representing these models and project them onto their first two principal components in order to visualise their paths through learning. The results are

70

60-

50

40

30-

20 -

20

40

Order 1

•0.81

0.6

•0.41

0,2

0.8

0.4

60

0.2

Model Rank

80

Order 301

0.6

0.4

0.2

0.6

-0.21

<!-- image -->

Normalized Training Time

1.0

-2.5

-2.0

-1.5

Order ∞o

Normalized Trajectory Time

Normalized Trajectory Time

Figure 3: All scatter plots are generated by tracking the training progress over 500,000 iterations of models with different orderk VE objectives. In each plot 120 models were tracked; at every 1000 timesteps the full set of models is converted into vector form and projected onto their first two principal components before being plotted (details in the appendix). Top row: points are colored according to their progress through training. Bottom row: points are colored according to the average value of the associated model's optimal policy on the environment. Rightmost plot: line-plot of the diameters of these scatter plots against the model-class order.

shown on the top row of Figure 3. In accordance with the theory, we see that the space of converged models, represented with the brightest yellow regions, grows with k . This trend is summarised in the rightmost plot, which shows the diameter of the scatter plots for each k . In the bottom row of Figure 3 we use color to show the value that the optimal policy of each model achieves in the true environment. As predicted by our theory, the space of models that are sufficient for optimal planning also grows with k .

Model classes containing many models with optimal planning performance are particularly advantageous when the set of models that an agent can represent is restricted, since the larger the set of suitable models the greater the chance of an overlap between this set and the set of models representable by the agent. Proposition 4 and Corollary 1 compared M ∞ ( GLYPH&lt;5&gt; ) and M ∞ ( GLYPH&lt;5&gt; det ) , showing that, although any model in either class is sufficient for planning, M ∞ ( GLYPH&lt;5&gt; ) ⊆ M ∞ ( GLYPH&lt;5&gt; det ) . This suggests that it might be better to learn a model in M ∞ ( GLYPH&lt;5&gt; det ) when the agent has limited capacity.

Figure 4: (a) Comparison of the performance of optimal policies obtained from capacity constrained models trained to be in M ∞ ( GLYPH&lt;5&gt; det ) and M ∞ ( GLYPH&lt;5&gt; ) . For each action a ∈ A , the transition dynamics ˜ P a is constrained to have a rank of at most k . The red dashed line represents the performance of the optimal environment policy. (b) Trajectories starting from the bottom-right state (red dot) sampled from the optimal environment policy in both the environment and a PVE model. Note the numerous diagonal transitions in the PVE model which are not permitted in the environment.

<!-- image -->

Order 60

Environment

PVE Model

Median human normalised score

5

3

— MuZero + Past Policies

MuZero Baseline

We illustrate that this is indeed the case in Figure 4b. We progressively restrict the space of models that the agent can represent and attempt to learn models in either M ∞ ( GLYPH&lt;5&gt; ) or M ∞ ( GLYPH&lt;5&gt; det ) . Indeed, we find that the larger class, M ∞ ( GLYPH&lt;5&gt; det ) , yields superior planning performance as agent capacity decreases.

Given their importance, we provide intuition on the ways that individual PVE models differ from the environment. In Figure 4a we compare trajectories starting at the same initial state (denoted by a red-circle) from the optimal environment policy in both the environment and in a randomly sampled model from M ∞ ( GLYPH&lt;5&gt; ) . In the PVE model there are numerous diagonal transitions not permitted by the environment. Note that while the PVE model has very different dynamics than the environment, these differences must 'balance out', as it still has the same values under any policy as the environment.

In Section 4 we showed that minimizing Muzero's loss function is analogous to minimizing an upper-bound on a PVE loss (9) with respect to the agent's current policy π -which corresponds to finding a model in M ∞ (Π) where Π = { π } . Note that our guarantee on the performance of PVE models (Corollary 1) holds when Π contains all deterministic policies. While it is not feasible to enforce Π = GLYPH&lt;5&gt; det , we can use previously seen policies by augmenting the MuZero algorithm with a buffer of past policies and their approximate value functions (we do so by periodically storing the corresponding parameters). We can then add an additional loss to MuZero with the form of the original value loss but using the past value functions. We still use sampled rewards to construct value targets, but use the stored policies to compute off-policy corrections using V-trace [9].

Figure 5: Comparison of our proposed modification to MuZero with an unmodified baseline.

<!-- image -->

To test this proposal we use an on policy (i.e., without a replay buffer) implementation of MuZero run for 500M frames (as opposed to 20B frames in the online result of [31]) on the Atari 57 benchmark and find that using our additional loss yields an advantage in the human normalised median performance shown in Figure 5. MuZero's data efficiency can also be improved with the aid of a replay buffer of trajectories from past policies [32], which may also capture some of the advantages of expanding the set of policies used for PVE.

## 6 Related work

Our work is closely related to the value-aware model learning (VAML, IterVAML, [11, 10]) which learns models to minimize the discrepancy between their own Bellman optimality operators and the environment's on the optimal value function-an a priori unknown quantity. To handle this VAML specifies a family of potential value functions and minimizes the worst-case discrepancy across them, whereas IterVAML minimizes the discrepancy with respect to model's current estimate of the value function in a value iteration inspired scheme. PVE and the V AML family are complementary works, with VAML addressing its induced optimization problems and PVE addressing its induced model classes. Both, however, advocate for learning models with their eventual use in mind-a view that is aligned with many criticisms of the maximum likelihood objective for model learning [11, 21, 2, 23]).

It is worth mentioning the relationship between PVE and TD models [35] which, for a given policy, defines any R ∈ R |S| and P ∈ R |S|×|S| with lim k →∞ P k = 0 as a valid model if V = R + P glyph[latticetop] V where V ∈ R |S| represents v π . Clearly all models in M ∞ ( { π } ) are valid models, however, since P is not necessarily a transition matrix, the converse does not hold. While TD models are restricted to prediction rather than control, their generality warrants further inquiry.

Orderk and PVE model classes form equivalences between MDPs and thus can be situated among other equivalence notions which can be formulated as state-aggregations [8, 26, 25, 16, 28, 13, 34, 39, 5, 40]. As pointed out by Grimm et al. [17], the interaction between arbitrary state-aggregation

1

2

and models can be captured with special cases of order-one VE. Our extension of higher-order VEs potentially offers the possibility of 'blending' existing notions of state aggregation with PVE.

A notable instance of state-aggregation is bisimulation [22], which uses a relation to aggregate states that have the same immediate rewards and transition dynamics into other aggregated states. Bisimulation metrics [13] provide smooth measures of how closely pairs of states satisfy bisimulation relations. These concepts have become increasingly popular in deep reinforcement learning where they are used to guide the learning of effective representations [43, 42, 15, 1]. Although both bisimulation and PVE provide direction for learning internal aspects of an agent, they are fundamentally different in their purview-bisimulation concerns representation learning, while PVE concerns the learning of models given a representation of state.

Beyond bisimulation, representation learning has a wide literature [41, 20, 6, 14, 4] including several modern works which explicitly study the conjunction of model learning with state representation [44, 42, 15]. These are further complemented by efforts to learn state representations and models jointly in the service of value-based planning [12, 33, 24, 18, 31, 38].

## 7 Conclusion and future work

We extended the value equivalence principle by defining a spectrum of orderk VE sets in which models induce the same k -step Bellman operators as the environment. We then explored the topology of the resulting equivalence classes and defined the limiting class when k →∞ as PVE. If a model is PVE to the environment with respect to a set of policies Π , then the value functions of all policies in Π are the same in the environment and the model. The fact that PVE classes can be defined using only a set of policies eliminates the need for specifying a set of functions to induce VE-resolving a fundamental issue left open by Grimm et al. [17]. Importantly, we showed that being PVE with respect to all deterministic policies is sufficient for a model to plan optimally in the environment. In the absence of additional information, this is the largest possible VE class that yields optimal planning. On the practical side, we showed how the MuZero algorithm can be understood as minimizing an upper bound on a PVE loss, and leveraged this insight to improve the algorithm's performance.

Though our efforts have advanced the understanding of value equivalence and proven useful algorithmically, there is still work to be done in developing a VE theory whose assumptions hold in practice. This remaining work can be broadly grouped into two areas (1) understanding the role of approximation in VE and (2) establishing performance guarantees for VE models with arbitrary sets of policies and functions. We leave these as future work.

## Acknowledgements

We thank Angelos Filos and Sonya Kotov for many thought-provoking discussions. Christopher Grimm's work was made possible by the support of the Lifelong Learning Machines (L2M) grant from the Defense Advanced Research Projects Agency. Any opinions, findings, conclusions, or recommendations expressed here are those of the authors and do not necessarily reflect the views of the sponsors.

## References

- [1] Rishabh Agarwal, Marlos C Machado, Pablo Samuel Castro, and Marc G Bellemare. Contrastive behavioral similarity embeddings for generalization in reinforcement learning. arXiv preprint arXiv:2101.05265 , 2021.
- [2] Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pages 463-474. PMLR, 2020.
- [3] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47:253-279, 2013.
- [4] Ondrej Biza, Robert Platt, Jan-Willem van de Meent, and Lawson LS Wong. Learning discrete state abstractions with deep variational inference. arXiv preprint arXiv:2003.04300 , 2020.
- [5] Pablo Samuel Castro. Scalable methods for computing state similarity in deterministic markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 10069-10076, 2020.
- [6] Dane Corneil, Wulfram Gerstner, and Johanni Brea. Efficient model-based deep reinforcement learning with variational state tabulation. In International Conference on Machine Learning , pages 1049-1058. PMLR, 2018.
- [7] Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In International conference on computers and games , pages 72-83. Springer, 2006.
- [8] Thomas Dean and Robert Givan. Model minimization in markov decision processes. In AAAI/IAAI , pages 106-111, 1997.
- [9] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International Conference on Machine Learning , pages 1407-1416. PMLR, 2018.
- [10] Amir-massoud Farahmand. Iterative value-aware model learning. In Advances in Neural Information Processing Systems (NeurIPS) , pages 9090-9101, 2018.
- [11] Amir-Massoud Farahmand, André Barreto, and Daniel Nikovski. Value-Aware Loss Function for Model-based Reinforcement Learning. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 54, pages 1486-1494, 2017.
- [12] G Farquhar, T Rocktäschel, M Igl, and S Whiteson. Treeqn and atreec: Differentiable treestructured models for deep reinforcement learning. In 6th International Conference on Learning Representations, ICLR 2018-Conference Track Proceedings , volume 6. ICLR, 2018.
- [13] Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite markov decision processes. In UAI , volume 4, pages 162-169, 2004.
- [14] Vincent François-Lavet, Yoshua Bengio, Doina Precup, and Joelle Pineau. Combined reinforcement learning via abstract representations. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 3582-3589, 2019.

- [15] Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G Bellemare. Deepmdp: Learning continuous latent space models for representation learning. In International Conference on Machine Learning , pages 2170-2179. PMLR, 2019.
- [16] Robert Givan, Thomas Dean, and Matthew Greig. Equivalence notions and model minimization in markov decision processes. Artificial Intelligence , 147(1-2):163-223, 2003.
- [17] Christopher Grimm, Andre Barreto, Satinder Singh, and David Silver. The value equivalence principle for model-based reinforcement learning. Advances in Neural Information Processing Systems , 33, 2020.
- [18] Matteo Hessel, Ivo Danihelka, Fabio Viola, Arthur Guez, Simon Schmitt, Laurent Sifre, Theophane Weber, David Silver, and Hado van Hasselt. Muesli: Combining improvements in policy optimization. arXiv preprint arXiv:2104.06159 , 2021.
- [19] Matteo Hessel, Manuel Kroiss, Aidan Clark, Iurii Kemaev, John Quan, Thomas Keck, Fabio Viola, and Hado van Hasselt. Podracer architectures for scalable Reinforcement Learning. CoRR , abs/2104.06272, 2021.
- [20] Maximilian Igl, Luisa Zintgraf, Tuan Anh Le, Frank Wood, and Shimon Whiteson. Deep variational reinforcement learning for pomdps. In International Conference on Machine Learning , pages 2117-2126. PMLR, 2018.
- [21] Joshua Joseph, Alborz Geramifard, John W Roberts, Jonathan P How, and Nicholas Roy. Reinforcement learning with misspecified model classes. In 2013 IEEE International Conference on Robotics and Automation , pages 939-946. IEEE, 2013.
- [22] Robin Milner. Communication and concurrency , volume 84.
- [23] Aditya Modi, Nan Jiang, Ambuj Tewari, and Satinder Singh. Sample complexity of reinforcement learning using linearly combined model ensembles. In International Conference on Artificial Intelligence and Statistics , pages 2010-2020. PMLR, 2020.
- [24] Junhyuk Oh, Satinder Singh, and Honglak Lee. Value prediction network. In Advances in Neural Information Processing Systems , pages 6118-6128, 2017.
- [25] Pascal Poupart and Craig Boutilier. Value-directed belief state approximation for POMDPs. CoRR , abs/1301.3887, 2013. URL http://arxiv.org/abs/1301.3887 .
- [26] Pascal Poupart, Craig Boutilier, et al. Value-directed compression of pomdps. Advances in neural information processing systems , pages 1579-1586, 2003.
- [27] Martin L. Puterman. Markov Decision Processes-Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, Inc., 1994.
- [28] Balaraman Ravindran and Andrew G Barto. Approximate homomorphisms: A framework for non-exact minimization in markov decision processes. 2004.
- [29] Stuart J. Russell and Peter Norvig. Artificial Intelligence: A Modern Approach . Pearson Education, 3 edition, 2003.
- [30] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. arXiv preprint arXiv:1911.08265 , 2019.
- [31] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature , 588(7839):604-609, 2020.
- [32] Julian Schrittwieser, Thomas Hubert, Amol Mandhane, Mohammadamin Barekatain, Ioannis Antonoglou, and David Silver. Online and offline reinforcement learning by planning with a learned model. arXiv preprint arXiv:2104.06294 , 2021.

- [33] David Silver, Hado van Hasselt, Matteo Hessel, Tom Schaul, Arthur Guez, Tim Harley, Gabriel Dulac-Arnold, David Reichert, Neil Rabinowitz, Andre Barreto, et al. The predictron: End-toend learning and planning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 , pages 3191-3199. JMLR. org, 2017.
- [34] John P Spencer, Michael SC Thomas, and JL McClelland. Toward a unified theory of development. JP Spencer, MSC Thomas, &amp; JL McClelland (Eds.) , pages 86-118, 2009.
- [35] Richard S. Sutton. TD models: Modeling the world at a mixture of time scales. In Proceedings of the Twelfth International Conference on Machine Learning , pages 531-539, 1995.
- [36] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . MIT Press, 2018. URL https://mitpress.mit.edu/books/ reinforcement-learning-second-edition . 2nd edition.
- [37] Csaba Szepesvári. Algorithms for Reinforcement Learning . Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan &amp; Claypool Publishers, 2010.
- [38] Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value iteration networks. In Advances in Neural Information Processing Systems , pages 2154-2162, 2016.
- [39] Jonathan Taylor, Doina Precup, and Prakash Panagaden. Bounding performance loss in approximate mdp homomorphisms. Advances in Neural Information Processing Systems , 21: 1649-1656, 2008.
- [40] Elise van der Pol, Thomas Kipf, Frans A Oliehoek, and Max Welling. Plannable approximations to mdp homomorphisms: Equivariance under actions. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems , pages 1431-1439, 2020.
- [41] Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: a locally linear latent dynamics model for control from raw images. In Proceedings of the 28th International Conference on Neural Information Processing Systems-Volume 2 , pages 2746-2754, 2015.
- [42] Amy Zhang, Zachary C Lipton, Luis Pineda, Kamyar Azizzadenesheli, Anima Anandkumar, Laurent Itti, Joelle Pineau, and Tommaso Furlanello. Learning causal state representations of partially observable environments. arXiv preprint arXiv:1906.10437 , 2019.
- [43] Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. arXiv preprint arXiv:2006.10742 , 2020.
- [44] Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew Johnson, and Sergey Levine. Solar: Deep structured representations for model-based reinforcement learning. In International Conference on Machine Learning , pages 7444-7453. PMLR, 2019.

## Checklist

1. For all authors...
2. (a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]
3. (b) Did you describe the limitations of your work? [Yes]
4. (c) Did you discuss any potential negative societal impacts of your work? [N/A]
5. (d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]
2. If you are including theoretical results...
7. (a) Did you state the full set of assumptions of all theoretical results? [Yes]
8. (b) Did you include complete proofs of all theoretical results? [Yes] See appendix.
3. If you ran experiments...
10. (a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] Code for the illustrative experiments is available at a URL provided in Appendix A.3. Sufficient detail for reproducability is provided in the same section.
11. (b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes]
12. (c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes]
13. (d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
15. (a) If your work uses existing assets, did you cite the creators? [Yes]
16. (b) Did you mention the license of the assets? [N/A]
17. (c) Did you include any new assets either in the supplemental material or as a URL? [Yes]
18. (d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]
19. (e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]
5. If you used crowdsourcing or conducted research with human subjects...
21. (a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]
22. (b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]
23. (c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

## A Appendix

## A.1 Illustrative MDPs

Several proofs in A.2 rely on constructing special MDPs to serve as examples or counterexamples. We reserve this section to describe these MDPs for later reference.

## A.1.1 Ring and false-ring MDPs

We consider a simple n -state, 1 action 'ring' MDP (Figure 6) denoted m n ◦ = ( r, p ) where:

<!-- formula-not-decoded -->

where g : i ↦→ R is some function that defines the reward from transitioning away from state i . Since |A| = 1 we omit actions from the reward and transition dynamics.

For each ring MDP and function g we additionally construct a corresponding 'false-ring' MDP (Figure 6) with the same state and actions spaces as m n ◦ but with states that only self-transition and with rewards designed to mimic the discounted n -step returns on Ring MDPs. We represent these as ˜ m n ◦ = (˜ r, ˜ p ) where

<!-- formula-not-decoded -->

and r n ( s i ) denotes the discounted n -step return starting from s i in m n ◦ . Note that the discounted n -step return of an n -state false-ring MDP is the same as that of an n -state ring MDP.

We now provide some basic results about pairs of ring and false-ring MDPs that we will use periodically in our proofs.

Lemma 1. For any n ∈ Z + ∪ {∞} if we treat the ring MDP m n ◦ as the environment and assume ˜ m n ◦ ∈ M it follows that when n &lt; ∞ and

when n = ∞ .

Proof. First we note that, since ring and false-ring MDPs only have one action, we can write GLYPH&lt;5&gt; = { π } where π takes this action at all states. We first consider the case when n &lt; ∞ , noting that that both MDPs are deterministic and that for any state s , performing n transitions will always return to s . We now consider an application of n -step Bellman operator of the false-ring model to an arbitrary function v ∈ V :

<!-- formula-not-decoded -->

implying that ˜ m n ◦ ∈ M n ( GLYPH&lt;5&gt; , V ) as needed. We now consider the case when n = ∞ : here, we note that for any state s ∈ S :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can then write:

<!-- formula-not-decoded -->

since ˜ m ◦ ∞ only self-transitions at each state. This shows that ˜ m ◦ ∞ ∈ M ∞ (Π) as needed.

Figure 6: Ring and false-ring environments with reward structure defined by g : Z + ↦→ R . States are numbered circles and outgoing arrows indicate possible transitions from each state. Arrows are labeled by the reward attained from performing their transition.

<!-- image -->

Lemma 2. Fix any k, K ∈ Z + ∪{∞} with k &lt; K and let f : S ↦→ R be any constant function. Let m = m K ◦ and ˜ m = ˜ m K ◦ . For any γ ∈ (0 , 1) it follows that glyph[negationslash]

<!-- formula-not-decoded -->

where m K ◦ and ˜ m K ◦ are K -state ring and false-ring MDPs with g ( i ) = 1 { i ∈ [1 , k ] } .

Proof. We begin by examining the k -step Bellman operator and Bellman fixed-point under the ring m K ◦ :

<!-- formula-not-decoded -->

where the second equality follows from the fact that g ensures that no reward is received after the first k steps from s 1 .

Next we examine the corresponding k -step Bellman operator under the false-ring ˜ m K ◦ :

<!-- formula-not-decoded -->

where the second equality follows from the construction of K -step false-ring MDPs to match the K -step returns of their corresponding ring MDP.

Taken together Eqs. (27-28) imply that in order for T k π f ( s 1 ) = ˜ T k π f ( s 1 ) it must be the case that ∑ K -1 t =0 γ t = ∑ k -1 t =0 γ t which can only happen when γ = 0 . Note that these properties hold when K = ∞ . This completes the proof.

## A.2 Proofs

In this section we provide proofs of the results in the main text.

Proposition 1. Let V be a set of functions such that if v ∈ V then T π v ∈ V for all π ∈ Π . Then, for k, K ∈ Z + such that k divides K , it follows that:

- (i) For any M⊆ M and any Π ⊆ GLYPH&lt;5&gt; , we have that M k (Π , V ) ⊆ M K (Π , V ) .
- (ii) If Π is non-empty and V contains at least one constant function, then there exist environments such that M k (Π , V ) ⊂ M K (Π , V ) .

Proof. Consider some m ∈ M k (Π , V ) . For any π ∈ Π and v ∈ V we know that ˜ T k π v = T k π v . Since k divides K we know that K = zk where z ∈ Z + . Hence

<!-- formula-not-decoded -->

Finally since V is closed under Bellman updates we can write ˜ T k π v = T k π v ∈ V , which allows us iteratively equate k -step environment and model operators on the right-hand side of Eq. (29) to obtain:

<!-- formula-not-decoded -->

This suffices to show that m ∈ M K (Π , V ) which means M k (Π , V ) ⊆ M K (Π , V ) .

We now assume that V contains at least one constant function and Π is non-empty and produce an instance of an environment and model class where the relation is strict. Let the environment be a K -state ring environment (see A.1.1): m K ◦ with g ( i ) = 1 { i ∈ [1 , k ] } and let M = M . Next we introduce a model given by the corresponding false-ring MDP (see A.1.1) ˜ m K ◦ . From Lemma 1 we have that ˜ m K ◦ ∈ M K (Π , V ) .

glyph[negationslash]

Since there is at least one constant function f ∈ V we know that T k π f ( s 1 ) = ˜ T k π f ( s 1 ) from Lemma 2. This is sufficient to show that ˜ m K ◦ / ∈ M k (Π , V ) and thus we have proven that there are instances where M k (Π , V ) ⊂ M K (Π , V ) .

Proposition 2. For any Π ⊆ GLYPH&lt;5&gt; and any k ∈ Z + it follows that

<!-- formula-not-decoded -->

Proof. We first note M ∞ (Π) = ⋂ π ∈ Π M ∞ ( { π } ) and consider any m ∈ M ∞ ( { π } ) for some π ∈ Π . From the definition of PVE we know ˜ v π = v π and thus can say:

<!-- formula-not-decoded -->

which suggests that m ∈ M k ( { π } , { v π } ) and thus M ∞ ( { π } ) ⊆ M k ( { π } , { v π } ) .

We now consider any element m ∈ M k ( { π } , { v π } ) , and note that from the definition of orderk VE we know that ˜ T k π v π = T k π v π , thus we can say:

<!-- formula-not-decoded -->

where we can repeat the process described in these implications ad-infinitum to obtain ˜ v π = lim n →∞ ˜ T nk π v π = v π . Hence m ∈ M ∞ ( { π } ) and thus M k ( { π } , { v π } ) .

Taken together this shows that M ∞ ( { π } ) = M k ( { π } , { v π } ) for any k and π thus:

<!-- formula-not-decoded -->

for any k ∈ Z + .

Corollary 2. Let Π ⊆ GLYPH&lt;5&gt; and let V be as in Proposition 1 for k ∈ Z + then we have that M k (Π , V ) ⊆ M ∞ (Π) . Moreover, if Π is non-empty and V contains at least one constant function, then there exist environments such that M k (Π , V ) ⊂ M ∞ (Π)

Proof. Consider some m ∈ M k (Π , V ) . From the generalization of Property 1 we know that m ∈ M zk (Π , V ) for any z ∈ Z + since k divides zk . Thus we know that ˜ T zk π v = T zk π v for any

choice of π ∈ Π , v ∈ V and z ∈ Z + . Accordingly the expressions are equal in the limit as z →∞ . Combining this with the fact that both ˜ T π and T π are contraction mappings, we obtain:

<!-- formula-not-decoded -->

which implies m ∈ M ∞ (Π) and thus M k (Π , V ) ⊆ M ∞ (Π) , as needed.

glyph[negationslash]

Moreover, so long that Π is nonempty and V contains some constant function f , we can construct a pair of ∞ -state ring / false-ring MDPs: m ◦ ∞ and ˜ m ◦ ∞ with g ( i ) = 1 { i ∈ [1 , k ] } (see A.1.1). By assuming that m ◦ ∞ is the environment, Lemma 1 tells us that ˜ m ◦ ∞ ∈ M ∞ (Π) and we know from Lemma 2 that T k π f ( s 1 ) = ˜ T k π f ( s 1 ) hence ˜ m ◦ ∞ / ∈ M k (Π) .

Proposition 3. Let Π ⊆ GLYPH&lt;5&gt; . If the environment state can be factored as S = X × Y where |Y| &gt; 1 and v π ( s ) = v π (( x, y )) = v π ( x ) for all π ∈ Π , then M 1 (Π , V ) ⊂ M ∞ (Π) .

Proof. Assume that M = M . Denote the environment reward and transition dynamics as ( r, p ) . For any value y 0 ∈ Y we consider a model m y 0 :

<!-- formula-not-decoded -->

We now examine the Bellman fixed-point induced by environment for any policy π ∈ Π :

<!-- formula-not-decoded -->

We can compare this to the Bellman operator induced by our model for the same policy:

<!-- formula-not-decoded -->

Notice that v π is a fixed point of this operator, hence ˜ v π = v π and and thus m y 0 ∈ M ∞ (Π) (since our particular choice of π ∈ Π was arbitrary). Moreover, we can construct different models for each y 0 ∈ Y , we know that

<!-- formula-not-decoded -->

Moreover, suppose m y 0 ∈ M 1 (Π , V ) . This implies that for all v ∈ V

<!-- formula-not-decoded -->

glyph[negationslash]

we now choose v (( x, y )) = 1 { y = y 0 } which reduces the above equations to:

glyph[negationslash]

<!-- formula-not-decoded -->

where P denotes the conditional probability of an event.

Now consider the class of models defined by Eq. (38). Suppose M Y ∈ M 1 (Π , V ) , by Eq. (40) this would mean that P ( y ′ = y 0 | x, y, π ) = 1 for all y 0 ∈ Y . This is impossible unless |Y| = 1 hence there must exist m y 0 / ∈ M 1 (Π , V ) and thus M 1 (Π , V ) ⊂ M ∞ (Π , V ) .

Corollary 1. Let GLYPH&lt;5&gt; det be the set of all deterministic policies. An optimal policy for any ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; det ) is also optimal in the environment.

Proof. Denote a deterministic optimal policy with respect to the environment as π ∗ . Let ˜ m ∈ M ∞ ( GLYPH&lt;5&gt; det ) and ˜ π ∗ be a deterministic optimal policy with respect ˜ m .

Suppose ˜ π ∗ were not optimal in the environment. This implies that v π ∗ ( s ) ≥ v ˜ π ∗ ( s ) ∀ s ∈ S with strict inequality for at least one state. However, since π ∗ and ˜ π ∗ are deterministic we have:

<!-- formula-not-decoded -->

for some s ∈ S . This contradicts ˜ π ∗ being optimal in the model.

Proposition 5. There exist environments and model classes for which M ∞ ( GLYPH&lt;5&gt; ) ⊂ M ∞ ( GLYPH&lt;5&gt; det ) .

Proof. Since the environment and model only differ when action R is taken from state 2 , we only need to consider deterministic policies that make this choice. Note that if action R is taken from state 2 , the values in both the model and environment at states 2 and 3 are necessarily 0 and the value of each in state 1 is either (1 -γ ) -1 or 0 depending on the action taken from state 1 . This suffices to show that the environment and model have the same values for all deterministic policies.

However, one can see that the model and environment differ for stochastic policies. Take, for instance, a policy for which π ( a | s ) = 0 . 5 for all a ∈ A , s ∈ S . The induced Markov reward processes from applying this policy to the environment and model, which share the same reward structure, have different transition dynamics at state 2 . It can be easily verified that this results in different values for the environment and model.

Proposition 6. For any π ∈ GLYPH&lt;5&gt; , v ∈ V and k, n ∈ Z + , we have that

<!-- formula-not-decoded -->

Proof. We begin by considering the left-hand side

<!-- formula-not-decoded -->

as needed.

Proposition 7. For any π ∈ GLYPH&lt;5&gt; , v ∈ V and k, n ∈ Z + , assuming ‖ v π -v ‖ ∞ &lt; g · ‖ v π -v ‖ d π for some g ≥ 0 , we have that:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

## A.3 Experimental details - illustrative experiments

## A.3.1 Code

Code to reproduce our illustrative experiments can be found at https://github.com/ chrisgrimm/proper\_value\_equivalence .

## A.3.2 Computational resources

Illustrative experiments were performed on three machines each with 4 NVIDIA GeForce GTX 1080 Ti graphics cards.

## A.3.3 Environment

All illustrative experiments depicted in Figures 3 and 4 were carried out in a stochastic version of the Four Rooms environment (depicted in Figure 7) where |S| = 104 and A consists of four actions corresponding to an intended movement in each of the cardinal directions. When an agent takes an action, it will move in the intended direction 80% of the time and otherwise move in a random direction. If the agent moves into a wall it will remain in place. When the agent transitions into the upper-right square it receives a reward of 1 , all other transitions yield 0 reward.

## A.3.4 Model representation and initialization

Figure 7: Visualization of the Four Rooms environment.

<!-- image -->

Models are represented tabularly by matrices ˜ R ∈ R |S|×|A| and ˜ P a ∈ R |S|×|S| for a ∈ A where ˜ R s,a = ˜ r ( s, a ) and ˜ P a s,s ′ = ˜ p ( s ′ | s, a ) . We generally constrain a matrix to be row-stochastic by parameterizing it with an unconstrained matrix of the same shape and applying a softmax with temperature 1 to each of its rows. In experiments with model capacity constraints we additionally impose that each ˜ P a has a rank of at most k by representing ˜ P a = D a K a where D a ∈ R |S|× k , K a ∈ R k ×|S| and both D a and K a are constrained to be row-stochastic (note that the product of rowstochastic matrices is itself row-stochastic). In this setting the parameters of the capacity-constrained transition dynamics are the unconstrained matrices parameterizing D a and K a .

Models are initialized by randomly sampling the entries of ˜ R according to U ( -1 , 1) and the entries of the matrices parameterizing the transition dynamics according to U ( -5 , 5) , where U ( l, u ) denotes a uniform distribution over the interval ( l, u ) .

In all illustrative experiments we train our models using the Adam optimizer with default hyperparameters ( β 1 = 0 . 99 , β 2 = 0 . 999 , glyph[epsilon1] =1e-8).

## A.3.5 Model space experiments

In Figure 3 we illustrate the properties of spaces of models trained to be in M k ( GLYPH&lt;5&gt; , V ) for k ∈ { 1 , 30 , 40 , 50 , 60 } and in M ∞ ( GLYPH&lt;5&gt; ) . To train each of these models we construct a set of policies

and functions D = { ( π i , v i ) } 100 , 000 i =1 . Each generated policy π i is, with equal probability, either a uniformly sampled deterministic policy or a stochastic policy for which at each state s , π i ( a | s ) = f a / ∑ a ∈A f a where f a ∼ U (0 , 1) for each a ∈ A . Each v i is sampled such that v i ( s ) ∼ U ( -1 , 1) for each s ∈ S . We then sample minibatches B ∼ D with | B | = 50 at each iteration and update models to minimize

<!-- formula-not-decoded -->

for orderk VE and PVE models respectively.

Each model is updated in this manner for 500 , 000 iterations with a learning rate of 1e-3 and a snapshot of the model is stored every 1000 iterations-creating a timeline of the model's progress through training. For each model class, this experiment is repeated with 120 randomly initialized models. To generate the points on the scatter plots depicted in Figure 3, we iterate through the snapshots of these 120 models. At snapshot t (training iteration 1000 × t ) we collect the snapshots of all the models and convert each model into a 1D vector representation by concatenating the entries from its reward and transition dynamics matrices. We then apply principle component analysis to these vectors, isolating the first two principle components, which we treat as (x, y) coordinates in the scatter plots. For the top row in Figure 3 we color these points according to progress through training: ( t/ 500) . On the bottom row, we compute the optimal policy with respect to each point's corresponding model: ˜ π ∗ and color the point according to ( ∑ s v ˜ π ∗ ( s )) / ( ∑ s v π ∗ ( s )) .

We produce the plot of model class diameters in Figure 3 by taking the scatter-plot points corresponding to the final snapshot ( t = 500 ) of models for each k , randomly grouping them into 4 sets of 30 points and computing the diameters of each set. We then use these 4 diameters to produce error bars.

## A.3.6 Individual model visualization

To generate the visualization of the dynamics of individual models displayed in Figure 4b, we randomly select a single PVE model trained in our model space experiments. We then collect 5000 length 30 trajectories starting from the bottom left state. The paths of these trajectories are then overlaid on top of a visualization of the environment and colored according to time along the trajectory ( t/ 30 ). This procedure is repeated using the environment in Figure 4a.

## A.3.7 Model capacity experiment

We compare the effect of capacity constraints on learning models in M ∞ ( GLYPH&lt;5&gt; ) and M ∞ ( GLYPH&lt;5&gt; det ) respectively by restricting the rank of the learned model's transition dynamics (as in A.3.4). We restrict the ranks of model transition dynamics to be at most k for k ∈ { 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100 , 104 } . To train each model we collect a set of 1000 policies by beginning with a random policy and repeatedly running the policy iteration algorithm in the environment, starting with a randomly initialized policy and stopping when the optimal policy is reached. The sequence of improved policies resulting from this process is stored. Whenever the algorithm terminates, a new random policy is generated and the process is repeated until 1000 policies have been stored. To increase the number of distinct policies generated by this process, at each step of policy iteration, we select, uniformly at random, 10% of states and update the policy at only these states. We then further boost the breadth of our collected policies and specialize them to GLYPH&lt;5&gt; and GLYPH&lt;5&gt; det by adding stochastic or deterministic 'noise.'

Precisely, when training a model to be in M ∞ ( GLYPH&lt;5&gt; ) we iterate over each of the 1000 policies generated by our policy iteration procedure and generate an additional 100 policies. Each additional policy is generated by selecting, uniformly at random, 10% of the original policy's states and replacing the its distribution at these states with a uniform distribution over actions.

When training a model to be in M ∞ ( GLYPH&lt;5&gt; det ) the same procedure is repeated but the original policy's distributions, at the selected states, are replaced by randomly generated deterministic distributions.

In either case, this produces 100 , 000 policies which are evaluated in the environment. Together this forms a set of policies and corresponding value functions: D = { ( π i , v i ) } 100 , 000 i =1 which can be used construct mini-batch PVE losses as described in (45). Models are trained according to these losses for 1 , 000 , 000 iterations with a learning rate of 5e-4. The errorbars around the environment value of the models' optimal policies at the end of training are reported across 10 seeds.

## A.4 MuZero experiment

Atari. We follow the Atari configuration used in Schrittwieser et al. [30], summarised in Table 1.

Table 1: Atari hyperparameters.

| PARAMETER                                                                                                                                                                           | VALUE                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Start no-ops Terminate on life loss Action set Max episode length Observation size Preprocessing Action repetitions Max-pool over last N action Total environment frames, including | [0, 30] Yes Valid actions 30 minutes (108,000 frames) 96 × 96 Grayscale 4 4 500M |

MuZero implementation. Our MuZero implementation largely follows the description given by Schrittwieser et al. [30], but uses a Sebulba distributed architecture as described in Hessel et al. [19], and TD( λ ) rather than n -step value targets. The hyperparameters are given in Table 2. Our network architecture is the same as used in MuZero [30].

The base MuZero loss is given by

<!-- formula-not-decoded -->

The reward loss glyph[lscript] r simply regresses the model-predicted rewards to the rewards seen in the environment. To compute the value and policy losses, MuZero performs a Monte-Carlo tree search using the learned model. The policy targets are proportional to the MCTS visitation counts at the root node. The value targets are computed using the MCTS value prediction ˜ v and the sequences of rewards. MuZero uses an n -step bootstrap return estimate v target t + k = ∑ n j =1 γ j -1 r t + k + j + γ n ˜ v t + k + n . We use the a TD( λ ) return estimate instead.

For our additional loss corresponding to past policies, we periodically store the parameters for the value function and policy (i.e. the network heads that take the model-predicted latent state as input). Then, we compute the same value loss glyph[lscript] v for each past value function. To account for the fact that the reward sequence was drawn from the current policy π rather than the stored policies, we use V-trace to compute a return estimate for the past policies.

The additional hyperparameters for the buffer of past value heads were tuned on MsPacman, over a buffer size in { 64 , 128 , 256 } and an update interval in { 10 , 50 , 100 , 500 } . Our experiments took roughly 35k TPU-v3 device-hours for both tuning and the full evaluation.

Full results. We report the final scores per game in Table 3. The mean scores are across the final 200 episodes in each of three seeds. We also report the standard error of the mean across seeds only. Performing a Wilcoxon signed rank test comparing per-game scores, we find that the version with the additional Past Policies loss has a better final performance with p = 0 . 044 .

Table 2: Hyperparameters for our MuZero experiment.

| HYPERPARAMETER                        | VALUE                               |
|---------------------------------------|-------------------------------------|
| Batch size length overlap             | 96 sequences                        |
| Sequence                              | 30 frames                           |
| Sequence                              | 10 frames                           |
| Model unroll length K                 | 5                                   |
| Optimiser                             | Adam                                |
| Initial learning rate                 | 1 × 10 - 4                          |
| Final learning rate (linear schedule) | 0                                   |
| Discount                              | 0.997                               |
| Target network update rate            | 0.1                                 |
| Value loss weight                     | 0.25                                |
| Reward loss weight                    | 1.0                                 |
| Policy loss weight                    | 1.0                                 |
| MCTS number of simulations            | 25                                  |
| λ for TD( λ )                         | 0.8                                 |
| MCTS Dirichlet prior fraction         | 0.3                                 |
| MCTS Dirichlet prior α                | 0.25                                |
| Search parameters update rate         | 0.1                                 |
| Value, reward number of bins          | 601                                 |
| Nonlinear value transform             | sgn ( z )( √ | z | +1 - 1)+0 . 01 z |
| Value buffer size                     | 128                                 |
| Value buffer update interval          | 50                                  |
| Value buffer loss weight              | 0.25                                |

Table 3: Final Atari scores for our deep RL experiments. We report the mean of the final 200 episodes over all three seeds, and the standard error of the mean across seeds.

| Environment         | MuZero (our impl.)   | MuZero (our impl.)   | MuZero + Past Policies   | MuZero + Past Policies   |
|---------------------|----------------------|----------------------|--------------------------|--------------------------|
| alien               | 38,698               | ± 2,809              | 52,821                   | ± 1,918                  |
| amidar              | 6,631                | ± 568                | 4,239                    | ± 1,550                  |
| assault             | 35,876               | ± 550                | 35,013                   | ± 738                    |
| asterix             | 674,573              | ± 88,318             | 549,421                  | ± 9,280                  |
| asteroids           | 214,034              | ± 4,719              | 235,543                  | ± 14,605                 |
| atlantis            | 835,445              | ± 92,290             | 845,409                  | ± 60,318                 |
| bank_heist          | 837                  | ± 265                | 552                      | ± 234                    |
| battle_zone         | 39,471               | ± 12,658             | 72,183                   | ± 11,385                 |
| beam_rider          | 120,675              | ± 16,588             | 130,129                  | ± 14,014                 |
| berzerk             | 22,449               | ± 3,780              | 35,249                   | ± 3,179                  |
| bowling             | 59                   | ± 0                  | 47                       | ± 7                      |
| boxing              | 99                   | ± 0                  | 99                       | ± 0                      |
| breakout            | 504                  | ± 165                | 770                      | ± 12                     |
| centipede           | 400,268              | ± 32,821             | 534,432                  | ± 38,912                 |
| chopper_command     | 524,655              | ± 154,540            | 660,503                  | ± 27,000                 |
| crazy_climber       | 189,621              | ± 7,313              | 217,204                  | ± 12,764                 |
| defender            | 322,472              | ± 105,043            | 483,394                  | ± 11,589                 |
| demon_attack        | 131,963              | ± 3,819              | 112,140                  | ± 17,739                 |
| double_dunk         | 3                    | ± 4                  | -1                       | ± 1                      |
| enduro              | 0                    | ± 0                  | 132                      | ± 86                     |
| fishing_derby       | -97                  | ± 0                  | -52                      | ± 29                     |
| freeway frostbite   | 0 3,439              | ± 0 ±                | 0 8,049                  | ± 0 ± 526                |
| gopher              | 121,984              | 1,401 ± 338          | 120,551                  | ± 923                    |
| gravitar            | 2,807                | ± 123                | 3,927                    | ± 54                     |
| hero                | 7,877                | ± 960                | 9,871                    | ± 523                    |
| ice_hockey          | -6                   | ± 4                  | -11                      | ± 3                      |
| jamesbond           | 23,475               | ± 1,586              | 13,668                   | ± 4,480                  |
| kangaroo            | 9,659                | ± 2,389              | 10,465                   | ± 2,835                  |
| krull               | 11,259               | ± 173                | 11,295                   | ± 108                    |
| kung_fu_master      | 55,242               | ± 4,267              | 83,705                   | ± 6,565                  |
| montezuma_revenge   | 0                    | ± 0                  | 0                        | ± 0                      |
| ms_pacman           | 40,263               | ± 387                | 43,700                   | ± 1,042                  |
| name_this_game      | 76,604               | ± 7,107              | 94,974                   | ± 9,942                  |
| phoenix             | 67,119               | ± 9,747              | 49,919                   | ± 10,573                 |
| pitfall pong        | -2 -7                | ± 1 ± 9              | -24 -6                   | ± 7 ± 9                  |
| private_eye         | 193                  | ± 101                | -6                       | ± 228                    |
| qbert               | 64,732               | ± 8,619              | 70,593                   | ± 16,955                 |
| riverraid           | 27,688               | ± 1,001              | 28,026                   | ± 1,823                  |
| road_runner         | 151,639              | ± 90,186             | 571,829                  | ± 106,184                |
| robotank            | 53                   | ± 2                  | 25                       | ± 8                      |
| seaquest            | 27,530               | ± 10,632             | 141,725                  | ± 48,000                 |
| skiing              | -27,968              |                      | -30,062                  | ± 248                    |
| solaris             | 1,544                | ± 1,346 ± 140        | 1,501                    | ± 193                    |
| space_invaders      | 3,962                | ± 102                | 5,367                    | ± 953                    |
| star_gunner         | 663,896              | ± 80,698             | 547,226                  | ± 126,538                |
| surround            | 7                    | 0                    |                          | ±                        |
|                     | -23                  | ±                    | 6                        | 1                        |
| tennis              | 267,331              | ± 0                  | 0                        | ± 0                      |
| time_pilot          |                      | ± 15,256             | 228,282                  | ± 10,844 ± 8             |
| tutankham up_n_down | 134 434,746          | ± 10 ± 3,905         | 150 432,240              | ± 4,221                  |
| venture             | 0                    | ± 0                  | 0                        | ± 0                      |
| video_pinball       | 376,660              | ± 37,647             | 378,897                  | ± 26,486                 |
| wizard_of_wor       | 79,425               | ± 1,458              | 54,093                   | ± 6,294                  |
| zaxxon              | 15,752               | ± 231                |                          | ± 196                    |
|                     |                      |                      | 15,790                   |                          |